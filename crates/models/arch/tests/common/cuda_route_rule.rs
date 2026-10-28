// SPDX-License-Identifier: Apache-2.0
// The goldens' routing dump and P9's per-layer rule, shared by the P9 body
// (`cuda_route_parity_body.rs`) and the parity body's P1 and P3 (`cuda_hf_parity_body.rs`), so the
// rule that accepts an expert-pick difference is one function (#933). Included with `include!` at
// the binary's root. The including binary supplies `goldens_dir()`; the layer count and the
// picks per token are the reader's arguments, so a binary that reads them off its tapes needs no
// constant of its own.

/// One run's routing over `rows` rows: per layer and row, the top-8 experts and the rank-8 and
/// rank-9 router probabilities, layer-major.
struct Picks {
    idx: Vec<i32>,
    p89: Vec<f32>,
    top_k: usize,
}

impl Picks {
    fn set(&self, rows: usize, layer: usize, row: usize) -> BTreeSet<i32> {
        self.idx[(layer * rows + row) * self.top_k..][..self.top_k]
            .iter()
            .copied()
            .collect()
    }

    fn p89(&self, rows: usize, layer: usize, row: usize) -> (f64, f64) {
        let at = (layer * rows + row) * 2;
        (f64::from(self.p89[at]), f64::from(self.p89[at + 1]))
    }
}

/// A routing file of the dump: `rows` rows of the f32 run then the bf16 run, each `[layers, rows,
/// top_k]` i32 then `[layers, rows, 2]` f32, little-endian (`meta.json`'s `route`).
struct Route {
    rows: usize,
    f32: Picks,
    bf16: Picks,
}

fn read_route_of(name: &str, rows: usize, (layers, top_k): (usize, usize)) -> Route {
    let path = goldens_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (n_idx, n_p) = (layers * rows * top_k, layers * rows * 2);
    assert_eq!(
        bytes.len(),
        2 * (n_idx + n_p) * 4,
        "{name}: two runs of [{layers}, {rows}, {top_k}] i32 and [{layers}, {rows}, 2] f32"
    );
    let words: Vec<[u8; 4]> = bytes.as_chunks::<4>().0.to_vec();
    let run = |from: usize| Picks {
        idx: (words[from..][..n_idx].iter())
            .map(|&w| i32::from_le_bytes(w))
            .collect(),
        p89: (words[from + n_idx..][..n_p].iter())
            .map(|&w| f32::from_le_bytes(w))
            .collect(),
        top_k,
    };
    Route {
        rows,
        f32: run(0),
        bf16: run(n_idx + n_p),
    }
}

impl Route {
    /// τ at `layer`: twice the largest |rank-8 probability, HF-bf16 − HF-f32| over the rows.
    fn tau(&self, layer: usize) -> f64 {
        let r = (0..self.rows)
            .map(|row| {
                let (f, _) = self.f32.p89(self.rows, layer, row);
                let (b, _) = self.bf16.p89(self.rows, layer, row);
                (b - f).abs()
            })
            .fold(0f64, f64::max);
        2.0 * r
    }

    /// Whether (layer, row) is a router tie under `tau`, and its f32 rank-8 minus rank-9 margin.
    fn tie(&self, layer: usize, row: usize, tau: f64) -> (bool, f64) {
        let (p8, p9) = self.f32.p89(self.rows, layer, row);
        (p8 - p9 <= tau, p8 - p9)
    }
}

/// P9's rule at `layer`, for both models (#933: one function, so the two cannot drift apart):
/// layer 0, no flip outside a router tie; layers ≥ 1, a flip inside a router tie (the same τ)
/// accepted with no count limit, and one outside a tie accepted only when the same token flipped
/// at an earlier layer (`unexcused` counts those that did not; the user's decision 2026-10-27,
/// option (a)), or when HF-bf16 flips at least as many tokens as CUDA in the layer (`bf16_flips`
/// against `flips`, CUDA's inside and outside; the user's decision 2026-10-27, option (a)
/// amended). Layer 0's `unexcused` is all its outside-tie flips: no token has an earlier flip.
/// Whether it passes, its text, and the excuse its outside-tie flips took.
fn layer_rule(
    layer: usize,
    (outside, excused, unexcused): (usize, usize, usize),
    (flips, bf16_flips): (usize, usize),
) -> (bool, String, &'static str) {
    if layer == 0 {
        return (outside == 0, "0 flips outside router ties".to_string(), "none");
    }
    // The excuse a layer >= 1's outside-tie flips with no earlier flip take: HF-bf16 flips at
    // least as many tokens there as CUDA (the user's decision 2026-10-27).
    let bf16_excuses = bf16_flips >= flips;
    let excuse = match (excused > 0, unexcused > 0 && bf16_excuses) {
        (false, false) => "none",
        (true, false) => "earlier flip",
        (false, true) => "HF-bf16 flips >= CUDA's",
        (true, true) => "earlier flip, HF-bf16 flips >= CUDA's",
    };
    (
        unexcused == 0 || bf16_excuses,
        "0 flips outside router ties on a token with no earlier flip unless HF-bf16 flips >= \
         CUDA's in the layer, those inside not capped (the user's decision 2026-10-27, option (a) \
         amended)"
            .to_string(),
        excuse,
    )
}
