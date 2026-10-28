// SPDX-License-Identifier: Apache-2.0
//! P9, the expert picks: Gemma 4 26B-A4B's CUDA router choices on the real checkpoint against HF
//! transformers' (AI-native-Systems-Research/scratchy#933).
//!
//! The tests are qwen3.6-35b-a3b's (`common/cuda_route_parity_body.rs`, shared with
//! `cuda_route_parity_35b.rs`), at Gemma's constants: 30 MoE layers, 128 experts, top 8, and no
//! GDN layer. Gemma's router is `layer.router` (`Gemma4TextRouter`): an RMS norm with no gain,
//! `× router.scale × 2816^-0.5`, `router.proj`, a softmax in f32, top 8. The reference is
//! `cuda_hf_reference.py --model gemma-4-26b-a4b-it`'s routing dump in the goldens
//! (`route_<p>.bin`, `route_greedy_<p>.bin`, described in `meta.json`'s `route`), whose rank-8 and
//! rank-9 probabilities are HF's own f32 softmax. CUDA's routing takes the top 8 of the raw logits
//! and a softmax over them (`RouteProgram::GemmaScaledSoftmax`): the same set, since a softmax
//! keeps the order.
//!
//! - A router tie at (layer, token): HF-f32's rank-8 and rank-9 probabilities differ by at most
//!   τ = 2·r, `r` the largest |rank-8 probability, HF-bf16 − HF-f32| over that prompt's tokens at
//!   that layer (P3's "twice the measured noise"; the 35B's rule, unchanged).
//! - Per prompt and layer, printed: the router ties, CUDA's flips inside and outside them, and
//!   HF-bf16's own flips against HF-f32 (the yardstick).
//! - Gate, layer 0: no flip outside a router tie (its input, the embedding and one attention
//!   layer, is the same up to rounding on both sides).
//! - Gate, layers 1-29: the shared body's one rule (`layer_rule`), the 35B's: a flip inside a
//!   router tie (the same τ) accepted with no count limit; one outside a tie accepted only when the
//!   same token flipped at an earlier layer, or when HF-bf16 flips at least as many tokens as CUDA
//!   in that layer, else it fails (the user's decisions 2026-10-27, option (a) and its amendment).
//!   Per layer, the outside-tie flips an earlier flip excuses, those it does not, and which excuse
//!   applied are printed.
//! - Attribution for P5/P7, printed: per greedy step of prompts 1-5 (HF-f32's sequence), the
//!   first layer at which the step's last token's CUDA pick differs from HF-f32's.
//!
//! Built only under `nvcc` and `gemma-4-26b-a4b-it`, and needs the H100, the checkpoint and the
//! goldens, as `cuda_hf_parity_gemma4`; run with `--test-threads=1`. A missing goldens dir fails
//! with the command that makes it, never skips. Every line prints before a test fails.

use scratchy_models::gemma4_moe::gemma_4_26b_a4b_it::{CUDA_TAPES, CUDA_WEIGHTS};

include!("common/cuda_route_parity_body.rs");

/// The goldens: `cuda_hf_reference.py --model gemma-4-26b-a4b-it`'s.
const GOLDENS: &str = "tests/goldens/gemma4-moe-gemma-4-26b-a4b-it-hf";
const MODEL: &str = "gemma-4-26b-a4b-it";
const CHECKPOINT: &str = "gemma-4-26B-A4B-it";
/// The MoE layers (every layer of the 30) and the picks per token.
const LAYERS: usize = 30;
const TOP_K: usize = 8;
const EXPERTS: i32 = 128;
/// No GDN layer: no GDN pool, and no `GdnSequence` in a forward.
const GDN_POOL: Option<GdnPoolConfig> = None;

/// The prompt and row #933's diagnosis, part 2, test B reads (`/tmp/v933-diag2.md`): prompt 3's
/// row 9, 6 logits off HF-f32 on CUDA.
const DIAG_PROMPT: usize = 3;
const DIAG_ROW: usize = 9;

/// #933's diagnosis, part 2, test B, printed only, nothing gated: at every layer, prompt 3 row 9's
/// CUDA pick against HF-f32's (and HF-bf16's), from one fresh m=512 forward cut after each layer's
/// routing, as `p9_expert_picks_match_hf_f32` reads them. Per layer: whether it is a router tie
/// (P9's τ), the f32 rank-8 minus rank-9 margin, both picks, the experts CUDA has and f32 lacks and
/// the reverse (so a flip inside a tie shows too), and HF-bf16's own. Then the layers whose CUDA
/// pick differs, inside and outside a tie.
#[test]
fn diag_prompt_3_row_9_picks_per_layer() {
    let (device, weights) = rig();
    let mut w = CudaWorker::new(&device, cut_tapes(), &weights, Some(KV_POOL), GDN_POOL)
        .unwrap_or_else(|e| panic!("CudaWorker::new over the {LAYERS} cut tapes: {e}"));
    let p = (prompts().into_iter().find(|p| p.p == DIAG_PROMPT))
        .unwrap_or_else(|| panic!("prompt {DIAG_PROMPT} in meta.json"));
    let n = p.ids.len();
    assert!(DIAG_ROW < n, "prompt {} has row {DIAG_ROW}", p.p);
    let route = read_route(&format!("route_{}.bin", p.p), n);
    let (mut inside, mut outside) = (Vec::new(), Vec::new());
    for layer in 0..LAYERS {
        let got = &cuda_picks(&mut w, layer, &p.ids)[DIAG_ROW];
        let want = route.f32.set(n, layer, DIAG_ROW);
        let bf16 = route.bf16.set(n, layer, DIAG_ROW);
        let tau = route.tau(layer);
        let (tie, margin) = route.tie(layer, DIAG_ROW, tau);
        let only_cuda: Vec<i32> = got.difference(&want).copied().collect();
        let only_f32: Vec<i32> = want.difference(got).copied().collect();
        if *got != want {
            match tie {
                true => inside.push(layer),
                false => outside.push(layer),
            }
        }
        eprintln!(
            "diag P9 prompt {} row {DIAG_ROW} (id {}) layer {layer}: router tie={tie} (f32 rank-8 \
             minus rank-9 {margin:.4e} vs tau {tau:.4e}) CUDA={got:?} HF-f32={want:?} same={} \
             CUDA-only={only_cuda:?} f32-only={only_f32:?} HF-bf16={bf16:?} HF-bf16 same as \
             f32={} (information only)",
            p.p,
            p.ids[DIAG_ROW],
            *got == want,
            bf16 == want
        );
    }
    eprintln!(
        "diag P9 prompt {} row {DIAG_ROW}: CUDA's pick differs from HF-f32's at {} layers inside \
         a router tie {inside:?} and {} outside {outside:?} (information only)",
        p.p,
        inside.len(),
        outside.len()
    );
}

/// The vocabulary of the goldens' logits, for test B's minimum-row search.
const DIAG_VOCAB: usize = 262_144;
/// The P1 prompt whose minimum-cosine row test B reads.
const DIAG_P1_PROMPT: usize = 6;
/// Prompt 6's rows test B reads besides its minimum row: the rows 25-41 that decode clears of a
/// prefill-only kernel (`/tmp/v933-diag3.md` §2).
const DIAG_P1_ROWS: [usize; 4] = [25, 30, 37, 41];

/// #933's diagnosis, test B (`/tmp/v933-diag2.md:73-87`), printed only, nothing gated: at every
/// layer, CUDA's top-8 against HF-f32's top-8 for prompt 3 row 9, for prompt 6's row of CUDA's
/// minimum logits cosine against HF-f32 (found here from one fresh m=512 forward of prompt 6 and the
/// goldens' `ref_f32_6.bin`) and for prompt 6's rows [`DIAG_P1_ROWS`], including flips inside a
/// router tie, and the f32 router gap (rank-8 minus rank-9) at each layer, as
/// [`diag_prompt_3_row_9_picks_per_layer`] prints them. Per row, a summary: CUDA's layers that
/// differ from f32's inside and outside a tie and the experts swapped in all, beside HF-bf16's own
/// against f32 from the routing dump's bf16 run.
#[test]
fn diag_picks_per_layer_at_the_bad_rows() {
    let (device, weights) = rig();
    // The cut tapes, then the full m=512 tape, one worker: the last index is the full forward.
    let full = (CUDA_TAPES.iter().position(|t| t.bucket_m == M)).expect("the m=512 tape");
    let tapes: &'static [CudaTape] = Box::leak(
        (cut_tapes().iter().copied())
            .chain([CUDA_TAPES[full]])
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    let mut w = CudaWorker::new(&device, tapes, &weights, Some(KV_POOL), GDN_POOL)
        .unwrap_or_else(|e| panic!("CudaWorker::new over the cut tapes and the m=512 tape: {e}"));
    let prompt = |n: usize| {
        (prompts().into_iter().find(|p| p.p == n))
            .unwrap_or_else(|| panic!("prompt {n} in meta.json"))
    };
    // Prompt 6's minimum-cosine row.
    let p6 = prompt(DIAG_P1_PROMPT);
    let n6 = p6.ids.len();
    let logits = {
        let tape = &tapes[LAYERS];
        let mut padded = vec![0u32; M as usize];
        padded[..n6].copy_from_slice(&p6.ids);
        let positions: Vec<u32> = (0..M).collect();
        let kv = KvSequence {
            block_table: SEAT.0,
            prefix_len: 0,
            q_len: n6 as u32,
        };
        let mut out = vec![0u16; tape.result.bytes as usize / 2];
        w.forward(
            LAYERS,
            Input::TokenIds(&padded),
            &positions,
            Some(&kv),
            None,
            &mut out,
        )
        .unwrap_or_else(|e| panic!("prompt {}'s m=512 forward: {e}", p6.p));
        out
    };
    let path = goldens_dir().join(format!("ref_f32_{}.bin", p6.p));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(bytes.len(), n6 * DIAG_VOCAB * 4, "ref_f32: [L, vocab] f32");
    let f32s: Vec<f32> = (bytes.as_chunks::<4>().0.iter())
        .map(|&q| f32::from_le_bytes(q))
        .collect();
    let cosine = |a: &[f32], b: &[f32]| {
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            let (x, y) = (f64::from(x), f64::from(y));
            ab += x * y;
            aa += x * x;
            bb += y * y;
        }
        ab / (aa.sqrt() * bb.sqrt())
    };
    let (min_row, min_cos) = (0..n6)
        .map(|row| {
            let c: Vec<f32> = (logits[row * DIAG_VOCAB..][..DIAG_VOCAB].iter())
                .map(|&h| half::bf16::from_bits(h).to_f32())
                .collect();
            (row, cosine(&c, &f32s[row * DIAG_VOCAB..][..DIAG_VOCAB]))
        })
        .fold((0, f64::INFINITY), |m, x| if x.1 < m.1 { x } else { m });
    eprintln!(
        "diag B prompt {} minimum CUDA row: row {min_row} (token {}) cosine(CUDA, f32)={min_cos:.6} \
         (information only)",
        p6.p, p6.ids[min_row]
    );
    let mut rows: Vec<(usize, usize)> = vec![(DIAG_PROMPT, DIAG_ROW), (DIAG_P1_PROMPT, min_row)];
    rows.extend(
        (DIAG_P1_ROWS.iter())
            .filter(|&&r| r != min_row)
            .map(|&r| (DIAG_P1_PROMPT, r)),
    );
    for (want, row) in rows {
        let p = prompt(want);
        let n = p.ids.len();
        assert!(row < n, "prompt {want} has row {row}");
        let route = read_route(&format!("route_{}.bin", p.p), n);
        let (mut inside, mut outside) = (Vec::new(), Vec::new());
        // Experts swapped against f32's (CUDA's, HF-bf16's), and HF-bf16's layers that differ.
        let (mut swapped, mut bf16_swapped) = (0usize, 0usize);
        let (mut bf16_inside, mut bf16_outside) = (Vec::new(), Vec::new());
        for layer in 0..LAYERS {
            let got = &cuda_picks(&mut w, layer, &p.ids)[row];
            let f32_pick = route.f32.set(n, layer, row);
            let bf16_pick = route.bf16.set(n, layer, row);
            let tau = route.tau(layer);
            let (tie, gap) = route.tie(layer, row, tau);
            let only_cuda: Vec<i32> = got.difference(&f32_pick).copied().collect();
            let only_f32: Vec<i32> = f32_pick.difference(got).copied().collect();
            if *got != f32_pick {
                match tie {
                    true => inside.push(layer),
                    false => outside.push(layer),
                }
            }
            swapped += only_cuda.len();
            let bf16_only = bf16_pick.difference(&f32_pick).count();
            bf16_swapped += bf16_only;
            if bf16_pick != f32_pick {
                match tie {
                    true => bf16_inside.push(layer),
                    false => bf16_outside.push(layer),
                }
            }
            eprintln!(
                "diag B prompt {want} row {row} (id {}) layer {layer}: f32 router gap (rank-8 \
                 minus rank-9) {gap:.4e}, router tie={tie} (tau {tau:.4e}); CUDA top-8={got:?} \
                 HF-f32 top-8={f32_pick:?} same={} CUDA-only={only_cuda:?} f32-only={only_f32:?} \
                 (information only)",
                p.ids[row],
                *got == f32_pick
            );
        }
        eprintln!(
            "diag B prompt {want} row {row}: CUDA's top-8 differs from HF-f32's at {} layers inside \
             a router tie {inside:?} and {} outside {outside:?} (information only)",
            inside.len(),
            outside.len()
        );
        eprintln!(
            "diag B prompt {want} row {row} summary: CUDA vs f32 {} inside a tie, {} outside, \
             {swapped} experts swapped; HF-bf16 vs f32 {} inside a tie {bf16_inside:?}, {} outside \
             {bf16_outside:?}, {bf16_swapped} experts swapped (information only)",
            inside.len(),
            outside.len(),
            bf16_inside.len(),
            bf16_outside.len()
        );
    }
}
