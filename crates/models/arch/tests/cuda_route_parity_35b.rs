// SPDX-License-Identifier: Apache-2.0
//! P9, the expert picks: Qwen3.6-35B-A3B's CUDA router choices on the real checkpoint against HF
//! transformers' (AI-native-Systems-Research/scratchy#859).
//!
//! At each of the 40 MoE layers the router scores 256 experts per token and the token runs
//! through its top 8 (its "pick"). A bf16 rounding can reorder the 8th- and 9th-best router
//! probabilities and so change the pick: a "flip". HF-bf16 flips against HF-f32 too, so a flip
//! where those two probabilities are near-equal is noise, and one elsewhere is a bug, as P3 treats
//! a top-1 disagreement.
//!
//! The reference is `cuda_hf_reference.py --model qwen3.6-35b-a3b`'s routing dump in the goldens
//! (`route_<p>.bin`, `route_greedy_<p>.bin`, described in `meta.json`'s `route`). CUDA's picks are
//! the `topk_idx` region (`MoeTape.topk_idx`, i32 `[m, 8]`) after each layer's `MoeRoute` step:
//! one forward of a copy of the m=512 tape cut just after that step, whose `result` is the region,
//! as `cuda_narrow_diag_9b` read its steps (`3e2845f8`). The 40 cut tapes are one worker's 40
//! buckets, so one arena serves them. Nothing in `src/` changes: the cut tapes are plain
//! `CudaTape` data and `CudaWorker::forward` the public API. The picks are compared as sets: the
//! kernel stores its slots ascending, HF's `topk` descending.
//!
//! - A router tie at (layer, token): HF-f32's rank-8 and rank-9 probabilities differ by at most
//!   τ = 2·r, `r` the largest |rank-8 probability, HF-bf16 − HF-f32| over that prompt's tokens at
//!   that layer (P3's "twice the measured noise").
//! - Per prompt and layer, printed: the router ties, CUDA's flips inside and outside them, and
//!   HF-bf16's own flips against HF-f32 (the yardstick).
//! - Gate, layer 0: no flip outside a router tie (its input, the embedding and one GDN layer, is
//!   the same up to rounding on both sides).
//! - Gate, layers 1-39: no flip outside a router tie (the same τ as layer 0); every flip inside
//!   one is accepted, with no count limit (the user's decision 2026-10-26/27).
//! - Attribution for P5/P7, printed: per greedy step of prompts 1-5 (HF-f32's sequence), the
//!   first layer at which the step's last token's CUDA pick differs from HF-f32's.
//!
//! Built only under `nvcc` and `qwen3.6-35b-a3b`, and needs the H100, the checkpoint and the
//! goldens, as `cuda_hf_parity_35b`; run with `--test-threads=1`. A missing goldens dir fails with
//! the command that makes it, never skips. Every line prints before a test fails.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::OnceLock;

use scratchy_models::qwen3_5_moe::qwen3_6_35b_a3b::{CUDA_TAPES, CUDA_WEIGHTS};
use scratchy_target_cuda::runtime::{
    CudaDevice, CudaWorker, DeviceBuffer, GdnPoolConfig, GdnSequence, Input, KvPoolConfig,
    KvSequence, load_weights,
};
use scratchy_target_cuda::step::{CudaStep, MoeRouteBf16};
use scratchy_target_cuda::tape::CudaTape;

/// The goldens: `cuda_hf_reference.py --model qwen3.6-35b-a3b`'s.
const GOLDENS: &str = "tests/goldens/qwen3-5-moe-qwen3.6-35b-a3b-hf";
/// The bucket the picks are read in; every prompt and greedy sequence fits it.
const M: u32 = 512;
/// The MoE layers and the picks per token.
const LAYERS: usize = 40;
const TOP_K: usize = 8;
const EXPERTS: i32 = 256;
/// The prompts with greedy goldens.
const GREEDY_PROMPTS: usize = 5;
const KV_POOL: KvPoolConfig = KvPoolConfig { num_blocks: 16 };
const GDN_POOL: GdnPoolConfig = GdnPoolConfig { num_slots: 2 };
const SEAT: (&[u32], u32) = (&[0, 1, 2, 3, 4, 5, 6, 7], 0);

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(GOLDENS)
}

/// The goldens' `meta.json`.
fn meta() -> &'static serde_json::Value {
    static META: OnceLock<serde_json::Value> = OnceLock::new();
    META.get_or_init(|| {
        let path = goldens_dir().join("meta.json");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "{}: {e}. Make the goldens first: python3 -I \
                 crates/models/arch/tests/cuda_hf_reference.py --model qwen3.6-35b-a3b \
                 --checkpoint <Qwen3.6-35B-A3B dir> --prompts crates/e2e/testdata/prompts.txt \
                 --out crates/models/arch/{GOLDENS}",
                path.display()
            )
        });
        serde_json::from_str(&text).expect("meta.json parses")
    })
}

/// One prompt of the goldens: its number and ids.
struct Prompt {
    p: usize,
    ids: Vec<u32>,
}

fn prompts() -> Vec<Prompt> {
    (meta()["prompts"]
        .as_array()
        .expect("meta.json lists prompts")
        .iter())
    .map(|e| Prompt {
        p: e["prompt"].as_u64().expect("a prompt number") as usize,
        ids: (e["ids"].as_array().expect("ids").iter())
            .map(|v| v.as_u64().expect("an id") as u32)
            .collect(),
    })
    .collect()
}

/// One run's routing over `rows` rows: per layer and row, the top-8 experts and the rank-8 and
/// rank-9 router probabilities, layer-major.
struct Picks {
    idx: Vec<i32>,
    p89: Vec<f32>,
}

impl Picks {
    fn set(&self, rows: usize, layer: usize, row: usize) -> BTreeSet<i32> {
        self.idx[(layer * rows + row) * TOP_K..][..TOP_K]
            .iter()
            .copied()
            .collect()
    }

    fn p89(&self, rows: usize, layer: usize, row: usize) -> (f64, f64) {
        let at = (layer * rows + row) * 2;
        (f64::from(self.p89[at]), f64::from(self.p89[at + 1]))
    }
}

/// A routing file of the dump: `rows` rows of the f32 run then the bf16 run, each `[40, rows, 8]`
/// i32 then `[40, rows, 2]` f32, little-endian (`meta.json`'s `route`).
struct Route {
    rows: usize,
    f32: Picks,
    bf16: Picks,
}

fn read_route(name: &str, rows: usize) -> Route {
    let path = goldens_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (n_idx, n_p) = (LAYERS * rows * TOP_K, LAYERS * rows * 2);
    assert_eq!(
        bytes.len(),
        2 * (n_idx + n_p) * 4,
        "{name}: two runs of [40, {rows}, 8] i32 and [40, {rows}, 2] f32"
    );
    let words: Vec<[u8; 4]> = bytes.as_chunks::<4>().0.to_vec();
    let run = |from: usize| Picks {
        idx: (words[from..][..n_idx].iter())
            .map(|&w| i32::from_le_bytes(w))
            .collect(),
        p89: (words[from + n_idx..][..n_p].iter())
            .map(|&w| f32::from_le_bytes(w))
            .collect(),
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

/// The m=512 tape cut just after each MoE layer's `MoeRoute` step, its result that layer's
/// `topk_idx`: `'static` copies for one worker, bucket `ℓ` the cut after layer `ℓ`. Every span
/// stays inside the same arena, so each validates as the full tape does. Built once per binary.
fn cut_tapes() -> &'static [CudaTape] {
    static CUTS: OnceLock<&'static [CudaTape]> = OnceLock::new();
    CUTS.get_or_init(|| {
        let full = (CUDA_TAPES.iter())
            .find(|t| t.bucket_m == M)
            .expect("the m=512 tape");
        let moe = full.moe.expect("the m=512 tape's MoE scratch");
        let cuts: Vec<CudaTape> = (full.steps.iter().enumerate())
            .filter_map(|(i, s)| match *s {
                CudaStep::MoeRoute(MoeRouteBf16 {
                    topk_idx, top_k, ..
                }) => {
                    assert_eq!(topk_idx, moe.topk_idx.offset, "step {i} writes topk_idx");
                    assert_eq!(top_k as usize, TOP_K, "step {i}'s top-k");
                    Some(CudaTape {
                        steps: Box::leak(full.steps[..=i].to_vec().into_boxed_slice()),
                        result: moe.topk_idx,
                        ..*full
                    })
                }
                _ => None,
            })
            .collect();
        assert_eq!(cuts.len(), LAYERS, "one MoeRoute step per layer");
        Box::leak(cuts.into_boxed_slice())
    })
}

/// The device and the loaded weights.
fn rig() -> (CudaDevice, Vec<DeviceBuffer>) {
    let device = CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"));
    let checkpoint = PathBuf::from(
        meta()["checkpoint"]
            .as_str()
            .expect("meta.json names the checkpoint"),
    );
    let weights = load_weights(&device, &checkpoint, CUDA_TAPES, CUDA_WEIGHTS)
        .unwrap_or_else(|e| panic!("load_weights: {e}"));
    (device, weights)
}

/// CUDA's picks at `layer` for every real row of `ids`, from fresh state: per row, its 8
/// experts as read from `topk_idx`.
fn cuda_picks(w: &mut CudaWorker<'_>, layer: usize, ids: &[u32]) -> Vec<BTreeSet<i32>> {
    let tape = &cut_tapes()[layer];
    let q = ids.len() as u32;
    let mut padded = vec![0u32; M as usize];
    padded[..ids.len()].copy_from_slice(ids);
    let positions: Vec<u32> = (0..M).collect();
    let kv = KvSequence {
        block_table: SEAT.0,
        prefix_len: 0,
        q_len: q,
    };
    let gdn = GdnSequence {
        slot: SEAT.1,
        q_len: q,
        fresh: true,
    };
    let mut out = vec![0u16; tape.result.bytes as usize / 2];
    w.forward(
        layer,
        Input::TokenIds(&padded),
        &positions,
        Some(&kv),
        Some(&gdn),
        &mut out,
    )
    .unwrap_or_else(|e| panic!("the forward cut after layer {layer}'s routing: {e}"));
    let idx: Vec<i32> = (out.as_chunks::<2>().0.iter())
        .map(|&[lo, hi]| (u32::from(lo) | u32::from(hi) << 16) as i32)
        .collect();
    (idx.chunks(TOP_K).take(ids.len()))
        .map(|row| row.iter().copied().collect())
        .collect()
}

/// A pick set that is not 8 distinct experts in range: counted as a flip, and printed.
fn malformed(set: &BTreeSet<i32>) -> bool {
    set.len() != TOP_K || set.iter().any(|&e| !(0..EXPERTS).contains(&e))
}

#[test]
fn p9_expert_picks_match_hf_f32() {
    let (device, weights) = rig();
    let mut w = CudaWorker::new(
        &device,
        cut_tapes(),
        &weights,
        Some(KV_POOL),
        Some(GDN_POOL),
    )
    .unwrap_or_else(|e| panic!("CudaWorker::new over the 40 cut tapes: {e}"));
    let mut pass = true;
    for p in prompts() {
        let n = p.ids.len();
        assert!(n > 0 && n <= M as usize, "prompt {}: {n} ids", p.p);
        let route = read_route(&format!("route_{}.bin", p.p), n);
        let (mut total_cuda, mut total_bf16) = (0usize, 0usize);
        for layer in 0..LAYERS {
            let cuda = cuda_picks(&mut w, layer, &p.ids);
            let tau = route.tau(layer);
            let (mut ties, mut inside, mut outside, mut bf16_flips, mut bad) = (0, 0, 0, 0, 0);
            for (row, got) in cuda.iter().enumerate() {
                let want = route.f32.set(n, layer, row);
                let (tie, _) = route.tie(layer, row, tau);
                ties += usize::from(tie);
                bad += usize::from(malformed(got));
                bf16_flips += usize::from(route.bf16.set(n, layer, row) != want);
                if *got != want {
                    if tie {
                        inside += 1;
                    } else {
                        outside += 1;
                    }
                }
            }
            let flips = inside + outside;
            total_cuda += flips;
            total_bf16 += bf16_flips;
            let (ok, rule) = if layer == 0 {
                (outside == 0, "0 flips outside router ties".to_string())
            } else {
                (
                    outside == 0,
                    "0 flips outside router ties, those inside not capped (the user's decision \
                     2026-10-26/27)"
                        .to_string(),
                )
            };
            eprintln!(
                "P9 prompt {} layer {layer}: {n} tokens, router ties={ties} (tau={tau:.4e}) \
                 CUDA flips={flips} (inside ties {inside}, outside {outside}) HF-bf16 \
                 flips={bf16_flips} malformed CUDA picks={bad} (need {rule}) {}",
                p.p,
                verdict(ok)
            );
            pass &= ok;
        }
        eprintln!(
            "P9 prompt {}: all layers CUDA flips={total_cuda} HF-bf16 flips={total_bf16} \
             (information only)",
            p.p
        );
    }
    assert!(
        pass,
        "an expert pick is out of agreement: see the table above"
    );
}

/// P5/P7 attribution, printed only: for each of HF-f32's 8 greedy steps on prompts 1-5, the
/// sequence so far forwarded fresh and, at each layer, its last token's CUDA pick against HF-f32's
/// for that step; the first layer where they differ, whether it is a router tie there (under the
/// step's own τ over the 8 steps), or none. A P5/P7 divergence at step `s` reads off row `s`.
#[test]
fn p9_greedy_steps_first_flip_layer() {
    let (device, weights) = rig();
    let mut w = CudaWorker::new(
        &device,
        cut_tapes(),
        &weights,
        Some(KV_POOL),
        Some(GDN_POOL),
    )
    .unwrap_or_else(|e| panic!("CudaWorker::new over the 40 cut tapes: {e}"));
    for p in prompts().into_iter().take(GREEDY_PROMPTS) {
        let path = goldens_dir().join(format!("greedy_{}.json", p.p));
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let golden: serde_json::Value = serde_json::from_str(&text).expect("greedy json");
        let tokens: Vec<u32> = (golden["steps"].as_array().expect("steps").iter())
            .map(|s| s["token"].as_u64().expect("a token") as u32)
            .collect();
        let route = read_route(&format!("route_greedy_{}.bin", p.p), tokens.len());
        for (step, &token) in tokens.iter().enumerate() {
            let mut seq = p.ids.clone();
            seq.extend_from_slice(&tokens[..step]);
            assert!(seq.len() <= M as usize, "prompt {} step {step}", p.p);
            let first = (0..LAYERS).find_map(|layer| {
                let cuda = cuda_picks(&mut w, layer, &seq);
                let got = &cuda[seq.len() - 1];
                (*got != route.f32.set(route.rows, layer, step)).then(|| {
                    let tau = route.tau(layer);
                    let (tie, margin) = route.tie(layer, step, tau);
                    format!(
                        "layer {layer} (f32 rank-8 minus rank-9 {margin:.4e} vs tau {tau:.4e}: {})",
                        if tie {
                            "a router tie"
                        } else {
                            "NOT a router tie"
                        }
                    )
                })
            });
            eprintln!(
                "P9 greedy prompt {} step {step} (HF-f32 token {token}): first layer whose \
                 last-token pick differs from HF-f32's: {} (information only)",
                p.p,
                first.as_deref().unwrap_or("none")
            );
        }
    }
}
