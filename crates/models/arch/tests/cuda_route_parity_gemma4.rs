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
