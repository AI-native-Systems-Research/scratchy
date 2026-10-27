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
//! - Layers 1-29: the shared body's one rule (`layer_rule`), the 35B's. The user's decision on it
//!   is open (#933): if it fails while every flip is inside a router tie, report the table beside
//!   the 35B's; that is not this binary's failure. A flip outside a router tie at layer 0 is.
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
