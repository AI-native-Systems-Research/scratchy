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
//! - Gate, layers 1-39: CUDA's flips ≤ 2 × HF-bf16's + 2 (P2's shape: within twice the yardstick;
//!   the +2 keeps a layer where bf16 flips nothing from failing on one rounding flip). After a
//!   flip the next layer's input legitimately differs, so its flips outside ties are printed, not
//!   gated. The multiplier and the +2 are read, not run: if this gate fails while every flip is
//!   inside a router tie, stop and report the table.
//! - Attribution for P5/P7, printed: per greedy step of prompts 1-5 (HF-f32's sequence), the
//!   first layer at which the step's last token's CUDA pick differs from HF-f32's.
//!
//! Built only under `nvcc` and `qwen3.6-35b-a3b`, and needs the H100, the checkpoint and the
//! goldens, as `cuda_hf_parity_35b`; run with `--test-threads=1`. A missing goldens dir fails with
//! the command that makes it, never skips. Every line prints before a test fails.

use scratchy_models::qwen3_5_moe::qwen3_6_35b_a3b::{CUDA_TAPES, CUDA_WEIGHTS};

include!("common/cuda_route_parity_body.rs");

/// The goldens: `cuda_hf_reference.py --model qwen3.6-35b-a3b`'s.
const GOLDENS: &str = "tests/goldens/qwen3-5-moe-qwen3.6-35b-a3b-hf";
const MODEL: &str = "qwen3.6-35b-a3b";
const CHECKPOINT: &str = "Qwen3.6-35B-A3B";
/// The MoE layers and the picks per token.
const LAYERS: usize = 40;
const TOP_K: usize = 8;
const EXPERTS: i32 = 256;
/// Two GDN slots; the cut tapes' GDN layers run as the full tape's.
const GDN_POOL: Option<GdnPoolConfig> = Some(GdnPoolConfig { num_slots: 2 });
