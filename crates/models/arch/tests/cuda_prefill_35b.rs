// SPDX-License-Identifier: Apache-2.0
//! Whole-model prefill and decode forwards of Qwen3.6-35B-A3B's emitted CUDA tapes on the H100.
//!
//! The same checks (a) to (o) as qwen3.5-9b's `cuda_prefill_9b.rs` (the tests are
//! `common/cuda_prefill_body.rs`), on the 35B's tapes (AI-native-Systems-Research/scratchy#859),
//! all 40 layers with their mixture-of-experts blocks. The tapes are `CUDA_TAPES`, exactly as
//! `#[forward]` emitted them; the weights are synthetic, each buffer sized from the tapes'
//! `weight_bytes` and filled by what reads it, as for the 9b, and each layer's MoE bundle as the
//! GEMMs that read it: the router `[256, 2048]` and every expert's gate and up rows uniform in
//! ±1/√2048, its down rows in ±1/√512. The lm_head is untied, its own `lm_head.weight`. The input
//! is token ids, uniform over the vocabulary and seeded; padding rows are id 0.
//!
//! The 35B's numbers: 523 weight buffers (69.3 GB, uploaded from the host, where they are built
//! once per binary), then 3 worker allocations (the arena, the KV pool, the GDN pool); the arena
//! is the m=4096 tape's 3,617,371,392 bytes, of which the MoE scratch is 737,939,712. One rig
//! holds about 73 GB of the H100's 80 GB, and the host copy another 69.3 GB of RAM, so run this
//! binary with `--test-threads=1`.
//!
//! (n)'s cosine is printed, not gated: the 35B's floor is set by the user from this run
//! (`boundary_floor_gates`); its finiteness still gates. Every other check is the 9b's, at the
//! 9b's thresholds.
//!
//! Built only under `nvcc` and `qwen3.6-35b-a3b` (`required-features`), and needs an sm_90 GPU:
//! no GPU fails rather than skips. Every check prints its line before a test fails, so one run
//! reports every number.

// `pub`: this binary uses part of `common`, and nothing in it is dead code.
#[path = "../../../targets/cuda/tests/common/numerics.rs"]
pub mod numerics;

use scratchy_models::qwen3_5_moe::qwen3_6_35b_a3b::CUDA_TAPES;

include!("common/cuda_prefill_body.rs");

const HIDDEN: usize = 2048;
/// The weight buffers: the 523 the tapes read (30 GDN bundles, 40 MoE bundles, the untied
/// lm_head's among them).
const WEIGHTS: u64 = 523;
/// The worker's arena: the largest tape's, m=4096 (`cuda_tapes_35b`'s `ARENAS`).
const ARENA: u64 = 3_617_371_392;
/// (n)'s line prints against the 9b's 0.998 for reference only: for this MoE model the floor is
/// the user's (`boundary_floor_gates` is false), set from this run's printed cosines.
const BOUNDARY_DECODE_COSINE: f64 = 0.998;
