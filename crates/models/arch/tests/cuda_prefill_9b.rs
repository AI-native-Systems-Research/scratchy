// SPDX-License-Identifier: Apache-2.0
//! Whole-model prefill and decode forwards of qwen3.5-9b's emitted CUDA tapes on the H100.
//!
//! The same checks (a) to (o) as qwen3.5-0.8b's `cuda_prefill.rs`, at the same thresholds but
//! (n)'s (`BOUNDARY_DECODE_COSINE`; the tests are `common/cuda_prefill_body.rs`), on the 9b's
//! tapes (AI-native-Systems-Research/scratchy#839). The tapes are `CUDA_TAPES`, exactly as `#[forward]` emitted them; the weights
//! are synthetic, each buffer sized from the tapes' `weight_bytes` and filled by what reads it: a
//! GEMM's `[N, K]` matrix uniform in ±1/√K, a norm's zero-centred gain in ±0.1, a GDN layer's
//! packed f32 bundle in the ranges the GDN kernel tests use. The embedding table is filled as a
//! `[vocab, hidden]` matrix of `k = hidden`: ±1/√4096. The 9b's lm_head is untied: its own
//! `lm_head.weight`, a `[vocab, hidden]` matrix of `k = 4096`, filled from its own seed, so (c)
//! checks the lm_head GEMM `(512, 248320, 4096)` against a B the embed never reads. The input is
//! token ids, uniform over the vocabulary and seeded; padding rows are id 0.
//!
//! The 9b's numbers: 355 weight buffers (17.9 GB, uploaded from the host, where they are built
//! once per binary), then 3 worker allocations (the arena, the KV pool, the GDN pool); the arena
//! is the m=4096 tape's 3,202,384,896 bytes. One rig holds about 21 GB of the H100's 80 GB, so
//! run this binary with `--test-threads=1`.
//!
//! Built only under `nvcc` and `qwen3.5-9b-text-only` (`required-features`), and needs an sm_90
//! GPU: no GPU fails rather than skips. The bare `qwen3.5-9b` feature selects the vision
//! architecture `qwen3-5-vl`; under `cuda` it fails the build with "no models selected". Every
//! check prints its line before a test fails, so one run reports every number.

// `pub`: this binary uses part of `common`, and nothing in it is dead code.
#[path = "../../../targets/cuda/tests/common/numerics.rs"]
pub mod numerics;

use scratchy_models::qwen3_5::qwen3_5_9b::CUDA_TAPES;

include!("common/cuda_prefill_body.rs");

const HIDDEN: usize = 4096;
/// The weight buffers: the 355 the tapes read (the untied lm_head's among them).
const WEIGHTS: u64 = 355;
/// The worker's arena: the largest tape's, m=4096.
const ARENA: u64 = 3_202_384_896;
/// (n)'s floor: each decode at the block boundary against the same row of one forward. 0.998, not
/// the 0.8b's 0.999: set from the measured 9b value, 0.998905 at P = 320 on the H100 (its
/// teacher-forced decodes were at 0.99926–0.99938), because a deeper, wider model accumulates
/// more bf16 difference between a decode and one forward.
const BOUNDARY_DECODE_COSINE: f64 = 0.998;
/// (k) teacher-forced's floor, #830.
const TEACHER_FORCED_DECODE_COSINE: f64 = 0.999;
