// SPDX-License-Identifier: Apache-2.0
//! Whole-model prefill and decode forwards of qwen3.5-0.8b's emitted CUDA tapes on the H100.
//!
//! The tapes are `CUDA_TAPES`, exactly as `#[forward]` emitted them; the weights are synthetic,
//! each buffer sized from the tapes' `weight_bytes` and filled by what reads it: a GEMM's `[N, K]`
//! matrix uniform in ±1/√K, a norm's zero-centred gain in ±0.1, a GDN layer's packed f32 bundle
//! in the ranges the GDN kernel tests use, the embedding table's rows uniform in ±1/32 (which the
//! tied lm_head reads too, a `[vocab, hidden]` matrix, so it is filled as a matrix of `k =
//! hidden`: ±1/√1024). The input is token ids, uniform over the vocabulary and seeded; padding
//! rows are id 0. Checks:
//!
//! - (a) the worker takes every tape, and its arena is the largest tape's;
//! - (b) a fresh 300-token forward in the m=512 bucket gives finite logits on every real row;
//! - (c) the lm_head GEMM, read back from the arena, against an f64 reference on six rows over
//!   every column, under the GEMM bound of `numerics.rs`;
//! - (d) repeating (b) in other KV blocks and another GDN slot gives the same logits, bit for
//!   bit;
//! - (e) the same 300 rows in the m=1024 bucket give the same real-row logits, bit for bit;
//! - (f) a 512-token sequence in one forward, and as 300 fresh tokens then 212 continuing, agree
//!   bit for bit on rows 300..512;
//!
//! Each compared pair of (d), (e) and (f) runs its second sequence in KV blocks and a GDN slot
//! disjoint from the first's, so a run that failed to write its cache or state cannot pass by
//! reading the other's.
//! - (g) a KV sequence and a GDN sequence that disagree on fresh are refused before any copy;
//! - (h) the worker allocates the weights, the arena and the two pools, and a forward nothing;
//! - (i) a token id outside the vocabulary, a count other than the bucket's and hidden rows are
//!   refused before any copy;
//! - (j) the worker takes all five tapes (the m=1 decode tape among them), its arena the largest's,
//!   266 + 3 allocations at setup and none across one prefill and 8 decodes;
//! - (k) 300 tokens prefilled, then a chain of decodes of positions 300..307, each against the same
//!   row of one 308-token forward: finite, and the same greedy token (argmax) unless that one-shot
//!   row is a near-tie (top-1 − top-2 ≤ 2 × the step's max |difference|); the near-tie count,
//!   cosine, max |difference| and relative L2 printed, not asserted (on these near-flat synthetic
//!   logits most steps are near-ties, so this is a smoke check; P7 is the chain-token check that
//!   carries weight); the last decode's lm_head GEMV within the GEMM bound of f64 over every
//!   column;
//!   (k, teacher-forced) each decode one step off the one-shot trajectory (300 + i prefilled fresh,
//!   then one decode at 300 + i) at cosine ≥ 0.999; (k-diag) prints, per decode, every KV layer's
//!   K/V and every GDN layer's state against the one-shot's, for the chain and teacher-forced;
//! - (l) the prefill and 8 decodes repeated in disjoint blocks and slot are bit-identical;
//! - (m) the 8 decodes leave the cache at positions 0..300 bit-unchanged;
//! - (n) decodes at P = 319 and P = 320 (a new block) are finite and at cosine ≥
//!   `BOUNDARY_DECODE_COSINE` (0.999);
//! - (o) at m=1, a wrong position, a short block table, fresh at P > 0 and q_len = 2 are refused
//!   before any copy.
//!
//! Built only under `nvcc` (`required-features`), and needs an sm_90 GPU: no GPU fails rather
//! than skips. Every check prints its line before a test fails, so one run reports every number.
//!
//! The tests are `common/cuda_prefill_body.rs`, shared with qwen3.5-9b's `cuda_prefill_9b.rs`;
//! this file is the 0.8b's module and its numbers.

// `pub`: this binary uses part of `common`, and nothing in it is dead code.
#[path = "../../../targets/cuda/tests/common/numerics.rs"]
pub mod numerics;

use scratchy_models::qwen3_5::qwen3_5_0_8b::CUDA_TAPES;

include!("common/cuda_prefill_body.rs");

const HIDDEN: usize = 1024;
/// The weight buffers: the 266 the tapes read (the tied lm_head reads the embedding table's).
const WEIGHTS: u64 = 266;
/// The worker's arena: the largest tape's, m=4096.
const ARENA: u64 = 2_585_035_776;
/// (n)'s floor: each decode at the block boundary against the same row of one forward.
const BOUNDARY_DECODE_COSINE: f64 = 0.999;
/// (k) teacher-forced's floor, #830.
const TEACHER_FORCED_DECODE_COSINE: f64 = 0.999;
