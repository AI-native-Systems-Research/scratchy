// SPDX-License-Identifier: Apache-2.0
//! Whole-model prefill and decode forwards of Gemma 4 26B-A4B's emitted CUDA tapes on the H100
//! (AI-native-Systems-Research/scratchy#925, part 13b-5 of #885).
//!
//! The checks (a) to (o) of the qwen binaries (`cuda_prefill.rs`, `cuda_prefill_9b.rs`,
//! `cuda_prefill_35b.rs`), where they apply, plus the sliding window's (p) to (r); the tests are
//! `common/cuda_prefill_gemma4_body.rs`, not the qwen body, which hard-codes qwen's vocabulary and
//! GDN pool. All 30 layers, as `#[forward]` emitted them (`CUDA_TAPES`, five buckets); the
//! weights are synthetic, each buffer sized from the tapes' `weight_bytes` and filled by what
//! reads it (the body's `weight`). The lm_head is tied: the embedding table, weight 0. The input
//! is token ids, uniform over the vocabulary and seeded; padding rows are id 0.
//!
//! The model's numbers: 627 weight buffers (50,466,283,580 bytes, uploaded from the host, where
//! they are built once per binary), then 2 worker allocations (the arena, the KV pool; there is no
//! GDN layer). The arena is the m=4096 tape's 3,513,339,392 bytes (`cuda_tapes_gemma4`'s
//! `ARENAS`; the soft-cap runs in place, #921), not #925's 5,660,823,040, which predates the
//! in-place soft-cap. The KV pool is 72 blocks of 64 tokens at 225,280 bytes a token (25 sliding
//! layers of 8 × 256, 5 global ones of 2 × 512, K and V): 1,038,090,240 bytes, two seats of 2,304
//! tokens. One rig holds about 55 GB of the H100's 80 GB, and the host copy another 50.5 GB of
//! RAM, so run this binary with `--test-threads=1`.
//!
//! Built only under `nvcc` and `gemma-4-26b-a4b-it` (`required-features`), and needs an sm_90
//! GPU: no GPU fails rather than skips. Every check prints its line before a test fails, so one
//! run reports every number.

// `pub`: this binary uses part of `common`, and nothing in it is dead code.
#[path = "../../../targets/cuda/tests/common/numerics.rs"]
pub mod numerics;

use scratchy_models::gemma4_moe::gemma_4_26b_a4b_it::CUDA_TAPES;

include!("common/cuda_prefill_gemma4_body.rs");

const HIDDEN: usize = 2816;
const VOCAB: usize = 262_144;
/// The weight buffers: the 627 the tapes read (30 expert banks, the tied lm_head the embedding
/// table's), and their bytes (`cuda_tapes_gemma4`'s `WEIGHT_BYTES`).
const WEIGHTS: u64 = 627;
const WEIGHT_BYTES: u64 = 50_466_283_580;
/// The worker's arena: the largest tape's, m=4096 (`cuda_tapes_gemma4`'s `ARENAS`).
const ARENA: u64 = 3_513_339_392;
/// The KV pool at [`KV_POOL`]'s 72 blocks: `72 · 64 · 225,280` bytes.
const KV_POOL_BYTES: u64 = 1_038_090_240;
/// The model's layers; the sliding layers' window, in keys (`sliding_window`), and how many of
/// the layers slide.
const LAYERS: usize = 30;
const WINDOW: usize = 1024;
const SLIDING_LAYERS: usize = 25;
/// The final soft-cap's cap (`final_logit_softcapping`).
const CAP: f64 = 30.0;
