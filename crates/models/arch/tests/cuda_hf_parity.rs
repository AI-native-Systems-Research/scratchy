// SPDX-License-Identifier: Apache-2.0
//! qwen3.5-0.8b's CUDA prefill and decode, run from the real `Qwen/Qwen3.5-0.8B` checkpoint on the
//! H100, against HuggingFace transformers on the same token ids.
//!
//! The reference is `tests/cuda_hf_reference.py`'s goldens (`tests/goldens/qwen3-5-qwen3.5-0.8b-hf/`):
//! per prompt, HF's float32 logits (the reference), its bfloat16 logits (the noise yardstick: how
//! far a bf16 forward of the same weights lands from the float32 one), the token ids it tokenized
//! once, and 8 greedy float32 tokens for prompts 1-5. "Noise" below means HF-bf16 against f32 on
//! the same rows. The weights load through `load_weights` and `CUDA_WEIGHTS`; the forward is
//! `CudaWorker` over `CUDA_TAPES`, the same calls an emitted forward will make.
//!
//! - load: 266 weight buffers and 3 worker allocations; every buffer's bytes are its
//!   `weight_bytes`; embed rows 0 and 248319 and layer 0's GDN bundle (unpacked) equal the
//!   checkpoint's bytes; the loader's time from opening the checkpoint to the last upload.
//! - P1 cosine: every real row of every prompt has cosine(CUDA, f32) ≥ 0.999.
//! - P2 error size: per prompt, RMS(CUDA − f32) ≤ 2 × RMS(noise), max |CUDA − f32| ≤ 2 × max
//!   |noise|.
//! - P3 top-1: CUDA's top-1 is f32's on every row whose f32 margin (top-1 − top-2) exceeds twice
//!   that row's max |noise|; the rest are near-ties, counted. The last row of each prompt agrees or
//!   is a near-tie. The share of all real rows that agree is printed, as information only.
//! - P4 top-5: on each prompt's last row, the CUDA and f32 top-5 share at least 4 tokens.
//! - P5 greedy: prompts 1-5, 8 tokens, each a fresh m=512 forward of the prompt and CUDA's own
//!   tokens so far, equal HF-f32's; at the first divergent step a margin under twice that step's
//!   max |noise| makes the prompt a near-tie.
//! - P6 chunked: prompt 6 as 128 fresh tokens, then the rest continuing in the same blocks and
//!   slot, gives rows 128.. bit-identical to one forward of it (in disjoint blocks and slot).
//! - P7 greedy via decode: prompts 1-5, one fresh m=512 prefill (its argmax is token 1), then 7
//!   one-token decodes through the m=1 tape: the 8 tokens equal HF-f32's under P5's near-tie
//!   rule (a margin under twice the step's max |noise|).
//! - P8 decode row: prompts 1-5, the first L − 1 tokens prefilled, then token L − 1 decoded: that
//!   row against HF-f32's last row under P1-P4 (cosine, 2 × noise, top-1 unless near-tie, top-5
//!   ≥ 4), and against CUDA's own one-shot last row (cosine ≥ 0.999, the same argmax unless
//!   near-tie, the max |difference| printed). A one-token prompt is skipped, saying so.
//!
//! The thresholds are the issues' (AI-native-Systems-Research/scratchy#816 and #830), read not
//! run. Built only under `nvcc` and `qwen3.5-0.8b`, and needs an sm_90 GPU and the goldens: a
//! missing goldens dir fails with the command that makes it, never skips. `prompt_1_one_shot` is one forward
//! alone, and `prompt_1_decode` a prefill and 7 decodes alone, for the sanitizers. Every check
//! prints its line before a test fails.
//!
//! The tests are `common/cuda_hf_parity_body.rs`, shared with qwen3.5-9b's
//! `cuda_hf_parity_9b.rs`; this file is the 0.8b's module and its numbers.

use scratchy_models::qwen3_5::qwen3_5_0_8b::{CUDA_TAPES, CUDA_WEIGHTS};

include!("common/cuda_hf_parity_body.rs");

const HIDDEN: usize = 1024;
const VOCAB: usize = 248_320;
/// Two GDN slots, one per seat; the worker's allocations are the arena, the KV pool and the GDN pool.
const GDN_POOL: Option<GdnPoolConfig> = Some(GdnPoolConfig { num_slots: 2 });
const WORKER_ALLOCATIONS: u64 = 3;
/// The weight buffers `load_weights` uploads: 266 (the tied lm_head reads the embedding
/// table's).
const WEIGHTS: u64 = 266;
/// Tied: the lm_head is the embedding table, whose rows `load_checks` checks already.
const UNTIED_LM_HEAD: Option<&str> = None;
/// The goldens: `cuda_hf_reference.py --model`, the checkpoint, and where they are written.
const MODEL: &str = "qwen3.5-0.8b";
const CHECKPOINT: &str = "Qwen3.5-0.8B";
const GOLDENS: &str = "tests/goldens/qwen3-5-qwen3.5-0.8b-hf";
/// P1's rule (#859): min cosine(CUDA, f32) ≥ 0.999 on every real row (0.8b runs, #816).
const P1: P1Rule = P1Rule::Floor(0.999);
/// P3 accepts no top-1 disagreement outside a near-tie (#816).
const P3: P3Rule = P3Rule::Strict;
