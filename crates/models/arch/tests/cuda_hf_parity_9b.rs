// SPDX-License-Identifier: Apache-2.0
//! qwen3.5-9b's CUDA prefill and decode, run from the real `Qwen/Qwen3.5-9B` checkpoint on the
//! H100, against HuggingFace transformers on the same token ids.
//!
//! The same checks as qwen3.5-0.8b's `cuda_hf_parity.rs`, at the same thresholds (the tests are
//! `common/cuda_hf_parity_body.rs`; AI-native-Systems-Research/scratchy#839). The reference is
//! `tests/cuda_hf_reference.py --model qwen3.5-9b`'s goldens
//! (`tests/goldens/qwen3-5-qwen3.5-9b-hf/`), of the pinned revision
//! `c202236235762e1c871ad0ccb60c8ee5ba337b9a`: per prompt, HF's float32 logits (the reference),
//! its bfloat16 logits (the noise yardstick), the token ids it tokenized once, and 8 greedy
//! float32 tokens for prompts 1-5. The weights load through `load_weights` and `CUDA_WEIGHTS`
//! from the four-shard checkpoint; the forward is `CudaWorker` over `CUDA_TAPES`.
//!
//! - load: 355 weight buffers and 3 worker allocations; every buffer's bytes are its
//!   `weight_bytes`; embed rows 0 and 248319, `lm_head.weight` rows 0 and 248319 (the 9b's
//!   lm_head is untied, its own tensor) and layer 0's GDN bundle (unpacked) equal the
//!   checkpoint's bytes; the loader's time from opening the checkpoint to the last upload.
//! - P1 cosine: every real row of every prompt has cosine(CUDA, f32) ≥ 0.999.
//! - P2 error size: per prompt, RMS(CUDA − f32) ≤ 2 × RMS(noise), max |CUDA − f32| ≤ 2 × max
//!   |noise|.
//! - P3 top-1: CUDA's top-1 is f32's on every row whose f32 margin exceeds twice that row's max
//!   |noise|; the last row of each prompt agrees or is a near-tie. The share of all real rows that
//!   agree is printed, as information only (it was 245/249 = 0.9839 here, every disagreement a
//!   near-tie).
//! - P4 top-5: on each prompt's last row, the CUDA and f32 top-5 share at least 4 tokens.
//! - P5 greedy: prompts 1-5, 8 tokens by fresh m=512 forwards, equal HF-f32's, each first
//!   divergence a near-tie (a margin under twice the step's max |noise|).
//! - P6 chunked: prompt 6 as 128 fresh tokens then the rest gives rows 128.. bit-identical to one
//!   forward of it.
//! - P7 greedy via decode: one prefill then 7 decodes give HF-f32's 8 tokens under P5's rule.
//! - P8 decode row: the first L − 1 tokens prefilled, then token L − 1 decoded, against HF-f32's
//!   last row under P1-P4 and against CUDA's own one-shot last row (cosine ≥ 0.999).
//!
//! The thresholds are #816's and #830's, read not run at the 9b's depth. One rig holds about
//! 21 GB of the H100's 80 GB (17.9 GB of weights, the 3.2 GB m=4096 arena, the pools), so run
//! this binary with `--test-threads=1`. Built only under `nvcc` and `qwen3.5-9b-text-only`
//! (`required-features`); the bare `qwen3.5-9b` feature selects the vision architecture
//! `qwen3-5-vl`, and under `cuda` fails the build with "no models selected". Needs an sm_90 GPU
//! and the goldens: a missing goldens dir fails with the command that makes it, never skips.
//! `prompt_1_one_shot` and `prompt_1_decode` run alone for the sanitizers. Every check prints its
//! line before a test fails.

use scratchy_models::qwen3_5::qwen3_5_9b::{CUDA_TAPES, CUDA_WEIGHTS};

include!("common/cuda_hf_parity_body.rs");

const HIDDEN: usize = 4096;
const VOCAB: usize = 248_320;
/// Two GDN slots, one per seat; the worker's allocations are the arena, the KV pool and the GDN pool.
const GDN_POOL: Option<GdnPoolConfig> = Some(GdnPoolConfig { num_slots: 2 });
const WORKER_ALLOCATIONS: u64 = 3;
/// The weight buffers `load_weights` uploads: 355, the untied lm_head's among them.
const WEIGHTS: u64 = 355;
/// Untied (`tie_word_embeddings` false): the lm_head's own tensor, whose rows `load_checks` checks.
const UNTIED_LM_HEAD: Option<&str> = Some("lm_head.weight");
/// The goldens: `cuda_hf_reference.py --model`, the checkpoint, and where they are written.
const MODEL: &str = "qwen3.5-9b";
const CHECKPOINT: &str = "Qwen3.5-9B";
const GOLDENS: &str = "tests/goldens/qwen3-5-qwen3.5-9b-hf";
/// P1's rule (#859): min cosine(CUDA, f32) ≥ 0.999 on every real row (0.8b runs, #816).
const P1: P1Rule = P1Rule::Floor(0.999);
