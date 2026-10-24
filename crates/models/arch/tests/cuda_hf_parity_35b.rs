// SPDX-License-Identifier: Apache-2.0
//! Qwen3.6-35B-A3B's CUDA prefill and decode, run from the real `Qwen/Qwen3.6-35B-A3B` checkpoint
//! on the H100, against HuggingFace transformers on the same token ids.
//!
//! The same checks as qwen3.5-9b's `cuda_hf_parity_9b.rs` (the tests are
//! `common/cuda_hf_parity_body.rs`; AI-native-Systems-Research/scratchy#859), plus the memory gate
//! and the MoE bundles' load rows. The reference is `tests/cuda_hf_reference.py --model
//! qwen3.6-35b-a3b`'s goldens (`tests/goldens/qwen3-5-moe-qwen3.6-35b-a3b-hf/`), of the pinned
//! revision `995ad96eacd98c81ed38be0c5b274b04031597b0` (26 shards), its experts run as HF's
//! per-expert `eager` loop. The weights load through `load_weights` and `CUDA_WEIGHTS`; the
//! forward is `CudaWorker` over `CUDA_TAPES`.
//!
//! - memory_fits, run first and alone: `cudaMemGetInfo` and `nvidia-smi` at four points, 523 + 3
//!   allocations, all five buckets, the peak after an m=4096 forward and a decode at most 78.0 GiB
//!   of the 79.6 GiB, with the headroom.
//! - load: 523 weight buffers and 3 worker allocations, each its `weight_bytes`; embed and
//!   `lm_head.weight` rows 0 and 248319, layer 0's GDN bundle, and layers 0 and 39's MoE bundles
//!   (the router, experts 0 and 255's first `gate_up` and `down` rows, the shared-expert gate)
//!   equal the checkpoint's; the load time.
//! - P1 cosine: printed, not gated. The rule is min cosine(CUDA, f32) ≥ min cosine(HF-bf16, f32)
//!   − δ, δ the user's; each prompt prints both and their difference beside the 0.999 floor, which
//!   came from 0.8b runs. A flipped expert pick moves a token by a whole expert, so HF-bf16 itself
//!   may sit below 0.999 here: that line is expected, not a failure.
//! - P2, P3, P4, P6: the 9b's rules, gated.
//! - P5, P7: every first divergence a near-tie, at most `near_tie_cap()` prompts.
//! - P8: the decode row's error, top-1 and top-5 gated; its two cosines (against HF-f32, and
//!   against CUDA's own one-shot row, the 35B's (n) floor) printed until the user sets them.
//!
//! The expert picks (P9) are `cuda_route_parity_35b.rs`'s. One rig holds about 68.5 GiB of the
//! H100's 79.6 GiB (69.3 GB of weights, the 3.6 GB m=4096 arena, the pools), so run this binary
//! with `--test-threads=1`. Built only under `nvcc` and `qwen3.6-35b-a3b` (`required-features`).
//! Needs an sm_90 GPU and the goldens: a missing goldens dir fails with the command that makes it,
//! never skips. `memory_fits` and `load_checks` read only the goldens' `checkpoint`.
//! `prompt_1_one_shot` and `prompt_1_decode` run alone for the sanitizers. Every check prints its
//! line before a test fails.

use scratchy_models::qwen3_5_moe::qwen3_6_35b_a3b::{CUDA_TAPES, CUDA_WEIGHTS};

include!("common/cuda_hf_parity_body.rs");

const HIDDEN: usize = 2048;
/// The weight buffers `load_weights` uploads: 523 (30 GDN bundles, 40 MoE bundles, the untied
/// lm_head's among them).
const WEIGHTS: u64 = 523;
/// Untied (`tie_word_embeddings` false): the lm_head's own tensor, whose rows `load_checks` checks.
const UNTIED_LM_HEAD: Option<&str> = Some("lm_head.weight");
/// The goldens: `cuda_hf_reference.py --model`, the checkpoint, and where they are written.
const MODEL: &str = "qwen3.6-35b-a3b";
const CHECKPOINT: &str = "Qwen3.6-35B-A3B";
const GOLDENS: &str = "tests/goldens/qwen3-5-moe-qwen3.6-35b-a3b-hf";
