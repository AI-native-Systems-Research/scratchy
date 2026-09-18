// SPDX-License-Identifier: Apache-2.0
//! `CudaWorker`: `Worker` implementation backed by the CUDA GPU runtime.
//!
//! CUDA tensor runtime + scratchy-forward-compiler (full feature set:
//! CUDA graphs, NCCL, FP8, etc.).
//!
//! The `scratchy_models` / `scratchy_builder_cuda` link anchors that keep the
//! `#[forward]` inventory rows + kernel `.a` files in the final binary live in
//! the worker binary crate (`scratchy-serving-worker`), not here — this crate
//! has no normal dep on either.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use scratchy_core_common::SamplingParams;
use scratchy_core_common::engine_io::EmbeddingData;
use scratchy_core_config::CudaGraphMode;
use scratchy_core_model::weight::HfModelConfig;
use scratchy_serving_engine::executor::ModelRunnerOutput;
use scratchy_serving_scheduler::scheduler::output::SchedulerOutput;
use tracing::info;

use crate::dtype::DType as GpuDType;
use crate::gdn_state::GdnStatePool;
#[cfg(feature = "cuda")]
use crate::kv_cache::CudaKvCacheExt;
use crate::kv_cache::KvCachePool;
#[cfg(feature = "cuda")]
use crate::weights::CudaWeightsExt;
use scratchy_forward_compiler::gdn_slot_allocator::GdnSlotAllocator;

use crate::OwnedTensor;
use crate::cpu_gpu_buf::PinnedBuf;
use crate::device::GpuDevice;
use crate::driver;
use crate::graph::{CudaGraphRunner, PrefillGraphRunner};
use crate::logits_processor::{
    AllowedTokenIdsProcessor, BadWordsProcessor, BatchUpdate, GrammarMaskProcessor,
    LogitBiasProcessor, LogitsProcessor, LogitsProcessorPipeline, MinTokensProcessor,
    PenaltiesProcessor, SealPadProcessor,
};
use crate::quant;
use crate::tensor::{GpuTensor, TensorView};

// Neutral worker interface lives in scratchy-serving-engine.
use scratchy_serving_engine::error::{ExecutorError, ExecutorResult};
use scratchy_serving_engine::input_batch::InputBatch;
use scratchy_serving_engine::input_batch::PreparedInputs;
use scratchy_serving_engine::worker::Worker;
use scratchy_serving_engine::worker_factory::WorkerCreateConfig;
use scratchy_serving_engine::worker_factory::resolve_model_path;

// Backend-neutral budget/bucket helpers from the engine (cycle-free seam).
use scratchy_serving_engine::gpu_budget::{compute_available_kv_bytes, gdn_slot_key};

/// Per-image MRoPE metadata in seq space: `(seq_offset, length, grid_t,
/// grid_h_merged, grid_w_merged)`, `seq_offset` relative to the req's full
/// sequence start (same coordinate system as `PlaceholderRange.offset`).
type SeqMmInfo = (u32, u32, u32, u32, u32);

/// Per-forward multimodal inputs threaded into `CudaModel::forward` /
/// `hidden_states`. Built once per step in `execute_model_inner` from the
/// active batch's cached `MultimodalData`, consumed only by the
/// `Self::Scratchy` arm — every other (text-only) arm ignores it.
///
/// `mm_embeds` is the projected vision-encoder output `[total_mm_tokens,
/// hidden]` produced by `MultimodalForward::vision_forward`;
/// `embed_patches` names where each image's slice lands in the flat input-
/// id sequence (token-space). The Scratchy arm copies both into
/// `ForwardCtx`, where `Instruction::Embed::eval` D2D-splices each patch
/// into the gather output. Empty / `None` is the text-only no-op path —
/// byte-identical to before MM landed.
#[cfg(feature = "cuda")]
pub struct MmForwardInputs<'a> {
    pub mm_embeds: TensorView<'a>,
    pub embed_patches: &'a [crate::EmbedPatch],
}

/// Per-step inputs for the GDN (Gated Delta-Net) hybrid path. Mirrors
/// [`MmForwardInputs`]: a `cfg(feature = "cuda")` aggregate the active
/// batch's executor parks once and the `CudaModel::forward` /
/// `CudaModel::hidden_states` arms unpack into `ForwardCtx::{gdn_state,
/// gdn_state_indices, gdn_is_fresh}`.
///
/// Only populated for hybrid arches (Qwen3.5 / Qwen3-Next) whose
/// `ScratchyWeights::gdn_runtime_config()` returns `Some(_)`. Pure-attention
/// arches leave the argument as `None`, the unpack degenerates to
/// `gdn_state: None, gdn_state_indices: None, gdn_is_fresh: None`, and
/// behavior is byte-identical to the pre-hybrid path.
///
/// Lifetimes: `state` borrows the worker-owned `GdnStatePool` (one per
/// `CudaWorker`, sized at `initialize_cache`); `indices` and `is_fresh`
/// borrow `TensorView`s built from `OwnedTensor`s H2D-uploaded each step
/// from `self.gdn_pending`. All three must outlive the forward call.
#[cfg(feature = "cuda")]
pub struct GdnForwardInputs<'a> {
    pub state: &'a crate::gdn_state::GdnStatePool<crate::PoolMem>,
    pub indices: TensorView<'a>,
    pub is_fresh: TensorView<'a>,
}

/// Per-step per-KV-group attention tensors for gemma4 hybrid SWA. Mirrors
/// [`GdnForwardInputs`]: the executor builds it once per forward and the
/// `CudaModel::{forward,hidden_states}` arms unpack it into
/// `ForwardCtx::{block_tables, slot_mappings}`. `None` on uniform models —
/// the ctx vecs stay empty and the attention ops fall back to the scalar
/// `block_table` / `slot_mapping` (byte-identical to the pre-hybrid path).
///
/// `block_tables[g]` / `slot_mappings[g]` are group `g`'s tensors (group 0 =
/// full/global, 1.. = sliding), built by `build_attention_tensors` from
/// `AttentionMetadata::sliding_groups`. Both slices and the `TensorView`s
/// they hold must outlive the forward call (they borrow `OwnedTensor`s parked
/// in the forward scope).
#[cfg(feature = "cuda")]
pub struct HybridKvInputs<'a> {
    pub block_tables: &'a [TensorView<'a>],
    pub slot_mappings: &'a [TensorView<'a>],
}

/// Return of [`build_attention_tensors`]: `(cu_seqlens_q, seqused_k, max_q,
/// max_k, per-group slot_mappings, per-group block_tables)`. Group 0 is the
/// full/global group (page-unified `full_block_size`), groups `1..` are the
/// sliding groups (base `block_size`); on uniform models there is one group.
#[cfg(feature = "cuda")]
type AttnTensors = (
    OwnedTensor,
    OwnedTensor,
    usize,
    usize,
    Vec<OwnedTensor>,
    Vec<OwnedTensor>,
);

/// A dense-bf16 Llama loaded through the new scratchy-forward-compiler
/// compiler. Holds the specialized `Weights` enum (one variant per
/// compiled config) + the `RotaryCache` the forward needs. Accessors
/// (num_layers / num_kv_heads / head_dim / vocab_size / hidden_size)
/// come directly from `Weights`'s methods — baked in at macro-
/// expansion time from each model's config.json.
///
/// Every model routes through this — `CudaModel` has no other variant
/// (the hand-written CUDA model forwards were removed in cbc59cc2d).
/// Any scratchy-forward-compiler-compiled model, loaded through
/// [`crate::try_load`]. The trait object erases the arch;
/// accessors + forward/backbone go through the `ScratchyWeights`
/// vtable. One struct covers every registered arch (llama, qwen2,
/// qwen3, gemma2, granite, and any future arch that lands a
/// `#[forward]` module) — adding a new arch requires zero lines
/// here.
#[cfg(feature = "cuda")]
pub struct ScratchyModel {
    pub weights: Box<dyn scratchy_forward_compiler::ScratchyWeights>,
    /// Vision-encoder handle for multimodal arches. `Some(_)` only
    /// when the loaded checkpoint carries `visual.*` tensors AND the
    /// arch has submitted a `ScratchyMmRegistration` row claiming the
    /// HF arch string. The worker calls
    /// `mm.vision_forward(pixel_batches, placeholders, device)` when
    /// the request batch carries `mm_data`; the result threads into
    /// `ForwardCtx.mm_embeds` / `embed_patches` for the
    /// `Instruction::Embed::eval` splice. `None` for every text-only
    /// arch — the splice arm sees `embed_patches.is_empty()` and
    /// degenerates to plain `embedding_gather_masked`.
    pub mm: Option<Box<dyn crate::MultimodalForward>>,
    /// NCCL communicator the worker injects via `set_tp_group` after
    /// construction. Threaded into `ForwardCtx::tp_group` on every
    /// forward call — the universal `Instruction::AllReduce` arm
    /// (gated under `feature = "nccl"`) reads it and dispatches into
    /// `NcclGroup::all_reduce_inplace`. `None` at tp=1 (the lowering
    /// pass emits zero AllReduce rows, so the field is never read).
    #[cfg(feature = "nccl")]
    pub tp_group: Option<std::sync::Arc<crate::nccl::NcclGroup>>,
    /// Tensor-parallel world size baked into the loaded `Weights`
    /// variant — used to derive the per-rank `num_kv_heads` for KV
    /// cache allocation. `ScratchyWeights::num_key_value_heads()` is
    /// the *unsharded* config value (sharded values flow through
    /// `CanonicalParams`); the executor must divide by `tp_world_size`
    /// here to match hand-written `self_attn.num_kv_heads` (which is
    /// already sharded) and Python's per-rank `num_kv_heads`. Without
    /// this, `KvCachePool` is sized with the unsharded head count and
    /// the per-block stride disagrees with the kernel-side per-rank
    /// `NUM_KV_HEADS` from `CanonicalParams`, producing decode garbage
    /// at tp>1.
    pub tp_world_size: usize,
}

/// Loaded model — always a scratchy-forward-compiler-compiled arch.
///
/// Hand-written CUDA model forwards have been removed; every architecture
/// now flows through `crate::try_load`. The single-variant enum
/// shape is preserved so existing match sites remain syntactically valid.
#[cfg(feature = "cuda")]
enum CudaModel {
    /// Any scratchy-forward-compiler-compiled arch. Populated by
    /// `crate::try_load(gw, stream, hf_arch_name)` —
    /// one variant covers every registered `#[forward]` module
    /// (llama / qwen2 / qwen3 / gemma2 / granite / …). Boxed
    /// because the per-arch `Weights` enum has one variant per
    /// compiled model config and can be large.
    Scratchy(Box<ScratchyModel>),
}

#[cfg(feature = "cuda")]
impl CudaModel {
    fn num_layers(&self) -> usize {
        match self {
            Self::Scratchy(m) => m.weights.num_hidden_layers() as usize,
        }
    }

    /// Whether this model is a scratchy-forward-compiler-compiled encoder (no lm_head Gemm,
    /// returns hidden states from `forward_backbone`). Encoders skip CUDA
    /// graph capture and use the fixed-estimate profiling path. Today only
    /// modernbert qualifies; add new arch names here as they land.
    fn is_scratchy_encoder(&self) -> bool {
        match self {
            Self::Scratchy(m) => matches!(m.weights.arch_name(), "modernbert"),
        }
    }

    fn num_kv_heads(&self) -> usize {
        match self {
            // ScratchyWeights reports the unsharded config value;
            // divide by tp_world_size so KvCachePool gets the
            // per-rank head count, matching Python's
            // `max(1, total // tp_size)`.
            Self::Scratchy(m) => {
                let total = m.weights.num_key_value_heads() as usize;
                let tp = m.tp_world_size.max(1);
                (total / tp).max(1)
            }
        }
    }

    fn head_dim(&self) -> usize {
        match self {
            Self::Scratchy(m) => m.weights.head_dim() as usize,
        }
    }

    /// Per-layer `kv_heads * head_dim` for hybrid-attention-geometry arches
    /// (Gemma-4: sliding 8×256 vs global 1×512 [12b] / 2×512 [26b]). `None` =
    /// uniform pool. The arch bakes the (tp=1) per-layer elems; this path is
    /// single-GPU — tensor-parallel per-rank KV sizing is a separate change.
    fn per_layer_kv_token_elems(&self) -> Option<Vec<usize>> {
        match self {
            Self::Scratchy(m) => m.weights.per_layer_kv_token_elems(),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::Scratchy(m) => m.weights.vocab_size() as usize,
        }
    }

    /// Whether this model uses Mixture-of-Experts layers.
    /// MoE models generate too many CUDA graph nodes for monolithic capture
    /// (router + N expert GEMMs + shared experts per layer) and exceed the
    /// CUDA driver's undocumented node limit. Piecewise capture is required.
    ///
    /// Identified by arch_name string; the set mirrors the previous
    /// hand-written `Mixtral / Qwen2Moe / Qwen3Moe / DeepSeekV2` enum check.
    fn is_moe(&self) -> bool {
        match self {
            Self::Scratchy(m) => matches!(
                m.weights.arch_name(),
                "mixtral"
                    | "qwen2_moe"
                    | "qwen3_moe"
                    | "deepseek_v2"
                    | "deepseek_v3"
                    | "deepseek_v3_flat"
                    // gemma4_moe has 128 experts/layer (far over the monolithic
                    // node limit) AND its global-attn decode does a blocking D2H
                    // gather that is illegal inside graph capture — both reasons
                    // require skipping monolithic capture. Without this, the
                    // default (no --enforce-eager) launch attempts Full capture
                    // and the in-capture stream_synchronize poisons the CUDA
                    // context (hard crash). See gather_global_kv_contiguous guard.
                    | "gemma4_moe"
                    // gemma4 (DENSE 12b/31b) is NOT an MoE, but this predicate also
                    // gates two conservative behaviors it requires: (1) skip
                    // monolithic graph capture (its global head_dim-512 attention
                    // uses the same capture-incompatible D2H gather), and (2) skip
                    // the activation-profiling dummy forward — both run a bs>1 dummy
                    // batch through the global-attn path, which is bs=1-only and
                    // would trip the eval.rs bs=1 guard. Same conservative path as
                    // gemma4_moe; runs eager by default.
                    | "gemma4"
            ),
        }
    }

    fn hidden_size(&self) -> usize {
        match self {
            Self::Scratchy(m) => m.weights.hidden_size() as usize,
        }
    }

    /// GDN (Gated-DeltaNet) runtime config for hybrid arches (Qwen3.5 /
    /// Qwen3-Next). `Some` only when the compiled `ScratchyWeights` reports
    /// per-layer `gdn_runtime_config` (macro-emitted from the unrolled IR);
    /// every dense / MoE / encoder arch returns `None`. The CUDA worker
    /// uses this for the GDN state pool sizing in `initialize_cache` and
    /// the pre-init reserve in `determine_available_memory`.
    fn gdn_runtime_config(
        &self,
    ) -> Option<scratchy_forward_compiler::gdn_state_layout::GdnRuntimeConfig> {
        match self {
            Self::Scratchy(m) => m.weights.gdn_runtime_config(),
        }
    }

    /// Inject NCCL process group into all model layers for TP.
    #[cfg(feature = "nccl")]
    fn set_tp_group(&mut self, group: std::sync::Arc<crate::nccl::NcclGroup>) {
        match self {
            Self::Scratchy(m) => m.tp_group = Some(group),
        }
    }

    /// Run backbone forward pass (without lm_head), returning hidden states
    /// `[num_tokens, hidden_size]` on GPU.
    ///
    /// # Safety
    /// All GpuTensors must be valid. CUDA context must be current.
    #[allow(clippy::too_many_arguments)]
    unsafe fn hidden_states(
        &self,
        input_ids: TensorView<'_>,
        positions: TensorView<'_>,
        slot_mapping: TensorView<'_>,
        cu_seqlens_q: TensorView<'_>,
        seqused_k: TensorView<'_>,
        block_table: TensorView<'_>,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
        mm_inputs: Option<&MmForwardInputs<'_>>,
        gdn_inputs: Option<&GdnForwardInputs<'_>>,
        hybrid_kv: Option<&HybridKvInputs<'_>>,
    ) -> crate::OwnedTensor {
        match self {
            // Scratchy arches all share this shape: build `ForwardCtx`
            // from the argument bag, dispatch through the
            // `ScratchyWeights` trait's `forward_backbone` vtable. The
            // trait impl inside each arch's compiled module routes
            // to the per-arch `forward_backbone(&weights, &ctx, ...)`.
            Self::Scratchy(m) => unsafe {
                let num_tokens = input_ids.dim(0) as u64;
                let (mm_embeds, embed_patches) = match mm_inputs {
                    Some(mm) => (Some(mm.mm_embeds), mm.embed_patches),
                    None => (None, &[][..]),
                };
                // Hybrid (GDN) unpack: when the caller built a
                // `GdnForwardInputs` (Qwen3.5 / Qwen3-Next), thread its
                // `state` pool plus per-step `indices` / `is_fresh` views
                // straight into `ForwardCtx`. `None` -> non-hybrid path
                // and `gdn_state*` stay `None` (byte-identical to before).
                let (gdn_state, gdn_state_indices, gdn_is_fresh) = match gdn_inputs {
                    Some(g) => (Some(g.state), Some(g.indices), Some(g.is_fresh)),
                    None => (None, None, None),
                };
                // Hybrid SWA unpack: per-group block tables / slot mappings, or
                // empty vecs on uniform models (scalar-field fallback).
                let (block_tables, slot_mappings) = match hybrid_kv {
                    Some(h) => (h.block_tables.to_vec(), h.slot_mappings.to_vec()),
                    None => (Vec::new(), Vec::new()),
                };
                let ctx = crate::ForwardCtx {
                    input_ids,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    block_tables,
                    slot_mappings,
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    mm_embeds,
                    embed_patches,
                    vision_rope_cos: None,
                    vision_rope_sin: None,
                    vision_rope_freqs: None,
                    pixels: None,
                    pos_embeds: None,
                    vision_cu_seqlens_full: None,
                    vision_cu_seqlens_window: None,
                    vision_max_seqlen_full: None,
                    vision_max_seqlen_window: None,
                    vision_window_index: None,
                    vision_reverse_indices: None,
                    vision_position_ids: None,
                    last_token_indices: None,
                    gdn_state,
                    gdn_state_indices,
                    gdn_is_fresh,
                    #[cfg(feature = "nccl")]
                    tp_group: m.tp_group.as_ref(),
                };
                m.weights.forward_backbone(
                    crate::ForwardCtxHandle::new(&ctx),
                    crate::ForwardDeviceHandle::new(device),
                    num_tokens,
                )
            },
        }
    }

    /// Forward using caching allocator (zero D2D copies between layers).
    /// Returns an `OwnedTensor` whose drop frees the logits allocation.
    #[allow(clippy::too_many_arguments)]
    unsafe fn forward(
        &self,
        input_ids: TensorView<'_>,
        positions: TensorView<'_>,
        slot_mapping: TensorView<'_>,
        cu_seqlens_q: TensorView<'_>,
        seqused_k: TensorView<'_>,
        block_table: TensorView<'_>,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
        last_token_indices: Option<TensorView<'_>>,
        mm_inputs: Option<&MmForwardInputs<'_>>,
        gdn_inputs: Option<&GdnForwardInputs<'_>>,
        hybrid_kv: Option<&HybridKvInputs<'_>>,
    ) -> crate::OwnedTensor {
        match self {
            // Every scratchy-forward-compiler-compiled arch routes through the
            // `ScratchyWeights` trait's `forward` vtable. The per-arch
            // macro-emitted impl delegates to that arch's specialized
            // `forward(&weights, &ctx, device, num_tokens)`.
            //
            // Selective last-token gather for prefill: when
            // `last_token_indices` is set we plumb it through
            // `ForwardCtx.last_token_indices` so the lm_head op
            // (`Instruction::CutlassFusedAddRmsNormGemm`) gathers
            // [num_seqs, hidden] from the post-norm activations
            // BEFORE the GEMM. The post-forward gather then becomes
            // a no-op (idx.dim(0) == logits.dim(0)). Mirrors metal's
            // GatherLastToken lowering and Python vLLM's
            // `logits_indices = query_start_loc[1:] - 1`.
            Self::Scratchy(m) => unsafe {
                // Encoder arches (modernbert) have no lm_head — both `forward`
                // and `forward_backbone` return hidden states. Routing those
                // through the logit-gather path below would mis-shape the
                // gather. Pooling callers use `hidden_states()` instead; this
                // arm is the logit-generation path and must hard-fail for
                // encoders.
                if self.is_scratchy_encoder() {
                    panic!(
                        "{}: encoder model does not support logit generation; use hidden_states()",
                        m.weights.arch_name()
                    );
                }
                let num_tokens = input_ids.dim(0) as u64;
                let (mm_embeds, embed_patches) = match mm_inputs {
                    Some(mm) => (Some(mm.mm_embeds), mm.embed_patches),
                    None => (None, &[][..]),
                };
                // Hybrid (GDN) unpack: when the caller built a
                // `GdnForwardInputs` (Qwen3.5 / Qwen3-Next), thread its
                // `state` pool plus per-step `indices` / `is_fresh` views
                // straight into `ForwardCtx`. `None` -> non-hybrid path
                // and `gdn_state*` stay `None` (byte-identical to before).
                let (gdn_state, gdn_state_indices, gdn_is_fresh) = match gdn_inputs {
                    Some(g) => (Some(g.state), Some(g.indices), Some(g.is_fresh)),
                    None => (None, None, None),
                };
                // Hybrid SWA unpack: per-group block tables / slot mappings, or
                // empty vecs on uniform models (scalar-field fallback).
                let (block_tables, slot_mappings) = match hybrid_kv {
                    Some(h) => (h.block_tables.to_vec(), h.slot_mappings.to_vec()),
                    None => (Vec::new(), Vec::new()),
                };
                let ctx = crate::ForwardCtx {
                    input_ids,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    block_tables,
                    slot_mappings,
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    mm_embeds,
                    embed_patches,
                    vision_rope_cos: None,
                    vision_rope_sin: None,
                    vision_rope_freqs: None,
                    pixels: None,
                    pos_embeds: None,
                    vision_cu_seqlens_full: None,
                    vision_cu_seqlens_window: None,
                    vision_max_seqlen_full: None,
                    vision_max_seqlen_window: None,
                    vision_window_index: None,
                    vision_reverse_indices: None,
                    vision_position_ids: None,
                    last_token_indices: last_token_indices.as_ref().copied(),
                    gdn_state,
                    gdn_state_indices,
                    gdn_is_fresh,
                    #[cfg(feature = "nccl")]
                    tp_group: m.tp_group.as_ref(),
                };
                let logits = m.weights.forward(
                    crate::ForwardCtxHandle::new(&ctx),
                    crate::ForwardDeviceHandle::new(device),
                    num_tokens,
                );
                // If the lm_head op already gathered (logits is at
                // [num_seqs, vocab]), skip the post-forward gather.
                // Otherwise (older codepaths / edge buckets that don't
                // hit the lm_head-with-indices arm) fall back to the
                // post-forward narrow.
                match last_token_indices {
                    Some(idx) if idx.dim(0) < num_tokens as usize && logits.dim(0) > idx.dim(0) => {
                        crate::kernels::embedding_gather(
                            logits.as_gpu_tensor(),
                            *idx,
                            &mut device.caching,
                            device.compute_stream,
                        )
                    }
                    _ => logits,
                }
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Per-arch HF-config helpers (`llama_config_from_hf`, `mixtral_config_from_hf`,
// …, `deepseek_v2_config_from_hf`) used to live here. They built strongly-
// typed configs that fed the hand-written `crate::model::*::*ForCausalLM`
// loaders. Both the loaders and these helpers are gone — scratchy-forward-compiler
// parses HF config inside its emitted per-arch `Weights::load` body.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Pinned host staging buffers
// ---------------------------------------------------------------------------

/// Pre-allocated pinned (page-locked) host buffers for CUDA graph replay.
///
/// Eliminates per-step heap Vec allocations and enables true async DMA
/// (pageable memory forces the CUDA driver to stage through an internal
/// pinned buffer, serializing the transfer).
#[cfg(feature = "cuda")]
struct HostStaging {
    /// `[max_batch]` u32 — input token IDs.
    input_ids: PinnedBuf,
    /// `[max_batch]` u32 — position indices.
    positions: PinnedBuf,
    /// `[max_batch]` i64 — KV cache slot mapping.
    slot_mapping: PinnedBuf,
    /// `[max_batch + 1]` i32 — cumulative query sequence lengths.
    cu_seqlens_q: PinnedBuf,
    /// `[max_batch]` i32 — per-sequence K lengths for paged FA2.
    seqused_k: PinnedBuf,
    /// `[max_batch * max_blocks_per_seq]` i32 — page table.
    block_table: PinnedBuf,
    /// Maximum blocks per sequence (computed from max_model_len / block_size).
    max_blocks_per_seq: usize,
    /// `[max_batch]` u32 — D2H token IDs from graph argmax (double-buffered).
    ///
    /// Two pinned buffers alternate per step so that the GPU can write to one
    /// while the CPU reads from the other (deferred D2H pattern).
    host_token_ids: [PinnedBuf; 2],
    /// Which of the two `host_token_ids` buffers is current (0 or 1).
    token_buf_idx: usize,
    /// `[max_batch * 5]` f32 — packed sampling params (temps, top_ks, top_ps, min_ps, randoms).
    sampling_packed: PinnedBuf,
}

#[cfg(feature = "cuda")]
impl HostStaging {
    /// Allocate pinned staging buffers for up to `max_batch` decode requests.
    ///
    /// # Safety
    /// Requires active CUDA context.
    unsafe fn new(max_batch: usize, max_blocks_per_seq: usize) -> anyhow::Result<Self> {
        Ok(Self {
            input_ids: unsafe { PinnedBuf::new(max_batch * 4)? },
            positions: unsafe { PinnedBuf::new(max_batch * 4)? },
            slot_mapping: unsafe { PinnedBuf::new(max_batch * 8)? },
            cu_seqlens_q: unsafe { PinnedBuf::new((max_batch + 1) * 4)? },
            seqused_k: unsafe { PinnedBuf::new(max_batch * 4)? },
            block_table: unsafe { PinnedBuf::new(max_batch * max_blocks_per_seq * 4)? },
            max_blocks_per_seq,
            host_token_ids: [unsafe { PinnedBuf::new(max_batch * 4)? }, unsafe {
                PinnedBuf::new(max_batch * 4)?
            }],
            token_buf_idx: 0,
            sampling_packed: unsafe { PinnedBuf::new(max_batch * 5 * 4)? },
        })
    }

    /// Fill block_table pinned buffer from attention metadata block_ids.
    /// Returns a slice of the pinned buffer with `graph_bs * max_blocks_per_seq` elements.
    unsafe fn fill_block_table<'a>(
        &'a self,
        block_ids: &[Vec<usize>],
        graph_bs: usize,
    ) -> &'a [i32] {
        let mbps = self.max_blocks_per_seq;
        let n = graph_bs * mbps;
        let bt = unsafe { self.block_table.slice_mut::<i32>(n) };
        bt.fill(0);
        for (i, blocks) in block_ids.iter().enumerate() {
            for (j, &bid) in blocks.iter().enumerate() {
                if j < mbps {
                    bt[i * mbps + j] = bid as i32;
                }
            }
        }
        &bt[..n]
    }
}

// ---------------------------------------------------------------------------
// PiecewiseDecodeRunner — TP>1 decode runner backed by piecewise CUDA graphs.
// ---------------------------------------------------------------------------
//
// At tp>1 the monolithic graph capture path is gated off: NCCL inside a graph
// fails with `CUDA_ERROR_ILLEGAL_ADDRESS` on L40S sm_89. Without graphs the
// decode forward runs eager — but on this branch the eager path also fails at
// tp>1 (something about the per-step input buffers / FI plan freshness or
// per-rank kernel ordering — the empirical failure is `H2D u32:
// CUDA_ERROR_ILLEGAL_ADDRESS` on the very first decode step). Piecewise graphs
// fix both: each segment is a separate CUgraph, NCCL collectives run eagerly
// between graphs, captured-pool input addresses are stable across replays.
//
// Mirrors `crate::graph::CudaGraphRunner`'s shape: one set of pre-allocated
// stable input buffers shared across batch sizes, one captured `PiecewiseRunner`
// per `cuda_graph_sizes` entry. `replay()` H2D-copies the step's inputs into
// the stable buffers, then walks the captured segments calling eager NCCL
// between them.
#[cfg(all(feature = "cuda", feature = "nccl"))]
struct PiecewiseDecodeRunner {
    /// One captured piecewise runner per batch size.
    runners: std::collections::HashMap<usize, crate::piecewise::PiecewiseRunner>,
    /// Stable per-step input buffers, sized for `max_batch`. Captured graphs
    /// reference these addresses; replay H2Ds new per-step contents in.
    input_ids: crate::RawGpuMem,
    positions: crate::RawGpuMem,
    slot_mapping: crate::RawGpuMem,
    cu_seqlens_q: crate::RawGpuMem,
    seqused_k: crate::RawGpuMem,
    block_table: crate::RawGpuMem,
    max_batch: usize,
    max_blocks_per_seq: usize,
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
impl PiecewiseDecodeRunner {
    /// # Safety
    /// CUDA context must be current on this thread.
    unsafe fn new(max_batch: usize, max_blocks_per_seq: usize) -> anyhow::Result<Self> {
        unsafe {
            let input_ids = crate::raw_cuda(driver::mem_alloc(max_batch * 4)?, max_batch * 4);
            let positions = crate::raw_cuda(driver::mem_alloc(max_batch * 4)?, max_batch * 4);
            let slot_mapping = crate::raw_cuda(driver::mem_alloc(max_batch * 8)?, max_batch * 8);
            let cu_seqlens_q =
                crate::raw_cuda(driver::mem_alloc((max_batch + 1) * 4)?, (max_batch + 1) * 4);
            let seqused_k = crate::raw_cuda(driver::mem_alloc(max_batch * 4)?, max_batch * 4);
            let block_table = crate::raw_cuda(
                driver::mem_alloc(max_batch * max_blocks_per_seq * 4)?,
                max_batch * max_blocks_per_seq * 4,
            );
            Ok(Self {
                runners: std::collections::HashMap::new(),
                input_ids,
                positions,
                slot_mapping,
                cu_seqlens_q,
                seqused_k,
                block_table,
                max_batch,
                max_blocks_per_seq,
            })
        }
    }

    /// Build per-batch-size GpuTensor views over the stable buffers.
    /// The captured graph reads from these addresses every replay.
    unsafe fn ctx_views(&self, bs: usize) -> PiecewiseInputs {
        unsafe {
            PiecewiseInputs {
                input_ids: GpuTensor::new(self.input_ids.ptr(), &[bs], crate::dtype::DType::U32),
                positions: GpuTensor::new(self.positions.ptr(), &[bs], crate::dtype::DType::U32),
                slot_mapping: GpuTensor::new(
                    self.slot_mapping.ptr(),
                    &[bs],
                    crate::dtype::DType::I64,
                ),
                cu_seqlens_q: GpuTensor::new(
                    self.cu_seqlens_q.ptr(),
                    &[bs + 1],
                    crate::dtype::DType::I32,
                ),
                seqused_k: GpuTensor::new(self.seqused_k.ptr(), &[bs], crate::dtype::DType::I32),
                block_table: GpuTensor::new(
                    self.block_table.ptr(),
                    &[bs, self.max_blocks_per_seq],
                    crate::dtype::DType::I32,
                ),
            }
        }
    }

    /// Fill the input buffers with dummy decode contents at batch_size=bs.
    /// Mirrors `crate::graph::CudaGraphRunner::fill_dummy_decode`.
    unsafe fn fill_dummy(&self, bs: usize, stream: crate::CUstream) -> anyhow::Result<()> {
        unsafe {
            driver::memset_d8(self.input_ids.ptr(), 0, bs * 4, stream)?;
            let pos: Vec<u32> = (0..bs as u32).collect();
            driver::memcpy_htod_async(
                self.positions.ptr(),
                pos.as_ptr() as *const u8,
                bs * 4,
                stream,
            )?;
            let slots: Vec<i64> = (0..bs as i64).collect();
            driver::memcpy_htod_async(
                self.slot_mapping.ptr(),
                slots.as_ptr() as *const u8,
                bs * 8,
                stream,
            )?;
            let cu_q: Vec<i32> = (0..=bs as i32).collect();
            driver::memcpy_htod_async(
                self.cu_seqlens_q.ptr(),
                cu_q.as_ptr() as *const u8,
                (bs + 1) * 4,
                stream,
            )?;
            let sk: Vec<i32> = vec![1; bs];
            driver::memcpy_htod_async(
                self.seqused_k.ptr(),
                sk.as_ptr() as *const u8,
                bs * 4,
                stream,
            )?;
            driver::memset_d8(
                self.block_table.ptr(),
                0,
                bs * self.max_blocks_per_seq * 4,
                stream,
            )?;
            driver::stream_synchronize(stream)?;
        }
        Ok(())
    }

    fn captured_sizes(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.runners.keys().copied().collect();
        v.sort();
        v
    }

    fn nearest_size(&self, bs: usize) -> Option<usize> {
        self.runners.keys().filter(|&&s| s >= bs).min().copied()
    }

    /// Capture piecewise CUDA graphs for batch_size = `bs`.
    ///
    /// Caller MUST have already called
    /// `device.caching.begin_allocate_to_pool()` so the captured tile
    /// addresses come from a private pool that stays live for the
    /// runner's lifetime.
    ///
    /// Two-pass: warmup forward populates the pool with stable
    /// addresses, then `forward_piecewise_capture` records each
    /// segment's kernel launches at those addresses. Replays at the
    /// same `bs` reuse the same addresses.
    ///
    /// # Safety
    /// CUDA context current; `model` is `CudaModel::Scratchy`; `kv_cache`
    /// outlives the call.
    unsafe fn capture(
        &mut self,
        bs: usize,
        max_seqlen_k: usize,
        model: &CudaModel,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
    ) -> anyhow::Result<()> {
        assert!(bs <= self.max_batch);
        let CudaModel::Scratchy(fm) = model;

        // Fill stable buffers with dummy decode contents at this bs.
        unsafe {
            self.fill_dummy(bs, device.compute_stream)?;
        }

        // FI replan happens at replay time only (matching
        // CudaGraphRunner). Calling it during capture triggers
        // `fi_plan_new returned null rc=-1` — appears to be a
        // double-init or state-collision against the plan already
        // built during model-load profiling.
        let _ = max_seqlen_k;

        let inputs = unsafe { self.ctx_views(bs) };
        let tp_group = fm.tp_group.as_ref();

        let build_ctx = || crate::ForwardCtx {
            input_ids: unsafe { TensorView::from_raw(inputs.input_ids) },
            positions: unsafe { TensorView::from_raw(inputs.positions) },
            slot_mapping: unsafe { TensorView::from_raw(inputs.slot_mapping) },
            cu_seqlens_q: unsafe { TensorView::from_raw(inputs.cu_seqlens_q) },
            seqused_k: unsafe { TensorView::from_raw(inputs.seqused_k) },
            block_table: unsafe { TensorView::from_raw(inputs.block_table) },
            // Graph capture/replay path is uniform-KV only (gemma4 is eager);
            // empty group vecs → attention ops use the scalar block_table.
            block_tables: Vec::new(),
            slot_mappings: Vec::new(),
            max_seqlen_q: 1,
            max_seqlen_k,
            kv_cache,
            mm_embeds: None,
            embed_patches: &[],
            vision_rope_cos: None,
            vision_rope_sin: None,
            vision_rope_freqs: None,
            pixels: None,
            pos_embeds: None,
            vision_cu_seqlens_full: None,
            vision_cu_seqlens_window: None,
            vision_max_seqlen_full: None,
            vision_max_seqlen_window: None,
            vision_window_index: None,
            vision_reverse_indices: None,
            vision_position_ids: None,
            last_token_indices: None,
            gdn_state: None,
            gdn_state_indices: None,
            gdn_is_fresh: None,
            tp_group,
        };

        // Warmup forward with NCCL suppressed — populates the private
        // allocator pool AND triggers FI's first-call `fi_plan_new`
        // (which does cudaMalloc) BEFORE we begin graph capture.
        // cudaMalloc inside `cuStreamBeginCapture` would fail. NCCL is
        // suppressed via the same guard `forward_piecewise_capture`
        // uses internally, so collectives don't fire here either —
        // ranks don't need to coordinate during this warmup.
        {
            let _suppress = crate::nccl::SuppressNcclGuard::new();
            let warmup = {
                let ctx = build_ctx();
                unsafe {
                    fm.weights.forward(
                        crate::ForwardCtxHandle::new(&ctx),
                        crate::ForwardDeviceHandle::new(device),
                        bs as u64,
                    )
                }
            };
            unsafe {
                driver::stream_synchronize(device.compute_stream)?;
            }
            drop(warmup);
        }

        // Capture: forward_piecewise_capture splits the tape at NCCL
        // boundaries, each segment becomes a CUgraph. The trait returns a
        // neutral `Box<dyn Any + Send>` (it can't name the cuda-only runner);
        // downcast it back to the concrete `PiecewiseRunner`.
        let runner = {
            let ctx = build_ctx();
            let boxed = unsafe {
                fm.weights.forward_piecewise_capture(
                    crate::ForwardCtxHandle::new(&ctx),
                    crate::ForwardDeviceHandle::new(device),
                    bs as u64,
                )?
            };
            *boxed
                .downcast::<crate::piecewise::PiecewiseRunner>()
                .expect("forward_piecewise_capture returned a non-PiecewiseRunner box")
        };
        self.runners.insert(bs, runner);
        Ok(())
    }

    /// Replay piecewise graphs for batch_size = `bs` with this step's
    /// host-side inputs.
    ///
    /// Returns the forward's logits as an `OwnedTensor` allocated from
    /// the regular caching allocator (not the private pool — independent
    /// lifetime, caller frees on drop).
    ///
    /// # Safety
    /// Buffers in `&[…]` slices have at least `bs` elements (or `bs+1`
    /// for cu_seqlens_q); `device` is the runner's capture device;
    /// caller has the NCCL group attached on `model.tp_group`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn replay(
        &self,
        bs: usize,
        input_ids: &[u32],
        positions: &[u32],
        slot_mapping: &[i64],
        cu_seqlens_q: &[i32],
        seqused_k: &[i32],
        block_table: &[i32],
        kv_cache: &KvCachePool,
        model: &CudaModel,
        device: &mut GpuDevice,
    ) -> anyhow::Result<OwnedTensor> {
        let runner = self.runners.get(&bs).ok_or_else(|| {
            anyhow::anyhow!("PiecewiseDecodeRunner: no captured runner for bs={bs}")
        })?;
        let CudaModel::Scratchy(fm) = model;
        let xfer = device.transfer_stream;

        // H2D the step's inputs into the stable buffers on the transfer
        // stream, then sync xfer→compute so the captured graph reads
        // the new contents.
        unsafe {
            driver::memcpy_htod_async(
                self.input_ids.ptr(),
                input_ids.as_ptr() as *const u8,
                bs * 4,
                xfer,
            )?;
            driver::memcpy_htod_async(
                self.positions.ptr(),
                positions.as_ptr() as *const u8,
                bs * 4,
                xfer,
            )?;
            driver::memcpy_htod_async(
                self.slot_mapping.ptr(),
                slot_mapping.as_ptr() as *const u8,
                bs * 8,
                xfer,
            )?;
            driver::memcpy_htod_async(
                self.cu_seqlens_q.ptr(),
                cu_seqlens_q.as_ptr() as *const u8,
                (bs + 1) * 4,
                xfer,
            )?;
            driver::memcpy_htod_async(
                self.seqused_k.ptr(),
                seqused_k.as_ptr() as *const u8,
                bs * 4,
                xfer,
            )?;
            driver::memcpy_htod_async(
                self.block_table.ptr(),
                block_table.as_ptr() as *const u8,
                bs * self.max_blocks_per_seq * 4,
                xfer,
            )?;
            device.sync_transfer_to_compute()?;
        }

        // FI replan with this step's max KV span. The captured graph's
        // FI kernels read int_ws_d, which replan rewrites on the
        // compute stream — same pattern `CudaGraphRunner::replay` uses.
        let max_seqlen_k = seqused_k.iter().copied().max().unwrap_or(0) as usize;
        unsafe {
            crate::attention_helpers::replan_fi_for_decode(
                bs,
                max_seqlen_k,
                kv_cache.block_size,
                device.compute_stream,
            );
        }

        // Build the same ctx shape capture saw — captured kernels read
        // the GpuTensor pointers, but the eager NCCL between segments
        // reads `tp_group` from this fresh ForwardCtx.
        let inputs = unsafe { self.ctx_views(bs) };
        let tp_group = fm.tp_group.as_ref();
        let ctx = crate::ForwardCtx {
            input_ids: unsafe { TensorView::from_raw(inputs.input_ids) },
            positions: unsafe { TensorView::from_raw(inputs.positions) },
            slot_mapping: unsafe { TensorView::from_raw(inputs.slot_mapping) },
            cu_seqlens_q: unsafe { TensorView::from_raw(inputs.cu_seqlens_q) },
            seqused_k: unsafe { TensorView::from_raw(inputs.seqused_k) },
            block_table: unsafe { TensorView::from_raw(inputs.block_table) },
            // Graph capture/replay path is uniform-KV only (gemma4 is eager);
            // empty group vecs → attention ops use the scalar block_table.
            block_tables: Vec::new(),
            slot_mappings: Vec::new(),
            max_seqlen_q: 1,
            max_seqlen_k,
            kv_cache,
            mm_embeds: None,
            embed_patches: &[],
            vision_rope_cos: None,
            vision_rope_sin: None,
            vision_rope_freqs: None,
            pixels: None,
            pos_embeds: None,
            vision_cu_seqlens_full: None,
            vision_cu_seqlens_window: None,
            vision_max_seqlen_full: None,
            vision_max_seqlen_window: None,
            vision_window_index: None,
            vision_reverse_indices: None,
            vision_position_ids: None,
            last_token_indices: None,
            gdn_state: None,
            gdn_state_indices: None,
            gdn_is_fresh: None,
            tp_group,
        };

        let logits = unsafe { crate::piecewise::run_piecewise_replay(runner, &ctx, device) };
        Ok(logits)
    }
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
struct PiecewiseInputs {
    input_ids: GpuTensor,
    positions: GpuTensor,
    slot_mapping: GpuTensor,
    cu_seqlens_q: GpuTensor,
    seqused_k: GpuTensor,
    block_table: GpuTensor,
}

// ---------------------------------------------------------------------------
// PiecewisePrefillRunner — TP>1 prefill runner backed by piecewise CUDA graphs.
// ---------------------------------------------------------------------------
//
// Mirror of `crate::graph::PrefillGraphRunner` but emits one captured
// `PiecewiseRunner` per `cuda_graph_sizes` prefill bucket. The monolithic
// PrefillGraphRunner deadlocks at tp>1 because its captured forward calls
// NCCL inside a single CUgraph; piecewise splits the tape at NCCL boundaries
// so collectives run eagerly between segments. Mirrors Python vLLM's
// FULL_AND_PIECEWISE mode (full graphs for decode, piecewise for prefill).
//
// Single-sequence shape: `cu_seqlens_q = [0, num_tokens]`, `seqused_k = [seq_len]`,
// `block_table = [1, max_blocks_per_seq]`, `last_token_indices = [num_tokens-1]`
// for selective lm_head gather (matching the monolithic prefill graph).
#[cfg(all(feature = "cuda", feature = "nccl"))]
struct PiecewisePrefillRunner {
    /// One captured piecewise runner per num_tokens bucket.
    runners: std::collections::HashMap<usize, crate::piecewise::PiecewiseRunner>,
    /// Stable per-step input buffers, sized for `max_prefill_tokens`.
    input_ids: crate::RawGpuMem,
    positions: crate::RawGpuMem,
    slot_mapping: crate::RawGpuMem,
    /// `[2]` i32 — `[0, num_tokens]`.
    cu_seqlens_q: crate::RawGpuMem,
    /// `[1]` i32 — `[seq_len]`.
    seqused_k: crate::RawGpuMem,
    /// `[1, max_blocks_per_seq]` i32.
    block_table: crate::RawGpuMem,
    /// `[1]` u32 — `[num_tokens - 1]` for selective lm_head gather.
    last_token_indices: crate::RawGpuMem,
    max_prefill_tokens: usize,
    max_blocks_per_seq: usize,
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
impl PiecewisePrefillRunner {
    /// # Safety
    /// CUDA context must be current on this thread.
    unsafe fn new(max_prefill_tokens: usize, max_blocks_per_seq: usize) -> anyhow::Result<Self> {
        unsafe {
            let input_ids = crate::raw_cuda(
                driver::mem_alloc(max_prefill_tokens * 4)?,
                max_prefill_tokens * 4,
            );
            let positions = crate::raw_cuda(
                driver::mem_alloc(max_prefill_tokens * 4)?,
                max_prefill_tokens * 4,
            );
            let slot_mapping = crate::raw_cuda(
                driver::mem_alloc(max_prefill_tokens * 8)?,
                max_prefill_tokens * 8,
            );
            // single-seq: cu_seqlens_q = [0, num_tokens], seqused_k = [seq_len].
            let cu_seqlens_q = crate::raw_cuda(driver::mem_alloc(2 * 4)?, 2 * 4);
            let seqused_k = crate::raw_cuda(driver::mem_alloc(4)?, 4);
            let block_table = crate::raw_cuda(
                driver::mem_alloc(max_blocks_per_seq * 4)?,
                max_blocks_per_seq * 4,
            );
            let last_token_indices = crate::raw_cuda(driver::mem_alloc(4)?, 4);
            Ok(Self {
                runners: std::collections::HashMap::new(),
                input_ids,
                positions,
                slot_mapping,
                cu_seqlens_q,
                seqused_k,
                block_table,
                last_token_indices,
                max_prefill_tokens,
                max_blocks_per_seq,
            })
        }
    }

    /// Build GpuTensor views over the stable buffers at this num_tokens.
    unsafe fn ctx_views(&self, num_tokens: usize) -> PiecewisePrefillInputs {
        unsafe {
            PiecewisePrefillInputs {
                input_ids: GpuTensor::new(
                    self.input_ids.ptr(),
                    &[num_tokens],
                    crate::dtype::DType::U32,
                ),
                positions: GpuTensor::new(
                    self.positions.ptr(),
                    &[num_tokens],
                    crate::dtype::DType::U32,
                ),
                slot_mapping: GpuTensor::new(
                    self.slot_mapping.ptr(),
                    &[num_tokens],
                    crate::dtype::DType::I64,
                ),
                cu_seqlens_q: GpuTensor::new(
                    self.cu_seqlens_q.ptr(),
                    &[2],
                    crate::dtype::DType::I32,
                ),
                seqused_k: GpuTensor::new(self.seqused_k.ptr(), &[1], crate::dtype::DType::I32),
                block_table: GpuTensor::new(
                    self.block_table.ptr(),
                    &[1, self.max_blocks_per_seq],
                    crate::dtype::DType::I32,
                ),
                last_token_indices: GpuTensor::new(
                    self.last_token_indices.ptr(),
                    &[1],
                    crate::dtype::DType::U32,
                ),
            }
        }
    }

    /// Fill stable buffers with dummy fresh-prefill contents at this
    /// num_tokens. Mirrors `crate::graph::PrefillGraphRunner::fill_dummy_prefill`.
    unsafe fn fill_dummy(&self, num_tokens: usize, stream: crate::CUstream) -> anyhow::Result<()> {
        unsafe {
            driver::memset_d8(self.input_ids.ptr(), 0, num_tokens * 4, stream)?;
            let pos: Vec<u32> = (0..num_tokens as u32).collect();
            driver::memcpy_htod_async(
                self.positions.ptr(),
                pos.as_ptr() as *const u8,
                num_tokens * 4,
                stream,
            )?;
            let slots: Vec<i64> = (0..num_tokens as i64).collect();
            driver::memcpy_htod_async(
                self.slot_mapping.ptr(),
                slots.as_ptr() as *const u8,
                num_tokens * 8,
                stream,
            )?;
            let cu_q: [i32; 2] = [0, num_tokens as i32];
            driver::memcpy_htod_async(
                self.cu_seqlens_q.ptr(),
                cu_q.as_ptr() as *const u8,
                2 * 4,
                stream,
            )?;
            let sk: [i32; 1] = [num_tokens as i32];
            driver::memcpy_htod_async(self.seqused_k.ptr(), sk.as_ptr() as *const u8, 4, stream)?;
            driver::memset_d8(
                self.block_table.ptr(),
                0,
                self.max_blocks_per_seq * 4,
                stream,
            )?;
            let lti: [u32; 1] = [(num_tokens - 1) as u32];
            driver::memcpy_htod_async(
                self.last_token_indices.ptr(),
                lti.as_ptr() as *const u8,
                4,
                stream,
            )?;
            driver::stream_synchronize(stream)?;
        }
        Ok(())
    }

    fn captured_sizes(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.runners.keys().copied().collect();
        v.sort();
        v
    }

    fn nearest_size(&self, num_tokens: usize) -> Option<usize> {
        self.runners
            .keys()
            .filter(|&&s| s >= num_tokens)
            .min()
            .copied()
    }

    /// Capture piecewise CUDA graphs for `num_tokens` prefill.
    ///
    /// Caller MUST have already called `device.caching.begin_allocate_to_pool()`
    /// so captured tile addresses come from a private pool that stays live for
    /// the runner's lifetime.
    ///
    /// # Safety
    /// CUDA context current; `model` is `CudaModel::Scratchy`; `kv_cache` outlives
    /// the call.
    unsafe fn capture(
        &mut self,
        num_tokens: usize,
        model: &CudaModel,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
    ) -> anyhow::Result<()> {
        assert!(num_tokens <= self.max_prefill_tokens);
        let CudaModel::Scratchy(fm) = model;

        unsafe {
            self.fill_dummy(num_tokens, device.compute_stream)?;
        }

        let inputs = unsafe { self.ctx_views(num_tokens) };
        let tp_group = fm.tp_group.as_ref();

        let build_ctx = || crate::ForwardCtx {
            input_ids: unsafe { TensorView::from_raw(inputs.input_ids) },
            positions: unsafe { TensorView::from_raw(inputs.positions) },
            slot_mapping: unsafe { TensorView::from_raw(inputs.slot_mapping) },
            cu_seqlens_q: unsafe { TensorView::from_raw(inputs.cu_seqlens_q) },
            seqused_k: unsafe { TensorView::from_raw(inputs.seqused_k) },
            block_table: unsafe { TensorView::from_raw(inputs.block_table) },
            // Graph capture/replay path is uniform-KV only (gemma4 is eager);
            // empty group vecs → attention ops use the scalar block_table.
            block_tables: Vec::new(),
            slot_mappings: Vec::new(),
            max_seqlen_q: num_tokens,
            max_seqlen_k: num_tokens,
            kv_cache,
            mm_embeds: None,
            embed_patches: &[],
            vision_rope_cos: None,
            vision_rope_sin: None,
            vision_rope_freqs: None,
            pixels: None,
            pos_embeds: None,
            vision_cu_seqlens_full: None,
            vision_cu_seqlens_window: None,
            vision_max_seqlen_full: None,
            vision_max_seqlen_window: None,
            vision_window_index: None,
            vision_reverse_indices: None,
            vision_position_ids: None,
            last_token_indices: Some(unsafe { TensorView::from_raw(inputs.last_token_indices) }),
            gdn_state: None,
            gdn_state_indices: None,
            gdn_is_fresh: None,
            tp_group,
        };

        // Warmup forward with NCCL suppressed — populates the private
        // allocator pool AND triggers FI's first-call `fi_plan_new`
        // (which does cudaMalloc) BEFORE we begin graph capture.
        // cudaMalloc inside `cuStreamBeginCapture` would fail. NCCL is
        // suppressed via the same guard `forward_piecewise_capture`
        // uses internally, so collectives don't fire here either —
        // ranks don't need to coordinate during this warmup.
        {
            let _suppress = crate::nccl::SuppressNcclGuard::new();
            let warmup = {
                let ctx = build_ctx();
                unsafe {
                    fm.weights.forward(
                        crate::ForwardCtxHandle::new(&ctx),
                        crate::ForwardDeviceHandle::new(device),
                        num_tokens as u64,
                    )
                }
            };
            unsafe {
                driver::stream_synchronize(device.compute_stream)?;
            }
            drop(warmup);
        }

        let runner = {
            let ctx = build_ctx();
            let boxed = unsafe {
                fm.weights.forward_piecewise_capture(
                    crate::ForwardCtxHandle::new(&ctx),
                    crate::ForwardDeviceHandle::new(device),
                    num_tokens as u64,
                )?
            };
            *boxed
                .downcast::<crate::piecewise::PiecewiseRunner>()
                .expect("forward_piecewise_capture returned a non-PiecewiseRunner box")
        };
        self.runners.insert(num_tokens, runner);
        Ok(())
    }

    /// Replay piecewise prefill graphs at `padded_num_tokens` with this step's
    /// host-side inputs.
    ///
    /// # Safety
    /// `device` matches the runner's capture device; caller has the NCCL group
    /// attached on `model.tp_group`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn replay(
        &self,
        padded_num_tokens: usize,
        input_ids: &[u32],
        positions: &[u32],
        slot_mapping: &[i64],
        seq_len: usize,
        block_table: &[i32],
        last_token_idx: u32,
        block_size: usize,
        kv_cache: &KvCachePool,
        model: &CudaModel,
        device: &mut GpuDevice,
    ) -> anyhow::Result<OwnedTensor> {
        let runner = self.runners.get(&padded_num_tokens).ok_or_else(|| {
            anyhow::anyhow!(
                "PiecewisePrefillRunner: no captured runner for num_tokens={padded_num_tokens}"
            )
        })?;
        let CudaModel::Scratchy(fm) = model;
        let xfer = device.transfer_stream;
        let num_real = input_ids.len();

        // H2D real inputs into the stable buffers, padding the rest with
        // zeros (input_ids/positions) or -1 (slot_mapping).
        unsafe {
            if num_real < padded_num_tokens {
                driver::memset_d8(self.input_ids.ptr(), 0, padded_num_tokens * 4, xfer)?;
            }
            driver::memcpy_htod_async(
                self.input_ids.ptr(),
                input_ids.as_ptr() as *const u8,
                num_real * 4,
                xfer,
            )?;

            if num_real < padded_num_tokens {
                driver::memset_d8(self.positions.ptr(), 0, padded_num_tokens * 4, xfer)?;
            }
            driver::memcpy_htod_async(
                self.positions.ptr(),
                positions.as_ptr() as *const u8,
                num_real * 4,
                xfer,
            )?;

            let mut padded_slots = vec![-1i64; padded_num_tokens];
            padded_slots[..num_real].copy_from_slice(slot_mapping);
            driver::memcpy_htod_async(
                self.slot_mapping.ptr(),
                padded_slots.as_ptr() as *const u8,
                padded_num_tokens * 8,
                xfer,
            )?;

            let cu_q: [i32; 2] = [0, padded_num_tokens as i32];
            let sk_val: [i32; 1] = [seq_len as i32];
            driver::memcpy_htod_async(
                self.cu_seqlens_q.ptr(),
                cu_q.as_ptr() as *const u8,
                8,
                xfer,
            )?;
            driver::memcpy_htod_async(self.seqused_k.ptr(), sk_val.as_ptr() as *const u8, 4, xfer)?;

            driver::memcpy_htod_async(
                self.block_table.ptr(),
                block_table.as_ptr() as *const u8,
                block_table.len().min(self.max_blocks_per_seq) * 4,
                xfer,
            )?;

            driver::memcpy_htod_async(
                self.last_token_indices.ptr(),
                &last_token_idx as *const u32 as *const u8,
                4,
                xfer,
            )?;

            device.sync_transfer_to_compute()?;
        }

        // FI replan with this step's max KV span. Fresh prefill: max_seqlen_k == seq_len.
        if block_size > 0 {
            unsafe {
                crate::attention_helpers::replan_fi_for_decode(
                    seq_len,
                    seq_len,
                    block_size,
                    device.compute_stream,
                );
            }
        }

        let inputs = unsafe { self.ctx_views(padded_num_tokens) };
        let tp_group = fm.tp_group.as_ref();
        let ctx = crate::ForwardCtx {
            input_ids: unsafe { TensorView::from_raw(inputs.input_ids) },
            positions: unsafe { TensorView::from_raw(inputs.positions) },
            slot_mapping: unsafe { TensorView::from_raw(inputs.slot_mapping) },
            cu_seqlens_q: unsafe { TensorView::from_raw(inputs.cu_seqlens_q) },
            seqused_k: unsafe { TensorView::from_raw(inputs.seqused_k) },
            block_table: unsafe { TensorView::from_raw(inputs.block_table) },
            // Graph capture/replay path is uniform-KV only (gemma4 is eager);
            // empty group vecs → attention ops use the scalar block_table.
            block_tables: Vec::new(),
            slot_mappings: Vec::new(),
            max_seqlen_q: padded_num_tokens,
            max_seqlen_k: seq_len,
            kv_cache,
            mm_embeds: None,
            embed_patches: &[],
            vision_rope_cos: None,
            vision_rope_sin: None,
            vision_rope_freqs: None,
            pixels: None,
            pos_embeds: None,
            vision_cu_seqlens_full: None,
            vision_cu_seqlens_window: None,
            vision_max_seqlen_full: None,
            vision_max_seqlen_window: None,
            vision_window_index: None,
            vision_reverse_indices: None,
            vision_position_ids: None,
            last_token_indices: Some(unsafe { TensorView::from_raw(inputs.last_token_indices) }),
            gdn_state: None,
            gdn_state_indices: None,
            gdn_is_fresh: None,
            tp_group,
        };

        let logits = unsafe { crate::piecewise::run_piecewise_replay(runner, &ctx, device) };
        Ok(logits)
    }
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
struct PiecewisePrefillInputs {
    input_ids: GpuTensor,
    positions: GpuTensor,
    slot_mapping: GpuTensor,
    cu_seqlens_q: GpuTensor,
    seqused_k: GpuTensor,
    block_table: GpuTensor,
    last_token_indices: GpuTensor,
}

// ---------------------------------------------------------------------------
// CudaWorker / MetalWorker
// ---------------------------------------------------------------------------

/// Deferred commit from a previous decode step.
///
/// Stored when the greedy graph fast path defers D2H sync. The token IDs
/// live in one of the double-buffered pinned host staging buffers. The
/// commit is resolved at the start of the next `execute_model_inner` call.
#[cfg(feature = "cuda")]
struct PendingCommit {
    /// Which host_token_ids buffer index holds the deferred token IDs.
    buf_idx: usize,
    /// Number of real requests (not graph padding).
    num_reqs: usize,
    /// Per-request IDs (same order as the host buffer).
    req_ids: Vec<String>,
    /// Per-request token count before this step (for `commit_step`).
    token_counts: Vec<usize>,
    /// Per-request flag: true if request had speculative tokens.
    has_spec_tokens: Vec<bool>,
}

/// A worker backed by the GPU runtime (cuda) for
/// zero-allocation inference.
#[cfg(feature = "cuda")]
pub struct CudaWorker {
    // ---------------------------------------------------------------
    // Backend-neutral state (cfg-mutexed only by transitive types).
    // ---------------------------------------------------------------
    config: WorkerCreateConfig,
    kv_cache: Option<KvCachePool>,
    /// Gated-DeltaNet recurrent-state pool for hybrid arches (Qwen3.5 /
    /// Qwen3-Next). `Some(_)` only when the loaded model's
    /// `gdn_runtime_config()` is `Some` (built in `initialize_cache`).
    /// The non-paged sibling of `kv_cache`: the metal forward binds its
    /// per-linear-layer conv/ssm buffers into the runtime through
    /// `ForwardCtx::gdn_state`.
    gdn_state: Option<GdnStatePool<crate::PoolMem>>,
    /// Per-request GDN state-slot allocator (one slot per concurrently
    /// resident sequence, recycled on finish — the "degeneration after
    /// N requests" guard). `Some` iff the model is hybrid; drives the
    /// per-step `gdn_state_indices` / `gdn_is_fresh` the forward reads.
    gdn_slot_allocator: Option<GdnSlotAllocator>,
    /// Per-step GDN scratch: the `(state_indices, is_fresh)` vectors
    /// built in `execute_model` (one entry per batched sequence in
    /// cu_seqlens order) and consumed by the metal
    /// `forward_argmax_blocking` to populate `ForwardCtx::{gdn_state_indices,
    /// gdn_is_fresh}`. `None` between steps / for non-hybrid arches.
    gdn_pending: Option<(Vec<i32>, Vec<u32>)>,
    model_dir: Option<PathBuf>,
    hf_config: Option<HfModelConfig>,
    /// Draft model's resolved snapshot dir, populated when
    /// `config.draft_model_path` is set.
    #[allow(dead_code)]
    draft_model_dir: Option<PathBuf>,
    /// Draft model's `config.json`, populated alongside `draft_model_dir`.
    #[allow(dead_code)]
    draft_hf_config: Option<HfModelConfig>,
    /// Second KV cache pool, sized for the draft model. Lives on the
    /// same `MetalAllocator` / residency set as the target's pool —
    /// one allocator covers all weights + both pools.
    #[allow(dead_code)]
    draft_kv_cache: Option<KvCachePool>,
    model_dtype: GpuDType,
    resolved_architecture: Option<String>,
    is_shutdown: bool,
    token_buffers: HashMap<String, Vec<u32>>,
    /// Per-request prompt length (for discard_request_mask on intermediate prefill chunks).
    prompt_lengths: HashMap<String, usize>,
    /// Per-request block annotations for span-aware RoPE.
    annotation_buffers: HashMap<String, scratchy_core_common::BlockAnnotations>,
    /// Per-request multimodal data (images), only populated for requests
    /// scheduled at first as image-bearing.
    mm_data_buffers: HashMap<String, scratchy_core_common::MultimodalData>,
    sampling_params_map: HashMap<String, SamplingParams>,
    input_batch: InputBatch,
    preloaded_tokenizer: Option<tokenizers::Tokenizer>,
    /// Resolved pooling strategy for embedding mode (cuda-only:
    /// pooling-mode codepaths live in the `cuda` impl block).
    pooling_strategy: scratchy_core_model::embedding::PoolingStrategy,
    /// Whether executing in pooling mode (--runner pooling).
    is_pooling: bool,
    /// True if batch composition changed this step (triggers BatchUpdate).
    batch_changed: bool,
    /// Ordered request IDs in the current batch (for pipeline update_state).
    batch_req_ids: Vec<String>,
    /// Per-request seeded RNGs for deterministic sampling.
    seeded_rngs: HashMap<String, rand::rngs::StdRng>,
    /// Optional progress callback for startup initialization.
    #[allow(clippy::type_complexity)]
    progress_callback: Option<std::sync::Arc<dyn Fn(&str) + Send + Sync>>,

    // ---------------------------------------------------------------
    // CUDA-only orchestration state.
    // ---------------------------------------------------------------
    model: Option<CudaModel>,
    /// CUDA graph runner for decode batches (monolithic mode).
    graph_runner: Option<CudaGraphRunner>,
    /// Piecewise CUDA graph runner for decode batches at tp>1. Captured
    /// once per `cuda_graph_sizes` entry in `compile_or_warm_up_model`;
    /// `execute_model_inner` routes pure-decode batches through
    /// `replay()` when batch matches a captured size, falling back to
    /// eager forward otherwise. `None` at tp=1 (monolithic graphs cover
    /// the same workload there).
    #[cfg(feature = "nccl")]
    piecewise_decode: Option<PiecewiseDecodeRunner>,
    /// Piecewise CUDA graph runner for single-sequence prefill at tp>1.
    /// Mirrors `prefill_graph_runner`'s shape but each tape segment is a
    /// separate CUgraph with eager NCCL between, so the cooperative collective
    /// inside-graph deadlock that monolithic prefill capture hits at tp>1
    /// doesn't apply. `None` at tp=1 (`prefill_graph_runner` covers the same
    /// workload there). Matches Python vLLM's FULL_AND_PIECEWISE mode.
    #[cfg(feature = "nccl")]
    piecewise_prefill: Option<PiecewisePrefillRunner>,
    /// CUDA graph runner for single-sequence prefill batches.
    prefill_graph_runner: Option<PrefillGraphRunner>,
    last_graph_batch_size: Option<(usize, u32)>,
    graph_metadata_valid: bool,
    /// True when the model uses GGML quantized layers (disables CUDA graphs).
    uses_ggml: bool,
    /// Pre-allocated pinned host staging buffers for graph replay.
    host_staging: Option<HostStaging>,
    /// Set once per thread to avoid redundant `ctx_set_current` driver calls.
    ctx_set_on_thread: bool,
    /// Deferred D2H commit from the previous greedy graph step.
    pending_commit: Option<PendingCommit>,
    /// Per-request grammar guide state for constrained decoding.
    #[cfg(feature = "guided-decoding")]
    grammar_states: HashMap<String, scratchy_core_model::grammar::GrammarGuide>,
    /// Parser factory for grammar-guided decoding (built once from tokenizer).
    #[cfg(feature = "guided-decoding")]
    grammar_factory: Option<std::sync::Arc<scratchy_core_model::grammar::LlgParserFactory>>,
    /// LogitsProcessor pipeline: persistent GPU state, rebuilt only on batch changes.
    logits_pipeline: Option<LogitsProcessorPipeline>,
    /// Grammar mask processor (separate from pipeline — needs backup logits).
    grammar_processor: GrammarMaskProcessor,
    /// Allowed token IDs processor.
    allowed_token_ids_processor: AllowedTokenIdsProcessor,
    /// Seal-pad processor: forces pad tokens after EOS for sealed requests.
    seal_pad_processor: SealPadProcessor,
    /// True when KV cache uses FP8 E4M3 quantization.
    kv_cache_is_fp8: bool,
    _calculate_kv_scales: bool,
    _k_scale_constant: f32,
    _v_scale_constant: f32,
    /// GPU weight allocations tracked for sleep/wake lifecycle.
    weight_gpu_allocs: Vec<crate::RawGpuMem>,
    /// Saved num_gpu_blocks for re-init after wake.
    num_gpu_blocks_saved: usize,
    /// The `CachingAllocator` (inside `GpuDevice`) is declared LAST among the
    /// CUDA fields so it drops AFTER every `OwnedTensor` holder above — `model`,
    /// the graph runners, `logits_pipeline` (which owns `PenaltiesProcessor` and
    /// its penalty tensors), the KV cache, etc. Rust drops fields in declaration
    /// order; if the allocator drops first, those fields' `OwnedTensor::drop`
    /// calls `free()` into an already-freed allocator — an ASan-confirmed
    /// heap-use-after-free that surfaces as the teardown `BTreeSet` "empty
    /// internal node". Keep `device` last, and do not drop it early in
    /// `shutdown()`.
    device: Option<GpuDevice>,
}
#[cfg(feature = "cuda")]
impl CudaWorker {
    /// Build `[3, n_tokens]` u32 MRoPE positions for an image-bearing
    /// batch. Mirrors Python vLLM's
    /// `Qwen2VLForConditionalGeneration.get_input_positions_tensor`:
    /// text tokens get `(cursor, cursor, cursor)` with `cursor`
    /// incrementing by 1; image tokens scan the post-spatial-merge
    /// grid in `(t, h, w)` row-major and emit `(st+t, st+h, st+w)`
    /// where `st` is the cursor at image entry; cursor advances by
    /// `max(grid_t, grid_h_merged, grid_w_merged)` after each image.
    ///
    /// Walks **every** seq position from 0 to `tokens_before + q_len`,
    /// advancing the cursor through preceding image patches even when
    /// they lie entirely in the cached prefix. Positions are written
    /// only for tokens in the current q range — but the cursor that
    /// gets written is the same one the encoder used to encode the
    /// cached KV, so `cached==prompt-1` (Bug 3) decodes correctly.
    /// Reqs with no patches get the linear cursor for every row —
    /// numerically identical to text-only 1D rope through the kernel's
    /// broadcast path, but in the 2D shape so a mixed batch (text req +
    /// image req) can share one positions tensor.
    fn build_mrope_positions_2d(
        prepared: &PreparedInputs,
        per_req_mm: &[Vec<SeqMmInfo>],
    ) -> Vec<u32> {
        let n_tokens = prepared.flat_positions.len();
        let mut t_row = vec![0u32; n_tokens];
        let mut h_row = vec![0u32; n_tokens];
        let mut w_row = vec![0u32; n_tokens];
        let meta = &prepared.attn_meta;

        for (req_idx, req_patches) in per_req_mm.iter().enumerate() {
            let req_start_batch = meta.query_start_loc[req_idx];
            let q_len = meta.q_lens[req_idx] as u32;
            let num_computed = meta.tokens_before[req_idx] as u32;
            let q_seq_end = num_computed + q_len;

            let mut cursor = 0u32;
            let mut seq_pos = 0u32;
            let mut patch_iter = req_patches.iter().peekable();

            while seq_pos < q_seq_end {
                if let Some(&&(s_off, length, gt, mh, mw)) = patch_iter.peek()
                    && s_off == seq_pos
                {
                    let st = cursor;
                    let stride_hw = mh * mw;
                    for img_idx in 0..length {
                        let pos_seq = seq_pos + img_idx;
                        if pos_seq >= num_computed && pos_seq < q_seq_end {
                            let local_q = (pos_seq - num_computed) as usize;
                            let flat_idx = req_start_batch + local_q;
                            let t = img_idx / stride_hw;
                            let rem = img_idx % stride_hw;
                            let h = rem / mw;
                            let w = rem % mw;
                            t_row[flat_idx] = st + t;
                            h_row[flat_idx] = st + h;
                            w_row[flat_idx] = st + w;
                        }
                    }
                    cursor = st + gt.max(mh).max(mw);
                    seq_pos += length;
                    patch_iter.next();
                    continue;
                }
                if seq_pos >= num_computed {
                    let local_q = (seq_pos - num_computed) as usize;
                    let flat_idx = req_start_batch + local_q;
                    t_row[flat_idx] = cursor;
                    h_row[flat_idx] = cursor;
                    w_row[flat_idx] = cursor;
                }
                cursor += 1;
                seq_pos += 1;
            }
        }

        let mut out = Vec::with_capacity(3 * n_tokens);
        out.extend_from_slice(&t_row);
        out.extend_from_slice(&h_row);
        out.extend_from_slice(&w_row);
        out
    }
}
#[cfg(feature = "cuda")]
impl CudaWorker {
    pub fn new(config: WorkerCreateConfig) -> Self {
        let is_pooling = config.is_pooling;
        let kv_cache_is_fp8 = config.kv_cache_dtype == "fp8_e4m3" || config.kv_cache_dtype == "fp8";
        let calculate_kv_scales = config.calculate_kv_scales;
        // Scale constants: match Python's defaults. Override via env vars.
        let k_scale_constant = std::env::var("VLLM_FP8_K_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0_f32);
        let v_scale_constant = std::env::var("VLLM_FP8_V_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0_f32);
        let seal_pad_processor = SealPadProcessor::new(
            config.eos_token_ids.clone(),
            0, // pad_token: default 0, updated during model load from tokenizer
            config.block_size,
        );
        Self {
            config,
            device: None,
            model: None,
            kv_cache: None,
            gdn_state: None,
            gdn_slot_allocator: None,
            gdn_pending: None,
            model_dir: None,
            hf_config: None,
            draft_model_dir: None,
            draft_hf_config: None,
            draft_kv_cache: None,
            model_dtype: GpuDType::BF16,
            resolved_architecture: None,
            is_shutdown: false,
            graph_runner: None,
            #[cfg(feature = "nccl")]
            piecewise_decode: None,
            #[cfg(feature = "nccl")]
            piecewise_prefill: None,
            prefill_graph_runner: None,
            last_graph_batch_size: None,
            graph_metadata_valid: false,
            uses_ggml: false,
            host_staging: None,
            token_buffers: HashMap::new(),
            prompt_lengths: HashMap::new(),
            annotation_buffers: HashMap::new(),
            mm_data_buffers: HashMap::new(),
            sampling_params_map: HashMap::new(),
            input_batch: InputBatch::new(),
            preloaded_tokenizer: None,
            ctx_set_on_thread: false,
            pooling_strategy: scratchy_core_model::embedding::PoolingStrategy::Last,
            is_pooling,
            pending_commit: None,
            #[cfg(feature = "guided-decoding")]
            grammar_states: HashMap::new(),
            #[cfg(feature = "guided-decoding")]
            grammar_factory: None,
            logits_pipeline: None,
            grammar_processor: GrammarMaskProcessor::new(),
            allowed_token_ids_processor: AllowedTokenIdsProcessor::new(),
            seal_pad_processor,
            batch_changed: false,
            batch_req_ids: Vec::new(),
            seeded_rngs: HashMap::new(),
            kv_cache_is_fp8,
            _calculate_kv_scales: calculate_kv_scales,
            _k_scale_constant: k_scale_constant,
            _v_scale_constant: v_scale_constant,
            weight_gpu_allocs: Vec::new(),
            num_gpu_blocks_saved: 0,
            progress_callback: None,
        }
    }

    /// Set progress callback for reporting model loading progress.
    pub fn set_progress_callback(&mut self, callback: std::sync::Arc<dyn Fn(&str) + Send + Sync>) {
        self.progress_callback = Some(callback);
    }

    /// Expose the GpuDevice (for NCCL stream access during TP init).
    pub fn device_ref(&self) -> Option<&GpuDevice> {
        self.device.as_ref()
    }

    /// Inject NCCL process group into the loaded model for TP communication.
    #[cfg(feature = "nccl")]
    pub fn set_tp_group(&mut self, group: std::sync::Arc<crate::nccl::NcclGroup>) {
        if let Some(ref mut model) = self.model {
            model.set_tp_group(group);
        }
    }

    /// PP plumbing was removed alongside the hand-written CUDA model
    /// forwards. These no-op stubs keep `scratchy-serving-api`'s init code compiling
    /// while load_model rejects `pp_size > 1` so the calls are unreachable.
    #[cfg(feature = "nccl")]
    pub fn set_pp_group(&mut self, _group: std::sync::Arc<crate::nccl::NcclGroup>) {}

    pub fn allocate_pp_recv_buffers(&mut self) {}

    /// Update max_num_batched_tokens (used for GPU-aware auto-detection).
    pub fn set_max_num_batched_tokens(&mut self, value: usize) {
        self.config.max_num_batched_tokens = value;
    }

    /// Expose the HF config after load_model.
    pub fn hf_config(&self) -> Option<&HfModelConfig> {
        self.hf_config.as_ref()
    }

    /// Maximum blocks per sequence: cdiv(max_model_len, block_size).
    /// Matches Python's `BlockTables.__init__` computation.
    fn max_blocks_per_seq(&self) -> usize {
        let max_model_len = self
            .hf_config
            .as_ref()
            .and_then(|c| c.max_position_embeddings)
            .unwrap_or(131072);
        max_model_len.div_ceil(self.config.block_size)
    }

    /// Expose the model directory after load_model.
    pub fn model_dir(&self) -> Option<&Path> {
        self.model_dir.as_deref()
    }

    /// Bytes per element for the model's KV cache dtype.
    pub fn resolved_dtype_elem_bytes(&self) -> usize {
        self.model_dtype.size_bytes()
    }

    /// Build grammar parser factory on demand (lazy — deferred from startup).
    #[cfg(feature = "guided-decoding")]
    fn ensure_grammar_factory(&mut self) {
        if self.grammar_factory.is_some() {
            return;
        }
        let Some(model_dir) = &self.model_dir else {
            return;
        };
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            info!("ScratchyWorker: no tokenizer.json found, grammar-guided decoding unavailable");
            return;
        }
        let tokenizer_bytes = match std::fs::read(&tokenizer_path) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    "ScratchyWorker: failed to read tokenizer.json for grammar factory: {e}"
                );
                return;
            }
        };

        match scratchy_core_model::grammar::build_parser_factory(&tokenizer_bytes) {
            Ok(factory) => {
                info!("ScratchyWorker: grammar parser factory built");
                self.grammar_factory = Some(factory);
            }
            Err(e) => {
                tracing::warn!("ScratchyWorker: failed to build grammar parser factory: {e}");
            }
        }
    }

    /// Resolve a LoRA adapter path: local directory or HF Hub download.
    fn resolve_adapter_path(&self, adapter_path: &str) -> ExecutorResult<PathBuf> {
        let path = Path::new(adapter_path);
        if path.is_dir() {
            return Ok(path.to_path_buf());
        }

        // Download from HuggingFace Hub.
        info!("Downloading LoRA adapter from HuggingFace Hub: {adapter_path}");
        let mut builder = hf_hub_downloader::Client::builder();
        if let Some(ref token) = self.config.hf_token {
            builder = builder.token(Some(token.clone()));
        }
        let client = builder.build();
        let repo = client.model(adapter_path);

        let config_path = repo.get("adapter_config.json").map_err(|e| {
            ExecutorError::WorkerInit(format!("failed to download adapter_config.json: {e}"))
        })?;
        let adapter_dir = config_path.parent().unwrap().to_path_buf();

        repo.get("adapter_model.safetensors").map_err(|e| {
            ExecutorError::WorkerInit(format!("failed to download adapter_model.safetensors: {e}"))
        })?;

        Ok(adapter_dir)
    }

    /// H2D copy a u32 slice into a caching-allocator tensor.
    fn h2d_u32(data: &[u32], device: &mut GpuDevice) -> ExecutorResult<OwnedTensor> {
        let t = device.caching.alloc_tensor(&[data.len()], GpuDType::U32);
        unsafe {
            driver::memcpy_htod_async(
                t.as_gpu_tensor().raw_ptr(),
                data.as_ptr() as *const u8,
                data.len() * 4,
                device.compute_stream,
            )
        }
        .map_err(|e| ExecutorError::WorkerExecution(format!("H2D u32: {e}")))?;
        Ok(t)
    }

    /// H2D copy an i32 slice into a caching-allocator tensor.
    fn h2d_i32(data: &[i32], device: &mut GpuDevice) -> ExecutorResult<OwnedTensor> {
        let t = device.caching.alloc_tensor(&[data.len()], GpuDType::I32);
        unsafe {
            driver::memcpy_htod_async(
                t.as_gpu_tensor().raw_ptr(),
                data.as_ptr() as *const u8,
                data.len() * 4,
                device.compute_stream,
            )
        }
        .map_err(|e| ExecutorError::WorkerExecution(format!("H2D i32: {e}")))?;
        Ok(t)
    }

    fn h2d_i64(data: &[i64], device: &mut GpuDevice) -> ExecutorResult<OwnedTensor> {
        let t = device.caching.alloc_tensor(&[data.len()], GpuDType::I64);
        unsafe {
            driver::memcpy_htod_async(
                t.as_gpu_tensor().raw_ptr(),
                data.as_ptr() as *const u8,
                data.len() * 8,
                device.compute_stream,
            )
        }
        .map_err(|e| ExecutorError::WorkerExecution(format!("H2D i64: {e}")))?;
        Ok(t)
    }

    /// Build attention metadata tensors from `AttentionMetadata`.
    ///
    /// Returns `(cu_seqlens_q, seqused_k, max_seqlen_q, max_seqlen_k,
    /// slot_mappings, block_tables)`. `seqused_k` has per-sequence K lengths
    /// `[num_reqs]` for the paged FA2 splitkv kernel.
    ///
    /// `slot_mappings` / `block_tables` are the per-KV-group set (index 0 =
    /// full/global, 1.. = sliding), one entry per group — length 1 on uniform
    /// models, `1 + N_sliding` on gemma4 hybrid SWA. Callers use `[0]` as the
    /// scalar `ForwardCtx.slot_mapping`/`block_table` and pass the whole set
    /// (when len > 1) as `HybridKvInputs`. The full/global group (0) is encoded
    /// with `full_block_size` (page-unified; == `block_size` on uniform
    /// models), the sliding groups with the base `block_size`.
    fn build_attention_tensors(
        meta: &scratchy_core_model::AttentionMetadata,
        block_size: usize,
        full_block_size: usize,
        device: &mut GpuDevice,
    ) -> ExecutorResult<AttnTensors> {
        let num_reqs = meta.num_reqs;

        // Invariant checks on attention metadata.
        debug_assert_eq!(meta.query_start_loc.len(), num_reqs + 1);
        debug_assert_eq!(meta.seq_lens.len(), num_reqs);
        debug_assert_eq!(meta.q_lens.len(), num_reqs);
        debug_assert_eq!(meta.block_ids.len(), num_reqs);
        debug_assert_eq!(meta.tokens_before.len(), num_reqs);
        debug_assert_eq!(
            meta.q_lens.iter().sum::<usize>(),
            meta.total_tokens,
            "sum(q_lens) != total_tokens"
        );
        debug_assert_eq!(*meta.query_start_loc.last().unwrap(), meta.total_tokens);
        for i in 0..num_reqs {
            debug_assert!(
                meta.seq_lens[i] >= meta.q_lens[i],
                "req {i}: seq_len={} < q_len={}",
                meta.seq_lens[i],
                meta.q_lens[i]
            );
            // `meta.block_ids` is the full/global group, so its block count is
            // sized by `full_block_size` (== base `block_size` on uniform).
            let needed_blocks = meta.seq_lens[i].div_ceil(full_block_size);
            debug_assert!(
                meta.block_ids[i].len() >= needed_blocks,
                "req {i}: block_ids.len()={} < needed_blocks={} for seq_len={} full_block_size={} \
                 tokens_before={} q_len={}",
                meta.block_ids[i].len(),
                needed_blocks,
                meta.seq_lens[i],
                full_block_size,
                meta.tokens_before[i],
                meta.q_lens[i]
            );
        }

        // cu_seqlens_q: cumulative query lengths [num_reqs + 1].
        let cu_seqlens_q: Vec<i32> = meta.query_start_loc.iter().map(|&x| x as i32).collect();
        let gpu_cu_seqlens_q = Self::h2d_i32(&cu_seqlens_q, device)?;

        // seqused_k: per-sequence K lengths [num_reqs] (used by paged FA2 splitkv kernel).
        let seqused_k: Vec<i32> = meta.seq_lens.iter().map(|&sl| sl as i32).collect();
        let gpu_seqused_k = Self::h2d_i32(&seqused_k, device)?;

        let max_seqlen_q = meta.q_lens.iter().copied().max().unwrap_or(0);
        let max_seqlen_k = meta.seq_lens.iter().copied().max().unwrap_or(0);

        // Per-KV-group tables. Group 0 = full/global (block_ids, full_block_size);
        // groups 1.. = sliding (sliding_groups[s], base block_size). On uniform
        // models `sliding_groups` is empty, so this is the single group-0 table
        // encoded with `full_block_size == block_size` — byte-identical to the
        // pre-hybrid single table.
        let groups: Vec<(&[Vec<usize>], usize)> =
            std::iter::once((meta.block_ids.as_slice(), full_block_size))
                .chain(
                    meta.sliding_groups
                        .iter()
                        .map(|s| (s.as_slice(), block_size)),
                )
                .collect();

        let mut slot_mappings: Vec<OwnedTensor> = Vec::with_capacity(groups.len());
        let mut block_tables: Vec<OwnedTensor> = Vec::with_capacity(groups.len());
        for (block_ids_per_req, bs) in groups {
            // slot_mapping: for each new token, (block_id * bs + offset).
            let mut slot_mapping = Vec::with_capacity(meta.total_tokens);
            for (i, block_ids) in block_ids_per_req.iter().enumerate().take(num_reqs) {
                let tokens_before = meta.tokens_before[i];
                let q_len = meta.q_lens[i];
                for t in 0..q_len {
                    let abs_pos = tokens_before + t;
                    let block_idx = abs_pos / bs;
                    let offset = abs_pos % bs;
                    if block_idx < block_ids.len() {
                        slot_mapping.push((block_ids[block_idx] * bs + offset) as i64);
                    } else {
                        slot_mapping.push(-1i64);
                    }
                }
            }
            slot_mappings.push(Self::h2d_i64(&slot_mapping, device)?);

            // block_table: [num_reqs, max_blocks] i32 (padded with 0). Each
            // group has its own max_blocks (the kernel reads the row stride
            // from the tensor shape at runtime — no baked constant).
            let max_blocks = block_ids_per_req.iter().map(|b| b.len()).max().unwrap_or(0);
            let bt = if max_blocks > 0 {
                let mut block_table = vec![0i32; num_reqs * max_blocks];
                for (i, blocks) in block_ids_per_req.iter().enumerate() {
                    for (j, &bid) in blocks.iter().enumerate() {
                        block_table[i * max_blocks + j] = bid as i32;
                    }
                }
                let mut t = Self::h2d_i32(&block_table, device)?;
                unsafe { t.reshape(&[num_reqs, max_blocks], GpuDType::I32) };
                t
            } else {
                device.caching.alloc_tensor(&[0], GpuDType::I32)
            };
            block_tables.push(bt);
        }

        Ok((
            gpu_cu_seqlens_q,
            gpu_seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mappings,
            block_tables,
        ))
    }

    /// Run the vision encoder for any active request whose first prefill
    /// chunk includes its image placeholders. Returns
    /// `(mm_embeds, embed_patches)` ready to thread into
    /// `MmForwardInputs` — `mm_embeds` is `[total_mm_tokens, hidden]`,
    /// `embed_patches[i].token_offset` is the destination row in the
    /// flat input-id sequence (post batch concatenation).
    ///
    /// Returns `None` when there's nothing to encode this step:
    /// - non-Scratchy arch, or Scratchy arch with no MM handle (text-only),
    /// - no req in the active batch carries `mm_data`,
    /// - no MM-bearing req is at its first scheduling step
    ///   (`tokens_before == 0`).
    ///
    /// MVP: assumes the prefill chunk for an MM-bearing req covers ALL
    /// its image placeholders (no chunked-prefill split across image
    /// boundaries). For typical Qwen2-VL-2B prompts (256 tokens / image)
    /// this holds inside a single 2048-token prefill chunk; the chunked
    /// case is a follow-up.
    ///
    /// # Safety
    /// `device` must be the live CUDA device the encoder kernels
    /// launch on. The returned `OwnedTensor` is on `device.compute_stream`
    /// — the caller must keep it alive across the language-side
    /// `model.forward` call so the `Instruction::Embed::eval` D2D
    /// splice still has live source memory.
    unsafe fn run_mm_vision_forward(
        model: &CudaModel,
        mm_data_buffers: &HashMap<String, scratchy_core_common::MultimodalData>,
        prepared: &PreparedInputs,
        device: &mut GpuDevice,
    ) -> Option<(OwnedTensor, Vec<crate::EmbedPatch>)> {
        // `CudaModel` collapsed to a single `Scratchy(_)` variant in
        // cbc59cc2d when the hand-written CUDA model forwards were
        // removed; the destructure is irrefutable.
        let CudaModel::Scratchy(fm) = model;
        let mm = fm.mm.as_ref()?;
        let meta = &prepared.attn_meta;
        let mut pixel_inputs: Vec<crate::PixelInput<'_>> = Vec::new();
        let mut placeholders: Vec<crate::EmbedPatch> = Vec::new();
        for (i, req) in prepared.req_inputs.iter().enumerate() {
            let Some(mm_data) = mm_data_buffers.get(&req.req_id) else {
                continue;
            };
            // Only run vision_forward for the first prefill chunk —
            // subsequent decode steps consume already-spliced KV cache.
            if meta.tokens_before[i] != 0 {
                continue;
            }
            let batch_offset = meta.query_start_loc[i] as u32;
            for (img, ph) in mm_data.images.iter().zip(mm_data.image_placeholders.iter()) {
                pixel_inputs.push(crate::PixelInput {
                    pixels: &img.pixels,
                    height: img.height as u32,
                    width: img.width as u32,
                });
                placeholders.push(crate::EmbedPatch {
                    token_offset: batch_offset + ph.offset as u32,
                    length: ph.length as u32,
                    // Filled by `vision_forward` from per-image
                    // `grid_thw` + `spatial_merge_size`. Default zero
                    // here is invalid for MRoPE; the encoder must
                    // populate before the executor reads them.
                    grid_t: 0,
                    grid_h_merged: 0,
                    grid_w_merged: 0,
                });
            }
        }
        if pixel_inputs.is_empty() {
            return None;
        }
        let (out, patches) = unsafe {
            mm.vision_forward(
                &pixel_inputs,
                &placeholders,
                crate::ForwardDeviceHandle::new(device),
            )
        };
        Some((out, patches))
    }

    /// Build per-req seq-space MM info for **every** MM-bearing req in
    /// the active batch — including ones whose vision encoder already
    /// ran on a prior step (cached prefix). Empty inner `Vec` for any
    /// req without mm_data. Returns an all-empty outer `Vec` when no
    /// req carries mm_data (text-only batch → caller falls back to 1D
    /// positions, byte-identical to pre-MM behavior).
    ///
    /// Tuple is `(seq_offset, length, grid_t, grid_h_merged, grid_w_merged)`
    /// in seq space (offset relative to the req's full sequence start,
    /// not batch start). [`Self::build_mrope_positions_2d`] walks the
    /// entire seq — including the cached prefix — so the cursor lands
    /// at the same position the encoder used to write KV the first
    /// time around.
    fn build_per_req_mm_seq_info(
        model: &CudaModel,
        mm_data_buffers: &HashMap<String, scratchy_core_common::MultimodalData>,
        prepared: &PreparedInputs,
    ) -> Vec<Vec<SeqMmInfo>> {
        let mut per_req: Vec<Vec<SeqMmInfo>> = vec![Vec::new(); prepared.req_inputs.len()];
        // `CudaModel` is a single-variant enum; the destructure is irrefutable.
        let CudaModel::Scratchy(fm) = model;
        let Some(mm) = fm.mm.as_ref() else {
            return per_req;
        };
        let mut any = false;
        for (i, req) in prepared.req_inputs.iter().enumerate() {
            let Some(mm_data) = mm_data_buffers.get(&req.req_id) else {
                continue;
            };
            if mm_data.images.is_empty() {
                continue;
            }
            let pixel_inputs: Vec<crate::PixelInput<'_>> = mm_data
                .images
                .iter()
                .map(|img| crate::PixelInput {
                    pixels: &img.pixels,
                    height: img.height as u32,
                    width: img.width as u32,
                })
                .collect();
            let grids = mm.embed_patch_grids(&pixel_inputs);
            for (ph, &(gt, mh, mw)) in mm_data.image_placeholders.iter().zip(grids.iter()) {
                per_req[i].push((ph.offset as u32, ph.length as u32, gt, mh, mw));
            }
            // Sort by seq offset so the walk visits patches in order.
            per_req[i].sort_by_key(|t| t.0);
            any = true;
        }
        if !any {
            return Vec::new();
        }
        per_req
    }

    /// D2H copy logits to CPU f32 vec.
    fn logits_to_cpu(logits: GpuTensor, device: &GpuDevice) -> ExecutorResult<Vec<f32>> {
        let num_elements = logits.numel();
        let nbytes = num_elements * logits.dtype().size_bytes();

        let mut host_buf = vec![0u8; nbytes];
        unsafe {
            driver::memcpy_dtoh_async(
                host_buf.as_mut_ptr(),
                logits.raw_ptr() as *const u8,
                nbytes,
                device.compute_stream,
            )
        }
        .map_err(|e| ExecutorError::WorkerExecution(format!("D2H logits: {e}")))?;
        unsafe { driver::stream_synchronize(device.compute_stream) }
            .map_err(|e| ExecutorError::WorkerExecution(format!("sync: {e}")))?;

        let f32_vec: Vec<f32> = match logits.dtype() {
            GpuDType::F32 => {
                let ptr = host_buf.as_ptr() as *const f32;
                unsafe { std::slice::from_raw_parts(ptr, num_elements) }.to_vec()
            }
            GpuDType::F16 => {
                let ptr = host_buf.as_ptr() as *const half::f16;
                let slice = unsafe { std::slice::from_raw_parts(ptr, num_elements) };
                slice.iter().map(|v| v.to_f32()).collect()
            }
            GpuDType::BF16 => {
                let ptr = host_buf.as_ptr() as *const half::bf16;
                let slice = unsafe { std::slice::from_raw_parts(ptr, num_elements) };
                slice.iter().map(|v| v.to_f32()).collect()
            }
            _ => {
                return Err(ExecutorError::WorkerExecution(
                    "unexpected logits dtype".into(),
                ));
            }
        };
        Ok(f32_vec)
    }

    /// Pool hidden states on GPU, D2H the small result, cast to f32, L2 normalize.
    ///
    /// `hidden_states` is `[num_tokens, hidden_size]` on GPU in model dtype.
    /// Returns `Vec<f32>` of length `hidden_size`, L2-normalized.
    fn pool_and_normalize(
        hidden_states: GpuTensor,
        num_tokens: usize,
        strategy: scratchy_core_model::embedding::PoolingStrategy,
        device: &mut GpuDevice,
    ) -> ExecutorResult<EmbeddingData> {
        use scratchy_core_model::embedding::PoolingStrategy;

        let hidden_size = hidden_states.dim(1);

        // AllTokens: D2H all rows, L2-normalize each, return as Multi.
        if strategy == PoolingStrategy::AllTokens {
            let all_f32 = Self::logits_to_cpu(hidden_states, device)?;
            let mut rows = Vec::with_capacity(num_tokens);
            for row in 0..num_tokens {
                let start = row * hidden_size;
                let end = start + hidden_size;
                let mut vec: Vec<f32> = all_f32[start..end].to_vec();
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in &mut vec {
                        *v /= norm;
                    }
                }
                rows.push(vec);
            }
            return Ok(EmbeddingData::Multi(rows));
        }

        // 1. GPU-side pooling: extract [hidden_size] from [num_tokens, hidden_size].
        let pooled_gpu = match strategy {
            PoolingStrategy::Last => unsafe {
                crate::kernels::pool_select_row(
                    hidden_states,
                    num_tokens - 1,
                    &mut device.caching,
                    device.compute_stream,
                )
            },
            PoolingStrategy::Cls => unsafe {
                crate::kernels::pool_select_row(
                    hidden_states,
                    0,
                    &mut device.caching,
                    device.compute_stream,
                )
            },
            PoolingStrategy::Mean => {
                // Mean pooling needs f32 for cuBLAS gemv. For bf16/f16 inputs,
                // D2H the small pooled tensor and compute mean on CPU.
                // (Avoids needing a bf16→f32 cast kernel for this small vector.)
                // Actually, for Mean we need to average ALL rows, so D2H all rows
                // is expensive. Instead, D2H just the pooled row for Last/CLS,
                // but for Mean we need GPU computation.
                //
                // Strategy: if dtype is f32, use GPU gemv. Otherwise, D2H the
                // full hidden_states and compute mean on CPU. For typical embedding
                // use cases, seq_len * hidden_size * 2 bytes is manageable
                // (512 * 4096 * 2 = 4MB). This is the simple path.
                //
                // TODO: add bf16→f32 cast kernel for full GPU mean pooling.
                if hidden_states.dtype() == GpuDType::F32 {
                    unsafe {
                        crate::kernels::pool_mean_f32(
                            hidden_states,
                            &device.cublas,
                            &mut device.caching,
                            device.compute_stream,
                        )
                    }
                } else {
                    // D2H all hidden states, compute mean on CPU.
                    let all_f32 = Self::logits_to_cpu(hidden_states, device)?;
                    let mut mean = vec![0.0f32; hidden_size];
                    let n = num_tokens as f32;
                    for row in 0..num_tokens {
                        let start = row * hidden_size;
                        for (j, val) in mean.iter_mut().enumerate() {
                            *val += all_f32[start + j] / n;
                        }
                    }
                    // L2 normalize on CPU.
                    let norm: f32 = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for v in &mut mean {
                            *v /= norm;
                        }
                    }
                    return Ok(EmbeddingData::Single(mean));
                }
            }
            PoolingStrategy::AllTokens => unreachable!("handled by early return"),
        };

        // 2. D2H the small [hidden_size] vector.
        let pooled_gpu_t = pooled_gpu.as_gpu_tensor();
        let nbytes = hidden_size * pooled_gpu_t.dtype().size_bytes();
        let mut host_buf = vec![0u8; nbytes];
        unsafe {
            driver::memcpy_dtoh_async(
                host_buf.as_mut_ptr(),
                pooled_gpu_t.raw_ptr() as *const u8,
                nbytes,
                device.compute_stream,
            )
        }
        .map_err(|e| ExecutorError::WorkerExecution(format!("D2H pooled: {e}")))?;
        unsafe { driver::stream_synchronize(device.compute_stream) }
            .map_err(|e| ExecutorError::WorkerExecution(format!("sync: {e}")))?;

        // 3. Cast to f32 on CPU.
        let f32_vec: Vec<f32> = match pooled_gpu_t.dtype() {
            GpuDType::F32 => {
                let ptr = host_buf.as_ptr() as *const f32;
                unsafe { std::slice::from_raw_parts(ptr, hidden_size) }.to_vec()
            }
            GpuDType::F16 => {
                let ptr = host_buf.as_ptr() as *const half::f16;
                let slice = unsafe { std::slice::from_raw_parts(ptr, hidden_size) };
                slice.iter().map(|v| v.to_f32()).collect()
            }
            GpuDType::BF16 => {
                let ptr = host_buf.as_ptr() as *const half::bf16;
                let slice = unsafe { std::slice::from_raw_parts(ptr, hidden_size) };
                slice.iter().map(|v| v.to_f32()).collect()
            }
            _ => {
                return Err(ExecutorError::WorkerExecution(
                    "unexpected dtype for pooled embedding".into(),
                ));
            }
        };

        // 4. L2 normalize on CPU (tiny vector, ~4096 floats = 16KB).
        let norm: f32 = f32_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            Ok(EmbeddingData::Single(
                f32_vec.into_iter().map(|x| x / norm).collect(),
            ))
        } else {
            Ok(EmbeddingData::Single(f32_vec))
        }
    }

    /// D2H copy token IDs using pinned staging if available.
    /// Enqueue async D2H + sync, returning token IDs. Uses the pinned double-
    /// buffer at `buf_idx` when staging is available.
    fn d2h_token_ids_sync(
        staging: Option<&HostStaging>,
        buf_idx: usize,
        gpu_tensor: &GpuTensor,
        num_reqs: usize,
        device: &mut GpuDevice,
    ) -> ExecutorResult<Vec<u32>> {
        if let Some(stg) = staging {
            // Async D2H: compute→event→transfer_stream→D2H→event→sync event.
            unsafe {
                device.async_d2h(
                    stg.host_token_ids[buf_idx].ptr(),
                    gpu_tensor.raw_ptr() as *const u8,
                    num_reqs * 4,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("async D2H token ids: {e}")))?;
            device
                .sync_d2h()
                .map_err(|e| ExecutorError::WorkerExecution(format!("sync d2h: {e}")))?;
            Ok(unsafe { stg.host_token_ids[buf_idx].slice::<u32>(num_reqs) }.to_vec())
        } else {
            let mut ids = vec![0u32; num_reqs];
            unsafe {
                device.async_d2h(
                    ids.as_mut_ptr() as *mut u8,
                    gpu_tensor.raw_ptr() as *const u8,
                    num_reqs * 4,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("async D2H token ids: {e}")))?;
            device
                .sync_d2h()
                .map_err(|e| ExecutorError::WorkerExecution(format!("sync d2h: {e}")))?;
            Ok(ids)
        }
    }

    /// Enqueue async D2H without sync. Returns the buffer index used.
    /// Caller must sync via `device.sync_d2h()` before reading the buffer.
    fn d2h_token_ids_async(
        staging: &HostStaging,
        buf_idx: usize,
        gpu_tensor: &GpuTensor,
        num_reqs: usize,
        device: &GpuDevice,
    ) -> ExecutorResult<()> {
        unsafe {
            device.async_d2h(
                staging.host_token_ids[buf_idx].ptr(),
                gpu_tensor.raw_ptr() as *const u8,
                num_reqs * 4,
            )
        }
        .map_err(|e| ExecutorError::WorkerExecution(format!("async D2H token ids: {e}")))?;
        Ok(())
    }
    /// Full GPU sampling pipeline: grammar mask → logit processors → sample → logprobs.
    /// No CPU fallback — everything stays on GPU, matching Python vLLM exactly.
    ///
    /// Uses the `LogitsProcessorPipeline` for persistent GPU state (logit_bias,
    /// penalties, min_tokens) and a separate `GrammarMaskProcessor` for grammar.
    #[allow(clippy::too_many_arguments)]
    fn gpu_sample_and_finalize(
        sampling_params_map: &HashMap<String, SamplingParams>,
        #[cfg(feature = "guided-decoding")] grammar_states: &mut HashMap<
            String,
            scratchy_core_model::grammar::GrammarGuide,
        >,
        grammar_processor: &GrammarMaskProcessor,
        allowed_token_ids_processor: &AllowedTokenIdsProcessor,
        seal_pad_processor: &SealPadProcessor,
        logits_pipeline: Option<&LogitsProcessorPipeline>,
        seeded_rngs: &mut HashMap<String, rand::rngs::StdRng>,
        host_staging: &Option<HostStaging>,
        input_batch: &mut InputBatch,
        token_buffers: &mut HashMap<String, Vec<u32>>,
        prompt_lengths: &HashMap<String, usize>,
        logits: GpuTensor,
        prepared: PreparedInputs,
        device: &mut GpuDevice,
        all_greedy: bool,
        vocab_size: usize,
    ) -> ExecutorResult<ModelRunnerOutput> {
        use rand::Rng;
        let num_reqs = prepared.req_inputs.len();
        let total_tokens = prepared.flat_token_ids.len();

        let any_spec_decode = prepared
            .req_inputs
            .iter()
            .any(|r| !r.spec_token_ids.is_empty());

        // --- Spec decode path: greedy rejection sampling ---
        // When any request has draft tokens, logits shape is [total_tokens, vocab_size]
        // (all positions). We argmax all rows, then do CPU greedy rejection.
        if any_spec_decode && all_greedy {
            return Self::spec_decode_greedy_sample(
                logits,
                total_tokens,
                prepared,
                device,
                host_staging,
                input_batch,
                token_buffers,
            );
        }

        // Helper: get a per-request random seed (u32) for GPU sampling kernels.
        //
        // Seeded requests draw from their per-request `StdRng`. Unseeded requests
        // hash `(req_id, generated-token-count)` via the shared, backend- and
        // TP-rank-independent `scratchy_core_common::fnv_seed` — the SAME formula
        // + the SAME per-request position the metal sampler uses (see
        // `scratchy_target_metal::sampling::gather_gpu_sample_params`), so an
        // unseeded request gets the identical seed (and sample) on either
        // backend. The
        // generated-token count is TP-stable (`token_buffers` is identical on all
        // ranks) and advances each decode step so the seed varies.
        //
        // The GPU Gumbel kernel uses the seed as a Philox seed for per-element
        // randomness (matching Python vLLM's tl.rand).
        let mut gen_seed = |req_id: &str, position: u32| -> u32 {
            if let Some(rng) = seeded_rngs.get_mut(req_id) {
                rng.random::<u32>()
            } else {
                scratchy_core_common::fnv_seed(req_id, position)
            }
        };

        // Determine which features are needed.
        let any_logprobs = prepared.req_inputs.iter().any(|r| {
            sampling_params_map
                .get(&r.req_id)
                .is_some_and(|p| p.logprobs.is_some())
        });

        let any_grammar = grammar_processor.is_active();
        let any_allowed = allowed_token_ids_processor.is_active();
        let any_seal_pad = seal_pad_processor.is_active();

        let pipeline_active = logits_pipeline.is_some_and(|p| p.any_active());
        let needs_f32 =
            pipeline_active || any_grammar || any_allowed || any_seal_pad || any_logprobs;

        if !needs_f32 {
            // Fast path: no modifications needed, use native dtype sampling.
            if all_greedy {
                let token_ids_owned = unsafe {
                    crate::kernels::argmax_batched(
                        logits,
                        &mut device.caching,
                        device.compute_stream,
                    )
                };
                let token_ids_gpu = token_ids_owned.as_gpu_tensor();
                return Self::finalize_d2h_and_commit(
                    &token_ids_gpu,
                    num_reqs,
                    prepared,
                    device,
                    host_staging,
                    0,
                    input_batch,
                    token_buffers,
                    prompt_lengths,
                );
            }

            // Non-greedy fast path: Gumbel or full sampling on native dtype.
            let all_no_filter = prepared.req_inputs.iter().all(|r| {
                sampling_params_map
                    .get(&r.req_id)
                    .is_none_or(|p| p.top_k <= 0 && p.top_p >= 1.0 && p.min_p <= 0.0)
            });

            let token_ids_owned = if all_no_filter {
                // Pack [temps: f32, seeds: u32] — both 4 bytes, same stride.
                let stride = num_reqs * 4;
                let total_bytes = stride * 2;
                let packed_ptr = Self::get_sampling_packed_ptr_s(host_staging, total_bytes);
                let temps_ptr = packed_ptr as *mut f32;
                let seeds_ptr = unsafe { packed_ptr.add(stride) as *mut u32 };
                for (i, req_slice) in prepared.req_inputs.iter().enumerate() {
                    let params = sampling_params_map.get(&req_slice.req_id);
                    let t = params.map_or(1.0f32, |p| p.temperature.max(1e-7) as f32);
                    let plen = prompt_lengths.get(&req_slice.req_id).copied().unwrap_or(0);
                    let position = token_buffers
                        .get(&req_slice.req_id)
                        .map(|b| b.len().saturating_sub(plen))
                        .unwrap_or(0) as u32;
                    let seed = gen_seed(&req_slice.req_id, position);
                    unsafe {
                        *temps_ptr.add(i) = t;
                        *seeds_ptr.add(i) = seed;
                    }
                }
                let gpu_packed_owned = device
                    .caching
                    .alloc_tensor(&[total_bytes / 4], GpuDType::F32);
                let gpu_packed = gpu_packed_owned.as_gpu_tensor();
                unsafe {
                    driver::memcpy_htod_async(
                        gpu_packed.raw_ptr(),
                        packed_ptr,
                        total_bytes,
                        device.compute_stream,
                    )
                }
                .map_err(|e| ExecutorError::WorkerExecution(format!("H2D sampling: {e}")))?;
                let base = gpu_packed.raw_ptr();
                let gpu_temps = unsafe { GpuTensor::new(base, &[num_reqs], GpuDType::F32) };
                let gpu_seeds =
                    unsafe { GpuTensor::new(base.add(stride), &[num_reqs], GpuDType::U32) };
                unsafe {
                    crate::kernels::sample_gumbel_batched(
                        logits,
                        gpu_temps,
                        gpu_seeds,
                        &mut device.caching,
                        device.compute_stream,
                    )
                }
            } else {
                let stride = num_reqs * 4;
                let total_bytes = stride * 5;
                let packed_ptr = Self::get_sampling_packed_ptr_s(host_staging, total_bytes);
                let temps_ptr = packed_ptr as *mut f32;
                let top_ks_ptr = unsafe { packed_ptr.add(stride) as *mut i32 };
                let top_ps_ptr = unsafe { packed_ptr.add(stride * 2) as *mut f32 };
                let min_ps_ptr = unsafe { packed_ptr.add(stride * 3) as *mut f32 };
                let randoms_ptr = unsafe { packed_ptr.add(stride * 4) as *mut f32 };
                for (i, req_slice) in prepared.req_inputs.iter().enumerate() {
                    let params = sampling_params_map.get(&req_slice.req_id);
                    let (t, k, p, mp) = params.map_or((1.0f32, 0i32, 1.0f32, 0.0f32), |p| {
                        (
                            p.temperature.max(1e-7) as f32,
                            p.top_k,
                            p.top_p as f32,
                            p.min_p as f32,
                        )
                    });
                    let plen = prompt_lengths.get(&req_slice.req_id).copied().unwrap_or(0);
                    let position = token_buffers
                        .get(&req_slice.req_id)
                        .map(|b| b.len().saturating_sub(plen))
                        .unwrap_or(0) as u32;
                    let random = scratchy_core_common::seed_to_uniform(gen_seed(
                        &req_slice.req_id,
                        position,
                    ));
                    unsafe {
                        *temps_ptr.add(i) = t;
                        *top_ks_ptr.add(i) = k;
                        *top_ps_ptr.add(i) = p;
                        *min_ps_ptr.add(i) = mp;
                        *randoms_ptr.add(i) = random;
                    }
                }
                let gpu_packed_owned = device
                    .caching
                    .alloc_tensor(&[total_bytes / 4], GpuDType::F32);
                let gpu_packed = gpu_packed_owned.as_gpu_tensor();
                unsafe {
                    driver::memcpy_htod_async(
                        gpu_packed.raw_ptr(),
                        packed_ptr,
                        total_bytes,
                        device.compute_stream,
                    )
                }
                .map_err(|e| ExecutorError::WorkerExecution(format!("H2D sampling: {e}")))?;
                let base = gpu_packed.raw_ptr();
                let gpu_temps = unsafe { GpuTensor::new(base, &[num_reqs], GpuDType::F32) };
                let gpu_top_ks =
                    unsafe { GpuTensor::new(base.add(stride), &[num_reqs], GpuDType::U32) };
                let gpu_top_ps =
                    unsafe { GpuTensor::new(base.add(stride * 2), &[num_reqs], GpuDType::F32) };
                let gpu_min_ps =
                    unsafe { GpuTensor::new(base.add(stride * 3), &[num_reqs], GpuDType::F32) };
                let gpu_randoms =
                    unsafe { GpuTensor::new(base.add(stride * 4), &[num_reqs], GpuDType::F32) };
                unsafe {
                    crate::kernels::sample_batched(
                        logits,
                        gpu_temps,
                        gpu_top_ks,
                        gpu_top_ps,
                        gpu_min_ps,
                        gpu_randoms,
                        &mut device.caching,
                        device.compute_stream,
                    )
                }
            };
            let token_ids_gpu = token_ids_owned.as_gpu_tensor();
            return Self::finalize_d2h_and_commit(
                &token_ids_gpu,
                num_reqs,
                prepared,
                device,
                host_staging,
                0,
                input_batch,
                token_buffers,
                prompt_lengths,
            );
        }

        // ---- Slow(er) path: cast to f32, apply modifications, sample on GPU ----

        // 1. Cast logits to f32.
        let logits_f32_owned = unsafe {
            crate::kernels::cast_logits_to_f32(logits, &mut device.caching, device.compute_stream)
        };
        let logits_f32 = logits_f32_owned.as_gpu_tensor();

        // 2. Save raw logits for logprobs (GPU copy, before modifications).
        let raw_logits_for_logprobs = if any_logprobs {
            let copy = device
                .caching
                .alloc_tensor(&[num_reqs, vocab_size], GpuDType::F32);
            unsafe {
                driver::memcpy_dtod_async(
                    copy.as_gpu_tensor().raw_ptr(),
                    logits_f32.raw_ptr() as *const u8,
                    num_reqs * vocab_size * 4,
                    device.compute_stream,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("D2D logits copy: {e}")))?;
            Some(copy)
        } else {
            None
        };

        // 3. Apply grammar mask, allowed_token_ids, and seal-pad on GPU (all need backup logits).
        let needs_mask_backup = (any_grammar && grammar_processor.needs_backup())
            || (any_allowed && allowed_token_ids_processor.needs_backup())
            || (any_seal_pad && seal_pad_processor.needs_backup());
        let _mask_backup_owned = if needs_mask_backup {
            let backup_owned = if raw_logits_for_logprobs.is_some() {
                None
            } else {
                let bk = device
                    .caching
                    .alloc_tensor(&[num_reqs, vocab_size], GpuDType::F32);
                unsafe {
                    driver::memcpy_dtod_async(
                        bk.as_gpu_tensor().raw_ptr(),
                        logits_f32.raw_ptr() as *const u8,
                        num_reqs * vocab_size * 4,
                        device.compute_stream,
                    )
                }
                .map_err(|e| ExecutorError::WorkerExecution(format!("D2D mask backup: {e}")))?;
                Some(bk)
            };
            let backup = if let Some(raw) = raw_logits_for_logprobs.as_ref() {
                raw.as_gpu_tensor()
            } else {
                backup_owned.as_ref().unwrap().as_gpu_tensor()
            };

            if any_grammar {
                grammar_processor.apply_with_backup(logits_f32, backup, device);
            }
            if any_allowed {
                allowed_token_ids_processor.apply_with_backup(logits_f32, backup, device);
            }
            if any_seal_pad {
                seal_pad_processor.apply_with_backup(logits_f32, backup, device);
            }
            backup_owned
        } else {
            None
        };

        // 4. Apply logit processors pipeline (min_tokens, logit_bias, penalties).
        if let Some(pipeline) = logits_pipeline {
            pipeline.apply_pre_sampling(logits_f32, device);
        }

        // 5. Sample on GPU (from modified f32 logits).
        let token_ids_owned = if all_greedy {
            // Argmax on f32 logits.
            unsafe {
                crate::kernels::argmax_batched(
                    logits_f32,
                    &mut device.caching,
                    device.compute_stream,
                )
            }
        } else {
            // Full sampling on f32 logits.
            let stride = num_reqs * 4;
            let total_bytes = stride * 5;
            let packed_ptr = Self::get_sampling_packed_ptr_s(host_staging, total_bytes);
            let temps_ptr = packed_ptr as *mut f32;
            let top_ks_ptr = unsafe { packed_ptr.add(stride) as *mut i32 };
            let top_ps_ptr = unsafe { packed_ptr.add(stride * 2) as *mut f32 };
            let min_ps_ptr = unsafe { packed_ptr.add(stride * 3) as *mut f32 };
            let randoms_ptr = unsafe { packed_ptr.add(stride * 4) as *mut f32 };
            for (i, req_slice) in prepared.req_inputs.iter().enumerate() {
                let params = sampling_params_map.get(&req_slice.req_id);
                let (t, k, p, mp) = params.map_or((1.0f32, 0i32, 1.0f32, 0.0f32), |p| {
                    (
                        p.temperature.max(1e-7) as f32,
                        p.top_k,
                        p.top_p as f32,
                        p.min_p as f32,
                    )
                });
                let plen = prompt_lengths.get(&req_slice.req_id).copied().unwrap_or(0);
                let position = token_buffers
                    .get(&req_slice.req_id)
                    .map(|b| b.len().saturating_sub(plen))
                    .unwrap_or(0) as u32;
                let random =
                    scratchy_core_common::seed_to_uniform(gen_seed(&req_slice.req_id, position));
                unsafe {
                    *temps_ptr.add(i) = t;
                    *top_ks_ptr.add(i) = k;
                    *top_ps_ptr.add(i) = p;
                    *min_ps_ptr.add(i) = mp;
                    *randoms_ptr.add(i) = random;
                }
            }
            let gpu_packed_owned = device
                .caching
                .alloc_tensor(&[total_bytes / 4], GpuDType::F32);
            let gpu_packed = gpu_packed_owned.as_gpu_tensor();
            unsafe {
                driver::memcpy_htod_async(
                    gpu_packed.raw_ptr(),
                    packed_ptr,
                    total_bytes,
                    device.compute_stream,
                )
            }
            .map_err(|e| ExecutorError::WorkerExecution(format!("H2D sampling: {e}")))?;
            let base = gpu_packed.raw_ptr();
            let gpu_temps = unsafe { GpuTensor::new(base, &[num_reqs], GpuDType::F32) };
            let gpu_top_ks =
                unsafe { GpuTensor::new(base.add(stride), &[num_reqs], GpuDType::U32) };
            let gpu_top_ps =
                unsafe { GpuTensor::new(base.add(stride * 2), &[num_reqs], GpuDType::F32) };
            let gpu_min_ps =
                unsafe { GpuTensor::new(base.add(stride * 3), &[num_reqs], GpuDType::F32) };
            let gpu_randoms =
                unsafe { GpuTensor::new(base.add(stride * 4), &[num_reqs], GpuDType::F32) };
            unsafe {
                crate::kernels::sample_batched(
                    logits_f32,
                    gpu_temps,
                    gpu_top_ks,
                    gpu_top_ps,
                    gpu_min_ps,
                    gpu_randoms,
                    &mut device.caching,
                    device.compute_stream,
                )
            }
        };
        let token_ids_gpu = token_ids_owned.as_gpu_tensor();

        // 7. Gather logprobs on GPU (if needed).
        let logprobs_output = if let Some(raw_logits_owned) = raw_logits_for_logprobs {
            let raw_logits = raw_logits_owned.as_gpu_tensor();
            // Find max num_logprobs across requests.
            let max_logprobs = prepared
                .req_inputs
                .iter()
                .filter_map(|r| sampling_params_map.get(&r.req_id).and_then(|p| p.logprobs))
                .max()
                .unwrap_or(0) as usize;

            if max_logprobs > 0 {
                let (topk_lp, topk_idx, topk_ranks) = unsafe {
                    crate::kernels::log_softmax_topk_gather(
                        raw_logits,
                        token_ids_gpu,
                        max_logprobs,
                        &mut device.caching,
                        device.compute_stream,
                    )
                };

                // D2H the small topk tensors.
                let k = max_logprobs + 1;
                let n_elements = num_reqs * k;
                let mut host_lp = vec![0.0f32; n_elements];
                let mut host_idx = vec![0i32; n_elements];
                let mut host_ranks = vec![0u32; n_elements];
                unsafe {
                    driver::memcpy_dtoh_async(
                        host_lp.as_mut_ptr() as *mut u8,
                        topk_lp.as_gpu_tensor().raw_ptr() as *const u8,
                        n_elements * 4,
                        device.compute_stream,
                    )
                    .map_err(|e| ExecutorError::WorkerExecution(format!("D2H logprobs: {e}")))?;
                    driver::memcpy_dtoh_async(
                        host_idx.as_mut_ptr() as *mut u8,
                        topk_idx.as_gpu_tensor().raw_ptr() as *const u8,
                        n_elements * 4,
                        device.compute_stream,
                    )
                    .map_err(|e| ExecutorError::WorkerExecution(format!("D2H logprob idx: {e}")))?;
                    driver::memcpy_dtoh_async(
                        host_ranks.as_mut_ptr() as *mut u8,
                        topk_ranks.as_gpu_tensor().raw_ptr() as *const u8,
                        n_elements * 4,
                        device.compute_stream,
                    )
                    .map_err(|e| {
                        ExecutorError::WorkerExecution(format!("D2H logprob ranks: {e}"))
                    })?;
                    driver::stream_synchronize(device.compute_stream)
                        .map_err(|e| ExecutorError::WorkerExecution(format!("sync: {e}")))?;
                }

                // Build per-request LogprobsOutput.
                let mut logprobs_map: HashMap<String, Vec<scratchy_core_common::LogprobsOutput>> =
                    HashMap::new();
                for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
                    let requested_n = sampling_params_map
                        .get(&req_slice.req_id)
                        .and_then(|p| p.logprobs);
                    if let Some(n) = requested_n {
                        let n = n as usize;
                        let base = req_idx * k;
                        // Slot 0 = sampled token.
                        let sampled = scratchy_core_common::TokenLogprob {
                            token_id: host_idx[base] as u32,
                            logprob: host_lp[base],
                            rank: host_ranks[base],
                        };
                        let mut top_logprobs = Vec::with_capacity(n);
                        for j in 1..=n.min(max_logprobs) {
                            top_logprobs.push(scratchy_core_common::TokenLogprob {
                                token_id: host_idx[base + j] as u32,
                                logprob: host_lp[base + j],
                                rank: host_ranks[base + j],
                            });
                        }
                        logprobs_map
                            .entry(req_slice.req_id.clone())
                            .or_default()
                            .push(scratchy_core_common::LogprobsOutput {
                                sampled,
                                top_logprobs,
                            });
                    }
                }
                Some(logprobs_map)
            } else {
                None
            }
        } else {
            None
        };

        // 8. D2H sampled tokens, commit, build output.
        let host_ids =
            Self::d2h_token_ids_sync(host_staging.as_ref(), 0, &token_ids_gpu, num_reqs, device)?;

        // Build discard mask for intermediate prefill chunks.
        let mut discard = vec![false; num_reqs];
        for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
            if req_slice.token_count > 1 {
                let prompt_len = prompt_lengths.get(&req_slice.req_id).copied().unwrap_or(0);
                let tokens_in_pool = input_batch.tokens_in_pool_for(&req_slice.req_id);
                let seq_len_after = tokens_in_pool + req_slice.token_count;
                if seq_len_after < prompt_len {
                    discard[req_idx] = true;
                }
            }
        }

        for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
            let tok = host_ids[req_idx];

            // Grammar advance on CPU (Python also does FSM state on CPU).
            #[cfg(feature = "guided-decoding")]
            if !discard[req_idx]
                && let Some(g) = grammar_states.get_mut(&req_slice.req_id)
            {
                g.advance(tok);
            }

            input_batch.commit_step(
                &req_slice.req_id,
                &[tok],
                req_slice.token_count,
                !req_slice.spec_token_ids.is_empty(),
            );
            if !discard[req_idx]
                && let Some(buf) = token_buffers.get_mut(&req_slice.req_id)
            {
                buf.push(tok);
            }
        }

        let req_ids: Vec<String> = prepared
            .req_inputs
            .iter()
            .map(|r| r.req_id.clone())
            .collect();
        input_batch.reclaim_buffers(prepared);
        let mut output = ModelRunnerOutput::from_ordered(req_ids, host_ids);

        // Clear sampled tokens for discarded (intermediate prefill) requests.
        for (idx, d) in discard.iter().enumerate() {
            if *d {
                output.sampled_token_ids[idx].clear();
            }
        }

        if let Some(mut logprobs_map) = logprobs_output {
            let logprobs_vec: Vec<Option<Vec<scratchy_core_common::LogprobsOutput>>> = output
                .req_ids
                .iter()
                .map(|rid| logprobs_map.remove(rid))
                .collect();
            output.logprobs = Some(logprobs_vec);
        }

        Ok(output)
    }

    /// Get a pointer to pinned host memory for packing sampling parameters.
    fn get_sampling_packed_ptr_s(host_staging: &Option<HostStaging>, min_bytes: usize) -> *mut u8 {
        if let Some(stg) = host_staging {
            debug_assert!(min_bytes <= stg.sampling_packed.capacity_bytes());
            stg.sampling_packed.ptr()
        } else {
            vec![0u8; min_bytes].leak().as_mut_ptr()
        }
    }

    /// Greedy rejection sampling for speculative decoding.
    ///
    /// When any request has draft tokens, the forward pass produces logits for
    /// ALL token positions `[total_tokens, vocab_size]`. This method:
    /// 1. Argmax all logit rows on GPU → `[total_tokens]` target token IDs
    /// 2. D2H sync all target IDs
    /// 3. CPU greedy rejection: for each spec decode request, compare target
    ///    argmax at each draft position against the draft token. Accept the
    ///    prefix of matches + 1 bonus/recovered token.
    /// 4. Build multi-token ModelRunnerOutput and commit_step.
    ///
    /// Matches Python vLLM's `rejection_sample()` with greedy (no draft probs).
    #[allow(clippy::too_many_arguments)]
    fn spec_decode_greedy_sample(
        logits: GpuTensor,
        total_tokens: usize,
        prepared: PreparedInputs,
        device: &mut GpuDevice,
        _host_staging: &Option<HostStaging>,
        input_batch: &mut InputBatch,
        token_buffers: &mut HashMap<String, Vec<u32>>,
    ) -> ExecutorResult<ModelRunnerOutput> {
        // Sync compute stream to catch any prior kernel errors.
        unsafe { crate::driver::stream_synchronize(device.compute_stream) }.map_err(|e| {
            ExecutorError::WorkerExecution(format!("spec decode pre-argmax sync: {e}"))
        })?;

        // Validate logits shape matches total_tokens.
        let logits_dim0 = logits.dim(0);
        assert_eq!(
            logits_dim0,
            total_tokens,
            "spec_decode_greedy_sample: logits dim0 ({}) != total_tokens ({}), num_reqs={}, \
             q_lens={:?}, spec_counts={:?}",
            logits_dim0,
            total_tokens,
            prepared.req_inputs.len(),
            prepared.attn_meta.q_lens,
            prepared
                .req_inputs
                .iter()
                .map(|r| r.spec_token_ids.len())
                .collect::<Vec<_>>(),
        );

        // 1. Argmax all logit rows on GPU.
        let token_ids_owned = unsafe {
            crate::kernels::argmax_batched(logits, &mut device.caching, device.compute_stream)
        };
        let token_ids_gpu = token_ids_owned.as_gpu_tensor();

        // Sync compute stream to catch argmax kernel errors before D2H.
        unsafe { crate::driver::stream_synchronize(device.compute_stream) }.map_err(|e| {
            ExecutorError::WorkerExecution(format!("spec decode post-argmax sync: {e}"))
        })?;

        // 2. D2H all target argmax IDs.
        // Bypass pinned staging — total_tokens (with drafts) may exceed staging capacity.
        let all_target_ids =
            Self::d2h_token_ids_sync(None, 0, &token_ids_gpu, total_tokens, device)?;

        // 3. Greedy rejection sampling per request.
        let num_reqs = prepared.req_inputs.len();
        let mut req_ids = Vec::with_capacity(num_reqs);
        let mut sampled_token_ids = Vec::with_capacity(num_reqs);
        let mut req_id_to_index = HashMap::with_capacity(num_reqs);

        let mut flat_offset = 0usize;
        for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
            let req_id = &req_slice.req_id;
            let n_tokens = req_slice.token_count; // 1 + num_drafts for spec, 1 for normal
            let target_ids = &all_target_ids[flat_offset..flat_offset + n_tokens];

            let rejection = ::scratchy_serving_engine::spec_decode::greedy_rejection_sample(
                target_ids,
                &req_slice.spec_token_ids,
            );
            let accepted_tokens = rejection.accepted_tokens;

            // 4. Commit step with accepted tokens.
            input_batch.commit_step(
                req_id,
                &accepted_tokens,
                req_slice.token_count,
                !req_slice.spec_token_ids.is_empty(),
            );
            if let Some(buf) = token_buffers.get_mut(req_id) {
                buf.extend_from_slice(&accepted_tokens);
            }

            req_id_to_index.insert(req_id.clone(), req_idx);
            req_ids.push(req_id.clone());
            sampled_token_ids.push(accepted_tokens);
            flat_offset += n_tokens;
        }

        input_batch.reclaim_buffers(prepared);

        Ok(ModelRunnerOutput {
            req_ids,
            req_id_to_index,
            sampled_token_ids,
            logprobs: None,
            prompt_logprobs_dict: HashMap::new(),
            draft_token_ids: None,
            draft_seed_inputs: None,
            pooler_output: None,
            // ⛔ EMPTY BY CONSTRUCTION, like metal: every request here is written at its own
            // `slot_mapping` position, so slot == token and the scheduler's own prefix arithmetic is
            // already right. `kv_extent` exists for a backend whose batched write appends every row of a
            // batch at ONE slot (the spyre pool), where token `t` stops being at slot `t`.
            //
            // ⛔⛔ AND THIS FILE, LIKE `gpu_worker.rs`, HAS NOT COMPILED SINCE THE FIELD WAS ADDED in
            // cc9d77c1 — four commits of `--features superdsc` checks while two other backends were
            // broken. There is no CUDA device on the machine this was written on, which is exactly why
            // the compile has to be run rather than reasoned about.
            kv_extent: HashMap::new(),
            // As metal: slot == token here, so there is no pool-wide reach to report.
            kv_pool_reach: None,
            d2h_resolver: None,
        })
    }

    /// D2H + sync + commit_step helper (static to avoid borrow conflicts).
    #[allow(clippy::too_many_arguments)]
    fn finalize_d2h_and_commit(
        token_ids_gpu: &GpuTensor,
        num_reqs: usize,
        prepared: PreparedInputs,
        device: &mut GpuDevice,
        host_staging: &Option<HostStaging>,
        buf_idx: usize,
        input_batch: &mut InputBatch,
        token_buffers: &mut HashMap<String, Vec<u32>>,
        prompt_lengths: &HashMap<String, usize>,
    ) -> ExecutorResult<ModelRunnerOutput> {
        let host_ids = Self::d2h_token_ids_sync(
            host_staging.as_ref(),
            buf_idx,
            token_ids_gpu,
            num_reqs,
            device,
        )?;
        // Build discard mask: intermediate prefill chunks (seq_len after
        // this step < prompt_len) should not have their sampled tokens
        // appended to token_buffers. Matches Python's discard_request_mask.
        let mut discard = vec![false; num_reqs];
        for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
            if req_slice.token_count > 1 {
                // This is a prefill request. Check if it finishes the prompt.
                let prompt_len = prompt_lengths.get(&req_slice.req_id).copied().unwrap_or(0);
                let tokens_in_pool = input_batch.tokens_in_pool_for(&req_slice.req_id);
                let seq_len_after = tokens_in_pool + req_slice.token_count;
                if seq_len_after < prompt_len {
                    discard[req_idx] = true;
                }
            }
        }

        for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
            let tok = host_ids[req_idx];
            input_batch.commit_step(
                &req_slice.req_id,
                &[tok],
                req_slice.token_count,
                !req_slice.spec_token_ids.is_empty(),
            );
            if !discard[req_idx]
                && let Some(buf) = token_buffers.get_mut(&req_slice.req_id)
            {
                buf.push(tok);
            }
        }

        // Build output, clearing tokens for discarded (intermediate prefill) requests.
        let req_ids: Vec<String> = prepared
            .req_inputs
            .iter()
            .map(|r| r.req_id.clone())
            .collect();
        input_batch.reclaim_buffers(prepared);
        let mut output = ModelRunnerOutput::from_ordered(req_ids, host_ids);
        for (idx, d) in discard.iter().enumerate() {
            if *d {
                output.sampled_token_ids[idx].clear();
            }
        }
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Worker trait implementation
// ---------------------------------------------------------------------------

// `CudaWorker` and `MetalWorker` are separate structs (cfg-mutex'd — each
// compile sees exactly one) implementing the shared `Worker` trait. Their
// method bodies genuinely differ (load_model dispatcher, initialize_cache
// shape, execute_model body, sampling, determine_available_memory profile
// path, compile_or_warm_up_model) and the cuda bodies call cuda-only helpers
// that don't compile under metal, so each worker owns its own impl rather than
// sharing one struct with per-method `#[cfg]` arms.

// Phase 5.2a stub: CUDA gets the `SpecDecodeBackend` trait surface so
// downstream architecture can hold the worker behind `dyn
// SpecDecodeBackend` from day one. Bodies stay `NotImplemented` —
// phase 5.2b ports them (cudaMalloc + cuda forward +
// `crate::kernels::argmax_batched`, lifting the existing
// `spec_decode_greedy_sample` logic).
#[cfg(feature = "cuda")]
impl ::scratchy_serving_engine::spec_decode::SpecDecodeBackend for CudaWorker {
    fn forward_argmax_blocking(
        &mut self,
        _model: ::scratchy_serving_engine::spec_decode::ModelHandle,
        _kv_pool: ::scratchy_serving_engine::spec_decode::KvPoolHandle,
        _req: &::scratchy_serving_engine::spec_decode::ForwardArgmaxRequest<'_>,
    ) -> Result<Vec<u32>, ::scratchy_serving_engine::spec_decode::BackendError> {
        Err(
            ::scratchy_serving_engine::spec_decode::BackendError::NotImplemented(
                "cuda: forward_argmax_blocking (phase 5.2b)",
            ),
        )
    }

    fn load_secondary_model(
        &mut self,
        _path: &::std::path::Path,
        _dtype: Option<&str>,
    ) -> Result<
        ::scratchy_serving_engine::spec_decode::ModelHandle,
        ::scratchy_serving_engine::spec_decode::BackendError,
    > {
        Err(
            ::scratchy_serving_engine::spec_decode::BackendError::NotImplemented(
                "cuda: load_secondary_model (phase 5.2b)",
            ),
        )
    }

    fn allocate_kv_pool(
        &mut self,
        _model: ::scratchy_serving_engine::spec_decode::ModelHandle,
        _num_blocks: usize,
    ) -> Result<
        ::scratchy_serving_engine::spec_decode::KvPoolHandle,
        ::scratchy_serving_engine::spec_decode::BackendError,
    > {
        Err(
            ::scratchy_serving_engine::spec_decode::BackendError::NotImplemented(
                "cuda: allocate_kv_pool (phase 5.2b)",
            ),
        )
    }

    fn kv_per_block_bytes(
        &self,
        _model: ::scratchy_serving_engine::spec_decode::ModelHandle,
    ) -> Result<usize, ::scratchy_serving_engine::spec_decode::BackendError> {
        Err(
            ::scratchy_serving_engine::spec_decode::BackendError::NotImplemented(
                "cuda: kv_per_block_bytes (phase 5.2b)",
            ),
        )
    }
}

#[cfg(feature = "cuda")]
impl Worker for CudaWorker {
    /// CUDA now implements the grouped SWA KV path: the executor builds
    /// per-KV-group block tables + slot mappings (`build_attention_tensors`),
    /// threads them through `ForwardCtx::{block_tables, slot_mappings}`, and
    /// every attention op / cache write resolves its group via
    /// `block_table_for(layer)` / `slot_mapping_for(layer)`; the pool allocates
    /// `group_size` shared physical tensors (`KvCachePool::new_grouped`). So
    /// the backend is hybrid-capable — `compute_kv_blocks` still decides
    /// per-arch (only page-differentiated arches like gemma4 get a
    /// `HybridKvConfig`; uniform arches fall back to the single pool, where the
    /// group vecs are length 1 and the ops use the scalar block table —
    /// byte-identical to the pre-hybrid path).
    ///
    /// Excluded: FP8 KV. The grouped pool (`new_grouped`) is bf16-only, and
    /// returning `true` here for FP8 would put the scheduler on the hybrid
    /// allocator (disjoint per-group ids) while the pool stayed uniform — a
    /// mismatch. FP8 KV keeps the uniform single pool (it also skips graph
    /// capture), so gate it off.
    fn supports_hybrid_swa_kv(&self) -> bool {
        !self.kv_cache_is_fp8
    }

    fn init_device(&mut self) -> ExecutorResult<()> {
        let device = GpuDevice::new(self.config.device_id)
            .map_err(|e| ExecutorError::WorkerInit(format!("GpuDevice init failed: {e}")))?;
        self.device = Some(device);
        Ok(())
    }

    fn load_model(&mut self) -> ExecutorResult<()> {
        let t0 = std::time::Instant::now();

        // Ensure CUDA context is current on this thread.
        {
            let device = self
                .device
                .as_ref()
                .ok_or_else(|| ExecutorError::WorkerInit("device not initialized".into()))?;
            unsafe { driver::ctx_set_current(device.ctx) }
                .map_err(|e| ExecutorError::WorkerInit(format!("ctx_set_current: {e}")))?;
        }

        // 1. Resolve model directory (or GGUF file path).
        let model_dir = resolve_model_path(
            &self.config.model_path,
            self.config.hf_token.as_deref(),
            self.config.gguf_file.as_deref(),
            None,
        )?;
        info!("ScratchyWorker: loading model from {}", model_dir.display());

        let device = self
            .device
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerInit("device not initialized".into()))?;

        // Tokenizer on background thread. Sources tried, in order:
        //   1. `tokenizer.json` in the dir (parent dir if model_dir
        //      is a bare `.gguf` file path).
        //   2. GGUF metadata `tokenizer.ggml.*` if a `.gguf` is found
        //      (covers HF GGUF repos that don't ship a sibling
        //      tokenizer.json — unsloth, bartowski, etc.).
        let tok_search_dir = if model_dir.is_file() {
            model_dir.parent().unwrap_or(&model_dir).to_path_buf()
        } else {
            model_dir.clone()
        };
        let tok_gguf_path: Option<PathBuf> =
            if model_dir.is_file() && model_dir.extension().is_some_and(|e| e == "gguf") {
                Some(model_dir.clone())
            } else {
                std::fs::read_dir(&tok_search_dir).ok().and_then(|mut it| {
                    it.find_map(|entry| {
                        let p = entry.ok()?.path();
                        (p.extension().is_some_and(|e| e == "gguf")).then_some(p)
                    })
                })
            };
        let tokenizer_handle = std::thread::spawn(move || {
            let json_path = tok_search_dir.join("tokenizer.json");
            if json_path.exists()
                && let Ok(t) = tokenizers::Tokenizer::from_file(&json_path)
            {
                return Some(t);
            }
            if let Some(gguf_path) = tok_gguf_path
                && let Ok(gguf) = scratchy_quantizations::gguf::GgufFile::open(&gguf_path)
            {
                match scratchy_quantizations::gguf::gguf_tokenizer(&gguf) {
                    Ok(Some(t)) => return Some(t),
                    Ok(None) => {}
                    Err(e) => tracing::warn!("gguf_tokenizer failed: {e}"),
                }
            }
            None
        });

        // 2. Parse config: GGUF metadata for `.gguf` files, otherwise
        //    `config.json` in the model dir. GGUF support lives in
        //    `scratchy-quantizations::gguf` — scratchy-core-model has no GGUF-awareness.
        let hf_config = if model_dir.is_file() && model_dir.extension().is_some_and(|e| e == "gguf")
        {
            let gguf = scratchy_quantizations::gguf::GgufFile::open(&model_dir)
                .map_err(|e| ExecutorError::WorkerInit(format!("GGUF parse failed: {e}")))?;
            scratchy_quantizations::gguf::gguf_model_config(&gguf)
                .map_err(|e| ExecutorError::WorkerInit(format!("GGUF config: {e}")))?
        } else {
            HfModelConfig::from_path(&model_dir)
                .map_err(|e| ExecutorError::WorkerInit(format!("config parse failed: {e}")))?
        };

        // 3. Resolve dtype.
        let dtype = match self.config.dtype.as_str() {
            "f16" | "float16" => GpuDType::F16,
            "bf16" | "bfloat16" => GpuDType::BF16,
            "f32" | "float32" => GpuDType::F32,
            _ => match hf_config.torch_dtype.as_deref() {
                Some("bfloat16") => GpuDType::BF16,
                Some("float16") => GpuDType::F16,
                _ => GpuDType::BF16,
            },
        };
        info!("ScratchyWorker: using dtype {:?}", dtype);

        // 4. Look up architecture.
        let arch = hf_config.architectures.first().cloned().unwrap_or_default();
        info!("ScratchyWorker: architecture = {arch}");

        // 5. Parse weight files. Safetensors stays CPU-mmap'd (lazy
        // upload via take()); GGUF is eagerly uploaded via the
        // inventory-registered loader. `from_path` dispatches.
        let tp_world = self.config.tp_world_size.max(1);
        let tp_rank = self.config.tp_rank;
        let stream = device.compute_stream;
        let t_parse = std::time::Instant::now();
        let mut weights = unsafe {
            let device_mut = self.device.as_mut().unwrap();
            crate::weights::from_path(
                &model_dir,
                stream,
                &mut device_mut.caching,
                dtype,
                tp_rank,
                tp_world,
            )
        }
        .map_err(|e| ExecutorError::WorkerInit(format!("weight load failed: {e}")))?;
        let t_parse = t_parse.elapsed();
        info!("ScratchyWorker: parsed {} weight tensors", weights.len());
        let uses_ggml = weights.is_gguf();
        let device = self.device.as_ref().unwrap();

        // Set target dtype so F32 weights are cast to model dtype on load.
        // Matches Python vLLM where model parameters are initialized with
        // torch_dtype and PyTorch auto-casts during weight_loader copy.
        weights.set_target_dtype(dtype);

        // 5b. Merge LoRA adapter weights (CPU-side, before H2D copy).
        if let Some(ref adapter_path) = self.config.lora_adapter {
            let adapter_dir = self.resolve_adapter_path(adapter_path)?;
            let merged = weights
                .merge_lora(&adapter_dir)
                .map_err(|e| ExecutorError::WorkerInit(format!("LoRA merge: {e}")))?;
            info!("ScratchyWorker: merged {merged} LoRA weight tensors");
        }

        // 6. Detect quantization config.
        let qconfig = quant::detect_quant_config(&model_dir)
            .map_err(|e| ExecutorError::WorkerInit(format!("quant config detection: {e}")))?;
        if qconfig.is_quantized() {
            info!("ScratchyWorker: detected quantization: {:?}", qconfig);
            if self.config.lora_adapter.is_some() {
                return Err(ExecutorError::WorkerInit(
                    "LoRA with quantized models requires Punica kernels (not yet implemented)"
                        .into(),
                ));
            }
        }

        // 6b. Start the background pre-stage pipeline. Worker threads fault
        // mmap pages and stage/cast tensors into pinned buffers concurrently
        // with model construction. Must be after set_target_dtype() and
        // merge_lora().
        let t_construct = std::time::Instant::now();
        weights.start_precast();

        // 7. Construct model. scratchy-forward-compiler is the sole model-construction
        //    path now; hand-written `crate::model::*` loaders + their
        //    PP/Qwen3Next-specific plumbing are gone. Pipeline parallelism is
        //    not supported until scratchy-forward-compiler grows native PP — fail fast
        //    rather than silently routing one rank into a stub.
        let tp_world = self.config.tp_world_size;
        let tp_rank = self.config.tp_rank;
        if self.config.pp_size > 1 {
            return Err(ExecutorError::WorkerInit(
                "pipeline parallelism is not supported on scratchy-forward-compiler forwards \
                 (the previous PP path lived in `crate::model::*` which \
                 has been removed). Run with `--pipeline-parallel-size 1`."
                    .into(),
            ));
        }

        // ── Hand off to scratchy-forward-compiler (all supported arches in one shot).
        //
        // `crate::try_load` walks every `#[forward]`-
        // registered arch; first whose `hf_arches` list contains
        // `arch.as_str()` wins. Auto-registered via
        // `inventory::submit!` at macro expansion — adding a new
        // arch to scratchy-models touches zero lines here.
        let scratchy_loaded: Option<CudaModel> = {
            let stream = device.compute_stream;
            // Thread a minimal HF-config view into scratchy so per-
            // variant `fingerprint_matches` can disambiguate
            // checkpoints that share tensor shapes but differ in
            // config-only fields (Phi-3-mini-4k vs Phi-3.5-mini-128k
            // — same weights, different `max_position_embeddings` +
            // `rope_scaling.type`). Scratchy owns the rest.
            // GGUF metadata's `<arch>.context_length` and rope_scaling
            // hash often disagree with the canonical HF config.json
            // (e.g. Qwen2.5-0.5B-Instruct GGUF reports 8192, JSON
            // reports 32768; unsloth Llama-3 GGUFs omit rope_scaling
            // entirely so we infer it). Treat those config-disambiguation
            // hints as permissive (None) for GGUF sources — the
            // fingerprint's positive shape + suffix checks plus
            // `is_gguf_gate` are already disjoint enough; the
            // compile-time variant's baked rope/max_pos values stay
            // authoritative.
            let suppress_hf_hints = weights.is_gguf();
            let hf_fp = scratchy_forward_compiler::HfFingerprint {
                rope_scaling_type: if suppress_hf_hints {
                    None
                } else {
                    hf_config
                        .extra
                        .get("rope_scaling")
                        .and_then(|rs| rs.get("rope_type").or_else(|| rs.get("type")))
                        .and_then(|v| v.as_str())
                },
                rope_scaling_hash: if suppress_hf_hints {
                    None
                } else {
                    hf_config
                        .extra
                        .get("rope_scaling")
                        .map(scratchy_forward_compiler::hash_json_value)
                },
                // Suppressed for GGUF for the same reason as the two above: the
                // metadata's rope base frequency routinely disagrees with the
                // canonical `config.json`, so the variant's baked value stays
                // authoritative rather than being second-guessed by a value we
                // don't trust.
                rope_theta: if suppress_hf_hints {
                    None
                } else {
                    hf_config.rope_theta
                },
            };
            // Runtime `max_model_len` — matches the serve-level
            // resolution (CLI `--max-model-len` ∨ HF
            // `max_position_embeddings` ∨ 4096) so Phi-3 LongRoPE's
            // `use_long_rope = max_model_len > original_max_pos`
            // flag agrees with Python vLLM at init time.
            let max_model_len = self
                .config
                .max_model_len
                .or(hf_config.max_position_embeddings)
                .unwrap_or(4096);
            // Pass the runtime (tp_world_size, tp_rank) so try_load
            // picks the matching `(arch, tp)` registration and the
            // per-(model, tp) emitted `Weights::load` body shards
            // weights by `tp_rank`. tp>1 is now end-to-end wired:
            // codegen's shard-kind dispatch (52615e881) routes load
            // calls through `_sharded` helpers; lowering injects
            // AllReduce after vocab-parallel Embed (aefa1de37) and
            // AllGather after lm_head Gemm (8a3ea25bc). At tp=1 every
            // emitted call site is byte-equivalent to before TP
            // landed — `_sharded` variants degrade to the unsharded
            // helpers when world == 1.
            let scratchy_tp = u8::try_from(tp_world).unwrap_or(1);
            let scratchy_rank = u8::try_from(tp_rank).unwrap_or(0);
            if let Some(scratchy_weights) = crate::try_load(
                &mut weights,
                stream,
                arch.as_str(),
                scratchy_tp,
                scratchy_rank,
                max_model_len,
                hf_fp,
            )
            .map_err(|e| {
                ExecutorError::WorkerInit(format!("scratchy-forward-compiler load: {e}"))
            })? {
                // Scratchy owns rotary construction: the emitted
                // `Weights::load` built the RotaryCache (plus any
                // dual-rotary arch-local variant) inline from the
                // manifest's bounds/scalars/rope_scaling. Nothing
                // for the executor to do.
                info!(
                    "ScratchyWorker: loaded {} via scratchy-forward-compiler ({})",
                    arch,
                    scratchy_weights.arch_name(),
                );
                // Probe for a sibling `MultimodalForward` registration.
                // Same `(arch_hint, tp_world_size)` filter as the text
                // try_load above; returns `Ok(None)` for text-only
                // arches and for MM-capable arches whose live
                // checkpoint has no `visual.*` tensors. The handle
                // rides on `ScratchyModel.mm` and is consumed at
                // forward time when the batch carries `mm_data`.
                let mm = crate::try_load_mm(
                    &mut weights,
                    stream,
                    arch.as_str(),
                    scratchy_tp,
                    scratchy_rank,
                    max_model_len,
                    hf_fp,
                )
                .map_err(|e| {
                    ExecutorError::WorkerInit(format!("scratchy-forward-compiler MM load: {e}"))
                })?;
                if mm.is_some() {
                    info!(
                        "ScratchyWorker: loaded {} vision encoder via scratchy-forward-compiler",
                        arch
                    );
                }
                Some(CudaModel::Scratchy(Box::new(ScratchyModel {
                    weights: scratchy_weights,
                    mm,
                    #[cfg(feature = "nccl")]
                    tp_group: None,
                    tp_world_size: self.config.tp_world_size,
                })))
            } else {
                None
            }
        };
        let model = scratchy_loaded.ok_or_else(|| {
            ExecutorError::WorkerInit(format!(
                "no scratchy-forward-compiler variant matched arch=`{arch}` at tp={tp_world}. \
                 Hand-written CUDA model forwards have been removed; scratchy-forward-compiler \
                 is the sole model-load path. Rebuild with the matching \
                 `scratchy-models/<stem>` feature (or \
                 `all` to compile every variant), and confirm \
                 the arch's `configs/quantizations.json` lists this checkpoint's \
                 quantization scheme."
            ))
        })?;

        let t_construct = t_construct.elapsed();

        // Sync to ensure all async H2D weight copies are complete.
        let t_sync = std::time::Instant::now();
        unsafe { driver::stream_synchronize(device.compute_stream) }
            .map_err(|e| ExecutorError::WorkerInit(format!("weight sync: {e}")))?;
        let t_sync = t_sync.elapsed();

        // Collect GPU weight allocation pointers for sleep/wake lifecycle.
        self.weight_gpu_allocs = weights.take_gpu_allocs();
        let t_drop = std::time::Instant::now();
        drop(weights); // CPU mmaps freed, GPU memory owned by model layers
        info!(
            "ScratchyWorker: load phases — parse {:.2}s, construct+upload {:.2}s, \
             sync {:.3}s, teardown {:.3}s",
            t_parse.as_secs_f64(),
            t_construct.as_secs_f64(),
            t_sync.as_secs_f64(),
            t_drop.elapsed().as_secs_f64(),
        );

        self.model_dtype = dtype;
        self.resolved_architecture = Some(arch);
        self.model = Some(model);
        self.uses_ggml = uses_ggml;
        self.model_dir = Some(model_dir.clone());
        self.hf_config = Some(hf_config);

        // Resolve pooling strategy.
        self.pooling_strategy = scratchy_core_model::embedding::resolve_pooling_strategy(
            &self.config.pooling_strategy,
            &model_dir,
        );

        // Collect tokenizer.
        if let Ok(Some(tok)) = tokenizer_handle.join() {
            self.preloaded_tokenizer = Some(tok);
        }

        // Initialize logits processor pipeline (grammar handled separately).
        let vocab_size = self.model.as_ref().unwrap().vocab_size();
        let processors: Vec<Box<dyn crate::logits_processor::LogitsProcessor>> = vec![
            Box::new(MinTokensProcessor::new()),
            Box::new(LogitBiasProcessor::new()),
            Box::new(PenaltiesProcessor::new(vocab_size)),
            Box::new(BadWordsProcessor::new()),
        ];
        self.logits_pipeline = Some(LogitsProcessorPipeline::new(processors));

        // Reinitialize SealPadProcessor with real EOS token IDs from model config.
        if let Some(ref hf_config) = self.hf_config {
            let eos_token_ids: Vec<u32> = hf_config
                .extra
                .get("eos_token_id")
                .map(|v| {
                    if let Some(id) = v.as_u64() {
                        vec![id as u32]
                    } else if let Some(arr) = v.as_array() {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|id| id as u32))
                            .collect()
                    } else {
                        vec![]
                    }
                })
                .unwrap_or_default();
            self.seal_pad_processor =
                SealPadProcessor::new(eos_token_ids, 0, self.config.block_size);
        }

        // Pre-allocate the FA3 scheduler-metadata buffer on Hopper
        // (sm_90+) before `determine_available_memory` runs its profile.
        // The 4 KB allocation lands as non-torch persistent memory and
        // is correctly accounted for in the KV-cache budget. On non-
        // Hopper devices the buffer is never read, so don't allocate.
        #[cfg(fa3_built)]
        if let Some(ref dev) = self.device
            && dev.sm_version >= 90
        {
            unsafe {
                crate::flash_attn_3::fa3_init_metadata();
            }
        }

        info!(
            "ScratchyWorker: model loaded in {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        Ok(())
    }

    fn initialize_cache(
        &mut self,
        num_gpu_blocks: usize,
        _num_cpu_blocks: usize,
    ) -> ExecutorResult<()> {
        if let Some(ref dev) = self.device {
            unsafe { driver::ctx_set_current(dev.ctx) }
                .map_err(|e| ExecutorError::WorkerInit(format!("ctx_set_current: {e}")))?;
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerInit("model not loaded".into()))?;

        // Resolve KV cache dtype: FP8 E4M3 when configured, otherwise model dtype.
        let kv_dtype = if self.kv_cache_is_fp8 {
            GpuDType::Fp8E4m3
        } else {
            self.model_dtype
        };

        // Capture the model-derived geometry as owned locals so the `&model`
        // borrow ends before we assign `self.kv_cache` below.
        let num_layers = model.num_layers();
        let num_kv_heads = model.num_kv_heads();
        let head_dim = model.head_dim();
        // Per-layer kv_heads*head_dim (Gemma-4: sliding 8×256=2048 vs global
        // 1×512/2×512). `None` on uniform arches.
        let per_layer_kv_elems = model.per_layer_kv_token_elems();
        let block_size = self.config.block_size;
        // Per-sequence block-table row capacity (runtime-derived). cuda drives
        // its block tables off `Weights::max_blocks_per_seq()`; this pool field
        // is sized the metal way so the type stays consistent.
        let max_bps = self
            .config
            .max_model_len
            .map(|l| l.div_ceil(block_size.max(1)))
            .unwrap_or(num_gpu_blocks)
            .min(num_gpu_blocks)
            .max(1);

        // vLLM group-shared hybrid SWA layout (gemma4): infer the sliding class
        // from `per_layer_kv_token_elems` (bigger page = sliding) and group.
        // Same trigger the scheduler's `hybrid_kv` uses, so group counts agree.
        // Only when the arch is page-differentiated AND KV is not FP8 (fp8 KV
        // keeps the uniform pool — it also skips graph capture). Gated on
        // `supports_hybrid_swa_kv()` — the SAME switch that puts the scheduler
        // on the per-group disjoint-id allocator. The grouped pool shares
        // physical tensors across same-position layers assuming per-group
        // disjoint block ids, so it is ONLY correct when the scheduler is also
        // hybrid; this keeps the two in lockstep (uniform pool until the flip).
        let hybrid_layout: Option<scratchy_core_config::HybridKvLayout> =
            if self.kv_cache_is_fp8 || !self.supports_hybrid_swa_kv() {
                None
            } else {
                per_layer_kv_elems.as_ref().and_then(|elems| {
                    let max_e = *elems.iter().max()?;
                    let geom: Vec<scratchy_core_config::LayerKvGeometry> = elems
                        .iter()
                        .map(|&e| scratchy_core_config::LayerKvGeometry {
                            is_sliding: e == max_e,
                            // Page proxy: head_size 1 keeps the per-class page RATIO
                            // (only the ratio drives grouping + block_size scaling).
                            num_kv_heads: e,
                            head_size: 1,
                            head_size_v: None,
                            sliding_window: if e == max_e { Some(1) } else { None },
                        })
                        .collect();
                    scratchy_core_config::compute_hybrid_kv_layout(
                        &geom,
                        block_size,
                        usize::MAX / 2,
                        kv_dtype.size_bytes(),
                    )
                })
            };

        let alloc = |bytes: usize| -> anyhow::Result<crate::PoolMem> {
            let ptr = unsafe { driver::mem_alloc(bytes) }?;
            Ok(unsafe { crate::raw_cuda(ptr, bytes) })
        };

        let pool = if let Some(layout) = hybrid_layout {
            // Grouped path: `group_size` shared physical tensors, per-layer
            // page-unified block sizes, per-group block tables.
            let elems = per_layer_kv_elems
                .clone()
                .expect("hybrid layout implies per_layer_kv_token_elems");
            let mut p = unsafe {
                KvCachePool::new_grouped(
                    num_layers,
                    num_gpu_blocks,
                    block_size,
                    num_kv_heads,
                    head_dim,
                    max_bps,
                    elems,
                    layout.layer_block_size.clone(),
                    layout.layer_to_tensor.clone(),
                    kv_dtype,
                    alloc,
                )
            }
            .map_err(|e| ExecutorError::WorkerInit(format!("KvCachePool(grouped): {e}")))?;
            p.set_kv_group_layout(layout.num_groups(), layout.layer_to_group_u32());
            info!(
                "ScratchyWorker(cuda): hybrid SWA KV — {} groups, {} physical tensors, {num_gpu_blocks} blocks",
                layout.num_groups(),
                layout.group_size
            );
            p
        } else {
            // Uniform pool: one physical tensor per layer, sized by geometry.
            let per_layer_block_elems = per_layer_kv_elems
                .map(|v| v.iter().map(|&e| e * block_size).collect::<Vec<usize>>());
            unsafe {
                KvCachePool::new(
                    num_layers,
                    num_gpu_blocks,
                    block_size,
                    num_kv_heads,
                    head_dim,
                    max_bps,
                    per_layer_block_elems,
                    kv_dtype,
                    alloc,
                    |scale_mem| {
                        // FP8 scales initialize to 1.0 (cuda H2D, null stream).
                        let one: f32 = 1.0;
                        driver::memcpy_htod_async(
                            scale_mem.ptr(),
                            &one as *const f32 as *const u8,
                            4,
                            std::ptr::null_mut(),
                        )?;
                        Ok(())
                    },
                )
            }
            .map_err(|e| ExecutorError::WorkerInit(format!("KvCachePool: {e}")))?
        };

        self.kv_cache = Some(pool);

        // Gated-DeltaNet (Qwen3.5 / Qwen3-Next) recurrent-state pool —
        // the non-paged sibling of the KV cache. Only hybrid arches
        // report a `gdn_runtime_config` (macro-emitted from the unrolled
        // IR's per-layer GDN dispatch); every other arch leaves
        // `self.gdn_state` / `self.gdn_slot_allocator` `None`. One
        // recurrent-state slot per concurrently-resident sequence
        // (`max_num_seqs`), matching the scheduler's `max_num_running_reqs`
        // so the slot allocator never exhausts. Mirrors the metal arm.
        if let Some(gdn_cfg) = model.gdn_runtime_config() {
            let num_slots = self.config.max_num_seqs.max(1);
            let num_layers = model.num_layers();
            let t_gdn = std::time::Instant::now();
            let gdn_pool = unsafe {
                crate::gdn_state::GdnStatePool::new(
                    num_layers,
                    &gdn_cfg.linear_layers,
                    num_slots,
                    gdn_cfg.conv_dim as usize,
                    gdn_cfg.conv_kernel as usize,
                    gdn_cfg.num_v_heads as usize,
                    gdn_cfg.head_v_dim as usize,
                    gdn_cfg.head_k_dim as usize,
                    |bytes| {
                        // f32 conv/ssm state, GPU-resident across forwards
                        // (no CPU touches). Same alloc path as the KV pool
                        // above — `driver::mem_alloc` + `raw_cuda`
                        // RAII wrapper auto-frees on drop.
                        let ptr = driver::mem_alloc(bytes)?;
                        Ok(crate::raw_cuda(ptr, bytes))
                    },
                )
            }
            .map_err(|e| ExecutorError::WorkerInit(format!("GdnStatePool: {e}")))?;
            info!(
                "ScratchyWorker(cuda): GDN state pool — {} linear / {} layers x {} slots in {:?}",
                gdn_cfg.num_linear_layers(),
                num_layers,
                num_slots,
                t_gdn.elapsed(),
            );
            self.gdn_state = Some(gdn_pool);
            self.gdn_slot_allocator = Some(GdnSlotAllocator::new(num_slots));
        }

        Ok(())
    }

    fn determine_available_memory(&mut self) -> ExecutorResult<usize> {
        if let Some(ref dev) = self.device {
            unsafe { driver::ctx_set_current(dev.ctx) }
                .map_err(|e| ExecutorError::WorkerInit(format!("ctx_set_current: {e}")))?;
        }

        // Skip profiling forward for models where the dummy forward is
        // incompatible or too expensive:
        // - GGML: flash attention triggers illegal memory access
        // - MoE: fused path uses ~55 MB/layer (OK), but profiling with
        //   max_num_batched_tokens can still OOM on the attention side
        // - Encoders: pooling-only, no logit head to drive the profile
        // - Hybrid (GDN: Qwen3.5 / Qwen3-Next): the profiling dummy
        //   forward would deref `ctx.gdn_state` / `gdn_state_indices`,
        //   which are `None` until `initialize_cache` runs. Skip and
        //   fixed-estimate for the first cut — matches the metal arm.
        let is_moe = self.model.as_ref().is_some_and(|m| m.is_moe());
        let is_encoder = self.model.as_ref().is_some_and(|m| m.is_scratchy_encoder());
        let is_hybrid = self
            .model
            .as_ref()
            .is_some_and(|m| m.gdn_runtime_config().is_some());
        if self.uses_ggml || is_moe || is_encoder || is_hybrid {
            let tag = if self.uses_ggml {
                "GGML"
            } else if is_moe {
                "MoE"
            } else if is_encoder {
                "encoder"
            } else {
                "hybrid-GDN"
            };
            info!(
                "ScratchyWorker: {tag} model — skipping activation profiling, using fixed estimate"
            );
            let (free, total) = cudarc::driver::result::mem_get_info()
                .map_err(|e| ExecutorError::WorkerInit(format!("cuMemGetInfo: {e}")))?;
            // Reserve room for the GDN recurrent-state pool. It's built in
            // `initialize_cache` (which runs AFTER this), so it isn't yet in
            // (total - free); fold it into the non-KV overhead so the engine
            // doesn't hand back KV blocks that leave no room for it. Zero
            // for non-hybrid arches (unwrap_or(0)).
            let gdn_reserve = self
                .model
                .as_ref()
                .and_then(|m| m.gdn_runtime_config())
                .map(|cfg| {
                    crate::gdn_state::GdnStatePool::<crate::PoolMem>::reserve_bytes(
                        cfg.num_linear_layers(),
                        self.config.max_num_seqs.max(1),
                        cfg.conv_dim as usize,
                        cfg.conv_kernel as usize,
                        cfg.num_v_heads as usize,
                        cfg.head_v_dim as usize,
                        cfg.head_k_dim as usize,
                    )
                })
                .unwrap_or(0);
            let weights_and_overhead = total.saturating_sub(free).saturating_add(gdn_reserve);
            let peak_activation_estimate = 512 * 1024 * 1024; // 512 MB conservative
            let utilization = self.config.gpu_memory_utilization;
            let available = compute_available_kv_bytes(
                total,
                weights_and_overhead,
                peak_activation_estimate,
                utilization,
            );
            info!(
                "Memory estimate: total={:.1} GiB, weights+overhead={:.1} GiB \
                 (incl. {:.1} MiB gdn_reserve), est_activations=512 MiB",
                total as f64 / 1_073_741_824.0,
                weights_and_overhead as f64 / 1_073_741_824.0,
                gdn_reserve as f64 / 1_048_576.0,
            );
            return Ok(available);
        }

        // Like Python vLLM: profile peak activation memory with a dummy forward
        // pass, then subtract it from available memory for KV cache sizing.
        //
        // Python uses `allocated_bytes.all.peak` (peak ACTIVE PyTorch allocations
        // during the profiling forward) + `non_torch_increase` (non-PyTorch CUDA
        // memory that persisted after the forward, e.g. cuBLAS workspace).
        //
        // We mirror this with:
        //   torch_peak     = caching.peak_active_bytes()  (peak live allocations)
        //   non_torch      = (total - free_after_trim) - caching.memory_reserved()
        //   peak_activations = torch_peak + non_torch
        let (free_before, _total) = cudarc::driver::result::mem_get_info()
            .map_err(|e| ExecutorError::WorkerInit(format!("cuMemGetInfo: {e}")))?;
        info!(
            "ScratchyWorker: {:.0} MB free VRAM",
            free_before as f64 / 1_048_576.0
        );

        let device = self
            .device
            .as_mut()
            .ok_or_else(|| ExecutorError::WorkerInit("device not initialized".into()))?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerInit("model not loaded".into()))?;

        // Run dummy forward with max_num_batched_tokens to measure peak activations.
        let prefill_tokens = self.config.max_num_batched_tokens;
        info!("Profiling activation memory with dummy forward ({prefill_tokens} tokens)...");

        // Allocate dummy inputs for the profiling forward pass.
        //
        // IDs and positions MUST be zero-initialized — the caching allocator
        // recycles freed blocks whose contents are undefined.  Garbage token
        // IDs cause out-of-bounds embedding lookups; garbage positions cause
        // out-of-bounds RoPE lookups.  Both segfault.  (Matches Python vLLM
        // which uses torch.zeros for its profiling dummy tensors.)
        let dummy_ids = device.alloc_gpu_tensor_zeroed(&[prefill_tokens], GpuDType::U32);
        let dummy_pos = device.alloc_gpu_tensor_zeroed(&[prefill_tokens], GpuDType::U32);

        // Slot mapping: slot[i] = i (sequential)
        let slot_data: Vec<i64> = (0..prefill_tokens as i64).collect();
        let dummy_slots =
            device.alloc_gpu_tensor_from_host(&[prefill_tokens], GpuDType::I64, unsafe {
                std::slice::from_raw_parts(slot_data.as_ptr() as *const u8, prefill_tokens * 8)
            });

        let cu_q: Vec<u32> = vec![0, prefill_tokens as u32];
        let gpu_cu_q = device.alloc_gpu_tensor_from_host(&[2], GpuDType::U32, unsafe {
            std::slice::from_raw_parts(cu_q.as_ptr() as *const u8, cu_q.len() * 4)
        });

        let seqused_data: Vec<u32> = vec![prefill_tokens as u32];
        let dummy_seqused = device.alloc_gpu_tensor_from_host(&[1], GpuDType::U32, unsafe {
            std::slice::from_raw_parts(seqused_data.as_ptr() as *const u8, 4)
        });

        // Block table: [1, num_blocks_needed] — sequential block indices
        let num_blocks_needed = prefill_tokens.div_ceil(self.config.block_size);
        let bt_data: Vec<u32> = (0..num_blocks_needed as u32).collect();
        let dummy_bt =
            device.alloc_gpu_tensor_from_host(&[1, num_blocks_needed], GpuDType::U32, unsafe {
                std::slice::from_raw_parts(bt_data.as_ptr() as *const u8, num_blocks_needed * 4)
            });

        // Single dummy seq → last_token_indices = [prefill_tokens - 1]. Without
        // this the lm_head gather (commit 14e043524a) is bypassed and lm_head
        // runs at full M=prefill_tokens, allocating an [M, vocab] output which
        // for vocab=152064, prefill_tokens=65536 (bench-latency bs=32 il=2048
        // setting `max_num_batched_tokens = bs*il`) is 19 GiB and OOMs the
        // profile pass before any KV-cache budget is computed. The runtime
        // path always sets `last_token_indices` at prefill, so the profile
        // peak measured here matches the runtime peak.
        let lti_data: Vec<u32> = vec![prefill_tokens as u32 - 1];
        let dummy_lti = device.alloc_gpu_tensor_from_host(&[1], GpuDType::U32, unsafe {
            std::slice::from_raw_parts(lti_data.as_ptr() as *const u8, 4)
        });

        // Create a KV cache large enough for the profiling tokens.
        let dummy_kv = unsafe {
            crate::KvCachePool::new(
                model.num_layers(),
                num_blocks_needed,
                self.config.block_size,
                model.num_kv_heads(),
                model.head_dim(),
                // Profiling pool: cap at the blocks allocated here.
                num_blocks_needed.max(1),
                // Per-layer geometry (Gemma-4) so the profiling prefill writes
                // global K/V (2×512) at the correct per-layer stride.
                model.per_layer_kv_token_elems().map(|v| {
                    v.iter()
                        .map(|&e| e * self.config.block_size)
                        .collect::<Vec<usize>>()
                }),
                self.model_dtype,
                |bytes| {
                    let ptr = driver::mem_alloc(bytes)?;
                    Ok(crate::raw_cuda(ptr, bytes))
                },
                // Profiling pool is never FP8 → scale init never invoked.
                |_scale_mem| -> anyhow::Result<()> { Ok(()) },
            )
        }
        .map_err(|e| ExecutorError::WorkerInit(format!("dummy KvCachePool: {e}")))?;

        // Reset peak stats immediately before the profiling forward so we measure
        // only the allocations made during this forward pass.
        device.caching.reset_peak_stats();

        // Run the forward pass to warm up cuBLAS and measure peak memory.
        unsafe {
            let _ = model.forward(
                TensorView::from_raw(dummy_ids),
                TensorView::from_raw(dummy_pos),
                TensorView::from_raw(dummy_slots),
                TensorView::from_raw(gpu_cu_q),
                TensorView::from_raw(dummy_seqused),
                TensorView::from_raw(dummy_bt),
                prefill_tokens,
                prefill_tokens,
                &dummy_kv,
                device,
                Some(TensorView::from_raw(dummy_lti)),
                None,
                // Profiling forward: hybrid GDN pool isn't built yet
                // (it's sized AFTER profiling determines KV memory). The
                // memory profiler's hybrid-arch bypass routes hybrid
                // models to the fixed-estimate path so this dummy
                // forward is never reached for them.
                None,
                None, // hybrid SWA: profiling dummy is uniform
            );
            if let Err(e) = driver::stream_synchronize(device.compute_stream) {
                tracing::error!("Memory profiling forward failed: {e}");
            }
        }

        // Capture peak active bytes from caching allocator — mirrors Python's
        // `allocated_bytes.all.peak` (peak ACTIVE allocations, freed blocks not
        // counted).
        let torch_peak = device.caching.peak_active_bytes();

        // Free the dummy KV cache and all cached allocator blocks, then trim to
        // return segments to the driver so cuMemGetInfo reflects only permanent
        // allocations (cuBLAS workspace, NCCL, etc.).
        drop(dummy_kv);
        device.caching.trim();

        // non_torch_increase = memory permanently held outside the caching
        // allocator after the profiling forward (cuBLAS workspace, etc.).
        // After trim(), caching.memory_reserved() == 0 for segments that were
        // released.  We compare against free_before to catch anything that
        // persisted.
        let (free_after_trim, total_memory) = cudarc::driver::result::mem_get_info()
            .map_err(|e| ExecutorError::WorkerInit(format!("cuMemGetInfo: {e}")))?;
        // memory_reserved() = bytes still held by caching allocator after trim
        // (private pool, graph capture pool, etc.)
        let caching_reserved = device.caching.memory_reserved();
        // non_torch = new permanent non-caching-allocator memory since free_before
        // (e.g. cuBLAS workspace created during forward).  Can be negative if
        // something was freed; clamp to 0.
        let used_after_trim = total_memory.saturating_sub(free_after_trim);
        let used_before = total_memory.saturating_sub(free_before);
        let non_torch = used_after_trim
            .saturating_sub(used_before)
            .saturating_sub(caching_reserved);

        let peak_activation_bytes = torch_peak + non_torch;

        // Match Python vLLM's memory calculation exactly.
        //
        // Python's flow (gpu_worker.py + mem_utils.py):
        //   non_kv_cache = weights_memory + torch_peak + non_torch + 150 MiB
        //   available_kv_bytes = requested - non_kv_cache
        //
        // Our free_before is measured AFTER model load (before profile run).
        // So (total - free_before) = weights + persistent pre-existing overhead.
        let weights_and_overhead = total_memory.saturating_sub(free_before);

        let utilization = self.config.gpu_memory_utilization;
        let available_kv_bytes = compute_available_kv_bytes(
            total_memory,
            weights_and_overhead,
            peak_activation_bytes,
            utilization,
        );

        info!(
            "Memory profiling: total={:.1} GiB, weights+overhead={:.1} GiB, \
             torch_peak={:.1} GiB, non_torch={:.1} GiB, peak_activations={:.1} GiB",
            total_memory as f64 / 1_073_741_824.0,
            weights_and_overhead as f64 / 1_073_741_824.0,
            torch_peak as f64 / 1_073_741_824.0,
            non_torch as f64 / 1_073_741_824.0,
            peak_activation_bytes as f64 / 1_073_741_824.0,
        );
        info!(
            "Available KV cache memory: {:.1} GiB \
             (requested={:.1} GiB [total*{:.2}] - non_kv={:.1} GiB)",
            available_kv_bytes as f64 / 1_073_741_824.0,
            (total_memory as f64 * utilization) / 1_073_741_824.0,
            utilization,
            (weights_and_overhead + peak_activation_bytes + 150 * 1024 * 1024) as f64
                / 1_073_741_824.0,
        );

        // Return the direct KV cache bytes. compute_num_blocks must NOT
        // apply gpu_memory_utilization again — it's already baked in.
        Ok(available_kv_bytes)
    }

    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
        self.execute_model_inner(scheduler_output)
    }

    fn compile_or_warm_up_model(&mut self) -> ExecutorResult<()> {
        if self.config.enforce_eager {
            info!("ScratchyWorker: --enforce-eager set, skipping CUDA graph capture");
            return Ok(());
        }

        if self.uses_ggml {
            info!(
                "ScratchyWorker: GGML model — skipping CUDA graph capture (incompatible with graph capture)"
            );
            return Ok(());
        }

        if self.kv_cache_is_fp8 {
            info!(
                "ScratchyWorker: FP8 KV cache — skipping CUDA graph capture (variable scratch buffer sizes)"
            );
            return Ok(());
        }

        // Hybrid (Gated-DeltaNet) arches: the GDN forward path uploads
        // per-step `gdn_state_indices` / `gdn_is_fresh` H2D each step,
        // which has no analogue in the captured-graph input metadata.
        // Run eager-only for the first cut (mirrors metal). A captured
        // hybrid path is a follow-up.
        let is_hybrid = self
            .model
            .as_ref()
            .is_some_and(|m| m.gdn_runtime_config().is_some());
        if is_hybrid {
            info!(
                "ScratchyWorker: hybrid-GDN model — skipping CUDA graph capture (eager-only for first cut)"
            );
            return Ok(());
        }

        // Hybrid SWA KV (gemma4): the per-KV-group block tables / slot mappings
        // are threaded through the EAGER forward (`HybridKvInputs`) but NOT yet
        // through the captured-graph input buffers (which hold a single block
        // table). Replaying a captured graph would make the sliding layers read
        // the full group's blocks → garbage / illegal access. Run eager-only
        // until the grouped graph path is wired; a captured hybrid-SWA path is a
        // follow-up (mirrors the hybrid-GDN first cut above).
        let is_swa_hybrid = self
            .kv_cache
            .as_ref()
            .is_some_and(|kv| kv.num_kv_groups() > 1);
        if is_swa_hybrid {
            info!(
                "ScratchyWorker: hybrid SWA KV model (gemma4) — skipping CUDA graph capture (eager-only until grouped graph path is wired)"
            );
            return Ok(());
        }

        let max_blocks_per_seq = self.max_blocks_per_seq();

        let (model, kv_cache, device) = match (&self.model, &self.kv_cache, &mut self.device) {
            (Some(m), Some(kv), Some(d)) => (m, kv, d),
            _ => return Ok(()), // Not fully initialized yet.
        };

        // Resolve Auto mode now that we know the SM version and TP config.
        {
            let resolved = self
                .config
                .cuda_graph_mode
                .resolve(device.sm_version, self.config.tp_world_size);
            info!(
                "ScratchyWorker: resolved cuda_graph_mode {:?} → {:?} (SM{}, TP={})",
                self.config.cuda_graph_mode, resolved, device.sm_version, self.config.tp_world_size
            );
            self.config.cuda_graph_mode = resolved;
        }

        // Encoder models don't support CUDA graph capture (no decode loop).
        if model.is_scratchy_encoder() {
            self.config.cuda_graph_mode = CudaGraphMode::None;
            info!("ScratchyWorker: encoder model — disabling CUDA graphs");
        }

        unsafe { driver::ctx_set_current(device.ctx) }
            .map_err(|e| ExecutorError::WorkerInit(format!("ctx_set_current: {e}")))?;

        let vocab_size = model.vocab_size();

        // Capture CUDA graphs for common decode batch sizes.
        // During decode, every request has q_len=1, so shapes are deterministic.
        let capture_sizes = if self.config.cuda_graph_sizes.is_empty() {
            // Fallback when no sizes were threaded in (the serve CLI
            // always populates `cuda_graph_sizes` via
            // `auto_capture_sizes`, so this fires only for callers that
            // leave it empty). Use the same `max_num_seqs`-clamped
            // Python-matching list rather than an unconditional 1..512
            // run — batch sizes above `max_num_seqs` are never replayed
            // (`nearest_graph_size` caps the runtime batch at it), so
            // capturing them is dead work.
            scratchy_core_config::CudaGraphConfig::auto_capture_sizes(
                self.config.max_num_seqs.max(1),
            )
        } else {
            self.config.cuda_graph_sizes.clone()
        };
        let max_bs = *capture_sizes.iter().max().unwrap();

        // Decode sk (KV-span) buckets to capture, clamped to what the
        // deployment's `max_model_len` can reach. `seqused_k` is bounded
        // by `max_model_len`, so for a context-capped model the larger
        // buckets are dead work — never selected by `pick_sk_bucket` at
        // replay — and capturing them wastes both startup time (two
        // forwards each) and the attention workspace they reserve as the
        // pool high-water mark. Keep every bucket strictly below
        // `max_model_len`, plus the first one at/above it as the
        // covering ceiling (a seq of length `L` needs a bucket `>= L`).
        // At default 32k context no bucket reaches the cap, so all four
        // survive — parity with the prior unconditional list.
        const DECODE_SK_BUCKETS_ALL: &[u32] = &[128, 512, 2048, 8192];
        let cap_max_model_len = self
            .config
            .max_model_len
            .or_else(|| {
                self.hf_config
                    .as_ref()
                    .and_then(|c| c.max_position_embeddings)
            })
            .unwrap_or(4096) as u32;
        let keep = DECODE_SK_BUCKETS_ALL
            .iter()
            .position(|&b| b >= cap_max_model_len)
            .map_or(DECODE_SK_BUCKETS_ALL.len(), |i| i + 1);
        let decode_sk_buckets = &DECODE_SK_BUCKETS_ALL[..keep];
        if decode_sk_buckets.len() < DECODE_SK_BUCKETS_ALL.len() {
            info!(
                "CUDA graph: capturing {} of {} decode sk buckets (max_model_len={cap_max_model_len} \
                 can't reach the rest): {decode_sk_buckets:?}",
                decode_sk_buckets.len(),
                DECODE_SK_BUCKETS_ALL.len(),
            );
        }
        let max_sk_bucket = *decode_sk_buckets.last().unwrap();

        // Piecewise CUDA-graph capture was removed alongside the hand-written
        // CUDA model forwards (the per-piece `execute_*_piece` bodies dispatched
        // through `crate::model::*` types directly). Treat
        // `Piecewise` / `FullAndPiecewise` modes as monolithic-only requests;
        // the scheduling-mode `resolve()` already maps them to safe defaults.

        // -----------------------------------------------------------------------
        // 2. Capture Monolithic (Full) Decode CUDA Graphs
        // -----------------------------------------------------------------------
        // Monolithic graphs capture the entire forward pass (including attention)
        // in a single CUDA graph — zero kernel launch overhead.
        //
        // MoE models are excluded: their per-layer kernel count (router +
        // N expert GEMMs + shared experts + gating) exceeds the CUDA driver's
        // undocumented graph node limit, causing CUDA_ERROR_ILLEGAL_ADDRESS that
        // permanently poisons the CUDA context. This matches Python vLLM's
        // approach of validating before capture rather than recovering after
        // failure. Piecewise graphs handle MoE models with ~1-3% decode overhead.
        //
        // Monolithic capture (TP=1) uses one captured graph per
        // `(batch_size, sk_bucket)` pair so each baked kernel matches
        // the cost solver's pick at that sk range; piecewise capture
        // (TP>1, `#[cfg(feature = "nccl")]` below) still uses a single
        // `padded_max_seqlen_k` constant since multi-bucket piecewise
        // would multiply NCCL coordination cost across each captured
        // segment — left as a follow-up.
        #[cfg(feature = "nccl")]
        let padded_max_seqlen_k: usize = 2048;
        let mut monolithic_failed = false;

        let model_supports_monolithic = !model.is_moe();
        let should_capture_monolithic = model_supports_monolithic
            && matches!(
                self.config.cuda_graph_mode,
                CudaGraphMode::Full
                    | CudaGraphMode::FullAndPiecewise
                    | CudaGraphMode::FullDecodeOnly
            );

        if !model_supports_monolithic {
            info!(
                "MoE model detected — skipping monolithic CUDA graph capture \
                 (too many graph nodes for driver limit). Piecewise graphs will be used."
            );
        }

        if !should_capture_monolithic {
            // Skip to prefill graphs / cublas autotune.
        } else {
            let mut runner = unsafe {
                CudaGraphRunner::new(max_bs, vocab_size, self.model_dtype, max_blocks_per_seq)
            }
            .map_err(|e| ExecutorError::WorkerInit(format!("CudaGraphRunner::new: {e}")))?;

            // FP8 KV cache: allocate persistent dequant buffers and cache scales.
            // Buffer is sized for the LARGEST decode sk bucket so it covers
            // every captured `(bs, sk_bucket)` graph below.
            if self.kv_cache_is_fp8 {
                unsafe {
                    runner
                        .init_fp8_buffers(
                            max_sk_bucket as usize, // largest decode sk bucket captured below
                            kv_cache.num_kv_heads,
                            kv_cache.head_dim,
                            self.model_dtype,
                        )
                        .map_err(|e| ExecutorError::WorkerInit(format!("init_fp8_buffers: {e}")))?;
                    runner
                        .cache_fp8_scales(kv_cache, device.compute_stream)
                        .map_err(|e| ExecutorError::WorkerInit(format!("cache_fp8_scales: {e}")))?;
                }
            }

            // Begin private pool for ALL graph captures (like PyTorch's shared graph pool).
            // One pool is shared across all batch sizes so blocks are reused.
            device.caching.begin_allocate_to_pool();

            // Set FP8 graph context thread-local so attention helpers use pre-allocated path.
            if let Some(ctx) = runner.fp8_graph_ctx() {
                crate::model::attention_helpers::set_fp8_graph_ctx(ctx);
            }

            // Capture multiple decode graphs per batch size — one per sk
            // bucket. The cost solver picks the cheapest Instruction
            // (FA2 vs FlashInfer paged attention) per `(num_tokens=1,
            // sk_bucket)` cell, so a single capture-time
            // `padded_max_seqlen_k` would bake the kernel that wins
            // ONLY at that sk and run it at every replay regardless of
            // the runtime span. Capturing one graph per declared sk
            // bucket lets replay dispatch (`pick_sk_bucket`) pick the
            // graph whose baked kernel matches the runtime
            // `max_seqlen_k` — short-context replays land on FA2,
            // long-context replays land on FlashInfer.
            //
            // Capture order: largest sk × largest bs first. The first
            // capture establishes the allocator pool's high-water mark
            // (sized by attention-workspace * KV span); subsequent
            // smaller (sk, bs) reuse the same memory rather than
            // growing the pool incrementally.
            'capture: for &sk_bucket in decode_sk_buckets.iter().rev() {
                for &bs in capture_sizes.iter().rev() {
                    info!("Capturing CUDA graph for batch_size={bs}, sk_bucket={sk_bucket}...");
                    let kv_ref = kv_cache;
                    let model_ref = model;

                    let result = unsafe {
                        runner.capture(bs, sk_bucket, device, |inputs, dev| {
                            model_ref.forward(
                                TensorView::from_raw(inputs.input_ids),
                                TensorView::from_raw(inputs.positions),
                                TensorView::from_raw(inputs.slot_mapping),
                                TensorView::from_raw(inputs.cu_seqlens_q),
                                TensorView::from_raw(inputs.seqused_k),
                                TensorView::from_raw(inputs.block_table),
                                1, // max_seqlen_q = 1 for decode
                                sk_bucket as usize,
                                kv_ref,
                                dev,
                                None, // no last_token_indices (decode: all tokens are last)
                                None, // text-only decode: no MM splice
                                None, // hybrid (GDN) arches are eager-only — graph capture is gated off, so this closure never fires for them
                                None, // hybrid SWA: capture path is uniform-KV
                            )
                        })
                    };

                    match result {
                        Ok(()) => {
                            info!("CUDA graph captured for batch_size={bs}, sk_bucket={sk_bucket}")
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to capture CUDA graph for bs={bs}, sk={sk_bucket}: {e}"
                            );
                            // Stop immediately — CUDA_ERROR_ILLEGAL_ADDRESS
                            // poisons the entire CUDA context. Piecewise
                            // graphs are the fallback.
                            monolithic_failed = true;
                            break 'capture;
                        }
                    }
                }
            }

            // Clear FP8 graph context thread-local.
            crate::model::attention_helpers::clear_fp8_graph_ctx();

            // End private pool after all captures. Blocks in the pool that are
            // still free are effectively owned by the captured graphs.
            device.caching.end_allocate_to_pool();

            if monolithic_failed {
                // Context is poisoned — discard any partially captured graphs.
                // Don't attempt staging allocation. Piecewise is the fallback.
                tracing::warn!(
                    "Discarding monolithic graphs (context poisoned). \
                 Piecewise graphs will handle all decode batches."
                );
            } else if !runner.captured_sizes().is_empty() {
                info!(
                    "CUDA graphs captured for (batch_size, sk_bucket): {:?}",
                    runner.captured_sizes()
                );
                // Allocate pinned host staging buffers sized for the largest captured graph.
                let staging_max_bs = *runner.captured_batch_sizes().last().unwrap();
                match unsafe { HostStaging::new(staging_max_bs, max_blocks_per_seq) } {
                    Ok(staging) => {
                        info!(
                            "Pinned host staging allocated for max_batch={}",
                            staging_max_bs
                        );
                        self.host_staging = Some(staging);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to allocate pinned staging: {e}");
                    }
                }
                self.graph_runner = Some(runner);
            }
        } // end should_capture_monolithic

        // -----------------------------------------------------------------------
        // 2b. Capture Piecewise Decode CUDA Graphs (TP>1)
        // -----------------------------------------------------------------------
        // At tp>1 monolithic graphs are off (NCCL inside a graph fails on
        // L40S). Piecewise splits the tape at AllReduce/AllGather boundaries:
        // each segment is its own captured CUgraph; collectives run eagerly
        // between graphs. Required because the eager fallback at tp>1 also
        // fails today (`H2D u32: ILLEGAL_ADDRESS` first decode step).
        //
        // MoE arches stay eager (router + expert dispatch produces too many
        // graph nodes; same node-count limit that disqualified them from
        // monolithic capture).
        #[cfg(feature = "nccl")]
        {
            let want_piecewise = self.config.tp_world_size > 1
                && !monolithic_failed
                && model_supports_monolithic // not MoE
                && self.config.cuda_graph_mode != CudaGraphMode::None;

            if want_piecewise {
                info!(
                    "ScratchyWorker: capturing piecewise CUDA graphs for tp>{} decode \
                     ({} batch sizes)",
                    1,
                    capture_sizes.len()
                );
                let pdr_res = unsafe { PiecewiseDecodeRunner::new(max_bs, max_blocks_per_seq) };
                match pdr_res {
                    Ok(mut pdr) => {
                        // FI plan replan for each sk_bucket BEFORE entering
                        // the private pool. The plan workspace lives outside
                        // the caching pool, and its int_ws_d is updated by
                        // these calls. Each captured kernel reads the plan
                        // for its (q=1, k, num_pages) tuple — without a
                        // pre-built plan the captured kernel hits a runtime
                        // fi_plan_new which is forbidden during graph
                        // capture. Mirrors the implicit warmup the eager
                        // path gets from per-step replans during profiling.
                        for &bs in capture_sizes.iter() {
                            unsafe {
                                crate::attention_helpers::replan_fi_for_decode(
                                    bs,
                                    padded_max_seqlen_k,
                                    kv_cache.block_size,
                                    device.compute_stream,
                                );
                            }
                        }
                        // One private allocator pool shared across all batch
                        // sizes — pinned addresses across captures.
                        device.caching.begin_allocate_to_pool();
                        let mut any_failed = false;
                        for &bs in capture_sizes.iter().rev() {
                            info!(
                                "Piecewise: capturing for batch_size={bs} (max_seqlen_k={padded_max_seqlen_k})…"
                            );
                            let res = unsafe {
                                pdr.capture(bs, padded_max_seqlen_k, model, kv_cache, device)
                            };
                            match res {
                                Ok(()) => {
                                    info!("Piecewise CUDA graphs captured for batch_size={bs}")
                                }
                                Err(e) => {
                                    tracing::warn!("Piecewise capture failed at bs={bs}: {e}");
                                    any_failed = true;
                                    break;
                                }
                            }
                        }
                        device.caching.end_allocate_to_pool();

                        if any_failed {
                            tracing::warn!(
                                "Discarding piecewise runner — partial captures unsafe \
                                 to use. Decode will run eager."
                            );
                        } else if !pdr.captured_sizes().is_empty() {
                            info!(
                                "Piecewise CUDA graphs captured for batch_sizes: {:?}",
                                pdr.captured_sizes()
                            );
                            // Need pinned host staging for replay's H2D
                            // step — same as monolithic.
                            if self.host_staging.is_none() {
                                let staging_max_bs = *pdr.captured_sizes().last().unwrap();
                                match unsafe {
                                    HostStaging::new(staging_max_bs, max_blocks_per_seq)
                                } {
                                    Ok(staging) => {
                                        info!(
                                            "Pinned host staging allocated for max_batch={} (piecewise)",
                                            staging_max_bs
                                        );
                                        self.host_staging = Some(staging);
                                    }
                                    Err(e) => tracing::warn!(
                                        "Failed to allocate pinned staging (piecewise): {e}"
                                    ),
                                }
                            }
                            self.piecewise_decode = Some(pdr);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("PiecewiseDecodeRunner::new failed: {e}");
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // 3. Capture Prefill CUDA Graphs
        // -----------------------------------------------------------------------
        // Skip if monolithic capture poisoned the CUDA context — prefill graphs
        // use the same full forward pass and would also fail.
        if monolithic_failed {
            info!(
                "Skipping prefill graph capture (CUDA context poisoned by monolithic failure). \
                 Piecewise graphs available for decode; prefill runs eagerly."
            );
        }
        // Encoder models don't use CUDA graphs at all — skip prefill capture.
        let skip_prefill_graphs =
            monolithic_failed || self.config.cuda_graph_mode == CudaGraphMode::None;
        if skip_prefill_graphs && !monolithic_failed {
            info!("Skipping prefill graph capture (CUDA graphs disabled).");
        }
        let max_prefill_tokens = self.config.max_num_batched_tokens;
        let prefill_sizes: Vec<usize> = [128, 256, 512, 1024, 2048, 4096, 8192]
            .iter()
            .copied()
            .filter(|&s| s <= max_prefill_tokens)
            .collect();

        // At tp>1, the monolithic PrefillGraphRunner deadlocks: its captured
        // forward calls NCCL inside a single CUgraph, and the all-reduce
        // points across ranks don't synchronize cleanly during capture
        // (rank-0 reaches the captured allreduce while rank-1 is still in
        // PiecewiseDecodeRunner::capture's NCCL-suppressed warmup, etc.).
        // Use piecewise prefill instead — splits the tape at NCCL boundaries
        // exactly like piecewise decode. Matches Python vLLM's
        // FULL_AND_PIECEWISE default.
        #[cfg(feature = "nccl")]
        let use_piecewise_prefill = self.config.tp_world_size > 1;
        #[cfg(not(feature = "nccl"))]
        let use_piecewise_prefill = false;

        #[cfg(feature = "nccl")]
        if use_piecewise_prefill && !prefill_sizes.is_empty() && !skip_prefill_graphs {
            let max_prefill = *prefill_sizes.last().unwrap();
            match unsafe { PiecewisePrefillRunner::new(max_prefill, max_blocks_per_seq) } {
                Ok(mut ppr) => {
                    info!(
                        "ScratchyWorker: capturing piecewise prefill CUDA graphs for tp>1 \
                         ({} prefill sizes)",
                        prefill_sizes.len()
                    );
                    // FI plan replan for prefill sizes BEFORE entering the
                    // private pool (cudaMalloc on first plan call would fail
                    // inside cuStreamBeginCapture). Mirrors the decode-side
                    // pre-capture replan loop above.
                    for &num_tokens in prefill_sizes.iter() {
                        unsafe {
                            crate::attention_helpers::replan_fi_for_decode(
                                num_tokens,
                                num_tokens,
                                kv_cache.block_size,
                                device.compute_stream,
                            );
                        }
                    }
                    // The piecewise decode capture above (and any prior
                    // monolithic pass) already opened the private pool —
                    // `end_allocate_to_pool` is a no-op, so the pool stays
                    // live and a second `begin` would assert. Reuse the
                    // existing active pool, mirroring how the monolithic
                    // PrefillGraphRunner branch shares the decode pool.
                    let opened_pool_here = if !device.caching.is_pool_active() {
                        device.caching.begin_allocate_to_pool();
                        true
                    } else {
                        false
                    };
                    let mut any_failed = false;
                    for &num_tokens in prefill_sizes.iter().rev() {
                        info!("Piecewise prefill: capturing for num_tokens={num_tokens}…");
                        let res = unsafe { ppr.capture(num_tokens, model, kv_cache, device) };
                        match res {
                            Ok(()) => info!(
                                "Piecewise prefill CUDA graphs captured for num_tokens={num_tokens}"
                            ),
                            Err(e) => {
                                tracing::warn!(
                                    "Piecewise prefill capture failed at num_tokens={num_tokens}: {e}"
                                );
                                any_failed = true;
                                break;
                            }
                        }
                    }
                    if opened_pool_here {
                        device.caching.end_allocate_to_pool();
                    }

                    if any_failed {
                        tracing::warn!(
                            "Discarding piecewise prefill runner — partial captures unsafe \
                             to use. Prefill will run eager."
                        );
                    } else if !ppr.captured_sizes().is_empty() {
                        info!(
                            "Piecewise prefill CUDA graphs captured for sizes: {:?}",
                            ppr.captured_sizes()
                        );
                        self.piecewise_prefill = Some(ppr);
                    }
                }
                Err(e) => {
                    tracing::warn!("PiecewisePrefillRunner::new failed: {e}");
                }
            }
        }

        if !use_piecewise_prefill && !prefill_sizes.is_empty() && !skip_prefill_graphs {
            let max_prefill = *prefill_sizes.last().unwrap();
            match unsafe {
                PrefillGraphRunner::new(
                    max_prefill,
                    vocab_size,
                    self.model_dtype,
                    max_blocks_per_seq,
                )
            } {
                Ok(mut prefill_runner) => {
                    // Capture largest first (matching Python vLLM).
                    for &num_tokens in prefill_sizes.iter().rev() {
                        info!("Capturing prefill CUDA graph for num_tokens={num_tokens}...");
                        let kv_ref = kv_cache;
                        let model_ref = model;

                        let result = unsafe {
                            prefill_runner.capture(num_tokens, device, |inputs, dev| {
                                model_ref.forward(
                                    TensorView::from_raw(inputs.input_ids),
                                    TensorView::from_raw(inputs.positions),
                                    TensorView::from_raw(inputs.slot_mapping),
                                    TensorView::from_raw(inputs.cu_seqlens_q),
                                    TensorView::from_raw(inputs.seqused_k),
                                    TensorView::from_raw(inputs.block_table),
                                    num_tokens, // max_seqlen_q
                                    num_tokens, // max_seqlen_k
                                    kv_ref,
                                    dev,
                                    Some(TensorView::from_raw(inputs.last_token_indices)),
                                    None, // graph-captured prefill is text-only
                                    None, // hybrid (GDN) prefill is eager-only — capture path is gated off for them
                                    None, // hybrid SWA: capture path is uniform-KV
                                )
                            })
                        };

                        match result {
                            Ok(()) => {
                                info!("Prefill CUDA graph captured for num_tokens={num_tokens}")
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to capture prefill graph for {num_tokens}: {e}"
                                );
                                break;
                            }
                        }
                    }

                    if !prefill_runner.captured_sizes().is_empty() {
                        info!(
                            "Prefill CUDA graphs captured for token counts: {:?}",
                            prefill_runner.captured_sizes()
                        );
                        self.prefill_graph_runner = Some(prefill_runner);
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to create PrefillGraphRunner: {e}");
                }
            }
        }

        if self.config.cublas_autotune {
            unsafe { device.cublas.benchmark_plans() };
        }

        Ok(())
    }

    fn sleep(&mut self, level: u32) -> ExecutorResult<()> {
        if level == 0 {
            // Level 0 = pause only, handled by the scheduler.
            return Ok(());
        }

        info!("ScratchyWorker: sleeping (level {level}) — freeing GPU memory");

        // Flush any deferred D2H commit.
        if self.pending_commit.take().is_some()
            && let Some(ref dev) = self.device
        {
            let _ = dev.sync_d2h();
        }

        // Save num_gpu_blocks for re-init on wake.
        if let Some(ref kv) = self.kv_cache {
            self.num_gpu_blocks_saved = kv.num_blocks;
        }

        // Drop CUDA graphs (Drop impl frees GPU memory).
        self.graph_runner = None;
        #[cfg(feature = "nccl")]
        {
            self.piecewise_decode = None;
            self.piecewise_prefill = None;
        }
        self.prefill_graph_runner = None;
        self.last_graph_batch_size = None;
        self.graph_metadata_valid = false;

        // Drop host staging (Drop impl frees pinned memory).
        self.host_staging = None;

        // Drop KV cache (Drop impl frees GPU memory).
        self.kv_cache = None;

        // Drop logits pipeline (frees GPU state).
        self.logits_pipeline = None;

        // Free weight GPU memory and drop model.
        if let Some(ref dev) = self.device {
            // Sync to ensure no in-flight ops reference weight memory.
            let _ = unsafe { driver::stream_synchronize(dev.compute_stream) };
        }
        // Drop tracked weight allocations (RawGpuMem::Drop frees GPU memory).
        self.weight_gpu_allocs.clear();
        self.model = None;

        // Clear per-request state.
        self.token_buffers.clear();
        self.prompt_lengths.clear();
        self.annotation_buffers.clear();
        self.sampling_params_map.clear();
        self.input_batch = InputBatch::new();
        self.batch_changed = false;
        self.batch_req_ids.clear();
        self.seeded_rngs.clear();
        #[cfg(feature = "guided-decoding")]
        self.grammar_states.clear();

        // Release all GPU memory held by the caching allocator (segments,
        // blocks, private pools). This returns memory to the CUDA driver and
        // resets the allocator so begin_allocate_to_pool() works on wake.
        if let Some(ref mut dev) = self.device {
            unsafe { dev.caching.release_all() };
        }

        info!("ScratchyWorker: sleep complete — GPU memory released");
        Ok(())
    }

    fn wake_up(&mut self, _tags: Option<&[String]>) -> ExecutorResult<()> {
        info!("ScratchyWorker: waking up — reloading model and KV cache");

        // Re-bind cuBLAS workspace — sleep's release_all() freed the old one.
        if let Some(ref mut dev) = self.device {
            unsafe { dev.cublas.rebind_workspace(&mut dev.caching) };
        }

        // Re-load model from disk (mmap'd safetensors, fast).
        self.load_model()?;

        // Re-init KV cache with saved block count.
        let num_gpu_blocks = self.num_gpu_blocks_saved;
        if num_gpu_blocks > 0 {
            self.initialize_cache(num_gpu_blocks, 0)?;
        }

        // Re-capture CUDA graphs.
        self.compile_or_warm_up_model()?;

        info!("ScratchyWorker: wake complete");
        Ok(())
    }

    fn shutdown(&mut self) {
        // Flush any deferred D2H commit before dropping the device.
        if self.pending_commit.take().is_some()
            && let Some(ref dev) = self.device
        {
            let _ = dev.sync_d2h();
        }
        // Drop tracked weight allocations (RawGpuMem::Drop frees GPU memory).
        self.weight_gpu_allocs.clear();
        self.model = None;
        self.kv_cache = None;
        // Do NOT drop `self.device` here. The allocator must outlive every
        // OwnedTensor holder still on `self` (logits_pipeline/PenaltiesProcessor,
        // graph runners, …). It is the last CUDA field, so it drops last when the
        // worker itself drops — after those fields. Freeing it here freed the
        // allocator while those tensors were still live → use-after-free in their
        // Drop (ASan-confirmed: PenaltiesProcessor::drop → CachingAllocator::free
        // → active_blocks.remove on a freed table).
        self.is_shutdown = true;
    }

    fn rank(&self) -> usize {
        0
    }

    fn local_rank(&self) -> usize {
        self.config.device_id as usize
    }

    fn is_driver_worker(&self) -> bool {
        true
    }

    fn take_preloaded_tokenizer(&mut self) -> Option<tokenizers::Tokenizer> {
        self.preloaded_tokenizer.take()
    }

    fn architecture(&self) -> Option<String> {
        self.resolved_architecture.clone()
    }

    fn embed(&mut self, token_id_seqs: &[&[u32]]) -> ExecutorResult<Vec<Vec<f32>>> {
        // Ensure CUDA context is current.
        if let Some(ref dev) = self.device {
            unsafe { driver::ctx_set_current(dev.ctx) }
                .map_err(|e| ExecutorError::WorkerExecution(format!("ctx_set_current: {e}")))?;
        }

        let (model, kv_cache, device) = match (&self.model, &self.kv_cache, &mut self.device) {
            (Some(m), Some(kv), Some(d)) => (m, kv, d),
            _ => {
                return Err(ExecutorError::WorkerExecution(
                    "model, KV cache, or device not initialized".into(),
                ));
            }
        };

        let block_size = self.config.block_size;
        let strategy = self.pooling_strategy;
        let mut results = Vec::with_capacity(token_id_seqs.len());

        for token_ids in token_id_seqs {
            let num_tokens = token_ids.len();
            if num_tokens == 0 {
                let hidden_size = model.hidden_size();
                results.push(vec![0.0f32; hidden_size]);
                continue;
            }

            // Build positions [0, 1, 2, ...].
            let positions: Vec<u32> = (0..num_tokens as u32).collect();

            // Upload inputs.
            let gpu_input_ids = Self::h2d_u32(token_ids, device)?;
            let gpu_positions = Self::h2d_u32(&positions, device)?;

            // Slot mapping: use block 0 sequentially.
            let slot_mapping: Vec<i64> = (0..num_tokens)
                .map(|t| {
                    let block_idx = t / block_size;
                    let offset = t % block_size;
                    (block_idx * block_size + offset) as i64
                })
                .collect();
            let gpu_slot_mapping = Self::h2d_i64(&slot_mapping, device)?;

            // Attention metadata: single sequence.
            let cu_seqlens_q = vec![0i32, num_tokens as i32];
            let gpu_cu_q = Self::h2d_i32(&cu_seqlens_q, device)?;
            let seqused_k = vec![num_tokens as i32];
            let gpu_seqused_k = Self::h2d_i32(&seqused_k, device)?;

            // Block table: [1, max_blocks].
            let max_blocks = num_tokens.div_ceil(block_size);
            let block_table: Vec<i32> = (0..max_blocks as i32).collect();
            let mut gpu_bt = Self::h2d_i32(&block_table, device)?;
            unsafe { gpu_bt.reshape(&[1, max_blocks], GpuDType::I32) };

            // Forward pass (backbone only).
            let hidden_states = unsafe {
                model.hidden_states(
                    gpu_input_ids.view(),
                    gpu_positions.view(),
                    gpu_slot_mapping.view(),
                    gpu_cu_q.view(),
                    gpu_seqused_k.view(),
                    gpu_bt.view(),
                    num_tokens,
                    num_tokens,
                    kv_cache,
                    device,
                    None,
                    None, // gdn_inputs: pooling embed is text-only — pooling models aren't hybrid
                    None, // hybrid SWA: pooling is uniform-KV
                )
            };

            // Pool + normalize. Dereference OwnedTensor → GpuTensor (Copy).
            let embedding = Self::pool_and_normalize(*hidden_states, num_tokens, strategy, device)?;
            // Side-channel embed always uses single-vector pooling (Last/Cls/Mean).
            match embedding {
                EmbeddingData::Single(v) => results.push(v),
                EmbeddingData::Multi(_) => {
                    return Err(ExecutorError::WorkerExecution(
                        "AllTokens pooling requires --runner pooling mode".into(),
                    ));
                }
            }
            // OwnedTensor dropped here — memory returns to caching allocator.
        }

        Ok(results)
    }
} // end impl Worker for CudaWorker

#[cfg(feature = "cuda")]
impl CudaWorker {
    fn execute_model_inner(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
        // Ensure CUDA context is current on this thread (once per thread).
        if !self.ctx_set_on_thread {
            if let Some(ref dev) = self.device {
                unsafe { driver::ctx_set_current(dev.ctx) }
                    .map_err(|e| ExecutorError::WorkerExecution(format!("ctx_set_current: {e}")))?;
            }
            self.ctx_set_on_thread = true;
        }

        let block_size = self.config.block_size;

        // NOTE: pending commit from the previous step is resolved lazily:
        // - Super fast path: resolved AFTER graph launch (overlaps with GPU)
        // - Normal path: resolved below, before prepare_inputs needs it

        // Clean up finished requests.
        let has_preempted = scheduler_output
            .preempted_req_ids
            .as_ref()
            .is_some_and(|s| !s.is_empty());
        self.batch_changed = !scheduler_output.finished_req_ids.is_empty()
            || !scheduler_output.scheduled_new_reqs.is_empty()
            || has_preempted
            || !scheduler_output
                .scheduled_cached_reqs
                .resumed_req_ids
                .is_empty();
        if self.batch_changed {
            // Batch composition changed — can't reuse persistent input_ids or metadata.
            self.last_graph_batch_size = None;
            self.graph_metadata_valid = false;
        }
        for req_id in &scheduler_output.finished_req_ids {
            self.token_buffers.remove(req_id);
            self.prompt_lengths.remove(req_id);
            self.annotation_buffers.remove(req_id);
            self.mm_data_buffers.remove(req_id);
            self.sampling_params_map.remove(req_id);
            self.seeded_rngs.remove(req_id);
            #[cfg(feature = "guided-decoding")]
            self.grammar_states.remove(req_id);
            // Recycle the request's GDN recurrent-state slot (hybrid
            // arches only). The slot's state is now stale; the next
            // request to claim it is flagged fresh so it zero-inits
            // rather than continuing from a finished sequence's state.
            // Mirrors the metal arm.
            if let Some(alloc) = self.gdn_slot_allocator.as_mut() {
                alloc.release(gdn_slot_key(req_id));
            }
        }
        self.input_batch
            .remove_finished(&scheduler_output.finished_req_ids);

        // Ensure grammar vocabulary is built if any new request needs it.
        #[cfg(feature = "guided-decoding")]
        {
            let needs_grammar = scheduler_output.scheduled_new_reqs.iter().any(|r| {
                r.sampling_params
                    .as_ref()
                    .is_some_and(|p| p.guided_grammar.is_some())
            });
            if needs_grammar {
                self.ensure_grammar_factory();
            }
        }

        // Process newly scheduled requests.
        for new_req in &scheduler_output.scheduled_new_reqs {
            let num_tokens = scheduler_output
                .num_scheduled_tokens
                .get(&new_req.req_id)
                .copied()
                .unwrap_or(0);
            if num_tokens == 0 {
                continue;
            }

            let prompt_ids = new_req.prompt_token_ids.as_deref().unwrap_or(&[]);
            let start = new_req.num_computed_tokens as usize;
            let end = (start + num_tokens).min(prompt_ids.len());
            let tokens_to_use = &prompt_ids[start..end];

            tracing::info!(
                req_id = %new_req.req_id,
                prompt_len = prompt_ids.len(),
                cached = start,
                new = tokens_to_use.len(),
                hit_rate = format_args!("{:.0}%", if prompt_ids.is_empty() { 0.0 } else { start as f64 / prompt_ids.len() as f64 * 100.0 }),
                "KV cache hit",
            );

            self.token_buffers
                .insert(new_req.req_id.clone(), prompt_ids.to_vec());
            self.prompt_lengths
                .insert(new_req.req_id.clone(), prompt_ids.len());
            if let Some(ref ann) = new_req.block_annotations {
                self.annotation_buffers
                    .insert(new_req.req_id.clone(), ann.clone());
            }
            if let Some(ref mm) = new_req.mm_data {
                self.mm_data_buffers
                    .insert(new_req.req_id.clone(), mm.clone());
            }
            if let Some(ref params) = new_req.sampling_params {
                self.sampling_params_map
                    .insert(new_req.req_id.clone(), params.clone());
                // Create per-request seeded RNG if seed is specified.
                if let Some(seed) = params.seed {
                    use rand::SeedableRng;
                    self.seeded_rngs.insert(
                        new_req.req_id.clone(),
                        rand::rngs::StdRng::seed_from_u64(seed),
                    );
                }
                #[cfg(feature = "guided-decoding")]
                if let Some(ref grammar) = params.guided_grammar
                    && let Some(ref factory) = self.grammar_factory
                {
                    match scratchy_core_model::grammar::GrammarGuide::from_guided_grammar(
                        grammar, factory,
                    ) {
                        Ok(guide) => {
                            self.grammar_states.insert(new_req.req_id.clone(), guide);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to compile grammar for request {}: {e}",
                                new_req.req_id
                            );
                        }
                    }
                }
            }

            // Group 0 = full/global; groups 1.. = sliding groups (gemma4 SWA),
            // present only for hybrid models. Mirror the metal split so the
            // grouped decode reads each layer's own group table.
            let block_ids = new_req.block_ids.first().cloned().unwrap_or_default();
            let sliding_groups: Vec<Vec<usize>> =
                new_req.block_ids.iter().skip(1).cloned().collect();
            self.input_batch.add_request_hybrid(
                new_req.req_id.clone(),
                tokens_to_use,
                block_ids,
                sliding_groups,
                new_req.num_computed_tokens,
            );
        }

        // Update cached requests' block tables. Track if any changed.
        let mut blocks_changed = !scheduler_output.scheduled_new_reqs.is_empty();
        for (i, req_id) in scheduler_output
            .scheduled_cached_reqs
            .req_ids
            .iter()
            .enumerate()
        {
            if let Some(Some(new_blocks)) =
                scheduler_output.scheduled_cached_reqs.new_block_ids.get(i)
                && let Some(group0) = new_blocks.first()
            {
                if !group0.is_empty() {
                    blocks_changed = true;
                }
                // Sliding groups (gemma4 SWA) ship each step too — carry them so
                // decode reads the sliding layers' own group tables (empty on
                // uniform models → update_blocks_hybrid leaves them untouched).
                let sliding_new: Vec<Vec<usize>> = new_blocks.iter().skip(1).cloned().collect();
                self.input_batch
                    .update_blocks_hybrid(req_id, group0.clone(), sliding_new);
            }
        }

        // ---------------------------------------------------------------
        // PP token fixup: when the scheduler provides new_token_ids (PP sync
        // scheduling), overwrite last_token_ids in the input batch so that
        // non-last PP stages embed the correct token instead of the dummy 0
        // committed in the previous step.
        // Matches Python: gpu_model_runner.py _update_states lines 1143-1150.
        // ---------------------------------------------------------------
        let cached = &scheduler_output.scheduled_cached_reqs;
        if !cached.new_token_ids.is_empty() {
            for (i, req_id) in cached.req_ids.iter().enumerate() {
                if let Some(tokens) = cached.new_token_ids.get(i)
                    && let Some(&last_token) = tokens.last()
                {
                    self.input_batch.set_last_token(req_id, last_token);
                }
            }
        }

        // ---------------------------------------------------------------
        // Chunked prefill re-arm: detect cached requests that need
        // more than 1 token (prefill continuation). Must happen before
        // the super fast graph path, which would otherwise replay a
        // decode graph instead of running the prefill eagerly.
        // ---------------------------------------------------------------
        let has_chunked_prefill =
            scheduler_output
                .scheduled_cached_reqs
                .req_ids
                .iter()
                .any(|req_id| {
                    let is_resumed = scheduler_output
                        .scheduled_cached_reqs
                        .resumed_req_ids
                        .contains(req_id);
                    let num_scheduled = scheduler_output
                        .num_scheduled_tokens
                        .get(req_id)
                        .copied()
                        .unwrap_or(0);
                    is_resumed || num_scheduled > 1
                });

        // ---------------------------------------------------------------
        // Super fast path: skip prepare_inputs entirely when the graph
        // has valid metadata from the previous step. This avoids ~50μs
        // of CPU work and, critically, lets us defer commit_step(N-1)
        // to AFTER the graph launch so it overlaps with GPU execution.
        // ---------------------------------------------------------------
        let num_active = self.input_batch.num_active();
        // Compute the runtime sk_bucket from the previous step's tokens-in-pool
        // (each will gain one this step), then look up the captured graph that
        // matches both `(num_active, sk_bucket)`. The fast-path stickiness
        // check (`last_graph_batch_size`) extends to the sk_bucket too: if the
        // bucket changed across steps (e.g. seq grew past 512), the captured
        // graph differs and we must fall through to the regular path so the
        // FlashInfer plan slot for the new bucket gets seeded.
        let fast_sk_bucket: Option<u32> = if self.graph_metadata_valid && !has_chunked_prefill {
            self.graph_runner.as_ref().and_then(|r| {
                let (_, _, tokens_in_pool) = self.input_batch.fast_path_info();
                let max_k = tokens_in_pool
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0)
                    .saturating_add(1);
                r.pick_sk_bucket(max_k)
            })
        } else {
            None
        };
        let fast_graph_bs = if self.graph_metadata_valid && !has_chunked_prefill {
            fast_sk_bucket.and_then(|sk| {
                self.graph_runner
                    .as_ref()
                    .and_then(|r| r.nearest_graph_size(num_active, sk))
                    .filter(|&gbs| self.last_graph_batch_size == Some((gbs, sk)))
            })
        } else {
            None
        };

        if let Some(graph_bs) = fast_graph_bs
            && let Some(sk_bucket) = fast_sk_bucket
            && let Some(ref mut stg) = self.host_staging
            && let Some(ref mut device) = self.device
        {
            // Collect info from InputBatch upfront (immutable borrow ends here).
            let (req_ids, block_tables, tokens_in_pool) = self.input_batch.fast_path_info();
            let out_req_ids: Vec<String> = req_ids.to_vec();
            let token_counts = self.input_batch.fast_path_token_counts();
            // Max seqlen_k for this step = max(tokens_already_in_cache + 1).
            // The +1 accounts for the token we're about to decode (written
            // to cache by the fused QKV+rope+cache kernel before attention).
            let fast_max_seqlen_k = tokens_in_pool.iter().copied().max().unwrap_or(0) + 1;

            // Check all-greedy and no-logprobs/grammar without prepare_inputs.
            let all_greedy_fast = out_req_ids.iter().all(|rid| {
                self.sampling_params_map
                    .get(rid)
                    .is_none_or(|p| p.temperature < 1e-6)
            });
            let any_needs_full = self.seal_pad_processor.is_active()
                || out_req_ids.iter().any(|rid| {
                    self.sampling_params_map
                        .get(rid)
                        .is_some_and(|p| p.logprobs.is_some())
                        || {
                            #[cfg(feature = "guided-decoding")]
                            {
                                self.grammar_states.contains_key(rid)
                            }
                            #[cfg(not(feature = "guided-decoding"))]
                            {
                                false
                            }
                        }
                });

            if all_greedy_fast && !any_needs_full {
                let block_size = self.config.block_size;

                // Block table update for the graph (only if blocks changed).
                let new_bt = if blocks_changed {
                    let bt = unsafe { stg.fill_block_table(block_tables, graph_bs) };
                    Some(bt)
                } else {
                    None
                };

                // Graph launch — GPU self-updates positions, slot_mapping, seqused_k.
                let runner = self.graph_runner.as_ref().unwrap();
                let replay_out = unsafe {
                    runner.replay_decode_fast(
                        graph_bs,
                        sk_bucket,
                        None,
                        new_bt,
                        block_size,
                        fast_max_seqlen_k,
                        device,
                    )
                }
                .map_err(|e| {
                    ExecutorError::WorkerExecution(format!("super fast replay_decode_fast: {e}"))
                })?;

                // Async D2H — enqueue on transfer stream, don't block.
                let buf_idx = stg.token_buf_idx;
                Self::d2h_token_ids_async(stg, buf_idx, &replay_out.token_ids, num_active, device)?;

                // NOW resolve the pending commit from the previous step.
                // The GPU is running step N, so this CPU work overlaps with it.
                if let Some(pending) = self.pending_commit.take() {
                    device.sync_d2h().map_err(|e| {
                        ExecutorError::WorkerExecution(format!("pending sync_d2h: {e}"))
                    })?;
                    let prev_ids = unsafe {
                        stg.host_token_ids[pending.buf_idx].slice::<u32>(pending.num_reqs)
                    };
                    for (i, &tok) in prev_ids.iter().enumerate().take(pending.num_reqs) {
                        self.input_batch.commit_step(
                            &pending.req_ids[i],
                            &[tok],
                            pending.token_counts[i],
                            pending.has_spec_tokens[i],
                        );
                        if let Some(buf) = self.token_buffers.get_mut(&pending.req_ids[i]) {
                            buf.push(tok);
                        }
                    }
                }

                self.pending_commit = Some(PendingCommit {
                    buf_idx,
                    num_reqs: num_active,
                    req_ids: out_req_ids.clone(),
                    token_counts,
                    has_spec_tokens: vec![false; num_active],
                });

                // Toggle double-buffer.
                stg.token_buf_idx ^= 1;

                // Build deferred output.
                let event_addr = device.d2h_done as usize;
                let buf_addr = stg.host_token_ids[buf_idx].ptr() as usize;
                let nr = num_active;
                return Ok(ModelRunnerOutput::deferred(
                    out_req_ids,
                    Box::new(move || unsafe {
                        driver::event_synchronize_raw(event_addr).expect("D2H event sync failed");
                        std::slice::from_raw_parts(buf_addr as *const u32, nr).to_vec()
                    }),
                ));
            }
        }

        // Resolve any deferred D2H commit from the previous step before
        // prepare_inputs (which reads positions, tokens_in_pool, last_token_ids).
        if let Some(pending) = self.pending_commit.take() {
            if let Some(ref dev) = self.device {
                dev.sync_d2h().map_err(|e| {
                    ExecutorError::WorkerExecution(format!("pending sync_d2h: {e}"))
                })?;
            }
            if let Some(ref stg) = self.host_staging {
                let host_ids =
                    unsafe { stg.host_token_ids[pending.buf_idx].slice::<u32>(pending.num_reqs) };
                for (i, &tok) in host_ids.iter().enumerate().take(pending.num_reqs) {
                    self.input_batch.commit_step(
                        &pending.req_ids[i],
                        &[tok],
                        pending.token_counts[i],
                        pending.has_spec_tokens[i],
                    );
                    if let Some(buf) = self.token_buffers.get_mut(&pending.req_ids[i]) {
                        buf.push(tok);
                    }
                }
            }
        }

        // ---------------------------------------------------------------
        // Preemption/resumption fixup — done AFTER pending_commit resolution
        // so that token_buffers has the final token from the previous step
        // before we rebuild the prefill sequence.
        //
        // Preempted requests: remove from InputBatch to stop zombie GPU writes
        // into blocks that have been freed and reallocated.  We keep
        // token_buffers / sampling_params_map / seeded_rngs so the request
        // can be cleanly re-admitted.
        //
        // Resumed requests: re-add to InputBatch as a fresh prefill using the
        // complete token sequence (prompt + all output tokens generated so
        // far).  The pending_commit above has already appended the very last
        // token to token_buffers, so the sequence is complete.
        //
        // Matches Python gpu_model_runner._update_states: unscheduled requests
        // are removed, resumed requests are re-added via add_request.
        // ---------------------------------------------------------------
        if let Some(preempted) = &scheduler_output.preempted_req_ids {
            for req_id in preempted {
                self.input_batch.remove_request(req_id);
            }
        }
        for (i, req_id) in scheduler_output
            .scheduled_cached_reqs
            .req_ids
            .iter()
            .enumerate()
        {
            let is_resumed = scheduler_output
                .scheduled_cached_reqs
                .resumed_req_ids
                .contains(req_id);
            // Chunked prefill continuation: a cached (running) request that
            // still has un-prefilled prompt tokens. We must re-arm the
            // InputBatch slot as prefill (remove + add_request) so
            // prepare_inputs emits this chunk's prompt tokens (q_len = chunk
            // size) instead of a single decode token.
            //
            // The request is still mid-prefill iff `num_computed < prompt_len`
            // (matches Python vLLM, where a request is "in prefill" while
            // num_computed_tokens < num_prompt_tokens). This INCLUDES the
            // final chunk, where num_computed + num_scheduled == prompt_len.
            //
            // (Earlier this gated on `num_computed + num_scheduled < prompt_len`
            // (strict), which dropped the FINAL chunk of every chunked prefill:
            // its prompt tokens — including the user's actual question — were
            // never processed, and the slot ran as a 1-token decode of the
            // discarded mid-prefill sample. So every prompt longer than
            // max_num_batched_tokens produced garbage / a prompt echo. A normal
            // decode has num_computed >= prompt_len and is still excluded.)
            let prompt_len = self.prompt_lengths.get(req_id).copied().unwrap_or(0);
            let num_computed_here = scheduler_output
                .scheduled_cached_reqs
                .num_computed_tokens
                .get(i)
                .copied()
                .unwrap_or(0) as usize;
            let is_chunked_prefill_continuation = !is_resumed && num_computed_here < prompt_len;
            if !is_resumed && !is_chunked_prefill_continuation {
                continue;
            }
            // Re-add the resumed/chunked-prefill request as a fresh prefill.
            let new_block_ids: Vec<usize> = scheduler_output
                .scheduled_cached_reqs
                .new_block_ids
                .get(i)
                .and_then(|opt| opt.as_ref())
                .and_then(|groups| groups.first())
                .cloned()
                .unwrap_or_default();
            let num_computed = scheduler_output
                .scheduled_cached_reqs
                .num_computed_tokens
                .get(i)
                .copied()
                .unwrap_or(0);
            // Truncate to the scheduled chunk: scheduler allocated blocks for
            // exactly `num_scheduled_tokens` tokens starting from `num_computed`.
            // Passing the full token_buffers (prompt + all outputs) would give
            // seq_lens > available blocks → slot_mapping = -1 → GPU fault.
            // This mirrors the new-request chunked-prefill logic (lines ~4906-4908).
            let num_scheduled = scheduler_output
                .num_scheduled_tokens
                .get(req_id)
                .copied()
                .unwrap_or(0);
            let tokens: Vec<u32> = self
                .token_buffers
                .get(req_id)
                .map(|buf| {
                    let start = num_computed as usize;
                    let end = (start + num_scheduled).min(buf.len());
                    buf[start..end].to_vec()
                })
                .unwrap_or_default();
            // For both resumed and chunked prefill, the scheduler's
            // allocate_slots returns ALL block IDs (not a delta).
            // For chunked prefill, update_blocks already set the full
            // block table on the input_batch slot — grab it before remove.
            let all_block_ids = if is_chunked_prefill_continuation {
                // update_blocks (line ~6359) already set the full block table
                self.input_batch
                    .block_table(req_id)
                    .map(|b| b.to_vec())
                    .unwrap_or(new_block_ids)
            } else {
                new_block_ids
            };
            // Sliding KV groups (gemma4 SWA) must survive the remove + re-add,
            // else the continuation/resume chunk drops to group 0 and sliding
            // layers read the full group's blocks → garbage. Chunked
            // continuation: the stored tables (set by update_blocks_hybrid).
            // Resumed: the scheduler ships the full per-group set this step.
            let all_sliding: Vec<Vec<usize>> = if is_chunked_prefill_continuation {
                self.input_batch
                    .sliding_groups(req_id)
                    .map(|s| s.to_vec())
                    .unwrap_or_default()
            } else {
                scheduler_output
                    .scheduled_cached_reqs
                    .new_block_ids
                    .get(i)
                    .and_then(|opt| opt.as_ref())
                    .map(|groups| groups.iter().skip(1).cloned().collect())
                    .unwrap_or_default()
            };
            // Remove the old (zombie) slot first, then re-add as prefill.
            self.input_batch.remove_request(req_id);
            self.input_batch.add_request_hybrid(
                req_id.clone(),
                &tokens,
                all_block_ids,
                all_sliding,
                num_computed,
            );
        }

        // Prepare flat inputs from InputBatch.
        let prepared = self
            .input_batch
            .prepare_inputs(&scheduler_output.scheduled_spec_decode_tokens);
        if prepared.flat_token_ids.is_empty() {
            return Ok(ModelRunnerOutput::from_token_map(HashMap::new()));
        }

        let total_tokens = prepared.flat_token_ids.len();
        // PP plumbing was removed alongside the hand-written CUDA model
        // forwards; load_model rejects pp_size > 1, so this path is decoder-
        // only and always pp_active == false.

        // Spans: mark per-block rotation flags based on BlockAnnotations.
        //
        // Relocatable blocks go through a rotate-attend-unrotate cycle:
        // - K is written WITH RoPE (normal QKV projection)
        // - Post-attention: un-rotated (inverse RoPE) → position-independent
        // - Pre-attention (next step): rotated to current position
        //
        // `block_is_unrotated[physical_block] = true` means:
        //   "this Relocatable block's K is currently stored WITHOUT RoPE
        //    (from a prior step's post-attention un-rotation pass)."
        //
        // Freshly written blocks are false — K has RoPE from QKV projection.
        // Non-Relocatable blocks are always false (never touched).
        if !self.annotation_buffers.is_empty()
            && let Some(kv_cache) = self.kv_cache.as_mut()
        {
            let meta = &prepared.attn_meta;
            for i in 0..meta.num_reqs {
                let req_id = &meta.req_ids[i];
                if let Some(annotations) = self.annotation_buffers.get(req_id) {
                    let block_ids = &meta.block_ids[i];
                    let tokens_before = meta.tokens_before[i];
                    let seq_len = meta.seq_lens[i];
                    for (block_idx, &physical_block) in block_ids.iter().enumerate() {
                        let block_start_pos = block_idx * block_size;
                        if block_start_pos >= seq_len {
                            break; // past the end of the sequence
                        }
                        let (is_relocatable, is_unrotated) =
                            scratchy_core_common::compute_block_flags(
                                annotations,
                                block_idx,
                                block_size,
                                seq_len,
                                tokens_before,
                            );
                        kv_cache.mark_block(physical_block, is_relocatable, is_unrotated);
                    }
                }
            }
            // Upload flags to GPU for the attention kernel.
            unsafe {
                let stream = self
                    .device
                    .as_ref()
                    .map(|d| d.compute_stream)
                    .unwrap_or(std::ptr::null_mut());
                kv_cache.sync_block_flags_to_gpu(stream);
            }
        }

        // Compute once before the split borrow below (self is borrowed mutably for device).
        let max_blocks_per_seq = self.max_blocks_per_seq();

        // GDN per-step state-slot indices (hybrid arches only). One i32
        // slot id + u32 fresh flag per batched sequence, in the SAME
        // order as `cu_seqlens_q` (== `prepared.attn_meta.req_ids`). The
        // CUDA forward downstream H2Ds these into `ForwardCtx::{
        // gdn_state_indices, gdn_is_fresh}` via `Self::h2d_i32` /
        // `Self::h2d_u32`. `slot_for` returns `is_fresh=true` on a
        // request's FIRST forward (including a recycled slot's new owner)
        // so the GDN conv1d/scan kernels zero-init the slot's conv/ssm
        // state instead of continuing from a finished sequence's stale
        // data (the degeneration guard). Mirrors the metal arm at line
        // 9551 onward.
        self.gdn_pending = if let Some(alloc) = self.gdn_slot_allocator.as_mut() {
            let req_ids = &prepared.attn_meta.req_ids;
            let mut indices: Vec<i32> = Vec::with_capacity(req_ids.len());
            let mut fresh: Vec<u32> = Vec::with_capacity(req_ids.len());
            for req_id in req_ids {
                match alloc.slot_for(gdn_slot_key(req_id)) {
                    Some((slot, is_fresh)) => {
                        indices.push(slot as i32);
                        fresh.push(u32::from(is_fresh));
                    }
                    None => {
                        return Err(ExecutorError::WorkerExecution(format!(
                            "GDN state-slot pool exhausted (capacity {}): scheduler \
                             admitted more concurrent sequences than max_num_seqs",
                            alloc.capacity(),
                        )));
                    }
                }
            }
            Some((indices, fresh))
        } else {
            None
        };

        // Split borrows: model + kv_cache (shared) vs device (mutable).
        // Use direct field access so the borrow checker sees disjoint borrows.
        // `self.gdn_state` and `self.gdn_pending` are read again at the eager
        // forward call site below; hoist their borrow here so the split
        // doesn't conflict.
        let gdn_state_ref = self.gdn_state.as_ref();
        let gdn_pending_taken = self.gdn_pending.take();
        let (model, kv_cache, device) = match (&self.model, &self.kv_cache, &mut self.device) {
            (Some(m), Some(kv), Some(d)) => (m, kv, d),
            _ => {
                return Err(ExecutorError::WorkerExecution(
                    "model, KV cache, or device not initialized".into(),
                ));
            }
        };
        let num_reqs = prepared.req_inputs.len();

        // Build batch_req_ids for this step (used by processor updates after forward).
        self.batch_req_ids.clear();
        self.batch_req_ids
            .extend(prepared.req_inputs.iter().map(|r| r.req_id.clone()));

        // --- Pooling mode: run backbone, pool, return embeddings ---
        if self.is_pooling {
            let gpu_input_ids = Self::h2d_u32(&prepared.flat_token_ids, device)?;
            let gpu_positions = Self::h2d_u32(&prepared.flat_positions, device)?;

            let full_block_size = kv_cache.max_block_size();
            let (cu_seqlens_q, seqused_k, max_seqlen_q, max_seqlen_k, slot_mappings, block_tables) =
                Self::build_attention_tensors(
                    &prepared.attn_meta,
                    block_size,
                    full_block_size,
                    device,
                )?;
            // Pooling models are never hybrid SWA → single group; use group 0.
            let slot_mapping = &slot_mappings[0];
            let block_table_gpu = &block_tables[0];

            let hidden_states = unsafe {
                model.hidden_states(
                    gpu_input_ids.view(),
                    gpu_positions.view(),
                    slot_mapping.view(),
                    cu_seqlens_q.view(),
                    seqused_k.view(),
                    block_table_gpu.view(),
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    device,
                    None,
                    None, // gdn_inputs: pooling-mode hidden_states is text-only — pooling models aren't hybrid
                    None, // hybrid SWA: pooling is uniform-KV
                )
            };

            // Pool each request's hidden states slice.
            let strategy = self.pooling_strategy;
            let meta = &prepared.attn_meta;
            let mut pooler_map: HashMap<String, EmbeddingData> = HashMap::new();

            for (req_idx, req_slice) in prepared.req_inputs.iter().enumerate() {
                let q_start = meta.query_start_loc[req_idx];
                let q_len = meta.q_lens[req_idx];
                let hidden_size = model.hidden_size();

                // Narrow hidden_states to this request's rows.
                let row_bytes = hidden_size * hidden_states.dtype().size_bytes();
                let req_hs = unsafe {
                    GpuTensor::new(
                        hidden_states.raw_ptr().add(q_start * row_bytes),
                        &[q_len, hidden_size],
                        hidden_states.dtype(),
                    )
                };

                let embedding = Self::pool_and_normalize(req_hs, q_len, strategy, device)?;
                pooler_map.insert(req_slice.req_id.clone(), embedding);

                // Commit step so InputBatch tracks progress.
                self.input_batch.commit_step(
                    &req_slice.req_id,
                    &[0], // dummy token — pooling doesn't generate tokens
                    req_slice.token_count,
                    false,
                );
            }

            self.input_batch.reclaim_buffers(prepared);

            // Build a ModelRunnerOutput with pooler_output and empty generation fields.
            let mut output = ModelRunnerOutput::from_token_map(HashMap::new());
            output.pooler_output = Some(pooler_map);
            return Ok(output);
        }

        let vocab_size = model.vocab_size();

        // Check if this is a pure decode batch (all q_len=1) and we have a graph.
        // We allow padding to the nearest captured graph size (e.g. BS=3 → graph BS=4).
        let is_decode = prepared.attn_meta.q_lens.iter().all(|&q| q == 1);
        // MM-bearing reqs MUST take the eager path: the decode CUDA graph was
        // captured with 1D position tensors (single rope-kernel invocation
        // per layer); replaying it for an MM-bearing req would feed 1D
        // positions to the MRoPE kernel, which disagrees with the 3D
        // positions used to encode the cached visual KV → attention misfires
        // and the model emits `<|im_end|>` immediately (Bug 3, the symmetric
        // counterpart of Bug 1's prefill-graph gate at `use_prefill_graph`).
        // The eager else-branch below builds `[3, n_tokens]` positions via
        // `build_all_mm_position_patches` whether or not the encoder runs
        // this step.
        let any_mm_in_batch = prepared
            .req_inputs
            .iter()
            .any(|r| self.mm_data_buffers.contains_key(&r.req_id));
        // Pure-decode batch: each request contributes 1 query token; the
        // KV span the captured kernel must cover is `max(seq_lens)` (already
        // includes the token-about-to-be-decoded since meta.seq_lens is
        // post-RopeAndCacheKV). Pick the captured graph keyed on
        // `(num_reqs, sk_bucket_for(max_seqlen_k))`.
        let decode_max_seqlen_k: usize = if is_decode && !any_mm_in_batch {
            prepared
                .attn_meta
                .seq_lens
                .iter()
                .copied()
                .max()
                .unwrap_or(1)
        } else {
            0
        };
        let decode_sk_bucket: Option<u32> = if is_decode && !any_mm_in_batch {
            self.graph_runner
                .as_ref()
                .and_then(|r| r.pick_sk_bucket(decode_max_seqlen_k))
        } else {
            None
        };
        let graph_bs = if is_decode && !any_mm_in_batch {
            decode_sk_bucket.and_then(|sk| {
                self.graph_runner
                    .as_ref()
                    .and_then(|r| r.nearest_graph_size(num_reqs, sk))
            })
        } else {
            None
        };
        let use_graph = graph_bs.is_some();

        // At tp>1 monolithic graph_runner is None; piecewise_decode is the
        // graph path. Same gating: pure decode, no MM in batch.
        #[cfg(feature = "nccl")]
        let piecewise_bs: Option<usize> = if !use_graph && is_decode && !any_mm_in_batch {
            self.piecewise_decode
                .as_ref()
                .and_then(|r| r.nearest_size(num_reqs))
        } else {
            None
        };
        #[cfg(not(feature = "nccl"))]
        let piecewise_bs: Option<usize> = None;
        let use_piecewise = piecewise_bs.is_some();

        if is_decode && !use_graph && !use_piecewise {
            tracing::debug!("CUDA graph miss: decode bs={num_reqs} has no matching graph");
        }

        // Check if all requests are greedy (temp < 1e-6). Used to decide
        // whether to use the in-graph argmax fast path.
        let all_greedy = prepared.req_inputs.iter().all(|r| {
            self.sampling_params_map
                .get(&r.req_id)
                .is_none_or(|p| p.temperature < 1e-6)
        });

        // ---------------------------------------------------------------------------
        // Mixed batch: split into decode (CUDA graph) + prefill (eager) passes.
        // This avoids running the entire batch through the slow eager path when
        // most requests are decode (q_len=1) but a few are prefill chunks.
        // ---------------------------------------------------------------------------
        let has_decode = prepared.attn_meta.q_lens.contains(&1);
        let any_spec_in_batch = prepared
            .req_inputs
            .iter()
            .any(|r| !r.spec_token_ids.is_empty());
        let is_mixed = !is_decode && has_decode;
        // Mixed batch: the decode subset's KV span is the max `seq_lens`
        // among rows with `q_len == 1`. Match against captured graphs by
        // `(n_decode, sk_bucket_for(max_seqlen_k_decode))`.
        let mixed_decode_max_seqlen_k: usize = if is_mixed && !any_spec_in_batch {
            prepared
                .attn_meta
                .q_lens
                .iter()
                .zip(prepared.attn_meta.seq_lens.iter())
                .filter_map(|(&q, &sk)| (q == 1).then_some(sk))
                .max()
                .unwrap_or(1)
        } else {
            0
        };
        let mixed_decode_sk_bucket: Option<u32> = if is_mixed && !any_spec_in_batch {
            self.graph_runner
                .as_ref()
                .and_then(|r| r.pick_sk_bucket(mixed_decode_max_seqlen_k))
        } else {
            None
        };
        let decode_graph_bs = if is_mixed && !any_spec_in_batch {
            mixed_decode_sk_bucket.and_then(|sk| {
                self.graph_runner.as_ref().and_then(|r| {
                    let n_decode = prepared
                        .attn_meta
                        .q_lens
                        .iter()
                        .filter(|&&q| q == 1)
                        .count();
                    r.nearest_graph_size(n_decode, sk)
                })
            })
        } else {
            None
        };

        if let Some(decode_graph_bs) = decode_graph_bs {
            // Partition requests into decode (q_len=1) and prefill (q_len>1).
            let mut decode_indices: Vec<usize> = Vec::new();
            let mut prefill_indices: Vec<usize> = Vec::new();
            for (i, &q) in prepared.attn_meta.q_lens.iter().enumerate() {
                if q == 1 {
                    decode_indices.push(i);
                } else {
                    prefill_indices.push(i);
                }
            }
            let n_decode = decode_indices.len();
            let n_prefill = prefill_indices.len();

            // --- Decode pass: run through CUDA graph ---
            // Build decode inputs from the subset of requests.
            let decode_logits = {
                let meta = &prepared.attn_meta;

                let mut input_ids = Vec::with_capacity(decode_graph_bs);
                let mut positions = Vec::with_capacity(decode_graph_bs);
                let mut slot_mapping: Vec<i64> = Vec::with_capacity(decode_graph_bs);
                let mut cu_seqlens_q: Vec<i32> = Vec::with_capacity(decode_graph_bs + 1);
                let mut seqused_k: Vec<i32> = Vec::with_capacity(decode_graph_bs);
                let mut block_table = vec![0i32; decode_graph_bs * max_blocks_per_seq];

                cu_seqlens_q.push(0);
                for (out_idx, &orig_idx) in decode_indices.iter().enumerate() {
                    // Each decode request has exactly 1 token.
                    let token_start = meta.query_start_loc[orig_idx];
                    input_ids.push(prepared.flat_token_ids[token_start]);
                    positions.push(prepared.flat_positions[token_start]);
                    cu_seqlens_q.push((out_idx + 1) as i32);
                    seqused_k.push(meta.seq_lens[orig_idx] as i32);

                    // Slot mapping: compute physical slot for this decode token.
                    let abs_pos = meta.tokens_before[orig_idx];
                    let block_idx = abs_pos / block_size;
                    let offset = abs_pos % block_size;
                    let block_ids = &meta.block_ids[orig_idx];
                    if block_idx < block_ids.len() {
                        slot_mapping.push((block_ids[block_idx] * block_size + offset) as i64);
                    } else {
                        slot_mapping.push(-1i64);
                    }

                    // Block table row.
                    for (j, &bid) in block_ids.iter().enumerate() {
                        if j < max_blocks_per_seq {
                            block_table[out_idx * max_blocks_per_seq + j] = bid as i32;
                        }
                    }
                }

                // Pad to graph batch size.
                input_ids.resize(decode_graph_bs, 0);
                positions.resize(decode_graph_bs, 0);
                slot_mapping.resize(decode_graph_bs, -1i64);
                for _ in n_decode..decode_graph_bs {
                    cu_seqlens_q.push(n_decode as i32);
                }
                seqused_k.resize(decode_graph_bs, 1);

                let runner = self.graph_runner.as_ref().unwrap();
                let sk_bucket = mixed_decode_sk_bucket
                    .expect("mixed_decode_sk_bucket present when decode_graph_bs is Some");
                let replay_out = unsafe {
                    runner.replay(
                        decode_graph_bs,
                        sk_bucket,
                        &input_ids,
                        &positions,
                        &slot_mapping,
                        &cu_seqlens_q,
                        &seqused_k,
                        &block_table,
                        self.config.block_size,
                        device,
                        false,
                    )
                }
                .map_err(|e| {
                    ExecutorError::WorkerExecution(format!("mixed decode graph replay: {e}"))
                })?;

                // Slice to real decode requests (discard padding rows).
                if decode_graph_bs > n_decode {
                    replay_out.logits.narrow_dim0(0, n_decode)
                } else {
                    replay_out.logits
                }
            };

            // --- Prefill pass: run through eager forward ---
            let prefill_logits = {
                let meta = &prepared.attn_meta;

                // Build flat token/position arrays for prefill requests.
                let mut pf_token_ids: Vec<u32> = Vec::new();
                let mut pf_positions: Vec<u32> = Vec::new();
                let mut pf_q_lens: Vec<usize> = Vec::new();
                let mut pf_seq_lens: Vec<usize> = Vec::new();
                let mut pf_block_ids: Vec<Vec<usize>> = Vec::new();
                let mut pf_tokens_before: Vec<usize> = Vec::new();
                let mut pf_query_start_loc: Vec<usize> = vec![0];

                let mut pf_offset = 0usize;
                for &orig_idx in &prefill_indices {
                    let q_len = meta.q_lens[orig_idx];
                    let token_start = meta.query_start_loc[orig_idx];
                    pf_token_ids.extend_from_slice(
                        &prepared.flat_token_ids[token_start..token_start + q_len],
                    );
                    pf_positions.extend_from_slice(
                        &prepared.flat_positions[token_start..token_start + q_len],
                    );
                    pf_q_lens.push(q_len);
                    pf_seq_lens.push(meta.seq_lens[orig_idx]);
                    pf_block_ids.push(meta.block_ids[orig_idx].clone());
                    pf_tokens_before.push(meta.tokens_before[orig_idx]);
                    pf_offset += q_len;
                    pf_query_start_loc.push(pf_offset);
                }
                let pf_total_tokens = pf_token_ids.len();

                let pf_meta = scratchy_core_model::AttentionMetadata::new(
                    n_prefill,
                    pf_total_tokens,
                    pf_query_start_loc,
                    pf_q_lens,
                    pf_seq_lens,
                    pf_block_ids,
                    pf_tokens_before,
                    vec![true; n_prefill],
                    prefill_indices
                        .iter()
                        .map(|&i| meta.req_ids[i].clone())
                        .collect(),
                );

                let gpu_input_ids = Self::h2d_u32(&pf_token_ids, device)?;
                let gpu_positions = Self::h2d_u32(&pf_positions, device)?;

                let full_block_size = kv_cache.max_block_size();
                let (
                    cu_seqlens_q,
                    seqused_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    slot_mappings,
                    block_tables,
                ) = Self::build_attention_tensors(&pf_meta, block_size, full_block_size, device)?;
                let slot_mapping = &slot_mappings[0];
                let block_table_gpu = &block_tables[0];
                // Per-group views for hybrid SWA (gemma4). len == 1 (uniform) →
                // `None`, and the attention ops use the scalar block_table.
                let bt_views: Vec<TensorView> = block_tables.iter().map(|t| t.view()).collect();
                let sm_views: Vec<TensorView> = slot_mappings.iter().map(|t| t.view()).collect();
                let mixed_hybrid_kv = if bt_views.len() > 1 {
                    Some(HybridKvInputs {
                        block_tables: &bt_views,
                        slot_mappings: &sm_views,
                    })
                } else {
                    None
                };

                // last_token_indices: for each prefill request, index of last token in flat array.
                let last_token_indices = if n_prefill < pf_total_tokens {
                    let mut indices = Vec::with_capacity(n_prefill);
                    let mut off = 0u32;
                    for &q in &pf_meta.q_lens {
                        indices.push(off + q as u32 - 1);
                        off += q as u32;
                    }
                    Some(Self::h2d_u32(&indices, device)?)
                } else {
                    None
                };

                unsafe {
                    model.forward(
                        gpu_input_ids.view(),
                        gpu_positions.view(),
                        slot_mapping.view(),
                        cu_seqlens_q.view(),
                        seqused_k.view(),
                        block_table_gpu.view(),
                        max_seqlen_q,
                        max_seqlen_k,
                        kv_cache,
                        device,
                        last_token_indices.as_ref().map(|t| t.view()),
                        // Mixed-batch eager prefill: MM splice not yet plumbed
                        // through this slow path. Single-prefill flows go
                        // through the eager main-forward branch below where
                        // mm_inputs is built. A VL prefill that lands in the
                        // mixed path (rare — would require concurrent decode
                        // reqs at first-image-prefill step) is a follow-up.
                        None,
                        // Hybrid (GDN) prefill — same logic as MM above. The
                        // gdn_inputs ctx is constructed in the eager
                        // main-forward branch below from `self.gdn_state`
                        // + `self.gdn_pending`. A hybrid-arch prefill landing
                        // in this mixed path (would require concurrent decode
                        // reqs at the same step) is a follow-up.
                        None,
                        // Hybrid SWA (gemma4): wired below at this call site once
                        // build_attention_tensors returns per-group tables.
                        mixed_hybrid_kv.as_ref(),
                    )
                }
            };

            // --- Merge logits in original request order ---
            // Allocate [num_reqs, vocab_size] and scatter decode/prefill logits.
            let row_bytes = vocab_size * decode_logits.dtype().size_bytes();
            let merged_owned = device
                .caching
                .alloc_tensor(&[num_reqs, vocab_size], decode_logits.dtype());
            let merged = merged_owned.as_gpu_tensor();

            // Copy decode logits into merged at their original positions.
            for (src_row, &orig_idx) in decode_indices.iter().enumerate() {
                let src = unsafe { decode_logits.raw_ptr().add(src_row * row_bytes) };
                let dst = unsafe { merged.raw_ptr().add(orig_idx * row_bytes) };
                unsafe { driver::memcpy_dtod_async(dst, src, row_bytes, device.compute_stream) }
                    .map_err(|e| {
                        ExecutorError::WorkerExecution(format!("mixed merge decode D2D: {e}"))
                    })?;
            }

            // Copy prefill logits into merged at their original positions.
            for (src_row, &orig_idx) in prefill_indices.iter().enumerate() {
                let src = unsafe { prefill_logits.raw_ptr().add(src_row * row_bytes) };
                let dst = unsafe { merged.raw_ptr().add(orig_idx * row_bytes) };
                unsafe { driver::memcpy_dtod_async(dst, src, row_bytes, device.compute_stream) }
                    .map_err(|e| {
                        ExecutorError::WorkerExecution(format!("mixed merge prefill D2D: {e}"))
                    })?;
            }

            // Invalidate graph metadata since we changed batch composition.
            self.last_graph_batch_size = None;
            self.graph_metadata_valid = false;

            // Drop the sub-logits so their memory returns to the caching allocator.
            // (decode_logits is a view into graph output — not owned. prefill_logits is
            // an OwnedTensor from forward(). merged_owned keeps the merged allocation.)
            let _ = decode_logits;
            drop(prefill_logits);

            // Fall through to sampling with merged logits.
            let logits = merged;

            // Update logits processors after forward, before sampling.
            Self::update_logits_processors(
                &LogitsUpdateCtx {
                    batch_changed: self.batch_changed,
                    num_reqs,
                    sampling_params_map: &self.sampling_params_map,
                    token_buffers: &self.token_buffers,
                    batch_req_ids: &self.batch_req_ids,
                },
                self.logits_pipeline.as_mut(),
                #[cfg(feature = "guided-decoding")]
                &mut self.grammar_states,
                &mut self.grammar_processor,
                &mut self.allowed_token_ids_processor,
                &mut self.seal_pad_processor,
                device,
            );

            // GPU sampling for mixed prefill+decode merged logits.
            return Self::gpu_sample_and_finalize(
                &self.sampling_params_map,
                #[cfg(feature = "guided-decoding")]
                &mut self.grammar_states,
                &self.grammar_processor,
                &self.allowed_token_ids_processor,
                &self.seal_pad_processor,
                self.logits_pipeline.as_ref(),
                &mut self.seeded_rngs,
                &self.host_staging,
                &mut self.input_batch,
                &mut self.token_buffers,
                &self.prompt_lengths,
                logits,
                prepared,
                device,
                all_greedy,
                vocab_size,
            );
        }

        // Check if any request needs logprobs, grammar, or logit processors
        // (these require the full GPU sampling pipeline instead of in-graph argmax).
        // Uses request-level state (sampling_params_map, grammar_states) rather than
        // processor GPU state, since processors are updated after forward/graph replay.
        let any_needs_full_sampling = prepared.req_inputs.iter().any(|r| {
            self.sampling_params_map.get(&r.req_id).is_some_and(|p| {
                p.logprobs.is_some()
                    || p.logit_bias.is_some()
                    || p.frequency_penalty != 0.0
                    || p.presence_penalty != 0.0
                    || p.repetition_penalty != 1.0
                    || p.min_tokens > 0
                    || p.bad_words_token_ids.is_some()
                    || p.allowed_token_ids.is_some()
                    || p.seal
            })
        }) || {
            #[cfg(feature = "guided-decoding")]
            {
                !self.grammar_states.is_empty()
            }
            #[cfg(not(feature = "guided-decoding"))]
            {
                false
            }
        };

        // Piecewise CUDA-graph dispatch and the Qwen3Next-specific GDN
        // state-pool lookup were removed alongside the hand-written CUDA model
        // forwards; everything below uses monolithic graph or eager.

        // -----------------------------------------------------------------------
        // Full CUDA Graph Path: Monolithic graph with fixed split-K
        // -----------------------------------------------------------------------
        if use_graph && all_greedy && !any_needs_full_sampling {
            // Fast path: CUDA graph with in-graph argmax. No separate sampling
            // kernel launch — argmax + D2D scatter are captured in the graph.
            let graph_bs = graph_bs.unwrap();
            let sk_bucket =
                decode_sk_bucket.expect("decode_sk_bucket present when graph_bs is Some");
            let meta = &prepared.attn_meta;
            let staging = self.host_staging.as_ref();

            let replay_out = if self.graph_metadata_valid
                && self.last_graph_batch_size == Some((graph_bs, sk_bucket))
            {
                // GPU-side metadata update: positions, slot_mapping, seqused_k
                // are incremented on GPU in a single kernel. Only block_table is
                // H2D-copied when blocks changed. input_ids were scattered by the
                // previous graph replay's in-graph argmax.
                let new_bt = if blocks_changed {
                    let bt =
                        unsafe { staging.unwrap().fill_block_table(&meta.block_ids, graph_bs) };
                    Some(bt)
                } else {
                    None
                };

                let max_seqlen_k_step = meta.seq_lens.iter().copied().max().unwrap_or(0);
                let runner = self.graph_runner.as_ref().unwrap();
                unsafe {
                    runner.replay_decode_fast(
                        graph_bs,
                        sk_bucket,
                        None, // input_ids already scattered by previous graph
                        new_bt,
                        block_size,
                        max_seqlen_k_step,
                        device,
                    )
                }
                .map_err(|e| {
                    ExecutorError::WorkerExecution(format!("graph replay_decode_fast: {e}"))
                })?
            } else if let Some(stg) = staging {
                // First step for this batch or batch composition changed:
                // full H2D of all metadata via pinned staging buffers.
                unsafe {
                    let ids = stg.input_ids.slice_mut::<u32>(graph_bs);
                    ids[..num_reqs].copy_from_slice(&prepared.flat_token_ids);
                    ids[num_reqs..].fill(0);

                    let pos = stg.positions.slice_mut::<u32>(graph_bs);
                    pos[..num_reqs].copy_from_slice(&prepared.flat_positions);
                    pos[num_reqs..].fill(0);

                    let cu_q = stg.cu_seqlens_q.slice_mut::<i32>(graph_bs + 1);
                    for (i, v) in cu_q[..=num_reqs].iter_mut().enumerate() {
                        *v = i as i32;
                    }
                    cu_q[(num_reqs + 1)..].fill(num_reqs as i32);

                    // seqused_k: per-sequence K lengths [graph_bs].
                    let sk = stg.seqused_k.slice_mut::<i32>(graph_bs);
                    for (i, &sl) in meta.seq_lens.iter().enumerate() {
                        sk[i] = sl as i32;
                    }
                    // Padded slots: use 1 (dummy seqs with 1 K token).
                    for s in &mut sk[num_reqs..] {
                        *s = 1;
                    }

                    let sm = stg.slot_mapping.slice_mut::<i64>(graph_bs);
                    for (i, slot) in sm[..num_reqs].iter_mut().enumerate() {
                        let abs_pos = meta.tokens_before[i];
                        let block_idx = abs_pos / block_size;
                        let offset = abs_pos % block_size;
                        let block_ids = &meta.block_ids[i];
                        if block_idx < block_ids.len() {
                            *slot = (block_ids[block_idx] * block_size + offset) as i64;
                        } else {
                            *slot = -1i64;
                        }
                    }
                    sm[num_reqs..].fill(-1i64);

                    let bt = stg.fill_block_table(&meta.block_ids, graph_bs);

                    let skip_input_ids = self.last_graph_batch_size == Some((graph_bs, sk_bucket))
                        && num_reqs == graph_bs;

                    let runner = self.graph_runner.as_ref().unwrap();
                    runner.replay(
                        graph_bs,
                        sk_bucket,
                        stg.input_ids.slice::<u32>(graph_bs),
                        stg.positions.slice::<u32>(graph_bs),
                        stg.slot_mapping.slice::<i64>(graph_bs),
                        stg.cu_seqlens_q.slice::<i32>(graph_bs + 1),
                        stg.seqused_k.slice::<i32>(graph_bs),
                        bt,
                        block_size,
                        device,
                        skip_input_ids,
                    )
                }
                .map_err(|e| ExecutorError::WorkerExecution(format!("graph replay: {e}")))?
            } else {
                // Fallback: no pinned staging (shouldn't happen but safe).
                let mut input_ids = prepared.flat_token_ids.clone();
                input_ids.resize(graph_bs, 0);
                let mut positions = prepared.flat_positions.clone();
                positions.resize(graph_bs, 0);

                let mut cu_seqlens_q: Vec<i32> = (0..=num_reqs as i32).collect();
                for _ in num_reqs..graph_bs {
                    cu_seqlens_q.push(num_reqs as i32);
                }

                // seqused_k: per-sequence K lengths [graph_bs].
                let mut seqused_k: Vec<i32> = meta.seq_lens.iter().map(|&sl| sl as i32).collect();
                // Padded slots: use 1 (dummy seqs with 1 K token).
                seqused_k.resize(graph_bs, 1);

                let mut slot_mapping = Vec::with_capacity(graph_bs);
                for i in 0..num_reqs {
                    let abs_pos = meta.tokens_before[i];
                    let block_idx = abs_pos / block_size;
                    let offset = abs_pos % block_size;
                    let block_ids = &meta.block_ids[i];
                    if block_idx < block_ids.len() {
                        slot_mapping.push((block_ids[block_idx] * block_size + offset) as i64);
                    } else {
                        slot_mapping.push(-1i64);
                    }
                }
                slot_mapping.resize(graph_bs, -1i64);

                let mut block_table = vec![0i32; graph_bs * max_blocks_per_seq];
                for (i, blocks) in meta.block_ids.iter().enumerate() {
                    for (j, &bid) in blocks.iter().enumerate() {
                        if j < max_blocks_per_seq {
                            block_table[i * max_blocks_per_seq + j] = bid as i32;
                        }
                    }
                }

                let skip_input_ids = self.last_graph_batch_size == Some((graph_bs, sk_bucket))
                    && num_reqs == graph_bs;

                let runner = self.graph_runner.as_ref().unwrap();
                unsafe {
                    runner.replay(
                        graph_bs,
                        sk_bucket,
                        &input_ids,
                        &positions,
                        &slot_mapping,
                        &cu_seqlens_q,
                        &seqused_k,
                        &block_table,
                        block_size,
                        device,
                        skip_input_ids,
                    )
                }
                .map_err(|e| ExecutorError::WorkerExecution(format!("graph replay: {e}")))?
            };

            // Record that graph buffers now have valid metadata for next step.
            self.last_graph_batch_size = Some((graph_bs, sk_bucket));
            self.graph_metadata_valid = true;

            // Deferred D2H: enqueue async copy on transfer stream, return
            // immediately without blocking on the GPU. The token IDs are
            // resolved lazily by the main thread (via D2HResolver) and the
            // commit_step is deferred to the start of the next execute_model.
            if let Some(ref mut stg) = self.host_staging {
                let buf_idx = stg.token_buf_idx;
                Self::d2h_token_ids_async(stg, buf_idx, &replay_out.token_ids, num_reqs, device)?;

                // Capture info needed for deferred commit_step.
                let req_ids: Vec<String> = prepared
                    .req_inputs
                    .iter()
                    .map(|r| r.req_id.clone())
                    .collect();
                let token_counts: Vec<usize> =
                    prepared.req_inputs.iter().map(|r| r.token_count).collect();
                let has_spec_tokens: Vec<bool> = prepared
                    .req_inputs
                    .iter()
                    .map(|r| !r.spec_token_ids.is_empty())
                    .collect();

                self.input_batch.reclaim_buffers(prepared);

                self.pending_commit = Some(PendingCommit {
                    buf_idx,
                    num_reqs,
                    req_ids: req_ids.clone(),
                    token_counts,
                    has_spec_tokens,
                });

                // Toggle double-buffer for next step.
                stg.token_buf_idx ^= 1;

                // Build deferred output: the resolver closure syncs the D2H
                // event and reads token IDs from the pinned host buffer.
                // Cast raw pointers to usize for Send safety (pinned buffer
                // and event outlive the closure — see PendingCommit safety doc).
                let event_addr = device.d2h_done as usize;
                let buf_addr = stg.host_token_ids[buf_idx].ptr() as usize;
                return Ok(ModelRunnerOutput::deferred(
                    req_ids,
                    Box::new(move || unsafe {
                        driver::event_synchronize_raw(event_addr).expect("D2H event sync failed");
                        std::slice::from_raw_parts(buf_addr as *const u32, num_reqs).to_vec()
                    }),
                ));
            }

            // No pinned staging — fall back to synchronous D2H.
            return Self::finalize_d2h_and_commit(
                &replay_out.token_ids,
                num_reqs,
                prepared,
                device,
                &self.host_staging,
                0,
                &mut self.input_batch,
                &mut self.token_buffers,
                &self.prompt_lengths,
            );
        }

        // Non-greedy graph path or eager path: need separate sampling.
        // IMPORTANT: skip graph replay when any logits processor or grammar is
        // active. CUDA graphs replay kernels at captured memory addresses —
        // intermediate forward-pass buffers reuse the same addresses as during
        // `_logits_owned` keeps the OwnedTensor alive for eager-forward paths
        // so the GPU memory backing `logits` survives until sampling completes.
        // `_mm_holder` keeps the vision-encoder OwnedTensor alive for the
        // same reason on the MM splice path (text-only batches: stays None).
        let mut _mm_holder: Option<(OwnedTensor, Vec<crate::EmbedPatch>)> = None;
        let (_logits_owned, logits) = if use_graph {
            // CUDA graph replay (non-greedy: in-graph argmax result is
            // discarded; we re-sample with temperature on the logits).
            let graph_bs = graph_bs.unwrap();
            let sk_bucket =
                decode_sk_bucket.expect("decode_sk_bucket present when graph_bs is Some");
            let meta = &prepared.attn_meta;
            let staging = self.host_staging.as_ref();

            let replay_out = if self.graph_metadata_valid
                && self.last_graph_batch_size == Some((graph_bs, sk_bucket))
            {
                // Fast path: GPU-side metadata update (same as greedy).
                // Only input_ids must be H2D'd (no in-graph argmax scatter for non-greedy).
                let new_bt = if blocks_changed {
                    let bt =
                        unsafe { staging.unwrap().fill_block_table(&meta.block_ids, graph_bs) };
                    Some(bt)
                } else {
                    None
                };

                // Must H2D input_ids since non-greedy doesn't use in-graph argmax scatter.
                let input_ids_slice = if let Some(stg) = staging {
                    unsafe {
                        let ids = stg.input_ids.slice_mut::<u32>(graph_bs);
                        ids[..num_reqs].copy_from_slice(&prepared.flat_token_ids);
                        ids[num_reqs..].fill(0);
                        stg.input_ids.slice::<u32>(graph_bs)
                    }
                } else {
                    let mut ids = prepared.flat_token_ids.clone();
                    ids.resize(graph_bs, 0);
                    ids.leak()
                };

                let max_seqlen_k_step = meta.seq_lens.iter().copied().max().unwrap_or(0);
                let runner = self.graph_runner.as_ref().unwrap();
                unsafe {
                    runner.replay_decode_fast(
                        graph_bs,
                        sk_bucket,
                        Some(input_ids_slice),
                        new_bt,
                        block_size,
                        max_seqlen_k_step,
                        device,
                    )
                }
                .map_err(|e| {
                    ExecutorError::WorkerExecution(format!("graph replay_decode_fast: {e}"))
                })?
            } else if let Some(stg) = staging {
                // First step or batch composition changed: full H2D via pinned staging.
                unsafe {
                    let ids = stg.input_ids.slice_mut::<u32>(graph_bs);
                    ids[..num_reqs].copy_from_slice(&prepared.flat_token_ids);
                    ids[num_reqs..].fill(0);

                    let pos = stg.positions.slice_mut::<u32>(graph_bs);
                    pos[..num_reqs].copy_from_slice(&prepared.flat_positions);
                    pos[num_reqs..].fill(0);

                    let cu_q = stg.cu_seqlens_q.slice_mut::<i32>(graph_bs + 1);
                    for (i, v) in cu_q[..=num_reqs].iter_mut().enumerate() {
                        *v = i as i32;
                    }
                    cu_q[(num_reqs + 1)..].fill(num_reqs as i32);

                    // seqused_k: per-sequence K lengths [graph_bs].
                    let sk = stg.seqused_k.slice_mut::<i32>(graph_bs);
                    for (i, &sl) in meta.seq_lens.iter().enumerate() {
                        sk[i] = sl as i32;
                    }
                    for s in &mut sk[num_reqs..] {
                        *s = 1;
                    }

                    let sm = stg.slot_mapping.slice_mut::<i64>(graph_bs);
                    for (i, slot) in sm[..num_reqs].iter_mut().enumerate() {
                        let abs_pos = meta.tokens_before[i];
                        let block_idx = abs_pos / block_size;
                        let offset = abs_pos % block_size;
                        let block_ids = &meta.block_ids[i];
                        if block_idx < block_ids.len() {
                            *slot = (block_ids[block_idx] * block_size + offset) as i64;
                        } else {
                            *slot = -1i64;
                        }
                    }
                    sm[num_reqs..].fill(-1i64);

                    let bt = stg.fill_block_table(&meta.block_ids, graph_bs);

                    let runner = self.graph_runner.as_ref().unwrap();
                    runner.replay(
                        graph_bs,
                        sk_bucket,
                        stg.input_ids.slice::<u32>(graph_bs),
                        stg.positions.slice::<u32>(graph_bs),
                        stg.slot_mapping.slice::<i64>(graph_bs),
                        stg.cu_seqlens_q.slice::<i32>(graph_bs + 1),
                        stg.seqused_k.slice::<i32>(graph_bs),
                        bt,
                        block_size,
                        device,
                        false,
                    )
                }
                .map_err(|e| ExecutorError::WorkerExecution(format!("graph replay: {e}")))?
            } else {
                // Fallback: no pinned staging.
                let mut input_ids = prepared.flat_token_ids.clone();
                input_ids.resize(graph_bs, 0);
                let mut positions = prepared.flat_positions.clone();
                positions.resize(graph_bs, 0);

                let mut cu_seqlens_q: Vec<i32> = (0..=num_reqs as i32).collect();
                for _ in num_reqs..graph_bs {
                    cu_seqlens_q.push(num_reqs as i32);
                }

                // seqused_k: per-sequence K lengths [graph_bs].
                let mut seqused_k: Vec<i32> = meta.seq_lens.iter().map(|&sl| sl as i32).collect();
                // Padded slots: use 1 (dummy seqs with 1 K token).
                seqused_k.resize(graph_bs, 1);

                let mut slot_mapping = Vec::with_capacity(graph_bs);
                for i in 0..num_reqs {
                    let abs_pos = meta.tokens_before[i];
                    let block_idx = abs_pos / block_size;
                    let offset = abs_pos % block_size;
                    let block_ids = &meta.block_ids[i];
                    if block_idx < block_ids.len() {
                        slot_mapping.push((block_ids[block_idx] * block_size + offset) as i64);
                    } else {
                        slot_mapping.push(-1i64);
                    }
                }
                slot_mapping.resize(graph_bs, -1i64);

                let mut block_table = vec![0i32; graph_bs * max_blocks_per_seq];
                for (i, blocks) in meta.block_ids.iter().enumerate() {
                    for (j, &bid) in blocks.iter().enumerate() {
                        if j < max_blocks_per_seq {
                            block_table[i * max_blocks_per_seq + j] = bid as i32;
                        }
                    }
                }

                let runner = self.graph_runner.as_ref().unwrap();
                unsafe {
                    runner.replay(
                        graph_bs,
                        sk_bucket,
                        &input_ids,
                        &positions,
                        &slot_mapping,
                        &cu_seqlens_q,
                        &seqused_k,
                        &block_table,
                        block_size,
                        device,
                        false,
                    )
                }
                .map_err(|e| ExecutorError::WorkerExecution(format!("graph replay: {e}")))?
            };

            // Track metadata validity for next step (works for non-greedy too).
            self.last_graph_batch_size = Some((graph_bs, sk_bucket));
            self.graph_metadata_valid = true;

            // Slice logits to only the real requests (discard padded rows).
            let logits = if graph_bs > num_reqs {
                replay_out.logits.narrow_dim0(0, num_reqs)
            } else {
                replay_out.logits
            };
            (None, logits)
        } else if use_piecewise {
            // -------------------------------------------------------------
            // Piecewise CUDA-graph decode (tp>1 path). Build padded inputs
            // matching the captured `pw_bs`, H2D into the runner's stable
            // buffers, replay segments with eager NCCL between them.
            // -------------------------------------------------------------
            #[cfg(feature = "nccl")]
            let result = {
                let pw_bs = piecewise_bs.unwrap();
                let meta = &prepared.attn_meta;

                // Build padded host vecs (same shape as the eager fallback
                // builds for monolithic graph replay below — easier to
                // reuse the existing pattern than to plumb staging here).
                let mut input_ids = prepared.flat_token_ids.clone();
                input_ids.resize(pw_bs, 0);
                let mut positions = prepared.flat_positions.clone();
                positions.resize(pw_bs, 0);
                let mut cu_seqlens_q: Vec<i32> = (0..=num_reqs as i32).collect();
                for _ in num_reqs..pw_bs {
                    cu_seqlens_q.push(num_reqs as i32);
                }
                let mut seqused_k: Vec<i32> = meta.seq_lens.iter().map(|&sl| sl as i32).collect();
                seqused_k.resize(pw_bs, 1);
                let mut slot_mapping = Vec::with_capacity(pw_bs);
                for i in 0..num_reqs {
                    let abs_pos = meta.tokens_before[i];
                    let block_idx = abs_pos / block_size;
                    let offset = abs_pos % block_size;
                    let block_ids = &meta.block_ids[i];
                    if block_idx < block_ids.len() {
                        slot_mapping.push((block_ids[block_idx] * block_size + offset) as i64);
                    } else {
                        slot_mapping.push(-1i64);
                    }
                }
                slot_mapping.resize(pw_bs, -1i64);
                let mut block_table = vec![0i32; pw_bs * max_blocks_per_seq];
                for (i, blocks) in meta.block_ids.iter().enumerate() {
                    for (j, &bid) in blocks.iter().enumerate() {
                        if j < max_blocks_per_seq {
                            block_table[i * max_blocks_per_seq + j] = bid as i32;
                        }
                    }
                }

                let pdr = self.piecewise_decode.as_ref().unwrap();
                unsafe {
                    pdr.replay(
                        pw_bs,
                        &input_ids,
                        &positions,
                        &slot_mapping,
                        &cu_seqlens_q,
                        &seqused_k,
                        &block_table,
                        kv_cache,
                        model,
                        device,
                    )
                }
                .map_err(|e| ExecutorError::WorkerExecution(format!("piecewise replay: {e}")))?
            };
            // Without nccl, `piecewise_bs` is hardcoded to None at the
            // dispatch site (see the `#[cfg(not(feature = "nccl"))]`
            // arm of `piecewise_bs:`), so `use_piecewise` is always
            // false and this branch is statically unreachable. Diverge
            // cleanly so clippy doesn't see a `let result = unreachable!()`
            // followed by dead-but-typed code.
            #[cfg(not(feature = "nccl"))]
            unreachable!("use_piecewise true requires nccl feature; gating bug");

            #[cfg(feature = "nccl")]
            {
                // logits is [pw_bs, vocab]; narrow to real reqs if padded.
                let logits_full = result;
                let logits_view = if num_reqs < piecewise_bs.unwrap() {
                    logits_full.as_gpu_tensor().narrow_dim0(0, num_reqs)
                } else {
                    logits_full.as_gpu_tensor()
                };
                self.last_graph_batch_size = None;
                self.graph_metadata_valid = false;
                (Some(logits_full), logits_view)
            }
        } else {
            // Non-decode path: try prefill graph, fall back to eager.
            self.last_graph_batch_size = None;
            self.graph_metadata_valid = false;

            // Check if we can use a prefill graph: single request, fresh prefill
            // (tokens_before == 0 means q_len == seq_len, so the model uses contiguous
            // FA2 — not paged — which is safe to capture in a CUDA graph).
            // MM-bearing reqs MUST take the eager path: the captured prefill graph
            // does not include the vision encoder + Embed splice, so replaying it
            // would skip vision_forward entirely and hand the decoder placeholder
            // tokens with no visual content (hallucinated output).
            let meta = &prepared.attn_meta;
            let req_has_mm = self.mm_data_buffers.contains_key(&meta.req_ids[0]);

            // At tp>1: piecewise prefill replay (NCCL eager between captured segments).
            // At tp=1: monolithic prefill replay.
            #[cfg(feature = "nccl")]
            let use_piecewise_prefill_replay = num_reqs == 1
                && meta.tokens_before[0] == 0
                && !req_has_mm
                && self
                    .piecewise_prefill
                    .as_ref()
                    .and_then(|r| r.nearest_size(total_tokens))
                    .is_some();
            #[cfg(not(feature = "nccl"))]
            let use_piecewise_prefill_replay = false;

            let use_prefill_graph = !use_piecewise_prefill_replay
                && num_reqs == 1
                && meta.tokens_before[0] == 0
                && !req_has_mm
                && self
                    .prefill_graph_runner
                    .as_ref()
                    .and_then(|r| r.nearest_graph_size(total_tokens))
                    .is_some();

            if use_piecewise_prefill_replay {
                #[cfg(feature = "nccl")]
                {
                    let ppr = self.piecewise_prefill.as_ref().unwrap();
                    let padded = ppr.nearest_size(total_tokens).unwrap();

                    let mut block_table = vec![0i32; max_blocks_per_seq];
                    for (j, &bid) in meta.block_ids[0].iter().enumerate() {
                        if j < max_blocks_per_seq {
                            block_table[j] = bid as i32;
                        }
                    }

                    let mut slot_mapping = Vec::with_capacity(total_tokens);
                    let block_ids = &meta.block_ids[0];
                    for t in 0..total_tokens {
                        let abs_pos = meta.tokens_before[0] + t;
                        let block_idx = abs_pos / block_size;
                        let offset = abs_pos % block_size;
                        if block_idx < block_ids.len() {
                            slot_mapping.push((block_ids[block_idx] * block_size + offset) as i64);
                        } else {
                            slot_mapping.push(-1i64);
                        }
                    }

                    let last_token_idx = (total_tokens - 1) as u32;

                    let logits = unsafe {
                        ppr.replay(
                            padded,
                            &prepared.flat_token_ids,
                            &prepared.flat_positions,
                            &slot_mapping,
                            meta.seq_lens[0],
                            &block_table,
                            last_token_idx,
                            block_size,
                            kv_cache,
                            model,
                            device,
                        )
                    }
                    .map_err(|e| {
                        ExecutorError::WorkerExecution(format!("piecewise prefill replay: {e}"))
                    })?;

                    let logits_view = logits.as_gpu_tensor();
                    (Some(logits), logits_view)
                }
                #[cfg(not(feature = "nccl"))]
                {
                    unreachable!(
                        "use_piecewise_prefill_replay true requires nccl feature; gating bug"
                    )
                }
            } else if use_prefill_graph {
                let padded = self
                    .prefill_graph_runner
                    .as_ref()
                    .unwrap()
                    .nearest_graph_size(total_tokens)
                    .unwrap();

                // Build block_table padded to MAX_BLOCKS_PER_SEQ.
                let mut block_table = vec![0i32; max_blocks_per_seq];
                for (j, &bid) in meta.block_ids[0].iter().enumerate() {
                    if j < max_blocks_per_seq {
                        block_table[j] = bid as i32;
                    }
                }

                // Build slot_mapping for real tokens.
                let mut slot_mapping = Vec::with_capacity(total_tokens);
                let block_ids = &meta.block_ids[0];
                for t in 0..total_tokens {
                    let abs_pos = meta.tokens_before[0] + t;
                    let block_idx = abs_pos / block_size;
                    let offset = abs_pos % block_size;
                    if block_idx < block_ids.len() {
                        slot_mapping.push((block_ids[block_idx] * block_size + offset) as i64);
                    } else {
                        slot_mapping.push(-1i64);
                    }
                }

                let last_token_idx = (total_tokens - 1) as u32;

                let replay_out = unsafe {
                    self.prefill_graph_runner.as_ref().unwrap().replay(
                        padded,
                        &prepared.flat_token_ids,
                        &prepared.flat_positions,
                        &slot_mapping,
                        meta.seq_lens[0],
                        &block_table,
                        last_token_idx,
                        block_size,
                        device,
                    )
                }
                .map_err(|e| {
                    ExecutorError::WorkerExecution(format!("prefill graph replay: {e}"))
                })?;

                (None, replay_out.logits)
            } else {
                // Eager forward path (multi-request prefill or uncaptured size).
                // Caching allocator: no reset needed — tensors freed on drop.

                // DEBUG: sync before forward to isolate errors from previous steps.
                /* {
                    let any_spec = prepared
                        .req_inputs
                        .iter()
                        .any(|r| !r.spec_token_ids.is_empty());
                    if any_spec {
                        unsafe { crate::driver::stream_synchronize(device.compute_stream) }
                            .map_err(|e| {
                                ExecutorError::WorkerExecution(format!(
                                    "eager pre-forward sync (spec batch): {e}"
                                ))
                            })?;
                        let meta = &prepared.attn_meta;
                        tracing::warn!(
                            "spec decode eager forward: num_reqs={}, total_tokens={}, q_lens={:?}, \
                             seq_lens={:?}, tokens_before={:?}, block_ids_lens={:?}",
                            meta.num_reqs,
                            meta.total_tokens,
                            meta.q_lens,
                            meta.seq_lens,
                            meta.tokens_before,
                            meta.block_ids.iter().map(|b| b.len()).collect::<Vec<_>>(),
                        );
                    }
                } */

                let gpu_input_ids = Self::h2d_u32(&prepared.flat_token_ids, device)?;
                let gpu_positions = Self::h2d_u32(&prepared.flat_positions, device)?;

                let full_block_size = kv_cache.max_block_size();
                let (
                    cu_seqlens_q,
                    seqused_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    slot_mappings,
                    block_tables,
                ) = Self::build_attention_tensors(
                    &prepared.attn_meta,
                    block_size,
                    full_block_size,
                    device,
                )?;
                let slot_mapping = &slot_mappings[0];
                let block_table = &block_tables[0];
                // Per-group views for hybrid SWA (gemma4). len == 1 (uniform) →
                // `None`, and the attention ops use the scalar block_table.
                let bt_views: Vec<TensorView> = block_tables.iter().map(|t| t.view()).collect();
                let sm_views: Vec<TensorView> = slot_mappings.iter().map(|t| t.view()).collect();
                let hybrid_kv = if bt_views.len() > 1 {
                    Some(HybridKvInputs {
                        block_tables: &bt_views,
                        slot_mappings: &sm_views,
                    })
                } else {
                    None
                };

                // For spec decode: skip last_token_indices so we get logits
                // for ALL token positions (needed for rejection sampling).
                // For normal batches: gather only the last token per request.
                let any_spec_decode = prepared
                    .req_inputs
                    .iter()
                    .any(|r| !r.spec_token_ids.is_empty());

                let last_token_indices = if any_spec_decode {
                    // Spec decode: need logits for all positions, not just last.
                    // Model returns [total_tokens, vocab_size].
                    None
                } else if num_reqs < total_tokens {
                    let mut indices = Vec::with_capacity(num_reqs);
                    let mut offset = 0u32;
                    for req_slice in &prepared.req_inputs {
                        indices.push(offset + req_slice.token_count as u32 - 1);
                        offset += req_slice.token_count as u32;
                    }
                    Some(Self::h2d_u32(&indices, device)?)
                } else {
                    None
                };

                {
                    // Run vision encoder for any MM-bearing req at its
                    // first prefill step; output rides into model.forward
                    // via mm_inputs and the `Instruction::Embed::eval`
                    // splice. The OwnedTensor lands in `_mm_holder` so
                    // its caching-allocator slot stays live until end of
                    // the surrounding fn — the splice D2D-copies on
                    // `device.compute_stream` and a parallel allocation
                    // could otherwise alias the same memory region
                    // before the splice kernel actually runs.
                    _mm_holder = unsafe {
                        Self::run_mm_vision_forward(model, &self.mm_data_buffers, &prepared, device)
                    };
                    // For image-bearing batches on MRoPE arches, override
                    // the 1D `gpu_positions` with a `[3, n_tokens]` u32
                    // tensor where image tokens carry their (T,H,W) grid
                    // coordinates (Python parity with
                    // `Qwen2VLForConditionalGeneration.get_input_positions_tensor`).
                    //
                    // Per-req seq-space MM info covers EVERY MM-bearing
                    // req in the batch — including ones with
                    // `tokens_before > 0` (cached prefix) where the
                    // encoder didn't run this step. Load-bearing for the
                    // cached-prefix path: the MRoPE positions written
                    // for the trailing new tokens must match the cursor
                    // the encoder advanced to when it wrote the cached
                    // KV blocks, otherwise attention misfires (Bug 3).
                    // Text-only batches get an empty outer `Vec` and keep
                    // the pre-built 1D `gpu_positions` (broadcast path).
                    let per_req_mm =
                        Self::build_per_req_mm_seq_info(model, &self.mm_data_buffers, &prepared);
                    // MRoPE override is per-arch — Qwen2-VL family
                    // packs (T, H, W) per token so MM tokens carry grid
                    // coords; Gemma3-MM / LLaVA-class use standard 1D
                    // RoPE in the text decoder and must keep the
                    // prebuilt sequence positions. The flag rides on
                    // `MultimodalForward::mm_metadata().mrope_positions`,
                    // declared per-arch on `pub const PROCESSOR:
                    // scratchy_vision::MmMetadata`.
                    // Single-variant enum — see `extract_mm_embeds`.
                    let CudaModel::Scratchy(fm) = model;
                    let mrope = fm
                        .mm
                        .as_ref()
                        .map(|m| m.mm_metadata().mrope_positions)
                        .unwrap_or(false);
                    let gpu_positions_2d = if mrope && !per_req_mm.is_empty() {
                        let pos_2d = Self::build_mrope_positions_2d(&prepared, &per_req_mm);
                        let mut t = Self::h2d_u32(&pos_2d, device)?;
                        unsafe { t.reshape(&[3, total_tokens], GpuDType::U32) };
                        Some(t)
                    } else {
                        None
                    };
                    let positions_view = match &gpu_positions_2d {
                        Some(t) => t.view(),
                        None => gpu_positions.view(),
                    };
                    let mm_inputs: Option<MmForwardInputs<'_>> =
                        _mm_holder.as_ref().map(|(t, p)| MmForwardInputs {
                            mm_embeds: t.view(),
                            embed_patches: p.as_slice(),
                        });

                    // Hybrid (GDN) per-step inputs. Hoisted refs above:
                    // `gdn_state_ref` is `self.gdn_state.as_ref()`,
                    // `gdn_pending_taken` is the per-step slot/fresh pair
                    // built at the top of execute_model_inner. Both must be
                    // `Some` to construct GdnForwardInputs; either being
                    // `None` (text-only build / non-hybrid arch) disables
                    // the GDN path for this step. The two H2D OwnedTensors
                    // must outlive the forward call (their `view()`s
                    // borrow into them); they're bound here so they drop
                    // after the forward returns.
                    let (gpu_gdn_indices, gpu_gdn_is_fresh) =
                        match (gdn_state_ref, gdn_pending_taken.as_ref()) {
                            (Some(_), Some((idx, fresh))) => (
                                Some(Self::h2d_i32(idx, device)?),
                                Some(Self::h2d_u32(fresh, device)?),
                            ),
                            _ => (None, None),
                        };
                    let gdn_inputs: Option<GdnForwardInputs<'_>> = match (
                        gdn_state_ref,
                        gpu_gdn_indices.as_ref(),
                        gpu_gdn_is_fresh.as_ref(),
                    ) {
                        (Some(state), Some(idx_t), Some(fresh_t)) => Some(GdnForwardInputs {
                            state,
                            indices: idx_t.view(),
                            is_fresh: fresh_t.view(),
                        }),
                        _ => None,
                    };

                    let owned = unsafe {
                        model.forward(
                            gpu_input_ids.view(),
                            positions_view,
                            slot_mapping.view(),
                            cu_seqlens_q.view(),
                            seqused_k.view(),
                            block_table.view(),
                            max_seqlen_q,
                            max_seqlen_k,
                            kv_cache,
                            device,
                            last_token_indices.as_ref().map(|t| t.view()),
                            mm_inputs.as_ref(),
                            gdn_inputs.as_ref(),
                            hybrid_kv.as_ref(),
                        )
                    };
                    drop(gpu_positions_2d);
                    let logits = *owned;
                    (Some(owned), logits)
                }
            }
        };

        // DEBUG: sync after forward to catch forward errors.
        /* {
            let any_spec = prepared
                .req_inputs
                .iter()
                .any(|r| !r.spec_token_ids.is_empty());
            if any_spec {
                unsafe { crate::driver::stream_synchronize(device.compute_stream) }.map_err(
                    |e| {
                        ExecutorError::WorkerExecution(format!(
                            "eager post-forward sync (spec batch): {e}"
                        ))
                    },
                )?;
            }
        } */

        // _logits_owned (if Some) keeps the OwnedTensor alive until after sampling.
        // It will be dropped at the end of this function, returning memory to the
        // caching allocator.

        // Update logits processors after forward/graph replay, before sampling.
        // This ensures processor GPU tensor allocations don't collide with CUDA
        // graph intermediate addresses.
        Self::update_logits_processors(
            &LogitsUpdateCtx {
                batch_changed: self.batch_changed,
                num_reqs,
                sampling_params_map: &self.sampling_params_map,
                token_buffers: &self.token_buffers,
                batch_req_ids: &self.batch_req_ids,
            },
            self.logits_pipeline.as_mut(),
            #[cfg(feature = "guided-decoding")]
            &mut self.grammar_states,
            &mut self.grammar_processor,
            &mut self.allowed_token_ids_processor,
            &mut self.seal_pad_processor,
            device,
        );

        // GPU sampling: handles all cases — greedy, non-greedy, penalties,
        // grammar, logit_bias, logprobs — entirely on GPU. No CPU fallback.
        Self::gpu_sample_and_finalize(
            &self.sampling_params_map,
            #[cfg(feature = "guided-decoding")]
            &mut self.grammar_states,
            &self.grammar_processor,
            &self.allowed_token_ids_processor,
            &self.seal_pad_processor,
            self.logits_pipeline.as_ref(),
            &mut self.seeded_rngs,
            &self.host_staging,
            &mut self.input_batch,
            &mut self.token_buffers,
            &self.prompt_lengths,
            logits,
            prepared,
            device,
            all_greedy,
            vocab_size,
        )
    }
    /// Update logits processor pipeline, grammar, and allowed_token_ids state.
    /// Called after forward/graph replay and before sampling, so processor GPU
    /// tensor allocations don't conflict with CUDA graph intermediate addresses
    /// (matching Python vLLM's architecture).
    fn update_logits_processors(
        ctx: &LogitsUpdateCtx<'_>,
        logits_pipeline: Option<&mut LogitsProcessorPipeline>,
        #[cfg(feature = "guided-decoding")] grammar_states: &mut HashMap<
            String,
            scratchy_core_model::grammar::GrammarGuide,
        >,
        grammar_processor: &mut GrammarMaskProcessor,
        allowed_token_ids_processor: &mut AllowedTokenIdsProcessor,
        seal_pad_processor: &mut SealPadProcessor,
        device: &mut GpuDevice,
    ) {
        let batch_update = if ctx.batch_changed {
            Some(BatchUpdate {
                batch_size: ctx.num_reqs,
                added: Vec::new(),
                removed: Vec::new(),
            })
        } else {
            None
        };

        if let Some(pipeline) = logits_pipeline {
            pipeline.update_state(
                batch_update.as_ref(),
                ctx.sampling_params_map,
                ctx.token_buffers,
                ctx.batch_req_ids,
                device,
            );
        }

        #[cfg(feature = "guided-decoding")]
        {
            let grammar_reqs: Vec<(usize, Vec<u32>)> = ctx
                .batch_req_ids
                .iter()
                .enumerate()
                .filter_map(|(idx, rid)| {
                    grammar_states
                        .get_mut(rid)
                        .and_then(|g| g.allowed_tokens())
                        .map(|allowed| (idx, allowed))
                })
                .collect();
            let refs: Vec<(usize, &[u32])> = grammar_reqs
                .iter()
                .map(|(idx, v)| (*idx, v.as_slice()))
                .collect();
            grammar_processor.update_from_allowed_tokens(&refs, device);
        }
        #[cfg(not(feature = "guided-decoding"))]
        {
            let empty: Vec<(usize, &[u32])> = Vec::new();
            grammar_processor.update_from_allowed_tokens(&empty, device);
        }

        allowed_token_ids_processor.update_state(
            batch_update.as_ref(),
            ctx.sampling_params_map,
            ctx.token_buffers,
            ctx.batch_req_ids,
            device,
        );

        seal_pad_processor.update_state(
            batch_update.as_ref(),
            ctx.sampling_params_map,
            ctx.token_buffers,
            ctx.batch_req_ids,
            device,
        );
    }
} // end impl CudaWorker (execute_model_inner)

/// Read-only batch context passed to `update_logits_processors`.
#[cfg(feature = "cuda")]
struct LogitsUpdateCtx<'a> {
    batch_changed: bool,
    num_reqs: usize,
    sampling_params_map: &'a HashMap<String, SamplingParams>,
    token_buffers: &'a HashMap<String, Vec<u32>>,
    batch_req_ids: &'a [String],
}
#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    /// Verify KV cache budget matches Python vLLM's formula on real hardware.
    ///
    /// Python (gpu_worker.py):
    ///   requested = total_memory * gpu_memory_utilization
    ///   non_kv_cache = weights + peak_activations + non_torch + 150 MiB
    ///   available_kv = requested - non_kv_cache
    ///
    /// This test allocates known amounts of GPU memory, then calls
    /// `compute_available_kv_bytes` (the same formula CudaWorker uses)
    /// and verifies the result matches manual Python-style calculation.
    #[test]
    fn test_kv_cache_budget_matches_python_formula() {
        // L40S-like GPU: 46 GiB total.
        let total_memory: usize = 46 * 1024 * 1024 * 1024;

        // Simulate Qwen2.5-3B-like numbers.
        let model_weights: usize = 6 * 1024 * 1024 * 1024; // 6 GiB
        let peak_activations: usize = 3 * 1024 * 1024 * 1024; // 3 GiB
        let utilization = 0.9;

        let result =
            compute_available_kv_bytes(total_memory, model_weights, peak_activations, utilization);

        // Manually compute expected using Python's exact formula:
        //   requested = total * util
        //   non_kv = weights + peak + 150 MiB
        //   available = requested - non_kv
        let redundancy: usize = 150 * 1024 * 1024;
        let non_kv = model_weights + peak_activations + redundancy;
        let requested = (total_memory as f64 * utilization) as usize;
        let expected = requested.saturating_sub(non_kv);

        assert_eq!(
            result,
            expected,
            "KV budget {:.2} GiB != expected {:.2} GiB",
            result as f64 / 1_073_741_824.0,
            expected as f64 / 1_073_741_824.0,
        );

        // Sanity: ~32 GiB for KV, matching Python's "Available KV cache memory: 33.49 GiB"
        let result_gib = result as f64 / 1_073_741_824.0;
        assert!(
            result_gib > 30.0 && result_gib < 35.0,
            "Expected ~32 GiB KV budget for 46 GiB GPU with 3B model, got {:.1} GiB",
            result_gib
        );
    }

    /// Verify that utilization=0.5 gives less KV budget than 0.9.
    #[test]
    fn test_kv_cache_budget_respects_utilization() {
        let total: usize = 48 * 1024 * 1024 * 1024; // 48 GiB
        let weights: usize = 6 * 1024 * 1024 * 1024;
        let peak: usize = 3 * 1024 * 1024 * 1024;

        let budget_90 = compute_available_kv_bytes(total, weights, peak, 0.9);
        let budget_50 = compute_available_kv_bytes(total, weights, peak, 0.5);

        assert!(
            budget_50 < budget_90,
            "util=0.5 ({:.1} GiB) should give less KV than util=0.9 ({:.1} GiB)",
            budget_50 as f64 / 1_073_741_824.0,
            budget_90 as f64 / 1_073_741_824.0,
        );
    }

    /// Verify that when non_kv_cache exceeds requested memory, we get 0 (not underflow).
    #[test]
    fn test_kv_cache_budget_saturates_at_zero() {
        let total: usize = 8 * 1024 * 1024 * 1024; // 8 GiB total
        let weights: usize = 6 * 1024 * 1024 * 1024; // 6 GiB weights
        let peak: usize = 3 * 1024 * 1024 * 1024; // 3 GiB peak
        // non_kv = 6 + 3 + 0.15 = 9.15 GiB > requested = 8 * 0.9 = 7.2 GiB

        let budget = compute_available_kv_bytes(total, weights, peak, 0.9);
        assert_eq!(budget, 0, "should saturate at 0, not underflow");
    }

    // =========================================================================
    // Piecewise CUDA Graph Tests
    // =========================================================================

    #[test]
    fn test_compute_optimal_splits_boundary_values() {
        // Test boundary: 0-256 should return 1 (key optimization)
        assert_eq!(compute_splits_logic(0), 1);
        assert_eq!(compute_splits_logic(1), 1);
        assert_eq!(compute_splits_logic(128), 1);
        assert_eq!(compute_splits_logic(256), 1);

        // Test boundary: 257-512 should return 2
        assert_eq!(compute_splits_logic(257), 2);
        assert_eq!(compute_splits_logic(384), 2);
        assert_eq!(compute_splits_logic(512), 2);

        // Test boundary: 513-1024 should return 4
        assert_eq!(compute_splits_logic(513), 4);
        assert_eq!(compute_splits_logic(768), 4);
        assert_eq!(compute_splits_logic(1024), 4);

        // Test boundary: 1025-2048 should return 8
        assert_eq!(compute_splits_logic(1025), 8);
        assert_eq!(compute_splits_logic(1536), 8);
        assert_eq!(compute_splits_logic(2048), 8);

        // Test boundary: >2048 should return 16
        assert_eq!(compute_splits_logic(2049), 16);
        assert_eq!(compute_splits_logic(4096), 16);
        assert_eq!(compute_splits_logic(8192), 16);
    }

    #[test]
    fn test_compute_optimal_splits_key_optimization() {
        // The key optimization: sequences ≤256 tokens use num_splits=1
        // This eliminates 72 kernel launches (36 transpose + 36 untranspose)
        for seqlen in [1, 64, 128, 192, 256] {
            assert_eq!(
                compute_splits_logic(seqlen),
                1,
                "Sequences ≤256 should use num_splits=1 to eliminate transpose overhead"
            );
        }

        // Verify that 257 triggers split-K
        assert_eq!(
            compute_splits_logic(257),
            2,
            "Sequences >256 should use split-K"
        );
    }

    #[test]
    fn test_compute_optimal_splits_progressive_scaling() {
        // Test that splits increase progressively with sequence length
        let splits_256 = compute_splits_logic(256);
        let splits_512 = compute_splits_logic(512);
        let splits_1024 = compute_splits_logic(1024);
        let splits_2048 = compute_splits_logic(2048);
        let splits_4096 = compute_splits_logic(4096);

        assert!(splits_256 <= splits_512);
        assert!(splits_512 <= splits_1024);
        assert!(splits_1024 <= splits_2048);
        assert!(splits_2048 <= splits_4096);

        // Verify specific values
        assert_eq!(splits_256, 1);
        assert_eq!(splits_512, 2);
        assert_eq!(splits_1024, 4);
        assert_eq!(splits_2048, 8);
        assert_eq!(splits_4096, 16);
    }

    #[test]
    fn test_compute_optimal_splits_power_of_two() {
        // All split values should be powers of 2 (1, 2, 4, 8, 16)
        for seqlen in [1, 100, 300, 600, 1200, 2400, 5000] {
            let splits = compute_splits_logic(seqlen);
            assert!(
                splits == 1 || splits == 2 || splits == 4 || splits == 8 || splits == 16,
                "Split value {} is not a power of 2 for seqlen {}",
                splits,
                seqlen
            );
        }
    }

    fn compute_splits_logic(max_seqlen_k: usize) -> usize {
        match max_seqlen_k {
            0..=256 => 1,
            257..=512 => 2,
            513..=1024 => 4,
            1025..=2048 => 8,
            _ => 16,
        }
    }
}
/// CUDA backend factory. Registered with `inventory::submit!` so
/// `create_worker` can find it without naming `CudaWorker`.
#[cfg(feature = "cuda")]
pub struct CudaWorkerFactory;

#[cfg(feature = "cuda")]
impl scratchy_serving_engine::worker_factory::WorkerFactory for CudaWorkerFactory {
    fn matches(&self, device: &str) -> bool {
        device.starts_with("cuda") || device == "auto"
    }

    fn create(
        &self,
        cuda_config: WorkerCreateConfig,
        progress: Option<scratchy_serving_engine::worker_factory::ProgressCallback>,
    ) -> anyhow::Result<scratchy_serving_engine::worker_factory::WorkerCreationResult> {
        use anyhow::Context as _;

        info!(
            "Using scratchy-serving-cuda backend (device={})",
            cuda_config.device_id
        );

        let mut worker = CudaWorker::new(cuda_config);
        worker
            .init_device()
            .context("failed to initialize CUDA device")?;

        // Store progress reference in worker for layer-by-layer updates
        if let Some(cb) = progress {
            worker.set_progress_callback(cb);
        }

        worker.load_model().context("failed to load CUDA model")?;

        let hf_config = worker
            .hf_config()
            .context("model config not available after CUDA load")?
            .clone();
        let model_dir = worker.model_dir().map(|p| p.to_path_buf());
        let dtype_elem_bytes = worker.resolved_dtype_elem_bytes();

        Ok((Box::new(worker), hf_config, model_dir, dtype_elem_bytes))
    }

    fn device_total_bytes_and_name(&self) -> Option<(u64, String)> {
        crate::current_device_total_bytes_and_name()
    }

    fn device_memory_free_total(&self) -> Option<(u64, u64)> {
        // Ensure the driver + a primary context exist on this thread (HTTP
        // handlers may run on any tokio worker thread), then query.
        unsafe {
            crate::driver::init().ok()?;
            crate::driver::ctx_create(0).ok()?;
            crate::driver::mem_get_info()
                .ok()
                .map(|(free, total)| (free as u64, total as u64))
        }
    }
}

#[cfg(feature = "cuda")]
inventory::submit!(
    &CudaWorkerFactory as &dyn scratchy_serving_engine::worker_factory::WorkerFactory
);

// Safety: CudaWorker contains raw GPU pointers (via GpuDevice, model weights,
// KV cache, PP buffers, CUDA graphs) and raw pointer fields in OwnedTensor /
// RawGpuAlloc / RawGpuMem. All GPU resources are allocated on a single CUDA
// context and accessed exclusively from the worker thread. Send is required
// because the worker is created on the main thread and moved to its dedicated
// worker thread via the executor's spawn.
unsafe impl Send for CudaWorker {}
