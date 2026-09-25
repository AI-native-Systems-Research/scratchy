// SPDX-License-Identifier: Apache-2.0
//! `MetalWorker`: arena + per-bucket MTL4 execution plan.
//!
//! One worker holds:
//!  - a private arena of `metal::Buffer`s, sized by the model's
//!    post-coloring slot count (one buffer per slot id);
//!  - one [`BucketBaking`] per bucket, holding a `Vec<BucketStep>`
//!    (the execution plan) and the pre-built MTL4 `Mtl4Step` list.
//!
//! The execution plan partitions the bucket's command stream into
//! [`BucketStep`]s: contiguous runs of commands sharing a pipeline
//! become a single `BucketStep::Dispatch`; dense GEMM (f16 and bf16)
//! bakes to a `gemm_{f16,bf16}_specialized` Dispatch step like every
//! other kernel. MTL4 execution reads the pre-baked `mtl4_steps` from
//! each baking; a bucket is ineligible only if a kernel exceeds the
//! 31-entry argument-table bind cap.

#[cfg(test)]
use crate::interpreter::metal::lowered::IntoBaked;
use std::sync::Arc;

use crate::interpreter::metal::__re::{
    Buffer, ComputePipelineState, Device, MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize,
};
use ::objc2::rc::Retained;
use ::objc2::runtime::ProtocolObject;

use super::lowered::{
    Binding, KernelId, LoweredCommand, LoweredMetalTape, MetalDtype, WeightBundleKind, WeightTensor,
};
use super::pipelines::{PipelineLookupError, SpecializedPipelines};
use super::runtime::RuntimeBindings;
use crate::MetalAllocator;
#[cfg(feature = "forward-telemetry")]
use scratchy_core_common::forward_telemetry::{
    ForwardRecord, ForwardTelemetry, KernelKind, TapeEntry,
};
use scratchy_ir::CanonicalParams;

/// Byte size of arena slot `i`. The macro's `colored_slot_map()`
/// computes this from the FUF's per-slot shape × dtype × max bucket;
/// for the Phase 5.C smoke test the test sets it explicitly.
pub type ArenaLayout = Vec<u64>;

/// One unit of execution in a bucket's plan.
///
/// `Dispatch` is a contiguous run of commands sharing a single pipeline.
/// Dense GEMM (f16 and bf16) is a `Dispatch` step like every other
/// kernel — `gemm_{f16,bf16}_specialized`, no MPS.
pub enum BucketStep {
    Dispatch {
        /// Kernel id of every dispatch in this step. Coalescing
        /// requires same pipeline (same kernel), so one id is
        /// authoritative for the whole step.
        kernel: super::lowered::KernelId,
        /// Pipeline state for this step's kernel(s).
        pipeline: ComputePipelineState,
        /// Per-command explicit bindings: (buffer, offset, index).
        direct_bindings: Vec<Vec<(Buffer, u64, u64)>>,
        /// Per-command dispatch shape: (threadgroups, threads_per_tg).
        direct_dispatch: Vec<(MTLSize, MTLSize)>,
        /// Per-sub-dispatch m-axis scaling hint, parallel to
        /// `direct_dispatch`. When `Some`, the runtime rewrites
        /// `threadgroups.{axis}` proportionally with actual
        /// `num_tokens` (see
        /// [`crate::interpreter::metal::lowered::MScaling`]),
        /// shrinking the grid to the actual M instead of paying
        /// the `bucket_m`-shaped over-dispatch cost.
        direct_m_scaling: Vec<Option<super::lowered::MScaling>>,
        /// Per-sub-dispatch barrier-before flag, sourced from the
        /// compile-time DAG hazard analysis in `LoweredMetalTape`.
        /// Consumed by the MTL4 path via `Mtl4Step.barrier_before`.
        barrier_before: Vec<bool>,
        /// Per-sub-dispatch runtime gate, mirroring
        /// [`LoweredMetalTape::runtime_gate`]. `None` (the common
        /// case) means always dispatch; `Some(OnlyIfSingleSeq)` /
        /// `Some(OnlyIfMultiSeq)` skips the dispatch unless the
        /// live `num_seqs` matches. Used by the lm_head slice +
        /// fallback pair so the cheap M=1 slice fires for
        /// single-seq forwards and the full M=bucket_m fallback
        /// fires for multi-seq batched/mixed forwards.
        runtime_gate: Vec<Option<super::lowered::RuntimeGate>>,
    },
}

/// A `(buffer, offset)` pair for a resolved dense-GEMM operand. The
/// arena/weight buffers themselves outlive the worker (the arena
/// lives on the worker; weight buffers live on the model meta which
/// the pool keeps alive), so a non-owning `Buffer` clone is
/// equivalent to an `Arc` clone — `metal::Buffer` is itself a
/// reference-counted handle.
pub struct BoundBuffer {
    pub buffer: Buffer,
    pub offset: u64,
}

/// One bucket's baked artifacts: the execution plan and MTL4 steps.
pub struct BucketBaking {
    pub bucket_m: u32,
    /// MTL4 steps built from the bake-time `BucketStep` plan. `None` iff
    /// a kernel exceeds the 31-binding argument-table cap (such buckets
    /// have no MTL4 execution path).
    pub mtl4_steps: Option<Vec<super::mtl4::Mtl4Step>>,
    /// Per-bucket inline-constants buffer. Backs every
    /// `Binding::Inline { value }` in this bucket's commands by packing
    /// all u32 values into one shared-storage MTLBuffer at consecutive
    /// 4-byte offsets, in command-then-binding order. `None` if the
    /// bucket has no Inline bindings. Lives on the baking (not the
    /// worker) because the values are baked at record-time from the
    /// command-specific literals in the lowered tape — different
    /// buckets fed by the same model have different axis_sizes /
    /// top_k stamps. dispatch-bound commands access this via
    /// `setBuffer_offset_atIndex` like any other device buffer.
    pub moe_inline_buf: Option<Buffer>,
}

#[derive(Debug)]
pub enum WorkerError {
    /// Lookup against [`SpecializedPipelines`] failed.
    PipelineLookup(PipelineLookupError),
    /// `arena_layout.len()` did not match the lowered tape's
    /// `num_arena_slots`. Indicates a mismatched lowering and arena
    /// computation upstream — the macro should keep these in sync.
    ArenaShapeMismatch { expected: u32, actual: usize },
    /// A `Binding::ArenaSlot { slot, .. }` referenced a slot id
    /// outside `[0, arena_layout.len())`.
    ArenaSlotOutOfRange {
        bucket_index: usize,
        command_index: usize,
        slot: u32,
        arena_len: usize,
    },
    /// `KernelId::Gemm` reached the bake step but its
    /// `LoweredCommand::gemm_dims` was `None`. Indicates a lowering
    /// bug — the lowering pass owns populating those for `Gemm`
    /// commands.
    MissingGemmDims {
        bucket_index: usize,
        command_index: usize,
    },
    /// `KernelId::Gemm` had unexpected bindings. The worker expects
    /// (output, input, weight) at indices 0/1/2 — anything else
    /// is a lowering / model-meta contract violation.
    GemmBindingsMalformed {
        bucket_index: usize,
        command_index: usize,
        reason: &'static str,
    },
    /// Resolving a `Binding::Weight` via the WtFn thunk + allocator
    /// failed. Either the layer struct didn't carry the requested
    /// tensor (e.g. bias absent on a no-bias linear) or the tensor's
    /// raw pointer didn't fall inside any of the allocator's arenas.
    WeightLookupFailed { reason: &'static str },
    /// A command referenced `Binding::Scratch` but the worker has no
    /// SplitK scratch buffer allocated. Indicates a lowering /
    /// `LoweredMetalTape::splitk_scratch_bytes` accounting bug —
    /// either lowering emitted Scratch without registering the
    /// scratch byte count, or the worker discarded the buffer.
    ScratchBufferMissing {
        bucket_index: usize,
        command_index: usize,
    },
}

impl std::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PipelineLookup(e) => write!(f, "MetalWorker: pipeline lookup: {e}"),
            Self::ArenaShapeMismatch { expected, actual } => write!(
                f,
                "MetalWorker: arena shape mismatch: tape expects {expected} slots, layout has {actual}"
            ),
            Self::ArenaSlotOutOfRange {
                bucket_index,
                command_index,
                slot,
                arena_len,
            } => write!(
                f,
                "MetalWorker: bucket {bucket_index} command {command_index} \
                 references arena slot {slot} but arena has only {arena_len} slots"
            ),
            Self::MissingGemmDims {
                bucket_index,
                command_index,
            } => write!(
                f,
                "MetalWorker: bucket {bucket_index} command {command_index}: \
                 KernelId::Gemm has no gemm_dims (lowering bug)"
            ),
            Self::GemmBindingsMalformed {
                bucket_index,
                command_index,
                reason,
            } => write!(
                f,
                "MetalWorker: bucket {bucket_index} command {command_index}: \
                 GEMM bindings malformed ({reason})"
            ),
            Self::WeightLookupFailed { reason } => {
                write!(f, "MetalWorker: weight lookup: {reason}")
            }
            Self::ScratchBufferMissing {
                bucket_index,
                command_index,
            } => write!(
                f,
                "MetalWorker: bucket {bucket_index} command {command_index}: \
                 Binding::Scratch with no splitk scratch buffer allocated \
                 (lowering / tape accounting bug)"
            ),
        }
    }
}

impl std::error::Error for WorkerError {}

/// One per concurrent forward. Owns its arena + per-bucket bakings;
/// borrows the model meta + runtime bindings + pipeline cache via
/// references threaded through `new`.
pub struct MetalWorker<W: CanonicalParams> {
    pub arena: Vec<Buffer>,
    pub bucket_bakings: Vec<BucketBaking>,
    /// Shared SplitK scratch buffer. `Some` when any bucket tape
    /// requested a non-zero `splitk_scratch_bytes` (i.e. at least one
    /// `Instruction::AffineQmm` in the tape picked
    /// `QmmTKernel::SplitK`); `None` otherwise. Sized to the max
    /// `splitk_scratch_bytes` across all bucket tapes, since
    /// successive `affine_qmm_t_splitk` calls inside a single dispatch
    /// run sequentially and can reuse the same buffer.
    pub splitk_scratch: Option<Buffer>,
    /// Shared MoE scratch buffer for `Binding::MoeScratch` resolution.
    /// Sized to `max(bucket_tapes.moe_scratch_bytes)`. `None` when no
    /// MoE instruction lowered (every bucket reports
    /// `moe_scratch_bytes = 0`). Layout decisions
    /// (router_logits / sorted_full / topk_inds / topk_scores /
    /// gate_out / up_out / down_out byte offsets) are owned by the
    /// lowering pass — see `MoeScratchLayout` in `lowering.rs`.
    pub moe_scratch: Option<Buffer>,
    /// Shared roped-K scratch buffer for `Binding::RopedKScratch`
    /// resolution (spans rope-on-read on NAX). Sized to
    /// `max(bucket_tapes.roped_k_scratch_bytes)`. `None` when no NAX
    /// spans prefill lowered. `RopeOnceNax` writes one layer's pre-roped
    /// K here; the following NAX attention reads it (no per-tile rope).
    pub roped_k_scratch: Option<Buffer>,
    /// Shared hd512-unfused-attention scratch (`Binding::AttnUnfusedScratch`),
    /// sized to `max(attn_unfused_scratch_bytes)`. `None` when no hd512-unfused
    /// attention was lowered.
    pub attn_unfused_scratch: Option<Buffer>,
    /// TurboQuant KV cache active for this worker (set from `runtime.tq` at
    /// build). Gates the TurboQuant per-layer commands (`gate_matches`).
    pub turboquant: bool,
    /// Per-forward block-table ROW WIDTH (== the host's `max_blocks_eff`
    /// stride) for the TurboQuant full-context staging grid. Stashed from
    /// `inputs.block_tables` before each forward (see the `write_runtime_inputs`
    /// call sites in `pool.rs`). The `TqStageRotated` dispatch reads this
    /// for `grid.x` so the staging covers the WHOLE active context. Using the
    /// static `W::MAX_BLOCKS_PER_SEQ` instead (128 for uniform arches) truncated
    /// it at 128 blocks / 2048 tokens, leaving the reused fp16 scratch
    /// tail holding the previous layer's KV → `!!!!` collapse past 2048 tokens.
    /// `0` until the first forward sets it (dispatch falls back to the const).
    pub tq_dequant_max_blocks: std::sync::atomic::AtomicU32,
    _marker: std::marker::PhantomData<fn() -> W>,
}

impl<W: CanonicalParams> MetalWorker<W> {
    /// Build a worker.
    ///
    /// `arena_layout[i]` = byte size of arena slot `i`. The arena is
    /// sized for the worker's largest bucket; downstream forwards
    /// re-use the same arena across buckets.
    ///
    /// `bucket_tapes` is the per-bucket lowered tape set, in the
    /// caller's bucket order. The worker bakes one dispatch + execution
    /// plan per tape.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: Arc<Device>,
        arena_layout: &ArenaLayout,
        bucket_tapes: &[LoweredMetalTape],
        pipelines: &SpecializedPipelines,
        weights: &W,
        allocator: &MetalAllocator,
        runtime: &RuntimeBindings,
    ) -> Result<Self, WorkerError> {
        Self::new_with_residency(
            device,
            arena_layout,
            bucket_tapes,
            pipelines,
            weights,
            allocator,
            runtime,
            None,
        )
    }

    /// Same as [`Self::new`] but accepts an optional `MetalResidencySet`
    /// that worker-local arena slots get inserted into. Used by the
    /// pool to pin every per-worker arena into the wired set so cmdbuf
    /// dispatches don't race against Apple's lazy paging on
    /// Llama-3.2-class working sets.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_residency(
        device: Arc<Device>,
        arena_layout: &ArenaLayout,
        bucket_tapes: &[LoweredMetalTape],
        pipelines: &SpecializedPipelines,
        weights: &W,
        allocator: &MetalAllocator,
        runtime: &RuntimeBindings,
        residency: Option<&crate::residency::MetalResidencySet>,
    ) -> Result<Self, WorkerError> {
        // Arena slot count comes from the lowered tape (post-FUF
        // coloring). The shared arena is sized to the MAX colored slot
        // count across buckets (see `for_buckets`); per-bucket counts can
        // differ (a fusion needing extra scratch carries more slots), so
        // a tape may legitimately use FEWER slots than the arena holds.
        // Only a tape needing MORE slots than the arena provides is a
        // genuine lowering/arena mismatch.
        if let Some(max_tape) = bucket_tapes.iter().map(|t| t.num_arena_slots).max()
            && (max_tape as usize) > arena_layout.len()
        {
            return Err(WorkerError::ArenaShapeMismatch {
                expected: max_tape,
                actual: arena_layout.len(),
            });
        }

        let arena: Vec<Buffer> = arena_layout
            .iter()
            .map(|&size| {
                // Shared storage so test code can seed/inspect arena
                // contents without staging copies. dispatch-bound buffers
                // are fine in shared on Apple silicon — the existing
                // `MetalAllocator` uses the same mode.
                let buf = device
                    .newBufferWithLength_options(
                        size as usize,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .expect("newBufferWithLength_options returned nil");
                if let Some(r) = residency {
                    r.insert(&buf);
                }
                buf
            })
            .collect();

        // Shared SplitK scratch buffer sized to the max across all
        // bucket tapes — one buffer suffices because successive
        // `affine_qmm_t_splitk` / `splitk_reduce_sum` pairs run
        // serially inside a single encoder. Allocated only when at
        // least one tape requested non-zero scratch; bakings that
        // don't reference `Binding::Scratch` pay nothing.
        let max_splitk_scratch_bytes: u32 = bucket_tapes
            .iter()
            .map(|t| t.splitk_scratch_bytes)
            .max()
            .unwrap_or(0);
        let splitk_scratch: Option<Buffer> = if max_splitk_scratch_bytes > 0 {
            let buf = device
                .newBufferWithLength_options(
                    max_splitk_scratch_bytes as usize,
                    MTLResourceOptions::StorageModePrivate,
                )
                .expect("newBufferWithLength_options returned nil (splitk scratch)");
            if let Some(r) = residency {
                r.insert(&buf);
            }
            Some(buf)
        } else {
            None
        };

        // Shared MoE scratch buffer for `Binding::MoeScratch`. Sized
        // to the max per-bucket `moe_scratch_bytes` because MoE blocks
        // within one bucket execute serially through the dispatch; cross-
        // bucket reuse is fine because only one bucket runs per
        // forward. Private storage — host never reads these
        // intermediates back.
        let max_moe_scratch_bytes: u32 = bucket_tapes
            .iter()
            .map(|t| t.moe_scratch_bytes)
            .max()
            .unwrap_or(0);
        let moe_scratch: Option<Buffer> = if max_moe_scratch_bytes > 0 {
            let buf = device
                .newBufferWithLength_options(
                    max_moe_scratch_bytes as usize,
                    MTLResourceOptions::StorageModePrivate,
                )
                .expect("newBufferWithLength_options returned nil (moe scratch)");
            if let Some(r) = residency {
                r.insert(&buf);
            }
            Some(buf)
        } else {
            None
        };

        // Shared roped-K scratch buffer for `Binding::RopedKScratch`
        // (spans rope-on-read, rope-once-to-scratch — NAX matrix-accel AND the
        // simdgroup steel prefill). Sized to the max `roped_k_scratch_bytes`
        // across buckets — one buffer suffices because the RopeOnce{Nax,Steel}
        // command and its following attention run serially per layer through
        // the dispatch, and the scratch is overwritten each layer (the cache, not
        // the scratch, is the persistent artifact).
        // Private storage — host never reads it back. Bound by dispatch-baked
        // address, so it MUST be made resident (lazy-pager hazard).
        let max_roped_k_scratch_bytes: u32 = bucket_tapes
            .iter()
            .map(|t| t.roped_k_scratch_bytes)
            .max()
            .unwrap_or(0);
        let roped_k_scratch: Option<Buffer> = if max_roped_k_scratch_bytes > 0 {
            let buf = device
                .newBufferWithLength_options(
                    max_roped_k_scratch_bytes as usize,
                    MTLResourceOptions::StorageModePrivate,
                )
                .expect("newBufferWithLength_options returned nil (roped-K scratch)");
            if let Some(r) = residency {
                r.insert(&buf);
            }
            Some(buf)
        } else {
            None
        };

        // Shared hd512-unfused-attention scratch (q_head/Kdense/Vdense_T/scores/
        // out_head packed at baked offsets). One buffer, overwritten per layer.
        let max_attn_unfused_scratch_bytes: u32 = bucket_tapes
            .iter()
            .map(|t| t.attn_unfused_scratch_bytes)
            .max()
            .unwrap_or(0);
        let attn_unfused_scratch: Option<Buffer> = if max_attn_unfused_scratch_bytes > 0 {
            let buf = device
                .newBufferWithLength_options(
                    max_attn_unfused_scratch_bytes as usize,
                    MTLResourceOptions::StorageModePrivate,
                )
                .expect("newBufferWithLength_options returned nil (attn-unfused scratch)");
            if let Some(r) = residency {
                r.insert(&buf);
            }
            Some(buf)
        } else {
            None
        };

        // Runtime metadata buffers (`input_ids`, `positions`,
        // `slot_mapping`, `cu_seqlens_q`, `seq_used_k`, `block_table`)
        // are allocated by the per-canonical `RuntimeFactory` closure
        // and never inserted into the residency set there. dispatch-recorded
        // commands bind them via `set_kernel_buffer` at bake time, so
        // the encoder firing the MTL4 compute encoder never sees a
        // `setBuffer` for them — without an explicit residency entry,
        // Apple's lazy paging can hand back stale pages and the dispatch
        // path produces garbage output (the dormant comment at
        // `pool.rs` flagging "wrong outputs for decode buckets" was
        // exactly this). The KV cache buffers in `kv_cache_k/v` are
        // already inserted by the executor that constructs the pool;
        // inserting them again would be a no-op but we skip to keep
        // the loop tight.
        if let Some(r) = residency {
            // Compile-time residency guard: destructure RuntimeBindings with
            // NO `..` rest so adding a new runtime buffer fails THIS build
            // until its residency is explicitly decided — either pinned here
            // (`r.insert(field)`) or bound to `field: _` with a reason. A new
            // dispatch-bound buffer that is silently left un-pinned reads stale
            // zero pages → token salad.
            let RuntimeBindings {
                input_ids,
                positions,
                slot_mappings,
                cu_seqlens_q,
                seq_used_k,
                span_ids,
                block_tables,
                // Plain `Vec<u32>` metadata, not a GPU buffer — no residency.
                layer_to_group: _,
                // Pinned at pool construction by the executor (see below).
                kv_cache_k: _,
                kv_cache_v: _,
                // Spans rope-on-read flags — bound by dispatch-baked address
                // when W::ROPE_ON_READ, so pinned below like block_tables
                // (16-byte placeholder + never bound on non-spans arches).
                block_unrotated_flags,
                // Written + read within the same encoder via setBuffer (not a
                // baked dispatch address), so the lazy-pager hazard doesn't apply.
                num_tokens_u32: _,
                num_sample_rows_u32: _,
                sample_indices: _,
                // GDN persistent state pools — pinned by the GdnStatePool owner.
                gdn_state_conv: _,
                gdn_state_ssm: _,
                gdn_state_indices,
                gdn_is_fresh,
                vision_rope_freqs,
                pixels,
                vision_pos_embeds,
                mm_embeds,
                mm_dst_rows,
                mrope_cos_sin,
                vision_cu_seqlens_full,
                vision_cu_seqlens_window,
                vision_window_index,
                vision_reverse_indices,
                vision_position_ids,
                // TurboQuant packed stores + codebook — pinned below when Some
                // (dispatch-bound by baked gpuAddress, same lazy-pager hazard).
                tq,
            } = runtime;
            r.insert(input_ids);
            r.insert(positions);
            r.insert(cu_seqlens_q);
            r.insert(seq_used_k);
            // Span labels: bound by dispatch-baked address on rope-on-read
            // arches (slot 8 of the prefill attention), so pin like seq_used_k.
            r.insert(span_ids);
            // Per-KV-cache-group slot_mappings + block tables (vLLM hybrid
            // layout). One group on uniform models; full + N sliding on gemma4
            // SWA. Each is bound by dispatch-baked gpuAddress (same lazy-pager
            // hazard as the inputs), so EVERY group buffer must be pinned.
            for b in slot_mappings {
                r.insert(b);
            }
            for b in block_tables {
                r.insert(b);
            }
            // Spans rope-on-read per-layer flag mirrors (placeholders on
            // non-spans arches; bound by baked gpuAddress when active).
            for b in block_unrotated_flags {
                r.insert(b);
            }
            // Gated-DeltaNet per-forward index buffers (hybrid arches:
            // Qwen3.5 / Qwen3-Next). Same contract as the inputs above —
            // the GDN conv1d/scan kernels bind them by baked gpuAddress in
            // the dispatch, never via an encoder `setBuffer`, so without an
            // explicit residency entry Apple's lazy pager hands the GPU
            // stale (zeroed) pages: the kernels then read `state_indices`
            // as all-zero, routing EVERY sequence's recurrent state to
            // slot 0 (so only the first sequence in a batch keeps its GDN
            // state; the rest read/write slot 0 and inherit seq-0's
            // answer). The factory allocates these unconditionally (a
            // 16 KiB Shared buffer even for non-hybrid arches), so the
            // insert is a harmless pin when no GDN command binds them.
            r.insert(gdn_state_indices);
            r.insert(gdn_is_fresh);
            // Vision per-forward externs (vision towers: Qwen3.5-VL ViT).
            // Same baked-gpuAddress / lazy-pager hazard as the GDN
            // buffers above — the `vision_rope_2d` / `vision_varlen_attn`
            // / `LoadPixels` kernels read `freqs` / `cu_seqlens` / `pixels`
            // by dispatch-baked address, so an un-pinned buffer is served stale
            // zero pages (garbage rope angles / all-token-0 pixels). The
            // factory allocates 16-byte placeholders on non-vision arches,
            // so the pin is harmless when no vision command binds them.
            r.insert(vision_rope_freqs);
            r.insert(pixels);
            r.insert(vision_pos_embeds);
            r.insert(mm_embeds);
            r.insert(mm_dst_rows);
            r.insert(mrope_cos_sin);
            r.insert(vision_cu_seqlens_full);
            r.insert(vision_cu_seqlens_window);
            r.insert(vision_window_index);
            r.insert(vision_reverse_indices);
            r.insert(vision_position_ids);
            // TurboQuant: the packed code stores + norms + codebook are read by
            // the dequant/quantize dispatch commands via baked gpuAddress, so pin them
            // (same lazy-pager hazard). `None` on non-turboquant runs → no-op.
            if let Some(t) = tq {
                for b in &t.packed_k {
                    r.insert(b);
                }
                for b in &t.packed_v {
                    r.insert(b);
                }
                for b in &t.norms_k {
                    r.insert(b);
                }
                for b in &t.norms_v {
                    r.insert(b);
                }
                r.insert(&t.signs);
                r.insert(&t.boundaries);
                r.insert(&t.centroids);
                // The fp16 scratch (table + backing data) is bound at every
                // layer's kv_cache slot + dereffed via the table's gpuAddress.
                r.insert(&t.scratch_k_table);
                r.insert(&t.scratch_v_table);
                r.insert(&t.scratch_k_data);
                r.insert(&t.scratch_v_data);
            }
            r.commit();
        }

        let mut bucket_bakings = Vec::with_capacity(bucket_tapes.len());
        for (bucket_idx, tape) in bucket_tapes.iter().enumerate() {
            let baking = bake_bucket(
                bucket_idx,
                tape,
                &arena,
                splitk_scratch.as_ref(),
                moe_scratch.as_ref(),
                roped_k_scratch.as_ref(),
                attn_unfused_scratch.as_ref(),
                pipelines,
                weights,
                allocator,
                runtime,
                device.clone(),
            )?;
            // Insert the per-bucket inline-constants buffer into the
            // residency set so the dispatch exec dispatch sees its
            // residency entry (the dispatch never `setBuffer`s these
            // explicitly — bindings are pre-recorded in the dispatch).
            if let (Some(r), Some(b)) = (residency, baking.moe_inline_buf.as_ref()) {
                r.insert(b);
            }
            bucket_bakings.push(baking);
        }
        if let (Some(r), Some(b)) = (residency, moe_scratch.as_ref()) {
            r.insert(b);
            r.commit();
        }

        Ok(Self {
            arena,
            bucket_bakings,
            splitk_scratch,
            moe_scratch,
            roped_k_scratch,
            attn_unfused_scratch,
            turboquant: runtime.tq.is_some(),
            tq_dequant_max_blocks: std::sync::atomic::AtomicU32::new(0),
            _marker: std::marker::PhantomData,
        })
    }

    /// Count total dispatches across all MTL4 steps in this bucket.
    pub fn count_dispatches(&self, bucket: usize) -> usize {
        self.bucket_bakings
            .get(bucket)
            .and_then(|b| b.mtl4_steps.as_ref())
            .map(|steps| steps.iter().map(|s| s.dispatches.len()).sum())
            .unwrap_or(0)
    }

    /// True if this bucket's tape contains an `AvgPool2d` dispatch — the
    /// gemma3-mm SigLIP projector's spatial pool. That op (and the
    /// `soft_emb_norm -> mm_input_projection` tail after it) hits an
    /// in-command-buffer write→read coherence failure at the 4096-patch
    /// scale: the projector gemm reads the soft-emb-norm output as
    /// all-zero (silent all-zero vision embeds → garbled text) UNLESS a
    /// command-buffer boundary (commit + host-wait) separates the
    /// writer from the reader. In-CB `Dispatch→Dispatch` barriers — even
    /// `visibility=Device`, even forced on every dispatch — do NOT fix
    /// it; only the CB boundary does. The pool serializes such buckets
    /// via [`run_dump_segment`]-style chunked commits. Cheap to scan
    /// (handful of steps) and the result gates a once-per-image path.
    pub fn bucket_has_avg_pool_2d(&self, bucket: usize) -> bool {
        self.bucket_bakings
            .get(bucket)
            .and_then(|b| b.mtl4_steps.as_ref())
            .map(|steps| {
                steps
                    .iter()
                    .any(|s| matches!(s.kernel, super::lowered::KernelId::AvgPool2d))
            })
            .unwrap_or(false)
    }

    /// Phase A.3 MTL4 execution path. Encodes the
    /// bucket's `mtl4_steps` onto a caller-provided MTL4 compute
    /// encoder using pre-baked `MTL4ArgumentTable`s. The caller owns
    /// command-buffer lifecycle (`begin`/`endCommandBuffer`),
    /// residency wiring (`useResidencySet`), commit, and event-based
    /// wait. Returns an error if MTL4 was not baked for this bucket
    /// (e.g. a kernel exceeds the 31-binding argument-table cap).
    pub fn run_bucket_mtl4(
        &self,
        bucket: usize,
        num_tokens: u32,
        num_seqs: u32,
        has_spec_tokens: bool,
        enc: &ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
    ) -> Result<(), WorkerError> {
        self.run_bucket_mtl4_inner(bucket, num_tokens, num_seqs, has_spec_tokens, enc, None)
    }

    /// Activation-dump replay segment: encode only the flat dispatch
    /// indices in `range` (counting every dispatch slot in step order,
    /// including runtime-gate-skipped ones, so indices stay aligned
    /// with the lowered command order / the [`DumpCmd`] sidecar).
    /// Used by the pool's `run_dump_pass` to re-run the tape in
    /// segments with a host wait + arena readback between them.
    pub fn run_bucket_mtl4_range(
        &self,
        bucket: usize,
        num_tokens: u32,
        num_seqs: u32,
        has_spec_tokens: bool,
        enc: &ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
        range: std::ops::Range<usize>,
    ) -> Result<(), WorkerError> {
        self.run_bucket_mtl4_inner(
            bucket,
            num_tokens,
            num_seqs,
            has_spec_tokens,
            enc,
            Some(range),
        )
    }

    fn run_bucket_mtl4_inner(
        &self,
        bucket: usize,
        num_tokens: u32,
        num_seqs: u32,
        has_spec_tokens: bool,
        enc: &ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
        range: Option<std::ops::Range<usize>>,
    ) -> Result<(), WorkerError> {
        use ::objc2_metal::{
            MTL4CommandEncoder, MTL4ComputeCommandEncoder as _, MTL4VisibilityOptions, MTLStages,
        };
        let baking = &self.bucket_bakings[bucket];
        let mtl4_steps =
            baking
                .mtl4_steps
                .as_ref()
                .ok_or(WorkerError::WeightLookupFailed {
                    reason: "MTL4 path requested but bucket has no mtl4_steps (Gemm or too-many-bindings fallback)",
                })?;
        // MTL4 compute encoders do NOT auto-serialize successive
        // dispatches the way MTL3's default-Serial encoder does;
        // the per-sub-dispatch `barrier_before` flag was computed
        // at macro time by `scratchy-forward-compiler-macro::interpreter_codegen
        // ::lower_bucket` from the FUF dataflow + `Implementation::
        // kv_layer_io`. Runtime does zero analysis — just emits a
        // `Dispatch→Dispatch` barrier wherever the flag fires.
        // Flat dispatch index across all steps. Counts EVERY dispatch
        // slot (including range-filtered and gate-skipped ones) so it
        // stays aligned with the lowered command order — the contract
        // the `DumpCmd` sidecar / `run_bucket_mtl4_range` rely on.
        let mut flat_idx: usize = 0;
        // Opt-in kernel tape (forward-local; published once at encode end).
        // Only the full forward is captured — dump-replay ranges are skipped so
        // the tape always reflects a complete dispatch sequence.
        #[cfg(feature = "forward-telemetry")]
        let tape_enabled = range.is_none() && ForwardTelemetry::global().is_enabled();
        #[cfg(feature = "forward-telemetry")]
        let mut tape: Vec<TapeEntry> = if tape_enabled {
            Vec::with_capacity(mtl4_steps.iter().map(|s| s.dispatches.len()).sum())
        } else {
            Vec::new()
        };
        for step in mtl4_steps {
            enc.setComputePipelineState(&step.pipeline);
            for ((((table, (tg, tpt)), need_barrier), scaling), gate) in step
                .tables
                .iter()
                .zip(step.dispatches.iter())
                .zip(step.barrier_before.iter())
                .zip(step.m_scaling.iter())
                .zip(step.runtime_gate.iter())
            {
                let this_idx = flat_idx;
                flat_idx += 1;
                // The compiler's own hazard-analysis verdict for the tape,
                // captured before runtime overrides (force/range) mutate it.
                #[cfg(feature = "forward-telemetry")]
                let compiler_barrier = *need_barrier;
                // In range mode: skip dispatches outside the segment,
                // and suppress the barrier on the segment's FIRST
                // dispatch — its predecessor ran in a previous command
                // buffer (commit + host wait = stronger ordering), and
                // a leading barrier on an empty encoder is something
                // the production path never emits.
                let mut need_barrier = *need_barrier;
                if let Some(r) = &range {
                    if !r.contains(&this_idx) {
                        continue;
                    }
                    if this_idx == r.start {
                        need_barrier = false;
                    }
                }
                if !gate_matches(
                    *gate,
                    num_tokens,
                    num_seqs,
                    has_spec_tokens,
                    self.turboquant,
                ) {
                    // Skipped: the lm_head slice's gather/qmv/scatter
                    // (gated single-seq) doesn't fire for batched
                    // batches; the M=bucket_m fallback (gated
                    // multi-seq) doesn't fire for single-seq prefill.
                    // Either way the timestamp slot, barrier, and
                    // dispatch are skipped together so the kernel
                    // doesn't run with stale per-dispatch state.
                    continue;
                }
                if need_barrier {
                    // Default to `None` visibility — measured -30 ms
                    // TTFT @ 1024-tok / -89 ms @ 2048-tok on M4
                    // Llama-3.2-3B-4bit, coherent on the standard probes
                    // (short prompts, 80-tok Apollo recall, haiku
                    // composition, Llama-3.2-1B math). Within a single
                    // MTL4 compute encoder, dispatch-to-dispatch
                    // sync is sufficient for correctness — full
                    // device-coherent visibility is over-conservative
                    // for back-to-back dispatches that aren't writing
                    // to memory other dispatches in the SAME encoder
                    // need cache-coherent reads of. The final cmdbuf
                    // commit point flushes everything before the next
                    // encoder runs.
                    if std::env::var_os("SCRATCHY_METAL_NO_BARRIERS").is_none() {
                        enc.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                            MTLStages::Dispatch,
                            MTLStages::Dispatch,
                            MTL4VisibilityOptions::None,
                        );
                    }
                }
                let tg_scaled = scale_tg_for_num_tokens(
                    *tg,
                    *scaling,
                    super::ids::NumTokens(num_tokens),
                    num_seqs,
                );
                // TurboQuant full-context staging: the baked grid is a
                // placeholder; the kernel needs (max_blocks, num_kv_heads,
                // num_seqs) — block-table-driven, with early-exit past
                // seqused_k; grid.z = the live batch's num_seqs.
                let tg_scaled = if matches!(step.kernel, super::lowered::KernelId::TqStageRotated) {
                    // grid.x must cover the host block-table ROW WIDTH
                    // (`max_blocks_eff`) — every block of the longest
                    // sequence. The static `W::MAX_BLOCKS_PER_SEQ` (128 for
                    // uniform arches like Llama) truncated the pass at 128
                    // blocks / 2048 tokens, so the reused fp16 scratch's tail
                    // kept the PREVIOUS layer's KV → attention collapse to
                    // `!!!!` past 2048 tokens. The per-forward runtime width is
                    // stashed from `inputs.block_tables` (== the stride the host
                    // padded the block_table to); fall back to the baked const
                    // when unset (0). Floor at the const so we never shrink
                    // below the host stride for short contexts.
                    let runtime_mb = self
                        .tq_dequant_max_blocks
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let baked =
                        <W as ::scratchy_forward_compiler::CanonicalParams>::MAX_BLOCKS_PER_SEQ;
                    // Cover the whole active context: the runtime block-table
                    // width (== the host's padded stride), floored at the baked
                    // const so short contexts never shrink below it.
                    let cover = runtime_mb.max(baked) as usize;
                    MTLSize {
                        width: cover,
                        height: tg_scaled.height,
                        depth: (num_seqs as usize).max(1),
                    }
                } else {
                    tg_scaled
                };
                // RopeOnce{Steel,Nax,GqaShared}: the pre-roped-K scratch is sized to
                // MAX_BLOCKS_PER_SEQ logical blocks (lowering), and the steel/gqa
                // attention reads the WHOLE sequence's roped K (computed prefix +
                // new). The baked grid M-scales by num_tokens (the NEW tokens only)
                // — too few for a chunked-prefill CONTINUATION, leaving the prefix
                // blocks past one bucket un-roped → garbage K → `!!!!` past 4096
                // tokens. Override grid.y to the live block-table width
                // (`tq_dequant_max_blocks`, == kv_len/block_size), capped at the
                // baked num_pages (the scratch's block capacity). Mirrors the
                // TqStageRotated grid override above.
                let tg_scaled = if matches!(
                    step.kernel,
                    super::lowered::KernelId::RopeOnceSteel
                        | super::lowered::KernelId::RopeOnceNax
                        | super::lowered::KernelId::RopeOnceGqaShared
                ) {
                    let runtime_mb = self
                        .tq_dequant_max_blocks
                        .load(std::sync::atomic::Ordering::Relaxed)
                        as usize;
                    let cap = tg.height; // baked num_pages = MAX_BLOCKS_PER_SEQ
                    let cover = if runtime_mb == 0 {
                        cap
                    } else {
                        runtime_mb.min(cap)
                    };
                    MTLSize {
                        width: tg_scaled.width,
                        height: cover,
                        depth: tg_scaled.depth,
                    }
                } else {
                    tg_scaled
                };
                enc.setArgumentTable(Some(table));
                enc.dispatchThreadgroups_threadsPerThreadgroup(tg_scaled, *tpt);
                #[cfg(feature = "forward-telemetry")]
                if tape_enabled {
                    tape.push(TapeEntry {
                        kind: kernel_kind(step.kernel),
                        barrier: compiler_barrier,
                        fused: is_fused(step.kernel),
                    });
                }
            }
        }
        #[cfg(feature = "forward-telemetry")]
        if tape_enabled {
            ForwardTelemetry::global().publish(ForwardRecord {
                num_tokens,
                num_seqs,
                tape,
            });
        }
        Ok(())
    }
}

/// Whether the compiler folded several ops into this single dispatch — the
/// `Fused*` kernels plus the multi-op rope/norm variants.
#[cfg(feature = "forward-telemetry")]
fn is_fused(id: KernelId) -> bool {
    use KernelId as K;
    matches!(
        id,
        K::FusedAddRmsNorm
            | K::FusedGateUpSiluMul
            | K::FusedQkvRopeCache
            | K::FusedAffineQkvRopeCache
            | K::RopeAppendNormed
            | K::NormAddScalarMul
            | K::SynthGateUpSiluMul
            | K::AttentionViaCacheTq
    )
}

/// Map a lowered kernel identifier onto the arch-agnostic family the UI colors
/// the live dispatch tape by. Exhaustive on purpose: a new `KernelId` variant
/// forces a classification decision here rather than silently defaulting.
#[cfg(feature = "forward-telemetry")]
fn kernel_kind(id: KernelId) -> KernelKind {
    use KernelId as K;
    match id {
        K::Embed | K::AffineEmbed | K::EmbeddingGather | K::MmEmbedSplice => KernelKind::Embed,
        K::RmsNorm | K::RmsNormUnit | K::FusedAddRmsNorm | K::NormAddScalarMul => KernelKind::Norm,
        K::RopeAppendNormed
        | K::RopeAppend
        | K::FusedQkvRopeCache
        | K::FusedAffineQkvRopeCache
        | K::RopeOnceNax
        | K::RopeOnceSteel
        | K::RopeOnceGqaShared => KernelKind::Rope,
        K::AttentionViaCache
        | K::AttentionViaCacheTq
        | K::AttentionPrefillSdpaPaged
        | K::AttnGatherKRope
        | K::AttnGatherVCopyT
        | K::AttnQConvert
        | K::AttnOConvert
        | K::AttnCausalSoftmax
        | K::AttnGemmQk
        | K::AttnGemmPv
        | K::SynthPreAttn => KernelKind::Attention,
        K::GatedDeltaNet => KernelKind::GatedDeltaNet,
        K::Gemm
        | K::AffineQmvQuad
        | K::AffineQmvFast
        | K::AffineQmv
        | K::AffineQmmT
        | K::AffineGatherQmmT
        | K::AffineGatherQmmTNax
        | K::AffineQmmTSplitK
        | K::AffineQmmTNax
        | K::Nvfp4Qmv
        | K::Nvfp4QmmT
        | K::Nvfp4QmmTNax
        | K::AffineGatherQmvFast
        | K::AffineGatherQmv
        | K::SplitKReduceSum => KernelKind::Gemm,
        K::MoeWeightedSum
        | K::MoeGroupOffsets
        | K::MoeGroupInit
        | K::MoeGroupScatter
        | K::MoeGroupGather
        | K::MoePerExpertScale
        | K::GateApply
        | K::GateScale
        | K::GateSplit
        | K::ArgPartitionTopK
        | K::TakeAlongAxis => KernelKind::Moe,
        K::FusedGateUpSiluMul
        | K::SiluMul
        | K::GeluMul
        | K::SynthGateUpSiluMul
        | K::SynthMlpPreDown => KernelKind::Mlp,
        K::GatherLastToken | K::ScatterFirstToLastRow | K::Softmax | K::SliceTrailingColsU32 => {
            KernelKind::Sample
        }
        K::VisionLayerNorm
        | K::VisionRope
        | K::VisionVarlenAttn
        | K::AvgPool2d
        | K::VisionGelu
        | K::VisionLoadPixels => KernelKind::Vision,
        K::ScalarWeightMul
        | K::ScalarMul
        | K::TanhSoftCap
        | K::Add
        | K::BiasAdd
        | K::Reshape
        | K::TqStageRotated
        | K::TqRotateRows
        | K::TqQuantizeToPacked => KernelKind::Elementwise,
    }
}

/// Bake one bucket's execution plan.
#[allow(clippy::too_many_arguments)]
fn bake_bucket<W: CanonicalParams>(
    bucket_index: usize,
    tape: &LoweredMetalTape,
    arena: &[Buffer],
    splitk_scratch: Option<&Buffer>,
    moe_scratch: Option<&Buffer>,
    roped_k_scratch: Option<&Buffer>,
    attn_unfused_scratch: Option<&Buffer>,
    pipelines: &SpecializedPipelines,
    weights: &W,
    allocator: &MetalAllocator,
    runtime: &RuntimeBindings,
    device: Arc<Device>,
) -> Result<BucketBaking, WorkerError> {
    // The shared arena is sized to the max colored slot count across
    // buckets; this bucket's tape may reference fewer slots (different
    // buckets pick different fusions). Only a tape demanding MORE slots
    // than the arena provides is a real mismatch.
    if tape.num_arena_slots as usize > arena.len() {
        return Err(WorkerError::ArenaShapeMismatch {
            expected: tape.num_arena_slots,
            actual: arena.len(),
        });
    }

    // Pre-pass: harvest every `Binding::Inline { value }` in
    // command-then-binding order, allocate one shared-storage MTLBuffer
    // sized to hold them all at 4-byte stride, write the values, and
    // ⭐ THE LAYER LOOP IS PLAYED OUT HERE, ONCE, AT LOAD. The baked tape carries ONE copy of
    // the body plus its iteration count; unrolling it at BAKE time is what made the emitted
    // `const` an order of magnitude larger. Everything below sees the same flat command list it
    // always did.
    let expanded_commands = tape.commands_expanded();
    let expanded_barriers = tape.barriers_expanded();

    // produce a flat offset list the main resolve loop walks via a
    // cursor. The cursor's invariant: at command `cmd_idx`, before
    // resolving any of its bindings, `inline_cursor` equals the number
    // of Inline bindings seen in commands `0..cmd_idx`. Each Inline
    // binding resolved consumes one slot (cursor += 1) and returns
    // `(moe_inline_buf, cursor_pre * 4, binding_index)`.
    // ⛔ SIZED FROM THE EXPANDED COMMANDS, because the cursor below walks them. The baked tape
    // holds ONE copy of the layer body, so staging from `tape.commands` would size this buffer
    // for a single iteration while the walk consumed one slot per Inline binding per LAYER —
    // every iteration after the first reading past the end.
    let inline_values: Vec<u32> = expanded_commands
        .iter()
        .flat_map(|c| {
            c.command.bindings.iter().filter_map(|b| match b {
                Binding::Inline { value, .. } => Some(*value),
                _ => None,
            })
        })
        .collect();
    let moe_inline_buf: Option<Buffer> = if inline_values.is_empty() {
        None
    } else {
        let n_bytes = inline_values.len() * std::mem::size_of::<u32>();
        let buf = device
            .newBufferWithLength_options(n_bytes, MTLResourceOptions::StorageModeShared)
            .expect("newBufferWithLength_options returned nil (moe inline)");
        // SAFETY: `contents()` returns a host-visible pointer for a
        // shared-storage buffer; we wrote `n_bytes` bytes through it
        // before any GPU command touches the buffer. dispatch recording
        // happens later in the same `bake_bucket`; the dispatch exec runs
        // strictly after this `new` returns to the caller.
        unsafe {
            let dst = buf.contents().as_ptr().cast::<u32>();
            std::ptr::copy_nonoverlapping(inline_values.as_ptr(), dst, inline_values.len());
        }
        Some(buf)
    };
    let mut inline_cursor: u32 = 0;

    let mut steps: Vec<BucketStep> = Vec::new();

    for (cmd_idx, gated) in expanded_commands.iter().enumerate() {
        let cmd = &gated.command;
        if matches!(cmd.kernel, KernelId::Gemm) {
            let dims = cmd.gemm_dims.ok_or(WorkerError::MissingGemmDims {
                bucket_index,
                command_index: cmd_idx,
            })?;
            let (a, b, c) = resolve_gemm_buffers(
                bucket_index,
                cmd_idx,
                cmd,
                arena,
                moe_scratch,
                moe_inline_buf.as_ref(),
                &mut inline_cursor,
                weights,
                allocator,
                runtime,
            )?;
            // f16 → `gemm_f16_specialized` (simdgroup_half8x8), bf16 →
            // `gemm_bf16_specialized` (simdgroup_bfloat8x8). Both are
            // custom MMA kernels routed through the same per-step
            // dispatch plumbing as every other compute kernel — no MPS,
            // no classic command buffer. (MPS rejects BFloat16, and we
            // drive f16 the same way so the backend has one MTL4 GEMM
            // path.)
            match W::METAL_DTYPE {
                MetalDtype::F16 | MetalDtype::Bf16 => {
                    let pipeline = match W::METAL_DTYPE {
                        MetalDtype::F16 => pipelines.pipeline_for_gemm_f16(dims.m, dims.n, dims.k),
                        _ => pipelines.pipeline_for_gemm_bf16(dims.m, dims.n, dims.k),
                    }
                    .map_err(WorkerError::PipelineLookup)?;
                    // gemm_{f16,bf16}_specialized binding contract:
                    //   buffer(0) = output, buffer(1) = input, buffer(2) = weight
                    let bindings_for_cmd: Vec<(Buffer, u64, u64)> = vec![
                        (c.buffer.clone(), c.offset, 0u64),
                        (a.buffer.clone(), a.offset, 1u64),
                        (b.buffer.clone(), b.offset, 2u64),
                    ];
                    // Dispatch: (ceil(N/8), ceil(M/8), 1) threadgroups,
                    // 32 threads (one simdgroup) per threadgroup.
                    let dispatch_for_cmd = (
                        MTLSize {
                            width: (dims.n as u64).div_ceil(8) as usize,
                            height: (dims.m as u64).div_ceil(8) as usize,
                            depth: 1_usize,
                        },
                        MTLSize {
                            width: 32_usize,
                            height: 1_usize,
                            depth: 1_usize,
                        },
                    );
                    let cmd_barrier = expanded_barriers.get(cmd_idx).copied().unwrap_or(true);
                    // Dense GEMM: M is the height axis but the bake here
                    // is for a dense linear that always dispatches at the
                    // actual M (no bucket_m baking), so leave m_scaling
                    // as None.
                    match steps.last_mut() {
                        Some(BucketStep::Dispatch {
                            pipeline: prev,
                            direct_bindings,
                            direct_dispatch,
                            direct_m_scaling,
                            barrier_before,
                            runtime_gate,
                            ..
                        }) if same_pipeline(prev, &pipeline) => {
                            direct_bindings.push(bindings_for_cmd);
                            direct_dispatch.push(dispatch_for_cmd);
                            direct_m_scaling.push(None);
                            barrier_before.push(cmd_barrier);
                            // Gemm path is never gated (no slice); push None
                            // to keep the Vec aligned with `direct_dispatch`.
                            runtime_gate.push(None);
                        }
                        _ => {
                            steps.push(BucketStep::Dispatch {
                                kernel: KernelId::Gemm,
                                pipeline,
                                direct_bindings: vec![bindings_for_cmd],
                                direct_dispatch: vec![dispatch_for_cmd],
                                direct_m_scaling: vec![None],
                                barrier_before: vec![cmd_barrier],
                                runtime_gate: vec![None],
                            });
                        }
                    }
                }
                MetalDtype::Int4 => {
                    return Err(WorkerError::PipelineLookup(
                        super::pipelines::PipelineLookupError::DtypeNotYetWired(
                            KernelId::Gemm,
                            MetalDtype::Int4,
                        ),
                    ));
                }
            }
            continue;
        }

        // The lowering pass baked `library` / `function` / `constants`
        // into the command directly — every per-layer scalar (eps,
        // attn_scale, paging strides) and the `W::METAL_DTYPE`-driven
        // symbol picks happen at lowering time, so this layer is a
        // thin cache lookup.
        let pipeline = pipelines
            .pipeline_for_command::<W>(cmd)
            .map_err(WorkerError::PipelineLookup)?;

        let bound = resolve_bindings(
            bucket_index,
            cmd_idx,
            cmd,
            arena,
            splitk_scratch,
            moe_scratch,
            roped_k_scratch,
            attn_unfused_scratch,
            moe_inline_buf.as_ref(),
            &mut inline_cursor,
            weights,
            allocator,
            runtime,
        )?;
        let bound_refs: Vec<(&Buffer, u64, u64)> =
            bound.iter().map(|(b, off, idx)| (b, *off, *idx)).collect();
        let (tg, tpt) = mtl_size_pair(cmd);

        // Coalesce with the previous step iff (a) it's an dispatch step
        // (a Gemm step forces an encoder boundary) and (b) its
        // pipeline shares the underlying ObjC pointer (specialized
        // pipelines are refcounted — same key returns same handle
        // from the cache).
        let bindings_for_cmd: Vec<(Buffer, u64, u64)> = bound_refs
            .iter()
            .map(|(b, off, idx)| ((*b).clone(), *off, *idx))
            .collect();
        let dispatch_for_cmd = (
            MTLSize {
                width: (tg.width),
                height: (tg.height),
                depth: (tg.depth),
            },
            MTLSize {
                width: (tpt.width),
                height: (tpt.height),
                depth: (tpt.depth),
            },
        );
        let cmd_barrier = expanded_barriers.get(cmd_idx).copied().unwrap_or(true);
        let cmd_gate = gated.gate;
        let cmd_m_scaling = cmd.dispatch.m_scaling;
        // Coalesce only when the gate matches too — a `OnlyIfSingleSeq`
        // dispatch can't share a step with an ungated dispatch since
        // they fire under different runtime conditions.
        let same_gate = match steps.last() {
            Some(BucketStep::Dispatch { runtime_gate, .. }) => {
                runtime_gate.last().copied().unwrap_or(None) == cmd_gate
            }
            _ => false,
        };
        match steps.last_mut() {
            Some(BucketStep::Dispatch {
                pipeline: prev,
                direct_bindings,
                direct_dispatch,
                direct_m_scaling,
                barrier_before,
                runtime_gate,
                ..
            }) if same_pipeline(prev, &pipeline) && same_gate => {
                direct_bindings.push(bindings_for_cmd);
                direct_dispatch.push(dispatch_for_cmd);
                direct_m_scaling.push(cmd_m_scaling);
                barrier_before.push(cmd_barrier);
                runtime_gate.push(cmd_gate);
            }
            _ => {
                steps.push(BucketStep::Dispatch {
                    kernel: cmd.kernel,
                    pipeline,
                    direct_bindings: vec![bindings_for_cmd],
                    direct_dispatch: vec![dispatch_for_cmd],
                    direct_m_scaling: vec![cmd_m_scaling],
                    barrier_before: vec![cmd_barrier],
                    runtime_gate: vec![cmd_gate],
                });
            }
        }
    }

    let mtl4_steps = super::mtl4::bake_mtl4_steps(&steps, &device);
    Ok(BucketBaking {
        bucket_m: tape.bucket_m,
        mtl4_steps,
        moe_inline_buf,
    })
}

/// Resolve `(out, in, weight)` buffers for a `KernelId::Gemm` command.
///
/// The lowering pass guarantees the binding order: index 0 → output
/// arena slot, index 1 → input arena slot, index 2 → LinearLayer
/// weight thunk. Anything else is a contract violation surfaced as
/// [`WorkerError::GemmBindingsMalformed`].
#[allow(clippy::too_many_arguments)]
fn resolve_gemm_buffers<W: CanonicalParams>(
    bucket_index: usize,
    command_index: usize,
    cmd: &LoweredCommand,
    arena: &[Buffer],
    moe_scratch: Option<&Buffer>,
    moe_inline_buf: Option<&Buffer>,
    inline_cursor: &mut u32,
    weights: &W,
    allocator: &MetalAllocator,
    runtime: &RuntimeBindings,
) -> Result<(BoundBuffer, BoundBuffer, BoundBuffer), WorkerError> {
    // KernelId::Gemm never references the SplitK scratch buffer
    // (dense GEMM has its own per-step dispatch path), so pass None.
    // MoE plumbing IS passed through: the §3b lowering emits the MoE
    // router-projection step as `Instruction::Gemm` with a
    // `Binding::MoeScratch` output, so the resolve has to be able to
    // unwrap MoE scratch refs the same as the dispatch path.
    let bound = resolve_bindings(
        bucket_index,
        command_index,
        cmd,
        arena,
        /*splitk_scratch=*/ None,
        moe_scratch,
        /*roped_k_scratch=*/ None,
        /*attn_unfused_scratch=*/ None,
        moe_inline_buf,
        inline_cursor,
        weights,
        allocator,
        runtime,
    )?;
    if bound.len() != 3 {
        return Err(WorkerError::GemmBindingsMalformed {
            bucket_index,
            command_index,
            reason: "expected exactly 3 bindings (out, in, weight)",
        });
    }
    // Bindings are produced in the order the lowering pass listed
    // them; their `binding_index` field carries the encoder slot but
    // we only care about positional ordering. The lowering pass uses
    // 0 = out, 1 = in, 2 = weight.
    let mut iter = bound.into_iter();
    let out = iter.next().expect("bound[0]");
    let inp = iter.next().expect("bound[1]");
    let wt = iter.next().expect("bound[2]");
    Ok((
        BoundBuffer {
            buffer: inp.0,
            offset: inp.1,
        },
        BoundBuffer {
            buffer: wt.0,
            offset: wt.1,
        },
        BoundBuffer {
            buffer: out.0,
            offset: out.1,
        },
    ))
}

/// Resolve a `Binding::Weight` against the loaded model `weights`
/// and the allocator that owns the underlying `MTLBuffer` arenas.
///
/// Calls into the per-arch [`scratchy_ir::WeightAccessors`] impl (emitted by
/// `scratchy-forward-compiler-macro::codegen::emit_weight_accessors_impl`) using
/// the binding's `(bucket, op_idx, slot, layer)` locator to recover the
/// `&Layer` struct (`&RmsNorm`, `&LinearLayer`, `&Embedding`), pulls
/// out the raw GpuTensor pointer matching `which`, and asks the
/// allocator which buffer + offset that pointer belongs to.
///
/// Same shape CUDA's interpreter uses: `WeightAccessors` → layer
/// struct → `GpuTensor`. The Metal-side delta is just the final
/// pointer → `(&Buffer, offset)` reverse lookup against the arena
/// allocator.
fn resolve_weight<W: scratchy_ir::CanonicalParams + scratchy_ir::WeightAccessors>(
    weights: &W,
    allocator: &MetalAllocator,
    kind: &WeightBundleKind,
    layer: u32,
    which: WeightTensor,
    locator: super::lowered::WeightLocator,
) -> Result<(Buffer, u64), WorkerError> {
    let bucket = locator.bucket;
    let op_idx = locator.op_idx;
    let slot = locator.slot;
    let tensor = match kind {
        WeightBundleKind::RmsNorm => weights.rms_norm_at(bucket, op_idx, slot, layer).weight,
        WeightBundleKind::Embedding => weights.embedding_at(bucket, op_idx, slot, layer).weight,
        WeightBundleKind::LinearLayer => {
            let l = weights.linear_at(bucket, op_idx, slot, layer);
            match (which, l) {
                // Dense path
                (WeightTensor::Weight, scratchy_layers::LinearLayer::Dense(_)) => l.dense_weight(),
                (WeightTensor::Bias, scratchy_layers::LinearLayer::Dense(_)) => {
                    l.dense_bias().ok_or(WorkerError::WeightLookupFailed {
                        reason: "LinearLayer bias requested but not present",
                    })?
                }
                // MLX-affine path: distinct accessors per tensor role.
                (WeightTensor::Weight, scratchy_layers::LinearLayer::AffineQuant(_)) => {
                    l.affine_weight()
                }
                (WeightTensor::AffineScales, scratchy_layers::LinearLayer::AffineQuant(_)) => {
                    l.affine_scales()
                }
                (WeightTensor::AffineBiases, scratchy_layers::LinearLayer::AffineQuant(_)) => {
                    l.affine_biases()
                }
                (WeightTensor::AffineLinearBias, scratchy_layers::LinearLayer::AffineQuant(_)) => l
                    .affine_linear_bias()
                    .ok_or(WorkerError::WeightLookupFailed {
                        reason: "AffineQuant linear_bias requested but not present",
                    })?,
                // NVFP4 path: packed weight via the shared `Weight` role,
                // folded per-group scales via `Nvfp4Scales`. No biases.
                (WeightTensor::Weight, scratchy_layers::LinearLayer::Nvfp4(_)) => l.nvfp4_weight(),
                (WeightTensor::Nvfp4Scales, scratchy_layers::LinearLayer::Nvfp4(_)) => {
                    l.nvfp4_scales()
                }
                // Mismatch: affine/nvfp4 WeightTensor on a Dense layer (or
                // vice versa), or any quant arm we don't expect to reach
                // the Metal worker.
                (WeightTensor::AffineScales, _)
                | (WeightTensor::AffineBiases, _)
                | (WeightTensor::AffineLinearBias, _)
                | (WeightTensor::Nvfp4Scales, _) => {
                    return Err(WorkerError::WeightLookupFailed {
                        reason: "AffineScales/AffineBiases/AffineLinearBias/Nvfp4Scales requested \
                                 but LinearLayer is not the matching quant variant",
                    });
                }
                (WeightTensor::Weight | WeightTensor::Bias, _) => {
                    return Err(WorkerError::WeightLookupFailed {
                        reason: "LinearLayer arm not reachable on the Metal worker — \
                                 macro should only emit Dense or AffineQuant on metal",
                    });
                }
                // `Moe*` WeightTensor variants are only valid against
                // `WeightBundleKind::{FusedMoe, SharedFusedMoe}`. The
                // lowering pass should never pair a Moe tensor with a
                // LinearLayer bundle — surface as a config bug rather
                // than a panic.
                (
                    WeightTensor::MoeRouterGate
                    | WeightTensor::MoeExpertGateW
                    | WeightTensor::MoeExpertGateS
                    | WeightTensor::MoeExpertGateB
                    | WeightTensor::MoeExpertUpW
                    | WeightTensor::MoeExpertUpS
                    | WeightTensor::MoeExpertUpB
                    | WeightTensor::MoeExpertDownW
                    | WeightTensor::MoeExpertDownS
                    | WeightTensor::MoeExpertDownB
                    | WeightTensor::MoeSharedGateUpW
                    | WeightTensor::MoeSharedGateUpS
                    | WeightTensor::MoeSharedGateUpB
                    | WeightTensor::MoeSharedDownW
                    | WeightTensor::MoeSharedDownS
                    | WeightTensor::MoeSharedDownB
                    | WeightTensor::MoeSharedExpertGate
                    | WeightTensor::GemmaRouterGate
                    | WeightTensor::GemmaPerExpertScale
                    | WeightTensor::GemmaRouterScale
                    | WeightTensor::GdnConv1d
                    | WeightTensor::GdnALog
                    | WeightTensor::GdnDtBias
                    | WeightTensor::GdnNorm,
                    _,
                ) => {
                    return Err(WorkerError::WeightLookupFailed {
                        reason: "Moe*/Gemma*/Gdn* WeightTensor variant requested against LinearLayer \
                                 bundle — expected FusedMoe / SharedFusedMoe / GemmaRouter / \
                                 GemmaSwitchGlu / GatedDeltaNet bundle",
                    });
                }
            }
        }
        WeightBundleKind::GatedDeltaNet => {
            let l = weights.gated_delta_net_at(bucket, op_idx, slot, layer);
            match which {
                WeightTensor::GdnConv1d => l.conv1d,
                WeightTensor::GdnALog => l.a_log,
                WeightTensor::GdnDtBias => l.dt_bias,
                WeightTensor::GdnNorm => l.norm,
                _ => {
                    return Err(WorkerError::WeightLookupFailed {
                        reason: "non-Gdn* WeightTensor requested against GatedDeltaNet bundle",
                    });
                }
            }
        }
        WeightBundleKind::LayerNorm => {
            // Torch-style LayerNorm-with-bias bundle (vision towers).
            // `which == Weight` → the gain, `which == Bias` → the
            // (mandatory) bias. The `vision_layernorm` kernel binds both
            // (buffer 2 = weight, buffer 3 = bias), so the lowering arm
            // emits one `Weight{Weight}` + one `Weight{Bias}` against the
            // same locator. An absent bias is a loader/DSL mismatch —
            // surface it rather than dropping the bias term.
            let l = weights.layer_norm_at(bucket, op_idx, slot, layer);
            match which {
                WeightTensor::Weight => l.weight,
                WeightTensor::Bias => l.bias.ok_or(WorkerError::WeightLookupFailed {
                    reason: "LayerNorm bundle has no .bias — vision LayerNorm \
                             requires both .weight and .bias",
                })?,
                _ => {
                    return Err(WorkerError::WeightLookupFailed {
                        reason: "non-Weight/Bias WeightTensor requested against LayerNorm bundle",
                    });
                }
            }
        }
        WeightBundleKind::CosSin => weights.cos_sin_at(bucket, op_idx, slot, layer),
        // Spans rope-on-read: cos/sin resolved by layer CLASS (the
        // binding's bucket/op_idx/slot locator is unused). Returns a
        // GpuTensor (the rotary/rotary_local cos_sin cache); the common
        // arena-lookup below maps it to (buffer, offset) like CosSin.
        WeightBundleKind::RopeOnReadCosSin { is_global } => {
            weights.rope_on_read_cos_sin(*is_global, layer)
        }
        // MLX-affine int4 quantized embedding (P6). The lowering's
        // `AffineEmbed` arm always uses `layer = 0` (embed_tokens is
        // not a layered weight) and the kernel expects three buffer
        // bindings: packed weight, scales, biases.
        WeightBundleKind::AffineQuantEmbedding => {
            let e = weights.affine_quant_embedding_at(bucket, op_idx, slot, layer);
            match which {
                WeightTensor::GdnConv1d
                | WeightTensor::GdnALog
                | WeightTensor::GdnDtBias
                | WeightTensor::GdnNorm => {
                    return Err(WorkerError::WeightLookupFailed {
                        reason: "Gdn* WeightTensor requested against AffineQuantEmbedding bundle",
                    });
                }
                WeightTensor::Weight => e.weight,
                WeightTensor::AffineScales => e.scales,
                WeightTensor::AffineBiases => e.affine_biases,
                WeightTensor::Bias | WeightTensor::AffineLinearBias | WeightTensor::Nvfp4Scales => {
                    return Err(WorkerError::WeightLookupFailed {
                        reason: "AffineQuantEmbedding carries only (weight, scales, biases) \
                                 — no linear-layer bias, and NVFP4 embeddings stay dense",
                    });
                }
                WeightTensor::MoeRouterGate
                | WeightTensor::MoeExpertGateW
                | WeightTensor::MoeExpertGateS
                | WeightTensor::MoeExpertGateB
                | WeightTensor::MoeExpertUpW
                | WeightTensor::MoeExpertUpS
                | WeightTensor::MoeExpertUpB
                | WeightTensor::MoeExpertDownW
                | WeightTensor::MoeExpertDownS
                | WeightTensor::MoeExpertDownB
                | WeightTensor::MoeSharedGateUpW
                | WeightTensor::MoeSharedGateUpS
                | WeightTensor::MoeSharedGateUpB
                | WeightTensor::MoeSharedDownW
                | WeightTensor::MoeSharedDownS
                | WeightTensor::MoeSharedDownB
                | WeightTensor::MoeSharedExpertGate
                | WeightTensor::GemmaRouterGate
                | WeightTensor::GemmaPerExpertScale
                | WeightTensor::GemmaRouterScale => {
                    return Err(WorkerError::WeightLookupFailed {
                        reason: "Moe*/Gemma* WeightTensor variant against AffineQuantEmbedding \
                                 bundle — expected FusedMoe / SharedFusedMoe / GemmaRouter / \
                                 GemmaSwitchGlu bundle",
                    });
                }
            }
        }
        // ── FusedMoe / SharedFusedMoe ──────────────────────────────
        //
        // §2: dispatch into the AffineFusedMoELayer /
        // AffineSharedFusedMoELayer struct fields. The Metal worker
        // can only resolve the `Affine` variant — the Dense variant
        // is cuda-only and never reaches this code path (the
        // MetalFusedMoeImpl only claims `OpKind::Moe` tiles whose
        // source weight has `StorageFormat::Affine`, and the macro
        // emits `load_affine` accordingly).
        WeightBundleKind::FusedMoe => {
            let layer_struct = weights.fused_moe_at(bucket, op_idx, slot, layer);
            let affine = match layer_struct {
                scratchy_layers::layers_moe::FusedMoELayer::Affine(a) => a.as_ref(),
                scratchy_layers::layers_moe::FusedMoELayer::Dense(_) => {
                    return Err(WorkerError::WeightLookupFailed {
                        reason: "FusedMoe bundle on Metal worker resolved to Dense variant — \
                                 the macro should emit FusedMoELayer::load_affine for Metal MoE; \
                                 reaching the Dense arm means a Dense BF16 checkpoint loaded \
                                 through the Metal MoE path",
                    });
                }
            };
            match which {
                WeightTensor::MoeRouterGate => affine.router_gate,
                WeightTensor::MoeExpertGateW => affine.expert_gate_w,
                WeightTensor::MoeExpertGateS => affine.expert_gate_scales,
                WeightTensor::MoeExpertGateB => affine.expert_gate_biases,
                WeightTensor::MoeExpertUpW => affine.expert_up_w,
                WeightTensor::MoeExpertUpS => affine.expert_up_scales,
                WeightTensor::MoeExpertUpB => affine.expert_up_biases,
                WeightTensor::MoeExpertDownW => affine.expert_down_w,
                WeightTensor::MoeExpertDownS => affine.expert_down_scales,
                WeightTensor::MoeExpertDownB => affine.expert_down_biases,
                // FusedMoe (Mixtral-style, no shared expert) cannot
                // legitimately request Shared* tensors. The macro
                // would only emit those bindings against
                // SharedFusedMoe — surface as a lowering bug.
                WeightTensor::MoeSharedGateUpW
                | WeightTensor::MoeSharedGateUpS
                | WeightTensor::MoeSharedGateUpB
                | WeightTensor::MoeSharedDownW
                | WeightTensor::MoeSharedDownS
                | WeightTensor::MoeSharedDownB
                | WeightTensor::MoeSharedExpertGate => {
                    return Err(WorkerError::WeightLookupFailed {
                        reason: "MoeShared* WeightTensor requested against FusedMoe bundle — \
                                 use SharedFusedMoe bundle for shared-expert tensors",
                    });
                }
                WeightTensor::Weight
                | WeightTensor::Bias
                | WeightTensor::AffineScales
                | WeightTensor::AffineBiases
                | WeightTensor::AffineLinearBias
                | WeightTensor::Nvfp4Scales
                | WeightTensor::GemmaRouterGate
                | WeightTensor::GemmaPerExpertScale
                | WeightTensor::GemmaRouterScale
                | WeightTensor::GdnConv1d
                | WeightTensor::GdnALog
                | WeightTensor::GdnDtBias
                | WeightTensor::GdnNorm => {
                    return Err(WorkerError::WeightLookupFailed {
                        reason: "non-Moe WeightTensor variant requested against FusedMoe bundle",
                    });
                }
            }
        }
        WeightBundleKind::SharedFusedMoe => {
            let layer_struct = weights.shared_fused_moe_at(bucket, op_idx, slot, layer);
            let shared = match layer_struct {
                scratchy_layers::layers_moe::SharedFusedMoELayer::Affine(a) => a.as_ref(),
                scratchy_layers::layers_moe::SharedFusedMoELayer::Dense(_) => {
                    return Err(WorkerError::WeightLookupFailed {
                        reason: "SharedFusedMoe bundle on Metal worker resolved to Dense variant",
                    });
                }
            };
            let routed = &shared.routed;
            // Lowering only emits shared-* bindings when the layer's
            // `shared_intermediate_size > 0`. None at runtime → the
            // loader was built with shared_inter=0 but the macro
            // baked a shared-expert tail into the Instruction
            // payload anyway → lowering / loader split bug.
            let shared_or_err = |opt: Option<scratchy_tensors::tensor::GpuTensor>| {
                opt.ok_or(WorkerError::WeightLookupFailed {
                    reason: "SharedFusedMoe shared-expert tensor binding fired but \
                                 layer was loaded with shared_intermediate_size=0",
                })
            };
            match which {
                WeightTensor::MoeRouterGate => routed.router_gate,
                WeightTensor::MoeExpertGateW => routed.expert_gate_w,
                WeightTensor::MoeExpertGateS => routed.expert_gate_scales,
                WeightTensor::MoeExpertGateB => routed.expert_gate_biases,
                WeightTensor::MoeExpertUpW => routed.expert_up_w,
                WeightTensor::MoeExpertUpS => routed.expert_up_scales,
                WeightTensor::MoeExpertUpB => routed.expert_up_biases,
                WeightTensor::MoeExpertDownW => routed.expert_down_w,
                WeightTensor::MoeExpertDownS => routed.expert_down_scales,
                WeightTensor::MoeExpertDownB => routed.expert_down_biases,
                WeightTensor::MoeSharedGateUpW => shared_or_err(shared.shared_gate_up_w)?,
                WeightTensor::MoeSharedGateUpS => shared_or_err(shared.shared_gate_up_scales)?,
                WeightTensor::MoeSharedGateUpB => shared_or_err(shared.shared_gate_up_biases)?,
                WeightTensor::MoeSharedDownW => shared_or_err(shared.shared_down_w)?,
                WeightTensor::MoeSharedDownS => shared_or_err(shared.shared_down_scales)?,
                WeightTensor::MoeSharedDownB => shared_or_err(shared.shared_down_biases)?,
                WeightTensor::MoeSharedExpertGate => shared_or_err(shared.shared_expert_gate)?,
                WeightTensor::Weight
                | WeightTensor::Bias
                | WeightTensor::AffineScales
                | WeightTensor::AffineBiases
                | WeightTensor::AffineLinearBias
                | WeightTensor::Nvfp4Scales
                | WeightTensor::GemmaRouterGate
                | WeightTensor::GemmaPerExpertScale
                | WeightTensor::GemmaRouterScale
                | WeightTensor::GdnConv1d
                | WeightTensor::GdnALog
                | WeightTensor::GdnDtBias
                | WeightTensor::GdnNorm => {
                    return Err(WorkerError::WeightLookupFailed {
                        reason: "non-Moe WeightTensor variant requested against SharedFusedMoe \
                                 bundle",
                    });
                }
            }
        }
        // ── Gemma-4 router bundle (GemmaMoe op) ─────────────────────
        //
        // Resolves the dequant'd dense router gate, the per-expert score
        // scale, and the RMSNorm gain off the GemmaRouterLayer struct.
        WeightBundleKind::GemmaRouter => {
            let r = weights.gemma_router_at(bucket, op_idx, slot, layer);
            match which {
                WeightTensor::GemmaRouterGate => r.gate,
                WeightTensor::GemmaPerExpertScale => r.per_expert_scale,
                WeightTensor::GemmaRouterScale => r.scale,
                _ => {
                    return Err(WorkerError::WeightLookupFailed {
                        reason: "non-GemmaRouter* WeightTensor requested against GemmaRouter bundle",
                    });
                }
            }
        }
        // ── Gemma-4 SwitchGLU experts bundle (GemmaMoe op) ──────────
        //
        // Reuses the MoeExpert* WeightTensor variants (same expert-major
        // affine layout) against the router-less SwitchGluExpertsLayer.
        WeightBundleKind::GemmaSwitchGlu => {
            let e = weights.gemma_switch_glu_at(bucket, op_idx, slot, layer);
            match which {
                WeightTensor::MoeExpertGateW => e.expert_gate_w,
                WeightTensor::MoeExpertGateS => e.expert_gate_scales,
                WeightTensor::MoeExpertGateB => e.expert_gate_biases,
                WeightTensor::MoeExpertUpW => e.expert_up_w,
                WeightTensor::MoeExpertUpS => e.expert_up_scales,
                WeightTensor::MoeExpertUpB => e.expert_up_biases,
                WeightTensor::MoeExpertDownW => e.expert_down_w,
                WeightTensor::MoeExpertDownS => e.expert_down_scales,
                WeightTensor::MoeExpertDownB => e.expert_down_biases,
                _ => {
                    return Err(WorkerError::WeightLookupFailed {
                        reason: "non-MoeExpert* WeightTensor requested against GemmaSwitchGlu \
                                 bundle (the router lives in the separate GemmaRouter bundle)",
                    });
                }
            }
        }
    };
    allocator
        .buffer_for(tensor.raw_ptr())
        .ok_or(WorkerError::WeightLookupFailed {
            reason: "weight pointer not in any MetalAllocator arena \
                     — was it loaded through this allocator?",
        })
}

/// Resolve every binding on `cmd` to (buffer, offset, binding-index).
///
/// Returns owned `Buffer` clones (cheap ObjC refcount) so callers
/// don't have to thread the [`MetalAllocator`]'s arenas-`Mutex` lock
/// guard through to the encoder.
#[allow(clippy::too_many_arguments)]
fn resolve_bindings<W: CanonicalParams>(
    bucket_index: usize,
    command_index: usize,
    cmd: &LoweredCommand,
    arena: &[Buffer],
    splitk_scratch: Option<&Buffer>,
    moe_scratch: Option<&Buffer>,
    roped_k_scratch: Option<&Buffer>,
    attn_unfused_scratch: Option<&Buffer>,
    moe_inline_buf: Option<&Buffer>,
    inline_cursor: &mut u32,
    weights: &W,
    allocator: &MetalAllocator,
    runtime: &RuntimeBindings,
) -> Result<Vec<(Buffer, u64, u64)>, WorkerError> {
    let mut out: Vec<(Buffer, u64, u64)> = Vec::with_capacity(cmd.bindings.len());
    for binding in cmd.bindings {
        let (buf, off, idx) = match binding {
            // Static-tape sources resolve through the tape's generated
            // resolver before this legacy per-dispatch path runs; a
            // Source binding reaching HERE means a static tape was fed
            // to the legacy resolver — a wiring bug, refused loudly.
            Binding::Source { .. } => {
                return Err(WorkerError::WeightLookupFailed {
                    reason: "Binding::Source reached the legacy binding resolver \
                             (static tapes resolve sources at load, not per dispatch)",
                });
            }
            Binding::ArenaSlot {
                slot,
                binding_index,
            } => {
                let s = *slot as usize;
                if s >= arena.len() {
                    return Err(WorkerError::ArenaSlotOutOfRange {
                        bucket_index,
                        command_index,
                        slot: *slot,
                        arena_len: arena.len(),
                    });
                }
                (arena[s].clone(), 0u64, *binding_index as u64)
            }
            Binding::Weight {
                kind,
                which,
                layer,
                locator,
                binding_index,
            } => {
                // MRoPE text decoders (Qwen3.5-VL) override the static
                // cos/sin RotaryCache with a per-forward, per-token
                // band-split table built host-side (option (b)). The
                // table lives in `runtime.mrope_cos_sin` and the rope
                // kernel reads it with identity positions (`positions[t]
                // = t`), so redirect the baked `WeightBundleKind::CosSin`
                // pointer to that runtime buffer. Compile-time gated on
                // `W::MROPE_SECTION` — folds away (zero cost, no behavior
                // change) on every 1D-rope arch. Covers all cos/sin
                // consumers at once (`RopeAppend`, `FusedQkvRopeCache`,
                // the affine-fused QKV-rope path) since they all bind
                // `CosSin` here.
                if W::MROPE_SECTION.is_some() && matches!(kind, WeightBundleKind::CosSin) {
                    (
                        runtime
                            .buffer_for(super::lowered::RuntimeBindingKind::MropeCosSin)
                            .clone(),
                        0u64,
                        *binding_index as u64,
                    )
                } else {
                    let (b, off) =
                        resolve_weight(weights, allocator, kind, layer.get(), *which, *locator)?;
                    (b, off, *binding_index as u64)
                }
            }
            Binding::Runtime {
                kind,
                binding_index,
            } => {
                let buf = runtime.buffer_for(*kind).clone();
                (buf, 0u64, *binding_index as u64)
            }
            Binding::Scratch { binding_index } => {
                let scratch = splitk_scratch.ok_or(WorkerError::ScratchBufferMissing {
                    bucket_index,
                    command_index,
                })?;
                (scratch.clone(), 0u64, *binding_index as u64)
            }
            Binding::RopedKScratch { binding_index } => {
                // Spans rope-on-read (NAX): the shared pre-roped-K buffer
                // (RopeOnceNax writes it at slot 0; the NAX attention reads
                // it at slot 7). Sized to `roped_k_scratch_bytes`.
                let scratch = roped_k_scratch.ok_or(WorkerError::ScratchBufferMissing {
                    bucket_index,
                    command_index,
                })?;
                (scratch.clone(), 0u64, *binding_index as u64)
            }
            Binding::AttnUnfusedScratch {
                offset,
                binding_index,
            } => {
                let scratch = attn_unfused_scratch.ok_or(WorkerError::ScratchBufferMissing {
                    bucket_index,
                    command_index,
                })?;
                (scratch.clone(), *offset as u64, *binding_index as u64)
            }
            Binding::Inline {
                binding_index,
                value: _,
            } => {
                let buf = moe_inline_buf.ok_or(WorkerError::WeightLookupFailed {
                    reason: "Binding::Inline reached resolve_bindings but the bucket's \
                             `moe_inline_buf` is None — bake_bucket's pre-pass should have \
                             allocated it from the tape's Inline bindings",
                })?;
                let off = (*inline_cursor as u64) * 4;
                *inline_cursor += 1;
                (buf.clone(), off, *binding_index as u64)
            }
            Binding::MoeScratch {
                binding_index,
                byte_offset,
            } => {
                let buf = moe_scratch.ok_or(WorkerError::WeightLookupFailed {
                    reason: "Binding::MoeScratch reached worker but `moe_scratch` is None — \
                             a MoE instruction lowered without `LoweredMetalTape::moe_scratch_bytes` \
                             being set (lowering pass bug)",
                })?;
                (buf.clone(), *byte_offset as u64, *binding_index as u64)
            }
        };
        out.push((buf, off, idx));
    }
    Ok(out)
}

/// Rewrite the m-axis of a baked threadgroup grid to match the
/// actual `num_tokens` of this forward, instead of the bucket_m the
/// grid was baked against.
///
/// `scaling = None` means the kernel's grid doesn't scale with M
/// (or the lowering pass hasn't yet been taught to emit a scaling
/// hint for it) — return the baked grid unchanged. When `scaling =
/// Some(MScaling { axis, tile })`, replace `tg.{axis}` with
/// `num_tokens.div_ceil(tile)`, clamped so we never grow above the
/// baked value (guards against `num_tokens > bucket_m`, which the
/// bucket picker already rules out but defense-in-depth).
/// Evaluate a per-dispatch [`RuntimeGate`] against the live
/// `num_seqs` of this forward (= `cu_seqlens_q.len() - 1`).
/// Returns `true` when the dispatch should fire, `false` when it
/// should be skipped. `gate == None` (the common case) always
/// fires.
///
/// `OnlyIfSingleSeq` fires when there's exactly one sequence in
/// the bucket — either pure single-seq prefill or a one-token
/// decode forward. The lm_head slice's gather/qmv/scatter trio is
/// gated this way.
///
/// `OnlyIfMultiSeq` fires when the bucket holds multiple
/// sequences (batched decode, mixed prefill+decode). The
/// full-`M=bucket_m` lm_head fallback is gated this way so the
/// slice's per-seq-incorrect logits get overwritten with a
/// correct multi-row GEMM result.
pub(super) fn gate_matches(
    gate: Option<super::lowered::RuntimeGate>,
    num_tokens: u32,
    num_seqs: u32,
    has_spec_tokens: bool,
    turboquant: bool,
) -> bool {
    // lm_head slice (`OnlyIfNoSpec`) fires only when there are EXTRA
    // tokens to drop (prefill / chunked-prefill / mixed batches);
    // steady-state decode has `num_tokens == num_seqs` and slicing
    // would just add 2 kernel launches with no GEMM-work savings
    // (qmv at M=num_seqs == qmm at M=num_seqs). Verified: gating
    // unconditionally on multi-seq regressed c=4 TPOT by +5% on
    // Llama-1B; gating on `num_tokens > num_seqs` keeps the prefill
    // win without hurting decode.
    let turboquant_decode = turboquant && num_tokens == num_seqs;
    match gate {
        None => true,
        Some(super::lowered::RuntimeGate::OnlyIfNoSpec) => {
            !has_spec_tokens && num_tokens > num_seqs
        }
        Some(super::lowered::RuntimeGate::OnlyIfSpec) => has_spec_tokens,
        Some(super::lowered::RuntimeGate::OnlyIfTurboquant) => turboquant,
        Some(super::lowered::RuntimeGate::OnlyIfTurboquantDecode) => turboquant_decode,
        Some(super::lowered::RuntimeGate::OnlyIfTurboquantNotDecode) => {
            turboquant && !turboquant_decode
        }
        Some(super::lowered::RuntimeGate::UnlessTurboquantDecode) => !turboquant_decode,
    }
}

fn scale_tg_for_num_tokens(
    mut tg: MTLSize,
    scaling: Option<super::lowered::MScaling>,
    num_tokens: super::ids::NumTokens,
    num_seqs: u32,
) -> MTLSize {
    let Some(s) = scaling else {
        return tg;
    };
    let bm = s.bucket_m.get().max(1) as u64;
    let n = (num_tokens.get().max(1) as u64).min(bm);
    let slot = match s.axis {
        super::lowered::MScaleAxis::X => &mut tg.width,
        super::lowered::MScaleAxis::Y => &mut tg.height,
        super::lowered::MScaleAxis::Z => &mut tg.depth,
    };
    // new = ceil(baseline * n / bucket_m). Clamped above to the
    // baseline so accidental num_tokens > bucket_m can't grow the
    // grid past what was baked.
    let baseline = *slot as u64;
    let scaled = baseline.saturating_mul(n).div_ceil(bm);
    *slot = scaled as usize;
    // `seq_axis`: SET (not scale) the chosen axis to the live num_seqs.
    // The steel paged prefill kernel needs one grid-Z layer per sequence
    // (`tid.z = seq_idx`) so a BQ-block tile never straddles a sequence
    // boundary; over-dispatched (seq, q-block) pairs early-out in the
    // kernel. See `MScaling::seq_axis`.
    if let Some(seq_ax) = s.seq_axis {
        let seq_slot = match seq_ax {
            super::lowered::MScaleAxis::X => &mut tg.width,
            super::lowered::MScaleAxis::Y => &mut tg.height,
            super::lowered::MScaleAxis::Z => &mut tg.depth,
        };
        *seq_slot = num_seqs.max(1) as usize;
    }
    tg
}

fn mtl_size_pair(cmd: &LoweredCommand) -> (MTLSize, MTLSize) {
    let tg = MTLSize {
        width: cmd.dispatch.threadgroups.0 as usize,
        height: cmd.dispatch.threadgroups.1 as usize,
        depth: cmd.dispatch.threadgroups.2 as usize,
    };
    let tpt = MTLSize {
        width: cmd.dispatch.threads_per_threadgroup.0 as usize,
        height: cmd.dispatch.threads_per_threadgroup.1 as usize,
        depth: cmd.dispatch.threads_per_threadgroup.2 as usize,
    };
    (tg, tpt)
}

/// Compare two `ComputePipelineState`s by ObjC handle. Cached
/// pipelines for the same `(kernel, bucket, extras)` tuple are
/// pointer-equal, so this is the right test for segment coalescing.
fn same_pipeline(a: &ComputePipelineState, b: &ComputePipelineState) -> bool {
    std::ptr::eq(
        Retained::as_ptr(a) as *const _,
        Retained::as_ptr(b) as *const _,
    )
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::interpreter::metal::lowered::{
        Binding, DispatchShape, LoweredCommand, RuntimeBindingKind, WeightBundleKind,
        WeightLocator, WeightTensor,
    };
    use crate::specialized_pipeline_cache::{ConstantValue, SpecializedPipelineCache};
    use scratchy_ir::CanonicalParams;
    use scratchy_layers::{Linear, LinearLayer, RmsNorm};
    use scratchy_tensors::{DType, DeviceAllocator, GpuTensor};
    use std::sync::Arc;

    /// Test fixture: holds `CanonicalParams` constants AND the layer
    /// instances the `WeightAccessors` impl below returns. Plays the
    /// role of the per-canonical `Weights` struct the macro will emit.
    struct TestWeights {
        rmsnorm_layer: RmsNorm,
        linear_layer: LinearLayer,
    }
    impl CanonicalParams for TestWeights {
        const HEAD_DIM: u32 = 64;
        const NUM_Q_HEADS: u32 = 32;
        const NUM_KV_HEADS: u32 = 4;
        const Q_SIZE: usize = 2048;
        const KV_SIZE: usize = 256;
        const INTERMEDIATE_SIZE: usize = 5632;
        const ATTN_SCALE: f32 = 0.125;
        const ATTN_SOFTCAP: f32 = 0.0;
        const SLIDING_WINDOW: i32 = -1;
        const KV_LORA_RANK: usize = 0;
        const QK_NOPE_HEAD_DIM: usize = 0;
        const QK_ROPE_HEAD_DIM: usize = 0;
        const V_HEAD_DIM: usize = 0;
        const FINAL_LOGIT_SOFTCAPPING: f32 = 0.0;
        const QK_HEAD_DIM: usize = 0;
        const MLA_ATTN_SCALE: f32 = 0.0;
        // Pin the smoke tests to the f16 / MPS GEMM path. The
        // `worker_routes_gemm_step` / `worker_interleaves_gemm_with_icb`
        // tests below assert specifically on the MPS GEMM branch. The
        // bf16 dispatch GEMM routing has its own coverage in the e2e tests.
        const METAL_DTYPE: MetalDtype = MetalDtype::F16;
    }

    // Weight resolution now flows through `WeightAccessors` (the WtFn
    // thunk seam is gone). The synthetic tapes below bind `RmsNorm` and
    // `LinearLayer` weight bundles, so this probe overrides exactly
    // those two accessors to return its held layer instances; every
    // other accessor keeps the trait's `unreachable!` default.
    impl scratchy_ir::WeightAccessors for TestWeights {
        fn rms_norm_at(&self, _bucket: u32, _op_idx: u32, _slot: u32, _layer: u32) -> &RmsNorm {
            &self.rmsnorm_layer
        }
        fn linear_at(&self, _bucket: u32, _op_idx: u32, _slot: u32, _layer: u32) -> &LinearLayer {
            &self.linear_layer
        }
    }

    /// Build a `TestWeights` + the `MetalAllocator` that owns its
    /// MTLBuffer arenas. The allocator is also used by the pool to
    /// resolve `Binding::Weight` lookups; tests that build a worker
    /// directly thread the same allocator into `MetalWorker::new`.
    ///
    /// Allocations are zero-filled — sufficient for verifying the
    /// recording flow. Numerical correctness lives in `pipelines.rs`'s
    /// `*_matches_cpu_golden` tests.
    fn build_test_weights() -> (Arc<TestWeights>, Arc<MetalAllocator>) {
        let device = crate::detect_device()
            .expect("test fixture: a Metal device")
            .device
            .clone();
        let mut allocator = MetalAllocator::new(device);

        // RmsNorm weight: [Q_SIZE] f16 = 4 KB.
        let rmsnorm_bytes = vec![0u8; TestWeights::Q_SIZE * 2];
        let rmsnorm_ptr = unsafe {
            allocator
                .alloc_and_copy_host(rmsnorm_bytes.as_ptr(), rmsnorm_bytes.len())
                .expect("rmsnorm tensor")
        };
        let rmsnorm_tensor =
            unsafe { GpuTensor::new(rmsnorm_ptr, &[TestWeights::Q_SIZE], DType::F16) };

        // LinearLayer weight: [Q_SIZE, Q_SIZE] f16 = ~8 MB. Sized to
        // the largest TinyLlama-class projection so the dense-GEMM dim
        // checks pass.
        let linear_bytes = vec![0u8; TestWeights::Q_SIZE * TestWeights::Q_SIZE * 2];
        let linear_ptr = unsafe {
            allocator
                .alloc_and_copy_host(linear_bytes.as_ptr(), linear_bytes.len())
                .expect("linear tensor")
        };
        let linear_tensor = unsafe {
            GpuTensor::new(
                linear_ptr,
                &[TestWeights::Q_SIZE, TestWeights::Q_SIZE],
                DType::F16,
            )
        };

        let weights = Arc::new(TestWeights {
            rmsnorm_layer: RmsNorm::new(rmsnorm_tensor, 1e-5),
            linear_layer: LinearLayer::Dense(Linear::new(linear_tensor, None)),
        });
        (weights, Arc::new(allocator))
    }

    fn alloc_buffer(device: &Device, bytes: u64) -> Buffer {
        device
            .newBufferWithLength_options(
                bytes.max(1) as usize,
                MTLResourceOptions::StorageModeShared,
            )
            .expect("newBufferWithLength_options returned nil")
    }

    fn empty_runtime(device: &Device, num_layers: usize) -> RuntimeBindings {
        RuntimeBindings {
            input_ids: alloc_buffer(device, 16),
            positions: alloc_buffer(device, 16),
            slot_mappings: vec![alloc_buffer(device, 16)],
            cu_seqlens_q: alloc_buffer(device, 16),
            seq_used_k: alloc_buffer(device, 16),
            span_ids: alloc_buffer(device, 16),
            block_tables: vec![alloc_buffer(device, 16)],
            layer_to_group: Vec::new(),
            kv_cache_k: (0..num_layers).map(|_| alloc_buffer(device, 16)).collect(),
            kv_cache_v: (0..num_layers).map(|_| alloc_buffer(device, 16)).collect(),
            block_unrotated_flags: (0..num_layers).map(|_| alloc_buffer(device, 16)).collect(),
            tq: None,
            num_tokens_u32: alloc_buffer(device, 4),
            num_sample_rows_u32: alloc_buffer(device, 4),
            sample_indices: alloc_buffer(device, 16),
            gdn_state_conv: ::std::vec::Vec::new(),
            gdn_state_ssm: ::std::vec::Vec::new(),
            gdn_state_indices: alloc_buffer(device, 16),
            gdn_is_fresh: alloc_buffer(device, 16),
            vision_rope_freqs: alloc_buffer(device, 16),
            pixels: alloc_buffer(device, 16),
            vision_pos_embeds: alloc_buffer(device, 16),
            mm_embeds: alloc_buffer(device, 16),
            mm_dst_rows: alloc_buffer(device, 16),
            mrope_cos_sin: alloc_buffer(device, 16),
            vision_cu_seqlens_full: alloc_buffer(device, 16),
            vision_cu_seqlens_window: alloc_buffer(device, 16),
            vision_window_index: alloc_buffer(device, 16),
            vision_reverse_indices: alloc_buffer(device, 16),
            vision_position_ids: alloc_buffer(device, 16),
        }
    }

    /// Build a synthetic 2-bucket lowered tape: each bucket runs the
    /// same `[RmsNorm, FusedAddRmsNorm]` shape twice. Verifies:
    /// arena allocation, dispatch recording per command, segment
    /// coalescing across same-pipeline neighbours.
    fn build_synthetic_tape(bucket_m: u32) -> LoweredMetalTape {
        let rmsnorm = LoweredCommand {
            kernel: KernelId::RmsNorm,
            library: "rmsnorm",
            function: "rmsnorm_f16_s_f16_specialized",
            constants: crate::interpreter::metal::lowered::baked(vec![
                ConstantValue::uint(0, bucket_m),
                ConstantValue::uint(1, <TestWeights as CanonicalParams>::Q_SIZE as u32),
                ConstantValue::float(2, <TestWeights as CanonicalParams>::RMS_NORM_EPS),
            ]),
            dispatch: DispatchShape {
                threadgroups: (bucket_m, 1, 1),
                threads_per_threadgroup: (256, 1, 1),
                m_scaling: None,
            },
            bindings: crate::interpreter::metal::lowered::baked(vec![
                Binding::ArenaSlot {
                    slot: 0,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: 1,
                    binding_index: 1,
                },
                Binding::Weight {
                    kind: WeightBundleKind::RmsNorm,
                    which: WeightTensor::Weight,
                    layer: crate::interpreter::metal::ids::LayerId(0),
                    locator: WeightLocator {
                        bucket: 0,
                        op_idx: 0,
                        slot: 0,
                    },
                    binding_index: 2,
                },
            ]),
            gemm_dims: None,
        };
        let fused_add_rmsnorm = LoweredCommand {
            kernel: KernelId::FusedAddRmsNorm,
            library: "fused_add_rmsnorm",
            function: "fused_add_rmsnorm_f16_s_f16_specialized",
            constants: crate::interpreter::metal::lowered::baked(vec![
                ConstantValue::uint(0, bucket_m),
                ConstantValue::uint(1, <TestWeights as CanonicalParams>::Q_SIZE as u32),
                ConstantValue::float(2, <TestWeights as CanonicalParams>::RMS_NORM_EPS),
            ]),
            dispatch: DispatchShape {
                threadgroups: (bucket_m, 1, 1),
                threads_per_threadgroup: (256, 1, 1),
                m_scaling: None,
            },
            bindings: crate::interpreter::metal::lowered::baked(vec![
                Binding::ArenaSlot {
                    slot: 0,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: 1,
                    binding_index: 1,
                },
                Binding::Weight {
                    kind: WeightBundleKind::RmsNorm,
                    which: WeightTensor::Weight,
                    layer: crate::interpreter::metal::ids::LayerId(0),
                    locator: WeightLocator {
                        bucket: 0,
                        op_idx: 0,
                        slot: 0,
                    },
                    binding_index: 2,
                },
            ]),
            gemm_dims: None,
        };
        // Two RmsNorm commands then two FusedAddRmsNorm commands —
        // exercises both kernels and the coalescer's same-pipeline
        // neighbour case (two RmsNorm with identical extras hit the
        // same cached pipeline; same for the FAR pair).
        LoweredMetalTape {
            bucket_m,
            num_arena_slots: 2,
            commands: crate::interpreter::metal::lowered::baked(vec![
                rmsnorm.into(),
                rmsnorm.into(),
                fused_add_rmsnorm.into(),
                fused_add_rmsnorm.into(),
            ]),
            splitk_scratch_bytes: 0,
            moe_scratch_bytes: 0,
            roped_k_scratch_bytes: 0,
            attn_unfused_scratch_bytes: 0,
            barrier_before: &[],
            // Straight-line: this fixture is one hand-built body, no rolled layer loop.
            loops: &[],
        }
    }

    /// Number of sub-dispatches a baked MTL4 step encodes — the modern
    /// observable that replaces the old `BucketStep::Dispatch { range }`
    /// width (one dispatch per coalesced command). Panics if MTL4 was
    /// not baked (a bucket carrying an f16 MPS GEMM has `mtl4_steps ==
    /// None`); the dispatch-only smoke tests below always bake MTL4.
    fn mtl4_step_dispatch_count(baking: &BucketBaking, step: usize) -> usize {
        baking
            .mtl4_steps
            .as_ref()
            .expect("dispatch-only bucket bakes MTL4 steps")[step]
            .dispatches
            .len()
    }

    #[test]
    fn worker_builds_and_segments_coalesce() {
        let Some(device) = crate::detect_device().filter(|_| crate::metal4_available()) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let device = Arc::new(device.device.clone());

        let cache = Arc::new(
            SpecializedPipelineCache::with_standard_shaders((*device).clone())
                .expect("compile standard shaders"),
        );
        let pipelines = SpecializedPipelines::new(cache);

        let (weights, allocator) = build_test_weights();
        let runtime = empty_runtime(&device, 1);

        // Two buckets: M=1 (decode) and M=8 (small prefill).
        let tapes = vec![build_synthetic_tape(1), build_synthetic_tape(8)];
        let arena_layout: ArenaLayout = vec![4 * 1024, 4 * 1024];

        let worker = MetalWorker::<TestWeights>::new(
            device,
            &arena_layout,
            &tapes,
            &pipelines,
            &weights,
            &allocator,
            &runtime,
        )
        .expect("worker builds");

        assert_eq!(worker.arena.len(), 2);
        assert_eq!(worker.bucket_bakings.len(), 2);

        // Each bucket has 4 commands: 2 RmsNorm then 2 FusedAddRmsNorm.
        // Adjacent same-kernel-with-same-extras commands must coalesce
        // into one segment (pipeline-pointer identity); cross-kernel
        // boundary forces a new segment. So: 2 dispatch steps per bucket,
        // no GEMM steps.
        for baking in &worker.bucket_bakings {
            let steps = baking
                .mtl4_steps
                .as_ref()
                .expect("dispatch-only bucket bakes MTL4 steps");
            assert_eq!(
                steps.len(),
                2,
                "expected RmsNorm + FusedAddRmsNorm coalesced"
            );
            // Each coalesced step carries its sub-dispatches: the two
            // RmsNorm commands collapse into step 0 (2 dispatches) and
            // the two FusedAddRmsNorm into step 1 (2 dispatches) —
            // the modern observable for the old `0..2` / `2..4` ranges.
            assert_eq!(mtl4_step_dispatch_count(baking, 0), 2);
            assert_eq!(mtl4_step_dispatch_count(baking, 1), 2);
        }
    }

    #[test]
    fn arena_shape_mismatch_errors() {
        let Some(device) = crate::detect_device().filter(|_| crate::metal4_available()) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let device = Arc::new(device.device.clone());

        let cache = Arc::new(
            SpecializedPipelineCache::with_standard_shaders((*device).clone())
                .expect("compile standard shaders"),
        );
        let pipelines = SpecializedPipelines::new(cache);
        let (weights, allocator) = build_test_weights();
        let runtime = empty_runtime(&device, 1);

        let tapes = vec![build_synthetic_tape(1)]; // num_arena_slots = 2

        // Layout has only 1 slot — should error.
        let bad_layout: ArenaLayout = vec![4 * 1024];
        let err = MetalWorker::<TestWeights>::new(
            device,
            &bad_layout,
            &tapes,
            &pipelines,
            &weights,
            &allocator,
            &runtime,
        )
        .err()
        .expect("expected arena shape mismatch error");
        assert!(matches!(
            err,
            WorkerError::ArenaShapeMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }

    /// Phase 5.C.4 smoke test: the worker bakes an AttentionViaCache
    /// command into a one-segment dispatch. Verifies that the new
    /// `attention_via_cache_v2_f16_specialized` kernel resolves through
    /// the specialized-pipeline cache and that the worker accepts the
    /// runtime bindings the lowering pass produces (Q + seq_used_k +
    /// block_table + per-layer kv_cache_k/v).
    #[test]
    fn worker_records_attention_via_cache() {
        let Some(device) = crate::detect_device().filter(|_| crate::metal4_available()) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let device = Arc::new(device.device.clone());

        let cache = Arc::new(
            SpecializedPipelineCache::with_standard_shaders((*device).clone())
                .expect("compile standard shaders"),
        );
        let pipelines = SpecializedPipelines::new(cache);

        let (weights, allocator) = build_test_weights();
        let runtime = empty_runtime(&device, 1);

        // Single AttentionViaCache command at decode bucket=1
        // (batch=1, num_q_heads heads).
        let attn = LoweredCommand {
            kernel: KernelId::AttentionViaCache,
            library: "attention",
            function: "attention_via_cache_v2_f16_specialized",
            constants: crate::interpreter::metal::lowered::baked(vec![
                ConstantValue::uint(0, TestWeights::HEAD_DIM),
                ConstantValue::uint(1, TestWeights::NUM_Q_HEADS),
                ConstantValue::uint(2, TestWeights::NUM_KV_HEADS),
                ConstantValue::float(3, TestWeights::ATTN_SCALE),
                ConstantValue::uint(4, TestWeights::BLOCK_SIZE),
                ConstantValue::uint(5, TestWeights::MAX_BLOCKS_PER_SEQ),
            ]),
            dispatch: DispatchShape {
                threadgroups: (1, TestWeights::NUM_Q_HEADS, 1),
                threads_per_threadgroup: (TestWeights::HEAD_DIM, 1, 1),
                m_scaling: None,
            },
            bindings: crate::interpreter::metal::lowered::baked(vec![
                Binding::ArenaSlot {
                    slot: 0,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: 1,
                    binding_index: 1,
                },
                Binding::Runtime {
                    kind: RuntimeBindingKind::SeqUsedK,
                    binding_index: 2,
                },
                Binding::Runtime {
                    kind: RuntimeBindingKind::BlockTable {
                        layer: crate::interpreter::metal::ids::LayerId(0),
                    },
                    binding_index: 3,
                },
                Binding::Runtime {
                    kind: RuntimeBindingKind::KvCacheK {
                        layer: crate::interpreter::metal::ids::LayerId(0),
                    },
                    binding_index: 4,
                },
                Binding::Runtime {
                    kind: RuntimeBindingKind::KvCacheV {
                        layer: crate::interpreter::metal::ids::LayerId(0),
                    },
                    binding_index: 5,
                },
            ]),
            gemm_dims: None,
        };
        let tape = LoweredMetalTape {
            bucket_m: 1,
            num_arena_slots: 2,
            commands: crate::interpreter::metal::lowered::baked(vec![attn.into()]),
            splitk_scratch_bytes: 0,
            moe_scratch_bytes: 0,
            roped_k_scratch_bytes: 0,
            attn_unfused_scratch_bytes: 0,
            barrier_before: &[],
            // Straight-line: this fixture is one hand-built body, no rolled layer loop.
            loops: &[],
        };

        let worker = MetalWorker::<TestWeights>::new(
            device,
            &vec![1024, 1024],
            &[tape],
            &pipelines,
            &weights,
            &allocator,
            &runtime,
        )
        .expect("worker bakes attention command");

        assert_eq!(worker.bucket_bakings.len(), 1);
        let baking = &worker.bucket_bakings[0];
        let steps = baking
            .mtl4_steps
            .as_ref()
            .expect("dispatch-only bucket bakes MTL4 steps");
        assert_eq!(steps.len(), 1);
        assert_eq!(mtl4_step_dispatch_count(baking, 0), 1);
    }

    /// Build a `KernelId::Gemm` lowered command at bucket=`bucket_m`
    /// projecting `[bucket_m, k]` × `[n, k]^T` → `[bucket_m, n]`.
    /// Bindings match the lowering pass: arena slots `(0, 1)` for
    /// (out, in) and a `LinearLayer` weight binding.
    fn build_gemm_command(bucket_m: u32, n: u32, k: u32) -> LoweredCommand {
        LoweredCommand {
            kernel: KernelId::Gemm,
            // GEMM is opaque to the unified pipeline picker — see the
            // matching note in `lowering.rs` for the production GEMM arm.
            library: "",
            function: "",
            constants: &[],
            dispatch: DispatchShape {
                threadgroups: (bucket_m.div_ceil(16), n.div_ceil(16), 1),
                threads_per_threadgroup: (16, 16, 1),
                m_scaling: None,
            },
            bindings: crate::interpreter::metal::lowered::baked(vec![
                Binding::ArenaSlot {
                    slot: 0,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: 1,
                    binding_index: 1,
                },
                Binding::Weight {
                    kind: WeightBundleKind::LinearLayer,
                    which: WeightTensor::Weight,
                    layer: crate::interpreter::metal::ids::LayerId(0),
                    locator: WeightLocator {
                        bucket: 0,
                        op_idx: 0,
                        slot: 0,
                    },
                    binding_index: 2,
                },
            ]),
            gemm_dims: Some(crate::interpreter::metal::lowered::GemmDims { m: bucket_m, n, k }),
        }
    }

    /// A tape carrying one `KernelId::Gemm` command bakes to an MTL4
    /// `Dispatch` step (the `gemm_{f16,bf16}_specialized` kernel), so the
    /// bucket is MTL4-eligible (`mtl4_steps` is `Some`). There is no
    /// longer an MPS / classic-command-buffer GEMM path.
    #[test]
    fn worker_routes_gemm_step() {
        let Some(device) = crate::detect_device().filter(|_| crate::metal4_available()) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let device = Arc::new(device.device.clone());

        let cache = Arc::new(
            SpecializedPipelineCache::with_standard_shaders((*device).clone())
                .expect("compile standard shaders"),
        );
        let pipelines = SpecializedPipelines::new(cache);
        let (weights, allocator) = build_test_weights();
        let runtime = empty_runtime(&device, 1);

        let tape = LoweredMetalTape {
            bucket_m: 1,
            num_arena_slots: 2,
            commands: crate::interpreter::metal::lowered::baked(vec![
                build_gemm_command(1, 2048, 2048).into(),
            ]),
            splitk_scratch_bytes: 0,
            moe_scratch_bytes: 0,
            roped_k_scratch_bytes: 0,
            attn_unfused_scratch_bytes: 0,
            barrier_before: &[],
            // Straight-line: this fixture is one hand-built body, no rolled layer loop.
            loops: &[],
        };

        let worker = MetalWorker::<TestWeights>::new(
            device,
            // Q-size buffers (2048 f16 = 4096 bytes; pad up).
            &vec![64 * 1024, 64 * 1024],
            &[tape],
            &pipelines,
            &weights,
            &allocator,
            &runtime,
        )
        .expect("worker bakes GEMM command");

        assert_eq!(worker.bucket_bakings.len(), 1);
        let baking = &worker.bucket_bakings[0];
        // GEMM bakes to a `gemm_*_specialized` Dispatch step → MTL4-eligible.
        assert!(
            baking.mtl4_steps.is_some(),
            "a GEMM bucket bakes to an MTL4 Dispatch step"
        );
    }

    /// Phase 5.C.5: an dispatch→Gemm→dispatch tape. The Gemm forces an encoder
    /// boundary, so the post-Gemm RmsNorm cannot coalesce with the
    /// pre-Gemm RmsNorm even though both share the same specialized
    /// pipeline.
    //
    // The three-segment structure (and the non-coalescing) is an internal
    // `bake_bucket` detail, not public on `BucketBaking`; the observable
    // consequence is that all three steps — including the GEMM — bake to
    // MTL4 `Dispatch` steps, so the bucket is MTL4-eligible.
    #[test]
    fn worker_interleaves_gemm_with_icb() {
        let Some(device) = crate::detect_device().filter(|_| crate::metal4_available()) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let device = Arc::new(device.device.clone());

        let cache = Arc::new(
            SpecializedPipelineCache::with_standard_shaders((*device).clone())
                .expect("compile standard shaders"),
        );
        let pipelines = SpecializedPipelines::new(cache);
        let (weights, allocator) = build_test_weights();
        let runtime = empty_runtime(&device, 1);

        let rmsnorm_pre = LoweredCommand {
            kernel: KernelId::RmsNorm,
            library: "rmsnorm",
            function: "rmsnorm_f16_s_f16_specialized",
            constants: crate::interpreter::metal::lowered::baked(vec![
                ConstantValue::uint(0, 1),
                ConstantValue::uint(1, <TestWeights as CanonicalParams>::Q_SIZE as u32),
                ConstantValue::float(2, <TestWeights as CanonicalParams>::RMS_NORM_EPS),
            ]),
            dispatch: DispatchShape {
                threadgroups: (1, 1, 1),
                threads_per_threadgroup: (256, 1, 1),
                m_scaling: None,
            },
            bindings: crate::interpreter::metal::lowered::baked(vec![
                Binding::ArenaSlot {
                    slot: 0,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: 1,
                    binding_index: 1,
                },
                Binding::Weight {
                    kind: WeightBundleKind::RmsNorm,
                    which: WeightTensor::Weight,
                    layer: crate::interpreter::metal::ids::LayerId(0),
                    locator: WeightLocator {
                        bucket: 0,
                        op_idx: 0,
                        slot: 0,
                    },
                    binding_index: 2,
                },
            ]),
            gemm_dims: None,
        };
        let rmsnorm_post = LoweredCommand {
            kernel: KernelId::RmsNorm,
            library: rmsnorm_pre.library,
            function: rmsnorm_pre.function,
            constants: rmsnorm_pre.constants.clone(),
            dispatch: rmsnorm_pre.dispatch,
            bindings: rmsnorm_pre
                .bindings
                .iter()
                .map(|b| match b {
                    Binding::ArenaSlot {
                        slot,
                        binding_index,
                    } => Binding::ArenaSlot {
                        slot: *slot,
                        binding_index: *binding_index,
                    },
                    Binding::Weight {
                        kind,
                        which,
                        layer,
                        locator,
                        binding_index,
                    } => Binding::Weight {
                        kind: *kind,
                        which: *which,
                        layer: *layer,
                        locator: *locator,
                        binding_index: *binding_index,
                    },
                    _ => unreachable!("rmsnorm_pre uses only ArenaSlot + RmsNorm Weight"),
                })
                .collect::<Vec<_>>()
                .into_baked(),
            gemm_dims: None,
        };

        let tape = LoweredMetalTape {
            bucket_m: 1,
            num_arena_slots: 2,
            commands: crate::interpreter::metal::lowered::baked(vec![
                rmsnorm_pre.into(),
                build_gemm_command(1, 2048, 2048).into(),
                rmsnorm_post.into(),
            ]),
            splitk_scratch_bytes: 0,
            moe_scratch_bytes: 0,
            roped_k_scratch_bytes: 0,
            attn_unfused_scratch_bytes: 0,
            barrier_before: &[],
            // Straight-line: this fixture is one hand-built body, no rolled layer loop.
            loops: &[],
        };

        let worker = MetalWorker::<TestWeights>::new(
            device,
            &vec![64 * 1024, 64 * 1024],
            &[tape],
            &pipelines,
            &weights,
            &allocator,
            &runtime,
        )
        .expect("worker bakes mixed tape");

        let baking = &worker.bucket_bakings[0];
        // The Dispatch|Gemm|Dispatch step plan is an internal `bake_bucket`
        // local; the observable consequence is that the GEMM bakes to a
        // `gemm_*_specialized` Dispatch step, so the whole bucket is
        // MTL4-eligible.
        assert!(
            baking.mtl4_steps.is_some(),
            "a bucket containing a GEMM bakes to MTL4 Dispatch steps"
        );
    }
}
