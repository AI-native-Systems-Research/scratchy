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
//! become a single `BucketStep::Dispatch`. MTL4 execution reads the
//! pre-baked `mtl4_steps` from each baking; a bucket is ineligible only
//! if a kernel exceeds the 31-entry argument-table bind cap.

#[cfg(test)]
use crate::interpreter::metal::lowered::IntoBaked;
use std::sync::Arc;

use crate::interpreter::metal::__re::{
    Buffer, ComputePipelineState, Device, MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize,
};
use ::objc2::rc::Retained;
use ::objc2::runtime::ProtocolObject;

use super::ids::LayerId;
use super::lowered::{
    Binding, GateCtx, LoweredCommand, LoweredMetalTape, MegakernelError, MegakernelTape, TapePlay,
    WeightTensor,
};
use super::pipelines::{PipelineLookupError, SpecializedPipelines};
use super::runtime::RuntimeBindings;
use crate::MetalAllocator;
use crate::tape::ids::SourceIx;
use crate::tape::lowered::ModelSources;
#[cfg(feature = "forward-telemetry")]
use scratchy_core_common::forward_telemetry::{
    ForwardRecord, ForwardTelemetry, KernelKind, TapeEntry,
};
use scratchy_ir::CanonicalParams;
use std::collections::HashMap;
use std::collections::hash_map::Entry;

/// Byte size of arena slot `i`. The macro's `colored_slot_map()`
/// computes this from the FUF's per-slot shape × dtype × max bucket;
/// for the Phase 5.C smoke test the test sets it explicitly.
pub type ArenaLayout = Vec<u64>;

/// One unit of execution in a bucket's plan.
///
/// `Dispatch` is a contiguous run of commands sharing a single pipeline.
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
    /// The bucket's decode megakernel in this worker's KV mode, when its tape carries one.
    pub megakernel: Option<super::megakernel::MegakernelBaking>,
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
    /// A per-bucket scratch or inline buffer a binding needs was not provisioned.
    WeightLookupFailed { reason: &'static str },
    /// A command's MLX-affine codes operand could not be bound stored the
    /// way its kernel reads them.
    AffineCodes(crate::metal_allocator::AffineCodesBindError),
    /// A model source a tape binds did not resolve at load.
    SourceUnresolved {
        ix: SourceIx,
        source: &'static str,
        which: WeightTensor,
        layer: LayerId,
        why: SourceMiss,
    },
    /// A command referenced `Binding::Scratch` but the worker has no
    /// SplitK scratch buffer allocated. Indicates a lowering /
    /// `LoweredMetalTape::splitk_scratch_bytes` accounting bug —
    /// either lowering emitted Scratch without registering the
    /// scratch byte count, or the worker discarded the buffer.
    ScratchBufferMissing {
        bucket_index: usize,
        command_index: usize,
    },
    /// The bucket's megakernel could not load.
    Megakernel(MegakernelError),
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
            Self::AffineCodes(e) => write!(f, "MetalWorker: {e}"),
            Self::WeightLookupFailed { reason } => {
                write!(f, "MetalWorker: weight lookup: {reason}")
            }
            Self::SourceUnresolved {
                ix,
                source,
                which,
                layer,
                why,
            } => write!(
                f,
                "MetalWorker: model source #{} `{source}` {which:?} at layer {}: {why:?}",
                ix.get(),
                layer.get()
            ),
            Self::ScratchBufferMissing {
                bucket_index,
                command_index,
            } => write!(
                f,
                "MetalWorker: bucket {bucket_index} command {command_index}: \
                 Binding::Scratch with no splitk scratch buffer allocated \
                 (lowering / tape accounting bug)"
            ),
            Self::Megakernel(e) => write!(f, "MetalWorker: {e}"),
        }
    }
}

impl std::error::Error for WorkerError {}

/// Why a [`WorkerError::SourceUnresolved`] source did not resolve.
#[derive(Debug)]
pub enum SourceMiss {
    /// The model's generated resolver has no such family.
    NoFamily,
    /// The family's bundle holds no such tensor (a `which` of another kind, an absent optional
    /// bias or shared expert, a Dense MoE on metal).
    NoTensor,
    /// The tensor lives in no arena of the pool's allocator.
    NotResident,
    /// A bake met a binding its pool did not resolve.
    NotResolved,
}

/// One per concurrent forward. Owns its arena + per-bucket bakings;
/// borrows the model meta + runtime bindings + pipeline cache via
/// references threaded through `new`.
pub struct MetalWorker<W: CanonicalParams> {
    pub arena: Vec<Buffer>,
    /// Residency pins for every buffer the baked dispatches reach by address.
    _pins: Vec<crate::residency::Pinned>,
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
    /// Whether some sequence of the step about to run has an unrotated
    /// (bit-31, span) block in its block table, set with
    /// `tq_dequant_max_blocks`. Only a prefill attention that re-ropes K can
    /// read such a block (`RuntimeGate::OnlyIfUnrotatedBlocks`).
    pub unrotated_blocks: std::sync::atomic::AtomicBool,
    /// How the next forward plays its tape ([`TapePlay`]); stashed from the forward's inputs
    /// like `tq_dequant_max_blocks`. `true` = [`TapePlay::Megakernel`].
    plays_megakernel: std::sync::atomic::AtomicBool,
    /// Megakernel runs the last forward played (the trace reads it).
    megakernel_runs: std::sync::atomic::AtomicU32,
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
        sources: &ResolvedSources,
        runtime: &RuntimeBindings,
    ) -> Result<Self, WorkerError> {
        Self::new_with_residency(
            device,
            arena_layout,
            bucket_tapes,
            &[],
            pipelines,
            sources,
            runtime,
            None,
        )
    }

    /// Same as [`Self::new`] but accepts an optional `MetalResidencySet`
    /// that worker-local arena slots get inserted into. Used by the
    /// pool to pin every per-worker arena into the wired set so cmdbuf
    /// dispatches don't race against Apple's lazy paging on
    /// Llama-3.2-class working sets.
    ///
    /// `megakernels[i]` is bucket `i`'s baked megakernel tapes (one per KV mode, usually none);
    /// the worker plays the one of its own KV mode.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_residency(
        device: Arc<Device>,
        arena_layout: &ArenaLayout,
        bucket_tapes: &[LoweredMetalTape],
        megakernels: &[&'static [MegakernelTape]],
        pipelines: &SpecializedPipelines,
        sources: &ResolvedSources,
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

        // Every buffer the baked dispatches reach by address, pinned for as
        // long as this worker lives.
        let mut pins: Vec<crate::residency::Pinned> = Vec::new();
        let mut pin = |b: &Buffer| pins.extend(residency.map(|r| r.pin(b.clone())));

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
                pin(&buf);
                buf
            })
            .collect();

        // A Private scratch buffer (the host never reads scratch back) sized to
        // the max across bucket tapes and pinned, since dispatches bind it by
        // baked address; `None` when no tape needs it.
        let mut scratch = |bytes: fn(&LoweredMetalTape) -> u32, what: &str| {
            let max = bucket_tapes.iter().map(bytes).max().unwrap_or(0);
            (max > 0).then(|| {
                let buf = device
                    .newBufferWithLength_options(
                        max as usize,
                        MTLResourceOptions::StorageModePrivate,
                    )
                    .unwrap_or_else(|| panic!("newBufferWithLength_options returned nil ({what})"));
                pin(&buf);
                buf
            })
        };
        // SplitK: one buffer suffices because successive `affine_qmm_t_splitk`
        // / `splitk_reduce_sum` pairs run serially inside a single encoder.
        let splitk_scratch = scratch(|t| t.splitk_scratch_bytes, "splitk scratch");
        // `Binding::MoeScratch`: MoE blocks within one bucket execute serially
        // through the dispatch; cross-bucket reuse is fine because only one
        // bucket runs per forward.
        let moe_scratch = scratch(|t| t.moe_scratch_bytes, "moe scratch");
        // `Binding::RopedKScratch` (spans rope-on-read, rope-once-to-scratch —
        // NAX matrix-accel AND the simdgroup steel prefill): the
        // RopeOnce{Nax,Steel} command and its following attention run serially
        // per layer through the dispatch, and the scratch is overwritten each
        // layer (the cache, not the scratch, is the persistent artifact).
        let roped_k_scratch = scratch(|t| t.roped_k_scratch_bytes, "roped-K scratch");
        // hd512-unfused attention (q_head/Kdense/Vdense_T/scores/out_head
        // packed at baked offsets). One buffer, overwritten per layer.
        let attn_unfused_scratch =
            scratch(|t| t.attn_unfused_scratch_bytes, "attn-unfused scratch");

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
            // (`pin(field)`) or bound to `field: _` with a reason. A new
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
            pin(input_ids);
            pin(positions);
            pin(cu_seqlens_q);
            pin(seq_used_k);
            // Span labels: bound by dispatch-baked address on rope-on-read
            // arches (slot 8 of the prefill attention), so pin like seq_used_k.
            pin(span_ids);
            // Per-KV-cache-group slot_mappings + block tables (vLLM hybrid
            // layout). One group on uniform models; full + N sliding on gemma4
            // SWA. Each is bound by dispatch-baked gpuAddress (same lazy-pager
            // hazard as the inputs), so EVERY group buffer must be pinned.
            for b in slot_mappings {
                pin(b);
            }
            for b in block_tables {
                pin(b);
            }
            // Spans rope-on-read per-layer flag mirrors (placeholders on
            // non-spans arches; bound by baked gpuAddress when active).
            for b in block_unrotated_flags {
                pin(b);
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
            pin(gdn_state_indices);
            pin(gdn_is_fresh);
            // Vision per-forward externs (vision towers: Qwen3.5-VL ViT).
            // Same baked-gpuAddress / lazy-pager hazard as the GDN
            // buffers above — the `vision_rope_2d` / `vision_varlen_attn`
            // / `LoadPixels` kernels read `freqs` / `cu_seqlens` / `pixels`
            // by dispatch-baked address, so an un-pinned buffer is served stale
            // zero pages (garbage rope angles / all-token-0 pixels). The
            // factory allocates 16-byte placeholders on non-vision arches,
            // so the pin is harmless when no vision command binds them.
            pin(vision_rope_freqs);
            pin(pixels);
            pin(vision_pos_embeds);
            pin(mm_embeds);
            pin(mm_dst_rows);
            pin(mrope_cos_sin);
            pin(vision_cu_seqlens_full);
            pin(vision_cu_seqlens_window);
            pin(vision_window_index);
            pin(vision_reverse_indices);
            pin(vision_position_ids);
            // TurboQuant: the packed code stores + norms + codebook are read by
            // the dequant/quantize dispatch commands via baked gpuAddress, so pin them
            // (same lazy-pager hazard). `None` for a dense model → no-op.
            if let Some(t) = tq {
                for b in &t.packed_k {
                    pin(b);
                }
                for b in &t.packed_v {
                    pin(b);
                }
                for b in &t.norms_k {
                    pin(b);
                }
                for b in &t.norms_v {
                    pin(b);
                }
                pin(&t.signs);
                pin(&t.boundaries);
                pin(&t.centroids);
                // The fp16 scratch (table + backing data) is bound at every
                // layer's kv_cache slot + dereffed via the table's gpuAddress.
                pin(&t.scratch_k_table);
                pin(&t.scratch_v_table);
                pin(&t.scratch_k_data);
                pin(&t.scratch_v_data);
            }
            r.commit();
        }

        let mut bucket_bakings = Vec::with_capacity(bucket_tapes.len());
        for (bucket_idx, tape) in bucket_tapes.iter().enumerate() {
            let megakernel = megakernels.get(bucket_idx).and_then(|m| m.first());
            let baking = bake_bucket::<W>(
                bucket_idx,
                tape,
                megakernel,
                &arena,
                splitk_scratch.as_ref(),
                moe_scratch.as_ref(),
                roped_k_scratch.as_ref(),
                attn_unfused_scratch.as_ref(),
                pipelines,
                sources,
                runtime,
                device.clone(),
            )?;
            // Insert the per-bucket inline-constants buffer into the
            // residency set so the dispatch exec dispatch sees its
            // residency entry (the dispatch never `setBuffer`s these
            // explicitly — bindings are pre-recorded in the dispatch).
            if let Some(b) = baking.moe_inline_buf.as_ref() {
                pin(b);
            }
            // The megakernel's addresses and sync words are bound by address too.
            if let Some(m) = baking.megakernel.as_ref() {
                m.buffers().into_iter().for_each(&mut pin);
            }
            bucket_bakings.push(baking);
        }
        // Every codes operand is now stored the way its readers read it.
        crate::metal_allocator::signal_affine_codes_settled();
        if let Some(r) = residency {
            r.commit();
        }

        Ok(Self {
            arena,
            _pins: pins,
            bucket_bakings,
            splitk_scratch,
            moe_scratch,
            roped_k_scratch,
            attn_unfused_scratch,
            tq_dequant_max_blocks: std::sync::atomic::AtomicU32::new(0),
            unrotated_blocks: std::sync::atomic::AtomicBool::new(false),
            plays_megakernel: std::sync::atomic::AtomicBool::new(false),
            megakernel_runs: std::sync::atomic::AtomicU32::new(0),
            _marker: std::marker::PhantomData,
        })
    }

    /// How the next forward plays its tape — stashed from the forward's inputs.
    pub fn set_tape_play(&self, play: TapePlay) {
        let mk = play == TapePlay::Megakernel;
        self.plays_megakernel
            .store(mk, std::sync::atomic::Ordering::Relaxed);
    }

    /// Megakernel runs the last encoded forward played.
    pub fn megakernel_runs_played(&self) -> u32 {
        self.megakernel_runs
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// After a forward's command buffer completed: the first bounded wait any megakernel run
    /// gave up on, as a typed error (every stall word is cleared).
    pub fn take_megakernel_stall(&self) -> Result<(), MegakernelError> {
        let stalls = self
            .bucket_bakings
            .iter()
            .filter_map(|b| b.megakernel.as_ref());
        let words: Vec<_> = stalls.map(|m| m.take_stall()).collect();
        words.into_iter().collect()
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
        // at macro time from the tape's dataflow (the metal compiler's
        // hazard walk). Runtime does zero analysis — just emits a
        // `Dispatch→Dispatch` barrier wherever the flag fires.
        // Flat dispatch index across all steps. Counts EVERY dispatch
        // slot (including gate-skipped ones) so it stays aligned with
        // the expanded command order the megakernel's span names.
        let mut flat_idx: usize = 0;
        // Opt-in kernel tape (forward-local; published once at encode end).
        #[cfg(feature = "forward-telemetry")]
        let tape_enabled = ForwardTelemetry::global().is_enabled();
        #[cfg(feature = "forward-telemetry")]
        let mut tape: Vec<TapeEntry> = if tape_enabled {
            Vec::with_capacity(mtl4_steps.iter().map(|s| s.dispatches.len()).sum())
        } else {
            Vec::new()
        };
        let ctx = GateCtx {
            num_tokens,
            num_seqs,
            has_spec_tokens,
            unrotated_blocks: self
                .unrotated_blocks
                .load(std::sync::atomic::Ordering::Relaxed),
        };
        // THE MEGAKERNEL plays only in the context it was planned under (bucket-1 decode of one
        // sequence, this worker's KV mode — its gates are constants there). Its ONE launch
        // replaces the forward's span of flat dispatch indices; a dispatch after it re-sets its
        // pipeline and fences.
        let plays = self
            .plays_megakernel
            .load(std::sync::atomic::Ordering::Relaxed);
        let megakernel = baking
            .megakernel
            .as_ref()
            .filter(|_| plays && ctx == GateCtx::decode_one(ctx.unrotated_blocks));
        let (mut runs_played, mut refence, mut repipeline) = (0u32, false, false);
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
                if let Some(mk) = megakernel {
                    let span = mk.commands();
                    if span.contains(&this_idx) {
                        if this_idx == span.start {
                            mk.encode(enc);
                            (runs_played, refence, repipeline) = (runs_played + 1, true, true);
                        }
                        continue;
                    }
                }
                // The compiler's own hazard-analysis verdict for the tape,
                // captured before the megakernel's re-fence overrides it.
                #[cfg(feature = "forward-telemetry")]
                let compiler_barrier = *need_barrier;
                let mut need_barrier = *need_barrier;
                if !gate.is_none_or(|g| g.admits(ctx)) {
                    // Skipped: e.g. the lm_head slice's gather/qmv/scatter
                    // (`OnlyIfNoSpec`) on a spec-decode verify step, or
                    // its M=bucket_m fallback (`OnlyIfSpec`) on any other.
                    // Either way the timestamp slot, barrier, and
                    // dispatch are skipped together so the kernel
                    // doesn't run with stale per-dispatch state.
                    continue;
                }
                if std::mem::take(&mut repipeline) {
                    enc.setComputePipelineState(&step.pipeline);
                }
                if std::mem::take(&mut refence) {
                    need_barrier = true;
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
        self.megakernel_runs
            .store(runs_played, std::sync::atomic::Ordering::Relaxed);
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
fn is_fused(id: super::lowered::KernelId) -> bool {
    use super::lowered::KernelId as K;
    matches!(
        id,
        K::FusedAddRmsNorm
            | K::FusedGateUpSiluMul
            | K::RopeAppendNormed
            | K::NormAddScalarMul
            | K::AttentionViaCacheTq
    )
}

/// Map a lowered kernel identifier onto the arch-agnostic family the UI colors
/// the live dispatch tape by. Exhaustive on purpose: a new `KernelId` variant
/// forces a classification decision here rather than silently defaulting.
#[cfg(feature = "forward-telemetry")]
fn kernel_kind(id: super::lowered::KernelId) -> KernelKind {
    use super::lowered::KernelId as K;
    match id {
        K::Embed | K::AffineEmbed | K::EmbeddingGather | K::MmEmbedSplice => KernelKind::Embed,
        K::RmsNorm | K::RmsNormUnit | K::FusedAddRmsNorm | K::NormAddScalarMul => KernelKind::Norm,
        K::RopeAppendNormed
        | K::RopeAppend
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
        | K::AttnGemmPv => KernelKind::Attention,
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
        | K::AffineQmmSmallM
        | K::AffineW4a8Quant
        | K::AffineQmmW4a8
        | K::AffineGatherW4a8Quant
        | K::AffineGatherQmmW4a8
        | K::AffineGatherQmvFast
        | K::AffineGatherQmv
        | K::SplitKReduceSum => KernelKind::Gemm,
        K::MoeWeightedSum
        | K::MoeGroupOffsets
        | K::MoeGroupInit
        | K::MoeGroupScatter
        | K::MoeGroupScatterQ8
        | K::MoeGroupGather
        | K::MoePerExpertScale
        | K::GateApply
        | K::GateScale
        | K::GateSplit
        | K::ArgPartitionTopK
        | K::TakeAlongAxis => KernelKind::Moe,
        K::FusedGateUpSiluMul | K::SiluMul | K::GeluMul => KernelKind::Mlp,
        K::GatherLastToken | K::ScatterFirstToLastRow | K::Softmax | K::SliceTrailingColsU32 => {
            KernelKind::Sample
        }
        K::VisionLayerNorm
        | K::VisionRope
        | K::VisionVarlenAttn
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
    megakernel: Option<&MegakernelTape>,
    arena: &[Buffer],
    splitk_scratch: Option<&Buffer>,
    moe_scratch: Option<&Buffer>,
    roped_k_scratch: Option<&Buffer>,
    attn_unfused_scratch: Option<&Buffer>,
    pipelines: &SpecializedPipelines,
    sources: &ResolvedSources,
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
    // Every command's resolved bindings, when the megakernel's step records need their addresses.
    let mut bound_by_command: Vec<Vec<super::megakernel::BoundBinding>> = Vec::new();

    for (cmd_idx, gated) in expanded_commands.iter().enumerate() {
        let cmd = &gated.command;
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
            sources,
            runtime,
        )?;
        let bound_refs: Vec<(&Buffer, u64, u64)> =
            bound.iter().map(|(b, off, idx)| (b, *off, *idx)).collect();
        let (tg, tpt) = mtl_size_pair(cmd);

        // Coalesce with the previous step iff its pipeline shares the
        // underlying ObjC pointer (specialized pipelines are refcounted —
        // same key returns same handle from the cache).
        let bindings_for_cmd: Vec<(Buffer, u64, u64)> = bound_refs
            .iter()
            .map(|(b, off, idx)| ((*b).clone(), *off, *idx))
            .collect();
        if megakernel.is_some() {
            bound_by_command.push(bindings_for_cmd.clone());
        }
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
    let megakernel = match megakernel {
        Some(mk) => {
            use super::megakernel::MegakernelBaking;
            let baked_of: Vec<usize> = tape.expanded_origins().iter().map(|o| o.baked).collect();
            let (baking, load) = MegakernelBaking::bake(
                mk,
                &expanded_commands,
                &baked_of,
                &bound_by_command,
                pipelines,
                &device,
            )?;
            if std::env::var_os("SCRATCHY_METAL_TRACE").is_some() {
                eprintln!(
                    "[megakernel] bucket_m={} ONE launch plays {} of {} commands; \
                     grid barriers per forward={}; library {} ({} bytes metallib) loaded in {:?}; \
                     pipeline built in {:?}; maxTotalThreadsPerThreadgroup={}; persistent \
                     threadgroups P={}",
                    tape.bucket_m,
                    load.steps,
                    expanded_commands.len(),
                    load.grid_barriers,
                    mk.library,
                    load.metallib_bytes,
                    load.loaded,
                    load.pipeline_built,
                    load.max_threads,
                    load.threadgroups,
                );
            }
            Some(baking)
        }
        None => None,
    };
    Ok(BucketBaking {
        bucket_m: tape.bucket_m,
        mtl4_steps,
        moe_inline_buf,
        megakernel,
    })
}

/// Every model tensor a pool's tapes bind, resolved ONCE at load: `(source, tensor, layer)` →
/// the buffer and offset holding it. Workers bake from this table; nothing resolves per
/// dispatch record, and a worker never touches the weights.
pub struct ResolvedSources(HashMap<(SourceIx, WeightTensor, LayerId), (Buffer, u64)>);

impl ResolvedSources {
    /// Resolve every [`Binding::Source`] of `tapes` (loops played out) through the model's
    /// generated [`ModelSources`] impl, each distinct `(ix, which, layer)` once.
    pub fn resolve<W: ModelSources>(
        weights: &W,
        allocator: &MetalAllocator,
        tapes: &[LoweredMetalTape],
    ) -> Result<Self, WorkerError> {
        let mut table = HashMap::new();
        for c in tapes.iter().flat_map(LoweredMetalTape::commands_expanded) {
            for b in c.command.bindings {
                let &Binding::Source {
                    ix, which, layer, ..
                } = b
                else {
                    continue;
                };
                let miss = |why| WorkerError::SourceUnresolved {
                    ix,
                    source: W::SOURCES.get(ix.get() as usize).copied().unwrap_or("?"),
                    which,
                    layer,
                    why,
                };
                let bundle = weights
                    .source(ix, layer)
                    .ok_or(miss(SourceMiss::NoFamily))?;
                // MLX-affine packed codes bind stored the way each reading command's kernel
                // reads them — every reader, so two that disagree refuse the load.
                if bundle.is_affine_codes(which) {
                    let tensor = bundle.tensor(which).ok_or(miss(SourceMiss::NoTensor))?;
                    let codes = crate::tape::kernel_constants::AffineCodes::of_constants(
                        c.command.constants,
                    );
                    let at = allocator
                        .bind_affine_codes(tensor.raw_ptr(), tensor.size_bytes(), codes)
                        .map_err(WorkerError::AffineCodes)?;
                    table.entry((ix, which, layer)).or_insert(at);
                } else if let Entry::Vacant(v) = table.entry((ix, which, layer)) {
                    let tensor = bundle.tensor(which).ok_or(miss(SourceMiss::NoTensor))?;
                    let at = allocator.buffer_for(tensor.raw_ptr());
                    v.insert(at.ok_or(miss(SourceMiss::NotResident))?);
                }
            }
        }
        Ok(Self(table))
    }

    fn get(
        &self,
        ix: SourceIx,
        which: WeightTensor,
        layer: LayerId,
    ) -> Result<(Buffer, u64), WorkerError> {
        let hit = self.0.get(&(ix, which, layer)).cloned();
        hit.ok_or(WorkerError::SourceUnresolved {
            ix,
            source: "?",
            which,
            layer,
            why: SourceMiss::NotResolved,
        })
    }
}

/// Resolve every binding on `cmd` to (buffer, offset, binding-index).
///
/// Returns owned `Buffer` clones (cheap ObjC refcount) so callers
/// don't have to thread the [`MetalAllocator`]'s arenas-`Mutex` lock
/// guard through to the encoder.
#[allow(clippy::too_many_arguments)]
fn resolve_bindings(
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
    sources: &ResolvedSources,
    runtime: &RuntimeBindings,
) -> Result<Vec<(Buffer, u64, u64)>, WorkerError> {
    let mut out: Vec<(Buffer, u64, u64)> = Vec::with_capacity(cmd.bindings.len());
    for binding in cmd.bindings {
        let (buf, off, idx) = match binding {
            Binding::Source {
                ix,
                which,
                layer,
                binding_index,
            } => {
                let (b, off) = sources.get(*ix, *which, *layer)?;
                (b, off, *binding_index as u64)
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
fn scale_tg_for_num_tokens(
    tg: MTLSize,
    scaling: Option<super::lowered::MScaling>,
    num_tokens: super::ids::NumTokens,
    num_seqs: u32,
) -> MTLSize {
    let Some(s) = scaling else {
        return tg;
    };
    let grid = [tg.width, tg.height, tg.depth].map(|v| v as u64);
    let [width, height, depth] = s.scale(grid, num_tokens, num_seqs).map(|v| v as usize);
    MTLSize {
        width,
        height,
        depth,
    }
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
        Binding, DispatchShape, KernelId, LoweredCommand, MetalDtype, RuntimeBindingKind,
        SourceRef, WeightTensor,
    };
    use crate::specialized_pipeline_cache::{ConstantValue, SpecializedPipelineCache};
    use scratchy_ir::CanonicalParams;
    use scratchy_layers::{Linear, LinearLayer, RmsNorm};
    use scratchy_tensors::{DType, DeviceAllocator, GpuTensor};
    use std::sync::Arc;

    /// The decode-step gates follow whether every sequence contributes one
    /// token; the sequence gates follow the step's sequence count, not its
    /// token count; the span-block gates follow the step's block tables; and
    /// `All` needs every gate: a decode step runs none of a TurboQuant prefill
    /// attention's variants.
    #[test]
    fn step_gates_follow_the_step() {
        use super::super::lowered::RuntimeGate::{
            All, OnlyIfDecodeStep, OnlyIfOneSequence, OnlyIfUnrotatedBlocks, UnlessDecodeStep,
            UnlessOneSequence, UnlessUnrotatedBlocks,
        };
        let step = |num_tokens, num_seqs, unrotated_blocks| GateCtx {
            num_tokens,
            num_seqs,
            has_spec_tokens: false,
            unrotated_blocks,
        };
        for (tokens, seqs, decode) in [
            (1, 1, true),
            (16, 16, true),
            (512, 1, false),
            (18, 16, false),
        ] {
            let s = step(tokens, seqs, false);
            assert_eq!(OnlyIfDecodeStep.admits(s), decode);
            assert_eq!(UnlessDecodeStep.admits(s), !decode);
        }
        for (tokens, seqs, one) in [(512, 1, true), (512, 2, false), (16, 16, false)] {
            let s = step(tokens, seqs, false);
            assert_eq!(OnlyIfOneSequence.admits(s), one);
            assert_eq!(UnlessOneSequence.admits(s), !one);
        }
        for unrotated in [false, true] {
            let s = step(512, 2, unrotated);
            assert_eq!(OnlyIfUnrotatedBlocks.admits(s), unrotated);
            assert_eq!(UnlessUnrotatedBlocks.admits(s), !unrotated);
        }
        let plain = All(&[UnlessDecodeStep, UnlessOneSequence, UnlessUnrotatedBlocks]);
        assert!(plain.admits(step(512, 2, false)));
        assert!(!plain.admits(step(512, 2, true)));
        assert!(!plain.admits(step(512, 1, false)));
        assert!(!plain.admits(step(2, 2, false)), "a decode step");
    }

    /// The lm_head sample slice, as the tape builds and dispatches it: the
    /// gather moves each sequence's last row to row `i`, the scatter moves
    /// row `i` back, both in place. In a mixed step, decodes and short chunks
    /// put some sequences' last rows inside `0..num_seqs`, the rows the
    /// others are moved to; every sequence must still get its own row.
    #[test]
    fn sample_slice_moves_each_sequences_own_row() {
        let Some(device) = crate::detect_device().filter(|_| crate::metal4_available()) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let device = device.device.clone();
        let cache = SpecializedPipelineCache::with_standard_shaders(device.clone())
            .expect("compile standard shaders");
        let p = crate::tape::model_consts::MetalModelConsts {
            metal_dtype: MetalDtype::F16,
            ..crate::tape::model_consts::MetalModelConsts::from_canonical::<TestWeights>()
        };
        // Decodes, then short prefix-hit chunks, then long chunks.
        let q_lens: Vec<u32> = [vec![1; 24], vec![2; 16], vec![3; 8], vec![40; 4]].concat();
        let cu: Vec<u32> = std::iter::once(0)
            .chain(q_lens.iter().scan(0, |end, n| {
                *end += n;
                Some(*end)
            }))
            .collect();
        let (num_seqs, num_tokens) = (q_lens.len(), *cu.last().unwrap() as usize);
        let last = |i: usize| cu[i + 1] as usize - 1;
        let width = 1024u32;
        let bucket_m = 512;
        for gather in [true, false] {
            let cmd = if gather {
                crate::tape::lowering::gather_last_token_command(
                    &p,
                    crate::tape::ids::ArenaSlotIdx(0),
                    width,
                )
            } else {
                crate::tape::lowering::scatter_first_to_last_row_command(
                    &p,
                    crate::tape::ids::ArenaSlotIdx(0),
                    width,
                )
            };
            let pso = cache
                .get_or_build(&crate::specialized_pipeline_cache::PipelineKey::new(
                    cmd.library,
                    cmd.function,
                    cmd.constants.to_vec(),
                ))
                .expect("pipeline");
            let (grid, threads) = mtl_size_pair(&cmd);
            let grid = scale_tg_for_num_tokens(
                grid,
                cmd.dispatch.m_scaling,
                super::super::ids::NumTokens(num_tokens as u32),
                num_seqs as u32,
            );
            // Every element of row `r` holds `r`.
            let rows: Vec<u16> = (0..bucket_m as usize)
                .flat_map(|r| {
                    std::iter::repeat_n(half::f16::from_f32(r as f32).to_bits(), width as usize)
                })
                .collect();
            let buf = crate::mtl4_dispatch::shared_slice(&device, &rows);
            let cu_buf = crate::mtl4_dispatch::shared_slice(&device, &cu);
            let n_buf = crate::mtl4_dispatch::shared_u32(&device, num_seqs as u32);
            assert!(crate::mtl4_dispatch::dispatch_threadgroups(
                &device,
                &pso,
                &[&buf, &cu_buf, &n_buf],
                grid,
                threads,
            ));
            let got: Vec<u16> = crate::mtl4_dispatch::read_slice(&buf, rows.len());
            let row = |r: usize| &got[r * width as usize..][..width as usize];
            for (i, q_len) in q_lens.iter().enumerate() {
                // Gather: row i holds sequence i's last row. Scatter: that
                // last row holds row i, sequence i's logits.
                let (at, want) = if gather { (i, last(i)) } else { (last(i), i) };
                let want = half::f16::from_f32(want as f32).to_bits();
                let wrong = row(at).iter().filter(|&&x| x != want).count();
                assert_eq!(
                    wrong,
                    0,
                    "{}: sequence {i} ({} tokens): {wrong} of {width} elements of row {at} are another row's",
                    if gather { "gather" } else { "scatter" },
                    q_len
                );
            }
        }
    }

    /// Test fixture: holds `CanonicalParams` constants AND the layer
    /// instances its `ModelSources` impl below returns. Plays the
    /// role of the per-canonical `Weights` struct the macro emits.
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

    // `WeightAccessors` is a supertrait of `CanonicalParams` (every method defaults). The
    // synthetic tapes below bind two model sources: the RmsNorm (0) and the LinearLayer (1).
    impl scratchy_ir::WeightAccessors for TestWeights {}
    const RMSNORM: SourceIx = SourceIx(0);
    const LINEAR: SourceIx = SourceIx(1);
    impl ModelSources for TestWeights {
        const SOURCES: &'static [&'static str] = &["rmsnorm", "linear"];
        fn source(&self, ix: SourceIx, _layer: LayerId) -> Option<SourceRef<'_>> {
            match ix {
                RMSNORM => Some(SourceRef::RmsNorm(&self.rmsnorm_layer)),
                LINEAR => Some(SourceRef::Linear(&self.linear_layer)),
                _ => None,
            }
        }
    }

    /// `tapes`' model sources, resolved as the pool resolves them.
    fn sources(w: &TestWeights, a: &MetalAllocator, tapes: &[LoweredMetalTape]) -> ResolvedSources {
        ResolvedSources::resolve(w, a, tapes).expect("sources resolve")
    }

    /// Build a `TestWeights` + the `MetalAllocator` that owns its
    /// MTLBuffer arenas. The allocator maps each resolved source's
    /// tensor back to its `(MTLBuffer, offset)`.
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
                Binding::Source {
                    ix: RMSNORM,
                    which: WeightTensor::Weight,
                    layer: LayerId(0),
                    binding_index: 2,
                },
            ]),
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
                Binding::Source {
                    ix: RMSNORM,
                    which: WeightTensor::Weight,
                    layer: LayerId(0),
                    binding_index: 2,
                },
            ]),
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
            &sources(&weights, &allocator, &tapes),
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
            &sources(&weights, &allocator, &tapes),
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
            &sources(&weights, &allocator, &[tape]),
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
        crate::tape::lowering::gemm_command(
            <TestWeights as CanonicalParams>::METAL_DTYPE,
            crate::interpreter::metal::lowered::GemmDims { m: bucket_m, n, k },
            vec![
                Binding::ArenaSlot {
                    slot: 0,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: 1,
                    binding_index: 1,
                },
                Binding::Source {
                    ix: LINEAR,
                    which: WeightTensor::Weight,
                    layer: LayerId(0),
                    binding_index: 2,
                },
            ],
        )
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
            &sources(&weights, &allocator, &[tape]),
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

    /// Phase 5.C.5: an dispatch→Gemm→dispatch tape. The Gemm's pipeline differs, so the
    /// post-Gemm RmsNorm cannot coalesce with the pre-Gemm RmsNorm even though both share the
    /// same specialized pipeline.
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
                Binding::Source {
                    ix: RMSNORM,
                    which: WeightTensor::Weight,
                    layer: LayerId(0),
                    binding_index: 2,
                },
            ]),
        };
        let rmsnorm_post = LoweredCommand {
            kernel: KernelId::RmsNorm,
            library: rmsnorm_pre.library,
            function: rmsnorm_pre.function,
            constants: rmsnorm_pre.constants,
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
                    Binding::Source {
                        ix,
                        which,
                        layer,
                        binding_index,
                    } => Binding::Source {
                        ix: *ix,
                        which: *which,
                        layer: *layer,
                        binding_index: *binding_index,
                    },
                    _ => unreachable!("rmsnorm_pre uses only ArenaSlot + the RmsNorm source"),
                })
                .collect::<Vec<_>>()
                .into_baked(),
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
            &sources(&weights, &allocator, &[tape]),
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
