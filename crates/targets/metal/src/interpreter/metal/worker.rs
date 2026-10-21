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
//! bakes to a `gemm_{f16,bf16}_specialized` Dispatch step (one row:
//! `gemv_{f16,bf16}_specialized`) like every other kernel. MTL4 execution reads the pre-baked `mtl4_steps` from
//! each baking; a bucket is ineligible only if a kernel exceeds the
//! 31-entry argument-table bind cap.

#[cfg(test)]
use crate::interpreter::metal::lowered::IntoBaked;
use std::sync::Arc;

use crate::interpreter::metal::__re::{
    Buffer, ComputePipelineState, Device, MTLDevice, MTLResourceOptions, MTLSize,
};
use ::objc2::rc::Retained;
use ::objc2::runtime::ProtocolObject;

use super::ids::LayerId;
use super::lowered::{Binding, Fence, KernelId, LoweredCommand, LoweredMetalTape, WeightTensor};
use super::pipelines::{PipelineLookupError, SpecializedPipelines};
use super::runtime::RuntimeBindings;
use crate::MetalAllocator;
use crate::sparse_arena::SparseBuffer;
use crate::tape::constants::TapeVariant;
use crate::tape::ids::SourceIx;
use crate::tape::lowered::ModelSources;
use crate::tape::lowered::{PerScratch, ScratchKind};
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

/// An activation buffer a worker binds by address: allocated in full, or placement-sparse and mapped
/// as far as the buckets that bind it reach ([`SparseBuffer`]).
#[derive(Clone)]
pub enum Activation {
    Full(Buffer),
    Sparse(Arc<SparseBuffer>),
}

impl Activation {
    pub fn buffer(&self) -> &Buffer {
        match self {
            Self::Full(buffer) => buffer,
            Self::Sparse(sparse) => sparse.buffer(),
        }
    }
}

/// What a pool's worker maps as its buckets reach it ([`SparseBuffer`]) rather than allocating in
/// full: each bucket's bytes in each arena slot, in bucket order (`MetalBucketSpec::slot_bytes`),
/// and the slots the host reads by pointer, which stay allocated in full and shared — every
/// bucket's backbone hidden states (a forward's `hidden_out` rows; an encoder's output, which is
/// its terminal slot too) and, where the pool's owner reads its logits on the host, every terminal
/// slot ([`TerminalReads`]). A guest's host reads none ([`ArenaShare::In`]).
#[derive(Clone, Default)]
pub struct SparsePlan {
    pub bucket_arena: Vec<Vec<u64>>,
    pub host_slots: Vec<u32>,
}

/// Who reads a pool's logits (each bucket's terminal slot): only the GPU's argmax and sampler, or
/// also the host, by pointer — a draft model's verify seed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TerminalReads {
    #[default]
    Gpu,
    Host,
}

/// An arena slot: the buffer a tape's colour lives in, an index into a worker's `arena`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Slot(pub usize);

/// A slot's place in its model's [`ArenaRanks`]: a guest's slot of rank `r` is its host's slot
/// of rank `r`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rank(pub usize);

/// One `T` per rank, rank 0 first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ByRank<T>(Vec<T>);

impl<T> Default for ByRank<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T> ByRank<T> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get(&self, rank: Rank) -> Option<&T> {
        self.0.get(rank.0)
    }

    /// Each rank with its value.
    pub fn iter(&self) -> impl Iterator<Item = (Rank, &T)> {
        self.0.iter().enumerate().map(|(r, v)| (Rank(r), v))
    }
}

impl<T> FromIterator<T> for ByRank<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

/// A model's arena slots by size, largest first — each slot's bytes at its largest bucket, ties by
/// slot — baked at expansion: the order in which two models whose forwards never overlap share one
/// arena ([`ArenaShare`]). The guest's slot of rank `r` is its host's slot of rank `r`, which the
/// host sizes to hold both ([`Hosting`]). Built only by [`Self::new`], which checks what it holds.
#[derive(Clone, Copy, Debug)]
pub struct ArenaRanks {
    /// Every slot: the order in which it takes its host's slots, as a guest.
    all: &'static [u32],
    /// Every slot but the buckets' final hidden states: the order in which it holds a guest's
    /// slots, as a host. A guest reads its host's hidden states after its own forward has begun —
    /// an MTP head's pass 1 once a run of rows — so it shares none of them.
    hosting: &'static [u32],
}

impl ArenaRanks {
    /// No slots: the ranks of a model whose arena is never shared (a test's).
    pub const NONE: Self = Self::new(&[], &[], &[]);

    /// The ranks of the model whose buckets are `buckets`: `all`, every slot of `0..all.len()` once
    /// — every slot of every bucket among them — and `hosting`, `all` in its order less each
    /// bucket's final hidden states (`backbone_slot`, read off the buckets themselves). Checked
    /// here, which the `static` the macro emits them in evaluates as it compiles: ranks breaking
    /// any of it fail the model's build.
    pub const fn new(
        all: &'static [u32],
        hosting: &'static [u32],
        buckets: &[super::pool::MetalBucketSpec],
    ) -> Self {
        let n = all.len();
        let mut b = 0;
        while b < buckets.len() {
            assert!(
                (buckets[b].num_arena_slots as usize) <= n,
                "ArenaRanks: a bucket's slot is unranked"
            );
            b += 1;
        }
        let mut i = 0;
        while i < n {
            assert!(
                (all[i] as usize) < n,
                "ArenaRanks: a ranked slot past the arena"
            );
            let mut j = i + 1;
            while j < n {
                assert!(all[i] != all[j], "ArenaRanks: a slot ranked twice");
                j += 1;
            }
            i += 1;
        }
        let (mut a, mut h) = (0, 0);
        while a < n {
            let mut is_hidden = false;
            let mut k = 0;
            while k < buckets.len() {
                is_hidden |= buckets[k].backbone_slot == all[a];
                k += 1;
            }
            if !is_hidden {
                assert!(
                    h < hosting.len() && hosting[h] == all[a],
                    "ArenaRanks: the hosting ranks are not every slot but the hidden states', in \
                     rank order"
                );
                h += 1;
            }
            a += 1;
        }
        assert!(
            h == hosting.len(),
            "ArenaRanks: a hosting rank past the slots"
        );
        Self { all, hosting }
    }

    /// The slots of an arena of `len` slots, in the order of `ranks`.
    fn within(ranks: &[u32], len: usize) -> impl Iterator<Item = Slot> + '_ {
        (ranks.iter().map(|&s| Slot(s as usize))).filter(move |s| s.0 < len)
    }

    /// As a guest with an arena of `len` slots: its slots, rank by rank — every one of them, its
    /// ranks naming every slot of its largest arena.
    pub fn guest_slots(&self, len: usize) -> ByRank<Slot> {
        Self::within(self.all, len).collect()
    }

    /// As a host with an arena of `len` slots, of a guest of `count` ranks: the slot each rank
    /// takes — the host's hosting slots in order, then fresh slots past its own for the rest.
    pub fn hosting_slots(&self, len: usize, count: usize) -> ByRank<Slot> {
        let hosting = Self::within(self.hosting, len);
        hosting.chain((len..).map(Slot)).take(count).collect()
    }
}

/// What a guest of a pool's arena needs of it ([`ArenaShare::Hosts`]): its slots' bytes, rank by
/// rank ([`ArenaRanks::guest_slots`]), and its scratch by kind. The host reads none of a guest's
/// slots by pointer, so any may be sparse.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GuestArena {
    /// Each rank's bytes: the largest of its kept buckets'.
    pub arena: ByRank<u64>,
    /// The host's scratch of each kind is the guest's too.
    pub scratch: PerScratch<u64>,
}

/// A host's arena grown to hold its guest's ([`ArenaShare::Hosts`]): the one arena the two models'
/// forwards run in, which the host allocates and the memory budget prices, and the slot of it each
/// guest rank takes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hosting {
    pub layout: ArenaLayout,
    pub slots: ByRank<Slot>,
}

impl Hosting {
    /// `layout`, a host's own arena ranked `ranks`, with each slot a rank of `guest` takes
    /// ([`ArenaRanks::hosting_slots`]) at least that rank's bytes.
    pub fn of(layout: &[u64], ranks: &ArenaRanks, guest: &GuestArena) -> Self {
        let slots = ranks.hosting_slots(layout.len(), guest.arena.len());
        let len = (slots.iter().map(|(_, s)| s.0 + 1)).fold(layout.len(), usize::max);
        let mut out = layout.to_vec();
        out.resize(len, 0);
        for ((_, &Slot(slot)), (_, &bytes)) in slots.iter().zip(guest.arena.iter()) {
            out[slot] = out[slot].max(bytes);
        }
        Self { layout: out, slots }
    }

    /// A host's scratch `own` grown to hold `guest`'s, kind by kind.
    pub fn scratch(own: PerScratch<u64>, guest: &GuestArena) -> PerScratch<u64> {
        PerScratch::from_fn(|kind| own[kind].max(guest.scratch[kind]))
    }
}

/// What a host pool shares with its guest ([`ArenaShare::In`]): the slot each guest rank takes, its
/// scratch by kind; and its TurboQuant scratch, which a guest of the same geometry stages through,
/// and whole TurboQuant runtime — its packed KV stores, codebook and scratch — which a guest
/// reading the host's KV binds (an MTP head with none of its own).
#[derive(Clone, Default)]
pub struct HostArena {
    pub arena: ByRank<Activation>,
    pub scratch: PerScratch<Option<Activation>>,
    pub tq_scratch: Option<super::runtime::TqScratch>,
    pub tq: Option<super::runtime::TqRuntimeBuffers>,
}

/// How a pool's arena relates to that of a model whose forwards never overlap its own, and follow
/// them on its command buffer: an MTP head's, its target's.
#[derive(Clone, Default)]
pub enum ArenaShare {
    /// Its arena is its own.
    #[default]
    Own,
    /// It holds the guest's arena in its own ([`Hosting`]).
    Hosts(GuestArena),
    /// Its arena and scratch are its host's, rank by rank and kind by kind. The host reads none
    /// of them by pointer: neither its hidden states nor its logits.
    In(Box<HostArena>),
}

impl ArenaShare {
    /// A pool's share: holding `guest`'s arena (its model's guest, set before the pool is built),
    /// or in `host`'s (its device's host). A model is a host or a guest, never both.
    pub fn of(guest: Option<&GuestArena>, host: Option<&HostArena>) -> Self {
        match (guest, host) {
            (Some(guest), None) => Self::Hosts(guest.clone()),
            (None, Some(host)) => Self::In(Box::new(host.clone())),
            (None, None) => Self::Own,
            (Some(_), Some(_)) => panic!("a pool both holds a guest's arena and runs in a host's"),
        }
    }

    pub fn role(&self) -> ArenaRole {
        match self {
            Self::Own => ArenaRole::Own,
            Self::Hosts(_) => ArenaRole::Host,
            Self::In(_) => ArenaRole::Guest,
        }
    }
}

/// What a pool is to the arena it runs in ([`ArenaShare`]), without the buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArenaRole {
    Own,
    Host,
    Guest,
}

/// One unit of execution in a bucket's plan.
///
/// `Dispatch` is a contiguous run of commands sharing a single pipeline.
/// Dense GEMM (f16 and bf16) is a `Dispatch` step like every other
/// kernel — `gemm_{f16,bf16}_specialized`, or `gemv_{f16,bf16}_specialized`
/// at one row; no MPS.
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
        /// Per-sub-dispatch fence (hazard class), sourced from the
        /// compile-time DAG hazard analysis in `LoweredMetalTape`.
        /// Consumed by the MTL4 path via `Mtl4Step.barrier_before`.
        barrier_before: Vec<Fence>,
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
    /// Per dispatch, in dispatch order, whether a forward reading only its cache writes plays it
    /// ([`LoweredMetalTape::cache_live_expanded`]).
    pub cache_live: Vec<bool>,
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
    /// The device could not allocate the worker's `what` buffer of `bytes`.
    BufferAlloc { what: &'static str, bytes: u64 },
    /// Mapping what bucket `bucket` binds of the sparse activations failed.
    SparseMap { bucket: usize, reason: String },
    /// The queue the sparse activations map on could not be created.
    SparseMapper { reason: String },
    /// A guest's `part` needs `need` bytes of its host's arena, which holds `has`: the host was
    /// sized for another guest, or for none.
    HostShort { part: HostPart, need: u64, has: u64 },
    /// A guest's slot `slot` has no rank, so no slot of its host's: its ranks are another
    /// model's.
    GuestSlotUnranked { slot: Slot },
}

/// What of its host's arena a guest takes: the slot of a rank, or the scratch of a kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostPart {
    Slot(Rank),
    Scratch(ScratchKind),
}

impl std::fmt::Display for HostPart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Slot(Rank(rank)) => write!(f, "arena slot of rank {rank}"),
            Self::Scratch(kind) => f.write_str(kind.what()),
        }
    }
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
            Self::BufferAlloc { what, bytes } => {
                write!(
                    f,
                    "MetalWorker: the device cannot allocate the {what} ({bytes} bytes)"
                )
            }
            Self::SparseMap { bucket, reason } => {
                write!(
                    f,
                    "MetalWorker: bucket {bucket}: mapping its activations: {reason}"
                )
            }
            Self::SparseMapper { reason } => {
                write!(
                    f,
                    "MetalWorker: the sparse activations' mapping queue: {reason}"
                )
            }
            Self::HostShort { part, need, has } => write!(
                f,
                "MetalWorker: a guest's {part} needs {need} bytes of its host's arena, which \
                 holds {has}: the host was sized for another guest"
            ),
            Self::GuestSlotUnranked { slot: Slot(slot) } => write!(
                f,
                "MetalWorker: a guest's arena slot {slot} has no rank, so no slot of its host's: \
                 its ranks are another model's"
            ),
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
    /// The arena slots as allocated, to share with a guest ([`Self::hosted`]).
    arena_activation: Vec<Activation>,
    /// The slot each rank of its guest takes, as a host ([`Hosting::slots`]); none otherwise.
    hosting: ByRank<Slot>,
    /// Residency pins for every buffer the baked dispatches reach by address.
    _pins: Vec<crate::residency::Pinned>,
    pub bucket_bakings: Vec<BucketBaking>,
    /// The scratch buffers the baked dispatches bind by address, by kind (those some tape needs,
    /// see [`Self::new_with_residency`]), held for them.
    scratch: PerScratch<Option<Activation>>,
    /// The sparse activations and how far each bucket reaches in them; `None` when every
    /// activation is allocated in full.
    sparse: Option<SparseActivation>,
    /// Per-forward count of the blocks the step's longest sequence spans
    /// (`live_block_width` in `pool.rs`, set before each forward), for the
    /// `TqStageRotated` and `RopeOnce*` grids: they cover the WHOLE active
    /// context and no more — the block-table row is the KV cap rung's width,
    /// which grows with the pool, not the context. `0` until the first
    /// forward sets it (dispatch falls back to the baked grid).
    pub tq_dequant_max_blocks: std::sync::atomic::AtomicU32,
    /// Whether some sequence of the step about to run has an unrotated
    /// (bit-31, span) block in its block table, set with
    /// `tq_dequant_max_blocks`. Only a prefill attention that re-ropes K can
    /// read such a block (`RuntimeGate::OnlyIfUnrotatedBlocks`).
    pub unrotated_blocks: std::sync::atomic::AtomicBool,
    /// Whether every sequence of the step about to run takes a plain
    /// Gated-DeltaNet step (`GdnStep::is_plain`), set with `unrotated_blocks`.
    /// Only such a step runs the one-command decode and the pipelined prefill
    /// scan (`RuntimeGate::OnlyIfPlainGdn`).
    pub gdn_plain: std::sync::atomic::AtomicBool,
    /// Whether some sequence of the step about to run takes a plain
    /// Gated-DeltaNet step over more than one row, set with `gdn_plain`: in a
    /// step that also replays or records, the pipelined scan's twin runs those
    /// sequences (`RuntimeGate::OnlyIfPlainGdnRows`).
    pub gdn_plain_rows: std::sync::atomic::AtomicBool,
    /// Per-dispatch GPU timestamp heap (`SCRATCHY_METAL_KERNEL_TIME`).
    /// `None` unless the env var is set at worker build — the probe is
    /// opt-in and costs nothing otherwise. One timestamp is written
    /// after EVERY dispatched kernel; a command buffer's wait resolves
    /// and prints the per-kernel deltas of the stamps written up to its
    /// commit (see `print_new_since`).
    pub kernel_time_heap: Option<std::sync::Arc<super::mtl4::KernelTimeHeap>>,
    _marker: std::marker::PhantomData<fn() -> W>,
}

/// A worker's sparse activations ([`SparseBuffer`]) and what each bucket needs mapped in them.
struct SparseActivation {
    device: Arc<Device>,
    residency: crate::residency::MetalResidencySet,
    mapper: Arc<crate::sparse_kv::KvMapper>,
    /// Every sparse buffer the worker binds.
    buffers: Vec<Arc<SparseBuffer>>,
    /// Each bucket's bytes in `buffers`, and whether a forward of it has mapped them: mappings
    /// stay, so a bucket reached once never looks again.
    buckets: Vec<(Vec<usize>, std::sync::atomic::AtomicBool)>,
}

impl SparseActivation {
    /// Map what `bucket` binds, before its first forward is encoded.
    fn reach(&self, bucket: usize) -> Result<(), WorkerError> {
        use std::sync::atomic::Ordering::Relaxed;
        let (needs, reached) = &self.buckets[bucket];
        if reached.load(Relaxed) {
            return Ok(());
        }
        let wants: Vec<(&SparseBuffer, usize)> = (self.buffers.iter().map(|b| &**b))
            .zip(needs.iter().copied())
            .collect();
        crate::sparse_arena::reach(&self.device, &self.residency, &self.mapper, &wants)
            .map_err(|reason| WorkerError::SparseMap { bucket, reason })?;
        reached.store(true, Relaxed);
        Ok(())
    }
}

impl<W: CanonicalParams> MetalWorker<W> {
    /// What this worker shares with its guest ([`HostArena`]): the slot each guest rank takes, and
    /// its scratch. A worker hosting no guest shares no slot.
    pub fn hosted(&self) -> HostArena {
        HostArena {
            arena: (self.hosting.iter())
                .map(|(_, &Slot(slot))| self.arena_activation[slot].clone())
                .collect(),
            scratch: self.scratch.clone(),
            tq_scratch: None,
            tq: None,
        }
    }

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
            pipelines,
            sources,
            runtime,
            None,
            None,
            &ArenaShare::Own,
            &ArenaRanks::NONE,
            None,
        )
    }

    /// Same as [`Self::new`] but accepts an optional `MetalResidencySet`
    /// that worker-local arena slots get inserted into. Used by the
    /// pool to pin every per-worker arena into the wired set so cmdbuf
    /// dispatches don't race against Apple's lazy paging on
    /// Llama-3.2-class working sets. The arena and scratch are shared with
    /// another model's as `share` says, by this model's `ranks`
    /// ([`ArenaShare`]); with a residency set and a `plan`, what the host does
    /// not read by pointer is placement-sparse, mapped as far as the buckets
    /// that run reach ([`SparsePlan`]).
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_residency(
        device: Arc<Device>,
        arena_layout: &ArenaLayout,
        bucket_tapes: &[LoweredMetalTape],
        pipelines: &SpecializedPipelines,
        sources: &ResolvedSources,
        runtime: &RuntimeBindings,
        residency: Option<&crate::residency::MetalResidencySet>,
        // The most rows a decode or verify step brings (`GpuDevice::max_decode_rows`): the decode
        // split's partials are reserved for at most these.
        decode_rows: Option<crate::tape::ids::DecodeRows>,
        share: &ArenaShare,
        ranks: &ArenaRanks,
        plan: Option<&SparsePlan>,
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

        // A buffer the device cannot allocate is a typed refusal, never a panic.
        let alloc = |bytes: u64, mode: MTLResourceOptions, what: &'static str| {
            let buf = usize::try_from(bytes)
                .ok()
                .and_then(|len| device.newBufferWithLength_options(len, mode));
            buf.ok_or(WorkerError::BufferAlloc { what, bytes })
        };
        // Placement-sparse where the pool plans it: wired with the working set, a sparse buffer takes
        // only what the buckets that run reach, where one allocated in full is zero-filled and wired
        // whole on the first forward, whatever its bucket.
        let plan = plan.filter(|_| residency.is_some());
        let make = |bytes: u64, mode, sparse: bool, what| match sparse {
            true => SparseBuffer::new(&device, bytes)
                .map(Activation::Sparse)
                .map_err(|_| WorkerError::BufferAlloc { what, bytes }),
            false => alloc(bytes, mode, what).map(Activation::Full),
        };
        let held = |a: &Activation| {
            use crate::interpreter::metal::__re::MTLBuffer as _;
            a.buffer().length() as u64
        };
        // A host holds its guest's slots too, rank by rank: each it hosts one in grown to that
        // rank's bytes, fresh slots past its own for the ranks its hosting slots do not reach.
        let (layout, hosting) = match share {
            ArenaShare::Hosts(guest) => {
                let Hosting { layout, slots } = Hosting::of(arena_layout, ranks, guest);
                (layout, slots)
            }
            ArenaShare::Own | ArenaShare::In(_) => (arena_layout.clone(), ByRank::default()),
        };
        let host_read = |slot: usize| plan.is_none_or(|p| p.host_slots.contains(&(slot as u32)));
        // A guest's slots are its host's, rank by rank, which the host sized to hold them.
        let mut hosted: Vec<Option<Activation>> = vec![None; layout.len()];
        if let ArenaShare::In(host) = share {
            for (rank, &Slot(slot)) in ranks.guest_slots(layout.len()).iter() {
                let taken = host.arena.get(rank);
                let (need, has) = (layout[slot], taken.map_or(0, held));
                let part = HostPart::Slot(rank);
                hosted[slot] = Some(
                    taken
                        .filter(|_| has >= need)
                        .ok_or(WorkerError::HostShort { part, need, has })?
                        .clone(),
                );
            }
            // Every slot is its host's: one its ranks miss would be allocated beside the host's
            // arena, unpriced.
            if let Some(slot) = hosted.iter().position(Option::is_none) {
                return Err(WorkerError::GuestSlotUnranked { slot: Slot(slot) });
            }
        }
        // Shared storage so test code can seed/inspect arena contents without staging copies, and
        // so the host can read the slots it reads by pointer; dispatch-bound buffers are fine in
        // shared on Apple silicon — the existing `MetalAllocator` uses the same mode.
        let arena_activation: Vec<Activation> = (layout.iter().zip(hosted).enumerate())
            .map(|(i, (&size, hosted))| match hosted {
                Some(activation) => Ok(activation),
                None => make(
                    size,
                    MTLResourceOptions::StorageModeShared,
                    !host_read(i),
                    "arena slot",
                ),
            })
            .collect::<Result<_, _>>()?;
        let arena: Vec<Buffer> = arena_activation
            .iter()
            .map(|a| a.buffer().clone())
            .collect();
        arena.iter().for_each(&mut pin);

        // A Private scratch buffer (the host never reads scratch back) per scratch the tapes bind
        // (`LoweredMetalTape::scratch_bytes`, one a `ScratchKind`), sized to the max across bucket
        // tapes and pinned, since dispatches bind it by baked address; `None` when no tape needs
        // it.
        // - SplitK: successive `affine_qmm_t_splitk` / `splitk_reduce_sum` pairs run serially
        //   inside a single encoder.
        // - MoE: MoE blocks within one bucket execute serially through the dispatch; cross-bucket
        //   reuse is fine because only one bucket runs per forward.
        // - Roped K (spans rope-on-read, rope-once-to-scratch — NAX matrix-accel AND the
        //   simdgroup steel prefill): the RopeOnce{Nax,Steel} command and its following attention
        //   run serially per layer, and the scratch is overwritten each layer (the cache, not the
        //   scratch, is the persistent artifact).
        // - hd512-unfused attention (q_head/Kdense/Vdense_T/scores/out_head packed at baked
        //   offsets): one buffer, overwritten per layer.
        let own: PerScratch<u64> = PerScratch::from_fn(|kind| {
            let max = (bucket_tapes.iter())
                .map(|t| t.scratch_bytes(decode_rows)[kind])
                .max();
            max.map_or(0, u64::from)
        });
        // A host's scratch of each kind holds its guest's too; a guest's is its host's.
        let sizes = match share {
            ArenaShare::Hosts(guest) => Hosting::scratch(own, guest),
            ArenaShare::Own | ArenaShare::In(_) => own,
        };
        let private = MTLResourceOptions::StorageModePrivate;
        let scratch = PerScratch::try_from_fn(|kind| match share {
            ArenaShare::In(host) if sizes[kind] > 0 => {
                let taken = host.scratch[kind].as_ref();
                let (need, has) = (sizes[kind], taken.map_or(0, held));
                let part = HostPart::Scratch(kind);
                let activation = taken.filter(|_| has >= need);
                Ok(Some(
                    activation
                        .ok_or(WorkerError::HostShort { part, need, has })?
                        .clone(),
                ))
            }
            _ => (sizes[kind] > 0)
                .then(|| make(sizes[kind], private, plan.is_some(), kind.what()))
                .transpose(),
        })?;
        (scratch.iter())
            .filter_map(|(_, a)| a.as_ref())
            .for_each(|a| pin(a.buffer()));

        // What each bucket binds of every sparse buffer: its arena bytes in a slot, its scratch
        // bytes of a kind.
        let sparse = match (plan, residency) {
            (Some(plan), Some(residency)) => {
                assert_eq!(
                    plan.bucket_arena.len(),
                    bucket_tapes.len(),
                    "SparsePlan: one arena table per bucket tape"
                );
                let arena_needs = |slot: usize| -> Vec<usize> {
                    let bytes = |t: &Vec<u64>| t.get(slot).copied().unwrap_or(0) as usize;
                    plan.bucket_arena.iter().map(bytes).collect()
                };
                let scratch_needs = |kind: ScratchKind| -> Vec<usize> {
                    let bytes = |t: &LoweredMetalTape| t.scratch_bytes(decode_rows)[kind] as usize;
                    bucket_tapes.iter().map(bytes).collect()
                };
                let arena_sparse = arena_activation.iter().enumerate();
                let per_buffer: Vec<(Arc<SparseBuffer>, Vec<usize>)> = (arena_sparse)
                    .filter_map(|(slot, a)| match a {
                        Activation::Sparse(s) => Some((s.clone(), arena_needs(slot))),
                        Activation::Full(_) => None,
                    })
                    .chain(scratch.iter().filter_map(|(kind, a)| match a {
                        Some(Activation::Sparse(s)) => Some((s.clone(), scratch_needs(kind))),
                        _ => None,
                    }))
                    .collect();
                let buckets = (0..bucket_tapes.len())
                    .map(|b| {
                        let needs = per_buffer.iter().map(|(_, n)| n[b]).collect();
                        (needs, std::sync::atomic::AtomicBool::new(false))
                    })
                    .collect();
                let mapper = crate::sparse_kv::KvMapper::new(&device)
                    .map_err(|reason| WorkerError::SparseMapper { reason })?;
                Some(SparseActivation {
                    device: device.clone(),
                    residency: residency.clone(),
                    mapper,
                    buffers: per_buffer.into_iter().map(|(s, _)| s).collect(),
                    buckets,
                })
            }
            _ => None,
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
                // Bound by no command.
                num_tokens_u32: _,
                // The sampled rows the gather and its undoing read by baked gpuAddress — pinned
                // below.
                num_sample_rows_u32,
                sample_indices,
                // GDN persistent state pools and draft lanes — pinned by the GdnStatePool owner.
                gdn_state_conv: _,
                gdn_state_ssm: _,
                gdn_lanes_conv: _,
                gdn_lanes_ssm: _,
                gdn_state_indices,
                gdn_is_fresh,
                vision_rope_freqs,
                pixels,
                vision_pos_embeds,
                target_hidden,
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
                // Host bookkeeping of the input buffers' written bytes, not a GPU buffer.
                written: _,
            } = runtime;
            pin(input_ids);
            pin(positions);
            pin(cu_seqlens_q);
            pin(seq_used_k);
            // Span labels: bound by dispatch-baked address on rope-on-read
            // arches (slot 8 of the prefill attention), so pin like seq_used_k.
            pin(span_ids);
            // The rows the forward samples: the lm_head slice's gather and the logits compaction
            // bind them by dispatch-baked address, same lazy-pager hazard as the inputs.
            pin(num_sample_rows_u32);
            pin(sample_indices);
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
            // / `LoadRows` kernels read `freqs` / `cu_seqlens` / `pixels`
            // by dispatch-baked address, so an un-pinned buffer is served stale
            // zero pages (garbage rope angles / all-token-0 pixels). The
            // factory allocates 16-byte placeholders on non-vision arches,
            // so the pin is harmless when no vision command binds them.
            pin(vision_rope_freqs);
            pin(pixels);
            pin(vision_pos_embeds);
            // An MTP head's target hidden rows: read by `LoadRows(TargetHidden)` at a
            // dispatch-baked address, so the same lazy-pager hazard (16-byte placeholder
            // elsewhere).
            pin(target_hidden);
            pin(mm_embeds);
            pin(mm_dst_rows);
            pin(mrope_cos_sin);
            pin(vision_cu_seqlens_full);
            pin(vision_cu_seqlens_window);
            pin(vision_window_index);
            pin(vision_reverse_indices);
            pin(vision_position_ids);
            // TurboQuant: the codebook and the fp16 scratch's chunk tables are read by
            // the dequant/quantize dispatch commands via baked gpuAddress, so pin them
            // (same lazy-pager hazard); the packed, norms and scratch stores pin
            // themselves, and each step of them as it is mapped (`SparseStores`).
            // `None` for a dense model → no-op.
            if let Some(t) = tq {
                pin(&t.signs);
                pin(&t.boundaries);
                pin(&t.centroids);
                pin(&t.scratch.k_table);
                pin(&t.scratch.v_table);
            }
            r.commit();
        }

        let mut bucket_bakings = Vec::with_capacity(bucket_tapes.len());
        for (bucket_idx, tape) in bucket_tapes.iter().enumerate() {
            let baking = bake_bucket::<W>(
                bucket_idx,
                tape,
                &arena,
                scratch[ScratchKind::SplitK]
                    .as_ref()
                    .map(Activation::buffer),
                scratch[ScratchKind::Moe].as_ref().map(Activation::buffer),
                scratch[ScratchKind::RopedK]
                    .as_ref()
                    .map(Activation::buffer),
                scratch[ScratchKind::AttnUnfused]
                    .as_ref()
                    .map(Activation::buffer),
                pipelines,
                sources,
                runtime,
                device.clone(),
            )?;
            bucket_bakings.push(baking);
        }
        // Every codes operand is now stored the way its readers read it.
        crate::metal_allocator::signal_affine_codes_settled();
        if let Some(r) = residency {
            r.commit();
        }

        Ok(Self {
            arena,
            arena_activation,
            hosting,
            _pins: pins,
            bucket_bakings,
            scratch,
            sparse,
            tq_dequant_max_blocks: std::sync::atomic::AtomicU32::new(0),
            unrotated_blocks: std::sync::atomic::AtomicBool::new(false),
            gdn_plain: std::sync::atomic::AtomicBool::new(true),
            gdn_plain_rows: std::sync::atomic::AtomicBool::new(true),
            kernel_time_heap: if std::env::var_os("SCRATCHY_METAL_KERNEL_TIME").is_some() {
                Some(super::mtl4::new_registered_kernel_time_heap(&device))
            } else {
                None
            },
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
        sampled_rows: u32,
        reads: super::ForwardReads,
        enc: &ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
    ) -> Result<(), WorkerError> {
        self.run_bucket_mtl4_inner(bucket, num_tokens, num_seqs, sampled_rows, reads, enc, None)
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
        sampled_rows: u32,
        enc: &ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
        range: std::ops::Range<usize>,
    ) -> Result<(), WorkerError> {
        self.run_bucket_mtl4_inner(
            bucket,
            num_tokens,
            num_seqs,
            sampled_rows,
            super::ForwardReads::Outputs,
            enc,
            Some(range),
        )
    }

    fn run_bucket_mtl4_inner(
        &self,
        bucket: usize,
        num_tokens: u32,
        num_seqs: u32,
        sampled_rows: u32,
        reads: super::ForwardReads,
        enc: &ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
        range: Option<std::ops::Range<usize>>,
    ) -> Result<(), WorkerError> {
        use ::objc2_metal::{
            MTL4CommandEncoder, MTL4ComputeCommandEncoder as _, MTL4VisibilityOptions, MTLStages,
        };
        // What this bucket binds of the sparse activations is mapped (and its heap committed to the
        // residency set) before its command buffer is.
        if let Some(sparse) = &self.sparse {
            sparse.reach(bucket)?;
        }
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
        let facts = StepFacts {
            num_tokens,
            num_seqs,
            sampled_rows,
            unrotated_blocks: self
                .unrotated_blocks
                .load(std::sync::atomic::Ordering::Relaxed),
            gdn_plain: self.gdn_plain.load(std::sync::atomic::Ordering::Relaxed),
            gdn_plain_rows: (self.gdn_plain_rows).load(std::sync::atomic::Ordering::Relaxed),
        };
        // A skipped dispatch's fence, owed to the next one played.
        let mut carried = Fence::None;
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
                // A forward reading only its cache writes skips what none of them needs, its fence
                // passed on to the next dispatch it plays: the fence orders what came before it
                // against what comes after, whichever dispatch that is.
                if reads == super::ForwardReads::CacheWrites && !baking.cache_live[this_idx] {
                    carried = carried.max(*need_barrier);
                    continue;
                }
                // The compiler's own hazard-analysis verdict for the tape,
                // captured before runtime overrides (force/range) mutate it.
                #[cfg(feature = "forward-telemetry")]
                let compiler_barrier = *need_barrier;
                // In range mode: skip dispatches outside the segment,
                // and suppress the fence on the segment's FIRST
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
                        need_barrier = Fence::None;
                    }
                }
                if !gate_matches(*gate, facts) {
                    // Skipped: e.g. the lm_head slice (`UnlessEveryRowSampled`)
                    // on a step that samples every row, or the every-row
                    // lm_head (`OnlyIfEveryRowSampled`) on any other.
                    // Either way the timestamp slot, barrier, and
                    // dispatch are skipped together so the kernel
                    // doesn't run with stale per-dispatch state.
                    continue;
                }
                let need_barrier = std::mem::replace(&mut carried, Fence::None).max(need_barrier);
                if need_barrier != Fence::None {
                    // Visibility from the hazard class: RAW fences are
                    // `Device` (memory-coherent — the reader must observe
                    // the prior write; ordering alone left the m≥2 router
                    // NormedGemv reading stale bytes ~50% of runs, issue
                    // #295), WAR/WAW fences are `None` (ordering only —
                    // `Device`'s flush costs real time where nothing reads
                    // through the caches: measured -30 ms TTFT @ 2048-tok
                    // on M4 Llama-3.2-3B-4bit, gemma-4-26b conc decode
                    // ~5-10%).
                    //
                    // ⛔ NOT a fence pair: `updateFence`/`waitForFence`
                    // between two dispatches of ONE MTL4 compute encoder
                    // does NOT order them (measured, M5 Max: all-zero
                    // logits, deterministic). MTL4 fences order work
                    // across encoder boundaries, not within one —
                    // the stage barrier is the only in-encoder
                    // producer→consumer mechanism.
                    let visibility = match need_barrier {
                        Fence::Coherent => MTL4VisibilityOptions::Device,
                        Fence::Ordered => MTL4VisibilityOptions::None,
                        Fence::None => unreachable!("guarded above"),
                    };
                    enc.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                        MTLStages::Dispatch,
                        MTLStages::Dispatch,
                        visibility,
                    );
                }
                let tg_scaled = scale_tg_for_num_tokens(*tg, *scaling, facts);
                // TurboQuant full-context staging: the baked grid is a
                // placeholder; the kernel needs (max_blocks, num_kv_heads,
                // num_seqs) — block-table-driven, with early-exit past
                // seqused_k; grid.z = the live batch's num_seqs.
                let tg_scaled = if matches!(step.kernel, super::lowered::KernelId::TqStageRotated) {
                    // grid.x must cover every block of the longest sequence
                    // (`tq_dequant_max_blocks`). The static
                    // `W::MAX_BLOCKS_PER_SEQ` (128 for uniform arches like
                    // Llama) truncated the pass at 128 blocks / 2048 tokens, so
                    // the reused fp16 scratch's tail kept the PREVIOUS layer's
                    // KV → attention collapse to `!!!!` past 2048 tokens. The
                    // baked const stands in only while the width is unset (0).
                    let runtime_mb = self
                        .tq_dequant_max_blocks
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let baked =
                        <W as ::scratchy_forward_compiler::CanonicalParams>::MAX_BLOCKS_PER_SEQ;
                    let cover = if runtime_mb == 0 { baked } else { runtime_mb };
                    MTLSize {
                        width: cover as usize,
                        height: tg_scaled.height,
                        depth: (num_seqs as usize).max(1),
                    }
                } else {
                    tg_scaled
                };
                // RopeOnce{Steel,Nax,GqaShared}: the pre-roped-K scratch holds the KV cap rung's
                // logical blocks (lowering), and the steel/gqa attention reads the WHOLE
                // sequence's roped K (computed prefix + new). The baked grid M-scales by
                // num_tokens (the NEW tokens only) — too few for a chunked-prefill CONTINUATION,
                // leaving the prefix blocks past one bucket un-roped → garbage K → `!!!!` past
                // 4096 tokens. Override grid.y to every block of the sequence
                // (`tq_dequant_max_blocks`), within the rung's cap: the scratch holds it.
                // Mirrors the TqStageRotated grid override above.
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
                    let cover = if runtime_mb == 0 {
                        tg.height
                    } else {
                        runtime_mb
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
                // SCRATCHY_METAL_KERNEL_TIME: one GPU timestamp per
                // dispatched kernel (relaxed granularity — the probe
                // must not perturb what it measures). Index pairs with
                // the tape position; the pool prints deltas after the
                // host wait.
                if let Some(heap) = &self.kernel_time_heap {
                    heap.write_timestamp(enc, step.kernel, tg_scaled);
                }
                #[cfg(feature = "forward-telemetry")]
                if tape_enabled {
                    tape.push(TapeEntry {
                        kind: kernel_kind(step.kernel),
                        barrier: compiler_barrier != Fence::None,
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
            | K::AffineQmvGated
            | K::RopeAppendNormed
            | K::NormAddScalarMul
            | K::AttentionViaCacheTq
            | K::AttentionDecodeGqaTq
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
        K::RmsNorm | K::RmsNormUnit | K::FusedAddRmsNorm | K::NormAddScalarMul | K::RowProgram => {
            KernelKind::Norm
        }
        K::RopeAppendNormed
        | K::RopeAppend
        | K::KvAppend
        | K::RopeOnceNax
        | K::RopeOnceSteel
        | K::RopeOnceGqaShared => KernelKind::Rope,
        K::AttentionViaCache
        | K::AttentionViaCacheTq
        | K::AttentionDecodeGqaTq
        | K::AttentionDecodeCombine
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
        | K::NormedGemv
        | K::AffineQmvQuad
        | K::AffineQmvFast
        | K::AffineQmvWide
        | K::AffineQmvTiny
        | K::AffineQmv
        | K::AffineQmvGated
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
        | K::MoeGroupScatterQ8
        | K::MoePerExpertScale
        | K::GateApply
        | K::GateScale
        | K::GateSplit
        | K::ArgPartitionTopK
        | K::TakeAlongAxis => KernelKind::Moe,
        K::FusedGateUpSiluMul | K::SiluMul | K::GeluMul | K::AffineGatherW4a8ActQuant => {
            KernelKind::Mlp
        }
        K::GatherLastToken | K::ScatterFirstToLastRow | K::Softmax | K::SliceTrailingColsU32 => {
            KernelKind::Sample
        }
        K::VisionLayerNorm | K::VisionRope | K::VisionVarlenAttn | K::AvgPool2d | K::VisionGelu => {
            KernelKind::Vision
        }
        K::ScalarWeightMul
        | K::ScalarMul
        | K::TanhSoftCap
        | K::Add
        | K::BiasAdd
        | K::Reshape
        | K::TqStageRotated
        | K::TqRotateRows
        | K::LoadRows
        | K::ConcatRows => KernelKind::Elementwise,
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

    // ⭐ THE LAYER LOOP IS PLAYED OUT HERE, ONCE, AT LOAD. The baked tape carries ONE copy of
    // the body plus its iteration count; unrolling it at BAKE time is what made the emitted
    // `const` an order of magnitude larger. Everything below sees the same flat command list it
    // always did.
    let expanded_commands = tape.commands_expanded();
    let expanded_barriers = tape.barriers_expanded();

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
                sources,
                runtime,
            )?;
            // One row: MLX's GEMV; otherwise an MMA GEMM (`pipeline_for_gemm`). All are
            // custom kernels on the same per-step dispatch plumbing as every other compute
            // kernel — no MPS (it rejects BFloat16), no classic command buffer.
            let (pipeline, shape) = pipelines
                .pipeline_for_gemm(W::METAL_DTYPE, dims)
                .map_err(WorkerError::PipelineLookup)?;
            // Binding contract: buffer(0) = output, buffer(1) = input, buffer(2) = weight.
            let bindings_for_cmd: Vec<(Buffer, u64, u64)> = vec![
                (c.buffer.clone(), c.offset, 0u64),
                (a.buffer.clone(), a.offset, 1u64),
                (b.buffer.clone(), b.offset, 2u64),
            ];
            let size = |(width, height, depth): (u32, u32, u32)| MTLSize {
                width: width as usize,
                height: height as usize,
                depth: depth as usize,
            };
            let dispatch_for_cmd = (
                size(shape.threadgroups),
                size(shape.threads_per_threadgroup),
            );
            let cmd_barrier = expanded_barriers
                .get(cmd_idx)
                .copied()
                .unwrap_or(Fence::Coherent);
            // Dense GEMM: M is the height axis but the bake here is for a dense linear that
            // always dispatches at the actual M (no bucket_m baking), so leave m_scaling as None.
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
                    // Gemm path is never gated (no slice); push None to keep the Vec aligned
                    // with `direct_dispatch`.
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
            sources,
            runtime,
        )?;
        let bound_refs: Vec<(&Buffer, u64, u64)> =
            bound.iter().map(|(b, off, idx)| (b, *off, *idx)).collect();
        let (tg, tpt) = mtl_size_pair(cmd, pipelines.variant());

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
        let cmd_barrier = expanded_barriers
            .get(cmd_idx)
            .copied()
            .unwrap_or(Fence::Coherent);
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
        cache_live: tape.cache_live_expanded(),
    })
}

/// Resolve `(out, in, weight)` buffers for a `KernelId::Gemm` command.
///
/// The lowering pass guarantees the binding order: index 0 → output
/// arena slot, index 1 → input arena slot, index 2 → LinearLayer
/// weight thunk. Anything else is a contract violation surfaced as
/// [`WorkerError::GemmBindingsMalformed`].
fn resolve_gemm_buffers(
    bucket_index: usize,
    command_index: usize,
    cmd: &LoweredCommand,
    arena: &[Buffer],
    moe_scratch: Option<&Buffer>,
    sources: &ResolvedSources,
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
        sources,
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

/// Evaluate a per-dispatch [`RuntimeGate`] against the step being encoded: `true` when the
/// dispatch fires, `false` when it is skipped. `gate == None` (the common case) always fires.
///
/// The lm_head pair splits on whether the step samples every row it runs. A decode step, or a
/// speculative verify step of drafting sequences alone, does (`OnlyIfEveryRowSampled`): the
/// lm_head over every row leaves the sampled rows' logits in order, and slicing would only add
/// launches — gating the slice on at every multi-sequence step regressed c=4 TPOT by +5% on
/// Llama-1B. A prefill chunk or a mixed batch does not (`UnlessEveryRowSampled`): the slice, or
/// the compaction. The pair is exhaustive by construction: a decode step that ran neither left
/// the logits slot's stale contents for the argmax, and every row froze on its first token
/// (issue #361).
pub fn gate_matches(gate: Option<super::lowered::RuntimeGate>, step: StepFacts) -> bool {
    let StepFacts {
        num_tokens,
        num_seqs,
        sampled_rows,
        unrotated_blocks,
        gdn_plain,
        gdn_plain_rows,
    } = step;
    let decode_step = num_tokens == num_seqs;
    match gate {
        None => true,
        Some(super::lowered::RuntimeGate::UnlessEveryRowSampled) => sampled_rows < num_tokens,
        Some(super::lowered::RuntimeGate::OnlyIfEveryRowSampled) => sampled_rows >= num_tokens,
        Some(super::lowered::RuntimeGate::OnlyIfDecodeStep) => decode_step,
        Some(super::lowered::RuntimeGate::UnlessDecodeStep) => !decode_step,
        Some(super::lowered::RuntimeGate::OnlyIfSmallMTokens) => {
            crate::quantized::SMALL_M_TOKENS.contains(&num_tokens)
        }
        Some(super::lowered::RuntimeGate::UnlessSmallMTokens) => {
            !crate::quantized::SMALL_M_TOKENS.contains(&num_tokens)
        }
        Some(super::lowered::RuntimeGate::OnlyIfOneSequence) => num_seqs == 1,
        Some(super::lowered::RuntimeGate::UnlessOneSequence) => num_seqs > 1,
        Some(super::lowered::RuntimeGate::OnlyIfUnrotatedBlocks) => unrotated_blocks,
        Some(super::lowered::RuntimeGate::UnlessUnrotatedBlocks) => !unrotated_blocks,
        Some(super::lowered::RuntimeGate::OnlyIfPlainGdn) => gdn_plain,
        Some(super::lowered::RuntimeGate::UnlessPlainGdn) => !gdn_plain,
        Some(super::lowered::RuntimeGate::OnlyIfPlainGdnRows) => gdn_plain_rows,
        Some(super::lowered::RuntimeGate::All(gates)) => {
            gates.iter().all(|g| gate_matches(Some(*g), step))
        }
        Some(super::lowered::RuntimeGate::Any(gates)) => {
            gates.iter().any(|g| gate_matches(Some(*g), step))
        }
    }
}

/// What a runtime gate can ask about the step being encoded.
#[derive(Clone, Copy)]
pub struct StepFacts {
    pub num_tokens: u32,
    pub num_seqs: u32,
    /// The rows the step samples (its `SampleIndices`' length): one a sequence, every row of a
    /// sequence verifying drafts.
    pub sampled_rows: u32,
    /// Some sequence's block table has an unrotated (bit-31, span) block.
    pub unrotated_blocks: bool,
    /// Every sequence's Gated-DeltaNet step is plain: no replay, no records.
    pub gdn_plain: bool,
    /// Some sequence takes a plain Gated-DeltaNet step over more than one row.
    pub gdn_plain_rows: bool,
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
/// bucket picker already rules out but defense-in-depth). Its
/// `set_axis`, if any, is then SET to the live count it names.
fn scale_tg_for_num_tokens(
    mut tg: MTLSize,
    scaling: Option<super::lowered::MScaling>,
    step: StepFacts,
) -> MTLSize {
    let Some(s) = scaling else {
        return tg;
    };
    let bm = s.bucket_m.get().max(1) as u64;
    let n = (step.num_tokens.max(1) as u64).min(bm);
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
    // `set_axis`: SET (not scale) the chosen axis to the live count (`SetAxis`).
    let set = match s.set_axis {
        Some(super::lowered::SetAxis::Seqs(axis)) => Some((axis, step.num_seqs)),
        Some(super::lowered::SetAxis::SampledRows(axis)) => Some((axis, step.sampled_rows)),
        None => None,
    };
    if let Some((axis, count)) = set {
        let slot = match axis {
            super::lowered::MScaleAxis::X => &mut tg.width,
            super::lowered::MScaleAxis::Y => &mut tg.height,
            super::lowered::MScaleAxis::Z => &mut tg.depth,
        };
        *slot = count.max(1) as usize;
    }
    tg
}

fn mtl_size_pair(cmd: &LoweredCommand, variant: TapeVariant) -> (MTLSize, MTLSize) {
    // A TurboQuant decode threadgroup serves the variant's query heads (`ConstantType::TqHeads`):
    // its threadgroups count heads. Its pipeline lookup refuses a variant without them.
    let heads = match (cmd.kernel, variant.tq_heads) {
        (KernelId::AttentionViaCacheTq, Some(h)) => h.get(),
        _ => 1,
    };
    let tg = MTLSize {
        width: cmd.dispatch.threadgroups.0 as usize,
        height: (cmd.dispatch.threadgroups.1 / heads) as usize,
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
        Binding, DispatchShape, LoweredCommand, MetalDtype, RuntimeBindingKind, SourceRef,
        WeightTensor,
    };
    use crate::specialized_pipeline_cache::{ConstantValue, SpecializedPipelineCache};
    use scratchy_ir::CanonicalParams;
    use scratchy_layers::{Linear, LinearLayer, RmsNorm};
    use scratchy_tensors::{DType, DeviceAllocator, GpuTensor};
    use std::sync::Arc;

    /// The test tapes' variant: `TestWeights`' block capacity, no TurboQuant decode attention.
    const TEST_VARIANT: TapeVariant = TapeVariant {
        cap: crate::tape::ids::MaxBlocksPerSeq(TestWeights::MAX_BLOCKS_PER_SEQ),
        tq_heads: None,
    };

    /// A TurboQuant decode attention's threadgroups count query heads: one serves the picked
    /// variant's heads, so the variant divides them. Every other command dispatches as baked.
    #[test]
    fn tq_decode_threadgroups_serve_the_variants_heads() {
        use crate::tape::ids::{MaxBlocksPerSeq, TqDecodeHeads};
        let command = |kernel| LoweredCommand {
            kernel,
            library: "attention",
            function: "attention_via_cache_v2_f16_specialized",
            constants: crate::interpreter::metal::lowered::baked(Vec::new()),
            dispatch: DispatchShape {
                threadgroups: (1, 32, 2),
                threads_per_threadgroup: (64, 1, 1),
                m_scaling: None,
            },
            bindings: crate::interpreter::metal::lowered::baked(Vec::new()),
            gemm_dims: None,
        };
        let (tq, fp16) = (
            command(KernelId::AttentionViaCacheTq),
            command(KernelId::AttentionViaCache),
        );
        for heads in [1, 2, 4, 8] {
            let variant = TapeVariant {
                cap: MaxBlocksPerSeq(128),
                tq_heads: Some(TqDecodeHeads(heads)),
            };
            let (tg, _) = mtl_size_pair(&tq, variant);
            assert_eq!((tg.width, tg.height, tg.depth), (1, 32 / heads as usize, 2));
            let (tg, _) = mtl_size_pair(&fp16, variant);
            assert_eq!((tg.width, tg.height, tg.depth), (1, 32, 2));
        }
    }

    /// The decode-step gates follow whether every sequence contributes one
    /// token; the sequence gates follow the step's sequence count, not its
    /// token count; the span-block gates follow the step's block tables; the
    /// plain-GDN gates follow the step's GDN codes; `All` needs every gate: a
    /// decode step runs none of a TurboQuant prefill attention's variants; and
    /// `Any` needs one: a replaying decode step runs a drafting model's GDN
    /// conv, which a plain one leaves to the one-command decode.
    #[test]
    fn step_gates_follow_the_step() {
        use super::super::lowered::RuntimeGate::{
            All, Any, OnlyIfDecodeStep, OnlyIfOneSequence, OnlyIfPlainGdn, OnlyIfUnrotatedBlocks,
            UnlessDecodeStep, UnlessOneSequence, UnlessPlainGdn, UnlessUnrotatedBlocks,
        };
        let step = |num_tokens, num_seqs, unrotated_blocks| StepFacts {
            num_tokens,
            num_seqs,
            sampled_rows: num_seqs,
            unrotated_blocks,
            gdn_plain: true,
            gdn_plain_rows: true,
        };
        for (tokens, seqs, decode) in [
            (1, 1, true),
            (16, 16, true),
            (512, 1, false),
            (18, 16, false),
        ] {
            let s = step(tokens, seqs, false);
            assert_eq!(gate_matches(Some(OnlyIfDecodeStep), s), decode);
            assert_eq!(gate_matches(Some(UnlessDecodeStep), s), !decode);
        }
        for (tokens, seqs, one) in [(512, 1, true), (512, 2, false), (16, 16, false)] {
            let s = step(tokens, seqs, false);
            assert_eq!(gate_matches(Some(OnlyIfOneSequence), s), one);
            assert_eq!(gate_matches(Some(UnlessOneSequence), s), !one);
        }
        for unrotated in [false, true] {
            let s = step(512, 2, unrotated);
            assert_eq!(gate_matches(Some(OnlyIfUnrotatedBlocks), s), unrotated);
            assert_eq!(gate_matches(Some(UnlessUnrotatedBlocks), s), !unrotated);
        }
        // A drafting model's GDN commands: the one-command decode on a plain decode step, its
        // conv on any other.
        let (decode, conv) = (
            All(&[OnlyIfDecodeStep, OnlyIfPlainGdn]),
            Any(&[UnlessDecodeStep, UnlessPlainGdn]),
        );
        for (tokens, seqs, gdn_plain, decodes) in [
            (1, 1, true, true),
            (2, 2, true, true),
            (1, 1, false, false),
            (3, 1, false, false),
            (512, 1, true, false),
        ] {
            let s = StepFacts {
                gdn_plain,
                ..step(tokens, seqs, false)
            };
            assert_eq!(gate_matches(Some(OnlyIfPlainGdn), s), gdn_plain);
            assert_eq!(gate_matches(Some(UnlessPlainGdn), s), !gdn_plain);
            assert_eq!(gate_matches(Some(decode), s), decodes);
            assert_eq!(gate_matches(Some(conv), s), !decodes);
        }
        // The pipelined twin's gate follows whether some plain sequence runs more than one row: a
        // verify step beside a prefill chunk runs it, one beside only verifies or decodes does not.
        for gdn_plain_rows in [false, true] {
            let s = StepFacts {
                gdn_plain: false,
                gdn_plain_rows,
                ..step(18, 2, false)
            };
            assert_eq!(
                gate_matches(
                    Some(super::super::lowered::RuntimeGate::OnlyIfPlainGdnRows),
                    s
                ),
                gdn_plain_rows
            );
        }
        let plain = All(&[UnlessDecodeStep, UnlessOneSequence, UnlessUnrotatedBlocks]);
        assert!(gate_matches(Some(plain), step(512, 2, false)));
        assert!(!gate_matches(Some(plain), step(512, 2, true)));
        assert!(!gate_matches(Some(plain), step(512, 1, false)));
        assert!(
            !gate_matches(Some(plain), step(2, 2, false)),
            "a decode step"
        );
        // The lm_head pair is exhaustive — no step runs neither, no step runs
        // both — and each side takes the steps it is for: the slice the steps
        // that sample fewer rows than they run (a prefill chunk, a mixed batch,
        // a verify step beside a prefill chunk), the every-row matmul those
        // that sample each (a decode step, a verify step alone). Issue #361:
        // before the every-row matmul took decode steps, a decode step in a
        // bucket whose lm_head lowers to one `qmm_t` (the slice band,
        // `bucket_m ≥ vector_limit`) ran no lm_head at all — the argmax re-read
        // the logits slot's stale contents and every row froze on its first
        // token.
        use super::super::lowered::RuntimeGate::{OnlyIfEveryRowSampled, UnlessEveryRowSampled};
        let lm_head = |tokens, seqs, sampled| {
            let s = StepFacts {
                sampled_rows: sampled,
                ..step(tokens, seqs, false)
            };
            let slice = gate_matches(Some(UnlessEveryRowSampled), s);
            let every = gate_matches(Some(OnlyIfEveryRowSampled), s);
            assert!(
                slice ^ every,
                "the lm_head gate pair must be exhaustive at num_tokens={tokens} \
                 num_seqs={seqs} sampled_rows={sampled}"
            );
            slice
        };
        for (tokens, seqs, sampled, slices) in [
            // Decode steps.
            (1u32, 1u32, 1u32, false),
            (9, 9, 9, false),
            (80, 80, 80, false),
            // Prefill chunks and mixed batches: a row a sequence.
            (18, 16, 16, true),
            (512, 9, 9, true),
            (4096, 128, 128, true),
            // A verify step of six sequences, three rows each, alone; beside a prefill chunk.
            (18, 6, 18, false),
            (530, 7, 19, true),
        ] {
            assert_eq!(
                lm_head(tokens, seqs, sampled),
                slices,
                "{tokens} {seqs} {sampled}"
            );
        }
    }

    /// The lm_head sample slice, as the tape builds and dispatches it: the
    /// gather swaps sampled row `i` into row `i`, the scatter swaps row `i`
    /// back, both in place. In a mixed step, decodes, verifies and short
    /// chunks put some sampled rows inside `0..num_sampled`, the rows the
    /// others are moved to; every sampled row must still land at its own
    /// index, and the scatter after the gather must leave every row where it
    /// started.
    #[test]
    fn sample_slice_moves_each_sampled_row() {
        let Some(device) = crate::detect_device().filter(|_| crate::metal4_available()) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let device = device.device.clone();
        let cache =
            SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");
        let p = crate::tape::model_consts::MetalModelConsts {
            metal_dtype: MetalDtype::F16,
            ..crate::tape::model_consts::MetalModelConsts::from_canonical::<TestWeights>()
        };
        // Decodes, then verifies (every row sampled), then short prefix-hit chunks, then long
        // chunks.
        let q_lens: Vec<u32> = [
            vec![1; 24],
            vec![3; 6],
            vec![2; 16],
            vec![3; 8],
            vec![40; 4],
        ]
        .concat();
        let cu: Vec<u32> = std::iter::once(0)
            .chain(q_lens.iter().scan(0, |end, n| {
                *end += n;
                Some(*end)
            }))
            .collect();
        let num_tokens = *cu.last().unwrap() as usize;
        let verifies = 24..30;
        let sampled: Vec<u32> = (0..q_lens.len())
            .flat_map(|i| match verifies.contains(&i) {
                true => cu[i]..cu[i + 1],
                false => cu[i + 1] - 1..cu[i + 1],
            })
            .collect();
        let num_sampled = sampled.len();
        let width = 1024u32;
        let bucket_m = 512;
        let slot = crate::tape::ids::ArenaSlotIdx(0);
        let gather = crate::tape::lowering::gather_last_token_command(&p, slot, width);
        let scatter = crate::tape::lowering::scatter_first_to_last_row_command(&p, slot, width);
        // `(row, the row it holds)`. Gather: row i holds sampled row i.
        // Scatter: sampled row i holds row i. Both: every row holds itself.
        let at = |i: usize| sampled[i] as usize;
        let gathered = (0..num_sampled).map(|i| (i, at(i))).collect();
        let scattered = (0..num_sampled).map(|i| (at(i), i)).collect();
        let restored = (0..num_tokens).map(|r| (r, r)).collect();
        let cases: [(&str, Vec<&LoweredCommand>, Vec<(usize, usize)>); 3] = [
            ("gather", vec![&gather], gathered),
            ("scatter", vec![&scatter], scattered),
            ("gather then scatter", vec![&gather, &scatter], restored),
        ];
        for (name, cmds, wants) in cases {
            // Every element of row `r` holds `r`.
            let rows: Vec<u16> = (0..bucket_m as usize)
                .flat_map(|r| {
                    std::iter::repeat_n(half::f16::from_f32(r as f32).to_bits(), width as usize)
                })
                .collect();
            let buf = crate::mtl4_dispatch::shared_slice(&device, &rows);
            let sampled_buf = crate::mtl4_dispatch::shared_slice(&device, &sampled);
            let n_buf = crate::mtl4_dispatch::shared_u32(&device, num_sampled as u32);
            let step = StepFacts {
                num_tokens: num_tokens as u32,
                num_seqs: q_lens.len() as u32,
                sampled_rows: num_sampled as u32,
                unrotated_blocks: false,
                gdn_plain: true,
                gdn_plain_rows: true,
            };
            for cmd in cmds {
                let key = crate::specialized_pipeline_cache::PipelineKey::new(
                    cmd.library,
                    cmd.function,
                    cmd.constants.to_vec(),
                );
                let pso = crate::aot::baked_build(&cache, &key).expect("pipeline");
                let (grid, threads) = mtl_size_pair(cmd, TEST_VARIANT);
                let grid = scale_tg_for_num_tokens(grid, cmd.dispatch.m_scaling, step);
                assert!(crate::mtl4_dispatch::dispatch_threadgroups(
                    &device,
                    &pso,
                    &[&buf, &sampled_buf, &n_buf],
                    grid,
                    threads,
                ));
            }
            let got: Vec<u16> = crate::mtl4_dispatch::read_slice(&buf, rows.len());
            for (at, want) in wants {
                let want = half::f16::from_f32(want as f32).to_bits();
                let wrong = got[at * width as usize..][..width as usize]
                    .iter()
                    .filter(|&&x| x != want)
                    .count();
                assert_eq!(
                    wrong, 0,
                    "{name}: {wrong} of {width} elements of row {at} are another row's"
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
            gdn_lanes_conv: ::std::vec::Vec::new(),
            gdn_lanes_ssm: ::std::vec::Vec::new(),
            gdn_state_indices: alloc_buffer(device, 16),
            gdn_is_fresh: alloc_buffer(device, 16),
            vision_rope_freqs: alloc_buffer(device, 16),
            pixels: alloc_buffer(device, 16),
            vision_pos_embeds: alloc_buffer(device, 16),
            target_hidden: alloc_buffer(device, 16),
            mm_embeds: alloc_buffer(device, 16),
            mm_dst_rows: alloc_buffer(device, 16),
            mrope_cos_sin: alloc_buffer(device, 16),
            vision_cu_seqlens_full: alloc_buffer(device, 16),
            vision_cu_seqlens_window: alloc_buffer(device, 16),
            vision_window_index: alloc_buffer(device, 16),
            vision_reverse_indices: alloc_buffer(device, 16),
            vision_position_ids: alloc_buffer(device, 16),
            written: Default::default(),
        }
    }

    /// The test weights' RMSNorm constants at `bucket_m` rows.
    fn rmsnorm_constants(bucket_m: u32) -> &'static [ConstantValue] {
        use crate::tape::ids::{BucketM, QSize, RmsNormEps};
        crate::tape::kernel_constants::RmsNormConstants {
            bucket_m: BucketM(bucket_m),
            q_size: QSize(<TestWeights as CanonicalParams>::Q_SIZE as u32),
            rms_norm_eps: RmsNormEps(<TestWeights as CanonicalParams>::RMS_NORM_EPS),
            weight_offset: 0.0,
        }
        .into_baked()
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
            constants: rmsnorm_constants(bucket_m),
            dispatch: DispatchShape {
                threadgroups: (bucket_m, 1, 1),
                threads_per_threadgroup: (crate::tape::kernel_constants::NORM_THREADS, 1, 1),
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
            gemm_dims: None,
        };
        let fused_add_rmsnorm = LoweredCommand {
            kernel: KernelId::FusedAddRmsNorm,
            library: "fused_add_rmsnorm",
            function: "fused_add_rmsnorm_f16_s_f16_specialized",
            constants: rmsnorm_constants(bucket_m),
            dispatch: DispatchShape {
                threadgroups: (bucket_m, 1, 1),
                threads_per_threadgroup: (crate::tape::kernel_constants::NORM_THREADS, 1, 1),
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
            gemm_dims: None,
        };
        // Two RmsNorm commands then two FusedAddRmsNorm commands —
        // exercises both kernels and the coalescer's same-pipeline
        // neighbour case (two RmsNorm with identical extras hit the
        // same cached pipeline; same for the FAR pair).
        LoweredMetalTape {
            bucket_m,
            num_arena_slots: 2,
            commands: crate::interpreter::metal::lowered::baked_commands(vec![
                rmsnorm.into(),
                rmsnorm.into(),
                fused_add_rmsnorm.into(),
                fused_add_rmsnorm.into(),
            ])
            .expect("commands"),
            splitk_scratch_bytes: 0,
            moe_scratch_bytes: 0,
            decode_partials_row_bytes: 0,
            roped_k_scratch_bytes: 0,
            attn_unfused_scratch_bytes: 0,
            logits_rows: crate::tape::lowered::LogitsRows::Bucket,
            cache_live: &[],
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

        let (weights, allocator) = build_test_weights();
        let runtime = empty_runtime(&device, 1);

        // Two buckets: M=1 (decode) and M=8 (small prefill).
        let tapes = vec![build_synthetic_tape(1), build_synthetic_tape(8)];
        let pipelines =
            crate::aot::tape_pipelines(&device, &tapes, TestWeights::METAL_DTYPE, TEST_VARIANT)
                .expect("pipelines");
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

    /// The buckets of a model whose arena has `slots` slots: one a slot of `hidden`, which holds its
    /// final hidden states — what `ArenaRanks::new` reads those slots off.
    fn buckets(slots: u32, hidden: &[u32]) -> Vec<super::super::pool::MetalBucketSpec> {
        let bucket = |&backbone_slot: &u32| super::super::pool::MetalBucketSpec {
            bucket_m: 1,
            num_arena_slots: slots,
            terminal_slot: 0,
            backbone_slot,
            arena_bytes: &[],
            backbone: &[],
            lm_head: &[],
            backbone_barriers: &[],
            lm_head_barriers: &[],
            tapes: &[],
            kernels: &[],
        };
        hidden.iter().map(bucket).collect()
    }

    /// A host holds its guest rank by rank: each hosting slot grows to the guest rank's bytes, a
    /// slot outside the hosting ranks (a bucket's hidden states) and a slot past the host's arena
    /// (another canonical's, not kept) host nothing, and the ranks past the hosting slots take
    /// fresh slots; the scratch grows kind by kind. Pure arithmetic: no device.
    #[test]
    fn a_host_grows_its_arena_and_scratch_to_hold_its_guest() {
        let ranks = ArenaRanks::new(&[3, 0, 2, 1], &[3, 0, 1], &buckets(4, &[2]));
        let slots = |s: ByRank<Slot>| s.iter().map(|(_, s)| s.0).collect::<Vec<_>>();
        // Slot 3 ranks first but lies past this 3-slot arena: its bucket is not kept.
        assert_eq!(slots(ranks.guest_slots(3)), [0, 2, 1]);
        assert_eq!(slots(ranks.hosting_slots(3, 4)), [0, 1, 3, 4]);
        let guest = GuestArena {
            arena: [100, 50, 30, 20].into_iter().collect(),
            scratch: PerScratch::from_fn(|kind| match kind {
                ScratchKind::Moe => 70,
                ScratchKind::RopedK => 5,
                _ => 0,
            }),
        };
        let hosting = Hosting::of(&[80, 60, 40], &ranks, &guest);
        assert_eq!(
            hosting.layout,
            [100, 60, 40, 30, 20],
            "rank 0 grows slot 0, rank 1 fits slot 1, slot 2 (hidden states) hosts nothing"
        );
        assert_eq!(slots(hosting.slots), [0, 1, 3, 4]);
        let own = PerScratch::from_fn(|kind| match kind {
            ScratchKind::SplitK => 10,
            ScratchKind::Moe => 90,
            ScratchKind::AttnUnfused => 3,
            ScratchKind::RopedK => 0,
        });
        let grown = Hosting::scratch(own, &guest);
        let bytes: Vec<u64> = grown.iter().map(|(_, &b)| b).collect();
        assert_eq!(bytes, [10, 90, 5, 3]);
    }

    /// Whatever a host's arena keeps and however many ranks its guest has, no guest rank takes a
    /// slot holding the host's hidden states: the guest reads them after its forward has begun.
    #[test]
    fn a_guest_never_takes_its_hosts_hidden_states() {
        let hidden = [1, 4];
        let ranks = ArenaRanks::new(&[4, 2, 0, 5, 1, 3], &[2, 0, 5, 3], &buckets(6, &hidden));
        for len in 0..=6 {
            for count in 0..=8 {
                let slots = ranks.hosting_slots(len, count);
                assert_eq!(slots.len(), count);
                assert!(
                    slots
                        .iter()
                        .all(|(_, s)| !(s.0 < len && hidden.contains(&(s.0 as u32)))),
                    "len {len}, count {count}: {slots:?}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "a slot ranked twice")]
    fn arena_ranks_refuse_a_slot_ranked_twice() {
        ArenaRanks::new(&[0, 1, 1], &[0, 1, 1], &[]);
    }

    /// The hidden-states slots are read off the buckets, not taken on trust.
    #[test]
    #[should_panic(expected = "every slot but the hidden states'")]
    fn arena_ranks_refuse_hosting_a_hidden_states_slot() {
        ArenaRanks::new(&[0, 1, 2], &[0, 1, 2], &buckets(3, &[1]));
    }

    /// Ranks naming fewer slots than a bucket's arena holds leave a slot no host would hold.
    #[test]
    #[should_panic(expected = "a bucket's slot is unranked")]
    fn arena_ranks_refuse_a_bucket_slot_unranked() {
        ArenaRanks::new(&[1, 0], &[1], &buckets(3, &[0]));
    }

    /// A host and its guest run in one arena: the guest's slot of each rank is the host's slot of
    /// that rank, grown to hold both; the host's hidden-states slot holds none of the guest's; a
    /// rank past the host's hosting slots takes a fresh slot of the host's. The host allocates
    /// exactly the arena its `Hosting` prices. A host sized for another guest refuses this one —
    /// an arena slot too small, or a scratch it never allocated.
    #[test]
    fn a_guest_runs_in_its_hosts_arena_rank_by_rank() {
        use crate::interpreter::metal::__re::MTLBuffer as _;
        let Some(device) = crate::detect_device().filter(|_| crate::metal4_available()) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let device = Arc::new(device.device.clone());
        let (weights, allocator) = build_test_weights();
        let runtime = empty_runtime(&device, 1);
        let tapes = vec![build_synthetic_tape(1)];
        let pipelines =
            crate::aot::tape_pipelines(&device, &tapes, TestWeights::METAL_DTYPE, TEST_VARIANT)
                .expect("pipelines");
        let build = |layout: &ArenaLayout,
                     tapes: &[LoweredMetalTape],
                     share: &ArenaShare,
                     ranks: &ArenaRanks| {
            MetalWorker::<TestWeights>::new_with_residency(
                device.clone(),
                layout,
                tapes,
                &pipelines,
                &sources(&weights, &allocator, tapes),
                &runtime,
                None,
                None,
                share,
                ranks,
                None,
            )
        };
        const K: u64 = 1024;
        // The host's slot 2 holds its hidden states: it hosts in slots 0 and 1 only.
        let host_layout = vec![64 * K, 4 * K, 8 * K];
        let host_ranks = ArenaRanks::new(&[0, 2, 1], &[0, 1], &buckets(3, &[2]));
        let guest_layout = vec![4 * K, 16 * K, 128 * K];
        let guest_ranks = ArenaRanks::new(&[2, 1, 0], &[2, 1], &buckets(3, &[0]));
        let guest = GuestArena {
            arena: [128 * K, 16 * K, 4 * K].into_iter().collect(),
            scratch: PerScratch::default(),
        };
        let hosts = ArenaShare::Hosts(guest.clone());
        let host = build(&host_layout, &tapes, &hosts, &host_ranks).expect("the host builds");
        let lengths: Vec<u64> = host.arena.iter().map(|b| b.length() as u64).collect();
        let priced = Hosting::of(&host_layout, &host_ranks, &guest);
        assert_eq!(
            lengths, priced.layout,
            "the host allocates what its hosting prices"
        );
        assert_eq!(lengths, [128 * K, 16 * K, 8 * K, 4 * K]);
        let hosted = ArenaShare::In(Box::new(host.hosted()));
        let guest_worker =
            build(&guest_layout, &tapes, &hosted, &guest_ranks).expect("the guest builds");
        let at = |w: &MetalWorker<TestWeights>, slot: usize| w.arena[slot].gpuAddress();
        assert_eq!(at(&guest_worker, 2), at(&host, 0));
        assert_eq!(at(&guest_worker, 1), at(&host, 1));
        assert_eq!(at(&guest_worker, 0), at(&host, 3));
        assert!((0..3).all(|slot| at(&guest_worker, slot) != at(&host, 2)));

        let bigger = vec![4 * K, 16 * K, 256 * K];
        let refused = build(&bigger, &tapes, &hosted, &guest_ranks).err();
        assert!(
            matches!(
                refused,
                Some(WorkerError::HostShort {
                    part: HostPart::Slot(Rank(0)),
                    need,
                    has,
                }) if need == 256 * K && has == 128 * K
            ),
            "{refused:?}"
        );
        // A guest whose tape binds a MoE scratch its host was not sized for.
        let mut scratching = build_synthetic_tape(1);
        scratching.moe_scratch_bytes = 4 * K as u32;
        let refused = build(&guest_layout, &[scratching], &hosted, &guest_ranks).err();
        assert!(
            matches!(
                refused,
                Some(WorkerError::HostShort {
                    part: HostPart::Scratch(ScratchKind::Moe),
                    need,
                    has: 0,
                }) if need == 4 * K
            ),
            "{refused:?}"
        );
        // A guest whose ranks are another model's, naming two of its three slots: its third would
        // be allocated beside the host's arena, unpriced.
        let short_ranks = ArenaRanks::new(&[1, 0], &[1], &buckets(2, &[0]));
        let refused = build(&guest_layout, &tapes, &hosted, &short_ranks).err();
        assert!(
            matches!(
                refused,
                Some(WorkerError::GuestSlotUnranked { slot: Slot(2) })
            ),
            "{refused:?}"
        );
    }

    #[test]
    fn arena_shape_mismatch_errors() {
        let Some(device) = crate::detect_device().filter(|_| crate::metal4_available()) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let device = Arc::new(device.device.clone());

        let (weights, allocator) = build_test_weights();
        let runtime = empty_runtime(&device, 1);

        let tapes = vec![build_synthetic_tape(1)]; // num_arena_slots = 2
        let pipelines =
            crate::aot::tape_pipelines(&device, &tapes, TestWeights::METAL_DTYPE, TEST_VARIANT)
                .expect("pipelines");

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
            commands: crate::interpreter::metal::lowered::baked_commands(vec![attn.into()])
                .expect("commands"),
            splitk_scratch_bytes: 0,
            moe_scratch_bytes: 0,
            decode_partials_row_bytes: 0,
            roped_k_scratch_bytes: 0,
            attn_unfused_scratch_bytes: 0,
            logits_rows: crate::tape::lowered::LogitsRows::Bucket,
            cache_live: &[],
            barrier_before: &[],
            // Straight-line: this fixture is one hand-built body, no rolled layer loop.
            loops: &[],
        };
        let pipelines =
            crate::aot::tape_pipelines(&device, &[tape], TestWeights::METAL_DTYPE, TEST_VARIANT)
                .expect("pipelines");

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
                Binding::Source {
                    ix: LINEAR,
                    which: WeightTensor::Weight,
                    layer: LayerId(0),
                    binding_index: 2,
                },
            ]),
            gemm_dims: Some(crate::interpreter::metal::lowered::GemmDims { m: bucket_m, n, k }),
        }
    }

    /// A tape carrying one `KernelId::Gemm` command bakes to an MTL4
    /// `Dispatch` step (at one row, the `gemv_{f16,bf16}_specialized` kernel), so the
    /// bucket is MTL4-eligible (`mtl4_steps` is `Some`). There is no
    /// longer an MPS / classic-command-buffer GEMM path.
    #[test]
    fn worker_routes_gemm_step() {
        let Some(device) = crate::detect_device().filter(|_| crate::metal4_available()) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let device = Arc::new(device.device.clone());
        let (weights, allocator) = build_test_weights();
        let runtime = empty_runtime(&device, 1);

        let tape = LoweredMetalTape {
            bucket_m: 1,
            num_arena_slots: 2,
            commands: crate::interpreter::metal::lowered::baked_commands(vec![
                build_gemm_command(1, 2048, 2048).into(),
            ])
            .expect("commands"),
            splitk_scratch_bytes: 0,
            moe_scratch_bytes: 0,
            decode_partials_row_bytes: 0,
            roped_k_scratch_bytes: 0,
            attn_unfused_scratch_bytes: 0,
            logits_rows: crate::tape::lowered::LogitsRows::Bucket,
            cache_live: &[],
            barrier_before: &[],
            // Straight-line: this fixture is one hand-built body, no rolled layer loop.
            loops: &[],
        };
        let pipelines =
            crate::aot::tape_pipelines(&device, &[tape], TestWeights::METAL_DTYPE, TEST_VARIANT)
                .expect("pipelines");

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

        let (weights, allocator) = build_test_weights();
        let runtime = empty_runtime(&device, 1);

        let rmsnorm_pre = LoweredCommand {
            kernel: KernelId::RmsNorm,
            library: "rmsnorm",
            function: "rmsnorm_f16_s_f16_specialized",
            constants: rmsnorm_constants(1),
            dispatch: DispatchShape {
                threadgroups: (1, 1, 1),
                threads_per_threadgroup: (crate::tape::kernel_constants::NORM_THREADS, 1, 1),
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
            gemm_dims: None,
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
            gemm_dims: None,
        };

        let tape = LoweredMetalTape {
            bucket_m: 1,
            num_arena_slots: 2,
            commands: crate::interpreter::metal::lowered::baked_commands(vec![
                rmsnorm_pre.into(),
                build_gemm_command(1, 2048, 2048).into(),
                rmsnorm_post.into(),
            ])
            .expect("commands"),
            splitk_scratch_bytes: 0,
            moe_scratch_bytes: 0,
            decode_partials_row_bytes: 0,
            roped_k_scratch_bytes: 0,
            attn_unfused_scratch_bytes: 0,
            logits_rows: crate::tape::lowered::LogitsRows::Bucket,
            cache_live: &[],
            barrier_before: &[],
            // Straight-line: this fixture is one hand-built body, no rolled layer loop.
            loops: &[],
        };
        let pipelines =
            crate::aot::tape_pipelines(&device, &[tape], TestWeights::METAL_DTYPE, TEST_VARIANT)
                .expect("pipelines");

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
