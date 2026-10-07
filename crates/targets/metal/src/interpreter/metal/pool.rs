// SPDX-License-Identifier: Apache-2.0
//! `MetalWorkerPool`: growable, capped, semaphore-bounded checkout/checkin.
//!
//! The pool starts at size 1 and grows on demand up to `max_workers`.
//! Each worker holds a private arena, a private [`RuntimeBindings`],
//! and one baked execution plan per bucket — never shared across workers.
//! `checkout()` blocks if every worker is in use *and* the pool is at
//! cap; otherwise it grows by one and hands the new worker out.
//!
//! `max_workers` is supplied by the caller (typically derived as
//! `floor((device_total - weights - misc) / per_worker_arena)` by
//! the model loader / `#[forward]` macro). Keeping the budget
//! computation outside the pool avoids the pool needing to
//! introspect device or weight memory.
//!
//! Synchronization uses `std::sync::{Mutex, Condvar}` — the host
//! side is sync (one forward per checked-out worker), so async
//! primitives buy nothing.

use std::sync::{Arc, Condvar, Mutex};

use crate::interpreter::metal::__re::{Buffer, Device, MTLBuffer};
use objc2_metal::{
    MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandEncoder, MTL4CommandQueue, MTLDevice,
};

use crate::specialized_pipeline_cache::SpecializedPipelineCache;

use super::forward::{Deferral, ForwardError, ForwardInputs, InFlight};
use super::lowered::{LoweredMetalTape, ModelSources};
use super::pipelines::SpecializedPipelines;
use super::runtime::{InputWrite, InputWrites, Padding, RuntimeBindings};
use super::worker::{ArenaLayout, LentActivation, MetalWorker, ResolvedSources, WorkerError};
use crate::MetalAllocator;
use crate::tape::constants::TapeVariant;
use crate::tape::ids::{HeadDim, MaxBlocksPerSeq, NumKvHeads, NumQHeads, TqDecodeHeads};
use crate::tape::lowered::{ClassedTape, KvAddressing};
use objc2::runtime::ProtocolObject;
use scratchy_ir::{CanonicalParams, Instruction};

/// One bucket's compile-time data, ready to be lowered + handed to a
/// [`MetalWorkerPool`].
///
/// Emitted by the `#[forward]` macro (Phase 5.F.5) as one row per
/// solved bucket per canonical model. The macro materializes:
///  - `bucket_m` — the workload point this bucket was specialized for;
///  - `num_arena_slots` — colored slot count from `colored_slot_map()`;
///  - `backbone` / `lm_head` — the bucket's `Instruction` static
///    slices, identical to the cuda-side `BACKBONE_M_<wp>` /
///    `LM_HEAD_M_<wp>` statics.
///
/// [`MetalWorkerPool::for_buckets`] calls [`lower`] on the concatenated
/// `(backbone ++ lm_head)` to produce a [`LoweredMetalTape`] per spec
/// at constructor time. Concatenation matches the cuda interpreter's
/// behavior — `forward()` runs backbone then lm_head as one logical
/// pass for a given bucket — and the metal worker bakes both halves
/// into the bucket's single execution plan so `forward()` dispatches
/// both halves without an extra mid-bucket boundary.
///
/// The slices are `&'static` because the macro emits them as static
/// items; the spec is `Copy` so callers can drop the bucket plan into
/// an `Arc<[MetalBucketSpec]>` cheaply.
pub struct MetalBucketSpec {
    pub bucket_m: u32,
    pub num_arena_slots: u32,
    /// Index in the colored arena where this bucket's terminal
    /// activation lands (lm_head output for decoder layouts; the
    /// backbone output for encoder layouts). The worker pool's
    /// `forward()` callback reads `worker.arena[terminal_slot]` to
    /// expose logits to the engine.
    pub terminal_slot: u32,
    /// Index in the colored arena of the lm_head's input — the final (post-norm) hidden states,
    /// `[num_tokens, hidden]`. Kept live through the lm_head, so it is readable after the
    /// forward (an MTP head's next input). Equal to `terminal_slot` for encoder layouts.
    pub backbone_slot: u32,
    /// Per-arena-slot byte sizes derived from the FUF tile shapes
    /// at macro-expansion time (post-coloring, with bucket-specific
    /// `num_tokens` baked in). `len() == num_arena_slots`. The
    /// pool takes the elementwise max across every bucket to size
    /// the single per-worker arena.
    pub arena_bytes: &'static [u64],
    pub backbone: &'static [Instruction],
    pub lm_head: &'static [Instruction],
    /// MTL4 encoder barrier flags computed at macro time from the
    /// FUF dataflow graph (one bool per `Instruction` in
    /// `backbone`/`lm_head`). `true` means the MTL4 path must emit a
    /// `Dispatch→Dispatch` barrier before this instruction's first
    /// dispatched `LoweredCommand`. The runtime carries these
    /// straight to `Mtl4Step.barrier_before`; no runtime hazard
    /// walk. See `scratchy-forward-compiler-macro::interpreter_codegen::
    /// lower_bucket` for the analysis.
    pub backbone_barriers: &'static [bool],
    pub lm_head_barriers: &'static [bool],
    /// The macro-baked tape rungs for this bucket — one per (generation
    /// class × chunked addressing × KV cap × TurboQuant decode heads). The
    /// pool picks one at load. Empty in cuda-macro builds, where the metal
    /// statics are cfg'd out anyway, and for a canonical the front end refused.
    pub tapes: &'static [crate::tape::lowered::ClassedTape],
    /// Every kernel of a baked library ([`crate::aot::baked_library`]) the model's tapes name,
    /// compiled with its constants: the model's one table, which each of its buckets names.
    pub kernels: &'static [crate::tape::lowered::BakedKernel],
}

impl Clone for MetalBucketSpec {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for MetalBucketSpec {}

/// Errors surfaced by [`MetalWorkerPool::for_buckets`] before the pool
/// reaches its first eager-spawn `WorkerError` path.
///
/// Covers (a) the pipeline cache's library loads —
/// distinct from per-worker `WorkerError::PipelineLookup` because the
/// failure is one-shot and pre-pool; (b) a bucket with no baked rung for
/// the device and KV capacity; (c) any `WorkerError` from the
/// inner [`MetalWorkerPool::new`] call.
#[derive(Debug)]
pub enum PoolBuildError {
    /// A baked or synthesized library failed to load. Message is the
    /// underlying [`MetalStreamError`](crate::stream::MetalStreamError).
    PipelineCacheBuild(String),
    /// Bucket `bucket_m` has no baked rung for this device whose KV cap holds `need` blocks per
    /// sequence (none at all for a canonical the front end refused).
    NoRung {
        bucket_m: u32,
        need: MaxBlocksPerSeq,
    },
    /// The device's name names no chip generation a tape is baked for.
    UnknownDevice(String),
    /// `ScratchyWeights::metal_rungs` handed out something other than [`MetalRungs`].
    NotMetalRungs,
    /// The eager-spawn first worker (or any structural pool prereq)
    /// reported a [`WorkerError`].
    Worker(WorkerError),
    /// Caller-provided `bucket_specs` was empty. The pool always
    /// needs at least one bucket; degenerate models that emit none
    /// would fail at `pick_bucket` time anyway, so we fail early.
    NoBuckets,
    /// The device's IO-registry entry has no GPU core count
    /// ([`crate::device::gpu_cores`]), which the TurboQuant decode tapes need.
    UnknownGpuCores,
}

impl std::fmt::Display for PoolBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PipelineCacheBuild(s) => write!(
                f,
                "MetalWorkerPool::for_buckets: pipeline cache build failed: {s}"
            ),
            Self::NoRung { bucket_m, need } => write!(
                f,
                "MetalWorkerPool::for_buckets: bucket M={bucket_m} has no baked tape rung for \
                 this device that holds {} KV blocks per sequence",
                need.get()
            ),
            Self::NotMetalRungs => f.write_str(
                "MetalWorkerPool: the model's `metal_rungs` is not a `MetalRungs` (stale build)",
            ),
            Self::UnknownDevice(name) => write!(
                f,
                "MetalWorkerPool::for_buckets: device `{name}` is no chip generation a tape is \
                 baked for"
            ),
            Self::Worker(e) => write!(f, "MetalWorkerPool::for_buckets: {e}"),
            Self::NoBuckets => write!(f, "MetalWorkerPool::for_buckets: bucket_specs is empty"),
            Self::UnknownGpuCores => write!(
                f,
                "MetalWorkerPool::for_buckets: the device's IO-registry entry has no gpu-core-count"
            ),
        }
    }
}

impl std::error::Error for PoolBuildError {}

impl From<WorkerError> for PoolBuildError {
    fn from(e: WorkerError) -> Self {
        Self::Worker(e)
    }
}

/// Factory closure invoked once per worker creation to produce a
/// fresh [`RuntimeBindings`] sized for the worker's largest bucket.
///
/// A type-erased `Arc<dyn Fn>` rather than a generic so the pool can
/// stay non-generic over the closure type — there's exactly one runtime
/// layout per (model, max bucket) and the factory captures it once at
/// pool construction.
///
/// The pool shares itself across its spawned worker threads (see the
/// [`MetalWorkerPool`] `Send`/`Sync` impls below), so the factory it
/// holds must be `Send + Sync` too. The macro-emitted closures capture
/// `Vec<Buffer>` — `objc2-metal`'s `Retained<ProtocolObject<dyn MTL*>>`,
/// whose protocol traits don't carry the auto markers — so the wrapper
/// asserts `Send`/`Sync` on the same grounds as the pool: the retained
/// handles' retain/release is thread-safe and the factory is only ever
/// *invoked* single-threaded, from the checkout that owns the device.
#[derive(Clone)]
pub struct RuntimeFactory(Arc<FactoryFn>);

struct FactoryFn(Box<dyn Fn(&Device, &LentActivation) -> RuntimeBindings>);

// SAFETY: the boxed closure captures only Metal object handles (objc2
// `Retained`, thread-safe retain/release) and POD; identical grounds to
// `MetalWorkerPool`'s own `Send`/`Sync` impls. See `RuntimeFactory` doc.
unsafe impl Send for FactoryFn {}
unsafe impl Sync for FactoryFn {}

impl RuntimeFactory {
    /// Wrap a per-worker `RuntimeBindings` builder.
    pub fn new(f: impl Fn(&Device, &LentActivation) -> RuntimeBindings + 'static) -> Self {
        Self(Arc::new(FactoryFn(Box::new(f))))
    }

    /// Build a fresh set of runtime bindings for a worker on `device`, placing what it can in
    /// `lent` ([`LentActivation::tq_scratch`]).
    pub fn build(&self, device: &Device, lent: &LentActivation) -> RuntimeBindings {
        ((self.0).0)(device, lent)
    }
}

/// One unit the pool hands out: a worker plus its private
/// [`RuntimeBindings`].
pub struct PooledWorker<W: CanonicalParams> {
    pub worker: MetalWorker<W>,
    pub runtime: RuntimeBindings,
}

/// RAII guard returned by [`MetalWorkerPool::checkout`]. Returns the
/// underlying [`PooledWorker`] to the pool on drop.
pub struct WorkerGuard<'pool, W: CanonicalParams> {
    pool: &'pool MetalWorkerPool<W>,
    inner: Option<PooledWorker<W>>,
}

impl<W: CanonicalParams> std::ops::Deref for WorkerGuard<'_, W> {
    type Target = PooledWorker<W>;
    fn deref(&self) -> &PooledWorker<W> {
        self.inner.as_ref().expect("guard inner already taken")
    }
}

impl<W: CanonicalParams> std::ops::DerefMut for WorkerGuard<'_, W> {
    fn deref_mut(&mut self) -> &mut PooledWorker<W> {
        self.inner.as_mut().expect("guard inner already taken")
    }
}

impl<W: CanonicalParams> Drop for WorkerGuard<'_, W> {
    fn drop(&mut self) {
        if let Some(pooled) = self.inner.take() {
            self.pool.checkin(pooled);
        }
    }
}

/// Growable, capped worker pool. Generic over the model's
/// [`CanonicalParams`] — one pool per loaded model variant.
///
/// The pool stays parameterized over `W` for the per-bucket
/// `LoweredMetalTape` (workers bake dispatchs from these), but it
/// does *not* hold a back-reference to the loaded `Weights`: it
/// resolves the tapes' model sources once at construction
/// ([`ResolvedSources`]) and keeps only their buffers, so it can be
/// stored as a field on the `Weights` struct without an `Arc`-cycle.
pub struct MetalWorkerPool<W: CanonicalParams> {
    device: Arc<Device>,
    /// Allocator that owns the `MTLBuffer` arenas the loaded
    /// `GpuTensor`s point into. The worker uses
    /// [`MetalAllocator::buffer_for`] to map a tensor's raw pointer
    /// back to `(&MTLBuffer, offset)` for encoder bindings.
    ///
    /// The allocator also owns the shared `MetalResidencySet` that
    /// pins every weight / arena / KV-cache buffer as resident across
    /// cmdbufs (so large Llama-3.2-class working sets don't race
    /// against Apple's lazy paging and produce non-deterministic
    /// decode output). The pool reads the set off the allocator and
    /// (a) hands it to spawned workers so per-worker arena buffers
    /// also get pinned, and (b) attaches it to the dispatch queue on
    /// the first `forward()`.
    allocator: Arc<MetalAllocator>,
    pipelines: Arc<SpecializedPipelines>,
    bucket_tapes: Arc<[LoweredMetalTape]>,
    /// The KV cap rung the tapes are baked for: the row stride each step's block tables are laid
    /// out at.
    block_table_stride: MaxBlocksPerSeq,
    /// Every model tensor the tapes bind, resolved once in [`Self::new`].
    sources: ResolvedSources,
    arena_layout: Arc<ArenaLayout>,
    runtime_factory: RuntimeFactory,
    max_workers: usize,
    /// Tracks whether the allocator's residency set has been committed
    /// yet. Committed lazily on the first `forward()` (after the
    /// `register_mmap` / arena / KV-cache inserts are queued); MTL4
    /// command buffers then declare it per-CB via `useResidencySet:`.
    /// Gated by an atomic so the one-time commit isn't repeated across
    /// thousands of forwards.
    residency_attached: std::sync::atomic::AtomicBool,
    /// Phase A.3 MTL4 surface. Lazily initialized on the first forward.
    /// Panics on init if MTL4 is unavailable — MTL4 is a hard assertion
    /// on the host (macOS 15+ / Apple Family 7+), not an opt-in gate
    /// (the metal backend is MTL4-only; MTL3 was removed).
    mtl4: Mutex<Option<Mtl4Pool>>,
    inner: Mutex<PoolInner<W>>,
    cv: Condvar,
}

/// Pool-owned MTL4 surface. The command buffer is re-created per forward
/// (cheap — Metal pools internally). The shared event is monotonically
/// signaled after each commit.
struct Mtl4Pool {
    queue: crate::interpreter::metal::__re::Mtl4Queue,
    /// Command allocators, each with the staging a deferred forward's input writes ride in, taken
    /// in turn: one is reset only once its last command buffer is done, so a forward is encoded
    /// while the ones before it run — the one in flight, the one queued, and this one.
    ring: [RingSlot; 3],
    /// The ring slot the next command buffer takes.
    next: usize,
    shared_event: crate::interpreter::metal::__re::SharedEvent,
    signal_counter: u64,
}

/// One of [`Mtl4Pool::ring`]'s command allocators.
struct RingSlot {
    allocator: crate::interpreter::metal::__re::Mtl4Allocator,
    /// What a deferred forward's recorded input writes are copied from.
    staging: Option<crate::residency::Pinned>,
    /// The shared-event value its last command buffer signals.
    signal: u64,
}

/// A command buffer [`MetalWorkerPool::commit`] committed, and how long encoding and committing it
/// took.
struct Committed {
    in_flight: InFlight,
    encode: std::time::Duration,
    commit: std::time::Duration,
}

/// Where one [`MetalWorkerPool::submit`] spent its time, for `SCRATCHY_METAL_TRACE`.
struct Submitted {
    encode: std::time::Duration,
    commit: std::time::Duration,
    wait: std::time::Duration,
}

impl std::fmt::Display for Submitted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            encode,
            commit,
            wait,
        } = self;
        write!(f, "encode={encode:?} commit={commit:?} wait={wait:?}")
    }
}

// `Retained<ProtocolObject<dyn MTL*>>` from objc2 isn't auto-Send/Sync
// because the protocol traits don't carry the markers; metal-rs bolted
// them on with `unsafe impl Send` on its own newtypes. The MTL retain/
// release / lifecycle ops are documented thread-safe and the worker
// pool spawns workers across threads, so we re-add the markers on the
// pool itself (and downstream worker types).
unsafe impl<W: CanonicalParams + Send + Sync> Send for MetalWorkerPool<W> {}
unsafe impl<W: CanonicalParams + Send + Sync> Sync for MetalWorkerPool<W> {}

struct PoolInner<W: CanonicalParams> {
    /// Workers ready to be handed out. `pop()` order is FIFO-ish
    /// (vec semantics: LIFO), which is fine — pool callers don't
    /// care about ordering.
    available: Vec<PooledWorker<W>>,
    /// Total workers ever created. `in_use = total_created - available.len()`.
    /// Capped at `max_workers`.
    total_created: usize,
}

/// Probe MTL4 availability once at pool construction. Logs the
/// result at info level so cold-start traces show whether the
/// upcoming Phase A side-by-side path is reachable on this host.
///
/// `newMTL4CommandQueue()` returns `Some` iff the host runs macOS 15+
/// on Apple Family 7+ silicon. The queue is dropped immediately — this
/// is a one-shot capability probe; the production queue is built lazily
/// by `ensure_mtl4` on first use.
fn probe_mtl4_availability(device: &Device) {
    let available = device.newMTL4CommandQueue().is_some();
    tracing::info!(target: "scratchy-target-metal", available, "mtl4 capability probe");
}

/// The TurboQuant decode group whose query heads per threadgroup the device picks
/// ([`TqDecodeHeads::for_group`]).
#[derive(Clone, Copy)]
pub struct TqGroup {
    pub head_dim: HeadDim,
    pub q_heads: NumQHeads,
    pub kv_heads: NumKvHeads,
}

/// A model's baked tape rungs: what a caller picks the rung its pool runs on from
/// ([`pick_rung`]) — before the pool exists (the KV pool's block-table width, the memory budget)
/// as the pool itself does.
#[derive(Clone, Copy)]
pub struct MetalRungs {
    pub buckets: &'static [MetalBucketSpec],
    pub tq: TqGroup,
}

/// The rung a pool runs on: its KV cap, which is the row width of every step's block tables, and
/// the tape each kept bucket runs.
pub struct PickedRung<'a> {
    pub cap: MaxBlocksPerSeq,
    /// The query heads one TurboQuant decode threadgroup serves on this device.
    pub tq_heads: TqDecodeHeads,
    pub tapes: Vec<(&'a MetalBucketSpec, &'a ClassedTape)>,
}

impl PickedRung<'_> {
    /// The values the picked tapes' variant-bound constants take.
    pub fn variant(&self) -> TapeVariant {
        TapeVariant {
            cap: self.cap,
            tq_heads: Some(self.tq_heads),
        }
    }
}

impl PickedRung<'_> {
    /// The bytes of each scratch buffer a worker allocates for these tapes (split-K, MoE, roped K,
    /// hd512 unfused): the largest any tape needs.
    pub fn scratch_sizes(&self) -> [u64; 4] {
        let tapes = || self.tapes.iter().map(|(_, rung)| rung.tape.scratch_bytes());
        std::array::from_fn(|i| tapes().map(|s| u64::from(s[i])).max().unwrap_or(0))
    }

    /// The bytes of the scratch buffers a worker allocates for these tapes.
    pub fn scratch_bytes(&self) -> u64 {
        self.scratch_sizes().iter().sum()
    }

    /// The worker's arena: sized for the largest activation across every bucket, AND for the
    /// largest colored slot count across buckets. Slot counts can differ per bucket: a bucket
    /// whose solver picked a fusion that needs extra scratch — e.g. the synth pre-attn /
    /// mlp-pre-down kernels, which write the updated residual to a distinct `residual_out` slot
    /// instead of in place to avoid a cross-threadgroup race — carries more colored slots than a
    /// bucket that didn't. A bucket's tape only ever references slots in `0..its own
    /// num_arena_slots`, so an arena sized to the max serves every bucket; smaller buckets simply
    /// leave the tail slots resident and idle. `arena_bytes` is elementwise-maxed over whatever
    /// slots each spec defines.
    pub fn arena_layout(&self) -> ArenaLayout {
        let num_slots = (self.tapes.iter())
            .map(|(s, _)| s.num_arena_slots as usize)
            .max()
            .unwrap_or(0);
        let mut arena_layout: ArenaLayout = vec![0u64; num_slots];
        for (spec, _) in &self.tapes {
            for (slot, &bytes) in spec.arena_bytes.iter().enumerate() {
                if bytes > arena_layout[slot] {
                    arena_layout[slot] = bytes;
                }
            }
        }
        arena_layout
    }
}

/// Pick the macro-baked rung a pool on `device` runs: the device's generation class and
/// TurboQuant decode heads, the workload's `addressing`, and the smallest KV cap rung that holds
/// `block_cap` blocks per sequence. Selection only; every fact of each tape was baked at macro
/// expansion.
///
/// The buckets are those up to `max_bucket_m`, the cap the worker derived from the memory budget
/// (at least the smallest, so decode and minimal prefill still run on a memory-starved device),
/// less any whose tape cannot exist at the rung (its scratch outgrows the 32-bit offsets its
/// kernels bind: no such rung was baked). Longer prompts chunk to the largest kept bucket via
/// `pick_bucket`.
pub fn pick_rung<'a>(
    device: &ProtocolObject<dyn MTLDevice>,
    bucket_specs: &'a [MetalBucketSpec],
    tq: TqGroup,
    max_bucket_m: Option<u32>,
    block_cap: usize,
    addressing: KvAddressing,
) -> Result<PickedRung<'a>, PoolBuildError> {
    let smallest = bucket_specs
        .iter()
        .min_by_key(|s| s.bucket_m)
        .ok_or(PoolBuildError::NoBuckets)?;
    let affords = |s: &&MetalBucketSpec| max_bucket_m.is_none_or(|cap| s.bucket_m <= cap);
    let mut buckets: Vec<&MetalBucketSpec> = bucket_specs.iter().filter(affords).collect();
    if buckets.is_empty() {
        buckets.push(smallest);
    }
    let profile = crate::device::known_profile(device)
        .ok_or_else(|| PoolBuildError::UnknownDevice(device.name().to_string()))?;
    let gen_class = crate::tape::lowered::GenClass::of(profile.generation);
    let need = MaxBlocksPerSeq(u32::try_from(block_cap).unwrap_or(u32::MAX));
    let gpu_cores = crate::device::gpu_cores(device).ok_or(PoolBuildError::UnknownGpuCores)?;
    let tq_heads = TqDecodeHeads::for_group(tq.head_dim, tq.q_heads, tq.kv_heads, gpu_cores);
    let serves = |t: &&ClassedTape| {
        t.gen_class == gen_class
            && t.addressing == addressing
            && t.tq_heads.is_none_or(|h| h == tq_heads)
    };
    let no_rung = |s: &MetalBucketSpec| PoolBuildError::NoRung {
        bucket_m: s.bucket_m,
        need,
    };
    // A bucket that baked no tape for the device is a canonical the front end refused.
    if let Some(s) = buckets.iter().find(|s| !s.tapes.iter().any(|t| serves(&t))) {
        return Err(no_rung(s));
    }
    let caps = buckets.iter().flat_map(|s| s.tapes.iter().filter(serves));
    let cap =
        (caps.map(|t| t.cap).filter(|&c| c >= need).min()).ok_or_else(|| no_rung(buckets[0]))?;
    let tapes: Vec<_> = (buckets.iter())
        .filter_map(|s| Some((*s, s.tapes.iter().filter(serves).find(|t| t.cap == cap)?)))
        .collect();
    tracing::info!(
        target: "scratchy-target-metal",
        gpu_cores = gpu_cores.get(),
        tq_decode_heads = tq_heads.get(),
        kv_cap_rung = cap.get(),
        buckets = tapes.len(),
        "baked tape rung"
    );
    Ok(PickedRung {
        cap,
        tq_heads,
        tapes,
    })
}

impl MetalRungs {
    /// The rungs `ScratchyWeights::metal_rungs` hands out (typed there as `Any`: the trait cannot
    /// name this crate).
    pub fn of(
        baked: &'static (dyn std::any::Any + Send + Sync),
    ) -> Result<&'static Self, PoolBuildError> {
        baked.downcast_ref().ok_or(PoolBuildError::NotMetalRungs)
    }

    /// [`pick_rung`] over this model's buckets.
    pub fn pick(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        max_bucket_m: Option<u32>,
        block_cap: usize,
        addressing: KvAddressing,
    ) -> Result<PickedRung<'static>, PoolBuildError> {
        pick_rung(
            device,
            self.buckets,
            self.tq,
            max_bucket_m,
            block_cap,
            addressing,
        )
    }
}

impl<W: CanonicalParams> MetalWorkerPool<W> {
    /// Build the pool and eagerly create the first worker.
    ///
    /// Eager creation surfaces allocation/recording/pipeline-lookup
    /// failures at construction time and warms the first-forward
    /// path (no creation cost on the first checkout). The first worker places its
    /// buffers in `lent` ([`LentActivation`]); any later one allocates its own.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: Arc<Device>,
        weights: &W,
        allocator: Arc<MetalAllocator>,
        pipelines: Arc<SpecializedPipelines>,
        bucket_tapes: Arc<[LoweredMetalTape]>,
        block_table_stride: MaxBlocksPerSeq,
        arena_layout: ArenaLayout,
        runtime_factory: RuntimeFactory,
        max_workers: usize,
        lent: LentActivation,
    ) -> Result<Self, WorkerError>
    where
        W: ModelSources,
    {
        assert!(max_workers >= 1, "max_workers must be >= 1");
        let sources = ResolvedSources::resolve(weights, &allocator, &bucket_tapes)?;

        // The shared `MetalResidencySet` lives on the allocator now —
        // arena buffers are pinned automatically as `push_arena_locked`
        // runs (in `MetalAllocator`), so the pool no longer manages
        // residency creation or arena-hook wiring. Per-worker arena
        // buffers (allocated outside the allocator, in `MetalWorker::new`)
        // still need explicit insertion; that happens in `spawn_worker`
        // below by reading `allocator.residency()`.

        // MTL4 probe. Side-effect-free: tries `device.newMTL4CommandQueue()`, logs
        // availability, drops the queue. Result is recomputed cheaply
        // when the side-by-side path (A.2/A.3) consults `self.mtl4` — kept
        // out of the pool struct until the hot path actually uses it.
        probe_mtl4_availability(&device);

        let pool = Self {
            device,
            allocator,
            pipelines,
            bucket_tapes,
            block_table_stride,
            sources,
            arena_layout: Arc::new(arena_layout),
            runtime_factory,
            max_workers,
            residency_attached: std::sync::atomic::AtomicBool::new(false),
            mtl4: Mutex::new(None),
            inner: Mutex::new(PoolInner {
                available: Vec::new(),
                total_created: 0,
            }),
            cv: Condvar::new(),
        };
        let first = pool.spawn_worker(&lent)?;
        {
            let mut inner = pool.inner.lock().unwrap();
            inner.total_created = 1;
            inner.available.push(first);
        }
        Ok(pool)
    }

    /// Build the pool from a flat `&[MetalBucketSpec]` plus the
    /// loaded model + its allocator + arena layout + runtime factory.
    ///
    /// This is the runtime-side prerequisite the `#[forward]` macro's
    /// emitted `metal_pool(...)` calls into. The macro materializes
    /// the static slices that back the `bucket_specs` and threads the
    /// loaded `Weights` + the `MetalAllocator` that owns its
    /// `MTLBuffer` arenas; the caller is responsible for the device,
    /// the arena byte-layout, the runtime factory, and the worker
    /// cap (typically derived from
    /// `floor((device_total - weights - misc) / per_worker_arena)`).
    pub fn for_buckets(
        device: Arc<Device>,
        weights: &W,
        allocator: Arc<MetalAllocator>,
        bucket_specs: &[MetalBucketSpec],
        runtime_factory: RuntimeFactory,
        max_workers: usize,
        max_bucket_m: Option<u32>,
        // Runtime per-sequence block-table capacity
        // (`KvCachePool::max_blocks_per_seq`), a KV cap rung itself (see
        // [`pick_rung`]): every step's block tables are rows of this width.
        block_cap: usize,
        addressing: KvAddressing,
        // Buffers of a pool whose forwards never overlap this one's, for the first worker to place
        // its own in ([`LentActivation`]).
        lent: LentActivation,
    ) -> Result<Self, PoolBuildError>
    where
        W: ModelSources,
    {
        let tq = TqGroup {
            head_dim: HeadDim(W::GLOBAL_HEAD_DIM),
            q_heads: NumQHeads(W::NUM_Q_HEADS),
            kv_heads: NumKvHeads(W::NUM_GLOBAL_KV_HEADS),
        };
        let rung = pick_rung(
            &device,
            bucket_specs,
            tq,
            max_bucket_m,
            block_cap,
            addressing,
        )?;
        let variant = rung.variant();
        let arena_layout = rung.arena_layout();
        let picked = rung.tapes;

        let cache = SpecializedPipelineCache::new((*device).clone(), &[])
            .map_err(|e| PoolBuildError::PipelineCacheBuild(format!("{e:?}")))?;
        let mut tables: Vec<&[crate::tape::lowered::BakedKernel]> = Vec::new();
        for (spec, _) in &picked {
            if !tables.iter().any(|t| std::ptr::eq(*t, spec.kernels)) {
                tables.push(spec.kernels);
                cache.register_baked(spec.kernels);
            }
        }
        let tapes: Vec<LoweredMetalTape> = picked.iter().map(|(_, rung)| rung.tape).collect();
        let bucket_tapes: Arc<[LoweredMetalTape]> = Arc::from(tapes);
        let pipelines = Arc::new(SpecializedPipelines::new(Arc::new(cache), variant));

        Self::new(
            device,
            weights,
            allocator,
            pipelines,
            bucket_tapes,
            variant.cap,
            arena_layout,
            runtime_factory,
            max_workers,
            lent,
        )
        .map_err(PoolBuildError::Worker)
    }

    pub fn max_workers(&self) -> usize {
        self.max_workers
    }

    /// The Metal device the pool's workers were allocated on. Callers
    /// use it to spin up a `CommandQueue` for [`Self::forward`] (the
    /// pool intentionally doesn't own the queue — the engine may
    /// share one across multiple pools / streams).
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Total workers currently allocated by the pool (whether or
    /// not they're checked out). Monotonically grows up to
    /// `max_workers`.
    pub fn current_size(&self) -> usize {
        self.inner.lock().unwrap().total_created
    }

    /// Number of workers currently sitting in the available queue.
    /// Test-facing — production callers don't need this.
    pub fn available(&self) -> usize {
        self.inner.lock().unwrap().available.len()
    }

    /// Block until a worker is available, then return a RAII guard.
    ///
    /// Three paths:
    /// 1. A worker is already idle → pop and return.
    /// 2. The pool is below cap → reserve a slot, drop the lock,
    ///    spawn (allocates GPU memory + records dispatchs), return.
    /// 3. The pool is at cap and all workers are busy → wait on the
    ///    condvar until a peer thread checks one back in.
    ///
    /// Spawn failures release the reserved slot and are propagated.
    pub fn checkout(&self) -> Result<WorkerGuard<'_, W>, WorkerError> {
        let mut inner = self.inner.lock().unwrap();
        loop {
            if let Some(pooled) = inner.available.pop() {
                return Ok(WorkerGuard {
                    pool: self,
                    inner: Some(pooled),
                });
            }
            if inner.total_created < self.max_workers {
                inner.total_created += 1;
                drop(inner);
                match self.spawn_worker(&LentActivation::default()) {
                    Ok(pooled) => {
                        return Ok(WorkerGuard {
                            pool: self,
                            inner: Some(pooled),
                        });
                    }
                    Err(e) => {
                        let mut inner = self.inner.lock().unwrap();
                        inner.total_created -= 1;
                        // A peer might be waiting on capacity that we
                        // just released by failing to spawn. Notify so
                        // they can re-check (they'll either find a
                        // freed worker or hit the same spawn path).
                        self.cv.notify_one();
                        return Err(e);
                    }
                }
            }
            inner = self.cv.wait(inner).unwrap();
        }
    }

    /// Non-blocking checkout. Returns `None` when every worker is
    /// busy *and* the pool is at cap.
    ///
    /// Spawn failures inside `try_checkout` produce `Some(Err(...))`
    /// so callers can distinguish "pool is full" (`None`) from
    /// "GPU allocation failed" (`Some(Err)`).
    pub fn try_checkout(&self) -> Option<Result<WorkerGuard<'_, W>, WorkerError>> {
        let mut inner = self.inner.lock().unwrap();
        if let Some(pooled) = inner.available.pop() {
            return Some(Ok(WorkerGuard {
                pool: self,
                inner: Some(pooled),
            }));
        }
        if inner.total_created < self.max_workers {
            inner.total_created += 1;
            drop(inner);
            match self.spawn_worker(&LentActivation::default()) {
                Ok(pooled) => Some(Ok(WorkerGuard {
                    pool: self,
                    inner: Some(pooled),
                })),
                Err(e) => {
                    let mut inner = self.inner.lock().unwrap();
                    inner.total_created -= 1;
                    self.cv.notify_one();
                    Some(Err(e))
                }
            }
        } else {
            None
        }
    }

    /// Map a real `num_tokens` to the bucket index that should run.
    ///
    /// Picks the smallest bucket whose `bucket_m >= num_tokens`. Tape
    /// ordering inside `bucket_tapes` is intentionally not assumed —
    /// the macro / model loader supplies whatever order it likes
    /// (typically ascending), and a linear scan over a handful of
    /// buckets is cheaper than maintaining a sorted invariant.
    ///
    /// **Safe-bucket floor:** multi-row buckets are only selected from
    /// `SAFE_MULTI_ROW_BUCKET_M` up. The historical reason — the fused
    /// MLP / MLX-steel kernels' BM=32 tile overrunning arenas sized for
    /// `bucket_m < 8` — is stale: the steel kernel has carried M-tail
    /// guards since its initial commit and the arenas are
    /// elementwise-maxed across buckets. Validated at 2–7 concurrent
    /// sequences on Llama-3.2-1B (dense fused-MLP) and gemma-4-26b-a4b
    /// (MoE, qmv_wide) with greedy-output checks; the floor now exists
    /// to keep `num_tokens=2..7` off the (correct but slow) M=8
    /// grouped-MoE bucket's padded compute only when no smaller bucket
    /// fits.
    pub fn pick_bucket(&self, num_tokens: u32) -> Result<usize, ForwardError> {
        const SAFE_MULTI_ROW_BUCKET_M: u32 = 2;
        if num_tokens == 0 {
            return Err(ForwardError::ZeroTokens);
        }
        let effective_min = if num_tokens == 1 {
            1
        } else {
            num_tokens.max(SAFE_MULTI_ROW_BUCKET_M)
        };
        let mut best: Option<(usize, u32)> = None;
        let mut max_bucket: u32 = 0;
        for (i, tape) in self.bucket_tapes.iter().enumerate() {
            if tape.bucket_m > max_bucket {
                max_bucket = tape.bucket_m;
            }
            if tape.bucket_m >= effective_min {
                best = match best {
                    Some((_, bm)) if bm <= tape.bucket_m => best,
                    _ => Some((i, tape.bucket_m)),
                };
            }
        }
        best.map(|(i, _)| i).ok_or(ForwardError::NoBucketFits {
            num_tokens,
            max_bucket,
        })
    }

    /// Lazily build the pool's MTL4 surface (queue + allocator +
    /// shared event). Panics if MTL4 is unavailable on this host
    /// (requires macOS 15+ on Apple Family 7+).
    fn ensure_mtl4(&self) {
        let mut slot = self.mtl4.lock().expect("mtl4 mutex");
        if slot.is_some() {
            return;
        }
        let queue = self.device.newMTL4CommandQueue().expect(
            "device.newMTL4CommandQueue() returned nil \
             — host does not support MTL4 (requires macOS 15+ on Apple Family 7+)",
        );
        let ring = std::array::from_fn(|_| RingSlot {
            allocator: self
                .device
                .newCommandAllocator()
                .expect("device.newCommandAllocator() returned nil"),
            staging: None,
            signal: 0,
        });
        let shared_event = self
            .device
            .newSharedEvent()
            .expect("device.newSharedEvent() returned nil");
        *slot = Some(Mtl4Pool {
            queue,
            ring,
            next: 0,
            shared_event,
            signal_counter: 0,
        });
    }

    /// MTL4 forward dispatch + optional encoder-tail hook.
    ///
    /// When `tail = Some(f)`, after the worker has encoded the bucket's
    /// dispatches and BEFORE the encoder is ended, the pool calls
    /// `f(&encoder)` so the caller can append additional dispatches
    /// (e.g. argmax) onto the same compute encoder. Forward + tail
    /// share one CB, one commit, and one host wait — no separate
    /// queue or shared event needed. A `deferred` forward is not waited for:
    /// its command buffer is recorded on the [`Deferral`].
    fn run_bucket_mtl4_with_tail<F>(
        &self,
        worker: &MetalWorker<W>,
        bucket_idx: usize,
        num_tokens: usize,
        num_seqs: u32,
        has_spec_tokens: bool,
        tail: Option<F>,
        writes: &InputWrites,
        deferred: Option<&Deferral>,
    ) -> Result<(), ForwardError>
    where
        F: FnOnce(
            &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
            &MetalWorker<W>,
            usize,
        ) -> Result<(), ForwardError>,
    {
        self.ensure_mtl4();
        // gemma3-mm SigLIP vision tower: the projector tail
        // (`AvgPool2d -> soft_emb_norm -> mm_input_projection`) hits an
        // in-command-buffer write→read coherence failure at the
        // 4096-patch scale — the projector gemm reads the soft-emb-norm
        // output as all-zero (→ silent all-zero vision embeds → garbled
        // text) unless a CB boundary (commit + host-wait) separates the
        // writer from the reader. No in-CB barrier fixes it (None,
        // Device, or forced-on-every-dispatch all fail); only the CB
        // boundary does. So run the WHOLE vision bucket as serialized
        // single-dispatch segments (each its own CB + host-wait), the
        // proven-correct execution from the dump replay. This is a
        // once-per-image prefill path (vision towers have no argmax
        // tail), so the per-segment host-wait overhead (~0.5 ms ×
        // dispatches) is immaterial against the multi-second tower.
        // The AvgPool2d-tape auto-trigger runs the segment at K=1.
        let needs_serialized = worker.bucket_has_avg_pool_2d(bucket_idx);
        if needs_serialized {
            assert!(
                deferred.is_none(),
                "a serialized forward waits for each of its command buffers"
            );
            assert!(
                tail.is_none(),
                "serialized chunked forward does not support a tail hook \
                 (vision towers have no argmax tail)",
            );
            let k = 1;
            let total = worker.count_dispatches(bucket_idx);
            let mut start = 0usize;
            while start < total {
                let end = (start + k).min(total);
                self.run_dump_segment(
                    worker,
                    bucket_idx,
                    num_tokens,
                    num_seqs,
                    has_spec_tokens,
                    start..end,
                    (start == 0).then_some(writes),
                )?;
                start = end;
            }
            return Ok(());
        }
        let trace = std::env::var_os("SCRATCHY_METAL_TRACE").is_some();
        let encode = |enc: &::objc2::runtime::ProtocolObject<
            dyn ::objc2_metal::MTL4ComputeCommandEncoder,
        >| {
            worker
                .run_bucket_mtl4(
                    bucket_idx,
                    num_tokens as u32,
                    num_seqs,
                    has_spec_tokens,
                    enc,
                )
                .map_err(ForwardError::Worker)?;
            // Caller-supplied encoder-tail hook (e.g. argmax dispatch)
            // runs on the SAME MTL4 compute encoder as the forward —
            // forward + tail share one CB, one commit, one host wait.
            if let Some(t) = tail {
                t(enc, worker, bucket_idx)?;
            }
            Ok(())
        };
        if let Some(deferral) = deferred {
            let ((), committed) = self.commit(Some(writes), encode)?;
            deferral.committed(committed.in_flight);
            return Ok(());
        }
        let ((), took) = self.submit(Some(writes), encode)?;
        if trace {
            let dispatches = worker.count_dispatches(bucket_idx);
            eprintln!(
                "[forward bucket={bucket_idx} num_tokens={num_tokens} mtl4 dispatches={dispatches}] {took}"
            );
        }
        Ok(())
    }

    /// ONE command buffer on the pool's MTL4 queue, returned once the GPU is done with it
    /// ([`wait_drained`]) — completed, or failed with the error the commit feedback reports — so
    /// nothing it reads is still in use when the caller gets control back. See [`Self::commit`].
    ///
    /// [`wait_drained`]: crate::mtl4_dispatch::wait_drained
    fn submit<R>(
        &self,
        writes: Option<&InputWrites>,
        encode: impl FnOnce(
            &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
        ) -> Result<R, ForwardError>,
    ) -> Result<(R, Submitted), ForwardError> {
        let t_pre = std::time::Instant::now();
        let (encoded, committed) = self.commit(writes, encode)?;
        committed.in_flight.wait()?;
        Ok((
            encoded,
            Submitted {
                encode: committed.encode,
                commit: committed.commit,
                wait: t_pre.elapsed() - committed.encode - committed.commit,
            },
        ))
    }

    /// ONE command buffer on the pool's MTL4 queue, committed without waiting for it. `encode`
    /// records onto its compute encoder, the pool's residency sets declared on it, after the
    /// forward's input `writes`; if it fails, its error returns with the command buffer ended and
    /// nothing committed. It starts once every command buffer committed before it is done.
    fn commit<R>(
        &self,
        writes: Option<&InputWrites>,
        encode: impl FnOnce(
            &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
        ) -> Result<R, ForwardError>,
    ) -> Result<(R, Committed), ForwardError> {
        use objc2::runtime::AnyObject;
        use objc2_metal::{
            MTL4ComputeCommandEncoder, MTL4VisibilityOptions, MTLSharedEvent, MTLStages,
        };
        use std::ptr::NonNull;
        let t_pre = std::time::Instant::now();
        let cb = self
            .device
            .newCommandBuffer()
            .expect("newCommandBuffer returned nil");
        let (encoded, t_encoded, in_flight) = {
            let mut slot = self.mtl4.lock().expect("mtl4 mutex");
            let mtl4 = slot.as_mut().expect("ensure_mtl4 succeeded");
            let taken = mtl4.next;
            mtl4.next = (taken + 1) % mtl4.ring.len();
            let ring = &mut mtl4.ring[taken];
            // Its allocator and staging are reused only once its last command buffer is done.
            if mtl4.shared_event.signaledValue() < ring.signal {
                crate::mtl4_dispatch::wait_drained(&mtl4.shared_event, ring.signal);
            }
            ring.allocator.reset();
            cb.beginCommandBufferWithAllocator(&ring.allocator);
            let staged = writes.map_or(&[][..], |w| &w.staged[..]);
            if !staged.is_empty() {
                if ring
                    .staging
                    .as_ref()
                    .is_none_or(|b| b.length() < staged.len())
                {
                    let buffer = crate::mtl4_dispatch::shared_zeroed(&self.device, staged.len());
                    ring.staging = Some(self.allocator.residency().pin(buffer));
                    self.allocator.residency().commit();
                }
                let staging = ring.staging.as_ref().expect("sized above");
                // SAFETY: `staging` is shared storage of at least `staged.len()` bytes, and no
                // command buffer reads it: the last one that did is done.
                unsafe {
                    let dst = staging.contents().as_ptr().cast::<u8>();
                    std::ptr::copy_nonoverlapping(staged.as_ptr(), dst, staged.len());
                }
            }
            // Residency: MTL4 cmdbufs declare per-cmdbuf.
            // Reuse the same set as the pool (weights + arenas + KV cache).
            let cb_ptr: *mut AnyObject =
                ::objc2::rc::Retained::as_ptr(&cb) as *const AnyObject as *mut AnyObject;
            unsafe {
                self.allocator
                    .residency()
                    .attach_to_mtl4_command_buffer(cb_ptr);
                // Also declare the weights set. A distinct un-wired set by
                // default, where this call is what keeps weights resident for
                // the duration of the forward; the same object as residency()
                // under WeightResidency::Wired (idempotent re-attach).
                self.allocator
                    .weights_residency()
                    .attach_to_mtl4_command_buffer(cb_ptr);
            }
            let enc = cb
                .computeCommandEncoder()
                .expect("MTL4 computeCommandEncoder returned nil");
            let ops = writes.map_or(&[][..], |w| &w.ops[..]);
            let barrier = |before| {
                enc.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                    MTLStages::Blit,
                    before,
                    MTL4VisibilityOptions::Device,
                )
            };
            let mut earlier_writes = true;
            for op in ops {
                // SAFETY: every range is inside its buffers, checked when it was recorded.
                unsafe {
                    match op {
                        InputWrite::Staged { to, at, len } => {
                            let staging = ring.staging.as_ref().expect("filled above");
                            enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                                staging, *at, to, 0, *len,
                            );
                        }
                        InputWrite::Fill { to, range, byte } => {
                            enc.fillBuffer_range_value(to, (*range).clone().into(), *byte);
                        }
                        InputWrite::Device {
                            src,
                            offset,
                            to,
                            at,
                        } => {
                            // It overwrites the placeholder a staged write put there.
                            if std::mem::take(&mut earlier_writes) {
                                barrier(MTLStages::Blit);
                            }
                            enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                                src,
                                *offset,
                                to,
                                *at,
                                size_of::<u32>(),
                            );
                        }
                    }
                }
            }
            if !ops.is_empty() {
                barrier(MTLStages::Dispatch);
            }
            let encoded = encode(&enc);
            enc.endEncoding();
            cb.endCommandBuffer();
            // Propagate an encode error AFTER the encoder/CB have been ended
            // (so allocator state stays consistent) and BEFORE committing
            // any half-encoded work to the GPU.
            let encoded = encoded?;
            mtl4.signal_counter = mtl4.signal_counter.checked_add(1).expect("event overflow");
            mtl4.ring[taken].signal = mtl4.signal_counter;
            let t_encoded = t_pre.elapsed();
            // MTL4 does not order command buffers on a queue: one still running when this is
            // committed holds it back (KV it appends, the token it samples, the inputs it reads).
            // Committed under the lock, so command buffers reach the queue in signal order.
            let (queue, event) = (&mtl4.queue, &mtl4.shared_event);
            let before = mtl4.signal_counter - 1;
            if event.signaledValue() < before {
                queue.waitForEvent_value(ProtocolObject::from_ref(&**event), before);
            }
            let error = std::sync::Arc::new(std::sync::Mutex::new(None));
            let feedback = commit_options(&error);
            let cb_protocol: &ProtocolObject<dyn ::objc2_metal::MTL4CommandBuffer> = &cb;
            let mut cb_array = [NonNull::from(cb_protocol)];
            unsafe {
                queue.commit_count_options(NonNull::from(&mut cb_array[0]), 1, &feedback);
            }
            // Signal AFTER the cmdbuf so the wait fires only once GPU work
            // is fully drained.
            queue.signalEvent_value(ProtocolObject::from_ref(&**event), mtl4.signal_counter);
            (
                encoded,
                t_encoded,
                InFlight {
                    event: event.clone(),
                    value: mtl4.signal_counter,
                    error,
                    _feedback: feedback,
                },
            )
        };
        Ok((
            encoded,
            Committed {
                in_flight,
                encode: t_encoded,
                commit: t_pre.elapsed() - t_encoded,
            },
        ))
    }

    /// One activation-dump replay segment: encode flat dispatch
    /// indices `range` of the bucket's baked tape on a fresh MTL4 CB,
    /// commit, and host-wait.
    fn run_dump_segment(
        &self,
        worker: &MetalWorker<W>,
        bucket_idx: usize,
        num_tokens: usize,
        num_seqs: u32,
        has_spec_tokens: bool,
        range: std::ops::Range<usize>,
        writes: Option<&InputWrites>,
    ) -> Result<(), ForwardError> {
        self.submit(writes, |enc| {
            worker
                .run_bucket_mtl4_range(
                    bucket_idx,
                    num_tokens as u32,
                    num_seqs,
                    has_spec_tokens,
                    enc,
                    range,
                )
                .map_err(ForwardError::Worker)
        })?;
        Ok(())
    }

    /// Phase 6 chain-driver primitive. Opens ONE MTL4 command buffer
    /// on the pool's internal MTL4 queue and invokes `body` with the
    /// checked-out worker, its `RuntimeBindings`, and the live compute
    /// encoder. The body drives N forward dispatches (typically the
    /// K-step draft chain) plus any caller-supplied dispatches
    /// (e.g. `argmax_dual_write`, `chain_advance`) onto the same
    /// encoder.
    ///
    /// The pool owns:
    ///   * worker checkout / checkin (via the `WorkerGuard` drop),
    ///   * iter-0 input upload (via `write_runtime_inputs`),
    ///   * residency attach (idempotent),
    ///   * MTL4 CB begin/end,
    ///   * commit, signal, host wait,
    ///   * allocator reset.
    ///
    /// One CB ⇒ one commit ⇒ one host wait for the entire chain.
    /// The residency set is committed on the first call (same lazy
    /// pattern as `forward_with_tail`) and declared per-CB via
    /// `useResidencySet:`; the commit runs on the pool's MTL4 queue.
    pub fn with_chain_encoder<F, R>(
        &self,
        inputs: &ForwardInputs<'_>,
        body: F,
    ) -> Result<R, ForwardError>
    where
        F: FnOnce(
            &MetalWorker<W>,
            &RuntimeBindings,
            &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
        ) -> Result<R, ForwardError>,
    {
        self.ensure_mtl4();
        let trace = std::env::var_os("SCRATCHY_METAL_TRACE").is_some();

        if !self
            .residency_attached
            .swap(true, std::sync::atomic::Ordering::AcqRel)
        {
            self.allocator.residency().commit();
            // Commit the weights set too. When it is a distinct un-wired set
            // (default), its inserts — including weight-memcpy arenas added at
            // load — take effect only on commit; without this the un-wired
            // weight buffers are not resident and the GPU reads garbage.
            // No-op when it aliases the wired set (WeightResidency::Wired).
            self.allocator.weights_residency().commit();
        }

        let guard = self.checkout()?;
        let writes = begin_step(&guard, inputs, self.block_table_stride)?;

        // Caller's body encodes the entire chain onto the encoder.
        let (body_result, took) = self.submit(Some(&writes), |enc| {
            body(&guard.worker, &guard.runtime, enc)
        })?;
        if trace {
            eprintln!("[chain encoder mtl4] {took}");
        }
        Ok(body_result)
    }

    /// Run one forward step via MTL4.
    ///
    /// Pipeline:
    ///  1. Pick the bucket from `inputs.num_tokens`.
    ///  2. Check out a worker (eagerly grow the pool if below cap;
    ///     block if at cap).
    ///  3. Validate every present input slice against its runtime
    ///     buffer's capacity; copy bytes into the buffer's `contents()`.
    ///  4. Encode the bucket's MTL4 steps, commit, and wait.
    ///  5. Run `with_output(&worker)` so the caller can read arena
    ///     buffers (e.g. logits) before the worker is checked back in.
    ///  6. Drop the guard — the worker returns to the pool.
    ///
    /// All validation runs before any GPU work is submitted.
    pub fn forward<R>(
        &self,
        inputs: &ForwardInputs<'_>,
        with_output: impl FnOnce(&MetalWorker<W>, usize) -> R,
    ) -> Result<R, ForwardError> {
        self.forward_with_tail::<R, fn(
            &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
            &MetalWorker<W>,
            usize,
        ) -> Result<(), ForwardError>>(inputs, with_output, None)
    }

    /// Same as [`forward`], but takes an optional encoder-tail hook
    /// invoked on the same MTL4 compute encoder as the forward,
    /// AFTER the bucket's dispatches and BEFORE `endEncoding`. Lets
    /// callers append additional dispatches (e.g. argmax sampling)
    /// onto the same CB so the whole step lives in one command
    /// buffer with one commit and one host wait.
    pub fn forward_with_tail<R, F>(
        &self,
        inputs: &ForwardInputs<'_>,
        with_output: impl FnOnce(&MetalWorker<W>, usize) -> R,
        tail: Option<F>,
    ) -> Result<R, ForwardError>
    where
        F: FnOnce(
            &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
            &MetalWorker<W>,
            usize,
        ) -> Result<(), ForwardError>,
    {
        let bucket_idx = self.pick_bucket(inputs.num_tokens)?;
        // num_seqs = number of sequences packed into this forward.
        // Computed once here from the staged `cu_seqlens_q` slice
        // (`batch + 1` entries) and threaded through the dispatch
        // path so the per-sub-dispatch `RuntimeGate` checks can
        // pick the lm_head slice (single-seq) vs the full
        // M=bucket_m fallback (multi-seq). Defaults to 1 when
        // `cu_seqlens_q` is absent — those are the
        // `Instruction::AttentionViaCache` decode buckets that
        // always run a single-token forward, never the slice.
        let num_seqs: u32 = inputs
            .cu_seqlens_q
            .map(|cu| (cu.len().saturating_sub(1)).max(1) as u32)
            .unwrap_or(1);
        let has_spec_tokens = inputs.has_spec_tokens;

        // Lazily commit the allocator's residency set on the first
        // forward — gated with an atomic bool so the one-time commit
        // isn't repeated across thousands of forwards. MTL4 command
        // buffers declare the set per-CB via `useResidencySet:`.
        //
        // commit() applies any inserts queued by `register_mmap` /
        // arena push / `gpu_worker::initialize_cache`'s KV-cache
        // wiring (~125ms on Llama-3.2-3B's 4.7 GB KV pool). Done
        // here rather than in init_cache so it overlaps with the
        // pool build / first-dispatch encoding instead of blocking
        // the engine init path.
        if !self
            .residency_attached
            .swap(true, std::sync::atomic::Ordering::AcqRel)
        {
            self.allocator.residency().commit();
            // Commit the weights set too. When it is a distinct un-wired set
            // (default), its inserts — including weight-memcpy arenas added at
            // load — take effect only on commit; without this the un-wired
            // weight buffers are not resident and the GPU reads garbage.
            // No-op when it aliases the wired set (WeightResidency::Wired).
            self.allocator.weights_residency().commit();
        }

        let guard = self.checkout()?;
        let writes = begin_step(&guard, inputs, self.block_table_stride)?;

        // All execution goes through the MTL4 path. (The opt-in MTL3
        // dispatch path was removed — it only ever ran the all-dispatch
        // buckets that MTL4 already handles, and its M4-era latency edge
        // no longer applies.) A bucket has no MTL4 plan
        // (`mtl4_steps == None`) only if a kernel exceeds the 31-entry
        // argument-table bind cap.
        assert!(
            guard.worker.bucket_bakings[bucket_idx].mtl4_steps.is_some(),
            "bucket {} is not MTL4-eligible (a kernel exceeds the 31-binding \
             argument-table cap)",
            bucket_idx,
        );
        self.run_bucket_mtl4_with_tail(
            &guard.worker,
            bucket_idx,
            inputs.num_tokens as usize,
            num_seqs,
            has_spec_tokens,
            tail,
            &writes,
            inputs.deferred,
        )?;

        // DIAGNOSTIC: dump non-zero counts for each arena slot. Tells
        // us where in the chain values transition from real to zero.
        if std::env::var_os("VLLM_DUMP_ARENA").is_some() {
            for slot in 0..guard.worker.arena.len() {
                let buf = &guard.worker.arena[slot];
                let len_bytes = buf.length();
                let row0 = unsafe {
                    std::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, len_bytes)
                };
                let nonzero_bytes = row0.iter().filter(|&&v| v != 0).count();
                eprintln!(
                    "[diag-arena] slot={:3} bytes={} nonzero_bytes={}/{}",
                    slot, len_bytes, nonzero_bytes, len_bytes,
                );
            }
        }

        // DIAGNOSTIC: dump first/last 4 bf16 values of every arena
        // slot. Used to bisect where forward N's outputs diverge from
        // forward N+1's across runs. Reads bytes directly as bf16.
        if std::env::var_os("VLLM_DUMP_ARENA_BF16").is_some() {
            fn bf16_bits_to_f32(bits: u16) -> f32 {
                f32::from_bits((bits as u32) << 16)
            }
            for slot in 0..guard.worker.arena.len() {
                let buf = &guard.worker.arena[slot];
                let len_bytes = buf.length();
                if len_bytes < 32 {
                    continue;
                }
                let bytes = unsafe {
                    std::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, len_bytes)
                };
                let head = (0..16)
                    .map(|j| {
                        let off = j * 2;
                        let bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                        format!("{:.4}", bf16_bits_to_f32(bits))
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                eprintln!("[diag-bf16] slot={:3} head=[{}]", slot, head);
            }
        }

        Ok(with_output(&guard.worker, bucket_idx))
    }

    fn spawn_worker(&self, lent: &LentActivation) -> Result<PooledWorker<W>, WorkerError> {
        let runtime = self.runtime_factory.build(&self.device, lent);
        let worker = MetalWorker::<W>::new_with_residency(
            self.device.clone(),
            &self.arena_layout,
            &self.bucket_tapes,
            &self.pipelines,
            &self.sources,
            &runtime,
            Some(self.allocator.residency()),
            lent,
        )?;
        Ok(PooledWorker { worker, runtime })
    }

    /// An idle worker's arena slots, scratch and TurboQuant scratch, to lend a pool whose forwards
    /// never overlap this one's ([`LentActivation`]); none while every worker is out.
    pub fn lend_activation(&self) -> LentActivation {
        let inner = self.inner.lock().unwrap();
        (inner.available.first()).map_or_else(LentActivation::default, |p| LentActivation {
            tq_scratch: p.runtime.tq.as_ref().map(|t| t.scratch.clone()),
            ..p.worker.activation()
        })
    }

    fn checkin(&self, pooled: PooledWorker<W>) {
        let mut inner = self.inner.lock().unwrap();
        inner.available.push(pooled);
        // Wake exactly one waiter — at most one of them can claim
        // this worker, the others stay blocked.
        self.cv.notify_one();
    }
}

/// Commit options for ONE command buffer, whose feedback handler records that command buffer's GPU
/// execution error (e.g. kIOGPUCommandBufferCallbackErrorOutOfMemory) in `error`. It is checked after
/// the wait — a silently-failed command buffer otherwise produces all-zero outputs and degenerate
/// logits (the macOS 26.5.1 Qwen3.5-MoE "!!!!" failure mode) — and a slot of its own means a later
/// command buffer's error can neither be taken for it nor consumed by it.
fn commit_options(
    error: &std::sync::Arc<std::sync::Mutex<Option<String>>>,
) -> objc2::rc::Retained<objc2_metal::MTL4CommitOptions> {
    let options = objc2_metal::MTL4CommitOptions::new();
    let err_slot = std::sync::Arc::clone(error);
    let block = block2::RcBlock::new(
        move |feedback: std::ptr::NonNull<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4CommitFeedback>,
        >| {
            use objc2_metal::MTL4CommitFeedback as _;
            let fb = unsafe { feedback.as_ref() };
            if let Some(e) = fb.error() {
                let msg = format!("{e}");
                eprintln!("[scratchy-target-metal] GPU COMMIT ERROR: {msg}");
                // Dump the FULL NSError — code/domain/userInfo. MTL
                // folds the faulting-encoder label + GPU fault info
                // into userInfo (MTLCommandBufferEncoderInfoErrorKey),
                // which the bare Display ("...error 1.") drops. This
                // is the only signal that names WHICH dispatch faulted.
                eprintln!(
                    "[scratchy-target-metal] GPU COMMIT ERROR code={} domain={}",
                    e.code(),
                    e.domain(),
                );
                // userInfo carries NSUnderlyingError /
                // NSMultipleUnderlyingErrorsKey whose NESTED NSError
                // userInfo holds the real per-encoder fault reason.
                // Debug prints only pointers; NSObject `description`
                // recurses and renders the whole tree.
                {
                    use objc2::runtime::AnyObject;
                    let ui = e.userInfo();
                    let ui_obj: &AnyObject = &ui;
                    let desc: objc2::rc::Retained<objc2_foundation::NSString> =
                        unsafe { objc2::msg_send![ui_obj, description] };
                    eprintln!("[scratchy-target-metal] GPU COMMIT ERROR userInfo: {desc}");
                }
                *err_slot.lock().expect("commit_error mutex") = Some(msg);
            }
        },
    );
    // The options object retains the handler block ("references your commit feedback handler
    // after you add it"); the in-flight command buffer keeps the options until it is done.
    unsafe { options.addFeedbackHandler(block2::RcBlock::as_ptr(&block) as _) };
    options
}

/// TurboQuant full-context dequant grid width = the host block-table ROW
/// STRIDE for this forward. The serving worker flattens the block table to
/// `[num_reqs * max_blocks_eff]` (gpu_worker.rs), with
/// `max_blocks_eff = max(W::MAX_BLOCKS_PER_SEQ, runtime_max_blocks)`, so the
/// stride recovers as `len / num_seqs`. The `TqStageRotated` dispatch must
/// cover exactly this many blocks so the reused fp16 scratch is filled for the
/// WHOLE active context (the kernel early-exits past `seqused_k`); the static
/// `W::MAX_BLOCKS_PER_SEQ` truncated it at 2048 tokens for uniform arches.
/// Returns 0 when there is no block table (decode-via-cache buckets etc.) — the
/// dispatch then falls back to the baked const.
/// Record the step's writes — into the worker's runtime buffers, its deferral's other buffers,
/// then the tokens an earlier forward wrote on the device — and set the per-step values its
/// dispatch reads.
fn begin_step<W: CanonicalParams>(
    worker: &PooledWorker<W>,
    inputs: &ForwardInputs<'_>,
    block_table_stride: MaxBlocksPerSeq,
) -> Result<InputWrites, ForwardError> {
    use std::sync::atomic::Ordering::Relaxed;
    let mut writes = InputWrites::default();
    write_runtime_inputs(&worker.runtime, inputs, block_table_stride, &mut writes)?;
    for (to, bytes) in inputs.deferred.iter().flat_map(|d| &d.host_writes) {
        writes.stage(to, bytes);
    }
    for input in inputs.deferred.iter().flat_map(|d| &d.device_inputs) {
        writes.ops.push(InputWrite::Device {
            src: input.src.clone(),
            offset: input.offset,
            to: worker.runtime.input_ids.clone(),
            at: input.flat_index * size_of::<u32>(),
        });
    }
    worker
        .worker
        .tq_dequant_max_blocks
        .store(tq_dequant_block_width(inputs), Relaxed);
    worker.worker.unrotated_blocks.store(
        step_has_unrotated_blocks(inputs, W::GLOBAL_BLOCK_SIZE),
        Relaxed,
    );
    worker.worker.gdn_plain.store(
        inputs.gdn_is_fresh.is_none_or(|codes| {
            codes
                .iter()
                .all(|&c| crate::gdn_state::GdnStep::decode(c).is_plain())
        }),
        Relaxed,
    );
    Ok(writes)
}

/// Whether some sequence of the step has an unrotated (bit-31, span) block
/// among the blocks it uses, in the full KV group's block table.
fn step_has_unrotated_blocks(inputs: &ForwardInputs<'_>, block_size: u32) -> bool {
    let (Some(table), stride) = (inputs.block_tables.first(), tq_dequant_block_width(inputs))
    else {
        return false;
    };
    let stride = stride as usize;
    if stride == 0 {
        return false;
    }
    table.chunks(stride).enumerate().any(|(seq, row)| {
        let used = inputs.seq_used_k.map_or(stride, |k| {
            (k.get(seq).copied().unwrap_or(0) as usize).div_ceil(block_size.max(1) as usize)
        });
        row[..used.min(row.len())]
            .iter()
            .any(|b| b & crate::UNROTATED_BLOCK_BIT != 0)
    })
}

fn tq_dequant_block_width(inputs: &ForwardInputs<'_>) -> u32 {
    let num_seqs = inputs
        .cu_seqlens_q
        .map(|cu| cu.len().saturating_sub(1).max(1))
        .unwrap_or(1);
    inputs
        .block_tables
        .first()
        .map(|b| (b.len() / num_seqs.max(1)) as u32)
        .unwrap_or(0)
}

/// Record each present input slice's write into the matching runtime
/// buffer ([`InputWrites`]). Validates length first; on overflow returns
/// [`ForwardError::BufferTooSmall`].
fn write_runtime_inputs(
    runtime: &RuntimeBindings,
    inputs: &ForwardInputs<'_>,
    block_table_stride: MaxBlocksPerSeq,
    writes: &mut InputWrites,
) -> Result<(), ForwardError> {
    // `src` at the head of runtime input `buffer`, every byte past it `padding`
    // ([`WrittenExtents::write`]: only what the last forward wrote past it is re-padded).
    let mut write = |kind: &'static str, buffer: &Buffer, src: &[u8], padding: Padding| {
        (runtime.written)
            .write(buffer, src, padding, writes)
            .map_err(|bytes_needed| ForwardError::BufferTooSmall {
                kind,
                bytes_needed,
                bytes_available: buffer.length(),
            })
    };
    write(
        "input_ids",
        &runtime.input_ids,
        bytes_of(inputs.input_ids),
        Padding::Zero,
    )?;
    write(
        "positions",
        &runtime.positions,
        bytes_of(inputs.positions),
        Padding::Zero,
    )?;
    // Per-KV-cache-group slot_mappings (vLLM hybrid layout). Padding lanes
    // get sentinel `u32::MAX` so the rope_append kernel early-outs before
    // writing the paged K/V cache. Zero-fill would otherwise route every
    // padding token's K projection into cache slot 0, overwriting the real
    // K of position 0 (each padding token has positions[t]=0 / input_ids[t]=0,
    // so they all write K_proj(token 0) to slot 0, racing with — and winning
    // against — the real position-0 write).
    for (g, s) in inputs.slot_mappings.iter().enumerate() {
        write(
            "slot_mapping",
            &runtime.slot_mappings[g],
            bytes_of(s),
            Padding::NoSlot,
        )?;
    }
    if let Some(s) = inputs.cu_seqlens_q {
        write(
            "cu_seqlens_q",
            &runtime.cu_seqlens_q,
            bytes_of(s),
            Padding::Zero,
        )?;
    }
    if let Some(s) = inputs.seq_used_k {
        write(
            "seq_used_k",
            &runtime.seq_used_k,
            bytes_of(s),
            Padding::Zero,
        )?;
    }
    // Span labels for block-diagonal attention: past the labels, zero — so a forward without
    // spans leaves it all-zero ⇒ the kernel's span mask is a no-op. The prefill kernel READS
    // span_ids under ATTN_ROR (every rope-on-read pipeline) whether or not the forward has spans:
    // labels left from a prior forward would make a query's q_span mismatch every key →
    // fully-masked rows → NaN/crash. Label 0 = attends-all = mask inert.
    let spans = inputs.span_ids.unwrap_or(&[]);
    write(
        "span_ids",
        &runtime.span_ids,
        bytes_of(spans),
        Padding::Zero,
    )?;
    if std::env::var("SCRATCHY_METAL_TRACE").is_ok() {
        if let Some(s) = inputs.block_tables.first() {
            eprintln!(
                "[runtime] block_table[0][0..min(8,len)] = {:?} (len={})",
                &s[..s.len().min(8)],
                s.len(),
            );
        }
        eprintln!(
            "[runtime] num_tokens = {} input_ids[0..min(8,len)] = {:?} positions[0..min(8,len)] = {:?}",
            inputs.num_tokens,
            &inputs.input_ids[..inputs.input_ids.len().min(8)],
            &inputs.positions[..inputs.positions.len().min(8)],
        );
    }
    // Per-KV-cache-group block tables (group 0 = full, then sliding): one row per sequence, each
    // the KV cap rung's width, the stride the tapes read them at.
    let num_seqs = (inputs.cu_seqlens_q).map_or(1, |cu| cu.len().saturating_sub(1).max(1));
    for (g, s) in inputs.block_tables.iter().enumerate() {
        if !s.is_empty() && s.len() != num_seqs * block_table_stride.get() as usize {
            return Err(ForwardError::BlockTableWidth {
                len: s.len(),
                num_seqs,
                rung: block_table_stride,
            });
        }
        write(
            "block_table",
            &runtime.block_tables[g],
            bytes_of(s),
            Padding::Zero,
        )?;
    }
    // Tell the gather kernel (used right before lm_head) what the
    // actual num_tokens of this forward is so it can pick the
    // last-token source row.
    write(
        "num_tokens",
        &runtime.num_tokens_u32,
        bytes_of(&[inputs.num_tokens]),
        Padding::Zero,
    )?;
    // Plumb the lm_head sample-row index list. When the caller
    // supplies `last_token_indices` (the common case: every forward
    // that produces a sampled token) the slice trio uses these as
    // the gather sources and scatter destinations; otherwise the
    // count is 0 and the slice is a no-op for this step.
    let num_sample_rows = inputs
        .last_token_indices
        .map(|s| s.len() as u32)
        .unwrap_or(0);
    write(
        "num_sample_rows",
        &runtime.num_sample_rows_u32,
        bytes_of(&[num_sample_rows]),
        Padding::Zero,
    )?;
    if let Some(s) = inputs.last_token_indices {
        write(
            "last_token_indices",
            &runtime.sample_indices,
            bytes_of(s),
            Padding::Zero,
        )?;
    }
    // GDN per-forward indices (hybrid arches only). i32 slot ids + u32
    // fresh flags, one per batched sequence in cu_seqlens order.
    if let Some(idx) = inputs.gdn_state_indices {
        // i32 and u32 share the 4-byte layout the kernels read as int.
        let as_u32 = unsafe { std::slice::from_raw_parts(idx.as_ptr() as *const u32, idx.len()) };
        write(
            "gdn_state_indices",
            &runtime.gdn_state_indices,
            bytes_of(as_u32),
            Padding::Zero,
        )?;
    }
    if let Some(fresh) = inputs.gdn_is_fresh {
        write(
            "gdn_is_fresh",
            &runtime.gdn_is_fresh,
            bytes_of(fresh),
            Padding::Zero,
        )?;
    }
    // Vision externs (vision-tower arches only). Copied verbatim as
    // bytes — `freqs` is f32, `pixels` is the model dtype (bf16); the
    // runtime buffers are untyped and the kernels reinterpret.
    if let Some(b) = inputs.vision_rope_freqs {
        write(
            "vision_rope_freqs",
            &runtime.vision_rope_freqs,
            b,
            Padding::Zero,
        )?;
    }
    if let Some(b) = inputs.pixels {
        write("pixels", &runtime.pixels, b, Padding::Zero)?;
    }
    if let Some(b) = inputs.pos_embeds {
        write("pos_embeds", &runtime.vision_pos_embeds, b, Padding::Zero)?;
    }
    if let Some(b) = inputs.target_hidden {
        write("target_hidden", &runtime.target_hidden, b, Padding::Zero)?;
    }
    // Qwen2.5-VL windowed-attention externs (i32/u32 bytes, verbatim).
    if let Some(b) = inputs.vision_cu_seqlens_full {
        write(
            "vision_cu_seqlens_full",
            &runtime.vision_cu_seqlens_full,
            b,
            Padding::Zero,
        )?;
    }
    if let Some(b) = inputs.vision_cu_seqlens_window {
        write(
            "vision_cu_seqlens_window",
            &runtime.vision_cu_seqlens_window,
            b,
            Padding::Zero,
        )?;
    }
    if let Some(b) = inputs.vision_window_index {
        write(
            "vision_window_index",
            &runtime.vision_window_index,
            b,
            Padding::Zero,
        )?;
    }
    if let Some(b) = inputs.vision_reverse_indices {
        write(
            "vision_reverse_indices",
            &runtime.vision_reverse_indices,
            b,
            Padding::Zero,
        )?;
    }
    if let Some(b) = inputs.vision_position_ids {
        write(
            "vision_position_ids",
            &runtime.vision_position_ids,
            b,
            Padding::Zero,
        )?;
    }
    // Multimodal splice (MM-bearing batches only). mm_embeds = the
    // projected vision output (bytes); mm_dst_rows = per-row dst (u32).
    if let Some(b) = inputs.mm_embeds {
        write("mm_embeds", &runtime.mm_embeds, b, Padding::Zero)?;
    }
    if let Some(d) = inputs.mm_dst_rows {
        write(
            "mm_dst_rows",
            &runtime.mm_dst_rows,
            bytes_of(d),
            Padding::Zero,
        )?;
    }
    // MRoPE cos/sin override (MRoPE text decoders only). Bytes in the
    // rope kernel's element dtype; the worker binds it at the cos/sin slot.
    if let Some(b) = inputs.mrope_cos_sin {
        write("mrope_cos_sin", &runtime.mrope_cos_sin, b, Padding::Zero)?;
    }
    Ok(())
}

/// `src`'s bytes.
fn bytes_of(src: &[u32]) -> &[u8] {
    // SAFETY: `u32` has no padding and every byte of it is initialized.
    unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), std::mem::size_of_val(src)) }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::interpreter::metal::__re::{Buffer, MTLDevice, MTLResourceOptions};
    use crate::interpreter::metal::lowered::{
        Binding, DispatchShape, KernelId, LoweredCommand, LoweredMetalTape, SourceRef, WeightTensor,
    };
    use crate::tape::ids::{LayerId, SourceIx};
    use scratchy_layers::RmsNorm;
    use scratchy_tensors::{DType, DeviceAllocator, GpuTensor};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    /// The test tapes' variant: `TestWeights`' block capacity, no TurboQuant decode attention.
    const TEST_VARIANT: TapeVariant = TapeVariant {
        cap: MaxBlocksPerSeq(TestWeights::MAX_BLOCKS_PER_SEQ),
        tq_heads: None,
    };

    /// Test fixture: holds `CanonicalParams` constants AND the layer
    /// its one model source resolves to. Pool tests only exercise
    /// RmsNorm bindings (the `synthetic_tape` builder below), so only
    /// the rmsnorm layer needs real backing.
    struct TestWeights {
        rmsnorm_layer: RmsNorm,
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
    }

    // `WeightAccessors` is a supertrait of `CanonicalParams` (every method defaults); metal
    // resolves the synthetic RmsNorm command's weight through the one-family source manifest.
    impl scratchy_ir::WeightAccessors for TestWeights {}
    impl ModelSources for TestWeights {
        const SOURCES: &'static [&'static str] = &["rmsnorm"];
        fn source(&self, ix: SourceIx, _layer: LayerId) -> Option<SourceRef<'_>> {
            (ix == SourceIx(0)).then_some(SourceRef::RmsNorm(&self.rmsnorm_layer))
        }
    }

    /// Build a `TestWeights` + the allocator that owns its RmsNorm
    /// weight's backing MTLBuffer. The allocator is also threaded
    /// into the pool so it can map `weight.raw_ptr()` back to
    /// `(&MTLBuffer, offset)` when it resolves the tapes' sources.
    fn build_test_weights(device: &Device) -> (Arc<TestWeights>, Arc<MetalAllocator>) {
        let mut allocator = MetalAllocator::new(device.clone());
        let bytes = vec![0u8; TestWeights::Q_SIZE * 2];
        let ptr = unsafe {
            allocator
                .alloc_and_copy_host(bytes.as_ptr(), bytes.len())
                .expect("rmsnorm weight alloc")
        };
        let tensor = unsafe { GpuTensor::new(ptr, &[TestWeights::Q_SIZE], DType::F16) };
        let weights = Arc::new(TestWeights {
            rmsnorm_layer: RmsNorm::new(tensor, 1e-5),
        });
        (weights, Arc::new(allocator))
    }

    fn alloc(device: &Device, bytes: u64) -> Buffer {
        device
            .newBufferWithLength_options(
                bytes.max(1) as usize,
                MTLResourceOptions::StorageModeShared,
            )
            .expect("newBuffer")
    }

    fn empty_runtime(device: &Device, num_layers: usize) -> RuntimeBindings {
        RuntimeBindings {
            input_ids: alloc(device, 16),
            positions: alloc(device, 16),
            slot_mappings: vec![alloc(device, 16)],
            cu_seqlens_q: alloc(device, 16),
            seq_used_k: alloc(device, 16),
            span_ids: alloc(device, 16),
            block_tables: vec![alloc(device, 16)],
            layer_to_group: Vec::new(),
            kv_cache_k: (0..num_layers).map(|_| alloc(device, 16)).collect(),
            kv_cache_v: (0..num_layers).map(|_| alloc(device, 16)).collect(),
            block_unrotated_flags: (0..num_layers).map(|_| alloc(device, 16)).collect(),
            tq: None,
            num_tokens_u32: alloc(device, 4),
            num_sample_rows_u32: alloc(device, 4),
            sample_indices: alloc(device, 16),
            gdn_state_conv: ::std::vec::Vec::new(),
            gdn_state_ssm: ::std::vec::Vec::new(),
            gdn_state_indices: alloc(device, 16),
            gdn_is_fresh: alloc(device, 16),
            vision_rope_freqs: alloc(device, 16),
            pixels: alloc(device, 16),
            vision_pos_embeds: alloc(device, 16),
            target_hidden: alloc(device, 16),
            mm_embeds: alloc(device, 16),
            mm_dst_rows: alloc(device, 16),
            mrope_cos_sin: alloc(device, 16),
            vision_cu_seqlens_full: alloc(device, 16),
            vision_cu_seqlens_window: alloc(device, 16),
            vision_window_index: alloc(device, 16),
            vision_reverse_indices: alloc(device, 16),
            vision_position_ids: alloc(device, 16),
            written: Default::default(),
        }
    }

    /// Single-bucket synthetic tape: one RmsNorm command. Enough to
    /// verify the worker bakes; the kernel itself isn't fired in
    /// 5.D tests (5.E hooks `run_bucket` to a real cmdbuf).
    fn synthetic_tape(bucket_m: u32) -> LoweredMetalTape {
        use crate::tape::lowered::IntoBaked;
        let cmd = LoweredCommand {
            kernel: KernelId::RmsNorm,
            library: "rmsnorm",
            function: "rmsnorm_f16_s_f16_specialized",
            constants: crate::tape::kernel_constants::RmsNormConstants {
                bucket_m: crate::tape::ids::BucketM(bucket_m),
                q_size: crate::tape::ids::QSize(
                    <TestWeights as scratchy_ir::CanonicalParams>::Q_SIZE as u32,
                ),
                rms_norm_eps: crate::tape::ids::RmsNormEps(
                    <TestWeights as scratchy_ir::CanonicalParams>::RMS_NORM_EPS,
                ),
                weight_offset: 0.0,
            }
            .into_baked(),
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
                    ix: SourceIx(0),
                    which: WeightTensor::Weight,
                    layer: LayerId(0),
                    binding_index: 2,
                },
            ]),
            gemm_dims: None,
        };
        LoweredMetalTape {
            bucket_m,
            num_arena_slots: 2,
            commands: crate::interpreter::metal::lowered::baked_commands(vec![cmd.into()])
                .expect("commands"),
            splitk_scratch_bytes: 0,
            barrier_before: &[],
            // Straight-line: this fixture is one hand-built body, no rolled layer loop.
            loops: &[],
            moe_scratch_bytes: 0,
            roped_k_scratch_bytes: 0,
            attn_unfused_scratch_bytes: 0,
        }
    }

    /// Build a pool with `max_workers = max` for a one-bucket
    /// TinyLlama-shaped synthetic tape. Returns `None` when no
    /// Metal device is present (lets each test silent-skip).
    fn build_pool(max: usize) -> Option<MetalWorkerPool<TestWeights>> {
        let device = crate::detect_device().filter(|_| crate::metal4_available())?;
        let device = Arc::new(device.device.clone());

        let tapes: Arc<[_]> = Arc::from(vec![synthetic_tape(1)]);
        let pipelines = Arc::new(
            crate::aot::tape_pipelines(&device, &tapes, TestWeights::METAL_DTYPE, TEST_VARIANT)
                .expect("pipelines"),
        );

        let (weights, allocator) = build_test_weights(&device);

        let arena_layout: ArenaLayout = vec![4096, 4096];
        let runtime_factory: RuntimeFactory = RuntimeFactory::new(|d, _| empty_runtime(d, 1));

        let pool = MetalWorkerPool::<TestWeights>::new(
            device,
            &weights,
            allocator,
            pipelines,
            tapes,
            MaxBlocksPerSeq(128),
            arena_layout,
            runtime_factory,
            max,
            LentActivation::default(),
        )
        .expect("pool builds");
        Some(pool)
    }

    #[test]
    fn pool_starts_with_one_worker() {
        let Some(pool) = build_pool(4) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        assert_eq!(pool.max_workers(), 4);
        assert_eq!(pool.current_size(), 1, "first worker eagerly created");
        assert_eq!(pool.available(), 1);
    }

    #[test]
    fn checkout_returns_eagerly_created_worker_first() {
        let Some(pool) = build_pool(4) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let _g = pool.checkout().expect("checkout 1");
        assert_eq!(pool.current_size(), 1, "first checkout reuses eager worker");
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn pool_grows_under_demand_up_to_cap() {
        let Some(pool) = build_pool(3) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let g1 = pool.checkout().expect("checkout 1");
        let g2 = pool.checkout().expect("checkout 2");
        let g3 = pool.checkout().expect("checkout 3");
        assert_eq!(pool.current_size(), 3, "grew to cap");
        assert_eq!(pool.available(), 0);
        // try_checkout at cap returns None, not Some(Err).
        assert!(pool.try_checkout().is_none());
        drop(g1);
        drop(g2);
        drop(g3);
        assert_eq!(pool.available(), 3);
        assert_eq!(pool.current_size(), 3, "cap unchanged after checkin");
    }

    #[test]
    fn guard_drop_returns_worker_to_pool() {
        let Some(pool) = build_pool(2) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        {
            let _g = pool.checkout().expect("checkout");
            assert_eq!(pool.available(), 0);
        }
        assert_eq!(pool.available(), 1, "drop returns worker");
        // Subsequent checkout reuses the existing worker rather than
        // growing the pool.
        let _g2 = pool.checkout().expect("checkout 2");
        assert_eq!(pool.current_size(), 1, "no growth on reuse");
    }

    /// Concurrency test: hold the only worker on the main thread,
    /// spawn a second thread that calls `checkout()` (must block),
    /// release the worker, verify the spawned thread unblocks.
    ///
    /// Uses an `AtomicBool` + a small bounded sleep to detect "still
    /// blocked". Sleep is short to keep the test fast; flake risk is
    /// low because the spawned thread only flips the flag *after*
    /// `checkout()` returns.
    #[test]
    fn checkout_blocks_when_at_cap_unblocks_on_checkin() {
        let Some(pool) = build_pool(1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let pool = Arc::new(pool);
        let g1 = pool.checkout().expect("checkout 1");
        assert_eq!(pool.available(), 0);

        let pool_c = pool.clone();
        let started = Arc::new(AtomicBool::new(false));
        let completed = Arc::new(AtomicBool::new(false));
        let started_c = started.clone();
        let completed_c = completed.clone();

        let handle = std::thread::spawn(move || {
            started_c.store(true, Ordering::SeqCst);
            let _g = pool_c.checkout().expect("blocking checkout");
            completed_c.store(true, Ordering::SeqCst);
        });

        // Wait for the spawned thread to enter checkout. SeqCst load
        // is sufficient — `started` is set before the blocking call.
        while !started.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        // Give the spawned thread a chance to enter the wait state.
        std::thread::sleep(Duration::from_millis(20));
        assert!(
            !completed.load(Ordering::SeqCst),
            "spawned thread should still be blocked while main holds the only worker"
        );

        drop(g1);
        handle.join().expect("spawned thread completed");
        assert!(
            completed.load(Ordering::SeqCst),
            "spawned thread unblocked once main checked the worker back in"
        );
        // Pool didn't grow — the spawned thread reused the existing one.
        assert_eq!(pool.current_size(), 1);
    }

    /// `try_checkout` returns `None` (not `Some(Err)`) when the pool
    /// is at cap and every worker is busy. Distinguishes "full" from
    /// "GPU OOM".
    #[test]
    fn try_checkout_at_cap_returns_none() {
        let Some(pool) = build_pool(1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let _g = pool.checkout().expect("checkout");
        match pool.try_checkout() {
            None => {}
            Some(Ok(_)) => panic!("try_checkout should not have succeeded at cap"),
            Some(Err(e)) => panic!("try_checkout should be None at cap, got Err({e:?})"),
        }
    }

    /// Two concurrent threads each grow the pool by one and run to
    /// completion. Verifies the spawn-while-locking-released path
    /// doesn't double-allocate or deadlock.
    #[test]
    fn concurrent_growth_to_cap() {
        let Some(pool) = build_pool(2) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let pool = Arc::new(pool);
        let pool_a = pool.clone();
        let pool_b = pool.clone();

        let h_a = std::thread::spawn(move || {
            let _g = pool_a.checkout().expect("checkout a");
            std::thread::sleep(Duration::from_millis(10));
        });
        let h_b = std::thread::spawn(move || {
            let _g = pool_b.checkout().expect("checkout b");
            std::thread::sleep(Duration::from_millis(10));
        });

        h_a.join().unwrap();
        h_b.join().unwrap();
        assert!(
            pool.current_size() <= 2,
            "pool must not exceed max_workers (got {})",
            pool.current_size()
        );
        assert_eq!(pool.available(), pool.current_size());
    }

    // ---------------- Phase 5.E: forward() ----------------

    /// Build a pool whose tape carries one bucket per provided
    /// `bucket_m`. Arena slots are sized for the largest bucket so the
    /// runtime buffers / arena handle every entry. The runtime factory
    /// sizes per-token arrays for the largest bucket too — smaller
    /// `num_tokens` values fit trivially.
    fn build_multi_bucket_pool(
        bucket_ms: &[u32],
        max_workers: usize,
    ) -> Option<MetalWorkerPool<TestWeights>> {
        let device = crate::detect_device().filter(|_| crate::metal4_available())?;
        let device = Arc::new(device.device.clone());
        let (weights, allocator) = build_test_weights(&device);
        let tapes: Arc<[_]> = bucket_ms
            .iter()
            .copied()
            .map(synthetic_tape)
            .collect::<Vec<_>>()
            .into();
        let pipelines = Arc::new(
            crate::aot::tape_pipelines(&device, &tapes, TestWeights::METAL_DTYPE, TEST_VARIANT)
                .expect("pipelines"),
        );
        // Arena slot for the synthetic RmsNorm: M × hidden_size f16 =
        // M × Q_SIZE × 2 bytes. Sized for the worst-case bucket.
        let max_m = bucket_ms.iter().copied().max().unwrap_or(1) as u64;
        let slot_bytes = max_m * (TestWeights::Q_SIZE as u64) * 2;
        let arena_layout: ArenaLayout = vec![slot_bytes, slot_bytes];
        // Per-token runtime arrays sized for max bucket.
        let max_m_bytes = (max_m * 4).max(16);
        let runtime_factory: RuntimeFactory = RuntimeFactory::new(move |d, _| RuntimeBindings {
            input_ids: alloc(d, max_m_bytes),
            positions: alloc(d, max_m_bytes),
            slot_mappings: vec![alloc(d, max_m_bytes)],
            cu_seqlens_q: alloc(d, 16),
            seq_used_k: alloc(d, 16),
            span_ids: alloc(d, 16),
            block_tables: vec![alloc(d, 16)],
            layer_to_group: Vec::new(),
            kv_cache_k: vec![alloc(d, 16)],
            kv_cache_v: vec![alloc(d, 16)],
            block_unrotated_flags: vec![alloc(d, 16)],
            tq: None,
            num_tokens_u32: alloc(d, 4),
            num_sample_rows_u32: alloc(d, 4),
            sample_indices: alloc(d, max_m_bytes),
            gdn_state_conv: ::std::vec::Vec::new(),
            gdn_state_ssm: ::std::vec::Vec::new(),
            gdn_state_indices: alloc(d, 16),
            gdn_is_fresh: alloc(d, 16),
            vision_rope_freqs: alloc(d, 16),
            pixels: alloc(d, 16),
            vision_pos_embeds: alloc(d, 16),
            target_hidden: alloc(d, 16),
            mm_embeds: alloc(d, 16),
            mm_dst_rows: alloc(d, 16),
            mrope_cos_sin: alloc(d, 16),
            vision_cu_seqlens_full: alloc(d, 16),
            vision_cu_seqlens_window: alloc(d, 16),
            vision_window_index: alloc(d, 16),
            vision_reverse_indices: alloc(d, 16),
            vision_position_ids: alloc(d, 16),
            written: Default::default(),
        });
        let pool = MetalWorkerPool::<TestWeights>::new(
            device,
            &weights,
            allocator,
            pipelines,
            tapes,
            MaxBlocksPerSeq(128),
            arena_layout,
            runtime_factory,
            max_workers,
            LentActivation::default(),
        )
        .expect("pool builds");
        Some(pool)
    }

    #[test]
    fn pick_bucket_returns_smallest_fit() {
        let Some(pool) = build_multi_bucket_pool(&[1, 8, 32], 1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        // Decode bucket.
        assert_eq!(pool.pick_bucket(1).unwrap(), 0);
        // Doesn't fit bucket 0; picks the next-smallest that fits.
        assert_eq!(pool.pick_bucket(2).unwrap(), 1);
        assert_eq!(pool.pick_bucket(8).unwrap(), 1);
        assert_eq!(pool.pick_bucket(9).unwrap(), 2);
        assert_eq!(pool.pick_bucket(32).unwrap(), 2);
    }

    #[test]
    fn pick_bucket_zero_tokens_errors() {
        let Some(pool) = build_multi_bucket_pool(&[1, 8], 1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        assert!(matches!(pool.pick_bucket(0), Err(ForwardError::ZeroTokens)));
    }

    #[test]
    fn pick_bucket_overflow_errors() {
        let Some(pool) = build_multi_bucket_pool(&[1, 8], 1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        match pool.pick_bucket(9) {
            Err(ForwardError::NoBucketFits {
                num_tokens: 9,
                max_bucket: 8,
            }) => {}
            other => panic!("expected NoBucketFits {{ 9, 8 }}, got {other:?}"),
        }
    }

    /// Tape order isn't required to be sorted — `pick_bucket` should
    /// still find the smallest fit when buckets come in arbitrary
    /// order.
    #[test]
    fn pick_bucket_handles_unsorted_tape_order() {
        let Some(pool) = build_multi_bucket_pool(&[32, 1, 8], 1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        assert_eq!(
            pool.pick_bucket(1).unwrap(),
            1,
            "smallest bucket at index 1"
        );
        assert_eq!(
            pool.pick_bucket(8).unwrap(),
            2,
            "8 fits index 2 (bucket_m=8)"
        );
        assert_eq!(pool.pick_bucket(9).unwrap(), 0, "9 only fits the 32 bucket");
    }

    /// A step has an unrotated block only if some sequence's USED blocks hold
    /// one: a bit-31 entry past a sequence's `seq_used_k` (stale padding in
    /// its row) does not count, and neither does another group's table.
    #[test]
    fn a_step_has_unrotated_blocks_only_among_the_blocks_it_uses() {
        let flagged = |b: u32| b | crate::UNROTATED_BLOCK_BIT;
        let cu = [0u32, 1, 2];
        let used = [32u32, 17];
        let has = |table: &[u32]| {
            let inputs = ForwardInputs {
                span_ids: None,
                num_tokens: 2,
                input_ids: &[0, 0],
                positions: &[31, 16],
                slot_mappings: Vec::new(),
                cu_seqlens_q: Some(&cu),
                seq_used_k: Some(&used),
                block_tables: vec![table],
                has_spec_tokens: false,
                last_token_indices: None,
                gdn_state_indices: None,
                gdn_is_fresh: None,
                vision_rope_freqs: None,
                vision_cu_seqlens_full: None,
                vision_cu_seqlens_window: None,
                vision_window_index: None,
                vision_reverse_indices: None,
                vision_position_ids: None,
                pixels: None,
                pos_embeds: None,
                target_hidden: None,
                mm_embeds: None,
                mm_dst_rows: None,
                mrope_cos_sin: None,
                deferred: None,
            };
            step_has_unrotated_blocks(&inputs, 16)
        };
        // Rows of 4 blocks; the first sequence uses 2 blocks, the second 2.
        assert!(!has(&[1, 2, 0, 0, 3, 4, 0, 0]));
        assert!(has(&[1, flagged(2), 0, 0, 3, 4, 0, 0]));
        assert!(has(&[1, 2, 0, 0, 3, flagged(4), 0, 0]));
        assert!(!has(&[1, 2, flagged(9), 0, 3, 4, 0, flagged(9)]));
    }

    #[test]
    fn forward_runs_one_decode_step() {
        let Some(pool) = build_multi_bucket_pool(&[1], 1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let inputs = ForwardInputs {
            span_ids: None,
            num_tokens: 1,
            input_ids: &[0u32],
            positions: &[0u32],
            slot_mappings: Vec::new(),
            cu_seqlens_q: None,
            seq_used_k: None,
            block_tables: Vec::new(),
            has_spec_tokens: false,
            last_token_indices: None,
            gdn_state_indices: None,
            gdn_is_fresh: None,
            vision_rope_freqs: None,
            vision_cu_seqlens_full: None,
            vision_cu_seqlens_window: None,
            vision_window_index: None,
            vision_reverse_indices: None,
            vision_position_ids: None,
            pixels: None,
            pos_embeds: None,
            target_hidden: None,
            mm_embeds: None,
            mm_dst_rows: None,
            mrope_cos_sin: None,
            deferred: None,
        };
        // Closure runs *while* the worker is checked out — assertion
        // is that we got it (not on numerical correctness; that's
        // 5.G's job via cpu_golden).
        let saw = pool
            .forward(&inputs, |worker, _bucket_idx| {
                // Worker arena is alive in the closure; reading its
                // contents would inspect the rmsnorm output. We only
                // assert structural facts here.
                assert_eq!(worker.bucket_bakings.len(), 1);
                42u32
            })
            .expect("forward succeeds");
        assert_eq!(saw, 42);
        assert_eq!(pool.available(), 1, "worker returned to pool after forward");
    }

    /// Two failed command buffers in flight each report their own failure, to every wait: the
    /// engine's resolver and the worker both wait on a deferred step, and a wait does not consume
    /// the error a later wait — or another command buffer's — needs.
    #[test]
    fn every_wait_on_a_failed_command_buffer_reports_it() {
        use objc2_metal::MTLSharedEvent;
        let Some(pool) = build_multi_bucket_pool(&[1], 1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let event = pool.device.newSharedEvent().expect("shared event");
        event.setSignaledValue(2);
        let failed = |value| {
            let error = Arc::new(Mutex::new(Some(format!("command buffer {value} failed"))));
            InFlight {
                event: event.clone(),
                value,
                _feedback: commit_options(&error),
                error,
            }
        };
        let (first, second) = (failed(1), failed(2));
        for step in [&first, &first, &second, &second] {
            assert!(
                matches!(step.wait(), Err(ForwardError::ExecutionFailed(_))),
                "a failed command buffer waited on reports its failure"
            );
        }
    }

    /// A forward's input writes are made on the device, not the host: the staged ones, re-padding
    /// what the forward before wrote, then each device input over its placeholder — a token the
    /// host never had.
    #[test]
    fn recorded_input_writes_land_on_the_device() {
        let Some(pool) = build_multi_bucket_pool(&[4], 1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        pool.ensure_mtl4();
        let sampled = pool
            .allocator
            .residency()
            .pin(crate::mtl4_dispatch::shared_bytes(
                &pool.device,
                bytes_of(&[7, 42]),
            ));
        pool.allocator.residency().commit();
        let guard = pool.checkout().expect("a worker");
        let input_ids = || crate::mtl4_dispatch::read_slice::<u32>(&guard.runtime.input_ids, 4);
        let run = |ids: &[u32], device_inputs| {
            let positions: Vec<u32> = (0..ids.len() as u32).collect();
            let deferral = Deferral::new(device_inputs);
            let inputs = ForwardInputs {
                span_ids: None,
                num_tokens: ids.len() as u32,
                input_ids: ids,
                positions: &positions,
                slot_mappings: Vec::new(),
                cu_seqlens_q: None,
                seq_used_k: None,
                block_tables: Vec::new(),
                has_spec_tokens: false,
                last_token_indices: None,
                gdn_state_indices: None,
                gdn_is_fresh: None,
                vision_rope_freqs: None,
                vision_cu_seqlens_full: None,
                vision_cu_seqlens_window: None,
                vision_window_index: None,
                vision_reverse_indices: None,
                vision_position_ids: None,
                pixels: None,
                pos_embeds: None,
                mm_embeds: None,
                mm_dst_rows: None,
                mrope_cos_sin: None,
                target_hidden: None,
                deferred: Some(&deferral),
            };
            let before = input_ids();
            let writes = begin_step(&guard, &inputs, pool.block_table_stride).expect("inputs fit");
            assert_eq!(
                input_ids(),
                before,
                "a recorded write leaves the host copy alone"
            );
            let ((), committed) = pool.commit(Some(&writes), |_| Ok(())).expect("commits");
            committed.in_flight.wait().expect("runs");
            input_ids()
        };
        let device_input = crate::interpreter::metal::DeviceInput {
            src: (*sampled).clone(),
            offset: size_of::<u32>(),
            flat_index: 1,
        };
        assert_eq!(run(&[5, 0, 9], vec![device_input]), [5, 42, 9, 0]);
        assert_eq!(run(&[8], Vec::new()), [8, 0, 0, 0]);
    }

    #[test]
    fn forward_rejects_zero_tokens() {
        let Some(pool) = build_multi_bucket_pool(&[1], 1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let inputs = ForwardInputs {
            span_ids: None,
            num_tokens: 0,
            input_ids: &[],
            positions: &[],
            slot_mappings: Vec::new(),
            cu_seqlens_q: None,
            seq_used_k: None,
            block_tables: Vec::new(),
            has_spec_tokens: false,
            last_token_indices: None,
            gdn_state_indices: None,
            gdn_is_fresh: None,
            vision_rope_freqs: None,
            vision_cu_seqlens_full: None,
            vision_cu_seqlens_window: None,
            vision_window_index: None,
            vision_reverse_indices: None,
            vision_position_ids: None,
            pixels: None,
            pos_embeds: None,
            target_hidden: None,
            mm_embeds: None,
            mm_dst_rows: None,
            mrope_cos_sin: None,
            deferred: None,
        };
        let err = pool
            .forward(&inputs, |_, _| ())
            .expect_err("zero-token forward rejected");
        assert!(matches!(err, ForwardError::ZeroTokens));
        // Worker was never checked out — pool stays at the eager 1.
        assert_eq!(pool.available(), 1);
    }

    #[test]
    fn forward_rejects_oversized_token_count() {
        let Some(pool) = build_multi_bucket_pool(&[1, 8], 1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let big = vec![0u32; 9];
        let inputs = ForwardInputs {
            span_ids: None,
            num_tokens: 9,
            input_ids: &big,
            positions: &big,
            slot_mappings: Vec::new(),
            cu_seqlens_q: None,
            seq_used_k: None,
            block_tables: Vec::new(),
            has_spec_tokens: false,
            last_token_indices: None,
            gdn_state_indices: None,
            gdn_is_fresh: None,
            vision_rope_freqs: None,
            vision_cu_seqlens_full: None,
            vision_cu_seqlens_window: None,
            vision_window_index: None,
            vision_reverse_indices: None,
            vision_position_ids: None,
            pixels: None,
            pos_embeds: None,
            target_hidden: None,
            mm_embeds: None,
            mm_dst_rows: None,
            mrope_cos_sin: None,
            deferred: None,
        };
        match pool.forward(&inputs, |_, _| ()) {
            Err(ForwardError::NoBucketFits {
                num_tokens: 9,
                max_bucket: 8,
            }) => {}
            other => panic!("expected NoBucketFits {{ 9, 8 }}, got {other:?}"),
        }
        assert_eq!(pool.available(), 1, "no checkout on bucket failure");
    }

    /// `BufferTooSmall` fires when the caller stages more bytes than
    /// the runtime buffer can hold. Crafted by sizing the runtime
    /// `input_ids` buffer to 16 bytes (default in this test setup) and
    /// passing a 5-element slice (20 bytes).
    #[test]
    fn forward_rejects_oversized_input_slice() {
        // Single-bucket pool with the *smaller* runtime layout — the
        // existing `build_pool` allocates 16-byte runtime buffers,
        // perfect for triggering the overflow path.
        let Some(pool) = build_pool(1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        // 5 × u32 = 20 bytes; runtime input_ids buffer is 16 bytes.
        let too_big = [0u32, 0u32, 0u32, 0u32, 0u32];
        let inputs = ForwardInputs {
            span_ids: None,
            num_tokens: 1, // pick_bucket succeeds (bucket 0 = decode)
            input_ids: &too_big,
            positions: &[0u32],
            slot_mappings: Vec::new(),
            cu_seqlens_q: None,
            seq_used_k: None,
            block_tables: Vec::new(),
            has_spec_tokens: false,
            last_token_indices: None,
            gdn_state_indices: None,
            gdn_is_fresh: None,
            vision_rope_freqs: None,
            vision_cu_seqlens_full: None,
            vision_cu_seqlens_window: None,
            vision_window_index: None,
            vision_reverse_indices: None,
            vision_position_ids: None,
            pixels: None,
            pos_embeds: None,
            target_hidden: None,
            mm_embeds: None,
            mm_dst_rows: None,
            mrope_cos_sin: None,
            deferred: None,
        };
        let err = pool
            .forward(&inputs, |_, _| ())
            .expect_err("oversized slice rejected");
        match err {
            ForwardError::BufferTooSmall {
                kind: "input_ids",
                bytes_needed: 20,
                bytes_available: 16,
            } => {}
            other => panic!("expected BufferTooSmall on input_ids, got {other:?}"),
        }
        // Checkout happened (validation runs after checkout) — verify
        // the worker came back via guard drop on the early Err return.
        assert_eq!(
            pool.available(),
            1,
            "worker returned to pool after validation failure"
        );
    }

    // ──────────────── Phase 5.F.4: for_buckets() ────────────────

    /// Empty backbone + lm_head per bucket — exercises the
    /// constructor's lower→pool-build path without requiring a real
    /// `Instruction` to be constructible at the test site (the
    /// macro emits those at codegen time; pool tests stay structural).
    /// The resulting tape carries zero commands; the worker still
    /// bakes a (trivially empty) dispatch per bucket and the pool still
    /// stands one worker up eagerly.
    const EMPTY_BACKBONE: &[Instruction] = &[];

    /// Baked tape rungs for an empty test bucket — what the macro
    /// would emit for `(empty, empty)` instruction streams: an
    /// empty-command tape per (gen class × chunked) pair at KV cap 128.
    fn test_empty_tapes(
        bucket_m: u32,
        num_arena_slots: u32,
    ) -> &'static [crate::tape::lowered::ClassedTape] {
        use crate::tape::lowered::{ClassedTape, GenClass, LoweredMetalTape, baked};
        let tape = LoweredMetalTape {
            bucket_m,
            num_arena_slots,
            commands: crate::tape::lowered::TapeCommands::EMPTY,
            barrier_before: &[],
            // Straight-line: this fixture is one hand-built body, no rolled layer loop.
            loops: &[],
            splitk_scratch_bytes: 0,
            moe_scratch_bytes: 0,
            roped_k_scratch_bytes: 0,
            attn_unfused_scratch_bytes: 0,
        };
        let mut v = Vec::new();
        for gen_class in [GenClass::M1, GenClass::Mid, GenClass::M5] {
            for addressing in [KvAddressing::Direct, KvAddressing::Chunked] {
                v.push(ClassedTape {
                    gen_class,
                    addressing,
                    cap: MaxBlocksPerSeq(128),
                    tq_heads: None,
                    tape,
                });
            }
        }
        baked(v)
    }
    const EMPTY_LM_HEAD: &[Instruction] = &[];
    const TEST_ARENA_BYTES: &[u64] = &[4096, 4096];
    // One barrier flag per backbone/lm_head `Instruction`; both slices
    // are empty here so the barrier slices are empty too.
    const EMPTY_BARRIERS: &[bool] = &[];

    fn build_via_for_buckets(
        bucket_specs: &[MetalBucketSpec],
        max_workers: usize,
    ) -> Option<Result<MetalWorkerPool<TestWeights>, PoolBuildError>> {
        let device = crate::detect_device().filter(|_| crate::metal4_available())?;
        let device = Arc::new(device.device.clone());
        let (weights, allocator) = build_test_weights(&device);
        let runtime_factory: RuntimeFactory = RuntimeFactory::new(|d, _| empty_runtime(d, 1));
        Some(MetalWorkerPool::<TestWeights>::for_buckets(
            device,
            &weights,
            allocator,
            bucket_specs,
            runtime_factory,
            max_workers,
            None,
            128, // block_cap (test default — matches the legacy MAX_BLOCKS_PER_SEQ)
            KvAddressing::Direct,
            LentActivation::default(),
        ))
    }

    /// Empty bucket_specs surfaces `PoolBuildError::NoBuckets` —
    /// ahead of any Metal-device interaction so this test runs even
    /// on non-Apple hosts.
    #[test]
    fn for_buckets_rejects_empty_specs() {
        // Note: this path runs without a Metal device because the
        // `NoBuckets` check fires before any device call — no
        // silent-skip needed.
        let device = match crate::detect_device() {
            Some(d) => Arc::new(d.device.clone()),
            None => {
                eprintln!("skipping: no Metal device");
                return;
            }
        };
        let (weights, allocator) = build_test_weights(&device);
        let runtime_factory: RuntimeFactory = RuntimeFactory::new(|d, _| empty_runtime(d, 1));
        let res = MetalWorkerPool::<TestWeights>::for_buckets(
            device,
            &weights,
            allocator,
            &[],
            runtime_factory,
            1,
            None,
            128, // block_cap (unused — NoBuckets fires first)
            KvAddressing::Direct,
            LentActivation::default(),
        );
        match res {
            Err(PoolBuildError::NoBuckets) => {}
            Err(other) => panic!("expected NoBuckets, got Err({other})"),
            Ok(_) => panic!("expected NoBuckets, got Ok(_)"),
        }
    }

    /// Single empty bucket builds: lower_pair on `(empty, empty)`
    /// produces an empty-command tape with the right `bucket_m` and
    /// `num_arena_slots`; the worker bakes one no-op dispatch; the pool
    /// stands up a worker eagerly. Verifies the constructor wires
    /// pipeline cache / lowering / pool spawn end-to-end.
    #[test]
    fn for_buckets_builds_pool_for_single_empty_bucket() {
        let specs = [MetalBucketSpec {
            bucket_m: 1,
            num_arena_slots: 2,
            terminal_slot: 1,
            backbone_slot: 0,
            arena_bytes: TEST_ARENA_BYTES,
            backbone: EMPTY_BACKBONE,
            lm_head: EMPTY_LM_HEAD,
            backbone_barriers: EMPTY_BARRIERS,
            lm_head_barriers: EMPTY_BARRIERS,
            tapes: test_empty_tapes(1, 2),
            kernels: &[],
        }];
        let Some(res) = build_via_for_buckets(&specs, 2) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let pool = res.expect("for_buckets constructs pool");
        assert_eq!(pool.max_workers(), 2);
        assert_eq!(pool.current_size(), 1, "first worker eagerly created");
        assert_eq!(pool.available(), 1);
    }

    /// A bucket spec with NO baked tape rungs (what the macro emits
    /// for a canonical the front end refused — dense MoE on metal) must
    /// refuse pool construction with `NoRung`, NOT build a pool
    /// whose forward runs a no-op dispatch and whose tail then panics on
    /// `arena[terminal_slot]` index-OOB. Pins the documented refusal
    /// contract: "the POOL refuses at load".
    #[test]
    fn for_buckets_refuses_bucket_with_no_tape_variants() {
        let specs = [MetalBucketSpec {
            bucket_m: 1,
            num_arena_slots: 2,
            terminal_slot: 1,
            backbone_slot: 0,
            arena_bytes: TEST_ARENA_BYTES,
            backbone: EMPTY_BACKBONE,
            lm_head: EMPTY_LM_HEAD,
            backbone_barriers: EMPTY_BARRIERS,
            lm_head_barriers: EMPTY_BARRIERS,
            tapes: &[],
            kernels: &[],
        }];
        let Some(res) = build_via_for_buckets(&specs, 1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        match res {
            Err(PoolBuildError::NoRung { bucket_m, .. }) => {
                assert_eq!(bucket_m, 1, "refusal names the offending bucket");
            }
            Err(other) => panic!("expected NoRung, got Err({other})"),
            Ok(_) => panic!(
                "expected NoRung, got Ok(_) — a rung-less bucket built a pool; \
                 its forward would no-op and its tail would panic on arena index-OOB"
            ),
        }
    }

    /// Two empty buckets — verifies pick_bucket sees both
    /// `bucket_m`s and the constructor preserves spec order via
    /// `Arc<[…]>`.
    #[test]
    fn for_buckets_preserves_bucket_order() {
        let specs = [
            MetalBucketSpec {
                bucket_m: 1,
                num_arena_slots: 2,
                terminal_slot: 1,
                backbone_slot: 0,
                arena_bytes: TEST_ARENA_BYTES,
                backbone: EMPTY_BACKBONE,
                lm_head: EMPTY_LM_HEAD,
                backbone_barriers: EMPTY_BARRIERS,
                lm_head_barriers: EMPTY_BARRIERS,
                tapes: test_empty_tapes(1, 2),
                kernels: &[],
            },
            MetalBucketSpec {
                bucket_m: 8,
                num_arena_slots: 2,
                terminal_slot: 1,
                backbone_slot: 0,
                arena_bytes: TEST_ARENA_BYTES,
                backbone: EMPTY_BACKBONE,
                lm_head: EMPTY_LM_HEAD,
                backbone_barriers: EMPTY_BARRIERS,
                lm_head_barriers: EMPTY_BARRIERS,
                tapes: test_empty_tapes(8, 2),
                kernels: &[],
            },
        ];
        let Some(res) = build_via_for_buckets(&specs, 1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let pool = res.expect("for_buckets constructs pool");
        assert_eq!(pool.pick_bucket(1).unwrap(), 0);
        assert_eq!(pool.pick_bucket(8).unwrap(), 1);
        assert!(matches!(
            pool.pick_bucket(9),
            Err(ForwardError::NoBucketFits { .. })
        ));
    }
}
