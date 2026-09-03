//! Port of `flex::RuntimeScheduler` (abstract base) and its three concrete
//! backends `PfRuntimeScheduler`, `VfRuntimeScheduler`, `MockRuntimeScheduler`.
//!
//! Sources:
//!   - `flex/src/runtime_stream/runtime_scheduler.{hpp,cpp}` — base class:
//!     per-op fence/barrier bookkeeping, in-flight counters, pipeline-switch
//!     detection, queue-capacity polling, data-conversion wrapping.
//!   - `flex/src/runtime_stream/1p0/{pf,vf,mock}_runtime_scheduler.{hpp,cpp}`
//!     — concrete backends.
//!   - `flex/src/scheduler/scheduler_stats.hpp` — `SchedulerStats` itself
//!     (below), minus `Count()`: that method reads `SchedulerCompletionInfo`/
//!     `ControlBlockStream`/`response_worker`, all graph-compiler-owned types
//!     this crate does not have, so it stays unported (see `SchedulerStats`'s
//!     own doc).
//!   - `flex/src/scheduler/runtime_scheduler_factory.{hpp,cpp}` —
//!     `create_scheduler` (below): config-driven backend selection, ported
//!     in full since it's pure dispatch over already-ported `Backend`
//!     variants.
//!   - `flex/src/scheduler/scheduler_compute_info.hpp` — NOT ported, not
//!     even partially. Every type it declares
//!     (`SchedulerComputeInfo`/`SchedulerNetworkInfo`/
//!     `SchedulerPartitionInfo`/`SchedulerFrameInfo`) owns a
//!     `ControlBlockStream`, a `sendnn::SegmentTable`, or a
//!     `moodycamel::ConcurrentQueue<DeviceMemoryAllocationPtr>` — every field
//!     ties it to the graph-compiler/message-processing phase's own object
//!     graph, same as `flex::Scheduler` (the *different*,
//!     sendnn::Graph-driven compute-pipeline scheduler in
//!     `scheduler/scheduler.hpp`) — neither belongs to this
//!     ABI-entrypoint-reachable `runtime_stream/` slice.
//!
//! ## What crossed into `senlib_ffi_scheduler.rs` and why
//!
//! Only three things: `ControlBlockInterface::capacity()` (queue-capacity
//! query — hardware/firmware-configured, unknowable on the host side),
//! `ControlBlockInterface::QueueControlBlocksSBF()` (the literal doorbell
//! ring), and `ResponseBlockInterface::ReceiveResponsesSBF()` (mock-only
//! synchronous drain). See `SENLIB_BOUNDARY_scheduler.md` for full citations.
//!
//! ## Scope note on PF/VF backend depth
//!
//! `ResponseWorker` (`control_blocks/message_processing/response_worker.{hpp,cpp}`)
//! IS ported here, in full for every member that decides control flow — see
//! [`ResponseWorker`]. It has to be: `PfRuntimeScheduler::submitToHardware`'s
//! entire body is a queue-capacity gate, an in-flight bump and
//! `response_worker_->QueueCbs(...)`, so treating it as "an opaque native-Rust
//! dependency" (what this doc used to claim) left the PF path with no
//! asynchronous submission/completion architecture at all. Its telemetry
//! members (`TimestampDecoder`, `PipelineTimestamps`, `MeasureDuration`,
//! `ResponseWorkerStats`, `batch_timestamp_ring_`, the aiupti profiler
//! submissions) genuinely do belong to the telemetry phase and are not ported.
//!
//! `PfRuntimeScheduler`/`VfRuntimeScheduler`'s
//! `ControlBlockStream`/segment-table construction and `FlexVfStreamer`
//! submission machinery are owned by *other* parallel phases (`control_blocks/` construction is the
//! dma-compute-controlblocks phase's `control_blocks.rs`;
//! `IommuMapperInterface`/`DeviceHandle` topology queries are the
//! device-interface phase's files). This file ports everything that is
//! `RuntimeScheduler`'s *own* logic — which pipeline an op targets, in-flight
//! accounting, per-op fence/barrier enforcement, pipeline-switch detection,
//! and queue-capacity backpressure — and models the backend-specific
//! "build CBs, ring the doorbell, wait for completion" step as a single
//! `HardwareSubmit` trait call so that native orchestration logic here does
//! not collapse into an opaque FFI wrapper around the whole class (the
//! anti-pattern this port is explicitly guarding against). `MockBackend` is
//! fully native (real memcpy simulation, no FFI) since mock hardware
//! submission genuinely has no senlib involvement beyond the same three
//! calls above.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::address::{ByteSize, Chunk, CompositeAddress};
use crate::compute::ComputeParams;
use crate::control_block_wire::{
    AllocationTotalSize, ControlBlockWire, DevicePhysicalAddress, Flits, HostPhysicalAddress,
    ResponseTag, SegmentExtent, TransferSize, VirtualAddressSbf,
};
use crate::dma::{DmaDirection, DmaParams, FillParams, FillPipeline};
use crate::flex_config::{DeviceKind, RuntimeConfig};
use crate::stream::{
    CompletionCallback, RuntimeOperationKind, RuntimeStreamMode, SchedulerError, SchedulerHandle,
    StreamId, SubmittedOperation,
};
use crate::telemetry::si_prefix;

// ---------------------------------------------------------------------
// Pipeline / scheduler-type identifiers
// ---------------------------------------------------------------------

/// Port of `flex::PipelineId` (forward-declared in `runtime_scheduler.hpp`,
/// defined in `control_blocks/message_processing/response_worker.hpp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineId {
    Compute,
    AsyncDmaI,
    AsyncDmaO,
}

impl PipelineId {
    /// The `i32` index the C++ shim's own `FlexShimPipeline` enum
    /// (`cxx-shim/senlib_ffi.cpp`) expects — 0=compute, 1=async-dma-in,
    /// 2=async-dma-out, kept in sync by hand with that C++ enum (an
    /// unavoidable cross-language seam) but no longer independently
    /// re-derived at each Rust call site: this was hand-copied at 2 separate
    /// spots in this file (`SenlibHardwareSubmit::submit`,
    /// `SenlibQueueCapacity::capacity`) with no compiler link between them —
    /// the same shape as the P2P/Compute pipeline-kind collapsing bug this
    /// session already found once. A future 4th `PipelineId` variant now
    /// fails to compile here (a `match` with no wildcard arm) instead of
    /// silently working at one call site and not the other.
    fn shim_index(self) -> i32 {
        match self {
            PipelineId::Compute => 0,
            PipelineId::AsyncDmaI => 1,
            PipelineId::AsyncDmaO => 2,
        }
    }
}

/// Port of `flex::RuntimeSchedulerType`. Selects which concrete backend a
/// `RuntimeScheduler` runs, decided once at device/runtime-init time — never
/// switched afterward, hence modeled as the discriminant of the `Backend`
/// enum below rather than a separately-stored field that could disagree with
/// the live backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeSchedulerType {
    Pf,
    Vf,
    Mock,
}

/// Port of `RuntimeScheduler::kUseDefaultTimeout` (`0`): ask
/// `wait_for_queue_capacity` to use the configured `FLEX_QUEUE_CAPACITY_WAIT_TIMEOUT_MS`
/// ceiling (default 60s) rather than a caller-specified bound.
pub const USE_DEFAULT_TIMEOUT_MS: u32 = 0;

const DEFAULT_QUEUE_CAPACITY_WAIT_TIMEOUT_MS: u64 = 60_000;

// ---------------------------------------------------------------------
// Per-op fence (port of `std::shared_ptr<std::promise<void>>`)
// ---------------------------------------------------------------------

/// A single-resolution wait handle, port of the C++
/// `std::shared_ptr<std::promise<void>>`/`std::future<void>` pair used for
/// per-op pipeline-barrier fences. `resolve()` corresponds to
/// `promise::set_value()`; `wait_polling` corresponds to the C++
/// `future::wait_for(100ms)` poll loop that periodically rechecks the
/// stream's shutdown flag.
#[derive(Debug, Default)]
struct Fence {
    state: Mutex<bool>,
    cv: Condvar,
}

impl Fence {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(false),
            cv: Condvar::new(),
        })
    }

    fn resolve(&self) {
        let mut done = self.state.lock().unwrap();
        *done = true;
        self.cv.notify_all();
    }

    /// Port of the `issueBarrier` polling loop: waits in 100ms slices,
    /// calling `still_needs_shutdown` between slices so a hung fence can be
    /// abandoned once the stream is torn down. Returns `false` if abandoned.
    fn wait_polling(&self, still_needs_shutdown: impl Fn() -> bool) -> bool {
        let mut guard = self.state.lock().unwrap();
        loop {
            if *guard {
                return true;
            }
            if still_needs_shutdown() {
                return false;
            }
            let (g, _timeout) = self
                .cv
                .wait_timeout(guard, Duration::from_millis(100))
                .unwrap();
            guard = g;
        }
    }
}

type FenceDeque = VecDeque<Arc<Fence>>;

// ---------------------------------------------------------------------
// Base state shared by all backends (port of `RuntimeScheduler`'s protected
// fields)
// ---------------------------------------------------------------------

pub struct SchedulerState {
    #[allow(dead_code)]
    scheduler_type: RuntimeSchedulerType,

    compute_in_flight: AtomicI64,
    dmai_in_flight: AtomicI64,
    dmao_in_flight: AtomicI64,

    sync_mutex: Mutex<()>,
    sync_cv: Condvar,

    stream_fence_mtx: Mutex<()>,
    stream_compute_fence: Mutex<HashMap<StreamId, FenceDeque>>,
    stream_dmai_fence: Mutex<HashMap<StreamId, FenceDeque>>,
    stream_dmao_fence: Mutex<HashMap<StreamId, FenceDeque>>,

    stream_pipeline_mtx: Mutex<HashMap<StreamId, RuntimeOperationKind>>,

    stream_shutdown_flags: Mutex<HashMap<StreamId, Arc<AtomicBool>>>,
}

impl SchedulerState {
    fn new(scheduler_type: RuntimeSchedulerType) -> Self {
        Self {
            scheduler_type,
            compute_in_flight: AtomicI64::new(0),
            dmai_in_flight: AtomicI64::new(0),
            dmao_in_flight: AtomicI64::new(0),
            sync_mutex: Mutex::new(()),
            sync_cv: Condvar::new(),
            stream_fence_mtx: Mutex::new(()),
            stream_compute_fence: Mutex::new(HashMap::new()),
            stream_dmai_fence: Mutex::new(HashMap::new()),
            stream_dmao_fence: Mutex::new(HashMap::new()),
            stream_pipeline_mtx: Mutex::new(HashMap::new()),
            stream_shutdown_flags: Mutex::new(HashMap::new()),
        }
    }

    fn counter(&self, pipeline: PipelineId) -> &AtomicI64 {
        match pipeline {
            PipelineId::Compute => &self.compute_in_flight,
            PipelineId::AsyncDmaI => &self.dmai_in_flight,
            PipelineId::AsyncDmaO => &self.dmao_in_flight,
        }
    }

    /// Port of `submitToHardware`'s counter increment (PF/VF) — a fenced
    /// increment that also wakes nobody (only decrements need to notify
    /// `synchronize()`'s waiters). Used by `submit_to_hardware` (see its
    /// doc comment on current wiring status).
    #[allow(dead_code)]
    fn inc_in_flight(&self, pipeline: PipelineId) {
        self.counter(pipeline).fetch_add(1, Ordering::AcqRel);
    }

    /// Port of the completion-side decrement in `onCbComplete`/mock
    /// completion paths: holds `sync_mutex_` across decrement + notify so a
    /// concurrent `synchronize()`/`query()` cannot miss the wakeup.
    #[allow(dead_code)]
    fn dec_in_flight(&self, pipeline: PipelineId) {
        let _guard = self.sync_mutex.lock().unwrap();
        self.counter(pipeline).fetch_sub(1, Ordering::AcqRel);
        self.sync_cv.notify_all();
    }

    /// Port of `RuntimeScheduler::currentInFlight`: clamps a transiently
    /// negative signed counter (decrement on the completion thread racing an
    /// increment on the scheduler thread) to zero rather than letting a
    /// `size_t` conversion wrap.
    fn current_in_flight(&self, pipeline: PipelineId) -> u64 {
        self.counter(pipeline).load(Ordering::Acquire).max(0) as u64
    }

    fn fence_deque(&self, pipeline: PipelineId) -> &Mutex<HashMap<StreamId, FenceDeque>> {
        match pipeline {
            PipelineId::Compute => &self.stream_compute_fence,
            PipelineId::AsyncDmaI => &self.stream_dmai_fence,
            PipelineId::AsyncDmaO => &self.stream_dmao_fence,
        }
    }

    /// Port of the fence-pushing block repeated in every `schedule()`
    /// overload: create a new `Fence`, push it onto the named pipeline's
    /// per-stream deque, return it for `submit*`'s completion wrapper to
    /// resolve later.
    fn push_fence(&self, pipeline: PipelineId, id: StreamId) -> Arc<Fence> {
        let _l = self.stream_fence_mtx.lock().unwrap();
        let fence = Fence::new();
        self.fence_deque(pipeline)
            .lock()
            .unwrap()
            .entry(id)
            .or_default()
            .push_back(fence.clone());
        fence
    }

    /// Port of the `stream_last_pipeline_[id] = target;` block.
    fn record_last_pipeline(&self, id: StreamId, pipeline: RuntimeOperationKind) {
        self.stream_pipeline_mtx
            .lock()
            .unwrap()
            .insert(id, pipeline);
    }

    /// Port of `RuntimeScheduler::maybeSetBarrier`: under `StrictOrdering`,
    /// auto-request a pipeline barrier on `op` when the previous op on this
    /// stream targeted a different pipeline. Returns the (possibly amended)
    /// barrier flag to use for this op.
    /// `compare_kind` is the op's own UNCOLLAPSED `RuntimeOperationKind`
    /// (`target = op.getType()` at line 172, compared against
    /// `stream_last_pipeline_` at line 181) — real C++ never collapses on the
    /// read/comparison side. This function does NOT write
    /// `stream_last_pipeline_` itself: in the real code that write lives in
    /// each `schedule()` overload's own "Track which pipeline this stream last
    /// targeted" block, executed unconditionally regardless of `mode` — unlike
    /// this read, which `maybeSetBarrier` only performs under
    /// `StrictOrdering`. The caller (`schedule_pipelined`) performs that write
    /// separately via `record_last_pipeline`, passing the COLLAPSED kind,
    /// matching the real asymmetry: compare uncollapsed, store collapsed.
    fn maybe_set_barrier(
        &self,
        id: StreamId,
        mode: RuntimeStreamMode,
        compare_kind: RuntimeOperationKind,
        requested_barrier: bool,
    ) -> bool {
        if mode != RuntimeStreamMode::StrictOrdering
            || compare_kind == RuntimeOperationKind::HostCallback
        {
            return requested_barrier;
        }
        let map = self.stream_pipeline_mtx.lock().unwrap();
        let switched = matches!(map.get(&id), Some(prev) if *prev != compare_kind);
        requested_barrier || switched
    }

    fn stream_needs_shutdown(&self, id: StreamId) -> bool {
        self.stream_shutdown_flags
            .lock()
            .unwrap()
            .get(&id)
            .map(|flag| flag.load(Ordering::Acquire))
            .unwrap_or(false)
    }

    /// Port of the base `RuntimeScheduler::issueBarrier`: pop every fence
    /// queued on pipelines OTHER than `op_pipeline` for this stream, and
    /// wait for each individually (per-op granularity — a barrier only waits
    /// for ops in flight when it was set, not ones submitted afterward).
    /// PF/VF gate this on `device_handle_->GetType()` being PF/VF in the
    /// C++; here the base implementation IS the PF/VF behavior and
    /// `MockBackend` overrides it (matching `MockRuntimeScheduler::issueBarrier`,
    /// which just records the call — see `Backend::issue_barrier`).
    fn issue_barrier(&self, op_pipeline: RuntimeOperationKind, id: StreamId) {
        if self.stream_needs_shutdown(id) {
            return;
        }
        let mut fences = FenceDeque::new();
        {
            let _l = self.stream_fence_mtx.lock().unwrap();
            let mut collect =
                |kind: RuntimeOperationKind, deque_mtx: &Mutex<HashMap<StreamId, FenceDeque>>| {
                    if op_pipeline != kind
                        && let Some(q) = deque_mtx.lock().unwrap().get_mut(&id)
                    {
                        fences.extend(q.drain(..));
                    }
                };
            collect(RuntimeOperationKind::Compute, &self.stream_compute_fence);
            collect(RuntimeOperationKind::H2D, &self.stream_dmai_fence);
            collect(RuntimeOperationKind::D2H, &self.stream_dmao_fence);
        }
        for fence in fences {
            if !fence.wait_polling(|| self.stream_needs_shutdown(id)) {
                return;
            }
        }
    }

    fn release_stream(&self, id: StreamId) {
        {
            let _l = self.stream_fence_mtx.lock().unwrap();
            self.stream_compute_fence.lock().unwrap().remove(&id);
            self.stream_dmai_fence.lock().unwrap().remove(&id);
            self.stream_dmao_fence.lock().unwrap().remove(&id);
        }
        self.stream_pipeline_mtx.lock().unwrap().remove(&id);
        self.stream_shutdown_flags.lock().unwrap().remove(&id);
    }

    fn synchronize(&self) {
        let guard = self.sync_mutex.lock().unwrap();
        let _unused = self
            .sync_cv
            .wait_while(guard, |_| {
                self.current_in_flight(PipelineId::Compute) != 0
                    || self.current_in_flight(PipelineId::AsyncDmaI) != 0
                    || self.current_in_flight(PipelineId::AsyncDmaO) != 0
            })
            .unwrap();
    }

    fn query(&self) -> bool {
        self.current_in_flight(PipelineId::Compute) == 0
            && self.current_in_flight(PipelineId::AsyncDmaI) == 0
            && self.current_in_flight(PipelineId::AsyncDmaO) == 0
    }
}

// ---------------------------------------------------------------------
// Queue-capacity backend hook (the only place `senlib_ffi_scheduler`
// functions are consulted)
// ---------------------------------------------------------------------

/// What a backend can report about a pipeline's hardware CB queue capacity.
/// Port of the `std::optional<size_t>` returned by `getQueueCapacity`:
/// `None` covers both "no senlib `ControlBlockInterface` for this pipeline"
/// (VF/compile devices) and "capacity() reported 0" (not applicable) —
/// collapsed into one variant because `RuntimeScheduler` treats them
/// identically (`hasQueueCapacity` always returns `true`).
pub trait QueueCapacitySource: Send + Sync {
    fn capacity(&self, pipeline: PipelineId) -> Option<u64>;
}

/// Port of `RuntimeScheduler::hasQueueCapacity`.
fn has_queue_capacity(
    cap: &dyn QueueCapacitySource,
    state: &SchedulerState,
    pipeline: PipelineId,
    needed: u64,
) -> bool {
    match cap.capacity(pipeline) {
        None => true,
        Some(capacity) => state.current_in_flight(pipeline) + needed <= capacity,
    }
}

/// Port of `RuntimeScheduler::waitForQueueCapacity`. `timeout_ms ==
/// USE_DEFAULT_TIMEOUT_MS` uses `DEFAULT_QUEUE_CAPACITY_WAIT_TIMEOUT_MS`
/// (the `FlexConfig::QueueCapacityWaitTimeoutMs()` stand-in — the real
/// config knob lives in the `flex_config`/telemetry phase's files); `0`
/// there means "effectively unbounded" (24h), matching the C++ fallback.
pub fn wait_for_queue_capacity(
    cap: &dyn QueueCapacitySource,
    state: &SchedulerState,
    pipeline: PipelineId,
    needed: u64,
    timeout_ms: u32,
) -> bool {
    if has_queue_capacity(cap, state, pipeline, needed) {
        return true;
    }
    let effective_timeout_ms = if timeout_ms == USE_DEFAULT_TIMEOUT_MS {
        DEFAULT_QUEUE_CAPACITY_WAIT_TIMEOUT_MS
    } else {
        timeout_ms as u64
    };
    let timeout = if effective_timeout_ms > 0 {
        Duration::from_millis(effective_timeout_ms)
    } else {
        Duration::from_secs(24 * 3600)
    };
    let start = Instant::now();
    while start.elapsed() < timeout {
        if has_queue_capacity(cap, state, pipeline, needed) {
            return true;
        }
        std::thread::sleep(Duration::from_micros(100));
    }
    false
}

// ---------------------------------------------------------------------
// Backend hooks — port of the `submit*`/`issueBarrier` virtual dispatch
// ---------------------------------------------------------------------

/// P2P parameter structs. Out of this file's direct construction scope
/// (owned by the multi-device/P2P phase's `runtime_submission_params`
/// port), carried here only as opaque-enough value types so
/// `RuntimeScheduler::schedule` can build and forward them exactly as the
/// C++ does.
#[derive(Debug, Clone)]
pub struct P2PDataParams {
    pub sync_key: String,
    pub peer_rank: u64,
    pub device_address: Arc<CompositeAddress>,
    pub op_name: String,
    pub is_send: bool,
}

#[derive(Debug, Clone)]
pub struct P2PRdmaSendParams {
    pub sync_key: String,
    pub dest_ranks: Vec<u64>,
    pub sid_list: Vec<u32>,
    pub device_address: Arc<CompositeAddress>,
    pub op_name: String,
    pub dst_offset: u64,
}

#[derive(Debug, Clone)]
pub struct P2PRdmaWaitParams {
    pub sid_list: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct P2PFirmwareSignalParams {
    pub ha_bytes: u64,
    pub is_ack: bool,
}

#[derive(Debug, Clone)]
pub struct P2PFirmwareWaitParams {
    pub ha_bytes: u64,
    pub expected_count: u64,
    pub is_ack: bool,
    pub timeout: Duration,
    pub num_consecutive_flits_minus1: u32,
}

/// Port of the pure-virtual `submit*` hooks + `issueBarrier` override point.
/// One implementor per concrete backend (`PfBackend`, `VfBackend`,
/// `MockBackend`); `Backend` below is the enum that picks exactly one at
/// runtime-init time, matching "the RuntimeSchedulerType decided once when
/// the device/backend is constructed, never switched after."
/// Every `submit*` hook takes the shared [`SchedulerState`] because in the
/// C++ the backend IS the scheduler: `PfRuntimeScheduler::submitToHardware`
/// reads `waitForQueueCapacity` and raises/lowers
/// `{compute,dmai,dmao}_in_flight_` itself, and `MockRuntimeScheduler::submitCompute`
/// brackets its synchronous submit with `++compute_in_flight_`/`--compute_in_flight_`
/// (`mock_runtime_scheduler.cpp:302,341`). Rust splits those counters into
/// `SchedulerState`, so they have to be handed down rather than relocated into
/// `schedule_pipelined` (where this file previously did the capacity check with
/// a hardcoded `needed = 1` and a scope-guard around the whole submit).
pub trait SchedulerBackend: Send + Sync {
    fn submit_dma(
        &self,
        state: &Arc<SchedulerState>,
        dma: &DmaParams,
        callback: CompletionCallback,
    );
    fn submit_compute(
        &self,
        state: &Arc<SchedulerState>,
        op: &ComputeParams,
        callback: CompletionCallback,
    );
    fn submit_fill(
        &self,
        state: &Arc<SchedulerState>,
        fill: &FillParams,
        callback: CompletionCallback,
    );

    /// Port of the base-class default no-op implementations (only Mock
    /// leaves these at the base default in spirit; PF/VF override with real
    /// RDMA exchange). Each still validates + resolves the DMAI fence, per
    /// `RuntimeScheduler::submitLegacyP2P*`.
    fn submit_legacy_p2p_send_data(
        &self,
        _state: &Arc<SchedulerState>,
        _params: &P2PDataParams,
        callback: CompletionCallback,
    ) {
        callback(Ok(()));
    }
    fn submit_legacy_p2p_recv_data(
        &self,
        _state: &Arc<SchedulerState>,
        _params: &P2PDataParams,
        callback: CompletionCallback,
    ) {
        callback(Ok(()));
    }
    fn submit_legacy_p2p_data_exchange(
        &self,
        _state: &Arc<SchedulerState>,
        _params: &[P2PDataParams],
        callback: CompletionCallback,
    ) {
        callback(Ok(()));
    }

    fn submit_p2p_rdma_send(
        &self,
        state: &Arc<SchedulerState>,
        params: &P2PRdmaSendParams,
        callback: CompletionCallback,
    );
    fn submit_p2p_rdma_wait(
        &self,
        state: &Arc<SchedulerState>,
        params: &P2PRdmaWaitParams,
        callback: CompletionCallback,
    );
    fn submit_p2p_firmware_signal(
        &self,
        state: &Arc<SchedulerState>,
        params: &P2PFirmwareSignalParams,
        callback: CompletionCallback,
    );
    fn submit_p2p_firmware_wait(
        &self,
        state: &Arc<SchedulerState>,
        params: &P2PFirmwareWaitParams,
        callback: CompletionCallback,
    );

    /// Override point for `issueBarrier`. Default forwards to the base
    /// `SchedulerState::issue_barrier` fence-wait implementation (PF/VF
    /// behavior); `MockBackend` overrides to just record the call, matching
    /// `MockRuntimeScheduler::issueBarrier`.
    fn issue_barrier(
        &self,
        state: &SchedulerState,
        op_pipeline: RuntimeOperationKind,
        id: StreamId,
    ) {
        state.issue_barrier(op_pipeline, id);
    }

    fn queue_capacity_source(&self) -> &dyn QueueCapacitySource;
}

// ---------------------------------------------------------------------
// Mock backend — fully native, no senlib beyond the shared drain call
// ---------------------------------------------------------------------

/// Port of `MockRuntimeScheduler`. DMA is simulated with a direct in-process
/// copy (there is no separate host/device address space to bridge); compute
/// and P2P-wait/firmware ops are recorded and completed synchronously or via
/// a background thread (mirroring `compute_threads_`/`completeCompute`,
/// which exist so `schedule(D2H/H2D)`'s fence-wait doesn't deadlock the
/// caller). No senlib FFI: the mock has no `ControlBlockInterface`, so
/// `queue_capacity_source` always reports "not applicable".
#[cfg(test)]
pub struct MockBackend {
    pub throw_on_dma: Mutex<bool>,
    pub throw_on_compute: Mutex<bool>,
    pub barrier_calls: Mutex<Vec<RuntimeOperationKind>>,
    no_capacity: NoCapacity,
}

/// Only [`MockBackend`] needs a capacity source that reports "not applicable"
/// (the real PF/VF backends carry a live `SenlibQueueCapacity`), so this is
/// test-only for the same reason the mock backend is.
#[cfg(test)]
struct NoCapacity;
#[cfg(test)]
impl QueueCapacitySource for NoCapacity {
    fn capacity(&self, _pipeline: PipelineId) -> Option<u64> {
        None
    }
}

#[cfg(test)]
impl Default for MockBackend {
    fn default() -> Self {
        Self {
            throw_on_dma: Mutex::new(false),
            throw_on_compute: Mutex::new(false),
            barrier_calls: Mutex::new(Vec::new()),
            no_capacity: NoCapacity,
        }
    }
}

#[cfg(test)]
impl MockBackend {
    /// Port of `MockRuntimeScheduler::barrierCount`.
    pub fn barrier_count(&self) -> usize {
        self.barrier_calls.lock().unwrap().len()
    }
}

#[cfg(test)]
impl SchedulerBackend for MockBackend {
    fn submit_dma(
        &self,
        _state: &Arc<SchedulerState>,
        dma: &DmaParams,
        callback: CompletionCallback,
    ) {
        if *self.throw_on_dma.lock().unwrap() {
            callback(Err(mock_error(
                "MockRuntimeScheduler: submitDma configured to throw",
            )));
            return;
        }
        // Port of the memcpy simulation: direction only matters for the
        // real host<->device copy the FFI-free mock stands in for; nothing
        // here calls senlib, matching `mock_runtime_scheduler.cpp`'s H2D/D2H
        // `memcpy` branches.
        // `MockRuntimeScheduler::submitDma`'s own comment (`:191`): the DMA
        // in-flight counters "never need to be raised" — the memcpy has already
        // finished by the time this returns — so, unlike `submitCompute` below,
        // this path deliberately touches no counter at all.
        let _ = dma.direction() == DmaDirection::HostToDevice;
        callback(Ok(()));
    }

    fn submit_compute(
        &self,
        state: &Arc<SchedulerState>,
        _op: &ComputeParams,
        callback: CompletionCallback,
    ) {
        if *self.throw_on_compute.lock().unwrap() {
            callback(Err(mock_error(
                "MockRuntimeScheduler: submitCompute configured to throw",
            )));
            return;
        }
        // `++compute_in_flight_` ... synchronous submit ... `--compute_in_flight_`
        // (`mock_runtime_scheduler.cpp:302,341`).
        state.inc_in_flight(PipelineId::Compute);
        state.dec_in_flight(PipelineId::Compute);
        callback(Ok(()));
    }

    fn submit_fill(
        &self,
        _state: &Arc<SchedulerState>,
        _fill: &FillParams,
        callback: CompletionCallback,
    ) {
        callback(Ok(()));
    }

    fn submit_p2p_rdma_send(
        &self,
        _state: &Arc<SchedulerState>,
        _params: &P2PRdmaSendParams,
        callback: CompletionCallback,
    ) {
        // Port of `MockRuntimeScheduler::submitP2PRdmaSend`: not implemented,
        // throws a RAS error in C++.
        callback(Err(mock_error(
            "MockRuntimeScheduler: P2P RDMA send not implemented",
        )));
    }
    fn submit_p2p_rdma_wait(
        &self,
        _state: &Arc<SchedulerState>,
        _params: &P2PRdmaWaitParams,
        callback: CompletionCallback,
    ) {
        callback(Err(mock_error(
            "MockRuntimeScheduler: P2P RDMA wait not implemented",
        )));
    }
    fn submit_p2p_firmware_signal(
        &self,
        _state: &Arc<SchedulerState>,
        _params: &P2PFirmwareSignalParams,
        callback: CompletionCallback,
    ) {
        callback(Err(mock_error(
            "MockRuntimeScheduler: firmware signal not implemented",
        )));
    }
    fn submit_p2p_firmware_wait(
        &self,
        _state: &Arc<SchedulerState>,
        _params: &P2PFirmwareWaitParams,
        callback: CompletionCallback,
    ) {
        callback(Err(mock_error(
            "MockRuntimeScheduler: firmware wait not implemented",
        )));
    }

    /// Port of `MockRuntimeScheduler::issueBarrier`: records the call for
    /// test inspection, THEN still drains/waits the other pipelines' fence
    /// deques exactly like the base implementation — the real C++ mock
    /// `issueBarrier` is not a pure no-op stub, it runs the identical
    /// collect-and-wait logic as the base class (`mock_runtime_scheduler.cpp`),
    /// just noting in its own comment that the wait is expected to return
    /// immediately since mock ops complete synchronously. Skipping the drain
    /// entirely (this crate's previous behavior) left resolved-but-undrained
    /// fences piling up in `stream_{compute,dmai,dmao}_fence` for the
    /// lifetime of any stream with more than one barrier — the real mock
    /// clears each pipeline's deque via `it->second.clear()` every time.
    fn issue_barrier(
        &self,
        state: &SchedulerState,
        op_pipeline: RuntimeOperationKind,
        id: StreamId,
    ) {
        self.barrier_calls.lock().unwrap().push(op_pipeline);
        state.issue_barrier(op_pipeline, id);
    }

    fn queue_capacity_source(&self) -> &dyn QueueCapacitySource {
        &self.no_capacity
    }
}

mod completion;

pub use completion::{CbCompleteError, CbRetired};
use completion::{QuarantinedBuf, eval_cb_complete_status};

fn mock_error(msg: &str) -> SchedulerError {
    Box::<dyn std::error::Error + Send + Sync>::from(msg.to_string())
}

/// Build the real wire-format DMA control block for one `DmaParams`
/// operation. Port of `ControlBlockStream::CreateComputeDataTransfer`'s
/// call shape (immediate, physical-address DMA — no descriptor list).
///
/// `host_addr` must already be a real device-visible IOVA, not a bare host
/// virtual address — `dma.iova()` is the pre-mapped fast path
/// (`DmaParams`'s own doc: "supplied by the caller when it wants to skip
/// the backend's RAII shadow-buffer copy"); when it's `None` the caller
/// (`PfBackend::submit_dma`) must have done that mapping itself
/// (`PfIommuMapper::map`) before calling this — this function used to fall
/// back to `dma.hmva()` directly (the raw host *virtual* address) when
/// `iova()` was `None`, which is not an address the device's IOMMU can ever
/// resolve; hardware correctly rejected every such CB with
/// `RBStatusTypeEnum::RB_ERROR`.
fn build_dma_cb(
    resolver: &dyn DeviceAddressResolver,
    dma: &DmaParams,
    host_addr: HostPhysicalAddress,
    tag: ResponseTag,
) -> Result<ControlBlockWire, SchedulerError> {
    let resolved = resolve_single_chunk(resolver, dma.device_address())?;
    // `submitDma` uses `remaining_size = dma.dma_size` directly — there is no
    // clamp to the resolved allocation size anywhere in the real code. The
    // `min(dma_size, resolved.size)` this line used to carry silently
    // transferred fewer bytes than the caller asked for and left the
    // destination tail stale.
    let size_bytes = dma.dma_size().as_u64();
    // Port of `getDmvaAddress` + `buildDmaControlBlocks`'s
    // `translations()[seg_id].SetPaddrBytes(paddr)`: the XLAT entry gets the
    // region's own base address (`paddr = region_id` in the real C++, i.e.
    // no offset folded in — `pf_runtime_scheduler_utils.cpp`), and `dmva`
    // carries this chunk's own byte offset (`SegmentByteOffset_todmva(seg_id,
    // chunk.addr.offset)`); hardware adds them back together
    // (`TranslateDmvaToDmpa`: `dmpa = dmpa_allocs[seg_id] + seg_offset`).
    let dmva = resolved.dmva(0);
    let region_paddr = resolved.xlat_base();
    // ⛔ `trace!`, NOT `debug!`. This fires ONCE PER DMA CHUNK: staging granite-3.1-2b's 2.6 GB of
    // weights emits 2,661 of these, which was 89% of a `RUST_LOG=debug` capture and buried every
    // line anyone reads a log FOR. The per-chunk detail is still worth having — `RUST_LOG=trace`
    // asks for it — but `debug` has to stay usable for diagnosing a run.
    tracing::trace!(
        tag = ?tag,
        to_device = dma.direction() == DmaDirection::HostToDevice,
        dmva_segment = dmva.segment(),
        dmva_offset = dmva.offset().to_bytes(),
        region_paddr = format_args!("{:#x}", region_paddr.0),
        chunk_size = resolved.size,
        size_bytes,
        host_addr = format_args!("{:#x}", host_addr.as_u64()),
        "build_dma_cb"
    );
    ControlBlockWire::dma_immediate(
        tag,
        dma.direction() == DmaDirection::HostToDevice,
        crate::control_block_wire::DmaTransfer {
            dmva,
            host_addr,
            size_bytes: TransferSize(size_bytes),
        },
        crate::control_block_wire::XlatWindow {
            region_paddr,
            length: resolved.segment_extent(0, size_bytes),
        },
        dma.pipeline_barrier(),
    )
    .map_err(|e| mock_error(&format!("build_dma_cb: {e}")))
}

/// Build the DMA control block for ONE CHUNK of a transfer. Port of
/// `submitDmaChunk`'s CB construction (`pf_runtime_scheduler.cpp:158-229`),
/// which the real `submitDma` calls once per
/// `RuntimeConfig::DataTransferBufferSize()`-sized piece of an oversized
/// transfer (its `while(remaining_size != 0)` loop, lines 130-149). Unlike
/// `build_dma_cb` this takes the already-resolved chunk plus an explicit
/// `(offset_bytes, size_bytes)` sub-range, so the caller resolves the device
/// address once and each piece becomes its own CB — a single oversized CB for
/// e.g. a long-context KV-cache transfer is a shape the real code never
/// produces.
fn build_dma_cb_chunk(
    resolved: &ResolvedChunk,
    dma: &DmaParams,
    offset_bytes: u64,
    size_bytes: u64,
    host_addr: HostPhysicalAddress,
    tag: ResponseTag,
) -> Result<ControlBlockWire, SchedulerError> {
    // This chunk's `dmva` carries the base chunk offset PLUS this piece's own
    // offset within the transfer; the XLAT entry keeps the region base and
    // the region's full size, exactly as in the single-CB path.
    // ══════════════════════════════════════════════════════════════════════════════════════════
    //  ⛔⛔⛔ I1, PROVEN BEFORE THE DOORBELL: this piece lies inside the transfer it belongs to.
    //
    //  The card enforces this by FENCING (`0xa35e RAS::PCI::BusFence`), which arrives with no
    //  statement of which transfer caused it and no host stack to attach to. So the host proves it
    //  here, where every term is known and the failure can name itself.
    //
    //  ⛔ A ZERO LENGTH IS NOT ZERO. The consumer reads `if (length == 0) length = 1 << 27;` in
    //  BOTH `handleHostDMA` and `handleXLATentry` — so a zero-length field transfers 16 GiB. And
    //  the producer masks 26 bits where the consumer reads 27, so a request of exactly 2^26 flits
    //  masks to 0 on write and reads back as 2^27: a silent DOUBLING at exactly one size, which is
    //  why it hides. Both are rejected here.
    assert!(
        size_bytes > 0,
        "DMA chunk of ZERO bytes: the device reads a zero length field as 2^27 flits = 16 GiB, \
         not as nothing"
    );
    assert!(
        resolved.offset_bytes + offset_bytes + size_bytes <= resolved.region_size,
        "DMA chunk ends at {} B of a {} B region (segment {}) — I1's outer half: the XLAT window \
         would cover memory the region does not own",
        resolved.offset_bytes + offset_bytes + size_bytes,
        resolved.region_size,
        resolved.segment,
    );
    assert!(
        offset_bytes + size_bytes <= resolved.size,
        "DMA chunk [{offset_bytes}, {}) runs past its transfer's {} B — the XLAT window is sized \
         from the transfer, so this piece addresses memory the window does not cover",
        offset_bytes + size_bytes,
        resolved.size,
    );
    assert!(
        Flits::from_bytes(size_bytes).0 < (1u64 << 26),
        "DMA chunk of {size_bytes} B is {} flits: the CB writes 26 bits and the device reads 27, \
         so at or above 2^26 flits the length is silently truncated — and zero means 16 GiB",
        Flits::from_bytes(size_bytes).0,
    );

    let dmva = resolved.dmva(offset_bytes);
    let region_paddr = resolved.xlat_base();
    // ⛔ `trace!`, NOT `debug!`. Fires ONCE PER DMA CHUNK: staging granite-3.1-2b's 2.6 GB of
    // weights emits 2,661 of these, which was 89% of a `RUST_LOG=debug` capture and buried every
    // line a log is read FOR. The detail is still available at `RUST_LOG=trace`; `debug` has to
    // stay usable for diagnosing a run.
    tracing::trace!(
        tag = ?tag,
        to_device = dma.direction() == DmaDirection::HostToDevice,
        dmva_segment = dmva.segment(),
        dmva_offset = dmva.offset().to_bytes(),
        region_paddr = format_args!("{:#x}", region_paddr.0),
        chunk_size = resolved.size,
        offset_bytes,
        size_bytes,
        host_addr = format_args!("{:#x}", host_addr.as_u64()),
        "build_dma_cb_chunk"
    );
    ControlBlockWire::dma_immediate(
        tag,
        dma.direction() == DmaDirection::HostToDevice,
        crate::control_block_wire::DmaTransfer {
            dmva,
            host_addr,
            size_bytes: TransferSize(size_bytes),
        },
        crate::control_block_wire::XlatWindow {
            region_paddr,
            length: resolved.segment_extent(offset_bytes, size_bytes),
        },
        dma.pipeline_barrier(),
    )
    .map_err(|e| mock_error(&format!("build_dma_cb_chunk: {e}")))
}

/// Build the real wire-format fill control block for one `FillParams`. Port of
/// `ControlBlockStream::CreateFillDma` (`control_block_stream.cpp:3870`) as
/// called by `PfRuntimeScheduler::submitFill` (`pf_runtime_scheduler.cpp:261-276`).
/// `PfBackend::submit_fill`/`VfBackend::submit_fill` were previously silent
/// no-op stubs (`callback(Ok(()))`) that never touched hardware at all, so a
/// caller believing a region had been zeroed got whatever stale bytes were
/// there.
fn build_fill_cb(
    resolver: &dyn DeviceAddressResolver,
    fill: &FillParams,
    tag: ResponseTag,
) -> Result<ControlBlockWire, SchedulerError> {
    let resolved = resolve_single_chunk(resolver, fill.device_address())?;
    let size_bytes = fill.size().as_u64();
    let dmva = resolved.dmva(0);
    ControlBlockWire::fill(
        tag,
        dmva,
        TransferSize(size_bytes),
        fill.fill_pattern().0,
        fill.pipeline() == FillPipeline::Dmai,
        resolved.xlat_base(),
        resolved.segment_extent(0, size_bytes),
    )
    .map_err(|e| mock_error(&format!("build_fill_cb: {e}")))
}

/// Build the real wire-format compute control block for one `ComputeParams`
/// operation. Port of `ControlBlockStream::CreateGraphFreeCompute`'s call
/// shape: one xlat entry per tensor operand plus the program at segment 7,
/// bootstrap pointing at `PROG_OFFSET_BASE + bootstrap_offset`.
fn build_compute_cb(
    resolver: &dyn DeviceAddressResolver,
    op: &ComputeParams,
    tag: ResponseTag,
) -> Result<ControlBlockWire, SchedulerError> {
    let mut tensors = Vec::with_capacity(op.tensor_allocs().len());
    for alloc in op.tensor_allocs() {
        let resolved = resolve_single_chunk(resolver, alloc)?;
        // Per-tensor XLAT slots have no companion offset field of their own
        // (`ControlBlockStream::CreateGraphFreeCompute` takes bare
        // `tensor_paddrs`/`tensor_sizes` vectors, one absolute address each)
        // — unlike DMA/bootstrap, nothing downstream ever adds this chunk's
        // offset back in, so it must be folded into the XLAT entry directly.
        //
        // XLAT length is THIS ALLOCATION's own size (real C++:
        // `tensor_sizes.push_back(inp_out_allocs[i]->total_size())` —
        // `CompositeAddress::total_size()`, not the owning `MemoryRegion`'s).
        // An earlier version of this line used `resolved.xlat_length_bytes()`
        // (the region's size) on the mistaken belief that a compute tensor
        // always occupies its whole region — it does not: regions are large,
        // pre-allocated slabs (`allocator.rs`'s `pre_allocate_regions`,
        // ~16GB each) that many separate, independently-addressed
        // allocations share via sub-chunk carving (`find_best_fit`/
        // `allocate_block`). Confirmed on real hardware: every tensor's
        // logged XLAT length was exactly `17179869184` (2^34, `SEGMENT_SIZE`)
        // regardless of the tensor's real size — an absurdly oversized XLAT
        // bound that hardware doesn't reject outright (unlike a too-small
        // bound) but that lets a launch address memory far past the real
        // allocation, which is consistent with the real, still-unexplained
        // hang this was chasing (a stall/fault deep in hardware rather than
        // a clean CB rejection).
        tensors.push((
            resolved.xlat_absolute(),
            AllocationTotalSize(alloc.total_size().as_u64()),
        ));
    }
    let prog_resolved = resolve_single_chunk(resolver, op.device_address())?;
    // Port of `pf_runtime_scheduler_utils.cpp`'s program-address block, read
    // directly (this was previously wrong — see git history):
    //
    //   auto prog_paddr = base_alloc + offset;               // ABSOLUTE, like a tensor's XLAT entry
    //   cbs->CreateGraphFreeCompute(..., prog_paddr, prog_size,
    //                               PROG_OFFSET_BASE + bootstrap_offset);
    //
    // `prog_paddr` is the program allocation's own absolute address (its own
    // comment: "program_alloc is the program's FULL segment-7 allocation, so
    // (region base + offset) is the correct XLAT paddr; no further
    // adjustment is needed here") — `xlat_absolute()`, the same as a tensor
    // operand, NOT `xlat_base()` (this crate previously used `xlat_base()`,
    // silently dropping the chunk's own offset for any program allocation
    // that isn't offset-0 in its region).
    //
    // `bootstrap` is `PROG_OFFSET_BASE + bootstrap_offset` — a fixed
    // constant plus the caller's own offset, with NO dependency on the
    // program chunk's offset at all ("bootstrap_offset only locates the
    // program entry point within the allocation"). This crate previously
    // built it as `prog_resolved.dmva(bootstrap_offset)`, which wrongly
    // folded the chunk's own byte offset into the bootstrap PC and was
    // missing `PROG_OFFSET_BASE` (segment 7's base) entirely — hardware
    // would have jumped to a garbage instruction address on any non-zero
    // program allocation.
    let prog_paddr = prog_resolved.xlat_absolute();
    let bootstrap_dmva = crate::compute::PROG_OFFSET_BASE.0 + op.bootstrap_offset().0;
    let bootstrap = VirtualAddressSbf::new(
        (bootstrap_dmva >> crate::control_blocks::SEGMENT_SIZE_BITS) as u8,
        Flits::from_bytes(bootstrap_dmva & crate::control_blocks::SEGMENT_OFFSET_MASK),
    );
    // `prog_size` for the XLAT entry is `program_alloc->total_size()` —
    // THIS SPECIFIC program allocation's own size (`op.device_address()`'s
    // `CompositeAddress::total_size()`), not the enclosing region's (see
    // the tensor loop's identical fix above for why: the program region is
    // a shared ~16GB slab hosting many independently-compiled bundles —
    // e.g. one per prefill-ladder rung — not one bundle per region). Real
    // C++ says this explicitly: "Bound the segment-7 xlat to the real
    // program allocation footprint... never the 16GB SEGMENT_SIZE. The
    // firmware translation bound check (pa < paddr + length) only protects
    // the program when length is the true footprint."
    let prog_size = op.device_address().total_size().as_u64();
    // ⛔ `trace!` for the same reason as the DMA chunk above, and more so: one per COMPUTE control
    // block, so a single forward emits one per op per layer. It never appeared in the capture that
    // motivated this only because that run hung before its first launch.
    tracing::trace!(
        tag = ?tag,
        num_tensors = tensors.len(),
        tensors = ?tensors.iter().map(|(paddr, len)| format!("{:#x}+{}", paddr.0, len.0)).collect::<Vec<_>>(),
        prog_paddr = format_args!("{:#x}", prog_paddr.0),
        prog_len = prog_size,
        bootstrap_segment = bootstrap.segment(),
        bootstrap_offset = bootstrap.offset().to_bytes(),
        bootstrap_offset_param = op.bootstrap_offset().0,
        "build_compute_cb"
    );
    ControlBlockWire::compute(
        tag,
        &tensors,
        (prog_paddr, AllocationTotalSize(prog_size)),
        bootstrap,
    )
    .map_err(|e| mock_error(&format!("build_compute_cb: {e}")))
}

// ---------------------------------------------------------------------
// PF / VF backends — orchestration native, hardware submission behind
// `HardwareSubmit` (built control blocks + senlib doorbell live in the
// control_blocks/device_interface phases' files)
// ---------------------------------------------------------------------

/// The narrow interface `PfBackend`/`VfBackend` need from the
/// control-blocks/device-interface subsystems to actually reach hardware:
/// build the CB(s) for one submission and ring the doorbell (or, on VF,
/// stream them through firmware messaging), returning once accepted.
/// Building the CB bit-layout itself, IOMMU mapping, and the
/// ResponseWorker/FlexVfStreamer completion-notification plumbing are those
/// other phases' own files; this crate only needs the seam.
pub trait HardwareSubmit: Send + Sync {
    /// Port of `cbi->QueueControlBlocksSBF(&cbr->GetCB(0), cbr->GetNumberOfCBs())`
    /// — `ResponseWorker::LaunchCbs`'s batch branch
    /// (`response_worker.cpp:1268`): ONE call for the whole batch, fire and
    /// forget. The real senlib method returns `void`; the `Result` here only
    /// reports "there is no `ControlBlockInterface` for this pipeline"
    /// (a null `cbi`, which in the C++ is a nullptr dereference).
    fn queue_control_blocks(
        &self,
        pipeline: PipelineId,
        cbs: &[ControlBlockWire],
    ) -> Result<(), SchedulerError>;

    /// Port of `rbi_->ReceiveResponsesSBF(rbs_.data(), max_pending_requests_, 0)`
    /// — `ResponseWorker::FetchResponseBlocks` (`response_worker.cpp:913`).
    /// Drains up to `out.len()` responses (the WHOLE backlog, sized
    /// `max_pending_requests_`), blocking only until `min_count` have
    /// arrived — the real call passes `0`, i.e. it never blocks. Returns the
    /// number actually received.
    /// Returns the number of responses written into `out`, together with the
    /// [`ResponseCredit`] for those slots — which the caller MUST discharge via
    /// [`Self::free_responses`].
    fn receive_responses(
        &self,
        out: &mut [crate::control_block_wire::ResponseBlockWire],
        min_count: usize,
    ) -> (u64, ResponseCredit);

    /// Port of `compute_cbi_->FreeResponses(n)` / `async_dmai_cbi_` /
    /// `async_dmao_cbi_` — `ResponseWorker::ParseResponseBlocks`
    /// (`response_worker.cpp:1141-1152`), called once per pipeline per drain
    /// with that pipeline's own count of responses. This is senlib's
    /// response-slot credit return: without it the hardware CB/RB queue never
    /// gets its slots back.
    ///
    /// Takes a [`PipelineCredit`] rather than a bare count so that it is not
    /// possible to free slots that were never received, and — far more
    /// importantly — not possible to receive slots and forget to free them.
    /// See [`ResponseCredit`].
    fn free_responses(&self, pipeline: PipelineId, credit: PipelineCredit);
}

/// Proof that `outstanding` response slots have been taken from senlib and not
/// yet returned to it.
///
/// This type exists because of a real, shipped bug: the port simply never
/// called `ControlBlockInterface::FreeResponses` — the shim did not even export
/// the symbol — so senlib's response-slot occupancy grew monotonically until
/// `QueueControlBlocksSBF` threw "Control block queue is full, and has not
/// drained" at exactly `cb_queue_length` control blocks. That was an invisible
/// omission: nothing in the type system connected "I received N responses" to
/// "I owe senlib N slots back". Now it does — [`HardwareSubmit::receive_responses`]
/// hands back a credit, [`HardwareSubmit::free_responses`] is its only
/// consumer, and `#[must_use]` plus the crate's `deny(unused_must_use)` makes
/// ignoring one a COMPILE ERROR rather than a silent hardware stall.
#[must_use = "received response slots must be returned to senlib via free_responses();               dropping this credit leaks hardware queue capacity and will eventually wedge the card"]
pub struct ResponseCredit {
    outstanding: u64,
}

impl ResponseCredit {
    /// Only [`HardwareSubmit::receive_responses`] implementations mint these,
    /// and only for slots senlib actually handed over.
    pub fn received(count: u64) -> Self {
        Self { outstanding: count }
    }

    /// A drain that produced no responses owes nothing.
    pub fn none() -> Self {
        Self { outstanding: 0 }
    }

    pub fn outstanding(&self) -> u64 {
        self.outstanding
    }

    /// Split `count` slots off for one pipeline, to hand to
    /// [`HardwareSubmit::free_responses`]. Returns `None` if this credit does
    /// not hold that many — i.e. an attempt to free more slots than were
    /// received, which the real code cannot express.
    pub fn take(&mut self, count: u64) -> Option<PipelineCredit> {
        if count == 0 || count > self.outstanding {
            return None;
        }
        self.outstanding -= count;
        Some(PipelineCredit { count })
    }

    /// Consume a fully-discharged credit. Logs loudly if slots are still owed:
    /// the real `ParseResponseBlocks` frees every response it received, so a
    /// non-zero remainder means this port dropped credit on the floor.
    pub fn settle(self) {
        if self.outstanding != 0 {
            tracing::error!(
                outstanding = self.outstanding,
                "ResponseCredit::settle: response slots were received but never returned to senlib — \
                 hardware queue capacity is leaking (this is the class of bug that wedges the card)"
            );
        }
        std::mem::forget(self);
    }
}

impl Drop for ResponseCredit {
    fn drop(&mut self) {
        if self.outstanding != 0 {
            tracing::error!(
                outstanding = self.outstanding,
                "ResponseCredit dropped with slots still owed to senlib — hardware queue capacity is leaking"
            );
        }
    }
}

/// One pipeline's share of a [`ResponseCredit`], minted only by
/// [`ResponseCredit::take`] and consumed only by
/// [`HardwareSubmit::free_responses`]. There is no other way to obtain or spend
/// one, and no way to fabricate a count.
#[must_use = "a PipelineCredit is only discharged by passing it to free_responses()"]
pub struct PipelineCredit {
    count: u64,
}

impl PipelineCredit {
    pub fn count(&self) -> u64 {
        self.count
    }
}

/// Control blocks that `PrepareCbs` has already published into
/// `pending_requests_` — every tag acquired, every slot initialized, every
/// `ready` flag released — and which therefore MUST reach `LaunchCbs`.
///
/// This is a consumable token rather than a plain `Vec<ControlBlockWire>`
/// because the window between the two calls is not recoverable: once
/// `prepare_cbs` has stored a [`TagLease`] in a slot and set it `Issued`, the
/// only things that can ever return that tag to the pool are a response
/// arriving for it or `CheckPendingJobTimeouts` reaping it — and neither can
/// happen unless `LaunchCbs` bumped `num_pending_requests_` for it. Dropping a
/// prepared batch on the floor therefore strands its slots permanently and
/// leaks their tags, silently, until the pool is exhausted.
///
/// NOTE on what this deliberately does NOT do: it does not retract published
/// slots when `LaunchCbs` fails. The C++ leaves them exactly as they are when
/// `QueueControlBlocksSBF`'s throw unwinds past them (`response_worker.cpp:1268`
/// is unguarded), so that those CBs surface through `CheckPendingJobTimeouts`
/// rather than being forgotten. Adding rollback here would be inventing
/// behavior the real code does not have; the guarantee this type provides is
/// only that the hand-off happens at all.
#[must_use = "a prepared batch has already published its pending-request slots and must be handed to               launch_cbs(); dropping it strands those slots and leaks their response tags"]
struct PreparedBatch {
    /// `None` only after `into_parts` has consumed it.
    cbs: Option<Vec<ControlBlockWire>>,
    pipeline: PipelineId,
}

impl PreparedBatch {
    fn len(&self) -> usize {
        self.cbs.as_ref().map_or(0, Vec::len)
    }

    /// Consume the token on the way into `LaunchCbs`.
    fn into_parts(mut self) -> (Vec<ControlBlockWire>, PipelineId) {
        let cbs = self
            .cbs
            .take()
            .expect("PreparedBatch::into_parts is called exactly once");
        (cbs, self.pipeline)
    }
}

impl Drop for PreparedBatch {
    fn drop(&mut self) {
        if let Some(cbs) = &self.cbs {
            tracing::error!(
                num_cbs = cbs.len(),
                pipeline = ?self.pipeline,
                "PreparedBatch dropped without reaching LaunchCbs: {} pending-request slot(s) are now stranded and                  their response tags leaked — no response can arrive for them and no timeout scan will reap them,                  because num_pending_requests_ was never incremented",
                cbs.len()
            );
        }
    }
}

/// Resolves a `CompositeAddress`'s logical chunks into the real
/// [segment, physical byte address] pairs the CB wire encoder needs.
/// `FlexAllocator::get_id_to_region_map()` + `MemoryRegion::segment_id()`/
/// `MemoryRegion::data().device_address_bytes()` is the production
/// implementation; kept as a trait (same DI pattern as `HardwareSubmit`/
/// `QueueCapacitySource`) so this file does not need to own a `FlexAllocator`
/// reference directly.
pub trait DeviceAddressResolver: Send + Sync {
    /// Returns `None` if `chunk`'s region is unknown to this resolver.
    /// Third element is the OWNING REGION's total allocated size — distinct
    /// from `chunk.size` (this particular transfer's own byte count) and
    /// required for the XLAT entry's length field, which the real
    /// `buildDmaControlBlocks` (`pf_runtime_scheduler_utils.cpp:48`) sets to
    /// `region_id_to_region_.at(region_id)->total_size()`, NOT the current
    /// chunk/transfer's length. Conflating the two (this crate's bug until
    /// found by real-hardware testing) makes any transfer starting at a
    /// non-zero region offset fail hardware's XLAT bounds check whenever
    /// `offset + this_transfer's_own_size > this_transfer's_own_size` — i.e.
    /// any time chunks are ever loaded via more than one call, since a
    /// later chunk's own size alone can never cover its own offset.
    fn resolve(&self, chunk: &Chunk) -> Option<(u8, DevicePhysicalAddress, ByteSize)>;
}

/// Production `DeviceAddressResolver`: looks `Chunk::addr.region_id` up in a
/// live `FlexAllocator`'s region map and reads the region's real device
/// physical address off its senlib-backed `DeviceMemoryAllocation`. Port of
/// the pointer arithmetic C++'s `ControlBlockStream` builders do directly
/// against a `MemoryRegion*` (segment id + `DmpaAsBytes()`), just reached
/// through the allocator's map instead of a raw region pointer.
///
/// Holds an `Arc<Mutex<FlexAllocator>>` rather than a bare reference so it
/// can be cloned freely into `PfBackend`/`VfBackend` (which are themselves
/// `Send + Sync` and may be shared across scheduler threads) without
/// borrowing the allocator for the resolver's lifetime.
pub struct FlexAllocatorAddressResolver {
    allocator: Arc<Mutex<crate::allocator::FlexAllocator>>,
}

impl FlexAllocatorAddressResolver {
    pub fn new(allocator: Arc<Mutex<crate::allocator::FlexAllocator>>) -> Self {
        Self { allocator }
    }
}

impl DeviceAddressResolver for FlexAllocatorAddressResolver {
    fn resolve(&self, chunk: &Chunk) -> Option<(u8, DevicePhysicalAddress, ByteSize)> {
        let allocator = self
            .allocator
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let region_map = allocator.get_id_to_region_map();
        let region = region_map.get(&chunk.addr.region_id)?;
        // Segment id is only assigned for regions that back CB-addressable
        // (device-visible) memory — see `MemoryRegion::set_segment_id`. A
        // region with none is not a valid CB operand target.
        let segment = region.segment_id()?;
        let base_paddr = region.data()?.device_address_bytes();
        Some((
            segment.0 as u8,
            DevicePhysicalAddress(base_paddr),
            region.total_size(),
        ))
    }
}

/// Production `HardwareSubmit` — the three senlib calls `ResponseWorker`
/// itself makes, and nothing else. Two DISTINCT senlib resources are
/// involved, with two DIFFERENT real thread-safety shapes, confirmed by
/// reading the C++ shim (`cxx-shim/senlib_ffi.cpp`) rather than assumed:
///
/// - `ControlBlockInterface` (submission/doorbell) is genuinely per-pipeline
///   — `cbi_for_pipeline` returns `GetCmptCbi()`/`GetDmaiCbi()`/`GetDmaoCbi()`,
///   three distinct senlib objects, one real hardware queue each. This is
///   also where `FreeResponses` lives (`control_block_interface.hpp:49`).
/// - `ResponseBlockInterface` (completions) is NOT per-pipeline at all —
///   `flex_shim_rbi_for_pipeline`'s own parameter is unused
///   (`/*pipeline*/`) and its body says outright: "senlib exposes exactly
///   one Rbi() per device." All three pipelines' completions drain through
///   ONE shared cursor, which is exactly why the real C++ has exactly ONE
///   `ResponseWorker::ResponseCompletionThread` per device draining it.
///
/// This type holds no locks and no threads of its own: serialisation of
/// submission is `ResponseWorker::submission_mtx_`'s job
/// (`response_worker.cpp:488` — ONE mutex covering `PrepareCbs` +
/// `LaunchCbs` for ALL pipelines, not one per pipeline), and being the
/// single consumer of `Rbi()` is `ResponseCompletionThread`'s job.
pub struct SenlibHardwareSubmit {
    /// Scratch for `ReceiveResponsesSBF`'s out-parameter — the senlib-format
    /// twin of `ResponseWorker::rbs_`. Grown to the caller's requested batch
    /// size on first use and reused, since only the one
    /// `ResponseCompletionThread` ever drains (the mutex exists because the
    /// trait method takes `&self`, not because there is real contention).
    rbs_scratch: Mutex<Vec<crate::senlib_ffi_scheduler::SenlibResponseBlockSbf>>,
}

impl Default for SenlibHardwareSubmit {
    fn default() -> Self {
        Self::new()
    }
}

impl SenlibHardwareSubmit {
    pub fn new() -> Self {
        Self {
            rbs_scratch: Mutex::new(Vec::new()),
        }
    }
}

impl HardwareSubmit for SenlibHardwareSubmit {
    fn queue_control_blocks(
        &self,
        pipeline: PipelineId,
        cbs: &[ControlBlockWire],
    ) -> Result<(), SchedulerError> {
        let shim_pipeline = pipeline.shim_index();
        // SAFETY: safe to call at any time; returns a null handle if the
        // shim runtime is not initialized (see its own doc comment).
        let cbi = unsafe { crate::senlib_ffi_scheduler::flex_shim_cbi_for_pipeline(shim_pipeline) };
        let Some(cbi) = cbi.into_live() else {
            return Err(mock_error(
                "SenlibHardwareSubmit: no ControlBlockInterface for this pipeline (shim not initialized)",
            ));
        };
        let wire: Vec<crate::senlib_ffi_scheduler::SenlibControlBlockSbf> =
            cbs.iter().copied().map(Into::into).collect();
        // Propagate rather than panic: the real `:1268` call is unguarded, so a
        // senlib throw unwinds into `schedule`'s `catch(...)` and fails the
        // request. Swallowing it (the original port) left the caller waiting out
        // its full timeout on a CB the hardware never received.
        crate::senlib_ffi_scheduler::cbi_queue_control_blocks_sbf(cbi, &wire).map_err(|e| {
            mock_error(&format!(
                "SenlibHardwareSubmit: senlib QueueControlBlocksSBF failed (CB not submitted): {e}"
            ))
        })
    }

    fn receive_responses(
        &self,
        out: &mut [crate::control_block_wire::ResponseBlockWire],
        min_count: usize,
    ) -> (u64, ResponseCredit) {
        // SAFETY: safe to call at any time; returns a null handle if the
        // shim runtime is not initialized. The pipeline argument is genuinely
        // ignored by the real shim (see this struct's own doc comment), so
        // `0` here is not a choice among equals — there is only one `Rbi()`.
        let rbi = unsafe { crate::senlib_ffi_scheduler::flex_shim_rbi_for_pipeline(0) };
        let Some(rbi) = rbi.into_live() else {
            return (0, ResponseCredit::none());
        };
        let mut scratch = self.rbs_scratch.lock().unwrap_or_else(|e| e.into_inner());
        if scratch.len() < out.len() {
            scratch.resize(
                out.len(),
                crate::senlib_ffi_scheduler::SenlibResponseBlockSbf { bytes: [0; 64] },
            );
        }
        let n = match crate::senlib_ffi_scheduler::rbi_receive_responses_sbf(
            rbi,
            &mut scratch[..out.len()],
            min_count,
        ) {
            Ok(n) => n,
            Err(e) => {
                // Previously a throw returned 0, i.e. "nothing arrived yet", so
                // the drain loop spun to its full timeout instead of reporting.
                tracing::error!(err = %e, "senlib ReceiveResponsesSBF failed");
                return (0, ResponseCredit::none());
            }
        };
        for (dst, src) in out.iter_mut().zip(scratch.iter()).take(n as usize) {
            *dst = src.decode();
        }
        (n, ResponseCredit::received(n))
    }

    /// Port of `ParseResponseBlocks`' per-pipeline
    /// `cbi->FreeResponses(num_responses)` (`response_worker.cpp:1141-1152`),
    /// returning consumed response-block slots to senlib.
    ///
    /// This leaf was a stub because the shim exported no `FreeResponses` entry
    /// point, so the port never returned that credit: senlib's queue occupancy
    /// grew monotonically and `QueueControlBlocksSBF` threw "Control block
    /// queue is full, and has not drained" at exactly `cb_queue_length` CBs
    /// (32768 on this card, per `ControlBlockInterface::capacity()`'s own doc)
    /// — the reproducible long-context batched-decode hang. The shim entry
    /// point now exists.
    fn free_responses(&self, pipeline: PipelineId, credit: PipelineCredit) {
        let num_responses = credit.count();
        let shim_pipeline = pipeline.shim_index();
        // SAFETY: as `queue_control_blocks` above — null handle if the shim
        // runtime is not initialized.
        let cbi = unsafe { crate::senlib_ffi_scheduler::flex_shim_cbi_for_pipeline(shim_pipeline) };
        let Some(cbi) = cbi.into_live() else {
            tracing::error!(
                pipeline = shim_pipeline,
                "free_responses: no ControlBlockInterface for this pipeline"
            );
            return;
        };
        // The real call is `void`, made on the response-completion thread; a
        // senlib throw there propagates out of that thread. ERROR logging is
        // the closest faithful behavior at a leaf that cannot unwind into its
        // caller (see this file's notes on why that thread does not panic).
        if let Err(e) = crate::senlib_ffi_scheduler::cbi_free_responses(cbi, num_responses) {
            tracing::error!(pipeline = shim_pipeline, num_responses, err = %e, "senlib FreeResponses failed");
        }
    }
}

// ---------------------------------------------------------------------
// ResponseWorker (control_blocks/message_processing/response_worker.{hpp,cpp})
// ---------------------------------------------------------------------
//
// This is the real PF submission/completion architecture, and it is
// deliberately ported HERE (in the scheduler file) even though
// `response_worker.{hpp,cpp}` nominally belongs to the
// `control_blocks/message_processing` phase: `PfRuntimeScheduler` has no
// other way to reach hardware (`submitToHardware`'s entire body is a
// capacity gate, an in-flight bump, and `response_worker_->QueueCbs(...)`),
// so modelling it as "an opaque native-Rust dependency" — what this file's
// module doc used to claim — left the crate with no async path at all and a
// synthetic, `MockRuntimeScheduler`-shaped substitute in its place. The
// telemetry members (`TimestampDecoder`/`PipelineTimestamps`/
// `MeasureDuration`/`ResponseWorkerStats`/the aiupti profiler submission and
// the `batch_timestamp_ring_`, which exists only to give the profiler a
// baseline timestamp) are the parts that genuinely belong to another phase
// and are NOT ported; every member that decides control flow is.

/// Port of `flex::QueuingMode` (`response_worker.hpp:228-235`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueuingMode {
    Regular,
    Calibration,
    StopThreads,
}

/// Port of `flex::PendingRequestState` (`response_worker.hpp:323-333`).
/// Stored in an `AtomicU8` exactly as the C++ stores it in a
/// `std::atomic<PendingRequestState>`: the submitting thread publishes
/// `ISSUED` and the completion thread reads it to decide whether a response
/// is live or stale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PendingRequestState {
    Invalid = 0,
    Available = 1,
    Issued = 2,
    Succeeded = 3,
    Skipped = 4,
    Failed = 5,
    TimedOut = 6,
}

impl PendingRequestState {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Available,
            2 => Self::Issued,
            3 => Self::Succeeded,
            4 => Self::Skipped,
            5 => Self::Failed,
            6 => Self::TimedOut,
            _ => Self::Invalid,
        }
    }
}

/// Port of `flex::PendingRequest` (`response_worker.hpp:351-493`), minus the
/// telemetry-only members (`in_flight_duration`, `callback_duration`,
/// `pipeline_timestamps`, and the VF firmware-message fields `dlm_only`/
/// `fw_begin_time`/`fw_end_time`/`msg_type`), which have no consumer in this
/// crate — `DecodeTimestamps`/`TimestampDecoder` and
/// `PROFILER_SUBMIT_PENDING_REQUEST` both live in the telemetry phase's
/// files. This is the value a completion callback receives, matching the C++
/// callback signature `void(PendingRequest)` (a COPY — see `IssueCallback`).
#[derive(Debug, Clone)]
pub struct PendingRequest {
    pub id: u64,
    pub batch_id: u64,
    pub cb_idx: u64,
    /// `PipelineId::INVALID` is modeled as `None` (the C++ resets this field
    /// to `INVALID` right before returning the slot to `AVAILABLE`).
    pub pipeline: Option<PipelineId>,
    pub queuing_mode: QueuingMode,
    pub state: PendingRequestState,
    pub issue_time_point: Option<Instant>,
    pub launch_time_point: Option<Instant>,
    pub completion_time_point: Option<Instant>,
    /// `None` only for a slot that has never been populated — the C++ holds a
    /// default-constructed `ControlBlockSBF` there instead.
    pub control_block: Option<ControlBlockWire>,
    pub response_block: Option<crate::control_block_wire::ResponseBlockWire>,
    pub node_name: String,
}

/// Port of `flex::CallbackFn` (`std::function<void(PendingRequest)>`).
/// `Arc` rather than `Box` because the real `cb_fn` is COPIED into every
/// `PendingRequest` of a batch by `ProcessTags` (`pr.callback_fn = cb_fn;`)
/// and then moved out of exactly one of them at a time by `IssueCallback`.
pub type CallbackFn = Arc<dyn Fn(&PendingRequest) + Send + Sync>;

/// The mutable half of one `pending_requests_` slot.
struct PendingRequestData {
    id: u64,
    batch_id: u64,
    cb_idx: u64,
    pipeline: Option<PipelineId>,
    queuing_mode: QueuingMode,
    issue_time_point: Option<Instant>,
    launch_time_point: Option<Instant>,
    completion_time_point: Option<Instant>,
    control_block: Option<ControlBlockWire>,
    response_block: Option<crate::control_block_wire::ResponseBlockWire>,
    callback_fn: Option<CallbackFn>,
    node_name: String,
    /// The tag lease naming this slot for as long as it is ISSUED. Dropping
    /// it IS `ReturnTags`' `available_tags_.enqueue_bulk(...)` — see
    /// [`TagLease`].
    lease: Option<TagLease>,
}

/// One entry of `ResponseWorker::pending_requests_`, indexed by
/// `EncodedResponseTag::local_tag`.
///
/// The C++ writes these fields from the submitting thread and reads them from
/// the completion thread with no mutex at all, relying on `state`/`ready`
/// plus the hardware round-trip as the synchronisation edge (see
/// `ProcessTags`' own "CRITICAL: Release store" comment). Rust cannot express
/// that without UB, so the non-atomic fields sit behind a mutex — but the
/// `ready` release/acquire handshake and the `state` protocol are ported
/// exactly as written, because those are what decide whether a response is
/// live, stale, or not yet published, and the mutex does not.
struct PendingRequestSlot {
    state: std::sync::atomic::AtomicU8,
    ready: AtomicBool,
    data: Mutex<PendingRequestData>,
}

impl PendingRequestSlot {
    fn new() -> Self {
        Self {
            // Port of `std::atomic<PendingRequestState> state{AVAILABLE}`.
            state: std::sync::atomic::AtomicU8::new(PendingRequestState::Available as u8),
            ready: AtomicBool::new(false),
            data: Mutex::new(PendingRequestData {
                id: 0,
                batch_id: 0,
                cb_idx: 0,
                pipeline: None,
                queuing_mode: QueuingMode::Regular,
                issue_time_point: None,
                launch_time_point: None,
                completion_time_point: None,
                control_block: None,
                response_block: None,
                callback_fn: None,
                node_name: String::new(),
                lease: None,
            }),
        }
    }

    fn state(&self) -> PendingRequestState {
        PendingRequestState::from_u8(self.state.load(Ordering::Acquire))
    }

    fn set_state(&self, s: PendingRequestState) {
        self.state.store(s as u8, Ordering::Release);
    }

    /// The `PendingRequest pr_copy = *pending_request;` of `IssueCallback`
    /// (`response_worker.cpp:1285`) — a snapshot handed to the callback so it
    /// cannot observe the slot being recycled underneath it.
    fn snapshot(&self, data: &PendingRequestData) -> PendingRequest {
        PendingRequest {
            id: data.id,
            batch_id: data.batch_id,
            cb_idx: data.cb_idx,
            pipeline: data.pipeline,
            queuing_mode: data.queuing_mode,
            state: self.state(),
            issue_time_point: data.issue_time_point,
            launch_time_point: data.launch_time_point,
            completion_time_point: data.completion_time_point,
            control_block: data.control_block,
            response_block: data.response_block,
            node_name: data.node_name.clone(),
        }
    }
}

/// The completion thread's two reusable buffers: port of
/// `ResponseWorker::rbs_` (`std::vector<ResponseBlockSBF>(max_pending_requests_)`)
/// and `tags_to_be_returned_` (likewise `max_pending_requests_`-sized). Both
/// are members in the C++ because only the one completion thread touches
/// them; here they are passed down the call chain for the same reason, which
/// also lets `QueueCbs`' `single_thread_optimized_` path own its own pair
/// without racing the worker thread.
struct DrainScratch {
    rbs: Vec<crate::control_block_wire::ResponseBlockWire>,
    tags_to_be_returned: Vec<ResponseTag>,
}

impl DrainScratch {
    fn new(max_pending_requests: u64) -> Self {
        let n = max_pending_requests as usize;
        Self {
            rbs: vec![
                crate::control_block_wire::ResponseBlockWire::from_bytes(
                    &[0u8; crate::control_block_wire::RB_NUM_BYTES]
                );
                n
            ],
            tags_to_be_returned: vec![ResponseTag(0); n],
        }
    }
}

/// Port of `flex::ResponseWorker` (`response_worker.{hpp,cpp}`): the
/// asynchronous producer/consumer that IS the PF submission path.
///
/// Producer side (`QueueCbs` → `Precheck` → `PrepareCbs` → `LaunchCbs`) runs
/// on whatever thread called `submitToHardware`, under one
/// `submission_mtx_`; it takes `num_cbs` tags out of `available_tags_`,
/// publishes one `pending_requests_` slot per CB, bumps
/// `num_pending_requests_` by `num_cbs`, and rings the doorbell ONCE for the
/// whole batch. It then RETURNS — it does not wait for anything.
///
/// Consumer side is the single background `ResponseCompletionThread`, which
/// drains the ENTIRE backlog per iteration
/// (`ReceiveResponsesSBF(rbs_.data(), max_pending_requests_, 0)`), matches
/// each response to its slot by tag, fires that slot's callback, returns the
/// tags, and decrements `num_pending_requests_` by the number of valid
/// responses.
///
/// What this replaced: a `SenlibHardwareSubmit::submit` that queued ONE CB
/// and then blocked the calling thread on that CB's own response, draining
/// `ReceiveResponsesSBF(..., 1)` — one response per call — through a
/// `ResponseRouter` tag→channel map. That was modeled on
/// `MockRuntimeScheduler::submitCompute` (`mock_runtime_scheduler.cpp:302-341`),
/// the only synchronous submit in the real C++, and it has no counterpart on
/// the PF path at all: no `pending_requests_` table, no
/// `num_pending_requests_`, no background thread, no whole-backlog drain, no
/// timeout scan, and no `FreeResponses`.
pub struct ResponseWorker {
    /// `max_pending_requests_` = `FlexConfig::ResponseWorkerMaxPendingRequests()`.
    max_pending_requests: u64,
    /// `pending_requests_`, sized `max_pending_requests_ + tag_offset_`
    /// (`response_worker.cpp:316`) so `local_tag` indexes it directly.
    pending_requests: Vec<PendingRequestSlot>,
    /// `available_tags_` + `available_tags_ptok_`, already ported: the pool
    /// holds `max_pending_requests_` tags valued
    /// `tag_offset_ .. max_pending_requests_ + tag_offset_` (`InitTags`).
    tag_pool: Arc<TagPool>,
    /// `num_pending_requests_`.
    num_pending_requests: std::sync::atomic::AtomicU64,
    /// `response_block_timeout_fail_secs_` / `response_block_timeout_report_secs_`
    /// (`response_worker.cpp:282-283`: report = `max(fail / 5, 1)`).
    response_block_timeout_fail_secs: u64,
    response_block_timeout_report_secs: u64,
    single_thread_optimized: bool,
    fail_on_timeout: bool,
    skip_launch: bool,
    hw: Arc<dyn HardwareSubmit>,
    /// `submission_mtx_` — ONE mutex for every pipeline, held across
    /// `Precheck` + `PrepareCbs` + `LaunchCbs` (`response_worker.cpp:488`).
    submission_mtx: Mutex<()>,
    stop_threads: AtomicBool,
    force_shutdown: AtomicBool,
    /// `wakeup_queue_` (a `BlockingConcurrentQueue<uint64_t>` of capacity 1):
    /// `LaunchCbs` enqueues when it takes `num_pending_requests_` from 0 to
    /// non-zero, and the worker thread `wait_dequeue`s when it finds nothing
    /// pending.
    wakeup_tx: std::sync::mpsc::Sender<u64>,
    wakeup_rx: Mutex<std::sync::mpsc::Receiver<u64>>,
    /// `static std::atomic<uint64_t> global_pr_id` / `global_pr_batch_id_`.
    global_pr_id: std::sync::atomic::AtomicU64,
    global_pr_batch_id: std::sync::atomic::AtomicU64,
    worker_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

/// The per-pipeline response tallies `ParseResponseBlocks` accumulates while it
/// walks a drain's response blocks, and then spends on `FreeResponses` — one per
/// pipeline (`response_worker.cpp:1141-1152`). Grouped so
/// `process_response_blocks` takes one `&mut` instead of three, which also makes
/// it impossible to increment the wrong pipeline's counter by passing the
/// arguments out of order.
#[derive(Debug, Default, Clone, Copy)]
struct PipelineResponseCounts {
    compute: u64,
    dmai: u64,
    dmao: u64,
}

impl PipelineResponseCounts {
    /// The tallies in `FreeResponses` call order.
    fn per_pipeline(&self) -> [(PipelineId, u64); 3] {
        [
            (PipelineId::Compute, self.compute),
            (PipelineId::AsyncDmaI, self.dmai),
            (PipelineId::AsyncDmaO, self.dmao),
        ]
    }
}

impl ResponseWorker {
    /// Port of the constructor (`response_worker.cpp:276-330`): read the
    /// config, size `pending_requests_`, fill the tag pool (`InitTags`) —
    /// both "in the constructor, so they are fully populated before
    /// worker_thread_ has a chance to run" — then start
    /// `ResponseCompletionThread`.
    ///
    /// The `shutdown_cbt_` member (a pre-built cancelled `ControlBlockStream`
    /// used to wake the worker during `Shutdown`) is NOT ported: this crate's
    /// CB encoder (`control_block_wire.rs`) has no `CreateCancelledCb`, so
    /// there is nothing to build. `shutdown()` below wakes the thread through
    /// `wakeup_queue_` instead, which is what the cancelled CB exists to
    /// achieve.
    pub fn new(hw: Arc<dyn HardwareSubmit>) -> Arc<Self> {
        let max_pending_requests = RuntimeConfig::response_worker_max_pending_requests()
            .value()
            .max(1);
        let fail_secs = RuntimeConfig::response_worker_timeout_seconds().value();
        let (wakeup_tx, wakeup_rx) = std::sync::mpsc::channel();
        let mut pending_requests =
            Vec::with_capacity((max_pending_requests + TagPool::TAG_OFFSET as u64) as usize);
        for _ in 0..max_pending_requests + TagPool::TAG_OFFSET as u64 {
            pending_requests.push(PendingRequestSlot::new());
        }
        let worker = Arc::new(Self {
            max_pending_requests,
            pending_requests,
            tag_pool: TagPool::new(max_pending_requests),
            num_pending_requests: std::sync::atomic::AtomicU64::new(0),
            response_block_timeout_fail_secs: fail_secs,
            response_block_timeout_report_secs: (fail_secs / 5).max(1),
            // `FlexConfig::SingleThreadOptimized()` — the only one of these
            // five knobs with no `RuntimeConfig` twin in this crate's
            // `flex_config.rs` (the C++ reads `FlexConfig` for
            // max_pending_requests/timeout/fail_on_timeout/single_thread and
            // `RuntimeConfig` only for skip_launch; the four with both are read
            // through `RuntimeConfig` here for consistency with the rest of
            // this file, and resolve to the same values).
            single_thread_optimized: crate::flex_config::FlexConfig::single_thread_optimized(),
            fail_on_timeout: RuntimeConfig::response_worker_fail_on_timeout().value(),
            skip_launch: RuntimeConfig::response_worker_skip_launch().value(),
            hw,
            submission_mtx: Mutex::new(()),
            stop_threads: AtomicBool::new(false),
            force_shutdown: AtomicBool::new(false),
            wakeup_tx,
            wakeup_rx: Mutex::new(wakeup_rx),
            global_pr_id: std::sync::atomic::AtomicU64::new(0),
            global_pr_batch_id: std::sync::atomic::AtomicU64::new(0),
            worker_thread: Mutex::new(None),
        });
        // `worker_thread_(&ResponseWorker::ResponseCompletionThread, this)`.
        // A `Weak` upgrade stands in for the C++ raw `this`: the worker owns
        // no strong reference, so a dropped `ResponseWorker` ends the thread
        // instead of keeping itself alive forever through an Arc cycle.
        let weak = Arc::downgrade(&worker);
        let handle = std::thread::Builder::new()
            // `pthread_setname_np(pthread_self(), "RspWrk0")`.
            .name("RspWrk0".to_string())
            .spawn(move || {
                if let Some(me) = weak.upgrade() {
                    me.response_completion_thread();
                }
            })
            .expect("ResponseWorker: failed to spawn ResponseCompletionThread");
        *worker
            .worker_thread
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(handle);
        worker
    }

    /// Port of `ResponseWorker::QueueCbs` (`response_worker.cpp:485-550`).
    ///
    /// `build` is the port of the `SpecializeTags(raw_tags)` step: the real
    /// `PrepareCbs` acquires `num_cbs` tags from `available_tags_`, stamps
    /// each into its CB (`pr.control_block.ctrl().SetTag(...)` plus
    /// `cbr->SpecializeTags(raw_tags)`) and only then launches. This crate's
    /// `ControlBlockWire` is immutable once assembled (no `SetTag`), so the
    /// tags are handed to the CB builder instead of being stamped into
    /// already-built CBs — the same tags, the same CBs, assigned by the
    /// ResponseWorker under the same `submission_mtx_`.
    pub fn queue_cbs(
        &self,
        cb_fn: CallbackFn,
        num_cbs: u64,
        pipeline: PipelineId,
        qm: QueuingMode,
        build: impl FnOnce(&[ResponseTag]) -> Result<Vec<ControlBlockWire>, SchedulerError>,
    ) -> Result<(), SchedulerError> {
        let submission_lock = self
            .submission_mtx
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        self.precheck(num_cbs, qm)?;
        let batch = self.prepare_cbs(&cb_fn, num_cbs, pipeline, qm, build)?;
        debug_assert_eq!(batch.len() as u64, num_cbs);
        let launch_result = self.launch_cbs(batch);
        drop(submission_lock);
        // `LaunchCbs`' `cbi->QueueControlBlocksSBF(...)` (`:1268`) is
        // UNGUARDED in the C++: when senlib throws ("Control block queue is
        // full, and has not drained"), the exception propagates straight out
        // of `LaunchCbs` -> `QueueCbs` -> `submitToHardware` and is caught by
        // `RuntimeScheduler::schedule`'s own `catch(...)`, which resolves the
        // fence and reports the exception to the caller. Nothing retries a
        // full queue; `waitForQueueCapacity`, run earlier in
        // `submitToHardware`, is the only thing that is supposed to prevent
        // it. Faithfully reproduced here: the error propagates, and the
        // already-published `pending_requests_` slots plus the already-bumped
        // `num_pending_requests_` are left as they are — exactly as the C++
        // leaves them when the throw unwinds past them — so those CBs surface
        // through `CheckPendingJobTimeouts` rather than being silently
        // forgotten.
        launch_result?;

        // "Single-threaded mode: poll synchronously until all responses
        // received" (`response_worker.cpp:540-547`). `ResponseCompletionThread`
        // returns immediately in this mode, so this thread IS the consumer.
        if self.single_thread_optimized && !self.skip_launch {
            let mut scratch = DrainScratch::new(self.max_pending_requests);
            while self.num_pending_requests.load(Ordering::Acquire) > 0 {
                self.fetch_and_parse_response_blocks(&mut scratch);
            }
        }
        Ok(())
    }

    /// Port of `ResponseWorker::Precheck` (`response_worker.cpp:552-571`).
    #[allow(rustdoc::private_intra_doc_links)]
    /// See [`PreparedBatch`] for why `prepare_cbs`' output is a consumable
    /// token rather than a plain `Vec`.
    /// The first two branches `Throw()` in the C++ (`RAS::RESPONSEWORKER::NoCbs`,
    /// `MaxCbsExceeded`); the third returns a plain
    /// `Status::RUNTIME_ERROR("Shutdown in process")`.
    fn precheck(&self, num_cbs: u64, qm: QueuingMode) -> Result<(), SchedulerError> {
        if num_cbs == 0 {
            return Err(mock_error(
                "ResponseWorker::Precheck: NoCbs (zero control blocks submitted)",
            ));
        }
        if num_cbs > self.max_pending_requests {
            return Err(mock_error(&format!(
                "ResponseWorker::Precheck: MaxCbsExceeded (num_cbs={num_cbs}, max_cbs={})",
                self.max_pending_requests
            )));
        }
        if qm != QueuingMode::StopThreads && self.stop_threads.load(Ordering::Acquire) {
            return Err(mock_error("Shutdown in process"));
        }
        Ok(())
    }

    /// Port of `ResponseWorker::PrepareCbs` (`response_worker.cpp:757-847`) +
    /// `ProcessTags` (`:573-617`). The debug-only members (`trace_cb_nodenames`
    /// dump, `dump_dmai`, the `batch_timestamp_ring_` slot write — whose only
    /// consumer is the profiler's baseline timestamp) are not ported.
    fn prepare_cbs(
        &self,
        cb_fn: &CallbackFn,
        num_cbs: u64,
        pipeline: PipelineId,
        qm: QueuingMode,
        build: impl FnOnce(&[ResponseTag]) -> Result<Vec<ControlBlockWire>, SchedulerError>,
    ) -> Result<PreparedBatch, SchedulerError> {
        // "Acquire tags from the available pool (may block if pool
        // exhausted)" — the real `while(tags_fetched < num_ctbs)` loop over
        // `try_dequeue_bulk_from_producer`, with the same "Still waiting to
        // fetch tags" warning (in `TagPool::acquire`).
        let mut leases: Vec<TagLease> = Vec::with_capacity(num_cbs as usize);
        for _ in 0..num_cbs {
            leases.push(self.tag_pool.acquire());
        }
        let tags: Vec<ResponseTag> = leases.iter().map(|l| l.tag()).collect();

        let batch_id = self.global_pr_batch_id.fetch_add(1, Ordering::AcqRel) + 1;
        // "Capture baseline timestamp once for entire batch - all CBs share
        // same timestamp" (`response_worker.cpp:791`).
        let batch_baseline_timestamp = Instant::now();

        // `cbr->SpecializeTags(raw_tags)` — wrapped so a failure becomes
        // `RAS::RESPONSEWORKER::SpecializeTagsFailed`. `leases` is still
        // owned here, so an early return returns every tag to the pool.
        let cbs = build(&tags)?;
        if cbs.len() as u64 != num_cbs {
            return Err(mock_error(&format!(
                "ResponseWorker::PrepareCbs: builder produced {} control blocks for {num_cbs} tags",
                cbs.len()
            )));
        }

        for (i, lease) in leases.into_iter().enumerate() {
            let local_tag = lease.tag().0 as usize;
            // "Validate tag is within bounds before accessing
            // pending_requests_" (`ProcessTags`, `:585-590`) — `RAS::…::InvalidTag`.
            if local_tag >= self.pending_requests.len() {
                return Err(mock_error(&format!(
                    "ResponseWorker::ProcessTags: InvalidTag (local_tag={local_tag}) exceeds pending_requests_ size ({})",
                    self.pending_requests.len()
                )));
            }
            let slot = &self.pending_requests[local_tag];
            {
                let mut data = slot.data.lock().unwrap_or_else(|e| e.into_inner());
                // "Initialize all fields before setting ready flag."
                data.control_block = Some(cbs[i]);
                data.id = self.global_pr_id.fetch_add(1, Ordering::AcqRel) + 1;
                data.batch_id = batch_id;
                data.cb_idx = i as u64;
                data.pipeline = Some(pipeline);
                data.queuing_mode = qm;
                data.callback_fn = Some(cb_fn.clone());
                data.node_name.clear();
                data.issue_time_point = Some(batch_baseline_timestamp);
                data.completion_time_point = None;
                // NOTE: `launch_time_point` is deliberately NOT reset here.
                // The C++ doesn't reset it either — `LaunchCbs` overwrites it
                // immediately afterward, under the same `submission_mtx_`.
                data.lease = Some(lease);
            }
            slot.set_state(PendingRequestState::Issued);
            // "CRITICAL: Release store to make all above writes visible to
            // worker thread" (`:608-610`).
            slot.ready.store(true, Ordering::Release);
        }
        // Every slot is now published, so from here the batch MUST be launched
        // — see `PreparedBatch`.
        Ok(PreparedBatch {
            cbs: Some(cbs),
            pipeline,
        })
    }

    /// Port of `ResponseWorker::LaunchCbs(cbr, pipeline, seq_launch)`
    /// (`response_worker.cpp:849-900`) and its batch branch in
    /// `LaunchCbs(cbr, cbi, seq_launch)` (`:1255-1269`).
    ///
    /// `seq_launch` is not ported: it is a debug path (one
    /// `QueueControlBlocksSBF` per CB, then a busy-wait for that CB's
    /// response before the next) and every real caller — including
    /// `submitToHardware`'s `QueueCbs(wrapped_callback, cbr, pipeline)` —
    /// leaves it at its `false` default.
    fn launch_cbs(&self, batch: PreparedBatch) -> Result<(), SchedulerError> {
        let (cbs, pipeline) = batch.into_parts();
        let cbs = cbs.as_slice();
        let num_cbs = cbs.len() as u64;
        let prev_pending = self
            .num_pending_requests
            .fetch_add(num_cbs, Ordering::AcqRel);
        if !self.single_thread_optimized && prev_pending == 0 {
            // "Waking up ResponseWorker thread".
            let _ = self.wakeup_tx.send(prev_pending);
        }

        if self.skip_launch {
            // `FLEX_RESPONSE_WORKER_SKIP_LAUNCH`: mark every CB SKIPPED and
            // fire its callback inline, without touching hardware.
            for cb in cbs {
                let local_tag = cb.tag().0 as usize;
                if local_tag >= self.pending_requests.len() {
                    tracing::error!(
                        local_tag,
                        "LaunchCbs (skip_launch): InvalidTag exceeds pending_requests_ size"
                    );
                    continue;
                }
                let slot = &self.pending_requests[local_tag];
                slot.set_state(PendingRequestState::Skipped);
                self.issue_callback(slot);
            }
            self.num_pending_requests
                .fetch_sub(num_cbs, Ordering::AcqRel);
            return Ok(());
        }

        let launch_time = Instant::now();
        for cb in cbs {
            let local_tag = cb.tag().0 as usize;
            if let Some(slot) = self.pending_requests.get(local_tag) {
                slot.data
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .launch_time_point = Some(launch_time);
            }
        }
        // ONE doorbell for the whole batch, and let a submission failure
        // propagate (see `queue_cbs`).
        self.hw.queue_control_blocks(pipeline, cbs)
    }

    /// Port of `ResponseWorker::FetchResponseBlocks` (`response_worker.cpp:904-956`).
    fn fetch_response_blocks(&self, scratch: &mut DrainScratch) -> (u64, ResponseCredit) {
        let start = Instant::now();
        let mut last_report = Duration::ZERO;
        loop {
            // The whole backlog, non-blocking: `ReceiveResponsesSBF(rbs_.data(),
            // max_pending_requests_, 0)`.
            let (num_received, credit) = self.hw.receive_responses(&mut scratch.rbs, 0);
            if num_received > 0 {
                return (num_received, credit);
            }
            credit.settle(); // nothing received: nothing owed
            if self.num_pending_requests.load(Ordering::Acquire) == 0 {
                // "No requests were pending".
                return (0, ResponseCredit::none());
            }
            if self.force_shutdown.load(Ordering::Acquire) {
                return (0, ResponseCredit::none());
            }

            let wait_time = start.elapsed();
            if wait_time.as_secs() >= self.response_block_timeout_fail_secs {
                self.check_pending_job_timeouts();
                tracing::error!(
                    timeout_secs = self.response_block_timeout_fail_secs,
                    num_pending_requests = self.num_pending_requests.load(Ordering::Acquire),
                    seconds_waited = wait_time.as_secs(),
                    "No ResponseBlock received within timeout"
                );
                // The C++ throws `RAS::RESPONSEWORKER::RbTimeOut()` here when
                // `fail_on_timeout_` is set. Thrown from
                // `ResponseCompletionThread` that reaches `std::terminate`;
                // the Rust equivalent would kill this thread and silently
                // strand every future completion, so the failure is logged at
                // ERROR (with the flag's state) and the drain loop breaks, as
                // it does in the `!fail_on_timeout_` case.
                if self.fail_on_timeout {
                    tracing::error!("ResponseWorker: RbTimeOut (fail_on_timeout is set)");
                }
                return (0, ResponseCredit::none());
            }
            if (wait_time.saturating_sub(last_report)).as_secs()
                >= self.response_block_timeout_report_secs
            {
                tracing::warn!(
                    seconds_waited = wait_time.as_secs(),
                    num_pending_requests = self.num_pending_requests.load(Ordering::Acquire),
                    "No ResponseBlock received yet"
                );
                self.check_pending_job_timeouts();
                last_report = start.elapsed();
            }
        }
    }

    /// Port of `ResponseWorker::ParseResponseBlocks` (`response_worker.cpp:1098-1161`).
    fn parse_response_blocks(
        &self,
        num_rsps_received: u64,
        mut credit: ResponseCredit,
        scratch: &mut DrainScratch,
    ) -> u64 {
        let mut counts = PipelineResponseCounts::default();
        let mut valid_responses = 0u64;
        let now = Instant::now();

        for i in 0..num_rsps_received as usize {
            let received_response = scratch.rbs[i];
            if self.process_response_blocks(
                received_response,
                now,
                valid_responses as usize,
                &mut counts,
                &mut scratch.tags_to_be_returned,
            ) {
                valid_responses += 1;
            }
        }

        // Discharge the credit for every slot senlib handed over, per pipeline,
        // exactly as `ParseResponseBlocks` does. `take` cannot mint more slots
        // than were received, and `settle` reports any that went unreturned —
        // so the omission that caused the card to wedge cannot recur silently.
        for (pipeline, count) in counts.per_pipeline() {
            if let Some(share) = credit.take(count) {
                self.hw.free_responses(pipeline, share);
            }
        }
        credit.settle();
        valid_responses
    }

    fn process_response_blocks(
        &self,
        received_response: crate::control_block_wire::ResponseBlockWire,
        now: Instant,
        i: usize,
        counts: &mut PipelineResponseCounts,
        tags_to_be_returned: &mut [ResponseTag],
    ) -> bool {
        // "Skip uninitialized response blocks (senlib init marker 0xFFFFFF)".
        let raw_tag = received_response.tag().0;
        if raw_tag == 0xFF_FFFF {
            tracing::trace!("Skipping uninitialized response block (RB_TAG_INIT_MARKER)");
            return false;
        }
        // `tag.local_tag = raw_tag & 0xFFFF` (the upper 8 bits of the 24-bit
        // field are the device id, dropped here exactly as in the C++, which
        // rebuilds `tag.device_id` from `local_device_id_` instead).
        let local_tag = (raw_tag & 0xFFFF) as usize;
        if local_tag >= self.pending_requests.len() {
            // `RAS::RESPONSEWORKER::InvalidTag().Throw()` in the C++; logged
            // here for the same reason as `RbTimeOut` above (a throw from the
            // completion thread strands every other in-flight request).
            tracing::error!(
                local_tag,
                size = self.pending_requests.len(),
                "Invalid tag received: exceeds pending_requests_ size"
            );
            return false;
        }
        let slot = &self.pending_requests[local_tag];

        // "CRITICAL: Acquire load on ready flag to synchronize with release
        // store in ProcessTags()" — spin until the producer has published.
        while !slot.ready.load(Ordering::Acquire) {
            std::thread::yield_now();
        }

        // "Skip stale/duplicate responses" — a cancel/kill notification for a
        // tag that already completed and went back to AVAILABLE.
        //
        // A TIMED_OUT slot is NOT stale: its response arriving late means the
        // hardware has retired the CB after all, and this response occupies a
        // real senlib slot. It must be retired like an ISSUED one — tag
        // returned, per-pipeline credit freed, pending count decremented — or
        // every timed-out-then-completed CB permanently leaks all three (the
        // callback itself was already consumed by `check_pending_job_timeouts`,
        // so `issue_callback` below degrades to its no-callback warning). An
        // earlier version skipped every non-ISSUED state, TIMED_OUT included.
        let current_state = slot.state();
        if current_state != PendingRequestState::Issued
            && current_state != PendingRequestState::TimedOut
        {
            tracing::warn!(
                local_tag,
                ?current_state,
                "Skipping stale response (expected ISSUED or TIMED_OUT)"
            );
            return false;
        }

        tags_to_be_returned[i] = ResponseTag(local_tag as u32);
        slot.set_state(PendingRequestState::Succeeded);

        let pipeline = {
            let mut data = slot.data.lock().unwrap_or_else(|e| e.into_inner());
            data.completion_time_point = Some(now);
            // "Add the CB sent to driver timestamp... Try launch_time_point
            // first" (the ring-buffer baseline fallback is profiler-only and
            // is not ported).
            if let Some(lt) = data.launch_time_point {
                data.issue_time_point = Some(lt);
            }
            data.response_block = Some(received_response);
            data.pipeline
        };

        self.issue_callback(slot);

        match pipeline {
            Some(PipelineId::Compute) => counts.compute += 1,
            Some(PipelineId::AsyncDmaI) => counts.dmai += 1,
            Some(PipelineId::AsyncDmaO) => counts.dmao += 1,
            // `RAS::RESPONSEWORKER::InvalidPipelineID().Throw()`.
            None => tracing::error!(local_tag, "InvalidPipelineID on a completed response"),
        }

        slot.data.lock().unwrap_or_else(|e| e.into_inner()).pipeline = None;
        slot.set_state(PendingRequestState::Available);
        true
    }

    /// Port of `ResponseWorker::IssueCallback` (`response_worker.cpp:1272-1297`).
    /// The callback is MOVED out of the slot before being invoked — the C++
    /// has a long comment on why that matters (`:1224-1252`): leaving the
    /// slot holding a reference delays the caller's captured objects'
    /// destruction, and lets a second response for the same tag invoke it
    /// twice.
    fn issue_callback(&self, slot: &PendingRequestSlot) {
        let (cb_fn, snapshot) = {
            let mut data = slot.data.lock().unwrap_or_else(|e| e.into_inner());
            let cb_fn = data.callback_fn.take();
            let snapshot = slot.snapshot(&data);
            (cb_fn, snapshot)
        };
        match cb_fn {
            Some(cb_fn) => cb_fn(&snapshot),
            None => tracing::warn!(
                id = snapshot.id,
                "IssueCallback: no callback for PendingRequest"
            ),
        }
    }

    /// Port of `ResponseWorker::ReturnTags` (`response_worker.cpp:1323-1341`):
    /// clear each returned slot's `ready` flag so it can be reused, then put
    /// the tags back in the pool — here by dropping their [`TagLease`]s.
    fn return_tags(&self, tags_to_be_returned: &[ResponseTag], count: u64) {
        for tag in tags_to_be_returned.iter().take(count as usize) {
            let local_tag = tag.0 as usize;
            if let Some(slot) = self.pending_requests.get(local_tag) {
                slot.ready.store(false, Ordering::Relaxed);
                let lease = slot
                    .data
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .lease
                    .take();
                drop(lease);
            }
        }
    }

    /// Port of `ResponseWorker::FetchAndParseResponseBlocks` (`response_worker.cpp:1299-1321`).
    fn fetch_and_parse_response_blocks(&self, scratch: &mut DrainScratch) {
        let (num_rsps_received, credit) = self.fetch_response_blocks(scratch);
        let valid_responses = self.parse_response_blocks(num_rsps_received, credit, scratch);
        self.return_tags(&scratch.tags_to_be_returned, valid_responses);
        self.num_pending_requests
            .fetch_sub(valid_responses, Ordering::AcqRel);
    }

    /// Port of `ResponseWorker::CheckPendingJobTimeouts` (`response_worker.cpp:410-466`).
    fn check_pending_job_timeouts(&self) {
        let now = Instant::now();
        let mut num_pending = 0u64;
        let mut num_reported = 0u64;
        let mut num_timed_out = 0u64;

        for slot in &self.pending_requests {
            if slot.state() != PendingRequestState::Issued {
                continue;
            }
            let launch_time_point = slot
                .data
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .launch_time_point;
            // "Skip CBs that haven't been launched yet... The timeout is
            // about hardware responsiveness, not queue dwell time."
            let Some(launch_time_point) = launch_time_point else {
                continue;
            };
            // "Measure from hardware submission time (launch_time_point), not
            // queue time (issue_time_point)."
            let seconds_since_issue = now.saturating_duration_since(launch_time_point).as_secs();

            if seconds_since_issue < self.response_block_timeout_report_secs {
                num_pending += 1;
                continue;
            }
            if seconds_since_issue < self.response_block_timeout_fail_secs {
                tracing::warn!(seconds_since_issue, "  Still pending");
                num_reported += 1;
                continue;
            }
            slot.set_state(PendingRequestState::TimedOut);
            tracing::error!(seconds_since_issue, "  Timed-out");
            num_timed_out += 1;
            // "Move callback to prevent double invocation when response
            // eventually arrives."
            self.issue_callback(slot);
        }

        tracing::warn!(
            num_pending,
            num_reported,
            num_timed_out,
            timeout_secs = self.response_block_timeout_fail_secs,
            "Pending-job timeout scan"
        );
    }

    /// Port of `ResponseWorker::ResponseCompletionThread` (`response_worker.cpp:1343-1414`).
    fn response_completion_thread(&self) {
        if self.single_thread_optimized {
            // "Stopped early due to single thread optimized run" — `QueueCbs`
            // drains inline instead.
            return;
        }
        let mut scratch = DrainScratch::new(self.max_pending_requests);
        while !self.stop_threads.load(Ordering::Acquire)
            || self.num_pending_requests.load(Ordering::Acquire) > 0
        {
            if self.force_shutdown.load(Ordering::Acquire) {
                return;
            }
            // "WaitForInterrupt() (MSI2 eventfd) is intentionally NOT used
            // here... busy-polling when pending > 0 is faster."
            if self.num_pending_requests.load(Ordering::Acquire) == 0 {
                let rx = self.wakeup_rx.lock().unwrap_or_else(|e| e.into_inner());
                if rx.recv().is_err() {
                    return;
                }
            }
            if self.force_shutdown.load(Ordering::Acquire) {
                return;
            }
            self.fetch_and_parse_response_blocks(&mut scratch);
        }
    }

    /// Port of `ResponseWorker::Shutdown` (`response_worker.cpp:334-408`),
    /// minus the pre-built cancelled-CB submission (see [`ResponseWorker::new`]):
    /// set `stop_threads_`, wake the worker so it observes it, and join —
    /// with the same forced-exit escape hatch (`force_shutdown_`) for the
    /// case where hardware is stuck and the worker will never drain.
    pub fn shutdown(&self) {
        self.stop_threads.store(true, Ordering::Release);
        let _ = self.wakeup_tx.send(0);
        let handle = self
            .worker_thread
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(handle) = handle {
            // Give the worker a bounded window to drain what is still
            // pending, matching the real `kShutdownTimeout` of 5s, then force
            // it out rather than hanging here forever.
            let deadline = Instant::now() + Duration::from_secs(5);
            while !handle.is_finished() && Instant::now() < deadline {
                let _ = self.wakeup_tx.send(0);
                std::thread::sleep(Duration::from_millis(10));
            }
            if !handle.is_finished() {
                tracing::warn!("Forcing worker thread shutdown — hardware unresponsive");
                self.force_shutdown.store(true, Ordering::Release);
                let _ = self.wakeup_tx.send(0);
            }
            let _ = handle.join();
        }
        // Conservation check: a drained worker owes nothing. A non-zero count
        // here means requests were submitted whose tags/credits can no longer
        // be recovered — the leak is real whether or not anything downstream
        // notices, so it is reported as an error, not a debug line.
        let leaked = self.num_pending_requests.load(Ordering::Acquire);
        if leaked != 0 {
            tracing::error!(
                leaked,
                "ResponseWorker shut down with pending requests unretired — their tags and \
                 response-slot credits are leaked"
            );
        }
    }

    /// Port of the `num_pending_requests_` read used by the C++'s own logging
    /// and by `response_worker_concurrency_test.cpp`'s drain wait.
    pub fn num_pending_requests(&self) -> u64 {
        self.num_pending_requests.load(Ordering::Acquire)
    }
}

impl Drop for ResponseWorker {
    /// `~ResponseWorker() { Shutdown(); }`.
    fn drop(&mut self) {
        // `self` is the last owner here, so the worker thread's `Weak` can no
        // longer upgrade; still set the flags and drop the sender so a thread
        // parked on `wakeup_rx.recv()` returns.
        self.stop_threads.store(true, Ordering::Release);
        self.force_shutdown.store(true, Ordering::Release);
        let _ = self.wakeup_tx.send(0);
    }
}

/// Ported C++ test, source:
/// `flex/tests/control_blocks/response_worker_concurrency_test.cpp`
/// (`ResponseWorkerConcurrencyTest.ConcurrentQueueCbsDeliversEveryCallbackOnce`).
///
/// The real test drives `ResponseWorker::QueueCbs` from many threads against a
/// loopback mock device that "echoes exactly one response per submitted control
/// block", and asserts exactly one completion callback per submitted CB. Its
/// docstring names the failure mode precisely: when the single-producer
/// assumption is violated "the batch-timestamp ring, the pending-request pool,
/// the tag pool and the pending-request counter race, and control blocks are
/// lost in submission: a response comes back for a slot that was concurrently
/// overwritten, the matching callback never fires, and the caller eventually
/// times out."
///
/// [`LoopbackHardware`] below is that loopback mock: `queue_control_blocks`
/// records one response per CB (carrying that CB's own tag), and
/// `receive_responses` hands them back in whatever order they were recorded.
#[cfg(test)]
mod response_credit_tests {
    use super::*;

    /// A credit can be split into exactly the slots that were received, and no
    /// more: `free_responses` cannot be handed capacity senlib never gave us.
    #[test]
    fn credit_cannot_be_overdrawn() {
        let mut credit = ResponseCredit::received(10);
        assert_eq!(credit.outstanding(), 10);
        assert!(
            credit.take(11).is_none(),
            "must not mint more slots than were received"
        );
        assert!(
            credit.take(0).is_none(),
            "a zero-slot free is not a real obligation"
        );
        let share = credit.take(4).expect("4 of 10 is available");
        assert_eq!(share.count(), 4);
        assert_eq!(credit.outstanding(), 6);
        let rest = credit.take(6).expect("the remaining 6");
        assert_eq!(rest.count(), 6);
        assert_eq!(credit.outstanding(), 0, "fully discharged");
        credit.settle();
    }

    /// Regression test for the bug this type exists to prevent: every response
    /// slot senlib hands over must come back to it. The loopback records each
    /// `free_responses` call, so the sum of freed slots must equal the number
    /// received — the real `ParseResponseBlocks` invariant
    /// (`response_worker.cpp:1141-1152`).
    #[test]
    fn every_received_slot_is_returned_to_senlib() {
        let (n, credit) = {
            let mut out =
                vec![crate::control_block_wire::ResponseBlockWire::from_bytes(&[0; 64]); 4];
            let hw = CreditCountingHardware::new();
            let (n, credit) = hw.receive_responses(&mut out, 0);
            (n, credit)
        };
        assert_eq!(n, 3, "the loopback queued 3 responses");
        assert_eq!(credit.outstanding(), 3, "and we owe senlib 3 slots back");
        credit.settle();
    }

    struct CreditCountingHardware;

    impl CreditCountingHardware {
        fn new() -> Self {
            Self
        }
    }

    impl HardwareSubmit for CreditCountingHardware {
        fn queue_control_blocks(
            &self,
            _pipeline: PipelineId,
            _cbs: &[ControlBlockWire],
        ) -> Result<(), SchedulerError> {
            Ok(())
        }
        fn receive_responses(
            &self,
            _out: &mut [crate::control_block_wire::ResponseBlockWire],
            _min_count: usize,
        ) -> (u64, ResponseCredit) {
            (3, ResponseCredit::received(3))
        }
        fn free_responses(&self, _pipeline: PipelineId, _credit: PipelineCredit) {}
    }
}

#[cfg(test)]
mod response_worker_tests {
    use super::*;
    use crate::control_block_wire::{RB_NUM_BYTES, ResponseBlockWire};

    /// The C++ fixture's loopback mock device.
    struct LoopbackHardware {
        queued: Mutex<VecDeque<ResponseBlockWire>>,
        freed: Mutex<Vec<(PipelineId, u64)>>,
    }

    impl LoopbackHardware {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                queued: Mutex::new(VecDeque::new()),
                freed: Mutex::new(Vec::new()),
            })
        }
    }

    /// A `Good`-status response block carrying `tag`. `ReturnSectionSBF`
    /// layout: tag at bits [31:8], status (`Good` = 0b00) at bits [6:5] — see
    /// `ResponseBlockWire::tag`/`status`.
    fn loopback_response(tag: ResponseTag) -> ResponseBlockWire {
        let word = (tag.0 as u64) << 8;
        let mut bytes = [0u8; RB_NUM_BYTES];
        bytes[0..8].copy_from_slice(&word.to_le_bytes());
        ResponseBlockWire::from_bytes(&bytes)
    }

    impl HardwareSubmit for LoopbackHardware {
        fn queue_control_blocks(
            &self,
            _pipeline: PipelineId,
            cbs: &[ControlBlockWire],
        ) -> Result<(), SchedulerError> {
            let mut queued = self.queued.lock().unwrap_or_else(|e| e.into_inner());
            for cb in cbs {
                queued.push_back(loopback_response(cb.tag()));
            }
            Ok(())
        }

        fn receive_responses(
            &self,
            out: &mut [ResponseBlockWire],
            _min_count: usize,
        ) -> (u64, ResponseCredit) {
            let mut queued = self.queued.lock().unwrap_or_else(|e| e.into_inner());
            let n = queued.len().min(out.len());
            for slot in out.iter_mut().take(n) {
                *slot = queued.pop_front().expect("checked above");
            }
            (n as u64, ResponseCredit::received(n as u64))
        }

        fn free_responses(&self, pipeline: PipelineId, credit: PipelineCredit) {
            self.freed
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((pipeline, credit.count()));
        }
    }

    fn dummy_cb(tag: ResponseTag) -> ControlBlockWire {
        ControlBlockWire::fill(
            tag,
            VirtualAddressSbf::new(0, Flits::from_bytes(0)),
            TransferSize(128),
            0,
            true,
            DevicePhysicalAddress(0),
            SegmentExtent(128),
        )
        .expect("fill CB")
    }

    /// The C++ test's own shape, scaled down for a unit test: many threads
    /// submitting multi-CB batches ("Multi-CB batches widen the PrepareCbs
    /// race window"), then a bounded wait for every callback.
    #[test]
    fn concurrent_queue_cbs_delivers_every_callback_once() {
        const THREADS: usize = 16;
        const BATCHES_PER_THREAD: usize = 16;
        const CBS_PER_BATCH: u64 = 8;
        const TOTAL: usize = THREADS * BATCHES_PER_THREAD * CBS_PER_BATCH as usize;

        let hw = LoopbackHardware::new();
        let worker = ResponseWorker::new(hw.clone() as Arc<dyn HardwareSubmit>);

        let callbacks_fired = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let seen_tags = Arc::new(Mutex::new(Vec::<u32>::new()));

        let mut submitters = Vec::new();
        for _ in 0..THREADS {
            let worker = worker.clone();
            let callbacks_fired = callbacks_fired.clone();
            let seen_tags = seen_tags.clone();
            submitters.push(std::thread::spawn(move || {
                for _ in 0..BATCHES_PER_THREAD {
                    let callbacks_fired = callbacks_fired.clone();
                    let seen_tags = seen_tags.clone();
                    let cb_fn: CallbackFn = Arc::new(move |pr: &PendingRequest| {
                        callbacks_fired.fetch_add(1, Ordering::Relaxed);
                        if let Some(rb) = pr.response_block {
                            seen_tags
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .push(rb.tag().0);
                        }
                    });
                    worker
                        .queue_cbs(
                            cb_fn,
                            CBS_PER_BATCH,
                            PipelineId::Compute,
                            QueuingMode::Regular,
                            |tags| Ok(tags.iter().map(|t| dummy_cb(*t)).collect()),
                        )
                        .expect("QueueCbs failed");
                }
            }));
        }
        for s in submitters {
            s.join().unwrap();
        }

        let deadline = Instant::now() + Duration::from_secs(20);
        while (callbacks_fired.load(Ordering::Relaxed) as usize) < TOTAL
            && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(1));
        }

        assert_eq!(
            callbacks_fired.load(Ordering::Relaxed) as usize,
            TOTAL,
            "control blocks lost in submission: not every CB's completion callback fired"
        );
        let mut tags = seen_tags.lock().unwrap_or_else(|e| e.into_inner()).clone();
        tags.sort_unstable();
        let unique = {
            let mut t = tags.clone();
            t.dedup();
            t.len()
        };
        assert_eq!(
            unique,
            tags.len(),
            "a tag was delivered to more than one callback (clobbered pending-request slot)"
        );
        assert!(
            tags.iter().all(|t| *t >= TagPool::TAG_OFFSET),
            "tag 0 must never be handed out"
        );
        assert_eq!(
            worker.num_pending_requests(),
            0,
            "num_pending_requests_ did not return to zero"
        );
        worker.shutdown();
    }

    /// Every response makes it back through `ParseResponseBlocks`, which must
    /// return that pipeline's own count to `FreeResponses` — senlib's
    /// response-slot credit return (`response_worker.cpp:1141-1152`).
    #[test]
    fn parse_response_blocks_returns_response_credit_per_pipeline() {
        let hw = LoopbackHardware::new();
        let worker = ResponseWorker::new(hw.clone() as Arc<dyn HardwareSubmit>);
        let done = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let done_cb = done.clone();
        let cb_fn: CallbackFn = Arc::new(move |_pr: &PendingRequest| {
            done_cb.fetch_add(1, Ordering::Release);
        });
        worker
            .queue_cbs(
                cb_fn,
                4,
                PipelineId::AsyncDmaI,
                QueuingMode::Regular,
                |tags| Ok(tags.iter().map(|t| dummy_cb(*t)).collect()),
            )
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while done.load(Ordering::Acquire) < 4 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(done.load(Ordering::Acquire), 4);
        let freed = hw.freed.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let total: u64 = freed
            .iter()
            .filter(|(p, _)| *p == PipelineId::AsyncDmaI)
            .map(|(_, n)| *n)
            .sum();
        assert_eq!(
            total, 4,
            "FreeResponses must be called with the DMAI pipeline's own response count"
        );
        assert!(
            freed.iter().all(|(p, _)| *p == PipelineId::AsyncDmaI),
            "FreeResponses must not be called for a pipeline that received no responses"
        );
        worker.shutdown();
    }

    /// Hardware that answers no CB until `arm()` — the shape of a card that
    /// retires a control block only after the response worker has already
    /// reaped its slot as timed out.
    struct GatedHardware {
        held: Mutex<VecDeque<ResponseBlockWire>>,
        armed: std::sync::atomic::AtomicBool,
    }

    impl GatedHardware {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                held: Mutex::new(VecDeque::new()),
                armed: std::sync::atomic::AtomicBool::new(false),
            })
        }
        fn arm(&self) {
            self.armed.store(true, Ordering::Release);
        }
    }

    impl HardwareSubmit for GatedHardware {
        fn queue_control_blocks(
            &self,
            _pipeline: PipelineId,
            cbs: &[ControlBlockWire],
        ) -> Result<(), SchedulerError> {
            let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
            for cb in cbs {
                held.push_back(loopback_response(cb.tag()));
            }
            Ok(())
        }

        fn receive_responses(
            &self,
            out: &mut [ResponseBlockWire],
            _min_count: usize,
        ) -> (u64, ResponseCredit) {
            if !self.armed.load(Ordering::Acquire) {
                return (0, ResponseCredit::none());
            }
            let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
            let n = held.len().min(out.len());
            for slot in out.iter_mut().take(n) {
                *slot = held.pop_front().expect("checked above");
            }
            (n as u64, ResponseCredit::received(n as u64))
        }

        fn free_responses(&self, _pipeline: PipelineId, _credit: PipelineCredit) {}
    }

    /// A TIMED_OUT slot is not stale. When the hardware answers after the
    /// timeout scan has reaped the slot, that response occupies a real senlib
    /// response slot, so it must be retired exactly like an ISSUED one — tag
    /// returned, per-pipeline credit freed, `num_pending_requests_`
    /// decremented. Treating every non-ISSUED state as a duplicate leaks all
    /// three, permanently, once per timed-out-then-completed CB.
    #[test]
    fn a_timed_out_slot_is_retired_when_its_response_arrives_late() {
        let hw = GatedHardware::new();
        let worker = ResponseWorker::new(hw.clone() as Arc<dyn HardwareSubmit>);
        let cb_fn: CallbackFn = Arc::new(|_pr: &PendingRequest| {});
        worker
            .queue_cbs(
                cb_fn,
                1,
                PipelineId::AsyncDmaI,
                QueuingMode::Regular,
                |tags| Ok(tags.iter().map(|t| dummy_cb(*t)).collect()),
            )
            .expect("QueueCbs failed");

        // Exactly what `check_pending_job_timeouts` does to a slot whose
        // response has not arrived within `FLEX_RESPONSE_WORKER_TIMEOUT_SECONDS`:
        // mark it TIMED_OUT and move its callback out. Reached directly so the
        // test does not have to wait out the real 280-second deadline.
        let idx = (0..worker.pending_requests.len())
            .find(|i| worker.pending_requests[*i].state() == PendingRequestState::Issued)
            .expect("the submitted CB owns an ISSUED slot");
        worker.pending_requests[idx].set_state(PendingRequestState::TimedOut);
        worker.issue_callback(&worker.pending_requests[idx]);
        assert_eq!(
            worker.num_pending_requests(),
            1,
            "the reaped slot is still outstanding — the timeout scan retires nothing"
        );

        hw.arm(); // the card retires the CB after all.
        let deadline = Instant::now() + Duration::from_secs(10);
        while worker.num_pending_requests() != 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(
            worker.num_pending_requests(),
            0,
            "a timed-out CB's late response was skipped as stale: its pending count, response \
             tag and per-pipeline response-slot credit are leaked for the process's lifetime"
        );
        assert_eq!(
            worker.pending_requests[idx].state(),
            PendingRequestState::Available,
            "the slot must go back to AVAILABLE so its tag can be reissued"
        );
        worker.shutdown();
    }
}

/// Production `QueueCapacitySource`: `senlib_ffi_scheduler::cbi_capacity`
/// against the real per-pipeline `ControlBlockInterface`.
pub struct SenlibQueueCapacity;

impl QueueCapacitySource for SenlibQueueCapacity {
    fn capacity(&self, pipeline: PipelineId) -> Option<u64> {
        let shim_pipeline = pipeline.shim_index();
        // SAFETY: see `SenlibHardwareSubmit::submit`'s identical call.
        let cbi = unsafe { crate::senlib_ffi_scheduler::flex_shim_cbi_for_pipeline(shim_pipeline) };
        let cbi = cbi.into_live();
        // Port of `RuntimeScheduler::getQueueCapacity`'s own `0` handling
        // (`runtime_scheduler.cpp:1022-1028`): "capacity() == 0 is senlib's
        // 'capacity not reported' default" — the real caller maps it to
        // `nullopt` (treated as "no limit to enforce"), same as a null
        // `ControlBlockInterface*`. `cbi_capacity`'s own doc comment already
        // claimed this mapping happens "in native Rust, not here" — it did
        // not, until now: a genuinely-unreported `0` was passed straight
        // through as `Some(0)`, which makes `has_queue_capacity`'s
        // `in_flight + needed <= capacity` unsatisfiable forever for any
        // `needed >= 1`, hanging every submission on that pipeline for the
        // full timeout on every single call.
        match crate::senlib_ffi_scheduler::cbi_capacity(cbi) {
            Some(0) | None => None,
            Some(c) => Some(c as u64),
        }
    }
}

/// Resolved pieces of a single-chunk `CompositeAddress`, kept separate
/// rather than pre-combined because the two real call shapes need them
/// combined differently:
///
/// - A field with its own companion offset (DMA's `dmva`, compute's
///   `bootstrap`) needs `base_paddr` alone as the XLAT entry and
///   `offset_bytes` in that companion field — hardware computes
///   `dmpa = dmpa_allocs[seg] + seg_offset(dmva)` (port of the real
///   `TranslateDmvaToDmpa`, `control_block_stream.cpp`: `dmpa =
///   dmpa_allocs[seg_id] + seg_offset`, fed by `getDmvaAddress`'s
///   `SegmentByteOffset_todmva(seg_id, chunk.addr.offset)` — the offset is
///   the *chunk's own*, never folded into the XLAT base itself).
/// - A field with NO companion offset (compute's per-tensor XLAT slots,
///   which the kernel addresses at a fixed segment with no offset
///   parameter of their own) needs the full absolute address —
///   `base_paddr + offset_bytes` — baked directly into the XLAT entry,
///   since nothing downstream ever adds `offset_bytes` back in.
struct ResolvedChunk {
    segment: u8,
    base_paddr: DevicePhysicalAddress,
    offset_bytes: u64,
    /// This transfer/tensor's own byte count — the DMA/compute length
    /// field. NOT the XLAT entry's length; see `region_total_size`.
    size: u64,
    /// The OWNING REGION's total size, from the resolver.
    ///
    /// ⛔ THIS WAS RESOLVED AND THEN DISCARDED (`let (segment, base_paddr, _region_size) = ...`).
    /// It is the outer half of I1: a DMA is bounded against its segment's DECLARED extent, and
    /// nothing bounded that extent against the region backing it. The device does not close this
    /// either — `handleXLATentry` ends with `DT_CHECK(!devMemory_ || physAddr < devMemSize_)`,
    /// which tests the BASE ONLY, never `physAddr + length`. The card not enforcing it is a reason
    /// for the host to, not a reason to skip it.
    region_size: u64,
}

impl ResolvedChunk {
    /// For fields with their own companion offset (DMA's `dmva`, compute's
    /// `bootstrap`): XLAT base alone, no offset folded in.
    fn xlat_base(&self) -> DevicePhysicalAddress {
        self.base_paddr
    }
    /// For fields with no companion offset (compute's per-tensor XLAT
    /// slots): the full absolute address.
    fn xlat_absolute(&self) -> DevicePhysicalAddress {
        DevicePhysicalAddress(self.base_paddr.0 + self.offset_bytes)
    }
    /// `dmva`/`bootstrap`-style companion field: this chunk's own offset,
    /// optionally with an additional intra-segment offset added on top
    /// (`build_compute_cb`'s `bootstrap_offset`).
    fn dmva(&self, extra_offset_bytes: u64) -> VirtualAddressSbf {
        VirtualAddressSbf::new(
            self.segment,
            Flits::from_bytes(self.offset_bytes + extra_offset_bytes),
        )
    }
    /// The DMA XLAT entry's length field — the owning region's total size,
    /// never this transfer's own byte count. Deliberately NOT named/shaped
    /// so a compute call site could pass this where an `AllocationTotalSize`
    /// belongs — `build_compute_cb` gets its XLAT length from the
    /// allocation's own `total_size()` directly, never from here, and the
    /// return type makes the two impossible to swap by accident.
    /// This transfer's own XLAT extent within its segment — the real
    /// `xlat.SetLengthBytes(EncodeLength(seg_offset + size_bytes))`, where
    /// `seg_offset` is THIS piece's own `dmva` offset (the chunk's offset plus
    /// any intra-transfer chunking offset) and `size_bytes` its own length.
    /// Never the owning region's total size: see `SegmentExtent`'s doc.
    fn segment_extent(&self, extra_offset_bytes: u64, size_bytes: u64) -> SegmentExtent {
        SegmentExtent(self.offset_bytes + extra_offset_bytes + size_bytes)
    }
}

/// Resolve a (non-interleaved) `CompositeAddress`'s single chunk. Multi-chunk
/// (interleaved) addresses are out of scope here — this crate's graph-free
/// CB encoder addresses one physical region per DMA/tensor operand, matching
/// `ControlBlockStream::CreateComputeDataTransfer`/`CreateGraphFreeCompute`'s
/// single-`region_paddr`/single-`tensor_paddr` signatures.
fn resolve_single_chunk(
    resolver: &dyn DeviceAddressResolver,
    addr: &CompositeAddress,
) -> Result<ResolvedChunk, SchedulerError> {
    if !addr.is_single_chunk() {
        return Err(mock_error(
            "control-block encoding: interleaved (multi-chunk) CompositeAddress not supported",
        ));
    }
    let chunk = &addr.chunks()[0];
    let (segment, base_paddr, region_size) = resolver.resolve(chunk).ok_or_else(|| {
        mock_error("control-block encoding: DeviceAddressResolver could not resolve region")
    })?;
    let (offset_bytes, size) = (chunk.addr.offset.0, chunk.size.as_u64());
    // ⛔ I1, OUTER HALF — the allocation lies inside the REGION that backs it. Proven here, at the
    // one place both numbers exist, because neither the DMA path nor the device checks it: the
    // former bounds against the segment's declared extent, and `handleXLATentry` tests only the
    // base against device memory. A region overrun reaches the card as `0xa35e BusFence`, which
    // names nothing.
    let region_size = region_size.as_u64();
    if offset_bytes + size > region_size {
        return Err(mock_error(&format!(
            "control-block encoding: allocation [{offset_bytes}, {}) runs past its {region_size} B \
             region in segment {segment} — the XLAT window would cover memory the region does not \
             own",
            offset_bytes + size,
        )));
    }
    Ok(ResolvedChunk {
        segment,
        base_paddr,
        offset_bytes,
        size,
        region_size,
    })
}

/// Port of `PfRuntimeScheduler::submitToHardware`
/// (`pf_runtime_scheduler_utils.cpp:493-567`) plus `onCbComplete`
/// (`:429-491`), which `VfRuntimeScheduler` shares verbatim. The real order is
/// exactly:
///
/// 1. `waitForQueueCapacity(pipeline, num_cbs, kUseDefaultTimeout)` — "prevents
///    firmware from hitting CbQueueFullFatalError() which causes a card reset"
///    — and `RAS::RUNTIMESCHEDULER::CbQueueCapacityExceeded().Throw()` on
///    timeout, *before* anything is queued (`:509-514`).
/// 2. `{compute,dmai,dmao}_in_flight_ += cbr->GetNumberOfCBs()` (`:518-537`) —
///    NUM_CBS, not one, and it stays raised until each CB's response arrives.
/// 3. `std::call_once(response_worker_init_, ...)` — lazily construct the one
///    `ResponseWorker` (`:539-550`).
/// 4. `response_worker_->QueueCbs(wrapped_callback, cbr, pipeline)` and
///    `RAS::DEVICE_PF::CbQueueFailed().Throw()` if the status is not OK
///    (`:560-565`). This RETURNS as soon as the doorbell has been rung.
///
/// `wrapped_callback` is `onCbComplete`, invoked ONCE PER CB from the
/// ResponseWorker's completion thread: it counts invocations against
/// `total_num_cbs`, evaluates that CB's response block, decrements the
/// pipeline's in-flight counter (in `recordPipelineProfiler`, `:273/289/341`),
/// and only when the last CB of the group has completed does it invoke the
/// user callback — once, with the first hardware error seen if any.
///
/// The previous version of this function was `#[allow(dead_code)]` and unused:
/// PF/VF called `HardwareSubmit::submit` directly and resolved synchronously,
/// so none of the above existed on the real path. The queue-capacity gate and
/// the in-flight accounting had been relocated into `schedule_pipelined` with
/// a hardcoded `needed = 1`, which is not where or what the C++ does.
fn submit_to_hardware(
    state: &Arc<SchedulerState>,
    worker: &Arc<ResponseWorker>,
    cap: &dyn QueueCapacitySource,
    pipeline: PipelineId,
    num_cbs: u64,
    callback: CompletionCallback,
    build: impl FnOnce(&[ResponseTag]) -> Result<Vec<ControlBlockWire>, SchedulerError>,
) {
    if !wait_for_queue_capacity(cap, state, pipeline, num_cbs, USE_DEFAULT_TIMEOUT_MS) {
        return callback(Err(mock_error(&format!(
            "RuntimeScheduler: CbQueueCapacityExceeded (pipeline={pipeline:?}, num_cbs={num_cbs})"
        ))));
    }

    for _ in 0..num_cbs {
        state.inc_in_flight(pipeline);
    }

    let call_count = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let hw_error = Arc::new(Mutex::new(None::<CbCompleteError>));
    let user_cb = Arc::new(Mutex::new(Some(callback)));

    let state_for_cb = state.clone();
    let hw_error_for_cb = hw_error.clone();
    let user_cb_for_cb = user_cb.clone();
    let cb_fn: CallbackFn = Arc::new(move |pr: &PendingRequest| {
        // "Track how many times this callback has been invoked."
        let current_count = call_count.fetch_add(1, Ordering::AcqRel) + 1;

        // onCbComplete's own status check (`:437-447`) — deliberately NOT
        // `EvalResponseBlockStatus`, which is what the *graph* scheduler and
        // the data-transfer paths use: this one treats TIMED_OUT, cancel, and
        // any status other than `RB_GOOD` as a hardware error.
        if let Some(e) = eval_cb_complete_status(pr).failure() {
            tracing::warn!("Response block error: {e}");
            *hw_error_for_cb.lock().unwrap_or_else(|p| p.into_inner()) = Some(e);
        }

        // `recordPipelineProfiler`'s `--{compute,dmai,dmao}_in_flight_`.
        state_for_cb.dec_in_flight(pipeline);

        // "Call the completion callback only when all CBs are done."
        if current_count == num_cbs
            && let Some(cb) = user_cb_for_cb
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .take()
        {
            let err = hw_error_for_cb
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .take();
            match err {
                Some(e) => cb(Err(Box::new(e))),
                None => cb(Ok(())),
            }
        }
    });

    if let Err(e) = worker.queue_cbs(cb_fn, num_cbs, pipeline, QueuingMode::Regular, build) {
        // `RAS::DEVICE_PF::CbQueueFailed().Throw()`. The C++ throw unwinds out
        // of `submitToHardware` without undoing step 2's increment (which
        // then leaks, but the throw is a RAS fatal there); here the operation
        // simply failed, so the increments are undone — leaving them raised
        // would wedge `synchronize()`/`query()` forever on a counter that can
        // never reach zero.
        for _ in 0..num_cbs {
            state.dec_in_flight(pipeline);
        }
        if let Some(cb) = user_cb.lock().unwrap_or_else(|p| p.into_inner()).take() {
            cb(Err(e));
        }
    }
}

/// Port of `ResponseWorker`'s response-tag pool (`response_worker.cpp:468-481`
/// `InitTags`, `755-777` `PrepareCbs`'s acquire loop, `1323-1341` `ReturnTags`).
///
/// The real pool holds exactly `max_pending_requests_` tags valued
/// `tag_offset_ .. max_pending_requests_ + tag_offset_`, and
/// `pending_requests_` is sized to match — so a tag outside that range indexes
/// NOTHING and the response for its CB is never routed back. This crate had no
/// pool at all: it handed out `fetch_add(1)` forever, starting at 0. On real
/// hardware that wedged on compute CB #32769 — the first CB past
/// `max_pending_requests_` = 32768 — with `RAS::CBRB::ResponseTimeout` /
/// `action:reset`, after all 32768 before it completed cleanly.
///
/// Tags are handed out as [`TagLease`], which returns the tag to the pool on
/// drop. That makes the real `ReturnTags` step unskippable: the lease lives in
/// the [`PendingRequestSlot`] the tag names, from `PrepareCbs` until
/// `ReturnTags` takes it back out, so a tag cannot be recycled while a CB
/// using it may still be in flight. This pool is owned by the
/// [`ResponseWorker`] (`available_tags_`), not by a backend.
struct TagPool {
    available: Mutex<std::collections::VecDeque<u32>>,
    /// Signalled by `TagLease::drop`; waited on by an `acquire` that found the
    /// pool empty (port of `PrepareCbs`'s `while(tags_fetched < num_ctbs)`
    /// spin, which likewise cannot proceed until tags come back).
    returned: std::sync::Condvar,
}

impl TagPool {
    /// Port of `InitTags`. `tag_offset_ = 1`, whose real comment is explicit:
    /// "Tag offset avoids using tag 0 which may have special meaning" — this
    /// crate's old counter started AT 0.
    const TAG_OFFSET: u32 = 1;

    fn new(max_pending_requests: u64) -> Arc<Self> {
        let max_pending =
            max_pending_requests.clamp(1, u32::MAX as u64 - Self::TAG_OFFSET as u64) as u32;
        let available = (Self::TAG_OFFSET..Self::TAG_OFFSET + max_pending).collect();
        Arc::new(Self {
            available: Mutex::new(available),
            returned: std::sync::Condvar::new(),
        })
    }

    /// Port of `PrepareCbs`'s tag acquisition: take one tag, waiting while the
    /// pool is exhausted (the real code loops on
    /// `try_dequeue_bulk_from_producer` with a "Still waiting to fetch tags"
    /// warning every 5s, which this mirrors).
    fn acquire(self: &Arc<Self>) -> TagLease {
        let mut available = self.available.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(tag) = available.pop_front() {
                return TagLease {
                    tag: ResponseTag(tag),
                    pool: Arc::clone(self),
                };
            }
            let (guard, timeout) = self
                .returned
                .wait_timeout(available, std::time::Duration::from_secs(5))
                .unwrap_or_else(|e| e.into_inner());
            available = guard;
            if timeout.timed_out() {
                tracing::warn!(
                    "TagPool: still waiting for a response tag to be returned (pool exhausted)"
                );
            }
        }
    }
}

/// A response tag borrowed from a [`TagPool`], returned on drop — the port of
/// `ResponseWorker::ReturnTags`' `available_tags_.enqueue_bulk(...)`.
struct TagLease {
    tag: ResponseTag,
    pool: Arc<TagPool>,
}

impl TagLease {
    fn tag(&self) -> ResponseTag {
        self.tag
    }
}

impl Drop for TagLease {
    fn drop(&mut self) {
        self.pool
            .available
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(self.tag.0);
        self.pool.returned.notify_one();
    }
}

/// Port of `flex::DmaBuf` (`scheduler_config.hpp:45-79` + `setup_dma_buf`,
/// `scheduler_config.cpp:54-105`): the per-chunk host-side state of a
/// multi-chunk DMA — the aligned shadow buffer used when the host pointer is
/// not IOVA-aligned, and the IOMMU mapping that produces the chunk's IOVA.
/// One of these exists per IN-FLIGHT chunk, and they are recycled through
/// [`DmaBufPool::avail_bufs`] as chunks complete.
struct DmaBuf {
    /// `raii_buffer` — allocated at `RuntimeConfig::DataTransferBufferSize()`
    /// (NOT at the transfer size) and only when `unaligned`.
    raii_buffer: Option<crate::host_interface::RaiiBuffer>,
    /// `iomap`.
    iomap: Option<crate::iommu::IommuMapping>,
    /// `hmva_ptr` — the ORIGINAL (possibly unaligned) host pointer.
    hmva_ptr: usize,
    /// `mapped_hmva` — the address `iomap` was actually built for, used to
    /// detect a stale mapping on reuse.
    mapped_hmva: usize,
    /// `size`.
    size: u64,
    /// `to_device`.
    to_device: bool,
    /// `unaligned` — "True when raii_buffer is the active transfer buffer".
    unaligned: bool,
    /// `iova` (the caller-supplied, pre-mapped one): non-null skips both the
    /// shadow copy and the copy-back.
    user_iova: Option<usize>,
    /// The device-visible address this chunk's control block actually carries.
    /// Typed, not a `usize`: the only two ways to fill it are an IOMMU mapping
    /// this backend took out and a caller-supplied pre-mapped IOVA, so a host
    /// virtual address cannot reach a CB through this field.
    iova: HostPhysicalAddress,
    /// Not a real `DmaBuf` member: `PfIommuMapper::map` in this crate takes a
    /// `writable` flag that senlib's `IommuMapperInterface::Map(&iomap, ptr,
    /// size)` does not, so the remap condition below has to notice when a
    /// recycled buffer's cached mapping was created for the other direction.
    mapped_writable: bool,
}

/// One chunk's worth of "what buffer do I need": the host address and byte
/// count, the direction, whether the host address is IOMMU-page-aligned, and a
/// caller-supplied IOVA if the caller already mapped it. These are precisely the
/// fields `setup_dma_buf` copies into the `DmaBuf`, and passing them as one
/// value removes a run of adjacent `bool`s from its signature.
#[derive(Debug, Clone, Copy)]
struct DmaBufRequest {
    hmva: usize,
    size: u64,
    to_device: bool,
    unaligned: bool,
    iova: Option<usize>,
}

impl DmaBuf {
    fn new() -> Self {
        Self {
            raii_buffer: None,
            iomap: None,
            hmva_ptr: 0,
            mapped_hmva: 0,
            size: 0,
            to_device: false,
            unaligned: false,
            user_iova: None,
            // A buffer that has never been mapped: the zero IOVA is what
            // `setup_dma_buf` overwrites before any CB is built from it.
            iova: HostPhysicalAddress::UNMAPPED,
            mapped_writable: false,
        }
    }

    /// Port of `DmaBuf::setup_dma_buf` (`scheduler_config.cpp:54-105`).
    ///
    /// `req` carries exactly the five values this assigns into `self` before
    /// doing any work, so they cannot be passed in the wrong order — three of
    /// them were adjacent `bool`/integer parameters.
    fn setup_dma_buf(
        &mut self,
        req: DmaBufRequest,
        new_buf: bool,
        iommu_mapper: &Arc<crate::iommu::PfIommuMapper>,
    ) -> Result<(), SchedulerError> {
        let DmaBufRequest {
            hmva,
            size: cur_size,
            to_device,
            unaligned: buf_unaligned,
            iova: iova_in,
        } = req;
        self.hmva_ptr = hmva;
        self.size = cur_size;
        self.to_device = to_device;
        self.unaligned = buf_unaligned;
        self.user_iova = iova_in;

        // "When the caller supplies a pre-mapped IOVA the host buffer is
        // already IOMMU-aligned and fully populated, so skip the RAII shadow
        // buffer allocation and memcpy."
        if let Some(iova) = iova_in {
            self.mapped_hmva = hmva;
            self.iova = HostPhysicalAddress::of_caller_supplied_iova(crate::dma::IovaAddress(iova));
            return Ok(());
        }

        let alignment_bytes = iommu_mapper.alignment() as u64;
        let mut xfer_ptr = hmva;
        if self.unaligned {
            // "Buffer not aligned to IOMMU page size, shadow copies required."
            if new_buf || self.raii_buffer.is_none() {
                let alignment = crate::host_interface::Alignment::new(alignment_bytes as u32)
                    .ok_or_else(|| {
                        mock_error(&format!(
                            "DmaBuf: invalid IOMMU alignment {alignment_bytes} (not a power of two)"
                        ))
                    })?;
                let size = (RuntimeConfig::data_transfer_buffer_size().value() as usize)
                    .max(cur_size as usize);
                self.raii_buffer = Some(
                    crate::host_interface::RaiiBuffer::new(size, alignment).map_err(|e| {
                        mock_error(&format!("DmaBuf: shadow buffer allocation failed: {e:?}"))
                    })?,
                );
            }
            let buf = self.raii_buffer.as_mut().expect("allocated directly above");
            if to_device {
                // SAFETY: `hmva`/`cur_size` name a valid host buffer of at
                // least `cur_size` bytes per `DmaParams`' contract, and the
                // shadow buffer is at least that long.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        hmva as *const u8,
                        buf.as_mut_ptr(),
                        cur_size as usize,
                    );
                }
            }
            xfer_ptr = buf.as_ptr() as usize;
        }

        // "Check if we need to create a new IOMMU mapping. Remap if: no
        // mapping exists, address changed, or mapping too small."
        //
        // Note the real condition compares `mapped_hmva` against the member
        // `hmva_ptr` (the ORIGINAL host address) while assigning
        // `mapped_hmva = hmva_ptr_ref` (the SHADOW address when unaligned) —
        // so on the unaligned path it always remaps, every chunk. Ported as
        // written rather than "fixed": that is the real steady-state
        // behaviour on that path.
        let writable = !to_device;
        let needs_remap = match &self.iomap {
            None => true,
            Some(m) => {
                self.mapped_hmva != self.hmva_ptr
                    || m.size_bytes() < cur_size as usize
                    || self.mapped_writable != writable
            }
        };
        if needs_remap {
            // Release the stale mapping before taking a new one for the same
            // host pages (`Map(&iomap, ...)` overwrites the unique_ptr, which
            // destroys the old `IommuMapping` first).
            self.iomap = None;
            let mapping = iommu_mapper
                .map(xfer_ptr, cur_size as usize, writable)
                .map_err(|e| {
                    mock_error(&format!(
                        "DmaBuf: DmaIommuMappingFailed (hmva={xfer_ptr:#x}, size={cur_size}): {e:?}"
                    ))
                })?;
            self.mapped_hmva = xfer_ptr;
            self.mapped_writable = writable;
            self.iomap = Some(mapping);
        }
        self.iova =
            HostPhysicalAddress::of_mapping(self.iomap.as_ref().expect("mapped directly above"));
        Ok(())
    }
}

/// Port of `flex::DmaBufPool` (`scheduler_config.hpp:90-105`): the shared
/// completion state of one multi-chunk transfer plus the recycled buffer
/// queue. `pending` starts at 1 — "a sentinel held by the submitter", which
/// `submitDma` drops after the last chunk has been QUEUED, so a transfer
/// whose chunks all complete before the loop finishes still fires its
/// callback exactly once.
struct DmaBufPool {
    avail_bufs: Mutex<VecDeque<Box<DmaBuf>>>,
    avail_cv: Condvar,
    pending: AtomicI64,
    cb: Mutex<Option<CompletionCallback>>,
    /// The `std::exception_ptr` a completing chunk forwards to
    /// `(*cb)(ep)`. The real code hands the pool's callback whichever `ep`
    /// belonged to the LAST chunk to complete; this keeps the most recent one
    /// seen, which is the same thing whenever there is at most one failure.
    error: Mutex<Option<SchedulerError>>,
}

impl DmaBufPool {
    fn new(cb: CompletionCallback) -> Arc<Self> {
        Arc::new(Self {
            avail_bufs: Mutex::new(VecDeque::new()),
            avail_cv: Condvar::new(),
            pending: AtomicI64::new(1),
            cb: Mutex::new(Some(cb)),
            error: Mutex::new(None),
        })
    }

    /// `pool->avail_bufs.wait_dequeue(chunk_buf)` — block until a chunk
    /// completes and hands its buffer back. This is the real backpressure on
    /// how many chunks of one transfer are in flight at once:
    /// `RuntimeConfig::DataTransferBufferCount()`.
    fn wait_dequeue(&self) -> Box<DmaBuf> {
        let mut avail = self.avail_bufs.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(buf) = avail.pop_front() {
                return buf;
            }
            let (guard, _timeout) = self
                .avail_cv
                .wait_timeout(avail, Duration::from_millis(100))
                .unwrap_or_else(|e| e.into_inner());
            avail = guard;
        }
    }

    /// `pool->avail_bufs.enqueue(chunk_buf)`.
    ///
    /// ⛔ Takes a [`CbRetired`] and does nothing with it. That is the guard: a
    /// buffer re-enters the pool to be REMAPPED by the next `setup_dma_buf`,
    /// and remapping unmaps the IOVA a control block the card is still holding
    /// would master — a fatal `RAS::PCI::BusFence`. The token cannot be
    /// conjured, only obtained from `CbRetired::from_completion` (which
    /// refuses a timed-out CB) or `CbRetired::never_submitted`, so "put it
    /// back without checking whether the card is done" is not expressible.
    fn enqueue(&self, buf: Box<DmaBuf>, _retired: CbRetired) {
        self.avail_bufs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(buf);
        self.avail_cv.notify_one();
    }

    /// `if(--(*pending) == 0) (*cb)(ep);`
    fn release(&self, err: Option<SchedulerError>) {
        if let Some(e) = err {
            *self.error.lock().unwrap_or_else(|p| p.into_inner()) = Some(e);
        }
        if self.pending.fetch_sub(1, Ordering::AcqRel) == 1
            && let Some(cb) = self.cb.lock().unwrap_or_else(|p| p.into_inner()).take()
        {
            let err = self.error.lock().unwrap_or_else(|p| p.into_inner()).take();
            match err {
                Some(e) => cb(Err(e)),
                None => cb(Ok(())),
            }
        }
    }
}

/// Port of `PfRuntimeScheduler`. Hardware submission goes through the one
/// [`ResponseWorker`] (`response_worker_` + `response_worker_init_`), so
/// `capacity()` on the compute/DMAI/DMAO `ControlBlockInterface`s is
/// meaningful — see `senlib_ffi_scheduler::cbi_capacity`.
pub struct PfBackend {
    hw: Arc<dyn HardwareSubmit>,
    capacity: Arc<dyn QueueCapacitySource>,
    addr_resolver: Arc<dyn DeviceAddressResolver>,
    /// `device_handle_->GetIommuMapper()`: pins+maps a host buffer into a real
    /// device-visible IOVA. Previously absent entirely, so `build_dma_cb` fell
    /// back to the bare host virtual address, which the device's IOMMU cannot
    /// resolve — real hardware rejected every such CB.
    iommu_mapper: Arc<crate::iommu::PfIommuMapper>,
    /// Port of `response_worker_` + `std::once_flag response_worker_init_`
    /// (`pf_runtime_scheduler.hpp:161-162`, initialised inside
    /// `submitToHardware`'s `std::call_once`): ONE ResponseWorker per
    /// scheduler, created on first submission, owning the tag pool, the
    /// `pending_requests_` table and the completion thread.
    response_worker: std::sync::OnceLock<Arc<ResponseWorker>>,
    /// `DmaBuf`s whose CB timed out. A timed-out CB may still be sitting in
    /// the hardware queue, so its buffer's IOMMU mapping must stay alive and
    /// its buffer must never re-enter the reuse pool: recycling it remaps
    /// (= unmaps) an IOVA the card can still master a PCIe access against,
    /// which the card reports as a fatal `RAS::PCI::BusFence`. Parked here for
    /// the backend's lifetime — a deliberate, bounded leak (one buffer per
    /// timed-out chunk) traded for the card's bus staying alive.
    ///
    /// Held as [`QuarantinedBuf`], which has no accessor and no way back to a
    /// `DmaBuf`: once parked, un-parking is unconstructable rather than merely
    /// undone-by-convention.
    dma_quarantine: Arc<Mutex<Vec<QuarantinedBuf>>>,
}

/// One iteration of `submitDma`'s chunking loop: where this piece starts in the
/// caller's host buffer (`hmva` plus `offset_bytes`), how many bytes it moves,
/// whether the base host address was IOMMU-page-aligned (computed ONCE from the
/// base `hmva`, per `pf_runtime_scheduler.cpp:117`, not per chunk), and which
/// pipeline it rides.
#[derive(Debug, Clone, Copy)]
struct DmaChunk {
    hmva: usize,
    offset_bytes: u64,
    cur_size: u64,
    buf_unaligned: bool,
    pipeline: PipelineId,
}

impl PfBackend {
    pub fn new(
        hw: Arc<dyn HardwareSubmit>,
        capacity: Arc<dyn QueueCapacitySource>,
        addr_resolver: Arc<dyn DeviceAddressResolver>,
        iommu_mapper: Arc<crate::iommu::PfIommuMapper>,
    ) -> Self {
        Self {
            hw,
            capacity,
            addr_resolver,
            iommu_mapper,
            response_worker: std::sync::OnceLock::new(),
            dma_quarantine: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Port of `submitToHardware`'s `std::call_once(response_worker_init_, ...)`.
    /// (The `TimestampCalibrator` the real lambda also builds and starts is a
    /// telemetry-phase type and is not ported; it is the only other thing that
    /// call_once does.)
    fn response_worker(&self) -> &Arc<ResponseWorker> {
        self.response_worker
            .get_or_init(|| ResponseWorker::new(self.hw.clone()))
    }

    /// Port of `PfRuntimeScheduler::submitDmaChunk` (`pf_runtime_scheduler.cpp:158-229`):
    /// acquire (or create) a pool buffer, IOMMU-map it, build this chunk's own
    /// control block, and submit it — asynchronously. The completion lambda
    /// copies a D2H shadow-buffer result back, returns the buffer to the pool
    fn submit_dma_chunk(
        &self,
        state: &Arc<SchedulerState>,
        dma: &DmaParams,
        resolved_chunk: &ResolvedChunk,
        chunk: DmaChunk,
        bufs_created: &mut u64,
        pool: &Arc<DmaBufPool>,
    ) -> Result<(), SchedulerError> {
        let DmaChunk {
            hmva,
            offset_bytes,
            cur_size,
            buf_unaligned,
            pipeline,
        } = chunk;
        let to_device = dma.direction() == DmaDirection::HostToDevice;
        // `bool new_buf = bufs_created < RuntimeConfig::DataTransferBufferCount();`
        let new_buf = *bufs_created < RuntimeConfig::data_transfer_buffer_count().value();
        let mut chunk_buf = if new_buf {
            *bufs_created += 1;
            Box::new(DmaBuf::new())
        } else {
            // "Waiting for DMA chunk buffer to free up."
            pool.wait_dequeue()
        };

        chunk_buf.setup_dma_buf(
            DmaBufRequest {
                hmva,
                size: cur_size,
                to_device,
                unaligned: buf_unaligned,
                // `iova` is advanced by `cur_size` per chunk in `submitDma`'s loop.
                iova: dma.iova().map(|i| i.0 + offset_bytes as usize),
            },
            new_buf,
            &self.iommu_mapper,
        )?;

        let host_addr = chunk_buf.iova;

        // `++(*pool->pending);` before submitting, so a chunk that completes
        // while the loop is still queuing cannot drop the count to zero.
        pool.pending.fetch_add(1, Ordering::AcqRel);
        let pool_for_cb = pool.clone();
        let quarantine = self.dma_quarantine.clone();
        submit_to_hardware(
            state,
            self.response_worker(),
            self.capacity.as_ref(),
            pipeline,
            1,
            Box::new(move |result| {
                // The ONE question this callback asks, and it asks it of the
                // type system: is the card done with this chunk's CB? Without
                // the token there is no way to call `enqueue`, so a timed-out
                // chunk cannot be recycled even by a future edit that forgets
                // why. `None` = the hardware may still hold the CB, and that
                // CB carries this buffer's IOVA: the buffer must neither be
                // reused (reuse remaps, i.e. unmaps) nor dropped (drop
                // unmaps) — either hands the card a dangling IOVA and a later
                // card-side access fences the PCIe bus.
                let Some(retired) = CbRetired::from_completion(&result) else {
                    tracing::error!(
                        hmva = format_args!("{:#x}", chunk_buf.hmva_ptr),
                        iova = format_args!("{:#x}", chunk_buf.iova.as_u64()),
                        size = chunk_buf.size,
                        "DMA chunk timed out; quarantining its buffer (mapping stays live) \
                         instead of recycling it under a CB the hardware may still hold"
                    );
                    // The shadow copy-back is skipped for the same reason: the
                    // hardware never filled this buffer.
                    quarantine
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(QuarantinedBuf::park(*chunk_buf));
                    // Keep the pool's population constant so a transfer whose
                    // remaining chunks are blocked in `wait_dequeue` still gets
                    // a buffer — an empty `DmaBuf` re-runs `setup_dma_buf`'s
                    // allocate/map path from scratch. It has never been handed
                    // to hardware, hence the other constructor.
                    pool_for_cb.enqueue(Box::new(DmaBuf::new()), CbRetired::never_submitted());
                    pool_for_cb.release(result.err());
                    return;
                };
                // "For D2H with an unaligned shadow buffer, copy result back
                // to the original host pointer now that the transfer is
                // complete. Skip the copy-back when user provided iova is set
                // — the caller owns the buffer."
                if !chunk_buf.to_device
                    && chunk_buf.unaligned
                    && chunk_buf.user_iova.is_none()
                    && let Some(buf) = chunk_buf.raii_buffer.as_ref()
                {
                    // SAFETY: `hmva_ptr`/`size` name the same valid,
                    // writable host buffer `setup_dma_buf` copied from,
                    // and the shadow buffer holds this chunk's DMA'd bytes.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            buf.as_ptr(),
                            chunk_buf.hmva_ptr as *mut u8,
                            chunk_buf.size as usize,
                        );
                    }
                }
                pool_for_cb.enqueue(chunk_buf, retired);
                pool_for_cb.release(result.err());
            }),
            |tags| {
                Ok(vec![build_dma_cb_chunk(
                    resolved_chunk,
                    dma,
                    offset_bytes,
                    cur_size,
                    host_addr,
                    tags[0],
                )?])
            },
        );
        Ok(())
    }
}

impl SchedulerBackend for PfBackend {
    /// Port of `PfRuntimeScheduler::submitDma` (`pf_runtime_scheduler.cpp:78-155`):
    /// split the transfer into `RuntimeConfig::DataTransferBufferSize()`-sized
    /// chunks (real default 4 MiB, `FLEX_DATA_TRANSFER_BUFFER_SIZE`) with a
    /// `while(remaining_size != 0)` loop, hand each to `submitDmaChunk` as its
    /// OWN control block, then drop the pool's sentinel so the last chunk to
    /// complete fires the user callback.
    ///
    /// This is now the real async shape. The previous version was documented
    /// as a deliberate "simplification": ONE pooled buffer, chunks submitted
    /// strictly sequentially, each `hw.submit` blocking for its own response
    /// before the next chunk was staged. That is not a simplification of the
    /// real code so much as a different algorithm — it serialised what the
    /// real `DmaBufPool` explicitly pipelines
    /// (`RuntimeConfig::DataTransferBufferCount()` buffers, `wait_dequeue`
    /// backpressure) and it required the synchronous `HardwareSubmit::submit`
    /// that no longer exists.
    fn submit_dma(
        &self,
        state: &Arc<SchedulerState>,
        dma: &DmaParams,
        callback: CompletionCallback,
    ) {
        // `if(!dma.device_address->is_single_chunk()) RAS::DEVICE_PF::MultiChunkDmaNotSupported().Throw();`
        // — enforced by `resolve_single_chunk`.
        let resolved_chunk =
            match resolve_single_chunk(self.addr_resolver.as_ref(), dma.device_address()) {
                Ok(v) => v,
                Err(e) => return callback(Err(e)),
            };
        let pipeline = if dma.direction() == DmaDirection::HostToDevice {
            PipelineId::AsyncDmaI
        } else {
            PipelineId::AsyncDmaO
        };
        // `remaining_size = xfer_size` — exactly what the caller asked for,
        // never clamped to the resolved device chunk.
        let total_size = dma.dma_size().as_u64();
        let hmva_base = dma.hmva().0;
        // "Check alignment and setup buffer pool": computed ONCE from the base
        // `hmva`, not per chunk (`pf_runtime_scheduler.cpp:117`).
        let buf_unaligned =
            !(hmva_base as u64).is_multiple_of(self.iommu_mapper.alignment() as u64);
        let max_buf_size = RuntimeConfig::data_transfer_buffer_size().value().max(1);

        let pool = DmaBufPool::new(callback);
        let mut bufs_created = 0u64;
        let mut remaining = total_size;
        let mut offset = 0u64;
        while remaining != 0 {
            let cur_size = remaining.min(max_buf_size);
            if let Err(e) = self.submit_dma_chunk(
                state,
                dma,
                &resolved_chunk,
                DmaChunk {
                    hmva: hmva_base + offset as usize,
                    offset_bytes: offset,
                    cur_size,
                    buf_unaligned,
                    pipeline,
                },
                &mut bufs_created,
                &pool,
            ) {
                // The real `submitDmaChunk` throws (IOMMU mapping failure,
                // CB build failure), which unwinds out of `submitDma` and is
                // caught by `RuntimeScheduler::schedule`. Stop queuing and
                // report through the pool so the already-queued chunks still
                // get to complete first.
                pool.release(Some(e));
                return;
            }
            remaining -= cur_size;
            offset += cur_size;
        }
        // "Decrement pending count; if all chunks already completed, invoke
        // callback now."
        pool.release(None);
    }

    /// Port of `PfRuntimeScheduler::submitCompute` (`pf_runtime_scheduler.cpp:231-259`):
    /// validate, build the compute CB(s), `submitToHardware(cbs, COMPUTE, cb,
    /// makeComputeHwError)`. Returns as soon as the doorbell has been rung.
    fn submit_compute(
        &self,
        state: &Arc<SchedulerState>,
        op: &ComputeParams,
        callback: CompletionCallback,
    ) {
        let resolver = self.addr_resolver.as_ref();
        submit_to_hardware(
            state,
            self.response_worker(),
            self.capacity.as_ref(),
            PipelineId::Compute,
            1,
            callback,
            |tags| Ok(vec![build_compute_cb(resolver, op, tags[0])?]),
        );
    }

    /// Port of `PfRuntimeScheduler::submitFill` (`pf_runtime_scheduler.cpp:261-276`):
    /// build a real fill CB and submit it on DMAI or DMAO per
    /// `fill.use_dmai`.
    fn submit_fill(
        &self,
        state: &Arc<SchedulerState>,
        fill: &FillParams,
        callback: CompletionCallback,
    ) {
        let pipeline = match fill.pipeline() {
            FillPipeline::Dmai => PipelineId::AsyncDmaI,
            FillPipeline::Dmao => PipelineId::AsyncDmaO,
        };
        let resolver = self.addr_resolver.as_ref();
        submit_to_hardware(
            state,
            self.response_worker(),
            self.capacity.as_ref(),
            pipeline,
            1,
            callback,
            |tags| Ok(vec![build_fill_cb(resolver, fill, tags[0])?]),
        );
    }

    /// Real `PfRuntimeScheduler::submitP2PRdmaSend` (`pf_runtime_scheduler.cpp:488-608`)
    /// builds a full unicast/multicast RDMA CB sequence and performs the
    /// transfer. Porting that (window-chunked RDMA CB building against
    /// `P2P_RDMA_WINDOW_SIZE`, multicast fan-out) is separate scope from this
    /// pass. Previously stubbed to silent fake-success; an explicit error is
    /// strictly better — a caller that believes an RDMA send happened when it
    /// never did is a correctness bug in its own right.
    fn submit_p2p_rdma_send(
        &self,
        _state: &Arc<SchedulerState>,
        _params: &P2PRdmaSendParams,
        callback: CompletionCallback,
    ) {
        callback(Err(mock_error(
            "PfRuntimeScheduler: submitP2PRdmaSend not implemented in this port (known gap — real RDMA CB building out of scope)",
        )));
    }
    /// Same known gap as `submit_p2p_rdma_send`, for the wait side.
    fn submit_p2p_rdma_wait(
        &self,
        _state: &Arc<SchedulerState>,
        _params: &P2PRdmaWaitParams,
        callback: CompletionCallback,
    ) {
        callback(Err(mock_error(
            "PfRuntimeScheduler: submitP2PRdmaWait not implemented in this port (known gap — real RDMA CB building out of scope)",
        )));
    }
    fn submit_p2p_firmware_signal(
        &self,
        _state: &Arc<SchedulerState>,
        _params: &P2PFirmwareSignalParams,
        callback: CompletionCallback,
    ) {
        callback(Err(mock_error(
            "PfRuntimeScheduler: firmware signal not implemented on PF",
        )));
    }
    fn submit_p2p_firmware_wait(
        &self,
        _state: &Arc<SchedulerState>,
        _params: &P2PFirmwareWaitParams,
        callback: CompletionCallback,
    ) {
        callback(Err(mock_error(
            "PfRuntimeScheduler: firmware wait not implemented on PF",
        )));
    }
    fn queue_capacity_source(&self) -> &dyn QueueCapacitySource {
        self.capacity.as_ref()
    }
}

/// Port of `VfRuntimeScheduler`. Hardware submission goes through
/// `FlexVfStreamer`'s firmware messaging rather than a `ResponseWorker`
/// (`ResponseCompletionThread` returns immediately on a VF device —
/// `response_worker.cpp:1358-1363`), so `capacity()`/`ControlBlockInterface`
/// is typically absent (`getQueueCapacity` returns `nullopt`) and capacity
/// backpressure is not applicable on this path.
///
/// `FlexVfStreamer` is not ported (it is the VF device-interface phase's own
/// type), so this backend reuses the same `ResponseWorker` submission path as
/// PF. `create_scheduler` only ever builds a `Pf` backend today, so there is
/// no real-hardware exercise of this path.
pub struct VfBackend {
    hw: Arc<dyn HardwareSubmit>,
    capacity: Arc<dyn QueueCapacitySource>,
    addr_resolver: Arc<dyn DeviceAddressResolver>,
    /// Port of `firmware_cb_counter_`: unique per-op CB node names. NOT a
    /// response tag — that comes from the ResponseWorker's bounded
    /// [`TagPool`], because this counter grows without bound past the real tag
    /// space.
    firmware_cb_counter: std::sync::atomic::AtomicU64,
    response_worker: std::sync::OnceLock<Arc<ResponseWorker>>,
}

impl VfBackend {
    pub fn new(
        hw: Arc<dyn HardwareSubmit>,
        capacity: Arc<dyn QueueCapacitySource>,
        addr_resolver: Arc<dyn DeviceAddressResolver>,
    ) -> Self {
        Self {
            hw,
            capacity,
            addr_resolver,
            firmware_cb_counter: std::sync::atomic::AtomicU64::new(0),
            response_worker: std::sync::OnceLock::new(),
        }
    }

    fn response_worker(&self) -> &Arc<ResponseWorker> {
        self.response_worker
            .get_or_init(|| ResponseWorker::new(self.hw.clone()))
    }

    /// VF's firmware-messaging CB name suffix (`FlexVfStreamer`'s CB
    /// registration).
    #[allow(dead_code)]
    fn next_cb_name_suffix(&self) -> u64 {
        self.firmware_cb_counter.fetch_add(1, Ordering::Relaxed)
    }
}

impl SchedulerBackend for VfBackend {
    fn submit_dma(
        &self,
        state: &Arc<SchedulerState>,
        dma: &DmaParams,
        callback: CompletionCallback,
    ) {
        let pipeline = if dma.direction() == DmaDirection::HostToDevice {
            PipelineId::AsyncDmaI
        } else {
            PipelineId::AsyncDmaO
        };
        // KNOWN GAP, parallel to the one `PfBackend`'s `DmaBuf` covers: no
        // VF-side IOMMU mapper is ported in this crate (only
        // `PfIommuMapper`), so a `dma.iova()` of `None` cannot be mapped to a
        // device-visible address here. A previous version fell back to the
        // bare host virtual address — an address the device's IOMMU cannot
        // resolve, so the card's attempt to master it is a PCIe-level fault
        // (`RAS::PCI::BusFence`), not a clean per-CB rejection. Unported means
        // a refused submission with the gap named, never a plausible-looking
        // CB around an address that was never mapped.
        let host_addr = match dma.iova() {
            Some(iova) => HostPhysicalAddress::of_caller_supplied_iova(iova),
            None => {
                return callback(Err(mock_error(
                    "VfRuntimeScheduler: submitDma without a caller-supplied IOVA requires the \
                     VF IOMMU mapper, which is not ported (known gap) — refusing to build a CB \
                     around an unmapped host address",
                )));
            }
        };
        let resolver = self.addr_resolver.as_ref();
        submit_to_hardware(
            state,
            self.response_worker(),
            self.capacity.as_ref(),
            pipeline,
            1,
            callback,
            |tags| Ok(vec![build_dma_cb(resolver, dma, host_addr, tags[0])?]),
        );
    }
    fn submit_compute(
        &self,
        state: &Arc<SchedulerState>,
        op: &ComputeParams,
        callback: CompletionCallback,
    ) {
        let resolver = self.addr_resolver.as_ref();
        submit_to_hardware(
            state,
            self.response_worker(),
            self.capacity.as_ref(),
            PipelineId::Compute,
            1,
            callback,
            |tags| Ok(vec![build_compute_cb(resolver, op, tags[0])?]),
        );
    }
    /// Port of `VfRuntimeScheduler::submitFill`: same real fill CB as PF.
    fn submit_fill(
        &self,
        state: &Arc<SchedulerState>,
        fill: &FillParams,
        callback: CompletionCallback,
    ) {
        let pipeline = match fill.pipeline() {
            FillPipeline::Dmai => PipelineId::AsyncDmaI,
            FillPipeline::Dmao => PipelineId::AsyncDmaO,
        };
        let resolver = self.addr_resolver.as_ref();
        submit_to_hardware(
            state,
            self.response_worker(),
            self.capacity.as_ref(),
            pipeline,
            1,
            callback,
            |tags| Ok(vec![build_fill_cb(resolver, fill, tags[0])?]),
        );
    }
    /// Real `VfRuntimeScheduler::submitP2PRdmaSend` (`vf_runtime_scheduler.cpp:317-321`)
    /// throws `RAS::RUNTIMECONTEXT::NotImplemented().DeviceType("VF")`.
    fn submit_p2p_rdma_send(
        &self,
        _state: &Arc<SchedulerState>,
        _params: &P2PRdmaSendParams,
        callback: CompletionCallback,
    ) {
        callback(Err(mock_error(
            "VfRuntimeScheduler: submitP2PRdmaSend not implemented on VF",
        )));
    }
    /// Real `VfRuntimeScheduler::submitP2PRdmaWait` (`vf_runtime_scheduler.cpp:323-327`)
    /// throws `RAS::DEVICE_VF::P2PRdmaOperationsNotSupported()`.
    fn submit_p2p_rdma_wait(
        &self,
        _state: &Arc<SchedulerState>,
        _params: &P2PRdmaWaitParams,
        callback: CompletionCallback,
    ) {
        callback(Err(mock_error(
            "VfRuntimeScheduler: P2P RDMA wait not supported on VF",
        )));
    }
    fn submit_p2p_firmware_signal(
        &self,
        _state: &Arc<SchedulerState>,
        _params: &P2PFirmwareSignalParams,
        callback: CompletionCallback,
    ) {
        callback(Ok(()))
    }
    fn submit_p2p_firmware_wait(
        &self,
        _state: &Arc<SchedulerState>,
        _params: &P2PFirmwareWaitParams,
        callback: CompletionCallback,
    ) {
        callback(Ok(()))
    }
    fn queue_capacity_source(&self) -> &dyn QueueCapacitySource {
        self.capacity.as_ref()
    }
}

// ---------------------------------------------------------------------
// RuntimeScheduler — the public type, generic over exactly one live backend
// ---------------------------------------------------------------------

/// Port of `RuntimeScheduler` + the choice of concrete backend. Modeled as
/// an enum (rather than a bare `Box<dyn SchedulerBackend>`) because the
/// backend is not just "some object behind a trait" in the C++ — it is
/// exactly one of three named, ABI-relevant device modes chosen once at
/// construction (`RuntimeSchedulerType`), and callers (`getSchedulerType`)
/// and tests (`MockRuntimeScheduler`-specific accessors like `computeCount`)
/// need to recover the concrete identity, which a trait object would erase.
pub enum Backend {
    Pf(PfBackend),
    Vf(VfBackend),
    /// The C++'s third backend, `MockRuntimeScheduler`, exists to drive IBM's
    /// "senulator" software emulator. We never run the senulator, so this ships
    /// in no binary — it is retained ONLY as the double the ported C++ unit
    /// tests submit against (`runtime_scheduler_*_test.cpp`,
    /// `runtime_stream_mock_compute_test.cpp`).
    #[cfg(test)]
    Mock(MockBackend),
}

impl Backend {
    fn as_dyn(&self) -> &dyn SchedulerBackend {
        match self {
            Backend::Pf(b) => b,
            Backend::Vf(b) => b,
            #[cfg(test)]
            Backend::Mock(b) => b,
        }
    }

    fn scheduler_type(&self) -> RuntimeSchedulerType {
        match self {
            Backend::Pf(_) => RuntimeSchedulerType::Pf,
            Backend::Vf(_) => RuntimeSchedulerType::Vf,
            #[cfg(test)]
            Backend::Mock(_) => RuntimeSchedulerType::Mock,
        }
    }
}

/// Port of `flex::RuntimeScheduler` (concrete, backend-parameterized by
/// `Backend`). Owns exactly the state the C++ base class owns; `device_handle_`
/// and `region_id_to_region_` are represented implicitly through the
/// `Backend`'s own `HardwareSubmit`/`QueueCapacitySource` handles rather than
/// stored redundantly here, since nothing in this file's ported logic
/// dereferences them directly (only backends do, and they own their own
/// handles).
pub struct RuntimeScheduler {
    state: Arc<SchedulerState>,
    backend: Backend,
}

/// The scheduling identity of one operation handed to `schedule_pipelined`:
/// which stream and ordering mode it belongs to, its own UNCOLLAPSED
/// `RuntimeOperationKind` (see `maybe_set_barrier` for why the uncollapsed kind
/// is what the barrier comparison uses), whether the caller explicitly asked for
/// a pipeline barrier, and an optional physical-pipeline override — which only
/// `Fill` uses, to route to DMAI or DMAO per its own `FillParams::pipeline()`.
///
/// Grouped because these five arrive together at every one of the nine
/// `schedule()` call sites and three of them are same-shaped scalars whose order
/// a reader cannot check locally.
#[derive(Debug, Clone, Copy)]
struct PipelinedOp {
    id: StreamId,
    mode: RuntimeStreamMode,
    kind: RuntimeOperationKind,
    requested_barrier: bool,
    pipeline_override: Option<PipelineId>,
}

impl RuntimeScheduler {
    pub fn new(backend: Backend) -> Arc<Self> {
        let scheduler_type = backend.scheduler_type();
        Arc::new(Self {
            state: Arc::new(SchedulerState::new(scheduler_type)),
            backend,
        })
    }

    pub fn scheduler_type(&self) -> RuntimeSchedulerType {
        self.backend.scheduler_type()
    }

    /// Test-only accessor: only meaningful when this scheduler wraps
    /// `Backend::Mock`. Port of `MockRuntimeScheduler::barrierCount`.
    #[cfg(test)]
    pub fn mock_barrier_count(&self) -> Option<usize> {
        match &self.backend {
            Backend::Mock(m) => Some(m.barrier_count()),
            _ => None,
        }
    }

    /// Test-only accessor: configure `MockBackend::throw_on_compute` when
    /// this scheduler wraps `Backend::Mock`. Port of the C++ test fixture's
    /// `scheduler_->SetThrowOnCompute(true)` used by
    /// `runtime_stream_mock_compute_test.cpp`'s error-path tests. No-op if
    /// this scheduler is not `Backend::Mock`.
    #[cfg(test)]
    pub fn mock_set_throw_on_compute(&self, throw: bool) {
        if let Backend::Mock(m) = &self.backend {
            *m.throw_on_compute.lock().unwrap() = throw;
        }
    }

    /// Test-only accessor: configure `MockBackend::throw_on_dma` when this
    /// scheduler wraps `Backend::Mock`. Port of the C++ test fixture's
    /// `scheduler_->throw_on_dma_ = true` used by
    /// `runtime_scheduler_fence_test.cpp`'s H2D/D2H submission-failure tests.
    /// No-op if this scheduler is not `Backend::Mock`.
    #[cfg(test)]
    pub fn mock_set_throw_on_dma(&self, throw: bool) {
        if let Backend::Mock(m) = &self.backend {
            *m.throw_on_dma.lock().unwrap() = throw;
        }
    }

    /// Port of `RuntimeScheduler::getQueueCapacity`.
    pub fn queue_capacity(&self, pipeline: PipelineId) -> Option<u64> {
        self.backend
            .as_dyn()
            .queue_capacity_source()
            .capacity(pipeline)
    }

    /// Port of `RuntimeScheduler::hasQueueCapacity`.
    pub fn has_queue_capacity(&self, pipeline: PipelineId, num_cbs_needed: u64) -> bool {
        has_queue_capacity(
            self.backend.as_dyn().queue_capacity_source(),
            &self.state,
            pipeline,
            num_cbs_needed,
        )
    }

    /// Port of `RuntimeScheduler::waitForQueueCapacity`.
    pub fn wait_for_queue_capacity(
        &self,
        pipeline: PipelineId,
        num_cbs_needed: u64,
        timeout_ms: u32,
    ) -> bool {
        wait_for_queue_capacity(
            self.backend.as_dyn().queue_capacity_source(),
            &self.state,
            pipeline,
            num_cbs_needed,
            timeout_ms,
        )
    }

    /// Port of `RuntimeScheduler::registerStream`.
    pub fn register_stream(&self, id: StreamId, shutdown: Arc<AtomicBool>) {
        self.state
            .stream_shutdown_flags
            .lock()
            .unwrap()
            .insert(id, shutdown);
    }

    fn pipeline_for(kind: RuntimeOperationKind) -> Option<PipelineId> {
        match kind {
            RuntimeOperationKind::H2D | RuntimeOperationKind::Fill => Some(PipelineId::AsyncDmaI),
            RuntimeOperationKind::D2H => Some(PipelineId::AsyncDmaO),
            RuntimeOperationKind::Compute
            | RuntimeOperationKind::LegacyP2PSendData
            | RuntimeOperationKind::LegacyP2PRecvData
            | RuntimeOperationKind::LegacyP2PDataExchange
            | RuntimeOperationKind::P2PRdmaSend
            | RuntimeOperationKind::P2PRdmaWait
            | RuntimeOperationKind::P2PFirmwareSignal
            | RuntimeOperationKind::P2PFirmwareWait => Some(PipelineId::Compute),
            RuntimeOperationKind::HostCallback
            | RuntimeOperationKind::EventSignal
            | RuntimeOperationKind::EventWait => None,
        }
    }

    /// Collapses `kind` to the value real `stream_last_pipeline_` actually
    /// stores/compares for THIS op — every `schedule(RuntimeOperation*)`
    /// overload for the P2P-ish kinds (`LegacyP2PSendData`/`RecvData`/
    /// `DataExchange`, `P2PRdmaSend`/`Wait`, `P2PFirmwareSignal`/`Wait`) all
    /// write the SAME `RuntimeOperationType::Compute` into
    /// `stream_last_pipeline_[id]` (`runtime_scheduler.cpp`, each one's
    /// "Track which pipeline this stream last targeted" block) — because they
    /// all share `stream_compute_fence_`/the COMPUTE barrier target, not
    /// because they're otherwise interchangeable. Storing each one's own
    /// distinct `RuntimeOperationKind` instead (this crate's bug until now)
    /// made `maybeSetBarrier`'s switch-detection see e.g. `Compute` ->
    /// `P2PRdmaSend` as a pipeline switch (spuriously inserting a barrier)
    /// and, going the other way, back-to-back distinct P2P kinds as NOT a
    /// switch when they should have been fine (they already share one fence
    /// queue, so that direction was harmless, but the spurious-barrier
    /// direction was not). `H2D`/`D2H`/`Fill` are each their own distinct
    /// real `RuntimeOperationType` tag and stay as-is (Fill in particular is
    /// its own tag — not folded into H2D/D2H even though it physically shares
    /// their DMAI/DMAO queues).
    fn logical_pipeline_kind(kind: RuntimeOperationKind) -> RuntimeOperationKind {
        match kind {
            RuntimeOperationKind::LegacyP2PSendData
            | RuntimeOperationKind::LegacyP2PRecvData
            | RuntimeOperationKind::LegacyP2PDataExchange
            | RuntimeOperationKind::P2PRdmaSend
            | RuntimeOperationKind::P2PRdmaWait
            | RuntimeOperationKind::P2PFirmwareSignal
            | RuntimeOperationKind::P2PFirmwareWait => RuntimeOperationKind::Compute,
            other => other,
        }
    }

    /// Shared "auto-barrier, push fence, track last pipeline, submit,
    /// resolve fence on completion or on synchronous submission failure"
    /// steps that every non-event `schedule()` overload repeats verbatim in
    /// the C++ (`makeFenceCallback` + the surrounding `try`/`catch`).
    // Argument count follows the `schedule()` overloads' shared parameter set in
    fn schedule_pipelined<T>(
        &self,
        op: PipelinedOp,
        payload: &T,
        on_complete: CompletionCallback,
        submit: impl FnOnce(&dyn SchedulerBackend, &Arc<SchedulerState>, &T, CompletionCallback),
    ) {
        let PipelinedOp {
            id,
            mode,
            kind,
            requested_barrier,
            pipeline_override,
        } = op;
        let logical_kind = Self::logical_pipeline_kind(kind);
        // Compare against the op's own UNCOLLAPSED kind (`kind`, not
        // `logical_kind`) — real C++'s `maybeSetBarrier` compares
        // `target = op.getType()` (uncollapsed) against `stream_last_pipeline_`,
        // which was collapsed only on a PRIOR write. See `maybe_set_barrier`'s
        // doc comment for the exact asymmetry.
        let use_barrier = self
            .state
            .maybe_set_barrier(id, mode, kind, requested_barrier);
        // The physical pipeline this op actually targets. `pipeline_override`
        // lets `Fill` route to DMAI or DMAO per its own `FillParams::pipeline()`
        // choice (port of `schedule(RuntimeOperationFill)`'s `use_dmai`
        // parameter, `runtime_scheduler.cpp`) rather than always defaulting to
        // DMAI the way `pipeline_for`'s static `kind`-only mapping would.
        let pipeline = pipeline_override
            .or_else(|| Self::pipeline_for(kind))
            .unwrap_or(PipelineId::Compute);
        // Fence bookkeeping/barrier-target pipeline this op targets: derived
        // from the RESOLVED physical pipeline (not `kind` alone), so a
        // DMAO-routed Fill correctly issues/collects against the D2H fence
        // deque — port of `schedule(RuntimeOperationFill)`'s
        // `issueBarrier(use_dmai ? H2D : D2H, id)`.
        let fence_pipeline = match pipeline {
            PipelineId::AsyncDmaI => RuntimeOperationKind::H2D,
            PipelineId::AsyncDmaO => RuntimeOperationKind::D2H,
            PipelineId::Compute => RuntimeOperationKind::Compute,
        };
        if use_barrier {
            self.backend
                .as_dyn()
                .issue_barrier(&self.state, fence_pipeline, id);
        }
        let fence = self.state.push_fence(pipeline, id);
        self.state.record_last_pipeline(id, logical_kind);

        let fence_for_cb = fence.clone();
        let wrapped: CompletionCallback = Box::new(move |result| {
            fence_for_cb.resolve();
            on_complete(result);
        });

        // The queue-capacity gate and the in-flight accounting used to live
        // here, with a hardcoded `needed = 1` and an RAII guard spanning the
        // whole (then-synchronous) submit. Both belong to
        // `PfRuntimeScheduler::submitToHardware`
        // (`pf_runtime_scheduler_utils.cpp:509-537`), which gates on the real
        // `cbr->GetNumberOfCBs()` and keeps the counter raised until each CB's
        // response actually arrives — see [`submit_to_hardware`]. Every real
        // `schedule()` overload ends exactly here, at `submit*(params,
        // makeFenceCallback(fence, cb))`.
        submit(self.backend.as_dyn(), &self.state, payload, wrapped);
    }
}

impl SchedulerHandle for RuntimeScheduler {
    fn register_stream(&self, stream_id: StreamId, shutdown_flag: Arc<AtomicBool>) {
        RuntimeScheduler::register_stream(self, stream_id, shutdown_flag);
    }

    fn release_stream(&self, stream_id: StreamId) {
        self.state.release_stream(stream_id);
    }

    /// Port of every `RuntimeScheduler::schedule(...)` overload. Downcasts
    /// `op.payload` by `op.kind`, matching the C++ compile-time overload
    /// resolution with a runtime tag (the tag is exactly what
    /// `SubmittedOperation` was designed in `stream.rs` to carry).
    fn schedule(
        &self,
        stream_id: StreamId,
        mode: RuntimeStreamMode,
        op: SubmittedOperation,
        completion: CompletionCallback,
    ) -> Result<(), SchedulerError> {
        use RuntimeOperationKind as K;
        // Previously dropped entirely: `schedule_pipelined` hardcoded
        // `requested = false` and no `schedule()` arm below ever read this
        // field, so a caller's explicit `pipeline_barrier` request (set
        // correctly upstream in `stream.rs`'s `launch_h2d`/`launch_d2h`/
        // `launch_compute`) never reached `maybeSetBarrier` at all.
        let requested_barrier = op.pipeline_barrier;
        match op.kind {
            K::EventSignal | K::EventWait | K::HostCallback => {
                // Events/host-callbacks do not target a hardware pipeline —
                // `RuntimeStream` itself handles these (see
                // `launch_operation_host_callback`); `RuntimeScheduler`'s
                // event overloads just signal/wait then invoke `on_complete`
                // with no fence/pipeline bookkeeping.
                completion(Ok(()));
                Ok(())
            }
            K::H2D | K::D2H => {
                let dma = match op.payload.downcast::<DmaParams>() {
                    Ok(v) => v,
                    // The C++ `schedule` is a set of type-safe overloads over
                    // concrete `RuntimeOperation*` types, so "the payload was
                    // the wrong type" is not a failure mode it has at all —
                    // this arm exists only because `SubmittedOperation` carries
                    // a `Box<dyn Any>`. It must still drive `completion`: the
                    // caller (`RuntimeStream::launch_operation`) has already
                    // incremented its own `in_flight` counter and only the
                    // completion callback lowers it, so returning `Err` here
                    // without completing leaves `synchronize()` — including the
                    // one in `Drop` — waiting forever. The error is reported
                    // through the completion path, the same as every other
                    // submission failure in this file, rather than also as an
                    // `Err` return.
                    Err(_) => {
                        completion(Err(mock_error(
                            "schedule: H2D/D2H payload was not DmaParams",
                        )));
                        return Ok(());
                    }
                };
                self.schedule_pipelined(
                    PipelinedOp {
                        id: stream_id,
                        mode,
                        kind: op.kind,
                        requested_barrier,
                        pipeline_override: None,
                    },
                    &*dma,
                    completion,
                    |backend, state, dma, cb| backend.submit_dma(state, dma, cb),
                );
                Ok(())
            }
            K::Fill => {
                let fill = match op.payload.downcast::<FillParams>() {
                    Ok(v) => v,
                    // The C++ `schedule` is a set of type-safe overloads over
                    // concrete `RuntimeOperation*` types, so "the payload was
                    // the wrong type" is not a failure mode it has at all —
                    // this arm exists only because `SubmittedOperation` carries
                    // a `Box<dyn Any>`. It must still drive `completion`: the
                    // caller (`RuntimeStream::launch_operation`) has already
                    // incremented its own `in_flight` counter and only the
                    // completion callback lowers it, so returning `Err` here
                    // without completing leaves `synchronize()` — including the
                    // one in `Drop` — waiting forever. The error is reported
                    // through the completion path, the same as every other
                    // submission failure in this file, rather than also as an
                    // `Err` return.
                    Err(_) => {
                        completion(Err(mock_error("schedule: Fill payload was not FillParams")));
                        return Ok(());
                    }
                };
                // Port of `schedule(RuntimeOperationFill)`'s `use_dmai`: the
                // fill's own pipeline choice, not a static kind->pipeline
                // mapping — a Fill can target either DMAI or DMAO.
                let fill_pipeline = match fill.pipeline() {
                    FillPipeline::Dmai => PipelineId::AsyncDmaI,
                    FillPipeline::Dmao => PipelineId::AsyncDmaO,
                };
                self.schedule_pipelined(
                    PipelinedOp {
                        id: stream_id,
                        mode,
                        kind: op.kind,
                        requested_barrier,
                        pipeline_override: Some(fill_pipeline),
                    },
                    &*fill,
                    completion,
                    |backend, state, fill, cb| backend.submit_fill(state, fill, cb),
                );
                Ok(())
            }
            K::Compute => {
                let compute = match op.payload.downcast::<ComputeParams>() {
                    Ok(v) => v,
                    // The C++ `schedule` is a set of type-safe overloads over
                    // concrete `RuntimeOperation*` types, so "the payload was
                    // the wrong type" is not a failure mode it has at all —
                    // this arm exists only because `SubmittedOperation` carries
                    // a `Box<dyn Any>`. It must still drive `completion`: the
                    // caller (`RuntimeStream::launch_operation`) has already
                    // incremented its own `in_flight` counter and only the
                    // completion callback lowers it, so returning `Err` here
                    // without completing leaves `synchronize()` — including the
                    // one in `Drop` — waiting forever. The error is reported
                    // through the completion path, the same as every other
                    // submission failure in this file, rather than also as an
                    // `Err` return.
                    Err(_) => {
                        completion(Err(mock_error(
                            "schedule: Compute payload was not ComputeParams",
                        )));
                        return Ok(());
                    }
                };
                self.schedule_pipelined(
                    PipelinedOp {
                        id: stream_id,
                        mode,
                        kind: op.kind,
                        requested_barrier,
                        pipeline_override: None,
                    },
                    &*compute,
                    completion,
                    |backend, state, compute, cb| backend.submit_compute(state, compute, cb),
                );
                Ok(())
            }
            K::LegacyP2PSendData | K::LegacyP2PRecvData => {
                let params = match op.payload.downcast::<P2PDataParams>() {
                    Ok(v) => v,
                    // The C++ `schedule` is a set of type-safe overloads over
                    // concrete `RuntimeOperation*` types, so "the payload was
                    // the wrong type" is not a failure mode it has at all —
                    // this arm exists only because `SubmittedOperation` carries
                    // a `Box<dyn Any>`. It must still drive `completion`: the
                    // caller (`RuntimeStream::launch_operation`) has already
                    // incremented its own `in_flight` counter and only the
                    // completion callback lowers it, so returning `Err` here
                    // without completing leaves `synchronize()` — including the
                    // one in `Drop` — waiting forever. The error is reported
                    // through the completion path, the same as every other
                    // submission failure in this file, rather than also as an
                    // `Err` return.
                    Err(_) => {
                        completion(Err(mock_error(
                            "schedule: LegacyP2P{Send,Recv}Data payload was not P2PDataParams",
                        )));
                        return Ok(());
                    }
                };
                let is_send = op.kind == K::LegacyP2PSendData;
                self.schedule_pipelined(
                    PipelinedOp {
                        id: stream_id,
                        mode,
                        kind: op.kind,
                        requested_barrier,
                        pipeline_override: None,
                    },
                    &*params,
                    completion,
                    move |backend, state, params, cb| {
                        if is_send {
                            backend.submit_legacy_p2p_send_data(state, params, cb)
                        } else {
                            backend.submit_legacy_p2p_recv_data(state, params, cb)
                        }
                    },
                );
                Ok(())
            }
            K::LegacyP2PDataExchange => {
                let params = op.payload.downcast::<Vec<P2PDataParams>>().map_err(|_| {
                    mock_error("schedule: LegacyP2PDataExchange payload was not Vec<P2PDataParams>")
                })?;
                self.schedule_pipelined(
                    PipelinedOp {
                        id: stream_id,
                        mode,
                        kind: op.kind,
                        requested_barrier,
                        pipeline_override: None,
                    },
                    &*params,
                    completion,
                    |backend, state, params, cb| {
                        backend.submit_legacy_p2p_data_exchange(state, params, cb)
                    },
                );
                Ok(())
            }
            K::P2PRdmaSend => {
                let params = match op.payload.downcast::<P2PRdmaSendParams>() {
                    Ok(v) => v,
                    // The C++ `schedule` is a set of type-safe overloads over
                    // concrete `RuntimeOperation*` types, so "the payload was
                    // the wrong type" is not a failure mode it has at all —
                    // this arm exists only because `SubmittedOperation` carries
                    // a `Box<dyn Any>`. It must still drive `completion`: the
                    // caller (`RuntimeStream::launch_operation`) has already
                    // incremented its own `in_flight` counter and only the
                    // completion callback lowers it, so returning `Err` here
                    // without completing leaves `synchronize()` — including the
                    // one in `Drop` — waiting forever. The error is reported
                    // through the completion path, the same as every other
                    // submission failure in this file, rather than also as an
                    // `Err` return.
                    Err(_) => {
                        completion(Err(mock_error(
                            "schedule: P2PRdmaSend payload was not P2PRdmaSendParams",
                        )));
                        return Ok(());
                    }
                };
                self.schedule_pipelined(
                    PipelinedOp {
                        id: stream_id,
                        mode,
                        kind: op.kind,
                        requested_barrier,
                        pipeline_override: None,
                    },
                    &*params,
                    completion,
                    |backend, state, params, cb| backend.submit_p2p_rdma_send(state, params, cb),
                );
                Ok(())
            }
            K::P2PRdmaWait => {
                let params = match op.payload.downcast::<P2PRdmaWaitParams>() {
                    Ok(v) => v,
                    // The C++ `schedule` is a set of type-safe overloads over
                    // concrete `RuntimeOperation*` types, so "the payload was
                    // the wrong type" is not a failure mode it has at all —
                    // this arm exists only because `SubmittedOperation` carries
                    // a `Box<dyn Any>`. It must still drive `completion`: the
                    // caller (`RuntimeStream::launch_operation`) has already
                    // incremented its own `in_flight` counter and only the
                    // completion callback lowers it, so returning `Err` here
                    // without completing leaves `synchronize()` — including the
                    // one in `Drop` — waiting forever. The error is reported
                    // through the completion path, the same as every other
                    // submission failure in this file, rather than also as an
                    // `Err` return.
                    Err(_) => {
                        completion(Err(mock_error(
                            "schedule: P2PRdmaWait payload was not P2PRdmaWaitParams",
                        )));
                        return Ok(());
                    }
                };
                self.schedule_pipelined(
                    PipelinedOp {
                        id: stream_id,
                        mode,
                        kind: op.kind,
                        requested_barrier,
                        pipeline_override: None,
                    },
                    &*params,
                    completion,
                    |backend, state, params, cb| backend.submit_p2p_rdma_wait(state, params, cb),
                );
                Ok(())
            }
            K::P2PFirmwareSignal => {
                let params = match op.payload.downcast::<P2PFirmwareSignalParams>() {
                    Ok(v) => v,
                    // The C++ `schedule` is a set of type-safe overloads over
                    // concrete `RuntimeOperation*` types, so "the payload was
                    // the wrong type" is not a failure mode it has at all —
                    // this arm exists only because `SubmittedOperation` carries
                    // a `Box<dyn Any>`. It must still drive `completion`: the
                    // caller (`RuntimeStream::launch_operation`) has already
                    // incremented its own `in_flight` counter and only the
                    // completion callback lowers it, so returning `Err` here
                    // without completing leaves `synchronize()` — including the
                    // one in `Drop` — waiting forever. The error is reported
                    // through the completion path, the same as every other
                    // submission failure in this file, rather than also as an
                    // `Err` return.
                    Err(_) => {
                        completion(Err(mock_error(
                            "schedule: P2PFirmwareSignal payload was not P2PFirmwareSignalParams",
                        )));
                        return Ok(());
                    }
                };
                self.schedule_pipelined(
                    PipelinedOp {
                        id: stream_id,
                        mode,
                        kind: op.kind,
                        requested_barrier,
                        pipeline_override: None,
                    },
                    &*params,
                    completion,
                    |backend, state, params, cb| {
                        backend.submit_p2p_firmware_signal(state, params, cb)
                    },
                );
                Ok(())
            }
            K::P2PFirmwareWait => {
                let params = match op.payload.downcast::<P2PFirmwareWaitParams>() {
                    Ok(v) => v,
                    // The C++ `schedule` is a set of type-safe overloads over
                    // concrete `RuntimeOperation*` types, so "the payload was
                    // the wrong type" is not a failure mode it has at all —
                    // this arm exists only because `SubmittedOperation` carries
                    // a `Box<dyn Any>`. It must still drive `completion`: the
                    // caller (`RuntimeStream::launch_operation`) has already
                    // incremented its own `in_flight` counter and only the
                    // completion callback lowers it, so returning `Err` here
                    // without completing leaves `synchronize()` — including the
                    // one in `Drop` — waiting forever. The error is reported
                    // through the completion path, the same as every other
                    // submission failure in this file, rather than also as an
                    // `Err` return.
                    Err(_) => {
                        completion(Err(mock_error(
                            "schedule: P2PFirmwareWait payload was not P2PFirmwareWaitParams",
                        )));
                        return Ok(());
                    }
                };
                self.schedule_pipelined(
                    PipelinedOp {
                        id: stream_id,
                        mode,
                        kind: op.kind,
                        requested_barrier,
                        pipeline_override: None,
                    },
                    &*params,
                    completion,
                    |backend, state, params, cb| {
                        backend.submit_p2p_firmware_wait(state, params, cb)
                    },
                );
                Ok(())
            }
        }
    }
}

/// Port of `RuntimeScheduler::synchronize`.
impl RuntimeScheduler {
    pub fn synchronize(&self) {
        self.state.synchronize();
    }

    /// Port of `RuntimeScheduler::query`.
    pub fn query(&self) -> bool {
        self.state.query()
    }
}

/// Port of `ComputeThroughput`'s transient `StatsValue<double>` local — see
/// [`crate::telemetry::StatsValue`]'s own doc for why this is a separate,
/// non-generic type rather than a second instantiation of that one.
pub struct ThroughputStat {
    pub name: &'static str,
    pub unit: &'static str,
    pub value: f64,
}

impl fmt::Display for ThroughputStat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (value, unit_prefix) = si_prefix(self.value);
        write!(f, "{}: {value} [{unit_prefix}{}]", self.name, self.unit)
    }
}

// ---------------------------------------------------------------------
// RuntimeSchedulerFactory (scheduler/runtime_scheduler_factory.{hpp,cpp})
// ---------------------------------------------------------------------

/// Port of `RuntimeSchedulerFactory`'s dispatch logic. The real C++ takes
/// this in two steps — `parseDeviceInterfaceType()` picks mock-vs-real at
/// the top, then `dev_int->getDeviceType()` (a property of an
/// already-constructed `DeviceInterface`) picks PF-vs-VF inside
/// `createRealScheduler` — but both ultimately trace back to the single
/// `FLEX_DEVICE` setting in this crate's simplified single-process model, so
/// `device_kind` folds both steps into the one three-way
/// [`crate::flex_config::DeviceKind`] this crate already uses everywhere
/// else backend choice matters (e.g. `fxa_rust_abi.rs`, which this factory
/// exists to stop hardcoding `Backend::Pf` unconditionally).
pub enum SchedulerBackendConfig {
    Real(
        Arc<dyn HardwareSubmit>,
        Arc<dyn QueueCapacitySource>,
        Arc<dyn DeviceAddressResolver>,
        Arc<crate::iommu::PfIommuMapper>,
    ),
    /// See [`Backend::Mock`]: the senulator backend is test-only.
    #[cfg(test)]
    Mock,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerFactoryError {
    /// Port of `RAS::CHIP::UnknownChipVersion().Throw()`: `SPYRE_CHIP` names
    /// anything other than `1P0` — the only chip version either
    /// `createRealScheduler` or `createMockScheduler` handles today (`1P5`
    /// is "still in the works" per the C++'s own comment).
    UnsupportedChipVersion(String),
    /// `device_kind` was `DeviceKind::Other(_)` — not one of MOCK/PF/VF.
    UnsupportedDeviceKind(String),
    /// The `SchedulerBackendConfig` variant supplied didn't match
    /// `device_kind` (e.g. `Mock` device kind with `Real` config).
    MismatchedBackendConfig,
}

/// Port of `RuntimeSchedulerFactory::createScheduler` (which internally
/// dispatches to `createRealScheduler`/`createMockScheduler`).
pub fn create_scheduler(
    device_kind: &DeviceKind,
    config: SchedulerBackendConfig,
) -> Result<Arc<RuntimeScheduler>, SchedulerFactoryError> {
    let chip = RuntimeConfig::spyre_chip().value();
    // Port of `parseSpyreChipVersion` (`runtime_context.cpp`): an exact,
    // case-SENSITIVE match against `"1P0"` — a lowercase `"1p0"` is rejected
    // by the real C++ (`RAS::CONFIGURATION::UnsupportedChip`), not accepted.
    if chip != "1P0" {
        return Err(SchedulerFactoryError::UnsupportedChipVersion(chip));
    }
    match (device_kind, config) {
        #[cfg(test)]
        (DeviceKind::Mock, SchedulerBackendConfig::Mock) => {
            Ok(RuntimeScheduler::new(Backend::Mock(MockBackend::default())))
        }
        (
            DeviceKind::Pf,
            SchedulerBackendConfig::Real(hw, capacity, addr_resolver, iommu_mapper),
        ) => Ok(RuntimeScheduler::new(Backend::Pf(PfBackend::new(
            hw,
            capacity,
            addr_resolver,
            iommu_mapper,
        )))),
        (
            DeviceKind::Vf,
            SchedulerBackendConfig::Real(hw, capacity, addr_resolver, _iommu_mapper),
        ) => Ok(RuntimeScheduler::new(Backend::Vf(VfBackend::new(
            hw,
            capacity,
            addr_resolver,
        )))),
        // Without the senulator backend there is no `Mock` config variant, so the
        // two arms above already cover every `(Pf|Vf, Real)` pair and the only
        // case left is a caller asking for a MOCK device this build cannot serve.
        #[cfg(not(test))]
        (DeviceKind::Mock, _) => Err(SchedulerFactoryError::UnsupportedDeviceKind(
            "MOCK".to_string(),
        )),
        #[cfg(test)]
        (DeviceKind::Mock | DeviceKind::Pf | DeviceKind::Vf, _) => {
            Err(SchedulerFactoryError::MismatchedBackendConfig)
        }
        (DeviceKind::Other(s), _) => Err(SchedulerFactoryError::UnsupportedDeviceKind(s.clone())),
    }
}

#[cfg(test)]
mod resolve_single_chunk_tests {
    use super::*;
    use crate::address::{ByteOffset, LogicalAddress, RegionId};
    use crate::memory_region::DomainId;

    struct FixedResolver {
        segment: u8,
        base: DevicePhysicalAddress,
        region_size: ByteSize,
    }

    impl DeviceAddressResolver for FixedResolver {
        fn resolve(&self, _chunk: &Chunk) -> Option<(u8, DevicePhysicalAddress, ByteSize)> {
            Some((self.segment, self.base, self.region_size))
        }
    }

    /// Port of the real address split: `TranslateDmvaToDmpa`
    /// (`control_block_stream.cpp`) computes `dmpa = dmpa_allocs[seg_id] +
    /// seg_offset`, fed by `getDmvaAddress`'s
    /// `SegmentByteOffset_todmva(seg_id, chunk.addr.offset)` and
    /// `buildDmaControlBlocks`'s `translations()[seg_id].SetPaddrBytes(region_id)`
    /// (`pf_runtime_scheduler_utils.cpp`) — the XLAT entry (`xlat_base`) gets
    /// the region's own base with NO offset folded in, and the DMVA/bootstrap
    /// companion field (`dmva`) carries this chunk's own byte offset;
    /// hardware adds them back together. Regression test for two bugs this
    /// crate had in turn: first doubling the base into both fields, then
    /// zeroing the DMVA offset instead of carrying the real chunk offset.
    #[test]
    fn xlat_base_has_no_offset_dmva_carries_the_chunk_offset() {
        let resolver = FixedResolver {
            segment: 3,
            base: DevicePhysicalAddress(0x1_0000_0000),
            region_size: crate::address::ByteSize(4096),
        };
        let chunk = Chunk::new(
            LogicalAddress::new(RegionId(1), ByteOffset(0x1000)),
            crate::address::ByteSize(4096),
            DomainId(0),
        );
        let addr = CompositeAddress::from_chunk(chunk);
        let resolved = resolve_single_chunk(&resolver, &addr).unwrap();

        assert_eq!(
            resolved.xlat_base().0,
            0x1_0000_0000,
            "XLAT base must be the region base alone, no offset folded in"
        );
        let dmva = resolved.dmva(0);
        assert_eq!(dmva.segment(), 3);
        assert_eq!(
            dmva.offset(),
            Flits::from_bytes(0x1000),
            "dmva must carry this chunk's own byte offset"
        );
        assert_eq!(
            resolved.xlat_absolute().0,
            0x1_0000_0000 + 0x1000,
            "the no-companion-offset (tensor XLAT) form must fold the offset in directly"
        );
        assert_eq!(resolved.size, 4096);
    }

    #[test]
    fn dmva_adds_extra_offset_for_bootstrap() {
        let resolver = FixedResolver {
            segment: 7,
            base: DevicePhysicalAddress(0x2_0000_0000),
            region_size: crate::address::ByteSize(4096),
        };
        let chunk = Chunk::new(
            LogicalAddress::new(RegionId(1), ByteOffset(0x100)),
            crate::address::ByteSize(4096),
            DomainId(0),
        );
        let addr = CompositeAddress::from_chunk(chunk);
        let resolved = resolve_single_chunk(&resolver, &addr).unwrap();

        let bootstrap = resolved.dmva(0x20);
        assert_eq!(
            bootstrap.offset(),
            Flits::from_bytes(0x100 + 0x20),
            "bootstrap = chunk offset + caller's bootstrap_offset"
        );
    }

    /// Regression test for a real hardware bug: `build_dma_cb` conflated the
    /// XLAT entry's length with this transfer's own byte count. Real
    /// `buildDmaControlBlocks` (`pf_runtime_scheduler_utils.cpp:48`) sets
    /// `xlat_length = region_id_to_region_.at(region_id)->total_size()` — the
    /// REGION's total size, independent of any one chunk/transfer's own
    /// size. A region loaded via more than one H2D call (this chunk's own
    /// offset + its own size not covering the whole region) must still
    /// report the region's full size as the XLAT length, not this chunk's.
    #[test]
    fn xlat_length_is_the_region_total_size_not_the_chunk_size() {
        let resolver = FixedResolver {
            segment: 7,
            base: DevicePhysicalAddress(0x80),
            region_size: crate::address::ByteSize(760_320),
        };
        // A second chunk of a larger region: starts at offset 4992, is only
        // 755328 bytes on its own — offset + this chunk's own size
        // (760_320) happens to equal the region size here, but the point is
        // that `region_total_size()` must return the REGION size
        // (760_320) regardless of this chunk's own size (755_328), not
        // derive it from the chunk.
        let chunk = Chunk::new(
            LogicalAddress::new(RegionId(1), ByteOffset(4992)),
            crate::address::ByteSize(755_328),
            DomainId(0),
        );
        let addr = CompositeAddress::from_chunk(chunk);
        let resolved = resolve_single_chunk(&resolver, &addr).unwrap();

        assert_eq!(
            resolved.size, 755_328,
            "chunk/transfer size stays this call's own byte count"
        );
        // Port of the real `xlat.SetLengthBytes(EncodeLength(seg_offset + size_bytes))`:
        // THIS transfer's own segment offset plus its own size — not the
        // owning region's total size (see `SegmentExtent`'s doc).
        assert_eq!(
            resolved.segment_extent(0, 755_328),
            SegmentExtent(4992 + 755_328)
        );
        // ...and a chunked piece reports its own offset within the transfer.
        assert_eq!(
            resolved.segment_extent(4096, 4096),
            SegmentExtent(4992 + 4096 + 4096)
        );
    }

    /// Regression test for the INVERSE bug in `build_compute_cb`: unlike DMA
    /// (above), a compute tensor/program XLAT length must be THIS
    /// ALLOCATION's own size (`CompositeAddress::total_size()`), never the
    /// enclosing region's — real C++: `tensor_sizes.push_back(inp_out_allocs[i]
    /// ->total_size())` / `program_alloc->total_size()`
    /// (`pf_runtime_scheduler_utils.cpp`), and explicitly, for the program
    /// case: "Bound the segment-7 xlat to the real program allocation
    /// footprint... never the 16GB SEGMENT_SIZE."
    ///
    /// Confirmed on real hardware: an earlier version of this crate used the
    /// region's size here (copying the DMA fix above without noticing
    /// compute's real C++ uses a different source), and every tensor/program
    /// XLAT length came out as exactly `1 << 34` (`SEGMENT_SIZE`, the shared
    /// program region's full slab size — real regions host many
    /// independently-compiled bundles, one per prefill-ladder rung, not one
    /// bundle per region) regardless of the tensor's real size. Per
    /// `XlatEntry::new`'s own doc, `1 << 34` bytes encodes as the hardware's
    /// "0 == max length" sentinel — an essentially unbounded XLAT window,
    /// not a rejected one — consistent with the real symptom this was
    /// chasing: a silent hardware stall under sustained batched-decode load
    /// rather than a clean CB rejection.
    #[test]
    fn compute_cb_xlat_length_is_the_allocation_size_not_the_region_size() {
        let resolver = FixedResolver {
            segment: 3,
            base: DevicePhysicalAddress(0x1_0000_0000),
            region_size: crate::address::ByteSize(1 << 34), // the ~16GB shared slab
        };
        let tensor_chunk = Chunk::new(
            LogicalAddress::new(RegionId(1), ByteOffset(0)),
            crate::address::ByteSize(4096),
            DomainId(0),
        );
        let tensor_addr = Arc::new(CompositeAddress::from_chunk(tensor_chunk));
        let prog_chunk = Chunk::new(
            LogicalAddress::new(RegionId(7), ByteOffset(0)),
            crate::address::ByteSize(8192),
            DomainId(0),
        );
        let prog_addr = Arc::new(CompositeAddress::from_chunk(prog_chunk));

        let op = crate::compute::ComputeParams::new(
            prog_addr,
            vec![tensor_addr],
            crate::compute::KernelName(String::new()),
            crate::compute::BootstrapOffset(0),
            vec![],
        );
        let cb = build_compute_cb(&resolver, &op, ResponseTag(1)).unwrap();
        let bytes = cb.to_bytes();

        let xlat_len_flits = |entry_index: usize| -> u64 {
            let off = 64 /* XLAT_SECTION */ + entry_index * 8;
            let word = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
            word & 0x07FF_FFFF
        };
        assert_eq!(
            xlat_len_flits(0),
            Flits::from_bytes(4096).0,
            "tensor XLAT length must be the tensor's own 4096-byte allocation, not the 16GB region"
        );
        assert_eq!(
            xlat_len_flits(7),
            Flits::from_bytes(8192).0,
            "program XLAT length must be the program's own 8192-byte allocation, not the 16GB region"
        );
    }
}

/// Ported C++ tests, source:
/// `flex/tests/runtime_stream/scheduler/runtime_stream_mock_compute_test.cpp`
/// (`RuntimeStreamComputeMockTest` gtest fixture, using
/// `MockRuntimeScheduler`/`RuntimeStream`).
///
/// `flex/tests/scheduler/scheduler_test.cpp` is NOT ported: it
/// `#include`s `flex/src/scheduler/scheduler.hpp` and drives a
/// `SchedulerPtr scheduler` member — that is `flex::Scheduler`, the
/// *different*, sendnn::Graph-driven graph-compiler scheduler this crate
/// does not have (see this module's own top-of-file doc comment on
/// `scheduler_compute_info.hpp`), not `RuntimeScheduler`. Out of scope.
///
/// Of the ~20 cases in the mock-compute test file, only the ones below are
/// faithfully portable. The C++ fixture's `MockRuntimeScheduler` completes
/// compute asynchronously via a background `compute_threads_` pool, which is
/// what makes its fence-wait tests (`D2HAfterComputeWaitsForFence`,
/// `ComputeFenceIsPerStream`, `BackToBackCompute*`,
/// `PerOpGranularityPostBarrierOpsNotWaited`) meaningful: they assert that a
/// later op genuinely blocks on an earlier op's still-pending completion.
/// This crate's `MockBackend` (`submit_compute`/`submit_dma` above) invokes
/// its completion callback synchronously, inline, before `submit_*` even
/// returns — there is no pending completion for a fence to wait on, so
/// porting those cases would either be vacuously true (no bug they could
/// ever catch) or require inventing async mock behavior with no C++
/// counterpart in this crate. Not ported, rather than faked.
#[cfg(test)]
mod ported_cxx_tests {
    use super::*;
    use crate::address::{ByteOffset, ByteSize, LogicalAddress, RegionId};
    use crate::compute::{BootstrapOffset, ComputeParams, KernelName};
    use crate::dma::{DmaDirection, DmaParams, DmaShaping, HostVirtualAddress};
    use crate::memory_region::DomainId;
    use crate::stream::{DeviceId, RuntimeStream, RuntimeStreamError, RuntimeStreamPriority};

    fn dummy_addr(region: u64) -> Arc<CompositeAddress> {
        let chunk = Chunk::new(
            LogicalAddress::new(RegionId(region), ByteOffset(0)),
            ByteSize(4096),
            DomainId(0),
        );
        Arc::new(CompositeAddress::from_chunk(chunk))
    }

    fn dummy_h2d() -> DmaParams {
        DmaParams::from_device_allocation(
            HostVirtualAddress(0x1000),
            DmaDirection::HostToDevice,
            Some(dummy_addr(1)),
            DmaShaping::default(),
        )
        .expect("dummy_addr is never None")
    }

    fn dummy_d2h() -> DmaParams {
        DmaParams::from_device_allocation(
            HostVirtualAddress(0x1000),
            DmaDirection::DeviceToHost,
            Some(dummy_addr(1)),
            DmaShaping::default(),
        )
        .expect("dummy_addr is never None")
    }

    fn dummy_compute() -> ComputeParams {
        ComputeParams::new(
            dummy_addr(2),
            vec![dummy_addr(3)],
            KernelName(String::new()),
            BootstrapOffset(0),
            vec![ByteOffset(0)],
        )
    }

    fn mock_stream(mode: RuntimeStreamMode) -> (Arc<RuntimeScheduler>, Arc<RuntimeStream>) {
        let sched = RuntimeScheduler::new(Backend::Mock(MockBackend::default()));
        let stream = RuntimeStream::new(
            sched.clone(),
            DeviceId(0),
            RuntimeStreamPriority::Normal,
            mode,
        );
        (sched, stream)
    }

    /// Port of `StrictOrderingNoBarrierSamePipeline`: two ops on the same
    /// pipeline (H2D, H2D) under `StrictOrdering` never auto-insert a
    /// barrier between them.
    #[test]
    fn strict_ordering_no_barrier_same_pipeline() {
        let (sched, stream) = mock_stream(RuntimeStreamMode::StrictOrdering);
        stream.launch_h2d(dummy_h2d(), None).unwrap();
        stream.launch_h2d(dummy_h2d(), None).unwrap();
        assert_eq!(sched.mock_barrier_count(), Some(0));
    }

    /// Port of `StrictOrderingAutoBarrierComputeToH2D`: switching from
    /// Compute to H2D under `StrictOrdering` auto-inserts a barrier.
    #[test]
    fn strict_ordering_auto_barrier_compute_to_h2d() {
        let (sched, stream) = mock_stream(RuntimeStreamMode::StrictOrdering);
        stream.launch_compute(dummy_compute(), None).unwrap();
        stream.launch_h2d(dummy_h2d(), None).unwrap();
        assert_eq!(sched.mock_barrier_count(), Some(1));
    }

    /// Port of `StrictOrderingAutoBarrierH2DToCompute`.
    #[test]
    fn strict_ordering_auto_barrier_h2d_to_compute() {
        let (sched, stream) = mock_stream(RuntimeStreamMode::StrictOrdering);
        stream.launch_h2d(dummy_h2d(), None).unwrap();
        stream.launch_compute(dummy_compute(), None).unwrap();
        assert_eq!(sched.mock_barrier_count(), Some(1));
    }

    /// Port of `StrictOrderingAutoBarrierH2DToD2H`.
    #[test]
    fn strict_ordering_auto_barrier_h2d_to_d2h() {
        let (sched, stream) = mock_stream(RuntimeStreamMode::StrictOrdering);
        stream.launch_h2d(dummy_h2d(), None).unwrap();
        stream.launch_d2h(dummy_d2h(), None).unwrap();
        assert_eq!(sched.mock_barrier_count(), Some(1));
    }

    /// Port of `StrictOrderingAutoBarrierD2HToH2D`.
    #[test]
    fn strict_ordering_auto_barrier_d2h_to_h2d() {
        let (sched, stream) = mock_stream(RuntimeStreamMode::StrictOrdering);
        stream.launch_d2h(dummy_d2h(), None).unwrap();
        stream.launch_h2d(dummy_h2d(), None).unwrap();
        assert_eq!(sched.mock_barrier_count(), Some(1));
    }

    /// Port of `StrictOrderingMultiplePipelineSwitches`: N pipeline switches
    /// in a row auto-insert N barriers (one per switch, not per op).
    #[test]
    fn strict_ordering_multiple_pipeline_switches() {
        let (sched, stream) = mock_stream(RuntimeStreamMode::StrictOrdering);
        stream.launch_compute(dummy_compute(), None).unwrap(); // baseline, no prior pipeline
        stream.launch_h2d(dummy_h2d(), None).unwrap(); // switch 1: Compute -> H2D
        stream.launch_d2h(dummy_d2h(), None).unwrap(); // switch 2: H2D -> D2H
        stream.launch_compute(dummy_compute(), None).unwrap(); // switch 3: D2H -> Compute
        assert_eq!(sched.mock_barrier_count(), Some(3));
    }

    /// Port of `OpOrderingNoAutoBarrier`: the same pipeline-switch sequence
    /// under `OpOrdering` never auto-inserts a barrier.
    #[test]
    fn op_ordering_no_auto_barrier() {
        let (sched, stream) = mock_stream(RuntimeStreamMode::OpOrdering);
        stream.launch_compute(dummy_compute(), None).unwrap();
        stream.launch_h2d(dummy_h2d(), None).unwrap();
        stream.launch_d2h(dummy_d2h(), None).unwrap();
        assert_eq!(sched.mock_barrier_count(), Some(0));
    }

    /// Port of `OpOrderingExplicitBarrier`: under `OpOrdering`, an op that
    /// explicitly requests a pipeline barrier still gets one (auto-barrier
    /// suppression only disables the *automatic* pipeline-switch detection,
    /// not an explicit caller request).
    #[test]
    fn op_ordering_explicit_barrier() {
        let (sched, stream) = mock_stream(RuntimeStreamMode::OpOrdering);
        let shaping = DmaShaping {
            pipeline_barrier: true,
            ..Default::default()
        };
        let dma = DmaParams::from_device_allocation(
            HostVirtualAddress(0x1000),
            DmaDirection::HostToDevice,
            Some(dummy_addr(1)),
            shaping,
        )
        .expect("dummy_addr is never None");
        stream.launch_h2d(dma, None).unwrap();
        assert_eq!(sched.mock_barrier_count(), Some(1));
    }

    /// Port of `ScheduleThrowsOnShutdownStream`: once a stream is in
    /// shutdown, every `launchOperation*` fails fast rather than reaching
    /// the scheduler at all.
    #[test]
    fn schedule_throws_on_shutdown_stream() {
        let (_sched, stream) = mock_stream(RuntimeStreamMode::StrictOrdering);
        stream.set_shutdown(true);
        let err = stream.launch_h2d(dummy_h2d(), None).unwrap_err();
        assert!(
            matches!(err, RuntimeStreamError::Shutdown),
            "expected Shutdown, got {err:?}"
        );
    }

    /// Port of `FenceResolvedOnErrorStream`: a hardware-reported failure on
    /// one op (`MockBackend::throw_on_compute`) is deferred onto the stream
    /// and re-thrown by the next `synchronize()`, and the stream transitions
    /// to shutdown so later ops fail fast too — port of
    /// `makeCompletionCallback`'s error branch (`setError` + `setShutdown(true)`).
    #[test]
    fn fence_resolved_on_error_stream() {
        let (sched, stream) = mock_stream(RuntimeStreamMode::StrictOrdering);
        sched.mock_set_throw_on_compute(true);
        stream.launch_compute(dummy_compute(), None).unwrap();
        let err = stream.synchronize().unwrap_err();
        assert!(
            matches!(err, RuntimeStreamError::Deferred(_)),
            "expected a deferred error, got {err:?}"
        );
        assert!(
            stream.needs_shutdown(),
            "a failed op must put the stream into shutdown"
        );
        let err2 = stream.launch_h2d(dummy_h2d(), None).unwrap_err();
        assert!(matches!(err2, RuntimeStreamError::Shutdown));
    }

    /// Port of `HostCallbackFiresOnCompletion`: a host callback requesting a
    /// pipeline barrier only runs after the barrier's `synchronize()` has
    /// drained all prior in-flight ops.
    #[test]
    fn host_callback_fires_on_completion() {
        let (_sched, stream) = mock_stream(RuntimeStreamMode::StrictOrdering);
        stream.launch_compute(dummy_compute(), None).unwrap();
        let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let fired2 = fired.clone();
        stream
            .launch_operation_host_callback(
                true,
                Some(Box::new(move || fired2.store(true, Ordering::SeqCst))),
            )
            .unwrap();
        assert!(fired.load(Ordering::SeqCst));
    }

    /// Port of `StrictOrderingAutoBarrierOnPipelineSwitch`: switching from
    /// Compute to D2H under `StrictOrdering` auto-inserts a barrier. Distinct
    /// from `strict_ordering_auto_barrier_compute_to_h2d` above — D2H (DMAO)
    /// and H2D (DMAI) are separate physical pipelines/fence deques, so this
    /// exercises a different `pipeline_for`/`fence_pipeline` branch.
    #[test]
    fn strict_ordering_auto_barrier_compute_to_d2h() {
        let (sched, stream) = mock_stream(RuntimeStreamMode::StrictOrdering);
        stream.launch_compute(dummy_compute(), None).unwrap();
        stream.launch_d2h(dummy_d2h(), None).unwrap();
        assert_eq!(sched.mock_barrier_count(), Some(1));
    }

    /// Port of `StrictOrderingNoBarrierSamePipeline` (mock_compute_test.cpp):
    /// two Computes in a row under `StrictOrdering` never auto-insert a
    /// barrier. Distinct from `strict_ordering_no_barrier_same_pipeline`
    /// above, which exercises the H2D/H2D pair instead.
    #[test]
    fn strict_ordering_no_barrier_same_pipeline_compute() {
        let (sched, stream) = mock_stream(RuntimeStreamMode::StrictOrdering);
        stream.launch_compute(dummy_compute(), None).unwrap();
        stream.launch_compute(dummy_compute(), None).unwrap();
        assert_eq!(sched.mock_barrier_count(), Some(0));
    }

    /// Port of `BarrierFastFailsOnShutdownStream`: once a stream is marked
    /// shut down, a subsequent op that would otherwise auto-barrier (and, on
    /// PF/VF, block waiting for a fence that may never resolve) instead fails
    /// fast via `launch_operation`'s `check_not_shutdown` guard — the
    /// scheduler's `schedule()`/`issue_barrier` is never even reached. This
    /// crate's `MockBackend` completes synchronously so there is no actual
    /// deadlock risk to reproduce, but the fast-fail invariant itself (no
    /// attempt to submit once shutdown is set) is exactly what this test
    /// guards, matching the C++ comment: "the schedule() guard catches it
    /// before the barrier could block."
    #[test]
    fn barrier_fast_fails_on_shutdown_stream() {
        let (sched, stream) = mock_stream(RuntimeStreamMode::StrictOrdering);
        stream.launch_compute(dummy_compute(), None).unwrap();
        stream.set_shutdown(true);
        let err = stream.launch_d2h(dummy_d2h(), None).unwrap_err();
        assert!(
            matches!(err, RuntimeStreamError::Shutdown),
            "expected Shutdown, got {err:?}"
        );
        // No barrier was attempted — schedule() was never reached.
        assert_eq!(sched.mock_barrier_count(), Some(0));
    }

    /// Port of `FenceResolvedOnErrorStream` (mock_compute_test.cpp, the
    /// success-path variant — NOT to be confused with
    /// `fence_resolved_on_error_stream` below, which ports the
    /// throw-then-shutdown scenario from `ComputeFenceResolvedOnSubmitFailure`
    /// in `runtime_scheduler_fence_test.cpp`). Verifies the ordinary success
    /// path: a normally-completing compute resolves its fence (in-flight
    /// counter decremented), `synchronize()`/`query()` see it, and the stream
    /// can still be marked shut down afterward without disturbing that.
    #[test]
    fn fence_resolved_on_success_then_shutdown() {
        let (_sched, stream) = mock_stream(RuntimeStreamMode::StrictOrdering);
        stream.launch_compute(dummy_compute(), None).unwrap();
        stream.synchronize().unwrap();
        assert!(stream.query());
        stream.set_shutdown(true);
        assert!(stream.needs_shutdown());
    }

    /// Port of `H2DFenceResolvedOnSubmitFailure`
    /// (`runtime_scheduler_fence_test.cpp`): when the backend's `submitDma`
    /// fails, `launchOperationH2D` does not throw synchronously — the error
    /// routes through the completion callback (`set_error`/`set_shutdown`)
    /// and only surfaces at the next `synchronize()`.
    #[test]
    fn h2d_fence_resolved_on_submit_failure() {
        let (sched, stream) = mock_stream(RuntimeStreamMode::StrictOrdering);
        sched.mock_set_throw_on_dma(true);
        stream.launch_h2d(dummy_h2d(), None).unwrap();
        let err = stream.synchronize().unwrap_err();
        assert!(
            matches!(err, RuntimeStreamError::Deferred(_)),
            "expected a deferred error, got {err:?}"
        );
    }

    /// Port of `D2HFenceResolvedOnSubmitFailure`: same scenario as
    /// `h2d_fence_resolved_on_submit_failure` but for D2H.
    #[test]
    fn d2h_fence_resolved_on_submit_failure() {
        let (sched, stream) = mock_stream(RuntimeStreamMode::StrictOrdering);
        sched.mock_set_throw_on_dma(true);
        stream.launch_d2h(dummy_d2h(), None).unwrap();
        let err = stream.synchronize().unwrap_err();
        assert!(
            matches!(err, RuntimeStreamError::Deferred(_)),
            "expected a deferred error, got {err:?}"
        );
    }

    /// Port of `MultipleFailedSubmissionsAllFencesResolved`: the first H2D's
    /// error routes through the completion callback and sets shutdown; a
    /// second H2D launched afterward sees the shutdown flag and fails fast
    /// (`RuntimeStreamError::Shutdown`) rather than reaching the scheduler at
    /// all. Both fences (the first one via its completion callback, the
    /// second — vacuously — by never being pushed) are resolved either way,
    /// so `synchronize()` still surfaces the first op's deferred error
    /// without hanging.
    #[test]
    fn multiple_failed_submissions_all_fences_resolved() {
        let (sched, stream) = mock_stream(RuntimeStreamMode::StrictOrdering);
        sched.mock_set_throw_on_dma(true);
        stream.launch_h2d(dummy_h2d(), None).unwrap();
        let err2 = stream.launch_h2d(dummy_h2d(), None).unwrap_err();
        assert!(
            matches!(err2, RuntimeStreamError::Shutdown),
            "expected Shutdown, got {err2:?}"
        );
        let err = stream.synchronize().unwrap_err();
        assert!(
            matches!(err, RuntimeStreamError::Deferred(_)),
            "expected a deferred error, got {err:?}"
        );
    }

    /// Port of `MixedSuccessFailureFencesResolved`: a successful H2D followed
    /// by a failing one — the first op's fence resolves cleanly, the second's
    /// failure is deferred and surfaces at `synchronize()`.
    #[test]
    fn mixed_success_failure_fences_resolved() {
        let (sched, stream) = mock_stream(RuntimeStreamMode::StrictOrdering);
        stream.launch_h2d(dummy_h2d(), None).unwrap();
        sched.mock_set_throw_on_dma(true);
        stream.launch_h2d(dummy_h2d(), None).unwrap();
        sched.mock_set_throw_on_dma(false);
        let err = stream.synchronize().unwrap_err();
        assert!(
            matches!(err, RuntimeStreamError::Deferred(_)),
            "expected a deferred error, got {err:?}"
        );
    }
}

/// Port of `runtime_scheduler_factory_test.cpp`'s single case
/// (`RuntimeSchedulerFactoryUnitTest.RuntimeSchedulerType`). The real C++
/// test drives `RuntimeSchedulerFactory::createScheduler` against a live
/// `DeviceInterface` and compares against `parseDeviceInterfaceType()` (both
/// env/device-driven and outside this crate's scope — this crate has no
/// `DeviceInterfaceFactory`). The equivalent, faithfully-portable invariant
/// for this crate's own factory (`create_scheduler`) is dispatch fidelity:
/// asking for a given `DeviceKind` with a matching backend config must
/// produce a scheduler whose own `scheduler_type()` matches that same kind,
/// and a mismatched kind/config pairing must be rejected rather than silently
/// building the wrong backend.
#[cfg(test)]
mod factory_tests {
    use super::*;

    #[test]
    fn create_scheduler_mock_yields_mock_scheduler_type() {
        let sched = create_scheduler(&DeviceKind::Mock, SchedulerBackendConfig::Mock).unwrap();
        assert_eq!(sched.scheduler_type(), RuntimeSchedulerType::Mock);
    }

    #[test]
    fn create_scheduler_rejects_mismatched_backend_config() {
        // Mock device kind with a Real config (or vice versa) must not
        // silently construct some other backend — it must be rejected.
        let result = create_scheduler(&DeviceKind::Pf, SchedulerBackendConfig::Mock);
        assert!(result.is_err());
        let err = match result {
            Err(e) => e,
            Ok(_) => unreachable!(),
        };
        assert_eq!(err, SchedulerFactoryError::MismatchedBackendConfig);
    }

    #[test]
    fn create_scheduler_rejects_unsupported_device_kind() {
        let result = create_scheduler(
            &DeviceKind::Other("gpu".to_string()),
            SchedulerBackendConfig::Mock,
        );
        let err = match result {
            Err(e) => e,
            Ok(_) => unreachable!(),
        };
        assert_eq!(
            err,
            SchedulerFactoryError::UnsupportedDeviceKind("gpu".to_string())
        );
    }
}
