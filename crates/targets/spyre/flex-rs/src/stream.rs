//! Port of `flex::RuntimeStream`'s OWN state machine — singleton-adjacent
//! bookkeeping owned by the stream itself, not the scheduler backend.
//! Source: `flex/include/flex/runtime_stream/runtime_stream.hpp` +
//! `flex/src/runtime_stream/runtime_stream.cpp`.
//!
//! Ported natively here: the in-flight operation counter, the shutdown flag,
//! deferred-exception storage (consume-once, first-error-wins), the
//! completion-callback lifecycle (`makeCompletionCallback`), `synchronize`'s
//! wait/notify protocol, and `query`.
//!
//! Deliberately NOT ported here (out of scope for this phase): the scheduler
//! backend itself (`RuntimeScheduler`/`PfRuntimeScheduler` — a parallel
//! agent's `crate::scheduler`) and the concrete per-operation parameter
//! structs (`DmaParams`, `ComputeParams`, `P2PRdmaSendParams`, etc. from
//! `runtime_submission_params.hpp`), which are a different subsystem's
//! artifacts. This file depends on the scheduler only through the
//! `SchedulerHandle` trait below — the minimal interface `RuntimeStream`
//! needs from it, matching what `runtime_stream.cpp` actually calls:
//! `registerStream`, `releaseStream`, and `schedule`. The scheduler phase is
//! expected to provide a concrete type implementing this trait (and may
//! relocate the trait itself into `crate::scheduler` during Assembly without
//! changing its shape).
//!
//! Also not ported: `launchOperationEventSignal`/`launchOperationEventWait`
//! (`runtime_stream.cpp:408-428`) and the two `fillAsync` overloads
//! (`runtime_stream.cpp:432-454`). The event pair takes an
//! `EventSignalOp*`/`EventWaitOp*` wrapping a shared `flex::Event`, and no
//! `Event` type exists anywhere in this crate — writing one would be
//! invented behavior rather than a port. `fillAsync`'s value+`DataFormats`
//! overload needs `RuntimeOperationFill::valueToFillPattern`, which lives in
//! the fill-params subsystem (`crate::dma`), not in `RuntimeStream`; once a
//! caller has a `FillPattern` the pattern overload is exactly
//! `launch_fill`. Every other `launchOperation*` is ported below.
//!
//! `RuntimeStream` never calls senlib directly in the C++ source — every
//! `launchOperation*` validates parameters, builds a `RuntimeOperation`, and
//! calls `scheduler_->schedule(...)`. The literal hardware submission lives
//! inside the scheduler backend. Hence there is no `unsafe`/FFI in this file.

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Opaque token uniquely identifying a stream to the scheduler, monotonically
/// assigned and unique per process lifetime. Port of `flex::StreamId`
/// (`runtime_stream_mode.hpp`: `using StreamId = uint64_t;`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StreamId(u64);

impl StreamId {
    fn next() -> Self {
        // Port of `nextStreamId()`'s `static std::atomic<StreamId> counter{0}`
        // (`runtime_stream.cpp:26-30`), where `StreamId = uint64_t`
        // (`runtime_stream_mode.hpp`). Previously used `AtomicI64` (signed) —
        // a StreamId should never be negative, and the real type is unsigned.
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        // Port of `nextStreamId()`'s `fetch_add(1, relaxed)` counter.
        StreamId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

/// Zero-based Spyre device index. Newtype over the raw `int` used by
/// `RuntimeStream::deviceId()`.
///
/// DELIBERATE WIDTH DEVIATION: the real C++ `device_id_` is a 32-bit `int`
/// (`RuntimeStream::RuntimeStream(RuntimeScheduler*, int device_id, ...)`,
/// `runtime_stream.cpp:31`), but this newtype is `i64`. Not narrowed to
/// `i32` to match exactly: `runtime.rs` uses `DeviceId(-1)` as a
/// no-device-selected sentinel threaded through `RuntimeContext::create`/
/// `build` and asserted on in multiple tests (`runtime.rs:145,357`), and
/// narrowing the public newtype would touch every one of those call sites
/// for a width change with no correctness benefit (both widths represent
/// the same small range of real device indices and the same sentinel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceId(pub i64);

/// Port of `flex::RuntimeStreamMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RuntimeStreamMode {
    /// Strict FIFO ordering: enqueue order == completion order, regardless
    /// of operation type.
    #[default]
    StrictOrdering,
    /// FIFO hardware-enqueue order only; a later op that must wait on an
    /// earlier op sets its own barrier flag rather than relying on ordering.
    OpOrdering,
}

/// Port of `flex::RuntimeStreamPriority`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RuntimeStreamPriority {
    #[default]
    Normal,
    High,
}

/// Port of `flex::RuntimeOperationType` — the type tag `RuntimeStream`
/// forwards to the scheduler so it can dispatch without needing the concrete
/// parameter struct (those structs are a different subsystem's artifact;
/// this crate only needs the tag to route completion/error handling
/// generically).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeOperationKind {
    H2D,
    D2H,
    Compute,
    HostCallback,
    Fill,
    LegacyP2PSendData,
    LegacyP2PRecvData,
    LegacyP2PDataExchange,
    P2PRdmaSend,
    P2PRdmaWait,
    P2PFirmwareSignal,
    P2PFirmwareWait,
    EventSignal,
    EventWait,
}

/// A submitted operation as `RuntimeStream` hands it to the scheduler:
/// the type tag plus the pipeline-barrier bit `RuntimeStream` itself
/// inspects (e.g. to decide whether `launchOperationHostCallback` must
/// `synchronize()` first). The concrete per-kind payload (device
/// addresses, kernel names, SIDs, ...) is opaque here — building and
/// interpreting it is the concrete-params subsystem's and the scheduler's
/// job, not `RuntimeStream`'s own state machine.
pub struct SubmittedOperation {
    pub kind: RuntimeOperationKind,
    pub pipeline_barrier: bool,
    pub payload: Box<dyn std::any::Any + Send>,
}

/// Errors surfaced by `RuntimeStream`'s own state machine. Distinct from
/// whatever error type the scheduler backend or concrete-params subsystem
/// defines for their own failures — `SchedulerError` below is intentionally
/// opaque (a boxed `dyn Error`) since this file must not assume a concrete
/// scheduler error type exists yet.
#[derive(Debug)]
pub enum RuntimeStreamError {
    /// Port of the `StreamInErrorState` runtime_error thrown by every
    /// `launchOperation*` when `needsShutdown()` is true.
    Shutdown,
    /// The scheduler rejected/failed the submission itself.
    Scheduler(SchedulerError),
    /// A previously-deferred exception, re-thrown (and cleared) by
    /// `synchronize()`.
    Deferred(Box<RuntimeStreamError>),
    /// Port of the `std::invalid_argument` guard in the callback path when a
    /// required parameter (e.g. a host callback) is null.
    NullCallback(&'static str),
    /// Port of `RAS::RUNTIMEOPERATION::InvalidChunkCount`
    /// (`runtime_stream_utils.hpp:132-142`).
    InvalidChunkCount {
        operation: &'static str,
        chunk_count: usize,
    },
    /// Port of `RAS::RUNTIMEOPERATION::InvalidTotalSize`
    /// (`runtime_stream_utils.hpp:150-160`).
    InvalidTotalSize {
        operation: &'static str,
        total_size: u64,
    },
    /// Port of `RAS::RUNTIMEOPERATION::ChunkOffsetNotAligned`
    /// (`runtime_stream_utils.hpp:168-186`).
    ChunkOffsetNotAligned {
        operation: &'static str,
        chunk_index: usize,
        offset: u64,
    },
}

impl std::fmt::Display for RuntimeStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shutdown => write!(f, "StreamInErrorState: stream is in error state"),
            Self::Scheduler(e) => write!(f, "scheduler rejected submission: {e}"),
            Self::Deferred(e) => write!(f, "deferred error from a prior operation: {e}"),
            Self::NullCallback(who) => write!(f, "{who}: required callback was null"),
            Self::InvalidChunkCount {
                operation,
                chunk_count,
            } => {
                write!(
                    f,
                    "{operation}: InvalidChunkCount (device_address has {chunk_count} chunks)"
                )
            }
            Self::InvalidTotalSize {
                operation,
                total_size,
            } => {
                write!(
                    f,
                    "{operation}: InvalidTotalSize (device_address total_size is {total_size})"
                )
            }
            Self::ChunkOffsetNotAligned {
                operation,
                chunk_index,
                offset,
            } => write!(
                f,
                "{operation}: ChunkOffsetNotAligned (chunk {chunk_index} offset {offset:#x} is not 128-byte aligned)"
            ),
        }
    }
}
impl std::error::Error for RuntimeStreamError {}

pub type SchedulerError = Box<dyn std::error::Error + Send + Sync>;

/// Best-effort extraction of a human-readable message from a caught panic
/// payload (`Box<dyn Any + Send>`), for wrapping into a `SchedulerError` in
/// `make_completion_callback`'s panic-catching path. Mirrors what
/// `std::exception::what()` gives the C++ side's `catch(...)` for a thrown
/// `std::exception`; Rust panic payloads are usually `&str`/`String` (from
/// `panic!`/`assert!`) but are not guaranteed to be, hence the fallback.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "user callback panicked with a non-string payload".to_string()
    }
}

/// The completion callback the scheduler invokes (typically from a
/// completion/response-worker thread, per the C++ doc comments) exactly once
/// per submitted operation, with `Ok(())` on success or `Err(..)` if the
/// operation itself failed on-device. Port of the closure built by
/// `makeCompletionCallback`, minus the user-callback invocation (which the
/// caller of `make_completion_callback` composes on top, matching the C++
/// lambda's `if(user_cb) user_cb(user_cb_data);` step).
pub type CompletionCallback = Box<dyn FnOnce(Result<(), SchedulerError>) + Send>;

/// The minimal interface `RuntimeStream` needs from the scheduler backend,
/// matching exactly what `runtime_stream.cpp` calls: `registerStream` (ctor),
/// `releaseStream` (dtor), and `schedule` (every `launchOperation*`).
///
/// Assumed to be implemented by a concrete type in `crate::scheduler` (a
/// parallel phase); this trait deliberately does not depend on that module so
/// this file compiles standalone regardless of scheduler.rs's existence or
/// exact shape. Assembly may re-export/relocate this trait under
/// `crate::scheduler` without changing its shape.
pub trait SchedulerHandle: Send + Sync {
    /// Port of `RuntimeScheduler::registerStream(StreamId, std::atomic<bool>*)`.
    /// `shutdown_flag` is the same `Arc<AtomicBool>` `RuntimeStream` itself
    /// reads/writes via `needs_shutdown`/`set_shutdown`, shared rather than
    /// passed as a raw pointer as in the C++ (the scheduler may want to poll
    /// or set it directly, e.g. on an unrecoverable hardware error).
    fn register_stream(&self, stream_id: StreamId, shutdown_flag: Arc<AtomicBool>);

    /// Port of `RuntimeScheduler::releaseStream(StreamId)`.
    fn release_stream(&self, stream_id: StreamId);

    /// Port of `RuntimeScheduler::schedule(StreamId, RuntimeStreamMode,
    /// RuntimeOperation&, CompletionCallback)`. Implementations own ordering
    /// semantics (`mode`) and eventual hardware submission; `completion` must
    /// be invoked exactly once, synchronously or asynchronously.
    fn schedule(
        &self,
        stream_id: StreamId,
        mode: RuntimeStreamMode,
        op: SubmittedOperation,
        completion: CompletionCallback,
    ) -> Result<(), SchedulerError>;
}

/// Port of `flex::RuntimeStream`. All fields below mirror the C++ class's
/// own state machine exactly; the scheduler pointer is the one abstract
/// dependency described above.
pub struct RuntimeStream {
    scheduler: Arc<dyn SchedulerHandle>,
    stream_id: StreamId,
    device_id: DeviceId,
    priority: RuntimeStreamPriority,
    mode: RuntimeStreamMode,

    // Port of `std::atomic<int> in_flight_{0}` (`runtime_stream.hpp:348`) —
    // `int` is 32-bit on every platform this runs on; previously `AtomicI64`,
    // widened past the real type for no correctness benefit (kept `AtomicI64`
    // for the underflow sentinel below only if it ever needed >32-bit
    // magnitude, which it never does — the guard clamps to 0 regardless).
    in_flight: AtomicI32,
    /// Shared with the scheduler at registration time (see
    /// `SchedulerHandle::register_stream`), matching the C++ passing
    /// `&shutdown_` into `registerStream`.
    shutdown: Arc<AtomicBool>,
    deferred_error: Mutex<Option<RuntimeStreamError>>,
    sync_count: AtomicI64,
    sync_mutex: Mutex<()>,
    sync_cv: Condvar,
}

impl RuntimeStream {
    /// Port of the `RuntimeStream` constructor, including
    /// `scheduler_->registerStream(streamId(), &shutdown_)`.
    pub fn new(
        scheduler: Arc<dyn SchedulerHandle>,
        device_id: DeviceId,
        priority: RuntimeStreamPriority,
        mode: RuntimeStreamMode,
    ) -> Arc<Self> {
        let stream_id = StreamId::next();
        let shutdown = Arc::new(AtomicBool::new(false));
        scheduler.register_stream(stream_id, shutdown.clone());
        Arc::new(Self {
            scheduler,
            stream_id,
            device_id,
            priority,
            mode,
            in_flight: AtomicI32::new(0),
            shutdown,
            deferred_error: Mutex::new(None),
            sync_count: AtomicI64::new(0),
            sync_mutex: Mutex::new(()),
            sync_cv: Condvar::new(),
        })
    }

    pub fn stream_id(&self) -> StreamId {
        self.stream_id
    }

    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    pub fn priority(&self) -> RuntimeStreamPriority {
        self.priority
    }

    pub fn execution_mode(&self) -> RuntimeStreamMode {
        self.mode
    }

    pub fn needs_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }

    pub fn set_shutdown(&self, shutdown: bool) {
        self.shutdown.store(shutdown, Ordering::Release);
    }

    fn increment_in_flight(&self) {
        self.in_flight.fetch_add(1, Ordering::AcqRel);
    }

    /// Port of `decrementInFlight`. Holds `sync_mutex_` across the decrement
    /// and notify so a concurrent `synchronize()` cannot observe a missed
    /// wakeup between its predicate check and going to sleep — same rationale
    /// as the C++ comment.
    fn decrement_in_flight(&self) {
        let _guard = self.sync_mutex.lock().unwrap();
        let remaining = self.in_flight.fetch_sub(1, Ordering::AcqRel) - 1;
        if remaining < 0 {
            // Port of the underflow guard: clamp and still notify so a
            // waiting synchronize() is not left stuck. Matches the C++
            // `LOG_ERROR << "RuntimeStream::decrementInFlight() underflow: ..."`
            // — this is a genuine bug signal (a completion fired more times
            // than an operation was launched) and must not be silent.
            tracing::error!(
                "RuntimeStream::decrementInFlight() underflow: remaining={remaining} device={:?}",
                self.device_id
            );
            self.in_flight.store(0, Ordering::Release);
        }
        if remaining <= 0 {
            self.sync_cv.notify_all();
        }
    }

    /// Port of `RuntimeStream::setError`: only the first deferred error is
    /// kept; later calls are ignored.
    pub fn set_error(&self, error: RuntimeStreamError) {
        let mut guard = self.deferred_error.lock().unwrap();
        if guard.is_none() {
            *guard = Some(error);
        }
    }

    /// Port of `makeCompletionCallback` (`runtime_stream.cpp:60-83`).
    /// Increments `in_flight_` immediately on the caller's thread (before the
    /// operation reaches the scheduler), and returns a `CompletionCallback`
    /// that, on the completion thread: records a deferred error and sets
    /// shutdown on failure, invokes `user_cb` on success, and always
    /// decrements `in_flight_`.
    ///
    /// PANIC-SAFETY BUG FOUND AND FIXED HERE: the real C++ lambda wraps the
    /// `user_cb(user_cb_data)` invocation itself inside the *same*
    /// `try { ... } catch(...) { setError(...); setShutdown(true); }` block
    /// that handles a deferred exception from the operation
    /// (`runtime_stream.cpp:65-77`) — so a throwing `user_cb` still reaches
    /// `decrementInFlight()`, just via the error path instead of falling
    /// straight through. A prior version of this port had no such wrapping:
    /// a panicking `user_cb` would unwind straight out of the closure,
    /// skipping `decrement_in_flight()` entirely — `in_flight` never reaches
    /// zero, and `synchronize()` (including the one implicitly run by
    /// `Drop`) hangs forever. `std::panic::catch_unwind` below reproduces the
    /// C++ catch-and-convert-to-deferred-error behavior, and
    /// `decrement_in_flight()` is called unconditionally afterward exactly
    /// like the C++ `decrementInFlight()` call sitting after (outside) the
    /// `try`/`catch`.
    ///
    /// `self` must be an `Arc<RuntimeStream>` because the returned closure
    /// outlives this call and may run on another thread.
    pub fn make_completion_callback(
        self: &Arc<Self>,
        user_cb: Option<Box<dyn FnOnce() + Send>>,
    ) -> CompletionCallback {
        self.increment_in_flight();
        let this = self.clone();
        Box::new(move |result: Result<(), SchedulerError>| {
            match result {
                Ok(()) => {
                    if let Some(cb) = user_cb {
                        // Port of the C++ `try { ... user_cb(user_cb_data);
                        // } catch(...) { setError(...); setShutdown(true); }`
                        // wrapping: a panicking user callback must not skip
                        // `decrement_in_flight()` below, so catch it here and
                        // convert it into a deferred error + shutdown, the
                        // same as an on-device operation failure.
                        if let Err(panic) =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(cb))
                        {
                            let msg = panic_message(&panic);
                            tracing::error!(
                                "stream entering error state: user callback panicked: {msg}"
                            );
                            this.set_error(RuntimeStreamError::Scheduler(msg.into()));
                            this.set_shutdown(true);
                        }
                    }
                }
                Err(e) => {
                    // This is often the ONLY place the real root cause of a
                    // hardware failure is ever visible: it happens on the
                    // scheduler's completion thread, asynchronously, and
                    // `set_error` only stores it for a later `synchronize()`
                    // to re-throw — a caller that never calls `synchronize()`
                    // before hitting the resulting `StreamInErrorState` on
                    // some unrelated later operation would otherwise see
                    // only that generic wrapper, never this actual cause.
                    tracing::error!("stream entering error state: {e}");
                    this.set_error(RuntimeStreamError::Scheduler(e));
                    this.set_shutdown(true);
                }
            }
            this.decrement_in_flight();
        })
    }

    fn check_not_shutdown(&self) -> Result<(), RuntimeStreamError> {
        if self.needs_shutdown() {
            Err(RuntimeStreamError::Shutdown)
        } else {
            Ok(())
        }
    }

    /// Port of every `launchOperation*` except `launchOperationHostCallback`
    /// (see `launch_operation_host_callback`): all of them are "check
    /// shutdown, wrap the user callback, call `scheduler_->schedule`" once
    /// their type-specific parameter validation has already happened in the
    /// concrete-params subsystem. `RuntimeStream`'s own contribution is
    /// exactly this shutdown check + callback lifecycle + scheduler handoff.
    pub fn launch_operation(
        self: &Arc<Self>,
        op: SubmittedOperation,
        user_cb: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<(), RuntimeStreamError> {
        self.check_not_shutdown()?;
        let completion = self.make_completion_callback(user_cb);
        self.scheduler
            .schedule(self.stream_id, self.mode, op, completion)
            .map_err(RuntimeStreamError::Scheduler)
    }

    /// Port of `RuntimeStream::launchOperationH2D`: builds the boxed
    /// `SubmittedOperation` from a caller-supplied `DmaParams` (host-to-device
    /// direction) and hands it to `launch_operation`, hiding the
    /// boxing/type-tag convention `SchedulerHandle::schedule` requires.
    /// Port of `containOneOrMoreChunks` (`runtime_stream_utils.hpp:132-142`).
    fn contain_one_or_more_chunks(
        device_address: &crate::address::CompositeAddress,
        operation: &'static str,
    ) -> Result<(), RuntimeStreamError> {
        if device_address.num_chunks() == 0 {
            return Err(RuntimeStreamError::InvalidChunkCount {
                operation,
                chunk_count: 0,
            });
        }
        Ok(())
    }

    /// Port of `totalSizeGreaterThanZero` (`runtime_stream_utils.hpp:150-160`).
    fn total_size_greater_than_zero(
        device_address: &crate::address::CompositeAddress,
        operation: &'static str,
    ) -> Result<(), RuntimeStreamError> {
        let total_size = device_address.total_size().as_u64();
        if total_size == 0 {
            return Err(RuntimeStreamError::InvalidTotalSize {
                operation,
                total_size,
            });
        }
        Ok(())
    }

    /// Port of `eachChunk128ByteAligned` (`runtime_stream_utils.hpp:168-186`):
    /// `chunk.addr.offset & 0x7F`.
    ///
    /// This one is load-bearing, not cosmetic. The control-block wire format
    /// encodes addresses at 128-byte flit granularity
    /// (`VirtualAddressSbf`/`Flits::from_bytes`, a truncating `>> 7`), so a
    /// chunk offset that is not 128-byte aligned does not get rejected — it
    /// gets silently truncated into a CB that addresses the wrong place. This
    /// check is what makes that a reported error instead of corrupt hardware
    /// behavior, and it was entirely absent from the port.
    fn each_chunk_128_byte_aligned(
        device_address: &crate::address::CompositeAddress,
        operation: &'static str,
    ) -> Result<(), RuntimeStreamError> {
        for (chunk_index, chunk) in device_address.chunks().iter().enumerate() {
            let offset = chunk.addr.offset.0;
            if offset & 0x7F != 0 {
                return Err(RuntimeStreamError::ChunkOffsetNotAligned {
                    operation,
                    chunk_index,
                    offset,
                });
            }
        }
        Ok(())
    }

    pub fn launch_h2d(
        self: &Arc<Self>,
        params: crate::dma::DmaParams,
        user_cb: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<(), RuntimeStreamError> {
        // `runtime_stream.cpp:136-142`: chunk count, total size, 128-byte alignment.
        Self::contain_one_or_more_chunks(params.device_address(), "RuntimeOperationH2D")?;
        Self::total_size_greater_than_zero(params.device_address(), "RuntimeOperationH2D")?;
        Self::each_chunk_128_byte_aligned(params.device_address(), "RuntimeOperationH2D")?;
        let pipeline_barrier = params.pipeline_barrier();
        let op = SubmittedOperation {
            kind: RuntimeOperationKind::H2D,
            pipeline_barrier,
            payload: Box::new(params),
        };
        self.launch_operation(op, user_cb)
    }

    /// Port of `RuntimeStream::launchOperationD2H`: same shape as
    /// `launch_h2d`, tagged for the device-to-host direction.
    pub fn launch_d2h(
        self: &Arc<Self>,
        params: crate::dma::DmaParams,
        user_cb: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<(), RuntimeStreamError> {
        // `runtime_stream.cpp:166-172`: same three as H2D.
        Self::contain_one_or_more_chunks(params.device_address(), "RuntimeOperationD2H")?;
        Self::total_size_greater_than_zero(params.device_address(), "RuntimeOperationD2H")?;
        Self::each_chunk_128_byte_aligned(params.device_address(), "RuntimeOperationD2H")?;
        let pipeline_barrier = params.pipeline_barrier();
        let op = SubmittedOperation {
            kind: RuntimeOperationKind::D2H,
            pipeline_barrier,
            payload: Box::new(params),
        };
        self.launch_operation(op, user_cb)
    }

    /// Port of `RuntimeStream::launchOperationCompute`: builds the boxed
    /// `SubmittedOperation` from a caller-supplied `ComputeParams` and hands
    /// it to `launch_operation`.
    pub fn launch_compute(
        self: &Arc<Self>,
        params: crate::compute::ComputeParams,
        user_cb: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<(), RuntimeStreamError> {
        // `runtime_stream.cpp:256-267`: chunk count and 128-byte alignment on the
        // program address, then 128-byte alignment on EVERY tensor operand.
        // Deliberately NO `totalSizeGreaterThanZero` — the C++ does not call it
        // for Compute.
        Self::contain_one_or_more_chunks(params.device_address(), "RuntimeOperationCompute")?;
        Self::each_chunk_128_byte_aligned(params.device_address(), "RuntimeOperationCompute")?;
        for address in params.tensor_allocs() {
            Self::each_chunk_128_byte_aligned(address, "RuntimeOperationCompute")?;
        }
        let pipeline_barrier = params.pipeline_barrier();
        let op = SubmittedOperation {
            kind: RuntimeOperationKind::Compute,
            pipeline_barrier,
            payload: Box::new(params),
        };
        self.launch_operation(op, user_cb)
    }

    /// Port of `RuntimeStream::launchOperationFill` (`runtime_stream.cpp:456-471`).
    /// Note the C++ passes `makeCompletionCallback(nullptr, nullptr)` — a fill
    /// carries no user callback (`FillParams` does not derive from
    /// `RuntimeOperationParams`, so it has no `callback` field), hence no
    /// `user_cb` parameter here. `pipeline_barrier` is likewise absent from
    /// `FillParams` in both the C++ and this crate's `crate::dma::FillParams`,
    /// so the submitted operation carries `pipeline_barrier: false` — matching
    /// the C++, which never calls `setPipelineBarrier` on a
    /// `RuntimeOperationFill` built from `FillParams`.
    pub fn launch_fill(
        self: &Arc<Self>,
        params: crate::dma::FillParams,
    ) -> Result<(), RuntimeStreamError> {
        // `runtime_stream.cpp:466`: chunk count ONLY — the C++ calls neither
        // `totalSizeGreaterThanZero` nor `eachChunk128ByteAligned` for Fill.
        Self::contain_one_or_more_chunks(params.device_address(), "RuntimeOperationFill")?;
        let op = SubmittedOperation {
            kind: RuntimeOperationKind::Fill,
            pipeline_barrier: false,
            payload: Box::new(params),
        };
        self.launch_operation(op, None)
    }

    /// Port of `RuntimeStream::launchOperationP2PRdmaSend` (`runtime_stream.cpp:182-200`).
    /// `pipeline_barrier` is an explicit argument because this crate's
    /// `P2PRdmaSendParams` (a `crate::scheduler` type, another phase's
    /// artifact) does not carry the `pipeline_barrier` field the C++
    /// `P2PRdmaSendParams` inherits from `RuntimeOperationParams` — the C++
    /// reads it off `params` and calls `op.setPipelineBarrier(...)`.
    pub fn launch_p2p_rdma_send(
        self: &Arc<Self>,
        params: crate::scheduler::P2PRdmaSendParams,
        pipeline_barrier: bool,
        user_cb: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<(), RuntimeStreamError> {
        let op = SubmittedOperation {
            kind: RuntimeOperationKind::P2PRdmaSend,
            pipeline_barrier,
            payload: Box::new(params),
        };
        self.launch_operation(op, user_cb)
    }

    /// Port of `RuntimeStream::launchOperationP2PRdmaWait` (`runtime_stream.cpp:202-214`).
    pub fn launch_p2p_rdma_wait(
        self: &Arc<Self>,
        params: crate::scheduler::P2PRdmaWaitParams,
        pipeline_barrier: bool,
        user_cb: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<(), RuntimeStreamError> {
        let op = SubmittedOperation {
            kind: RuntimeOperationKind::P2PRdmaWait,
            pipeline_barrier,
            payload: Box::new(params),
        };
        self.launch_operation(op, user_cb)
    }

    /// Port of `RuntimeStream::launchOperationP2PFirmwareSignal` (`runtime_stream.cpp:216-228`).
    pub fn launch_p2p_firmware_signal(
        self: &Arc<Self>,
        params: crate::scheduler::P2PFirmwareSignalParams,
        pipeline_barrier: bool,
        user_cb: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<(), RuntimeStreamError> {
        let op = SubmittedOperation {
            kind: RuntimeOperationKind::P2PFirmwareSignal,
            pipeline_barrier,
            payload: Box::new(params),
        };
        self.launch_operation(op, user_cb)
    }

    /// Port of `RuntimeStream::launchOperationP2PFirmwareWait` (`runtime_stream.cpp:230-243`).
    pub fn launch_p2p_firmware_wait(
        self: &Arc<Self>,
        params: crate::scheduler::P2PFirmwareWaitParams,
        pipeline_barrier: bool,
        user_cb: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<(), RuntimeStreamError> {
        let op = SubmittedOperation {
            kind: RuntimeOperationKind::P2PFirmwareWait,
            pipeline_barrier,
            payload: Box::new(params),
        };
        self.launch_operation(op, user_cb)
    }

    /// Port of `RuntimeStream::launchOperationLegacyP2PDataExchange`
    /// (`runtime_stream.cpp:293-330`): the whole ordered send/recv list is
    /// submitted as ONE operation with ONE shared pipeline barrier and ONE
    /// shared completion callback (the C++ appends every child op into a
    /// single `RuntimeOperationLegacyP2PDataExchange` and schedules that),
    /// which is why the payload is the full `Vec` and there is exactly one
    /// `make_completion_callback` (i.e. one `in_flight` increment) for the
    /// entire exchange. Each element's `is_send` selects send vs recv,
    /// matching the C++ `op_param->is_send_` branch.
    pub fn launch_legacy_p2p_data_exchange(
        self: &Arc<Self>,
        operations: Vec<crate::scheduler::P2PDataParams>,
        pipeline_barrier: bool,
        user_cb: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<(), RuntimeStreamError> {
        let op = SubmittedOperation {
            kind: RuntimeOperationKind::LegacyP2PDataExchange,
            pipeline_barrier,
            payload: Box::new(operations),
        };
        self.launch_operation(op, user_cb)
    }

    /// Port of `launchOperationHostCallback`: uniquely among the
    /// `launchOperation*` family, this one does NOT go through the
    /// scheduler's async completion path at all. If a barrier was
    /// requested it blocks on `synchronize()` first, then invokes the
    /// callback inline on the caller's thread.
    pub fn launch_operation_host_callback(
        self: &Arc<Self>,
        pipeline_barrier: bool,
        callback: Option<Box<dyn FnOnce()>>,
    ) -> Result<(), RuntimeStreamError> {
        let callback = callback.ok_or(RuntimeStreamError::NullCallback(
            "RuntimeOperationHostCallback",
        ))?;
        if pipeline_barrier {
            self.synchronize()?;
        }
        callback();
        Ok(())
    }

    /// Port of `RuntimeStream::synchronize`. Skips the wait loop entirely
    /// once the stream is already in shutdown (matching the C++
    /// `if(!needsShutdown())` guard), then always consumes (swaps out) any
    /// deferred error exactly once.
    pub fn synchronize(&self) -> Result<(), RuntimeStreamError> {
        if !self.needs_shutdown() {
            let mut guard = self.sync_mutex.lock().unwrap();
            let wait_start = std::time::Instant::now();
            // Port of the C++ `wait_for(lock, 60s, predicate)` loop: wake up
            // every 60s while still waiting and log, so a lost completion
            // (ResponseWorker processed the hardware response but the
            // completion callback never reached decrementInFlight()) is
            // visible instead of synchronize() hanging silently forever.
            while self.in_flight.load(Ordering::Acquire) > 0 {
                let (new_guard, timeout_result) = self
                    .sync_cv
                    .wait_timeout_while(guard, std::time::Duration::from_secs(60), |_| {
                        self.in_flight.load(Ordering::Acquire) > 0
                    })
                    .unwrap();
                guard = new_guard;
                if timeout_result.timed_out() {
                    let elapsed_ms = wait_start.elapsed().as_millis();
                    tracing::error!(
                        "RuntimeStream::synchronize() still waiting after {elapsed_ms}ms: in_flight_={} device={:?} — possible lost completion",
                        self.in_flight.load(Ordering::Acquire),
                        self.device_id
                    );
                }
            }
            // Port of `runtime_stream.cpp:388-391`:
            //
            //     auto sync_id = sync_count_++;
            //     telemetry::UnifiedProfiler::Instance().FlushToJson("stream-sync-" + std::to_string(sync_id));
            //
            // The counter is kept (it is this stream's sync sequence number and
            // costs nothing), but the flush is NOT ported: flex's profiler
            // subsystem was removed from this crate — every one of its backends
            // was permanently disabled here, so `FlushToJson` drained an event
            // buffer nothing ever wrote to, while taking a process-global mutex
            // on every `synchronize()`. If flex telemetry is ever wanted, this
            // is where the per-sync drain belongs.
            let _sync_id = self.sync_count.fetch_add(1, Ordering::Relaxed);
            drop(guard);
        }

        let deferred = self.deferred_error.lock().unwrap().take();
        match deferred {
            Some(err) => Err(RuntimeStreamError::Deferred(Box::new(err))),
            None => Ok(()),
        }
    }

    /// Port of `RuntimeStream::query`: `return in_flight_ == 0;`
    /// (`runtime_stream.cpp:430`). `std::atomic<int>::operator==` (via
    /// implicit conversion / `operator T`) uses `memory_order_seq_cst`, so
    /// this uses `Ordering::SeqCst` to match exactly (previously
    /// `Ordering::Acquire`).
    pub fn query(&self) -> bool {
        self.in_flight.load(Ordering::SeqCst) == 0
    }
}

impl Drop for RuntimeStream {
    /// Port of `~RuntimeStream`: best-effort drain (errors are logged and
    /// suppressed, never propagated from a destructor) followed by
    /// `scheduler_->releaseStream(streamId())`.
    fn drop(&mut self) {
        if let Err(e) = self.synchronize() {
            tracing::warn!(
                "RuntimeStream::drop: deferred error suppressed while draining stream for device {:?}: {e}",
                self.device_id
            );
        }
        self.scheduler.release_stream(self.stream_id);
    }
}

/// Ported from the real C++ suite `flex/tests/runtime_stream/stream/`.
///
/// This crate has no RuntimeContext/RuntimeScheduler/hardware layer yet (those
/// are parallel phases), so every test here exercises `RuntimeStream`'s own
/// orchestration logic in isolation against a local `MockScheduler` test
/// double implementing `SchedulerHandle` — analogous to how the real C++
/// tests exercise `RuntimeStream` against a live `RuntimeContext`, except we
/// have no such fixture to create streams through, so tests construct
/// `RuntimeStream::new` directly.
///
/// Per-file disposition (10 C++ files, ported/skipped as noted):
///
/// - `device_interface_factory_test.cpp` (1 test): SKIPPED — tests
///   `DeviceInterfaceFactory::createDevice()`, a device-bring-up type with no
///   Rust counterpart in this crate.
/// - `event_primitives_test.cpp` (8 tests), `event_wait_test.cpp` (9 tests):
///   SKIPPED — both test `flex::Event`/`EventWaitOp`/`EventSignalOp`, a
///   synchronization-primitive class with no Rust counterpart; `stream.rs`
///   only carries the `EventSignal`/`EventWait` *tag* in
///   `RuntimeOperationKind`, not an `Event` object to test.
/// - `runtime_stream_api_test.cpp` (7 tests): 3 ported
///   (`create_default_stream_has_default_properties`,
///   `create_stream_with_parameters`,
///   `create_multiple_streams_have_independent_properties`) plus 2 adapted
///   (`stream_device_id_matches_constructor_arg`,
///   `concurrent_stream_creation_yields_unique_stream_ids`); `GetDefaultStream`,
///   `DestroyStream`, `CannotDestroyDefaultStream` SKIPPED as `RuntimeContext`
///   API, not `RuntimeStream`'s own state machine (a parallel phase).
/// - `runtime_stream_compute_test.cpp` (9 tests): 1 ported
///   (`compute_completion_invokes_user_callback`, adapted from
///   `ComputeWithHostCallback`); the rest SKIPPED — hardware-integration tests
///   (real kernel execution/data verification) or `ComputeParams`
///   construction-validation tests (`NullProgramAddressThrows`,
///   `ZeroedProgramBufferThrows`), neither of which is `RuntimeStream`'s own
///   logic.
/// - `runtime_stream_dma_test.cpp` (8 tests): 3 ported
///   (`query_reflects_in_flight_operations`, adapted from
///   `StreamQueryDuringDma`; `dma_callback_functionality`; and the important
///   `deferred_error_first_error_wins`, ported from `DeferredErrorHandling`);
///   the rest SKIPPED — hardware DMA + data-verification integration tests.
/// - `runtime_stream_fill_test.cpp` (11 tests): SKIPPED — the 7
///   hardware-fill tests need a real device, and the 4 `FillPatternUnittest`
///   tests exercise `RuntimeOperationFill::valueToFillPattern`, a
///   fill-parameter-subsystem function `stream.rs` has no `launch_fill`
///   method for.
/// - `runtime_stream_p2p_firmware_test.cpp` (7 tests),
///   `runtime_stream_p2p_wdone_test.cpp` (5 tests): SKIPPED — all require real
///   VF/PF hardware (pinned PCIe memory, firmware flit counters, device wdone
///   registers).
/// - `runtime_stream_utils_test.cpp` (5 tests): SKIPPED — tests free functions
///   in `runtime_stream_utils.hpp` (`deviceTypesToStr`,
///   `parseDeviceInterfaceType`, ...), not the `RuntimeStream` class; out of
///   this file's scope (device-types module, ported separately).
#[cfg(test)]
mod validator_tests {
    use super::*;
    use crate::address::{ByteOffset, ByteSize, Chunk, CompositeAddress, LogicalAddress, RegionId};
    use crate::memory_region::DomainId;

    fn addr_with_offset(offset: u64) -> CompositeAddress {
        CompositeAddress::from_chunk(Chunk::new(
            LogicalAddress::new(RegionId(1), ByteOffset(offset)),
            ByteSize(4096),
            DomainId(0),
        ))
    }

    /// Port of `eachChunk128ByteAligned`'s `chunk.addr.offset & 0x7F` check
    /// (`runtime_stream_utils.hpp:168-186`). This matters because the CB wire
    /// format encodes addresses at 128-byte flit granularity with a truncating
    /// shift, so an unaligned offset would otherwise produce a CB pointing
    /// somewhere else entirely rather than an error.
    #[test]
    fn chunk_offset_must_be_128_byte_aligned() {
        let aligned = addr_with_offset(128 * 7);
        RuntimeStream::each_chunk_128_byte_aligned(&aligned, "RuntimeOperationH2D")
            .expect("a 128-byte-aligned offset is accepted");

        for bad in [1u64, 0x7F, 129, 4095] {
            let err = RuntimeStream::each_chunk_128_byte_aligned(
                &addr_with_offset(bad),
                "RuntimeOperationH2D",
            )
            .expect_err("an unaligned chunk offset must be rejected");
            match err {
                RuntimeStreamError::ChunkOffsetNotAligned {
                    operation,
                    chunk_index,
                    offset,
                } => {
                    assert_eq!(operation, "RuntimeOperationH2D");
                    assert_eq!(chunk_index, 0);
                    assert_eq!(offset, bad);
                }
                other => panic!("wrong error: {other}"),
            }
        }
    }

    /// Port of `containOneOrMoreChunks` (`runtime_stream_utils.hpp:132-142`).
    #[test]
    fn empty_composite_address_is_rejected() {
        let empty = CompositeAddress::from_chunks(vec![]);
        let err = RuntimeStream::contain_one_or_more_chunks(&empty, "RuntimeOperationFill")
            .expect_err("zero chunks must be rejected");
        assert!(
            matches!(
                err,
                RuntimeStreamError::InvalidChunkCount { chunk_count: 0, .. }
            ),
            "got {err}"
        );
    }

    /// Port of `totalSizeGreaterThanZero` (`runtime_stream_utils.hpp:150-160`).
    #[test]
    fn zero_total_size_is_rejected() {
        let zero = CompositeAddress::from_chunk(Chunk::new(
            LogicalAddress::new(RegionId(1), ByteOffset(0)),
            ByteSize(0),
            DomainId(0),
        ));
        let err = RuntimeStream::total_size_greater_than_zero(&zero, "RuntimeOperationD2H")
            .expect_err("a zero total size must be rejected");
        assert!(
            matches!(
                err,
                RuntimeStreamError::InvalidTotalSize { total_size: 0, .. }
            ),
            "got {err}"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::AtomicUsize;

    use super::*;

    /// Test double for `SchedulerHandle`, standing in for the real
    /// `RuntimeScheduler` the C++ tests exercise through a live
    /// `RuntimeContext`. `outcomes` is a queue of per-`schedule()`-call
    /// results (`Ok`/`Err` string reason), consumed FIFO, defaulting to `Ok`
    /// once exhausted. When `auto_complete` is `true` (the default),
    /// `schedule()` invokes `completion` synchronously with the next outcome;
    /// when `false`, completions are stashed in `pending` for the test to
    /// fire manually via `fire_next`, modeling an async completion thread.
    struct MockScheduler {
        registered: Mutex<Vec<(StreamId, Arc<AtomicBool>)>>,
        released: Mutex<Vec<StreamId>>,
        outcomes: Mutex<VecDeque<Result<(), String>>>,
        auto_complete: bool,
        pending: Mutex<Vec<CompletionCallback>>,
        schedule_calls: AtomicUsize,
    }

    impl MockScheduler {
        fn new(auto_complete: bool) -> Arc<Self> {
            Arc::new(Self {
                registered: Mutex::new(Vec::new()),
                released: Mutex::new(Vec::new()),
                outcomes: Mutex::new(VecDeque::new()),
                auto_complete,
                pending: Mutex::new(Vec::new()),
                schedule_calls: AtomicUsize::new(0),
            })
        }

        fn push_outcome(&self, outcome: Result<(), String>) {
            self.outcomes.lock().unwrap().push_back(outcome);
        }

        /// Fires the oldest still-pending completion (only meaningful when
        /// `auto_complete` is `false`).
        fn fire_next(&self, result: Result<(), SchedulerError>) {
            let cb = self.pending.lock().unwrap().remove(0);
            cb(result);
        }

        fn pending_count(&self) -> usize {
            self.pending.lock().unwrap().len()
        }
    }

    impl SchedulerHandle for MockScheduler {
        fn register_stream(&self, stream_id: StreamId, shutdown_flag: Arc<AtomicBool>) {
            self.registered
                .lock()
                .unwrap()
                .push((stream_id, shutdown_flag));
        }

        fn release_stream(&self, stream_id: StreamId) {
            self.released.lock().unwrap().push(stream_id);
        }

        fn schedule(
            &self,
            _stream_id: StreamId,
            _mode: RuntimeStreamMode,
            _op: SubmittedOperation,
            completion: CompletionCallback,
        ) -> Result<(), SchedulerError> {
            self.schedule_calls.fetch_add(1, Ordering::Relaxed);
            if self.auto_complete {
                let outcome = self.outcomes.lock().unwrap().pop_front().unwrap_or(Ok(()));
                completion(outcome.map_err(|e| -> SchedulerError { e.into() }));
            } else {
                self.pending.lock().unwrap().push(completion);
            }
            Ok(())
        }
    }

    fn dummy_op(kind: RuntimeOperationKind) -> SubmittedOperation {
        SubmittedOperation {
            kind,
            pipeline_barrier: false,
            payload: Box::new(()),
        }
    }

    // Port of `RuntimeStreamUnittest.CreateDefaultStream`.
    #[test]
    fn create_default_stream_has_default_properties() {
        let scheduler = MockScheduler::new(true);
        let stream = RuntimeStream::new(
            scheduler,
            DeviceId(0),
            RuntimeStreamPriority::default(),
            RuntimeStreamMode::default(),
        );

        assert_eq!(stream.execution_mode(), RuntimeStreamMode::StrictOrdering);
        assert_eq!(stream.priority(), RuntimeStreamPriority::Normal);
    }

    // Port of `RuntimeStreamUnittest.CreateStreamWithParameters`.
    #[test]
    fn create_stream_with_parameters() {
        let scheduler = MockScheduler::new(true);
        let stream = RuntimeStream::new(
            scheduler,
            DeviceId(0),
            RuntimeStreamPriority::High,
            RuntimeStreamMode::OpOrdering,
        );

        assert_eq!(stream.execution_mode(), RuntimeStreamMode::OpOrdering);
        assert_eq!(stream.priority(), RuntimeStreamPriority::High);
    }

    // Port of `RuntimeStreamUnittest.CreateMultipleStreams`.
    #[test]
    fn create_multiple_streams_have_independent_properties() {
        let scheduler = MockScheduler::new(true);
        let stream1 = RuntimeStream::new(
            scheduler.clone(),
            DeviceId(0),
            RuntimeStreamPriority::Normal,
            RuntimeStreamMode::StrictOrdering,
        );
        let stream2 = RuntimeStream::new(
            scheduler.clone(),
            DeviceId(0),
            RuntimeStreamPriority::Normal,
            RuntimeStreamMode::OpOrdering,
        );
        let stream3 = RuntimeStream::new(
            scheduler,
            DeviceId(0),
            RuntimeStreamPriority::High,
            RuntimeStreamMode::StrictOrdering,
        );

        assert_ne!(stream1.stream_id(), stream2.stream_id());
        assert_ne!(stream1.stream_id(), stream3.stream_id());
        assert_ne!(stream2.stream_id(), stream3.stream_id());

        assert_eq!(stream1.execution_mode(), RuntimeStreamMode::StrictOrdering);
        assert_eq!(stream1.priority(), RuntimeStreamPriority::Normal);

        assert_eq!(stream2.execution_mode(), RuntimeStreamMode::OpOrdering);
        assert_eq!(stream2.priority(), RuntimeStreamPriority::Normal);

        assert_eq!(stream3.execution_mode(), RuntimeStreamMode::StrictOrdering);
        assert_eq!(stream3.priority(), RuntimeStreamPriority::High);
    }

    // Adapted from `RuntimeStreamUnittest.StreamDeviceId`: the C++ test
    // checks `stream->deviceId() == context_->getDeviceID()`; with no
    // `RuntimeContext` here, this checks `RuntimeStream::new` faithfully
    // stores the `DeviceId` it was constructed with.
    #[test]
    fn stream_device_id_matches_constructor_arg() {
        let scheduler = MockScheduler::new(true);
        let stream = RuntimeStream::new(
            scheduler,
            DeviceId(7),
            RuntimeStreamPriority::default(),
            RuntimeStreamMode::default(),
        );

        assert_eq!(stream.device_id(), DeviceId(7));
    }

    // Adapted from `RuntimeStreamUnittest.ConcurrentStreamCreation`: the C++
    // test creates streams from many threads via `context_->createStream()`
    // and checks none are null; with no `RuntimeContext`, this creates
    // `RuntimeStream`s directly from many threads and checks `StreamId::next`
    // never hands out a duplicate under concurrent construction.
    #[test]
    fn concurrent_stream_creation_yields_unique_stream_ids() {
        let scheduler = MockScheduler::new(true);
        let num_threads = 4;
        let streams_per_thread = 10;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let scheduler = scheduler.clone();
                std::thread::spawn(move || {
                    let mut ids = Vec::with_capacity(streams_per_thread);
                    for _ in 0..streams_per_thread {
                        let stream = RuntimeStream::new(
                            scheduler.clone(),
                            DeviceId(0),
                            RuntimeStreamPriority::default(),
                            RuntimeStreamMode::default(),
                        );
                        ids.push(stream.stream_id());
                    }
                    ids
                })
            })
            .collect();

        let mut all_ids: Vec<StreamId> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        let total = all_ids.len();
        assert_eq!(total, num_threads * streams_per_thread);

        all_ids.sort();
        all_ids.dedup();
        assert_eq!(
            all_ids.len(),
            total,
            "StreamId::next() handed out a duplicate under concurrent creation"
        );
    }

    // Adapted from `RuntimeStreamComputeTest.ComputeWithHostCallback`: the
    // C++ test attaches a callback to `ComputeParams` and checks it fired
    // after `synchronize()`. This crate's `launch_compute` takes the user
    // callback directly (no concrete-params callback field), so this ports
    // the same behavior against `launch_operation` + a dummy payload,
    // exercising `make_completion_callback`'s "invoke user_cb on success"
    // path.
    #[test]
    fn compute_completion_invokes_user_callback() {
        let scheduler = MockScheduler::new(true);
        let stream = RuntimeStream::new(
            scheduler,
            DeviceId(0),
            RuntimeStreamPriority::default(),
            RuntimeStreamMode::default(),
        );

        let fired = Arc::new(AtomicBool::new(false));
        let fired_clone = fired.clone();
        let user_cb: Box<dyn FnOnce() + Send> =
            Box::new(move || fired_clone.store(true, Ordering::SeqCst));

        stream
            .launch_operation(dummy_op(RuntimeOperationKind::Compute), Some(user_cb))
            .unwrap();
        stream.synchronize().unwrap();

        assert!(
            fired.load(Ordering::SeqCst),
            "user callback was not invoked after compute completion"
        );
    }

    // Adapted from `RuntimeStreamDmaUnittest.StreamQueryDuringDma`: the C++
    // test only checks `query()` doesn't throw while an op is outstanding
    // (Rust's `query` is infallible so that isn't a meaningful port); this
    // instead checks the actual value `query()` reports across the
    // outstanding-then-completed lifecycle, using a non-auto-completing mock
    // to hold an operation "in flight" under test control.
    #[test]
    fn query_reflects_in_flight_operations() {
        let scheduler = MockScheduler::new(false);
        let stream = RuntimeStream::new(
            scheduler.clone(),
            DeviceId(0),
            RuntimeStreamPriority::default(),
            RuntimeStreamMode::default(),
        );

        assert!(
            stream.query(),
            "stream should be idle before any operation is launched"
        );

        stream
            .launch_operation(dummy_op(RuntimeOperationKind::H2D), None)
            .unwrap();
        assert!(
            !stream.query(),
            "stream should be busy while the operation is still in flight"
        );
        assert_eq!(scheduler.pending_count(), 1);

        scheduler.fire_next(Ok(()));
        stream.synchronize().unwrap();
        assert!(
            stream.query(),
            "stream should be idle again after synchronize()"
        );
    }

    // Port of `RuntimeStreamDmaUnittest.DmaCallbackFunctionality`.
    #[test]
    fn dma_callback_functionality() {
        let scheduler = MockScheduler::new(true);
        let stream = RuntimeStream::new(
            scheduler,
            DeviceId(0),
            RuntimeStreamPriority::default(),
            RuntimeStreamMode::default(),
        );

        let callback_count = Arc::new(AtomicUsize::new(0));
        let callback_count_clone = callback_count.clone();
        let user_cb: Box<dyn FnOnce() + Send> = Box::new(move || {
            callback_count_clone.fetch_add(1, Ordering::SeqCst);
        });

        stream
            .launch_operation(dummy_op(RuntimeOperationKind::H2D), Some(user_cb))
            .unwrap();
        stream.synchronize().unwrap();

        assert!(
            callback_count.load(Ordering::SeqCst) >= 1,
            "callback was not called"
        );
    }

    // Port of `RuntimeStreamDmaUnittest.DeferredErrorHandling`: launches
    // three operations whose completions each fail with a distinct error;
    // the first failure sets shutdown, so any subsequent `launch_operation`
    // call observes `needs_shutdown() == true` and is rejected up front with
    // `RuntimeStreamError::Shutdown` (matching the C++ `StreamInErrorState`
    // synchronous throw) rather than reaching the scheduler at all. Either
    // way, `set_error`'s first-error-wins rule means only "Error First"
    // survives to be re-thrown once, by `synchronize()`.
    #[test]
    fn deferred_error_first_error_wins() {
        let scheduler = MockScheduler::new(true);
        let stream = RuntimeStream::new(
            scheduler.clone(),
            DeviceId(0),
            RuntimeStreamPriority::default(),
            RuntimeStreamMode::default(),
        );

        scheduler.push_outcome(Err("Deferred Error First".to_string()));
        scheduler.push_outcome(Err("Deferred Error Second".to_string()));
        scheduler.push_outcome(Err("Deferred Error Third".to_string()));

        let mut rejected_after_shutdown = false;
        for _ in 0..3 {
            match stream.launch_operation(dummy_op(RuntimeOperationKind::H2D), None) {
                Ok(()) => {}
                Err(RuntimeStreamError::Shutdown) => rejected_after_shutdown = true,
                Err(e) => panic!("unexpected error launching operation: {e}"),
            }
        }

        // At least the first op always makes it to the (auto-completing) mock
        // scheduler and fails, setting shutdown; later launches then observe
        // needs_shutdown() == true and are synchronously rejected — matching
        // the C++ test's absorbed `StreamInErrorState` throws.
        assert!(stream.needs_shutdown());
        let _ = rejected_after_shutdown;

        match stream.synchronize() {
            Err(RuntimeStreamError::Deferred(inner)) => {
                assert_eq!(
                    inner.to_string(),
                    "scheduler rejected submission: Deferred Error First"
                );
            }
            other => {
                panic!("expected synchronize() to re-throw the first deferred error, got {other:?}")
            }
        }

        // The deferred error is consumed exactly once.
        assert!(stream.synchronize().is_ok());
    }
}
