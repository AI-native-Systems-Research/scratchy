//! Port of `flex::RuntimeContext`'s OWN state machine — singleton lifecycle,
//! stream table, and default-stream bookkeeping. Source:
//! `flex/include/flex/runtime_stream/runtime_context.hpp` +
//! `flex/src/runtime_stream/runtime_context.cpp`.
//!
//! Deliberately NOT ported here (out of scope for this phase, owned by other
//! subsystems): `DeviceInterfaceFactory`/`DeviceInterface` construction,
//! `FlexAllocator` *construction* (Foundation phase, `allocator.rs`), and
//! `RuntimeSchedulerFactory`/`RuntimeScheduler` construction (scheduler
//! phase, `crate::scheduler` — this file depends on it only through
//! `crate::stream::SchedulerHandle`). `RuntimeContext::create` in the real
//! C++ builds all three internally; here they are supplied by the caller
//! (Assembly wires the concrete scheduler/device/allocator together), which
//! keeps this file's job exactly what the task brief asks for: the
//! singleton/stream-table bookkeeping, not the subsystems it wires up.
//!
//! `RuntimeContext` does, however, port `getAllocator()`
//! (`runtime_context.cpp`'s `RuntimeContextImpl::getAllocator` /
//! `RuntimeContext::getAllocator`): once built elsewhere, the
//! `FlexAllocator` is handed to `RuntimeContext::create`/`build` and stored
//! alongside the scheduler for the lifetime of the context, exactly as
//! `RuntimeContextImpl::flex_allocator_` does relative to `scheduler_`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::allocator::FlexAllocator;
use crate::senlib_ffi_runtime;
use crate::stream::{
    DeviceId, RuntimeStream, RuntimeStreamMode, RuntimeStreamPriority, SchedulerHandle, StreamId,
};

#[derive(Debug)]
pub enum RuntimeContextError {
    /// Port of `RAS::RUNTIMECONTEXT::ContextAlreadyCreated`: `create()` was
    /// called with a specific device id while a context for a *different*
    /// device id already exists.
    AlreadyCreatedForDifferentDevice {
        existing: DeviceId,
        requested: DeviceId,
    },
    /// Port of `RAS::RUNTIMECONTEXT::ContextNotCreated`: any accessor called
    /// before `create()`, or after `reset()`.
    NotCreated,
    /// Port of the `destroyStream` null-handle / default-stream guards.
    DestroyStreamError(&'static str),
}

impl std::fmt::Display for RuntimeContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyCreatedForDifferentDevice {
                existing,
                requested,
            } => write!(
                f,
                "RuntimeContext already created for device {existing:?}, cannot recreate for {requested:?}"
            ),
            Self::NotCreated => write!(f, "RuntimeContext has not been created, or has been reset"),
            Self::DestroyStreamError(msg) => write!(f, "destroyStream: {msg}"),
        }
    }
}
impl std::error::Error for RuntimeContextError {}

/// Port of `senlib::v2::SenPci::ncards()` as used by
/// `detail::getNumDevicesImpl`/`parseSpyreDevices`
/// (`runtime_context.cpp:44,109`): the raw count of Spyre cards in the system,
/// obtainable without opening any device.
///
/// NOTE the memoization here is this port's own (the C++ calls `ncards()`
/// afresh each time; only the *result of `getNumDevicesImpl`* is memoized, by
/// `getNumDevices`'s `std::call_once` — see `get_num_devices` below). Kept
/// because `ncards()` is itself a stable property of the machine.
pub fn num_physical_cards() -> u64 {
    static CACHED: OnceLock<u64> = OnceLock::new();
    *CACHED.get_or_init(|| {
        // SAFETY: `flex_senlib_pci_ncards` takes no arguments and returns a
        // plain integer; no ownership crosses the FFI boundary.
        unsafe { senlib_ffi_runtime::flex_senlib_pci_ncards() }
    })
}

/// Port of the failure modes of `detail::getNumDevicesImpl`
/// (`runtime_context.cpp:93-141`).
#[derive(Debug)]
pub enum GetNumDevicesError {
    /// Port of `RAS::CONFIGURATION::InvalidAIUWorldSizeVar().AIUWorldSize(n).Throw()`
    /// (`runtime_context.cpp:99-102`): `FLEX_DEVICE=MOCK` with `AIU_WORLD_SIZE < 1`.
    InvalidAiuWorldSize(i64),
    /// Propagated from `RuntimeConfig::SpyreDevices()`, which
    /// `parseSpyreDevices` reads (`runtime_context.cpp:39`) and which throws
    /// `RAS::CONFIGURATION::IncorrectSpyreDevicesVar` when fewer device
    /// mappings are given than the world size needs.
    SpyreDevices(crate::flex_config::SpyreDevicesError),
}

impl std::fmt::Display for GetNumDevicesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidAiuWorldSize(n) => {
                write!(f, "invalid AIU_WORLD_SIZE for MOCK device: {n}")
            }
            Self::SpyreDevices(e) => write!(f, "SPYRE_DEVICES is invalid: {e}"),
        }
    }
}
impl std::error::Error for GetNumDevicesError {}

/// Port of `flex::detail::parseSpyreDevices()` (`runtime_context.cpp:36-91`):
/// take `RuntimeConfig::SpyreDevices().str()` — i.e. the comma-joined
/// rendering of the already-resolved setting, NOT a fresh `getenv` — and turn
/// it into a de-duplicated, ascending set of in-range device indices.
///
/// Every rejection path in the C++ is a `LOG_WARNING` + `continue`, never an
/// error: a non-numeric item, an item with trailing garbage (`from_chars`
/// must consume the whole token, unlike the `strtol` used by
/// `FlexConfig::FlexSpyreDevices`), an out-of-range index (`< 0` or
/// `>= ncards()`), and a duplicate are all silently dropped. An empty
/// setting yields an empty set without even calling `ncards()`.
pub fn parse_spyre_devices() -> Result<std::collections::BTreeSet<i64>, GetNumDevicesError> {
    let spyre_devices_str = crate::flex_config::RuntimeConfig::spyre_devices()
        .map_err(GetNumDevicesError::SpyreDevices)?
        .str();

    let mut devices = std::collections::BTreeSet::new();
    if spyre_devices_str.is_empty() {
        return Ok(devices);
    }

    let total_devices = num_physical_cards() as i64;

    for item in spyre_devices_str.split(',') {
        // Port of the explicit whitespace trim (`runtime_context.cpp:51-53`).
        let item = item.trim_matches(|c| c == ' ' || c == '\t' || c == '\n' || c == '\r');
        if item.is_empty() {
            continue;
        }
        // Port of `std::from_chars` + the `ptr != end` check: the entire token
        // must parse, so `"3x"` is rejected here (it would be accepted as `3`
        // by `FlexConfig::FlexSpyreDevices`'s `strtol`).
        let Ok(device_idx) = item.parse::<i64>() else {
            tracing::warn!("Invalid device index: {item}");
            continue;
        };
        if device_idx < 0 || device_idx >= total_devices {
            tracing::warn!("Device index out of bounds: {item}");
            continue;
        }
        if !devices.insert(device_idx) {
            tracing::warn!("Duplicate device index {item}");
        }
    }
    Ok(devices)
}

/// Port of `flex::detail::getNumDevicesImpl()` (`runtime_context.cpp:93-141`) —
/// the resolution order is load-bearing:
///
/// 1. `FLEX_DEVICE == "MOCK"` (via `RuntimeConfig`, so config-file-resolvable):
///    `AIU_WORLD_SIZE` is authoritative and must be `>= 1`, else throw.
/// 2. Otherwise, if `ncards() == 0`: warn and return 0.
/// 3. Otherwise, if `SPYRE_DEVICES` names at least one valid device: that count wins.
/// 4. Otherwise fall back to `AIU_WORLD_SIZE`, clamped to `ncards()` when it is
///    unset/`< 1` or larger than the real card count.
fn get_num_devices_impl() -> Result<i64, GetNumDevicesError> {
    let aiu_world_size = crate::flex_config::FlexConfig::aiu_world_size().0;

    if crate::flex_config::RuntimeConfig::flex_device().value() == "MOCK" {
        if aiu_world_size < 1 {
            return Err(GetNumDevicesError::InvalidAiuWorldSize(aiu_world_size));
        }
        tracing::info!("getNumDevicesImpl(): {aiu_world_size} devices available");
        return Ok(aiu_world_size);
    }

    let total_devices = num_physical_cards() as i64;
    if total_devices == 0 {
        tracing::warn!("No available device");
        return Ok(0);
    }

    let num_devices = parse_spyre_devices()?.len() as i64;
    if num_devices > 0 {
        tracing::info!("getNumDevicesImpl(): {num_devices} devices available");
        return Ok(num_devices);
    }

    let mut aiu_world_size = aiu_world_size;
    if aiu_world_size > total_devices || aiu_world_size < 1 {
        tracing::warn!(
            "AIU_WORLD_SIZE {aiu_world_size} does not match the total number of devices {total_devices}, or unset. Use the total number of devices."
        );
        aiu_world_size = total_devices;
    } else if aiu_world_size < total_devices {
        tracing::warn!(
            "AIU_WORLD_SIZE {aiu_world_size} does not match the total number of devices {total_devices}"
        );
    }

    tracing::info!("getNumDevicesImpl(): {aiu_world_size} devices available");
    Ok(aiu_world_size)
}

/// Port of `flex::getNumDevices()` (`runtime_context.cpp:227-235`): the number
/// of Spyre devices usable by this process, computed exactly once
/// (`std::call_once`) and cached for the process lifetime.
///
/// PORT GAP FIXED: this used to be nothing but `num_physical_cards()` — the raw
/// `ncards()` — with a comment claiming the `SPYRE_DEVICES`/`AIU_WORLD_SIZE`
/// resolution was "out of scope". It is not out of scope: it is the entire body
/// of `getNumDevicesImpl`, it lives in `runtime_context.cpp`, and skipping it
/// silently changes the answer for every MOCK run (where `ncards()` is
/// irrelevant and `AIU_WORLD_SIZE` decides) and for every run that restricts
/// itself with `SPYRE_DEVICES`.
///
/// Only a successful result is memoized: a failing `getNumDevicesImpl` in the
/// C++ throws out of the `std::call_once` lambda, which leaves the once-flag
/// unset so a later call retries.
pub fn get_num_devices() -> Result<i64, GetNumDevicesError> {
    static CACHED: OnceLock<Mutex<Option<i64>>> = OnceLock::new();
    let slot = CACHED.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().unwrap();
    if let Some(n) = *guard {
        return Ok(n);
    }
    let n = get_num_devices_impl()?;
    *guard = Some(n);
    Ok(n)
}

/// Port of `flex::RuntimeContext`. Streams are owned exclusively by the
/// context, matching the C++ `unordered_map<RuntimeStream*, unique_ptr<...>>`
/// ownership model; callers hold `StreamId`s (the safe-Rust equivalent of a
/// non-owning `RuntimeStream*` that can't dangle after `destroy_stream`,
/// since lookups go back through the context rather than through a raw
/// pointer).
pub struct RuntimeContext {
    /// Port of `RuntimeContextImpl::device_id_`: the raw value supplied to
    /// `create()` (-1 if the caller let flex pick the device, matching the
    /// C++ default `device_id = -1`). Used, as in the C++, only for
    /// constructing new `RuntimeStream`s (`createStream`,
    /// `runtime_context.cpp:446`) — NOT for the already-created-for-a-
    /// different-device check, which the real C++ compares against the
    /// hardware-resolved id instead (see `resolved_device_id`).
    device_id: DeviceId,
    /// Port of what `RuntimeContextImpl::getDevice()->getDeviceID()`
    /// (`runtime_context.cpp:373-379`) resolves to once the real
    /// `DeviceInterface` has bound to a concrete device: this is the value
    /// `create()`'s already-created-for-a-different-device check
    /// (`runtime_context.cpp:396-416`) and `getDeviceID()`
    /// (`runtime_context.cpp:373-379`) actually use — NOT the raw
    /// constructor argument. This crate has no `DeviceInterfaceFactory`/
    /// `DeviceInterface` (out of scope, see module doc), so there is no real
    /// hardware to query: when the caller requests a specific device id,
    /// resolution is exact (`Some(id) => id`, mirroring
    /// `DeviceInterfaceFactory::createDevice(device_id)` opening exactly
    /// that device); when the caller leaves it unspecified (`None`/-1,
    /// "flex sets the device index itself"), this stands in with
    /// `DeviceId(0)` — the lowest-numbered/default device — since genuine
    /// auto-selection would require the unported `DeviceInterfaceFactory`.
    resolved_device_id: DeviceId,
    /// Port of the "has this `RuntimeContext*` been `reset()`" check every
    /// real C++ accessor performs (`prtc_impl_ == nullptr` ->
    /// `RAS::RUNTIMECONTEXT::ContextNotCreated().Throw()`,
    /// `runtime_context.cpp:330-334,342-346,447-452`, etc). Rust's `reset()`
    /// clears the global singleton slot, but a caller may still be holding
    /// an `Arc<RuntimeContext>` clone from before the reset; without this
    /// flag such a stale handle would silently keep working forever.
    active: AtomicBool,
    scheduler: Arc<dyn SchedulerHandle>,
    /// Port of `RuntimeContextImpl::flex_allocator_`. Shared (rather than
    /// exclusively owned) since `getAllocator()` in the C++ hands out a
    /// `shared_ptr<FlexAllocator>` that outlives any single caller; wrapped
    /// in a `Mutex` because `FlexAllocator`'s own `allocate`/`deallocate`
    /// take `&mut self`, unlike the C++ type which synchronizes internally.
    allocator: Arc<Mutex<FlexAllocator>>,
    streams: Mutex<HashMap<StreamId, Arc<RuntimeStream>>>,
    default_stream_id: StreamId,
}

static INSTANCE: OnceLock<Mutex<Option<Arc<RuntimeContext>>>> = OnceLock::new();

fn instance_slot() -> &'static Mutex<Option<Arc<RuntimeContext>>> {
    INSTANCE.get_or_init(|| Mutex::new(None))
}

impl RuntimeContext {
    fn build(
        device_id: DeviceId,
        resolved_device_id: DeviceId,
        scheduler: Arc<dyn SchedulerHandle>,
        allocator: Arc<Mutex<FlexAllocator>>,
    ) -> Arc<Self> {
        // Port of `RuntimeContextImpl`'s construction of the default stream,
        // which "lives for the lifetime of the context and cannot be
        // destroyed via destroyStream()".
        let default_stream = RuntimeStream::new(
            scheduler.clone(),
            device_id,
            RuntimeStreamPriority::Normal,
            RuntimeStreamMode::StrictOrdering,
        );
        let default_stream_id = default_stream.stream_id();
        let mut streams = HashMap::new();
        streams.insert(default_stream_id, default_stream);

        Arc::new(Self {
            device_id,
            resolved_device_id,
            active: AtomicBool::new(true),
            scheduler,
            allocator,
            streams: Mutex::new(streams),
            default_stream_id,
        })
    }

    /// Port of `RuntimeContext::create(device_id, alloc_init)`. Unlike the
    /// real C++ (which builds the `FlexAllocator` internally from
    /// `alloc_init`), the already-constructed allocator is supplied by the
    /// caller — see the module doc comment on why `FlexAllocator`
    /// construction itself stays out of scope here — but it is stored on the
    /// context exactly as `RuntimeContextImpl` stores `flex_allocator_`,
    /// reachable afterward via `get_allocator()`.
    /// `device_id = None` mirrors the C++ default `device_id = -1`
    /// ("flex sets the device index itself").
    ///
    /// Idempotent like the C++: if a context already exists for the same (or
    /// unspecified) device id, returns the existing singleton rather than
    /// rebuilding it (the supplied `allocator` is ignored in that case, same
    /// as the C++ discarding a redundant `alloc_init`).
    pub fn create(
        device_id: Option<DeviceId>,
        scheduler: Arc<dyn SchedulerHandle>,
        allocator: Arc<Mutex<FlexAllocator>>,
    ) -> Result<Arc<Self>, RuntimeContextError> {
        let mut slot = instance_slot().lock().unwrap();
        if let Some(existing) = slot.as_ref() {
            // Port of `runtime_context.cpp:396-398`'s
            // `device_id != -1 && device_id != rtc->prtc_impl_->getDevice()->getDeviceID()`:
            // compares the requested id against the hardware-*resolved* id,
            // not the raw construction-time argument (which would stay -1
            // forever for a default-created context and spuriously reject
            // every later `create(Some(id))` call).
            if let Some(requested) = device_id
                && requested != existing.resolved_device_id
            {
                return Err(RuntimeContextError::AlreadyCreatedForDifferentDevice {
                    existing: existing.resolved_device_id,
                    requested,
                });
            }
            return Ok(existing.clone());
        }
        let raw_device_id = device_id.unwrap_or(DeviceId(-1));
        let resolved_device_id = device_id.unwrap_or(DeviceId(0));
        let ctx = Self::build(raw_device_id, resolved_device_id, scheduler, allocator);
        // Port of `runtime_context.cpp:407-409`'s `LOG_INFO << "Logical Device ID: "`.
        // The same C++ block also touches `aiupti::GlobalProfiler()` for its side
        // effect only — forcing that singleton onto the creating thread rather
        // than lazily onto whichever worker first emits an activity. That is not
        // ported: `aiupti::Profiler` belongs to `libaiupti.so`, a separate
        // library this crate does not reimplement, so there is no singleton here
        // to force. If libaiupti is ever bound over FFI, this is where its
        // eager-construction call belongs.
        tracing::info!("Logical Device ID: {raw_device_id:?}");
        *slot = Some(ctx.clone());
        Ok(ctx)
    }

    /// Port of `RuntimeContext::getInstance`.
    pub fn get_instance() -> Result<Arc<Self>, RuntimeContextError> {
        instance_slot()
            .lock()
            .unwrap()
            .clone()
            .ok_or(RuntimeContextError::NotCreated)
    }

    /// Port of `RuntimeContext::reset`. Note the C++ comment's deadlock
    /// warning is preserved structurally: this must never be called from
    /// within another `RuntimeContext` method.
    pub fn reset() -> Result<(), RuntimeContextError> {
        let mut slot = instance_slot().lock().unwrap();
        let Some(ctx) = slot.take() else {
            return Err(RuntimeContextError::NotCreated);
        };
        // Port of `RuntimeContextImpl::~RuntimeContextImpl`'s unconditional
        // per-stream drain before the stream map is cleared
        // (`runtime_context.cpp:263-288`): every registered stream is
        // `synchronize()`d, with any deferred error logged and swallowed —
        // never propagated out of `reset()` — exactly like
        // `destroy_stream`'s existing suppress-and-log pattern below.
        {
            let streams = ctx.streams.lock().unwrap();
            for (id, stream) in streams.iter() {
                if let Err(e) = stream.synchronize() {
                    tracing::error!(
                        "RuntimeContext::reset: deferred error suppressed while draining stream {id:?} during teardown: {e}"
                    );
                }
            }
        }
        // Mark this instance inactive so any `Arc<RuntimeContext>` clone a
        // caller is still holding past this point observes the reset
        // (port of every accessor's `prtc_impl_ == nullptr` check —
        // `runtime_context.cpp:330-334` etc — which must keep tripping even
        // though Rust drops `prtc_impl_`'s equivalent state by clearing the
        // singleton slot rather than mutating the shared object in place).
        ctx.active.store(false, Ordering::SeqCst);
        drop(slot);
        Ok(())
    }

    /// Returns `Err(NotCreated)` if this context has been `reset()`, mirroring
    /// every real C++ accessor's `prtc_impl_ == nullptr` ->
    /// `RAS::RUNTIMECONTEXT::ContextNotCreated().Throw()` guard.
    fn check_active(&self) -> Result<(), RuntimeContextError> {
        if self.active.load(Ordering::SeqCst) {
            Ok(())
        } else {
            Err(RuntimeContextError::NotCreated)
        }
    }

    /// Port of `RuntimeContext::getDeviceID` (`runtime_context.cpp:373-379`):
    /// returns the hardware-resolved device id, not the raw constructor
    /// argument — see the `resolved_device_id` field doc comment.
    pub fn device_id(&self) -> Result<DeviceId, RuntimeContextError> {
        self.check_active()?;
        Ok(self.resolved_device_id)
    }

    /// Port of `RuntimeContext::getAllocator`. Returns a shared handle to the
    /// context's `FlexAllocator` rather than the C++'s `shared_ptr`, since
    /// `FlexAllocator`'s allocate/deallocate need `&mut` access — see the
    /// field doc comment on `allocator`.
    pub fn get_allocator(&self) -> Result<Arc<Mutex<FlexAllocator>>, RuntimeContextError> {
        self.check_active()?;
        Ok(self.allocator.clone())
    }

    /// Port of `RuntimeContext::createStream`.
    pub fn create_stream(
        &self,
        priority: RuntimeStreamPriority,
        mode: RuntimeStreamMode,
    ) -> Result<StreamId, RuntimeContextError> {
        self.check_active()?;
        let stream = RuntimeStream::new(self.scheduler.clone(), self.device_id, priority, mode);
        let id = stream.stream_id();
        self.streams.lock().unwrap().insert(id, stream);
        Ok(id)
    }

    /// Non-owning lookup — the safe-Rust equivalent of receiving a
    /// `RuntimeStream*` from `createStream`/`getDefaultStream`.
    pub fn get_stream(
        &self,
        id: StreamId,
    ) -> Result<Option<Arc<RuntimeStream>>, RuntimeContextError> {
        self.check_active()?;
        Ok(self.streams.lock().unwrap().get(&id).cloned())
    }

    /// Port of `RuntimeContext::destroyStream`: refuses to destroy the
    /// default stream, then blocks (via `synchronize`, called on drop of the
    /// removed `Arc` once its refcount reaches zero — matching the C++
    /// contract that outstanding operations run to completion) until the
    /// stream reaches synchronization.
    pub fn destroy_stream(&self, id: StreamId) -> Result<(), RuntimeContextError> {
        self.check_active()?;
        if id == self.default_stream_id {
            return Err(RuntimeContextError::DestroyStreamError(
                "Cannot destroy the default stream",
            ));
        }
        // INVENTED BEHAVIOR REMOVED: this previously returned
        // `Err(UnknownStream)` for an id that isn't in the table. The real
        // `destroyStream` (`runtime_context.cpp:461-479`) validates only two
        // things — null handle and default stream — and then does a bare
        // `prtc_impl_->streams_.erase(handle)`, which is a silent no-op for a
        // handle that isn't present (e.g. a double destroy). There is no
        // "unknown handle" error anywhere in the C++.
        let Some(stream) = self.streams.lock().unwrap().remove(&id) else {
            return Ok(());
        };
        // Explicit synchronize (matching the C++ blocking contract) rather
        // than relying solely on `Drop::drop`'s best-effort drain, since
        // other `Arc<RuntimeStream>` clones (e.g. held by in-flight
        // completion callbacks) may keep the stream alive past this point.
        //
        // Matches `~RuntimeStream`'s try/catch (invoked when C++'s
        // `streams_.erase(handle)` drops the owning `unique_ptr`): any
        // deferred error is logged and swallowed, never propagated to the
        // caller of destroyStream()/destroy_stream(). The previous version
        // here mapped a real deferred error (e.g. a DMA failure) into a
        // misleading `UnknownStream` error and surfaced it to the caller,
        // which the real C++ never does.
        if let Err(e) = stream.synchronize() {
            tracing::warn!(
                "RuntimeContext::destroy_stream: deferred error suppressed while draining stream {id:?}: {e}"
            );
        }
        Ok(())
    }

    /// Port of `RuntimeContext::getDefaultStream`.
    pub fn default_stream_id(&self) -> Result<StreamId, RuntimeContextError> {
        self.check_active()?;
        Ok(self.default_stream_id)
    }

    /// Port of `RuntimeContext::hasStreamError`: true if any registered
    /// stream's shutdown flag is set.
    pub fn has_stream_error(&self) -> Result<bool, RuntimeContextError> {
        self.check_active()?;
        Ok(self
            .streams
            .lock()
            .unwrap()
            .values()
            .any(|s| s.needs_shutdown()))
    }
}

/// Ported from the real C++ suite
/// `flex/tests/runtime_stream/context/{get_device_info_test,
/// runtime_context_stream_error_test,runtime_context_test}.cpp` (27 cases
/// total).
///
/// - `get_device_info_test.cpp` (11 `TEST_F` cases): SKIPPED IN FULL. Every
///   case exercises `RuntimeContext::create()`'s real device handle plus
///   `DeviceInfo`/`QueryInfoInterface()` — an actual hardware/firmware query.
///   Neither `DeviceInfo`, `QueryInfoInterface`, nor a device handle exists
///   anywhere in this crate (confirmed: `grep -rl
///   "DeviceInfo|QueryInfoInterface|GetDeviceHandle" src/` finds nothing), so
///   there is no port to exercise; fabricating one here would test
///   Rust-invented behavior with no C++ counterpart, which the audit
///   explicitly forbids.
/// - `runtime_context_test.cpp` (9 cases): 7 ported below (2 of them —
///   `GetAllocatorDefaultInit`/`GetAllocatorUserInit` — adapted, see their
///   doc comments); 2 SKIPPED:
///   - `GetDefaultDeviceHandle`: needs `getDeviceHandle()`/`DeviceInterface`,
///     not ported (see the module doc comment's scope list).
///   - `AllocatorAllocMemory`: calls `FlexAllocator::allocate` against a
///     *real* device-backed region, which requires an actual senlib
///     allocation to succeed (`allocator.rs::pre_allocate_regions` ->
///     `allocate_new_region` -> `DeviceMemoryAllocator::try_allocate` ->
///     `senlib_ffi_allocator::flex_senlib_memory_allocate`). This crate has
///     no mock senlib seam (see `device_memory_allocator.rs`'s own comment
///     on why its 6 `TryAllocate` tests aren't ported either) — hardware is
///     genuinely required for a `FlexAllocator` with any real *non-empty*
///     region to exist, so this stays untestable here for the same reason.
/// - `runtime_context_stream_error_test.cpp` (7 cases): all 7 ported — none
///   of them touch `FlexAllocator`/hardware at all, only
///   `RuntimeContext`'s + `RuntimeStream`'s own bookkeeping against a mock
///   `SchedulerHandle`.
#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;
    use std::sync::atomic::AtomicBool;

    use super::*;
    use crate::allocator::FlexAllocatorInit;
    use crate::device_memory_allocator::DeviceMemoryAllocator;
    use crate::stream::{CompletionCallback, SchedulerError, SubmittedOperation};
    use crate::topology::DeviceTopology;

    /// Serializes every test below against the process-wide `RuntimeContext`
    /// singleton (`INSTANCE`), which — unlike gtest's serial-by-default
    /// `TEST_F` execution — is not safe under Rust's parallel-by-default test
    /// runner: two tests racing `create`/`reset` concurrently would flake in
    /// ways the real C++ suite never has to guard against.
    static SINGLETON_GUARD: StdMutex<()> = StdMutex::new(());

    /// Trivial `SchedulerHandle` double: good enough for every test below,
    /// none of which inspect scheduling behavior itself (that's
    /// `stream.rs`'s own `MockScheduler`'s job) — they only need something
    /// that satisfies `RuntimeStream::new`'s `register_stream` call and, if a
    /// `launch_operation` ever runs, completes immediately.
    struct NullScheduler;
    impl SchedulerHandle for NullScheduler {
        fn register_stream(&self, _stream_id: StreamId, _shutdown_flag: Arc<AtomicBool>) {}
        fn release_stream(&self, _stream_id: StreamId) {}
        fn schedule(
            &self,
            _stream_id: StreamId,
            _mode: RuntimeStreamMode,
            _op: SubmittedOperation,
            completion: CompletionCallback,
        ) -> Result<(), SchedulerError> {
            completion(Ok(()));
            Ok(())
        }
    }

    fn test_scheduler() -> Arc<dyn SchedulerHandle> {
        Arc::new(NullScheduler)
    }

    /// A `FlexAllocator` buildable with zero senlib/hardware involvement: an
    /// empty-domain `DeviceTopology` makes `preAllocateRegions`'s per-domain
    /// region loop (`allocator.rs::pre_allocate_regions`) iterate zero times,
    /// so `allocate_new_region` — the one function on this path that reaches
    /// real senlib — is never called. Stands in for the real C++ tests'
    /// actual device-backed `FlexAllocator`, which this crate cannot
    /// construct in a unit test at all (see the `mod tests` doc comment
    /// above on `AllocatorAllocMemory`).
    fn test_allocator() -> Arc<Mutex<FlexAllocator>> {
        let topo = DeviceTopology::new(vec![]);
        let init = FlexAllocatorInit {
            max_regions: 1,
            num_program_regions: 0,
            ..Default::default()
        };
        Arc::new(Mutex::new(
            FlexAllocator::new(init, topo, DeviceMemoryAllocator::new(1)).unwrap(),
        ))
    }

    /// Runs `body` with the `RuntimeContext` singleton guaranteed empty
    /// beforehand and cleaned up afterward, holding `SINGLETON_GUARD` for the
    /// duration — the Rust analogue of the C++ fixtures'
    /// `SetUp`/`TearDown` pair (`TearDown` swallows `reset()`'s
    /// "not created" error exactly like the C++ `catch(std::runtime_error&)`).
    fn with_clean_context<F: FnOnce()>(body: F) {
        let _guard = SINGLETON_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let _ = RuntimeContext::reset();
        body();
        let _ = RuntimeContext::reset();
    }

    // Port of `RuntimeContextUnittest.NoDevice`.
    #[test]
    fn get_instance_before_create_errors() {
        with_clean_context(|| {
            assert!(matches!(
                RuntimeContext::get_instance(),
                Err(RuntimeContextError::NotCreated)
            ));
        });
    }

    // Port of `RuntimeContextUnittest.DefaultDevice`: the real test just
    // checks `create()`+`getDevice()` don't throw; this checks the resulting
    // context's `getDeviceID()`-equivalent reports the hardware-resolved
    // device id — this crate has no real `DeviceInterfaceFactory` to resolve
    // against, so a default-created context (`None`/-1, "flex sets the
    // device index itself") resolves to the stand-in default `DeviceId(0)`
    // rather than staying at the -1 sentinel forever (see
    // `resolved_device_id`'s field doc comment).
    #[test]
    fn create_default_device_succeeds() {
        with_clean_context(|| {
            let ctx = RuntimeContext::create(None, test_scheduler(), test_allocator()).unwrap();
            assert_eq!(ctx.device_id().unwrap(), DeviceId(0));
        });
    }

    // Port of `RuntimeContextUnittest.UserSpecifiedDevice`.
    #[test]
    fn create_with_user_specified_device_succeeds() {
        with_clean_context(|| {
            let ctx = RuntimeContext::create(Some(DeviceId(0)), test_scheduler(), test_allocator())
                .unwrap();
            assert_eq!(ctx.device_id().unwrap(), DeviceId(0));
        });
    }

    // Port of `RuntimeContextUnittest.ReallocateDevice`.
    #[test]
    fn reallocate_same_device_returns_existing_reallocate_different_device_errors() {
        with_clean_context(|| {
            let ctx1 =
                RuntimeContext::create(Some(DeviceId(0)), test_scheduler(), test_allocator())
                    .unwrap();
            let ctx2 =
                RuntimeContext::create(Some(DeviceId(0)), test_scheduler(), test_allocator())
                    .unwrap();
            assert!(
                Arc::ptr_eq(&ctx1, &ctx2),
                "create() with the same device id must return the existing context"
            );

            let _ = RuntimeContext::get_instance().unwrap();

            match RuntimeContext::create(Some(DeviceId(1)), test_scheduler(), test_allocator()) {
                Err(RuntimeContextError::AlreadyCreatedForDifferentDevice {
                    existing,
                    requested,
                }) => {
                    assert_eq!(existing, DeviceId(0));
                    assert_eq!(requested, DeviceId(1));
                }
                Ok(_) => panic!("expected AlreadyCreatedForDifferentDevice, got Ok"),
                Err(other) => panic!("expected AlreadyCreatedForDifferentDevice, got {other:?}"),
            }
        });
    }

    // Regression test for the bug where `create(None)` followed by
    // `create(Some(DeviceId(0)))` always raised
    // `AlreadyCreatedForDifferentDevice`: the mismatch check must compare
    // against the *resolved* device id (which a `None`-created context
    // resolves to `DeviceId(0)`, see `resolved_device_id`), not the raw -1
    // sentinel stored for stream construction.
    #[test]
    fn create_default_then_matching_explicit_device_succeeds() {
        with_clean_context(|| {
            let ctx1 = RuntimeContext::create(None, test_scheduler(), test_allocator()).unwrap();
            let ctx2 =
                RuntimeContext::create(Some(DeviceId(0)), test_scheduler(), test_allocator())
                    .unwrap();
            assert!(
                Arc::ptr_eq(&ctx1, &ctx2),
                "create(Some(0)) after create(None) must return the existing context, not error"
            );
        });
    }

    // Port of `RuntimeContextUnittest.ResetContext`.
    #[test]
    fn reset_removes_context_and_double_reset_errors() {
        with_clean_context(|| {
            RuntimeContext::create(Some(DeviceId(0)), test_scheduler(), test_allocator()).unwrap();
            RuntimeContext::reset().unwrap();
            assert!(matches!(
                RuntimeContext::get_instance(),
                Err(RuntimeContextError::NotCreated)
            ));
            assert!(matches!(
                RuntimeContext::reset(),
                Err(RuntimeContextError::NotCreated)
            ));
        });
    }

    // Regression test: every real C++ accessor throws `ContextNotCreated`
    // once `reset()` has run (`runtime_context.cpp:330-334` etc), even when
    // called through a handle obtained before the reset. A held
    // `Arc<RuntimeContext>` must observe this too, not silently keep working.
    #[test]
    fn stale_handle_errors_after_reset() {
        with_clean_context(|| {
            let ctx = RuntimeContext::create(Some(DeviceId(0)), test_scheduler(), test_allocator())
                .unwrap();
            let default_id = ctx.default_stream_id().unwrap();
            RuntimeContext::reset().unwrap();

            assert!(matches!(
                ctx.device_id(),
                Err(RuntimeContextError::NotCreated)
            ));
            assert!(matches!(
                ctx.get_allocator(),
                Err(RuntimeContextError::NotCreated)
            ));
            assert!(matches!(
                ctx.create_stream(
                    RuntimeStreamPriority::default(),
                    RuntimeStreamMode::default()
                ),
                Err(RuntimeContextError::NotCreated)
            ));
            assert!(matches!(
                ctx.get_stream(default_id),
                Err(RuntimeContextError::NotCreated)
            ));
            assert!(matches!(
                ctx.destroy_stream(default_id),
                Err(RuntimeContextError::NotCreated)
            ));
            assert!(matches!(
                ctx.default_stream_id(),
                Err(RuntimeContextError::NotCreated)
            ));
            assert!(matches!(
                ctx.has_stream_error(),
                Err(RuntimeContextError::NotCreated)
            ));
        });
    }

    // Adapted from `RuntimeContextUnittest.GetAllocatorDefaultInit`: the real
    // test exercises a real device-backed `FlexAllocator`'s topology (needs
    // hardware — see `AllocatorAllocMemory`'s skip note above). The part that
    // is actually `RuntimeContext`'s own job — storing whatever
    // `FlexAllocator` the caller hands to `create()` and returning that SAME
    // instance from `get_allocator()` — is what's ported here.
    #[test]
    fn get_allocator_returns_the_instance_supplied_at_create() {
        with_clean_context(|| {
            let allocator = test_allocator();
            let ctx =
                RuntimeContext::create(Some(DeviceId(0)), test_scheduler(), allocator.clone())
                    .unwrap();
            assert!(Arc::ptr_eq(&ctx.get_allocator().unwrap(), &allocator));
        });
    }

    // Adapted from `RuntimeContextUnittest.GetAllocatorUserInit`: the real
    // test constructs the `FlexAllocator` from a user-supplied
    // `FlexAllocatorInit` (that construction itself is `FlexAllocator`'s own
    // concern, not `RuntimeContext`'s — see the module doc comment on why
    // `FlexAllocator` construction is out of `RuntimeContext`'s scope here).
    // What remains genuinely `RuntimeContext`'s: `create()` on an
    // already-existing context discards a redundant caller-supplied
    // allocator rather than swapping it in, matching the C++ "idempotent
    // create ignores a redundant alloc_init" contract.
    #[test]
    fn create_ignores_a_redundant_allocator_on_idempotent_create() {
        with_clean_context(|| {
            let first = test_allocator();
            let ctx1 =
                RuntimeContext::create(Some(DeviceId(0)), test_scheduler(), first.clone()).unwrap();
            let second = test_allocator();
            let ctx2 = RuntimeContext::create(Some(DeviceId(0)), test_scheduler(), second).unwrap();
            assert!(Arc::ptr_eq(&ctx1, &ctx2));
            assert!(
                Arc::ptr_eq(&ctx1.get_allocator().unwrap(), &first),
                "the redundant second allocator must be discarded"
            );
        });
    }

    // Port of `StreamErrorIsolationTest.CleanContextReportsNoError`.
    #[test]
    fn clean_context_reports_no_stream_error() {
        with_clean_context(|| {
            let ctx = RuntimeContext::create(None, test_scheduler(), test_allocator()).unwrap();
            assert!(!ctx.has_stream_error().unwrap());
        });
    }

    // Port of `StreamErrorIsolationTest.NewStreamReportsNoError`.
    #[test]
    fn new_stream_reports_no_error() {
        with_clean_context(|| {
            let ctx = RuntimeContext::create(None, test_scheduler(), test_allocator()).unwrap();
            let id = ctx
                .create_stream(
                    RuntimeStreamPriority::default(),
                    RuntimeStreamMode::default(),
                )
                .unwrap();
            let stream = ctx.get_stream(id).unwrap().unwrap();
            assert!(!ctx.has_stream_error().unwrap());
            assert!(!stream.needs_shutdown());
        });
    }

    // Port of `StreamErrorIsolationTest.ErrorStateDetectedAfterSetShutdown`.
    #[test]
    fn error_state_detected_after_set_shutdown() {
        with_clean_context(|| {
            let ctx = RuntimeContext::create(None, test_scheduler(), test_allocator()).unwrap();
            let id = ctx
                .create_stream(
                    RuntimeStreamPriority::default(),
                    RuntimeStreamMode::default(),
                )
                .unwrap();
            let stream = ctx.get_stream(id).unwrap().unwrap();
            assert!(!ctx.has_stream_error().unwrap());

            stream.set_shutdown(true);

            assert!(ctx.has_stream_error().unwrap());
            assert!(stream.needs_shutdown());
        });
    }

    // Port of `StreamErrorIsolationTest.ErrorClearsAfterBrokenStreamDestroyed`.
    #[test]
    fn error_clears_after_broken_stream_destroyed() {
        with_clean_context(|| {
            let ctx = RuntimeContext::create(None, test_scheduler(), test_allocator()).unwrap();
            let id = ctx
                .create_stream(
                    RuntimeStreamPriority::default(),
                    RuntimeStreamMode::default(),
                )
                .unwrap();
            let stream = ctx.get_stream(id).unwrap().unwrap();
            stream.set_shutdown(true);
            assert!(ctx.has_stream_error().unwrap());

            ctx.destroy_stream(id).unwrap();

            assert!(!ctx.has_stream_error().unwrap());
        });
    }

    // Port of `StreamErrorIsolationTest.CreateStreamSucceedsAfterPriorStreamError`.
    #[test]
    fn create_stream_succeeds_after_prior_stream_error() {
        with_clean_context(|| {
            let ctx = RuntimeContext::create(None, test_scheduler(), test_allocator()).unwrap();
            let broken_id = ctx
                .create_stream(
                    RuntimeStreamPriority::default(),
                    RuntimeStreamMode::default(),
                )
                .unwrap();
            let broken = ctx.get_stream(broken_id).unwrap().unwrap();
            broken.set_shutdown(true);
            ctx.destroy_stream(broken_id).unwrap();

            let fresh_id = ctx
                .create_stream(
                    RuntimeStreamPriority::default(),
                    RuntimeStreamMode::default(),
                )
                .unwrap();
            let fresh = ctx.get_stream(fresh_id).unwrap().unwrap();
            assert!(!fresh.needs_shutdown());
            assert!(!ctx.has_stream_error().unwrap());
        });
    }

    // Port of `StreamErrorIsolationTest.SynchronizeRethrowsDeferredException`.
    #[test]
    fn synchronize_rethrows_deferred_error() {
        with_clean_context(|| {
            let ctx = RuntimeContext::create(None, test_scheduler(), test_allocator()).unwrap();
            let id = ctx
                .create_stream(
                    RuntimeStreamPriority::default(),
                    RuntimeStreamMode::default(),
                )
                .unwrap();
            let stream = ctx.get_stream(id).unwrap().unwrap();

            let error_msg = "simulated hardware fault";
            stream.set_error(crate::stream::RuntimeStreamError::Scheduler(
                error_msg.into(),
            ));
            stream.set_shutdown(true);

            match stream.synchronize() {
                Err(crate::stream::RuntimeStreamError::Deferred(inner)) => {
                    assert!(
                        inner.to_string().contains(error_msg),
                        "original error message not preserved: {inner}"
                    );
                }
                other => {
                    panic!("expected synchronize() to rethrow the deferred error, got {other:?}")
                }
            }
        });
    }

    // Port of `StreamErrorIsolationTest.DefaultStreamErrorDetectedByHasStreamError`.
    #[test]
    fn default_stream_error_detected_by_has_stream_error() {
        with_clean_context(|| {
            let ctx = RuntimeContext::create(None, test_scheduler(), test_allocator()).unwrap();
            let default_stream = ctx
                .get_stream(ctx.default_stream_id().unwrap())
                .unwrap()
                .unwrap();
            assert!(!ctx.has_stream_error().unwrap());

            default_stream.set_shutdown(true);
            assert!(ctx.has_stream_error().unwrap());

            default_stream.set_shutdown(false);
        });
    }
}
