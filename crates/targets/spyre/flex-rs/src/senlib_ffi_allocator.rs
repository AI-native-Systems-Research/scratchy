//! Foundation-phase senlib FFI surface: the allocator subsystem's slice of the
//! senlib boundary. Per-phase file (see task brief) to avoid symbol/module
//! collisions with parallel agents' own `senlib_ffi_<label>.rs` files; the
//! Assembly phase consolidates all of them later.
//!
//! Everything in `flex::FlexAllocator`, `flex::MemoryRegion`, `flex::DeviceTopology`
//! is flex's own bookkeeping/algorithm and has been ported natively in this crate
//! (`allocator.rs`, `memory_region.rs`, `topology.rs`, `domain.rs`, `address.rs`).
//! The literal bottom of the allocator call stack is exactly two senlib primitives:
//!
//! 1. `senlib::MemoryAllocator::Allocate(bytes) -> void*` (PF mode) /
//!    `senlib::v2::VfWrapper::Allocate(flits) -> AllocationIndex` (VF mode) —
//!    called from `flex::DeviceMemoryAllocator::TryAllocate`, verified at
//!    `flex/src/device_interface/device_memory_allocator.cpp:212` (PF,
//!    `memory_allocator_->Allocate`) and `:176`/`:187` (VF, `vfw_->Allocate`,
//!    including the retry-with-backoff loop at `:180-191`). Both backends are
//!    unified behind one `flex_senlib_memory_allocate` entry point here since
//!    which backend is active is a hardware-driver concern, not flex logic —
//!    `flex::DeviceMemoryAllocator` itself just branches on `vfw_ == nullptr`.
//! 2. `senlib::MemoryAllocator::Free(void*)` (PF) / `senlib::v2::VfWrapper::Deallocate(index)`
//!    (VF) — called from `flex::DeviceMemoryAllocator::Free`, verified at
//!    `flex/src/device_interface/device_memory_allocator.cpp:272` (PF, `memory_allocator_->Free`)
//!    and `:261` (VF, `vfw_->Deallocate`).
//!
//! Everything else in `device_memory_allocator.cpp` — `WriteDebugInfo`'s fence-string
//! formatting, `MemoryTracker::RecordAllocation`/`RecordDeallocation` bookkeeping,
//! padding/alignment math (`sendnn::ceil_to`), and the retry/backoff loop around VF
//! `Allocate` — is flex/telemetry-side logic, not a senlib call itself. Padding math
//! is ported natively in `device_memory_allocator.rs::DeviceMemoryAllocator::try_allocate`
//! directly (not `allocator.rs::align_spyre_allocation`, which is a different,
//! higher-level alignment concern). The VF retry-with-backoff loop
//! (`:180-191`) is likewise ported natively there — as a loop around calls
//! to `flex_senlib_memory_allocate` below, not folded into this FFI call's
//! contract (see that function's doc comment). Debug fencing and telemetry
//! counters remain intentionally out of scope for this slice (not on the
//! ABI-reachable path).
//!
//! `senlib::v2::SenPci::ncards()` / per-card topology query (device_topology.hpp
//! construction) is a *different* subsystem's FFI surface (RuntimeContext
//! initialization) and stays out of this file; see the runtime-phase agent's own
//! `senlib_ffi_<label>.rs` / `SENLIB_BOUNDARY_<label>.md` for that boundary.
//!
//! Not built/linked on this machine (no senlib headers/libs present): honest
//! `extern "C"` declarations with no bodies, not stub implementations.

/// Opaque handle to a senlib-owned device memory allocation (PF `void*` from
/// `senlib::MemoryAllocator::Allocate`, or a VF `AllocationIndex` widened to a
/// handle-shaped integer — the two backends are unified at this FFI boundary
/// so callers above it never need to know which one is active). `handle == 0`
/// is the failure sentinel in both the VF (`alloc_index == 0` in
/// `device_memory_allocator.cpp:180`) and unified-here PF case.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SenlibAllocationHandle(pub u64);

impl SenlibAllocationHandle {
    /// The only way to turn a raw FFI return value into a handle this crate
    /// will actually store/use/free. Checks the `handle == 0` failure
    /// sentinel ONCE, here, and returns `None` for it — every other function
    /// in this crate that needs a live handle takes `LiveAllocationHandle`,
    /// not this type, so "forgot to check for the 0 sentinel before storing
    /// it" cannot compile. Same shape as the `capacity() == 0` vs `None`
    /// mapping already fixed in the scheduler subsystem.
    pub fn into_live(self) -> Option<LiveAllocationHandle> {
        std::num::NonZeroU64::new(self.0).map(LiveAllocationHandle)
    }
}

/// A `SenlibAllocationHandle` KNOWN to be live (non-zero) — obtainable only
/// via `SenlibAllocationHandle::into_live`. `DeviceMemoryAllocation` stores
/// this, not the raw FFI type, so it can never hold onto a failure sentinel
/// by accident.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveAllocationHandle(std::num::NonZeroU64);

impl LiveAllocationHandle {
    /// Widen back to the raw FFI type for the one call site that needs it
    /// (`flex_senlib_memory_free`, which shares the ABI-level type with
    /// `flex_senlib_memory_allocate`'s return value).
    pub fn as_raw(self) -> SenlibAllocationHandle {
        SenlibAllocationHandle(self.0.get())
    }

    pub fn as_u64(self) -> u64 {
        self.0.get()
    }
}

unsafe extern "C" {
    /// Corresponds to `senlib::MemoryAllocator::Allocate` (PF,
    /// `device_memory_allocator.cpp:212`) / `senlib::v2::VfWrapper::Allocate`
    /// (VF, `device_memory_allocator.cpp:176`). This FFI call is the bare
    /// senlib primitive only — it does NOT include the retry-with-backoff
    /// loop wrapping the VF call at `device_memory_allocator.cpp:180-191`.
    /// That loop is flex-side C++ logic (`getMaxRetries()`/
    /// `getInitialVfAllocRetryDelay()`, reading `FlexConfig::VfAllocMaxRetries()`/
    /// `VfAllocInitialRetryDelayMs()`, plus a plain `std::this_thread::sleep_for`
    /// exponential backoff), not a senlib/firmware behavior. A prior version
    /// of this comment claimed the retry was "folded into this call's
    /// contract... since VF retry policy is a senlib/firmware behavior" —
    /// that was incorrect on both counts: nothing on the senlib side
    /// retries, and the retry policy belongs to flex, not firmware. The
    /// retry loop is now implemented on the Rust side, in
    /// `device_memory_allocator.rs`'s `DeviceMemoryAllocator::try_allocate`,
    /// which calls this function in a loop rather than assuming retries
    /// happen underneath it.
    /// Returns a null-equivalent (`handle == 0`) on failure; flex-side OOM
    /// handling (memory-pressure callback + retry across regions) lives in
    /// `allocator.rs`, above this call, not here.
    ///
    /// Takes only a byte count, matching the real signatures exactly:
    /// `senlib::MemoryAllocator::Allocate(total_bytes)` and
    /// `VfWrapper::Allocate(total_flits)` — neither accepts a domain id.
    /// `TryAllocate`'s `std::optional<uint32_t> domain_id` argument is
    /// currently logged only, never forwarded to either real call (see
    /// `device_memory_allocator.cpp:153`'s
    /// `TODO(unknown): Pass domain_id to senlib::MemoryAllocator / VfWrapper
    /// when firmware adds bank-targeted allocation support`), so a prior
    /// version of this declaration that took a `domain_id_or_negative_one`
    /// parameter here invented an FFI argument with no real senlib analog.
    pub fn flex_senlib_memory_allocate(num_bytes: u64) -> SenlibAllocationHandle;

    /// Corresponds to `senlib::MemoryAllocator::Free` (PF,
    /// `device_memory_allocator.cpp:272`) / `senlib::v2::VfWrapper::Deallocate`
    /// (VF, `device_memory_allocator.cpp:261`).
    /// Returns 0 on success, -1 if the free did not reach senlib (unknown
    /// handle, or a senlib throw). Was `void`: a failed deallocation was
    /// indistinguishable from a successful one, the same swallowed-failure shape
    /// that caused the batched-decode hang on the submission path.
    pub fn flex_senlib_memory_free(handle: SenlibAllocationHandle) -> i32;
}
