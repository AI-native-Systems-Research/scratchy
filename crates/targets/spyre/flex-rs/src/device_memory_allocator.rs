//! Port of `flex/include/flex/device_interface/device_memory_allocator.hpp`.
//!
//! This is THE senlib boundary. `DeviceMemoryAllocation`/`DeviceMemoryAllocator`
//! are thin RAII/dispatch wrappers around `senlib_ffi`; all `unsafe` in this
//! file is a direct, unavoidable consequence of owning a senlib handle.
//! Everything above this module (`memory_region.rs`, `allocator.rs`) is pure
//! Rust bookkeeping and never touches `unsafe` or `senlib_ffi` directly.

use crate::address::ByteSize;
use crate::memory_region::DomainId;
use crate::senlib_ffi_allocator::{self, LiveAllocationHandle};

/// Permanent (long-lived, e.g. model weights) vs Temporary (short-lived,
/// e.g. activations) allocation lifetime hint. Port of `flex::AllocType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AllocType {
    #[default]
    Permanent,
    Temporary,
}

/// RAII wrapper for one senlib device-memory allocation. Port of
/// `flex::DeviceMemoryAllocation`. The `Drop` impl is the only place besides
/// `senlib_ffi` itself where `unsafe` appears in this module — it frees the
/// senlib-owned handle.
#[derive(Debug)]
pub struct DeviceMemoryAllocation {
    size: ByteSize,
    // `LiveAllocationHandle`, not the raw FFI `SenlibAllocationHandle` — this
    // type cannot exist except by going through
    // `SenlibAllocationHandle::into_live()`'s zero-sentinel check, so a
    // `DeviceMemoryAllocation` can never be constructed holding onto a
    // failed allocation's handle by accident.
    handle: LiveAllocationHandle,
}

impl DeviceMemoryAllocation {
    fn new(size: ByteSize, handle: LiveAllocationHandle) -> Self {
        Self { size, handle }
    }

    pub fn size(&self) -> ByteSize {
        self.size
    }

    /// Device memory physical address in bytes (or the VF-firmware-relative
    /// equivalent) — derived directly from the senlib handle.
    pub fn device_address_bytes(&self) -> u64 {
        self.handle.as_u64()
    }

    // Real `DeviceMemoryAllocation::AlignmentBytes()`/`AllocIndex()`
    // (device_memory_allocator.hpp) are intentionally NOT ported as
    // accessors here. `AlignmentBytes()` would require storing the
    // per-allocation alignment the real type carries
    // (`alignment_bytes_`), which this type currently doesn't retain
    // (`DeviceMemoryAllocator::platform_alignment()` above already exposes
    // the allocator-level alignment that would-be value is derived from).
    // `AllocIndex()` is the VF-only `AIUMsg::V1::AllocationIndex` used by
    // `DeviceMemoryAllocator::Free`'s VF branch to call
    // `vfw_->Deallocate(index)`; this crate's unified
    // `flex_senlib_memory_free(handle)` (see `senlib_ffi_allocator.rs`)
    // already carries that index inside the opaque handle itself, so a
    // separate public accessor for it has no caller here. This is an
    // intentionally narrower accessor surface than the real type's, not an
    // oversight; add both if a caller ever needs them.
}

impl Drop for DeviceMemoryAllocation {
    fn drop(&mut self) {
        // SAFETY: `self.handle` was obtained from a successful
        // `flex_senlib_memory_allocate` call in `DeviceMemoryAllocator::try_allocate`
        // and has not been freed before (ownership is unique — `DeviceMemoryAllocation`
        // is not `Clone`), matching the C++ `~DeviceMemoryAllocation()` -> `allocator_->Free(this)`.
        // The C++ `~DeviceMemoryAllocation()` cannot report a failed `Free`
        // either, but it does not hide one: log it rather than dropping it on
        // the floor (a leaked device allocation is silent otherwise).
        let rc = unsafe { senlib_ffi_allocator::flex_senlib_memory_free(self.handle.as_raw()) };
        if rc != 0 {
            tracing::error!(handle = ?self.handle, "senlib device-memory Free failed; allocation leaked");
        }
    }
}

/// Errors from `DeviceMemoryAllocator::try_allocate`, replacing the C++
/// `sendnn::Status` return convention with a proper `Result`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceMemoryAllocatorError {
    /// Real `TryAllocate` throws two distinct RAS errors for a senlib
    /// allocation failure depending on which backend is active:
    /// `RAS::MEMORY::PFAllocFailed()` (device_memory_allocator.cpp:219,
    /// `memory_allocator_->Allocate` returning `nullptr`) or
    /// `RAS::MEMORY::VFAllocFailed()` (device_memory_allocator.cpp:194,
    /// `vfw_->Allocate` returning a zero index after exhausting retries).
    /// This is intentionally kept as ONE variant rather than split into
    /// `PfAllocFailed`/`VfAllocFailed`: `senlib_ffi_allocator`'s
    /// `flex_senlib_memory_allocate` unifies both backends behind a single
    /// call whose failure sentinel (`handle == 0`) does not indicate which
    /// backend produced it (see that module's doc comment for why). This
    /// type has no independent way to know which backend is active at this
    /// call site, so distinguishing the two here would require either a
    /// real FFI signature change (no evidence one exists) or duplicating
    /// backend-tracking state that already lives elsewhere — not a clean
    /// fix, hence the collapse.
    SenlibAllocationFailed,
    /// Port of `RAS::MEMORY::AllocEmptyFailed().Throw()` — real `TryAllocate`
    /// rejects `num_bytes == 0` before doing any padding/senlib work.
    EmptyAllocation,
}

impl std::fmt::Display for DeviceMemoryAllocatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SenlibAllocationFailed => write!(f, "senlib device memory allocation failed"),
            Self::EmptyAllocation => write!(f, "cannot allocate zero device memory bytes"),
        }
    }
}
impl std::error::Error for DeviceMemoryAllocatorError {}

/// Port of `flex::DeviceMemoryAllocator`. Unlike the C++ version, this does
/// not need a PF/VF branch at the Rust level: `senlib_ffi::flex_senlib_memory_allocate`
/// is documented (see `SENLIB_BOUNDARY.md`) to unify both backends behind one
/// call, since which backend is active is a hardware-driver concern, not a
/// flex-logic concern.
#[derive(Debug, Default)]
pub struct DeviceMemoryAllocator {
    platform_alignment: u64,
}

impl DeviceMemoryAllocator {
    pub fn new(platform_alignment: u64) -> Self {
        Self { platform_alignment }
    }

    pub fn platform_alignment(&self) -> u64 {
        self.platform_alignment
    }

    /// Port of `DeviceMemoryAllocator::TryAllocate` (single-allocation overload).
    /// `context_id`/debug-fencing bookkeeping from the C++ version
    /// (`WriteDebugInfo`) is omitted here as genuinely orthogonal
    /// diagnostics tooling, not part of the 58-function slice.
    pub fn try_allocate(
        &self,
        num_bytes: ByteSize,
        _alloc_type: AllocType,
        // `domain_id` is accepted (matching the real `TryAllocate`'s
        // `std::optional<uint32_t> domain_id` parameter) but not forwarded to
        // the senlib call below: real `TryAllocate` never passes it to either
        // `senlib::MemoryAllocator::Allocate(total_bytes)` or
        // `VfWrapper::Allocate(total_flits)` — both take only a byte/flit
        // count — it is only logged (`TODO(unknown): Pass domain_id to
        // senlib::MemoryAllocator / VfWrapper when firmware adds bank-targeted
        // allocation support`, device_memory_allocator.cpp:153).
        _domain_id: Option<DomainId>,
    ) -> Result<DeviceMemoryAllocation, DeviceMemoryAllocatorError> {
        // Port of `if(num_bytes == 0) { RAS::MEMORY::AllocEmptyFailed().Throw(); }`.
        if num_bytes.as_u64() == 0 {
            return Err(DeviceMemoryAllocatorError::EmptyAllocation);
        }
        // Port of `DeviceMemoryAllocator::TryAllocate`'s PF path: the
        // senlib-facing request is padded up to `alignment` (real
        // `sendnn::ceil_to(num_bytes, alignment)`; the `2 * debug_allocations_`
        // term alongside it is the orthogonal debug-fencing padding this file
        // already omits) — previously missing here entirely, so every
        // allocation this crate made through senlib was unpadded relative to
        // what the real allocator does. The logical size recorded on the
        // returned `DeviceMemoryAllocation` stays the caller's original,
        // unpadded `num_bytes` (real: `std::make_shared<DeviceMemoryAllocation>(num_bytes, ...)`
        // — the padded size is only ever used for the underlying `Allocate` call).
        //
        // `.max(1)` has no real-C++ counterpart (real `GetPlatformAlignment()`
        // is always seeded from `sysconf(_SC_PAGESIZE)`, never 0) — it is an
        // intentional defensive addition here purely to keep `div_ceil` below
        // from panicking on a `platform_alignment` of 0 (e.g. a
        // `DeviceMemoryAllocator::default()` in a test that never called
        // `new`), not a port of any real behavior. Kept as documented
        // defensive code rather than removed.
        let alignment = self.platform_alignment.max(1);
        let padded_bytes = num_bytes.as_u64().div_ceil(alignment) * alignment;

        // Port of `DeviceMemoryAllocator::TryAllocate`'s VF retry-with-backoff
        // loop (device_memory_allocator.cpp:180-191): on an allocation
        // failure (unified `handle == 0` sentinel here, `alloc_index == 0`
        // there), retry up to `FlexConfig::VfAllocMaxRetries()` times
        // (default 1, `FLEX_VF_ALLOC_MAX_RETRIES`), sleeping
        // `FlexConfig::VfAllocInitialRetryDelayMs()` (default 2000ms,
        // `FLEX_VF_ALLOC_INITIAL_RETRY_DELAY_MS`) before the first retry and
        // doubling the delay each subsequent retry
        // (`BACKOFF_MULTIPLIER = 2`, device_memory_allocator.cpp:44/191).
        // Real `TryAllocate` only runs this loop on the VF (`vfw_ != nullptr`)
        // path, never for a PF `memory_allocator_->Allocate` failure — this
        // crate's FFI boundary unifies both backends behind one
        // `flex_senlib_memory_allocate` call (see `senlib_ffi_allocator.rs`)
        // and cannot tell which backend a given failure came from, so the
        // retry loop below runs unconditionally for any failure. For a real
        // VF failure this exactly matches the real retry behavior; for a
        // real PF failure it means this port will (harmlessly) wait out one
        // retry-with-backoff cycle before surfacing the same
        // `SenlibAllocationFailed` the real PF path would have returned
        // immediately — a spurious delay on the failure path, not a
        // correctness bug, and unavoidable without a backend tag at the FFI
        // boundary (see `DeviceMemoryAllocatorError::SenlibAllocationFailed`'s
        // doc comment).
        let max_retries = crate::flex_config::FlexConfig::vf_alloc_max_retries().0;
        let mut retry_delay_ms =
            crate::flex_config::FlexConfig::vf_alloc_initial_retry_delay_ms().0;
        const BACKOFF_MULTIPLIER: i32 = 2;

        // SAFETY: FFI call into senlib with plain integer arguments; no
        // pointers, no aliasing concerns. Failure is signaled via a
        // zero handle per the `senlib_ffi` contract.
        let mut handle = unsafe { senlib_ffi_allocator::flex_senlib_memory_allocate(padded_bytes) };
        let mut retry_count = 0;
        while handle.into_live().is_none() && retry_count < max_retries {
            retry_count += 1;
            std::thread::sleep(std::time::Duration::from_millis(
                retry_delay_ms.max(0) as u64
            ));
            handle = unsafe { senlib_ffi_allocator::flex_senlib_memory_allocate(padded_bytes) };
            retry_delay_ms = retry_delay_ms.saturating_mul(BACKOFF_MULTIPLIER);
        }
        let handle = handle
            .into_live()
            .ok_or(DeviceMemoryAllocatorError::SenlibAllocationFailed)?;
        Ok(DeviceMemoryAllocation::new(num_bytes, handle))
    }
}

// flex/tests/device_types/device_memory_allocator_test.cpp's 6 tests
// (TryAllocateSucceeds/Fails, TryAllocateMultiple*, alignment/domain
// passthrough) are NOT ported here. Every one of them constructs the C++
// `DeviceMemoryAllocator` over a fake/mock senlib allocator seam
// (`MockSenlibAllocator` or equivalent) so it can assert on success/failure
// without real hardware. This Rust type has no such seam: `try_allocate`
// above calls `senlib_ffi_allocator::flex_senlib_memory_allocate`
// unconditionally, so a test here would either require real senlib
// hardware (not exercisable as a unit test) or inventing a mock backend
// this crate's production code doesn't have — fabricating test coverage
// rather than porting real coverage. If a mock seam is added to this type
// for other reasons, these 6 cases should be revisited then.
