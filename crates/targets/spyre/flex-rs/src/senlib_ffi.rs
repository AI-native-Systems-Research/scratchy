//! THE senlib FFI boundary — every `extern "C"` declaration this crate makes
//! against senlib lives here, and nowhere else.
//!
//! The port was written by parallel phases, each with its own
//! `senlib_ffi_<label>.rs` file (to avoid symbol collisions while phases ran
//! concurrently); this module is the full consolidation — the per-phase files
//! are gone, their declarations moved here uncurated (the crate's scope is
//! checked against a CodeQL class-hierarchy closure, and a silently-shrunk
//! boundary is a silent API regression). The one subsystem whose declarations
//! did NOT survive is `multi_device` — see "The multi_device deletion" below.
//!
//! See `SENLIB_BOUNDARY.md` at the crate root for the full per-subsystem
//! boundary rationale; this file carries module wiring, the declarations, the
//! safe wrappers, and the audit trail.
//!
//! ## Duplicate found and fixed: "how many Spyre cards are there"
//!
//! Three phases independently reached `senlib::v2::SenPci::ncards()` (PCI
//! enumeration of physical cards) from three different call sites in the
//! C++ source, and each declared its own `extern "C"` binding for it:
//!   - the runtime-core phase, as `flex_senlib_pci_ncards` (from
//!     `RuntimeContext`/`getNumDevices`);
//!   - the config/util phase, as `senlib_senpci_ncards` (from
//!     `FlexConfig::FlexSpyreDevices`);
//!   - a pre-split placeholder, as part of a since-removed
//!     `flex_senlib_query_topology` (topology-phase card+HBM-layout query).
//!
//! `flex_senlib_query_topology` was never implemented in
//! `cxx-shim/senlib_ffi.cpp` (no such symbol exists in the real SDK either —
//! see the "REAL LINK-TARGET AUDIT" below) and real-hardware testing hit a
//! `FlexAllocator` OOM traced to scratchy's runtime bring-up
//! (`scratchy-target-spyre`'s `runtime::build_runtime`) sizing its
//! single domain off a 32 GiB placeholder instead of the card's real HBM
//! capacity. It is replaced below by `flex_senlib_device_memory_size`, a
//! one-line real call (`DeviceHandle::GetDmpaSize()`) matching exactly what
//! the real C++ path does (`device_memory_topology.cpp::populateDomainTopology`
//! fed by `getDeviceMemorySize()`) — this crate's 1p0 target has exactly one
//! domain, so a full multi-domain query was never actually needed. The other
//! two — `flex_senlib_pci_ncards` and `senlib_senpci_ncards` — are ASSUMED to
//! be the same zero-argument `ncards() -> count` call declared twice under
//! different names, both citing `senlib::v2::SenPci`/`SenPciShared::ncards()`
//! from their respective call sites in `flex_config.cpp`. This is an
//! unverified assumption, not a confirmed fact: `flex-cxx` does not vendor
//! senlib itself, so there is no header available in this audit to check
//! that `SenPci::ncards()` and `SenPciShared::ncards()` really are the exact
//! same underlying call rather than, say, two related-but-distinct methods
//! on different senlib-side wrapper types. `flex_senlib_pci_ncards` is kept
//! as the one canonical declaration (it has the closer-to-source name); the
//! config/util phase's `senlib_senpci_ncards` extern was deleted and its
//! safe wrapper (`sen_pci_card_count`) now calls the canonical declaration
//! instead. `flex_config.rs`, which calls `sen_pci_card_count()`, needed no
//! changes.
//!
//! The old pre-split placeholder externs that used to live directly in the
//! pre-consolidation hub file (`flex_senlib_stream_create/destroy/submit/
//! poll_completions`, `SenlibStreamHandle`, `RawControlBlock`) had zero
//! callers anywhere in the crate — they were superseded by the scheduler
//! phase's real, call-site-verified declarations (`senlib_cbi_*`/
//! `senlib_rbi_*`, used by `scheduler.rs`/`stream.rs`) and were removed
//! rather than re-exported as dead code.
//!
//! ## The multi_device deletion (resolved 2026-10-16, issue #530)
//!
//! The RDMA/HDMA subsystem's thirteen declarations (`flex_senlib_pin_and_map`,
//! `flex_senlib_pinned_memory_*`, `flex_senlib_direct_set_wdone`,
//! `flex_senlib_ainc_wdone`, `flex_senlib_rdma_*`) and their would-be caller
//! `multi_device.rs` (1928 lines: `BitmaskHelper`/`BitSetHelper`/
//! `HdmaAllocationManager`/`RdmaUnit`-shaped surface, barrier/broadcast over
//! a POSIX rendezvous file) were DELETED, not folded, on evidence:
//!   1. **No callers.** Nothing in the workspace referenced
//!      `flex_rs::multi_device` — only `lib.rs`'s module declaration and this
//!      hub's re-export (verified by a workspace-wide grep).
//!   2. **No link targets.** `cxx-shim/senlib_ffi.cpp` implements ZERO of the
//!      thirteen symbols; anything that called them would fail at `ld` with
//!      undefined references. They were written against flex-cxx's
//!      mock-device shared-memory layer (`RdmaShm`), not even the real P2P
//!      surface.
//!
//! That half-state was a trap for whoever wires P2P next: they would find a
//! fully-shaped module whose FFI bottom links to nothing. Git history
//! preserves all of it; re-adding with real shim entrypoints when P2P wiring
//! actually happens is the honest shape. The per-call-site citations for the
//! deleted declarations live in `SENLIB_BOUNDARY.md`'s deletion record.
//!
//! ## The multi_device re-add (issue #550)
//!
//! #550 re-added the multi-device surface with REAL shim entrypoints. The
//! five senlib-real declarations (the pin-and-map family and the wdone
//! register pokes, restored below with their original names) are now
//! implemented in `cxx-shim/senlib_ffi.cpp`; the eight mock-`RdmaShm`
//! declarations were NOT restored — the rebuilt `multi_device` bottoms out
//! on senlib's cross-port P2P IOMMU mapping instead
//! (`flex_senlib_p2p_create_mapping`, senlib
//! `vfio_backend::create_iommu_mapping_for_cross_port_p2p_rdma`) plus the
//! BAR DMABuf export (`flex_senlib_get_dmabuf`). Every signature below that
//! could not be checked against the on-pod SDK from this worktree carries a
//! `FIXME(pod-verify)` at its declaration; the transport-layering rationale
//! (why senlib's mapping primitive and not spyre-comms' collectives API) is
//! in `SENLIB_BOUNDARY.md`'s multi-device section.
//!
//! ## The controlblocks phase: no declarations, on purpose
//!
//! The DMA/compute-params + control-blocks subsystem (`crate::dma`,
//! `crate::compute`, `crate::control_blocks`) makes **zero senlib calls** —
//! verified by reading every file assigned to that phase:
//! `runtime_submission_params.{hpp,cpp}` (pure parameter validation),
//! `control_block_stream.hpp`/`segment_addressing.hpp` (pure arithmetic),
//! `dma_descriptor_pack_*` (plain value types), and the mock/software CB
//! executors `dma_control_block_handler`/`control_block_handler`/
//! `compute_pipeline_control_block_handler` (memcpy/pointer arithmetic into
//! a flex-owned `DeviceMemory` mock buffer, selected when
//! `RuntimeConfig::FlexCompute()` is `"NULL"`/`"SENULATOR"`). The one
//! senlib-adjacent type `DmaControlBlockHandler` names is a read-only
//! on-wire struct layout, not a call. The real hardware submission lives in
//! the scheduler section below — nothing duplicates it here.
//!
//! # ---------------------------------------------------------------------
//! # REAL LINK-TARGET AUDIT (2026-08-15, against the actual on-pod SDK at
//! # /opt/ibm/spyre)
//! # ---------------------------------------------------------------------
//!
//! Every `flex_senlib_*`/`senlib_*` symbol declared in this module is a NAME
//! THAT DOES NOT EXIST in any installed SDK library (`nm -D --defined-only`
//! on every .so under /opt/ibm/spyre/{senlib,runtime,deeptools}/lib turns up
//! zero matches for `flex_senlib` or `senlib_control_block`/
//! `senlib_response_block`). These declarations were written as an
//! aspirational C shim contract; a real senlib/flex install is a C++ API
//! (mangled `_Z...` symbols) with no C shim anywhere in the SDK exporting
//! these plain names, so linking against a real build fails at `ld` with
//! "undefined reference" for every one of them — not a runtime bug.
//!
//! The resolution, confirmed by reading the real headers on the pod:
//! `cxx-shim/senlib_ffi.cpp` is the hand-written C++ shim TU that exports
//! every one of these names as a plain `extern "C"` symbol, each a one-line
//! marshal onto the real, PUBLIC `flex::DeviceHandle` /
//! `senlib::ControlBlockInterface` / `senlib::ResponseBlockInterface`
//! methods cited in each declaration's doc comment. It is compiled by
//! flex-rs's `build.rs` unconditionally by default and links against the
//! real senlib on the pod (`/opt/ibm/spyre/senlib/lib`, plus its own
//! transitive deps); `SCRATCHY_SKIP_SENDNN_CXX=1` skips the compile for
//! check/clippy on a machine without the SDK headers (a link then fails
//! loudly).
//!
//! `flex_shim_cbi_for_pipeline`/`flex_shim_rbi_for_pipeline` are the shim
//! entry points that turn a `PipelineId` into the actual
//! `ControlBlockInterface*`/`ResponseBlockInterface*` opaque handle;
//! `flex_shim_init_runtime` (below) brings up the real device and caches its
//! `DeviceHandle` process-wide — it must be called once, successfully, before
//! any other function here.
//!
//! `scheduler::FlexAllocatorAddressResolver` is the production
//! `DeviceAddressResolver` — it resolves `Chunk::addr.region_id` through a
//! live `FlexAllocator`'s `get_id_to_region_map()` and reads the region's
//! real device physical address off `MemoryRegion::data().
//! device_address_bytes()`. `PfBackend`/`VfBackend` can be constructed
//! end-to-end by a caller that owns an `Arc<Mutex<FlexAllocator>>` and an
//! `Arc<dyn HardwareSubmit>` built on top of this module's functions.

use std::ffi::CStr;
use std::ffi::c_void;
use std::os::raw::{c_char, c_int, c_uint};

// ════════════════════════════════════════════════════════════════════════════════════════════
//  Runtime bring-up, topology, and IOMMU
// ════════════════════════════════════════════════════════════════════════════════════════════

unsafe extern "C" {
    /// Brings up the real device runtime (idempotent, thread-safe on the
    /// C++ side via `std::call_once` — see `cxx-shim/flex_shim.cpp`).
    /// MUST be called (and succeed, i.e. return 0) before any other
    /// function in this module is called; every other function
    /// assumes a live `flex::DeviceHandle` is already cached.
    /// `logical_device_id` mirrors `flex::initializeRuntime`'s own parameter
    /// (which Spyre card to bring up; 0 for a single-device host).
    pub fn flex_shim_init_runtime(logical_device_id: i32) -> i32;

    /// Real device memory capacity in bytes for the currently-open card —
    /// `flex::DeviceHandle::GetDmpaSize()`, the exact call the real C++
    /// allocator bring-up path uses to size its single 1p0 domain
    /// (`flex/src/runtime_stream/1p0/device_memory_topology.cpp`'s
    /// `populateDomainTopology(mem_size)`, fed by
    /// `getDevice()->getDeviceMemorySize() -> DeviceHandle::GetDmpaSize()`).
    /// For a PF device this resolves to `PfDeviceHandle::GetDmpaSize() ->
    /// device_memory_->DeviceMemorySize() -> lpddr_size_`, itself queried
    /// once at device-open time from `senlib::v2::PfInterface::GetLPDDRSize()`
    /// — this is the real HBM capacity of the open card, not a placeholder.
    /// Requires `flex_shim_init_runtime` to have already succeeded; returns
    /// 0 (and logs to stderr) otherwise.
    pub fn flex_senlib_device_memory_size() -> u64;

    /// Real RCU (compute-core) count on the open card —
    /// `senlib::v2::SenPci::RCU_num_cores()`, the exact call
    /// `flex::populateDomainTopology` uses to build the 1p0 target's
    /// single-domain core-affinity vector
    /// (`flex/src/runtime_stream/1p0/device_memory_topology.cpp:32`). Static
    /// senlib call, no `flex_shim_init_runtime` prerequisite, but declared
    /// here since it feeds the same topology-population call as
    /// `flex_senlib_device_memory_size`.
    pub fn flex_senlib_rcu_num_cores() -> u32;

    /// Corresponds to `senlib::v2::SenPci::ncards()`
    /// (`flex/src/runtime_stream/runtime_context.cpp:44`), which returns the
    /// number of Spyre cards physically present in the system, independent
    /// of any process-level device filtering (`SPYRE_DEVICES`/
    /// `AIU_WORLD_SIZE`, applied by flex-side logic in `runtime.rs`, not
    /// here). The canonical declaration for the duplicate resolved in this
    /// module's doc.
    pub fn flex_senlib_pci_ncards() -> u64;

    /// `PfIommuMapper::Map`: `RegisterMemForIOMMU` + `pin_and_map`, first
    /// sglist entry. 0 on success.
    ///
    /// `writable` mirrors the real `PfIommuMapper::Map(IommuMappingPtr*,
    /// void* hmva, size_t size_bytes, bool writable)` signature
    /// (pf_iommu_mapper.cpp:44/pf_iommu_mapper.hpp:64) — added here to match
    /// it exactly; a prior version of this declaration omitted it. Note
    /// that in the real implementation this bool is NOT forwarded into
    /// either underlying senlib call (`RegisterMemForIOMMU`/`pin_and_map`
    /// take no writable/permission argument in the real code,
    /// pf_iommu_mapper.cpp:48-49) — real `Map` only uses it to construct the
    /// `IommuMapping` bookkeeping object (`iomap->writable_`) returned to the
    /// caller. This shim function mirrors that: the parameter is
    /// accepted (so the signature matches the real `Map()` this function
    /// corresponds to) but is not passed to the two senlib calls beneath
    /// it, matching real behavior rather than inventing a senlib argument
    /// that doesn't exist. The Rust-side `writable` bookkeeping equivalent
    /// to `iomap->writable_` is handled by the caller
    /// (`iommu.rs::PfIommuMapper::map`, which already stores it on
    /// `IommuMapping` independent of this FFI call).
    pub fn flex_senlib_iommu_register_and_map(
        hmva: *mut std::ffi::c_void,
        size_bytes: u64,
        writable: bool,
        out_iova: *mut *mut std::ffi::c_void,
        out_size: *mut u64,
    ) -> i32;

    /// `PfIommuMapper::Unmap`: `unpin_and_unmap` + `UnregisterMemForIOMMU`.
    pub fn flex_senlib_iommu_unmap_and_unregister(
        hmva: *mut std::ffi::c_void,
        iova: *mut std::ffi::c_void,
        size_bytes: u64,
    ) -> i32;
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  Device-memory allocator
// ════════════════════════════════════════════════════════════════════════════════════════════
//
// Everything in `flex::FlexAllocator`, `flex::MemoryRegion`,
// `flex::DeviceTopology` is flex's own bookkeeping/algorithm and is ported
// natively (`allocator.rs`, `memory_region.rs`, `topology.rs`, `domain.rs`,
// `address.rs`). The literal bottom of the allocator call stack is exactly
// two senlib primitives, both below, unified across PF/VF backends since
// which backend is active is a hardware-driver concern, not flex logic.
// Everything else in `device_memory_allocator.cpp` — `WriteDebugInfo`'s
// fence-string formatting, `MemoryTracker` bookkeeping, padding/alignment
// math (`sendnn::ceil_to`), and the VF retry-with-backoff loop — is
// flex-side logic ported natively (`device_memory_allocator.rs`), not folded
// into these calls' contracts.

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
    /// exponential backoff), not a senlib/firmware behavior. The
    /// retry loop is implemented on the Rust side, in
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

// ════════════════════════════════════════════════════════════════════════════════════════════
//  Scheduler: the CBI/RBI submission + completion surface
// ════════════════════════════════════════════════════════════════════════════════════════════
//
// `flex::RuntimeScheduler` (abstract base) + `PfRuntimeScheduler`/
// `VfRuntimeScheduler`/`MockRuntimeScheduler`. Everything else this subsystem
// does — fence/barrier bookkeeping, in-flight counters, pipeline-switch
// detection, DMA chunk looping, control-block *orchestration* — is flex's own
// logic and is ported natively in `scheduler.rs`. Building the literal CB bit
// layout (SBF structures) is `control_block_wire`'s scope; this section only
// models the narrow senlib-facing surface the scheduler itself touches.

/// Opaque handle mirroring `senlib::ControlBlockInterface*`. Never
/// dereferenced field-by-field in this crate — only passed back into the
/// `extern "C"` entry points below.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SenlibControlBlockInterfaceHandle(pub *mut c_void);

impl SenlibControlBlockInterfaceHandle {
    /// The only way to get a `LiveCbiHandle`: checks the null sentinel
    /// (shim runtime not initialized / pipeline out of range) exactly once.
    /// Every safe wrapper below (`cbi_capacity`/`cbi_queue_control_blocks_sbf`)
    /// takes `LiveCbiHandle`, not this raw type, so a caller cannot pass a
    /// possibly-null handle to senlib without the compiler forcing the
    /// `into_live()` check first — three separate hand-written
    /// `cbi.0.is_null()` checks in `scheduler.rs` collapse to this one.
    pub fn into_live(self) -> Option<LiveCbiHandle> {
        std::ptr::NonNull::new(self.0).map(LiveCbiHandle)
    }
}

/// A `SenlibControlBlockInterfaceHandle` known to be non-null.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveCbiHandle(std::ptr::NonNull<c_void>);

impl LiveCbiHandle {
    fn as_raw(self) -> SenlibControlBlockInterfaceHandle {
        SenlibControlBlockInterfaceHandle(self.0.as_ptr())
    }
}

/// Opaque handle mirroring `senlib::ResponseBlockInterface*`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SenlibResponseBlockInterfaceHandle(pub *mut c_void);

impl SenlibResponseBlockInterfaceHandle {
    /// See `SenlibControlBlockInterfaceHandle::into_live` — same shape.
    pub fn into_live(self) -> Option<LiveRbiHandle> {
        std::ptr::NonNull::new(self.0).map(LiveRbiHandle)
    }
}

/// A `SenlibResponseBlockInterfaceHandle` known to be non-null.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveRbiHandle(std::ptr::NonNull<c_void>);

impl LiveRbiHandle {
    fn as_raw(self) -> SenlibResponseBlockInterfaceHandle {
        SenlibResponseBlockInterfaceHandle(self.0.as_ptr())
    }
}

/// One control block in senlib's wire format (`senlib::ControlBlockSBF`).
/// Opaque, fixed-size, POD from senlib's perspective — this crate never
/// interprets its bytes; `control_block_wire` is responsible for populating
/// it (via the `From` impl below).
///
/// SIZE VERIFIED 2026-08-15 against the real SDK on-pod
/// (`hal/1p0/control_block_sbf.hpp`: `constexpr uint64_t CB_NUM_BYTES = 128;`,
/// six 8/16-byte sections — CTRL/R5/DMI/DMO/CMPT/XLAT). The previous `[u8; 256]`
/// here was never checked against a real senlib header and was simply
/// wrong — double the real size. Getting this wrong would have silently
/// corrupted every other queued CB (stride mismatch) the moment a real
/// `senlib_control_block_interface_queue_control_blocks_sbf` was linked.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SenlibControlBlockSbf {
    pub bytes: [u8; 128],
}

/// One response block in senlib's wire format (`senlib::ResponseBlockSBF`).
///
/// SIZE VERIFIED 2026-08-15 against the real SDK on-pod
/// (`hal/1p0/response_block_sbf.hpp`: `constexpr uint64_t RB_NUM_BYTES = 64;`,
/// enforced there by a `static_assert(sizeof(ResponseBlockSBF) == RB_NUM_BYTES)`).
/// Same "never verified against a real header" issue as
/// `SenlibControlBlockSbf` above — was `[u8; 256]`, four times too large.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SenlibResponseBlockSbf {
    pub bytes: [u8; 64],
}

// SEALED: the `senlib_*` declarations in this block are module-private on
// purpose. The only way to reach senlib's submission/completion surface from
// the rest of the crate is through the safe `Result`-returning wrappers
// below, so a call site cannot skip a status check or ignore a failure —
// the port's original `void` submit + swallowed exception is unrepresentable
// here now. The two `pub fn flex_shim_*` handle getters at the top of the
// block are the deliberate exception: they only fetch opaque handles (whose
// null sentinel is checked by `into_live()`) and cannot fail silently. Do
// not add `pub` to anything else in this block.
unsafe extern "C" {
    /// Returns the `senlib::ControlBlockInterface*` for `pipeline`
    /// (0=compute, 1=async-dma-in, 2=async-dma-out — see
    /// `cxx-shim/flex_shim.cpp`'s `FlexShimPipeline` enum, kept in sync by
    /// hand with `PipelineId` here) as an opaque handle, or a null handle if
    /// the shim runtime has not been initialized (see
    /// `flex_shim_init_runtime` above) or `pipeline` is out of range.
    /// Backs `DeviceHandle::ComputeCbi()/AsyncDmaICbi()/AsyncDmaOCbi()`.
    pub fn flex_shim_cbi_for_pipeline(pipeline: i32) -> SenlibControlBlockInterfaceHandle;

    /// Returns the process's single `senlib::ResponseBlockInterface*`
    /// (`DeviceHandle::Rbi()`) as an opaque handle. `pipeline` is accepted
    /// for symmetry with `flex_shim_cbi_for_pipeline` but ignored by the
    /// shim — senlib exposes exactly one RBI per device.
    pub fn flex_shim_rbi_for_pipeline(pipeline: i32) -> SenlibResponseBlockInterfaceHandle;

    /// Port of `senlib::ControlBlockInterface::capacity() const`.
    /// Call site: `runtime_scheduler.cpp:1025` (`RuntimeScheduler::getQueueCapacity`).
    #[link_name = "senlib_control_block_interface_capacity"]
    fn senlib_cbi_capacity(cbi: SenlibControlBlockInterfaceHandle, out_capacity: *mut usize)
    -> i32;

    /// The calling thread's most recent shim error (`""` if none). Owned by the
    /// shim, valid until this thread's next failed shim call.
    #[link_name = "flex_shim_last_error"]
    fn flex_shim_last_error() -> *const c_char;

    /// Port of `senlib::ControlBlockInterface::QueueControlBlocksSBF(const ControlBlockSBF*, uint64_t)`
    /// (`senlib/1p0/control_block_interface.hpp:46,76`).
    /// Call site: `mock_runtime_scheduler.cpp:322` (direct); reached indirectly
    /// by PF/VF through `response_worker.cpp:1214,1268` (out of this
    /// subsystem's file scope — see the section doc above).
    ///
    /// Returns 0 on success, -1 if the submission did not reach senlib. The
    /// senlib method itself is `void`; the status is the shim's stand-in for
    /// the C++ exception that the real call site
    /// (`response_worker.cpp:1268`, unguarded by any try/catch) lets
    /// propagate. See `cxx-shim/senlib_ffi.cpp` for why this must not be
    /// swallowed.
    #[link_name = "senlib_control_block_interface_queue_control_blocks_sbf"]
    fn senlib_cbi_queue_control_blocks_sbf(
        cbi: SenlibControlBlockInterfaceHandle,
        cbs: *const SenlibControlBlockSbf,
        count: usize,
    ) -> i32;

    /// Port of `senlib::ControlBlockInterface::FreeResponses(uint64_t)`
    /// (`senlib/1p0/control_block_interface.hpp:49`). Returns 0 on success,
    /// -1 if the call did not reach senlib.
    ///
    /// Call site: `ResponseWorker::ParseResponseBlocks`
    /// (`response_worker.cpp:1141-1152`), once per pipeline per drain. This
    /// symbol did not exist in the port at all, so consumed response-block
    /// slots were never returned to senlib: occupancy grew monotonically until
    /// `QueueControlBlocksSBF` threw "Control block queue is full, and has not
    /// drained" at exactly `cb_queue_length` CBs. That was the long-standing
    /// long-context batched-decode hang.
    #[link_name = "senlib_control_block_interface_free_responses"]
    fn senlib_cbi_free_responses(cbi: SenlibControlBlockInterfaceHandle, num_responses: u64)
    -> i32;

    /// Port of `senlib::ResponseBlockInterface::ReceiveResponsesSBF(ResponseBlockSBF*, size_t, size_t)`.
    /// Call site: `mock_runtime_scheduler.cpp:329`.
    #[link_name = "senlib_response_block_interface_receive_responses_sbf"]
    fn senlib_rbi_receive_responses_sbf(
        rbi: SenlibResponseBlockInterfaceHandle,
        out: *mut SenlibResponseBlockSbf,
        max_count: usize,
        min_count: usize,
        out_received: *mut u64,
    ) -> i32;
}

impl From<crate::control_block_wire::ControlBlockWire> for SenlibControlBlockSbf {
    /// `ControlBlockWire::to_bytes()` is byte-for-byte the same 128-byte
    /// `SentientSoc::V1::ControlBlockSBF` layout this type mirrors — see
    /// `crate::control_block_wire`'s module doc for the verified layout.
    fn from(wire: crate::control_block_wire::ControlBlockWire) -> Self {
        Self {
            bytes: wire.to_bytes(),
        }
    }
}

impl SenlibResponseBlockSbf {
    /// Decode the RB's `ReturnSectionSBF` (completion status/tag/cancel).
    /// See `crate::control_block_wire::ResponseBlockWire` for the verified
    /// bit layout this delegates to.
    pub fn decode(&self) -> crate::control_block_wire::ResponseBlockWire {
        crate::control_block_wire::ResponseBlockWire::from_bytes(&self.bytes)
    }
}

/// The calling thread's most recent shim error message, for attaching to a
/// Rust-side error. Empty if the shim has not recorded one.
pub fn last_shim_error() -> String {
    // SAFETY: the shim returns a NUL-terminated, thread-local buffer that stays
    // valid until this thread's next failed shim call; we copy it immediately.
    unsafe {
        let p = flex_shim_last_error();
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

/// Safe wrapper: query a pipeline's configured CB queue capacity.
///
/// Returns `None` when `cbi` is `None` (mirrors the C++ `if(!cbi) return
/// nullopt`) — the caller (this subsystem's `getQueueCapacity`) additionally
/// treats a reported `0` as "not applicable", which is *not* senlib's
/// concern and is handled in native Rust, not here. Taking `LiveCbiHandle`
/// instead of the raw, possibly-null type means there is no null check left
/// to write (or omit) in this function's own body.
pub fn cbi_capacity(cbi: Option<LiveCbiHandle>) -> Option<usize> {
    let cbi = cbi?;
    let mut capacity = 0usize;
    // SAFETY: `cbi` is a live senlib::ControlBlockInterface* obtained from
    // DeviceHandle for the lifetime of the owning RuntimeScheduler; capacity()
    // is a const, side-effect-free query per the senlib API contract.
    // A senlib throw must NOT be reported as capacity 0: upper layers read 0 as
    // "capacity not reported" and stop enforcing backpressure entirely.
    // `None` here means the same thing, so the failure is logged loudly.
    let rc = unsafe { senlib_cbi_capacity(cbi.as_raw(), &mut capacity) };
    if rc != 0 {
        tracing::error!(err = %last_shim_error(), "senlib ControlBlockInterface::capacity() failed");
        return None;
    }
    Some(capacity)
}

/// Safe wrapper: ring the doorbell — hand a batch of already-built control
/// blocks to senlib for hardware submission. `cbs` must have been fully
/// populated by `control_block_wire` before this call.
///
/// # Errors
/// Returns `Err` with the shim's message if the submission did not reach
/// senlib. The real call site (`response_worker.cpp:1268`) makes an unguarded
/// `void` call, so a senlib throw there propagates out of the worker; the
/// caller must FAIL the request rather than observe-and-ignore. Swallowing it
/// leaves the caller's already-incremented in-flight count waiting on a CB
/// hardware never received.
pub fn cbi_queue_control_blocks_sbf(
    cbi: LiveCbiHandle,
    cbs: &[SenlibControlBlockSbf],
) -> Result<(), String> {
    // SAFETY: `cbi` is a live handle; `cbs`/`cbs.len()` describe a valid
    // slice for the duration of this call, matching
    // `QueueControlBlocksSBF(const ControlBlockSBF*, uint64_t)`'s contract.
    let rc = unsafe { senlib_cbi_queue_control_blocks_sbf(cbi.as_raw(), cbs.as_ptr(), cbs.len()) };
    if rc == 0 {
        Ok(())
    } else {
        Err(last_shim_error())
    }
}

/// Safe wrapper: return `num_responses` consumed response-block slots to
/// senlib, the port of `ParseResponseBlocks`' per-pipeline
/// `cbi->FreeResponses(n)` (`response_worker.cpp:1141-1152`). Without this the
/// hardware queue never drains from senlib's point of view.
///
/// # Errors
/// Returns `Err` with the shim's message if the call did not reach senlib.
/// The real method is `void` and its call site is unguarded, so a senlib
/// throw there propagates out of the response-completion thread; the caller
/// decides how to surface that (see `scheduler.rs`).
pub fn cbi_free_responses(cbi: LiveCbiHandle, num_responses: u64) -> Result<(), String> {
    // SAFETY: `cbi` is a live senlib::ControlBlockInterface* for the lifetime
    // of the owning scheduler; `FreeResponses` takes a plain count.
    let rc = unsafe { senlib_cbi_free_responses(cbi.as_raw(), num_responses) };
    if rc == 0 {
        Ok(())
    } else {
        Err(last_shim_error())
    }
}

/// Safe wrapper: drain up to `out.len()` response blocks, blocking until at
/// least `min_count` have arrived. Returns the number actually received.
/// # Errors
/// Returns `Err` with the shim's recorded message if the senlib call threw. A
/// count of `0` is NOT an error — with the real `min_count == 0` it is the
/// normal "nothing has arrived yet" answer, which is exactly why the failure
/// needs its own channel.
pub fn rbi_receive_responses_sbf(
    rbi: LiveRbiHandle,
    out: &mut [SenlibResponseBlockSbf],
    min_count: usize,
) -> Result<u64, String> {
    // SAFETY: `rbi` is a live handle; `out` is a valid mutable slice sized
    // for `max_count`, matching `ReceiveResponsesSBF`'s out-parameter contract.
    let mut received = 0u64;
    let rc = unsafe {
        senlib_rbi_receive_responses_sbf(
            rbi.as_raw(),
            out.as_mut_ptr(),
            out.len(),
            min_count,
            &mut received,
        )
    };
    if rc == 0 {
        Ok(received)
    } else {
        Err(last_shim_error())
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  Config/util: card-count and PCIe-address lookups
// ════════════════════════════════════════════════════════════════════════════════════════════
//
// Almost entirely env-var/JSON-config parsing with process-wide caching
// (`flex_config.rs`, `util.rs`); exactly two calls, both inside `FlexConfig`
// (not `RuntimeConfig`), and both irreducible hardware/driver topology
// queries with no flex-owned bookkeeping above them:
//   - `senlib::v2::SenPci::ncards()` / `SenPciShared::ncards()` in
//     `flex_config.cpp:379,408` (`FlexConfig::FlexSpyreDevices`) — the
//     duplicate resolved onto `flex_senlib_pci_ncards` above.
//   - `senlib::v2::SenPci::pf(card_index).to_string()` / `SenPciShared::pf(...)`
//     in `flex_config.cpp:442,471` (`FlexConfig::RdmaGetPCIeAddress`).
// Whether `SenPci` and `SenPciShared`'s methods are the exact same underlying
// senlib call rather than two distinct-but-similar ones is NOT independently
// verified — `flex-cxx` does not vendor senlib itself, so no senlib header
// was available to check during the audit; it is carried as an assumption.

// NOTE: `senlib::v2::SenPci::ncards()` used to be declared here a second time
// as `senlib_senpci_ncards`. It is ASSUMED to be the same real senlib call as
// `flex_senlib_pci_ncards` above — see the module doc's duplicate-resolution
// note for the unverified-assumption caveat.
unsafe extern "C" {
    /// `senlib::v2::SenPci::pf(card_index).to_string()` /
    /// `SenPciShared::pf(...)`: PCIe bus address string for the physical
    /// function of card `card_index`. Returns a senlib-owned, null-terminated
    /// buffer valid for the duration of the call (no ownership transfer).
    fn senlib_senpci_pf_address(card_index: c_uint) -> *const c_char;
}

/// Safe wrapper over `senlib::v2::SenPci::ncards()`. Delegates to the
/// canonical `flex_senlib_pci_ncards` declaration (see the module doc for why
/// this used to have its own duplicate extern).
pub fn sen_pci_card_count() -> i64 {
    // SAFETY: `flex_senlib_pci_ncards` takes no arguments and returns a plain
    // integer; senlib guarantees it is callable at any point after library
    // init (card enumeration happens once at senlib load time).
    unsafe { flex_senlib_pci_ncards() as i64 }
}

/// Safe wrapper over `senlib::v2::SenPci::pf(card_index).to_string()`.
pub fn pcie_address_for_card(card_index: u32) -> String {
    // SAFETY: `card_index` is caller-checked (by `FlexConfig`) to be less than
    // `sen_pci_card_count()` before this is called; senlib returns a valid
    // null-terminated C string for any in-range index, and the buffer is not
    // retained by us beyond this call.
    unsafe {
        let ptr = senlib_senpci_pf_address(card_index);
        if ptr.is_null() {
            return String::new();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════
//  Multi-device / cross-port P2P (re-added per issue #550)
// ════════════════════════════════════════════════════════════════════════════════════════════
//
// `flex/multi_device`'s FFI bottom. #530 deleted this subsystem's thirteen
// declarations because none of them had a shim implementation — eight were
// written against flex-cxx's mock `RdmaShm` layer, and even the five
// senlib-real ones (the pin-and-map family, the wdone register pokes)
// linked to nothing. #550 re-adds the surface against the real transport:
//
//   - the five senlib-real declarations, restored under their original
//     names and now implemented in `cxx-shim/senlib_ffi.cpp`;
//   - the eight mock-RdmaShm declarations are NOT restored. Their roles are
//     filled by the real cross-port P2P surface: address translation
//     (`rdma_xseg_*`) is the IOMMU mapping itself (`p2p_create_mapping`
//     below); the data movement (`rdma_copy_to_remote`) is an ordinary H2D
//     DMA through the scheduler aimed at the mapped address; the
//     write-done counters (`rdma_wdone_inc/check`) are the framed VSID
//     instructions plus the wdone register pokes restored here.
//
// Peer ENUMERATION was already this module's surface: `sen_pci_card_count()`
// above is the card count and `pcie_address_for_card(i)` names each peer;
// `p2p_create_mapping(peer_card_index, ..)` is the piece that was missing.
//
// The transport-layering decision (see `SENLIB_BOUNDARY.md`): the vendor's
// collectives library, spyre-comms, cannot be scratchy's seam — its
// `initialize_library` demands live C++ `flex::RuntimeContext*`/
// `flex::RuntimeStream*` objects (torch-spyre
// `csrc/distributed/spyre_ccl.cpp:49-51`, `module.h:41-42`), exactly the
// objects #512's fxa collapse removed. scratchy binds the senlib primitive
// BELOW spyre-comms instead, and `multi_device.rs`'s restored native
// sections are scratchy's counterpart of the comms layer above it.
//
// Everything in this section follows the scheduler section's sealed
// pattern: the externs are module-private and the only way to reach them is
// through the safe `Result`-returning wrappers below. Every signature that
// could not be checked against the on-pod SDK from this worktree carries a
// `FIXME(pod-verify)`.

/// Opaque handle to a senlib-owned pinned host-memory region (the product of
/// `PfWrapper::allocate_pin_and_map`, cited at flex-cxx `hdma_shm.cpp:903-914`
/// — page-locked host memory mapped into the device's DMA address space,
/// optionally backed by a named shm file so peer processes can attach to the
/// same region). `handle == 0` is the failure sentinel, same shape as
/// `SenlibAllocationHandle`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SenlibPinnedMemoryHandle(pub u64);

impl SenlibPinnedMemoryHandle {
    /// The only way to turn a raw FFI return value into a handle this crate
    /// will actually use: checks the `handle == 0` failure sentinel once,
    /// here. Same shape as `SenlibAllocationHandle::into_live`.
    pub fn into_live(self) -> Option<LivePinnedMemoryHandle> {
        std::num::NonZeroU64::new(self.0).map(LivePinnedMemoryHandle)
    }
}

/// A `SenlibPinnedMemoryHandle` KNOWN to be live (non-zero).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LivePinnedMemoryHandle(std::num::NonZeroU64);

impl LivePinnedMemoryHandle {
    /// Widen back to the raw FFI type for `pinned_memory_release`, which
    /// shares the ABI-level type with `flex_senlib_pin_and_map`'s return.
    pub fn as_raw(self) -> SenlibPinnedMemoryHandle {
        SenlibPinnedMemoryHandle(self.0.get())
    }

    pub fn as_u64(self) -> u64 {
        self.0.get()
    }
}

/// One entry of senlib `vfio_backend`'s public `peer_dev_info` vector
/// (`std::vector<std::pair<int, void*>>`) after a cross-port P2P mapping is
/// created — see `p2p_create_mapping`. `aux` is the `int` half of the pair,
/// a senlib-side id for the mapped peer window; `addr` is the mapped
/// address of the peer window (the cross-port target an ordinary H2D DMA
/// can be aimed at, per the #550 experiment design).
///
/// FIXME(pod-verify): whether `aux` is a file descriptor, BAR index, or
/// peer slot — and whether `addr` is DMA-targetable, host-mapped, or both —
/// are exactly the facts the first pod run must establish by logging
/// `peer_dev_info` raw; do not interpret either field before then.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SenlibP2pSegment {
    pub aux: c_int,
    pub addr: *mut c_void,
}

unsafe extern "C" {
    /// Pin and map `num_bytes` of host memory for device DMA access,
    /// optionally backed by a named shm file (`name`; null for anonymous) so
    /// peer processes attach to the same pinned region. Cited:
    /// hdma_shm.cpp:903 `this_pfw_->allocate_pin_and_map(shm_sz_instr)`,
    /// hdma_shm.cpp:907 `allocate_pin_and_map(shm_sz_mgmt + world_size_,
    /// shm_fname_mgmt)`, hdma_shm.cpp:914 `allocate_pin_and_map(
    /// shm_sz_data_p2p, shm_fname_data)` — the HDMA instruction/mgmt/data
    /// buffers cross-process P2P exchange is built on. Irreducible:
    /// page-locking host memory plus IOMMU-mapping it is a senlib/kernel
    /// operation that cannot be emulated in ordinary Rust heap allocation.
    /// Returns a null-equivalent (`handle == 0`) on failure.
    ///
    /// FIXME(pod-verify): the exact post-reorg signature of
    /// `PfWrapper::allocate_pin_and_map` (overload pair vs. defaulted
    /// name parameter) and the concrete wrapper type it returns; the shim
    /// stores the wrapper type-erased and captures `hptr()` at creation, so
    /// only the shim's one construction line depends on the answer.
    fn flex_senlib_pin_and_map(num_bytes: u64, name: *const c_char) -> SenlibPinnedMemoryHandle;

    /// Releases a pinned-memory mapping obtained from `flex_senlib_pin_and_map`
    /// (unpins + unmaps + drops senlib's wrapper). Cited: hdma_shm.hpp
    /// lifetime of the `PinnedMemoryWrapperPtr` members, destructed in
    /// `HdmaShmMgmt::~HdmaShmMgmt` (hdma_shm.hpp:479). Returns 0 on success,
    /// -1 if the release did not reach senlib (unknown handle, or a throw).
    fn flex_senlib_pinned_memory_release(handle: SenlibPinnedMemoryHandle) -> i32;

    /// Returns the host-virtual pointer backing a pinned-memory region
    /// (`PinnedMemoryWrapper::hptr()`, used throughout hdma_shm.cpp, e.g.
    /// line 644). Non-null for any live handle — the allocation succeeded
    /// when the handle was minted.
    fn flex_senlib_pinned_memory_hptr(handle: SenlibPinnedMemoryHandle) -> *mut c_void;

    /// Directly sets the on-device write-done counter for `sid` to `val`.
    /// Cited: host_wdone.cpp:413 `pfw_->interface()->direct_set_wdone(
    /// instr.sid, val)`. Irreducible: this pokes a device-side hardware
    /// register through senlib's PF interface. Returns 0 on success, -1 if
    /// the poke did not reach senlib.
    ///
    /// FIXME(pod-verify): `sid`/`val` parameter widths are taken from the
    /// pre-#530 declaration's citation; check them against the pod's
    /// `pf_interface.hpp` before relying on values beyond u8/i32 range.
    fn flex_senlib_direct_set_wdone(sid: u8, val: i32) -> i32;

    /// Atomically increments the on-device write-done counter for `sid`.
    /// Cited: host_wdone.cpp:417 and host_wdone.cpp:1025
    /// `pfw_->interface()->ainc_wdone(sid)`. Same width caveat as
    /// `flex_senlib_direct_set_wdone` (FIXME(pod-verify)).
    fn flex_senlib_ainc_wdone(sid: u8) -> i32;

    /// Creates the cross-port P2P IOMMU mapping for peer card
    /// `peer_card_index`, sized `len_bytes`: senlib
    /// `vfio_backend::create_iommu_mapping_for_cross_port_p2p_rdma(
    /// PciDevId peer_dev, size_t len)` — pod header
    /// `/opt/ibm/spyre/senlib/include/vfio/1p0/vfio_backend.hpp:73`, mangled
    /// on the pod .so as
    /// `_ZN6senlib2v212vfio_backend44create_iommu_mapping_for_cross_port_
    /// p2p_rdmaENS0_8PciDevIdEm`. The resulting segments land
    /// in the vfio_backend's PUBLIC `peer_dev_info`
    /// (`std::vector<std::pair<int, void*>>`); this call marshals up to
    /// `max_segments` of them into `out_segments` and ALWAYS writes the
    /// total count to `out_count`, so a too-small buffer is reported (retry
    /// with a bigger one), never silently truncated. Idempotent per
    /// (peer, len): the shim memoizes, because nothing above senlib calls
    /// the mapping today and double-mapping a peer is undefined territory.
    /// The peer's `PciDevId` comes from the same `SenPciShared::pf(index)`
    /// lookup the shim's device bring-up and `senlib_senpci_pf_address`
    /// use. Returns 0 on success, -1 on failure.
    ///
    /// FIXME(pod-verify): the vfio_backend's include path/namespace
    /// post-reorg, its reachability from the public `PfInterface` this
    /// shim holds, and whether `peer_dev_info` is per-peer or cumulative
    /// across peers — see `SENLIB_BOUNDARY.md`'s ledger for the recorded
    /// disassembly facts.
    fn flex_senlib_p2p_create_mapping(
        peer_card_index: c_uint,
        len_bytes: u64,
        out_segments: *mut SenlibP2pSegment,
        max_segments: usize,
        out_count: *mut usize,
    ) -> i32;

    /// Exports `size` bytes of the open card's `bar` at `offset` as a
    /// DMABuf file descriptor: `PfInterface::get_dmabuf(int bar,
    /// uint64_t offset, size_t size)` (pod `pf_interface.hpp:111` — the
    /// same header this shim already includes for the interfaces above).
    /// Returns the fd, or -1 on failure. Irreducible: turning a BAR region
    /// into a dma-buf is a driver operation.
    ///
    /// FIXME(pod-verify): whether the real `get_dmabuf` returns the fd
    /// directly or a wrapper object the shim must unwrap; the declaration
    /// assumes the former.
    fn flex_senlib_get_dmabuf(bar: c_int, offset: u64, size: u64) -> c_int;
}

/// Safe wrapper: pin and map `num_bytes` of host memory, optionally backed
/// by the named shm file `name` (built by the caller as a `CString`; pass
/// `None` for anonymous), for device DMA access.
///
/// # Errors
/// Returns `Err` with the shim's message if the pin/map did not reach senlib
/// or failed — the HDMA buffers cross-process exchange depends on are not
/// constructible by silently proceeding.
pub fn pin_and_map(
    num_bytes: usize,
    name: Option<&CStr>,
) -> Result<LivePinnedMemoryHandle, String> {
    // SAFETY: `name`, when present, is a valid NUL-terminated C string for
    // the duration of this call; `num_bytes` is a plain byte count. The
    // returned handle is checked for the 0 failure sentinel via `into_live`
    // before anything stores it.
    let handle = unsafe {
        flex_senlib_pin_and_map(
            num_bytes as u64,
            name.map_or(std::ptr::null(), |n| n.as_ptr()),
        )
    };
    handle.into_live().ok_or_else(last_shim_error)
}

/// Safe wrapper: the host-virtual pointer backing a live pinned-memory
/// region. Non-null by construction (the handle was minted by a successful
/// `pin_and_map`); the region stays valid for the handle's lifetime.
pub fn pinned_memory_hptr(handle: LivePinnedMemoryHandle) -> *mut c_void {
    // SAFETY: `handle` is live (non-zero, from `pin_and_map`'s success path)
    // and has not been released; `hptr()` is a const accessor on senlib's
    // wrapper.
    unsafe { flex_senlib_pinned_memory_hptr(handle.as_raw()) }
}

/// Safe wrapper: release a pinned-memory mapping back to senlib.
///
/// # Errors
/// Returns `Err` with the shim's message if the release did not reach
/// senlib — a pinned region that silently stays mapped leaks page-locked
/// memory, which is precisely the failure that must not be swallowed.
pub fn pinned_memory_release(handle: SenlibPinnedMemoryHandle) -> Result<(), String> {
    // SAFETY: `handle` is the raw FFI type by design (mirrors
    // `flex_senlib_memory_free`'s shape): releasing an unknown/zero handle
    // is senlib's error to report, not a memory-safety hazard — the shim's
    // table rejects it before any senlib call.
    let rc = unsafe { flex_senlib_pinned_memory_release(handle) };
    if rc == 0 {
        Ok(())
    } else {
        Err(last_shim_error())
    }
}

/// Safe wrapper: set the on-device write-done counter for `sid` to `val`
/// (`host_wdone.cpp:413`).
///
/// # Errors
/// Returns `Err` with the shim's message if the register poke did not reach
/// senlib.
pub fn direct_set_wdone(sid: u8, val: i32) -> Result<(), String> {
    // SAFETY: plain integer arguments; assumes the shim runtime was brought
    // up (`flex_shim_init_runtime` succeeded), which the shim checks itself.
    let rc = unsafe { flex_senlib_direct_set_wdone(sid, val) };
    if rc == 0 {
        Ok(())
    } else {
        Err(last_shim_error())
    }
}

/// Safe wrapper: atomically increment the on-device write-done counter for
/// `sid` (`host_wdone.cpp:417,1025`).
///
/// # Errors
/// Returns `Err` with the shim's message if the register poke did not reach
/// senlib.
pub fn ainc_wdone(sid: u8) -> Result<(), String> {
    // SAFETY: plain integer argument; same shim-runtime assumption as
    // `direct_set_wdone`.
    let rc = unsafe { flex_senlib_ainc_wdone(sid) };
    if rc == 0 {
        Ok(())
    } else {
        Err(last_shim_error())
    }
}

/// Safe wrapper: create (or idempotently re-read) the cross-port P2P IOMMU
/// mapping for peer card `peer_card_index`, sized `len_bytes`, returning the
/// mapped segments senlib reported in `peer_dev_info`. Enumerate peers with
/// `sen_pci_card_count()` + `pcie_address_for_card()`; the index passed here
/// is the same PCI enumeration index.
///
/// The first pod run should LOG the returned segments before interpreting
/// them — the `aux` field's meaning and the `addr` field's
/// DMA-vs-host nature are the experiment's open questions.
///
/// # Errors
/// Returns `Err` with the shim's message if the mapping call did not reach
/// senlib or threw, or if the peer was already mapped with a different
/// length.
pub fn p2p_create_mapping(
    peer_card_index: u32,
    len_bytes: u64,
) -> Result<Vec<SenlibP2pSegment>, String> {
    let mut segments = vec![
        SenlibP2pSegment {
            aux: 0,
            addr: std::ptr::null_mut()
        };
        8
    ];
    let mut count = 0usize;
    // SAFETY: `segments` is a valid mutable slice of `max_segments` entries
    // for the duration of the call; `out_count` is a valid out-pointer. The
    // shim is memoized per (peer, len), so the retry below re-marshals the
    // same mapping rather than creating a second one.
    let rc = unsafe {
        flex_senlib_p2p_create_mapping(
            peer_card_index,
            len_bytes,
            segments.as_mut_ptr(),
            segments.len(),
            &mut count,
        )
    };
    if rc != 0 {
        return Err(last_shim_error());
    }
    if count > segments.len() {
        // The mapping produced more segments than the first buffer held
        // (`out_count` always carries the true total): re-marshal with an
        // exactly-sized buffer — idempotent by the shim's memoization.
        segments.resize(
            count,
            SenlibP2pSegment {
                aux: 0,
                addr: std::ptr::null_mut(),
            },
        );
        // SAFETY: same as above, now sized for the reported count.
        let rc = unsafe {
            flex_senlib_p2p_create_mapping(
                peer_card_index,
                len_bytes,
                segments.as_mut_ptr(),
                segments.len(),
                &mut count,
            )
        };
        if rc != 0 {
            return Err(last_shim_error());
        }
    }
    segments.truncate(count);
    Ok(segments)
}

/// Safe wrapper: export `size` bytes of the open card's `bar` at `offset`
/// as a DMABuf file descriptor (`PfInterface::get_dmabuf`,
/// `pf_interface.hpp:111`). The caller owns the returned fd (close it with
/// `libc::close` when done).
///
/// # Errors
/// Returns `Err` with the shim's message if the export did not reach senlib
/// or failed.
pub fn get_dmabuf(bar: i32, offset: u64, size: u64) -> Result<i32, String> {
    // SAFETY: plain integer arguments; the returned fd is a process-owned
    // descriptor the caller closes — no senlib lifetime attaches to it.
    let fd = unsafe { flex_senlib_get_dmabuf(bar, offset, size) };
    if fd >= 0 {
        Ok(fd)
    } else {
        Err(last_shim_error())
    }
}
