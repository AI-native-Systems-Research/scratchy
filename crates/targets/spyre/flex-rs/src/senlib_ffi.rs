//! Consolidated senlib FFI boundary.
//!
//! Each parallel phase produced its own `senlib_ffi_<label>.rs` file (per the
//! task brief, to avoid symbol collisions while phases ran concurrently).
//! This module is the Assembly-phase consolidation point: it re-exports each
//! phase's FFI surface under one root, and is also where genuine cross-phase
//! duplicates were found and merged into a single canonical declaration.
//!
//! See `SENLIB_BOUNDARY.md` at the crate root for the full boundary
//! rationale; this file only carries module wiring + the dedup fix below.
//!
//! ## Duplicate found and fixed: "how many Spyre cards are there"
//!
//! Three phases independently reached `senlib::v2::SenPci::ncards()` (PCI
//! enumeration of physical cards) from three different call sites in the
//! C++ source, and each declared its own
//! `extern "C"` binding for it:
//!   - the runtime-core phase, as `flex_senlib_pci_ncards` in
//!     `senlib_ffi_runtime.rs` (from `RuntimeContext`/`getNumDevices`);
//!   - the config/util phase, as `senlib_senpci_ncards` in
//!     `senlib_ffi_config_util.rs` (from `FlexConfig::FlexSpyreDevices`);
//!   - this file's own pre-split placeholder, as part of a since-removed
//!     `flex_senlib_query_topology` (topology-phase card+HBM-layout query).
//!
//! UPDATE (2026-08-15): `flex_senlib_query_topology` was never implemented in
//! `cxx-shim/flex_shim.cpp` (no such symbol exists in the real SDK either —
//! see the "REAL LINK-TARGET AUDIT" below) and real-hardware testing hit a
//! `FlexAllocator` OOM traced to `fxa_rust_abi.rs::build_runtime()` sizing its
//! single domain off a 32 GiB placeholder instead of the card's real HBM
//! capacity. It is replaced below by `flex_senlib_device_memory_size`, a
//! one-line real call (`DeviceHandle::GetDmpaSize()`) matching exactly what
//! the real C++ path does (`device_memory_topology.cpp::populateDomainTopology`
//! fed by `getDeviceMemorySize()`) — this crate's 1p0 target has exactly one
//! domain, so a full multi-domain query was never actually needed. The other
//! two —
//! `flex_senlib_pci_ncards` and `senlib_senpci_ncards` — are ASSUMED to be
//! the same zero-argument `ncards() -> count` call declared twice under
//! different names, both citing `senlib::v2::SenPci`/`SenPciShared::ncards()`
//! from their respective call sites in `flex_config.cpp`. This is an
//! unverified assumption, not a confirmed fact: `flex-cxx` does not vendor
//! senlib itself, so there is no header available in this audit to check
//! that `SenPci::ncards()` and `SenPciShared::ncards()` really are the exact
//! same underlying call rather than, say, two related-but-distinct methods
//! on different senlib-side wrapper types. `senlib_ffi_runtime::flex_senlib_pci_ncards`
//! is kept as the one canonical declaration (it has the closer-to-source name); the
//! config/util phase's `senlib_senpci_ncards` extern was deleted from
//! `senlib_ffi_config_util.rs` and its safe wrapper (`sen_pci_card_count`)
//! now calls the canonical declaration instead. `flex_config.rs`, which
//! calls `sen_pci_card_count()`, needed no changes.
//!
//! The old pre-split placeholder externs that used to live directly in this
//! file (`flex_senlib_stream_create/destroy/submit/poll_completions`,
//! `SenlibStreamHandle`, `RawControlBlock`) had zero callers anywhere in the
//! crate — they were superseded by the scheduler phase's real,
//! call-site-verified `senlib_ffi_scheduler.rs` (`senlib_cbi_*`/`senlib_rbi_*`,
//! used by `scheduler.rs`/`stream.rs`) before this file was consolidated, and
//! have been removed rather than re-exported as dead code.

pub use crate::senlib_ffi_allocator as allocator;
pub use crate::senlib_ffi_config_util as config_util;
pub use crate::senlib_ffi_controlblocks as controlblocks;
pub use crate::senlib_ffi_multi_device as multi_device;
pub use crate::senlib_ffi_runtime as runtime;
pub use crate::senlib_ffi_scheduler as scheduler;

/// Canonical "how many Spyre cards" call — see module doc. Re-exported here
/// under the name most other code refers to it by; the real declaration
/// lives in `senlib_ffi_runtime.rs` since that is the phase whose call site
/// (`RuntimeContext`/`getNumDevices`) is the most direct match to the C++
/// symbol name.
pub use crate::senlib_ffi_runtime::flex_senlib_pci_ncards;

unsafe extern "C" {
    /// Brings up the real device runtime (idempotent, thread-safe on the
    /// C++ side via `std::call_once` — see `cxx-shim/flex_shim.cpp`).
    /// MUST be called (and succeed, i.e. return 0) before any other
    /// `senlib_ffi*` function in this crate is called; every other function
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
    /// `IommuMapping` bookkeeping object (`iomap->writable_`) returned to
    /// the caller. This shim function mirrors that: the parameter is
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

// ---------------------------------------------------------------------
// REAL LINK-TARGET AUDIT (2026-08-15, against the actual on-pod SDK at
// /opt/ibm/spyre) — see also senlib_ffi_allocator.rs and
// senlib_ffi_scheduler.rs for the two other files this applies to.
//
// Every `flex_senlib_*`/`senlib_*` symbol declared across
// senlib_ffi{,_allocator,_runtime,_multi_device,_scheduler,_controlblocks,
// _config_util}.rs is a NAME THAT DOES NOT EXIST in any installed SDK
// library (`nm -D --defined-only` on every .so under
// /opt/ibm/spyre/{senlib,runtime,deeptools}/lib turns up zero matches for
// `flex_senlib` or `senlib_control_block`/`senlib_response_block`). These
// modules' own doc comments already say why: "Not built/linked on this
// machine (no senlib headers/libs present)" — they were written against
// nothing, as an aspirational C shim contract that nobody has yet written.
// A real senlib/flex install is a C++ API (mangled `_Z...` symbols); there
// is no C shim anywhere in the SDK exporting these plain names, so linking
// scratchy's `spyre_sdk_abi.cpp`-replacement against a real build fails at
// `ld` with "undefined reference" for every one of them, not a runtime bug.
//
// The good news, confirmed by reading the real headers on the pod: a
// *legitimate*, minimal, hand-written C++ shim TU (compiled alongside these
// declarations, exporting exactly these names via `extern "C"`) can
// implement most of them by delegating to real, PUBLIC, `FLEX_EXPORT` flex
// C++ API reachable from a live `flex::RuntimeContext` — not by
// reimplementing raw hardware bring-up:
//
//   flex::getFlexRuntimeContext()->getDeviceHandle()
//     -> shared_ptr<const flex::DeviceHandle>, which itself exposes (all
//        public, `flex/device_types/device_handle.hpp`):
//          ->GetDeviceMemoryAllocator()  // flex::DeviceMemoryAllocator
//              ->TryAllocate(&alloc, nbytes, context_id, alloc_type, domain)
//              ->Free(alloc.get())
//            i.e. a real target for flex_senlib_memory_allocate/_free —
//            NOTE: real signature takes a context_id and returns a
//            DeviceMemoryAllocationPtr (dmpa + size + alignment + VF alloc
//            index), richer than this crate's `SenlibAllocationHandle(u64)`;
//            the shim would need to own the DeviceMemoryAllocationPtr behind
//            the u64 handle (e.g. a process-wide handle table), not just
//            reinterpret a pointer.
//          ->AsyncDmaICbi() / ->AsyncDmaOCbi() / ->ComputeCbi()
//              -> senlib::ControlBlockInterface* (exactly the type
//                 senlib_ffi_scheduler.rs's `SenlibControlBlockInterfaceHandle`
//                 is meant to be an opaque handle to)
//          ->Rbi() -> senlib::ResponseBlockInterface*
//
// `senlib::v2::SenPci::ncards()` (`senlib/1p0/senpci.hpp`) is a plain static
// method with no bring-up prerequisite at all — a trivial, safe real target
// for `flex_senlib_pci_ncards`.
//
// THE ACTUAL BLOCKER, not fixable this way: `PfBackend`/`VfBackend` in
// `scheduler.rs` never build a `SentientSoc::V1::ControlBlockSBF` (or
// `ResponseBlockSBF`) at all — `HardwareSubmit::submit(pipeline, num_cbs,
// callback)` carries no DMA/compute payload whatsoever (no addresses, no
// sizes, no kernel name, no bootstrap offset). So even with every senlib
// symbol above wired to a real, correct C++ shim, calling into real hardware
// through this path would ring the doorbell on control blocks that were
// never populated with the actual work — the CB wire-format encoder (what
// `flex/src/control_blocks/*` does in the real C++: turning `DmaParams`/
// `ComputeParams` into CTRL/DMI/DMO/CMPT/XLAT section bytes per
// `hal/1p0/{dma,compute,translation,ctrl,r5}_control_block_section_sbf.hpp`)
// does not exist anywhere in this crate. This is the real, structural gap —
// a whole unimplemented subsystem, not a signature mismatch — and is why
// `SenlibControlBlockSbf`/`SenlibResponseBlockSbf`'s sizes (see
// senlib_ffi_scheduler.rs, now fixed to the real 128/64 bytes) had never
// been exercised against anything: nothing in this crate has ever populated
// one.
//
// UPDATE (2026-08-15): the CB/RB wire-format encoder now exists —
// `crate::control_block_wire` (`ControlBlockWire`/`ResponseBlockWire`),
// ported from the real headers pulled off the pod
// (`hal/1p0/{control_block,ctrl_section,dma_control_block_section{,_base},
// compute_control_block_section,translation_control_block_section,
// response_block,return_section,virtual_address}_sbf.hpp` under
// `/opt/ibm/spyre/senlib/include/`), covering the graph-free compute/DMA/fill
// CB shapes (`CreateGraphFreeCompute`/`CreateComputeDataTransfer`/
// `CreateFillDma`) this crate actually needs — RDMA/HDMA/R5 section encoding
// is intentionally not ported (this runtime never emits it). `scheduler.rs`'s
// `HardwareSubmit::submit` now takes `&[ControlBlockWire]` instead of a bare
// CB count, and `PfBackend`/`VfBackend::submit_dma`/`submit_compute` build
// real CBs via `build_dma_cb`/`build_compute_cb` before calling it.
// `senlib_ffi_scheduler::SenlibControlBlockSbf` has a `From<ControlBlockWire>`
// impl (byte-identical layout) so the two link up directly.
//
// STILL OPEN: the senlib symbol-name problem described above is unchanged —
// `senlib_control_block_interface_queue_control_blocks_sbf` et al. are still
// not real C ABI names; a hand-written C++ shim TU (delegating to the real
// `flex::DeviceHandle::AsyncDmaICbi()/AsyncDmaOCbi()/ComputeCbi()/Rbi()` +
// `senlib::ControlBlockInterface::QueueControlBlocksSBF`/
// `senlib::ResponseBlockInterface::ReceiveResponsesSBF`, per this file's own
// citations above) still needs to be written and linked before this crate
// can reach real hardware. `DeviceAddressResolver` (scheduler.rs) — the
// `CompositeAddress` -> physical-address lookup the encoder needs — also
// still needs a production implementation backed by
// `FlexAllocator::get_id_to_region_map()`; no such impl exists in this crate
// yet, so `PfBackend`/`VfBackend` cannot be constructed end-to-end without
// one being supplied by the caller.
// ---------------------------------------------------------------------
//
// UPDATE (2026-08-15, later same day): both remaining gaps are closed.
//
// `cxx-shim/flex_shim.cpp` is the real, hand-written C++ shim TU this audit
// called for — it exports every `flex_senlib_*`/`senlib_*`/`flex_shim_*`
// name declared across these `senlib_ffi*.rs` files as a plain `extern "C"`
// symbol, each a one-line marshal onto the real
// `flex::DeviceHandle`/`senlib::ControlBlockInterface`/
// `senlib::ResponseBlockInterface` methods cited in this file's and
// `senlib_ffi_scheduler.rs`'s doc comments. It links against the real
// `libflex.so`/senlib on the pod (`/opt/ibm/spyre/{runtime,senlib}/lib`) and
// is compiled by `build.rs` only when `FLEX_RS_BUILD_CXX_SHIM=1` is set in
// the environment (unset on this laptop, which has none of the SDK headers
// — `cargo check`/`build`/`clippy` here never invoke a C++ compiler).
//
// `flex_shim_cbi_for_pipeline`/`flex_shim_rbi_for_pipeline` (new, in
// `senlib_ffi_scheduler.rs`) are the shim entry points that turn a
// `PipelineId` into the actual `ControlBlockInterface*`/
// `ResponseBlockInterface*` opaque handle — the missing piece needed before
// `senlib_cbi_capacity`/`senlib_cbi_queue_control_blocks_sbf`/
// `senlib_rbi_receive_responses_sbf` have anything real to be called with.
// `flex_shim_init_runtime` (this file) brings up the real device and caches
// its `DeviceHandle` process-wide; it must be called once, successfully,
// before any other function here.
//
// `scheduler::FlexAllocatorAddressResolver` is the production
// `DeviceAddressResolver` this audit's second gap called for — it resolves
// `Chunk::addr.region_id` through a live `FlexAllocator`'s
// `get_id_to_region_map()` and reads the region's real device physical
// address off `MemoryRegion::data().device_address_bytes()`
// (`DeviceMemoryAllocation::DmpaAsBytes()`, surfaced through
// `flex_senlib_memory_allocate`'s return value). `PfBackend`/`VfBackend` can
// now be constructed end-to-end by a caller that owns a
// `Arc<Mutex<FlexAllocator>>` and an `Arc<dyn HardwareSubmit>` built on top
// of the shim's `senlib_ffi_scheduler` functions.
// ---------------------------------------------------------------------
