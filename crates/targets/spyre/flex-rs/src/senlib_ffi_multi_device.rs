//! FFI boundary for the multi_device (RDMA/HDMA) subsystem.
//!
//! Every declaration here corresponds to a call site verified in
//! `flex-cxx/flex/src/multi_device/**` that reaches senlib's PF/VF wrapper,
//! pinned-memory subsystem, or the mock-device shared-memory transfer layer
//! that stands in for real RDMA hardware submission. See
//! `SENLIB_BOUNDARY_multi-device.md` at the crate root for the rationale
//! behind each boundary choice.
//!
//! All of these are opaque handles/pointers on the Rust side: the safe
//! wrappers in `multi_device.rs` own the bookkeeping (queue slots, bitmaps,
//! counters, message framing) and only reach across this boundary for the
//! literal pin/allocate/transfer/counter operation.

use std::os::raw::{c_int, c_void};

/// Opaque senlib PF/VF wrapper handle.
/// Cited: hdma_shm.cpp:528 `HdmaShmMgmt::HdmaShmMgmt(std::shared_ptr<senlib::v2::PfWrapper> pfw)`
/// and hdma_shm.cpp:549 (VfWrapper overload).
#[repr(C)]
pub struct SenlibDeviceWrapper {
    _private: [u8; 0],
}

/// Opaque pinned host-memory region handle returned by
/// `PfWrapper::allocate_pin_and_map` / `VfWrapper::allocate_pin_and_map`.
/// Cited: hdma_shm.cpp:903,907,914,1033,1045,1107,1112 (`this_pfw_->allocate_pin_and_map`,
/// `this_vfw_->allocate_pin_and_map`).
#[repr(C)]
pub struct SenlibPinnedMemory {
    _private: [u8; 0],
}

/// Opaque handle to the mock-device shared-memory region that RdmaUnit uses
/// as its stand-in for real device memory reachable over RDMA.
/// Cited: rdma_unit.cpp:340 (`rdma_shm_ = std::make_shared<RdmaShm>(local_rank_)`).
#[repr(C)]
pub struct SenlibRdmaShm {
    _private: [u8; 0],
}

unsafe extern "C" {
    /// Pin and map `num_bytes` of host memory for device DMA access, optionally
    /// backed by a named shm file (`name_ptr`/`name_len`, null/0 for anonymous).
    /// Cited: hdma_shm.cpp:903 `this_pfw_->allocate_pin_and_map(shm_sz_instr)`,
    /// hdma_shm.cpp:907 `this_pfw_->allocate_pin_and_map(shm_sz_mgmt + world_size_, shm_fname_mgmt)`,
    /// hdma_shm.cpp:914 `this_pfw_->allocate_pin_and_map(shm_sz_data_p2p, shm_fname_data)`.
    /// Irreducible: pinning host memory for device DMA is a senlib/kernel-driver
    /// operation (page-locking + IOMMU mapping) that cannot be emulated in
    /// ordinary Rust heap allocation.
    pub fn flex_senlib_pin_and_map(
        device: *mut SenlibDeviceWrapper,
        num_bytes: usize,
        name_ptr: *const u8,
        name_len: usize,
    ) -> *mut SenlibPinnedMemory;

    /// Releases a pinned-memory mapping obtained from `flex_senlib_pin_and_map`.
    /// Cited: hdma_shm.hpp lifetime of `PinnedMemoryWrapperPtr` members (destructed
    /// in `HdmaShmMgmt::~HdmaShmMgmt`, hdma_shm.hpp:479).
    pub fn flex_senlib_pinned_memory_release(mem: *mut SenlibPinnedMemory);

    /// Returns the host-virtual pointer backing a pinned-memory region
    /// (`PinnedMemoryWrapper::hptr()`, used throughout hdma_shm.cpp, e.g. line 644).
    pub fn flex_senlib_pinned_memory_hptr(mem: *mut SenlibPinnedMemory) -> *mut c_void;

    /// Directly sets the on-device write-done counter for `sid` to `val`.
    /// Cited: host_wdone.cpp:413 `pfw_->interface()->direct_set_wdone(instr.sid, val)`.
    /// Irreducible: this pokes a device-side hardware register through senlib's
    /// PF interface; there is no host-side equivalent.
    pub fn flex_senlib_direct_set_wdone(
        device: *mut SenlibDeviceWrapper,
        sid: u8,
        val: i32,
    ) -> c_int;

    /// Atomically increments the on-device write-done counter for `sid`.
    /// Cited: host_wdone.cpp:417 and host_wdone.cpp:1025 `pfw_->interface()->ainc_wdone(sid)`.
    pub fn flex_senlib_ainc_wdone(device: *mut SenlibDeviceWrapper, sid: u8) -> c_int;

    /// Opens or creates the mock-RDMA shared-memory region for `local_rank`.
    /// Cited: rdma_unit.cpp:340 `rdma_shm_ = std::make_shared<RdmaShm>(local_rank_)`.
    pub fn flex_senlib_rdma_shm_open(local_rank: i64) -> *mut SenlibRdmaShm;

    /// Releases a mock-RDMA shared-memory handle.
    pub fn flex_senlib_rdma_shm_close(shm: *mut SenlibRdmaShm);

    /// Performs the literal device-to-device data transfer: copies `size_bytes`
    /// from `local_ptr` into the remote rank's mapped device-memory region at
    /// translated offset `xlat`. Cited: rdma_unit.cpp:415-416
    /// `remote_ptr = rdma_devmems_.at(tid)->DeviceMemoryPointer(xlat); memcpy(remote_ptr, local_ptr, size_bytes)`.
    /// Irreducible: this is the RDMA write itself (or its mock-hardware stand-in);
    /// everything above it (validation, xseg lookup, sizing) is flex bookkeeping
    /// and is ported natively.
    pub fn flex_senlib_rdma_copy_to_remote(
        shm: *mut SenlibRdmaShm,
        target_rank: u32,
        xlat_bytes: u64,
        local_ptr: *const u8,
        size_bytes: usize,
    ) -> c_int;

    /// Atomically increments the remote write-done counter `(tid, sid)` and
    /// returns its new value. Cited: rdma_unit.cpp:499
    /// `rdma_shm_->WriteDoneCounterInc(tid, sid)`.
    pub fn flex_senlib_rdma_wdone_inc(shm: *mut SenlibRdmaShm, tid: u32, sid: u32) -> i64;

    /// Reads the current write-done counter `(tid, sid)` and, if it observes
    /// a positive value, DECREMENTS it by one before returning -- the
    /// pre-decrement value is what's returned. This is a mutating
    /// read-and-consume operation, not a pure read: a caller that polls this
    /// repeatedly is draining the counter by one unit per successful poll.
    /// Cited: rdma_unit.cpp:293-306 `RdmaShm::WriteDoneCounterCheck`
    /// ("Decrements $sid's WriteDone counter if value is larger than zero
    /// then returns the previous value"), consumed at rdma_unit.cpp:457
    /// `rdma_shm_->WriteDoneCounterCheck(tid, sid)`.
    pub fn flex_senlib_rdma_wdone_check(shm: *mut SenlibRdmaShm, tid: u32, sid: u32) -> i64;

    /// Reads the translation-segment (Xseg) register for `(tid, sid)`.
    /// Cited: rdma_unit.cpp:409 `rdma_shm_->XsegBytes(tid, sid)`.
    pub fn flex_senlib_rdma_xseg_bytes(shm: *mut SenlibRdmaShm, tid: u32, sid: u32) -> u64;

    /// Returns whether the Xseg register for `(tid, sid)` has its valid bit set.
    /// Cited: rdma_unit.cpp:402 `rdma_shm_->XsegValid(tid, sid)`.
    pub fn flex_senlib_rdma_xseg_valid(shm: *mut SenlibRdmaShm, tid: u32, sid: u32) -> bool;

    /// Writes the raw Xseg register for `(tid, sid)`.
    /// Cited: rdma_unit.cpp:524 `rdma_shm_->Xseg(tid, sid) = xseg`.
    pub fn flex_senlib_rdma_xseg_write(shm: *mut SenlibRdmaShm, tid: u32, sid: u32, xseg: u32);
}
