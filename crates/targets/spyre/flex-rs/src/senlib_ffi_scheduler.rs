//! The ONLY `extern "C"`/senlib-touching declarations for the scheduler subsystem.
//!
//! Every call site below was verified against `flex-cxx`. See
//! `SENLIB_BOUNDARY_scheduler.md` at the crate root for the full rationale of
//! why each one is irreducible (cannot be pushed further down into native
//! Rust) — this file only carries the declarations + call-site citations.
//!
//! Boundary calls identified for this subsystem:
//!
//! 1. `senlib::ControlBlockInterface::capacity()` — queried from
//!    `RuntimeScheduler::getQueueCapacity()`
//!    (`flex/src/runtime_stream/runtime_scheduler.cpp:1025`) via
//!    `device_handle_->ComputeCbi()/AsyncDmaICbi()/AsyncDmaOCbi()`
//!    (`flex/include/flex/device_interface/device_handle.hpp:206,212,218`).
//!    Reports the hardware CB queue depth; nothing on the host side can know
//!    this number, it comes from firmware/hardware configuration.
//!
//! 2. `senlib::ControlBlockInterface::QueueControlBlocksSBF()` — the literal
//!    "ring the doorbell" call. In this subsystem's own files it is called
//!    directly by `MockRuntimeScheduler::submitCompute`
//!    (`flex/src/runtime_stream/1p0/mock_runtime_scheduler.cpp:322`) via
//!    `device_handle_->ComputeCbi()`. The PF/VF backends reach the same
//!    senlib entry point indirectly through `ResponseWorker::QueueCbs()`
//!    (`flex/src/control_blocks/message_processing/response_worker.cpp:1214,1268`,
//!    calling `cbi->QueueControlBlocksSBF(...)`), but `ResponseWorker` itself
//!    belongs to the `control_blocks/message_processing/` subsystem (a
//!    different parallel phase's files), so this crate treats
//!    `ResponseWorker::QueueCbs` as an opaque native-Rust dependency (like
//!    `DeviceHandle`) rather than re-declaring senlib bindings for it here.
//!
//! 3. `senlib::ResponseBlockInterface::ReceiveResponsesSBF()` — mock-only
//!    synchronous drain of response blocks after `QueueControlBlocksSBF`,
//!    `flex/src/runtime_stream/1p0/mock_runtime_scheduler.cpp:329`, via
//!    `device_handle_->Rbi()` (`device_handle.hpp:224`).
//!
//! Everything else this subsystem does — fence/barrier bookkeeping, in-flight
//! counters, pipeline-switch detection, DMA chunk looping, P2P/RDMA parameter
//! validation, control-block *orchestration* (which fields get set, in what
//! order) — is flex's own logic and is ported natively in `scheduler.rs`.
//! Building the literal CB bit layout (SBF structures) is the
//! `control_blocks/` subsystem's own scope (a different parallel phase); this
//! file/module does not redeclare or re-implement that layer, it only models
//! the narrow senlib-facing surface `RuntimeScheduler` itself touches.

use std::ffi::c_void;

/// Opaque handle mirroring `senlib::ControlBlockInterface*`. Never
/// dereferenced field-by-field in this crate — only passed back into the
/// three `extern "C"` entry points below.
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
/// interprets its bytes; the `control_blocks/` subsystem's own file is
/// responsible for populating it.
///
/// SIZE VERIFIED 2026-08-15 against the real SDK on-pod
/// (`hal/1p0/control_block_sbf.hpp`: `constexpr uint64_t CB_NUM_BYTES = 128;`,
/// six 8/16-byte sections — CTRL/R5/DMI/DMO/CMPT/XLAT). The previous `[u8; 256]`
/// here was never checked against a real senlib header (this whole module's
/// own doc says as much: "Not built/linked on this machine") and was simply
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
/// Same "never verified against a real header" issue as `SenlibControlBlockSbf`
/// above — was `[u8; 256]`, four times too large.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SenlibResponseBlockSbf {
    pub bytes: [u8; 64],
}

// SEALED: these declarations are module-private on purpose. The only way to
// reach senlib from the rest of the crate is through the safe `Result`-returning
// wrappers below, so a call site cannot skip a status check or ignore a failure
// — the port's original `void` submit + swallowed exception is unrepresentable
// here now. Do not add `pub` to anything in this block.
unsafe extern "C" {
    /// Returns the `senlib::ControlBlockInterface*` for `pipeline`
    /// (0=compute, 1=async-dma-in, 2=async-dma-out — see
    /// `cxx-shim/flex_shim.cpp`'s `FlexShimPipeline` enum, kept in sync by
    /// hand with `PipelineId` here) as an opaque handle, or a null handle if
    /// the shim runtime has not been initialized (see
    /// `flex_shim_init_runtime` below) or `pipeline` is out of range.
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
    fn flex_shim_last_error() -> *const std::ffi::c_char;

    /// Port of `senlib::ControlBlockInterface::QueueControlBlocksSBF(const ControlBlockSBF*, uint64_t)`
    /// (`senlib/1p0/control_block_interface.hpp:46,76`).
    /// Call site: `mock_runtime_scheduler.cpp:322` (direct); reached indirectly
    /// by PF/VF through `response_worker.cpp:1214,1268` (out of this
    /// subsystem's file scope — see module doc above).
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
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
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

/// Safe wrapper: ring the doorbell — hand a batch of already-built control
/// blocks to senlib for hardware submission. `cbs` must have been fully
/// populated by the `control_blocks/` subsystem before this call.
///
/// # Errors
/// Returns `Err(())` if the submission did not reach senlib. The real call site
/// (`response_worker.cpp:1268`) makes an unguarded `void` call, so a senlib
/// throw there propagates out of the worker; the caller must FAIL the request
/// rather than observe-and-ignore. Swallowing it leaves the caller's
/// already-incremented in-flight count waiting on a CB hardware never received.
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
/// Returns `Err(())` if the call did not reach senlib. The real method is
/// `void` and its call site is unguarded, so a senlib throw there propagates
/// out of the response-completion thread; the caller decides how to surface
/// that (see `scheduler.rs`).
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
