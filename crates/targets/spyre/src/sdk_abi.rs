// SPDX-License-Identifier: Apache-2.0
//! Rust bindings for `csrc/spyre_sdk_abi.cpp` — the one C-ABI seam to the IBM Spyre SDK
//! (flex device runtime + the two deeptools host functions). Everything above this module is
//! Rust; everything below it is the SDK.
//!
//! The raw externs are wrapped immediately in typed handles: a [`DevAddr`] is an owned
//! `flex::CompositeAddress*` (freed on drop), a [`Stream`] is a created `RuntimeStream*`
//! (destroyed on drop — which is also the per-forward fence-deque drain the leak fix relies on).

use std::ffi::CString;
use std::os::raw::{c_char, c_void};

unsafe extern "C" {
    fn fxa_init_runtime() -> i32;
    fn fxa_prewarm();
    fn fxa_stream_create() -> *mut c_void;
    fn fxa_stream_destroy(stream: *mut c_void);
    fn fxa_sync(stream: *mut c_void) -> i32;
    fn fxa_runtime_reset() -> i32;
    fn fxa_alloc(nbytes: u64, program: i32) -> *mut c_void;
    fn fxa_addr_free(addr: *mut c_void);
    fn fxa_addr_total_size(addr: *const c_void) -> u64;
    fn fxa_addr_is_single_chunk(addr: *const c_void) -> i32;
    fn fxa_addr_chunk0(
        addr: *const c_void,
        region: *mut u64,
        offset: *mut u64,
        size: *mut u64,
        domain: *mut u64,
    ) -> i32;
    fn fxa_addr_from_chunk(region: u64, offset: u64, size: u64, domain: u64) -> *mut c_void;
    fn fxa_h2d(stream: *mut c_void, host: *const c_void, size: u64, addr: *const c_void) -> i32;
    fn fxa_d2h(stream: *mut c_void, host: *mut c_void, size: u64, addr: *const c_void) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn fxa_compute(
        stream: *mut c_void,
        prog: *const c_void,
        tensor_allocs: *const *const c_void,
        n: u64,
        kernel_name: *const c_char,
        bootstrap_offset: u64,
        tensor_byte_offsets: *const u64,
        n_tbo: u64,
        pipeline_barrier: i32,
    ) -> i32;
    fn fxa_prog_offset_base() -> u64;
}

/// Bring the process-wide flex runtime up (idempotent, ~5.7s cold). `Err` carries the rc.
pub fn init_runtime() -> Result<(), i32> {
    let rc = unsafe { fxa_init_runtime() };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

/// Kick the runtime bring-up on a detached background thread (overlaps host weight load).
pub fn prewarm_runtime() {
    unsafe { fxa_prewarm() }
}

/// Tear the whole RuntimeContext down — the one-shot leak probe. BREAKS every live session.
pub fn runtime_reset() -> Result<(), i32> {
    let rc = unsafe { fxa_runtime_reset() };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

/// `flex::PROG_OFFSET_BASE` — the device VA base every `job_bin_ptr` is rebased by at launch.
pub fn prog_offset_base() -> u64 {
    unsafe { fxa_prog_offset_base() }
}

/// One created `flex::RuntimeStream`. Dropping it destroys the stream, which drains the
/// scheduler's per-stream fence deques (the resident-session leak fix recreates one per forward).
/// [`Stream::h2d`]'s refusal code for a transfer longer than the device region it targets. Distinct
/// from every `fxa_*` rc (those are 0 or negative small values from the runtime) so a caller
/// rendering `rc={rc}` cannot confuse "we refused" with "the card refused".
pub const RC_H2D_OVERRUNS_REGION: i32 = -0x4844; // 'HD'

pub struct Stream(*mut c_void);

/// ⛔⛔⛔ STREAM CREATION IS SERIALIZED, AND IT IS NOT A PERFORMANCE HEDGE.
///
/// `fxa_stream_create` lazily initialises the global flex runtime (`fxa_init_runtime()`) and then calls
/// `g_rt->createStream()`. The prefill ladder builds its rungs IN PARALLEL — measured on granite-3.1-8b:
/// `prefill ladder: 20 rung(s) ready in 6.19s wall — built in parallel` — so twenty-odd threads enter that
/// function at once on every load.
///
/// MEASURED: model load aborts intermittently, roughly one run in three, with `Signal Received: 6` and NO
/// message, symbolized (`addr2line` on the backtrace the abort prints) to
/// `fxa_stream_create ← Executor::prepare ← SuperDscSession::new_inner`. The C++ wrapper already catches
/// every exception and returns null, so flex is ABORTING INTERNALLY — nothing can be caught or retried, and
/// the process dies before serving a single token, taking a `scr batch` run's output file with it.
///
/// A concurrent lazy-init plus a runtime call with no documented thread-safety is the shape of that abort, so
/// the calls are taken one at a time. The cost is bounded and small — 20 creations against a 6.19 s parallel
/// build — and correctness here is not tradeable: an aborted load is a lost run.
static STREAM_CREATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

impl Stream {
    pub fn create() -> Option<Stream> {
        // A poisoned lock still serializes (nothing is protected but the CALL), so recover rather than
        // panic: a panic here would turn one thread's failure into the whole load's.
        let _serialize = STREAM_CREATE.lock().unwrap_or_else(|e| e.into_inner());
        let p = unsafe { fxa_stream_create() };
        if p.is_null() { None } else { Some(Stream(p)) }
    }

    pub fn synchronize(&self) -> Result<(), i32> {
        let rc = unsafe { fxa_sync(self.0) };
        if rc == 0 { Ok(()) } else { Err(rc) }
    }

    /// ⛔⛔⛔ ASYNC H2D OF `bytes` INTO `addr`. THE LENGTH IS THE HOST SLICE'S — THERE IS NO OTHER FORM.
    ///
    /// This is the only door, and that is the point. It replaces an
    /// `h2d_raw(host: *const u8, size: u64, addr)` whose `size` came from a DIFFERENT OBJECT than
    /// `host` — and every one of its four callers sourced that size from `addr.total_size()`, the
    /// DEVICE region's alignment-padded extent. Whenever the pad exceeded the host buffer, the DMA
    /// pinned past its end.
    ///
    /// That over-read was real: `alloc_ops_at` sent `addr.total_size()` bytes out of
    /// `op.code.init_binary`. It was NOT, however, the cause of the failure that found it — see that
    /// function, where a granite-3.1-8b program's VFIO pin returned `EFAULT` and cost prefill rung
    /// m=11. Fixing the length alone moved that failure zero bytes. Both are defects; only one was
    /// the bug being chased, and this doc used to claim otherwise.
    ///
    /// A raw pointer plus an independent length cannot be made safe by documentation — the old form
    /// said "`host` must point at `size` readable bytes" and four callers still got it wrong. A
    /// slice carries its own length, so the over-read is now UNREPRESENTABLE: a caller that wants
    /// fewer bytes than the region slices the buffer, and the slice bound is the proof.
    ///
    /// The device side is the other half of the same pairing, so it is checked here rather than
    /// trusted: a transfer LONGER than the region would write past what was allocated.
    pub fn h2d(&self, bytes: &[u8], addr: &DevAddr) -> Result<(), i32> {
        if bytes.len() as u64 > addr.total_size() {
            return Err(RC_H2D_OVERRUNS_REGION);
        }
        let rc = unsafe {
            fxa_h2d(
                self.0,
                bytes.as_ptr() as *const c_void,
                bytes.len() as u64,
                addr.raw(),
            )
        };
        if rc == 0 { Ok(()) } else { Err(rc) }
    }

    /// Async D2H into `bytes` from the device window `addr`.
    pub fn d2h(&self, bytes: &mut [u8], addr: &DevAddr) -> Result<(), i32> {
        let rc = unsafe {
            fxa_d2h(
                self.0,
                bytes.as_mut_ptr() as *mut c_void,
                bytes.len() as u64,
                addr.raw(),
            )
        };
        if rc == 0 { Ok(()) } else { Err(rc) }
    }

    /// One ComputeOnDevice launch (async). `tensor_allocs` is POSITIONAL — entry i backs the
    /// program's logical segment i.
    pub fn compute(
        &self,
        prog: &DevAddr,
        tensor_allocs: &[&DevAddr],
        kernel_name: &str,
        bootstrap_offset: u64,
        tensor_byte_offsets: &[u64],
        pipeline_barrier: bool,
    ) -> Result<(), i32> {
        let ptrs: Vec<*const c_void> = tensor_allocs.iter().map(|a| a.raw()).collect();
        let name = CString::new(kernel_name).unwrap_or_default();
        let rc = unsafe {
            fxa_compute(
                self.0,
                prog.raw(),
                ptrs.as_ptr(),
                ptrs.len() as u64,
                name.as_ptr(),
                bootstrap_offset,
                tensor_byte_offsets.as_ptr(),
                tensor_byte_offsets.len() as u64,
                pipeline_barrier as i32,
            )
        };
        if rc == 0 { Ok(()) } else { Err(rc) }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { fxa_stream_destroy(self.0) }
    }
}

// The stream handle is only ever used from the session's own thread, but the session moves
// across worker threads between forwards, exactly as the C++ session pointer did.
unsafe impl Send for Stream {}

/// One chunk's identity — what a shifted or sliced sub-address is built from.
#[derive(Clone, Copy, Debug)]
pub struct Chunk0 {
    pub region: u64,
    pub offset: u64,
    pub size: u64,
    pub domain: u64,
}

/// An owned `flex::CompositeAddress*`. Either a real allocation ([`DevAddr::alloc`] — device
/// memory lives until process exit; the handle frees only the descriptor, matching the C++
/// executor which never freed regions) or a non-owning window ([`DevAddr::from_chunk`]).
pub struct DevAddr(*mut c_void);

/// Which device memory type an allocation lands in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemKind {
    Tensor,
    Program,
}

impl DevAddr {
    pub fn alloc(nbytes: u64, kind: MemKind) -> Option<DevAddr> {
        let p = unsafe { fxa_alloc(nbytes, (kind == MemKind::Program) as i32) };
        if p.is_null() { None } else { Some(DevAddr(p)) }
    }

    /// A new single-chunk address naming the window (region, offset, size, domain) — the shifted
    /// per-layer base / sliced-DMA form.
    pub fn from_chunk(c: Chunk0) -> Option<DevAddr> {
        let p = unsafe { fxa_addr_from_chunk(c.region, c.offset, c.size, c.domain) };
        if p.is_null() { None } else { Some(DevAddr(p)) }
    }

    pub fn total_size(&self) -> u64 {
        unsafe { fxa_addr_total_size(self.0) }
    }

    pub fn is_single_chunk(&self) -> bool {
        unsafe { fxa_addr_is_single_chunk(self.0) != 0 }
    }

    pub fn chunk0(&self) -> Option<Chunk0> {
        let (mut r, mut o, mut s, mut d) = (0u64, 0u64, 0u64, 0u64);
        let rc = unsafe { fxa_addr_chunk0(self.0, &mut r, &mut o, &mut s, &mut d) };
        (rc == 0).then_some(Chunk0 {
            region: r,
            offset: o,
            size: s,
            domain: d,
        })
    }

    /// A shifted single-chunk window `sh` bytes into this region (size = remaining bytes, the
    /// same `c0.size > sh ? c0.size - sh : c0.size` rule the C++ per-layer advance used).
    pub fn shifted(&self, sh: u64) -> Option<DevAddr> {
        let c0 = self.chunk0()?;
        let nsz = if c0.size > sh { c0.size - sh } else { c0.size };
        DevAddr::from_chunk(Chunk0 {
            region: c0.region,
            offset: c0.offset + sh,
            size: nsz,
            domain: c0.domain,
        })
    }

    /// A sliced single-chunk window `[off, off+len)` of this region.
    pub fn window(&self, off: u64, len: u64) -> Option<DevAddr> {
        let c0 = self.chunk0()?;
        DevAddr::from_chunk(Chunk0 {
            region: c0.region,
            offset: c0.offset + off,
            size: len,
            domain: c0.domain,
        })
    }

    fn raw(&self) -> *const c_void {
        self.0
    }
}

impl Drop for DevAddr {
    fn drop(&mut self) {
        unsafe { fxa_addr_free(self.0) }
    }
}

unsafe impl Send for DevAddr {}
