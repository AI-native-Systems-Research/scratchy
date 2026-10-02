// SPDX-License-Identifier: Apache-2.0
//! Single-op Metal-4 dispatch helper for tests and cost-sweeps.
//!
//! Production runs every kernel through the MTL4 tape (`bake_mtl4_steps`
//! / `run_bucket_mtl4`). This helper lets out-of-band callers (isolated
//! kernel parity tests and the cost-sweep profiler)
//! exercise the SAME kernel on the SAME MTL4 path — argument-table
//! `setAddress`/`gpuAddress` bindings, a committed residency set, and the
//! `begin → useResidencySet → encode → commit → event-wait` command
//! lifecycle — instead of a classic `queue.commandBuffer()` encoder.
//! Scalars that the classic path passed via `setBytes` become tiny
//! address-bound `StorageModeShared` buffers (see [`shared_u32`]).
//!
//! Every MTL4 commit whose results a caller reads — these helpers' and the
//! worker pool's — goes through [`commit_and_wait`].

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4CommandQueue, MTL4ComputeCommandEncoder, MTLBuffer,
    MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSharedEvent, MTLSize,
};

use crate::residency::{MetalResidencySet, Pinned};

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;

/// Block until `event` reaches `value`: the GPU is done with every command buffer committed
/// before the signal, however long that takes. Returning earlier hands back buffers a kernel is
/// still using — the caller frees, reuses or unpins them under it, and a process that then exits
/// leaves the kernel running on the GPU. So a kernel that never ends holds this wait, and the
/// thread, until the system ends its command buffer or the process is killed; each minute the wait
/// goes on is reported. Reaching the signal says nothing about whether the command buffers
/// completed: see [`commit_and_wait`].
fn wait_drained(event: &ProtocolObject<dyn MTLSharedEvent>, value: u64) {
    let started = std::time::Instant::now();
    while !event.waitUntilSignaledValue_timeoutMS(value, 60_000) {
        eprintln!(
            "[scratchy-target-metal] GPU work committed {:?} ago is still running; waiting for it",
            started.elapsed(),
        );
    }
}

/// How long [`commit_and_wait`] waits for a commit's feedback once the GPU has signalled past the
/// commit. Metal delivers it tens of microseconds after the signal; a report that has not come
/// within this is not coming.
const FEEDBACK_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// Why a command buffer did not complete: the cause its commit feedback reports (Metal's
/// `MTL4CommandQueueError` codes), or no feedback at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandFailure {
    Timeout,
    NotPermitted,
    /// What a command buffer whose working set the GPU cannot hold resident reports (it does not
    /// run): `kIOGPUCommandBufferCallbackErrorOutOfMemory`.
    OutOfMemory,
    DeviceRemoved,
    AccessRevoked,
    Internal,
    /// No error in `MTL4CommandQueueErrorDomain`, or a code the domain does not define: the top
    /// error's code.
    Other {
        code: isize,
    },
    /// No feedback within [`FEEDBACK_WAIT`] of the GPU signalling past the commit.
    Unreported,
}

impl CommandFailure {
    /// The cause: the innermost `MTL4CommandQueueErrorDomain` error along `error`'s underlying
    /// errors. Metal wraps it — a command buffer that ran out of memory reports a code-1
    /// ("timeout") error whose underlying error is the out-of-memory one.
    fn of(error: &objc2_foundation::NSError) -> Self {
        use objc2_metal::MTL4CommandQueueError as E;
        fn cause(error: &objc2_foundation::NSError) -> Option<isize> {
            let domain = unsafe { objc2_metal::MTL4CommandQueueErrorDomain };
            (error.underlyingErrors().iter())
                .find_map(|under| cause(&under))
                .or_else(|| error.domain().isEqualToString(domain).then(|| error.code()))
        }
        let Some(code) = cause(error) else {
            return Self::Other { code: error.code() };
        };
        match E(code) {
            E::Timeout => Self::Timeout,
            E::NotPermitted => Self::NotPermitted,
            E::OutOfMemory => Self::OutOfMemory,
            E::DeviceRemoved => Self::DeviceRemoved,
            E::AccessRevoked => Self::AccessRevoked,
            E::Internal => Self::Internal,
            _ => Self::Other { code },
        }
    }
}

impl std::fmt::Display for CommandFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Other { code } => {
                write!(f, "the GPU failed the command buffer (error code {code})")
            }
            Self::Unreported => write!(
                f,
                "Metal sent no report on the command buffer within {FEEDBACK_WAIT:?} of its completion"
            ),
            failure => write!(f, "the GPU failed the command buffer: {failure:?}"),
        }
    }
}

impl std::error::Error for CommandFailure {}

/// Commit `cb` alone on `queue`, then signal `event` to `value`; return once the GPU is done with
/// it ([`wait_drained`]) AND Metal has reported on it — the error of a command buffer that did not
/// complete. A failed command buffer still reaches the signal (one refused as out of memory under
/// memory pressure does not run at all, its outputs left as they were), so the report is the only
/// thing that tells it from a completed one. The commit carries its OWN `MTL4CommitOptions` and
/// feedback handler: Metal calls a handler once, for the first commit of the options it was added
/// to — a handler on options reused across commits never hears about any later one.
pub fn commit_and_wait(
    queue: &ProtocolObject<dyn MTL4CommandQueue>,
    cb: &ProtocolObject<dyn MTL4CommandBuffer>,
    event: &ProtocolObject<dyn MTLSharedEvent>,
    value: u64,
) -> Result<(), CommandFailure> {
    commit_and_time(queue, cb, event, value).map(drop)
}

/// [`commit_and_wait`], returning how long the GPU ran the command buffer: from its start to its
/// end, as Metal reports them.
pub fn commit_and_time(
    queue: &ProtocolObject<dyn MTL4CommandQueue>,
    cb: &ProtocolObject<dyn MTL4CommandBuffer>,
    event: &ProtocolObject<dyn MTLSharedEvent>,
    value: u64,
) -> Result<std::time::Duration, CommandFailure> {
    use block2::RcBlock;
    use objc2_metal::{MTL4CommitFeedback, MTL4CommitOptions};
    use std::sync::{Arc, Condvar, Mutex};
    type Report = Option<Result<std::time::Duration, CommandFailure>>;
    let report: Arc<(Mutex<Report>, Condvar)> = Arc::default();
    let options = MTL4CommitOptions::new();
    let handler = {
        let report = Arc::clone(&report);
        RcBlock::new(
            move |feedback: std::ptr::NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
                let feedback = unsafe { feedback.as_ref() };
                let outcome = match feedback.error() {
                    None => Ok(std::time::Duration::from_secs_f64(
                        (feedback.GPUEndTime() - feedback.GPUStartTime()).max(0.0),
                    )),
                    Some(error) => {
                        // The NSError's userInfo (nested underlying errors) names the faulting
                        // encoder, which the typed failure cannot carry; `description` renders
                        // the whole tree.
                        let tree: Retained<objc2_foundation::NSString> =
                            unsafe { objc2::msg_send![&*error, description] };
                        eprintln!("[scratchy-target-metal] GPU command buffer failed: {tree}");
                        Err(CommandFailure::of(&error))
                    }
                };
                let (slot, reported) = &*report;
                *slot.lock().expect("commit report") = Some(outcome);
                reported.notify_one();
            },
        )
    };
    unsafe { options.addFeedbackHandler(RcBlock::as_ptr(&handler) as _) };
    let mut cbs = [std::ptr::NonNull::from(cb)];
    unsafe { queue.commit_count_options(std::ptr::NonNull::from(&mut cbs[0]), 1, &options) };
    queue.signalEvent_value(ProtocolObject::from_ref(event), value);
    wait_drained(event, value);
    let (slot, reported) = &*report;
    let slot = slot.lock().expect("commit report");
    let (mut slot, _) = reported
        .wait_timeout_while(slot, FEEDBACK_WAIT, |outcome| outcome.is_none())
        .expect("commit report");
    slot.take().unwrap_or(Err(CommandFailure::Unreported))
}

/// `StorageModeShared` buffer initialized from `data`.
pub fn shared_bytes(device: &Device, data: &[u8]) -> Buffer {
    let buf = device
        .newBufferWithLength_options(data.len().max(1), MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buf.contents().as_ptr() as *mut u8,
            data.len(),
        );
    }
    buf
}

/// Zeroed `StorageModeShared` buffer of `len` bytes.
pub fn shared_zeroed(device: &Device, len: usize) -> Buffer {
    let buf = device
        .newBufferWithLength_options(len.max(1), MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe {
        std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, len);
    }
    buf
}

/// A scalar `u32` as an address-bindable buffer (replaces the classic
/// path's `setBytes` for a `device const uint&` kernel argument).
pub fn shared_u32(device: &Device, v: u32) -> Buffer {
    shared_bytes(device, &v.to_ne_bytes())
}

/// A scalar `f32` as an address-bindable buffer.
pub fn shared_f32(device: &Device, v: f32) -> Buffer {
    shared_bytes(device, &v.to_ne_bytes())
}

/// `StorageModeShared` buffer holding a `Copy` slice verbatim (e.g. a
/// `&[half::bf16]` / `&[f16]` / `&[u32]` of kernel input).
pub fn shared_slice<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    shared_bytes(device, bytes)
}

/// Read `n` `Copy` elements back out of a `StorageModeShared` buffer.
pub fn read_slice<T: Copy>(buf: &Buffer, n: usize) -> Vec<T> {
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const T, n) }.to_vec()
}

/// Dispatch one compute kernel on MTL4. `buffers[i].gpuAddress()` is
/// bound at argument-table index `i` (so the kernel's `[[buffer(i)]]`
/// resolves to it), the residency set covers every buffer, and the
/// command buffer declares it via `useResidencySet:` exactly like
/// production. Blocks on a shared event until the GPU completes.
/// Hard occupancy guard. A compute dispatch that requests more
/// threads/threadgroup than the pipeline's `maxTotalThreadsPerThreadgroup`
/// is illegal: the driver silently under-launches and produces WRONG results
/// with no error. Observed on M1 — the hd256/512 `attention_via_cache_v2`
/// decode kernel requests 1024 threads but under its register pressure the M1
/// pipeline max was 640, so simdgroups dropped out and the online-softmax
/// combine read uninitialized threadgroup slots -> garbage. Panic loudly
/// rather than corrupt silently; a kernel that needs the launch must declare
/// `[[max_total_threads_per_threadgroup(N)]]`. (The forward path is guarded up
/// front in `bake_mtl4_steps`; this covers the auxiliary dispatch paths —
/// argmax / sampling / grammar-mask / chain-advance.)
#[inline]
fn assert_within_pipeline_cap(pso: &Pipeline, threads_per_threadgroup: MTLSize) {
    let req = threads_per_threadgroup.width
        * threads_per_threadgroup.height
        * threads_per_threadgroup.depth;
    let max = pso.maxTotalThreadsPerThreadgroup();
    assert!(
        req <= max,
        "dispatch requests {req} threads/threadgroup ({}x{}x{}) but pipeline \
         maxTotalThreadsPerThreadgroup is {max} — the GPU would under-launch and \
         silently corrupt output. Add [[max_total_threads_per_threadgroup({req})]] \
         to the kernel (or reduce its threadgroup size).",
        threads_per_threadgroup.width,
        threads_per_threadgroup.height,
        threads_per_threadgroup.depth,
    );
}

///
/// Returns `false` if the host has no MTL4 queue (caller should skip); panics if the GPU fails the
/// command buffer ([`commit_and_wait`]).
pub fn dispatch_threadgroups(
    device: &Device,
    pso: &Pipeline,
    buffers: &[&Buffer],
    threadgroups: MTLSize,
    threads_per_threadgroup: MTLSize,
) -> bool {
    let Some(queue4) = device.newMTL4CommandQueue() else {
        eprintln!("skipping: no MTL4 queue");
        return false;
    };

    // Residency: MTL4 declares residency explicitly (no implicit
    // tracking), so every address-bound buffer must be in a committed
    // set attached to the command buffer.
    let res = crate::residency::MetalResidencySet::new(device);
    let _pins: Vec<Pinned> = buffers.iter().map(|&b| res.pin(b.clone())).collect();
    res.commit();

    let desc = MTL4ArgumentTableDescriptor::new();
    desc.setMaxBufferBindCount(buffers.len());
    let table = device
        .newArgumentTableWithDescriptor_error(&desc)
        .expect("argument table");
    for (i, b) in buffers.iter().enumerate() {
        unsafe {
            table.setAddress_atIndex(b.gpuAddress(), i);
        }
    }

    let alloc4 = device
        .newCommandAllocator()
        .expect("MTL4 command allocator");
    let event = device.newSharedEvent().expect("shared event");
    let cb = device.newCommandBuffer().expect("mtl4 command buffer");
    cb.beginCommandBufferWithAllocator(&alloc4);
    let cb_ptr: *mut AnyObject = Retained::as_ptr(&cb) as *const AnyObject as *mut AnyObject;
    unsafe {
        res.attach_to_mtl4_command_buffer(cb_ptr);
    }
    let enc = cb.computeCommandEncoder().expect("mtl4 encoder");
    enc.setComputePipelineState(pso);
    enc.setArgumentTable(Some(&table));
    assert_within_pipeline_cap(pso, threads_per_threadgroup);
    enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_threadgroup);
    enc.endEncoding();
    cb.endCommandBuffer();

    if let Err(failure) = commit_and_wait(&queue4, &cb, &event, 1) {
        panic!("MTL4 dispatch: {failure}");
    }
    true
}

/// Build a per-dispatch MTL4 argument table from an index→address map.
/// The table is sized to `max(index) + 1`; every index `0..=max` is first
/// set to `gap_fill` (a throwaway resident address), then each `(addr, index)`
/// overwrites its slot. This is how a kernel with NON-CONSECUTIVE bindings
/// (e.g. buffers at 0..6 + 16, scalars at 7..15 + 17) gets a dense table with
/// no unbound holes (an unbound slot is a GPU fault on some drivers).
pub fn build_arg_table(
    device: &Device,
    bindings: &[(u64, usize)],
    gap_fill: u64,
) -> Retained<ProtocolObject<dyn MTL4ArgumentTable>> {
    let max_idx = bindings.iter().map(|&(_, i)| i).max().unwrap_or(0);
    let desc = MTL4ArgumentTableDescriptor::new();
    desc.setMaxBufferBindCount(max_idx + 1);
    let table = device
        .newArgumentTableWithDescriptor_error(&desc)
        .expect("argument table");
    for i in 0..=max_idx {
        unsafe {
            table.setAddress_atIndex(gap_fill, i);
        }
    }
    for &(addr, i) in bindings {
        unsafe {
            table.setAddress_atIndex(addr, i);
        }
    }
    table
}

/// A single MTL4 command buffer + residency set carrying N argument-table
/// dispatches — the production-path replacement for a classic
/// `queue.commandBuffer()` + `MTLComputeCommandEncoder`.
///
/// Lifecycle: [`begin`](Self::begin) opens the CB + compute encoder + a fresh
/// residency set; [`encode`](Self::encode) appends one dispatch (its own
/// argument table, its buffers made resident, its `setBytes` scalars turned
/// into address-bound scalar buffers); [`commit`](Self::commit) ends encoding,
/// commits the residency set, ends the CB, submits and waits.
///
/// MTL4 has NO implicit resource tracking, so every address-bound buffer (and
/// every buffer the kernel dereferences via a stored `gpuAddress`, e.g. the
/// paged chunk-data buffers) must be in the committed residency set, and the
/// scalar buffers / argument tables must stay ALIVE until the GPU drains:
/// `commit` waits, so `self`'s resources drop only after completion.
pub struct Mtl4DispatchBatch {
    device: Device,
    queue4: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    res: MetalResidencySet,
    /// The command buffer's allocator, alive until it completes.
    _alloc4: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    cb: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    enc: Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>,
    /// Every buffer the batch binds or makes resident (bound, dereferenced,
    /// `setBytes`-replacement scalars, the gap filler), pinned until completion.
    pins: Vec<Pinned>,
    /// Per-dispatch argument tables, kept alive until completion.
    tables: Vec<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>,
    /// Address of the throwaway zeroed buffer bound at any unused argument-table
    /// gap index (the buffer itself is `pins[0]`).
    zero: u64,
}

impl Mtl4DispatchBatch {
    /// Open a fresh MTL4 command buffer + compute encoder + residency set.
    /// Returns `None` if the host has no MTL4 queue (caller should skip).
    pub fn begin(device: &Device) -> Option<Self> {
        let queue4 = device.newMTL4CommandQueue()?;
        let res = MetalResidencySet::new(device);
        let alloc4 = device
            .newCommandAllocator()
            .expect("MTL4 command allocator");
        let event = device.newSharedEvent().expect("shared event");
        let zero = res.pin(shared_zeroed(device, 16));
        let cb = device.newCommandBuffer().expect("mtl4 command buffer");
        cb.beginCommandBufferWithAllocator(&alloc4);
        let enc = cb.computeCommandEncoder().expect("mtl4 encoder");
        Some(Self {
            device: device.clone(),
            queue4,
            res,
            _alloc4: alloc4,
            event,
            cb,
            enc,
            zero: zero.gpuAddress(),
            pins: vec![zero],
            tables: Vec::new(),
        })
    }

    /// Encode one compute dispatch.
    ///
    /// * `buffer_bindings`: `(buffer, index)` bound by `gpuAddress` at `index`
    ///   and inserted into the residency set.
    /// * `u32_scalars` / `f32_scalars`: `(value, index)` — each becomes a fresh
    ///   `StorageModeShared` scalar buffer bound at `index` (the MTL4 stand-in
    ///   for the classic `setBytes`), made resident and kept alive.
    /// * `extra_resident`: buffers the kernel dereferences via a stored
    ///   `gpuAddress` (e.g. the paged chunk-data buffers) — made resident but
    ///   NOT bound in the argument table.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        pso: &Pipeline,
        buffer_bindings: &[(&Buffer, usize)],
        u32_scalars: &[(u32, usize)],
        f32_scalars: &[(f32, usize)],
        extra_resident: &[&Buffer],
        threadgroups: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        let mut binds: Vec<(u64, usize)> =
            Vec::with_capacity(buffer_bindings.len() + u32_scalars.len() + f32_scalars.len());
        for &(b, i) in buffer_bindings {
            self.pins.push(self.res.pin(b.clone()));
            binds.push((b.gpuAddress(), i));
        }
        for &b in extra_resident {
            self.pins.push(self.res.pin(b.clone()));
        }
        for &(v, i) in u32_scalars {
            let sb = self.res.pin(shared_u32(&self.device, v));
            binds.push((sb.gpuAddress(), i));
            self.pins.push(sb);
        }
        for &(v, i) in f32_scalars {
            let sb = self.res.pin(shared_f32(&self.device, v));
            binds.push((sb.gpuAddress(), i));
            self.pins.push(sb);
        }
        let table = build_arg_table(&self.device, &binds, self.zero);
        self.enc.setComputePipelineState(pso);
        self.enc.setArgumentTable(Some(&table));
        assert_within_pipeline_cap(pso, threads_per_threadgroup);
        self.enc
            .dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_threadgroup);
        self.tables.push(table);
    }

    /// Insert an intra-encoder dispatch→dispatch barrier so a later
    /// [`encode`](Self::encode) reads what an earlier one wrote. MTL4 compute
    /// encoders do NOT auto-serialize same-encoder dispatches, so a producer→
    /// consumer chain within ONE command buffer (e.g. cast → penalties →
    /// sample, each reading the previous kernel's `device`-space store) needs
    /// this between the dependent dispatches. `Device` visibility because the
    /// producer's store may live in L2 only. Mirrors the pre-argmax barrier in
    /// `argmax::encode_argmax_into_mtl4_inner`.
    pub fn barrier(&self) {
        use objc2_metal::{MTL4CommandEncoder as _, MTL4VisibilityOptions, MTLStages};
        self.enc
            .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                MTLStages::Dispatch,
                MTLStages::Dispatch,
                MTL4VisibilityOptions::Device,
            );
    }

    /// End encoding, commit the residency set, end + submit the command buffer
    /// and block until the GPU is done with it ([`commit_and_wait`]: however long
    /// that takes); `self`'s resources drop only then. Panics if the GPU fails
    /// the command buffer.
    pub fn commit(self) {
        if let Err(failure) = self.try_commit() {
            panic!("MTL4 dispatch batch: {failure}");
        }
    }

    /// [`Self::commit`], the GPU's failure returned; on success, how long the GPU ran the batch
    /// ([`commit_and_time`]).
    pub fn try_commit(self) -> Result<std::time::Duration, CommandFailure> {
        self.enc.endEncoding();
        // Commit the now-populated residency set, THEN attach it to the CB
        // (between begin and endCommandBuffer) so the driver wires every bound +
        // dereferenced buffer for this CB. Attaching after the commit (rather
        // than while empty) avoids any chance of binding an empty snapshot.
        self.res.commit();
        let cb_ptr: *mut AnyObject =
            Retained::as_ptr(&self.cb) as *const AnyObject as *mut AnyObject;
        unsafe {
            self.res.attach_to_mtl4_command_buffer(cb_ptr);
        }
        self.cb.endCommandBuffer();
        commit_and_time(&self.queue4, &self.cb, &self.event, 1)
    }
}
