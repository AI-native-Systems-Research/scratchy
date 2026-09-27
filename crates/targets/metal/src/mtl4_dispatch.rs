// SPDX-License-Identifier: Apache-2.0
//! Single-op Metal-4 dispatch helper for tests, benches, and cost-sweeps.
//!
//! Production runs every kernel through the MTL4 tape (`bake_mtl4_steps`
//! / `run_bucket_mtl4`). This helper lets out-of-band callers (isolated
//! kernel parity tests, the perf benches, and the cost-sweep profiler)
//! exercise the SAME kernel on the SAME MTL4 path — argument-table
//! `setAddress`/`gpuAddress` bindings, a committed residency set, and the
//! `begin → useResidencySet → encode → commit → event-wait` command
//! lifecycle — instead of a classic `queue.commandBuffer()` encoder.
//! Scalars that the classic path passed via `setBytes` become tiny
//! address-bound `StorageModeShared` buffers (see [`shared_u32`]).

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4CommandQueue, MTL4ComputeCommandEncoder, MTLBuffer,
    MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSharedEvent, MTLSize,
};

use crate::residency::{MetalResidencySet, Pinned};
use crate::stream::MetalStreamError;

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;

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
/// Returns `false` if the host has no MTL4 queue (caller should skip).
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

    let cb_protocol: &ProtocolObject<dyn MTL4CommandBuffer> = &cb;
    let mut cb_array = [std::ptr::NonNull::from(cb_protocol)];
    unsafe {
        queue4.commit_count(std::ptr::NonNull::from(&mut cb_array[0]), 1);
    }
    queue4.signalEvent_value(ProtocolObject::from_ref(&*event), 1);
    assert!(
        event.waitUntilSignaledValue_timeoutMS(1, 30_000),
        "MTL4 dispatch helper timed out"
    );
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
/// commits the residency set, ends the CB, and submits.
///
/// MTL4 has NO implicit resource tracking, so every address-bound buffer (and
/// every buffer the kernel dereferences via a stored `gpuAddress`, e.g. the
/// paged chunk-data buffers) must be in the committed residency set, and the
/// scalar buffers / argument tables must stay ALIVE until the GPU drains. The
/// sync `commit(true)` path event-waits, so `self`'s resources drop only after
/// completion; the async `commit(false)` path can't wait, so it hands the
/// resources to a commit-feedback handler that releases them on GPU completion.
pub struct Mtl4DispatchBatch {
    device: Device,
    queue4: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    res: MetalResidencySet,
    alloc4: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
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
            alloc4,
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

    /// End encoding, commit the residency set, end + submit the command buffer.
    /// `wait == true` signals a shared event and blocks until the GPU drains
    /// (sync path: returns `Err` on timeout). `wait == false` submits fire-and-
    /// forget — queue ordering guarantees a later dispatch on the same device
    /// sees the result — and keeps the batch's resources alive via a commit-
    /// feedback handler that drops them only after GPU completion.
    pub fn commit(mut self, wait: bool) -> Result<(), MetalStreamError> {
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
        let cb_protocol: &ProtocolObject<dyn MTL4CommandBuffer> = &self.cb;
        let cb_nn = std::ptr::NonNull::from(cb_protocol);
        self.cb.endCommandBuffer();
        if wait {
            let mut cb_array = [cb_nn];
            unsafe {
                self.queue4
                    .commit_count(std::ptr::NonNull::from(&mut cb_array[0]), 1);
            }
            self.queue4
                .signalEvent_value(ProtocolObject::from_ref(&*self.event), 1);
            if !self.event.waitUntilSignaledValue_timeoutMS(1, 60_000) {
                return Err(MetalStreamError::ShaderCompilationFailed(
                    "MTL4 dispatch batch timed out".into(),
                ));
            }
            // `self` drops here → residency set + pinned buffers + tables freed
            // AFTER the GPU has drained. Safe.
            Ok(())
        } else {
            use block2::RcBlock;
            use objc2_metal::{MTL4CommitFeedback, MTL4CommitOptions};
            // No host wait: the GPU still references the residency set, the
            // pinned buffers, and the argument tables. MTL4 does
            // NOT implicitly retain any of them, so move them into a commit-
            // feedback handler that fires on completion — its captured handles
            // (Arc/Retained clones) keep everything alive until the GPU is done.
            let opts = MTL4CommitOptions::new();
            let res = self.res.clone();
            let pins = std::mem::take(&mut self.pins);
            let tables = std::mem::take(&mut self.tables);
            let alloc4 = self.alloc4.clone();
            let queue4 = self.queue4.clone();
            let cb_keep = self.cb.clone();
            let block = RcBlock::new(
                move |_fb: std::ptr::NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
                    // Touch every captured handle so the closure owns them; the
                    // handler runs once on GPU completion, then Metal releases
                    // the block and these handles drop.
                    let _ = (&res, &pins, &tables, &alloc4, &queue4, &cb_keep);
                },
            );
            unsafe {
                opts.addFeedbackHandler(RcBlock::as_ptr(&block) as _);
            }
            let mut cb_array = [cb_nn];
            unsafe {
                self.queue4.commit_count_options(
                    std::ptr::NonNull::from(&mut cb_array[0]),
                    1,
                    &opts,
                );
            }
            // Drop our `block` handle: Metal retained it in `addFeedbackHandler`
            // and releases it after firing, dropping the captured resources.
            Ok(())
        }
    }
}
