// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Minimal MTL4 single-dispatch helper.
//!
//! `clippy.toml`'s `disallowed-methods` bans `MTLCommandQueue::commandBuffer`
//! (classic MTL3) everywhere the workspace builds against `objc2_metal` — this
//! crate's own Metal use (a CPU-validation accelerator for the emulator's
//! large matmuls, unrelated to `scratchy-target-metal`'s model-serving Metal
//! backend) is no exception. MTL4 has NO implicit resource tracking: every
//! buffer a kernel touches — including the tiny address-bound buffers a
//! classic `setBytes` scalar becomes — must be in a committed residency set
//! AND bound through an argument table by its `gpuAddress()`, never
//! `setBuffer_offset_atIndex`/`setBytes_length_atIndex`.
//!
//! Self-contained: no dependency on `scratchy-target-metal`. That crate is
//! the METAL target (model-serving); this one is the SPYRE emulator, and the
//! architecture keeps target crates from reaching into each other — so this
//! is its own copy of the same MTL4 plumbing
//! (`scratchy_target_metal::mtl4_dispatch`), not a shared import.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandBuffer, MTL4CommandEncoder,
    MTL4CommandQueue, MTL4ComputeCommandEncoder, MTLAllocation, MTLBuffer, MTLComputePipelineState,
    MTLDevice, MTLResidencySet, MTLResidencySetDescriptor, MTLResourceOptions, MTLSharedEvent,
    MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;

/// Whether `device` can actually mint an MTL4 command queue — a real
/// capability probe, not string-matching a `dispatch`/`Batch::begin` error
/// after the fact. `newMTL4CommandQueue()` returns `None` on some hosted CI
/// runners even though `MTLCreateSystemDefaultDevice` succeeds (a real GPU,
/// but the virtualized/shared device rejects MTL4 queues) — callers that
/// need to skip in that environment should check this BEFORE dispatching,
/// not classify the error string a failed dispatch produced.
///
/// `#[cfg(test)]`: currently only the test suite's skip-guards need this: the
/// crate's own dispatch/`Batch::begin` paths propagate a real `Err` on
/// failure to mint a queue instead of pre-checking, since a queue that stops
/// working mid-run there is a real bug to surface, not a skip condition.
#[cfg(test)]
pub(crate) fn mtl4_available(device: &Device) -> bool {
    device.newMTL4CommandQueue().is_some()
}

/// One kernel argument: a buffer already on the device, or scalar bytes that
/// [`dispatch`] turns into a tiny address-bound `StorageModeShared` buffer —
/// MTL4 has no `setBytes`, every binding is an address.
pub enum Arg<'a> {
    Buffer(&'a Buffer),
    Bytes(&'a [u8]),
}

/// How the encoder sizes its dispatch: total THREADS (the driver picks the
/// threadgroup grid), or an explicit THREADGROUP count (the caller's own
/// output-block tiling).
pub enum Extent {
    Threads(MTLSize),
    Threadgroups(MTLSize),
}

/// Run one compute kernel to completion on MTL4: build a residency set + an
/// argument table over `args` (bound at their slice index, so index `i`
/// answers a kernel's `[[buffer(i)]]`), dispatch, and block until the GPU is
/// done. Mirrors `scratchy-target-metal`'s `mtl4_dispatch::dispatch_threadgroups`,
/// minus the multi-dispatch batching this crate's single-shot call sites don't
/// need. Returns the submit→complete wall-clock elapsed time — MTL4 command
/// buffers drop the classic `GPUStartTime`/`GPUEndTime` counters, so callers
/// that were reading those (perf telemetry, not correctness) get this coarser
/// proxy instead.
pub fn dispatch(
    device: &Device,
    pso: &Pipeline,
    args: &[Arg<'_>],
    extent: Extent,
    threads_per_threadgroup: MTLSize,
) -> Result<std::time::Duration, String> {
    dispatch_readback(device, pso, args, extent, threads_per_threadgroup, None)
        .map(|(_, elapsed)| elapsed)
}

/// Same as [`dispatch`], additionally reading back `readback.1` bytes from
/// the argument at index `readback.0` once the GPU completes — the shape
/// every host-readback call site here needs (dispatch, then read the output
/// buffer). Returns the bytes and the submit→complete elapsed time.
pub fn dispatch_readback(
    device: &Device,
    pso: &Pipeline,
    args: &[Arg<'_>],
    extent: Extent,
    threads_per_threadgroup: MTLSize,
    readback: Option<(usize, usize)>,
) -> Result<(Vec<u8>, std::time::Duration), String> {
    let queue4 = device
        .newMTL4CommandQueue()
        .ok_or("mtl4: no MTL4 command queue on this device")?;

    // Every `Bytes` arg becomes an address-bound scratch buffer up front — it
    // must outlive the dispatch, and its address goes in the argument table
    // exactly like a real buffer's.
    let res = MTLResourceOptions::StorageModeShared;
    let mut scalars: Vec<Buffer> = Vec::new();
    let mut bound: Vec<Buffer> = Vec::with_capacity(args.len());
    for a in args {
        match a {
            Arg::Buffer(b) => bound.push((*b).clone()),
            Arg::Bytes(bytes) => {
                let buf = unsafe {
                    device
                        .newBufferWithBytes_length_options(
                            NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                            bytes.len().max(1),
                            res,
                        )
                        .ok_or("mtl4: scalar buffer alloc failed")?
                };
                bound.push(buf.clone());
                scalars.push(buf);
            }
        }
    }

    let residency = residency_set(device)?;
    for b in &bound {
        let alloc: &ProtocolObject<dyn MTLAllocation> = b.as_ref();
        residency.addAllocation(alloc);
    }
    residency.commit();

    let desc = MTL4ArgumentTableDescriptor::new();
    desc.setMaxBufferBindCount(bound.len().max(1));
    let table = device
        .newArgumentTableWithDescriptor_error(&desc)
        .map_err(|e| format!("mtl4: argument table alloc failed: {e:?}"))?;
    for (i, b) in bound.iter().enumerate() {
        unsafe {
            table.setAddress_atIndex(b.gpuAddress(), i);
        }
    }

    let alloc4 = device
        .newCommandAllocator()
        .ok_or("mtl4: command allocator alloc failed")?;
    let event = device
        .newSharedEvent()
        .ok_or("mtl4: shared event alloc failed")?;
    let cb = device
        .newCommandBuffer()
        .ok_or("mtl4: command buffer alloc failed")?;
    cb.beginCommandBufferWithAllocator(&alloc4);
    cb.useResidencySet(&residency);
    let enc = cb
        .computeCommandEncoder()
        .ok_or("mtl4: compute encoder alloc failed")?;
    enc.setComputePipelineState(pso);
    enc.setArgumentTable(Some(&table));
    match extent {
        Extent::Threads(t) => enc.dispatchThreads_threadsPerThreadgroup(t, threads_per_threadgroup),
        Extent::Threadgroups(t) => {
            enc.dispatchThreadgroups_threadsPerThreadgroup(t, threads_per_threadgroup)
        }
    }
    enc.endEncoding();
    cb.endCommandBuffer();

    let cb_protocol: &ProtocolObject<dyn MTL4CommandBuffer> = &cb;
    let mut cb_array = [std::ptr::NonNull::from(cb_protocol)];
    let start = std::time::Instant::now();
    unsafe {
        queue4.commit_count(std::ptr::NonNull::from(&mut cb_array[0]), 1);
    }
    queue4.signalEvent_value(ProtocolObject::from_ref(&*event), 1);
    if !event.waitUntilSignaledValue_timeoutMS(1, 30_000) {
        return Err("mtl4: dispatch timed out".into());
    }
    let elapsed = start.elapsed();

    let bytes = match readback {
        Some((idx, len)) => {
            let buf = bound.get(idx).ok_or("mtl4: readback index out of range")?;
            let raw =
                unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, len) };
            raw.to_vec()
        }
        None => Vec::new(),
    };
    Ok((bytes, elapsed))
}

/// A single MTL4 command buffer carrying N sequential dispatches — for a
/// chain like [`super::MetalGemm::run_chain`] where each step reads the
/// previous step's output and the ~250µs submit/wait round trip is paid once
/// for the whole chain rather than once per step. Lifecycle: [`begin`] opens
/// the CB + compute encoder + residency set; [`encode`] appends one dispatch
/// ([`barrier`] first if it depends on the previous one's output); [`commit`]
/// ends encoding, commits residency, submits, and blocks until the GPU
/// drains.
pub struct Batch {
    device: Device,
    queue4: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    alloc4: Retained<ProtocolObject<dyn objc2_metal::MTL4CommandAllocator>>,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    cb: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    enc: Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>,
    /// `setBytes`-replacement scalar buffers and per-dispatch argument
    /// tables, kept alive until [`commit`] — MTL4 does not retain either.
    scalars: Vec<Buffer>,
    tables: Vec<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>,
}

impl Batch {
    pub fn begin(device: &Device) -> Result<Self, String> {
        let queue4 = device
            .newMTL4CommandQueue()
            .ok_or("mtl4: no MTL4 command queue on this device")?;
        let residency = residency_set(device)?;
        let alloc4 = device
            .newCommandAllocator()
            .ok_or("mtl4: command allocator alloc failed")?;
        let event = device
            .newSharedEvent()
            .ok_or("mtl4: shared event alloc failed")?;
        let cb = device
            .newCommandBuffer()
            .ok_or("mtl4: command buffer alloc failed")?;
        cb.beginCommandBufferWithAllocator(&alloc4);
        let enc = cb
            .computeCommandEncoder()
            .ok_or("mtl4: compute encoder alloc failed")?;
        Ok(Self {
            device: device.clone(),
            queue4,
            residency,
            alloc4,
            event,
            cb,
            enc,
            scalars: Vec::new(),
            tables: Vec::new(),
        })
    }

    /// Encode one dispatch: `args` bind at their slice index exactly like
    /// [`dispatch`]'s. Call [`barrier`] first if this dispatch reads a buffer
    /// an earlier one in this batch wrote — MTL4 compute encoders do not
    /// auto-serialize same-encoder dispatches.
    pub fn encode(
        &mut self,
        pso: &Pipeline,
        args: &[Arg<'_>],
        extent: Extent,
        threads_per_threadgroup: MTLSize,
    ) -> Result<(), String> {
        let res = MTLResourceOptions::StorageModeShared;
        let mut bound: Vec<Buffer> = Vec::with_capacity(args.len());
        for a in args {
            match a {
                Arg::Buffer(b) => bound.push((*b).clone()),
                Arg::Bytes(bytes) => {
                    let buf = unsafe {
                        self.device
                            .newBufferWithBytes_length_options(
                                NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                                bytes.len().max(1),
                                res,
                            )
                            .ok_or("mtl4: scalar buffer alloc failed")?
                    };
                    bound.push(buf.clone());
                    self.scalars.push(buf);
                }
            }
        }
        for b in &bound {
            let alloc: &ProtocolObject<dyn MTLAllocation> = b.as_ref();
            self.residency.addAllocation(alloc);
        }

        let desc = MTL4ArgumentTableDescriptor::new();
        desc.setMaxBufferBindCount(bound.len().max(1));
        let table = self
            .device
            .newArgumentTableWithDescriptor_error(&desc)
            .map_err(|e| format!("mtl4: argument table alloc failed: {e:?}"))?;
        for (i, b) in bound.iter().enumerate() {
            unsafe {
                table.setAddress_atIndex(b.gpuAddress(), i);
            }
        }

        self.enc.setComputePipelineState(pso);
        self.enc.setArgumentTable(Some(&table));
        match extent {
            Extent::Threads(t) => self
                .enc
                .dispatchThreads_threadsPerThreadgroup(t, threads_per_threadgroup),
            Extent::Threadgroups(t) => self
                .enc
                .dispatchThreadgroups_threadsPerThreadgroup(t, threads_per_threadgroup),
        }
        self.tables.push(table);
        Ok(())
    }

    /// Intra-encoder dispatch→dispatch barrier: a later [`encode`] depending
    /// on an earlier one's output needs this between them (see the struct
    /// doc). `Device` visibility because the producer's store may live in L2
    /// only.
    pub fn barrier(&self) {
        use objc2_metal::{MTL4CommandEncoder as _, MTL4VisibilityOptions, MTLStages};
        self.enc
            .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                MTLStages::Dispatch,
                MTLStages::Dispatch,
                MTL4VisibilityOptions::Device,
            );
    }

    /// End encoding, commit residency, submit, and block until the GPU
    /// drains. Returns the submit→complete elapsed time (see [`dispatch`]'s
    /// doc on why this is wall-clock, not `GPUStartTime`/`GPUEndTime`).
    pub fn commit(self) -> Result<std::time::Duration, String> {
        // The command allocator backs `self.cb`'s recorded commands for its
        // whole lifetime (through submission below) — held only to keep it
        // alive that long, never otherwise read.
        let _allocator_keepalive = &self.alloc4;
        self.enc.endEncoding();
        // Commit the now-populated residency set, THEN attach it to the CB, so
        // the driver wires every bound buffer this batch touched — mirrors
        // `scratchy-target-metal`'s `Mtl4DispatchBatch::commit`.
        self.residency.commit();
        self.cb.useResidencySet(&self.residency);
        self.cb.endCommandBuffer();

        let cb_protocol: &ProtocolObject<dyn MTL4CommandBuffer> = &self.cb;
        let mut cb_array = [std::ptr::NonNull::from(cb_protocol)];
        let start = std::time::Instant::now();
        unsafe {
            self.queue4
                .commit_count(std::ptr::NonNull::from(&mut cb_array[0]), 1);
        }
        self.queue4
            .signalEvent_value(ProtocolObject::from_ref(&*self.event), 1);
        if !self.event.waitUntilSignaledValue_timeoutMS(1, 60_000) {
            return Err("mtl4: dispatch batch timed out".into());
        }
        Ok(start.elapsed())
    }
}

/// A committed [`MTLResidencySet`] with every dispatch's buffers inserted.
/// One-shot per dispatch (this crate's call sites are not hot enough to
/// justify a persistent wired set — see `scratchy-target-metal`'s
/// `MetalResidencySet` for that tradeoff, which does not apply here).
fn residency_set(device: &Device) -> Result<Retained<ProtocolObject<dyn MTLResidencySet>>, String> {
    let desc = MTLResidencySetDescriptor::new();
    device
        .newResidencySetWithDescriptor_error(&desc)
        .map_err(|e| format!("mtl4: residency set alloc failed: {e:?}"))
}
