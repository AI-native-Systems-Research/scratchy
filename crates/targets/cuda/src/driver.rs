// SPDX-License-Identifier: Apache-2.0
//! Thin wrappers around CUDA driver API via cudarc's sys module.
//!
//! Each function is 1-3 lines: call the sys function, check CUresult, return.
//! These are low-level building blocks — higher-level code uses `GpuDevice`.

use anyhow::{Result, bail};
use cudarc::driver::sys::{self, CUcontext, CUdevice, CUdeviceptr, CUevent, CUresult, CUstream};

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

fn check(result: CUresult) -> Result<()> {
    if result != sys::cudaError_enum::CUDA_SUCCESS {
        bail!("CUDA driver error: {:?}", result);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

/// Initialize the CUDA driver. Must be called once before any other CUDA call.
pub unsafe fn init() -> Result<()> {
    check(sys::cuInit(0))
}

/// Get the number of CUDA devices.
pub unsafe fn device_count() -> Result<i32> {
    let mut count = 0i32;
    check(sys::cuDeviceGetCount(&mut count))?;
    Ok(count)
}

/// Get a device handle.
pub unsafe fn device_get(ordinal: i32) -> Result<CUdevice> {
    let mut dev: CUdevice = 0;
    check(sys::cuDeviceGet(&mut dev, ordinal))?;
    Ok(dev)
}

/// Get the CUDA device currently attached to the calling thread's
/// primary context. The returned handle is suitable for passing into
/// [`device_get_num_sm`] / [`device_get_sm_version`] and for use as
/// the `device_id` argument to Marlin kernels.
///
/// Requires the caller to have already entered a CUDA context
/// (`ctx_create` / `ctx_set_current`); errors if no context is current.
pub unsafe fn current_device() -> Result<CUdevice> {
    let mut dev: CUdevice = 0;
    check(sys::cuCtxGetDevice(&mut dev))?;
    Ok(dev)
}

/// Get the number of streaming multiprocessors on a device.
pub unsafe fn device_get_num_sm(device: CUdevice) -> Result<i32> {
    let mut value = 0i32;
    check(sys::cuDeviceGetAttribute(
        &mut value,
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        device,
    ))?;
    Ok(value)
}

/// Get the SM version (compute capability) as a single number: major*10 + minor.
/// E.g. SM89 (Ada) → 89, SM80 (Ampere) → 80, SM90 (Hopper) → 90.
pub unsafe fn device_get_sm_version(device: CUdevice) -> Result<u32> {
    let mut major = 0i32;
    let mut minor = 0i32;
    check(sys::cuDeviceGetAttribute(
        &mut major,
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        device,
    ))?;
    check(sys::cuDeviceGetAttribute(
        &mut minor,
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        device,
    ))?;
    Ok((major * 10 + minor) as u32)
}

/// Retain the primary CUDA context on the given device.
///
/// Uses the primary context (shared with the CUDA runtime API) rather than
/// creating a standalone context. This is required for NCCL compatibility —
/// NCCL's internal proxy threads use the runtime API and expect the primary context.
pub unsafe fn ctx_create(device: CUdevice) -> Result<CUcontext> {
    let mut ctx: CUcontext = std::ptr::null_mut();
    check(sys::cuDevicePrimaryCtxRetain(&mut ctx, device))?;
    check(sys::cuCtxSetCurrent(ctx))?;
    Ok(ctx)
}

/// Set the current context for this thread.
pub unsafe fn ctx_set_current(ctx: CUcontext) -> Result<()> {
    check(sys::cuCtxSetCurrent(ctx))
}

/// Get the calling thread's current context. Spawned threads start with
/// no current context — pass this handle in and `ctx_set_current` it
/// before any driver call that needs one (e.g. `mem_alloc_host`).
pub unsafe fn ctx_get_current() -> Result<CUcontext> {
    let mut ctx: CUcontext = std::ptr::null_mut();
    check(sys::cuCtxGetCurrent(&mut ctx))?;
    Ok(ctx)
}

// ---------------------------------------------------------------------------
// Memory allocation
// ---------------------------------------------------------------------------

/// Allocate device memory. Called only during init/warmup, never on hot path.
pub unsafe fn mem_alloc(size: usize) -> Result<*mut u8> {
    let mut ptr: CUdeviceptr = 0;
    check(sys::cuMemAlloc_v2(&mut ptr, size))?;
    Ok(ptr as *mut u8)
}

/// Free device memory.
pub unsafe fn mem_free(ptr: *mut u8) -> Result<()> {
    check(sys::cuMemFree_v2(ptr as CUdeviceptr))
}

/// Allocate pinned (page-locked) host memory for async transfers.
pub unsafe fn mem_alloc_host(size: usize) -> Result<*mut u8> {
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    check(sys::cuMemAllocHost_v2(&mut ptr, size))?;
    Ok(ptr as *mut u8)
}

/// Free pinned host memory.
pub unsafe fn mem_free_host(ptr: *mut u8) -> Result<()> {
    check(sys::cuMemFreeHost(ptr as *mut std::ffi::c_void))
}

/// Query free and total device memory in bytes.
pub unsafe fn mem_get_info() -> Result<(usize, usize)> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    check(sys::cuMemGetInfo_v2(&mut free, &mut total))?;
    Ok((free, total))
}

/// Set device memory to zero.
pub unsafe fn memset_d8(ptr: *mut u8, value: u8, bytes: usize, stream: CUstream) -> Result<()> {
    check(sys::cuMemsetD8Async(
        ptr as CUdeviceptr,
        value,
        bytes,
        stream,
    ))
}

// ---------------------------------------------------------------------------
// Async memory transfers
// ---------------------------------------------------------------------------

/// Async host-to-device copy on a specific stream.
/// Synchronous (blocking) H2D copy. Unlike the `_async` variant this is
/// not stream-ordered: it returns once the bytes are on the device, so
/// the host source can be reused immediately — the pre-stage workers
/// rely on that to recycle one small pinned slot per worker without
/// events or stream synchronization.
pub unsafe fn memcpy_htod(dst_device: *mut u8, src_host: *const u8, bytes: usize) -> Result<()> {
    check(sys::cuMemcpyHtoD_v2(
        dst_device as CUdeviceptr,
        src_host as *const std::ffi::c_void,
        bytes,
    ))
}

pub unsafe fn memcpy_htod_async(
    dst: *mut u8,
    src: *const u8,
    bytes: usize,
    stream: CUstream,
) -> Result<()> {
    check(sys::cuMemcpyHtoDAsync_v2(
        dst as CUdeviceptr,
        src as *const std::ffi::c_void,
        bytes,
        stream,
    ))
}

/// Async device-to-host copy on a specific stream.
pub unsafe fn memcpy_dtoh_async(
    dst: *mut u8,
    src: *const u8,
    bytes: usize,
    stream: CUstream,
) -> Result<()> {
    check(sys::cuMemcpyDtoHAsync_v2(
        dst as *mut std::ffi::c_void,
        src as CUdeviceptr,
        bytes,
        stream,
    ))
}

/// Async device-to-device copy on a specific stream.
pub unsafe fn memcpy_dtod_async(
    dst: *mut u8,
    src: *const u8,
    bytes: usize,
    stream: CUstream,
) -> Result<()> {
    check(sys::cuMemcpyDtoDAsync_v2(
        dst as CUdeviceptr,
        src as CUdeviceptr,
        bytes,
        stream,
    ))
}

// ---------------------------------------------------------------------------
// Streams
// ---------------------------------------------------------------------------

/// Create a non-default, non-blocking stream (suitable for CUDA graph capture).
pub unsafe fn stream_create() -> Result<CUstream> {
    let mut stream: CUstream = std::ptr::null_mut();
    check(sys::cuStreamCreate(
        &mut stream,
        sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
    ))?;
    Ok(stream)
}

/// Destroy a stream.
pub unsafe fn stream_destroy(stream: CUstream) -> Result<()> {
    check(sys::cuStreamDestroy_v2(stream))
}

/// Synchronize a stream (block until all queued work completes).
pub unsafe fn stream_synchronize(stream: CUstream) -> Result<()> {
    check(sys::cuStreamSynchronize(stream))
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// Create a CUDA event (default flags).
pub unsafe fn event_create() -> Result<CUevent> {
    let mut event: CUevent = std::ptr::null_mut();
    check(sys::cuEventCreate(
        &mut event,
        sys::CUevent_flags::CU_EVENT_DEFAULT as u32,
    ))?;
    Ok(event)
}

/// Create a CUDA event that only tracks completion (no timing).
pub unsafe fn event_create_disable_timing() -> Result<CUevent> {
    let mut event: CUevent = std::ptr::null_mut();
    check(sys::cuEventCreate(
        &mut event,
        sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32,
    ))?;
    Ok(event)
}

/// Record an event on a stream.
pub unsafe fn event_record(event: CUevent, stream: CUstream) -> Result<()> {
    check(sys::cuEventRecord(event, stream))
}

/// Make a stream wait for an event.
pub unsafe fn stream_wait_event(stream: CUstream, event: CUevent) -> Result<()> {
    check(sys::cuStreamWaitEvent(stream, event, 0))
}

/// Destroy an event.
pub unsafe fn event_destroy(event: CUevent) -> Result<()> {
    check(sys::cuEventDestroy_v2(event))
}

/// Block the calling thread until the event has been recorded.
pub unsafe fn event_synchronize(event: CUevent) -> Result<()> {
    check(sys::cuEventSynchronize(event))
}

/// Like `event_synchronize` but takes a raw `usize` (for Send-safe closures).
pub unsafe fn event_synchronize_raw(event_addr: usize) -> Result<()> {
    check(sys::cuEventSynchronize(event_addr as CUevent))
}

/// Compute elapsed time in milliseconds between two recorded events.
pub unsafe fn event_elapsed(start: CUevent, end: CUevent) -> Result<f32> {
    let mut ms: f32 = 0.0;
    check(sys::cuEventElapsedTime(&mut ms, start, end))?;
    Ok(ms)
}

// ---------------------------------------------------------------------------
// CUDA graphs
// ---------------------------------------------------------------------------

/// Begin CUDA graph capture on a stream.
pub unsafe fn stream_begin_capture(stream: CUstream) -> Result<()> {
    check(sys::cuStreamBeginCapture_v2(
        stream,
        sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
    ))
}

/// End capture and return a CUgraph.
pub unsafe fn stream_end_capture(stream: CUstream) -> Result<sys::CUgraph> {
    let mut graph: sys::CUgraph = std::ptr::null_mut();
    check(sys::cuStreamEndCapture(stream, &mut graph))?;
    Ok(graph)
}

/// Instantiate a captured graph for replay.
pub unsafe fn graph_instantiate(graph: sys::CUgraph) -> Result<sys::CUgraphExec> {
    let mut exec: sys::CUgraphExec = std::ptr::null_mut();
    // Use cuGraphInstantiateWithFlags for newer CUDA (12+).
    check(sys::cuGraphInstantiateWithFlags(
        &mut exec,
        graph,
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH as u64,
    ))?;
    Ok(exec)
}

/// Launch an instantiated graph on a stream.
pub unsafe fn graph_launch(exec: sys::CUgraphExec, stream: CUstream) -> Result<()> {
    check(sys::cuGraphLaunch(exec, stream))
}

/// Destroy a graph.
pub unsafe fn graph_destroy(graph: sys::CUgraph) -> Result<()> {
    check(sys::cuGraphDestroy(graph))
}

/// Destroy an instantiated graph.
pub unsafe fn graph_exec_destroy(exec: sys::CUgraphExec) -> Result<()> {
    check(sys::cuGraphExecDestroy(exec))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    fn init_cuda() -> CUcontext {
        unsafe {
            init().expect("CUDA init");
            let dev = device_get(0).expect("device");
            ctx_create(dev).expect("context")
        }
    }

    #[test]
    fn test_init_and_device_count() {
        unsafe {
            init().expect("CUDA init");
            let count = device_count().expect("device count");
            assert!(count >= 1, "expected at least 1 GPU, got {count}");
        }
    }

    #[test]
    fn test_device_get() {
        unsafe {
            init().expect("CUDA init");
            let dev = device_get(0).expect("device 0");
            // Device handle should be a non-negative integer.
            assert!(dev >= 0);
        }
    }

    #[test]
    fn test_ctx_create_and_set() {
        let ctx = init_cuda();
        unsafe {
            ctx_set_current(ctx).expect("set current");
        }
    }

    #[test]
    fn test_mem_alloc_and_free() {
        let _ctx = init_cuda();
        unsafe {
            let ptr = mem_alloc(4096).expect("alloc");
            assert!(!ptr.is_null());
            mem_free(ptr).expect("free");
        }
    }

    #[test]
    fn test_mem_alloc_host_and_free() {
        let _ctx = init_cuda();
        unsafe {
            let ptr = mem_alloc_host(4096).expect("alloc host");
            assert!(!ptr.is_null());
            // Write to pinned memory to verify it's usable.
            std::ptr::write_bytes(ptr, 0xAB, 4096);
            mem_free_host(ptr).expect("free host");
        }
    }

    #[test]
    fn test_stream_create_sync_destroy() {
        let _ctx = init_cuda();
        unsafe {
            let stream = stream_create().expect("create");
            assert!(!stream.is_null());
            stream_synchronize(stream).expect("sync");
            stream_destroy(stream).expect("destroy");
        }
    }

    #[test]
    fn test_htod_dtoh_roundtrip() {
        let _ctx = init_cuda();
        unsafe {
            let stream = stream_create().expect("stream");
            let host_src = mem_alloc_host(256).expect("host src");
            let gpu = mem_alloc(256).expect("gpu");
            let host_dst = mem_alloc_host(256).expect("host dst");

            // Fill source with known pattern.
            for i in 0..256 {
                *host_src.add(i) = i as u8;
            }

            // H2D
            memcpy_htod_async(gpu, host_src, 256, stream).expect("htod");
            // D2H
            memcpy_dtoh_async(host_dst, gpu, 256, stream).expect("dtoh");
            stream_synchronize(stream).expect("sync");

            // Verify roundtrip.
            for i in 0..256 {
                assert_eq!(*host_dst.add(i), i as u8, "mismatch at byte {i}");
            }

            mem_free_host(host_src).expect("free src");
            mem_free(gpu).expect("free gpu");
            mem_free_host(host_dst).expect("free dst");
            stream_destroy(stream).expect("destroy stream");
        }
    }

    #[test]
    fn test_dtod_copy() {
        let _ctx = init_cuda();
        unsafe {
            let stream = stream_create().expect("stream");
            let host = mem_alloc_host(128).expect("host");
            let gpu_a = mem_alloc(128).expect("gpu_a");
            let gpu_b = mem_alloc(128).expect("gpu_b");

            // Fill host, copy to gpu_a.
            for i in 0..128 {
                *host.add(i) = (i * 3) as u8;
            }
            memcpy_htod_async(gpu_a, host, 128, stream).expect("htod");

            // D2D: gpu_a → gpu_b.
            memcpy_dtod_async(gpu_b, gpu_a, 128, stream).expect("dtod");

            // Read back from gpu_b.
            let host_out = mem_alloc_host(128).expect("host_out");
            memcpy_dtoh_async(host_out, gpu_b, 128, stream).expect("dtoh");
            stream_synchronize(stream).expect("sync");

            for i in 0..128 {
                assert_eq!(*host_out.add(i), (i * 3) as u8, "mismatch at {i}");
            }

            mem_free_host(host).unwrap();
            mem_free_host(host_out).unwrap();
            mem_free(gpu_a).unwrap();
            mem_free(gpu_b).unwrap();
            stream_destroy(stream).unwrap();
        }
    }

    #[test]
    fn test_memset() {
        let _ctx = init_cuda();
        unsafe {
            let stream = stream_create().expect("stream");
            let gpu = mem_alloc(512).expect("gpu");

            // Memset to 0.
            memset_d8(gpu, 0, 512, stream).expect("memset");

            let host = mem_alloc_host(512).expect("host");
            memcpy_dtoh_async(host, gpu, 512, stream).expect("dtoh");
            stream_synchronize(stream).expect("sync");

            for i in 0..512 {
                assert_eq!(*host.add(i), 0, "not zeroed at {i}");
            }

            mem_free_host(host).unwrap();
            mem_free(gpu).unwrap();
            stream_destroy(stream).unwrap();
        }
    }

    #[test]
    fn test_memset_nonzero() {
        let _ctx = init_cuda();
        unsafe {
            let stream = stream_create().expect("stream");
            let gpu = mem_alloc(256).expect("gpu");

            memset_d8(gpu, 0xFF, 256, stream).expect("memset");

            let host = mem_alloc_host(256).expect("host");
            memcpy_dtoh_async(host, gpu, 256, stream).expect("dtoh");
            stream_synchronize(stream).expect("sync");

            for i in 0..256 {
                assert_eq!(*host.add(i), 0xFF, "wrong value at {i}");
            }

            mem_free_host(host).unwrap();
            mem_free(gpu).unwrap();
            stream_destroy(stream).unwrap();
        }
    }

    #[test]
    fn test_event_create_record_wait() {
        let _ctx = init_cuda();
        unsafe {
            let s1 = stream_create().expect("s1");
            let s2 = stream_create().expect("s2");
            let event = event_create_disable_timing().expect("event");

            // Alloc + memset on s1.
            let gpu = mem_alloc(256).expect("gpu");
            memset_d8(gpu, 0x42, 256, s1).expect("memset");

            // Record event on s1, wait on s2.
            event_record(event, s1).expect("record");
            stream_wait_event(s2, event).expect("wait");

            // D2H on s2 — should see the memset data.
            let host = mem_alloc_host(256).expect("host");
            memcpy_dtoh_async(host, gpu, 256, s2).expect("dtoh");
            stream_synchronize(s2).expect("sync");

            assert_eq!(*host, 0x42);

            event_destroy(event).unwrap();
            mem_free_host(host).unwrap();
            mem_free(gpu).unwrap();
            stream_destroy(s1).unwrap();
            stream_destroy(s2).unwrap();
        }
    }

    #[test]
    fn test_graph_capture_memset() {
        let _ctx = init_cuda();
        unsafe {
            let stream = stream_create().expect("stream");
            let gpu = mem_alloc(256).expect("gpu");

            // Begin capture.
            stream_begin_capture(stream).expect("begin capture");

            // Captured op: memset.
            memset_d8(gpu, 0xAB, 256, stream).expect("memset during capture");

            // End capture.
            let graph = stream_end_capture(stream).expect("end capture");
            assert!(!graph.is_null());

            let exec = graph_instantiate(graph).expect("instantiate");

            // Launch graph.
            graph_launch(exec, stream).expect("launch");
            stream_synchronize(stream).expect("sync");

            // Verify.
            let host = mem_alloc_host(256).expect("host");
            memcpy_dtoh_async(host, gpu, 256, stream).expect("dtoh");
            stream_synchronize(stream).expect("sync2");

            assert_eq!(*host, 0xAB);

            graph_exec_destroy(exec).unwrap();
            graph_destroy(graph).unwrap();
            mem_free_host(host).unwrap();
            mem_free(gpu).unwrap();
            stream_destroy(stream).unwrap();
        }
    }

    #[test]
    fn test_multiple_streams() {
        let _ctx = init_cuda();
        unsafe {
            // Create 4 streams — verify they're all distinct.
            let streams: Vec<CUstream> = (0..4).map(|_| stream_create().expect("stream")).collect();

            for i in 0..streams.len() {
                for j in (i + 1)..streams.len() {
                    assert_ne!(
                        streams[i] as usize, streams[j] as usize,
                        "streams {i} and {j} are the same"
                    );
                }
            }

            for s in streams {
                stream_destroy(s).unwrap();
            }
        }
    }

    #[test]
    fn test_large_alloc() {
        let _ctx = init_cuda();
        unsafe {
            // Allocate 64 MB.
            let size = 64 * 1024 * 1024;
            let ptr = mem_alloc(size).expect("large alloc");
            assert!(!ptr.is_null());
            mem_free(ptr).expect("free");
        }
    }
}
