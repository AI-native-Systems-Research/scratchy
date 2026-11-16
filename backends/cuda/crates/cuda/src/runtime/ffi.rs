// SPDX-License-Identifier: Apache-2.0
//! The CUDA runtime API (`libcudart`) calls the runtime makes, declared by hand.
//!
//! `build.rs` links `cudart_static` under `nvcc`, and the kernel launchers use that same copy, so
//! these calls share the launchers' context and streams. No binding crate: one would load its own
//! `libcudart` or `libcuda` and put a second runtime instance in the process. These are the only
//! runtime calls declared; every `cudaError_t` is checked where it is returned, so nothing reads
//! or clears the last-error state.

use std::ffi::{c_char, c_void};

/// `cudaError_t`; 0 is `cudaSuccess`.
pub(crate) type CudaErrorT = i32;
/// `cudaStream_t`.
pub(crate) type CudaStreamT = *mut c_void;

/// `cudaMemcpyHostToDevice`.
pub(crate) const MEMCPY_HOST_TO_DEVICE: i32 = 1;
/// `cudaMemcpyDeviceToHost`.
pub(crate) const MEMCPY_DEVICE_TO_HOST: i32 = 2;
/// `cudaDevAttrComputeCapabilityMajor`.
pub(crate) const ATTR_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
/// `cudaDevAttrComputeCapabilityMinor`.
pub(crate) const ATTR_COMPUTE_CAPABILITY_MINOR: i32 = 76;
/// `cudaStreamNonBlocking`: the stream never waits on the legacy default stream.
pub(crate) const STREAM_NON_BLOCKING: u32 = 0x01;
/// `cudaStreamCaptureModeThreadLocal`: a capture refuses only this thread's unsafe calls (a
/// synchronous copy, an allocation) while it records.
pub(crate) const STREAM_CAPTURE_MODE_THREAD_LOCAL: i32 = 1;
/// `cudaGraph_t` and `cudaGraphExec_t`.
pub(crate) type CudaGraphT = *mut c_void;
pub(crate) type CudaGraphExecT = *mut c_void;

unsafe extern "C" {
    pub(crate) fn cudaGetDeviceCount(count: *mut i32) -> CudaErrorT;
    pub(crate) fn cudaSetDevice(device: i32) -> CudaErrorT;
    pub(crate) fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> CudaErrorT;
    pub(crate) fn cudaStreamCreateWithFlags(stream: *mut CudaStreamT, flags: u32) -> CudaErrorT;
    pub(crate) fn cudaStreamSynchronize(stream: CudaStreamT) -> CudaErrorT;
    pub(crate) fn cudaStreamDestroy(stream: CudaStreamT) -> CudaErrorT;
    pub(crate) fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> CudaErrorT;
    pub(crate) fn cudaFree(dev_ptr: *mut c_void) -> CudaErrorT;
    pub(crate) fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: i32,
        stream: CudaStreamT,
    ) -> CudaErrorT;
    pub(crate) fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> CudaErrorT;
    pub(crate) fn cudaGetErrorString(error: CudaErrorT) -> *const c_char;
    // The CUDA Graph calls (#1048): one capture per bucket at worker construction, one launch per
    // forward. `flags` 0 is the default instantiation.
    pub(crate) fn cudaStreamBeginCapture(stream: CudaStreamT, mode: i32) -> CudaErrorT;
    pub(crate) fn cudaStreamEndCapture(stream: CudaStreamT, graph: *mut CudaGraphT) -> CudaErrorT;
    pub(crate) fn cudaGraphInstantiateWithFlags(
        exec: *mut CudaGraphExecT,
        graph: CudaGraphT,
        flags: u64,
    ) -> CudaErrorT;
    pub(crate) fn cudaGraphLaunch(exec: CudaGraphExecT, stream: CudaStreamT) -> CudaErrorT;
    pub(crate) fn cudaGraphExecDestroy(exec: CudaGraphExecT) -> CudaErrorT;
    pub(crate) fn cudaGraphDestroy(graph: CudaGraphT) -> CudaErrorT;
}
