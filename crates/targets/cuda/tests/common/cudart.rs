// SPDX-License-Identifier: Apache-2.0
//! The few CUDA runtime API (`libcudart`) calls the kernel tests need, declared by hand.
//!
//! `cudart_static` is already linked under `nvcc` (the launchers' `<<<>>>` needs it), so these
//! share the kernels' context and default stream with no setup. They live in `tests/` so no public
//! runtime API is committed before the runtime issue chooses one.

use std::ffi::{CStr, c_char, c_void};

/// `cudaMemcpyHostToDevice`.
const MEMCPY_HOST_TO_DEVICE: i32 = 1;
/// `cudaMemcpyDeviceToHost`.
const MEMCPY_DEVICE_TO_HOST: i32 = 2;
/// `cudaDevAttrComputeCapabilityMajor`.
const ATTR_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
/// `cudaDevAttrComputeCapabilityMinor`.
const ATTR_COMPUTE_CAPABILITY_MINOR: i32 = 76;

unsafe extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaMemset(dev_ptr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetErrorString(error: i32) -> *const c_char;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
}

/// Panic, naming `what` and the CUDA error, unless `code` is `cudaSuccess`.
pub fn check(code: i32, what: &str) {
    if code != 0 {
        panic!("{what} failed: {}", error_string(code));
    }
}

/// `cudaGetErrorString`, with the numeric code.
pub fn error_string(code: i32) -> String {
    // SAFETY: cudaGetErrorString returns a static NUL-terminated string for any code.
    let msg = unsafe { CStr::from_ptr(cudaGetErrorString(code)) };
    format!("CUDA error {code} ({})", msg.to_string_lossy())
}

/// Device 0's compute capability `(major, minor)`. Panics, with the CUDA error, when there is no
/// usable device.
pub fn compute_capability() -> (i32, i32) {
    let (mut major, mut minor) = (0, 0);
    // SAFETY: both out-pointers are valid for one i32.
    let code = unsafe { cudaDeviceGetAttribute(&mut major, ATTR_COMPUTE_CAPABILITY_MAJOR, 0) };
    if code != 0 {
        panic!(
            "the `nvcc` feature's tests need an sm_90 GPU (H100), but reading device 0's compute \
             capability failed: {}",
            error_string(code)
        );
    }
    // SAFETY: as above.
    check(
        unsafe { cudaDeviceGetAttribute(&mut minor, ATTR_COMPUTE_CAPABILITY_MINOR, 0) },
        "cudaDeviceGetAttribute(minor)",
    );
    (major, minor)
}

/// Wait for all work on the device; panics with the error an asynchronous kernel fault leaves.
pub fn synchronize() {
    // SAFETY: no arguments.
    check(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize");
}

/// One `cudaMalloc` allocation of `u16` elements (bf16 bits), freed on drop.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    len: usize,
}

impl DeviceBuffer {
    /// `len` uninitialized `u16` elements on the device.
    pub fn new(len: usize) -> Self {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: `ptr` is a valid out-pointer.
        check(unsafe { cudaMalloc(&mut ptr, len * 2) }, "cudaMalloc");
        Self { ptr, len }
    }

    /// A device copy of `host`.
    pub fn from_host(host: &[u16]) -> Self {
        let buf = Self::new(host.len());
        // SAFETY: `buf` holds exactly `host.len()` elements.
        check(
            unsafe {
                cudaMemcpy(
                    buf.ptr,
                    host.as_ptr().cast(),
                    host.len() * 2,
                    MEMCPY_HOST_TO_DEVICE,
                )
            },
            "cudaMemcpy(host to device)",
        );
        buf
    }

    /// Set every byte of the buffer to `byte`.
    pub fn fill_bytes(&self, byte: u8) {
        // SAFETY: the count is the allocation's size.
        check(
            unsafe { cudaMemset(self.ptr, i32::from(byte), self.len * 2) },
            "cudaMemset",
        );
    }

    /// The whole buffer, copied to the host.
    pub fn to_host(&self) -> Vec<u16> {
        let mut host = vec![0u16; self.len];
        // SAFETY: `host` holds exactly `self.len` elements.
        check(
            unsafe {
                cudaMemcpy(
                    host.as_mut_ptr().cast(),
                    self.ptr,
                    self.len * 2,
                    MEMCPY_DEVICE_TO_HOST,
                )
            },
            "cudaMemcpy(device to host)",
        );
        host
    }

    /// The device pointer, for a kernel launcher.
    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from cudaMalloc and is freed once. A failure here cannot be reported
        // from a destructor, and a sticky error shows up at the next checked call.
        let _ = unsafe { cudaFree(self.ptr) };
    }
}
