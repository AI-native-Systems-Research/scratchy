// SPDX-License-Identifier: Apache-2.0
//! One device allocation: a weight, or a worker's arena.

use std::ffi::c_void;

use super::device::CudaDevice;
use super::ffi;
use super::{CudaError, check};

/// One `cudaMalloc` allocation, freed on drop; or [`DeviceBuffer::empty`], which holds nothing.
///
/// Copies through [`DeviceBuffer::upload`] and [`DeviceBuffer::download`] run on the legacy
/// default stream and wait for it, so they are synchronous. A worker's stream is non-blocking and
/// never waits on that stream, which is why a buffer is complete before any forward reads it.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    bytes: u64,
}

/// The legacy default stream.
const DEFAULT_STREAM: ffi::CudaStreamT = std::ptr::null_mut();

impl DeviceBuffer {
    /// `bytes` of uninitialized device memory, allocated through `device`.
    pub(crate) fn alloc(device: &CudaDevice, bytes: u64) -> Result<Self, CudaError> {
        let len = usize::try_from(bytes).expect("a device allocation's bytes fit the host's usize");
        let ptr = device.malloc(len)?;
        Ok(Self { ptr, bytes })
    }

    /// A buffer of no bytes and no allocation: the entry a worker's weight table holds for a
    /// source that is not a weight (the embedded hidden, whose `weight_bytes` is 0).
    pub fn empty() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            bytes: 0,
        }
    }

    /// A device copy of `host` (bf16 bits), allocated through `device`; returns once it landed.
    pub fn upload(device: &CudaDevice, host: &[u16]) -> Result<Self, CudaError> {
        // SAFETY: `host` is `size_of_val(host)` bytes, borrowed for the call.
        unsafe { Self::upload_raw(device, host.as_ptr().cast(), std::mem::size_of_val(host)) }
    }

    /// A device copy of `host`'s bytes, allocated through `device` (one `cudaMalloc` and one
    /// copy, as [`Self::upload`]); returns once it landed. The weight loader's upload: a weight's
    /// staged bytes, whatever its element type.
    pub fn upload_bytes(device: &CudaDevice, host: &[u8]) -> Result<Self, CudaError> {
        // SAFETY: `host` is `host.len()` bytes, borrowed for the call.
        unsafe { Self::upload_raw(device, host.as_ptr().cast(), host.len()) }
    }

    /// A device copy of the `bytes` host bytes at `host`.
    ///
    /// # Safety
    ///
    /// `host` must point to `bytes` readable host bytes for the whole call.
    unsafe fn upload_raw(
        device: &CudaDevice,
        host: *const c_void,
        bytes: usize,
    ) -> Result<Self, CudaError> {
        let buf = Self::alloc(device, bytes as u64)?;
        // SAFETY: `buf` holds exactly `bytes` bytes, and the caller keeps `host`'s `bytes` bytes
        // readable until the stream wait below has returned.
        check(
            unsafe {
                ffi::cudaMemcpyAsync(
                    buf.ptr,
                    host,
                    bytes,
                    ffi::MEMCPY_HOST_TO_DEVICE,
                    DEFAULT_STREAM,
                )
            },
            "cudaMemcpyAsync(host to device)",
        )?;
        // SAFETY: the legacy default stream always exists.
        check(
            unsafe { ffi::cudaStreamSynchronize(DEFAULT_STREAM) },
            "cudaStreamSynchronize",
        )?;
        Ok(buf)
    }

    /// The whole buffer, copied to the host as bf16 bits, once all work on the legacy default
    /// stream before it has finished.
    pub fn download(&self) -> Result<Vec<u16>, CudaError> {
        let len = usize::try_from(self.bytes / 2).expect("the buffer was allocated on this host");
        let mut host = vec![0u16; len];
        // SAFETY: `host` holds `len * 2 <= self.bytes` bytes, and it stays borrowed until the
        // stream wait below has returned.
        check(
            unsafe {
                ffi::cudaMemcpyAsync(
                    host.as_mut_ptr().cast(),
                    self.ptr,
                    len * 2,
                    ffi::MEMCPY_DEVICE_TO_HOST,
                    DEFAULT_STREAM,
                )
            },
            "cudaMemcpyAsync(device to host)",
        )?;
        // SAFETY: the legacy default stream always exists.
        check(
            unsafe { ffi::cudaStreamSynchronize(DEFAULT_STREAM) },
            "cudaStreamSynchronize",
        )?;
        Ok(host)
    }

    /// The allocation's size.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The device pointer, for a kernel launcher. Null for [`DeviceBuffer::empty`].
    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        // SAFETY: `ptr` came from cudaMalloc and is freed once. A failure here cannot be reported
        // from a destructor; a sticky error shows up at the next checked call.
        let _ = unsafe { ffi::cudaFree(self.ptr) };
    }
}
