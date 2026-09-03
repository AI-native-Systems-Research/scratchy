// SPDX-License-Identifier: Apache-2.0
//! `GpuDevice`: the central runtime object combining streams, cuBLAS, and allocator.
//!
//! One `GpuDevice` per GPU. All kernel launches go on `compute_stream`.
//! The `transfer_stream` handles async H2D/D2H copies for overlap.

use crate::alloc::CachingAllocator;
use crate::cublas::CublasHandle;
use crate::driver;
use anyhow::Result;
use cudarc::driver::sys::{CUcontext, CUevent, CUstream};

/// The GPU device runtime. Owns streams, cuBLAS handle, and caching allocator.
///
/// All model forward passes operate through this struct. One instance per GPU.
/// Memory management uses a caching allocator (like PyTorch's CUDACachingAllocator)
/// — tensors are freed on drop and their blocks reused from a free list.
pub struct GpuDevice {
    pub device_id: i32,
    pub ctx: CUcontext,
    pub compute_stream: CUstream,
    pub transfer_stream: CUstream,
    pub cublas: CublasHandle,
    /// Caching allocator — the ONLY allocator. Like PyTorch's CUDACachingAllocator.
    pub caching: CachingAllocator,
    /// Event for gating CPU reuse of pinned buffers after H2D transfer.
    pub transfer_done: CUevent,
    /// Event recorded on transfer_stream after D2H copy for async output.
    pub d2h_done: CUevent,
    /// Number of streaming multiprocessors on this device.
    pub num_sm: i32,
    /// SM version (compute capability): major*10 + minor. E.g. 89 for L40S, 80 for A100.
    pub sm_version: u32,
}

impl GpuDevice {
    /// Initialize a GPU device.
    ///
    /// Creates CUDA context, two streams, cuBLAS handle, and caching allocator.
    pub fn new(device_id: i32) -> Result<Self> {
        unsafe {
            driver::init()?;
            let cu_device = driver::device_get(device_id)?;
            let ctx = driver::ctx_create(cu_device)?;

            let compute_stream = driver::stream_create()?;
            let transfer_stream = driver::stream_create()?;
            let transfer_done = driver::event_create_disable_timing()?;
            let d2h_done = driver::event_create_disable_timing()?;

            let mut caching = CachingAllocator::new();
            let cublas = CublasHandle::new(compute_stream, &mut caching)?;
            let num_sm = driver::device_get_num_sm(cu_device)?;
            let sm_version = driver::device_get_sm_version(cu_device)?;

            tracing::info!(
                "GpuDevice initialized: device={}, SMs={}, SM{}",
                device_id,
                num_sm,
                sm_version,
            );

            Ok(Self {
                device_id,
                ctx,
                compute_stream,
                transfer_stream,
                cublas,
                caching,
                transfer_done,
                d2h_done,
                num_sm,
                sm_version,
            })
        }
    }

    /// Synchronize the compute stream (block until all compute completes).
    pub fn sync_compute(&self) -> Result<()> {
        unsafe { driver::stream_synchronize(self.compute_stream) }
    }

    /// Synchronize the transfer stream.
    pub fn sync_transfer(&self) -> Result<()> {
        unsafe { driver::stream_synchronize(self.transfer_stream) }
    }

    /// Record an event on the transfer stream and make compute wait for it.
    /// Used after H2D copy to ensure inputs are ready before forward pass.
    pub fn sync_transfer_to_compute(&self) -> Result<()> {
        unsafe {
            driver::event_record(self.transfer_done, self.transfer_stream)?;
            driver::stream_wait_event(self.compute_stream, self.transfer_done)?;
        }
        Ok(())
    }

    /// Record event on compute_stream, make transfer_stream wait for it,
    /// then do D2H copy on transfer_stream and record d2h_done event.
    /// The caller must call `sync_d2h()` before reading the host buffer.
    pub unsafe fn async_d2h(
        &self,
        host_dst: *mut u8,
        device_src: *const u8,
        bytes: usize,
    ) -> Result<()> {
        // Gate transfer_stream on compute_stream completion.
        driver::event_record(self.transfer_done, self.compute_stream)?;
        driver::stream_wait_event(self.transfer_stream, self.transfer_done)?;
        // D2H on transfer_stream.
        driver::memcpy_dtoh_async(host_dst, device_src, bytes, self.transfer_stream)?;
        // Record d2h_done so caller can sync on it.
        driver::event_record(self.d2h_done, self.transfer_stream)?;
        Ok(())
    }

    /// Block until the D2H copy initiated by `async_d2h` is complete.
    pub fn sync_d2h(&self) -> Result<()> {
        unsafe { driver::event_synchronize(self.d2h_done) }
    }

    /// Allocate persistent device memory (not from caching allocator).
    /// For weight buffers, KV cache, etc.
    ///
    /// # Safety
    /// Caller owns the returned pointer and must free it with `driver::mem_free`.
    pub unsafe fn alloc_persistent(&self, bytes: usize) -> Result<*mut u8> {
        driver::mem_alloc(bytes)
    }

    /// Async D2D copy on compute stream.
    pub unsafe fn copy_dtod(&self, dst: *mut u8, src: *const u8, bytes: usize) -> Result<()> {
        driver::memcpy_dtod_async(dst, src, bytes, self.compute_stream)
    }

    /// Zero GPU memory on compute stream.
    pub unsafe fn memset_zero(&self, ptr: *mut u8, bytes: usize) -> Result<()> {
        driver::memset_d8(ptr, 0, bytes, self.compute_stream)
    }

    /// Allocate a GPU tensor from the caching allocator, zero-initialized.
    ///
    /// Unlike [`CachingAllocator::alloc_gpu_tensor`], the returned tensor is
    /// guaranteed to contain zeros.  Uses an async memset on the compute stream
    /// so no synchronization is added.
    pub fn alloc_gpu_tensor_zeroed(
        &mut self,
        shape: &[usize],
        dtype: crate::dtype::DType,
    ) -> crate::tensor::GpuTensor {
        let tensor = self.caching.alloc_gpu_tensor(shape, dtype);
        let numel: usize = shape.iter().product();
        let bytes = numel * dtype.size_bytes();
        unsafe {
            // Cannot fail for valid allocations; ignore result like PyTorch.
            let _ = driver::memset_d8(tensor.raw_ptr(), 0, bytes, self.compute_stream);
        }
        tensor
    }

    /// Allocate a GPU tensor and upload host data into it on the compute stream.
    ///
    /// `data` must be a contiguous host buffer whose byte length equals
    /// `numel(shape) * dtype.size_bytes()`.  The copy is async on the compute
    /// stream — the host buffer can be dropped after this call returns (the
    /// driver copies from it before the memcpy_htod_async call returns for
    /// pageable memory).
    pub fn alloc_gpu_tensor_from_host(
        &mut self,
        shape: &[usize],
        dtype: crate::dtype::DType,
        data: &[u8],
    ) -> crate::tensor::GpuTensor {
        let tensor = self.caching.alloc_gpu_tensor(shape, dtype);
        let numel: usize = shape.iter().product();
        let bytes = numel * dtype.size_bytes();
        debug_assert_eq!(
            data.len(),
            bytes,
            "host data length ({}) != tensor size ({})",
            data.len(),
            bytes
        );
        unsafe {
            let _ = driver::memcpy_htod_async(
                tensor.raw_ptr(),
                data.as_ptr(),
                bytes,
                self.compute_stream,
            );
        }
        tensor
    }
}

impl Drop for GpuDevice {
    fn drop(&mut self) {
        unsafe {
            // CachingAllocator is dropped automatically (frees all GPU blocks).
            // CublasHandle is dropped automatically.
            let _ = driver::event_destroy(self.transfer_done);
            let _ = driver::stream_destroy(self.transfer_stream);
            if !self.compute_stream.is_null() {
                let _ = driver::stream_destroy(self.compute_stream);
            }
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::tensor::GpuTensor;

    #[test]
    fn test_device_create_and_drop() {
        let dev = GpuDevice::new(0).expect("GpuDevice::new");
        assert_eq!(dev.device_id, 0);
        assert!(!dev.compute_stream.is_null());
        assert!(!dev.transfer_stream.is_null());
        assert_ne!(
            dev.compute_stream as usize, dev.transfer_stream as usize,
            "compute and transfer streams should differ"
        );
        drop(dev);
    }

    #[test]
    fn test_device_caching_alloc() {
        let mut dev = GpuDevice::new(0).expect("GpuDevice::new");
        let t = dev.caching.alloc_tensor(&[32, 4096], DType::BF16);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.dim(0), 32);
        assert_eq!(t.dim(1), 4096);
    }

    #[test]
    fn test_device_sync_compute() {
        let dev = GpuDevice::new(0).expect("GpuDevice::new");
        dev.sync_compute().expect("sync compute");
    }

    #[test]
    fn test_device_sync_transfer() {
        let dev = GpuDevice::new(0).expect("GpuDevice::new");
        dev.sync_transfer().expect("sync transfer");
    }

    #[test]
    fn test_device_sync_transfer_to_compute() {
        let dev = GpuDevice::new(0).expect("GpuDevice::new");
        dev.sync_transfer_to_compute()
            .expect("sync transfer to compute");
    }

    #[test]
    fn test_device_alloc_persistent() {
        let dev = GpuDevice::new(0).expect("GpuDevice::new");
        unsafe {
            let ptr = dev.alloc_persistent(1024).expect("persistent alloc");
            assert!(!ptr.is_null());
            driver::mem_free(ptr).expect("free");
        }
    }

    #[test]
    fn test_device_memset_zero() {
        let dev = GpuDevice::new(0).expect("GpuDevice::new");
        unsafe {
            let ptr = dev.alloc_persistent(256).expect("alloc");
            dev.memset_zero(ptr, 256).expect("memset");
            dev.sync_compute().expect("sync");

            let host = driver::mem_alloc_host(256).expect("host");
            driver::memcpy_dtoh_async(host, ptr, 256, dev.compute_stream).expect("dtoh");
            driver::stream_synchronize(dev.compute_stream).expect("sync");

            let bytes = std::slice::from_raw_parts(host, 256);
            for (i, &b) in bytes.iter().enumerate() {
                assert_eq!(b, 0, "not zero at byte {i}");
            }

            driver::mem_free_host(host).unwrap();
            driver::mem_free(ptr).unwrap();
        }
    }

    #[test]
    fn test_device_copy_dtod() {
        let dev = GpuDevice::new(0).expect("GpuDevice::new");
        unsafe {
            let src = dev.alloc_persistent(128).expect("src");
            let dst = dev.alloc_persistent(128).expect("dst");

            let host = driver::mem_alloc_host(128).expect("host");
            for i in 0..128 {
                *host.add(i) = (i * 5) as u8;
            }
            driver::memcpy_htod_async(src, host, 128, dev.compute_stream).expect("htod");

            dev.copy_dtod(dst, src, 128).expect("dtod");
            dev.sync_compute().expect("sync");

            let host_out = driver::mem_alloc_host(128).expect("host_out");
            driver::memcpy_dtoh_async(host_out, dst, 128, dev.compute_stream).expect("dtoh");
            driver::stream_synchronize(dev.compute_stream).expect("sync");

            for i in 0..128 {
                assert_eq!(*host_out.add(i), (i * 5) as u8, "mismatch at {i}");
            }

            driver::mem_free_host(host).unwrap();
            driver::mem_free_host(host_out).unwrap();
            driver::mem_free(src).unwrap();
            driver::mem_free(dst).unwrap();
        }
    }

    #[test]
    fn test_device_gemm_f32() {
        let mut dev = GpuDevice::new(0).expect("GpuDevice::new");
        unsafe {
            let host_a = driver::mem_alloc_host(16).unwrap();
            let host_b = driver::mem_alloc_host(16).unwrap();
            std::slice::from_raw_parts_mut(host_a as *mut f32, 4)
                .copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
            std::slice::from_raw_parts_mut(host_b as *mut f32, 4)
                .copy_from_slice(&[2.0, 3.0, 4.0, 5.0]);

            let gpu_a = driver::mem_alloc(16).unwrap();
            let gpu_b = driver::mem_alloc(16).unwrap();
            driver::memcpy_htod_async(gpu_a, host_a, 16, dev.compute_stream).unwrap();
            driver::memcpy_htod_async(gpu_b, host_b, 16, dev.compute_stream).unwrap();

            let a = GpuTensor::new(gpu_a, &[2, 2], DType::F32);
            let b = GpuTensor::new(gpu_b, &[2, 2], DType::F32);

            let c = dev.cublas.gemm(a, b, &mut dev.caching);

            let host_c = driver::mem_alloc_host(16).unwrap();
            driver::memcpy_dtoh_async(host_c, c.as_gpu_tensor().raw_ptr(), 16, dev.compute_stream)
                .unwrap();
            driver::stream_synchronize(dev.compute_stream).unwrap();

            let result = std::slice::from_raw_parts(host_c as *const f32, 4);
            let expected = [2.0, 4.0, 3.0, 5.0];
            for (i, (got, exp)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-3,
                    "gemm mismatch at {i}: got {got}, expected {exp}"
                );
            }

            driver::mem_free_host(host_a).unwrap();
            driver::mem_free_host(host_b).unwrap();
            driver::mem_free_host(host_c).unwrap();
            driver::mem_free(gpu_a).unwrap();
            driver::mem_free(gpu_b).unwrap();
        }
    }
}
