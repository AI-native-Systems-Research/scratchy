// SPDX-License-Identifier: Apache-2.0
//! `CpuGpuBuf`: paired pinned-CPU + GPU buffer, pre-allocated at init.
//!
//! Mirrors Python vLLM's `CpuGpuBuffer` exactly. Used for model runner inputs
//! (input_ids, positions, seq_lens, block_tables) that are filled on CPU each
//! step and async-copied to GPU.

use crate::driver;
use crate::dtype::DType;
use crate::tensor::GpuTensor;
use anyhow::Result;
use cudarc::driver::sys::CUstream;

/// Paired pinned-host + device buffer for async H2D/D2H transfers.
///
/// Pre-allocated at init, reused every engine step. Zero allocation on hot path.
pub struct CpuGpuBuf {
    cpu: *mut u8,
    gpu: *mut u8,
    capacity_elements: usize,
    dtype: DType,
}

// Safety: pinned host memory + device memory are accessible from any thread.
unsafe impl Send for CpuGpuBuf {}
unsafe impl Sync for CpuGpuBuf {}

impl CpuGpuBuf {
    /// Allocate a paired buffer with room for `capacity` elements of `dtype`.
    ///
    /// # Safety
    /// Must be called with an active CUDA context.
    pub unsafe fn new(capacity_elements: usize, dtype: DType) -> Result<Self> {
        let bytes = capacity_elements * dtype.size_bytes();
        let cpu = driver::mem_alloc_host(bytes)?;
        let gpu = match driver::mem_alloc(bytes) {
            Ok(ptr) => ptr,
            Err(e) => {
                let _ = driver::mem_free_host(cpu);
                return Err(e);
            }
        };
        Ok(Self {
            cpu,
            gpu,
            capacity_elements,
            dtype,
        })
    }

    /// Capacity in elements.
    pub fn capacity(&self) -> usize {
        self.capacity_elements
    }

    /// Data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Raw pointer to pinned CPU buffer.
    pub fn cpu_ptr(&self) -> *mut u8 {
        self.cpu
    }

    /// Raw pointer to GPU buffer.
    pub fn gpu_ptr(&self) -> *mut u8 {
        self.gpu
    }

    /// CPU buffer as a typed mutable slice.
    ///
    /// # Safety
    /// Caller must ensure `T` matches `self.dtype` and `n <= self.capacity_elements`.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn cpu_slice_mut<T>(&self, n: usize) -> &mut [T] {
        debug_assert!(n <= self.capacity_elements);
        std::slice::from_raw_parts_mut(self.cpu as *mut T, n)
    }

    /// CPU buffer as a typed slice.
    ///
    /// # Safety
    /// Caller must ensure `T` matches `self.dtype`, `n <= self.capacity_elements`,
    /// and the first `n` elements have been initialized.
    pub unsafe fn cpu_slice<T>(&self, n: usize) -> &[T] {
        debug_assert!(n <= self.capacity_elements);
        std::slice::from_raw_parts(self.cpu as *const T, n)
    }

    /// Get a `GpuTensor` view of the first `n` elements as a 1D tensor.
    pub fn gpu_tensor_1d(&self, n: usize) -> GpuTensor {
        debug_assert!(n <= self.capacity_elements);
        unsafe { GpuTensor::new(self.gpu, &[n], self.dtype) }
    }

    /// Get a `GpuTensor` view with the given shape.
    pub fn gpu_tensor(&self, shape: &[usize]) -> GpuTensor {
        let numel: usize = shape.iter().product();
        debug_assert!(numel <= self.capacity_elements);
        unsafe { GpuTensor::new(self.gpu, shape, self.dtype) }
    }

    /// Async copy first `n` elements from CPU to GPU on `stream`.
    ///
    /// # Safety
    /// The first `n` elements of the CPU buffer must be initialized.
    /// The stream must be valid.
    pub unsafe fn copy_to_gpu(&self, n: usize, stream: CUstream) -> Result<()> {
        debug_assert!(n <= self.capacity_elements);
        let bytes = n * self.dtype.size_bytes();
        driver::memcpy_htod_async(self.gpu, self.cpu, bytes, stream)
    }

    /// Async copy first `n` elements from GPU to CPU on `stream`.
    ///
    /// # Safety
    /// The stream must be valid. Results are only available after stream sync
    /// or event wait.
    pub unsafe fn copy_to_cpu(&self, n: usize, stream: CUstream) -> Result<()> {
        debug_assert!(n <= self.capacity_elements);
        let bytes = n * self.dtype.size_bytes();
        driver::memcpy_dtoh_async(self.cpu, self.gpu, bytes, stream)
    }
}

impl Drop for CpuGpuBuf {
    fn drop(&mut self) {
        unsafe {
            let _ = driver::mem_free(self.gpu);
            let _ = driver::mem_free_host(self.cpu);
        }
    }
}

/// Host-only pinned (page-locked) buffer for async H2D staging.
///
/// Unlike `CpuGpuBuf`, this does NOT allocate GPU memory — use this when the
/// GPU destination is managed elsewhere (e.g., CUDA graph persistent buffers).
/// Provides zero-copy async DMA when passed to `memcpy_htod_async`.
pub struct PinnedBuf {
    ptr: *mut u8,
    capacity_bytes: usize,
}

unsafe impl Send for PinnedBuf {}
unsafe impl Sync for PinnedBuf {}

impl PinnedBuf {
    /// Allocate `capacity_bytes` of pinned host memory.
    ///
    /// # Safety
    /// Must be called with an active CUDA context.
    pub unsafe fn new(capacity_bytes: usize) -> Result<Self> {
        let ptr = if capacity_bytes > 0 {
            driver::mem_alloc_host(capacity_bytes)?
        } else {
            std::ptr::null_mut()
        };
        Ok(Self {
            ptr,
            capacity_bytes,
        })
    }

    /// Raw pointer to the pinned host memory.
    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Capacity in bytes.
    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    /// Get a mutable typed slice of the first `n` elements.
    ///
    /// # Safety
    /// `T` must be compatible with the intended use and `n * size_of::<T>()` must
    /// fit in `capacity_bytes`.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice_mut<T>(&self, n: usize) -> &mut [T] {
        debug_assert!(n * std::mem::size_of::<T>() <= self.capacity_bytes);
        std::slice::from_raw_parts_mut(self.ptr as *mut T, n)
    }

    /// Get a typed slice of the first `n` elements.
    ///
    /// # Safety
    /// `T` must be compatible and elements must be initialized.
    pub unsafe fn slice<T>(&self, n: usize) -> &[T] {
        debug_assert!(n * std::mem::size_of::<T>() <= self.capacity_bytes);
        std::slice::from_raw_parts(self.ptr as *const T, n)
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = driver::mem_free_host(self.ptr);
            }
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    fn init_cuda() {
        unsafe {
            driver::init().expect("CUDA init");
            let dev = driver::device_get(0).expect("device");
            let _ctx = driver::ctx_create(dev).expect("context");
        }
    }

    #[test]
    fn test_create_and_drop() {
        init_cuda();
        let buf = unsafe { CpuGpuBuf::new(1024, DType::F32).unwrap() };
        assert_eq!(buf.capacity(), 1024);
        assert_eq!(buf.dtype(), DType::F32);
        assert!(!buf.cpu_ptr().is_null());
        assert!(!buf.gpu_ptr().is_null());
        drop(buf);
    }

    #[test]
    fn test_cpu_slice_write_read() {
        init_cuda();
        let buf = unsafe { CpuGpuBuf::new(64, DType::F32).unwrap() };

        unsafe {
            let slice = buf.cpu_slice_mut::<f32>(64);
            for (i, v) in slice.iter_mut().enumerate() {
                *v = i as f32;
            }

            let read = buf.cpu_slice::<f32>(64);
            for (i, v) in read.iter().enumerate() {
                assert_eq!(*v, i as f32);
            }
        }
    }

    #[test]
    fn test_gpu_tensor_1d() {
        init_cuda();
        let buf = unsafe { CpuGpuBuf::new(256, DType::F16).unwrap() };

        let t = buf.gpu_tensor_1d(128);
        assert_eq!(t.ndim(), 1);
        assert_eq!(t.dim(0), 128);
        assert_eq!(t.dtype(), DType::F16);
        assert_eq!(t.raw_ptr(), buf.gpu_ptr());
    }

    #[test]
    fn test_gpu_tensor_2d() {
        init_cuda();
        let buf = unsafe { CpuGpuBuf::new(512, DType::BF16).unwrap() };

        let t = buf.gpu_tensor(&[16, 32]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.dim(0), 16);
        assert_eq!(t.dim(1), 32);
        assert_eq!(t.numel(), 512);
    }

    #[test]
    fn test_roundtrip_htod_dtoh() {
        init_cuda();
        unsafe {
            let stream = driver::stream_create().expect("stream");
            let buf = CpuGpuBuf::new(128, DType::U32).unwrap();

            // Write pattern to CPU side.
            let cpu = buf.cpu_slice_mut::<u32>(128);
            for (i, v) in cpu.iter_mut().enumerate() {
                *v = (i * 7 + 13) as u32;
            }

            // Copy to GPU.
            buf.copy_to_gpu(128, stream).expect("htod");
            driver::stream_synchronize(stream).expect("sync");

            // Zero the CPU buffer.
            let cpu = buf.cpu_slice_mut::<u32>(128);
            for v in cpu.iter_mut() {
                *v = 0;
            }

            // Copy back from GPU.
            buf.copy_to_cpu(128, stream).expect("dtoh");
            driver::stream_synchronize(stream).expect("sync2");

            // Verify.
            let cpu = buf.cpu_slice::<u32>(128);
            for (i, v) in cpu.iter().enumerate() {
                assert_eq!(*v, (i * 7 + 13) as u32, "mismatch at index {i}");
            }

            driver::stream_destroy(stream).unwrap();
        }
    }

    #[test]
    fn test_partial_copy() {
        init_cuda();
        unsafe {
            let stream = driver::stream_create().expect("stream");
            let buf = CpuGpuBuf::new(1024, DType::F32).unwrap();

            // Write only first 10 elements.
            let cpu = buf.cpu_slice_mut::<f32>(10);
            for (i, v) in cpu.iter_mut().enumerate() {
                *v = i as f32 * 2.0;
            }

            // Copy only first 10 elements to GPU.
            buf.copy_to_gpu(10, stream).expect("htod partial");

            // Zero CPU, copy back only 10.
            let cpu = buf.cpu_slice_mut::<f32>(10);
            for v in cpu.iter_mut() {
                *v = -1.0;
            }
            buf.copy_to_cpu(10, stream).expect("dtoh partial");
            driver::stream_synchronize(stream).expect("sync");

            let cpu = buf.cpu_slice::<f32>(10);
            for (i, v) in cpu.iter().enumerate() {
                assert_eq!(*v, i as f32 * 2.0);
            }

            driver::stream_destroy(stream).unwrap();
        }
    }

    #[test]
    fn test_different_dtypes() {
        init_cuda();
        for dtype in [
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::U32,
            DType::I32,
            DType::I64,
        ] {
            let buf = unsafe { CpuGpuBuf::new(64, dtype).unwrap() };
            assert_eq!(buf.dtype(), dtype);
            assert_eq!(buf.capacity(), 64);
        }
    }

    // -----------------------------------------------------------------------
    // PinnedBuf tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pinned_buf_create_and_drop() {
        init_cuda();
        let buf = unsafe { PinnedBuf::new(4096).unwrap() };
        assert!(!buf.ptr().is_null());
        assert_eq!(buf.capacity_bytes(), 4096);
        drop(buf);
    }

    #[test]
    fn test_pinned_buf_zero_capacity() {
        init_cuda();
        let buf = unsafe { PinnedBuf::new(0).unwrap() };
        assert!(buf.ptr().is_null());
        assert_eq!(buf.capacity_bytes(), 0);
    }

    #[test]
    fn test_pinned_buf_slice_write_read_u32() {
        init_cuda();
        let buf = unsafe { PinnedBuf::new(256 * 4).unwrap() };
        unsafe {
            let s = buf.slice_mut::<u32>(256);
            for (i, v) in s.iter_mut().enumerate() {
                *v = (i * 3 + 7) as u32;
            }
            let r = buf.slice::<u32>(256);
            for (i, v) in r.iter().enumerate() {
                assert_eq!(*v, (i * 3 + 7) as u32);
            }
        }
    }

    #[test]
    fn test_pinned_buf_slice_write_read_i64() {
        init_cuda();
        let buf = unsafe { PinnedBuf::new(64 * 8).unwrap() };
        unsafe {
            let s = buf.slice_mut::<i64>(64);
            for (i, v) in s.iter_mut().enumerate() {
                *v = -(i as i64);
            }
            let r = buf.slice::<i64>(64);
            for (i, v) in r.iter().enumerate() {
                assert_eq!(*v, -(i as i64));
            }
        }
    }

    #[test]
    fn test_pinned_buf_h2d_async() {
        init_cuda();
        unsafe {
            let stream = driver::stream_create().expect("stream");
            let buf = PinnedBuf::new(128 * 4).unwrap();

            // Write pattern into pinned memory.
            let s = buf.slice_mut::<u32>(128);
            for (i, v) in s.iter_mut().enumerate() {
                *v = (i * 11) as u32;
            }

            // Allocate GPU memory and do async H2D from pinned buffer.
            let gpu = driver::mem_alloc(128 * 4).expect("gpu alloc");
            driver::memcpy_htod_async(gpu, buf.ptr(), 128 * 4, stream).expect("htod");

            // D2H back into a second pinned buffer to verify.
            let buf2 = PinnedBuf::new(128 * 4).unwrap();
            driver::memcpy_dtoh_async(buf2.ptr(), gpu, 128 * 4, stream).expect("dtoh");
            driver::stream_synchronize(stream).expect("sync");

            let r = buf2.slice::<u32>(128);
            for (i, v) in r.iter().enumerate() {
                assert_eq!(*v, (i * 11) as u32, "mismatch at {i}");
            }

            driver::mem_free(gpu).unwrap();
            driver::stream_destroy(stream).unwrap();
        }
    }
}
