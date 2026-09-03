// SPDX-License-Identifier: Apache-2.0
//! CUDA implementation of the [`DeviceAllocator`] trait.
//!
//! Wraps `mem_alloc` + `memcpy_htod_async` + `RawGpuMem` so
//! `GpuWeights::take` can drive H2D transfers through a backend-
//! neutral interface. Behaviorally identical to the inline DMA path
//! `weights.rs` used pre-refactor.

use anyhow::Result;
use cudarc::driver::sys::CUstream;

use crate::alloc::RawGpuMem;
use crate::device_allocator::DeviceAllocator;
use crate::driver;

/// CUDA-backed [`DeviceAllocator`]. Holds the H2D stream and the
/// allocation tracker; allocations are RAII-freed on drop or
/// transferred out via [`take_allocations`].
///
/// [`take_allocations`]: CudaAllocator::take_allocations
pub struct CudaAllocator {
    stream: CUstream,
    gpu_allocs: Vec<RawGpuMem>,
}

unsafe impl Send for CudaAllocator {}
unsafe impl Sync for CudaAllocator {}

impl CudaAllocator {
    pub fn new(stream: CUstream) -> Self {
        Self {
            stream,
            gpu_allocs: Vec::new(),
        }
    }

    /// The H2D stream, exposed for backend-specific callers (the
    /// precast pipeline + `take_into`/`take_shard*`) that queue
    /// their own DMAs alongside this allocator's.
    pub fn stream(&self) -> CUstream {
        self.stream
    }

    /// Transfer ownership of all live GPU allocations out. After
    /// this call the allocator's internal tracker is empty; the
    /// allocations are freed when the returned `Vec` is dropped.
    /// Mirrors `GpuWeights::take_gpu_allocs`'s semantics.
    pub fn take_allocations(&mut self) -> Vec<RawGpuMem> {
        std::mem::take(&mut self.gpu_allocs)
    }

    /// Push an externally-allocated `RawGpuMem` onto the tracker —
    /// used by the GGUF loader, which dequantizes weights on the
    /// GPU outside this allocator's main path.
    pub fn push_alloc(&mut self, alloc: RawGpuMem) {
        self.gpu_allocs.push(alloc);
    }

    /// Remove a tracked allocation by raw pointer and `leak()` its
    /// wrapper, leaving the GPU memory live for the caller to free.
    /// Used by quant repack paths that swap in a freshly-allocated
    /// destination buffer.
    pub fn unrecord_alloc(&mut self, ptr: *mut u8) {
        if let Some(pos) = self.gpu_allocs.iter().position(|m| m.ptr() == ptr) {
            let removed = self.gpu_allocs.swap_remove(pos);
            removed.leak();
        }
    }
}

impl DeviceAllocator for CudaAllocator {
    type Stream = CUstream;
    type Mem = RawGpuMem;

    fn stream(&self) -> CUstream {
        CudaAllocator::stream(self)
    }
    fn push_alloc(&mut self, alloc: RawGpuMem) {
        CudaAllocator::push_alloc(self, alloc)
    }
    fn take_allocations(&mut self) -> Vec<RawGpuMem> {
        CudaAllocator::take_allocations(self)
    }
    fn unrecord_alloc(&mut self, ptr: *mut u8) {
        CudaAllocator::unrecord_alloc(self, ptr)
    }
    // register_mmap: cuda uses the trait default (no-op) — cudaMalloc + DMA
    // happens regardless of mmap layout.

    unsafe fn alloc_and_copy_host(&mut self, src_host: *const u8, bytes: usize) -> Result<*mut u8> {
        let gpu_ptr = unsafe { driver::mem_alloc(bytes)? };
        self.gpu_allocs
            .push(unsafe { crate::raw_cuda(gpu_ptr, bytes) });
        unsafe {
            driver::memcpy_htod_async(gpu_ptr, src_host, bytes, self.stream)?;
            // Sync so the caller can release `src_host` (often a
            // shared pinned scratch about to be overwritten by the
            // next caller's cast).
            driver::stream_synchronize(self.stream)?;
        }
        Ok(gpu_ptr)
    }

    fn adopt_raw(&mut self, mem: RawGpuMem) {
        self.gpu_allocs.push(mem);
    }

    fn prefault_mmap(&self, _mmap: &memmap2::Mmap) {
        // Overlap disk readahead with parsing + H2D (cuda only). Unix-only;
        // a no-op elsewhere.
        #[cfg(unix)]
        unsafe {
            libc::madvise(
                _mmap.as_ptr() as *mut libc::c_void,
                _mmap.len(),
                libc::MADV_WILLNEED,
            );
            // Sequential access hint for better readahead chunk sizes.
            libc::madvise(
                _mmap.as_ptr() as *mut libc::c_void,
                _mmap.len(),
                libc::MADV_SEQUENTIAL,
            );
        }
    }

    unsafe fn alloc_and_copy_device(
        &mut self,
        src_dev: *const u8,
        bytes: usize,
    ) -> Result<*mut u8> {
        let gpu_ptr = unsafe { driver::mem_alloc(bytes)? };
        self.gpu_allocs
            .push(unsafe { crate::raw_cuda(gpu_ptr, bytes) });
        unsafe {
            driver::memcpy_dtod_async(gpu_ptr, src_dev, bytes, self.stream)?;
            // Sync before returning so the caller may drop the device
            // source (a precast staging buffer freed on entry drop).
            driver::stream_synchronize(self.stream)?;
        }
        Ok(gpu_ptr)
    }
}
