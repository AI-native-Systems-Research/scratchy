// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral device allocation + host→device transfer.
//!
//! `GpuWeights` does the disk → host pipeline (mmap, dtype cast,
//! name lookup) once for every backend. The platform-specific bit —
//! "give me `N` bytes of device memory and put these host bytes on
//! it" — lives behind this trait.
//!
//! - **Metal impl** (Apple silicon, unified memory) bump-allocates
//!   inside an arena `MTLBuffer` in `StorageModeShared`. The
//!   "transfer" is a `memcpy(buffer.contents() + offset, src, len)`
//!   — same-RAM copy, no DMA.
//!
//! - **Spyre impl** (`SpyreAllocator`) holds host memory the worker stages
//!   onto the card.

use anyhow::Result;
use std::path::Path;
use std::sync::Arc;

/// A device-memory handle stored by the shared GPU containers
/// (`GpuWeights` / `KvCachePool` / `GdnStatePool`) in `Vec<Self::Mem>`.
///
/// Exposes the device-visible pointer so the generic, backend-neutral pool
/// code can build [`GpuTensor`](crate::tensor::GpuTensor)s from a stored
/// allocation without naming a concrete `RawGpuMem` / `MetalMem`.
pub trait PoolMemory: Send + Sync {
    /// Device-visible pointer to the first byte of this allocation.
    fn ptr(&self) -> *mut u8;
}

/// The platform-specific bit of weight loading: device-side
/// allocation plus host→device byte transfer.
///
/// `GpuWeights` is generic over `A: DeviceAllocator` so backend-
/// specific accessors live on `impl GpuWeights<ConcreteAllocator>`
/// blocks that already know the concrete allocator type — no `Any`
/// downcasts required at any call site.
///
/// Implementations own the device memory they hand out and free it
/// on drop (or transfer ownership out via a backend-specific method).
pub trait DeviceAllocator: Send + Sync {
    /// Allocate `bytes` of device memory and copy `src_host` (host
    /// memory) into it. Returns the device-visible pointer suitable
    /// for [`GpuTensor::new`](crate::tensor::GpuTensor::new).
    ///
    /// The implementation synchronizes before returning, so the
    /// device sees the bytes by the time the caller receives the
    /// pointer.
    ///
    /// # Safety
    ///
    /// `src_host` must point to `bytes` valid host bytes for the
    /// duration of this call. Metal implementations `memcpy` from
    /// `src_host`, so any host memory is fine.
    unsafe fn alloc_and_copy_host(&mut self, src_host: *const u8, bytes: usize) -> Result<*mut u8>;

    /// Variant of [`alloc_and_copy_host`] that promises the resulting
    /// device pointer will only be bound to kernel arguments whose
    /// scalar binding type requires at most `min_align` bytes of
    /// offset alignment (e.g. 2 for `device const half*` /
    /// `device const bfloat*`, 4 for `device const uint32_t*`).
    ///
    /// Metal's zero-copy mmap-alias path is gated on the offset
    /// being a multiple of `MIN_BIND_ALIGN = 16` (covers u32 packed
    /// int4 weights + simdgroup_float4 / any SIMD-wide reads).
    /// `mlx-community` 4bit safetensors land tensor offsets at
    /// `mod 16 = 2`, so the 16-byte gate rejects every F16/BF16
    /// scales/biases/RMSNorm-gain tensor even though those bindings
    /// only do scalar reads. Threading dtype-aware `min_align` lets
    /// those tensors take zero-copy.
    ///
    /// A backend with no analogous gate keeps the default, which
    /// forwards to [`alloc_and_copy_host`] and ignores `min_align`.
    ///
    /// # Safety
    ///
    /// Same as [`alloc_and_copy_host`]. Caller is responsible for
    /// ensuring the returned pointer is only bound to kernels that
    /// read at `min_align` granularity — wider SIMD-vector reads
    /// (e.g. `vec<half, 4>`) require strictly higher alignment.
    ///
    /// [`alloc_and_copy_host`]: Self::alloc_and_copy_host
    unsafe fn alloc_and_copy_host_aligned(
        &mut self,
        src_host: *const u8,
        bytes: usize,
        _min_align: usize,
    ) -> Result<*mut u8> {
        unsafe { self.alloc_and_copy_host(src_host, bytes) }
    }

    /// `bytes` of device memory the host writes in place, from where
    /// [`alloc_and_copy_host`] would copy to: `None` when the device's
    /// memory is not host-visible (spyre), so a caller fills host
    /// memory and copies it instead. Metal's unified memory is.
    ///
    /// # Safety
    ///
    /// No kernel may read the memory before the host has written it.
    ///
    /// [`alloc_and_copy_host`]: Self::alloc_and_copy_host
    unsafe fn alloc_host_writable(&mut self, _bytes: usize) -> Result<Option<*mut u8>> {
        Ok(None)
    }

    // ---- Backend-specific surface that generic `GpuWeights<A>` / pools need ----
    //
    // These let the shared containers stay backend-neutral: they name only
    // `Self::Stream` / `Self::Mem`, never a concrete `MetalMem`. Used monomorphized (one backend per build), never as
    // `dyn DeviceAllocator`, so associated types are fine.

    /// The host→device stream this allocator queues transfers on
    /// (`()` under metal and spyre).
    type Stream;

    /// The device-memory handle this allocator hands out and tracks
    /// (`MetalMem` under metal). Lets generic
    /// `GpuWeights` / KV / GDN pools store `Vec<Self::Mem>` without naming a
    /// backend type. The [`PoolMemory`] bound lets that generic pool code read
    /// back the device pointer to build views.
    type Mem: PoolMemory;

    fn stream(&self) -> Self::Stream;

    /// Track an externally-allocated device buffer. Default: no-op — the
    /// metal arenas own their memory.
    fn push_alloc(&mut self, _alloc: Self::Mem) {}

    /// Drain + transfer ownership of all tracked allocations. Default: empty.
    fn take_allocations(&mut self) -> Vec<Self::Mem> {
        Vec::new()
    }

    /// Untrack one allocation by pointer, leaking its wrapper so the caller
    /// owns the device memory. Default: no-op.
    fn unrecord_alloc(&mut self, _ptr: *mut u8) {}

    /// Register a safetensors mmap for the zero-copy alias path (metal does a
    /// realign-pack so tensor offsets become bind-aligned). Default: no-op.
    fn register_mmap(&self, _path: &Path, _mmap: Arc<memmap2::Mmap>) -> Result<()> {
        Ok(())
    }

    /// Hint the OS to start paging in a freshly-mapped safetensors shard
    /// (`MADV_WILLNEED`) to overlap disk I/O with parsing + the H2D copy.
    /// Metal skips it (it prefaults per-shard inside
    /// [`register_mmap`](Self::register_mmap), keeping the working set small).
    /// Default: no-op.
    fn prefault_mmap(&self, _mmap: &memmap2::Mmap) {}
}
