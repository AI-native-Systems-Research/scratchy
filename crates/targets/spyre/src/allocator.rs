// SPDX-License-Identifier: Apache-2.0
//! Host-memory allocator for the Spyre/KTIR backend.

use anyhow::Result;
use scratchy_tensors::{DeviceAllocator, PoolMemory};

/// Host-memory "device" allocator for Spyre/KTIR.
///
/// Spyre/KTIR weights live in plain host RAM (the `ktir-emulator` emulator and the
/// AIU both consume host-resident bundles), so the [`DeviceAllocator`]
/// contract — "allocate `bytes` and copy `src_host` onto it" — is a host
/// allocation + `memcpy`. The allocator owns every buffer it hands out and frees
/// them on drop; the returned pointers stay valid for the allocator's lifetime
/// (the same ownership model as the CUDA/Metal allocators).
#[derive(Default)]
pub struct SpyreAllocator {
    /// Owns the host buffers backing the pointers returned below.
    allocs: Vec<Box<[u8]>>,
}

impl SpyreAllocator {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Device-memory witness for the host-backed Spyre allocator. The buffers are
/// owned by [`SpyreAllocator::allocs`] (the same model metal uses for its
/// arenas), so the `take_allocations` / `push_alloc` tracking hooks stay at
/// their no-op defaults and no `SpyreMem` is ever constructed — it exists only
/// to satisfy the [`DeviceAllocator::Mem`] associated type (`Mem: PoolMemory`)
/// that generic `GpuWeights<A>` / pool code names.
pub struct SpyreMem(*mut u8);
// SAFETY: the pointer addresses a host buffer owned for the allocator's
// lifetime; `SpyreMem` is never actually instantiated (see above).
unsafe impl Send for SpyreMem {}
unsafe impl Sync for SpyreMem {}
impl PoolMemory for SpyreMem {
    fn ptr(&self) -> *mut u8 {
        self.0
    }
}

impl DeviceAllocator for SpyreAllocator {
    type Stream = ();
    type Mem = SpyreMem;

    fn stream(&self) {}

    unsafe fn alloc_and_copy_host(&mut self, src_host: *const u8, bytes: usize) -> Result<*mut u8> {
        let mut buf = vec![0u8; bytes].into_boxed_slice();
        if bytes > 0 {
            // SAFETY: the caller guarantees `src_host` is valid for `bytes`
            // bytes; `buf` is exactly `bytes` long and freshly allocated, so the
            // ranges cannot overlap.
            unsafe { std::ptr::copy_nonoverlapping(src_host, buf.as_mut_ptr(), bytes) };
        }
        let ptr = buf.as_mut_ptr();
        self.allocs.push(buf);
        Ok(ptr)
    }
}
