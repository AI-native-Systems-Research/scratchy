// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral raw device-memory ownership.
//!
//! A device allocation reduced to its neutral essence: a pointer, a size, and
//! an optional free thunk run on `Drop`. The thunk captures the backend's free
//! routine at construction (CUDA's `driver::mem_free`; Metal passes a thunk
//! that simply owns the `MTLBuffer` so it is released on drop), so this type
//! names no backend. Backends needing richer handles (Metal's `MTLBuffer` +
//! GPU address) wrap this in their own type.

/// Opaque device allocation with a backend-supplied free thunk.
pub struct RawGpuMem {
    ptr: *mut u8,
    size: usize,
    /// Run once on `Drop` with `ptr` iff `Some`. Captured at construction so
    /// the backend's free symbol need not be namable from this neutral crate.
    free: Option<Box<dyn FnMut(*mut u8)>>,
}

// SAFETY: the pointer is a device-allocation handle whose threading is governed
// by the backend that constructed it (same invariant the previous per-backend
// `RawGpuMem` upheld with its raw fields).
unsafe impl Send for RawGpuMem {}
unsafe impl Sync for RawGpuMem {}

impl RawGpuMem {
    /// Construct from a raw pointer + size and an optional free thunk.
    ///
    /// # Safety
    /// `ptr` must be a valid device pointer of at least `size` bytes for the
    /// lifetime of the returned value. If `free` is `Some`, it must correctly
    /// release `ptr` and be safe to call once on `Drop`.
    pub unsafe fn from_parts(
        ptr: *mut u8,
        size: usize,
        free: Option<Box<dyn FnMut(*mut u8)>>,
    ) -> Self {
        Self { ptr, size, free }
    }

    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    // (PoolMemory::ptr forwards to the inherent accessor above.)

    /// Relinquish ownership: returns `(ptr, size)` and suppresses the free
    /// thunk so `Drop` does nothing.
    pub fn leak(mut self) -> (*mut u8, usize) {
        let result = (self.ptr, self.size);
        self.ptr = std::ptr::null_mut();
        self.free = None;
        std::mem::forget(self);
        result
    }
}

impl Drop for RawGpuMem {
    fn drop(&mut self) {
        let Some(mut free) = self.free.take() else {
            return;
        };
        if !self.ptr.is_null() {
            free(self.ptr);
        }
    }
}

impl crate::device_allocator::PoolMemory for RawGpuMem {
    fn ptr(&self) -> *mut u8 {
        self.ptr
    }
}
