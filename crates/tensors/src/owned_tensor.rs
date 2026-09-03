// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral RAII tensor wrapper.
//!
//! Owns a device allocation behind a [`GpuTensor`] descriptor and releases it
//! on `Drop`. The backend-specific lifecycle is captured in two thunks at
//! construction so this type names no backend:
//! - `free` runs on `Drop` (CUDA returns the block to its caching allocator;
//!   Metal's thunk simply owns the `MTLBuffer`, releasing it when dropped).
//! - `detach` runs in [`OwnedTensor::into_gpu_tensor`] to hand ownership out
//!   without freeing (CUDA unregisters the block from its allocator's active
//!   set; Metal has nothing to detach).

use crate::dtype::DType;
use crate::tensor::{GpuTensor, TensorView};

// Not `Send + Sync`: the cuda thunk captures a raw allocator pointer and the
// metal thunk a retained buffer. The struct's `unsafe impl Send/Sync` below
// asserts the whole `OwnedTensor` is safe to move across threads (same
// contract the previous per-backend struct upheld with its raw fields).
type FreeThunk = Box<dyn FnMut(*mut u8, usize)>;
type DetachThunk = Box<dyn FnOnce(*mut u8)>;

pub struct OwnedTensor {
    inner: GpuTensor,
    size_bytes: usize,
    /// Run on `Drop` with `(ptr, size_bytes)` iff `Some`.
    free: Option<FreeThunk>,
    /// Run by `into_gpu_tensor` with `ptr` iff `Some`, to relinquish
    /// ownership without freeing.
    detach: Option<DetachThunk>,
}

// SAFETY: same invariant the previous per-backend `OwnedTensor` upheld — the
// raw pointer is a device handle whose threading is governed by the backend
// that constructed it; the thunks are `Send + Sync`.
unsafe impl Send for OwnedTensor {}
unsafe impl Sync for OwnedTensor {}

impl OwnedTensor {
    /// Construct from a descriptor + size and backend lifecycle thunks.
    ///
    /// # Safety
    /// `inner` must describe a valid device allocation of `size_bytes` bytes.
    /// `free` (if `Some`) must correctly release it and be safe to call once
    /// on `Drop`; `detach` (if `Some`) must relinquish tracking without
    /// freeing.
    pub unsafe fn from_parts(
        inner: GpuTensor,
        size_bytes: usize,
        free: Option<FreeThunk>,
        detach: Option<DetachThunk>,
    ) -> Self {
        Self {
            inner,
            size_bytes,
            free,
            detach,
        }
    }

    pub fn as_gpu_tensor(&self) -> GpuTensor {
        self.inner
    }

    /// Relinquish ownership of the underlying allocation and return its
    /// descriptor. Runs the `detach` thunk (if any) to stop the backend
    /// tracking the block, then suppresses the `free` thunk.
    pub fn into_gpu_tensor(mut self) -> GpuTensor {
        let t = self.inner;
        if let Some(detach) = self.detach.take() {
            detach(t.raw_ptr());
        }
        self.free = None;
        std::mem::forget(self);
        t
    }

    pub unsafe fn reshape(&mut self, shape: &[usize], dtype: DType) {
        self.inner = GpuTensor::new(self.inner.raw_ptr(), shape, dtype);
    }

    pub fn view(&self) -> TensorView<'_> {
        unsafe { TensorView::from_raw(self.inner) }
    }

    pub unsafe fn view_offset(
        &self,
        byte_offset: usize,
        new_shape: &[usize],
        dtype: DType,
    ) -> TensorView<'_> {
        let inner = GpuTensor::new(self.inner.raw_ptr().add(byte_offset), new_shape, dtype);
        TensorView::from_raw(inner)
    }
}

impl Drop for OwnedTensor {
    fn drop(&mut self) {
        if let Some(mut free) = self.free.take() {
            free(self.inner.raw_ptr(), self.size_bytes);
        }
    }
}

impl std::ops::Deref for OwnedTensor {
    type Target = GpuTensor;
    fn deref(&self) -> &GpuTensor {
        &self.inner
    }
}

impl std::fmt::Debug for OwnedTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OwnedTensor({:?})", self.inner)
    }
}
