// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
//! Backend-neutral GPU tensor core.
//!
//! The foundational types every backend builds on: [`GpuTensor`] (32-byte
//! descriptor), [`TensorView`] (lifetime-checked borrow), [`DType`], and
//! the [`DeviceAllocator`] host→device transfer trait. Pure metadata +
//! traits — no CUDA, no Metal, no device calls — so it compiles everywhere
//! and gives the `targets/{cuda,metal}` crates a shared home that cannot
//! name a backend.

pub mod device_allocator;
pub mod dtype;
pub mod lowering_dtype;
pub mod owned_tensor;
pub mod precast;
pub mod raw_mem;
pub mod tensor;
pub mod weight_path;
pub mod weight_source;

pub use device_allocator::{DeviceAllocator, PoolMemory};
pub use dtype::DType;
pub use lowering_dtype::{MetalDtype, ScaleDtype};
pub use owned_tensor::OwnedTensor;
pub use precast::{PrecastEntry, PrecastPipeline};
pub use raw_mem::RawGpuMem;
pub use tensor::{GpuTensor, TensorView};
pub use weight_path::{layer_weight_path_with_root, vision_block_weight_path};
pub use weight_source::WeightSource;

/// Backend-neutral load-stream handle: a CUDA stream under the `cuda` feature,
/// otherwise a unit placeholder so backend-agnostic signatures stay cfg-free.
#[cfg(feature = "cuda")]
pub type LoadStream = cudarc::driver::sys::CUstream;
#[cfg(not(feature = "cuda"))]
pub type LoadStream = ();

/// Backend-neutral opaque handle for the ambient forward-args bundle the
/// `ScratchyWeights::forward*` trait methods take.
///
/// The concrete bundle (`scratchy_target_cuda::ForwardCtx`) names the
/// backend-coupled `KvCachePool` / `GdnStatePool` types, so it cannot live in
/// the cfg-free dispatcher. Threading this opaque, lifetime-carrying newtype
/// through the trait (the same neutralization [`LoadStream`] applies to the
/// load stream) keeps the trait method signatures cfg-free of cuda/metal while
/// the per-arch `impl ScratchyWeights` recovers the concrete `&ForwardCtx` via
/// [`ForwardCtxHandle::as_ref`]. Object-safe (a thin pointer; no associated
/// types), so `Box<dyn ScratchyWeights>` and its fat-pointer transmute in the
/// worker are unchanged.
#[derive(Clone, Copy)]
pub struct ForwardCtxHandle<'a> {
    ptr: *const core::ffi::c_void,
    _marker: core::marker::PhantomData<&'a ()>,
}

impl<'a> ForwardCtxHandle<'a> {
    /// Wrap a borrow of the concrete forward-args bundle as the neutral
    /// handle. The borrow's lifetime is carried so the recovered reference
    /// in the callee cannot outlive the bundle.
    #[inline]
    pub fn new<T>(ctx: &'a T) -> Self {
        Self {
            ptr: ctx as *const T as *const core::ffi::c_void,
            _marker: core::marker::PhantomData,
        }
    }

    /// Recover the concrete forward-args bundle reference.
    ///
    /// # Safety
    /// `T` must be the exact concrete type the handle was built from with
    /// [`Self::new`] (every `impl ScratchyWeights` and its caller agree on
    /// `scratchy_target_cuda::ForwardCtx`).
    #[inline]
    pub unsafe fn as_ref<T>(self) -> &'a T {
        &*(self.ptr as *const T)
    }
}

/// Backend-neutral opaque handle for the live device the
/// `ScratchyWeights::forward*` trait methods launch on. The concrete device
/// (`scratchy_target_cuda::GpuDevice`) wraps cublas/streams/allocator and must
/// not move into the leaf, so the trait names this newtype instead and the
/// per-arch `impl ScratchyWeights` recovers `&mut GpuDevice` via
/// [`ForwardDeviceHandle::as_mut`]. Same object-safety story as
/// [`ForwardCtxHandle`].
pub struct ForwardDeviceHandle<'a> {
    ptr: *mut core::ffi::c_void,
    _marker: core::marker::PhantomData<&'a mut ()>,
}

impl<'a> ForwardDeviceHandle<'a> {
    /// Wrap a mutable borrow of the concrete device as the neutral handle.
    #[inline]
    pub fn new<T>(device: &'a mut T) -> Self {
        Self {
            ptr: device as *mut T as *mut core::ffi::c_void,
            _marker: core::marker::PhantomData,
        }
    }

    /// Recover the concrete device reference.
    ///
    /// # Safety
    /// `T` must be the exact concrete type the handle was built from with
    /// [`Self::new`]; the caller must not alias the returned `&mut` while the
    /// handle is live.
    #[inline]
    pub unsafe fn as_mut<T>(self) -> &'a mut T {
        &mut *(self.ptr as *mut T)
    }
}

/// Opaque handle to the backend's `GpuWeights<A>`, erasing the allocator so
/// the cross-arch registry fn-pointers name no backend allocator.
///
/// The cross-arch load registry (`ArchTryLoadFn` / `MmTryLoadFn`) lives in the
/// cfg-free compiler, but its fn-pointers would otherwise have to name the
/// concrete `GpuWeights<BackendAllocator>` weight-store the per-arch loaders
/// fill — a type the compiler cannot express (it has no backend allocator).
/// Threading this opaque, lifetime-carrying newtype through the fn-pointer (the
/// same neutralization [`ForwardDeviceHandle`] applies to the device) keeps the
/// registry signatures free of any backend allocator while `try_load`'s
/// generic wrapper and the macro-emitted closure recover the concrete
/// `&mut GpuWeights<A>` via [`GpuWeightsHandle::as_mut`].
///
/// `Copy` so the registry's `find_map` can thread one handle through each
/// candidate registration's fn-pointer in turn (the fn-pointer takes it by
/// value); exactly one fn-pointer recovers the `&mut`, mirroring the
/// auto-reborrow the old `&mut GpuWeights` argument relied on.
#[derive(Clone, Copy)]
pub struct GpuWeightsHandle<'a> {
    ptr: *mut (),
    _marker: core::marker::PhantomData<&'a mut ()>,
}

impl<'a> GpuWeightsHandle<'a> {
    /// Wrap a mutable borrow of the concrete weight store as the neutral
    /// handle. The borrow's lifetime is carried so the recovered reference in
    /// the callee cannot outlive the store.
    #[inline]
    pub fn new<T>(w: &'a mut T) -> Self {
        Self {
            ptr: w as *mut T as *mut (),
            _marker: core::marker::PhantomData,
        }
    }

    /// Recover the concrete weight-store reference.
    ///
    /// # Safety
    /// `T` must be the exact concrete type the handle was built from with
    /// [`Self::new`]; the caller must not alias the returned `&mut` while the
    /// handle is live.
    #[inline]
    pub unsafe fn as_mut<T>(self) -> &'a mut T {
        &mut *(self.ptr as *mut T)
    }
}

/// Backend-neutral opaque handle for the Metal worker's per-forward
/// `RuntimeBindings` (the Metal analogue of CUDA's `ForwardCtx`: input-id /
/// position / KV-cache / GDN buffers). `RuntimeBindings` is a metal-rs struct
/// in `scratchy-target-metal`, so the cfg-free dispatcher's `MetalChainBody`
/// type cannot name it directly without re-coupling the compiler to the metal
/// target. Threading this thin pointer through the closure keeps the chain
/// body signature target-free; the worker that constructs the body and the
/// macro-emitted `forward_chain_with_encoder` that invokes it both name the
/// concrete `RuntimeBindings` and agree on the recovered type. Same
/// object-safety / lifetime story as [`ForwardCtxHandle`] (shared borrow).
#[derive(Clone, Copy)]
pub struct MetalRuntimeHandle<'a> {
    ptr: *const core::ffi::c_void,
    _marker: core::marker::PhantomData<&'a ()>,
}

impl<'a> MetalRuntimeHandle<'a> {
    /// Wrap a borrow of the concrete `RuntimeBindings` as the neutral handle.
    #[inline]
    pub fn new<T>(runtime: &'a T) -> Self {
        Self {
            ptr: runtime as *const T as *const core::ffi::c_void,
            _marker: core::marker::PhantomData,
        }
    }

    /// Recover the concrete `RuntimeBindings` reference.
    ///
    /// # Safety
    /// `T` must be the exact concrete type the handle was built from with
    /// [`Self::new`] (the worker's body closure and the macro-emitted chain
    /// caller agree on `scratchy_target_metal::interpreter::metal::RuntimeBindings`).
    #[inline]
    pub unsafe fn as_ref<T>(self) -> &'a T {
        &*(self.ptr as *const T)
    }
}
