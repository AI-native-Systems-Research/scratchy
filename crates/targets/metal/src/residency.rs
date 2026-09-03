// SPDX-License-Identifier: Apache-2.0
//
// Wrapper around `MTLResidencySet` (macOS 15+ / iOS 18+).
// Pins MTL allocations as resident across command buffers so Apple's
// lazy paging doesn't inject non-deterministic stale-page reads when
// the working set is large enough to push the implicit-residency
// tracker over its threshold.
//
// Mirrors `mlx/backend/metal/resident.{h,cpp}` from the MLX repo —
// same API shape (`new`, `insert`, `commit`),
// adapted to objc2. On older macOS the wrapper is a no-op (the
// objc2-metal `Device` and `CommandQueue` paths still work; just no
// pinning).

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Bool, ProtocolObject};
use objc2::{class, msg_send, sel};
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLDevice};
use std::sync::{Arc, Mutex};

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;

#[derive(Clone)]
pub struct MetalResidencySet {
    inner: Arc<Mutex<Inner>>,
}

struct Inner {
    /// `MTLResidencySet*` (or null on macOS < 15 / unsupported devices).
    set_ptr: *mut AnyObject,
    /// Whether `requestResidency` was issued at construction. When false
    /// (an *un-wired* set, see [`MetalResidencySet::new_unwired`]) the set
    /// never persistently pins its allocations — members are made resident
    /// only per-command-buffer via `useResidencySet:`. Teardown then skips
    /// the `endResidency` half of the pair (unbalanced `endResidency` is
    /// undefined when no request was ever made).
    requested: bool,
    /// Whether `endResidency` has already been issued (via [`MetalResidencySet::shutdown`]).
    /// Guards against ending the residency request twice (request was made once).
    ended: bool,
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        if !self.set_ptr.is_null() {
            unsafe {
                // Undo the system-wired residency request and drop every pinned
                // allocation BEFORE releasing the set, so the GPU driver reclaims
                // the wired pages deterministically. A bare `release` can leave
                // the residency request dangling if the set is retained anywhere
                // (e.g. still attached to a command queue), stranding wired GPU
                // memory after the process exits. Skip the end-pair if
                // `shutdown()` already issued it. An un-wired set (never
                // `requestResidency`'d) has no residency request to end —
                // just drop its allocations before release.
                if !self.ended {
                    let _: () = msg_send![self.set_ptr, removeAllAllocations];
                    let _: () = msg_send![self.set_ptr, commit];
                    if self.requested {
                        let _: () = msg_send![self.set_ptr, endResidency];
                    }
                }
                let _: () = msg_send![self.set_ptr, release];
            }
        }
    }
}

impl MetalResidencySet {
    /// Build a **wired** residency set: after construction the set issues
    /// `requestResidency`, so every inserted allocation is pinned resident
    /// (non-pageable) until teardown. Correct for the transient working set
    /// (activation arenas, KV cache) that the GPU touches every forward and
    /// that must not be paged out mid-decode.
    pub fn new(device: &Device) -> Self {
        Self::build(device, true)
    }

    /// Build an **un-wired** residency set: the set is created and its
    /// allocations can be inserted/committed, but `requestResidency` is
    /// NOT issued. Members are therefore made resident only for the
    /// lifetime of a command buffer that declares the set via
    /// `useResidencySet:` — resident *while a forward executes*, pageable
    /// between forwards. Correct for immutable, mmap-backed weights: MTL4
    /// still needs them declared (no implicit residency tracking), but
    /// pinning multi-GB of read-only weights permanently is what drives
    /// the machine into wired-memory pressure / OOM.
    pub fn new_unwired(device: &Device) -> Self {
        Self::build(device, false)
    }

    fn build(device: &Device, wire: bool) -> Self {
        let set_ptr = unsafe { try_create_residency_set(device) };
        if !set_ptr.is_null() && wire {
            unsafe {
                let _: () = msg_send![set_ptr, requestResidency];
            }
        }
        Self {
            inner: Arc::new(Mutex::new(Inner {
                set_ptr,
                requested: wire && !set_ptr.is_null(),
                ended: false,
            })),
        }
    }

    /// Deterministically end the residency request and drop every pinned
    /// allocation. Call on graceful shutdown (SIGTERM/Ctrl-C → worker
    /// teardown) BEFORE dropping the device, so the GPU driver reclaims the
    /// system-wired pages immediately rather than leaving them for the
    /// kernel's process-teardown reclaim — which is intermittent when GPU
    /// work was in flight and orphans wired GPU memory (esp. on `kill -9`
    /// mid-decode, which can't reach this at all). Idempotent.
    pub fn shutdown(&self) {
        let mut inner = self.inner.lock().expect("residency set mutex");
        if inner.set_ptr.is_null() || inner.ended {
            return;
        }
        unsafe {
            let _: () = msg_send![inner.set_ptr, removeAllAllocations];
            let _: () = msg_send![inner.set_ptr, commit];
            if inner.requested {
                let _: () = msg_send![inner.set_ptr, endResidency];
            }
        }
        inner.ended = true;
    }

    pub fn is_active(&self) -> bool {
        let inner = self.inner.lock().expect("residency set mutex");
        !inner.set_ptr.is_null()
    }

    pub fn insert(&self, buffer: &Buffer) {
        let inner = self.inner.lock().expect("residency set mutex");
        if inner.set_ptr.is_null() {
            return;
        }
        unsafe {
            let buf_ptr: *mut AnyObject =
                Retained::as_ptr(buffer) as *const AnyObject as *mut AnyObject;
            let _: () = msg_send![inner.set_ptr, addAllocation: buf_ptr];
        }
    }

    /// Remove a previously-inserted allocation from the residency set
    /// (reactive shrink). Pair with [`Self::commit`] to apply, then the
    /// caller may free the buffer — its pages are no longer wired.
    pub fn remove(&self, buffer: &Buffer) {
        let inner = self.inner.lock().expect("residency set mutex");
        if inner.set_ptr.is_null() {
            return;
        }
        unsafe {
            let buf_ptr: *mut AnyObject =
                Retained::as_ptr(buffer) as *const AnyObject as *mut AnyObject;
            let _: () = msg_send![inner.set_ptr, removeAllocation: buf_ptr];
        }
    }

    pub fn commit(&self) {
        let inner = self.inner.lock().expect("residency set mutex");
        if inner.set_ptr.is_null() {
            return;
        }
        unsafe {
            let _: () = msg_send![inner.set_ptr, commit];
        }
    }

    /// Invoke `useResidencySet:` on a `MTL4CommandBuffer`. MTL4
    /// command buffers declare residency per command buffer, so each
    /// fresh `MTL4CommandBuffer` needs this call between
    /// `beginCommandBufferWithAllocator` and `endCommandBuffer`.
    /// Takes a raw `*mut AnyObject` rather than a typed reference so
    /// the kernels crate can stay free of an `objc2-metal` MTL4 dep.
    ///
    /// # Safety
    /// `cb_ptr` must be a non-null pointer to a live MTL4CommandBuffer
    /// in the recording state.
    pub unsafe fn attach_to_mtl4_command_buffer(&self, cb_ptr: *mut AnyObject) {
        let inner = self.inner.lock().expect("residency set mutex");
        if inner.set_ptr.is_null() || cb_ptr.is_null() {
            return;
        }
        let _: () = msg_send![cb_ptr, useResidencySet: inner.set_ptr];
    }
}

/// Build a residency set on `device`. Returns null on macOS < 15 or
/// on any failure (device family check fails, descriptor alloc
/// fails, `newResidencySet:error:` returns nil).
unsafe fn try_create_residency_set(device: &Device) -> *mut AnyObject {
    let device_ptr: *mut AnyObject = Retained::as_ptr(device) as *const AnyObject as *mut AnyObject;

    // Probe for `newResidencySetWithDescriptor:error:` selector. The
    // selector exists on macOS 15+ runtimes; on older systems
    // `respondsToSelector:` returns NO and we bail out cleanly rather
    // than triggering an unrecognized-selector exception.
    let sel_check = sel!(newResidencySetWithDescriptor:error:);
    let responds: Bool = msg_send![device_ptr, respondsToSelector: sel_check];
    if !responds.as_bool() {
        return std::ptr::null_mut();
    }

    // Build a default `MTLResidencySetDescriptor` (no label, no
    // initialCapacity hints). MLX uses the same path.
    let desc_class = class!(MTLResidencySetDescriptor);
    let desc: *mut AnyObject = msg_send![desc_class, alloc];
    let desc: *mut AnyObject = msg_send![desc, init];
    if desc.is_null() {
        return std::ptr::null_mut();
    }
    let _: *mut AnyObject = msg_send![desc, autorelease];

    let mut error: *mut AnyObject = std::ptr::null_mut();
    let set: *mut AnyObject = msg_send![
        device_ptr,
        newResidencySetWithDescriptor: desc,
        error: &mut error
    ];
    if set.is_null() {
        return std::ptr::null_mut();
    }
    set
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLResourceOptions;

    #[test]
    fn construction_does_not_panic() {
        let Some(device_info) = crate::detect_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let set = MetalResidencySet::new(&device_info.device);
        let _ = set.is_active();
        set.commit();
    }

    #[test]
    fn insert_and_commit_roundtrip() {
        let Some(device_info) = crate::detect_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let device = device_info.device.clone();
        let set = MetalResidencySet::new(&device);

        let buf = device
            .newBufferWithLength_options(1024, MTLResourceOptions::StorageModeShared)
            .expect("newBufferWithLength");
        set.insert(&buf);
        set.commit();
    }
}
