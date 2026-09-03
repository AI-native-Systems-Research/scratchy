// SPDX-License-Identifier: Apache-2.0
//! Metal constructor for the neutral [`OwnedTensor`].

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use scratchy_tensors::{GpuTensor, OwnedTensor};

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

/// Wrap an `MTLBuffer`-backed allocation as a backend-neutral [`OwnedTensor`].
///
/// The buffer is moved into the `free` thunk, so it stays alive exactly as
/// long as the `OwnedTensor` and is released (ref-count drop) when the tensor
/// drops — Metal needs no explicit free. There is nothing to `detach`.
pub fn owned_from_metal_buffer(inner: GpuTensor, buffer: Buffer, size_bytes: usize) -> OwnedTensor {
    unsafe {
        OwnedTensor::from_parts(
            inner,
            size_bytes,
            Some(Box::new(move |_ptr, _size| {
                let _ = &buffer;
            })),
            None,
        )
    }
}
