// SPDX-License-Identifier: Apache-2.0
//! CUDA device-side vision helpers: K-pad of a vision Linear's weight
//! plus the env-driven intermediate-tensor trace dump. These name CUDA
//! runtime types (`CUstream` / `GpuTensor` / `GpuWeights` / `Linear`),
//! so they live in the target crate; the backend-neutral host glue
//! (`VisionConfig`, `MmMetadata`, cu_seqlens, patch permute) stays in
//! `scratchy-vision`.

use crate::CUstream;
use crate::driver;
use crate::layers::Linear;
use crate::tensor::GpuTensor;
#[cfg(feature = "cuda")]
use crate::weights::CudaWeightsExt;
use crate::weights::GpuWeights;
use anyhow::Result;

/// Pad a `[D0, K]` Linear's weight to `[D0, K_pad]` where
/// `K_pad = round_up(K, 8)`. Tail columns are zero so contribution is
/// preserved. No-op if K is already a multiple of 8.
///
/// # Safety
/// `weights` must outlive the returned `Linear`'s underlying allocation
/// (the new pointer is registered via `record_alloc` so the
/// caching/weights allocator owns it).
pub unsafe fn pad_linear_k_to_mult8(
    linear: Linear,
    weights: &mut GpuWeights,
    stream: CUstream,
) -> Result<Linear> {
    let w = linear.weight;
    debug_assert_eq!(w.ndim(), 2);
    let d0 = w.dim(0);
    let k = w.dim(1);
    let k_pad = k.next_multiple_of(8);
    if k_pad == k {
        return Ok(linear);
    }
    let dtype = w.dtype();
    let elem = dtype.size_bytes();
    let src_pitch = k * elem;
    let dst_pitch = k_pad * elem;
    let total_bytes = d0 * dst_pitch;
    let new_ptr = unsafe { driver::mem_alloc(total_bytes) }?;
    weights.record_alloc(new_ptr, total_bytes);
    unsafe { driver::memset_d8(new_ptr, 0, total_bytes, stream) }?;
    let src_base = w.raw_ptr() as *const u8;
    for r in 0..d0 {
        unsafe {
            driver::memcpy_dtod_async(
                new_ptr.add(r * dst_pitch),
                src_base.add(r * src_pitch),
                src_pitch,
                stream,
            )?;
        }
    }
    let new_w = unsafe { GpuTensor::new(new_ptr, &[d0, k_pad], dtype) };
    Ok(Linear::new(new_w, linear.bias))
}
