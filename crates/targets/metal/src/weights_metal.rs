// SPDX-License-Identifier: Apache-2.0
//! Metal-only `GpuWeights` operations: the metal device-memory accessor, the
//! rayon parallel upload helper, and the MLX-affine int4 dequant hop.
//!
//! The neutral `GpuWeights` struct lives in `scratchy-layers`; these methods
//! issue metal-specific work (and the affine-dequant calls into this crate's
//! `layers_quant`), so they ride this extension trait. Callers `use` it.

use anyhow::Result;
use scratchy_layers::weights::{GpuWeights, UploadSrc};
use scratchy_tensors::{DType, GpuTensor};

/// Metal-only `GpuWeights` operations. Implemented for the metal-allocator pool.
pub trait MetalWeightsExt {
    /// MLX-affine int4 dequantize-then-upload (router-gate load).
    fn take_affine_dequant_b4(
        &mut self,
        prefix: &str,
        group_size: u32,
        bits: u32,
        dtype_out: DType,
    ) -> Result<GpuTensor>;

    /// Copy a tensor's (cast-aware) bytes directly into `dst` via a parallel
    /// rayon memcpy. Returns `(bytes_written, shape, dtype)`.
    ///
    /// # Safety
    /// `dst` must be valid for the post-cast byte size; the shared cast scratch
    /// must not be borrowed across the call.
    unsafe fn take_into_metal(
        &mut self,
        name: &str,
        dst: *mut u8,
    ) -> Result<(usize, Vec<usize>, DType)>;

    /// The underlying `MetalAllocator` (loaders call `alloc_uninit` on it).
    fn metal_allocator(&self) -> &crate::MetalAllocator;
}

impl MetalWeightsExt for GpuWeights<crate::MetalAllocator> {
    fn take_affine_dequant_b4(
        &mut self,
        prefix: &str,
        group_size: u32,
        bits: u32,
        dtype_out: DType,
    ) -> Result<GpuTensor> {
        crate::layers_quant::affine_dequant_b4(self, prefix, group_size, bits, dtype_out)
    }

    unsafe fn take_into_metal(
        &mut self,
        name: &str,
        dst: *mut u8,
    ) -> Result<(usize, Vec<usize>, DType)> {
        // Metal has no precast pipeline, so this is always the host path.
        let (ptr, size_bytes, dtype, shape) = match self.take_upload_src(name)? {
            UploadSrc::Host {
                ptr,
                bytes,
                dtype,
                shape,
                ..
            } => (ptr, bytes, dtype, shape),
            _ => unreachable!("metal take_upload_src is always Host (no precast)"),
        };
        // Parallel memcpy via rayon — saturates Apple Silicon system bandwidth
        // (~3x vs a single-threaded copy on the gate_up pack hot path).
        // SAFETY: `ptr`/`dst` are valid for `size_bytes`; chunks are
        // non-overlapping by construction.
        const CHUNK: usize = 4 * 1024 * 1024;
        if size_bytes >= 2 * CHUNK {
            use rayon::prelude::*;
            let src_addr = ptr as usize;
            let dst_addr = dst as usize;
            (0..size_bytes)
                .into_par_iter()
                .step_by(CHUNK)
                .for_each(|off| {
                    let len = (size_bytes - off).min(CHUNK);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            (src_addr + off) as *const u8,
                            (dst_addr + off) as *mut u8,
                            len,
                        );
                    }
                });
        } else {
            unsafe { std::ptr::copy_nonoverlapping(ptr, dst, size_bytes) };
        }
        Ok((size_bytes, shape, dtype))
    }

    fn metal_allocator(&self) -> &crate::MetalAllocator {
        self.allocator()
    }
}
