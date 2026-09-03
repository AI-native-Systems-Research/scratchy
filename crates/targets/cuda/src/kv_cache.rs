// SPDX-License-Identifier: Apache-2.0
//! Re-export of the backend-neutral [`KvCachePool`](scratchy_layers::kv_cache::KvCachePool)
//! plus the CUDA extension trait that drives its FP8 / span / gather paths.
//!
//! The pool storage + layout is generic over `M: PoolMemory` and lives in
//! `scratchy-layers`; the defaulted alias preserves the historical
//! `scratchy_target_cuda::kv_cache::KvCachePool` (with `M = PoolMem`) path.
//! The methods that issue CUDA `driver` calls (`set_k_scale` / `set_v_scale`,
//! `gather_kv_contiguous`, `sync_block_flags_to_gpu`) cannot live in the
//! neutral crate, so they ride this extension trait over the pool's public
//! accessors.

/// Backend-neutral paged KV cache pool, with the per-backend mem type defaulted
/// to this crate's [`PoolMem`](crate::PoolMem).
pub type KvCachePool<M = crate::PoolMem> = scratchy_layers::kv_cache::KvCachePool<M>;

#[cfg(feature = "cuda")]
pub use cuda_ext::CudaKvCacheExt;

#[cfg(feature = "cuda")]
mod cuda_ext {
    use crate::driver;
    use scratchy_tensors::GpuTensor;

    /// CUDA-only KV-cache operations that issue `driver` DMAs — kept out of the
    /// neutral `scratchy-layers` pool. Implemented for the concrete
    /// cuda-allocator pool; callers `use` this trait to get method syntax.
    pub trait CudaKvCacheExt {
        /// Set K scale for a layer from a host value.
        ///
        /// # Safety
        /// Requires a valid CUDA context + stream.
        unsafe fn set_k_scale(&self, layer: usize, val: f32, stream: crate::CUstream);
        /// Set V scale for a layer from a host value. See [`Self::set_k_scale`].
        ///
        /// # Safety
        /// Requires a valid CUDA context + stream.
        unsafe fn set_v_scale(&self, layer: usize, val: f32, stream: crate::CUstream);
        /// Gather K or V from blocks into a contiguous
        /// `[total_tokens, kv_heads, head_dim]` tensor (CPU-driven D2D copy —
        /// correct, not fast; the paged-FA2 fallback).
        ///
        /// # Safety
        /// Requires a valid CUDA context + stream; `arena` must outlive the
        /// returned tensor.
        #[allow(clippy::too_many_arguments)]
        unsafe fn gather_kv_contiguous(
            &self,
            layer: usize,
            is_key: bool,
            block_table: GpuTensor,
            total_tokens: usize,
            num_kv_heads: usize,
            head_dim: usize,
            arena: &mut crate::arena::ScratchArena,
            stream: crate::CUstream,
        ) -> GpuTensor;
        /// Lazily allocate + upload the rotation/span flag GPU mirrors.
        ///
        /// # Safety
        /// Requires a valid CUDA context + stream.
        unsafe fn sync_block_flags_to_gpu(&mut self, stream: crate::CUstream);
    }

    impl CudaKvCacheExt for super::KvCachePool<crate::PoolMem> {
        unsafe fn set_k_scale(&self, layer: usize, val: f32, stream: crate::CUstream) {
            driver::memcpy_htod_async(
                self.k_scale_ptr_mut(layer) as *mut u8,
                &val as *const f32 as *const u8,
                4,
                stream,
            )
            .expect("set_k_scale H2D");
        }

        unsafe fn set_v_scale(&self, layer: usize, val: f32, stream: crate::CUstream) {
            driver::memcpy_htod_async(
                self.v_scale_ptr_mut(layer) as *mut u8,
                &val as *const f32 as *const u8,
                4,
                stream,
            )
            .expect("set_v_scale H2D");
        }

        unsafe fn gather_kv_contiguous(
            &self,
            layer: usize,
            is_key: bool,
            block_table: GpuTensor,
            total_tokens: usize,
            num_kv_heads: usize,
            head_dim: usize,
            arena: &mut crate::arena::ScratchArena,
            stream: crate::CUstream,
        ) -> GpuTensor {
            let cache = if is_key {
                self.k_cache_raw(layer)
            } else {
                self.v_cache_raw(layer)
            };
            let out = arena.alloc(&[total_tokens, num_kv_heads, head_dim], cache.dtype());

            // D2H the block table to get block IDs on CPU.
            let num_blocks_in_table = block_table.dim(1);
            let mut block_ids = vec![0i32; num_blocks_in_table];
            driver::memcpy_dtoh_async(
                block_ids.as_mut_ptr() as *mut u8,
                block_table.raw_ptr() as *const u8,
                num_blocks_in_table * 4,
                stream,
            )
            .expect("D2H block_table");
            driver::stream_synchronize(stream).expect("sync block_table D2H");

            // Copy block by block. Cache layout: [num_blocks, block_size, kv_heads, head_dim].
            let elem_size = cache.dtype().size_bytes();
            let tokens_per_block = self.block_size;
            let row_bytes = num_kv_heads * head_dim * elem_size;
            let block_stride_bytes = tokens_per_block * row_bytes;

            let mut tokens_remaining = total_tokens;
            let mut dst_offset: usize = 0;
            for &bid in &block_ids {
                if tokens_remaining == 0 {
                    break;
                }
                let n = tokens_remaining.min(tokens_per_block);
                let src = (cache.raw_ptr() as *const u8).add(bid as usize * block_stride_bytes);
                let dst = (out.raw_ptr() as *mut u8).add(dst_offset);
                driver::memcpy_dtod_async(dst, src, n * row_bytes, stream)
                    .expect("D2D gather block");
                dst_offset += n * row_bytes;
                tokens_remaining -= n;
            }

            out
        }

        unsafe fn sync_block_flags_to_gpu(&mut self, stream: crate::CUstream) {
            // Lazily allocate the one-byte-per-block GPU mirror buffers.
            self.ensure_block_flag_mirrors(|n| {
                let p = unsafe { driver::mem_alloc(n)? };
                Ok(unsafe { crate::raw_cuda(p, n) })
            })
            .expect("alloc block-flag mirrors");

            let unrot = self.block_unrotated_gpu() as *mut u8;
            if !unrot.is_null() {
                let flags: Vec<u8> = self
                    .block_is_unrotated
                    .iter()
                    .map(|&b| u8::from(b))
                    .collect();
                driver::memcpy_htod_async(unrot, flags.as_ptr(), flags.len(), stream)
                    .expect("sync block_unrotated H2D");
            }
            let span = self.block_span_gpu() as *mut u8;
            if !span.is_null() {
                let flags: Vec<u8> = self.block_is_span.iter().map(|&b| u8::from(b)).collect();
                driver::memcpy_htod_async(span, flags.as_ptr(), flags.len(), stream)
                    .expect("sync block_span H2D");
            }
        }
    }
}
