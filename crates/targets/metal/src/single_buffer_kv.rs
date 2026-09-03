// SPDX-License-Identifier: Apache-2.0
//
//! Per-layer single-buffer KV storage for the reactive (chunked) metal KV pool.
//!
//! One `SingleBufferKvLayer` owns ONE `StorageModePrivate` MTLBuffer per
//! `(layer, K/V)`, sized to span the maximum chunk count for the model. The
//! `KvCachePool`'s logical "chunks" are byte offsets within this buffer
//! (`chunk_idx * chunk_bytes`) rather than separate `MTLBuffer` allocations.
//!
//! Two properties this gives us:
//!
//! 1. **One VA range per layer-kv.** The reactive chunked design's original
//!    "N small chunk buffers per layer" exposed `O(num_layers * num_chunks)`
//!    distinct GPU virtual address ranges to the residency set + UAT. With
//!    one buffer per layer-kv, the kernel-time VA-range count is constant in
//!    the chunk count — the GPU sees just `2 * num_layers` distinct ranges
//!    plus the per-layer chunk-address tables. (This alone is a wash with
//!    the multi-chunk design — the real win is the BPC=0 kernel fast path
//!    layered on top; see `attention.metal`.)
//!
//! 2. **Lazy physical commit via Apple's pager.** The buffer's VA is reserved
//!    up-front but `StorageModePrivate` doesn't pre-fault the pages — Apple's
//!    pager only commits physical pages on first GPU access. The reactive
//!    pool's "1 chunk at init, grow on demand" behavior is preserved: at
//!    idle, only the touched chunks' pages are physically resident.
//!
//! The pool's chunk-address table is filled with `layer_base + chunk_idx *
//! chunk_bytes` per chunk. The decode attention reader's BPC=0 fast path
//! treats `chunk_table[0]` as the layer base and reads
//! `(half*)chunk_table[0] + physical_block * kv_blk_stride`, which equals
//! the multi-chunk addressing math by construction.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

pub type Result<T> = std::result::Result<T, String>;

macro_rules! bail {
    ($($t:tt)*) => { return Err(format!($($t)*)) };
}

/// One per `(layer, K/V)`. Pools `Vec<SingleBufferKvLayer>` are indexed
/// `layer * 2 + kv_idx` (kv_idx = 0 for K, 1 for V) — matches the order
/// `KvCachePool::new_metal_chunked` calls `alloc_chunk` in.
pub struct SingleBufferKvLayer {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    chunk_bytes: usize,
    committed_chunks: usize,
    max_chunks: usize,
    base_address: u64,
}

impl SingleBufferKvLayer {
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        chunk_bytes: usize,
        max_chunks: usize,
    ) -> Result<Self> {
        if max_chunks == 0 {
            bail!("SingleBufferKvLayer::new: max_chunks must be > 0");
        }
        if chunk_bytes == 0 {
            bail!("SingleBufferKvLayer::new: chunk_bytes must be > 0");
        }
        let buf_bytes = chunk_bytes
            .checked_mul(max_chunks)
            .ok_or_else(|| "layer buffer size overflows usize".to_string())?;
        let buf = device
            .newBufferWithLength_options(buf_bytes, MTLResourceOptions::StorageModePrivate)
            .ok_or_else(|| {
                format!(
                    "newBufferWithLength_options(StorageModePrivate, len={buf_bytes}) returned nil"
                )
            })?;
        let base_address = buf.gpuAddress();
        Ok(Self {
            buf,
            chunk_bytes,
            committed_chunks: 0,
            max_chunks,
            base_address,
        })
    }

    /// Mark chunks `[committed_chunks, target_chunks)` as in-use. No physical
    /// allocation happens here — Apple's pager commits pages on first GPU
    /// access. Bookkeeping only, so the alloc-chunk closure in
    /// `KvCachePool::new_metal_chunked` / `grow_to_cover` can derive the
    /// next chunk's byte offset from `committed_chunks`.
    pub fn commit_through(&mut self, target_chunks: usize) -> Result<()> {
        if target_chunks > self.max_chunks {
            bail!(
                "SingleBufferKvLayer::commit_through: target={target_chunks} > max_chunks={}",
                self.max_chunks
            );
        }
        if target_chunks > self.committed_chunks {
            self.committed_chunks = target_chunks;
        }
        Ok(())
    }

    /// Pair to `commit_through`: drop the bookkeeping counter back to `keep`.
    /// No explicit page release available without a range-level
    /// `setPurgeableState`; trailing chunks' pages stay backed until Apple's
    /// pager evicts them under memory pressure.
    pub fn shrink_to(&mut self, keep: usize) -> Result<()> {
        if keep < self.committed_chunks {
            self.committed_chunks = keep;
        }
        Ok(())
    }

    /// gpuAddress of chunk `chunk_idx`'s first byte (`base + chunk_idx *
    /// chunk_bytes`). Used by `fill_chunk_tables` to populate the per-layer
    /// chunk-address table the kernel binds at the `KvCacheK/V` slot.
    pub fn gpu_address_of_chunk(&self, chunk_idx: usize) -> u64 {
        self.base_address + (chunk_idx * self.chunk_bytes) as u64
    }

    pub fn chunk_byte_offset(&self, chunk_idx: usize) -> usize {
        chunk_idx * self.chunk_bytes
    }

    /// Borrow the underlying MTLBuffer so the worker can insert it into the
    /// `MTLResidencySet` exactly once for the buffer's lifetime.
    pub fn buffer(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.buf
    }

    /// Clone of `Retained` — shares ownership without freeing. The pool
    /// packages chunk sub-ranges as `MetalMem::from_buffer_with_offset`
    /// wrapping clones of this buffer.
    pub fn buffer_clone(&self) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.buf.clone()
    }

    pub fn chunk_bytes(&self) -> usize {
        self.chunk_bytes
    }

    pub fn committed_chunks(&self) -> usize {
        self.committed_chunks
    }

    pub fn max_chunks(&self) -> usize {
        self.max_chunks
    }
}

unsafe impl Send for SingleBufferKvLayer {}
unsafe impl Sync for SingleBufferKvLayer {}
