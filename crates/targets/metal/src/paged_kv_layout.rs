// SPDX-License-Identifier: Apache-2.0
//! Canonical paged-KV-cache layout.
//!
//! Captures the single invariant
//!
//! ```text
//! KV cache buffer shape: [num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]
//! ```
//!
//! and exposes one definition of the four strides that address into it:
//!
//! * `kv_blk_stride`   — elements between block_id and block_id + 1
//!   (= `num_kv_heads * BLOCK_SIZE * head_dim`)
//! * `kv_head_stride`  — elements between kv_head and kv_head + 1
//!   within one block (= `BLOCK_SIZE * head_dim`)
//! * `per_token_stride`— elements between slot_in_block S and S + 1
//!   (= `head_dim`)
//! * `buffer_elems`    — total elements per K (or V) buffer
//!   (= `num_blocks * kv_blk_stride`)
//!
//! Catches bug class #5 — the four-call-site stride-math drift the
//! plan calls out (`cpu_golden::rope_append_paged`,
//! `cpu_golden::attention_via_cache`, `cpu_golden::attention_prefill_paged`,
//! `interpreter::metal::pipelines` test harness).
//!
//! Lives at the crate root (not feature-gated) so both `cpu_golden`
//! (unconditional) and the metal-feature-gated callers can share one
//! definition. Args are bare `u32` to avoid pulling the metal-side
//! `ids` newtypes into CPU paths; metal call sites can wrap their
//! `PhysicalBlockIdx(_)` / `SlotInBlock(_)` payloads at the boundary.

/// Per-layer paged KV-cache layout.
///
/// The layer index is *not* part of the layout — each layer gets its
/// own `K` and `V` buffer (`RuntimeBindingKind::KvCacheK { layer }`).
/// The layout describes the within-buffer addressing only.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PagedKvLayout {
    pub num_blocks: u32,
    pub num_kv_heads: u32,
    pub block_size: u32,
    pub head_dim: u32,
}

impl PagedKvLayout {
    /// Derive a layout from the flat buffer element count plus the
    /// three within-block dims (`num_kv_heads`, `block_size`,
    /// `head_dim`). The CPU golden code paths know those three
    /// up-front and infer `num_blocks` from the K (or V) buffer
    /// length — this constructor folds the stride math the call
    /// site used to inline. Asserts that the inputs divide evenly.
    pub fn from_buffer_elems(
        buffer_elems: usize,
        num_kv_heads: u32,
        block_size: u32,
        head_dim: u32,
    ) -> Self {
        let blk_stride = (num_kv_heads as usize) * (block_size as usize) * (head_dim as usize);
        assert!(blk_stride > 0, "PagedKvLayout: zero-size block stride");
        assert_eq!(
            buffer_elems % blk_stride,
            0,
            "PagedKvLayout: buffer length {buffer_elems} not divisible by per-block stride {blk_stride}",
        );
        Self {
            num_blocks: (buffer_elems / blk_stride) as u32,
            num_kv_heads,
            block_size,
            head_dim,
        }
    }

    /// Elements per block: `num_kv_heads * block_size * head_dim`.
    /// Equivalently, the stride along the leading "block_id" axis.
    pub fn kv_blk_stride(&self) -> usize {
        (self.num_kv_heads as usize) * (self.block_size as usize) * (self.head_dim as usize)
    }

    /// Elements per kv-head within one block: `block_size * head_dim`.
    pub fn kv_head_stride(&self) -> usize {
        (self.block_size as usize) * (self.head_dim as usize)
    }

    /// Elements per slot within one (block, kv-head) pair: `head_dim`.
    pub fn per_token_stride(&self) -> usize {
        self.head_dim as usize
    }

    /// Total elements in a single K-or-V cache buffer for this layout.
    pub fn buffer_elems(&self) -> usize {
        (self.num_blocks as usize) * self.kv_blk_stride()
    }

    /// Element offset of
    /// `cache[physical_block, kv_head, slot_in_block, dim=0]`. Add
    /// the per-dim index (`0..head_dim`) to address one element.
    pub fn elem_offset(&self, physical_block: u32, kv_head: u32, slot_in_block: u32) -> usize {
        (physical_block as usize) * self.kv_blk_stride()
            + (kv_head as usize) * self.kv_head_stride()
            + (slot_in_block as usize) * self.per_token_stride()
    }

    /// Convenience: element offset given a flat `global_slot` index
    /// that the engine writes into `RuntimeBindingKind::SlotMapping`.
    /// Same arithmetic as `cpu_golden::rope_append_paged` did inline.
    pub fn elem_offset_for_global_slot(&self, global_slot: u32, kv_head: u32) -> usize {
        let block = global_slot / self.block_size;
        let slot = global_slot % self.block_size;
        self.elem_offset(block, kv_head, slot)
    }

    /// Decompose a physical block id into `(chunk_index, block_in_chunk)`
    /// for the metal chunked KV pool.
    ///
    /// On metal the KV pool is allocated as a series of fixed-size
    /// *chunk* buffers (`blocks_per_chunk` blocks each, one MTLBuffer
    /// per chunk per layer per K/V) so the `MTLResidencySet` can hold
    /// only the live chunks — physical RAM then tracks live KV instead
    /// of pinning the whole pool resident. A physical block id maps to
    /// the chunk that backs it (`id / blocks_per_chunk`) and the block
    /// offset within that chunk (`id % blocks_per_chunk`); the kernel
    /// looks the chunk's base `gpuAddress` up in a per-layer table and
    /// then addresses within it with the *block-in-chunk* index in
    /// place of `physical_block`.
    ///
    /// CUDA (and any single-buffer caller) passes
    /// `blocks_per_chunk == num_blocks`, so `chunk_index` is always 0
    /// and `block_in_chunk == physical_block` — i.e. a no-op.
    ///
    /// `blocks_per_chunk` must be non-zero.
    #[inline]
    pub fn chunk_decompose(physical_block: u32, blocks_per_chunk: u32) -> (u32, u32) {
        debug_assert!(blocks_per_chunk > 0, "blocks_per_chunk must be non-zero");
        (
            physical_block / blocks_per_chunk,
            physical_block % blocks_per_chunk,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_decompose_single_buffer_is_noop() {
        // blocks_per_chunk == num_blocks: everything lands in chunk 0
        // at its own index (the CUDA / single-buffer contract).
        for blk in [0u32, 1, 17, 255] {
            assert_eq!(PagedKvLayout::chunk_decompose(blk, 256), (0, blk));
        }
    }

    #[test]
    fn chunk_decompose_crosses_chunk_boundaries() {
        let bpc = 512; // one chunk == one 8192-token seq at block_size 16
        assert_eq!(PagedKvLayout::chunk_decompose(0, bpc), (0, 0));
        assert_eq!(PagedKvLayout::chunk_decompose(511, bpc), (0, 511));
        assert_eq!(PagedKvLayout::chunk_decompose(512, bpc), (1, 0));
        assert_eq!(PagedKvLayout::chunk_decompose(513, bpc), (1, 1));
        assert_eq!(PagedKvLayout::chunk_decompose(1024, bpc), (2, 0));
    }

    #[test]
    fn chunk_decompose_reassembles_to_global_block() {
        let bpc = 300;
        for blk in [0u32, 1, 299, 300, 301, 899, 900, 12_345] {
            let (chunk, in_chunk) = PagedKvLayout::chunk_decompose(blk, bpc);
            assert_eq!(chunk * bpc + in_chunk, blk);
            assert!(in_chunk < bpc);
        }
    }
}
