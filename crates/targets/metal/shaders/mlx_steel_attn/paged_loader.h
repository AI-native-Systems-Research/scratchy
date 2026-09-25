// SPDX-License-Identifier: Apache-2.0
//
// Paged-K/V-cache adapter for MLX `steel_attention` kernel.
// Mirrors the interface of MLX's `BlockLoaderT` (loader.h:153) —
// constructor + `load_unsafe()` + `load_safe()` + `next()` — but
// reads tiles from the scratchy-target-metal paged KV cache instead of a
// contiguous device buffer.
//
// Cache layout (matches `attention_via_cache_v2`/`rope_append`):
//   k_cache[num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]
//   v_cache[num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]
//
// Per-sequence indirection through `block_table[seq_idx][logical_block]`
// → physical_block. Caller is responsible for computing the absolute
// K-position via `seq_used_k[seq_idx]` and the per-Q-block start.
//
// IMPORTANT: this loader currently assumes BK == BLOCK_SIZE, so one
// MLX kb iteration corresponds to exactly one paged block (no
// intra-tile block boundary). For Llama-3.2-3B (head_dim=128 →
// BK=16, BLOCK_SIZE=16) the assumption holds. For head_dim < 128
// (BK=32) we'd need to handle straddled tiles — a follow-up if/when
// we model archs with smaller head_dim on metal.

#pragma once

#include "defines.h"
#include "loader.h"  // for BlockLoaderT

namespace mlx {
namespace steel {

// Minimal-delta paged loader: mirrors `BlockLoaderT` (loader.h) byte-
// for-byte — same struct layout, same `load_unsafe`/`load_safe`
// methods, same per-thread `src` member that load* reads from. The
// ONLY difference vs `BlockLoaderT` is `next()`: instead of advancing
// `src` linearly by `tile_stride`, we re-resolve it through
// `block_table[logical_block]` so non-sequential paged blocks work.
//
// Why not inherit from `BlockLoaderT` and override only `next()`?
// Tried — Metal's template-dependent-base name lookup refused to find
// inherited members ("'BlockLoaderT::src' is not a member of class
// 'PagedBlockLoaderT'") even with `this->Base::src`. Standalone class
// sidesteps that.
//
// Why this should fix the MMA-frag-accumulation bug we found in
// `PagedKVBlockLoader`: that loader recomputes `src` *inside*
// `load_unsafe()` as a function-local variable; this loader keeps
// `src` as a struct member that `load_unsafe()` reads from — exactly
// like `BlockLoaderT`, which the contig kernel uses successfully.
// Same compiler input → same codegen path → no compiler-induced
// frag-dropping in the downstream MMA loop.
//
// Constraint: `BROWS == BLOCK_SIZE_` so one kb iter = one paged block.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short kDstStrRow,
    short kDstStrCol,
    short tgp_size,
    short BLOCK_SIZE_,
    short n_reads = (BCOLS * BROWS) / tgp_size,
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct PagedBlockLoaderT {
  // A K-tile spans `kPagesPerTile` consecutive paged blocks of BLOCK_SIZE
  // rows. For the steel-attention shapes each thread owns exactly one tile
  // row (TROWS == BROWS), so a thread's row never straddles a page boundary
  // — load_unsafe/load_safe stay byte-identical to BlockLoaderT; only
  // resolve()/next()/seek() become page-aware.
  STEEL_CONST short kPagesPerTile = BROWS / BLOCK_SIZE_;
  static_assert(
      BROWS % BLOCK_SIZE_ == 0,
      "PagedBlockLoaderT requires BK to be a whole multiple of BLOCK_SIZE.");
  static_assert(
      BROWS == BLOCK_SIZE_ || TROWS >= BROWS,
      "multi-page K-tiles require one tile row per thread (TROWS >= BROWS).");

  STEEL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  STEEL_CONST short vec_size = n_reads;

  // Layout — these mirror `BlockLoaderT` exactly (same names + types
  // + order) so load_unsafe/load_safe below are textually identical.
  const int src_ld;
  const int tile_stride;  // unused for paged (next() uses block_table) but kept for layout parity
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const device T* src;  // re-bound in next() via block_table

  // Block-indirection extras. Chunked KV: `chunk_table` is the
  // per-layer chunk-address table (device uint64 gpuAddresses), not a
  // single cache buffer. A physical block id derefs
  // `chunk_table[physical / blocks_per_chunk]` then addresses
  // `physical % blocks_per_chunk` within that chunk. `kv_head_off`
  // (= kv_head_idx * kv_head_stride) is added per-block since the
  // chunk base is resolved fresh on every block.
  const device uint* block_table_row;
  const device uint64_t* chunk_table;
  const int kv_blk_stride;
  const int kv_head_off;
  const int blocks_per_chunk;
  const int num_pages;          // valid logical blocks for this seq (clamp bound)
  const short row_page;         // bi / BLOCK_SIZE_ — sub-page this thread's row reads
  const int row_col_offset;     // (bi % BLOCK_SIZE_) * src_ld + bj + kv_head_off
  int logical_block;            // first page of the current K-tile

  // Rope-on-read (spans, ROPE-ONCE-TO-SCRATCH): when `scratch_base != nullptr`
  // this loader reads K from a DENSE pre-roped scratch buffer (written once by
  // `rope_once_steel`) indexed by LOGICAL block — no block_table / chunk_table
  // indirection. The scratch mirrors the cache's per-block strides, so
  // `scratch_base + lb*kv_blk_stride + row_col_offset` reuses the exact same
  // per-thread offset math. The `scratch_base` is only ever non-null when the
  // kernel sets it from `ATTN_PAGED_ROR != 0` (a compile-time function-constant
  // fold), so the non-spans path keeps the byte-identical chunk_table resolve.
  const device T* scratch_base;

  // Resolve this thread's `src` for the K-tile whose first page is
  // `first_lb`. The thread's row lives in sub-page `row_page`; clamp to the
  // last allocated page so a partial tail tile never derefs an unallocated
  // block-table slot (those out-of-range rows are zeroed by load_safe).
  METAL_FUNC const device T* resolve(int first_lb) const thread {
    int lb = first_lb + int(row_page);
    if (lb >= num_pages) {
      lb = num_pages - 1;
    }
    if (scratch_base != nullptr) {
      // Dense pre-roped scratch: logical-block indexed, no indirection.
      return scratch_base + lb * kv_blk_stride + row_col_offset;
    }
    // Spans: mask off block_table bit 31 (rope-on-read flag); identity
    // for non-spans (worker only sets bit 31 when ROPE_ON_READ).
    const uint physical = uint(block_table_row[lb]) & 0x7FFFFFFFu;
    const uint chunk    = physical / uint(blocks_per_chunk);
    const uint bic      = physical % uint(blocks_per_chunk);
    return (const device T*)chunk_table[chunk]
        + int(bic) * kv_blk_stride + row_col_offset;
  }

  METAL_FUNC PagedBlockLoaderT(
      const device uint64_t* chunk_table_,
      const int src_ld_,
      const int kv_blk_stride_,
      const int kv_head_off_,
      const int blocks_per_chunk_,
      const int num_pages_,
      const device uint* block_table_row_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      const device T* scratch_base_ = nullptr) thread
      : src_ld(src_ld_),
        tile_stride(BROWS * src_ld_),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * kDstStrRow + bj * kDstStrCol),
        src(nullptr),
        block_table_row(block_table_row_),
        chunk_table(chunk_table_),
        kv_blk_stride(kv_blk_stride_),
        kv_head_off(kv_head_off_),
        blocks_per_chunk(blocks_per_chunk_),
        num_pages(num_pages_),
        row_page(bi / BLOCK_SIZE_),
        row_col_offset(int(bi % BLOCK_SIZE_) * src_ld_ + int(bj) + kv_head_off_),
        logical_block(0),
        scratch_base(scratch_base_) {
    src = resolve(0);
  }

  // ===== Methods copied verbatim from `BlockLoaderT` =====================

  /* Load from device memory into threadgroup memory — without bound checking */
  METAL_FUNC void load_unsafe() const thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = src[i * src_ld + j];
      }
    }
  }

  /* Load from device memory into threadgroup memory — with bound checking */
  METAL_FUNC void load_safe(short2 src_tile_dim) const thread {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * kDstStrRow + j * kDstStrCol] = T(0);
        }
      }
      return;
    }

    bool tmp_idx[vec_size];
    T tmp_val[vec_size];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      }
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      }
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      }
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = tmp_val[j];
      }
    }
  }

  // ===== End verbatim copy ==============================================

  /* Iteration helper — advance to the next K-tile (kPagesPerTile pages). */
  METAL_FUNC void next() thread {
    logical_block += kPagesPerTile;
    src = resolve(logical_block);
  }

  /* Jump to BK-tile index `tile` (converted to the page-unit first block).
   * Used by the sliding-window kernel to skip K/V tiles entirely older than
   * the attention window. The kernel speaks in BK-tile units; the loader owns
   * the page-unit conversion so BK=BLOCK_SIZE (1 page/tile) is unchanged. */
  METAL_FUNC void seek(int tile) thread {
    logical_block = tile * kPagesPerTile;
    src = resolve(logical_block);
  }
};

template <
    typename T,
    short BROWS,             // BK (K-seq tile rows)
    short BCOLS,             // BD (head dim)
    short kDstStrRow,        // threadgroup-dest row stride
    short kDstStrCol,        // threadgroup-dest col stride
    short tgp_size,          // WM*WN*32
    short BLOCK_SIZE_,       // paged-cache block size (assumed == BROWS)
    short n_reads = (BCOLS * BROWS) / tgp_size,
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct PagedKVBlockLoader {
  static_assert(
      BROWS == BLOCK_SIZE_,
      "PagedKVBlockLoader currently requires BROWS == BLOCK_SIZE; "
      "intra-tile block boundary not supported yet.");

  STEEL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  STEEL_CONST short vec_size = n_reads;
  STEEL_CONST short BLOCK_SIZE = BLOCK_SIZE_;

  // Cache base + per-block stride
  const device T* cache_base;          // points at k_cache or v_cache
  const int kv_blk_stride;             // num_kv_heads * BLOCK_SIZE * head_dim
  const int kv_head_stride;            // BLOCK_SIZE * head_dim
  const int per_token_stride;          // head_dim (cols stride within a block)
  // Per-sequence indirection
  const device uint* block_table_row;  // = block_table + seq_idx * max_blocks
  // Current logical block index in this sequence
  int logical_block;
  // Threadgroup destination + thread coords
  threadgroup T* dst;
  const short thread_idx;
  const short bi;  // row within the BK tile
  const short bj;  // starting col within BD

  METAL_FUNC PagedKVBlockLoader(
      const device T* cache_base_,
      const int kv_blk_stride_,
      const int kv_head_stride_,
      const int per_token_stride_,
      const device uint* block_table_row_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) thread
      : cache_base(cache_base_),
        kv_blk_stride(kv_blk_stride_),
        kv_head_stride(kv_head_stride_),
        per_token_stride(per_token_stride_),
        block_table_row(block_table_row_),
        logical_block(0),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * kDstStrRow + bj * kDstStrCol) {}

  /* Load this thread's slice of the current paged block. */
  METAL_FUNC void load_unsafe() const thread {
    // Spans: block_table bit 31 carries the rope-on-read unrotated flag;
    // mask it off for addressing. Identity for non-spans (the worker only
    // ever sets bit 31 when ROPE_ON_READ), so this is a free ALU op there.
    const uint physical_block = block_table_row[logical_block] & 0x7FFFFFFFu;
    const device T* src =
        cache_base + physical_block * kv_blk_stride + bi * per_token_stride + bj;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = src[i * per_token_stride + j];
      }
    }
  }

  /* Bounded variant: zero out positions past `src_tile_dim`.
   * For paged cache the bounds-checked path fires on the last
   * partial K block (kL_rem < BROWS). */
  METAL_FUNC void load_safe(short2 src_tile_dim) const thread {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * kDstStrRow + j * kDstStrCol] = T(0);
        }
      }
      return;
    }

    // Spans: block_table bit 31 carries the rope-on-read unrotated flag;
    // mask it off for addressing. Identity for non-spans (the worker only
    // ever sets bit 31 when ROPE_ON_READ), so this is a free ALU op there.
    const uint physical_block = block_table_row[logical_block] & 0x7FFFFFFFu;
    const device T* src =
        cache_base + physical_block * kv_blk_stride + bi * per_token_stride + bj;

    bool tmp_idx[vec_size];
    T tmp_val[vec_size];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      }
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = src[tmp_idx[j] ? (i * per_token_stride + j) : 0];
      }
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      }
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = tmp_val[j];
      }
    }
  }

  METAL_FUNC void next() thread {
    logical_block += 1;
  }
};

} // namespace steel
} // namespace mlx
