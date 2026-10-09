// SPDX-License-Identifier: Apache-2.0

#pragma once

// Paged-cache base-pointer resolution, shared by every paged attention
// kernel that loads K/V fragments straight from device memory (the NAX
// `matmul2d` kernels and the simdgroup wide kernel) instead of staging
// tiles through threadgroup memory (`PagedBlockLoaderT` keeps its own
// copy inside the loader). Extracted from
// `steel_attention_nax_paged_kernel.h` when the simdgroup wide kernel
// needed the same two resolves.

#include <metal_stdlib>

using namespace metal;

namespace mlx {
namespace steel {

// Resolve the device base pointer for paged block `logical_block` of the
// current sequence: indirect through the per-seq block table, then through
// the per-layer chunk-address table, then add the kv-head offset.
//   chunk_table[ physical / blocks_per_chunk ]
//     + (physical % blocks_per_chunk) * kv_blk_stride
//     + kv_head_off
// Mirrors `PagedBlockLoaderT::resolve` (paged_loader.h).
//
// Spans: block_table bit 31 carries the rope-on-read unrotated flag; mask
// it off for addressing (identity for non-spans — the worker only ever
// sets bit 31 when ROPE_ON_READ, so this is a free ALU op there). This
// mirrors `paged_loader.h`'s `& 0x7FFFFFFFu` in all three resolve paths.
template <typename T>
METAL_FUNC const device T* paged_resolve_block(
    const device uint64_t* chunk_table,
    const device uint* block_table_row,
    int logical_block,
    int num_pages,
    int blocks_per_chunk,
    int kv_blk_stride,
    int kv_head_off) {
  int lb = logical_block;
  if (lb >= num_pages) {
    lb = num_pages - 1;
  }
  const uint physical = uint(block_table_row[lb]) & 0x7FFFFFFFu;
  const uint chunk = physical / uint(blocks_per_chunk);
  const uint bic = physical % uint(blocks_per_chunk);
  return (const device T*)chunk_table[chunk] + int(bic) * kv_blk_stride + kv_head_off;
}

// Resolve the base pointer for LOGICAL block `logical_block` of the roped-K
// SCRATCH (rope-on-read spans). The scratch is a single dense contiguous
// buffer written by the rope-once kernels, indexed by LOGICAL block (no
// block_table indirection, no chunk table) with the SAME per-block strides
// as the cache:
//   scratch + logical_block * kv_blk_stride + kv_head_off
// so attention can reuse the cache's load math verbatim.
template <typename T>
METAL_FUNC const device T* paged_resolve_scratch(
    const device T* scratch,
    int logical_block,
    int num_pages,
    int kv_blk_stride,
    int kv_head_off) {
  int lb = logical_block;
  if (lb >= num_pages) {
    lb = num_pages - 1;
  }
  return scratch + lb * kv_blk_stride + kv_head_off;
}

} // namespace steel
} // namespace mlx
