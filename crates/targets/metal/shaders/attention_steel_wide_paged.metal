// SPDX-License-Identifier: Apache-2.0
//
// Simdgroup (pre-NAX) WIDE-head paged attention instantiations — the steel
// port of `attention_steel_nax_paged.metal`'s `attention_nax_paged_wide`
// rows, for head_dims one warp's whole-head O cannot hold on GPUs without
// the matrix accelerator (M1-M4: Gemma 4's head_dim 512 global layers).
//
// Includes the classic `steel_attention_paged_kernel.h` for its
// ATTN_PAGED_* constants, its reduce ops and its `rope_once_steel_kernel`
// (instantiated below at the wide kernel's page size); the wide kernel
// itself is `mlx_steel_attn/steel_attention_wide_paged_kernel.h`.
// Every kernel here takes its tile from the bake's constants (the lowering's `SteelWideTile`).

#include "baked.h"
#include "mlx_steel_attn/steel_attention_paged_kernel.h"
#include "mlx_steel_attn/steel_attention_wide_paged_kernel.h"

// The tile a bake runs at (`SteelWideTile`, the lowering's law): Q rows a threadgroup, keys a
// K-tile, 16-row Q blocks a threadgroup and head-dim slices a block. The head dim and the page are
// the bake's ATTN_PAGED_HEAD_DIM / ATTN_PAGED_BLOCK_SIZE.
SCRATCHY_CONSTANT_OPTIONAL(uint, STEEL_TILE_BQ, 30);
SCRATCHY_CONSTANT_OPTIONAL(uint, STEEL_TILE_BK, 31);
SCRATCHY_CONSTANT_OPTIONAL(uint, STEEL_TILE_WM, 32);
SCRATCHY_CONSTANT_OPTIONAL(uint, STEEL_TILE_WN, 33);

#define INST_STEEL_WIDE_PAGED(dt_tag, dt_type) \
  SCRATCHY_KERNEL(attention_steel_wide_paged_##dt_tag, attention_steel_paged_wide<dt_type, STEEL_TILE_BQ, STEEL_TILE_BK, ATTN_PAGED_HEAD_DIM, STEEL_TILE_WM, STEEL_TILE_WN, ATTN_PAGED_BLOCK_SIZE, float>) \
  SCRATCHY_KERNEL(rope_once_steel_wide_##dt_tag, rope_once_steel_kernel<dt_type, ATTN_PAGED_HEAD_DIM, ATTN_PAGED_BLOCK_SIZE>)

INST_STEEL_WIDE_PAGED(f16, half)
INST_STEEL_WIDE_PAGED(bf16, bfloat)
