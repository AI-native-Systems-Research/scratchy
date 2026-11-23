// SPDX-License-Identifier: Apache-2.0
//
// NAX (matrix-accelerator) paged attention instantiations.
//
// Like `quantized_qmm_nax.metal`, this kernel uses MetalPerformancePrimitives
// `matmul2d` cooperative tensors, so build.rs compiles it with the MPP flags
// (see `reaches_mpp`). Every kernel here takes its tile from the bake's constants.

#include "baked.h"
#include "metal_nax.h"
#include "mlx_steel_attn/steel_attention_nax_paged_kernel.h"

// The tile a bake runs at (`NaxTile`, the lowering's law): Q rows a threadgroup, keys a step,
// warps along the Q rows and along the head dims. The head dim and the page are the bake's
// NAXP_HEAD_DIM / NAXP_BLOCK_SIZE.
SCRATCHY_CONSTANT_OPTIONAL(uint, NAXP_TILE_BQ, 30);
SCRATCHY_CONSTANT_OPTIONAL(uint, NAXP_TILE_BK, 31);
SCRATCHY_CONSTANT_OPTIONAL(uint, NAXP_TILE_WM, 32);
SCRATCHY_CONSTANT_OPTIONAL(uint, NAXP_TILE_WN, 33);
// The runs a verify step's packed tile splits its keys into (`attention_nax_paged_verify_split`).
SCRATCHY_CONSTANT_OPTIONAL(uint, NAXP_SPLITS, 34);

// One warp's whole slice of the head dims (`attention_nax_paged`), or several warps a Q-row block,
// each a slice, over whole-page `matmul2d`s (`attention_nax_paged_wide`).
#define INST_NAX_PAGED(dt_tag, dt_type) \
  SCRATCHY_KERNEL(attention_nax_paged_##dt_tag, attention_nax_paged<dt_type, NAXP_TILE_BQ, NAXP_TILE_BK, NAXP_HEAD_DIM, NAXP_TILE_WM, NAXP_TILE_WN, NAXP_BLOCK_SIZE, float>) \
  SCRATCHY_KERNEL(attention_nax_paged_wide_##dt_tag, attention_nax_paged_wide<dt_type, NAXP_TILE_BQ, NAXP_TILE_BK, NAXP_HEAD_DIM, NAXP_TILE_WM, NAXP_TILE_WN, NAXP_BLOCK_SIZE, float>) \
  SCRATCHY_KERNEL(rope_once_nax_##dt_tag, rope_once_nax_kernel<dt_type, NAXP_HEAD_DIM, NAXP_BLOCK_SIZE>)

INST_NAX_PAGED(f16, half)
INST_NAX_PAGED(bf16, bfloat)

// A verify step's lanes: `attention_verify_pack_q` packs a KV head's group of query heads' 8-row
// lanes into one Q tile, so each KV page is read once a group, and the wide kernel's packed form
// splits its keys NAXP_SPLITS ways, its partials merged by `attention_via_cache_v2_combine`.
#define INST_VERIFY(dt_tag, dt_type) \
  SCRATCHY_KERNEL(attention_verify_pack_q_##dt_tag, attention_verify_pack_q<dt_type, NAXP_HEAD_DIM, NAXP_TILE_BQ>) \
  SCRATCHY_KERNEL(attention_nax_paged_verify_split_##dt_tag, attention_nax_paged_wide<dt_type, NAXP_TILE_BQ, NAXP_TILE_BK, NAXP_HEAD_DIM, NAXP_TILE_WM, NAXP_TILE_WN, NAXP_BLOCK_SIZE, float, true, NAXP_SPLITS>)
INST_VERIFY(f16, half)
INST_VERIFY(bf16, bfloat)
