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

// One warp's whole slice of the head dims (`attention_nax_paged`), or several warps a Q-row block,
// each a slice, over whole-page `matmul2d`s (`attention_nax_paged_wide`).
#define INST_NAX_PAGED(dt_tag, dt_type) \
  SCRATCHY_KERNEL(attention_nax_paged_##dt_tag, attention_nax_paged<dt_type, NAXP_TILE_BQ, NAXP_TILE_BK, NAXP_HEAD_DIM, NAXP_TILE_WM, NAXP_TILE_WN, NAXP_BLOCK_SIZE, float>) \
  SCRATCHY_KERNEL(attention_nax_paged_wide_##dt_tag, attention_nax_paged_wide<dt_type, NAXP_TILE_BQ, NAXP_TILE_BK, NAXP_HEAD_DIM, NAXP_TILE_WM, NAXP_TILE_WN, NAXP_BLOCK_SIZE, float>) \
  SCRATCHY_KERNEL(rope_once_nax_##dt_tag, rope_once_nax_kernel<dt_type, NAXP_HEAD_DIM, NAXP_BLOCK_SIZE>)

INST_NAX_PAGED(f16, half)
INST_NAX_PAGED(bf16, bfloat)

// A verify step's lanes at head_dim 256: `attention_verify_pack_q` packs a KV head's group of query
// heads' 8-row lanes into one 64-row tile, so each KV page is read once a group.
#define INST_VERIFY_PACK(dt_tag, dt_type, bd) \
  SCRATCHY_KERNEL(attention_verify_pack_q_##dt_tag##_bd##bd, attention_verify_pack_q<dt_type, bd>)
INST_VERIFY_PACK(f16, half, 256)
INST_VERIFY_PACK(bf16, bfloat, 256)
// The packed tile with its keys split VERIFY_SPLITS ways (`attention_verify_split`), its partials
// merged by `attention_via_cache_v2_combine`.
#define INST_VERIFY_SPLIT(dt_tag, dt_type, bd, sp) \
  SCRATCHY_KERNEL(attention_steel_nax_paged_##dt_tag##_bq64_bk32_bd##bd##_wm4_wn2_bs16_packed_split##sp, attention_nax_paged_wide<dt_type, 64, 32, bd, 4, 2, 16, float, true, sp>)
INST_VERIFY_SPLIT(f16, half, 256, 4)
INST_VERIFY_SPLIT(bf16, bfloat, 256, 4)
