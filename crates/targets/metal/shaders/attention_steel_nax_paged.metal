// SPDX-License-Identifier: Apache-2.0
//
// NAX (matrix-accelerator) paged attention instantiations.
//
// Like `quantized_qmm_nax.metal`, this kernel uses MetalPerformancePrimitives
// `matmul2d` cooperative tensors, so build.rs compiles it with the MPP flags
// (see `reaches_mpp`).
//
// Tile shape: BQ=64, BK=32, WM=4, WN=1, BLOCK_SIZE=16. BQ=64 = 4 warps *
// 16 rows (one NAX Q-frag per warp). BK=32 = 2 paged blocks per K-tile.

#include "baked.h"
#include "metal_nax.h"
#include "mlx_steel_attn/steel_attention_nax_paged_kernel.h"

#define INST_STEEL_NAX_PAGED(dt_tag, dt_type, bd) \
  SCRATCHY_KERNEL(attention_steel_nax_paged_##dt_tag##_bq64_bk32_bd##bd##_wm4_wn1_bs16, attention_nax_paged<dt_type, 64, 32, bd, 4, 1, 16, float>)
// The whole-page `matmul2d` kernel, four Q-row blocks a threadgroup: two warps a block, each half
// the head dims of O, over 16-token pages (a step one K-tile) ...
#define INST_STEEL_NAX_PAGED_WN2(dt_tag, dt_type, bd) \
  SCRATCHY_KERNEL(attention_steel_nax_paged_##dt_tag##_bq64_bk32_bd##bd##_wm4_wn2_bs16, attention_nax_paged_wide<dt_type, 64, 32, bd, 4, 2, 16, float>)
// ... or four, each a quarter, over 32-token pages (a step four K-tiles).
#define INST_STEEL_NAX_PAGED_WN4_BS32(dt_tag, dt_type, bd) \
  SCRATCHY_KERNEL(attention_steel_nax_paged_##dt_tag##_bq64_bk32_bd##bd##_wm4_wn4_bs32, attention_nax_paged_wide<dt_type, 64, 32, bd, 4, 4, 32, float>)

INST_STEEL_NAX_PAGED(f16, half, 128)
INST_STEEL_NAX_PAGED(bf16, bfloat, 128)
// head_dim 64 (llama-class): same BQ=64/BK=32 tiling, BD=64. Lets hd64
// attention run on NAX (matmul2d) with the block-diagonal span cut, instead of
// the simdgroup steel kernel.
INST_STEEL_NAX_PAGED(f16, half, 64)
INST_STEEL_NAX_PAGED(bf16, bfloat, 64)
// head_dim 256 (Qwen3.6's full attention, Gemma 4's sliding layers): same tiling, twice the
// head-dim fragments a warp.
INST_STEEL_NAX_PAGED_WN2(f16, half, 256)
INST_STEEL_NAX_PAGED_WN2(bf16, bfloat, 256)
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
// head_dim 512 over 32-token pages (Gemma 4's global layers).
INST_STEEL_NAX_PAGED_WN4_BS32(f16, half, 512)
INST_STEEL_NAX_PAGED_WN4_BS32(bf16, bfloat, 512)

// Rope-once kernel (spans rope-on-read): ropes a request's K ONCE into the
// dense scratch, so the attention reads pre-roped K with no per-tile rotation.
// Plain compute kernel (no MPP) but lives in this library to share the cache
// resolve + cos_sin layout. One instantiation per (dtype, head_dim).
#define INST_ROPE_ONCE_NAX(dt_tag, dt_type, bd) \
  SCRATCHY_KERNEL(rope_once_nax_##dt_tag##_bd##bd##_bs16, rope_once_nax_kernel<dt_type, bd, 16>)
#define INST_ROPE_ONCE_NAX_BS32(dt_tag, dt_type, bd) \
  SCRATCHY_KERNEL(rope_once_nax_##dt_tag##_bd##bd##_bs32, rope_once_nax_kernel<dt_type, bd, 32>)

INST_ROPE_ONCE_NAX(f16, half, 128)
INST_ROPE_ONCE_NAX(bf16, bfloat, 128)
INST_ROPE_ONCE_NAX(f16, half, 64)
INST_ROPE_ONCE_NAX(bf16, bfloat, 64)
INST_ROPE_ONCE_NAX(f16, half, 256)
INST_ROPE_ONCE_NAX(bf16, bfloat, 256)
INST_ROPE_ONCE_NAX_BS32(f16, half, 512)
INST_ROPE_ONCE_NAX_BS32(bf16, bfloat, 512)
