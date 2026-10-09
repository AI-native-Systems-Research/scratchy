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
//
// Tile shape for head_dim 512 (see the measured-optimum note at the
// instantiations below): BQ=32, BK=32, WM=2, WN=8, BLOCK_SIZE=32 — two
// 16-row Q blocks a threadgroup, eight head-dim slices (DW=64, a
// 32-register O a warp), a step one page group a warp (eight 32-key
// K-tiles), 512 threads.

#include "baked.h"
#include "mlx_steel_attn/steel_attention_paged_kernel.h"
#include "mlx_steel_attn/steel_attention_wide_paged_kernel.h"

#define INST_STEEL_WIDE_PAGED(dt_tag, dt_type, bq, bk, bd, wm, wn, bs) \
  SCRATCHY_KERNEL(attention_steel_wide_paged_##dt_tag##_bq##bq##_bk##bk##_bd##bd##_wm##wm##_wn##wn##_bs##bs, attention_steel_paged_wide<dt_type, bq, bk, bd, wm, wn, bs, float>)

// Rope-once twin at the wide kernel's page size (the classic library's
// INST_STEEL_PAGED macro hardcodes bs16; the wide kernel runs on 32-token
// pages). Same template, same symbol scheme as
// `rope_once_steel_symbol`'s bs16 rows.
#define INST_ROPE_ONCE_STEEL_BS32(dt_tag, dt_type, bd) \
  SCRATCHY_KERNEL(rope_once_steel_##dt_tag##_bd##bd##_bs32, rope_once_steel_kernel<dt_type, bd, 32>)

// Tile shape (measured on M1 Max, m=4096 8q/2kv): BQ=32, BK=32, WM=2,
// WN=8, BLOCK_SIZE=32 — 512 threads, 32 KB exchange (one buffer). The
// head-dim slice count is the perf dial: every doubling of WN halves a
// warp's V/Q fragment-load instructions per key (loads, not MMA, bound
// this kernel — hoisting the V fragment out of the Q-row-frag loop alone
// was +41%) and doubles the WN-times-duplicated softmax. wn8 is the
// measured optimum of that trade (wn4 1.30, wn8 1.64, wn16 1.54 TF/s).
INST_STEEL_WIDE_PAGED(f16, half, 32, 32, 512, 2, 8, 32)
INST_STEEL_WIDE_PAGED(bf16, bfloat, 32, 32, 512, 2, 8, 32)
INST_ROPE_ONCE_STEEL_BS32(f16, half, 512)
INST_ROPE_ONCE_STEEL_BS32(bf16, bfloat, 512)
