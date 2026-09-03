// SPDX-License-Identifier: Apache-2.0
//
// NAX (matrix-accelerator) paged attention instantiations.
//
// IMPORTANT: like `quantized_qmm_nax.metal`, this kernel uses
// MetalPerformancePrimitives `matmul2d` cooperative tensors and MUST be
// compiled at runtime via `newLibraryWithSource` — the offline
// `xcrun metal` / `metallib` toolchain miscompiles MPP (each matmul2d
// reduces only half its K). It is therefore NOT registered as an
// embedded metallib in `build.rs`; the test / dispatcher compiles it
// from source (`metal_nax.h` inlined) the same way
// `compile_nax_library_from_source` does for the qmm.
//
// Tile shape: BQ=64, BK=32, WM=4, WN=1, BLOCK_SIZE=16. BQ=64 = 4 warps *
// 16 rows (one NAX Q-frag per warp). BK=32 = 2 paged blocks per K-tile.

#include "metal_nax.h"
#include "mlx_steel_attn/steel_attention_nax_paged_kernel.h"

#define INST_STEEL_NAX_PAGED(dt_tag, dt_type, bd)                              \
  template [[host_name(                                                        \
      "attention_steel_nax_paged_" #dt_tag "_bq64_bk32_bd" #bd "_wm4_wn1_bs16" \
  )]] [[kernel]]                                                               \
  decltype(attention_nax_paged<dt_type, 64, 32, bd, 4, 1, 16, float>)          \
      attention_nax_paged<dt_type, 64, 32, bd, 4, 1, 16, float>;

INST_STEEL_NAX_PAGED(f16, half, 128)
INST_STEEL_NAX_PAGED(bf16, bfloat, 128)
// head_dim 64 (llama-class): same BQ=64/BK=32 tiling, BD=64. Lets hd64
// attention run on NAX (matmul2d) with the block-diagonal span cut, instead of
// the simdgroup steel kernel.
INST_STEEL_NAX_PAGED(f16, half, 64)
INST_STEEL_NAX_PAGED(bf16, bfloat, 64)

// Rope-once kernel (spans rope-on-read): ropes a request's K ONCE into the
// dense scratch, so the attention reads pre-roped K with no per-tile rotation.
// Plain compute kernel (no MPP) but lives in this library to share the cache
// resolve + cos_sin layout. One instantiation per (dtype, head_dim).
#define INST_ROPE_ONCE_NAX(dt_tag, dt_type, bd)                                \
  template [[host_name("rope_once_nax_" #dt_tag "_bd" #bd "_bs16")]] [[kernel]] \
  decltype(rope_once_nax_kernel<dt_type, bd, 16>)                              \
      rope_once_nax_kernel<dt_type, bd, 16>;

INST_ROPE_ONCE_NAX(f16, half, 128)
INST_ROPE_ONCE_NAX(bf16, bfloat, 128)
INST_ROPE_ONCE_NAX(f16, half, 64)
INST_ROPE_ONCE_NAX(bf16, bfloat, 64)
