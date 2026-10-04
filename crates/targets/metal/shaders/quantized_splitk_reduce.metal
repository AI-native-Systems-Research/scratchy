// SPDX-License-Identifier: Apache-2.0
//
// Sum-along-axis-0 reduction for the [split_k, M, N] intermediate
// produced by `affine_qmm_t_splitk` (see `quantized_qmm.metal`).
// Mirrors MLX's `strided_reduce_general_dispatch` invocation at
// `mlx/backend/metal/quantized.cpp:861`, which sum-reduces the
// splitk intermediate down to the final [M, N] output.
//
// MLX uses a generic strided reduction kernel; we land a purpose-
// built sum-axis-0 kernel here because (a) the shape is fully known
// (rank-3 contiguous, axis 0 = split_k), (b) the lower-side
// `MetalSplitKReduce` only ever runs in this configuration, and
// (c) the resulting one-pass kernel is easier to keep ICB-recordable
// than a generic-reduce port. If a future MoE / KV path needs a
// broader reduce surface this kernel can grow into it; until then,
// scope-limited keeps the surface honest.
//
// Baked constants:
//   REDUCE_M       — output row count (= qmm_t_splitk's M)
//   REDUCE_N       — output column count (= qmm_t_splitk's N)
//   REDUCE_SPLIT_K — partition count along axis 0
//
// Dispatch: 1 thread per output element. Grid size = REDUCE_M *
// REDUCE_N. The kernel sums float-accumulated to match the float
// accumulator inside `qmm_t_impl_inline` so SplitK + reduce composes
// to the same fp result as the equivalent qmm_t Standard run (modulo
// summation order, which differs by O(eps * split_k) — within the
// noise floor for greedy decode).

#include <metal_stdlib>
#include "baked.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, REDUCE_M, 0);
SCRATCHY_CONSTANT(uint, REDUCE_N, 1);
SCRATCHY_CONSTANT(uint, REDUCE_SPLIT_K, 2);

template <typename T>
[[kernel]] void splitk_reduce_sum(
    device       T* out [[buffer(0)]],
    const device T* in  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
  const uint mn = REDUCE_M * REDUCE_N;
  if (gid >= mn) {
    return;
  }
  // The SplitK intermediate is laid out [split_k, M, N] contiguous,
  // so `in[k * M*N + m*N + n]` is the (k, m, n) element. We reduce
  // along k for each output index `gid = m*N + n`.
  float acc = 0.0f;
  for (uint k = 0; k < REDUCE_SPLIT_K; ++k) {
    acc += float(in[k * mn + gid]);
  }
  out[gid] = static_cast<T>(acc);
}

#define INST_REDUCE(dtype_tag, mtl_type) \
  SCRATCHY_KERNEL(splitk_reduce_sum_##dtype_tag, splitk_reduce_sum<mtl_type>)

INST_REDUCE(f16,  half)
INST_REDUCE(bf16, bfloat)
