// SPDX-License-Identifier: Apache-2.0
//
// `mx.take_along_axis(gates, inds, axis=-1)` lowered to a Metal
// kernel. Specialization of MLX's `gather_axis` (see
// `mlx/backend/metal/kernels/indexing/gather_axis.h`) to the 2D
// contiguous case both source and index sides — the only shape the
// MoE router uses:
//
//   gates [N, num_experts]  T   (router probs)
//   inds  [N, top_k]        u32 (top-k indices from argpartition)
//   out   [N, top_k]        T   (per-token expert scores)
//
// Both are row-contiguous; axis = -1. Per MLX gather_axis.h the
// generic formula collapses for SrcC=IdxC=true to:
//
//   out[n, k] = src[n, indices[n, k]]
//
// One thread per (n, k). Dispatch (top_k, N, 1).

#include <metal_stdlib>
#include "baked.h"

using namespace metal;

// `MoeTopKConstants`, compiled in: the source row's width and the index row's.
SCRATCHY_CONSTANT(int, SRC_AXIS_SIZE, 0);
SCRATCHY_CONSTANT(int, IDX_AXIS_SIZE, 1);

template <typename T>
[[kernel]] void take_along_axis_2d_contig(
    const device T*    src        [[buffer(0)]],
    const device uint* indices    [[buffer(1)]],
    device T*          out        [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  uint k = gid.x;
  uint n = gid.y;
  if (k >= grid.x || n >= grid.y) return;
  uint idx = indices[n * uint(IDX_AXIS_SIZE) + k];
  // No negative-index normalization: argpartition outputs u32 in
  // [0, SRC_AXIS_SIZE). MLX's `is_signed_v<IdxT>` branch is dead
  // for our uint indices.
  out[n * uint(IDX_AXIS_SIZE) + k] = src[n * uint(SRC_AXIS_SIZE) + idx];
}

SCRATCHY_KERNEL(take_along_axis_2d_contig_float16, take_along_axis_2d_contig<half>)
SCRATCHY_KERNEL(take_along_axis_2d_contig_bfloat16, take_along_axis_2d_contig<bfloat>)
