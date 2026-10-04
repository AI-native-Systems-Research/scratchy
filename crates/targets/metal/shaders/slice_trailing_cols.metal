// SPDX-License-Identifier: Apache-2.0
//
// `out[n, k] = in[n, axis_size - top_k + k]`. Used after the
// argpartition kernel (which produces a full ascending sort over
// [N, num_experts]) to extract the top-k indices.
//
// Why a dedicated kernel: MTLBuffer offsets are per-binding, not
// per-row. A row-wise trailing slice can't be expressed as a
// buffer offset alone, so the gather happens explicitly. One
// thread per (n, k); dispatch (top_k, N, 1).

#include <metal_stdlib>
#include "baked.h"

using namespace metal;

// `MoeTopKConstants`, compiled in.
SCRATCHY_CONSTANT(int, AXIS_SIZE, 0);
SCRATCHY_CONSTANT(int, TOP_K, 1);

template <typename U>
[[kernel]] void slice_trailing_cols(
    const device U* src [[buffer(0)]],
    device U*       dst [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]) {
  uint k = gid.x;
  uint n = gid.y;
  uint src_col = uint(AXIS_SIZE - TOP_K) + k;
  dst[n * uint(TOP_K) + k] = src[n * uint(AXIS_SIZE) + src_col];
}

SCRATCHY_KERNEL(slice_trailing_cols_u32, slice_trailing_cols<uint>)
