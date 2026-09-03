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

using namespace metal;

[[kernel]] void slice_trailing_cols_u32(
    const device uint* src [[buffer(0)]],
    device uint*       dst [[buffer(1)]],
    const constant int& axis_size [[buffer(2)]],
    const constant int& top_k     [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  uint k = gid.x;
  uint n = gid.y;
  uint src_col = uint(axis_size - top_k) + k;
  dst[n * uint(top_k) + k] = src[n * uint(axis_size) + src_col];
}
