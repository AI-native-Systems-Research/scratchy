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
#include "megakernel/mk_common.h"

using namespace metal;

// Body shared by the dispatch kernel and the megakernel adapter (src / dst coherent there; the
// sizes are the inline scalars, by reference).
template <typename SP, typename DP, typename CI>
METAL_FUNC void slice_trailing_cols_body(SP src, DP dst, CI axis_size, CI top_k, uint2 gid) {
  uint k = gid.x;
  uint n = gid.y;
  uint src_col = uint(axis_size - top_k) + k;
  dst[n * uint(top_k) + k] = src[n * uint(axis_size) + src_col];
}

MK_FUNC void mk_slice_trailing_cols_u32(thread const MkStep& s, MkLane l, threadgroup uchar*) {
  if (!l.live) return;
  slice_trailing_cols_body<mk_cptr<uint>, mk_ptr<uint>, const device int&>(
      (mk_cptr<uint>)s.addr[0], (mk_ptr<uint>)s.addr[1], *(const device int*)s.addr[2],
      *(const device int*)s.addr[3], mk_thread_in_grid(l).xy);
}

#ifndef MK_BODIES_ONLY
[[kernel]] void slice_trailing_cols_u32(
    const device uint* src [[buffer(0)]],
    device uint*       dst [[buffer(1)]],
    const constant int& axis_size [[buffer(2)]],
    const constant int& top_k     [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  slice_trailing_cols_body<const device uint*, device uint*, const constant int&>(
      src, dst, axis_size, top_k, gid);
}
#else
MK_TAIL(slice_trailing_cols_u32, 0x3, (mk_slice_trailing_cols_u32), MK_NO_CONSTS)
#endif
