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
#include "megakernel/mk_common.h"

using namespace metal;

// Body shared by the dispatch kernel and the megakernel adapter (the sizes are the inline
// scalars, by reference).
template <typename SP, typename IP, typename OP, typename CI>
METAL_FUNC void take_along_axis_body(SP src, IP indices, OP out, CI src_axis_size,
                                     CI idx_axis_size, uint2 gid, uint2 grid) {
  uint k = gid.x;
  uint n = gid.y;
  if (k >= grid.x || n >= grid.y) return;
  uint idx = indices[n * uint(idx_axis_size) + k];
  // No negative-index normalization: argpartition outputs u32 in
  // [0, src_axis_size). MLX's `is_signed_v<IdxT>` branch is dead
  // for our uint indices.
  out[n * uint(idx_axis_size) + k] = src[n * uint(src_axis_size) + idx];
}

template <typename T>
MK_FUNC void mk_take_along_axis(thread const MkStep& s, MkLane l, threadgroup uchar*) {
  if (!l.live) return;
  take_along_axis_body<mk_cptr<T>, mk_cptr<uint>, mk_ptr<T>, const device int&>(
      (mk_cptr<T>)s.addr[0], (mk_cptr<uint>)s.addr[1], (mk_ptr<T>)s.addr[2],
      *(const device int*)s.addr[3], *(const device int*)s.addr[4], mk_thread_in_grid(l).xy,
      mk_threads_per_grid(s).xy);
}

#ifndef MK_BODIES_ONLY
template <typename T>
[[kernel]] void take_along_axis_2d_contig(
    const device T*    src        [[buffer(0)]],
    const device uint* indices    [[buffer(1)]],
    device T*          out        [[buffer(2)]],
    const constant int& src_axis_size [[buffer(3)]],
    const constant int& idx_axis_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  take_along_axis_body<const device T*, const device uint*, device T*, const constant int&>(
      src, indices, out, src_axis_size, idx_axis_size, gid, grid);
}

#define INSTANTIATE_TAKE(tag, type)                                     \
  template [[host_name("take_along_axis_2d_contig_" #tag)]]             \
  [[kernel]] void take_along_axis_2d_contig<type>(                      \
      const device type*  src     [[buffer(0)]],                        \
      const device uint*  indices [[buffer(1)]],                        \
      device type*        out     [[buffer(2)]],                        \
      const constant int& src_axis_size [[buffer(3)]],                  \
      const constant int& idx_axis_size [[buffer(4)]],                  \
      uint2 gid [[thread_position_in_grid]],                            \
      uint2 grid [[threads_per_grid]]);
#else
#define INSTANTIATE_TAKE(tag, type)                                                  \
  MK_TAIL(take_along_axis_2d_contig_##tag, (mk_take_along_axis<type>),      \
             MK_NO_CONSTS)
#endif

INSTANTIATE_TAKE(float32, float)
INSTANTIATE_TAKE(float16, half)
INSTANTIATE_TAKE(bfloat16, bfloat)
