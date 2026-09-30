// SPDX-License-Identifier: Apache-2.0
//
// MoE reduction: `out[n, d] = Σ_k expert_out[n, k, d] * scores[n, k]`.
// Faithful translation of the Python expression in
//   qwen3_moe.py:137 / qwen2_moe.py:138 / mixtral.py:119
//     y = (y * scores[..., None]).sum(axis=-2)
//
// Layout:
//   expert_out  [N, top_k, hidden]   T_act (bf16/f16/f32)
//   scores      [N, top_k]           T_act
//   out         [N, hidden]          T_act
//
// `MWS_TOP_K` and `MWS_HIDDEN` are baked as function_constants so
// the inner-loop bound is a compile-time constant the optimizer
// can fully unroll for the small top_k values we care about (2-8).
// One thread per (n, d). Dispatch (hidden, N, 1).

#include <metal_stdlib>
#include "megakernel/mk_common.h"

using namespace metal;

#define MWS_CONSTS(X) X(int, top_k, MWS_TOP_K, 0) X(int, hidden, MWS_HIDDEN, 1)
#ifndef MK_BODIES_ONLY
MWS_CONSTS(MK_FC_DECLARE)
struct MwsFc {
  MWS_CONSTS(MK_FC_ACCESSOR)
};
#endif

// Body shared by the dispatch kernel and the megakernel adapter (every operand coherent there).
template <typename T, typename C, typename EP, typename SP, typename OP>
METAL_FUNC void moe_weighted_sum_body(EP expert_out, SP scores, OP out, uint2 gid, uint2 grid) {
  uint d = gid.x;
  uint n = gid.y;
  if (d >= uint(C::hidden()) || n >= grid.y) return;
  float acc = 0.0f;
  auto row_scores = scores + n * uint(C::top_k());
  auto row_expert = expert_out + n * uint(C::top_k()) * uint(C::hidden());
  for (int k = 0; k < C::top_k(); ++k) {
    acc = fma(float(row_expert[uint(k) * uint(C::hidden()) + d]),
              float(row_scores[k]),
              acc);
  }
  out[n * uint(C::hidden()) + d] = T(acc);
}

template <typename T, typename C>
MK_FUNC void mk_moe_weighted_sum(thread const MkStep& s, MkLane l, threadgroup uchar*) {
  if (!l.live) return;
  moe_weighted_sum_body<T, C>((mk_cptr<T>)s.addr[0], (mk_cptr<T>)s.addr[1], (mk_ptr<T>)s.addr[2],
                              mk_thread_in_grid(l).xy, mk_threads_per_grid(s).xy);
}

#ifndef MK_BODIES_ONLY
template <typename T>
[[kernel]] void moe_weighted_sum(
    const device T* expert_out [[buffer(0)]],
    const device T* scores     [[buffer(1)]],
    device T*       out        [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  moe_weighted_sum_body<T, MwsFc>(expert_out, scores, out, gid, grid);
}

#define INSTANTIATE_MWS(tag, type)                                       \
  template [[host_name("moe_weighted_sum_" #tag)]]                       \
  [[kernel]] void moe_weighted_sum<type>(                                \
      const device type* expert_out [[buffer(0)]],                       \
      const device type* scores     [[buffer(1)]],                       \
      device type*       out        [[buffer(2)]],                       \
      uint2 gid [[thread_position_in_grid]],                             \
      uint2 grid [[threads_per_grid]]);
#else
#define INSTANTIATE_MWS(tag, type)                                                            \
  MK_TAIL(moe_weighted_sum_##tag, 0x7, (mk_moe_weighted_sum<type, MK_C>), MWS_CONSTS)
#endif

INSTANTIATE_MWS(float16, half)
INSTANTIATE_MWS(bfloat16, bfloat)
INSTANTIATE_MWS(float32, float)
