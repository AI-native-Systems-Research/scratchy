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
// `MWS_TOP_K` and `MWS_HIDDEN` are baked constants so
// the inner-loop bound is a compile-time constant the optimizer
// can fully unroll for the small top_k values we care about (2-8).
// One thread per (n, d). Dispatch (hidden, N, 1).

#include <metal_stdlib>
#include "baked.h"

using namespace metal;

SCRATCHY_CONSTANT(int, MWS_TOP_K, 0);
SCRATCHY_CONSTANT(int, MWS_HIDDEN, 1);

template <typename T>
[[kernel]] void moe_weighted_sum(
    const device T* expert_out [[buffer(0)]],
    const device T* scores     [[buffer(1)]],
    device T*       out        [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  uint d = gid.x;
  uint n = gid.y;
  if (d >= uint(MWS_HIDDEN) || n >= grid.y) return;
  float acc = 0.0f;
  device const T* row_scores = scores + n * uint(MWS_TOP_K);
  device const T* row_expert = expert_out + n * uint(MWS_TOP_K) * uint(MWS_HIDDEN);
  for (int k = 0; k < MWS_TOP_K; ++k) {
    acc = fma(float(row_expert[uint(k) * uint(MWS_HIDDEN) + d]),
              float(row_scores[k]),
              acc);
  }
  out[n * uint(MWS_HIDDEN) + d] = T(acc);
}

#define INSTANTIATE_MWS(tag, type) \
  SCRATCHY_KERNEL(moe_weighted_sum_##tag, moe_weighted_sum<type>)

INSTANTIATE_MWS(float16, half)
INSTANTIATE_MWS(bfloat16, bfloat)
INSTANTIATE_MWS(float32, float)
