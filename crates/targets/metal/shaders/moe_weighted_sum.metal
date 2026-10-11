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
// A sorted bake (`MWS_SORTED`) reads pair (n, k)'s row where the sort put
// it, `expert_out[pos[n * top_k + k]]` (`moe_group.metal`'s `pos`).
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
SCRATCHY_CONSTANT_OPTIONAL(bool, MWS_SORTED, 2);
// 3 (`MWS_LINEAR_BIAS`): gpt-oss's grouped bake — pair k's expert adds its down LINEAR bias
// before its score weights it ([E, hidden], the expert's row at `down_bias[e·hidden + d]`),
// the decomposed sibling of the gathered `affine_gather_qmv_combine` epilogue (buffer(10)
// there). Unset: no buffer(4)/(5), and the add const-folds away.
SCRATCHY_CONSTANT_OPTIONAL(bool, MWS_LINEAR_BIAS_FC, 3);
constant constexpr bool MWS_LINEAR_BIAS = MWS_LINEAR_BIAS_FC_SET && MWS_LINEAR_BIAS_FC;

template <typename T>
[[kernel]] void moe_weighted_sum(
    const device T* expert_out [[buffer(0)]],
    const device T* scores     [[buffer(1)]],
    device T*       out        [[buffer(2)]],
    const device uint* pos     [[buffer(3)]],
    const device uint* topk_indices [[buffer(4)]],
    const device T*  down_bias      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  uint d = gid.x;
  uint n = gid.y;
  if (d >= uint(MWS_HIDDEN) || n >= grid.y) return;
  float acc = 0.0f;
  device const T* row_scores = scores + n * uint(MWS_TOP_K);
  for (int k = 0; k < MWS_TOP_K; ++k) {
    const uint pair = n * uint(MWS_TOP_K) + uint(k), row = MWS_SORTED ? pos[pair] : pair;
    float r = float(expert_out[size_t(row) * uint(MWS_HIDDEN) + d]);
    if (MWS_LINEAR_BIAS) {
      r += float(down_bias[size_t(topk_indices[pair]) * size_t(MWS_HIDDEN) + d]);
    }
    acc = fma(r, float(row_scores[k]), acc);
  }
  out[n * uint(MWS_HIDDEN) + d] = T(acc);
}

#define INSTANTIATE_MWS(tag, type) \
  SCRATCHY_KERNEL(moe_weighted_sum_##tag, moe_weighted_sum<type>)

INSTANTIATE_MWS(float16, half)
INSTANTIATE_MWS(bfloat16, bfloat)
INSTANTIATE_MWS(float32, float)
