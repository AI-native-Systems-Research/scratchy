// SPDX-License-Identifier: Apache-2.0
//
// A MoE block's routing from its router logits, one threadgroup per token, in one command:
//
//   [softmax over the experts]          ROUTE_PRE (Qwen's shared-expert router)
//   the top-k experts, ascending        as the argsort kernel's (`argpartition.metal`) stable
//                                       ascending sort and `slice_trailing_cols`' last k
//   their scores                        as `take_along_axis`
//   [the scores × ROUTE_SCALE]          as `scalar_mul` (Gemma's `hidden^-0.5`)
//   [softmax | renorm over the top-k]   ROUTE_POST, as `softmax.metal`'s kernels
//   [each score × its expert's scale]   ROUTE_EXPERT_SCALE, as `moe_per_expert_scale`
//
// The top-k are picked, not sorted: simdgroup 0 takes the max of every expert's packed key k
// times. The key is the score's 16 bits mapped to an unsigned order (`LessThan`'s: every NaN
// above everything, -0 equal to +0) over the expert's index, so equal scores rank by index — the
// order a stable ascending sort leaves them in — and the k picks, read back to front, are the
// sort's last k. Every other step runs the code of the kernel named beside it, over the same rows,
// so the routing gives the same bits as those kernels in sequence. A softmax or renorm row's
// threads past its scores hold the reduction's identity, so the BN-thread threadgroup
// (BN * 4 >= experts) gives the bits of the 256-thread kernel too.
//
// Bindings: logits @ 0 (in place under ROUTE_PRE), top-k indices @ 1, top-k scores @ 2,
// per-expert scales @ 3 (bound under ROUTE_EXPERT_SCALE only). Dispatch (1, tokens, 1) ×
// (BN, 1, 1).

#include <metal_stdlib>
#include "baked.h"
#include "softmax_row.h"

using namespace metal;

// `MoeRouteConstants`, compiled in.
SCRATCHY_CONSTANT(int, ROUTE_EXPERTS, 0);
SCRATCHY_CONSTANT(int, ROUTE_TOP_K, 1);
SCRATCHY_CONSTANT(int, ROUTE_PRE, 2);
SCRATCHY_CONSTANT_OPTIONAL(float, ROUTE_SCALE, 3);
// 0: none, 1: softmax, 2: renorm.
SCRATCHY_CONSTANT(int, ROUTE_POST, 4);
SCRATCHY_CONSTANT(int, ROUTE_EXPERT_SCALE, 5);

// Expert `e`'s score `v` as a key whose unsigned order is `LessThan`'s, with ties broken by index.
template <typename T>
METAL_FUNC uint route_key(T v, uint e) {
  ushort bits = as_type<ushort>(v);
  float f = float(v);
  ushort order;
  if (isnan(f)) {
    order = 0xffff;
  } else if (f == 0.0f) {
    order = 0x8000;
  } else if (bits & 0x8000) {
    order = ushort(~bits);
  } else {
    order = ushort(bits | 0x8000);
  }
  return (uint(order) << 16) | e;
}

template <typename T, short BN>
[[kernel, max_total_threads_per_threadgroup(BN)]] void moe_route(
    device T*       logits       [[buffer(0)]],
    device uint*    inds         [[buffer(1)]],
    device T*       scores       [[buffer(2)]],
    const device T* expert_scale [[buffer(3)]],
    uint3 tid           [[threadgroup_position_in_grid]],
    uint3 lid           [[thread_position_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float local_a[32];
  threadgroup float local_b[32];
  constexpr int E = ROUTE_EXPERTS;
  constexpr int K = ROUTE_TOP_K;
  constexpr int PER_LANE = (E + 31) / 32;
  uint row = tid.y;
  device T* row_logits = logits + size_t(row) * E;
  device T* row_scores = scores + size_t(row) * K;
  device uint* row_inds = inds + size_t(row) * K;

  if (ROUTE_PRE) {
    mlx_softmax::softmax_row<T, float, 4>(
        row_logits, row_logits, E, int(lid.x), simd_lane_id, simd_group_id, local_a, local_b);
    threadgroup_barrier(mem_flags::mem_device);
  }
  if (simd_group_id == 0) {
    uint keys[PER_LANE];
    for (int j = 0; j < PER_LANE; ++j) {
      uint e = simd_lane_id + 32 * uint(j);
      // No expert's key is 0: the lowest score, -inf, maps above it.
      keys[j] = e < uint(E) ? route_key(row_logits[e], e) : 0u;
    }
    for (int r = 0; r < K; ++r) {
      uint m = 0;
      for (int j = 0; j < PER_LANE; ++j) {
        m = max(m, keys[j]);
      }
      uint best = simd_max(m);
      for (int j = 0; j < PER_LANE; ++j) {
        keys[j] = keys[j] == best ? 0u : keys[j];
      }
      if (simd_lane_id == 0) {
        row_inds[K - 1 - r] = best & 0xffff;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  uint k = lid.x;
  if (k < uint(K)) {
    T s = row_logits[row_inds[k]];
    if (ROUTE_SCALE_SET) {
      s = T(float(s) * ROUTE_SCALE);
    }
    row_scores[k] = s;
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (ROUTE_POST == 1) {
    mlx_softmax::softmax_row<T, float, 4>(
        row_scores, row_scores, K, int(lid.x), simd_lane_id, simd_group_id, local_a, local_b);
  } else if (ROUTE_POST == 2) {
    mlx_softmax::renorm_row<T, float, 4>(
        row_scores, row_scores, K, int(lid.x), simd_lane_id, simd_group_id, local_a);
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (ROUTE_EXPERT_SCALE && k < uint(K)) {
    uint expert = row_inds[k];
    float s = float(expert_scale[expert]);
    float w = float(row_scores[k]);
    row_scores[k] = static_cast<T>(w * s);
  }
}

#define INST_MOE_ROUTE(tag, type, bn) \
  SCRATCHY_KERNEL(moe_route_##tag##_bn##bn, moe_route<type, bn>)

INST_MOE_ROUTE(float16, half, 32)
INST_MOE_ROUTE(bfloat16, bfloat, 32)
INST_MOE_ROUTE(float16, half, 64)
INST_MOE_ROUTE(bfloat16, bfloat, 64)
