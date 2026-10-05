// SPDX-License-Identifier: Apache-2.0
//
// A MoE block's routing from its router logits, one threadgroup per token, in one command: the
// program `moe_route.h` spells, over device rows — the softmax over the experts in place over the
// logits.
//
// Bindings: logits @ 0 (in place under ROUTE_PRE), top-k indices @ 1, top-k scores @ 2,
// per-expert scales @ 3 (bound under ROUTE_EXPERT_SCALE only). Dispatch (1, tokens, 1) ×
// (BN, 1, 1).

#include <metal_stdlib>
#include "baked.h"
#include "moe_route.h"

using namespace metal;

// `MoeRouteConstants`, compiled in.
SCRATCHY_CONSTANT(int, ROUTE_EXPERTS, 0);
SCRATCHY_CONSTANT(int, ROUTE_TOP_K, 1);
SCRATCHY_CONSTANT(int, ROUTE_PRE, 2);
SCRATCHY_CONSTANT_OPTIONAL(float, ROUTE_SCALE, 3);
// 0: none, 1: softmax, 2: renorm.
SCRATCHY_CONSTANT(int, ROUTE_POST, 4);
SCRATCHY_CONSTANT(int, ROUTE_EXPERT_SCALE, 5);

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
  uint row = tid.y;
  device T* row_logits = logits + size_t(row) * E;
  device T* row_scores = scores + size_t(row) * K;
  device uint* row_inds = inds + size_t(row) * K;
  route_top_k<T, E, K, ROUTE_PRE != 0>(
      row_logits, row_logits, row_inds, lid.x, simd_lane_id, simd_group_id, local_a, local_b);
  route_scores<T, K, ROUTE_PRE != 0, ROUTE_SCALE_SET, ROUTE_POST, ROUTE_EXPERT_SCALE != 0>(
      row_logits, row_logits, row_inds, row_scores, expert_scale, ROUTE_SCALE, lid.x,
      simd_lane_id, simd_group_id, local_a, local_b);
}

#define INST_MOE_ROUTE(tag, type, bn) \
  SCRATCHY_KERNEL(moe_route_##tag##_bn##bn, moe_route<type, bn>)

INST_MOE_ROUTE(float16, half, 32)
INST_MOE_ROUTE(bfloat16, bfloat, 32)
INST_MOE_ROUTE(float16, half, 64)
INST_MOE_ROUTE(bfloat16, bfloat, 64)
