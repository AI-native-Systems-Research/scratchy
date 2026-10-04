// SPDX-License-Identifier: Apache-2.0
//
// A MoE block's routing from its router logits, one threadgroup per token, in one command:
//
//   [softmax over the experts]          ROUTE_PRE (Qwen's shared-expert router)
//   ascending sort of the experts       the argsort kernel's (`argpartition.metal`) block sort
//   the last top-k sorted indices       as `slice_trailing_cols`
//   their scores                        as `take_along_axis`
//   [the scores × ROUTE_SCALE]          as `scalar_mul` (Gemma's `hidden^-0.5`)
//   [softmax | renorm over the top-k]   ROUTE_POST, as `softmax.metal`'s kernels
//   [each score × its expert's scale]   ROUTE_EXPERT_SCALE, as `moe_per_expert_scale`
//
// Each step runs the code of the kernel named beside it, over the same rows, so the routing
// gives the same bits as those kernels in sequence. The threadgroup is the sort's (BN threads,
// BN * 4 >= experts); a softmax or renorm row's threads past its scores hold the reduction's
// identity, so it gives the bits of the 256-thread kernel too. Steps hand off through the
// buffers those kernels use: the logits (in place under ROUTE_PRE), the sorted experts, the
// top-k indices and the scores.
//
// Bindings: logits @ 0, sorted @ 1, top-k indices @ 2, top-k scores @ 3, per-expert scales @ 4
// (bound under ROUTE_EXPERT_SCALE only). Dispatch (1, tokens, 1) × (BN, 1, 1).

#include <metal_stdlib>
#include "baked.h"
#include "block_sort.h"
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
constant constexpr int ROUTE_ONE = 1;

template <typename T, short BN>
[[kernel, max_total_threads_per_threadgroup(BN)]] void moe_route(
    device T*       logits       [[buffer(0)]],
    device uint*    sorted       [[buffer(1)]],
    device uint*    inds         [[buffer(2)]],
    device T*       scores       [[buffer(3)]],
    const device T* expert_scale [[buffer(4)]],
    uint3 tid           [[threadgroup_position_in_grid]],
    uint3 lid           [[thread_position_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]]) {
  using sort_kernel = mlx_sort::KernelMergeSort<T, uint, true, BN, 4>;
  threadgroup T tgp_vals[sort_kernel::N_PER_BLOCK];
  threadgroup uint tgp_idxs[sort_kernel::N_PER_BLOCK];
  threadgroup float local_a[32];
  threadgroup float local_b[32];
  constexpr int E = ROUTE_EXPERTS;
  constexpr int K = ROUTE_TOP_K;
  uint row = tid.y;
  device T* row_logits = logits + size_t(row) * E;
  device T* row_scores = scores + size_t(row) * K;

  if (ROUTE_PRE) {
    mlx_softmax::softmax_row<T, float, 4>(
        row_logits, row_logits, E, int(lid.x), simd_lane_id, simd_group_id, local_a, local_b);
    threadgroup_barrier(mem_flags::mem_device);
  }
  sort_kernel::block_sort_impl(
      logits, sorted, ROUTE_EXPERTS, ROUTE_ONE, ROUTE_ONE, ROUTE_EXPERTS, ROUTE_EXPERTS, tgp_vals,
      tgp_idxs, tid, lid);
  threadgroup_barrier(mem_flags::mem_device);
  uint k = lid.x;
  if (k < uint(K)) {
    uint idx = sorted[size_t(row) * E + uint(E - K) + k];
    inds[size_t(row) * K + k] = idx;
    T s = row_logits[idx];
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
    uint expert = inds[size_t(row) * K + k];
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
