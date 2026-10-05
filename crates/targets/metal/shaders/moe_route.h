// SPDX-License-Identifier: Apache-2.0
//
// A MoE block's routing of one token's router logits, run by one threadgroup — the program
// `moe_route.metal` runs as a command, and the one-row expert kernels run themselves
// (`MetalFusion::MoeRouted`), each from the logits:
//
//   [softmax over the experts]          PRE (Qwen's shared-expert router)
//   the top-k experts, ascending        as the argsort kernel's (`argpartition.metal`) stable
//                                       ascending sort and `slice_trailing_cols`' last k
//   their scores                        as `take_along_axis`
//   [the scores × scale]                SCALED, as `scalar_mul` (Gemma's `hidden^-0.5`)
//   [softmax | renorm over the top-k]   POST 1 | 2, as `softmax.metal`'s kernels
//   [each score × its expert's scale]   EXPERT_SCALED, as `moe_per_expert_scale`
//
// The top-k are picked, not sorted: simdgroup 0 takes the max of every expert's packed key k
// times. The key is the score's 16 bits mapped to an unsigned order (`LessThan`'s: every NaN
// above everything, -0 equal to +0) over the expert's index, so equal scores rank by index — the
// order a stable ascending sort leaves them in — and the k picks, read back to front, are the
// sort's last k. Every other step runs the code of the kernel named beside it, over the same rows,
// so the routing gives the same bits as those kernels in sequence. A softmax or renorm row's
// threads past its scores hold the reduction's identity, so any threadgroup of at least E / 4
// threads gives the bits of the 256-thread kernel too.
//
// The rows are pointers of any address space: the routing command's are device memory (the
// softmax in place over the logits), an expert kernel's its threadgroup memory.

#pragma once

#include <metal_stdlib>
#include "softmax_row.h"

using namespace metal;

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

// Every thread of the threadgroup, after its writes to either memory, before its reads of them.
METAL_FUNC void route_barrier() {
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
}

// The top-k of E experts' scores into `inds`: `logits`, under PRE softmaxed into `soft` first.
// Thread `lid` of the threadgroup; every thread returns with the picks in `inds`.
template <typename T, int E, int K, bool PRE, typename PL, typename PS, typename PI>
METAL_FUNC void route_top_k(
    PL logits,
    PS soft,
    PI inds,
    uint lid,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup float* local_a,
    threadgroup float* local_b) {
  constexpr int PER_LANE = (E + 31) / 32;
  if (PRE) {
    mlx_softmax::softmax_row<T, float, 4>(
        logits, soft, E, int(lid), simd_lane_id, simd_group_id, local_a, local_b);
    route_barrier();
  }
  if (simd_group_id == 0) {
    uint keys[PER_LANE];
    for (int j = 0; j < PER_LANE; ++j) {
      uint e = simd_lane_id + 32 * uint(j);
      // No expert's key is 0: the lowest score, -inf, maps above it.
      keys[j] = e < uint(E) ? route_key(PRE ? T(soft[e]) : T(logits[e]), e) : 0u;
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
        inds[K - 1 - r] = best & 0xffff;
      }
    }
  }
  route_barrier();
}

// The top-k picks' scores into `scores`, from the scores `route_top_k` picked from (`soft` under
// PRE, else `logits`): each [× scale], [softmaxed | renormed over the k], [× its expert's scale].
// Thread `lid` of the threadgroup; every thread returns with the scores in `scores`.
template <typename T, int K, bool PRE, bool SCALED, int POST, bool EXPERT_SCALED, typename PL,
          typename PS, typename PI, typename PO>
METAL_FUNC void route_scores(
    PL logits,
    PS soft,
    PI inds,
    PO scores,
    const device T* expert_scale,
    float scale,
    uint lid,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup float* local_a,
    threadgroup float* local_b) {
  if (lid < uint(K)) {
    T s = PRE ? T(soft[inds[lid]]) : T(logits[inds[lid]]);
    if (SCALED) {
      s = T(float(s) * scale);
    }
    scores[lid] = s;
  }
  route_barrier();
  if (POST == 1) {
    mlx_softmax::softmax_row<T, float, 4>(
        scores, scores, K, int(lid), simd_lane_id, simd_group_id, local_a, local_b);
  } else if (POST == 2) {
    mlx_softmax::renorm_row<T, float, 4>(
        scores, scores, K, int(lid), simd_lane_id, simd_group_id, local_a);
  }
  route_barrier();
  if (EXPERT_SCALED && lid < uint(K)) {
    float s = float(expert_scale[inds[lid]]);
    float w = float(scores[lid]);
    scores[lid] = static_cast<T>(w * s);
  }
  route_barrier();
}
