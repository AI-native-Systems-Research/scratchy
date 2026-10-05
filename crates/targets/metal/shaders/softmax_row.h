// SPDX-License-Identifier: Apache-2.0
//
// One row's softmax (MLX `softmax_single_row`, `mlx/backend/metal/kernels/softmax.h` lines 10-98)
// and one row's renormalization to sum one, run by one threadgroup over `axis_size` scores.
// Shared by the softmax kernels (`softmax.metal`) and the MoE routing kernel (`moe_route.metal`).
//
// `in` / `out` point at the row, in any address space. `local_a` / `local_b` are the caller's threadgroup scratch, 32
// entries each (a kernel declares threadgroup memory; a function cannot). A thread past the row
// reads the reduction's identity, so a threadgroup of any size covering `axis_size / N_READS`
// threads gives the same bits.

#pragma once

#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>

namespace mlx_softmax {

using namespace metal;

template <typename T>
struct Limits {
  static constant constexpr const T min = numeric_limits<T>::lowest();
  static constant constexpr const T finite_min = numeric_limits<T>::lowest();
};

template <typename T>
inline T softmax_exp(T x) {
  // MLX softmax.h:4 — x is in (-oo, 0] post max-subtract, so
  // fast::exp is fine; the subsequent divide-by-sum normalizes.
  return fast::exp(x);
}

template <typename T, typename AccT, int N_READS, typename PI, typename PO>
METAL_FUNC void softmax_row(
    PI in,
    PO out,
    int axis_size,
    int lid,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup AccT* local_max,
    threadgroup AccT* local_normalizer) {
  AccT ld[N_READS];

  in += lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      ld[i] = AccT(in[i]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      ld[i] = ((lid * N_READS + i) < axis_size) ? AccT(in[i])
                                                : Limits<AccT>::min;
    }
  }
  if (simd_group_id == 0) {
    local_max[simd_lane_id] = Limits<AccT>::min;
    local_normalizer[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  AccT maxval = Limits<AccT>::finite_min;
  for (int i = 0; i < N_READS; i++) {
    maxval = (maxval < ld[i]) ? ld[i] : maxval;
  }
  maxval = simd_max(maxval);
  if (simd_lane_id == 0) {
    local_max[simd_group_id] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    maxval = simd_max(local_max[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_max[0] = maxval;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = local_max[0];

  AccT normalizer = 0;
  for (int i = 0; i < N_READS; i++) {
    AccT exp_x = softmax_exp(ld[i] - maxval);
    ld[i] = exp_x;
    normalizer += exp_x;
  }
  normalizer = simd_sum(normalizer);
  if (simd_lane_id == 0) {
    local_normalizer[simd_group_id] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    normalizer = simd_sum(local_normalizer[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_normalizer[0] = normalizer;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = 1 / local_normalizer[0];

  out += lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      out[i] = T(ld[i] * normalizer);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        out[i] = T(ld[i] * normalizer);
      }
    }
  }
}

template <typename T, typename AccT, int N_READS, typename PI, typename PO>
METAL_FUNC void renorm_row(
    PI in,
    PO out,
    int axis_size,
    int lid,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup AccT* local_sum) {
  AccT ld[N_READS];
  in  += lid * N_READS;
  out += lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) ld[i] = AccT(in[i]);
  } else {
    for (int i = 0; i < N_READS; i++) {
      ld[i] = ((lid * N_READS + i) < axis_size) ? AccT(in[i]) : AccT(0);
    }
  }

  if (simd_group_id == 0) local_sum[simd_lane_id] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  AccT s = 0;
  for (int i = 0; i < N_READS; i++) s += ld[i];
  s = simd_sum(s);
  if (simd_lane_id == 0) local_sum[simd_group_id] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    s = simd_sum(local_sum[simd_lane_id]);
    if (simd_lane_id == 0) local_sum[0] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  AccT total = local_sum[0];
  AccT inv = (total > AccT(0)) ? (AccT(1) / total) : AccT(0);

  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) out[i] = T(ld[i] * inv);
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (lid * N_READS + i < axis_size) out[i] = T(ld[i] * inv);
    }
  }
}

}  // namespace mlx_softmax
