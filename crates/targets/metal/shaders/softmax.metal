// SPDX-License-Identifier: Apache-2.0
//
// Faithful port of MLX's `softmax_single_row` from
// `mlx/backend/metal/kernels/softmax.h` (lines 10-98). MLX template
// args (T, AccT, N_READS) are baked here as monomorphic
// instantiations covering the MoE-router shape we need:
//
//   block_softmax_precise_<dtype>   T=<dtype>, AccT=float, N_READS=4
//
// `precise=true` matches `mx.softmax(..., precise=True)` from
// `qwen3_moe.py:128`, `qwen2_moe.py:131`, `mixtral.py:116` — the
// router probs must use float accumulation. axis_size rides as
// `constant int& [[buffer(2)]]` (NOT a function constant) so the
// same pipeline serves every num_experts value Mixtral / Qwen MoE
// hands us at runtime.
//
// Dispatch: threadgroups (rows, 1, 1), threads_per_threadgroup
// (BLOCK_THREADS, 1, 1) where BLOCK_THREADS * N_READS >= axis_size.
// For MoE router widths up to ~256, BLOCK_THREADS = ceildiv(axis_size,
// N_READS) rounded up to the nearest power of two works; the
// callers in scratchy-target-metal dispatch with BLOCK_THREADS = 1024
// (max simd-cooperative size) which covers axis_size <= 4096.
//
// Symbol naming follows the MLX precedent
// (`block_softmax_precise_float16`, `block_softmax_precise_bfloat16`)
// so the symbol parses uniformly under `shader_cache::library_for`.

#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

#define MLX_N_READS 4

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

template <typename T, typename AccT = T, int N_READS = MLX_N_READS>
[[kernel]] void softmax_single_row(
    const device T* in,
    device T* out,
    constant int& axis_size,
    uint gid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  int lid = _lid;

  constexpr int SIMD_SIZE = 32;

  threadgroup AccT local_max[SIMD_SIZE];
  threadgroup AccT local_normalizer[SIMD_SIZE];

  AccT ld[N_READS];

  in += gid * size_t(axis_size) + lid * N_READS;
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

  out += gid * size_t(axis_size) + lid * N_READS;
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

#define INSTANTIATE_PRECISE(tag, type)                          \
  template [[host_name("block_softmax_precise_" #tag)]]         \
  [[kernel]] void softmax_single_row<type, float, MLX_N_READS>( \
      const device type* in,                                    \
      device type* out,                                         \
      constant int& axis_size,                                  \
      uint gid [[threadgroup_position_in_grid]],                \
      uint _lid [[thread_position_in_threadgroup]],             \
      uint simd_lane_id [[thread_index_in_simdgroup]],          \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define INSTANTIATE_NONPRECISE(tag, type)                       \
  template [[host_name("block_softmax_" #tag)]]                 \
  [[kernel]] void softmax_single_row<type, type, MLX_N_READS>(  \
      const device type* in,                                    \
      device type* out,                                         \
      constant int& axis_size,                                  \
      uint gid [[threadgroup_position_in_grid]],                \
      uint _lid [[thread_position_in_threadgroup]],             \
      uint simd_lane_id [[thread_index_in_simdgroup]],          \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

// Precise variants use AccT=float so all simd reductions land on
// float; non-precise variants stay in the input dtype. MSL's native
// `bfloat` lacks simd_max / simd_sum overloads under the current
// metal toolchain, so we only instantiate the precise bf16 form —
// which is the only one the MoE router needs (qwen3_moe.py:128
// passes `precise=True`).
INSTANTIATE_PRECISE(float16, half)
INSTANTIATE_PRECISE(bfloat16, bfloat)
INSTANTIATE_NONPRECISE(float32, float)
INSTANTIATE_NONPRECISE(float16, half)

// ─────────────────────────────────────────────────────────────────
// topk_renorm — Qwen3-MoE `norm_topk_prob=True` row renormalize.
//
// MLX reference: `mx.softmax(router_logits)` → `take_along_axis(probs, topk_inds)`
// → `weights = weights / weights.sum(axis=-1, keepdims=True)` (when
// norm_topk_prob is set). The Metal lowering pulls the top-k gathered
// scores from `topk_scores` and applies this kernel in-place; AffineGatherQmv
// downstream consumes the renormalized weights.
//
// Dispatch: threadgroups (num_tokens, 1, 1), one threadgroup per row,
// threads_per_threadgroup (BLOCK_THREADS, 1, 1). axis_size = top_k
// (typically 4 or 8); float accumulation in registers; cast back to
// `T` on write.
//
// Symbol naming follows the softmax precedent.
// ─────────────────────────────────────────────────────────────────

template <typename T, typename AccT = float, int N_READS = MLX_N_READS>
[[kernel]] void topk_renorm_single_row(
    const device T* in,
    device T* out,
    constant int& axis_size,
    uint gid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  int lid = _lid;
  constexpr int SIMD_SIZE = 32;
  threadgroup AccT local_sum[SIMD_SIZE];

  AccT ld[N_READS];
  in  += gid * size_t(axis_size) + lid * N_READS;
  out += gid * size_t(axis_size) + lid * N_READS;
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

#define INSTANTIATE_TOPK_RENORM(tag, type)                          \
  template [[host_name("topk_renorm_" #tag)]]                       \
  [[kernel]] void topk_renorm_single_row<type, float, MLX_N_READS>( \
      const device type* in,                                        \
      device type* out,                                             \
      constant int& axis_size,                                      \
      uint gid [[threadgroup_position_in_grid]],                    \
      uint _lid [[thread_position_in_threadgroup]],                 \
      uint simd_lane_id [[thread_index_in_simdgroup]],              \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

INSTANTIATE_TOPK_RENORM(float16, half)
INSTANTIATE_TOPK_RENORM(bfloat16, bfloat)
