// SPDX-License-Identifier: Apache-2.0
//
// Splash's staged decode tiles (~/git/splash @ 1482f6bb runtime/metal/kernels/shared/gguf_linear.metal:60-180
// `gguf_decode_tile`, `gguf_decode_gate_up`; kernels/common/gguf_staged_tile.h:45-211 `staged_first_step`,
// `gguf_staged_steps`, `StagedColumn`, `gguf_staged_pair_steps`; kernels/common/split_reduce.h), over MLX-affine
// 4-bit weights. With these changes, and no other:
//
//   - The weights are read either as MLX keeps them (codes `[N][K/8]` u32, scales and biases `[N][K/G]`, the codes
//     offset-8 where `AffineCodes::Offset8` stores them so, XORed back first) or, with PLANES, as Splash's af4g64
//     image, byte for byte (metal/abi/QuantFormat.h): a 64-column tile's group of 32 is its columns' 16 code bytes
//     side by side, in lane-owned chunk order, and its meta unit their (scale, bias) bf16 pairs. Either decodes as
//     Splash's `FmtAffine<4, 2>` does, value s * code + z in fp32 rounded once to half.
//   - The rows the tile reads are the bucket's (`QMM_M`), bounds-checked by the matmul's dynamic extent: a bucket's
//     arena holds its own rows, not a tile's (Splash's `LinearPlan::storageRows`).
//   - The K partitions are the grid's height: the dispatch sets them from the device's GPU cores (Splash's
//     `decodeSplits`), so no constant names them.
//
// The decode tile is 64 columns a threadgroup: two simdgroups of 32 (`GGUF_STAGED_COLUMNS`), each staging its own. A
// step is 32 inputs (`GGUF_STAGED_STEP`): every lane dequantizes its column's step into the simdgroup's half stage,
// the simdgroup meets, loads the next step's codes and scale while the step's `matmul2d` runs over the bucket's rows,
// and the stages alternate. The gate/up tile is one simdgroup staging the gate's and the up's same 32 columns each
// step, so the step's latency serves both tensors, and writes bf16(up) * silu(bf16(gate)). A split threadgroup stores
// its fp32 partials and arrives at its tile's counter; the last to arrive adds the partitions in split order and
// returns the counter to zero. With STAGED_SPREAD each column tile starts its walk at its own step.
//
// Baked constants: QMM_K, QMM_N, QMM_M (slots 0-2), STAGED_SPREAD (slot 3), AFFINE_CODES_OFFSET8 (slot 5).
// Decode tile buffers: w[0], scales[1], biases[2], x[3], y[4], partials[5] (`[splits][M][N]` fp32), counters[6]
// (one u32 a 64-column tile, zero between dispatches); grid (N/64, splits), 64 threads.
// Gate/up buffers: gate w, scales, biases [0-2], up w, scales, biases [3-5], x[6], y[7], partials[8]
// (`[splits][M][2N]` fp32: the gate's columns, then the up's at N on), counters[9] (one a 32-column tile); grid
// (N/32, splits), 32 threads. PLANES reads each tensor's meta at its scales binding.

#pragma clang fp reassociate(off)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include "baked.h"

using namespace metal;
using namespace mpp::tensor_ops;

SCRATCHY_CONSTANT_OPTIONAL(int, QMM_K, 0);
SCRATCHY_CONSTANT_OPTIONAL(int, QMM_N, 1);
SCRATCHY_CONSTANT_OPTIONAL(int, QMM_M, 2);
SCRATCHY_CONSTANT_OPTIONAL(bool, STAGED_SPREAD, 3);
SCRATCHY_CONSTANT_OPTIONAL(bool, AFFINE_CODES_OFFSET8, 5);

// GGUF_TILE_COLUMNS, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP (metal/abi/Gguf.h).
constant constexpr uint kTileColumns = 64;
constant constexpr uint kStagedColumns = 32;
constant constexpr uint kStagedStep = 32;

// Splash's `staged_first_step`: spread starts tile t's walk (t * 40503 mod 2^16) / 2^16 of the way into its
// partition, the golden ratio's fraction in 16 bits, so a run of tiles starts evenly over it and most find the input
// slice they fetch brought in by others.
inline uint staged_first_step(uint tile, uint step_begin, uint step_end, bool spread) {
  if (!spread) return step_begin;
  return step_begin + uint(ulong((tile * 40503u) & 0xFFFFu) * (step_end - step_begin) >> 16);
}

// Splash's af4g64 decode of one group into the column's stage row: value = s * code + z in fp32, rounded once to
// half (`staged_linear`, kernels/common/gguf_staged.h:12-17), the 4-bit code exact in half's mantissa.
inline void affine_dequant32(uint4 packed, float2 meta, threadgroup half *dst) {
#pragma unroll
  for (ushort j = 0; j < 4; ++j) {
    const uint w = AFFINE_CODES_OFFSET8 ? packed[j] ^ 0x88888888u : packed[j];
    // Elements 8j..8j+7: even codes in t's bytes, odd codes in u's.
    const uint t = w & 0x0F0F0F0Fu, u = (w >> 4) & 0x0F0F0F0Fu;
    half4 lo, hi;
#pragma unroll
    for (ushort p = 0; p < 4; ++p) {
      // 0x6400 is half 1024, whose ulp is 1: or-ing a code into its mantissa makes 1024 + code, exactly.
      const uint pair = ((t >> (8 * p)) & 0xFFu) | (((u >> (8 * p)) & 0xFFu) << 16);
      const float2 code = float2(as_type<half2>(pair | 0x64006400u) - half2(1024.0h));
      const half2 v = half2(fma(code, float2(meta.x), float2(meta.y)));
      if (p < 2) {
        lo[2 * p] = v.x;
        lo[2 * p + 1] = v.y;
      } else {
        hi[2 * (p - 2)] = v.x;
        hi[2 * (p - 2) + 1] = v.y;
      }
    }
    *((threadgroup half4 *)(dst + 8 * j)) = lo;
    *((threadgroup half4 *)(dst + 8 * j + 4)) = hi;
  }
}

// Splash's af4g64 group as its planes hold it (`dequant32` over `FmtQ4K`'s chunks, kernels/common/gguf_staged.h,
// quant_formats.h `quant_nibble_pairs`): word c holds chunk c's pairs, pair p's e0 at bits 4p and e1 at 16 + 4p;
// pairs 0, 1 are elements 4c..4c+3 and pairs 2, 3 elements 16+4c..16+4c+3.
inline void splash_dequant32(uint4 packed, float2 meta, threadgroup half *dst) {
#pragma unroll
  for (ushort c = 0; c < 4; ++c) {
    const uint4 pairs = (uint4(packed[c]) >> uint4(0, 4, 8, 12)) & 0x000F000Fu;
    half2 v[4];
#pragma unroll
    for (ushort p = 0; p < 4; ++p) {
      const float2 code = float2(as_type<half2>(pairs[p] | 0x64006400u) - half2(1024.0h));
      v[p] = half2(fma(code, float2(meta.x), float2(meta.y)));
    }
    *((threadgroup half4 *)(dst + 4 * c)) = half4(v[0], v[1]);
    *((threadgroup half4 *)(dst + 16 + 4 * c)) = half4(v[2], v[3]);
  }
}

// One column of a tensor as a lane stages it (Splash's `StagedColumn`): where its codes and meta are, the payload of
// the step it stages next and that step's (scale, bias).
template <typename S, int G, bool PLANES> struct StagedColumn {
  static constexpr constant uint kStepsPerGroup = G / kStagedStep;
  static constexpr constant uint kGroups = uint(QMM_K) / G;
  const device uint32_t *w;
  const device S *scales, *biases;
  uint column;
  uint4 packed;
  float2 meta;

  uint4 codes(uint step) const {
    if (PLANES) {
      const uint tile = column / kTileColumns, in_tile = column % kTileColumns;
      return ((device const uint4 *)w)[(ulong(tile) * (uint(QMM_K) / kStagedStep) + step) * kTileColumns + in_tile];
    }
    return ((device const uint4 *)(w + ulong(column) * (uint(QMM_K) / 8)))[step];
  }
  float2 coefficients(uint unit) const {
    if (PLANES) {
      const uint tile = column / kTileColumns, in_tile = column % kTileColumns;
      const uint m = ((device const uint *)scales)[(ulong(tile) * kGroups + unit) * kTileColumns + in_tile];
      return float2(float(as_type<bfloat>(ushort(m & 0xFFFFu))), float(as_type<bfloat>(ushort(m >> 16))));
    }
    return float2(float(scales[ulong(column) * kGroups + unit]), float(biases[ulong(column) * kGroups + unit]));
  }
  // The column's planes, and step `first`'s payload (when the walk has a step) and coefficients.
  void begin(const device uint32_t *w_, const device S *scales_, const device S *biases_, uint column_, uint first,
             bool any) thread {
    w = w_;
    scales = scales_;
    biases = biases_;
    column = column_;
    packed = any ? codes(first) : uint4(0);
    meta = coefficients(first / kStepsPerGroup);
  }
  // Dequantizes the staged step into the lane's row of `stage`.
  void stage(threadgroup half *stage, uint lane) thread {
    if (PLANES) splash_dequant32(packed, meta, stage + lane * kStagedStep);
    else affine_dequant32(packed, meta, stage + lane * kStagedStep);
  }
  // After `step` is staged: loads `next`'s payload, and its coefficients when it enters a new group.
  void load(uint step, uint next) thread {
    packed = codes(next);
    const uint unit = next / kStepsPerGroup;
    if (unit != step / kStepsPerGroup) meta = coefficients(unit);
  }
};

// Splash's cross-threadgroup reduction of a K split (kernels/common/split_reduce.h): each partition publishes its
// fp32 partials through a device coherent pointer and arrives at its tile's counter; the last to arrive adds the
// partitions in split order, its own from registers, and returns the counter to zero. The result does not depend on
// scheduling, and no threadgroup waits for another.

// True in every thread of the threadgroup that arrives last.
inline bool split_arrive_last(device atomic_uint *counter, uint splits, uint thread_index,
                              threadgroup uint *arrival) {
  // Every writer publishes its partials before thread zero signals arrival. The device-scope fences pair through
  // the counter.
  threadgroup_barrier(mem_flags::mem_device);
  if (thread_index == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    *arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  return *arrival == splits - 1;
}

// The splits' partials added in split order: `own` for split `self`, `partial(s)` for every other split s.
template <typename T, typename Partial>
inline T split_sum(T own, uint self, uint splits, Partial partial) {
  T total = T(0);
  for (uint s = 0; s < splits; ++s) total += s == self ? own : partial(s);
  return total;
}

// Returns the tile's counter to zero; called by the reducing threadgroup.
inline void split_release(device atomic_uint *counter, uint thread_index) {
  if (thread_index == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);
}

// The walk over steps [sb, se) from `first`, wrapping (`gguf_staged_steps`, `gguf_staged_pair_steps`): `run(step,
// next, more, buffer)` for each, `more` while a step follows, the stages alternating per step walked.
template <typename Run> inline void staged_walk(uint sb, uint se, uint first, Run run) {
  uint walked = 0;
  for (uint pass = 0; pass < 2; ++pass) {
    const uint from = pass ? sb : first, to = pass ? first : se;
    for (uint step = from; step < to; ++step, ++walked)
      run(step, step + 1 < to ? step + 1 : sb, walked + 1 < se - sb, walked & 1);
  }
}

template <typename T, typename S, int G, int ROWS, bool PLANES>
[[kernel, max_total_threads_per_threadgroup(64)]] void affine_staged_decode(
    const device uint32_t *w [[buffer(0)]],
    const device S *scales [[buffer(1)]],
    const device S *biases [[buffer(2)]],
    const device T *x [[buffer(3)]],
    device T *y [[buffer(4)]],
    device coherent(device) float *partials [[buffer(5)]],
    device atomic_uint *counters [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 grid [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  using Ext = dextents<int32_t, 2>;
  constexpr int K = QMM_K, N = QMM_N, M = QMM_M;
  static_assert(G % kStagedStep == 0, "a group holds whole steps");
  const uint splits = grid.y;
  const uint per = uint(K) / kStagedStep / splits, sb = group.y * per, se = sb + per;
  const uint origin = group.x * kTileColumns + simd_group * kStagedColumns;

  threadgroup half stages[2][2][kStagedColumns * kStagedStep];
  threadgroup half *stage0 = stages[simd_group][0], *stage1 = stages[simd_group][1];
  tensor<device T, Ext, tensor_inline> a((device T *)x, Ext(K, M));
  tensor<threadgroup half, Ext, tensor_inline> bt0(stage0, Ext(kStagedStep, kStagedColumns));
  tensor<threadgroup half, Ext, tensor_inline> bt1(stage1, Ext(kStagedStep, kStagedColumns));
  auto b0 = bt0.template slice<kStagedStep, kStagedColumns>(0, 0);
  auto b1 = bt1.template slice<kStagedStep, kStagedColumns>(0, 0);
  constexpr auto descriptor = matmul2d_descriptor(ROWS, kStagedColumns, kStagedStep, false, true, false,
                                                  matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  auto a0 = a.template slice<kStagedStep, dynamic_extent>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

  const uint first = staged_first_step(group.x, sb, se, STAGED_SPREAD);
  StagedColumn<S, G, PLANES> column;
  column.begin(w, scales, biases, origin + lane, first, sb < se);
  staged_walk(sb, se, first, [&](uint step, uint next, bool more, uint buffer) __attribute__((always_inline)) {
    column.stage(buffer ? stage1 : stage0, lane);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (more) column.load(step, next);
    auto a_slice = a.template slice<kStagedStep, dynamic_extent>(step * kStagedStep, 0);
    if (buffer) operation.run(a_slice, b1, acc);
    else operation.run(a_slice, b0, acc);
  });

  if (splits == 1) {
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      const auto index = acc.get_multidimensional_index(i);
      const int row = index[1];
      if (row < M) y[ulong(row) * N + origin + index[0]] = T(acc[i]);
    }
    return;
  }
  // Split: publish the partials, arrive, and the last partition adds them in split order.
  const auto at = [&](uint split, uint row, uint col) { return (ulong(split) * M + row) * N + origin + col; };
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto index = acc.get_multidimensional_index(i);
    if (index[1] < M) partials[at(group.y, index[1], index[0])] = acc[i];
  }
  threadgroup uint arrival;
  const uint thread_index = simd_group * 32 + lane;
  device atomic_uint *counter = counters + group.x;
  if (!split_arrive_last(counter, splits, thread_index, &arrival)) return;
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto index = acc.get_multidimensional_index(i);
    const uint row = index[1], col = index[0];
    if (int(row) >= M) continue;
    const float own = acc[i];
    y[ulong(row) * N + origin + col] =
        T(split_sum(own, group.y, splits, [&](uint p) { return partials[at(p, row, col)]; }));
  }
  split_release(counter, thread_index);
}

// Splash's silu (kernels/common/activation.h): g / (1 + e^-g).
inline float splash_silu(float x) { return x / (1.0f + fast::exp2(-1.44269504089f * x)); }

template <typename T, typename S, int G, int ROWS, bool PLANES>
[[kernel, max_total_threads_per_threadgroup(32)]] void affine_staged_gate_up(
    const device uint32_t *gw [[buffer(0)]],
    const device S *gscales [[buffer(1)]],
    const device S *gbiases [[buffer(2)]],
    const device uint32_t *uw [[buffer(3)]],
    const device S *uscales [[buffer(4)]],
    const device S *ubiases [[buffer(5)]],
    const device T *x [[buffer(6)]],
    device T *y [[buffer(7)]],
    device coherent(device) float *partials [[buffer(8)]],
    device atomic_uint *counters [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 grid [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  using Ext = dextents<int32_t, 2>;
  constexpr int K = QMM_K, N = QMM_N, M = QMM_M;
  const uint splits = grid.y;
  const uint per = uint(K) / kStagedStep / splits, sb = group.y * per, se = sb + per;
  const uint origin = group.x * kStagedColumns;

  threadgroup half stage[2][kStagedColumns * kStagedStep];
  tensor<device T, Ext, tensor_inline> a((device T *)x, Ext(K, M));
  tensor<threadgroup half, Ext, tensor_inline> bt0(stage[0], Ext(kStagedStep, kStagedColumns));
  tensor<threadgroup half, Ext, tensor_inline> bt1(stage[1], Ext(kStagedStep, kStagedColumns));
  auto bg = bt0.template slice<kStagedStep, kStagedColumns>(0, 0);
  auto bu = bt1.template slice<kStagedStep, kStagedColumns>(0, 0);
  constexpr auto descriptor = matmul2d_descriptor(ROWS, kStagedColumns, kStagedStep, false, true, false,
                                                  matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  auto a0 = a.template slice<kStagedStep, dynamic_extent>(0, 0);
  auto gate = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(bg), float>();
  auto up = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(bg), float>();
#pragma unroll
  for (ushort i = 0; i < gate.get_capacity(); ++i) {
    gate[i] = 0.0f;
    up[i] = 0.0f;
  }
  // Each walks from the step its 64-column tile walks from in the two passes (`group.x / 2`), so at the same split
  // gate and up are the bits of the gate pass's and the up pass's sums.
  const uint first = staged_first_step(group.x / 2, sb, se, STAGED_SPREAD);
  StagedColumn<S, G, PLANES> g, u;
  g.begin(gw, gscales, gbiases, origin + lane, first, sb < se);
  u.begin(uw, uscales, ubiases, origin + lane, first, sb < se);
  staged_walk(sb, se, first, [&](uint step, uint next, bool more, uint) __attribute__((always_inline)) {
    g.stage(stage[0], lane);
    u.stage(stage[1], lane);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (more) {
      g.load(step, next);
      u.load(step, next);
    }
    auto a_slice = a.template slice<kStagedStep, dynamic_extent>(step * kStagedStep, 0);
    operation.run(a_slice, bg, gate);
    operation.run(a_slice, bu, up);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  });

  const auto value = [](float gv, float uv) { return T(float(bfloat(uv)) * splash_silu(float(bfloat(gv)))); };
  if (splits == 1) {
#pragma unroll
    for (ushort i = 0; i < gate.get_capacity(); ++i) {
      if (!gate.is_valid_element(i)) continue;
      const auto index = gate.get_multidimensional_index(i);
      const int row = index[1];
      if (row < M) y[ulong(row) * N + origin + index[0]] = value(gate[i], up[i]);
    }
    return;
  }
  const auto at = [&](uint split, uint row, uint col) {
    return (ulong(split) * M + row) * (2 * N) + origin + col;
  };
#pragma unroll
  for (ushort i = 0; i < gate.get_capacity(); ++i) {
    if (!gate.is_valid_element(i)) continue;
    const auto index = gate.get_multidimensional_index(i);
    if (index[1] >= M) continue;
    partials[at(group.y, index[1], index[0])] = gate[i];
    partials[at(group.y, index[1], index[0]) + N] = up[i];
  }
  threadgroup uint arrival;
  device atomic_uint *counter = counters + group.x;
  if (!split_arrive_last(counter, splits, lane, &arrival)) return;
#pragma unroll
  for (ushort i = 0; i < gate.get_capacity(); ++i) {
    if (!gate.is_valid_element(i)) continue;
    const auto index = gate.get_multidimensional_index(i);
    const uint row = index[1], col = index[0];
    if (int(row) >= M) continue;
    const float own_g = gate[i], own_u = up[i];
    const float gs = split_sum(own_g, group.y, splits, [&](uint p) { return partials[at(p, row, col)]; });
    const float us = split_sum(own_u, group.y, splits, [&](uint p) { return partials[at(p, row, col) + N]; });
    y[ulong(row) * N + origin + col] = value(gs, us);
  }
  split_release(counter, lane);
}

#define INST_STAGED(act_tag, act_type, scale_tag, scale_type, gs, rows)                                \
  SCRATCHY_KERNEL(affine_staged_decode_##act_tag##_s_##scale_tag##_gs_##gs##_b_4_m_##rows,             \
                  affine_staged_decode<act_type, scale_type, gs, rows, false>)                         \
  SCRATCHY_KERNEL(affine_staged_decode_planes_##act_tag##_s_##scale_tag##_gs_##gs##_b_4_m_##rows,      \
                  affine_staged_decode<act_type, scale_type, gs, rows, true>)                          \
  SCRATCHY_KERNEL(affine_staged_gate_up_##act_tag##_s_##scale_tag##_gs_##gs##_b_4_m_##rows,            \
                  affine_staged_gate_up<act_type, scale_type, gs, rows, false>)                        \
  SCRATCHY_KERNEL(affine_staged_gate_up_planes_##act_tag##_s_##scale_tag##_gs_##gs##_b_4_m_##rows,     \
                  affine_staged_gate_up<act_type, scale_type, gs, rows, true>)

#define INST_STAGED_ROWS(act_tag, act_type, scale_tag, scale_type, gs)                                 \
  INST_STAGED(act_tag, act_type, scale_tag, scale_type, gs, 8)                                         \
  INST_STAGED(act_tag, act_type, scale_tag, scale_type, gs, 16)

INST_STAGED_ROWS(bf16, bfloat, bf16, bfloat, 64)
INST_STAGED_ROWS(f16, half, f16, half, 64)
INST_STAGED_ROWS(bf16, bfloat, f16, half, 64)
INST_STAGED_ROWS(f16, half, bf16, bfloat, 64)
