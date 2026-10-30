// SPDX-License-Identifier: Apache-2.0
//
// Splash's DFlash2 draft kernels (~/git/splash @ 1482f6bb: runtime/metal/kernels/decode/draft.metal,
// runtime/metal/kernels/common/draft_context_kv.h, runtime/metal/kernels/prefill/draft_context.metal,
// runtime/metal/kernels/common/rms_inverse.h), with these changes, and no other:
//
//   - Splash's compiled-in geometry (`SPLASH_DRAFT_*`, runtime/metal/abi/DraftAttention.h and
//     ExecutionGeometry.h) is the model's, baked: every value below is a constant of the command.
//   - A lane's rings are one ring pool's: Splash binds each of its four lanes' rings apart
//     (`SPLASH_LANE_BINDING`); a batch here holds more lanes, so a lane's ring is the pool's at the
//     lane's state slot (`ring_slots`). The pool is `[slot][kv head][window][dim]` keys and
//     `[slot][kv head][dim][window]` values, a layer's own (the binding is the layer's).
//   - A row's RoPE pair is computed where it is used, from the row's position and the baked rotary
//     base, as Splash's table pass computes it (runtime/model/RopeScaling.hpp:46
//     `rotaryInverseFrequencies`, runtime/metal/kernels/shared/rope.metal
//     `rope_build_draft_range_tables`): fp32 throughout, where Splash first writes each row's
//     into a table.
//   - A lane's cache length is the position of its block's first row, its anchor: Splash passes
//     the same value (`item.logicalPosition`) in the kernel's parameters.
//
// Splash's kernels read their parameters' lane arrays; here each lane is a sequence of the step,
// in the order its rows are.

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include "baked.h"

using namespace metal;
using namespace mpp::tensor_ops;

// The block (SPLASH_DRAFT_QUERY_ROWS = SPLASH_TARGET_VERIFY_ROWS).
SCRATCHY_CONSTANT(uint, DRAFT_ROWS, 0);
// SPLASH_DRAFT_QUERY_HEADS, SPLASH_DRAFT_KV_HEADS, SPLASH_DRAFT_HEAD_DIMENSION.
SCRATCHY_CONSTANT(uint, DRAFT_QUERY_HEADS, 1);
SCRATCHY_CONSTANT(uint, DRAFT_KV_HEADS, 2);
SCRATCHY_CONSTANT(uint, DRAFT_HEAD_DIM, 3);
// SPLASH_DRAFT_SLIDING_WINDOW.
SCRATCHY_CONSTANT(uint, DRAFT_WINDOW, 4);
// SPLASH_DRAFT_ATTENTION_SPLITS.
SCRATCHY_CONSTANT(uint, DRAFT_SPLITS, 5);
// The hidden width a convolution runs at, and its channel group (SPLASH_DRAFT_CONVOLUTION_GROUP).
SCRATCHY_CONSTANT(uint, DRAFT_HIDDEN, 6);
SCRATCHY_CONSTANT(uint, DRAFT_CONV_GROUP, 7);
// A convolution's stage: 0 prepares, 1 finishes with the residual.
SCRATCHY_CONSTANT(uint, DRAFT_CONV_FINISH, 8);
// The epsilon of every RMS norm (SPLASH_RMS_EPSILON), and the attention's softmax scale.
SCRATCHY_CONSTANT(float, DRAFT_RMS_EPS, 9);
SCRATCHY_CONSTANT(float, DRAFT_SCALE, 10);
// The rotary base (the drafter's `rope_parameters.rope_theta`).
SCRATCHY_CONSTANT(float, DRAFT_ROPE_THETA, 11);

constant constexpr uint DRAFT_ATTENTION_WIDTH = DRAFT_QUERY_HEADS * DRAFT_HEAD_DIM;
constant constexpr uint DRAFT_KV_WIDTH = DRAFT_KV_HEADS * DRAFT_HEAD_DIM;
constant constexpr uint DRAFT_QKV_WIDTH = DRAFT_ATTENTION_WIDTH + 2 * DRAFT_KV_WIDTH;
constant constexpr uint DRAFT_ATTENTION_ROWS = DRAFT_QUERY_HEADS / DRAFT_KV_HEADS * DRAFT_ROWS;
constant constexpr uint DRAFT_ROPE_PAIRS = DRAFT_HEAD_DIM / 2;
constant constexpr uint DRAFT_CONV_TAPS = 2;
constant constexpr uint DRAFT_DYNAMIC_WIDTH =
    DRAFT_CONV_GROUP ? 2 * DRAFT_CONV_TAPS * (DRAFT_HIDDEN / DRAFT_CONV_GROUP) : 0;
// A lane's ring of one layer: its kv heads' window slots of one head dimension each.
constant constexpr ulong DRAFT_RING_LANE = ulong(DRAFT_KV_HEADS) * DRAFT_WINDOW * DRAFT_HEAD_DIM;
// One split's partial: its attention rows' accumulators, then their maxima and sums.
constant constexpr uint DRAFT_PARTIAL_FLOATS =
    DRAFT_ATTENTION_ROWS * DRAFT_HEAD_DIM + 2 * DRAFT_ATTENTION_ROWS;

// runtime/metal/kernels/common/rms_inverse.h `rms_inverse_of_sums`, at 8 simdgroups.
inline float draft_rms_inverse_of_sums(float sum, uint width, threadgroup float *reductions,
                                       uint thread_index, uint lane, uint simd_group) {
#pragma clang fp reassociate(off)
  sum = simd_sum(sum);
  if (lane == 0)
    reductions[simd_group] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i)
      total += reductions[i];
    reductions[0] = rsqrt(total / width + DRAFT_RMS_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return reductions[0];
}

// The cosine and sine of RoPE pair `pair` at `position`: the pair's inverse frequency
// θ^(−pair / pairs) (RopeScaling.hpp:46), its angle at the position, then both
// (rope.metal:32-35, 51-54), in fp32.
inline float2 draft_rope(uint position, uint pair) {
  const float inverse_frequency = pow(DRAFT_ROPE_THETA, -float(pair) / float(DRAFT_ROPE_PAIRS));
  const float angle = float(position) * inverse_frequency;
  return float2(cos(angle), sin(angle));
}

// ── draft_conv (draft.metal draft_conv_phase / draft_conv) ─────────────────────────────────────

template <typename T>
inline void draft_conv_phase(device const T *input, device const T *dynamic,
                             device const T *base, device const T *residual,
                             device T *output, bool finish, uint group, uint thread_index) {
  constexpr uint Rows = DRAFT_ROWS, Taps = DRAFT_CONV_TAPS, hidden = DRAFT_HIDDEN;
  const uint conv_groups = hidden / DRAFT_CONV_GROUP;
  const uint dynamic_width = DRAFT_DYNAMIC_WIDTH;
  const uint element = group * 256 + thread_index;
  if (element >= Rows * hidden)
    return;
  uint row = element / hidden;
  uint channel = element % hidden;
  uint conv_group = channel / DRAFT_CONV_GROUP;
  uint kind = finish ? 1 : 0;
  float value =
      float(input[element]) *
      (float(base[(kind * Taps) * hidden + channel]) +
       float(dynamic[row * dynamic_width + (kind * Taps) * conv_groups + conv_group]));
  if (row > 0) {
    value += float(input[(row - 1) * hidden + channel]) *
             (float(base[(kind * Taps + 1) * hidden + channel]) +
              float(dynamic[row * dynamic_width + (kind * Taps + 1) * conv_groups + conv_group]));
  }
  if (finish)
    value += float(residual[element]);
  output[element] = T(value);
}

template <typename T>
[[kernel]] void draft_conv(device const T *input [[buffer(0)]],
                           device const T *dynamic [[buffer(1)]],
                           device const T *base [[buffer(2)]],
                           device const T *residual [[buffer(3)]],
                           device T *output [[buffer(4)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr ulong Rows = DRAFT_ROWS;
  const ulong hidden = DRAFT_HIDDEN, dynamic_width = DRAFT_DYNAMIC_WIDTH;
  uint batch = group.y;
  draft_conv_phase(input + batch * Rows * hidden, dynamic + batch * Rows * dynamic_width, base,
                   residual + batch * Rows * hidden, output + batch * Rows * hidden,
                   DRAFT_CONV_FINISH != 0, group.x, thread_index);
}

// ── draft_attention_qkv (draft.metal draft_qkv_prepare_phase / draft_attention_qkv) ────────────

template <typename T>
inline void draft_qkv_prepare_phase(device const T *proposal_qkv, device T *queries,
                                    device const T *q_norm, device const T *k_norm,
                                    device const uint *positions,
                                    device T *query_keys, device T *query_values,
                                    threadgroup float *reductions, threadgroup T *head, uint group,
                                    uint thread_index, uint lane, uint simd_group) {
  constexpr uint Rows = DRAFT_ROWS, QHeads = DRAFT_QUERY_HEADS, KVHeads = DRAFT_KV_HEADS,
                 HeadDim = DRAFT_HEAD_DIM;
  constexpr uint QWidth = DRAFT_ATTENTION_WIDTH, KWidth = DRAFT_KV_WIDTH;
  constexpr uint PackedWidth = DRAFT_QKV_WIDTH;
  constexpr uint QueryTasks = Rows * QHeads;
  constexpr uint Tasks = QueryTasks + Rows * KVHeads;

  const uint element = group * 256 + thread_index;
  if (element < Rows * KVHeads * HeadDim) {
    uint row = element / (KVHeads * HeadDim);
    uint remainder = element % (KVHeads * HeadDim);
    uint attention_head = remainder / HeadDim;
    uint dim = remainder % HeadDim;
    query_values[(attention_head * HeadDim + dim) * Rows + row] =
        proposal_qkv[row * PackedWidth + QWidth + KWidth + remainder];
  }

  // The group's normalization task; the condition is uniform across the threadgroup, so every
  // thread meets its barriers.
  if (group < Tasks) {
    const uint task = group;
    bool query = task < QueryTasks;
    uint local_task = query ? task : task - QueryTasks;
    uint heads = query ? QHeads : KVHeads;
    uint row = local_task / heads;
    uint attention_head = local_task % heads;
    device const T *source =
        query ? proposal_qkv + row * PackedWidth + attention_head * HeadDim
              : proposal_qkv + row * PackedWidth + QWidth + attention_head * HeadDim;
    device const T *weight = query ? q_norm : k_norm;
    device T *destination =
        (query ? queries : query_keys) + (attention_head * Rows + row) * HeadDim;

    float value = thread_index < HeadDim ? float(source[thread_index]) : 0.0f;
    const float inverse = draft_rms_inverse_of_sums(value * value, HeadDim, reductions,
                                                    thread_index, lane, simd_group);
    if (thread_index < HeadDim)
      head[thread_index] = T(value * inverse * float(weight[thread_index]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < HeadDim / 2) {
      float first = float(head[thread_index]);
      float second = float(head[thread_index + HeadDim / 2]);
      const float2 cs = draft_rope(positions[row], thread_index);
      destination[thread_index] = T(first * cs.x - second * cs.y);
      destination[thread_index + HeadDim / 2] = T(second * cs.x + first * cs.y);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Grid {groups, lanes}: `queries` holds the grouped queries, `query_keys` / `query_values` each
// lane's block keys and values, all in the op scratch.
template <typename T>
[[kernel]] void draft_attention_qkv(device const T *proposal_qkv [[buffer(0)]],
                                    device T *queries [[buffer(1)]],
                                    device const T *q_norm [[buffer(2)]],
                                    device const T *k_norm [[buffer(3)]],
                                    device const uint *positions [[buffer(5)]],
                                    device T *query_keys [[buffer(6)]],
                                    device T *query_values [[buffer(7)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint thread_index [[thread_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr ulong Rows = DRAFT_ROWS, Packed = DRAFT_QKV_WIDTH, Attention = DRAFT_ATTENTION_WIDTH,
                  KVHeads = DRAFT_KV_HEADS, HeadDim = DRAFT_HEAD_DIM;
  uint batch = group.y;
  threadgroup float reductions[8];
  threadgroup T head[DRAFT_HEAD_DIM];
  draft_qkv_prepare_phase(proposal_qkv + batch * Rows * Packed, queries + batch * Rows * Attention,
                          q_norm, k_norm, positions + batch * Rows,
                          query_keys + batch * KVHeads * Rows * HeadDim,
                          query_values + batch * KVHeads * HeadDim * Rows, reductions, head,
                          group.x, thread_index, lane, simd_group);
}

// ── draft_attention_bf16_split (draft.metal draft_attention_split_phase) ───────────────────────
//
// One split of the attention of one KV head over the 32 query rows of one lane (four query heads x
// eight proposals). The live window occupies one or two physical ranges of the ring; the
// ring-aligned tiles touching them are dealt round-robin to the splits, so every context length
// spreads over all splits and no tile reads past the ring. Each split runs an online softmax over
// its tiles, the last split adds the eight current rows, and the unnormalized fp32 accumulator
// with its row maxima and sums lands in `partial` for the fixed-order reduce.
template <typename T>
inline void draft_attention_split_phase(
    device T *queries, device T *keys, device T *values, device T *query_keys,
    device T *query_values, device float *partial, uint value_stride, uint cache_length,
    uint split, threadgroup float *score_storage, threadgroup float *row_max,
    threadgroup float *row_sum, threadgroup float *previous_scale, uint thread_index, uint lane,
    uint simd_group) {
  constexpr ushort M = DRAFT_ATTENTION_ROWS, N = 128, D = DRAFT_HEAD_DIM, TileK = 64;
  constexpr uint Rows = DRAFT_ROWS, Window = DRAFT_WINDOW, Splits = DRAFT_SPLITS;
  uint common_start = cache_length >= Window - 1 ? cache_length - (Window - 1) : 0;
  uint old_count = cache_length - common_start;
  uint physical_start = common_start % Window;
  // Live tiles: [0, wrapped) when the window wraps, then [resume, tail_end); a tile the wrap and
  // the tail both touch is visited once.
  uint physical_end = physical_start + old_count;
  uint wrapped = physical_end > Window ? (physical_end - Window + N - 1) / N : 0;
  uint tail_end = (min(physical_end, Window) + N - 1) / N;
  uint resume = max(physical_start / N, wrapped);
  uint live_tiles = wrapped + tail_end - resume;
  // A split with no tile that is not the last one has nothing to add: it leaves only row maxima
  // of -inf, which the reduce skips, instead of loading Q and storing a zero accumulator at short
  // contexts.
  if (split >= live_tiles && split + 1 != Splits) {
    if (thread_index < M)
      partial[M * D + thread_index] = -INFINITY;
    return;
  }
  auto qt = tensor(queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  constexpr auto qk_descriptor = matmul2d_descriptor(M, N, TileK, false, true, false);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  auto q0 = qt.template slice<TileK, M>(0, 0);
  auto pt = tensor(score_storage, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto p0 = pt.template slice<TileK, M>(0, 0);
  auto first_v = tensor(values, dextents<int, 2>{N, D}, array<int, 2>{1, int(value_stride)});
  auto first_v0 = first_v.template slice<TileK, D>(0, 0);
  constexpr auto pv_descriptor = matmul2d_descriptor(M, D, TileK, false, true, false);
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto running = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(first_v0),
                                                                float>();
  for (ushort i = 0; i < running.get_capacity(); ++i)
    running[i] = 0.0f;
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint tile = split; tile < live_tiles; tile += Splits) {
    uint slot = (tile < wrapped ? tile : resume + tile - wrapped) * N;
    auto kt = tensor(keys + slot * D, dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto k0 = kt.template slice<TileK, N>(0, 0);
    auto scores =
        qk.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
    for (ushort i = 0; i < scores.get_capacity(); ++i)
      scores[i] = 0.0f;
    for (ushort chunk = 0; chunk < D / TileK; ++chunk) {
      auto qs = qt.template slice<TileK, M>(chunk * TileK, 0);
      auto ks = kt.template slice<TileK, N>(chunk * TileK, 0);
      auto partial_scores =
          qk.template get_destination_cooperative_tensor<decltype(qs), decltype(ks), float>();
      qk.run(qs, ks, partial_scores);
      for (ushort i = 0; i < scores.get_capacity(); ++i) {
        scores[i] += partial_scores[i];
      }
    }
    scores.store(pt.template slice<N, M>(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort batch = 0; batch < M; batch += 8) {
      ushort matrix_row = batch + simd_group;
      uint proposal_row = matrix_row % Rows;
      uint query_position = cache_length + proposal_row;
      uint row_start = query_position >= Window - 1 ? query_position - (Window - 1) : 0;
      uint hidden_prefix = row_start - common_start;
      float local_scores[4];
      float tile_max = -INFINITY;
      for (ushort i = 0; i < 4; ++i) {
        uint column = lane + i * 32;
        uint key = slot + column;
        uint logical_key =
            key >= physical_start ? key - physical_start : key + Window - physical_start;
        bool valid = logical_key < old_count && logical_key >= hidden_prefix;
        float score = score_storage[matrix_row * N + column] * DRAFT_SCALE;
        local_scores[i] = valid ? score : -INFINITY;
        tile_max = max(tile_max, local_scores[i]);
      }
      tile_max = simd_max(tile_max);
      float next_max = max(row_max[matrix_row], tile_max);
      // A row with no visible key so far keeps its running state instead of scaling it by
      // exp(-inf - -inf).
      bool empty = next_max == -INFINITY;
      float scale = empty ? 1.0f : fast::exp(row_max[matrix_row] - next_max);
      float tile_sum = 0.0f;
      for (ushort i = 0; i < 4; ++i) {
        local_scores[i] = empty ? 0.0f : fast::exp(local_scores[i] - next_max);
        tile_sum += local_scores[i];
      }
      tile_sum = simd_sum(tile_sum);
      if (lane == 0) {
        previous_scale[matrix_row] = scale;
        row_sum[matrix_row] = row_sum[matrix_row] * scale + tile_sum;
        row_max[matrix_row] = next_max;
      }
      for (ushort i = 0; i < 4; ++i) {
        uint column = lane + i * 32;
        score_storage[matrix_row * N + column] = local_scores[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto vt = tensor(values + slot, dextents<int, 2>{N, D}, array<int, 2>{1, int(value_stride)});
    auto v0 = vt.template slice<TileK, D>(0, 0);
    auto partial_output =
        pv.template get_destination_cooperative_tensor<decltype(p0), decltype(v0), float>();
    for (ushort i = 0; i < partial_output.get_capacity(); ++i) {
      partial_output[i] = 0.0f;
    }
    for (ushort chunk = 0; chunk < N / TileK; ++chunk) {
      auto ps = pt.template slice<TileK, M>(chunk * TileK, 0);
      auto vs = vt.template slice<TileK, D>(chunk * TileK, 0);
      auto partial_values =
          pv.template get_destination_cooperative_tensor<decltype(ps), decltype(vs), float>();
      pv.run(ps, vs, partial_values);
      for (ushort i = 0; i < partial_output.get_capacity(); ++i) {
        partial_output[i] += partial_values[i];
      }
    }
    for (ushort i = 0; i < running.get_capacity(); ++i) {
      auto index = running.get_multidimensional_index(i);
      running[i] = running[i] * previous_scale[index[1]] + partial_output[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (split + 1 == Splits) {
    // Persistent query K/V contains exactly eight rows. The K=16 padding required by the MPP value
    // product exists only in threadgroup memory.
    constexpr ushort CurrentN = Rows;
    constexpr ushort CurrentValueTile = 16;
    threadgroup float *current_raw_scores = score_storage + M * CurrentValueTile;
    threadgroup T *current_values = reinterpret_cast<threadgroup T *>(
        score_storage + M * CurrentValueTile + M * CurrentN);
    auto current_scores =
        tensor(current_raw_scores, dextents<int, 2>{CurrentN, M}, array<int, 2>{1, CurrentN});
    constexpr auto current_qk_descriptor =
        matmul2d_descriptor(M, CurrentN, TileK, false, true, false);
    matmul2d<current_qk_descriptor, execution_simdgroups<8>> current_qk;
    auto current_kt = tensor(query_keys, dextents<int, 2>{D, CurrentN}, array<int, 2>{1, D});
    auto current_k0 = current_kt.template slice<TileK, CurrentN>(0, 0);
    auto current = current_qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(current_k0), float>();
    for (ushort i = 0; i < current.get_capacity(); ++i)
      current[i] = 0.0f;
    for (ushort chunk = 0; chunk < D / TileK; ++chunk) {
      auto qs = qt.template slice<TileK, M>(chunk * TileK, 0);
      auto ks = current_kt.template slice<TileK, CurrentN>(chunk * TileK, 0);
      auto partial_scores = current_qk.template get_destination_cooperative_tensor<
          decltype(qs), decltype(ks), float>();
      current_qk.run(qs, ks, partial_scores);
      for (ushort i = 0; i < current.get_capacity(); ++i) {
        current[i] += partial_scores[i];
      }
    }
    current.store(current_scores.template slice<CurrentN, M>(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort batch = 0; batch < M; batch += 8) {
      ushort matrix_row = batch + simd_group;
      float score = lane < CurrentN
                        ? current_raw_scores[matrix_row * CurrentN + lane] * DRAFT_SCALE
                        : -INFINITY;
      float tile_max = simd_max(score);
      float next_max = max(row_max[matrix_row], tile_max);
      float scale = fast::exp(row_max[matrix_row] - next_max);
      float probability = fast::exp(score - next_max);
      float tile_sum = simd_sum(probability);
      if (lane == 0) {
        previous_scale[matrix_row] = scale;
        row_sum[matrix_row] = row_sum[matrix_row] * scale + tile_sum;
        row_max[matrix_row] = next_max;
      }
      if (lane < CurrentValueTile) {
        score_storage[matrix_row * CurrentValueTile + lane] = lane < CurrentN ? probability : 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto current_probabilities = tensor(score_storage, dextents<int, 2>{CurrentValueTile, M},
                                        array<int, 2>{1, CurrentValueTile});
    auto current_p0 = current_probabilities.template slice<CurrentValueTile, M>(0, 0);
    for (uint index = thread_index; index < D * CurrentValueTile; index += 256) {
      uint dimension = index / CurrentValueTile;
      uint token = index % CurrentValueTile;
      current_values[index] = token < CurrentN ? query_values[dimension * CurrentN + token] : T(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto current_vt = tensor(current_values, dextents<int, 2>{CurrentValueTile, D},
                             array<int, 2>{1, int(CurrentValueTile)});
    auto current_v0 = current_vt.template slice<CurrentValueTile, D>(0, 0);
    constexpr auto current_pv_descriptor =
        matmul2d_descriptor(M, D, CurrentValueTile, false, true, false);
    matmul2d<current_pv_descriptor, execution_simdgroups<8>> current_pv;
    auto current_output = current_pv.template get_destination_cooperative_tensor<
        decltype(current_p0), decltype(current_v0), float>();
    current_pv.run(current_p0, current_v0, current_output);
    for (ushort i = 0; i < running.get_capacity(); ++i) {
      auto index = running.get_multidimensional_index(i);
      running[i] = running[i] * previous_scale[index[1]] + current_output[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  auto accumulator = tensor(partial, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  running.store(accumulator.template slice<D, M>(0, 0));
  if (thread_index < M) {
    partial[M * D + thread_index] = row_max[thread_index];
    partial[M * D + M + thread_index] = row_sum[thread_index];
  }
}

// Grid {kv heads, lanes, splits}: every split streams its share of the live ring tiles of its
// head and the last one adds the eight current rows. Partials sit in the op scratch, one
// 32 x 130 fp32 block per (lane, head, split).
template <typename T>
[[kernel]] void draft_attention_split(device T *queries [[buffer(0)]],
                                      device T *ring_keys [[buffer(1)]],
                                      device T *ring_values [[buffer(2)]],
                                      device const uint *ring_slots [[buffer(3)]],
                                      device const uint *positions [[buffer(4)]],
                                      device T *query_keys [[buffer(5)]],
                                      device T *query_values [[buffer(6)]],
                                      device float *partials [[buffer(7)]],
                                      uint3 group [[threadgroup_position_in_grid]],
                                      uint thread_index [[thread_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint AttentionM = DRAFT_ATTENTION_ROWS, AttentionN = 128, KVHeads = DRAFT_KV_HEADS;
  constexpr ulong Rows = DRAFT_ROWS, Attention = DRAFT_ATTENTION_WIDTH, HeadDim = DRAFT_HEAD_DIM,
                  Window = DRAFT_WINDOW;
  uint batch = group.y;
  const ulong ring = ulong(ring_slots[batch]) * DRAFT_RING_LANE;
  device float *partial =
      partials + ((batch * KVHeads + group.x) * DRAFT_SPLITS + group.z) * DRAFT_PARTIAL_FLOATS;
  threadgroup float workspace[AttentionM * AttentionN + 3 * AttentionM];
  draft_attention_split_phase(
      queries + batch * Rows * Attention + group.x * AttentionM * HeadDim,
      ring_keys + ring + group.x * Window * HeadDim, ring_values + ring + group.x * Window * HeadDim,
      query_keys + batch * KVHeads * Rows * HeadDim + group.x * Rows * HeadDim,
      query_values + batch * KVHeads * HeadDim * Rows + group.x * Rows * HeadDim, partial,
      uint(Window), positions[batch * Rows], group.z, workspace, workspace + AttentionM * AttentionN,
      workspace + AttentionM * AttentionN + AttentionM,
      workspace + AttentionM * AttentionN + 2 * AttentionM, thread_index, lane, simd_group);
}

// ── draft_attention_bf16_reduce (draft.metal draft_attention_reduce_phase) ─────────────────────
//
// Fixed-order combine of one (lane, head) pair's split partials into the normalized rows, so the
// result does not depend on which split finished first or on how many lanes ran.
template <typename T>
inline void draft_attention_reduce_phase(device const float *partials, device T *output,
                                         uint thread_index) {
  constexpr uint M = DRAFT_ATTENTION_ROWS, D = DRAFT_HEAD_DIM, Stride = M * D + 2 * M,
                 Splits = DRAFT_SPLITS;
  constexpr uint Chunk = D / 8;
  uint row = thread_index / 8;
  uint dim = (thread_index % 8) * Chunk;
  float maximum = -INFINITY;
  for (uint split = 0; split < Splits; ++split)
    maximum = max(maximum, partials[split * Stride + M * D + row]);
  float total = 0.0f;
  float accumulated[Chunk];
  for (uint i = 0; i < Chunk; ++i)
    accumulated[i] = 0.0f;
  for (uint split = 0; split < Splits; ++split) {
    device const float *partial = partials + split * Stride;
    float split_max = partial[M * D + row];
    // A split with no visible key contributes nothing rather than exp(-inf - max) times an
    // all-zero accumulator.
    if (split_max == -INFINITY)
      continue;
    float scale = fast::exp(split_max - maximum);
    total += scale * partial[M * D + M + row];
    for (uint i = 0; i < Chunk; ++i)
      accumulated[i] += scale * partial[row * D + dim + i];
  }
  for (uint i = 0; i < Chunk; ++i)
    output[row * D + dim + i] = T(accumulated[i] / total);
}

template <typename T>
[[kernel]] void draft_attention_reduce(device T *queries [[buffer(0)]],
                                       device const float *partials [[buffer(1)]],
                                       uint2 group [[threadgroup_position_in_grid]],
                                       uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr uint AttentionM = DRAFT_ATTENTION_ROWS, KVHeads = DRAFT_KV_HEADS;
  constexpr ulong Rows = DRAFT_ROWS, Attention = DRAFT_ATTENTION_WIDTH, HeadDim = DRAFT_HEAD_DIM;
  uint batch = group.y;
  draft_attention_reduce_phase(
      partials + (batch * KVHeads + group.x) * DRAFT_SPLITS * DRAFT_PARTIAL_FLOATS,
      queries + batch * Rows * Attention + group.x * AttentionM * HeadDim, thread_index);
}

// ── draft_attention_reorder (draft.metal draft_attention_reorder_phase) ────────────────────────

template <typename T>
[[kernel]] void draft_attention_reorder(device const T *grouped [[buffer(0)]],
                                        device T *row_major [[buffer(1)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr uint Rows = DRAFT_ROWS, QHeads = DRAFT_QUERY_HEADS, HeadDim = DRAFT_HEAD_DIM;
  constexpr ulong Attention = DRAFT_ATTENTION_WIDTH;
  uint batch = group.y;
  device const T *lane_grouped = grouped + batch * Rows * Attention;
  device T *lane_row_major = row_major + batch * Rows * Attention;
  const uint element = group.x * 256 + thread_index;
  if (element >= Rows * QHeads * HeadDim)
    return;
  uint row = element / (QHeads * HeadDim);
  uint remainder = element % (QHeads * HeadDim);
  uint query_head = remainder / HeadDim;
  uint dim = remainder % HeadDim;
  uint grouped_index = (query_head * Rows + row) * HeadDim + dim;
  lane_row_major[element] = lane_grouped[grouped_index];
}

// ── The context writers (common/draft_context_kv.h draft_context_kv_phase) ─────────────────────
//
// One KV head of one context row: a row of context_kv holds the row's keys, then its values
// (KWidth each). The keys are normalized and rotated into the ring slot of the row's position; the
// values are copied there.
template <typename T>
inline void draft_context_kv_phase(device const T *context_kv, device const T *k_norm,
                                   device T *keys, device T *values,
                                   uint position, uint head_index, uint thread_index, uint lane,
                                   uint simd_group, threadgroup float *reductions,
                                   threadgroup T *normalized) {
  constexpr uint HeadDim = DRAFT_HEAD_DIM, Window = DRAFT_WINDOW;
  constexpr uint KWidth = DRAFT_KV_WIDTH;
  uint slot = position % Window;
  device const T *source = context_kv + head_index * HeadDim;
  device T *key = keys + (ulong(head_index) * Window + slot) * HeadDim;
  device T *value = values + ulong(head_index) * HeadDim * Window + slot;

  float element = thread_index < HeadDim ? float(source[thread_index]) : 0.0f;
  const float inverse = draft_rms_inverse_of_sums(element * element, HeadDim, reductions,
                                                  thread_index, lane, simd_group);
  if (thread_index < HeadDim) {
    normalized[thread_index] = T(element * inverse * float(k_norm[thread_index]));
    value[ulong(thread_index) * Window] = source[KWidth + thread_index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < HeadDim / 2) {
    float first = float(normalized[thread_index]);
    float second = float(normalized[thread_index + HeadDim / 2]);
    const float2 cs = draft_rope(position, thread_index);
    key[thread_index] = T(first * cs.x - second * cs.y);
    key[thread_index + HeadDim / 2] = T(second * cs.x + first * cs.y);
  }
}

// A cycle's commit (draft.metal draft_context_kv_commit): each lane's verify rows up to the ones
// it retained (`retained`), one threadgroup per (row, kv head). A lane's verify rows are those of
// its block its verify budget chose (`cu_seqlens`), where Splash's are its block's every row.
template <typename T>
[[kernel]] void draft_context_kv_commit(device const T *context_kv [[buffer(0)]],
                                        device const T *k_norm [[buffer(1)]],
                                        device const uint *positions [[buffer(3)]],
                                        device T *ring_keys [[buffer(4)]],
                                        device T *ring_values [[buffer(5)]],
                                        device const uint *ring_slots [[buffer(6)]],
                                        device const uint *retained [[buffer(7)]],
                                        device const uint *cu_seqlens [[buffer(8)]],
                                        device const uint *num_tokens [[buffer(9)]],
                                        uint group [[threadgroup_position_in_grid]],
                                        uint thread_index [[thread_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint KVHeads = DRAFT_KV_HEADS;
  constexpr uint RowWidth = 2 * DRAFT_KV_WIDTH;
  const uint r = group / KVHeads;
  if (r >= num_tokens[0])
    return;
  uint batch = 0;
  while (cu_seqlens[batch + 1] <= r)
    ++batch;
  const uint row = r - cu_seqlens[batch];
  if (row >= retained[batch])
    return;
  const ulong ring = ulong(ring_slots[batch]) * DRAFT_RING_LANE;
  threadgroup float reductions[8];
  threadgroup T normalized[DRAFT_HEAD_DIM];
  draft_context_kv_phase(context_kv + ulong(r) * RowWidth, k_norm, ring_keys + ring,
                         ring_values + ring, positions[cu_seqlens[batch]] + row, group % KVHeads,
                         thread_index, lane, simd_group, reductions, normalized);
}

// A prompt's commit (draft_context.metal prefill_draft_context_kv): every row of the step, each at
// its position in its sequence's ring, one threadgroup per (row, kv head). A step's rows of one
// sequence are at most a window: no two of them share a slot (Splash's DraftContextPlan).
template <typename T>
[[kernel]] void prefill_draft_context_kv(device const T *context_kv [[buffer(0)]],
                                         device const T *k_norm [[buffer(1)]],
                                         device const uint *positions [[buffer(3)]],
                                         device T *ring_keys [[buffer(4)]],
                                         device T *ring_values [[buffer(5)]],
                                         device const uint *ring_slots [[buffer(6)]],
                                         device const uint *cu_seqlens [[buffer(7)]],
                                         device const uint *num_tokens [[buffer(8)]],
                                         uint task [[threadgroup_position_in_grid]],
                                         uint thread_index [[thread_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]],
                                         uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint KVHeads = DRAFT_KV_HEADS;
  constexpr uint RowWidth = 2 * DRAFT_KV_WIDTH;
  const uint row = task / KVHeads;
  if (row >= num_tokens[0])
    return;
  uint sequence = 0;
  while (cu_seqlens[sequence + 1] <= row)
    ++sequence;
  const ulong ring = ulong(ring_slots[sequence]) * DRAFT_RING_LANE;
  threadgroup float reductions[8];
  threadgroup T normalized[DRAFT_HEAD_DIM];
  draft_context_kv_phase(context_kv + ulong(row) * RowWidth, k_norm, ring_keys + ring,
                         ring_values + ring, positions[row], task % KVHeads, thread_index, lane,
                         simd_group, reductions, normalized);
}

#define INST_DRAFT_ACT(tag, T)                                                     \
  SCRATCHY_KERNEL(draft_conv_##tag, draft_conv<T>)                                 \
  SCRATCHY_KERNEL(draft_attention_split_##tag, draft_attention_split<T>)           \
  SCRATCHY_KERNEL(draft_attention_reduce_##tag, draft_attention_reduce<T>)         \
  SCRATCHY_KERNEL(draft_attention_reorder_##tag, draft_attention_reorder<T>)       \
  SCRATCHY_KERNEL(draft_attention_qkv_##tag, draft_attention_qkv<T>)               \
  SCRATCHY_KERNEL(draft_context_kv_commit_##tag, draft_context_kv_commit<T>)       \
  SCRATCHY_KERNEL(prefill_draft_context_kv_##tag, prefill_draft_context_kv<T>)

INST_DRAFT_ACT(f16, half)
INST_DRAFT_ACT(bf16, bfloat)
