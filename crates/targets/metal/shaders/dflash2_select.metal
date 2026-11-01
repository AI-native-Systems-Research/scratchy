// SPDX-License-Identifier: Apache-2.0
//
// Splash's DFlash2 draft selector and verify rows (~/git/splash @ 1482f6bb:
// runtime/metal/kernels/decode/sampling.metal `draft_select_top16_sharded`, `draft_select_edges`,
// `draft_select_dflash`, `draft_verify_rows`, and their helpers), with these changes, and no
// other:
//
//   - Splash's compiled-in geometry (`SPLASH_DRAFT_*`, runtime/metal/abi/ExecutionGeometry.h) and
//     the vocabulary its parameters carry are the model's, baked.
//   - A lane's anchor is its block's first input token, which Splash passes as `params.anchor`.
//   - A lane's sampling policy (temperature, whether it samples) and its verify rows' maximum are
//     the step's `lane_params` (dflash2_cycle.h), which Splash passes in its parameters.
//   - The logits are the model's activation type; Splash's are f32.
//   - The verify-rows model (every row, or the budget with its acceptance model) is baked, and a
//     budget's costs are Splash's seed (runtime/model/VerifyBudget.cpp `VerifyBudget::seed`),
//     where Splash passes the table of the line it refits online once it has timed its cycles.

#include <metal_stdlib>
#include "baked.h"
#include "dflash2_cycle.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, SELECT_VOCAB, 0);
// SPLASH_DRAFT_QUERY_ROWS (the block), SPLASH_DRAFT_PROPOSAL_TOKENS (its proposals).
SCRATCHY_CONSTANT(uint, SELECT_ROWS, 1);
SCRATCHY_CONSTANT(uint, SELECT_POSITIONS, 2);
// SPLASH_DRAFT_CANDIDATES, SPLASH_DRAFT_SELECTOR_RANK, SPLASH_DRAFT_SAMPLING_SHARDS.
SCRATCHY_CONSTANT(uint, SELECT_CANDIDATES, 3);
SCRATCHY_CONSTANT(uint, SELECT_RANK, 4);
SCRATCHY_CONSTANT(uint, SELECT_SHARDS, 5);
// The verify-rows model (VerifyRowsParams): logit(p) = slope · logit(q) + intercept[position].
SCRATCHY_CONSTANT(float, SELECT_SLOPE, 6);
SCRATCHY_CONSTANT(float, SELECT_INTERCEPT_0, 7);
SCRATCHY_CONSTANT(float, SELECT_INTERCEPT_1, 8);
SCRATCHY_CONSTANT(float, SELECT_INTERCEPT_2, 9);
SCRATCHY_CONSTANT(float, SELECT_INTERCEPT_3, 10);
SCRATCHY_CONSTANT(float, SELECT_INTERCEPT_4, 11);
SCRATCHY_CONSTANT(float, SELECT_INTERCEPT_5, 12);
SCRATCHY_CONSTANT(float, SELECT_INTERCEPT_6, 13);
// The most lanes a step of the bucket holds.
SCRATCHY_CONSTANT(uint, SELECT_LANES, 14);
// Each lane verifies its maximum (VerifyRowsParams `every_row`): the budget's slots go unread.
SCRATCHY_CONSTANT(bool, SELECT_EVERY_ROW, 15);
// A cycle's cost at 1, 2, 4 and 8 rows a lane, for one to four lanes (VerifyBudget.cpp `kSeed`):
// SELECT_SEED_<lanes - 1>_<knot>.
SCRATCHY_CONSTANT(float, SELECT_SEED_0_0, 16);
SCRATCHY_CONSTANT(float, SELECT_SEED_0_1, 17);
SCRATCHY_CONSTANT(float, SELECT_SEED_0_2, 18);
SCRATCHY_CONSTANT(float, SELECT_SEED_0_3, 19);
SCRATCHY_CONSTANT(float, SELECT_SEED_1_0, 20);
SCRATCHY_CONSTANT(float, SELECT_SEED_1_1, 21);
SCRATCHY_CONSTANT(float, SELECT_SEED_1_2, 22);
SCRATCHY_CONSTANT(float, SELECT_SEED_1_3, 23);
SCRATCHY_CONSTANT(float, SELECT_SEED_2_0, 24);
SCRATCHY_CONSTANT(float, SELECT_SEED_2_1, 25);
SCRATCHY_CONSTANT(float, SELECT_SEED_2_2, 26);
SCRATCHY_CONSTANT(float, SELECT_SEED_2_3, 27);
SCRATCHY_CONSTANT(float, SELECT_SEED_3_0, 28);
SCRATCHY_CONSTANT(float, SELECT_SEED_3_1, 29);
SCRATCHY_CONSTANT(float, SELECT_SEED_3_2, 30);
SCRATCHY_CONSTANT(float, SELECT_SEED_3_3, 31);

constant constexpr uint K = SELECT_CANDIDATES;

inline float select_intercept(uint position) {
  constexpr float intercepts[7] = {SELECT_INTERCEPT_0, SELECT_INTERCEPT_1, SELECT_INTERCEPT_2,
                                   SELECT_INTERCEPT_3, SELECT_INTERCEPT_4, SELECT_INTERCEPT_5,
                                   SELECT_INTERCEPT_6};
  return intercepts[position];
}

// What a cycle of `lanes` lanes verifying `rows` rows in all costs (VerifyBudget::seed, linear
// between its knots); one of more lanes than the seed prices costs what one of four does.
inline float select_cost(uint lanes, uint rows) {
  constexpr float seed[4][4] = {
      {SELECT_SEED_0_0, SELECT_SEED_0_1, SELECT_SEED_0_2, SELECT_SEED_0_3},
      {SELECT_SEED_1_0, SELECT_SEED_1_1, SELECT_SEED_1_2, SELECT_SEED_1_3},
      {SELECT_SEED_2_0, SELECT_SEED_2_1, SELECT_SEED_2_2, SELECT_SEED_2_3},
      {SELECT_SEED_3_0, SELECT_SEED_3_1, SELECT_SEED_3_2, SELECT_SEED_3_3}};
  constexpr float knots[4] = {1.0f, 2.0f, 4.0f, 8.0f};
  const uint shape = clamp(lanes, 1u, 4u) - 1;
  const float per_lane = float(rows) / float(max(lanes, 1u));
  uint upper = 1;
  while (upper < 3 && per_lane > knots[upper])
    ++upper;
  const float t = (per_lane - knots[upper - 1]) / (knots[upper] - knots[upper - 1]);
  return seed[shape][upper - 1] + t * (seed[shape][upper] - seed[shape][upper - 1]);
}

inline bool top_beats(float value, uint token, float other, uint other_token) {
  return value > other || (value == other && token < other_token);
}

// Register-resident sorted top-16 insert for an entry the caller has already checked against the
// last slot; the unrolled shift keeps every index static.
inline void top16_insert(thread float (&values)[K], thread uint (&ids)[K], float value,
                         uint token) {
#pragma clang loop unroll(full)
  for (uint slot = K - 1; slot > 0; --slot) {
    bool here = top_beats(value, token, values[slot], ids[slot]);
    bool above = top_beats(value, token, values[slot - 1], ids[slot - 1]);
    values[slot] = here ? (above ? values[slot - 1] : value) : values[slot];
    ids[slot] = here ? (above ? ids[slot - 1] : token) : ids[slot];
  }
  bool top = top_beats(value, token, values[0], ids[0]);
  values[0] = top ? value : values[0];
  ids[0] = top ? token : ids[0];
}

// Pops the head of a register-resident sorted list of Count entries.
template <uint Count>
inline void top_pop(thread float (&values)[Count], thread uint (&ids)[Count]) {
#pragma clang loop unroll(full)
  for (uint slot = 0; slot + 1 < Count; ++slot) {
    values[slot] = values[slot + 1];
    ids[slot] = ids[slot + 1];
  }
  values[Count - 1] = -INFINITY;
  ids[Count - 1] = 0xffffffffu;
}

// The simdgroup's best list head: (value desc, id asc), so ties and the empty sentinel (-inf, ~0u)
// resolve the same way everywhere.
inline void simd_best_head(float value, uint token, thread float &best, thread uint &best_token) {
  best = simd_max(value);
  best_token = simd_min(value == best ? token : 0xffffffffu);
}

// One shard of one proposal row (sampling.metal draft_select_top16_sharded). Threads stream their
// share of the shard in vectors, keep the chunk in registers, and first find the 16th largest of
// the per-thread maxima: at least sixteen tokens are that large, so nothing below it can be in the
// row's top-16 and the exact sorted insert only runs for the few survivors. The group then pops
// its best sixteen in rank order. The (value desc, id asc) order is total, so the partial is the
// same set in the same order whatever the thread partition.
template <typename L>
[[kernel]] void draft_select_top16_sharded(device const L *logits [[buffer(0)]],
                                           device uint *partial_ids [[buffer(1)]],
                                           device float *partial_values [[buffer(2)]],
                                           uint group [[threadgroup_position_in_grid]],
                                           uint thread_index [[thread_index_in_threadgroup]],
                                           uint lane [[thread_index_in_simdgroup]],
                                           uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Rows = SELECT_ROWS, Positions = SELECT_POSITIONS, Shards = SELECT_SHARDS;
  constexpr uint VectorTokens = 4, ChunkVectors = 16;
  constexpr uint vocabulary = SELECT_VOCAB;
  uint batch = group / (Positions * Shards);
  uint local = group % (Positions * Shards);
  uint position = local / Shards;
  uint shard = local % Shards;
  ulong row_start = (ulong(batch) * Rows + position + 1) * vocabulary;
  uint shard_tokens = (vocabulary + Shards - 1) / Shards;
  uint begin = min(shard * shard_tokens, vocabulary);
  uint end = min(begin + shard_tokens, vocabulary);
  // Vector loads require alignment. Handle the shard's unaligned head and tail with scalar
  // inserts.
  uint head = uint((VectorTokens - (row_start + begin) % VectorTokens) % VectorTokens);
  head = min(head, end - begin);
  uint vectors = (end - begin - head) / VectorTokens;
  uint vector_begin = begin + head;
  uint tail_begin = vector_begin + vectors * VectorTokens;
  device const L *row = logits + row_start;
  device const vec<L, 4> *vector_row = reinterpret_cast<device const vec<L, 4> *>(row + vector_begin);

  float values[K];
  uint ids[K];
  for (uint i = 0; i < K; ++i) {
    values[i] = -INFINITY;
    ids[i] = 0xffffffffu;
  }
  if (thread_index < head) {
    uint token = begin + thread_index;
    float value = float(row[token]);
    if (top_beats(value, token, values[K - 1], ids[K - 1]))
      top16_insert(values, ids, value, token);
  }
  if (thread_index < end - tail_begin) {
    uint token = tail_begin + thread_index;
    float value = float(row[token]);
    if (top_beats(value, token, values[K - 1], ids[K - 1]))
      top16_insert(values, ids, value, token);
  }

  threadgroup float maxima[256];
  threadgroup float thresholds[8];
  for (uint chunk = 0; chunk < vectors; chunk += 256 * ChunkVectors) {
    float4 loaded[ChunkVectors];
    float best = -INFINITY;
    for (uint i = 0; i < ChunkVectors; ++i) {
      uint index = chunk + thread_index + i * 256;
      loaded[i] = index < vectors ? float4(vector_row[index]) : float4(0.0f);
      if (index < vectors) {
        for (uint j = 0; j < VectorTokens; ++j)
          if (loaded[i][j] > best)
            best = loaded[i][j];
      }
    }
    // The 16th largest thread maximum: the minimum over the maxima that fewer than sixteen others
    // exceed.
    maxima[thread_index] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint above = 0;
    for (uint other = 0; other < 256; ++other)
      above += maxima[other] > best ? 1u : 0u;
    float candidate = simd_min(above < K ? best : INFINITY);
    if (lane == 0)
      thresholds[simd_group] = candidate;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float threshold = thresholds[0];
    for (uint other = 1; other < 8; ++other)
      threshold = min(threshold, thresholds[other]);

    for (uint i = 0; i < ChunkVectors; ++i) {
      uint index = chunk + thread_index + i * 256;
      if (index >= vectors)
        continue;
      uint token = vector_begin + index * VectorTokens;
      for (uint j = 0; j < VectorTokens; ++j) {
        float value = loaded[i][j];
        if (value >= threshold && top_beats(value, token + j, values[K - 1], ids[K - 1]))
          top16_insert(values, ids, value, token + j);
      }
    }
  }

  // Sixteen rounds pop the group-wide best head; the simdgroup bests alternate between two slots
  // so one barrier per round suffices.
  threadgroup float round_values[2][8];
  threadgroup uint round_ids[2][8];
  for (uint rank = 0; rank < K; ++rank) {
    float head_value = values[0];
    uint head_id = ids[0];
    float simd_value;
    uint simd_id;
    simd_best_head(head_value, head_id, simd_value, simd_id);
    uint slot = rank & 1;
    if (lane == 0) {
      round_values[slot][simd_group] = simd_value;
      round_ids[slot][simd_group] = simd_id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float best = round_values[slot][0];
    uint best_id = round_ids[slot][0];
    for (uint other = 1; other < 8; ++other) {
      float value = round_values[slot][other];
      uint token = round_ids[slot][other];
      if (top_beats(value, token, best, best_id)) {
        best = value;
        best_id = token;
      }
    }
    if (thread_index == rank) {
      partial_ids[group * K + rank] = best_id;
      partial_values[group * K + rank] = best;
    }
    if (head_value == best && head_id == best_id)
      top_pop<K>(values, ids);
  }
}

// Merges the shard partials of one row into the simdgroup's lanes: each lane holds consecutive
// entries of one shard's sorted list and sixteen rounds pop the simdgroup-wide best into rank
// order, lane `rank` keeping it.
inline void top16_merge_shards(device const uint *partial_ids, device const float *partial_values,
                               uint row, uint lane, thread float &value, thread uint &token) {
  constexpr uint Shards = SELECT_SHARDS;
  constexpr uint Entries = Shards * K / 32;
  uint origin = row * Shards * K + lane * Entries;
  float values[Entries];
  uint ids[Entries];
  for (uint i = 0; i < Entries; ++i) {
    values[i] = partial_values[origin + i];
    ids[i] = partial_ids[origin + i];
  }
  value = -INFINITY;
  token = 0xffffffffu;
  for (uint rank = 0; rank < K; ++rank) {
    float best;
    uint best_id;
    simd_best_head(values[0], ids[0], best, best_id);
    if (lane == rank) {
      value = best;
      token = best_id;
    }
    if (values[0] == best && ids[0] == best_id)
      top_pop<Entries>(values, ids);
  }
}

// One group per proposal row (sampling.metal draft_select_edges). Simdgroup 0 merges its
// candidates and unary scores; simdgroup 1 merges the preceding row (the anchor for row 0). The
// remaining simdgroups score the 16 x 16 predecessor/candidate edge table in fixed per-lane
// reduction order. The table follows the shard partials in partial-values scratch.
template <typename T>
[[kernel]] void draft_select_edges(device const uint *partial_ids [[buffer(0)]],
                                   device float *partial_values [[buffer(1)]],
                                   device uint *candidates [[buffer(2)]],
                                   device float *unary [[buffer(3)]],
                                   device const T *hidden [[buffer(4)]],
                                   device const bfloat *predecessor_codebook [[buffer(5)]],
                                   device const bfloat *successor_codebook [[buffer(6)]],
                                   device const uint *input_ids [[buffer(7)]],
                                   device const uint *num_tokens [[buffer(8)]],
                                   uint row [[threadgroup_position_in_grid]],
                                   uint threads [[threads_per_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Rows = SELECT_ROWS, Positions = SELECT_POSITIONS, Shards = SELECT_SHARDS;
  constexpr uint Rank = SELECT_RANK, vocabulary = SELECT_VOCAB;
  uint batch = row / Positions;
  uint position = row % Positions;
  const uint lanes = num_tokens[0] / SELECT_ROWS;
  threadgroup uint successors[K];
  threadgroup uint predecessors[K];
  if (simd_group == 0) {
    float value;
    uint token;
    top16_merge_shards(partial_ids, partial_values, row, lane, value, token);
    if (lane < K) {
      candidates[row * K + lane] = token;
      unary[row * K + lane] = value;
      successors[lane] = token;
    }
  } else if (simd_group == 1) {
    uint token = input_ids[batch * Rows];
    if (position > 0) {
      float value;
      top16_merge_shards(partial_ids, partial_values, row - 1, lane, value, token);
    }
    if (lane < K)
      predecessors[lane] = token;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // One task is one predecessor and eight of the sixteen candidates, so a simdgroup issues all its
  // codebook loads at once instead of one cold row per edge.
  constexpr uint TaskCandidates = 8, Dims = Rank / 32;
  uint tasks = (position > 0 ? K : 1) * (K / TaskCandidates);
  device float *table =
      partial_values + ulong(lanes) * Positions * Shards * K + ulong(row) * K * K;
  device const T *row_hidden = hidden + (ulong(batch) * Rows + position + 1) * Rank;
  for (uint task = simd_group; task < tasks; task += threads / 32) {
    uint predecessor_index = task / (K / TaskCandidates);
    uint first_candidate = task % (K / TaskCandidates) * TaskCandidates;
    // Ids are produced by the top-k selection and are always in range; the clamp only keeps a
    // corrupted id inside the codebooks.
    uint safe_predecessor = min(predecessors[predecessor_index], vocabulary - 1u);
    float context[Dims];
    float successor[TaskCandidates][Dims];
    for (uint i = 0; i < Dims; ++i) {
      uint dim = lane + i * 32;
      context[i] =
          float(predecessor_codebook[safe_predecessor * Rank + dim]) * float(row_hidden[dim]);
    }
    for (uint j = 0; j < TaskCandidates; ++j) {
      uint safe_candidate = min(successors[first_candidate + j], vocabulary - 1u);
      for (uint i = 0; i < Dims; ++i)
        successor[j][i] = float(successor_codebook[safe_candidate * Rank + lane + i * 32]);
    }
    for (uint j = 0; j < TaskCandidates; ++j) {
      float score = 0.0f;
      for (uint i = 0; i < Dims; ++i)
        score += context[i] * successor[j][i];
      score = simd_sum(score);
      if (lane == 0)
        table[predecessor_index * K + first_candidate + j] = score;
    }
  }
}

// One thread per lane walks the seven positions (sampling.metal draft_select_dflash): the score
// of a candidate is its unary score plus the edge from the previously chosen candidate, read from
// the table draft_select_edges left in the partial-values scratch. It writes the lane's confidence
// at each position as it walks. A sampling lane draws its proposal at position p with its step's
// uniform DRAFT_UNIFORM_PROPOSALS + p (`draft_uniform`).
[[kernel]] void draft_select_dflash(device const uint *candidates [[buffer(0)]],
                                    device const float *unary [[buffer(1)]],
                                    device const float *partial_values [[buffer(2)]],
                                    device uint *tokens [[buffer(3)]],
                                    device float *q_probs [[buffer(4)]],
                                    device float *confidence [[buffer(5)]],
                                    device const DraftLaneParams *lane_params [[buffer(6)]],
                                    device const uint *num_tokens [[buffer(7)]],
                                    uint batch [[thread_position_in_grid]]) {
  constexpr ulong Positions = SELECT_POSITIONS, Shards = SELECT_SHARDS, Candidates = K;
  const uint lanes = num_tokens[0] / SELECT_ROWS;
  if (batch >= lanes)
    return;
  candidates += batch * Positions * Candidates;
  unary += batch * Positions * Candidates;
  device const float *tables =
      partial_values + lanes * Positions * Shards * Candidates + batch * Positions * Candidates * Candidates;
  tokens += batch * Positions;
  q_probs += batch * Positions * Candidates;
  confidence += batch * Positions;

  const DraftLaneParams lp = lane_params[batch];
  const bool sampling = lp.sampling != 0;
  uint predecessor_index = 0;
  for (uint position = 0; position < Positions; ++position) {
    device const float *edges = tables + (position * Candidates + predecessor_index) * Candidates;
    float scores[Candidates];
    for (uint i = 0; i < Candidates; ++i)
      scores[i] = unary[position * Candidates + i] + edges[i];
    // The largest of the position's candidate probabilities at temperature 1, before the lane
    // chooses among them.
    float best = scores[0];
    for (uint i = 1; i < Candidates; ++i)
      best = max(best, scores[i]);
    float mass = 0.0f;
    for (uint i = 0; i < Candidates; ++i)
      mass += exp(scores[i] - best);
    confidence[position] = 1.0f / mass;
    uint selected = 0;
    if (sampling) {
      float maximum = scores[0];
      for (uint i = 1; i < Candidates; ++i)
        maximum = max(maximum, scores[i]);
      float sum = 0.0f;
      for (uint i = 0; i < Candidates; ++i) {
        float probability = exp((scores[i] - maximum) / lp.temperature);
        q_probs[position * Candidates + i] = probability;
        sum += probability;
      }
      const float uniform =
          draft_uniform(lp.seed_lo, lp.seed_hi, lp.position, DRAFT_UNIFORM_PROPOSALS + position);
      float cumulative = 0.0f;
      selected = Candidates - 1;
      for (uint i = 0; i < Candidates; ++i) {
        float probability = q_probs[position * Candidates + i] / sum;
        q_probs[position * Candidates + i] = probability;
        cumulative += probability;
        if (selected == Candidates - 1 && cumulative > uniform) {
          selected = i;
        }
      }
    } else {
      for (uint i = 1; i < Candidates; ++i) {
        if (scores[i] > scores[selected])
          selected = i;
      }
    }
    predecessor_index = selected;
    tokens[position] = candidates[position * Candidates + selected];
  }
}

// One thread chooses every lane's verify rows (sampling.metal draft_verify_rows): it ranks the
// survivals of the lanes' proposal positions, largest first, the earlier position first among
// equal ones, so that a lane's chosen rows are always the first of its block, and takes as many as
// maximize the expected tokens over the cost of the rows verified in all. With every row, each
// lane verifies its maximum.
[[kernel]] void draft_verify_rows(device const float *confidence [[buffer(0)]],
                                  device uint *verify_rows [[buffer(1)]],
                                  device const DraftLaneParams *lane_params [[buffer(2)]],
                                  device const uint *num_tokens [[buffer(3)]],
                                  uint index [[thread_position_in_grid]]) {
  constexpr uint Positions = SELECT_POSITIONS;
  constexpr uint Lanes = SELECT_LANES;
  if (index != 0)
    return;
  const uint lanes = min(num_tokens[0] / SELECT_ROWS, Lanes);
  uint rows[Lanes];
  uint total = 0;
  for (uint lane = 0; lane < lanes; ++lane) {
    rows[lane] = SELECT_EVERY_ROW ? lane_params[lane].maximum : 1;
    total += rows[lane];
  }
  if (!SELECT_EVERY_ROW) {
    float survivals[Lanes * Positions];
    uint owners[Lanes * Positions];
    uint count = 0;
    for (uint lane = 0; lane < lanes; ++lane) {
      float survival = 1.0f;
      // Row position + 2 verifies the proposal at `position`.
      for (uint position = 0; position + 1 < lane_params[lane].maximum; ++position) {
        const float q = clamp(confidence[lane * Positions + position], 1.0e-6f, 1.0f - 1.0e-6f);
        const float z = SELECT_SLOPE * log(q / (1.0f - q)) + select_intercept(position);
        survival *= 1.0f / (1.0f + exp(-z));
        uint at = count++;
        for (; at > 0 && survivals[at - 1] < survival; --at) {
          survivals[at] = survivals[at - 1];
          owners[at] = owners[at - 1];
        }
        survivals[at] = survival;
        owners[at] = lane;
      }
    }
    // The host's cost line when it gives one (a row's bytes past zero), else the seed.
    const float fixed = lane_params[0].cost_fixed, per_row = lane_params[0].cost_per_row;
    auto cost = [&](uint rows) {
      return per_row > 0.0f ? fixed + per_row * float(rows) : select_cost(lanes, rows);
    };
    float expected = float(lanes);
    float best = expected / cost(total);
    uint taken = 0;
    for (uint i = 0; i < count; ++i) {
      expected += survivals[i];
      const float value = expected / cost(total + i + 1);
      if (value > best) {
        best = value;
        taken = i + 1;
      }
    }
    for (uint i = 0; i < taken; ++i)
      ++rows[owners[i]];
  }
  for (uint lane = 0; lane < lanes; ++lane)
    verify_rows[lane] = rows[lane];
}

SCRATCHY_KERNEL(draft_select_top16_sharded_f16, draft_select_top16_sharded<half>)
SCRATCHY_KERNEL(draft_select_top16_sharded_bf16, draft_select_top16_sharded<bfloat>)
SCRATCHY_KERNEL(draft_select_top16_sharded_f32, draft_select_top16_sharded<float>)
SCRATCHY_KERNEL(draft_select_edges_f16, draft_select_edges<half>)
SCRATCHY_KERNEL(draft_select_edges_bf16, draft_select_edges<bfloat>)
