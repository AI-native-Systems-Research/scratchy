// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

#include <metal_stdlib>
#include "baked.h"
using namespace metal;

// ---------------------------------------------------------------------------
// logprobs_topk — the OpenAI `logprobs` / `top_logprobs` of emitted tokens: for each scored entry,
// the log-softmax of its logits row at the token emitted from it, that token's rank, and the
// row's LOGPROBS_TOP (20, OpenAI's `top_logprobs` maximum) most likely tokens, as cuda's
// `log_softmax_topk_gather` computes them; a request keeps as many as it asked for.
//
// Input:  logits [rows, LOGPROBS_VOCAB]  (half / bfloat), the step's sampled rows
//         rows   [count]  uint            each entry's logits row
//         tokens [rows]   uint            the token emitted from each logits row
// Output: per entry, LOGPROBS_TOP + 1 slots: slot 0 the emitted token, slots 1.. the top tokens
//         in descending order (ties to the smaller id), each a token id, its log-probability and
//         its 1-based rank (1 + the logits strictly above its own).
//
// Dispatch: threadgroups (count, 1, 1), threads (LOGPROBS_THREADS, 1, 1). One threadgroup an entry;
// its threads stride the vocabulary. Each keeps an online log-sum-exp (its maximum and the sum of
// exp(logit − maximum)), the count of logits above the emitted token's, and its own top
// LOGPROBS_TOP; the threadgroup merges the sums, the counts, and the lists (LOGPROBS_TOP rounds of
// its threads' best remaining candidate). NaN logits count nowhere.
//
// Bindings:
//   buffer(0) = logits  buffer(1) = rows  buffer(2) = tokens  buffer(3) = count (constant uint)
//   buffer(4) = out_logprob [count, TOP + 1] float
//   buffer(5) = out_token   [count, TOP + 1] uint
//   buffer(6) = out_rank    [count, TOP + 1] uint
// ---------------------------------------------------------------------------

SCRATCHY_CONSTANT(uint, LOGPROBS_VOCAB, 0);

constant constexpr uint LOGPROBS_THREADS = 1024;
constant constexpr uint LOGPROBS_SIMDGROUPS = LOGPROBS_THREADS / 32;
// The most top tokens any request asks for: OpenAI's `top_logprobs` maximum.
constant constexpr uint LOGPROBS_TOP = 20;

// `a` before `b` in the top list: larger, or equal at a smaller id.
inline bool logprobs_before(float a, uint a_id, float b, uint b_id) {
  return a > b || (a == b && a_id < b_id);
}

// Two online log-sum-exp partials, (maximum, sum of exp(x − maximum)), merged.
inline float2 logprobs_merge(float2 a, float2 b) {
  if (b.x == -INFINITY)
    return a;
  if (a.x == -INFINITY)
    return b;
  const float m = max(a.x, b.x);
  return float2(m, a.y * exp(a.x - m) + b.y * exp(b.x - m));
}

template <typename T>
[[kernel]] void logprobs_topk(device const T *logits [[buffer(0)]],
                              device const uint *rows [[buffer(1)]],
                              device const uint *tokens [[buffer(2)]],
                              constant uint &count [[buffer(3)]],
                              device float *out_logprob [[buffer(4)]],
                              device uint *out_token [[buffer(5)]],
                              device uint *out_rank [[buffer(6)]],
                              uint entry [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint K = LOGPROBS_TOP;
  threadgroup float2 lse_shared[LOGPROBS_SIMDGROUPS];
  threadgroup uint above_shared[LOGPROBS_SIMDGROUPS];
  threadgroup float best_value[LOGPROBS_SIMDGROUPS];
  threadgroup uint best_id[LOGPROBS_SIMDGROUPS];
  threadgroup uint winner;
  if (entry >= count)
    return;
  const uint row = rows[entry];
  device const T *x = logits + ulong(row) * LOGPROBS_VOCAB;
  const uint token = tokens[row];
  const float emitted = float(x[min(token, LOGPROBS_VOCAB - 1)]);

  // This thread's pass: log-sum-exp, the logits above the emitted one, its own top K.
  float2 lse = float2(-INFINITY, 0.0f);
  uint above = 0;
  float top_value[K];
  uint top_id[K];
  for (uint k = 0; k < K; ++k) {
    top_value[k] = -INFINITY;
    top_id[k] = 0xFFFFFFFFu;
  }
  for (uint i = tid; i < LOGPROBS_VOCAB; i += LOGPROBS_THREADS) {
    const float v = float(x[i]);
    if (isnan(v))
      continue;
    lse = logprobs_merge(lse, float2(v, 1.0f));
    above += v > emitted ? 1u : 0u;
    if (logprobs_before(v, i, top_value[LOGPROBS_TOP - 1], top_id[LOGPROBS_TOP - 1])) {
      uint slot = LOGPROBS_TOP - 1;
      while (slot > 0 && logprobs_before(v, i, top_value[slot - 1], top_id[slot - 1])) {
        top_value[slot] = top_value[slot - 1];
        top_id[slot] = top_id[slot - 1];
        --slot;
      }
      top_value[slot] = v;
      top_id[slot] = i;
    }
  }

  // The threadgroup's log-sum-exp and count.
  for (ushort off = 16; off > 0; off >>= 1) {
    lse = logprobs_merge(lse, float2(simd_shuffle_down(lse.x, off), simd_shuffle_down(lse.y, off)));
    above += simd_shuffle_down(above, off);
  }
  if (lane == 0) {
    lse_shared[simd_group] = lse;
    above_shared[simd_group] = above;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float2 total = float2(-INFINITY, 0.0f);
    uint above_total = 0;
    for (uint s = 0; s < LOGPROBS_SIMDGROUPS; ++s) {
      total = logprobs_merge(total, lse_shared[s]);
      above_total += above_shared[s];
    }
    lse_shared[0] = total;
    above_shared[0] = above_total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float log_sum = lse_shared[0].x + log(lse_shared[0].y);
  const ulong out = ulong(entry) * (LOGPROBS_TOP + 1);
  if (tid == 0) {
    out_logprob[out] = emitted - log_sum;
    out_token[out] = token;
    out_rank[out] = above_shared[0] + 1;
  }

  // The top list: each round the threadgroup's best head; its thread pops it.
  uint head = 0;
  for (uint k = 0; k < LOGPROBS_TOP; ++k) {
    float v = head < LOGPROBS_TOP ? top_value[head] : -INFINITY;
    uint id = head < LOGPROBS_TOP ? top_id[head] : 0xFFFFFFFFu;
    for (ushort off = 16; off > 0; off >>= 1) {
      const float ov = simd_shuffle_down(v, off);
      const uint oid = simd_shuffle_down(id, off);
      if (logprobs_before(ov, oid, v, id)) {
        v = ov;
        id = oid;
      }
    }
    if (lane == 0) {
      best_value[simd_group] = v;
      best_id[simd_group] = id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float bv = -INFINITY;
      uint bid = 0xFFFFFFFFu;
      for (uint s = 0; s < LOGPROBS_SIMDGROUPS; ++s) {
        if (logprobs_before(best_value[s], best_id[s], bv, bid)) {
          bv = best_value[s];
          bid = best_id[s];
        }
      }
      winner = bid;
      out_logprob[out + 1 + k] = bv - log_sum;
      out_token[out + 1 + k] = bid;
      out_rank[out + 1 + k] = k + 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (head < LOGPROBS_TOP && top_id[head] == winner)
      ++head;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

SCRATCHY_KERNEL(logprobs_topk_f16, logprobs_topk<half>)
SCRATCHY_KERNEL(logprobs_topk_bf16, logprobs_topk<bfloat>)
