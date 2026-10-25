// SPDX-License-Identifier: Apache-2.0
//
// Splash's target policy over a DFlash2 cycle's verify rows (~/git/splash @ 1482f6bb
// runtime/metal/kernels/decode/sampling.metal:1-920: `decode_sample_mass_sharded`,
// `decode_sample_vocabulary_search`, `decode_sample_vocabulary_draw` and their helpers;
// runtime/metal/kernels/common/split_reduce.h), with these changes, and no other:
//
//   - The logits are the model's activation type (half or bfloat); Splash's are f32. Every value
//     is read as f32.
//   - A row admits every token but the stop tokens of a lane that ignores end-of-sequence: a lane
//     whose grammar constrains it drafts nothing here (Splash's `constrained` mask rows).
//   - A lane's policy and its rows are its `VerifyLane` (dflash2_cycle.h): selected row s is row
//     s % ROWS of lane s / ROWS, at its logits row `sampled + s % ROWS`; a lane has `drafts + 1`
//     rows, the others return at once. Splash reads them from `TargetSamplingParams`.
//   - A drafted row's draft token is the lane's `drafted` token, which Splash reads from its verify
//     input; its draft candidates and their probabilities are the drafter's selection's.
//
// The rows of the target policy. A sampled lane draws from each row's min-p/top-k/top-p
// distribution over the whole vocabulary: a sharded scan sums the row's softmax denominator, one
// group per row finds the last token the distribution keeps in the order of the logits (logit
// descending, id ascending) without sorting the vocabulary, and the draw takes one pass over the
// kept tokens in id order, shared by the row's GROUPS groups: each sums ranges of the vocabulary,
// and the one that finishes last (split_arrive_last) draws. Every reduction runs in a fixed order,
// so a row selects the same token on every run.
//
// Baked constants:
//   TSAMPLE_VOCAB — the logits' width
//   TSAMPLE_ROWS  — a lane's block: its anchor and its proposals (SPLASH_TARGET_VERIFY_ROWS)

#include <metal_stdlib>
#include "baked.h"
#include "dflash2_cycle.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, TSAMPLE_VOCAB, 0);
SCRATCHY_CONSTANT(uint, TSAMPLE_ROWS, 1);

constant constexpr uint kProposals = TSAMPLE_ROWS - 1;
// SPLASH_TARGET_SAMPLING_SHARDS, SPLASH_TARGET_VOCABULARY_THREADS, SPLASH_TARGET_VOCABULARY_GROUPS,
// SPLASH_TARGET_VOCABULARY_RANGES, SPLASH_DRAFT_CANDIDATES (runtime/metal/abi/ExecutionGeometry.h).
constant constexpr uint kShards = 16;
constant constexpr uint kVocabularyThreads = 1024;
constant constexpr uint kVocabularySimdgroups = kVocabularyThreads / 32;
constant constexpr uint kVocabularyGroups = 8;
constant constexpr uint kVocabularyRanges = kVocabularyGroups * (kVocabularyThreads / 32);
static_assert(kVocabularyRanges <= kVocabularyThreads, "a group holds one draw range per thread");
constant constexpr uint kDraftCandidates = 16;
// Pivots per pass: each pass splits a bracket sixteen ways. In logit order kKeyPivots of them
// split its keys evenly and the others follow the logits' scale (place_pivots).
constant constexpr uint kPivots = 15;
constant constexpr uint kKeyPivots = 4;

// ── split_reduce.h ─────────────────────────────────────────────────────────────────────────────

// True in every thread of the threadgroup that arrives last.
inline bool split_arrive_last(device atomic_uint *counter, uint splits, uint thread_index,
                              threadgroup uint *arrival) {
  // Every writer publishes its partials before thread zero signals arrival. The device-scope
  // fences pair through the counter.
  threadgroup_barrier(mem_flags::mem_device);
  if (thread_index == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    *arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  return *arrival == splits - 1;
}

// Returns the tile's counter to zero; called by the reducing threadgroup.
inline void split_release(device atomic_uint *counter, uint thread_index) {
  if (thread_index == 0)
    atomic_store_explicit(counter, 0u, memory_order_relaxed);
}

// ── The rows ───────────────────────────────────────────────────────────────────────────────────

// The row a lane selects from: its logits, the tokens it admits (all but the stop tokens when the
// lane ignores end-of-sequence) and, for a sampled row whose maximum is known, their softmax
// weights.
template <typename L>
struct TargetRow {
  device const L *logits;
  bool exclude_stop;
  uint stop_token_0;
  uint stop_token_1;
  uint vocabulary;
  float temperature;
  float maximum;

  bool admits(uint token) const {
    return !exclude_stop || (token != stop_token_0 && token != stop_token_1);
  }
  float logit(uint token) const { return float(logits[token]); }
  float weight(float value) const { return exp((value - maximum) / temperature); }
};

// Whether selected row s of a dispatch is a row of a sampled lane.
inline bool samples(device const VerifyLane *lanes, uint s) {
  const VerifyLane lane = lanes[s / TSAMPLE_ROWS];
  return lane.sampling != 0 && s % TSAMPLE_ROWS <= lane.drafts;
}

// Selected row s of a dispatch.
template <typename L>
inline TargetRow<L> selected_row(device const L *logits, device const VerifyLane *lanes, uint s) {
  const VerifyLane lane = lanes[s / TSAMPLE_ROWS];
  return {logits + ulong(lane.sampled + s % TSAMPLE_ROWS) * TSAMPLE_VOCAB,
          lane.exclude_stop != 0,
          lane.stop[0],
          lane.stop[1],
          TSAMPLE_VOCAB,
          lane.temperature,
          0.0f};
}

// Shares of a row's softmax denominator combined in order: their largest maximum, their sums
// rescaled to it and their admitted counts. A group combines its simdgroups' shares, and a row's
// search its shards'.
template <class Shares>
inline TargetShardMass merge_masses(Shares shares, uint count, float temperature) {
  TargetShardMass total{-FLT_MAX, 0.0f, 0};
  for (uint index = 0; index < count; ++index)
    total.maximum = max(total.maximum, shares[index].maximum);
  for (uint index = 0; index < count; ++index) {
    const TargetShardMass share = shares[index];
    if (share.sum > 0.0f)
      total.sum += share.sum * exp((share.maximum - total.maximum) / temperature);
    total.admitted += share.admitted;
  }
  return total;
}

// One shard's share of a sampled row's softmax denominator: each thread keeps a running maximum
// and sum of its admitted tokens' weights (an online softmax), combined over the group in a fixed
// order. The running maximum starts below every finite logit, so a -inf logit weighs nothing.
template <typename L>
inline void shard_mass(TargetRow<L> row, uint shard, device TargetShardMass &partial,
                       threadgroup TargetShardMass *group_masses, uint thread_index, uint lane,
                       uint simd_group) {
  TargetShardMass mass{-FLT_MAX, 0.0f, 0};
  for (uint token = shard * 256 + thread_index; token < row.vocabulary; token += kShards * 256) {
    if (!row.admits(token))
      continue;
    const float value = row.logit(token);
    if (value > mass.maximum) {
      mass.sum = mass.sum * exp((mass.maximum - value) / row.temperature) + 1.0f;
      mass.maximum = value;
    } else {
      mass.sum += exp((value - mass.maximum) / row.temperature);
    }
    ++mass.admitted;
  }
  const float simd_maximum = simd_max(mass.maximum);
  const float scaled =
      mass.sum > 0.0f ? mass.sum * exp((mass.maximum - simd_maximum) / row.temperature) : 0.0f;
  const float simd_total = simd_sum(scaled);
  const uint simd_admitted = simd_sum(mass.admitted);
  if (lane == 0)
    group_masses[simd_group] = {simd_maximum, simd_total, simd_admitted};
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0)
    partial = merge_masses(group_masses, 8, row.temperature);
}

// One shard of each selected row of the sampled lanes: SHARDS groups of 256 threads a row.
template <typename L>
[[kernel]] void dflash2_sample_mass(device const L *logits [[buffer(0)]],
                                    device const VerifyLane *lanes [[buffer(1)]],
                                    device TargetShardMass *partial_masses [[buffer(2)]],
                                    uint group [[threadgroup_position_in_grid]],
                                    uint thread_index [[thread_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup TargetShardMass group_masses[8];
  const uint s = group / kShards;
  if (!samples(lanes, s))
    return;
  shard_mass(selected_row(logits, lanes, s), group % kShards, partial_masses[group], group_masses,
             thread_index, lane, simd_group);
}

// ── The search ─────────────────────────────────────────────────────────────────────────────────

// The order of the logits as an unsigned key: a larger logit has a larger key, and -0 and +0,
// which compare equal, share one. Every logit's key is above kNoKey.
inline uint logit_key(float value) {
  uint bits = as_type<uint>(value);
  bits = (bits << 1) ? bits : 0u;
  return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}
inline float key_logit(uint key) {
  return as_type<float>((key & 0x80000000u) ? key & 0x7fffffffu : ~key);
}
constant constexpr uint kNoKey = 0u;
// The key of -infinity, at or below that of every admitted logit.
constant constexpr uint kLowestKey = 0x007fffffu;

// A position in the order of the logits: the tokens at or before it have a larger key, or the
// same key and an id at most last. {kNoKey, 0} lies after every token.
struct OrderBoundary {
  uint key;
  uint last;
};

inline bool at_or_before(uint key, uint token, OrderBoundary boundary) {
  return key > boundary.key || (key == boundary.key && token <= boundary.last);
}

// The orders a search runs in. The logit order is the order of the logits. Among the tokens tied
// at one logit, the tie order puts a smaller id first; every other token has kNoKey.
struct LogitOrder {
  static constexpr constant bool by_logit = true;
  uint key(float value, uint) const { return logit_key(value); }
  float logit(uint key) const { return key_logit(key); }
};
struct TieOrder {
  uint tied_key;
  uint vocabulary;
  uint key(float value, uint token) const {
    return logit_key(value) == tied_key ? vocabulary - token : kNoKey;
  }
  static constexpr constant bool by_logit = false;
  float logit(uint) const { return key_logit(tied_key); }
};

// What a search keeps: the tokens in order until their count, or their mass (the sum of their
// weights), exceeds the target. A count search measures counts alone (top-k, and the tie order's
// first ids), a mass search counts and masses (top-p).
struct CountTarget {
  static constexpr constant bool by_mass = false;
  uint count;

  bool exceeded(uint measured_count, float) const { return measured_count > count; }
};
struct MassTarget {
  static constexpr constant bool by_mass = true;
  float mass;

  bool exceeded(uint, float measured_mass) const { return measured_mass > mass; }
};

// A key range [lo, hi) and the count and mass of the tokens with at least each end's key: lo's
// exceed the search target, hi's do not. A count search narrows the counts alone and leaves the
// masses as they came in.
struct Bracket {
  uint lo;
  uint hi;
  uint lo_count;
  uint hi_count;
  float lo_mass;
  float hi_mass;

  uint tokens() const { return lo_count - hi_count; }
};

// The last token a search keeps, and the count and mass of the tokens at or before it. A ranked
// selection leaves those tokens in the scratch keys and ids, in order (resolve).
struct Selection {
  OrderBoundary last;
  uint count;
  float mass;
  bool ranked;
};

struct VocabularyScratch {
  uint counts[kVocabularySimdgroups][kPivots];
  float masses[kVocabularySimdgroups][kPivots];
  uint total_counts[kPivots];
  float total_masses[kPivots];
  atomic_uint gathered;
  uint keys[kVocabularyThreads];
  uint ids[kVocabularyThreads];
  Selection selection;
};

struct DrawScratch {
  uint arrival;
  // Per range of the draw: the kept weight of its tokens other than the draft's candidates, one
  // past the last of those with weight, and its weight in the draw.
  float range_rest[kVocabularyRanges];
  uint range_last[kVocabularyRanges];
  float range_weights[kVocabularyRanges];
  // The draft's candidates, each listed once: its draft probability, kept weight, weight in the
  // draw and range.
  uint draft_ids[kDraftCandidates];
  float draft_probabilities[kDraftCandidates];
  float draft_weights[kDraftCandidates];
  float draft_draw_weights[kDraftCandidates];
  uint draft_ranges[kDraftCandidates];
  float kept_mass;
  uint drawn_range;
  float drawn_target;
  uint drawn_token;
};

// Pivots inside [lo, hi) that split the bracket. The tie order splits its keys evenly. In logit
// order the last kKeyPivots do, so a pass keeps at most a fifth of the bracket's keys, rounded
// up, and a search ends within fourteen passes whatever the logits, temperature or penalties; the
// others follow the logits' scale, which ends most searches within a few: geometric distances
// below the bracket's top, in units of the temperature, while it is open at the bottom, and even
// steps in logit value once it is not.
template <class Order>
inline void place_pivots(Order order, Bracket bracket, float temperature,
                         thread uint (&pivots)[kPivots]) {
  const ulong width = bracket.hi - bracket.lo;
  if (!order.by_logit) {
    for (uint pivot = 0; pivot < kPivots; ++pivot)
      pivots[pivot] = bracket.lo + uint(width * (pivot + 1) / (kPivots + 1));
    return;
  }
  constexpr uint kScaled = kPivots - kKeyPivots;
  const float top = key_logit(bracket.hi - 1);
  const float bottom = key_logit(bracket.lo);
  const bool open = bracket.lo == kLowestKey;
  for (uint pivot = 0; pivot < kScaled; ++pivot) {
    const float value = open ? top - temperature * 0.5f * exp2(float(kScaled - 1 - pivot))
                             : bottom + (top - bottom) * float(pivot + 1) / float(kScaled + 1);
    pivots[pivot] = clamp(logit_key(value), bracket.lo, bracket.hi - 1);
  }
  for (uint pivot = 0; pivot < kKeyPivots; ++pivot)
    pivots[kScaled + pivot] = bracket.lo + uint(width * (pivot + 1) / (kKeyPivots + 1));
}

// The count and, with Masses, the mass of the admitted tokens at or before the floor whose keys
// are at least each pivot's, summed per thread in token order and then over the group in a fixed
// order. Counts alone weigh no token.
template <bool Masses, uint Pivots, class Order, typename L>
inline void measure_pivots(TargetRow<L> row, Order order, OrderBoundary floor,
                           thread const uint (&pivots)[Pivots],
                           threadgroup VocabularyScratch &scratch, uint thread_index, uint lane,
                           uint simd_group) {
  uint counts[Pivots];
  float masses[Pivots];
  uint lowest = pivots[0];
  for (uint pivot = 0; pivot < Pivots; ++pivot) {
    counts[pivot] = 0;
    if constexpr (Masses)
      masses[pivot] = 0.0f;
    lowest = min(lowest, pivots[pivot]);
  }
  for (uint token = thread_index; token < row.vocabulary; token += kVocabularyThreads) {
    const float value = row.logit(token);
    const uint key = order.key(value, token);
    // Tokens below every pivot add nothing.
    if (key < lowest || !row.admits(token) || !at_or_before(logit_key(value), token, floor))
      continue;
    if constexpr (Masses) {
      const float weight = row.weight(value);
      for (uint pivot = 0; pivot < Pivots; ++pivot) {
        const bool above = key >= pivots[pivot];
        counts[pivot] += above ? 1u : 0u;
        masses[pivot] += above ? weight : 0.0f;
      }
    } else {
      for (uint pivot = 0; pivot < Pivots; ++pivot)
        counts[pivot] += key >= pivots[pivot] ? 1u : 0u;
    }
  }
  for (uint pivot = 0; pivot < Pivots; ++pivot) {
    const uint count = simd_sum(counts[pivot]);
    if (lane == 0)
      scratch.counts[simd_group][pivot] = count;
    if constexpr (Masses) {
      const float mass = simd_sum(masses[pivot]);
      if (lane == 0)
        scratch.masses[simd_group][pivot] = mass;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < Pivots) {
    uint count = 0;
    for (uint simd = 0; simd < kVocabularySimdgroups; ++simd)
      count += scratch.counts[simd][thread_index];
    scratch.total_counts[thread_index] = count;
    if constexpr (Masses) {
      float mass = 0.0f;
      for (uint simd = 0; simd < kVocabularySimdgroups; ++simd)
        mass += scratch.masses[simd][thread_index];
      scratch.total_masses[thread_index] = mass;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The count and mass of the admitted tokens at or before the floor whose logits' keys are at
// least key, summed as measure_pivots sums a pivot's.
struct Measure {
  uint count;
  float mass;
};
template <typename L>
inline Measure measure_at_or_above(TargetRow<L> row, OrderBoundary floor, uint key,
                                   threadgroup VocabularyScratch &scratch, uint thread_index,
                                   uint lane, uint simd_group) {
  const uint pivots[1] = {key};
  measure_pivots<true>(row, LogitOrder{}, floor, pivots, scratch, thread_index, lane, simd_group);
  const Measure measure{scratch.total_counts[0], scratch.total_masses[0]};
  // Every thread reads the totals before the next pass rewrites them.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return measure;
}

// Narrows the bracket until it holds at most one token per thread or a single key.
template <class Order, class Target, typename L>
inline Bracket narrow(TargetRow<L> row, Order order, OrderBoundary floor, Bracket bracket,
                      Target target, threadgroup VocabularyScratch &scratch, uint thread_index,
                      uint lane, uint simd_group) {
  while (bracket.tokens() > kVocabularyThreads && bracket.hi - bracket.lo > 1) {
    uint pivots[kPivots];
    place_pivots(order, bracket, row.temperature, pivots);
    measure_pivots<Target::by_mass>(row, order, floor, pivots, scratch, thread_index, lane,
                                    simd_group);
    // The highest pivot whose measure exceeds the target and the lowest one whose measure does
    // not bound the new bracket. A pivot at lo never becomes hi: lo's measure may come from
    // another sum (the shards' masses, or the top-k selection's), and rounding must not empty the
    // bracket.
    for (uint pivot = 0; pivot < kPivots; ++pivot) {
      const uint count = scratch.total_counts[pivot];
      const float mass = Target::by_mass ? scratch.total_masses[pivot] : 0.0f;
      if (target.exceeded(count, mass)) {
        if (pivots[pivot] >= bracket.lo) {
          bracket.lo = pivots[pivot];
          bracket.lo_count = count;
          if (Target::by_mass)
            bracket.lo_mass = mass;
        }
      } else if (pivots[pivot] > bracket.lo && pivots[pivot] < bracket.hi) {
        bracket.hi = pivots[pivot];
        bracket.hi_count = count;
        if (Target::by_mass)
          bracket.hi_mass = mass;
      }
    }
    // Every thread reads the totals before the next pass rewrites them.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return bracket;
}

// The last token the target keeps in a bracket of at most one token per thread: its tokens are
// gathered, put in order by rank, and added to the measure at hi one at a time until it exceeds
// the target. A count search in logit order measures no masses while it narrows: its gather sums
// the mass at hi as measure_pivots would have, or, when every token from lo up fits the group,
// gathers them all and adds them from the first, which leaves the tokens it keeps ranked in the
// scratch (Selection::ranked).
template <class Order, class Target, typename L>
inline Selection resolve(TargetRow<L> row, Order order, OrderBoundary floor, Bracket bracket,
                         Target target, threadgroup VocabularyScratch &scratch, uint thread_index,
                         uint lane, uint simd_group) {
  constexpr bool unmeasured = Order::by_logit && !Target::by_mass;
  const bool ranked = unmeasured && bracket.lo_count <= kVocabularyThreads;
  const bool measures = unmeasured && !ranked;
  if (thread_index == 0)
    atomic_store_explicit(&scratch.gathered, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float above = 0.0f;
  for (uint token = thread_index; token < row.vocabulary; token += kVocabularyThreads) {
    const float value = row.logit(token);
    const uint key = order.key(value, token);
    if (key < bracket.lo || !row.admits(token) || !at_or_before(logit_key(value), token, floor))
      continue;
    if (!ranked && key >= bracket.hi) {
      if (measures)
        above += row.weight(value);
      continue;
    }
    const uint slot = atomic_fetch_add_explicit(&scratch.gathered, 1u, memory_order_relaxed);
    if (slot < kVocabularyThreads) {
      scratch.keys[slot] = key;
      scratch.ids[slot] = token;
    }
  }
  if (measures) {
    above = simd_sum(above);
    if (lane == 0)
      scratch.masses[simd_group][0] = above;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint gathered =
      min(atomic_load_explicit(&scratch.gathered, memory_order_relaxed), kVocabularyThreads);
  uint key = 0;
  uint id = 0;
  uint rank = 0;
  if (thread_index < gathered) {
    key = scratch.keys[thread_index];
    id = scratch.ids[thread_index];
    for (uint other = 0; other < gathered; ++other) {
      const uint other_key = scratch.keys[other];
      rank += other_key > key || (other_key == key && scratch.ids[other] < id) ? 1u : 0u;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < gathered) {
    scratch.keys[rank] = key;
    scratch.ids[rank] = id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    uint count = ranked ? 0 : bracket.hi_count;
    float mass = ranked ? 0.0f : bracket.hi_mass;
    if (measures) {
      mass = 0.0f;
      for (uint simd = 0; simd < kVocabularySimdgroups; ++simd)
        mass += scratch.masses[simd][0];
    }
    // Rounding may leave a mass target unreached: keep the whole bracket.
    Selection selection{{bracket.lo, 0xffffffffu},
                        bracket.lo_count,
                        Target::by_mass ? bracket.lo_mass : mass,
                        ranked};
    for (uint index = 0; index < gathered; ++index) {
      count += 1;
      mass += row.weight(order.logit(scratch.keys[index]));
      selection = {{scratch.keys[index], scratch.ids[index]}, count, mass, ranked};
      if (target.exceeded(count, mass))
        break;
    }
    scratch.selection = selection;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return scratch.selection;
}

// The last token the target keeps among the admitted tokens at or before the floor, starting
// from a bracket in logit order.
template <class Target, typename L>
inline Selection select_last(TargetRow<L> row, OrderBoundary floor, Bracket bracket,
                             Target target, threadgroup VocabularyScratch &scratch,
                             uint thread_index, uint lane, uint simd_group) {
  bracket = narrow(row, LogitOrder{}, floor, bracket, target, scratch, thread_index, lane,
                   simd_group);
  if (bracket.tokens() <= kVocabularyThreads)
    return resolve(row, LogitOrder{}, floor, bracket, target, scratch, thread_index, lane,
                   simd_group);
  // More tokens tie at one logit than a group gathers: the target keeps its first ones by id,
  // each of the same weight, after the mass above them, which a count search measures in one more
  // pass.
  const uint ties = bracket.tokens();
  const float weight = row.weight(key_logit(bracket.lo));
  float above;
  uint keep;
  if constexpr (Target::by_mass) {
    above = bracket.hi_mass;
    keep = 1u + uint(clamp((target.mass - above) / weight, 0.0f, float(ties - 1)));
  } else {
    above = measure_at_or_above(row, floor, bracket.hi, scratch, thread_index, lane, simd_group)
                .mass;
    keep = target.count + 1 - bracket.hi_count;
  }
  uint last = bracket.lo == floor.key ? floor.last : 0xffffffffu;
  if (keep < ties) {
    const TieOrder order{bracket.lo, row.vocabulary};
    const CountTarget first{keep - 1};
    Bracket tied{1, row.vocabulary + 1, ties, 0, 0.0f, 0.0f};
    tied = narrow(row, order, floor, tied, first, scratch, thread_index, lane, simd_group);
    last = resolve(row, order, floor, tied, first, scratch, thread_index, lane, simd_group)
               .last.last;
  }
  return {{bracket.lo, last}, bracket.hi_count + keep, above + float(keep) * weight, false};
}

// The last token of the row's distribution: the tokens that weigh at least min_p of the heaviest,
// then the top_k of those, then within those the top_p nucleus, whose mass is measured against the
// mass of what the two cuts before it keep. A top-k selection the scratch holds ranked is walked
// for the nucleus in that order, with the sums it was selected with; otherwise the nucleus is
// searched for. A row that keeps every admitted token ends at {kNoKey, 0}.
template <typename L>
inline OrderBoundary distribution_end(TargetRow<L> row, float min_p, uint top_k, float top_p,
                                      float mass, uint admitted,
                                      threadgroup VocabularyScratch &scratch, uint thread_index,
                                      uint lane, uint simd_group) {
  OrderBoundary end{kNoKey, 0};
  Bracket bracket{kLowestKey, logit_key(row.maximum) + 1, admitted, 0, mass, 0.0f};
  if (min_p > 0.0f) {
    // A token weighs min_p of the heaviest, which weighs 1, at the logit -temperature * log(min_p)
    // below the maximum. The tokens at or above that logit stay, the heaviest always.
    const uint lowest = logit_key(min(row.maximum + row.temperature * log(min_p), row.maximum));
    const Measure kept =
        measure_at_or_above(row, end, lowest, scratch, thread_index, lane, simd_group);
    if (kept.count < admitted) {
      end = {lowest, 0xffffffffu};
      bracket.lo = lowest;
      bracket.lo_count = kept.count;
      bracket.lo_mass = kept.mass;
    }
  }
  if (top_k < bracket.lo_count) {
    // A min_p cut ends on a whole key, which the bracket bounds from below, so this search needs
    // no floor.
    const Selection top = select_last(row, {kNoKey, 0}, bracket, CountTarget{top_k - 1}, scratch,
                                      thread_index, lane, simd_group);
    end = top.last;
    if (top.ranked && top_p < 1.0f) {
      // Every thread reads the selection before the walk rewrites it.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (thread_index == 0) {
        // Rounding may leave the nucleus unreached: keep the whole top-k.
        const float nucleus = top_p * top.mass;
        float walked = 0.0f;
        for (uint index = 0; index < top.count; ++index) {
          walked += row.weight(key_logit(scratch.keys[index]));
          if (walked > nucleus) {
            end = {scratch.keys[index], scratch.ids[index]};
            break;
          }
        }
        scratch.selection.last = end;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      return scratch.selection.last;
    }
    bracket.lo = end.key;
    bracket.lo_count = top.count;
    bracket.lo_mass = top.mass;
  }
  if (top_p < 1.0f)
    end = select_last(row, end, bracket, MassTarget{top_p * bracket.lo_mass}, scratch,
                      thread_index, lane, simd_group)
              .last;
  return end;
}

// Where a sampled row's distribution ends: its shards' masses, merged in shard order into the
// row's largest admitted logit, softmax denominator and admitted count, then the search. The
// record keeps the maximum and the end, which the draw reads. A top-k of 0 keeps every token.
template <typename L>
inline void search_row(TargetRow<L> row, device const TargetShardMass *masses, float min_p,
                       uint top_k, float top_p, device TargetVocabularyRow &record,
                       threadgroup VocabularyScratch &scratch, uint thread_index, uint lane,
                       uint simd_group) {
  const TargetShardMass merged = merge_masses(masses, kShards, row.temperature);
  row.maximum = merged.maximum;
  const OrderBoundary end =
      distribution_end(row, min_p, top_k == 0 ? 0xffffffffu : top_k, top_p, merged.sum,
                       merged.admitted, scratch, thread_index, lane, simd_group);
  if (thread_index == 0)
    record = {merged.maximum, end.key, end.last, 0.0f};
}

// Where the distribution of each selected row of the sampled lanes ends, one group of THREADS
// threads per row.
template <typename L>
[[kernel]] void dflash2_sample_search(device const L *logits [[buffer(0)]],
                                      device const VerifyLane *lanes [[buffer(1)]],
                                      device const TargetShardMass *partial_masses [[buffer(2)]],
                                      device TargetVocabularyRow *vocabulary_rows [[buffer(3)]],
                                      uint s [[threadgroup_position_in_grid]],
                                      uint thread_index [[thread_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup VocabularyScratch scratch;
  if (!samples(lanes, s))
    return;
  const VerifyLane params = lanes[s / TSAMPLE_ROWS];
  search_row(selected_row(logits, lanes, s), partial_masses + ulong(s) * kShards, params.min_p,
             params.top_k, params.top_p, vocabulary_rows[s], scratch, thread_index, lane,
             simd_group);
}

// ── The draw ───────────────────────────────────────────────────────────────────────────────────

// A token's weight in the row's distribution: its softmax weight if the row admits it at or
// before the distribution's end, else 0.
template <typename L>
inline float kept_weight(TargetRow<L> row, OrderBoundary end, uint token) {
  if (token >= row.vocabulary || !row.admits(token))
    return 0.0f;
  const float value = row.logit(token);
  return at_or_before(logit_key(value), token, end) ? row.weight(value) : 0.0f;
}

// The draft's candidates in one range of the draw, one bit per staged entry.
inline uint range_candidates(uint range, threadgroup DrawScratch &scratch) {
  uint held = 0;
  for (uint index = 0; index < kDraftCandidates; ++index)
    held |= scratch.draft_ranges[index] == range ? 1u << index : 0u;
  return held;
}

// One range's weight in the draw: the kept weight of its tokens other than the draft's
// candidates, plus its candidates' draw weights.
inline float range_weight(uint range, threadgroup DrawScratch &scratch) {
  float weight = scratch.range_rest[range];
  for (uint held = range_candidates(range, scratch); held; held &= held - 1)
    weight += scratch.draft_draw_weights[ctz(held)];
  return weight;
}

// The last token of a range with weight in the draw, which the draw takes when rounding leaves
// its target unreached.
inline uint range_last(uint range, threadgroup DrawScratch &scratch) {
  uint after = scratch.range_last[range];
  for (uint held = range_candidates(range, scratch); held; held &= held - 1) {
    const uint index = ctz(held);
    if (scratch.draft_draw_weights[index] > 0.0f)
      after = max(after, scratch.draft_ids[index] + 1);
  }
  return after - 1;
}

// Draws the row's token, shared by its groups; returns true in the group that finishes last, with
// the drawn token and the draft token's probability. Each kept token weighs its softmax weight;
// with a draft, each of the draft's candidates weighs its weight less its draft probability times
// the kept mass, never below zero, so the draw follows the residual distribution acceptance
// corrects a rejected draft token from (the whole distribution when nothing remains). Each
// simdgroup of the row's groups sums one range of the vocabulary without the candidates; the last
// group computes their weights once, adds the ranges in order and walks the one that holds the
// draw with prefix sums. A range's weight and its walk add the same token weights, so the range
// the draw picks always has a token to draw.
template <typename L>
inline bool vocabulary_draw(TargetRow<L> row, OrderBoundary end, bool drafted, uint draft_token,
                            device const uint *draft_ids, device const float *draft_probabilities,
                            float uniform, device coherent(device) TargetVocabularyRange *ranges,
                            device atomic_uint *arrivals, uint slice,
                            threadgroup DrawScratch &scratch, uint thread_index, uint lane,
                            uint simd_group, thread uint &drawn, thread float &draft_probability) {
  const uint range_tokens =
      (row.vocabulary + kVocabularyRanges * 32 - 1) / (kVocabularyRanges * 32) * 32;
  if (thread_index < kDraftCandidates) {
    // A candidate the draft lists twice counts once, as lookups find its first entry.
    uint id = drafted ? draft_ids[thread_index] : 0xffffffffu;
    for (uint earlier = 0; drafted && earlier < thread_index; ++earlier)
      id = draft_ids[earlier] == id ? 0xffffffffu : id;
    scratch.draft_ids[thread_index] = id;
    scratch.draft_ranges[thread_index] = id < row.vocabulary ? id / range_tokens : kVocabularyRanges;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint own = slice * kVocabularySimdgroups + simd_group;
  const uint begin = min(own * range_tokens, row.vocabulary);
  const uint finish = min(begin + range_tokens, row.vocabulary);
  const uint held = range_candidates(own, scratch);
  float rest = 0.0f;
  uint after = 0;
  for (uint first = begin; first < finish; first += 32) {
    const uint token = first + lane;
    float weight = token < finish ? kept_weight(row, end, token) : 0.0f;
    for (uint bits = held; bits; bits &= bits - 1)
      weight = scratch.draft_ids[ctz(bits)] == token ? 0.0f : weight;
    rest += weight;
    after = weight > 0.0f ? token + 1 : after;
  }
  rest = simd_sum(rest);
  after = simd_max(after);
  if (lane == 0)
    ranges[own] = {rest, after};
  if (!split_arrive_last(arrivals, kVocabularyGroups, thread_index, &scratch.arrival))
    return false;
  if (thread_index < kVocabularyRanges) {
    const TargetVocabularyRange measured = ranges[thread_index];
    scratch.range_rest[thread_index] = measured.rest;
    scratch.range_last[thread_index] = measured.after;
  }
  if (thread_index < kDraftCandidates) {
    scratch.draft_probabilities[thread_index] = drafted ? draft_probabilities[thread_index] : 0.0f;
    scratch.draft_weights[thread_index] = kept_weight(row, end, scratch.draft_ids[thread_index]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float kept_mass = 0.0f;
    for (uint range = 0; range < kVocabularyRanges; ++range)
      kept_mass += scratch.range_rest[range];
    for (uint index = 0; index < kDraftCandidates; ++index)
      kept_mass += scratch.draft_weights[index];
    bool residual = false;
    for (uint range = 0; drafted && range < kVocabularyRanges; ++range)
      residual = residual || scratch.range_rest[range] > 0.0f;
    for (uint index = 0; drafted && index < kDraftCandidates; ++index) {
      const float left = max(
          scratch.draft_weights[index] - scratch.draft_probabilities[index] * kept_mass, 0.0f);
      scratch.draft_draw_weights[index] = left;
      residual = residual || left > 0.0f;
    }
    // Without a draft, or with nothing left of its residual, the draw follows the distribution
    // itself.
    for (uint index = 0; !residual && index < kDraftCandidates; ++index)
      scratch.draft_draw_weights[index] = scratch.draft_weights[index];
    scratch.kept_mass = kept_mass;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < kVocabularyRanges)
    scratch.range_weights[thread_index] = range_weight(thread_index, scratch);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint range = 0; range < kVocabularyRanges; ++range)
      total += scratch.range_weights[range];
    const float target = uniform * total;
    float cumulative = 0.0f;
    uint range_drawn = 0;
    float range_target = 0.0f;
    for (uint range = 0; range < kVocabularyRanges; ++range) {
      const float weight = scratch.range_weights[range];
      if (!(weight > 0.0f))
        continue;
      // Rounding may leave the target unreached: the last range with weight.
      range_drawn = range;
      range_target = target - cumulative;
      cumulative += weight;
      if (cumulative > target)
        break;
    }
    scratch.drawn_range = range_drawn;
    scratch.drawn_target = range_target;
    scratch.drawn_token = range_last(range_drawn, scratch);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    const uint walked = scratch.drawn_range;
    const uint walk_begin = min(walked * range_tokens, row.vocabulary);
    const uint walk_finish = min(walk_begin + range_tokens, row.vocabulary);
    const uint walk_held = range_candidates(walked, scratch);
    const float target = scratch.drawn_target;
    float cumulative = 0.0f;
    for (uint first = walk_begin; first < walk_finish; first += 32) {
      const uint token = first + lane;
      float weight = token < walk_finish ? kept_weight(row, end, token) : 0.0f;
      for (uint bits = walk_held; bits; bits &= bits - 1) {
        const uint index = ctz(bits);
        if (scratch.draft_ids[index] == token)
          weight = scratch.draft_draw_weights[index];
      }
      const float prefix = simd_prefix_inclusive_sum(weight);
      const uint hit = simd_min(weight > 0.0f && cumulative + prefix > target ? lane : 32u);
      if (hit < 32) {
        if (lane == 0)
          scratch.drawn_token = first + hit;
        break;
      }
      cumulative += simd_broadcast(prefix, 31);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  drawn = scratch.drawn_token;
  draft_probability = drafted ? kept_weight(row, end, draft_token) / scratch.kept_mass : 0.0f;
  split_release(arrivals, thread_index);
  return true;
}

// The draw of each selected row of the sampled lanes, GROUPS groups of THREADS threads per row: a
// drafted row's draft token's probability and the correction a rejection takes or, for the row
// after the whole draft, its bonus token. Each row's token lands at its sampled row of `tokens`.
template <typename L>
[[kernel]] void dflash2_sample_draw(device const L *logits [[buffer(0)]],
                                    device const VerifyLane *lanes [[buffer(1)]],
                                    device TargetVocabularyRow *vocabulary_rows [[buffer(2)]],
                                    device const uint *drafted [[buffer(3)]],
                                    device const uint *draft_ids [[buffer(4)]],
                                    device const float *draft_probabilities [[buffer(5)]],
                                    device uint *tokens [[buffer(6)]],
                                    device coherent(device) TargetVocabularyRange *ranges
                                    [[buffer(7)]],
                                    device atomic_uint *arrivals [[buffer(8)]],
                                    uint group [[threadgroup_position_in_grid]],
                                    uint thread_index [[thread_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup DrawScratch scratch;
  const uint s = group / kVocabularyGroups;
  if (!samples(lanes, s))
    return;
  const uint batch = s / TSAMPLE_ROWS;
  const uint index = s % TSAMPLE_ROWS;
  const VerifyLane params = lanes[batch];
  device TargetVocabularyRow &record = vocabulary_rows[s];
  TargetRow<L> row = selected_row(logits, lanes, s);
  row.maximum = record.maximum;
  // A drafted row follows its draft token, the next verify input row.
  const bool is_drafted = index < params.drafts;
  const ulong position = ulong(batch) * kProposals + (is_drafted ? index : 0);
  uint token;
  float draft_probability;
  if (vocabulary_draw(row, {record.end_key, record.end_last}, is_drafted,
                      is_drafted ? drafted[position] : 0u,
                      draft_ids + position * kDraftCandidates,
                      draft_probabilities + position * kDraftCandidates,
                      draft_uniform(params.seed_lo, params.seed_hi, params.position,
                                    DRAFT_UNIFORM_CORRECTION),
                      ranges + ulong(s) * kVocabularyRanges, arrivals + s,
                      group % kVocabularyGroups, scratch, thread_index, lane, simd_group, token,
                      draft_probability) &&
      thread_index == 0) {
    tokens[params.sampled + index] = token;
    if (is_drafted)
      record.draft_probability = draft_probability;
  }
}

#define INST_TSAMPLE(tag, L)                                         \
  SCRATCHY_KERNEL(dflash2_sample_mass_##tag, dflash2_sample_mass<L>)     \
  SCRATCHY_KERNEL(dflash2_sample_search_##tag, dflash2_sample_search<L>) \
  SCRATCHY_KERNEL(dflash2_sample_draw_##tag, dflash2_sample_draw<L>)

INST_TSAMPLE(f16, half)
INST_TSAMPLE(bf16, bfloat)
