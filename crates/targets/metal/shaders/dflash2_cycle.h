// SPDX-License-Identifier: Apache-2.0
//
// What a DFlash2 cycle's kernels share (Splash ~/git/splash @ 1482f6bb runtime/metal/abi/Sampling.h):
// the uniforms a lane draws its cycle with, made by the kernels that draw them, and the per-lane
// parameters Splash's batch parameter structs carry in their lane arrays (`SelectorBatchParams`,
// `VerifyRowsParams`, `AcceptBatchParams`, `TargetSamplingParams`). Mirrored by
// `scratchy_target_metal::dflash2` (the host writes them).

#pragma once

// The uniforms one lane draws a step with, in [0, 1): the draft's sampled proposal at position p
// draws DRAFT_UNIFORM_PROPOSALS + p; acceptance tests draft token p against
// DRAFT_UNIFORM_ACCEPTANCE + p; a sampled verify row draws its correction, or the bonus token
// after the whole draft, with DRAFT_UNIFORM_CORRECTION (Splash's SPLASH_UNIFORM_*, with its seven
// proposals; its initial draw is the target's sampler's here).
#define DRAFT_UNIFORM_PROPOSALS 1u
#define DRAFT_UNIFORM_ACCEPTANCE (DRAFT_UNIFORM_PROPOSALS + 7u)
#define DRAFT_UNIFORM_CORRECTION (DRAFT_UNIFORM_ACCEPTANCE + 7u)
#define DRAFT_SAMPLING_UNIFORMS (DRAFT_UNIFORM_CORRECTION + 1u)

// Draw `draw` of a lane's step: Splash's sampling uniform (runtime/model/Runtime.cpp:834-840
// `nextUniform`, SplitMix64 of the request's seed and a draw counter, its top 24 bits), made by
// the kernel that draws it rather than staged by the host each cycle. Its counter is the step's
// verify position times DRAFT_SAMPLING_UNIFORMS plus the draw, not a count of the request's draws
// (Splash's `rngCounter`): a step's position is past every earlier step's — each commits a token
// at least — so no two draws of a request share a counter, and a step run again from its position
// draws what it drew (what Splash restores `rngCounter` for, Runtime.cpp:2689-2739).
inline float draft_uniform(uint seed_lo, uint seed_hi, uint position, uint draw) {
  const ulong counter = ulong(position) * DRAFT_SAMPLING_UNIFORMS + draw;
  ulong value = ((ulong(seed_hi) << 32) | seed_lo) + counter * 0x9e3779b97f4a7c15ul;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ul;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebul;
  value ^= value >> 31;
  return float(value >> 40) * (1.0f / 16777216.0f);
}

// One lane's policy for a cycle: its temperature and whether it samples (Splash's
// SelectorBatchParams `temperature`, `sampling_mask`), the most rows its verify takes (its
// retention limit, VerifyRowsParams `maximum`) and the tokens it may still emit
// (AcceptBatchParams `remaining`); and the request's seed and the step's verify position, which
// its proposals' uniforms are drawn from (`draft_uniform`).
struct DraftLaneParams {
  float temperature;
  uint sampling;
  uint maximum;
  uint remaining;
  uint seed_lo;
  uint seed_hi;
  uint position;
};

// One lane of a step's verify, as Splash's AcceptBatchParams and TargetSamplingParams carry it:
// its first verify row among the step's sampled rows, the drafts its verify carried (0 for a lane
// that decoded without), that row's position, the tokens it may still emit (at least one) and its
// stop tokens; whether it samples, and its distribution's cuts — top-k (0: every token), its
// temperature, top-p and min-p — and whether it ignores end-of-sequence (never selects a stop
// token); and the request's seed, which with `position` its acceptance's and correction's uniforms
// are drawn from (`draft_uniform`).
struct VerifyLane {
  uint sampled;
  uint drafts;
  uint position;
  uint remaining;
  uint stop[2];
  uint sampling;
  uint top_k;
  float temperature;
  float top_p;
  float min_p;
  uint exclude_stop;
  uint seed_lo;
  uint seed_hi;
};

// A sampled verify row's selection over the whole vocabulary (Splash's TargetShardMass,
// TargetVocabularyRow, TargetVocabularyRange).
struct TargetShardMass {
  float maximum;
  float sum;
  uint admitted;
};

struct TargetVocabularyRow {
  float maximum;
  uint end_key;
  uint end_last;
  float draft_probability;
};

struct TargetVocabularyRange {
  float rest;
  uint after;
};

// One penalized sequence of a step (Splash's `SamplingPenaltyParams`, an entry of it): its first
// row in the step's logits and how many it has (a verify step's anchor and drafts, or one), its
// penalty words' row, its penalties (repetition with its saturated inverse, presence, frequency),
// and the step's input tokens of those rows: row r's context adds inputs 1..r, its drafts.
struct PenalizeEntry {
  uint logits_row;
  uint rows;
  uint table_row;
  float repetition;
  float repetition_inverse;
  float presence;
  float frequency;
  uint inputs[8];
};
