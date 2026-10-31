// SPDX-License-Identifier: Apache-2.0
//
// What a DFlash2 cycle's kernels share (Splash ~/git/splash @ 1482f6bb runtime/metal/abi/Sampling.h):
// the uniforms a lane draws its cycle with, and the per-lane parameters Splash's batch parameter
// structs carry in their lane arrays (`SelectorBatchParams`, `VerifyRowsParams`,
// `AcceptBatchParams`, `TargetSamplingParams`). Mirrored by `scratchy_target_metal::dflash2` (the
// host writes them).

#pragma once

// The uniforms one lane draws a cycle with, in [0, 1): the draft's sampled proposal at position p
// draws DRAFT_UNIFORM_PROPOSALS + p; acceptance tests draft token p against
// DRAFT_UNIFORM_ACCEPTANCE + p; a sampled verify row draws its correction, or the bonus token
// after the whole draft, with DRAFT_UNIFORM_CORRECTION. Lane l's start at
// l * DRAFT_SAMPLING_UNIFORMS (Splash's SPLASH_UNIFORM_*, with its seven proposals).
#define DRAFT_UNIFORM_INITIAL 0u
#define DRAFT_UNIFORM_PROPOSALS 1u
#define DRAFT_UNIFORM_ACCEPTANCE (DRAFT_UNIFORM_PROPOSALS + 7u)
#define DRAFT_UNIFORM_CORRECTION (DRAFT_UNIFORM_ACCEPTANCE + 7u)
#define DRAFT_SAMPLING_UNIFORMS (DRAFT_UNIFORM_CORRECTION + 1u)

// One lane's policy for a cycle: its temperature and whether it samples (Splash's
// SelectorBatchParams `temperature`, `sampling_mask`), the most rows its verify takes (its
// retention limit, VerifyRowsParams `maximum`) and the tokens it may still emit
// (AcceptBatchParams `remaining`).
struct DraftLaneParams {
  float temperature;
  uint sampling;
  uint maximum;
  uint remaining;
};

// One lane of a step's verify, as Splash's AcceptBatchParams and TargetSamplingParams carry it:
// its first verify row among the step's sampled rows, the drafts its verify carried (0 for a lane
// that decoded without), that row's position, the tokens it may still emit (at least one) and its
// stop tokens; whether it samples, and its distribution's cuts — top-k (0: every token), its
// temperature, top-p and min-p — and whether it ignores end-of-sequence (never selects a stop
// token).
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
