// SPDX-License-Identifier: Apache-2.0
//
// Splash's acceptance of a DFlash2 cycle's verify (~/git/splash @ 1482f6bb
// runtime/metal/kernels/decode/sampling.metal `decode_accept_dflash`, `accept_greedy_lane`,
// `accept_sampled_lane`, `sparse_lookup`, `finish_acceptance`), and the block its drafter drafts
// next: the lane's last retained token, at the position after the rows it retained
// (runtime/model/Runtime.cpp, the next cycle's anchor and `item.logicalPosition`). With these
// changes, and no other:
//
//   - A greedy lane's target tokens are the argmax the target's sampler picked for its verify rows;
//     a sampled lane's, the draws `dflash2_target_sampling.metal` made (a correction a row, the
//     bonus last), with each drafted row's target probability of its draft token.
//   - A lane's parameters are its `VerifyLane` (dflash2_cycle.h), Splash's `AcceptBatchParams`
//     lane arrays and sampling mask.
//   - The next block's inputs are written on the device: Splash's host lays them out from the
//     retained count it reads back after the cycle.
//   - A lane verifies the drafts it carried, none for a lane that decoded without: it retains its
//     one row.
//
// Baked constants:
//   ACCEPT_ROWS — the block: an anchor and its proposals (SPLASH_DRAFT_QUERY_ROWS)
//   ACCEPT_MASK — the token a proposal row embeds (`dflash_config.mask_token_id`)
//
// Dispatch: a threadgroup of one thread a lane.

#include <metal_stdlib>
#include "baked.h"
#include "dflash2_cycle.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, ACCEPT_ROWS, 0);
SCRATCHY_CONSTANT(uint, ACCEPT_MASK, 1);

constant constexpr uint ACCEPT_PROPOSALS = ACCEPT_ROWS - 1;
// SPLASH_DRAFT_CANDIDATES.
constant constexpr uint kDraftCandidates = 16;

inline float sparse_lookup(device const uint *ids, device const float *probabilities, uint count,
                           uint token) {
  for (uint i = 0; i < count; ++i) {
    if (ids[i] == token)
      return probabilities[i];
  }
  return 0.0f;
}

// Keeps at most `remaining` of the accepted tokens plus the correction, cut after the first stop
// token, and records the retained and accepted counts.
inline void finish_acceptance(device const uint *tokens, uint accepted, VerifyLane params,
                              device uint &retained, device uint &accepted_count) {
  accepted_count = accepted;
  uint kept = min(accepted + 1, params.remaining);
  for (uint i = 0; i < kept; ++i) {
    if (tokens[i] == params.stop[0] || tokens[i] == params.stop[1]) {
      kept = i + 1;
      break;
    }
  }
  retained = kept;
}

// A sampled lane's verify rows carry their draft tokens' target probabilities
// (TargetVocabularyRow), and its output tokens their draws: the correction a rejection takes or,
// for the last row, the bonus token. Acceptance overwrites the accepted rows with the draft
// tokens, so row `accepted` keeps its draw.
inline void accept_sampled_lane(device const uint *draft_tokens, device const uint *draft_ids,
                                device const float *draft_probs,
                                device const TargetVocabularyRow *target_rows,
                                device const float *uniforms, device uint *output_tokens,
                                VerifyLane params, device uint &retained,
                                device uint &accepted_count) {
  uint accepted = 0;
  while (accepted < params.drafts) {
    const uint token = draft_tokens[accepted];
    const float q = sparse_lookup(draft_ids + accepted * kDraftCandidates,
                                  draft_probs + accepted * kDraftCandidates, kDraftCandidates,
                                  token);
    const float p = target_rows[accepted].draft_probability;
    if (!(uniforms[DRAFT_UNIFORM_ACCEPTANCE + accepted] * q < p))
      break;
    output_tokens[accepted] = token;
    ++accepted;
  }
  finish_acceptance(output_tokens, accepted, params, retained, accepted_count);
}

inline void accept_greedy_lane(device const uint *draft_tokens, device uint *target_tokens,
                               VerifyLane params, device uint &retained,
                               device uint &accepted_count) {
  uint accepted = 0;
  while (accepted < params.drafts && draft_tokens[accepted] == target_tokens[accepted])
    ++accepted;
  finish_acceptance(target_tokens, accepted, params, retained, accepted_count);
}

[[kernel]] void dflash2_accept(device const VerifyLane *lanes [[buffer(0)]],
                               device const uint *drafted [[buffer(1)]],
                               device uint *tokens [[buffer(2)]],
                               device uint *retained [[buffer(3)]],
                               device uint *accepted_count [[buffer(4)]],
                               device uint *block_ids [[buffer(5)]],
                               device uint *block_positions [[buffer(6)]],
                               device const uint *draft_ids [[buffer(7)]],
                               device const float *draft_probs [[buffer(8)]],
                               device const TargetVocabularyRow *target_rows [[buffer(9)]],
                               device const float *uniforms [[buffer(10)]],
                               uint lane [[threadgroup_position_in_grid]]) {
  const VerifyLane params = lanes[lane];
  device const uint *draft = drafted + lane * ACCEPT_PROPOSALS;
  device uint *target = tokens + params.sampled;
  if (params.sampling != 0) {
    const ulong candidates = ulong(lane) * ACCEPT_PROPOSALS * kDraftCandidates;
    accept_sampled_lane(draft, draft_ids + candidates, draft_probs + candidates,
                        target_rows + lane * ACCEPT_ROWS, uniforms + lane * DRAFT_SAMPLING_UNIFORMS,
                        target, params, retained[lane], accepted_count[lane]);
  } else {
    accept_greedy_lane(draft, target, params, retained[lane], accepted_count[lane]);
  }
  const uint kept = retained[lane];
  device uint *ids = block_ids + lane * ACCEPT_ROWS;
  device uint *positions = block_positions + lane * ACCEPT_ROWS;
  for (uint row = 0; row < ACCEPT_ROWS; ++row) {
    ids[row] = row == 0 ? target[kept - 1] : ACCEPT_MASK;
    positions[row] = params.position + kept + row;
  }
}
