// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

#include <metal_stdlib>
#include "baked.h"
using namespace metal;

// ---------------------------------------------------------------------------
// grammar_mask — constrained-decoding logit mask.
//
// Applies a per-request grammar allow-set to the logits in place: every
// token NOT permitted by the request's grammar FSM at the current step
// is forced to -inf so the downstream greedy `argmax` can only pick an
// allowed token. The allowed set is supplied as a dense bitset (one bit
// per vocab entry, set = allowed), one bitset row per masked request.
//
// Only the rows that actually have an active grammar are dispatched; the
// `rows` table maps grammar-row gid -> the logits batch row to mask, so a
// mixed batch (some requests constrained, others free) is handled by
// dispatching exactly `num_rows` threadgroups and leaving every other
// logits row untouched.
//
// Dispatch: threadgroups (num_rows, 1, 1), threads_per_threadgroup
// (TG_SIZE, 1, 1). One threadgroup per constrained row; threads in a
// group stride over the vocab axis. No reduction / no threadgroup
// memory — each (row, token) is touched by exactly one thread, so the
// in-place write is hazard-free.
//
// Bindings (must match `grammar_mask::encode_grammar_mask_into_mtl4`):
//   buffer(0) = logits      [total_n, GRAMMAR_VOCAB]               half / bfloat (in/out)
//   buffer(1) = allow_bits  [num_rows * GRAMMAR_WORDS_PER_ROW]  uint  (1 bit/token, set=allowed)
//   buffer(2) = rows        [num_rows]                          uint  (grammar row -> logits row)
// ---------------------------------------------------------------------------

SCRATCHY_CONSTANT(uint, GRAMMAR_VOCAB, 0);
SCRATCHY_CONSTANT(uint, GRAMMAR_WORDS_PER_ROW, 1);

template <typename T>
[[kernel]] void grammar_mask(
    device       T*     logits     [[buffer(0)]],
    device const uint*  allow_bits [[buffer(1)]],
    device const uint*  rows       [[buffer(2)]],
    uint  gid [[threadgroup_position_in_grid]],
    uint  tid [[thread_position_in_threadgroup]],
    uint  tg  [[threads_per_threadgroup]])
{
    uint logits_row = rows[gid];
    device       T*    row  = logits     + uint(logits_row) * GRAMMAR_VOCAB;
    device const uint* bits = allow_bits + uint(gid) * GRAMMAR_WORDS_PER_ROW;
    uint bitset_cap = GRAMMAR_WORDS_PER_ROW * 32u;

    for (uint v = tid; v < GRAMMAR_VOCAB; v += tg) {
        // Slots beyond the bitset (vocab padding the lm_head emits past the
        // tokenizer vocab) are never grammar-allowed → always -inf. Guarding
        // here also keeps the bits[] read in bounds when vocab > bitset_cap.
        bool allowed = (v < bitset_cap) && (((bits[v >> 5] >> (v & 31u)) & 1u) != 0u);
        if (!allowed) {
            row[v] = T(-INFINITY);
        }
    }
}

SCRATCHY_KERNEL(grammar_mask_f16, grammar_mask<half>)
SCRATCHY_KERNEL(grammar_mask_bf16, grammar_mask<bfloat>)
