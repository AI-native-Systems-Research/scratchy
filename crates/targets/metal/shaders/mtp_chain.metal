// SPDX-License-Identifier: Apache-2.0
//
// mtp_chain — between a multi-token-prediction head's passes over one speculative step, the next
// pass's inputs, picked on the device, so the passes run in the step's command buffer with no
// host round trip between them.
//
// One threadgroup per drafting sequence `s`, after pass `depth` (from 1) ran and argmaxed. The
// pass's row it reads: after pass 1 the sequence's newest — its first token row plus the drafts
// the target accepted, the leading token rows whose target token is the draft their successor row
// verifies (recorded in `accepted`) — and after a later pass, row `s`. That row's argmax is draft
// `depth` and the next pass's input token; its final hidden state is the next pass's target hidden
// row `s`; the next pass's position, slot and used KV length — and an MRoPE head's rotary cos/sin
// row for that position — are the host's for the accepted count.
//
// Bindings:
//   0  picked       uint  the pass's argmax: after pass 1 over its argmax window, else one a row
//   1  hidden       2-byte elements, [rows, MTP_HIDDEN]: the pass's final hidden states
//   2  target       uint  [step rows]: the target's token at each row of the step
//   3  drafted      uint  [step rows]: the token each row's successor reads, the draft it verifies
//   4  seqs         uint  [n, 3]: per sequence, its first token row in the step, its token rows,
//                         and its first token row in pass 1's rows
//   5  accepted     uint  [n]: written after pass 1, read after a later one
//   6  next         uint  [n, MTP_DRAFTS + 1, 3]: per sequence and accepted count, the next pass's
//                         position, slot and used KV length
//   7  next_ids     uint  [n]
//   8  next_pos     uint  [n]
//   9  next_slot    uint  [n]
//   10 next_used    uint  [n]
//   11 next_hidden  2-byte elements, [n, MTP_HIDDEN]
//   12 drafts       uint  [n, MTP_DRAFTS]
//   13 depth        uint
//   14 window_start uint: the first of pass 1's rows its argmax ran
//   15 rope         2-byte elements, [n, MTP_DRAFTS + 1, MTP_ROPE]: per sequence and accepted
//                   count, the next pass's rotary cos/sin row (MTP_ROPE 0: none)
//   16 next_rope    2-byte elements, [n, MTP_ROPE]
//
// Dispatch: n threadgroups of 256 threads; thread 0 picks, every thread copies the rows.

#include <metal_stdlib>
#include "baked.h"
using namespace metal;

SCRATCHY_CONSTANT(uint, MTP_HIDDEN, 0);
SCRATCHY_CONSTANT(uint, MTP_DRAFTS, 1);
SCRATCHY_CONSTANT(uint, MTP_ROPE, 2);

kernel void mtp_chain(
    device const uint*   picked       [[buffer(0)]],
    device const ushort* hidden       [[buffer(1)]],
    device const uint*   target       [[buffer(2)]],
    device const uint*   drafted      [[buffer(3)]],
    device const uint*   seqs         [[buffer(4)]],
    device       uint*   accepted     [[buffer(5)]],
    device const uint*   next         [[buffer(6)]],
    device       uint*   next_ids     [[buffer(7)]],
    device       uint*   next_pos     [[buffer(8)]],
    device       uint*   next_slot    [[buffer(9)]],
    device       uint*   next_used    [[buffer(10)]],
    device       ushort* next_hidden  [[buffer(11)]],
    device       uint*   drafts       [[buffer(12)]],
    constant     uint&   depth        [[buffer(13)]],
    constant     uint&   window_start [[buffer(14)]],
    device const ushort* rope         [[buffer(15)]],
    device       ushort* next_rope    [[buffer(16)]],
    uint s   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
    threadgroup uint row_shared;
    threadgroup uint kept_shared;
    if (tid == 0) {
        uint row = s;
        uint kept = 0;
        if (depth == 1) {
            const uint first = seqs[s * 3], rows = seqs[s * 3 + 1];
            while (kept + 1 < rows && target[first + kept] == drafted[first + kept]) {
                ++kept;
            }
            accepted[s] = kept;
            row = seqs[s * 3 + 2] + kept;
        } else {
            kept = accepted[s];
        }
        const uint token = picked[depth == 1 ? row - window_start : row];
        drafts[s * MTP_DRAFTS + depth - 1] = token;
        device const uint* at = next + (s * (MTP_DRAFTS + 1) + kept) * 3;
        next_ids[s] = token;
        next_pos[s] = at[0];
        next_slot[s] = at[1];
        next_used[s] = at[2];
        row_shared = row;
        kept_shared = kept;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const ushort* from = hidden + ulong(row_shared) * MTP_HIDDEN;
    device ushort* to = next_hidden + ulong(s) * MTP_HIDDEN;
    for (uint i = tid; i < MTP_HIDDEN; i += tg) {
        to[i] = from[i];
    }
    device const ushort* rope_row = rope + ulong(s * (MTP_DRAFTS + 1) + kept_shared) * MTP_ROPE;
    for (uint i = tid; i < MTP_ROPE; i += tg) {
        next_rope[ulong(s) * MTP_ROPE + i] = rope_row[i];
    }
}
