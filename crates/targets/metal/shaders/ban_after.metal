// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

#include <metal_stdlib>
#include "baked.h"
using namespace metal;

// ---------------------------------------------------------------------------
// ban_after — a two-token bad word's ban, read on the device.
//
// A bad word `[after, banned]` bans `banned` at a sampled row whose token before
// it is `after`: the row's own input token (a decode row's last token, a verify
// row's draft before it). That token is in the forward's `input_ids`, on the
// device even when the step that sampled it has not finished on the host — so the
// ban needs no host read and a deferred or pipelined step keeps it.
//
// Per job `(row, input, after, banned)`: `logits[row][banned] = -inf` iff `after`
// is `BAN_ALWAYS` (a one-token bad word, or one the host found banned) or
// `input_ids[input] == after`. One thread per job; a job writes one logit, and two
// jobs that write the same one write the same value, so the in-place write is
// hazard-free.
//
// Bindings (must match `ban_after::encode_ban_after_into_mtl4`):
//   buffer(0) = logits     [total_n, BAN_VOCAB]  half / bfloat (in/out)
//   buffer(1) = input_ids  [num_tokens]          uint
//   buffer(2) = jobs       [1 + count * 4]       uint: count, then each job's four words
// ---------------------------------------------------------------------------

SCRATCHY_CONSTANT(uint, BAN_VOCAB, 0);

// `after` of a job that bans whatever the token before it.
constant constexpr uint BAN_ALWAYS = 0xFFFFFFFFu;

template <typename T>
[[kernel]] void ban_after(
    device       T*    logits    [[buffer(0)]],
    device const uint* input_ids [[buffer(1)]],
    device const uint* jobs      [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= jobs[0]) return;
    device const uint* job = jobs + 1u + gid * 4u;
    uint row = job[0], input = job[1], after = job[2], banned = job[3];
    bool fires = after == BAN_ALWAYS || input_ids[input] == after;
    if (fires && banned < BAN_VOCAB) {
        logits[row * BAN_VOCAB + banned] = T(-INFINITY);
    }
}

SCRATCHY_KERNEL(ban_after_f16, ban_after<half>)
SCRATCHY_KERNEL(ban_after_bf16, ban_after<bfloat>)
