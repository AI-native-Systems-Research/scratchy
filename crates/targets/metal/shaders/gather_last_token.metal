// SPDX-License-Identifier: Apache-2.0
//
// gather_last_token / scatter_first_to_last_row — the sampled rows'
// kernels around the lm_head. Index-driven by the step's sampled rows
// (`sampled`, ascending: each sequence's last row, every row of a
// sequence verifying drafts), `num_sampled` of them:
//
//   gather:  swap rows i and sampled[i]   for i = 0 .. num_sampled - 1
//   scatter: swap rows i and sampled[i]   for i = num_sampled - 1 .. 0
//
// Both swap rows in place, and a sampled row can be another's i (a
// decode's row inside a mixed step's front). `sampled` is strictly
// increasing, so sampled[i] >= i and sampled[i] > sampled[k] for k < i:
// the gather's swap i reads a row sampled[i] no earlier swap touched,
// and no later swap touches row i, so row i ends holding the original
// sampled[i]. The scatter is the same swaps in reverse, the gather
// undone: on the gathered hidden states it puts every row back, so they
// read as the backbone wrote them after the lm_head ran. The gather over
// every row's logits keeps the sampled rows', in order, at the front.
// Columns are independent, so 1D dispatch:
// tg = (ceil(GATHER_ROW_STRIDE/256), 1, 1).
//
// `GATHER_ROW_STRIDE` (constant slot 0) is hidden_size (the gather and
// the scatter over the hidden states) or vocab_size (the gather over the
// logits).

#include <metal_stdlib>
#include "baked.h"
using namespace metal;

SCRATCHY_CONSTANT(uint, GATHER_ROW_STRIDE, 0);

template <typename T_act>
inline void swap_rows(device T_act* rows, uint a, uint b, uint h) {
    const ulong at_a = (ulong)a * (ulong)GATHER_ROW_STRIDE + (ulong)h;
    const ulong at_b = (ulong)b * (ulong)GATHER_ROW_STRIDE + (ulong)h;
    const T_act moved = rows[at_a];
    rows[at_a] = rows[at_b];
    rows[at_b] = moved;
}

template <typename T_act>
[[kernel]] void gather_last_token(
    device T_act* rows                   [[buffer(0)]],
    device const uint* sampled           [[buffer(1)]],
    device const uint* num_sampled_buf   [[buffer(2)]],
    uint h [[thread_position_in_grid]])
{
    if (h >= GATHER_ROW_STRIDE) return;
    const uint num_sampled = num_sampled_buf[0];
    for (uint row = 0; row < num_sampled; ++row) {
        swap_rows(rows, row, sampled[row], h);
    }
}

template <typename T_act>
[[kernel]] void scatter_first_to_last_row(
    device T_act* rows                   [[buffer(0)]],
    device const uint* sampled           [[buffer(1)]],
    device const uint* num_sampled_buf   [[buffer(2)]],
    uint h [[thread_position_in_grid]])
{
    if (h >= GATHER_ROW_STRIDE) return;
    for (uint row = num_sampled_buf[0]; row-- > 0;) {
        swap_rows(rows, row, sampled[row], h);
    }
}

SCRATCHY_KERNEL(gather_last_token_f16_specialized, gather_last_token<half>)
SCRATCHY_KERNEL(gather_last_token_bf16_specialized, gather_last_token<bfloat>)
SCRATCHY_KERNEL(scatter_first_to_last_row_f16_specialized, scatter_first_to_last_row<half>)
SCRATCHY_KERNEL(scatter_first_to_last_row_bf16_specialized, scatter_first_to_last_row<bfloat>)
