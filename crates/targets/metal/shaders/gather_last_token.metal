// SPDX-License-Identifier: Apache-2.0
//
// gather_last_token / scatter_first_to_last_row — sampling-position
// slice kernels around the lm_head GEMM. Index-driven via the
// existing `cu_seqlens_q` runtime buffer. For seq i in [0, num_seqs):
//   src_row[i] = cu_seqlens_q[i + 1] - 1
//
//   gather:  swap rows i and src_row[i]   for i = 0 .. num_seqs - 1
//   scatter: swap rows i and src_row[i]   for i = num_seqs - 1 .. 0
//
// Both swap rows in place, and in a mixed step a short sequence's
// src_row can be another sequence's i. cu_seqlens_q is increasing, so
// src_row[i] >= i and src_row[i] > src_row[k] for k < i: the gather's
// swap i reads a row src_row[i] no earlier swap touched, and no later
// swap touches row i, so row i ends holding the original src_row[i].
// The scatter is the same swaps in reverse, the gather undone: on the
// GEMM's output it lands row i on src_row[i]; on the gathered hidden
// states it puts every row back, so they read as the backbone wrote
// them after the lm_head ran. Columns are independent, so 1D dispatch:
// tg = (ceil(GATHER_ROW_STRIDE/256), 1, 1).
//
// `GATHER_ROW_STRIDE` (constant slot 0) is hidden_size (gather, and the
// scatter over the hidden states) or vocab_size (the scatter over the
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
    device T_act* hidden                 [[buffer(0)]],
    device const uint* cu_seqlens_q      [[buffer(1)]],
    device const uint* num_seqs_buf      [[buffer(2)]],
    uint h [[thread_position_in_grid]])
{
    if (h >= GATHER_ROW_STRIDE) return;
    const uint num_seqs = num_seqs_buf[0];
    for (uint row = 0; row < num_seqs; ++row) {
        swap_rows(hidden, row, cu_seqlens_q[row + 1] - 1, h);
    }
}

template <typename T_act>
[[kernel]] void scatter_first_to_last_row(
    device T_act* rows                   [[buffer(0)]],
    device const uint* cu_seqlens_q      [[buffer(1)]],
    device const uint* num_seqs_buf      [[buffer(2)]],
    uint h [[thread_position_in_grid]])
{
    if (h >= GATHER_ROW_STRIDE) return;
    for (uint row = num_seqs_buf[0]; row-- > 0;) {
        swap_rows(rows, row, cu_seqlens_q[row + 1] - 1, h);
    }
}

SCRATCHY_KERNEL(gather_last_token_f16_specialized, gather_last_token<half>)
SCRATCHY_KERNEL(gather_last_token_bf16_specialized, gather_last_token<bfloat>)
SCRATCHY_KERNEL(scatter_first_to_last_row_f16_specialized, scatter_first_to_last_row<half>)
SCRATCHY_KERNEL(scatter_first_to_last_row_bf16_specialized, scatter_first_to_last_row<bfloat>)
