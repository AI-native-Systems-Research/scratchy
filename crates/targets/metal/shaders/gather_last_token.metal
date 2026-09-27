// SPDX-License-Identifier: Apache-2.0
//
// gather_last_token / scatter_first_to_last_row — sampling-position
// slice kernels around the lm_head GEMM. Index-driven via the
// existing `cu_seqlens_q` runtime buffer. For seq i in [0, num_seqs):
//   src_row[i] = cu_seqlens_q[i + 1] - 1
//
//   gather:  hidden[i, :]      <-  hidden[src_row[i], :]   for i in 0..num_seqs
//   scatter: logits[src_row[i], :] <- logits[i, :]         for i in 0..num_seqs
//
// Both move rows in place, and in a mixed step a short sequence's
// src_row can be another sequence's i. One thread per column walks the
// sequences in order, which is safe: cu_seqlens_q is non-decreasing, so
// src_row[i] >= i. The gather walks i up and reads row src_row[i] >= i
// before any row >= i is written; the scatter walks i down and reads
// row i before any row <= i is written (every earlier write went to a
// src_row[k] >= k > i). Columns are independent, so 1D dispatch:
// tg = (ceil(GATHER_ROW_STRIDE/256), 1, 1).
//
// `GATHER_ROW_STRIDE` (function constant 0) is hidden_size (gather)
// or vocab_size (scatter).

#include <metal_stdlib>
using namespace metal;

constant uint GATHER_ROW_STRIDE [[function_constant(0)]];

#define GATHER_KERNEL(NAME, T_act) \
kernel void NAME( \
    device T_act* hidden                 [[buffer(0)]], \
    device const uint* cu_seqlens_q      [[buffer(1)]], \
    device const uint* num_seqs_buf      [[buffer(2)]], \
    uint h [[thread_position_in_grid]]) \
{ \
    if (h >= GATHER_ROW_STRIDE) return; \
    const uint num_seqs = num_seqs_buf[0]; \
    for (uint dst_row = 0; dst_row < num_seqs; ++dst_row) { \
        const uint src_row = cu_seqlens_q[dst_row + 1] - 1; \
        if (src_row == dst_row) continue; \
        hidden[(ulong)dst_row * (ulong)GATHER_ROW_STRIDE + (ulong)h] = \
            hidden[(ulong)src_row * (ulong)GATHER_ROW_STRIDE + (ulong)h]; \
    } \
}

#define SCATTER_KERNEL(NAME, T_act) \
kernel void NAME( \
    device T_act* logits                 [[buffer(0)]], \
    device const uint* cu_seqlens_q      [[buffer(1)]], \
    device const uint* num_seqs_buf      [[buffer(2)]], \
    uint h [[thread_position_in_grid]]) \
{ \
    if (h >= GATHER_ROW_STRIDE) return; \
    const uint num_seqs = num_seqs_buf[0]; \
    for (uint src_row = num_seqs; src_row-- > 0;) { \
        const uint dst_row = cu_seqlens_q[src_row + 1] - 1; \
        if (src_row == dst_row) continue; \
        logits[(ulong)dst_row * (ulong)GATHER_ROW_STRIDE + (ulong)h] = \
            logits[(ulong)src_row * (ulong)GATHER_ROW_STRIDE + (ulong)h]; \
    } \
}

GATHER_KERNEL(gather_last_token_f16_specialized,  half)
GATHER_KERNEL(gather_last_token_bf16_specialized, bfloat)
SCATTER_KERNEL(scatter_first_to_last_row_f16_specialized,  half)
SCATTER_KERNEL(scatter_first_to_last_row_bf16_specialized, bfloat)
