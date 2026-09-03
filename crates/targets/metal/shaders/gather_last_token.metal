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
// 2D dispatch: tg = (ceil(GATHER_ROW_STRIDE/256), bucket_m, 1).
// Y dim is sized for the worst-case (bucket_m); threads with
// tid.y >= num_seqs early-out so the actual M = num_seqs at runtime.
//
// In-place safety: cu_seqlens_q is monotonically non-decreasing,
// so src_row[i] >= i (each seq contributes at least one token).
// Gather writes row i after reading row >= i (no previously-written
// destination row gets re-read as a source). Scatter is symmetric
// and safe when src_row[i] >= num_seqs for all i (the common case
// at prefill — every seq has more than `num_seqs` total tokens).
// Edge case where some seq has <= num_seqs tokens (e.g. tiny prompts
// mixed with longer prompts) can race; document and fix later.
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
    uint2 tid [[thread_position_in_grid]]) \
{ \
    const uint h = tid.x; \
    const uint dst_row = tid.y; \
    if (h >= GATHER_ROW_STRIDE) return; \
    const uint num_seqs = num_seqs_buf[0]; \
    if (dst_row >= num_seqs) return; \
    const uint src_row = cu_seqlens_q[dst_row + 1] - 1; \
    if (src_row == dst_row) return; \
    hidden[(ulong)dst_row * (ulong)GATHER_ROW_STRIDE + (ulong)h] = \
        hidden[(ulong)src_row * (ulong)GATHER_ROW_STRIDE + (ulong)h]; \
}

#define SCATTER_KERNEL(NAME, T_act) \
kernel void NAME( \
    device T_act* logits                 [[buffer(0)]], \
    device const uint* cu_seqlens_q      [[buffer(1)]], \
    device const uint* num_seqs_buf      [[buffer(2)]], \
    uint2 tid [[thread_position_in_grid]]) \
{ \
    const uint h = tid.x; \
    const uint src_row = tid.y; \
    if (h >= GATHER_ROW_STRIDE) return; \
    const uint num_seqs = num_seqs_buf[0]; \
    if (src_row >= num_seqs) return; \
    const uint dst_row = cu_seqlens_q[src_row + 1] - 1; \
    if (src_row == dst_row) return; \
    logits[(ulong)dst_row * (ulong)GATHER_ROW_STRIDE + (ulong)h] = \
        logits[(ulong)src_row * (ulong)GATHER_ROW_STRIDE + (ulong)h]; \
}

GATHER_KERNEL(gather_last_token_f16_specialized,  half)
GATHER_KERNEL(gather_last_token_bf16_specialized, bfloat)
SCATTER_KERNEL(scatter_first_to_last_row_f16_specialized,  half)
SCATTER_KERNEL(scatter_first_to_last_row_bf16_specialized, bfloat)
