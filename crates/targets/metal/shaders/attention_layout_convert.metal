// SPDX-License-Identifier: Apache-2.0
//
// Layout + dtype converts for the gemma4 hd512 GLOBAL unfused-attention path.
// MPS GEMM is fp16-only, and per-head matmuls want each head's matrix
// contiguous, so:
//   q_convert: Q from q_proj  [Lq, num_heads, head_dim] (token-major, bf16)
//              -> q_f16        [num_heads, Lq, head_dim] (head-major, f16)
//   o_convert: PV output       [num_heads, Lq, head_dim] (head-major, f16)
//              -> attn output  [Lq, num_heads, head_dim] (token-major, bf16)
// so the rest of the model (o_proj) sees the exact layout/dtype gqa_shared
// produced. One thread per (token i, head h), copying head_dim elements.
//
// Lq / num_heads / head_dim are compiled in (slots 0 / 1 / 2).

#include <metal_stdlib>
#include "baked.h"
using namespace metal;

SCRATCHY_CONSTANT(uint, CVT_LQ, 0);
SCRATCHY_CONSTANT(uint, CVT_NH, 1);
SCRATCHY_CONSTANT(uint, CVT_HD, 2);

// Q: [Lq, nh, hd] token-major bf16 -> [nh, Lq, hd] head-major bf16
#if SCRATCHY_COMPILES(q_convert_prod_bf16)
kernel void q_convert_prod_bf16(
    device bfloat*       out [[buffer(0)]],
    const device bfloat* in  [[buffer(1)]],
    uint2                gid [[thread_position_in_grid]])
{
    const uint i = gid.x, h = gid.y;
    if (i >= CVT_LQ || h >= CVT_NH) return;
    const device bfloat* s = in + (uint64_t(i) * CVT_NH + h) * CVT_HD;
    device bfloat* o = out + (uint64_t(h) * CVT_LQ + i) * CVT_HD;
    for (uint k = 0u; k < CVT_HD; k++) o[k] = s[k];
}
#endif

// O: [nh, Lq, hd] head-major bf16 -> [Lq, nh, hd] token-major bf16
#if SCRATCHY_COMPILES(o_convert_prod_bf16)
kernel void o_convert_prod_bf16(
    device bfloat*       out [[buffer(0)]],
    const device bfloat* in  [[buffer(1)]],
    uint2                gid [[thread_position_in_grid]])
{
    const uint i = gid.x, h = gid.y;
    if (i >= CVT_LQ || h >= CVT_NH) return;
    const device bfloat* s = in + (uint64_t(h) * CVT_LQ + i) * CVT_HD;
    device bfloat* o = out + (uint64_t(i) * CVT_NH + h) * CVT_HD;
    for (uint k = 0u; k < CVT_HD; k++) o[k] = s[k];
}
#endif
