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
// params[0] = Lq, params[1] = num_heads, params[2] = head_dim.

#include <metal_stdlib>
using namespace metal;

template <typename SrcT, typename DstT>
inline void to_head_major_body(
    device DstT* out, const device SrcT* in, const device uint* p, uint2 gid)
{
    const uint lq = p[0];
    const uint h_ = p[1];
    const uint d_ = p[2];
    const uint i = gid.x;
    const uint h = gid.y;
    if (i >= lq || h >= h_) {
        return;
    }
    const device SrcT* s = in + (uint64_t(i) * h_ + h) * d_;   // token-major src
    device DstT* o = out + (uint64_t(h) * lq + i) * d_;          // head-major dst
    for (uint k = 0u; k < d_; k++) {
        o[k] = DstT(s[k]);
    }
}

template <typename SrcT, typename DstT>
inline void to_token_major_body(
    device DstT* out, const device SrcT* in, const device uint* p, uint2 gid)
{
    const uint lq = p[0];
    const uint h_ = p[1];
    const uint d_ = p[2];
    const uint i = gid.x;
    const uint h = gid.y;
    if (i >= lq || h >= h_) {
        return;
    }
    const device SrcT* s = in + (uint64_t(h) * lq + i) * d_;   // head-major src
    device DstT* o = out + (uint64_t(i) * h_ + h) * d_;          // token-major dst
    for (uint k = 0u; k < d_; k++) {
        o[k] = DstT(s[k]);
    }
}

kernel void q_convert_bf16_to_f16(
    device half*         out [[buffer(0)]],
    const device bfloat* in  [[buffer(1)]],
    const device uint*   p   [[buffer(2)]],
    uint2                gid [[thread_position_in_grid]])
{
    to_head_major_body<bfloat, half>(out, in, p, gid);
}

kernel void o_convert_f16_to_bf16(
    device bfloat*     out [[buffer(0)]],
    const device half* in  [[buffer(1)]],
    const device uint* p   [[buffer(2)]],
    uint2              gid [[thread_position_in_grid]])
{
    to_token_major_body<half, bfloat>(out, in, p, gid);
}

// f16 parity variants (correctness test / non-bf16 arches).
kernel void q_convert_f16_to_f16(
    device half*       out [[buffer(0)]],
    const device half* in  [[buffer(1)]],
    const device uint* p   [[buffer(2)]],
    uint2              gid [[thread_position_in_grid]])
{
    to_head_major_body<half, half>(out, in, p, gid);
}

kernel void o_convert_f16_to_f16(
    device half*       out [[buffer(0)]],
    const device half* in  [[buffer(1)]],
    const device uint* p   [[buffer(2)]],
    uint2              gid [[thread_position_in_grid]])
{
    to_token_major_body<half, half>(out, in, p, gid);
}

// bf16->bf16 layout variants (production hd512 path is all bf16).
kernel void q_convert_bf16_to_bf16(
    device bfloat*       out [[buffer(0)]],
    const device bfloat* in  [[buffer(1)]],
    const device uint*   p   [[buffer(2)]],
    uint2                gid [[thread_position_in_grid]])
{
    to_head_major_body<bfloat, bfloat>(out, in, p, gid);
}

kernel void o_convert_bf16_to_bf16(
    device bfloat*       out [[buffer(0)]],
    const device bfloat* in  [[buffer(1)]],
    const device uint*   p   [[buffer(2)]],
    uint2                gid [[thread_position_in_grid]])
{
    to_token_major_body<bfloat, bfloat>(out, in, p, gid);
}

// ── Production variants: Lq/nh/hd as function constants (no params buffer) ──
constant uint CVT_LQ [[function_constant(0)]];
constant uint CVT_NH [[function_constant(1)]];
constant uint CVT_HD [[function_constant(2)]];

// Q: [Lq, nh, hd] token-major bf16 -> [nh, Lq, hd] head-major bf16
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

// O: [nh, Lq, hd] head-major bf16 -> [Lq, nh, hd] token-major bf16
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
