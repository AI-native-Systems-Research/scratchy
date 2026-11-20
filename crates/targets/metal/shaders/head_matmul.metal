// SPDX-License-Identifier: Apache-2.0
//
// A matmul per head (`MetalStep::HeadMatmul`): head `h` of each input row (the
// first `K` lanes of its `IN_STRIDE`-wide part) times weight block `h`, into head
// `h`'s `N_HEAD` output lanes. The weight is every head's block of `N_HEAD` rows
// in order, `HEADS * N_HEAD` rows of `K`: MLX's `MultiLinear` `[heads, out, in]`.
// MLA's query absorption (`embed_q`) and value decompression (`unembed_out`).
//
// Constants (compiled in; `HeadMatmulConstants`):
//   0 = HEADS      heads
//   1 = K          lanes of a head's input part each row dots
//   2 = N_HEAD     output lanes per head
//   3 = IN_STRIDE  a head's part of an input row (`in_cols / HEADS`)
//   4 = GROUP      MLX-affine group size (affine kernels)
//   5 = BITS       MLX-affine code width (affine kernels)
//   6 = SIMDGROUPS a threadgroup's simdgroups
//   7 = ROWS_PER_SG output rows a simdgroup computes
//
// Bindings: buffer(0) = weight (packed codes, or dense rows), buffer(1) / (2) =
// scales / biases (affine), buffer(3) = x [rows, HEADS * IN_STRIDE], buffer(4) =
// y [rows, HEADS * N_HEAD].
//
// Dispatch: threadgroups (rows, HEADS * N_HEAD / (SIMDGROUPS * ROWS_PER_SG)
// rounded up, 1), threads (32, SIMDGROUPS, 1): each simdgroup computes
// ROWS_PER_SG output rows of one token row, its lanes splitting K.
//
// An affine row packs eight codes into BITS bytes, little-endian, as MLX packs
// every width it supports (2, 3, 4, 5, 6, 8): code `i` sits at bits
// `[BITS * i, BITS * (i + 1))` of its row's bytes. A lane reads whole packs of
// eight, each inside one group (GROUP is a multiple of eight).

#include <metal_stdlib>
#include "baked.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, HM_HEADS, 0);
SCRATCHY_CONSTANT(uint, HM_K, 1);
SCRATCHY_CONSTANT(uint, HM_N_HEAD, 2);
SCRATCHY_CONSTANT(uint, HM_IN_STRIDE, 3);
SCRATCHY_CONSTANT_OPTIONAL(uint, HM_GROUP, 4);
SCRATCHY_CONSTANT_OPTIONAL(uint, HM_BITS, 5);

SCRATCHY_CONSTANT(uint, HM_SIMDGROUPS, 6);
SCRATCHY_CONSTANT(uint, HM_ROWS_PER_SG, 7);
constant constexpr uint HM_N = HM_HEADS * HM_N_HEAD;
// An affine row's bytes, and its packs of eight codes.
constant constexpr uint HM_ROW_BYTES = HM_BITS_SET ? 4u * ((HM_K * HM_BITS + 31u) / 32u) : 0u;
constant constexpr uint HM_PACKS = HM_K / 8u;
constant constexpr uint HM_GROUPS = HM_BITS_SET ? HM_K / HM_GROUP : 1u;
static_assert(!HM_BITS_SET || (HM_K % HM_GROUP == 0u && HM_GROUP % 8u == 0u),
              "an affine row is whole groups, each whole packs of eight codes");

// Pack `p`'s eight codes of the row at `row`, as one 64-bit little-endian word.
inline ulong hm_pack(device const uchar* row, uint p) {
    device const uchar* at = row + p * HM_BITS;
    if (HM_BITS == 4u) {
        return ulong(*(device const uint*)at);
    }
    if (HM_BITS == 8u) {
        return ulong(*(device const uint*)at) | (ulong(*(device const uint*)(at + 4)) << 32);
    }
    if (HM_BITS == 2u) {
        return ulong(*(device const ushort*)at);
    }
    ulong w = 0;
    for (uint b = 0; b < HM_BITS; ++b) {
        w |= ulong(at[b]) << (8u * b);
    }
    return w;
}

template <typename T, typename S>
[[kernel]] void head_qmv(
    device const uchar* w      [[buffer(0)]],
    device const S*     scales [[buffer(1)]],
    device const S*     biases [[buffer(2)]],
    device const T*     x      [[buffer(3)]],
    device       T*     y      [[buffer(4)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint t = tg.x;
    const uint first = (tg.y * HM_SIMDGROUPS + sg) * HM_ROWS_PER_SG;
    constexpr uint mask = (1u << HM_BITS) - 1u;
    for (uint r = 0; r < HM_ROWS_PER_SG; ++r) {
        const uint n = first + r;
        if (n >= HM_N) {
            return;
        }
        const uint head = n / HM_N_HEAD;
        device const T* xh = x + t * (HM_HEADS * HM_IN_STRIDE) + head * HM_IN_STRIDE;
        device const uchar* row = w + n * HM_ROW_BYTES;
        float acc = 0.0f;
        for (uint p = lane; p < HM_PACKS; p += 32u) {
            const ulong codes = hm_pack(row, p);
            const uint g = (p * 8u) / HM_GROUP;
            const float scale = float(scales[n * HM_GROUPS + g]);
            const float bias = float(biases[n * HM_GROUPS + g]);
            float dot = 0.0f;
            float sum = 0.0f;
            for (uint j = 0; j < 8u; ++j) {
                const float xv = float(xh[p * 8u + j]);
                dot += float((codes >> (HM_BITS * j)) & mask) * xv;
                sum += xv;
            }
            acc += scale * dot + bias * sum;
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            y[t * HM_N + n] = T(acc);
        }
    }
}

template <typename T>
[[kernel]] void head_mv(
    device const T* w [[buffer(0)]],
    device const T* x [[buffer(3)]],
    device       T* y [[buffer(4)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint t = tg.x;
    const uint first = (tg.y * HM_SIMDGROUPS + sg) * HM_ROWS_PER_SG;
    for (uint r = 0; r < HM_ROWS_PER_SG; ++r) {
        const uint n = first + r;
        if (n >= HM_N) {
            return;
        }
        const uint head = n / HM_N_HEAD;
        device const T* xh = x + t * (HM_HEADS * HM_IN_STRIDE) + head * HM_IN_STRIDE;
        device const T* row = w + n * HM_K;
        float acc = 0.0f;
        for (uint i = lane; i < HM_K; i += 32u) {
            acc += float(row[i]) * float(xh[i]);
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            y[t * HM_N + n] = T(acc);
        }
    }
}

SCRATCHY_KERNEL(head_qmv_f16_s_f16, head_qmv<half, half>)
SCRATCHY_KERNEL(head_qmv_f16_s_bf16, head_qmv<half, bfloat>)
SCRATCHY_KERNEL(head_qmv_bf16_s_f16, head_qmv<bfloat, half>)
SCRATCHY_KERNEL(head_qmv_bf16_s_bf16, head_qmv<bfloat, bfloat>)
SCRATCHY_KERNEL(head_mv_f16, head_mv<half>)
SCRATCHY_KERNEL(head_mv_bf16, head_mv<bfloat>)
