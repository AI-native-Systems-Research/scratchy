// SPDX-License-Identifier: Apache-2.0
//
// A KV writer's row work — its rope rotation and its int8 encode of the K and V rows it holds —
// ONE definition, included by the KV writers (rope.metal, kv_append.metal; `MetalFusion::KvEncoded`)
// and by the decode attention that runs its writer itself (attention.metal;
// `MetalFusion::RopedAttention`), so both compute the same bits.

#pragma once
#include "kv_int8.h"

// The int8 store a KV writer's folded-in encode writes, by binding: K's codes and scales (16, 17),
// V's (18, 19). Every KV writer (rope.metal, kv_append.metal) binds it alike.
#define KV_WRITER_INT8_BUFFERS                                                   \
    device char*  kv_codes_k  [[buffer(16)]],                                    \
    device float* kv_scales_k [[buffer(17)]],                                    \
    device char*  kv_codes_v  [[buffer(18)]],                                    \
    device float* kv_scales_v [[buffer(19)]],

// One NeoX pair `(x0, x1)` rotated by its position's `(c, s)`: `(x0·c − x1·s, x1·c + x0·s)`.
inline float2 rope_rotate(float x0, float x1, float c, float s) {
    return float2(x0 * c - x1 * s, x1 * c + x0 * s);
}

// Encode element `d` of a token's K and V rows (of `dim`, KV head `kv_head` of `num_kv`) into the
// int8 store at the token's slot (bit 31, the spans "stored unrotated" flag, stripped), K's and
// V's encodes sharing every barrier. Threadgroup-uniform; a thread with `d` past `dim` keeps the
// barriers and encodes nothing.
inline void kv_row_encode(float k, float v, uint d, uint dim, uint num_kv, uint kv_head,
                          uint slot_raw, device char* codes_k, device float* scales_k,
                          device char* codes_v, device float* scales_v,
                          threadgroup float* scratch) {
    const uint row = (slot_raw & 0x7FFFFFFFu) * num_kv + kv_head;
    kv_encode<2>({k, v}, d, dim, {codes_k + row * dim, codes_v + row * dim},
                 {scales_k + row, scales_v + row}, scratch);
}
