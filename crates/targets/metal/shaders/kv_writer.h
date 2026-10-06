// SPDX-License-Identifier: Apache-2.0
//
// A KV writer's row work — its rope rotation and its TurboQuant encode of the K and V rows it
// holds — ONE definition, included by the KV writers (rope.metal; `MetalFusion::KvEncoded`) and by
// the decode attention that runs its writer itself (attention.metal; `MetalFusion::RopedAttention`),
// so both compute the same bits.

#pragma once
#include "turboquant_encode.h"
#include "turboquant_offset.h"

// One NeoX pair `(x0, x1)` rotated by its position's `(c, s)`: `(x0·c − x1·s, x1·c + x0·s)`.
inline float2 rope_rotate(float x0, float x1, float c, float s) {
    return float2(x0 * c - x1 * s, x1 * c + x0 * s);
}

// Encode element `d` of a token's K and V rows (of `dim`, KV head `kv_head` of `num_kv`) into the
// packed store at the token's slot, each operand's offset (turboquant_offset.h's `k_offset` /
// `v_offset` mode) removed first: the same codes `tq_compress_paged` makes of them read back from
// the pool, K's and V's encodes sharing every barrier. Threadgroup-uniform; a thread with `d` past
// `dim` keeps the barriers and encodes nothing.
template <typename T>
inline void kv_row_encode(float k, float v, uint d, uint dim, uint num_kv, uint kv_head,
                          uint slot_raw, uint pos, uint bits, uint vals_per_word, uint packed_dim,
                          uint k_offset, uint v_offset, uint rot_dim, uint pair_off,
                          device const float* signs, device const float* boundaries,
                          device uint* packed_k, device float* norms_k, device uint* packed_v,
                          device float* norms_v, device const T* k_bias, device const T* v_bias,
                          device const T* cos_sin, threadgroup float* scratch,
                          threadgroup uint* codes) {
    const uint store = (slot_raw & 0x7FFFFFFFu) * num_kv + kv_head;
    const bool unrotated = (slot_raw & 0x80000000u) != 0u;
    const bool live = d < dim;
    const float k_off = live ? tq_offset<T>(k_offset, k_bias + kv_head * dim, cos_sin, rot_dim,
                                            pair_off, pos, unrotated, d)
                             : 0.0f;
    const float v_off = live ? tq_offset<T>(v_offset, v_bias + kv_head * dim, cos_sin, rot_dim,
                                            pair_off, pos, unrotated, d)
                             : 0.0f;
    tq_encode<2>({k - k_off, v - v_off}, d, dim, bits, vals_per_word, packed_dim, 1u << bits, signs,
                 boundaries, {packed_k + store * packed_dim, packed_v + store * packed_dim},
                 {norms_k + store, norms_v + store}, scratch, codes);
}
