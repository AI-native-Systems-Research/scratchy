// SPDX-License-Identifier: Apache-2.0
//
// The additive offset a TurboQuant'd K/V carries — ONE definition, included by
// the kernels that remove it (turboquant.metal's compress) and restore it
// (attention.metal's prefill staging).
//
// The codec quantizes each vector relative to its own L2 norm, so its error
// scales with that norm. A K/V that is `projection(x) + bias` (Qwen2) has a norm
// the bias can dominate, and the error then swamps the part that tells one key
// from another; the codes therefore hold the vector MINUS this offset.

#pragma once
#include <metal_stdlib>
using namespace metal;

// Element `e` of the offset (lowering: `tq_offset_bindings`). `mode` 0: none.
// 1: `bias`. 2: `bias` rotated exactly as RopeAppend rotated the key at position
// `pos` — NeoX pairs (d, d + pair_off) for d < rot_dim/2 at cos/sin index d,
// every other element unrotated. A slot/block whose K is stored unrotated
// (spans, bit 31) holds the bias unrotated too.
template <typename T>
inline float tq_offset(uint mode, device const T* bias, device const T* cos_sin, uint rot_dim,
                       uint pair_off, uint pos, bool unrotated, uint e) {
    if (mode == 0u) return 0.0f;
    const float b = float(bias[e]);
    if (mode == 1u || unrotated) return b;
    const uint half_rot = rot_dim / 2u;
    device const T* cos_row = cos_sin + pos * rot_dim;
    device const T* sin_row = cos_row + half_rot;
    if (e < half_rot) {
        return b * float(cos_row[e]) - float(bias[e + pair_off]) * float(sin_row[e]);
    }
    if (e >= pair_off && e < pair_off + half_rot) {
        const uint d = e - pair_off;
        return b * float(cos_row[d]) + float(bias[d]) * float(sin_row[d]);
    }
    return b;
}
