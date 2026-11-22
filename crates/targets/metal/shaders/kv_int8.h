// SPDX-License-Identifier: Apache-2.0
//
// The int8 KV cache's rows, ONE definition every KV writer and reader shares
// (`scratchy_layers::kv_int8`, oMLX's `affine8`): a head row of K or V is rotated — each element
// signed by a fixed ±1 (`kv_sign`), then an orthonormal Walsh–Hadamard transform — and stored as
// signed 8-bit codes against one f32 scale for the row, `scale = max(max/127, −min/128)`.
//
// A layer's store holds a row per (KV slot, KV head), `slot · num_kv + kv_head`: its `dim` code
// bytes at `row · dim`, its scale at `row`. Attention runs in the rotated domain — a rotated
// element is `code · scale` — its query rotated once, its output rotated back once.

#pragma once
#include <metal_stdlib>
using namespace metal;

// Element `e`'s sign before the transform: `kv_int8::sign`'s hash.
inline float kv_sign(uint e) {
    ulong z = ulong(e) + 0x9E3779B97F4A7C15ul;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    z ^= z >> 31;
    return (z & 1ul) == 0ul ? 1.0f : -1.0f;
}

// A rotated element's code against its row's scale.
inline char kv_code(float r, float scale) {
    return scale > 0.0f ? char(clamp(rint(r / scale), -128.0f, 127.0f)) : char(0);
}

// A row's scale from its rotated extremes.
inline float kv_scale(float hi, float lo) {
    return max(max(hi, 0.0f) / 127.0f, min(lo, 0.0f) / -128.0f);
}

// Thread `e` of a one-dimensional threadgroup is lane `e % KV_SIMD` of simdgroup `e / KV_SIMD`.
constant constexpr uint KV_SIMD = 32;

// The threadgroup floats `kv_encode` takes for `n` vectors of `dim`: the butterfly's two buffers.
constexpr uint kv_encode_floats(uint n, uint dim) { return 2u * n * dim; }

// Encode `N` `dim`-vectors together — vector `n`'s element `x[n]` on thread `elem` — sharing every
// barrier: rotated, then coded into `codes[n][0, dim)` against the scale written to `*scale[n]`.
// `scratch` is `kv_encode_floats(N, dim)` threadgroup floats. Threadgroup-uniform; `dim` a power of
// two. A threadgroup wider than `dim` runs it whole: its threads past `dim` keep every barrier and
// touch no memory.
template <uint N>
inline void kv_encode(array<float, N> x, uint elem, uint dim, array<device char*, N> codes,
                      array<device float*, N> scale, threadgroup float* scratch) {
    const bool live = elem < dim;
    const uint lane = elem % KV_SIMD;
    array<float, N> y;
    for (uint n = 0; n < N; n++) y[n] = live ? x[n] * kv_sign(elem) : 0.0f;

    // The butterfly: the lower partner keeps the sum, the upper the difference.
    uint h = 1;
    for (; h < min(dim, KV_SIMD); h <<= 1)
        for (uint n = 0; n < N; n++) {
            const float p = simd_shuffle_xor(y[n], ushort(h));
            y[n] = (elem & h) ? p - y[n] : y[n] + p;
        }
    // Wider stages alternate buffers: a buffer is written again two stages on, past the barrier
    // every thread reaches only after its reads of it.
    threadgroup float* bufs[2] = {scratch, scratch + N * dim};
    bool odd = false;
    for (; h < dim; h <<= 1, odd = !odd) {
        threadgroup float* buf = bufs[odd ? 1 : 0];
        if (live)
            for (uint n = 0; n < N; n++) buf[n * dim + elem] = y[n];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (live)
            for (uint n = 0; n < N; n++) {
                const float p = buf[n * dim + (elem ^ h)];
                y[n] = (elem & h) ? p - y[n] : y[n] + p;
            }
    }
    const float norm = rsqrt(float(dim));
    for (uint n = 0; n < N; n++) y[n] *= norm;

    // The extremes: each simdgroup's, then simdgroup 0's over theirs, through the buffer the last
    // butterfly stage did not read.
    threadgroup float* ext = bufs[odd ? 1 : 0];
    const uint groups = (dim + KV_SIMD - 1u) / KV_SIMD;
    array<float, N> hi, lo;
    for (uint n = 0; n < N; n++) {
        hi[n] = simd_max(live ? y[n] : -INFINITY);
        lo[n] = simd_min(live ? y[n] : INFINITY);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (live && lane == 0)
        for (uint n = 0; n < N; n++) {
            ext[(2u * n) * groups + elem / KV_SIMD] = hi[n];
            ext[(2u * n + 1u) * groups + elem / KV_SIMD] = lo[n];
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint n = 0; n < N; n++) {
        float h2 = -INFINITY, l2 = INFINITY;
        for (uint g = 0; g < groups; g++) {
            h2 = max(h2, ext[(2u * n) * groups + g]);
            l2 = min(l2, ext[(2u * n + 1u) * groups + g]);
        }
        const float s = kv_scale(h2, l2);
        if (live) codes[n][elem] = kv_code(y[n], s);
        if (elem == 0) *scale[n] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
