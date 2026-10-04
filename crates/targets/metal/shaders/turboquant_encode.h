// SPDX-License-Identifier: Apache-2.0
//
// The TurboQuant encode of K/V vectors — ONE definition, included by the kernels that encode:
// turboquant.metal's compress, which reads a vector back from the pool, and the KV writers in
// rope.metal, which encode the K and V rows they hold as they write them (`MetalFusion::KvEncoded`).
// The codebook math (norm, signs, WHT butterfly, nearest centroid, bit packing) is a port of
// arozanov's `turboquant_mlx/metal.py`.

#pragma once
#include <metal_stdlib>
using namespace metal;

// Unnormalized Walsh-Hadamard transform of the `dim` floats in `shared`, one element per thread.
// Threadgroup-uniform (every thread runs every barrier).
inline void tq_wht_tg(threadgroup float* shared, uint dim, uint elem) {
    for (uint h = 1; h < dim; h *= 2) {
        uint blk = elem / (2 * h), off = elem % (2 * h);
        if (off < h) { uint j = blk * 2 * h + off; float a = shared[j], b = shared[j + h]; shared[j] = a + b; shared[j + h] = a - b; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Thread `e` of a one-dimensional threadgroup is lane `e % TQ_SIMD` of simdgroup `e / TQ_SIMD`: an
// encode stage whose partners lie within `TQ_SIMD` of each other exchanges through a shuffle, a
// wider one through the threadgroup.
constant constexpr uint TQ_SIMD = 32;

// The threadgroup floats `tq_encode` takes for `n` vectors of `dim`: the norm's sums (which the
// butterfly then reuses), the butterfly's other buffer, and each vector's squared norm.
constexpr uint tq_encode_floats(uint n, uint dim) { return 2u * n * dim + n; }

// An encoded element: its code, and its vector's L2 norm.
struct TqEncoded {
    uint code;
    float norm;
};

// Encode `N` `dim`-vectors together — vector `n`'s element `x[n]` (its offset removed) on thread
// `elem` — sharing every barrier: its codes bit-packed `vals_per_word` to a word into
// `packed[n][0, packed_dim)`, its norm into `*norm[n]`. `scratch` is `tq_encode_floats(N, dim)`
// threadgroup floats, `codes` `N * dim` threadgroup uints. Each vector's arithmetic is the same
// op for op, in the same order — the norm's pairwise sums widest stride first, the butterfly's
// stages narrowest first — whichever way a stage's partners exchange and however many vectors
// encode together, so its codes are the same bits. Threadgroup-uniform; `dim` a power of two.
template <uint N>
inline array<TqEncoded, N> tq_encode(array<float, N> x, uint elem, uint dim, uint bits,
                                     uint vals_per_word, uint packed_dim, uint centroids,
                                     device const float* signs, device const float* boundaries,
                                     array<device uint*, N> packed, array<device float*, N> norm,
                                     threadgroup float* scratch, threadgroup uint* codes) {
    threadgroup float* sq = scratch;
    threadgroup float* other = scratch + N * dim;
    threadgroup float* total = scratch + 2 * N * dim;
    const uint lane = elem % TQ_SIMD;

    // The norm: the squares summed pairwise, stride halving; simdgroup 0 takes the last strides.
    for (uint n = 0; n < N; n++) sq[n * dim + elem] = x[n] * x[n];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s = dim / 2;
    for (; s > TQ_SIMD; s >>= 1) {
        if (elem < s)
            for (uint n = 0; n < N; n++) sq[n * dim + elem] += sq[n * dim + elem + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (elem < TQ_SIMD) {
        array<float, N> v;
        for (uint n = 0; n < N; n++) v[n] = sq[n * dim + elem];
        if (s == TQ_SIMD) {
            for (uint n = 0; n < N; n++) v[n] += sq[n * dim + elem + s];
            s >>= 1;
        }
        for (; s > 0; s >>= 1)
            for (uint n = 0; n < N; n++) {
                const float p = simd_shuffle_down(v[n], ushort(s));
                if (lane < s) v[n] += p;
            }
        if (elem == 0)
            for (uint n = 0; n < N; n++) total[n] = v[n];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    array<TqEncoded, N> out;
    array<float, N> y;
    for (uint n = 0; n < N; n++) {
        const float vec_norm = sqrt(total[n]);
        const float safe_norm = max(vec_norm, 1e-8f);
        y[n] = (x[n] / safe_norm) * signs[elem];
        out[n].norm = vec_norm;
    }

    // The butterfly: the lower partner keeps the sum, the upper the difference.
    uint h = 1;
    for (; h < min(dim, TQ_SIMD); h <<= 1)
        for (uint n = 0; n < N; n++) {
            const float p = simd_shuffle_xor(y[n], ushort(h));
            y[n] = (elem & h) ? p - y[n] : y[n] + p;
        }
    // Wider stages alternate buffers: a buffer is written again two stages on, past the barrier
    // every thread reaches only after its reads of it; the norm's sums are dead past the one above.
    for (bool odd = false; h < dim; h <<= 1, odd = !odd) {
        threadgroup float* buf = odd ? other : sq;
        for (uint n = 0; n < N; n++) buf[n * dim + elem] = y[n];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint n = 0; n < N; n++) {
            const float p = buf[n * dim + (elem ^ h)];
            y[n] = (elem & h) ? p - y[n] : y[n] + p;
        }
    }

    for (uint n = 0; n < N; n++) {
        uint idx = 0;
        for (uint b = 0; b < centroids - 1; b++) if (y[n] > boundaries[b]) idx++;
        codes[n * dim + elem] = idx;
        out[n].code = idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint word_idx = elem / vals_per_word, pos_in_word = elem % vals_per_word;
    if (pos_in_word == 0 && word_idx < packed_dim)
        for (uint n = 0; n < N; n++) {
            uint word = 0;
            for (uint i = 0; i < vals_per_word && (word_idx * vals_per_word + i) < dim; i++)
                word |= (codes[n * dim + word_idx * vals_per_word + i] & ((1u << bits) - 1u)) << (i * bits);
            packed[n][word_idx] = word;
        }
    if (elem == 0)
        for (uint n = 0; n < N; n++) *norm[n] = out[n].norm;
    return out;
}
