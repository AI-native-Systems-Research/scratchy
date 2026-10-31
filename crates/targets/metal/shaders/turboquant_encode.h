// SPDX-License-Identifier: Apache-2.0
//
// The TurboQuant encode of K/V vectors — ONE definition, included by the kernels that encode:
// turboquant.metal's compress, which reads a vector back from the pool, and the KV writers in
// rope.metal, which encode the K and V rows they hold as they write them (`MetalFusion::KvEncoded`).
// The codebook math (norm, signs, WHT butterfly, nearest centroid, bit packing) is a port of
// arozanov's `turboquant_mlx/metal.py`; the stored norm is corrected for the codes' decoded
// length (`PolarQuantizer::quantize`).

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

// The threadgroup floats `tq_encode` takes for `n` vectors of `dim`: the sums' operands (which the
// butterfly reuses between them), the butterfly's other buffer, and each vector's sum.
constexpr uint tq_encode_floats(uint n, uint dim) { return 2u * n * dim + n; }

// An encoded element: its code, and its vector's stored norm.
struct TqEncoded {
    uint code;
    float norm;
};

// `sum[n]`, on every thread, the sum over `dim` threads of each one's `v[n]`: written to `terms`,
// summed pairwise, stride halving, simdgroup 0 taking the last strides. Threadgroup-uniform; the
// caller has every thread past its last read of `terms` and `sum` (a barrier) before the call.
template <uint N>
inline array<float, N> tq_sum_tg(array<float, N> v, uint elem, uint dim, threadgroup float* terms,
                                 threadgroup float* sum) {
    const uint lane = elem % TQ_SIMD;
    if (elem < dim)
        for (uint n = 0; n < N; n++) terms[n * dim + elem] = v[n];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s = dim / 2;
    for (; s > TQ_SIMD; s >>= 1) {
        if (elem < s)
            for (uint n = 0; n < N; n++) terms[n * dim + elem] += terms[n * dim + elem + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (elem < TQ_SIMD) {
        array<float, N> t;
        for (uint n = 0; n < N; n++) t[n] = terms[n * dim + elem];
        if (s == TQ_SIMD) {
            for (uint n = 0; n < N; n++) t[n] += terms[n * dim + elem + s];
            s >>= 1;
        }
        for (; s > 0; s >>= 1)
            for (uint n = 0; n < N; n++) {
                const float p = simd_shuffle_down(t[n], ushort(s));
                if (lane < s) t[n] += p;
            }
        if (elem == 0)
            for (uint n = 0; n < N; n++) sum[n] = t[n];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    array<float, N> out;
    for (uint n = 0; n < N; n++) out[n] = sum[n];
    return out;
}

// Encode `N` `dim`-vectors together — vector `n`'s element `x[n]` (its offset removed) on thread
// `elem` — sharing every barrier: its codes bit-packed `vals_per_word` to a word into
// `packed[n][0, packed_dim)`, its norm into `*norm[n]`. `table` is the codebook's encode table
// (`PolarQuantizer::encode_table`): the `centroids - 1` boundaries the codes are digitized against,
// then the `centroids` levels. The norm stored is the vector's own over its codes' decoded length,
// `|levels[code]| / sqrt(dim)`, so a vector decodes at its own length rather than a little short of
// it. `scratch` is `tq_encode_floats(N, dim)` threadgroup floats, `codes` `N * dim` threadgroup
// uints. Each vector's arithmetic is the same op for op, in the same order — the sums pairwise,
// widest stride first, the butterfly's stages narrowest first — whichever way a stage's partners
// exchange and however many vectors encode together, so its codes are the same bits.
// Threadgroup-uniform; `dim` a power of two. A threadgroup wider than `dim` (a decode attention's)
// runs it whole: its threads past `dim` keep every barrier and touch no memory.
template <uint N>
inline array<TqEncoded, N> tq_encode(array<float, N> x, uint elem, uint dim, uint bits,
                                     uint vals_per_word, uint packed_dim, uint centroids,
                                     device const float* signs, device const float* table,
                                     array<device uint*, N> packed, array<device float*, N> norm,
                                     threadgroup float* scratch, threadgroup uint* codes) {
    threadgroup float* sq = scratch;
    threadgroup float* other = scratch + N * dim;
    threadgroup float* total = scratch + 2 * N * dim;
    device const float* levels = table + (centroids - 1);
    const bool live = elem < dim;

    array<float, N> x2;
    for (uint n = 0; n < N; n++) x2[n] = x[n] * x[n];
    const array<float, N> norm2 = tq_sum_tg<N>(x2, elem, dim, sq, total);

    array<TqEncoded, N> out;
    array<float, N> y;
    for (uint n = 0; n < N; n++) {
        const float vec_norm = sqrt(norm2[n]);
        const float safe_norm = max(vec_norm, 1e-8f);
        y[n] = live ? (x[n] / safe_norm) * signs[elem] : 0.0f;
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
        if (live)
            for (uint n = 0; n < N; n++) buf[n * dim + elem] = y[n];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (live)
            for (uint n = 0; n < N; n++) {
                const float p = buf[n * dim + (elem ^ h)];
                y[n] = (elem & h) ? p - y[n] : y[n] + p;
            }
    }

    array<float, N> level2;
    for (uint n = 0; n < N; n++) {
        uint idx = 0;
        for (uint b = 0; b < centroids - 1; b++) if (y[n] > table[b]) idx++;
        if (live) codes[n * dim + elem] = idx;
        out[n].code = idx;
        level2[n] = live ? levels[idx] * levels[idx] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The codes' decoded length: the levels' L2 norm over sqrt(dim). Its sums reuse `sq`, which
    // every thread is past its last butterfly read of. `precise::`, so fast math rewrites it the
    // same way in every kernel that inlines it, and a writer's norms equal the compress kernel's.
    const array<float, N> decoded2 = tq_sum_tg<N>(level2, elem, dim, sq, total);
    for (uint n = 0; n < N; n++) {
        const float decoded = precise::sqrt(precise::divide(decoded2[n], float(dim)));
        if (decoded > 0.0f) out[n].norm = precise::divide(out[n].norm, decoded);
    }
    const uint word_idx = elem / vals_per_word, pos_in_word = elem % vals_per_word;
    if (live && pos_in_word == 0 && word_idx < packed_dim)
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
