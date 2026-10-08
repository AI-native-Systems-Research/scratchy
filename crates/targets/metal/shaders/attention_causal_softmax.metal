// SPDX-License-Identifier: Apache-2.0
//
// Fused scale + causal-mask + row softmax for the gemma4 hd512 GLOBAL unfused
// attention path. Operates in place on the QK^T scores S[q_heads, Lq, kv_len].
//
// The plain `causal_softmax_{f16,bf16}` bodies: one thread per (query i, head
// h); the thread streams the kv_len axis. The full row (Lkv up to ~30k) is
// never staged in threadgroup memory — each j is read straight from device, so
// there is no 32 KiB budget concern. The row's original scores are preserved
// until the final write pass (3 reads, 1 write), so the exp is computed in
// fp32 and only the normalized probability is rounded to the storage dtype
// (no intermediate fp16 round-trip of exp). The production
// `causal_softmax_prod_bf16` variant cooperates per row instead — see its
// block comment below.
//
// Causal model: in chunked prefill the Lq queries of this forward are the LAST
// Lq positions of the live sequence, so query i sits at absolute position
// (kv_len - Lq) + i and attends keys [0, (kv_len - Lq) + i]. Keys at or beyond
// that are masked; their probability is written as 0 so the downstream PV GEMM
// (which spans all kv_len columns) ignores them.
//
// params[0] = Lq, params[1] = kv_len. scale = attention softmax scale
// (gemma4 = 1.0; passed explicitly so other arches reuse the kernel).

#include <metal_stdlib>
#include "baked.h"
using namespace metal;

template <typename T>
inline void causal_softmax_body(
    device T*           scores, // [q_heads, Lq, kv_len], in place
    const device uint*  params, // [Lq, kv_len]
    constant float&     scale,
    uint2               gid)    // x = query i, y = head h
{
    const uint lq     = params[0];
    const uint kv_len = params[1];
    const uint i      = gid.x;
    const uint h      = gid.y;
    if (i >= lq) {
        return;
    }
    device T* row = scores + (uint64_t(h) * lq + i) * kv_len;

    // Causal key count for this query (chunk_start = kv_len - Lq).
    const uint valid = (kv_len - lq) + i + 1u;

    // Pass 1: row max of the scaled scores.
    float m = -INFINITY;
    for (uint j = 0u; j < valid; j++) {
        m = max(m, float(row[j]) * scale);
    }
    // Pass 2: denominator (fp32 accumulate).
    float l = 0.0f;
    for (uint j = 0u; j < valid; j++) {
        l += exp(float(row[j]) * scale - m);
    }
    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    // Pass 3: write normalized probabilities; zero the masked tail.
    for (uint j = 0u; j < valid; j++) {
        row[j] = T(exp(float(row[j]) * scale - m) * inv);
    }
    for (uint j = valid; j < kv_len; j++) {
        row[j] = T(0.0f);
    }
}

kernel void causal_softmax_f16(
    device half*       scores [[buffer(0)]],
    const device uint* params [[buffer(1)]],
    constant float&    scale  [[buffer(2)]],
    uint2              gid    [[thread_position_in_grid]])
{
    causal_softmax_body<half>(scores, params, scale, gid);
}

kernel void causal_softmax_bf16(
    device bfloat*     scores [[buffer(0)]],
    const device uint* params [[buffer(1)]],
    constant float&    scale  [[buffer(2)]],
    uint2              gid    [[thread_position_in_grid]])
{
    causal_softmax_body<bfloat>(scores, params, scale, gid);
}

// ── Production variant: scale compiled in; the ACTUAL query count
// (num new tokens this forward) comes from cu_seqlens_q (NOT the baked
// bucket_m — using bucket_m underflows `kv_len - lq` when bucket_m > kv_len).
// scores rows are one head [bucket_m, kv_len]; only rows [0, lq_actual) are real
// queries (the rest are padding from the over-sized bucket and are zeroed).
//
// One SOFT_T-thread threadgroup per row (dispatch: threadgroups = bucket rows):
// the row max and denominator are cooperative strided scans + simdgroup
// reduces, NOT one thread streaming kv_len serially ×3 passes — the serial
// form ran ~30× under bandwidth and was 11.6% of gemma-4 4k prefill on M1.
// The row is never staged in threadgroup memory (no 32 KiB budget concern at
// long kv); original scores survive until the write pass (3 reads, 1 write),
// the exp is fp32, and only the normalized probability rounds to the storage
// dtype. All early-outs are per-ROW (uniform across the threadgroup), so the
// barriers below never see a partially-returned group.
#if SCRATCHY_COMPILES(causal_softmax_prod_bf16)
SCRATCHY_CONSTANT(float, SOFT_SCALE, 1);

// Block-diagonal span attention: lower-bound each query's valid key range at its
// own span's first block (mirrors the QKᵀ gemm bound, so the skipped band is
// never read). Granularity == metal KV block size; unset → 0 (disabled →
// full causal range, byte-identical).
SCRATCHY_CONSTANT_OPTIONAL(uint, SOFT_SPAN_BLOCK, 3);

// Threads cooperating on one row. MUST match the lowering's
// threads_per_threadgroup for this kernel (the stride and reduce assume it).
constant constexpr uint SOFT_T = 64;

template <typename T>
inline void causal_softmax_prod_body(
    device T* scores, const device uint* seq_used, const device uint* cu_seqlens_q,
    const device uint* span_ids, threadgroup float* red, uint row, uint tid, uint simd_lid)
{
    const uint kv_len    = seq_used[0];
    const uint lq_actual = cu_seqlens_q[1] - cu_seqlens_q[0]; // single-seq prefill: new query tokens
    if (row >= lq_actual) {
        // ZERO padding rows (not just skip): the unguarded NAX PV gemm over-reads its
        // contraction K-tail, which reads the NEXT row's scores — so a NaN/garbage
        // padding row poisons a real row's output (NaN×0=NaN). Steel guards its K-tail
        // so this is harmless there.
        device T* prow = scores + uint64_t(row) * kv_len;
        for (uint j = tid; j < kv_len; j += SOFT_T) { prow[j] = T(0); }
        return; // padding row from the over-sized prefill bucket
    }
    // Query `row` is the row-th NEW token → absolute key position
    // (kv_len - lq_actual) + row; it attends keys [0, that] inclusive.
    // kv_len >= lq_actual always (prefill appends).
    device T* r = scores + uint64_t(row) * kv_len;
    const uint valid = (kv_len - lq_actual) + row + 1u;
    // Block-diagonal lower bound: span query attends only [span_lo, valid).
    uint span_lo = 0u;
    if (SOFT_SPAN_BLOCK != 0u) {
        const uint sf = span_ids[(kv_len - lq_actual) + row]; // per-TOKEN index (no block divide)
        if (sf != 0u) {
            span_lo = sf - 1u; // label-1 IS the span's first TOKEN (exact; no block round)
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Pass 1: cooperative row max of the scaled scores.
    float m = -INFINITY;
    for (uint j = span_lo + tid; j < valid; j += SOFT_T) {
        m = max(m, float(r[j]) * SOFT_SCALE);
    }
    for (uint off = 16u; off > 0u; off >>= 1u) {
        m = max(m, simd_shuffle(m, simd_lid ^ off));
    }
    if (simd_lid == 0u) { red[tid / 32u] = m; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = red[0];
    for (uint s = 1u; s < SOFT_T / 32u; ++s) { m = max(m, red[s]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Pass 2: denominator (fp32 accumulate), same strided scan.
    float l = 0.0f;
    for (uint j = span_lo + tid; j < valid; j += SOFT_T) {
        l += exp(float(r[j]) * SOFT_SCALE - m);
    }
    for (uint off = 16u; off > 0u; off >>= 1u) {
        l += simd_shuffle(l, simd_lid ^ off);
    }
    if (simd_lid == 0u) { red[tid / 32u] = l; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    l = red[0];
    for (uint s = 1u; s < SOFT_T / 32u; ++s) { l += red[s]; }
    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    // Pass 3: write normalized probabilities; zero the masked bands.
    for (uint j = span_lo + tid; j < valid; j += SOFT_T) {
        r[j] = T(exp(float(r[j]) * SOFT_SCALE - m) * inv);
    }
    for (uint j = tid; j < span_lo; j += SOFT_T) { r[j] = T(0.0f); }
    for (uint j = valid + tid; j < kv_len; j += SOFT_T) { r[j] = T(0.0f); }
}

kernel void causal_softmax_prod_bf16(
    device bfloat*     scores       [[buffer(0)]],
    const device uint* seq_used     [[buffer(1)]],
    const device uint* cu_seqlens_q [[buffer(2)]],
    const device uint* span_ids     [[buffer(3)]],
    uint3              tgid         [[threadgroup_position_in_grid]],
    uint               tid          [[thread_index_in_threadgroup]],
    uint               simd_lid     [[thread_index_in_simdgroup]])
{
    threadgroup float red[SOFT_T / 32u];
    causal_softmax_prod_body<bfloat>(scores, seq_used, cu_seqlens_q, span_ids, red,
                                     tgid.x, tid, simd_lid);
}
#endif
