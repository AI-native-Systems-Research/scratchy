// SPDX-License-Identifier: Apache-2.0
//
// Multi-head latent attention's kernels: the query rotation
// (`MetalStep::MlaRopeQ`), the cache write (`MetalStep::MlaAppend`) and the
// absorbed attention over the cache (`MetalStep::MlaAttend`).
//
// The cache holds one row per token of a layer: the compressed latent
// (LATENT lanes) then the rotated shared rope key (ROPE lanes), `ROW` wide in
// all. The paged pool holds the row as its K and V halves, `HALF = ROW / 2`
// lanes each: lane `e` of a row sits in the K store when `e < HALF`, else at
// `e - HALF` in the V store, both at the token's slot of the layer's one
// kv head. A block is `BLOCK_SIZE` rows; a store binding is the layer's chunk
// table (a physical block derefs `table[block / BLOCKS_PER_CHUNK]`, then block
// `block % BLOCKS_PER_CHUNK` of it; BLOCKS_PER_CHUNK 0: one base).
//
// The rope pairs lanes `(2i, 2i + 1)` of a rope part, rotated by frequency
// `i`'s cos and sin at the row's position (`cos_sin[pos]` holds the ROPE / 2
// cosines, then the sines).
//
// Constants (compiled in; `MlaConstants`), every kernel's the same slots:
//   0 = HEADS       query heads
//   1 = LATENT      the compressed latent's width
//   2 = NOPE        a query head's un-rotated part
//   3 = ROPE        a query head's rotated part, the shared key's width
//   4 = BLOCK_SIZE  rows per paged block
//   5 = MAX_BLOCKS  the block table's row stride
//   6 = BLOCKS_PER_CHUNK
//   7 = SCALE       the softmax scale
//   8 = HEADS_PER_TG  query heads an attention threadgroup serves

#include <metal_stdlib>
#include "baked.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, MLA_HEADS, 0);
SCRATCHY_CONSTANT(uint, MLA_LATENT, 1);
SCRATCHY_CONSTANT(uint, MLA_NOPE, 2);
SCRATCHY_CONSTANT(uint, MLA_ROPE, 3);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_BLOCK_SIZE, 4);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_MAX_BLOCKS, 5);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_BLOCKS_PER_CHUNK, 6);
SCRATCHY_CONSTANT_OPTIONAL(float, MLA_SCALE, 7);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_HEADS_PER_TG, 8);

constant constexpr uint MLA_Q_HEAD = MLA_NOPE + MLA_ROPE;
constant constexpr uint MLA_ROW = MLA_LATENT + MLA_ROPE;
constant constexpr uint MLA_HALF = MLA_ROW / 2u;
constant constexpr uint MLA_HALF_ROPE = MLA_ROPE / 2u;

// Lane `d` of a rope part `x` (ROPE lanes) rotated at `pos`: its pair's
// partner is `d ^ 1`, the pair's frequency `d / 2`.
template <typename T>
inline float mla_rotate(device const T* x, uint d, device const T* cos_sin, uint pos) {
    device const T* row = cos_sin + pos * MLA_ROPE;
    const float c = float(row[d / 2u]);
    const float s = float(row[MLA_HALF_ROPE + d / 2u]);
    const float x0 = float(x[d & ~1u]);
    const float x1 = float(x[d | 1u]);
    return (d & 1u) == 0u ? x0 * c - x1 * s : x1 * c + x0 * s;
}

// The address of lane `e`'s store element at physical slot `slot`: whether it
// is in the V store, and its index in its chunk; `chunk` is set to its chunk.
inline uint mla_cache_at(uint slot, uint e, thread uint& chunk, thread bool& in_v) {
    const uint block = slot / MLA_BLOCK_SIZE;
    chunk = MLA_BLOCKS_PER_CHUNK == 0u ? 0u : block / MLA_BLOCKS_PER_CHUNK;
    const uint blk = MLA_BLOCKS_PER_CHUNK == 0u ? block : block % MLA_BLOCKS_PER_CHUNK;
    in_v = e >= MLA_HALF;
    return blk * (MLA_BLOCK_SIZE * MLA_HALF) + (slot % MLA_BLOCK_SIZE) * MLA_HALF
        + (in_v ? e - MLA_HALF : e);
}

// ── Query rotation ──────────────────────────────────────────────────────────
// buffer(0) = q [rows, HEADS * Q_HEAD], buffer(1) = out (same), buffer(2) =
// positions [rows], buffer(3) = cos_sin [positions, ROPE].
// Dispatch: threadgroups (rows, HEADS, 1), threads (Q_HEAD, 1, 1).
template <typename T>
[[kernel]] void mla_rope_q(
    device const T*    q         [[buffer(0)]],
    device       T*    out       [[buffer(1)]],
    device const uint* positions [[buffer(2)]],
    device const T*    cos_sin   [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint  d  [[thread_index_in_threadgroup]])
{
    if (d >= MLA_Q_HEAD) {
        return;
    }
    const uint base = (tg.x * MLA_HEADS + tg.y) * MLA_Q_HEAD;
    out[base + d] = d < MLA_NOPE
        ? q[base + d]
        : T(mla_rotate(q + base + MLA_NOPE, d - MLA_NOPE, cos_sin, positions[tg.x]));
}

// ── Cache write ─────────────────────────────────────────────────────────────
// buffer(0) = latent [rows, LATENT], buffer(1) = k_pe [rows, ROPE], buffer(2)
// = out [rows, ROPE] (the rotated key), buffer(3) = positions [rows],
// buffer(4) = cos_sin, buffer(5) = slot_mapping [rows] (0xFFFFFFFF: a padding
// row, written nowhere), buffer(6) / (7) = the layer's K / V chunk tables.
// Dispatch: threadgroups (rows, 1, 1), threads (ROW, 1, 1).
template <typename T>
[[kernel]] void mla_append(
    device const T*        latent       [[buffer(0)]],
    device const T*        k_pe         [[buffer(1)]],
    device       T*        out          [[buffer(2)]],
    device const uint*     positions    [[buffer(3)]],
    device const T*        cos_sin      [[buffer(4)]],
    device const uint*     slot_mapping [[buffer(5)]],
    device const uint64_t* k_cache      [[buffer(6)]],
    device const uint64_t* v_cache      [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint  e  [[thread_index_in_threadgroup]])
{
    if (e >= MLA_ROW) {
        return;
    }
    const uint t = tg.x;
    T v;
    if (e < MLA_LATENT) {
        v = latent[t * MLA_LATENT + e];
    } else {
        const uint d = e - MLA_LATENT;
        v = T(mla_rotate(k_pe + t * MLA_ROPE, d, cos_sin, positions[t]));
        out[t * MLA_ROPE + d] = v;
    }
    const uint slot_raw = slot_mapping[t];
    if (slot_raw == 0xFFFFFFFFu) {
        return;
    }
    uint chunk;
    bool in_v;
    const uint at = mla_cache_at(slot_raw & 0x7FFFFFFFu, e, chunk, in_v);
    ((device T*)(in_v ? v_cache[chunk] : k_cache[chunk]))[at] = v;
}

// ── Absorbed attention ──────────────────────────────────────────────────────
// buffer(0) = out [rows, HEADS * LATENT], buffer(1) = q_latent [rows, HEADS *
// LATENT], buffer(2) = q [rows, HEADS * Q_HEAD] (rotated: its rope parts are
// read), buffer(3) = seq_used_k [seqs], buffer(4) = block_table [seqs,
// MAX_BLOCKS], buffer(5) / (6) = the layer's K / V chunk tables, buffer(7) =
// cu_seqlens_q [seqs + 1].
//
// Row `t` of sequence `s` attends to the first `seq_used_k[s] - (cu[s + 1] -
// 1 - t)` cached rows of `s`: its own position and every one before it. A
// head's query over a cached row is its latent query · the row's latent plus
// its rope part · the row's rotated key, both scaled by SCALE.
//
// Dispatch: threadgroups (rows, HEADS / HEADS_PER_TG, 1), threads (1024, 1,
// 1): 32 simdgroups split the cached rows, each lane holds ELEMS contiguous
// lanes of a row (and of each served head's query), and the simdgroups'
// online softmaxes combine at the end (MLX `sdpa_vector`'s merge).
constant constexpr uint MLA_SG = 32u;
constant constexpr uint MLA_ELEMS = (MLA_ROW + 31u) / 32u;
constant constexpr uint MLA_HPT = MLA_HEADS_PER_TG_SET ? MLA_HEADS_PER_TG : 1u;

template <typename T>
[[kernel, max_total_threads_per_threadgroup(1024)]] void mla_attend(
    device       T*        out          [[buffer(0)]],
    device const T*        q_latent     [[buffer(1)]],
    device const T*        q            [[buffer(2)]],
    device const uint*     seq_used_k   [[buffer(3)]],
    device const uint*     block_table  [[buffer(4)]],
    device const uint64_t* k_cache      [[buffer(5)]],
    device const uint64_t* v_cache      [[buffer(6)]],
    device const uint*     cu_seqlens_q [[buffer(7)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint row = tg.x;
    const uint head0 = tg.y * MLA_HPT;
    uint seq = 0;
    while (cu_seqlens_q[seq + 1u] <= row) {
        ++seq;
    }
    const uint kv_len = seq_used_k[seq] - (cu_seqlens_q[seq + 1u] - 1u - row);
    device const uint* blocks = block_table + seq * MLA_MAX_BLOCKS;

    // This lane's lanes of a row: [e0, e0 + MLA_ELEMS).
    const uint e0 = lane * MLA_ELEMS;
    float qr[MLA_HPT * MLA_ELEMS];
    float o[MLA_HPT * MLA_ELEMS];
    float m[MLA_HPT];
    float l[MLA_HPT];
    for (uint h = 0; h < MLA_HPT; ++h) {
        const uint head = head0 + h;
        for (uint j = 0; j < MLA_ELEMS; ++j) {
            const uint e = e0 + j;
            float v = 0.0f;
            if (e < MLA_LATENT) {
                v = float(q_latent[(row * MLA_HEADS + head) * MLA_LATENT + e]);
            } else if (e < MLA_ROW) {
                v = float(q[(row * MLA_HEADS + head) * MLA_Q_HEAD + MLA_NOPE + (e - MLA_LATENT)]);
            }
            qr[h * MLA_ELEMS + j] = MLA_SCALE * v;
            o[h * MLA_ELEMS + j] = 0.0f;
        }
        m[h] = -INFINITY;
        l[h] = 0.0f;
    }

    float k[MLA_ELEMS];
    for (uint pos = sg; pos < kv_len; pos += MLA_SG) {
        const uint slot = (blocks[pos / MLA_BLOCK_SIZE] & 0x7FFFFFFFu) * MLA_BLOCK_SIZE
            + pos % MLA_BLOCK_SIZE;
        for (uint j = 0; j < MLA_ELEMS; ++j) {
            const uint e = e0 + j;
            k[j] = 0.0f;
            if (e < MLA_ROW) {
                uint chunk;
                bool in_v;
                const uint at = mla_cache_at(slot, e, chunk, in_v);
                k[j] = float(((device const T*)(in_v ? v_cache[chunk] : k_cache[chunk]))[at]);
            }
        }
        for (uint h = 0; h < MLA_HPT; ++h) {
            float s = 0.0f;
            for (uint j = 0; j < MLA_ELEMS; ++j) {
                s += qr[h * MLA_ELEMS + j] * k[j];
            }
            s = simd_sum(s);
            const float m_new = max(m[h], s);
            const float factor = exp(m[h] - m_new);
            const float p = exp(s - m_new);
            l[h] = l[h] * factor + p;
            for (uint j = 0; j < MLA_ELEMS; ++j) {
                const bool latent_lane = e0 + j < MLA_LATENT;
                o[h * MLA_ELEMS + j] = o[h * MLA_ELEMS + j] * factor + (latent_lane ? p * k[j] : 0.0f);
            }
            m[h] = m_new;
        }
    }

    // Merge the simdgroups' softmaxes: each head's max and sum over them, then
    // each output lane's sum through a [lane][simdgroup] transpose.
    threadgroup float tg_m[MLA_HPT * MLA_SG];
    threadgroup float tg_l[MLA_HPT * MLA_SG];
    threadgroup float tg_o[MLA_SG * 32u];
    if (lane == 0) {
        for (uint h = 0; h < MLA_HPT; ++h) {
            tg_m[h * MLA_SG + sg] = m[h];
            tg_l[h * MLA_SG + sg] = l[h];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 0; h < MLA_HPT; ++h) {
        // Lane `i` stands for simdgroup `i`'s partial.
        const float mi = tg_m[h * MLA_SG + lane];
        const float m_all = simd_max(mi);
        const float factor = mi == -INFINITY ? 0.0f : exp(mi - m_all);
        const float l_all = simd_sum(tg_l[h * MLA_SG + lane] * factor);
        for (uint j = 0; j < MLA_ELEMS; ++j) {
            tg_o[lane * MLA_SG + sg] = o[h * MLA_ELEMS + j];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Simdgroup `sg` now holds lane `sg`'s element `j` of every simdgroup.
            const float merged = simd_sum(tg_o[sg * 32u + lane] * factor);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint e = sg * MLA_ELEMS + j;
            if (lane == 0 && e < MLA_LATENT) {
                out[(row * MLA_HEADS + head0 + h) * MLA_LATENT + e] = T(merged / l_all);
            }
        }
    }
}

SCRATCHY_KERNEL(mla_rope_q_f16, mla_rope_q<half>)
SCRATCHY_KERNEL(mla_rope_q_bf16, mla_rope_q<bfloat>)
SCRATCHY_KERNEL(mla_append_f16, mla_append<half>)
SCRATCHY_KERNEL(mla_append_bf16, mla_append<bfloat>)
SCRATCHY_KERNEL(mla_attend_f16, mla_attend<half>)
SCRATCHY_KERNEL(mla_attend_bf16, mla_attend<bfloat>)
