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
//   9 = SIMDGROUPS  an attention threadgroup's simdgroups
//  10 = PARTS       the partitions an attention splits each sequence's cached
//                   rows into (unset: one, no merge pass)
//  11 = PARTIAL     a partial's elements: its max, its sum, then its LATENT lanes
//  12 = MERGE_THREADS  a merge threadgroup's threads
//  13 = ROW_THREADS a row kernel's threads (the query rotation, the cache write)

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
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_SIMDGROUPS, 9);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_PARTS, 10);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_PARTIAL_STRIDE, 11);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_MERGE_THREADS, 12);
SCRATCHY_CONSTANT_OPTIONAL(uint, MLA_ROW_THREADS, 13);

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
// Dispatch: threadgroups (rows, HEADS, 1), threads (ROW_THREADS, 1, 1), each
// thread every ROW_THREADS-th lane of its head.
template <typename T>
[[kernel]] void mla_rope_q(
    device const T*    q         [[buffer(0)]],
    device       T*    out       [[buffer(1)]],
    device const uint* positions [[buffer(2)]],
    device const T*    cos_sin   [[buffer(3)]],
    uint3 tg  [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]])
{
    const uint base = (tg.x * MLA_HEADS + tg.y) * MLA_Q_HEAD;
    for (uint d = tid; d < MLA_Q_HEAD; d += MLA_ROW_THREADS) {
        out[base + d] = d < MLA_NOPE
            ? q[base + d]
            : T(mla_rotate(q + base + MLA_NOPE, d - MLA_NOPE, cos_sin, positions[tg.x]));
    }
}

// ── Cache write ─────────────────────────────────────────────────────────────
// buffer(0) = latent [rows, LATENT], buffer(1) = k_pe [rows, ROPE], buffer(2)
// = out [rows, ROPE] (the rotated key), buffer(3) = positions [rows],
// buffer(4) = cos_sin, buffer(5) = slot_mapping [rows] (0xFFFFFFFF: a padding
// row, written nowhere), buffer(6) / (7) = the layer's K / V chunk tables.
// Dispatch: threadgroups (rows, 1, 1), threads (ROW_THREADS, 1, 1), each
// thread every ROW_THREADS-th lane of its row.
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
    uint3 tg  [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]])
{
    const uint t = tg.x;
    const uint slot_raw = slot_mapping[t];
    for (uint e = tid; e < MLA_ROW; e += MLA_ROW_THREADS) {
        T v;
        if (e < MLA_LATENT) {
            v = latent[t * MLA_LATENT + e];
        } else {
            const uint d = e - MLA_LATENT;
            v = T(mla_rotate(k_pe + t * MLA_ROPE, d, cos_sin, positions[t]));
            out[t * MLA_ROPE + d] = v;
        }
        if (slot_raw != 0xFFFFFFFFu) {
            uint chunk;
            bool in_v;
            const uint at = mla_cache_at(slot_raw & 0x7FFFFFFFu, e, chunk, in_v);
            ((device T*)(in_v ? v_cache[chunk] : k_cache[chunk]))[at] = v;
        }
    }
}

// ── Absorbed attention ──────────────────────────────────────────────────────
// buffer(0) = out [rows, HEADS * LATENT], buffer(1) = q_latent [rows, HEADS *
// LATENT], buffer(2) = q [rows, HEADS * Q_HEAD] (rotated: its rope parts are
// read), buffer(3) = seq_used_k [seqs], buffer(4) = block_table [seqs,
// MAX_BLOCKS], buffer(5) / (6) = the layer's K / V chunk tables, buffer(7) =
// cu_seqlens_q [seqs + 1], buffer(8) = the partials (PARTS > 1): per row,
// head and partition, its softmax max, its sum, then its LATENT weighted
// lanes, all f32.
//
// Row `t` of sequence `s` attends to the first `seq_used_k[s] - (cu[s + 1] -
// 1 - t)` cached rows of `s`: its own position and every one before it. A
// head's score over a cached row is its latent query · the row's latent plus
// its rope part · the row's rotated key, both scaled by SCALE.
//
// Dispatch: threadgroups (rows, HEADS / HEADS_PER_TG, PARTS), threads (32 *
// SIMDGROUPS, 1, 1). Partition `p` takes a contiguous `1 / PARTS` of the row's
// cached rows, its simdgroups every SIMDGROUPS-th of them; lane `l` holds lanes
// `l + 32 j` of a cached row and of each served head's query. Its simdgroups'
// online softmaxes merge in threadgroup memory, a head at a time: into `out`
// when there is one partition, else into the partials `mla_attend_merge`
// combines. Fewer simdgroups leave each thread more registers for its heads.
constant constexpr uint MLA_SG = MLA_SIMDGROUPS_SET ? MLA_SIMDGROUPS : 1u;
constant constexpr uint MLA_ELEMS = (MLA_ROW + 31u) / 32u;
constant constexpr uint MLA_HPT = MLA_HEADS_PER_TG_SET ? MLA_HEADS_PER_TG : 1u;
constant constexpr uint MLA_NPARTS = MLA_PARTS_SET ? MLA_PARTS : 1u;
// A partial: its max, its sum, then its latent lanes.
constant constexpr uint MLA_PARTIAL = MLA_PARTIAL_STRIDE;
static_assert(!MLA_PARTS_SET || MLA_PARTIAL >= MLA_LATENT + 2u,
              "an MLA partial holds its max, its sum and its latent lanes");

template <typename T>
[[kernel]] void mla_attend(
    device       T*        out          [[buffer(0)]],
    device const T*        q_latent     [[buffer(1)]],
    device const T*        q            [[buffer(2)]],
    device const uint*     seq_used_k   [[buffer(3)]],
    device const uint*     block_table  [[buffer(4)]],
    device const uint64_t* k_cache      [[buffer(5)]],
    device const uint64_t* v_cache      [[buffer(6)]],
    device const uint*     cu_seqlens_q [[buffer(7)]],
    device       float*    partials     [[buffer(8)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint row = tg.x;
    const uint head0 = tg.y * MLA_HPT;
    const uint part = tg.z;
    uint seq = 0;
    while (cu_seqlens_q[seq + 1u] <= row) {
        ++seq;
    }
    const uint kv_len = seq_used_k[seq] - (cu_seqlens_q[seq + 1u] - 1u - row);
    const uint span = (kv_len + MLA_NPARTS - 1u) / MLA_NPARTS;
    const uint first = part * span;
    const uint last = min(first + span, kv_len);
    device const uint* blocks = block_table + seq * MLA_MAX_BLOCKS;

    float qr[MLA_HPT * MLA_ELEMS];
    float o[MLA_HPT * MLA_ELEMS];
    float m[MLA_HPT];
    float l[MLA_HPT];
    for (uint h = 0; h < MLA_HPT; ++h) {
        const uint head = head0 + h;
        for (uint j = 0; j < MLA_ELEMS; ++j) {
            const uint e = lane + 32u * j;
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
    for (uint pos = first + sg; pos < last; pos += MLA_SG) {
        const uint slot = (blocks[pos / MLA_BLOCK_SIZE] & 0x7FFFFFFFu) * MLA_BLOCK_SIZE
            + pos % MLA_BLOCK_SIZE;
        uint chunk;
        bool in_v;
        const uint at = mla_cache_at(slot, 0u, chunk, in_v);
        device const T* k_row = (device const T*)k_cache[chunk] + at;
        device const T* v_row = (device const T*)v_cache[chunk] + at;
        for (uint j = 0; j < MLA_ELEMS; ++j) {
            const uint e = lane + 32u * j;
            k[j] = e < MLA_HALF ? float(k_row[e]) : (e < MLA_ROW ? float(v_row[e - MLA_HALF]) : 0.0f);
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
                const bool latent_lane = lane + 32u * j < MLA_LATENT;
                o[h * MLA_ELEMS + j] = o[h * MLA_ELEMS + j] * factor + (latent_lane ? p * k[j] : 0.0f);
            }
            m[h] = m_new;
        }
    }

    // Merge the simdgroups' softmaxes, a head at a time: each simdgroup's max, sum and latent
    // lanes into threadgroup memory, then each output lane summed over them, rescaled to the
    // head's max.
    threadgroup float tg_m[MLA_HPT * MLA_SG];
    threadgroup float tg_l[MLA_HPT * MLA_SG];
    threadgroup float tg_o[MLA_SG * MLA_LATENT];
    if (lane == 0) {
        for (uint h = 0; h < MLA_HPT; ++h) {
            tg_m[h * MLA_SG + sg] = m[h];
            tg_l[h * MLA_SG + sg] = l[h];
        }
    }
    const uint tid = sg * 32u + lane;
    for (uint h = 0; h < MLA_HPT; ++h) {
        for (uint j = 0; j < MLA_ELEMS; ++j) {
            const uint e = lane + 32u * j;
            if (e < MLA_LATENT) {
                tg_o[sg * MLA_LATENT + e] = o[h * MLA_ELEMS + j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float m_all = -INFINITY;
        for (uint s = 0; s < MLA_SG; ++s) {
            m_all = max(m_all, tg_m[h * MLA_SG + s]);
        }
        float l_all = 0.0f;
        for (uint s = 0; s < MLA_SG; ++s) {
            const float ms = tg_m[h * MLA_SG + s];
            l_all += ms == -INFINITY ? 0.0f : tg_l[h * MLA_SG + s] * exp(ms - m_all);
        }
        const uint head = head0 + h;
        device float* partial = partials + ((row * MLA_HEADS + head) * MLA_NPARTS + part) * MLA_PARTIAL;
        if (MLA_NPARTS > 1u && tid == 0) {
            partial[0] = m_all;
            partial[1] = l_all;
        }
        for (uint e = tid; e < MLA_LATENT; e += 32u * MLA_SG) {
            float acc = 0.0f;
            for (uint s = 0; s < MLA_SG; ++s) {
                const float ms = tg_m[h * MLA_SG + s];
                acc += ms == -INFINITY ? 0.0f : tg_o[s * MLA_LATENT + e] * exp(ms - m_all);
            }
            if (MLA_NPARTS > 1u) {
                partial[2u + e] = acc;
            } else {
                out[(row * MLA_HEADS + head) * MLA_LATENT + e] = T(acc / l_all);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ── The partitions' merge ───────────────────────────────────────────────────
// buffer(0) = out [rows, HEADS * LATENT], buffer(1) = the partials
// `mla_attend` wrote. Each head's lanes: the PARTS partials' weighted lanes,
// each rescaled to the head's max over them, over their sums.
// Dispatch: threadgroups (rows, HEADS, 1), threads (MERGE_THREADS, 1, 1).
template <typename T>
[[kernel]] void mla_attend_merge(
    device       T*     out      [[buffer(0)]],
    device const float* partials [[buffer(1)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 at [[thread_position_in_threadgroup]])
{
    const uint tid = at.x;
    const uint row = tg.x;
    const uint head = tg.y;
    device const float* p = partials + (row * MLA_HEADS + head) * MLA_NPARTS * MLA_PARTIAL;
    float m_all = -INFINITY;
    for (uint i = 0; i < MLA_NPARTS; ++i) {
        m_all = max(m_all, p[i * MLA_PARTIAL]);
    }
    float l_all = 0.0f;
    for (uint i = 0; i < MLA_NPARTS; ++i) {
        const float mi = p[i * MLA_PARTIAL];
        l_all += mi == -INFINITY ? 0.0f : p[i * MLA_PARTIAL + 1u] * exp(mi - m_all);
    }
    for (uint e = tid; e < MLA_LATENT; e += MLA_MERGE_THREADS) {
        float acc = 0.0f;
        for (uint i = 0; i < MLA_NPARTS; ++i) {
            const float mi = p[i * MLA_PARTIAL];
            acc += mi == -INFINITY ? 0.0f : p[i * MLA_PARTIAL + 2u + e] * exp(mi - m_all);
        }
        out[(row * MLA_HEADS + head) * MLA_LATENT + e] = T(acc / l_all);
    }
}

SCRATCHY_KERNEL(mla_rope_q_f16, mla_rope_q<half>)
SCRATCHY_KERNEL(mla_rope_q_bf16, mla_rope_q<bfloat>)
SCRATCHY_KERNEL(mla_append_f16, mla_append<half>)
SCRATCHY_KERNEL(mla_append_bf16, mla_append<bfloat>)
SCRATCHY_KERNEL(mla_attend_f16, mla_attend<half>)
SCRATCHY_KERNEL(mla_attend_bf16, mla_attend<bfloat>)
SCRATCHY_KERNEL(mla_attend_merge_f16, mla_attend_merge<half>)
SCRATCHY_KERNEL(mla_attend_merge_bf16, mla_attend_merge<bfloat>)
