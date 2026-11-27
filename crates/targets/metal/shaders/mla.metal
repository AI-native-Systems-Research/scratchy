// SPDX-License-Identifier: Apache-2.0
//
// Multi-head latent attention's kernels: the query rotation
// (`MetalStep::MlaRopeQ`), the cache write (`MetalStep::MlaAppend`) and the
// absorbed attention over the cache (`MetalStep::MlaAttend`).
//
// The cache holds one row per token of a layer: the compressed latent
// (LATENT lanes) then the rotated shared rope key (ROPE lanes), `ROW` wide in
// all. The paged pool holds the row as its K and V halves, HALF lanes each: lane `e` of a row sits in the K store when `e < HALF`, else at
// `e - HALF` in the V store, both at the token's slot of the layer's one
// kv head. A block is `BLOCK_SIZE` rows; a store binding is the layer's chunk
// table (a physical block derefs `table[block / BLOCKS_PER_CHUNK]`, then block
// `block % BLOCKS_PER_CHUNK` of it; BLOCKS_PER_CHUNK 0: one base).
//
// The rope pairs a rope part's lanes as PAIRING says (the model's rope facts:
// adjacent `(2i, 2i + 1)`, or the halves `(i, i + ROPE / 2)`), each pair rotated
// by its frequency's cos and sin at the row's position (`cos_sin[pos]` holds the
// ROPE / 2 cosines, then the sines).
//
// Constants (compiled in; `MlaConstants`), every kernel's the same slots:
//   0 = HEADS       query heads
//   1 = LATENT      the compressed latent's width
//   2 = NOPE        a query head's un-rotated part
//   3 = ROPE        a query head's rotated part, the shared key's width
//   4 = BLOCK_SIZE  rows per paged block (with 5, 6 and 14: the cache's)
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
//  14 = HALF        the lanes of a row the pool's K store holds; the rest are
//                   the V store's
//  15 = PAIRING     the rope's lane pairing: 0 adjacent `(2i, 2i + 1)`, 1 the
//                   halves `(i, i + ROPE / 2)` (NeoX)
//  16 = WARP_LANES  the tiled attention's: the latent lanes of a warp's output
//  17 = KEYS        the tiled attention's: the cached rows a warp scores a step
//  18 = BLOCKS      the tiled attention's: the 16-query blocks of a threadgroup

#include <metal_stdlib>
#include "baked.h"
#include "mla.h"
#include "mlx_steel_attn/attn.h"
#include "mlx_steel_attn/attn_ops.h"

using namespace metal;

constant constexpr uint MLA_HALF_ROPE = MLA_ROPE / 2u;

// Lane `d` of a rope part `x` (ROPE lanes) rotated at `pos`. Interleaved
// (PAIRING 0): its partner is `d ^ 1`, its frequency `d / 2`. Half-split
// (PAIRING 1, NeoX): its partner is `d ± ROPE / 2`, its frequency `d % (ROPE / 2)`.
template <typename T>
inline float mla_rotate(device const T* x, uint d, device const T* cos_sin, uint pos) {
    device const T* row = cos_sin + pos * MLA_ROPE;
    const bool interleaved = MLA_PAIRING == 0u;
    const uint f = interleaved ? d / 2u : d % MLA_HALF_ROPE;
    const bool first = interleaved ? (d & 1u) == 0u : d < MLA_HALF_ROPE;
    const uint i0 = interleaved ? (d & ~1u) : f;
    const uint i1 = interleaved ? (d | 1u) : f + MLA_HALF_ROPE;
    const float c = float(row[f]);
    const float s = float(row[MLA_HALF_ROPE + f]);
    const float x0 = float(x[i0]);
    const float x1 = float(x[i1]);
    return first ? x0 * c - x1 * s : x1 * c + x0 * s;
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

// ── Tiled absorbed attention (a matmul-band bucket) ─────────────────────────
// The absorbed attention is one key head every query head shares: over a
// sequence's `rows x HEADS` queries (row-major, a row's heads consecutive —
// exactly `q_latent`'s and `out`'s rows), each a LATENT + ROPE query (its
// latent query, then its rotated rope part) against the cached rows, its
// output the softmax-weighted latent lanes of the rows. This kernel is the
// steel wide paged attention's FA-2 (`attention_steel_paged_wide`) over those
// queries: a threadgroup takes BLOCKS blocks of 16 of them, a block LATENT /
// WARP_LANES warps; warp `wn` of a block scores page group `wn` of a step (KEYS cached
// rows) over the whole query, the block's scores cross its warps through
// threadgroup memory once a step, and each warp applies the online softmax
// and accumulates its WARP_LANES-lane slice of the output. Every fragment
// loads straight from device memory: a query fragment from `q_latent` or
// `q`, a cached row's from the K or V store (HALF is whole fragments, so no
// fragment straddles the two). The heads of a row share its keys and its
// causal limit; a query attends to the cached rows up to its row's position.
//
// Bindings as `mla_attend`'s, less the partials (buffer(8)).
// Dispatch: threadgroups (ceil(rows * HEADS / (16 * BLOCKS)), 1, seqs), threads
// (32 * BLOCKS * LATENT / WARP_LANES, 1, 1); threadgroups past a sequence's
// queries return.
// Unset (another kernel's compile of this file): one fragment each, never dispatched.
constant constexpr uint MLA_TILE_LANES = MLA_WARP_LANES_SET ? MLA_WARP_LANES : 8u;
constant constexpr uint MLA_TILE_KEYS = MLA_KEYS_SET ? MLA_KEYS : 8u;
static_assert(!MLA_WARP_LANES_SET || (MLA_LATENT % MLA_WARP_LANES == 0 && MLA_WARP_LANES % 8u == 0),
              "whole fragments a warp");
static_assert(!MLA_WARP_LANES_SET || (MLA_ROPE % 8u == 0 && MLA_HALF % 8u == 0),
              "a fragment never straddles the query's parts or the row's stores");
static_assert(!MLA_KEYS_SET || (MLA_KEYS % 8u == 0 && MLA_BLOCK_SIZE % MLA_KEYS == 0),
              "whole page groups a step");

template <typename T>
[[kernel, max_total_threads_per_threadgroup(MLA_TILE_THREADS)]]
void mla_attend_tiles(
    device       T*        out          [[buffer(0)]],
    device const T*        q_latent     [[buffer(1)]],
    device const T*        q            [[buffer(2)]],
    device const uint*     seq_used_k   [[buffer(3)]],
    device const uint*     block_table  [[buffer(4)]],
    device const uint64_t* k_cache      [[buffer(5)]],
    device const uint64_t* v_cache      [[buffer(6)]],
    device const uint*     cu_seqlens_q [[buffer(7)]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid           [[threadgroup_position_in_grid]])
{
    using namespace mlx::steel;
    typedef float AccumType;
    constexpr short kU = 8;                    // a fragment's edge
    constexpr short ROWS = 16;                 // queries a warp's block
    constexpr int TQ = ROWS / kU;
    constexpr int WM = int(MLA_TILE_BLOCKS);
    constexpr int BQ = WM * ROWS;
    constexpr int WN = int(MLA_TILE_WN);
    constexpr int DW = int(MLA_TILE_LANES);
    constexpr int TDW = DW / kU;
    constexpr int KQ = int(MLA_TILE_KEYS);     // a step's page group a warp, and a K-tile
    constexpr short TK = KQ / kU;
    constexpr int QK_FRAGS = int(MLA_ROW) / kU;
    constexpr int LATENT_FRAGS = int(MLA_LATENT) / kU;
    constexpr int HALF_FRAGS = int(MLA_HALF) / kU;
    constexpr int ld_lat = int(MLA_LATENT);
    constexpr int ld_q = int(MLA_Q_HEAD);
    constexpr int ld_row = int(MLA_HALF);

    const uint seq = tid.z;
    const uint seq_start = cu_seqlens_q[seq];
    const uint new_q = cu_seqlens_q[seq + 1u] - seq_start;
    const uint kv_len = seq_used_k[seq];
    const uint prefix = kv_len - new_q;
    const uint queries = new_q * MLA_HEADS;
    const uint item0 = tid.x * uint(BQ);
    if (item0 >= queries) {
        return;
    }
    const uint tile_items = min(uint(BQ), queries - item0);

    using MMAFrag_acc_t = BaseMMAFrag<AccumType, kU, kU>;
    using otile_t = MMATile<AccumType, TQ, TDW, MMAFrag_acc_t>;
    using stile_t = MMATile<AccumType, TQ, TK, MMAFrag_acc_t>;

    const short rb = short(simd_group_id / uint(WN));
    const short wn = short(simd_group_id % uint(WN));
    const short tm = ROWS * rb;
    const short lim_rows = short(int(tile_items) - int(tm));
    // A block past the tile's queries still meets every step: it reads the tile's first queries
    // and stores nothing.
    const uint load_item = seq_start * MLA_HEADS + item0 + (lim_rows > 0 ? uint(tm) : 0u);
    device const T* ql = q_latent + load_item * MLA_LATENT;
    device const T* qr = q + load_item * MLA_Q_HEAD + MLA_NOPE;

    const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
    const short sm = simd_coord.y;
    const short sn = simd_coord.x;

    constexpr short kRowsPT = otile_t::kRowsPerThread;
    AccumType max_score[kRowsPT];
    AccumType sum_score[kRowsPT] = {0};
    for (short i = 0; i < kRowsPT; ++i) {
        max_score[i] = Limits<AccumType>::finite_min;
    }
    otile_t Otile;
    Otile.clear();

    // Positions: a query's is its row's, `prefix + item / HEADS`.
    const int pos_block = int(prefix + (item0 + uint(tm)) / MLA_HEADS);
    const int pos_last = int(prefix + (item0 + tile_items - 1u) / MLA_HEADS);
    const int kv_tiles = int((kv_len + uint(KQ) - 1u) / uint(KQ));
    const int kb_lim = min((pos_last + KQ) / KQ, kv_tiles);
    const int num_pages = int((kv_len + MLA_BLOCK_SIZE - 1u) / MLA_BLOCK_SIZE);
    device const uint* blocks = block_table + seq * MLA_MAX_BLOCKS;
    const AccumType scale2 = AccumType(MLA_SCALE) * AccumType(M_LOG2E_F);

    auto rows_at = [&](device const uint64_t* store, int key) -> device const T* {
        return mla_rows_at<T>(store, blocks, num_pages, key);
    };

    // S = Q K^T of page group `g` over the whole query: its latent lanes then its rope lanes,
    // against the rows' lanes in the K store then the V store.
    auto do_qk_group = [&](int g, thread stile_t& Sg) {
        const short klim = short(clamp(int(kv_len) - g * KQ, 0, KQ));
        const short rows = max(min(lim_rows, ROWS), short(1));
        device const T* kg = rows_at(k_cache, g * KQ);
        device const T* vg = rows_at(v_cache, g * KQ);
        STEEL_PRAGMA_UNROLL
        for (short dd = 0; dd < QK_FRAGS; dd++) {
            MMATile<AccumType, TQ, 1> Qt;
            MMATile<AccumType, 1, TK> Kt;
            const bool latent = dd < LATENT_FRAGS;
            const int ld = latent ? ld_lat : ld_q;
            device const T* qsrc = (latent ? ql + dd * kU : qr + (dd - LATENT_FRAGS) * kU)
                + int(sm) * ld + int(sn);
            if (rows == ROWS) {
                Qt.template load<T, 1, 1>(qsrc, ld);
            } else {
                Qt.template load_safe<T, 1, 1>(qsrc, ld, short2(kU, short(rows - sm)));
            }
            device const T* ksrc = (dd < HALF_FRAGS ? kg + dd * kU : vg + (dd - HALF_FRAGS) * kU)
                + int(sm);
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < TK; j++) {
                const short k0 = j * kU;
                device const T* src = ksrc + int(k0 + sn) * ld_row;
                if (k0 + kU <= klim) {
                    MMAFrag_acc_t::load(Kt.frag_at(0, j), src, Int<1>{}, ld_row);
                } else {
                    MMAFrag_acc_t::load_safe(
                        Kt.frag_at(0, j), src, Int<1>{}, ld_row, short(kU),
                        short(klim - k0 - sn), 0, 0);
                }
            }
            tile_matmad(Sg, Qt, Kt, Sg);
        }
    };

    auto scale_and_mask = [&](int kb, thread stile_t& S) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < stile_t::kElemsPerTile; i++) {
            S.elems()[i] *= scale2;
        }
        const int key0 = kb * KQ;
        if (key0 + KQ > int(kv_len) || key0 + KQ - 1 > pos_block) {
            constexpr auto neg_inf = Limits<AccumType>::finite_min;
            STEEL_PRAGMA_UNROLL
            for (short iq = 0; iq < TQ; iq++) {
                const int pos = int(prefix + (item0 + uint(tm + sm + iq * kU)) / MLA_HEADS);
                STEEL_PRAGMA_UNROLL
                for (short ik = 0; ik < TK; ik++) {
                    thread auto& fg = S.frag_at(iq, ik);
                    STEEL_PRAGMA_UNROLL
                    for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
                        const int c = key0 + ik * kU + sn + jj;
                        fg[jj] = (c >= int(kv_len) || c > pos) ? neg_inf : fg[jj];
                    }
                }
            }
        }
    };

    auto softmax_update = [&](thread stile_t& S) {
        AccumType new_max[kRowsPT];
        AccumType factor[kRowsPT];
        for (short i = 0; i < kRowsPT; ++i) {
            new_max[i] = max_score[i];
        }
        S.template row_reduce<MaxOp>(new_max);
        S.template row_bin_op<ExpSubOp>(new_max);
        for (short i = 0; i < kRowsPT; ++i) {
            factor[i] = fast::exp2(max_score[i] - new_max[i]);
            max_score[i] = new_max[i];
            sum_score[i] = sum_score[i] * factor[i];
        }
        S.template row_reduce<SumOp>(sum_score);
        Otile.template row_bin_op<MulOp>(factor);
    };

    // O += P V over K-tile `kb`: this warp's latent lanes of its rows, from the K store below
    // HALF, the V store past it.
    auto do_pv = [&](int kb, thread stile_t& S) {
        const int key0 = kb * KQ;
        device const T* kg = rows_at(k_cache, key0);
        device const T* vg = rows_at(v_cache, key0);
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TDW; id++) {
            const int lane = int(wn) * DW + id * kU;
            device const T* vp = (lane < int(MLA_HALF) ? kg + lane : vg + (lane - int(MLA_HALF)))
                + int(sn);
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
                const int k_abs = key0 + ik * kU;
                typename MMAFrag_acc_t::frag_type Vf;
                device const T* vsrc = vp + int(ik * kU + sm) * ld_row;
                if (k_abs + kU <= int(kv_len)) {
                    MMAFrag_acc_t::load(Vf, vsrc, ld_row, Int<1>{});
                } else {
                    MMAFrag_acc_t::load_safe(
                        Vf, vsrc, ld_row, Int<1>{},
                        short(clamp(int(kv_len) - k_abs - sm, 0, int(kU))), short(kU), 0, 0);
                }
                STEEL_PRAGMA_UNROLL
                for (short iq = 0; iq < TQ; iq++) {
                    MMAFrag_acc_t::mma(
                        Otile.frag_at(iq, id), S.frag_at(iq, ik), Vf, Otile.frag_at(iq, id));
                }
            }
        }
    };

    // A block's scores of a step, lane by lane (lane-preserving, as the wide kernel's), a buffer
    // a step parity where two fit.
    constexpr int kSPerLane = stile_t::kElemsPerTile;
    constexpr int kPart = 32 * kSPerLane;
    constexpr int kBufs = 2 * WM * WN * kPart * int(sizeof(AccumType)) <= 32768 ? 2 : 1;
    threadgroup AccumType sx[kBufs * WM * WN * kPart];

    for (int kb0 = 0; kb0 < kb_lim; kb0 += WN) {
        const int g = kb0 + wn;
        stile_t Sg;
        Sg.clear();
        if (g < kb_lim) {
            do_qk_group(g, Sg);
        }
        threadgroup AccumType* part = sx + (kb0 / WN % kBufs) * (WM * WN * kPart);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kSPerLane; i++) {
            part[(simd_group_id * kSPerLane + i) * 32 + simd_lane_id] = Sg.elems()[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (short t = 0; t < WN; t++) {
            const int kb = kb0 + t;
            if (kb >= kb_lim) {
                break;
            }
            stile_t S;
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kSPerLane; i++) {
                S.elems()[i] = part[((rb * WN + t) * kSPerLane + i) * 32 + simd_lane_id];
            }
            scale_and_mask(kb, S);
            softmax_update(S);
            do_pv(kb, S);
        }
        if (kBufs == 1) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    AccumType rcp[kRowsPT];
    for (short i = 0; i < kRowsPT; ++i) {
        rcp[i] = 1.f / sum_score[i];
    }
    Otile.template row_bin_op<MulOp>(rcp);
    if (lim_rows <= 0) {
        return;
    }
    device T* o = out + (seq_start * MLA_HEADS + item0 + uint(tm)) * MLA_LATENT
        + uint(wn) * uint(DW) + uint(sm) * MLA_LATENT + uint(sn);
    if (lim_rows < ROWS) {
        const short2 dims = short2(short(DW) - sn, lim_rows - sm);
        if (dims.x <= 0 || dims.y <= 0) {
            return;
        }
        Otile.template store_safe<T, 1, 1>(o, ld_lat, dims);
    } else {
        Otile.template store<T, 1, 1>(o, ld_lat);
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
SCRATCHY_KERNEL(mla_attend_tiles_f16, mla_attend_tiles<half>)
SCRATCHY_KERNEL(mla_attend_tiles_bf16, mla_attend_tiles<bfloat>)
SCRATCHY_KERNEL(mla_attend_merge_f16, mla_attend_merge<half>)
SCRATCHY_KERNEL(mla_attend_merge_bf16, mla_attend_merge<bfloat>)
