// SPDX-License-Identifier: Apache-2.0
//
// Scratchy metal attention kernels — direct ports of MLX attention.
//
//   `attention_via_cache_v2_f16/bf16_specialized` —
//       paged-cache adaptation of MLX `sdpa_vector` from
//       `mlx/backend/metal/kernels/sdpa_vector.h`. Decode path.
//   `attention_prefill_sdpa_v2_paged_f16/bf16_specialized` —
//       Paged-cache prefill, shares the decode kernel's outer
//       structure (1 Q per TG, online softmax + per-simdgroup
//       K-axis split). K/V read through `block_table` indirection;
//       K-axis covers the FULL `seqused_k[seq]` (prefix + new), and
//       the per-Q causal mask shifts by `(seqused_k[seq] -
//       new_q_for_seq)` to account for prior cached prefix.
//       Required for chunked prefill, prefix caching, mixed
//       prefill/decode batches, and multi-turn chat continuation.
//       The only prefill kernel emitted on metal post-Phase B
//       (Phase B macro adapter at metal/attention.rs always emits
//       `Instruction::AttentionPrefillPaged`).
//
// Function constant indices (must match `pipelines::constants_for`):
//   0  ATTN_HEAD_DIM           uint
//   1  ATTN_NUM_Q_HEADS        uint
//   2  ATTN_NUM_KV_HEADS       uint
//   3  ATTN_SCALE_FC           float
//   4  ATTN_BLOCK_SIZE         uint
//   5  ATTN_MAX_BLOCKS_PER_SEQ uint

#include <metal_stdlib>
using namespace metal;



constant uint  ATTN_HEAD_DIM           [[function_constant(0)]];
constant uint  ATTN_NUM_Q_HEADS        [[function_constant(1)]];
constant uint  ATTN_NUM_KV_HEADS       [[function_constant(2)]];
constant float ATTN_SCALE_FC           [[function_constant(3)]];
constant uint  ATTN_BLOCK_SIZE         [[function_constant(4)]];
constant uint  ATTN_MAX_BLOCKS_PER_SEQ [[function_constant(5)]];
// Reactive (chunked) KV pool: the `k_cache`/`v_cache` bindings are
// per-layer chunk-address TABLES (device uint64 gpuAddresses), not the
// cache buffers. A resolved physical block id derefs
// `table[physical_block / BLOCKS_PER_CHUNK]` then addresses with
// `physical_block % BLOCKS_PER_CHUNK`. See
// scratchy-target-metal's `BLOCKS_PER_CHUNK`. (`attention_via_cache_v2_*`
// reads constant slot 6 via `AttentionViaCacheConstants`;
// `attention_prefill_sdpa_v2_paged_*` via `AttentionPrefillPagedConstants`.)
constant uint  ATTN_BLOCKS_PER_CHUNK   [[function_constant(6)]];

// Sliding-window attention (Gemma2/3/4 alternating layers). A query at
// absolute position `q` attends to keys `k` with `0 <= q - k < window`
// (self + window-1 prior — matches both mlx `create_causal_mask`
// `linds < rinds + window_size` and HF's `(q-k) >= window` masking).
// `0` disables the window entirely; the compiler folds the checks away
// for non-sliding pipelines (full-attention models pass 0).
constant int   ATTN_WINDOW             [[function_constant(7)]];

// Cap on `seq_used_k[seq]` the shared-logits buffer can hold.
// Each token uses 4 bytes; this cap × 4 == threadgroup memory bytes
// dedicated to the partial-logits scratch. Smaller is better for
// concurrent-threadgroup occupancy on Apple GPU clusters (each
// cluster reserves the per-threadgroup TGSM budget per resident
// group). 256 * 4 = 1KB still covers TinyLlama-class decode lengths
// without spilling logits to device memory; sequences longer than
// this cap fall back to a larger-cap pipeline (TODO).
#define ATTN_MAX_SHARED_LOGITS 256u

// ── Rope-on-read (spans / position-independent KV) ──────────────────
//
// Span (relocatable) K blocks are stored UNROTATED; attention re-ropes
// each cached K to the reader's own position on read, so one cached copy
// is shared across every reuse position (zero-copy block sharing). These
// constants are OPTIONAL: when a pipeline does not set slot 10, the
// `is_function_constant_defined` guard folds the whole rotation path away
// and buffers 6/7 are never accessed — non-spans pipelines are byte-
// identical to before. Set by `AttentionViaCacheConstants` only when
// rope-on-read is active.
//   8  ATTN_ROT_DIM       uint  — rotary dim (full head_dim for NeoX;
//                                  partial_rotary_factor*head_dim for
//                                  gemma4 global proportional rope)
//   9  ATTN_PAIR_OFF      uint  — NeoX pairing offset: rot_dim/2 for full
//                                  rope, head_dim/2 for proportional rope
//                                  (MUST match rope_append's ROPE_PAIR_OFF)
//  10  ATTN_ROPE_ON_READ  uint  — 0/1 master switch
constant uint ATTN_ROT_DIM      [[function_constant(8)]];
constant uint ATTN_PAIR_OFF     [[function_constant(9)]];
constant uint ATTN_ROPE_ON_READ [[function_constant(10)]];
constant bool ATTN_ROR_DEFINED  = is_function_constant_defined(ATTN_ROPE_ON_READ);
constant uint ATTN_ROR          = ATTN_ROR_DEFINED ? ATTN_ROPE_ON_READ : 0u;

// ── Rope-once-to-scratch (spans, gqa_shared path) ───────────────────
//
//  11  ATTN_K_SCRATCH  uint  — 0/1. When set, the gqa_shared prefill kernel
//      reads PRE-ROPED K from a dense logical-block-indexed scratch buffer
//      (written ONCE by `rope_once_gqa_shared_*`) instead of re-roping each
//      staged K tile in smem on every q-tile. The scratch mirrors the
//      cache's per-block strides, so the staging load is byte-identical to a
//      cache read — there is no per-tile rotation in the attention. This is
//      the gqa_shared twin of steel/NAX's `rope_once_{steel,nax}` (head_dim
//      512 has no steel instantiation, so gemma4 global prefill falls to
//      gqa_shared and pays the per-tile rope without this). When unset the
//      `is_function_constant_defined` guard folds the scratch path away and
//      the kernel keeps the in-kernel cos_sin rotation (or, with ROR also
//      unset, the byte-identical non-spans cache read). The rope-once kernel
//      itself reads ATTN_ROT_DIM / ATTN_PAIR_OFF / the per-block bit-31 flag.
constant uint ATTN_K_SCRATCH        [[function_constant(11)]];
constant bool ATTN_K_SCRATCH_DEF    = is_function_constant_defined(ATTN_K_SCRATCH);
constant uint ATTN_KSCR             = ATTN_K_SCRATCH_DEF ? ATTN_K_SCRATCH : 0u;

// ── Co-resident NeoX-pair lane layout (decode rope-on-read) ─────────
//
//  12  ATTN_PAIR_CORESIDENT  uint — 0/1. Decode kernels only. When 1
//      each lane owns its NeoX pairs `{d, d+half_dim}` co-resident (j in
//      [0,np) → first-side element `base + j`; j in [np,qk) → its pair
//      `base + (j-np) + half_dim`, where np = qk/2, base = simd_lid*np).
//      The on-read rope then has BOTH pair members in-lane: no
//      `simd_shuffle`, no `k_pair[]` staging array (the shuffle was ~45%
//      of the decode rope overhead). Set by the lowering ONLY for full
//      NeoX rope (rot_dim == head_dim, pair_off == head_dim/2); the
//      addressing helpers below fold to the plain contiguous slice when
//      this is 0 (every non-spans dispatch + the A/B-off case), keeping
//      that path byte-identical.
constant uint ATTN_PAIR_CORESIDENT     [[function_constant(12)]];
constant bool ATTN_PAIR_CORESIDENT_DEF = is_function_constant_defined(ATTN_PAIR_CORESIDENT);
constant uint ATTN_PCR                 = ATTN_PAIR_CORESIDENT_DEF ? ATTN_PAIR_CORESIDENT : 0u;

// Per-lane element offset (within a key's head_dim row) for local index
// `j` in [0, qk_per_thread). Contiguous layout: `simd_lid*qk + j`.
// Co-resident layout: the lane owns `np = qk/2` first-side elements then
// their `+half_dim` pairs, so element `j` is in-lane paired with element
// `j ± np`. half_dim = head_dim/2 (full NeoX, the only case PCR is set).
inline uint attn_elem_off(uint simd_lid, uint j, uint qk_per_thread, uint head_dim) {
    if (ATTN_PCR != 0u) {
        const uint np    = qk_per_thread / 2u;
        const uint base  = simd_lid * np;
        const uint halfd = head_dim / 2u;
        return (j < np) ? (base + j) : (base + (j - np) + halfd);
    }
    return simd_lid * qk_per_thread + j;
}

// Rotate this lane's contiguous K slice to the reuse position whose
// cos/sin row is `cos_row`/`sin_row` ([0,half_dim) wide each), matching
// `rope_append`'s NeoX pairing EXACTLY (rope.metal): element at absolute
// index `d` pairs with `d + pair_off`; the rotary index into cos/sin is
// `d` (first side) or `d - pair_off` (second side), both in [0,half_dim).
// For proportional rope (pair_off > half_dim) the lanes whose elements
// fall outside the two rotary sub-ranges pass through unchanged.
//
// In `attention_via_cache_v2` / `attention_prefill_sdpa_v2_paged`, the 32
// lanes of a simdgroup split one key's head_dim into `qk_per_thread`-wide
// contiguous slices, so element `d`'s pair `d + pair_off` lives in lane
// `simd_lid ± pair_off/qk_per_thread`, same local index — reachable via
// `simd_shuffle`. MUST be called simdgroup-uniformly; the caller gates on
// a per-physical-block flag that is uniform across the simdgroup (all
// lanes process the same key), so there is no shuffle divergence.
// TK = the KV-cache element type (half / bfloat). The rotated K is rounded
// back to TK so it bit-matches what `rope_append` STORED — fp16(x0*c - x1*s) —
// making rope-on-read transparent vs rotate-on-write under greedy decode.
template <typename TK, typename TCS>
inline void rope_on_read_k_slice(
    thread float*    k_loc,          // [qk_per_thread] this lane's K (in/out)
    uint             qk_per_thread,
    uint             simd_lid,
    device const TCS* cos_row,       // cos for the reuse position, [0,half_dim)
    device const TCS* sin_row,       // sin for the reuse position, [0,half_dim)
    uint             half_dim,
    uint             pair_off)
{
    const uint base_d        = simd_lid * qk_per_thread;
    const uint pair_lane_off = pair_off / qk_per_thread;
    const bool first_side    = base_d < half_dim;
    const bool second_side   = (base_d >= pair_off) && (base_d < pair_off + half_dim);
    const uint src_lane      = first_side  ? (simd_lid + pair_lane_off)
                             : second_side ? (simd_lid - pair_lane_off)
                                           : simd_lid;
    // Uniform shuffle: every lane fetches its pair's slice from `src_lane`
    // (pass-through lanes fetch themselves and discard the result).
    float k_pair[16];
    for (uint j = 0; j < qk_per_thread; ++j) {
        k_pair[j] = simd_shuffle(k_loc[j], src_lane);
    }
    if (first_side) {
        for (uint j = 0; j < qk_per_thread; ++j) {
            const uint d = base_d + j;                  // rotary index < half_dim
            const float c = float(cos_row[d]);
            const float s = float(sin_row[d]);
            // Round to cache dtype TK to match rope_append's stored fp16.
            k_loc[j] = float(TK(k_loc[j] * c - k_pair[j] * s)); // x0*c - x1*s
        }
    } else if (second_side) {
        for (uint j = 0; j < qk_per_thread; ++j) {
            const uint d = base_d + j - pair_off;       // rotary index < half_dim
            const float c = float(cos_row[d]);
            const float s = float(sin_row[d]);
            k_loc[j] = float(TK(k_loc[j] * c + k_pair[j] * s)); // x1*c + x0*s
        }
    }
}

// In-lane NeoX rope for the CO-RESIDENT layout (ATTN_PAIR_CORESIDENT).
// This lane holds, for each pair p in [0,np): k_loc[p] = element
// `base + p` (first side) and k_loc[np + p] = its pair
// `base + p + head_dim/2` (second side). Both members are in
// registers, so the rotation needs NO simd_shuffle and NO staging
// array. Bit-matches `rope_on_read_k_slice` / `rope_append` (round to
// the cache dtype TK).
//
// FULL NeoX (rot_dim == head_dim): `half_dim == head_dim/2`, so every
// pair's first-side absolute index `base + p` is a valid rotary index
// (< half_dim) and all pairs rotate.
//
// PROPORTIONAL (rot_dim < head_dim, gemma4 global hd512/rot128): the
// co-resident pair offset is still `head_dim/2` (element `d` pairs with
// `d + head_dim/2`, matching `rope_once`'s `pair_off == head_dim/2`),
// but only the first `half_dim == rot_dim/2` first-side elements are
// rotated — elements with `base + p >= half_dim` are outside the rotary
// window and pass through unchanged (both pair members already loaded).
// The rotary index into cos/sin is the absolute first-side index
// `base + p` either way, matching `rope_once`'s `cr[d]` / `cr[half_dim+d]`.
template <typename TK, typename TCS>
inline void rope_on_read_k_pairs_inlane(
    thread float*    k_loc,          // [qk_per_thread] this lane's K (in/out)
    uint             qk_per_thread,
    uint             simd_lid,
    device const TCS* cos_row,       // cos for the reuse position, [0,half_dim)
    device const TCS* sin_row,       // sin for the reuse position, [0,half_dim)
    uint             half_dim)       // rotary half = rot_dim/2 (<= head_dim/2)
{
    const uint np   = qk_per_thread / 2u;
    const uint base = simd_lid * np;                    // first-side abs index
    for (uint p = 0; p < np; ++p) {
        const uint d = base + p;                        // first-side abs index
        if (d >= half_dim) continue;                    // outside rotary window
        const float c  = float(cos_row[d]);
        const float s  = float(sin_row[d]);
        const float x0 = k_loc[p];                      // element d
        const float x1 = k_loc[np + p];                 // element d + head_dim/2
        k_loc[p]      = float(TK(x0 * c - x1 * s));      // first side
        k_loc[np + p] = float(TK(x1 * c + x0 * s));      // second side
    }
}

// ── TurboQuant KV read (decode) ──────────────────────────────────────
//
//  13  ATTN_TQ_BITS  uint — codebook width (bits per code) of the TurboQuant
//      packed KV store. Set on the TurboQuant twin of the decode command and
//      on the prefill staging pass (`tq_stage_rotated`); unset, the
//      `is_function_constant_defined` guard folds the decode path away and
//      buffers 7..13 are never accessed.
//
// A TurboQuant vector is stored as codes `c` into an N(0,1) codebook plus its
// L2 norm, and decodes as `x = norm · s² · D · H · c` (turboquant.metal: D =
// diag(signs), H the unnormalized Walsh-Hadamard transform, s² = 1/head_dim).
// H is symmetric and D diagonal, so attention can stay in the codebook domain:
//   q · x_t      = norm_t · (s² · H · D · q) · c_t       — rotate q once;
//   Σ_t p_t x_t  = s² · D · H · (Σ_t p_t · norm_t · c_t)  — rotate the output once.
// No key is ever decoded, so the per-layer full-context dequant pass is gone
// for decode.
constant uint ATTN_TQ_BITS [[function_constant(13)]];
constant bool ATTN_TQ_DEF  = is_function_constant_defined(ATTN_TQ_BITS);
constant uint ATTN_TQ      = ATTN_TQ_DEF ? ATTN_TQ_BITS : 0u;

// Unnormalized Walsh-Hadamard transform (H·x) of the head_dim vector a
// simdgroup holds as `qk_per_thread` elements per lane (`attn_elem_off`
// ownership). Under both the contiguous and the co-resident layout the bits of
// an element's index are a permutation of (the local index's bits, the lane's 5
// bits), and H factors into one 2x2 butterfly per index bit — so a register
// butterfly per local bit plus a `simd_shuffle_xor` butterfly per lane bit is
// H·x for either layout. MUST be called simdgroup-uniformly.
inline void tq_wht(thread float* x, uint qk_per_thread, uint simd_lid) {
    for (uint h = 1; h < qk_per_thread; h <<= 1) {
        for (uint j = 0; j < qk_per_thread; ++j) {
            if ((j & h) == 0u) {
                const float a = x[j];
                const float b = x[j | h];
                x[j]     = a + b;
                x[j | h] = a - b;
            }
        }
    }
    for (ushort m = 1; m < 32; m <<= 1) {
        for (uint j = 0; j < qk_per_thread; ++j) {
            const float o = simd_shuffle_xor(x[j], m);
            x[j] = (simd_lid & m) ? (o - x[j]) : (x[j] + o);
        }
    }
}

// Code of element `e` of the packed vector starting at word `row_word`:
// `32 / bits` codes per u32, LSB first, none straddling a word (turboquant.metal).
inline uint tq_code(device const uint* packed, uint row_word, uint e) {
    const uint vpw = 32u / ATTN_TQ;
    return (packed[row_word + e / vpw] >> ((e % vpw) * ATTN_TQ)) & ((1u << ATTN_TQ) - 1u);
}

// Spans rope-on-read: rotate this lane's K slice to key position `i`.
template <typename T>
inline void attn_rope_on_read(thread float* k_loc, uint qk_per_thread, uint simd_lid,
                              uint i, device const T* cos_sin) {
    const uint half_dim = ATTN_ROT_DIM / 2u;
    device const T* cos_row = cos_sin + i * ATTN_ROT_DIM;
    device const T* sin_row = cos_row + half_dim;
    if (ATTN_PCR != 0u) {
        // Co-resident: both pair members in-lane, no shuffle/staging.
        rope_on_read_k_pairs_inlane<T>(k_loc, qk_per_thread, simd_lid,
                                       cos_row, sin_row, half_dim);
    } else {
        rope_on_read_k_slice<T>(k_loc, qk_per_thread, simd_lid,
                                cos_row, sin_row, half_dim, ATTN_PAIR_OFF);
    }
}

// ── TurboQuant prefill: the rotated-domain KV image ─────────────────────
//
// Prefill attention — every kernel family, unchanged — runs in the codebook's
// rotated domain. R = s·H·D is orthonormal, so q·k = (R·q)·(R·k) and
// Σ p·v = Rᵀ·Σ p·(R·v). `tq_stage_rotated` writes R·k (or R·v) for every key
// of the step's sequences into the layer's cache scratch; `tq_rotate_rows`
// turns q into R·q before the attention and its output back with Rᵀ after it.
// A cached key needs no transform at all — R·x̃ = norm·s·centroid[code] — so the
// pass is a table lookup over the context, and a Walsh-Hadamard transform only
// over the step's new keys.
//
// Span blocks (block_table bit 31) are re-roped by the attention itself
// (rope-on-read at key position i), so they are stored as rope₋ᵢ(R·ropeᵢ(k)):
// the attention's own ropeᵢ turns that into R·ropeᵢ(k).

// NeoX rope of this lane's contiguous slice (`simd_lid*qk + j`) to position
// `i`, forward (`sign` 1) or inverse (-1). Pairs `(d, d + ATTN_PAIR_OFF)` for
// `d < ATTN_ROT_DIM/2`, the partner slice fetched from lane
// `simd_lid ± ATTN_PAIR_OFF/qk` (as `rope_on_read_k_slice`). Simdgroup-uniform.
template <typename T>
inline void tq_rope_slice(thread float* x, uint qk_per_thread, uint simd_lid, uint i,
                          device const T* cos_sin, float sign) {
    const uint half_dim = ATTN_ROT_DIM / 2u;
    const uint pair_off = ATTN_PAIR_OFF;
    device const T* cos_row = cos_sin + i * ATTN_ROT_DIM;
    device const T* sin_row = cos_row + half_dim;
    const uint base_d = simd_lid * qk_per_thread;
    const bool first_side = base_d < half_dim;
    const bool second_side = (base_d >= pair_off) && (base_d < pair_off + half_dim);
    const uint src = first_side  ? simd_lid + pair_off / qk_per_thread
                   : second_side ? simd_lid - pair_off / qk_per_thread
                                 : simd_lid;
    float pair[16];
    for (uint j = 0; j < qk_per_thread; ++j) {
        pair[j] = simd_shuffle(x[j], src);
    }
    for (uint j = 0; j < qk_per_thread; ++j) {
        if (first_side) {
            const uint d = base_d + j;
            x[j] = x[j] * float(cos_row[d]) - pair[j] * sign * float(sin_row[d]);
        } else if (second_side) {
            const uint d = base_d + j - pair_off;
            x[j] = x[j] * float(cos_row[d]) + pair[j] * sign * float(sin_row[d]);
        }
    }
}

// One (logical block, kv head, sequence) per 32-lane threadgroup; each lane
// owns `qk` contiguous elements of every row. Grid (block-table width,
// num_kv_heads, num_seqs), rows past `seq_used_k` skipped. Bindings:
//   0 cache (the layer's K or V scratch, chunk table)  1 block_table
//   2 seq_used_k  3 cu_seqlens_q  4 slot_mapping  5 packed codes  6 norms
//   7 signs  8 centroids  9 cos_sin (K under ATTN_ROPE_ON_READ only)
template <typename T>
kernel void tq_stage_rotated(
    device const uint64_t* cache        [[buffer(0)]],
    device const uint*     block_table  [[buffer(1)]],
    device const uint*     seq_used_k   [[buffer(2)]],
    device const uint*     cu_seqlens_q [[buffer(3)]],
    device const uint*     slot_mapping [[buffer(4)]],
    device const uint*     packed       [[buffer(5)]],
    device const float*    norms        [[buffer(6)]],
    device const float*    signs        [[buffer(7)]],
    device const float*    centroids    [[buffer(8)]],
    device const T*        cos_sin      [[buffer(9)]],
    uint3 tg       [[threadgroup_position_in_grid]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    const uint head_dim = ATTN_HEAD_DIM;
    const uint num_kv = ATTN_NUM_KV_HEADS;
    const uint block_size = ATTN_BLOCK_SIZE;
    const uint qk_per_thread = head_dim / 32u;
    const uint logical_block = tg.x;
    const uint kv_head = tg.y;
    const uint seq = tg.z;
    const uint kv_len = seq_used_k[seq];
    if (logical_block * block_size >= kv_len) {
        return;
    }
    const uint q_start = cu_seqlens_q[seq];
    const uint prefix_len = kv_len - (cu_seqlens_q[seq + 1] - q_start);
    const uint bt_raw = block_table[seq * ATTN_MAX_BLOCKS_PER_SEQ + logical_block];
    const uint physical_block = bt_raw & 0x7FFFFFFFu;
    const bool span = (ATTN_ROR != 0u) && ((bt_raw & 0x80000000u) != 0u);
    const uint chunk = (ATTN_BLOCKS_PER_CHUNK == 0u) ? 0u : physical_block / ATTN_BLOCKS_PER_CHUNK;
    const uint blk_in_chunk =
        (ATTN_BLOCKS_PER_CHUNK == 0u) ? physical_block : physical_block % ATTN_BLOCKS_PER_CHUNK;
    device T* block = (device T*)cache[chunk]
        + (blk_in_chunk * num_kv + kv_head) * block_size * head_dim;
    const uint vpw = 32u / ATTN_TQ;
    const uint pdim = (head_dim + vpw - 1u) / vpw;
    const uint e0 = simd_lid * qk_per_thread;
    const float s = 1.0f / sqrt(float(head_dim));
    for (uint t = 0; t < block_size; ++t) {
        const uint i = logical_block * block_size + t;
        if (i >= kv_len) {
            break;
        }
        // The step's new keys come from the cache their writer just filled —
        // unless the slot is the spans write-skip sentinel (a reused block:
        // nothing was written, the key lives only in the packed store).
        const bool cached =
            i < prefix_len || slot_mapping[q_start + (i - prefix_len)] == 0xFFFFFFFFu;
        device T* row = block + t * head_dim + e0;
        float x[16];
        if (cached) {
            const uint store_row = (physical_block * block_size + t) * num_kv + kv_head;
            const float n = norms[store_row] * s;
            for (uint j = 0; j < qk_per_thread; ++j) {
                x[j] = centroids[tq_code(packed, store_row * pdim, e0 + j)] * n;
            }
            if (!span) {
                for (uint j = 0; j < qk_per_thread; ++j) {
                    row[j] = T(x[j]);
                }
                continue;
            }
            // x̃ = Rᵀ·x = s·D·H·x, to rope in the plain domain.
            tq_wht(x, qk_per_thread, simd_lid);
            for (uint j = 0; j < qk_per_thread; ++j) {
                x[j] *= s * signs[e0 + j];
            }
        } else {
            for (uint j = 0; j < qk_per_thread; ++j) {
                x[j] = float(row[j]);
            }
        }
        if (span) {
            tq_rope_slice<T>(x, qk_per_thread, simd_lid, i, cos_sin, 1.0f);
        }
        for (uint j = 0; j < qk_per_thread; ++j) {
            x[j] *= signs[e0 + j];
        }
        tq_wht(x, qk_per_thread, simd_lid);
        for (uint j = 0; j < qk_per_thread; ++j) {
            x[j] *= s;
        }
        if (span) {
            tq_rope_slice<T>(x, qk_per_thread, simd_lid, i, cos_sin, -1.0f);
        }
        for (uint j = 0; j < qk_per_thread; ++j) {
            row[j] = T(x[j]);
        }
    }
}

// Rotate each (token, q head) row of `rows` [tokens, num_q_heads, head_dim] in
// place: R·x, or Rᵀ·x when INVERSE. Grid (tokens, num_q_heads), 32 lanes.
template <typename T, bool INVERSE>
kernel void tq_rotate_rows(
    device T*           rows  [[buffer(0)]],
    device const float* signs [[buffer(1)]],
    uint2 tg       [[threadgroup_position_in_grid]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    const uint head_dim = ATTN_HEAD_DIM;
    const uint qk_per_thread = head_dim / 32u;
    const uint e0 = simd_lid * qk_per_thread;
    device T* row = rows + (tg.x * ATTN_NUM_Q_HEADS + tg.y) * head_dim + e0;
    const float s = 1.0f / sqrt(float(head_dim));
    float x[16];
    for (uint j = 0; j < qk_per_thread; ++j) {
        x[j] = float(row[j]) * (INVERSE ? 1.0f : signs[e0 + j]);
    }
    tq_wht(x, qk_per_thread, simd_lid);
    for (uint j = 0; j < qk_per_thread; ++j) {
        row[j] = T(x[j] * s * (INVERSE ? signs[e0 + j] : 1.0f));
    }
}

#define INSTANTIATE_TQ_PREFILL(tag, T)                                                   \
    template [[host_name("tq_stage_rotated_" #tag)]] [[kernel]] void tq_stage_rotated<T>( \
        device const uint64_t* cache [[buffer(0)]],                                     \
        device const uint* block_table [[buffer(1)]],                                   \
        device const uint* seq_used_k [[buffer(2)]],                                    \
        device const uint* cu_seqlens_q [[buffer(3)]],                                  \
        device const uint* slot_mapping [[buffer(4)]],                                  \
        device const uint* packed [[buffer(5)]],                                        \
        device const float* norms [[buffer(6)]],                                        \
        device const float* signs [[buffer(7)]],                                        \
        device const float* centroids [[buffer(8)]],                                    \
        device const T* cos_sin [[buffer(9)]],                                          \
        uint3 tg [[threadgroup_position_in_grid]],                                      \
        uint simd_lid [[thread_index_in_simdgroup]]);                                   \
    template [[host_name("tq_rotate_rows_" #tag)]] [[kernel]] void tq_rotate_rows<T, false>( \
        device T* rows [[buffer(0)]], device const float* signs [[buffer(1)]],          \
        uint2 tg [[threadgroup_position_in_grid]],                                      \
        uint simd_lid [[thread_index_in_simdgroup]]);                                   \
    template [[host_name("tq_unrotate_rows_" #tag)]] [[kernel]] void tq_rotate_rows<T, true>( \
        device T* rows [[buffer(0)]], device const float* signs [[buffer(1)]],          \
        uint2 tg [[threadgroup_position_in_grid]],                                      \
        uint simd_lid [[thread_index_in_simdgroup]]);

INSTANTIATE_TQ_PREFILL(f16, half)
INSTANTIATE_TQ_PREFILL(bf16, bfloat)

// ============================================================================
// attention_via_cache_v2_{f16,bf16}_specialized — paged-cache decode attention
// ============================================================================
//
// Paged-cache adaptation of MLX's `sdpa_vector` (from
// `mlx/backend/metal/kernels/sdpa_vector.h`):
//
//   1. Online softmax: max + sum_exp accumulated per-step in
//      registers; output accumulator is rescaled when a new max is
//      seen. No `shared_logits[]` threadgroup buffer, no two-pass
//      structure, no per-token `threadgroup_barrier` in the K loop.
//
//   2. BN simdgroups split the K-axis. Each simdgroup processes
//      keys at index `simd_gid, simd_gid + BN, simd_gid + 2*BN, ...`
//      so the work fans out across simdgroups without cross-simd
//      reductions in the inner loop.
//
//   3. Each lane handles `qk_per_thread = HEAD_DIM / 32` elements of
//      K and V. The dot product reduces inside one simdgroup via
//      `simd_sum`.
//
// The combine step at the end (after the K loop) collects per-
// simdgroup partials, reconciles their max+sum_exp via simd_max +
// simd_sum on threadgroup-mem-staged values, then produces the
// final output.
//
// Bindings (must match `AttentionViaCacheBindingSet` /
// `TqAttentionBindingSet` in tape/kernel_bindings.rs):
//   buffer(0)  = output      [batch, num_q_heads, head_dim]
//   buffer(1)  = q           [batch, num_q_heads, head_dim]
//   buffer(2)  = seq_used_k  [batch]
//   buffer(3)  = block_table [batch, MAX_BLOCKS_PER_SEQ]
//   buffer(4)  = k_cache     chunk table → [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
//   buffer(5)  = v_cache     chunk table → same
//   buffer(6)  = cos_sin     (ATTN_ROPE_ON_READ only)
//   buffer(7..13) (ATTN_TQ only) = packed K codes, packed V codes, K norms,
//                V norms ([slot, num_kv_heads, ...], by physical slot), signs
//                [head_dim], centroids [2^bits], slot_mapping [batch].
//
// Dispatch: threadgroups (batch, num_q_heads, 1), threads (1024, 1, 1)
// = 32 simdgroups × 32 lanes. HEAD_DIM must be a multiple of 32.
//
// max_total_threads_per_threadgroup(1024) = BN*BD (32*32) — REQUIRED. Without
// it the metal compiler picks a per-GPU register budget optimized for speed;
// at Gemma head_dim 256/512 the register pressure (q_reg[16]/o_reg[16]/k_loc[16]
// fp32) drops the pipeline's maxTotalThreadsPerThreadgroup below 1024 on M1
// (measured 640), so the 1024-thread launch SILENTLY under-launches — dropped
// simdgroups (20..31) never run, leaving the 32x32 softmax-combine transpose +
// tg_max/tg_sum reading uninitialized threadgroup slots → garbage decode. The
// attribute forces the compiler to guarantee the full 1024-thread launch is
// dispatchable (spilling registers on M1 if needed) or fail pipeline creation
// loudly instead of corrupting silently. Must equal the dispatched
// threads_per_threadgroup at lowering.rs (AttentionViaCache / Sliding).
template <typename T>
[[kernel, max_total_threads_per_threadgroup(1024)]] void attention_via_cache_v2(
    device       T* output         [[buffer(0)]],
    device const T* q              [[buffer(1)]],
    device const uint* seq_used_k  [[buffer(2)]],
    device const uint* block_table [[buffer(3)]],
    device const uint64_t* k_cache [[buffer(4)]],
    device const uint64_t* v_cache [[buffer(5)]],
    // Rope-on-read (spans): cos/sin for the active layer class. The
    // per-block "stored unrotated → rotate on read" flag rides in
    // block_table bit 31 (free — the K-loop already loads block_table
    // for addressing), so there is NO separate flag buffer. Only
    // accessed when ATTN_ROPE_ON_READ; dead-eliminated otherwise.
    device const T*     cos_sin      [[buffer(6)]],
    device const uint*  tq_packed_k  [[buffer(7)]],
    device const uint*  tq_packed_v  [[buffer(8)]],
    device const float* tq_norms_k   [[buffer(9)]],
    device const float* tq_norms_v   [[buffer(10)]],
    device const float* tq_signs     [[buffer(11)]],
    device const float* tq_centroids [[buffer(12)]],
    device const uint*  slot_mapping [[buffer(13)]],
    uint3  tg_pos    [[threadgroup_position_in_grid]],
    uint   simd_gid  [[simdgroup_index_in_threadgroup]],
    uint   simd_lid  [[thread_index_in_simdgroup]])
{
    constexpr int BN = 32; // simdgroups per threadgroup
    constexpr int BD = 32; // lanes per simdgroup
    typedef float U;
    // Spans: block_table entries carry the unrotated flag in bit 31 when
    // ATTN_ROR; mask it off for the physical block id. Compile-const
    // false (non-spans) → no mask, byte-identical.
    const uint ATTN_BT_MASK = (ATTN_ROR != 0u) ? 0x7FFFFFFFu : 0xFFFFFFFFu;

    const uint head_dim    = ATTN_HEAD_DIM;
    const uint num_q       = ATTN_NUM_Q_HEADS;
    const uint num_kv      = ATTN_NUM_KV_HEADS;
    const uint block_size  = ATTN_BLOCK_SIZE;
    const uint max_blocks  = ATTN_MAX_BLOCKS_PER_SEQ;
    const float scale      = ATTN_SCALE_FC;

    const uint qk_per_thread = head_dim / uint(BD);

    const uint seq_idx     = tg_pos.x;            // batch index
    const uint q_head_idx  = tg_pos.y;            // 0..NUM_Q_HEADS
    const uint group_ratio = num_q / num_kv;
    const uint kv_head_idx = q_head_idx / group_ratio;
    const uint kv_len      = seq_used_k[seq_idx];

    const uint kv_blk_stride  = num_kv * block_size * head_dim;
    const uint kv_head_stride = block_size * head_dim;
    const uint kv_tok_stride  = head_dim;

    thread U q_reg[16];                 // qk_per_thread <= 16 (head_dim<=512;
    thread U o_reg[16];                 // Gemma4 global layers are 512)

    // Threadgroup scratch for per-simdgroup max + sum_exp combine.
    threadgroup U tg_outputs[BN * BD];
    threadgroup U tg_max[BN];
    threadgroup U tg_sum[BN];

    device const T*    q_row = q + (seq_idx * num_q + q_head_idx) * head_dim;
    device       T*    o_row = output + (seq_idx * num_q + q_head_idx) * head_dim;
    device const uint* row_block_table = block_table + seq_idx * max_blocks;

    // Pre-multiply Q by scale (MLX `sdpa_vector`: `q[i] = scale * queries[i]`).
    // Element ownership follows attn_elem_off (contiguous, or co-resident
    // NeoX pairs under ATTN_PAIR_CORESIDENT) — Q must match K's per-lane set.
    for (uint i = 0; i < qk_per_thread; ++i) {
        q_reg[i] = U(scale) * U(q_row[attn_elem_off(simd_lid, i, qk_per_thread, head_dim)]);
        o_reg[i] = 0;
    }

    // TurboQuant: q rotated into the codebook domain, `s²·H·D·q`. The
    // codebook domain is laid out contiguously — lane `l` owns elements
    // `l*qk_per_thread + j` (`tq_e`) whatever the q/K layout, as H·D mixes
    // every element anyway — so a lane's codes sit in adjacent packed words.
    // Every simdgroup owns the same slices and rotates its own copy in
    // registers. The codebook (<= 16 centroids) is staged once.
    const uint tq_e = simd_lid * qk_per_thread;
    thread U qt_reg[16];
    threadgroup U tq_lut[16];
    if (ATTN_TQ != 0u) {
        for (uint j = 0; j < qk_per_thread; ++j) {
            qt_reg[j] = U(scale) * U(q_row[tq_e + j]) * tq_signs[tq_e + j];
        }
        tq_wht(qt_reg, qk_per_thread, simd_lid);
        for (uint j = 0; j < qk_per_thread; ++j) {
            qt_reg[j] /= U(head_dim);
        }
        const uint tid = simd_gid * uint(BD) + simd_lid;
        if (tid < (1u << ATTN_TQ)) {
            tq_lut[tid] = tq_centroids[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint tq_pdim = (ATTN_TQ == 0u) ? 0u : (head_dim + 32u / ATTN_TQ - 1u) / (32u / ATTN_TQ);

    // Initialize per-thread max with finite minimum (MLX uses
    // `Limits<U>::finite_min`; -FLT_MAX is the f32 equivalent).
    // fast::exp doesn't handle -INFINITY safely so we avoid it.
    U max_score = -FLT_MAX;
    U sum_exp_score = 0;

    // TurboQuant: the key this step appended (the query's own, at kv_len-1)
    // is not in the packed store yet — uniform arches quantize after
    // attention, hybrid ones only lossily before it — so it is read from the
    // cache its writer just filled, exactly as the dequant path read it. A
    // reused span block's slot is the write-skip sentinel: nothing was
    // written, and that key lives only in the packed store.
    const bool tail_in_cache = (ATTN_TQ == 0u) || (slot_mapping[seq_idx] != 0xFFFFFFFFu);
    // For each key, simdgroup `simd_gid` handles tokens at indices
    // simd_gid, simd_gid+BN, simd_gid+2*BN, ... The simdgroup that
    // overshoots `kv_len` skips its iteration and contributes 0.
    for (uint i = simd_gid; i < kv_len; i += uint(BN)) {
        // Sliding window: decode Q sits at absolute position kv_len-1;
        // skip keys older than the window. Branch is simdgroup-uniform
        // (i derives from simd_gid) and folds away when ATTN_WINDOW=0.
        if (ATTN_WINDOW > 0 && (int(kv_len) - 1 - int(i)) >= ATTN_WINDOW) {
            continue;
        }
        // Resolve paged cache pointer for token i in this simdgroup.
        const uint logical_block = i / block_size;
        const uint bt_raw = row_block_table[logical_block];
        const uint physical_block = bt_raw & ATTN_BT_MASK;
        // Spans: bit 31 = this block's K is stored unrotated → rotate on
        // read. Free — bt_raw is loaded for addressing anyway. Uniform
        // across the simdgroup (same block per simdgroup-iteration).
        const bool do_rot = (ATTN_ROR != 0u) && ((bt_raw & 0x80000000u) != 0u);
        const uint token_in_block = i - logical_block * block_size;
        // TurboQuant: every key but the tail comes from the packed store,
        // indexed by physical slot (bit 31 stripped — spans or not).
        const bool packed = (ATTN_TQ != 0u) && !(tail_in_cache && i + 1u == kv_len);
        const uint tq_row =
            ((bt_raw & 0x7FFFFFFFu) * block_size + token_in_block) * num_kv + kv_head_idx;

        U score = 0;
        U k_scale = 1;
        device const T* v_ptr = nullptr;
        if (packed) {
            const uint k_word = tq_row * tq_pdim;
            if (do_rot) {
                // Span block: the codes hold UNROTATED K and RoPE does not
                // commute with H·D, so decode this key the way the dequant
                // pass did (rounded to T) and re-rope it in the plain domain.
                U k_loc[16];
                for (uint j = 0; j < qk_per_thread; ++j) {
                    k_loc[j] = tq_lut[tq_code(tq_packed_k, k_word,
                                              attn_elem_off(simd_lid, j, qk_per_thread, head_dim))];
                }
                tq_wht(k_loc, qk_per_thread, simd_lid);
                const U k_norm = tq_norms_k[tq_row] / U(head_dim);
                for (uint j = 0; j < qk_per_thread; ++j) {
                    const uint e = attn_elem_off(simd_lid, j, qk_per_thread, head_dim);
                    k_loc[j] = U(T(k_loc[j] * tq_signs[e] * k_norm));
                }
                attn_rope_on_read<T>(k_loc, qk_per_thread, simd_lid, i, cos_sin);
                for (uint j = 0; j < qk_per_thread; ++j) {
                    score += q_reg[j] * k_loc[j];
                }
            } else {
                for (uint j = 0; j < qk_per_thread; ++j) {
                    score += qt_reg[j] * tq_lut[tq_code(tq_packed_k, k_word, tq_e + j)];
                }
                k_scale = tq_norms_k[tq_row];
            }
        } else {
            // Chunked KV: deref the chunk backing this physical block.
            // ATTN_BLOCKS_PER_CHUNK is a function constant. When set to 0 the
            // compiler dead-eliminates the chunked branch — used by the
            // single-buffer-per-layer mode where `k_cache[0]` holds the layer
            // base address and physical_block is the full offset (no modulo,
            // no per-block chunk_table load). When non-zero the path matches
            // the reactive chunked KV pool.
            uint chunk;
            uint blk_in_chunk;
            if (ATTN_BLOCKS_PER_CHUNK == 0u) {
                chunk = 0u;
                blk_in_chunk = physical_block;
            } else {
                chunk = physical_block / ATTN_BLOCKS_PER_CHUNK;
                blk_in_chunk = physical_block % ATTN_BLOCKS_PER_CHUNK;
            }
            // Row base (no per-lane offset); element ownership via
            // attn_elem_off — contiguous, or co-resident NeoX pairs.
            device const T* k_ptr =
                (device const T*)k_cache[chunk]
                + blk_in_chunk   * kv_blk_stride
                + kv_head_idx    * kv_head_stride
                + token_in_block * kv_tok_stride;
            v_ptr =
                (device const T*)v_cache[chunk]
                + blk_in_chunk   * kv_blk_stride
                + kv_head_idx    * kv_head_stride
                + token_in_block * kv_tok_stride;

            // Dot product q·k for this lane's slice. Span blocks (do_rot,
            // simdgroup-uniform) are re-roped to this key's position (= `i`)
            // in registers first; every other key dots directly from k_ptr —
            // byte-identical to the rope-on-write hot path (the rope-on-read
            // parity guarantee).
            if (do_rot) {
                U k_loc[16];
                for (uint j = 0; j < qk_per_thread; ++j) {
                    k_loc[j] = U(k_ptr[attn_elem_off(simd_lid, j, qk_per_thread, head_dim)]);
                }
                attn_rope_on_read<T>(k_loc, qk_per_thread, simd_lid, i, cos_sin);
                for (uint j = 0; j < qk_per_thread; ++j) {
                    score += q_reg[j] * k_loc[j];
                }
            } else {
                for (uint j = 0; j < qk_per_thread; ++j) {
                    score += q_reg[j] * U(k_ptr[attn_elem_off(simd_lid, j, qk_per_thread, head_dim)]);
                }
            }
        }
        score = simd_sum(score) * k_scale;

        // Online softmax update. Match MLX `sdpa_vector`: fast::exp
        // for both factor + exp_score.
        U new_max = max(max_score, score);
        U factor = metal::fast::exp(max_score - new_max);
        U exp_score = metal::fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        // Accumulate weighted V; rescale prior accumulator with factor.
        // V element ownership matches K/Q (attn_elem_off). Under TurboQuant
        // the accumulator lives in the codebook domain.
        if (packed) {
            const uint v_word = tq_row * tq_pdim;
            const U p_norm = exp_score * tq_norms_v[tq_row];
            for (uint j = 0; j < qk_per_thread; ++j) {
                o_reg[j] = o_reg[j] * factor + p_norm * tq_lut[tq_code(tq_packed_v, v_word, tq_e + j)];
            }
        } else if (ATTN_TQ != 0u) {
            // The tail's plain V into the codebook domain: s²·D·H·(H·D·v) = v.
            U v_loc[16];
            for (uint j = 0; j < qk_per_thread; ++j) {
                v_loc[j] = U(v_ptr[tq_e + j]) * tq_signs[tq_e + j];
            }
            tq_wht(v_loc, qk_per_thread, simd_lid);
            for (uint j = 0; j < qk_per_thread; ++j) {
                o_reg[j] = o_reg[j] * factor + exp_score * v_loc[j];
            }
        } else {
            for (uint j = 0; j < qk_per_thread; ++j) {
                o_reg[j] = o_reg[j] * factor
                         + exp_score * U(v_ptr[attn_elem_off(simd_lid, j, qk_per_thread, head_dim)]);
            }
        }
    }

    // ── Combine per-simdgroup partials ───────────────────────────
    //
    // Each simdgroup's lane 0 publishes its max + sum_exp; all
    // simdgroups then read all values via lane id and reduce.
    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each lane (within simdgroup_id 0..BN-1) reads tg_max[simd_lid]
    // / tg_sum[simd_lid]; simd_max + simd_sum produce the global max
    // and (factor-rescaled) global sum_exp.
    U other_max = tg_max[simd_lid];
    U global_max = simd_max(other_max);
    U factor = metal::fast::exp(other_max - global_max);
    U global_sum = simd_sum(tg_sum[simd_lid] * factor);

    // Combine output partials. Each simdgroup wrote o_reg[j] for
    // its slice; we need to weight each simdgroup's contribution by
    // its `factor` (the rescaling for the global max), then sum
    // across simdgroups, then divide by global_sum.
    for (uint j = 0; j < qk_per_thread; ++j) {
        tg_outputs[simd_lid * BD + simd_gid] = o_reg[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Each simdgroup reads its column from tg_outputs and sums
        // across the BD partials, weighted by per-simdgroup factor.
        U val = tg_outputs[simd_gid * BD + simd_lid] * factor;
        U combined = simd_sum(val);
        if (global_sum != 0) {
            combined = combined / global_sum;
        }
        o_reg[j] = combined;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // The combine transposes lane<->simdgroup, so simdgroup `simd_gid`
    // now owns the element set that lane `simd_gid` owned during the K
    // loop.
    if (ATTN_TQ != 0u) {
        // Codebook-domain output: gather it back into lane slices and
        // rotate once, o = s²·D·H·a.
        if (simd_lid == 0) {
            for (uint j = 0; j < qk_per_thread; ++j) {
                tg_outputs[simd_gid * qk_per_thread + j] = o_reg[j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_gid == 0) {
            for (uint j = 0; j < qk_per_thread; ++j) {
                o_reg[j] = tg_outputs[tq_e + j];
            }
            tq_wht(o_reg, qk_per_thread, simd_lid);
            for (uint j = 0; j < qk_per_thread; ++j) {
                o_row[tq_e + j] = T(o_reg[j] * tq_signs[tq_e + j] / U(head_dim));
            }
        }
    } else if (simd_lid == 0) {
        // Lane 0 of each simdgroup writes its qk_per_thread output slice.
        for (uint j = 0; j < qk_per_thread; ++j) {
            o_row[attn_elem_off(simd_gid, j, qk_per_thread, head_dim)] = T(o_reg[j]);
        }
    }
}

#define INSTANTIATE_ATTENTION_VIA_CACHE_V2(name, T)                                   \
    template [[host_name(name)]] [[kernel]]                                            \
    void attention_via_cache_v2<T>(                                                    \
        device T* output [[buffer(0)]], device const T* q [[buffer(1)]],               \
        device const uint* seq_used_k [[buffer(2)]],                                   \
        device const uint* block_table [[buffer(3)]],                                  \
        device const uint64_t* k_cache [[buffer(4)]],                                  \
        device const uint64_t* v_cache [[buffer(5)]],                                  \
        device const T* cos_sin [[buffer(6)]],                                         \
        device const uint* tq_packed_k [[buffer(7)]],                                  \
        device const uint* tq_packed_v [[buffer(8)]],                                  \
        device const float* tq_norms_k [[buffer(9)]],                                  \
        device const float* tq_norms_v [[buffer(10)]],                                 \
        device const float* tq_signs [[buffer(11)]],                                   \
        device const float* tq_centroids [[buffer(12)]],                               \
        device const uint* slot_mapping [[buffer(13)]],                                \
        uint3 tg_pos [[threadgroup_position_in_grid]],                                 \
        uint simd_gid [[simdgroup_index_in_threadgroup]],                              \
        uint simd_lid [[thread_index_in_simdgroup]]);

INSTANTIATE_ATTENTION_VIA_CACHE_V2("attention_via_cache_v2_f16_specialized", half)
INSTANTIATE_ATTENTION_VIA_CACHE_V2("attention_via_cache_v2_bf16_specialized", bfloat)

// ─────────────────────────────────────────────────────────────────────
// attention_prefill_sdpa_v2_paged — paged-cache variant of the prefill
// sdpa_vector port. Same outer structure as
// `attention_prefill_sdpa_v2_*` (1 Q per TG, dispatch on
// `(num_q_heads, total_q, 1)`, online softmax + per-simdgroup K-axis
// split) but K/V are read from the paged cache via `block_table`
// (mirroring the decode kernel `attention_via_cache_v2_*`).
//
// Use cases (where contiguous prefill cannot be used):
//   - Chunked prefill: prompts > max_num_batched_tokens get split
//     across steps; chunks 2+ have prior K already in the paged cache.
//   - Prefix caching: requests sharing a prefix start with
//     num_computed_tokens > 0; the new tokens must attend over the
//     prior cached K.
//   - Mixed prefill/decode batches: V1 scheduler interleaves a
//     prefilling sequence (with prior K) alongside decoding sequences.
//   - Multi-turn chat continuation: each new turn extends a sequence
//     whose prior turns are already cached.
//
// Differences vs the contiguous prefill kernel:
//   1. K/V read from `k_cache`/`v_cache` via `block_table[seq_idx]`
//      indirection (paged-cache layout) — same access shape as the
//      decode kernel.
//   2. K-axis loop runs over the FULL cached length `seqused_k[seq]`,
//      not just `seq_end - seq_start` — the prior cached K is in
//      cache slots `[0, seqused_k[seq] - new_q_for_seq)` and the new
//      tokens just appended (by the upstream `RopeAppend`) are at
//      `[seqused_k[seq] - new_q_for_seq, seqused_k[seq])`.
//   3. Causal mask compares K position `i` against the absolute Q
//      position `q_abs_pos = (seqused_k[seq] - new_q_for_seq) +
//      q_pos_in_new`, where `new_q_for_seq = cu_seqlens_q[seq+1] -
//      cu_seqlens_q[seq]` and `q_pos_in_new = global_q -
//      cu_seqlens_q[seq]`. The `(seqused_k - new_q_for_seq)` shift
//      is the prefix-length offset that contiguous prefill doesn't
//      need (it has no prior cached K).
//   4. Function constants extend to BLOCK_SIZE + MAX_BLOCKS_PER_SEQ
//      (paging) — same set as the decode kernel.
//
// Buffer bindings:
//   buffer(0) = output       [total_q, num_q_heads, head_dim]
//   buffer(1) = q            [total_q, num_q_heads, head_dim]
//   buffer(2) = cu_seqlens_q [batch+1]
//   buffer(3) = seqused_k    [batch]   (total cached K including new)
//   buffer(4) = block_table  [batch, MAX_BLOCKS_PER_SEQ]
//   buffer(5) = k_cache      [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
//   buffer(6) = v_cache      [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
//
// Function constants 0..5: HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS,
// ATTN_SCALE_FC, BLOCK_SIZE, MAX_BLOCKS_PER_SEQ. Same indices as
// `attention_via_cache_v2_*`.
//
// Dispatch: threadgroups `(num_q_heads, total_q, 1)`, threads
// `(1024, 1, 1)` = 32 simdgroups × 32 lanes (same as the contiguous
// prefill kernel and decode kernel).
//
// Constraint: HEAD_DIM must be a multiple of 32 (qk_per_thread =
// HEAD_DIM / 32). Llama-3.2 / Qwen / Mistral / Phi all satisfy.

// max_total_threads_per_threadgroup(1024) = BN*BD (32*32) — REQUIRED, same as
// attention_via_cache_v2 (see there): the 1024-thread launch under-launches on
// M1 at Gemma head_dim 256/512 (pipeline cap 768) without it → silent garbage.
[[kernel, max_total_threads_per_threadgroup(1024)]] void attention_prefill_sdpa_v2_paged_f16_specialized(
    device       half* output       [[buffer(0)]],   // [total_q, num_q_heads, head_dim]
    device const half* q            [[buffer(1)]],   // [total_q, num_q_heads, head_dim]
    device const uint* cu_seqlens_q [[buffer(2)]],   // [batch+1]
    device const uint* seq_used_k   [[buffer(3)]],   // [batch]
    device const uint* block_table  [[buffer(4)]],   // [batch, MAX_BLOCKS_PER_SEQ]
    device const uint64_t* k_cache  [[buffer(5)]],   // chunk-address table
    device const uint64_t* v_cache  [[buffer(6)]],   // chunk-address table
    // Rope-on-read (spans): see attention_via_cache_v2. f16 cos_sin; the
    // unrotated flag rides in block_table bit 31 (no flag buffer).
    device const half*     cos_sin               [[buffer(7)]],
    // Block-diagonal span attention: per-logical-block span label. Bound only
    // when ATTN_ROR; read only under ATTN_ROR (const-folded away otherwise).
    device const uint*     span_ids              [[buffer(8)]],
    uint3  tg_pos    [[threadgroup_position_in_grid]],
    uint3  tid       [[thread_position_in_threadgroup]],
    uint   simd_gid  [[simdgroup_index_in_threadgroup]],
    uint   simd_lid  [[thread_index_in_simdgroup]])
{
    constexpr int BN = 32;
    constexpr int BD = 32;
    typedef float U;
    const uint ATTN_BT_MASK = (ATTN_ROR != 0u) ? 0x7FFFFFFFu : 0xFFFFFFFFu;

    const uint head_dim    = ATTN_HEAD_DIM;
    const uint num_q       = ATTN_NUM_Q_HEADS;
    const uint num_kv      = ATTN_NUM_KV_HEADS;
    const uint block_size  = ATTN_BLOCK_SIZE;
    const uint max_blocks  = ATTN_MAX_BLOCKS_PER_SEQ;
    const float scale      = ATTN_SCALE_FC;
    const uint qk_per_thread = head_dim / uint(BD);

    const uint q_head_idx  = tg_pos.x;            // 0..NUM_Q_HEADS
    const uint global_q    = tg_pos.y;            // 0..total_q
    const uint group_ratio = num_q / num_kv;
    const uint kv_head_idx = q_head_idx / group_ratio;

    // Locate this Q token's sequence via cu_seqlens_q. Same linear
    // scan as the contiguous prefill kernel; sentinel exit on the
    // first `hi <= lo` (production buffer is `(max_m+1)*4` bytes,
    // OOB reads land in zero-init memory).
    uint seq_idx   = 0;
    uint seq_start = 0;
    uint seq_end   = 0;
    bool in_range  = false;
    for (uint b = 0; b < 1024u; ++b) {
        const uint lo = cu_seqlens_q[b];
        const uint hi = cu_seqlens_q[b + 1];
        if (global_q >= lo && global_q < hi) {
            seq_idx   = b;
            seq_start = lo;
            seq_end   = hi;
            in_range  = true;
            break;
        }
        if (hi <= lo) break;       // sentinel: end of batch
    }
    if (!in_range) {
        // Padding lane (global_q past the last sequence). Match
        // contiguous-prefill: lane 0 of each simdgroup writes zero.
        if (simd_lid == 0) {
            device half* o_ptr =
                output + (global_q * num_q + q_head_idx) * head_dim
                       + simd_gid * qk_per_thread;
            for (uint j = 0; j < qk_per_thread; ++j) o_ptr[j] = half(0);
        }
        return;
    }

    const uint new_q_for_seq = seq_end - seq_start;
    const uint q_pos_in_new  = global_q - seq_start;
    const uint kv_len        = seq_used_k[seq_idx];
    // Absolute position of this Q in the K axis. Prefix length is
    // `kv_len - new_q_for_seq` (caller guarantees seq_used_k already
    // includes the just-appended new tokens; upstream RopeAppend ran
    // before this attention dispatch).
    const uint q_abs_pos     = (kv_len - new_q_for_seq) + q_pos_in_new;

    const uint kv_blk_stride  = num_kv * block_size * head_dim;
    const uint kv_head_stride = block_size * head_dim;
    const uint kv_tok_stride  = head_dim;

    thread U q_reg[16];                 // qk_per_thread <= 16 (head_dim<=512;
    thread U o_reg[16];                 // Gemma4 global layers are 512)

    threadgroup U tg_outputs[BN * BD];
    threadgroup U tg_max[BN];
    threadgroup U tg_sum[BN];

    device const half* q_row = q + (global_q * num_q + q_head_idx) * head_dim;
    device       half* o_row = output + (global_q * num_q + q_head_idx) * head_dim;
    device const uint* row_block_table = block_table + seq_idx * max_blocks;

    // Pre-multiply Q by scale (MLX `sdpa_vector`: `q[i] = scale * queries[i]`).
    for (uint i = 0; i < qk_per_thread; ++i) {
        q_reg[i] = U(scale) * U(q_row[simd_lid * qk_per_thread + i]);
        o_reg[i] = 0;
    }

    U max_score = -FLT_MAX;
    U sum_exp_score = 0;

    // Block-diagonal span attention. span_ids holds (the span's first block + 1),
    // or 0 for the shared prefix / query (attends everything). A query inside a
    // span attends ONLY [span_lo, q_abs_pos] — restricting the loop's LOWER bound
    // (here) and UPPER bound (q_abs_pos) is the real O(N^2)->O(N*span) cut, not a
    // per-key skip. ATTN_ROR-gated so the non-spans pipeline const-folds it away.
    uint q_span = 0u;
    uint span_lo = 0u;
    if (ATTN_ROR != 0u) {
        q_span = span_ids[q_abs_pos]; // per-TOKEN index (no block divide)
        if (q_span != 0u) { span_lo = q_span - 1u; } // label-1 IS the span's first TOKEN (exact)
    }

    // Online softmax over the FULL cached K. Each simdgroup `simd_gid`
    // covers tokens at indices simd_gid, simd_gid + BN, simd_gid +
    // 2*BN, … Causal: skip K positions > q_abs_pos. The branch is
    // simdgroup-uniform (`i` derives from simd_gid; `q_abs_pos` is
    // threadgroup-uniform).
    // Loop is bounded [span_lo, q_abs_pos]: span_lo (0 for non-spans) skips the
    // prefix + sibling spans; `i <= q_abs_pos` is the causal upper bound (so we
    // no longer iterate-then-`continue` past it). For non-spans span_lo==0 and
    // the visited keys/order are identical to the old `i<kv_len; if(i>q_abs_pos)`
    // form → byte-identical. For a span query the loop runs O(span), not O(N).
    for (uint i = span_lo + simd_gid; i < kv_len; i += uint(BN)) {
        // Causal upper bound. SPAN query (q_span != 0): BREAK — the loop is bounded
        // [span_lo, q_abs_pos] = the span only (the real O(N)->O(span) cut). NON-spans
        // (q_span == 0): CONTINUE, byte-identical to the original full-causal scan.
        if (i > q_abs_pos) { if (q_span != 0u) { break; } else { continue; } }
        // Sliding window: attend iff q_abs_pos - i < window.
        if (ATTN_WINDOW > 0 && int(q_abs_pos) - int(i) >= ATTN_WINDOW) {
            continue;
        }

        const uint logical_block = i / block_size;
        const uint bt_raw = row_block_table[logical_block];
        const uint physical_block = bt_raw & ATTN_BT_MASK;
        const bool do_rot = (ATTN_ROR != 0u) && ((bt_raw & 0x80000000u) != 0u);
        const uint token_in_block = i - logical_block * block_size;
        // Chunked KV: deref the chunk backing this physical block.
        // ATTN_BLOCKS_PER_CHUNK is a function constant. When set to 0 the
        // compiler dead-eliminates the chunked branch — used by the
        // single-buffer-per-layer mode where `k_cache[0]` holds the layer
        // base address and physical_block is the full offset (no modulo,
        // no per-block chunk_table load). When non-zero the path matches
        // the reactive chunked KV pool.
        uint chunk;
        uint blk_in_chunk;
        if (ATTN_BLOCKS_PER_CHUNK == 0u) {
            chunk = 0u;
            blk_in_chunk = physical_block;
        } else {
            chunk = physical_block / ATTN_BLOCKS_PER_CHUNK;
            blk_in_chunk = physical_block % ATTN_BLOCKS_PER_CHUNK;
        }
        device const half* k_ptr =
            (device const half*)k_cache[chunk]
            + blk_in_chunk   * kv_blk_stride
            + kv_head_idx    * kv_head_stride
            + token_in_block * kv_tok_stride
            + simd_lid * qk_per_thread;
        device const half* v_ptr =
            (device const half*)v_cache[chunk]
            + blk_in_chunk   * kv_blk_stride
            + kv_head_idx    * kv_head_stride
            + token_in_block * kv_tok_stride
            + simd_lid * qk_per_thread;

        // q·k; span blocks (do_rot) re-roped to position `i` first, every
        // other key dots directly from k_ptr. See attention_via_cache_v2.
        U score = 0;
        if (do_rot) {
            U k_loc[16];
            for (uint j = 0; j < qk_per_thread; ++j) {
                k_loc[j] = U(k_ptr[j]);
            }
            const uint half_dim = ATTN_ROT_DIM / 2u;
            device const half* cos_row = cos_sin + i * ATTN_ROT_DIM;
            device const half* sin_row = cos_row + half_dim;
            rope_on_read_k_slice<half>(k_loc, qk_per_thread, simd_lid,
                                 cos_row, sin_row, half_dim, ATTN_PAIR_OFF);
            for (uint j = 0; j < qk_per_thread; ++j) {
                score += q_reg[j] * k_loc[j];
            }
        } else {
            for (uint j = 0; j < qk_per_thread; ++j) {
                score += q_reg[j] * U(k_ptr[j]);
            }
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = metal::fast::exp(max_score - new_max);
        U exp_score = metal::fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint j = 0; j < qk_per_thread; ++j) {
            o_reg[j] = o_reg[j] * factor + exp_score * U(v_ptr[j]);
        }
    }

    // Combine per-simdgroup partials (identical to decode + contiguous
    // prefill kernels).
    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    U other_max = tg_max[simd_lid];
    U global_max = simd_max(other_max);
    U factor = metal::fast::exp(other_max - global_max);
    U global_sum = simd_sum(tg_sum[simd_lid] * factor);

    for (uint j = 0; j < qk_per_thread; ++j) {
        tg_outputs[simd_lid * BD + simd_gid] = o_reg[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        U val = tg_outputs[simd_gid * BD + simd_lid] * factor;
        U combined = simd_sum(val);
        if (global_sum != 0) {
            combined = combined / global_sum;
        }
        o_reg[j] = combined;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        device half* o_ptr = o_row + simd_gid * qk_per_thread;
        for (uint j = 0; j < qk_per_thread; ++j) {
            o_ptr[j] = half(o_reg[j]);
        }
    }
}

/// BF16 sibling of `attention_prefill_sdpa_v2_paged_f16_specialized`.
/// Same algorithm; sole difference is the `bfloat`/`half` element
/// type on the device pointers. f32 accumulator preserved.
///
/// max_total_threads_per_threadgroup(1024): see the f16 sibling — REQUIRED so
/// the 1024-thread (32-simdgroup) launch is guaranteed dispatchable on M1.
[[kernel, max_total_threads_per_threadgroup(1024)]] void attention_prefill_sdpa_v2_paged_bf16_specialized(
    device       bfloat* output       [[buffer(0)]],   // [total_q, num_q_heads, head_dim]
    device const bfloat* q            [[buffer(1)]],   // [total_q, num_q_heads, head_dim]
    device const uint*   cu_seqlens_q [[buffer(2)]],   // [batch+1]
    device const uint*   seq_used_k   [[buffer(3)]],   // [batch]
    device const uint*   block_table  [[buffer(4)]],   // [batch, MAX_BLOCKS_PER_SEQ]
    device const uint64_t* k_cache    [[buffer(5)]],   // chunk-address table
    device const uint64_t* v_cache    [[buffer(6)]],   // chunk-address table
    // Rope-on-read (spans): see attention_via_cache_v2. bf16 cos_sin; the
    // unrotated flag rides in block_table bit 31 (no flag buffer).
    device const bfloat*   cos_sin               [[buffer(7)]],
    // Block-diagonal span attention: per-logical-block span label (slot 8).
    device const uint*     span_ids              [[buffer(8)]],
    uint3  tg_pos    [[threadgroup_position_in_grid]],
    uint3  tid       [[thread_position_in_threadgroup]],
    uint   simd_gid  [[simdgroup_index_in_threadgroup]],
    uint   simd_lid  [[thread_index_in_simdgroup]])
{
    constexpr int BN = 32;
    constexpr int BD = 32;
    typedef float U;
    const uint ATTN_BT_MASK = (ATTN_ROR != 0u) ? 0x7FFFFFFFu : 0xFFFFFFFFu;

    const uint head_dim    = ATTN_HEAD_DIM;
    const uint num_q       = ATTN_NUM_Q_HEADS;
    const uint num_kv      = ATTN_NUM_KV_HEADS;
    const uint block_size  = ATTN_BLOCK_SIZE;
    const uint max_blocks  = ATTN_MAX_BLOCKS_PER_SEQ;
    const float scale      = ATTN_SCALE_FC;
    const uint qk_per_thread = head_dim / uint(BD);

    const uint q_head_idx  = tg_pos.x;
    const uint global_q    = tg_pos.y;
    const uint group_ratio = num_q / num_kv;
    const uint kv_head_idx = q_head_idx / group_ratio;

    uint seq_idx   = 0;
    uint seq_start = 0;
    uint seq_end   = 0;
    bool in_range  = false;
    for (uint b = 0; b < 1024u; ++b) {
        const uint lo = cu_seqlens_q[b];
        const uint hi = cu_seqlens_q[b + 1];
        if (global_q >= lo && global_q < hi) {
            seq_idx   = b;
            seq_start = lo;
            seq_end   = hi;
            in_range  = true;
            break;
        }
        if (hi <= lo) break;
    }
    if (!in_range) {
        if (simd_lid == 0) {
            device bfloat* o_ptr =
                output + (global_q * num_q + q_head_idx) * head_dim
                       + simd_gid * qk_per_thread;
            for (uint j = 0; j < qk_per_thread; ++j) o_ptr[j] = bfloat(0);
        }
        return;
    }

    const uint new_q_for_seq = seq_end - seq_start;
    const uint q_pos_in_new  = global_q - seq_start;
    const uint kv_len        = seq_used_k[seq_idx];
    const uint q_abs_pos     = (kv_len - new_q_for_seq) + q_pos_in_new;

    const uint kv_blk_stride  = num_kv * block_size * head_dim;
    const uint kv_head_stride = block_size * head_dim;
    const uint kv_tok_stride  = head_dim;

    thread U q_reg[16];                 // qk_per_thread <= 16 (head_dim<=512)
    thread U o_reg[16];

    threadgroup U tg_outputs[BN * BD];
    threadgroup U tg_max[BN];
    threadgroup U tg_sum[BN];

    device const bfloat* q_row = q + (global_q * num_q + q_head_idx) * head_dim;
    device       bfloat* o_row = output + (global_q * num_q + q_head_idx) * head_dim;
    device const uint*   row_block_table = block_table + seq_idx * max_blocks;

    for (uint i = 0; i < qk_per_thread; ++i) {
        q_reg[i] = U(scale) * U(q_row[simd_lid * qk_per_thread + i]);
        o_reg[i] = 0;
    }

    U max_score = -FLT_MAX;
    U sum_exp_score = 0;

    // Block-diagonal span attention. span_ids holds (the span's first block + 1),
    // or 0 for the shared prefix / query (attends everything). A query inside a
    // span attends ONLY [span_lo, q_abs_pos] — restricting the loop's LOWER bound
    // (here) and UPPER bound (q_abs_pos) is the real O(N^2)->O(N*span) cut, not a
    // per-key skip. ATTN_ROR-gated so the non-spans pipeline const-folds it away.
    uint q_span = 0u;
    uint span_lo = 0u;
    if (ATTN_ROR != 0u) {
        q_span = span_ids[q_abs_pos]; // per-TOKEN index (no block divide)
        if (q_span != 0u) { span_lo = q_span - 1u; } // label-1 IS the span's first TOKEN (exact)
    }

    // Loop is bounded [span_lo, q_abs_pos]: span_lo (0 for non-spans) skips the
    // prefix + sibling spans; `i <= q_abs_pos` is the causal upper bound (so we
    // no longer iterate-then-`continue` past it). For non-spans span_lo==0 and
    // the visited keys/order are identical to the old `i<kv_len; if(i>q_abs_pos)`
    // form → byte-identical. For a span query the loop runs O(span), not O(N).
    for (uint i = span_lo + simd_gid; i < kv_len; i += uint(BN)) {
        // Causal upper bound. SPAN query (q_span != 0): BREAK — the loop is bounded
        // [span_lo, q_abs_pos] = the span only (the real O(N)->O(span) cut). NON-spans
        // (q_span == 0): CONTINUE, byte-identical to the original full-causal scan.
        if (i > q_abs_pos) { if (q_span != 0u) { break; } else { continue; } }
        // Sliding window: attend iff q_abs_pos - i < window.
        if (ATTN_WINDOW > 0 && int(q_abs_pos) - int(i) >= ATTN_WINDOW) {
            continue;
        }

        const uint logical_block = i / block_size;
        const uint bt_raw = row_block_table[logical_block];
        const uint physical_block = bt_raw & ATTN_BT_MASK;
        const bool do_rot = (ATTN_ROR != 0u) && ((bt_raw & 0x80000000u) != 0u);
        const uint token_in_block = i - logical_block * block_size;
        // Chunked KV: deref the chunk backing this physical block.
        // ATTN_BLOCKS_PER_CHUNK is a function constant. When set to 0 the
        // compiler dead-eliminates the chunked branch — used by the
        // single-buffer-per-layer mode where `k_cache[0]` holds the layer
        // base address and physical_block is the full offset (no modulo,
        // no per-block chunk_table load). When non-zero the path matches
        // the reactive chunked KV pool.
        uint chunk;
        uint blk_in_chunk;
        if (ATTN_BLOCKS_PER_CHUNK == 0u) {
            chunk = 0u;
            blk_in_chunk = physical_block;
        } else {
            chunk = physical_block / ATTN_BLOCKS_PER_CHUNK;
            blk_in_chunk = physical_block % ATTN_BLOCKS_PER_CHUNK;
        }
        device const bfloat* k_ptr =
            (device const bfloat*)k_cache[chunk]
            + blk_in_chunk   * kv_blk_stride
            + kv_head_idx    * kv_head_stride
            + token_in_block * kv_tok_stride
            + simd_lid * qk_per_thread;
        device const bfloat* v_ptr =
            (device const bfloat*)v_cache[chunk]
            + blk_in_chunk   * kv_blk_stride
            + kv_head_idx    * kv_head_stride
            + token_in_block * kv_tok_stride
            + simd_lid * qk_per_thread;

        // q·k; span blocks (do_rot) re-roped to position `i` first, every
        // other key dots directly from k_ptr. See attention_via_cache_v2.
        U score = 0;
        if (do_rot) {
            U k_loc[16];
            for (uint j = 0; j < qk_per_thread; ++j) {
                k_loc[j] = U(k_ptr[j]);
            }
            const uint half_dim = ATTN_ROT_DIM / 2u;
            device const bfloat* cos_row = cos_sin + i * ATTN_ROT_DIM;
            device const bfloat* sin_row = cos_row + half_dim;
            rope_on_read_k_slice<bfloat>(k_loc, qk_per_thread, simd_lid,
                                 cos_row, sin_row, half_dim, ATTN_PAIR_OFF);
            for (uint j = 0; j < qk_per_thread; ++j) {
                score += q_reg[j] * k_loc[j];
            }
        } else {
            for (uint j = 0; j < qk_per_thread; ++j) {
                score += q_reg[j] * U(k_ptr[j]);
            }
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = metal::fast::exp(max_score - new_max);
        U exp_score = metal::fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint j = 0; j < qk_per_thread; ++j) {
            o_reg[j] = o_reg[j] * factor + exp_score * U(v_ptr[j]);
        }
    }

    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    U other_max = tg_max[simd_lid];
    U global_max = simd_max(other_max);
    U factor = metal::fast::exp(other_max - global_max);
    U global_sum = simd_sum(tg_sum[simd_lid] * factor);

    for (uint j = 0; j < qk_per_thread; ++j) {
        tg_outputs[simd_lid * BD + simd_gid] = o_reg[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        U val = tg_outputs[simd_gid * BD + simd_lid] * factor;
        U combined = simd_sum(val);
        if (global_sum != 0) {
            combined = combined / global_sum;
        }
        o_reg[j] = combined;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        device bfloat* o_ptr = o_row + simd_gid * qk_per_thread;
        for (uint j = 0; j < qk_per_thread; ++j) {
            o_ptr[j] = bfloat(o_reg[j]);
        }
    }
}

// ── rope-once kernel (spans rope-on-read, gqa_shared global prefill) ─────────
//
// One pass over a request's K for one layer: read the UNROTATED K from the
// paged cache, rope each row to its absolute position (cos/sin from the table,
// done ONCE per key), write the roped K into the dense scratch. The cache is
// never mutated. O(K) — amortizes to ~0% against the O(NQ·K) attention. This is
// the gqa_shared twin of `rope_once_{steel,nax}`; head_dim 512 (gemma4 global)
// has no steel instantiation, so the steel/NAX rope-once symbols don't cover
// it. Unlike those (template BLOCK_SIZE_), this reads ATTN_HEAD_DIM /
// ATTN_BLOCK_SIZE / ATTN_NUM_KV_HEADS / ATTN_ROT_DIM / ATTN_PAIR_OFF /
// ATTN_BLOCKS_PER_CHUNK / ATTN_MAX_BLOCKS_PER_SEQ from function constants, so
// one symbol covers any (head_dim, block_size) the gqa_shared kernel takes
// (gemma4 global: hd 512, bs 32).
//
// The scratch it writes is DENSE, indexed by LOGICAL block with the SAME
// per-block strides as the cache:
//   scratch + logical_block * (num_kv_heads*BLOCK_SIZE*head_dim)
//           + kv_head * (BLOCK_SIZE*head_dim) + tib*head_dim + dim
// so the gqa_shared kernel's TG-cooperative K stage reads it with the byte-
// identical `k_blk + sub*head_dim + idx` math (logical-block base, no chunk
// indirection).
//
// Grid: gid.y = logical block index; gid.x decodes to
//   (kv_head, token_in_block, rotary pair index d in [0, rot_dim/2)).
// A thread ropes the PAIR (d, d+pair_off) of one K row. Rows whose block is
// NOT flagged unrotated (block_table bit 31 clear) are COPIED through
// unchanged so the scratch is a complete K image either way (the attention
// always reads the scratch when ATTN_K_SCRATCH is set). Bit-exact with
// `rope_append` / `cpu_rope_k` (round to the cache dtype on store).
//
//   buffer(0) k_scratch     dense roped-K (dtype T), logical-block indexed
//   buffer(1) block_table   [batch, max_blocks_per_seq]  (bit 31 = unrotated)
//   buffer(2) k_cache       per-layer chunk-address table (uint64 gpuAddrs)
//   buffer(3) seq_used_k    [batch]
//   buffer(4) cos_sin       [max_pos, rot_dim]
template <typename T>
inline void rope_once_gqa_shared_body(
    device T*          k_scratch,
    const device uint* block_table,
    const device uint64_t* k_cache,
    const device uint* seq_used_k,
    const device T*    cos_sin,
    uint2 gid)
{
    const uint logical_block = gid.y;
    // ⛔⛔⛔⭐⭐⭐⭐⭐ THIS HARD-CODED 0 IS THE BATCHED-DECODE / BATCHED-PREFILL CORRUPTION.
    // MEASURED 2026-08-11 on metal (13s repro, granite-3.3-8b-4bit):
    //   * 12 short distinct prompts, no prefix-cache hits: 1 of 12 correct. The ONLY correct one is the
    //     request whose prefill had a step to ITSELF.
    //   * same 4 prompts with --max-num-seqs 1: 4 of 4. Solo: 4 of 4.
    // WHY: rope-once re-ropes K out of the paged cache into a dense `k_scratch` that the steel/NAX paged
    // prefill attention then reads INSTEAD of the cache. With `seq_idx = 0` it reads
    // `block_table + 0 * MAX_BLOCKS_PER_SEQ` and `seq_used_k[0]`, and the scratch has NO batch dimension —
    // so ONE K image, built entirely from batch row 0's history, is served to EVERY row of the batch.
    // Row 0 is therefore always right (a decode sharing the step IS row 0) and every seq_idx > 0 attends
    // row 0's keys, which reads as fluent, confidently wrong output.
    // ⛔ NOT spans-gated: `ROPE_ON_READ` derives from `fuf_uses_rotary` (macros/src/lib.rs:1841), i.e. it is
    // on for EVERY rope model including granite — `SPANS_OFF=1` changes nothing (measured, identical 1/12).
    // ⛔ THE OBVIOUS FIX DOES NOT FIT IN MEMORY. Adding a batch dimension means
    // `num_pages(block_cap=8192) * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM * 2B` ≈ 268 MB PER SEQUENCE
    // (lowering.rs ~4272 sizes exactly one sequence). x max_num_seqs is not an option.
    // ⏭ THE FIX IS TO STOP STAGING: for a multi-sequence step, read the paged cache per row (each request
    // through its OWN block_table row) rather than through this scratch — which is what vLLM does (K is
    // stored already-roped and `block_table.py:186` commits all `num_reqs` rows; there is no staging buffer
    // at all). The comment below was true when written — the caller really was single-sequence — and it
    // outlived that fact, which is why this survived every host-side audit.
    const uint seq_idx       = 0; // single-sequence prefill ONLY — see above; WRONG for num_reqs > 1

    const uint head_dim    = ATTN_HEAD_DIM;
    const uint num_kv      = ATTN_NUM_KV_HEADS;
    const uint block_size  = ATTN_BLOCK_SIZE;
    const uint max_blocks  = ATTN_MAX_BLOCKS_PER_SEQ;
    const uint rot_dim     = ATTN_ROT_DIM;
    const uint pair_off    = ATTN_PAIR_OFF;
    const uint half_dim    = rot_dim / 2u;

    const uint kv_len    = seq_used_k[seq_idx];
    const uint num_pages = (kv_len + block_size - 1u) / block_size;
    if (logical_block >= num_pages) {
        return;
    }

    // gid.x decodes to (kv_head, token_in_block, rotary index d in [0,half_dim)).
    const uint per_head = block_size * half_dim;
    const uint kv_head  = gid.x / per_head;
    const uint rem      = gid.x % per_head;
    const uint tib      = rem / half_dim;     // token in block
    const uint d        = rem % half_dim;     // rotary pair index
    if (kv_head >= num_kv) {
        return;
    }

    const uint pos = logical_block * block_size + tib;

    const uint kv_blk_stride = num_kv * block_size * head_dim;
    const uint kv_head_off   = kv_head * (block_size * head_dim);
    const device uint* row_block_table = block_table + seq_idx * max_blocks;

    // Source row (cache, possibly unrotated) and dest row (scratch, dense).
    const uint bt_raw   = row_block_table[logical_block];
    const uint physical = bt_raw & 0x7FFFFFFFu;
    uint chunk;
    uint bic;
    if (ATTN_BLOCKS_PER_CHUNK == 0u) {
        chunk = 0u;
        bic   = physical;
    } else {
        chunk = physical / ATTN_BLOCKS_PER_CHUNK;
        bic   = physical % ATTN_BLOCKS_PER_CHUNK;
    }
    const device T* k_blk =
        (const device T*)k_cache[chunk] + bic * kv_blk_stride + kv_head_off;
    device T* dst_blk = k_scratch + logical_block * kv_blk_stride + kv_head_off;
    const device T* k_row = k_blk + tib * head_dim;
    device T* dst_row = dst_blk + tib * head_dim;

    const bool past_len = pos >= kv_len;
    const bool flagged  = (bt_raw & 0x80000000u) != 0u;

    // Pass-through copy of all NON-rotated head dims (proportional rope leaves
    // dims outside {[0,half_dim) ∪ [pair_off,pair_off+half_dim)} untouched).
    // Thread d==0 mirrors them once. For full NeoX (rot_dim==head_dim) this
    // loop is empty (every dim is in a rotary pair).
    if (d == 0u) {
        for (uint dd = 0u; dd < head_dim; dd++) {
            const bool is_low  = dd < half_dim;
            const bool is_high = (dd >= pair_off) && (dd < pair_off + half_dim);
            if (!is_low && !is_high) dst_row[dd] = k_row[dd];
        }
    }

    if (past_len || !flagged) {
        // Padding row (attention masks it) OR a block stored rotated already
        // (non-span) — copy the rotary pair through unchanged.
        dst_row[d]            = k_row[d];
        dst_row[pair_off + d] = k_row[pair_off + d];
        return;
    }

    // Re-rope the (d, pair_off+d) pair to absolute position `pos`.
    const device T* cr = cos_sin + pos * rot_dim;
    const float c  = float(cr[d]);
    const float s  = float(cr[half_dim + d]);
    const float x0 = float(k_row[d]);
    const float x1 = float(k_row[pair_off + d]);
    dst_row[d]            = T(x0 * c - x1 * s);
    dst_row[pair_off + d] = T(x1 * c + x0 * s);
}

kernel void rope_once_gqa_shared_f16_specialized(
    device       half*     k_scratch   [[buffer(0)]],
    const device uint*     block_table [[buffer(1)]],
    const device uint64_t* k_cache     [[buffer(2)]],
    const device uint*     seq_used_k  [[buffer(3)]],
    const device half*     cos_sin     [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    rope_once_gqa_shared_body<half>(
        k_scratch, block_table, k_cache, seq_used_k, cos_sin, gid);
}

kernel void rope_once_gqa_shared_bf16_specialized(
    device       bfloat*   k_scratch   [[buffer(0)]],
    const device uint*     block_table [[buffer(1)]],
    const device uint64_t* k_cache     [[buffer(2)]],
    const device uint*     seq_used_k  [[buffer(3)]],
    const device bfloat*   cos_sin     [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    rope_once_gqa_shared_body<bfloat>(
        k_scratch, block_table, k_cache, seq_used_k, cos_sin, gid);
}

/// GQA-cooperative paged SDPA prefill (f16): one threadgroup per
/// (kv_head, query); each simdgroup owns ONE q-head of the GQA group
/// and K/V blocks are staged through threadgroup memory ONCE per
/// query, shared by all `gqa = NUM_Q_HEADS / NUM_KV_HEADS` heads.
///
/// Motivation: `attention_prefill_sdpa_v2_paged_*` launches one TG
/// per (q_head, query) — at high GQA every K/V byte is re-streamed
/// from device `gqa` times. Gemma4's global layers (head_dim 512,
/// 16:1 GQA) ran bandwidth-bound at ~350 ms/layer on T=2930 prefill;
/// staging cuts device K/V traffic by `gqa`×.
///
/// Layout/semantics identical to the v2 kernel (same buffers, same
/// fn-consts incl. ATTN_WINDOW, same causal shift). Differences:
///   - grid = (NUM_KV_HEADS, total_q, 1); threads = (32*gqa, 1, 1)
///     (lowering asserts 32*gqa <= 1024 i.e. gqa <= 32).
///   - chunked online softmax per paged block (BLOCK_SIZE <= 16 keys
///     per stage; one block_table lookup per stage).
///   - no cross-simdgroup merge: each simdgroup covers the FULL key
///     range for its head; lanes store their own dim slice.
///
/// Constraints (enforced by the lowering arm): HEAD_DIM % 32 == 0,
/// HEAD_DIM <= 512, BLOCK_SIZE <= 16, 2 <= gqa <= 32.
kernel void attention_prefill_sdpa_gqa_shared_f16_specialized(
    device       half* output       [[buffer(0)]],   // [total_q, num_q_heads, head_dim]
    device const half* q            [[buffer(1)]],   // [total_q, num_q_heads, head_dim]
    device const uint* cu_seqlens_q [[buffer(2)]],   // [batch+1]
    device const uint* seq_used_k   [[buffer(3)]],   // [batch]
    device const uint* block_table  [[buffer(4)]],   // [batch, MAX_BLOCKS_PER_SEQ]
    device const uint64_t* k_cache  [[buffer(5)]],   // chunk-address table
    device const uint64_t* v_cache  [[buffer(6)]],   // chunk-address table
    // Spans, slot 7 has TWO mutually-exclusive uses (only one is bound per
    // dispatch; folded by ATTN_KSCR / ATTN_ROR function constants):
    //   ATTN_K_SCRATCH (rope-once-to-scratch): the DENSE pre-roped K scratch
    //     written ONCE by `rope_once_gqa_shared_*`. K is staged from here with
    //     the byte-identical cache-load math (logical-block base) and NO
    //     per-tile rotation — the amortized-to-~0% long-context path.
    //   else ATTN_ROPE_ON_READ (in-kernel rope): f16 cos_sin; K is staged from
    //     the cache and rotated in-place in smem on every q-tile (the per-tile
    //     redundancy the scratch path replaces).
    // The unrotated flag rides in block_table bit 31 (no flag buffer).
    device const half*     cos_sin               [[buffer(7)]],
    uint3  tg_pos    [[threadgroup_position_in_grid]],
    uint3  tid       [[thread_position_in_threadgroup]],
    uint   simd_gid  [[simdgroup_index_in_threadgroup]],
    uint   simd_lid  [[thread_index_in_simdgroup]])
{
    typedef float U;
    // Slot 7 aliases cos_sin or the pre-roped K scratch depending on ATTN_KSCR.
    device const half* k_scratch = cos_sin;

    const uint head_dim    = ATTN_HEAD_DIM;
    const uint num_q       = ATTN_NUM_Q_HEADS;
    const uint num_kv      = ATTN_NUM_KV_HEADS;
    const uint block_size  = ATTN_BLOCK_SIZE;
    const uint max_blocks  = ATTN_MAX_BLOCKS_PER_SEQ;
    const float scale      = ATTN_SCALE_FC;
    const uint qk_per_thread = head_dim / 32u;
    const uint gqa         = num_q / num_kv;
    const uint tg_threads  = gqa * 32u;

    const uint kv_head_idx = tg_pos.x;            // 0..NUM_KV_HEADS
    const uint global_q    = tg_pos.y;            // 0..total_q
    const uint q_head_idx  = kv_head_idx * gqa + simd_gid;

    // Staged K/V block: BLOCK_SIZE x HEAD_DIM elements, statically
    // sized to the 16 x 512 maximum (16 KB at 2 B/elem).
    threadgroup half kv_smem[16 * 512];

    // Locate this Q token's sequence via cu_seqlens_q (same scan +
    // sentinel as the v2 kernel).
    uint seq_idx   = 0;
    uint seq_start = 0;
    uint seq_end   = 0;
    bool in_range  = false;
    for (uint b = 0; b < 1024u; ++b) {
        const uint lo = cu_seqlens_q[b];
        const uint hi = cu_seqlens_q[b + 1];
        if (global_q >= lo && global_q < hi) {
            seq_idx   = b;
            seq_start = lo;
            seq_end   = hi;
            in_range  = true;
            break;
        }
        if (hi <= lo) break;       // sentinel: end of batch
    }
    if (!in_range) {
        // Padding lane: zero this TG's gqa output rows (each lane
        // writes its own dim slice of its simdgroup's head).
        device half* o_ptr =
            output + (global_q * num_q + q_head_idx) * head_dim
                   + simd_lid * qk_per_thread;
        for (uint j = 0; j < qk_per_thread; ++j) o_ptr[j] = half(0);
        return;
    }

    const uint new_q_for_seq = seq_end - seq_start;
    const uint q_pos_in_new  = global_q - seq_start;
    const uint kv_len        = seq_used_k[seq_idx];
    const uint q_abs_pos     = (kv_len - new_q_for_seq) + q_pos_in_new;

    const uint kv_blk_stride  = num_kv * block_size * head_dim;
    const uint kv_head_stride = block_size * head_dim;

    thread U q_reg[16];                 // qk_per_thread <= 16
    thread U o_reg[16];
    thread U p_reg[16];                 // per-chunk probs (block_size <= 16)

    device const half* q_row = q + (global_q * num_q + q_head_idx) * head_dim;
    device       half* o_row = output + (global_q * num_q + q_head_idx) * head_dim;
    device const uint* row_block_table = block_table + seq_idx * max_blocks;

    for (uint i = 0; i < qk_per_thread; ++i) {
        q_reg[i] = U(scale) * U(q_row[simd_lid * qk_per_thread + i]);
        o_reg[i] = 0;
    }

    U run_max = -FLT_MAX;
    U sum_exp = 0;

    // Key range for THIS query: causal cap at q_abs_pos, window floor
    // at q_abs_pos - window + 1. Blocks are processed stage-by-stage;
    // all simdgroups walk the same stages (the staging is TG-wide) but
    // skip invalid keys per-element. The block range uses the TG-wide
    // bounds = this query's own bounds (one query per TG).
    const uint last_key = min(kv_len - 1u, q_abs_pos);
    uint first_key = 0;
    if (ATTN_WINDOW > 0 && int(q_abs_pos) - ATTN_WINDOW + 1 > 0) {
        first_key = uint(int(q_abs_pos) - ATTN_WINDOW + 1);
    }
    const uint blk_lo = first_key / block_size;
    const uint blk_hi = last_key / block_size;          // inclusive

    // Staging is done in <=16-token SUB-STAGES so the smem (16*head_dim) and
    // p_reg[16] stay within limits for a page-unified block_size > 16 (gemma4
    // full class: 32). block_size <= 16 → one sub-stage (identical to before).
    const uint STAGE_MAX = 16u;
    for (uint blk = blk_lo; blk <= blk_hi; ++blk) {
        // Spans: block_table bit 31 = this block's K is stored unrotated.
        const uint bt_raw = row_block_table[blk];
        const uint physical_block =
            (ATTN_ROR != 0u) ? (bt_raw & 0x7FFFFFFFu) : bt_raw;
        const bool blk_do_rot = (ATTN_ROR != 0u) && ((bt_raw & 0x80000000u) != 0u);
        uint chunk;
        uint blk_in_chunk;
        if (ATTN_BLOCKS_PER_CHUNK == 0u) {
            chunk = 0u;
            blk_in_chunk = physical_block;
        } else {
            chunk = physical_block / ATTN_BLOCKS_PER_CHUNK;
            blk_in_chunk = physical_block % ATTN_BLOCKS_PER_CHUNK;
        }
        // Rope-once-to-scratch (spans): when ATTN_KSCR, K is read PRE-ROPED
        // from the dense scratch (logical-block indexed, no chunk indirection),
        // already roped to each key's position by `rope_once_gqa_shared` — so
        // the in-kernel per-tile rotation below folds away. V always reads from
        // the cache (V has no RoPE). `blk_do_rot` is unused on this path.
        device const half* k_blk =
            (ATTN_KSCR != 0u)
                ? (k_scratch + blk * kv_blk_stride + kv_head_idx * kv_head_stride)
                : ((device const half*)k_cache[chunk]
                   + blk_in_chunk * kv_blk_stride + kv_head_idx * kv_head_stride);
        device const half* v_blk =
            (device const half*)v_cache[chunk]
            + blk_in_chunk * kv_blk_stride + kv_head_idx * kv_head_stride;

        for (uint sub = 0; sub < block_size; sub += STAGE_MAX) {
            const uint stage     = min(STAGE_MAX, block_size - sub);
            const uint base_key   = blk * block_size + sub;
            const uint stage_elems = stage * head_dim;

            // ── Stage K sub-block (TG-cooperative) ──────────────────
            {
                device const half* k_base = k_blk + sub * head_dim;
                for (uint idx = tid.x; idx < stage_elems; idx += tg_threads) {
                    kv_smem[idx] = k_base[idx];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ── Rope-on-read: re-rope the staged K in smem to each key's
            // position for blocks stored unrotated (matches rope_append).
            // Each thread owns disjoint (kk,d) pairs → no smem race; the
            // whole block (incl. its barrier) folds away when ROR is off OR
            // when K is read pre-roped from the scratch (ATTN_KSCR).
            if (ATTN_ROR != 0u && ATTN_KSCR == 0u) {
                if (blk_do_rot) {
                    const uint half_dim = ATTN_ROT_DIM / 2u;
                    const uint pair_off = ATTN_PAIR_OFF;
                    const uint npairs   = stage * half_dim;
                    for (uint idx = tid.x; idx < npairs; idx += tg_threads) {
                        const uint kk  = idx / half_dim;
                        const uint d   = idx % half_dim;
                        const uint key = base_key + kk;
                        device const half* cos_row = cos_sin + key * ATTN_ROT_DIM;
                        device const half* sin_row = cos_row + half_dim;
                        const float c  = float(cos_row[d]);
                        const float s  = float(sin_row[d]);
                        const uint  b  = kk * head_dim;
                        const float x0 = float(kv_smem[b + d]);
                        const float x1 = float(kv_smem[b + pair_off + d]);
                        kv_smem[b + d]            = half(x0 * c - x1 * s);
                        kv_smem[b + pair_off + d] = half(x1 * c + x0 * s);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // ── Scores for this sub-stage's keys (per simdgroup) ────
            U chunk_max = -FLT_MAX;
            for (uint kk = 0; kk < stage; ++kk) {
                const uint key = base_key + kk;
                const bool valid = key >= first_key && key <= last_key;
                U score = -FLT_MAX;
                if (valid) {
                    U partial = 0;
                    threadgroup const half* k_row =
                        kv_smem + kk * head_dim + simd_lid * qk_per_thread;
                    for (uint j = 0; j < qk_per_thread; ++j) {
                        partial += q_reg[j] * U(k_row[j]);
                    }
                    score = simd_sum(partial);
                    chunk_max = max(chunk_max, score);
                }
                p_reg[kk] = score;
            }

            // ── Chunked online-softmax update ───────────────────────
            if (chunk_max > -FLT_MAX) {
                const U new_max = max(run_max, chunk_max);
                const U factor = metal::fast::exp(run_max - new_max);
                sum_exp *= factor;
                for (uint j = 0; j < qk_per_thread; ++j) {
                    o_reg[j] *= factor;
                }
                for (uint kk = 0; kk < stage; ++kk) {
                    if (p_reg[kk] > -FLT_MAX) {
                        const U p = metal::fast::exp(p_reg[kk] - new_max);
                        p_reg[kk] = p;
                        sum_exp += p;
                    } else {
                        p_reg[kk] = 0;
                    }
                }
                run_max = new_max;
            } else {
                for (uint kk = 0; kk < stage; ++kk) p_reg[kk] = 0;
            }

            // ── Stage V sub-block over the same smem ────────────────
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                device const half* v_base = v_blk + sub * head_dim;
                for (uint idx = tid.x; idx < stage_elems; idx += tg_threads) {
                    kv_smem[idx] = v_base[idx];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ── Accumulate O over this sub-stage's keys ─────────────
            for (uint kk = 0; kk < stage; ++kk) {
                const U p = p_reg[kk];
                if (p != 0) {
                    threadgroup const half* v_row =
                        kv_smem + kk * head_dim + simd_lid * qk_per_thread;
                    for (uint j = 0; j < qk_per_thread; ++j) {
                        o_reg[j] += p * U(v_row[j]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // ── Store: lanes own disjoint dim slices; normalize by sum ──────
    device half* o_ptr = o_row + simd_lid * qk_per_thread;
    const U inv = (sum_exp != 0) ? (U(1) / sum_exp) : U(0);
    for (uint j = 0; j < qk_per_thread; ++j) {
        o_ptr[j] = half(o_reg[j] * inv);
    }
}

/// GQA-cooperative paged SDPA prefill (bf16): one threadgroup per
/// (kv_head, query); each simdgroup owns ONE q-head of the GQA group
/// and K/V blocks are staged through threadgroup memory ONCE per
/// query, shared by all `gqa = NUM_Q_HEADS / NUM_KV_HEADS` heads.
///
/// Motivation: `attention_prefill_sdpa_v2_paged_*` launches one TG
/// per (q_head, query) — at high GQA every K/V byte is re-streamed
/// from device `gqa` times. Gemma4's global layers (head_dim 512,
/// 16:1 GQA) ran bandwidth-bound at ~350 ms/layer on T=2930 prefill;
/// staging cuts device K/V traffic by `gqa`×.
///
/// Layout/semantics identical to the v2 kernel (same buffers, same
/// fn-consts incl. ATTN_WINDOW, same causal shift). Differences:
///   - grid = (NUM_KV_HEADS, total_q, 1); threads = (32*gqa, 1, 1)
///     (lowering asserts 32*gqa <= 1024 i.e. gqa <= 32).
///   - chunked online softmax per paged block (BLOCK_SIZE <= 16 keys
///     per stage; one block_table lookup per stage).
///   - no cross-simdgroup merge: each simdgroup covers the FULL key
///     range for its head; lanes store their own dim slice.
///
/// Constraints (enforced by the lowering arm): HEAD_DIM % 32 == 0,
/// HEAD_DIM <= 512, BLOCK_SIZE <= 16, 2 <= gqa <= 32.
kernel void attention_prefill_sdpa_gqa_shared_bf16_specialized(
    device       bfloat* output       [[buffer(0)]],   // [total_q, num_q_heads, head_dim]
    device const bfloat* q            [[buffer(1)]],   // [total_q, num_q_heads, head_dim]
    device const uint* cu_seqlens_q [[buffer(2)]],   // [batch+1]
    device const uint* seq_used_k   [[buffer(3)]],   // [batch]
    device const uint* block_table  [[buffer(4)]],   // [batch, MAX_BLOCKS_PER_SEQ]
    device const uint64_t* k_cache  [[buffer(5)]],   // chunk-address table
    device const uint64_t* v_cache  [[buffer(6)]],   // chunk-address table
    // Spans, slot 7 = cos_sin (in-kernel rope) OR the pre-roped K scratch
    // (ATTN_K_SCRATCH). See the f16 sibling. Unrotated flag rides in bit 31.
    device const bfloat*   cos_sin               [[buffer(7)]],
    uint3  tg_pos    [[threadgroup_position_in_grid]],
    uint3  tid       [[thread_position_in_threadgroup]],
    uint   simd_gid  [[simdgroup_index_in_threadgroup]],
    uint   simd_lid  [[thread_index_in_simdgroup]])
{
    typedef float U;
    // Slot 7 aliases cos_sin or the pre-roped K scratch depending on ATTN_KSCR.
    device const bfloat* k_scratch = cos_sin;

    const uint head_dim    = ATTN_HEAD_DIM;
    const uint num_q       = ATTN_NUM_Q_HEADS;
    const uint num_kv      = ATTN_NUM_KV_HEADS;
    const uint block_size  = ATTN_BLOCK_SIZE;
    const uint max_blocks  = ATTN_MAX_BLOCKS_PER_SEQ;
    const float scale      = ATTN_SCALE_FC;
    const uint qk_per_thread = head_dim / 32u;
    const uint gqa         = num_q / num_kv;
    const uint tg_threads  = gqa * 32u;

    const uint kv_head_idx = tg_pos.x;            // 0..NUM_KV_HEADS
    const uint global_q    = tg_pos.y;            // 0..total_q
    const uint q_head_idx  = kv_head_idx * gqa + simd_gid;

    // Staged K/V block: BLOCK_SIZE x HEAD_DIM elements, statically
    // sized to the 16 x 512 maximum (16 KB at 2 B/elem).
    threadgroup bfloat kv_smem[16 * 512];

    // Locate this Q token's sequence via cu_seqlens_q (same scan +
    // sentinel as the v2 kernel).
    uint seq_idx   = 0;
    uint seq_start = 0;
    uint seq_end   = 0;
    bool in_range  = false;
    for (uint b = 0; b < 1024u; ++b) {
        const uint lo = cu_seqlens_q[b];
        const uint hi = cu_seqlens_q[b + 1];
        if (global_q >= lo && global_q < hi) {
            seq_idx   = b;
            seq_start = lo;
            seq_end   = hi;
            in_range  = true;
            break;
        }
        if (hi <= lo) break;       // sentinel: end of batch
    }
    if (!in_range) {
        // Padding lane: zero this TG's gqa output rows (each lane
        // writes its own dim slice of its simdgroup's head).
        device bfloat* o_ptr =
            output + (global_q * num_q + q_head_idx) * head_dim
                   + simd_lid * qk_per_thread;
        for (uint j = 0; j < qk_per_thread; ++j) o_ptr[j] = bfloat(0);
        return;
    }

    const uint new_q_for_seq = seq_end - seq_start;
    const uint q_pos_in_new  = global_q - seq_start;
    const uint kv_len        = seq_used_k[seq_idx];
    const uint q_abs_pos     = (kv_len - new_q_for_seq) + q_pos_in_new;

    const uint kv_blk_stride  = num_kv * block_size * head_dim;
    const uint kv_head_stride = block_size * head_dim;

    thread U q_reg[16];                 // qk_per_thread <= 16
    thread U o_reg[16];
    thread U p_reg[16];                 // per-chunk probs (block_size <= 16)

    device const bfloat* q_row = q + (global_q * num_q + q_head_idx) * head_dim;
    device       bfloat* o_row = output + (global_q * num_q + q_head_idx) * head_dim;
    device const uint* row_block_table = block_table + seq_idx * max_blocks;

    for (uint i = 0; i < qk_per_thread; ++i) {
        q_reg[i] = U(scale) * U(q_row[simd_lid * qk_per_thread + i]);
        o_reg[i] = 0;
    }

    U run_max = -FLT_MAX;
    U sum_exp = 0;

    // Key range for THIS query: causal cap at q_abs_pos, window floor
    // at q_abs_pos - window + 1. Blocks are processed stage-by-stage;
    // all simdgroups walk the same stages (the staging is TG-wide) but
    // skip invalid keys per-element. The block range uses the TG-wide
    // bounds = this query's own bounds (one query per TG).
    const uint last_key = min(kv_len - 1u, q_abs_pos);
    uint first_key = 0;
    if (ATTN_WINDOW > 0 && int(q_abs_pos) - ATTN_WINDOW + 1 > 0) {
        first_key = uint(int(q_abs_pos) - ATTN_WINDOW + 1);
    }
    const uint blk_lo = first_key / block_size;
    const uint blk_hi = last_key / block_size;          // inclusive

    // <=16-token sub-stages so smem (16*head_dim) + p_reg[16] fit a page-
    // unified block_size > 16 (gemma4 full class: 32). bs <= 16 → one stage.
    const uint STAGE_MAX = 16u;
    for (uint blk = blk_lo; blk <= blk_hi; ++blk) {
        // Spans: block_table bit 31 = this block's K is stored unrotated.
        const uint bt_raw = row_block_table[blk];
        const uint physical_block =
            (ATTN_ROR != 0u) ? (bt_raw & 0x7FFFFFFFu) : bt_raw;
        const bool blk_do_rot = (ATTN_ROR != 0u) && ((bt_raw & 0x80000000u) != 0u);
        uint chunk;
        uint blk_in_chunk;
        if (ATTN_BLOCKS_PER_CHUNK == 0u) {
            chunk = 0u;
            blk_in_chunk = physical_block;
        } else {
            chunk = physical_block / ATTN_BLOCKS_PER_CHUNK;
            blk_in_chunk = physical_block % ATTN_BLOCKS_PER_CHUNK;
        }
        // Rope-once-to-scratch (spans): K pre-roped from the dense scratch
        // (logical-block indexed) when ATTN_KSCR; else the cache. V always
        // from the cache. See the f16 sibling.
        device const bfloat* k_blk =
            (ATTN_KSCR != 0u)
                ? (k_scratch + blk * kv_blk_stride + kv_head_idx * kv_head_stride)
                : ((device const bfloat*)k_cache[chunk]
                   + blk_in_chunk * kv_blk_stride + kv_head_idx * kv_head_stride);
        device const bfloat* v_blk =
            (device const bfloat*)v_cache[chunk]
            + blk_in_chunk * kv_blk_stride + kv_head_idx * kv_head_stride;

        for (uint sub = 0; sub < block_size; sub += STAGE_MAX) {
            const uint stage      = min(STAGE_MAX, block_size - sub);
            const uint base_key    = blk * block_size + sub;
            const uint stage_elems = stage * head_dim;

            // ── Stage K sub-block (TG-cooperative) ──────────────────
            {
                device const bfloat* k_base = k_blk + sub * head_dim;
                for (uint idx = tid.x; idx < stage_elems; idx += tg_threads) {
                    kv_smem[idx] = k_base[idx];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ── Rope-on-read: re-rope staged K in smem (see f16 sibling).
            // Folds away when K is read pre-roped from the scratch (ATTN_KSCR).
            if (ATTN_ROR != 0u && ATTN_KSCR == 0u) {
                if (blk_do_rot) {
                    const uint half_dim = ATTN_ROT_DIM / 2u;
                    const uint pair_off = ATTN_PAIR_OFF;
                    const uint npairs   = stage * half_dim;
                    for (uint idx = tid.x; idx < npairs; idx += tg_threads) {
                        const uint kk  = idx / half_dim;
                        const uint d   = idx % half_dim;
                        const uint key = base_key + kk;
                        device const bfloat* cos_row = cos_sin + key * ATTN_ROT_DIM;
                        device const bfloat* sin_row = cos_row + half_dim;
                        const float c  = float(cos_row[d]);
                        const float s  = float(sin_row[d]);
                        const uint  b  = kk * head_dim;
                        const float x0 = float(kv_smem[b + d]);
                        const float x1 = float(kv_smem[b + pair_off + d]);
                        kv_smem[b + d]            = bfloat(x0 * c - x1 * s);
                        kv_smem[b + pair_off + d] = bfloat(x1 * c + x0 * s);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // ── Scores for this sub-stage's keys (per simdgroup) ────
            U chunk_max = -FLT_MAX;
            for (uint kk = 0; kk < stage; ++kk) {
                const uint key = base_key + kk;
                const bool valid = key >= first_key && key <= last_key;
                U score = -FLT_MAX;
                if (valid) {
                    U partial = 0;
                    threadgroup const bfloat* k_row =
                        kv_smem + kk * head_dim + simd_lid * qk_per_thread;
                    for (uint j = 0; j < qk_per_thread; ++j) {
                        partial += q_reg[j] * U(k_row[j]);
                    }
                    score = simd_sum(partial);
                    chunk_max = max(chunk_max, score);
                }
                p_reg[kk] = score;
            }

            // ── Chunked online-softmax update ───────────────────────
            if (chunk_max > -FLT_MAX) {
                const U new_max = max(run_max, chunk_max);
                const U factor = metal::fast::exp(run_max - new_max);
                sum_exp *= factor;
                for (uint j = 0; j < qk_per_thread; ++j) {
                    o_reg[j] *= factor;
                }
                for (uint kk = 0; kk < stage; ++kk) {
                    if (p_reg[kk] > -FLT_MAX) {
                        const U p = metal::fast::exp(p_reg[kk] - new_max);
                        p_reg[kk] = p;
                        sum_exp += p;
                    } else {
                        p_reg[kk] = 0;
                    }
                }
                run_max = new_max;
            } else {
                for (uint kk = 0; kk < stage; ++kk) p_reg[kk] = 0;
            }

            // ── Stage V sub-block over the same smem ────────────────
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                device const bfloat* v_base = v_blk + sub * head_dim;
                for (uint idx = tid.x; idx < stage_elems; idx += tg_threads) {
                    kv_smem[idx] = v_base[idx];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ── Accumulate O over this sub-stage's keys ─────────────
            for (uint kk = 0; kk < stage; ++kk) {
                const U p = p_reg[kk];
                if (p != 0) {
                    threadgroup const bfloat* v_row =
                        kv_smem + kk * head_dim + simd_lid * qk_per_thread;
                    for (uint j = 0; j < qk_per_thread; ++j) {
                        o_reg[j] += p * U(v_row[j]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // ── Store: lanes own disjoint dim slices; normalize by sum ──────
    device bfloat* o_ptr = o_row + simd_lid * qk_per_thread;
    const U inv = (sum_exp != 0) ? (U(1) / sum_exp) : U(0);
    for (uint j = 0; j < qk_per_thread; ++j) {
        o_ptr[j] = bfloat(o_reg[j] * inv);
    }
}
