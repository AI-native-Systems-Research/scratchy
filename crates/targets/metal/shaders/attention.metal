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
// Baked constant slots (must match the typed `*Constants` in `tape/kernel_constants.rs`):
//   0  ATTN_HEAD_DIM           uint
//   1  ATTN_NUM_Q_HEADS        uint
//   2  ATTN_NUM_KV_HEADS       uint
//   3  ATTN_SCALE_FC           float
//   4  ATTN_BLOCK_SIZE         uint
//   5  ATTN_MAX_BLOCKS_PER_SEQ uint

#include <metal_stdlib>
#include "baked.h"
#include "kv_writer.h"
using namespace metal;



SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_HEAD_DIM, 0);
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_NUM_Q_HEADS, 1);
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_NUM_KV_HEADS, 2);
SCRATCHY_CONSTANT_OPTIONAL(float, ATTN_SCALE_FC, 3);
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_BLOCK_SIZE, 4);
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_MAX_BLOCKS_PER_SEQ, 5);
// Reactive (chunked) KV pool: the `k_cache`/`v_cache` bindings are
// per-layer chunk-address TABLES (device uint64 gpuAddresses), not the
// cache buffers. A resolved physical block id derefs
// `table[physical_block / BLOCKS_PER_CHUNK]` then addresses with
// `physical_block % BLOCKS_PER_CHUNK`. See
// scratchy-target-metal's `BLOCKS_PER_CHUNK`. (`attention_via_cache_v2_*`
// reads constant slot 6 via `AttentionViaCacheConstants`;
// `attention_prefill_sdpa_v2_paged_*` via `AttentionPrefillPagedConstants`.)
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_BLOCKS_PER_CHUNK, 6);

// Sliding-window attention (Gemma2/3/4 alternating layers). A query at
// absolute position `q` attends to keys `k` with `0 <= q - k < window`
// (self + window-1 prior — matches both mlx `create_causal_mask`
// `linds < rinds + window_size` and HF's `(q-k) >= window` masking).
// `0` disables the window entirely; the compiler folds the checks away
// for non-sliding pipelines (full-attention models pass 0).
SCRATCHY_CONSTANT_OPTIONAL(int, ATTN_WINDOW, 7);

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
// constants are OPTIONAL: when the bake does not set slot 10, that
// slot's `_SET` flag folds the whole rotation path away and buffers
// 6/7 are never accessed — non-spans pipelines are byte-
// identical to before. Set by `AttentionViaCacheConstants` only when
// rope-on-read is active.
//   8  ATTN_ROT_DIM       uint  — rotary dim (full head_dim for NeoX;
//                                  partial_rotary_factor*head_dim for
//                                  gemma4 global proportional rope)
//   9  ATTN_PAIR_OFF      uint  — NeoX pairing offset: rot_dim/2 for full
//                                  rope, head_dim/2 for proportional rope
//                                  (MUST match rope_append's ROPE_PAIR_OFF)
//  10  ATTN_ROPE_ON_READ  uint  — 0/1 master switch
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_ROT_DIM, 8);
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_PAIR_OFF, 9);
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_ROPE_ON_READ, 10);
constant bool ATTN_ROR_DEFINED  = ATTN_ROPE_ON_READ_SET;
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
//      slot's `_SET` flag folds the scratch path away and
//      the kernel keeps the in-kernel cos_sin rotation (or, with ROR also
//      unset, the byte-identical non-spans cache read). The rope-once kernel
//      itself reads ATTN_ROT_DIM / ATTN_PAIR_OFF / the per-block bit-31 flag.
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_K_SCRATCH, 11);
constant bool ATTN_K_SCRATCH_DEF    = ATTN_K_SCRATCH_SET;
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
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_PAIR_CORESIDENT, 12);
constant bool ATTN_PAIR_CORESIDENT_DEF = ATTN_PAIR_CORESIDENT_SET;
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
//      slot's `_SET` flag folds the decode path away and
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
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_TQ_BITS, 13);
constant bool ATTN_TQ_DEF  = ATTN_TQ_BITS_SET;
constant uint ATTN_TQ      = ATTN_TQ_DEF ? ATTN_TQ_BITS : 0u;

//  14  ATTN_TQ_K_BIAS / 15  ATTN_TQ_V_BIAS — set when the packed codes hold that
//      operand MINUS its projection bias (turboquant_offset.h; the codec's
//      error scales with the coded vector's norm). Decode (bias at buffers
//      14 / 15): K's bias is rotated with the key, so each packed key's score
//      gains q·R_i·b (needs the rope-on-read table and pairing); V's is added
//      once to the output, since the softmax weights sum to 1. Prefill staging
//      (bias at buffer 10) restores it into each cached key's rotated image.
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_TQ_K_BIAS, 14);
constant bool ATTN_TQ_KB = ATTN_TQ_K_BIAS_SET;
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_TQ_V_BIAS, 15);
constant bool ATTN_TQ_VB = ATTN_TQ_V_BIAS_SET;

//  16  ATTN_TQ_HEADS — query heads per decode threadgroup under TurboQuant:
//      consecutive heads of one KV head, so each key's codes are decoded once
//      for all of them (decode is ALU-bound on that decode). Unset: 1.
//      heads * head_dim / 32 <= 32.
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_TQ_HEADS_FC, 16);
constant uint ATTN_TQ_HEADS = ATTN_TQ_HEADS_FC_SET ? ATTN_TQ_HEADS_FC : 1u;

//  17  ATTN_TQ_STAGE_PASS — the rows one `tq_stage_rotated` dispatch stages:
//      1 = the step's new rows (read from the cache their writer just filled),
//      2 = the cached rows (decoded from the packed store). The lowering runs
//      pass 1 then pass 2: a row that is new for one sequence can be a
//      prefix-cache hit for another in the same step, and both rewrite it in
//      place, so the two writes must be ordered, not concurrent.
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_TQ_STAGE_PASS, 17);

// 18  ATTN_SPLITS — the threadgroups `attention_decode_gqa_tq` spreads a KV head's keys over, a
//     constant of the bake's KV cap rung: each takes a contiguous run of the sequence's key blocks
//     and stores its query heads' partials (the max, the sum, the unnormalized output) for
//     `attention_via_cache_v2_combine` to merge. Unset: one.
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_SPLITS_FC, 18);
constant constexpr uint ATTN_SPLITS = ATTN_SPLITS_FC_SET ? ATTN_SPLITS_FC : 1u;
// The decode merge's output slices per barrier round (`attn_decode_merge`): its scratch holds
// this many 32 x 32 transposes, each row padded to 33 floats so a simdgroup's transposed stores
// (a lane a row) land in 32 different banks.
constant constexpr uint ATTN_MERGE_SLICES = 4u;
constant constexpr uint ATTN_MERGE_STRIDE = 33u;

// 19 / 20  ATTN_FOLD_ROT_DIM / ATTN_FOLD_PAIR_OFF — set: a one-row step's decode attention runs
//     its KV writer (`MetalFusion::RopedAttention`), roping as the writer would — NeoX pairs
//     (d, d + pair off), d < rot dim / 2. It ropes its query heads in place, builds the new K/V
//     row in threadgroup memory, where the key loop reads the step's own key, and the threadgroup
//     owning the KV head writes the row to the cache and, under TurboQuant, encodes it into the
//     packed store, exactly as `rope_append_*` does. Buffers 17..22: the raw K and V rows, the
//     positions, the codebook's boundaries, the writer's rotary table and its rotated K bias's.
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_FOLD_ROT_DIM, 19);
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_FOLD_PAIR_OFF, 20);
constant constexpr bool ATTN_FOLD = ATTN_FOLD_ROT_DIM_SET;
constant constexpr uint ATTN_FOLD_DIM = ATTN_FOLD ? ATTN_HEAD_DIM : 1u;

// 21  ATTN_SINKS — gpt-oss attention sinks: the layer's per-head sink logits
//     (bound at the kernel family's sinks buffer) are an extra softmax column,
//     added UNSCALED (never multiplied by sm_scale — HF gpt_oss appends the raw
//     sink logit after the scaled qk) and dropped before ·V (the sink
//     contributes weight to the denominator only). Online softmax seeds the
//     row max with the sink so the running rescale accounts for it, and the
//     denominator gains exp(sink − rowmax). Unset: the column folds away and
//     the kernel is byte-identical (the sinks buffer is never dereferenced).
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_SINKS, 21);
constant bool ATTN_SINKS_ON = ATTN_SINKS_SET;

// 22  ATTN_ROW_QUERIES — the decode kernel's threadgroup rows. Unset/0: one
//     per sequence (a decode step). 1: one per token of a few-row step (a
//     speculative verify step's last token and drafts, a short chunk) — the
//     row's sequence from `cu_seqlens_q` (buffer 23), its keys its
//     sequence's up to its own position: each row attends as it would
//     decoding. A decode step's rows are its sequences, so it reads the same.
SCRATCHY_CONSTANT_OPTIONAL(uint, ATTN_ROW_QUERIES, 22);

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
//   10 the projection bias (ATTN_TQ_K_BIAS on K / ATTN_TQ_V_BIAS on V only)
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
    device const T*        bias         [[buffer(10)]],
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
    // The codes hold the operand minus its projection bias (turboquant_offset.h);
    // a cached key gets it back in the rotated domain. V's bias is one vector for
    // every key, so its image R·b_v is taken once; K's is rotated to each key's
    // position, so its image costs one transform per cached key.
    const uint off_mode = ATTN_TQ_KB ? 2u : (ATTN_TQ_VB ? 1u : 0u);
    device const T* b = bias + kv_head * head_dim;
    float rb[16];
    if (off_mode == 1u) {
        for (uint j = 0; j < qk_per_thread; ++j) {
            rb[j] = float(b[e0 + j]) * signs[e0 + j];
        }
        tq_wht(rb, qk_per_thread, simd_lid);
        for (uint j = 0; j < qk_per_thread; ++j) {
            rb[j] *= s;
        }
    }
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
        if (cached != (ATTN_TQ_STAGE_PASS == 2u)) {
            continue;
        }
        device T* row = block + t * head_dim + e0;
        float x[16];
        if (cached) {
            const uint store_row = (physical_block * block_size + t) * num_kv + kv_head;
            const float n = norms[store_row] * s;
            for (uint j = 0; j < qk_per_thread; ++j) {
                x[j] = centroids[tq_code(packed, store_row * pdim, e0 + j)] * n;
            }
            if (!span) {
                if (off_mode == 2u) {
                    for (uint j = 0; j < qk_per_thread; ++j) {
                        rb[j] = tq_offset<T>(2u, b, cos_sin, ATTN_ROT_DIM, ATTN_PAIR_OFF, i, false,
                                             e0 + j) * signs[e0 + j];
                    }
                    tq_wht(rb, qk_per_thread, simd_lid);
                    for (uint j = 0; j < qk_per_thread; ++j) {
                        rb[j] *= s;
                    }
                }
                for (uint j = 0; j < qk_per_thread; ++j) {
                    row[j] = T(off_mode != 0u ? x[j] + rb[j] : x[j]);
                }
                continue;
            }
            // x̃ = Rᵀ·x = s·D·H·x, to rope in the plain domain — where a span
            // key, stored unrotated, gets its bias back unrotated.
            tq_wht(x, qk_per_thread, simd_lid);
            for (uint j = 0; j < qk_per_thread; ++j) {
                x[j] *= s * signs[e0 + j];
                if (off_mode != 0u) {
                    x[j] += tq_offset<T>(off_mode, b, cos_sin, ATTN_ROT_DIM, ATTN_PAIR_OFF, i,
                                         true, e0 + j);
                }
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

#define INSTANTIATE_TQ_PREFILL(tag, T) \
  SCRATCHY_KERNEL(tq_stage_rotated_##tag, tq_stage_rotated<T>) \
  SCRATCHY_KERNEL(tq_rotate_rows_##tag, tq_rotate_rows<T, false>) \
  SCRATCHY_KERNEL(tq_unrotate_rows_##tag, tq_rotate_rows<T, true>)

INSTANTIATE_TQ_PREFILL(f16, half)
INSTANTIATE_TQ_PREFILL(bf16, bfloat)

// Merge a decode threadgroup's 32 simdgroup partials — each simdgroup's per-head running max,
// sum of exponentials and output accumulator — and write the heads' output rows. Run by the one-pass
// kernel on its own simdgroups' partials and by the combine pass on the split threadgroups' stored
// ones, so both merge the same values the same way.
template <typename T>
inline void attn_decode_merge(thread float* o_reg, thread float* max_score,
                              thread float* sum_exp_score, uint heads, uint simd_gid,
                              uint simd_lid, threadgroup float* tg_outputs,
                              threadgroup float* tg_max, threadgroup float* tg_sum,
                              device T* o_row, device const float* tq_signs,
                              device const T* vb)
{
    constexpr int BN = 32; // simdgroups per threadgroup
    constexpr int BD = 32; // lanes per simdgroup
    typedef float U;
    const uint head_dim = ATTN_HEAD_DIM;
    const uint qk_per_thread = head_dim / uint(BD);
    const uint tq_e = simd_lid * qk_per_thread;
    // ── Combine per-simdgroup partials ───────────────────────────
    //
    // Each simdgroup's lane 0 publishes its max + sum_exp per head; all
    // simdgroups then read all values via lane id and reduce.
    if (simd_lid == 0) {
        for (uint h = 0; h < heads; ++h) {
            tg_max[h * BN + simd_gid] = max_score[h];
            tg_sum[h * BN + simd_gid] = sum_exp_score[h];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint h = 0; h < heads; ++h) {
        // Each lane (within simdgroup_id 0..BN-1) reads tg_max[simd_lid]
        // / tg_sum[simd_lid]; simd_max + simd_sum produce the global max
        // and (factor-rescaled) global sum_exp.
        U other_max = tg_max[h * BN + simd_lid];
        U global_max = simd_max(other_max);
        U factor = metal::fast::exp(other_max - global_max);
        U global_sum = simd_sum(tg_sum[h * BN + simd_lid] * factor);

        // Combine output partials. Each simdgroup wrote o_reg[j] for
        // its slice; we need to weight each simdgroup's contribution by
        // its `factor` (the rescaling for the global max), then sum
        // across simdgroups, then divide by global_sum. ATTN_MERGE_SLICES
        // of the slices share each pair of barriers.
        for (uint j0 = 0; j0 < qk_per_thread; j0 += ATTN_MERGE_SLICES) {
            const uint nj = min(ATTN_MERGE_SLICES, qk_per_thread - j0);
            for (uint jj = 0; jj < nj; ++jj) {
                tg_outputs[(jj * BD + simd_lid) * ATTN_MERGE_STRIDE + simd_gid] = o_reg[h * qk_per_thread + j0 + jj];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint jj = 0; jj < nj; ++jj) {
                // Each simdgroup reads its column from tg_outputs and sums
                // across the BD partials, weighted by per-simdgroup factor.
                U val = tg_outputs[(jj * BD + simd_gid) * ATTN_MERGE_STRIDE + simd_lid] * factor;
                U combined = simd_sum(val);
                if (global_sum != 0) {
                    combined = combined / global_sum;
                }
                o_reg[h * qk_per_thread + j0 + jj] = combined;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // The combine transposes lane<->simdgroup, so simdgroup `simd_gid`
    // now owns the element set that lane `simd_gid` owned during the K
    // loop.
    if (ATTN_TQ != 0u) {
        // Codebook-domain output: gather it back into lane slices and
        // rotate once, o = s²·D·H·a — simdgroup `h` for head `h`.
        if (simd_lid == 0) {
            for (uint h = 0; h < heads; ++h) {
                for (uint j = 0; j < qk_per_thread; ++j) {
                    tg_outputs[h * head_dim + simd_gid * qk_per_thread + j] =
                        o_reg[h * qk_per_thread + j];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_gid < heads) {
            U o_loc[16];
            for (uint j = 0; j < qk_per_thread; ++j) {
                o_loc[j] = tg_outputs[simd_gid * head_dim + tq_e + j];
            }
            tq_wht(o_loc, qk_per_thread, simd_lid);
            for (uint j = 0; j < qk_per_thread; ++j) {
                const U o = o_loc[j] * tq_signs[tq_e + j] / U(head_dim);
                o_row[simd_gid * head_dim + tq_e + j] = T(ATTN_TQ_VB ? o + U(vb[tq_e + j]) : o);
            }
        }
    } else if (simd_lid == 0) {
        // Lane 0 of each simdgroup writes its qk_per_thread output slice.
        for (uint j = 0; j < qk_per_thread; ++j) {
            o_row[attn_elem_off(simd_gid, j, qk_per_thread, head_dim)] = T(o_reg[j]);
        }
    }
}

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
//   buffer(14/15) (ATTN_TQ_K_BIAS / ATTN_TQ_V_BIAS) = K / V projection bias
//                [num_kv_heads * head_dim].
//   buffer(16) (ATTN_SINKS) = the layer's sink logits [num_q_heads], the
//                model dtype (gpt-oss ships bf16). Free between the TQ biases
//                (14/15) and the fold buffers (17..22) so a TurboQuant
//                dispatch carrying sinks can't collide with the TQ slots
//                7..=13.
//   buffer(23)   (ATTN_ROW_QUERIES only) = cu_seqlens_q [num_seqs + 1].
//
// Dispatch: threadgroups (batch — ATTN_ROW_QUERIES: tokens — , num_q_heads, 1), threads (1024, 1, 1)
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
    device       T* q              [[buffer(1)]],
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
    device       uint*  tq_packed_k  [[buffer(7)]],
    device       uint*  tq_packed_v  [[buffer(8)]],
    device       float* tq_norms_k   [[buffer(9)]],
    device       float* tq_norms_v   [[buffer(10)]],
    device const float* tq_signs     [[buffer(11)]],
    device const float* tq_centroids [[buffer(12)]],
    device const uint*  slot_mapping [[buffer(13)]],
    device const T*     tq_k_bias    [[buffer(14)]],
    device const T*     tq_v_bias    [[buffer(15)]],
    // gpt-oss attention sinks (ATTN_SINKS): the layer's [num_q_heads] sink
    // logits, model dtype. Only dereferenced when ATTN_SINKS; unbound
    // otherwise (the cos_sin slot-6 pattern).
    device const T*     sinks        [[buffer(16)]],
    device const T*     fold_k        [[buffer(17)]],
    device const T*     fold_v        [[buffer(18)]],
    device const uint*  positions     [[buffer(19)]],
    device const float* tq_boundaries [[buffer(20)]],
    device const T*     fold_cos_sin  [[buffer(21)]],
    device const T*     tq_bias_cos_sin [[buffer(22)]],
    device const uint*  cu_seqlens_q  [[buffer(23)]],
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
    // Query heads this threadgroup serves (ATTN_TQ_HEADS): `heads`
    // consecutive heads of one KV head; per-head state is indexed
    // `[h * qk_per_thread + j]`.
    const uint heads = (ATTN_TQ != 0u) ? ATTN_TQ_HEADS : 1u;

    const uint row         = tg_pos.x;            // query row (ATTN_ROW_QUERIES: a token)
    const uint q_head_idx  = tg_pos.y * heads;    // first of `heads` query heads
    const uint group_ratio = num_q / num_kv;
    const uint kv_head_idx = q_head_idx / group_ratio;
    uint seq_idx = row;
    uint kv_len  = seq_used_k[row];
    if (ATTN_ROW_QUERIES != 0u) {
        seq_idx = 0;
        while (cu_seqlens_q[seq_idx + 1] <= row) {
            ++seq_idx;
        }
        kv_len = seq_used_k[seq_idx] - (cu_seqlens_q[seq_idx + 1] - 1u - row);
    }

    const uint kv_blk_stride  = num_kv * block_size * head_dim;
    const uint kv_head_stride = block_size * head_dim;
    const uint kv_tok_stride  = head_dim;

    thread U q_reg[16];                 // qk_per_thread <= 16 (head_dim <= 512)
    thread U o_reg[32];                 // [h * qk_per_thread + j], <= 32 (ATTN_TQ_HEADS)

    // Threadgroup scratch for per-simdgroup max + sum_exp combine; first, under the TurboQuant
    // fold, the KV row encode's scratch and codes.
    constexpr uint MERGE_FLOATS = ATTN_MERGE_SLICES * uint(BN) * ATTN_MERGE_STRIDE;
    constexpr uint ENCODE_FLOATS =
        ATTN_TQ_BITS_SET && ATTN_FOLD ? tq_encode_floats(2u, ATTN_HEAD_DIM) + 2u * ATTN_HEAD_DIM : 0u;
    threadgroup U tg_outputs[MERGE_FLOATS > ENCODE_FLOATS ? MERGE_FLOATS : ENCODE_FLOATS];
    threadgroup U tg_max[BN * 8];       // [head][simdgroup], heads <= 8
    threadgroup U tg_sum[BN * 8];

    device const T*    q_row = q + (row * num_q + q_head_idx) * head_dim;
    device       T*    o_row = output + (row * num_q + q_head_idx) * head_dim;
    device const uint* row_block_table = block_table + seq_idx * max_blocks;

    // Fold: the step's KV writer, run here (`ATTN_FOLD`). The threadgroup ropes its query heads in
    // place and builds its KV head's new row in `fold_kv` (K, then V), each element rounded to T as
    // the writer rounds it, K left unrotated where the writer leaves it (a span block). The
    // threadgroup owning the KV head, its first query head's, writes the row to the cache and
    // encodes it into the packed store; no other threadgroup reads that slot this step.
    threadgroup T fold_kv[2u * ATTN_FOLD_DIM];
    const uint fold_slot = ATTN_FOLD ? slot_mapping[seq_idx] : 0xFFFFFFFFu;
    if (ATTN_FOLD) {
        const uint tid = simd_gid * uint(BD) + simd_lid;
        const uint half_rot = ATTN_FOLD_ROT_DIM / 2u;
        const uint pair_off = ATTN_FOLD_PAIR_OFF;
        const uint pos = positions[seq_idx];
        device const T* cos_row = fold_cos_sin + pos * ATTN_FOLD_ROT_DIM;
        device const T* sin_row = cos_row + half_rot;
        device T* q_own = q + (seq_idx * num_q + q_head_idx) * head_dim;
        for (uint p = tid; p < heads * half_rot; p += uint(BN * BD)) {
            device T* qh = q_own + (p / half_rot) * head_dim;
            const uint d = p % half_rot;
            const float2 r = rope_rotate(float(qh[d]), float(qh[pair_off + d]), float(cos_row[d]),
                                         float(sin_row[d]));
            qh[d] = T(r.x);
            qh[pair_off + d] = T(r.y);
        }
        threadgroup T* fk = fold_kv;
        threadgroup T* fv = fold_kv + ATTN_FOLD_DIM;
        const uint row = (seq_idx * num_kv + kv_head_idx) * head_dim;
        if (tid < head_dim) {
            fk[tid] = fold_k[row + tid];
            fv[tid] = fold_v[row + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const bool skip_k_rot = (ATTN_ROR != 0u) && ((fold_slot & 0x80000000u) != 0u);
        if (tid < half_rot && !skip_k_rot) {
            const float2 r = rope_rotate(float(fk[tid]), float(fk[pair_off + tid]),
                                         float(cos_row[tid]), float(sin_row[tid]));
            fk[tid] = T(r.x);
            fk[pair_off + tid] = T(r.y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        if (q_head_idx % group_ratio == 0u && fold_slot != 0xFFFFFFFFu) {
            const uint phys = (ATTN_ROR != 0u) ? (fold_slot & 0x7FFFFFFFu) : fold_slot;
            const uint block = phys / block_size;
            const uint chunk = (ATTN_BLOCKS_PER_CHUNK == 0u) ? 0u : block / ATTN_BLOCKS_PER_CHUNK;
            const uint blk_in_chunk =
                (ATTN_BLOCKS_PER_CHUNK == 0u) ? block : block % ATTN_BLOCKS_PER_CHUNK;
            const uint at = blk_in_chunk * kv_blk_stride + kv_head_idx * kv_head_stride
                + (phys % block_size) * kv_tok_stride;
            if (tid < head_dim) {
                ((device T*)k_cache[chunk])[at + tid] = fk[tid];
                ((device T*)v_cache[chunk])[at + tid] = fv[tid];
            }
            if (ATTN_TQ != 0u) {
                // The encode's scratch is the merge's, free until the key loop is done.
                const uint vpw = 32u / ATTN_TQ;
                threadgroup float* scratch = tg_outputs;
                threadgroup uint* codes = (threadgroup uint*)(tg_outputs + tq_encode_floats(2u, head_dim));
                const bool live = tid < head_dim;
                kv_row_encode<T>(live ? float(fk[tid]) : 0.0f, live ? float(fv[tid]) : 0.0f, tid,
                                 head_dim, num_kv, kv_head_idx, fold_slot, pos, ATTN_TQ, vpw,
                                 (head_dim + vpw - 1u) / vpw, ATTN_TQ_KB ? 2u : 0u,
                                 ATTN_TQ_VB ? 1u : 0u, ATTN_FOLD_ROT_DIM, pair_off, tq_signs,
                                 tq_boundaries, tq_packed_k, tq_norms_k, tq_packed_v, tq_norms_v,
                                 tq_k_bias, tq_v_bias, tq_bias_cos_sin, scratch, codes);
            }
        }
    }

    // Pre-multiply Q by scale (MLX `sdpa_vector`: `q[i] = scale * queries[i]`).
    // Element ownership follows attn_elem_off (contiguous, or co-resident
    // NeoX pairs under ATTN_PAIR_CORESIDENT) — Q must match K's per-lane set.
    // Under TurboQuant only the rare plain-domain keys (the tail, span blocks)
    // use it, and they read it per head from q_row instead (`q_plain`).
    if (ATTN_TQ == 0u) {
        for (uint i = 0; i < qk_per_thread; ++i) {
            q_reg[i] = U(scale) * U(q_row[attn_elem_off(simd_lid, i, qk_per_thread, head_dim)]);
        }
    }
    for (uint i = 0; i < heads * qk_per_thread; ++i) {
        o_reg[i] = 0;
    }
    auto q_plain = [&](uint h, uint j) -> U {
        return ATTN_TQ == 0u
            ? q_reg[j]
            : U(scale) * U(q_row[h * head_dim + attn_elem_off(simd_lid, j, qk_per_thread, head_dim)]);
    };

    // TurboQuant: q rotated into the codebook domain, `s²·H·D·q`. The
    // codebook domain is laid out contiguously — lane `l` owns elements
    // `l*qk_per_thread + j` (`tq_e`) whatever the q/K layout, as H·D mixes
    // every element anyway — so a lane's codes sit in adjacent packed words.
    // Every simdgroup owns the same slices and rotates its own copy in
    // registers. The codebook (<= 16 centroids) is staged once.
    const uint tq_e = simd_lid * qk_per_thread;
    thread U qt_reg[32];
    threadgroup U tq_lut[16];
    if (ATTN_TQ != 0u) {
        for (uint h = 0; h < heads; ++h) {
            thread U* qt = qt_reg + h * qk_per_thread;
            for (uint j = 0; j < qk_per_thread; ++j) {
                qt[j] = U(scale) * U(q_row[h * head_dim + tq_e + j]) * tq_signs[tq_e + j];
            }
            tq_wht(qt, qk_per_thread, simd_lid);
            for (uint j = 0; j < qk_per_thread; ++j) {
                qt[j] /= U(head_dim);
            }
        }
        const uint tid = simd_gid * uint(BD) + simd_lid;
        if (tid < (1u << ATTN_TQ)) {
            tq_lut[tid] = tq_centroids[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint tq_pdim = (ATTN_TQ == 0u) ? 0u : (head_dim + 32u / ATTN_TQ - 1u) / (32u / ATTN_TQ);

    // TurboQuant K bias: q·R_i·b = Σ_d (kb_a[d]·cos_{i,d} + kb_b[d]·sin_{i,d}) + kb_c
    // over the NeoX pairs (d, d + ATTN_PAIR_OFF), d < ATTN_ROT_DIM/2, plus the
    // unrotated rest (kb_c). Per query; lane l owns the `kb_np` adjacent pairs
    // from d = l·kb_np, so each key's cos/sin reads are contiguous per lane.
    device const T* kb = tq_k_bias + kv_head_idx * head_dim;
    device const T* vb = tq_v_bias + kv_head_idx * head_dim;
    const uint kb_np = (ATTN_ROT_DIM / 2u + 31u) / 32u;
    thread U kb_a[16];                  // [h * kb_np + j]
    thread U kb_b[16];
    thread U kb_c[8];
    if (ATTN_TQ_KB) {
        const uint half_rot = ATTN_ROT_DIM / 2u;
        for (uint h = 0; h < heads; ++h) {
            device const T* qh = q_row + h * head_dim;
            for (uint j = 0; j < kb_np; ++j) {
                const uint d = simd_lid * kb_np + j;
                kb_a[h * kb_np + j] = 0;
                kb_b[h * kb_np + j] = 0;
                if (d < half_rot) {
                    const U q0 = U(scale) * U(qh[d]);
                    const U q1 = U(scale) * U(qh[d + ATTN_PAIR_OFF]);
                    const U b0 = U(kb[d]);
                    const U b1 = U(kb[d + ATTN_PAIR_OFF]);
                    kb_a[h * kb_np + j] = q0 * b0 + q1 * b1;
                    kb_b[h * kb_np + j] = q1 * b0 - q0 * b1;
                }
            }
            U c = 0;
            for (uint e = simd_lid; e < head_dim; e += 32u) {
                if (e >= half_rot && (e < ATTN_PAIR_OFF || e >= ATTN_PAIR_OFF + half_rot)) {
                    c += U(scale) * U(qh[e]) * U(kb[e]);
                }
            }
            kb_c[h] = simd_sum(c);
        }
    }

    // Initialize per-thread max with finite minimum (MLX uses
    // `Limits<U>::finite_min`; -FLT_MAX is the f32 equivalent).
    // fast::exp doesn't handle -INFINITY safely so we avoid it.
    // Attention sinks: every simdgroup seeds its max with the UNSCALED sink
    // logit instead, so the merge's global max is ≥ the sink and the
    // denominator term simdgroup 0 adds below can't overflow.
    U max_score[8];
    U sum_exp_score[8];
    for (uint h = 0; h < heads; ++h) {
        max_score[h] = ATTN_SINKS_ON ? U(sinks[q_head_idx + h]) : -FLT_MAX;
        sum_exp_score[h] = 0;
    }

    // TurboQuant: the key this step appended (the query's own, at kv_len-1)
    // is not in the packed store yet — uniform arches quantize after
    // attention, hybrid ones only lossily before it — so it is read from the
    // cache its writer just filled, exactly as the dequant path read it. A
    // reused span block's slot is the write-skip sentinel: nothing was
    // written, and that key lives only in the packed store.
    const bool tail_in_cache = (ATTN_TQ == 0u) || (slot_mapping[row] != 0xFFFFFFFFu);
    // For each key, simdgroup `vsg` of the 32 handles tokens at indices
    // vsg, vsg+BN, vsg+2*BN, ... The simdgroup that overshoots `kv_len`
    // skips its iteration and contributes 0. A sliding window (decode Q at
    // absolute position kv_len-1) starts each simdgroup at its first key inside
    // the window: the same keys in the same order as walking every key and
    // skipping the older ones, without the walk.
    const uint vsg = simd_gid;
    const uint window_lo =
        (ATTN_WINDOW > 0 && kv_len > uint(ATTN_WINDOW)) ? kv_len - uint(ATTN_WINDOW) : 0u;
    const uint first_key =
        vsg + (window_lo > vsg ? (window_lo - vsg + uint(BN) - 1u) / uint(BN) * uint(BN) : 0u);
    for (uint i = first_key; i < kv_len; i += uint(BN)) {
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
        // Fold: the step's own key, from the row this threadgroup built.
        const bool fold_tail = ATTN_FOLD && fold_slot != 0xFFFFFFFFu && i + 1u == kv_len;
        const uint tq_row =
            ((bt_raw & 0x7FFFFFFFu) * block_size + token_in_block) * num_kv + kv_head_idx;

        U score[8];                     // this lane's share of each head's q·k
        for (uint h = 0; h < heads; ++h) {
            score[h] = 0;
        }
        U k_scale = 1;
        U k_off[8];                     // this lane's share of q·R_i·b (TurboQuant K bias)
        const bool k_biased = ATTN_TQ_KB && packed && !do_rot;
        device const T* v_ptr = nullptr;
        if (packed) {
            const uint k_word = tq_row * tq_pdim;
            if (do_rot) {
                // Span block: the codes hold UNROTATED K and RoPE does not
                // commute with H·D, so decode this key the way the dequant
                // pass did (rounded to T, its unrotated bias restored) and
                // re-rope it in the plain domain.
                U k_loc[16];
                for (uint j = 0; j < qk_per_thread; ++j) {
                    k_loc[j] = tq_lut[tq_code(tq_packed_k, k_word,
                                              attn_elem_off(simd_lid, j, qk_per_thread, head_dim))];
                }
                tq_wht(k_loc, qk_per_thread, simd_lid);
                const U k_norm = tq_norms_k[tq_row] / U(head_dim);
                for (uint j = 0; j < qk_per_thread; ++j) {
                    const uint e = attn_elem_off(simd_lid, j, qk_per_thread, head_dim);
                    const U x = k_loc[j] * tq_signs[e] * k_norm;
                    k_loc[j] = U(T(ATTN_TQ_KB ? x + U(kb[e]) : x));
                }
                attn_rope_on_read<T>(k_loc, qk_per_thread, simd_lid, i, cos_sin);
                for (uint h = 0; h < heads; ++h) {
                    for (uint j = 0; j < qk_per_thread; ++j) {
                        score[h] += q_plain(h, j) * k_loc[j];
                    }
                }
            } else {
                U k_code[16];
                for (uint j = 0; j < qk_per_thread; ++j) {
                    k_code[j] = tq_lut[tq_code(tq_packed_k, k_word, tq_e + j)];
                }
                for (uint h = 0; h < heads; ++h) {
                    for (uint j = 0; j < qk_per_thread; ++j) {
                        score[h] += qt_reg[h * qk_per_thread + j] * k_code[j];
                    }
                }
                k_scale = tq_norms_k[tq_row];
                if (k_biased) {
                    const uint d0 = simd_lid * kb_np;
                    device const T* cos_row = cos_sin + i * ATTN_ROT_DIM + d0;
                    device const T* sin_row = cos_row + ATTN_ROT_DIM / 2u;
                    for (uint h = 0; h < heads; ++h) {
                        k_off[h] = 0;
                    }
                    for (uint j = 0; j < kb_np; ++j) {
                        if (d0 + j < ATTN_ROT_DIM / 2u) {
                            const U c = U(cos_row[j]);
                            const U s = U(sin_row[j]);
                            for (uint h = 0; h < heads; ++h) {
                                k_off[h] += kb_a[h * kb_np + j] * c + kb_b[h * kb_np + j] * s;
                            }
                        }
                    }
                }
            }
        } else {
            // Chunked KV: deref the chunk backing this physical block.
            // ATTN_BLOCKS_PER_CHUNK is a baked constant. When it is 0 the
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
            U k_loc[16];
            for (uint j = 0; j < qk_per_thread; ++j) {
                const uint e = attn_elem_off(simd_lid, j, qk_per_thread, head_dim);
                k_loc[j] = U(fold_tail ? fold_kv[e] : k_ptr[e]);
            }
            if (do_rot) {
                attn_rope_on_read<T>(k_loc, qk_per_thread, simd_lid, i, cos_sin);
            }
            for (uint h = 0; h < heads; ++h) {
                for (uint j = 0; j < qk_per_thread; ++j) {
                    score[h] += q_plain(h, j) * k_loc[j];
                }
            }
        }

        // Online softmax update per head. Match MLX `sdpa_vector`:
        // fast::exp for both factor + exp_score. k_scale is
        // simdgroup-uniform, so a K bias's term joins the one reduction;
        // without one this is exactly the plain score.
        U factor[8];
        U exp_score[8];
        for (uint h = 0; h < heads; ++h) {
            const U s = k_biased ? simd_sum(score[h] * k_scale + k_off[h]) + kb_c[h]
                                 : simd_sum(score[h]) * k_scale;
            const U new_max = max(max_score[h], s);
            factor[h] = metal::fast::exp(max_score[h] - new_max);
            exp_score[h] = metal::fast::exp(s - new_max);
            max_score[h] = new_max;
            sum_exp_score[h] = sum_exp_score[h] * factor[h] + exp_score[h];
        }

        // Accumulate weighted V; rescale prior accumulator with factor.
        // V element ownership matches K/Q (attn_elem_off). Under TurboQuant
        // the accumulator lives in the codebook domain.
        U v_loc[16];
        if (packed) {
            const uint v_word = tq_row * tq_pdim;
            for (uint j = 0; j < qk_per_thread; ++j) {
                v_loc[j] = tq_lut[tq_code(tq_packed_v, v_word, tq_e + j)];
            }
            const U v_norm = tq_norms_v[tq_row];
            for (uint h = 0; h < heads; ++h) {
                exp_score[h] *= v_norm;
            }
        } else if (ATTN_TQ != 0u) {
            // The tail's plain V into the codebook domain: s²·D·H·(H·D·v) = v.
            // Centered like the packed codes; the bias returns at the output.
            for (uint j = 0; j < qk_per_thread; ++j) {
                const U v = U(fold_tail ? fold_kv[ATTN_FOLD_DIM + tq_e + j] : v_ptr[tq_e + j]);
                v_loc[j] = (ATTN_TQ_VB ? v - U(vb[tq_e + j]) : v) * tq_signs[tq_e + j];
            }
            tq_wht(v_loc, qk_per_thread, simd_lid);
        } else {
            for (uint j = 0; j < qk_per_thread; ++j) {
                const uint e = attn_elem_off(simd_lid, j, qk_per_thread, head_dim);
                v_loc[j] = U(fold_tail ? fold_kv[ATTN_FOLD_DIM + e] : v_ptr[e]);
            }
        }
        for (uint h = 0; h < heads; ++h) {
            for (uint j = 0; j < qk_per_thread; ++j) {
                o_reg[h * qk_per_thread + j] =
                    o_reg[h * qk_per_thread + j] * factor[h] + exp_score[h] * v_loc[j];
            }
        }
    }

    // Attention sinks: the denominator gains exp(sink − rowmax) — the extra
    // softmax column, UNSCALED by sm_scale, contributing no V. Every
    // simdgroup's max was seeded with the sink, so the argument is ≤ 0;
    // simdgroup 0 adds it once so the merged denominator carries exactly one
    // sink column (all lanes of a simdgroup hold identical partials; the
    // merge reads lane 0's). The column's value is ZERO, and under TurboQuant
    // the merge epilogue adds vb to the whole output — so the sink must
    // cancel its own share: its "value" enters the codebook accumulator as
    // −vb (centered like the tail's plain V above), and the epilogue's +vb
    // then restores exactly the keys' weight sum. Without this a sink that
    // dominates the softmax (a biased-KV layer's scaled scores can sit far
    // below an unscaled sink) hands the output a full vb vector.
    if (ATTN_SINKS_ON && simd_gid == 0) {
        for (uint h = 0; h < heads; ++h) {
            const U w = metal::fast::exp(U(sinks[q_head_idx + h]) - max_score[h]);
            sum_exp_score[h] += w;
            if (ATTN_TQ != 0u && ATTN_TQ_VB) {
                U v_loc[16];
                for (uint j = 0; j < qk_per_thread; ++j) {
                    v_loc[j] = -U(vb[tq_e + j]) * tq_signs[tq_e + j];
                }
                tq_wht(v_loc, qk_per_thread, simd_lid);
                for (uint j = 0; j < qk_per_thread; ++j) {
                    o_reg[h * qk_per_thread + j] += w * v_loc[j];
                }
            }
        }
    }

    attn_decode_merge<T>(o_reg, max_score, sum_exp_score, heads, simd_gid, simd_lid, tg_outputs,
                         tg_max, tg_sum, o_row, tq_signs, vb);
}

SCRATCHY_KERNEL(attention_via_cache_v2_f16_specialized, attention_via_cache_v2<half>)
SCRATCHY_KERNEL(attention_via_cache_v2_bf16_specialized, attention_via_cache_v2<bfloat>)

// attention_via_cache_v2_combine_{f16,bf16}_specialized — `attention_decode_gqa_tq`'s second pass: a
// threadgroup per query head merges the partials its KV head's ATTN_SPLITS threadgroups stored
// (buffer 16, `[batch, num_q_heads, ATTN_SPLITS, 2 + head_dim]`: max, sum, unnormalized output in
// the codebook domain) and writes the head's output as `attention_via_cache_v2`'s merge does.
// Its ATTN_COMBINE_GROUPS simdgroups each sum their share of every lane's elements over the
// splits; the first divides and rotates them. Dispatch: threadgroups (batch, num_q_heads, 1), threads
// (32 · ATTN_COMBINE_GROUPS, 1, 1).
constant constexpr uint ATTN_COMBINE_GROUPS = 4u;
template <typename T>
[[kernel, max_total_threads_per_threadgroup(32 * ATTN_COMBINE_GROUPS)]] void
attention_via_cache_v2_combine(
    device       T*     output        [[buffer(0)]],
    device const float* tq_signs      [[buffer(11)]],
    device const T*     tq_v_bias     [[buffer(15)]],
    device const float* attn_partials [[buffer(16)]],
    uint3  tg_pos    [[threadgroup_position_in_grid]],
    uint   simd_gid  [[simdgroup_index_in_threadgroup]],
    uint   simd_lid  [[thread_index_in_simdgroup]])
{
    typedef float U;
    const uint head_dim = ATTN_HEAD_DIM;
    const uint num_q = ATTN_NUM_Q_HEADS;
    const uint qk_per_thread = head_dim / 32u;
    const uint seq_idx = tg_pos.x;
    const uint q_head_idx = tg_pos.y;
    const uint kv_head_idx = q_head_idx / (num_q / ATTN_NUM_KV_HEADS);
    const uint row = head_dim + 2u;
    device const float* p = attn_partials + (seq_idx * num_q + q_head_idx) * ATTN_SPLITS * row;
    U global_max = -FLT_MAX;
    for (uint s = 0; s < ATTN_SPLITS; ++s) {
        global_max = max(global_max, U(p[s * row]));
    }
    // Lane l the elements l·qk_per_thread + j: the codebook domain's lane slices under TurboQuant;
    // this simdgroup QG of them from `j0`, through `sums` (a lane's row padded by one).
    // Another kernel's bake may set a head dim this one cannot take: its arrays need a size all
    // the same (the assert refuses baking this kernel with one).
    constexpr bool D_OK = ATTN_HEAD_DIM % (32u * ATTN_COMBINE_GROUPS) == 0u && ATTN_HEAD_DIM != 0u;
    static_assert(sizeof(T) && D_OK, "head_dim a multiple of 32 lanes x the combine's groups");
    constexpr uint QG = D_OK ? ATTN_HEAD_DIM / 32u / ATTN_COMBINE_GROUPS : 1u;
    threadgroup U sums[32u * (QG * ATTN_COMBINE_GROUPS + 1u)];
    const uint e0 = simd_lid * qk_per_thread;
    const uint j0 = simd_gid * QG;
    U global_sum = 0;
    U part[QG];
    for (uint j = 0; j < QG; ++j) {
        part[j] = 0;
    }
    for (uint s = 0; s < ATTN_SPLITS; ++s) {
        const U factor = metal::fast::exp(U(p[s * row]) - global_max);
        global_sum += U(p[s * row + 1u]) * factor;
        for (uint j = 0; j < QG; ++j) {
            part[j] += U(p[s * row + 2u + e0 + j0 + j]) * factor;
        }
    }
    for (uint j = 0; j < QG; ++j) {
        sums[simd_lid * (qk_per_thread + 1u) + j0 + j] = part[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid != 0u) {
        return;
    }
    U o_loc[16];
    for (uint j = 0; j < qk_per_thread; ++j) {
        o_loc[j] = sums[simd_lid * (qk_per_thread + 1u) + j];
    }
    for (uint j = 0; j < qk_per_thread; ++j) {
        o_loc[j] = global_sum != 0 ? o_loc[j] / global_sum : o_loc[j];
    }
    device T* o_row = output + (seq_idx * num_q + q_head_idx) * head_dim;
    if (ATTN_TQ != 0u) {
        // o = s²·D·H·a, as the one-pass merge rotates it.
        device const T* vb = tq_v_bias + kv_head_idx * head_dim;
        tq_wht(o_loc, qk_per_thread, simd_lid);
        for (uint j = 0; j < qk_per_thread; ++j) {
            const U o = o_loc[j] * tq_signs[e0 + j] / U(head_dim);
            o_row[e0 + j] = T(ATTN_TQ_VB ? o + U(vb[e0 + j]) : o);
        }
    } else {
        for (uint j = 0; j < qk_per_thread; ++j) {
            o_row[e0 + j] = T(o_loc[j]);
        }
    }
}

SCRATCHY_KERNEL(attention_via_cache_v2_combine_f16_specialized, attention_via_cache_v2_combine<half>)
SCRATCHY_KERNEL(attention_via_cache_v2_combine_bf16_specialized, attention_via_cache_v2_combine<bfloat>)

// attention_decode_gqa_tq_{f16,bf16}_specialized — the TurboQuant decode attention of a KV head's
// 8 query heads together (`ATTN_NUM_Q_HEADS == 8 · ATTN_NUM_KV_HEADS`, 4-bit codes, no projection
// bias): each key's codes decode once for all 8, straight into 8x8 simdgroup-matrix fragments (MLX
// `BaseMMAFrag<T, 8, 8>`: lane `l` holds row fm = (l/4 & 4) + (l/2 % 4) and columns
// fn = (l/4 & 2)·2 + (l % 2)·2, fn + 1; the lanes sharing a row differ in bits 0 and 3), so
// S = Q·Kᵀ and O += P·V run on the matrix units a block of 8 keys at a time. A key group's
// simdgroups own a slice of the head dims each, their S partials meeting in threadgroup memory
// once a block; the threadgroup's key groups take interleaved blocks and merge at the end. A span
// block's keys (K stored unrotated) decode to the plain domain and re-rope, a simdgroup a key or
// two, into the same S. Threadgroup z of ATTN_SPLITS takes a contiguous run of the sequence's packed
// keys' blocks and stores its heads' partials for `attention_via_cache_v2_combine` (one head a
// threadgroup); the last also takes the step's own key when its writer left it in the cache, or
// under the fold (`ATTN_FOLD`) built it here, exactly as `attention_via_cache_v2` does. Under the
// fold each split ropes the query heads, never writing the query back, the last builds the KV row
// in threadgroup memory, and one threadgroup more (z = ATTN_SPLITS) builds it, writes it to the
// cache and encodes it. Bindings: `attention_via_cache_v2`'s, buffer 0 unread. Dispatch:
// threadgroups (batch, num_kv_heads, ATTN_SPLITS + the fold's writer), threads (head_dim, 1, 1).
template <typename T>
[[kernel, max_total_threads_per_threadgroup(ATTN_HEAD_DIM)]] void attention_decode_gqa_tq(
    device const T*     q             [[buffer(1)]],
    device const uint*  seq_used_k    [[buffer(2)]],
    device const uint*  block_table   [[buffer(3)]],
    device const uint64_t* k_cache    [[buffer(4)]],
    device const uint64_t* v_cache    [[buffer(5)]],
    device const T*     cos_sin       [[buffer(6)]],
    device       uint*  tq_packed_k   [[buffer(7)]],
    device       uint*  tq_packed_v   [[buffer(8)]],
    device       float* tq_norms_k    [[buffer(9)]],
    device       float* tq_norms_v    [[buffer(10)]],
    device const float* tq_signs      [[buffer(11)]],
    device const float* tq_centroids  [[buffer(12)]],
    device const uint*  slot_mapping  [[buffer(13)]],
    device const T*     tq_k_bias     [[buffer(14)]],
    device const T*     tq_v_bias     [[buffer(15)]],
    device       float* attn_partials [[buffer(16)]],
    device const T*     fold_k        [[buffer(17)]],
    device const T*     fold_v        [[buffer(18)]],
    device const uint*  positions     [[buffer(19)]],
    device const float* tq_boundaries [[buffer(20)]],
    device const T*     fold_cos_sin  [[buffer(21)]],
    device const T*     tq_bias_cos_sin [[buffer(22)]],
    uint3  tg_pos    [[threadgroup_position_in_grid]],
    uint   simd_gid  [[simdgroup_index_in_threadgroup]],
    uint   simd_lid  [[thread_index_in_simdgroup]])
{
    constexpr uint G = 8u;
    constexpr uint KB = 8u;
    // Another kernel's bake may set a head dim this one cannot take: its arrays need a size all
    // the same (the asserts below refuse baking this kernel with one).
    constexpr bool D_OK = ATTN_HEAD_DIM % 128u == 0u && ATTN_HEAD_DIM - 1u < 512u;
    constexpr uint D = D_OK ? ATTN_HEAD_DIM : 128u;
    constexpr uint NSG = D / 32u;
    // A key group's simdgroups, each a slice of the head dims (8 at head_dim 512: 8 Q and 8 O
    // fragments a lane), and its keys of a span block.
    constexpr uint SG = NSG >= 16u ? 8u : 4u;
    constexpr uint KG = NSG / SG;
    constexpr uint KPS = KB / SG;
    constexpr uint DS = D / SG;
    constexpr uint TILES = DS / 8u;
    constexpr uint QK = D / 32u;
    // `qb`'s row stride: a head's row padded by 4 floats, so the 8 rows of a fragment's load and
    // the merge's stores fall in different banks.
    constexpr uint QS = D + 4u;
    constexpr uint XS_S = KG * 2u * SG * 64u;
    constexpr uint XS_ENC = tq_encode_floats(2u, D) + 2u * D;
    constexpr uint XS = XS_S > XS_ENC ? XS_S : XS_ENC;
    // Checked where this kernel is baked (`sizeof(T)` defers them to its instantiation).
    static_assert(sizeof(T) && ATTN_TQ == 4u, "a packed word holds one 8-wide tile row");
    static_assert(sizeof(T) && ATTN_NUM_Q_HEADS == G * ATTN_NUM_KV_HEADS, "8 query heads a KV head");
    static_assert(sizeof(T) && D_OK, "head_dim a multiple of 128, at most 512");
    static_assert(sizeof(T) && ATTN_BLOCK_SIZE % KB == 0u, "a key block in one KV block");
    static_assert(sizeof(T) && !ATTN_TQ_KB && !ATTN_TQ_VB, "no projection bias");
    static_assert(sizeof(T) && ATTN_WINDOW <= 0, "full attention");
    static_assert(sizeof(T) && (ATTN_ROR == 0u || ATTN_PAIR_OFF % QK == 0u),
                  "a rotary pair's partner a whole lane stride away");
    const uint num_q = ATTN_NUM_Q_HEADS;
    const uint num_kv = ATTN_NUM_KV_HEADS;
    const uint bs = ATTN_BLOCK_SIZE;
    const uint pdim = D / 8u;
    const uint kv_blk_stride = num_kv * bs * D;
    const uint kv_head_stride = bs * D;
    const uint seq = tg_pos.x;
    const uint kvh = tg_pos.y;
    const uint z = tg_pos.z;
    const uint q0 = kvh * G;
    const uint kv_len = seq_used_k[seq];
    const uint tid = simd_gid * 32u + simd_lid;
    const uint kg = simd_gid / SG;
    const uint sg = simd_gid % SG;
    device const uint* row_bt = block_table + seq * ATTN_MAX_BLOCKS_PER_SEQ;
    const uint slot = slot_mapping[seq];
    // The step's own key is not in the packed store: its writer left it in the cache, or the fold
    // builds it here. A write-skip slot's key is packed like the rest.
    const bool own_key = slot != 0xFFFFFFFFu;
    const uint n_packed = kv_len - (own_key ? 1u : 0u);

    threadgroup float qb[G * QS];
    threadgroup float xs[XS];
    threadgroup T fold_kv[2u * ATTN_FOLD_DIM];
    threadgroup float lut[16];
    threadgroup float ml[2u * KG * G];
    threadgroup float own_s[G];
    if (tid < 16u) {
        lut[tid] = tq_centroids[tid];
    }

    // The query heads' element `e` as the writer leaves it: roped under the fold, rounded to T.
    const uint fold_pos = ATTN_FOLD ? positions[seq] : 0u;
    auto q_plain = [&](uint h, uint e) -> float {
        device const T* qh = q + (seq * num_q + q0 + h) * D;
        if (!ATTN_FOLD) {
            return float(qh[e]);
        }
        const uint half_rot = ATTN_FOLD_ROT_DIM / 2u;
        const uint off = ATTN_FOLD_PAIR_OFF;
        device const T* c = fold_cos_sin + fold_pos * ATTN_FOLD_ROT_DIM;
        if (e < half_rot) {
            return float(T(rope_rotate(float(qh[e]), float(qh[off + e]), float(c[e]),
                                       float(c[half_rot + e])).x));
        }
        if (e >= off && e < off + half_rot) {
            const uint d = e - off;
            return float(T(rope_rotate(float(qh[d]), float(qh[e]), float(c[d]),
                                       float(c[half_rot + d])).y));
        }
        return float(qh[e]);
    };
    // The fold's KV row in `fold_kv` (K, then V), K roped unless its block is a span's.
    auto build_row = [&]() {
        threadgroup T* fk = fold_kv;
        threadgroup T* fv = fold_kv + ATTN_FOLD_DIM;
        const uint row = (seq * num_kv + kvh) * D;
        fk[tid] = fold_k[row + tid];
        fv[tid] = fold_v[row + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint half_rot = ATTN_FOLD_ROT_DIM / 2u;
        const uint off = ATTN_FOLD_PAIR_OFF;
        const bool skip_k_rot = (ATTN_ROR != 0u) && ((slot & 0x80000000u) != 0u);
        if (tid < half_rot && !skip_k_rot) {
            device const T* c = fold_cos_sin + fold_pos * ATTN_FOLD_ROT_DIM;
            const float2 r = rope_rotate(float(fk[tid]), float(fk[off + tid]), float(c[tid]),
                                         float(c[half_rot + tid]));
            fk[tid] = T(r.x);
            fk[off + tid] = T(r.y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    };
    // Under the fold the threadgroup past the splits is the writer's: it builds the row, writes it
    // to the cache and encodes it into the packed store, as `attention_via_cache_v2`'s fold does,
    // off the splits' path.
    if (ATTN_FOLD && z == ATTN_SPLITS) {
        build_row();
        if (own_key) {
            const uint phys = (ATTN_ROR != 0u) ? (slot & 0x7FFFFFFFu) : slot;
            const uint block = phys / bs;
            const uint chunk = (ATTN_BLOCKS_PER_CHUNK == 0u) ? 0u : block / ATTN_BLOCKS_PER_CHUNK;
            const uint blk_in_chunk =
                (ATTN_BLOCKS_PER_CHUNK == 0u) ? block : block % ATTN_BLOCKS_PER_CHUNK;
            const uint at = blk_in_chunk * kv_blk_stride + kvh * kv_head_stride + (phys % bs) * D;
            ((device T*)k_cache[chunk])[at + tid] = fold_kv[tid];
            ((device T*)v_cache[chunk])[at + tid] = fold_kv[ATTN_FOLD_DIM + tid];
            const uint vpw = 32u / ATTN_TQ;
            threadgroup uint* codes = (threadgroup uint*)(xs + tq_encode_floats(2u, D));
            kv_row_encode<T>(float(fold_kv[tid]), float(fold_kv[ATTN_FOLD_DIM + tid]), tid, D,
                             num_kv, kvh, slot, fold_pos, ATTN_TQ, vpw, pdim, 0u, 0u,
                             ATTN_FOLD_ROT_DIM, ATTN_FOLD_PAIR_OFF, tq_signs, tq_boundaries,
                             tq_packed_k, tq_norms_k, tq_packed_v, tq_norms_v, tq_k_bias,
                             tq_v_bias, tq_bias_cos_sin, xs, codes);
        }
        return;
    }
    // q in the codebook domain, `s·H·D·q / D`, a simdgroup a head; its fragments; then the plain
    // query, scaled, for the keys the plain domain scores. With two simdgroups a head, each takes
    // half its dims (the index's top bit) and the transform's stages run in the same order: the
    // low bits in a lane, the lane bits by shuffles, the top bit across the pair through `qb`.
    if (NSG == 2u * G) {
        constexpr uint HQ = QK / 2u;
        const uint h = simd_gid % G;
        const uint hi = simd_gid / G;
        const uint own = h * QS + hi * (D / 2u) + simd_lid * HQ;
        float x[HQ];
        for (uint j = 0; j < HQ; ++j) {
            const uint e = hi * (D / 2u) + simd_lid * HQ + j;
            x[j] = ATTN_SCALE_FC * q_plain(h, e) * tq_signs[e];
        }
        tq_wht(x, HQ, simd_lid);
        for (uint j = 0; j < HQ; ++j) {
            qb[own + j] = x[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint pair = h * QS + (1u - hi) * (D / 2u) + simd_lid * HQ;
        for (uint j = 0; j < HQ; ++j) {
            const float o = qb[pair + j];
            x[j] = hi != 0u ? (o - x[j]) : (x[j] + o);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < HQ; ++j) {
            qb[own + j] = x[j] / float(D);
        }
    } else {
        for (uint h = simd_gid; h < G; h += NSG) {
            float x[QK];
            for (uint j = 0; j < QK; ++j) {
                const uint e = simd_lid * QK + j;
                x[j] = ATTN_SCALE_FC * q_plain(h, e) * tq_signs[e];
            }
            tq_wht(x, QK, simd_lid);
            for (uint j = 0; j < QK; ++j) {
                qb[h * QS + simd_lid * QK + j] = x[j] / float(D);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 Qf[TILES];
    for (uint t = 0; t < TILES; ++t) {
        simdgroup_load(Qf[t], qb + sg * DS + t * 8u, QS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < G * D; i += D) {
        qb[i / D * QS + i % D] = ATTN_SCALE_FC * q_plain(i / D, i % D);
    }

    // Fold: the last split builds the KV row in `fold_kv` (K, then V), K roped unless its block is
    // a span's, for the step's own key; the writer's threadgroup (above) built it too.
    if (ATTN_FOLD && z == ATTN_SPLITS - 1u) {
        build_row();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // A plain-domain K row in this lane's elements `lane·QK + j`, re-roped to key `i` when its
    // block holds K unrotated (the partner of element e sits ATTN_PAIR_OFF / QK lanes away).
    auto rope_row = [&](thread float* x, uint i) {
        const uint half_rot = ATTN_ROT_DIM / 2u;
        const uint sh = ATTN_PAIR_OFF / QK;
        const bool first = simd_lid * QK < ATTN_PAIR_OFF;
        const ushort src = ushort(first ? simd_lid + sh : simd_lid - sh);
        device const T* c = cos_sin + i * ATTN_ROT_DIM;
        for (uint j = 0; j < QK; ++j) {
            const uint e = simd_lid * QK + j;
            const float p = simd_shuffle(x[j], src);
            if (e < half_rot) {
                x[j] = rope_rotate(x[j], p, float(c[e]), float(c[half_rot + e])).x;
            } else if (e >= ATTN_PAIR_OFF && e < ATTN_PAIR_OFF + half_rot) {
                const uint d = e - ATTN_PAIR_OFF;
                x[j] = rope_rotate(p, x[j], float(c[d]), float(c[half_rot + d])).y;
            }
        }
    };
    // Every query head's score against a plain K row, on every lane.
    auto plain_scores = [&](thread float* x, thread float* s) {
        for (uint h = 0; h < G; ++h) {
            float acc = 0.0f;
            for (uint j = 0; j < QK; ++j) {
                acc += qb[h * QS + simd_lid * QK + j] * x[j];
            }
            s[h] = simd_sum(acc);
        }
    };

    const uint qid = simd_lid / 4u;
    const uint fm = (qid & 4u) + ((simd_lid / 2u) % 4u);
    const uint fn = (qid & 2u) * 2u + (simd_lid % 2u) * 2u;
    const uint n_blocks = (n_packed + KB - 1u) / KB;
    const uint per = (n_blocks + ATTN_SPLITS - 1u) / ATTN_SPLITS;
    const uint b_lo = min(z * per, n_blocks);
    const uint b_hi = min(b_lo + per, n_blocks);
    const uint word0 = sg * TILES;

    simdgroup_float8x8 Of[TILES];
    for (uint t = 0; t < TILES; ++t) {
        Of[t] = simdgroup_float8x8(0.0f);
    }
    float m = -FLT_MAX;
    float l = 0.0f;
    uint buf = 0u;
    for (uint it = 0; it < (b_hi - b_lo + KG - 1u) / KG; ++it) {
        const uint b = b_lo + it * KG + kg;
        const bool live = b < b_hi;
        const uint kb = b * KB;
        const uint bt_raw = live ? row_bt[kb / bs] : 0u;
        const bool do_rot = (ATTN_ROR != 0u) && ((bt_raw & 0x80000000u) != 0u);
        const uint base = (bt_raw & 0x7FFFFFFFu) * bs + kb % bs;
        auto packed_row = [&](uint key) { return (base + key - kb) * num_kv + kvh; };
        const uint k0 = kb + fn;
        const uint k1 = k0 + 1u;
        const bool v0 = live && k0 < n_packed;
        const bool v1 = live && k1 < n_packed;
        float s0 = 0.0f;
        float s1 = 0.0f;
        if (live && !do_rot) {
            // S over this simdgroup's dims: Kᵀ's tile t is dims (word0 + t)·8 + fm of keys k0, k1
            // — nibble fm of each key's word word0 + t.
            const uint r0 = packed_row(v0 ? k0 : kb);
            const uint r1 = packed_row(v1 ? k1 : kb);
            const float n0 = v0 ? tq_norms_k[r0] : 0.0f;
            const float n1 = v1 ? tq_norms_k[r1] : 0.0f;
            device const uint* w0 = tq_packed_k + r0 * pdim + word0;
            device const uint* w1 = tq_packed_k + r1 * pdim + word0;
            simdgroup_float8x8 S = simdgroup_float8x8(0.0f);
            for (uint t = 0; t < TILES; ++t) {
                simdgroup_float8x8 Kt;
                Kt.thread_elements()[0] = lut[(w0[t] >> (fm * 4u)) & 15u] * n0;
                Kt.thread_elements()[1] = lut[(w1[t] >> (fm * 4u)) & 15u] * n1;
                simdgroup_multiply_accumulate(S, Qf[t], Kt, S);
            }
            s0 = S.thread_elements()[0];
            s1 = S.thread_elements()[1];
        } else if (live) {
            // A span block: this simdgroup's KPS keys decoded whole, rounded to T as the dequant
            // pass leaves them, re-roped, and scored in the plain domain.
            float sc[KPS][G];
            for (uint u = 0; u < KPS; ++u) {
                const uint key = kb + KPS * sg + u;
                for (uint h = 0; h < G; ++h) {
                    sc[u][h] = 0.0f;
                }
                if (key < n_packed) {
                    const uint r = packed_row(key);
                    float x[QK];
                    for (uint j = 0; j < QK; ++j) {
                        x[j] = lut[tq_code(tq_packed_k, r * pdim, simd_lid * QK + j)];
                    }
                    tq_wht(x, QK, simd_lid);
                    const float n = tq_norms_k[r] / float(D);
                    for (uint j = 0; j < QK; ++j) {
                        x[j] = float(T(x[j] * tq_signs[simd_lid * QK + j] * n));
                    }
                    rope_row(x, key);
                    plain_scores(x, sc[u]);
                }
            }
            // The whole score lands in this simdgroup's partial (the lanes holding its keys'
            // columns), the others' hold none of it.
            const uint c0 = fn - KPS * sg;
            const uint c1 = c0 + 1u;
            s0 = c0 < KPS ? sc[c0][fm] : 0.0f;
            s1 = c1 < KPS ? sc[c1][fm] : 0.0f;
        }
        threadgroup float* sx = xs + (kg * 2u + buf) * SG * 64u;
        sx[sg * 64u + fm * 8u + fn] = s0;
        sx[sg * 64u + fm * 8u + fn + 1u] = s1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s0 = 0.0f;
        s1 = 0.0f;
        for (uint g = 0; g < SG; ++g) {
            s0 += sx[g * 64u + fm * 8u + fn];
            s1 += sx[g * 64u + fm * 8u + fn + 1u];
        }
        buf ^= 1u;
        if (!live) {
            continue;
        }
        // Online softmax of row fm (head q0 + fm).
        s0 = v0 ? s0 : -FLT_MAX;
        s1 = v1 ? s1 : -FLT_MAX;
        float bm = max(s0, s1);
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bm = max(bm, simd_shuffle_xor(bm, 8));
        const float new_m = max(m, bm);
        const float factor = metal::fast::exp(m - new_m);
        const float p0 = v0 ? metal::fast::exp(s0 - new_m) : 0.0f;
        const float p1 = v1 ? metal::fast::exp(s1 - new_m) : 0.0f;
        float ps = p0 + p1;
        ps += simd_shuffle_xor(ps, 1);
        ps += simd_shuffle_xor(ps, 8);
        l = l * factor + ps;
        m = new_m;
        simdgroup_float8x8 P;
        P.thread_elements()[0] = p0;
        P.thread_elements()[1] = p1;
        // O = O·factor + P·V: V's tile t is key kb + fm's dims (word0 + t)·8 + fn, fn + 1.
        const uint kv_key = kb + fm;
        const bool vv = kv_key < n_packed;
        const uint rv = packed_row(vv ? kv_key : kb);
        const float nv = vv ? tq_norms_v[rv] : 0.0f;
        device const uint* wv = tq_packed_v + rv * pdim + word0;
        for (uint t = 0; t < TILES; ++t) {
            const uint w = wv[t];
            simdgroup_float8x8 Vt;
            Vt.thread_elements()[0] = lut[(w >> (fn * 4u)) & 15u] * nv;
            Vt.thread_elements()[1] = lut[(w >> (fn * 4u + 4u)) & 15u] * nv;
            Of[t].thread_elements()[0] *= factor;
            Of[t].thread_elements()[1] *= factor;
            simdgroup_multiply_accumulate(Of[t], P, Vt, Of[t]);
        }
    }

    // The step's own key (the last split's): every head's score from the plain query, its V into
    // the codebook domain, `H·D·v`, in `xs`.
    const bool takes_own = own_key && z == ATTN_SPLITS - 1u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (takes_own && simd_gid == 0u) {
        const uint i = kv_len - 1u;
        const uint bt_raw = row_bt[i / bs];
        const bool do_rot = (ATTN_ROR != 0u) && ((bt_raw & 0x80000000u) != 0u);
        const uint phys = bt_raw & ((ATTN_ROR != 0u) ? 0x7FFFFFFFu : 0xFFFFFFFFu);
        const uint chunk = (ATTN_BLOCKS_PER_CHUNK == 0u) ? 0u : phys / ATTN_BLOCKS_PER_CHUNK;
        const uint blk_in_chunk =
            (ATTN_BLOCKS_PER_CHUNK == 0u) ? phys : phys % ATTN_BLOCKS_PER_CHUNK;
        const uint at = blk_in_chunk * kv_blk_stride + kvh * kv_head_stride + (i % bs) * D;
        device const T* k_ptr = (device const T*)k_cache[chunk] + at;
        device const T* v_ptr = (device const T*)v_cache[chunk] + at;
        float x[QK];
        float y[QK];
        for (uint j = 0; j < QK; ++j) {
            const uint e = simd_lid * QK + j;
            x[j] = float(ATTN_FOLD ? fold_kv[e] : k_ptr[e]);
            y[j] = float(ATTN_FOLD ? fold_kv[ATTN_FOLD_DIM + e] : v_ptr[e]) * tq_signs[e];
        }
        if (do_rot) {
            rope_row(x, i);
        }
        float s[G];
        plain_scores(x, s);
        tq_wht(y, QK, simd_lid);
        for (uint j = 0; j < QK; ++j) {
            xs[simd_lid * QK + j] = y[j];
        }
        if (simd_lid < G) {
            own_s[simd_lid] = s[simd_lid];
        }
    }

    // The key groups' merge into `qb` (the plain query's last reader is above): each row's max
    // over the groups, each group's output and sum scaled to it.
    if (sg == 0u && (simd_lid & 9u) == 0u) {
        ml[kg * G + fm] = m;
        ml[KG * G + kg * G + fm] = l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gm = -FLT_MAX;
    for (uint g = 0; g < KG; ++g) {
        gm = max(gm, ml[g * G + fm]);
    }
    const float to_gm = metal::fast::exp(m - gm);
    for (uint g = 0; g < KG; ++g) {
        if (g == kg) {
            for (uint t = 0; t < TILES; ++t) {
                const uint e = fm * QS + sg * DS + t * 8u + fn;
                const float o0 = Of[t].thread_elements()[0] * to_gm;
                const float o1 = Of[t].thread_elements()[1] * to_gm;
                qb[e] = g == 0u ? o0 : qb[e] + o0;
                qb[e + 1u] = g == 0u ? o1 : qb[e + 1u] + o1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Partials, `[batch, num_q_heads, ATTN_SPLITS, 2 + D]`: each head's max and sum, then its
    // output; the step's own key folded in by the split taking it.
    for (uint i = tid; i < G * D; i += D) {
        const uint h = i / D;
        float hm = -FLT_MAX;
        for (uint g = 0; g < KG; ++g) {
            hm = max(hm, ml[g * G + h]);
        }
        float hl = 0.0f;
        for (uint g = 0; g < KG; ++g) {
            hl += ml[KG * G + g * G + h] * metal::fast::exp(ml[g * G + h] - hm);
        }
        float o = qb[h * QS + i % D];
        if (takes_own) {
            const float s = own_s[h];
            const float new_m = max(hm, s);
            const float factor = metal::fast::exp(hm - new_m);
            const float p = metal::fast::exp(s - new_m);
            o = o * factor + p * xs[i % D];
            hl = hl * factor + p;
            hm = new_m;
        }
        device float* part = attn_partials + ((seq * num_q + q0 + h) * ATTN_SPLITS + z) * (D + 2u);
        if (i % D == 0u) {
            part[0] = hm;
            part[1] = hl;
        }
        part[2u + i % D] = o;
    }
}

SCRATCHY_KERNEL(attention_decode_gqa_tq_f16_specialized, attention_decode_gqa_tq<half>)
SCRATCHY_KERNEL(attention_decode_gqa_tq_bf16_specialized, attention_decode_gqa_tq<bfloat>)

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
//   4. Baked constants extend to BLOCK_SIZE + MAX_BLOCKS_PER_SEQ
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
// Baked constants 0..5: HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS,
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
#if SCRATCHY_COMPILES(attention_prefill_sdpa_v2_paged_f16_specialized)
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
    // gpt-oss attention sinks (ATTN_SINKS): the layer's [num_q_heads] sink
    // logits, model dtype. Only dereferenced when ATTN_SINKS.
    device const half*     sinks                 [[buffer(9)]],
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

    // Attention sinks: seed the max with the UNSCALED sink logit (see
    // attention_via_cache_v2) so the merged global max is ≥ the sink.
    U max_score = ATTN_SINKS_ON ? U(sinks[q_head_idx]) : -FLT_MAX;
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
    // The keys a query attends — causal (i <= q_abs_pos) and, sliding, inside the window
    // (q_abs_pos - i < window) — walked from the first this simdgroup takes to the last: the same
    // keys in the same order as walking all of them and skipping the rest.
    const uint win_lo = (ATTN_WINDOW > 0 && q_abs_pos + 1u > uint(ATTN_WINDOW))
        ? q_abs_pos + 1u - uint(ATTN_WINDOW) : 0u;
    const uint lane_lo = span_lo + simd_gid;
    const uint key_lo =
        lane_lo + (win_lo > lane_lo ? (win_lo - lane_lo + uint(BN) - 1u) / uint(BN) * uint(BN) : 0u);
    const uint key_hi = min(kv_len, q_abs_pos + 1u);
    for (uint i = key_lo; i < key_hi; i += uint(BN)) {

        const uint logical_block = i / block_size;
        const uint bt_raw = row_block_table[logical_block];
        const uint physical_block = bt_raw & ATTN_BT_MASK;
        const bool do_rot = (ATTN_ROR != 0u) && ((bt_raw & 0x80000000u) != 0u);
        const uint token_in_block = i - logical_block * block_size;
        // Chunked KV: deref the chunk backing this physical block.
        // ATTN_BLOCKS_PER_CHUNK is a baked constant. When it is 0 the
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
    // Attention sinks: the denominator gains the extra column's weight,
    // exp(sink − global_max) — every simdgroup computes this merge copy
    // redundantly, so each copy's divide stays consistent. The seed makes
    // global_max ≥ sink, so the argument is ≤ 0.
    if (ATTN_SINKS_ON) {
        global_sum += metal::fast::exp(U(sinks[q_head_idx]) - global_max);
    }

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
#endif

/// BF16 sibling of `attention_prefill_sdpa_v2_paged_f16_specialized`.
/// Same algorithm; sole difference is the `bfloat`/`half` element
/// type on the device pointers. f32 accumulator preserved.
///
/// max_total_threads_per_threadgroup(1024): see the f16 sibling — REQUIRED so
/// the 1024-thread (32-simdgroup) launch is guaranteed dispatchable on M1.
#if SCRATCHY_COMPILES(attention_prefill_sdpa_v2_paged_bf16_specialized)
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
    // gpt-oss attention sinks (ATTN_SINKS): the layer's [num_q_heads] sink
    // logits, model dtype. Only dereferenced when ATTN_SINKS.
    device const bfloat*   sinks                 [[buffer(9)]],
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

    // Attention sinks: seed the max with the UNSCALED sink logit (see
    // attention_via_cache_v2) so the merged global max is ≥ the sink.
    U max_score = ATTN_SINKS_ON ? U(sinks[q_head_idx]) : -FLT_MAX;
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
    // The keys a query attends — causal (i <= q_abs_pos) and, sliding, inside the window
    // (q_abs_pos - i < window) — walked from the first this simdgroup takes to the last: the same
    // keys in the same order as walking all of them and skipping the rest.
    const uint win_lo = (ATTN_WINDOW > 0 && q_abs_pos + 1u > uint(ATTN_WINDOW))
        ? q_abs_pos + 1u - uint(ATTN_WINDOW) : 0u;
    const uint lane_lo = span_lo + simd_gid;
    const uint key_lo =
        lane_lo + (win_lo > lane_lo ? (win_lo - lane_lo + uint(BN) - 1u) / uint(BN) * uint(BN) : 0u);
    const uint key_hi = min(kv_len, q_abs_pos + 1u);
    for (uint i = key_lo; i < key_hi; i += uint(BN)) {

        const uint logical_block = i / block_size;
        const uint bt_raw = row_block_table[logical_block];
        const uint physical_block = bt_raw & ATTN_BT_MASK;
        const bool do_rot = (ATTN_ROR != 0u) && ((bt_raw & 0x80000000u) != 0u);
        const uint token_in_block = i - logical_block * block_size;
        // Chunked KV: deref the chunk backing this physical block.
        // ATTN_BLOCKS_PER_CHUNK is a baked constant. When it is 0 the
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
    // Attention sinks: the denominator gains the extra column's weight,
    // exp(sink − global_max). The seed makes global_max ≥ sink.
    if (ATTN_SINKS_ON) {
        global_sum += metal::fast::exp(U(sinks[q_head_idx]) - global_max);
    }

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
#endif

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
// ATTN_BLOCKS_PER_CHUNK / ATTN_MAX_BLOCKS_PER_SEQ from baked constants, so
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
    // The scratch has no batch dimension: this stages batch row 0 only, and the
    // tape runs it only on single-sequence steps (`RuntimeGate::OnlyIfOneSequence`);
    // a step with several sequences runs the attention's per-row twin instead.
    const uint seq_idx       = 0;

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

#if SCRATCHY_COMPILES(rope_once_gqa_shared_f16_specialized)
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
#endif

#if SCRATCHY_COMPILES(rope_once_gqa_shared_bf16_specialized)
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
#endif

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
#if SCRATCHY_COMPILES(attention_prefill_sdpa_gqa_shared_f16_specialized)
kernel void attention_prefill_sdpa_gqa_shared_f16_specialized(
    device       half* output       [[buffer(0)]],   // [total_q, num_q_heads, head_dim]
    device const half* q            [[buffer(1)]],   // [total_q, num_q_heads, head_dim]
    device const uint* cu_seqlens_q [[buffer(2)]],   // [batch+1]
    device const uint* seq_used_k   [[buffer(3)]],   // [batch]
    device const uint* block_table  [[buffer(4)]],   // [batch, MAX_BLOCKS_PER_SEQ]
    device const uint64_t* k_cache  [[buffer(5)]],   // chunk-address table
    device const uint64_t* v_cache  [[buffer(6)]],   // chunk-address table
    // Spans, slot 7 has TWO mutually-exclusive uses (only one is bound per
    // dispatch; folded by the baked ATTN_KSCR / ATTN_ROR):
    //   ATTN_K_SCRATCH (rope-once-to-scratch): the DENSE pre-roped K scratch
    //     written ONCE by `rope_once_gqa_shared_*`. K is staged from here with
    //     the byte-identical cache-load math (logical-block base) and NO
    //     per-tile rotation — the amortized-to-~0% long-context path.
    //   else ATTN_ROPE_ON_READ (in-kernel rope): f16 cos_sin; K is staged from
    //     the cache and rotated in-place in smem on every q-tile (the per-tile
    //     redundancy the scratch path replaces).
    // The unrotated flag rides in block_table bit 31 (no flag buffer).
    device const half*     cos_sin               [[buffer(7)]],
    // gpt-oss attention sinks (ATTN_SINKS): the layer's [num_q_heads] sink
    // logits, model dtype. Only dereferenced when ATTN_SINKS. Slot 9 — the
    // binding set's SpanIds rides 8.
    device const half*     sinks                [[buffer(9)]],
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

    // Attention sinks: seed the max with the UNSCALED sink logit (see
    // attention_via_cache_v2). Each simdgroup owns one head's full key
    // range (no cross-simdgroup merge), so the seed doubles as the row max.
    U run_max = ATTN_SINKS_ON ? U(sinks[q_head_idx]) : -FLT_MAX;
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
    // Attention sinks: the denominator gains the extra column's weight,
    // exp(sink − run_max) ≤ 1 by the seed, contributing no V.
    if (ATTN_SINKS_ON) {
        sum_exp += metal::fast::exp(U(sinks[q_head_idx]) - run_max);
    }
    device half* o_ptr = o_row + simd_lid * qk_per_thread;
    const U inv = (sum_exp != 0) ? (U(1) / sum_exp) : U(0);
    for (uint j = 0; j < qk_per_thread; ++j) {
        o_ptr[j] = half(o_reg[j] * inv);
    }
}
#endif

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
#if SCRATCHY_COMPILES(attention_prefill_sdpa_gqa_shared_bf16_specialized)
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
    // gpt-oss attention sinks (ATTN_SINKS): see the f16 sibling (slot 9).
    device const bfloat*   sinks                [[buffer(9)]],
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

    // Attention sinks: seed the max with the UNSCALED sink logit (see
    // attention_via_cache_v2). Each simdgroup owns one head's full key
    // range (no cross-simdgroup merge), so the seed doubles as the row max.
    U run_max = ATTN_SINKS_ON ? U(sinks[q_head_idx]) : -FLT_MAX;
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
    // Attention sinks: the denominator gains the extra column's weight,
    // exp(sink − run_max) ≤ 1 by the seed, contributing no V.
    if (ATTN_SINKS_ON) {
        sum_exp += metal::fast::exp(U(sinks[q_head_idx]) - run_max);
    }
    device bfloat* o_ptr = o_row + simd_lid * qk_per_thread;
    const U inv = (sum_exp != 0) ? (U(1) / sum_exp) : U(0);
    for (uint j = 0; j < qk_per_thread; ++j) {
        o_ptr[j] = bfloat(o_reg[j] * inv);
    }
}
#endif
