// SPDX-License-Identifier: Apache-2.0
//
// NAX (matrix-accelerator) paged-K/V variant of MLX's steel attention.
//
// This is the FA-2 attention algorithm from MLX's `steel_attention_nax.h`
// (which drives the Apple matrix accelerator via `NAXTile` / the MPP
// `matmul2d` cooperative tensors in `metal_nax.h`), with the K/V load
// path rewritten to read from scratchy-target-metal's PAGED KV cache instead of
// a contiguous device buffer — exactly mirroring how
// `steel_attention_paged_kernel.h` is the paged port of the simdgroup
// `steel_attention_kernel.h`.
//
// Why a separate kernel rather than reusing PagedBlockLoaderT: the NAX
// kernel does NOT stage K/V through threadgroup memory in the non-spans
// path. `NAXTile::load` reads tiles straight from device memory into
// matrix-accelerator register fragments (16 rows per fragment). Since the
// paged cache uses BLOCK_SIZE=16, one 16-row NAX fragment maps to EXACTLY
// one paged block, so the page indirection is a per-fragment device-pointer
// resolve through the block table — no threadgroup loader needed.
//
// Spans / rope-on-read (ROPE-ONCE-TO-SCRATCH): K is stored UNROTATED in the
// cache (the cache is the position-independent reuse artifact — never mutated).
// A span cached at one position and reused at another MUST be re-roped to the
// reader's absolute position. Rather than re-rope each K tile INSIDE the
// attention's per-query-tile loop (flash re-reads/re-ropes each K tile ~NQ =
// m/BQ times — ≈128× at m=8192 — which measured ~1.99× of the non-spans
// baseline, dominated by the per-tile REDUNDANCY, NOT the cos/sin source),
// the K is roped ONCE — outside the attention — into a transient device
// SCRATCH buffer by the separate `rope_once_nax` kernel (one pass over K =
// O(K)). The attention then reads PRE-ROPED K from the scratch with an
// ordinary load and NO per-tile rotation, so it runs at the non-spans
// baseline. For long context the one-time O(K) rope is negligible vs the
// O(NQ·K) attention, so ON/OFF approaches ~1.0×.
//
// V is never roped (no RoPE on V) — V is always read from the cache. When
// ROPE_ON_READ is unset, no scratch is allocated, the rope-once kernel is not
// dispatched, and the attention reads K directly from the cache exactly as
// before (non-spans path byte-identical).
//
// Two kernels live in this header:
//   1. `rope_once_nax` — reads unrotated K from the paged cache (block_table
//      bit 31 flags "unrotated"), ropes each row to its absolute position
//      using the cos_sin TABLE (the sincos is done ONCE per key here, not per
//      q-tile), and writes the roped K into the scratch in a DENSE
//      logical-block layout (same per-block strides as the cache, but indexed
//      by logical block — single contiguous buffer).
//   2. `attention_nax_paged` — the FA-2 attention. When ROPE_ON_READ it reads
//      K from the scratch (buffer 7 = scratch base pointer) via the dense
//      logical-block resolve; otherwise reads K from the cache (buffer 5).
//
// attention_nax_paged bindings (mirror `attention_steel_paged`):
//   buffer(0) O             [total_q, num_q_heads, head_dim]
//   buffer(1) Q             [total_q, num_q_heads, head_dim]
//   buffer(2) cu_seqlens_q  [batch+1]
//   buffer(3) seq_used_k    [batch]
//   buffer(4) block_table   [batch, max_blocks_per_seq]  (bit 31 = unrotated)
//   buffer(5) k_cache       per-layer chunk-address table (uint64 gpuAddrs)
//   buffer(6) v_cache       per-layer chunk-address table (uint64 gpuAddrs)
//   buffer(7) k_scratch     roped-K dense buffer (dtype T); only read when
//                           ROPE_ON_READ — the rope-once kernel's output
//
// rope_once_nax bindings:
//   buffer(0) k_scratch     [num_pages, num_kv_heads, BLOCK_SIZE, head_dim] T
//   buffer(1) block_table   [batch, max_blocks_per_seq]  (bit 31 = unrotated)
//   buffer(2) k_cache       per-layer chunk-address table (uint64 gpuAddrs)
//   buffer(3) seq_used_k    [batch]
//   buffer(4) cos_sin       [max_pos, rot_dim]
//
// Gate: NAX hardware only (M5+/Apple9). The dispatcher selects this
// kernel iff `is_nax_capable(profile.generation)` — same gate the NAX
// qmm uses.

#pragma once

#include <metal_stdlib>

#include "metal_nax.h"  // NAXTile + BaseNAXFrag + tile_matmad_nax

using namespace metal;
using namespace mlx::steel;

// ---- minimal numeric limits (mlx pulls these from utils.h) ---------------
template <typename T>
struct NaxLimits {
  static constexpr constant float finite_min = -3.38953139e+38f;
  static constexpr constant float finite_max = 3.38953139e+38f;
};

constant float NAX_M_LOG2E = 1.44269504088896340736f;

// Function constants (same slot layout as attention_steel_paged).
constant uint  NAXP_HEAD_DIM           [[function_constant(0)]];  // == BD
constant uint  NAXP_NUM_Q_HEADS        [[function_constant(1)]];
constant uint  NAXP_NUM_KV_HEADS       [[function_constant(2)]];
constant float NAXP_SCALE              [[function_constant(3)]];
constant uint  NAXP_BLOCK_SIZE         [[function_constant(4)]];  // == BLOCK_SIZE_
constant uint  NAXP_MAX_BLOCKS_PER_SEQ [[function_constant(5)]];
constant uint  NAXP_BLOCKS_PER_CHUNK   [[function_constant(6)]];
constant int   NAXP_WINDOW             [[function_constant(7)]];  // 0 = full attn

// Rope-on-read (spans): same slots/semantics as ATTN_PAGED_ROT_DIM /
// ATTN_PAGED_PAIR_OFF / ATTN_PAGED_ROPE_ON_READ in the simdgroup steel
// kernel. ROPE_ON_READ selects the scratch K-source in attention_nax_paged
// (the K is pre-roped by rope_once_nax); rot_dim/pair_off are consumed only
// by the rope_once_nax kernel. The is_function_constant_defined guard folds
// buffer 7 (the scratch) away when unset, so non-spans NAX attention reads K
// straight from the cache — byte-identical to the prior direct-load kernel.
constant uint  NAXP_ROT_DIM            [[function_constant(8)]];
constant uint  NAXP_PAIR_OFF           [[function_constant(9)]];
constant uint  NAXP_ROPE_ON_READ       [[function_constant(10)]];
constant bool  NAXP_ROR_DEFINED = is_function_constant_defined(NAXP_ROPE_ON_READ);
constant uint  NAXP_ROR = NAXP_ROR_DEFINED ? NAXP_ROPE_ON_READ : 0u;

struct NMaxOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); }
};
struct NSumOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x + y; }
};
struct NMulOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x * y; }
};
struct NExpSubOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return fast::exp2(x - y); }
};
struct NDivOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x / y; }
};

// Resolve the device base pointer for paged block `logical_block` of the
// current sequence: indirect through the per-seq block table, then through
// the per-layer chunk-address table, then add the kv-head offset.
//   chunk_table[ physical / blocks_per_chunk ]
//     + (physical % blocks_per_chunk) * kv_blk_stride
//     + kv_head_off
// Mirrors `PagedBlockLoaderT::resolve` (paged_loader.h).
//
// Spans: block_table bit 31 carries the rope-on-read unrotated flag; mask
// it off for addressing (identity for non-spans — the worker only ever
// sets bit 31 when ROPE_ON_READ, so this is a free ALU op there). This
// mirrors `paged_loader.h`'s `& 0x7FFFFFFFu` in all three resolve paths.
template <typename T>
METAL_FUNC const device T* nax_resolve_block(
    const device uint64_t* chunk_table,
    const device uint* block_table_row,
    int logical_block,
    int num_pages,
    int blocks_per_chunk,
    int kv_blk_stride,
    int kv_head_off) {
  int lb = logical_block;
  if (lb >= num_pages) {
    lb = num_pages - 1;
  }
  const uint physical = uint(block_table_row[lb]) & 0x7FFFFFFFu;
  const uint chunk = physical / uint(blocks_per_chunk);
  const uint bic = physical % uint(blocks_per_chunk);
  return (const device T*)chunk_table[chunk] + int(bic) * kv_blk_stride + kv_head_off;
}

// Resolve the base pointer for LOGICAL block `logical_block` of the roped-K
// SCRATCH (rope-on-read spans). The scratch is a single dense contiguous
// buffer written by `rope_once_nax`, indexed by LOGICAL block (no block_table
// indirection, no chunk table) with the SAME per-block strides as the cache:
//   scratch + logical_block * kv_blk_stride + kv_head_off
// so attention can reuse the cache's `do_qk_direct` load math verbatim.
template <typename T>
METAL_FUNC const device T* nax_resolve_scratch(
    const device T* scratch,
    int logical_block,
    int num_pages,
    int kv_blk_stride,
    int kv_head_off) {
  int lb = logical_block;
  if (lb >= num_pages) {
    lb = num_pages - 1;
  }
  return scratch + lb * kv_blk_stride + kv_head_off;
}

// ── rope-once kernel ──────────────────────────────────────────────────────
//
// One pass over a request's K for one layer: read the UNROTATED K from the
// paged cache, rope each row to its absolute position (cos/sin from the table,
// done ONCE per key), write the roped K into the dense scratch. The cache is
// never mutated. O(K) — amortizes to ~0% against the O(NQ·K) attention.
//
// Grid: one thread per (logical_block, kv_head, token_in_block, half-dim
// pair). A thread ropes the PAIR (d, d+pair_off) of one K row. Rows whose
// block is NOT flagged unrotated (block_table bit 31 clear) are COPIED
// through unchanged so the scratch is a complete K image either way (the
// attention always reads the scratch when ROR is on). Bit-exact with
// `rope_append` / `cpu_rope_k` (round to T on store).
//
//   tid.x in [0, num_kv_heads * BLOCK_SIZE * half_dim)   (head, row, pair)
//   tid.y = logical block index
template <
    typename T,
    int BD,             // head_dim
    int BLOCK_SIZE_>
[[kernel]]
void rope_once_nax_kernel(
    device T*          k_scratch    [[buffer(0)]],
    const device uint* block_table  [[buffer(1)]],
    const device uint64_t* k_cache  [[buffer(2)]],
    const device uint* seq_used_k   [[buffer(3)]],
    const device T*    cos_sin      [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint logical_block = gid.y;
  const uint seq_idx = 0; // single-sequence prefill (rope-once is per-request)

  const uint kv_len = seq_used_k[seq_idx];
  const uint num_pages = (kv_len + uint(BLOCK_SIZE_) - 1u) / uint(BLOCK_SIZE_);
  if (logical_block >= num_pages) {
    return;
  }

  const uint half_dim = NAXP_ROT_DIM / 2u;
  const uint pair_off = NAXP_PAIR_OFF;
  // gid.x decodes to (kv_head, token_in_block, rotary index d in [0,half_dim)).
  const uint per_head = uint(BLOCK_SIZE_) * half_dim;
  const uint kv_head  = gid.x / per_head;
  const uint rem      = gid.x % per_head;
  const uint tib      = rem / half_dim;     // token in block
  const uint d        = rem % half_dim;     // rotary pair index
  if (kv_head >= NAXP_NUM_KV_HEADS) {
    return;
  }

  const uint pos = logical_block * uint(BLOCK_SIZE_) + tib;

  const int kv_blk_stride  = int(NAXP_NUM_KV_HEADS) * BLOCK_SIZE_ * BD;
  const int kv_head_off    = int(kv_head) * (BLOCK_SIZE_ * BD);
  const device uint* row_block_table =
      block_table + seq_idx * NAXP_MAX_BLOCKS_PER_SEQ;

  // Source row (cache, possibly unrotated) and dest row (scratch, dense).
  const device T* k_blk = nax_resolve_block<T>(
      k_cache, row_block_table, int(logical_block), int(num_pages),
      int(NAXP_BLOCKS_PER_CHUNK), kv_blk_stride, kv_head_off);
  device T* dst_blk = k_scratch + int(logical_block) * kv_blk_stride + kv_head_off;
  const device T* k_row = k_blk + tib * uint(BD);
  device T* dst_row = dst_blk + tib * uint(BD);

  const bool past_len = pos >= kv_len;
  const bool flagged =
      (row_block_table[logical_block] & 0x80000000u) != 0u;

  // Pass-through copy of all NON-rotated head dims (proportional rope leaves
  // dims outside {[0,half_dim) ∪ [pair_off,pair_off+half_dim)} untouched).
  // Thread d==0 mirrors them once. For full NeoX (rot_dim==head_dim) this
  // loop is empty (every dim is in a rotary pair).
  if (d == 0u) {
    for (uint dd = 0u; dd < uint(BD); dd++) {
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
  const device T* cr = cos_sin + pos * NAXP_ROT_DIM;
  const float c = float(cr[d]);
  const float s = float(cr[half_dim + d]);
  const float x0 = float(k_row[d]);
  const float x1 = float(k_row[pair_off + d]);
  dst_row[d]            = T(x0 * c - x1 * s);
  dst_row[pair_off + d] = T(x1 * c + x0 * s);
}

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    int BLOCK_SIZE_,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]]
void attention_nax_paged(
    device T*          O            [[buffer(0)]],
    const device T*    Q            [[buffer(1)]],
    const device uint* cu_seqlens_q [[buffer(2)]],
    const device uint* seq_used_k   [[buffer(3)]],
    const device uint* block_table  [[buffer(4)]],
    const device uint64_t* k_cache  [[buffer(5)]],
    const device uint64_t* v_cache  [[buffer(6)]],
    // Rope-on-read (spans): PRE-ROPED K scratch (dtype T), dense by logical
    // block, written by `rope_once_nax`. Only read when NAXP_ROPE_ON_READ;
    // K then comes from here (no per-tile rotation) instead of the cache.
    const device T*    k_scratch    [[buffer(7)]],
    // Block-diagonal span attention (spans): PER-TOKEN span label buffer,
    // indexed by ABSOLUTE token position. label = (span's first_token + 1);
    // 0 = shared/attends-all. Only read when NAXP_ROR != 0 (the branch below
    // folds away on non-spans), so buffer 8 is left unbound and the non-spans
    // path is byte-identical to the prior kernel. The host already binds this
    // at buffer(8) for the NAX arm (shared binding set with steel).
    const device uint* span_ids     [[buffer(8)]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  (void)lid;

  const uint seq_idx    = tid.z;
  const uint q_head_idx = tid.y;
  const uint gqa_factor = NAXP_NUM_Q_HEADS / NAXP_NUM_KV_HEADS;
  const uint kv_head_idx = q_head_idx / gqa_factor;

  const uint seq_start     = cu_seqlens_q[seq_idx];
  const uint seq_end       = cu_seqlens_q[seq_idx + 1];
  const uint new_q_for_seq = seq_end - seq_start;
  const uint kv_len        = seq_used_k[seq_idx];
  const uint prefix_len    = kv_len - new_q_for_seq;
  const uint q_block_base  = tid.x * uint(BQ);
  const uint global_q_base = seq_start + q_block_base;

  if (q_block_base >= new_q_for_seq) {
    return;
  }
  const uint q_tile_rows = min(uint(BQ), new_q_for_seq - q_block_base);

  const int Q_stride_tok = int(NAXP_NUM_Q_HEADS) * BD;
  Q += int(global_q_base) * Q_stride_tok + int(q_head_idx) * BD;
  O += int(global_q_base) * Q_stride_tok + int(q_head_idx) * BD;

  const int kv_blk_stride  = int(NAXP_NUM_KV_HEADS) * BLOCK_SIZE_ * BD;
  const int kv_head_stride = BLOCK_SIZE_ * BD;
  const int kv_head_off    = int(kv_head_idx) * kv_head_stride;
  const int per_token_stride = BD; // row stride inside a block
  const device uint* row_block_table =
      block_table + seq_idx * NAXP_MAX_BLOCKS_PER_SEQ;
  const int num_pages = int((kv_len + uint(BLOCK_SIZE_) - 1u) / uint(BLOCK_SIZE_));

  const float scale2 = NAXP_SCALE * NAX_M_LOG2E;

  // ----- NAX tile geometry -------------------------------------------------
  constexpr short kU = 16; // NAX fragment edge (rows/cols)
  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0,
      "Each simdgroup must host atleast 1 NAX matrix along Q sequence.");
  static_assert(
      BLOCK_SIZE_ == kU,
      "NAX paged kernel assumes BLOCK_SIZE == 16 (one NAX frag = one page).");

  constexpr int TQ = BQ / (kNWarps * kU); // Q-seq frags per warp
  constexpr int TD = BD / kU;             // head-dim frags
  constexpr short TK = BK / kU;           // KV-seq frags per K-tile (== pages)

  static_assert(TQ == 1, "Check TQ");

  // Rope-on-read (spans): when NAXP_ROR is set, K is read from the scratch
  // (already roped once by `rope_once_nax`) via `resolve_k`, so attention does
  // the SAME direct device→fragment load as the non-spans path — NO per-tile
  // rotation, no extra cos_sin reads inside this O(NQ·K) loop. The earlier
  // in-register re-rope (rotate each K tile on every q-tile) was ~1.99× of the
  // baseline because flash re-ropes K ~m/BQ× (the per-tile REDUNDANCY); roping
  // ONCE outside (O(K)) amortizes to ~0% for long context.

  using otile_t = NAXTile<AccumType, TQ, TD>;
  otile_t Otile;
  Otile.clear();

  // Q row offset this warp owns.
  const short tm = kU * TQ * simd_group_id;
  Q += tm * Q_stride_tok;
  O += tm * Q_stride_tok;

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  constexpr short kRowsPT = otile_t::kRowsPerThread;
  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = NaxLimits<AccumType>::finite_min;
  }

  // Causal/window iteration bounds (absolute K-axis coords). The Q tile
  // rows are [prefix_len + q_block_base + tm, ... + q_tile_rows).
  const int abs_q_min = int(prefix_len) + int(q_block_base) + int(tm);
  const int abs_q_max_excl =
      int(prefix_len) + int(q_block_base) + int(q_tile_rows);
  const int kv_tiles_total = int((kv_len + uint(BK) - 1u) / uint(BK));
  int kb_lim = (abs_q_max_excl + BK - 1) / BK;
  if (kb_lim > kv_tiles_total) kb_lim = kv_tiles_total;
  const int kb_min_causal = abs_q_min / BK;

  int kb_start = 0;
  if (NAXP_WINDOW > 0) {
    const int first_k = abs_q_min - NAXP_WINDOW + 1;
    if (first_k > 0) kb_start = first_k / BK;
  }
  // Block-diagonal span attention: a SPAN-UNIFORM Q tile (all rows in one span)
  // attends ONLY that span's K-tiles → start the kb-loop at the span's first
  // tile (real O(T²)→O(T·span) cut). Ported from the simdgroup steel kernel
  // (steel_attention_paged_kernel.h ~L552), NAXP_ROR-gated (folds away on
  // non-spans). span_ids is PER-TOKEN: label = (span's first_token + 1),
  // 0 = shared/attends-all; index by ABSOLUTE token position (abs_q_min /
  // abs_q_max_excl are already absolute). Convert the per-token first_token to
  // the first KV TILE via `/ BK` — the KB-LOOP stride (NAX BK != BLOCK_SIZE_,
  // e.g. BK=32 at BD=128), NOT the page size — flooring a mid-tile span start
  // to its containing tile. A tile straddling a span boundary (sf != sl) keeps
  // the full causal range. Raising kb_start skips leading tiles directly (the
  // kb-loop honors it; no loader.seek() — NAX resolves per-block each iter).
  if (NAXP_ROR != 0u) {
    const uint sf = span_ids[uint(abs_q_min)];
    const uint sl = span_ids[uint(abs_q_max_excl - 1)];
    if (sf != 0u && sf == sl) {
      const int span_kb = (int(sf) - 1) / int(BK);
      if (span_kb > kb_start) { kb_start = span_kb; }
    }
  }

  const int kv_aligned_tiles = int(kv_len / uint(BK));
  const uint kv_rem = kv_len % uint(BK);
  const short lim_rows_q = short(int(q_tile_rows) - int(tm));

  using stile_t = NAXTile<AccumType, TQ, TK>;

  // Resolve the K base pointer for logical block `lb`. Spans (NAXP_ROR != 0):
  // K comes PRE-ROPED from the dense scratch (rope_once_nax wrote it). Else:
  // straight from the paged cache. The branch is on the ROR function constant,
  // so the compiler keeps only one path per pipeline variant.
  auto resolve_k = [&](int lb) -> const device T* {
    if (NAXP_ROR != 0u) {
      return nax_resolve_scratch<T>(
          k_scratch, lb, num_pages, kv_blk_stride, kv_head_off);
    }
    return nax_resolve_block<T>(
        k_cache, row_block_table, lb, num_pages,
        int(NAXP_BLOCKS_PER_CHUNK), kv_blk_stride, kv_head_off);
  };

  // ── S = Q @ K^T loading K DIRECTLY from device into NAX frags (NO
  //    threadgroup staging; the NAX accelerator wants device→fragment loads).
  //    K source = the scratch (pre-roped) for spans, else the cache — both
  //    via the SAME load math (the scratch mirrors the cache block layout). ─
  auto do_qk_direct = [&](int kb_, thread stile_t& Stile) {
    Stile.clear();
    const bool is_last_k = (kb_ == kv_aligned_tiles) && (kv_rem != 0u);
    const short lim_rows_k = is_last_k ? short(kv_rem) : short(BK);
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik += 2) {
        const int lb0 = kb_ * TK + ik;
        const int lb1 = kb_ * TK + ik + 1;
        const device T* Kp0 = resolve_k(lb0);
        const device T* Kp1 = resolve_k(lb1);
        const short klim0 = short(lim_rows_k - ik * kU);
        const short klim1 = short(lim_rows_k - (ik + 1) * kU);
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TD; id++) {
          NAXTile<T, 1, 1> Qtile;
          NAXTile<T, 1, 1> Ktile0;
          NAXTile<T, 1, 1> Ktile1;
          const int Q_load_off = iq * kU * Q_stride_tok + id * kU;
          const int K_load_off = id * kU;
          if (lim_rows_q < BQ) {
            Qtile.load_rows(Q + Q_load_off, Q_stride_tok, short(lim_rows_q - iq * kU));
          } else {
            Qtile.load(Q + Q_load_off, Q_stride_tok);
          }
          if (klim0 < kU) {
            Ktile0.load_rows(Kp0 + K_load_off, per_token_stride, klim0);
          } else {
            Ktile0.load(Kp0 + K_load_off, per_token_stride);
          }
          if (klim1 < kU) {
            Ktile1.load_rows(Kp1 + K_load_off, per_token_stride, klim1);
          } else {
            Ktile1.load(Kp1 + K_load_off, per_token_stride);
          }
          stile_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qtile.frag_at(0, 0),
              metal::false_type{},
              Ktile0.frag_at(0, 0),
              Ktile1.frag_at(0, 0),
              metal::true_type{});
        }
      }
    }
  };

  // ── scale + partial-tail/causal/window masks on Stile (register-only). ─
  auto scale_and_mask = [&](int kb_, thread stile_t& Stile) {
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale2;
    }

    const bool is_last_k = (kb_ == kv_aligned_tiles) && (kv_rem != 0u);
    if (is_last_k) {
      constexpr auto neg_inf = NaxLimits<AccumType>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = ik * kU + sn;
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < int(kv_rem)) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    if (kb_ >= kb_min_causal) {
      constexpr auto neg_inf = NaxLimits<AccumType>::finite_min;
      const int base_row = int(prefix_len) + int(q_block_base) + tm + sm;
      const int base_col = kb_ * BK + sn;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const int r = base_row + iq * kU + ii * stile_t::kFragRowsJump;
              const int c = base_col + ik * kU + jj;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    if (NAXP_WINDOW > 0 && (abs_q_max_excl - 1 - kb_ * BK) >= NAXP_WINDOW) {
      constexpr auto neg_inf = NaxLimits<AccumType>::finite_min;
      const int base_row = int(prefix_len) + int(q_block_base) + tm + sm;
      const int base_col = kb_ * BK + sn;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const int r = base_row + iq * kU + ii * stile_t::kFragRowsJump;
              const int c = base_col + ik * kU + jj;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = ((r - c) >= NAXP_WINDOW) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }
  };

  // ── online-softmax update of (max,sum,Otile) from the masked Stile. ────
  auto softmax_update = [&](thread stile_t& Stile) {
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];

    Stile.template row_reduce<NMaxOp>(new_max);
    Stile.template row_bin_op<NExpSubOp>(new_max);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }
    Stile.template row_reduce<NSumOp>(sum_score);

    Otile.template row_bin_op<NMulOp>(factor);
  };

  // ── O += P @ V for block kb_ (V read direct from device, never roped). ─
  auto do_pv = [&](int kb_, thread stile_t& Stile) {
    const short lim_rows_k = (kb_ == kv_aligned_tiles) && (kv_rem != 0u)
        ? short(kv_rem)
        : short(BK);
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id += 2) {
        if (BD == 128 && id == 4) {
          threadgroup_barrier(mem_flags::mem_none);
        }
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const int lb = kb_ * TK + ik;
          const device T* Vp = nax_resolve_block<T>(
              v_cache, row_block_table, lb, num_pages,
              int(NAXP_BLOCKS_PER_CHUNK), kv_blk_stride, kv_head_off);
          const short vlim = short(lim_rows_k - ik * kU);

          NAXTile<T, 1, 2> Vtile;
          const int V_load_off = id * kU;
          if (vlim < kU) {
            Vtile.load_rows(Vp + V_load_off, per_token_stride, vlim);
          } else {
            Vtile.load(Vp + V_load_off, per_token_stride);
          }

          otile_t::NAXFrag_t::mma(
              Otile.frag_at(iq, id),
              Otile.frag_at(iq, id + 1),
              Stile.frag_at(iq, ik),
              metal::false_type{},
              Vtile.frag_at(0, 0),
              Vtile.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }
  };

  // ----- KV loop -----------------------------------------------------------
  // One direct-load loop for both paths. `do_qk_direct`'s K source is the
  // scratch (pre-roped, spans) or the cache (non-spans) via `resolve_k`, which
  // folds on the ROR function constant — so the spans path runs at the
  // non-spans baseline (the rope was done ONCE by rope_once_nax). V is always
  // read from the cache (never roped).
  for (int kb = kb_start; kb < kb_lim; kb++) {
    stile_t Stile;
    do_qk_direct(kb, Stile);
    scale_and_mask(kb, Stile);
    softmax_update(Stile);
    simdgroup_barrier(mem_flags::mem_none);
    do_pv(kb, Stile);
  }

  // ----- normalize + store -------------------------------------------------
  threadgroup_barrier(mem_flags::mem_none);

  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }
  Otile.template row_bin_op<NMulOp>(rcp);

  if (lim_rows_q < BQ) {
    if (lim_rows_q <= 0) return;
    Otile.store_rows(O, Q_stride_tok, lim_rows_q);
  } else {
    Otile.store(O, Q_stride_tok);
  }
}
