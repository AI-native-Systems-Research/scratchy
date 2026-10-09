// SPDX-License-Identifier: Apache-2.0
//
// Simdgroup (pre-NAX) wide-head paged attention — the steel port of
// `attention_nax_paged_wide` (steel_attention_nax_paged_kernel.h), for
// heads one warp's O cannot hold (head_dim 512: Gemma 4's global layers)
// on GPUs without the matrix accelerator.
//
// Same FA-2 algorithm, same warp arrangement, same exchange cadence as the
// NAX wide kernel; only the matrix ops differ (simdgroup MMA + fragment
// loads vs `matmul2d` cooperative tensors):
//
//   - Warps are WM Q-row blocks x WN head-dim slices; a warp's O is its
//     block's 16 rows over its BD/WN head-dim slice (64 f32 registers at
//     BD=512/WN=4 — the same per-warp O budget as the head_dim 256 steel
//     kernel, which is exactly why the classic kernel cannot take BD=512:
//     its whole-head O would be 256 registers a thread).
//   - Q @ K^T is KV-split across the block's warps: each warp forms the
//     WHOLE-HEAD S of one page group a step (WN ops over the head-dim
//     slices summed in slice order — no partial-S summation across warps),
//     so S crosses warps through threadgroup memory ONCE a step as a pure
//     relocation (the #306 exchange granularity, not the per-tile partial
//     sums it replaced).
//   - Softmax and P @ V run D-split: every warp of a block applies the
//     online-softmax update to the step's K-tiles (duplicated WN x a
//     block — cheap ALU on 16 x BK elements) and accumulates its own
//     head-dim slice of O from the exchanged P.
//   - Q/K/V fragments load straight from device memory (no threadgroup
//     staging at all — the property that makes BD=512 fit: the classic
//     kernel's smem staging needs 33 KB for the Q tile alone at BQ=32,
//     over the whole 32 KB budget; this kernel's only threadgroup memory
//     is the S exchange buffer).
//
// Bindings, constants, masks, spans seek and the rope-once-to-scratch
// contract are `attention_paged`'s (steel_attention_paged_kernel.h) — this
// header is included after it in `attention_steel_wide_paged.metal` and
// reuses its ATTN_PAGED_* constants, its reduce ops (MaxOp/SumOp/...) and
// its Limits. The rope-once twin is the classic header's
// `rope_once_steel_kernel`, instantiated by the same .metal at the wide
// kernel's page size.

#include "attn.h"
#include "paged_resolve.h"

using namespace metal;
using namespace mlx::steel;

// clang-format off
template <
    typename T,
    int BQ,          // Q rows a threadgroup (WM * 16)
    int BK,          // keys a K-tile of the softmax/PV loop
    int BD,          // head_dim
    int WM,          // Q-row blocks (16 rows) a threadgroup
    int WN,          // head-dim slices (warps a Q block)
    int BLOCK_SIZE_, // paged-cache block size (page)
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]]
void attention_steel_paged_wide(
    device T*          O            [[buffer(0)]],   // [total_q, num_q_heads, head_dim]
    const device T*    Q            [[buffer(1)]],   // [total_q, num_q_heads, head_dim]
    const device uint* cu_seqlens_q [[buffer(2)]],   // [batch+1]
    const device uint* seq_used_k   [[buffer(3)]],   // [batch] total cached K
    const device uint* block_table  [[buffer(4)]],   // [batch, max_blocks_per_seq]
    const device uint64_t* k_cache  [[buffer(5)]],   // per-layer chunk-address table
    const device uint64_t* v_cache  [[buffer(6)]],   // per-layer chunk-address table
    const device T*    k_scratch    [[buffer(7)]],   // pre-roped K (ROPE_ON_READ)
    const device uint* span_ids     [[buffer(8)]],   // per-token span labels (ROR)
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) { // clang-format on

  const uint seq_idx    = tid.z;
  const uint q_head_idx = tid.y;
  const uint gqa_factor = ATTN_PAGED_NUM_Q_HEADS / ATTN_PAGED_NUM_KV_HEADS;
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

  const int Q_stride_tok = int(ATTN_PAGED_NUM_Q_HEADS) * BD;
  Q += int(global_q_base) * Q_stride_tok + int(q_head_idx) * BD;
  O += int(global_q_base) * Q_stride_tok + int(q_head_idx) * BD;

  const int kv_blk_stride = int(ATTN_PAGED_NUM_KV_HEADS) * BLOCK_SIZE_ * BD;
  const int kv_head_off   = int(kv_head_idx) * (BLOCK_SIZE_ * BD);
  const device uint* row_block_table =
      block_table + seq_idx * ATTN_PAGED_MAX_BLOCKS_PER_SEQ;
  const int num_pages = int((kv_len + uint(BLOCK_SIZE_) - 1u) / uint(BLOCK_SIZE_));

  const AccumType scale2 = static_cast<AccumType>(ATTN_PAGED_SCALE) * AccumType(M_LOG2E_F);

  // ----- warp geometry -----------------------------------------------------
  constexpr short kU = 8;                  // simdgroup MMA fragment edge
  constexpr short ROWS_PER_BLOCK = 16;     // Q rows a warp's block (2 frags)
  constexpr int TQ = ROWS_PER_BLOCK / kU;  // Q-row frags a warp (2)
  constexpr int DW = BD / WN;              // head dims a warp's O covers
  constexpr int TDW = DW / kU;             // ... as fragments
  // Keys one warp's Q @ K^T takes: one page group, at most 32 (as the NAX
  // wide kernel's KQ; BLOCK_SIZE_ % KQ == 0 so a group never straddles a
  // page and the resolve stays a single block-table read). Must equal BK:
  // the exchange readback below is lane-preserving only when Stile's
  // fragment grid is Sg's (BK == KQ, one group a K-tile) — any other split
  // interleaves the two fragment orders.
  constexpr int KQ = BLOCK_SIZE_ < 32 ? BLOCK_SIZE_ : 32;
  constexpr short TK = BK / kU;            // key frags a K-tile
  constexpr int kTiles = WN * KQ / BK;     // K-tiles a step
  static_assert(
      BQ == WM * ROWS_PER_BLOCK && WN > 1,
      "one 16-row Q block a warp, WN warps a block");
  static_assert(
      BK % KQ == 0 && BLOCK_SIZE_ % KQ == 0 && KQ % kU == 0,
      "whole page groups a K-tile");
  static_assert(DW % kU == 0, "whole 8-dim fragments a warp");
  static_assert(kTiles * BK == WN * KQ, "whole K-tiles a step");

  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kU, kU>;
  using otile_t = MMATile<AccumType, TQ, TDW, MMAFrag_acc_t>; // O: 16 x DW
  using stile_t = MMATile<AccumType, TQ, TK, MMAFrag_acc_t>;  // S: 16 x BK
  using sgtile_t = MMATile<AccumType, TQ, KQ / kU, MMAFrag_acc_t>; // S: 16 x KQ

  const short rb = short(simd_group_id / WN);
  const short wn = short(simd_group_id % WN);
  const short tm = ROWS_PER_BLOCK * rb;
  const short lim_rows_q = short(int(q_tile_rows) - int(tm));
  // A warp past the tile's rows still runs its ops (its block meets every
  // step): it reads the tile's first rows and stores nothing (the NAX wide
  // kernel's `max(min(lim, kU), 1)` extent — finite S, no NaN path).
  const device T* Qw = Q + (lim_rows_q > 0 ? int(tm) : 0) * Q_stride_tok;
  O += int(tm) * Q_stride_tok + int(wn) * DW;

  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  constexpr short kRowsPT = otile_t::kRowsPerThread;
  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  otile_t Otile;
  Otile.clear();

  // Causal/window iteration bounds (absolute K-axis coords), shared by the
  // threadgroup (its warps meet every step): the Q tile rows are
  // [prefix_len + q_block_base, ... + q_tile_rows).
  const int abs_q_min = int(prefix_len) + int(q_block_base) + int(tm);
  const int abs_q_first = int(prefix_len) + int(q_block_base);
  const int abs_q_max_excl = abs_q_first + int(q_tile_rows);
  const int kv_tiles_total = int((kv_len + uint(BK) - 1u) / uint(BK));
  int kb_lim = (abs_q_max_excl + BK - 1) / BK;
  if (kb_lim > kv_tiles_total) kb_lim = kv_tiles_total;
  int kb_start = 0;
  if (ATTN_PAGED_WINDOW > 0) {
    const int first_k = abs_q_first - ATTN_PAGED_WINDOW + 1;
    if (first_k > 0) kb_start = first_k / BK;
  }
  // Block-diagonal span attention, as `attention_paged`: a span-uniform Q
  // tile starts at its span's first K-tile (ROR-gated; folds away on
  // non-spans so span_ids is never read unbound).
  if (ATTN_PAGED_ROR != 0u || ATTN_PAGED_SELFONLY != 0u) {
    const uint sf = span_ids[uint(abs_q_first)];
    const uint sl = span_ids[uint(abs_q_max_excl - 1)];
    if (sf != 0u && sf == sl) {
      const int span_kb = (int(sf) - 1) / int(BK);
      if (span_kb > kb_start) kb_start = span_kb;
    }
  }

  // The first row of the KQ-key group `g`'s page, for K (pre-roped scratch
  // for spans, else the cache: one path per pipeline, on the baked ROR) and
  // V (always the cache, never roped). Same math as the NAX wide kernel's
  // resolve_k / resolve_v.
  auto resolve_k = [&](int g) -> const device T* {
    const int lb = g / (BLOCK_SIZE_ / KQ);
    const int rows = (g % (BLOCK_SIZE_ / KQ)) * KQ * BD;
    if (ATTN_PAGED_ROR != 0u) {
      return paged_resolve_scratch<T>(
          k_scratch, lb, num_pages, kv_blk_stride, kv_head_off) + rows;
    }
    return paged_resolve_block<T>(
        k_cache, row_block_table, lb, num_pages,
        int(ATTN_PAGED_BLOCKS_PER_CHUNK), kv_blk_stride, kv_head_off) + rows;
  };
  auto resolve_v = [&](int g) -> const device T* {
    return paged_resolve_block<T>(
        v_cache, row_block_table, g / (BLOCK_SIZE_ / KQ), num_pages,
        int(ATTN_PAGED_BLOCKS_PER_CHUNK), kv_blk_stride, kv_head_off)
        + (g % (BLOCK_SIZE_ / KQ)) * KQ * BD;
  };

  // The block's S of a step, lane by lane, a buffer a step parity where two
  // fit (a warp writing step s + 2's has met its block at step s + 1, after
  // everyone read step s's), else one buffer and a second barrier. The
  // layout is lane-preserving: a lane writes its S fragments' elements
  // contiguously and reads them back at the same positions, so the
  // exchanged Stile has exactly the fragment ownership a freshly computed
  // one would (the NAX wide kernel's `sx` contract).
  constexpr int kSPerLane = sgtile_t::kElemsPerTile; // == KQ / 2
  constexpr int kPart = 32 * kSPerLane;              // S elements a warp
  constexpr int kBufs = 2 * WM * WN * kPart * int(sizeof(AccumType)) <= 32768 ? 2 : 1;
  threadgroup AccumType sx[kBufs * WM * WN * kPart];

  // ── S = Q @ K^T of page group `g` over the whole head: the WN head-dim
  //    slices' tile_matmads summed in slice order, K and Q fragments loaded
  //    straight from device memory (K row-major [keys, BD]: a K^T fragment
  //    is 8 dims x 8 keys at strides 1 / BD). Keys past the group's live
  //    rows load as 0 (a partial last page). ─────────────────────────────
  auto load_kt_frag = [&](
                          thread typename MMAFrag_acc_t::frag_type& frag,
                          const device T* Kg,
                          short k0,
                          short klim) {
    // Kg: the group's base + this thread's dim coordinate; the fragment's
    // keys are [k0, k0 + 8), this thread's two at k0 + sn, k0 + sn + 1.
    const device T* src = Kg + int(k0 + sn) * int(BD);
    if (k0 + kU <= klim) {
      MMAFrag_acc_t::load(frag, src, Int<1>{}, int(BD));
    } else {
      // Live keys from this thread's first column on (sn): j < klim - k0 - sn.
      MMAFrag_acc_t::load_safe(
          frag, src, Int<1>{}, int(BD), short(kU), short(klim - k0 - sn), 0, 0);
    }
  };

  auto do_qk_group = [&](int g, thread sgtile_t& Sg) {
    const short klim = short(
        clamp(int(kv_len) - g * KQ, 0, int(KQ)));
    const short rows = max(min(lim_rows_q, short(ROWS_PER_BLOCK)), short(1));
    STEEL_PRAGMA_UNROLL
    for (short w = 0; w < WN; w++) {
      const device T* Qb = Qw + int(w) * DW;
      const device T* Kg = resolve_k(g) + int(w * DW + sm);
      STEEL_PRAGMA_UNROLL
      for (short dd = 0; dd < DW / kU; dd++) {
        MMATile<AccumType, TQ, 1> Qt;
        MMATile<AccumType, 1, KQ / kU> Kt;
        // Q fragment tile: 16 rows x 8 dims at [w*DW + dd*8, +8); rows past
        // `rows` load as 0.
        const int qoff = int(sm) * Q_stride_tok + int(dd * kU + sn);
        if (rows == ROWS_PER_BLOCK) {
          Qt.template load<T, 1, 1>(Qb + qoff, Q_stride_tok);
        } else {
          Qt.template load_safe<T, 1, 1>(
              Qb + qoff, Q_stride_tok, short2(kU, short(rows - sm)));
        }
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < KQ / kU; j++) {
          load_kt_frag(Kt.frag_at(0, j), Kg + int(dd * kU), short(j * kU), klim);
        }
        tile_matmad(Sg, Qt, Kt, Sg);
      }
    }
  };

  // ── scale + the partial tail / causal / window masks on Stile
  //    (register-only; per-element r/c exactly as the NAX wide kernel). ──
  auto scale_and_mask = [&](int kb_, thread stile_t& Stile) {
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale2;
    }

    const int key0 = kb_ * BK;
    const bool tail = key0 + BK > int(kv_len);
    const bool causal = key0 + BK - 1 > abs_q_min;
    const bool window =
        ATTN_PAGED_WINDOW > 0 && (abs_q_max_excl - 1 - key0) >= ATTN_PAGED_WINDOW;
    if (tail || causal || window) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            const int r = abs_q_min + sm + iq * kU;
            const int c = key0 + ik * kU + sn + jj;
            const bool dead = c >= int(kv_len) || r < c ||
                (ATTN_PAGED_WINDOW > 0 && (r - c) >= ATTN_PAGED_WINDOW);
            fg[jj] = dead ? neg_inf : fg[jj];
          }
        }
      }
    }
  };

  // ── online-softmax update of (max, sum, Otile) from the masked Stile ──
  auto softmax_update = [&](thread stile_t& Stile) {
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];

    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }
    Stile.template row_reduce<SumOp>(sum_score);
    Otile.template row_bin_op<MulOp>(factor);
  };

  // ── O += P @ V for K-tile kb_ (V fragments straight from the device
  //    cache, one 8x8 at a time, loaded once per (D-frag, key-frag) and
  //    mma'd into every Q-row frag; keys past kv_len load as 0 so a masked
  //    P = 0 cannot hit an uninitialized V element). ─────────────────────
  auto do_pv = [&](int kb_, thread stile_t& Stile) {
    const int key0 = kb_ * BK;
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDW; id++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        const int k_abs = key0 + int(ik) * kU;
        const int g = k_abs / KQ;
        const short k_in_g = short(k_abs % KQ);
        const device T* Vp =
            resolve_v(g) + int(wn) * DW + int(id * kU + sn);
        typename MMAFrag_acc_t::frag_type Vf;
        const device T* vsrc = Vp + int(k_in_g + sm) * int(BD);
        if (k_abs + kU <= int(kv_len)) {
          MMAFrag_acc_t::load(Vf, vsrc, int(BD), Int<1>{});
        } else {
          MMAFrag_acc_t::load_safe(
              Vf, vsrc, int(BD), Int<1>{},
              short(clamp(int(kv_len) - k_abs - sm, 0, int(kU))), short(kU),
              0, 0);
        }
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vf,
              Otile.frag_at(iq, id));
        }
      }
    }
  };

  // ----- KV loop: a step is WN page groups (kTiles K-tiles) ───────────────
  for (int kb0 = kb_start; kb0 < kb_lim; kb0 += kTiles) {
    // This warp's page group: S = Σ_w (0 + Q_w K_w^T), in slice order.
    const int g = kb0 * (BK / KQ) + wn;
    sgtile_t Sg;
    Sg.clear();
    // A group past the causal limit or kv_len is never read back — skip its
    // loads/mmas (Sg stays 0; the exchange write below is unconditional so
    // every warp of the block meets every step).
    if (g * KQ < int(kv_len) && g * KQ < kb_lim * BK) {
      do_qk_group(g, Sg);
    }
    threadgroup AccumType* part =
        sx + ((kb0 - kb_start) / kTiles % kBufs) * (WM * WN * kPart);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kSPerLane; i++) {
      part[(simd_group_id * kSPerLane + i) * 32 + simd_lane_id] = Sg.elems()[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short t = 0; t < kTiles; t++) {
      const int kb = kb0 + t;
      if (kb >= kb_lim) break;
      stile_t Stile;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kElemsPerTile; i++) {
        Stile.elems()[i] =
            part[(rb * WN * kSPerLane + t * (BK / 2) + i) * 32 + simd_lane_id];
      }

      scale_and_mask(kb, Stile);
      softmax_update(Stile);
      do_pv(kb, Stile);
    }
    if (kBufs == 1) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // ----- normalize + store -------------------------------------------------
  AccumType rcp[kRowsPT];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }
  Otile.template row_bin_op<MulOp>(rcp);

  // O sits at (tm + sm) rows / (wn*DW + sn) cols; the store's tile offsets
  // are relative to that (the classic kernel's pre-offset convention).
  O += int(sm) * Q_stride_tok + int(sn);
  if (lim_rows_q < ROWS_PER_BLOCK) {
    const short2 dims = short2(DW - sn, lim_rows_q - sm);
    if (dims.x <= 0 || dims.y <= 0) return;
    Otile.template store_safe<T, 1, 1>(O, Q_stride_tok, dims);
  } else {
    Otile.template store<T, 1, 1>(O, Q_stride_tok);
  }
}
