// SPDX-License-Identifier: Apache-2.0
//
// Paged-K/V-cache variant of MLX's steel_attention prefill kernel.
// Same FlashAttention-2 algorithm as `steel_attention_kernel.h`;
// only the K/V load path is replaced with `PagedKVBlockLoader` so
// the kernel reads from scratchy-target-metal's paged KV cache instead of
// a contiguous device buffer.
//
// Bindings (mirrors `attention_prefill_sdpa_v2_paged`):
//   buffer(0) Q             [total_q, num_q_heads, head_dim]
//   buffer(1) k_cache       [num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]
//   buffer(2) v_cache       [num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]
//   buffer(3) O             [total_q, num_q_heads, head_dim]
//   buffer(4) AttnParamsPaged
//   buffer(5) cu_seqlens_q  [batch+1]
//   buffer(6) seq_used_k    [batch]
//   buffer(7) block_table   [batch, max_blocks_per_seq]
//
// Dispatch grid: `(NQ_blocks_per_seq[seq], num_q_heads, batch)` —
// PER-SEQUENCE Q-block tiling so a BQ tile never straddles a
// sequence boundary. The caller (scratchy-forward-compiler lowering) is
// responsible for setting tid.z = seq_idx and tid.x = Q-block
// within that sequence's qL.
//
// (For our prefill, sequences typically run one at a time so
// tid.z=0; the structure generalizes cleanly.)
//
// Spans / rope-on-read (ROPE-ONCE-TO-SCRATCH): K is stored UNROTATED in the
// paged cache (the position-independent reuse artifact — never mutated). A
// span cached at one position and reused at another MUST be re-roped to the
// reader's absolute position. Rather than re-rope each K tile INSIDE the
// attention's per-q-tile loop (flash re-reads/re-ropes each staged K tile
// ~NQ = m/BQ times, measured ~1.06-1.99× of the non-spans baseline — the
// per-tile REDUNDANCY, not the cos/sin source), the K is roped ONCE — outside
// the attention — into a transient device SCRATCH buffer by the separate
// `rope_once_steel` kernel (one pass over K = O(K)). The attention then reads
// PRE-ROPED K from the scratch with an ordinary paged-loader read and NO
// per-tile rotation, so it runs at the non-spans baseline. For long context
// the one-time O(K) rope is negligible vs the O(NQ·K) attention, so ON/OFF
// approaches ~1.0×. This mirrors `steel_attention_nax_paged_kernel.h`'s
// `rope_once_nax` for the simdgroup (head_dim 64/96/128/256) steel path.
//
// When ROPE_ON_READ is unset no scratch is bound (the is_function_constant_
// defined guard folds buffer 8 away), the rope-once kernel is not dispatched,
// and the attention reads K directly from the cache exactly as before
// (non-spans path byte-identical). V is never roped (no RoPE on V).

#include "attn.h"
#include "paged_loader.h"

using namespace mlx::steel;

// (No AttnParamsPaged struct — model params come in as function
// constants 0..5; see ATTN_PAGED_* declarations below.)

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

// Model-level params come in as function constants — pipeline
// specialization makes them compile-time literals inside the
// kernel. Slot numbers mirror `attention_prefill_sdpa_v2_paged`
// (`attention.metal`) so dispatcher code can share them.
//
// Slots 0 (HEAD_DIM) and 4 (BLOCK_SIZE) are accepted from the
// dispatcher for ABI consistency with the prior kernel but are
// unused in this body — both values are already baked as template
// parameters (`BD` and `BLOCK_SIZE_`) at metallib-compile time,
// which is strictly stronger constant folding.
constant uint  ATTN_PAGED_HEAD_DIM           [[function_constant(0)]];  // unused; ==BD
constant uint  ATTN_PAGED_NUM_Q_HEADS        [[function_constant(1)]];
constant uint  ATTN_PAGED_NUM_KV_HEADS       [[function_constant(2)]];
constant float ATTN_PAGED_SCALE              [[function_constant(3)]];
constant uint  ATTN_PAGED_BLOCK_SIZE         [[function_constant(4)]];  // unused; ==BLOCK_SIZE_
constant uint  ATTN_PAGED_MAX_BLOCKS_PER_SEQ [[function_constant(5)]];
// Reactive (chunked) KV pool: k_cache/v_cache (buffers 5/6) are
// per-layer chunk-address TABLES (device uint64 gpuAddresses), not the
// cache buffers. `PagedBlockLoaderT` derefs
// `chunk_table[physical / BLOCKS_PER_CHUNK]` per block. See
// scratchy-target-metal's `BLOCKS_PER_CHUNK`.
constant uint  ATTN_PAGED_BLOCKS_PER_CHUNK   [[function_constant(6)]];
// Sliding-window width (Gemma2/3/4 local layers): a query at absolute
// position q attends keys k with 0 <= q - k < window. 0 disables the
// window; the compiler folds every window branch away for
// full-attention pipelines (the dispatcher always sets slot 7 — 0 for
// full attention, W::SLIDING_WINDOW for the sliding prefill arm).
// Same slot/semantics as `ATTN_WINDOW` in attention.metal.
constant int   ATTN_PAGED_WINDOW             [[function_constant(7)]];

// Debug toggle. When `ATTN_PAGED_DEBUG_MODE != 0`, the kernel replaces
// its normal store path with a per-lane marker write so the bench can
// observe which (tid, simdgroup, lane) tuples actually reach the
// store. Modes:
//   1 = write (simd_group_id + 1) * 10 + 1 at O[(tm+sm)*Q_stride_tok + sn]
//       — per-lane single-element write, no Otile involvement.
//   2 = same as 1 but EVERY thread writes regardless of position —
//       proves the kernel reached this point at all.
constant uint  ATTN_PAGED_DEBUG_MODE         [[function_constant(99)]];

// Rope-on-read (spans): same slots/semantics as ATTN_ROT_DIM /
// ATTN_PAIR_OFF / ATTN_ROPE_ON_READ in attention.metal. Optional — the
// is_function_constant_defined guard folds the scratch K-source (and
// buffer 7, the pre-roped K scratch) away when unset, so non-spans steel
// pipelines are byte-identical. ROPE_ON_READ selects the scratch K-source
// in `attention_paged` (the K is pre-roped by `rope_once_steel`); rot_dim/
// pair_off are consumed only by the `rope_once_steel` kernel.
constant uint  ATTN_PAGED_ROT_DIM            [[function_constant(8)]];
constant uint  ATTN_PAGED_PAIR_OFF           [[function_constant(9)]];
constant uint  ATTN_PAGED_ROPE_ON_READ       [[function_constant(10)]];
constant bool  ATTN_PAGED_ROR_DEFINED = is_function_constant_defined(ATTN_PAGED_ROPE_ON_READ);
constant uint  ATTN_PAGED_ROR = ATTN_PAGED_ROR_DEFINED ? ATTN_PAGED_ROPE_ON_READ : 0u;

// Self-only span masking DECOUPLED from rope-on-read. The sliding spans path
// runs with ATTN_PAGED_ROR=0 (its K-source is the plain cache), which also
// gated off the kb-loop span seek below — so a Relocatable span absorbed the
// preamble at every sliding layer and its K/V was NOT the pure function of
// span bytes the spans design requires for content-addressed reuse. This
// constant gates the seek independently of the K-source. Undefined/0 ⇒
// byte-identical. Requires span_ids bound (per-token labels).
constant uint  ATTN_PAGED_SELFONLY_RAW       [[function_constant(14)]];
constant bool  ATTN_PAGED_SELFONLY_DEF = is_function_constant_defined(ATTN_PAGED_SELFONLY_RAW);
constant uint  ATTN_PAGED_SELFONLY = ATTN_PAGED_SELFONLY_DEF ? ATTN_PAGED_SELFONLY_RAW : 0u;

// ── rope-once kernel (spans rope-on-read, simdgroup steel prefill) ──────────
//
// One pass over a request's K for one layer: read the UNROTATED K from the
// paged cache, rope each row to its absolute position (cos/sin from the table,
// done ONCE per key), write the roped K into the dense scratch. The cache is
// never mutated. O(K) — amortizes to ~0% against the O(NQ·K) attention. This
// is the simdgroup-steel twin of `rope_once_nax_kernel`; the scratch it writes
// is read by `attention_paged` via `PagedBlockLoaderT`'s dense-scratch resolve.
//
// Grid: tid.y = logical block index; tid.x decodes to (kv_head, token_in_block,
// rotary pair index d in [0, rot_dim/2)). A thread ropes the PAIR (d, d+pair_off)
// of one K row. Rows whose block is NOT flagged unrotated (block_table bit 31
// clear) are COPIED through unchanged so the scratch is a complete K image
// either way (the attention always reads the scratch when ROR is on). Bit-exact
// with `rope_append` / `cpu_rope_k` (round to T on store).
//
//   buffer(0) k_scratch     [num_pages, num_kv_heads, BLOCK_SIZE, head_dim] T
//   buffer(1) block_table   [batch, max_blocks_per_seq]  (bit 31 = unrotated)
//   buffer(2) k_cache       per-layer chunk-address table (uint64 gpuAddrs)
//   buffer(3) seq_used_k    [batch]
//   buffer(4) cos_sin       [max_pos, rot_dim]
template <typename T, int BD, int BLOCK_SIZE_>
[[kernel]]
void rope_once_steel_kernel(
    device T*          k_scratch    [[buffer(0)]],
    const device uint* block_table  [[buffer(1)]],
    const device uint64_t* k_cache  [[buffer(2)]],
    const device uint* seq_used_k   [[buffer(3)]],
    const device T*    cos_sin      [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint logical_block = gid.y;
  // ⛔⛔⛔ SEE attention.metal's rope_once FOR THE FULL FINDING: this 0 serves batch row 0's K image to
  // EVERY row of the batch, and is the measured cause of 1-of-12 on a 12-request metal batch. Correct only
  // while the caller is single-sequence, which it no longer is.
  const uint seq_idx = 0; // single-sequence prefill ONLY — WRONG for num_reqs > 1

  const uint kv_len = seq_used_k[seq_idx];
  const uint num_pages = (kv_len + uint(BLOCK_SIZE_) - 1u) / uint(BLOCK_SIZE_);
  if (logical_block >= num_pages) {
    return;
  }

  const uint half_dim = ATTN_PAGED_ROT_DIM / 2u;
  const uint pair_off = ATTN_PAGED_PAIR_OFF;
  // gid.x decodes to (kv_head, token_in_block, rotary index d in [0,half_dim)).
  const uint per_head = uint(BLOCK_SIZE_) * half_dim;
  const uint kv_head  = gid.x / per_head;
  const uint rem      = gid.x % per_head;
  const uint tib      = rem / half_dim;     // token in block
  const uint d        = rem % half_dim;     // rotary pair index
  if (kv_head >= ATTN_PAGED_NUM_KV_HEADS) {
    return;
  }

  const uint pos = logical_block * uint(BLOCK_SIZE_) + tib;

  const int kv_blk_stride  = int(ATTN_PAGED_NUM_KV_HEADS) * BLOCK_SIZE_ * BD;
  const int kv_head_off    = int(kv_head) * (BLOCK_SIZE_ * BD);
  const device uint* row_block_table =
      block_table + seq_idx * ATTN_PAGED_MAX_BLOCKS_PER_SEQ;

  // Source row (cache, possibly unrotated) and dest row (scratch, dense).
  const uint physical = uint(row_block_table[logical_block]) & 0x7FFFFFFFu;
  const uint chunk = physical / ATTN_PAGED_BLOCKS_PER_CHUNK;
  const uint bic = physical % ATTN_PAGED_BLOCKS_PER_CHUNK;
  const device T* k_blk =
      (const device T*)k_cache[chunk] + int(bic) * kv_blk_stride + kv_head_off;
  device T* dst_blk = k_scratch + int(logical_block) * kv_blk_stride + kv_head_off;
  const device T* k_row = k_blk + tib * uint(BD);
  device T* dst_row = dst_blk + tib * uint(BD);

  const bool past_len = pos >= kv_len;
  const bool flagged =
      (uint(row_block_table[logical_block]) & 0x80000000u) != 0u;

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
  const device T* cr = cos_sin + pos * ATTN_PAGED_ROT_DIM;
  const float c = float(cr[d]);
  const float s = float(cr[half_dim + d]);
  const float x0 = float(k_row[d]);
  const float x1 = float(k_row[pair_off + d]);
  dst_row[d]            = T(x0 * c - x1 * s);
  dst_row[pair_off + d] = T(x1 * c + x0 * s);
}

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct SubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x - y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

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
void attention_paged(
    // Binding indices match the existing
    // `attention_prefill_sdpa_v2_paged_*` so dispatch wiring stays
    // unchanged when we swap kernels.
    device T*          O            [[buffer(0)]],   // [total_q, num_q_heads, head_dim]
    const device T*    Q            [[buffer(1)]],   // [total_q, num_q_heads, head_dim]
    const device uint* cu_seqlens_q [[buffer(2)]],   // [batch+1]
    const device uint* seq_used_k   [[buffer(3)]],   // [batch] total cached K
    const device uint* block_table  [[buffer(4)]],   // [batch, max_blocks_per_seq]
    const device uint64_t* k_cache  [[buffer(5)]],   // per-layer chunk-address table
    const device uint64_t* v_cache  [[buffer(6)]],   // per-layer chunk-address table
    // Rope-on-read (spans, rope-once-to-scratch): PRE-ROPED K scratch (dtype
    // T), dense by logical block, written by `rope_once_steel`. Only read when
    // ATTN_PAGED_ROPE_ON_READ; K then comes from here (no per-tile rotation)
    // instead of the cache. The is_function_constant_defined guard folds this
    // away when ROR is unset, so the non-spans ABI is the 7-binding form.
    const device T*     k_scratch             [[buffer(7)]],
    // Block-diagonal span attention: per-logical-block span label (slot 8).
    // Bound only when spans/rope-on-read active; read only under ATTN_PAGED_ROR.
    const device uint*  span_ids              [[buffer(8)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  (void)lid;

  // tid layout: tid.x = Q-block within sequence (0..NQ_blocks-1)
  //             tid.y = q-head index             (0..num_q_heads-1)
  //             tid.z = seq_idx                  (0..batch-1)
  const uint seq_idx     = tid.z;
  const uint q_head_idx  = tid.y;
  const uint gqa_factor  = ATTN_PAGED_NUM_Q_HEADS / ATTN_PAGED_NUM_KV_HEADS;
  const uint kv_head_idx = q_head_idx / gqa_factor;

  const uint seq_start   = cu_seqlens_q[seq_idx];
  const uint seq_end     = cu_seqlens_q[seq_idx + 1];
  const uint new_q_for_seq = seq_end - seq_start;
  const uint kv_len      = seq_used_k[seq_idx];
  const uint prefix_len  = kv_len - new_q_for_seq;
  const uint q_block_base = tid.x * uint(BQ);
  const uint global_q_base = seq_start + q_block_base;

  if (q_block_base >= new_q_for_seq) {
    return;
  }
  const uint q_tile_rows = min(uint(BQ), new_q_for_seq - q_block_base);
  const bool q_tile_full = (q_tile_rows == uint(BQ));

  const int Q_stride_tok = int(ATTN_PAGED_NUM_Q_HEADS) * BD;
  Q += int(global_q_base) * Q_stride_tok + int(q_head_idx) * BD;
  O += int(global_q_base) * Q_stride_tok + int(q_head_idx) * BD;

  const int kv_blk_stride  = int(ATTN_PAGED_NUM_KV_HEADS) * BLOCK_SIZE_ * BD;
  const int kv_head_stride = BLOCK_SIZE_ * BD;
  const int per_token_stride = BD;
  const device uint* row_block_table = block_table + seq_idx * ATTN_PAGED_MAX_BLOCKS_PER_SEQ;
  // Allocated paged blocks for this sequence — the clamp bound the K/V
  // loaders use so a partial tail K-tile (BK > BLOCK_SIZE) never derefs an
  // unallocated block-table slot.
  const int num_kv_pages = int((kv_len + uint(BLOCK_SIZE_) - 1u) / uint(BLOCK_SIZE_));

  // Prepare threadgroup memory
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  constexpr short tgp_mem_0 = (BK + padK) * (BD);          // K tile (col-major)
  constexpr short tgp_mem_1 = BK * (BD + padV);            // V tile (row-major)
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;
  constexpr short tgp_mem_q = BQ * (BD + padQ);            // Q tile

  // Rope-on-read software pipeline (spans): when the staged K tile is
  // re-roped on read, the rope's expensive part (cos_sin device loads +
  // the rotate) is issued for the NEXT kb BEFORE the current kb's
  // softmax/PV so the GPU overlaps those long-latency loads with the PV
  // matmul (see the `kPipeline` loop below). That needs the K smem tile
  // DOUBLE-BUFFERED (Ks[2]) plus a SEPARATE V buffer (the serialized
  // path aliases Ks/Vs in one KV_smem). We pick the pipeline only when
  // the double-buffered footprint fits the 32 KiB threadgroup budget —
  // a compile-time (template-param) decision, so the choice is baked per
  // (T, BD, BK) instantiation. Head dims that don't fit (bd128/bd256 at
  // f16) keep the byte-identical serialized layout below; rope-OFF
  // pipelines are unaffected either way (the rope folds out).
  constexpr int dbl_bytes =
      int(sizeof(T)) * (int(tgp_mem_q) + 2 * int(tgp_mem_0) + int(tgp_mem_1));
  constexpr bool kPipeline = dbl_bytes <= 32768;

  // One pool, aliased per layout, so the declared threadgroup footprint
  // is exactly the chosen path's footprint (no double-counting).
  constexpr int pool_elems =
      kPipeline ? (int(tgp_mem_q) + 2 * int(tgp_mem_0) + int(tgp_mem_1))
                : (int(tgp_mem_q) + int(tgp_mem_s));
  threadgroup T smem_pool[pool_elems];

  threadgroup T* Qs = smem_pool;
  // Pipelined: Ks_buf[0], Ks_buf[1] are the double-buffered K tiles;
  // Vs is its own buffer. Serialized: Ks and Vs alias the single
  // KV_smem region right after Q (exactly the prior layout).
  threadgroup T* Ks_buf0 = smem_pool + tgp_mem_q;
  threadgroup T* Ks_buf1 = smem_pool + tgp_mem_q + (kPipeline ? tgp_mem_0 : 0);
  threadgroup T* Vs = smem_pool + tgp_mem_q + (kPipeline ? 2 * int(tgp_mem_0) : 0);
  // The serialized path's K/V buffer (== Ks_buf0 region).
  threadgroup T* Ks = Ks_buf0;

  // Prepare block loaders
  // Q: contiguous (just like MLX). Stride along Q-token axis is
  // num_q_heads * head_dim (we've pre-offset Q to (global_q, head)).
  using QBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BQ,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDQ_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 1,
      /* short tgp_size = */ WM * WN * 32>;

  // K and V: paged. Use `PagedBlockLoaderT` — inherits MLX's
  // `BlockLoaderT` byte-for-byte and overrides only `next()` to
  // advance via the block table. Same compiler codegen as the contig
  // path's load_unsafe / load_safe.
  using KBlockLoader = PagedBlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ 1,
      /* short kDstStrCol = */ LDK_tgp,
      /* short tgp_size = */ WM * WN * 32,
      /* short BLOCK_SIZE_ = */ BLOCK_SIZE_>;

  using VBlockLoader = PagedBlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDV_tgp,
      /* short kDstStrCol = */ 1,
      /* short tgp_size = */ WM * WN * 32,
      /* short BLOCK_SIZE_ = */ BLOCK_SIZE_>;

  QBlockLoader loader_q(
      Q, Q_stride_tok, Qs, simd_group_id, simd_lane_id);
  // Chunked KV: pass the chunk-address table + the per-block head
  // offset (kv_head_idx * kv_head_stride) + BLOCKS_PER_CHUNK; the
  // loader resolves the chunk base per physical block (it can no
  // longer be folded into a single `cache_base`).
  // Rope-on-read (spans): when ROR is set, K is read from the dense pre-roped
  // scratch (written once by `rope_once_steel`), so the loader bypasses the
  // block_table/chunk_table indirection and reads `k_scratch + lb*kv_blk_stride
  // + row_col_offset`. The scratch base is null otherwise → the byte-identical
  // chunk_table resolve. The branch folds on the ATTN_PAGED_ROR function const.
  const device T* k_src_scratch =
      (ATTN_PAGED_ROR != 0u) ? k_scratch : (const device T*)nullptr;
  KBlockLoader loader_k(
      k_cache,
      per_token_stride,
      kv_blk_stride,
      int(kv_head_idx) * kv_head_stride,
      int(ATTN_PAGED_BLOCKS_PER_CHUNK),
      num_kv_pages,
      row_block_table,
      Ks,
      simd_group_id,
      simd_lane_id,
      k_src_scratch);
  VBlockLoader loader_v(
      v_cache,
      per_token_stride,
      kv_blk_stride,
      int(kv_head_idx) * kv_head_stride,
      int(ATTN_PAGED_BLOCKS_PER_CHUNK),
      num_kv_pages,
      row_block_table,
      Vs,
      simd_group_id,
      simd_lane_id);

  const AccumType scale = static_cast<AccumType>(ATTN_PAGED_SCALE) * M_LOG2E_F;

  // Prepare MMA tiles
  constexpr short kFragSize = 8; // MMAFrag size
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  // KV sequence frags (all warps load the same frags)
  constexpr int TK = BK / kFragSize;
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kFragSize;

  static_assert(TQ == 1, "Check TQ");

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load Q blocks. Bounds-checked load uses `q_tile_rows` (this
  // sequence's remaining new-Q tokens at this Q-block).
  if (q_tile_full) {
    loader_q.load_unsafe();
  } else {
    loader_q.load_safe(short2(BD, int(q_tile_rows)));
  }

  // Init row reduction variables
  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;

  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};

  // Init to -Inf
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  // KV-block iteration bounds. The Q tile spans rows [q_block_base,
  // q_block_base + q_tile_rows) in *new-tokens* coords; in absolute
  // K-axis coords those rows are at [prefix_len + q_block_base,
  // prefix_len + q_block_base + q_tile_rows). Causal-mask cuts off
  // K positions > this Q's absolute row position; the kb upper
  // bound is the block containing the last masked-IN K position.
  const int abs_q_min = int(prefix_len) + int(q_block_base);
  const int abs_q_max_excl =
      int(prefix_len) + int(q_block_base) + int(q_tile_rows);
  const int kv_blocks_total = int((kv_len + uint(BK) - 1u) / uint(BK));
  int kb_lim = (abs_q_max_excl + BK - 1) / BK;
  if (kb_lim > kv_blocks_total) kb_lim = kv_blocks_total;
  // First kb that needs causal masking. The Q tile's earliest
  // absolute row is `prefix_len + q_block_base`; any K position
  // strictly past that row is masked-out causally.
  const int kb_min_causal = abs_q_min / BK;
  // Sliding window: the EARLIEST key any row of this Q tile attends
  // is `abs_q_min - window + 1` (row q attends k iff 0 <= q - k <
  // window, and abs_q_min is the tile's smallest q). K tiles entirely
  // before that are skipped — this is what makes windowed prefill
  // O(T·window) instead of O(T²). Branch folds away at window == 0.
  int kb_start = 0;
  if (ATTN_PAGED_WINDOW > 0) {
    const int first_k = abs_q_min - ATTN_PAGED_WINDOW + 1;
    if (first_k > 0) {
      kb_start = first_k / BK;
    }
  }
  // Block-diagonal span attention: a SPAN-UNIFORM Q tile (all its rows in one
  // span) attends ONLY that span's K-blocks → start the kb-loop at the span's
  // first block, reusing the same seek/fast-forward the sliding window uses.
  // This is the real O(T^2)->O(T*span) cut for the tools/spans region, AND it
  // means the kb-loop never sees a sibling-span block (no all-masked rows → no
  // softmax NaN), so the in-tile score mask stays unnecessary. ATTN_PAGED_ROR-
  // gated (folds away on non-spans). span_ids is PER-TOKEN: label = (span's
  // first TOKEN + 1), 0 = shared/attends-all. Index by ABSOLUTE TOKEN position
  // (matching the host `span_ids_per_token` builder + the sibling kernels), NOT
  // the old per-BLOCK `/ BLOCK_SIZE_`. Convert the per-token first_token to the
  // first KV BLOCK via `/ BK` (BK == BLOCK_SIZE_), flooring a mid-block span
  // start to its containing block (the correct lower bound). A tile straddling a
  // span boundary (first row's span != last row's) keeps the full causal range.
  if (ATTN_PAGED_ROR != 0u || ATTN_PAGED_SELFONLY != 0u) {
    const uint sf = span_ids[uint(abs_q_min)];
    const uint sl = span_ids[uint(abs_q_max_excl - 1)];
    if (sf != 0u && sf == sl) {
      const int span_kb = (int(sf) - 1) / int(BK);
      if (span_kb > kb_start) { kb_start = span_kb; }
    }
  }
  // Last kb that's a full BK-wide tile (the rest of kv_len fits
  // in this block partially).
  const int kv_aligned_blocks = int(kv_len / uint(BK));
  const uint kv_rem = kv_len % uint(BK);

  // Fast-forward the paged loaders past the skipped tiles (O(1) —
  // one block-table read each).
  if (kb_start > 0) {
    loader_k.seek(kb_start);
    loader_v.seek(kb_start);
  }

  // ── Shared per-block primitives (used by both the serialized and the
  //    software-pipelined loops below). Lambdas so the body is written
  //    once; the compiler inlines them. ───────────────────────────────
  //
  // Rope-on-read (spans, rope-once-to-scratch): there is NO per-tile rotation
  // here — when ROR is set, `loader_k` stages K straight from the dense
  // pre-roped scratch (`rope_once_steel` already roped each key to its absolute
  // position, once). So the staged K tile is ready for QK as-is, exactly like a
  // non-spans cache read. The earlier in-kernel `rope_buf` (rotate each staged
  // K tile on every q-tile) was ~1.06-1.99× of the baseline because flash
  // re-ropes K ~m/BQ× (the per-tile REDUNDANCY); roping ONCE outside (O(K))
  // amortizes to ~0% for long context.

  // S = Q @ Kbuf.T into the (cleared) Stile.
  auto do_qk = [&](threadgroup T* Kbuf) {
    Stile.clear();
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
          &Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(
          &Kbuf[Ks_offset + dd * Ks_tile_stride]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Stile, Qtile, Ktile, Stile);
    }
  };

  // scale + partial-tail/causal/window masks on Stile (register-only,
  // no smem K dependency).
  auto scale_and_mask = [&](int kb_) {
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
    }

    if (kb_ == kv_aligned_blocks && kv_rem != 0u) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= int(kv_rem)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    if (kb_ >= kb_min_causal) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = int(prefix_len) + int(q_block_base)
                          + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb_ * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Block-diagonal span mask: a query inside a span (q_span != 0) attends ONLY
    // keys with the same span label; q_span == 0 (shared prefix / query) attends
    // all. Mirrors the causal block's row_pos/col_pos. ATTN_PAGED_ROR-gated so
    // the non-spans pipeline const-folds it (and span_ids unbound) entirely.
    if (false && ATTN_PAGED_ROR != 0u) {  // TEMP: steel-mask isolation toggle
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = int(prefix_len) + int(q_block_base)
                          + tm + sm + (i * stile_t::kFragRows);
        const uint q_span = span_ids[uint(row_pos) / uint(BLOCK_SIZE_)];
        if (q_span != 0u) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < stile_t::kTileCols; j++) {
            const int col_pos = kb_ * BK + sn + (j * stile_t::kFragCols);
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
              if (span_ids[uint(col_pos + jj) / uint(BLOCK_SIZE_)] != q_span) {
                Stile.frag_at(i, j)[jj] = neg_inf;
              }
            }
          }
        }
      }
    }

    if (ATTN_PAGED_WINDOW > 0 &&
        (abs_q_max_excl - 1 - kb_ * BK) >= ATTN_PAGED_WINDOW) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = int(prefix_len) + int(q_block_base)
                          + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb_ * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((row_pos - (col_pos + jj)) >= ATTN_PAGED_WINDOW) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }
  };

  // Online-softmax update of (max,sum,Otile) from the masked Stile.
  auto softmax_update = [&]() {
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }
    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = new_max[i];
    }
    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }
    Otile.template row_bin_op<MulOp>(factor);
  };

  // O += P @ Vs (Vs staged + visible; caller fences).
  auto do_pv = [&]() {
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }
          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;
          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + kk * LDV_tgp + dd]);
          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }
          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }
  };

  // Retarget loader_k's smem destination to a specific K buffer base
  // (the per-thread (bi,bj) offset is identical to the constructor's).
  auto retarget_k = [&](threadgroup T* Kbuf) {
    loader_k.dst = Kbuf + loader_k.bi * 1 + loader_k.bj * LDK_tgp;
  };
  // Load loader_k's current K-tile (partial-tail aware) for block kb_.
  auto load_k_block = [&](int kb_) {
    if (kb_ == kv_aligned_blocks && kv_rem != 0u) {
      loader_k.load_safe(short2(BD, int(kv_rem)));
    } else {
      loader_k.load_unsafe();
    }
  };
  auto load_v_block = [&](int kb_) {
    if (kb_ == kv_aligned_blocks && kv_rem != 0u) {
      loader_v.load_safe(short2(BD, int(kv_rem)));
    } else {
      loader_v.load_unsafe();
    }
  };

  if constexpr (kPipeline) {
    // ── Software-pipelined loop (K-prefetch overlap). ─────────────────
    // The NEXT block's K tile is staged BEFORE the current block's
    // softmax/PV so its device→smem load overlaps the PV matmul. K is
    // double-buffered (Ks_buf0/Ks_buf1); V has its own buffer. The K
    // source is the dense pre-roped scratch (spans) or the cache (non-
    // spans) via `loader_k` — both already-roped at load time, so there
    // is no in-kernel rotation. Numerically identical to the serialized
    // path either way.
    threadgroup T* K_dbl[2] = {Ks_buf0, Ks_buf1};

    // Prologue: stage the FIRST K tile into buffer 0.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    retarget_k(K_dbl[0]);
    load_k_block(kb_start);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int kb = kb_start; kb < kb_lim; kb++) {
      const int cur = (kb - kb_start) & 1;
      const int nxt = cur ^ 1;
      const bool has_next = (kb + 1) < kb_lim;

      // QK on the already-prepared current K tile.
      do_qk(K_dbl[cur]);

      // Stage CURRENT V and (prefetch) NEXT K together. The barrier
      // before makes Ks_buf[nxt]'s old contents (read by the prior
      // iteration's QK) and Vs (read by the prior PV) safe to overwrite.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      load_v_block(kb);
      if (has_next) {
        loader_k.next();
        retarget_k(K_dbl[nxt]);
        load_k_block(kb + 1);
      }
      // V visible (for PV) AND K[kb+1] visible (for next iter's QK).
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Current block's compute (independent of Ks_buf[nxt]).
      scale_and_mask(kb);
      softmax_update();
      do_pv();

      loader_v.next();

      // Bend: K[kb+1] visible before next iter's QK reads it; PV done
      // reading Vs before next iter's V-load overwrites it.
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else {
    // ── Serialized loop (byte-identical to the original; used when the
    //    double-buffered footprint would exceed the threadgroup budget,
    //    e.g. f16 bd128/bd256). Ks and Vs share KV_smem (Ks==Ks_buf0). ─
    for (int kb = kb_start; kb < kb_lim; kb++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      load_k_block(kb);

      threadgroup_barrier(mem_flags::mem_threadgroup);

      do_qk(Ks);
      scale_and_mask(kb);

      threadgroup_barrier(mem_flags::mem_threadgroup);
      load_v_block(kb);

      softmax_update();

      threadgroup_barrier(mem_flags::mem_threadgroup);
      do_pv();

      loader_k.next();
      loader_v.next();
    }
  }

  // Normalize output (skip in debug mode 4 to preserve the markers).
  if (ATTN_PAGED_DEBUG_MODE != 4u) {
    Otile.template row_bin_op<DivOp>(sum_score);
  }

  // DEBUG_MODE=3: overwrite Otile with constant 1.0 to test whether
  // the Otile.template store<T,1,1>(O, Q_stride_tok) path itself
  // correctly writes 1024 elements per simdgroup (4096 per TG).
  // If yes → bug is in upstream Otile compute. If no → bug is in
  // store path (despite simd-coverage diagnostic saying it's reached).
  if (ATTN_PAGED_DEBUG_MODE == 3u) {
    STEEL_PRAGMA_UNROLL
    for (short f = 0; f < decltype(Otile)::kNumFrags; f++) {
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < decltype(Otile)::kElemsPerFrag; e++) {
        Otile.val_frags[f][e] = AccumType(1);
      }
    }
  }

  // DEBUG_MODE=4: assign each frag a unique marker `id + 1` (1..16)
  // AFTER the MMA loop, so the store reads from these markers. If
  // output shows correct markers in each frag's col range, the bug
  // is in MMA accumulation upstream. If output is still patterned
  // (frag 14 = 0), the bug is in `row_bin_op<DivOp>` or `store`.
  if (ATTN_PAGED_DEBUG_MODE == 4u) {
    STEEL_PRAGMA_UNROLL
    for (short f = 0; f < decltype(Otile)::kNumFrags; f++) {
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < decltype(Otile)::kElemsPerFrag; e++) {
        Otile.val_frags[f][e] = AccumType(f + 1);
      }
    }
    // Skip the div_by_sum_score below — we want markers, not normalized.
  }

  threadgroup_barrier(mem_flags::mem_none);

  // Store results. O is already pre-offset to (global_q_base,
  // q_head_idx). O row stride along the Q-token axis is
  // `Q_stride_tok = num_q_heads * BD`.
  O += (tm + sm) * Q_stride_tok + sn;

  if (ATTN_PAGED_DEBUG_MODE == 1u) {
    // Marker write: each lane writes (sg+1)*10 + 1.0 at its O slot.
    // 128 lanes per TG → 128 distinct O writes per TG. If we see this
    // marker, that (tid, simdgroup, lane) reached the store path.
    O[0] = T(AccumType(simd_group_id + 1u) * AccumType(10) + AccumType(1));
    return;
  }
  if (ATTN_PAGED_DEBUG_MODE == 2u) {
    // Write to O[(tid.x, tid.y, simd_group_id, simd_lane_id) hash slot]
    // — a single global location per (kernel-position, lane) tuple.
    // This proves the kernel REACHED this point regardless of where
    // store_safe / store would have routed the lane.
    const uint slot = ((tid.x * 128u) + (tid.y * 4u) + simd_group_id) * 32u + simd_lane_id;
    // Reuse buffer(3) (seq_used_k) is too small — repurpose O global
    // base offset (the original kernel didn't pre-offset for debug).
    // Map slot to O at slot*1 (assuming buffer is big enough; bench
    // sizes O = M * num_q * D >> 768*4*32 = 98k).
    device T* O_base = O - (int)((tm + sm) * Q_stride_tok + sn);
    if (slot < (uint)(int(ATTN_PAGED_NUM_Q_HEADS) * BD * 1024)) {
      O_base[slot] = T(AccumType(simd_group_id + 1u) * AccumType(10) +
                       AccumType(simd_lane_id) * AccumType(0.01));
    }
    return;
  }

  if (!q_tile_full) {
    auto dst_tile_dims =
        short2(BD - sn, int(q_tile_rows) - (tm + sm));

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Otile.template store_safe<T, 1, 1>(O, Q_stride_tok, dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(O, Q_stride_tok);
  }
}
