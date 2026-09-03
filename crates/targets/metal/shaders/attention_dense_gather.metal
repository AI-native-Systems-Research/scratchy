// SPDX-License-Identifier: Apache-2.0
//
// Dense kv-head-major gather for the gemma4 hd512 GLOBAL unfused-attention
// path. Gathers paged K/V into a contiguous [num_kv, kv_len, head_dim] buffer
// so each kv-head's slab is a contiguous [kv_len, head_dim] matrix that MPS
// (MPSMatrixMultiplication) can consume directly for the QK^T / PV GEMMs.
//
// The existing `rope_once_gqa_shared` produces a PAGE-major layout
// ([block, kv_head, tib, dim]) for the cooperative kernel; MPS needs each
// kv-head contiguous, hence this twin. The row stride per kv-head is derived
// from the live `seq_used[0]` (kv_len), so the dense buffer is packed exactly
// [num_kv * kv_len * head_dim] with no padding between heads.
//
// Two variants:
//   * copy  (V): plain gather, no rotation.
//   * rope  (K): rope-on-read (proportional/NeoX), mirroring
//     `rope_once_gqa_shared_body`'s math; only the rotary pairs are rotated,
//     gated on the block_table bit-31 "stored unrotated" flag (spans).

#include <metal_stdlib>
using namespace metal;

constant uint  GDK_HEAD_DIM          [[function_constant(0)]];
constant uint  GDK_NUM_KV            [[function_constant(1)]];
constant uint  GDK_BLOCK_SIZE        [[function_constant(2)]];
constant uint  GDK_BLOCKS_PER_CHUNK  [[function_constant(3)]];
constant uint  GDK_ROT_DIM           [[function_constant(4)]];
constant uint  GDK_PAIR_OFF          [[function_constant(5)]];
// STATIC dest row stride (= block_cap*block_size) so per-head GEMM weight
// offsets are bakeable into the tape. Only positions [0,kv_len) are written.
constant uint  GDK_MAX_KV            [[function_constant(6)]];

// One thread per (pos, kv_head). gid.x = pos (key position), gid.y = kv_head.
// SrcT = paged-cache dtype (gemma4: bfloat); DstT = MPS-GEMM compute dtype (half).
template <typename SrcT, typename DstT, bool APPLY_ROPE>
inline void gather_dense_kvmajor_body(
    device DstT*            dst,         // [num_kv, kv_len, head_dim] kv-head-major
    const device uint*      block_table, // [max_blocks]
    const device uint64_t*  cache,       // chunk base addresses (paged)
    const device uint*      seq_used,    // [1] live kv_len
    const device SrcT*      cos_sin,     // [max_pos, rot_dim] (read iff APPLY_ROPE)
    uint2                   gid)
{
    const uint pos     = gid.x;
    const uint kv_head = gid.y;
    const uint kv_len  = seq_used[0];
    if (pos >= kv_len || kv_head >= GDK_NUM_KV) {
        return;
    }

    const uint hd = GDK_HEAD_DIM;
    const uint bs = GDK_BLOCK_SIZE;
    const uint logical_block = pos / bs;
    const uint tib           = pos % bs;

    // Source (paged cache): [block, kv_head, tib, dim], block-major.
    const uint kv_blk_stride = GDK_NUM_KV * bs * hd;
    const uint kv_head_off   = kv_head * (bs * hd);
    const uint bt_raw   = block_table[logical_block];
    const uint physical = bt_raw & 0x7FFFFFFFu;
    uint chunk;
    uint bic;
    if (GDK_BLOCKS_PER_CHUNK == 0u) {
        chunk = 0u;
        bic   = physical;
    } else {
        chunk = physical / GDK_BLOCKS_PER_CHUNK;
        bic   = physical % GDK_BLOCKS_PER_CHUNK;
    }
    const device SrcT* src =
        (const device SrcT*)cache[chunk] + bic * kv_blk_stride + kv_head_off + tib * hd;

    // Dest (dense): [kv_head, pos, dim], kv-head-major; STATIC stride GDK_MAX_KV*hd.
    device DstT* d = dst + kv_head * (GDK_MAX_KV * hd) + pos * hd;

    if (!APPLY_ROPE) {
        for (uint i = 0u; i < hd; i++) {
            d[i] = DstT(src[i]);
        }
        return;
    }

    // K rope-on-read. Non-rotated dims pass through; rotary pairs (dd, pair_off+dd)
    // for dd in [0, half_dim) are rotated to absolute position `pos` unless the
    // block was stored already-rotated (flag bit clear).
    const uint half_dim = GDK_ROT_DIM / 2u;
    const uint pair_off = GDK_PAIR_OFF;
    const bool flagged  = (bt_raw & 0x80000000u) != 0u;

    for (uint dd = 0u; dd < hd; dd++) {
        const bool is_low  = dd < half_dim;
        const bool is_high = (dd >= pair_off) && (dd < pair_off + half_dim);
        if (!is_low && !is_high) {
            d[dd] = DstT(src[dd]);
        }
    }
    if (!flagged) {
        for (uint dd = 0u; dd < half_dim; dd++) {
            d[dd]            = DstT(src[dd]);
            d[pair_off + dd] = DstT(src[pair_off + dd]);
        }
        return;
    }
    const device SrcT* cr = cos_sin + pos * GDK_ROT_DIM;
    for (uint dd = 0u; dd < half_dim; dd++) {
        const float c  = float(cr[dd]);
        const float s  = float(cr[half_dim + dd]);
        const float x0 = float(src[dd]);
        const float x1 = float(src[pair_off + dd]);
        d[dd]            = DstT(x0 * c - x1 * s);
        d[pair_off + dd] = DstT(x1 * c + x0 * s);
    }
}

// Same-dtype f16 (used by the correctness test).
kernel void gather_dense_kvmajor_copy_f16(
    device half*           dst         [[buffer(0)]],
    const device uint*     block_table [[buffer(1)]],
    const device uint64_t* cache       [[buffer(2)]],
    const device uint*     seq_used    [[buffer(3)]],
    uint2                  gid         [[thread_position_in_grid]])
{
    gather_dense_kvmajor_body<half, half, false>(dst, block_table, cache, seq_used, nullptr, gid);
}

// gemma4 production: paged cache is bfloat, MPS GEMM compute is half → convert on gather.
kernel void gather_dense_kvmajor_copy_bf16_to_f16(
    device half*           dst         [[buffer(0)]],
    const device uint*     block_table [[buffer(1)]],
    const device uint64_t* cache       [[buffer(2)]],
    const device uint*     seq_used    [[buffer(3)]],
    uint2                  gid         [[thread_position_in_grid]])
{
    gather_dense_kvmajor_body<bfloat, half, false>(dst, block_table, cache, seq_used, nullptr, gid);
}

kernel void gather_dense_kvmajor_rope_bf16_to_f16(
    device half*           dst         [[buffer(0)]],
    const device uint*     block_table [[buffer(1)]],
    const device uint64_t* cache       [[buffer(2)]],
    const device uint*     seq_used    [[buffer(3)]],
    const device bfloat*   cos_sin     [[buffer(4)]],
    uint2                  gid         [[thread_position_in_grid]])
{
    gather_dense_kvmajor_body<bfloat, half, true>(dst, block_table, cache, seq_used, cos_sin, gid);
}

// Same-dtype f16 rope (parity / non-bf16 arches).
kernel void gather_dense_kvmajor_rope_f16(
    device half*           dst         [[buffer(0)]],
    const device uint*     block_table [[buffer(1)]],
    const device uint64_t* cache       [[buffer(2)]],
    const device uint*     seq_used    [[buffer(3)]],
    const device half*     cos_sin     [[buffer(4)]],
    uint2                  gid         [[thread_position_in_grid]])
{
    gather_dense_kvmajor_body<half, half, true>(dst, block_table, cache, seq_used, cos_sin, gid);
}

// ── bf16->bf16 variants for the production (bf16) hd512 unfused path ──
kernel void gather_dense_kvmajor_copy_bf16(
    device bfloat*         dst         [[buffer(0)]],
    const device uint*     block_table [[buffer(1)]],
    const device uint64_t* cache       [[buffer(2)]],
    const device uint*     seq_used    [[buffer(3)]],
    uint2                  gid         [[thread_position_in_grid]])
{
    gather_dense_kvmajor_body<bfloat, bfloat, false>(dst, block_table, cache, seq_used, nullptr, gid);
}

kernel void gather_dense_kvmajor_rope_bf16(
    device bfloat*         dst         [[buffer(0)]],
    const device uint*     block_table [[buffer(1)]],
    const device uint64_t* cache       [[buffer(2)]],
    const device uint*     seq_used    [[buffer(3)]],
    const device bfloat*   cos_sin     [[buffer(4)]],
    uint2                  gid         [[thread_position_in_grid]])
{
    gather_dense_kvmajor_body<bfloat, bfloat, true>(dst, block_table, cache, seq_used, cos_sin, gid);
}

// Transposed copy for V: dst is [kv_head, head_dim, kv_len] (head-dim major)
// so the PV GEMM (C=A@B^T) gets B = V^T directly. dst[kv,d,pos] = paged V[..,d].
kernel void gather_dense_kvmajor_copyT_bf16(
    device bfloat*         dst         [[buffer(0)]],   // [num_kv, head_dim, kv_len]
    const device uint*     block_table [[buffer(1)]],
    const device uint64_t* cache       [[buffer(2)]],
    const device uint*     seq_used    [[buffer(3)]],
    uint2                  gid         [[thread_position_in_grid]])
{
    const uint pos     = gid.x;
    const uint kv_head = gid.y;
    const uint kv_len  = seq_used[0];
    const uint hd = GDK_HEAD_DIM;
    // The NAX PV gemm (gemm_t_nax_impl) does NOT guard its contraction K-tail (MLX
    // assumes K is BK=64 aligned), so it over-reads V^T in [kv_len, padded). That tail
    // is STALE across forwards (the dense scratch is reused) → garbage on chunked-
    // prefill CONTINUATIONS. (A single chunk has a fresh, zero tail, which masked it.)
    // Zero the tail so the over-read contracts 0. Steel guards its K-tail → harmless.
    const uint kv_len_padded = (kv_len + 63u) & ~63u;
    if (pos >= kv_len_padded || kv_head >= GDK_NUM_KV) return;
    if (pos >= kv_len) {
        device bfloat* dz = dst + kv_head * (hd * GDK_MAX_KV) + pos;
        for (uint i = 0u; i < hd; i++) { dz[i * GDK_MAX_KV] = bfloat(0); }
        return;
    }
    const uint bs = GDK_BLOCK_SIZE;
    const uint logical_block = pos / bs;
    const uint tib = pos % bs;
    const uint kv_blk_stride = GDK_NUM_KV * bs * hd;
    const uint kv_head_off = kv_head * (bs * hd);
    const uint physical = block_table[logical_block] & 0x7FFFFFFFu;
    uint chunk, bic;
    if (GDK_BLOCKS_PER_CHUNK == 0u) { chunk = 0u; bic = physical; }
    else { chunk = physical / GDK_BLOCKS_PER_CHUNK; bic = physical % GDK_BLOCKS_PER_CHUNK; }
    const device bfloat* src = (const device bfloat*)cache[chunk] + bic * kv_blk_stride + kv_head_off + tib * hd;
    device bfloat* d = dst + kv_head * (hd * GDK_MAX_KV) + pos;  // [kv, d, pos], STATIC stride
    for (uint i = 0u; i < hd; i++) {
        d[i * GDK_MAX_KV] = src[i];
    }
}
