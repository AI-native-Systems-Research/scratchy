// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused affine-int4 Q/K/V projection + NeoX RoPE + paged KV-cache write.
// Replaces [AffineQmv(Q), AffineQmv(K), AffineQmv(V), RopeAppend] with one
// dispatch. Written using MetalKittens (metal_kittens.h) for the cooperative
// simdgroup GEMV inner product; matches qmv_fast_impl perf on each head.
//
// Dispatch shape:
//   threadgroups: (M, NUM_Q + 2*NUM_KV, 1)
//   threads_per_threadgroup: (MK_SIMD_SIZE * HEAD_DIM / MK_ROWS_PER_SIMDGROUP, 1, 1)
//                          = (32 * HEAD_DIM / 4, 1, 1)
//                          = 1024 for HEAD_DIM=128
//
// Each threadgroup handles one (token, head). All HEAD_DIM output elements are
// computed by MK simdgroups cooperatively, staged through threadgroup memory,
// then the epilogue applies RoPE and writes to q_out / kv_cache.
//
// Function constants (must match lowering.rs KernelId::FusedAffineQkvRopeCache):
//   0 = HIDDEN        (uint) — K dimension
//   1 = NUM_Q_HEADS   (uint)
//   2 = NUM_KV_HEADS  (uint)
//   3 = HEAD_DIM      (uint) — must be multiple of MK_ROWS_PER_SIMDGROUP (4)
//   4 = ROT_DIM       (uint) — must be multiple of MK_ROWS_PER_SIMDGROUP (4)
//   5 = BLOCK_SIZE    (uint) — paged KV cache block size
//   6 = M             (uint) — token count for this bucket
//   7 = GROUP_SIZE    (uint) — int4 quantization group size (32, 64, or 128)
//
// Bindings:
//   buffer(0) = q_out         [M, NUM_Q  * HEAD_DIM]                  T_act w
//   buffer(1) = input         [M, HIDDEN]                              T_act r
//   buffer(2) = weight_packed [(NUM_Q+2*NUM_KV)*HEAD_DIM, HIDDEN/8]   u32   r
//   buffer(3) = scales        [(NUM_Q+2*NUM_KV)*HEAD_DIM, HIDDEN/GS]  T_scale r
//   buffer(4) = biases        same shape as scales                     T_scale r
//   buffer(5) = cos_sin       [max_pos, ROT_DIM]                       T_act r
//   buffer(6) = positions     [M]                                      u32   r
//   buffer(7) = slot_mapping  [M] sentinel 0xFFFFFFFF → skip write     u32   r
//   buffer(8) = kv_cache_k    [num_blocks, NUM_KV, BLOCK_SIZE, HEAD_DIM] T_act rw
//   buffer(9) = kv_cache_v    same shape as kv_cache_k                T_act rw
//
// RoPE constraint: ROT_DIM/2 must be a multiple of MK_ROWS_PER_SIMDGROUP (4)
// so that no simdgroup spans the half-dim boundary. True for all Llama/Qwen2
// configs (half_dim is always a power of 2 >= 32).

#include <metal_stdlib>
#include "metal_kittens.h"
using namespace metal;

constant uint FAQRC_HIDDEN       [[function_constant(0)]];
constant uint FAQRC_NUM_Q_HEADS  [[function_constant(1)]];
constant uint FAQRC_NUM_KV_HEADS [[function_constant(2)]];
constant uint FAQRC_HEAD_DIM     [[function_constant(3)]];
constant uint FAQRC_ROT_DIM      [[function_constant(4)]];
constant uint FAQRC_BLOCK_SIZE   [[function_constant(5)]];
constant uint FAQRC_M            [[function_constant(6)]];
constant uint FAQRC_GROUP_SIZE   [[function_constant(7)]];
// Spans / rope-on-read: store K UNROTATED for slot_mapping-bit-31 blocks
// (attention re-ropes on read). Default-off → non-spans byte-identical.
// (Shader is currently unreached — no FusedAffineQkvRopeCache lowering arm —
// hardened so wiring the arm can't silently double-rotate span K.)
constant uint FAQRC_ROPE_ON_READ [[function_constant(8)]];
constant bool FAQRC_ROR_DEFINED  = is_function_constant_defined(FAQRC_ROPE_ON_READ);
constant uint FAQRC_ROR          = FAQRC_ROR_DEFINED ? FAQRC_ROPE_ON_READ : 0u;

// Compile-time ceiling for threadgroup memory. HEAD_DIM <= 256 for all
// supported models; increasing this wastes tg memory per threadgroup.
constant constexpr uint FAQRC_HEAD_DIM_MAX = 256;

template <typename T_act, typename T_scale, int group_size>
[[kernel]] void fused_affine_qkv_rope_cache_mk_impl(
    device       T_act*    q_out         [[buffer(0)]],
    device const T_act*    input         [[buffer(1)]],
    device const uint32_t* weight_packed [[buffer(2)]],
    device const T_scale*  scales        [[buffer(3)]],
    device const T_scale*  biases        [[buffer(4)]],
    device const T_act*    cos_sin       [[buffer(5)]],
    device const uint*     positions     [[buffer(6)]],
    device const uint*     slot_mapping  [[buffer(7)]],
    device       T_act*    kv_cache_k    [[buffer(8)]],
    device       T_act*    kv_cache_v    [[buffer(9)]],
    uint3 tg_pos   [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    const uint t               = tg_pos.x;
    const uint head            = tg_pos.y;
    const uint head_dim        = FAQRC_HEAD_DIM;
    const uint hidden          = FAQRC_HIDDEN;
    const uint num_q           = FAQRC_NUM_Q_HEADS;
    const uint num_kv          = FAQRC_NUM_KV_HEADS;
    const uint rot_dim         = FAQRC_ROT_DIM;
    const uint half_dim        = rot_dim / 2;
    const uint block_sz        = FAQRC_BLOCK_SIZE;
    const uint num_heads_total = num_q + 2u * num_kv;

    if (t >= FAQRC_M || head >= num_heads_total) return;

    // ── Pointer adjustment (matches qmv_fast_impl layout exactly) ────────────
    //
    // simd_gid selects which 4 consecutive output rows this simdgroup computes.
    // Global output row base: head * HEAD_DIM + simd_gid * MK_ROWS_PER_SIMDGROUP.
    //
    constexpr int bits              = 4;
    constexpr int pack_factor       = mk_get_pack_factor<bits, 32>();
    constexpr int bytes_per_pack    = mk_get_bytes_per_pack<bits, 32>();
    constexpr int values_per_thread = pack_factor * MK_PACKS_PER_THREAD;
    constexpr int scale_step        = group_size / values_per_thread;

    const uint global_out_row_base = head * head_dim + simd_gid * MK_ROWS_PER_SIMDGROUP;
    const int  in_vec_size_w       = (int)hidden * bytes_per_pack / pack_factor;
    const int  in_vec_size_g       = (int)hidden / group_size;

    const device uint8_t* ws = (const device uint8_t*)weight_packed
        + (size_t)global_out_row_base * (size_t)in_vec_size_w
        + (size_t)simd_lid * MK_PACKS_PER_THREAD * bytes_per_pack;
    const device T_scale* sc = scales
        + (size_t)global_out_row_base * (size_t)in_vec_size_g
        + simd_lid / scale_step;
    const device T_scale* bi = biases
        + (size_t)global_out_row_base * (size_t)in_vec_size_g
        + simd_lid / scale_step;
    const device T_act* x = input
        + (size_t)t * (size_t)hidden
        + (size_t)simd_lid * values_per_thread;

    // ── Compute 4 output elements → threadgroup memory ────────────────────────
    //
    // After mk_qmv_fast_to_smem + mk_sync(), smem[0..HEAD_DIM-1] holds the
    // full projection output for this (token, head).
    //
    threadgroup float smem[FAQRC_HEAD_DIM_MAX];

    mk_qmv_fast_to_smem<T_act, T_scale, group_size, bits>(
        ws, sc, bi, x, (int)hidden, smem, simd_gid, simd_lid);

    mk_sync();

    // ── Epilogue: RoPE + Q output / KV cache write ────────────────────────────
    //
    // Only simd_lid == 0 writes (4 elements per simdgroup). simd_sum in
    // mk_qmv_fast broadcasts results to all lanes so any lane could write,
    // but lane 0 is the canonical choice to avoid 32-way write conflicts.
    //
    if (simd_lid != 0) return;

    // Rows this simdgroup owns.
    const uint base_d = simd_gid * MK_ROWS_PER_SIMDGROUP;

    const uint kQ_END = num_q;
    const uint kK_END = num_q + num_kv;

    // ── Q head ────────────────────────────────────────────────────────────────
    if (head < kQ_END) {
        device T_act* q_row = q_out
            + (size_t)t    * (size_t)(num_q * head_dim)
            + (size_t)head * (size_t)head_dim;

        if (base_d < half_dim) {
            // Owns d in [base_d, base_d+3], all < half_dim (constraint: half_dim
            // is a multiple of MK_ROWS_PER_SIMDGROUP so no group spans the boundary).
            const uint pos = positions[t];
            device const T_act* cos_row = cos_sin + (size_t)pos * (size_t)rot_dim;
            device const T_act* sin_row = cos_row + half_dim;
            for (int r = 0; r < MK_ROWS_PER_SIMDGROUP; r++) {
                const uint d  = base_d + r;
                const float c = float(cos_row[d]);
                const float s = float(sin_row[d]);
                const float x0 = smem[d];
                const float x1 = smem[half_dim + d];
                q_row[d]            = T_act(x0 * c - x1 * s);
                q_row[half_dim + d] = T_act(x1 * c + x0 * s);
            }
        } else if (base_d >= rot_dim) {
            // Pass-through tail (partial rope only; full rope has rot_dim == head_dim).
            for (int r = 0; r < MK_ROWS_PER_SIMDGROUP; r++)
                q_row[base_d + r] = T_act(smem[base_d + r]);
        }
        // base_d in [half_dim, rot_dim): written by the paired lower simdgroup.
        return;
    }

    // ── K head ────────────────────────────────────────────────────────────────
    if (head < kK_END) {
        const uint kv_head = head - num_q;
        const uint slot    = slot_mapping[t];
        if (slot == 0xFFFFFFFFu) return;

        // Spans: bit 31 = store this block's K UNROTATED (rope-on-read).
        const bool skip_k_rot = (FAQRC_ROR != 0u) && ((slot & 0x80000000u) != 0u);
        const uint phys_slot    = (FAQRC_ROR != 0u) ? (slot & 0x7FFFFFFFu) : slot;
        const uint block_id     = phys_slot / block_sz;
        const uint block_offset = phys_slot % block_sz;
        device T_act* k_dst = kv_cache_k
            + (size_t)block_id     * (size_t)(num_kv * block_sz * head_dim)
            + (size_t)kv_head      * (size_t)(block_sz * head_dim)
            + (size_t)block_offset * (size_t)head_dim;

        if (skip_k_rot) {
            // Store K verbatim (every lane writes its own range); attention
            // re-ropes to the reuse position on read.
            for (int r = 0; r < MK_ROWS_PER_SIMDGROUP; r++)
                k_dst[base_d + r] = T_act(smem[base_d + r]);
        } else if (base_d < half_dim) {
            const uint pos = positions[t];
            device const T_act* cos_row = cos_sin + (size_t)pos * (size_t)rot_dim;
            device const T_act* sin_row = cos_row + half_dim;
            for (int r = 0; r < MK_ROWS_PER_SIMDGROUP; r++) {
                const uint d  = base_d + r;
                const float c = float(cos_row[d]);
                const float s = float(sin_row[d]);
                const float x0 = smem[d];
                const float x1 = smem[half_dim + d];
                k_dst[d]            = T_act(x0 * c - x1 * s);
                k_dst[half_dim + d] = T_act(x1 * c + x0 * s);
            }
        } else if (base_d >= rot_dim) {
            for (int r = 0; r < MK_ROWS_PER_SIMDGROUP; r++)
                k_dst[base_d + r] = T_act(smem[base_d + r]);
        }
        return;
    }

    // ── V head — pass-through to kv_cache_v ───────────────────────────────────
    {
        const uint kv_head = head - kK_END;
        const uint slot    = slot_mapping[t];
        if (slot == 0xFFFFFFFFu) return;

        // Spans: strip the bit-31 flag for addressing (identity when ROR off).
        const uint phys_slot    = (FAQRC_ROR != 0u) ? (slot & 0x7FFFFFFFu) : slot;
        const uint block_id     = phys_slot / block_sz;
        const uint block_offset = phys_slot % block_sz;
        device T_act* v_dst = kv_cache_v
            + (size_t)block_id     * (size_t)(num_kv * block_sz * head_dim)
            + (size_t)kv_head      * (size_t)(block_sz * head_dim)
            + (size_t)block_offset * (size_t)head_dim;

        for (int r = 0; r < MK_ROWS_PER_SIMDGROUP; r++)
            v_dst[base_d + r] = T_act(smem[base_d + r]);
    }
}

// Instantiate for all (T_act, T_scale, group_size) combinations used in
// mlx-community 4-bit models. group_size ∈ {32, 64, 128}.
#define INST_FAQRC_MK(act_tag, act_type, scale_type, gs)                       \
  template [[host_name("fused_affine_qkv_rope_cache_mk_" #act_tag "_gs" #gs)]] \
  [[kernel]] decltype(fused_affine_qkv_rope_cache_mk_impl<act_type, scale_type, gs>) \
      fused_affine_qkv_rope_cache_mk_impl<act_type, scale_type, gs>;

INST_FAQRC_MK(f16,  half,    half,    32)
INST_FAQRC_MK(f16,  half,    half,    64)
INST_FAQRC_MK(f16,  half,    half,   128)
INST_FAQRC_MK(bf16, bfloat,  bfloat,  32)
INST_FAQRC_MK(bf16, bfloat,  bfloat,  64)
INST_FAQRC_MK(bf16, bfloat,  bfloat, 128)
