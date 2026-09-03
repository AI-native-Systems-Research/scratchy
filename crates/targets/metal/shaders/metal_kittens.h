// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0
//
// MetalKittens — composable MSL primitive library for Apple Silicon.
//
// Core primitive: mk_qmv_fast
//   Cooperative int4 GEMV: 2 simdgroups × 32 threads compute
//   MK_ROWS_PER_SIMDGROUP=4 consecutive output elements each.
//   Matches MLX's qmv_fast_impl (mlx quantized.h:749-814) exactly.
//
// Fused kernel dispatch shape (per-head):
//   Grid: (M, NUM_HEADS_TOTAL, 1)
//   Threads per threadgroup: MK_SIMD_SIZE * (HEAD_DIM / MK_ROWS_PER_SIMDGROUP)
//     = 32 * 32 = 1024 for HEAD_DIM=128
//
// Each threadgroup handles all HEAD_DIM output elements for one (token, head).
// After mk_qmv_fast, lane-0s write 4 results to threadgroup memory; mk_sync()
// makes all HEAD_DIM elements visible for the epilogue (RoPE, cache write).

#pragma once

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

static constant constexpr int MK_SIMD_SIZE          = 32;
static constant constexpr int MK_ROWS_PER_SIMDGROUP = 4;
static constant constexpr int MK_PACKS_PER_THREAD   = 2;  // bits==4 fast path

// ─────────────────────────────────────────────────────────────────────────────
// Pack helpers — quantized.h:1-26
// ─────────────────────────────────────────────────────────────────────────────

template <int bits, int wsize = 8>
inline constexpr short mk_get_pack_factor() {
    return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short mk_get_bytes_per_pack() {
    constexpr int pow2 = (bits & (bits - 1)) == 0;
    return pow2 ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

// ─────────────────────────────────────────────────────────────────────────────
// mk_load_vector — quantized.h:28-189 (bits==4 path)
// ─────────────────────────────────────────────────────────────────────────────

template <typename T, typename U, int values_per_thread, int bits>
inline U mk_load_vector(const device T* x, thread U* x_thread) {
    U sum = 0;
    if (bits == 4) {
        for (int i = 0; i < values_per_thread; i += 4) {
            sum += x[i] + x[i+1] + x[i+2] + x[i+3];
            x_thread[i]   = x[i];
            x_thread[i+1] = x[i+1] / 16.0f;
            x_thread[i+2] = x[i+2] / 256.0f;
            x_thread[i+3] = x[i+3] / 4096.0f;
        }
    } else if (bits == 8) {
        for (int i = 0; i < values_per_thread; i++) {
            sum += x[i];
            x_thread[i] = x[i];
        }
    }
    return sum;
}

// ─────────────────────────────────────────────────────────────────────────────
// mk_qdot — quantized.h:191-392 (bits==4 path)
// ─────────────────────────────────────────────────────────────────────────────

template <typename U, int values_per_thread, int bits>
inline U mk_qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale, U bias, U sum)
{
    U accum = 0;
    if (bits == 4) {
        const device uint16_t* ws = (const device uint16_t*)w;
        for (int i = 0; i < (values_per_thread / 4); i++) {
            accum += (x_thread[4*i]   * (ws[i] & 0x000f) +
                      x_thread[4*i+1] * (ws[i] & 0x00f0) +
                      x_thread[4*i+2] * (ws[i] & 0x0f00) +
                      x_thread[4*i+3] * (ws[i] & 0xf000));
        }
    } else if (bits == 8) {
        for (int i = 0; i < values_per_thread; i++)
            accum += x_thread[i] * w[i];
    }
    return scale * accum + sum * bias;
}

// ─────────────────────────────────────────────────────────────────────────────
// mk_qdot_safe — quantized.h qdot_safe (bits==4/8). Bounded variant of
// mk_qdot: dots only the first `N` (of values_per_thread) weight values,
// so the partial tail block of an unaligned `in_vec_size` never reads
// past the weight buffer. `x_thread[N..]` must already be zeroed by
// mk_load_vector_safe — the loop simply stops at `N`.
// ─────────────────────────────────────────────────────────────────────────────

template <typename U, int values_per_thread, int bits>
inline U mk_qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale, U bias, U sum, int N)
{
    U accum = 0;
    if (bits == 4) {
        const device uint16_t* ws = (const device uint16_t*)w;
        for (int i = 0; i < (N / 4); i++) {
            accum += (x_thread[4*i]   * (ws[i] & 0x000f) +
                      x_thread[4*i+1] * (ws[i] & 0x00f0) +
                      x_thread[4*i+2] * (ws[i] & 0x0f00) +
                      x_thread[4*i+3] * (ws[i] & 0xf000));
        }
    } else if (bits == 8) {
        for (int i = 0; i < N; i++)
            accum += x_thread[i] * w[i];
    }
    return scale * accum + sum * bias;
}

// ─────────────────────────────────────────────────────────────────────────────
// mk_qmv_fast
//
// Core of qmv_fast_impl (quantized.h:749-814). Computes MK_ROWS_PER_SIMDGROUP
// consecutive output elements cooperatively across MK_SIMD_SIZE threads.
//
// Precondition — caller pre-adjusts pointers:
//   ws     += simdgroup_out_row * in_vec_size_w
//              + simd_lid * MK_PACKS_PER_THREAD * bytes_per_pack
//   scales += simdgroup_out_row * in_vec_size_g
//              + simd_lid / scale_step
//   biases += same as scales
//   x      += token * in_vec_size + simd_lid * values_per_thread
//
// where:
//   in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor
//   in_vec_size_g = in_vec_size / group_size
//   scale_step    = group_size / values_per_thread
//
// On return: result[0..MK_ROWS_PER_SIMDGROUP-1] holds the dot products.
// Valid on ALL simd lanes (simd_sum broadcasts). Only simd_lid==0 should
// write to device or threadgroup memory to avoid races.
// ─────────────────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits>
inline void mk_qmv_fast(
    const device uint8_t* ws,
    const device T_scale* scales,
    const device T_scale* biases,
    const device T_act*   x,
    int                   in_vec_size,
    thread float          result[MK_ROWS_PER_SIMDGROUP],
    uint                  simd_lid)
{
    constexpr int pack_factor       = mk_get_pack_factor<bits, 32>();
    constexpr int bytes_per_pack    = mk_get_bytes_per_pack<bits, 32>();
    constexpr int values_per_thread = pack_factor * MK_PACKS_PER_THREAD;
    constexpr int block_size        = values_per_thread * MK_SIMD_SIZE;

    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / group_size;

    thread float x_thread[values_per_thread];
    for (int r = 0; r < MK_ROWS_PER_SIMDGROUP; r++) result[r] = 0.0f;

    for (int k = 0; k < in_vec_size; k += block_size) {
        float sum = mk_load_vector<T_act, float, values_per_thread, bits>(x, x_thread);

        for (int row = 0; row < MK_ROWS_PER_SIMDGROUP; row++) {
            const device uint8_t* wl = ws + row * in_vec_size_w;
            float s = float(scales[row * in_vec_size_g]);
            float b = float(biases[row * in_vec_size_g]);
            result[row] += mk_qdot<float, values_per_thread, bits>(
                wl, x_thread, s, b, sum);
        }

        ws     += block_size * bytes_per_pack / pack_factor;
        scales += block_size / group_size;
        biases += block_size / group_size;
        x      += block_size;
    }

    for (int row = 0; row < MK_ROWS_PER_SIMDGROUP; row++)
        result[row] = simd_sum(result[row]);
}

// ─────────────────────────────────────────────────────────────────────────────
// mk_sync — threadgroup memory barrier
// ─────────────────────────────────────────────────────────────────────────────

inline void mk_sync() {
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ─────────────────────────────────────────────────────────────────────────────
// mk_qmv_fast_to_smem
//
// Calls mk_qmv_fast and writes the 4 results into threadgroup memory at
// smem[smem_base + simd_gid * MK_ROWS_PER_SIMDGROUP + 0..3] (lane-0 only).
//
// After all simdgroups in the threadgroup call this, do mk_sync() so that
// smem[0..HEAD_DIM-1] is fully populated and visible to all threads.
// ─────────────────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits>
inline void mk_qmv_fast_to_smem(
    const device uint8_t* ws,
    const device T_scale* scales,
    const device T_scale* biases,
    const device T_act*   x,
    int                   in_vec_size,
    threadgroup float*    smem,
    uint                  simd_gid,
    uint                  simd_lid)
{
    thread float result[MK_ROWS_PER_SIMDGROUP];
    mk_qmv_fast<T_act, T_scale, group_size, bits>(
        ws, scales, biases, x, in_vec_size, result, simd_lid);

    if (simd_lid == 0) {
        const uint base = simd_gid * MK_ROWS_PER_SIMDGROUP;
        for (int r = 0; r < MK_ROWS_PER_SIMDGROUP; r++)
            smem[base + r] = result[r];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Decode-atom primitives (Phase 1 of the fusion-synthesis plan)
// ─────────────────────────────────────────────────────────────────────────────
//
// These are the building blocks the fuse pass stitches together when
// synthesizing per-(model, bucket, dtype) decoder kernels. Each is a
// pure register / threadgroup-memory operation — no device-memory
// side effects beyond declared output channels. The atom adapters in
// `scratchy-forward-compiler-macro/src/atom.rs` emit calls to these from
// `emit_metal_body`.

// ─────────────────────────────────────────────────────────────────────────────
// mk_load_vector — threadgroup-source overload
//
// Identical body to the device-source `mk_load_vector` above; used by
// fused kernels that stage `x` into TG memory (e.g. RmsNorm-fold
// fusions that compute `x = (residual + delta) * scale * rms_weight`
// once per TG and then run mk_qmv_fast's inner loop reading from TG mem).
// ─────────────────────────────────────────────────────────────────────────────

template <typename T, typename U, int values_per_thread, int bits>
inline U mk_load_vector(const threadgroup T* x, thread U* x_thread) {
    U sum = 0;
    if (bits == 4) {
        for (int i = 0; i < values_per_thread; i += 4) {
            sum += x[i] + x[i+1] + x[i+2] + x[i+3];
            x_thread[i]   = x[i];
            x_thread[i+1] = x[i+1] / 16.0f;
            x_thread[i+2] = x[i+2] / 256.0f;
            x_thread[i+3] = x[i+3] / 4096.0f;
        }
    } else if (bits == 8) {
        for (int i = 0; i < values_per_thread; i++) {
            sum += x[i];
            x_thread[i] = x[i];
        }
    }
    return sum;
}

// ─────────────────────────────────────────────────────────────────────────────
// mk_load_vector_safe — threadgroup-source, bounded (quantized.h
// load_vector_safe). Reads only the first `N` activation values and
// zeros `x_thread[N..values_per_thread]`, so a partial tail block of an
// unaligned `in_vec_size` never reads past the staged `x` row (and the
// zeroed lanes make mk_qdot_safe's omitted weight reads a true no-op).
// ─────────────────────────────────────────────────────────────────────────────

template <typename T, typename U, int values_per_thread, int bits>
inline U mk_load_vector_safe(const threadgroup T* x, thread U* x_thread, int N) {
    U sum = 0;
    if (bits == 4) {
        for (int i = 0; i < N; i += 4) {
            sum += x[i] + x[i+1] + x[i+2] + x[i+3];
            x_thread[i]   = x[i];
            x_thread[i+1] = x[i+1] / 16.0f;
            x_thread[i+2] = x[i+2] / 256.0f;
            x_thread[i+3] = x[i+3] / 4096.0f;
        }
    } else if (bits == 8) {
        for (int i = 0; i < N; i++) {
            sum += x[i];
            x_thread[i] = x[i];
        }
    }
    for (int i = N; i < values_per_thread; i++) {
        x_thread[i] = 0;
    }
    return sum;
}

// ─────────────────────────────────────────────────────────────────────────────
// mk_tg_sum — cooperative sum across an entire threadgroup
//
// Each thread contributes `local`; returns the total broadcast to all
// threads. Two-stage hardware-supported reduction: `simd_sum` within
// each simdgroup, then a single simdgroup folds the per-simdgroup
// partials via `scratch`. `scratch` must hold ≥ `num_simdgroups`
// floats. Bounded by `num_simdgroups ≤ MK_SIMD_SIZE` (32 lanes) —
// 1024-thread TGs at the upper edge.
// ─────────────────────────────────────────────────────────────────────────────

inline float mk_tg_sum(
    float              local,
    threadgroup float* scratch,
    uint               num_simdgroups,
    uint               simd_gid,
    uint               simd_lid)
{
    const float simd_partial = simd_sum(local);
    if (simd_lid == 0) {
        scratch[simd_gid] = simd_partial;
    }
    mk_sync();

    if (simd_gid == 0) {
        float v = (simd_lid < num_simdgroups) ? scratch[simd_lid] : 0.0f;
        v = simd_sum(v);
        if (simd_lid == 0) scratch[0] = v;
    }
    mk_sync();
    return scratch[0];
}

// ─────────────────────────────────────────────────────────────────────────────
// mk_tg_rmsnorm_scale — RMSNorm scale factor `rsqrt(mean(x²) + eps)`
//
// Each thread contributes its `local_sumsq` (partial sum-of-squares
// over its slice of the input vector). Returns the broadcast scale
// to all TG threads. `scratch` and `num_simdgroups` as for `mk_tg_sum`.
// ─────────────────────────────────────────────────────────────────────────────

inline float mk_tg_rmsnorm_scale(
    float              local_sumsq,
    uint               n,
    float              eps,
    threadgroup float* scratch,
    uint               num_simdgroups,
    uint               simd_gid,
    uint               simd_lid)
{
    const float sumsq = mk_tg_sum(local_sumsq, scratch, num_simdgroups,
                                  simd_gid, simd_lid);
    return rsqrt(sumsq / float(n) + eps);
}

// ─────────────────────────────────────────────────────────────────────────────
// mk_silu — SiLU activation (x * sigmoid(x))
//
// SiLU(x) = x / (1 + exp(-x)). Pure register operation. Typed to
// promote to float for the exp/divide, returning float so callers can
// stage further FMAs before casting back to T_act.
// ─────────────────────────────────────────────────────────────────────────────

template <typename T>
inline float mk_silu(T x) {
    const float v = float(x);
    return v / (1.0f + exp(-v));
}

// ─────────────────────────────────────────────────────────────────────────────
// mk_rope_pair — in-register NeoX-style RoPE pair rotation
//
// Rotates (x0, x1) by (cos, sin) in place:
//   x0' = x0 * cos - x1 * sin
//   x1' = x1 * cos + x0 * sin
//
// Used by attention-block fusions where Q/K vector elements live in
// per-thread registers and the rotation happens before the paged-cache
// or arena write.
// ─────────────────────────────────────────────────────────────────────────────

inline void mk_rope_pair(
    thread float& x0,
    thread float& x1,
    float         cos_v,
    float         sin_v)
{
    const float n0 = x0 * cos_v - x1 * sin_v;
    const float n1 = x1 * cos_v + x0 * sin_v;
    x0 = n0;
    x1 = n1;
}

// ─────────────────────────────────────────────────────────────────────────────
// mk_paged_kv_write_row — paged KV-cache element write
//
// Writes one element at `cache[block_id, kv_head, block_offset, d]`
// where `(block_id, block_offset) = divmod(slot, block_size)`. The
// caller is responsible for sentinel-slot handling
// (slot == 0xFFFFFFFF marks a padding token and the write should be
// skipped).
// ─────────────────────────────────────────────────────────────────────────────

template <typename T>
inline void mk_paged_kv_write_row(
    device T* cache_base,
    uint      slot,
    uint      kv_head,
    uint      num_kv,
    uint      block_size,
    uint      head_dim,
    uint      d,
    T         value)
{
    const uint block_id     = slot / block_size;
    const uint block_offset = slot % block_size;
    device T* dst = cache_base
        + (size_t)block_id     * (size_t)(num_kv * block_size * head_dim)
        + (size_t)kv_head      * (size_t)(block_size * head_dim)
        + (size_t)block_offset * (size_t)head_dim;
    dst[d] = value;
}

// ─────────────────────────────────────────────────────────────────────────────
// mk_apply_rms_scale — per-element RMSNorm output (x * scale * weight)
//
// One step of the RMSNorm output computation, applied after
// `mk_tg_rmsnorm_scale` has produced `scale` for this token's
// `[HIDDEN]` vector. Promotes through float for FMA single-rounding
// — matches MLX rmsnorm behavior.
// ─────────────────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale>
inline T_act mk_apply_rms_scale(T_act x, float scale, T_scale weight) {
    return T_act(float(x) * scale * float(weight));
}
