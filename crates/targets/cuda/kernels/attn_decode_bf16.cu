// SPDX-License-Identifier: Apache-2.0
//
// Grouped-query decode attention of one query row over the paged bf16 KV cache, for the H100 at
// head_dim 256: the one-row bucket's attention (a decode step's one new token). Q and O are the
// tape's `[1, q_heads · 256]` bf16; K and V are read from one layer's paged cache
// `[num_blocks, kv_heads, 64, 256]` bf16 through the sequence's block table.
//
//   o[h, :] = Σ_j softmax_j(scale · q[h, :] · k[j, h/hr, :]) · v[j, h/hr, :],   j < kv_len
//
// with `kv_len = P + L` read from the tape's `sequence` region `[P, L]` (L = 1 for a decode, the
// new token's key already appended by its layer's `RopeAppend`), and `hr = q_heads / kv_heads` the
// GQA group: query head h reads kv head h / hr.
//
// Plain CUDA, not ThunderKittens: the pinned ThunderKittens has no decode kernel, and one query row
// is below both the 16-row register tile and the 64-row WGMMA. The structure mirrors metal's
// one-pass decode (`shaders/attention.metal`, `attention_via_cache_v2_*`): the warps of one CTA
// split the keys, and their partials merge in shared memory in a fixed order.
//
//   - Grid `(q_heads)`, 256 threads (8 warps). CTA h reads kv head h / hr. Lane l holds query lanes
//     `8l .. 8l + 8` in f32 (one 16-byte load; head_dim 256 = 32 lanes × 8).
//   - Warp w walks keys `j = w, w + 8, w + 16, … < kv_len`. For each key it reads
//     `block = block_table[j / 64]`, slot `j % 64`, then 16 bytes of K's row
//     `((block · kv_heads + kvh) · 64 + slot) · 256 + 8l`. The dot product sums the lane's 8
//     products in ascending order with `fmaf`, then a fixed xor butterfly (16, 8, 4, 2, 1) gives
//     every lane the same sum.
//   - The online softmax runs in f32 with `c = scale · log2(e)`. The score `s = dot · c` is rounded
//     on its own (`__fmul_rn`), so fast-math cannot contract it into `fma(dot, c, −m_new)` and the
//     max key's `p` is exactly 1. `m_new = max(m, s)`; the rescale is taken as a difference first,
//     `r = (m == m_new) ? 1 : exp2(m − m_new)` (#812's fast-math lesson: an unchanged max rescales
//     by exactly 1, never by an `exp2` of a rounded difference); `p = exp2(s − m_new)`;
//     `l = l·r + p`; `o[i] = o[i]·r + p·v[i]`, V's 16 bytes loaded from the same cache row. `p`
//     stays f32, as in metal's decode.
//   - Merge: each warp writes `(m, l, o[256])` to static shared memory (8 × 258 f32 = 8,256 bytes,
//     under 48 KiB, so no attribute and no one-time setup). After `__syncthreads`, thread d (one
//     per output lane) merges warps 0..7 in ascending order: `M = max m_w`, `L = Σ l_w · r_w`,
//     `o = Σ o_w[d] · r_w`, skipping a warp that saw no key (`l_w == 0`), and rounds `o / L` once
//     to bf16 into `out[h · 256 + d]`.
//
// No key at or past kv_len is ever loaded. This is not a mask: the pool is never cleared, so an
// unwritten slot can hold NaN, and 0 · NaN = NaN (#791). The loop bound is the only guard needed.
//
// The order of every addition is a function of kv_len alone (key j always lands on warp j % 8, in
// ascending j, and the merge is in ascending warp order); it does not depend on which physical
// blocks the table names, so a shuffled block table gives the same bits, and so does a repeat. No
// atomics.
//
// Speed is not this kernel's aim. One CTA per query head means 8 CTAs on 132 SMs for qwen3.5-0.8b:
// at the parity lengths (under a few hundred keys) each warp walks about 25 keys and the GEMVs
// dominate a decode step, but at 8192 keys this kernel is slow (correct, with 1024 keys per warp).
// Cross-CTA flash-decoding (split-K over many CTAs, then a combine launch over an arena scratch of
// partials) is deferred to a later speed issue.
//
// Built with `--use_fast_math` (build.rs): `exp2f` is the approximate `ex2.approx.ftz`; the
// rescale's exact-1 select above is what keeps an unchanged max exact under it.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

// The launcher's return code that is not a `cudaError_t` (which is non-negative). `src/kernels.rs`
// mirrors it.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)

namespace {

// `step::ATTN_DECODE_BF16_HEAD_DIM`: 32 lanes × 8.
constexpr int HEAD_DIM = 256;
constexpr int VEC = 8;
constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
// Token slots per KV block (`step::CUDA_KV_BLOCK_SIZE`).
constexpr uint32_t BLOCK_SIZE = 64;
// log2(e): the softmax runs in base 2 with the scale folded in.
constexpr float LOG2E = 1.44269504089f;
// A warp's 32 lanes hold one query row of 8 lanes each, and the merge's thread d writes output
// lane d.
static_assert(32 * VEC == HEAD_DIM, "one warp holds one head_dim row");
static_assert(THREADS == HEAD_DIM, "the merge has one thread per output lane");

__device__ __forceinline__ void load8(const __nv_bfloat16* p, float (&f)[VEC]) {
    const uint4 raw = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < VEC / 2; ++i) {
        const float2 v = __bfloat1622float2(h[i]);
        f[2 * i] = v.x;
        f[2 * i + 1] = v.y;
    }
}

__global__ void __launch_bounds__(THREADS)
attn_decode_bf16_kernel(const __nv_bfloat16* __restrict__ q,
                        const __nv_bfloat16* __restrict__ k_cache,
                        const __nv_bfloat16* __restrict__ v_cache,
                        const uint32_t* __restrict__ block_table,
                        const uint32_t* __restrict__ sequence, __nv_bfloat16* __restrict__ out,
                        int kv_heads, int group, uint32_t num_blocks, float c) {
    __shared__ float s_o[WARPS][HEAD_DIM];
    __shared__ float s_m[WARPS];
    __shared__ float s_l[WARPS];

    const int h = blockIdx.x;
    const int kvh = h / group;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    // `forward` checked `P + L` fits the block table; both are u32 on the device.
    const uint32_t kv_len = sequence[0] + sequence[1];

    float qf[VEC];
    load8(q + size_t(h) * HEAD_DIM + size_t(lane) * VEC, qf);

    float m = -INFINITY;
    float l = 0.0f;
    float o[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        o[i] = 0.0f;
    }
    for (uint32_t j = warp; j < kv_len; j += WARPS) {
        const uint32_t block = block_table[j / BLOCK_SIZE];
        // `forward` refused any block outside the pool; one here is skipped, never read (defence
        // in depth, as `kv_append_bf16.cu` writes nothing for one).
        if (block >= num_blocks) {
            continue;
        }
        const size_t row =
            ((size_t(block) * size_t(kv_heads) + size_t(kvh)) * BLOCK_SIZE + (j % BLOCK_SIZE)) *
                HEAD_DIM +
            size_t(lane) * VEC;
        float kf[VEC];
        load8(k_cache + row, kf);
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            dot = fmaf(qf[i], kf[i], dot);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, offset);
        }
        // `__fmul_rn` is never contracted into the `exp2f` argument below: a contracted
        // `fma(dot, c, −m_new)` would give the max key `exp2(rounding error)`, not exactly 1.
        const float s = __fmul_rn(dot, c);
        const float m_new = fmaxf(m, s);
        // The first key: the empty state has no sum to rescale. An unchanged max: exactly 1.
        const float r = (m == m_new) ? 1.0f : (m == -INFINITY ? 0.0f : exp2f(m - m_new));
        const float p = exp2f(s - m_new);
        float vf[VEC];
        load8(v_cache + row, vf);
        l = fmaf(l, r, p);
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            o[i] = fmaf(o[i], r, p * vf[i]);
        }
        m = m_new;
    }
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        s_o[warp][lane * VEC + i] = o[i];
    }
    if (lane == 0) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
    __syncthreads();

    // Thread d merges lane d of every warp's partial, in ascending warp order.
    const int d = threadIdx.x;
    float big = -INFINITY;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
        if (s_l[w] != 0.0f) {
            big = fmaxf(big, s_m[w]);
        }
    }
    float sum = 0.0f;
    float acc = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
        if (s_l[w] == 0.0f) {
            continue;
        }
        const float r = (s_m[w] == big) ? 1.0f : exp2f(s_m[w] - big);
        sum = fmaf(s_l[w], r, sum);
        acc = fmaf(s_o[w][d], r, acc);
    }
    // `kv_len ≥ 1` (`forward` checked `L ≥ 1`), so warp 0 saw a key and `sum > 0`; an empty
    // sequence writes +0.0 rather than 0 / 0.
    out[size_t(h) * HEAD_DIM + d] = __float2bfloat16_rn(sum > 0.0f ? acc / sum : 0.0f);
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched; any other value = the
// cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// Refused before any pointer is touched: `head_dim` not 256; `q_heads` or `kv_heads` zero, or
// `q_heads` not a multiple of `kv_heads`; `block_size` not 64; `num_blocks` zero; `scale` not
// finite or not above 0; a dimension above INT_MAX.
//
// `q` and `out` must be 16-byte-aligned device pointers to `q_heads · 256` bf16 (one row), `out`
// overlapping no input; `k_cache` and `v_cache` to one layer's `num_blocks · kv_heads · 64 · 256`
// bf16; `sequence` to `[P, L]` u32 with `L ≥ 1`; `block_table` to the sequence's block ids,
// `ceil((P + L) / 64)` of them, each below `num_blocks`. No dynamic shared memory, so no one-time
// setup. The launch is asynchronous on `stream`; the caller owns it and synchronizes.
extern "C" int scratchy_cuda_attn_decode_bf16(const void* q, const void* k_cache,
                                              const void* v_cache, const void* block_table,
                                              const void* sequence, void* out, uint32_t q_heads,
                                              uint32_t kv_heads, uint32_t head_dim,
                                              uint32_t block_size, uint32_t num_blocks,
                                              float scale,
                                              void* stream /* cudaStream_t; null = default */) {
    if (head_dim != uint32_t(HEAD_DIM) || q_heads == 0 || kv_heads == 0 ||
        q_heads % kv_heads != 0 || block_size != BLOCK_SIZE || num_blocks == 0 ||
        !std::isfinite(scale) || !(scale > 0.0f) || q_heads > uint32_t(INT_MAX) ||
        num_blocks > uint32_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const __nv_bfloat16* qp = static_cast<const __nv_bfloat16*>(q);
    const __nv_bfloat16* kp = static_cast<const __nv_bfloat16*>(k_cache);
    const __nv_bfloat16* vp = static_cast<const __nv_bfloat16*>(v_cache);
    const uint32_t* tp = static_cast<const uint32_t*>(block_table);
    const uint32_t* sp = static_cast<const uint32_t*>(sequence);
    __nv_bfloat16* op = static_cast<__nv_bfloat16*>(out);
    int ikv = static_cast<int>(kv_heads);
    int group = static_cast<int>(q_heads / kv_heads);
    float c = scale * LOG2E;
    void* args[] = {&qp, &kp, &vp, &tp, &sp, &op, &ikv, &group, &num_blocks, &c};
    const dim3 grid(q_heads), block(THREADS);
    return static_cast<int>(cudaLaunchKernel(attn_decode_bf16_kernel, grid, block, args, 0,
                                             static_cast<cudaStream_t>(stream)));
}
