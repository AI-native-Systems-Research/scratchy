// SPDX-License-Identifier: Apache-2.0
// Split-KV flash-decode for ARBITRARY head_dim — the fast path for Gemma-4
// GLOBAL attention (head_dim 512 > the vllm-flash-attn cap of 256) when the
// query count is small (decode: num_q == 1, or a short continuation).
//
// Why this exists: `sdpa_naive_kernel` launches a grid of only
// (num_q, num_q_heads) blocks and each block streams ALL key positions
// serially with a full tree reduction + several __syncthreads per key. At
// decode (num_q == 1) that is just `num_q_heads` blocks (e.g. 32) — a tiny
// fraction of the GPU — each doing an O(num_k) serial loop. For a multi-k
// context that kernel dominates decode wall-clock (measured >120 ms/call at
// ~4k ctx). This kernel parallelizes the KV dimension across `num_splits`
// blocks per (query, head) and combines the partial softmaxes in a second
// pass, so decode saturates the GPU and drops to well under a millisecond.
//
// Two passes on the same stream (ordered, no host sync between them):
//   1. `sdpa_flash_decode_split_kernel`: grid (num_q*num_q_heads, num_splits),
//      block = head_dim threads. Each block runs an online (numerically
//      stable) softmax over its key sub-range and writes a partial
//      (accumulator, running-max m, running-denom l).
//   2. `sdpa_flash_decode_combine_kernel`: grid (num_q*num_q_heads),
//      block = head_dim threads. Rescales and sums the `num_splits` partials
//      into the final output row.
//
// Layouts (contiguous, row-major), identical to sdpa_naive:
//   q:   [num_q,  num_q_heads,  head_dim]
//   k/v: [num_k,  num_kv_heads, head_dim]
//   out: [num_q,  num_q_heads,  head_dim]
// GQA: kv_head = q_head / (num_q_heads / num_kv_heads).
// Causal + optional sliding window use absolute positions:
//   query token i has position q_pos_base + i; key j has position j; i attends
//   to j iff j <= q_pos and (window < 0 || q_pos - j < window).
//
// Scratch (caller-allocated, f32):
//   partial_out: [num_q*num_q_heads, num_splits, head_dim]
//   partial_m:   [num_q*num_q_heads, num_splits]
//   partial_l:   [num_q*num_q_heads, num_splits]

#include <cstdint>
#include <cuda_bf16.h>
#include <math.h>

#include "sdpa_reduce.cuh"  // block_reduce_sum

template <typename T>
__global__ void sdpa_flash_decode_split_kernel(
    const T* __restrict__ q,       // [Nq, Hq, D]
    const T* __restrict__ k,       // [Nk, Hkv, D]
    const T* __restrict__ v,       // [Nk, Hkv, D]
    float* __restrict__ partial_out,  // [Nq*Hq, S, D]
    float* __restrict__ partial_m,    // [Nq*Hq, S]
    float* __restrict__ partial_l,    // [Nq*Hq, S]
    int num_q, int num_k,
    int num_q_heads, int num_kv_heads,
    int head_dim,
    float scale, float softcap,
    int window, int q_pos_base,
    int num_splits)
{
    const int qh_flat = blockIdx.x;       // qi * num_q_heads + qh
    const int split = blockIdx.y;         // [0, num_splits)
    const int qi = qh_flat / num_q_heads;
    const int qh = qh_flat % num_q_heads;
    if (qi >= num_q) return;

    const int kv_h = qh / (num_q_heads / num_kv_heads);
    const int tid = threadIdx.x;          // 0..head_dim-1
    const int q_pos = q_pos_base + qi;

    __shared__ float red[32];             // warp partials (num_warps <= 32)
    __shared__ float s_score;             // broadcast dot-product score

    // Per-split key range.
    const int per = (num_k + num_splits - 1) / num_splits;
    const int k0 = split * per;
    int k1 = k0 + per;
    if (k1 > num_k) k1 = num_k;

    const float qd = (float)q[(size_t)(qi * num_q_heads + qh) * head_dim + tid];
    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f;                     // this thread owns output dim `tid`

    for (int kj = k0; kj < k1; ++kj) {
        const int k_pos = kj;
        if (k_pos > q_pos) break;                          // keys ordered
        if (window >= 0 && (q_pos - k_pos) >= window) continue;

        const float kd = (float)k[(size_t)(kj * num_kv_heads + kv_h) * head_dim + tid];
        const float dot = block_reduce_sum(qd * kd, red);
        if (tid == 0) {
            float sc = dot * scale;
            if (softcap > 0.0f) sc = softcap * tanhf(sc / softcap);
            s_score = sc;
        }
        __syncthreads();
        const float score = s_score;

        const float new_m = fmaxf(m, score);
        const float alpha = (m == -INFINITY) ? 0.0f : __expf(m - new_m);
        const float p = __expf(score - new_m);
        l = l * alpha + p;
        const float vd = (float)v[(size_t)(kj * num_kv_heads + kv_h) * head_dim + tid];
        acc = acc * alpha + p * vd;
        m = new_m;
        __syncthreads();  // all threads read s_score before warp 0 overwrites it
    }

    const size_t po = ((size_t)qh_flat * num_splits + split) * head_dim + tid;
    partial_out[po] = acc;
    if (tid == 0) {
        partial_m[(size_t)qh_flat * num_splits + split] = m;
        partial_l[(size_t)qh_flat * num_splits + split] = l;
    }
}

template <typename T>
__global__ void sdpa_flash_decode_combine_kernel(
    const float* __restrict__ partial_out,  // [Nq*Hq, S, D]
    const float* __restrict__ partial_m,     // [Nq*Hq, S]
    const float* __restrict__ partial_l,     // [Nq*Hq, S]
    T* __restrict__ out,                     // [Nq, Hq, D]
    int num_q, int num_q_heads, int head_dim,
    int num_splits)
{
    const int qh_flat = blockIdx.x;
    const int qi = qh_flat / num_q_heads;
    if (qi >= num_q) return;
    const int tid = threadIdx.x;

    extern __shared__ float sh[];   // [num_splits] m, then [num_splits] l
    float* sm = sh;
    float* sl = sh + num_splits;
    for (int s = tid; s < num_splits; s += blockDim.x) {
        sm[s] = partial_m[(size_t)qh_flat * num_splits + s];
        sl[s] = partial_l[(size_t)qh_flat * num_splits + s];
    }
    __syncthreads();

    // Global max over splits (skip empty splits: m == -inf, l == 0).
    float gm = -INFINITY;
    for (int s = 0; s < num_splits; ++s) {
        if (sl[s] > 0.0f) gm = fmaxf(gm, sm[s]);
    }
    float denom = 0.0f;
    float o = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        if (sl[s] <= 0.0f) continue;
        const float w = __expf(sm[s] - gm);
        denom += sl[s] * w;
        o += partial_out[((size_t)qh_flat * num_splits + s) * head_dim + tid] * w;
    }
    const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    const float res = o * inv;

    T* ovec = out + (size_t)qh_flat * head_dim;
    if (sizeof(T) == 2) {
        __nv_bfloat16 ob = __float2bfloat16(res);
        ((uint16_t*)ovec)[tid] = *reinterpret_cast<uint16_t*>(&ob);
    } else {
        ovec[tid] = (T)res;
    }
}

extern "C" {

// Launch both passes on `stream`. Scratch buffers are caller-allocated and
// sized for the given num_splits (see file header). head_dim == blockDim.x,
// a power of two <= 1024.
void sdpa_flash_decode_bf16(
    const void* q, const void* k, const void* v, void* out,
    float* partial_out, float* partial_m, float* partial_l,
    int num_q, int num_k, int num_q_heads, int num_kv_heads, int head_dim,
    float scale, float softcap, int window, int q_pos_base,
    int num_splits, cudaStream_t stream)
{
    dim3 split_grid(num_q * num_q_heads, num_splits, 1);
    dim3 block(head_dim, 1, 1);
    sdpa_flash_decode_split_kernel<__nv_bfloat16><<<split_grid, block, 0, stream>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
        partial_out, partial_m, partial_l,
        num_q, num_k, num_q_heads, num_kv_heads, head_dim,
        scale, softcap, window, q_pos_base, num_splits);

    dim3 comb_grid(num_q * num_q_heads, 1, 1);
    size_t comb_shmem = (size_t)num_splits * 2 * sizeof(float);
    sdpa_flash_decode_combine_kernel<__nv_bfloat16><<<comb_grid, block, comb_shmem, stream>>>(
        partial_out, partial_m, partial_l,
        (__nv_bfloat16*)out,
        num_q, num_q_heads, head_dim, num_splits);
}

} // extern "C"
