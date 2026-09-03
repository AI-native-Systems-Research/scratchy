// SPDX-License-Identifier: Apache-2.0
// Naive (non-flash) scaled-dot-product attention for ARBITRARY head_dim —
// used for Gemma-4 GLOBAL attention (head_dim 512), which exceeds the
// vllm-flash-attn cap of 256 (HEADDIM_SWITCH instantiates ≤256). Correctness-
// first: one threadblock per (query position, query head); the block streams
// over all key positions with an online (numerically-stable) softmax and
// accumulates the value vector. head_dim ≤ 1024 (block of `head_dim` threads;
// the running output lives in shared memory as f32).
//
// Layouts (contiguous, row-major):
//   q: [num_q_tokens, num_q_heads,  head_dim]
//   k: [num_kv_tokens, num_kv_heads, head_dim]
//   v: [num_kv_tokens, num_kv_heads, head_dim]
//   out: [num_q_tokens, num_q_heads, head_dim]
// GQA: kv_head = q_head / (num_q_heads / num_kv_heads).
//
// Causal mask uses absolute positions: query token i (global pos
// q_pos_base + i) attends to key token j iff key_pos(j) <= query_pos(i) and
// (window < 0 || query_pos(i) - key_pos(j) < window). For a fresh full prefill
// key_pos(j) == j and query_pos(i) == q_pos_base + i.

#include <cstdint>
#include <cuda_bf16.h>
#include <math.h>

#include "sdpa_reduce.cuh"  // block_reduce_sum

template <typename T>
__global__ void sdpa_naive_kernel(
    const T* __restrict__ q,    // [Nq, Hq, D]
    const T* __restrict__ k,    // [Nk, Hkv, D]
    const T* __restrict__ v,    // [Nk, Hkv, D]
    T* __restrict__ out,        // [Nq, Hq, D]
    int num_q, int num_k,
    int num_q_heads, int num_kv_heads,
    int head_dim,
    float scale,
    float softcap,              // 0 => disabled
    int window,                 // <0 => no sliding window
    int q_pos_base)             // absolute position of query token 0
{
    const int qi = blockIdx.x;        // query token index [0, num_q)
    const int qh = blockIdx.y;        // query head index  [0, num_q_heads)
    if (qi >= num_q || qh >= num_q_heads) return;

    const int kv_h = qh / (num_q_heads / num_kv_heads);
    const int tid = threadIdx.x;      // 0..head_dim-1 (blockDim.x == head_dim)
    const int q_pos = q_pos_base + qi;

    // Warp-shuffle block reduction (block_reduce_sum, sdpa_reduce.cuh) replaces
    // the old per-key tree reduction: ~2 __syncthreads/key instead of ~12. The
    // online-softmax state (m, l) and the per-dim output accumulator live in
    // registers; only the reduced score is broadcast via one __shared__.
    __shared__ float red[32];         // warp partials (num_warps == head_dim/32)
    __shared__ float s_score;         // broadcast dot-product score

    const T* qvec = q + (size_t)(qi * num_q_heads + qh) * head_dim;
    const float qd = (float)qvec[tid];
    float m = -INFINITY;              // running max
    float l = 0.0f;                   // running denom (sum exp)
    float acc = 0.0f;                 // this thread owns output dim `tid`

    for (int kj = 0; kj < num_k; ++kj) {
        const int k_pos = kj;  // contiguous prefill: key pos == index
        // Causal + sliding-window mask.
        if (k_pos > q_pos) break;                       // keys are in order
        if (window >= 0 && (q_pos - k_pos) >= window) continue;

        const T* kvec = k + (size_t)(kj * num_kv_heads + kv_h) * head_dim;
        const float dot = block_reduce_sum(qd * (float)kvec[tid], red);
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

    // normalize + write
    float denom = (l > 0.0f) ? l : 1.0f;
    T* ovec = out + (size_t)(qi * num_q_heads + qh) * head_dim;
    float o = acc / denom;
    if (sizeof(T) == 2) {
        __nv_bfloat16 ob = __float2bfloat16(o);
        ((uint16_t*)ovec)[tid] = *reinterpret_cast<uint16_t*>(&ob);
    } else {
        ovec[tid] = (T)o;
    }
}

extern "C" {

void sdpa_naive_bf16(
    const void* q, const void* k, const void* v, void* out,
    int num_q, int num_k, int num_q_heads, int num_kv_heads, int head_dim,
    float scale, float softcap, int window, int q_pos_base,
    cudaStream_t stream)
{
    dim3 grid(num_q, num_q_heads, 1);
    dim3 block(head_dim, 1, 1);
    // acc is register-resident now; only the static red[32]/s_score smem is used.
    sdpa_naive_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
        (__nv_bfloat16*)out, num_q, num_k, num_q_heads, num_kv_heads, head_dim,
        scale, softcap, window, q_pos_base);
}

} // extern "C"
