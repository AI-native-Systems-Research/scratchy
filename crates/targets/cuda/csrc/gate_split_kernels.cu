// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5 attention output-gate split — per-head deinterleave of the
// DOUBLED q_proj output into `query` and `gate`.
//
// Mirrors crates/targets/metal/shaders/gate_split.metal byte-for-byte.
// transformers Qwen3_5Attention:
//   q_proj(x).view(*, num_heads, 2*head_dim) -> chunk(2, dim=-1)
// i.e. each head's 2*head_dim block is [query(head_dim) | gate(head_dim)].
//
//   Input  qg:    [M, num_heads * 2 * head_dim]   (T-typed, contiguous)
//   Output q_out: [M, num_heads * head_dim]       (T-typed, contiguous)
//   Output gate:  [M, num_heads * head_dim]       (T-typed, contiguous)
//
// 1 thread per output element; each thread copies one (q, gate) pair.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void gate_split_kernel(
    T* __restrict__ q_out,            // [M, num_heads * head_dim]
    T* __restrict__ gate_out,         // [M, num_heads * head_dim]
    const T* __restrict__ qg,         // [M, num_heads * 2 * head_dim]
    int n,                            // M * num_heads * head_dim
    int head_dim,
    int num_heads)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    const int hd   = head_dim;
    const int cols = num_heads * hd;          // per-output row width
    const int row  = gid / cols;
    const int rem  = gid - row * cols;
    const int head = rem / hd;
    const int d    = rem - head * hd;
    // qg row width = num_heads * 2 * head_dim; within a head: [query | gate].
    const int base = row * (cols * 2) + head * (2 * hd) + d;
    q_out[gid]    = qg[base];
    gate_out[gid] = qg[base + hd];
}

extern "C" {

void gate_split_bf16(
    void* q_out, void* gate_out, const void* qg,
    int num_tokens, int num_heads, int head_dim,
    cudaStream_t stream)
{
    const int n = num_tokens * num_heads * head_dim;
    if (n == 0) return;
    constexpr int THREADS = 256;
    const int blocks = (n + THREADS - 1) / THREADS;
    gate_split_kernel<__nv_bfloat16><<<blocks, THREADS, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(q_out),
        reinterpret_cast<__nv_bfloat16*>(gate_out),
        reinterpret_cast<const __nv_bfloat16*>(qg),
        n, head_dim, num_heads);
}

void gate_split_f16(
    void* q_out, void* gate_out, const void* qg,
    int num_tokens, int num_heads, int head_dim,
    cudaStream_t stream)
{
    const int n = num_tokens * num_heads * head_dim;
    if (n == 0) return;
    constexpr int THREADS = 256;
    const int blocks = (n + THREADS - 1) / THREADS;
    gate_split_kernel<__half><<<blocks, THREADS, 0, stream>>>(
        reinterpret_cast<__half*>(q_out),
        reinterpret_cast<__half*>(gate_out),
        reinterpret_cast<const __half*>(qg),
        n, head_dim, num_heads);
}

} // extern "C"
