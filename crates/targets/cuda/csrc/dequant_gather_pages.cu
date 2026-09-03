// SPDX-License-Identifier: Apache-2.0
// FP8 dequantize + gather pages CUDA kernel.
//
// Reads FP8 E4M3 pages from a paged KV cache using a block table,
// dequantizes to BF16/F16, and writes a contiguous output tensor
// [total_kv_tokens, num_heads, head_dim].
//
// This allows using our existing BF16 FlashAttention-2 kernels with
// FP8 KV cache by dequantizing on-the-fly before attention.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "fp8_utils.cuh"

// ---------------------------------------------------------------------------
// Dequant+gather kernel (BF16 output)
// ---------------------------------------------------------------------------
// Grid:  total_kv_tokens blocks (one per output token position)
// Block: min(num_heads * head_dim, 1024) threads
//
// Each block maps a flat token index to (seq_idx, pos_in_seq) using
// cu_seqlens_k (prefix sum of sequence lengths), then looks up the
// page block from block_table, reads FP8 from cache, dequants to
// BF16, and writes to contiguous output.

__global__ void dequant_gather_pages_bf16_kernel(
    const uint8_t* __restrict__ cache,       // [num_blocks, block_size, num_heads, head_dim] FP8
    const int32_t* __restrict__ block_table,  // [batch_size, max_pages_per_seq]
    const int32_t* __restrict__ cu_seqlens_k, // [batch_size + 1] prefix sum
    float scale,
    int num_heads,
    int head_dim,
    int block_size,
    int max_pages_per_seq,
    int batch_size,
    __nv_bfloat16* __restrict__ output)       // [total_kv_tokens, num_heads, head_dim]
{
    const int token_flat = blockIdx.x;  // flat position across all sequences
    const int n_elems = num_heads * head_dim;

    // Binary search to find which sequence this flat token belongs to.
    int seq_idx = 0;
    int lo = 0, hi = batch_size;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cu_seqlens_k[mid + 1] <= token_flat) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    seq_idx = lo;

    // Bounds check: blocks beyond actual total_kv_tokens exit early.
    // This enables over-sized grid launch during CUDA graph capture.
    if (seq_idx >= batch_size) return;

    // Position within this sequence
    const int pos_in_seq = token_flat - cu_seqlens_k[seq_idx];

    // Map to page block
    const int page_idx = pos_in_seq / block_size;
    const int page_offset = pos_in_seq % block_size;
    const int block_id = block_table[seq_idx * max_pages_per_seq + page_idx];

    // Source: FP8 cache at [block_id, page_offset, :, :]
    const uint8_t* src = cache + (block_id * block_size + page_offset) * n_elems;
    // Destination: BF16 output at [token_flat, :, :]
    __nv_bfloat16* dst = output + token_flat * n_elems;

    // Vectorized: load 16 FP8 bytes, convert to 16 BF16, store as 2×uint4
    constexpr int VEC_FP8 = 16;   // 16 FP8 bytes = 16 elements
    const int num_vecs = n_elems / VEC_FP8;
    const int tail_start = num_vecs * VEC_FP8;

    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        // Load 16 FP8 bytes as uint4 (16 bytes)
        uint4 fp8_vec = *reinterpret_cast<const uint4*>(&src[vi * VEC_FP8]);
        uint8_t* fp8_bytes = reinterpret_cast<uint8_t*>(&fp8_vec);

        // Convert 16 FP8 → 16 BF16 (stored in 2 uint4s = 32 bytes)
        __nv_bfloat16 bf16_vals[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            bf16_vals[i] = fp8e4m3_to_bf16(
                fp8_bytes[i], scale);
        }

        // Store 16 BF16 values = 32 bytes = 2 × uint4
        *reinterpret_cast<uint4*>(&dst[vi * VEC_FP8]) =
            *reinterpret_cast<uint4*>(&bf16_vals[0]);
        *reinterpret_cast<uint4*>(&dst[vi * VEC_FP8 + 8]) =
            *reinterpret_cast<uint4*>(&bf16_vals[8]);
    }

    // Scalar tail
    for (int i = tail_start + threadIdx.x; i < n_elems; i += blockDim.x) {
        dst[i] = fp8e4m3_to_bf16(
            src[i], scale);
    }
}

// ---------------------------------------------------------------------------
// Dequant+gather kernel (F16 output)
// ---------------------------------------------------------------------------

__global__ void dequant_gather_pages_f16_kernel(
    const uint8_t* __restrict__ cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cu_seqlens_k,
    float scale,
    int num_heads,
    int head_dim,
    int block_size,
    int max_pages_per_seq,
    int batch_size,
    __half* __restrict__ output)
{
    const int token_flat = blockIdx.x;
    const int n_elems = num_heads * head_dim;

    int seq_idx = 0;
    int lo = 0, hi = batch_size;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cu_seqlens_k[mid + 1] <= token_flat) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    seq_idx = lo;

    // Bounds check: blocks beyond actual total_kv_tokens exit early.
    if (seq_idx >= batch_size) return;

    const int pos_in_seq = token_flat - cu_seqlens_k[seq_idx];
    const int page_idx = pos_in_seq / block_size;
    const int page_offset = pos_in_seq % block_size;
    const int block_id = block_table[seq_idx * max_pages_per_seq + page_idx];

    const uint8_t* src = cache + (block_id * block_size + page_offset) * n_elems;
    __half* dst = output + token_flat * n_elems;

    constexpr int VEC_FP8 = 16;
    const int num_vecs = n_elems / VEC_FP8;
    const int tail_start = num_vecs * VEC_FP8;

    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        uint4 fp8_vec = *reinterpret_cast<const uint4*>(&src[vi * VEC_FP8]);
        uint8_t* fp8_bytes = reinterpret_cast<uint8_t*>(&fp8_vec);

        __half f16_vals[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            f16_vals[i] = fp8e4m3_to_f16(
                fp8_bytes[i], scale);
        }

        *reinterpret_cast<uint4*>(&dst[vi * VEC_FP8]) =
            *reinterpret_cast<uint4*>(&f16_vals[0]);
        *reinterpret_cast<uint4*>(&dst[vi * VEC_FP8 + 8]) =
            *reinterpret_cast<uint4*>(&f16_vals[8]);
    }

    for (int i = tail_start + threadIdx.x; i < n_elems; i += blockDim.x) {
        dst[i] = fp8e4m3_to_f16(
            src[i], scale);
    }
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

extern "C" {

void dequant_gather_pages_bf16(
    const uint8_t* cache,
    const int32_t* block_table,
    const int32_t* cu_seqlens_k,
    float scale,
    int total_kv_tokens,
    int num_heads,
    int head_dim,
    int block_size,
    int max_pages_per_seq,
    int batch_size,
    uint16_t* output,
    cudaStream_t stream)
{
    if (total_kv_tokens == 0) return;
    int n_elems = num_heads * head_dim;
    int threads = (n_elems < 1024) ? n_elems : 1024;
    dequant_gather_pages_bf16_kernel<<<total_kv_tokens, threads, 0, stream>>>(
        cache, block_table, cu_seqlens_k, scale,
        num_heads, head_dim, block_size, max_pages_per_seq, batch_size,
        reinterpret_cast<__nv_bfloat16*>(output));
}

void dequant_gather_pages_f16(
    const uint8_t* cache,
    const int32_t* block_table,
    const int32_t* cu_seqlens_k,
    float scale,
    int total_kv_tokens,
    int num_heads,
    int head_dim,
    int block_size,
    int max_pages_per_seq,
    int batch_size,
    uint16_t* output,
    cudaStream_t stream)
{
    if (total_kv_tokens == 0) return;
    int n_elems = num_heads * head_dim;
    int threads = (n_elems < 1024) ? n_elems : 1024;
    dequant_gather_pages_f16_kernel<<<total_kv_tokens, threads, 0, stream>>>(
        cache, block_table, cu_seqlens_k, scale,
        num_heads, head_dim, block_size, max_pages_per_seq, batch_size,
        reinterpret_cast<__half*>(output));
}

} // extern "C"

// ---------------------------------------------------------------------------
// BF16 paged gather (NO dequant) — Gemma-4 GLOBAL attention (head_dim 512 >
// flash cap) gathers K or V from the paged cache into a contiguous
// [total_kv, num_heads, head_dim] BF16 buffer, then runs naive SDPA.
//
// Single-sequence (bs=1) specialization: the global-attn path is bs=1, so this
// takes `total_kv` directly instead of cu_seqlens_k — no host readback, no H2D.
// That makes it CUDA-graph-capture-safe (unlike the old host-driven D2H +
// per-block D2D loop, which was illegal inside capture). Over-sized grid is
// safe: blocks with `token >= total_kv` exit early.
//
// Grid:  >= total_kv blocks (one per output token); Block: min(n_elems, 1024).
__global__ void gather_pages_bf16_kernel(
    const __nv_bfloat16* __restrict__ cache,   // [num_blocks, block_size, num_heads, head_dim]
    const int32_t* __restrict__ block_table,   // [1, max_pages] (single seq)
    int total_kv,
    int num_heads,
    int head_dim,
    int block_size,
    __nv_bfloat16* __restrict__ output)        // [total_kv, num_heads, head_dim]
{
    const int token = blockIdx.x;
    if (token >= total_kv) return;  // over-sized-grid / capture-safe guard

    const int n_elems = num_heads * head_dim;
    const int page_idx = token / block_size;
    const int page_offset = token % block_size;
    const int block_id = block_table[page_idx];

    const __nv_bfloat16* src = cache + (block_id * block_size + page_offset) * n_elems;
    __nv_bfloat16* dst = output + token * n_elems;

    // Vectorized copy: uint4 = 16 bytes = 8 bf16.
    constexpr int VEC = 8;
    const int num_vecs = n_elems / VEC;
    const int tail_start = num_vecs * VEC;
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        reinterpret_cast<uint4*>(dst)[vi] = reinterpret_cast<const uint4*>(src)[vi];
    }
    for (int i = tail_start + threadIdx.x; i < n_elems; i += blockDim.x) {
        dst[i] = src[i];
    }
}

extern "C" void gather_pages_bf16(
    const uint16_t* cache,
    const int32_t* block_table,
    int total_kv,
    int num_heads,
    int head_dim,
    int block_size,
    uint16_t* output,
    cudaStream_t stream)
{
    if (total_kv == 0) return;
    int n_elems = num_heads * head_dim;
    int threads = (n_elems < 1024) ? n_elems : 1024;
    gather_pages_bf16_kernel<<<total_kv, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(cache),
        block_table, total_kv, num_heads, head_dim, block_size,
        reinterpret_cast<__nv_bfloat16*>(output));
}
