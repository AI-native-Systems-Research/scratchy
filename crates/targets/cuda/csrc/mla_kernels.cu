// SPDX-License-Identifier: Apache-2.0
// MLA (Multi-Latent Attention) data movement kernels for DeepSeek V2/V3.
//
// These kernels replace CPU-driven per-token per-head D2D copy loops with
// proper GPU-parallel operations, matching what Python does with vectorized
// tensor ops (split, cat, slice, zero-pad, broadcast).

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// 1. split_kv_a: split [num_tokens, kv_lora_rank + rope_dim] into
//    latent [num_tokens, kv_lora_rank] and k_pe [num_tokens, rope_dim]
// ---------------------------------------------------------------------------

template <typename T>
__global__ void mla_split_kv_a_kernel(
    const T* __restrict__ src,    // [num_tokens, kv_lora_rank + rope_dim]
    T* __restrict__ dst_latent,   // [num_tokens, kv_lora_rank]
    T* __restrict__ dst_k_pe,     // [num_tokens, rope_dim]
    int num_tokens,
    int kv_lora_rank,
    int rope_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_row = kv_lora_rank + rope_dim;
    int total = num_tokens * total_row;
    if (idx >= total) return;

    int t = idx / total_row;
    int d = idx % total_row;
    T val = src[t * total_row + d];

    if (d < kv_lora_rank) {
        dst_latent[t * kv_lora_rank + d] = val;
    } else {
        dst_k_pe[t * rope_dim + (d - kv_lora_rank)] = val;
    }
}

// ---------------------------------------------------------------------------
// 2. extract_q_pe: extract rope portion from Q
//    src [num_tokens, num_heads * qk_head_dim] ->
//    dst [num_tokens, num_heads * rope_dim]  (last rope_dim of each head)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void mla_extract_q_pe_kernel(
    const T* __restrict__ src,  // [num_tokens, num_heads * qk_head_dim]
    T* __restrict__ dst,        // [num_tokens, num_heads * rope_dim]
    int num_tokens,
    int num_heads,
    int qk_head_dim,
    int qk_nope_head_dim,
    int rope_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * num_heads * rope_dim;
    if (idx >= total) return;

    int t = idx / (num_heads * rope_dim);
    int rem = idx % (num_heads * rope_dim);
    int h = rem / rope_dim;
    int d = rem % rope_dim;

    dst[idx] = src[t * num_heads * qk_head_dim + h * qk_head_dim + qk_nope_head_dim + d];
}

// ---------------------------------------------------------------------------
// 3. write_q_pe: write rope portion back into Q after RoPE
//    src [num_tokens, num_heads * rope_dim] ->
//    dst [num_tokens, num_heads * qk_head_dim]  (last rope_dim of each head)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void mla_write_q_pe_kernel(
    const T* __restrict__ src,  // [num_tokens, num_heads * rope_dim]
    T* __restrict__ dst,        // [num_tokens, num_heads * qk_head_dim]
    int num_tokens,
    int num_heads,
    int qk_head_dim,
    int qk_nope_head_dim,
    int rope_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * num_heads * rope_dim;
    if (idx >= total) return;

    int t = idx / (num_heads * rope_dim);
    int rem = idx % (num_heads * rope_dim);
    int h = rem / rope_dim;
    int d = rem % rope_dim;

    dst[t * num_heads * qk_head_dim + h * qk_head_dim + qk_nope_head_dim + d] = src[idx];
}

// ---------------------------------------------------------------------------
// 4. assemble_k: concat k_nope from kv_b + broadcast k_pe
//    kv_b [num_tokens, num_heads * (nope_dim + v_head_dim)]
//    k_pe [num_tokens, rope_dim]  (single head, broadcast to all)
//    dst  [num_tokens, num_heads * qk_head_dim]
// ---------------------------------------------------------------------------

template <typename T>
__global__ void mla_assemble_k_kernel(
    const T* __restrict__ kv_b,   // [num_tokens, num_heads * (nope_dim + v_head_dim)]
    const T* __restrict__ k_pe,   // [num_tokens, rope_dim]
    T* __restrict__ dst,          // [num_tokens, num_heads * qk_head_dim]
    int num_tokens,
    int num_heads,
    int qk_nope_head_dim,
    int qk_rope_head_dim,
    int v_head_dim,
    int qk_head_dim  // = nope + rope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * num_heads * qk_head_dim;
    if (idx >= total) return;

    int t = idx / (num_heads * qk_head_dim);
    int rem = idx % (num_heads * qk_head_dim);
    int h = rem / qk_head_dim;
    int d = rem % qk_head_dim;

    int kv_b_stride = num_heads * (qk_nope_head_dim + v_head_dim);

    if (d < qk_nope_head_dim) {
        // k_nope from kv_b: head h, dim d
        dst[idx] = kv_b[t * kv_b_stride + h * (qk_nope_head_dim + v_head_dim) + d];
    } else {
        // k_pe: broadcast from single head
        int rope_d = d - qk_nope_head_dim;
        dst[idx] = k_pe[t * qk_rope_head_dim + rope_d];
    }
}

// ---------------------------------------------------------------------------
// 5. assemble_v: copy v_head_dim from kv_b into zero-padded V
//    kv_b [num_tokens, num_heads * (nope_dim + v_head_dim)]
//    dst  [num_tokens, num_heads * qk_head_dim]  (zero-padded)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void mla_assemble_v_kernel(
    const T* __restrict__ kv_b,  // [num_tokens, num_heads * (nope_dim + v_head_dim)]
    T* __restrict__ dst,         // [num_tokens, num_heads * qk_head_dim] (pre-zeroed)
    int num_tokens,
    int num_heads,
    int qk_nope_head_dim,
    int v_head_dim,
    int qk_head_dim
) {
    // One thread per V element: num_tokens * num_heads * v_head_dim
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * num_heads * v_head_dim;
    if (idx >= total) return;

    int t = idx / (num_heads * v_head_dim);
    int rem = idx % (num_heads * v_head_dim);
    int h = rem / v_head_dim;
    int d = rem % v_head_dim;

    int kv_b_stride = num_heads * (qk_nope_head_dim + v_head_dim);
    T val = kv_b[t * kv_b_stride + h * (qk_nope_head_dim + v_head_dim) + qk_nope_head_dim + d];
    dst[t * num_heads * qk_head_dim + h * qk_head_dim + d] = val;
}

// ---------------------------------------------------------------------------
// 6. slice_attn_output: slice from qk_head_dim to v_head_dim per head
//    src [num_tokens, num_heads * qk_head_dim]
//    dst [num_tokens, num_heads * v_head_dim]
// ---------------------------------------------------------------------------

template <typename T>
__global__ void mla_slice_attn_output_kernel(
    const T* __restrict__ src,  // [num_tokens, num_heads * qk_head_dim]
    T* __restrict__ dst,        // [num_tokens, num_heads * v_head_dim]
    int num_tokens,
    int num_heads,
    int qk_head_dim,
    int v_head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * num_heads * v_head_dim;
    if (idx >= total) return;

    int t = idx / (num_heads * v_head_dim);
    int rem = idx % (num_heads * v_head_dim);
    int h = rem / v_head_dim;
    int d = rem % v_head_dim;

    dst[idx] = src[t * num_heads * qk_head_dim + h * qk_head_dim + d];
}

// ---------------------------------------------------------------------------
// Extern "C" entry points — typed variants
// ---------------------------------------------------------------------------

#define LAUNCH_GRID(total, block) (((total) + (block) - 1) / (block))

#define DEFINE_MLA_SPLIT_KV_A(SUFFIX, T) \
extern "C" void mla_split_kv_a_##SUFFIX( \
    const T* src, T* dst_latent, T* dst_k_pe, \
    int num_tokens, int kv_lora_rank, int rope_dim, \
    cudaStream_t stream \
) { \
    int total = num_tokens * (kv_lora_rank + rope_dim); \
    int block = 256; \
    mla_split_kv_a_kernel<<<LAUNCH_GRID(total, block), block, 0, stream>>>( \
        src, dst_latent, dst_k_pe, num_tokens, kv_lora_rank, rope_dim); \
}

DEFINE_MLA_SPLIT_KV_A(f32, float)
DEFINE_MLA_SPLIT_KV_A(f16, __half)
DEFINE_MLA_SPLIT_KV_A(bf16, __nv_bfloat16)

#define DEFINE_MLA_EXTRACT_Q_PE(SUFFIX, T) \
extern "C" void mla_extract_q_pe_##SUFFIX( \
    const T* src, T* dst, \
    int num_tokens, int num_heads, int qk_head_dim, int qk_nope_head_dim, int rope_dim, \
    cudaStream_t stream \
) { \
    int total = num_tokens * num_heads * rope_dim; \
    int block = 256; \
    mla_extract_q_pe_kernel<<<LAUNCH_GRID(total, block), block, 0, stream>>>( \
        src, dst, num_tokens, num_heads, qk_head_dim, qk_nope_head_dim, rope_dim); \
}

DEFINE_MLA_EXTRACT_Q_PE(f32, float)
DEFINE_MLA_EXTRACT_Q_PE(f16, __half)
DEFINE_MLA_EXTRACT_Q_PE(bf16, __nv_bfloat16)

#define DEFINE_MLA_WRITE_Q_PE(SUFFIX, T) \
extern "C" void mla_write_q_pe_##SUFFIX( \
    const T* src, T* dst, \
    int num_tokens, int num_heads, int qk_head_dim, int qk_nope_head_dim, int rope_dim, \
    cudaStream_t stream \
) { \
    int total = num_tokens * num_heads * rope_dim; \
    int block = 256; \
    mla_write_q_pe_kernel<<<LAUNCH_GRID(total, block), block, 0, stream>>>( \
        src, dst, num_tokens, num_heads, qk_head_dim, qk_nope_head_dim, rope_dim); \
}

DEFINE_MLA_WRITE_Q_PE(f32, float)
DEFINE_MLA_WRITE_Q_PE(f16, __half)
DEFINE_MLA_WRITE_Q_PE(bf16, __nv_bfloat16)

#define DEFINE_MLA_ASSEMBLE_K(SUFFIX, T) \
extern "C" void mla_assemble_k_##SUFFIX( \
    const T* kv_b, const T* k_pe, T* dst, \
    int num_tokens, int num_heads, int qk_nope_head_dim, int qk_rope_head_dim, \
    int v_head_dim, int qk_head_dim, \
    cudaStream_t stream \
) { \
    int total = num_tokens * num_heads * qk_head_dim; \
    int block = 256; \
    mla_assemble_k_kernel<<<LAUNCH_GRID(total, block), block, 0, stream>>>( \
        kv_b, k_pe, dst, num_tokens, num_heads, qk_nope_head_dim, qk_rope_head_dim, \
        v_head_dim, qk_head_dim); \
}

DEFINE_MLA_ASSEMBLE_K(f32, float)
DEFINE_MLA_ASSEMBLE_K(f16, __half)
DEFINE_MLA_ASSEMBLE_K(bf16, __nv_bfloat16)

#define DEFINE_MLA_ASSEMBLE_V(SUFFIX, T) \
extern "C" void mla_assemble_v_##SUFFIX( \
    const T* kv_b, T* dst, \
    int num_tokens, int num_heads, int qk_nope_head_dim, int v_head_dim, int qk_head_dim, \
    cudaStream_t stream \
) { \
    int total = num_tokens * num_heads * v_head_dim; \
    int block = 256; \
    mla_assemble_v_kernel<<<LAUNCH_GRID(total, block), block, 0, stream>>>( \
        kv_b, dst, num_tokens, num_heads, qk_nope_head_dim, v_head_dim, qk_head_dim); \
}

DEFINE_MLA_ASSEMBLE_V(f32, float)
DEFINE_MLA_ASSEMBLE_V(f16, __half)
DEFINE_MLA_ASSEMBLE_V(bf16, __nv_bfloat16)

#define DEFINE_MLA_SLICE_ATTN_OUTPUT(SUFFIX, T) \
extern "C" void mla_slice_attn_output_##SUFFIX( \
    const T* src, T* dst, \
    int num_tokens, int num_heads, int qk_head_dim, int v_head_dim, \
    cudaStream_t stream \
) { \
    int total = num_tokens * num_heads * v_head_dim; \
    int block = 256; \
    mla_slice_attn_output_kernel<<<LAUNCH_GRID(total, block), block, 0, stream>>>( \
        src, dst, num_tokens, num_heads, qk_head_dim, v_head_dim); \
}

DEFINE_MLA_SLICE_ATTN_OUTPUT(f32, float)
DEFINE_MLA_SLICE_ATTN_OUTPUT(f16, __half)
DEFINE_MLA_SLICE_ATTN_OUTPUT(bf16, __nv_bfloat16)
