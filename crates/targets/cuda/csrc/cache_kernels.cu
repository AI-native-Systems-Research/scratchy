// SPDX-License-Identifier: Apache-2.0
// Fused reshape_and_cache CUDA kernel for vLLM Rust.
//
// Port of csrc/cache_kernels.cu from Python vLLM.
// Scatters newly computed K/V tokens into the paged KV cache using
// a slot_mapping that maps each token to its target cache slot.
//
// Cache layout: [num_blocks, block_size, num_kv_heads, head_dim] (NHD)
// This matches the "flash" variant in Python vLLM.
//
// Uses vectorized 128-bit loads/stores for throughput.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "vec_utils.cuh"
#include "fp8_utils.cuh"

// ---------------------------------------------------------------------------
// reshape_and_cache kernel (vectorized)
// ---------------------------------------------------------------------------
// Grid:  num_tokens blocks (one per token)
// Block: min(num_heads * head_dim, 1024) threads
//
// Each token's K/V data ([num_heads, head_dim] elements) is copied from
// the input tensors to the target slot in the paged cache.

template <typename T>
__global__ void reshape_and_cache_kernel(
    const T* __restrict__ key,           // [num_tokens, num_heads, head_dim]
    const T* __restrict__ value,         // [num_tokens, num_heads, head_dim]
    T* __restrict__ key_cache,           // [num_blocks, block_size, num_heads, head_dim]
    T* __restrict__ value_cache,         // [num_blocks, block_size, num_heads, head_dim]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    int num_heads,
    int head_dim,
    int block_size)
{
    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const int token_idx = blockIdx.x;
    const int64_t slot = slot_mapping[token_idx];

    // slot < 0 means padding token — skip.
    if (slot < 0) {
        return;
    }

    const int n_elems = num_heads * head_dim;
    const int num_vecs = n_elems / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    // Source: contiguous input at [token_idx, :, :]
    const T* key_src = key + token_idx * n_elems;
    const T* value_src = value + token_idx * n_elems;

    // Destination: cache[block_idx, block_offset, :, :]
    T* key_dst = key_cache + slot * n_elems;
    T* value_dst = value_cache + slot * n_elems;

    // Vectorized copy.
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        vec_store(&key_dst[vi * VEC_SIZE], vec_load(&key_src[vi * VEC_SIZE]));
        vec_store(&value_dst[vi * VEC_SIZE], vec_load(&value_src[vi * VEC_SIZE]));
    }

    // Scalar tail.
    for (int i = tail_start + threadIdx.x; i < n_elems; i += blockDim.x) {
        key_dst[i] = key_src[i];
        value_dst[i] = value_src[i];
    }
}

// ---------------------------------------------------------------------------
// FP8 E4M3 reshape_and_cache kernel
// ---------------------------------------------------------------------------
// Reads BF16/F16 key/value, quantizes to FP8 E4M3 and writes to FP8 cache.
// Scale convention: fp8_stored = cast(input / scale) = cast(input * inv_scale)
// Each token: load BF16 input → convert to FP8 → store to FP8 cache.
// Vectorized: process 16 elements per iteration (load 2×uint4 of BF16 = 32B,
// convert 16 elements to FP8, store 16 bytes).

template <typename SrcT>
__global__ void reshape_and_cache_fp8_kernel(
    const SrcT* __restrict__ key,          // [num_tokens, num_heads, head_dim] BF16/F16
    const SrcT* __restrict__ value,        // [num_tokens, num_heads, head_dim] BF16/F16
    uint8_t* __restrict__ key_cache,       // [num_blocks, block_size, num_heads, head_dim] FP8
    uint8_t* __restrict__ value_cache,     // same layout
    const int64_t* __restrict__ slot_mapping,
    const float* __restrict__ k_scale,     // GPU scalar
    const float* __restrict__ v_scale,     // GPU scalar
    int num_heads,
    int head_dim,
    int block_size)
{
    const int token_idx = blockIdx.x;
    const int64_t slot = slot_mapping[token_idx];

    if (slot < 0) return;

    const int n_elems = num_heads * head_dim;
    const float k_inv_scale = 1.0f / (*k_scale);
    const float v_inv_scale = 1.0f / (*v_scale);

    // Source: BF16/F16 input
    const SrcT* key_src = key + token_idx * n_elems;
    const SrcT* value_src = value + token_idx * n_elems;

    // Destination: FP8 cache (1 byte per element)
    uint8_t* key_dst = key_cache + slot * n_elems;
    uint8_t* value_dst = value_cache + slot * n_elems;

    // Process 8 elements at a time (one uint4 of BF16/F16 = 16 bytes = 8 elements).
    // Output: 8 FP8 bytes stored as uint2 (8 bytes).
    constexpr int VEC_ELEMS = 8;  // elements per BF16/F16 uint4 load
    const int num_vecs = n_elems / VEC_ELEMS;
    const int tail_start = num_vecs * VEC_ELEMS;

    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        // Load 8 BF16/F16 elements as uint4
        uint4 k_vec = *reinterpret_cast<const uint4*>(&key_src[vi * VEC_ELEMS]);
        uint4 v_vec = *reinterpret_cast<const uint4*>(&value_src[vi * VEC_ELEMS]);

        // Convert to float, then to FP8
        SrcT* k_elems = reinterpret_cast<SrcT*>(&k_vec);
        SrcT* v_elems = reinterpret_cast<SrcT*>(&v_vec);

        uint8_t k_fp8[8], v_fp8[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if constexpr (sizeof(SrcT) == 2) {
                // BF16 or F16 — both are 2 bytes
                k_fp8[i] = bf16_to_fp8e4m3(
                    *reinterpret_cast<__nv_bfloat16*>(&k_elems[i]), k_inv_scale);
                v_fp8[i] = bf16_to_fp8e4m3(
                    *reinterpret_cast<__nv_bfloat16*>(&v_elems[i]), v_inv_scale);
            }
        }

        // Store 8 FP8 bytes as uint2 (8 bytes)
        *reinterpret_cast<uint2*>(&key_dst[vi * VEC_ELEMS]) =
            *reinterpret_cast<uint2*>(k_fp8);
        *reinterpret_cast<uint2*>(&value_dst[vi * VEC_ELEMS]) =
            *reinterpret_cast<uint2*>(v_fp8);
    }

    // Scalar tail
    for (int i = tail_start + threadIdx.x; i < n_elems; i += blockDim.x) {
        if constexpr (sizeof(SrcT) == 2) {
            key_dst[i] = bf16_to_fp8e4m3(
                *reinterpret_cast<const __nv_bfloat16*>(&key_src[i]), k_inv_scale);
            value_dst[i] = bf16_to_fp8e4m3(
                *reinterpret_cast<const __nv_bfloat16*>(&value_src[i]), v_inv_scale);
        }
    }
}

// Specialization for F16 input (uses f16_to_fp8e4m3 instead)
__global__ void reshape_and_cache_fp8_f16_kernel(
    const __half* __restrict__ key,
    const __half* __restrict__ value,
    uint8_t* __restrict__ key_cache,
    uint8_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    int num_heads,
    int head_dim,
    int block_size)
{
    const int token_idx = blockIdx.x;
    const int64_t slot = slot_mapping[token_idx];

    if (slot < 0) return;

    const int n_elems = num_heads * head_dim;
    const float k_inv_scale = 1.0f / (*k_scale);
    const float v_inv_scale = 1.0f / (*v_scale);

    const __half* key_src = key + token_idx * n_elems;
    const __half* value_src = value + token_idx * n_elems;
    uint8_t* key_dst = key_cache + slot * n_elems;
    uint8_t* value_dst = value_cache + slot * n_elems;

    constexpr int VEC_ELEMS = 8;
    const int num_vecs = n_elems / VEC_ELEMS;
    const int tail_start = num_vecs * VEC_ELEMS;

    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        uint4 k_vec = *reinterpret_cast<const uint4*>(&key_src[vi * VEC_ELEMS]);
        uint4 v_vec = *reinterpret_cast<const uint4*>(&value_src[vi * VEC_ELEMS]);

        __half* k_elems = reinterpret_cast<__half*>(&k_vec);
        __half* v_elems = reinterpret_cast<__half*>(&v_vec);

        uint8_t k_fp8[8], v_fp8[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            k_fp8[i] = f16_to_fp8e4m3(k_elems[i], k_inv_scale);
            v_fp8[i] = f16_to_fp8e4m3(v_elems[i], v_inv_scale);
        }

        *reinterpret_cast<uint2*>(&key_dst[vi * VEC_ELEMS]) =
            *reinterpret_cast<uint2*>(k_fp8);
        *reinterpret_cast<uint2*>(&value_dst[vi * VEC_ELEMS]) =
            *reinterpret_cast<uint2*>(v_fp8);
    }

    for (int i = tail_start + threadIdx.x; i < n_elems; i += blockDim.x) {
        key_dst[i] = f16_to_fp8e4m3(key_src[i], k_inv_scale);
        value_dst[i] = f16_to_fp8e4m3(value_src[i], v_inv_scale);
    }
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

extern "C" {

void reshape_and_cache_f32(
    const float* key, const float* value,
    float* key_cache, float* value_cache,
    const int64_t* slot_mapping,
    int num_tokens, int num_heads, int head_dim, int block_size,
    cudaStream_t stream)
{
    int n_elems = num_heads * head_dim;
    int threads = (n_elems < 1024) ? n_elems : 1024;
    reshape_and_cache_kernel<float><<<num_tokens, threads, 0, stream>>>(
        key, value, key_cache, value_cache, slot_mapping,
        num_heads, head_dim, block_size);
}

void reshape_and_cache_f16(
    const __half* key, const __half* value,
    __half* key_cache, __half* value_cache,
    const int64_t* slot_mapping,
    int num_tokens, int num_heads, int head_dim, int block_size,
    cudaStream_t stream)
{
    int n_elems = num_heads * head_dim;
    int threads = (n_elems < 1024) ? n_elems : 1024;
    reshape_and_cache_kernel<__half><<<num_tokens, threads, 0, stream>>>(
        key, value, key_cache, value_cache, slot_mapping,
        num_heads, head_dim, block_size);
}

void reshape_and_cache_bf16(
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* key_cache, __nv_bfloat16* value_cache,
    const int64_t* slot_mapping,
    int num_tokens, int num_heads, int head_dim, int block_size,
    cudaStream_t stream)
{
    int n_elems = num_heads * head_dim;
    int threads = (n_elems < 1024) ? n_elems : 1024;
    reshape_and_cache_kernel<__nv_bfloat16><<<num_tokens, threads, 0, stream>>>(
        key, value, key_cache, value_cache, slot_mapping,
        num_heads, head_dim, block_size);
}

void reshape_and_cache_fp8_bf16(
    const uint16_t* key, const uint16_t* value,
    uint8_t* key_cache, uint8_t* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale, const float* v_scale,
    int num_tokens, int num_heads, int head_dim, int block_size,
    cudaStream_t stream)
{
    int n_elems = num_heads * head_dim;
    int threads = (n_elems < 1024) ? n_elems : 1024;
    reshape_and_cache_fp8_kernel<__nv_bfloat16><<<num_tokens, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(key),
        reinterpret_cast<const __nv_bfloat16*>(value),
        key_cache, value_cache, slot_mapping,
        k_scale, v_scale,
        num_heads, head_dim, block_size);
}

void reshape_and_cache_fp8_f16(
    const uint16_t* key, const uint16_t* value,
    uint8_t* key_cache, uint8_t* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale, const float* v_scale,
    int num_tokens, int num_heads, int head_dim, int block_size,
    cudaStream_t stream)
{
    int n_elems = num_heads * head_dim;
    int threads = (n_elems < 1024) ? n_elems : 1024;
    reshape_and_cache_fp8_f16_kernel<<<num_tokens, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(key),
        reinterpret_cast<const __half*>(value),
        key_cache, value_cache, slot_mapping,
        k_scale, v_scale,
        num_heads, head_dim, block_size);
}

} // extern "C"
