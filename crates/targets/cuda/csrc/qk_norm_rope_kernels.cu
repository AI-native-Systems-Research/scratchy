// SPDX-License-Identifier: Apache-2.0
// Fused per-head QK RMS normalization + NeoX RoPE CUDA kernel.
//
// Used by Gemma3 which applies per-head RMS norm to Q/K vectors before
// RoPE rotation. Fusing eliminates the intermediate global memory round-trip.
//
// Grid:  (num_tokens, num_q_heads + num_kv_heads)
// Block: min(head_dim, 1024) threads
//
// Algorithm per block:
//   1. Identify Q or K head from blockIdx.y
//   2. Pass 1: sum-of-squares reduction → inv_rms
//   3. Pass 2: normalize with weight, store to shared memory
//   4. Pass 3: NeoX RoPE from shared memory → write back to global memory

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Warp-level reduce sum via shuffle.
__device__ __forceinline__ float warp_reduce_sum_qknr(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduce sum via shared memory.
// Uses static shared memory (coexists with extern __shared__).
__device__ float block_reduce_sum_qknr(float val) {
    __shared__ float shared_reduce[32]; // one per warp
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum_qknr(val);
    if (lane == 0) shared_reduce[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? shared_reduce[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum_qknr(val);
    return val;
}

// ---------------------------------------------------------------------------
// Fused QK-norm + RoPE kernel
// ---------------------------------------------------------------------------

template <typename T>
__global__ void qk_norm_rope_kernel(
    T* __restrict__ query,              // [num_tokens, num_q_heads, head_dim] in/out
    T* __restrict__ key,                // [num_tokens, num_kv_heads, head_dim] in/out
    const T* __restrict__ q_weight,     // [head_dim]
    const T* __restrict__ k_weight,     // [head_dim]
    const T* __restrict__ cos_cache,    // [max_pos, head_dim]
    const T* __restrict__ sin_cache,    // [max_pos, head_dim]
    const uint32_t* __restrict__ positions,  // [num_tokens]
    float epsilon,
    float q_weight_offset,
    float k_weight_offset,
    int num_q_heads,
    int num_kv_heads,
    int head_dim)
{
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int half_dim = head_dim / 2;
    const int pos = static_cast<int>(positions[token_idx]);

    // Determine if this block handles a Q head or K head.
    const bool is_q = (head_idx < num_q_heads);
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);

    // Pointer to the head vector and weight.
    T* head_ptr;
    const T* weight;
    float w_offset;
    if (is_q) {
        head_ptr = query + (token_idx * num_q_heads + local_head) * head_dim;
        weight = q_weight;
        w_offset = q_weight_offset;
    } else {
        head_ptr = key + (token_idx * num_kv_heads + local_head) * head_dim;
        weight = k_weight;
        w_offset = k_weight_offset;
    }

    // Dynamic shared memory for normalized values.
    extern __shared__ float smem[];

    // Pass 1: compute sum of squares for RMS norm.
    float ss = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = static_cast<float>(head_ptr[i]);
        ss += v * v;
    }
    ss = block_reduce_sum_qknr(ss);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(ss / head_dim + epsilon);
    }
    __syncthreads();

    // Pass 2: normalize with weight (+ optional offset for Gemma's
    // (1+w) convention), store to shared memory.
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = static_cast<float>(head_ptr[i]) * s_inv_rms;
        smem[i] = v * (static_cast<float>(weight[i]) + w_offset);
    }
    __syncthreads();

    // Pass 3: apply NeoX RoPE from shared memory and write back.
    // cos/sin cache: [max_pos, head_dim] — duplicated format (cos[i] == cos[i+half]).
    const T* cos_ptr = cos_cache + pos * head_dim;
    const T* sin_ptr = sin_cache + pos * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        float a = smem[i];
        float b = smem[i + half_dim];
        float c = static_cast<float>(cos_ptr[i]);
        float s = static_cast<float>(sin_ptr[i]);

        head_ptr[i]            = static_cast<T>(a * c - b * s);
        head_ptr[i + half_dim] = static_cast<T>(b * c + a * s);
    }
}

// ---------------------------------------------------------------------------
// QK-norm ONLY kernel (no RoPE) — for models with partial RoPE
// ---------------------------------------------------------------------------

template <typename T>
__global__ void qk_norm_kernel(
    T* __restrict__ query,              // [num_tokens, num_q_heads, head_dim] in/out
    T* __restrict__ key,                // [num_tokens, num_kv_heads, head_dim] in/out
    const T* __restrict__ q_weight,     // [head_dim]
    const T* __restrict__ k_weight,     // [head_dim]
    float epsilon,
    float q_weight_offset,
    float k_weight_offset,
    int num_q_heads,
    int num_kv_heads,
    int head_dim)
{
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    const bool is_q = (head_idx < num_q_heads);
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);

    T* head_ptr;
    const T* weight;
    float w_offset;
    if (is_q) {
        head_ptr = query + (token_idx * num_q_heads + local_head) * head_dim;
        weight = q_weight;
        w_offset = q_weight_offset;
    } else {
        head_ptr = key + (token_idx * num_kv_heads + local_head) * head_dim;
        weight = k_weight;
        w_offset = k_weight_offset;
    }

    // Compute sum of squares for RMS norm.
    float ss = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = static_cast<float>(head_ptr[i]);
        ss += v * v;
    }
    ss = block_reduce_sum_qknr(ss);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(ss / head_dim + epsilon);
    }
    __syncthreads();

    // Normalize and write back (with weight + optional offset).
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = static_cast<float>(head_ptr[i]) * s_inv_rms;
        head_ptr[i] = static_cast<T>(v * (static_cast<float>(weight[i]) + w_offset));
    }
}

// ---------------------------------------------------------------------------
// Sigmoid-mul kernel: out = sigmoid(gate) * input (element-wise, in-place on input)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void sigmoid_mul_kernel(
    T* __restrict__ input,        // [numel] in/out — overwritten with result
    const T* __restrict__ gate,   // [numel]
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float x = static_cast<float>(input[idx]);
    float g = static_cast<float>(gate[idx]);
    float sig = 1.0f / (1.0f + expf(-g));
    input[idx] = static_cast<T>(sig * x);
}

// ---------------------------------------------------------------------------
// Sigmoid-rowgate-combine kernel: out = routed + sh * sigmoid(gate[row])
// (Qwen3.5-MoE shared-expert combine; gate is [rows, 1], row-broadcast
// across cols). Mirrors gate_scale.metal's 4-buffer form so the caller
// binds the arena's routed/out slots directly — no staging copy. Aliasing
// out == routed is safe (each thread reads then writes only element idx),
// which preserves the old in-place accumulate as a special case.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void sigmoid_rowgate_combine_kernel(
    T* __restrict__ out,          // [rows*cols] combined output
    const T* __restrict__ routed, // [rows*cols] routed-expert sum
    const T* __restrict__ sh,     // [rows*cols] shared-expert output
    const T* __restrict__ gate,   // [rows] shared_expert_gate output ([T, 1])
    int numel,
    int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    int row = idx / cols;
    float r = static_cast<float>(routed[idx]);
    float s = static_cast<float>(sh[idx]);
    float g = static_cast<float>(gate[row]);
    float sig = 1.0f / (1.0f + expf(-g));
    out[idx] = static_cast<T>(r + s * sig);
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

extern "C" {

void qk_norm_rope_f32(
    float* query, float* key,
    const float* q_weight, const float* k_weight,
    const float* cos_cache, const float* sin_cache,
    const uint32_t* positions,
    float epsilon,
    float q_weight_offset, float k_weight_offset,
    int num_q_heads, int num_kv_heads,
    int head_dim, int num_tokens,
    cudaStream_t stream)
{
    dim3 grid(num_tokens, num_q_heads + num_kv_heads);
    int threads = (head_dim < 1024) ? head_dim : 1024;
    int smem_bytes = head_dim * sizeof(float);
    qk_norm_rope_kernel<float><<<grid, threads, smem_bytes, stream>>>(
        query, key, q_weight, k_weight,
        cos_cache, sin_cache, positions,
        epsilon, q_weight_offset, k_weight_offset,
        num_q_heads, num_kv_heads, head_dim);
}

void qk_norm_rope_f16(
    __half* query, __half* key,
    const __half* q_weight, const __half* k_weight,
    const __half* cos_cache, const __half* sin_cache,
    const uint32_t* positions,
    float epsilon,
    float q_weight_offset, float k_weight_offset,
    int num_q_heads, int num_kv_heads,
    int head_dim, int num_tokens,
    cudaStream_t stream)
{
    dim3 grid(num_tokens, num_q_heads + num_kv_heads);
    int threads = (head_dim < 1024) ? head_dim : 1024;
    int smem_bytes = head_dim * sizeof(float);
    qk_norm_rope_kernel<__half><<<grid, threads, smem_bytes, stream>>>(
        query, key, q_weight, k_weight,
        cos_cache, sin_cache, positions,
        epsilon, q_weight_offset, k_weight_offset,
        num_q_heads, num_kv_heads, head_dim);
}

void qk_norm_rope_bf16(
    __nv_bfloat16* query, __nv_bfloat16* key,
    const __nv_bfloat16* q_weight, const __nv_bfloat16* k_weight,
    const __nv_bfloat16* cos_cache, const __nv_bfloat16* sin_cache,
    const uint32_t* positions,
    float epsilon,
    float q_weight_offset, float k_weight_offset,
    int num_q_heads, int num_kv_heads,
    int head_dim, int num_tokens,
    cudaStream_t stream)
{
    dim3 grid(num_tokens, num_q_heads + num_kv_heads);
    int threads = (head_dim < 1024) ? head_dim : 1024;
    int smem_bytes = head_dim * sizeof(float);
    qk_norm_rope_kernel<__nv_bfloat16><<<grid, threads, smem_bytes, stream>>>(
        query, key, q_weight, k_weight,
        cos_cache, sin_cache, positions,
        epsilon, q_weight_offset, k_weight_offset,
        num_q_heads, num_kv_heads, head_dim);
}

// QK-norm only (no RoPE)

void qk_norm_f32(
    float* query, float* key,
    const float* q_weight, const float* k_weight,
    float epsilon,
    float q_weight_offset, float k_weight_offset,
    int num_q_heads, int num_kv_heads,
    int head_dim, int num_tokens,
    cudaStream_t stream)
{
    dim3 grid(num_tokens, num_q_heads + num_kv_heads);
    int threads = (head_dim < 1024) ? head_dim : 1024;
    qk_norm_kernel<float><<<grid, threads, 0, stream>>>(
        query, key, q_weight, k_weight,
        epsilon, q_weight_offset, k_weight_offset,
        num_q_heads, num_kv_heads, head_dim);
}

void qk_norm_f16(
    __half* query, __half* key,
    const __half* q_weight, const __half* k_weight,
    float epsilon,
    float q_weight_offset, float k_weight_offset,
    int num_q_heads, int num_kv_heads,
    int head_dim, int num_tokens,
    cudaStream_t stream)
{
    dim3 grid(num_tokens, num_q_heads + num_kv_heads);
    int threads = (head_dim < 1024) ? head_dim : 1024;
    qk_norm_kernel<__half><<<grid, threads, 0, stream>>>(
        query, key, q_weight, k_weight,
        epsilon, q_weight_offset, k_weight_offset,
        num_q_heads, num_kv_heads, head_dim);
}

void qk_norm_bf16(
    __nv_bfloat16* query, __nv_bfloat16* key,
    const __nv_bfloat16* q_weight, const __nv_bfloat16* k_weight,
    float epsilon,
    float q_weight_offset, float k_weight_offset,
    int num_q_heads, int num_kv_heads,
    int head_dim, int num_tokens,
    cudaStream_t stream)
{
    dim3 grid(num_tokens, num_q_heads + num_kv_heads);
    int threads = (head_dim < 1024) ? head_dim : 1024;
    qk_norm_kernel<__nv_bfloat16><<<grid, threads, 0, stream>>>(
        query, key, q_weight, k_weight,
        epsilon, q_weight_offset, k_weight_offset,
        num_q_heads, num_kv_heads, head_dim);
}

// Sigmoid-mul

void sigmoid_mul_f32(
    float* input, const float* gate, int numel, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    sigmoid_mul_kernel<float><<<blocks, threads, 0, stream>>>(input, gate, numel);
}

void sigmoid_mul_f16(
    __half* input, const __half* gate, int numel, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    sigmoid_mul_kernel<__half><<<blocks, threads, 0, stream>>>(input, gate, numel);
}

void sigmoid_mul_bf16(
    __nv_bfloat16* input, const __nv_bfloat16* gate, int numel, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    sigmoid_mul_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(input, gate, numel);
}

// Sigmoid-rowgate-combine (Qwen3.5-MoE shared-expert combine)

void sigmoid_rowgate_combine_f32(
    float* out, const float* routed, const float* sh, const float* gate, int numel, int cols,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    sigmoid_rowgate_combine_kernel<float>
        <<<blocks, threads, 0, stream>>>(out, routed, sh, gate, numel, cols);
}

void sigmoid_rowgate_combine_f16(
    __half* out, const __half* routed, const __half* sh, const __half* gate, int numel, int cols,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    sigmoid_rowgate_combine_kernel<__half>
        <<<blocks, threads, 0, stream>>>(out, routed, sh, gate, numel, cols);
}

void sigmoid_rowgate_combine_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* routed, const __nv_bfloat16* sh,
    const __nv_bfloat16* gate, int numel, int cols, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    sigmoid_rowgate_combine_kernel<__nv_bfloat16>
        <<<blocks, threads, 0, stream>>>(out, routed, sh, gate, numel, cols);
}

} // extern "C"
