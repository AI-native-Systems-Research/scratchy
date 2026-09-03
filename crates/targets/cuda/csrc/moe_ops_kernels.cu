// SPDX-License-Identifier: Apache-2.0
// Element-wise CUDA kernels for MoE operations:
// 1. sigmoid_mul_add: out = a + sigmoid(gate) * b  (shared expert gating)
// 2. add_inplace: a += b  (simple accumulation)
//
// Uses vectorized 128-bit loads/stores via vec_utils.cuh.

#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "vec_utils.cuh"

// ---------------------------------------------------------------------------
// sigmoid_mul_add: out[i] = a[i] + sigmoid(gate_val) * b[i]
// gate is [num_tokens, 1], broadcast across hidden dimension.
// a, b, out are [num_tokens, hidden_size].
// ---------------------------------------------------------------------------

template <typename T>
__global__ void sigmoid_mul_add_kernel(
    T* __restrict__ out,
    const T* __restrict__ a,
    const T* __restrict__ b,
    const T* __restrict__ gate,  // [num_tokens, 1]
    int hidden_size)
{
    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const int row = blockIdx.x;
    const T* a_row = a + row * hidden_size;
    const T* b_row = b + row * hidden_size;
    T* o_row = out + row * hidden_size;

    // Load gate value for this row and compute sigmoid.
    float g = static_cast<float>(gate[row]);
    float sig = 1.0f / (1.0f + expf(-g));

    // Vectorized path.
    const int num_vecs = hidden_size / VEC_SIZE;
    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        float fa[VEC_SIZE], fb[VEC_SIZE], fo[VEC_SIZE];
        unpack_vec<T>(vec_load(&a_row[i * VEC_SIZE]), fa);
        unpack_vec<T>(vec_load(&b_row[i * VEC_SIZE]), fb);

        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            fo[j] = fa[j] + sig * fb[j];
        }
        vec_store(&o_row[i * VEC_SIZE], pack_vec<T>(fo));
    }

    // Scalar tail.
    const int tail_start = num_vecs * VEC_SIZE;
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float fa = static_cast<float>(a_row[i]);
        float fb = static_cast<float>(b_row[i]);
        o_row[i] = static_cast<T>(fa + sig * fb);
    }
}

// ---------------------------------------------------------------------------
// add_inplace: a[i] += b[i]
// a, b are [num_tokens, hidden_size]. Modifies a in-place.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void add_inplace_kernel(
    T* __restrict__ a,
    const T* __restrict__ b,
    int hidden_size)
{
    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const int row = blockIdx.x;
    T* a_row = a + row * hidden_size;
    const T* b_row = b + row * hidden_size;

    const int num_vecs = hidden_size / VEC_SIZE;
    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        float fa[VEC_SIZE], fb[VEC_SIZE];
        unpack_vec<T>(vec_load(&a_row[i * VEC_SIZE]), fa);
        unpack_vec<T>(vec_load(&b_row[i * VEC_SIZE]), fb);

        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            fa[j] += fb[j];
        }
        vec_store(&a_row[i * VEC_SIZE], pack_vec<T>(fa));
    }

    const int tail_start = num_vecs * VEC_SIZE;
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float fa = static_cast<float>(a_row[i]);
        float fb = static_cast<float>(b_row[i]);
        a_row[i] = static_cast<T>(fa + fb);
    }
}

// =====================================================================
// extern "C" launchers
// =====================================================================

static constexpr int BLOCK_SIZE = 256;

// sigmoid_mul_add: out = a + sigmoid(gate) * b
extern "C" void sigmoid_mul_add_bf16(
    void* out, const void* a, const void* b, const void* gate,
    int num_tokens, int hidden_size, cudaStream_t stream)
{
    sigmoid_mul_add_kernel<__nv_bfloat16><<<num_tokens, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(out),
        reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<const __nv_bfloat16*>(gate),
        hidden_size);
}

extern "C" void sigmoid_mul_add_f16(
    void* out, const void* a, const void* b, const void* gate,
    int num_tokens, int hidden_size, cudaStream_t stream)
{
    sigmoid_mul_add_kernel<__half><<<num_tokens, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<__half*>(out),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<const __half*>(gate),
        hidden_size);
}

extern "C" void sigmoid_mul_add_f32(
    void* out, const void* a, const void* b, const void* gate,
    int num_tokens, int hidden_size, cudaStream_t stream)
{
    sigmoid_mul_add_kernel<float><<<num_tokens, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<float*>(out),
        reinterpret_cast<const float*>(a),
        reinterpret_cast<const float*>(b),
        reinterpret_cast<const float*>(gate),
        hidden_size);
}

// add_inplace: a += b
extern "C" void add_inplace_bf16(
    void* a, const void* b,
    int num_tokens, int hidden_size, cudaStream_t stream)
{
    add_inplace_kernel<__nv_bfloat16><<<num_tokens, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b),
        hidden_size);
}

extern "C" void add_inplace_f16(
    void* a, const void* b,
    int num_tokens, int hidden_size, cudaStream_t stream)
{
    add_inplace_kernel<__half><<<num_tokens, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<__half*>(a),
        reinterpret_cast<const __half*>(b),
        hidden_size);
}

extern "C" void add_inplace_f32(
    void* a, const void* b,
    int num_tokens, int hidden_size, cudaStream_t stream)
{
    add_inplace_kernel<float><<<num_tokens, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<float*>(a),
        reinterpret_cast<const float*>(b),
        hidden_size);
}

// ---------------------------------------------------------------------------
// Gemma-4 routing kernel A: softmax over the gathered top-k logits, with a
// temperature multiply folded in BEFORE exp.
//
//   scores[m, j] = softmax_j( logits[m, ids[m, j]] * temp )      (f32 out)
//
// Gemma selects the top-k on RAW logits, then softmaxes over ONLY those k
// (NOT over all experts) with temp = hidden^-0.5. One block per token, top_k
// (≤ 32) lanes; the F32 scores feed fused_moe_gemm / moe_sum.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void moe_softmax_topk_temp_kernel(
    float* __restrict__ scores,        // [M, top_k] f32, out
    const T* __restrict__ logits,      // [M, num_experts]
    const int* __restrict__ ids,       // [M, top_k] i32
    int num_experts, int top_k, float temp)
{
    // One warp (32 lanes) per token. ALL lanes stay live through the warp
    // shuffles — lanes >= top_k contribute neutral values (−inf for max, 0 for
    // sum) so no __syncthreads / divergent early-return is needed (top_k ≤ 32).
    const int m = blockIdx.x;
    const int j = threadIdx.x;
    const bool active = (j < top_k);

    // Gather this lane's selected logit, scaled by temperature.
    float x = -INFINITY;
    if (active) {
        const int eid = ids[m * top_k + j];
        x = float(logits[m * num_experts + eid]) * temp;
    }

    // Warp-max over the top_k lanes (inactive lanes are −inf → ignored).
    float vmax = x;
    for (int o = 16; o > 0; o >>= 1) {
        vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, o));
    }

    // exp(x - max); inactive lanes produce 0 (exp(-inf)).
    float e = active ? __expf(x - vmax) : 0.0f;
    float vsum = e;
    for (int o = 16; o > 0; o >>= 1) {
        vsum += __shfl_xor_sync(0xffffffff, vsum, o);
    }

    if (active) {
        scores[m * top_k + j] = e / vsum;
    }
}

extern "C" void moe_softmax_topk_temp_bf16(
    float* scores, const void* logits, const int* ids,
    int num_tokens, int num_experts, int top_k, float temp, cudaStream_t stream)
{
    moe_softmax_topk_temp_kernel<__nv_bfloat16><<<num_tokens, 32, 0, stream>>>(
        scores, reinterpret_cast<const __nv_bfloat16*>(logits), ids,
        num_experts, top_k, temp);
}

extern "C" void moe_softmax_topk_temp_f16(
    float* scores, const void* logits, const int* ids,
    int num_tokens, int num_experts, int top_k, float temp, cudaStream_t stream)
{
    moe_softmax_topk_temp_kernel<__half><<<num_tokens, 32, 0, stream>>>(
        scores, reinterpret_cast<const __half*>(logits), ids,
        num_experts, top_k, temp);
}

extern "C" void moe_softmax_topk_temp_f32(
    float* scores, const void* logits, const int* ids,
    int num_tokens, int num_experts, int top_k, float temp, cudaStream_t stream)
{
    moe_softmax_topk_temp_kernel<float><<<num_tokens, 32, 0, stream>>>(
        scores, reinterpret_cast<const float*>(logits), ids,
        num_experts, top_k, temp);
}

// ---------------------------------------------------------------------------
// Gemma-4 routing kernel B: per-expert score scale gather-multiply (in place).
//
//   scores[g] *= per_expert_scale[ids[g]]        g in [0, M*top_k)
//
// Port of moe_per_expert_scale.metal. `scores` are the F32 routing weights;
// `per_expert_scale` is the router bundle's bf16 [num_experts] vector. Float
// intermediate so the multiply matches the reference's single rounding.
// ---------------------------------------------------------------------------

__global__ void moe_per_expert_scale_kernel(
    float* __restrict__ scores,                     // [M*top_k] f32, in place
    const int* __restrict__ ids,                    // [M*top_k] i32
    const __nv_bfloat16* __restrict__ per_expert_scale,  // [num_experts] bf16
    int n)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    const int eid = ids[g];
    const float s = __bfloat162float(per_expert_scale[eid]);
    scores[g] = scores[g] * s;
}

extern "C" void moe_per_expert_scale_f32(
    float* scores, const int* ids, const void* per_expert_scale,
    int n, cudaStream_t stream)
{
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    moe_per_expert_scale_kernel<<<blocks, threads, 0, stream>>>(
        scores, ids,
        reinterpret_cast<const __nv_bfloat16*>(per_expert_scale), n);
}
