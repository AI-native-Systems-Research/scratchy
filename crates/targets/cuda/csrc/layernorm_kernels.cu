// SPDX-License-Identifier: Apache-2.0
// Fused RMS normalization CUDA kernel for vLLM Rust.
//
// Port of csrc/layernorm_kernels.cu from Python vLLM.
// One thread block per row (token). Each block cooperatively computes
// the variance, then applies weight * x * rsqrt(variance + eps).
//
// Uses vectorized 128-bit loads/stores for throughput.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "vec_utils.cuh"

// Warp-level reduce sum via shuffle.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduce sum via shared memory.
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32]; // one per warp
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

// ---- RMS Norm (out-of-place, vectorized) ----
// out[row, :] = weight * input[row, :] * rsqrt(mean(input[row, :]^2) + eps)

template <typename T>
__global__ void rms_norm_kernel(
    T* __restrict__ out,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    float epsilon,
    float weight_offset,  // e.g. 1.0 for Gemma2's (1+w); 0.0 for Llama.
    int hidden_size)
{
    constexpr int VEC_SIZE = VecType<T>::SIZE;
    using VT = typename VecType<T>::Type;

    const int row = blockIdx.x;
    const T* x = input + row * hidden_size;
    T* y = out + row * hidden_size;

    const int num_vecs = hidden_size / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    // Pass 1: Compute sum of squares with vectorized loads.
    float ss = 0.0f;
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float buf[VEC_SIZE];
        unpack_vec<T>(vec_load(&x[vi * VEC_SIZE]), buf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            ss += buf[j] * buf[j];
        }
    }
    // Scalar tail.
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = static_cast<float>(x[i]);
        ss += v * v;
    }
    ss = block_reduce_sum(ss);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(ss / hidden_size + epsilon);
    }
    __syncthreads();

    // Pass 2: Apply normalization with vectorized loads/stores.
    // All math in fp32; cast to T happens at pack/store (Gemma
    // cast-last semantics — matches vllm Python's GemmaRMSNorm).
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float xbuf[VEC_SIZE], wbuf[VEC_SIZE], obuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&x[vi * VEC_SIZE]), xbuf);
        unpack_vec<T>(vec_load(&weight[vi * VEC_SIZE]), wbuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            obuf[j] = xbuf[j] * s_inv_rms * (wbuf[j] + weight_offset);
        }
        vec_store(&y[vi * VEC_SIZE], pack_vec<T>(obuf));
    }
    // Scalar tail.
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = static_cast<float>(x[i]) * s_inv_rms;
        y[i] = static_cast<T>(v * (static_cast<float>(weight[i]) + weight_offset));
    }
}

// ---- Fused Add + RMS Norm (in-place on input, residual updated, vectorized) ----
// residual += input; input = weight * residual * rsqrt(mean(residual^2) + eps)

template <typename T>
__global__ void fused_add_rms_norm_kernel(
    T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    float epsilon,
    float weight_offset,
    int hidden_size)
{
    constexpr int VEC_SIZE = VecType<T>::SIZE;
    using VT = typename VecType<T>::Type;

    const int row = blockIdx.x;
    T* inp = input + row * hidden_size;
    T* res = residual + row * hidden_size;

    const int num_vecs = hidden_size / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    // Pass 1: Fused add + variance computation with vectorized loads/stores.
    // Match Python GemmaRMSNorm._forward_static_with_residual: `x = x + residual`
    // on bf16 tensors rounds each sum to bf16 BEFORE the `.float()` upcast, so
    // variance is computed from bf16-precision values. Round-trip through T
    // to quantize the sum before squaring.
    float ss = 0.0f;
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float ibuf[VEC_SIZE], rbuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&inp[vi * VEC_SIZE]), ibuf);
        unpack_vec<T>(vec_load(&res[vi * VEC_SIZE]), rbuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            rbuf[j] = static_cast<float>(static_cast<T>(rbuf[j] + ibuf[j]));
            ss += rbuf[j] * rbuf[j];
        }
        vec_store(&res[vi * VEC_SIZE], pack_vec<T>(rbuf));
    }
    // Scalar tail.
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        T r_t = static_cast<T>(static_cast<float>(res[i]) + static_cast<float>(inp[i]));
        res[i] = r_t;
        float r = static_cast<float>(r_t);
        ss += r * r;
    }
    ss = block_reduce_sum(ss);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(ss / hidden_size + epsilon);
    }
    __syncthreads();

    // Pass 2: Normalize with vectorized loads/stores. Gemma's
    // `(1+w)` convention rides `weight_offset`; Llama passes 0.0.
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float rbuf[VEC_SIZE], wbuf[VEC_SIZE], obuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&res[vi * VEC_SIZE]), rbuf);
        unpack_vec<T>(vec_load(&weight[vi * VEC_SIZE]), wbuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            obuf[j] = rbuf[j] * s_inv_rms * (wbuf[j] + weight_offset);
        }
        vec_store(&inp[vi * VEC_SIZE], pack_vec<T>(obuf));
    }
    // Scalar tail.
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = static_cast<float>(res[i]) * s_inv_rms;
        inp[i] = static_cast<T>(v * (static_cast<float>(weight[i]) + weight_offset));
    }
}

// ---- C entry points ----

extern "C" {

// ---- RMS Norm entry points ----

void rms_norm_f32(
    float* out, const float* input, const float* weight,
    float epsilon, float weight_offset, int num_tokens, int hidden_size,
    cudaStream_t stream)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    rms_norm_kernel<float><<<num_tokens, threads, 0, stream>>>(
        out, input, weight, epsilon, weight_offset, hidden_size);
}

void rms_norm_f16(
    __half* out, const __half* input, const __half* weight,
    float epsilon, float weight_offset, int num_tokens, int hidden_size,
    cudaStream_t stream)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    rms_norm_kernel<__half><<<num_tokens, threads, 0, stream>>>(
        out, input, weight, epsilon, weight_offset, hidden_size);
}

void rms_norm_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* input, const __nv_bfloat16* weight,
    float epsilon, float weight_offset, int num_tokens, int hidden_size,
    cudaStream_t stream)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    rms_norm_kernel<__nv_bfloat16><<<num_tokens, threads, 0, stream>>>(
        out, input, weight, epsilon, weight_offset, hidden_size);
}

// ---- Fused Add + RMS Norm entry points ----

void fused_add_rms_norm_f32(
    float* input, float* residual, const float* weight,
    float epsilon, float weight_offset, int num_tokens, int hidden_size,
    cudaStream_t stream)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    fused_add_rms_norm_kernel<float><<<num_tokens, threads, 0, stream>>>(
        input, residual, weight, epsilon, weight_offset, hidden_size);
}

void fused_add_rms_norm_f16(
    __half* input, __half* residual, const __half* weight,
    float epsilon, float weight_offset, int num_tokens, int hidden_size,
    cudaStream_t stream)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    fused_add_rms_norm_kernel<__half><<<num_tokens, threads, 0, stream>>>(
        input, residual, weight, epsilon, weight_offset, hidden_size);
}

void fused_add_rms_norm_bf16(
    __nv_bfloat16* input, __nv_bfloat16* residual, const __nv_bfloat16* weight,
    float epsilon, float weight_offset, int num_tokens, int hidden_size,
    cudaStream_t stream)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    fused_add_rms_norm_kernel<__nv_bfloat16><<<num_tokens, threads, 0, stream>>>(
        input, residual, weight, epsilon, weight_offset, hidden_size);
}

// ---- Cohere LayerNorm (out-of-place, vectorized) ----
// out[row, :] = weight * (input[row, :] - mean(input[row, :])) / sqrt(var(input[row, :]) + eps)
// Full LayerNorm with mean subtraction, weight only (no bias).

} // extern "C"

template <typename T>
__global__ void cohere_layer_norm_kernel(
    T* __restrict__ out,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    float epsilon,
    int hidden_size)
{
    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const int row = blockIdx.x;
    const T* x = input + row * hidden_size;
    T* y = out + row * hidden_size;

    const int num_vecs = hidden_size / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    // Pass 1: Compute sum and sum of squares.
    float sum_val = 0.0f;
    float ss = 0.0f;
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float buf[VEC_SIZE];
        unpack_vec<T>(vec_load(&x[vi * VEC_SIZE]), buf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            sum_val += buf[j];
            ss += buf[j] * buf[j];
        }
    }
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = static_cast<float>(x[i]);
        sum_val += v;
        ss += v * v;
    }
    sum_val = block_reduce_sum(sum_val);
    ss = block_reduce_sum(ss);

    __shared__ float s_mean, s_inv_std;
    if (threadIdx.x == 0) {
        float mean = sum_val / hidden_size;
        float var = ss / hidden_size - mean * mean;
        s_mean = mean;
        s_inv_std = rsqrtf(var + epsilon);
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;

    // Pass 2: Normalize.
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float xbuf[VEC_SIZE], wbuf[VEC_SIZE], obuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&x[vi * VEC_SIZE]), xbuf);
        unpack_vec<T>(vec_load(&weight[vi * VEC_SIZE]), wbuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            obuf[j] = (xbuf[j] - mean) * inv_std * wbuf[j];
        }
        vec_store(&y[vi * VEC_SIZE], pack_vec<T>(obuf));
    }
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = (static_cast<float>(x[i]) - mean) * inv_std;
        y[i] = static_cast<T>(v * static_cast<float>(weight[i]));
    }
}

// ---- Fused Add + Cohere LayerNorm (in-place) ----
// residual += input; input = weight * (residual - mean(residual)) / sqrt(var(residual) + eps)

template <typename T>
__global__ void fused_add_cohere_layer_norm_kernel(
    T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    float epsilon,
    int hidden_size)
{
    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const int row = blockIdx.x;
    T* inp = input + row * hidden_size;
    T* res = residual + row * hidden_size;

    const int num_vecs = hidden_size / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    // Pass 1: Fused add + stats.
    float sum_val = 0.0f;
    float ss = 0.0f;
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float ibuf[VEC_SIZE], rbuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&inp[vi * VEC_SIZE]), ibuf);
        unpack_vec<T>(vec_load(&res[vi * VEC_SIZE]), rbuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            rbuf[j] += ibuf[j];
            sum_val += rbuf[j];
            ss += rbuf[j] * rbuf[j];
        }
        vec_store(&res[vi * VEC_SIZE], pack_vec<T>(rbuf));
    }
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float r = static_cast<float>(res[i]) + static_cast<float>(inp[i]);
        res[i] = static_cast<T>(r);
        sum_val += r;
        ss += r * r;
    }
    sum_val = block_reduce_sum(sum_val);
    ss = block_reduce_sum(ss);

    __shared__ float s_mean, s_inv_std;
    if (threadIdx.x == 0) {
        float mean = sum_val / hidden_size;
        float var = ss / hidden_size - mean * mean;
        s_mean = mean;
        s_inv_std = rsqrtf(var + epsilon);
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;

    // Pass 2: Normalize.
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float rbuf[VEC_SIZE], wbuf[VEC_SIZE], obuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&res[vi * VEC_SIZE]), rbuf);
        unpack_vec<T>(vec_load(&weight[vi * VEC_SIZE]), wbuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            obuf[j] = (rbuf[j] - mean) * inv_std * wbuf[j];
        }
        vec_store(&inp[vi * VEC_SIZE], pack_vec<T>(obuf));
    }
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = (static_cast<float>(res[i]) - mean) * inv_std;
        inp[i] = static_cast<T>(v * static_cast<float>(weight[i]));
    }
}

extern "C" {

// ---- Cohere LayerNorm entry points ----

void cohere_layer_norm_f32(
    float* out, const float* input, const float* weight,
    float epsilon, int num_tokens, int hidden_size,
    cudaStream_t stream)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    cohere_layer_norm_kernel<float><<<num_tokens, threads, 0, stream>>>(
        out, input, weight, epsilon, hidden_size);
}

void cohere_layer_norm_f16(
    __half* out, const __half* input, const __half* weight,
    float epsilon, int num_tokens, int hidden_size,
    cudaStream_t stream)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    cohere_layer_norm_kernel<__half><<<num_tokens, threads, 0, stream>>>(
        out, input, weight, epsilon, hidden_size);
}

void cohere_layer_norm_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* input, const __nv_bfloat16* weight,
    float epsilon, int num_tokens, int hidden_size,
    cudaStream_t stream)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    cohere_layer_norm_kernel<__nv_bfloat16><<<num_tokens, threads, 0, stream>>>(
        out, input, weight, epsilon, hidden_size);
}

// ---- Fused Add + Cohere LayerNorm entry points ----

void fused_add_cohere_layer_norm_f32(
    float* input, float* residual, const float* weight,
    float epsilon, int num_tokens, int hidden_size,
    cudaStream_t stream)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    fused_add_cohere_layer_norm_kernel<float><<<num_tokens, threads, 0, stream>>>(
        input, residual, weight, epsilon, hidden_size);
}

void fused_add_cohere_layer_norm_f16(
    __half* input, __half* residual, const __half* weight,
    float epsilon, int num_tokens, int hidden_size,
    cudaStream_t stream)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    fused_add_cohere_layer_norm_kernel<__half><<<num_tokens, threads, 0, stream>>>(
        input, residual, weight, epsilon, hidden_size);
}

void fused_add_cohere_layer_norm_bf16(
    __nv_bfloat16* input, __nv_bfloat16* residual, const __nv_bfloat16* weight,
    float epsilon, int num_tokens, int hidden_size,
    cudaStream_t stream)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    fused_add_cohere_layer_norm_kernel<__nv_bfloat16><<<num_tokens, threads, 0, stream>>>(
        input, residual, weight, epsilon, hidden_size);
}

} // extern "C"

// ---- LayerNorm with bias (out-of-place, vectorized) ----
// out[row, :] = weight * (input[row, :] - mean) / sqrt(var + eps) + bias
// Standard nn.LayerNorm with both weight and bias parameters.

template <typename T>
__global__ void layer_norm_bias_kernel(
    T* __restrict__ out,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    float epsilon,
    int hidden_size)
{
    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const int row = blockIdx.x;
    const T* x = input + row * hidden_size;
    T* y = out + row * hidden_size;

    const int num_vecs = hidden_size / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    // Pass 1: Compute sum and sum of squares.
    float sum_val = 0.0f;
    float ss = 0.0f;
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float buf[VEC_SIZE];
        unpack_vec<T>(vec_load(&x[vi * VEC_SIZE]), buf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            sum_val += buf[j];
            ss += buf[j] * buf[j];
        }
    }
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = static_cast<float>(x[i]);
        sum_val += v;
        ss += v * v;
    }
    sum_val = block_reduce_sum(sum_val);
    ss = block_reduce_sum(ss);

    __shared__ float s_mean, s_inv_std;
    if (threadIdx.x == 0) {
        float mean = sum_val / hidden_size;
        float var = ss / hidden_size - mean * mean;
        s_mean = mean;
        s_inv_std = rsqrtf(var + epsilon);
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;

    // Pass 2: Normalize with weight and bias.
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float xbuf[VEC_SIZE], wbuf[VEC_SIZE], bbuf[VEC_SIZE], obuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&x[vi * VEC_SIZE]), xbuf);
        unpack_vec<T>(vec_load(&weight[vi * VEC_SIZE]), wbuf);
        unpack_vec<T>(vec_load(&bias[vi * VEC_SIZE]), bbuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            obuf[j] = (xbuf[j] - mean) * inv_std * wbuf[j] + bbuf[j];
        }
        vec_store(&y[vi * VEC_SIZE], pack_vec<T>(obuf));
    }
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = (static_cast<float>(x[i]) - mean) * inv_std;
        y[i] = static_cast<T>(v * static_cast<float>(weight[i]) + static_cast<float>(bias[i]));
    }
}

// ---- Fused Add + LayerNorm with bias (in-place) ----
// residual += input; input = weight * (residual - mean) / sqrt(var + eps) + bias

template <typename T>
__global__ void fused_add_layer_norm_bias_kernel(
    T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    float epsilon,
    int hidden_size)
{
    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const int row = blockIdx.x;
    T* inp = input + row * hidden_size;
    T* res = residual + row * hidden_size;

    const int num_vecs = hidden_size / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    // Pass 1: Fused add + stats.
    float sum_val = 0.0f;
    float ss = 0.0f;
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float ibuf[VEC_SIZE], rbuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&inp[vi * VEC_SIZE]), ibuf);
        unpack_vec<T>(vec_load(&res[vi * VEC_SIZE]), rbuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            rbuf[j] += ibuf[j];
            sum_val += rbuf[j];
            ss += rbuf[j] * rbuf[j];
        }
        vec_store(&res[vi * VEC_SIZE], pack_vec<T>(rbuf));
    }
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float r = static_cast<float>(res[i]) + static_cast<float>(inp[i]);
        res[i] = static_cast<T>(r);
        sum_val += r;
        ss += r * r;
    }
    sum_val = block_reduce_sum(sum_val);
    ss = block_reduce_sum(ss);

    __shared__ float s_mean, s_inv_std;
    if (threadIdx.x == 0) {
        float mean = sum_val / hidden_size;
        float var = ss / hidden_size - mean * mean;
        s_mean = mean;
        s_inv_std = rsqrtf(var + epsilon);
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;

    // Pass 2: Normalize with weight and bias.
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float rbuf[VEC_SIZE], wbuf[VEC_SIZE], bbuf[VEC_SIZE], obuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&res[vi * VEC_SIZE]), rbuf);
        unpack_vec<T>(vec_load(&weight[vi * VEC_SIZE]), wbuf);
        unpack_vec<T>(vec_load(&bias[vi * VEC_SIZE]), bbuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            obuf[j] = (rbuf[j] - mean) * inv_std * wbuf[j] + bbuf[j];
        }
        vec_store(&inp[vi * VEC_SIZE], pack_vec<T>(obuf));
    }
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = (static_cast<float>(res[i]) - mean) * inv_std;
        inp[i] = static_cast<T>(v * static_cast<float>(weight[i]) + static_cast<float>(bias[i]));
    }
}

extern "C" {

// ---- LayerNorm with bias entry points ----

void layer_norm_bias_f32(
    float* out, const float* input, const float* weight, const float* bias,
    float epsilon, int num_tokens, int hidden_size,
    cudaStream_t stream)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    layer_norm_bias_kernel<float><<<num_tokens, threads, 0, stream>>>(
        out, input, weight, bias, epsilon, hidden_size);
}

void layer_norm_bias_f16(
    __half* out, const __half* input, const __half* weight, const __half* bias,
    float epsilon, int num_tokens, int hidden_size,
    cudaStream_t stream)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    layer_norm_bias_kernel<__half><<<num_tokens, threads, 0, stream>>>(
        out, input, weight, bias, epsilon, hidden_size);
}

void layer_norm_bias_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* input,
    const __nv_bfloat16* weight, const __nv_bfloat16* bias,
    float epsilon, int num_tokens, int hidden_size,
    cudaStream_t stream)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    layer_norm_bias_kernel<__nv_bfloat16><<<num_tokens, threads, 0, stream>>>(
        out, input, weight, bias, epsilon, hidden_size);
}

// ---- Fused Add + LayerNorm with bias entry points ----

void fused_add_layer_norm_bias_f32(
    float* input, float* residual, const float* weight, const float* bias,
    float epsilon, int num_tokens, int hidden_size,
    cudaStream_t stream)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    fused_add_layer_norm_bias_kernel<float><<<num_tokens, threads, 0, stream>>>(
        input, residual, weight, bias, epsilon, hidden_size);
}

void fused_add_layer_norm_bias_f16(
    __half* input, __half* residual, const __half* weight, const __half* bias,
    float epsilon, int num_tokens, int hidden_size,
    cudaStream_t stream)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    fused_add_layer_norm_bias_kernel<__half><<<num_tokens, threads, 0, stream>>>(
        input, residual, weight, bias, epsilon, hidden_size);
}

void fused_add_layer_norm_bias_bf16(
    __nv_bfloat16* input, __nv_bfloat16* residual,
    const __nv_bfloat16* weight, const __nv_bfloat16* bias,
    float epsilon, int num_tokens, int hidden_size,
    cudaStream_t stream)
{
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    fused_add_layer_norm_bias_kernel<__nv_bfloat16><<<num_tokens, threads, 0, stream>>>(
        input, residual, weight, bias, epsilon, hidden_size);
}

} // extern "C"
