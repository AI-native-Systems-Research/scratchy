// SPDX-License-Identifier: Apache-2.0
// bf16 <-> fp32 elementwise casts. Used by `NcclGroup::all_reduce_
// inplace_promote` to do fp32-precision NCCL all-reduce on bf16
// tensors — matches Python vLLM's `custom_all_reduce.cuh` precision
// (upcast bf16 -> fp32 -> sum -> downcast fp32 -> bf16) which
// avoids ~1 ULP per-reduce drift compared to NCCL's native bf16 sum.
// Compounded over ~73 AllReduces (per Qwen2.5-3B forward) the bf16
// drift is enough to produce wrong sampled tokens on small models.

#include <cuda_bf16.h>
#include <cstdint>

template <int VEC>
__global__ void bf16_to_fp32_vec_kernel(
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx * VEC;
    if (base >= n) return;
    #pragma unroll
    for (int i = 0; i < VEC; i++) {
        int j = base + i;
        if (j < n) dst[j] = __bfloat162float(src[j]);
    }
}

template <int VEC>
__global__ void fp32_to_bf16_vec_kernel(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx * VEC;
    if (base >= n) return;
    #pragma unroll
    for (int i = 0; i < VEC; i++) {
        int j = base + i;
        if (j < n) dst[j] = __float2bfloat16(src[j]);
    }
}

extern "C" {

void bf16_to_fp32(const void* src, void* dst, int n, cudaStream_t stream) {
    constexpr int VEC = 4;
    int threads = 256;
    int total_vecs = (n + VEC - 1) / VEC;
    int blocks = (total_vecs + threads - 1) / threads;
    bf16_to_fp32_vec_kernel<VEC><<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)src, (float*)dst, n);
}

void fp32_to_bf16(const void* src, void* dst, int n, cudaStream_t stream) {
    constexpr int VEC = 4;
    int threads = 256;
    int total_vecs = (n + VEC - 1) / VEC;
    int blocks = (total_vecs + threads - 1) / threads;
    fp32_to_bf16_vec_kernel<VEC><<<blocks, threads, 0, stream>>>(
        (const float*)src, (__nv_bfloat16*)dst, n);
}

}  // extern "C"
