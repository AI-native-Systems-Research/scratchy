// SPDX-License-Identifier: Apache-2.0
// FP8 KV scale computation: find abs-max of a BF16/F16 tensor and
// compute scale = abs_max / divisor.
//
// Used for --calculate-kv-scales (one-shot after first forward pass).

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>

// Atomic max for floats using __float_as_uint / __uint_as_float trick.
__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
    if (val >= 0.0f) {
        atomicMax(reinterpret_cast<unsigned int*>(addr), __float_as_uint(val));
    }
}

// ---------------------------------------------------------------------------
// Phase 1: per-block abs-max reduction
// Phase 2: write scale = abs_max / divisor
// ---------------------------------------------------------------------------

__global__ void compute_abs_max_bf16_kernel(
    const uint16_t* __restrict__ tensor,
    int num_elements,
    float* __restrict__ abs_max_out)  // single float on GPU, initialized to 0
{
    __shared__ float smem[256];
    const int tid = threadIdx.x;
    float local_max = 0.0f;

    // Grid-stride loop
    for (int i = blockIdx.x * blockDim.x + tid; i < num_elements;
         i += blockDim.x * gridDim.x) {
        float val = __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16*>(&tensor[i]));
        float abs_val = fabsf(val);
        if (abs_val > local_max) local_max = abs_val;
    }

    smem[tid] = local_max;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other = smem[tid + s];
            if (other > smem[tid]) smem[tid] = other;
        }
        __syncthreads();
    }

    // Write block result to global via atomic
    if (tid == 0) {
        atomicMaxFloat(abs_max_out, smem[0]);
    }
}

__global__ void finalize_scale_kernel(
    float* __restrict__ abs_max_and_scale,  // in-place: reads abs_max, writes scale
    float divisor)
{
    float abs_max = *abs_max_and_scale;
    // scale = abs_max / divisor. If abs_max is 0, scale = 1.0 (safe default).
    float scale = (abs_max > 0.0f) ? (abs_max / divisor) : 1.0f;
    *abs_max_and_scale = scale;
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

extern "C" {

void compute_abs_max_and_scale_bf16(
    const uint16_t* tensor,
    int num_elements,
    float divisor,
    float* scale_out,       // GPU scalar — will contain the computed scale
    cudaStream_t stream)
{
    // Zero the output first (abs_max accumulator)
    cudaMemsetAsync(scale_out, 0, sizeof(float), stream);

    // Launch abs-max reduction
    const int threads = 256;
    const int blocks = min((num_elements + threads - 1) / threads, 1024);
    compute_abs_max_bf16_kernel<<<blocks, threads, 0, stream>>>(
        tensor, num_elements, scale_out);

    // Finalize: scale = abs_max / divisor
    finalize_scale_kernel<<<1, 1, 0, stream>>>(scale_out, divisor);
}

} // extern "C"
