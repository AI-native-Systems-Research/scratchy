// SPDX-License-Identifier: Apache-2.0
// Fused GDN gating CUDA kernel for Qwen3-Next.
//
// Computes:
//   g[i] = -exp(A_log[h]) * softplus(a[i] + dt_bias[h])
//   beta[i] = sigmoid(b[i])
//
// Port of: fused_gdn_gating_kernel (Triton) from qwen3_next.py

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

__device__ __forceinline__ float softplus_f(float x) {
    if (x > 20.0f) return x;
    return logf(1.0f + expf(x));
}

__global__ void fused_gdn_gating_kernel(
    float* __restrict__ g_out,
    float* __restrict__ beta_out,
    const float* __restrict__ A_log,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ dt_bias,
    int num_heads,
    int batch_size
) {
    int i_b = blockIdx.x;
    int head = blockIdx.y * blockDim.x + threadIdx.x;

    if (i_b >= batch_size || head >= num_heads) return;

    int off = i_b * num_heads + head;

    float x = a[off] + dt_bias[head];
    float sp = softplus_f(x);
    g_out[off] = -expf(A_log[head]) * sp;
    beta_out[off] = 1.0f / (1.0f + expf(-b[off]));
}

// C entry point
extern "C" void fused_gdn_gating(
    float* g_out, float* beta_out,
    const float* A_log, const float* a, const float* b, const float* dt_bias,
    int num_heads, int batch_size,
    cudaStream_t stream)
{
    int threads = (num_heads < 256) ? num_heads : 256;
    dim3 grid(batch_size, (num_heads + threads - 1) / threads);
    fused_gdn_gating_kernel<<<grid, threads, 0, stream>>>(
        g_out, beta_out, A_log, a, b, dt_bias, num_heads, batch_size);
}
