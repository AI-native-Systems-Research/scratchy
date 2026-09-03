/*
 * MoE alignment and reduction kernels, adapted from Python vLLM's
 * csrc/moe/moe_align_sum_kernels.cu.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stripped PyTorch/ATen dependencies, added extern "C" launchers for Rust FFI.
 * Kept: moe_sum_kernel (weighted reduction across top-k experts).
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Inline cuda_compat.h macros (CUDA-only)
#define VLLM_LDG(arg) __ldg(arg)

namespace vllm {
namespace moe {

// ====================== moe_sum_kernel ===============================
// Reduces [num_tokens, topk, d] -> [num_tokens, d] by summing across topk.
template <typename scalar_t, int TOPK>
__global__ void moe_sum_kernel(
    scalar_t* __restrict__ out,          // [num_tokens, d]
    const scalar_t* __restrict__ input,  // [num_tokens, topk, d]
    const int d)
{
    const int64_t token_idx = blockIdx.x;
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
        scalar_t x = static_cast<scalar_t>(0);
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            x += VLLM_LDG(&input[token_idx * TOPK * d + k * d + idx]);
        }
        out[token_idx * d + idx] = x;
    }
}

} // namespace moe
} // namespace vllm

// =====================================================================
// extern "C" launchers for Rust FFI
// =====================================================================

// Helper: launch moe_sum for a given type and topk
template <typename scalar_t>
static void launch_moe_sum(
    scalar_t* out,
    const scalar_t* input,
    int num_tokens,
    int hidden_size,
    int topk,
    cudaStream_t stream)
{
    dim3 grid(num_tokens);
    dim3 block(min(hidden_size, 1024));

    switch (topk) {
        case 1:
            vllm::moe::moe_sum_kernel<scalar_t, 1><<<grid, block, 0, stream>>>(out, input, hidden_size);
            break;
        case 2:
            vllm::moe::moe_sum_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(out, input, hidden_size);
            break;
        case 3:
            vllm::moe::moe_sum_kernel<scalar_t, 3><<<grid, block, 0, stream>>>(out, input, hidden_size);
            break;
        case 4:
            vllm::moe::moe_sum_kernel<scalar_t, 4><<<grid, block, 0, stream>>>(out, input, hidden_size);
            break;
        case 5:
            vllm::moe::moe_sum_kernel<scalar_t, 5><<<grid, block, 0, stream>>>(out, input, hidden_size);
            break;
        case 6:
            vllm::moe::moe_sum_kernel<scalar_t, 6><<<grid, block, 0, stream>>>(out, input, hidden_size);
            break;
        case 8:
            vllm::moe::moe_sum_kernel<scalar_t, 8><<<grid, block, 0, stream>>>(out, input, hidden_size);
            break;
        default:
            vllm::moe::moe_sum_kernel<scalar_t, 4><<<grid, block, 0, stream>>>(out, input, hidden_size);
            break;
    }
}

extern "C" void moe_sum_f32(
    float* out,
    const float* input,
    int num_tokens,
    int hidden_size,
    int topk,
    cudaStream_t stream)
{
    launch_moe_sum<float>(out, input, num_tokens, hidden_size, topk, stream);
}

extern "C" void moe_sum_f16(
    void* out,
    const void* input,
    int num_tokens,
    int hidden_size,
    int topk,
    cudaStream_t stream)
{
    launch_moe_sum<__half>(
        reinterpret_cast<__half*>(out),
        reinterpret_cast<const __half*>(input),
        num_tokens, hidden_size, topk, stream);
}

extern "C" void moe_sum_bf16(
    void* out,
    const void* input,
    int num_tokens,
    int hidden_size,
    int topk,
    cudaStream_t stream)
{
    launch_moe_sum<__nv_bfloat16>(
        reinterpret_cast<__nv_bfloat16*>(out),
        reinterpret_cast<const __nv_bfloat16*>(input),
        num_tokens, hidden_size, topk, stream);
}
