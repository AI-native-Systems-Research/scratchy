// SPDX-License-Identifier: Apache-2.0
// FP8 block dequantization: FP8 weight [N, K] + per-block scale_inv
// [ceil(N/block_n), ceil(K/block_k)] → BF16/F16 [N, K].
//
// Each element: out[i][j] = fp8_to_float(weight[i][j]) * scale_inv[i/block_n][j/block_k]
// where scale_inv is the inverse of the quantization scale for that block.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ---------------------------------------------------------------------------
// Kernel: per-element FP8 → BF16 dequant with block-indexed scales
// ---------------------------------------------------------------------------

__global__ void fp8_block_dequant_bf16_kernel(
    const uint8_t* __restrict__ weight,       // [N, K] FP8
    const float* __restrict__ scale_inv,      // [ceil(N/block_n), ceil(K/block_k)] f32
    uint16_t* __restrict__ output,            // [N, K] BF16
    int N, int K,
    int block_n, int block_k,
    int scale_stride_n)  // = ceil(K/block_k)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    const int row = idx / K;
    const int col = idx % K;

    // Look up the block-level scale.
    int block_row = row / block_n;
    int block_col = col / block_k;
    float scale = scale_inv[block_row * scale_stride_n + block_col];

    // Dequantize: fp8 → float → scale → bf16
    __nv_fp8_e4m3 fp8 = *reinterpret_cast<const __nv_fp8_e4m3*>(&weight[idx]);
    float fval = float(fp8) * scale;
    __nv_bfloat16 bf16_val = __float2bfloat16(fval);
    output[idx] = *reinterpret_cast<uint16_t*>(&bf16_val);
}

__global__ void fp8_block_dequant_f16_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ scale_inv,
    uint16_t* __restrict__ output,
    int N, int K,
    int block_n, int block_k,
    int scale_stride_n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    const int row = idx / K;
    const int col = idx % K;

    int block_row = row / block_n;
    int block_col = col / block_k;
    float scale = scale_inv[block_row * scale_stride_n + block_col];

    __nv_fp8_e4m3 fp8 = *reinterpret_cast<const __nv_fp8_e4m3*>(&weight[idx]);
    float fval = float(fp8) * scale;
    __half h_val = __float2half(fval);
    output[idx] = *reinterpret_cast<uint16_t*>(&h_val);
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

extern "C" {

void fp8_block_dequant_bf16(
    const uint8_t* weight,
    const float* scale_inv,
    uint16_t* output,
    int N, int K,
    int block_n, int block_k,
    cudaStream_t stream)
{
    int total = N * K;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    int scale_stride_n = (K + block_k - 1) / block_k;
    fp8_block_dequant_bf16_kernel<<<blocks, threads, 0, stream>>>(
        weight, scale_inv, output, N, K, block_n, block_k, scale_stride_n);
}

void fp8_block_dequant_f16(
    const uint8_t* weight,
    const float* scale_inv,
    uint16_t* output,
    int N, int K,
    int block_n, int block_k,
    cudaStream_t stream)
{
    int total = N * K;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    int scale_stride_n = (K + block_k - 1) / block_k;
    fp8_block_dequant_f16_kernel<<<blocks, threads, 0, stream>>>(
        weight, scale_inv, output, N, K, block_n, block_k, scale_stride_n);
}

} // extern "C"
