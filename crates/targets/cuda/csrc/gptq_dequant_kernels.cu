// SPDX-License-Identifier: Apache-2.0
// GPU dequantization kernel for GPTQ INT4 packed weights.
//
// Dequantizes packed INT4-in-INT32 weights to BF16/F16/F32 on GPU.
// GPTQ packs along rows (dim 0): qweight has shape [in_features/8, out_features].

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// GPTQ dequantize kernel
// ---------------------------------------------------------------------------

template <typename T>
__global__ void gptq_dequantize_kernel(
    T* __restrict__ out,                 // [in_features, out_features]
    const int32_t* __restrict__ qweight, // [in_features/8, out_features]
    const int32_t* __restrict__ qzeros,  // [num_groups, out_features/8]
    const T* __restrict__ scales,        // [num_groups, out_features]
    const int32_t* __restrict__ g_idx,   // [in_features] or nullptr
    int in_features, int out_features, int group_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= in_features || col >= out_features) return;

    // Unpack the 4-bit value from qweight
    int packed_row = row / 8;
    int bit_offset = (row % 8) * 4;
    int32_t packed = qweight[packed_row * out_features + col];
    int val = (packed >> bit_offset) & 0xF;

    // Determine group index
    int group = g_idx ? g_idx[row] : row / group_size;

    // Unpack zero point from qzeros (packed along columns)
    int packed_zero_col = col / 8;
    int zero_bit = (col % 8) * 4;
    int32_t packed_zero = qzeros[group * ((out_features + 7) / 8) + packed_zero_col];
    int zero = (packed_zero >> zero_bit) & 0xF;

    // Scale and output
    float scale = float(scales[group * out_features + col]);
    float result = scale * (float(val) - float(zero));
    out[row * out_features + col] = T(result);
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

extern "C" void gptq_dequantize_f32(
    float* out, const int32_t* qweight, const int32_t* qzeros,
    const float* scales, const int32_t* g_idx,
    int in_features, int out_features, int group_size)
{
    dim3 block(32, 32);
    dim3 grid((out_features + 31) / 32, (in_features + 31) / 32);
    gptq_dequantize_kernel<float><<<grid, block>>>(
        out, qweight, qzeros, scales, g_idx,
        in_features, out_features, group_size);
}

extern "C" void gptq_dequantize_f16(
    __half* out, const int32_t* qweight, const int32_t* qzeros,
    const __half* scales, const int32_t* g_idx,
    int in_features, int out_features, int group_size)
{
    dim3 block(32, 32);
    dim3 grid((out_features + 31) / 32, (in_features + 31) / 32);
    gptq_dequantize_kernel<__half><<<grid, block>>>(
        out, qweight, qzeros, scales, g_idx,
        in_features, out_features, group_size);
}

extern "C" void gptq_dequantize_bf16(
    __nv_bfloat16* out, const int32_t* qweight, const int32_t* qzeros,
    const __nv_bfloat16* scales, const int32_t* g_idx,
    int in_features, int out_features, int group_size)
{
    dim3 block(32, 32);
    dim3 grid((out_features + 31) / 32, (in_features + 31) / 32);
    gptq_dequantize_kernel<__nv_bfloat16><<<grid, block>>>(
        out, qweight, qzeros, scales, g_idx,
        in_features, out_features, group_size);
}
