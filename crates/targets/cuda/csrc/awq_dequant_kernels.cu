// SPDX-License-Identifier: Apache-2.0
// GPU dequantization kernel for AWQ INT4 packed weights.
//
// Dequantizes packed INT4-in-INT32 weights to BF16/F16/F32 on GPU.
// AWQ packs along columns (dim 1): qweight has shape [in_features, out_features/8].
// AWQ uses interleave order: nibble j stores logical column [0,2,4,6,1,3,5,7][j].
//
// Reverse-interleave (logical col offset -> nibble index):
//   col 0->nibble 0, col 1->nibble 4, col 2->nibble 1, col 3->nibble 5,
//   col 4->nibble 2, col 5->nibble 6, col 6->nibble 3, col 7->nibble 7
// Pattern: even cols -> col/2, odd cols -> (col-1)/2 + 4

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// AWQ dequantize kernel
// ---------------------------------------------------------------------------

template <typename T>
__global__ void awq_dequantize_kernel(
    T* __restrict__ out,                 // [in_features, out_features]
    const int32_t* __restrict__ qweight, // [in_features, out_features/8]
    const int32_t* __restrict__ qzeros,  // [num_groups, out_features/8]
    const T* __restrict__ scales,        // [num_groups, out_features]
    int in_features, int out_features, int group_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= in_features || col >= out_features) return;

    int packed_col = col / 8;
    int col_within = col % 8;

    // AWQ reverse-interleave: logical col offset -> nibble index
    int nibble_idx = (col_within % 2 == 0) ? col_within / 2 : (col_within - 1) / 2 + 4;
    int bit_offset = nibble_idx * 4;

    int packed_cols = (out_features + 7) / 8;

    // Unpack weight value
    int32_t packed = qweight[row * packed_cols + packed_col];
    int val = (packed >> bit_offset) & 0xF;

    // Group and zero point (qzeros uses same interleave packing)
    int group = row / group_size;
    int32_t packed_zero = qzeros[group * packed_cols + packed_col];
    int zero = (packed_zero >> bit_offset) & 0xF;

    // Scale and output
    float scale = float(scales[group * out_features + col]);
    float result = scale * (float(val) - float(zero));
    out[row * out_features + col] = T(result);
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

extern "C" void awq_dequantize_f32(
    float* out, const int32_t* qweight, const int32_t* qzeros,
    const float* scales, int in_features, int out_features, int group_size)
{
    dim3 block(32, 32);
    dim3 grid((out_features + 31) / 32, (in_features + 31) / 32);
    awq_dequantize_kernel<float><<<grid, block>>>(
        out, qweight, qzeros, scales,
        in_features, out_features, group_size);
}

extern "C" void awq_dequantize_f16(
    __half* out, const int32_t* qweight, const int32_t* qzeros,
    const __half* scales, int in_features, int out_features, int group_size)
{
    dim3 block(32, 32);
    dim3 grid((out_features + 31) / 32, (in_features + 31) / 32);
    awq_dequantize_kernel<__half><<<grid, block>>>(
        out, qweight, qzeros, scales,
        in_features, out_features, group_size);
}

extern "C" void awq_dequantize_bf16(
    __nv_bfloat16* out, const int32_t* qweight, const int32_t* qzeros,
    const __nv_bfloat16* scales, int in_features, int out_features, int group_size)
{
    dim3 block(32, 32);
    dim3 grid((out_features + 31) / 32, (in_features + 31) / 32);
    awq_dequantize_kernel<__nv_bfloat16><<<grid, block>>>(
        out, qweight, qzeros, scales,
        in_features, out_features, group_size);
}
