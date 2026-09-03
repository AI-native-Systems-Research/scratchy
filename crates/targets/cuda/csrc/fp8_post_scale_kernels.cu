// SPDX-License-Identifier: Apache-2.0
// Post-GEMM per-row scale multiply for FP8 dynamic activation quantization.
//
// After cublasLt FP8 GEMM (which only supports scalar scale pointers),
// apply the per-token activation scales as a row-wise multiply:
//   output[i, :] *= activation_scale[i]
//
// This matches the behavior of CUTLASS cutlass_scaled_mm which fuses
// per-row scale_a into the GEMM kernel.
//
// Also includes FP8 re-quantization kernel for fused module scale merging.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ---------------------------------------------------------------------------
// Kernel 1: Row-wise scale multiply (BF16 in-place)
//
// output[row, col] *= scales[row]
// One block per row, vectorized 8-wide for coalesced access.
// ---------------------------------------------------------------------------

__global__ void row_scale_multiply_bf16_kernel(
    uint16_t* __restrict__ output,       // [M, N] BF16 — modified in place
    const float* __restrict__ scales,    // [M] f32
    int N)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float scale = scales[row];
    const int row_offset = row * N;

    // Vectorized: 8 BF16 at a time (16 bytes = uint4)
    const int vec_elems = 8;
    const int vec_iters = N / vec_elems;
    const int remainder_start = vec_iters * vec_elems;

    for (int vi = tid; vi < vec_iters; vi += blockDim.x) {
        int base = row_offset + vi * vec_elems;
        uint4 in_vec = *reinterpret_cast<uint4*>(&output[base]);
        uint16_t* vals = reinterpret_cast<uint16_t*>(&in_vec);

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float fval = __bfloat162float(
                *reinterpret_cast<__nv_bfloat16*>(&vals[j]));
            fval *= scale;
            __nv_bfloat16 bf = __float2bfloat16(fval);
            vals[j] = *reinterpret_cast<uint16_t*>(&bf);
        }

        *reinterpret_cast<uint4*>(&output[base]) = in_vec;
    }

    // Remainder
    for (int i = remainder_start + tid; i < N; i += blockDim.x) {
        int idx = row_offset + i;
        float fval = __bfloat162float(
            *reinterpret_cast<__nv_bfloat16*>(&output[idx]));
        fval *= scale;
        __nv_bfloat16 bf = __float2bfloat16(fval);
        output[idx] = *reinterpret_cast<uint16_t*>(&bf);
    }
}

// F16 variant
__global__ void row_scale_multiply_f16_kernel(
    uint16_t* __restrict__ output,       // [M, N] F16
    const float* __restrict__ scales,    // [M] f32
    int N)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float scale = scales[row];
    const int row_offset = row * N;

    const int vec_elems = 8;
    const int vec_iters = N / vec_elems;
    const int remainder_start = vec_iters * vec_elems;

    for (int vi = tid; vi < vec_iters; vi += blockDim.x) {
        int base = row_offset + vi * vec_elems;
        uint4 in_vec = *reinterpret_cast<uint4*>(&output[base]);
        uint16_t* vals = reinterpret_cast<uint16_t*>(&in_vec);

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float fval = __half2float(*reinterpret_cast<__half*>(&vals[j]));
            fval *= scale;
            __half hf = __float2half(fval);
            vals[j] = *reinterpret_cast<uint16_t*>(&hf);
        }

        *reinterpret_cast<uint4*>(&output[base]) = in_vec;
    }

    for (int i = remainder_start + tid; i < N; i += blockDim.x) {
        int idx = row_offset + i;
        float fval = __half2float(*reinterpret_cast<__half*>(&output[idx]));
        fval *= scale;
        __half hf = __float2half(fval);
        output[idx] = *reinterpret_cast<uint16_t*>(&hf);
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: FP8 re-quantize rows with new scale
//
// For fused module scale merging (QKV, gate_up):
// Given FP8 weight quantized with old_scale, re-quantize with new_scale:
//   new_fp8[i] = quantize_fp8(dequantize(old_fp8[i], old_scale), new_scale)
//             = quantize_fp8(old_fp8[i] * old_scale / new_scale)
//
// Processes a contiguous shard of rows [start_row, start_row + num_rows).
// ---------------------------------------------------------------------------

__global__ void fp8_requantize_rows_kernel(
    uint8_t* __restrict__ weight,       // [total_rows, K] FP8 — modified in place
    int K,
    int start_row,
    float scale_ratio)                  // old_scale / new_scale
{
    const int row = blockIdx.x + start_row;
    const int tid = threadIdx.x;
    const int row_offset = row * K;

    // Vectorized: 8 FP8 bytes at a time (= 2 uint32)
    const int vec_elems = 8;
    const int vec_iters = K / vec_elems;
    const int remainder_start = vec_iters * vec_elems;

    for (int vi = tid; vi < vec_iters; vi += blockDim.x) {
        int base = row_offset + vi * vec_elems;
        uint2 in_vec = *reinterpret_cast<uint2*>(&weight[base]);
        uint8_t* bytes = reinterpret_cast<uint8_t*>(&in_vec);

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            // Dequant: FP8 → float, apply ratio, re-quantize
            __nv_fp8_e4m3 old_fp8 = *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[j]);
            float fval = float(old_fp8) * scale_ratio;
            // Clamp to FP8 range before re-quantizing
            fval = fminf(fmaxf(fval, -448.0f), 448.0f);
            __nv_fp8_e4m3 new_fp8(fval);
            bytes[j] = *reinterpret_cast<uint8_t*>(&new_fp8);
        }

        *reinterpret_cast<uint2*>(&weight[base]) = in_vec;
    }

    // Remainder
    for (int i = remainder_start + tid; i < K; i += blockDim.x) {
        int idx = row_offset + i;
        __nv_fp8_e4m3 old_fp8 = *reinterpret_cast<__nv_fp8_e4m3*>(&weight[idx]);
        float fval = float(old_fp8) * scale_ratio;
        fval = fminf(fmaxf(fval, -448.0f), 448.0f);
        __nv_fp8_e4m3 new_fp8(fval);
        weight[idx] = *reinterpret_cast<uint8_t*>(&new_fp8);
    }
}

// F32 variant — used by GGML MoE forward to apply per-task topk weights.
__global__ void row_scale_multiply_f32_kernel(
    float* __restrict__ output,          // [M, N] F32
    const float* __restrict__ scales,    // [M] f32
    int N)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float scale = scales[row];
    const int row_offset = row * N;

    for (int i = tid; i < N; i += blockDim.x) {
        output[row_offset + i] *= scale;
    }
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

extern "C" {

// Per-row scale multiply: output[i,:] *= scales[i]
void fp8_row_scale_multiply_bf16(
    uint16_t* output,           // [M, N] BF16 — modified in place
    const float* scales,        // [M] f32
    int M,
    int N,
    cudaStream_t stream)
{
    const int threads = 256;
    row_scale_multiply_bf16_kernel<<<M, threads, 0, stream>>>(
        output, scales, N);
}

void fp8_row_scale_multiply_f16(
    uint16_t* output,           // [M, N] F16
    const float* scales,        // [M] f32
    int M,
    int N,
    cudaStream_t stream)
{
    const int threads = 256;
    row_scale_multiply_f16_kernel<<<M, threads, 0, stream>>>(
        output, scales, N);
}

void fp8_row_scale_multiply_f32(
    float* output,              // [M, N] F32 — modified in place
    const float* scales,        // [M] f32
    int M,
    int N,
    cudaStream_t stream)
{
    const int threads = 256;
    row_scale_multiply_f32_kernel<<<M, threads, 0, stream>>>(
        output, scales, N);
}

// Re-quantize FP8 rows with new scale.
// Processes rows [start_row, start_row + num_rows) in weight.
// scale_ratio = old_scale / new_scale
void fp8_requantize_rows(
    uint8_t* weight,            // [total_rows, K] FP8 — modified in place
    int K,
    int start_row,
    int num_rows,
    float scale_ratio,
    cudaStream_t stream)
{
    if (num_rows == 0) return;
    const int threads = 256;
    fp8_requantize_rows_kernel<<<num_rows, threads, 0, stream>>>(
        weight, K, start_row, scale_ratio);
}

} // extern "C"
