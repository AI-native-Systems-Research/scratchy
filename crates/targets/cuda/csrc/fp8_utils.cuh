// SPDX-License-Identifier: Apache-2.0
// FP8 E4M3 conversion utilities for CUDA kernels.
//
// Scale convention (matching Python vLLM exactly):
//   fp8_stored = fp8_cast(bf16_val / scale)
//   bf16_recovered = fp8_to_float(fp8_stored) * scale
//
// Requires SM89+ (Ada Lovelace / Hopper) for __nv_fp8_e4m3 hardware support.

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ---------------------------------------------------------------------------
// BF16 <-> FP8 E4M3 (raw uint8_t storage)
// ---------------------------------------------------------------------------

/// Convert BF16 to FP8 E4M3 raw byte with scale: fp8 = cast(bf16 * inv_scale).
/// Uses saturation to clamp out-of-range values to FP8 max (±448).
__device__ __forceinline__ uint8_t bf16_to_fp8e4m3(
    __nv_bfloat16 val, float inv_scale)
{
    float fval = __bfloat162float(val) * inv_scale;
    __nv_fp8_e4m3 fp8(fval);  // saturating cast
    return *reinterpret_cast<uint8_t*>(&fp8);
}

/// Convert FP8 E4M3 raw byte to BF16 with scale: bf16 = fp8_to_float(fp8) * scale.
__device__ __forceinline__ __nv_bfloat16 fp8e4m3_to_bf16(
    uint8_t fp8_byte, float scale)
{
    __nv_fp8_e4m3 fp8 = *reinterpret_cast<__nv_fp8_e4m3*>(&fp8_byte);
    float fval = float(fp8) * scale;
    return __float2bfloat16(fval);
}

// ---------------------------------------------------------------------------
// F16 <-> FP8 E4M3 (raw uint8_t storage)
// ---------------------------------------------------------------------------

/// Convert F16 to FP8 E4M3 raw byte with scale: fp8 = cast(f16 * inv_scale).
__device__ __forceinline__ uint8_t f16_to_fp8e4m3(
    __half val, float inv_scale)
{
    float fval = __half2float(val) * inv_scale;
    __nv_fp8_e4m3 fp8(fval);  // saturating cast
    return *reinterpret_cast<uint8_t*>(&fp8);
}

/// Convert FP8 E4M3 raw byte to F16 with scale: f16 = fp8_to_float(fp8) * scale.
__device__ __forceinline__ __half fp8e4m3_to_f16(
    uint8_t fp8_byte, float scale)
{
    __nv_fp8_e4m3 fp8 = *reinterpret_cast<__nv_fp8_e4m3*>(&fp8_byte);
    float fval = float(fp8) * scale;
    return __float2half(fval);
}
