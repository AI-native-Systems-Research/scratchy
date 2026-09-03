// SPDX-License-Identifier: Apache-2.0
// CUTLASS 2.x scaled_mm kernel instantiation for SM89 (Ada Lovelace / L40S).
// Implements FP8 E4M3 GEMM with fused per-row/per-tensor scale epilogue.
// Matches Python vLLM's cutlass_scaled_mm_sm89 exactly.

#include "scaled_mm_c2x.cuh"
#include "scaled_mm_c2x_sm89_fp8_dispatch.cuh"
#include "scaled_mm_epilogues_c2x.hpp"

using namespace vllm;

// ---------------------------------------------------------------------------
// Internal dispatch: FP8 E4M3 → BF16 output
// ---------------------------------------------------------------------------
static void cutlass_scaled_mm_sm89_fp8_bf16(
    void* c, const void* a, const void* b,
    const float* a_scales, int a_scales_numel,
    const float* b_scales, int b_scales_numel,
    int32_t m, int32_t n, int32_t k,
    int64_t lda, int64_t ldb, int64_t ldc,
    cudaStream_t stream)
{
  cutlass_gemm_sm89_fp8_dispatch<
      cutlass::float_e4m3_t, cutlass::bfloat16_t, c2x::ScaledEpilogue>(
      c, a, b, m, n, k, lda, ldb, ldc, stream,
      a_scales, a_scales_numel,
      b_scales, b_scales_numel);
}

// ---------------------------------------------------------------------------
// Internal dispatch: FP8 E4M3 → F16 output
// ---------------------------------------------------------------------------
static void cutlass_scaled_mm_sm89_fp8_f16(
    void* c, const void* a, const void* b,
    const float* a_scales, int a_scales_numel,
    const float* b_scales, int b_scales_numel,
    int32_t m, int32_t n, int32_t k,
    int64_t lda, int64_t ldb, int64_t ldc,
    cudaStream_t stream)
{
  cutlass_gemm_sm89_fp8_dispatch<
      cutlass::float_e4m3_t, cutlass::half_t, c2x::ScaledEpilogue>(
      c, a, b, m, n, k, lda, ldb, ldc, stream,
      a_scales, a_scales_numel,
      b_scales, b_scales_numel);
}

// ---------------------------------------------------------------------------
// Internal dispatch: FP8 E4M3 → BF16 output with bias
// ---------------------------------------------------------------------------
static void cutlass_scaled_mm_sm89_fp8_bf16_bias(
    void* c, const void* a, const void* b,
    const float* a_scales, int a_scales_numel,
    const float* b_scales, int b_scales_numel,
    const void* bias,
    int32_t m, int32_t n, int32_t k,
    int64_t lda, int64_t ldb, int64_t ldc,
    cudaStream_t stream)
{
  cutlass_gemm_sm89_fp8_dispatch<
      cutlass::float_e4m3_t, cutlass::bfloat16_t, c2x::ScaledEpilogueBias>(
      c, a, b, m, n, k, lda, ldb, ldc, stream,
      a_scales, a_scales_numel,
      b_scales, b_scales_numel,
      static_cast<const cutlass::bfloat16_t*>(bias));
}

// ---------------------------------------------------------------------------
// Internal dispatch: FP8 E4M3 → F16 output with bias
// ---------------------------------------------------------------------------
static void cutlass_scaled_mm_sm89_fp8_f16_bias(
    void* c, const void* a, const void* b,
    const float* a_scales, int a_scales_numel,
    const float* b_scales, int b_scales_numel,
    const void* bias,
    int32_t m, int32_t n, int32_t k,
    int64_t lda, int64_t ldb, int64_t ldc,
    cudaStream_t stream)
{
  cutlass_gemm_sm89_fp8_dispatch<
      cutlass::float_e4m3_t, cutlass::half_t, c2x::ScaledEpilogueBias>(
      c, a, b, m, n, k, lda, ldb, ldc, stream,
      a_scales, a_scales_numel,
      b_scales, b_scales_numel,
      static_cast<const cutlass::half_t*>(bias));
}

// ---------------------------------------------------------------------------
// C entry points (called from Rust via FFI)
// ---------------------------------------------------------------------------
extern "C" {

/// FP8 scaled matmul on SM89, no bias.
///
/// c [M, N] = scale_a * (A_fp8 [M, K] @ B_fp8 [N, K]^T) * scale_b
///
/// A is row-major [M, K] FP8 E4M3.
/// B is row-major [N, K] FP8 E4M3 (treated as column-major [K, N] by CUTLASS).
/// C is row-major [M, N] BF16 or F16.
///
/// a_scales: [M] f32 (per-token) or [1] f32 (per-tensor)
/// b_scales: [N] f32 (per-channel) or [1] f32 (per-tensor)
/// out_dtype: 0 = BF16, 1 = F16
void cutlass_scaled_mm_sm89(
    void* c,
    const void* a,
    const void* b,
    const float* a_scales,
    int a_scales_numel,
    const float* b_scales,
    int b_scales_numel,
    int M, int N, int K,
    int out_dtype,       // 0 = BF16, 1 = F16
    cudaStream_t stream)
{
  int64_t lda = K;
  int64_t ldb = K;  // B is [N, K] row-major = [K, N] col-major, stride = K
  int64_t ldc = N;

  if (out_dtype == 0) {
    // BF16 output
    cutlass_scaled_mm_sm89_fp8_bf16(
        c, a, b, a_scales, a_scales_numel, b_scales, b_scales_numel,
        M, N, K, lda, ldb, ldc, stream);
  } else {
    // F16 output
    cutlass_scaled_mm_sm89_fp8_f16(
        c, a, b, a_scales, a_scales_numel, b_scales, b_scales_numel,
        M, N, K, lda, ldb, ldc, stream);
  }
}

/// FP8 scaled matmul on SM89 with bias.
void cutlass_scaled_mm_bias_sm89(
    void* c,
    const void* a,
    const void* b,
    const float* a_scales,
    int a_scales_numel,
    const float* b_scales,
    int b_scales_numel,
    const void* bias,     // [N] same dtype as output
    int M, int N, int K,
    int out_dtype,        // 0 = BF16, 1 = F16
    cudaStream_t stream)
{
  int64_t lda = K;
  int64_t ldb = K;
  int64_t ldc = N;

  if (out_dtype == 0) {
    cutlass_scaled_mm_sm89_fp8_bf16_bias(
        c, a, b, a_scales, a_scales_numel, b_scales, b_scales_numel,
        bias, M, N, K, lda, ldb, ldc, stream);
  } else {
    cutlass_scaled_mm_sm89_fp8_f16_bias(
        c, a, b, a_scales, a_scales_numel, b_scales, b_scales_numel,
        bias, M, N, K, lda, ldb, ldc, stream);
  }
}

} // extern "C"
