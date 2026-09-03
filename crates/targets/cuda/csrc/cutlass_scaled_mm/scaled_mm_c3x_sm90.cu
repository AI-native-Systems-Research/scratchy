// SPDX-License-Identifier: Apache-2.0
// CUTLASS 3.x scaled_mm kernel instantiation for SM90a (Hopper / H100).
// Implements FP8 E4M3 GEMM with a fused per-row/per-tensor scale epilogue,
// matching Python vLLM's cutlass_scaled_mm_sm90 (the C3X path).
//
// The extern "C" entry points are ALWAYS defined so the Rust FFI links on any
// build host. The real kernel body is compiled only when ENABLE_SCALED_MM_SM90
// is set (i.e. when vllm-kernels-cuda's build.rs detects an sm_90+ build host
// and targets sm_90a). On other hosts the body is a hard-fail stub — it is
// never reached at runtime because the Rust dispatch only routes SM90 devices
// here (see scratchy-target-cuda/src/kernels.rs::cutlass_scaled_mm).

#include <cstdio>
#include <cstdlib>
#include "cuda_runtime.h"

#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
  #include "scaled_mm_c3x_sm90_fp8_dispatch.cuh"
#endif

extern "C" {

/// FP8 scaled matmul on SM90 (Hopper), no bias.
///
/// c [M, N] = scale_a * (A_fp8 [M, K] @ B_fp8 [N, K]^T) * scale_b
///
/// A row-major [M, K] FP8 E4M3; B row-major [N, K] FP8 E4M3 (col-major [K, N]);
/// C row-major [M, N] BF16 or F16.
/// a_scales: [M] (per-token) or [1] (per-tensor); b_scales: [N] or [1].
/// out_dtype: 0 = BF16, 1 = F16
void cutlass_scaled_mm_sm90(void* c, const void* a, const void* b,
                            const float* a_scales, int a_scales_numel,
                            const float* b_scales, int b_scales_numel, int M,
                            int N, int K, int out_dtype, cudaStream_t stream) {
#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
  if (out_dtype == 0) {
    vllm::cutlass_gemm_sm90_fp8_dispatch<cutlass::float_e4m3_t,
                                         cutlass::bfloat16_t, /*EnableBias=*/false>(
        c, a, b, M, N, K, stream, a_scales, a_scales_numel, b_scales,
        b_scales_numel);
  } else {
    vllm::cutlass_gemm_sm90_fp8_dispatch<cutlass::float_e4m3_t, cutlass::half_t,
                                         /*EnableBias=*/false>(
        c, a, b, M, N, K, stream, a_scales, a_scales_numel, b_scales,
        b_scales_numel);
  }
#else
  (void)c; (void)a; (void)b; (void)a_scales; (void)a_scales_numel;
  (void)b_scales; (void)b_scales_numel; (void)M; (void)N; (void)K;
  (void)out_dtype; (void)stream;
  fprintf(stderr,
          "FATAL: cutlass_scaled_mm_sm90 called but the SM90 FP8 kernel was "
          "not compiled (build host was not sm_90+).\n");
  abort();
#endif
}

/// FP8 scaled matmul on SM90 (Hopper) with per-output-channel bias.
void cutlass_scaled_mm_bias_sm90(void* c, const void* a, const void* b,
                                 const float* a_scales, int a_scales_numel,
                                 const float* b_scales, int b_scales_numel,
                                 const void* bias, int M, int N, int K,
                                 int out_dtype, cudaStream_t stream) {
#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
  if (out_dtype == 0) {
    vllm::cutlass_gemm_sm90_fp8_dispatch<cutlass::float_e4m3_t,
                                         cutlass::bfloat16_t, /*EnableBias=*/true>(
        c, a, b, M, N, K, stream, a_scales, a_scales_numel, b_scales,
        b_scales_numel, static_cast<const cutlass::bfloat16_t*>(bias));
  } else {
    vllm::cutlass_gemm_sm90_fp8_dispatch<cutlass::float_e4m3_t, cutlass::half_t,
                                         /*EnableBias=*/true>(
        c, a, b, M, N, K, stream, a_scales, a_scales_numel, b_scales,
        b_scales_numel, static_cast<const cutlass::half_t*>(bias));
  }
#else
  (void)c; (void)a; (void)b; (void)a_scales; (void)a_scales_numel;
  (void)b_scales; (void)b_scales_numel; (void)bias; (void)M; (void)N; (void)K;
  (void)out_dtype; (void)stream;
  fprintf(stderr,
          "FATAL: cutlass_scaled_mm_bias_sm90 called but the SM90 FP8 kernel "
          "was not compiled (build host was not sm_90+).\n");
  abort();
#endif
}

}  // extern "C"
