// Standalone cuBLAS bf16 GEMM benchmark for the LLaMA 1B prefill shapes.
// Measures FLOPS and latency at the same M/K/N as our megakernel hot path.
//
// Build: nvcc -O3 -arch=sm_89 bench.cu -lcublas -o bench
// Run:   ./bench

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cassert>

#define CHECK_CUDA(call) do { cudaError_t e = call; if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }} while(0)

#define CHECK_CUBLAS(call) do { cublasStatus_t s = call; if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); exit(1); }} while(0)

struct Shape {
    const char *name;
    int m, k, n;
};

// At seq=1024, LLaMA 1B per layer:
//   gate: A[1024,2048] x B[2048,8192]^T -> C[1024,8192]
//   up:   same
//   down: A[1024,8192] x B[8192,2048]^T -> C[1024,2048]
//   qkv:  A[1024,2048] x B[2048,2304]^T -> C[1024,2304]
//   o:    A[1024,2048] x B[2048,2048]^T -> C[1024,2048]
// per-layer flops = 2*M*K*N for each gemm.
// 16 layers total.
int main(int argc, char **argv) {
    int seq = 1024;
    if (argc > 1) seq = atoi(argv[1]);

    Shape shapes[] = {
        {"qkv",  seq, 2048, 2304},
        {"o",    seq, 2048, 2048},
        {"gate", seq, 2048, 8192},
        {"up",   seq, 2048, 8192},
        {"down", seq, 8192, 2048},
    };
    constexpr int n_shapes = sizeof(shapes)/sizeof(shapes[0]);
    constexpr int n_layers = 16;

    // L4 dense bf16 peak (from datasheet): 121 TFLOPS
    constexpr double l4_bf16_peak_tflops = 121.0;

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    // Force tensor-core math.
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    printf("=== L4 cuBLAS bf16 GEMM benchmark (seq=%d, %d layers) ===\n", seq, n_layers);
    printf("L4 dense bf16 peak: %.1f TFLOPS\n\n", l4_bf16_peak_tflops);
    printf("%-8s %6s %6s %6s  %10s  %12s  %8s  %12s\n",
           "name", "M", "K", "N", "lat_us", "TFLOPS", "% peak", "1L *16(ms)");

    double total_flops_per_pass = 0.0;
    double total_time_ms = 0.0;

    for (int s = 0; s < n_shapes; s++) {
        int M = shapes[s].m, K = shapes[s].k, N = shapes[s].n;

        // Allocate on device.
        __nv_bfloat16 *dA, *dB, *dC;
        CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&dC, M * N * sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMemset(dA, 0, M * K * sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMemset(dB, 0, K * N * sizeof(__nv_bfloat16)));

        // Compute D = A @ B.
        // Our convention: A is row-major [M,K], B is row-major [N,K] (i.e. weight stored
        // as out × in), and we want D = A @ B^T = [M,N].
        // cuBLAS is column-major. So with row-major A interpreted as col-major A^T [K,M]
        // and row-major B as col-major B^T [K,N], computing D = A @ B^T row-major
        // = (B @ A^T)^T col-major = treat as col-major dC[N,M] = B [N,K] * A [K,M].
        // i.e. cublasGemmEx with:
        //   m=N, n=M, k=K
        //   A_ptr = dB, op=N, lda=K (since B is row-major [N,K] = col-major [K,N], lda=K)
        //   wait this is getting confusing.
        //
        // Simpler: just call gemmEx with A^T x B = C convention.
        // We want C[M,N] = A[M,K] * B[N,K]^T (row-major for all).
        // In cuBLAS column-major convention this is:
        //   C^T[N,M] = (A * B^T)^T = B * A^T
        //   C^T col-major [N,M] = B col-major [N,K] * A^T col-major [K,M]
        // But our A row-major [M,K] = col-major [K,M] (= A^T_col).
        // And our B row-major [N,K] = col-major [K,N].
        // We need B as col-major [N,K]. That's B_row [K,N] which we don't have.
        //
        // Easier: just declare opA=N, opB=T and let cuBLAS handle.
        // For row-major output C[M,N] = A_row[M,K] * B_row[K,N], i.e. B is [K,N] not [N,K]
        // — but our weights are stored as [N,K] (out × in)! Let me just use opB=T.
        //
        // C[M,N] = A[M,K] * B[N,K]^T  (B is the weight, [N,K], so we transpose).
        // In col-major:
        //   C^T[N,M] = B[N,K] * A[M,K]^T
        // cublasGemmEx(opA=N, opB=T) with m=N, n=M, k=K, A=dB(weight, ld=K), B=dA(act, ld=K).
        // Result is C[M,N] in row-major = C^T[N,M] in col-major. Good.

        const float alpha = 1.0f, beta = 0.0f;

        // Warmup
        for (int w = 0; w < 4; w++) {
            CHECK_CUBLAS(cublasGemmEx(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                dB, CUDA_R_16BF, K,    // weight, row-major [N,K] = col-major [K,N], ldB=K
                dA, CUDA_R_16BF, K,    // act,    row-major [M,K] = col-major [K,M], ldA=K
                &beta,
                dC, CUDA_R_16BF, N,    // out,    row-major [M,N] = col-major [N,M], ldC=N
                CUBLAS_COMPUTE_32F,    // fp32 accumulator (matches our megakernel)
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        // Time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        const int iters = 200;
        cudaEventRecord(start);
        for (int it = 0; it < iters; it++) {
            CHECK_CUBLAS(cublasGemmEx(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K,
                &alpha, dB, CUDA_R_16BF, K,
                dA, CUDA_R_16BF, K,
                &beta, dC, CUDA_R_16BF, N,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= iters;

        double flops = 2.0 * (double)M * (double)K * (double)N;
        double tflops_achieved = flops / (ms * 1e-3) / 1e12;
        double pct_peak = tflops_achieved / l4_bf16_peak_tflops * 100.0;
        double per_layer_ms = ms;
        // gate and up are called once each per layer; same for qkv/o/down.
        // Our megakernel runs gate+up as one fused phase (count both).
        double total_for_16 = per_layer_ms * n_layers;

        printf("%-8s %6d %6d %6d  %8.3f us  %10.2f  %6.1f%%  %10.3f\n",
               shapes[s].name, M, K, N, ms * 1000.0, tflops_achieved, pct_peak, total_for_16);

        total_flops_per_pass += flops * n_layers;
        total_time_ms += per_layer_ms * n_layers;

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Note: 'gate' and 'up' have the same shape but are separate gemms in the fwd pass.
    // The above loop counts both. So total_time_ms = full 16-layer prefill GEMM-only time.
    double total_tflops = total_flops_per_pass / (total_time_ms * 1e-3) / 1e12;
    double pct = total_tflops / l4_bf16_peak_tflops * 100.0;

    printf("\n=== TOTALS for 16-layer LLaMA 1B prefill (GEMMs only) ===\n");
    printf("Total flops:      %.2f Tflops\n", total_flops_per_pass / 1e12);
    printf("Total time:       %.3f ms\n", total_time_ms);
    printf("Achieved:         %.2f TFLOPS (%.1f%% of L4 peak %.1f)\n",
           total_tflops, pct, l4_bf16_peak_tflops);

    cublasDestroy(handle);
    return 0;
}
