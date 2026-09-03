// SPDX-License-Identifier: Apache-2.0
// CUTLASS 2.x BF16 GEMM with bias broadcast.
//
// Computes: D[M, N] = A[M, K] @ B[N, K]^T + bias[N]
//
// Built on the same template family as `cutlass_standalone_gemm.cu`:
// plain `cutlass::gemm::device::Gemm` + `LinearCombination` epilogue.
// The bias is passed as the C operand at stride 0 (RowMajor with
// ldc=0), so every M-row indexes the same bias[0..N] — built-in row
// broadcast via the standard GEMM API, no EVT visitor tree.
//
// One launcher per (TB_M, TB_N, STAGES) tuple in CUTLASS_TILE_ZOO.
// The DP solver picks per (workload M, weight N, K) from the
// calibrated CSV; the macro-emitted forward calls the matching
// launch fn directly via `launch_fn_for_bias`.

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cuda_runtime.h>

template <typename GemmOp>
static int run_gemm_bias(
    void* d, const void* a, const void* b, const void* bias,
    int M, int N, int K,
    cudaStream_t stream
) {
    using BF16 = cutlass::bfloat16_t;
    typename GemmOp::Arguments args(
        {M, N, K},
        {(BF16 const*)a, K},        // A [M, K] row-major,  lda=K
        {(BF16 const*)b, K},        // B [N, K] col-major,  ldb=K
        {(BF16 const*)bias, 0},     // C row-major, ldc=0 → broadcast bias[N]
        {(BF16*)d, N},              // D [M, N] row-major,  ldd=N
        {1.0f, 1.0f}                // alpha=1, beta=1 → D = A@B^T + bias
    );

    GemmOp op;
    auto status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    status = op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) return -2;

    status = op(stream);
    return (status == cutlass::Status::kSuccess) ? 0 : -3;
}

#define BIAS_GEMM_CONFIG(TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)        \
    using GemmBias_##TB_M##x##TB_N##x##TB_K##_s##STAGES = cutlass::gemm::device::Gemm< \
        cutlass::bfloat16_t, cutlass::layout::RowMajor,                           \
        cutlass::bfloat16_t, cutlass::layout::ColumnMajor,                        \
        cutlass::bfloat16_t, cutlass::layout::RowMajor,                           \
        float,                                                                    \
        cutlass::arch::OpClassTensorOp,                                           \
        cutlass::arch::Sm80,                                                      \
        cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>,                               \
        cutlass::gemm::GemmShape<WARP_M, WARP_N, WARP_K>,                         \
        cutlass::gemm::GemmShape<16, 8, 16>,                                      \
        cutlass::epilogue::thread::LinearCombination<                             \
            cutlass::bfloat16_t, 8, float, float>,                                \
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,             \
        STAGES                                                                    \
    >;

#define BIAS_GEMM_LAUNCH(TB_M, TB_N, TB_K, STAGES)                                \
    extern "C" int cutlass_gemm_bias_##TB_M##x##TB_N##_s##STAGES##_launch(        \
        void* d, const void* a, const void* b, const void* bias,                  \
        int M, int N, int K,                                                      \
        uint64_t stream                                                           \
    ) {                                                                           \
        return run_gemm_bias<GemmBias_##TB_M##x##TB_N##x##TB_K##_s##STAGES>(      \
            d, a, b, bias, M, N, K, (cudaStream_t)stream);                        \
    }

#define BIAS_GEMM(TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)               \
    BIAS_GEMM_CONFIG(TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)            \
    BIAS_GEMM_LAUNCH(TB_M, TB_N, TB_K, STAGES)

// Tile zoo — must match `CUTLASS_TILE_ZOO` in
// scratchy-forward-compiler-macro/src/impl_lib.rs and the extern-decl block in
// scratchy-target-cuda/src/cutlass.rs. One row per zoo entry.

// TB_M  TB_N  TB_K  WARP_M  WARP_N  WARP_K  STAGES
BIAS_GEMM( 16,   64,  32,    16,     32,     32,    3)
BIAS_GEMM( 16,   64,  32,    16,     32,     32,    4)
BIAS_GEMM( 16,  128,  32,    16,     64,     32,    3)
BIAS_GEMM( 16,  128,  32,    16,     64,     32,    4)
BIAS_GEMM( 32,   64,  32,    32,     32,     32,    3)
BIAS_GEMM( 32,   64,  32,    32,     32,     32,    4)
BIAS_GEMM( 32,  128,  32,    32,     64,     32,    3)
BIAS_GEMM( 32,  128,  32,    32,     64,     32,    4)
BIAS_GEMM( 32,  256,  32,    32,     64,     32,    3)
BIAS_GEMM( 64,   64,  32,    32,     32,     32,    3)
BIAS_GEMM( 64,   64,  32,    32,     32,     32,    4)
BIAS_GEMM( 64,  128,  32,    32,     64,     32,    3)
BIAS_GEMM( 64,  128,  32,    32,     64,     32,    4)
BIAS_GEMM(128,   64,  32,    64,     32,     32,    3)
BIAS_GEMM(128,   64,  32,    64,     32,     32,    4)
BIAS_GEMM(128,  128,  32,    64,     32,     32,    3)
BIAS_GEMM(128,  128,  32,    64,     32,     32,    4)
BIAS_GEMM(128,  256,  32,    64,     64,     32,    3)
BIAS_GEMM(256,   64,  32,    64,     32,     32,    3)
BIAS_GEMM(256,   64,  32,    64,     32,     32,    4)
