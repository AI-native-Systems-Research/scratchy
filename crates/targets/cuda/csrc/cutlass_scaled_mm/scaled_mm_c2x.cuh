// SPDX-License-Identifier: Apache-2.0
// CUTLASS 2.x GEMM template for scaled matmul with fused epilogue.
// Ported from Python vLLM, stripped of PyTorch dependencies.
#pragma once

#include <stddef.h>

// clang-format off
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"

#include "math.hpp"
#include "common.hpp"
// clang-format on

using namespace cute;

/*
   Epilogues defined in scaled_mm_epilogues_c2x.hpp
   must contain a public type named EVTCompute of type Sm80EVT,
   as well as a static prepare_args function.
*/

namespace vllm {

template <typename Arch, template <typename> typename ArchGuard,
          typename ElementAB_, typename ElementD_,
          template <typename, typename> typename Epilogue_, typename TileShape,
          typename WarpShape, typename InstructionShape, int32_t MainLoopStages,
          typename FP8MathOperator = cutlass::arch::OpMultiplyAdd>
struct cutlass_2x_gemm {
  using ElementAB = ElementAB_;
  using ElementD = ElementD_;

  using ElementAcc =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>, int32_t,
                                float>::type;

  using Operator =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>,
                                cutlass::arch::OpMultiplyAddSaturate,
                                FP8MathOperator>::type;

  using OutputTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          TileShape, WarpShape, float, 4, 1 /* epilogue stages */
          >;

  using Epilogue = Epilogue_<ElementD, OutputTileThreadMap>;
  using EVTCompute = typename Epilogue::EVTCompute;

  using D = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementD, cutlass::FloatRoundStyle::round_to_nearest,
      Stride<int64_t, Int<1>, Int<0>>>;

  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<D, EVTCompute>;

  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD = 4;

  // clang-format off
  using RowMajor = typename cutlass::layout::RowMajor;
  using ColumnMajor = typename cutlass::layout::ColumnMajor;
  using KernelType =
    ArchGuard<typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementAB, RowMajor, cutlass::ComplexTransform::kNone, AlignmentAB,
      ElementAB, ColumnMajor, cutlass::ComplexTransform::kNone, AlignmentAB,
      float, cutlass::layout::RowMajor, AlignmentCD,
      ElementAcc, float, cutlass::arch::OpClassTensorOp,
      Arch,
      TileShape, WarpShape, InstructionShape,
      EVTD,
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
      MainLoopStages, Operator,
      1 /* epilogue stages */
      >::GemmKernel>;
  // clang-format on

  using Op = cutlass::gemm::device::GemmUniversalAdapter<KernelType>;
};

// Raw-pointer version of cutlass_gemm_caller (no torch::Tensor).
// All tensor metadata (pointers, shapes, strides) passed explicitly.
template <typename Gemm, typename... EpilogueArgs>
inline void cutlass_gemm_caller(
    void* c_ptr,                  // output [M, N]
    const void* a_ptr,            // [M, K] row-major
    const void* b_ptr,            // [K, N] column-major (= [N, K] row-major)
    int32_t m, int32_t n, int32_t k,
    int64_t lda,                  // A row stride (= K for contiguous)
    int64_t ldb,                  // B column stride (= K for [N,K] row-major)
    int64_t ldc,                  // C row stride (= N for contiguous)
    cudaStream_t stream,
    EpilogueArgs&&... epilogue_params) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  cutlass::gemm::GemmCoord problem_size{m, n, k};

  using StrideC = Stride<int64_t, Int<1>, Int<0>>;
  StrideC c_stride{ldc, Int<1>{}, Int<0>{}};

  auto a = static_cast<ElementAB const*>(a_ptr);
  auto b = static_cast<ElementAB const*>(b_ptr);
  auto c = static_cast<ElementD*>(c_ptr);

  typename Gemm::D::Arguments d_args{c, c_stride};

  using Epilogue = typename Gemm::Epilogue;
  auto evt_args =
      Epilogue::prepare_args(std::forward<EpilogueArgs>(epilogue_params)...);

  typename Gemm::EVTD::Arguments epilogue_args{
      evt_args,
      d_args,
  };

  typename Gemm::Op::Arguments args{
      // Match vllm csrc/quantization/w8a8/cutlass/scaled_mm_c2x.cuh.
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
      problem_size,
      1,
      epilogue_args,
      a,
      b,
      nullptr,
      nullptr,
      0,
      0,
      0,
      0,
      lda,
      ldb,
      ldc,
      ldc};

  typename Gemm::Op gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(args);

  // Use cudaMallocAsync for graph-capture compatibility
  void* workspace = nullptr;
  if (workspace_size > 0) {
    cudaError_t malloc_err = cudaMallocAsync(&workspace, workspace_size, stream);
    if (malloc_err != cudaSuccess) {
      fprintf(stderr, "FATAL: cudaMallocAsync failed: %s (size=%zu)\n",
              cudaGetErrorString(malloc_err), workspace_size);
      abort();
    }
  }

  CUTLASS_CHECK(gemm_op.can_implement(args));
  cutlass::Status status = gemm_op(args, workspace, stream);
  CUTLASS_CHECK(status);

  if (workspace != nullptr) {
    cudaFreeAsync(workspace, stream);
  }
}

// Fallback version that checks shared memory and falls back to smaller tile.
template <typename Gemm, typename FallbackGemm, typename... EpilogueArgs>
inline void fallback_cutlass_gemm_caller(
    void* c_ptr, const void* a_ptr, const void* b_ptr,
    int32_t m, int32_t n, int32_t k,
    int64_t lda, int64_t ldb, int64_t ldc,
    cudaStream_t stream,
    EpilogueArgs&&... args) {
  static const int max_shared_mem_per_block_opt_in =
      get_cuda_max_shared_memory_per_block_opt_in(0);

  size_t const gemm_shared_mem_size =
      sizeof(typename Gemm::KernelType::SharedStorage);
  size_t const fallback_gemm_shared_mem_size =
      sizeof(typename FallbackGemm::KernelType::SharedStorage);

  if (gemm_shared_mem_size <= max_shared_mem_per_block_opt_in) {
    return cutlass_gemm_caller<Gemm>(
        c_ptr, a_ptr, b_ptr, m, n, k, lda, ldb, ldc, stream,
        std::forward<EpilogueArgs>(args)...);
  } else {
    // Fallback: use smaller tile configuration
    if (fallback_gemm_shared_mem_size > max_shared_mem_per_block_opt_in) {
      fprintf(stderr, "CUTLASS: fallback GEMM also exceeds shared memory\n");
      abort();
    }
    return cutlass_gemm_caller<FallbackGemm>(
        c_ptr, a_ptr, b_ptr, m, n, k, lda, ldb, ldc, stream,
        std::forward<EpilogueArgs>(args)...);
  }
}

}  // namespace vllm
