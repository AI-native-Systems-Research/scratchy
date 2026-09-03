// SPDX-License-Identifier: Apache-2.0
// SM90 (Hopper) FP8 CUTLASS 3.x GEMM configurations + dispatch.
// Ported from Python vLLM's
//   csrc/.../w8a8/cutlass/c3x/scaled_mm_sm90_fp8_dispatch.cuh
// stripped of PyTorch / torch::stable dependencies: tensors become raw device
// pointers + (m, n, k) + element counts, and the workspace / stream are driven
// by cudaMallocAsync + a cudaStream_t (matching the C2X SM89 port here).
//
// The tile-config structs and the per-(M,N) dispatch ladder are a faithful copy
// of upstream; only the calling convention differs.
#pragma once

// clang-format off
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "common.hpp"
#include "scaled_mm_epilogues_c3x.hpp"
// clang-format on

using namespace cute;

namespace vllm {

// ---------------------------------------------------------------------------
// Raw-pointer GemmKernel-level caller (no torch::Tensor / torch::stable).
// Allocates the CUTLASS workspace with cudaMallocAsync for graph-capture
// compatibility and runs on the supplied stream.
// ---------------------------------------------------------------------------
namespace c3x {

template <typename GemmKernel>
void cutlass_gemm_caller(cute::Shape<int, int, int, int> prob_shape,
                         typename GemmKernel::MainloopArguments mainloop_args,
                         typename GemmKernel::EpilogueArguments epilogue_args,
                         cudaStream_t stream,
                         typename GemmKernel::TileSchedulerArguments scheduler =
                             {}) {
  cutlass::KernelHardwareInfo hw_info;
  typename GemmKernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                      prob_shape,
                                      mainloop_args,
                                      epilogue_args,
                                      hw_info,
                                      scheduler};

  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));

  size_t workspace_size = gemm_op.get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_size > 0) {
    cudaError_t malloc_err = cudaMallocAsync(&workspace, workspace_size, stream);
    if (malloc_err != cudaSuccess) {
      fprintf(stderr, "FATAL: cudaMallocAsync failed: %s (size=%zu)\n",
              cudaGetErrorString(malloc_err), workspace_size);
      abort();
    }
  }

  cutlass::Status status = gemm_op.run(args, workspace, stream);
  CUTLASS_CHECK(status);

  if (workspace != nullptr) {
    cudaFreeAsync(workspace, stream);
  }
}

}  // namespace c3x

// ---------------------------------------------------------------------------
// cutlass_3x_gemm_sm90_fp8 — verbatim from upstream (swap_ab supported).
// ---------------------------------------------------------------------------
template <typename ElementAB_, typename ElementD_,
          template <typename, typename, typename> typename Epilogue_,
          typename TileShape, typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule, bool swap_ab_ = false>
struct cutlass_3x_gemm_sm90_fp8 {
  using ElementAB = ElementAB_;
  using ElementC = ElementD_;
  using ElementD = ElementD_;
  using ElementAcc =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>, int32_t,
                                float>::type;

  using Epilogue = Epilogue_<ElementAcc, ElementD, TileShape>;

  using EVTCompute = typename Epilogue::EVTCompute;

  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD =
      128 / cutlass::sizeof_bits<ElementD>::value;

  // Compile-time swap_ab flag
  static constexpr bool swap_ab = swap_ab_;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutA_T = typename cutlass::layout::LayoutTranspose<LayoutA>::type;

  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutB_T = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

  using LayoutD = cutlass::layout::RowMajor;
  using LayoutD_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutD>::type;

  using LayoutC = LayoutD;
  using LayoutC_Transpose = LayoutD_Transpose;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAcc, float, ElementC,
          conditional_t<swap_ab, LayoutC_Transpose, LayoutC>, AlignmentCD,
          ElementD, conditional_t<swap_ab, LayoutD_Transpose, LayoutD>,
          AlignmentCD, EpilogueSchedule, EVTCompute>::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);

  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(CEStorageSize)>;

  using CollectiveMainloop = conditional_t<
      swap_ab,
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementAB,
          LayoutB_T, AlignmentAB,             // Swapped B (as A)
          ElementAB, LayoutA_T, AlignmentAB,  // Swapped A (as B)
          ElementAcc, TileShape, ClusterShape, Stages,
          KernelSchedule>::CollectiveOp,
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementAB,
          LayoutA, AlignmentAB, ElementAB, LayoutB, AlignmentAB, ElementAcc,
          TileShape, ClusterShape, Stages, KernelSchedule>::CollectiveOp>;

  using KernelType = enable_sm90_or_later<cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>>;

  struct GemmKernel : public KernelType {};
};

// ---------------------------------------------------------------------------
// Per-(M,N) tile configurations — verbatim from upstream.
// ---------------------------------------------------------------------------
template <typename InType, typename OutType, bool EnableBias>
struct sm90_fp8_config_default {
  // M in (128, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;

  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogueBias,
                               TileShape, ClusterShape, KernelSchedule,
                               EpilogueSchedule>,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                               ClusterShape, KernelSchedule, EpilogueSchedule>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm90_fp8_config_M8192_K6144 {
  // M >= 8192, K >= 6144
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
  using EpilogueSchedule =
      typename cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileShape = Shape<_256, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;

  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogueBias,
                               TileShape, ClusterShape, KernelSchedule,
                               EpilogueSchedule>,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                               ClusterShape, KernelSchedule, EpilogueSchedule>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm90_fp8_config_M128 {
  // M in (64, 128]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogueBias,
                               TileShape, ClusterShape, KernelSchedule,
                               EpilogueSchedule>,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                               ClusterShape, KernelSchedule, EpilogueSchedule>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm90_fp8_config_M64_N1280 {
  // M in (16, 64], N in [1 1280]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _16, _256>;
  using ClusterShape = Shape<_1, _4, _1>;

  // enable swap AB for M < 64
  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogueColumnBias,
                               TileShape, ClusterShape, KernelSchedule,
                               EpilogueSchedule, true>,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                               ClusterShape, KernelSchedule, EpilogueSchedule,
                               true>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm90_fp8_config_M64_N8192 {
  // M in (16, 64], N > 1280
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _64, _256>;
  using ClusterShape = Shape<_1, _1, _1>;

  // enable swap AB for M < 64
  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogueColumnBias,
                               TileShape, ClusterShape, KernelSchedule,
                               EpilogueSchedule, true>,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                               ClusterShape, KernelSchedule, EpilogueSchedule,
                               true>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm90_fp8_config_M16_N1280 {
  // M in [1, 16], N in [1, 1280]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _16, _256>;
  using ClusterShape = Shape<_1, _2, _1>;

  // enable swap AB for M < 64
  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogueColumnBias,
                               TileShape, ClusterShape, KernelSchedule,
                               EpilogueSchedule, true>,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                               ClusterShape, KernelSchedule, EpilogueSchedule,
                               true>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm90_fp8_config_M16_N8192 {
  // M in [1, 16], N > 1280
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _16, _256>;
  using ClusterShape = Shape<_1, _1, _1>;

  // enable swap AB for M < 64
  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogueColumnBias,
                               TileShape, ClusterShape, KernelSchedule,
                               EpilogueSchedule, true>,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                               ClusterShape, KernelSchedule, EpilogueSchedule,
                               true>>;
};

// ---------------------------------------------------------------------------
// Raw-pointer GEMM caller for a chosen config. Builds packed strides from
// (m, n, k) — all operands are contiguous (A:[M,K] row-major, B:[N,K]
// row-major == [K,N] col-major, D:[M,N] row-major).
// EpilogueArgs carries the scale pointers/counts (+ optional bias pointer),
// forwarded to Gemm::Epilogue::prepare_args by the dispatch below.
// ---------------------------------------------------------------------------
template <typename Gemm, typename... EpilogueArgs>
void cutlass_gemm_caller_sm90_fp8(void* out, const void* a, const void* b,
                                  int32_t m, int32_t n, int32_t k,
                                  cudaStream_t stream,
                                  EpilogueArgs&&... epilogue_params) {
  static constexpr bool swap_ab = Gemm::swap_ab;
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;
  using GemmKernel = typename Gemm::GemmKernel;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;

  auto prob_shape =
      swap_ab ? cute::make_shape(n, m, k, 1) : cute::make_shape(m, n, k, 1);

  StrideA a_stride =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB b_stride =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC c_stride = cutlass::make_cute_packed_stride(
      StrideC{},
      swap_ab ? cute::make_shape(n, m, 1) : cute::make_shape(m, n, 1));

  auto a_ptr = static_cast<ElementAB*>(const_cast<void*>(a));
  auto b_ptr = static_cast<ElementAB*>(const_cast<void*>(b));
  auto c_ptr = static_cast<ElementD*>(out);

  typename GemmKernel::MainloopArguments mainloop_args =
      swap_ab ? typename GemmKernel::MainloopArguments{b_ptr, b_stride, a_ptr,
                                                       a_stride}
              : typename GemmKernel::MainloopArguments{a_ptr, a_stride, b_ptr,
                                                       b_stride};

  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(
          std::forward<EpilogueArgs>(epilogue_params)...),
      c_ptr, c_stride, c_ptr, c_stride};

  c3x::cutlass_gemm_caller<GemmKernel>(prob_shape, mainloop_args, epilogue_args,
                                       stream);
}

// ---------------------------------------------------------------------------
// Per-(M,N) dispatch. Mirrors upstream exactly, including the swapped scale
// argument order for the M<=64 (swap_ab) configs.
// `bias_args...` is either empty (ScaledEpilogue) or a single ElementD* bias
// pointer (ScaledEpilogue{,Column}Bias).
// ---------------------------------------------------------------------------
template <typename InType, typename OutType, bool EnableBias,
          typename... BiasArgs>
inline void cutlass_gemm_sm90_fp8_dispatch(void* out, const void* a,
                                           const void* b, int32_t m, int32_t n,
                                           int32_t k, cudaStream_t stream,
                                           const float* a_scales,
                                           int a_scales_numel,
                                           const float* b_scales,
                                           int b_scales_numel,
                                           BiasArgs&&... bias_args) {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());

  using Cutlass3xGemmDefault =
      typename sm90_fp8_config_default<InType, OutType,
                                       EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM8192_K6144 =
      typename sm90_fp8_config_M8192_K6144<InType, OutType,
                                           EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM128 =
      typename sm90_fp8_config_M128<InType, OutType, EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM64_N1280 =
      typename sm90_fp8_config_M64_N1280<InType, OutType,
                                         EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM64_N8192 =
      typename sm90_fp8_config_M64_N8192<InType, OutType,
                                         EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM16_N1280 =
      typename sm90_fp8_config_M16_N1280<InType, OutType,
                                         EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM16_N8192 =
      typename sm90_fp8_config_M16_N8192<InType, OutType,
                                         EnableBias>::Cutlass3xGemm;

  if (m <= 16) {
    // m in [1, 16] — swap_ab configs take (b_scales, a_scales) order.
    if (n <= 1280) {
      return cutlass_gemm_caller_sm90_fp8<Cutlass3xGemmM16_N1280>(
          out, a, b, m, n, k, stream, b_scales, b_scales_numel, a_scales,
          a_scales_numel, std::forward<BiasArgs>(bias_args)...);
    }
    return cutlass_gemm_caller_sm90_fp8<Cutlass3xGemmM16_N8192>(
        out, a, b, m, n, k, stream, b_scales, b_scales_numel, a_scales,
        a_scales_numel, std::forward<BiasArgs>(bias_args)...);
  } else if (m <= 64) {
    // m in (16, 64] — swap_ab configs take (b_scales, a_scales) order.
    if (n <= 1280) {
      return cutlass_gemm_caller_sm90_fp8<Cutlass3xGemmM64_N1280>(
          out, a, b, m, n, k, stream, b_scales, b_scales_numel, a_scales,
          a_scales_numel, std::forward<BiasArgs>(bias_args)...);
    }
    return cutlass_gemm_caller_sm90_fp8<Cutlass3xGemmM64_N8192>(
        out, a, b, m, n, k, stream, b_scales, b_scales_numel, a_scales,
        a_scales_numel, std::forward<BiasArgs>(bias_args)...);
  } else if (m <= 128) {
    // m in (64, 128]
    return cutlass_gemm_caller_sm90_fp8<Cutlass3xGemmM128>(
        out, a, b, m, n, k, stream, a_scales, a_scales_numel, b_scales,
        b_scales_numel, std::forward<BiasArgs>(bias_args)...);
  } else if (m >= 8192 && k >= 6144) {
    return cutlass_gemm_caller_sm90_fp8<Cutlass3xGemmM8192_K6144>(
        out, a, b, m, n, k, stream, a_scales, a_scales_numel, b_scales,
        b_scales_numel, std::forward<BiasArgs>(bias_args)...);
  } else {
    // m in (128, inf)
    return cutlass_gemm_caller_sm90_fp8<Cutlass3xGemmDefault>(
        out, a, b, m, n, k, stream, a_scales, a_scales_numel, b_scales,
        b_scales_numel, std::forward<BiasArgs>(bias_args)...);
  }
}

}  // namespace vllm
