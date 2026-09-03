// SPDX-License-Identifier: Apache-2.0
// Custom epilogues for fusing per-row/per-tensor scales onto a CUTLASS GEMM.
// Ported from Python vLLM's scaled_mm_epilogues_c2x.hpp, stripped of PyTorch.
#pragma once

#include "broadcast_load_epilogue_c2x.hpp"

namespace vllm::c2x {

using namespace cute;

/*
 * Common load descriptors for ScaledEpilogue classes.
 */
template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogueBase {
 protected:
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  template <typename T>
  using ColOrScalarLoad =
      cutlass::epilogue::threadblock::VisitorColOrScalarBroadcast<
          OutputTileThreadMap, T, Stride<Int<1>, Int<0>, Int<0>>>;

  template <typename T>
  using RowOrScalarLoad =
      cutlass::epilogue::threadblock::VisitorRowOrScalarBroadcast<
          OutputTileThreadMap, T, Stride<Int<0>, Int<1>, Int<0>>>;

  template <typename T>
  using ColLoad = cutlass::epilogue::threadblock::VisitorColBroadcast<
      OutputTileThreadMap, T, Stride<Int<1>, Int<0>, Int<0>>>;

  template <typename T>
  using RowLoad = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, T, Stride<Int<0>, Int<1>, Int<0>>>;

  template <typename T>
  using RowOrZeroLoad =
      cutlass::epilogue::threadblock::VisitorRowOrZeroBroadcast<
          OutputTileThreadMap, T, Stride<Int<0>, Int<1>, Int<0>>>;
};

/*
 ScaledEpilogue: D = (a_scales * A) (b_scales * B)
 a_scales can be per-token [M] or per-tensor [1].
 b_scales can be per-channel [N] or per-tensor [1].
*/
template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogue
    : private ScaledEpilogueBase<ElementD, OutputTileThreadMap> {
 private:
  using SUPER = ScaledEpilogueBase<ElementD, OutputTileThreadMap>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;

  using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::threadblock::Sm80EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::threadblock::Sm80EVT<Compute1, ScaleA, EVTCompute0>;
  using ArgumentType = typename EVTCompute::Arguments;

  // prepare_args takes raw pointers + numel instead of torch::Tensor.
  // a_scales_ptr: float*, a_scales_numel: 1 (scalar) or M (per-token)
  // b_scales_ptr: float*, b_scales_numel: 1 (scalar) or N (per-channel)
  static ArgumentType prepare_args(
      const float* a_scales_ptr, int a_scales_numel,
      const float* b_scales_ptr, int b_scales_numel) {
    typename ScaleA::Arguments a_args{a_scales_ptr, a_scales_numel != 1};
    typename ScaleB::Arguments b_args{b_scales_ptr, b_scales_numel != 1};
    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, {}};
  }
};

/*
 ScaledEpilogueBias: D = (a_scales * A)(b_scales * B) + bias
 bias is per-output channel [N].
*/
template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogueBias
    : protected ScaledEpilogueBase<ElementD, OutputTileThreadMap> {
 protected:
  using SUPER = ScaledEpilogueBase<ElementD, OutputTileThreadMap>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template RowLoad<ElementD>;
  using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::threadblock::Sm80EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute = cutlass::epilogue::threadblock::Sm80EVT<Compute1, ScaleA,
                                                             EVTCompute0, Bias>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(
      const float* a_scales_ptr, int a_scales_numel,
      const float* b_scales_ptr, int b_scales_numel,
      const ElementD* bias_ptr) {
    typename ScaleA::Arguments a_args{a_scales_ptr, a_scales_numel != 1};
    typename ScaleB::Arguments b_args{b_scales_ptr, b_scales_numel != 1};
    typename Bias::Arguments bias_args{const_cast<ElementD*>(bias_ptr)};
    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, bias_args, {}};
  }
};

}  // namespace vllm::c2x
