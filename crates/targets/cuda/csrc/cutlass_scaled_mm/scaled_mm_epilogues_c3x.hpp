// SPDX-License-Identifier: Apache-2.0
// CUTLASS 3.x (Hopper / SM90) scale epilogues for fused FP8 scaled_mm.
// Ported from Python vLLM's
//   csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp
// stripped of all torch::Tensor dependencies: scales/bias are passed as raw
// device pointers + element counts instead of tensors.
//
// Only the descriptors the FP8 SM90 dispatch needs are kept:
//   ScaledEpilogue        D = (a_scales * A)(b_scales * B)
//   ScaledEpilogueBias    D = (a_scales * A)(b_scales * B) + bias   (row bias)
//   ScaledEpilogueColumnBias  same, but bias is a column vector (swap_ab path)
// The azp / array (grouped) epilogues are intentionally omitted.
#pragma once

#include "broadcast_load_epilogue_c3x.hpp"

/*
   Epilogues must contain a public type named EVTCompute of type Sm90EVT, as
   well as a static prepare_args function that constructs an
   EVTCompute::Arguments struct. Here prepare_args takes raw device pointers
   and element counts rather than torch tensors.
*/

namespace vllm::c3x {

using namespace cute;

/*
 * Common load descriptors for the ScaledEpilogue[...] classes.
 */
template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogueBase {
 protected:
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  template <typename T>
  using ColOrScalarLoad = cutlass::epilogue::fusion::Sm90ColOrScalarBroadcast<
      0 /*Stages*/, TileShape, T, Stride<Int<1>, Int<0>, Int<0>>>;

  template <typename T>
  using RowOrScalarLoad = cutlass::epilogue::fusion::Sm90RowOrScalarBroadcast<
      0 /*Stages*/, TileShape, T, Stride<Int<0>, Int<1>, Int<0>>>;

  template <typename T, bool EnableNullPtr = false>
  using ColLoad = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0 /*Stages*/, TileShape, T, T, Stride<Int<1>, Int<0>, Int<0>>,
      128 / sizeof_bits_v<T>, EnableNullPtr>;

  template <typename T, bool EnableNullPtr = false>
  using RowLoad = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0 /*Stages*/, TileShape, T, T, Stride<Int<0>, Int<1>, Int<0>>,
      128 / sizeof_bits_v<T>, EnableNullPtr>;

  // Build load-descriptor Arguments from a raw pointer + element count.
  // A `numel == 1` scale is a scalar broadcast; otherwise it is a per-row /
  // per-column vector. Matches torch path's `tensor.numel() != 1` flag.
  template <typename Descriptor, typename T>
  static auto args_from_ptr(const T* data_ptr, int numel) {
    using Arguments = typename Descriptor::Arguments;
    if constexpr (std::is_same_v<Descriptor, ColOrScalarLoad<T>> ||
                  std::is_same_v<Descriptor, RowOrScalarLoad<T>>) {
      return Arguments{const_cast<T*>(data_ptr), numel != 1};
    } else {
      static_assert(!std::is_same_v<Descriptor, ColLoad<T, true>> &&
                    !std::is_same_v<Descriptor, RowLoad<T, true>>);
      return Arguments{const_cast<T*>(data_ptr)};
    }
  }
};

/*
   D = (a_scales * A) (b_scales * B)
   a_scales: per-token [M] or per-tensor [1]; b_scales: per-channel [N] or [1].
*/
template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogue
    : private ScaledEpilogueBase<ElementAcc, ElementD, TileShape> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, TileShape>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(const float* a_scales_ptr, int a_scales_numel,
                                   const float* b_scales_ptr,
                                   int b_scales_numel) {
    auto a_args =
        SUPER::template args_from_ptr<ScaleA, float>(a_scales_ptr, a_scales_numel);
    auto b_args =
        SUPER::template args_from_ptr<ScaleB, float>(b_scales_ptr, b_scales_numel);

    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, {}};
  }
};

/*
 * D = (a_scales * A)(b_scales * B) + bias, bias per-output-channel (row).
 */
template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogueBias
    : private ScaledEpilogueBase<ElementAcc, ElementD, TileShape> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, TileShape>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template RowLoad<ElementD>;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0, Bias>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(const float* a_scales_ptr, int a_scales_numel,
                                   const float* b_scales_ptr, int b_scales_numel,
                                   const ElementD* bias_ptr) {
    auto a_args =
        SUPER::template args_from_ptr<ScaleA, float>(a_scales_ptr, a_scales_numel);
    auto b_args =
        SUPER::template args_from_ptr<ScaleB, float>(b_scales_ptr, b_scales_numel);
    auto bias_args = SUPER::template args_from_ptr<Bias, ElementD>(bias_ptr, 0);

    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, bias_args, {}};
  }
};

/*
 * Same as ScaledEpilogueBias but bias is a column vector instead of a row
 * vector. Used by the swap_ab (small-M) configs which compute C^T = B^T A^T.
 */
template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogueColumnBias
    : private ScaledEpilogueBase<ElementAcc, ElementD, TileShape> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, TileShape>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;
  using Bias = typename SUPER::template ColLoad<ElementD>;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::homogeneous_multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0, Bias>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(const float* a_scales_ptr, int a_scales_numel,
                                   const float* b_scales_ptr, int b_scales_numel,
                                   const ElementD* bias_ptr) {
    auto a_args =
        SUPER::template args_from_ptr<ScaleA, float>(a_scales_ptr, a_scales_numel);
    auto b_args =
        SUPER::template args_from_ptr<ScaleB, float>(b_scales_ptr, b_scales_numel);
    auto bias_args = SUPER::template args_from_ptr<Bias, ElementD>(bias_ptr, 0);

    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, bias_args, {}};
  }
};

}  // namespace vllm::c3x
