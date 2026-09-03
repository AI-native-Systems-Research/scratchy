// SPDX-License-Identifier: Apache-2.0
// CUTLASS 2.x EVT epilogue for fused GEMM + bias broadcast add.
//
// Computes: D[m, n] = accumulator[m, n] + bias[n]
//
// Where bias is a `[N]` vector broadcast across the M dimension.
// One kernel launch, one `[M, N]` write, no intermediate delta tensor.
//
// Peer to `cutlass_silu_mul_epilogue.hpp`; pulls in the same EVT
// primitives from scaled_mm_c2x.cuh so it rides the existing CUTLASS
// build infrastructure.

#pragma once

#include <cutlass/epilogue/threadblock/fusion/visitor_2x.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitor_load.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitor_compute.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>

namespace vllm::c2x {

using namespace cute;

/// EVT epilogue: D = accumulator + row_broadcast(bias)
///
/// Tree structure:
///   Accum             → ┐
///                         ├→ Add → Store
///   RowBroadcast(bias) → ┘
template <typename ElementD, typename OutputTileThreadMap>
struct BiasAddEpilogue {
 private:
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  // Row broadcast: bias is [N] with stride (0, 1, 0) broadcasting
  // identical values across the M axis.
  using BiasLoad = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, ElementD,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using ComputeAdd = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeAdd, Accum, BiasLoad>;
  using ArgumentType = typename EVTCompute::Arguments;

  /// Build EVT arguments from a raw bias pointer.
  /// bias_ptr: pointer to the per-N bias vector (length N, bf16).
  static ArgumentType prepare_args(const ElementD* bias_ptr) {
    // RowBroadcast arguments: pointer + null-indicator + stride.
    typename BiasLoad::Arguments bias_args{
        bias_ptr,
        ElementD(0),
        {cute::Int<0>{}, cute::Int<1>{}, cute::Int<0>{}}};
    // Top-level: Add(Accum, RowBroadcast(bias))
    return ArgumentType{{}, bias_args, {}};
  }
};

}  // namespace vllm::c2x
