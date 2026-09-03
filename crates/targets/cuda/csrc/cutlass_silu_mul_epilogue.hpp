// SPDX-License-Identifier: Apache-2.0
// CUTLASS 2.x EVT epilogue for fused Gate GEMM + SiLU + Mul.
//
// Computes: D[i] = silu(accumulator[i]) * aux[i]
//
// Where:
//   accumulator = result of Gate GEMM (A @ W_gate)
//   aux         = Up GEMM output (A @ W_up), loaded from GMEM
//   D           = silu(gate) * up — the MLP activation output
//
// This fuses {GemmGate + GateUpConcat + SiluMul} into a single kernel,
// saving one kernel launch + one full GMEM round-trip of the gate output.

#pragma once

#include <cutlass/epilogue/threadblock/fusion/visitor_2x.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitor_load.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitor_compute.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/epilogue/thread/activation.h>

namespace vllm::c2x {

using namespace cute;

/// EVT epilogue: D = silu(accumulator) * aux_load(up_output)
///
/// Tree structure:
///   Accum → SiLU → ┐
///                    ├→ Multiply → Store
///   AuxLoad(up) → ┘
template <typename ElementD, typename OutputTileThreadMap>
struct SiluMulEpilogue {
 private:
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  // Load the up-projection output from GMEM (same shape as D: [M, N]).
  using AuxLoad = cutlass::epilogue::threadblock::VisitorAuxLoad<
      OutputTileThreadMap, ElementD, cute::Stride<int64_t, cute::Int<1>, int64_t>>;

  // Step 1: Apply SiLU to accumulator → float
  // SiLu is template<class T> struct — matches the template<class> class expected by VisitorCompute.
  using ComputeSilu = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::epilogue::thread::SiLu, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTSilu =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeSilu, Accum>;

  // Step 2: Multiply silu(accum) * aux_load → ElementD
  using ComputeMul = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeMul, AuxLoad, EVTSilu>;
  using ArgumentType = typename EVTCompute::Arguments;

  /// Build EVT arguments from raw pointers.
  /// up_ptr: pointer to the up-projection output [M, N], row-major.
  static ArgumentType prepare_args(ElementD* up_ptr, int N) {
    // AuxLoad arguments: pointer + stride.
    // Row-major [M, N]: stride is (N, 1, 0) for (M, N, L).
    typename AuxLoad::Arguments aux_args{
        up_ptr,
        ElementD(0),
        {(int64_t)N, cute::Int<1>{}, (int64_t)0}};
    // SiLU arguments: empty (no params).
    typename EVTSilu::Arguments silu_args{{}, {}};
    // Top-level: Multiply(AuxLoad, SiLU(Accum))
    return ArgumentType{aux_args, silu_args, {}};
  }
};

}  // namespace vllm::c2x
