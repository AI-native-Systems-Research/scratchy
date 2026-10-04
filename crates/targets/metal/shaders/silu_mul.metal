// SPDX-License-Identifier: Apache-2.0
//
// Standalone fused `silu(gate) * up` for the decomposed q-MLP path.
//
// Plan P12 branch (i): when both gate_proj and up_proj are MLX-affine
// quantized, the macro emits the SwiGLU MLP as three instructions:
//   AffineQmm(gate_proj)  -> gate scratch [M, I]
//   AffineQmm(up_proj)    -> up   scratch [M, I]
//   SiluMul(gate, up)     -> out          [M, I]
// instead of the single `FusedGateUpSiluMul` (which assumes Dense
// storage and produces a packed [M, 2I] gate_up via cuBLAS GEMM).
// This is one extra device round-trip per MLP layer vs. the dense
// path; eliminating it would require a hand-rolled fused
// affine_qmm + silu + mul kernel which the
// `feedback_no_handcoded_fusion` rule forbids.
//
// Baked constants:
//   SILU_MUL_N — total output element count (= M * intermediate_size)
//
// Dispatch: 1 thread per output element. Float accumulator on the
// silu so denormalized half/bfloat exp() doesn't flush to zero on
// the negative tail.

#include <metal_stdlib>
#include "baked.h"
#include "gated_act.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, SILU_MUL_N, 0);

template <typename T>
[[kernel]] void silu_mul(
    device       T* out  [[buffer(0)]],
    const device T* gate [[buffer(1)]],
    const device T* up   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= SILU_MUL_N) {
    return;
  }
  out[gid] = static_cast<T>(silu_mul_f(float(gate[gid]), float(up[gid])));
}

#define INST_SILU_MUL(dtype_tag, mtl_type) \
  SCRATCHY_KERNEL(silu_mul_##dtype_tag, silu_mul<mtl_type>)

INST_SILU_MUL(f16,  half)
INST_SILU_MUL(bf16, bfloat)

// GELU (tanh approximation) sibling for the decomposed GeGLU q-MLP
// path (Gemma2/3/4: `gelu_pytorch_tanh(gate) * up`). Same three
// instructions as the SwiGLU decomposition, with `GeluMul` as the
// elementwise tail.
template <typename T>
[[kernel]] void gelu_mul(
    device       T* out  [[buffer(0)]],
    const device T* gate [[buffer(1)]],
    const device T* up   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= SILU_MUL_N) {
    return;
  }
  out[gid] = static_cast<T>(gelu_mul_f(float(gate[gid]), float(up[gid])));
}

#define INST_GELU_MUL(dtype_tag, mtl_type) \
  SCRATCHY_KERNEL(gelu_mul_##dtype_tag, gelu_mul<mtl_type>)

INST_GELU_MUL(f16,  half)
INST_GELU_MUL(bf16, bfloat)
