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
// Function constants:
//   SILU_MUL_N — total output element count (= M * intermediate_size)
//
// Dispatch: 1 thread per output element. Float accumulator on the
// silu so denormalized half/bfloat exp() doesn't flush to zero on
// the negative tail.

#include <metal_stdlib>

using namespace metal;

constant uint SILU_MUL_N [[function_constant(0)]];

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
  // SiLU(x) = x / (1 + exp(-x)) — kept in float so the denormalized
  // tail of the half/bfloat exp() stays representable.
  float g = float(gate[gid]);
  float u = float(up[gid]);
  float silu_g = g / (1.0f + exp(-g));
  out[gid] = static_cast<T>(silu_g * u);
}

#define INST_SILU_MUL(dtype_tag, mtl_type)                                \
  template [[host_name("silu_mul_" #dtype_tag)]] [[kernel]] void          \
  silu_mul<mtl_type>(                                                     \
      device       mtl_type* out  [[buffer(0)]],                          \
      const device mtl_type* gate [[buffer(1)]],                          \
      const device mtl_type* up   [[buffer(2)]],                          \
      uint gid [[thread_position_in_grid]]);

INST_SILU_MUL(f16,  half)
INST_SILU_MUL(bf16, bfloat)

// GELU (tanh approximation) sibling for the decomposed GeGLU q-MLP
// path (Gemma2/3/4: `gelu_pytorch_tanh(gate) * up`). Same three
// instructions as the SwiGLU decomposition, with `GeluMul` as the
// elementwise tail. Formula matches `gelu_approx` in
// `fused_gate_up_silu_mul.metal` and mlx `nn.gelu_approx`:
//   GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 x³)))
// Float accumulator throughout (the tanh argument overflows half).
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
  float g = float(gate[gid]);
  float u = float(up[gid]);
  const float sqrt_2_over_pi = 0.7978845608f;
  const float coeff = 0.044715f;
  // Clamp the tanh argument: Metal's fast-math tanh computes
  // (exp(2x)-1)/(exp(2x)+1), which is inf/inf = NaN once 2x
  // overflows exp (|x| ≳ 44 — i.e. ANY gate ≥ ~10.06; Gemma4 layer-0
  // gates reach 57.5). tanh(15) rounds to exactly 1.0f, so the clamp
  // is bit-exact vs a saturating tanh. Same fix as activation.metal.
  float inner = clamp(
      sqrt_2_over_pi * (g + coeff * g * g * g), -15.0f, 15.0f);
  float gelu_g = 0.5f * g * (1.0f + tanh(inner));
  out[gid] = static_cast<T>(gelu_g * u);
}

#define INST_GELU_MUL(dtype_tag, mtl_type)                                \
  template [[host_name("gelu_mul_" #dtype_tag)]] [[kernel]] void          \
  gelu_mul<mtl_type>(                                                     \
      device       mtl_type* out  [[buffer(0)]],                          \
      const device mtl_type* gate [[buffer(1)]],                          \
      const device mtl_type* up   [[buffer(2)]],                          \
      uint gid [[thread_position_in_grid]]);

INST_GELU_MUL(f16,  half)
INST_GELU_MUL(bf16, bfloat)
