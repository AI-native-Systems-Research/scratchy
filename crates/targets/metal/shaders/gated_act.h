// SPDX-License-Identifier: Apache-2.0
//
// The gated activations `act(gate) * up`, in float. The elementwise kernels (`silu_mul.metal`) and
// the MoE gate·up kernel (`quantized_qmv.metal`) both compute them here, so both give the same
// bits.

#pragma once

#include <metal_stdlib>

// A shared expert's combine, `routed + shared · σ(g)` (Qwen3.5-MoE), in float: `gate_scale.metal`
// and the expert combine that stores it (`quantized_qmv.metal`) both compute it here.
inline float gate_scale_f(float routed, float shared, float g) {
  float sig_g = 1.0f / (1.0f + metal::exp(-g));
  return routed + shared * sig_g;
}

// SiLU(g) * u, SiLU(g) = g / (1 + exp(-g)) — kept in float so the denormalized tail of the
// half/bfloat exp() stays representable.
inline float silu_mul_f(float g, float u) {
  float silu_g = g / (1.0f + metal::exp(-g));
  return silu_g * u;
}

// GELU(g) * u, GELU in its tanh approximation (`gelu_approx` in `fused_gate_up_silu_mul.metal`,
// mlx `nn.gelu_approx`): GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 x³))). Float
// throughout (the tanh argument overflows half).
inline float gelu_mul_f(float g, float u) {
  const float sqrt_2_over_pi = 0.7978845608f;
  const float coeff = 0.044715f;
  // Clamp the tanh argument: Metal's fast-math tanh computes
  // (exp(2x)-1)/(exp(2x)+1), which is inf/inf = NaN once 2x
  // overflows exp (|x| ≳ 44 — i.e. ANY gate ≥ ~10.06; Gemma4 layer-0
  // gates reach 57.5). tanh(15) rounds to exactly 1.0f, so the clamp
  // is bit-exact vs a saturating tanh. Same fix as activation.metal.
  float inner = metal::clamp(
      sqrt_2_over_pi * (g + coeff * g * g * g), -15.0f, 15.0f);
  float gelu_g = 0.5f * g * (1.0f + metal::tanh(inner));
  return gelu_g * u;
}
