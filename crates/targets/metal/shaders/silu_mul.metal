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
//   SILU_MUL_ROW, SILU_MUL_EXPERTS — set by a grouped MoE bake: its padded
//     rows of SILU_MUL_ROW elements, a row whose expert in `row_expert` is
//     not below SILU_MUL_EXPERTS (the sort's sentinel) holding no pair and
//     skipped
//
// Dispatch: 1 thread per output element. Float accumulator on the
// silu so denormalized half/bfloat exp() doesn't flush to zero on
// the negative tail.

#include <metal_stdlib>
#include "baked.h"
#include "gated_act.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, SILU_MUL_N, 0);
SCRATCHY_CONSTANT_OPTIONAL(uint, SILU_MUL_ROW, 1);
SCRATCHY_CONSTANT_OPTIONAL(uint, SILU_MUL_EXPERTS, 2);
// 3 (`SILU_MUL_LINEAR_BIAS`): gpt-oss's grouped bake — the pair's expert adds its gate and up
// LINEAR biases before the activation, the decomposed sibling of the gathered
// `affine_gather_qmv_gated` epilogue (buffer(14) there). Unset: no buffer(4), and the adds
// const-fold away.
SCRATCHY_CONSTANT_OPTIONAL(bool, SILU_MUL_LINEAR_BIAS_FC, 3);
constant constexpr bool SILU_MUL_LINEAR_BIAS =
    SILU_MUL_LINEAR_BIAS_FC_SET && SILU_MUL_LINEAR_BIAS_FC;
// 4 (`SILU_MUL_SWIGLU_LIMIT`): SwiGLU-OAI's clamp, the model's `swiglu_limit` — set exactly on
// the `swiglu_oai_mul_*` kernels.
SCRATCHY_CONSTANT_OPTIONAL(float, SILU_MUL_SWIGLU_LIMIT, 4);
static_assert((SCRATCHY_COMPILES(swiglu_oai_mul_f16) || SCRATCHY_COMPILES(swiglu_oai_mul_bf16)) ==
                  SILU_MUL_SWIGLU_LIMIT_SET,
              "SwiGLU-OAI takes its limit compiled in");

inline bool silu_mul_skips(const device uint* row_expert, uint gid) {
  return gid >= SILU_MUL_N ||
      (SILU_MUL_ROW_SET && row_expert[gid / max(SILU_MUL_ROW, 1u)] >= SILU_MUL_EXPERTS);
}

template <typename T>
[[kernel]] void silu_mul(
    device       T* out  [[buffer(0)]],
    const device T* gate [[buffer(1)]],
    const device T* up   [[buffer(2)]],
    const device uint* row_expert [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (silu_mul_skips(row_expert, gid)) {
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
    const device uint* row_expert [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (silu_mul_skips(row_expert, gid)) {
    return;
  }
  out[gid] = static_cast<T>(gelu_mul_f(float(gate[gid]), float(up[gid])));
}

#define INST_GELU_MUL(dtype_tag, mtl_type) \
  SCRATCHY_KERNEL(gelu_mul_##dtype_tag, gelu_mul<mtl_type>)

INST_GELU_MUL(f16,  half)
INST_GELU_MUL(bf16, bfloat)

// gpt-oss SwiGLU-OAI sibling (`swiglu_oai_mul_f` in gated_act.h): the
// clamped affine activation OpenAI ships instead of SiLU. Same three
// instructions as the SwiGLU decomposition, with the OAI act as the
// elementwise tail — the decomposed fallback of a bake whose gate/up fold
// did not fuse (the fused path is `affine_gather_qmv_gated`'s
// GATED_ACT == 2 epilogue).
template <typename T>
[[kernel]] void swiglu_oai_mul(
    device       T* out  [[buffer(0)]],
    const device T* gate [[buffer(1)]],
    const device T* up   [[buffer(2)]],
    const device uint* row_expert [[buffer(3)]],
    const device T* linear_bias [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  if (silu_mul_skips(row_expert, gid)) {
    return;
  }
  float g = float(gate[gid]);
  float u = float(up[gid]);
  if (SILU_MUL_LINEAR_BIAS) {
    // [E, 2·row]: the expert's gate rows then its up rows.
    const uint row = max(SILU_MUL_ROW, 1u), col = gid % row;
    const device T* b = linear_bias + size_t(row_expert[gid / row]) * 2 * size_t(row);
    g += float(b[col]);
    u += float(b[row + col]);
  }
  out[gid] = static_cast<T>(swiglu_oai_mul_f(g, u, SILU_MUL_SWIGLU_LIMIT));
}

#define INST_SWIGLU_OAI_MUL(dtype_tag, mtl_type) \
  SCRATCHY_KERNEL(swiglu_oai_mul_##dtype_tag, swiglu_oai_mul<mtl_type>)

INST_SWIGLU_OAI_MUL(f16,  half)
INST_SWIGLU_OAI_MUL(bf16, bfloat)
