// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5-MoE shared-expert combine: `out = routed + shared_y * sigmoid(g)`.
//
// The sparse MoE layer's routed-expert output (`moe_block`) is combined
// with the always-on shared expert: a SwiGLU MLP whose output is scaled
// by `sigmoid(shared_expert_gate(x))`. The gate is `[T, 1]` — ONE scalar
// per token — so unlike `gate_apply` (flat element-wise) the gate read is
// row-indexed: `row = gid / cols` (transformers `Qwen3_5MoeSparseMoeBlock`:
// `routed + shared_expert_output * torch.sigmoid(gate)`; mlx-lm
// `Qwen3NextSparseMoeBlock`: `y + mx.sigmoid(shared_expert_gate(x)) *
// shared_y`).
//
// Function constants:
//   GATE_SCALE_N    — total output element count (= M * hidden_size)
//   GATE_SCALE_COLS — hidden_size (the gate's row stride)
//
// Dispatch: 1 thread per output element (mirrors `gate_apply.metal`).
// Sigmoid kept in float so the half/bfloat exp() tail stays representable.

#include <metal_stdlib>

using namespace metal;

constant uint GATE_SCALE_N [[function_constant(0)]];
constant uint GATE_SCALE_COLS [[function_constant(1)]];

template <typename T>
[[kernel]] void gate_scale(
    device       T* out      [[buffer(0)]],
    const device T* routed   [[buffer(1)]],
    const device T* shared_y [[buffer(2)]],
    const device T* g        [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= GATE_SCALE_N) {
    return;
  }
  uint row = gid / GATE_SCALE_COLS;
  float r = float(routed[gid]);
  float s = float(shared_y[gid]);
  float gv = float(g[row]);
  float sig_g = 1.0f / (1.0f + exp(-gv));
  out[gid] = static_cast<T>(r + s * sig_g);
}

#define INST_GATE_SCALE(dtype_tag, mtl_type)                              \
  template [[host_name("gate_scale_" #dtype_tag)]] [[kernel]] void        \
  gate_scale<mtl_type>(                                                   \
      device       mtl_type* out      [[buffer(0)]],                      \
      const device mtl_type* routed   [[buffer(1)]],                      \
      const device mtl_type* shared_y [[buffer(2)]],                      \
      const device mtl_type* g        [[buffer(3)]],                      \
      uint gid [[thread_position_in_grid]]);

INST_GATE_SCALE(f16,  half)
INST_GATE_SCALE(bf16, bfloat)
