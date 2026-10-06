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
// Baked constants:
//   GATE_SCALE_N    — total output element count (= M * hidden_size)
//   GATE_SCALE_COLS — hidden_size (the gate's row stride)
//
// Dispatch: 1 thread per output element (mirrors `gate_apply.metal`).
// Sigmoid kept in float so the half/bfloat exp() tail stays representable.

#include <metal_stdlib>
#include "baked.h"
#include "gated_act.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, GATE_SCALE_N, 0);
SCRATCHY_CONSTANT(uint, GATE_SCALE_COLS, 1);

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
  out[gid] = static_cast<T>(gate_scale_f(float(routed[gid]), float(shared_y[gid]), float(g[row])));
}

#define INST_GATE_SCALE(dtype_tag, mtl_type) \
  SCRATCHY_KERNEL(gate_scale_##dtype_tag, gate_scale<mtl_type>)

INST_GATE_SCALE(f16,  half)
INST_GATE_SCALE(bf16, bfloat)
