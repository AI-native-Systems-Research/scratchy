// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5 attention output gate: `out = attn * sigmoid(gate)`.
//
// The full-attention layer's `q_proj` is doubled; `gate_split`
// deinterleaves the per-head `[query | gate]` blocks, attention runs on
// `query`, and the result is gated by `sigmoid(gate)` before `o_proj`
// (transformers `Qwen3_5Attention`: `attn_output * torch.sigmoid(gate)`).
//
// Baked constants:
//   GATE_APPLY_N — total output element count (= M * num_heads * head_dim)
//
// Dispatch: 1 thread per output element (mirrors `silu_mul.metal`).
// Sigmoid kept in float so the half/bfloat exp() tail stays representable.

#include <metal_stdlib>
#include "baked.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, GATE_APPLY_N, 0);

template <typename T>
[[kernel]] void gate_apply(
    device       T* out  [[buffer(0)]],
    const device T* attn [[buffer(1)]],
    const device T* gate [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= GATE_APPLY_N) {
    return;
  }
  float a = float(attn[gid]);
  float g = float(gate[gid]);
  float sig_g = 1.0f / (1.0f + exp(-g));
  out[gid] = static_cast<T>(a * sig_g);
}

#define INST_GATE_APPLY(dtype_tag, mtl_type) \
  SCRATCHY_KERNEL(gate_apply_##dtype_tag, gate_apply<mtl_type>)

INST_GATE_APPLY(f16,  half)
INST_GATE_APPLY(bf16, bfloat)
