// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5 attention output-gate split: per-head deinterleave of the
// DOUBLED `q_proj` output into `query` and `gate`.
//
// transformers `Qwen3_5Attention`:
//   q_proj(x).view(*, num_heads, 2*head_dim) -> chunk(2, dim=-1)
// i.e. each head's `2*head_dim` block is `[query(head_dim) | gate(head_dim)]`.
// Input  qg:   [M, num_heads * 2 * head_dim]
// Output query:[M, num_heads * head_dim]   (= qg[:, h, 0:head_dim])
// Output gate: [M, num_heads * head_dim]   (= qg[:, h, head_dim:2*head_dim])
//
// Function constants:
//   GATE_SPLIT_N         — per-output element count (= M * num_heads * head_dim)
//   GATE_SPLIT_HEAD_DIM  — head_dim
//   GATE_SPLIT_NUM_HEADS — num_heads
//
// Dispatch: 1 thread per output element; each writes one query elem and
// the matching gate elem from the interleaved source row.

#include <metal_stdlib>

using namespace metal;

constant uint GATE_SPLIT_N         [[function_constant(0)]];
constant uint GATE_SPLIT_HEAD_DIM  [[function_constant(1)]];
constant uint GATE_SPLIT_NUM_HEADS [[function_constant(2)]];

template <typename T>
[[kernel]] void gate_split(
    device       T* q_out    [[buffer(0)]],
    device       T* gate_out [[buffer(1)]],
    const device T* qg       [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= GATE_SPLIT_N) {
    return;
  }
  uint hd   = GATE_SPLIT_HEAD_DIM;
  uint cols = GATE_SPLIT_NUM_HEADS * hd;   // per-output row width
  uint row  = gid / cols;
  uint rem  = gid % cols;
  uint head = rem / hd;
  uint d    = rem % hd;
  // qg row width = num_heads * 2 * head_dim; within a head: [query | gate].
  uint base = row * (cols * 2) + head * (2 * hd) + d;
  q_out[gid]    = qg[base];
  gate_out[gid] = qg[base + hd];
}

#define INST_GATE_SPLIT(dtype_tag, mtl_type)                              \
  template [[host_name("gate_split_" #dtype_tag)]] [[kernel]] void        \
  gate_split<mtl_type>(                                                   \
      device       mtl_type* q_out    [[buffer(0)]],                      \
      device       mtl_type* gate_out [[buffer(1)]],                      \
      const device mtl_type* qg       [[buffer(2)]],                      \
      uint gid [[thread_position_in_grid]]);

INST_GATE_SPLIT(f16,  half)
INST_GATE_SPLIT(bf16, bfloat)
