// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5 attention output gate: `out = attn * sigmoid(gate)`.
//
// The full-attention layer's `q_proj` is doubled; `gate_split`
// deinterleaves the per-head `[query | gate]` blocks, attention runs on
// `query`, and the result is gated by `sigmoid(gate)` before `o_proj`
// (transformers `Qwen3_5Attention`: `attn_output * torch.sigmoid(gate)`).
//
// Function constants:
//   GATE_APPLY_N — total output element count (= M * num_heads * head_dim)
//
// Dispatch: 1 thread per output element (mirrors `silu_mul.metal`).
// Sigmoid kept in float so the half/bfloat exp() tail stays representable.

#include <metal_stdlib>
#include "megakernel/mk_common.h"

using namespace metal;

#define GATE_APPLY_CONSTS(X) X(uint, n, GATE_APPLY_N, 0)
#ifndef MK_BODIES_ONLY
GATE_APPLY_CONSTS(MK_FC_DECLARE)
#endif

// Body shared by the dispatch kernels and the megakernel adapter: element `gid` (< n).
template <typename T, typename OP, typename IP>
METAL_FUNC void gate_apply_body(OP out, IP attn, IP gate, uint gid) {
  float a = float(attn[gid]);
  float g = float(gate[gid]);
  float sig_g = 1.0f / (1.0f + exp(-g));
  out[gid] = static_cast<T>(a * sig_g);
}

// Megakernel adapter: out (0), attn (1) and gate (2) device-coherent.
template <typename T, typename C>
MK_FUNC void mk_gate_apply(thread const MkStep& s, MkLane l, threadgroup uchar*) {
  const uint gid = mk_thread_in_grid(l).x;
  if (!l.live || gid >= C::n()) return;
  gate_apply_body<T>((mk_ptr<T>)s.addr[0], (mk_cptr<T>)s.addr[1], (mk_cptr<T>)s.addr[2], gid);
}

#ifndef MK_BODIES_ONLY
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
  gate_apply_body<T>(out, attn, gate, gid);
}

#define INST_GATE_APPLY(dtype_tag, mtl_type)                              \
  template [[host_name("gate_apply_" #dtype_tag)]] [[kernel]] void        \
  gate_apply<mtl_type>(                                                   \
      device       mtl_type* out  [[buffer(0)]],                          \
      const device mtl_type* attn [[buffer(1)]],                          \
      const device mtl_type* gate [[buffer(2)]],                          \
      uint gid [[thread_position_in_grid]]);
#else
#define INST_GATE_APPLY(dtype_tag, mtl_type) \
  MK_TAIL(gate_apply_##dtype_tag, 0x7, (mk_gate_apply<mtl_type, MK_C>), GATE_APPLY_CONSTS)
#endif

INST_GATE_APPLY(f16,  half)
INST_GATE_APPLY(bf16, bfloat)
