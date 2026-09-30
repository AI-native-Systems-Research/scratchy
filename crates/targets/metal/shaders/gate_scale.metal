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
#include "megakernel/mk_common.h"

using namespace metal;

#define GATE_SCALE_CONSTS(X) X(uint, n, GATE_SCALE_N, 0) X(uint, cols, GATE_SCALE_COLS, 1)
#ifndef MK_BODIES_ONLY
GATE_SCALE_CONSTS(MK_FC_DECLARE)
struct GateScaleFc {
  GATE_SCALE_CONSTS(MK_FC_ACCESSOR)
};
#endif

// Body shared by the dispatch kernels and the megakernel adapter: element `gid` (< n).
template <typename T, typename C, typename OP, typename IP>
METAL_FUNC void gate_scale_body(OP out, IP routed, IP shared_y, IP g, uint gid) {
  uint row = gid / C::cols();
  float r = float(routed[gid]);
  float s = float(shared_y[gid]);
  float gv = float(g[row]);
  float sig_g = 1.0f / (1.0f + exp(-gv));
  out[gid] = static_cast<T>(r + s * sig_g);
}

// Megakernel adapter: out (0), routed (1), shared_y (2) and g (3) device-coherent.
template <typename T, typename C>
MK_FUNC void mk_gate_scale(thread const MkStep& s, MkLane l, threadgroup uchar*) {
  const uint gid = mk_thread_in_grid(l).x;
  if (!l.live || gid >= C::n()) return;
  gate_scale_body<T, C>((mk_ptr<T>)s.addr[0], (mk_cptr<T>)s.addr[1], (mk_cptr<T>)s.addr[2],
                        (mk_cptr<T>)s.addr[3], gid);
}

#ifndef MK_BODIES_ONLY
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
  gate_scale_body<T, GateScaleFc>(out, routed, shared_y, g, gid);
}

#define INST_GATE_SCALE(dtype_tag, mtl_type)                              \
  template [[host_name("gate_scale_" #dtype_tag)]] [[kernel]] void        \
  gate_scale<mtl_type>(                                                   \
      device       mtl_type* out      [[buffer(0)]],                      \
      const device mtl_type* routed   [[buffer(1)]],                      \
      const device mtl_type* shared_y [[buffer(2)]],                      \
      const device mtl_type* g        [[buffer(3)]],                      \
      uint gid [[thread_position_in_grid]]);
#else
#define INST_GATE_SCALE(dtype_tag, mtl_type) \
  MK_TAIL(gate_scale_##dtype_tag, 0xf, (mk_gate_scale<mtl_type, MK_C>), GATE_SCALE_CONSTS)
#endif

INST_GATE_SCALE(f16,  half)
INST_GATE_SCALE(bf16, bfloat)
