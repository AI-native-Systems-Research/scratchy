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
#include "megakernel/mk_common.h"

using namespace metal;

#define GATE_SPLIT_CONSTS(X)                                                              \
  X(uint, n, GATE_SPLIT_N, 0) X(uint, head_dim, GATE_SPLIT_HEAD_DIM, 1)                   \
  X(uint, num_heads, GATE_SPLIT_NUM_HEADS, 2)
#ifndef MK_BODIES_ONLY
GATE_SPLIT_CONSTS(MK_FC_DECLARE)
struct GateSplitFc {
  GATE_SPLIT_CONSTS(MK_FC_ACCESSOR)
};
#endif

// Body shared by the dispatch kernels and the megakernel adapter: element `gid` (< n).
template <typename C, typename QP, typename GP, typename SP>
METAL_FUNC void gate_split_body(QP q_out, GP gate_out, SP qg, uint gid) {
  uint hd   = C::head_dim();
  uint cols = C::num_heads() * hd;   // per-output row width
  uint row  = gid / cols;
  uint rem  = gid % cols;
  uint head = rem / hd;
  uint d    = rem % hd;
  // qg row width = num_heads * 2 * head_dim; within a head: [query | gate].
  uint base = row * (cols * 2) + head * (2 * hd) + d;
  q_out[gid]    = qg[base];
  gate_out[gid] = qg[base + hd];
}

// Megakernel adapter: q_out (0), gate_out (1) and qg (2) device-coherent.
template <typename T, typename C>
MK_FUNC void mk_gate_split(thread const MkStep& s, MkLane l, threadgroup uchar*) {
  const uint gid = mk_thread_in_grid(l).x;
  if (!l.live || gid >= C::n()) return;
  gate_split_body<C>((mk_ptr<T>)s.addr[0], (mk_ptr<T>)s.addr[1], (mk_cptr<T>)s.addr[2], gid);
}

#ifndef MK_BODIES_ONLY
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
  gate_split_body<GateSplitFc>(q_out, gate_out, qg, gid);
}

#define INST_GATE_SPLIT(dtype_tag, mtl_type)                              \
  template [[host_name("gate_split_" #dtype_tag)]] [[kernel]] void        \
  gate_split<mtl_type>(                                                   \
      device       mtl_type* q_out    [[buffer(0)]],                      \
      device       mtl_type* gate_out [[buffer(1)]],                      \
      const device mtl_type* qg       [[buffer(2)]],                      \
      uint gid [[thread_position_in_grid]]);
#else
#define INST_GATE_SPLIT(dtype_tag, mtl_type) \
  MK_TAIL(gate_split_##dtype_tag, 0x7, (mk_gate_split<mtl_type, MK_C>), GATE_SPLIT_CONSTS)
#endif

INST_GATE_SPLIT(f16,  half)
INST_GATE_SPLIT(bf16, bfloat)
