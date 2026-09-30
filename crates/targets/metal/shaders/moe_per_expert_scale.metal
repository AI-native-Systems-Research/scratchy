// SPDX-License-Identifier: Apache-2.0
//
// Gemma-4 `gemma_moe` op: per-expert score scale gather-multiply.
//
// Oracle (mlx_lm/models/gemma4_text.py Router): after the softmax over the
// top-k gathered router scores, each weight is multiplied by the routed
// expert's `per_expert_scale`:
//
//     w[m, j] *= per_expert_scale[idx[m, j]]
//
// This is the one genuinely-new metal kernel of the gemma_moe lowering: an
// in-place [M, top_k] gather-multiply. The per_expert_scale tensor is the
// router bundle's bf16/f16 `[num_experts]` vector; `idx` is the [M, top_k]
// u32 top-k index buffer (the same `topk_inds` the affine_gather_qmv steps
// read). The softmax already wrote `topk_scores` in place; this scales it.
//
// Function constants:
//   MPES_N — total element count = M * top_k.
//
// Dispatch: 1 thread per (m, j). Float intermediate so the bf16/f16
// multiply matches the host reference's single rounding.

#include <metal_stdlib>
#include "megakernel/mk_common.h"

using namespace metal;

#define MPES_CONSTS(X) X(uint, n, MPES_N, 0)
#ifndef MK_BODIES_ONLY
MPES_CONSTS(MK_FC_DECLARE)
struct MpesFc {
  MPES_CONSTS(MK_FC_ACCESSOR)
};
#endif

// Body shared by the dispatch kernel and the megakernel adapter (scores / indices coherent there).
template <typename T, typename C, typename SP, typename IP>
METAL_FUNC void moe_per_expert_scale_body(SP topk_scores, IP topk_inds,
                                          const device T* per_expert_scale, uint gid) {
  if (gid >= C::n()) {
    return;
  }
  uint expert = topk_inds[gid];
  float s = float(per_expert_scale[expert]);
  float w = float(topk_scores[gid]);
  topk_scores[gid] = static_cast<T>(w * s);
}

template <typename T, typename C>
MK_FUNC void mk_moe_per_expert_scale(thread const MkStep& s, MkLane l, threadgroup uchar*) {
  if (!l.live) return;
  moe_per_expert_scale_body<T, C>((mk_ptr<T>)s.addr[0], (mk_cptr<uint>)s.addr[1],
                                  (const device T*)s.addr[2], mk_thread_in_grid(l).x);
}

#ifndef MK_BODIES_ONLY
template <typename T>
[[kernel]] void moe_per_expert_scale(
    device       T*        topk_scores      [[buffer(0)]],
    const device uint*     topk_inds        [[buffer(1)]],
    const device T*        per_expert_scale [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  moe_per_expert_scale_body<T, MpesFc>(topk_scores, topk_inds, per_expert_scale, gid);
}

#define INST_MPES(dtype_tag, mtl_type)                                    \
  template [[host_name("moe_per_expert_scale_" #dtype_tag)]] [[kernel]]   \
  void moe_per_expert_scale<mtl_type>(                                    \
      device       mtl_type* topk_scores      [[buffer(0)]],             \
      const device uint*     topk_inds        [[buffer(1)]],             \
      const device mtl_type* per_expert_scale [[buffer(2)]],             \
      uint gid [[thread_position_in_grid]]);
#else
#define INST_MPES(dtype_tag, mtl_type)                                                       \
  MK_TAIL(moe_per_expert_scale_##dtype_tag, 0x3,                                    \
             (mk_moe_per_expert_scale<mtl_type, MK_C>), MPES_CONSTS)
#endif

INST_MPES(float16,  half)
INST_MPES(bfloat16, bfloat)
