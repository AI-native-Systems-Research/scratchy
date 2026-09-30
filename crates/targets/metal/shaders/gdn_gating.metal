// SPDX-License-Identifier: Apache-2.0
//
// Gated-DeltaNet input-dependent gating (Qwen3.5 / Qwen3-Next).
// Mirrors `cpu_golden::gdn_gating` + the CUDA `fused_gdn_gating` kernel:
//
//   g[t,h]    = -exp(A_log[h]) * softplus(a[t,h] + dt_bias[h])   (<= 0)
//   beta[t,h] = sigmoid(b[t,h])
//
// where `softplus(x) = ln(1+exp(x))` for x <= 20, else `x` (the stable
// threshold the reference uses). `h` indexes value-heads (num_v_heads).
//
// Inputs a/b are `[T, num_heads]` and read in the model dtype (`T_act`);
// `dt_bias` is `[num_heads]` model-dtype (bf16 on disk). `A_log` is
// `[num_heads]` and is **float32 on disk** (`cast_predicate` keeps it
// un-cast; Python `.float()`s it at use) — so it is bound as `float*`
// regardless of `T_act`, NOT the model dtype. All values f32-accumulate.
// Outputs g/beta are f32 (consumed by the f32 recurrent-scan kernel).
//
// Function constants:
//   GDN_GATING_N         — total element count (= T * num_heads)
//   GDN_GATING_NUM_HEADS — num_v_heads (to recover h = gid % num_heads)
//
// Dispatch: 1 thread per (t, h) element.

#include <metal_stdlib>
#include "megakernel/mk_common.h"

using namespace metal;

#define GDN_GATING_CONSTS(X) X(uint, n, GDN_GATING_N, 0) X(uint, num_heads, GDN_GATING_NUM_HEADS, 1)
#ifndef MK_BODIES_ONLY
GDN_GATING_CONSTS(MK_FC_DECLARE)
struct GdnGatingFc {
  GDN_GATING_CONSTS(MK_FC_ACCESSOR)
};
#endif

// Body shared by the dispatch kernels and the megakernel adapter: element `gid` (< n).
template <typename T, typename C, typename FP, typename IP>
METAL_FUNC void gdn_gating_body(FP g_out, FP beta_out, IP a, IP b, const device float* a_log,
                                const device T* dt_bias, uint gid) {
  uint h = gid % C::num_heads();
  // g = -exp(A_log[h]) * softplus(a + dt_bias[h])
  float av = float(a[gid]) + float(dt_bias[h]);
  float sp = av <= 20.0f ? log(1.0f + exp(av)) : av;
  g_out[gid] = -exp(float(a_log[h])) * sp;
  // beta = sigmoid(b)
  float bv = float(b[gid]);
  beta_out[gid] = 1.0f / (1.0f + exp(-bv));
}

// Megakernel adapter: g_out (0), beta_out (1), a (2) and b (3) device-coherent.
template <typename T, typename C>
MK_FUNC void mk_gdn_gating(thread const MkStep& s, MkLane l, threadgroup uchar*) {
  const uint gid = mk_thread_in_grid(l).x;
  if (!l.live || gid >= C::n()) return;
  gdn_gating_body<T, C>((mk_ptr<float>)s.addr[0], (mk_ptr<float>)s.addr[1],
                        (mk_cptr<T>)s.addr[2], (mk_cptr<T>)s.addr[3],
                        (const device float*)s.addr[4], (const device T*)s.addr[5], gid);
}

#ifndef MK_BODIES_ONLY
template <typename T>
[[kernel]] void gdn_gating(
    device       float* g_out    [[buffer(0)]],
    device       float* beta_out [[buffer(1)]],
    const device T*     a        [[buffer(2)]],
    const device T*     b        [[buffer(3)]],
    const device float* a_log    [[buffer(4)]],
    const device T*     dt_bias  [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= GDN_GATING_N) {
    return;
  }
  gdn_gating_body<T, GdnGatingFc>(g_out, beta_out, a, b, a_log, dt_bias, gid);
}

#define INST_GDN_GATING(dtype_tag, mtl_type)                              \
  template [[host_name("gdn_gating_" #dtype_tag)]] [[kernel]] void        \
  gdn_gating<mtl_type>(                                                   \
      device       float*   g_out    [[buffer(0)]],                       \
      device       float*   beta_out [[buffer(1)]],                       \
      const device mtl_type* a       [[buffer(2)]],                       \
      const device mtl_type* b       [[buffer(3)]],                       \
      const device float*    a_log   [[buffer(4)]],                       \
      const device mtl_type* dt_bias [[buffer(5)]],                       \
      uint gid [[thread_position_in_grid]]);
#else
#define INST_GDN_GATING(dtype_tag, mtl_type) \
  MK_ADAPTER(gdn_gating_##dtype_tag, 0, 0xf, (mk_gdn_gating<mtl_type, MK_C>), GDN_GATING_CONSTS)
#endif

INST_GDN_GATING(f16,  half)
INST_GDN_GATING(bf16, bfloat)
INST_GDN_GATING(f32,  float)
