// SPDX-License-Identifier: Apache-2.0
//
// Gated-DeltaNet gated RMSNorm (Qwen3.5 / Qwen3-Next), norm_before_gate.
// Mirrors `cpu_golden::gdn_rms_norm_gated` + transformers `RMSNormGated`:
//
//   out[r,i] = rmsnorm_over_d(x[r])[i] * weight[i] * silu(z[r,i])
//
// where rmsnorm uses `var = mean(x^2)`, `inv = rsqrt(var + eps)`, and the
// gate is **SiLU(z) = z * sigmoid(z)** (NOT plain sigmoid — matches the
// silu fix in gdn_recurrent_kernels.cu). `d = head_v_dim`; normalization
// is per value-head, so `total_rows = num_tokens * num_v_heads`.
//
// `x` is f32 (the recurrent-scan output `o`); `z`/`out` are the model dtype
// (`T`) — `out` feeds the `out_proj` gemm directly, so it is written in
// model dtype (the GDN op's `out_slot` is a model-dtype arena slot). The
// `weight` (`linear_attn.norm.weight`) is **float32 on disk** (kept un-cast
// by the loader), so it is bound as `float*` regardless of `T`.
//
// Function constants:
//   GDN_RMS_D    — head_v_dim (the per-row reduction width)
//   GDN_RMS_ROWS — total_rows (= num_tokens * num_v_heads)
//   GDN_RMS_EPS  — rms_norm_eps
//
// Dispatch: one threadgroup per row; threadgroup reduction over d.

#include <metal_stdlib>
#include "megakernel/mk_common.h"

using namespace metal;

#define GDN_RMS_CONSTS(X) \
  X(uint, d, GDN_RMS_D, 0) X(uint, rows, GDN_RMS_ROWS, 1) X(float, eps, GDN_RMS_EPS, 2)
#ifndef MK_BODIES_ONLY
GDN_RMS_CONSTS(MK_FC_DECLARE)
struct GdnRmsFc {
  GDN_RMS_CONSTS(MK_FC_ACCESSOR)
};
#endif

// Body shared by the dispatch kernels and the megakernel adapter. `live` is a compile-time `true`
// on the dispatch path (its early return already ran); the megakernel passes whether this thread
// belongs to a real row, and every thread runs every barrier.
template <typename T, typename C, typename OP, typename XP, typename ZP>
METAL_FUNC void gdn_rms_norm_gated_body(OP out, XP x, ZP z, const device float* weight, uint gid,
                                        uint tid, uint tg_size, bool live,
                                        threadgroup float* sdata) {
  uint d = C::d();
  XP x_row = x + gid * d;
  ZP z_row = z + gid * d;
  OP o_row = out + gid * d;

  float ss = 0.0f;
  if (live) {
    for (uint i = tid; i < d; i += tg_size) {
      float v = x_row[i];
      ss += v * v;
    }
  }
  sdata[tid] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tg_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv = rsqrt(sdata[0] / float(d) + C::eps());

  // `x · inv · w · SiLU(z)`, SiLU(z) = z / (1 + exp(-z)), pinned to the order the dispatch kernel's
  // element loop computes it in — `((inv · z) · x) · w / (1 + exp(-z))` — so a row of at most
  // `tg_size` elements (one pass, no loop) is not reassociated differently.
  {
#pragma clang fp reassociate(off)
    if (live) {
      for (uint i = tid; i < d; i += tg_size) {
        float zi = float(z_row[i]);
        o_row[i] = T((((inv * zi) * x_row[i]) * float(weight[i])) / (1.0f + exp(-zi)));
      }
    }
  }
}

// Megakernel adapter: out (0), x (1) and z (2) device-coherent; one virtual threadgroup per row
// owns the dispatch kernel's `sdata[1024]`.
template <typename T, typename C>
MK_FUNC void mk_gdn_rms_norm_gated(thread const MkStep& s, MkLane l, threadgroup uchar* tg) {
  gdn_rms_norm_gated_body<T, C>((mk_ptr<T>)s.addr[0], (mk_cptr<float>)s.addr[1],
                                (mk_cptr<T>)s.addr[2], (const device float*)s.addr[3],
                                l.tg_pos.x, l.tid, l.tpg.x, l.live && l.tg_pos.x < C::rows(),
                                (threadgroup float*)mk_region(s, l, tg));
}

#ifndef MK_BODIES_ONLY
template <typename T>
[[kernel]] void gdn_rms_norm_gated(
    device       T*     out    [[buffer(0)]],
    const device float* x      [[buffer(1)]],
    const device T*     z      [[buffer(2)]],
    const device float* weight [[buffer(3)]],
    uint gid     [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
  if (gid >= GDN_RMS_ROWS) {
    return;
  }
  threadgroup float sdata[1024];
  gdn_rms_norm_gated_body<T, GdnRmsFc>(out, x, z, weight, gid, tid, tg_size, true, sdata);
}

#define INST_GDN_RMS_NORM_GATED(dtype_tag, mtl_type)                      \
  template [[host_name("gdn_rms_norm_gated_" #dtype_tag)]] [[kernel]] void\
  gdn_rms_norm_gated<mtl_type>(                                          \
      device       mtl_type* out    [[buffer(0)]],                        \
      const device float*    x      [[buffer(1)]],                        \
      const device mtl_type* z      [[buffer(2)]],                        \
      const device float*    weight [[buffer(3)]],                        \
      uint gid     [[threadgroup_position_in_grid]],                      \
      uint tid     [[thread_position_in_threadgroup]],                    \
      uint tg_size [[threads_per_threadgroup]]);
#else
#define INST_GDN_RMS_NORM_GATED(dtype_tag, mtl_type)                                    \
  MK_ADAPTER(gdn_rms_norm_gated_##dtype_tag, 4096, 0x7,                                 \
             (mk_gdn_rms_norm_gated<mtl_type, MK_C>), GDN_RMS_CONSTS)
#endif

INST_GDN_RMS_NORM_GATED(f16,  half)
INST_GDN_RMS_NORM_GATED(bf16, bfloat)
INST_GDN_RMS_NORM_GATED(f32,  float)
