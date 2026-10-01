// SPDX-License-Identifier: Apache-2.0
//
// LayerNorm-with-bias (Qwen3.5-VL / Qwen3-VL ViT norm1/norm2/merger.norm).
//
// The ViT norms are LayerNorm WITH BIAS (each carries .weight AND .bias) —
// NOT RMSNorm. No metal matcher / kernel existed; this is net-new. Faithful to
// `mlx.nn.LayerNorm` (biased variance, eps inside the sqrt, affine):
//     y = (x - mean) * rsqrt(var + eps) * weight + bias
//     mean = E[x],  var = E[x^2] - mean^2   (population / biased, ddof=0)
// Verified == mlx.nn.LayerNorm: max_abs_err 2.4e-7.
//
// One threadgroup per row (token); two threadgroup reductions (sum, sum-of-
// squares). Layout: row-major [M, HIDDEN]. weight/bias are [HIDDEN].
//
// The gain/bias buffers carry the canonical's SCALE dtype, which is not
// always the activation dtype (ModernBERT: bf16 activations, f16 gains),
// so `T_scale` is a second template parameter and the host name encodes
// it as `_s_<scale>` — the same convention `rmsnorm` uses. Reading an
// f16 gain as bf16 turns 0.575 into 7.3e-05 and the whole model with it.
//
// Function constants: LN_M (rows), LN_HIDDEN (D), LN_EPS, LN_HAS_BIAS.

#include <metal_stdlib>
#include "megakernel/mk_common.h"

using namespace metal;

// LN_HAS_BIAS: 1 = the norm carries a `.bias` (ViT norms); 0 = weight-only
// LayerNorm (ModernBERT's `norm_bias=False`), where the `bias` buffer
// is bound to a valid-but-unread allocation and must NOT be applied.
#define LN_CONSTS(X)                                                                      \
  X(uint, m, LN_M, 0) X(uint, hidden, LN_HIDDEN, 1) X(float, eps, LN_EPS, 2)              \
  X(uint, has_bias, LN_HAS_BIAS, 3)
#ifndef MK_BODIES_ONLY
LN_CONSTS(MK_FC_DECLARE)
struct LnFc {
  LN_CONSTS(MK_FC_ACCESSOR)
};
#endif

// Body shared by the dispatch kernels and the megakernel adapter (see `rmsnorm_body`): `live` is
// a compile-time `true` on the dispatch path; the megakernel passes whether this thread belongs to
// a real row, and every thread runs every barrier.
template <typename T, typename T_scale, typename C, typename OP, typename IP>
METAL_FUNC void vision_layernorm_body(OP output, IP input, device const T_scale* weight,
                                      device const T_scale* bias, uint gid, uint tid,
                                      uint tg_size, bool live, threadgroup float* sh_sum,
                                      threadgroup float* sh_sq) {
  float ls = 0.0f, lsq = 0.0f;
  if (live) {
    for (uint i = tid; i < C::hidden(); i += tg_size) {
      float v = float(input[gid * C::hidden() + i]);
      ls += v;
      lsq += v * v;
    }
  }
  sh_sum[tid] = ls;
  sh_sq[tid] = lsq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
    if (tid < stride) {
      sh_sum[tid] += sh_sum[tid + stride];
      sh_sq[tid] += sh_sq[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  float inv_n = 1.0f / float(C::hidden());
  float mean = sh_sum[0] * inv_n;
  float var = sh_sq[0] * inv_n - mean * mean;
  var = max(var, 0.0f);                  // guard fp cancellation
  float inv = rsqrt(var + C::eps());

  if (live) {
    for (uint i = tid; i < C::hidden(); i += tg_size) {
      float v = float(input[gid * C::hidden() + i]);
      float w = float(weight[i]);
      float b = (C::has_bias() != 0u) ? float(bias[i]) : 0.0f;
      output[gid * C::hidden() + i] = T((v - mean) * inv * w + b);
    }
  }
}

// Megakernel adapter: one virtual threadgroup per row owns the dispatch kernel's `sh_sum[1024]` +
// `sh_sq[1024]`.
template <typename T, typename T_scale, typename C>
MK_FUNC void mk_vision_layernorm(thread const MkStep& s, MkLane l, threadgroup uchar* tg) {
  threadgroup float* sh = (threadgroup float*)mk_region(s, l, tg);
  vision_layernorm_body<T, T_scale, C>((mk_ptr<T>)s.addr[0], (mk_cptr<T>)s.addr[1],
                                       (device const T_scale*)s.addr[2],
                                       (device const T_scale*)s.addr[3], l.tg_pos.x, l.tid,
                                       l.tpg.x, l.live && l.tg_pos.x < C::m(), sh, sh + 1024);
}

#ifndef MK_BODIES_ONLY
template <typename T, typename T_scale>
[[kernel]] void vision_layernorm(
    device       T* output       [[buffer(0)]],
    device const T* input        [[buffer(1)]],
    device const T_scale* weight [[buffer(2)]],
    device const T_scale* bias   [[buffer(3)]],
    uint gid     [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
  if (gid >= LN_M) {
    return;
  }
  threadgroup float sh_sum[1024];
  threadgroup float sh_sq[1024];
  vision_layernorm_body<T, T_scale, LnFc>(output, input, weight, bias, gid, tid, tg_size, true,
                                          sh_sum, sh_sq);
}

#define INST_VISION_LAYERNORM(dtype_tag, mtl_type, scale_tag, mtl_scale)      \
  template [[host_name("vision_layernorm_" #dtype_tag "_s_" #scale_tag)]]     \
  [[kernel]] void                                                            \
  vision_layernorm<mtl_type, mtl_scale>(                                     \
      device       mtl_type* output       [[buffer(0)]],                     \
      device const mtl_type* input        [[buffer(1)]],                     \
      device const mtl_scale* weight      [[buffer(2)]],                     \
      device const mtl_scale* bias        [[buffer(3)]],                     \
      uint gid     [[threadgroup_position_in_grid]],                         \
      uint tid     [[thread_position_in_threadgroup]],                       \
      uint tg_size [[threads_per_threadgroup]]);
#else
#define INST_VISION_LAYERNORM(dtype_tag, mtl_type, scale_tag, mtl_scale)                   \
  MK_ADAPTER(vision_layernorm_##dtype_tag##_s_##scale_tag, 8192,                          \
             (mk_vision_layernorm<mtl_type, mtl_scale, MK_C>), LN_CONSTS)
#endif

INST_VISION_LAYERNORM(f16,  half,   f16,  half)
INST_VISION_LAYERNORM(f16,  half,   bf16, bfloat)
INST_VISION_LAYERNORM(bf16, bfloat, f16,  half)
INST_VISION_LAYERNORM(bf16, bfloat, bf16, bfloat)
INST_VISION_LAYERNORM(f32,  float,  f32,  float)
