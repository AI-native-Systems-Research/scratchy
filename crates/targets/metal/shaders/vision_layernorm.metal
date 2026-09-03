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
// Function constants: LN_M (rows), LN_HIDDEN (D), LN_EPS.

#include <metal_stdlib>

using namespace metal;

constant uint  LN_M      [[function_constant(0)]];
constant uint  LN_HIDDEN [[function_constant(1)]];
constant float LN_EPS    [[function_constant(2)]];
// 1 = the norm carries a `.bias` (ViT norms); 0 = weight-only
// LayerNorm (ModernBERT's `norm_bias=False`), where the `bias` buffer
// is bound to a valid-but-unread allocation and must NOT be applied.
constant uint  LN_HAS_BIAS [[function_constant(3)]];

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

  float ls = 0.0f, lsq = 0.0f;
  for (uint i = tid; i < LN_HIDDEN; i += tg_size) {
    float v = float(input[gid * LN_HIDDEN + i]);
    ls += v;
    lsq += v * v;
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

  float inv_n = 1.0f / float(LN_HIDDEN);
  float mean = sh_sum[0] * inv_n;
  float var = sh_sq[0] * inv_n - mean * mean;
  var = max(var, 0.0f);                  // guard fp cancellation
  float inv = rsqrt(var + LN_EPS);

  for (uint i = tid; i < LN_HIDDEN; i += tg_size) {
    float v = float(input[gid * LN_HIDDEN + i]);
    float w = float(weight[i]);
    float b = (LN_HAS_BIAS != 0u) ? float(bias[i]) : 0.0f;
    output[gid * LN_HIDDEN + i] = T((v - mean) * inv * w + b);
  }
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

INST_VISION_LAYERNORM(f16,  half,   f16,  half)
INST_VISION_LAYERNORM(f16,  half,   bf16, bfloat)
INST_VISION_LAYERNORM(bf16, bfloat, f16,  half)
INST_VISION_LAYERNORM(bf16, bfloat, bf16, bfloat)
INST_VISION_LAYERNORM(f32,  float,  f32,  float)
