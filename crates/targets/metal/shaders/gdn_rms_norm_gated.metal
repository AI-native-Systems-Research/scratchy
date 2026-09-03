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

using namespace metal;

constant uint  GDN_RMS_D    [[function_constant(0)]];
constant uint  GDN_RMS_ROWS [[function_constant(1)]];
constant float GDN_RMS_EPS  [[function_constant(2)]];

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
  uint d = GDN_RMS_D;
  const device float* x_row = x + gid * d;
  const device T*     z_row = z + gid * d;
  device T*           o_row = out + gid * d;

  threadgroup float sdata[1024];
  float ss = 0.0f;
  for (uint i = tid; i < d; i += tg_size) {
    float v = x_row[i];
    ss += v * v;
  }
  sdata[tid] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tg_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv = rsqrt(sdata[0] / float(d) + GDN_RMS_EPS);

  for (uint i = tid; i < d; i += tg_size) {
    float zi = float(z_row[i]);
    float silu_z = zi / (1.0f + exp(-zi));  // SiLU(z) = z * sigmoid(z)
    o_row[i] = T(x_row[i] * inv * float(weight[i]) * silu_z);
  }
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

INST_GDN_RMS_NORM_GATED(f16,  half)
INST_GDN_RMS_NORM_GATED(bf16, bfloat)
INST_GDN_RMS_NORM_GATED(f32,  float)
