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
// Mamba-2 (Nemotron-H `MambaRMSNormGated`, mlx-lm `nemotron_h.py`) gates BEFORE the norm, and
// normalizes per group of `d_inner / n_groups`:
//
//   out[r,i] = rmsnorm_over_d(silu(z[r]) * x[r])[i] * weight[g*d + i]
//
// its `z` the leading columns of the in-projection row and its weight spanning the whole
// `d_inner` row (row r = token t, group g = r % groups).
//
// Baked constants:
//   GDN_RMS_D           — head_v_dim / group width (the per-row reduction width)
//   GDN_RMS_ROWS        — total_rows (= num_tokens * num_v_heads; Mamba-2: * n_groups)
//   GDN_RMS_EPS         — rms_norm_eps
//   GDN_RMS_Z_STRIDE    — optional: `z`'s row stride, a token's rows a column slice of it (unset:
//                         `z` is `[total_rows, d]`)
//   GDN_RMS_GROUPS      — optional: rows a token has (with GDN_RMS_Z_STRIDE); the weight then spans
//                         all of them (`[groups * d]`) instead of one (`[d]`)
//   GDN_RMS_GATE_BEFORE — optional: the Mamba-2 order above
//
// Dispatch: one threadgroup per row, a power of two of threads; threadgroup
// reduction over d. Every power of two ≥ d sums in the same order (its extra
// tree levels add zeros), so the lowering dispatches the smallest, at most 256.

#include <metal_stdlib>
#include "baked.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, GDN_RMS_D, 0);
SCRATCHY_CONSTANT(uint, GDN_RMS_ROWS, 1);
SCRATCHY_CONSTANT(float, GDN_RMS_EPS, 2);
SCRATCHY_CONSTANT_OPTIONAL(uint, GDN_RMS_Z_STRIDE, 3);
SCRATCHY_CONSTANT_OPTIONAL(uint, GDN_RMS_GROUPS, 4);
SCRATCHY_CONSTANT_OPTIONAL(bool, GDN_RMS_GATE_BEFORE, 5);

constant constexpr uint GDN_RMS_TOKEN_ROWS = GDN_RMS_GROUPS_SET ? GDN_RMS_GROUPS : 1u;

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
  const uint group = gid % GDN_RMS_TOKEN_ROWS;
  const device float* x_row = x + gid * d;
  const device T*     z_row = GDN_RMS_Z_STRIDE_SET
                                  ? z + (gid / GDN_RMS_TOKEN_ROWS) * GDN_RMS_Z_STRIDE + group * d
                                  : z + gid * d;
  const device float* w_row = weight + group * d;
  device T*           o_row = out + gid * d;

  threadgroup float sdata[1024];
  float ss = 0.0f;
  for (uint i = tid; i < d; i += tg_size) {
    float v = x_row[i];
    if (GDN_RMS_GATE_BEFORE) {
      float zi = float(z_row[i]);
      v *= zi / (1.0f + exp(-zi));
    }
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
    o_row[i] = GDN_RMS_GATE_BEFORE ? T(x_row[i] * silu_z * inv * float(w_row[i]))
                                   : T(x_row[i] * inv * float(w_row[i]) * silu_z);
  }
}

#define INST_GDN_RMS_NORM_GATED(dtype_tag, mtl_type) \
  SCRATCHY_KERNEL(gdn_rms_norm_gated_##dtype_tag, gdn_rms_norm_gated<mtl_type>)

INST_GDN_RMS_NORM_GATED(f16,  half)
INST_GDN_RMS_NORM_GATED(bf16, bfloat)
INST_GDN_RMS_NORM_GATED(f32,  float)
