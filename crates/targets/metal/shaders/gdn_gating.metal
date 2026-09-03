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

using namespace metal;

constant uint GDN_GATING_N         [[function_constant(0)]];
constant uint GDN_GATING_NUM_HEADS [[function_constant(1)]];

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
  uint h = gid % GDN_GATING_NUM_HEADS;
  // g = -exp(A_log[h]) * softplus(a + dt_bias[h])
  float av = float(a[gid]) + float(dt_bias[h]);
  float sp = av <= 20.0f ? log(1.0f + exp(av)) : av;
  g_out[gid] = -exp(float(a_log[h])) * sp;
  // beta = sigmoid(b)
  float bv = float(b[gid]);
  beta_out[gid] = 1.0f / (1.0f + exp(-bv));
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

INST_GDN_GATING(f16,  half)
INST_GDN_GATING(bf16, bfloat)
INST_GDN_GATING(f32,  float)
