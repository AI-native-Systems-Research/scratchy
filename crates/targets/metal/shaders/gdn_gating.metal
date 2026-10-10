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
// Mamba-2 (Nemotron-H) gates by the same `-exp(A_log) * softplus(dt + dt_bias)`, its `dt` the
// trailing columns of the in-projection row: with `GDN_GATING_MAMBA` set, `a` is read as that column
// slice and `beta` carries `dt = softplus(dt + dt_bias)` itself (the scan's input scale); `b` is
// unread.
//
// Baked constants:
//   GDN_GATING_N         — total element count (= T * num_heads)
//   GDN_GATING_NUM_HEADS — num_v_heads (to recover h = gid % num_heads)
//   GDN_GATING_MAMBA     — optional: the Mamba-2 form above
//   GDN_GATING_A_STRIDE  — optional: `a`'s row stride (unset: num_heads)
//   GDN_GATING_A_OFFSET  — optional: `a`'s first column in that row (unset: 0)
//
// Dispatch: 1 thread per (t, h) element.

#include <metal_stdlib>
#include "baked.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, GDN_GATING_N, 0);
SCRATCHY_CONSTANT(uint, GDN_GATING_NUM_HEADS, 1);
SCRATCHY_CONSTANT_OPTIONAL(bool, GDN_GATING_MAMBA, 2);
SCRATCHY_CONSTANT_OPTIONAL(uint, GDN_GATING_A_STRIDE, 3);
SCRATCHY_CONSTANT_OPTIONAL(uint, GDN_GATING_A_OFFSET, 4);

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
  uint a_ix = GDN_GATING_A_STRIDE_SET
                  ? (gid / GDN_GATING_NUM_HEADS) * GDN_GATING_A_STRIDE + GDN_GATING_A_OFFSET + h
                  : gid;
  // g = -exp(A_log[h]) * softplus(a + dt_bias[h])
  float av = float(a[a_ix]) + float(dt_bias[h]);
  float sp = av <= 20.0f ? log(1.0f + exp(av)) : av;
  g_out[gid] = -exp(float(a_log[h])) * sp;
  if (GDN_GATING_MAMBA) {
    beta_out[gid] = sp;  // dt
    return;
  }
  // beta = sigmoid(b)
  float bv = float(b[gid]);
  beta_out[gid] = 1.0f / (1.0f + exp(-bv));
}

#define INST_GDN_GATING(dtype_tag, mtl_type) \
  SCRATCHY_KERNEL(gdn_gating_##dtype_tag, gdn_gating<mtl_type>)

INST_GDN_GATING(f16,  half)
INST_GDN_GATING(bf16, bfloat)
INST_GDN_GATING(f32,  float)
