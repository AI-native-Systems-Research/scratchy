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
//   GDN_RMS_ROWS_PER_TG — the simdgroup form's rows a threadgroup (one per simdgroup); the
//                         threadgroup form's keys carry no slot 6
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

// The simdgroup form of the same norm, for a reduction width that is a power
// of two in [32, 256] — the widths whose covering threadgroup above (the
// smallest power of two ≥ d) gives every thread exactly ONE element, so its
// tree pairs thread i with thread i+s at every level s = tg/2 … 1. The
// lowering dispatches this form only for those widths, one simdgroup a row
// (`GDN_RMS_ROWS_PER_TG` rows a threadgroup, one per simdgroup) and
// `GDN_RMS_D / 32` strided elements a lane (element `lane + 32*j`), where
// that SAME tree needs no threadgroup memory and no barrier:
//
//   - every level with `s >= 32` pairs elements whose indices differ by a
//     multiple of 32 — both live in ONE lane's registers as `j` and
//     `j + s/32` — so the top `log2(d/32)` levels are register adds;
//   - every level with `s <= 16` pairs lanes of one simdgroup, so the bottom
//     five levels are simd shuffles — every lane folds (as `gdn_decode`
//     reduces), and lane 0's chain — the only one read — draws solely
//     in-range sources, so it is the threadgroup form's `sdata[0]` tree. The
//     sums agree BIT FOR BIT, and `inv` (broadcast lane 0's sum, then the
//     same rsqrt) and the per-element output formula are the threadgroup
//     form's own.
//
// Shuffle calls are simdgroup-uniform (a row never splits a simdgroup, and
// the row bound returns whole simdgroups); Mamba-2's gate-before-norm and
// strided `z` fold in exactly as above.
//
// `GDN_RMS_ROWS_PER_TG` is optional only because the threadgroup form's keys
// carry no slot 6; the simd form's own static_assert below refuses a bake that
// leaves it unset or zero (a zero would make every threadgroup recompute the
// first rows instead of striding past them).
SCRATCHY_CONSTANT_OPTIONAL(uint, GDN_RMS_ROWS_PER_TG, 6);

template <typename T>
[[kernel]] void gdn_rms_norm_gated_simd(
    device       T*     out    [[buffer(0)]],
    const device float* x      [[buffer(1)]],
    const device T*     z      [[buffer(2)]],
    const device float* weight [[buffer(3)]],
    uint gid     [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
  uint d = GDN_RMS_D;
  uint row = gid * GDN_RMS_ROWS_PER_TG + tid / 32;
  if (row >= GDN_RMS_ROWS) {
    return;
  }
  uint lane = tid % 32;
  // A lane's strided elements. The clamp keeps this template parseable in the threadgroup
  // form's own bakes — clang checks its non-template-dependent code there too, where slot 6
  // is unset and `GDN_RMS_D` can be below 32 — and is the identity wherever the static_assert
  // below lets it instantiate.
  constexpr uint e = GDN_RMS_D < 32u ? 1u : GDN_RMS_D / 32u;
  // Checked where this kernel is baked (`sizeof(T)` defers to its instantiation): the lane
  // layout above rests on the width being a power of two in [32, 256] and the rows a
  // threadgroup being set — anything else sums the wrong elements or strides rows wrong.
  static_assert(sizeof(T) && GDN_RMS_ROWS_PER_TG_SET && GDN_RMS_ROWS_PER_TG > 0u &&
                    GDN_RMS_D >= 32u && GDN_RMS_D <= 256u &&
                    (GDN_RMS_D & (GDN_RMS_D - 1u)) == 0u,
                "one simdgroup a row, a power-of-two width in [32, 256]");
  const uint group = row % GDN_RMS_TOKEN_ROWS;
  const device float* x_row = x + row * d;
  const device T*     z_row = GDN_RMS_Z_STRIDE_SET
                                  ? z + (row / GDN_RMS_TOKEN_ROWS) * GDN_RMS_Z_STRIDE + group * d
                                  : z + row * d;
  const device float* w_row = weight + group * d;
  device T*           o_row = out + row * d;

  // The lane's squares, one per strided element — the gate (Mamba-2) folding
  // in per element, exactly as the threadgroup form's load loop does. The
  // block's `contract(off)` pragma keeps `p[j] = v*v` and the register-tree
  // adds below separate roundings, as the threadgroup form's are: fast-math
  // would otherwise fuse them into FMAs, whose single roundings differ by an
  // ULP from the tree's (and only for some values — a silent parity drift).
  // The block ends the effect before the store loop, which compiles with the
  // threadgroup form's own contraction.
  float v;
  {
#pragma clang fp contract(off)
    float p[e];
    _Pragma("unroll") for (uint j = 0; j < e; j++) {
      float vj = x_row[lane + 32 * j];
      if (GDN_RMS_GATE_BEFORE) {
        float zi = float(z_row[lane + 32 * j]);
        vj *= zi / (1.0f + exp(-zi));
      }
      p[j] = vj * vj;
    }
    // The tree's top log2(e) levels (every pairing s ≥ 32): register adds.
    _Pragma("unroll") for (uint s = e / 2; s > 0; s >>= 1) {
      _Pragma("unroll") for (uint j = 0; j < s; j++) {
        p[j] += p[j + s];
      }
    }
    v = p[0];
  }
  // The tree's bottom five levels (every pairing s ≤ 16): simd shuffles, as
  // the decode path reduces (`gdn_decode`). Every lane folds, but only lane
  // 0's chain matters — its sources are all in range, so its sum is the
  // threadgroup form's sdata[0] bit for bit.
  for (uint s = 16; s > 0; s >>= 1) {
    v += simd_shuffle_down(v, s);
  }
  float inv = rsqrt(simd_broadcast(v, 0) / float(d) + GDN_RMS_EPS);

  // The store loop is the threadgroup form's own shape — a runtime stride off
  // the thread id (here the lane, strided by the simdgroup width, a runtime
  // quotient of `tg_size`) — so the compiler lowers its `exp`, division and
  // multiply chain exactly as the threadgroup form's, bit for bit. The clamp
  // is the identity wherever the static_assert lets this kernel instantiate;
  // it only keeps the never-instantiated parse (slot 6 unset) free of a
  // division-by-zero.
  uint simdw = tg_size / (GDN_RMS_ROWS_PER_TG < 1u ? 1u : GDN_RMS_ROWS_PER_TG);
  for (uint i = lane; i < d; i += simdw) {
    float zi = float(z_row[i]);
    float silu_z = zi / (1.0f + exp(-zi));  // SiLU(z) = z * sigmoid(z)
    o_row[i] = GDN_RMS_GATE_BEFORE ? T(x_row[i] * silu_z * inv * float(w_row[i]))
                                   : T(x_row[i] * inv * float(w_row[i]) * silu_z);
  }
}

#define INST_GDN_RMS_NORM_GATED_SIMD(dtype_tag, mtl_type) \
  SCRATCHY_KERNEL(gdn_rms_norm_gated_simd_##dtype_tag, gdn_rms_norm_gated_simd<mtl_type>)

INST_GDN_RMS_NORM_GATED_SIMD(f16,  half)
INST_GDN_RMS_NORM_GATED_SIMD(bf16, bfloat)
INST_GDN_RMS_NORM_GATED_SIMD(f32,  float)
