// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
#include "megakernel/mk_common.h"
using namespace metal;

#ifndef MK_BODIES_ONLY
// ============================================================================
// CopyRows: out[i] = in[i]  (flat element-wise copy, bounds-guarded)
//
// Materializes the vision `pixels` runtime extern into an arena tile
// (`Instruction::LoadPixels`). out @ buffer(0), in @ buffer(1), the
// element count `n` @ buffer(2) as a runtime `constant uint&` (NOT a
// function constant — bound via `setBytes` inline, like `gelu_tanh`),
// so the m_scaling tail and any bucket-padding rows are no-ops.
// ============================================================================

kernel void copy_rows_f16(
    device half* out [[buffer(0)]],
    device const half* in [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = in[gid];
}

kernel void copy_rows_bf16(
    device bfloat* out [[buffer(0)]],
    device const bfloat* in [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = in[gid];
}

// ============================================================================
// Add: out = a + b
// ============================================================================

kernel void add_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] + b[gid];
}

kernel void add_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] + b[gid];
}

// ── In-place residual add: residual += delta ────────────────────────────────
//
// The `KernelId::Add` lowering arm (`Instruction::Add`, interpreter/metal/
// lowering.rs) binds buffer(0) = residual (in/out) + buffer(1) = delta and
// dispatches token-parallel over `eff_m * width` exact threads (no bounds
// guard needed — same dispatchThreads convention as `bias_add_*_specialized`).
// The `_specialized` suffix matches the elementwise naming family the
// lowering's `pick_specialized_symbol` helper expects; this variant carries
// no function constants. First exercised by the Qwen3.5-VL vision tower —
// the text path fuses its residual into `FusedAddRmsNorm`, so the standalone
// add never reached metal before.
#endif // MK_BODIES_ONLY

// Bodies shared by the dispatch kernels below and their megakernel adapters: element `gid` (the
// dispatch's [[thread_position_in_grid]]; exact grids, no bounds check).
template <typename RP, typename DP>
METAL_FUNC void residual_add_body(RP residual, DP delta, uint gid) {
    residual[gid] = residual[gid] + delta[gid];
}
template <typename T>
MK_FUNC void mk_residual_add(thread const MkStep& s, MkLane l, threadgroup uchar*) {
    if (!l.live) return;
    residual_add_body((mk_ptr<T>)s.addr[0], (mk_cptr<T>)s.addr[1], mk_thread_in_grid(l).x);
}

#ifndef MK_BODIES_ONLY
#define INST_RESIDUAL_ADD(tag, T)                                              \
  kernel void residual_add_##tag##_specialized(                                \
      device T* residual [[buffer(0)]],                                         \
      device const T* delta [[buffer(1)]],                                      \
      uint gid [[thread_position_in_grid]]) {                                   \
    residual_add_body(residual, delta, gid);                                    \
  }
#else
#define INST_RESIDUAL_ADD(tag, T) \
  MK_TAIL(residual_add_##tag##_specialized, (mk_residual_add<T>), MK_NO_CONSTS)
#endif
INST_RESIDUAL_ADD(f16, half)
INST_RESIDUAL_ADD(bf16, bfloat)

// ── Multimodal embed splice: scatter vision embeddings into the text
// embedding stream ───────────────────────────────────────────────────
//
// `Instruction::SpliceMmEmbeds` (interpreter/metal/lowering.rs). For each
// source row `s` of `mm` (the projected vision output), copy it into text
// embedding row `dst_rows[s]`; `dst_rows[s] == 0xFFFFFFFF` skips (text-only
// batches and the padding tail past `total_mm` are all marked skip).
// One thread per element; `embed` is in/out (buffer 0). Dispatched over
// `num_tokens * hidden` (m_scaling shrinks from the baked `bucket_m *
// hidden` to the live num_tokens), so `s < num_tokens` always indexes
// `dst_rows`.
template <typename T, typename EP, typename CU>
METAL_FUNC void mm_embed_splice_body(EP embed, device const T* mm, device const uint* dst_rows,
                                     CU hidden, uint gid) {
    uint s = gid / hidden;
    uint c = gid % hidden;
    uint dst = dst_rows[s];
    if (dst == 0xFFFFFFFFu) return;
    embed[dst * hidden + c] = mm[s * hidden + c];
}
template <typename T>
MK_FUNC void mk_mm_embed_splice(thread const MkStep& s, MkLane l, threadgroup uchar*) {
    if (!l.live) return;
    mm_embed_splice_body<T, mk_ptr<T>, const device uint&>(
        (mk_ptr<T>)s.addr[0], (device const T*)s.addr[1], (device const uint*)s.addr[2],
        *(const device uint*)s.addr[3], mk_thread_in_grid(l).x);
}

#ifndef MK_BODIES_ONLY
#define INST_MM_EMBED_SPLICE(tag, T)                                           \
  kernel void mm_embed_splice_##tag(                                           \
      device T* embed [[buffer(0)]],                                            \
      device const T* mm [[buffer(1)]],                                         \
      device const uint* dst_rows [[buffer(2)]],                                \
      constant uint& hidden [[buffer(3)]],                                      \
      uint gid [[thread_position_in_grid]]) {                                   \
    mm_embed_splice_body<T, device T*, constant uint&>(embed, mm, dst_rows, hidden, gid); \
  }
#else
#define INST_MM_EMBED_SPLICE(tag, T) \
  MK_TAIL(mm_embed_splice_##tag, (mk_mm_embed_splice<T>), MK_NO_CONSTS)
#endif
INST_MM_EMBED_SPLICE(f16, half)
INST_MM_EMBED_SPLICE(bf16, bfloat)

#ifndef MK_BODIES_ONLY

// ============================================================================
// Mul: out = a * b
// ============================================================================

kernel void mul_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] * b[gid];
}

kernel void mul_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] * b[gid];
}

// ============================================================================
// Sub: out = a - b
// ============================================================================

kernel void sub_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] - b[gid];
}

kernel void sub_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] - b[gid];
}

// ============================================================================
// ScalarMul: out = scalar * input
// ============================================================================

kernel void scalar_mul_f16(
    device const half* input [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = half(scalar) * input[gid];
}

kernel void scalar_mul_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = bfloat(scalar) * input[gid];
}

// ============================================================================
// BiasAdd: out = input + bias (broadcast bias across last dimension)
// ============================================================================

kernel void bias_add_f16(
    device const half* input [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& num_cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint col = gid % num_cols;
    out[gid] = input[gid] + bias[col];
}

kernel void bias_add_bf16(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* bias [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& num_cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint col = gid % num_cols;
    out[gid] = input[gid] + bias[col];
}

// ── Specialized BiasAdd (function-constant num_cols) ────────────────────────
//
// Pipeline-time num_cols binding so the Metal driver can constant-fold the
// modulus on dispatches of fixed projection width (Q/K/V each ship a separate
// specialized pipeline at lowering time — `function_constant(0)` is the only
// axis). Bound by `Instruction::MetalBiasAdd` via the
// `KernelId::BiasAdd` lowering arm.
#endif // MK_BODIES_ONLY
#define BIAS_ADD_CONSTS(X) X(uint, cols, BIAS_ADD_NUM_COLS, 0)
#ifndef MK_BODIES_ONLY
BIAS_ADD_CONSTS(MK_FC_DECLARE)
struct BiasAddFc {
    BIAS_ADD_CONSTS(MK_FC_ACCESSOR)
};
#endif

// Body shared by the dispatch kernels and the megakernel adapter.
template <typename C, typename IP, typename BP, typename OP>
METAL_FUNC void bias_add_body(IP input, BP bias, OP out, uint gid) {
    out[gid] = input[gid] + bias[gid % C::cols()];
}
template <typename T, typename C>
MK_FUNC void mk_bias_add(thread const MkStep& s, MkLane l, threadgroup uchar*) {
    if (!l.live) return;
    bias_add_body<C>((mk_cptr<T>)s.addr[0], (device const T*)s.addr[1], (mk_ptr<T>)s.addr[2],
                     mk_thread_in_grid(l).x);
}

#ifndef MK_BODIES_ONLY
#define INST_BIAS_ADD(tag, T)                                                  \
  kernel void bias_add_##tag##_specialized(                                    \
      device const T* input [[buffer(0)]],                                      \
      device const T* bias [[buffer(1)]],                                       \
      device T* out [[buffer(2)]],                                              \
      uint gid [[thread_position_in_grid]]) {                                   \
    bias_add_body<BiasAddFc>(input, bias, out, gid);                            \
  }
#else
#define INST_BIAS_ADD(tag, T) \
  MK_TAIL(bias_add_##tag##_specialized, (mk_bias_add<T, MK_C>), BIAS_ADD_CONSTS)
#endif
INST_BIAS_ADD(f16, half)
INST_BIAS_ADD(bf16, bfloat)

#ifndef MK_BODIES_ONLY

// ============================================================================
// TanhSoftCap: out = cap * tanh(input / cap)
// Used in Gemma2 for attention logit capping
// ============================================================================

kernel void tanh_soft_cap_f16(
    device const half* input [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant float& cap [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = float(input[gid]);
    float result = cap * tanh(x / cap);
    out[gid] = half(result);
}

kernel void tanh_soft_cap_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant float& cap [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = float(input[gid]);
    float result = cap * tanh(x / cap);
    out[gid] = bfloat(result);
}

// Specialized ScalarMul: out = in * SCALE with the compile-time
// constant baked as function constant 2 (slots 0/1 belong to
// BIAS_ADD_NUM_COLS / TANH_SOFTCAP_CAP — fn-const indices are
// file-scoped). Used by the `Instruction::ScalarMul` lowering arm
// (Gemma-family embed scaling `embed(...) * sqrt(hidden_size)`).
// First exercised by Gemma4-on-metal — the arm previously referenced
// these symbols without any .metal definition (latent dead arm).
#endif // MK_BODIES_ONLY
#define SCALAR_MUL_CONSTS(X) X(float, scale, SCALAR_MUL_SCALE, 2)
#ifndef MK_BODIES_ONLY
SCALAR_MUL_CONSTS(MK_FC_DECLARE)
struct ScalarMulFc {
    SCALAR_MUL_CONSTS(MK_FC_ACCESSOR)
};
#endif

template <typename T, typename C, typename OP, typename IP>
METAL_FUNC void scalar_mul_body(OP out, IP input, uint gid) {
    out[gid] = T(float(input[gid]) * C::scale());
}
template <typename T, typename C>
MK_FUNC void mk_scalar_mul(thread const MkStep& s, MkLane l, threadgroup uchar*) {
    if (!l.live) return;
    scalar_mul_body<T, C>((mk_ptr<T>)s.addr[0], (mk_cptr<T>)s.addr[1], mk_thread_in_grid(l).x);
}

// ScalarWeightMul: out = in * w[0] — multiply by a loaded [1]-shaped
// weight (Gemma4 `layer_scalar`, applied to the hidden state at the
// end of every decoder layer). Exact-thread dispatch like
// `residual_add_*_specialized`; no function constants.
template <typename T, typename OP, typename IP>
METAL_FUNC void scalar_weight_mul_body(OP out, IP input, device const T* weight, uint gid) {
    out[gid] = T(float(input[gid]) * float(weight[0]));
}
template <typename T>
MK_FUNC void mk_scalar_weight_mul(thread const MkStep& s, MkLane l, threadgroup uchar*) {
    if (!l.live) return;
    scalar_weight_mul_body<T>((mk_ptr<T>)s.addr[0], (mk_cptr<T>)s.addr[1],
                              (device const T*)s.addr[2], mk_thread_in_grid(l).x);
}

#ifndef MK_BODIES_ONLY
#define INST_SCALAR_MUL(tag, T)                                                \
  kernel void scalar_mul_##tag##_specialized(                                  \
      device       T* out    [[buffer(0)]],                                     \
      device const T* input  [[buffer(1)]],                                     \
      uint gid [[thread_position_in_grid]]) {                                   \
    scalar_mul_body<T, ScalarMulFc>(out, input, gid);                           \
  }
#define INST_SCALAR_WEIGHT_MUL(tag, T)                                         \
  kernel void scalar_weight_mul_##tag##_specialized(                           \
      device       T* out    [[buffer(0)]],                                     \
      device const T* input  [[buffer(1)]],                                     \
      device const T* weight [[buffer(2)]],                                     \
      uint gid [[thread_position_in_grid]]) {                                   \
    scalar_weight_mul_body<T>(out, input, weight, gid);                         \
  }
#else
#define INST_SCALAR_MUL(tag, T)                                                             \
  MK_TAIL(scalar_mul_##tag##_specialized, (mk_scalar_mul<T, MK_C>),                 \
             SCALAR_MUL_CONSTS)
#define INST_SCALAR_WEIGHT_MUL(tag, T)                                                      \
  MK_TAIL(scalar_weight_mul_##tag##_specialized, (mk_scalar_weight_mul<T>),        \
             MK_NO_CONSTS)
#endif
INST_SCALAR_MUL(f16, half)
INST_SCALAR_MUL(bf16, bfloat)
INST_SCALAR_WEIGHT_MUL(f16, half)
INST_SCALAR_WEIGHT_MUL(bf16, bfloat)

// Specialized variants: the cap is a per-model compile-time constant
// (`W::FINAL_LOGIT_SOFTCAPPING`), so it bakes into the pipeline as a
// function constant instead of a runtime scalar buffer — same Phase
// 5.B pattern as `bias_add_*_specialized`. Used by the metal
// `Instruction::TanhSoftCap` lowering arm (Gemma2/4 final logit
// softcapping: `out = cap * tanh(x / cap)`). Slot 1: function-constant
// indices are file-scoped and slot 0 belongs to BIAS_ADD_NUM_COLS.
#define TANH_SOFTCAP_CONSTS(X) X(float, cap, TANH_SOFTCAP_CAP, 1)
#ifndef MK_BODIES_ONLY
TANH_SOFTCAP_CONSTS(MK_FC_DECLARE)
struct TanhSoftcapFc {
    TANH_SOFTCAP_CONSTS(MK_FC_ACCESSOR)
};
#endif

template <typename T, typename C, typename IP, typename OP>
METAL_FUNC void tanh_soft_cap_body(IP input, OP out, uint gid) {
    float x = float(input[gid]);
    out[gid] = T(C::cap() * tanh(x / C::cap()));
}
template <typename T, typename C>
MK_FUNC void mk_tanh_soft_cap(thread const MkStep& s, MkLane l, threadgroup uchar*) {
    if (!l.live) return;
    tanh_soft_cap_body<T, C>((mk_cptr<T>)s.addr[0], (mk_ptr<T>)s.addr[1], mk_thread_in_grid(l).x);
}

#ifndef MK_BODIES_ONLY
#define INST_TANH_SOFT_CAP(tag, T)                                             \
  kernel void tanh_soft_cap_##tag##_specialized(                               \
      device const T* input [[buffer(0)]],                                      \
      device T* out [[buffer(1)]],                                              \
      uint gid [[thread_position_in_grid]]) {                                   \
    tanh_soft_cap_body<T, TanhSoftcapFc>(input, out, gid);                      \
  }
#else
#define INST_TANH_SOFT_CAP(tag, T)                                                          \
  MK_TAIL(tanh_soft_cap_##tag##_specialized, (mk_tanh_soft_cap<T, MK_C>),           \
             TANH_SOFTCAP_CONSTS)
#endif
INST_TANH_SOFT_CAP(f16, half)
INST_TANH_SOFT_CAP(bf16, bfloat)
