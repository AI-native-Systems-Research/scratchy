// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
#include "baked.h"
using namespace metal;

// Constant slots are file-scoped: each kernel family below is defined only in
// a bake that carries its slots (its command's typed constants).

// ============================================================================
// CopyRows: out[i] = in[i]  (flat element-wise copy, bounds-guarded)
//
// Materializes the vision `pixels` runtime extern into an arena tile
// (`Instruction::LoadPixels`). out @ buffer(0), in @ buffer(1); the
// element count is `CopyRowsConstants` (slot 4), so the m_scaling tail and
// any bucket-padding rows are no-ops.
// ============================================================================

#ifdef SCRATCHY_CONSTANT_4
SCRATCHY_CONSTANT(uint, COPY_ROWS_N, 4);

template <typename T>
[[kernel]] void copy_rows(
    device T* out [[buffer(0)]],
    device const T* in [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= COPY_ROWS_N) return;
    out[gid] = in[gid];
}
#endif

SCRATCHY_KERNEL(copy_rows_f16, copy_rows<half>)
SCRATCHY_KERNEL(copy_rows_bf16, copy_rows<bfloat>)

// ── In-place residual add: residual += delta ────────────────────────────────
//
// The `KernelId::Add` lowering arm (`Instruction::Add`, interpreter/metal/
// lowering.rs) binds buffer(0) = residual (in/out) + buffer(1) = delta and
// dispatches token-parallel over `eff_m * width` exact threads (no bounds
// guard needed — same dispatchThreads convention as `bias_add_*_specialized`).
// No constants. First exercised by the Qwen3.5-VL vision tower — the text
// path fuses its residual into `FusedAddRmsNorm`, so the standalone add never
// reached metal before.
template <typename T>
[[kernel]] void residual_add(
    device T* residual [[buffer(0)]],
    device const T* delta [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    residual[gid] = residual[gid] + delta[gid];
}

SCRATCHY_KERNEL(residual_add_f16_specialized, residual_add<half>)
SCRATCHY_KERNEL(residual_add_bf16_specialized, residual_add<bfloat>)

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
// `dst_rows`. The row width is `MmEmbedSpliceConstants` (slot 5).
#ifdef SCRATCHY_CONSTANT_5
SCRATCHY_CONSTANT(uint, MM_EMBED_HIDDEN, 5);

template <typename T>
[[kernel]] void mm_embed_splice(
    device T* embed [[buffer(0)]],
    device const T* mm [[buffer(1)]],
    device const uint* dst_rows [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint s = gid / MM_EMBED_HIDDEN;
    uint c = gid % MM_EMBED_HIDDEN;
    uint dst = dst_rows[s];
    if (dst == 0xFFFFFFFFu) return;
    embed[dst * MM_EMBED_HIDDEN + c] = mm[s * MM_EMBED_HIDDEN + c];
}
#endif

SCRATCHY_KERNEL(mm_embed_splice_f16, mm_embed_splice<half>)
SCRATCHY_KERNEL(mm_embed_splice_bf16, mm_embed_splice<bfloat>)

// ── Specialized BiasAdd (compiled-in num_cols) ──────────────────────────────
//
// The projection width is a compiled-in constant so the compiler folds the
// modulus (Q/K/V each bake their own kernel). Slot 0. Bound by
// `Instruction::MetalBiasAdd` via the `KernelId::BiasAdd` lowering arm.
#ifdef SCRATCHY_CONSTANT_0
SCRATCHY_CONSTANT(uint, BIAS_ADD_NUM_COLS, 0);

template <typename T>
[[kernel]] void bias_add(
    device const T* input [[buffer(0)]],
    device const T* bias [[buffer(1)]],
    device T* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = input[gid] + bias[gid % BIAS_ADD_NUM_COLS];
}
#endif

SCRATCHY_KERNEL(bias_add_f16_specialized, bias_add<half>)
SCRATCHY_KERNEL(bias_add_bf16_specialized, bias_add<bfloat>)

// Specialized ScalarMul: out = in * SCALE, the scale and the element count
// compiled in (`ScalarMulConstants`, slots 2 / 3). Used by the
// `Instruction::ScalarMul` lowering arm (Gemma-family embed scaling
// `embed(...) * sqrt(hidden_size)`).
#ifdef SCRATCHY_CONSTANT_2
SCRATCHY_CONSTANT(float, SCALAR_MUL_SCALE, 2);
// The elements the buffer holds: the dispatch rounds up to whole threadgroups.
SCRATCHY_CONSTANT(uint, SCALAR_MUL_N, 3);

template <typename T>
[[kernel]] void scalar_mul(
    device       T* out    [[buffer(0)]],
    device const T* input  [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= SCALAR_MUL_N) return;
    out[gid] = T(float(input[gid]) * SCALAR_MUL_SCALE);
}
#endif

SCRATCHY_KERNEL(scalar_mul_f16_specialized, scalar_mul<half>)
SCRATCHY_KERNEL(scalar_mul_bf16_specialized, scalar_mul<bfloat>)

// ScalarWeightMul: out = in * w[0] — multiply by a loaded [1]-shaped
// weight (Gemma4 `layer_scalar`, applied to the hidden state at the
// end of every decoder layer). Exact-thread dispatch like
// `residual_add_*_specialized`; no constants.
template <typename T>
[[kernel]] void scalar_weight_mul(
    device       T* out    [[buffer(0)]],
    device const T* input  [[buffer(1)]],
    device const T* weight [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = T(float(input[gid]) * float(weight[0]));
}

SCRATCHY_KERNEL(scalar_weight_mul_f16_specialized, scalar_weight_mul<half>)
SCRATCHY_KERNEL(scalar_weight_mul_bf16_specialized, scalar_weight_mul<bfloat>)

// TanhSoftCap: out = cap * tanh(input / cap). The cap is a per-model
// constant (`W::FINAL_LOGIT_SOFTCAPPING`), compiled in at slot 1. Used by
// the metal `Instruction::TanhSoftCap` lowering arm (Gemma2/4 final logit
// softcapping).
#ifdef SCRATCHY_CONSTANT_1
SCRATCHY_CONSTANT(float, TANH_SOFTCAP_CAP, 1);

template <typename T>
[[kernel]] void tanh_soft_cap(
    device const T* input [[buffer(0)]],
    device T* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = float(input[gid]);
    out[gid] = T(TANH_SOFTCAP_CAP * tanh(x / TANH_SOFTCAP_CAP));
}
#endif

SCRATCHY_KERNEL(tanh_soft_cap_f16_specialized, tanh_soft_cap<half>)
SCRATCHY_KERNEL(tanh_soft_cap_bf16_specialized, tanh_soft_cap<bfloat>)
