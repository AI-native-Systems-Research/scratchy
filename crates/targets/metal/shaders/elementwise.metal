// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

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
kernel void residual_add_f16_specialized(
    device half* residual [[buffer(0)]],
    device const half* delta [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    residual[gid] = residual[gid] + delta[gid];
}

kernel void residual_add_bf16_specialized(
    device bfloat* residual [[buffer(0)]],
    device const bfloat* delta [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    residual[gid] = residual[gid] + delta[gid];
}

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
kernel void mm_embed_splice_f16(
    device half* embed [[buffer(0)]],
    device const half* mm [[buffer(1)]],
    device const uint* dst_rows [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint s = gid / hidden;
    uint c = gid % hidden;
    uint dst = dst_rows[s];
    if (dst == 0xFFFFFFFFu) return;
    embed[dst * hidden + c] = mm[s * hidden + c];
}

kernel void mm_embed_splice_bf16(
    device bfloat* embed [[buffer(0)]],
    device const bfloat* mm [[buffer(1)]],
    device const uint* dst_rows [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint s = gid / hidden;
    uint c = gid % hidden;
    uint dst = dst_rows[s];
    if (dst == 0xFFFFFFFFu) return;
    embed[dst * hidden + c] = mm[s * hidden + c];
}

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
constant uint BIAS_ADD_NUM_COLS [[function_constant(0)]];

kernel void bias_add_f16_specialized(
    device const half* input [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = input[gid] + bias[gid % BIAS_ADD_NUM_COLS];
}

kernel void bias_add_bf16_specialized(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* bias [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = input[gid] + bias[gid % BIAS_ADD_NUM_COLS];
}

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
constant float SCALAR_MUL_SCALE [[function_constant(2)]];

kernel void scalar_mul_f16_specialized(
    device       half* out    [[buffer(0)]],
    device const half* input  [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = half(float(input[gid]) * SCALAR_MUL_SCALE);
}

kernel void scalar_mul_bf16_specialized(
    device       bfloat* out    [[buffer(0)]],
    device const bfloat* input  [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = bfloat(float(input[gid]) * SCALAR_MUL_SCALE);
}

// ScalarWeightMul: out = in * w[0] — multiply by a loaded [1]-shaped
// weight (Gemma4 `layer_scalar`, applied to the hidden state at the
// end of every decoder layer). Exact-thread dispatch like
// `residual_add_*_specialized`; no function constants.
kernel void scalar_weight_mul_f16_specialized(
    device       half* out    [[buffer(0)]],
    device const half* input  [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = half(float(input[gid]) * float(weight[0]));
}

kernel void scalar_weight_mul_bf16_specialized(
    device       bfloat* out    [[buffer(0)]],
    device const bfloat* input  [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = bfloat(float(input[gid]) * float(weight[0]));
}

// Specialized variants: the cap is a per-model compile-time constant
// (`W::FINAL_LOGIT_SOFTCAPPING`), so it bakes into the pipeline as a
// function constant instead of a runtime scalar buffer — same Phase
// 5.B pattern as `bias_add_*_specialized`. Used by the metal
// `Instruction::TanhSoftCap` lowering arm (Gemma2/4 final logit
// softcapping: `out = cap * tanh(x / cap)`). Slot 1: function-constant
// indices are file-scoped and slot 0 belongs to BIAS_ADD_NUM_COLS.
constant float TANH_SOFTCAP_CAP [[function_constant(1)]];

kernel void tanh_soft_cap_f16_specialized(
    device const half* input [[buffer(0)]],
    device half* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = float(input[gid]);
    out[gid] = half(TANH_SOFTCAP_CAP * tanh(x / TANH_SOFTCAP_CAP));
}

kernel void tanh_soft_cap_bf16_specialized(
    device const bfloat* input [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = float(input[gid]);
    out[gid] = bfloat(TANH_SOFTCAP_CAP * tanh(x / TANH_SOFTCAP_CAP));
}
