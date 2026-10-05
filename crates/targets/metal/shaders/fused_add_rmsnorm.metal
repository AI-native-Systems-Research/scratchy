// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
#include "baked.h"
#include "row_sum.h"
using namespace metal;

/// `RmsNormConstants`, compiled in: 0 = M (uint), 1 = N/HIDDEN_SIZE (uint), 2 = EPS (float).
SCRATCHY_CONSTANT(uint,  FUSED_ARN_M,             0);
SCRATCHY_CONSTANT(uint,  FUSED_ARN_HIDDEN_SIZE,   1);
SCRATCHY_CONSTANT(float, FUSED_ARN_EPS,           2);
// Zero-centered (Gemma / Qwen3.5) RMSNorm: effective gain = weight + offset.
SCRATCHY_CONSTANT(float, FUSED_ARN_WEIGHT_OFFSET, 3);
// The threads of a row's threadgroup (`NORM_THREADS`): a constant, so a thread's elements — every
// FUSED_ARN_THREADS-th from its index — are a count the compiler knows, and load together.
SCRATCHY_CONSTANT(uint,  FUSED_ARN_THREADS,       4);
constant constexpr uint FUSED_ARN_PER_THREAD =
    (FUSED_ARN_HIDDEN_SIZE + FUSED_ARN_THREADS - 1) / FUSED_ARN_THREADS;

/// Specialized fused add+rmsnorm matching the CUDA `fused_add_rms_norm_inplace`
/// semantics (`scratchy-target-cuda::kernels::fused_add_rms_norm_inplace`):
///   - `residual += delta` in place
///   - `delta` is overwritten with `rmsnorm(residual_after_add, weight, eps)`
///
/// Template form (`<T_act, T_scale>`) — same P10b in-register cast
/// pattern: residual/delta in the activation dtype, weight in its
/// on-disk dtype (F16 for every sampled mlx-community / Llama-3.x
/// checkpoint), all promoted to `float` for the reduction.
///
/// Bindings (must match `interpreter::metal::lowering::lower_one` for
/// `Instruction::FusedAddRmsNorm`):
///   buffer(0) = residual (in/out)
///   buffer(1) = delta    (in/out — overwritten with normed result)
///   buffer(2) = weight   (in; on-disk dtype)
///
/// Dispatch: `(M, 1, 1)` threadgroups × `FUSED_ARN_THREADS` threads,
/// cooperative reduction over `HIDDEN_SIZE`.
template <typename T_act, typename T_scale>
[[kernel]] void fused_add_rmsnorm_specialized_impl(
    device       T_act*   residual [[buffer(0)]],
    device       T_act*   delta    [[buffer(1)]],
    device const T_scale* weight   [[buffer(2)]],
    uint gid     [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]]
) {
    if (gid >= FUSED_ARN_M) return;

    threadgroup float shared_sum[FUSED_ARN_THREADS];
    device T_act* res = residual + gid * FUSED_ARN_HIDDEN_SIZE;
    device T_act* del = delta + gid * FUSED_ARN_HIDDEN_SIZE;

    // Pass 1: residual += delta in place, accumulate sum-of-squares. Pass 2
    // reads each sum back as stored — from registers.
    float stored[FUSED_ARN_PER_THREAD];
    float local_sum = 0.0f;
    for (uint k = 0; k < FUSED_ARN_PER_THREAD; ++k) {
        const uint i = tid + k * FUSED_ARN_THREADS;
        if (i < FUSED_ARN_HIDDEN_SIZE) {
            float s = float(res[i]) + float(del[i]);
            const T_act t = T_act(s);
            res[i] = t;
            stored[k] = float(t);
            local_sum += s * s;
        }
    }
    const float sum_sq = row_sum(local_sum, tid, FUSED_ARN_THREADS, shared_sum);

    float rms = sqrt(sum_sq / float(FUSED_ARN_HIDDEN_SIZE) + FUSED_ARN_EPS);

    // Pass 2: write `rmsnorm(residual, weight)` back into `delta`.
    for (uint k = 0; k < FUSED_ARN_PER_THREAD; ++k) {
        const uint i = tid + k * FUSED_ARN_THREADS;
        if (i < FUSED_ARN_HIDDEN_SIZE) {
            float w = float(weight[i]) + FUSED_ARN_WEIGHT_OFFSET;
            del[i] = T_act((stored[k] / rms) * w);
        }
    }
}

#define INST_FUSED_ARN(act_tag, act_type, scale_tag, scale_type)              \
  SCRATCHY_KERNEL(fused_add_rmsnorm_##act_tag##_s_##scale_tag##_specialized,  \
                  fused_add_rmsnorm_specialized_impl<act_type, scale_type>)

// Coverage: T_scale tracks on-disk gain dtype. Llama-3.x ships F16
// gains; Qwen3 family ships BF16. See INST_RMSNORM in `rmsnorm.metal`.
INST_FUSED_ARN(f16,  half,   f16,  half)
INST_FUSED_ARN(bf16, bfloat, f16,  half)
INST_FUSED_ARN(bf16, bfloat, bf16, bfloat)
INST_FUSED_ARN(f16,  half,   bf16, bfloat)

/// Gemma4 post-FFN tail, fused (the norm-THEN-add mirror of
/// `fused_add_rmsnorm_specialized_impl`):
///
///   out = (rmsnorm(delta, gains, eps) + residual) * layer_scalar
///
/// i.e. the DSL chain `post_ffwd_normed = rmsnorm(down, post_ffwd_ln);
/// hidden = add(post_ffwd_normed, hidden); hidden = scalar_weight_mul(
/// hidden, layer_scalar[layer])` in one dispatch. Reduction runs over
/// `delta` only (the residual is NOT part of the norm — order differs
/// from FusedAddRmsNorm). No in-place writes: `out` is a distinct slot.
///
/// Bindings (must match `interpreter::metal::lowering` for
/// `Instruction::NormAddScalarMul`):
///   buffer(0) = delta        (in; norm input — down-proj output)
///   buffer(1) = residual     (in)
///   buffer(2) = out          (out; the new hidden_states)
///   buffer(3) = gains        (in; [hidden], on-disk dtype)
///   buffer(4) = layer_scalar (in; [1],     on-disk dtype)
///
/// Constants: FUSED_ARN_{M, HIDDEN_SIZE, EPS, WEIGHT_OFFSET}
/// (same quartet/slots as fused_add_rmsnorm — shared RmsNormConstants).
///
/// Dispatch: `(M, 1, 1)` threadgroups × `FUSED_ARN_THREADS` threads,
/// cooperative reduction over `HIDDEN_SIZE`.
template <typename T_act, typename T_scale>
[[kernel]] void norm_add_scalar_mul_impl(
    device const T_act*   delta        [[buffer(0)]],
    device const T_act*   residual     [[buffer(1)]],
    device       T_act*   out          [[buffer(2)]],
    device const T_scale* gains        [[buffer(3)]],
    device const T_scale* layer_scalar [[buffer(4)]],
    uint gid     [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]]
) {
    if (gid >= FUSED_ARN_M) return;

    threadgroup float shared_sum[FUSED_ARN_THREADS];

    // Pass 1: sum-of-squares over delta (the norm input), kept for pass 2.
    float dv[FUSED_ARN_PER_THREAD];
    float local_sum = 0.0f;
    for (uint k = 0; k < FUSED_ARN_PER_THREAD; ++k) {
        const uint i = tid + k * FUSED_ARN_THREADS;
        if (i < FUSED_ARN_HIDDEN_SIZE) {
            dv[k] = float(delta[gid * FUSED_ARN_HIDDEN_SIZE + i]);
            local_sum += dv[k] * dv[k];
        }
    }
    const float sum_sq = row_sum(local_sum, tid, FUSED_ARN_THREADS, shared_sum);

    float rms = sqrt(sum_sq / float(FUSED_ARN_HIDDEN_SIZE) + FUSED_ARN_EPS);
    float s = float(layer_scalar[0]);

    // Pass 2: out = (normed + residual) * layer_scalar. Round to T_act
    // at EVERY op boundary the unfused chain rounds at (RmsNorm store,
    // Add store, ScalarWeightMul store) — keeps the fused kernel
    // BIT-IDENTICAL to the rmsnorm/add/scalar_weight_mul sequence, so
    // verbatim greedy parity is preserved by construction.
    for (uint k = 0; k < FUSED_ARN_PER_THREAD; ++k) {
        const uint i = tid + k * FUSED_ARN_THREADS;
        if (i < FUSED_ARN_HIDDEN_SIZE) {
            float r = float(residual[gid * FUSED_ARN_HIDDEN_SIZE + i]);
            float w = float(gains[i]) + FUSED_ARN_WEIGHT_OFFSET;
            T_act normed = T_act((dv[k] / rms) * w);
            T_act summed = T_act(float(normed) + r);
            out[gid * FUSED_ARN_HIDDEN_SIZE + i] = T_act(float(summed) * s);
        }
    }
}

#define INST_NORM_ADD_SCALAR_MUL(act_tag, act_type, scale_tag, scale_type)    \
  SCRATCHY_KERNEL(norm_add_scalar_mul_##act_tag##_s_##scale_tag##_specialized, \
                  norm_add_scalar_mul_impl<act_type, scale_type>)

INST_NORM_ADD_SCALAR_MUL(f16,  half,   f16,  half)
INST_NORM_ADD_SCALAR_MUL(bf16, bfloat, f16,  half)
INST_NORM_ADD_SCALAR_MUL(bf16, bfloat, bf16, bfloat)
INST_NORM_ADD_SCALAR_MUL(f16,  half,   bf16, bfloat)
