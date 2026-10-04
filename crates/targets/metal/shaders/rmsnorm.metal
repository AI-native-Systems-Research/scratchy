// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
#include "baked.h"
using namespace metal;

/// `RmsNormConstants`, compiled in: 0 = M (uint), 1 = N/HIDDEN_SIZE (uint), 2 = EPS (float).
SCRATCHY_CONSTANT(uint,  RMSNORM_M,             0);
SCRATCHY_CONSTANT(uint,  RMSNORM_HIDDEN_SIZE,   1);
SCRATCHY_CONSTANT(float, RMSNORM_EPS,           2);
// Zero-centered (Gemma / Qwen3.5) RMSNorm: effective gain = weight + offset.
// `offset` = 1.0 for `(1 + weight)` arches, 0.0 for plain RMSNorm.
SCRATCHY_CONSTANT(float, RMSNORM_WEIGHT_OFFSET, 3);

// Template form (`<T_act, T_scale>`): same in-register cast pattern as
// the affine quant kernels (`shaders/quantized_*.metal`). The kernel
// reads activations through a `T_act` device pointer, the gain through
// a separate `T_scale` device pointer, and promotes both to `float`
// for the reduction. Lets the loader keep RMSNorm gains in their
// on-disk dtype (F16 for every sampled mlx-community / Llama-3.x
// checkpoint) instead of F16→BF16 truncating at load on the bf16
// stack — the same regression repair P10b applied to quant scales.
//
// Bindings (must match `interpreter::metal::lowering::lower_one` for
// `Instruction::RmsNorm`):
//   buffer(0) = output (out_slot — written)
//   buffer(1) = input  (in_slot  — read)
//   buffer(2) = weight (read; on-disk dtype)
template <typename T_act, typename T_scale>
[[kernel]] void rmsnorm_specialized_impl(
    device       T_act*   output [[buffer(0)]],
    device const T_act*   input  [[buffer(1)]],
    device const T_scale* weight [[buffer(2)]],
    uint gid     [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= RMSNORM_M) return;

    threadgroup float shared_sum[1024];

    float local_sum = 0.0f;
    for (uint i = tid; i < RMSNORM_HIDDEN_SIZE; i += tg_size) {
        float val = float(input[gid * RMSNORM_HIDDEN_SIZE + i]);
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = sqrt(shared_sum[0] / float(RMSNORM_HIDDEN_SIZE) + RMSNORM_EPS);

    for (uint i = tid; i < RMSNORM_HIDDEN_SIZE; i += tg_size) {
        float val = float(input[gid * RMSNORM_HIDDEN_SIZE + i]);
        float w   = float(weight[i]) + RMSNORM_WEIGHT_OFFSET;
        output[gid * RMSNORM_HIDDEN_SIZE + i] = T_act((val / rms) * w);
    }
}

#define INST_RMSNORM(act_tag, act_type, scale_tag, scale_type)                  \
  SCRATCHY_KERNEL(rmsnorm_##act_tag##_s_##scale_tag##_specialized,               \
                  rmsnorm_specialized_impl<act_type, scale_type>)

// Coverage: T_scale tracks on-disk gain dtype. Llama-3.x / Qwen2.5 /
// SmolLM mlx-community 4bit ship F16 RMSNorm gains; Qwen3 family ships
// BF16. Both are loaded with `take_keep_dtype` (no loader-side cast),
// so the kernel template must cover both. `W::SCALE_DTYPE` picks the
// arm at lowering time.
INST_RMSNORM(f16,  half,   f16, half)
INST_RMSNORM(bf16, bfloat, f16, half)
INST_RMSNORM(bf16, bfloat, bf16, bfloat)
INST_RMSNORM(f16,  half,   bf16, bfloat)

// Unit-gain RMSNorm — no learnable scale (gain ≡ 1, no weight buffer).
// Faithful port of mlx `RMSNormNoScale` (Gemma4 `v_norm`: V is
// rms-normalized per head before the cache write, with NO weights on
// disk). Same constants as the weighted variant minus the offset.
template <typename T_act>
[[kernel]] void rmsnorm_unit_impl(
    device       T_act* output [[buffer(0)]],
    device const T_act* input  [[buffer(1)]],
    uint gid     [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (gid >= RMSNORM_M) return;

    // Same tree reduction as `rmsnorm_specialized_impl` — identical
    // accumulation order keeps the two norms bit-consistent.
    threadgroup float shared_sum[1024];

    float local_sum = 0.0f;
    for (uint i = tid; i < RMSNORM_HIDDEN_SIZE; i += tg_size) {
        float val = float(input[gid * RMSNORM_HIDDEN_SIZE + i]);
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = sqrt(shared_sum[0] / float(RMSNORM_HIDDEN_SIZE) + RMSNORM_EPS);
    for (uint i = tid; i < RMSNORM_HIDDEN_SIZE; i += tg_size) {
        float val = float(input[gid * RMSNORM_HIDDEN_SIZE + i]);
        output[gid * RMSNORM_HIDDEN_SIZE + i] = T_act(val / rms);
    }
}

#define INST_RMSNORM_UNIT(act_tag, act_type)                                  \
  SCRATCHY_KERNEL(rmsnorm_unit_##act_tag##_specialized, rmsnorm_unit_impl<act_type>)

INST_RMSNORM_UNIT(f16,  half)
INST_RMSNORM_UNIT(bf16, bfloat)
