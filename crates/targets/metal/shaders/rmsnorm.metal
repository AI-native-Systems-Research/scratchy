// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

/// RMSNorm kernel: y = x * weight / sqrt(mean(x^2) + eps)
///
/// Grid: (M, 1, 1) where M = batch_size
/// Threadgroup: (min(N, 1024), 1, 1) where N = hidden_size
kernel void rmsnorm_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    // Compute mean of squares using threadgroup reduction
    threadgroup float shared_sum[1024];
    
    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float val = float(input[gid * N + i]);
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction in shared memory
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Broadcast RMS to all threads
    float rms = sqrt(shared_sum[0] / float(N) + eps);
    
    // Normalize and scale
    for (uint i = tid; i < N; i += tg_size) {
        float val = float(input[gid * N + i]);
        float w = float(weight[i]);
        output[gid * N + i] = half((val / rms) * w);
    }
}

/// Phase 5.B.3 specialized variant: layer-independent params baked
/// in via `[[function_constant(N)]]`, no runtime constants buffer.
/// Index assignments must match `scratchy-target-metal::interpreter::metal::pipelines`:
///   0 = M (uint), 1 = N/HIDDEN_SIZE (uint), 2 = EPS (float).
constant uint  RMSNORM_M             [[function_constant(0)]];
constant uint  RMSNORM_HIDDEN_SIZE   [[function_constant(1)]];
constant float RMSNORM_EPS           [[function_constant(2)]];
// Zero-centered (Gemma / Qwen3.5) RMSNorm: effective gain = weight + offset.
// `offset` = 1.0 for `(1 + weight)` arches, 0.0 for plain RMSNorm.
constant float RMSNORM_WEIGHT_OFFSET [[function_constant(3)]];

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

#define INST_RMSNORM(act_tag, act_type, scale_tag, scale_type)              \
  template [[host_name("rmsnorm_" #act_tag "_s_" #scale_tag "_specialized")]] \
  [[kernel]] decltype(rmsnorm_specialized_impl<act_type, scale_type>)       \
      rmsnorm_specialized_impl<act_type, scale_type>;

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
// disk). Same fn-consts as the weighted variant minus the offset.
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

#define INST_RMSNORM_UNIT(act_tag, act_type)                          \
  template [[host_name("rmsnorm_unit_" #act_tag "_specialized")]]     \
  [[kernel]] decltype(rmsnorm_unit_impl<act_type>)                    \
      rmsnorm_unit_impl<act_type>;

INST_RMSNORM_UNIT(f16,  half)
INST_RMSNORM_UNIT(bf16, bfloat)

/// BF16 variant (uses float16 as Metal doesn't have native bfloat16)
kernel void rmsnorm_bf16(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    threadgroup float shared_sum[1024];
    
    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float val = input[gid * N + i];
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
    
    float rms = sqrt(shared_sum[0] / float(N) + eps);
    
    for (uint i = tid; i < N; i += tg_size) {
        float val = input[gid * N + i];
        float w = weight[i];
        output[gid * N + i] = (val / rms) * w;
    }
}
