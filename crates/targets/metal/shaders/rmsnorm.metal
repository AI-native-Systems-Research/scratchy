// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
#include "megakernel/mk_common.h"
using namespace metal;

/// RMSNorm kernel: y = x * weight / sqrt(mean(x^2) + eps)
///
/// Grid: (M, 1, 1) where M = batch_size
/// Threadgroup: (min(N, 1024), 1, 1) where N = hidden_size
#ifndef MK_BODIES_ONLY
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
#endif // MK_BODIES_ONLY

/// Phase 5.B.3 specialized variant: layer-independent params baked
/// in via `[[function_constant(N)]]`, no runtime constants buffer.
/// Index assignments must match `scratchy-target-metal::interpreter::metal::pipelines`:
///   0 = M (uint), 1 = N/HIDDEN_SIZE (uint), 2 = EPS (float).
// RMSNORM_WEIGHT_OFFSET: zero-centered (Gemma / Qwen3.5) RMSNorm: effective gain = weight +
// offset. `offset` = 1.0 for `(1 + weight)` arches, 0.0 for plain RMSNorm.
#define RMSNORM_UNIT_CONSTS(X) \
  X(uint, m, RMSNORM_M, 0) X(uint, n, RMSNORM_HIDDEN_SIZE, 1) X(float, eps, RMSNORM_EPS, 2)
#define RMSNORM_CONSTS(X) RMSNORM_UNIT_CONSTS(X) X(float, off, RMSNORM_WEIGHT_OFFSET, 3)
#ifndef MK_BODIES_ONLY
RMSNORM_CONSTS(MK_FC_DECLARE)
// The dispatch kernels' constants, as the bodies read them.
struct RmsnormFc {
  RMSNORM_CONSTS(MK_FC_ACCESSOR)
};
#endif // MK_BODIES_ONLY

// Body shared by the dispatch kernel and the megakernel adapter. `live` is a compile-time `true`
// on the dispatch path (its early return already ran); the megakernel passes whether this thread
// belongs to a real virtual threadgroup, and every thread runs every barrier. `unit` = no gain.
template <typename T_act, typename T_scale, typename C, bool unit, typename OP, typename IP>
METAL_FUNC void rmsnorm_body(OP output, IP input, const device T_scale* weight, uint gid, uint tid,
                             uint tg_size, bool live, threadgroup float* shared_sum) {
    float local_sum = 0.0f;
    if (live) {
        for (uint i = tid; i < C::n(); i += tg_size) {
            float val = float(input[gid * C::n() + i]);
            local_sum += val * val;
        }
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = sqrt(shared_sum[0] / float(C::n()) + C::eps());

    // `x / rms * w`, pinned to the arithmetic the dispatch kernel's element loop computes — the
    // loop-invariant reciprocal of `rms`, then `(x · 1/rms) · w` — so a row of at most `tg_size`
    // elements (one pass, no loop to hoist the reciprocal out of) is not folded back into a
    // division.
    {
#pragma clang fp reassociate(off)
        const float inv = 1.0f / rms;
        if (live) {
            for (uint i = tid; i < C::n(); i += tg_size) {
                float val = float(input[gid * C::n() + i]);
                if constexpr (unit) {
                    output[gid * C::n() + i] = T_act(val * inv);
                } else {
                    float w   = float(weight[i]) + C::off();
                    output[gid * C::n() + i] = T_act((val * inv) * w);
                }
            }
        }
    }
}

// Megakernel adapters: one virtual threadgroup per dispatch row.
template <typename T_act, typename T_scale, typename C>
MK_FUNC void mk_rmsnorm(thread const MkStep& s, MkLane l, threadgroup uchar* tg) {
    rmsnorm_body<T_act, T_scale, C, false>((mk_ptr<T_act>)s.addr[0], (mk_cptr<T_act>)s.addr[1],
                                           (const device T_scale*)s.addr[2], l.tg_pos.x, l.tid,
                                           l.tpg.x, l.live && l.tg_pos.x < C::m(),
                                           (threadgroup float*)mk_region(s, l, tg));
}
template <typename T_act, typename C>
MK_FUNC void mk_rmsnorm_unit(thread const MkStep& s, MkLane l, threadgroup uchar* tg) {
    rmsnorm_body<T_act, T_act, C, true>((mk_ptr<T_act>)s.addr[0], (mk_cptr<T_act>)s.addr[1],
                                        (const device T_act*)nullptr, l.tg_pos.x, l.tid, l.tpg.x,
                                        l.live && l.tg_pos.x < C::m(),
                                        (threadgroup float*)mk_region(s, l, tg));
}
#ifndef MK_BODIES_ONLY

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
    rmsnorm_body<T_act, T_scale, RmsnormFc, false>(output, input, weight, gid, tid, tg_size, true,
                                                   shared_sum);
}

#define INST_RMSNORM(act_tag, act_type, scale_tag, scale_type)              \
  template [[host_name("rmsnorm_" #act_tag "_s_" #scale_tag "_specialized")]] \
  [[kernel]] decltype(rmsnorm_specialized_impl<act_type, scale_type>)       \
      rmsnorm_specialized_impl<act_type, scale_type>;
#else
// Megakernel mode: the same instantiation lines name the adapters. One virtual threadgroup owns
// the dispatch kernel's `shared_sum[1024]`.
#define INST_RMSNORM(act_tag, act_type, scale_tag, scale_type)                  \
  MK_ADAPTER(rmsnorm_##act_tag##_s_##scale_tag##_specialized, 4096,          \
             (mk_rmsnorm<act_type, scale_type, MK_C>), RMSNORM_CONSTS)
#endif

// Coverage: T_scale tracks on-disk gain dtype. Llama-3.x / Qwen2.5 /
// SmolLM mlx-community 4bit ship F16 RMSNorm gains; Qwen3 family ships
// BF16. Both are loaded with `take_keep_dtype` (no loader-side cast),
// so the kernel template must cover both. `W::SCALE_DTYPE` picks the
// arm at lowering time.
INST_RMSNORM(f16,  half,   f16, half)
INST_RMSNORM(bf16, bfloat, f16, half)
INST_RMSNORM(bf16, bfloat, bf16, bfloat)
INST_RMSNORM(f16,  half,   bf16, bfloat)

#ifndef MK_BODIES_ONLY

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
    rmsnorm_body<T_act, T_act, RmsnormFc, true>(output, input, nullptr, gid, tid, tg_size, true,
                                                shared_sum);
}

#define INST_RMSNORM_UNIT(act_tag, act_type)                          \
  template [[host_name("rmsnorm_unit_" #act_tag "_specialized")]]     \
  [[kernel]] decltype(rmsnorm_unit_impl<act_type>)                    \
      rmsnorm_unit_impl<act_type>;
#else
#define INST_RMSNORM_UNIT(act_tag, act_type)                                  \
  MK_ADAPTER(rmsnorm_unit_##act_tag##_specialized, 4096,                   \
             (mk_rmsnorm_unit<act_type, MK_C>), RMSNORM_UNIT_CONSTS)
#endif

INST_RMSNORM_UNIT(f16,  half)
INST_RMSNORM_UNIT(bf16, bfloat)

#ifndef MK_BODIES_ONLY

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
#endif // MK_BODIES_ONLY
