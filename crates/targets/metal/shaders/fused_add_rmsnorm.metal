// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
#include "megakernel/mk_common.h"
using namespace metal;

/// Fused Add + RMSNorm kernel: y = rmsnorm(x + residual, weight, eps)
///
/// This fusion eliminates one memory round-trip by computing the residual add
/// and normalization in a single pass. Critical for memory-bound Apple Silicon.
///
/// Pattern: x' = rmsnorm(x + residual, w, eps)
/// Used in: Pre-norm and post-norm residual patterns in transformer layers
///
/// Grid: (M, 1, 1) where M = batch_size
/// Threadgroup: (min(N, 1024), 1, 1) where N = hidden_size
///
/// Outputs:
/// - output: normalized result [M, N]
/// - residual_out: (x + residual) for next layer's residual [M, N]
#ifndef MK_BODIES_ONLY
kernel void fused_add_rmsnorm_f16(
    device const half* input [[buffer(0)]],
    device const half* residual [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device half* output [[buffer(3)]],
    device half* residual_out [[buffer(4)]],  // Optional: pass null if not needed
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    // Pass 1: Compute x + residual and sum of squares
    threadgroup float shared_sum[1024];
    
    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float x_val = float(input[gid * N + i]);
        float r_val = float(residual[gid * N + i]);
        float sum_val = x_val + r_val;
        
        // Write out residual sum if output buffer provided
        if (residual_out != nullptr) {
            residual_out[gid * N + i] = half(sum_val);
        }
        
        local_sum += sum_val * sum_val;
    }
    shared_sum[tid] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction for sum of squares
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Compute RMS normalization factor
    float rms = sqrt(shared_sum[0] / float(N) + eps);
    
    // Pass 2: Normalize and scale with weight
    for (uint i = tid; i < N; i += tg_size) {
        float x_val = float(input[gid * N + i]);
        float r_val = float(residual[gid * N + i]);
        float sum_val = x_val + r_val;
        float w = float(weight[i]);
        output[gid * N + i] = half((sum_val / rms) * w);
    }
}

/// BF16 variant (uses float as Metal doesn't have native bfloat16)
kernel void fused_add_rmsnorm_bf16(
    device const float* input [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* residual_out [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    threadgroup float shared_sum[1024];
    
    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float x_val = input[gid * N + i];
        float r_val = residual[gid * N + i];
        float sum_val = x_val + r_val;
        
        if (residual_out != nullptr) {
            residual_out[gid * N + i] = sum_val;
        }
        
        local_sum += sum_val * sum_val;
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
        float x_val = input[gid * N + i];
        float r_val = residual[gid * N + i];
        float sum_val = x_val + r_val;
        float w = weight[i];
        output[gid * N + i] = (sum_val / rms) * w;
    }
}

#endif // MK_BODIES_ONLY

/// Phase 5.B.3 specialized variant: layer-independent params baked
/// in via `[[function_constant(N)]]`. Index assignments must match
/// `scratchy-target-metal::interpreter::metal::pipelines`:
///   0 = M (uint), 1 = N/HIDDEN_SIZE (uint), 2 = EPS (float).
// FUSED_ARN_WEIGHT_OFFSET: zero-centered (Gemma / Qwen3.5) RMSNorm: effective gain = weight + offset.
#define FUSED_ARN_CONSTS(X)                                                                  \
  X(uint, m, FUSED_ARN_M, 0) X(uint, n, FUSED_ARN_HIDDEN_SIZE, 1) X(float, eps, FUSED_ARN_EPS, 2) \
  X(float, off, FUSED_ARN_WEIGHT_OFFSET, 3)
#ifndef MK_BODIES_ONLY
FUSED_ARN_CONSTS(MK_FC_DECLARE)
// The dispatch kernels' constants, as the bodies read them.
struct FusedArnFc {
  FUSED_ARN_CONSTS(MK_FC_ACCESSOR)
};
#endif // MK_BODIES_ONLY

// Body shared by the dispatch kernel and the megakernel adapter (see `rmsnorm_body`): `residual`
// and `delta` are both read and written in place.
template <typename T_act, typename T_scale, typename C, typename RP, typename DP>
METAL_FUNC void fused_add_rmsnorm_body(RP residual, DP delta, const device T_scale* weight,
                                       uint gid, uint tid, uint tg_size, bool live,
                                       threadgroup float* shared_sum) {
    // Pass 1: residual += delta in place, accumulate sum-of-squares.
    float local_sum = 0.0f;
    if (live) {
        for (uint i = tid; i < C::n(); i += tg_size) {
            float r = float(residual[gid * C::n() + i]);
            float d = float(delta[gid * C::n() + i]);
            float s = r + d;
            residual[gid * C::n() + i] = T_act(s);
            local_sum += s * s;
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

    // Pass 2: write `rmsnorm(residual, weight)` back into `delta`.
    if (live) {
        for (uint i = tid; i < C::n(); i += tg_size) {
            float s = float(residual[gid * C::n() + i]);
            float w = float(weight[i]) + C::off();
            delta[gid * C::n() + i] = T_act((s / rms) * w);
        }
    }
}

// Megakernel adapter: residual (0) and delta (1) device-coherent.
template <typename T_act, typename T_scale, typename C>
MK_FUNC void mk_fused_add_rmsnorm(thread const MkStep& s, MkLane l, threadgroup uchar* tg) {
    fused_add_rmsnorm_body<T_act, T_scale, C>((mk_ptr<T_act>)s.addr[0], (mk_ptr<T_act>)s.addr[1],
                                              (const device T_scale*)s.addr[2], l.tg_pos.x, l.tid,
                                              l.tpg.x, l.live && l.tg_pos.x < C::m(),
                                              (threadgroup float*)mk_region(s, l, tg));
}
#ifndef MK_BODIES_ONLY

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
/// Dispatch: `(M, 1, 1)` threadgroups × `tg_size` threads, cooperative
/// reduction over `HIDDEN_SIZE`.
template <typename T_act, typename T_scale>
[[kernel]] void fused_add_rmsnorm_specialized_impl(
    device       T_act*   residual [[buffer(0)]],
    device       T_act*   delta    [[buffer(1)]],
    device const T_scale* weight   [[buffer(2)]],
    uint gid     [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= FUSED_ARN_M) return;

    threadgroup float shared_sum[1024];
    fused_add_rmsnorm_body<T_act, T_scale, FusedArnFc>(residual, delta, weight, gid, tid, tg_size,
                                                       true, shared_sum);
}

#define INST_FUSED_ARN(act_tag, act_type, scale_tag, scale_type)              \
  template [[host_name("fused_add_rmsnorm_" #act_tag "_s_" #scale_tag         \
                       "_specialized")]]                                      \
  [[kernel]] decltype(fused_add_rmsnorm_specialized_impl<act_type, scale_type>) \
      fused_add_rmsnorm_specialized_impl<act_type, scale_type>;
#else
// Megakernel mode: the same instantiation lines name the adapters (the dispatch kernel's
// `shared_sum[1024]` per virtual threadgroup).
#define INST_FUSED_ARN(act_tag, act_type, scale_tag, scale_type)                          \
  MK_ADAPTER(fused_add_rmsnorm_##act_tag##_s_##scale_tag##_specialized, 4096, 0x3,     \
             (mk_fused_add_rmsnorm<act_type, scale_type, MK_C>), FUSED_ARN_CONSTS)
#endif

// Coverage: T_scale tracks on-disk gain dtype. Llama-3.x ships F16
// gains; Qwen3 family ships BF16. See INST_RMSNORM in `rmsnorm.metal`.
INST_FUSED_ARN(f16,  half,   f16,  half)
INST_FUSED_ARN(bf16, bfloat, f16,  half)
INST_FUSED_ARN(bf16, bfloat, bf16, bfloat)
INST_FUSED_ARN(f16,  half,   bf16, bfloat)

#ifndef MK_BODIES_ONLY

/// Optimized variant with vectorized loads (half4) for better memory bandwidth
/// Requires N to be multiple of 4
kernel void fused_add_rmsnorm_f16_vec4(
    device const half4* input [[buffer(0)]],
    device const half4* residual [[buffer(1)]],
    device const half4* weight [[buffer(2)]],
    device half4* output [[buffer(3)]],
    device half4* residual_out [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N_div4 [[buffer(6)]],  // N / 4
    constant float& eps [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    threadgroup float shared_sum[1024];
    
    float local_sum = 0.0f;
    for (uint i = tid; i < N_div4; i += tg_size) {
        half4 x_val = input[gid * N_div4 + i];
        half4 r_val = residual[gid * N_div4 + i];
        half4 sum_val = x_val + r_val;
        
        if (residual_out != nullptr) {
            residual_out[gid * N_div4 + i] = sum_val;
        }
        
        // Accumulate sum of squares for all 4 elements
        float4 sum_f = float4(sum_val);
        local_sum += dot(sum_f, sum_f);
    }
    shared_sum[tid] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float rms = sqrt(shared_sum[0] / float(N_div4 * 4) + eps);
    
    for (uint i = tid; i < N_div4; i += tg_size) {
        half4 x_val = input[gid * N_div4 + i];
        half4 r_val = residual[gid * N_div4 + i];
        half4 sum_val = x_val + r_val;
        half4 w = weight[i];
        output[gid * N_div4 + i] = half4((float4(sum_val) / rms) * float4(w));
    }
}

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
/// Function constants: FUSED_ARN_{M, HIDDEN_SIZE, EPS, WEIGHT_OFFSET}
/// (same quartet/slots as fused_add_rmsnorm — shared RmsNormConstants).
///
/// Dispatch: `(M, 1, 1)` threadgroups × `tg_size` threads, cooperative
/// reduction over `HIDDEN_SIZE`.
#endif // MK_BODIES_ONLY

// Body shared by the dispatch kernels and the megakernel adapter (see `rmsnorm_body`).
template <typename T_act, typename T_scale, typename C, typename DP, typename RP, typename OP>
METAL_FUNC void norm_add_scalar_mul_body(DP delta, RP residual, OP out,
                                         const device T_scale* gains,
                                         const device T_scale* layer_scalar, uint gid, uint tid,
                                         uint tg_size, bool live, threadgroup float* shared_sum) {
    // Pass 1: sum-of-squares over delta (the norm input).
    float local_sum = 0.0f;
    if (live) {
        for (uint i = tid; i < C::n(); i += tg_size) {
            float d = float(delta[gid * C::n() + i]);
            local_sum += d * d;
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
    float s = float(layer_scalar[0]);

    // Pass 2: out = (normed + residual) * layer_scalar. Round to T_act
    // at EVERY op boundary the unfused chain rounds at (RmsNorm store,
    // Add store, ScalarWeightMul store) — keeps the fused kernel
    // BIT-IDENTICAL to the rmsnorm/add/scalar_weight_mul sequence, so
    // verbatim greedy parity is preserved by construction.
    if (live) {
        for (uint i = tid; i < C::n(); i += tg_size) {
            float d = float(delta[gid * C::n() + i]);
            float r = float(residual[gid * C::n() + i]);
            float w = float(gains[i]) + C::off();
            T_act normed = T_act((d / rms) * w);
            T_act summed = T_act(float(normed) + r);
            out[gid * C::n() + i] = T_act(float(summed) * s);
        }
    }
}

// Megakernel adapter: delta (0), residual (1) and out (2) device-coherent.
template <typename T_act, typename T_scale, typename C>
MK_FUNC void mk_norm_add_scalar_mul(thread const MkStep& s, MkLane l, threadgroup uchar* tg) {
    norm_add_scalar_mul_body<T_act, T_scale, C>(
        (mk_cptr<T_act>)s.addr[0], (mk_cptr<T_act>)s.addr[1], (mk_ptr<T_act>)s.addr[2],
        (const device T_scale*)s.addr[3], (const device T_scale*)s.addr[4], l.tg_pos.x, l.tid,
        l.tpg.x, l.live && l.tg_pos.x < C::m(), (threadgroup float*)mk_region(s, l, tg));
}

#ifndef MK_BODIES_ONLY
template <typename T_act, typename T_scale>
[[kernel]] void norm_add_scalar_mul_impl(
    device const T_act*   delta        [[buffer(0)]],
    device const T_act*   residual     [[buffer(1)]],
    device       T_act*   out          [[buffer(2)]],
    device const T_scale* gains        [[buffer(3)]],
    device const T_scale* layer_scalar [[buffer(4)]],
    uint gid     [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= FUSED_ARN_M) return;

    threadgroup float shared_sum[1024];
    norm_add_scalar_mul_body<T_act, T_scale, FusedArnFc>(delta, residual, out, gains,
                                                         layer_scalar, gid, tid, tg_size, true,
                                                         shared_sum);
}

#define INST_NORM_ADD_SCALAR_MUL(act_tag, act_type, scale_tag, scale_type)    \
  template [[host_name("norm_add_scalar_mul_" #act_tag "_s_" #scale_tag      \
                       "_specialized")]]                                      \
  [[kernel]] decltype(norm_add_scalar_mul_impl<act_type, scale_type>)        \
      norm_add_scalar_mul_impl<act_type, scale_type>;
#else
#define INST_NORM_ADD_SCALAR_MUL(act_tag, act_type, scale_tag, scale_type)                \
  MK_ADAPTER(norm_add_scalar_mul_##act_tag##_s_##scale_tag##_specialized, 4096, 0x7,     \
             (mk_norm_add_scalar_mul<act_type, scale_type, MK_C>), FUSED_ARN_CONSTS)
#endif

INST_NORM_ADD_SCALAR_MUL(f16,  half,   f16,  half)
INST_NORM_ADD_SCALAR_MUL(bf16, bfloat, f16,  half)
INST_NORM_ADD_SCALAR_MUL(bf16, bfloat, bf16, bfloat)
INST_NORM_ADD_SCALAR_MUL(f16,  half,   bf16, bfloat)
