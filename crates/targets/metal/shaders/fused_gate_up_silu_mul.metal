// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// MLX's pragma helpers from `mlx/backend/metal/kernels/utils.h`.
#ifndef MLX_MTL_PRAGMA_UNROLL
#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#endif

/// Fused Gate-Up-SiLU-Mul kernel for SwiGLU activation
///
/// Pattern: silu(gate_proj(x)) * up_proj(x)
/// Where: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
///
/// This fusion eliminates memory round-trips by computing the activation
/// in a single pass. Critical for memory-bound MLP layers on Apple Silicon.
///
/// Two variants:
/// 1. Separate GEMMs: Takes gate_out and up_out as inputs
/// 2. Fused GEMM: Takes concatenated gate_up output [B, 2*I] as input
///
/// Grid: (M, 1, 1) where M = batch_size
/// Threadgroup: (min(N, 1024), 1, 1) where N = intermediate_size

/// SiLU activation: x * sigmoid(x)
inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

/// Variant 1: Separate gate and up outputs
/// Input: gate_out [M, N], up_out [M, N]
/// Output: silu(gate_out) * up_out [M, N]
kernel void fused_gate_up_silu_mul_f16(
    device const half* gate_out [[buffer(0)]],
    device const half* up_out [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N; i += tg_size) {
        float gate = float(gate_out[gid * N + i]);
        float up = float(up_out[gid * N + i]);
        output[gid * N + i] = half(silu(gate) * up);
    }
}

/// Variant 2: Fused gate_up output (concatenated in column dimension)
/// Input: gate_up [M, 2*N] where first N columns are gate, next N are up
/// Output: silu(gate) * up [M, N]
kernel void fused_gate_up_silu_mul_concat_f16(
    device const half* gate_up [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& M [[buffer(2)]],
    constant uint& N [[buffer(3)]],  // intermediate_size (not 2*N)
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N; i += tg_size) {
        float gate = float(gate_up[gid * (2 * N) + i]);
        float up = float(gate_up[gid * (2 * N) + N + i]);
        output[gid * N + i] = half(silu(gate) * up);
    }
}

/// BF16 variant - separate outputs
kernel void fused_gate_up_silu_mul_bf16(
    device const float* gate_out [[buffer(0)]],
    device const float* up_out [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N; i += tg_size) {
        float gate = gate_out[gid * N + i];
        float up = up_out[gid * N + i];
        output[gid * N + i] = silu(gate) * up;
    }
}

/// BF16 variant - concatenated input
kernel void fused_gate_up_silu_mul_concat_bf16(
    device const float* gate_up [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& M [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N; i += tg_size) {
        float gate = gate_up[gid * (2 * N) + i];
        float up = gate_up[gid * (2 * N) + N + i];
        output[gid * N + i] = silu(gate) * up;
    }
}

/// Vectorized variant (half4) for better memory bandwidth
/// Requires N to be multiple of 4
kernel void fused_gate_up_silu_mul_f16_vec4(
    device const half4* gate_out [[buffer(0)]],
    device const half4* up_out [[buffer(1)]],
    device half4* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N_div4 [[buffer(4)]],  // N / 4
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N_div4; i += tg_size) {
        half4 gate = gate_out[gid * N_div4 + i];
        half4 up = up_out[gid * N_div4 + i];
        
        // Apply SiLU element-wise
        float4 gate_f = float4(gate);
        float4 up_f = float4(up);
        float4 result;
        result.x = silu(gate_f.x) * up_f.x;
        result.y = silu(gate_f.y) * up_f.y;
        result.z = silu(gate_f.z) * up_f.z;
        result.w = silu(gate_f.w) * up_f.w;
        
        output[gid * N_div4 + i] = half4(result);
    }
}

/// Vectorized concatenated variant
kernel void fused_gate_up_silu_mul_concat_f16_vec4(
    device const half4* gate_up [[buffer(0)]],
    device half4* output [[buffer(1)]],
    constant uint& M [[buffer(2)]],
    constant uint& N_div4 [[buffer(3)]],  // N / 4
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N_div4; i += tg_size) {
        half4 gate = gate_up[gid * (2 * N_div4) + i];
        half4 up = gate_up[gid * (2 * N_div4) + N_div4 + i];
        
        float4 gate_f = float4(gate);
        float4 up_f = float4(up);
        float4 result;
        result.x = silu(gate_f.x) * up_f.x;
        result.y = silu(gate_f.y) * up_f.y;
        result.z = silu(gate_f.z) * up_f.z;
        result.w = silu(gate_f.w) * up_f.w;
        
        output[gid * N_div4 + i] = half4(result);
    }
}

/// GELU variant for Gemma2/3 models
/// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
inline float gelu_approx(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    // Clamp: fast-math tanh = (exp(2x)-1)/(exp(2x)+1) → NaN once
    // exp overflows (|inner| ≳ 44, i.e. any gate ≥ ~10.06). tanh(15)
    // rounds to exactly 1.0f — bit-exact vs a saturating tanh.
    float inner = clamp(sqrt_2_over_pi * (x + coeff * x3), -15.0f, 15.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

/// Fused Gate-Up-GELU-Mul for Gemma models
kernel void fused_gate_up_gelu_mul_f16(
    device const half* gate_out [[buffer(0)]],
    device const half* up_out [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (gid >= M) return;
    
    for (uint i = tid; i < N; i += tg_size) {
        float gate = float(gate_out[gid * N + i]);
        float up = float(up_out[gid * N + i]);
        output[gid * N + i] = half(gelu_approx(gate) * up);
    }
}

// Note: Metal does not have erf() function, so exact GELU is not available
// Use approximate GELU instead (gelu_approx above)

// ============================================================================
// fused_gate_up_silu_mul_decode_f16_specialized
// ============================================================================
//
// M=1 fast path, fused gate+up SwiGLU. Direct port of MLX's
// GEMVKernel from `mlx/backend/metal/kernels/gemv.metal`
// (BM=1, BN=8, SM=1, SN=32, TM=4, TN=4) extended to compute
// `output = silu(gate_w @ input) * (up_w @ input)` in one pass.
//
// Per-thread contract (matches MLX exactly):
//   - Threadgroup: (BN*SN, BM*SM, 1) = (256, 1, 1) threads = 8
//     simdgroups × 32 lanes.
//   - Per threadgroup output: blockM = BM*SM*TM = 4 output rows.
//   - Per threadgroup K chunk: blockN = BN*SN*TN = 1024 K-elements
//     per main-loop iteration.
//   - Each thread accumulates TM=4 outputs over the K dimension,
//     reading TN=4 input + TN=4 gate-weight + TN=4 up-weight elements
//     per inner iter. Threadgroup-memory reduction across BN
//     simdgroups happens at the end.
//
// Bindings (must match the matrix variant exactly so the lowering
// stays kernel-agnostic):
//   buffer(0) = output [1, N]
//   buffer(1) = input  [1, K]
//   buffer(2) = weight [2*N, K]   packed [gate; up]
//
// Function constants:
//   3 = M (uint) — must be 1 for this kernel; defensive early-out.
//   4 = N (uint) — INTERMEDIATE_SIZE.
//   5 = K (uint) — Q_SIZE / hidden_size.
//
// Threadgroup grid: ((N + blockM - 1) / blockM, 1, 1) =
// ((N + 3) / 4, 1, 1) for the 4-output blockM. For TinyLlama
// N=5632 that's 1408 threadgroups; each threadgroup runs ~K/blockN
// = K/1024 main-loop iterations (= 2 for K=2048) plus the simdgroup
// reduction.
//
// Empirically validated against the MLX reference; copy faithful to
// the upstream kernel except for the fusion epilogue (silu(gate)*up
// instead of writing both rows separately) and the function-constant
// shape arguments instead of MLX's runtime constants.

constant uint FUSED_MLP_DECODE_M [[function_constant(3)]];
constant uint FUSED_MLP_DECODE_N [[function_constant(4)]];
constant uint FUSED_MLP_DECODE_K [[function_constant(5)]];
// GELU (Gemma GeGLU) vs SiLU (SwiGLU) activation selector. Default
// false (SiLU) so existing FusedGateUpSiluMul stays bit-identical.
constant bool FUSED_MLP_DECODE_IS_GELU [[function_constant(9)]];

kernel void fused_gate_up_silu_mul_decode_f16_specialized(
    device       half* output  [[buffer(0)]],   // [1, N]
    device const half* input   [[buffer(1)]],   // [1, K]
    device const half* weight  [[buffer(2)]],   // [2*N, K] packed [gate; up]
    uint3 tid     [[threadgroup_position_in_grid]],
    uint3 lid     [[thread_position_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    if (FUSED_MLP_DECODE_M != 1u) return;

    // Compile-time params chosen from MLX's `instantiate_gemv_blocks`
    // standard non-edge case: `instantiate_gemv(name, itype, 1, 8, 1, 32, 4, 4)`.
    constexpr int BM = 1;
    constexpr int BN = 8;
    constexpr int SM = 1;
    constexpr int SN = 32;
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int threadsM = BM * SM;        // 1
    constexpr int threadsN = BN * SN;        // 256
    constexpr int blockM   = threadsM * TM;  // 4
    constexpr int blockN   = threadsN * TN;  // 1024

    const uint N = FUSED_MLP_DECODE_N;
    const uint K = FUSED_MLP_DECODE_K;
    const uint matrix_ld = K;

    // Per-thread accumulators (TM outputs per thread).
    thread float gate_result[TM] = {0};
    thread float up_result  [TM] = {0};
    thread half  in_buf [TN];
    thread half  gate_buf[TN];
    thread half  up_buf  [TN];

    const int thrM = SN != 32 ? int(simd_lid) / SN : 0;
    const int thrN = SN != 32 ? int(simd_lid) % SN : int(simd_lid);

    const int sgN = BN != 1 ? int(simd_gid) % BN : 0;
    const int simdM = BN != 1 ? SM * (int(simd_gid) / BN) : SM * int(simd_gid);
    const int simdN = BN != 1 ? SN * (int(simd_gid) % BN) : 0;

    int bm = (simdM + thrM) * TM;
    int bn = (simdN + thrN) * TN;

    // Block position: which output rows this threadgroup is computing.
    int out_row = int(tid.x) * blockM + bm;
    if (out_row >= int(N)) return;

    // Adjust the tail simdgroup so the last threadgroup's writes stay
    // in bounds (matches MLX's edge handling).
    const int N_int = int(N);
    out_row = out_row + TM <= N_int ? out_row : N_int - TM;

    // Pointer pair: gate row out_row, up row N + out_row.
    device const half* gate_mat = weight + uint(out_row) * matrix_ld;
    device const half* up_mat   = weight + (N + uint(out_row)) * matrix_ld;

    // Loop over K in blocks of blockN = 1024.
    const int K_int = int(K);
    const int n_iter = K_int / blockN;
    const int last_iter = blockN * n_iter;
    const int leftover = K_int - last_iter;

    for (int i = 0; i < n_iter; ++i) {
        // Load TN input elements for this thread's K-slice.
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
            in_buf[tn] = input[bn + tn];
        }

        int mat_offset = 0;
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
            // Load TN gate-weight + TN up-weight elements for row tm.
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
                gate_buf[tn] = gate_mat[mat_offset + bn + tn];
                up_buf[tn]   = up_mat  [mat_offset + bn + tn];
            }
            // Accumulate.
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
                gate_result[tm] += float(gate_buf[tn]) * float(in_buf[tn]);
                up_result[tm]   += float(up_buf[tn])   * float(in_buf[tn]);
            }
            mat_offset += int(matrix_ld);
        }

        bn += blockN;
    }

    if (leftover > 0) {
        // Bounds-checked tail — copy MLX's load_safe pattern inline.
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
            in_buf[tn] = (bn + tn < K_int) ? input[bn + tn] : half(0);
        }
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
                gate_buf[tn] = (bn + tn < K_int)
                    ? gate_mat[tm * int(matrix_ld) + bn + tn]
                    : half(0);
                up_buf[tn] = (bn + tn < K_int)
                    ? up_mat  [tm * int(matrix_ld) + bn + tn]
                    : half(0);
            }
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
                gate_result[tm] += float(gate_buf[tn]) * float(in_buf[tn]);
                up_result[tm]   += float(up_buf[tn])   * float(in_buf[tn]);
            }
        }
    }

    // Simdgroup reduction (32-lane shuffle-down).
    MLX_MTL_PRAGMA_UNROLL
    for (int tm = 0; tm < TM; tm++) {
        MLX_MTL_PRAGMA_UNROLL
        for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
            gate_result[tm] += simd_shuffle_down(gate_result[tm], sn);
            up_result[tm]   += simd_shuffle_down(up_result[tm], sn);
        }
    }

    // Threadgroup reduction across BN=8 simdgroups (only sgN=0 will
    // hold the final partials and write outputs).
    threadgroup float tgp_gate[BN * (blockM + TM)];
    threadgroup float tgp_up  [BN * (blockM + TM)];

    threadgroup float* gate_results = tgp_gate + sgN * (blockM + TM) + bm;
    threadgroup float* up_results   = tgp_up   + sgN * (blockM + TM) + bm;
    if (thrN == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
            gate_results[tm] = gate_result[tm];
            up_results[tm]   = up_result[tm];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgN == 0 && thrN == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int sgn = 1; sgn < BN; sgn++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tm = 0; tm < TM; tm++) {
                gate_result[tm] += tgp_gate[sgn * (blockM + TM) + bm + tm];
                up_result[tm]   += tgp_up  [sgn * (blockM + TM) + bm + tm];
            }
        }

        // SwiGLU epilogue + write.
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
            const float g = gate_result[tm];
            const float u = up_result[tm];
            const float act = FUSED_MLP_DECODE_IS_GELU
                ? gelu_approx(g)
                : g / (1.0f + exp(-g));
            output[out_row + tm] = half(act * u);
        }
    }
}

/// BF16 specialized variant of the M=1 decode fused MLP. Direct
/// translation of `..._decode_f16_specialized` with `bfloat` device
/// pointers and `bfloat` thread-local buffers; accumulators stay
/// f32 (matching the f16 variant's accumulation semantics).
kernel void fused_gate_up_silu_mul_decode_bf16_specialized(
    device       bfloat* output  [[buffer(0)]],
    device const bfloat* input   [[buffer(1)]],
    device const bfloat* weight  [[buffer(2)]],
    uint3 tid     [[threadgroup_position_in_grid]],
    uint3 lid     [[thread_position_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    if (FUSED_MLP_DECODE_M != 1u) return;

    constexpr int BM = 1;
    constexpr int BN = 8;
    constexpr int SM = 1;
    constexpr int SN = 32;
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int threadsM = BM * SM;
    constexpr int threadsN = BN * SN;
    constexpr int blockM   = threadsM * TM;
    constexpr int blockN   = threadsN * TN;

    const uint N = FUSED_MLP_DECODE_N;
    const uint K = FUSED_MLP_DECODE_K;
    const uint matrix_ld = K;

    thread float gate_result[TM] = {0};
    thread float up_result  [TM] = {0};
    thread bfloat in_buf [TN];
    thread bfloat gate_buf[TN];
    thread bfloat up_buf  [TN];

    const int thrM = SN != 32 ? int(simd_lid) / SN : 0;
    const int thrN = SN != 32 ? int(simd_lid) % SN : int(simd_lid);

    const int sgN = BN != 1 ? int(simd_gid) % BN : 0;
    const int simdM = BN != 1 ? SM * (int(simd_gid) / BN) : SM * int(simd_gid);
    const int simdN = BN != 1 ? SN * (int(simd_gid) % BN) : 0;

    int bm = (simdM + thrM) * TM;
    int bn = (simdN + thrN) * TN;

    int out_row = int(tid.x) * blockM + bm;
    if (out_row >= int(N)) return;

    const int N_int = int(N);
    out_row = out_row + TM <= N_int ? out_row : N_int - TM;

    device const bfloat* gate_mat = weight + uint(out_row) * matrix_ld;
    device const bfloat* up_mat   = weight + (N + uint(out_row)) * matrix_ld;

    const int K_int = int(K);
    const int n_iter = K_int / blockN;
    const int last_iter = blockN * n_iter;
    const int leftover = K_int - last_iter;

    for (int i = 0; i < n_iter; ++i) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
            in_buf[tn] = input[bn + tn];
        }

        int mat_offset = 0;
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
                gate_buf[tn] = gate_mat[mat_offset + bn + tn];
                up_buf[tn]   = up_mat  [mat_offset + bn + tn];
            }
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
                gate_result[tm] += float(gate_buf[tn]) * float(in_buf[tn]);
                up_result[tm]   += float(up_buf[tn])   * float(in_buf[tn]);
            }
            mat_offset += int(matrix_ld);
        }

        bn += blockN;
    }

    if (leftover > 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
            in_buf[tn] = (bn + tn < K_int) ? input[bn + tn] : bfloat(0);
        }
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
                gate_buf[tn] = (bn + tn < K_int)
                    ? gate_mat[tm * int(matrix_ld) + bn + tn]
                    : bfloat(0);
                up_buf[tn] = (bn + tn < K_int)
                    ? up_mat  [tm * int(matrix_ld) + bn + tn]
                    : bfloat(0);
            }
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
                gate_result[tm] += float(gate_buf[tn]) * float(in_buf[tn]);
                up_result[tm]   += float(up_buf[tn])   * float(in_buf[tn]);
            }
        }
    }

    MLX_MTL_PRAGMA_UNROLL
    for (int tm = 0; tm < TM; tm++) {
        MLX_MTL_PRAGMA_UNROLL
        for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
            gate_result[tm] += simd_shuffle_down(gate_result[tm], sn);
            up_result[tm]   += simd_shuffle_down(up_result[tm], sn);
        }
    }

    threadgroup float tgp_gate[BN * (blockM + TM)];
    threadgroup float tgp_up  [BN * (blockM + TM)];

    threadgroup float* gate_results = tgp_gate + sgN * (blockM + TM) + bm;
    threadgroup float* up_results   = tgp_up   + sgN * (blockM + TM) + bm;
    if (thrN == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
            gate_results[tm] = gate_result[tm];
            up_results[tm]   = up_result[tm];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgN == 0 && thrN == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int sgn = 1; sgn < BN; sgn++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tm = 0; tm < TM; tm++) {
                gate_result[tm] += tgp_gate[sgn * (blockM + TM) + bm + tm];
                up_result[tm]   += tgp_up  [sgn * (blockM + TM) + bm + tm];
            }
        }

        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
            const float g = gate_result[tm];
            const float u = up_result[tm];
            const float act = FUSED_MLP_DECODE_IS_GELU
                ? gelu_approx(g)
                : g / (1.0f + exp(-g));
            output[out_row + tm] = bfloat(act * u);
        }
    }
}

// ============================================================================
// fused_gate_up_silu_mul_gemm_steel_{f16,bf16}_specialized
// ============================================================================
//
// Higher-throughput variant of the matrix-path fused MLP kernel.
// 32x32 output tile, 4 simdgroups per threadgroup (WM=2, WN=2),
// BK=16. Two `simdgroup_float8x8` accumulator tiles per simdgroup
// for gate and up, one shared SwiGLU epilogue, register-cached A/B
// fragments. Direct adaptation of the MLX `steel/gemm/` pattern
// (BlockMMA + BlockLoader at BM=BN=32, BK=16, WM=WN=2) inlined to
// avoid vendoring the full MLX header tree (cf. prior dead-end on
// the generic-gemm port at lm_head shapes — the MLP shapes
// (N <= 8192, grid_x <= 256) stay well under any 1024-tg-per-axis
// concern).
//
// Bindings (must match the lowering in
// `interpreter::metal::lowering::lower_one`):
//   buffer(0) = output  [M, N]            silu(gate) * up
//   buffer(1) = input   [M, K]            post-rmsnorm hidden
//   buffer(2) = weight  [2*N, K]          packed [gate; up]
//
// Function constants (distinct from the 8x8 variant's 0/1/2 + the
// decode variant's 3/4/5 so the same library can host all three
// kernels without colliding):
//   6 = M     (uint) — bucket_m
//   7 = N     (uint) — INTERMEDIATE_SIZE (per-side; 2N is gate+up width)
//   8 = K     (uint) — Q_SIZE  (hidden_size)
//
// Threadgroup grid: (ceil(N/32), ceil(M/32), 1)
// Threads per group: (128, 1, 1)  — 4 simdgroups × 32 lanes
//
// Tile layout per simdgroup:
//   sgM = simdgroup_id / WN ∈ {0, 1}; sgN = simdgroup_id % WN ∈ {0, 1}.
//   Each simdgroup owns rows [sgM*16, sgM*16 + 16) along M and cols
//   [sgN*16, sgN*16 + 16) along N. Within that 16×16 region the four
//   8×8 frags are contiguous: frag (i, j) at rows [i*8, i*8+8) cols
//   [j*8, j*8+8), with TM=TN=2.
//
// Loads:
//   Each threadgroup-pass reads one 32×16 A-tile and two 32×16 B-tiles
//   (gate + up) into threadgroup memory. With 128 threads × 4 halves
//   per thread = 512 halves = exactly 32×16, one round-trip. Vector
//   loads (half4 / bfloat4) keep per-thread issue at one device
//   transaction each. Per-row padding `STEEL_PAD = 8` halves keeps
//   the fragment loads off-bank.
//
// Edge handling:
//   M tail (M % 32 != 0) handled via per-row guard in the A loader
//   plus a guarded per-cell store in the epilogue. N tail handled
//   symmetrically on the B-side. K is assumed to be a multiple of
//   BK=16 — true for every model we ship today (Q_SIZE ∈ {2048, 3072,
//   …, 8192} are all multiples of 16). If a future model breaks
//   that, add a K-tail load_safe equivalent à la steel/gemm.h.

constant uint FUSED_MLP_STEEL_M [[function_constant(6)]];
constant uint FUSED_MLP_STEEL_N [[function_constant(7)]];
constant uint FUSED_MLP_STEEL_K [[function_constant(8)]];
constant bool FUSED_MLP_STEEL_IS_GELU [[function_constant(10)]];

#define STEEL_BM   32
#define STEEL_BN   32
#define STEEL_BK   16
#define STEEL_WM   2
#define STEEL_WN   2
#define STEEL_TM   2   // BM / (8 * WM)
#define STEEL_TN   2   // BN / (8 * WN)
#define STEEL_KFR  2   // BK / 8
#define STEEL_TGP  128 // WM * WN * 32
#define STEEL_PAD  8
#define STEEL_ALD  24  // BK + PAD
#define STEEL_BLD  24  // BK + PAD

inline float silu_steel(float x) {
    return x / (1.0f + exp(-x));
}

kernel void fused_gate_up_silu_mul_gemm_steel_f16_specialized(
    device       half* output  [[buffer(0)]],   // [M, N]
    device const half* input   [[buffer(1)]],   // [M, K]
    device const half* weight  [[buffer(2)]],   // [2*N, K]
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint3 tgid          [[threadgroup_position_in_grid]],
    uint3 tid3          [[thread_position_in_threadgroup]])
{
    (void)tid3;
    const uint M = FUSED_MLP_STEEL_M;
    const uint N = FUSED_MLP_STEEL_N;
    const uint K = FUSED_MLP_STEEL_K;

    const uint c_row = tgid.y * STEEL_BM;
    const uint c_col = tgid.x * STEEL_BN;
    if (c_row >= M || c_col >= N) return;

    threadgroup half As[STEEL_BM * STEEL_ALD];
    threadgroup half Bs_gate[STEEL_BN * STEEL_BLD];
    threadgroup half Bs_up  [STEEL_BN * STEEL_BLD];

    // Per-thread BlockLoader-equivalent indices.  Each thread reads
    // N_READS = (BM*BK)/TGP = 32*16/128 = 4 halves per pass; TCOLS =
    // BK/N_READS = 4 → bj walks 0,4,8,12; TROWS = TGP/TCOLS = 32 ==
    // BM, so the whole 32×16 tile is staged in a single round.
    constexpr int N_READS = (STEEL_BM * STEEL_BK) / STEEL_TGP;  // 4
    constexpr int TCOLS = STEEL_BK / N_READS;                    // 4

    const uint thread_idx = simd_group_id * 32u + simd_lane_id;  // [0, 128)
    const uint bi = thread_idx / uint(TCOLS);                    // [0, 32)
    const uint bj = uint(N_READS) * (thread_idx % uint(TCOLS));  // 0,4,8,12

    const int sgM = int(simd_group_id) / STEEL_WN;
    const int sgN = int(simd_group_id) % STEEL_WN;

    // Per-tile valid extents (M / N tail handling).
    const uint m_tile = (c_row + STEEL_BM <= M) ? uint(STEEL_BM) : (M - c_row);
    const uint n_tile = (c_col + STEEL_BN <= N) ? uint(STEEL_BN) : (N - c_col);
    const bool m_full = (m_tile == STEEL_BM);
    const bool n_full = (n_tile == STEEL_BN);

    simdgroup_float8x8 acc_gate[STEEL_TM][STEEL_TN];
    simdgroup_float8x8 acc_up  [STEEL_TM][STEEL_TN];
    MLX_MTL_PRAGMA_UNROLL
    for (int i = 0; i < STEEL_TM; ++i) {
        MLX_MTL_PRAGMA_UNROLL
        for (int j = 0; j < STEEL_TN; ++j) {
            acc_gate[i][j] = simdgroup_float8x8(0.0f);
            acc_up  [i][j] = simdgroup_float8x8(0.0f);
        }
    }

    device const half* A_src  = input  + (c_row + bi) * K + bj;
    device const half* Bg_src = weight + (c_col + bi) * K + bj;
    device const half* Bu_src = weight + (N + c_col + bi) * K + bj;

    threadgroup half* As_dst = As      + bi * STEEL_ALD + bj;
    threadgroup half* Bg_dst = Bs_gate + bi * STEEL_BLD + bj;
    threadgroup half* Bu_dst = Bs_up   + bi * STEEL_BLD + bj;

    const uint k_iter_count = K / uint(STEEL_BK);

    for (uint kk = 0; kk < k_iter_count; ++kk) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative load of one 32×16 tile per buffer. Guarded
        // loads gate the *dereference*, not just the stored value,
        // so the device read itself never goes OOB on M / N tails.
        if (m_full) {
            *((threadgroup half4*)As_dst) = *((const device half4*)A_src);
        } else if (bi < m_tile) {
            *((threadgroup half4*)As_dst) = *((const device half4*)A_src);
        } else {
            *((threadgroup half4*)As_dst) = half4(0);
        }
        if (n_full) {
            *((threadgroup half4*)Bg_dst) = *((const device half4*)Bg_src);
            *((threadgroup half4*)Bu_dst) = *((const device half4*)Bu_src);
        } else if (bi < n_tile) {
            *((threadgroup half4*)Bg_dst) = *((const device half4*)Bg_src);
            *((threadgroup half4*)Bu_dst) = *((const device half4*)Bu_src);
        } else {
            *((threadgroup half4*)Bg_dst) = half4(0);
            *((threadgroup half4*)Bu_dst) = half4(0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // K-fragment loop: BK=16 → 2 8-wide K-frags per BK iter.
        MLX_MTL_PRAGMA_UNROLL
        for (int kf = 0; kf < STEEL_KFR; ++kf) {
            simdgroup_half8x8 A_frag[STEEL_TM];
            MLX_MTL_PRAGMA_UNROLL
            for (int i = 0; i < STEEL_TM; ++i) {
                threadgroup const half* a_ptr =
                    As + (sgM * 16 + i * 8) * STEEL_ALD + kf * 8;
                simdgroup_load(A_frag[i], a_ptr, STEEL_ALD);
            }

            simdgroup_half8x8 Bg_frag[STEEL_TN];
            simdgroup_half8x8 Bu_frag[STEEL_TN];
            MLX_MTL_PRAGMA_UNROLL
            for (int j = 0; j < STEEL_TN; ++j) {
                int n_off = sgN * 16 + j * 8;
                threadgroup const half* bg_ptr =
                    Bs_gate + n_off * STEEL_BLD + kf * 8;
                threadgroup const half* bu_ptr =
                    Bs_up   + n_off * STEEL_BLD + kf * 8;
                simdgroup_load(Bg_frag[j], bg_ptr, STEEL_BLD,
                               ulong2(0, 0), /*transpose*/ true);
                simdgroup_load(Bu_frag[j], bu_ptr, STEEL_BLD,
                               ulong2(0, 0), /*transpose*/ true);
            }

            MLX_MTL_PRAGMA_UNROLL
            for (int i = 0; i < STEEL_TM; ++i) {
                MLX_MTL_PRAGMA_UNROLL
                for (int j = 0; j < STEEL_TN; ++j) {
                    simdgroup_multiply_accumulate(
                        acc_gate[i][j], A_frag[i], Bg_frag[j], acc_gate[i][j]);
                    simdgroup_multiply_accumulate(
                        acc_up  [i][j], A_frag[i], Bu_frag[j], acc_up  [i][j]);
                }
            }
        }

        A_src  += STEEL_BK;
        Bg_src += STEEL_BK;
        Bu_src += STEEL_BK;
    }

    // Epilogue: store gate + up accumulators to threadgroup scratch,
    // apply silu(g)*u per cell, then write the live region of the
    // 32×32 output tile to device memory.
    threadgroup float gate_scratch[STEEL_BM * STEEL_BN];
    threadgroup float up_scratch  [STEEL_BM * STEEL_BN];
    threadgroup half  c_scratch   [STEEL_BM * STEEL_BN];

    MLX_MTL_PRAGMA_UNROLL
    for (int i = 0; i < STEEL_TM; ++i) {
        MLX_MTL_PRAGMA_UNROLL
        for (int j = 0; j < STEEL_TN; ++j) {
            int row_base = sgM * 16 + i * 8;
            int col_base = sgN * 16 + j * 8;
            simdgroup_store(acc_gate[i][j],
                            gate_scratch + row_base * STEEL_BN + col_base,
                            STEEL_BN);
            simdgroup_store(acc_up[i][j],
                            up_scratch + row_base * STEEL_BN + col_base,
                            STEEL_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 32×32 = 1024 cells / 128 threads = 8 cells / thread.
    for (uint t = thread_idx; t < uint(STEEL_BM * STEEL_BN); t += STEEL_TGP) {
        float g = gate_scratch[t];
        float u = up_scratch[t];
        c_scratch[t] = half((FUSED_MLP_STEEL_IS_GELU ? gelu_approx(g) : silu_steel(g)) * u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (m_full && n_full) {
        for (uint t = thread_idx; t < uint(STEEL_BM * STEEL_BN); t += STEEL_TGP) {
            uint r = t / uint(STEEL_BN);
            uint c = t % uint(STEEL_BN);
            output[(c_row + r) * N + (c_col + c)] = c_scratch[t];
        }
    } else {
        for (uint t = thread_idx; t < uint(STEEL_BM * STEEL_BN); t += STEEL_TGP) {
            uint r = t / uint(STEEL_BN);
            uint c = t % uint(STEEL_BN);
            if (r < m_tile && c < n_tile) {
                output[(c_row + r) * N + (c_col + c)] = c_scratch[t];
            }
        }
    }
}

/// BF16 mirror of `fused_gate_up_silu_mul_gemm_steel_f16_specialized`.
/// `simdgroup_bfloat8x8` MMA tiles (Metal 3.1+, native on M3+).
/// Accumulators stay `simdgroup_float8x8` — bf16→f32 accumulation is
/// the standard pattern for matmul kernels and matches CUDA's bf16 GEMM.
kernel void fused_gate_up_silu_mul_gemm_steel_bf16_specialized(
    device       bfloat* output  [[buffer(0)]],
    device const bfloat* input   [[buffer(1)]],
    device const bfloat* weight  [[buffer(2)]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint3 tgid          [[threadgroup_position_in_grid]],
    uint3 tid3          [[thread_position_in_threadgroup]])
{
    (void)tid3;
    const uint M = FUSED_MLP_STEEL_M;
    const uint N = FUSED_MLP_STEEL_N;
    const uint K = FUSED_MLP_STEEL_K;

    const uint c_row = tgid.y * STEEL_BM;
    const uint c_col = tgid.x * STEEL_BN;
    if (c_row >= M || c_col >= N) return;

    threadgroup bfloat As[STEEL_BM * STEEL_ALD];
    threadgroup bfloat Bs_gate[STEEL_BN * STEEL_BLD];
    threadgroup bfloat Bs_up  [STEEL_BN * STEEL_BLD];

    constexpr int N_READS = (STEEL_BM * STEEL_BK) / STEEL_TGP;
    constexpr int TCOLS = STEEL_BK / N_READS;

    const uint thread_idx = simd_group_id * 32u + simd_lane_id;
    const uint bi = thread_idx / uint(TCOLS);
    const uint bj = uint(N_READS) * (thread_idx % uint(TCOLS));

    const int sgM = int(simd_group_id) / STEEL_WN;
    const int sgN = int(simd_group_id) % STEEL_WN;

    const uint m_tile = (c_row + STEEL_BM <= M) ? uint(STEEL_BM) : (M - c_row);
    const uint n_tile = (c_col + STEEL_BN <= N) ? uint(STEEL_BN) : (N - c_col);
    const bool m_full = (m_tile == STEEL_BM);
    const bool n_full = (n_tile == STEEL_BN);

    simdgroup_float8x8 acc_gate[STEEL_TM][STEEL_TN];
    simdgroup_float8x8 acc_up  [STEEL_TM][STEEL_TN];
    MLX_MTL_PRAGMA_UNROLL
    for (int i = 0; i < STEEL_TM; ++i) {
        MLX_MTL_PRAGMA_UNROLL
        for (int j = 0; j < STEEL_TN; ++j) {
            acc_gate[i][j] = simdgroup_float8x8(0.0f);
            acc_up  [i][j] = simdgroup_float8x8(0.0f);
        }
    }

    device const bfloat* A_src  = input  + (c_row + bi) * K + bj;
    device const bfloat* Bg_src = weight + (c_col + bi) * K + bj;
    device const bfloat* Bu_src = weight + (N + c_col + bi) * K + bj;

    threadgroup bfloat* As_dst = As      + bi * STEEL_ALD + bj;
    threadgroup bfloat* Bg_dst = Bs_gate + bi * STEEL_BLD + bj;
    threadgroup bfloat* Bu_dst = Bs_up   + bi * STEEL_BLD + bj;

    const uint k_iter_count = K / uint(STEEL_BK);

    for (uint kk = 0; kk < k_iter_count; ++kk) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (m_full) {
            *((threadgroup bfloat4*)As_dst) = *((const device bfloat4*)A_src);
        } else if (bi < m_tile) {
            *((threadgroup bfloat4*)As_dst) = *((const device bfloat4*)A_src);
        } else {
            *((threadgroup bfloat4*)As_dst) = bfloat4(0);
        }
        if (n_full) {
            *((threadgroup bfloat4*)Bg_dst) = *((const device bfloat4*)Bg_src);
            *((threadgroup bfloat4*)Bu_dst) = *((const device bfloat4*)Bu_src);
        } else if (bi < n_tile) {
            *((threadgroup bfloat4*)Bg_dst) = *((const device bfloat4*)Bg_src);
            *((threadgroup bfloat4*)Bu_dst) = *((const device bfloat4*)Bu_src);
        } else {
            *((threadgroup bfloat4*)Bg_dst) = bfloat4(0);
            *((threadgroup bfloat4*)Bu_dst) = bfloat4(0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        MLX_MTL_PRAGMA_UNROLL
        for (int kf = 0; kf < STEEL_KFR; ++kf) {
            simdgroup_bfloat8x8 A_frag[STEEL_TM];
            MLX_MTL_PRAGMA_UNROLL
            for (int i = 0; i < STEEL_TM; ++i) {
                threadgroup const bfloat* a_ptr =
                    As + (sgM * 16 + i * 8) * STEEL_ALD + kf * 8;
                simdgroup_load(A_frag[i], a_ptr, STEEL_ALD);
            }

            simdgroup_bfloat8x8 Bg_frag[STEEL_TN];
            simdgroup_bfloat8x8 Bu_frag[STEEL_TN];
            MLX_MTL_PRAGMA_UNROLL
            for (int j = 0; j < STEEL_TN; ++j) {
                int n_off = sgN * 16 + j * 8;
                threadgroup const bfloat* bg_ptr =
                    Bs_gate + n_off * STEEL_BLD + kf * 8;
                threadgroup const bfloat* bu_ptr =
                    Bs_up   + n_off * STEEL_BLD + kf * 8;
                simdgroup_load(Bg_frag[j], bg_ptr, STEEL_BLD,
                               ulong2(0, 0), true);
                simdgroup_load(Bu_frag[j], bu_ptr, STEEL_BLD,
                               ulong2(0, 0), true);
            }

            MLX_MTL_PRAGMA_UNROLL
            for (int i = 0; i < STEEL_TM; ++i) {
                MLX_MTL_PRAGMA_UNROLL
                for (int j = 0; j < STEEL_TN; ++j) {
                    simdgroup_multiply_accumulate(
                        acc_gate[i][j], A_frag[i], Bg_frag[j], acc_gate[i][j]);
                    simdgroup_multiply_accumulate(
                        acc_up  [i][j], A_frag[i], Bu_frag[j], acc_up  [i][j]);
                }
            }
        }

        A_src  += STEEL_BK;
        Bg_src += STEEL_BK;
        Bu_src += STEEL_BK;
    }

    threadgroup float  gate_scratch[STEEL_BM * STEEL_BN];
    threadgroup float  up_scratch  [STEEL_BM * STEEL_BN];
    threadgroup bfloat c_scratch   [STEEL_BM * STEEL_BN];

    MLX_MTL_PRAGMA_UNROLL
    for (int i = 0; i < STEEL_TM; ++i) {
        MLX_MTL_PRAGMA_UNROLL
        for (int j = 0; j < STEEL_TN; ++j) {
            int row_base = sgM * 16 + i * 8;
            int col_base = sgN * 16 + j * 8;
            simdgroup_store(acc_gate[i][j],
                            gate_scratch + row_base * STEEL_BN + col_base,
                            STEEL_BN);
            simdgroup_store(acc_up[i][j],
                            up_scratch + row_base * STEEL_BN + col_base,
                            STEEL_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = thread_idx; t < uint(STEEL_BM * STEEL_BN); t += STEEL_TGP) {
        float g = gate_scratch[t];
        float u = up_scratch[t];
        c_scratch[t] = bfloat((FUSED_MLP_STEEL_IS_GELU ? gelu_approx(g) : silu_steel(g)) * u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (m_full && n_full) {
        for (uint t = thread_idx; t < uint(STEEL_BM * STEEL_BN); t += STEEL_TGP) {
            uint r = t / uint(STEEL_BN);
            uint c = t % uint(STEEL_BN);
            output[(c_row + r) * N + (c_col + c)] = c_scratch[t];
        }
    } else {
        for (uint t = thread_idx; t < uint(STEEL_BM * STEEL_BN); t += STEEL_TGP) {
            uint r = t / uint(STEEL_BN);
            uint c = t % uint(STEEL_BN);
            if (r < m_tile && c < n_tile) {
                output[(c_row + r) * N + (c_col + c)] = c_scratch[t];
            }
        }
    }
}

