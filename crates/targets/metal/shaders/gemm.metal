// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "baked.h"
using namespace metal;

// ---------------------------------------------------------------------------
// gemm_bf16_specialized
// ---------------------------------------------------------------------------
//
// Generic dense GEMM for bf16 inputs:
//
//     C = A @ B^T,   A: [M, K], B: [N, K] (transposed-right), C: [M, N]
//
// Hardcodes the "Linear layer" convention used by every Llama / Qwen /
// Mistral GEMM in the metal backend: `transpose_a = false`,
// `transpose_b = true`, `alpha = 1.0`, `beta = 0.0`. The runtime never
// invokes the more general MPS surface, so baking these in lets the
// shader stay short and the baked constants stay at three
// dimensions.
//
// Why a custom kernel: `MPSMatrixMultiplication` only accepts
// `MPSDataTypeFloat32`, `MPSDataTypeFloat16`, `MPSDataTypeInt8`,
// `MPSDataTypeInt16` (asserted at runtime — see
// `MPSMatrixMultiplication.mm:3260`). Even though the hardware on
// Apple Silicon M3+ has native bf16 MMA and `MPSDataTypeBFloat16`
// is a valid MPSCore type, the matmul kernel itself doesn't take
// it. We use `simdgroup_bfloat8x8` (typedef of
// `simdgroup_matrix<bfloat, 8, 8>`, available since Metal 3.1) which
// drives the same hardware MMA from a compute kernel.
//
// Bindings (must match `interpreter::metal::lowering::lower_one`'s
// `Instruction::Gemm` arm — same shape as MPS path: out, in, weight):
//   buffer(0) = output  [M, N]
//   buffer(1) = input   [M, K]
//   buffer(2) = weight  [N, K]
//
// Baked constants:
//   0 = M
//   1 = N
//   2 = K
//
// Dispatch: threadgroups (ceil(N/8), ceil(M/8), 1), threads (32, 1, 1).
// One simdgroup per threadgroup; each simdgroup computes an 8×8
// output tile.
// ---------------------------------------------------------------------------

SCRATCHY_CONSTANT_OPTIONAL(uint, GEMM_M, 0);
SCRATCHY_CONSTANT_OPTIONAL(uint, GEMM_N, 1);
SCRATCHY_CONSTANT_OPTIONAL(uint, GEMM_K, 2);

// Shared C = A @ B^T body (A:[M,K], B:[N,K], C:[M,N]) with M/N/K passed
// explicitly: the dense kernels pass the baked GEMM_M/N/K.
inline void gemm_t_bf16_body(
    device bfloat*       output,
    device const bfloat* input,
    device const bfloat* weight,
    uint M, uint N, uint K,
    uint w_ld,           // weight (B) ROW stride
    threadgroup bfloat*  a_pad,
    threadgroup bfloat*  b_pad,
    threadgroup float*   c_scratch,
    uint3 tgid, uint tid)
{
    constexpr uint TILE = 8u;
    const uint m_base = tgid.y * TILE;
    const uint n_base = tgid.x * TILE;
    if (m_base >= M || n_base >= N) return;

    const bool m_full = (m_base + TILE <= M);
    const bool n_full = (n_base + TILE <= N);

    simdgroup_float8x8 acc = simdgroup_float8x8(0.0f);

    for (uint k_base = 0u; k_base < K; k_base += TILE) {
        const bool k_full = (k_base + TILE <= K);

        simdgroup_bfloat8x8 A;
        simdgroup_bfloat8x8 B;

        if (m_full && k_full) {
            simdgroup_load(A, input + m_base * K + k_base, K);
        } else {
            for (uint t = tid; t < TILE * TILE; t += 32u) {
                uint r = t / TILE;
                uint c = t % TILE;
                uint mr = m_base + r;
                uint kc = k_base + c;
                a_pad[r * TILE + c] = (mr < M && kc < K) ? input[mr * K + kc] : bfloat(0);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(A, a_pad, TILE);
        }

        // weight is [N, K] with ROW stride w_ld; compute C = A @ W^T → load B
        // with transpose=true.
        if (n_full && k_full) {
            simdgroup_load(B, weight + n_base * w_ld + k_base, w_ld, ulong2(0, 0), true);
        } else {
            for (uint t = tid; t < TILE * TILE; t += 32u) {
                uint r = t / TILE;
                uint c = t % TILE;
                uint nr = n_base + r;
                uint kc = k_base + c;
                b_pad[r * TILE + c] = (nr < N && kc < K) ? weight[nr * w_ld + kc] : bfloat(0);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(B, b_pad, TILE, ulong2(0, 0), true);
        }
        simdgroup_multiply_accumulate(acc, A, B, acc);
    }

    simdgroup_store(acc, c_scratch, TILE);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (m_full && n_full) {
        for (uint t = tid; t < TILE * TILE; t += 32u) {
            uint r = t / TILE;
            uint c = t % TILE;
            output[(m_base + r) * N + (n_base + c)] = bfloat(c_scratch[r * TILE + c]);
        }
    } else {
        for (uint t = tid; t < TILE * TILE; t += 32u) {
            uint r = t / TILE;
            uint c = t % TILE;
            uint mr = m_base + r;
            uint nc = n_base + c;
            if (mr < M && nc < N) {
                output[mr * N + nc] = bfloat(c_scratch[r * TILE + c]);
            }
        }
    }
}

#if SCRATCHY_COMPILES(gemm_bf16_specialized)
kernel void gemm_bf16_specialized(
    device       bfloat* output [[buffer(0)]],
    device const bfloat* input  [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    constexpr uint TILE = 8u;
    threadgroup bfloat a_pad[TILE * TILE];
    threadgroup bfloat b_pad[TILE * TILE];
    threadgroup float  c_scratch[TILE * TILE];
    // dense: w row stride == contraction == GEMM_K.
    gemm_t_bf16_body(output, input, weight, GEMM_M, GEMM_N, GEMM_K, GEMM_K,
                     a_pad, b_pad, c_scratch, tgid, tid3.x);
}
#endif

// f16 counterpart of `gemm_t_bf16_body` — identical tiling and dispatch
// convention, with `half` inputs and `simdgroup_half8x8` MMA (float
// accumulators). Lets f16 dense GEMM run as a normal MTL4 compute
// dispatch instead of MPS' `MPSMatrixMultiplication`, so the metal
// backend has no classic command-buffer GEMM path. Output matches the
// MPS path modulo accumulation order.
inline void gemm_t_f16_body(
    device half*       output,
    device const half* input,
    device const half* weight,
    uint M, uint N, uint K,
    uint w_ld,
    threadgroup half*  a_pad,
    threadgroup half*  b_pad,
    threadgroup float* c_scratch,
    uint3 tgid, uint tid)
{
    constexpr uint TILE = 8u;
    const uint m_base = tgid.y * TILE;
    const uint n_base = tgid.x * TILE;
    if (m_base >= M || n_base >= N) return;

    const bool m_full = (m_base + TILE <= M);
    const bool n_full = (n_base + TILE <= N);

    simdgroup_float8x8 acc = simdgroup_float8x8(0.0f);

    for (uint k_base = 0u; k_base < K; k_base += TILE) {
        const bool k_full = (k_base + TILE <= K);

        simdgroup_half8x8 A;
        simdgroup_half8x8 B;

        if (m_full && k_full) {
            simdgroup_load(A, input + m_base * K + k_base, K);
        } else {
            for (uint t = tid; t < TILE * TILE; t += 32u) {
                uint r = t / TILE;
                uint c = t % TILE;
                uint mr = m_base + r;
                uint kc = k_base + c;
                a_pad[r * TILE + c] = (mr < M && kc < K) ? input[mr * K + kc] : half(0);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(A, a_pad, TILE);
        }

        if (n_full && k_full) {
            simdgroup_load(B, weight + n_base * w_ld + k_base, w_ld, ulong2(0, 0), true);
        } else {
            for (uint t = tid; t < TILE * TILE; t += 32u) {
                uint r = t / TILE;
                uint c = t % TILE;
                uint nr = n_base + r;
                uint kc = k_base + c;
                b_pad[r * TILE + c] = (nr < N && kc < K) ? weight[nr * w_ld + kc] : half(0);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(B, b_pad, TILE, ulong2(0, 0), true);
        }
        simdgroup_multiply_accumulate(acc, A, B, acc);
    }

    simdgroup_store(acc, c_scratch, TILE);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (m_full && n_full) {
        for (uint t = tid; t < TILE * TILE; t += 32u) {
            uint r = t / TILE;
            uint c = t % TILE;
            output[(m_base + r) * N + (n_base + c)] = half(c_scratch[r * TILE + c]);
        }
    } else {
        for (uint t = tid; t < TILE * TILE; t += 32u) {
            uint r = t / TILE;
            uint c = t % TILE;
            uint mr = m_base + r;
            uint nc = n_base + c;
            if (mr < M && nc < N) {
                output[mr * N + nc] = half(c_scratch[r * TILE + c]);
            }
        }
    }
}

#if SCRATCHY_COMPILES(gemm_f16_specialized)
kernel void gemm_f16_specialized(
    device       half* output [[buffer(0)]],
    device const half* input  [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    constexpr uint TILE = 8u;
    threadgroup half  a_pad[TILE * TILE];
    threadgroup half  b_pad[TILE * TILE];
    threadgroup float c_scratch[TILE * TILE];
    // dense: w row stride == contraction == GEMM_K.
    gemm_t_f16_body(output, input, weight, GEMM_M, GEMM_N, GEMM_K, GEMM_K,
                    a_pad, b_pad, c_scratch, tgid, tid3.x);
}
#endif

// ── Blocked simdgroup GEMM: bf16 dense ─────────────────────────────────────
// C = A[M,K] @ B[N,K]^T. 32×32 output tile, 4 simdgroups (WM=WN=2), BK=16,
// register-cached 8×8 frags, shared A/B — a single-output adaptation of the MoE
// steel GEMM (`fused_gate_up_silu_mul_gemm_steel`). Far higher reuse than the
// 8×8 reference body (which re-streamed K per output tile). Handles M/N tails
// AND a K-tail (K need NOT be a multiple of 16 — guarded cooperative load).
// `w_ld` is the B row stride.
#define ATTN_BM 32
#define ATTN_BN 32
#define ATTN_BK 16
#define ATTN_WN 2
#define ATTN_TM 2
#define ATTN_TN 2
#define ATTN_KFR 2
#define ATTN_TGP 128
#define ATTN_LD 24 // BK + 8 pad (keeps frag loads off-bank)

inline void gemm_t_bf16_blocked(
    device bfloat*       output,
    device const bfloat* input,
    device const bfloat* weight,
    uint M, uint N, uint K, uint w_ld,
    threadgroup bfloat*  As,        // [ATTN_BM * ATTN_LD]
    threadgroup bfloat*  Bs,        // [ATTN_BN * ATTN_LD]
    threadgroup float*   c_scratch, // [ATTN_BM * ATTN_BN]
    uint simd_group_id, uint simd_lane_id, uint3 tgid)
{
    const uint c_row = tgid.y * ATTN_BM;
    const uint c_col = tgid.x * ATTN_BN;
    if (c_row >= M || c_col >= N) {
        return;
    }

    constexpr int N_READS = (ATTN_BM * ATTN_BK) / ATTN_TGP; // 4
    constexpr int TCOLS = ATTN_BK / N_READS;                // 4
    const uint thread_idx = simd_group_id * 32u + simd_lane_id; // [0,128)
    const uint bi = thread_idx / uint(TCOLS);                   // [0,32)
    const uint bj = uint(N_READS) * (thread_idx % uint(TCOLS)); // 0,4,8,12

    const int sgM = int(simd_group_id) / ATTN_WN;
    const int sgN = int(simd_group_id) % ATTN_WN;

    const uint m_tile = (c_row + ATTN_BM <= M) ? uint(ATTN_BM) : (M - c_row);
    const uint n_tile = (c_col + ATTN_BN <= N) ? uint(ATTN_BN) : (N - c_col);

    simdgroup_float8x8 acc[ATTN_TM][ATTN_TN];
    for (int i = 0; i < ATTN_TM; ++i) {
        for (int j = 0; j < ATTN_TN; ++j) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }

    const uint k_iters = (K + ATTN_BK - 1u) / ATTN_BK; // ceil → K-tail safe
    for (uint kk = 0; kk < k_iters; ++kk) {
        const uint k_base = kk * ATTN_BK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Cooperative guarded load of one 32×16 A-tile and 32×16 B-tile. Guards
        // gate the dereference, so device reads never go OOB on M/N/K tails.
        threadgroup bfloat* as_dst = As + bi * ATTN_LD + bj;
        threadgroup bfloat* bs_dst = Bs + bi * ATTN_LD + bj;
        const device bfloat* a_src = input + (c_row + bi) * K + (k_base + bj);
        const device bfloat* b_src = weight + (c_col + bi) * w_ld + (k_base + bj);
        for (int c = 0; c < N_READS; ++c) {
            const uint kc = k_base + bj + uint(c);
            as_dst[c] = (bi < m_tile && kc < K) ? a_src[c] : bfloat(0);
            bs_dst[c] = (bi < n_tile && kc < K) ? b_src[c] : bfloat(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kf = 0; kf < ATTN_KFR; ++kf) {
            simdgroup_bfloat8x8 A_frag[ATTN_TM];
            for (int i = 0; i < ATTN_TM; ++i) {
                threadgroup const bfloat* a_ptr = As + (sgM * 16 + i * 8) * ATTN_LD + kf * 8;
                simdgroup_load(A_frag[i], a_ptr, ATTN_LD);
            }
            simdgroup_bfloat8x8 B_frag[ATTN_TN];
            for (int j = 0; j < ATTN_TN; ++j) {
                threadgroup const bfloat* b_ptr = Bs + (sgN * 16 + j * 8) * ATTN_LD + kf * 8;
                simdgroup_load(B_frag[j], b_ptr, ATTN_LD, ulong2(0, 0), true);
            }
            for (int i = 0; i < ATTN_TM; ++i) {
                for (int j = 0; j < ATTN_TN; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], A_frag[i], B_frag[j], acc[i][j]);
                }
            }
        }
    }

    for (int i = 0; i < ATTN_TM; ++i) {
        for (int j = 0; j < ATTN_TN; ++j) {
            const int rb = sgM * 16 + i * 8;
            const int cb = sgN * 16 + j * 8;
            simdgroup_store(acc[i][j], c_scratch + rb * ATTN_BN + cb, ATTN_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = thread_idx; t < uint(ATTN_BM * ATTN_BN); t += ATTN_TGP) {
        const uint r = t / uint(ATTN_BN);
        const uint c = t % uint(ATTN_BN);
        if (r < m_tile && c < n_tile) {
            output[(c_row + r) * N + (c_col + c)] = bfloat(c_scratch[t]);
        }
    }
}

// Dense bf16 from `GEMM_BLOCKED_ROWS` rows: `gemm_bf16_specialized`'s MMAs in its order, its bits.
#if SCRATCHY_COMPILES(gemm_bf16_blocked)
kernel void gemm_bf16_blocked(
    device       bfloat* output   [[buffer(0)]],
    device const bfloat* input    [[buffer(1)]],
    device const bfloat* weight   [[buffer(2)]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint3 tgid          [[threadgroup_position_in_grid]])
{
    threadgroup bfloat As[ATTN_BM * ATTN_LD];
    threadgroup bfloat Bs[ATTN_BN * ATTN_LD];
    threadgroup float  c_scratch[ATTN_BM * ATTN_BN];
    gemm_t_bf16_blocked(output, input, weight, GEMM_M, GEMM_N, GEMM_K, GEMM_K,
                        As, Bs, c_scratch, simd_group_id, simd_lane_id, tgid);
}
#endif

// ---------------------------------------------------------------------------
// gemv_{f16,bf16}_specialized
// ---------------------------------------------------------------------------
//
// A few rows (M <= GEMV_MAX_ROWS): output = input @ weight^T, each row's product as MLX's
// GEMVKernel computes it at its standard instantiation (`instantiate_gemv(name, itype, 1, 8, 1,
// 32, 4, 4)`; the same loop `fused_gate_up_silu_mul_decode_*_specialized` runs for its gate rows).
// Each threadgroup's 8 simdgroups split K, each thread holding 4 outputs' float sums per row over
// its 4-element slices; the simdgroups reduce by shuffle, then through threadgroup memory. Each
// weight slice is loaded once for every row — a verify step's rows read the matrix once.
//
// Bindings and constants are the GEMM's: buffer(0) = output [M, N], buffer(1) = input [M, K],
// buffer(2) = weight [N, K]; 0 / 1 / 2 = M / N / K.
//
// `gemv_normed_*` (`MetalFusion::NormedRouter`, `NormedQmv`): a router's pre-norm, or the RMSNorm
// whose rows it reads, folded in. Constant 5 is its epsilon, 6 its gain offset, buffer(3) its
// gain: each thread dots the weights with `x ⊙ (gain + offset)` and sums `x²` as it loads `x`,
// and each row is scaled by its own `1 / rms(x)` when it stores.
//
// Dispatch: threadgroups (ceil(N/4), 1, 1), threads (256, 1, 1). Needs N >= 4: the last
// threadgroup moves back to the last 4 rows.
// ---------------------------------------------------------------------------

#ifndef MLX_MTL_PRAGMA_UNROLL
#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#endif

SCRATCHY_CONSTANT_OPTIONAL(float, GEMV_NORM_EPS, 5);
SCRATCHY_CONSTANT_OPTIONAL(float, GEMV_NORM_W_OFFSET, 6);
constant constexpr bool GEMV_NORMED = GEMV_NORM_EPS_SET;

// `M` (<= MAX_ROWS) rows of `input`, `tgp` holding MAX_ROWS rows' partials.
template <typename T, typename G, int MAX_ROWS>
METAL_FUNC void gemv_specialized_impl(
    device       T*       output,
    device const T*       input,
    device const T*       weight,
    device const G*       gain,
    threadgroup float*    tgp,
    int                    M,
    uint                   tid_x,
    uint                   simd_gid,
    uint                   simd_lid)
{
    constexpr int BN = 8;              // simdgroups per threadgroup, all along K
    constexpr int SN = 32;             // threads per simdgroup, all along K
    constexpr int TM = 4;              // outputs per thread
    constexpr int TN = 4;              // K elements per thread per step
    constexpr int blockM = TM;         // outputs per threadgroup
    constexpr int blockN = BN * SN * TN;
    constexpr int row_tgp = BN * (blockM + TM);  // one row's partials in `tgp`

    const int N = int(GEMM_N);
    const int K = int(GEMM_K);

    thread float result[MAX_ROWS][TM] = {{0}};
    thread float in_buf[MAX_ROWS][TN];
    thread T w_buf[TN];
    float sum_sq[MAX_ROWS] = {0};

    const int sgN = int(simd_gid) % BN;
    int bn = (SN * sgN + int(simd_lid)) * TN;

    int out_row = int(tid_x) * blockM;
    if (out_row >= N) return;
    out_row = out_row + TM <= N ? out_row : N - TM;
    device const T* mat = weight + uint(out_row) * uint(K);

    const int n_iter = K / blockN;
    const int leftover = K - blockN * n_iter;

    for (int i = 0; i < n_iter; ++i) {
        for (int r = 0; r < M; r++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
                const float x = float(input[r * K + bn + tn]);
                sum_sq[r] += GEMV_NORMED ? x * x : 0.0f;
                in_buf[r][tn] = GEMV_NORMED ? x * (float(gain[bn + tn]) + GEMV_NORM_W_OFFSET) : x;
            }
        }
        int mat_offset = 0;
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
                w_buf[tn] = mat[mat_offset + bn + tn];
            }
            for (int r = 0; r < M; r++) {
                MLX_MTL_PRAGMA_UNROLL
                for (int tn = 0; tn < TN; tn++) {
                    result[r][tm] += float(w_buf[tn]) * in_buf[r][tn];
                }
            }
            mat_offset += K;
        }
        bn += blockN;
    }

    if (leftover > 0) {
        for (int r = 0; r < M; r++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
                const float x = (bn + tn < K) ? float(input[r * K + bn + tn]) : 0.0f;
                sum_sq[r] += GEMV_NORMED ? x * x : 0.0f;
                in_buf[r][tn] = GEMV_NORMED && bn + tn < K
                                    ? x * (float(gain[bn + tn]) + GEMV_NORM_W_OFFSET)
                                    : x;
            }
        }
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
                w_buf[tn] = (bn + tn < K) ? mat[tm * K + bn + tn] : T(0);
            }
            for (int r = 0; r < M; r++) {
                MLX_MTL_PRAGMA_UNROLL
                for (int tn = 0; tn < TN; tn++) {
                    result[r][tm] += float(w_buf[tn]) * in_buf[r][tn];
                }
            }
        }
    }

    for (int r = 0; r < M; r++) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
            MLX_MTL_PRAGMA_UNROLL
            for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                result[r][tm] += simd_shuffle_down(result[r][tm], sn);
            }
        }
        // A simdgroup's sum of squares of the row rides in its slot's spare word.
        sum_sq[r] = GEMV_NORMED ? simd_sum(sum_sq[r]) : 0.0f;
    }

    if (simd_lid == 0) {
        for (int r = 0; r < M; r++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tm = 0; tm < TM; tm++) {
                tgp[r * row_tgp + sgN * (blockM + TM) + tm] = result[r][tm];
            }
            tgp[r * row_tgp + sgN * (blockM + TM) + TM] = sum_sq[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgN == 0 && simd_lid == 0) {
        for (int r = 0; r < M; r++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int sgn = 1; sgn < BN; sgn++) {
                MLX_MTL_PRAGMA_UNROLL
                for (int tm = 0; tm < TM; tm++) {
                    result[r][tm] += tgp[r * row_tgp + sgn * (blockM + TM) + tm];
                }
                sum_sq[r] += tgp[r * row_tgp + sgn * (blockM + TM) + TM];
            }
            const float scale =
                GEMV_NORMED ? 1.0f / sqrt(sum_sq[r] / float(K) + GEMV_NORM_EPS) : 1.0f;
            MLX_MTL_PRAGMA_UNROLL
            for (int tm = 0; tm < TM; tm++) {
                output[r * N + out_row + tm] = T(GEMV_NORMED ? result[r][tm] * scale : result[r][tm]);
            }
        }
    }
}

// Up to GEMV_MAX_ROWS rows (GEMM_M): the gemv above, on the GEMM's buffers.
template <typename T, typename G>
[[kernel]] void gemv_specialized(
    device       T* output [[buffer(0)]],
    device const T* input  [[buffer(1)]],
    device const T* weight [[buffer(2)]],
    device const G* gain   [[buffer(3)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    // Rows in chunks of GEMV_CHUNK_ROWS, at most GEMV_MAX_ROWS (pipelines.rs) in all, a chunk a
    // threadgroup row (tid.y): a chunk's sums stay in registers, and the chunks of a verify step of
    // up to four 8-row lanes read each weight slice from DRAM once and from cache after. Each row's
    // sum is the 8-row GEMV's.
    constexpr int CHUNK = 8;      // = GEMV_CHUNK_ROWS (pipelines.rs)
    constexpr int MAX_ROWS = 32;  // = GEMV_MAX_ROWS (pipelines.rs)
    if (GEMM_M > uint(MAX_ROWS)) return;
    const int r0 = int(tid.y) * CHUNK;
    if (r0 >= int(GEMM_M)) return;
    constexpr int BN = 8;         // simdgroups per threadgroup, all along K
    constexpr int TM = 4;         // outputs per thread
    threadgroup float tgp[CHUNK * BN * (TM + TM)];
    // The chunk's rows, a constant wherever GEMM_M makes it one (at most a chunk, or whole chunks):
    // the impl's row loops then unroll over registers, as the 8-row GEMV's do.
    const int rows = int(GEMM_M) <= CHUNK         ? int(GEMM_M)
                     : int(GEMM_M) % CHUNK == 0   ? CHUNK
                                                  : min(CHUNK, int(GEMM_M) - r0);
    gemv_specialized_impl<T, G, CHUNK>(
        output + size_t(r0) * uint(GEMM_N), input + size_t(r0) * uint(GEMM_K), weight, gain, tgp,
        rows, tid.x, simd_gid, simd_lid);
}

// `gemv_normed_rows_*`: the normed gemv over EVERY row of an M-row batch — a router's pre-norm
// (`MetalFusion::NormedRouter`) on the multi-row buckets, where the plain Gemm path cannot take
// the fold. Dispatch (ceil(N/4), M, 1): tid.y picks the row; the threadgroup's 256 threads all
// collaborate on that one row, so the sum-of-squares reduce is the one-row kernel's own.
template <typename T, typename G>
[[kernel]] void gemv_specialized_rows(
    device       T* output [[buffer(0)]],
    device const T* input  [[buffer(1)]],
    device const T* weight [[buffer(2)]],
    device const G* gain   [[buffer(3)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    constexpr int BN = 8;   // simdgroups per threadgroup, all along K
    constexpr int TM = 4;   // outputs per thread
    threadgroup float tgp[BN * (TM + TM)];
    gemv_specialized_impl<T, G, 1>(
        output + size_t(tid.y) * uint(GEMM_N),
        input + size_t(tid.y) * uint(GEMM_K),
        weight,
        gain,
        tgp,
        1,
        tid.x,
        simd_gid,
        simd_lid);
}

#define INST_GEMV(tag, T) \
  SCRATCHY_KERNEL(gemv_##tag##_specialized, gemv_specialized<T, T>)

INST_GEMV(f16, half)
INST_GEMV(bf16, bfloat)

#define INST_GEMV_NORMED(act_tag, T, gain_tag, G) \
  SCRATCHY_KERNEL(gemv_normed_##act_tag##_s_##gain_tag, gemv_specialized<T, G>)

INST_GEMV_NORMED(f16, half, f16, half)
INST_GEMV_NORMED(bf16, bfloat, f16, half)
INST_GEMV_NORMED(bf16, bfloat, bf16, bfloat)
INST_GEMV_NORMED(f16, half, bf16, bfloat)

#define INST_GEMV_NORMED_ROWS(act_tag, T, gain_tag, G) \
  SCRATCHY_KERNEL(gemv_normed_rows_##act_tag##_s_##gain_tag, gemv_specialized_rows<T, G>)

INST_GEMV_NORMED_ROWS(f16, half, f16, half)
INST_GEMV_NORMED_ROWS(bf16, bfloat, f16, half)
INST_GEMV_NORMED_ROWS(bf16, bfloat, bf16, bfloat)
INST_GEMV_NORMED_ROWS(f16, half, bf16, bfloat)
