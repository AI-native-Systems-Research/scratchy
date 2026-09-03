// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
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
// shader stay short and the function-constant bag stay at three
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
// Function constants:
//   0 = M
//   1 = N
//   2 = K
//
// Dispatch: threadgroups (ceil(N/8), ceil(M/8), 1), threads (32, 1, 1).
// One simdgroup per threadgroup; each simdgroup computes an 8×8
// output tile.
// ---------------------------------------------------------------------------

constant uint GEMM_M [[function_constant(0)]];
constant uint GEMM_N [[function_constant(1)]];
constant uint GEMM_K [[function_constant(2)]];

// Block-diagonal span attention for the hd512 unfused QKᵀ: granularity of the
// per-position span-label buffer (== metal KV block size). The bound only fires
// where span_ids is non-zero, so non-span tiles keep the full causal range and
// the dense path stays byte-identical. Undefined → 0 (disabled).
constant uint QK_SPAN_BLOCK_RAW [[function_constant(3)]];
constant uint QK_SPAN_BLOCK =
    is_function_constant_defined(QK_SPAN_BLOCK_RAW) ? QK_SPAN_BLOCK_RAW : 0u;

// Shared C = A @ B^T body (A:[M,K], B:[N,K], C:[M,N]) with M/N/K passed
// explicitly. The dense kernel passes the baked GEMM_M/N/K; the hd512 unfused
// attention variants substitute a runtime kv_len (from seq_used) for N (QKᵀ) or
// K (PV) while the tape bakes the grid at the max — tiles past the live N/K
// early-return. This is the pre-M5 (non-NAX) counterpart of gemm_nax_bf16_qk/pv
// and uses the identical [M,N,K] convention, so its output matches bit-for-bit
// modulo accumulation order.
inline void gemm_t_bf16_body(
    device bfloat*       output,
    device const bfloat* input,
    device const bfloat* weight,
    uint M, uint N, uint K,
    uint w_ld,           // weight (B) ROW stride; == K for dense/QKᵀ, but for PV
                         // the V^T dense buffer is strided by static max_kv while
                         // the contraction K = kv_len — they MUST be decoupled.
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
        // with transpose=true. Row stride is w_ld (not K) so PV can read the
        // max_kv-strided V^T while contracting only kv_len columns.
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

// ── Blocked simdgroup GEMM for the hd512 unfused attention (pre-M5) ──────────
// C = A[M,K] @ B[N,K]^T. 32×32 output tile, 4 simdgroups (WM=WN=2), BK=16,
// register-cached 8×8 frags, shared A/B — a single-output adaptation of the MoE
// steel GEMM (`fused_gate_up_silu_mul_gemm_steel`). Far higher reuse than the
// 8×8 reference body (which re-streamed K per output tile). Handles M/N tails
// AND a K-tail (kv_len need NOT be a multiple of 16 — guarded cooperative load).
// `w_ld` decouples the B (V^T) row stride from the contraction K.
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

// hd512 unfused attention (non-NAX / pre-M5) QKᵀ: N = kv_len from seq_used,
// M = GEMM_M (Lq), K = GEMM_K (hd). Output scores [Lq, kv_len].
kernel void gemm_bf16_qk(
    device       bfloat* output   [[buffer(0)]],   // scores [M=Lq, N=kv_len]
    device const bfloat* input    [[buffer(1)]],   // Q head [M=Lq, K=hd]
    device const bfloat* weight   [[buffer(2)]],   // K dense [N=kv_len, K=hd]
    device const uint*   seq_used     [[buffer(3)]],   // [1] kv_len
    device const uint*   span_ids     [[buffer(4)]],   // per-block span label; 0 = no span
    device const uint*   cu_seqlens_q [[buffer(5)]],   // new-query span (ACTUAL, not bucket)
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint3 tgid          [[threadgroup_position_in_grid]])
{
    const uint M = GEMM_M;
    const uint N = seq_used[0];
    const uint c_row = tgid.y * ATTN_BM;  // query-tile
    const uint c_col = tgid.x * ATTN_BN;  // key-tile
    // Block-diagonal span bound: a span-uniform query-tile attends only its own
    // span's keys [span_lo, causal]. Skip both bands BEFORE the hd contraction
    // (real FLOP cut, mirroring steel's kb_start). span_ids 0 or a tile that
    // straddles a span boundary → full causal range (byte-identical).
    if (QK_SPAN_BLOCK != 0u) {
        // Causal offset from the ACTUAL new-query count (cu_seqlens), NOT GEMM_M:
        // GEMM_M is the over-sized prefill bucket (bucket_m can exceed kv_len), so
        // N - GEMM_M underflows. lq_actual ≤ kv_len ⇒ no underflow. Query gemm-row
        // r → absolute key position (kv_len - lq_actual) + r.
        const uint lq_actual = cu_seqlens_q[1] - cu_seqlens_q[0];
        const uint causal_off = N - lq_actual;
        const uint sf = span_ids[causal_off + c_row]; // per-TOKEN index
        const uint sl = span_ids[causal_off + c_row + ATTN_BM - 1u];
        if (sf != 0u && sf == sl) {
            // causal upper: key-tile entirely past the last query's causal key
            if (c_col >= causal_off + c_row + ATTN_BM) {
                return;
            }
            // span lower: key-tile entirely before the span's first token
            const uint span_lo = sf - 1u; // label-1 IS the span's first TOKEN (exact)
            if (c_col + ATTN_BN <= span_lo) {
                return;
            }
        }
    }
    threadgroup bfloat As[ATTN_BM * ATTN_LD];
    threadgroup bfloat Bs[ATTN_BN * ATTN_LD];
    threadgroup float  c_scratch[ATTN_BM * ATTN_BN];
    // QKᵀ: N=kv_len dynamic; kdense row stride == hd == GEMM_K.
    gemm_t_bf16_blocked(output, input, weight, M, N, GEMM_K, GEMM_K,
                        As, Bs, c_scratch, simd_group_id, simd_lane_id, tgid);
}

// hd512 unfused attention (non-NAX / pre-M5) PV: K = kv_len from seq_used,
// M = GEMM_M (Lq), N = GEMM_N (hd), and the V^T dense weight is strided by the
// STATIC max_kv passed in GEMM_K (decoupled from the kv_len contraction).
kernel void gemm_bf16_pv(
    device       bfloat* output   [[buffer(0)]],   // out head [M=Lq, N=hd]
    device const bfloat* input    [[buffer(1)]],   // probs [M=Lq, K=kv_len]
    device const bfloat* weight   [[buffer(2)]],   // V^T dense [N=hd, rows strided by max_kv]
    device const uint*   seq_used [[buffer(3)]],   // [1] kv_len
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint3 tgid          [[threadgroup_position_in_grid]])
{
    threadgroup bfloat As[ATTN_BM * ATTN_LD];
    threadgroup bfloat Bs[ATTN_BN * ATTN_LD];
    threadgroup float  c_scratch[ATTN_BM * ATTN_BN];
    gemm_t_bf16_blocked(output, input, weight, GEMM_M, GEMM_N, seq_used[0], GEMM_K,
                        As, Bs, c_scratch, simd_group_id, simd_lane_id, tgid);
}
