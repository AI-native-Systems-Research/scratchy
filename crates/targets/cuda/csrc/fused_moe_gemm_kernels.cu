/*
 * Fused MoE GEMM kernel — expert-indexed tiled matrix multiplication.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Matches the semantics of Python vLLM's Triton fused_moe_kernel:
 * - Input A: [num_tokens, K] (hidden states)
 * - Weights B: [E, N, K] (stacked expert weights, row-major)
 * - Output C: [num_tokens * top_k, N]
 * - Token dispatch via sorted_token_ids, expert dispatch via expert_ids
 *
 * Kernel variants:
 * 1. BF16 WMMA (SM80+): BF16 tensor core compute via WMMA API, BF16 output
 * 2. F16  WMMA (SM70+): F16 tensor core compute via WMMA API, F16 output
 * 3. FP8  (SM89+): FP8 E4M3 inputs/weights, BF16 output, with per-token/expert scales
 *    - SM89+: True FP8 tensor core compute via PTX mma.sync m16n8k32
 *    - SM80-SM88: FP8 storage + dequant-to-BF16 in shared memory + BF16 WMMA compute
 *    Both paths get the 2x memory bandwidth win from FP8 storage.
 *
 * Tile sizes: BLOCK_M=variable (16/32/64/128), BLOCK_N=128, BLOCK_K=32
 *
 * Variable BLOCK_M: BF16/F16 kernels are templated on BLOCK_M_T to reduce
 * wasted compute during decode (few tokens per expert). Warps are
 * redistributed across M and N dimensions to keep all 8 warps active:
 *   warps_per_m = BLOCK_M_T / WMMA_M
 *   warps_per_n = 8 / warps_per_m
 *   n_tiles_per_warp = 8 / warps_per_n
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;

// Fixed tile sizes (BLOCK_M only used by FP8 SM89 PTX kernel, which is not yet templated)
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 32

// GROUP_SIZE_M: L2 cache locality optimization from Python vLLM's Triton kernel.
// Groups GROUP_SIZE_M consecutive M-blocks with the same N-block before advancing N,
// so B-tile loads hit L2 cache. Python vLLM default = 8.
#define GROUP_SIZE_M 8

// WMMA fragment size (used by BF16, F16, and FP8 dequant kernels)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Thread block size: 256 threads.
#define THREADS_PER_BLOCK 256
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK / 32)

#define WARP_TILES_N (BLOCK_N / WMMA_N)  // 8 (total N-tiles, always fixed)

// FP8 PTX MMA tile sizes: m16n8k32
#define FP8_WMMA_M 16
#define FP8_WMMA_N 8
#define FP8_WMMA_K 32
#define FP8_WARP_TILES_N (BLOCK_N / FP8_WMMA_N)  // 16

namespace vllm {
namespace moe {

// =========================================================================
// BF16 WMMA GEMM kernel — tensor core compute via WMMA API (SM80+)
//
// Faithful port of Python vLLM's Triton fused_moe_kernel.
// Templated on BLOCK_M_T (16/32/64/128) for variable tile height.
// 8 warps redistributed across M and N dimensions:
//   warps_per_m = BLOCK_M_T / 16, warps_per_n = 8 / warps_per_m
// Shared memory: smem_a[BLOCK_M_T][32] + smem_b[32][128] + staging.
// =========================================================================

template <int BLOCK_M_T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
fused_moe_gemm_bf16_wmma(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weights,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    int32_t num_valid_tokens,
    int32_t K,
    int32_t N,
    int32_t top_k,
    int32_t apply_weights)
{
    // Warp-to-tile mapping: redistribute 8 warps across M and N
    constexpr int WARPS_PER_M = BLOCK_M_T / WMMA_M;  // M-stripes available
    constexpr int WARPS_PER_N = WARPS_PER_BLOCK / WARPS_PER_M;  // warps sharing each M-stripe
    constexpr int N_TILES_PER_WARP = WARP_TILES_N / WARPS_PER_N;  // N-tiles each warp handles

    const int32_t total_padded = *num_tokens_post_padded;
    const int32_t num_n_blocks = (N + BLOCK_N - 1) / BLOCK_N;

    // GROUP_SIZE_M swizzle: group M-blocks for L2 locality on B tiles.
    const int32_t num_m_blocks = gridDim.x / num_n_blocks;
    const int32_t num_pid_in_group = GROUP_SIZE_M * num_n_blocks;
    const int32_t group_id = blockIdx.x / num_pid_in_group;
    const int32_t first_pid_m = group_id * GROUP_SIZE_M;
    const int32_t group_size_m = min(num_m_blocks - first_pid_m, GROUP_SIZE_M);
    const int32_t pid_m = first_pid_m + (blockIdx.x % group_size_m);
    const int32_t pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;

    if (pid_m * BLOCK_M_T >= total_padded) return;

    const int32_t expert_id = expert_ids[pid_m];
    if (expert_id < 0) return;

    const __nv_bfloat16* expert_w = weights + (int64_t)expert_id * N * K;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m = warp_id / WARPS_PER_N;
    const int warp_n_start = (warp_id % WARPS_PER_N) * N_TILES_PER_WARP;

    __shared__ __nv_bfloat16 smem_a[BLOCK_M_T][BLOCK_K];
    __shared__ __nv_bfloat16 smem_b[BLOCK_K][BLOCK_N];

    // WMMA accumulators: one per N-tile this warp handles
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[N_TILES_PER_WARP];
    #pragma unroll
    for (int wn = 0; wn < N_TILES_PER_WARP; ++wn) {
        wmma::fill_fragment(acc[wn], 0.0f);
    }

    for (int32_t k_start = 0; k_start < K; k_start += BLOCK_K) {
        // Cooperative load A tile [BLOCK_M_T × BLOCK_K]
        for (int32_t idx = threadIdx.x; idx < BLOCK_M_T * BLOCK_K; idx += THREADS_PER_BLOCK) {
            int32_t m_local = idx / BLOCK_K;
            int32_t k_local = idx % BLOCK_K;
            int32_t global_m = pid_m * BLOCK_M_T + m_local;
            int32_t global_k = k_start + k_local;

            __nv_bfloat16 val = __float2bfloat16(0.0f);
            if (global_m < total_padded && global_k < K) {
                int32_t token_id = sorted_token_ids[global_m];
                if (token_id < num_valid_tokens) {
                    val = input[(int64_t)(token_id / top_k) * K + global_k];
                }
            }
            smem_a[m_local][k_local] = val;
        }

        // Cooperative load B tile [BLOCK_K × BLOCK_N]
        for (int32_t idx = threadIdx.x; idx < BLOCK_K * BLOCK_N; idx += THREADS_PER_BLOCK) {
            int32_t k_local = idx / BLOCK_N;
            int32_t n_local = idx % BLOCK_N;
            int32_t global_k = k_start + k_local;
            int32_t global_n = pid_n * BLOCK_N + n_local;

            __nv_bfloat16 val = __float2bfloat16(0.0f);
            if (global_k < K && global_n < N) {
                val = expert_w[(int64_t)global_n * K + global_k];
            }
            smem_b[k_local][n_local] = val;
        }

        __syncthreads();

        // WMMA compute: each warp processes its M-stripe across its assigned N-tiles
        #pragma unroll
        for (int32_t ki = 0; ki < BLOCK_K; ki += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, &smem_a[warp_m * WMMA_M][ki], BLOCK_K);

            #pragma unroll
            for (int wn = 0; wn < N_TILES_PER_WARP; ++wn) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &smem_b[ki][(warp_n_start + wn) * WMMA_N], BLOCK_N);
                wmma::mma_sync(acc[wn], a_frag, b_frag, acc[wn]);
            }
        }

        __syncthreads();
    }

    // Epilogue: store WMMA fragments to staging, apply routing weight, write output
    __shared__ float warp_staging[WARPS_PER_BLOCK][WMMA_M * WMMA_N];

    #pragma unroll
    for (int wn = 0; wn < N_TILES_PER_WARP; ++wn) {
        wmma::store_matrix_sync(
            &warp_staging[warp_id][0], acc[wn], WMMA_N, wmma::mem_row_major);
        __syncwarp();

        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int32_t m_local = warp_m * WMMA_M + i / WMMA_N;
            int32_t n_local = (warp_n_start + wn) * WMMA_N + i % WMMA_N;
            int32_t global_m = pid_m * BLOCK_M_T + m_local;
            int32_t global_n = pid_n * BLOCK_N + n_local;

            if (global_m >= total_padded || global_n >= N) continue;

            int32_t token_id = sorted_token_ids[global_m];
            if (token_id >= num_valid_tokens) continue;

            float val = warp_staging[warp_id][i];
            if (apply_weights) {
                val *= topk_weights[token_id];
            }
            output[(int64_t)token_id * N + global_n] = __float2bfloat16(val);
        }
    }
}

// =========================================================================
// F16 WMMA GEMM kernel — tensor core compute via WMMA API (SM70+)
//
// Same structure as BF16 WMMA kernel above, with __half types.
// Templated on BLOCK_M_T with redistributed warp-to-tile mapping.
// =========================================================================

template <int BLOCK_M_T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
fused_moe_gemm_f16_wmma(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ weights,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    int32_t num_valid_tokens,
    int32_t K,
    int32_t N,
    int32_t top_k,
    int32_t apply_weights)
{
    constexpr int WARPS_PER_M = BLOCK_M_T / WMMA_M;
    constexpr int WARPS_PER_N = WARPS_PER_BLOCK / WARPS_PER_M;
    constexpr int N_TILES_PER_WARP = WARP_TILES_N / WARPS_PER_N;

    const int32_t total_padded = *num_tokens_post_padded;
    const int32_t num_n_blocks = (N + BLOCK_N - 1) / BLOCK_N;

    // GROUP_SIZE_M swizzle
    const int32_t num_m_blocks = gridDim.x / num_n_blocks;
    const int32_t num_pid_in_group = GROUP_SIZE_M * num_n_blocks;
    const int32_t group_id = blockIdx.x / num_pid_in_group;
    const int32_t first_pid_m = group_id * GROUP_SIZE_M;
    const int32_t group_size_m = min(num_m_blocks - first_pid_m, GROUP_SIZE_M);
    const int32_t pid_m = first_pid_m + (blockIdx.x % group_size_m);
    const int32_t pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;

    if (pid_m * BLOCK_M_T >= total_padded) return;

    const int32_t expert_id = expert_ids[pid_m];
    if (expert_id < 0) return;

    const __half* expert_w = weights + (int64_t)expert_id * N * K;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m = warp_id / WARPS_PER_N;
    const int warp_n_start = (warp_id % WARPS_PER_N) * N_TILES_PER_WARP;

    __shared__ __half smem_a[BLOCK_M_T][BLOCK_K];
    __shared__ __half smem_b[BLOCK_K][BLOCK_N];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[N_TILES_PER_WARP];
    #pragma unroll
    for (int wn = 0; wn < N_TILES_PER_WARP; ++wn) {
        wmma::fill_fragment(acc[wn], 0.0f);
    }

    for (int32_t k_start = 0; k_start < K; k_start += BLOCK_K) {
        for (int32_t idx = threadIdx.x; idx < BLOCK_M_T * BLOCK_K; idx += THREADS_PER_BLOCK) {
            int32_t m_local = idx / BLOCK_K;
            int32_t k_local = idx % BLOCK_K;
            int32_t global_m = pid_m * BLOCK_M_T + m_local;
            int32_t global_k = k_start + k_local;

            __half val = __float2half(0.0f);
            if (global_m < total_padded && global_k < K) {
                int32_t token_id = sorted_token_ids[global_m];
                if (token_id < num_valid_tokens) {
                    val = input[(int64_t)(token_id / top_k) * K + global_k];
                }
            }
            smem_a[m_local][k_local] = val;
        }

        for (int32_t idx = threadIdx.x; idx < BLOCK_K * BLOCK_N; idx += THREADS_PER_BLOCK) {
            int32_t k_local = idx / BLOCK_N;
            int32_t n_local = idx % BLOCK_N;
            int32_t global_k = k_start + k_local;
            int32_t global_n = pid_n * BLOCK_N + n_local;

            __half val = __float2half(0.0f);
            if (global_k < K && global_n < N) {
                val = expert_w[(int64_t)global_n * K + global_k];
            }
            smem_b[k_local][n_local] = val;
        }

        __syncthreads();

        #pragma unroll
        for (int32_t ki = 0; ki < BLOCK_K; ki += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, &smem_a[warp_m * WMMA_M][ki], BLOCK_K);

            #pragma unroll
            for (int wn = 0; wn < N_TILES_PER_WARP; ++wn) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &smem_b[ki][(warp_n_start + wn) * WMMA_N], BLOCK_N);
                wmma::mma_sync(acc[wn], a_frag, b_frag, acc[wn]);
            }
        }

        __syncthreads();
    }

    __shared__ float warp_staging[WARPS_PER_BLOCK][WMMA_M * WMMA_N];

    #pragma unroll
    for (int wn = 0; wn < N_TILES_PER_WARP; ++wn) {
        wmma::store_matrix_sync(
            &warp_staging[warp_id][0], acc[wn], WMMA_N, wmma::mem_row_major);
        __syncwarp();

        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int32_t m_local = warp_m * WMMA_M + i / WMMA_N;
            int32_t n_local = (warp_n_start + wn) * WMMA_N + i % WMMA_N;
            int32_t global_m = pid_m * BLOCK_M_T + m_local;
            int32_t global_n = pid_n * BLOCK_N + n_local;

            if (global_m >= total_padded || global_n >= N) continue;

            int32_t token_id = sorted_token_ids[global_m];
            if (token_id >= num_valid_tokens) continue;

            float val = warp_staging[warp_id][i];
            if (apply_weights) {
                val *= topk_weights[token_id];
            }
            output[(int64_t)token_id * N + global_n] = __float2half(val);
        }
    }
}

// =========================================================================
// FP8 E4M3 fused MoE GEMM kernel
//
// Loads FP8 data from global memory (2x bandwidth savings over BF16).
// On SM89+: uses PTX mma.sync.aligned.m16n8k32 for FP8 tensor core compute.
// On SM80-SM88: dequantizes FP8→BF16 in shared memory, then uses BF16 WMMA.
//
// Epilogue applies: output = acc * a_scale[token] * w_scale[expert] [* topk_weight]
// Output is always BF16.
// =========================================================================

// Helper: dequant FP8 byte → BF16 (no scale, just type conversion).
__device__ __forceinline__ __nv_bfloat16 fp8_to_bf16_raw(uint8_t fp8_byte) {
    __nv_fp8_e4m3 fp8 = *reinterpret_cast<__nv_fp8_e4m3*>(&fp8_byte);
    return __float2bfloat16(float(fp8));
}

// PTX mma.sync for FP8 E4M3: m16n8k32 with f32 accumulation.
// Only available on SM89+ (Ada Lovelace). Guarded at the asm level.
__device__ __forceinline__ void mma_fp8_m16n8k32(
    float (&d)[4],
    const uint32_t (&a)[4],
    const uint32_t (&b)[2],
    const float (&c)[4])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
#else
    // Fallback: should never be called on < SM89.
    d[0] = c[0]; d[1] = c[1]; d[2] = c[2]; d[3] = c[3];
#endif
}

// SM89+ path: true FP8 tensor core compute via PTX mma.sync m16n8k32.
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
fused_moe_gemm_fp8_sm89(
    __nv_bfloat16* __restrict__ output,
    const uint8_t* __restrict__ input,    // FP8 E4M3 [M, K]
    const uint8_t* __restrict__ weights,  // FP8 E4M3 [E, N, K]
    const float* __restrict__ a_scales,   // [M] per-token activation scales
    const float* __restrict__ w_scales,   // [E] per-expert weight scales
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    int32_t num_valid_tokens,
    int32_t K,
    int32_t N,
    int32_t top_k,
    int32_t apply_weights)
{
    const int32_t total_padded = *num_tokens_post_padded;
    const int32_t num_n_blocks = (N + BLOCK_N - 1) / BLOCK_N;
    const int32_t pid_m = blockIdx.x / num_n_blocks;
    const int32_t pid_n = blockIdx.x % num_n_blocks;

    if (pid_m * BLOCK_M >= total_padded) return;

    const int32_t expert_id = expert_ids[pid_m];
    if (expert_id < 0) return;

    const uint8_t* expert_w = weights + (int64_t)expert_id * N * K;
    const float expert_w_scale = w_scales[expert_id];

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m = warp_id;
    const int group_id = lane_id >> 2;
    const int tid_in_group = lane_id & 3;

    // A: row-major [BLOCK_M][BLOCK_K] (1 byte/elem) — row = M, col = K.
    // B: col-major [BLOCK_N][BLOCK_K] (1 byte/elem) — stored as smem_b[n][k],
    //    so each row (fixed n) is contiguous in K, matching PTX col_major B layout.
    __shared__ uint8_t smem_a[BLOCK_M][BLOCK_K];    // 4KB
    __shared__ uint8_t smem_b[BLOCK_N][BLOCK_K];    // 4KB

    float acc[FP8_WARP_TILES_N][4];
    #pragma unroll
    for (int wn = 0; wn < FP8_WARP_TILES_N; ++wn) {
        acc[wn][0] = 0.0f; acc[wn][1] = 0.0f;
        acc[wn][2] = 0.0f; acc[wn][3] = 0.0f;
    }

    for (int32_t k_start = 0; k_start < K; k_start += BLOCK_K) {
        // Load A tile.
        for (int32_t idx = threadIdx.x; idx < BLOCK_M * BLOCK_K; idx += THREADS_PER_BLOCK) {
            int32_t m_local = idx / BLOCK_K;
            int32_t k_local = idx % BLOCK_K;
            int32_t global_m = pid_m * BLOCK_M + m_local;
            int32_t global_k = k_start + k_local;

            uint8_t val = 0;
            if (global_m < total_padded && global_k < K) {
                int32_t token_id = sorted_token_ids[global_m];
                if (token_id < num_valid_tokens * top_k) {
                    val = input[(int64_t)(token_id / top_k) * K + global_k];
                }
            }
            smem_a[m_local][k_local] = val;
        }

        // Load B tile — store as [N][K] so each N-row is K-contiguous (col-major for PTX).
        for (int32_t idx = threadIdx.x; idx < BLOCK_N * BLOCK_K; idx += THREADS_PER_BLOCK) {
            int32_t n_local = idx / BLOCK_K;
            int32_t k_local = idx % BLOCK_K;
            int32_t global_n = pid_n * BLOCK_N + n_local;
            int32_t global_k = k_start + k_local;

            uint8_t val = 0;
            if (global_n < N && global_k < K) {
                val = expert_w[(int64_t)global_n * K + global_k];
            }
            smem_b[n_local][k_local] = val;
        }

        __syncthreads();

        // BLOCK_K / FP8_WMMA_K = 32 / 32 = 1 MMA step per K-tile.
        // Load A fragment from smem_a (row-major).
        // PTX m16n8k32 .row A layout: 4 × uint32 per thread.
        // Thread (group_id, tid_in_group):
        //   a[0] = A[group_id,   tid_in_group*4 .. tid_in_group*4+3]
        //   a[1] = A[group_id,   16+tid_in_group*4 .. 16+tid_in_group*4+3]
        //   a[2] = A[group_id+8, tid_in_group*4 .. tid_in_group*4+3]
        //   a[3] = A[group_id+8, 16+tid_in_group*4 .. 16+tid_in_group*4+3]
        const int a_row0 = warp_m * FP8_WMMA_M + group_id;
        const int a_row1 = a_row0 + 8;

        uint32_t a_reg[4];
        a_reg[0] = *reinterpret_cast<const uint32_t*>(&smem_a[a_row0][tid_in_group * 4]);
        a_reg[1] = *reinterpret_cast<const uint32_t*>(&smem_a[a_row0][16 + tid_in_group * 4]);
        a_reg[2] = *reinterpret_cast<const uint32_t*>(&smem_a[a_row1][tid_in_group * 4]);
        a_reg[3] = *reinterpret_cast<const uint32_t*>(&smem_a[a_row1][16 + tid_in_group * 4]);

        #pragma unroll
        for (int wn = 0; wn < FP8_WARP_TILES_N; ++wn) {
            // Load B fragment from smem_b (col-major = [N][K] with K contiguous).
            // PTX m16n8k32 .col B layout: 2 × uint32 per thread.
            // Thread (group_id, tid_in_group):
            //   b[0] = B[tid_in_group*4..tid_in_group*4+3, group_id]
            //         = smem_b[wn*8 + group_id][tid_in_group*4 .. tid_in_group*4+3]
            //   b[1] = B[16+tid_in_group*4..16+tid_in_group*4+3, group_id]
            //         = smem_b[wn*8 + group_id][16 + tid_in_group*4 .. 16+tid_in_group*4+3]
            const int b_n = wn * FP8_WMMA_N + group_id;
            uint32_t b_reg[2];
            b_reg[0] = *reinterpret_cast<const uint32_t*>(&smem_b[b_n][tid_in_group * 4]);
            b_reg[1] = *reinterpret_cast<const uint32_t*>(&smem_b[b_n][16 + tid_in_group * 4]);

            mma_fp8_m16n8k32(acc[wn], a_reg, b_reg, acc[wn]);
        }

        __syncthreads();
    }

    // Epilogue: write results with scale application.
    // m16n8 output mapping per thread:
    //   d[0] = C[groupID,     tid_in_group * 2]
    //   d[1] = C[groupID,     tid_in_group * 2 + 1]
    //   d[2] = C[groupID + 8, tid_in_group * 2]
    //   d[3] = C[groupID + 8, tid_in_group * 2 + 1]
    #pragma unroll
    for (int wn = 0; wn < FP8_WARP_TILES_N; ++wn) {
        int32_t base_n = pid_n * BLOCK_N + wn * FP8_WMMA_N;
        int32_t local_n0 = tid_in_group * 2;
        int32_t local_n1 = local_n0 + 1;

        // Rows 0..7
        {
            int32_t global_m = pid_m * BLOCK_M + warp_m * FP8_WMMA_M + group_id;
            if (global_m < total_padded) {
                int32_t token_id = sorted_token_ids[global_m];
                if (token_id < num_valid_tokens * top_k) {
                    float scale = a_scales[token_id / top_k] * expert_w_scale;
                    float v0 = acc[wn][0] * scale;
                    float v1 = acc[wn][1] * scale;
                    if (apply_weights) {
                        float tw = topk_weights[token_id];
                        v0 *= tw; v1 *= tw;
                    }
                    int32_t gn0 = base_n + local_n0;
                    int32_t gn1 = base_n + local_n1;
                    if (gn0 < N) output[(int64_t)token_id * N + gn0] = __float2bfloat16(v0);
                    if (gn1 < N) output[(int64_t)token_id * N + gn1] = __float2bfloat16(v1);
                }
            }
        }
        // Rows 8..15
        {
            int32_t global_m = pid_m * BLOCK_M + warp_m * FP8_WMMA_M + group_id + 8;
            if (global_m < total_padded) {
                int32_t token_id = sorted_token_ids[global_m];
                if (token_id < num_valid_tokens * top_k) {
                    float scale = a_scales[token_id / top_k] * expert_w_scale;
                    float v2 = acc[wn][2] * scale;
                    float v3 = acc[wn][3] * scale;
                    if (apply_weights) {
                        float tw = topk_weights[token_id];
                        v2 *= tw; v3 *= tw;
                    }
                    int32_t gn0 = base_n + local_n0;
                    int32_t gn1 = base_n + local_n1;
                    if (gn0 < N) output[(int64_t)token_id * N + gn0] = __float2bfloat16(v2);
                    if (gn1 < N) output[(int64_t)token_id * N + gn1] = __float2bfloat16(v3);
                }
            }
        }
    }
}

// SM80+ fallback: load FP8 from global, dequant to BF16 in shared memory, BF16 WMMA compute.
// Same interface as SM89 kernel. Still gets 2x bandwidth savings from FP8 storage.
// Templated on BLOCK_M_T with warp redistribution + GROUP_SIZE_M swizzle.
template <int BLOCK_M_T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
fused_moe_gemm_fp8_dequant_wmma(
    __nv_bfloat16* __restrict__ output,
    const uint8_t* __restrict__ input,    // FP8 E4M3 [M, K]
    const uint8_t* __restrict__ weights,  // FP8 E4M3 [E, N, K]
    const float* __restrict__ a_scales,   // [M] per-token activation scales
    const float* __restrict__ w_scales,   // [E] per-expert weight scales
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    int32_t num_valid_tokens,
    int32_t K,
    int32_t N,
    int32_t top_k,
    int32_t apply_weights)
{
    constexpr int WARPS_PER_M = BLOCK_M_T / WMMA_M;
    constexpr int WARPS_PER_N = WARPS_PER_BLOCK / WARPS_PER_M;
    constexpr int N_TILES_PER_WARP = WARP_TILES_N / WARPS_PER_N;

    const int32_t total_padded = *num_tokens_post_padded;
    const int32_t num_n_blocks = (N + BLOCK_N - 1) / BLOCK_N;

    // GROUP_SIZE_M swizzle
    const int32_t num_m_blocks = gridDim.x / num_n_blocks;
    const int32_t num_pid_in_group = GROUP_SIZE_M * num_n_blocks;
    const int32_t group_id = blockIdx.x / num_pid_in_group;
    const int32_t first_pid_m = group_id * GROUP_SIZE_M;
    const int32_t group_size_m = min(num_m_blocks - first_pid_m, GROUP_SIZE_M);
    const int32_t pid_m = first_pid_m + (blockIdx.x % group_size_m);
    const int32_t pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;

    if (pid_m * BLOCK_M_T >= total_padded) return;

    const int32_t expert_id = expert_ids[pid_m];
    if (expert_id < 0) return;

    const uint8_t* expert_w = weights + (int64_t)expert_id * N * K;
    const float expert_w_scale = w_scales[expert_id];

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m = warp_id / WARPS_PER_N;
    const int warp_n_start = (warp_id % WARPS_PER_N) * N_TILES_PER_WARP;

    // Dequantized BF16 tiles in shared memory (same layout as BF16 kernel).
    __shared__ __nv_bfloat16 smem_a[BLOCK_M_T][BLOCK_K];
    __shared__ __nv_bfloat16 smem_b[BLOCK_K][BLOCK_N];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[N_TILES_PER_WARP];
    #pragma unroll
    for (int wn = 0; wn < N_TILES_PER_WARP; ++wn) {
        wmma::fill_fragment(acc[wn], 0.0f);
    }

    for (int32_t k_start = 0; k_start < K; k_start += BLOCK_K) {
        // Load A: FP8 → BF16 dequant in smem. No scale applied here (scale in epilogue).
        for (int32_t idx = threadIdx.x; idx < BLOCK_M_T * BLOCK_K; idx += THREADS_PER_BLOCK) {
            int32_t m_local = idx / BLOCK_K;
            int32_t k_local = idx % BLOCK_K;
            int32_t global_m = pid_m * BLOCK_M_T + m_local;
            int32_t global_k = k_start + k_local;

            __nv_bfloat16 val = __float2bfloat16(0.0f);
            if (global_m < total_padded && global_k < K) {
                int32_t token_id = sorted_token_ids[global_m];
                if (token_id < num_valid_tokens * top_k) {
                    val = fp8_to_bf16_raw(input[(int64_t)(token_id / top_k) * K + global_k]);
                }
            }
            smem_a[m_local][k_local] = val;
        }

        // Load B: FP8 → BF16 dequant in smem.
        for (int32_t idx = threadIdx.x; idx < BLOCK_K * BLOCK_N; idx += THREADS_PER_BLOCK) {
            int32_t k_local = idx / BLOCK_N;
            int32_t n_local = idx % BLOCK_N;
            int32_t global_k = k_start + k_local;
            int32_t global_n = pid_n * BLOCK_N + n_local;

            __nv_bfloat16 val = __float2bfloat16(0.0f);
            if (global_k < K && global_n < N) {
                val = fp8_to_bf16_raw(expert_w[(int64_t)global_n * K + global_k]);
            }
            smem_b[k_local][n_local] = val;
        }

        __syncthreads();

        // Standard BF16 WMMA compute with warp redistribution.
        #pragma unroll
        for (int32_t ki = 0; ki < BLOCK_K; ki += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, &smem_a[warp_m * WMMA_M][ki], BLOCK_K);

            #pragma unroll
            for (int wn = 0; wn < N_TILES_PER_WARP; ++wn) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &smem_b[ki][(warp_n_start + wn) * WMMA_N], BLOCK_N);
                wmma::mma_sync(acc[wn], a_frag, b_frag, acc[wn]);
            }
        }

        __syncthreads();
    }

    // Epilogue: apply a_scale * w_scale, then optional topk_weight.
    __shared__ float warp_staging[WARPS_PER_BLOCK][WMMA_M * WMMA_N];

    #pragma unroll
    for (int wn = 0; wn < N_TILES_PER_WARP; ++wn) {
        wmma::store_matrix_sync(
            &warp_staging[warp_id][0], acc[wn], WMMA_N, wmma::mem_row_major);
        __syncwarp();

        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int32_t m_local = warp_m * WMMA_M + i / WMMA_N;
            int32_t n_local = (warp_n_start + wn) * WMMA_N + i % WMMA_N;
            int32_t global_m = pid_m * BLOCK_M_T + m_local;
            int32_t global_n = pid_n * BLOCK_N + n_local;

            if (global_m >= total_padded || global_n >= N) continue;

            int32_t token_id = sorted_token_ids[global_m];
            if (token_id >= num_valid_tokens * top_k) continue;

            float val = warp_staging[warp_id][i];
            // Apply FP8 dequant scales: a_scale[token] * w_scale[expert].
            val *= a_scales[token_id / top_k] * expert_w_scale;
            if (apply_weights) {
                val *= topk_weights[token_id];
            }
            output[(int64_t)token_id * N + global_n] = __float2bfloat16(val);
        }
    }
}

// SM80+ FP8 block-quantized MoE GEMM — applies per-block weight scales during
// the FP8→BF16 dequant step in shared memory, so WMMA accumulates correctly-
// scaled BF16 values. The epilogue only applies per-token activation scale.
//
// w_scale_inv: [E, ceil(N/block_n), ceil(K/block_k)] f32  (3D block scales)
// scale_stride_e: stride to advance one expert in w_scale_inv
// scale_stride_n: stride to advance one N-block in w_scale_inv
template <int BLOCK_M_T>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
fused_moe_gemm_fp8_block_dequant_wmma(
    __nv_bfloat16* __restrict__ output,
    const uint8_t* __restrict__ input,       // FP8 E4M3 [M, K]
    const uint8_t* __restrict__ weights,     // FP8 E4M3 [E, N, K]
    const float* __restrict__ a_scales,      // [M] per-token activation scales
    const float* __restrict__ w_scale_inv,   // [E, ceil(N/bn), ceil(K/bk)] block scales
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    int32_t num_valid_tokens,
    int32_t K,
    int32_t N,
    int32_t top_k,
    int32_t apply_weights,
    int32_t block_n,           // quantization block size along N
    int32_t block_k,           // quantization block size along K
    int32_t scale_stride_e,    // ceil(N/block_n) * ceil(K/block_k)
    int32_t scale_stride_n)    // ceil(K/block_k)
{
    constexpr int WARPS_PER_M = BLOCK_M_T / WMMA_M;
    constexpr int WARPS_PER_N = WARPS_PER_BLOCK / WARPS_PER_M;
    constexpr int N_TILES_PER_WARP = WARP_TILES_N / WARPS_PER_N;

    const int32_t total_padded = *num_tokens_post_padded;
    const int32_t num_n_blocks = (N + BLOCK_N - 1) / BLOCK_N;

    // GROUP_SIZE_M swizzle
    const int32_t num_m_blocks = gridDim.x / num_n_blocks;
    const int32_t num_pid_in_group = GROUP_SIZE_M * num_n_blocks;
    const int32_t group_id = blockIdx.x / num_pid_in_group;
    const int32_t first_pid_m = group_id * GROUP_SIZE_M;
    const int32_t group_size_m = min(num_m_blocks - first_pid_m, GROUP_SIZE_M);
    const int32_t pid_m = first_pid_m + (blockIdx.x % group_size_m);
    const int32_t pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;

    if (pid_m * BLOCK_M_T >= total_padded) return;

    const int32_t expert_id = expert_ids[pid_m];
    if (expert_id < 0) return;

    const uint8_t* expert_w = weights + (int64_t)expert_id * N * K;
    const float* expert_scale = w_scale_inv + expert_id * scale_stride_e;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m = warp_id / WARPS_PER_N;
    const int warp_n_start = (warp_id % WARPS_PER_N) * N_TILES_PER_WARP;

    __shared__ __nv_bfloat16 smem_a[BLOCK_M_T][BLOCK_K];
    __shared__ __nv_bfloat16 smem_b[BLOCK_K][BLOCK_N];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[N_TILES_PER_WARP];
    #pragma unroll
    for (int wn = 0; wn < N_TILES_PER_WARP; ++wn) {
        wmma::fill_fragment(acc[wn], 0.0f);
    }

    for (int32_t k_start = 0; k_start < K; k_start += BLOCK_K) {
        // Load A: FP8 → BF16 dequant in smem. No scale applied (activation scale in epilogue).
        for (int32_t idx = threadIdx.x; idx < BLOCK_M_T * BLOCK_K; idx += THREADS_PER_BLOCK) {
            int32_t m_local = idx / BLOCK_K;
            int32_t k_local = idx % BLOCK_K;
            int32_t global_m = pid_m * BLOCK_M_T + m_local;
            int32_t global_k = k_start + k_local;

            __nv_bfloat16 val = __float2bfloat16(0.0f);
            if (global_m < total_padded && global_k < K) {
                int32_t token_id = sorted_token_ids[global_m];
                if (token_id < num_valid_tokens * top_k) {
                    val = fp8_to_bf16_raw(input[(int64_t)(token_id / top_k) * K + global_k]);
                }
            }
            smem_a[m_local][k_local] = val;
        }

        // Load B: FP8 → BF16 with per-block weight scale applied during dequant.
        for (int32_t idx = threadIdx.x; idx < BLOCK_K * BLOCK_N; idx += THREADS_PER_BLOCK) {
            int32_t k_local = idx / BLOCK_N;
            int32_t n_local = idx % BLOCK_N;
            int32_t global_k = k_start + k_local;
            int32_t global_n = pid_n * BLOCK_N + n_local;

            __nv_bfloat16 val = __float2bfloat16(0.0f);
            if (global_k < K && global_n < N) {
                __nv_fp8_e4m3 fp8 = *reinterpret_cast<const __nv_fp8_e4m3*>(
                    &expert_w[(int64_t)global_n * K + global_k]);
                float raw = float(fp8);
                int32_t sn = global_n / block_n;
                int32_t sk = global_k / block_k;
                float scale = expert_scale[sn * scale_stride_n + sk];
                val = __float2bfloat16(raw * scale);
            }
            smem_b[k_local][n_local] = val;
        }

        __syncthreads();

        // Standard BF16 WMMA compute with warp redistribution.
        #pragma unroll
        for (int32_t ki = 0; ki < BLOCK_K; ki += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, &smem_a[warp_m * WMMA_M][ki], BLOCK_K);

            #pragma unroll
            for (int wn = 0; wn < N_TILES_PER_WARP; ++wn) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &smem_b[ki][(warp_n_start + wn) * WMMA_N], BLOCK_N);
                wmma::mma_sync(acc[wn], a_frag, b_frag, acc[wn]);
            }
        }

        __syncthreads();
    }

    // Epilogue: apply a_scale only (w_scale already applied during B-tile load).
    __shared__ float warp_staging[WARPS_PER_BLOCK][WMMA_M * WMMA_N];

    #pragma unroll
    for (int wn = 0; wn < N_TILES_PER_WARP; ++wn) {
        wmma::store_matrix_sync(
            &warp_staging[warp_id][0], acc[wn], WMMA_N, wmma::mem_row_major);
        __syncwarp();

        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int32_t m_local = warp_m * WMMA_M + i / WMMA_N;
            int32_t n_local = (warp_n_start + wn) * WMMA_N + i % WMMA_N;
            int32_t global_m = pid_m * BLOCK_M_T + m_local;
            int32_t global_n = pid_n * BLOCK_N + n_local;

            if (global_m >= total_padded || global_n >= N) continue;

            int32_t token_id = sorted_token_ids[global_m];
            if (token_id >= num_valid_tokens * top_k) continue;

            float val = warp_staging[warp_id][i];
            // Only activation scale — weight block scale was applied during load.
            val *= a_scales[token_id / top_k];
            if (apply_weights) {
                val *= topk_weights[token_id];
            }
            output[(int64_t)token_id * N + global_n] = __float2bfloat16(val);
        }
    }
}

} // namespace moe
} // namespace vllm

// =====================================================================
// extern "C" launchers
// =====================================================================

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

extern "C" void fused_moe_gemm_bf16(
    void* output,
    const void* input,
    const void* weights,
    const float* topk_weights,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded,
    int num_valid_tokens,
    int in_features,
    int out_features,
    int top_k,
    int num_tokens_padded_total,
    int apply_weights,
    int block_m,
    cudaStream_t stream)
{
    int max_m_blocks = CEILDIV(num_tokens_padded_total, block_m);
    int num_n_blocks = CEILDIV(out_features, BLOCK_N);
    int grid = max_m_blocks * num_n_blocks;

    #define LAUNCH_BF16(BM) \
        vllm::moe::fused_moe_gemm_bf16_wmma<BM> \
            <<<grid, THREADS_PER_BLOCK, 0, stream>>>( \
                reinterpret_cast<__nv_bfloat16*>(output), \
                reinterpret_cast<const __nv_bfloat16*>(input), \
                reinterpret_cast<const __nv_bfloat16*>(weights), \
                topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, \
                num_valid_tokens, in_features, out_features, top_k, apply_weights)

    switch (block_m) {
        case 16:  LAUNCH_BF16(16);  break;
        case 32:  LAUNCH_BF16(32);  break;
        case 64:  LAUNCH_BF16(64);  break;
        case 128: LAUNCH_BF16(128); break;
        default:  LAUNCH_BF16(128); break;
    }
    #undef LAUNCH_BF16
}

extern "C" void fused_moe_gemm_f16(
    void* output,
    const void* input,
    const void* weights,
    const float* topk_weights,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded,
    int num_valid_tokens,
    int in_features,
    int out_features,
    int top_k,
    int num_tokens_padded_total,
    int apply_weights,
    int block_m,
    cudaStream_t stream)
{
    int max_m_blocks = CEILDIV(num_tokens_padded_total, block_m);
    int num_n_blocks = CEILDIV(out_features, BLOCK_N);
    int grid = max_m_blocks * num_n_blocks;

    #define LAUNCH_F16(BM) \
        vllm::moe::fused_moe_gemm_f16_wmma<BM> \
            <<<grid, THREADS_PER_BLOCK, 0, stream>>>( \
                reinterpret_cast<__half*>(output), \
                reinterpret_cast<const __half*>(input), \
                reinterpret_cast<const __half*>(weights), \
                topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, \
                num_valid_tokens, in_features, out_features, top_k, apply_weights)

    switch (block_m) {
        case 16:  LAUNCH_F16(16);  break;
        case 32:  LAUNCH_F16(32);  break;
        case 64:  LAUNCH_F16(64);  break;
        case 128: LAUNCH_F16(128); break;
        default:  LAUNCH_F16(128); break;
    }
    #undef LAUNCH_F16
}

// FP8 E4M3 fused MoE GEMM — SM89+ path (true FP8 tensor cores).
// On SM89+, uses PTX mma.sync m16n8k32 for 2x compute throughput.
// On pre-SM89, this is a stub (never called — Rust dispatches to dequant path).
extern "C" void fused_moe_fp8_gemm_sm89(
    void* output,
    const void* input,
    const void* weights,
    const float* a_scales,
    const float* w_scales,
    const float* topk_weights,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded,
    int num_valid_tokens,
    int in_features,
    int out_features,
    int top_k,
    int apply_weights,
    cudaStream_t stream)
{
    int max_m_blocks = CEILDIV(num_valid_tokens * top_k, BLOCK_M) + 64;
    int num_n_blocks = CEILDIV(out_features, BLOCK_N);
    int grid = max_m_blocks * num_n_blocks;

    vllm::moe::fused_moe_gemm_fp8_sm89
        <<<grid, THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(output),
            reinterpret_cast<const uint8_t*>(input),
            reinterpret_cast<const uint8_t*>(weights),
            a_scales, w_scales, topk_weights,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            num_valid_tokens, in_features, out_features, top_k, apply_weights);
}

// FP8 E4M3 fused MoE GEMM — SM80+ dequant fallback path.
// Loads FP8 from global, dequants to BF16 in shared memory, uses BF16 WMMA.
// Still gets 2x memory bandwidth savings from FP8 storage.
extern "C" void fused_moe_fp8_gemm_dequant(
    void* output,
    const void* input,
    const void* weights,
    const float* a_scales,
    const float* w_scales,
    const float* topk_weights,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded,
    int num_valid_tokens,
    int in_features,
    int out_features,
    int top_k,
    int apply_weights,
    int block_m,
    cudaStream_t stream)
{
    int max_m_blocks = CEILDIV(num_valid_tokens * top_k, block_m) + 64;
    int num_n_blocks = CEILDIV(out_features, BLOCK_N);
    int grid = max_m_blocks * num_n_blocks;

    #define LAUNCH_FP8_DEQUANT(BM) \
        vllm::moe::fused_moe_gemm_fp8_dequant_wmma<BM> \
            <<<grid, THREADS_PER_BLOCK, 0, stream>>>( \
                reinterpret_cast<__nv_bfloat16*>(output), \
                reinterpret_cast<const uint8_t*>(input), \
                reinterpret_cast<const uint8_t*>(weights), \
                a_scales, w_scales, topk_weights, \
                sorted_token_ids, expert_ids, num_tokens_post_padded, \
                num_valid_tokens, in_features, out_features, top_k, apply_weights)

    switch (block_m) {
        case 16:  LAUNCH_FP8_DEQUANT(16);  break;
        case 32:  LAUNCH_FP8_DEQUANT(32);  break;
        case 64:  LAUNCH_FP8_DEQUANT(64);  break;
        case 128: LAUNCH_FP8_DEQUANT(128); break;
        default:  LAUNCH_FP8_DEQUANT(128); break;
    }
    #undef LAUNCH_FP8_DEQUANT
}

// FP8 block-quantized fused MoE GEMM — per-block weight scales applied during dequant.
extern "C" void fused_moe_fp8_block_gemm_dequant(
    void* output,
    const void* input,
    const void* weights,
    const float* a_scales,
    const float* w_scale_inv,
    const float* topk_weights,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded,
    int num_valid_tokens,
    int in_features,
    int out_features,
    int top_k,
    int apply_weights,
    int block_m,
    int quant_block_n,
    int quant_block_k,
    int scale_stride_e,
    int scale_stride_n,
    cudaStream_t stream)
{
    int max_m_blocks = CEILDIV(num_valid_tokens * top_k, block_m) + 64;
    int num_n_blocks = CEILDIV(out_features, BLOCK_N);
    int grid = max_m_blocks * num_n_blocks;

    #define LAUNCH_FP8_BLOCK_DEQUANT(BM) \
        vllm::moe::fused_moe_gemm_fp8_block_dequant_wmma<BM> \
            <<<grid, THREADS_PER_BLOCK, 0, stream>>>( \
                reinterpret_cast<__nv_bfloat16*>(output), \
                reinterpret_cast<const uint8_t*>(input), \
                reinterpret_cast<const uint8_t*>(weights), \
                a_scales, w_scale_inv, topk_weights, \
                sorted_token_ids, expert_ids, num_tokens_post_padded, \
                num_valid_tokens, in_features, out_features, top_k, apply_weights, \
                quant_block_n, quant_block_k, scale_stride_e, scale_stride_n)

    switch (block_m) {
        case 16:  LAUNCH_FP8_BLOCK_DEQUANT(16);  break;
        case 32:  LAUNCH_FP8_BLOCK_DEQUANT(32);  break;
        case 64:  LAUNCH_FP8_BLOCK_DEQUANT(64);  break;
        case 128: LAUNCH_FP8_BLOCK_DEQUANT(128); break;
        default:  LAUNCH_FP8_BLOCK_DEQUANT(128); break;
    }
    #undef LAUNCH_FP8_BLOCK_DEQUANT
}

// =====================================================================
// Gather kernels for unfused (graph-capture-safe) MoE path
// =====================================================================

// Gather top_k experts' outputs for each token from all-expert buffer.
// all_expert_out: [num_experts, num_tokens, out_features] (BF16, row-major)
// topk_ids:      [num_tokens * top_k] (int32, flat view of [num_tokens, top_k])
// output:        [num_tokens * top_k, out_features] (BF16)
__global__ void moe_expert_gather_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ all_expert_out,
    const int32_t* __restrict__ topk_ids,
    int num_tokens, int top_k, int out_features)
{
    int row = blockIdx.x;
    if (row >= num_tokens * top_k) return;
    int t = row / top_k;
    int expert_id = topk_ids[row];
    const __nv_bfloat16* src = all_expert_out + ((int64_t)expert_id * num_tokens + t) * out_features;
    __nv_bfloat16* dst = output + (int64_t)row * out_features;
    for (int f = threadIdx.x; f < out_features; f += blockDim.x)
        dst[f] = src[f];
}

extern "C" void moe_expert_gather_bf16(
    void* output, const void* all_expert_out, const int32_t* topk_ids,
    int num_tokens, int top_k, int out_features, cudaStream_t stream)
{
    int num_rows = num_tokens * top_k;
    int threads = out_features < 256 ? out_features : 256;
    moe_expert_gather_bf16_kernel<<<num_rows, threads, 0, stream>>>(
        (__nv_bfloat16*)output, (const __nv_bfloat16*)all_expert_out,
        topk_ids, num_tokens, top_k, out_features);
}

// Gather top_k experts' w2 outputs, weight, and sum across top_k.
// all_expert_out: [num_experts, rows_per_expert, out_features] (BF16)
//   Each expert has rows_per_expert rows. Token t's data for expert e
//   is at row (e * rows_per_expert + t).
// topk_ids:      [num_tokens * top_k] (int32)
// topk_weights:  [num_tokens * top_k] (float32)
// output:        [num_tokens, out_features] (BF16)
__global__ void moe_expert_gather_weighted_sum_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ all_expert_out,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    int num_tokens, int top_k, int out_features, int rows_per_expert)
{
    int t = blockIdx.x;
    if (t >= num_tokens) return;
    __nv_bfloat16* dst = output + (int64_t)t * out_features;
    for (int f = threadIdx.x; f < out_features; f += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            int idx = t * top_k + k;
            int expert_id = topk_ids[idx];
            float w = topk_weights[idx];
            acc += w * __bfloat162float(
                all_expert_out[((int64_t)expert_id * rows_per_expert + t) * out_features + f]);
        }
        dst[f] = __float2bfloat16(acc);
    }
}

extern "C" void moe_expert_gather_weighted_sum_bf16(
    void* output, const void* all_expert_out,
    const int32_t* topk_ids, const float* topk_weights,
    int num_tokens, int top_k, int out_features, int rows_per_expert,
    cudaStream_t stream)
{
    int threads = out_features < 256 ? out_features : 256;
    moe_expert_gather_weighted_sum_bf16_kernel<<<num_tokens, threads, 0, stream>>>(
        (__nv_bfloat16*)output, (const __nv_bfloat16*)all_expert_out,
        topk_ids, topk_weights, num_tokens, top_k, out_features, rows_per_expert);
}
