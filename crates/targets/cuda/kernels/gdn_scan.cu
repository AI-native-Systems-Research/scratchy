// SPDX-License-Identifier: Apache-2.0
//
// Gated-DeltaNet recurrent gated delta-rule scan for the H100, one prefill chunk of one sequence:
// launch 3 of the four a `GatedDeltaNet` step makes (`src/step.rs`). Per value head `h`, the state
// `S [head_v, head_k]` (f32) starts from the slot's (or zero on a fresh start), and per real token
// `t < q_len`, in order (metal's `gdn_scan_varlen.metal:155-245`, `cpu_golden::gdn_recurrent`):
//
//     q̂ = q · scale / sqrt(‖q‖² + 1e-6),  k̂ = k / sqrt(‖k‖² + 1e-6)    (key head h / (nv/nk))
//     S ← S · exp(g)                       the decay, applied first
//     u ← β · (v − S · k̂)                  the delta-rule correction
//     S ← S + u ⊗ k̂                        the rank-1 update
//     o ← S · q̂
//
// `q`, `k` and `v` are the f32 conv output's column slices (`q | k | v`, `[rows, conv_dim]`), `g`
// and `β` the gating's (`[rows, nv]`), `o` is f32 `[rows, value_dim]`, all the step's scratch. `S`
// is written back to the slot once, after the last real token.
//
// Plain CUDA, not ThunderKittens: there is no tile-shaped matmul. Each token is three
// 128-long dot products per value row and a rank-1 update; ThunderKittens' tiles are 16×16 at
// least, and its linear-attention kernels have no correction term, no β and no state in or out.
//
// THE MAPPING (the issue's; metal's `gdn_scan_pipelined.metal` is the closest): grid (head_v / 32,
// nv), 128 threads per CTA = 32 value rows of one value head × 4 lanes per row. Lane `l` of a row
// keeps `S[row, 32l .. 32l + 32)` in 32 f32 registers, indexed by unrolled constants only, so the
// state never spills. A row's four lanes are consecutive threads of one warp, so its dot products
// are each lane's 32 products summed in ascending order, then an xor butterfly over offsets 1 and
// 2: the same order in every row and every token. A float add is commutative, so the four lanes
// end with the same bits.
//
// STAGING: the CTA stages blocks of 32 tokens in shared memory, `q` and `k` of its key head
// (`32 · 256` f32), its 32 rows' `v` (`32 · 32`), and `g` and `β` (`32 + 32`): 37,120 bytes,
// under the 48 KiB of static shared memory a launch gets with no attribute. Plain loads and two
// `__syncthreads` per block (no TMA), so racecheck sees every access. Only real tokens are staged:
// no thread reads a padding row of any input.
//
// THE STEP-5B LESSON: the state pool is never cleared, so a fresh slot may hold NaN. A fresh start
// sets the registers to 0 by a select and never loads the slot: `0 · NaN` is NaN, so no masking
// multiply would do. A continued start loads the slot.
//
// CONTINUITY: each token's arithmetic depends only on the state before it and its own inputs, not
// on where it sits in a chunk or a staging block, so one forward of a sequence and the same
// sequence in chunks give the same bits, the state included.
//
// Built with `--use_fast_math` (build.rs): `expf` is `__expf` and `rsqrtf` the approximate one,
// subnormals flush to zero. The long-recurrence bound allows for both.

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

// The launcher's return code that is not a `cudaError_t` (which is non-negative). `src/kernels.rs`
// mirrors it.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)

namespace {

// `step::GDN_HEAD_K`: 4 lanes × 32 registers.
constexpr int HEAD_K = 128;
constexpr int LANES = 4;
constexpr int PER_LANE = HEAD_K / LANES;
// `step::GDN_HEAD_V_MULTIPLE`: the value rows of one CTA.
constexpr int ROWS = 32;
constexpr int THREADS = ROWS * LANES;
// `step::GDN_HEAD_V_MAX`.
constexpr uint32_t HEAD_V_MAX = 256;
// `step::GDN_SCAN_TOKEN_BLOCK`.
constexpr int TOKENS = 32;
// The L2 norms' eps, inside the sqrt (metal's and fla's).
constexpr float L2_EPS = 1e-6f;

// The row's four lanes' partials, summed in a fixed order: (l0 + l1) + (l2 + l3), on every lane.
__device__ __forceinline__ float row_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v;
}

__global__ void __launch_bounds__(THREADS)
gdn_scan_kernel(const float* __restrict__ conv_out, const float* __restrict__ g,
                const float* __restrict__ beta, float* __restrict__ ssm_state,
                unsigned long long slot_elems, const uint32_t* __restrict__ sequence,
                float* __restrict__ o, int conv_dim, int nk, int nv, int head_v, float scale) {
    __shared__ float qs[TOKENS][HEAD_K];
    __shared__ float ks[TOKENS][HEAD_K];
    __shared__ float vs[TOKENS][ROWS];
    __shared__ float gs[TOKENS];
    __shared__ float bs[TOKENS];

    const int h = blockIdx.y;
    const int kh = h / (nv / nk);
    const int row0 = blockIdx.x * ROWS;
    const int r = threadIdx.x / LANES;
    const int lane = threadIdx.x % LANES;
    const int i = row0 + r;  // this row's value dim within head h
    const int key_dim = nk * HEAD_K;
    const int value_dim = nv * head_v;

    const uint32_t slot = sequence[0];
    const int q_len = int(sequence[1]);
    const bool fresh = sequence[2] == 1u;
    float* state = ssm_state + (unsigned long long)slot * slot_elems +
                   (size_t(h) * head_v + i) * HEAD_K + lane * PER_LANE;

    float s[PER_LANE];
    if (fresh) {
        // A select: the slot is not read.
#pragma unroll
        for (int j = 0; j < PER_LANE; ++j) {
            s[j] = 0.0f;
        }
    } else {
#pragma unroll
        for (int j = 0; j < PER_LANE; ++j) {
            s[j] = state[j];
        }
    }

    for (int tb = 0; tb < q_len; tb += TOKENS) {
        const int nt = min(TOKENS, q_len - tb);
        // Stage the block's real tokens: q and k of key head kh, v of the CTA's rows, g and β.
        for (int e = threadIdx.x; e < nt * HEAD_K; e += THREADS) {
            const int tt = e / HEAD_K, j = e % HEAD_K;
            const float* row = conv_out + size_t(tb + tt) * conv_dim;
            qs[tt][j] = row[kh * HEAD_K + j];
            ks[tt][j] = row[key_dim + kh * HEAD_K + j];
        }
        for (int e = threadIdx.x; e < nt * ROWS; e += THREADS) {
            const int tt = e / ROWS, rr = e % ROWS;
            vs[tt][rr] = conv_out[size_t(tb + tt) * conv_dim + 2 * key_dim + h * head_v + row0 + rr];
        }
        for (int tt = threadIdx.x; tt < nt; tt += THREADS) {
            gs[tt] = g[size_t(tb + tt) * nv + h];
            bs[tt] = beta[size_t(tb + tt) * nv + h];
        }
        __syncthreads();

        for (int tt = 0; tt < nt; ++tt) {
            const float* q = &qs[tt][lane * PER_LANE];
            const float* k = &ks[tt][lane * PER_LANE];
            // 1. The norms of q and k.
            float qq = 0.0f, kk = 0.0f;
#pragma unroll
            for (int j = 0; j < PER_LANE; ++j) {
                qq = fmaf(q[j], q[j], qq);
                kk = fmaf(k[j], k[j], kk);
            }
            qq = row_sum(qq);
            kk = row_sum(kk);
            // 2. q̂ = q · scale / sqrt(‖q‖² + eps), k̂ = k / sqrt(‖k‖² + eps).
            const float q_inv = rsqrtf(qq + L2_EPS) * scale;
            const float k_inv = rsqrtf(kk + L2_EPS);
            // 3. The decay, first.
            const float decay = expf(gs[tt]);
#pragma unroll
            for (int j = 0; j < PER_LANE; ++j) {
                s[j] *= decay;
            }
            // 4. S · k̂.
            float sk = 0.0f;
#pragma unroll
            for (int j = 0; j < PER_LANE; ++j) {
                sk = fmaf(s[j], k[j] * k_inv, sk);
            }
            sk = row_sum(sk);
            // 5. The correction: write only what S does not already predict.
            const float u = (vs[tt][r] - sk) * bs[tt];
            // 6. The rank-1 update, and 7. o = S · q̂.
            float out = 0.0f;
#pragma unroll
            for (int j = 0; j < PER_LANE; ++j) {
                s[j] = fmaf(u, k[j] * k_inv, s[j]);
                out = fmaf(s[j], q[j] * q_inv, out);
            }
            out = row_sum(out);
            if (lane == 0) {
                o[size_t(tb + tt) * value_dim + h * head_v + i] = out;
            }
        }
        // The next block's staging overwrites what this one read.
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < PER_LANE; ++j) {
        state[j] = s[j];
    }
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched; any other value = the
// cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `head_k` must be 128, `head_v` a non-zero multiple of 32 and at most 256, `nk` and `nv` non-zero
// with `nv` a multiple of `nk`, `rows` non-zero, `conv_dim` exactly `2 · nk · 128 + nv · head_v`,
// `rows · conv_dim` at most INT_MAX, `scale` finite and above 0, and `slot_elems`
// (`GdnStateLayout::ssm_elems_per_slot`) exactly `nv · head_v · 128`; all are checked before any
// pointer is touched. `conv_out` is `[rows, conv_dim]` f32, `g` and `beta` `[rows, nv]` f32, `o`
// `[rows, nv · head_v]` f32, `sequence` `[slot, q_len, start, 0]` u32 with `1 ≤ q_len ≤ rows`, and
// `ssm_state` one layer's state buffer holding slot `slot`'s `slot_elems` f32 at
// `slot · slot_elems`. `o` and `ssm_state` overlap nothing else. 37,120 bytes of static shared
// memory, under the default 48 KiB, so no one-time setup. The launch is asynchronous on `stream`.
extern "C" int scratchy_cuda_gdn_scan(const void* conv_out, const void* g, const void* beta,
                                      void* ssm_state, uint64_t slot_elems, const void* sequence,
                                      void* o, uint32_t rows, uint32_t nk, uint32_t nv,
                                      uint32_t head_k, uint32_t head_v, float scale,
                                      void* stream /* cudaStream_t; null = default stream */) {
    if (head_k != uint32_t(HEAD_K) || head_v == 0 || head_v % ROWS != 0 || head_v > HEAD_V_MAX ||
        nk == 0 || nv == 0 || nv % nk != 0 || rows == 0 || !std::isfinite(scale) ||
        !(scale > 0.0f)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const uint64_t conv_dim = 2 * uint64_t(nk) * HEAD_K + uint64_t(nv) * head_v;
    if (uint64_t(rows) * conv_dim > uint64_t(INT_MAX) ||
        slot_elems != uint64_t(nv) * head_v * HEAD_K) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const float* cp = static_cast<const float*>(conv_out);
    const float* gp = static_cast<const float*>(g);
    const float* bp = static_cast<const float*>(beta);
    float* sp = static_cast<float*>(ssm_state);
    unsigned long long se = slot_elems;
    const uint32_t* qp = static_cast<const uint32_t*>(sequence);
    float* op = static_cast<float*>(o);
    int icd = static_cast<int>(conv_dim), ink = static_cast<int>(nk), inv = static_cast<int>(nv);
    int ihv = static_cast<int>(head_v);
    void* args[] = {&cp, &gp, &bp, &sp, &se, &qp, &op, &icd, &ink, &inv, &ihv, &scale};
    const dim3 grid(head_v / ROWS, nv), block(THREADS);
    return static_cast<int>(cudaLaunchKernel(gdn_scan_kernel, grid, block, args, 0,
                                             static_cast<cudaStream_t>(stream)));
}
