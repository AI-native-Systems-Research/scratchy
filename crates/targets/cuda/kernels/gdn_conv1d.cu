// SPDX-License-Identifier: Apache-2.0
//
// Gated-DeltaNet causal depthwise conv1d for the H100, one prefill chunk of one sequence: launch 1
// of the four a `GatedDeltaNet` step makes (`src/step.rs`). For each real token `t < q_len` and
// channel `c < conv_dim` of `qkv [rows, conv_dim]` (bf16, `q | k | v`):
//
//     conv_out[t, c] = SiLU( Σ_{j=0}^{K-1} w[c, j] · x[t − K + 1 + j, c] )     (f32)
//
// where a row before 0 is the slot's conv ring: the last `K − 1` inputs of the sequence's earlier
// chunks, or zero on a fresh start. Then the ring becomes the last `K − 1` entries of
// `[seed ++ x[0 .. q_len)]`, the CORRECT general shift (metal's `gdn_conv1d_varlen.metal`), so
// `forward(T)` then `forward(1)` equals `forward(T + 1)`, a chunk shorter than `K − 1` included.
//
// Plain CUDA, not ThunderKittens: 4 taps per channel, no matrix multiply.
//
// One thread per (channel, block of 64 tokens): grid (ceil(conv_dim / 128), ceil(rows / 64)), 128
// threads. The thread keeps the causal window (`K − 1` f32) in registers and walks its block's
// tokens in order. Its window is seeded
//   - in block 0, from the slot's ring, or with zeros WITHOUT READING THE RING on a fresh start:
//     the pool is never cleared, so a fresh slot may hold anything, NaN included, and a select
//     keeps it out where `0 · ring` would not;
//   - in any other block, from `x` rows `t0 − K + 1 .. t0 − 1`, which are real rows (`t0 ≥ 64 ≥
//     K − 1` and `t0 < q_len`).
// Only block 0's thread then writes the new ring. So each channel's ring has one reader and one
// writer, the same thread, and no race (metal's argument, `gdn_conv1d_varlen.metal:8-25`).
//
// A block with `t0 ≥ q_len` returns before reading anything else, and no thread reads a row at or
// past `q_len`: padding rows are never read, so a NaN in one cannot reach an output. The taps are
// summed in ascending `j`, each token's arithmetic the same wherever the token sits in a chunk, so
// a chunked sequence gives the same bits as one forward of it. The ring holds copies of widened
// bf16 inputs (or the seed's), so it is bit-exact.
//
// Built with `--use_fast_math` (build.rs): `expf` and the division in SiLU are the approximate
// ones; the tests' bound (2⁻¹⁶·(|ref| + Σ|w·x|) + 2⁻²⁴) allows for both.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

// The launcher's return code that is not a `cudaError_t` (which is non-negative). `src/kernels.rs`
// mirrors it.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)

namespace {

constexpr int THREADS = 128;
// `step::GDN_CONV_TOKEN_BLOCK`.
constexpr int TOKEN_BLOCK = 64;
// `step::GDN_CONV_KERNEL_RANGE`.
constexpr uint32_t KMIN = 2;
constexpr uint32_t KMAX = 8;

__global__ void __launch_bounds__(THREADS)
gdn_conv1d_kernel(const __nv_bfloat16* __restrict__ x, const float* __restrict__ w,
                  float* __restrict__ conv_state, unsigned long long slot_elems,
                  const uint32_t* __restrict__ sequence, float* __restrict__ conv_out,
                  int conv_dim, int kernel) {
    const int c = blockIdx.x * THREADS + threadIdx.x;
    if (c >= conv_dim) {
        return;
    }
    const int q_len = int(sequence[1]);
    const int t0 = blockIdx.y * TOKEN_BLOCK;
    if (t0 >= q_len) {
        return;
    }
    const uint32_t slot = sequence[0];
    const bool fresh = sequence[2] == 1u;
    const int state_len = kernel - 1;
    float* ring = conv_state + (unsigned long long)slot * slot_elems + size_t(c) * state_len;
    const size_t stride = size_t(conv_dim);

    float wl[KMAX];
#pragma unroll
    for (int j = 0; j < int(KMAX); ++j) {
        wl[j] = j < kernel ? w[size_t(c) * kernel + j] : 0.0f;
    }

    // The window before row t0: the last `state_len` entries of `[seed ++ x]`. Block 0's seed is
    // kept for the ring write below.
    float seed[KMAX - 1];
    float win[KMAX - 1];
#pragma unroll
    for (int j = 0; j < int(KMAX) - 1; ++j) {
        float v = 0.0f;
        if (j < state_len) {
            if (t0 == 0) {
                // A fresh start keeps the 0: the ring is not read (a branch every thread of the
                // grid takes the same way, not `fresh · ring`).
                if (!fresh) {
                    v = ring[j];
                }
            } else {
                v = __bfloat162float(x[size_t(t0 - state_len + j) * stride + c]);
            }
        }
        seed[j] = v;
        win[j] = v;
    }

    // Every register array is indexed by unrolled constants only (selects, not `win[state_len -
    // 1]`), so the window stays in registers.
    const int t_end = min(t0 + TOKEN_BLOCK, q_len);
    for (int t = t0; t < t_end; ++t) {
        const float xt = __bfloat162float(x[size_t(t) * stride + c]);
        // The taps in ascending j: the window's `state_len`, then the token's own.
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < int(KMAX) - 1; ++j) {
            if (j < state_len) {
                sum = fmaf(wl[j], win[j], sum);
            }
        }
#pragma unroll
        for (int j = 0; j < int(KMAX); ++j) {
            if (j == state_len) {
                sum = fmaf(wl[j], xt, sum);
            }
        }
        conv_out[size_t(t) * stride + c] = sum / (1.0f + expf(-sum));
        // Shift left, the token entering last. Ascending j reads `win[j + 1]` before it is
        // written.
#pragma unroll
        for (int j = 0; j < int(KMAX) - 1; ++j) {
            if (j + 1 < state_len) {
                win[j] = win[j + 1];
            } else if (j + 1 == state_len) {
                win[j] = xt;
            }
        }
    }

    if (t0 == 0) {
        // The new ring: entry j is `[seed ++ x[0 .. q_len)]`'s `q_len + j`-th, a seed entry when
        // `q_len + j < state_len`, else row `q_len + j − state_len` of x, a real row.
#pragma unroll
        for (int j = 0; j < int(KMAX) - 1; ++j) {
            if (j < state_len) {
                const int p = q_len + j - state_len;
                float v = 0.0f;
                if (p < 0) {
#pragma unroll
                    for (int i = 0; i < int(KMAX) - 1; ++i) {
                        if (i == q_len + j) {
                            v = seed[i];
                        }
                    }
                } else {
                    v = __bfloat162float(x[size_t(p) * stride + c]);
                }
                ring[j] = v;
            }
        }
    }
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched; any other value = the
// cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `rows` and `conv_dim` must be non-zero, `kernel` in 2..=8, `rows · conv_dim` at most INT_MAX,
// and `slot_elems` (`GdnStateLayout::conv_elems_per_slot`) exactly `conv_dim · (kernel − 1)`; all
// are checked before any pointer is touched. `x` is `[rows, conv_dim]` bf16, `w` `[conv_dim,
// kernel]` f32, `conv_out` `[rows, conv_dim]` f32, `sequence` `[slot, q_len, start, 0]` u32 with
// `1 ≤ q_len ≤ rows`, and `conv_state` one layer's conv buffer, holding slot `slot`'s
// `slot_elems` f32 at `slot · slot_elems`. `conv_out` and `conv_state` overlap nothing else. No
// shared memory, so no one-time setup. The launch is asynchronous on `stream`.
extern "C" int scratchy_cuda_gdn_conv1d(const void* x, const void* w, void* conv_state,
                                        uint64_t slot_elems, const void* sequence, void* conv_out,
                                        uint32_t rows, uint32_t conv_dim, uint32_t kernel,
                                        void* stream /* cudaStream_t; null = default stream */) {
    if (rows == 0 || conv_dim == 0 || kernel < KMIN || kernel > KMAX ||
        uint64_t(rows) * uint64_t(conv_dim) > uint64_t(INT_MAX) ||
        slot_elems != uint64_t(conv_dim) * uint64_t(kernel - 1)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const __nv_bfloat16* xp = static_cast<const __nv_bfloat16*>(x);
    const float* wp = static_cast<const float*>(w);
    float* sp = static_cast<float*>(conv_state);
    unsigned long long se = slot_elems;
    const uint32_t* qp = static_cast<const uint32_t*>(sequence);
    float* op = static_cast<float*>(conv_out);
    int icd = static_cast<int>(conv_dim), ik = static_cast<int>(kernel);
    void* args[] = {&xp, &wp, &sp, &se, &qp, &op, &icd, &ik};
    const dim3 grid((conv_dim + THREADS - 1) / THREADS, (rows + TOKEN_BLOCK - 1) / TOKEN_BLOCK);
    const dim3 block(THREADS);
    return static_cast<int>(cudaLaunchKernel(gdn_conv1d_kernel, grid, block, args, 0,
                                             static_cast<cudaStream_t>(stream)));
}
