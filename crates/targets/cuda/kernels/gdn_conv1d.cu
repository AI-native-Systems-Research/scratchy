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
// ThunderKittens' warp vector (PLAN-tk100 §2 S4, §3.12): TK adds the container and the loads and
// stores here, no matrix multiply. One warp per (group of 128 channels, block of 64 tokens), 4
// warps per CTA: grid (ceil(conv_dim / 128 / 4), ceil(rows / 64)). The warp's channels
// `c0 .. c0 + 128` are a naive `rv_fl<128>` (lane `l` holds channels `c0 + l + 32i`). The kernel is
// instantiated for each `K` in 2..=8, so every window and tap vector is indexed by a compile-time
// constant. The warp keeps the causal window (`K − 1` vectors) in registers and walks its block's
// tokens in order:
//   - the taps `w_j[c] = w[c, j]` are strided by `K` in memory, so each is gathered with TK's
//     vector `apply` over `gl::operator[]`, once;
//   - the window is seeded in block 0 from the slot's ring (the same `apply` form, `[slot][c][j]`),
//     or with `warp::zero` WITHOUT READING THE RING on a fresh start: the pool is never cleared,
//     so a fresh slot may hold anything, NaN included, and a select keeps it out where
//     `0 · ring` would not; in any other block from `x` rows `t0 − K + 1 .. t0 − 1` by
//     `warp::load` (bf16 to f32, exact), real rows since `t0 ≥ 64 ≥ K − 1` and `t0 < q_len`;
//   - per token, `xt` by `warp::load`, then the taps in ascending `j`: `sum = w_0 · win_0`, then
//     `sum = fma(w_j, win_j, sum)`, the token's tap last, each through the op
//     `scratchy::ops::fma_acc` (`fmaf`, one rounding per tap, the plain-CUDA kernel's order and
//     rounding) by the `rv_fma` gap; then `unary_op<scratchy::ops::silu>` and `warp::store` (f32);
//     then the window shifted left by `warp::copy`, the token entering last.
// Only block 0's warp then writes the new ring, with the same `apply` form over a `gl<float>`
// (each lane writes its own channels' entries). So each channel's ring has one reader and one
// writer, the same lane, and no race (metal's argument, `gdn_conv1d_varlen.metal:8-25`). No shared
// memory, so no shared-tile hand-off needs a barrier.
//
// A block with `t0 ≥ q_len` returns before reading anything else, and no lane reads a row at or
// past `q_len`: padding rows are never read, so a NaN in one cannot reach an output. Each token's
// arithmetic is the same wherever the token sits in a chunk, so a chunked sequence gives the same
// bits as one forward of it. The ring holds copies of widened bf16 inputs (or the seed's), so it is
// bit-exact.
//
// Built with `--use_fast_math` (build.rs): `expf` and the division in SiLU are the approximate
// ones; the tests' bound (2⁻¹⁶·(|ref| + Σ|w·x|) + 2⁻²⁴) allows for both.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::GDN_CONV_CHANNEL_GROUP`: one warp's channels, a naive `rv_fl<128>`.
constexpr int GROUP = 128;
constexpr int WARPS = 4;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;
// `step::GDN_CONV_TOKEN_BLOCK`.
constexpr int TOKEN_BLOCK = 64;
// `step::GDN_CONV_KERNEL_RANGE`.
constexpr uint32_t KMIN = 2;
constexpr uint32_t KMAX = 8;

using chan_vec = rv_fl<GROUP>;
using x_gl = gl<bf16, 1, 1, -1, -1>;
using f_gl = gl<float, 1, 1, -1, -1>;
// `[slot, q_len, start, 0]`.
using seq_gl = gl<int, 1, 1, 1, -1>;

struct conv_globals {
    // `[rows, conv_dim]` bf16.
    x_gl x;
    // `[conv_dim, K]` f32.
    f_gl w;
    // `[slots, conv_dim · (K − 1)]` f32: slot s's ring, `[c][j]`.
    f_gl ring;
    seq_gl sequence;
    // `[rows, conv_dim]` f32.
    f_gl conv_out;
    int rows;
    int groups;
};

template <int K>
__global__ __launch_bounds__(NUM_THREADS) void gdn_conv1d_ker(const __grid_constant__ conv_globals g) {
    constexpr int STATE = K - 1;
    const int group = blockIdx.x * WARPS + kittens::warpid();
    if (group >= g.groups) {
        return;
    }
    const int q_len = g.sequence[{0, 0, 0, 1}];
    const int t0 = blockIdx.y * TOKEN_BLOCK;
    if (t0 >= q_len) {
        return;
    }
    const int slot = g.sequence[{0, 0, 0, 0}];
    const bool fresh = g.sequence[{0, 0, 0, 2}] == 1;
    const int c0 = group * GROUP;

    chan_vec taps[K];
#pragma unroll
    for (int j = 0; j < K; ++j) {
        warp::apply(taps[j], taps[j], [&](int i, float) { return g.w[{0, 0, c0 + i, j}]; });
    }

    // The window before row t0: the last `STATE` entries of `[seed ++ x]`. Block 0's seed is kept
    // for the ring write below.
    chan_vec seed[STATE];
    chan_vec win[STATE];
#pragma unroll
    for (int j = 0; j < STATE; ++j) {
        if (t0 == 0) {
            if (fresh) {
                // A fresh start keeps the 0: the ring is not read (a branch every thread of the
                // grid takes the same way, not `fresh · ring`).
                warp::zero(seed[j]);
            } else {
                warp::apply(seed[j], seed[j], [&](int i, float) {
                    return g.ring[{0, 0, slot, (c0 + i) * STATE + j}];
                });
            }
        } else {
            warp::load(seed[j], g.x, {0, 0, t0 - STATE + j, group});
        }
        warp::copy(win[j], seed[j]);
    }

    const int t_end = min(t0 + TOKEN_BLOCK, q_len);
    for (int t = t0; t < t_end; ++t) {
        chan_vec xt, sum;
        warp::load(xt, g.x, {0, 0, t, group});
        // The taps in ascending j: the window's `STATE`, then the token's own.
        warp::mul(sum, taps[0], win[0]);
#pragma unroll
        for (int j = 1; j < STATE; ++j) {
            scratchy::ops::rv_fma<scratchy::ops::fma_acc>(sum, taps[j], win[j], sum);
        }
        scratchy::ops::rv_fma<scratchy::ops::fma_acc>(sum, taps[STATE], xt, sum);
        warp::unary_op<scratchy::ops::silu>(sum, sum);
        warp::store(g.conv_out, sum, {0, 0, t, group});
        // Shift left, the token entering last.
#pragma unroll
        for (int j = 0; j + 1 < STATE; ++j) {
            warp::copy(win[j], win[j + 1]);
        }
        warp::copy(win[STATE - 1], xt);
    }

    if (t0 == 0) {
        // The new ring: entry j is `[seed ++ x[0 .. q_len)]`'s `q_len + j`-th, a seed entry when
        // `q_len + j < STATE`, else row `q_len + j − STATE` of x, a real row.
#pragma unroll
        for (int j = 0; j < STATE; ++j) {
            const int p = q_len + j - STATE;
            chan_vec entry;
            if (p < 0) {
#pragma unroll
                for (int i = 0; i < STATE; ++i) {
                    if (i == q_len + j) {
                        warp::copy(entry, seed[i]);
                    }
                }
            } else {
                warp::load(entry, g.x, {0, 0, p, group});
            }
            warp::apply(entry, entry, [&](int i, float v) {
                g.ring[{0, 0, slot, (c0 + i) * STATE + j}] = v;
                return v;
            });
        }
    }
}

template <int K>
int launch(const conv_globals& G, uint32_t rows, cudaStream_t stream) {
    void* args[] = {const_cast<conv_globals*>(&G)};
    const dim3 grid((uint32_t(G.groups) + uint32_t(WARPS) - 1) / uint32_t(WARPS),
                    (rows + uint32_t(TOKEN_BLOCK) - 1) / uint32_t(TOKEN_BLOCK));
    return static_cast<int>(
        cudaLaunchKernel(gdn_conv1d_ker<K>, grid, dim3(NUM_THREADS), args, 0, stream));
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught; any other value = the cudaError_t
// cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the runtime's last-error
// state.
//
// `rows` and `conv_dim` must be non-zero, `conv_dim` a multiple of 128
// (`step::GDN_CONV_CHANNEL_GROUP`, one warp's channels), `kernel` in 2..=8, `rows · conv_dim` at
// most INT_MAX, and `slot_elems` (`GdnStateLayout::conv_elems_per_slot`) exactly
// `conv_dim · (kernel − 1)`; all are checked before any pointer is touched. `x` is `[rows,
// conv_dim]` bf16, `w` `[conv_dim, kernel]` f32, `conv_out` `[rows, conv_dim]` f32, `sequence`
// `[slot, q_len, start, 0]` u32 with `1 ≤ q_len ≤ rows`, and `conv_state` one layer's conv buffer,
// holding slot `slot`'s `slot_elems` f32 at `slot · slot_elems`. `conv_out` and `conv_state`
// overlap nothing else. No shared memory, so no one-time setup. The launch is asynchronous on
// `stream`.
extern "C" int scratchy_cuda_gdn_conv1d(const void* x, const void* w, void* conv_state,
                                        uint64_t slot_elems, const void* sequence, void* conv_out,
                                        uint32_t rows, uint32_t conv_dim, uint32_t kernel,
                                        void* stream /* cudaStream_t; null = default stream */) {
    if (rows == 0 || conv_dim == 0 || kernel < KMIN || kernel > KMAX ||
        uint64_t(rows) * uint64_t(conv_dim) > uint64_t(INT_MAX) ||
        slot_elems != uint64_t(conv_dim) * uint64_t(kernel - 1) ||
        conv_dim % uint32_t(GROUP) != 0) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        // No view carries a TMA type, so building them encodes nothing: they are the pointers and
        // the dimensions the kernel indexes with. The ring is viewed one slot per row, as wide as
        // a slot: `slot · slot_elems` is its row `slot`, and the slot count is the most a slot
        // index can name.
        x_gl xg{static_cast<bf16*>(const_cast<void*>(x)), nullptr, nullptr, size_t(rows),
                size_t(conv_dim)};
        f_gl wg{static_cast<float*>(const_cast<void*>(w)), nullptr, nullptr, size_t(conv_dim),
                size_t(kernel)};
        f_gl rg{static_cast<float*>(conv_state), nullptr, nullptr, size_t(INT_MAX),
                size_t(slot_elems)};
        seq_gl qg{static_cast<int*>(const_cast<void*>(sequence)), nullptr, nullptr, nullptr,
                  size_t(4)};
        f_gl og{static_cast<float*>(conv_out), nullptr, nullptr, size_t(rows), size_t(conv_dim)};
        conv_globals G{xg, wg, rg, qg, og, int(rows), int(conv_dim / uint32_t(GROUP))};
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        switch (kernel) {
            case 2: return launch<2>(G, rows, s);
            case 3: return launch<3>(G, rows, s);
            case 4: return launch<4>(G, rows, s);
            case 5: return launch<5>(G, rows, s);
            case 6: return launch<6>(G, rows, s);
            case 7: return launch<7>(G, rows, s);
            default: return launch<8>(G, rows, s);
        }
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
