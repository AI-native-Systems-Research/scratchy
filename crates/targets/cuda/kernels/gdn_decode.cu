// SPDX-License-Identifier: Apache-2.0
//
// Gated-DeltaNet one-token decode for the H100 (#1049): the four launches of a one-row
// `GatedDeltaNet` step (`gdn_conv1d.cu`, `gdn_gating.cu`, `gdn_scan.cu`, `gdn_rms_norm_gated.cu`)
// as one, `src/step.rs`'s `CudaStep::GdnDecode`. Each stage is its own kernel's code at one token,
// on the same f32 values in the same order, so the output, the conv ring and the state are the
// four launches' bits; only the f32 intermediates stay in shared memory instead of the step's GDN
// scratch.
//
// ONE CTA PER KEY HEAD, 8 warps, grid `nk`. A key head's `nv / nk` value heads (`heads`, 1 for the
// 0.8B, 2 for the 9B and the 35B) read its q and k, so the CTA that owns the key head also owns its
// value heads: every channel's conv ring and every state tile has one reader and one writer, in one
// CTA. A CTA per value head would have two CTAs read and rewrite one key head's ring, a race no
// single-writer rule closes (one may read the ring after the other rewrote it).
//
//   1. The conv (`gdn_conv1d.cu`'s body at token 0 of block 0): warp 0 runs the key head's 128 q
//      channels, warp 1 its k channels, warp `2 + i` the `i`-th 128 channels of the CTA's value
//      heads' v. Each warp's taps and seed (the slot's ring, or `warp::zero` without reading it on
//      a fresh start) by TK's `apply` over `gl::operator[]`, the token by `warp::load`, then
//      `warp::mul` and the `rv_fma` gap's `fma_acc` taps in ascending order and
//      `unary_op<silu>`; the result `warp::store`d (f32) into the CTA's shared q, k or v vector. The
//      new ring is the seed shifted by one, the token last, written by the same `apply` form.
//   2. The gating (`gdn_gating.cu`'s body): warp 0, after its conv, loads the 16-head chunk holding
//      the CTA's value heads (`16 % heads == 0`, so they share one chunk) and computes `g` and `β`
//      for the chunk, `warp::store`d (f32) into shared vectors.
//   3. The scan (`gdn_scan.cu`'s per-token body): every warp normalises q and k (each warp the
//      same bits), and for each value head its 16-row state tiles `w, w + 8, …`: the tile loaded
//      (or zeroed on a fresh start), decayed, corrected and updated, its `o` rows `warp::store`d
//      (f32) into the head's shared `o` vector, the tile stored back.
//   4. The gated norm (`gdn_rms_norm_gated.cu`'s body): warp `j` takes value head `j`, its `o` a
//      naive `rv_fl<HV>` loaded from shared memory (the same lane-to-element map as the global
//      load), then the norm, `SiLU(z)` and the gain, stored bf16.
// `__syncthreads` separates the stages. The token is row 0: the launcher's contract is the four
// launchers' `1 ≤ q_len ≤ rows` at `rows = 1`, so `q_len` is 1. The sequence record (slot, q_len,
// fresh) is read on the device from the tape's GDN sequence region, never a host argument.
//
// ThunderKittens' operations only (PLAN-tk100 §1), each the one its stage's kernel uses: the
// `rv`/`sv` loads and stores, `warp::apply` with `gl::operator[]`, the `rv_fma` gap, `warp::mul`,
// `add`, `sub`, `sum`, `copy`, `mul_col`, `row_sum`, `broadcast_col`, `row_map<fma_AxCtB>`, and
// the op structs `silu`, `gdn_g`, `sigmoid`, `l2_inv`, `exp_of`, `rms_inv`. Instantiated for each
// conv width `K` in 2..=8 and `HV` (`head_v`) 128 or 256: every window, tap and tile index is a
// compile-time constant after unrolling. Static shared memory only (under 10 KiB), so the one-time
// setup only loads the kernel.
//
// Built with `--use_fast_math` (build.rs), as the four kernels it replaces.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cmath>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::GDN_HEAD_K` and `step::GDN_CONV_CHANNEL_GROUP`: one warp's channels.
constexpr int HEAD_K = 128;
constexpr int GROUP = 128;
constexpr int WARPS = 8;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;
constexpr int WARP_ROWS = 16;
// `step::GDN_GATING_NV_MULTIPLE` and `step::GDN_DECODE_MAX_HEADS`.
constexpr int NV_CHUNK = 16;
constexpr int MAX_HEADS = 4;
// `step::GDN_CONV_KERNEL_RANGE`.
constexpr uint32_t KMIN = 2;
constexpr uint32_t KMAX = 8;
// The traced phases (`TRACE`, test support): thread 0's `clock` cycles from its start at the end
// of each. The one-key-head form: 0 the conv, 1 the gating, 2 the barrier, 3 the scan, 4 the
// barrier, 5 the norm. The one-value-head form: 0 the prefetch issued, 1 the conv and gating, 2
// the cluster barrier, 3 the q/k ring written, 4 the scan, 5 the barrier, 6 the norm.
constexpr int PHASES = 8;
// The conv width the traced instances take (every model's `conv_kernel`).
constexpr uint32_t TRACE_K = 4;

using chan_vec = rv_fl<GROUP>;
using chunk_rv = rv_fl<NV_CHUNK>;
using state_tile = rt_fl<WARP_ROWS, HEAD_K>;
// One value per state column (the `align` layout): k̂ and q̂.
using key_vec = state_tile::row_vec;
// One value per state row (the `ortho` layout): sk, v, u and o.
using row_vals = state_tile::col_vec;
using key_naive = rv_fl<HEAD_K>;
using row_naive = rv_fl<WARP_ROWS>;

using x_gl = gl<bf16, 1, 1, -1, -1>;
using f_gl = gl<float, 1, 1, -1, -1>;
using vec_gl = gl<float, 1, 1, 1, -1>;
// `[slots, nv, head_v, 128]`, as `gdn_scan.cu`'s: read and written by `warp::load`/`store` only,
// which never read the slot dimension, so its size is 1.
using state_gl = gl<float, -1, -1, -1, HEAD_K>;
// `[slot, q_len, start, 0]`.
using seq_gl = gl<int, 1, 1, 1, -1>;

struct decode_globals {
    // `[1, conv_dim]`: `q | k | v`.
    x_gl qkv;
    // `[1, nv]` each.
    x_gl a;
    x_gl b;
    // `[nv, head_v]`: the token's gate, one row per value head.
    x_gl z;
    // `[conv_dim, K]` f32: the conv taps.
    f_gl w;
    // `[nv]` each.
    vec_gl a_log;
    vec_gl dt_bias;
    // `[head_v]`: the gated norm's gain.
    vec_gl norm;
    // `[slots, conv_dim · (K − 1)]` f32: slot s's ring, `[c][j]`.
    f_gl ring;
    state_gl ssm_state;
    seq_gl sequence;
    // `[nv, head_v]` bf16.
    x_gl out;
    // `[CTAs, PHASES]` f32 (`TRACE` only): thread 0's cycles from its start at each phase's end.
    f_gl trace;
    int nk;
    // The value heads per key head.
    int heads;
    float scale;
    float eps;
};

template <int K, int HV, bool TRACE = false>
__global__ __launch_bounds__(NUM_THREADS, 1) void gdn_decode_ker(
    const __grid_constant__ decode_globals g) {
    constexpr int STATE = K - 1;
    // A value head's 128-channel vectors and its 16-row state tiles.
    constexpr int V_VECS = HV / GROUP;
    constexpr int TILES = HV / WARP_ROWS;
    __shared__ sv_fl<HEAD_K> qs;
    __shared__ sv_fl<HEAD_K> ks;
    __shared__ sv_fl<HV> vs[MAX_HEADS];
    __shared__ sv_fl<HV> os[MAX_HEADS];
    __shared__ sv_fl<NV_CHUNK> gs;
    __shared__ sv_fl<NV_CHUNK> bs;

    const int kh = blockIdx.x;
    const int w = kittens::warpid();
    const int h0 = kh * g.heads;
    const int key_dim = g.nk * HEAD_K;
    const int slot = g.sequence[{0, 0, 0, 0}];
    const bool fresh = g.sequence[{0, 0, 0, 2}] == 1;
    // `TRACE`: thread 0's cycles since its start at the end of each phase.
    const uint32_t t0 = TRACE ? uint32_t(clock()) : 0;
    float marks[PHASES] = {};
    const auto mark = [&](int p) {
        if constexpr (TRACE) {
            marks[p] = float(uint32_t(clock()) - t0);
        }
    };

    // 1. The conv's one step, and the ring after it.
    if (w < 2 + g.heads * V_VECS) {
        const int group = w == 0   ? kh
                          : w == 1 ? g.nk + kh
                                   : (2 * key_dim + h0 * HV) / GROUP + (w - 2);
        const int c0 = group * GROUP;
        chan_vec taps[K];
#pragma unroll
        for (int j = 0; j < K; ++j) {
            warp::apply(taps[j], taps[j], [&](int i, float) { return g.w[{0, 0, c0 + i, j}]; });
        }
        chan_vec seed[STATE];
#pragma unroll
        for (int j = 0; j < STATE; ++j) {
            if (fresh) {
                // A fresh start keeps the 0: the ring is not read.
                warp::zero(seed[j]);
            } else {
                warp::apply(seed[j], seed[j], [&](int i, float) {
                    return g.ring[{0, 0, slot, (c0 + i) * STATE + j}];
                });
            }
        }
        chan_vec xt, sum;
        warp::load(xt, g.qkv, {0, 0, 0, group});
        warp::mul(sum, taps[0], seed[0]);
#pragma unroll
        for (int j = 1; j < STATE; ++j) {
            scratchy::ops::rv_fma<scratchy::ops::fma_acc>(sum, taps[j], seed[j], sum);
        }
        scratchy::ops::rv_fma<scratchy::ops::fma_acc>(sum, taps[STATE], xt, sum);
        warp::unary_op<scratchy::ops::silu>(sum, sum);
        if (w == 0) {
            warp::store(qs, sum);
        } else if (w == 1) {
            warp::store(ks, sum);
        } else {
            const int v = w - 2;
            warp::store(vs[v / V_VECS].template subvec<GROUP>(v % V_VECS), sum);
        }
        // The new ring: the seed shifted by one, the token last.
#pragma unroll
        for (int j = 0; j + 1 < STATE; ++j) {
            chan_vec entry;
            warp::copy(entry, seed[j + 1]);
            warp::apply(entry, entry, [&](int i, float v) {
                g.ring[{0, 0, slot, (c0 + i) * STATE + j}] = v;
                return v;
            });
        }
        warp::apply(xt, xt, [&](int i, float v) {
            g.ring[{0, 0, slot, (c0 + i) * STATE + STATE - 1}] = v;
            return v;
        });
    }

    mark(0);
    // 2. The gating of the 16-head chunk holding the CTA's value heads.
    if (w == 0) {
        const int c = h0 / NV_CHUNK;
        chunk_rv av, bv, a_log, dt_bias;
        warp::load(av, g.a, {0, 0, 0, c});
        warp::load(bv, g.b, {0, 0, 0, c});
        warp::load(a_log, g.a_log, {0, 0, 0, c});
        warp::load(dt_bias, g.dt_bias, {0, 0, 0, c});
        warp::add(av, av, dt_bias);
        warp::bin_op<scratchy::ops::gdn_g>(av, av, a_log);
        warp::unary_op<scratchy::ops::sigmoid>(bv, bv);
        warp::store(gs, av);
        warp::store(bs, bv);
    }
    mark(1);
    __syncthreads();
    mark(2);

    // 3. The scan's token: q̂ and k̂, then each value head's tiles.
    key_naive qn, kn, sq;
    warp::load(qn, qs);
    warp::load(kn, ks);
    warp::mul(sq, kn, kn);
    const float kk = warp::sum(sq);
    warp::mul(sq, qn, qn);
    const float qq = warp::sum(sq);
    const float k_inv = scratchy::ops::l2_inv::op<float>(kk);
    const float q_inv = scratchy::ops::l2_inv::op<float>(qq) * g.scale;
    warp::mul(kn, kn, k_inv);
    warp::mul(qn, qn, q_inv);
    for (int hg = 0; hg < g.heads; ++hg) {
        const int h = h0 + hg;
        const float decay = scratchy::ops::exp_of::op<float>(gs[h % NV_CHUNK]);
        const float beta = bs[h % NV_CHUNK];
        for (int r = w; r < TILES; r += WARPS) {
            const coord<state_tile> at = {slot, h, r, 0};
            state_tile S, T;
            if (fresh) {
                // A select: the slot is not read.
                warp::zero(S);
            } else {
                warp::load(S, g.ssm_state, at);
            }
            warp::mul(S, S, decay);
            key_vec kv;
            warp::copy(kv, kn);
            warp::mul_col(T, S, kv);
            row_vals sk;
            warp::row_sum(sk, T);
            row_naive vn;
            warp::load(vn, vs[hg].template subvec<WARP_ROWS>(r));
            row_vals u;
            warp::copy(u, vn);
            warp::sub(u, u, sk);
            warp::mul(u, u, beta);
            warp::broadcast_col(T, kv);
            warp::row_map<base_ops::fma_AxCtB>(S, T, S, u);
            warp::copy(kv, qn);
            warp::mul_col(T, S, kv);
            row_vals ov;
            warp::row_sum(ov, T);
            warp::store(os[hg].template subvec<WARP_ROWS>(r), ov);
            warp::store(g.ssm_state, S, at);
        }
    }
    mark(3);
    __syncthreads();
    mark(4);

    // 4. The gated norm of value head `w`.
    if (w < g.heads) {
        const int h = h0 + w;
        rv_fl<HV> v, sqv, z, gain;
        warp::load(v, os[w]);
        warp::mul(sqv, v, v);
        const float m = warp::sum(sqv) / float(HV);
        const float inv = scratchy::ops::rms_inv::op<float>(m, g.eps);
        warp::load(z, g.z, {0, 0, h, 0});
        warp::unary_op<scratchy::ops::silu>(z, z);
        warp::load(gain, g.norm, {0, 0, 0, 0});
        warp::mul(v, v, inv);
        warp::mul(v, v, gain);
        warp::mul(v, v, z);
        warp::store(g.out, v, {0, 0, h, 0});
    }
    mark(5);
    if constexpr (TRACE) {
        if (threadIdx.x == 0) {
#pragma unroll
            for (int p = 0; p < PHASES; ++p) {
                g.trace[{0, 0, int(blockIdx.x), p}] = marks[p];
            }
        }
    }
}

// THE ONE-VALUE-HEAD FORM (FP8 S8 D1, the launcher's): `gdn_decode_ker`'s stages and ops in the same
// order on the same values, so the same bits, over one CTA per value head instead of one per key
// head (the 35B: 32 CTAs, not 16; the 27B: 48), the key head's `heads` CTAs one cluster:
//   - every load issued first: each warp's first state tile, the norm's `z` and gain;
//   - the conv: warps 0 and 1 the key head's q and k channels (each CTA of the cluster computes
//     them, from the ring as it was), warps `2 ..` the CTA's own value head's v channels, whose
//     ring the CTA alone owns and rewrites at once; the gating on the last warp, in parallel;
//   - the cluster's barrier (`everyone::tma::cluster::sync`): every CTA has read the key head's q
//     and k ring, so rank 0 alone rewrites it after (one reader-writer race the one-key-head form
//     avoided by owning the key head in one CTA);
//   - the scan of the one value head, a tile a warp (two at `head_v` 256), the first prefetched;
//     then the gated norm by warp 0 on the prefetched `z` and gain.
template <int K, int HV, bool TRACE = false>
__global__ __launch_bounds__(NUM_THREADS, 1) void gdn_decode_head_ker(
    const __grid_constant__ decode_globals g) {
    constexpr int STATE = K - 1;
    constexpr int V_VECS = HV / GROUP;
    constexpr int TILES = HV / WARP_ROWS;
    constexpr int GATE_WARP = WARPS - 1;
    static_assert(2 + V_VECS <= GATE_WARP, "the gating's warp is not a conv warp");
    __shared__ sv_fl<HEAD_K> qs;
    __shared__ sv_fl<HEAD_K> ks;
    __shared__ sv_fl<HV> vs;
    __shared__ sv_fl<HV> os;
    __shared__ sv_fl<NV_CHUNK> gs;
    __shared__ sv_fl<NV_CHUNK> bs;

    const int hg = kittens::cluster_ctarank();
    const int kh = blockIdx.y;
    const int h = kh * g.heads + hg;
    const int w = kittens::warpid();
    const int key_dim = g.nk * HEAD_K;
    const int slot = g.sequence[{0, 0, 0, 0}];
    const bool fresh = g.sequence[{0, 0, 0, 2}] == 1;
    const uint32_t t0 = TRACE ? uint32_t(clock()) : 0;
    float marks[PHASES] = {};
    const auto mark = [&](int p) {
        if constexpr (TRACE) {
            marks[p] = float(uint32_t(clock()) - t0);
        }
    };

    // 0. Every load first: the warp's first state tile; warp 0's `z` and gain.
    state_tile S;
    if (w < TILES) {
        if (fresh) {
            // A select: the slot is not read.
            warp::zero(S);
        } else {
            warp::load(S, g.ssm_state, {slot, h, w, 0});
        }
    }
    rv_fl<HV> z, gain;
    if (w == 0) {
        warp::load(z, g.z, {0, 0, h, 0});
        warp::load(gain, g.norm, {0, 0, 0, 0});
    }
    mark(0);

    // 1. The conv's one step (`gdn_decode_ker`'s code), and the v channels' ring after it; the
    //    gating of the 16-head chunk on its own warp.
    chan_vec seed[STATE], xt;
    int c0 = 0;
    if (w < 2 + V_VECS) {
        const int group = w == 0   ? kh
                          : w == 1 ? g.nk + kh
                                   : (2 * key_dim + h * HV) / GROUP + (w - 2);
        c0 = group * GROUP;
        chan_vec taps[K];
#pragma unroll
        for (int j = 0; j < K; ++j) {
            warp::apply(taps[j], taps[j], [&](int i, float) { return g.w[{0, 0, c0 + i, j}]; });
        }
#pragma unroll
        for (int j = 0; j < STATE; ++j) {
            if (fresh) {
                // A fresh start keeps the 0: the ring is not read.
                warp::zero(seed[j]);
            } else {
                warp::apply(seed[j], seed[j], [&](int i, float) {
                    return g.ring[{0, 0, slot, (c0 + i) * STATE + j}];
                });
            }
        }
        chan_vec sum;
        warp::load(xt, g.qkv, {0, 0, 0, group});
        warp::mul(sum, taps[0], seed[0]);
#pragma unroll
        for (int j = 1; j < STATE; ++j) {
            scratchy::ops::rv_fma<scratchy::ops::fma_acc>(sum, taps[j], seed[j], sum);
        }
        scratchy::ops::rv_fma<scratchy::ops::fma_acc>(sum, taps[STATE], xt, sum);
        warp::unary_op<scratchy::ops::silu>(sum, sum);
        if (w == 0) {
            warp::store(qs, sum);
        } else if (w == 1) {
            warp::store(ks, sum);
        } else {
            warp::store(vs.template subvec<GROUP>(w - 2), sum);
        }
    }
    // The ring after the step: the seed shifted by one, the token last.
    const auto write_ring = [&]() {
#pragma unroll
        for (int j = 0; j + 1 < STATE; ++j) {
            chan_vec entry;
            warp::copy(entry, seed[j + 1]);
            warp::apply(entry, entry, [&](int i, float v) {
                g.ring[{0, 0, slot, (c0 + i) * STATE + j}] = v;
                return v;
            });
        }
        warp::apply(xt, xt, [&](int i, float v) {
            g.ring[{0, 0, slot, (c0 + i) * STATE + STATE - 1}] = v;
            return v;
        });
    };
    if (w >= 2 && w < 2 + V_VECS) {
        write_ring();
    }
    if (w == GATE_WARP) {
        const int c = h / NV_CHUNK;
        chunk_rv av, bv, a_log, dt_bias;
        warp::load(av, g.a, {0, 0, 0, c});
        warp::load(bv, g.b, {0, 0, 0, c});
        warp::load(a_log, g.a_log, {0, 0, 0, c});
        warp::load(dt_bias, g.dt_bias, {0, 0, 0, c});
        warp::add(av, av, dt_bias);
        warp::bin_op<scratchy::ops::gdn_g>(av, av, a_log);
        warp::unary_op<scratchy::ops::sigmoid>(bv, bv);
        warp::store(gs, av);
        warp::store(bs, bv);
    }
    mark(1);
    // Every CTA of the key head has read its q and k ring (and its shared vectors are written):
    // rank 0 alone rewrites that ring after.
    everyone::tma::cluster::sync();
    mark(2);
    if (hg == 0 && w < 2) {
        write_ring();
    }
    mark(3);

    // 3. The scan's token for value head `h`, `gdn_decode_ker`'s code; the warp's first tile is the
    //    prefetched one.
    key_naive qn, kn, sq;
    warp::load(qn, qs);
    warp::load(kn, ks);
    warp::mul(sq, kn, kn);
    const float kk = warp::sum(sq);
    warp::mul(sq, qn, qn);
    const float qq = warp::sum(sq);
    const float k_inv = scratchy::ops::l2_inv::op<float>(kk);
    const float q_inv = scratchy::ops::l2_inv::op<float>(qq) * g.scale;
    warp::mul(kn, kn, k_inv);
    warp::mul(qn, qn, q_inv);
    const float decay = scratchy::ops::exp_of::op<float>(gs[h % NV_CHUNK]);
    const float beta = bs[h % NV_CHUNK];
    for (int r = w; r < TILES; r += WARPS) {
        const coord<state_tile> at = {slot, h, r, 0};
        if (r != w) {
            if (fresh) {
                warp::zero(S);
            } else {
                warp::load(S, g.ssm_state, at);
            }
        }
        state_tile T;
        warp::mul(S, S, decay);
        key_vec kv;
        warp::copy(kv, kn);
        warp::mul_col(T, S, kv);
        row_vals sk;
        warp::row_sum(sk, T);
        row_naive vn;
        warp::load(vn, vs.template subvec<WARP_ROWS>(r));
        row_vals u;
        warp::copy(u, vn);
        warp::sub(u, u, sk);
        warp::mul(u, u, beta);
        warp::broadcast_col(T, kv);
        warp::row_map<base_ops::fma_AxCtB>(S, T, S, u);
        warp::copy(kv, qn);
        warp::mul_col(T, S, kv);
        row_vals ov;
        warp::row_sum(ov, T);
        warp::store(os.template subvec<WARP_ROWS>(r), ov);
        warp::store(g.ssm_state, S, at);
    }
    mark(4);
    __syncthreads();
    mark(5);

    // 4. The gated norm of value head `h`, `gdn_decode_ker`'s code on the prefetched `z` and gain.
    if (w == 0) {
        rv_fl<HV> v, sqv;
        warp::load(v, os);
        warp::mul(sqv, v, v);
        const float m = warp::sum(sqv) / float(HV);
        const float inv = scratchy::ops::rms_inv::op<float>(m, g.eps);
        warp::unary_op<scratchy::ops::silu>(z, z);
        warp::mul(v, v, inv);
        warp::mul(v, v, gain);
        warp::mul(v, v, z);
        warp::store(g.out, v, {0, 0, h, 0});
    }
    mark(6);
    if constexpr (TRACE) {
        if (threadIdx.x == 0) {
            const int cta = blockIdx.y * gridDim.x + blockIdx.x;
#pragma unroll
            for (int p = 0; p < PHASES; ++p) {
                g.trace[{0, 0, cta, p}] = marks[p];
            }
        }
    }
}

// The one-key-head form (`variant` 0) on `nk` CTAs, or the one-value-head form (1) on `heads ×
// nk` CTAs, the key head's `heads` one cluster.
template <int K, int HV, bool TRACE>
int launch_decode(const decode_globals& G, uint32_t nk, uint32_t heads, uint32_t variant,
                  cudaStream_t stream) {
    if (variant == 0) {
        void* args[] = {const_cast<decode_globals*>(&G)};
        return static_cast<int>(cudaLaunchKernel(gdn_decode_ker<K, HV, TRACE>, dim3(nk),
                                                 dim3(NUM_THREADS), args, 0, stream));
    }
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = heads;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(heads, nk);
    cfg.blockDim = dim3(NUM_THREADS);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return static_cast<int>(cudaLaunchKernelEx(&cfg, gdn_decode_head_ker<K, HV, TRACE>, G));
}

template <int HV>
int launch_width(const decode_globals& G, uint32_t nk, uint32_t heads, uint32_t kernel,
                 uint32_t variant, bool trace, cudaStream_t stream) {
    if (trace) {
        return launch_decode<int(TRACE_K), HV, true>(G, nk, heads, variant, stream);
    }
    switch (kernel) {
        case 2: return launch_decode<2, HV, false>(G, nk, heads, variant, stream);
        case 3: return launch_decode<3, HV, false>(G, nk, heads, variant, stream);
        case 4: return launch_decode<4, HV, false>(G, nk, heads, variant, stream);
        case 5: return launch_decode<5, HV, false>(G, nk, heads, variant, stream);
        case 6: return launch_decode<6, HV, false>(G, nk, heads, variant, stream);
        case 7: return launch_decode<7, HV, false>(G, nk, heads, variant, stream);
        default: return launch_decode<8, HV, false>(G, nk, heads, variant, stream);
    }
}

// `cudaFuncGetAttributes` of every instance at `HV` (both forms, every conv width, and the traced
// ones): the first failure, else cudaSuccess.
template <int HV>
cudaError_t load_width() {
    cudaFuncAttributes attr;
    for (auto ker : {gdn_decode_ker<2, HV>, gdn_decode_ker<3, HV>, gdn_decode_ker<4, HV>,
                     gdn_decode_ker<5, HV>, gdn_decode_ker<6, HV>, gdn_decode_ker<7, HV>,
                     gdn_decode_ker<8, HV>, gdn_decode_ker<int(TRACE_K), HV, true>,
                     gdn_decode_head_ker<2, HV>, gdn_decode_head_ker<3, HV>,
                     gdn_decode_head_ker<4, HV>, gdn_decode_head_ker<5, HV>,
                     gdn_decode_head_ker<6, HV>, gdn_decode_head_ker<7, HV>,
                     gdn_decode_head_ker<8, HV>, gdn_decode_head_ker<int(TRACE_K), HV, true>}) {
        if (const cudaError_t e = cudaFuncGetAttributes(&attr, ker); e != cudaSuccess) {
            return e;
        }
    }
    return cudaSuccess;
}

}  // namespace

// The kernel's one-time setup, run once per device by `kernels::prepare_all` (`CudaDevice::open`):
// query every instance's attributes, which loads its module, so a worker's graph capture
// (`CudaWorker::new`) never loads a kernel mid-capture. It changes nothing on the device.
//
// 0 = done; any other value = the cudaError_t of the first query that failed.
extern "C" int scratchy_cuda_gdn_decode_prepare(void) {
    if (const cudaError_t e = load_width<128>(); e != cudaSuccess) {
        return static_cast<int>(e);
    }
    return static_cast<int>(load_width<256>());
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught; any other value = the cudaError_t
// cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the runtime's last-error
// state.
//
// The four GDN launchers' rules at one row, and the kernel's own: `head_k` 128, `head_v` 128 or
// 256, `nk` and `nv` non-zero, `nv` a multiple of `nk` and of 16, `heads = nv / nk` at most 4 and
// dividing 16, `2 + heads · head_v / 128` at most 8 (the conv's warps), `kernel` in 2..=8,
// `conv_slot_elems` exactly `conv_dim · (kernel − 1)` and `ssm_slot_elems` exactly
// `nv · head_v · 128` (`conv_dim = 2 · nk · 128 + nv · head_v`), `conv_dim · kernel` at most
// INT_MAX, and `scale` and `eps` finite and above 0; all are checked before any pointer is
// touched. `qkv` is `[1, conv_dim]` bf16, `a` and `b` `[1, nv]` bf16, `z` and `out` `[1, nv ·
// head_v]` bf16, `w` `[conv_dim, kernel]` f32, `a_log` and `dt_bias` `[nv]` f32, `norm` `[head_v]`
// f32, `sequence` `[slot, q_len, start, 0]` u32 with `q_len` 1, and `conv_state` and `ssm_state`
// one layer's buffers holding slot `slot`'s entries at `slot · slot_elems`. `out` and the two
// state buffers overlap nothing else. The output, the ring and the state are the bits of
// `scratchy_cuda_gdn_conv1d`, `_gating`, `_scan` and `_rms_norm_gated` at `rows = 1`.
// `scratchy_cuda_gdn_decode_prepare` must have succeeded on the current device. The launch is
// asynchronous on `stream`.
static int gdn_decode(const void* qkv, const void* a, const void* b, const void* z, const void* w,
                      const void* a_log, const void* dt_bias, const void* norm, void* conv_state,
                      uint64_t conv_slot_elems, void* ssm_state, uint64_t ssm_slot_elems,
                      const void* sequence, void* out, uint32_t nk, uint32_t nv, uint32_t head_k,
                      uint32_t head_v, uint32_t kernel, float scale, float eps, uint32_t variant,
                      void* trace, void* stream) {
    if (head_k != uint32_t(HEAD_K) || (head_v != 128 && head_v != 256) || nk == 0 || nv == 0 ||
        nv % nk != 0 || nv % uint32_t(NV_CHUNK) != 0 || kernel < KMIN || kernel > KMAX ||
        !std::isfinite(scale) || !(scale > 0.0f) || !std::isfinite(eps) || !(eps > 0.0f)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const uint32_t heads = nv / nk;
    const uint64_t conv_dim = 2 * uint64_t(nk) * HEAD_K + uint64_t(nv) * head_v;
    if (heads > uint32_t(MAX_HEADS) || uint32_t(NV_CHUNK) % heads != 0 ||
        2 + heads * head_v / uint32_t(GROUP) > uint32_t(WARPS) ||
        conv_dim * kernel > uint64_t(INT_MAX) ||
        conv_slot_elems != conv_dim * uint64_t(kernel - 1) ||
        ssm_slot_elems != uint64_t(nv) * head_v * HEAD_K) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const auto bf = [](const void* p) { return static_cast<bf16*>(const_cast<void*>(p)); };
        const auto fl = [](const void* p) { return static_cast<float*>(const_cast<void*>(p)); };
        // No view carries a TMA type, so building them encodes nothing: they are the pointers and
        // the dimensions the kernel indexes with. The ring is viewed one slot per row, as wide as
        // a slot, as `gdn_conv1d.cu` views it.
        x_gl qg{bf(qkv), nullptr, nullptr, size_t(1), size_t(conv_dim)};
        x_gl ag{bf(a), nullptr, nullptr, size_t(1), size_t(nv)};
        x_gl bg{bf(b), nullptr, nullptr, size_t(1), size_t(nv)};
        x_gl zg{bf(z), nullptr, nullptr, size_t(nv), size_t(head_v)};
        f_gl wg{fl(w), nullptr, nullptr, size_t(conv_dim), size_t(kernel)};
        vec_gl lg{fl(a_log), nullptr, nullptr, nullptr, size_t(nv)};
        vec_gl dg{fl(dt_bias), nullptr, nullptr, nullptr, size_t(nv)};
        vec_gl ng{fl(norm), nullptr, nullptr, nullptr, size_t(head_v)};
        f_gl rg{fl(conv_state), nullptr, nullptr, size_t(INT_MAX), size_t(conv_slot_elems)};
        state_gl sg{fl(ssm_state), size_t(1), size_t(nv), size_t(head_v), nullptr};
        seq_gl seqg{static_cast<int*>(const_cast<void*>(sequence)), nullptr, nullptr, nullptr,
                    size_t(4)};
        x_gl og{bf(out), nullptr, nullptr, size_t(nv), size_t(head_v)};
        // With no trace the view is never written; `out` is a valid address for it.
        const size_t ctas = variant == 0 ? size_t(nk) : size_t(nv);
        f_gl tg{fl(trace != nullptr ? trace : out), nullptr, nullptr,
                trace != nullptr ? ctas : size_t(1), trace != nullptr ? size_t(PHASES) : 1};
        decode_globals G{qg, ag, bg, zg, wg, lg, dg, ng, rg, sg, seqg, og, tg,
                         int(nk), int(heads), scale, eps};
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        const bool traced = trace != nullptr;
        return head_v == 128 ? launch_width<128>(G, nk, heads, kernel, variant, traced, s)
                             : launch_width<256>(G, nk, heads, kernel, variant, traced, s);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

extern "C" int scratchy_cuda_gdn_decode(const void* qkv, const void* a, const void* b,
                                        const void* z, const void* w, const void* a_log,
                                        const void* dt_bias, const void* norm, void* conv_state,
                                        uint64_t conv_slot_elems, void* ssm_state,
                                        uint64_t ssm_slot_elems, const void* sequence, void* out,
                                        uint32_t nk, uint32_t nv, uint32_t head_k,
                                        uint32_t head_v, uint32_t kernel, float scale, float eps,
                                        void* stream /* cudaStream_t; null = default stream */) {
    return gdn_decode(qkv, a, b, z, w, a_log, dt_bias, norm, conv_state, conv_slot_elems,
                      ssm_state, ssm_slot_elems, sequence, out, nk, nv, head_k, head_v, kernel,
                      scale, eps, 1, nullptr, stream);
}

// Test support for `tests/gdn.rs` only (FP8 S8 D1): `scratchy_cuda_gdn_decode` at `variant` 0 (the
// one-key-head form, #1049's) or 1 (the one-value-head form, the launcher's), the same bits; with
// `trace` (`[CTAs, 8]` f32, the CTAs `nk` at variant 0 and `nv` at 1, CTA `y · heads + x`), the
// traced instance's phase cycles (kernel 4 only). Refused as `scratchy_cuda_gdn_decode`, for
// `variant` above 1, and for a `trace` at another `kernel`.
extern "C" int scratchy_cuda_gdn_decode_variant(
    const void* qkv, const void* a, const void* b, const void* z, const void* w,
    const void* a_log, const void* dt_bias, const void* norm, void* conv_state,
    uint64_t conv_slot_elems, void* ssm_state, uint64_t ssm_slot_elems, const void* sequence,
    void* out, uint32_t nk, uint32_t nv, uint32_t head_k, uint32_t head_v, uint32_t kernel,
    float scale, float eps, uint32_t variant, void* trace,
    void* stream /* cudaStream_t; null = default stream */) {
    if (variant > 1 || (trace != nullptr && kernel != TRACE_K)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    return gdn_decode(qkv, a, b, z, w, a_log, dt_bias, norm, conv_state, conv_slot_elems,
                      ssm_state, ssm_slot_elems, sequence, out, nk, nv, head_k, head_v, kernel,
                      scale, eps, variant, trace, stream);
}
