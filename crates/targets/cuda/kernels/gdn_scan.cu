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
// THE STATE AS A THUNDERKITTENS REGISTER TILE (PLAN-tk100 §2 S6, §3.14): grid (head_v / 32, nv), as
// before; two warps per CTA, warp `w` holding value rows `row0 + 16w .. row0 + 16w + 16` of head h
// as one `rt_fl<16, 128>` (64 f32 per lane). Per token, every step is a TK op:
//   - q and k are loaded from shared memory as naive-layout `rv_fl<128>` (4 f32 per lane, each at a
//     fixed register: `warp::load` from an `sv` indexes the naive layout by unrolled constants);
//     their squared norms are `warp::mul` then `warp::sum`, the inverse norms the op structs
//     `scratchy::ops::l2_inv` (`rsqrtf(x + 1e-6)`) times `scale` for q, and each vector is scaled
//     by `warp::mul`. Each is then converted to the tile's `row_vec` (the `align` layout) by
//     `warp::copy`, TK's shuffle-only layout conversion. An `align` vector loaded straight from
//     shared memory is written at a lane-dependent register index, which ptxas places in a stack
//     frame (TK-2's RmsNorm, #850's review §11.1); the naive load and the conversion index every
//     register by a constant.
//   - the decay: `scratchy::ops::exp_of` (`expf`) of g, then `warp::mul(S, S, decay)`;
//   - `sk = S · k̂`: `warp::mul_col(T, S, k̂)`, then `warp::row_sum(sk, T)` into the tile's
//     `col_vec`;
//   - `u = (v − sk) · β` on the `col_vec` (`warp::sub`, `warp::mul`), `v` loaded as a naive
//     `rv_fl<16>` from the warp's 16 staged entries and converted by `warp::copy`;
//   - the rank-1 update: `warp::broadcast_col(T, k̂)`, then `warp::row_map<base_ops::fma_AxCtB>(S,
//     T, S, u)`, which is `T · u + S` with `u` per row (R8, checked by
//     `tests/gdn.rs`'s `the_rank_one_update_is_t_times_u_plus_s` through the probe kernel below);
//   - `o = S · q̂`: `warp::mul_col(T, S, q̂)`, `warp::row_sum(ov, T)`, and `warp::store` of the 16
//     values to `o[t, h·head_v + row0 + 16w ..]`.
// One temporary tile `T` serves both `mul_col`s and the `broadcast_col`, in sequence.
//
// THE ARITHMETIC ORDER, a new fixed one: `mul_col` rounds each product and `row_sum` adds the
// tile's products in tile order, then a fixed shuffle across the four lanes of a row; the norms are
// each lane's 4 products then a fixed shuffle tree. The plain-CUDA kernel summed fused `fmaf`
// products per lane, then a butterfly. The decay is a separate multiply in both, and the update
// `S + k̂ · u` fused in both (`fma_AxCtB`'s `a*c+b`, contracted under `--use_fast_math`).
//
// STAGING: the CTA stages blocks of 32 tokens in shared memory, `q` and `k` of its key head (512
// bytes each per token, `sv_fl<128>`), its 32 rows' `v` (128 bytes, `sv_fl<32>`), by TK's
// descriptor-free bulk copy (`tma::load_async`, one `tma::expect_bytes` per block on one
// semaphore), and `g` and `β` (one f32 each, read through `gl::operator[]` into an `sv_fl<32>`):
// 37,120 bytes, under the 48 KiB of static shared memory a launch gets with no attribute. Only real
// tokens are staged: no thread reads a padding row of any input.
//
// THE STEP-5B LESSON: the state pool is never cleared, so a fresh slot may hold NaN. A fresh start
// sets the tile to 0 (`warp::zero`, a select on a value every thread reads the same way) and never
// loads the slot: `0 · NaN` is NaN, so no masking multiply would do. A continued start loads the
// slot (`warp::load` from the f32 `gl`, a plain copy, so the state round-trips bit for bit).
//
// CONTINUITY: each token's arithmetic depends only on the state before it and its own inputs, not
// on where it sits in a chunk or a staging block, so one forward of a sequence and the same
// sequence in chunks give the same bits, the state included. No atomics: repeats are bit-identical.
//
// Built with `--use_fast_math` (build.rs): `expf` is `__expf` and `rsqrtf` the approximate one,
// subnormals flush to zero. The long-recurrence bound allows for both.

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

// `step::GDN_HEAD_K`.
constexpr int HEAD_K = 128;
// `step::GDN_HEAD_V_MULTIPLE`: the value rows of one CTA, 16 per warp.
constexpr int ROWS = 32;
constexpr int WARP_ROWS = 16;
constexpr int WARPS = ROWS / WARP_ROWS;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;
// `step::GDN_HEAD_V_MAX`.
constexpr uint32_t HEAD_V_MAX = 256;
// `step::GDN_SCAN_TOKEN_BLOCK`.
constexpr int TOKENS = 32;
// The bytes one token stages by bulk copy: q and k of the key head, v of the CTA's rows.
constexpr uint32_t QK_BYTES = HEAD_K * sizeof(float);
constexpr uint32_t V_BYTES = ROWS * sizeof(float);
constexpr uint32_t TOKEN_BYTES = 2 * QK_BYTES + V_BYTES;

using state_tile = rt_fl<WARP_ROWS, HEAD_K>;
// One value per state column (the `align` layout): k̂ and q̂.
using key_vec = state_tile::row_vec;
// One value per state row (the `ortho` layout): sk, v, u and o.
using row_vals = state_tile::col_vec;
// The naive layouts the shared-memory loads fill, before the conversion.
using key_naive = rv_fl<HEAD_K>;
using row_naive = rv_fl<WARP_ROWS>;

using row_gl = gl<float, 1, 1, -1, -1>;
// `[slots, nv, head_v, 128]`: slot, value head, state row, key lane. No TMA type: read and written
// by `warp::load`/`warp::store` only, which never read the slot dimension, so its size is 1.
using state_gl = gl<float, -1, -1, -1, HEAD_K>;
// `[slot, q_len, start, 0]`.
using seq_gl = gl<int, 1, 1, 1, -1>;

struct scan_globals {
    // `[rows, conv_dim]`: `q | k | v`.
    row_gl conv_out;
    // The gating's g, the log of the decay, and β: `[rows, nv]` each.
    row_gl log_decay;
    row_gl beta;
    // `[rows, nv · head_v]`.
    row_gl o;
    state_gl ssm_state;
    seq_gl sequence;
    int nk;
    int nv;
    int head_v;
    float scale;
};

static_assert(2 * TOKENS * sizeof(sv_fl<HEAD_K>) + TOKENS * sizeof(sv_fl<ROWS>) +
                      2 * sizeof(sv_fl<TOKENS>) ==
                  37120,
              "the staging is 37,120 bytes");

__global__ __launch_bounds__(NUM_THREADS) void gdn_scan_ker(const __grid_constant__ scan_globals g) {
    __shared__ sv_fl<HEAD_K> qs[TOKENS];
    __shared__ sv_fl<HEAD_K> ks[TOKENS];
    __shared__ sv_fl<ROWS> vs[TOKENS];
    __shared__ sv_fl<TOKENS> gs;
    __shared__ sv_fl<TOKENS> bs;
    __shared__ kittens::semaphore staged;

    const int h = blockIdx.y;
    const int kh = h / (g.nv / g.nk);
    const int row0 = blockIdx.x * ROWS;
    const int w = kittens::warpid();
    const int key_dim = g.nk * HEAD_K;

    const int slot = g.sequence[{0, 0, 0, 0}];
    const int q_len = g.sequence[{0, 0, 0, 1}];
    const bool fresh = g.sequence[{0, 0, 0, 2}] == 1;
    // This warp's 16 state rows, in units of 16 rows.
    const coord<state_tile> at = {slot, h, (row0 + w * WARP_ROWS) / WARP_ROWS, 0};

    state_tile S, T;
    if (fresh) {
        // A select: the slot is not read.
        warp::zero(S);
    } else {
        warp::load(S, g.ssm_state, at);
    }

    if (threadIdx.x == 0) {
        init_semaphore(staged, 0, 1);
    }
    __syncthreads();

    for (int tb = 0, block = 0; tb < q_len; tb += TOKENS, block++) {
        const int nt = min(TOKENS, q_len - tb);
        // Stage the block's real tokens: q and k of key head kh, v of the CTA's rows, g and β.
        if (threadIdx.x == 0) {
            tma::expect_bytes(staged, uint32_t(nt) * TOKEN_BYTES);
            for (int tt = 0; tt < nt; ++tt) {
                const int t = tb + tt;
                tma::load_async(qs[tt][0], g.conv_out[{0, 0, t, kh * HEAD_K}], QK_BYTES, staged);
                tma::load_async(ks[tt][0], g.conv_out[{0, 0, t, key_dim + kh * HEAD_K}], QK_BYTES,
                                staged);
                tma::load_async(vs[tt][0], g.conv_out[{0, 0, t, 2 * key_dim + h * g.head_v + row0}],
                                V_BYTES, staged);
                gs[tt] = g.log_decay[{0, 0, t, h}];
                bs[tt] = g.beta[{0, 0, t, h}];
            }
        }
        __syncthreads();
        wait(staged, block % 2);

        for (int tt = 0; tt < nt; ++tt) {
            // 1. The norms of q and k, and q̂, k̂.
            key_naive qn, kn, sq;
            warp::load(qn, qs[tt]);
            warp::load(kn, ks[tt]);
            warp::mul(sq, kn, kn);
            const float kk = warp::sum(sq);
            warp::mul(sq, qn, qn);
            const float qq = warp::sum(sq);
            const float k_inv = scratchy::ops::l2_inv::op<float>(kk);
            const float q_inv = scratchy::ops::l2_inv::op<float>(qq) * g.scale;
            warp::mul(kn, kn, k_inv);
            warp::mul(qn, qn, q_inv);
            // 2. The decay, first.
            const float decay = scratchy::ops::exp_of::op<float>(gs[tt]);
            warp::mul(S, S, decay);
            // 3. sk = S · k̂.
            key_vec kv;
            warp::copy(kv, kn);
            warp::mul_col(T, S, kv);
            row_vals sk;
            warp::row_sum(sk, T);
            // 4. The correction: write only what S does not already predict.
            row_naive vn;
            warp::load(vn, vs[tt].subvec<WARP_ROWS>(w));
            row_vals u;
            warp::copy(u, vn);
            warp::sub(u, u, sk);
            warp::mul(u, u, bs[tt]);
            // 5. The rank-1 update, S = k̂ · u + S per row.
            warp::broadcast_col(T, kv);
            warp::row_map<base_ops::fma_AxCtB>(S, T, S, u);
            // 6. o = S · q̂.
            warp::copy(kv, qn);
            warp::mul_col(T, S, kv);
            row_vals ov;
            warp::row_sum(ov, T);
            warp::store(g.o, ov,
                        {0, 0, tb + tt, (h * g.head_v + row0 + w * WARP_ROWS) / WARP_ROWS});
        }
        // The next block's staging overwrites what this one read.
        __syncthreads();
    }

    warp::store(g.ssm_state, S, at);
}

// TEST SUPPORT, for `tests/gdn.rs`'s `the_rank_one_update_is_t_times_u_plus_s` only (R8): no
// `CudaStep` uses it. One warp loads `S` and `T` (`[16, 128]` f32) and `u` (`[16]` f32) into the
// scan's own register types, applies the scan's rank-1 update `warp::row_map<fma_AxCtB>(S, T, S,
// u)`, and stores `S` into `out`: so a TK or flag change that alters the argument order shows.
struct rank_one_globals {
    row_gl s;
    row_gl t;
    row_gl u;
    row_gl out;
};

__global__ __launch_bounds__(kittens::WARP_THREADS) void gdn_rank_one_probe_ker(
    const __grid_constant__ rank_one_globals g) {
    state_tile S, T;
    row_vals u;
    warp::load(S, g.s, {0, 0, 0, 0});
    warp::load(T, g.t, {0, 0, 0, 0});
    warp::load(u, g.u, {0, 0, 0, 0});
    warp::row_map<base_ops::fma_AxCtB>(S, T, S, u);
    warp::store(g.out, S, {0, 0, 0, 0});
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught; any other value = the cudaError_t
// cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the runtime's last-error
// state.
//
// `head_k` must be 128, `head_v` a non-zero multiple of 32 and at most 256, `nk` and `nv` non-zero
// with `nv` a multiple of `nk`, `rows` non-zero, `conv_dim` exactly `2 · nk · 128 + nv · head_v`,
// `rows · conv_dim` at most INT_MAX, `scale` finite and above 0, and `slot_elems`
// (`GdnStateLayout::ssm_elems_per_slot`) exactly `nv · head_v · 128`; all are checked before any
// pointer is touched. `conv_out` is `[rows, conv_dim]` f32, `g` and `beta` `[rows, nv]` f32, `o`
// `[rows, nv · head_v]` f32, `sequence` `[slot, q_len, start, 0]` u32 with `1 ≤ q_len ≤ rows`, and
// `ssm_state` one layer's state buffer holding slot `slot`'s `slot_elems` f32 at
// `slot · slot_elems`. `conv_out` is 16-byte aligned (the bulk copies'). `o` and `ssm_state`
// overlap nothing else. 37,120 bytes of static shared memory, under the default 48 KiB, so no
// one-time setup. The launch is asynchronous on `stream`.
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
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        float* cp = static_cast<float*>(const_cast<void*>(conv_out));
        float* gp = static_cast<float*>(const_cast<void*>(g));
        float* bp = static_cast<float*>(const_cast<void*>(beta));
        float* sp = static_cast<float*>(ssm_state);
        int* qp = static_cast<int*>(const_cast<void*>(sequence));
        float* op = static_cast<float*>(o);
        // No view carries a TMA type, so building them encodes nothing: they are the pointers and
        // the dimensions the kernel indexes with.
        row_gl cg{cp, nullptr, nullptr, size_t(rows), size_t(conv_dim)};
        row_gl gg{gp, nullptr, nullptr, size_t(rows), size_t(nv)};
        row_gl bg{bp, nullptr, nullptr, size_t(rows), size_t(nv)};
        row_gl og{op, nullptr, nullptr, size_t(rows), size_t(nv) * head_v};
        state_gl sg{sp, size_t(1), size_t(nv), size_t(head_v), nullptr};
        seq_gl qg{qp, nullptr, nullptr, nullptr, size_t(4)};
        scan_globals G{cg, gg, bg, og, sg, qg, int(nk), int(nv), int(head_v), scale};
        void* args[] = {&G};
        const dim3 grid(head_v / ROWS, nv), block(NUM_THREADS);
        return static_cast<int>(cudaLaunchKernel(gdn_scan_ker, grid, block, args, 0,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// Test support for `tests/gdn.rs` only (R8): `out = T · u + S` by the scan's rank-1 update, `S`,
// `T` and `out` `[16, 128]` f32 and `u` `[16]` f32, 16-byte-aligned device pointers, `out`
// overlapping no input. 0 = launched; SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught; any
// other value = the launch's cudaError_t. Asynchronous on `stream`.
extern "C" int scratchy_cuda_gdn_rank_one_probe(const void* s, const void* t, const void* u,
                                                void* out,
                                                void* stream /* cudaStream_t; null = default */) {
    try {
        row_gl sg{static_cast<float*>(const_cast<void*>(s)), nullptr, nullptr, size_t(WARP_ROWS),
                  size_t(HEAD_K)};
        row_gl tg{static_cast<float*>(const_cast<void*>(t)), nullptr, nullptr, size_t(WARP_ROWS),
                  size_t(HEAD_K)};
        row_gl ug{static_cast<float*>(const_cast<void*>(u)), nullptr, nullptr, size_t(1),
                  size_t(WARP_ROWS)};
        row_gl og{static_cast<float*>(out), nullptr, nullptr, size_t(WARP_ROWS), size_t(HEAD_K)};
        rank_one_globals G{sg, tg, ug, og};
        void* args[] = {&G};
        return static_cast<int>(cudaLaunchKernel(gdn_rank_one_probe_ker, dim3(1),
                                                 dim3(kittens::WARP_THREADS), args, 0,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
