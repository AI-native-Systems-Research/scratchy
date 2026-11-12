// SPDX-License-Identifier: Apache-2.0
//
// Gated-DeltaNet chunked gated delta-rule scan for the H100 (FP8 S8 F6a, /tmp/a-f6-plan.md §2): the
// same recurrence as `gdn_scan.cu`, the same launcher contract, computed 64 tokens at a time on
// the tensor cores (fla's `chunk_gated_delta_rule`, the WY/UT form). No tape launches it yet.
//
// Per value head `h` (key head `h / (nv/nk)`), with q̂, k̂ the scan's normalised rows (q̂ scaled),
// over a chunk of C = 64 tokens (rows past `q_len` are zero: k̂ = q̂ = v = 0, g = β = 0), with
// `G_i = Σ_{j≤i} g_j` and the decay ratios `exp(G_i − G_j)` (differences, so nothing overflows):
//
//     A = strictly_lower(diag(β) · (K̂K̂ᵀ ∘ Γ))       Γ_ij = exp(G_i − G_j)
//     T = (I + A)⁻¹
//     U = T · diag(β) · V,   W = T · diag(β·exp(G)) · K̂
//     V_new = U − W · S_prevᵀ
//     O = diag(exp(G)) · Q̂ · S_prevᵀ + ((Q̂K̂ᵀ) ∘ Γ ∘ causal) · V_new
//     S_next = exp(G_C) · S_prev + V_newᵀ · diag(exp(G_C − G)) · K̂
//
// S as `[head_v, 128]`, as the slot holds it. The per-token rule `S ← S·exp(g)`, `u = β(v − S·k̂)`,
// `S ← S + u ⊗ k̂`, `o = S·q̂` is this form at C = 1; at C = 64 its order and precision move: the
// products are bf16 WGMMAs with f32 accumulation (q̂, k̂, v, T, W, P, V_new and the state rounded to
// bf16 as operands, once a chunk), the state itself stays f32 in registers.
//
// THE CTA: grid (head_v / 32, nv), as `gdn_scan`'s: 32 value rows of one head, so today's state slab
// and slot layout. One warpgroup (4 warps); warp `w` holds tokens `16w .. 16w + 16` of every
// `[64, *]` product. The state is held TRANSPOSED, `Sᵀ [128, 32]` f32, as two `rt_fl<16, 32>` per
// warp (key lanes `16w ..` of each 64-lane half): its update `Sᵀ += K̂ᵀ · (e ∘ V_new)` is then a
// 64-row `warpgroup::mma_AtB`, and `Sᵀ` rounded to bf16 is directly the B operand of `Q̂·Sᵀ` and
// `W·Sᵀ`. The slot is read and written through `gl::operator[]` via an f32 shared tile (the
// transposition), once at the start (never on a fresh start: the step-5b lesson) and once at the end.
//
// THE SOLVE, `T = (I + A)⁻¹` (A strictly lower, so T unit lower): A's four 16×16 diagonal blocks
// are inverted in f32 in shared memory, one warp each, in parallel: forward substitution on rows
// (`rv_fl<16>`, `warp::mul`/`warp::add`, A's entries by `st::operator[]`), 120 row steps. With D
// that block-diagonal inverse and N A's strictly block-lower part, `I + A = B(I + D·N)` and
// `M = D·N` is nilpotent of index 4, so `T = (I − M + M² − M³)·D = D + Y·D` with
// `Y = −M·(I − M·(I − M))`: four 64×64 bf16 WGMMAs (fla's 16-row blocking, its block products on the
// tensor cores). T's diagonal blocks are D itself, f32; its off-diagonal blocks carry the bf16
// operand rounding, as T does anyway as the operand of U and W.
//
// THE INPUTS: each warp loads its 16 token rows of q, k (two 64-lane halves each) and v (the CTA's
// 32 rows) from `conv_out` with `warp::load` (naive `rv_fl`), normalises q and k (`warp::sum` of
// the squares, `scratchy::ops::l2_inv`), and stores the bf16 rows into the operand tiles with
// `warp::store`. Only real rows are loaded: no thread reads a padding row. `g` and `β` reach
// shared vectors through `gl::operator[]`, and one thread sums `G` in token order (a fixed order).
//
// THE OUTPUT: `O` through an f32 shared tile, each real row stored by `warp::store` of an
// `rv_fl<32>`: padding rows are never written.
//
// Every product is a TK warpgroup MMA from shared tiles, every element map a TK map (`warp::apply`
// with the row and column, `mul`, `sub`), every exp `scratchy::ops::exp_of`. Writes to an MMA
// operand are made visible to the async proxy by `warpgroup::mma_fence()` before the barrier.
//
// CONTINUITY: chunks start at multiples of 64 from the launch's first token, so one forward and the
// same sequence in pieces give the same bits when each piece but the last is a multiple of 64
// tokens (the state between pieces is the f32 slot, read back exactly); elsewhere within the bound.
// No atomics: repeats are bit-identical.
//
// Built with `--use_fast_math` (build.rs): `expf` is `__expf`, `rsqrtf` the approximate one,
// subnormals flush to zero.
//
// TEST SUPPORT: `PART` instantiates the timing split (`scratchy_cuda_gdn_chunk_scan_part`): 1 skips
// the solve (T = I), 2 stops after the precompute (no `V_new`, `O` or state update). Neither is a
// correct scan; `PART` 0 is the kernel.

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

// `step::GDN_HEAD_K`, and the key half one 128-byte-swizzled bf16 tile holds.
constexpr int HEAD_K = 128;
constexpr int HALF = 64;
// `step::GDN_HEAD_V_MULTIPLE`: the value rows of one CTA.
constexpr int ROWS = 32;
// `step::GDN_HEAD_V_MAX`.
constexpr uint32_t HEAD_V_MAX = 256;
// The chunk: WGMMA's M.
constexpr int CHUNK = 64;
constexpr int WARPS = 4;
constexpr int WARP_TOKENS = CHUNK / WARPS;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;

constexpr int PART_FULL = 0;
constexpr int PART_NO_SOLVE = 1;
constexpr int PART_PRECOMPUTE = 2;

// A `[64, 64]` bf16 operand: a K̂ or Q̂ half, A's strictly block-lower part, D, M, the chain's
// factors, T's two scalings, W's halves, P.
using tok_sq = st_bf<CHUNK, HALF>;
// `[64 tokens, 32 rows]` bf16: V, V_new (then e ∘ V_new).
using tok_rows = st_bf<CHUNK, ROWS>;
// `[64 key lanes, 32 rows]` bf16: a half of Sᵀ, the B operand of Q̂·Sᵀ and W·Sᵀ.
using key_rows = st_bf<HALF, ROWS>;
// f32 `[64, 16]`: row `16w + r` holds warp w's diagonal block row r of A, then of D.
using diag_f = st_fl<CHUNK, 16>;
// f32 `[64, 32]`: O's rows on their way out, and a half of Sᵀ on its way in or out of the slot.
using rows_f = st_fl<CHUNK, ROWS>;

// A warp's 16 rows of a `[64, 64]`, `[64, 32]` and `[64, 16]` product.
using sq_rt = rt_fl<16, CHUNK>;
using rows_rt = rt_fl<16, ROWS>;
using diag_rt = rt_fl<16, 16>;

using row_gl = gl<float, 1, 1, -1, -1>;
// `[slots, nv, head_v, 128]`: indexed through `operator[]` only, which never reads the slot
// dimension's size, so it is 1.
using state_gl = gl<float, -1, -1, -1, HEAD_K>;
// `[slot, q_len, start, 0]`.
using seq_gl = gl<int, 1, 1, 1, -1>;

struct chunk_globals {
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

constexpr size_t SMEM_USED = 4 * sizeof(tok_sq) + 5 * sizeof(tok_sq) + 2 * sizeof(tok_rows) +
                             2 * sizeof(key_rows) + sizeof(diag_f) + sizeof(rows_f);
// The allocator's 1024-byte alignment of the first tile.
constexpr int SMEM_BYTES = int(SMEM_USED) + 1024;
static_assert(SMEM_BYTES <= 113 * 1024, "two CTAs fit an SM's 228 KiB");

// Warp `w`'s 16×16 block of a warp's `[16, 64]` rows: `w` is a runtime value, so each case names
// its block by a constant index.
__device__ inline void diag_block(diag_rt& dst, const sq_rt& src, int w) {
    switch (w) {
        case 0: dst.tiles[0][0] = src.tiles[0][0]; break;
        case 1: dst.tiles[0][0] = src.tiles[0][1]; break;
        case 2: dst.tiles[0][0] = src.tiles[0][2]; break;
        default: dst.tiles[0][0] = src.tiles[0][3]; break;
    }
}
__device__ inline void set_diag_block(sq_rt& dst, const diag_rt& src, int w) {
    switch (w) {
        case 0: dst.tiles[0][0] = src.tiles[0][0]; break;
        case 1: dst.tiles[0][1] = src.tiles[0][0]; break;
        case 2: dst.tiles[0][2] = src.tiles[0][0]; break;
        default: dst.tiles[0][3] = src.tiles[0][0]; break;
    }
}

// Every warp's operand stores visible to the next WGMMA: the async-proxy fence, then the barrier.
__device__ inline void operands_ready() {
    warpgroup::mma_fence();
    __syncthreads();
}

template <int PART>
__global__ __launch_bounds__(NUM_THREADS, 2) void gdn_chunk_scan_ker(
    const __grid_constant__ chunk_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    // [0] K̂'s halves, [1] Q̂'s.
    tok_sq(&kq)[2][2] = al.allocate<tok_sq, 2, 2>();
    // The chain's operands: [0] N, then Y, then W's low half; [1] D, then W's high half; [2] M,
    // then T·diag(β); [3] I − M, then I − M·(I − M), then T·diag(β·exp(G)); [4] P.
    tok_sq(&bt)[5] = al.allocate<tok_sq, 5>();
    tok_rows& vb = al.allocate<tok_rows>();
    tok_rows& vn = al.allocate<tok_rows>();
    key_rows(&stb)[2] = al.allocate<key_rows, 2>();
    diag_f& ad = al.allocate<diag_f>();
    rows_f& rf = al.allocate<rows_f>();
    // The chunk's `G` (its prefix sums of g) and β, zero past `q_len`.
    __shared__ sv_fl<CHUNK> gc;
    __shared__ sv_fl<CHUNK> bc;

    const int h = blockIdx.y;
    const int kh = h / (g.nv / g.nk);
    const int row0 = blockIdx.x * ROWS;
    const int w = kittens::warpid();
    const int i0 = w * WARP_TOKENS;
    const int key_dim = g.nk * HEAD_K;
    // The CTA's first column of v, and of o.
    const int v_col = 2 * key_dim + h * g.head_v + row0;
    const int o_col = h * g.head_v + row0;

    const int slot = g.sequence[{0, 0, 0, 0}];
    const int q_len = g.sequence[{0, 0, 0, 1}];
    const bool fresh = g.sequence[{0, 0, 0, 2}] == 1;

    // Sᵀ: [half] holds key lanes `64·half + 16w ..` of the CTA's 32 rows.
    rows_rt st[2];
    if (fresh) {
        // A select: the slot is not read.
        warp::zero(st[0]);
        warp::zero(st[1]);
    } else {
        for (int half = 0; half < 2; ++half) {
            // The slot's `[32 rows, 64 lanes]` into `[64 lanes, 32 rows]`; consecutive threads read
            // consecutive lanes.
            for (int e = threadIdx.x; e < HALF * ROWS; e += NUM_THREADS) {
                const int lane = e % HALF;
                const int r = e / HALF;
                rf[{lane, r}] = g.ssm_state[{slot, h, row0 + r, HALF * half + lane}];
            }
            __syncthreads();
            warpgroup::load(st[half], rf);
            __syncthreads();
        }
    }
    warpgroup::store(stb[0], st[0]);
    warpgroup::store(stb[1], st[1]);

    for (int c0 = 0; c0 < q_len; c0 += CHUNK) {
        const int nt = min(CHUNK, q_len - c0);
        // 1. The chunk's rows: q̂, k̂ and v as bf16 operands, zero past `q_len`; g and β.
        for (int r = 0; r < WARP_TOKENS; ++r) {
            const int tt = i0 + r;
            rv_fl<HALF> k0, k1, q0, q1, sq;
            rv_fl<ROWS> v;
            if (tt < nt) {
                const int t = c0 + tt;
                warp::load(q0, g.conv_out, {0, 0, t, kh * HEAD_K / HALF});
                warp::load(q1, g.conv_out, {0, 0, t, kh * HEAD_K / HALF + 1});
                warp::load(k0, g.conv_out, {0, 0, t, (key_dim + kh * HEAD_K) / HALF});
                warp::load(k1, g.conv_out, {0, 0, t, (key_dim + kh * HEAD_K) / HALF + 1});
                warp::load(v, g.conv_out, {0, 0, t, v_col / ROWS});
                warp::mul(sq, k0, k0);
                float kk = warp::sum(sq);
                warp::mul(sq, k1, k1);
                kk += warp::sum(sq);
                warp::mul(sq, q0, q0);
                float qq = warp::sum(sq);
                warp::mul(sq, q1, q1);
                qq += warp::sum(sq);
                const float k_inv = scratchy::ops::l2_inv::op<float>(kk);
                const float q_inv = scratchy::ops::l2_inv::op<float>(qq) * g.scale;
                warp::mul(k0, k0, k_inv);
                warp::mul(k1, k1, k_inv);
                warp::mul(q0, q0, q_inv);
                warp::mul(q1, q1, q_inv);
            } else {
                warp::zero(k0);
                warp::zero(k1);
                warp::zero(q0);
                warp::zero(q1);
                warp::zero(v);
            }
            warp::store(kq[0][0], k0, {tt, 0});
            warp::store(kq[0][1], k1, {tt, 0});
            warp::store(kq[1][0], q0, {tt, 0});
            warp::store(kq[1][1], q1, {tt, 0});
            warp::store(vb, v, {tt, 0});
        }
        if (threadIdx.x < CHUNK) {
            const int tt = threadIdx.x;
            gc[tt] = tt < nt ? g.log_decay[{0, 0, c0 + tt, h}] : 0.0f;
            bc[tt] = tt < nt ? g.beta[{0, 0, c0 + tt, h}] : 0.0f;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float run = 0.0f;
#pragma unroll
            for (int i = 0; i < CHUNK; ++i) {
                run += gc[i];
                gc[i] = run;
            }
        }
        operands_ready();

        // 2. K̂K̂ᵀ, Q̂K̂ᵀ, and Q̂·Sᵀ.
        sq_rt kk, qk;
        rows_rt o;
        warpgroup::mm_ABt(kk, kq[0][0], kq[0][0]);
        warpgroup::mma_ABt(kk, kq[0][1], kq[0][1]);
        warpgroup::mm_ABt(qk, kq[1][0], kq[0][0]);
        warpgroup::mma_ABt(qk, kq[1][1], kq[0][1]);
        warpgroup::mm_AB(o, kq[1][0], stb[0]);
        warpgroup::mma_AB(o, kq[1][1], stb[1]);
        warpgroup::mma_async_wait();

        // 3. A (its diagonal blocks f32, N its strictly block-lower part bf16), P, and the inter-
        //    chunk part of O, `exp(G_i) · (Q̂·Sᵀ)_i`.
        warp::apply(kk, kk, [&](int r, int c, float x) {
            const int i = i0 + r;
            return c < i ? x * bc[i] * scratchy::ops::exp_of::op<float>(gc[i] - gc[c]) : 0.0f;
        });
        diag_rt dg;
        diag_block(dg, kk, w);
        warpgroup::store(ad, dg);
        warp::apply(kk, kk, [&](int, int c, float x) { return c < i0 ? x : 0.0f; });
        warpgroup::store(bt[0], kk);
        warp::apply(qk, qk, [&](int r, int c, float x) {
            const int i = i0 + r;
            return c <= i ? x * scratchy::ops::exp_of::op<float>(gc[i] - gc[c]) : 0.0f;
        });
        warpgroup::store(bt[4], qk);
        warp::apply(o, o, [&](int r, int, float x) {
            return x * scratchy::ops::exp_of::op<float>(gc[i0 + r]);
        });

        // 4. T = (I + A)⁻¹ into `d`.
        sq_rt d;
        if constexpr (PART == PART_NO_SOLVE) {
            warp::zero(d);
            warp::apply(d, d, [&](int r, int c, float) { return c == i0 + r ? 1.0f : 0.0f; });
        } else {
            // D: warp w's diagonal block, in place, its rows `i0 ..` only. Row 0 is e_0 (A's row
            // 0 block is zero); row r is e_r − Σ_{j<r} A_rj · D_j, whose entry r is 0 (D_j is lower)
            // and becomes 1.
            __syncwarp();
            if (warp::laneid() == 0) {
                ad[{i0, 0}] = 1.0f;
            }
            __syncwarp();
            for (int r = 1; r < WARP_TOKENS; ++r) {
                rv_fl<16> acc, dj;
                warp::zero(acc);
                for (int j = 0; j < r; ++j) {
                    warp::load(dj, ad, {i0 + j, 0});
                    warp::mul(dj, dj, ad[{i0 + r, j}]);
                    warp::add(acc, acc, dj);
                }
                warp::mul(acc, acc, -1.0f);
                __syncwarp();
                warp::store(ad, acc, {i0 + r, 0});
                __syncwarp();
                if (warp::laneid() == 0) {
                    ad[{i0 + r, r}] = 1.0f;
                }
                __syncwarp();
            }
            warpgroup::load(dg, ad);
            warp::zero(d);
            set_diag_block(d, dg, w);
            warpgroup::store(bt[1], d);
            operands_ready();
            // M = D·N; then I − M.
            sq_rt m;
            warpgroup::mm_AB(m, bt[1], bt[0]);
            warpgroup::mma_async_wait();
            warpgroup::store(bt[2], m);
            warp::apply(m, m, [&](int r, int c, float x) {
                return (c == i0 + r ? 1.0f : 0.0f) - x;
            });
            warpgroup::store(bt[3], m);
            operands_ready();
            // I − M·(I − M).
            warpgroup::mm_AB(m, bt[2], bt[3]);
            warpgroup::mma_async_wait();
            __syncthreads();
            warp::apply(m, m, [&](int r, int c, float x) {
                return (c == i0 + r ? 1.0f : 0.0f) - x;
            });
            warpgroup::store(bt[3], m);
            operands_ready();
            // Y = −M·(I − M·(I − M)).
            warpgroup::mm_AB(m, bt[2], bt[3]);
            warpgroup::mma_async_wait();
            warp::mul(m, m, -1.0f);
            warpgroup::store(bt[0], m);
            operands_ready();
            // T = D + Y·D.
            warpgroup::mma_AB(d, bt[0], bt[1]);
            warpgroup::mma_async_wait();
            __syncthreads();
        }

        // 5. T·diag(β) and T·diag(β·exp(G)); U = T·diag(β)·V and W = T·diag(β·exp(G))·K̂.
        sq_rt tw;
        warp::apply(tw, d, [&](int, int c, float x) { return x * bc[c]; });
        warpgroup::store(bt[2], tw);
        warp::apply(tw, d, [&](int, int c, float x) {
            return x * bc[c] * scratchy::ops::exp_of::op<float>(gc[c]);
        });
        warpgroup::store(bt[3], tw);
        operands_ready();
        rows_rt u;
        sq_rt w_lo, w_hi;
        warpgroup::mm_AB(u, bt[2], vb);
        warpgroup::mm_AB(w_lo, bt[3], kq[0][0]);
        warpgroup::mm_AB(w_hi, bt[3], kq[0][1]);
        warpgroup::mma_async_wait();
        __syncthreads();
        warpgroup::store(bt[0], w_lo);
        warpgroup::store(bt[1], w_hi);
        operands_ready();

        if constexpr (PART != PART_PRECOMPUTE) {
            // 6. V_new = U − W·Sᵀ.
            rows_rt ws;
            warpgroup::mm_AB(ws, bt[0], stb[0]);
            warpgroup::mma_AB(ws, bt[1], stb[1]);
            warpgroup::mma_async_wait();
            warp::sub(u, u, ws);
            warpgroup::store(vn, u);
            operands_ready();

            // 7. O += P·V_new, its real rows stored.
            warpgroup::mma_AB(o, bt[4], vn);
            warpgroup::mma_async_wait();
            warpgroup::store(rf, o);
            __syncwarp();
            for (int r = 0; r < WARP_TOKENS; ++r) {
                const int tt = i0 + r;
                if (tt < nt) {
                    rv_fl<ROWS> ov;
                    warp::load(ov, rf, {tt, 0});
                    warp::store(g.o, ov, {0, 0, c0 + tt, o_col / ROWS});
                }
            }

            // 8. Sᵀ = exp(G_C)·Sᵀ + K̂ᵀ·(exp(G_C − G) ∘ V_new).
            const float g_last = gc[CHUNK - 1];
            warp::apply(u, u, [&](int r, int, float x) {
                return x * scratchy::ops::exp_of::op<float>(g_last - gc[i0 + r]);
            });
            __syncthreads();
            warpgroup::store(vn, u);
            operands_ready();
            const float decay = scratchy::ops::exp_of::op<float>(g_last);
            warp::mul(st[0], st[0], decay);
            warp::mul(st[1], st[1], decay);
            warpgroup::mma_AtB(st[0], kq[0][0], vn);
            warpgroup::mma_AtB(st[1], kq[0][1], vn);
            warpgroup::mma_async_wait();
            warpgroup::store(stb[0], st[0]);
            warpgroup::store(stb[1], st[1]);
        }
        // The next chunk's rows overwrite what this one's products read.
        __syncthreads();
    }

    // The state, back into the slot through the transposition tile.
    for (int half = 0; half < 2; ++half) {
        warpgroup::store(rf, st[half]);
        __syncthreads();
        for (int e = threadIdx.x; e < HALF * ROWS; e += NUM_THREADS) {
            const int lane = e % HALF;
            const int r = e / HALF;
            g.ssm_state[{slot, h, row0 + r, HALF * half + lane}] = rf[{lane, r}];
        }
        __syncthreads();
    }
}

// The launch shared by the kernel's launcher and the timing split's.
template <int PART>
int launch(const void* conv_out, const void* g, const void* beta, void* ssm_state,
           uint64_t slot_elems, const void* sequence, void* o, uint32_t rows, uint32_t nk,
           uint32_t nv, uint32_t head_k, uint32_t head_v, float scale, void* stream) {
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
        // No view carries a TMA type, so building them encodes nothing.
        row_gl cg{cp, nullptr, nullptr, size_t(rows), size_t(conv_dim)};
        row_gl gg{gp, nullptr, nullptr, size_t(rows), size_t(nv)};
        row_gl bg{bp, nullptr, nullptr, size_t(rows), size_t(nv)};
        row_gl og{op, nullptr, nullptr, size_t(rows), size_t(nv) * head_v};
        state_gl sg{sp, size_t(1), size_t(nv), size_t(head_v), nullptr};
        seq_gl qg{qp, nullptr, nullptr, nullptr, size_t(4)};
        chunk_globals G{cg, gg, bg, og, sg, qg, int(nk), int(nv), int(head_v), scale};
        void* args[] = {&G};
        const dim3 grid(head_v / ROWS, nv), block(NUM_THREADS);
        return static_cast<int>(cudaLaunchKernel(gdn_chunk_scan_ker<PART>, grid, block, args,
                                                 SMEM_BYTES, static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

}  // namespace

// The kernel's one-time setup, run once per device by `kernels::prepare_all`: allow each instance
// `SMEM_BYTES` of dynamic shared memory (which also loads its module, so a graph capture never
// loads a kernel mid-capture). 0 = done; any other value = the first failing cudaError_t.
extern "C" int scratchy_cuda_gdn_chunk_scan_prepare(void) {
    cudaError_t e = cudaFuncSetAttribute(gdn_chunk_scan_ker<PART_FULL>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) {
        return static_cast<int>(e);
    }
    e = cudaFuncSetAttribute(gdn_chunk_scan_ker<PART_NO_SOLVE>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) {
        return static_cast<int>(e);
    }
    return static_cast<int>(cudaFuncSetAttribute(gdn_chunk_scan_ker<PART_PRECOMPUTE>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 SMEM_BYTES));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught; any other value = the cudaError_t
// cudaLaunchKernel returned for THIS launch.
//
// `scratchy_cuda_gdn_scan`'s contract exactly (its arguments, its refusals, its buffers), computed
// in chunks of 64 tokens: `head_k` must be 128, `head_v` a non-zero multiple of 32 and at most 256,
// `nk` and `nv` non-zero with `nv` a multiple of `nk`, `rows` non-zero, `conv_dim` exactly
// `2 · nk · 128 + nv · head_v`, `rows · conv_dim` at most INT_MAX, `scale` finite and above 0, and
// `slot_elems` exactly `nv · head_v · 128`; all are checked before any pointer is touched.
// `conv_out` is `[rows, conv_dim]` f32, `g` and `beta` `[rows, nv]` f32, `o` `[rows, nv · head_v]`
// f32, `sequence` `[slot, q_len, start, 0]` u32 with `1 ≤ q_len ≤ rows`, `ssm_state` one layer's
// state buffer holding slot `slot`'s `slot_elems` f32 at `slot · slot_elems`. `o` and `ssm_state`
// overlap nothing else. `scratchy_cuda_gdn_chunk_scan_prepare` must have succeeded. Asynchronous
// on `stream`.
extern "C" int scratchy_cuda_gdn_chunk_scan(const void* conv_out, const void* g, const void* beta,
                                            void* ssm_state, uint64_t slot_elems,
                                            const void* sequence, void* o, uint32_t rows,
                                            uint32_t nk, uint32_t nv, uint32_t head_k,
                                            uint32_t head_v, float scale,
                                            void* stream /* cudaStream_t; null = default */) {
    return launch<PART_FULL>(conv_out, g, beta, ssm_state, slot_elems, sequence, o, rows, nk, nv,
                             head_k, head_v, scale, stream);
}

// Test support for `tests/gdn.rs`'s timing split only: `scratchy_cuda_gdn_chunk_scan` with `part`
// 0 (the kernel), 1 (the solve skipped, T = I) or 2 (stopped after the precompute: no V_new, O
// or state update). Parts 1 and 2 are not a scan; they time what they leave out. Any other `part`
// is SCRATCHY_CUDA_ERR_SHAPE.
extern "C" int scratchy_cuda_gdn_chunk_scan_part(const void* conv_out, const void* g,
                                                 const void* beta, void* ssm_state,
                                                 uint64_t slot_elems, const void* sequence,
                                                 void* o, uint32_t rows, uint32_t nk, uint32_t nv,
                                                 uint32_t head_k, uint32_t head_v, float scale,
                                                 uint32_t part, void* stream) {
    switch (part) {
        case uint32_t(PART_FULL):
            return launch<PART_FULL>(conv_out, g, beta, ssm_state, slot_elems, sequence, o, rows,
                                     nk, nv, head_k, head_v, scale, stream);
        case uint32_t(PART_NO_SOLVE):
            return launch<PART_NO_SOLVE>(conv_out, g, beta, ssm_state, slot_elems, sequence, o,
                                         rows, nk, nv, head_k, head_v, scale, stream);
        case uint32_t(PART_PRECOMPUTE):
            return launch<PART_PRECOMPUTE>(conv_out, g, beta, ssm_state, slot_elems, sequence, o,
                                           rows, nk, nv, head_k, head_v, scale, stream);
        default:
            return SCRATCHY_CUDA_ERR_SHAPE;
    }
}
