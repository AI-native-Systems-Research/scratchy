// SPDX-License-Identifier: Apache-2.0
//
// The one-row rope append for the H100 (FP8 S8 W4c): at one row, the decode step's query rope, its
// key rope and the K/V append into the e4m3 cache in ONE launch (`step::RopeAppendBf16::q`), and,
// on Gemma's sliding layers, the per-head q and k RMSNorms and the V unit norm before them
// (`::norms`). It replaces, at one row, `rope_bf16.cu`'s one-row form on q, the same on k and
// `kv_append_fp8.cu` (3 launches), and, normed, `rms_norm_bf16.cu`'s 16-row kernel on q's and k's
// head views and its unit form on V's head view (5 launches). Each output is the bits of the
// launches it replaces, by running their code on the same bytes:
//
//   - Grid: one CTA of one warp per head, `q_heads + kv_heads` of them, as the one-row rope
//     (`rope_bf16_ker<ROT, HALF_HEAD, true>`). CTA `h < q_heads` is q's head h; CTA `q_heads + j`
//     is k's head j, and also appends k's and v's head j.
//   - **The head norm** (normed only) is `rms_norm_bf16_ker`'s code at its one CTA over the head
//     view (`[heads, head_dim]`, `rows = heads ≤ 16`: one 16-row block, `head_dim / 256` column
//     tiles): each tile TMA-loaded into a plain `st_bf<16, 256, false>` (rows past `heads` zeros),
//     loaded into `rt_fl<16, 256>`, squared (`warp::mul`), the 3-operand `warp::row_sum`, the parts
//     added in warp order, `warp::div` by `cols`, `rms_inv` with `eps`, `mul_row` by `inv`,
//     `mul_col` by `0 + gain` (not under the unit norm), `warp::store` (the one RNE rounding). The
//     16-row kernel's warp `t` does exactly this for tile `t` of the block (tile 0 at Gemma's
//     sliding 256, tiles 0 and 1 at its global 512); its other warps own no tile and add exact
//     zeros in warp order, so one warp doing each owning warp's part gives the same total. Every
//     CTA of a view computes the whole block (a few hundred FMAs a tile), and keeps its own head's
//     row: `head % 16`.
//   - **The rope** is `rope_bf16_ker`'s one-row unit code on the head: its boxes of
//     `st_bf<16, 64, false>` (from the normed tile's row, or TMA-loaded), the table's boxes, the same
//     maps in the same order, each box stored back (the one rounding), then TMA-stored to its output.
//   - **The append** is `kv_append_fp8_ker`'s: k's roped head and v's (normed) head as `sv_bf` rows
//     converted to e4m3 (`warp::copy`, satfinite), TMA-stored at the slot's `{block, head, off, 0}`,
//     after the same checks (a slot of u32::MAX, or a block at or past `num_blocks`, writes nothing).
//     It reads the roped k from shared memory, as the append kernel read it from `k_out`; both are
//     the same bf16 bytes. `k_out` is written too: the attention reads it as its operand 3.
//
// One warp per CTA, so no CTA barrier; each stage's output is in shared memory before the next
// stage's `warp::load` of it (`__syncwarp`). Built with `--use_fast_math`, as each stage's kernel.
//
// THE FOUR-WARP FORM (FP8 S8 R1, `rope_append_wide_ker`, the launcher's): the same stages and the
// same TK ops on the same tiles, so the same bits, scheduled across four warps of the head's CTA
// instead of one warp's serial chain (25 × 10.4 µs a Gemma decode token):
//   - every load at once, by warp 0: the table, q's or k's head (its view tiles and gains when
//     normed), and, on k's heads that append, v's (its view tiles, or its row), each on its own
//     semaphore, where the one-warp form loaded v only after the rope;
//   - the norms' pass 1 in parallel: the q/k view's tile `t` by warp `t`, v's by warp `2 + t`, as
//     the 16-row kernel's warp `t` owns tile `t`; each warp's row sums into a shared vector, added
//     back in warp order (tile 0's, then tile 1's, then the exact zeros), then pass 2 by the same
//     warps;
//   - the rope's box pairs by warps `0 ..` (two at Gemma's sliding 256, one elsewhere), the head's
//     boxes stored by warps `b % 4`, k's and v's e4m3 rows converted and stored by warps 0 and 1.
// The one-warp form, `rope_append_ker`, stays as the bits the four-warp form is checked against
// (`scratchy_cuda_rope_append_fp8_variant`, test support).
//
// Instances (`step::ROPE_BF16_KERNEL_SHAPES` by head dim, and `NORMS` on the one the fold takes):
// `<64, false, 256>` (27B, 35B), `<256, false, 256>` and `<256, false, 256, true>` (Gemma sliding),
// `<128, true, 512>` and `<128, true, 512, true>` (Gemma global). Shared memory: the head's boxes
// (16 KiB at 512), the table's, and each view's norm tiles and gains (8 KiB and 512 B a tile),
// under 48 KiB but on the normed global instance (two tiles a view, 54 KiB), whose setup raises
// its limit.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

constexpr int ROWS = 16;
constexpr int BOX_COLS = 64;
constexpr int NORM_COLS = 256;
constexpr int NUM_THREADS = kittens::WARP_THREADS;
// `step::CUDA_KV_BLOCK_SIZE`, and a padding row's slot.
constexpr uint32_t BLOCK_SIZE = 64;
constexpr uint32_t NO_SLOT = 0xFFFFFFFFu;
constexpr uint32_t MAX_SLOT_BLOCKS = (NO_SLOT / BLOCK_SIZE) + 1;
constexpr int HALF_HEAD_DIM = 512;

using box = st_bf<ROWS, BOX_COLS, false>;
using box_gl = gl<bf16, 1, 1, -1, -1, box>;
using box_rt = rt_fl<ROWS, BOX_COLS>;
using half_rt = rt_fl<ROWS, BOX_COLS / 2>;
using norm_tile = st_bf<ROWS, NORM_COLS, false>;
using norm_gl = gl<bf16, 1, 1, -1, -1, norm_tile>;
using norm_rt = rt_fl<ROWS, NORM_COLS>;
using norm_rows = norm_rt::col_vec;
using norm_cols = norm_rt::row_vec;
using gain_vec = sv_bf<NORM_COLS>;
using gain_gl = gl<bf16, 1, 1, 1, -1, gain_vec>;
using slot_gl = gl<int, 1, 1, 1, -1>;
static_assert(half_rt::width == 2 && box_rt::width == 4, "the rope's halves, `rope_bf16.cu`'s");

// `rope_bf16.cu`'s `rope_shape`: `BOXES` boxes per unit, the table's `TABLE_BOXES`, the partner's
// box offset `PAIR_OFF`.
template <int ROT, bool HALF_HEAD>
struct rope_shape {
    static_assert((ROT == 64 || ROT == 256) != HALF_HEAD && (!HALF_HEAD || ROT == 128),
                  "the instantiated rotated widths and pairings");
    static constexpr int TABLE_BOXES = ROT / BOX_COLS;
    static constexpr int BOXES = HALF_HEAD ? HALF_HEAD_DIM / BOX_COLS : TABLE_BOXES;
    static constexpr int PAIR_OFF = HALF_HEAD ? BOXES / 2 : TABLE_BOXES / 2;
};

template <int HEAD_DIM>
using cache_vec = sv_fp8e4m3<HEAD_DIM>;
template <int HEAD_DIM>
using cache_gl = gl<fp8e4m3, -1, -1, int(BLOCK_SIZE), HEAD_DIM, cache_vec<HEAD_DIM>>;

template <int HEAD_DIM>
struct ra_globals {
    // q's input and output; k's input and roped output (the attention's operand 3); v's input.
    box_gl q;
    box_gl q_out;
    box_gl k;
    box_gl k_out;
    box_gl v;
    box_gl table;
    slot_gl slot_mapping;
    cache_gl<HEAD_DIM> k_cache;
    cache_gl<HEAD_DIM> v_cache;
    // The norms' head views `[heads, head_dim]` of q, k and v, and q's and k's gains (normed only).
    norm_gl qn;
    norm_gl kn;
    norm_gl vn;
    gain_gl q_gain;
    gain_gl k_gain;
    float eps;
    // `head_dim`, the norm's `cols`.
    float cols;
    int q_heads;
    int kv_heads;
    uint32_t num_blocks;
};

// `rms_norm_bf16_ker<FORM>`'s code at its one CTA over a head view of `heads ≤ 16` rows and
// `TILES ≤ 4` 256-column tiles, by one warp doing the work of its warps 0..TILES-1 (warp `t` owns
// tile `t`; the others own none): the block's tiles into `x_s`, normed, rounded, stored back.
// `UNIT`: no gain.
template <bool UNIT, int TILES>
__device__ inline void view_norm(norm_tile (&x_s)[TILES], gain_vec (&g_s)[TILES], const norm_gl& x,
                                 const gain_gl& gain, float eps, float cols, semaphore& arrived) {
    static_assert(TILES >= 1 && TILES <= 4, "one tile per warp of the 16-row kernel's four");
    warp::tma::expect_bytes(arrived, TILES * (sizeof(norm_tile) + (UNIT ? 0 : sizeof(gain_vec))));
    for (int t = 0; t < TILES; t++) {
        warp::tma::load_async(x_s[t], x, coord<norm_tile>{0, 0, 0, t}, arrived);
        if constexpr (!UNIT) {
            warp::tma::load_async(g_s[t], gain, coord<gain_vec>{0, 0, 0, t}, arrived);
        }
    }
    wait(arrived, 0);
    // Pass 1: warp `w`'s sum of squares of its tile into `part[w]` (warps past `TILES` hold `+0`),
    // then the four parts in warp order, then `inv` as the 16-row kernel's SASS forms it and the
    // one-row norm writes it out (`rms_norm_one_row_ker`, W1, bit-identical over 8 M outputs):
    // MUFU.RCP of `cols`, one FFMA `total · rcp + eps`, MUFU.RSQ (`scratchy::ops::rms_inv_fused`).
    norm_rows part[4];
    for (int w = 0; w < 4; w++) {
        warp::zero(part[w]);
    }
    norm_rt v;
    for (int t = 0; t < TILES; t++) {
        warp::load(v, x_s[t]);
        warp::mul(v, v, v);
        warp::row_sum(part[t], v, part[t]);
    }
    for (int w = 1; w < 4; w++) {
        warp::add(part[0], part[0], part[w]);
    }
    norm_rows rcp, inv;
    warp::one(rcp);
    warp::div(rcp, rcp, cols);
    scratchy::ops::rv_ternary_map<scratchy::ops::rms_inv_fused>(inv, part[0], rcp, eps);
    // Pass 2: `x · inv · (0 + gain)` per tile, the one rounding.
    for (int t = 0; t < TILES; t++) {
        warp::load(v, x_s[t]);
        warp::mul_row(v, v, inv);
        if constexpr (!UNIT) {
            norm_cols g;
            warp::load(g, g_s[t]);
            warp::add(g, g, 0.0f);
            warp::mul_col(v, v, g);
        }
        warp::store(x_s[t], v);
    }
    __syncwarp();
}

// The norm tiles an instance allocates per view: the head's 256-column tiles when normed, else
// one (never read).
template <int HEAD_DIM, bool NORMS>
constexpr int norm_tiles() {
    return NORMS ? HEAD_DIM / NORM_COLS : 1;
}

template <int ROT, bool HALF_HEAD, int HEAD_DIM, bool NORMS>
__global__ __launch_bounds__(NUM_THREADS, 1) void rope_append_ker(
    const __grid_constant__ ra_globals<HEAD_DIM> g) {
    constexpr int BOXES = rope_shape<ROT, HALF_HEAD>::BOXES;
    constexpr int TABLE_BOXES = rope_shape<ROT, HALF_HEAD>::TABLE_BOXES;
    constexpr int PAIR_OFF = rope_shape<ROT, HALF_HEAD>::PAIR_OFF;
    constexpr int HEAD_BOXES = HEAD_DIM / BOX_COLS;
    constexpr int NORM_TILES = norm_tiles<HEAD_DIM, NORMS>();
    static_assert(!NORMS || HEAD_DIM % NORM_COLS == 0, "a normed head is whole 256-lane tiles");
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    box(&x_s)[HEAD_BOXES] = al.allocate<box, HEAD_BOXES>();
    box(&t_s)[TABLE_BOXES] = al.allocate<box, TABLE_BOXES>();
    norm_tile(&n_s)[NORM_TILES] = al.allocate<norm_tile, NORM_TILES>();
    norm_tile(&nv_s)[NORM_TILES] = al.allocate<norm_tile, NORM_TILES>();
    gain_vec(&g_s)[NORM_TILES] = al.allocate<gain_vec, NORM_TILES>();
    __shared__ sv_bf<HEAD_DIM> k_buf;
    __shared__ sv_bf<HEAD_DIM> v_buf;
    __shared__ sv_fp8e4m3<HEAD_DIM> k_q;
    __shared__ sv_fp8e4m3<HEAD_DIM> v_q;
    __shared__ semaphore arrived, table_arrived, norm_arrived, v_arrived;

    const int head = blockIdx.x;
    const bool is_q = head < g.q_heads;
    const int h = is_q ? head : head - g.q_heads;

    if (warp::laneid() == 0) {
        init_semaphore(arrived, 0, 1);
        init_semaphore(table_arrived, 0, 1);
        init_semaphore(norm_arrived, 0, 1);
        init_semaphore(v_arrived, 0, 1);
    }
    __syncwarp();
    warp::tma::expect_bytes(table_arrived, TABLE_BOXES * sizeof(box));
    for (int b = 0; b < TABLE_BOXES; b++) {
        warp::tma::load_async(t_s[b], g.table, coord<box>{0, 0, 0, b}, table_arrived);
    }

    // The head's input boxes: its norm's row (normed), or its own bytes.
    if constexpr (NORMS) {
        view_norm<false>(n_s, g_s, is_q ? g.qn : g.kn, is_q ? g.q_gain : g.k_gain, g.eps, g.cols,
                         norm_arrived);
        // Row `h % 16` of the view's tiles is the head: box `b` is its lanes `64b .. 64b + 63`, into
        // row 0 of the box, through the tiles' accessors (a copy of bf16 bytes). Rows 1..15 are
        // zeros, as the one-row rope's TMA reads them past its one-row array; its store drops
        // them, as this one's does.
        for (int b = 0; b < HEAD_BOXES; b++) {
            warp::zero(x_s[b]);
        }
        __syncwarp();
        for (int e = warp::laneid(); e < HEAD_DIM; e += kittens::WARP_THREADS) {
            x_s[e / BOX_COLS][{0, e % BOX_COLS}] = n_s[e / NORM_COLS][{h % ROWS, e % NORM_COLS}];
        }
        __syncwarp();
    } else {
        const box_gl& src = is_q ? g.q : g.k;
        warp::tma::expect_bytes(arrived, HEAD_BOXES * sizeof(box));
        for (int b = 0; b < HEAD_BOXES; b++) {
            warp::tma::load_async(x_s[b], src, coord<box>{0, 0, 0, h * HEAD_BOXES + b}, arrived);
        }
        wait(arrived, 0);
    }
    wait(table_arrived, 0);

    // The rope: `rope_bf16_ker`'s one-row unit code on each unit of the head (its first rotated,
    // the rest passed through).
    for (int u = 0; u < HEAD_BOXES / BOXES; u++) {
        if (u == 0) {
            if constexpr (ROT == 64) {
                box_rt t;
                warp::load(t, t_s[0]);
                half_rt c, s;
                c.tiles[0][0] = t.tiles[0][0];
                c.tiles[0][1] = t.tiles[0][1];
                s.tiles[0][0] = t.tiles[0][2];
                s.tiles[0][1] = t.tiles[0][3];
                box_rt x;
                warp::load(x, x_s[0]);
                half_rt x0, x1, a, b;
                x0.tiles[0][0] = x.tiles[0][0];
                x0.tiles[0][1] = x.tiles[0][1];
                x1.tiles[0][0] = x.tiles[0][2];
                x1.tiles[0][1] = x.tiles[0][3];
                // y0 = x0·c − x1·s, into `a`.
                warp::mul(a, x0, c);
                warp::mul(b, x1, s);
                warp::sub(a, a, b);
                // y1 = x1·c + x0·s, into `b`.
                warp::mul(x1, x1, c);
                warp::mul(x0, x0, s);
                warp::add(b, x1, x0);
                x.tiles[0][0] = a.tiles[0][0];
                x.tiles[0][1] = a.tiles[0][1];
                x.tiles[0][2] = b.tiles[0][0];
                x.tiles[0][3] = b.tiles[0][1];
                warp::store(x_s[0], x);
            } else {
                constexpr int PAIR = TABLE_BOXES / 2;
                for (int b = 0; b < PAIR; b++) {
                    box_rt x0, x1, t, y0, y1;
                    warp::load(x0, x_s[b]);
                    warp::load(x1, x_s[b + PAIR_OFF]);
                    warp::load(t, t_s[b]);
                    warp::mul(y0, x0, t);
                    warp::mul(y1, x1, t);
                    warp::load(t, t_s[b + PAIR]);
                    // y0 = x0·c − x1·s.
                    warp::mul(x1, x1, t);
                    warp::sub(y0, y0, x1);
                    // y1 = x1·c + x0·s.
                    warp::mul(x0, x0, t);
                    warp::add(y1, y1, x0);
                    warp::store(x_s[b], y0);
                    warp::store(x_s[b + PAIR_OFF], y1);
                }
            }
            __syncwarp();
        }
    }
    const box_gl& dst = is_q ? g.q_out : g.k_out;
    for (int b = 0; b < HEAD_BOXES; b++) {
        warp::tma::store_async(dst, x_s[b], coord<box>{0, 0, 0, h * HEAD_BOXES + b});
    }

    // The append, k's heads only: `kv_append_fp8_ker`'s checks, conversion and stores.
    if (!is_q) {
        const uint32_t slot = uint32_t(g.slot_mapping[{0, 0, 0, 0}]);
        const uint32_t block = slot / BLOCK_SIZE, off = slot % BLOCK_SIZE;
        if (slot != NO_SLOT && block < g.num_blocks) {
            // V's head: its unit norm's row (normed), or its own bytes.
            if constexpr (NORMS) {
                // Its own tile: the k norm's generic-proxy writes to `n_s` precede no TMA load here.
                view_norm<true>(nv_s, g_s, g.vn, g.k_gain, g.eps, g.cols, v_arrived);
                for (int e = warp::laneid(); e < HEAD_DIM; e += kittens::WARP_THREADS) {
                    v_buf[e] = nv_s[e / NORM_COLS][{h % ROWS, e % NORM_COLS}];
                }
            } else {
                // `kv_append_fp8_ker`'s bulk load of the head's row.
                constexpr uint32_t bytes = HEAD_DIM * sizeof(bf16);
                warp::tma::expect_bytes(v_arrived, bytes);
                warp::tma::load_async(v_buf[0], g.v[{0, 0, 0, h * HEAD_DIM}], bytes, v_arrived);
                wait(v_arrived, 0);
            }
            // k's roped head from the boxes' row 0 (the bf16 bytes `k_out` holds).
            for (int e = warp::laneid(); e < HEAD_DIM; e += kittens::WARP_THREADS) {
                k_buf[e] = x_s[e / BOX_COLS][{0, e % BOX_COLS}];
            }
            __syncwarp();
            // Each element to e4m3, satfinite (`kv_append_fp8_ker`'s converting copy).
            warp::copy(k_q, k_buf);
            warp::copy(v_q, v_buf);
            __syncwarp();
            const coord<cache_vec<HEAD_DIM>> at = {int(block), h, int(off), 0};
            warp::tma::store_async(g.k_cache, k_q, at);
            warp::tma::store_async(g.v_cache, v_q, at);
        }
    }
    warp::tma::store_async_wait();
}

// The four-warp form's warps.
constexpr int WIDE_WARPS = 4;
constexpr int WIDE_THREADS = WIDE_WARPS * kittens::WARP_THREADS;

// `view_norm`'s loads: the view's tiles (and, but under `UNIT`, its gains) on `arrived`.
template <bool UNIT, int TILES>
__device__ inline void view_load(norm_tile (&x_s)[TILES], gain_vec (&g_s)[TILES], const norm_gl& x,
                                 const gain_gl& gain, semaphore& arrived) {
    warp::tma::expect_bytes(arrived, TILES * (sizeof(norm_tile) + (UNIT ? 0 : sizeof(gain_vec))));
    for (int t = 0; t < TILES; t++) {
        warp::tma::load_async(x_s[t], x, coord<norm_tile>{0, 0, 0, t}, arrived);
        if constexpr (!UNIT) {
            warp::tma::load_async(g_s[t], gain, coord<gain_vec>{0, 0, 0, t}, arrived);
        }
    }
}

// `view_norm`'s pass 1 on one tile: its rows' sums of squares (`part[t]` there) into `part_s`.
__device__ inline void view_part(const norm_tile& x_t, sv_fl<ROWS>& part_s) {
    norm_rows part;
    warp::zero(part);
    norm_rt v;
    warp::load(v, x_t);
    warp::mul(v, v, v);
    warp::row_sum(part, v, part);
    warp::store(part_s, part);
}

// `view_norm`'s total of the four warps' parts (the `TILES` tiles', then exact zeros, in warp order),
// `inv`, and pass 2 on one tile.
template <bool UNIT, int TILES>
__device__ inline void view_scale(norm_tile& x_t, const gain_vec& g_t,
                                  const sv_fl<ROWS> (&parts)[TILES], float eps, float cols) {
    norm_rows part[4];
    for (int w = 0; w < 4; w++) {
        warp::zero(part[w]);
    }
    for (int t = 0; t < TILES; t++) {
        warp::load(part[t], parts[t]);
    }
    for (int w = 1; w < 4; w++) {
        warp::add(part[0], part[0], part[w]);
    }
    norm_rows rcp, inv;
    warp::one(rcp);
    warp::div(rcp, rcp, cols);
    scratchy::ops::rv_ternary_map<scratchy::ops::rms_inv_fused>(inv, part[0], rcp, eps);
    norm_rt v;
    warp::load(v, x_t);
    warp::mul_row(v, v, inv);
    if constexpr (!UNIT) {
        norm_cols g;
        warp::load(g, g_t);
        warp::add(g, g, 0.0f);
        warp::mul_col(v, v, g);
    }
    warp::store(x_t, v);
    __syncwarp();
}

template <int ROT, bool HALF_HEAD, int HEAD_DIM, bool NORMS>
__global__ __launch_bounds__(WIDE_THREADS, 1) void rope_append_wide_ker(
    const __grid_constant__ ra_globals<HEAD_DIM> g) {
    constexpr int BOXES = rope_shape<ROT, HALF_HEAD>::BOXES;
    constexpr int TABLE_BOXES = rope_shape<ROT, HALF_HEAD>::TABLE_BOXES;
    constexpr int PAIR_OFF = rope_shape<ROT, HALF_HEAD>::PAIR_OFF;
    constexpr int HEAD_BOXES = HEAD_DIM / BOX_COLS;
    constexpr int NORM_TILES = norm_tiles<HEAD_DIM, NORMS>();
    static_assert(!NORMS || HEAD_DIM % NORM_COLS == 0, "a normed head is whole 256-lane tiles");
    static_assert(NORM_TILES <= 2, "the q/k view's tiles on warps 0, 1, v's on 2, 3");
    // Only the head's first unit is rotated, as `rope_append_ker` runs it (`BOXES` its boxes).
    static_assert(BOXES <= HEAD_BOXES, "a head is whole units");
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    box(&x_s)[HEAD_BOXES] = al.allocate<box, HEAD_BOXES>();
    box(&t_s)[TABLE_BOXES] = al.allocate<box, TABLE_BOXES>();
    norm_tile(&n_s)[NORM_TILES] = al.allocate<norm_tile, NORM_TILES>();
    norm_tile(&nv_s)[NORM_TILES] = al.allocate<norm_tile, NORM_TILES>();
    gain_vec(&g_s)[NORM_TILES] = al.allocate<gain_vec, NORM_TILES>();
    __shared__ sv_bf<HEAD_DIM> k_buf;
    __shared__ sv_bf<HEAD_DIM> v_buf;
    __shared__ sv_fp8e4m3<HEAD_DIM> k_q;
    __shared__ sv_fp8e4m3<HEAD_DIM> v_q;
    // Per view (q/k's, then v's), each tile's row sums.
    __shared__ sv_fl<ROWS> qk_parts[NORM_TILES];
    __shared__ sv_fl<ROWS> v_parts[NORM_TILES];
    __shared__ semaphore arrived, table_arrived, norm_arrived, v_arrived;

    const int head = blockIdx.x;
    const bool is_q = head < g.q_heads;
    const int h = is_q ? head : head - g.q_heads;
    const int w = kittens::warpid();
    // The append's slot and checks first (k's heads only), so v's load starts with the others.
    const uint32_t slot = is_q ? NO_SLOT : uint32_t(g.slot_mapping[{0, 0, 0, 0}]);
    const uint32_t block = slot / BLOCK_SIZE, off = slot % BLOCK_SIZE;
    const bool append = !is_q && slot != NO_SLOT && block < g.num_blocks;

    if (threadIdx.x == 0) {
        init_semaphore(arrived, 0, 1);
        init_semaphore(table_arrived, 0, 1);
        init_semaphore(norm_arrived, 0, 1);
        init_semaphore(v_arrived, 0, 1);
    }
    __syncthreads();
    // 1. Every load, at once.
    if (w == 0) {
        warp::tma::expect_bytes(table_arrived, TABLE_BOXES * sizeof(box));
        for (int b = 0; b < TABLE_BOXES; b++) {
            warp::tma::load_async(t_s[b], g.table, coord<box>{0, 0, 0, b}, table_arrived);
        }
        if constexpr (NORMS) {
            view_load<false>(n_s, g_s, is_q ? g.qn : g.kn, is_q ? g.q_gain : g.k_gain,
                             norm_arrived);
            if (append) {
                view_load<true>(nv_s, g_s, g.vn, g.k_gain, v_arrived);
            }
        } else {
            const box_gl& src = is_q ? g.q : g.k;
            warp::tma::expect_bytes(arrived, HEAD_BOXES * sizeof(box));
            for (int b = 0; b < HEAD_BOXES; b++) {
                warp::tma::load_async(x_s[b], src, coord<box>{0, 0, 0, h * HEAD_BOXES + b},
                                      arrived);
            }
            if (append) {
                // `kv_append_fp8_ker`'s bulk load of the head's row.
                constexpr uint32_t bytes = HEAD_DIM * sizeof(bf16);
                warp::tma::expect_bytes(v_arrived, bytes);
                warp::tma::load_async(v_buf[0], g.v[{0, 0, 0, h * HEAD_DIM}], bytes, v_arrived);
            }
        }
    }

    // 2. The norms (normed only), the q/k view on warps 0 .., v's on warps 2 ..; then the head's
    //    row into the boxes' row 0 (rows 1..15 zeros), v's into `v_buf`.
    if constexpr (NORMS) {
        const bool qk_warp = w < NORM_TILES;
        const bool v_warp = append && w >= 2 && w - 2 < NORM_TILES;
        if (qk_warp) {
            wait(norm_arrived, 0);
            view_part(n_s[w], qk_parts[w]);
        } else if (v_warp) {
            wait(v_arrived, 0);
            view_part(nv_s[w - 2], v_parts[w - 2]);
        }
        __syncthreads();
        if (qk_warp) {
            view_scale<false>(n_s[w], g_s[w], qk_parts, g.eps, g.cols);
        } else if (v_warp) {
            view_scale<true>(nv_s[w - 2], g_s[w - 2], v_parts, g.eps, g.cols);
        }
        for (int b = w; b < HEAD_BOXES; b += WIDE_WARPS) {
            warp::zero(x_s[b]);
        }
        __syncthreads();
        for (int e = threadIdx.x; e < HEAD_DIM; e += WIDE_THREADS) {
            x_s[e / BOX_COLS][{0, e % BOX_COLS}] = n_s[e / NORM_COLS][{h % ROWS, e % NORM_COLS}];
            if (append) {
                v_buf[e] = nv_s[e / NORM_COLS][{h % ROWS, e % NORM_COLS}];
            }
        }
    } else {
        wait(arrived, 0);
    }
    __syncthreads();

    // 3. The rope: `rope_append_ker`'s code, its pairs on warps 0 ...
    if constexpr (ROT == 64) {
        if (w == 0) {
            wait(table_arrived, 0);
            box_rt t;
            warp::load(t, t_s[0]);
            half_rt c, s;
            c.tiles[0][0] = t.tiles[0][0];
            c.tiles[0][1] = t.tiles[0][1];
            s.tiles[0][0] = t.tiles[0][2];
            s.tiles[0][1] = t.tiles[0][3];
            box_rt x;
            warp::load(x, x_s[0]);
            half_rt x0, x1, a, b;
            x0.tiles[0][0] = x.tiles[0][0];
            x0.tiles[0][1] = x.tiles[0][1];
            x1.tiles[0][0] = x.tiles[0][2];
            x1.tiles[0][1] = x.tiles[0][3];
            // y0 = x0·c − x1·s, into `a`.
            warp::mul(a, x0, c);
            warp::mul(b, x1, s);
            warp::sub(a, a, b);
            // y1 = x1·c + x0·s, into `b`.
            warp::mul(x1, x1, c);
            warp::mul(x0, x0, s);
            warp::add(b, x1, x0);
            x.tiles[0][0] = a.tiles[0][0];
            x.tiles[0][1] = a.tiles[0][1];
            x.tiles[0][2] = b.tiles[0][0];
            x.tiles[0][3] = b.tiles[0][1];
            warp::store(x_s[0], x);
        }
    } else {
        constexpr int PAIR = TABLE_BOXES / 2;
        static_assert(PAIR <= WIDE_WARPS, "one pair a warp");
        if (w < PAIR) {
            wait(table_arrived, 0);
            const int b = w;
            box_rt x0, x1, t, y0, y1;
            warp::load(x0, x_s[b]);
            warp::load(x1, x_s[b + PAIR_OFF]);
            warp::load(t, t_s[b]);
            warp::mul(y0, x0, t);
            warp::mul(y1, x1, t);
            warp::load(t, t_s[b + PAIR]);
            // y0 = x0·c − x1·s.
            warp::mul(x1, x1, t);
            warp::sub(y0, y0, x1);
            // y1 = x1·c + x0·s.
            warp::mul(x0, x0, t);
            warp::add(y1, y1, x0);
            warp::store(x_s[b], y0);
            warp::store(x_s[b + PAIR_OFF], y1);
        }
    }
    __syncthreads();

    // 4. The roped head to its output, its boxes on warps `b % 4`.
    const box_gl& dst = is_q ? g.q_out : g.k_out;
    for (int b = w; b < HEAD_BOXES; b += WIDE_WARPS) {
        warp::tma::store_async(dst, x_s[b], coord<box>{0, 0, 0, h * HEAD_BOXES + b});
    }

    // 5. The append: k's roped row (the boxes' row 0, `k_out`'s bytes) and v's, each to e4m3 and
    //    its cache, on warps 0 and 1.
    if (append) {
        for (int e = threadIdx.x; e < HEAD_DIM; e += WIDE_THREADS) {
            k_buf[e] = x_s[e / BOX_COLS][{0, e % BOX_COLS}];
        }
        __syncthreads();
        const coord<cache_vec<HEAD_DIM>> at = {int(block), h, int(off), 0};
        if (w == 0) {
            warp::copy(k_q, k_buf);
            __syncwarp();
            warp::tma::store_async(g.k_cache, k_q, at);
        } else if (w == 1) {
            if constexpr (!NORMS) {
                wait(v_arrived, 0);
            }
            warp::copy(v_q, v_buf);
            __syncwarp();
            warp::tma::store_async(g.v_cache, v_q, at);
        }
    }
    warp::tma::store_async_wait();
}

// The instance's dynamic shared memory: the head's boxes, the table's, the two views' norm tiles
// and the gain vectors, plus 1,024 bytes of alignment slack.
template <int ROT, bool HALF_HEAD, int HEAD_DIM, bool NORMS>
constexpr int smem_bytes() {
    return int((HEAD_DIM / BOX_COLS + rope_shape<ROT, HALF_HEAD>::TABLE_BOXES) * sizeof(box) +
               norm_tiles<HEAD_DIM, NORMS>() * (2 * sizeof(norm_tile) + sizeof(gain_vec))) +
           1024;
}
static_assert(smem_bytes<128, true, 512, false>() + 3 * 512 * 2 <= 48 * 1024 &&
                  smem_bytes<256, false, 256, true>() + 3 * 256 * 2 <= 48 * 1024,
              "under the default dynamic shared memory limit, with the static vectors");
// The normed global instance's two tiles per view take it past 48 KiB: its setup raises the limit.
constexpr int HALF_NORMED_SMEM = smem_bytes<128, true, 512, true>();
static_assert(HALF_NORMED_SMEM > 48 * 1024 && HALF_NORMED_SMEM + 3 * 512 * 2 <= 227 * 1024,
              "past the default limit, under the H100's opt-in one, with the static vectors");

// The launch's operands, as the launcher was given them.
struct ra_args {
    const void* q = nullptr;
    void* q_out = nullptr;
    const void* k = nullptr;
    void* k_out = nullptr;
    const void* v = nullptr;
    const void* table = nullptr;
    const void* slot_mapping = nullptr;
    void* k_cache = nullptr;
    void* v_cache = nullptr;
    const void* q_gain = nullptr;
    const void* k_gain = nullptr;
};

template <int ROT, bool HALF_HEAD, int HEAD_DIM, bool NORMS>
int launch(const ra_args& a, uint32_t q_heads, uint32_t kv_heads, uint32_t num_blocks, float eps,
           bool wide, cudaStream_t s) {
    const auto bf = [](const void* p) { return static_cast<bf16*>(const_cast<void*>(p)); };
    const size_t qc = size_t(q_heads) * HEAD_DIM, kc = size_t(kv_heads) * HEAD_DIM;
    const size_t blocks = size_t(num_blocks < MAX_SLOT_BLOCKS ? num_blocks : MAX_SLOT_BLOCKS);
    // Building each view encodes its tensor map into host memory; nothing is allocated on the
    // device. The norm views exist on every instance and are read only by `NORMS`.
    box_gl qg{bf(a.q), nullptr, nullptr, size_t(1), qc};
    box_gl qog{bf(a.q_out), nullptr, nullptr, size_t(1), qc};
    box_gl kg{bf(a.k), nullptr, nullptr, size_t(1), kc};
    box_gl kog{bf(a.k_out), nullptr, nullptr, size_t(1), kc};
    box_gl vg{bf(a.v), nullptr, nullptr, size_t(1), kc};
    box_gl tg{bf(a.table), nullptr, nullptr, size_t(1), size_t(ROT)};
    slot_gl sg{static_cast<int*>(const_cast<void*>(a.slot_mapping)), nullptr, nullptr, nullptr,
               size_t(1)};
    cache_gl<HEAD_DIM> kcg{static_cast<fp8e4m3*>(a.k_cache), blocks, size_t(kv_heads), nullptr,
                           nullptr};
    cache_gl<HEAD_DIM> vcg{static_cast<fp8e4m3*>(a.v_cache), blocks, size_t(kv_heads), nullptr,
                           nullptr};
    norm_gl qn{bf(a.q), nullptr, nullptr, size_t(q_heads), size_t(HEAD_DIM)};
    norm_gl kn{bf(a.k), nullptr, nullptr, size_t(kv_heads), size_t(HEAD_DIM)};
    norm_gl vn{bf(a.v), nullptr, nullptr, size_t(kv_heads), size_t(HEAD_DIM)};
    // Unnormed, the gain views are never read: they view q, which holds at least `head_dim` bf16.
    const void* qgain = NORMS ? a.q_gain : a.q;
    const void* kgain = NORMS ? a.k_gain : a.q;
    gain_gl qgg{bf(qgain), nullptr, nullptr, nullptr, size_t(HEAD_DIM)};
    gain_gl kgg{bf(kgain), nullptr, nullptr, nullptr, size_t(HEAD_DIM)};
    ra_globals<HEAD_DIM> G{qg, qog, kg, kog, vg, tg, sg, kcg, vcg, qn, kn, vn, qgg, kgg,
                           eps, float(HEAD_DIM), int(q_heads), int(kv_heads), num_blocks};
    void* args[] = {&G};
    const dim3 grid{q_heads + kv_heads};
    constexpr int smem = smem_bytes<ROT, HALF_HEAD, HEAD_DIM, NORMS>();
    if (wide) {
        return static_cast<int>(cudaLaunchKernel(rope_append_wide_ker<ROT, HALF_HEAD, HEAD_DIM, NORMS>,
                                                 grid, dim3(WIDE_THREADS), args, smem, s));
    }
    return static_cast<int>(cudaLaunchKernel(rope_append_ker<ROT, HALF_HEAD, HEAD_DIM, NORMS>, grid,
                                             dim3(NUM_THREADS), args, smem, s));
}

}  // namespace

// The pairings (`step::RopePairing`, in its declaration order), as `rope_bf16.cu`'s.
constexpr uint32_t PAIRING_HALF_ROTATED = 0;
constexpr uint32_t PAIRING_HALF_HEAD = 1;

// The kernel's one-time setup, run once per device by `kernels::prepare_all`: query each
// instance's attributes, which loads its module, so a graph capture never loads one mid-capture,
// and allow the normed global instance its `HALF_NORMED_SMEM` of dynamic shared memory.
//
// 0 = done; any other value = the cudaError_t of the first call that failed.
extern "C" int scratchy_cuda_rope_append_fp8_prepare(void) {
    cudaFuncAttributes attr;
    for (const cudaError_t e : {
             cudaFuncGetAttributes(&attr, rope_append_ker<64, false, 256, false>),
             cudaFuncGetAttributes(&attr, rope_append_ker<256, false, 256, false>),
             cudaFuncGetAttributes(&attr, rope_append_ker<256, false, 256, true>),
             cudaFuncGetAttributes(&attr, rope_append_ker<128, true, 512, false>),
             cudaFuncGetAttributes(&attr, rope_append_ker<128, true, 512, true>),
             cudaFuncSetAttribute(rope_append_ker<128, true, 512, true>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, HALF_NORMED_SMEM),
             cudaFuncGetAttributes(&attr, rope_append_wide_ker<64, false, 256, false>),
             cudaFuncGetAttributes(&attr, rope_append_wide_ker<256, false, 256, false>),
             cudaFuncGetAttributes(&attr, rope_append_wide_ker<256, false, 256, true>),
             cudaFuncGetAttributes(&attr, rope_append_wide_ker<128, true, 512, false>),
             cudaFuncGetAttributes(&attr, rope_append_wide_ker<128, true, 512, true>),
             cudaFuncSetAttribute(rope_append_wide_ker<128, true, 512, true>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, HALF_NORMED_SMEM),
         }) {
        if (e != cudaSuccess) {
            return static_cast<int>(e);
        }
    }
    return 0;
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched; SCRATCHY_CUDA_ERR_HOST
// = a C++ exception was caught (a tensor-map encode); any other value = the cudaError_t
// cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the runtime's last-error
// state.
//
// At one row: `q_out = rope(q)`, `k_out = rope(k)`, and k_out's and v's head rows appended to the
// e4m3 caches at `slot_mapping[0]`, as `scratchy_cuda_rope_paired_bf16` on q, the same on k and
// `scratchy_cuda_kv_append_fp8`, bit for bit. With `q_gain` (null: no norms), first q's and k's
// heads by their RMSNorm (gain `q_gain`, `k_gain`, the stored convention, `eps`) and v's heads by
// the unit norm (`eps`), as `scratchy_cuda_rms_norm_bf16` and `scratchy_cuda_rms_norm_unit_bf16` on
// the head views `[heads, head_dim]`. Refused before any pointer is touched: `(rot_dim, pairing,
// head_dim)` not (64, half-rotated, 256), (256, half-rotated, 256) or (128, half-head, 512); the
// heads zero or above 16 under the norms, which take (256, half-rotated, 256) and (128, half-head,
// 512) only, at a finite
// positive `eps`; `block_size` not 64. `q`, `q_out` hold `q_heads · head_dim` bf16, `k`, `k_out`,
// `v` `kv_heads · head_dim`, the table `rot_dim`, the gains `head_dim` bf16; the outputs overlap no
// input; the caches as `scratchy_cuda_kv_append_fp8`'s. `num_blocks` 0 appends nothing. The launch
// is asynchronous on `stream`.
static int rope_append(const void* q, void* q_out, const void* k, void* k_out, const void* v,
                       const void* table, const void* slot_mapping, void* k_cache, void* v_cache,
                       uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t rot_dim,
                       uint32_t pairing, uint32_t block_size, uint32_t num_blocks,
                       const void* q_gain, const void* k_gain, float eps, bool wide,
                       void* stream) {
    const bool narrow = pairing == PAIRING_HALF_ROTATED && rot_dim == 64 && head_dim == 256;
    const bool full = pairing == PAIRING_HALF_ROTATED && rot_dim == 256 && head_dim == 256;
    const bool half = pairing == PAIRING_HALF_HEAD && rot_dim == 128 && head_dim == 512;
    const bool norms = q_gain != nullptr;
    if (!(narrow || full || half) || q_heads == 0 || kv_heads == 0 || block_size != BLOCK_SIZE ||
        uint64_t(q_heads + kv_heads) > 65535 ||
        (norms && (!(full || half) || k_gain == nullptr || q_heads > uint32_t(ROWS) ||
                   kv_heads > uint32_t(ROWS) || !(eps > 0.0f && eps <= 3.0e38f)))) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const ra_args a{q, q_out, k, k_out, v, table, slot_mapping, k_cache, v_cache, q_gain, k_gain};
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        if (half) {
            return norms ? launch<128, true, 512, true>(a, q_heads, kv_heads, num_blocks, eps, wide, s)
                         : launch<128, true, 512, false>(a, q_heads, kv_heads, num_blocks, eps, wide, s);
        }
        if (narrow) {
            return launch<64, false, 256, false>(a, q_heads, kv_heads, num_blocks, eps, wide, s);
        }
        return norms ? launch<256, false, 256, true>(a, q_heads, kv_heads, num_blocks, eps, wide, s)
                     : launch<256, false, 256, false>(a, q_heads, kv_heads, num_blocks, eps, wide, s);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

extern "C" int scratchy_cuda_rope_append_fp8(
    const void* q, void* q_out, const void* k, void* k_out, const void* v, const void* table,
    const void* slot_mapping, void* k_cache, void* v_cache, uint32_t q_heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t rot_dim, uint32_t pairing, uint32_t block_size,
    uint32_t num_blocks, const void* q_gain, const void* k_gain, float eps,
    void* stream /* cudaStream_t; null = default */) {
    return rope_append(q, q_out, k, k_out, v, table, slot_mapping, k_cache, v_cache, q_heads,
                       kv_heads, head_dim, rot_dim, pairing, block_size, num_blocks, q_gain, k_gain,
                       eps, true, stream);
}

// Test support for `tests/rope_gate.rs` only (FP8 S8 R1): `scratchy_cuda_rope_append_fp8` at
// `variant` 0 (`rope_append_ker`, the one-warp form W4c built) or 1 (`rope_append_wide_ker`, the
// launcher's own), the same bits. Refused as `scratchy_cuda_rope_append_fp8`, or for `variant`
// above 1.
extern "C" int scratchy_cuda_rope_append_fp8_variant(
    const void* q, void* q_out, const void* k, void* k_out, const void* v, const void* table,
    const void* slot_mapping, void* k_cache, void* v_cache, uint32_t q_heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t rot_dim, uint32_t pairing, uint32_t block_size,
    uint32_t num_blocks, const void* q_gain, const void* k_gain, float eps, uint32_t variant,
    void* stream /* cudaStream_t; null = default */) {
    if (variant > 1) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    return rope_append(q, q_out, k, k_out, v, table, slot_mapping, k_cache, v_cache, q_heads,
                       kv_heads, head_dim, rot_dim, pairing, block_size, num_blocks, q_gain, k_gain,
                       eps, variant == 1, stream);
}
