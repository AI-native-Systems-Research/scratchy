// SPDX-License-Identifier: Apache-2.0
//
// bf16 RMSNorm for the H100: y[r, c] = x[r, c] / sqrt(mean(x[r, :]²) + eps) · (gain_offset +
// gain[c]), `x` and `y` `[rows, cols]` row-major bf16, `gain` `[cols]` bf16. Computed in f32,
// rounded once to bf16 (round-to-nearest-even) per output. `gain_offset` is the gain convention's
// (`GainConvention::offset`: 0 for Llama's gain, 1 for qwen3.5's zero-centred one).
//
// The kernel is `template <bool UNIT>`. `rms_norm_bf16_ker<false>` is the norm above. Under `UNIT`
// (`scratchy_cuda_rms_norm_unit_bf16`, Gemma's `v_norm`, Metal's `rmsnorm_unit_impl`) there is no
// gain: `y = x · inv`, the gain vector neither loaded nor multiplied (its TMA load and
// `warp::mul_col` are compiled out). Everything else, the two passes, the fixed reduction order
// and `rms_inv`, is the same code, so the unit norm gives the bits of `<false>` with a gain of
// all ones and offset 0 (a multiply by exactly 1.0 in f32 changes nothing).
//
// Two more forms fold the residual add into the norm (#1049), each the bits of the two launches it
// replaces:
//   - `ADD_NORM` (`scratchy_cuda_add_rms_norm_bf16`, Qwen's `norm(add(a, b))`): every tile loads
//     `a` and `b`, adds them in f32 (`warp::add`, `add_bf16.cu`'s one add) and rounds the sum to
//     bf16 through a bf16 register tile (`warp::copy`, the convertor `warp::store` rounds with);
//     the norm reads that rounded sum, widened back (exact). Pass 1 stores the sum (`sum`, the
//     add's output); pass 2 recomputes it from `a` and `b`, the same bits, and norms it.
//   - `NORM_ADD` (`scratchy_cuda_norm_add_bf16`, Gemma's `add(norm(x), residual)`): the norm
//     above, its output rounded to bf16 through a bf16 register tile, widened back (exact) and
//     added to the tile of `residual` in f32, then stored (the add's one rounding).
// The forms are `template <int FORM>`: the two passes, the reduction order and `rms_inv` are the
// plain norm's code, so each output is the bits of today's `Add` then `RmsNorm`, or `RmsNorm` then
// `Add`.
// ThunderKittens' 16-row tile (PLAN-tk100 §2 S2, §3.3): one CTA per block of 16 rows, one
// warpgroup of 4 warps. Warp `w` takes the 256-column tiles `w, w + 4, …` of the block, in two
// passes over `x`, each tile TMA-loaded by the warp's lane 0 into a plain `st_bf<16, 256, false>`
// through a 2-stage ring per warp:
//   1. The sum of squares: `warp::load` into `rt_fl<16, 256>` (exact), `warp::mul` by itself,
//      `warp::row_sum` into the warp's `col_vec` across its tiles in ascending order. Each warp
//      stores its sums to an `sv_fl<16>`; after the warpgroup's barrier, every warp loads the four
//      and adds them in warp order 0, 1, 2, 3, so all four hold the same bits. Then
//      `inv = rsqrt(sum / cols + eps)`: `warp::div` by `cols`, then `scratchy::ops::rms_inv` with
//      `eps`, today's expression.
//   2. The output: `x` re-read by TMA (the plain-CUDA kernel also read it twice), `warp::mul_row`
//      by `inv`, `warp::mul_col` by the tile's gain plus `gain_offset` (a `row_vec`), stored into
//      the tile (`warp::store`, the one RNE rounding) and TMA-stored.
// A tile's gain is TMA-loaded as an `sv_bf<256>`: a vector box is never swizzled and is bounded by
// the array's `cols`, so the columns past `cols` of a partial last tile read zeros. Every tile is
// plain, never swizzled: a box past the array's rows (the last block's padding) or columns
// (`cols % 256 ≠ 0`) reads zeros and stores nothing (TK-1's R1 gate, `tests/tma_bounds.rs`). A
// padding row's sum is 0, its `inv` `rsqrt(eps)`, finite, and it is never stored; rows never mix.
//
// The reduction order, fixed: `row_sum`'s within a tile (`ops/group/register/tile/
// reductions.cuh`), the warp's tiles ascending, then the four warps in order. It is not the
// plain-CUDA kernel's (each lane's `fmaf` chain, a xor butterfly, then 8 warps), so the bits
// differ within the f64 bound; it is the same on every run, so repeats are bit-identical. A product
// is no longer fused into its sum (`mul`, then `row_sum`): one more rounding per element, within
// the bound's 2⁻⁸.
//
// THE ONE-ROW FORM (`rms_norm_one_row_ker<FORM>`, FP8 S8 W1): at `rows == 1` (every decode step)
// the 16-row kernel is one CTA walking the whole row on one SM. The launcher then takes this
// kernel instead, the same bits on `col_tiles` CTAs, chosen from `rows` alone (a constant of each
// step), with no cross-CTA read and no scratch:
//   - CTA `c` owns column tile `c`. Its 4 warps each take the tiles `w, w + 4, …` of the row as
//     the 16-row kernel's do, and build `part_w` the same way: tile `t`'s sum of squares by the
//     same `warp::load`, `warp::mul` and 3-operand `warp::row_sum(part, x², part)`, in ascending
//     `t`. Then the four `part_w` are added in warp order, `rms_inv`, all as the 16-row kernel.
//     So every CTA computes the row's `inv` itself, in today's order: the same bits on every CTA.
//     The cost is that each CTA reads the whole row (and, under `ADD_NORM`, `b`): L2 traffic of
//     `col_tiles · 2 · cols` bytes, which a one-row norm affords.
//   - Then CTA `c`'s warp 0 runs pass 2 on tile `c` only: the 16-row kernel's pass-2 body, the
//     same TK ops in the same order. `ADD_NORM`'s sum is stored by the owner too.
//   - A one-row tile is still `st_bf<16, 256, false>`: rows 1..15 of the box are past the
//     array's one row, so they read zeros and store nothing (TK-1's R1 gate), as the 16-row
//     kernel's padding rows do. So each element's arithmetic is unchanged.
//
// THE ROW FORM (`rms_norm_row_ker<FORM>`, FP8 S8 N1): the one-row form, with pass 1's 16-row tile
// walk (15 zero rows, a 2-stage ring of 4 warps: ~8.7 µs a launch on the H100 in a decode graph)
// replaced by the same sums on the one row, in the same order:
//   - The row (and `ADD_NORM`'s `b`) is TMA-loaded once as `sv_bf<256>` chunks (zero past `cols`).
//   - Thread `(t, q)` (`4t + q`) forms lane `q`'s `row_sum` chain of tile `t`'s row 0
//     (`reductions.cuh`'s `row_reduce` on `rt_fl<16, 256>`): row 0 of a 16 × 16 subtile `j` is
//     `data[0]`, columns `16j + 2q, +1`, and `data[2]`, columns `16j + 8 + 2q, +1` (`ldmatrix`'s
//     fragment), so the float2 chain is `x²[2q] + x²[8 + 2q]`, then `+ x²[16j + 2q]` and
//     `+ x²[16j + 8 + 2q]` for `j = 1 .. 15`, then `.x + .y`: `s_q`. The square is `warp::mul`'s
//     (`base_ops::mul` on float2), every add `base_ops::sum`'s, so the dataflow is the tile's.
//   - Tile `t`'s sum is `row_reduce`'s two xor shuffles, `(s_0 + s_2) + (s_1 + s_3)`; warp `w`'s
//     partial `part_w` is `((0 + T_w) + T_{w+4}) + …`, tiles ascending, as `row_sum(part, x², part)`.
//   - The four partials go to `sums` (row 0; the padding rows 0, as the tile's zero rows sum to),
//     and from there the one-row form's code: the warp-order add, `rms_inv_fused`, and pass 2 by
//     warp 0 on the owned tile, whose loads are issued first so they land during pass 1.
// So the bits are the one-row form's, and so the 16-row kernel's. Rows up to `ROW_TILES` tiles take
// it (`launch`); wider ones keep the one-row form.
//
// Built with `--use_fast_math` (build.rs): `rsqrtf` is the approximate reciprocal square root, as
// before. Shared memory: 4 warps × 2 stages × (two 8 KiB tiles and a 512-byte gain vector), and
// the four 64-byte sums, plus 1,024 bytes of alignment slack, set once by
// `scratchy_cuda_rms_norm_bf16_prepare` (in `PREPARE`). The plain and unit forms never touch the
// second tile.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// The launcher's return codes that are not a `cudaError_t` (which is non-negative).
// `src/kernels.rs` mirrors them.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::RMS_NORM_BF16_COLS`: `cols` is a multiple of it, so every row stride is a multiple of 16
// bytes, as the tensor maps need.
constexpr uint32_t VEC = 8;
constexpr int ROWS = 16;
constexpr int COLS = 256;
constexpr int WARPS = 4;
constexpr int STAGES = 2;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;
// The warpgroup's named barrier (0 is `__syncthreads`).
constexpr int SUMS_BARRIER = 1;

// The kernel's forms: the norm, the unit norm, and the two residual folds (#1049).
constexpr int NORM = 0;
constexpr int UNIT = 1;
constexpr int ADD_NORM = 2;
constexpr int NORM_ADD = 3;

using tile = st_bf<ROWS, COLS, false>;
using gain_vec = sv_bf<COLS>;
using sums_vec = sv_fl<ROWS>;
using x_gl = gl<bf16, 1, 1, -1, -1, tile>;
using g_gl = gl<bf16, 1, 1, 1, -1, gain_vec>;
// The row form's widest row, in 256-column tiles (17,408 columns).
constexpr int ROW_TILES = 68;
using acc_tile = rt_fl<ROWS, COLS>;
using bf16_tile = rt_bf<ROWS, COLS>;
using row_values = acc_tile::col_vec;
using col_values = acc_tile::row_vec;

struct rms_globals {
    x_gl x;
    g_gl gain;
    x_gl y;
    // `ADD_NORM`: `x`'s addend `b`, and the sum's output. `NORM_ADD`: the residual, read into the
    // `other` tile; `sum` is unused. The plain and unit forms read neither.
    x_gl other;
    x_gl sum;
    // The row form's views of row 0 of `x` and `other` as `[cols]` vectors (`gain`'s view).
    g_gl x_row;
    g_gl other_row;
    // The 256-column tiles of a row, the last one partial when `cols % 256 ≠ 0`.
    int col_tiles;
    float cols;
    float eps;
    float gain_offset;
};

constexpr size_t SMEM_USED =
    WARPS * STAGES * (2 * sizeof(tile) + sizeof(gain_vec)) + WARPS * sizeof(sums_vec);
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int RMS_SMEM_BYTES = int(SMEM_USED) + 1024;
static_assert(RMS_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

// `x` rounded to bf16 and widened back, in place: `warp::store`'s rounding (RNE, the
// `convertor<bf16_2, float2>` both use), kept in registers.
__device__ inline void round_bf16(acc_tile& x) {
    bf16_tile r;
    warp::copy(r, x);
    warp::copy(x, r);
}

template <int FORM>
__global__ __launch_bounds__(NUM_THREADS, 1) void rms_norm_bf16_ker(
    const __grid_constant__ rms_globals g) {
    constexpr bool GAIN = FORM != UNIT;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    tile(&x_smem)[WARPS][STAGES] = al.allocate<tile, WARPS, STAGES>();
    tile(&o_smem)[WARPS][STAGES] = al.allocate<tile, WARPS, STAGES>();
    gain_vec(&g_smem)[WARPS][STAGES] = al.allocate<gain_vec, WARPS, STAGES>();
    sums_vec(&sums)[WARPS] = al.allocate<sums_vec, WARPS>();
    __shared__ kittens::semaphore arrived[WARPS][STAGES];

    const int w = kittens::warpid();
    const int row_block = blockIdx.x;
    // This warp's tiles; its load `i` of `2 · mine` (pass 1's, then pass 2's) is in stage `i % 2`
    // at phase `(i / 2) % 2`.
    const int mine = (g.col_tiles - w + WARPS - 1) / WARPS;
    if (warp::laneid() == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_semaphore(arrived[w][s], 0, 1);
        }
    }
    __syncwarp();
    // Pass 1 loads `b` for the sum, pass 2 for the sum again (`ADD_NORM`); only pass 2 the
    // residual (`NORM_ADD`).
    auto issue = [&](int i) {
        const int stage = i % STAGES, t = w + (i % mine) * WARPS;
        const bool second = GAIN && i >= mine;
        const bool other = FORM == ADD_NORM || (FORM == NORM_ADD && i >= mine);
        coord<tile> at = {0, 0, row_block, t};
        warp::tma::expect_bytes(arrived[w][stage], sizeof(tile) + (other ? sizeof(tile) : 0) +
                                                       (second ? sizeof(gain_vec) : 0));
        warp::tma::load_async(x_smem[w][stage], g.x, at, arrived[w][stage]);
        if (other) {
            warp::tma::load_async(o_smem[w][stage], g.other, at, arrived[w][stage]);
        }
        if (second) {
            coord<gain_vec> gat = {0, 0, 0, t};
            warp::tma::load_async(g_smem[w][stage], g.gain, gat, arrived[w][stage]);
        }
    };

    // Pass 1: this warp's sums of squares.
    row_values part;
    warp::zero(part);
    if (mine > 0) {
        issue(0);
    }
    for (int i = 0; i < mine; i++) {
        const int stage = i % STAGES;
        // Pass 1 stores nothing but `ADD_NORM`'s sum: the next load may take the other stage once
        // its tiles were read (the previous iteration's `warp::load`, which every lane finished at
        // this barrier) and the sum's store from it was read out.
        if constexpr (FORM == ADD_NORM) {
            warp::tma::store_async_read_wait();
        }
        __syncwarp();
        issue(i + 1);  // pass 1's next tile, or pass 2's first (`i + 1 == mine`)
        wait(arrived[w][stage], (i / STAGES) % 2);
        acc_tile x;
        warp::load(x, x_smem[w][stage]);
        if constexpr (FORM == ADD_NORM) {
            acc_tile b;
            warp::load(b, o_smem[w][stage]);
            warp::add(x, x, b);
            round_bf16(x);
            warp::store(x_smem[w][stage], x);
            __syncwarp();
            coord<tile> at = {0, 0, row_block, w + i * WARPS};
            warp::tma::store_async(g.sum, x_smem[w][stage], at);
        }
        warp::mul(x, x, x);
        warp::row_sum(part, x, part);
    }
    warp::store(sums[w], part);
    warpgroup::sync(SUMS_BARRIER);
    // Every warp adds the four partial sums in warp order, so all four hold the same bits.
    row_values total, other;
    warp::load(total, sums[0]);
    for (int v = 1; v < WARPS; v++) {
        warp::load(other, sums[v]);
        warp::add(total, total, other);
    }
    row_values inv;
    warp::div(inv, total, g.cols);
    warp::bin_op<scratchy::ops::rms_inv>(inv, inv, g.eps);

    // Pass 2: the output of this warp's tiles.
    for (int i = mine; i < 2 * mine; i++) {
        const int stage = i % STAGES, t = w + (i - mine) * WARPS;
        // The other stage's previous store must have read its tile before the next load
        // overwrites it.
        warp::tma::store_async_read_wait();
        __syncwarp();
        if (i + 1 < 2 * mine) {
            issue(i + 1);
        }
        wait(arrived[w][stage], (i / STAGES) % 2);
        acc_tile x;
        warp::load(x, x_smem[w][stage]);
        if constexpr (FORM == ADD_NORM) {
            acc_tile b;
            warp::load(b, o_smem[w][stage]);
            warp::add(x, x, b);
            round_bf16(x);
        }
        warp::mul_row(x, x, inv);
        if constexpr (GAIN) {
            col_values gain;
            warp::load(gain, g_smem[w][stage]);
            warp::add(gain, gain, g.gain_offset);
            warp::mul_col(x, x, gain);
        }
        if constexpr (FORM == NORM_ADD) {
            acc_tile r;
            round_bf16(x);
            warp::load(r, o_smem[w][stage]);
            warp::add(x, x, r);
        }
        warp::store(x_smem[w][stage], x);
        __syncwarp();
        coord<tile> at = {0, 0, row_block, t};
        warp::tma::store_async(g.y, x_smem[w][stage], at);
    }
    warp::tma::store_async_wait();
}

// `rows == 1`: CTA `blockIdx.x` owns column tile `blockIdx.x`. Pass 1 is the 16-row kernel's, on
// the one row block: every CTA computes the whole row's `inv` in the same order. Pass 2 is the
// 16-row kernel's on the owned tile only, by warp 0.
template <int FORM>
__global__ __launch_bounds__(NUM_THREADS, 1) void rms_norm_one_row_ker(
    const __grid_constant__ rms_globals g) {
    constexpr bool GAIN = FORM != UNIT;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    tile(&x_smem)[WARPS][STAGES] = al.allocate<tile, WARPS, STAGES>();
    tile(&o_smem)[WARPS][STAGES] = al.allocate<tile, WARPS, STAGES>();
    gain_vec(&g_smem)[WARPS][STAGES] = al.allocate<gain_vec, WARPS, STAGES>();
    sums_vec(&sums)[WARPS] = al.allocate<sums_vec, WARPS>();
    __shared__ kittens::semaphore arrived[WARPS][STAGES];
    __shared__ kittens::semaphore owned;

    const int w = kittens::warpid();
    const int t_own = blockIdx.x;
    const int mine = (g.col_tiles - w + WARPS - 1) / WARPS;
    if (warp::laneid() == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_semaphore(arrived[w][s], 0, 1);
        }
        if (w == 0) {
            init_semaphore(owned, 0, 1);
        }
    }
    __syncwarp();
    // Pass 1's load `i` of this warp's `mine` tiles, the 16-row kernel's `issue` at `i < mine`.
    auto issue = [&](int i) {
        const int stage = i % STAGES, t = w + i * WARPS;
        coord<tile> at = {0, 0, 0, t};
        warp::tma::expect_bytes(arrived[w][stage],
                                sizeof(tile) + (FORM == ADD_NORM ? sizeof(tile) : 0));
        warp::tma::load_async(x_smem[w][stage], g.x, at, arrived[w][stage]);
        if constexpr (FORM == ADD_NORM) {
            warp::tma::load_async(o_smem[w][stage], g.other, at, arrived[w][stage]);
        }
    };

    // Pass 1: this warp's sums of squares, the 16-row kernel's loop with no sum store.
    row_values part;
    warp::zero(part);
    if (mine > 0) {
        issue(0);
    }
    for (int i = 0; i < mine; i++) {
        const int stage = i % STAGES;
        __syncwarp();
        if (i + 1 < mine) {
            issue(i + 1);
        }
        wait(arrived[w][stage], (i / STAGES) % 2);
        acc_tile x;
        warp::load(x, x_smem[w][stage]);
        if constexpr (FORM == ADD_NORM) {
            acc_tile b;
            warp::load(b, o_smem[w][stage]);
            warp::add(x, x, b);
            round_bf16(x);
        }
        warp::mul(x, x, x);
        warp::row_sum(part, x, part);
    }
    warp::store(sums[w], part);
    warpgroup::sync(SUMS_BARRIER);
    // The four partial sums in warp order, as the 16-row kernel; then `inv`.
    row_values total, other;
    warp::load(total, sums[0]);
    for (int v = 1; v < WARPS; v++) {
        warp::load(other, sums[v]);
        warp::add(total, total, other);
    }
    // `inv` as the 16-row kernel's SASS forms it in every form: MUFU.RCP of `cols`, one FFMA
    // `total · rcp + eps`, MUFU.RSQ (`scratchy::ops::rms_inv_fused`). Left to ptxas, this kernel's
    // `ADD_NORM` instance kept the bottom packed row's product and `+ eps` apart (the H100 SASS at
    // 6b6a32f7 and a5b097c9: an FMUL, then an FADD past the warps' branch), even without an early
    // return; the explicit FFMA makes both rows that FFMA.
    row_values rcp, inv;
    warp::one(rcp);
    warp::div(rcp, rcp, g.cols);
    scratchy::ops::rv_ternary_map<scratchy::ops::rms_inv_fused>(inv, total, rcp, g.eps);
    // Pass 2 is warp 0's alone.
    if (w == 0) {
        // Pass 2 on the owned tile only: the 16-row kernel's pass-2 body, on stage 0's tiles (pass 1
        // waited on every load it issued, so none is in flight) and its own semaphore.
        __syncwarp();
        {
            coord<tile> at = {0, 0, 0, t_own};
            constexpr bool OTHER = FORM == ADD_NORM || FORM == NORM_ADD;
            warp::tma::expect_bytes(owned, sizeof(tile) + (OTHER ? sizeof(tile) : 0) +
                                               (GAIN ? sizeof(gain_vec) : 0));
            warp::tma::load_async(x_smem[0][0], g.x, at, owned);
            if constexpr (OTHER) {
                warp::tma::load_async(o_smem[0][0], g.other, at, owned);
            }
            if constexpr (GAIN) {
                coord<gain_vec> gat = {0, 0, 0, t_own};
                warp::tma::load_async(g_smem[0][0], g.gain, gat, owned);
            }
        }
        wait(owned, 0);
        acc_tile x;
        warp::load(x, x_smem[0][0]);
        if constexpr (FORM == ADD_NORM) {
            acc_tile b;
            warp::load(b, o_smem[0][0]);
            warp::add(x, x, b);
            round_bf16(x);
            // The owner stores the tile's sum, as the 16-row kernel's pass 1 does: the same rounded
            // bits. Through its own tile, so pass 2's `x` stays in registers.
            warp::store(o_smem[0][1], x);
            __syncwarp();
            coord<tile> sat = {0, 0, 0, t_own};
            warp::tma::store_async(g.sum, o_smem[0][1], sat);
        }
        warp::mul_row(x, x, inv);
        if constexpr (GAIN) {
            col_values gain;
            warp::load(gain, g_smem[0][0]);
            warp::add(gain, gain, g.gain_offset);
            warp::mul_col(x, x, gain);
        }
        if constexpr (FORM == NORM_ADD) {
            acc_tile r;
            round_bf16(x);
            warp::load(r, o_smem[0][0]);
            warp::add(x, x, r);
        }
        warp::store(x_smem[0][1], x);
        __syncwarp();
        coord<tile> at = {0, 0, 0, t_own};
        warp::tma::store_async(g.y, x_smem[0][1], at);
        warp::tma::store_async_wait();
    }
}

// `rows == 1`, the row form (N1): CTA `blockIdx.x` owns column tile `blockIdx.x`; pass 1's sums on
// the row's vectors in the 16-row kernel's order; then the one-row form's tail and pass 2.
template <int FORM>
__global__ __launch_bounds__(NUM_THREADS, 1) void rms_norm_row_ker(
    const __grid_constant__ rms_globals g) {
    constexpr bool GAIN = FORM != UNIT;
    constexpr bool OTHER = FORM == ADD_NORM || FORM == NORM_ADD;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    // Pass 2's tiles: [0] the owned tile's `x` (and `other`), [1] the output's (and the sum's).
    tile(&x_smem)[STAGES] = al.allocate<tile, STAGES>();
    tile(&o_smem)[STAGES] = al.allocate<tile, STAGES>();
    gain_vec& g_smem = al.allocate<gain_vec>();
    gain_vec(&x_row)[ROW_TILES] = al.allocate<gain_vec, ROW_TILES>();
    gain_vec(&b_row)[ROW_TILES] = al.allocate<gain_vec, ROW_TILES>();
    sums_vec(&sums)[WARPS] = al.allocate<sums_vec, WARPS>();
    // Each lane chain's sum, and each tile's (`ROW_TILES` rounded up to 16).
    __shared__ sv_fl<4 * ROW_TILES> lane_s;
    __shared__ sv_fl<80> tile_s;
    __shared__ kittens::semaphore row_arrived, owned;

    const int w = kittens::warpid();
    const int tid = int(threadIdx.x);
    const int t_own = blockIdx.x;
    const int tiles = g.col_tiles;
    if (tid == 0) {
        init_semaphore(row_arrived, 0, 1);
        init_semaphore(owned, 0, 1);
    }
    __syncthreads();
    if (w == 0) {
        // The row's vectors, then pass 2's owned tiles.
        warp::tma::expect_bytes(row_arrived,
                                uint32_t(tiles * sizeof(gain_vec) * (FORM == ADD_NORM ? 2 : 1)));
        for (int t = 0; t < tiles; t++) {
            warp::tma::load_async(x_row[t], g.x_row, {0, 0, 0, t}, row_arrived);
            if constexpr (FORM == ADD_NORM) {
                warp::tma::load_async(b_row[t], g.other_row, {0, 0, 0, t}, row_arrived);
            }
        }
        coord<tile> at = {0, 0, 0, t_own};
        warp::tma::expect_bytes(owned, sizeof(tile) + (OTHER ? sizeof(tile) : 0) +
                                           (GAIN ? sizeof(gain_vec) : 0));
        warp::tma::load_async(x_smem[0], g.x, at, owned);
        if constexpr (OTHER) {
            warp::tma::load_async(o_smem[0], g.other, at, owned);
        }
        if constexpr (GAIN) {
            coord<gain_vec> gat = {0, 0, 0, t_own};
            warp::tma::load_async(g_smem, g.gain, gat, owned);
        }
    }
    wait(row_arrived, 0);

    // Pass 1, `row_reduce`'s lane chains: thread `(t, q)`'s float2 sum of its squares.
    for (int i = tid; i < 4 * tiles; i += NUM_THREADS) {
        const int t = i / 4, q = i % 4;
        // The element pair at column `c` of tile `t`, as `warp::load` widens it (and `ADD_NORM`'s
        // `warp::add` then `round_bf16`), squared by `warp::mul`'s op.
        const auto square = [&](int c) {
            float2 v = base_types::convertor<float2, bf16_2>::convert(
                bf16_2{x_row[t][c], x_row[t][c + 1]});
            if constexpr (FORM == ADD_NORM) {
                const float2 b = base_types::convertor<float2, bf16_2>::convert(
                    bf16_2{b_row[t][c], b_row[t][c + 1]});
                v = base_types::convertor<float2, bf16_2>::convert(
                    base_types::convertor<bf16_2, float2>::convert(base_ops::sum::op<float2>(v, b)));
            }
            return base_ops::mul::op<float2>(v, v);
        };
        float2 top = base_ops::sum::op<float2>(square(2 * q), square(8 + 2 * q));
#pragma unroll
        for (int j = 1; j < COLS / 16; j++) {
            top = base_ops::sum::op<float2>(top, square(16 * j + 2 * q));
            top = base_ops::sum::op<float2>(top, square(16 * j + 8 + 2 * q));
        }
        lane_s[i] = base_ops::sum::op<float>(top.x, top.y);
    }
    __syncthreads();
    // Each tile's sum, `row_reduce`'s xor-2 then xor-1 shuffle adds.
    for (int t = tid; t < tiles; t += NUM_THREADS) {
        const float a = base_ops::sum::op<float>(lane_s[4 * t], lane_s[4 * t + 2]);
        const float b = base_ops::sum::op<float>(lane_s[4 * t + 1], lane_s[4 * t + 3]);
        tile_s[t] = base_ops::sum::op<float>(a, b);
    }
    __syncthreads();
    // Warp `v`'s partial over its tiles `v, v + 4, …`, ascending, from `warp::zero`; the padding
    // rows' sums are 0.
    for (int i = tid; i < WARPS * ROWS; i += NUM_THREADS) {
        const int v = i / ROWS, r = i % ROWS;
        float part = 0.0f;
        if (r == 0) {
            for (int t = v; t < tiles; t += WARPS) {
                part = base_ops::sum::op<float>(part, tile_s[t]);
            }
        }
        sums[v][r] = part;
    }
    __syncthreads();
    // The one-row form's tail: the four partial sums in warp order, then `inv`.
    row_values total, other;
    warp::load(total, sums[0]);
    for (int v = 1; v < WARPS; v++) {
        warp::load(other, sums[v]);
        warp::add(total, total, other);
    }
    row_values rcp, inv;
    warp::one(rcp);
    warp::div(rcp, rcp, g.cols);
    scratchy::ops::rv_ternary_map<scratchy::ops::rms_inv_fused>(inv, total, rcp, g.eps);
    // Pass 2, warp 0's on the owned tile: the one-row form's.
    if (w == 0) {
        wait(owned, 0);
        acc_tile x;
        warp::load(x, x_smem[0]);
        if constexpr (FORM == ADD_NORM) {
            acc_tile b;
            warp::load(b, o_smem[0]);
            warp::add(x, x, b);
            round_bf16(x);
            warp::store(o_smem[1], x);
            __syncwarp();
            coord<tile> sat = {0, 0, 0, t_own};
            warp::tma::store_async(g.sum, o_smem[1], sat);
        }
        warp::mul_row(x, x, inv);
        if constexpr (GAIN) {
            col_values gain;
            warp::load(gain, g_smem);
            warp::add(gain, gain, g.gain_offset);
            warp::mul_col(x, x, gain);
        }
        if constexpr (FORM == NORM_ADD) {
            acc_tile r;
            round_bf16(x);
            warp::load(r, o_smem[0]);
            warp::add(x, x, r);
        }
        warp::store(x_smem[1], x);
        __syncwarp();
        coord<tile> at = {0, 0, 0, t_own};
        warp::tma::store_async(g.y, x_smem[1], at);
        warp::tma::store_async_wait();
    }
}

constexpr size_t ROW_SMEM_USED = 2 * STAGES * sizeof(tile) + sizeof(gain_vec) +
                                 2 * ROW_TILES * sizeof(gain_vec) + WARPS * sizeof(sums_vec);
constexpr int ROW_SMEM_BYTES = int(ROW_SMEM_USED) + 1024;
static_assert(ROW_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

// Build the views and launch `rms_norm_bf16_ker<FORM>`, or at `rows == 1` the one-row form
// `rms_norm_one_row_ker<FORM>` (`col_tiles` CTAs, the same bits). Can throw (a tensor-map
// encode); the caller catches. `gain` is never read under `UNIT`, `other` only by the residual
// folds and `sum` only by `ADD_NORM`, but every view is still built (the callers give a valid
// address).
//
// `one_row` (test support only): at `rows == 1`, 0 the one-row form, 1 the row form (the launcher's
// at up to `ROW_TILES` tiles).
template <int FORM>
int launch(const void* x, const void* gain, void* y, const void* other, void* sum, uint32_t rows,
           uint32_t cols, float eps, float gain_offset, cudaStream_t stream, int one_row = -1) {
    const auto view = [&](const void* p) {
        // Building a `gl` encodes its tensor map into host memory; nothing is allocated on the
        // device. Every row stride is `2·cols` bytes, a multiple of 16.
        return x_gl{static_cast<bf16*>(const_cast<void*>(p)), nullptr, nullptr, size_t(rows),
                    size_t(cols)};
    };
    g_gl gg{static_cast<bf16*>(const_cast<void*>(gain)), nullptr, nullptr, nullptr, size_t(cols)};
    const int col_tiles = int((cols + uint32_t(COLS) - 1) / uint32_t(COLS));
    const auto row = [&](const void* p) {
        return g_gl{static_cast<bf16*>(const_cast<void*>(p)), nullptr, nullptr, nullptr,
                    size_t(cols)};
    };
    rms_globals G{view(x),   gg,  view(y),          view(other), view(sum), row(x), row(other),
                  col_tiles, float(cols), eps, gain_offset};
    void* args[] = {&G};
    if (rows == 1 && (one_row == 1 || (one_row == -1 && col_tiles <= ROW_TILES))) {
        const dim3 grid{uint32_t(col_tiles)}, block{NUM_THREADS};
        return static_cast<int>(cudaLaunchKernel(rms_norm_row_ker<FORM>, grid, block, args,
                                                 ROW_SMEM_BYTES, stream));
    }
    if (rows == 1) {
        // Braces: `grid(uint32_t(col_tiles))` parses as a function declaration.
        const dim3 grid{uint32_t(col_tiles)}, block{NUM_THREADS};
        return static_cast<int>(cudaLaunchKernel(rms_norm_one_row_ker<FORM>, grid, block, args,
                                                 RMS_SMEM_BYTES, stream));
    }
    const dim3 grid((rows + uint32_t(ROWS) - 1) / uint32_t(ROWS)), block(NUM_THREADS);
    return static_cast<int>(
        cudaLaunchKernel(rms_norm_bf16_ker<FORM>, grid, block, args, RMS_SMEM_BYTES, stream));
}

bool refused(uint32_t rows, uint32_t cols) {
    return rows == 0 || cols == 0 || cols % VEC != 0 ||
           uint64_t(rows) * uint64_t(cols) > uint64_t(INT_MAX);
}

}  // namespace

// The kernel's one-time setup: allow the four forms, and their one-row instances, `RMS_SMEM_BYTES`
// of dynamic shared memory on the current device (the default limit is 48 KB). Run once per device
// before any launch (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch.
//
// 0 = done; any other value = the first cudaError_t of setting the attribute.
extern "C" int scratchy_cuda_rms_norm_bf16_prepare(void) {
    for (auto ker : {rms_norm_bf16_ker<NORM>, rms_norm_bf16_ker<UNIT>, rms_norm_bf16_ker<ADD_NORM>,
                     rms_norm_bf16_ker<NORM_ADD>, rms_norm_one_row_ker<NORM>,
                     rms_norm_one_row_ker<UNIT>, rms_norm_one_row_ker<ADD_NORM>,
                     rms_norm_one_row_ker<NORM_ADD>}) {
        const cudaError_t e =
            cudaFuncSetAttribute(ker, cudaFuncAttributeMaxDynamicSharedMemorySize, RMS_SMEM_BYTES);
        if (e != cudaSuccess) {
            return static_cast<int>(e);
        }
    }
    for (auto ker : {rms_norm_row_ker<NORM>, rms_norm_row_ker<UNIT>, rms_norm_row_ker<ADD_NORM>,
                     rms_norm_row_ker<NORM_ADD>}) {
        const cudaError_t e =
            cudaFuncSetAttribute(ker, cudaFuncAttributeMaxDynamicSharedMemorySize, ROW_SMEM_BYTES);
        if (e != cudaSuccess) {
            return static_cast<int>(e);
        }
    }
    return 0;
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value
// = the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `rows` and `cols` must be non-zero, `cols` a multiple of 8, and `rows · cols` at most INT_MAX;
// all three are checked before any pointer is touched. `x` and `y` must be 16-byte-aligned device
// pointers to `rows·cols` bf16 and `gain` to `cols` bf16, `y` overlapping neither input.
// `scratchy_cuda_rms_norm_bf16_prepare` must have succeeded on the current device. The launch is
// asynchronous on `stream`; the caller owns it and synchronizes.
extern "C" int scratchy_cuda_rms_norm_bf16(const void* x, const void* gain, void* y, uint32_t rows,
                                           uint32_t cols, float eps, float gain_offset,
                                           void* stream /* cudaStream_t; null = default stream */) {
    if (refused(rows, cols)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        // `other` and `sum` are never read by this form; `x` is a valid address for both views.
        return launch<NORM>(x, gain, y, x, y, rows, cols, eps, gain_offset,
                            static_cast<cudaStream_t>(stream));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// `y = x / sqrt(mean(x²) + eps)`: the norm with no gain (`rms_norm_bf16_ker<true>`). Return codes,
// shape rule and setup as `scratchy_cuda_rms_norm_bf16`'s; `x` and `y` must be 16-byte-aligned
// device pointers to `rows·cols` bf16, `y` not overlapping `x`.
extern "C" int scratchy_cuda_rms_norm_unit_bf16(const void* x, void* y, uint32_t rows,
                                                uint32_t cols, float eps,
                                                void* stream /* cudaStream_t; null = default */) {
    if (refused(rows, cols)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        // The gain view still encodes a tensor map, so it is given `x`, a valid 16-byte-aligned
        // address of at least `cols` bf16; `<UNIT>` never reads it, nor `other` and `sum`.
        return launch<UNIT>(x, x, y, x, y, rows, cols, eps, 0.0f,
                            static_cast<cudaStream_t>(stream));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// `sum = a + b` (one f32 add, rounded once to bf16) and `y = sum / sqrt(mean(sum²) + eps) ·
// (gain_offset + gain)` (#1049, `rms_norm_bf16_ker<ADD_NORM>`): the bits of
// `scratchy_cuda_add_bf16` then `scratchy_cuda_rms_norm_bf16` on its output. Return codes, shape
// rule and setup as `scratchy_cuda_rms_norm_bf16`'s; `a`, `b`, `sum` and `y` must be
// 16-byte-aligned device pointers to `rows·cols` bf16 and `gain` to `cols` bf16, `sum` and `y`
// overlapping neither each other nor an input.
extern "C" int scratchy_cuda_add_rms_norm_bf16(const void* a, const void* b, const void* gain,
                                               void* sum, void* y, uint32_t rows, uint32_t cols,
                                               float eps, float gain_offset,
                                               void* stream /* cudaStream_t; null = default */) {
    if (refused(rows, cols)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        return launch<ADD_NORM>(a, gain, y, b, sum, rows, cols, eps, gain_offset,
                                static_cast<cudaStream_t>(stream));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// `out = residual + bf16(x / sqrt(mean(x²) + eps) · (gain_offset + gain))`, the sum in f32 and
// rounded once (#1049, `rms_norm_bf16_ker<NORM_ADD>`): the bits of `scratchy_cuda_rms_norm_bf16`
// then `scratchy_cuda_add_bf16` of its output and `residual`. Return codes, shape rule and setup
// as `scratchy_cuda_rms_norm_bf16`'s; `x`, `residual` and `out` must be 16-byte-aligned device
// pointers to `rows·cols` bf16 and `gain` to `cols` bf16, `out` overlapping neither input.
extern "C" int scratchy_cuda_norm_add_bf16(const void* x, const void* gain, const void* residual,
                                           void* out, uint32_t rows, uint32_t cols, float eps,
                                           float gain_offset,
                                           void* stream /* cudaStream_t; null = default */) {
    if (refused(rows, cols)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        // `sum` is never written by this form; `out` is a valid address for its view.
        return launch<NORM_ADD>(x, gain, out, residual, out, rows, cols, eps, gain_offset,
                                static_cast<cudaStream_t>(stream));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// Test support (`tests/row_ops.rs`, FP8 S8 N1): one row of norm form `form` (0 the norm, 1 the
// unit norm, 2 `ADD_NORM`, 3 `NORM_ADD`) at `variant` 0, the one-row form, or 1, the row form,
// the same bits. Every pointer a valid 16-byte-aligned device address of `cols` bf16 (`other`
// and `sum` read or written only by the forms that take them). Return codes as
// `scratchy_cuda_rms_norm_bf16`'s; refused also when `form` or `variant` is another value, or
// `cols` is past `ROW_TILES` tiles.
extern "C" int scratchy_cuda_rms_norm_one_row_variant(const void* x, const void* gain, void* y,
                                                      const void* other, void* sum, uint32_t cols,
                                                      float eps, float gain_offset, uint32_t form,
                                                      uint32_t variant,
                                                      void* stream /* cudaStream_t */) {
    if (refused(1, cols) || form > 3 || variant > 1 ||
        cols > uint32_t(ROW_TILES) * uint32_t(COLS)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        const int v = int(variant);
        switch (form) {
            case 0:
                return launch<NORM>(x, gain, y, other, sum, 1, cols, eps, gain_offset, s, v);
            case 1:
                return launch<UNIT>(x, gain, y, other, sum, 1, cols, eps, gain_offset, s, v);
            case 2:
                return launch<ADD_NORM>(x, gain, y, other, sum, 1, cols, eps, gain_offset, s, v);
            default:
                return launch<NORM_ADD>(x, gain, y, other, sum, 1, cols, eps, gain_offset, s, v);
        }
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
