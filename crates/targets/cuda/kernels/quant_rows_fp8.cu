// SPDX-License-Identifier: Apache-2.0
//
// The per-token dynamic e4m3 quantization of an fp8 GEMM's bf16 input for the H100 (FP8 S3,
// `step::QuantRowsFp8`): for each row `r` of `x [rows, cols]` bf16,
//
//     scale[r] = max(max_c |x[r, c]| / 448, 1 / (448 · 512))
//     q[r, c]  = e4m3_satfinite(clamp(x[r, c] / scale[r], −448, 448))
//
// `q [rows, cols]` e4m3 and `scale [rows]` f32, both in the tape's quant scratch. This is vLLM
// v0.31.0's `dynamic_per_token_scaled_fp8_quant_kernel_strided`
// (`csrc/libtorch_stable/quantization/w8a8/fp8/common.cu`): its scale (`token_scale`, with
// `min_scaling_factor`) and its element conversion `scaled_fp8_conversion<false>`, which DIVIDES
// by the scale. Both operations are op structs (`scratchy::ops::fp8_scale`, `fp8_quant_pre`);
// the cast is TK's `warp::copy` of an `rt_fl` to an `rt_fp8e4m3` (`__nv_fp8x4_e4m3`,
// round-to-nearest-even, saturating to finite).
//
// `amax` is exact under any reduction order (a max), so the same bf16 row gives the same scale
// and the same bytes whichever kernel computes them: the FP8 GEMV's prologue (S4) must equal this.
//
// ThunderKittens' 16-row tile, `rms_norm_bf16.cu`'s structure (PLAN-tk100 §2 S2): one warp per 16
// rows, 4 warps per CTA, so `rows / 16` warps and a wide grid. Each warp walks its rows'
// 256-column tiles in two passes, each tile TMA-loaded by its lane 0 into a plain
// `st_bf<16, 256, false>` through a 2-stage ring:
//   1. `warp::load` into `rt_fl<16, 256>` (exact), `warp::abs`, `warp::row_max` into the warp's
//      `col_vec` across its tiles; then `fp8_scale` on it, stored to `scale` (`warp::store` of the
//      column vector to its `gl`).
//   2. The tile re-read, `warp::row_map<fp8_quant_pre>` by the scale, `warp::copy` to
//      `rt_fp8e4m3<16, 256>`, `warp::store` into a plain `st_fp8e4m3<16, 256, false>`, TMA-stored.
// Every tile is plain: a box past the array's columns (`cols % 256 ≠ 0`) reads zeros and stores
// nothing (TK-1's R1 gate, `tests/tma_bounds.rs`). A zero adds nothing to a max of absolute values,
// so the partial tile's scale is the row's. `rows` is a multiple of 16, so no box crosses the rows.
//
// Shared memory: 4 warps × 2 stages × (an 8 KiB bf16 tile and a 4 KiB e4m3 tile), plus 1,024 bytes
// of alignment slack, set once by `scratchy_cuda_quant_rows_fp8_prepare` (in `PREPARE`).

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

// `step::GEMM_FP8_K_MULTIPLE`: `cols` is a multiple of it, so every row stride (`cols` e4m3 bytes,
// `2·cols` bf16 bytes) is a multiple of 16, as the tensor maps need, and the GEMM's K steps tile it.
constexpr uint32_t COL_MULTIPLE = 64;
// One tile's rows: `rows` is a multiple of it.
constexpr int ROWS = 16;
constexpr int COLS = 256;
constexpr int WARPS = 4;
constexpr int STAGES = 2;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;

using x_tile = st_bf<ROWS, COLS, false>;
using q_tile = st_fp8e4m3<ROWS, COLS, false>;
using x_gl = gl<bf16, 1, 1, -1, -1, x_tile>;
using q_gl = gl<fp8e4m3, 1, 1, -1, -1, q_tile>;
// `[1, rows]` f32: written by `warp::store` of a column vector, no tensor map.
using scale_gl = gl<float, 1, 1, 1, -1>;
using acc_tile = rt_fl<ROWS, COLS>;
using fp8_tile = rt_fp8e4m3<ROWS, COLS>;
using row_values = acc_tile::col_vec;

struct quant_globals {
    x_gl x;
    q_gl q;
    scale_gl scale;
    // The rows' 16-row blocks, and the 256-column tiles of a row (the last one partial when
    // `cols % 256 ≠ 0`).
    int row_blocks;
    int col_tiles;
};

constexpr size_t SMEM_USED = WARPS * STAGES * (sizeof(x_tile) + sizeof(q_tile));
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int QUANT_SMEM_BYTES = int(SMEM_USED) + 1024;
static_assert(QUANT_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

__global__ __launch_bounds__(NUM_THREADS, 1) void quant_rows_fp8_ker(
    const __grid_constant__ quant_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    x_tile(&x_smem)[WARPS][STAGES] = al.allocate<x_tile, WARPS, STAGES>();
    q_tile(&q_smem)[WARPS][STAGES] = al.allocate<q_tile, WARPS, STAGES>();
    __shared__ kittens::semaphore arrived[WARPS][STAGES];

    const int w = kittens::warpid();
    const int row_block = blockIdx.x * WARPS + w;
    // A warp past the last block (the last CTA's, when `rows / 16` is not a multiple of 4) has
    // nothing to do: every warp works alone, so it may leave.
    if (row_block >= g.row_blocks) {
        return;
    }
    const int tiles = g.col_tiles;
    if (warp::laneid() == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_semaphore(arrived[w][s], 0, 1);
        }
    }
    __syncwarp();
    // Load `i` of `2 · tiles` (pass 1's, then pass 2's): tile `i % tiles`, stage `i % 2`, phase
    // `(i / 2) % 2`.
    auto issue = [&](int i) {
        const int stage = i % STAGES;
        coord<x_tile> at = {0, 0, row_block, i % tiles};
        warp::tma::expect_bytes(arrived[w][stage], sizeof(x_tile));
        warp::tma::load_async(x_smem[w][stage], g.x, at, arrived[w][stage]);
    };

    // Pass 1: the rows' max of absolute values, then their scales.
    row_values amax;
    warp::zero(amax);
    issue(0);
    for (int i = 0; i < tiles; i++) {
        const int stage = i % STAGES;
        // The other stage's tile was read by the previous iteration's `warp::load`, which every
        // lane finished at this barrier.
        __syncwarp();
        issue(i + 1);  // pass 1's next tile, or pass 2's first (`i + 1 == tiles`)
        wait(arrived[w][stage], (i / STAGES) % 2);
        acc_tile x;
        warp::load(x, x_smem[w][stage]);
        warp::abs(x, x);
        warp::row_max(amax, x, amax);
    }
    row_values scale;
    warp::unary_op<scratchy::ops::fp8_scale>(scale, amax);
    warp::store(g.scale, scale, {0, 0, 0, row_block});

    // Pass 2: each tile divided by its rows' scales, cast to e4m3 and stored.
    for (int i = tiles; i < 2 * tiles; i++) {
        const int stage = i % STAGES;
        // The store from this stage's e4m3 tile two iterations ago must have read it before this
        // one overwrites it; the other stage's bf16 tile was read last iteration.
        warp::tma::store_async_read_wait<1>();
        __syncwarp();
        if (i + 1 < 2 * tiles) {
            issue(i + 1);
        }
        wait(arrived[w][stage], (i / STAGES) % 2);
        acc_tile x;
        warp::load(x, x_smem[w][stage]);
        warp::row_map<scratchy::ops::fp8_quant_pre>(x, x, scale);
        fp8_tile q;
        warp::copy(q, x);
        warp::store(q_smem[w][stage], q);
        __syncwarp();
        coord<q_tile> at = {0, 0, row_block, i - tiles};
        warp::tma::store_async(g.q, q_smem[w][stage], at);
    }
    warp::tma::store_async_wait();
}

// ── F3: eight warps a 16-row block (FP8 S8's TTFT plan) ──
//
// `quant_rows_fp8_ker` gives each 16-row block one warp, which walks every tile of the row twice:
// ~165 µs a launch at 5,120 columns on the H100 (W4's prefill traces), whatever the rows, and its
// grid is `rows / 64` CTAs. `quant_rows_fp8_wide_ker` gives each block a CTA of 8 warps: warp `w`
// takes the block's tiles `w, w + 8, …` through the same 2-stage ring and the same pass-1 ops,
// building its rows' `row_max` over its tiles; the 8 warps' maxima meet in shared memory and every
// warp takes their max (a max of absolute values is exact in any order, so it is the one warp's
// `amax`); then `fp8_scale` and the same pass 2 on its tiles, with `fp8_quant_pre_nz`, which skips
// the division of a ±0 (the value is the same; the padding rows of a prefill bucket stop taking
// `__fdiv_rn`'s slow path). So the bytes and the scales are `quant_rows_fp8_ker`'s.
// A narrow row leaves most of 8 warps idle (512 columns: 2 tiles, ~3× slower than the one-warp
// form at 16,384 rows), so the kernel is generic over `BW`, the warps of a block: a CTA of 8 warps
// holds `8 / BW` blocks, and the launcher takes the smallest of 1, 2, 4, 8 that is at least the
// row's tiles (8 past that, `wide_warps`). Still, a narrow row's many blocks are the one-warp
// form's: its 96 KiB CTAs fit two an SM, these 192 KiB ones one, so past 512 blocks (the experts'
// 16,384 pair rows of a 2,048-token prefill) the one-warp form runs in one wave where this one
// needs two (on the H100, 16,384 × 512: 28 µs against 35). The launcher takes it there
// (`narrow_many`, from `rows` and `cols`, constants of each step).
// Shared memory: 8 warps × 2 stages × (an 8 KiB bf16 tile and a 4 KiB e4m3 tile), plus slack.

constexpr int WIDE_WARPS = 8;
constexpr int WIDE_THREADS = WIDE_WARPS * kittens::WARP_THREADS;
constexpr size_t WIDE_SMEM_USED = WIDE_WARPS * STAGES * (sizeof(x_tile) + sizeof(q_tile));
constexpr int WIDE_SMEM_BYTES = int(WIDE_SMEM_USED) + 1024;
static_assert(WIDE_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

// Whether `quant_rows_fp8_ker` takes a shape: a row of at most 4 tiles in more than 512 blocks.
bool narrow_many(uint32_t row_blocks, int col_tiles) {
    return col_tiles <= 4 && row_blocks > 512;
}

// The warps of a block for a row of `col_tiles` tiles.
int wide_warps(int col_tiles) {
    return col_tiles <= 1 ? 1 : col_tiles <= 2 ? 2 : col_tiles <= 4 ? 4 : WIDE_WARPS;
}

template <int BW>
__global__ __launch_bounds__(WIDE_THREADS, 1) void quant_rows_fp8_wide_ker(
    const __grid_constant__ quant_globals g) {
    static_assert(WIDE_WARPS % BW == 0, "whole blocks a CTA");
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    x_tile(&x_smem)[WIDE_WARPS][STAGES] = al.allocate<x_tile, WIDE_WARPS, STAGES>();
    q_tile(&q_smem)[WIDE_WARPS][STAGES] = al.allocate<q_tile, WIDE_WARPS, STAGES>();
    __shared__ kittens::semaphore arrived[WIDE_WARPS][STAGES];
    __shared__ sv_fl<ROWS> amax_s[WIDE_WARPS];

    const int w = kittens::warpid();
    // This warp's block of the CTA's `8 / BW`, its index `wb` in the block's warps.
    const int first = w / BW * BW, wb = w - first;
    const int row_block = blockIdx.x * (WIDE_WARPS / BW) + w / BW;
    // A block past the last one (the last CTA's) has no tiles, but its warps meet the barrier.
    const bool live = row_block < g.row_blocks;
    // This warp's tiles of the row: `wb + BW i` for `i < mine` (none when the row has fewer).
    const int mine = live ? (g.col_tiles - wb + BW - 1) / BW : 0;
    if (warp::laneid() == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_semaphore(arrived[w][s], 0, 1);
        }
    }
    __syncwarp();
    // Load `i` of `2 · mine` (pass 1's, then pass 2's): this warp's tile `i % mine`, stage `i % 2`,
    // phase `(i / 2) % 2`.
    auto issue = [&](int i) {
        const int stage = i % STAGES;
        coord<x_tile> at = {0, 0, row_block, wb + (i % mine) * BW};
        warp::tma::expect_bytes(arrived[w][stage], sizeof(x_tile));
        warp::tma::load_async(x_smem[w][stage], g.x, at, arrived[w][stage]);
    };

    // Pass 1: this warp's tiles' max of absolute values, `quant_rows_fp8_ker`'s ops.
    row_values amax;
    warp::zero(amax);
    if (mine > 0) {
        issue(0);
    }
    for (int i = 0; i < mine; i++) {
        const int stage = i % STAGES;
        __syncwarp();
        issue(i + 1);  // pass 1's next tile, or pass 2's first (`i + 1 == mine`)
        wait(arrived[w][stage], (i / STAGES) % 2);
        acc_tile x;
        warp::load(x, x_smem[w][stage]);
        warp::abs(x, x);
        warp::row_max(amax, x, amax);
    }
    // The block's `BW` warps' maxima, then their max in each of them.
    warp::store(amax_s[w], amax);
    __syncthreads();
    row_values all, other;
    warp::load(all, amax_s[first]);
    for (int v = 1; v < BW; v++) {
        warp::load(other, amax_s[first + v]);
        warp::max(all, all, other);
    }
    row_values scale;
    warp::unary_op<scratchy::ops::fp8_scale>(scale, all);
    if (live && wb == 0) {
        warp::store(g.scale, scale, {0, 0, 0, row_block});
    }

    // Pass 2: each of this warp's tiles divided by its rows' scales, cast to e4m3 and stored.
    for (int i = mine; i < 2 * mine; i++) {
        const int stage = i % STAGES;
        // The store from this stage's e4m3 tile two iterations ago must have read it before this
        // one overwrites it; the other stage's bf16 tile was read last iteration.
        warp::tma::store_async_read_wait<1>();
        __syncwarp();
        if (i + 1 < 2 * mine) {
            issue(i + 1);
        }
        wait(arrived[w][stage], (i / STAGES) % 2);
        acc_tile x;
        warp::load(x, x_smem[w][stage]);
        warp::row_map<scratchy::ops::fp8_quant_pre_nz>(x, x, scale);
        fp8_tile q;
        warp::copy(q, x);
        warp::store(q_smem[w][stage], q);
        __syncwarp();
        coord<q_tile> at = {0, 0, row_block, wb + (i - mine) * BW};
        warp::tma::store_async(g.q, q_smem[w][stage], at);
    }
    warp::tma::store_async_wait();
}

}  // namespace

// The kernels' one-time setup: allow `quant_rows_fp8_ker` `QUANT_SMEM_BYTES` and each
// `quant_rows_fp8_wide_ker` instance `WIDE_SMEM_BYTES` of dynamic shared memory on the current
// device (the default limit is 48 KB). Run once per device before any launch
// (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch.
//
// 0 = done; any other value = the cudaError_t of the first attribute that failed.
extern "C" int scratchy_cuda_quant_rows_fp8_prepare(void) {
    cudaError_t e = cudaFuncSetAttribute(quant_rows_fp8_ker,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         QUANT_SMEM_BYTES);
    for (const auto ker : {quant_rows_fp8_wide_ker<1>, quant_rows_fp8_wide_ker<2>,
                           quant_rows_fp8_wide_ker<4>, quant_rows_fp8_wide_ker<WIDE_WARPS>}) {
        if (e == cudaSuccess) {
            e = cudaFuncSetAttribute(ker, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     WIDE_SMEM_BYTES);
        }
    }
    return static_cast<int>(e);
}

// Test support: `scratchy_cuda_quant_rows_fp8` at `variant` (0 `quant_rows_fp8_ker`, one warp a
// 16-row block, FP8 S3's form; 1 the launcher's: `quant_rows_fp8_wide_ker` at `wide_warps`, or
// FP8 S3's form at `narrow_many`), the same bits.
extern "C" int scratchy_cuda_quant_rows_fp8_variant(const void* x, void* q, void* scale,
                                                    uint32_t rows, uint32_t cols, uint32_t variant,
                                                    void* stream /* cudaStream_t */) {
    if (rows == 0 || rows % uint32_t(ROWS) != 0 || cols == 0 || cols % COL_MULTIPLE != 0 ||
        uint64_t(rows) * uint64_t(cols) > uint64_t(INT_MAX) || variant > 1) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        // Building a tile `gl` encodes its tensor map into host memory; nothing is allocated on
        // the device. The scale view carries no tensor map.
        x_gl xg{static_cast<bf16*>(const_cast<void*>(x)), nullptr, nullptr, size_t(rows),
                size_t(cols)};
        q_gl qg{static_cast<fp8e4m3*>(q), nullptr, nullptr, size_t(rows), size_t(cols)};
        scale_gl sg{static_cast<float*>(scale), nullptr, nullptr, nullptr, size_t(rows)};
        const uint32_t row_blocks = rows / uint32_t(ROWS);
        const int col_tiles = int((cols + uint32_t(COLS) - 1) / uint32_t(COLS));
        quant_globals G{xg, qg, sg, int(row_blocks), col_tiles};
        void* args[] = {&G};
        const cudaStream_t st = static_cast<cudaStream_t>(stream);
        if (variant == 1 && !narrow_many(row_blocks, col_tiles)) {
            const int bw = wide_warps(col_tiles);
            const auto ker = bw == 1   ? quant_rows_fp8_wide_ker<1>
                             : bw == 2 ? quant_rows_fp8_wide_ker<2>
                             : bw == 4 ? quant_rows_fp8_wide_ker<4>
                                       : quant_rows_fp8_wide_ker<WIDE_WARPS>;
            const uint32_t per_cta = uint32_t(WIDE_WARPS / bw);
            const dim3 grid((row_blocks + per_cta - 1) / per_cta);
            return static_cast<int>(
                cudaLaunchKernel(ker, grid, dim3(WIDE_THREADS), args, WIDE_SMEM_BYTES, st));
        }
        const dim3 grid((row_blocks + uint32_t(WARPS) - 1) / uint32_t(WARPS)), block(NUM_THREADS);
        return static_cast<int>(
            cudaLaunchKernel(quant_rows_fp8_ker, grid, block, args, QUANT_SMEM_BYTES, st));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value
// = the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `rows` must be a non-zero multiple of 16 and `cols` of 64 (`step::GEMM_FP8_K_MULTIPLE`), and
// `rows · cols` at most INT_MAX; all are checked before any pointer is touched. `x` must be a
// 16-byte-aligned device pointer to `rows·cols` bf16, `q` to `rows·cols` e4m3 bytes and `scale`
// to `rows` f32 (4-byte aligned), neither overlapping `x` nor each other.
// `scratchy_cuda_quant_rows_fp8_prepare` must have succeeded on the current device. The launch is
// asynchronous on `stream`; the caller owns it and synchronizes. It runs
// `quant_rows_fp8_wide_ker` (F3), whose bits are `quant_rows_fp8_ker`'s, or `quant_rows_fp8_ker`
// itself at `narrow_many`.
extern "C" int scratchy_cuda_quant_rows_fp8(const void* x, void* q, void* scale, uint32_t rows,
                                            uint32_t cols,
                                            void* stream /* cudaStream_t; null = default */) {
    return scratchy_cuda_quant_rows_fp8_variant(x, q, scale, rows, cols, 1, stream);
}

// ── Q1: the one-row form (the decode bucket's `QuantRowsFp8`) ──
//
// `rows == 1` in `quant_rows_fp8_ker` is one warp walking every tile of the row twice through a
// 2-stage ring (454 µs at 5,120 columns, 1,529 at 17,408 on the H100). The one-row form is the
// one-row RMSNorm's shape (`rms_norm_bf16.cu`): `col_tiles` CTAs of 8 warps, CTA `t` owning
// column chunk `t`.
//   1. Every CTA computes the whole row's `amax`, as the fp8 GEMV's prologue does (`gemv_fp8.cu`'s
//      `quantize_row`): warp `w` TMA-loads its 256-column chunks `w, w + 8, …` as `sv_bf<256>` in
//      one burst, `warp::load` into a naive `rv_fl<256>` (exact), `warp::abs`, `warp::max`, then
//      `warp::max` to one value, written twice into an `sv_fl<16>`, and every warp reduces the 16.
//      A max of absolute values is exact in any order, so it is `quant_rows_fp8`'s `amax`.
//   2. `scale = fp8_scale(amax)`; CTA 0 stores it. The owned chunk is already in shared memory
//      (pass 1 loaded it), and each of the CTA's 256 threads quantizes one of its 256 elements, the
//      same `fp8_quant_pre` call as `quant_rows_fp8`, then TK's scalar e4m3 convertor (round to
//      nearest even, saturating: the byte `warp::copy` to an fp8 tile writes,
//      `scratchy::ops::fp8_round`'s), into row 0 of an e4m3 tile, TMA-stored through the `[1, cols]`
//      map (rows past 1 dropped).
// The same ops on the same values, so row 0's bytes and scale are `quant_rows_fp8`'s
// (`tests/quant_rows_fp8.rs`). Two tile forms before it (dividing the tile's 15 zero-fill rows,
// then row 0 only) took ~24 and ~11 µs: a division of a zero takes `__fdiv_rn`'s slow path, and
// the zero rows' branch serialized warp 0's divisions.

namespace {

constexpr int ONE_WARPS = 8;
constexpr int ONE_THREADS = ONE_WARPS * kittens::WARP_THREADS;
// The longest row's chunks (17,408 / 256), and a warp's share of them, `⌈68 / 8⌉`.
constexpr int ONE_ROW_CHUNKS = 68;
constexpr int ONE_CHUNKS = (ONE_ROW_CHUNKS + 8 - 1) / 8;

using chunk_sv = sv_bf<COLS>;
using chunk_rv = rv_fl<COLS>;
using amax_rv = rv_fl<16>;
using one_x_gl = gl<bf16, 1, 1, -1, -1, x_tile, chunk_sv>;

struct one_row_globals {
    one_x_gl x;
    q_gl q;
    scale_gl scale;
    int col_tiles;
    int cols;
};

constexpr size_t ONE_SMEM_USED = size_t(ONE_WARPS) * ONE_CHUNKS * sizeof(chunk_sv) + sizeof(q_tile);
constexpr int ONE_SMEM_BYTES = int(ONE_SMEM_USED) + 1024;
static_assert(ONE_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

__global__ __launch_bounds__(ONE_THREADS, 1) void quant_row_fp8_one_ker(
    const __grid_constant__ one_row_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    chunk_sv(&row)[ONE_WARPS][ONE_CHUNKS] = al.allocate<chunk_sv, ONE_WARPS, ONE_CHUNKS>();
    q_tile& q_s = al.allocate<q_tile>();
    __shared__ kittens::semaphore chunks_arrived[ONE_WARPS];
    __shared__ sv_fl<16> amax_s;
    const int w = kittens::warpid();
    const int t_own = blockIdx.x;
    const int mine = (g.col_tiles - w + ONE_WARPS - 1) / ONE_WARPS;
    if (warp::laneid() == 0) {
        init_semaphore(chunks_arrived[w], 0, 1);
    }
    __syncwarp();
    // Pass 1: this warp's chunks in one burst, their max of absolute values.
    warp::tma::expect_bytes(chunks_arrived[w], uint32_t(mine * sizeof(chunk_sv)));
    for (int i = 0; i < mine; i++) {
        warp::tma::load_async(row[w][i], g.x, {0, 0, 0, w + i * ONE_WARPS}, chunks_arrived[w]);
    }
    chunk_rv x, m;
    warp::zero(m);
    if (mine > 0) {
        wait(chunks_arrived[w], 0);
    }
    for (int i = 0; i < mine; i++) {
        warp::load(x, row[w][i]);
        warp::abs(x, x);
        warp::max(m, m, x);
    }
    const float part = warp::max(m);
    if (warp::laneid() == 0) {
        amax_s[w] = part;
        amax_s[w + ONE_WARPS] = part;
    }
    __syncthreads();
    amax_rv all;
    warp::load(all, amax_s);
    const float scale = scratchy::ops::fp8_scale::op<float>(warp::max(all));
    if (t_own == 0 && threadIdx.x == 0) {
        g.scale[{0, 0, 0, 0}] = scale;
    }
    // Pass 2, one element a thread: the owned chunk is warp `t_own % 8`'s load `t_own / 8` (its
    // arrival waited on by every thread: the phase has completed).
    const int owner = t_own % ONE_WARPS;
    wait(chunks_arrived[owner], 0);
    const int col = int(threadIdx.x);
    if (t_own * COLS + col < g.cols) {
        const float v = base_types::convertor<float, bf16>::convert(row[owner][t_own / ONE_WARPS][col]);
        q_s[{0, col}] = base_types::convertor<fp8e4m3, float>::convert(
            scratchy::ops::fp8_quant_pre::op<float>(v, scale));
    }
    // Every thread's byte is written before warp 0's TMA store reads the tile (TK's store fences
    // the async proxy).
    __syncthreads();
    if (w == 0) {
        coord<q_tile> at = {0, 0, 0, t_own};
        warp::tma::store_async(g.q, q_s, at);
        warp::tma::store_async_wait();
    }
}

}  // namespace

// The one-row form's one-time setup (in `PREPARE`, `kernels::prepare_all`). 0 = done; any other
// value = the cudaError_t.
extern "C" int scratchy_cuda_quant_row_fp8_one_prepare(void) {
    return static_cast<int>(cudaFuncSetAttribute(
        quant_row_fp8_one_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_SMEM_BYTES));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = refused, nothing launched; SCRATCHY_CUDA_ERR_HOST = a
// C++ exception was caught; any other value = the cudaError_t of the launch.
//
// `quant_rows_fp8` of one row: `x [1, cols]` bf16 into `q [1, cols]` e4m3 and `scale [1]` f32, the
// bits of `scratchy_cuda_quant_rows_fp8`'s row 0 for the same row. `cols` a non-zero multiple of 64
// at most 17,408 (the shared chunks hold 68 256-column chunks), checked first. Pointers as
// `scratchy_cuda_quant_rows_fp8`'s. `scratchy_cuda_quant_row_fp8_one_prepare` must have succeeded on
// the current device. Asynchronous on `stream`.
extern "C" int scratchy_cuda_quant_row_fp8_one(const void* x, void* q, void* scale, uint32_t cols,
                                               void* stream /* cudaStream_t; null = default */) {
    if (cols == 0 || cols % COL_MULTIPLE != 0 || cols > uint32_t(ONE_ROW_CHUNKS) * uint32_t(COLS)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const int col_tiles = int((cols + uint32_t(COLS) - 1) / uint32_t(COLS));
        one_row_globals G{
            one_x_gl{static_cast<bf16*>(const_cast<void*>(x)), nullptr, nullptr, size_t(1),
                     size_t(cols)},
            q_gl{static_cast<fp8e4m3*>(q), nullptr, nullptr, size_t(1), size_t(cols)},
            scale_gl{static_cast<float*>(scale), nullptr, nullptr, nullptr, size_t(1)},
            col_tiles, int(cols)};
        void* args[] = {&G};
        return static_cast<int>(cudaLaunchKernel(quant_row_fp8_one_ker, dim3(col_tiles),
                                                 dim3(ONE_THREADS), args, ONE_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
