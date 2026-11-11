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

}  // namespace

// The kernel's one-time setup: allow it `QUANT_SMEM_BYTES` of dynamic shared memory on the current
// device (the default limit is 48 KB). Run once per device before any launch
// (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch.
//
// 0 = done; any other value = the cudaError_t of setting the attribute.
extern "C" int scratchy_cuda_quant_rows_fp8_prepare(void) {
    return static_cast<int>(cudaFuncSetAttribute(
        quant_rows_fp8_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, QUANT_SMEM_BYTES));
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
// asynchronous on `stream`; the caller owns it and synchronizes.
extern "C" int scratchy_cuda_quant_rows_fp8(const void* x, void* q, void* scale, uint32_t rows,
                                            uint32_t cols,
                                            void* stream /* cudaStream_t; null = default */) {
    if (rows == 0 || rows % uint32_t(ROWS) != 0 || cols == 0 || cols % COL_MULTIPLE != 0 ||
        uint64_t(rows) * uint64_t(cols) > uint64_t(INT_MAX)) {
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
        const dim3 grid((row_blocks + uint32_t(WARPS) - 1) / uint32_t(WARPS)), block(NUM_THREADS);
        return static_cast<int>(cudaLaunchKernel(quant_rows_fp8_ker, grid, block, args,
                                                 QUANT_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// ── Q1: the one-row form (FP8 S8's ceiling follow-up, test-only: no tape launches it yet) ──
//
// `rows == 1` in `quant_rows_fp8_ker` is one warp walking every tile of the row twice through a
// 2-stage ring (454 µs at 5,120 columns, 1,529 at 17,408 on the H100). The one-row form is the
// one-row RMSNorm's shape (`rms_norm_bf16.cu`): `col_tiles` CTAs of 8 warps, CTA `t` owning
// column tile `t`.
//   1. Every CTA computes the whole row's `amax`, as the fp8 GEMV's prologue does (`gemv_fp8.cu`'s
//      `quantize_row`): warp `w` TMA-loads its 256-column chunks `w, w + 8, …` as `sv_bf<256>` in
//      one burst, `warp::load` into a naive `rv_fl<256>` (exact), `warp::abs`, `warp::max`, then
//      `warp::max` to one value, written twice into an `sv_fl<16>`, and every warp reduces the 16.
//      A max of absolute values is exact in any order, so it is `quant_rows_fp8`'s `amax`.
//   2. `scale = fp8_scale(amax)`; CTA 0 stores it. Warp 0 TMA-loads tile `t` as `quant_rows_fp8`'s
//      plain `st_bf<16, 256>` from the `[1, cols]` map (rows 1..15 zero), and runs that kernel's
//      pass 2 on it: `row_map<fp8_quant_pre>` by a column vector of `scale`, `warp::copy` to
//      `rt_fp8e4m3`, stored and TMA-stored (rows past 1 and columns past `cols` dropped).
// The same ops on the same values, so row 0's bytes and scale are `quant_rows_fp8`'s.
//
// Q1 as first built (variant 0) took ~24 µs at any `cols` on the H100: pass 2 divided the tile's
// 15 TMA-zero-filled rows too, and `__fdiv_rn` of a zero takes its slow path (SASS `FCHK` then a
// `CALL` per element: 120 of a lane's 128). Variant 1, the launcher's, divides row 0 only:
// `row_map<fp8_quant_pre_row>` by the column vector `[scale, 0, …, 0]` (an `sv_fl<16>` written by
// lane 0..15 and `warp::load`ed), the same `fp8_quant_pre` call on row 0's values. A test-only
// `TRACE` instance records each CTA's phases (`clock64`, cycles from its start) into `trace`.

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

// `[ctas, 8]` f32 phase cycles (the `TRACE` instances'); no tensor map.
using trace_gl = gl<float, 1, 1, -1, -1>;

struct one_row_globals {
    one_x_gl x;
    q_gl q;
    scale_gl scale;
    trace_gl trace;
    int col_tiles;
};

constexpr size_t ONE_SMEM_USED =
    size_t(ONE_WARPS) * ONE_CHUNKS * sizeof(chunk_sv) + sizeof(x_tile) + sizeof(q_tile);
constexpr int ONE_SMEM_BYTES = int(ONE_SMEM_USED) + 1024;
static_assert(ONE_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

// The trace's phases: 0 entry, 1 chunks arrived, 2 amax shared, 3 tile arrived, 4 tile divided,
// 5 stored.
template <bool ROW_ONLY, bool TRACE>
__global__ __launch_bounds__(ONE_THREADS, 1) void quant_row_fp8_one_ker(
    const __grid_constant__ one_row_globals g) {
    const long long t0 = TRACE ? clock64() : 0;
    const auto mark = [&](int phase) {
        if constexpr (TRACE) {
            if (threadIdx.x == 0) {
                g.trace[{0, 0, int(blockIdx.x), phase}] = float(clock64() - t0);
            }
        }
    };
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    chunk_sv(&row)[ONE_WARPS][ONE_CHUNKS] = al.allocate<chunk_sv, ONE_WARPS, ONE_CHUNKS>();
    x_tile& x_s = al.allocate<x_tile>();
    q_tile& q_s = al.allocate<q_tile>();
    __shared__ kittens::semaphore chunks_arrived[ONE_WARPS];
    __shared__ kittens::semaphore owned;
    __shared__ sv_fl<16> amax_s;
    __shared__ sv_fl<16> row_scale_s;
    const int w = kittens::warpid();
    const int t_own = blockIdx.x;
    const int mine = (g.col_tiles - w + ONE_WARPS - 1) / ONE_WARPS;
    if (warp::laneid() == 0) {
        init_semaphore(chunks_arrived[w], 0, 1);
        if (w == 0) {
            init_semaphore(owned, 0, 1);
        }
    }
    __syncwarp();
    // Pass 1: this warp's chunks in one burst, their max of absolute values.
    warp::tma::expect_bytes(chunks_arrived[w], uint32_t(mine * sizeof(chunk_sv)));
    for (int i = 0; i < mine; i++) {
        warp::tma::load_async(row[w][i], g.x, {0, 0, 0, w + i * ONE_WARPS}, chunks_arrived[w]);
    }
    // Warp 0 also loads its owned tile now, for pass 2.
    if (w == 0) {
        warp::tma::expect_bytes(owned, sizeof(x_tile));
        warp::tma::load_async(x_s, g.x, {0, 0, 0, t_own}, owned);
    }
    chunk_rv x, m;
    warp::zero(m);
    if (mine > 0) {
        wait(chunks_arrived[w], 0);
    }
    mark(1);
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
    mark(2);
    amax_rv all;
    warp::load(all, amax_s);
    const float scale = scratchy::ops::fp8_scale::op<float>(warp::max(all));
    if (w == 0) {
        if (t_own == 0 && warp::laneid() == 0) {
            g.scale[{0, 0, 0, 0}] = scale;
        }
        // Pass 2: `quant_rows_fp8_ker`'s, on the owned tile.
        row_values sc;
        if constexpr (ROW_ONLY) {
            // `[scale, 0, …, 0]`: row 0's scale, and 0 for the 15 zero-fill rows.
            if (warp::laneid() < 16) {
                row_scale_s[warp::laneid()] = warp::laneid() == 0 ? scale : 0.0f;
            }
            __syncwarp();
            warp::load(sc, row_scale_s);
        } else {
            warp::zero(sc);
            warp::add(sc, sc, scale);
        }
        wait(owned, 0);
        mark(3);
        acc_tile v;
        warp::load(v, x_s);
        if constexpr (ROW_ONLY) {
            warp::row_map<scratchy::ops::fp8_quant_pre_row>(v, v, sc);
        } else {
            warp::row_map<scratchy::ops::fp8_quant_pre>(v, v, sc);
        }
        mark(4);
        fp8_tile q;
        warp::copy(q, v);
        warp::store(q_s, q);
        __syncwarp();
        coord<q_tile> at = {0, 0, 0, t_own};
        warp::tma::store_async(g.q, q_s, at);
        warp::tma::store_async_wait();
        mark(5);
    }
}

template <bool ROW_ONLY, bool TRACE>
cudaError_t allow_one() {
    return cudaFuncSetAttribute(quant_row_fp8_one_ker<ROW_ONLY, TRACE>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_SMEM_BYTES);
}

}  // namespace

// The one-row form's one-time setup, every instance (in `PREPARE`, `kernels::prepare_all`).
// 0 = done; any other value = the first failing cudaError_t.
extern "C" int scratchy_cuda_quant_row_fp8_one_prepare(void) {
    cudaError_t e = cudaSuccess;
    for (const cudaError_t r : {allow_one<true, false>(), allow_one<false, false>(),
                                allow_one<true, true>(), allow_one<false, true>()}) {
        if (e == cudaSuccess) {
            e = r;
        }
    }
    return static_cast<int>(e);
}

// Test support: Q1 at `variant` (0 the first form, dividing the zero rows; 1 the launcher's), its
// phase trace into `trace` (`[col_tiles, 8]` f32 cycles) when `trace` is not null. Otherwise as
// `scratchy_cuda_quant_row_fp8_one`.
extern "C" int scratchy_cuda_quant_row_fp8_one_variant(const void* x, void* q, void* scale,
                                                       uint32_t cols, uint32_t variant,
                                                       void* trace,
                                                       void* stream /* cudaStream_t */) {
    if (cols == 0 || cols % COL_MULTIPLE != 0 || cols > uint32_t(ONE_ROW_CHUNKS) * uint32_t(COLS) ||
        variant > 1) {
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
            // With no trace the view is never read; `scale` is a valid address for it.
            trace_gl{static_cast<float*>(trace != nullptr ? trace : scale), nullptr, nullptr,
                     size_t(trace != nullptr ? col_tiles : 1), size_t(trace != nullptr ? 8 : 1)},
            col_tiles};
        void* args[] = {&G};
        const bool traced = trace != nullptr;
        const auto ker = variant == 1 ? (traced ? quant_row_fp8_one_ker<true, true>
                                                : quant_row_fp8_one_ker<true, false>)
                                      : (traced ? quant_row_fp8_one_ker<false, true>
                                                : quant_row_fp8_one_ker<false, false>);
        return static_cast<int>(cudaLaunchKernel(ker, dim3(col_tiles), dim3(ONE_THREADS), args,
                                                 ONE_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
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
    return scratchy_cuda_quant_row_fp8_one_variant(x, q, scale, cols, 1, nullptr, stream);
}
