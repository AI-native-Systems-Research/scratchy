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
// Built with `--use_fast_math` (build.rs): `rsqrtf` is the approximate reciprocal square root, as
// before. Shared memory: 4 warps × 2 stages × (an 8 KiB tile and a 512-byte gain vector), and the
// four 64-byte sums, plus 1,024 bytes of alignment slack, set once by
// `scratchy_cuda_rms_norm_bf16_prepare` (in `PREPARE`).

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

using tile = st_bf<ROWS, COLS, false>;
using gain_vec = sv_bf<COLS>;
using sums_vec = sv_fl<ROWS>;
using x_gl = gl<bf16, 1, 1, -1, -1, tile>;
using g_gl = gl<bf16, 1, 1, 1, -1, gain_vec>;
using acc_tile = rt_fl<ROWS, COLS>;
using row_values = acc_tile::col_vec;
using col_values = acc_tile::row_vec;

struct rms_globals {
    x_gl x;
    g_gl gain;
    x_gl y;
    // The 256-column tiles of a row, the last one partial when `cols % 256 ≠ 0`.
    int col_tiles;
    float cols;
    float eps;
    float gain_offset;
};

constexpr size_t SMEM_USED =
    WARPS * STAGES * (sizeof(tile) + sizeof(gain_vec)) + WARPS * sizeof(sums_vec);
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int RMS_SMEM_BYTES = int(SMEM_USED) + 1024;
static_assert(RMS_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

template <bool UNIT>
__global__ __launch_bounds__(NUM_THREADS, 1) void rms_norm_bf16_ker(
    const __grid_constant__ rms_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    tile(&x_smem)[WARPS][STAGES] = al.allocate<tile, WARPS, STAGES>();
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
    auto issue = [&](int i) {
        const int stage = i % STAGES, t = w + (i % mine) * WARPS;
        const bool second = !UNIT && i >= mine;
        coord<tile> at = {0, 0, row_block, t};
        warp::tma::expect_bytes(arrived[w][stage], sizeof(tile) + (second ? sizeof(gain_vec) : 0));
        warp::tma::load_async(x_smem[w][stage], g.x, at, arrived[w][stage]);
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
        // Pass 1 stores nothing: the next load may take the other stage once its tile was read
        // (the previous iteration's `warp::load`, which every lane finished at this barrier).
        __syncwarp();
        issue(i + 1);  // pass 1's next tile, or pass 2's first (`i + 1 == mine`)
        wait(arrived[w][stage], (i / STAGES) % 2);
        acc_tile x;
        warp::load(x, x_smem[w][stage]);
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
        warp::mul_row(x, x, inv);
        if constexpr (!UNIT) {
            col_values gain;
            warp::load(gain, g_smem[w][stage]);
            warp::add(gain, gain, g.gain_offset);
            warp::mul_col(x, x, gain);
        }
        warp::store(x_smem[w][stage], x);
        __syncwarp();
        coord<tile> at = {0, 0, row_block, t};
        warp::tma::store_async(g.y, x_smem[w][stage], at);
    }
    warp::tma::store_async_wait();
}

// Build the views and launch `rms_norm_bf16_ker<UNIT>`. Can throw (a tensor-map encode); the
// caller catches. `gain` is never read under `UNIT`, but its view is still built.
template <bool UNIT>
int launch(const void* x, const void* gain, void* y, uint32_t rows, uint32_t cols, float eps,
           float gain_offset, cudaStream_t stream) {
    bf16* xp = static_cast<bf16*>(const_cast<void*>(x));
    bf16* gp = static_cast<bf16*>(const_cast<void*>(gain));
    bf16* yp = static_cast<bf16*>(y);
    // Building each `gl` encodes its tensor map into host memory; nothing is allocated on the
    // device. Every row stride is `2·cols` bytes, a multiple of 16.
    x_gl xg{xp, nullptr, nullptr, size_t(rows), size_t(cols)};
    g_gl gg{gp, nullptr, nullptr, nullptr, size_t(cols)};
    x_gl yg{yp, nullptr, nullptr, size_t(rows), size_t(cols)};
    const int col_tiles = int((cols + uint32_t(COLS) - 1) / uint32_t(COLS));
    rms_globals G{xg, gg, yg, col_tiles, float(cols), eps, gain_offset};
    void* args[] = {&G};
    const dim3 grid((rows + uint32_t(ROWS) - 1) / uint32_t(ROWS)), block(NUM_THREADS);
    return static_cast<int>(
        cudaLaunchKernel(rms_norm_bf16_ker<UNIT>, grid, block, args, RMS_SMEM_BYTES, stream));
}

}  // namespace

// The kernel's one-time setup: allow both instantiations `RMS_SMEM_BYTES` of dynamic shared memory
// on the current device (the default limit is 48 KB). Run once per device before any launch
// (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch.
//
// 0 = done; any other value = the first cudaError_t of setting the attribute.
extern "C" int scratchy_cuda_rms_norm_bf16_prepare(void) {
    for (auto ker : {rms_norm_bf16_ker<false>, rms_norm_bf16_ker<true>}) {
        const cudaError_t e =
            cudaFuncSetAttribute(ker, cudaFuncAttributeMaxDynamicSharedMemorySize, RMS_SMEM_BYTES);
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
    if (rows == 0 || cols == 0 || cols % VEC != 0 ||
        uint64_t(rows) * uint64_t(cols) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        return launch<false>(x, gain, y, rows, cols, eps, gain_offset,
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
    if (rows == 0 || cols == 0 || cols % VEC != 0 ||
        uint64_t(rows) * uint64_t(cols) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        // The gain view still encodes a tensor map, so it is given `x`, a valid 16-byte-aligned
        // address of at least `cols` bf16; `<true>` never reads it.
        return launch<true>(x, x, y, rows, cols, eps, 0.0f,
                            static_cast<cudaStream_t>(stream));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
