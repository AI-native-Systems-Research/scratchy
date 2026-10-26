// SPDX-License-Identifier: Apache-2.0
//
// Gemma 4's one-operand bf16 row maps for the H100, over `x` and `out` `[rows, cols]` row-major bf16
// of the same shape (no broadcast), each element widened to f32 (exact), computed in f32 and
// rounded once to bf16 (round-to-nearest-even):
//   - `Scale`: `out = x · scale`, `scale` a compile-time constant of the step (Gemma's embedding
//     scale, `√hidden`); Metal's `scalar_mul` (`shaders/elementwise.metal`).
//   - `LayerScale`: `out = x · w[0]`, `w` a `[1]` bf16 weight (Gemma's per-layer `layer_scalar`);
//     Metal's `scalar_weight_mul`.
//   - `SoftCap`: `out = cap · tanh(x / cap)` (Gemma's final logits, cap 30), the op struct
//     `scratchy::ops::soft_cap`; Metal's `tanh_soft_cap`.
//
// ThunderKittens' 16-row tile (PLAN-tk100 §2 S2), `silu_mul_bf16.cu`'s structure with one operand:
// one CTA per block of 16 rows, one warpgroup of 4 warps. Warp `w` takes the 256-column tiles
// `w, w + 4, …` of the block. Per tile, lane 0 of the warp TMA-loads `x` into a plain shared tile
// `st_bf<16, 256, false>` (`tma::expect_bytes` on the stage's semaphore), the warp loads it into
// `rt_fl<16, 256>` (bf16 to f32, exact), applies the map (`warp::mul` by the scalar, or
// `warp::bin_map<scratchy::ops::soft_cap>` with the cap), stores the result back into the same
// tile (`warp::store`, the one RNE rounding), `__syncwarp()`, and lane 0 TMA-stores it. Every tile
// is plain, never swizzled: a TMA box past the array's rows (the last block's padding) or columns
// (a partial last tile, `cols % 256 ≠ 0`: Gemma's 704 and 2112) reads zeros and stores nothing
// (TK-1's R1 gate, `tests/tma_bounds.rs`), so the output's tail is never written. Each warp keeps
// a 2-stage ring: the next tile's load is in flight while this one computes. A warp's tiles are
// its own, so no warp waits on another.
//
// `out == x` is safe (the soft-cap's logits may be capped in place): each element is read and
// written by the same warp through the same shared tile, whose TMA load has completed (the `wait`
// on its semaphore) before its TMA store is issued; no warp or CTA reads another's tiles, and the
// next tile's prefetch reads other addresses. Any other overlap of `out` and `x` is not allowed.
//
// No reduction: each element's result depends on its own input only, so the bits do not depend on
// the launch shape or on `out == x`. Built with `--use_fast_math` (build.rs): f32 subnormals are
// flushed to zero, `x / cap` is the fast division and `tanhf` the intrinsic `__tanhf`. Shared
// memory: 4 warps × 2 stages × 8 KiB, plus 1,024 bytes of alignment slack = 66,560 bytes of dynamic
// shared memory, set once for the three instantiations by `scratchy_cuda_pointwise_bf16_prepare`
// (in `PREPARE`).

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

// `step::SCALE_BF16_COLS` and `step::SOFT_CAP_BF16_COLS`: `cols` is a multiple of it, so every row
// stride is a multiple of 16 bytes, as the tiles' tensor maps need.
constexpr uint32_t VEC = 8;
constexpr int ROWS = 16;
constexpr int COLS = 256;
constexpr int WARPS = 4;
constexpr int STAGES = 2;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;

// Which map a launch runs.
constexpr int SCALE = 0;
constexpr int LAYER_SCALE = 1;
constexpr int SOFT_CAP = 2;

using tile = st_bf<ROWS, COLS, false>;
using row_gl = gl<bf16, 1, 1, -1, -1, tile>;
// The layer scale's `[1]` weight, read through `gl::operator[]`; no TMA type, so building it
// encodes nothing. The other two maps never read it.
using scalar_gl = gl<bf16, 1, 1, 1, -1>;

struct pw_globals {
    row_gl x;
    row_gl out;
    scalar_gl w;
    // The 256-column tiles of a row, the last one partial when `cols % 256 ≠ 0`.
    int col_tiles;
    // The scale's factor or the soft-cap's cap; unused by the layer scale.
    float param;
};

constexpr size_t SMEM_TILES = WARPS * STAGES * sizeof(tile);
static_assert(SMEM_TILES == 65536, "4 warps x 2 stages x 8 KiB");
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int PW_SMEM_BYTES = int(SMEM_TILES) + 1024;
static_assert(PW_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

template <int OP>
__global__ __launch_bounds__(NUM_THREADS, 1) void pointwise_bf16_ker(
    const __grid_constant__ pw_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    tile(&x_smem)[WARPS][STAGES] = al.allocate<tile, WARPS, STAGES>();
    __shared__ kittens::semaphore arrived[WARPS][STAGES];

    const int w = kittens::warpid();
    const int row_block = blockIdx.x;
    if (warp::laneid() == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_semaphore(arrived[w][s], 0, 1);
        }
    }
    __syncwarp();
    // The layer scale's factor: the weight's one element, widened exactly to f32. Every lane reads
    // the same address once, before any tile.
    float param = g.param;
    if constexpr (OP == LAYER_SCALE) {
        param = base_types::convertor<float, bf16>::convert(g.w[{0, 0, 0, 0}]);
    }

    // This warp's tiles `w, w + 4, …`; the `i`-th of them is in stage `i % 2`.
    auto issue = [&](int i) {
        const int stage = i % STAGES;
        coord<tile> at = {0, 0, row_block, w + i * WARPS};
        warp::tma::expect_bytes(arrived[w][stage], sizeof(tile));
        warp::tma::load_async(x_smem[w][stage], g.x, at, arrived[w][stage]);
    };
    const int mine = (g.col_tiles - w + WARPS - 1) / WARPS;
    if (mine > 0) {
        issue(0);
    }
    for (int i = 0; i < mine; i++) {
        const int stage = i % STAGES;
        // The other stage's previous store must have read its tile before the next load
        // overwrites it.
        warp::tma::store_async_read_wait();
        __syncwarp();
        if (i + 1 < mine) {
            issue(i + 1);
        }
        wait(arrived[w][stage], (i / STAGES) % 2);
        rt_fl<ROWS, COLS> x;
        warp::load(x, x_smem[w][stage]);
        if constexpr (OP == SOFT_CAP) {
            warp::bin_map<scratchy::ops::soft_cap>(x, x, param);
        } else {
            warp::mul(x, x, param);
        }
        warp::store(x_smem[w][stage], x);
        __syncwarp();
        coord<tile> at = {0, 0, row_block, w + i * WARPS};
        warp::tma::store_async(g.out, x_smem[w][stage], at);
    }
    warp::tma::store_async_wait();
}

// The shape rule every launcher checks before it touches a pointer.
bool refused(uint32_t rows, uint32_t cols) {
    return rows == 0 || cols == 0 || cols % VEC != 0 ||
           uint64_t(rows) * uint64_t(cols) > uint64_t(INT_MAX);
}

// Build the views and launch map `OP`. Can throw (a tensor-map encode); the caller catches.
template <int OP>
int launch(const void* x, const void* w, void* out, uint32_t rows, uint32_t cols, float param,
           cudaStream_t stream) {
    bf16* xp = static_cast<bf16*>(const_cast<void*>(x));
    bf16* wp = static_cast<bf16*>(const_cast<void*>(w));
    bf16* op = static_cast<bf16*>(out);
    // Building each tile `gl` encodes its tensor map into host memory; nothing is allocated on the
    // device. Every row stride is `2·cols` bytes, a multiple of 16.
    row_gl xg{xp, nullptr, nullptr, size_t(rows), size_t(cols)};
    row_gl og{op, nullptr, nullptr, size_t(rows), size_t(cols)};
    scalar_gl wg{wp, nullptr, nullptr, nullptr, size_t(1)};
    pw_globals G{xg, og, wg, int((cols + uint32_t(COLS) - 1) / uint32_t(COLS)), param};
    void* args[] = {&G};
    const dim3 grid((rows + uint32_t(ROWS) - 1) / uint32_t(ROWS)), block(NUM_THREADS);
    return static_cast<int>(
        cudaLaunchKernel(pointwise_bf16_ker<OP>, grid, block, args, PW_SMEM_BYTES, stream));
}

}  // namespace

// The kernels' one-time setup: allow each of the three instantiations `PW_SMEM_BYTES` of dynamic
// shared memory on the current device (the default limit is 48 KB). Run once per device before any
// launch (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch.
//
// 0 = done; any other value = the first cudaError_t of setting the attribute.
extern "C" int scratchy_cuda_pointwise_bf16_prepare(void) {
    for (auto ker : {pointwise_bf16_ker<SCALE>, pointwise_bf16_ker<LAYER_SCALE>,
                     pointwise_bf16_ker<SOFT_CAP>}) {
        const cudaError_t e =
            cudaFuncSetAttribute(ker, cudaFuncAttributeMaxDynamicSharedMemorySize, PW_SMEM_BYTES);
        if (e != cudaSuccess) {
            return static_cast<int>(e);
        }
    }
    return 0;
}

// Every launcher below: 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value =
// the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `rows` and `cols` must be non-zero, `cols` a multiple of 8, and `rows · cols` at most INT_MAX;
// all three are checked before any pointer is touched. `x` and `out` must be 16-byte-aligned
// device pointers to `rows·cols` bf16 each; `out` may equal `x` exactly, and may overlap it in no
// other way. `scratchy_cuda_pointwise_bf16_prepare` must have succeeded on the current device. The
// launch is asynchronous on `stream`; the caller owns it and synchronizes.

// `out = x · scale`.
extern "C" int scratchy_cuda_scale_bf16(const void* x, void* out, uint32_t rows, uint32_t cols,
                                        float scale,
                                        void* stream /* cudaStream_t; null = default stream */) {
    if (refused(rows, cols)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        return launch<SCALE>(x, nullptr, out, rows, cols, scale,
                             static_cast<cudaStream_t>(stream));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// `out = x · w[0]`, `w` a device pointer to one bf16 (any alignment of a bf16), overlapping
// neither `x` nor `out`.
extern "C" int scratchy_cuda_layer_scale_bf16(const void* x, const void* w, void* out,
                                              uint32_t rows, uint32_t cols,
                                              void* stream /* cudaStream_t; null = default */) {
    if (refused(rows, cols)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        return launch<LAYER_SCALE>(x, w, out, rows, cols, 0.0f,
                                   static_cast<cudaStream_t>(stream));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// `out = cap · tanh(x / cap)`.
extern "C" int scratchy_cuda_soft_cap_bf16(const void* x, void* out, uint32_t rows, uint32_t cols,
                                           float cap,
                                           void* stream /* cudaStream_t; null = default stream */) {
    if (refused(rows, cols)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        return launch<SOFT_CAP>(x, nullptr, out, rows, cols, cap,
                                static_cast<cudaStream_t>(stream));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
