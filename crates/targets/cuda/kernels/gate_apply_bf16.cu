// SPDX-License-Identifier: Apache-2.0
//
// bf16 attention output gate for the H100: out = attn / (1 + e^(−gate)) over [rows, cols], `attn`, `gate` and `out` row-major bf16 of the
// same shape (no broadcast). Each element is widened to f32 (exact), computed in f32 and rounded
// once to bf16 (round-to-nearest-even). The expression is the plain-CUDA kernel's, `a / (1.0f + __expf(-g))`, now the op struct `scratchy::ops::gate_apply`.
//
// ThunderKittens' 16-row tile (PLAN-tk100 §2 S2): one CTA per block of 16 rows, one warpgroup of
// 4 warps. Warp `w` takes the 256-column tiles `w, w + 4, …` of the block. Per tile, lane 0 of
// the warp TMA-loads both operands into plain shared tiles `st_bf<16, 256, false>`
// (`tma::expect_bytes` on the stage's semaphore), the warp loads them into `rt_fl<16, 256>`
// (bf16 to f32, exact), computes with TK's map (`warp::bin_map<scratchy::ops::gate_apply>`), stores the result into the first
// operand's tile (`warp::store`, the one RNE rounding), and lane 0 TMA-stores it. Every tile is
// plain, never swizzled: a TMA box past the array's rows (the last block's padding) or columns (a
// partial last tile, `cols % 256 ≠ 0`) then reads zeros and stores nothing (TK-1's R1 gate,
// `tests/tma_bounds.rs`), so the output's tail is never written. Each warp keeps a 2-stage ring
// per operand: the next tile's load is in flight while this one computes. A warp's tiles are its
// own, so no warp waits on another.
//
// No reduction: each element's result depends on its own inputs only, so the bits do not depend
// on the launch shape. Built with `--use_fast_math` (build.rs): `__expf` is the fast exponential. Shared memory: 4 warps ×
// 2 stages × 2 operands × 8 KiB, plus 1,024 bytes of alignment slack = 132,096 bytes of dynamic
// shared memory, set once by `scratchy_cuda_gate_apply_bf16_prepare` (in `PREPARE`).

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

// `step::GATE_APPLY_BF16_COLS`: `cols` is a multiple of it, so every row stride is a multiple of 16 bytes,
// as the tiles' tensor maps need.
constexpr uint32_t VEC = 8;
constexpr int ROWS = 16;
constexpr int COLS = 256;
constexpr int WARPS = 4;
constexpr int STAGES = 2;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;

using tile = st_bf<ROWS, COLS, false>;
using row_gl = gl<bf16, 1, 1, -1, -1, tile>;

struct ew_globals {
    row_gl attn;
    row_gl gate;
    row_gl out;
    // The 256-column tiles of a row, the last one partial when `cols % 256 ≠ 0`.
    int col_tiles;
};

constexpr size_t SMEM_TILES = WARPS * STAGES * 2 * sizeof(tile);
static_assert(SMEM_TILES == 131072, "4 warps x 2 stages x 2 operands x 8 KiB");
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int EW_SMEM_BYTES = int(SMEM_TILES) + 1024;
static_assert(EW_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

__global__ __launch_bounds__(NUM_THREADS, 1) void gate_apply_bf16_ker(const __grid_constant__ ew_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    tile(&a_smem)[WARPS][STAGES] = al.allocate<tile, WARPS, STAGES>();
    tile(&b_smem)[WARPS][STAGES] = al.allocate<tile, WARPS, STAGES>();
    __shared__ kittens::semaphore arrived[WARPS][STAGES];

    const int w = kittens::warpid();
    const int row_block = blockIdx.x;
    if (warp::laneid() == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_semaphore(arrived[w][s], 0, 1);
        }
    }
    __syncwarp();

    // This warp's tiles `w, w + 4, …`; the `i`-th of them is in stage `i % 2`.
    auto issue = [&](int i) {
        const int stage = i % STAGES;
        coord<tile> at = {0, 0, row_block, w + i * WARPS};
        warp::tma::expect_bytes(arrived[w][stage], 2 * sizeof(tile));
        warp::tma::load_async(a_smem[w][stage], g.attn, at, arrived[w][stage]);
        warp::tma::load_async(b_smem[w][stage], g.gate, at, arrived[w][stage]);
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
        rt_fl<ROWS, COLS> x, y;
        warp::load(x, a_smem[w][stage]);
        warp::load(y, b_smem[w][stage]);
        warp::bin_map<scratchy::ops::gate_apply>(x, x, y);
        warp::store(a_smem[w][stage], x);
        __syncwarp();
        coord<tile> at = {0, 0, row_block, w + i * WARPS};
        warp::tma::store_async(g.out, a_smem[w][stage], at);
    }
    warp::tma::store_async_wait();
}

}  // namespace

// The kernel's one-time setup: allow it `EW_SMEM_BYTES` of dynamic shared memory on the current
// device (the default limit is 48 KB). Run once per device before any launch
// (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch.
//
// 0 = done; any other value = the cudaError_t of setting the attribute.
extern "C" int scratchy_cuda_gate_apply_bf16_prepare(void) {
    return static_cast<int>(cudaFuncSetAttribute(
        gate_apply_bf16_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, EW_SMEM_BYTES));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value
// = the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `rows` and `cols` must be non-zero, `cols` a multiple of 8, and `rows · cols` at most INT_MAX;
// all three are checked before any pointer is touched. `attn`, `gate` and `out` must be
// 16-byte-aligned device pointers to `rows·cols` bf16 each, `out` overlapping neither input.
// `scratchy_cuda_gate_apply_bf16_prepare` must have succeeded on the current device. The launch is
// asynchronous on `stream`; the caller owns it and synchronizes.
extern "C" int scratchy_cuda_gate_apply_bf16(const void* attn, const void* gate, void* out,
                                             uint32_t rows, uint32_t cols,
                                             void* stream /* cudaStream_t; null = default */) {
    if (rows == 0 || cols == 0 || cols % VEC != 0 ||
        uint64_t(rows) * uint64_t(cols) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        bf16* ap = static_cast<bf16*>(const_cast<void*>(attn));
        bf16* bp = static_cast<bf16*>(const_cast<void*>(gate));
        bf16* op = static_cast<bf16*>(out);
        // Building each `gl` encodes its tensor map into host memory; nothing is allocated on
        // the device. Every row stride is `2·cols` bytes, a multiple of 16.
        row_gl ag{ap, nullptr, nullptr, size_t(rows), size_t(cols)};
        row_gl bg{bp, nullptr, nullptr, size_t(rows), size_t(cols)};
        row_gl og{op, nullptr, nullptr, size_t(rows), size_t(cols)};
        ew_globals G{ag, bg, og, int((cols + uint32_t(COLS) - 1) / uint32_t(COLS))};
        void* args[] = {&G};
        const dim3 grid((rows + uint32_t(ROWS) - 1) / uint32_t(ROWS)), block(NUM_THREADS);
        return static_cast<int>(cudaLaunchKernel(gate_apply_bf16_ker, grid, block, args, EW_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
