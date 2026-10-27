// SPDX-License-Identifier: Apache-2.0
//
// bf16 shared-expert gate scale for the H100: out = routed + shared · (1 / (1 + e^(−g))) over
// [rows, cols], `routed`, `shared` and `out` row-major bf16 of the same shape, `g` [rows, 1] bf16
// broadcast across its row (qwen3-5-moe's `shared_expert_gate`, MoE step M6 of #857). Each element
// is widened to f32 (exact), computed in f32 and rounded once to bf16 (round-to-nearest-even). The
// sigmoid is the op struct `scratchy::ops::sigmoid`, `1 / (1 + expf(−g))`, Metal's and the shared
// `GateScale`'s formula.
//
// ThunderKittens' 16-row tile (PLAN-tk100 §2 S2), `gate_apply_bf16.cu`'s structure: one CTA per
// block of 16 rows, one warpgroup of 4 warps. Warp `w` takes the 256-column tiles `w, w + 4, …` of
// the block. Per tile, lane 0 of the warp TMA-loads `routed` and `shared` into plain shared tiles
// `st_bf<16, 256, false>` (`tma::expect_bytes` on the stage's semaphore), the warp loads them into
// `rt_fl<16, 256>` (bf16 to f32, exact), scales each row of `shared` by its σ(g)
// (`warp::mul_row`), adds `routed` (`warp::add`), stores the sum into the first operand's tile
// (`warp::store`, the one RNE rounding), and lane 0 TMA-stores it. Every tile is plain, never
// swizzled: a TMA box past the array's rows (the last block's padding) or columns (a partial last
// tile, `cols % 256 ≠ 0`) then reads zeros and stores nothing (TK-1's R1 gate,
// `tests/tma_bounds.rs`), so the output's tail is never written. Each warp keeps a 2-stage ring
// per operand.
//
// The row gate: `g` is `[rows, 1]`, so a plain column-vector load would read 16 values past a
// one-row region. Each warp builds the tile's column vector once with TK's `apply` over
// `gl::operator[]`, `row < rows ? g[row] : 0`, then `unary_op<scratchy::ops::sigmoid>` on it.
//
// No reduction: each element's result depends on its own inputs only, so the bits do not depend
// on the launch shape. Built with `--use_fast_math` (build.rs): `expf` is the fast exponential.
// Shared memory: 4 warps × 2 stages × 2 operands × 8 KiB, plus 1,024 bytes of alignment slack =
// 132,096 bytes of dynamic shared memory, set once by `scratchy_cuda_gate_scale_bf16_prepare` (in
// `PREPARE`).

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

// `step::GATE_SCALE_BF16_COLS`: `cols` is a multiple of it, so every row stride is a multiple of
// 16 bytes, as the tiles' tensor maps need.
constexpr uint32_t VEC = 8;
constexpr int ROWS = 16;
constexpr int COLS = 256;
constexpr int WARPS = 4;
constexpr int STAGES = 2;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;

using tile = st_bf<ROWS, COLS, false>;
using row_gl = gl<bf16, 1, 1, -1, -1, tile>;
// `g`, `[rows, 1]` bf16: no TMA type, read element by element.
using gate_gl = gl<bf16, 1, 1, -1, 1>;
using acc_tile = rt_fl<ROWS, COLS>;
// One value per tile row: the rows' σ(g).
using row_values = acc_tile::col_vec;

struct ew_globals {
    row_gl routed;
    row_gl shared;
    gate_gl g;
    row_gl out;
    int rows;
    // The 256-column tiles of a row, the last one partial when `cols % 256 ≠ 0`.
    int col_tiles;
};

constexpr size_t SMEM_TILES = WARPS * STAGES * 2 * sizeof(tile);
static_assert(SMEM_TILES == 131072, "4 warps x 2 stages x 2 operands x 8 KiB");
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int EW_SMEM_BYTES = int(SMEM_TILES) + 1024;
static_assert(EW_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

__global__ __launch_bounds__(NUM_THREADS, 1) void gate_scale_bf16_ker(const __grid_constant__ ew_globals g) {
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
        warp::tma::load_async(a_smem[w][stage], g.routed, at, arrived[w][stage]);
        warp::tma::load_async(b_smem[w][stage], g.shared, at, arrived[w][stage]);
    };
    const int mine = (g.col_tiles - w + WARPS - 1) / WARPS;
    // The block's 16 rows' σ(g), once per warp: `apply` over `gl::operator[]`, a row past
    // `rows` read as 0 (its outputs are never stored).
    row_values sig;
    const int row0 = row_block * ROWS;
    warp::apply(sig, sig, [&](int r, float) {
        return row0 + r < g.rows
                   ? base_types::convertor<float, bf16>::convert(g.g[{0, 0, row0 + r, 0}])
                   : 0.0f;
    });
    warp::unary_op<scratchy::ops::sigmoid>(sig, sig);
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
        acc_tile x, y;
        warp::load(x, a_smem[w][stage]);
        warp::load(y, b_smem[w][stage]);
        warp::mul_row(y, y, sig);
        warp::add(x, x, y);
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
extern "C" int scratchy_cuda_gate_scale_bf16_prepare(void) {
    return static_cast<int>(cudaFuncSetAttribute(
        gate_scale_bf16_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, EW_SMEM_BYTES));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value
// = the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `rows` and `cols` must be non-zero, `cols` a multiple of 8, and `rows · cols` at most INT_MAX;
// all three are checked before any pointer is touched. `routed`, `shared` and `out` must be
// 16-byte-aligned device pointers to `rows·cols` bf16 each and `g` to `rows` bf16, `out`
// overlapping no input. `scratchy_cuda_gate_scale_bf16_prepare` must have succeeded on the current
// device. The launch is asynchronous on `stream`; the caller owns it and synchronizes.
extern "C" int scratchy_cuda_gate_scale_bf16(const void* routed, const void* shared,
                                             const void* g, void* out, uint32_t rows,
                                             uint32_t cols,
                                             void* stream /* cudaStream_t; null = default */) {
    if (rows == 0 || cols == 0 || cols % VEC != 0 ||
        uint64_t(rows) * uint64_t(cols) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        bf16* rp = static_cast<bf16*>(const_cast<void*>(routed));
        bf16* sp = static_cast<bf16*>(const_cast<void*>(shared));
        bf16* gp = static_cast<bf16*>(const_cast<void*>(g));
        bf16* op = static_cast<bf16*>(out);
        // Building each tile `gl` encodes its tensor map into host memory; nothing is allocated
        // on the device. Every row stride is `2·cols` bytes, a multiple of 16. `g`'s view has no
        // tensor map.
        row_gl rg{rp, nullptr, nullptr, size_t(rows), size_t(cols)};
        row_gl sg{sp, nullptr, nullptr, size_t(rows), size_t(cols)};
        gate_gl gg{gp, nullptr, nullptr, size_t(rows), nullptr};
        row_gl og{op, nullptr, nullptr, size_t(rows), size_t(cols)};
        ew_globals G{rg, sg, gg, og, int(rows), int((cols + uint32_t(COLS) - 1) / uint32_t(COLS))};
        void* args[] = {&G};
        const dim3 grid((rows + uint32_t(ROWS) - 1) / uint32_t(ROWS)), block(NUM_THREADS);
        return static_cast<int>(cudaLaunchKernel(gate_scale_bf16_ker, grid, block, args,
                                                 EW_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
