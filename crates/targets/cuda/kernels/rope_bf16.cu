// SPDX-License-Identifier: Apache-2.0
//
// bf16 partial rotary embedding (RoPE) for the H100, over x and out [rows, cols], cols = heads ·
// head_dim, all row-major bf16. Only the first `rot_dim` lanes of each head are rotated, in NeoX
// pairing within that width: lane d pairs with lane d + rot_dim/2, for d < rot_dim/2, and
//
//     out[d]             = x[d] · c − x[d + R/2] · s
//     out[d + R/2]       = x[d + R/2] · c + x[d] · s
//
// with c = table[row, d] and s = table[row, R/2 + d], computed in f32 and rounded once to bf16
// (round-to-nearest-even). The lanes rot_dim..head_dim pass through bit for bit: `out` is a fresh
// buffer (CUDA's colour rule), so they are copied, not left in place. qwen3.5 rotates 64 of each
// head's 256 lanes. Metal's counterpart is `shaders/rope.metal`'s pair at `(d, d + pair_off)`.
//
// `table` is the tape's rotary table, [rows, rot_dim] bf16, row r laid out [cos(0..R/2) |
// sin(0..R/2)] for token r's position (built on the host, `src/rotary.rs`).
//
// ThunderKittens' 16-row tile (PLAN-tk100 §2 S2): one CTA of one warp per block of 16 rows,
// instantiated for rot_dim 64. ThunderKittens' `kernels/rotary/rotary.cu` (at the vendored
// 67845f5f) rotates the full head on [B, H, N, D] tiles; this kernel keeps its idiom on a partial
// width. x, table and out are viewed with the plain tile `st_bf<16, 64, false>`, so a row block
// is `cols / 64` boxes and head h is the `head_dim / 64` boxes from `h · head_dim / 64` (a tile
// coordinate's column is scaled by the tile's 64 columns, `coord::unit_coord`); a head's first
// box is lanes 0..63, the rotated ones, and the rest pass through. A plain `st_subtile` of a wider
// plain tile is indexed by its own width, so a head is never loaded as one 256-wide tile and
// viewed 64 wide.
//
// Per block, lane 0 TMA-loads the table's box once, then walks the row block's boxes through a
// 4-stage ring (`tma::expect_bytes` on each stage's semaphore), the next boxes' loads in flight
// while this one is handled. A rotated box is loaded whole into `rt_fl<16, 64>` (bf16 to f32,
// exact), split with the table into halves by copying base tiles (`x0` = tiles 0..1, `x1` = 2..3;
// `cos`, `sin` likewise), rotated with TK's maps (`warp::mul`, then `warp::sub` or `warp::add`),
// rejoined, and stored back into its box (`warp::store`, the one RNE rounding). Lane 0 then
// TMA-stores every box to `out`: a pass-through box as it was loaded, so those lanes never leave
// shared memory as anything but the loaded bytes. Every tile is plain, never swizzled: a box past
// the arrays' rows (the last block's padding) reads zeros and stores nothing (TK-1's R1 gate,
// `tests/tma_bounds.rs`), so the output's tail is never written.
//
// Position 0 (c = 1, s = 0) is exact: `x0·1 − x1·0 = x0` and `x1·1 + x0·0 = x1` in f32, and under
// an FMA contraction too, signed zeros included. Each product is its own `warp::mul` and the sum
// or difference its own map; built with `--use_fast_math` (build.rs), nvcc may still contract a
// product into the add or subtract after inlining, as it could in the plain-CUDA kernel's
// `x0·c − x1·s`. So the rotated lanes may differ from that kernel by about one f32 rounding before
// the bf16 one, which the RoPE bound's `2⁻¹⁶` term covers. Each element depends only on its own
// row's inputs, so the bits do not depend on the launch shape.
//
// Shared memory: 4 stages × 2 KiB, the table's 2 KiB box, plus 1,024 bytes of alignment slack =
// 11,264 bytes of dynamic shared memory, under the 48 KiB default, so no one-time setup.

#include "kittens.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// The launcher's return codes that are not a `cudaError_t` (which is non-negative).
// `src/kernels.rs` mirrors them.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::ROPE_BF16_KERNEL_ROT_DIM`: the one rotated width the kernel is instantiated for, one box.
constexpr int ROT_DIM = 64;
constexpr int HALF = ROT_DIM / 2;
// `step::ROPE_BF16_KERNEL_HEAD_MULTIPLE`: `head_dim` is whole boxes, so every head starts at a box.
constexpr uint32_t HEAD_MULTIPLE = ROT_DIM;
constexpr int ROWS = 16;
constexpr int STAGES = 4;
constexpr int NUM_THREADS = kittens::WARP_THREADS;

using box = st_bf<ROWS, ROT_DIM, false>;
using box_gl = gl<bf16, 1, 1, -1, -1, box>;
using half_rt = rt_fl<ROWS, HALF>;
static_assert(half_rt::width == 2 && rt_fl<ROWS, ROT_DIM>::width == 4,
              "each half of the rotated width is two 16-column base tiles");

struct rope_globals {
    box_gl x;
    box_gl table;
    box_gl out;
    // The 64-column boxes of a row, and of a head.
    int row_boxes;
    int head_boxes;
};

constexpr size_t SMEM_BOXES = (STAGES + 1) * sizeof(box);
static_assert(SMEM_BOXES == 10240, "4 stages x 2 KiB, plus the table's box");
// The boxes plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int ROPE_SMEM_BYTES = int(SMEM_BOXES) + 1024;
static_assert(ROPE_SMEM_BYTES <= 48 * 1024, "under the default dynamic shared memory limit");

__global__ __launch_bounds__(NUM_THREADS, 1) void rope_bf16_ker(const __grid_constant__ rope_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    box(&x_smem)[STAGES] = al.allocate<box, STAGES>();
    box(&t_smem) = al.allocate<box>();
    __shared__ kittens::semaphore arrived[STAGES];
    __shared__ kittens::semaphore table_arrived;

    const int row_block = blockIdx.x;
    if (warp::laneid() == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_semaphore(arrived[s], 0, 1);
        }
        init_semaphore(table_arrived, 0, 1);
    }
    __syncwarp();

    warp::tma::expect_bytes(table_arrived, sizeof(box));
    warp::tma::load_async(t_smem, g.table, coord<box>{0, 0, row_block, 0}, table_arrived);
    // Box `i` of the row block into stage `i % 4`.
    auto issue = [&](int i) {
        const int stage = i % STAGES;
        warp::tma::expect_bytes(arrived[stage], sizeof(box));
        warp::tma::load_async(x_smem[stage], g.x, coord<box>{0, 0, row_block, i}, arrived[stage]);
    };
    for (int i = 0; i < STAGES - 1 && i < g.row_boxes; i++) {
        issue(i);
    }

    wait(table_arrived, 0);
    rt_fl<ROWS, ROT_DIM> t;
    warp::load(t, t_smem);
    half_rt c, s;
    c.tiles[0][0] = t.tiles[0][0];
    c.tiles[0][1] = t.tiles[0][1];
    s.tiles[0][0] = t.tiles[0][2];
    s.tiles[0][1] = t.tiles[0][3];

    for (int i = 0; i < g.row_boxes; i++) {
        const int stage = i % STAGES;
        // Box `i + 3` lands in the stage box `i − 1` was stored from, the newest store: it must
        // have read the stage before that load overwrites it.
        warp::tma::store_async_read_wait();
        __syncwarp();
        if (i + STAGES - 1 < g.row_boxes) {
            issue(i + STAGES - 1);
        }
        wait(arrived[stage], (i / STAGES) % 2);
        if (i % g.head_boxes == 0) {
            rt_fl<ROWS, ROT_DIM> x;
            warp::load(x, x_smem[stage]);
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
            warp::store(x_smem[stage], x);
            __syncwarp();
        }
        warp::tma::store_async(g.out, x_smem[stage], coord<box>{0, 0, row_block, i});
    }
    warp::tma::store_async_wait();
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value =
// the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// Refused before any pointer is touched: `rows`, `cols` or `head_dim` zero; `rot_dim` not 64
// (`step::ROPE_BF16_KERNEL_ROT_DIM`, the one rotated width the kernel is instantiated for);
// `head_dim` not a multiple of 64 (`step::ROPE_BF16_KERNEL_HEAD_MULTIPLE`, whole boxes);
// `cols` not a multiple of `head_dim`;
// `rows · cols` above INT_MAX. `x`, `table` and `out` must be 16-byte-aligned device pointers to
// `rows·cols`, `rows·rot_dim` and `rows·cols` bf16, `out` overlapping neither input. Its dynamic
// shared memory is under the 48 KiB default, so no one-time setup. The launch is asynchronous on
// `stream`; the caller owns it and synchronizes.
extern "C" int scratchy_cuda_rope_bf16(const void* x, const void* table, void* out, uint32_t rows,
                                       uint32_t cols, uint32_t head_dim, uint32_t rot_dim,
                                       void* stream /* cudaStream_t; null = default stream */) {
    if (rows == 0 || cols == 0 || head_dim == 0 || rot_dim != uint32_t(ROT_DIM) ||
        head_dim % HEAD_MULTIPLE != 0 || cols % head_dim != 0 || uint64_t(rows) * uint64_t(cols) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        bf16* xp = static_cast<bf16*>(const_cast<void*>(x));
        bf16* tp = static_cast<bf16*>(const_cast<void*>(table));
        bf16* op = static_cast<bf16*>(out);
        // Building each `gl` encodes its tensor map into host memory; nothing is allocated on
        // the device. Every row stride is a multiple of 128 bytes.
        box_gl xg{xp, nullptr, nullptr, size_t(rows), size_t(cols)};
        box_gl tg{tp, nullptr, nullptr, size_t(rows), size_t(rot_dim)};
        box_gl og{op, nullptr, nullptr, size_t(rows), size_t(cols)};
        rope_globals G{xg, tg, og, int(cols / uint32_t(ROT_DIM)), int(head_dim / uint32_t(ROT_DIM))};
        void* args[] = {&G};
        const dim3 grid((rows + uint32_t(ROWS) - 1) / uint32_t(ROWS)), block(NUM_THREADS);
        return static_cast<int>(cudaLaunchKernel(rope_bf16_ker, grid, block, args,
                                                 ROPE_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
