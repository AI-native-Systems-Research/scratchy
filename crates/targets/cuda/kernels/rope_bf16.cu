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
// instantiated for rot_dim 64 (qwen3.5's partial rope) and 256 (Gemma 4's sliding layers, a full
// rope on a 256-lane head: lane d < 128 pairs with lane d + 128). ThunderKittens'
// `kernels/rotary/rotary.cu` (at the vendored 67845f5f) rotates the full head on [B, H, N, D]
// tiles; this kernel keeps its idiom on token-major rows. x, table and out are viewed with the
// plain tile `st_bf<16, 64, false>` (a tile coordinate's column is scaled by the tile's 64
// columns, `coord::unit_coord`), and a row block is walked in units of `rot_dim / 64` boxes,
// `rot_dim` lanes: head h is the `head_dim / rot_dim` units from `h · head_dim / rot_dim`, its
// first unit is the rotated lanes, and the rest pass through. A plain `st_subtile` of a wider
// plain tile is indexed by its own width, so a head is never loaded as one 256-wide tile and
// viewed 64 wide.
//
// Per block, lane 0 TMA-loads the table's unit once, then walks the row block's units through a
// ring of stages (`tma::expect_bytes` on each stage's semaphore, one per unit's boxes), the next
// units' loads in flight while this one is handled: 4 stages of one box at rot_dim 64, 2 of four
// boxes at 256. At rot_dim 64 a rotated box is loaded whole into `rt_fl<16, 64>` (bf16 to f32,
// exact), split with the table into halves by copying base tiles (`x0` = tiles 0..1, `x1` = 2..3;
// `cos`, `sin` likewise), rotated with TK's maps (`warp::mul`, then `warp::sub` or `warp::add`),
// rejoined, and stored back into its box (`warp::store`, the one RNE rounding). At rot_dim 256
// the halves are whole boxes: boxes 0 and 1 are `x0` (lanes 0..127), boxes 2 and 3 `x1`, and the
// table's boxes 0, 1 are `cos` and 2, 3 `sin`; box b < 2 is rotated with box b + 2 by table boxes
// b and b + 2, the same maps on `rt_fl<16, 64>` boxes, each box stored back after. Lane 0 then
// TMA-stores every box to `out`: a pass-through box as it was loaded, so those lanes never leave
// shared memory as anything but the loaded bytes. Every tile is plain, never swizzled: a box past
// the arrays' rows (the last block's padding) reads zeros and stores nothing (TK-1's R1 gate,
// `tests/tma_bounds.rs`), so the output's tail is never written; no box is ever past a column
// edge (`cols` and the table's `rot_dim` are whole boxes).
//
// Position 0 (c = 1, s = 0) is exact: `x0·1 − x1·0 = x0` and `x1·1 + x0·0 = x1` in f32, and under
// an FMA contraction too, signed zeros included. Each product is its own `warp::mul` and the sum
// or difference its own map; built with `--use_fast_math` (build.rs), nvcc may still contract a
// product into the add or subtract after inlining, as it could in the plain-CUDA kernel's
// `x0·c − x1·s`. So the rotated lanes may differ from that kernel by about one f32 rounding before
// the bf16 one, which the RoPE bound's `2⁻¹⁶` term covers. Each element depends only on its own
// row's inputs, so the bits do not depend on the launch shape.
//
// Shared memory: at rot_dim 64, 4 stages × 2 KiB and the table's 2 KiB box, plus 1,024 bytes of
// alignment slack = 11,264 bytes; at rot_dim 256, 2 stages × 8 KiB and the table's 8 KiB, plus the
// slack = 25,600 bytes. Both are dynamic shared memory under the 48 KiB default, so no one-time
// setup.
//
// Gemma 4's global layers (#885's 13b-3) rotate 128 of each 512-lane head in the half-head
// pairing (HF `_compute_proportional_rope_parameters`, metal's `ROPE_PAIR_OFF = HEAD_DIM/2`):
// lane d < 64 pairs with lane d + 256, and lanes 64..255 and 320..511 pass through. That is the
// `<128, true>` instantiation (`HALF_HEAD`): its unit is a whole head, 8 boxes, box 0 rotated with
// box 4 by the table's box 0 (cos) and box 1 (sin), the rot_dim-256 path's maps on one box pair,
// and boxes 1..3 and 5..7 stored as loaded. The table is `[rows, 128]`, two boxes, its frequencies
// over the head dim (`RotaryTable::freq_dim`, the host's business). Shared memory: 2 stages × 16
// KiB and the table's 4 KiB, plus the slack = 37,888 bytes, under the default too. The `<64,
// false>` and `<256, false>` instantiations are the code above, unchanged: `HALF_HEAD` only picks
// the unit's and the pair's box counts, which are those instantiations' own constants.

#include "kittens.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// The launcher's return codes that are not a `cudaError_t` (which is non-negative).
// `src/kernels.rs` mirrors them.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::ROPE_BF16_KERNEL_HEAD_MULTIPLE`: `head_dim` is whole boxes, so every head starts at a box.
constexpr uint32_t HEAD_MULTIPLE = 64;
constexpr int ROWS = 16;
constexpr int BOX_COLS = 64;
constexpr int NUM_THREADS = kittens::WARP_THREADS;

using box = st_bf<ROWS, BOX_COLS, false>;
using box_gl = gl<bf16, 1, 1, -1, -1, box>;
using box_rt = rt_fl<ROWS, BOX_COLS>;
using half_rt = rt_fl<ROWS, BOX_COLS / 2>;
static_assert(half_rt::width == 2 && box_rt::width == 4,
              "each half of a 64-lane rotated width is two 16-column base tiles");

// The half-head instantiation's one head dim: 8 boxes, a whole head per unit.
constexpr int HALF_HEAD_DIM = 512;
// One instantiation per rotated width and pairing (`step::ROPE_BF16_KERNEL_SHAPES`): `BOXES` boxes
// per unit (`ROT` lanes, or a whole head for the half-head pairing), the table's `TABLE_BOXES`,
// the box offset `PAIR_OFF` of a rotated box's partner, and the ring's `STAGES` units.
template <int ROT, bool HALF_HEAD>
struct rope_shape {
    static_assert((ROT == 64 || ROT == 256) != HALF_HEAD && (!HALF_HEAD || ROT == 128),
                  "the instantiated rotated widths and pairings");
    static constexpr int TABLE_BOXES = ROT / BOX_COLS;
    static constexpr int BOXES = HALF_HEAD ? HALF_HEAD_DIM / BOX_COLS : TABLE_BOXES;
    static constexpr int PAIR_OFF = HALF_HEAD ? BOXES / 2 : TABLE_BOXES / 2;
    static constexpr int STAGES = ROT == 64 ? 4 : 2;
    static constexpr size_t SMEM_BOXES = (STAGES * BOXES + TABLE_BOXES) * sizeof(box);
    // The boxes plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
    static constexpr int SMEM_BYTES = int(SMEM_BOXES) + 1024;
};
static_assert(rope_shape<64, false>::SMEM_BOXES == 10240, "4 stages x 2 KiB, plus the table's box");
static_assert(rope_shape<256, false>::SMEM_BOXES == 24576,
              "2 stages x 8 KiB, plus the table's 8 KiB");
static_assert(rope_shape<128, true>::SMEM_BOXES == 36864,
              "2 stages x 16 KiB (a 512-lane head), plus the table's 4 KiB");
static_assert(rope_shape<64, false>::SMEM_BYTES <= 48 * 1024 &&
                  rope_shape<256, false>::SMEM_BYTES <= 48 * 1024 &&
                  rope_shape<128, true>::SMEM_BYTES <= 48 * 1024,
              "under the default dynamic shared memory limit");

struct rope_globals {
    box_gl x;
    box_gl table;
    box_gl out;
    // The units of a row, and of a head (one unit is `rot_dim` lanes, or a half-head head).
    int row_units;
    int head_units;
};

template <int ROT, bool HALF_HEAD>
__global__ __launch_bounds__(NUM_THREADS, 1) void rope_bf16_ker(const __grid_constant__ rope_globals g) {
    constexpr int BOXES = rope_shape<ROT, HALF_HEAD>::BOXES;
    constexpr int TABLE_BOXES = rope_shape<ROT, HALF_HEAD>::TABLE_BOXES;
    constexpr int STAGES = rope_shape<ROT, HALF_HEAD>::STAGES;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    box(&x_smem)[STAGES][BOXES] = al.allocate<box, STAGES, BOXES>();
    box(&t_smem)[TABLE_BOXES] = al.allocate<box, TABLE_BOXES>();
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

    warp::tma::expect_bytes(table_arrived, TABLE_BOXES * sizeof(box));
    for (int b = 0; b < TABLE_BOXES; b++) {
        warp::tma::load_async(t_smem[b], g.table, coord<box>{0, 0, row_block, b}, table_arrived);
    }
    // Unit `i` of the row block (boxes `i · BOXES ..`) into stage `i % STAGES`.
    auto issue = [&](int i) {
        const int stage = i % STAGES;
        warp::tma::expect_bytes(arrived[stage], BOXES * sizeof(box));
        for (int b = 0; b < BOXES; b++) {
            warp::tma::load_async(x_smem[stage][b], g.x, coord<box>{0, 0, row_block, i * BOXES + b},
                                  arrived[stage]);
        }
    };
    for (int i = 0; i < STAGES - 1 && i < g.row_units; i++) {
        issue(i);
    }

    wait(table_arrived, 0);
    // At rot_dim 64 the table's one box is held in registers, split into its halves; at 256 its
    // four boxes stay in shared memory and are loaded per pair (four `rt_fl<16, 64>` held across
    // the walk would be 128 registers a lane).
    [[maybe_unused]] half_rt c, s;
    if constexpr (ROT == 64) {
        box_rt t;
        warp::load(t, t_smem[0]);
        c.tiles[0][0] = t.tiles[0][0];
        c.tiles[0][1] = t.tiles[0][1];
        s.tiles[0][0] = t.tiles[0][2];
        s.tiles[0][1] = t.tiles[0][3];
    }

    for (int i = 0; i < g.row_units; i++) {
        const int stage = i % STAGES;
        // Unit `i + STAGES − 1` lands in the stage unit `i − 1` was stored from, the newest store:
        // it must have read the stage before that load overwrites it.
        warp::tma::store_async_read_wait();
        __syncwarp();
        if (i + STAGES - 1 < g.row_units) {
            issue(i + STAGES - 1);
        }
        wait(arrived[stage], (i / STAGES) % 2);
        if (i % g.head_units == 0) {
            if constexpr (ROT == 64) {
                box_rt x;
                warp::load(x, x_smem[stage][0]);
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
                warp::store(x_smem[stage][0], x);
            } else {
                // Box b < PAIR (lanes 64b .. 64b + 63) pairs with box b + PAIR_OFF, by the
                // table's box b (cos) and b + PAIR (sin): at rot_dim 256 PAIR = PAIR_OFF = 2; in
                // the half-head pairing PAIR = 1 and PAIR_OFF = 4. The rot_dim-64 path's products,
                // difference and sum, each its own map, ordered so the cos box is dead before the
                // sin box is loaded into the same registers: five `rt_fl<16, 64>` live at most.
                constexpr int PAIR = TABLE_BOXES / 2;
                constexpr int PAIR_OFF = rope_shape<ROT, HALF_HEAD>::PAIR_OFF;
                for (int b = 0; b < PAIR; b++) {
                    box_rt x0, x1, t, y0, y1;
                    warp::load(x0, x_smem[stage][b]);
                    warp::load(x1, x_smem[stage][b + PAIR_OFF]);
                    warp::load(t, t_smem[b]);
                    warp::mul(y0, x0, t);
                    warp::mul(y1, x1, t);
                    warp::load(t, t_smem[b + PAIR]);
                    // y0 = x0·c − x1·s.
                    warp::mul(x1, x1, t);
                    warp::sub(y0, y0, x1);
                    // y1 = x1·c + x0·s.
                    warp::mul(x0, x0, t);
                    warp::add(y1, y1, x0);
                    warp::store(x_smem[stage][b], y0);
                    warp::store(x_smem[stage][b + PAIR_OFF], y1);
                }
            }
            __syncwarp();
        }
        for (int b = 0; b < BOXES; b++) {
            warp::tma::store_async(g.out, x_smem[stage][b], coord<box>{0, 0, row_block, i * BOXES + b});
        }
    }
    warp::tma::store_async_wait();
}

// Launch the `<ROT, HALF_HEAD>` instantiation over the launcher's checked shape.
template <int ROT, bool HALF_HEAD>
int launch_rope(bf16* xp, bf16* tp, bf16* op, uint32_t rows, uint32_t cols, uint32_t head_dim,
                cudaStream_t stream) {
    // The lanes of one unit: `ROT`, or a whole half-head head.
    constexpr uint32_t UNIT = uint32_t(rope_shape<ROT, HALF_HEAD>::BOXES * BOX_COLS);
    // Building each `gl` encodes its tensor map into host memory; nothing is allocated on the
    // device. Every row stride is a multiple of 128 bytes.
    box_gl xg{xp, nullptr, nullptr, size_t(rows), size_t(cols)};
    box_gl tg{tp, nullptr, nullptr, size_t(rows), size_t(ROT)};
    box_gl og{op, nullptr, nullptr, size_t(rows), size_t(cols)};
    rope_globals G{xg, tg, og, int(cols / UNIT), int(head_dim / UNIT)};
    void* args[] = {&G};
    const dim3 grid((rows + uint32_t(ROWS) - 1) / uint32_t(ROWS)), block(NUM_THREADS);
    return static_cast<int>(cudaLaunchKernel(rope_bf16_ker<ROT, HALF_HEAD>, grid, block, args,
                                             rope_shape<ROT, HALF_HEAD>::SMEM_BYTES, stream));
}

}  // namespace

// The pairings (`step::RopePairing`, in its declaration order): lane d pairs with d + rot_dim/2,
// or with d + head_dim/2.
constexpr uint32_t PAIRING_HALF_ROTATED = 0;
constexpr uint32_t PAIRING_HALF_HEAD = 1;

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value =
// the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// Refused before any pointer is touched: `rows`, `cols` or `head_dim` zero; `(rot_dim, pairing)`
// not (64, half-rotated), (256, half-rotated) or (128, half-head) (`step::ROPE_BF16_KERNEL_SHAPES`,
// the instantiations); `head_dim` not a multiple of 64 (`step::ROPE_BF16_KERNEL_HEAD_MULTIPLE`,
// whole boxes), or not a multiple of `rot_dim` (whole units, which at rot_dim 64 the 64 rule
// already is); a half-head `head_dim` not 512, the instantiation's head; `cols` not a multiple of
// `head_dim`; `rows · cols` above INT_MAX. `x`, `table` and `out` must be 16-byte-aligned device
// pointers to `rows·cols`, `rows·rot_dim` and `rows·cols` bf16, `out` overlapping neither input.
// Its dynamic shared memory is under the 48 KiB default, so no one-time setup. The launch is
// asynchronous on `stream`; the caller owns it and synchronizes.
extern "C" int scratchy_cuda_rope_paired_bf16(const void* x, const void* table, void* out,
                                              uint32_t rows, uint32_t cols, uint32_t head_dim,
                                              uint32_t rot_dim, uint32_t pairing,
                                              void* stream /* cudaStream_t; null = default */) {
    const bool half_rotated = pairing == PAIRING_HALF_ROTATED && (rot_dim == 64 || rot_dim == 256);
    const bool half_head =
        pairing == PAIRING_HALF_HEAD && rot_dim == 128 && head_dim == uint32_t(HALF_HEAD_DIM);
    if (rows == 0 || cols == 0 || head_dim == 0 || !(half_rotated || half_head) ||
        head_dim % HEAD_MULTIPLE != 0 || head_dim % rot_dim != 0 || cols % head_dim != 0 ||
        uint64_t(rows) * uint64_t(cols) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        bf16* xp = static_cast<bf16*>(const_cast<void*>(x));
        bf16* tp = static_cast<bf16*>(const_cast<void*>(table));
        bf16* op = static_cast<bf16*>(out);
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        if (half_head) {
            return launch_rope<128, true>(xp, tp, op, rows, cols, head_dim, s);
        }
        return rot_dim == 64 ? launch_rope<64, false>(xp, tp, op, rows, cols, head_dim, s)
                             : launch_rope<256, false>(xp, tp, op, rows, cols, head_dim, s);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// The half-rotated rope, `scratchy_cuda_rope_paired_bf16` with `pairing` half-rotated: its codes
// and refusals, so `rot_dim` not 64 or 256 is refused (the half-rotated entries of
// `step::ROPE_BF16_KERNEL_SHAPES`).
extern "C" int scratchy_cuda_rope_bf16(const void* x, const void* table, void* out, uint32_t rows,
                                       uint32_t cols, uint32_t head_dim, uint32_t rot_dim,
                                       void* stream /* cudaStream_t; null = default stream */) {
    return scratchy_cuda_rope_paired_bf16(x, table, out, rows, cols, head_dim, rot_dim,
                                          PAIRING_HALF_ROTATED, stream);
}
