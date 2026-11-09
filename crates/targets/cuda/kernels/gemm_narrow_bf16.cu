// SPDX-License-Identifier: Apache-2.0
//
// Narrow bf16 GEMM for the H100: C[m, n] = A[m, k] · B[n, k]ᵀ, all row-major bf16, accumulated in
// f32 and rounded once to bf16 (round-to-nearest-even), for `n` of at most 64 output columns. The
// gated delta net's `a` and `b` projections are `[hidden, linear_num_value_heads]`: 16 columns in
// qwen3.5-0.8b, 32 in 9b, which the tiled GEMM (`gemm_bf16.cu`, 256-column tiles) refuses.
//
// ThunderKittens' GEMM structure (PLAN-tk100 §3.2), as `gemv_bf16.cu`'s: a producer warp loads
// tiles with TMA, `tma::expect_bytes` before every load, into a ring of `STAGES` stages, and two
// consumer warpgroups multiply them with WGMMA (`warpgroup::mma_ABt` against the `[N, K]` B,
// `mma_async_wait` before each stage goes back to the producer). One CTA per 128 output rows, 64
// per consumer warpgroup; the output tile is 64 columns wide and covers all of B's `n ≤ 64` rows,
// whose rows `n .. 63` TMA reads as zeros. So for n = 16, 48 of the 64 tensor-core columns
// multiply zeros, and for n = 32, 32: the op is bound by reading A once (`m · k` bf16), so that
// costs issue slots, not bandwidth. Two consumer warpgroups, not one, as `gemm_bf16.cu`'s
// `M_BLOCK = 2`: chosen by reading, not by measured time; `tests/gemm_bf16.rs` prints the time.
//
// A hand-built producer/consumer kernel, structured as `attn_prefill_bf16.cu`, not TK's `lcf`
// template: the K tail below has no place in `lcf`'s `compute`.
//
// Which shared tile is swizzled follows from TMA's edges (the GKE probe's R1 at c8d80fda, re-run
// by `tests/tma_bounds.rs`; see `gemv_bf16.cu`): rows past an array's edge read zeros on a
// swizzled tile, columns past it only on a plain one; WGMMA reads swizzled tiles only. So:
//   - the full K tiles (`k0 + 64 ≤ k`) of A and B are swizzled `st_bf<64, 64>`, multiplied by
//     WGMMA;
//   - the K tail, when `k` is not a multiple of 64, is a plain `st_bf<64, 64, false>` of each
//     warpgroup's A rows and one of B, zero past `k`, multiplied by warp MMA from registers into
//     the same `rt_fl<16, 64>` accumulator after the last WGMMA's wait (`warpgroup::load` of each
//     warp's 16 rows of A, `warp::load` of all of B's tail, `warp::mma_ABt`), whole tiles only;
//   - a consumer warpgroup whose 64 rows are all past `m` (the last CTA's second, at any `m ≤ 64`)
//     loads, multiplies and stores nothing, so no box lies wholly past an edge;
//   - C is a plain `st_bf<64, 64, false>` per consumer warpgroup, never swizzled: `warpgroup::store`
//     (the one RNE rounding to bf16), then one lane's `tma::store_async`, which issues its own
//     async-proxy fence. A plain store drops the columns `n .. 63` and the rows past `m`; a
//     swizzled one writes the columns past `n` (R1). Its row stride, `2n` bytes with `n` a
//     multiple of 8, is a multiple of 16, as its tensor map needs.
// No real model's `k` has a tail; the test shape `k` = 8 does. The zeros past `k`, in B's rows
// past `n` and in A's rows past `m` add exactly 0.
//
// A WGMMA output row depends only on its own row of A and on B, accumulated in a fixed K order,
// with no atomics: the same row gives the same bits wherever it sits in A and at any `m`
// (`an_output_row_does_not_depend_on_where_its_input_row_sits`, `the_narrow_gemm_at_one_row_is_
// its_row_at_128`), so decode's GDN `a`/`b` projections are bit-identical to prefill's rows.
//
// Shared memory: `STAGES` × (two A tiles and one B tile), the three plain tail tiles and the two
// plain C tiles, 8 KiB each, plus 1,024 bytes of alignment slack = 140,288 bytes of dynamic shared
// memory, set once by `scratchy_cuda_gemm_narrow_bf16_abt_prepare` (in `PREPARE`). The
// semaphores are static shared memory.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::GEMM_NARROW_BF16_MULTIPLE`: `n` and `k` are multiples of it, so every row stride is a
// multiple of 16 bytes, as the tensor maps need.
constexpr uint32_t VEC = 8;
// `step::GEMM_NARROW_BF16_MAX_N`: B's rows, all in one 64-row tile.
constexpr uint32_t MAX_N = 64;
// Rows and columns of a tile, and K per tile.
constexpr int TILE = 64;
constexpr int STAGES = 4;
constexpr int CONSUMER_WARPGROUPS = 2;
constexpr int PRODUCER_WARPGROUP = CONSUMER_WARPGROUPS;
constexpr int NUM_THREADS = (CONSUMER_WARPGROUPS + 1) * kittens::WARPGROUP_THREADS;
// Output rows per CTA.
constexpr int CTA_ROWS = CONSUMER_WARPGROUPS * TILE;
static_assert(MAX_N == uint32_t(TILE), "every column of C is in one tile");

using tile = st_bf<TILE, TILE>;
using plain_tile = st_bf<TILE, TILE, false>;
// A `[m, k]` and B `[n, k]`, each with both tensor maps: swizzled for the full K tiles, plain for
// the tail.
using operand_gl = gl<bf16, 1, 1, -1, -1, tile, plain_tile>;
// C `[m, n]`: the plain tensor map only.
using output_gl = gl<bf16, 1, 1, -1, -1, plain_tile>;
// FP8 S8 W4a, the one-row prologue norm: A's row as 256-element chunks (their vector tensor map),
// the norm's gain `[1, k]` (no map), and the shared row (a normed `k` at most `NORM_CHUNKS`
// chunks; the norm's tile views of chunk `t` read the 15 after it, `scratchy_ops.cuh`).
using chunk_sv = sv_bf<256>;
using row_gl = gl<bf16, 1, 1, -1, -1, chunk_sv>;
using gain_gl = gl<bf16, 1, 1, -1, -1>;
constexpr int NORM_CHUNKS = 24;
constexpr int ROW_CHUNKS = NORM_CHUNKS + 15;

struct narrow_globals {
    operand_gl a;
    operand_gl b;
    output_gl c;
    // The rows of A and C; the full K tiles, `k / 64`; whether a K tail follows them.
    int m;
    int k_tiles;
    bool tail;
    // FP8 S8 W4a: at `m == 1`, the prologue norm when `norm` (A's row `arow` normalised in shared
    // memory, `rms_norm_one_row_ker`'s bits, then written over row 0 of each A tile before its
    // WGMMA), its `gain`, `eps`, `offset` and `cols` (`k`), and the row's `chunks`.
    row_gl arow;
    gain_gl gain;
    float eps;
    float offset;
    float cols;
    int chunks;
    int norm;
};

constexpr size_t SMEM_TILES = STAGES * (CONSUMER_WARPGROUPS + 1) * sizeof(tile) +
                              (CONSUMER_WARPGROUPS + 1) * sizeof(plain_tile) +
                              CONSUMER_WARPGROUPS * sizeof(plain_tile);
static_assert(SMEM_TILES == 139264, "4 stages of two A and one B, three tail tiles, two C tiles");
// The prologue norm's row (FP8 S8 W4a), after the tiles, so their offsets are unchanged.
constexpr size_t SMEM_ROW = size_t(ROW_CHUNKS) * sizeof(chunk_sv);
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int NARROW_SMEM_BYTES = int(SMEM_TILES + SMEM_ROW) + 1024;
static_assert(NARROW_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

// A narrow CTA's shared tiles: the A and B ring, the tails and the C tiles, in the order the
// one-GEMM kernel has always carved them.
struct narrow_smem {
    tile (&a_smem)[STAGES][CONSUMER_WARPGROUPS];
    tile (&b_smem)[STAGES];
    plain_tile (&a_tail)[CONSUMER_WARPGROUPS];
    plain_tile& b_tail;
    plain_tile (&c_smem)[CONSUMER_WARPGROUPS];
    // The prologue norm's row and the four warp parts (FP8 S8 W4a).
    chunk_sv (&row)[ROW_CHUNKS];
    scratchy::ops::norm_parts (&parts)[4];
};

// One CTA of one narrow GEMM over `g`'s views and `sm`'s tiles: output rows `128 · row_cta ..`.
// The grouped kernel runs the same CTA for each member, so each member is its own GEMM's bits.
__device__ inline void narrow_cta(const narrow_globals& g, const narrow_smem& sm, int row_cta) {
    auto& a_smem = sm.a_smem;
    auto& b_smem = sm.b_smem;
    auto& a_tail = sm.a_tail;
    plain_tile& b_tail = sm.b_tail;
    auto& c_smem = sm.c_smem;

    // The CTA's first 64-row tile of A and C, and how many of its consumer warpgroups have a real
    // row: the last CTA's second warpgroup may have none (any `m ≤ 64`, decode's `m = 1`
    // included). Such a warpgroup loads, multiplies and stores nothing, so no TMA box lies wholly
    // past the array's edge: the GKE probe measured partly-outside boxes only.
    const int row_tile0 = row_cta * CONSUMER_WARPGROUPS;
    const int live = min(CONSUMER_WARPGROUPS, (g.m - row_tile0 * TILE + TILE - 1) / TILE);
    const int warpgroupid = threadIdx.x / kittens::WARPGROUP_THREADS;

    __shared__ kittens::semaphore arrived[STAGES], done[STAGES], tail_arrived, row_arrived;
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_semaphore(arrived[s], 0, 1);
            init_semaphore(done[s], live, 0);
        }
        init_semaphore(tail_arrived, 0, 1);
        init_semaphore(row_arrived, 0, 1);
    }
    __syncthreads();
    // FP8 S8 W4a: the prologue norm, every thread of the CTA (`m == 1`: the one CTA). Warp 0
    // TMA-loads A's row as chunks; the CTA normalises it in place (`scratchy_ops.cuh`'s
    // `row_norm_inv` and `row_norm_apply`, `rms_norm_one_row_ker`'s bits). A uniform branch.
    if (g.norm) {
        const int w = kittens::warpid();
        if (w == 0) {
            warp::tma::expect_bytes(row_arrived, uint32_t(g.chunks * sizeof(chunk_sv)));
            for (int c = 0; c < g.chunks; c++) {
                warp::tma::load_async(sm.row[c], g.arow, {0, 0, 0, c}, row_arrived);
            }
        }
        wait(row_arrived, 0);
        const float inv = scratchy::ops::row_norm_inv(sm.row, sm.parts, g.chunks, g.cols, g.eps, w,
                                                      [] { __syncthreads(); });
        scratchy::ops::row_norm_apply(sm.row, g.gain, g.chunks, inv, g.offset, w,
                                      NUM_THREADS / kittens::WARP_THREADS);
        __syncthreads();
    }

    if (warpgroupid == PRODUCER_WARPGROUP) {
        // The producer: one warp loads the tails first (their tiles are their own), then K tile
        // `j` into stage `j % STAGES`, after every live consumer released that stage's previous
        // tile. Only the live consumers' A tiles are loaded.
        if (warpgroup::warpid() == 0) {
            if (g.tail) {
                warp::tma::expect_bytes(tail_arrived, (live + 1) * sizeof(plain_tile));
                for (int w = 0; w < live; w++) {
                    coord<plain_tile> at_a = {0, 0, row_tile0 + w, g.k_tiles};
                    warp::tma::load_async(a_tail[w], g.a, at_a, tail_arrived);
                }
                coord<plain_tile> at_b = {0, 0, 0, g.k_tiles};
                warp::tma::load_async(b_tail, g.b, at_b, tail_arrived);
            }
            for (int j = 0; j < g.k_tiles; j++) {
                const int stage = j % STAGES;
                if (j >= STAGES) {
                    wait(done[stage], ((j - STAGES) / STAGES) % 2);
                }
                warp::tma::expect_bytes(arrived[stage], (live + 1) * sizeof(tile));
                for (int w = 0; w < live; w++) {
                    coord<tile> at_a = {0, 0, row_tile0 + w, j};
                    warp::tma::load_async(a_smem[stage][w], g.a, at_a, arrived[stage]);
                }
                coord<tile> at_b = {0, 0, 0, j};
                warp::tma::load_async(b_smem[stage], g.b, at_b, arrived[stage]);
            }
        }
    } else if (warpgroupid < live) {
        // Consumer `warpgroupid`: its 64 rows of C, 16 per warp.
        rt_fl<16, TILE> acc;
        warp::zero(acc);
        for (int j = 0; j < g.k_tiles; j++) {
            const int stage = j % STAGES;
            wait(arrived[stage], (j / STAGES) % 2);
            if (g.norm) {
                // FP8 S8 W4a: row 0 of the A tile (the one row) is the normed row's K tile `j`,
                // written through the tile's swizzled accessor; then the generic-proxy writes are
                // made visible to the tensor cores (`warpgroup::mma_fence`, TK's `wgmma.fence` and
                // `fence.proxy.async`) and the warpgroup meets, as S5's widened K tiles.
                const int t = warpgroup::laneid();
                if (t < TILE) {
                    const int e = j * TILE + t;
                    a_smem[stage][warpgroupid][{0, t}] = sm.row[e / 256][e % 256];
                }
                warpgroup::mma_fence();
                warpgroup::sync(warpgroupid + 1);
            }
            warpgroup::mma_ABt(acc, a_smem[stage][warpgroupid], b_smem[stage]);
            // The WGMMA reads shared memory and writes `acc` asynchronously: wait for it before
            // releasing the stage, and before anything else reads `acc`.
            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) {
                arrive(done[stage], 1);
            }
        }
        if (g.tail) {
            wait(tail_arrived, 0);
            rt_bf<16, TILE> a_reg;
            rt_bf<TILE, TILE> b_reg;
            warpgroup::load(a_reg, a_tail[warpgroupid]);
            warp::load(b_reg, b_tail);
            warp::mma_ABt(acc, a_reg, b_reg, acc);
        }
        warpgroup::store(c_smem[warpgroupid], acc);
        // Each warpgroup's own named barrier (0 is `__syncthreads`).
        warpgroup::sync(warpgroupid + 1);
        if (warpgroup::warpid() == 0) {
            coord<plain_tile> at = {0, 0, row_tile0 + warpgroupid, 0};
            warp::tma::store_async(g.c, c_smem[warpgroupid], at);
        }
        warp::tma::store_async_wait();
    }
}

__global__ __launch_bounds__(NUM_THREADS, 1) void gemm_narrow_bf16_ker(
    const __grid_constant__ narrow_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    st_bf<TILE, TILE>(&a_smem)[STAGES][CONSUMER_WARPGROUPS] =
        al.allocate<tile, STAGES, CONSUMER_WARPGROUPS>();
    st_bf<TILE, TILE>(&b_smem)[STAGES] = al.allocate<tile, STAGES>();
    st_bf<TILE, TILE, false>(&a_tail)[CONSUMER_WARPGROUPS] =
        al.allocate<plain_tile, CONSUMER_WARPGROUPS>();
    st_bf<TILE, TILE, false>& b_tail = al.allocate<plain_tile>();
    st_bf<TILE, TILE, false>(&c_smem)[CONSUMER_WARPGROUPS] =
        al.allocate<plain_tile, CONSUMER_WARPGROUPS>();
    chunk_sv(&row_s)[ROW_CHUNKS] = al.allocate<chunk_sv, ROW_CHUNKS>();
    __shared__ sv_fl<16> parts[4];
    narrow_cta(g, narrow_smem{a_smem, b_smem, a_tail, b_tail, c_smem, row_s, parts}, blockIdx.x);
}

// The grouped narrow GEMM (#1049): up to three narrow GEMMs of one `A`, one launch. Member `i` is
// `m[i]`, a narrow GEMM's views and counts; the grid's y is the member, its x the row CTA. An
// absent third member repeats the second's views; the grid has no y for it.
struct narrow_group_globals {
    narrow_globals m0;
    narrow_globals m1;
    narrow_globals m2;
};

__global__ __launch_bounds__(NUM_THREADS, 1) void gemm_narrow_group_bf16_ker(
    const __grid_constant__ narrow_group_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    st_bf<TILE, TILE>(&a_smem)[STAGES][CONSUMER_WARPGROUPS] =
        al.allocate<tile, STAGES, CONSUMER_WARPGROUPS>();
    st_bf<TILE, TILE>(&b_smem)[STAGES] = al.allocate<tile, STAGES>();
    st_bf<TILE, TILE, false>(&a_tail)[CONSUMER_WARPGROUPS] =
        al.allocate<plain_tile, CONSUMER_WARPGROUPS>();
    st_bf<TILE, TILE, false>& b_tail = al.allocate<plain_tile>();
    st_bf<TILE, TILE, false>(&c_smem)[CONSUMER_WARPGROUPS] =
        al.allocate<plain_tile, CONSUMER_WARPGROUPS>();
    chunk_sv(&row_s)[ROW_CHUNKS] = al.allocate<chunk_sv, ROW_CHUNKS>();
    __shared__ sv_fl<16> parts[4];
    const narrow_globals& m = blockIdx.y == 2 ? g.m2 : blockIdx.y == 1 ? g.m1 : g.m0;
    narrow_cta(m, narrow_smem{a_smem, b_smem, a_tail, b_tail, c_smem, row_s, parts}, blockIdx.x);
}

// M7 (#857): the one-column projection, `c[r] = a[r, :] · b[0, :]` (a shared expert's gate,
// `n = 1`). C's row stride is 2 bytes, below a tensor map's 16, so this is ThunderKittens' warp
// vector (PLAN-tk100 §2 S4), not the tile GEMM above: one warp per row, 8 rows per CTA. The row
// of `a` and B's one row are walked in `k / 256` chunks of naive `rv_fl<256>` (`warp::load`, bf16
// to f32, exact), each chunk's products `warp::mul` then accumulated by `warp::add` in chunk
// order, then summed by `warp::sum` (each lane's elements in order, then a fixed shuffle tree into
// lane 0 and a broadcast). Lane 0 stores the row's one bf16 through `gl::operator[]`, rounded once
// (`base_types::convertor<bf16, float>`, round-to-nearest-even). A fixed order, no atomics: a row's
// bits do not depend on where it sits in `a`.
constexpr int ONE_WARPS = 8;
constexpr int ONE_THREADS = ONE_WARPS * kittens::WARP_THREADS;
// `step::GEMM_NARROW_BF16_ONE_K_MULTIPLE`: one chunk, a naive `rv_fl<256>`.
constexpr int ONE_CHUNK = 256;

using one_vec = rv_fl<ONE_CHUNK>;
using one_in_gl = gl<bf16, 1, 1, -1, -1>;
// `c`, `[m, 1]` bf16: written element by element.
using one_out_gl = gl<bf16, 1, 1, -1, 1>;

struct one_globals {
    one_in_gl a;
    one_in_gl b;
    one_out_gl c;
    int m;
    int chunks;
    // FP8 S8 W4a: at `m == 1`, the prologue norm when `norm` (the narrow GEMM's, above).
    gain_gl gain;
    float eps;
    float offset;
    float cols;
    int norm;
};

__global__ __launch_bounds__(ONE_THREADS) void gemm_narrow_bf16_n1_ker(
    const __grid_constant__ one_globals g) {
    __shared__ sv_bf<ONE_CHUNK> row_s[ROW_CHUNKS];
    __shared__ sv_fl<16> parts[4];
    // FP8 S8 W4a: the prologue norm, every warp of the one CTA (`m == 1`), before any warp leaves
    // (W1's FFMA lesson: no early return before a value another warp reads). The row is loaded
    // into shared memory (`warp::load`, `warp::store`: exact), normalised in place.
    if (g.norm) {
        const int wid = kittens::warpid();
        for (int ch = wid; ch < g.chunks; ch += ONE_WARPS) {
            one_vec v;
            warp::load(v, g.a, {0, 0, 0, ch});
            warp::store(row_s[ch], v);
        }
        __syncthreads();
        const float inv = scratchy::ops::row_norm_inv(row_s, parts, g.chunks, g.cols, g.eps, wid,
                                                      [] { __syncthreads(); });
        scratchy::ops::row_norm_apply(row_s, g.gain, g.chunks, inv, g.offset, wid, ONE_WARPS);
        __syncthreads();
    }
    const int row = blockIdx.x * ONE_WARPS + kittens::warpid();
    if (row >= g.m) {
        return;
    }
    one_vec acc, x, w;
    warp::zero(acc);
    for (int ch = 0; ch < g.chunks; ++ch) {
        if (g.norm) {
            warp::load(x, row_s[ch]);
        } else {
            warp::load(x, g.a, {0, 0, row, ch});
        }
        warp::load(w, g.b, {0, 0, 0, ch});
        warp::mul(x, x, w);
        warp::add(acc, acc, x);
    }
    const float dot = warp::sum(acc);
    if (warp::laneid() == 0) {
        g.c[{0, 0, row, 0}] = base_types::convertor<bf16, float>::convert(dot);
    }
}

}  // namespace

namespace {

// The prologue norm a launcher was given (FP8 S8 W4a): `gain` (null: none), `eps`, `offset`.
struct norm_end {
    const void* gain = nullptr;
    float eps = 0.0f;
    float offset = 0.0f;
};
constexpr norm_end NO_NORM{nullptr, 0.0f, 0.0f};

// The norm's rule: one row (`m == 1`), `k` a multiple of 256 of at most `NORM_CHUNKS` chunks,
// `eps` finite and positive.
bool norm_shape(const norm_end& e, uint32_t m, uint32_t k) {
    return e.gain == nullptr || (m == 1 && k % 256 == 0 && k / 256 <= uint32_t(NORM_CHUNKS) &&
                                 e.eps > 0.0f && e.eps <= 3.0e38f);
}

}  // namespace

// The kernel's one-time setup: allow it `NARROW_SMEM_BYTES` of dynamic shared memory on the
// current device (the default limit is 48 KB). Run once per device before any launch
// (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch.
//
// 0 = done; any other value = the cudaError_t of setting the attribute.
extern "C" int scratchy_cuda_gemm_narrow_bf16_abt_prepare(void) {
    // The `n = 1` warp-vector kernel needs no attribute; querying its attributes loads it, so a
    // worker's graph capture never loads it mid-capture.
    cudaFuncAttributes attr;
    if (const cudaError_t e = cudaFuncGetAttributes(&attr, gemm_narrow_bf16_n1_ker); e != cudaSuccess) {
        return static_cast<int>(e);
    }
    return static_cast<int>(cudaFuncSetAttribute(
        gemm_narrow_bf16_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, NARROW_SMEM_BYTES));
}

// The grouped kernel's one-time setup (#1049): `scratchy_cuda_gemm_narrow_bf16_abt_prepare`'s
// attribute, which loads it before any graph capture.
//
// 0 = done; any other value = the cudaError_t of setting the attribute.
extern "C" int scratchy_cuda_gemm_narrow_group_bf16_abt_prepare(void) {
    return static_cast<int>(cudaFuncSetAttribute(gemm_narrow_group_bf16_ker,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 NARROW_SMEM_BYTES));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value
// = the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `m` must be non-zero, `n` 1 or a non-zero multiple of 8 and at most 64, `k` a non-zero multiple
// of 8 (of 256 when `n` is 1, `step::GEMM_NARROW_BF16_ONE_K_MULTIPLE`: whole warp-vector chunks),
// and `m` and `k` at most INT_MAX; all are checked before any pointer is touched. `a`, `b` and `c`
// must be 16-byte-aligned device pointers to `m·k`, `n·k` and `m·n` bf16, `c` overlapping neither
// input. `n` = 1 runs the warp-vector kernel above, with no shared memory; any other `n` the tile
// GEMM, for which `scratchy_cuda_gemm_narrow_bf16_abt_prepare` must have succeeded on the current
// device. The launch is asynchronous on `stream`.
namespace {

// The tile GEMM's rule, for one narrow member: `n` a non-zero multiple of 8 at most 64 (not the
// `n = 1` warp-vector form), with the shared `m` and `k`.
bool narrow_tile_shape(uint32_t m, uint32_t n, uint32_t k) {
    return m != 0 && m <= uint32_t(INT_MAX) && n != 0 && n % VEC == 0 && n <= MAX_N && k != 0 &&
           k % VEC == 0 && k <= uint32_t(INT_MAX);
}

// Member `i`'s views and counts, as `scratchy_cuda_gemm_narrow_bf16_abt` builds them. It can
// throw (TK's host code).
narrow_globals narrow_member(const void* a, const void* b, void* c, uint32_t m, uint32_t n,
                             uint32_t k, const norm_end& e) {
    operand_gl ag{static_cast<bf16*>(const_cast<void*>(a)), nullptr, nullptr, size_t(m), size_t(k)};
    operand_gl bg{static_cast<bf16*>(const_cast<void*>(b)), nullptr, nullptr, size_t(n), size_t(k)};
    output_gl cg{static_cast<bf16*>(c), nullptr, nullptr, size_t(m), size_t(n)};
    // The row's vector map: A itself (`m` rows), read only when normed.
    row_gl rg{static_cast<bf16*>(const_cast<void*>(a)), nullptr, nullptr, size_t(m), size_t(k)};
    gain_gl gg{static_cast<bf16*>(const_cast<void*>(e.gain)), nullptr, nullptr, size_t(1),
               size_t(k)};
    return narrow_globals{ag,
                          bg,
                          cg,
                          int(m),
                          int(k / uint32_t(TILE)),
                          k % uint32_t(TILE) != 0,
                          rg,
                          gg,
                          e.eps,
                          e.offset,
                          float(k),
                          int((k + 255) / 256),
                          e.gain != nullptr ? 1 : 0};
}


int narrow(const void* a, const void* b, void* c, uint32_t m, uint32_t n, uint32_t k,
           const norm_end& e, void* stream) {
    if (m == 0 || m > uint32_t(INT_MAX) || n == 0 || (n != 1 && n % VEC != 0) || n > MAX_N ||
        k == 0 || k % VEC != 0 || k > uint32_t(INT_MAX) || (n == 1 && k % ONE_CHUNK != 0) ||
        !norm_shape(e, m, k)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    if (n == 1) {
        // ThunderKittens' host code can throw; an exception must never unwind into Rust.
        try {
            // No view carries a TMA type, so building them encodes nothing.
            one_in_gl ag{static_cast<bf16*>(const_cast<void*>(a)), nullptr, nullptr, size_t(m),
                         size_t(k)};
            one_in_gl bg{static_cast<bf16*>(const_cast<void*>(b)), nullptr, nullptr, size_t(1),
                         size_t(k)};
            one_out_gl cg{static_cast<bf16*>(c), nullptr, nullptr, size_t(m), nullptr};
            gain_gl gg{static_cast<bf16*>(const_cast<void*>(e.gain)), nullptr, nullptr, size_t(1),
                       size_t(k)};
            one_globals G{ag,    bg,       cg,       int(m), int(k / uint32_t(ONE_CHUNK)),
                          gg,    e.eps,    e.offset, float(k), e.gain != nullptr ? 1 : 0};
            void* args[] = {&G};
            const dim3 grid((m + uint32_t(ONE_WARPS) - 1) / uint32_t(ONE_WARPS)),
                block(ONE_THREADS);
            return static_cast<int>(cudaLaunchKernel(gemm_narrow_bf16_n1_ker, grid, block, args,
                                                     0, static_cast<cudaStream_t>(stream)));
        } catch (...) {
            return SCRATCHY_CUDA_ERR_HOST;
        }
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        // Building each `gl` encodes its TMA tensor maps into host memory; nothing is allocated
        // on the device. Row strides are `2k` and `2n` bytes, multiples of 16.
        narrow_globals G = narrow_member(a, b, c, m, n, k, e);
        void* args[] = {&G};
        const dim3 grid((m + uint32_t(CTA_ROWS) - 1) / uint32_t(CTA_ROWS)), block(NUM_THREADS);
        return static_cast<int>(cudaLaunchKernel(gemm_narrow_bf16_ker, grid, block, args,
                                                 NARROW_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

}  // namespace

extern "C" int scratchy_cuda_gemm_narrow_bf16_abt(const void* a, const void* b, void* c,
                                                  uint32_t m, uint32_t n, uint32_t k,
                                                  void* stream /* cudaStream_t; null = default */) {
    return narrow(a, b, c, m, n, k, NO_NORM, stream);
}


// As `scratchy_cuda_gemm_narrow_bf16_abt` for `count` (2 or 3) members of one `a [m, k]` (#1049):
// member `i` is `c[i] [m, n[i]] = a · b[i]ᵀ`, each `n[i]` a non-zero multiple of 8 at most 64 (the
// tile GEMM's; `n = 1` is refused), checked before any pointer is touched. Each member's output
// is the bits of its own `scratchy_cuda_gemm_narrow_bf16_abt`.
// `scratchy_cuda_gemm_narrow_group_bf16_abt_prepare` must have succeeded on the current device.
namespace {

int narrow_group(const void* a, const void* const* b, void* const* c, const uint32_t* n, uint32_t m,
                 uint32_t k, uint32_t count, const norm_end& e, void* stream) {
    if (count < 2 || count > 3 || !norm_shape(e, m, k)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    for (uint32_t i = 0; i < count; i++) {
        if (!narrow_tile_shape(m, n[i], k)) {
            return SCRATCHY_CUDA_ERR_SHAPE;
        }
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const uint32_t last = count - 1;
        narrow_group_globals G{narrow_member(a, b[0], c[0], m, n[0], k, e),
                               narrow_member(a, b[1], c[1], m, n[1], k, e),
                               narrow_member(a, b[last], c[last], m, n[last], k, e)};
        void* args[] = {&G};
        const dim3 grid((m + uint32_t(CTA_ROWS) - 1) / uint32_t(CTA_ROWS), count),
            block(NUM_THREADS);
        return static_cast<int>(cudaLaunchKernel(gemm_narrow_group_bf16_ker, grid, block, args,
                                                 NARROW_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

}  // namespace

extern "C" int scratchy_cuda_gemm_narrow_group_bf16_abt(const void* a, const void* const* b,
                                                        void* const* c, const uint32_t* n,
                                                        uint32_t m, uint32_t k, uint32_t count,
                                                        void* stream /* cudaStream_t */) {
    return narrow_group(a, b, c, n, m, k, count, NO_NORM, stream);
}

// FP8 S8 W4a: the narrow GEMMs with the one-row prologue norm: `gain` (null: none) the absorbed
// RMSNorm's `[k]` bf16 weight, `eps` and `gain_offset` its scalars. The row is normalised in
// shared memory (the bits of `scratchy_cuda_rms_norm_bf16` at one row) and is what each A tile's
// row 0 holds when its WGMMA (or the `n = 1` warp kernel's dot) reads it, so each output is the
// launcher above's bits on the norm's output. Only `m == 1`, `k` a multiple of 256 of at most
// 6,144 (the norm's shared row) and `eps` finite and positive, checked before any pointer is
// touched; the plain launchers' rules otherwise.
extern "C" int scratchy_cuda_gemm_narrow_bf16_ends(const void* a, const void* b, void* c,
                                                   uint32_t m, uint32_t n, uint32_t k,
                                                   const void* gain, float eps, float gain_offset,
                                                   void* stream /* cudaStream_t; null = default */) {
    return narrow(a, b, c, m, n, k, norm_end{gain, eps, gain_offset}, stream);
}

extern "C" int scratchy_cuda_gemm_narrow_group_bf16_ends(const void* a, const void* const* b,
                                                         void* const* c, const uint32_t* n,
                                                         uint32_t m, uint32_t k, uint32_t count,
                                                         const void* gain, float eps,
                                                         float gain_offset,
                                                         void* stream /* cudaStream_t */) {
    return narrow_group(a, b, c, n, m, k, count, norm_end{gain, eps, gain_offset}, stream);
}
