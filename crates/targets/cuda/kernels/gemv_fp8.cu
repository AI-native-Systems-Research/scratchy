// SPDX-License-Identifier: Apache-2.0
//
// The decode fp8 GEMVs for the H100 (FP8 S4): c[n] = bf16((Σ_k q[k] · B[n, k]) · w_scale[n] ·
// a_scale), one bf16 activation row `a [1, k]` quantized in the kernel's prologue to `q` and
// `a_scale`, against a row-major e4m3 weight `B [n, k]` with its per-channel f32 `w_scale [n]`
// (`step::fp8_weight`). The instances, as `gemv_bf16.cu`'s:
//   - `gemv_fp8_ker<false>`: every one-row fp8 projection (`CudaStep::GemvFp8`), and
//     `gemv_group_fp8_ker` the grouped one (`GemvGroupFp8`);
//   - `gemv_fp8_ker<true>`: the MoE block's gathered expert GEMV over an fp8 bank (`ExpertGemv`
//     over `MoeWeight::Fp8Bundle`/`Fp8Own`), pair `p` reading the first `n` rows of expert
//     `e = topk_idx[p]`'s `[expert_rows, k]` block and its scale row;
//   - `gate_up_gemv_fp8_ker`: the gate/up fold of either (`GateUpAct` with
//     `GateUpKernel::GemvFp8`, `ExpertGateUpAct` over an fp8 bank).
//
// The CUDA-core form, FP8 S4's measured winner (S0 part ii, 2026-11-06: 2.15 TB/s on the 27B's
// gate+up against 0.56 for WGMMA with `m` padded to 64): the e4m3 bytes are widened in registers
// inside the dot product, so no wider copy of a weight ever exists in memory.
//   - **The prologue quantization is `quant_rows_fp8.cu`'s, element for element.** Warp 0 TMA-loads
//     `a`'s whole row into shared memory, one `sv_bf<256>` per chunk on one semaphore (the `k` tail
//     is TMA's zero fill). Each warp takes chunks `w, w + 8, ...`: `warp::load` into a naive
//     `rv_fl<256>` (exact), `warp::abs`, `warp::max`; then `warp::max` to one value, written twice
//     into an `sv_fl<16>` so its 16 slots are the 8 warps', and every warp reduces those to the
//     row's amax. A max is exact in any order: this is `quant_rows_fp8`'s amax, and
//     `scratchy::ops::fp8_scale` its scale. Then the CTA's K split's chunks, in place:
//     `warp::bin_op<fp8_quant_pre>` by the scale (vLLM v0.31.0's division and clamp),
//     `warp::unary_op<fp8_round>` (TK's e4m3 convertor, round to nearest even, saturating, widened
//     back), `warp::store` to the bf16 chunk: every e4m3 value is a bf16, so the chunk holds the
//     values of `quant_rows_fp8`'s bytes exactly. `scratchy_cuda_gemv_fp8_quant_probe` stores them
//     and the scale, so a test proves they are `quant_rows_fp8`'s at every split.
//   - **The weights stream as in `gemv_bf16.cu`.** Grid `(column blocks, K splits, pairs)`, 8 warps
//     of `ROWS` rows (`rows_per_warp`), `step::gemv_splits`' split. Each warp runs its own 4-stage
//     ring of `sv_fp8e4m3<256>` (256 bytes, one vector TMA per row and chunk, the `k` tail zero),
//     primed before the prologue so the loads overlap it; it waits once per stage, then per row
//     widens the chunk with the one gap, `scratchy::ops::sv_fp8_to_rv_fl` (TK has no `rv<fp8>`),
//     into a naive `rv_fl<256>` in TK's own element order, and adds its products with the chunk
//     of `q` by the `rv_fma` gap's `fma_acc`. An e4m3 times an e4m3 value is exact in f32, so each
//     row's sum is its exact products added in ascending K, then `warp::sum`'s fixed tree.
//   - **The scales**, as S3's epilogue (`gemm_fp8.cu`'s `scale_acc`): lane 0 takes
//     `(dot · w_scale[row]) · a_scale` (`base_ops::mul`). With one split it rounds to bf16
//     (`base_types::convertor<bf16, float>`) and stores through `gl::operator[]`. With `S > 1` it
//     stores the scaled f32 partial into `partials [S, P · n_pad]` (`n_pad` = `n` rounded up to the
//     32-row block; rows past `n` store +0), and `gemv_reduce_ker` adds each row's `S` partials in
//     ascending split order and rounds once: `gemv_bf16.cu`'s reduce.
//   - **The gate/up fold** scales the gate's and the up's sums by their own `w_scale` and by
//     `a_scale`, applies `OP` (`silu_mul` or `gelu_tanh_mul`) on the two f32 values and rounds
//     once, as S3's fp8 gate/up GEMM; above one split its reduce adds both, then does the same.
//
// The split changes the order the products are added in, so a shape's bits depend on its split:
// the split is a constant of the shape (`step::gemv_splits`), never of the launch. The grouped,
// gate/up and expert instances run the same CTA routine at the same split, so each member, each
// projection and each pair is its own GEMV's bits. Only `c`'s `P · n` elements and, at `S > 1`,
// the scratch's `S · P · n_pad` f32 are written.
//
// ThunderKittens' operations only (PLAN-tk100 §1), each checked against the pin:
//   - `tma::expect_bytes`, `tma::load_async` of an `sv` through a `gl` with an `sv` descriptor
//     (`ops/group/util/tma.cuh`, `ops/group/memory/vec/tma.cuh:209`; the descriptor, e4m3 as
//     `UINT8`, its zero fill, `types/global/tma.cuh:233`); `init_semaphore`, `wait`;
//   - `warp::load`/`warp::store` between a naive `rv` and an `sv` or a `gl`
//     (`ops/group/memory/vec/shared_to_register.cuh`, `global_to_register.cuh`);
//   - `warp::zero`, `abs`, `max`, `unary_op`, `bin_op` on naive `rv_fl`
//     (`ops/group/register/vec/maps.cuh`); `warp::max`, `warp::sum` (`reductions.cuh`);
//   - `base_ops::mul`, `base_types::convertor`; `shared_allocator<1024>`, `sv::operator[]`,
//     `gl::operator[]`;
//   - the gaps `scratchy::ops::sv_fp8_to_rv_fl` (FP8 S4's, approved 2026-11-06) and `rv_fma`.
//
// Shared memory: the row, 68 vectors of 512 bytes (`k` at most 17,408, the 27B's down projection,
// the largest fp8 `k` of the three FP8 checkpoints; a longer row is refused), and the weight rings,
// 8 warps × 4 stages × 4 rows of 256 bytes, plus 1,024 of alignment slack = 68,608 bytes of dynamic
// shared memory; two CTAs per SM, as `gemv_bf16.cu`'s grid assumes. Set once per instance by
// `_prepare` (in `PREPARE`). The semaphores and the amax vector are static.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::GEMM_FP8_K_MULTIPLE`: `k` is a multiple of it, so every e4m3 row stride is a multiple of
// 16 bytes, as the tensor maps need.
constexpr uint32_t K_MULTIPLE = 64;
// `gemv_bf16.cu`'s chunk, rows, splits and target (`step::GEMV_CHUNK`, `GEMV_ROWS_PER_WARP`,
// `GEMV_BLOCK_ROWS`, `GEMV_MAX_SPLITS`, `GEMV_TARGET_CTAS`): `step::gemv_splits` is one table.
constexpr int CHUNK = 256;
constexpr int ROWS_PER_WARP = 4;
constexpr int WARPS = 8;
constexpr int BLOCK_ROWS = WARPS * ROWS_PER_WARP;
constexpr int MAX_SPLITS = 16;
constexpr int STAGES = 4;
// The chunks of the longest row the shared row holds: 17,408 / 256.
constexpr int ROW_CHUNKS = 68;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;
constexpr int ROW_CHOICES[] = {4, 3, 2};
static_assert(ROW_CHOICES[0] == ROWS_PER_WARP, "the 4-row instance is the split one");
constexpr uint64_t TARGET_CTAS = 264;
// The reduce kernel: one lane per output row, 8 warps per CTA.
constexpr int REDUCE_WARPS = 8;
constexpr int REDUCE_ROWS = kittens::WARP_THREADS;
constexpr int REDUCE_THREADS = REDUCE_WARPS * kittens::WARP_THREADS;
// The expert GEMV's bounds (`op_abi::MOE_EXPERT_COUNTS`, `step::MOE_ROUTE_MAX_TOP_K`), and the
// grid's z extent, one slice per pair.
constexpr uint32_t EXPERTS_SMALL = 128;
constexpr uint32_t EXPERTS_LARGE = 256;
constexpr uint32_t MAX_TOP_K = 32;
constexpr uint32_t MAX_GRID_Z = 65535;

using row_sv = sv_bf<CHUNK>;
using w_sv = sv_fp8e4m3<CHUNK>;
using chunk_rv = rv_fl<CHUNK>;
using amax_sv = sv_fl<16>;
using amax_rv = rv_fl<16>;
using reduce_rv = rv_fl<REDUCE_ROWS>;
static_assert(2 * WARPS == 16, "the amax vector holds each warp's value twice");
// `a [rows, k]` bf16 and `B [depth, rows, k]` e4m3, each with its vector tensor map.
using in_gl = gl<bf16, 1, 1, -1, -1, row_sv>;
using weight_gl = gl<fp8e4m3, 1, -1, -1, -1, w_sv>;
// `w_scale [depth, rows]` f32, `c [P, n]` bf16 and the partials `[S, P · n_pad]` f32: no map.
using scale_gl = gl<float, 1, 1, -1, -1>;
using out_gl = gl<bf16, 1, 1, -1, -1>;
using partial_gl = gl<float, 1, 1, -1, -1>;
using idx_gl = gl<int, 1, 1, -1, -1>;

struct gemv_globals {
    in_gl a;
    weight_gl b;
    scale_gl b_scale;
    out_gl c;
    partial_gl partials;
    // `[1, P]` expert ids; the dense instance never reads it.
    idx_gl topk_idx;
    // The rows of `c` and of a pair's run of partials (`n` rounded up to 32).
    int n;
    int n_pad;
    // The chunks of a row, `⌈k / 256⌉`, and of one split, `⌈chunks / splits⌉`.
    int chunks;
    int split_chunks;
    int splits;
    // Pair `p` reads `a`'s row `p / row_div` (`top_k` for the token rows, 1 for the pair rows and
    // the dense GEMV).
    int row_div;
    // FP8 S8 W4a, the residual epilogue: `residual` `[P, n]` bf16, added to each stored row when
    // `add` (the reduce adds it at `S > 1`).
    out_gl residual;
    int add;
};

struct reduce_globals {
    partial_gl partials;
    out_gl c;
    int n;
    int n_pad;
    int splits;
    // FP8 S8 W4a: the residual epilogue, as `gemv_globals`'.
    out_gl residual;
    int add;
};

constexpr size_t SMEM_ROW = size_t(ROW_CHUNKS) * sizeof(row_sv);
constexpr size_t SMEM_RING = size_t(WARPS) * STAGES * ROWS_PER_WARP * sizeof(w_sv);
static_assert(sizeof(row_sv) == 512 && sizeof(w_sv) == 256, "a row chunk, a weight chunk");
static_assert(SMEM_ROW % 1024 == 0 && SMEM_RING % 1024 == 0, "no allocation pads the next");
// The vectors plus the slack the allocator may spend aligning the first to 1024 bytes.
constexpr int GEMV_SMEM_BYTES = int(SMEM_ROW + SMEM_RING) + 1024;
static_assert(GEMV_SMEM_BYTES == 68608, "the row, the rings, the slack");
static_assert(2 * GEMV_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "two CTAs per SM");

// The CTA's shared memory and semaphores: `a`'s row, each warp's weight ring.
template <int ROWS>
struct gemv_smem {
    row_sv (&row)[ROW_CHUNKS];
    w_sv (&b)[WARPS][STAGES][ROWS];
    kittens::semaphore& row_arrived;
    kittens::semaphore (&arrived)[WARPS][STAGES];
    amax_sv& amax;
};

// The prologue: `a`'s row `a_row` (`chunks` chunks) into `sm.row`, its amax and scale, then its
// chunks `c0 .. c0 + count` quantized in place; returns the scale, in every thread. Every thread
// of the CTA calls it once.
template <int ROWS>
__device__ inline float quantize_row(const in_gl& a, int a_row, int chunks, int c0, int count,
                                     const gemv_smem<ROWS>& sm) {
    const int w = kittens::warpid();
    if (w == 0) {
        if (warp::laneid() == 0) {
            init_semaphore(sm.row_arrived, 0, 1);
        }
        __syncwarp();
        warp::tma::expect_bytes(sm.row_arrived, uint32_t(chunks * sizeof(row_sv)));
        for (int c = 0; c < chunks; c++) {
            warp::tma::load_async(sm.row[c], a, {0, 0, a_row, c}, sm.row_arrived);
        }
    }
    // The semaphore is initialised before any thread waits on it.
    __syncthreads();
    wait(sm.row_arrived, 0);
    chunk_rv x, m;
    warp::zero(m);
    for (int c = w; c < chunks; c += WARPS) {
        warp::load(x, sm.row[c]);
        warp::abs(x, x);
        warp::max(m, m, x);
    }
    const float part = warp::max(m);
    if (warp::laneid() == 0) {
        sm.amax[w] = part;
        sm.amax[w + WARPS] = part;
    }
    // Every warp's amax is written, and every warp has read the row for it, before any chunk is
    // overwritten.
    __syncthreads();
    amax_rv all;
    warp::zero(all);
    warp::load(all, sm.amax);
    const float scale = scratchy::ops::fp8_scale::op<float>(warp::max(all));
    for (int c = c0 + w; c < c0 + count; c += WARPS) {
        warp::load(x, sm.row[c]);
        warp::bin_op<scratchy::ops::fp8_quant_pre>(x, x, scale);
        warp::unary_op<scratchy::ops::fp8_round>(x, x);
        warp::store(sm.row[c], x);
    }
    // Every chunk of the slice is quantized before any warp reads it.
    __syncthreads();
    return scale;
}

// The CTA's arithmetic, `gemv_bf16.cu`'s `gemv_dots` over the e4m3 ring and the quantized row:
// each of the warp's rows' unscaled partial sums, in every lane, rows past `n` (and a split past
// the row's chunks) +0. `before` runs in every thread after the warp primes its ring (the
// prologue, or nothing). `j0` is the ring index of the call's first load: 0 for the first call,
// which initialises the semaphores; the gate/up fold's second call continues the ring.
template <bool EXPERT, int ROWS, typename BEFORE>
__device__ inline void gemv_dots(const gemv_globals& g, const gemv_smem<ROWS>& sm, int block,
                                 int split, int p, int j0, BEFORE&& before, float (&dot)[ROWS]) {
    const int w = kittens::warpid();
    const int row0 = (block * WARPS + w) * ROWS;
    const int rows = max(0, min(ROWS, g.n - row0));
    const int c0 = split * g.split_chunks;
    const int count = max(0, min(g.split_chunks, g.chunks - c0));
    const bool live = rows > 0 && count > 0;
    int depth = 0;
    if constexpr (EXPERT) {
        depth = g.topk_idx[{0, 0, 0, p}];
    }
    // Chunk `j` of the split (`c0 + j` of the row) into stage `(j0 + j) % STAGES`.
    const auto issue = [&](int j) {
        const int s = (j0 + j) % STAGES;
        warp::tma::expect_bytes(sm.arrived[w][s], uint32_t(rows * sizeof(w_sv)));
        // A constant trip count, as the dot loop's. With `r < rows` as the loop bound, the 2-row
        // instance's refills (a split of more than 4 chunks) summed wrong bytes, or hung with
        // more than one CTA, on the H100; the 3- and 4-row instances never did.
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            if (r < rows) {
                warp::tma::load_async(sm.b[w][s][r], g.b, {0, depth, row0 + r, c0 + j},
                                      sm.arrived[w][s]);
            }
        }
    };
    if (live) {
        if (j0 == 0 && warp::laneid() == 0) {
            for (int s = 0; s < STAGES; s++) {
                init_semaphore(sm.arrived[w][s], 0, 1);
            }
        }
        __syncwarp();
        for (int j = 0; j < min(STAGES, count); j++) {
            issue(j);
        }
    }
    before();

    chunk_rv acc[ROWS];
    chunk_rv x, y;
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        warp::zero(acc[r]);
    }
    if (live) {
        for (int j = 0; j < count; j++) {
            const int s = (j0 + j) % STAGES;
            warp::load(x, sm.row[c0 + j]);
            wait(sm.arrived[w][s], ((j0 + j) / STAGES) % 2);
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                if (r < rows) {
                    scratchy::ops::sv_fp8_to_rv_fl(y, sm.b[w][s][r]);
                    scratchy::ops::rv_fma<scratchy::ops::fma_acc>(acc[r], y, x, acc[r]);
                }
            }
            // Every lane has read the stage into registers before its next load overwrites it.
            __syncwarp();
            if (j + STAGES < count) {
                issue(j + STAGES);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        dot[r] = warp::sum(acc[r]);
    }
}

// `dot` of row `row` of weight depth `depth`, scaled as S3's epilogue: `(dot · w_scale) · a_scale`.
__device__ inline float fp8_scaled(const gemv_globals& g, int depth, int row, float dot,
                                   float a_scale) {
    return base_ops::mul::op<float>(base_ops::mul::op<float>(dot, g.b_scale[{0, 0, depth, row}]),
                                    a_scale);
}

// One CTA of one GEMV: the prologue inside `gemv_dots`, then lane 0 stores each row scaled:
// rounded at one split, else its f32 partial (+0 past `n`, so the reduce reads written values).
template <bool EXPERT, int ROWS>
__device__ inline void gemv_cta(const gemv_globals& g, const gemv_smem<ROWS>& sm, int block,
                                int split, int p) {
    const int c0 = split * g.split_chunks;
    const int count = max(0, min(g.split_chunks, g.chunks - c0));
    float a_scale = 0.0f;
    float dot[ROWS];
    gemv_dots<EXPERT, ROWS>(
        g, sm, block, split, p, 0,
        [&] { a_scale = quantize_row(g.a, p / g.row_div, g.chunks, c0, count, sm); }, dot);
    const int row0 = (block * WARPS + kittens::warpid()) * ROWS;
    const int rows = max(0, min(ROWS, g.n - row0));
    int depth = 0;
    if constexpr (EXPERT) {
        depth = g.topk_idx[{0, 0, 0, p}];
    }
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const int row = row0 + r;
        if (warp::laneid() == 0) {
            const float v = r < rows ? fp8_scaled(g, depth, row, dot[r], a_scale) : 0.0f;
            if (g.splits == 1) {
                if (r < rows) {
                    // FP8 S8 W4a: the residual epilogue, `add_bf16`'s add on the stored row.
                    g.c[{0, 0, p, row}] =
                        g.add ? scratchy::ops::with_residual(v, g.residual[{0, 0, p, row}])
                              : base_types::convertor<bf16, float>::convert(v);
                }
            } else {
                g.partials[{0, 0, split, p * g.n_pad + row}] = v;
            }
        }
    }
}

// The dense or expert GEMV: the grid is `(column blocks, K splits, pairs)`.
template <bool EXPERT, int ROWS>
__global__ __launch_bounds__(NUM_THREADS, 2) void gemv_fp8_ker(const __grid_constant__ gemv_globals g) {
    extern __shared__ int __shm[];
    shared_allocator<1024> al((int*)&__shm[0]);
    row_sv(&row_s)[ROW_CHUNKS] = al.allocate<row_sv, ROW_CHUNKS>();
    w_sv(&b)[WARPS][STAGES][ROWS] = al.allocate<w_sv, WARPS, STAGES, ROWS>();
    __shared__ kittens::semaphore row_arrived;
    __shared__ kittens::semaphore arrived[WARPS][STAGES];
    __shared__ sv_fl<16> amax;
    const gemv_smem<ROWS> sm{row_s, b, row_arrived, arrived, amax};
    gemv_cta<EXPERT, ROWS>(g, sm, blockIdx.x, blockIdx.y, blockIdx.z);
}

// The grouped GEMV: up to three dense fp8 GEMVs of one `a`, one launch, `gemv_bf16.cu`'s grid
// (member `i`'s CTAs are `start[i] .. start[i + 1]` of the grid's x). A CTA runs its member's
// `gemv_cta`, so each member is its own GEMV's bits. An absent third member repeats the second's
// views and starts at the grid's end, so it owns no CTA.
struct gemv_group_globals {
    gemv_globals m0;
    gemv_globals m1;
    gemv_globals m2;
    int start1;
    int start2;
    int blocks0;
    int blocks1;
    int blocks2;
};

__global__ __launch_bounds__(NUM_THREADS, 2) void gemv_group_fp8_ker(
    const __grid_constant__ gemv_group_globals g) {
    extern __shared__ int __shm[];
    shared_allocator<1024> al((int*)&__shm[0]);
    row_sv(&row_s)[ROW_CHUNKS] = al.allocate<row_sv, ROW_CHUNKS>();
    w_sv(&b)[WARPS][STAGES][ROWS_PER_WARP] = al.allocate<w_sv, WARPS, STAGES, ROWS_PER_WARP>();
    __shared__ kittens::semaphore row_arrived;
    __shared__ kittens::semaphore arrived[WARPS][STAGES];
    __shared__ sv_fl<16> amax;
    const gemv_smem<ROWS_PER_WARP> sm{row_s, b, row_arrived, arrived, amax};
    const int x = blockIdx.x;
    const int member = x >= g.start2 ? 2 : x >= g.start1 ? 1 : 0;
    const int local = x - (member == 2 ? g.start2 : member == 1 ? g.start1 : 0);
    const int blocks = member == 2 ? g.blocks2 : member == 1 ? g.blocks1 : g.blocks0;
    const gemv_globals& m = member == 2 ? g.m2 : member == 1 ? g.m1 : g.m0;
    gemv_cta<false, ROWS_PER_WARP>(m, sm, local % blocks, local / blocks, 0);
}

// `gemv_bf16.cu`'s reduce: rows `32v .. 32v + 32` of `g`'s `[S, P · n_pad]` scaled partials, one
// per lane, added split by split in ascending order, rounded once.
__device__ inline void reduce_warp(const reduce_globals& g, int v) {
    if (v * REDUCE_ROWS >= g.partials.cols()) {
        return;
    }
    reduce_rv acc, part;
    warp::zero(acc);
    for (int s = 0; s < g.splits; s++) {
        warp::load(part, g.partials, {0, 0, s, v});
        warp::add(acc, acc, part);
    }
    const int rr = v * REDUCE_ROWS + warp::laneid();
    const int p = rr / g.n_pad;
    const int col = rr % g.n_pad;
    if (col < g.n) {
        g.c[{0, 0, p, col}] = g.add ? scratchy::ops::with_residual(acc[0][0], g.residual[{0, 0, p, col}])
                                    : base_types::convertor<bf16, float>::convert(acc[0][0]);
    }
}

__global__ __launch_bounds__(REDUCE_THREADS) void gemv_reduce_ker(
    const __grid_constant__ reduce_globals g) {
    reduce_warp(g, blockIdx.x * REDUCE_WARPS + kittens::warpid());
}

// The grouped GEMV's reduce: the split members' reduces, one launch (`gemv_bf16.cu`'s).
struct reduce_group_globals {
    reduce_globals m0;
    reduce_globals m1;
    reduce_globals m2;
    int start1;
    int start2;
    int end;
};

__global__ __launch_bounds__(REDUCE_THREADS) void gemv_reduce_group_ker(
    const __grid_constant__ reduce_group_globals g) {
    const int v = blockIdx.x * REDUCE_WARPS + kittens::warpid();
    if (v >= g.end) {
        return;
    }
    const int member = v >= g.start2 ? 2 : v >= g.start1 ? 1 : 0;
    const int start = member == 2 ? g.start2 : member == 1 ? g.start1 : 0;
    const reduce_globals& m = member == 2 ? g.m2 : member == 1 ? g.m1 : g.m0;
    reduce_warp(m, v - start);
}

// The gate/up GEMV: `c = OP(g, u)` per row, rounded once, `g` and `u` the gate's and the up's
// scaled sums (each its own GEMV's value before its rounding), S3's fp8 gate/up form. One
// prologue serves both; the up pass's loads continue the gate pass's ring.
struct gate_up_globals {
    gemv_globals gate;
    gemv_globals up;
};

template <bool EXPERT, typename OP>
__global__ __launch_bounds__(NUM_THREADS, 2) void gate_up_gemv_fp8_ker(
    const __grid_constant__ gate_up_globals g) {
    extern __shared__ int __shm[];
    shared_allocator<1024> al((int*)&__shm[0]);
    row_sv(&row_s)[ROW_CHUNKS] = al.allocate<row_sv, ROW_CHUNKS>();
    w_sv(&b)[WARPS][STAGES][ROWS_PER_WARP] = al.allocate<w_sv, WARPS, STAGES, ROWS_PER_WARP>();
    __shared__ kittens::semaphore row_arrived;
    __shared__ kittens::semaphore arrived[WARPS][STAGES];
    __shared__ sv_fl<16> amax;
    const gemv_smem<ROWS_PER_WARP> sm{row_s, b, row_arrived, arrived, amax};
    const int block = blockIdx.x, split = blockIdx.y, p = blockIdx.z;
    const int c0 = split * g.gate.split_chunks;
    const int count = max(0, min(g.gate.split_chunks, g.gate.chunks - c0));
    float a_scale = 0.0f;
    float gate[ROWS_PER_WARP], up[ROWS_PER_WARP];
    gemv_dots<EXPERT, ROWS_PER_WARP>(
        g.gate, sm, block, split, p, 0,
        [&] {
            a_scale = quantize_row(g.gate.a, p / g.gate.row_div, g.gate.chunks, c0, count, sm);
        },
        gate);
    // Every lane has read the gate's last stage (its loop's last `__syncwarp`), so the up's loads
    // may take the ring's next stages. A warp with no rows ran no loads, and runs none now.
    gemv_dots<EXPERT, ROWS_PER_WARP>(g.up, sm, block, split, p, count, [] {}, up);
    const int row0 = (block * WARPS + kittens::warpid()) * ROWS_PER_WARP;
    const int rows = max(0, min(ROWS_PER_WARP, g.gate.n - row0));
    int depth = 0;
    if constexpr (EXPERT) {
        depth = g.gate.topk_idx[{0, 0, 0, p}];
    }
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        const int row = row0 + r;
        if (warp::laneid() == 0) {
            const float gv = r < rows ? fp8_scaled(g.gate, depth, row, gate[r], a_scale) : 0.0f;
            const float uv = r < rows ? fp8_scaled(g.up, depth, row, up[r], a_scale) : 0.0f;
            if (g.gate.splits == 1) {
                if (r < rows) {
                    g.gate.c[{0, 0, p, row}] =
                        base_types::convertor<bf16, float>::convert(OP::template op<float>(gv, uv));
                }
            } else {
                g.gate.partials[{0, 0, split, p * g.gate.n_pad + row}] = gv;
                g.up.partials[{0, 0, split, p * g.up.n_pad + row}] = uv;
            }
        }
    }
}

// The gate/up GEMV's reduce: `reduce_warp`'s sums of the gate's and the up's scaled partials,
// then `OP` through TK's vector map and the one rounding. `gate.c` is the output.
struct gate_up_reduce_globals {
    reduce_globals gate;
    reduce_globals up;
};

template <typename OP>
__global__ __launch_bounds__(REDUCE_THREADS) void gate_up_reduce_fp8_ker(
    const __grid_constant__ gate_up_reduce_globals g) {
    const int v = blockIdx.x * REDUCE_WARPS + kittens::warpid();
    if (v * REDUCE_ROWS >= g.gate.partials.cols()) {
        return;
    }
    reduce_rv gate, up, part;
    warp::zero(gate);
    warp::zero(up);
    for (int s = 0; s < g.gate.splits; s++) {
        warp::load(part, g.gate.partials, {0, 0, s, v});
        warp::add(gate, gate, part);
        warp::load(part, g.up.partials, {0, 0, s, v});
        warp::add(up, up, part);
    }
    warp::bin_op<OP>(gate, gate, up);
    const int rr = v * REDUCE_ROWS + warp::laneid();
    const int p = rr / g.gate.n_pad;
    const int col = rr % g.gate.n_pad;
    if (col < g.gate.n) {
        g.gate.c[{0, 0, p, col}] = base_types::convertor<bf16, float>::convert(gate[0][0]);
    }
}

// Test support (`tests/gemv_fp8.rs`): the prologue alone, one CTA per K split of a `[1, k]` row,
// each storing its quantized chunks' f32 values into `out` and its scale into `scale[split]`. No
// `CudaStep` launches it.
struct probe_globals {
    in_gl a;
    partial_gl out;
    partial_gl scale;
    int chunks;
    int split_chunks;
};

__global__ __launch_bounds__(NUM_THREADS) void gemv_fp8_quant_probe_ker(
    const __grid_constant__ probe_globals g) {
    extern __shared__ int __shm[];
    shared_allocator<1024> al((int*)&__shm[0]);
    row_sv(&row_s)[ROW_CHUNKS] = al.allocate<row_sv, ROW_CHUNKS>();
    w_sv(&b)[WARPS][STAGES][ROWS_PER_WARP] = al.allocate<w_sv, WARPS, STAGES, ROWS_PER_WARP>();
    __shared__ kittens::semaphore row_arrived;
    __shared__ kittens::semaphore arrived[WARPS][STAGES];
    __shared__ sv_fl<16> amax;
    const gemv_smem<ROWS_PER_WARP> sm{row_s, b, row_arrived, arrived, amax};
    const int split = blockIdx.x;
    const int c0 = split * g.split_chunks;
    const int count = max(0, min(g.split_chunks, g.chunks - c0));
    const float a_scale = quantize_row(g.a, 0, g.chunks, c0, count, sm);
    for (int c = c0 + kittens::warpid(); c < c0 + count; c += WARPS) {
        chunk_rv x;
        warp::load(x, sm.row[c]);
        warp::store(g.out, x, {0, 0, 0, c});
    }
    if (kittens::warpid() == 0 && warp::laneid() == 0) {
        g.scale[{0, 0, 0, split}] = a_scale;
    }
}

// `n` rounded up to the 32-row block.
uint32_t padded(uint32_t n) {
    return (n + uint32_t(BLOCK_ROWS) - 1) / uint32_t(BLOCK_ROWS) * uint32_t(BLOCK_ROWS);
}

// The chunks of a row of `k`, or 0 when `k` is not a non-zero multiple of 64 or the row is longer
// than the shared row holds.
uint32_t row_chunks(uint32_t k) {
    const uint32_t chunks = (k + uint32_t(CHUNK) - 1) / uint32_t(CHUNK);
    return k == 0 || k % K_MULTIPLE != 0 || chunks > uint32_t(ROW_CHUNKS) ? 0 : chunks;
}

// The chunks of one split, or 0 when `splits` does not cut `chunks` into non-empty runs (1 to
// `MAX_SPLITS` of them): `gemv_bf16.cu`'s rule, `step::gemv_split_chunks`.
uint32_t split_chunks(uint32_t chunks, uint32_t splits) {
    if (splits == 0 || splits > uint32_t(MAX_SPLITS) || splits > chunks) {
        return 0;
    }
    const uint32_t each = (chunks + splits - 1) / splits;
    return (splits - 1) * each < chunks ? each : 0;
}

// The CTAs of a grid of `pairs` slices of `n` rows at `rows` rows per warp.
uint64_t ctas(uint32_t n, uint32_t pairs, int rows) {
    const uint64_t block = uint64_t(WARPS) * uint64_t(rows);
    return (uint64_t(n) + block - 1) / block * uint64_t(pairs);
}

// `gemv_bf16.cu`'s rows per warp, a constant of the shape.
int rows_per_warp(uint32_t n, uint32_t pairs, uint32_t splits) {
    if (splits > 1) {
        return ROWS_PER_WARP;
    }
    int best = ROWS_PER_WARP;
    uint64_t best_ctas = ctas(n, pairs, best);
    uint64_t best_waves = (best_ctas + TARGET_CTAS - 1) / TARGET_CTAS;
    for (const int rows : ROW_CHOICES) {
        const uint64_t c = ctas(n, pairs, rows);
        const uint64_t waves = (c + TARGET_CTAS - 1) / TARGET_CTAS;
        if (20 * c >= 17 * TARGET_CTAS * waves) {
            return rows;
        }
        if (c * best_waves > best_ctas * waves) {
            best = rows;
            best_ctas = c;
            best_waves = waves;
        }
    }
    return best;
}

template <bool EXPERT, int ROWS>
cudaError_t launch_main(gemv_globals& G, uint32_t n, uint32_t pairs, uint32_t splits,
                        cudaStream_t s) {
    void* args[] = {&G};
    const uint32_t block_rows = uint32_t(WARPS * ROWS);
    const dim3 grid((n + block_rows - 1) / block_rows, splits, pairs), block(NUM_THREADS);
    return cudaLaunchKernel(gemv_fp8_ker<EXPERT, ROWS>, grid, block, args, GEMV_SMEM_BYTES, s);
}

// The main kernel of instance `EXPERT` at `rows_per_warp`'s rows, then at `splits > 1` the reduce,
// on `stream`.
template <bool EXPERT>
int launch(gemv_globals G, uint32_t n, uint32_t pairs, uint32_t splits, void* stream) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int rows = rows_per_warp(n, pairs, splits);
    const cudaError_t launched =
        rows == 2   ? launch_main<EXPERT, 2>(G, n, pairs, splits, s)
        : rows == 3 ? launch_main<EXPERT, 3>(G, n, pairs, splits, s)
                    : launch_main<EXPERT, ROWS_PER_WARP>(G, n, pairs, splits, s);
    if (launched != cudaSuccess || splits == 1) {
        return static_cast<int>(launched);
    }
    reduce_globals R{G.partials, G.c, int(n), int(padded(n)), int(splits), G.residual, G.add};
    void* rargs[] = {&R};
    const uint32_t vectors = pairs * padded(n) / uint32_t(REDUCE_ROWS);
    const dim3 rgrid((vectors + REDUCE_WARPS - 1) / REDUCE_WARPS), rblock(REDUCE_THREADS);
    return static_cast<int>(cudaLaunchKernel(gemv_reduce_ker, rgrid, rblock, rargs, 0, s));
}

// The dense GEMV's shape rule at `splits`: the chunks of one split, or 0 when it is refused.
uint32_t dense_split(void* partials, uint32_t n, uint32_t k, uint32_t splits) {
    const uint32_t chunks = row_chunks(k);
    if (n == 0 || n > uint32_t(INT_MAX) || chunks == 0) {
        return 0;
    }
    const uint32_t each = split_chunks(chunks, splits);
    if (each == 0 || (splits > 1 && partials == nullptr) ||
        uint64_t(splits) * uint64_t(padded(n)) > uint64_t(INT_MAX)) {
        return 0;
    }
    return each;
}

// The views and counts of a GEMV of `pairs` slices of `n` rows over `B [depth, rows, k]` at
// `splits`, `each` chunks a split. Building each view encodes its vector tensor map into host
// memory; nothing is allocated on the device. It can throw (TK's host code).
gemv_globals views(const void* a, uint32_t a_rows, const void* b, const void* b_scale,
                   uint32_t depth, uint32_t rows, const void* topk_idx, void* c, void* partials,
                   uint32_t pairs, uint32_t n, uint32_t k, uint32_t splits, uint32_t each,
                   int row_div) {
    in_gl ag{static_cast<bf16*>(const_cast<void*>(a)), nullptr, nullptr, size_t(a_rows),
             size_t(k)};
    weight_gl bg{static_cast<fp8e4m3*>(const_cast<void*>(b)), nullptr, size_t(depth), size_t(rows),
                 size_t(k)};
    scale_gl sg{static_cast<float*>(const_cast<void*>(b_scale)), nullptr, nullptr, size_t(depth),
                size_t(rows)};
    out_gl cg{static_cast<bf16*>(c), nullptr, nullptr, size_t(pairs), size_t(n)};
    partial_gl pg{static_cast<float*>(partials), nullptr, nullptr, size_t(splits),
                  size_t(uint64_t(pairs) * padded(n))};
    idx_gl ig{static_cast<int*>(const_cast<void*>(topk_idx)), nullptr, nullptr, size_t(1),
              size_t(pairs)};
    out_gl rg{nullptr, nullptr, nullptr, size_t(pairs), size_t(n)};
    return gemv_globals{ag,     bg,
                        sg,     cg,
                        pg,     ig,
                        int(n), int(padded(n)),
                        int(row_chunks(k)), int(each),
                        int(splits), row_div,
                        rg,     0};
}

// The dense GEMV's views (`views` over one row and one weight).
gemv_globals dense_globals(const void* a, const void* b, const void* b_scale, void* c,
                           void* partials, uint32_t n, uint32_t k, uint32_t splits,
                           uint32_t each) {
    return views(a, 1, b, b_scale, 1, n, nullptr, c, partials, 1, n, k, splits, each, 1);
}

// The dense GEMV at `splits`: the shape rule, then the views and the launch.
int dense(const void* a, const void* b, const void* b_scale, void* c, void* partials, uint32_t n,
          uint32_t k, uint32_t splits, const void* residual, void* stream) {
    const uint32_t each = dense_split(partials, n, k, splits);
    if (each == 0) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        gemv_globals G = dense_globals(a, b, b_scale, c, partials, n, k, splits, each);
        // FP8 S8 W4a: the residual epilogue (null: none), `[1, n]` as `c`.
        if (residual != nullptr) {
            G.residual = out_gl{static_cast<bf16*>(const_cast<void*>(residual)), nullptr, nullptr,
                                size_t(1), size_t(n)};
            G.add = 1;
        }
        return launch<false>(G, n, 1, splits, stream);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// The grouped GEMV: `count` (2 or 3) dense GEMVs of one `a [1, k]`, member `i` writing `c[i]`
// from `b[i]` and `b_scale[i]` at `splits[i]`, its partials at `partials[i]`. Each member meets
// the dense GEMV's rule on its own. The main launch runs every member's CTAs at 4 rows per warp;
// then one reduce over the split members, if any: `gemv_bf16.cu`'s `launch_group`.
int launch_group(const void* a, const void* const* b, const void* const* b_scale, void* const* c,
                 void* const* partials, const uint32_t* n, uint32_t k, const uint32_t* splits,
                 uint32_t count, void* stream) {
    if (count < 2 || count > 3) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    uint32_t each[3] = {0, 0, 0};
    uint64_t ctas = 0, vectors = 0;
    for (uint32_t i = 0; i < count; i++) {
        each[i] = dense_split(partials[i], n[i], k, splits[i]);
        if (each[i] == 0) {
            return SCRATCHY_CUDA_ERR_SHAPE;
        }
        ctas += uint64_t(padded(n[i]) / uint32_t(BLOCK_ROWS)) * splits[i];
        vectors += splits[i] > 1 ? padded(n[i]) / uint32_t(REDUCE_ROWS) : 0;
    }
    if (ctas > uint64_t(INT_MAX) || vectors > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const uint32_t last = count - 1;
        gemv_globals m[3] = {
            dense_globals(a, b[0], b_scale[0], c[0], partials[0], n[0], k, splits[0], each[0]),
            dense_globals(a, b[1], b_scale[1], c[1], partials[1], n[1], k, splits[1], each[1]),
            dense_globals(a, b[last], b_scale[last], c[last], partials[last], n[last], k,
                          splits[last], each[last]),
        };
        int blocks[3], start[4] = {0, 0, 0, 0};
        for (uint32_t i = 0; i < 3; i++) {
            blocks[i] = int(padded(n[i < count ? i : last]) / uint32_t(BLOCK_ROWS));
            const int own = i < count ? blocks[i] * int(splits[i]) : 0;
            start[i + 1] = start[i] + own;
        }
        gemv_group_globals G{m[0],     m[1],      m[2],      start[1],
                             start[2], blocks[0], blocks[1], blocks[2]};
        void* args[] = {&G};
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        const cudaError_t launched = cudaLaunchKernel(
            gemv_group_fp8_ker, dim3(start[3]), dim3(NUM_THREADS), args, GEMV_SMEM_BYTES, s);
        if (launched != cudaSuccess || vectors == 0) {
            return static_cast<int>(launched);
        }
        // The split members' reduces, in member order; a one-split member owns no warp.
        reduce_globals r[3] = {
            {m[0].partials, m[0].c, m[0].n, m[0].n_pad, m[0].splits, m[0].residual, m[0].add},
            {m[1].partials, m[1].c, m[1].n, m[1].n_pad, m[1].splits, m[1].residual, m[1].add},
            {m[2].partials, m[2].c, m[2].n, m[2].n_pad, m[2].splits, m[2].residual, m[2].add},
        };
        int rstart[4] = {0, 0, 0, 0};
        for (uint32_t i = 0; i < 3; i++) {
            const bool split = i < count && splits[i] > 1;
            rstart[i + 1] = rstart[i] + (split ? int(padded(n[i]) / uint32_t(REDUCE_ROWS)) : 0);
        }
        reduce_group_globals R{r[0], r[1], r[2], rstart[1], rstart[2], rstart[3]};
        void* rargs[] = {&R};
        const dim3 rgrid((uint32_t(rstart[3]) + REDUCE_WARPS - 1) / REDUCE_WARPS),
            rblock(REDUCE_THREADS);
        return static_cast<int>(cudaLaunchKernel(gemv_reduce_group_ker, rgrid, rblock, rargs, 0, s));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// The expert GEMV's shape rule (`gemv_bf16.cu`'s `expert`, with the fp8 `k`): the chunks of one
// split, or 0 when it is refused.
uint32_t expert_split(void* partials, uint32_t m, uint32_t top_k, uint32_t n, uint32_t k,
                      uint32_t experts, uint32_t expert_rows, uint64_t a_rows, uint32_t splits) {
    const uint64_t pairs = uint64_t(m) * uint64_t(top_k);
    const uint32_t chunks = row_chunks(k);
    if (m == 0 || top_k == 0 || top_k > MAX_TOP_K || pairs > uint64_t(MAX_GRID_Z) || n == 0 ||
        n % 64 != 0 || chunks == 0 || (experts != EXPERTS_SMALL && experts != EXPERTS_LARGE) ||
        expert_rows < n || a_rows * uint64_t(k) > uint64_t(INT_MAX) ||
        pairs * uint64_t(n) > uint64_t(INT_MAX) ||
        uint64_t(experts) * uint64_t(expert_rows) * uint64_t(k) > uint64_t(INT_MAX)) {
        return 0;
    }
    const uint32_t each = split_chunks(chunks, splits);
    if (each == 0 || (splits > 1 && partials == nullptr) ||
        uint64_t(splits) * pairs * uint64_t(padded(n)) > uint64_t(INT_MAX)) {
        return 0;
    }
    return each;
}

// The expert GEMV at `splits`: the shape rule, then the views and the launch.
int expert(const void* a, const void* b, const void* b_scale, const void* topk_idx, void* c,
           void* partials, uint32_t m, uint32_t top_k, uint32_t n, uint32_t k, uint32_t experts,
           uint32_t expert_rows, uint32_t pair_rows, uint32_t splits, void* stream) {
    const uint32_t pairs = m * top_k;
    const uint64_t a_rows = pair_rows != 0 ? uint64_t(m) * uint64_t(top_k) : uint64_t(m);
    const uint32_t each =
        expert_split(partials, m, top_k, n, k, experts, expert_rows, a_rows, splits);
    if (each == 0) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        gemv_globals G = views(a, uint32_t(a_rows), b, b_scale, experts, expert_rows, topk_idx, c,
                               partials, pairs, n, k, splits, each,
                               pair_rows != 0 ? 1 : int(top_k));
        return launch<true>(G, n, pairs, splits, stream);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// The gate/up GEMV's activation: 0 SwiGLU (`scratchy::ops::silu_mul`), 1 GeGLU
// (`scratchy::ops::gelu_tanh_mul`), as `step::GatedOp`; any other value is refused.
constexpr uint32_t ACT_SILU = 0;
constexpr uint32_t ACT_GELU = 1;

// The gate/up GEMV's launches on `G`'s gate and up views (each a GEMV of the same `n`, `pairs`
// and split): the fused CTAs, then at `splits > 1` the fused reduce, on `stream`.
template <bool EXPERT, typename OP>
int launch_gate_up(gate_up_globals G, uint32_t n, uint32_t pairs, uint32_t splits,
                   void* stream) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    void* args[] = {&G};
    const dim3 grid(padded(n) / uint32_t(BLOCK_ROWS), splits, pairs), block(NUM_THREADS);
    const cudaError_t launched =
        cudaLaunchKernel(gate_up_gemv_fp8_ker<EXPERT, OP>, grid, block, args, GEMV_SMEM_BYTES, s);
    if (launched != cudaSuccess || splits == 1) {
        return static_cast<int>(launched);
    }
    gate_up_reduce_globals R{
        {G.gate.partials, G.gate.c, int(n), int(padded(n)), int(splits), G.gate.residual, 0},
        {G.up.partials, G.up.c, int(n), int(padded(n)), int(splits), G.up.residual, 0},
    };
    void* rargs[] = {&R};
    const uint32_t vectors = pairs * padded(n) / uint32_t(REDUCE_ROWS);
    const dim3 rgrid((vectors + REDUCE_WARPS - 1) / REDUCE_WARPS), rblock(REDUCE_THREADS);
    return static_cast<int>(
        cudaLaunchKernel(gate_up_reduce_fp8_ker<OP>, rgrid, rblock, rargs, 0, s));
}

// `launch_gate_up` at the activation `act` names.
template <bool EXPERT>
int launch_gate_up_act(uint32_t act, const gate_up_globals& G, uint32_t n, uint32_t pairs,
                       uint32_t splits, void* stream) {
    return act == ACT_SILU
               ? launch_gate_up<EXPERT, scratchy::ops::silu_mul>(G, n, pairs, splits, stream)
               : launch_gate_up<EXPERT, scratchy::ops::gelu_tanh_mul>(G, n, pairs, splits, stream);
}

// The dense gate/up GEMV: each of the gate and the up meets the dense GEMV's rule at `splits`,
// with its partials at `partials` (the gate's) and past the gate's `splits · n_pad` f32 (the up's).
int dense_gate_up(const void* a, const void* gate, const void* gate_scale, const void* up,
                  const void* up_scale, void* c, void* partials, uint32_t n, uint32_t k,
                  uint32_t splits, uint32_t act, void* stream) {
    const uint32_t each = dense_split(partials, n, k, splits);
    if (each == 0 || (act != ACT_SILU && act != ACT_GELU)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        float* up_partials =
            partials == nullptr ? nullptr
                                : static_cast<float*>(partials) + size_t(splits) * padded(n);
        gate_up_globals G{dense_globals(a, gate, gate_scale, c, partials, n, k, splits, each),
                          dense_globals(a, up, up_scale, c, up_partials, n, k, splits, each)};
        return launch_gate_up_act<false>(act, G, n, 1, splits, stream);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// The expert gate/up GEMV: `expert`'s rule for the gate's and the up's slabs of one bank (each
// `[experts, expert_rows, k]` with its `[experts, expert_rows]` scales at its own base), the token
// rows read, the gate's partials at `partials` and the up's past them.
int expert_gate_up(const void* a, const void* gate, const void* gate_scale, const void* up,
                   const void* up_scale, const void* topk_idx, void* c, void* partials, uint32_t m,
                   uint32_t top_k, uint32_t n, uint32_t k, uint32_t experts, uint32_t expert_rows,
                   uint32_t splits, uint32_t act, void* stream) {
    const uint32_t each =
        expert_split(partials, m, top_k, n, k, experts, expert_rows, uint64_t(m), splits);
    if (each == 0 || (act != ACT_SILU && act != ACT_GELU)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const uint32_t pairs = m * top_k;
        float* up_partials =
            partials == nullptr
                ? nullptr
                : static_cast<float*>(partials) + size_t(splits) * size_t(pairs) * padded(n);
        gate_up_globals G{views(a, m, gate, gate_scale, experts, expert_rows, topk_idx, c,
                                partials, pairs, n, k, splits, each, int(top_k)),
                          views(a, m, up, up_scale, experts, expert_rows, topk_idx, c,
                                up_partials, pairs, n, k, splits, each, int(top_k))};
        return launch_gate_up_act<true>(act, G, n, pairs, splits, stream);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// Allow `ker` `GEMV_SMEM_BYTES` of dynamic shared memory, which loads it.
template <typename K>
cudaError_t allow(K ker) {
    return cudaFuncSetAttribute(ker, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                GEMV_SMEM_BYTES);
}

}  // namespace

// The file's one-time setup: allow every main instance `GEMV_SMEM_BYTES` of dynamic shared memory
// on the current device (the default limit is 48 KB), and load the reduces, so a worker's graph
// capture never loads a kernel mid-capture. Run once per device before any launch
// (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch.
//
// 0 = done; any other value = the cudaError_t of the first call that failed.
extern "C" int scratchy_cuda_gemv_fp8_prepare(void) {
    cudaError_t e = cudaSuccess;
    for (const cudaError_t r :
         {allow(gemv_fp8_ker<false, 4>), allow(gemv_fp8_ker<false, 3>),
          allow(gemv_fp8_ker<false, 2>), allow(gemv_fp8_ker<true, 4>),
          allow(gemv_fp8_ker<true, 3>), allow(gemv_fp8_ker<true, 2>), allow(gemv_group_fp8_ker),
          allow(gate_up_gemv_fp8_ker<false, scratchy::ops::silu_mul>),
          allow(gate_up_gemv_fp8_ker<false, scratchy::ops::gelu_tanh_mul>),
          allow(gate_up_gemv_fp8_ker<true, scratchy::ops::silu_mul>),
          allow(gate_up_gemv_fp8_ker<true, scratchy::ops::gelu_tanh_mul>),
          allow(gemv_fp8_quant_probe_ker)}) {
        if (e == cudaSuccess) {
            e = r;
        }
    }
    cudaFuncAttributes attr;
    if (e == cudaSuccess) {
        e = cudaFuncGetAttributes(&attr, gemv_reduce_ker);
    }
    if (e == cudaSuccess) {
        e = cudaFuncGetAttributes(&attr, gemv_reduce_group_ker);
    }
    for (const auto ker : {gate_up_reduce_fp8_ker<scratchy::ops::silu_mul>,
                           gate_up_reduce_fp8_ker<scratchy::ops::gelu_tanh_mul>}) {
        if (e == cudaSuccess) {
            e = cudaFuncGetAttributes(&attr, ker);
        }
    }
    return static_cast<int>(e);
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value
// = the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `c [n] = bf16((q · bᵀ) · b_scale[n] · a_scale)`, `q` and `a_scale` the per-token e4m3
// quantization of `a [1, k]` bf16 (`scratchy_cuda_quant_rows_fp8`'s), `b [n, k]` e4m3 and
// `b_scale [n]` f32, its `k` cut into `splits` runs of whole 256-element chunks
// (`step::gemv_splits`). `n` non-zero and at most INT_MAX, `k` a non-zero multiple of 64 at most
// 17,408, `splits` in 1..=16, at most `⌈k / 256⌉` and cutting the chunks into non-empty runs, and
// `splits · n_pad` at most INT_MAX (`n_pad` = `n` rounded up to 32); all are checked before any
// pointer is touched. `a`, `b` and `c` must be 16-byte-aligned device pointers, `b_scale` 4-byte
// aligned, `c` overlapping no input. At `splits > 1`, `partials` must be a 16-byte-aligned device
// pointer to `splits · n_pad` f32 overlapping no operand (null is refused), and a second launch
// adds the splits. `scratchy_cuda_gemv_fp8_prepare` must have succeeded on the current device. The
// launches are asynchronous on `stream`.
extern "C" int scratchy_cuda_gemv_fp8_abt(const void* a, const void* b, const void* b_scale,
                                          void* c, void* partials, uint32_t n, uint32_t k,
                                          uint32_t splits,
                                          void* stream /* cudaStream_t; null = default */) {
    return dense(a, b, b_scale, c, partials, n, k, splits, nullptr, stream);
}

// FP8 S8 W4a: `scratchy_cuda_gemv_fp8_abt` with the residual epilogue: `residual` (null: none) is
// a `[n]` bf16 the stored row is added to, `add_bf16`'s add on that launcher's output, written into
// `c`; it may not overlap `c`. Rule and return codes as that launcher's.
extern "C" int scratchy_cuda_gemv_fp8_residual(const void* a, const void* b, const void* b_scale,
                                               void* c, void* partials, uint32_t n, uint32_t k,
                                               uint32_t splits, const void* residual,
                                               void* stream /* cudaStream_t; null = default */) {
    return dense(a, b, b_scale, c, partials, n, k, splits, residual, stream);
}

// As `scratchy_cuda_gemv_fp8_abt` for `count` (2 or 3) members of one `a [1, k]`: member `i`
// writes `c[i] [n[i]]` from `b[i] [n[i], k]` and `b_scale[i] [n[i]]` at `splits[i]`, its partials
// (at `splits[i] > 1`) at `partials[i]`, overlapping no operand nor another member's partials.
// Each member meets that launcher's rule on its own, checked before any pointer is touched, and
// its output is that launcher's bits at the same split. Two launches at most.
extern "C" int scratchy_cuda_gemv_group_fp8_abt(const void* a, const void* const* b,
                                                const void* const* b_scale, void* const* c,
                                                void* const* partials, const uint32_t* n,
                                                uint32_t k, const uint32_t* splits, uint32_t count,
                                                void* stream /* cudaStream_t; null = default */) {
    return launch_group(a, b, b_scale, c, partials, n, k, splits, count, stream);
}

// The gate/up GEMV: `c [n] = act(g, u)`, rounded once, `g` and `u` the gate's and the up's
// `scratchy_cuda_gemv_fp8_abt` values before their rounding (each scaled by its own `b_scale` and
// by `a_scale`), `act` `ACT_SILU` or `ACT_GELU`. Return codes as the GEMV's; each of the gate and
// the up meets `scratchy_cuda_gemv_fp8_abt`'s rule at `splits`, and `act` is one of the two,
// checked before any pointer is touched. At `splits > 1`, `partials` holds `2 · splits · n_pad`
// f32 (the gate's, then the up's). Two launches at most.
extern "C" int scratchy_cuda_gate_up_gemv_fp8(const void* a, const void* gate,
                                              const void* gate_scale, const void* up,
                                              const void* up_scale, void* c, void* partials,
                                              uint32_t n, uint32_t k, uint32_t splits,
                                              uint32_t act,
                                              void* stream /* cudaStream_t; null = default */) {
    return dense_gate_up(a, gate, gate_scale, up, up_scale, c, partials, n, k, splits, act,
                         stream);
}

// The gathered expert GEMV over an fp8 bank: per pair `p`, `c[p] = bf16((q_r · B_eᵀ) ·
// b_scale[e, :n] · a_scale_r)`, `e = topk_idx[p]`, `r` the pair's row of `a` (`p / top_k`, or `p`
// when `pair_rows`), quantized as `scratchy_cuda_gemv_fp8_abt` quantizes it. `b` is expert 0's
// block of `[experts, expert_rows, k]` e4m3 and `b_scale` its row of `[experts, expert_rows]` f32
// (only the first `n` rows of each are read). The rule is `scratchy_cuda_expert_gemv_split_bf16`'s
// with `k` a multiple of 64 at most 17,408, checked before any pointer is touched; `partials`
// holds `splits · m · top_k · n` f32 at `splits > 1`. Each pair's row is
// `scratchy_cuda_gemv_fp8_abt`'s bits at the same split on its row and expert slice.
extern "C" int scratchy_cuda_expert_gemv_fp8(const void* a, const void* b, const void* b_scale,
                                             const void* topk_idx, void* c, void* partials,
                                             uint32_t m, uint32_t top_k, uint32_t n, uint32_t k,
                                             uint32_t experts, uint32_t expert_rows,
                                             uint32_t pair_rows, uint32_t splits,
                                             void* stream /* cudaStream_t; null = default */) {
    return expert(a, b, b_scale, topk_idx, c, partials, m, top_k, n, k, experts, expert_rows,
                  pair_rows, splits, stream);
}

// The expert gate/up GEMV over an fp8 bank: per pair, `scratchy_cuda_gate_up_gemv_fp8`'s fold of
// the gate's and the up's `scratchy_cuda_expert_gemv_fp8` values on the token rows, `gate` and
// `up` (with their scales) expert 0's blocks of two `[experts, expert_rows, k]` slabs. Rule as
// `scratchy_cuda_expert_gemv_fp8`'s with `pair_rows` 0, `act` as the gate/up GEMV's; `partials`
// holds `2 · splits · m · top_k · n` f32 at `splits > 1`.
extern "C" int scratchy_cuda_expert_gate_up_gemv_fp8(
    const void* a, const void* gate, const void* gate_scale, const void* up, const void* up_scale,
    const void* topk_idx, void* c, void* partials, uint32_t m, uint32_t top_k, uint32_t n,
    uint32_t k, uint32_t experts, uint32_t expert_rows, uint32_t splits, uint32_t act,
    void* stream /* cudaStream_t; null = default */) {
    return expert_gate_up(a, gate, gate_scale, up, up_scale, topk_idx, c, partials, m, top_k, n,
                          k, experts, expert_rows, splits, act, stream);
}

// Test support for `tests/gemv_fp8.rs` only: the GEMV's prologue on `a [1, k]` bf16 at `splits`,
// one CTA per split. Split `s` stores the f32 values of its chunks' quantized elements into `out`
// (`⌈k / 256⌉ · 256` f32; the elements past `k` are 0) and its scale into `scale[s]`. Refused, with
// SCRATCHY_CUDA_ERR_SHAPE before any pointer is touched, unless `k` and `splits` meet
// `scratchy_cuda_gemv_fp8_abt`'s rule. `out` 16-byte aligned, `scale` 4-byte aligned.
extern "C" int scratchy_cuda_gemv_fp8_quant_probe(const void* a, void* out, void* scale,
                                                  uint32_t k, uint32_t splits,
                                                  void* stream /* cudaStream_t; null = default */) {
    const uint32_t chunks = row_chunks(k);
    const uint32_t each = chunks == 0 ? 0 : split_chunks(chunks, splits);
    if (each == 0) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        in_gl ag{static_cast<bf16*>(const_cast<void*>(a)), nullptr, nullptr, size_t(1), size_t(k)};
        partial_gl og{static_cast<float*>(out), nullptr, nullptr, size_t(1),
                      size_t(chunks) * size_t(CHUNK)};
        partial_gl sg{static_cast<float*>(scale), nullptr, nullptr, size_t(1), size_t(splits)};
        probe_globals G{ag, og, sg, int(chunks), int(each)};
        void* args[] = {&G};
        return static_cast<int>(cudaLaunchKernel(gemv_fp8_quant_probe_ker, dim3(splits),
                                                 dim3(NUM_THREADS), args, GEMV_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
