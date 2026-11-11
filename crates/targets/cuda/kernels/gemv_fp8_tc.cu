// SPDX-License-Identifier: Apache-2.0
//
// FP8 S8's tensor-core decode GEMV for the H100 (step 1, the timing test's kernel; no tape launches
// it yet): for up to three members off one quantized activation row,
//
//     c_i[j] = bf16(((Σ_k q[k] · b_i[j, k]) · b_scale_i[j]) · a_scale)
//
// `q [1, k]` e4m3 and `a_scale` f32 are `quant_rows_fp8`'s row 0 and its scale (the activation is
// quantized once per group, before this launch); `b_i [n_i, k]` e4m3 and `b_scale_i [n_i]` f32 the
// checkpoint's weight and its per-channel scale; `c_i [n_i]` bf16. The epilogue's order is
// `gemm_fp8.cu`'s (`mul_col` by the weight scale, then by the token scale), rounded once.
//
// WGMMA e4m3 × e4m3 (`warpgroup::mma_ABt`, as `gemm_fp8.cu`): A is the activation row padded to the
// WGMMA's 64 rows (its `[1, k]` tensor map reads rows 1..63 as zeros, TK-1's R1 rule), B a 128-row
// tile of the weight, K in 64-wide steps (two `m64n128k32`s; every fp8 `k` is a multiple of 64,
// `step::GEMM_FP8_K_MULTIPLE`). Only the accumulator's row 0 is the GEMV; the others stay zero.
//
// The grid is stream-K: the members' 128-row tiles, concatenated, times the K steps are one
// sequence of `units`; CTA `c` of `CTAS` (132, one per H100 SXM SM) takes the contiguous run
// `[⌊c · units / CTAS⌋, ⌊(c + 1) · units / CTAS⌋)`, so every SM streams the same bytes to within
// one 8 KB step at any shape (the 27B gate+up's 34,816 rows are 2^11 · 17, which no tile count
// divides into 132). A producer warp TMA-loads A and B through a `STAGES`-deep ring; one consumer
// warpgroup runs the WGMMAs. Each run of a tile inside a CTA (a "piece") ends with its unscaled
// row 0 (`warp::col_sum` of warp 0's 16 rows: rows 1..15 are exact zeros) stored to `partials`
// row `c + t` (unique: along the sequence `c` and `t` never decrease and never both stay). Then
// `gemv_fp8_tc_reduce_ker` adds a tile's pieces in ascending K order, scales, rounds.
//
// Bits are a function of the shape alone (CTAS is a constant, not the device's SM count). They are
// not `gemv_fp8.cu`'s, and a member of a group is not its own launch's (the pieces fall elsewhere):
// the approved kind of change (WGMMA accumulation, K order), against S3's f64 bound.
//
// Shared memory: 12 stages × (a 4 KB A tile and an 8 KB B tile) plus 1,024 bytes of alignment
// slack (B1; the ceiling probe's instances below differ), set by
// `scratchy_cuda_gemv_fp8_tc_prepare`.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>
#include <type_traits>

using namespace kittens;

// The launcher's return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors them.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// One CTA per SM of an H100 SXM; the bits depend on it, so it is a constant.
constexpr int CTAS = 132;
constexpr int TILE_N = 128;
constexpr int K_STEP = 64;
// The ceiling probe's B4 (`2cta`): two CTAs per SM. Every `partials` is sized for it.
constexpr int MAX_CTAS = 264;
constexpr int NUM_THREADS = 2 * kittens::WARPGROUP_THREADS;
constexpr int REDUCE_ROWS = 32;
constexpr int REDUCE_WARPS = 8;
constexpr int REDUCE_THREADS = REDUCE_WARPS * kittens::WARP_THREADS;
// `n` a multiple of it: the reduce's 32-row vectors and the scales' reads stay inside the arrays.
constexpr uint32_t N_MULTIPLE = 64;

// The kernel's instance: its K step, ring, CTAs per SM and WGMMA release. `tc_b1` is the one this
// file's launcher runs; the others are the ceiling probe's test-only knobs (/tmp/d-cap.md §3 P-B),
// launched only by `scratchy_cuda_gemv_fp8_tc_variant`:
//   B2 `wait1`: the stage before is released after `mma_async_wait<1>()`, so one WGMMA group stays
//     in flight (the same WGMMAs in the same order: the same bits);
//   B3 `k128`: 128-wide K steps (`st_fp8e4m3<128, 128>` B and `<64, 128>` A boxes, 128-byte
//     swizzle), four `m64n128k32` a stage, 8 stages;
//   B4 `2cta`: 264 CTAs, two per SM, 6 stages each;
//   B6 `k256`: B3 at 256-wide K steps (`st_fp8e4m3<128, 256>` B, `<64, 256>` A: P-A's 256-byte
//     rows), eight `m64n128k32` a stage, 4 stages (193 KB).
//   B8 `swapab32`: vLLM's geometry (its `cutlass_3x_gemm_sm90_fp8` at TileShape 64x16x256) at the
//     smallest fp8 N TK has at the pin (32): the weight's 64-row, 256-wide tile is the WGMMA's M
//     side and the activation's `st_fp8e4m3<32, 256>` (1 real row, 31 TMA zero fill) its N side,
//     so `D` is 64 × 32 and its column 0 the GEMV (`warp::row_sum`: the other columns are exact
//     zeros). One CTA per 64 weight rows over the whole K, no stream-K, its epilogue stored
//     directly; two CTAs per SM, 4 stages each (2 × 97 KB);
//   B9 `swapab32sk`: B8's tile under stream-K (264 CTAs), its pieces through the reduce.
// B3 to B9 change the K grouping and the split: the approved kind, under S3's bound.
// `KIND` 0 is the activation as the M side (B1 to B6), 1 B8, 2 B9.
template <int K_STEP_, int STAGES_, int CTAS_, bool WAIT1_, int TILE_ = TILE_N, int KIND_ = 0>
struct tc_variant {
    static constexpr int k_step = K_STEP_;
    static constexpr int stages = STAGES_;
    static constexpr int ctas = CTAS_;
    static constexpr bool wait1 = WAIT1_;
    static constexpr int tile = TILE_;
    static constexpr int kind = KIND_;
    static constexpr int min_blocks = CTAS_ / CTAS;
    // The activation's box (the M side's 64 rows, or B8's N side's 32) and the weight's.
    using a_tile = st_fp8e4m3<KIND_ == 0 ? 64 : 32, K_STEP_>;
    using b_tile = st_fp8e4m3<TILE_, K_STEP_>;
    static constexpr size_t smem = STAGES_ * (sizeof(a_tile) + sizeof(b_tile)) + 1024;
    static_assert(min_blocks * smem <= kittens::MAX_SHARED_MEMORY, "the CTAs per SM fit");
};
using tc_b1 = tc_variant<K_STEP, 12, CTAS, false>;
using tc_b2 = tc_variant<K_STEP, 12, CTAS, true>;
using tc_b3 = tc_variant<128, 8, CTAS, false>;
using tc_b4 = tc_variant<K_STEP, 6, MAX_CTAS, false>;
using tc_b6 = tc_variant<256, 4, CTAS, false>;
using tc_b8 = tc_variant<256, 4, MAX_CTAS, false, 64, 1>;
using tc_b9 = tc_variant<256, 4, MAX_CTAS, false, 64, 2>;

using acc_tile = rt_fl<16, TILE_N>;
using piece_rv = acc_tile::row_vec;
using reduce_rv = rv_fl<REDUCE_ROWS>;
using a_gl = gl<fp8e4m3, 1, 1, -1, -1, tc_b1::a_tile, tc_b3::a_tile, tc_b6::a_tile, tc_b8::a_tile>;
using b_gl = gl<fp8e4m3, 1, 1, -1, -1, tc_b1::b_tile, tc_b3::b_tile, tc_b6::b_tile, tc_b8::b_tile>;
// `[CTAS + tiles, 128]` f32 pieces, `[1, n]` scales and outputs: no tensor map.
using partial_gl = gl<float, 1, 1, -1, -1>;
using scale_gl = gl<float, 1, 1, -1, -1>;
using out_gl = gl<bf16, 1, 1, -1, -1>;


struct tc_globals {
    a_gl a;
    b_gl b0, b1, b2;
    partial_gl partials;
    // The members' scales and outputs (the reduce's). An absent member repeats the last.
    scale_gl s0, s1, s2;
    out_gl c0, c1, c2;
    scale_gl a_scale;
    int n0, n1, n2;
    // The first tile of members 1 and 2 (`tiles` for an absent one), all tiles, the K steps.
    int start1, start2, tiles, k_steps, ctas;
};

// The first unit of CTA `c`'s run.
__device__ inline long long run_start(int c, long long units, int ctas) {
    return units * c / ctas;
}

// The CTA whose run holds unit `u`.
__device__ inline int cta_of(long long u, long long units, int ctas) {
    int c = int(u * ctas / units);
    while (c + 1 < ctas && run_start(c + 1, units, ctas) <= u) {
        c++;
    }
    while (run_start(c, units, ctas) > u) {
        c--;
    }
    return c;
}

template <typename V>
__global__ __launch_bounds__(NUM_THREADS, V::min_blocks) void gemv_fp8_tc_ker(
    const __grid_constant__ tc_globals g) {
    using a_tile = typename V::a_tile;
    using b_tile = typename V::b_tile;
    constexpr int STAGES = V::stages;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    a_tile(&a_s)[STAGES] = al.allocate<a_tile, STAGES>();
    b_tile(&b_s)[STAGES] = al.allocate<b_tile, STAGES>();
    __shared__ semaphore arrived[STAGES], done[STAGES];
    const long long units = (long long)g.tiles * g.k_steps;
    const int c = blockIdx.x;
    const long long u0 = run_start(c, units, g.ctas);
    const int len = int(run_start(c + 1, units, g.ctas) - u0);
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_semaphore(arrived[s], 0, 1);
            init_semaphore(done[s], 1, 0);
        }
    }
    __syncthreads();
    if (warpgroup::groupid() == 1) {
        if constexpr (V::min_blocks == 2) {
            // B4: two CTAs fit an SM's registers only with the producer's handed to the consumer
            // (128 · 24 + 128 · 232 = 32,768, half the SM's file).
            warpgroup::decrease_registers<24>();
        }
        // The producer: warp 0 of the second warpgroup.
        if (warpgroup::warpid() == 0) {
            for (int i = 0; i < len; i++) {
                const long long u = u0 + i;
                const int t = int(u / g.k_steps), j = int(u % g.k_steps), s = i % STAGES;
                if (i >= STAGES) {
                    wait(done[s], ((i - STAGES) / STAGES) % 2);
                }
                const b_gl& b = t >= g.start2 ? g.b2 : t >= g.start1 ? g.b1 : g.b0;
                const int local = t >= g.start2 ? t - g.start2 : t >= g.start1 ? t - g.start1 : t;
                warp::tma::expect_bytes(arrived[s], sizeof(a_tile) + sizeof(b_tile));
                warp::tma::load_async(a_s[s], g.a, {0, 0, 0, j}, arrived[s]);
                warp::tma::load_async(b_s[s], b, {0, 0, local, j}, arrived[s]);
            }
        }
    } else {
        if constexpr (V::min_blocks == 2) {
            warpgroup::increase_registers<232>();
        }
        acc_tile acc;
        warp::zero(acc);
        // B2: the stage whose WGMMA group may still be in flight, or -1.
        int pending = -1;
        const auto release = [&](int stage) {
            if (warpgroup::laneid() == 0) {
                arrive(done[stage], 1);
            }
        };
        for (int i = 0; i < len; i++) {
            const long long u = u0 + i;
            const int t = int(u / g.k_steps), j = int(u % g.k_steps), s = i % STAGES;
            wait(arrived[s], (i / STAGES) % 2);
            warpgroup::mma_ABt(acc, a_s[s], b_s[s]);
            const bool end = j == g.k_steps - 1 || i == len - 1;
            if constexpr (V::wait1) {
                // Every group but this stage's is done: release the one before.
                warpgroup::mma_async_wait<1>();
                if (pending >= 0) {
                    release(pending);
                }
                pending = s;
                if (end) {
                    warpgroup::mma_async_wait();
                    release(s);
                    pending = -1;
                }
            } else {
                // The WGMMA reads the stage and writes `acc` asynchronously: wait before releasing
                // the stage and before anything reads `acc`.
                warpgroup::mma_async_wait();
                release(s);
            }
            if (end) {
                // The piece's end: warp 0 holds rows 0..15, row 0 the GEMV's.
                if (warpgroup::warpid() == 0) {
                    piece_rv v;
                    warp::col_sum(v, acc);
                    warp::store(g.partials, v, {0, 0, c + t, 0});
                }
                warp::zero(acc);
            }
        }
    }
}

// B8 and B9: the weight tile as the WGMMA's M side, the activation's 32-row box as its N side.
template <typename V>
__global__ __launch_bounds__(NUM_THREADS, V::min_blocks) void gemv_fp8_swapab_ker(
    const __grid_constant__ tc_globals g) {
    using x_tile = typename V::a_tile;
    using w_tile = typename V::b_tile;
    using acc32 = rt_fl<16, 32>;
    constexpr int STAGES = V::stages;
    constexpr bool SK = V::kind == 2;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    w_tile(&w_s)[STAGES] = al.allocate<w_tile, STAGES>();
    x_tile(&x_s)[STAGES] = al.allocate<x_tile, STAGES>();
    __shared__ semaphore arrived[STAGES], done[STAGES];
    const long long units = (long long)g.tiles * g.k_steps;
    const int c = blockIdx.x;
    // B9's stream-K run, or B8's whole tile `c`.
    const long long u0 = SK ? run_start(c, units, g.ctas) : (long long)c * g.k_steps;
    const int len = SK ? int(run_start(c + 1, units, g.ctas) - u0) : g.k_steps;
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_semaphore(arrived[s], 0, 1);
            init_semaphore(done[s], 1, 0);
        }
    }
    __syncthreads();
    if (warpgroup::groupid() == 1) {
        if (warpgroup::warpid() == 0) {
            for (int i = 0; i < len; i++) {
                const long long u = u0 + i;
                const int t = int(u / g.k_steps), j = int(u % g.k_steps), s = i % STAGES;
                if (i >= STAGES) {
                    wait(done[s], ((i - STAGES) / STAGES) % 2);
                }
                const b_gl& b = t >= g.start2 ? g.b2 : t >= g.start1 ? g.b1 : g.b0;
                const int local = t >= g.start2 ? t - g.start2 : t >= g.start1 ? t - g.start1 : t;
                warp::tma::expect_bytes(arrived[s], sizeof(x_tile) + sizeof(w_tile));
                warp::tma::load_async(x_s[s], g.a, {0, 0, 0, j}, arrived[s]);
                warp::tma::load_async(w_s[s], b, {0, 0, local, j}, arrived[s]);
            }
        }
    } else {
        acc32 acc;
        warp::zero(acc);
        for (int i = 0; i < len; i++) {
            const long long u = u0 + i;
            const int t = int(u / g.k_steps), j = int(u % g.k_steps), s = i % STAGES;
            wait(arrived[s], (i / STAGES) % 2);
            warpgroup::mma_ABt(acc, w_s[s], x_s[s]);
            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) {
                arrive(done[s], 1);
            }
            if (j == g.k_steps - 1 || i == len - 1) {
                // Warp `w` holds weight rows `16w .. 16w + 16` of the tile; column 0 is theirs.
                const int w = warpgroup::warpid();
                typename acc32::col_vec v;
                warp::row_sum(v, acc);
                if constexpr (SK) {
                    warp::store(g.partials, v, {0, 0, c + t, w});
                } else {
                    const int m = t >= g.start2 ? 2 : t >= g.start1 ? 1 : 0;
                    const int local = t - (m == 2 ? g.start2 : m == 1 ? g.start1 : 0);
                    const int row = local * V::tile + 16 * w;
                    const scale_gl& sc = m == 2 ? g.s2 : m == 1 ? g.s1 : g.s0;
                    const out_gl& out = m == 2 ? g.c2 : m == 1 ? g.c1 : g.c0;
                    typename acc32::col_vec ws;
                    warp::load(ws, sc, {0, 0, 0, row / 16});
                    warp::mul(v, v, ws);
                    warp::mul(v, v, g.a_scale[{0, 0, 0, 0}]);
                    warp::store(out, v, {0, 0, 0, row / 16});
                }
                warp::zero(acc);
            }
        }
    }
}

// Rows `32v .. 32v + 32` of the concatenated members: their tile's pieces added in ascending K
// order, then `· b_scale`, `· a_scale`, rounded once. `TILE` is the kernel's tile rows (128, or B9's 64).
template <int TILE>
__global__ __launch_bounds__(REDUCE_THREADS) void gemv_fp8_tc_reduce_ker(
    const __grid_constant__ tc_globals g) {
    const int v = blockIdx.x * REDUCE_WARPS + kittens::warpid();
    const int t = v * REDUCE_ROWS / TILE;
    if (t >= g.tiles) {
        return;
    }
    const int m = t >= g.start2 ? 2 : t >= g.start1 ? 1 : 0;
    const int start = m == 2 ? g.start2 : m == 1 ? g.start1 : 0;
    const int n = m == 2 ? g.n2 : m == 1 ? g.n1 : g.n0;
    // The vector's first row inside its member, and inside its tile.
    const int row = (t - start) * TILE + (v * REDUCE_ROWS) % TILE;
    if (row >= n) {
        return;
    }
    const long long units = (long long)g.tiles * g.k_steps;
    const int first = cta_of((long long)t * g.k_steps, units, g.ctas);
    const int last = cta_of((long long)(t + 1) * g.k_steps - 1, units, g.ctas);
    const int col = (v * REDUCE_ROWS) % TILE / REDUCE_ROWS;
    reduce_rv acc, part, ws;
    warp::zero(acc);
    for (int cta = first; cta <= last; cta++) {
        warp::load(part, g.partials, {0, 0, cta + t, col});
        warp::add(acc, acc, part);
    }
    const scale_gl& s = m == 2 ? g.s2 : m == 1 ? g.s1 : g.s0;
    const out_gl& out = m == 2 ? g.c2 : m == 1 ? g.c1 : g.c0;
    warp::load(ws, s, {0, 0, 0, row / REDUCE_ROWS});
    warp::mul(acc, acc, ws);
    warp::mul(acc, acc, g.a_scale[{0, 0, 0, 0}]);
    warp::store(out, acc, {0, 0, 0, row / REDUCE_ROWS});
}

int tiles_of(uint32_t n, int tile = TILE_N) {
    return int((n + uint32_t(tile) - 1) / uint32_t(tile));
}

// Instance `V`'s main kernel.
template <typename V>
constexpr auto main_ker() {
    if constexpr (V::kind == 0) {
        return gemv_fp8_tc_ker<V>;
    } else {
        return gemv_fp8_swapab_ker<V>;
    }
}

template <typename V>
cudaError_t allow() {
    return cudaFuncSetAttribute(main_ker<V>(), cudaFuncAttributeMaxDynamicSharedMemorySize,
                                int(V::smem));
}

// The views of one launch at instance `V`'s K step and CTA count. Can throw (TK's host code).
template <typename V>
tc_globals views(const void* q, const void* a_scale, const void* const* b,
                 const void* const* b_scale, void* const* c, const uint32_t* n, uint32_t count,
                 uint32_t k, void* partials) {
    const uint32_t last = count - 1;
    const auto bv = [&](uint32_t i) {
        return b_gl{static_cast<fp8e4m3*>(const_cast<void*>(b[i])), nullptr, nullptr,
                    size_t(n[i]), size_t(k)};
    };
    const auto sv = [&](uint32_t i) {
        return scale_gl{static_cast<float*>(const_cast<void*>(b_scale[i])), nullptr, nullptr,
                        size_t(1), size_t(n[i])};
    };
    const auto cv = [&](uint32_t i) {
        return out_gl{static_cast<bf16*>(c[i]), nullptr, nullptr, size_t(1), size_t(n[i])};
    };
    const int t0 = tiles_of(n[0], V::tile);
    const int t1 = count > 1 ? tiles_of(n[1], V::tile) : 0;
    const int t2 = count > 2 ? tiles_of(n[2], V::tile) : 0;
    const int tiles = t0 + t1 + t2;
    const int k_steps = int(k / uint32_t(V::k_step));
    const long long units = (long long)tiles * k_steps;
    const int ctas = units < V::ctas ? int(units) : V::ctas;
    return tc_globals{
        a_gl{static_cast<fp8e4m3*>(const_cast<void*>(q)), nullptr, nullptr, size_t(1), size_t(k)},
        bv(0), bv(count > 1 ? 1 : last), bv(last),
        partial_gl{static_cast<float*>(partials), nullptr, nullptr, size_t(ctas + tiles),
                   size_t(V::tile)},
        sv(0), sv(count > 1 ? 1 : last), sv(last),
        cv(0), cv(count > 1 ? 1 : last), cv(last),
        scale_gl{static_cast<float*>(const_cast<void*>(a_scale)), nullptr, nullptr, size_t(1),
                 size_t(1)},
        int(n[0]), int(n[count > 1 ? 1 : last]), int(n[last]),
        // An absent member starts at `tiles`: no tile is its.
        count > 1 ? t0 : tiles, count > 2 ? t0 + t1 : tiles, tiles, k_steps, ctas};
}

// The stream-K kernel at instance `V`, then (unless `main_only`) its reduce, on `stream`.
template <typename V>
int launch(const void* q, const void* a_scale, const void* const* b, const void* const* b_scale,
           void* const* c, const uint32_t* n, uint32_t count, uint32_t k, void* partials,
           bool main_only, void* stream) {
    if (k % uint32_t(V::k_step) != 0) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    tc_globals G = views<V>(q, a_scale, b, b_scale, c, n, count, k, partials);
    void* args[] = {&G};
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    // B8 runs one CTA per tile, every other instance its stream-K CTAs.
    const int grid = V::kind == 1 ? G.tiles : G.ctas;
    cudaError_t e = cudaLaunchKernel(main_ker<V>(), dim3(grid), dim3(NUM_THREADS), args, V::smem, st);
    // B8 stores its rows itself: it has no reduce.
    if (e != cudaSuccess || main_only || V::kind == 1) {
        return static_cast<int>(e);
    }
    const int vectors = G.tiles * (V::tile / REDUCE_ROWS);
    return static_cast<int>(cudaLaunchKernel(gemv_fp8_tc_reduce_ker<V::tile>,
                                             dim3((vectors + REDUCE_WARPS - 1) / REDUCE_WARPS),
                                             dim3(REDUCE_THREADS), args, 0, st));
}

// ── B8 in production form (FP8 S8 T1, /tmp/a-tc-plan.md §1, §5.2) ───────────────────────────
//
// `tc_b8`'s geometry and its main loop, unchanged: the weight's 64-row, 256-wide tile is the
// WGMMA's M side, the activation's `st_fp8e4m3<32, 256>` box its N side (1 real row, 31 rows of
// TMA zero fill), `D [64, 32]`'s column 0 the GEMV (`warp::row_sum`: the other columns are exact
// zeros), one CTA per 64-row tile over the whole K, 4 stages, two CTAs per SM. What T1 adds:
//   - **The expert form** (`EXPERT`): grid `(tiles, pairs)`; pair `p` reads weight depth
//     `topk_idx[p]` of `[experts, expert_rows, k]` and activation row `p / row_div`
//     (`gemv_fp8.cu`'s `EXPERT` contract). The dense form is `experts` 1, `topk_idx` unread. A
//     pair is the dense GEMV on its slice, bit for bit: the same CTA on the same bytes.
//   - **The activation as `[rows, 1, k]`**: each row its own depth, so every box's rows past 0 are
//     TMA's zero fill even when the activation holds several rows (the expert pairs' 8). Without
//     it a box would read the next rows into `D`'s other columns, which the row sum adds.
//   - **The K tail**: `⌈k / 256⌉` steps; both tensor maps' K extent is `k`, so TMA zero-fills both
//     boxes past it (`tests/gemv_fp8_b8.rs`'s swizzled-box probe), and a zero adds 0 exactly.
//   - **The residual epilogue** (`add`): `round_bf16`, the residual row widened, one f32 add, the
//     store's one rounding: `scratchy::ops::with_residual`'s bits (W4a's `MatvecEpilogue`).
//   - **The f32 store** (`F32`): the scaled f32 sums, unrounded, for the gate/up members, so the
//     act + quantize step after them rounds the activation once (`act_quant_fp8.cu`).
//   - Up to 3 members, as `tc_b8`: each tile's member by `start1`/`start2`; each member is its own
//     launch's bits (no split: a tile's CTA does not depend on the other members).
// Bits are a function of the shape alone. They are not `gemv_fp8.cu`'s (WGMMA's fp8 accumulation
// and K grouping): the approved kind, against S3's f64 bound.

using b8_x_tile = tc_b8::a_tile;
using b8_w_tile = tc_b8::b_tile;
// The activation `[rows, 1, k]` e4m3, and a member's weight `[depth, rows, k]` e4m3.
using b8_x_gl = gl<fp8e4m3, 1, -1, 1, -1, b8_x_tile>;
using b8_w_gl = gl<fp8e4m3, 1, -1, -1, -1, b8_w_tile>;
// No tensor map: the weight scales `[depth, rows]`, the activation's `[rows]`, the outputs and
// residuals `[pairs, n]`, the experts `[pairs]`.
using b8_scale_gl = gl<float, 1, 1, -1, -1>;
using b8_ascale_gl = gl<float, 1, 1, 1, -1>;
template <bool F32>
using b8_out_gl = gl<std::conditional_t<F32, float, bf16>, 1, 1, -1, -1>;
using b8_res_gl = gl<bf16, 1, 1, -1, -1>;
using b8_idx_gl = gl<int, 1, 1, 1, -1>;

template <bool F32>
struct b8_globals {
    b8_x_gl x;
    b8_ascale_gl a_scale;
    b8_w_gl b0, b1, b2;
    b8_scale_gl s0, s1, s2;
    b8_out_gl<F32> c0, c1, c2;
    b8_res_gl r0, r1, r2;
    b8_idx_gl topk_idx;
    // The first tile of members 1 and 2 (the tile count for an absent one), the K steps, the
    // activation row of pair `p` (`p / row_div`), and the residual epilogue's switch.
    int start1, start2, k_steps, row_div, add;
};

template <bool EXPERT, bool F32>
__global__ __launch_bounds__(NUM_THREADS, tc_b8::min_blocks) void gemv_fp8_b8_ker(
    const __grid_constant__ b8_globals<F32> g) {
    using acc32 = rt_fl<16, 32>;
    constexpr int STAGES = tc_b8::stages;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    b8_w_tile(&w_s)[STAGES] = al.allocate<b8_w_tile, STAGES>();
    b8_x_tile(&x_s)[STAGES] = al.allocate<b8_x_tile, STAGES>();
    __shared__ semaphore arrived[STAGES], done[STAGES];
    const int t = blockIdx.x, p = blockIdx.y;
    const int m = t >= g.start2 ? 2 : t >= g.start1 ? 1 : 0;
    const int local = t - (m == 2 ? g.start2 : m == 1 ? g.start1 : 0);
    const int xrow = p / g.row_div;
    int depth = 0;
    if constexpr (EXPERT) {
        depth = g.topk_idx[{0, 0, 0, p}];
    }
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_semaphore(arrived[s], 0, 1);
            init_semaphore(done[s], 1, 0);
        }
    }
    __syncthreads();
    if (warpgroup::groupid() == 1) {
        // The producer: warp 0 of the second warpgroup.
        if (warpgroup::warpid() == 0) {
            const b8_w_gl& b = m == 2 ? g.b2 : m == 1 ? g.b1 : g.b0;
            for (int j = 0; j < g.k_steps; j++) {
                const int s = j % STAGES;
                if (j >= STAGES) {
                    wait(done[s], ((j - STAGES) / STAGES) % 2);
                }
                warp::tma::expect_bytes(arrived[s], sizeof(b8_x_tile) + sizeof(b8_w_tile));
                warp::tma::load_async(x_s[s], g.x, {0, xrow, 0, j}, arrived[s]);
                warp::tma::load_async(w_s[s], b, {0, depth, local, j}, arrived[s]);
            }
        }
    } else {
        acc32 acc;
        warp::zero(acc);
        for (int j = 0; j < g.k_steps; j++) {
            const int s = j % STAGES;
            wait(arrived[s], (j / STAGES) % 2);
            warpgroup::mma_ABt(acc, w_s[s], x_s[s]);
            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) {
                arrive(done[s], 1);
            }
        }
        // Warp `w` holds weight rows `16w .. 16w + 16` of the tile; column 0 is theirs.
        const int w = warpgroup::warpid();
        const int row = local * tc_b8::tile + 16 * w;
        typename acc32::col_vec v, side;
        warp::row_sum(v, acc);
        warp::load(side, m == 2 ? g.s2 : m == 1 ? g.s1 : g.s0, {0, 0, depth, row / 16});
        warp::mul(v, v, side);
        warp::mul(v, v, g.a_scale[{0, 0, 0, xrow}]);
        if (g.add) {
            warp::unary_op<scratchy::ops::round_bf16>(v, v);
            warp::load(side, m == 2 ? g.r2 : m == 1 ? g.r1 : g.r0, {0, 0, p, row / 16});
            warp::add(v, v, side);
        }
        warp::store(m == 2 ? g.c2 : m == 1 ? g.c1 : g.c0, v, {0, 0, p, row / 16});
    }
}

// The swizzled-box probe (test support, `tests/gemv_fp8_b8.rs`, TK-1's R1 on B8's 128-byte-swizzled
// fp8 boxes): each box first TMA-loaded with all-NaN `poison`, then B8's own box at K step `j`,
// activation row `row` (the `[rows, 1, k]` view) and weight tile `tile`, then both TMA-stored to
// exact-fit dumps. Every byte past `k`, past the activation's row 0 or past the weight's rows must
// come back 0, not NaN.
using b8_dump_x_gl = gl<fp8e4m3, 1, 1, -1, -1, b8_x_tile>;
using b8_dump_w_gl = gl<fp8e4m3, 1, 1, -1, -1, b8_w_tile>;

struct b8_probe_globals {
    b8_x_gl x;
    b8_w_gl w;
    b8_dump_x_gl poison_x, dump_x;
    b8_dump_w_gl poison_w, dump_w;
    int row, tile, j;
};

__global__ void gemv_fp8_b8_box_probe_ker(const __grid_constant__ b8_probe_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    b8_x_tile& xs = al.allocate<b8_x_tile>();
    b8_w_tile& ws = al.allocate<b8_w_tile>();
    __shared__ semaphore poisoned, loaded;
    if (warp::laneid() == 0) {
        init_semaphore(poisoned, 0, 1);
        init_semaphore(loaded, 0, 1);
    }
    __syncwarp();
    warp::tma::expect_bytes(poisoned, sizeof(b8_x_tile) + sizeof(b8_w_tile));
    warp::tma::load_async(xs, g.poison_x, {0, 0, 0, 0}, poisoned);
    warp::tma::load_async(ws, g.poison_w, {0, 0, 0, 0}, poisoned);
    wait(poisoned, 0);
    warp::tma::expect_bytes(loaded, sizeof(b8_x_tile) + sizeof(b8_w_tile));
    warp::tma::load_async(xs, g.x, {0, g.row, 0, g.j}, loaded);
    warp::tma::load_async(ws, g.w, {0, 0, g.tile, g.j}, loaded);
    wait(loaded, 0);
    warp::tma::store_async(g.dump_x, xs, {0, 0, 0, 0});
    warp::tma::store_async(g.dump_w, ws, {0, 0, 0, 0});
    warp::tma::store_async_wait();
}

constexpr int B8_PROBE_SMEM = int(sizeof(b8_x_tile) + sizeof(b8_w_tile)) + 1024;

// B8's production instances.
template <bool EXPERT, bool F32>
cudaError_t allow_b8() {
    return cudaFuncSetAttribute(gemv_fp8_b8_ker<EXPERT, F32>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, int(tc_b8::smem));
}

// A B8 launch's operands, as the launcher was given them.
struct b8_args {
    const void* q = nullptr;
    const void* a_scale = nullptr;
    uint32_t a_rows = 0;
    const void* const* b = nullptr;
    const void* const* b_scale = nullptr;
    void* const* c = nullptr;
    const void* const* residual = nullptr;
    const uint32_t* n = nullptr;
    uint32_t count = 0;
    uint32_t k = 0;
    uint32_t experts = 0;
    uint32_t expert_rows = 0;
    const void* topk_idx = nullptr;
    uint32_t pairs = 0;
    uint32_t row_div = 0;
};

// The views and counts of `a`'s launch, then the launch on `stream`. Can throw (TK's host code).
template <bool EXPERT, bool F32>
int launch_b8(const b8_args& a, void* stream) {
    using out_t = std::conditional_t<F32, float, bf16>;
    const uint32_t last = a.count - 1;
    const auto at = [&](uint32_t i) { return i < a.count ? i : last; };
    const auto rows = [&](uint32_t i) { return EXPERT ? a.expert_rows : a.n[at(i)]; };
    const auto wv = [&](uint32_t i) {
        return b8_w_gl{static_cast<fp8e4m3*>(const_cast<void*>(a.b[at(i)])), nullptr,
                       size_t(a.experts), size_t(rows(i)), size_t(a.k)};
    };
    const auto sv = [&](uint32_t i) {
        return b8_scale_gl{static_cast<float*>(const_cast<void*>(a.b_scale[at(i)])), nullptr,
                           nullptr, size_t(a.experts), size_t(rows(i))};
    };
    const auto cv = [&](uint32_t i) {
        return b8_out_gl<F32>{static_cast<out_t*>(a.c[at(i)]), nullptr, nullptr,
                              size_t(a.pairs), size_t(a.n[at(i)])};
    };
    // A member with no residual views its own output (never read: `add` is 0 for the launch).
    const auto rv = [&](uint32_t i) {
        const void* r = a.residual != nullptr ? a.residual[at(i)] : a.c[at(i)];
        return b8_res_gl{static_cast<bf16*>(const_cast<void*>(r)), nullptr, nullptr,
                         size_t(a.pairs), size_t(a.n[at(i)])};
    };
    int tiles[3] = {0, 0, 0};
    for (uint32_t i = 0; i < a.count; i++) {
        tiles[i] = tiles_of(a.n[i], tc_b8::tile);
    }
    const int all = tiles[0] + tiles[1] + tiles[2];
    // With no experts the view is never read; `a_scale` is a valid address for it.
    const void* idx = EXPERT ? a.topk_idx : a.a_scale;
    b8_globals<F32> G{
        b8_x_gl{static_cast<fp8e4m3*>(const_cast<void*>(a.q)), nullptr, size_t(a.a_rows), nullptr,
                size_t(a.k)},
        b8_ascale_gl{static_cast<float*>(const_cast<void*>(a.a_scale)), nullptr, nullptr, nullptr,
                     size_t(a.a_rows)},
        wv(0), wv(1), wv(2),
        sv(0), sv(1), sv(2),
        cv(0), cv(1), cv(2),
        rv(0), rv(1), rv(2),
        b8_idx_gl{static_cast<int*>(const_cast<void*>(idx)), nullptr, nullptr, nullptr,
                  size_t(EXPERT ? a.pairs : 1)},
        a.count > 1 ? tiles[0] : all,
        a.count > 2 ? tiles[0] + tiles[1] : all,
        int((a.k + uint32_t(tc_b8::k_step) - 1) / uint32_t(tc_b8::k_step)),
        int(a.row_div),
        a.residual != nullptr ? 1 : 0};
    void* args[] = {&G};
    const dim3 grid{uint32_t(all), a.pairs}, block{uint32_t(NUM_THREADS)};
    return static_cast<int>(cudaLaunchKernel(gemv_fp8_b8_ker<EXPERT, F32>, grid, block, args,
                                             tc_b8::smem, static_cast<cudaStream_t>(stream)));
}

// Whether `a` is refused (`scratchy_cuda_gemv_fp8_b8`'s list).
bool b8_refused(const b8_args& a, bool f32) {
    constexpr uint64_t LIMIT = uint64_t(INT_MAX);
    const bool expert = a.topk_idx != nullptr;
    if (a.count < 1 || a.count > 3 || a.k == 0 || a.k % N_MULTIPLE != 0 || a.pairs == 0 ||
        a.pairs > 65535 || a.row_div == 0 || a.a_rows == 0 ||
        uint64_t(a.pairs - 1) / a.row_div >= a.a_rows || a.experts == 0 ||
        (!expert && a.experts != 1) || (f32 && a.residual != nullptr) || a.q == nullptr ||
        a.a_scale == nullptr || uint64_t(a.a_rows) * a.k > LIMIT) {
        return true;
    }
    for (uint32_t i = 0; i < a.count; i++) {
        const uint64_t rows = expert ? a.expert_rows : a.n[i];
        if (a.n[i] == 0 || a.n[i] % N_MULTIPLE != 0 || rows < a.n[i] ||
            uint64_t(a.experts) * rows * a.k > LIMIT || uint64_t(a.pairs) * a.n[i] > LIMIT ||
            a.b[i] == nullptr || a.b_scale[i] == nullptr || a.c[i] == nullptr ||
            (a.residual != nullptr && a.residual[i] == nullptr)) {
            return true;
        }
    }
    return false;
}

}  // namespace

// The kernels' one-time setup: allow every instance its dynamic shared memory on the current
// device (B1, the launcher's, and the ceiling probe's B2 to B4, so no probe launch sets an
// attribute inside a timed region). 0 = done; any other value = the first failing cudaError_t.
extern "C" int scratchy_cuda_gemv_fp8_tc_prepare(void) {
    cudaError_t e = cudaSuccess;
    for (const cudaError_t r :
         {allow<tc_b1>(), allow<tc_b2>(), allow<tc_b3>(), allow<tc_b4>(), allow<tc_b6>(),
          allow<tc_b8>(), allow<tc_b9>(), allow_b8<false, false>(), allow_b8<false, true>(),
          allow_b8<true, false>(), allow_b8<true, true>()}) {
        if (e == cudaSuccess) {
            e = r;
        }
    }
    return static_cast<int>(e);
}

// The `partials` bytes a launch of these members needs at any instance:
// `(min(264, units) + tiles) · 128` f32 (264 is B4's CTAs; B1 uses at most 132 of those rows).
// 0 = the shape is refused.
extern "C" uint64_t scratchy_cuda_gemv_fp8_tc_partial_bytes(const uint32_t* n, uint32_t count,
                                                           uint32_t k) {
    if (count < 1 || count > 3 || k == 0 || k % K_STEP != 0 || k > uint32_t(INT_MAX)) {
        return 0;
    }
    uint64_t tiles = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (n[i] == 0 || n[i] % N_MULTIPLE != 0 || n[i] > uint32_t(INT_MAX) / 2) {
            return 0;
        }
        tiles += uint64_t(tiles_of(n[i]));
    }
    const uint64_t units = tiles * (k / K_STEP);
    const uint64_t ctas = units < uint64_t(MAX_CTAS) ? units : uint64_t(MAX_CTAS);
    return (ctas + tiles) * TILE_N * sizeof(float);
}

// The ceiling probe's test-only form (/tmp/d-cap.md §3 P-B): as `scratchy_cuda_gemv_fp8_tc` at
// instance `variant` (0 B1, the launcher's; 1 B2 `wait1`; 2 B3 `k128`, which also needs `k` a
// multiple of 128; 3 B4 `2cta`; 4 B6 `k256`, `k` a multiple of 256; 5 B8 `swapab32` and 6 B9
// `swapab32sk`, `k` a multiple of 256), and with `main_only` (B5
// `noreduce`) the stream-K kernel alone,
// leaving `c` unwritten. No tape launches it.
extern "C" int scratchy_cuda_gemv_fp8_tc_variant(const void* q, const void* a_scale,
                                                 const void* const* b, const void* const* b_scale,
                                                 void* const* c, const uint32_t* n,
                                                 uint32_t count, uint32_t k, void* partials,
                                                 uint32_t variant, uint32_t main_only,
                                                 void* stream /* cudaStream_t */) {
    if (scratchy_cuda_gemv_fp8_tc_partial_bytes(n, count, k) == 0 || variant > 6) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const bool m = main_only != 0;
        switch (variant) {
            case 0: return launch<tc_b1>(q, a_scale, b, b_scale, c, n, count, k, partials, m, stream);
            case 1: return launch<tc_b2>(q, a_scale, b, b_scale, c, n, count, k, partials, m, stream);
            case 2: return launch<tc_b3>(q, a_scale, b, b_scale, c, n, count, k, partials, m, stream);
            case 3: return launch<tc_b4>(q, a_scale, b, b_scale, c, n, count, k, partials, m, stream);
            case 4: return launch<tc_b6>(q, a_scale, b, b_scale, c, n, count, k, partials, m, stream);
            case 5: return launch<tc_b8>(q, a_scale, b, b_scale, c, n, count, k, partials, m, stream);
            default: return launch<tc_b9>(q, a_scale, b, b_scale, c, n, count, k, partials, m, stream);
        }
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught; any other value = the cudaError_t of the
// first launch that failed.
//
// `count` (1 to 3) members off `q [1, k]` e4m3 and its `a_scale` (one f32): member `i` is
// `c[i] [n[i]] = bf16(((q · b[i]ᵀ) · b_scale[i]) · a_scale)`, `n[i]` a non-zero multiple of 64 and `k`
// of 64, checked before any pointer is touched. `partials` holds
// `scratchy_cuda_gemv_fp8_tc_partial_bytes` bytes. Every pointer is 16-byte aligned device memory,
// no output overlapping an input. `scratchy_cuda_gemv_fp8_tc_prepare` must have succeeded on the
// current device. Two launches (the stream-K kernel, then its reduce), asynchronous on `stream`.
extern "C" int scratchy_cuda_gemv_fp8_tc(const void* q, const void* a_scale,
                                         const void* const* b, const void* const* b_scale,
                                         void* const* c, const uint32_t* n, uint32_t count,
                                         uint32_t k, void* partials,
                                         void* stream /* cudaStream_t; null = default */) {
    return scratchy_cuda_gemv_fp8_tc_variant(q, a_scale, b, b_scale, c, n, count, k, partials, 0, 0,
                                             stream);
}

// B8, production form (FP8 S8 T1; no tape launches it yet). 0 = launched; SCRATCHY_CUDA_ERR_SHAPE
// = refused, nothing launched; SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught; any other
// value = the launch's cudaError_t. One launch, asynchronous on `stream`.
//
// `count` (1 to 3) members off the e4m3 activation `q [a_rows, k]` and its per-row `a_scale
// [a_rows]` f32 (`quant_rows_fp8`'s bytes): pair `p` (`0 .. pairs`) of member `i` is
//
//     c[i][p, j] = (((q[p / row_div] · b_e[j, :]) · b_scale_e[j]) · a_scale[p / row_div])
//
// over the member's `n[i]` rows, rounded once to bf16 (`out_f32` 0), with `residual[i][p, j]`
// added as `scratchy::ops::with_residual` does when `residual` is not null; or stored as f32
// (`out_f32` 1, no residual). Dense (`topk_idx` null): `b[i]` is `[n[i], k]` and `b_scale[i]
// [n[i]]`, `experts` 1, `expert_rows` unread, `e` 0. Expert: `b[i]` is the bank's `[experts,
// expert_rows, k]` and `b_scale[i] [experts, expert_rows]`, `e = topk_idx[p]`, each member's first
// `n[i]` rows of its expert (`gemv_fp8.cu`'s `EXPERT` contract).
//
// Refused before any pointer is touched: `count` not 1..3; `k` or an `n[i]` zero or not a multiple
// of 64; `pairs` 0 or above 65,535; `row_div` 0; a pair's row at or past `a_rows`; `experts` 0,
// or not 1 for the dense form; `expert_rows` below an `n[i]` (expert); a residual with `out_f32`;
// a size above INT_MAX; a null pointer. Every pointer 16-byte-aligned device memory, no output
// overlapping an input. `scratchy_cuda_gemv_fp8_tc_prepare` must have succeeded.
extern "C" int scratchy_cuda_gemv_fp8_b8(
    const void* q, const void* a_scale, uint32_t a_rows, const void* const* b,
    const void* const* b_scale, void* const* c, const void* const* residual, const uint32_t* n,
    uint32_t count, uint32_t k, uint32_t experts, uint32_t expert_rows, const void* topk_idx,
    uint32_t pairs, uint32_t row_div, uint32_t out_f32,
    void* stream /* cudaStream_t; null = default */) {
    const b8_args a{q,     a_scale, a_rows, b,           b_scale,  c,     residual, n,
                    count, k,       experts, expert_rows, topk_idx, pairs, row_div};
    if (b8_refused(a, out_f32 != 0)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const bool expert = topk_idx != nullptr;
        if (out_f32 != 0) {
            return expert ? launch_b8<true, true>(a, stream) : launch_b8<false, true>(a, stream);
        }
        return expert ? launch_b8<true, false>(a, stream) : launch_b8<false, false>(a, stream);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// Test support (`tests/gemv_fp8_b8.rs`): `gemv_fp8_b8_box_probe_ker` on the activation `x [rows, 1,
// k]` at row `row` and the weight `w [n, k]` at tile `tile`, K step `j`; `poison_x [32, 256]`,
// `poison_w [64, 256]` all-NaN e4m3, `dump_x`, `dump_w` the same shapes. Refused: a zero or
// misaligned size (`k` a multiple of 16), a coordinate past its array, a null pointer.
extern "C" int scratchy_cuda_gemv_fp8_b8_box_probe(const void* x, uint32_t rows, const void* w,
                                                   uint32_t n, uint32_t k, uint32_t row,
                                                   uint32_t tile, uint32_t j, const void* poison_x,
                                                   const void* poison_w, void* dump_x,
                                                   void* dump_w,
                                                   void* stream /* cudaStream_t; null = default */) {
    if (rows == 0 || n == 0 || k == 0 || k % 16 != 0 || row >= rows || tile * 64 >= n ||
        uint64_t(j) * 256 >= k || x == nullptr || w == nullptr || poison_x == nullptr ||
        poison_w == nullptr || dump_x == nullptr || dump_w == nullptr ||
        uint64_t(rows) * k > uint64_t(INT_MAX) || uint64_t(n) * k > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const auto e4 = [](const void* p) { return static_cast<fp8e4m3*>(const_cast<void*>(p)); };
        b8_probe_globals G{
            b8_x_gl{e4(x), nullptr, size_t(rows), nullptr, size_t(k)},
            b8_w_gl{e4(w), nullptr, size_t(1), size_t(n), size_t(k)},
            b8_dump_x_gl{e4(poison_x), nullptr, nullptr, size_t(32), size_t(256)},
            b8_dump_x_gl{e4(dump_x), nullptr, nullptr, size_t(32), size_t(256)},
            b8_dump_w_gl{e4(poison_w), nullptr, nullptr, size_t(64), size_t(256)},
            b8_dump_w_gl{e4(dump_w), nullptr, nullptr, size_t(64), size_t(256)},
            int(row), int(tile), int(j)};
        void* args[] = {&G};
        const dim3 grid{1u}, block{uint32_t(kittens::WARP_THREADS)};
        return static_cast<int>(cudaLaunchKernel(gemv_fp8_b8_box_probe_ker, grid, block, args,
                                                 B8_PROBE_SMEM, static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
