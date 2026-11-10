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
// Shared memory: `STAGES` × (a 4 KB A tile and an 8 KB B tile) plus 1,024 bytes of alignment slack,
// set by `scratchy_cuda_gemv_fp8_tc_prepare`.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// The launcher's return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors them.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// One CTA per SM of an H100 SXM; the bits depend on it, so it is a constant.
constexpr int CTAS = 132;
constexpr int TILE_N = 128;
constexpr int K_STEP = 64;
constexpr int STAGES = 12;
constexpr int NUM_THREADS = 2 * kittens::WARPGROUP_THREADS;
constexpr int REDUCE_ROWS = 32;
constexpr int REDUCE_WARPS = 8;
constexpr int REDUCE_THREADS = REDUCE_WARPS * kittens::WARP_THREADS;
// `n` a multiple of it: the reduce's 32-row vectors and the scales' reads stay inside the arrays.
constexpr uint32_t N_MULTIPLE = 64;

using a_tile = st_fp8e4m3<64, K_STEP>;
using b_tile = st_fp8e4m3<TILE_N, K_STEP>;
using acc_tile = rt_fl<16, TILE_N>;
using piece_rv = acc_tile::row_vec;
using reduce_rv = rv_fl<REDUCE_ROWS>;
using a_gl = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
using b_gl = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
// `[CTAS + tiles, 128]` f32 pieces, `[1, n]` scales and outputs: no tensor map.
using partial_gl = gl<float, 1, 1, -1, -1>;
using scale_gl = gl<float, 1, 1, -1, -1>;
using out_gl = gl<bf16, 1, 1, -1, -1>;

constexpr size_t TC_SMEM_BYTES = STAGES * (sizeof(a_tile) + sizeof(b_tile)) + 1024;
static_assert(TC_SMEM_BYTES <= 227 * 1024, "one CTA per SM");

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

__global__ __launch_bounds__(NUM_THREADS, 1) void gemv_fp8_tc_ker(
    const __grid_constant__ tc_globals g) {
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
        acc_tile acc;
        warp::zero(acc);
        for (int i = 0; i < len; i++) {
            const long long u = u0 + i;
            const int t = int(u / g.k_steps), j = int(u % g.k_steps), s = i % STAGES;
            wait(arrived[s], (i / STAGES) % 2);
            warpgroup::mma_ABt(acc, a_s[s], b_s[s]);
            // The WGMMA reads the stage and writes `acc` asynchronously: wait before releasing the
            // stage and before anything reads `acc`.
            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) {
                arrive(done[s], 1);
            }
            if (j == g.k_steps - 1 || i == len - 1) {
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

// Rows `32v .. 32v + 32` of the concatenated members: their tile's pieces added in ascending K
// order, then `· b_scale`, `· a_scale`, rounded once.
__global__ __launch_bounds__(REDUCE_THREADS) void gemv_fp8_tc_reduce_ker(
    const __grid_constant__ tc_globals g) {
    const int v = blockIdx.x * REDUCE_WARPS + kittens::warpid();
    const int t = v * REDUCE_ROWS / TILE_N;
    if (t >= g.tiles) {
        return;
    }
    const int m = t >= g.start2 ? 2 : t >= g.start1 ? 1 : 0;
    const int start = m == 2 ? g.start2 : m == 1 ? g.start1 : 0;
    const int n = m == 2 ? g.n2 : m == 1 ? g.n1 : g.n0;
    // The vector's first row inside its member, and inside its tile.
    const int row = (t - start) * TILE_N + (v * REDUCE_ROWS) % TILE_N;
    if (row >= n) {
        return;
    }
    const long long units = (long long)g.tiles * g.k_steps;
    const int first = cta_of((long long)t * g.k_steps, units, g.ctas);
    const int last = cta_of((long long)(t + 1) * g.k_steps - 1, units, g.ctas);
    const int col = (v * REDUCE_ROWS) % TILE_N / REDUCE_ROWS;
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

int tiles_of(uint32_t n) {
    return int((n + uint32_t(TILE_N) - 1) / uint32_t(TILE_N));
}

}  // namespace

// The main kernel's one-time setup: allow it `TC_SMEM_BYTES` of dynamic shared memory on the current
// device. 0 = done; any other value = the cudaError_t.
extern "C" int scratchy_cuda_gemv_fp8_tc_prepare(void) {
    return static_cast<int>(cudaFuncSetAttribute(
        gemv_fp8_tc_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, TC_SMEM_BYTES));
}

// The `partials` bytes a launch of these members needs: `(min(CTAS, units) + tiles) · 128` f32.
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
    const uint64_t ctas = units < uint64_t(CTAS) ? units : uint64_t(CTAS);
    return (ctas + tiles) * TILE_N * sizeof(float);
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
    if (scratchy_cuda_gemv_fp8_tc_partial_bytes(n, count, k) == 0) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
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
        const int t0 = tiles_of(n[0]);
        const int t1 = count > 1 ? tiles_of(n[1]) : 0;
        const int t2 = count > 2 ? tiles_of(n[2]) : 0;
        const int tiles = t0 + t1 + t2;
        const int k_steps = int(k / K_STEP);
        const long long units = (long long)tiles * k_steps;
        const int ctas = units < CTAS ? int(units) : CTAS;
        tc_globals G{
            a_gl{static_cast<fp8e4m3*>(const_cast<void*>(q)), nullptr, nullptr, size_t(1),
                 size_t(k)},
            bv(0), bv(count > 1 ? 1 : last), bv(last),
            partial_gl{static_cast<float*>(partials), nullptr, nullptr, size_t(ctas + tiles),
                       size_t(TILE_N)},
            sv(0), sv(count > 1 ? 1 : last), sv(last),
            cv(0), cv(count > 1 ? 1 : last), cv(last),
            scale_gl{static_cast<float*>(const_cast<void*>(a_scale)), nullptr, nullptr, size_t(1),
                     size_t(1)},
            int(n[0]), int(n[count > 1 ? 1 : last]), int(n[last]),
            // An absent member starts at `tiles`: no tile is its.
            count > 1 ? t0 : tiles, count > 2 ? t0 + t1 : tiles, tiles, k_steps, ctas};
        void* args[] = {&G};
        const cudaStream_t st = static_cast<cudaStream_t>(stream);
        cudaError_t e = cudaLaunchKernel(gemv_fp8_tc_ker, dim3(ctas), dim3(NUM_THREADS), args,
                                         TC_SMEM_BYTES, st);
        if (e != cudaSuccess) {
            return static_cast<int>(e);
        }
        const int vectors = tiles * (TILE_N / REDUCE_ROWS);
        return static_cast<int>(cudaLaunchKernel(gemv_fp8_tc_reduce_ker,
                                                 dim3((vectors + REDUCE_WARPS - 1) / REDUCE_WARPS),
                                                 dim3(REDUCE_THREADS), args, 0, st));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
