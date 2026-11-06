// SPDX-License-Identifier: Apache-2.0
//
// Narrow per-channel fp8 GEMM for the H100 (FP8 S3): C[m, n] = bf16((A[m, k] · B[n, k]ᵀ) ·
// w_scale[n] · a_scale[m]), A and B row-major e4m3, accumulated in f32 and rounded once, for `n`
// of at most 64 output columns: axteam's Qwen3.8-27B quantizes the gated delta net's `in_proj_a`
// and `in_proj_b` (48 columns each), which the tiled fp8 GEMM (`gemm_fp8.cu`, 64-column tiles at
// least) refuses. The lowering runs the two as one grouped launch off one quantized input.
//
// `gemm_narrow_bf16.cu`'s structure (a producer warp, a `STAGES`-deep TMA ring, two consumer
// warpgroups of 64 rows each, `warpgroup::mma_ABt` against the `[N, K]` B, one CTA per 128 output
// rows), on `gemm_fp8.cu`'s operands: swizzled `st_fp8e4m3<64, 64>` tiles (64-byte rows), two
// `k32` WGMMAs per stage, the scale epilogue `mul_col(acc, w_scale)` then `mul_row(acc, a_scale)`.
// What differs from the bf16 kernel:
//   - no K tail: `k` is a multiple of 64 (`step::GEMM_FP8_K_MULTIPLE`), so no warp-MMA tail;
//   - the scales are TMA-loaded once per CTA, before the K loop, into `sv_fl<64>` vectors: B's
//     `w_scale[0 .. 64)` and each live consumer's 64 entries of `a_scale`. A vector box is never
//     swizzled and is bounded by its array's length, so the entries past `n` and past `m` read
//     zeros (the R1 rule, `tests/tma_bounds.rs`); they scale only columns and rows the store drops.
// The rest is the bf16 kernel's: B's rows `n .. 63` and A's rows past `m` read as zeros on the
// swizzled tiles; a consumer warpgroup whose 64 rows are all past `m` loads, multiplies and stores
// nothing; C is a plain `st_bf<64, 64, false>` per consumer, whose TMA store drops the columns
// `n .. 63` and the rows past `m` (its row stride `2n` bytes, `n` a multiple of 8, is a multiple of
// 16). A fixed K order, no atomics: an output row depends only on its own row of A.
//
// Shared memory: `STAGES` × (two A tiles and one B tile, 4 KiB each), the two plain C tiles (8 KiB
// each), three 256-byte scale vectors, plus 1,024 bytes of alignment slack, set once by
// `scratchy_cuda_gemm_narrow_fp8_prepare` (in `PREPARE`). The semaphores are static shared memory.

#include "kittens.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::GEMM_NARROW_BF16_MULTIPLE`: `n` is a multiple of it, so C's row stride is a multiple of 16
// bytes, as its tensor map needs.
constexpr uint32_t VEC = 8;
// `step::GEMM_NARROW_BF16_MAX_N`: B's rows, all in one 64-row tile.
constexpr uint32_t MAX_N = 64;
// `quant_rows_fp8.cu`'s 16-row tile: `m` is a multiple of it.
constexpr uint32_t ROW_MULTIPLE = 16;
// `step::GEMM_FP8_K_MULTIPLE`: one K step.
constexpr uint32_t K_STEP = 64;
// Rows and columns of a tile, and K per tile.
constexpr int TILE = 64;
constexpr int STAGES = 4;
constexpr int CONSUMER_WARPGROUPS = 2;
constexpr int PRODUCER_WARPGROUP = CONSUMER_WARPGROUPS;
constexpr int NUM_THREADS = (CONSUMER_WARPGROUPS + 1) * kittens::WARPGROUP_THREADS;
// Output rows per CTA.
constexpr int CTA_ROWS = CONSUMER_WARPGROUPS * TILE;
static_assert(MAX_N == uint32_t(TILE), "every column of C is in one tile");
static_assert(K_STEP == uint32_t(TILE), "one K step per tile");

using tile = st_fp8e4m3<TILE, TILE>;
using out_tile = st_bf<TILE, TILE, false>;
using scale_vec = sv_fl<TILE>;
using operand_gl = gl<fp8e4m3, 1, 1, -1, -1, tile>;
using output_gl = gl<bf16, 1, 1, -1, -1, out_tile>;
// `[1, len]` f32 scales, TMA-loaded as 64-entry vectors.
using scale_gl = gl<float, 1, 1, 1, -1, scale_vec>;
using acc_tile = rt_fl<16, TILE>;

struct narrow_globals {
    operand_gl a;
    operand_gl b;
    output_gl c;
    scale_gl a_scale;
    scale_gl b_scale;
    // The rows of A and C; the K steps, `k / 64`.
    int m;
    int k_tiles;
};

constexpr size_t SMEM_USED = STAGES * (CONSUMER_WARPGROUPS + 1) * sizeof(tile) +
                             CONSUMER_WARPGROUPS * sizeof(out_tile) +
                             (CONSUMER_WARPGROUPS + 1) * sizeof(scale_vec);
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int NARROW_SMEM_BYTES = int(SMEM_USED) + 1024;
static_assert(NARROW_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

// A narrow CTA's shared tiles and vectors.
struct narrow_smem {
    tile (&a_smem)[STAGES][CONSUMER_WARPGROUPS];
    tile (&b_smem)[STAGES];
    out_tile (&c_smem)[CONSUMER_WARPGROUPS];
    scale_vec (&a_scale)[CONSUMER_WARPGROUPS];
    scale_vec& b_scale;
};

// One CTA of one narrow fp8 GEMM over `g`'s views and `sm`'s tiles: output rows `128 · row_cta ..`.
// The grouped kernel runs the same CTA for each member, so each member is its own GEMM's bits.
__device__ inline void narrow_cta(const narrow_globals& g, const narrow_smem& sm, int row_cta) {
    auto& a_smem = sm.a_smem;
    auto& b_smem = sm.b_smem;
    auto& c_smem = sm.c_smem;

    // The CTA's first 64-row tile of A and C, and how many of its consumer warpgroups have a real
    // row (`gemm_narrow_bf16.cu`'s): a dead one loads, multiplies and stores nothing.
    const int row_tile0 = row_cta * CONSUMER_WARPGROUPS;
    const int live = min(CONSUMER_WARPGROUPS, (g.m - row_tile0 * TILE + TILE - 1) / TILE);
    const int warpgroupid = threadIdx.x / kittens::WARPGROUP_THREADS;

    __shared__ kittens::semaphore arrived[STAGES], done[STAGES], scales_arrived;
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_semaphore(arrived[s], 0, 1);
            init_semaphore(done[s], live, 0);
        }
        init_semaphore(scales_arrived, 0, 1);
    }
    __syncthreads();

    if (warpgroupid == PRODUCER_WARPGROUP) {
        // The producer: one warp loads the scales first, then K tile `j` into stage `j % STAGES`,
        // after every live consumer released that stage's previous tile.
        if (warpgroup::warpid() == 0) {
            warp::tma::expect_bytes(scales_arrived, (live + 1) * sizeof(scale_vec));
            for (int w = 0; w < live; w++) {
                coord<scale_vec> at_a = {0, 0, 0, row_tile0 + w};
                warp::tma::load_async(sm.a_scale[w], g.a_scale, at_a, scales_arrived);
            }
            coord<scale_vec> at_b = {0, 0, 0, 0};
            warp::tma::load_async(sm.b_scale, g.b_scale, at_b, scales_arrived);
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
        acc_tile acc;
        warp::zero(acc);
        for (int j = 0; j < g.k_tiles; j++) {
            const int stage = j % STAGES;
            wait(arrived[stage], (j / STAGES) % 2);
            warpgroup::mma_ABt(acc, a_smem[stage][warpgroupid], b_smem[stage]);
            // The WGMMA reads shared memory and writes `acc` asynchronously: wait for it before
            // releasing the stage, and before anything else reads `acc`.
            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) {
                arrive(done[stage], 1);
            }
        }
        wait(scales_arrived, 0);
        acc_tile::row_vec ws;
        acc_tile::col_vec as;
        warp::load(ws, sm.b_scale);
        warpgroup::load(as, sm.a_scale[warpgroupid]);
        warp::mul_col(acc, acc, ws);
        warp::mul_row(acc, acc, as);
        warpgroup::store(c_smem[warpgroupid], acc);
        // Each warpgroup's own named barrier (0 is `__syncthreads`).
        warpgroup::sync(warpgroupid + 1);
        if (warpgroup::warpid() == 0) {
            coord<out_tile> at = {0, 0, row_tile0 + warpgroupid, 0};
            warp::tma::store_async(g.c, c_smem[warpgroupid], at);
        }
        warp::tma::store_async_wait();
    }
}

__global__ __launch_bounds__(NUM_THREADS, 1) void gemm_narrow_fp8_ker(
    const __grid_constant__ narrow_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    st_fp8e4m3<TILE, TILE>(&a_smem)[STAGES][CONSUMER_WARPGROUPS] =
        al.allocate<tile, STAGES, CONSUMER_WARPGROUPS>();
    st_fp8e4m3<TILE, TILE>(&b_smem)[STAGES] = al.allocate<tile, STAGES>();
    st_bf<TILE, TILE, false>(&c_smem)[CONSUMER_WARPGROUPS] =
        al.allocate<out_tile, CONSUMER_WARPGROUPS>();
    sv_fl<TILE>(&a_scale)[CONSUMER_WARPGROUPS] = al.allocate<scale_vec, CONSUMER_WARPGROUPS>();
    sv_fl<TILE>& b_scale = al.allocate<scale_vec>();
    const narrow_smem sm{a_smem, b_smem, c_smem, a_scale, b_scale};
    narrow_cta(g, sm, blockIdx.x);
}

// The grouped narrow GEMM: up to three narrow GEMMs of one `A` (and its `a_scale`), one launch.
// Member `i` is `m[i]`; the grid's y is the member, its x the row CTA. An absent third member
// repeats the second's views; the grid has no y for it.
struct narrow_group_globals {
    narrow_globals m0;
    narrow_globals m1;
    narrow_globals m2;
};

__global__ __launch_bounds__(NUM_THREADS, 1) void gemm_narrow_group_fp8_ker(
    const __grid_constant__ narrow_group_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    st_fp8e4m3<TILE, TILE>(&a_smem)[STAGES][CONSUMER_WARPGROUPS] =
        al.allocate<tile, STAGES, CONSUMER_WARPGROUPS>();
    st_fp8e4m3<TILE, TILE>(&b_smem)[STAGES] = al.allocate<tile, STAGES>();
    st_bf<TILE, TILE, false>(&c_smem)[CONSUMER_WARPGROUPS] =
        al.allocate<out_tile, CONSUMER_WARPGROUPS>();
    sv_fl<TILE>(&a_scale)[CONSUMER_WARPGROUPS] = al.allocate<scale_vec, CONSUMER_WARPGROUPS>();
    sv_fl<TILE>& b_scale = al.allocate<scale_vec>();
    const narrow_smem sm{a_smem, b_smem, c_smem, a_scale, b_scale};
    const narrow_globals& m = blockIdx.y == 2 ? g.m2 : blockIdx.y == 1 ? g.m1 : g.m0;
    narrow_cta(m, sm, blockIdx.x);
}

// The narrow rule for one member: `m` a non-zero multiple of 16 (A is `QuantRowsFp8`'s whole
// 16-row tiles, and `a_scale`'s tensor map needs a 16-byte multiple), `n` a non-zero multiple of 8
// at most 64, `k` a non-zero multiple of 64, `m` and `k` at most INT_MAX.
bool narrow_shape(uint32_t m, uint32_t n, uint32_t k) {
    return m != 0 && m % ROW_MULTIPLE == 0 && m <= uint32_t(INT_MAX) && n != 0 && n % VEC == 0 && n <= MAX_N && k != 0 &&
           k % K_STEP == 0 && k <= uint32_t(INT_MAX);
}

// One member's views and counts. Building each tile and vector `gl` encodes its TMA tensor map into
// host memory; nothing is allocated on the device. Can throw (TK's host code).
narrow_globals narrow_member(const void* a, const void* a_scale, const void* b,
                             const void* b_scale, void* c, uint32_t m, uint32_t n, uint32_t k) {
    operand_gl ag{static_cast<fp8e4m3*>(const_cast<void*>(a)), nullptr, nullptr, size_t(m),
                  size_t(k)};
    operand_gl bg{static_cast<fp8e4m3*>(const_cast<void*>(b)), nullptr, nullptr, size_t(n),
                  size_t(k)};
    output_gl cg{static_cast<bf16*>(c), nullptr, nullptr, size_t(m), size_t(n)};
    scale_gl asg{static_cast<float*>(const_cast<void*>(a_scale)), nullptr, nullptr, nullptr,
                 size_t(m)};
    scale_gl bsg{static_cast<float*>(const_cast<void*>(b_scale)), nullptr, nullptr, nullptr,
                 size_t(n)};
    return narrow_globals{ag, bg, cg, asg, bsg, int(m), int(k / K_STEP)};
}

}  // namespace

// The kernels' one-time setup: allow the plain and the grouped kernel `NARROW_SMEM_BYTES` of
// dynamic shared memory on the current device (the default limit is 48 KB). Run once per device
// before any launch (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch.
//
// 0 = done; any other value = the cudaError_t of the first call that failed.
extern "C" int scratchy_cuda_gemm_narrow_fp8_prepare(void) {
    cudaError_t e = cudaFuncSetAttribute(
        gemm_narrow_fp8_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, NARROW_SMEM_BYTES);
    if (e == cudaSuccess) {
        e = cudaFuncSetAttribute(gemm_narrow_group_fp8_ker,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, NARROW_SMEM_BYTES);
    }
    return static_cast<int>(e);
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value
// = the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `c [m, n] = bf16((a · bᵀ) · b_scale[n] · a_scale[m])`: `m` a non-zero multiple of 16, `n` a non-zero multiple of
// 8 at most 64, `k` a non-zero multiple of 64, `m` and `k` at most INT_MAX, all checked before any
// pointer is touched. `a [m, k]` and `b [n, k]` e4m3, `a_scale [m]` and `b_scale [n]` f32 and
// `c [m, n]` bf16 must be 16-byte-aligned device pointers, `c` overlapping no input.
// `scratchy_cuda_gemm_narrow_fp8_prepare` must have succeeded on the current device. The launch
// is asynchronous on `stream`.
extern "C" int scratchy_cuda_gemm_narrow_fp8_abt(const void* a, const void* a_scale, const void* b,
                                                 const void* b_scale, void* c, uint32_t m,
                                                 uint32_t n, uint32_t k,
                                                 void* stream /* cudaStream_t; null = default */) {
    if (!narrow_shape(m, n, k)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        narrow_globals G = narrow_member(a, a_scale, b, b_scale, c, m, n, k);
        void* args[] = {&G};
        const dim3 grid((m + uint32_t(CTA_ROWS) - 1) / uint32_t(CTA_ROWS)), block(NUM_THREADS);
        return static_cast<int>(cudaLaunchKernel(gemm_narrow_fp8_ker, grid, block, args,
                                                 NARROW_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// As `scratchy_cuda_gemm_narrow_fp8_abt` for `count` (2 or 3) members of one `a [m, k]` and its
// `a_scale [m]`: member `i` is `c[i] [m, n[i]] = bf16((a · b[i]ᵀ) · b_scale[i][n] · a_scale[m])`,
// each `n[i]` under the narrow rule, checked before any pointer is touched. Each member's output
// is the bits of its own `scratchy_cuda_gemm_narrow_fp8_abt`.
// `scratchy_cuda_gemm_narrow_fp8_prepare` must have succeeded on the current device.
extern "C" int scratchy_cuda_gemm_narrow_group_fp8_abt(const void* a, const void* a_scale,
                                                       const void* const* b,
                                                       const void* const* b_scale,
                                                       void* const* c, const uint32_t* n,
                                                       uint32_t m, uint32_t k, uint32_t count,
                                                       void* stream /* cudaStream_t */) {
    if (count < 2 || count > 3) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    for (uint32_t i = 0; i < count; i++) {
        if (!narrow_shape(m, n[i], k)) {
            return SCRATCHY_CUDA_ERR_SHAPE;
        }
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const uint32_t last = count - 1;
        const auto member = [&](uint32_t i) {
            return narrow_member(a, a_scale, b[i], b_scale[i], c[i], m, n[i], k);
        };
        narrow_group_globals G{member(0), member(1), member(last)};
        void* args[] = {&G};
        const dim3 grid((m + uint32_t(CTA_ROWS) - 1) / uint32_t(CTA_ROWS), count),
            block(NUM_THREADS);
        return static_cast<int>(cudaLaunchKernel(gemm_narrow_group_fp8_ker, grid, block, args,
                                                 NARROW_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
