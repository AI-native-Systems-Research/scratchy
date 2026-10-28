// SPDX-License-Identifier: Apache-2.0
//
// bf16 GEMV for the H100: c[n] = Σ_k a[k] · B[n, k], one activation row `a [1, k]` against a
// row-major `B [n, k]` weight, all bf16, accumulated in f32 and rounded once to bf16
// (round-to-nearest-even). It is the `m = 1` case of `gemm_bf16.cu`'s `C = A · Bᵀ`: a decode
// step's every projection and its lm_head.
//
// ThunderKittens' GEMM structure at 64×64 (PLAN-tk100 §3.1), from `gemm_bf16.cu` (itself adapted
// from TK's `kernels/gemm/bf16_h100` at 67845f5f, MIT, see `thunderkittens/LICENSE`): a producer
// warp loads tiles with TMA, `tma::expect_bytes` before every load, into a ring of `STAGES`
// stages; one consumer warpgroup multiplies them with WGMMA (`warpgroup::mma_ABt` against the
// `[N, K]` B), and waits for each WGMMA (`mma_async_wait`) before handing its stage back. One CTA
// per 64 output columns: 64 rows of `B` against `a`, whose 63 rows past the array's edge TMA reads
// as zeros. So 63 of every 64 tensor-core rows multiply zeros; at one real row the op does two
// flops per weight byte, far below the H100's ridge point, and its time is the time to read `B`
// (`n · k` bf16), as the plain-CUDA GEMV's was.
//
// A hand-built producer/consumer kernel, structured as `attn_prefill_bf16.cu` (explicit
// semaphores, one producer warp), not TK's `lcf` template: the K tail and the unaligned `c` store
// below have no place in `lcf`'s `compute` and `finish`.
//
// Which shared tile is swizzled follows from what TMA does past an array's edge (the GKE probe's
// R1 at c8d80fda, re-run by `tests/tma_bounds.rs`): a swizzled tile reads zeros for rows past the
// edge, but for columns past it reads real memory, since its tensor map rounds a row up to the
// 64-element swizzle chunk (`types/global/tma.cuh`); a plain (non-swizzled) tile reads zeros past
// both. WGMMA reads swizzled tiles only (`types/shared/descriptor.cuh`). So:
//   - the full K tiles (`k0 + 64 ≤ k`) of `a` and `B` are swizzled `st_bf<64, 64>`, multiplied by
//     WGMMA;
//   - the K tail, when `k` is not a multiple of 64, is one plain `st_bf<64, 64, false>` of each,
//     zero past `k`, multiplied by warp MMA from registers into the same `rt_fl<16, 64>`
//     accumulator after the last WGMMA's wait: `warpgroup::load` gives each warp its 16 rows of
//     `a`, `warp::load` each warp all of `B`'s tail, and `warp::mma_ABt` accumulates. Each warp
//     works on its own registers. Whole plain tiles are loaded, never a narrower subtile view
//     (`st_subtile::idx` indexes a plain subtile by its own width);
//   - `c` has no tensor map: a tensor map needs every row stride a multiple of 16 bytes, and `n`
//     may be any non-zero value (one column is a test shape). The accumulator goes to a plain
//     shared tile (`warpgroup::store`, the one RNE rounding to bf16), then row 0's
//     `min(64, n − c0)` columns are copied through TK's accessors. Nothing else of `c` is written.
// No real model's `k` has a tail (1024, 2048, 3584, 4096, 12288); the test shapes `k` = 8 and 72
// do. The zeros past `k` and in `a`'s rows 1..63 add exactly 0.
//
// Every product is accumulated inside WGMMA or warp MMA in a fixed K order, with no atomics, so
// the same input gives the same bits. Shared memory: `STAGES` × (`a` + `B`) swizzled tiles, the
// two plain tail tiles and the plain `c` tile, 8 KiB each, plus 1,024 bytes of alignment slack =
// 91,136 bytes of dynamic shared memory, set once by `scratchy_cuda_gemv_bf16_abt_prepare` (in
// `PREPARE`). The semaphores are static shared memory.

#include "kittens.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::GEMV_BF16_MULTIPLE`: `k` is a multiple of it, so every row stride of `a` and `B` is a
// multiple of 16 bytes, as their tensor maps need.
constexpr uint32_t VEC = 8;
// Output columns per CTA (rows of `B`), and K per tile.
constexpr int TILE = 64;
constexpr int STAGES = 4;
// One consumer warpgroup, then one producer warpgroup whose first warp issues every load.
constexpr int CONSUMER_WARPGROUP = 0;
constexpr int PRODUCER_WARPGROUP = 1;
constexpr int NUM_THREADS = 2 * kittens::WARPGROUP_THREADS;
// The consumer warpgroup's named barrier (0 is `__syncthreads`).
constexpr int CONSUMER_BARRIER = 1;

using tile = st_bf<TILE, TILE>;
using plain_tile = st_bf<TILE, TILE, false>;
// `a [1, k]` and `B [n, k]`, each with both tensor maps: swizzled for the full K tiles, plain for
// the tail.
using operand_gl = gl<bf16, 1, 1, -1, -1, tile, plain_tile>;
// `c [1, n]`: no tensor map, written element by element.
using output_gl = gl<bf16, 1, 1, -1, -1>;

struct gemv_globals {
    operand_gl a;
    operand_gl b;
    output_gl c;
    // The columns of `c`; the full K tiles, `k / 64`; whether a K tail follows them.
    int n;
    int k_tiles;
    bool tail;
};

constexpr size_t SMEM_TILES = STAGES * 2 * sizeof(tile) + 3 * sizeof(plain_tile);
static_assert(SMEM_TILES == 90112, "4 stages of a and B, two tail tiles and c, 8 KiB each");
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int GEMV_SMEM_BYTES = int(SMEM_TILES) + 1024;
static_assert(GEMV_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

__global__ __launch_bounds__(NUM_THREADS, 1) void gemv_bf16_ker(
    const __grid_constant__ gemv_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    tile(&a_smem)[STAGES] = al.allocate<tile, STAGES>();
    tile(&b_smem)[STAGES] = al.allocate<tile, STAGES>();
    plain_tile& a_tail = al.allocate<plain_tile>();
    plain_tile& b_tail = al.allocate<plain_tile>();
    plain_tile& c_smem = al.allocate<plain_tile>();

    // This CTA's 64 output columns: rows `64 · col_tile ..` of `B`.
    const int col_tile = blockIdx.x;
    const int warpgroupid = threadIdx.x / kittens::WARPGROUP_THREADS;

    __shared__ kittens::semaphore arrived[STAGES], done[STAGES], tail_arrived;
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_semaphore(arrived[s], 0, 1);
            init_semaphore(done[s], 1, 0);
        }
        init_semaphore(tail_arrived, 0, 1);
    }
    __syncthreads();

    if (warpgroupid == PRODUCER_WARPGROUP) {
        // The producer: one warp loads the tail first (its tiles are its own), then K tile `j`
        // into stage `j % STAGES`, after the consumer released that stage's previous tile.
        if (warpgroup::warpid() == 0) {
            if (g.tail) {
                coord<plain_tile> at_a = {0, 0, 0, g.k_tiles};
                coord<plain_tile> at_b = {0, 0, col_tile, g.k_tiles};
                warp::tma::expect_bytes(tail_arrived, 2 * sizeof(plain_tile));
                warp::tma::load_async(a_tail, g.a, at_a, tail_arrived);
                warp::tma::load_async(b_tail, g.b, at_b, tail_arrived);
            }
            for (int j = 0; j < g.k_tiles; j++) {
                const int stage = j % STAGES;
                if (j >= STAGES) {
                    wait(done[stage], ((j - STAGES) / STAGES) % 2);
                }
                coord<tile> at_a = {0, 0, 0, j};
                coord<tile> at_b = {0, 0, col_tile, j};
                warp::tma::expect_bytes(arrived[stage], 2 * sizeof(tile));
                warp::tma::load_async(a_smem[stage], g.a, at_a, arrived[stage]);
                warp::tma::load_async(b_smem[stage], g.b, at_b, arrived[stage]);
            }
        }
    } else if (warpgroupid == CONSUMER_WARPGROUP) {
        // The consumer: each warp holds 16 rows of the 64-row accumulator; only row 0 is real.
        rt_fl<16, TILE> acc;
        warp::zero(acc);
        for (int j = 0; j < g.k_tiles; j++) {
            const int stage = j % STAGES;
            wait(arrived[stage], (j / STAGES) % 2);
            warpgroup::mma_ABt(acc, a_smem[stage], b_smem[stage]);
            // The WGMMA reads shared memory and writes `acc` asynchronously: wait for it before
            // handing the stage back to the producer, and before anything else reads `acc`.
            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) {
                arrive(done[stage], 1);
            }
        }
        if (g.tail) {
            wait(tail_arrived, 0);
            rt_bf<16, TILE> a_reg;
            rt_bf<TILE, TILE> b_reg;
            warpgroup::load(a_reg, a_tail);
            warp::load(b_reg, b_tail);
            warp::mma_ABt(acc, a_reg, b_reg, acc);
        }
        warpgroup::store(c_smem, acc);
        warpgroup::sync(CONSUMER_BARRIER);
        // Row 0's real columns, `c0 .. c0 + valid`, one per thread, through TK's accessors.
        const int c0 = col_tile * TILE;
        const int valid = min(TILE, g.n - c0);
        const int t = warpgroup::laneid();
        if (t < valid) {
            g.c[{0, 0, 0, c0 + t}] = c_smem[{0, t}];
        }
    }
}

}  // namespace

// The kernel's one-time setup: allow it `GEMV_SMEM_BYTES` of dynamic shared memory on the current
// device (the default limit is 48 KB). Run once per device before any launch
// (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch.
//
// 0 = done; any other value = the cudaError_t of setting the attribute.
extern "C" int scratchy_cuda_gemv_bf16_abt_prepare(void) {
    return static_cast<int>(cudaFuncSetAttribute(
        gemv_bf16_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMV_SMEM_BYTES));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value
// = the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `n` must be non-zero, `k` a non-zero multiple of 8, and both at most INT_MAX; all are checked
// before any pointer is touched. `a`, `b` and `c` must be 16-byte-aligned device pointers to `k`,
// `n·k` and `n` bf16, `c` overlapping neither input. `scratchy_cuda_gemv_bf16_abt_prepare` must
// have succeeded on the current device. The launch is asynchronous on `stream`.
extern "C" int scratchy_cuda_gemv_bf16_abt(const void* a, const void* b, void* c, uint32_t n,
                                           uint32_t k,
                                           void* stream /* cudaStream_t; null = default */) {
    if (n == 0 || n > uint32_t(INT_MAX) || k == 0 || k % VEC != 0 || k > uint32_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        bf16* ap = static_cast<bf16*>(const_cast<void*>(a));
        bf16* bp = static_cast<bf16*>(const_cast<void*>(b));
        bf16* cp = static_cast<bf16*>(c);
        // Building each operand's `gl` encodes its two TMA tensor maps into host memory; nothing
        // is allocated on the device. Every row stride is `2k` bytes, a multiple of 16.
        operand_gl ag{ap, nullptr, nullptr, size_t(1), size_t(k)};
        operand_gl bg{bp, nullptr, nullptr, size_t(n), size_t(k)};
        output_gl cg{cp, nullptr, nullptr, size_t(1), size_t(n)};
        gemv_globals G{ag, bg, cg, int(n), int(k / uint32_t(TILE)), k % uint32_t(TILE) != 0};
        void* args[] = {&G};
        const dim3 grid((n + uint32_t(TILE) - 1) / uint32_t(TILE)), block(NUM_THREADS);
        return static_cast<int>(cudaLaunchKernel(gemv_bf16_ker, grid, block, args,
                                                 GEMV_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
