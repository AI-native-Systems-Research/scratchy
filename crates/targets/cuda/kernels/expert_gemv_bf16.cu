// SPDX-License-Identifier: Apache-2.0
//
// The qwen3-5-moe block's gathered expert GEMV for the H100 (MoE step M4 of #857): for each of the
// `P = m · top_k` (token, pick) pairs, with `e = topk_idx[p]`,
//
//     c[p, :] = a[row(p), :] · B_eᵀ          (bf16, f32 accumulate, rounded once)
//
// `row(p)` is the pair's token row `p / top_k` (the gate and up, over the token rows) or its own
// row `p` (the down, over the activation rows). `B_e` is rows `0 .. n` of expert `e`'s
// `[expert_rows, k]` block of the bundle part `b` points at, as `expert_gemm_bf16.cu`'s. It runs
// at m = 1 (decode), where the pairs are not sorted: each reads its own expert's weights.
//
// Built from `kernels/gemv_bf16.cu` (ThunderKittens' GEMM structure at 64×64, PLAN-tk100 §3.1,
// §3.M M4): one producer warp loads tiles with TMA, `tma::expect_bytes` before every load, into a
// 4-stage ring of swizzled `st_bf<64, 64>`; one consumer warpgroup multiplies them with WGMMA
// (`warpgroup::mma_ABt` against the `[N, K]` B) into an `rt_fl<16, 64>` accumulator, and waits
// for each WGMMA (`mma_async_wait`) before handing its stage back. What changes:
//   - the grid is `(n / 64, P)`: one CTA per (64 output columns, pair). The producer's lane 0
//     reads `e = topk_idx[p]` from a `gl<int>` and loads B from depth `e`;
//   - the A box is the 64 rows from `row(p)`, an element-unit `coord<>`. Its other rows are other
//     real rows, or TMA's zero fill past the region; an output row depends on its own A row only,
//     so they are discarded with their outputs, and row 0 is the bits `gemv_bf16` gives for the
//     same row against the same expert's `[n, k]` (the same template and K order);
//   - `k` is a multiple of 64, so there is no K tail and no plain tail tile.
// No box crosses a column edge, so A and B may be swizzled (the TK-1 R1 rule). C goes to a plain
// `st_bf<64, 64, false>` (`warpgroup::store`, the one round-to-nearest-even), and only row 0 is
// written to `c[p]`, through TK's accessors, as `gemv_bf16.cu` writes its row 0.
//
// Shared memory: 4 stages of A and B and the plain C tile, 8 KiB each, plus 1,024 bytes of
// alignment slack = 74,752 bytes of dynamic shared memory, set once by
// `scratchy_cuda_expert_gemv_bf16_prepare` (in `PREPARE`). The semaphores are static.

#include "kittens.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::MOE_ROUTE_EXPERTS`: the depth of B.
constexpr int EXPERTS = 256;
// `step::MOE_ROUTE_MAX_TOP_K`.
constexpr uint32_t MAX_TOP_K = 32;
// The grid's y extent: one CTA row per pair.
constexpr uint32_t MAX_GRID_Y = 65535;
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
// `a [rows, k]`, swizzled.
using operand_gl = gl<bf16, 1, 1, -1, -1, tile>;
// `[experts, expert_rows, k]`: expert `e`'s weights are depth `e`.
using expert_gl = gl<bf16, 1, -1, -1, -1, tile>;
// `c [P, n]`: no tensor map, written element by element.
using output_gl = gl<bf16, 1, 1, -1, -1>;
using idx_gl = gl<int, 1, 1, -1, -1>;

struct expert_gemv_globals {
    operand_gl a;
    expert_gl b;
    output_gl c;
    // `[1, P]`.
    idx_gl topk_idx;
    // The full K tiles, `k / 64`; `a`'s row of pair `p` is `p / row_div` (`top_k` for the token
    // rows, 1 for the pair rows).
    int k_tiles;
    int row_div;
};

constexpr size_t SMEM_TILES = STAGES * 2 * sizeof(tile) + sizeof(plain_tile);
static_assert(SMEM_TILES == 73728, "4 stages of a and B and c, 8 KiB each");
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int EXPERT_GEMV_SMEM_BYTES = int(SMEM_TILES) + 1024;
static_assert(EXPERT_GEMV_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

__global__ __launch_bounds__(NUM_THREADS, 1) void expert_gemv_bf16_ker(
    const __grid_constant__ expert_gemv_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    tile(&a_smem)[STAGES] = al.allocate<tile, STAGES>();
    tile(&b_smem)[STAGES] = al.allocate<tile, STAGES>();
    plain_tile& c_smem = al.allocate<plain_tile>();

    // This CTA's 64 output columns (rows `64 · col_tile ..` of `B_e`) of pair `p`.
    const int col_tile = blockIdx.x;
    const int p = blockIdx.y;
    const int warpgroupid = threadIdx.x / kittens::WARPGROUP_THREADS;

    __shared__ kittens::semaphore arrived[STAGES], done[STAGES];
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_semaphore(arrived[s], 0, 1);
            init_semaphore(done[s], 1, 0);
        }
    }
    __syncthreads();

    if (warpgroupid == PRODUCER_WARPGROUP) {
        // The producer: one warp loads K tile `j` into stage `j % STAGES`, after the consumer
        // released that stage's previous tile.
        if (warpgroup::warpid() == 0) {
            const int e = g.topk_idx[{0, 0, 0, p}];
            const int row = p / g.row_div;
            for (int j = 0; j < g.k_tiles; j++) {
                const int stage = j % STAGES;
                if (j >= STAGES) {
                    wait(done[stage], ((j - STAGES) / STAGES) % 2);
                }
                // A's box starts at the pair's row, in elements; B's at expert `e`'s tile.
                coord<> at_a = {0, 0, row, j * TILE};
                coord<tile> at_b = {0, e, col_tile, j};
                warp::tma::expect_bytes(arrived[stage], 2 * sizeof(tile));
                warp::tma::load_async(a_smem[stage], g.a, at_a, arrived[stage]);
                warp::tma::load_async(b_smem[stage], g.b, at_b, arrived[stage]);
            }
        }
    } else if (warpgroupid == CONSUMER_WARPGROUP) {
        // The consumer: each warp holds 16 rows of the 64-row accumulator; only row 0 is kept.
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
        warpgroup::store(c_smem, acc);
        warpgroup::sync(CONSUMER_BARRIER);
        // Row 0's 64 columns, one per thread, through TK's accessors.
        const int t = warpgroup::laneid();
        if (t < TILE) {
            g.c[{0, 0, p, col_tile * TILE + t}] = c_smem[{0, t}];
        }
    }
}

}  // namespace

// The kernel's one-time setup: allow it `EXPERT_GEMV_SMEM_BYTES` of dynamic shared memory on the
// current device (the default limit is 48 KB). Run once per device before any launch
// (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch.
//
// 0 = done; any other value = the cudaError_t of setting the attribute.
extern "C" int scratchy_cuda_expert_gemv_bf16_prepare(void) {
    return static_cast<int>(cudaFuncSetAttribute(expert_gemv_bf16_ker,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 EXPERT_GEMV_SMEM_BYTES));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value
// = the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `m` non-zero, `top_k` in 1..=32, `m · top_k` at most 65,535 (the grid's y), `n` and `k`
// non-zero multiples of 64, `experts` 256, `expert_rows` at least `n`, and `a_rows · k`,
// `m · top_k · n` and `experts · expert_rows · k` at most INT_MAX, with `a_rows` `m · top_k`
// when `pair_rows`, else `m`; all are checked before any pointer is touched. `a` is
// `[a_rows, k]` bf16; `b` points at expert 0's block of `[experts, expert_rows, k]` bf16 (only
// the first `n` rows of each are read); `topk_idx` `[m · top_k]` i32, each below 256; `c`
// `[m · top_k, n]` bf16. All 16-byte aligned, `c` overlapping no input.
// `scratchy_cuda_expert_gemv_bf16_prepare` must have succeeded on the current device. The launch
// is asynchronous on `stream`.
extern "C" int scratchy_cuda_expert_gemv_bf16(const void* a, const void* b, const void* topk_idx,
                                              void* c, uint32_t m, uint32_t top_k, uint32_t n,
                                              uint32_t k, uint32_t experts, uint32_t expert_rows,
                                              uint32_t pair_rows,
                                              void* stream /* cudaStream_t; null = default */) {
    const uint64_t pairs = uint64_t(m) * uint64_t(top_k);
    const uint64_t a_rows = pair_rows != 0 ? pairs : uint64_t(m);
    if (m == 0 || top_k == 0 || top_k > MAX_TOP_K || pairs > uint64_t(MAX_GRID_Y) || n == 0 ||
        n % uint32_t(TILE) != 0 || k == 0 || k % uint32_t(TILE) != 0 ||
        experts != uint32_t(EXPERTS) || expert_rows < n ||
        a_rows * uint64_t(k) > uint64_t(INT_MAX) || pairs * uint64_t(n) > uint64_t(INT_MAX) ||
        uint64_t(experts) * uint64_t(expert_rows) * uint64_t(k) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        // Building each tile `gl` encodes its TMA tensor map into host memory; nothing is
        // allocated on the device. Every row stride is `2k` bytes, a multiple of 16.
        operand_gl ag{static_cast<bf16*>(const_cast<void*>(a)), nullptr, nullptr, size_t(a_rows),
                      size_t(k)};
        expert_gl bg{static_cast<bf16*>(const_cast<void*>(b)), nullptr, size_t(experts),
                     size_t(expert_rows), size_t(k)};
        output_gl cg{static_cast<bf16*>(c), nullptr, nullptr, size_t(pairs), size_t(n)};
        idx_gl ig{static_cast<int*>(const_cast<void*>(topk_idx)), nullptr, nullptr, size_t(1),
                  size_t(pairs)};
        expert_gemv_globals G{ag, bg, cg, ig, int(k / uint32_t(TILE)),
                              pair_rows != 0 ? 1 : int(top_k)};
        void* args[] = {&G};
        const dim3 grid(n / uint32_t(TILE), uint32_t(pairs)), block(NUM_THREADS);
        return static_cast<int>(cudaLaunchKernel(expert_gemv_bf16_ker, grid, block, args,
                                                 EXPERT_GEMV_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
