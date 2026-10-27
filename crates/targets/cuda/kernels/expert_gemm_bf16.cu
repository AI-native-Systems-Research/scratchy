// SPDX-License-Identifier: Apache-2.0
//
// The qwen3-5-moe block's grouped expert GEMM for the H100 (MoE step M3 of #857): for every live
// 128-row tile `t` of the sorted pair rows (`kernels/moe_sort.cu`), with `e = tile_expert[t]`,
//
//     C[128t .. 128t + 128, :] = A[128t .. 128t + 128, :] · B_eᵀ          (bf16, f32 accumulate)
//
// `B_e` is rows `0 .. n` of expert `e`'s `[expert_rows, k]` block of the bundle part `b` points
// at: the gate (`[2·inter, hidden]` per expert, its first `inter` rows), the up (the same blocks,
// from row `inter`) or the down (`[hidden, inter]`). A tile at or past `offsets[256] / 128` is
// never read or written.
//
// Built from `kernels/gemm_bf16.cu` (ThunderKittens' `lcf` `matmul_template<2, 4, 8>`, adapted
// from TK's `kernels/gemm/bf16_h100` at 67845f5f, MIT, see `thunderkittens/LICENSE`), PLAN-tk100
// §3.M M3. Its producer, consumer and finish are that kernel's: swizzled `st_bf<64, 64>` input
// tiles in a 4-stage ring, `rt_fl<16, 256>` accumulators, `warpgroup::mma_ABt` against the `[N, K]`
// B (`b_stack_tile`), `tma::expect` and `tma::load_async` on the input semaphore, and
// `warpgroup::mma_async_wait` before each stage's release. So every live tile's rows are the bits
// `gemm_bf16` gives for the same 128 rows against the same expert's `[n, k]`: the same WGMMA K
// order. What changes:
//   - the task walk: a task is a live tile and a 256-column block, `task_id < live · n / 256`,
//     tiles fastest so neighbouring CTAs share an expert's weights. `common_setup` reads the live
//     count `offsets[256] / 128` from a `gl<int>` (capped at the region's `max_tiles`). In `lcf`
//     a task with `num_iters < 0` ends the CTA's whole persistent loop, so only a task past the
//     last live one sets it: a dead tile is never handed out, and no TMA or semaphore is spent on
//     it;
//   - B is a `gl` whose depth axis is the expert, `[experts, expert_rows, k]`, and the producer
//     loads it at `{0, e, row block, k step}`, `e` read from `tile_expert` in `common_setup`.
// No box crosses a column edge (`n` and `k` are multiples of 256 and 64), so every tile may be
// swizzled (the TK-1 R1 rule). Shared memory: `gemm_bf16`'s, set once by
// `scratchy_cuda_expert_gemm_bf16_prepare` (in `PREPARE`).

#include "kittens.cuh"
#include "prototype.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// One thread block per SM of an H100 SXM, as `gemm_bf16.cu`: each loops over tasks until none are
// left, so correctness does not depend on this number; only speed does.
constexpr int H100_SXM_SMS = 132;
// `step::MOE_ROUTE_EXPERTS`: the depth of B, and `offsets[EXPERTS]` the padded total.
constexpr int EXPERTS = 256;

template <int M_BLOCK, int N_BLOCK>
struct expert_layout {
    using base_tile = st_bf<64, 64>;
    using global_layout = gl<bf16, 1, 1, -1, -1, base_tile>;
    // `[experts, expert_rows, k]`: expert `e`'s weights are depth `e`.
    using expert_layout_gl = gl<bf16, 1, -1, -1, -1, base_tile>;
    using idx_gl = gl<int, 1, 1, -1, -1>;
    struct globals {
        global_layout A;
        expert_layout_gl B;
        global_layout C;
        // `[1, 257]` and `[1, max_tiles]`.
        idx_gl offsets;
        idx_gl tile_expert;
    };
    struct input_block { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct finish_block { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state { int2 coord; int expert; };
    struct consumer_state { rt_fl<16, N_BLOCK * base_tile::cols> accum; };
};
template <int _M_BLOCK = 2, int _N_BLOCK = 4>
struct expert_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK;
    using layout = expert_layout<M_BLOCK, N_BLOCK>;
    // As `gemm_bf16.cu`'s: one consumer's N_BLOCK column-adjacent 64x64 finish tiles seen as one
    // [64, 64*N_BLOCK] tile, and the input block's N_BLOCK row-stacked B tiles as one [64*N_BLOCK,
    // 64] tile, layout-identical (each 8 KB, 128-byte swizzled rows, `input.b` 1024-aligned).
    using wide_tile = st_bf<64, 64 * N_BLOCK>;
    using b_stack_tile = st_bf<64 * N_BLOCK, 64>;
    static constexpr int NUM_CONSUMER_WARPS = M_BLOCK * 4, INPUT_PIPE_STAGES = 4,
                         PRODUCER_BARRIER_ARRIVALS = 1;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        const int max_tiles = args.globals.C.rows() / (M_BLOCK * 64);
        const int live = min(args.globals.offsets[{0, 0, 0, EXPERTS}] / (M_BLOCK * 64), max_tiles);
        const int Cblocks = args.globals.C.cols() / (N_BLOCK * 64);
        const int task_id = args.task_iter * gridDim.x + blockIdx.x;
        if (task_id >= live * Cblocks) {  // past the last live task: no more work to do
            args.num_iters = -1;
            return;
        }
        const int t = task_id % live;
        args.common.expert = args.globals.tile_expert[{0, 0, 0, t}];
        args.num_iters = args.globals.A.cols() / 64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS / 4 ? 0 : warpgroup::groupid();  // producer sets as 0
        args.common.coord = {t * M_BLOCK + id, (task_id / live) * N_BLOCK};
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();  // decrease registers for producers
        }
        __device__ static void load(producer_load_args<layout> args) {
            // As `gemm_bf16.cu`: one thread sets the byte count and issues every TMA load.
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for (int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x + i, args.iter}, args.inputs_arrived);
                // B is expert `e`'s [n, K]: depth = expert, tile row = N block, column = K step.
                for (int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {0, args.common.expert, args.common.coord.y + i, args.iter},
                                    args.inputs_arrived);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();  // increase registers for consumers
            kittens::warp::zero(args.state.accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_ABt(
                args.state.accum,                                // dest registers
                args.input.a[warpgroup::groupid()],              // A: this consumer's 64 rows x 64 K
                reinterpret_cast<b_stack_tile&>(args.input.b));  // B: 256 N rows x 64 K
            // The WGMMA reads shared memory asynchronously: wait for it before handing the input
            // block back to the producer.
            warpgroup::mma_async_wait();
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]),
                             args.state.accum);
            warpgroup::sync(warpgroup::groupid() + 4);
            if (warpgroup::laneid() == 0) for (int i = 0; i < N_BLOCK; i++) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i],
                                 {args.common.coord.x, args.common.coord.y + i});
                tma::store_async_read_wait();  // wait that store is finished before reusing finish memory
            }
            kittens::warp::zero(args.state.accum);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

using expert_gemm_bf16_mmt = expert_template<2, 4>;

// `gemm_bf16.cu`'s: the whole H100 budget minus the 1024 bytes `lcf` reserves for alignment.
constexpr int EXPERT_GEMM_BF16_SMEM_BYTES = kittens::MAX_SHARED_MEMORY - 1024;

}  // namespace

// The kernel's one-time setup: allow it `EXPERT_GEMM_BF16_SMEM_BYTES` of dynamic shared memory on
// the current device (the default limit is 48 KB). Run once per device before any launch
// (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch.
//
// 0 = done; any other value = the cudaError_t of setting the attribute.
extern "C" int scratchy_cuda_expert_gemm_bf16_prepare(void) {
    return static_cast<int>(
        cudaFuncSetAttribute(kittens::prototype::lcf::kernel<expert_gemm_bf16_mmt>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             EXPERT_GEMM_BF16_SMEM_BYTES));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value
// = the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `max_tiles` non-zero, `n` a non-zero multiple of 256 and `k` of 64 (whole tiles: the task walk
// truncates, so a tail would be left unwritten), `experts` 256, `expert_rows` at least `n`, and
// `max_tiles · 128 · max(n, k)` and `experts · expert_rows · k` at most INT_MAX; all are checked
// before any pointer is touched. `a` is `[max_tiles · 128, k]` and `c` `[max_tiles · 128, n]`
// bf16; `b` points at expert 0's block of `[experts, expert_rows, k]` bf16 (only the first `n`
// rows of each are read); `offsets` `[257]` and `tile_expert` `[max_tiles]` are the sort's u32
// regions, `offsets[256] / 128` live tiles each naming an expert below 256. All 16-byte aligned,
// `c` overlapping no input. `scratchy_cuda_expert_gemm_bf16_prepare` must have succeeded on the
// current device. The launch is asynchronous on `stream`.
extern "C" int scratchy_cuda_expert_gemm_bf16(const void* a, const void* b, const void* offsets,
                                              const void* tile_expert, void* c,
                                              uint32_t max_tiles, uint32_t n, uint32_t k,
                                              uint32_t experts, uint32_t expert_rows,
                                              void* stream /* cudaStream_t; null = default */) {
    using mmt = expert_gemm_bf16_mmt;
    constexpr uint32_t M_TILE = mmt::M_BLOCK * 64, N_TILE = mmt::N_BLOCK * 64, K_TILE = 64;
    const uint64_t rows = uint64_t(max_tiles) * uint64_t(M_TILE);
    const uint64_t wide = n > k ? n : k;
    if (max_tiles == 0 || n == 0 || k == 0 || n % N_TILE != 0 || k % K_TILE != 0 ||
        experts != uint32_t(EXPERTS) || expert_rows < n || rows * wide > uint64_t(INT_MAX) ||
        uint64_t(experts) * uint64_t(expert_rows) * uint64_t(k) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        using layout = typename mmt::layout;
        // Building each tile `gl` encodes its TMA tensor map into host memory; nothing is
        // allocated on the device. The index views carry no tensor map.
        typename layout::global_layout Ag{static_cast<bf16*>(const_cast<void*>(a)), nullptr,
                                          nullptr, size_t(rows), size_t(k)};
        typename layout::expert_layout_gl Bg{static_cast<bf16*>(const_cast<void*>(b)), nullptr,
                                             size_t(experts), size_t(expert_rows), size_t(k)};
        typename layout::global_layout Cg{static_cast<bf16*>(c), nullptr, nullptr, size_t(rows),
                                          size_t(n)};
        typename layout::idx_gl Og{static_cast<int*>(const_cast<void*>(offsets)), nullptr,
                                   nullptr, size_t(1), size_t(EXPERTS + 1)};
        typename layout::idx_gl Tg{static_cast<int*>(const_cast<void*>(tile_expert)), nullptr,
                                   nullptr, size_t(1), size_t(max_tiles)};
        typename layout::globals G{Ag, Bg, Cg, Og, Tg};
        void* args[] = {&G};
        const dim3 grid(H100_SXM_SMS), block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
        return static_cast<int>(cudaLaunchKernel(kittens::prototype::lcf::kernel<mmt>, grid,
                                                 block, args, EXPERT_GEMM_BF16_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
