// SPDX-License-Identifier: Apache-2.0
//
// bf16 GEMM for the H100: C = A · Bᵀ, with A [M, K], B [N, K] and C [M, N], all row-major bf16,
// accumulated in f32. B is the Linear weight layout HF checkpoints store, the one Metal's
// `gemm.metal` assumes ("C = A @ B^T, B: [N, K]").
//
// Adapted from ThunderKittens `kernels/gemm/bf16_h100/bf16_h100_gemm.cu` at 67845f5f (MIT, see
// `thunderkittens/LICENSE`). What changed from upstream:
//   - B is [N, K], not [K, N]: the producer loads B tiles at (N-block row, K-step column), and
//     the consumer calls `warpgroup::mma_ABt` instead of `mma_AB`.
//   - The persistent grid size is the named constant `H100_SXM_SMS`.
//   - Upstream's benchmark harness (`run_benchmark`, `main`, `../common.cuh`) is dropped; only the
//     kernel and an `extern "C"` launcher are here.
//
// The kernel is ThunderKittens' `lcf` producer/consumer template: one producer warpgroup issues
// TMA loads into a 4-stage ring of shared-memory input blocks, and two consumer warpgroups run
// WGMMA on them, each owning 64 of the tile's 128 output rows and all 256 of its columns.

#include "kittens.cuh"
#include "prototype.cuh"

#include <cstdint>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

// Launcher return codes that are not a `cudaError_t`. `cudaError_t` is non-negative, so negative
// codes cannot collide with it. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

// One thread block per SM of an H100 SXM. Each block loops over output tiles until none are left,
// so correctness does not depend on this number; only speed does.
constexpr int H100_SXM_SMS = 132;

template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using  base_tile      = st_bf<64, 64>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    struct globals        { global_layout A, B, C; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; };
};
template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    // The finish block's N_BLOCK column-adjacent 64x64 tiles of one consumer, seen as one
    // [64, 64*N_BLOCK] tile for `warpgroup::store` (as upstream).
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    // The input block's N_BLOCK row-stacked 64x64 B tiles, seen as one [64*N_BLOCK, 64] tile: the
    // [N, K] operand `mma_ABt` takes. Layout-identical: each 64x64 bf16 tile is 8 KB with 128-byte
    // swizzled rows, so tile i's row r sits at byte (64*i + r)*128 of `input.b`, exactly where the
    // [256, 64] tile keeps its row 64*i + r. The swizzle is a function of address bits 7-9 and
    // `input.b` starts 1024-byte aligned (the `lcf` input blocks are 1024-aligned and `a` before it
    // is a multiple of 8 KB), so the swizzle pattern matches too.
    using b_stack_tile = st_bf<64*N_BLOCK, 64>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    // ThunderKittens template functions
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (M_BLOCK*64), Cblocks = args.globals.C.cols() / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M,
                           (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else { // Id is too high, no more work to do
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid(); // producer sets as 0
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        __device__ static void load(producer_load_args<layout> args) {
            // One thread of the producer warpgroup sets the byte count and issues every TMA load;
            // the `inputs_arrived` semaphore (PRODUCER_BARRIER_ARRIVALS = 1) completes when they land.
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                // B is [N, K]: tile row = N block, tile column = K step.
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.common.coord.y+i, args.iter}, args.inputs_arrived);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>(); // increase registers for consumers
            kittens::warp::zero(args.state.accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_ABt(
                args.state.accum, // dest registers
                args.input.a[warpgroup::groupid()], // A: this consumer's 64 rows x 64 K
                reinterpret_cast<b_stack_tile&>(args.input.b) // B: 256 N rows x 64 K
            );
            // The WGMMA reads shared memory asynchronously: wait for it before handing the input
            // block back to the producer.
            warpgroup::mma_async_wait();
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if (warpgroup::laneid() == 0) for(int i = 0; i < N_BLOCK; i++) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i],
                                             {args.common.coord.x, args.common.coord.y+i});
                tma::store_async_read_wait(); // wait that store is finished before reusing finish memory
            }
            kittens::warp::zero(args.state.accum);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

using gemm_bf16_abt_mmt = matmul_template<2, 4, 8>;

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode);
// any other value = the cudaError_t from cudaFuncSetAttribute / cudaGetLastError.
//
// `m`, `n` and `k` must be non-zero multiples of 128, 256 and 64: the tile order truncates
// (`C.rows() / (M_BLOCK*64)`), so a tail would silently be left unwritten. The launch is
// asynchronous on `stream`; the caller owns it and synchronizes.
extern "C" int scratchy_cuda_gemm_bf16_abt(const void* a, const void* b, void* c,
                                           uint32_t m, uint32_t n, uint32_t k,
                                           void* stream /* cudaStream_t; null = default stream */) {
    using mmt = gemm_bf16_abt_mmt;
    constexpr uint32_t M_TILE = mmt::M_BLOCK * 64, N_TILE = mmt::N_BLOCK * 64, K_TILE = 64;
    if (m == 0 || n == 0 || k == 0 || m % M_TILE != 0 || n % N_TILE != 0 || k % K_TILE != 0) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        using global_layout = typename mmt::layout::global_layout;
        using globals       = typename mmt::layout::globals;
        // Building each `gl` encodes its TMA tensor map (`cuTensorMapEncodeTiled`).
        global_layout Ag{static_cast<bf16*>(const_cast<void*>(a)), nullptr, nullptr, size_t(m), size_t(k)};
        global_layout Bg{static_cast<bf16*>(const_cast<void*>(b)), nullptr, nullptr, size_t(n), size_t(k)};
        global_layout Cg{static_cast<bf16*>(c),                    nullptr, nullptr, size_t(m), size_t(n)};
        globals G{Ag, Bg, Cg};

        constexpr int SMEM_BYTES = MAX_SHARED_MEMORY - 1024;
        cudaError_t err = cudaFuncSetAttribute(kittens::prototype::lcf::kernel<mmt>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        if (err != cudaSuccess) return static_cast<int>(err);
        // Three warpgroups: two consumers and one producer.
        kittens::prototype::lcf::kernel<mmt><<<H100_SXM_SMS, kittens::prototype::detail::NUM_THREADS_v<mmt>,
                                               SMEM_BYTES, static_cast<cudaStream_t>(stream)>>>(G);
        return static_cast<int>(cudaGetLastError());
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
