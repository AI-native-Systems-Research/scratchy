// SPDX-License-Identifier: Apache-2.0
//
// Per-channel fp8 GEMM for the H100 (FP8 S3): C = bf16((A · Bᵀ) · w_scale[n] · a_scale[m]), with
// A [M, K] and B [N, K] row-major e4m3, accumulated in f32, `w_scale [N]` and `a_scale [M]` f32,
// and C [M, N] row-major bf16, rounded once. A is the quant scratch `QuantRowsFp8` wrote (its rows
// and their per-token scales); B is a checkpoint's e4m3 Linear weight, `[N, K]` verbatim, with its
// per-channel scale widened to f32 at load (`step::fp8_weight`).
//
// Adapted from ThunderKittens `kernels/gemm/fp8_h100_scaled/fp8_h100_gemm_scaled.cu` at 67845f5f
// (MIT, see `thunderkittens/LICENSE`; that file is not vendored), through `gemm_bf16.cu`'s
// templates, which keep their structure here. What changed from upstream:
//   - The K step is 64, not 128: A is `st_fp8e4m3<64, 64>` and B `N_BLOCK` row-stacked
//     `st_fp8e4m3<64, 64>` (64-byte rows, 64-byte swizzle), so each stage runs two `k32` WGMMAs.
//     Every fp8 `k` of the three FP8 checkpoints is a multiple of 64 (Gemma's 704 and 2112 are not
//     of 128): `step::GEMM_FP8_K_MULTIPLE`.
//   - B is `N_BLOCK` 64-row tiles, as `gemm_bf16.cu`'s, so the accumulator is `rt_fl<16, 256>`
//     (or 64 at col64), not upstream's 128.
//   - The epilogue is upstream's, `mul_col(acc, w_scale)` then `mul_row(acc, a_scale)`
//     (`scale_acc`), and the store is bf16 (`st_bf` finish tiles, `gemm_bf16.cu`'s), the one
//     rounding.
//   - Upstream's benchmark harness is dropped; only the kernels and `extern "C"` launchers are here.
//
// The instances, each `gemm_bf16.cu`'s with fp8 operands and the scale epilogue:
//   - `matmul_template<2, 4>` and `<2, 1>`: the plain GEMM at 256 and 64 output columns per tile
//     (`scratchy_cuda_gemm_fp8_abt`, `col64` 0 or 1);
//   - `group_template`: up to three GEMMs off one A, each with its own `w_scale`
//     (`scratchy_cuda_gemm_group_fp8_abt`);
//   - `gate_up_template`: the gate's and the up's GEMMs, each accumulator scaled by its own
//     `w_scale` and by `a_scale`, then `OP` (`silu_mul` or `gelu_tanh_mul`) in f32, rounded once
//     (`scratchy_cuda_gate_up_gemm_fp8`). Unlike the bf16 fold, the gate and the up are never
//     rounded to bf16 on their own: the one rounding is the activation's output's.
// Each instance's K loop is the same 64-wide steps in the same WGMMA order, so a member of a
// group, or a 64-column tile, is the plain GEMM's bits at that element.
//
// Numerics: Hopper's fp8 WGMMA accumulates inside the tensor core at reduced precision (FP8 plan
// Risk A); `tests/gemm_fp8.rs` measures it against f64 at every `k` of the three models.
//
// No 64-column box crosses a column edge (`n` and `k` are multiples of 64) and `m` is a multiple of
// 128, so every tile stays swizzled (the TK-1 R1 rule). The scales are read with `warp::load` /
// `warpgroup::load` straight from their `gl`, inside their arrays for the same reason.

#include "kittens.cuh"
#include "prototype.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// One thread block per SM of an H100 SXM, as `gemm_bf16.cu`: correctness does not depend on it.
constexpr int H100_SXM_SMS = 132;

using in_tile = st_fp8e4m3<64, 64>;
using out_tile = st_bf<64, 64>;
using in_gl = gl<fp8e4m3, 1, 1, -1, -1, in_tile>;
using out_gl = gl<bf16, 1, 1, -1, -1, out_tile>;
// `[1, len]` f32: read by `warp::load`/`warpgroup::load` into register vectors, no tensor map.
using scale_gl = gl<float, 1, 1, 1, -1>;

// `acc · w_scale[cols]` per column, then `· a_scale[rows]` per row: upstream's epilogue, in its
// order. `acc` is one consumer warpgroup's `[64, 64·N_BLOCK]` (its warp's 16 rows), `row_tile` its
// 64-row tile of `a`, `col_block` its first 64-column block (`coord.y`, a multiple of `N_BLOCK`)
// and `w_row` the row of `w` holding its scales (0 for a 1-D scale; the expert for a bank's).
template <typename ACC, typename WGL>
__device__ inline void scale_acc(ACC& acc, const WGL& w, int w_row, int col_block,
                                 const scale_gl& a, int row_tile) {
    constexpr int BLOCKS = ACC::cols / 64;
    typename ACC::row_vec ws;
    warp::load(ws, w, {0, 0, w_row, col_block / BLOCKS});
    warp::mul_col(acc, acc, ws);
    typename ACC::col_vec as;
    warpgroup::load(as, a, {0, 0, 0, row_tile});
    warp::mul_row(acc, acc, as);
}

template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    struct globals        { in_gl A, B; out_gl C; scale_gl A_scale, B_scale; };
    struct input_block    { in_tile a[M_BLOCK], b[N_BLOCK]; };
    struct finish_block   { out_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK*64> accum; };
};
template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    // `gemm_bf16.cu`'s two views. The finish block's N_BLOCK column-adjacent 64x64 bf16 tiles as
    // one [64, 64*N_BLOCK] tile. The input block's N_BLOCK row-stacked 64x64 e4m3 B tiles as one
    // [64*N_BLOCK, 64] tile, layout-identical: each is 4 KB with 64-byte swizzled rows, so tile i's
    // row r sits at byte (64*i + r)*64 of `input.b`, where the [256, 64] tile keeps its row
    // 64*i + r; the 64-byte swizzle repeats every 512 bytes (a function of address bits 4-8) and
    // `input.b` starts 1024-byte aligned (the `lcf` input blocks are, and `a` before it is 8 KB),
    // so the swizzle pattern matches too.
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    using b_stack_tile = st_fp8e4m3<64*N_BLOCK, 64>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
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
            // Two `m64nNk32` WGMMAs: the 64-byte K step.
            warpgroup::mma_ABt(
                args.state.accum,
                args.input.a[warpgroup::groupid()],
                reinterpret_cast<b_stack_tile&>(args.input.b)
            );
            warpgroup::mma_async_wait();
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            scale_acc(args.state.accum, args.globals.B_scale, 0, args.common.coord.y,
                               args.globals.A_scale, args.common.coord.x);
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

// The grouped GEMM: `gemm_bf16.cu`'s `group_template` (up to three members off one A, each its own
// GEMM's bits), each member with its own B, C and `w_scale`. An absent third member repeats the
// second's views and owns no task.
template<int M_BLOCK, int N_BLOCK>
struct group_layout {
    struct globals        { in_gl A, B0, B1, B2; out_gl C0, C1, C2;
                            scale_gl A_scale, S0, S1, S2; int members; };
    struct input_block    { in_tile a[M_BLOCK], b[N_BLOCK]; };
    struct finish_block   { out_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int2 coord; int member; };
    struct consumer_state { rt_fl<16, N_BLOCK*64> accum; };
};
template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct group_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = group_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    using b_stack_tile = st_fp8e4m3<64*N_BLOCK, 64>;
    using globals = typename layout::globals;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    __device__ static inline const in_gl& b_of(const globals& g, int m) {
        return m == 2 ? g.B2 : m == 1 ? g.B1 : g.B0;
    }
    __device__ static inline const out_gl& c_of(const globals& g, int m) {
        return m == 2 ? g.C2 : m == 1 ? g.C1 : g.C0;
    }
    __device__ static inline const scale_gl& s_of(const globals& g, int m) {
        return m == 2 ? g.S2 : m == 1 ? g.S1 : g.S0;
    }
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        const auto& G = args.globals;
        int Rblocks = G.A.rows() / (M_BLOCK*64);
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        int member = 0;
        for (; member < G.members; member++) {
            int tasks = Rblocks * (c_of(G, member).cols() / (N_BLOCK*64));
            if (task_id < tasks) break;
            task_id -= tasks;
        }
        if (member == G.members) { // no more work to do
            args.num_iters = -1;
            return;
        }
        // `matmul_template`'s tile order, within the member.
        int Cblocks = c_of(G, member).cols() / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M,
                           (task_id%super_repeat)/SUPER_M };
        else {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        args.common.member = member;
        args.num_iters = G.A.cols()/64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid(); // producer sets as 0
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], b_of(args.globals, args.common.member),
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
                args.state.accum,
                args.input.a[warpgroup::groupid()],
                reinterpret_cast<b_stack_tile&>(args.input.b)
            );
            warpgroup::mma_async_wait();
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            const int member = args.common.member;
            scale_acc(args.state.accum, s_of(args.globals, member), 0,
                               args.common.coord.y, args.globals.A_scale, args.common.coord.x);
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if (warpgroup::laneid() == 0) for(int i = 0; i < N_BLOCK; i++) {
                tma::store_async(c_of(args.globals, member),
                                 args.finish.c[warpgroup::groupid()][i],
                                 {args.common.coord.x, args.common.coord.y+i});
                tma::store_async_read_wait();
            }
            kittens::warp::zero(args.state.accum);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

// The gate/up GEMM: `gemm_bf16.cu`'s `gate_up_template` at half the column block (`N_BLOCK = 2`,
// two `rt_fl<16, 128>` accumulators), each accumulator scaled by its own projection's `w_scale`
// and by `a_scale` (so each is the plain fp8 GEMM's value before its rounding), then `OP` by TK's
// map in f32, then the one rounding at the store. Only the result is stored, so the finish block
// holds one set of tiles.
template<int M_BLOCK, int N_BLOCK>
struct gate_up_layout {
    struct globals        { in_gl A, G, U; out_gl C; scale_gl A_scale, G_scale, U_scale; };
    struct input_block    { in_tile a[M_BLOCK], g[N_BLOCK], u[N_BLOCK]; };
    struct finish_block   { out_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK*64> gate, up; };
};
template<typename OP, int _M_BLOCK=2, int _N_BLOCK=2, int _SUPER_M=12>
struct gate_up_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = gate_up_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    using b_stack_tile = st_fp8e4m3<64*N_BLOCK, 64>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    // `matmul_template`'s tile order.
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
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++) {
                    tma::load_async(args.input.g[i], args.globals.G,
                                    {args.common.coord.y+i, args.iter}, args.inputs_arrived);
                    tma::load_async(args.input.u[i], args.globals.U,
                                    {args.common.coord.y+i, args.iter}, args.inputs_arrived);
                }
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>(); // increase registers for consumers
            kittens::warp::zero(args.state.gate);
            kittens::warp::zero(args.state.up);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_ABt(args.state.gate, args.input.a[warpgroup::groupid()],
                               reinterpret_cast<b_stack_tile&>(args.input.g));
            warpgroup::mma_ABt(args.state.up, args.input.a[warpgroup::groupid()],
                               reinterpret_cast<b_stack_tile&>(args.input.u));
            warpgroup::mma_async_wait();
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            const int c = warpgroup::groupid();
            scale_acc(args.state.gate, args.globals.G_scale, 0, args.common.coord.y,
                               args.globals.A_scale, args.common.coord.x);
            scale_acc(args.state.up, args.globals.U_scale, 0, args.common.coord.y,
                               args.globals.A_scale, args.common.coord.x);
            kittens::warp::bin_map<OP>(args.state.gate, args.state.gate, args.state.up);
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[c]), args.state.gate);
            warpgroup::sync(c+4);
            if (warpgroup::laneid() == 0) for(int i = 0; i < N_BLOCK; i++) {
                tma::store_async(args.globals.C, args.finish.c[c][i],
                                 {args.common.coord.x, args.common.coord.y+i});
                tma::store_async_read_wait(); // wait that store is finished before reusing finish memory
            }
            kittens::warp::zero(args.state.gate);
            kittens::warp::zero(args.state.up);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

// The MoE block's grouped expert GEMM over an fp8 bank: `expert_gemm_bf16.cu`'s task walk (a task
// is a live 128-row tile of the sorted pair rows and a column block, `e = tile_expert[t]`) with
// `matmul_template`'s fp8 producer, consumer and epilogue. B is expert `e`'s e4m3 `[n, k]` (depth
// `e` of `[E, expert_rows, k]`), its scale row `e` of `[E, expert_rows]` f32, and `a_scale` the
// pair rows' (each pair row was quantized on its own). So every live tile's rows are the bits
// `matmul_template` gives for the same 128 rows against the same expert's `[n, k]` and scales.
//
// FP8 S8 P3(a): `M_BLOCK` 1 makes a task one 64-row tile (one consumer warpgroup), so the sort's
// padding is per 64 rows; `K_STEP` 128 makes a stage one `st_fp8e4m3<64, 128>` (128-byte rows,
// 128-byte swizzle) of A and of each B block, four `k32` WGMMAs a stage in place of two. The K
// loop is the same `k32` WGMMAs in the same order, and each output row is its own A row's K
// sequence, so every row is the bits of `M_BLOCK` 2, `K_STEP` 64 (`tests/moe.rs`). `K_STEP` 128
// needs `k` a multiple of 128: a 128-byte box past `k` reads the next row's bytes (TK-1's R1).
template<int M_BLOCK, int N_BLOCK, int K_STEP = 64>
struct expert_layout {
    using k_tile = st_fp8e4m3<64, K_STEP>;
    using a_gl = gl<fp8e4m3, 1, 1, -1, -1, k_tile>;
    // `[experts, expert_rows, k]`: expert `e`'s weights are depth `e`.
    using expert_gl = gl<fp8e4m3, 1, -1, -1, -1, k_tile>;
    // `[experts, expert_rows]` f32 per-channel scales, and the sort's `[1, len]` index regions.
    using expert_scale_gl = gl<float, 1, 1, -1, -1>;
    using idx_gl = gl<int, 1, 1, -1, -1>;
    struct globals {
        a_gl A;
        expert_gl B;
        out_gl C;
        scale_gl A_scale;
        expert_scale_gl B_scale;
        // `[1, E + 1]` and `[1, max_tiles]`.
        idx_gl offsets;
        idx_gl tile_expert;
    };
    struct input_block    { k_tile a[M_BLOCK], b[N_BLOCK]; };
    struct finish_block   { out_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int2 coord; int expert; };
    struct consumer_state { rt_fl<16, N_BLOCK*64> accum; };
};
template<int _EXPERTS, int _N_BLOCK, int _M_BLOCK=2, int _K_STEP=64>
struct expert_template {
    static constexpr int EXPERTS = _EXPERTS, M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK;
    static constexpr int K_STEP = _K_STEP;
    using layout = expert_layout<M_BLOCK, N_BLOCK, K_STEP>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    using b_stack_tile = st_fp8e4m3<64*N_BLOCK, K_STEP>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        const int max_tiles = args.globals.C.rows() / (M_BLOCK*64);
        const int live = min(args.globals.offsets[{0, 0, 0, EXPERTS}] / (M_BLOCK*64), max_tiles);
        const int Cblocks = args.globals.C.cols() / (N_BLOCK*64);
        const int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id >= live * Cblocks) {  // past the last live task: no more work to do
            args.num_iters = -1;
            return;
        }
        const int t = task_id % live;
        args.common.expert = args.globals.tile_expert[{0, 0, 0, t}];
        args.num_iters = args.globals.A.cols()/K_STEP;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid(); // producer sets as 0
        args.common.coord = {t*M_BLOCK + id, (task_id / live)*N_BLOCK};
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                // B is expert `e`'s [n, K]: depth = expert, tile row = N block, column = K step.
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {0, args.common.expert, args.common.coord.y+i, args.iter},
                                    args.inputs_arrived);
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
                args.state.accum,
                args.input.a[warpgroup::groupid()],
                reinterpret_cast<b_stack_tile&>(args.input.b)
            );
            warpgroup::mma_async_wait();
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            scale_acc(args.state.accum, args.globals.B_scale, args.common.expert,
                               args.common.coord.y, args.globals.A_scale, args.common.coord.x);
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

using gemm_fp8_abt_mmt = matmul_template<2, 4, 8>;
using gemm_col64_fp8_abt_mmt = matmul_template<2, 1, 8>;
using gemm_group_fp8_abt_mmt = group_template<2, 4, 8>;
using gate_up_silu_fp8_mmt = gate_up_template<scratchy::ops::silu_mul, 2, 2, 8>;
using gate_up_gelu_fp8_mmt = gate_up_template<scratchy::ops::gelu_tanh_mul, 2, 2, 8>;
// One 64-column block per accumulator: the 64-column GEMM's shapes (Gemma's 2112-column MLP).
using gate_up_silu_col64_fp8_mmt = gate_up_template<scratchy::ops::silu_mul, 2, 1, 8>;
using gate_up_gelu_col64_fp8_mmt = gate_up_template<scratchy::ops::gelu_tanh_mul, 2, 1, 8>;
// `op_abi::MOE_EXPERT_COUNTS` and `step::EXPERT_GEMM_N_TILES`, as `expert_gemm_bf16.cu`'s three.
constexpr int EXPERTS_SMALL = 128;
constexpr int EXPERTS_LARGE = 256;
constexpr uint32_t N_TILE_NARROW = 64;
constexpr uint32_t N_TILE_WIDE = 256;
using expert_gemm_fp8_mmt = expert_template<EXPERTS_LARGE, 4>;
using expert_gemm_e128_fp8_mmt = expert_template<EXPERTS_SMALL, 4>;
using expert_gemm_e128_col64_fp8_mmt = expert_template<EXPERTS_SMALL, 1>;
// FP8 S8 P3(a): the 64-row tiles at K step 64 and 128, each expert count and column tile.
using expert_gemm_m64_k64_fp8_mmt = expert_template<EXPERTS_LARGE, 4, 1, 64>;
using expert_gemm_m64_k128_fp8_mmt = expert_template<EXPERTS_LARGE, 4, 1, 128>;
using expert_gemm_e128_m64_k64_fp8_mmt = expert_template<EXPERTS_SMALL, 4, 1, 64>;
using expert_gemm_e128_m64_k128_fp8_mmt = expert_template<EXPERTS_SMALL, 4, 1, 128>;
using expert_gemm_e128_col64_m64_k64_fp8_mmt = expert_template<EXPERTS_SMALL, 1, 1, 64>;
using expert_gemm_e128_col64_m64_k128_fp8_mmt = expert_template<EXPERTS_SMALL, 1, 1, 128>;

// `gemm_bf16.cu`'s: the whole H100 budget minus the 1024 bytes `lcf` reserves for alignment.
constexpr int GEMM_FP8_SMEM_BYTES = kittens::MAX_SHARED_MEMORY - 1024;

// `GEMM_FP8_SMEM_BYTES` of dynamic shared memory for `mmt`'s instance. One call per instance: the
// instances take different `globals` types, so no braced list holds them.
template <typename mmt>
cudaError_t smem_attr() {
    return cudaFuncSetAttribute(kittens::prototype::lcf::kernel<mmt>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM_FP8_SMEM_BYTES);
}

// The shape rule of `mmt`'s tile: `m`, `n` and `k` non-zero multiples of `M_BLOCK·64`, `N_BLOCK·64`
// and 64, none above INT_MAX.
template <typename mmt>
bool tile_shape(uint32_t m, uint32_t n, uint32_t k) {
    constexpr uint32_t M_TILE = mmt::M_BLOCK * 64, N_TILE = mmt::N_BLOCK * 64, K_TILE = 64;
    constexpr uint32_t DIM_MAX = static_cast<uint32_t>(INT_MAX);
    return m != 0 && n != 0 && k != 0 && m % M_TILE == 0 && n % N_TILE == 0 && k % K_TILE == 0 &&
           m <= DIM_MAX && n <= DIM_MAX && k <= DIM_MAX;
}

// A row-major `[rows, cols]` operand view; building it encodes its TMA tensor map into host memory
// (nothing is allocated on the device). Can throw.
in_gl in_view(const void* p, uint32_t rows, uint32_t cols) {
    return in_gl{static_cast<fp8e4m3*>(const_cast<void*>(p)), nullptr, nullptr, size_t(rows),
                 size_t(cols)};
}
out_gl out_view(void* p, uint32_t rows, uint32_t cols) {
    return out_gl{static_cast<bf16*>(p), nullptr, nullptr, size_t(rows), size_t(cols)};
}
// A `[len]` f32 scale view: no tensor map.
scale_gl scale_view(const void* p, uint32_t len) {
    return scale_gl{static_cast<float*>(const_cast<void*>(p)), nullptr, nullptr, nullptr,
                    size_t(len)};
}

// Launch `mmt`'s instance on `G`: three warpgroups (two consumers, one producer), the persistent
// grid, `G` by value (`__grid_constant__`; cudaLaunchKernel copies it before returning).
template <typename mmt>
int launch(typename mmt::layout::globals& G, void* stream) {
    void* args[] = {&G};
    const dim3 grid(H100_SXM_SMS), block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    return static_cast<int>(cudaLaunchKernel(kittens::prototype::lcf::kernel<mmt>, grid, block,
                                             args, GEMM_FP8_SMEM_BYTES,
                                             static_cast<cudaStream_t>(stream)));
}

template <typename mmt>
int launch_abt(const void* a, const void* a_scale, const void* b, const void* b_scale, void* c,
               uint32_t m, uint32_t n, uint32_t k, void* stream) {
    if (!tile_shape<mmt>(m, n, k)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        typename mmt::layout::globals G{in_view(a, m, k), in_view(b, n, k), out_view(c, m, n),
                                        scale_view(a_scale, m), scale_view(b_scale, n)};
        return launch<mmt>(G, stream);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

template <typename mmt, typename rule>
int launch_gate_up(const void* a, const void* a_scale, const void* gate, const void* gate_scale,
                   const void* up, const void* up_scale, void* c, uint32_t m, uint32_t n,
                   uint32_t k, void* stream) {
    if (!tile_shape<rule>(m, n, k)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        typename mmt::layout::globals G{in_view(a, m, k),        in_view(gate, n, k),
                                        in_view(up, n, k),       out_view(c, m, n),
                                        scale_view(a_scale, m),  scale_view(gate_scale, n),
                                        scale_view(up_scale, n)};
        return launch<mmt>(G, stream);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// The launch of `mmt`'s expert instance, the shape already checked against its tile. Can throw.
template <typename mmt>
int launch_expert(const void* a, const void* a_scale, const void* b, const void* b_scale,
                  const void* offsets, const void* tile_expert, void* c, uint64_t rows, uint32_t n,
                  uint32_t k, uint32_t max_tiles, uint32_t expert_rows, void* stream) {
    using layout = typename mmt::layout;
    typename layout::expert_gl Bg{static_cast<fp8e4m3*>(const_cast<void*>(b)), nullptr,
                                  size_t(mmt::EXPERTS), size_t(expert_rows), size_t(k)};
    typename layout::expert_scale_gl Sg{static_cast<float*>(const_cast<void*>(b_scale)), nullptr,
                                        nullptr, size_t(mmt::EXPERTS), size_t(expert_rows)};
    typename layout::idx_gl Og{static_cast<int*>(const_cast<void*>(offsets)), nullptr, nullptr,
                               size_t(1), size_t(mmt::EXPERTS + 1)};
    typename layout::idx_gl Tg{static_cast<int*>(const_cast<void*>(tile_expert)), nullptr,
                               nullptr, size_t(1), size_t(max_tiles)};
    typename layout::a_gl Ag{static_cast<fp8e4m3*>(const_cast<void*>(a)), nullptr, nullptr,
                             size_t(rows), size_t(k)};
    typename layout::globals G{Ag, Bg, out_view(c, uint32_t(rows), n),
                               scale_view(a_scale, uint32_t(rows)), Sg, Og, Tg};
    return launch<mmt>(G, stream);
}

}  // namespace

// The kernels' one-time setup: allow every instance of this file `GEMM_FP8_SMEM_BYTES` of dynamic
// shared memory on the current device (the default limit is 48 KB). Run once per device before any
// launch (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch.
//
// 0 = done; any other value = the cudaError_t of the first call that failed.
extern "C" int scratchy_cuda_gemm_fp8_prepare(void) {
    cudaError_t e = smem_attr<gemm_fp8_abt_mmt>();
    if (e == cudaSuccess) {
        e = smem_attr<gemm_col64_fp8_abt_mmt>();
    }
    if (e == cudaSuccess) {
        e = smem_attr<gemm_group_fp8_abt_mmt>();
    }
    if (e == cudaSuccess) {
        e = smem_attr<gate_up_silu_fp8_mmt>();
    }
    if (e == cudaSuccess) {
        e = smem_attr<gate_up_gelu_fp8_mmt>();
    }
    if (e == cudaSuccess) {
        e = smem_attr<gate_up_silu_col64_fp8_mmt>();
    }
    if (e == cudaSuccess) {
        e = smem_attr<gate_up_gelu_col64_fp8_mmt>();
    }
    if (e == cudaSuccess) {
        e = smem_attr<expert_gemm_fp8_mmt>();
    }
    if (e == cudaSuccess) {
        e = smem_attr<expert_gemm_e128_fp8_mmt>();
    }
    if (e == cudaSuccess) {
        e = smem_attr<expert_gemm_e128_col64_fp8_mmt>();
    }
    if (e == cudaSuccess) {
        e = smem_attr<expert_gemm_m64_k64_fp8_mmt>();
    }
    if (e == cudaSuccess) {
        e = smem_attr<expert_gemm_m64_k128_fp8_mmt>();
    }
    if (e == cudaSuccess) {
        e = smem_attr<expert_gemm_e128_m64_k64_fp8_mmt>();
    }
    if (e == cudaSuccess) {
        e = smem_attr<expert_gemm_e128_m64_k128_fp8_mmt>();
    }
    if (e == cudaSuccess) {
        e = smem_attr<expert_gemm_e128_col64_m64_k64_fp8_mmt>();
    }
    if (e == cudaSuccess) {
        e = smem_attr<expert_gemm_e128_col64_m64_k128_fp8_mmt>();
    }
    return static_cast<int>(e);
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value
// = the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `c [m, n] = bf16((a · bᵀ) · b_scale[n] · a_scale[m])`: `a [m, k]` and `b [n, k]` e4m3, `a_scale
// [m]` and `b_scale [n]` f32, `c` bf16. At `col64` 0, `m`, `n` and `k` must be non-zero multiples
// of 128, 256 and 64; at `col64` 1, of 128, 64 and 64 (the 64-column tiles); none above INT_MAX,
// `col64` 0 or 1, all checked before any pointer is touched. On an `n` both take, each output
// element is the same bits. `a` and `b` must be 16-byte-aligned device pointers, the scales
// 8-byte-aligned, `c` 16-byte-aligned and overlapping no input. `scratchy_cuda_gemm_fp8_prepare`
// must have succeeded on the current device. The launch is asynchronous on `stream`.
extern "C" int scratchy_cuda_gemm_fp8_abt(const void* a, const void* a_scale, const void* b,
                                          const void* b_scale, void* c, uint32_t m, uint32_t n,
                                          uint32_t k, uint32_t col64,
                                          void* stream /* cudaStream_t; null = default */) {
    if (col64 > 1) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    return col64 == 1
               ? launch_abt<gemm_col64_fp8_abt_mmt>(a, a_scale, b, b_scale, c, m, n, k, stream)
               : launch_abt<gemm_fp8_abt_mmt>(a, a_scale, b, b_scale, c, m, n, k, stream);
}

// As `scratchy_cuda_gemm_fp8_abt` at `col64` 0 for `count` (2 or 3) members of one `a [m, k]`
// and `a_scale [m]`: member `i` is `c[i] [m, n[i]] = bf16((a · b[i]ᵀ) · b_scale[i][n] ·
// a_scale[m])`, each `n[i]` a non-zero multiple of 256 at most INT_MAX (with the shared `m` and
// `k` under the plain GEMM's rule), checked before any pointer is touched. Each member's output is
// the bits of its own `scratchy_cuda_gemm_fp8_abt`. `scratchy_cuda_gemm_fp8_prepare` must have
// succeeded on the current device.
extern "C" int scratchy_cuda_gemm_group_fp8_abt(const void* a, const void* a_scale,
                                                const void* const* b, const void* const* b_scale,
                                                void* const* c, const uint32_t* n, uint32_t m,
                                                uint32_t k, uint32_t count,
                                                void* stream /* cudaStream_t; null = default */) {
    using mmt = gemm_group_fp8_abt_mmt;
    if (count < 2 || count > 3) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    for (uint32_t i = 0; i < count; i++) {
        if (!tile_shape<mmt>(m, n[i], k)) {
            return SCRATCHY_CUDA_ERR_SHAPE;
        }
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const uint32_t last = count - 1;
        const auto weight = [&](uint32_t i) { return in_view(b[i], n[i], k); };
        const auto out = [&](uint32_t i) { return out_view(c[i], m, n[i]); };
        const auto ws = [&](uint32_t i) { return scale_view(b_scale[i], n[i]); };
        typename mmt::layout::globals G{in_view(a, m, k),
                                        weight(0), weight(1), weight(last),
                                        out(0), out(1), out(last),
                                        scale_view(a_scale, m), ws(0), ws(1), ws(last),
                                        int(count)};
        return launch<mmt>(G, stream);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// The fp8 gate/up GEMM: `c [m, n] = bf16(act(g, u))`, `g = (a · gateᵀ) · gate_scale[n] ·
// a_scale[m]` and `u` the up's, both in f32, rounded once; `act` 0 for SwiGLU's `silu_mul`, 1 for
// GeGLU's `gelu_tanh_mul`. Under `scratchy_cuda_gemm_fp8_abt`'s shape rule at the same `col64`
// (`m`, `n`, `k` non-zero multiples of 128, 256 or 64 at `col64` 1, and 64), none above INT_MAX,
// `act` and `col64` 0 or 1, all checked before any pointer is touched. Each accumulator is the
// value `scratchy_cuda_gemm_fp8_abt` rounds. Pointers as `scratchy_cuda_gemm_fp8_abt`'s; `gate`
// and `up` `[n, k]`, `c` overlapping no input. `scratchy_cuda_gemm_fp8_prepare` must have
// succeeded on the current device.
extern "C" int scratchy_cuda_gate_up_gemm_fp8(const void* a, const void* a_scale, const void* gate,
                                              const void* gate_scale, const void* up,
                                              const void* up_scale, void* c, uint32_t m,
                                              uint32_t n, uint32_t k, uint32_t act, uint32_t col64,
                                              void* stream /* cudaStream_t; null = default */) {
    if (act > 1 || col64 > 1) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    if (col64 == 1) {
        return act == 0 ? launch_gate_up<gate_up_silu_col64_fp8_mmt, gemm_col64_fp8_abt_mmt>(
                              a, a_scale, gate, gate_scale, up, up_scale, c, m, n, k, stream)
                        : launch_gate_up<gate_up_gelu_col64_fp8_mmt, gemm_col64_fp8_abt_mmt>(
                              a, a_scale, gate, gate_scale, up, up_scale, c, m, n, k, stream);
    }
    return act == 0 ? launch_gate_up<gate_up_silu_fp8_mmt, gemm_fp8_abt_mmt>(
                          a, a_scale, gate, gate_scale, up, up_scale, c, m, n, k, stream)
                    : launch_gate_up<gate_up_gelu_fp8_mmt, gemm_fp8_abt_mmt>(
                          a, a_scale, gate, gate_scale, up, up_scale, c, m, n, k, stream);
}

// The grouped expert GEMM over an fp8 bank (`expert_gemm_bf16.cu`'s, FP8 S3): for every live
// 128-row tile `t` of the sorted pair rows, `e = tile_expert[t]`, `C[128t .., :] =
// bf16((A[128t .., :] · B_eᵀ) · b_scale[e, :n] · a_scale[128t ..])`. Return codes as
// `scratchy_cuda_gemm_fp8_abt`'s; the shape rule, refusals and the index regions are
// `scratchy_cuda_expert_gemm_bf16`'s (`max_tiles` non-zero, `(experts, n_tile)` one of (256, 256),
// (128, 256) and (128, 64), `n` a non-zero multiple of `n_tile` and `k` of 64, `expert_rows` at
// least `n`, `max_tiles · 128 · max(n, k)` and `E · expert_rows · k` at most INT_MAX), all checked
// before any pointer is touched. `a` is `[max_tiles · 128, k]` e4m3 and `a_scale` its rows' f32;
// `b` points at expert 0's block of `[E, expert_rows, k]` e4m3 and `b_scale` at expert 0's row of
// `[E, expert_rows]` f32 (only the first `n` of each are read); `c` is `[max_tiles · 128, n]` bf16.
// The tensors 16-byte aligned, the scales 8-byte aligned, `c` overlapping no input.
// `scratchy_cuda_gemm_fp8_prepare` must have succeeded on the current device.
extern "C" int scratchy_cuda_expert_gemm_fp8(const void* a, const void* a_scale, const void* b,
                                             const void* b_scale, const void* offsets,
                                             const void* tile_expert, void* c, uint32_t max_tiles,
                                             uint32_t n, uint32_t k, uint32_t experts,
                                             uint32_t expert_rows, uint32_t n_tile,
                                             void* stream /* cudaStream_t; null = default */) {
    constexpr uint32_t M_TILE = 128, K_TILE = 64;
    const bool wide_large = experts == uint32_t(EXPERTS_LARGE) && n_tile == N_TILE_WIDE;
    const bool wide_small = experts == uint32_t(EXPERTS_SMALL) && n_tile == N_TILE_WIDE;
    const bool narrow_small = experts == uint32_t(EXPERTS_SMALL) && n_tile == N_TILE_NARROW;
    const uint64_t rows = uint64_t(max_tiles) * uint64_t(M_TILE);
    const uint64_t wide = n > k ? n : k;
    if (max_tiles == 0 || !(wide_large || wide_small || narrow_small) || n == 0 || k == 0 ||
        n % n_tile != 0 || k % K_TILE != 0 || expert_rows < n || rows * wide > uint64_t(INT_MAX) ||
        uint64_t(experts) * uint64_t(expert_rows) * uint64_t(k) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    static_assert(expert_gemm_fp8_mmt::M_BLOCK * 64 == int(M_TILE), "the sort's row tile");
    static_assert(expert_gemm_fp8_mmt::N_BLOCK * 64 == int(N_TILE_WIDE), "the wide column tile");
    static_assert(expert_gemm_e128_col64_fp8_mmt::N_BLOCK * 64 == int(N_TILE_NARROW),
                  "the narrow column tile");
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        if (wide_large) {
            return launch_expert<expert_gemm_fp8_mmt>(a, a_scale, b, b_scale, offsets,
                                                      tile_expert, c, rows, n, k, max_tiles,
                                                      expert_rows, stream);
        }
        if (wide_small) {
            return launch_expert<expert_gemm_e128_fp8_mmt>(a, a_scale, b, b_scale, offsets,
                                                           tile_expert, c, rows, n, k, max_tiles,
                                                           expert_rows, stream);
        }
        return launch_expert<expert_gemm_e128_col64_fp8_mmt>(a, a_scale, b, b_scale, offsets,
                                                             tile_expert, c, rows, n, k,
                                                             max_tiles, expert_rows, stream);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// Test support (`tests/moe.rs`, FP8 S8 P3(a)): `scratchy_cuda_expert_gemm_fp8` at row tile `m_tile`
// (128, today's instances, K step 64; or 64) and K step `k_step` (64; or 128, `k` a multiple of
// 128). `max_tiles` counts `m_tile`-row tiles, and `tile_expert` names each one's expert: a
// sort padded to `m_tile` rows. Each pair row is the same bits at every row tile and K step.
// Return codes and rule otherwise as that launcher's.
extern "C" int scratchy_cuda_expert_gemm_fp8_variant(
    const void* a, const void* a_scale, const void* b, const void* b_scale, const void* offsets,
    const void* tile_expert, void* c, uint32_t max_tiles, uint32_t n, uint32_t k, uint32_t experts,
    uint32_t expert_rows, uint32_t n_tile, uint32_t m_tile, uint32_t k_step,
    void* stream /* cudaStream_t; null = default */) {
    if (m_tile == 128 && k_step == 64) {
        return scratchy_cuda_expert_gemm_fp8(a, a_scale, b, b_scale, offsets, tile_expert, c,
                                             max_tiles, n, k, experts, expert_rows, n_tile,
                                             stream);
    }
    const bool wide_large = experts == uint32_t(EXPERTS_LARGE) && n_tile == N_TILE_WIDE;
    const bool wide_small = experts == uint32_t(EXPERTS_SMALL) && n_tile == N_TILE_WIDE;
    const bool narrow_small = experts == uint32_t(EXPERTS_SMALL) && n_tile == N_TILE_NARROW;
    const uint64_t rows = uint64_t(max_tiles) * 64;
    const uint64_t wide = n > k ? n : k;
    if (m_tile != 64 || (k_step != 64 && k_step != 128) || max_tiles == 0 ||
        !(wide_large || wide_small || narrow_small) || n == 0 || k == 0 || n % n_tile != 0 ||
        k % k_step != 0 || expert_rows < n || rows * wide > uint64_t(INT_MAX) ||
        uint64_t(experts) * uint64_t(expert_rows) * uint64_t(k) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const auto go = [&](auto mmt) {
            return launch_expert<decltype(mmt)>(a, a_scale, b, b_scale, offsets, tile_expert, c,
                                                rows, n, k, max_tiles, expert_rows, stream);
        };
        if (wide_large) {
            return k_step == 128 ? go(expert_gemm_m64_k128_fp8_mmt{}) : go(expert_gemm_m64_k64_fp8_mmt{});
        }
        if (wide_small) {
            return k_step == 128 ? go(expert_gemm_e128_m64_k128_fp8_mmt{})
                                 : go(expert_gemm_e128_m64_k64_fp8_mmt{});
        }
        return k_step == 128 ? go(expert_gemm_e128_col64_m64_k128_fp8_mmt{})
                             : go(expert_gemm_e128_col64_m64_k64_fp8_mmt{});
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
