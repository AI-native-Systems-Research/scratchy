// SPDX-License-Identifier: Apache-2.0
//
// The qwen3-5-moe block's combine for the H100 (MoE step M5 of #857): per token row `t`,
//
//     out[t, :] = Σ_{j < top_k} scores[t, j] · down[row(t, j), :]          (bf16 out)
//
// over the slots in order, in f32, rounded once. `row(t, j)` is `pos[t · top_k + j]` when the
// pairs were sorted by expert (the prefill buckets), else `t · top_k + j` (the gathered decode
// bucket). `down` is the expert GEMM's or GEMV's bf16 output, `scores` the route's f32 weights.
//
// ThunderKittens' warp vector (PLAN-tk100 §2 S4, §3.M M5): one warp per token row, 8 warps per CTA.
// The row is `hidden / 256` column chunks of a naive `rv_fl<256>`. Per chunk, the accumulator is
// zeroed, then for each slot in order its down row's chunk is loaded (`warp::load`, bf16 to f32,
// exact), scaled by the slot's score (`warp::mul`) and added (`warp::add`), and the chunk stored
// once (`warp::store`, the one round-to-nearest-even). The product and the sum are two map calls,
// so each term rounds twice and is never fused: a fixed order, which #841's combine bound allows.
// The row and score indices are read through `gl<int>` and `gl<float>` views. Every token row of
// the bucket is written.
//
// THE ONE-ROW FORM (FP8 S8 W1): at `m == 1` (every decode step) the grid is one CTA, one warp of
// which walks every chunk of the row. The launcher then launches the `ONE_ROW` instance on a grid
// of `chunks` CTAs of one warp, warp `c` running chunk `c`'s loop unchanged: the same slots in the
// same order, the same loads, maps and store. A chunk's sum never reads another chunk, so the
// bits are the warp-per-row kernel's. Chosen from `m`, a constant of each step.

#include "kittens.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::MOE_COMBINE_BF16_CHUNK`: one naive `rv_fl<256>`; `hidden` is a multiple of it.
constexpr int CHUNK = 256;
// `step::MOE_ROUTE_MAX_TOP_K`.
constexpr uint32_t MAX_TOP_K = 32;
constexpr int WARPS = 8;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;

using chunk_vec = rv_fl<CHUNK>;
using row_gl = gl<bf16, 1, 1, -1, -1>;
using idx_gl = gl<int, 1, 1, -1, -1>;
using score_gl = gl<float, 1, 1, -1, -1>;

struct combine_globals {
    // `[down_rows, hidden]`.
    row_gl down;
    // `[1, m · top_k]`: pair to sorted row, read only when `sorted`.
    idx_gl pos;
    // `[m, top_k]`.
    score_gl scores;
    // `[m, hidden]`.
    row_gl out;
    int m;
    int top_k;
    int chunks;
    bool sorted;
};

template <bool ONE_ROW>
__global__ __launch_bounds__(NUM_THREADS) void moe_combine_bf16_ker(
    const __grid_constant__ combine_globals g) {
    // The token row, and the chunks this warp writes: every chunk of row `t`, or under `ONE_ROW`
    // (one warp a CTA) chunk `blockIdx.x` of row 0 alone.
    const int t = ONE_ROW ? 0 : int(blockIdx.x) * WARPS + kittens::warpid();
    if (t >= g.m) {
        return;
    }
    const int ch0 = ONE_ROW ? int(blockIdx.x) : 0;
    const int ch1 = ONE_ROW ? ch0 + 1 : g.chunks;
    for (int ch = ch0; ch < ch1; ++ch) {
        chunk_vec acc, d;
        warp::zero(acc);
        for (int j = 0; j < g.top_k; ++j) {
            const int pair = t * g.top_k + j;
            const int row = g.sorted ? g.pos[{0, 0, 0, pair}] : pair;
            const float s = g.scores[{0, 0, t, j}];
            warp::load(d, g.down, {0, 0, row, ch});
            warp::mul(d, d, s);
            warp::add(acc, acc, d);
        }
        warp::store(g.out, acc, {0, 0, t, ch});
    }
}

}  // namespace

// The kernel's one-time setup, run once per device by `kernels::prepare_all` (`CudaDevice::open`):
// query its attributes (`cudaFuncGetAttributes`), which loads its module, so a worker's graph
// capture (`CudaWorker::new`) never loads a kernel mid-capture. It changes nothing on the device.
//
// 0 = done; any other value = the cudaError_t of the first query that failed.
extern "C" int scratchy_cuda_moe_combine_bf16_prepare(void) {
    cudaFuncAttributes attr;
    for (auto ker : {moe_combine_bf16_ker<false>, moe_combine_bf16_ker<true>}) {
        if (const cudaError_t e = cudaFuncGetAttributes(&attr, ker); e != cudaSuccess) {
            return static_cast<int>(e);
        }
    }
    return 0;
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught; any other value = the cudaError_t
// cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the runtime's last-error
// state.
//
// `m` non-zero, `top_k` in 1..=32, `hidden` a non-zero multiple of 256
// (`step::MOE_COMBINE_BF16_CHUNK`), `down_rows` at least `m · top_k`, and `m · top_k` and
// `down_rows · hidden` at most INT_MAX; all are checked before any pointer is touched. `down` is
// `[down_rows, hidden]` bf16, `pos` `[m · top_k]` u32 (each below `down_rows`; null when not
// `sorted`), `scores` `[m, top_k]` f32 and `out` `[m, hidden]` bf16, overlapping no input. No
// shared memory, so its one-time setup (`_prepare`, in `PREPARE`) only loads it. The launch is
// asynchronous on `stream`.
extern "C" int scratchy_cuda_moe_combine_bf16(const void* down, const void* pos,
                                              const void* scores, void* out, uint32_t m,
                                              uint32_t top_k, uint32_t hidden, uint32_t down_rows,
                                              uint32_t sorted,
                                              void* stream /* cudaStream_t; null = default */) {
    const uint64_t pairs = uint64_t(m) * uint64_t(top_k);
    if (m == 0 || top_k == 0 || top_k > MAX_TOP_K || hidden == 0 || hidden % CHUNK != 0 ||
        uint64_t(down_rows) < pairs || pairs > uint64_t(INT_MAX) ||
        uint64_t(down_rows) * uint64_t(hidden) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        // No view carries a TMA type, so building them encodes nothing. A gathered launch has
        // no `pos`: its view is the scores' pointer, never read.
        const void* pos_or_any = sorted != 0 ? pos : scores;
        row_gl dg{static_cast<bf16*>(const_cast<void*>(down)), nullptr, nullptr,
                  size_t(down_rows), size_t(hidden)};
        idx_gl pg{static_cast<int*>(const_cast<void*>(pos_or_any)), nullptr, nullptr, size_t(1),
                  size_t(pairs)};
        score_gl sg{static_cast<float*>(const_cast<void*>(scores)), nullptr, nullptr, size_t(m),
                    size_t(top_k)};
        row_gl og{static_cast<bf16*>(out), nullptr, nullptr, size_t(m), size_t(hidden)};
        combine_globals G{dg, pg, sg, og, int(m), int(top_k), int(hidden / uint32_t(CHUNK)),
                          sorted != 0};
        void* args[] = {&G};
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        if (m == 1) {
            // One warp a CTA, one CTA a chunk.
            const dim3 grid{hidden / uint32_t(CHUNK)}, block{uint32_t(kittens::WARP_THREADS)};
            return static_cast<int>(
                cudaLaunchKernel(moe_combine_bf16_ker<true>, grid, block, args, 0, s));
        }
        const dim3 grid((m + uint32_t(WARPS) - 1) / uint32_t(WARPS)), block(NUM_THREADS);
        return static_cast<int>(
            cudaLaunchKernel(moe_combine_bf16_ker<false>, grid, block, args, 0, s));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
