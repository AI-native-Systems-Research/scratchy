// SPDX-License-Identifier: Apache-2.0
//
// The greedy token of one bf16 logits row on the H100 (FP8 S8 W7): the worker's `greedy_token`
// (`worker/src/plan.rs`) on the device, captured into a one-row tape's graph after its last step,
// so a decode copies 4 bytes back, not the row. The same answer, bit for bit:
//   - the argmax, the LOWEST id among the logits equal to the maximum (bf16 widens to f32 exactly,
//     so `x == M` is the host's `v > best` tie rule: −0 and +0 are equal, as there);
//   - a row holding a NaN is refused: the output is `−1 − i`, `i` the lowest NaN id, which the host
//     turns into the same `PlanError::NanLogit { id: i }` (the host loop stops at its first NaN).
//
// One CTA of 8 warps, two passes over the row in chunks of 256 logits, chunk `c` by warp `c % 8`:
//   1. each warp's running max (`warp::max`, which skips a NaN) and its NaN flag (`nan_flag`);
//      both reduced across the warps through shared memory, in warp order;
//   2. each logit's candidate index (`index_if` of `equals_key`, its id when it equals the max, or
//      of `nan_flag` for a row holding a NaN; else +inf), the ids an f32 iota of one
//      chunk (`iota`, 0 .. 256, the caller's) plus the chunk's base, exact below 2^24; the running
//      min, reduced across the warps the same way.
// The answer is a min over the asked set's ids, so it does not depend on the order of the warps.
// No atomics.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::ARGMAX_CHUNK`: the logits one warp reads at a time, and the iota's length.
constexpr int CHUNK = 256;
constexpr int WARPS = 8;
constexpr int THREADS = WARPS * kittens::WARP_THREADS;
// The largest row whose ids are exact f32s.
constexpr uint32_t MAX_VOCAB = 1u << 24;

// A chunk, naive: each logit in one lane (8 a lane), widened from bf16 by TK's load.
using chunk_rv = rv_fl<CHUNK>;
using part_sv = sv_fl<16>;
using part_rv = rv_fl<16>;

using row_gl = gl<bf16, 1, 1, 1, -1>;
using iota_gl = gl<float, 1, 1, 1, CHUNK>;
using out_gl = gl<int, 1, 1, 1, 1>;

struct argmax_globals {
    row_gl row;
    iota_gl iota;
    out_gl out;
    int chunks;
};

// This warp's `v` into slots `w` and `w + 8` of `part` (16 slots, 8 warps), then, after the CTA's
// barrier, `REDUCE` of the 16 in every warp.
template <bool MAX>
__device__ inline float across_warps(part_sv& part, float v) {
    const int w = kittens::warpid();
    if (warp::laneid() == 0) {
        part[w] = v;
        part[w + WARPS] = v;
    }
    __syncthreads();
    part_rv all;
    warp::load(all, part);
    if constexpr (MAX) {
        return warp::max(all);
    } else {
        return warp::min(all);
    }
}

__global__ __launch_bounds__(THREADS, 1) void argmax_bf16_ker(const __grid_constant__ argmax_globals g) {
    __shared__ sv_fl<16> max_part, nan_part, idx_part;
    const int w = kittens::warpid();
    chunk_rv x, run, flag;

    // Pass 1: the max, and whether the row holds a NaN.
    warp::neg_infty(run);
    warp::zero(flag);
    for (int c = w; c < g.chunks; c += WARPS) {
        warp::load(x, g.row, {0, 0, 0, c});
        warp::max(run, run, x);
        warp::unary_op<scratchy::ops::nan_flag>(x, x);
        warp::max(flag, flag, x);
    }
    const float row_max = across_warps<true>(max_part, warp::max(run));
    const bool has_nan = across_warps<true>(nan_part, warp::max(flag)) > 0.0f;

    // Pass 2: the lowest id of the asked set.
    chunk_rv iota, idx;
    warp::load(iota, g.iota, {0, 0, 0, 0});
    warp::pos_infty(run);
    for (int c = w; c < g.chunks; c += WARPS) {
        warp::load(x, g.row, {0, 0, 0, c});
        warp::add(idx, iota, float(c * CHUNK));
        // `has_nan` is the CTA's, so every warp takes the same branch.
        if (has_nan) {
            warp::unary_op<scratchy::ops::nan_flag>(x, x);
        } else {
            warp::bin_op<scratchy::ops::equals_key>(x, x, row_max);
        }
        warp::bin_op<scratchy::ops::index_if>(x, x, idx);
        warp::min(run, run, x);
    }
    const int first = int(across_warps<false>(idx_part, warp::min(run)));
    if (threadIdx.x == 0) {
        g.out[{0, 0, 0, 0}] = has_nan ? -1 - first : first;
    }
}

}  // namespace

// The greedy token of `row` (`vocab` bf16 logits) into `out` (one int): its argmax, the lowest id on
// a tie; `−1 − i` when the row holds a NaN, `i` the lowest NaN id. `iota` holds the f32s `0 .. 256`.
//
// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = refused, nothing launched: `vocab` zero, not a multiple of
// 256 (`step::ARGMAX_CHUNK`) or above 2^24, or a null pointer; SCRATCHY_CUDA_ERR_HOST = a C++
// exception was caught; any other value = the cudaError_t of the launch. Asynchronous on `stream`;
// every pointer a device pointer to its elements, `out` overlapping no input.
extern "C" int scratchy_cuda_argmax_bf16(const void* row, uint32_t vocab, const void* iota,
                                         void* out,
                                         void* stream /* cudaStream_t; null = default */) {
    if (vocab == 0 || vocab % uint32_t(CHUNK) != 0 || vocab > MAX_VOCAB || row == nullptr ||
        iota == nullptr || out == nullptr) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        row_gl rg{static_cast<bf16*>(const_cast<void*>(row)), nullptr, nullptr, nullptr,
                  size_t(vocab)};
        iota_gl ig{static_cast<float*>(const_cast<void*>(iota)), nullptr, nullptr, nullptr,
                   nullptr};
        out_gl og{static_cast<int*>(out), nullptr, nullptr, nullptr, nullptr};
        argmax_globals G{rg, ig, og, int(vocab / uint32_t(CHUNK))};
        void* args[] = {&G};
        const dim3 grid{1u}, block{uint32_t(THREADS)};
        return static_cast<int>(cudaLaunchKernel(argmax_bf16_ker, grid, block, args, 0,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
