// SPDX-License-Identifier: Apache-2.0
//
// The sampled row of a prefill's result matmul for the H100 (FP8 S8 F2, `step::GatherLastRow`):
// row `q_len − 1` of `x [rows, cols]` bf16 copied over row 0, in place, `q_len` entry 1 of the KV
// sequence region (`[prefix_len, q_len]` u32, written by the forward before the graph runs). The
// lm_head then runs over one 128-row tile, and the sequence's last logits row is its row 0. A
// plain copy: the bytes are moved, never widened, so row 0 is row `q_len − 1`'s bits.
//
// `embed_bf16.cu`'s TK form: a `gl<int>` index read, then the bulk copy through shared memory. One
// warp per CTA, CTA `c` the row's `c`-th 4096-element chunk: every lane reads `q_len` from the
// `gl<int>` sequence view (`gl::operator[]`), lane 0 bulk-loads the chunk of row `q_len − 1` into
// the warp's `sv_bf<4096>` (`tma::load_async`, `tma::expect_bytes` on its semaphore), the warp
// waits, and lane 0 bulk-stores it over the chunk of row 0 (`tma::store_async`). The chunks are
// disjoint columns and the two rows differ, so no CTA reads what another writes. A `q_len` of 0
// or 1 (the last row is row 0 already) or above `rows` copies nothing: the worker never uploads
// one above the bucket's rows.
//
// Shared memory: an 8 KiB buffer and a semaphore, static (under the 48 KiB default), so its
// one-time setup (`_prepare`, in `PREPARE`) only loads it.

#include "kittens.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// The launcher's return codes that are not a `cudaError_t` (which is non-negative).
// `src/kernels.rs` mirrors them.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::GATHER_LAST_ROW_BF16_COLS`: `cols` is a multiple of it, so every row is a whole number
// of 16-byte units and starts 16-byte aligned, as the bulk copy needs.
constexpr uint32_t VEC = 8;
// The bf16 elements one bulk copy moves at most: one 8 KiB buffer.
constexpr int CHUNK = 4096;

using seq_gl = gl<int, 1, 1, 1, -1>;
using row_gl = gl<bf16, 1, 1, -1, -1>;

struct gather_globals {
    seq_gl sequence;
    row_gl x;
    int rows;
    int cols;
};

__global__ __launch_bounds__(kittens::WARP_THREADS, 1) void gather_last_row_bf16_ker(
    const __grid_constant__ gather_globals g) {
    __shared__ sv_bf<CHUNK> buf;
    __shared__ kittens::semaphore arrived;

    const int src = g.sequence[{0, 0, 0, 1}] - 1;
    if (src <= 0 || src >= g.rows) {
        return;
    }
    const int c0 = int(blockIdx.x) * CHUNK;
    const uint32_t bytes = uint32_t(min(CHUNK, g.cols - c0)) * sizeof(bf16);
    if (warp::laneid() == 0) {
        init_semaphore(arrived, 0, 1);
    }
    __syncwarp();
    warp::tma::expect_bytes(arrived, bytes);
    warp::tma::load_async(buf[0], g.x[{0, 0, src, c0}], bytes, arrived);
    wait(arrived, 0);
    warp::tma::store_async(g.x[{0, 0, 0, c0}], buf[0], bytes);
    warp::tma::store_async_wait();
}

}  // namespace

// The kernel's one-time setup, run once per device by `kernels::prepare_all` (`CudaDevice::open`):
// query its attributes (`cudaFuncGetAttributes`), which loads its module, so a worker's graph
// capture never loads a kernel mid-capture. It changes nothing on the device.
//
// 0 = done; any other value = the cudaError_t of the query.
extern "C" int scratchy_cuda_gather_last_row_bf16_prepare(void) {
    cudaFuncAttributes attr;
    return static_cast<int>(cudaFuncGetAttributes(&attr, gather_last_row_bf16_ker));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught; any other value = the cudaError_t
// cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the runtime's last-error
// state.
//
// `rows` and `cols` must be non-zero, `cols` a multiple of 8, and `rows · cols` at most INT_MAX;
// all are checked before any pointer is touched. `x` must be a 16-byte-aligned device pointer to
// `rows·cols` bf16 and `sequence` a 4-byte-aligned one to 2 u32, the `[prefix_len, q_len]` of the
// forward. Static shared memory only. The launch is asynchronous on `stream`; the caller owns it
// and synchronizes.
extern "C" int scratchy_cuda_gather_last_row_bf16(void* x, const void* sequence, uint32_t rows,
                                                  uint32_t cols,
                                                  void* stream /* cudaStream_t; null = default */) {
    if (rows == 0 || cols == 0 || cols % VEC != 0 ||
        uint64_t(rows) * uint64_t(cols) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        // No view carries a TMA type, so building them encodes nothing.
        seq_gl sg{static_cast<int*>(const_cast<void*>(sequence)), nullptr, nullptr, nullptr,
                  size_t(2)};
        row_gl xg{static_cast<bf16*>(x), nullptr, nullptr, size_t(rows), size_t(cols)};
        gather_globals G{sg, xg, int(rows), int(cols)};
        void* args[] = {&G};
        const dim3 grid((cols + uint32_t(CHUNK) - 1) / uint32_t(CHUNK)),
            block(kittens::WARP_THREADS);
        return static_cast<int>(cudaLaunchKernel(gather_last_row_bf16_ker, grid, block, args, 0,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
