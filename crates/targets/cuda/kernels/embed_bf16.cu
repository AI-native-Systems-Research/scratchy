// SPDX-License-Identifier: Apache-2.0
//
// bf16 token-embedding gather for the H100: out[r, :] = table[tokens[r], :] for every one of the
// bucket's `rows`, with tokens [rows] u32, table [vocab, hidden] and out [rows, hidden], all
// row-major. A plain copy: each element is moved as raw 16-byte vectors, never widened, so the
// result is the table's bits.
//
// Plain CUDA, not ThunderKittens: a row gather is no tile op, and ThunderKittens has no gather.
// Metal's counterpart is `shaders/embed.metal` (one row per token, a plain copy).
//
// One 128-thread block per row. Every row is written, padding rows included, so no unwritten
// arena state reaches the first norm or GEMM. An id at or above `vocab` writes a +0.0 row and
// reads nothing from the table: the worker refuses such an id before any copy, so this is defence
// in depth, never a path a checked forward takes.

#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

// The launcher's return code that is not a `cudaError_t` (which is non-negative). `src/kernels.rs`
// mirrors it.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)

namespace {

// bf16 values per 16-byte vector: every load and store is one `uint4`.
constexpr int VEC = 8;
// Threads per block, one block per row.
constexpr int THREADS = 128;

__global__ void __launch_bounds__(THREADS)
embed_bf16_kernel(const uint32_t* __restrict__ tokens, const uint4* __restrict__ table,
                  uint4* __restrict__ out, int hidden_vecs, uint32_t vocab) {
    const size_t row = blockIdx.x;
    const uint32_t id = tokens[row];
    uint4* dst = out + row * size_t(hidden_vecs);
    if (id >= vocab) {
        const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
        for (int v = threadIdx.x; v < hidden_vecs; v += THREADS) {
            dst[v] = zero;
        }
        return;
    }
    const uint4* src = table + size_t(id) * size_t(hidden_vecs);
    for (int v = threadIdx.x; v < hidden_vecs; v += THREADS) {
        dst[v] = src[v];
    }
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched; any other value = the
// cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `rows`, `hidden` and `vocab` must be non-zero and at most INT_MAX, `hidden` a multiple of 8
// (every row then starts 16-byte aligned and is whole vectors), and `rows · hidden` at most
// INT_MAX; all are checked before any pointer is touched. `tokens` must be a 4-byte-aligned device
// pointer to `rows` u32; `table` and `out` 16-byte-aligned device pointers to `vocab·hidden` and
// `rows·hidden` bf16, `out` overlapping neither input. No shared memory, so no one-time setup.
// The launch is asynchronous on `stream`; the caller owns it and synchronizes.
extern "C" int scratchy_cuda_embed_bf16(const void* tokens, const void* table, void* out,
                                        uint32_t rows, uint32_t hidden, uint32_t vocab,
                                        void* stream /* cudaStream_t; null = default stream */) {
    if (rows == 0 || hidden == 0 || vocab == 0 || hidden % VEC != 0 ||
        rows > uint32_t(INT_MAX) || hidden > uint32_t(INT_MAX) || vocab > uint32_t(INT_MAX) ||
        uint64_t(rows) * uint64_t(hidden) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const uint32_t* tp = static_cast<const uint32_t*>(tokens);
    const uint4* wp = static_cast<const uint4*>(table);
    uint4* op = static_cast<uint4*>(out);
    int hidden_vecs = static_cast<int>(hidden / VEC);
    void* args[] = {&tp, &wp, &op, &hidden_vecs, &vocab};
    const dim3 grid(rows), block(THREADS);
    return static_cast<int>(cudaLaunchKernel(embed_bf16_kernel, grid, block, args, 0,
                                             static_cast<cudaStream_t>(stream)));
}
