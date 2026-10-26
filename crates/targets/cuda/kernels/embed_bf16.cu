// SPDX-License-Identifier: Apache-2.0
//
// bf16 token-embedding gather for the H100: out[r, :] = table[tokens[r], :] for every one of the
// bucket's `rows`, with tokens [rows] u32, table [vocab, hidden] and out [rows, hidden], all
// row-major. A plain copy: the bytes are moved, never widened, so the result is the table's bits.
// Metal's counterpart is `shaders/embed.metal` (one row per token, a plain copy).
//
// ThunderKittens' bulk copy through shared memory (PLAN-tk100 §2 S3). TK has no gather, so a
// `gl<int>` index read plus the bulk copy is its TK form. One warp per output row, 4 warps per
// CTA. Every lane reads the row's id from the `gl<int>` token view (`gl::operator[]`; a u32 above
// 2³¹ reads as negative). For an id in `[0, vocab)`, lane 0 bulk-loads the table row into the
// warp's `sv_bf<4096>` (`tma::load_async`, descriptor-free, `tma::expect_bytes` on the warp's
// semaphore), the warp waits, and lane 0 bulk-stores it to the output row (`tma::store_async`).
// Rows wider than 4096 elements move in 4096-element chunks; the next chunk's load waits until
// the previous store has read the buffer (`tma::store_async_read_wait`).
//
// Every row is written, padding rows included, so no unwritten arena state reaches the first norm
// or GEMM. An id outside `[0, vocab)` writes a +0.0 row (`warp::zero` on the buffer, then the bulk
// store) and reads nothing from the table: the worker refuses such an id before any copy, so this
// is defence in depth, never a path a checked forward takes. A bulk store writes exactly `hidden ·
// 2` bytes, so the output's tail past `rows · hidden` is never written.
//
// Shared memory: 4 warps × 8 KiB of buffer plus a semaphore each, static (under the 48 KiB
// default), so no one-time setup.

#include "kittens.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// The launcher's return codes that are not a `cudaError_t` (which is non-negative).
// `src/kernels.rs` mirrors them.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::EMBED_BF16_COLS`: `hidden` is a multiple of it, so every row is a whole number of 16-byte
// units and starts 16-byte aligned, as the bulk copy needs.
constexpr uint32_t VEC = 8;
// The bf16 elements one bulk copy moves at most: one 8 KiB buffer.
constexpr int CHUNK = 4096;
constexpr int WARPS = 4;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;

using token_gl = gl<int, 1, 1, 1, -1>;
using row_gl = gl<bf16, 1, 1, -1, -1>;

struct embed_globals {
    token_gl tokens;
    row_gl table;
    row_gl out;
    int rows;
    int hidden;
    int vocab;
};

__global__ __launch_bounds__(NUM_THREADS, 1) void embed_bf16_ker(const __grid_constant__ embed_globals g) {
    __shared__ sv_bf<CHUNK> buf[WARPS];
    __shared__ kittens::semaphore arrived[WARPS];

    const int w = kittens::warpid();
    const int row = blockIdx.x * WARPS + w;
    if (row >= g.rows) {
        return;
    }
    const int id = g.tokens[{0, 0, 0, row}];
    // `hidden` is at most INT_MAX, so the chunks are counted without forming `c0 + CHUNK`.
    const int chunks = (g.hidden - 1) / CHUNK + 1;
    if (id < 0 || id >= g.vocab) {
        warp::zero(buf[w]);
        __syncwarp();
        for (int k = 0; k < chunks; k++) {
            const int c0 = k * CHUNK;
            const uint32_t bytes = uint32_t(min(CHUNK, g.hidden - c0)) * sizeof(bf16);
            warp::tma::store_async(g.out[{0, 0, row, c0}], buf[w][0], bytes);
        }
        warp::tma::store_async_wait();
        return;
    }
    if (warp::laneid() == 0) {
        init_semaphore(arrived[w], 0, 1);
    }
    __syncwarp();
    for (int k = 0; k < chunks; k++) {
        const int c0 = k * CHUNK;
        const uint32_t bytes = uint32_t(min(CHUNK, g.hidden - c0)) * sizeof(bf16);
        // The previous chunk's store must have read the buffer before this load overwrites it.
        warp::tma::store_async_read_wait();
        __syncwarp();
        warp::tma::expect_bytes(arrived[w], bytes);
        warp::tma::load_async(buf[w][0], g.table[{0, 0, id, c0}], bytes, arrived[w]);
        wait(arrived[w], k % 2);
        warp::tma::store_async(g.out[{0, 0, row, c0}], buf[w][0], bytes);
    }
    warp::tma::store_async_wait();
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught; any other value = the cudaError_t
// cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the runtime's last-error
// state.
//
// `rows`, `hidden` and `vocab` must be non-zero and at most INT_MAX, `hidden` a multiple of 8
// (every row then starts 16-byte aligned and is whole 16-byte units), and `rows · hidden` at most
// INT_MAX; all are checked before any pointer is touched. `tokens` must be a 4-byte-aligned device
// pointer to `rows` u32; `table` and `out` 16-byte-aligned device pointers to `vocab·hidden` and
// `rows·hidden` bf16, `out` overlapping neither input. Static shared memory only, so no one-time
// setup. The launch is asynchronous on `stream`; the caller owns it and synchronizes.
extern "C" int scratchy_cuda_embed_bf16(const void* tokens, const void* table, void* out,
                                        uint32_t rows, uint32_t hidden, uint32_t vocab,
                                        void* stream /* cudaStream_t; null = default stream */) {
    if (rows == 0 || hidden == 0 || vocab == 0 || hidden % VEC != 0 ||
        rows > uint32_t(INT_MAX) || hidden > uint32_t(INT_MAX) || vocab > uint32_t(INT_MAX) ||
        uint64_t(rows) * uint64_t(hidden) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        int* tp = static_cast<int*>(const_cast<void*>(tokens));
        bf16* wp = static_cast<bf16*>(const_cast<void*>(table));
        bf16* op = static_cast<bf16*>(out);
        // No view carries a TMA type, so building them encodes nothing: they are the pointers
        // and the dimensions the kernel indexes with.
        token_gl tg{tp, nullptr, nullptr, nullptr, size_t(rows)};
        row_gl wg{wp, nullptr, nullptr, size_t(vocab), size_t(hidden)};
        row_gl og{op, nullptr, nullptr, size_t(rows), size_t(hidden)};
        embed_globals G{tg, wg, og, int(rows), int(hidden), int(vocab)};
        void* args[] = {&G};
        const dim3 grid((rows + uint32_t(WARPS) - 1) / uint32_t(WARPS)), block(NUM_THREADS);
        return static_cast<int>(cudaLaunchKernel(embed_bf16_ker, grid, block, args, 0,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
