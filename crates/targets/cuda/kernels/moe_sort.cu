// SPDX-License-Identifier: Apache-2.0
//
// The qwen3-5-moe block's sort for the H100 (MoE step M2 of #857): the `P = m · top_k` (token,
// pick) pairs, pair `p` naming expert `topk_idx[p]`, sorted by expert, stably (equal experts keep
// pair order). Each expert's run starts at `offsets[e]` and is padded to a whole number of the
// grouped GEMM's 128-row tiles, so a tile belongs to one expert. Out:
//   - `rows[pos[p], :] = x[p / top_k, :]` (the pair's token row, its bits copied), and every
//     padding row of every live tile `+0.0`; no row outside a live tile is written;
//   - `counts[e]` (the expert's pairs), `offsets[0..=256]` (the padded exclusive scan of
//     `counts`), `tile_expert[t]` for every live tile `t < offsets[256] / 128`, and `pos[p]`;
//   - `chunk_hist [⌈P / 256⌉, 256]`, scratch: the chunks' counts, then their bases.
// The region layout is `step::SortLayout` / `step::sort_index_regions` (#841).
//
// A stable counting sort in three stream-ordered launches, with no atomics, shuffles or ballots
// (PLAN-tk100 §3.M M2): where order matters the work is serial, so the result is the same bits on
// every run.
//   (a) One warp per chunk of 256 pairs (`step::MOE_SORT_CHUNK`). The counters are a shared
//       `sv<int, 256>`, zeroed by `warp::zero`. Lane 0 walks the chunk's pairs in pair order:
//       `pos[p] = cnt[e]` (the pair's rank in its chunk, scratch until (c)), then `cnt[e] += 1`,
//       through `sv::operator[]` and `gl::operator[]`. The counters go to `chunk_hist`'s row by
//       `warp::store`.
//   (b) One warp. For each chunk in order, its counts row is loaded (`warp::load`, an
//       `rv<int, 256>`), the running sum stored over it as the chunk's base, and the row added in
//       (`warp::add`). The last sum is `counts`. Moved into a shared `sv<int, 256>`, it is scanned
//       by the `TK-GAP(exclusive_scan_sv)` helper of `scratchy_ops.cuh` (one lane, serial, each
//       count padded by `scratchy::ops::tile_pad`), giving `offsets[0..256)` and the total,
//       `offsets[256]`. Lane 0 then writes `tile_expert` for the live tiles, expert by expert.
//   (c) One warp per pair, 8 warps per CTA: `pos[p] = offsets[e] + base[c][e] + rank[p]` (integer
//       scalars), and the token row bulk-copied into `rows[pos[p]]` through the warp's
//       `sv_bf<2048>` (PLAN-tk100 §2 S3, as `embed_bf16.cu`), bit-exact, NaN payloads included.
//       Past the pairs' CTAs, one warp per expert writes its run's padding rows from a zeroed
//       buffer (`warp::zero`, then bulk stores).
// The points where one warp's TK operations meet on one shared vector, each a `__syncwarp()`
// (the TK-4 lesson): (a) the zero before lane 0's counting, and the counting before the store;
// (b) the register-to-shared store before the scan, and the scan before the offsets' store;
// (c) the zero before the padding's bulk stores, and each copy buffer between the previous
// store's `store_async_read_wait` and the next load.
//
// Shared memory: at most 8 warps × 4 KiB of buffer plus a semaphore each, static (under the
// 48 KiB default), so no one-time setup.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::MOE_ROUTE_EXPERTS`: the one expert count, one counter per expert.
constexpr int EXPERTS = 256;
// `step::MOE_SORT_CHUNK`: pairs per `chunk_hist` row, one warp of (a) each.
constexpr int CHUNK = 256;
// `step::MOE_ROUTE_MAX_TOP_K`.
constexpr uint32_t MAX_TOP_K = 32;
// `op_abi::CUDA_EXPERT_TILE_ROWS`: the grouped GEMM's row tile, the runs' padding.
constexpr int TILE_ROWS = 128;
static_assert(scratchy::ops::tile_pad::ROWS == TILE_ROWS, "the scan pads to the GEMM's tile");
// `step::EMBED_BF16_COLS`'s rule: `hidden` a multiple of 8 bf16, so a row is whole 16-byte units
// and starts 16-byte aligned, as the bulk copy needs.
constexpr uint32_t VEC = 8;
// The bf16 elements one bulk copy moves at most: one 4 KiB buffer, qwen3-5-moe's whole row.
constexpr int ROW_CHUNK = 2048;
constexpr int WARPS = 8;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;

using count_rv = rv<int, EXPERTS>;
using idx_gl = gl<int, 1, 1, -1, -1>;
using row_gl = gl<bf16, 1, 1, -1, -1>;

struct sort_globals {
    // `[m, hidden]`.
    row_gl x;
    // `[1, P]`.
    idx_gl topk_idx;
    // `[chunks, 256]`.
    idx_gl chunk_hist;
    // `[1, 256]`.
    idx_gl counts;
    // `[1, 257]`.
    idx_gl offsets;
    // `[1, max_tiles]`.
    idx_gl tile_expert;
    // `[1, P]`.
    idx_gl pos;
    // `[max_tiles · 128, hidden]`.
    row_gl rows;
    int pairs;
    int chunks;
    int top_k;
    int hidden;
    // The CTAs of (c) that copy pairs; the ones after them write the padding.
    int pair_ctas;
};

// (a): one warp per chunk.
__global__ __launch_bounds__(kittens::WARP_THREADS) void moe_sort_count_ker(
    const __grid_constant__ sort_globals g) {
    __shared__ sv_int<EXPERTS> cnt;
    const int c = blockIdx.x;
    warp::zero(cnt);
    __syncwarp();
    if (warp::laneid() == 0) {
        const int first = c * CHUNK;
        const int last = min(first + CHUNK, g.pairs);
        for (int p = first; p < last; ++p) {
            const int e = g.topk_idx[{0, 0, 0, p}];
            g.pos[{0, 0, 0, p}] = cnt[e];
            cnt[e] += 1;
        }
    }
    __syncwarp();
    warp::store(g.chunk_hist, cnt, {0, 0, c, 0});
}

// (b): one warp.
__global__ __launch_bounds__(kittens::WARP_THREADS) void moe_sort_offsets_ker(
    const __grid_constant__ sort_globals g) {
    __shared__ sv_int<EXPERTS> scan;
    count_rv run, h;
    warp::zero(run);
    for (int c = 0; c < g.chunks; ++c) {
        // Each lane loads and stores the same elements of the row, so the store over the row it
        // just read needs no barrier.
        warp::load(h, g.chunk_hist, {0, 0, c, 0});
        warp::store(g.chunk_hist, run, {0, 0, c, 0});
        warp::add(run, run, h);
    }
    warp::store(g.counts, run, {0, 0, 0, 0});
    warp::store(scan, run);
    __syncwarp();
    if (warp::laneid() == 0) {
        const int total = scratchy::ops::exclusive_scan_sv<scratchy::ops::tile_pad>(scan);
        g.offsets[{0, 0, 0, EXPERTS}] = total;
        for (int e = 0; e < EXPERTS; ++e) {
            const int end = e + 1 < EXPERTS ? scan[e + 1] : total;
            for (int t = scan[e] / TILE_ROWS; t < end / TILE_ROWS; ++t) {
                g.tile_expert[{0, 0, 0, t}] = e;
            }
        }
    }
    __syncwarp();
    warp::store(g.offsets, scan, {0, 0, 0, 0});
}

// (c): one warp per pair, then one warp per expert's padding.
__global__ __launch_bounds__(NUM_THREADS, 1) void moe_sort_scatter_ker(
    const __grid_constant__ sort_globals g) {
    __shared__ sv_bf<ROW_CHUNK> buf[WARPS];
    __shared__ kittens::semaphore arrived[WARPS];

    const int w = kittens::warpid();
    // `hidden` is at most INT_MAX, so the chunks are counted without forming `c0 + ROW_CHUNK`.
    const int row_chunks = (g.hidden - 1) / ROW_CHUNK + 1;
    if (int(blockIdx.x) >= g.pair_ctas) {
        const int e = (int(blockIdx.x) - g.pair_ctas) * WARPS + w;
        const int first = g.offsets[{0, 0, 0, e}] + g.counts[{0, 0, 0, e}];
        const int last = g.offsets[{0, 0, 0, e + 1}];
        if (first >= last) {
            return;
        }
        warp::zero(buf[w]);
        __syncwarp();
        for (int r = first; r < last; ++r) {
            for (int k = 0; k < row_chunks; ++k) {
                const int c0 = k * ROW_CHUNK;
                const uint32_t bytes = uint32_t(min(ROW_CHUNK, g.hidden - c0)) * sizeof(bf16);
                warp::tma::store_async(g.rows[{0, 0, r, c0}], buf[w][0], bytes);
            }
        }
        warp::tma::store_async_wait();
        return;
    }
    const int p = int(blockIdx.x) * WARPS + w;
    if (p >= g.pairs) {
        return;
    }
    const int e = g.topk_idx[{0, 0, 0, p}];
    const int row = g.offsets[{0, 0, 0, e}] + g.chunk_hist[{0, 0, p / CHUNK, e}] +
                    g.pos[{0, 0, 0, p}];
    // Every lane has read the rank before lane 0 overwrites it.
    __syncwarp();
    if (warp::laneid() == 0) {
        g.pos[{0, 0, 0, p}] = row;
        init_semaphore(arrived[w], 0, 1);
    }
    __syncwarp();
    const int token = p / g.top_k;
    for (int k = 0; k < row_chunks; k++) {
        const int c0 = k * ROW_CHUNK;
        const uint32_t bytes = uint32_t(min(ROW_CHUNK, g.hidden - c0)) * sizeof(bf16);
        // The previous chunk's store must have read the buffer before this load overwrites it.
        warp::tma::store_async_read_wait();
        __syncwarp();
        warp::tma::expect_bytes(arrived[w], bytes);
        warp::tma::load_async(buf[w][0], g.x[{0, 0, token, c0}], bytes, arrived[w]);
        wait(arrived[w], k % 2);
        warp::tma::store_async(g.rows[{0, 0, row, c0}], buf[w][0], bytes);
    }
    warp::tma::store_async_wait();
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught; any other value = the cudaError_t
// cudaLaunchKernel returned for the first of the three launches that failed (the ones after it
// are not launched). Nothing here reads or clears the runtime's last-error state.
//
// `m` non-zero, `experts` 256 (`step::MOE_ROUTE_EXPERTS`), `top_k` in 1..=32
// (`step::MOE_ROUTE_MAX_TOP_K`), `hidden` a non-zero multiple of 8, `m · top_k` and
// `m · hidden` at most INT_MAX, and `max_tiles` at least `tape::moe_tiles`' bound
// `(P + 127 · min(256, P)) / 128` with `max_tiles · 128 · hidden` at most INT_MAX; all are
// checked before any pointer is touched. The bound is the most tiles any routing pads to, so every
// row and tile index the kernels form is in its region. `x` is `[m, hidden]` bf16 and `rows`
// `[max_tiles · 128, hidden]` bf16, 16-byte aligned; `topk_idx` `[P]` i32, each in `[0, 256)`;
// `chunk_hist` `[⌈P / 256⌉, 256]`, `counts` `[256]`, `offsets` `[257]`, `tile_expert`
// `[max_tiles]` and `pos` `[P]` u32, each 16-byte aligned; no output overlaps another or an input.
// Static shared memory only, so no one-time setup. The launches are asynchronous on `stream`, in
// order.
extern "C" int scratchy_cuda_moe_sort(const void* x, const void* topk_idx, void* chunk_hist,
                                      void* counts, void* offsets, void* tile_expert, void* pos,
                                      void* rows, uint32_t m, uint32_t experts, uint32_t top_k,
                                      uint32_t hidden, uint32_t max_tiles,
                                      void* stream /* cudaStream_t; null = default */) {
    const uint64_t pairs = uint64_t(m) * uint64_t(top_k);
    const uint64_t runs = pairs < uint64_t(EXPERTS) ? pairs : uint64_t(EXPERTS);
    const uint64_t bound = (pairs + uint64_t(TILE_ROWS - 1) * runs) / uint64_t(TILE_ROWS);
    if (m == 0 || experts != uint32_t(EXPERTS) || top_k == 0 || top_k > MAX_TOP_K ||
        hidden == 0 || hidden % VEC != 0 || pairs > uint64_t(INT_MAX) ||
        uint64_t(m) * uint64_t(hidden) > uint64_t(INT_MAX) || uint64_t(max_tiles) < bound ||
        uint64_t(max_tiles) * uint64_t(TILE_ROWS) * uint64_t(hidden) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const uint32_t chunks = uint32_t((pairs + uint64_t(CHUNK) - 1) / uint64_t(CHUNK));
        const uint32_t pair_ctas = uint32_t((pairs + uint64_t(WARPS) - 1) / uint64_t(WARPS));
        // No view carries a TMA type, so building them encodes nothing.
        row_gl xg{static_cast<bf16*>(const_cast<void*>(x)), nullptr, nullptr, size_t(m),
                  size_t(hidden)};
        idx_gl ig{static_cast<int*>(const_cast<void*>(topk_idx)), nullptr, nullptr, size_t(1),
                  size_t(pairs)};
        idx_gl hg{static_cast<int*>(chunk_hist), nullptr, nullptr, size_t(chunks),
                  size_t(EXPERTS)};
        idx_gl cg{static_cast<int*>(counts), nullptr, nullptr, size_t(1), size_t(EXPERTS)};
        idx_gl og{static_cast<int*>(offsets), nullptr, nullptr, size_t(1), size_t(EXPERTS + 1)};
        idx_gl tg{static_cast<int*>(tile_expert), nullptr, nullptr, size_t(1), size_t(max_tiles)};
        idx_gl pg{static_cast<int*>(pos), nullptr, nullptr, size_t(1), size_t(pairs)};
        row_gl rg{static_cast<bf16*>(rows), nullptr, nullptr,
                  size_t(max_tiles) * size_t(TILE_ROWS), size_t(hidden)};
        sort_globals G{xg, ig, hg, cg, og, tg, pg, rg, int(pairs), int(chunks), int(top_k),
                       int(hidden), int(pair_ctas)};
        void* args[] = {&G};
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        const dim3 warp(kittens::WARP_THREADS);
        cudaError_t err = cudaLaunchKernel(moe_sort_count_ker, dim3(chunks), warp, args, 0, s);
        if (err != cudaSuccess) {
            return static_cast<int>(err);
        }
        err = cudaLaunchKernel(moe_sort_offsets_ker, dim3(1), warp, args, 0, s);
        if (err != cudaSuccess) {
            return static_cast<int>(err);
        }
        const dim3 grid(pair_ctas + uint32_t(EXPERTS / WARPS)), block(NUM_THREADS);
        return static_cast<int>(cudaLaunchKernel(moe_sort_scatter_ker, grid, block, args, 0, s));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
