// SPDX-License-Identifier: Apache-2.0
//
// The MoE block's sort for the H100 (MoE step M2 of #857): the `P = m · top_k` (token,
// pick) pairs, pair `p` naming expert `topk_idx[p]`, sorted by expert, stably (equal experts keep
// pair order). Each expert's run starts at `offsets[e]` and is padded to a whole number of the
// grouped GEMM's 128-row tiles, so a tile belongs to one expert. Out:
//   - `rows[pos[p], :] = x[p / top_k, :]` (the pair's token row, its bits copied), and every
//     padding row of every live tile `+0.0`; no row outside a live tile is written;
//   - `counts[e]` (the expert's pairs), `offsets[0..=E]` (the padded exclusive scan of
//     `counts`), `tile_expert[t]` for every live tile `t < offsets[E] / 128`, and `pos[p]`;
//   - `chunk_hist [⌈P / 256⌉, E]`, scratch: the chunks' counts, then their bases.
// The region layout is `step::SortLayout` / `step::sort_index_regions` (#841). `E` is the expert
// count, a template parameter instantiated at both of `op_abi::MOE_EXPERT_COUNTS`: 256
// (qwen3-5-moe) and 128 (Gemma 4 26B-A4B, #924). Nothing else depends on it.
//
// A stable counting sort in three stream-ordered launches, with no atomics, shuffles or ballots
// (PLAN-tk100 §3.M M2): where order matters the work is serial, so the result is the same bits on
// every run.
//   (a) One warp per chunk of 256 pairs (`step::MOE_SORT_CHUNK`). The counters are a shared
//       `sv<int, E>`, zeroed by `warp::zero`. Lane 0 walks the chunk's pairs in pair order:
//       `pos[p] = cnt[e]` (the pair's rank in its chunk, scratch until (c)), then `cnt[e] += 1`,
//       through `sv::operator[]` and `gl::operator[]`. The counters go to `chunk_hist`'s row by
//       `warp::store`.
//   (b) One warp. For each chunk in order, its counts row is loaded (`warp::load`, an
//       `rv<int, E>`), the running sum stored over it as the chunk's base, and the row added in
//       (`warp::add`). The last sum is `counts`. Moved into a shared `sv<int, E>`, it is scanned
//       by the `TK-GAP(exclusive_scan_sv)` helper of `scratchy_ops.cuh` (one lane, serial, each
//       count padded by `scratchy::ops::tile_pad`), giving `offsets[0..E)` and the total,
//       `offsets[E]`. Lane 0 then writes `tile_expert` for the live tiles, expert by expert.
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
// THE E4M3 SCATTER (FP8 S8 P1, `scratchy_cuda_moe_sort_fp8`): an fp8 bank's gate and up read
// e4m3 rows with per-row scales, `quant_rows_fp8`'s, which depend on the row alone. So the token
// rows are quantized once, before the sort (`xq [m, hidden]` e4m3, `x_scale [m]` f32), and (c)
// copies each pair's e4m3 row and scale instead of its bf16 row: `rows [capacity, hidden]` e4m3
// then `row_scale [capacity]` f32, the layout of `quant_rows_fp8` over the sorted rows. Each
// padding row of a live tile is +0 bytes with `fp8_scale(0)`, `quant_rows_fp8`'s of the +0.0 row
// today's sort writes; no row outside a live tile is written. (a) and (b) are the sort's own. The
// copies are TK's bulk copy through the warp's `sv_fp8e4m3<4096>`, the bytes moved as they are.
//
// Shared memory: at most 8 warps × 4 KiB of buffer plus a semaphore each, static (under the 48 KiB
// default), so its one-time setup (`_prepare`, in `PREPARE`) only loads it.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `op_abi::MOE_EXPERT_COUNTS`: the expert counts the sort is instantiated at, one counter per
// expert.
constexpr int EXPERTS_SMALL = 128;
constexpr int EXPERTS_LARGE = 256;
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

template <int EXPERTS>
using count_rv = rv<int, EXPERTS>;
using idx_gl = gl<int, 1, 1, -1, -1>;
using row_gl = gl<bf16, 1, 1, -1, -1>;

struct sort_globals {
    // `[m, hidden]`.
    row_gl x;
    // `[1, P]`.
    idx_gl topk_idx;
    // `[chunks, E]`.
    idx_gl chunk_hist;
    // `[1, E]`.
    idx_gl counts;
    // `[1, E + 1]`.
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
template <int EXPERTS>
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

// (b): one warp, each run padded to `TILE` rows.
template <int EXPERTS, int TILE = TILE_ROWS>
__global__ __launch_bounds__(kittens::WARP_THREADS) void moe_sort_offsets_ker(
    const __grid_constant__ sort_globals g) {
    __shared__ sv_int<EXPERTS> scan;
    count_rv<EXPERTS> run, h;
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
        const int total = scratchy::ops::exclusive_scan_sv<scratchy::ops::tile_pad_at<TILE>>(scan);
        g.offsets[{0, 0, 0, EXPERTS}] = total;
        for (int e = 0; e < EXPERTS; ++e) {
            const int end = e + 1 < EXPERTS ? scan[e + 1] : total;
            for (int t = scan[e] / TILE; t < end / TILE; ++t) {
                g.tile_expert[{0, 0, 0, t}] = e;
            }
        }
    }
    __syncwarp();
    warp::store(g.offsets, scan, {0, 0, 0, 0});
}

// (c): one warp per pair, then one warp per expert's padding. Only the grid depends on the expert
// count (`launch_sort`), so one instance serves both.
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

// The e4m3 scatter's views: the quantized token rows and their scales, the sort's index regions,
// and the sorted e4m3 rows and scales.
using q_gl = gl<fp8e4m3, 1, 1, -1, -1>;
using scale_gl = gl<float, 1, 1, 1, -1>;

struct sort_fp8_globals {
    // `[m, hidden]` e4m3, `[1, m]` f32.
    q_gl xq;
    scale_gl x_scale;
    idx_gl topk_idx;
    idx_gl chunk_hist;
    idx_gl counts;
    idx_gl offsets;
    idx_gl pos;
    // `[max_tiles · 128, hidden]` e4m3, `[1, max_tiles · 128]` f32.
    q_gl rows;
    scale_gl row_scale;
    int pairs;
    int top_k;
    int hidden;
    int pair_ctas;
};

// The e4m3 rows one bulk copy moves at most: the warp's 4 KiB e4m3 buffer.
constexpr int ROW_CHUNK_BYTES = ROW_CHUNK * int(sizeof(bf16));

// (c) on e4m3 rows (P1): one warp per pair, then one warp per expert's padding.
__global__ __launch_bounds__(NUM_THREADS, 1) void moe_sort_scatter_fp8_ker(
    const __grid_constant__ sort_fp8_globals g) {
    __shared__ sv_fp8e4m3<ROW_CHUNK_BYTES> buf[WARPS];
    __shared__ kittens::semaphore arrived[WARPS];

    const int w = kittens::warpid();
    const int row_chunks = (g.hidden - 1) / ROW_CHUNK_BYTES + 1;
    if (int(blockIdx.x) >= g.pair_ctas) {
        const int e = (int(blockIdx.x) - g.pair_ctas) * WARPS + w;
        const int first = g.offsets[{0, 0, 0, e}] + g.counts[{0, 0, 0, e}];
        const int last = g.offsets[{0, 0, 0, e + 1}];
        if (first >= last) {
            return;
        }
        // The zero row's bytes (+0) and scale (`fp8_scale(0)`).
        warp::zero(buf[w]);
        __syncwarp();
        const float zero_scale = scratchy::ops::fp8_scale::op<float>(0.0f);
        for (int r = first; r < last; ++r) {
            for (int k = 0; k < row_chunks; ++k) {
                const int c0 = k * ROW_CHUNK_BYTES;
                const uint32_t bytes = uint32_t(min(ROW_CHUNK_BYTES, g.hidden - c0));
                warp::tma::store_async(g.rows[{0, 0, r, c0}], buf[w][0], bytes);
            }
            if (warp::laneid() == 0) {
                g.row_scale[{0, 0, 0, r}] = zero_scale;
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
    const int token = p / g.top_k;
    if (warp::laneid() == 0) {
        g.pos[{0, 0, 0, p}] = row;
        g.row_scale[{0, 0, 0, row}] = g.x_scale[{0, 0, 0, token}];
        init_semaphore(arrived[w], 0, 1);
    }
    __syncwarp();
    for (int k = 0; k < row_chunks; k++) {
        const int c0 = k * ROW_CHUNK_BYTES;
        const uint32_t bytes = uint32_t(min(ROW_CHUNK_BYTES, g.hidden - c0));
        // The previous chunk's store must have read the buffer before this load overwrites it.
        warp::tma::store_async_read_wait();
        __syncwarp();
        warp::tma::expect_bytes(arrived[w], bytes);
        warp::tma::load_async(buf[w][0], g.xq[{0, 0, token, c0}], bytes, arrived[w]);
        wait(arrived[w], k % 2);
        warp::tma::store_async(g.rows[{0, 0, row, c0}], buf[w][0], bytes);
    }
    warp::tma::store_async_wait();
}

// The three launches at `EXPERTS`, the shape already checked: the counts, the offsets, then the
// scatter, whose grid is the pairs' CTAs plus one warp per expert's padding.
template <int EXPERTS>
int launch_sort(const sort_globals& G, uint32_t chunks, uint32_t pair_ctas, cudaStream_t s) {
    static_assert(EXPERTS % WARPS == 0, "the padding CTAs hold whole warps of experts");
    void* args[] = {const_cast<sort_globals*>(&G)};
    const dim3 warp(kittens::WARP_THREADS);
    cudaError_t err = cudaLaunchKernel(moe_sort_count_ker<EXPERTS>, dim3(chunks), warp, args, 0, s);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    err = cudaLaunchKernel(moe_sort_offsets_ker<EXPERTS>, dim3(1), warp, args, 0, s);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    const dim3 grid(pair_ctas + uint32_t(EXPERTS / WARPS)), block(NUM_THREADS);
    return static_cast<int>(cudaLaunchKernel(moe_sort_scatter_ker, grid, block, args, 0, s));
}

// The e4m3 sort at `EXPERTS` (P1): the sort's counts and offsets on `G` (runs padded to `TILE`
// rows), then the e4m3 scatter on `F`.
template <int EXPERTS, int TILE>
int launch_sort_fp8(const sort_globals& G, const sort_fp8_globals& F, uint32_t chunks,
                    uint32_t pair_ctas, cudaStream_t s) {
    static_assert(EXPERTS % WARPS == 0, "the padding CTAs hold whole warps of experts");
    void* args[] = {const_cast<sort_globals*>(&G)};
    const dim3 warp(kittens::WARP_THREADS);
    cudaError_t err = cudaLaunchKernel(moe_sort_count_ker<EXPERTS>, dim3(chunks), warp, args, 0, s);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    err = cudaLaunchKernel(moe_sort_offsets_ker<EXPERTS, TILE>, dim3(1), warp, args, 0, s);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    void* fargs[] = {const_cast<sort_fp8_globals*>(&F)};
    const dim3 grid(pair_ctas + uint32_t(EXPERTS / WARPS)), block(NUM_THREADS);
    return static_cast<int>(cudaLaunchKernel(moe_sort_scatter_fp8_ker, grid, block, fargs, 0, s));
}

}  // namespace

// The kernel's one-time setup, run once per device by `kernels::prepare_all` (`CudaDevice::open`):
// query its attributes (`cudaFuncGetAttributes`), which loads its module, so a worker's graph
// capture (`CudaWorker::new`) never loads a kernel mid-capture. It changes nothing on the device.
//
// 0 = done; any other value = the cudaError_t of the first query that failed.
extern "C" int scratchy_cuda_moe_sort_prepare(void) {
    cudaFuncAttributes attr;
    if (const cudaError_t e = cudaFuncGetAttributes(&attr, moe_sort_count_ker<EXPERTS_SMALL>); e != cudaSuccess) {
        return static_cast<int>(e);
    }
    if (const cudaError_t e = cudaFuncGetAttributes(&attr, moe_sort_offsets_ker<EXPERTS_SMALL>); e != cudaSuccess) {
        return static_cast<int>(e);
    }
    if (const cudaError_t e = cudaFuncGetAttributes(&attr, moe_sort_count_ker<EXPERTS_LARGE>); e != cudaSuccess) {
        return static_cast<int>(e);
    }
    if (const cudaError_t e = cudaFuncGetAttributes(&attr, moe_sort_offsets_ker<EXPERTS_LARGE>); e != cudaSuccess) {
        return static_cast<int>(e);
    }
    if (const cudaError_t e = cudaFuncGetAttributes(&attr, moe_sort_scatter_ker); e != cudaSuccess) {
        return static_cast<int>(e);
    }
    if (const cudaError_t e = cudaFuncGetAttributes(&attr, moe_sort_scatter_fp8_ker);
        e != cudaSuccess) {
        return static_cast<int>(e);
    }
    if (const cudaError_t e = cudaFuncGetAttributes(&attr, moe_sort_offsets_ker<EXPERTS_SMALL, 64>);
        e != cudaSuccess) {
        return static_cast<int>(e);
    }
    if (const cudaError_t e = cudaFuncGetAttributes(&attr, moe_sort_offsets_ker<EXPERTS_LARGE, 64>);
        e != cudaSuccess) {
        return static_cast<int>(e);
    }
    return 0;
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught; any other value = the cudaError_t
// cudaLaunchKernel returned for the first of the three launches that failed (the ones after it
// are not launched). Nothing here reads or clears the runtime's last-error state.
//
// `m` non-zero, `experts` `E` one of 128 and 256 (`op_abi::MOE_EXPERT_COUNTS`), `top_k` in 1..=32
// (`step::MOE_ROUTE_MAX_TOP_K`), `hidden` a non-zero multiple of 8, `m · top_k` and `m · hidden` at
// most INT_MAX, and `max_tiles` at least `tape::moe_tiles`' bound `(P + 127 · min(E, P)) / 128`
// with `max_tiles · 128 · hidden` at most INT_MAX; all are checked before any pointer is touched.
// The bound is the most tiles any routing pads to, so every row and tile index the kernels form is
// in its region. `x` is `[m, hidden]` bf16 and `rows` `[max_tiles · 128, hidden]` bf16, 16-byte
// aligned; `topk_idx` `[P]` i32, each in `[0, E)`; `chunk_hist` `[⌈P / 256⌉, E]`, `counts` `[E]`,
// `offsets` `[E + 1]`, `tile_expert` `[max_tiles]` and `pos` `[P]` u32, each 16-byte aligned; no
// output overlaps another or an input. Static shared memory only, so its one-time setup
// (`_prepare`, in `PREPARE`) only loads it. The launches are asynchronous on `stream`, in order.
extern "C" int scratchy_cuda_moe_sort(const void* x, const void* topk_idx, void* chunk_hist,
                                      void* counts, void* offsets, void* tile_expert, void* pos,
                                      void* rows, uint32_t m, uint32_t experts, uint32_t top_k,
                                      uint32_t hidden, uint32_t max_tiles,
                                      void* stream /* cudaStream_t; null = default */) {
    const uint64_t pairs = uint64_t(m) * uint64_t(top_k);
    const uint64_t runs = pairs < uint64_t(experts) ? pairs : uint64_t(experts);
    const uint64_t bound = (pairs + uint64_t(TILE_ROWS - 1) * runs) / uint64_t(TILE_ROWS);
    const bool counted = experts == uint32_t(EXPERTS_SMALL) || experts == uint32_t(EXPERTS_LARGE);
    if (m == 0 || !counted || top_k == 0 || top_k > MAX_TOP_K ||
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
                  size_t(experts)};
        idx_gl cg{static_cast<int*>(counts), nullptr, nullptr, size_t(1), size_t(experts)};
        idx_gl og{static_cast<int*>(offsets), nullptr, nullptr, size_t(1), size_t(experts + 1)};
        idx_gl tg{static_cast<int*>(tile_expert), nullptr, nullptr, size_t(1), size_t(max_tiles)};
        idx_gl pg{static_cast<int*>(pos), nullptr, nullptr, size_t(1), size_t(pairs)};
        row_gl rg{static_cast<bf16*>(rows), nullptr, nullptr,
                  size_t(max_tiles) * size_t(TILE_ROWS), size_t(hidden)};
        sort_globals G{xg, ig, hg, cg, og, tg, pg, rg, int(pairs), int(chunks), int(top_k),
                       int(hidden), int(pair_ctas)};
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        if (experts == uint32_t(EXPERTS_SMALL)) {
            return launch_sort<EXPERTS_SMALL>(G, chunks, pair_ctas, s);
        }
        return launch_sort<EXPERTS_LARGE>(G, chunks, pair_ctas, s);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// FP8 S8 P1: the sort with the e4m3 scatter. Return codes, launches and rule as
// `scratchy_cuda_moe_sort`'s, `hidden` a multiple of 16 (each e4m3 row starts 16-byte aligned for
// the bulk copy), with `xq` `[m, hidden]` e4m3 then `x_scale` `[m]` f32 (the token rows'
// `quant_rows_fp8`) in place of `x`, and `rows` `[max_tiles · 128, hidden]` e4m3 then
// `row_scale` `[max_tiles · 128]` f32 in place of the bf16 rows: pair `p`'s row is its token's e4m3
// row and scale, each live tile's padding row +0 bytes with `fp8_scale(0)`, every other row
// unwritten. `xq` and `rows` 16-byte aligned, the scales 4-byte aligned; no output overlaps another
// or an input.
namespace {

// The e4m3 sort's launcher at `TILE`-row tiles: the rule, the views and the launches.
template <int TILE>
int sort_fp8(const void* xq, const void* x_scale, const void* topk_idx, void* chunk_hist,
             void* counts, void* offsets, void* tile_expert, void* pos, void* rows, void* row_scale,
             uint32_t m, uint32_t experts, uint32_t top_k, uint32_t hidden, uint32_t max_tiles,
             void* stream) {
    const uint64_t pairs = uint64_t(m) * uint64_t(top_k);
    const uint64_t runs = pairs < uint64_t(experts) ? pairs : uint64_t(experts);
    const uint64_t bound = (pairs + uint64_t(TILE - 1) * runs) / uint64_t(TILE);
    const bool counted = experts == uint32_t(EXPERTS_SMALL) || experts == uint32_t(EXPERTS_LARGE);
    if (m == 0 || !counted || top_k == 0 || top_k > MAX_TOP_K || hidden == 0 ||
        hidden % (2 * VEC) != 0 || pairs > uint64_t(INT_MAX) ||
        uint64_t(m) * uint64_t(hidden) > uint64_t(INT_MAX) || uint64_t(max_tiles) < bound ||
        uint64_t(max_tiles) * uint64_t(TILE) * uint64_t(hidden) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const uint32_t chunks = uint32_t((pairs + uint64_t(CHUNK) - 1) / uint64_t(CHUNK));
        const uint32_t pair_ctas = uint32_t((pairs + uint64_t(WARPS) - 1) / uint64_t(WARPS));
        const size_t capacity = size_t(max_tiles) * size_t(TILE);
        // No view carries a TMA type, so building them encodes nothing. (a) and (b) read neither
        // row view; the e4m3 buffers stand in for them.
        row_gl xg{static_cast<bf16*>(const_cast<void*>(xq)), nullptr, nullptr, size_t(m),
                  size_t(hidden / 2)};
        idx_gl ig{static_cast<int*>(const_cast<void*>(topk_idx)), nullptr, nullptr, size_t(1),
                  size_t(pairs)};
        idx_gl hg{static_cast<int*>(chunk_hist), nullptr, nullptr, size_t(chunks),
                  size_t(experts)};
        idx_gl cg{static_cast<int*>(counts), nullptr, nullptr, size_t(1), size_t(experts)};
        idx_gl og{static_cast<int*>(offsets), nullptr, nullptr, size_t(1), size_t(experts + 1)};
        idx_gl tg{static_cast<int*>(tile_expert), nullptr, nullptr, size_t(1), size_t(max_tiles)};
        idx_gl pg{static_cast<int*>(pos), nullptr, nullptr, size_t(1), size_t(pairs)};
        row_gl rg{static_cast<bf16*>(rows), nullptr, nullptr, capacity, size_t(hidden / 2)};
        sort_globals G{xg, ig, hg, cg, og, tg, pg, rg, int(pairs), int(chunks), int(top_k),
                       int(hidden), int(pair_ctas)};
        sort_fp8_globals F{
            q_gl{static_cast<fp8e4m3*>(const_cast<void*>(xq)), nullptr, nullptr, size_t(m),
                 size_t(hidden)},
            scale_gl{static_cast<float*>(const_cast<void*>(x_scale)), nullptr, nullptr, nullptr,
                     size_t(m)},
            ig,
            hg,
            cg,
            og,
            pg,
            q_gl{static_cast<fp8e4m3*>(rows), nullptr, nullptr, capacity, size_t(hidden)},
            scale_gl{static_cast<float*>(row_scale), nullptr, nullptr, nullptr, capacity},
            int(pairs),
            int(top_k),
            int(hidden),
            int(pair_ctas)};
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        if (experts == uint32_t(EXPERTS_SMALL)) {
            return launch_sort_fp8<EXPERTS_SMALL, TILE>(G, F, chunks, pair_ctas, s);
        }
        return launch_sort_fp8<EXPERTS_LARGE, TILE>(G, F, chunks, pair_ctas, s);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

}  // namespace

extern "C" int scratchy_cuda_moe_sort_fp8(const void* xq, const void* x_scale, const void* topk_idx,
                                          void* chunk_hist, void* counts, void* offsets,
                                          void* tile_expert, void* pos, void* rows,
                                          void* row_scale, uint32_t m, uint32_t experts,
                                          uint32_t top_k, uint32_t hidden, uint32_t max_tiles,
                                          void* stream /* cudaStream_t; null = default */) {
    return sort_fp8<TILE_ROWS>(xq, x_scale, topk_idx, chunk_hist, counts, offsets, tile_expert,
                               pos, rows, row_scale, m, experts, top_k, hidden, max_tiles, stream);
}

// FP8 S8 P3(a): `scratchy_cuda_moe_sort_fp8` with each expert's run padded to 64 rows (an fp8 bank's
// tile, `op_abi::CUDA_FP8_EXPERT_TILE_ROWS`): `max_tiles` counts 64-row tiles, `rows` and
// `row_scale` hold `max_tiles · 64` rows, `tile_expert` names each 64-row tile's expert, and the
// bound is `(P + 63 · min(E, P)) / 64`. Each pair's sorted row is still its token's e4m3 row and
// scale; each live tile's padding row +0 bytes with the zero row's scale.
extern "C" int scratchy_cuda_moe_sort_fp8_m64(const void* xq, const void* x_scale,
                                              const void* topk_idx, void* chunk_hist, void* counts,
                                              void* offsets, void* tile_expert, void* pos,
                                              void* rows, void* row_scale, uint32_t m,
                                              uint32_t experts, uint32_t top_k, uint32_t hidden,
                                              uint32_t max_tiles,
                                              void* stream /* cudaStream_t; null = default */) {
    return sort_fp8<64>(xq, x_scale, topk_idx, chunk_hist, counts, offsets, tile_expert, pos, rows,
                        row_scale, m, experts, top_k, hidden, max_tiles, stream);
}
