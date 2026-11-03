// SPDX-License-Identifier: Apache-2.0
//
// The decode GEMVs for the H100 (#1031): c[n] = Σ_k a[k] · B[n, k], one activation row `a [1, k]`
// against a row-major weight `B [n, k]`, all bf16, accumulated in f32 and rounded once to bf16
// (round to nearest even). One kernel template, two instances:
//   - `gemv_bf16_ker<false>`: every one-row projection, the GEMV router and the lm_head
//     (`scratchy_cuda_gemv_bf16_abt`, `scratchy_cuda_gemv_split_bf16_abt`);
//   - `gemv_bf16_ker<true>`: the MoE block's gathered expert GEMV, one launch over the
//     `P = m · top_k` (token, pick) pairs, pair `p` reading the first `n` rows of expert
//     `e = topk_idx[p]`'s `[expert_rows, k]` block (`scratchy_cuda_expert_gemv_bf16`,
//     `scratchy_cuda_expert_gemv_split_bf16`).
// Both run one row routine with the same chunking, and take the same split for the same `(n, k)`
// (`step::gemv_splits`), so a pair's row is the dense GEMV's bits on the same row and expert slice
// (`tests/moe.rs`, `expert_gemv_bf16_is_gemv_bf16_per_pair`).
//
// A one-row matrix product reads each weight byte once for two flops, so its time is the time to
// stream `B`. The kernel is written for that, not as a GEMM at one row (the kernel it replaces ran
// `gemm_bf16.cu`'s WGMMA template, 63 of its 64 rows zero, one CTA per 64 columns):
//   - **Every SM streams.** The grid is `(column blocks, K splits, pairs)`. A CTA is 8 warps, each
//     owning `ROWS_PER_WARP` = 4 consecutive output rows, so a column block is 32 rows; the `k`
//     axis is cut into `splits` runs of whole 256-element chunks. `step::gemv_splits` picks the
//     split from `(n, k)` so a narrow projection still puts two CTAs on each of the 132 SMs.
//   - **16-byte transfers, one wait per chunk of several rows.** Each warp runs its own 4-stage
//     ring of shared vectors: per stage, lane 0 sets the byte count (`tma::expect_bytes`) and
//     issues one vector TMA load (`tma::load_async` of an `sv_bf<256>`, 512 bytes) of `a`'s chunk
//     and one of each of its rows' chunks, all on the stage's semaphore. The warp waits once per
//     stage, then per row loads the chunk into a naive `rv_fl<256>` (`warp::load`, bf16 to f32,
//     exact), `warp::mul` by `a`'s and `warp::add` into the row's accumulator. Then it hands the
//     stage back to its own next load (`__syncwarp`, then lane 0 issues chunk `j + 4`). A warp
//     waits on nothing but its own loads: no producer warp, no CTA barrier.
//   - **The `k` tail is TMA's zero fill.** The views' columns end at `k`, a multiple of 8
//     (`step::GEMV_BF16_MULTIPLE`); a row's last chunk may run past it, and a plain (non-swizzled)
//     box reads zeros there (the TK-1 probe's R1, `tests/tma_bounds.rs`), for `a` and `B` alike,
//     so the lanes past `k` add exactly 0. A split ends on a chunk boundary, so only the row's last
//     chunk ever crosses `k`. Rows past `n` are never loaded: a warp's row count is clamped.
//   - **The order is fixed.** Per row, each lane adds its elements over the chunks in ascending
//     K, in f32; `warp::sum` (each lane's 8 elements in order, then a fixed shuffle tree into lane
//     0) gives the split's f32 partial. No atomics: the same bits on every run, wherever the
//     operands sit.
//   - **The splits' sum.** With one split, lane 0 rounds the row's partial to bf16
//     (`base_types::convertor<bf16, float>`) and stores it through `gl::operator[]`. With `S > 1`
//     lane 0 stores the f32 partial into `partials [S, P · n_pad]` (`n_pad` = `n` rounded up to the
//     32-row block; the rows past `n` store +0, so every element the reduce reads is written), and
//     a second launch, `gemv_reduce_ker`, adds each row's `S` partials in ascending split order in
//     f32 and rounds once: one warp per 32 rows, `warp::load` of a naive `rv_fl<32>` per split
//     (lane `l` holds row `32v + l`), `warp::add` split by split. TK at the pin has no cross-CTA
//     acquire or release, so the ordered sum is a second launch on the same stream, never the last
//     CTA of a column block. `partials` is the tape's GEMV scratch (`tape::CudaTape::gemv_scratch`).
//
// Every product and sum is f32 and the only bf16 rounding is the final one. The split changes the
// order the products are added in, so a shape's bits depend on its split: the split is a constant
// of the shape (`step::gemv_splits`), never of the launch. Only `c`'s `P · n` elements and, at
// `S > 1`, the scratch's `S · P · n_pad` f32 are written.
//
// ThunderKittens' operations only (PLAN-tk100 §1), each checked against the pin:
//   - `tma::expect_bytes`, `tma::load_async` of an `sv` through a `gl` with an `sv` descriptor
//     (`ops/group/util/tma.cuh:18`, `ops/group/memory/vec/tma.cuh:209`; the descriptor
//     `types/global/tma.cuh:233`, its fill `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`, zeros, `:55`);
//   - `init_semaphore`, `wait` (`ops/thread/util/sync.cuh:37, :106`);
//   - `warp::load` of a naive `rv` from an `sv` and from a `gl`
//     (`ops/group/memory/vec/shared_to_register.cuh:14, :73`; `global_to_register.cuh:14, :66`);
//   - `warp::zero`, `warp::add`, `warp::mul` on naive `rv_fl` (`ops/group/register/vec/maps.cuh:132,
//     :333, :359`); `warp::sum` (`reductions.cuh:133`, its naive tree `:66`);
//   - `base_types::convertor<bf16, float>` (`common/base_types.cuh:561`);
//   - `shared_allocator<1024>` (`common/util.cuh:252`), `gl::operator[]` (`types/global/gl.cuh:174`).
//
// Shared memory: 8 warps × 4 stages × (1 + 4) vectors of 512 bytes = 81,920 bytes, plus 1,024 of
// alignment slack = 82,944 bytes of dynamic shared memory, so two CTAs fit an SM's 227 KiB; set
// once per instance by `_prepare` (in `PREPARE`). The semaphores are static.

#include "kittens.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::GEMV_BF16_MULTIPLE`: `k` is a multiple of it, so every row stride is a multiple of 16
// bytes, as the tensor maps need.
constexpr uint32_t VEC = 8;
// `step::GEMV_CHUNK`, `GEMV_ROWS_PER_WARP`, `GEMV_BLOCK_ROWS`, `GEMV_MAX_SPLITS`.
constexpr int CHUNK = 256;
constexpr int ROWS_PER_WARP = 4;
constexpr int WARPS = 8;
constexpr int BLOCK_ROWS = WARPS * ROWS_PER_WARP;
constexpr int MAX_SPLITS = 16;
constexpr int STAGES = 4;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;
// The reduce kernel: one lane per output row, 8 warps per CTA.
constexpr int REDUCE_WARPS = 8;
constexpr int REDUCE_ROWS = kittens::WARP_THREADS;
constexpr int REDUCE_THREADS = REDUCE_WARPS * kittens::WARP_THREADS;
// The expert GEMV's bounds (`op_abi::MOE_EXPERT_COUNTS`, `step::MOE_ROUTE_MAX_TOP_K`), and the
// grid's z extent, one slice per pair.
constexpr uint32_t EXPERTS_SMALL = 128;
constexpr uint32_t EXPERTS_LARGE = 256;
constexpr uint32_t MAX_TOP_K = 32;
constexpr uint32_t MAX_GRID_Z = 65535;

using chunk_sv = sv_bf<CHUNK>;
using chunk_rv = rv_fl<CHUNK>;
using reduce_rv = rv_fl<REDUCE_ROWS>;
// `a [rows, k]` and `B [depth, rows, k]` bf16, each with its vector tensor map.
using in_gl = gl<bf16, 1, 1, -1, -1, chunk_sv>;
using weight_gl = gl<bf16, 1, -1, -1, -1, chunk_sv>;
// `c [P, n]` bf16 and the partials `[S, P · n_pad]` f32: no tensor map.
using out_gl = gl<bf16, 1, 1, -1, -1>;
using partial_gl = gl<float, 1, 1, -1, -1>;
using idx_gl = gl<int, 1, 1, -1, -1>;

struct gemv_globals {
    in_gl a;
    weight_gl b;
    out_gl c;
    partial_gl partials;
    // `[1, P]` expert ids; the dense instance never reads it.
    idx_gl topk_idx;
    // The rows of `c` and of a pair's run of partials (`n` rounded up to 32).
    int n;
    int n_pad;
    // The chunks of a row, `⌈k / 256⌉`, and of one split, `⌈chunks / splits⌉`.
    int chunks;
    int split_chunks;
    int splits;
    // Pair `p` reads `a`'s row `p / row_div` (`top_k` for the token rows, 1 for the pair rows and
    // the dense GEMV).
    int row_div;
};

struct reduce_globals {
    partial_gl partials;
    out_gl c;
    int n;
    int n_pad;
    int splits;
};

constexpr size_t SMEM_VECTORS = size_t(WARPS) * STAGES * (1 + ROWS_PER_WARP) * sizeof(chunk_sv);
static_assert(sizeof(chunk_sv) == 512, "a chunk is 512 bytes");
static_assert(SMEM_VECTORS == 81920, "8 warps, 4 stages, a and 4 rows");
// The vectors plus the slack the allocator may spend aligning the first to 1024 bytes.
constexpr int GEMV_SMEM_BYTES = int(SMEM_VECTORS) + 1024;
static_assert(2 * GEMV_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "two CTAs per SM");

template <bool EXPERT>
__global__ __launch_bounds__(NUM_THREADS, 2) void gemv_bf16_ker(const __grid_constant__ gemv_globals g) {
    extern __shared__ int __shm[];
    shared_allocator<1024> al((int*)&__shm[0]);
    chunk_sv(&a_s)[WARPS][STAGES] = al.allocate<chunk_sv, WARPS, STAGES>();
    chunk_sv(&b_s)[WARPS][STAGES][ROWS_PER_WARP] =
        al.allocate<chunk_sv, WARPS, STAGES, ROWS_PER_WARP>();
    __shared__ kittens::semaphore arrived[WARPS][STAGES];

    const int w = kittens::warpid();
    const int split = blockIdx.y;
    const int p = blockIdx.z;
    const int row0 = blockIdx.x * BLOCK_ROWS + w * ROWS_PER_WARP;
    const int rows = max(0, min(ROWS_PER_WARP, g.n - row0));
    const int c0 = split * g.split_chunks;
    const int count = max(0, min(g.split_chunks, g.chunks - c0));
    const int a_row = p / g.row_div;
    int depth = 0;
    if constexpr (EXPERT) {
        depth = g.topk_idx[{0, 0, 0, p}];
    }

    chunk_rv acc[ROWS_PER_WARP];
    chunk_rv x, y;
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        warp::zero(acc[r]);
    }
    if (rows > 0 && count > 0) {
        if (warp::laneid() == 0) {
            for (int s = 0; s < STAGES; s++) {
                init_semaphore(arrived[w][s], 0, 1);
            }
        }
        __syncwarp();
        // Chunk `j` of the split (`c0 + j` of the row) into stage `j % STAGES`.
        const auto issue = [&](int j) {
            const int s = j % STAGES;
            warp::tma::expect_bytes(arrived[w][s], uint32_t((1 + rows) * sizeof(chunk_sv)));
            warp::tma::load_async(a_s[w][s], g.a, {0, 0, a_row, c0 + j}, arrived[w][s]);
            for (int r = 0; r < rows; r++) {
                warp::tma::load_async(b_s[w][s][r], g.b, {0, depth, row0 + r, c0 + j},
                                      arrived[w][s]);
            }
        };
        for (int j = 0; j < min(STAGES, count); j++) {
            issue(j);
        }
        for (int j = 0; j < count; j++) {
            const int s = j % STAGES;
            wait(arrived[w][s], (j / STAGES) % 2);
            warp::load(x, a_s[w][s]);
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                if (r < rows) {
                    warp::load(y, b_s[w][s][r]);
                    warp::mul(y, y, x);
                    warp::add(acc[r], acc[r], y);
                }
            }
            // Every lane has read the stage into registers before its next load overwrites it.
            __syncwarp();
            if (j + STAGES < count) {
                issue(j + STAGES);
            }
        }
    }
    // Each row's partial, in every lane; lane 0 stores it. The rows past `n` (and a split past the
    // row's chunks) are +0, stored only into the scratch so the reduce reads written values.
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        const float dot = warp::sum(acc[r]);
        const int row = row0 + r;
        if (warp::laneid() == 0) {
            if (g.splits == 1) {
                if (r < rows) {
                    g.c[{0, 0, p, row}] = base_types::convertor<bf16, float>::convert(dot);
                }
            } else {
                g.partials[{0, 0, split, p * g.n_pad + row}] = dot;
            }
        }
    }
}

__global__ __launch_bounds__(REDUCE_THREADS) void gemv_reduce_ker(
    const __grid_constant__ reduce_globals g) {
    // Rows `32v .. 32v + 32` of the `[S, P · n_pad]` partials, one per lane.
    const int v = blockIdx.x * REDUCE_WARPS + kittens::warpid();
    if (v * REDUCE_ROWS >= g.partials.cols()) {
        return;
    }
    reduce_rv acc, part;
    warp::zero(acc);
    for (int s = 0; s < g.splits; s++) {
        warp::load(part, g.partials, {0, 0, s, v});
        warp::add(acc, acc, part);
    }
    const int rr = v * REDUCE_ROWS + warp::laneid();
    const int p = rr / g.n_pad;
    const int col = rr % g.n_pad;
    if (col < g.n) {
        g.c[{0, 0, p, col}] = base_types::convertor<bf16, float>::convert(acc[0][0]);
    }
}

// `n` rounded up to the 32-row block.
uint32_t padded(uint32_t n) {
    return (n + uint32_t(BLOCK_ROWS) - 1) / uint32_t(BLOCK_ROWS) * uint32_t(BLOCK_ROWS);
}

// The chunks of one split, or 0 when `splits` does not cut `chunks` into non-empty runs (1 to
// `MAX_SPLITS` of them).
uint32_t split_chunks(uint32_t chunks, uint32_t splits) {
    if (splits == 0 || splits > uint32_t(MAX_SPLITS) || splits > chunks) {
        return 0;
    }
    const uint32_t each = (chunks + splits - 1) / splits;
    return (splits - 1) * each < chunks ? each : 0;
}

// The main kernel of instance `EXPERT`, then at `splits > 1` the reduce, on `stream`.
template <bool EXPERT>
int launch(gemv_globals G, uint32_t n, uint32_t pairs, uint32_t splits, void* stream) {
    void* args[] = {&G};
    const dim3 grid(padded(n) / uint32_t(BLOCK_ROWS), splits, pairs), block(NUM_THREADS);
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const cudaError_t launched =
        cudaLaunchKernel(gemv_bf16_ker<EXPERT>, grid, block, args, GEMV_SMEM_BYTES, s);
    if (launched != cudaSuccess || splits == 1) {
        return static_cast<int>(launched);
    }
    reduce_globals R{G.partials, G.c, int(n), int(padded(n)), int(splits)};
    void* rargs[] = {&R};
    const uint32_t vectors = pairs * padded(n) / uint32_t(REDUCE_ROWS);
    const dim3 rgrid((vectors + REDUCE_WARPS - 1) / REDUCE_WARPS),
        rblock(REDUCE_THREADS);
    return static_cast<int>(cudaLaunchKernel(gemv_reduce_ker, rgrid, rblock, rargs, 0, s));
}

// The dense GEMV at `splits`: the shape rule, then the views and the launch.
int dense(const void* a, const void* b, void* c, void* partials, uint32_t n, uint32_t k,
          uint32_t splits, void* stream) {
    if (n == 0 || n > uint32_t(INT_MAX) || k == 0 || k % VEC != 0 || k > uint32_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const uint32_t chunks = (k + uint32_t(CHUNK) - 1) / uint32_t(CHUNK);
    const uint32_t each = split_chunks(chunks, splits);
    if (each == 0 || (splits > 1 && partials == nullptr) ||
        uint64_t(splits) * uint64_t(padded(n)) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        // Building each view encodes its vector tensor map into host memory; nothing is allocated
        // on the device. Every row stride is `2k` bytes, a multiple of 16.
        in_gl ag{static_cast<bf16*>(const_cast<void*>(a)), nullptr, nullptr, size_t(1), size_t(k)};
        weight_gl bg{static_cast<bf16*>(const_cast<void*>(b)), nullptr, size_t(1), size_t(n),
                     size_t(k)};
        out_gl cg{static_cast<bf16*>(c), nullptr, nullptr, size_t(1), size_t(n)};
        partial_gl pg{static_cast<float*>(partials), nullptr, nullptr, size_t(splits),
                      size_t(padded(n))};
        idx_gl ig{nullptr, nullptr, nullptr, size_t(1), size_t(1)};
        gemv_globals G{ag,          bg,        cg,        pg,          ig, int(n), int(padded(n)),
                       int(chunks), int(each), int(splits), 1};
        return launch<false>(G, n, 1, splits, stream);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// The expert GEMV at `splits`: the shape rule, then the views and the launch.
int expert(const void* a, const void* b, const void* topk_idx, void* c, void* partials,
           uint32_t m, uint32_t top_k, uint32_t n, uint32_t k, uint32_t experts,
           uint32_t expert_rows, uint32_t pair_rows, uint32_t splits, void* stream) {
    const uint64_t pairs = uint64_t(m) * uint64_t(top_k);
    const uint64_t a_rows = pair_rows != 0 ? pairs : uint64_t(m);
    if (m == 0 || top_k == 0 || top_k > MAX_TOP_K || pairs > uint64_t(MAX_GRID_Z) || n == 0 ||
        n % 64 != 0 || k == 0 || k % 64 != 0 ||
        (experts != EXPERTS_SMALL && experts != EXPERTS_LARGE) || expert_rows < n ||
        a_rows * uint64_t(k) > uint64_t(INT_MAX) || pairs * uint64_t(n) > uint64_t(INT_MAX) ||
        uint64_t(experts) * uint64_t(expert_rows) * uint64_t(k) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const uint32_t chunks = (k + uint32_t(CHUNK) - 1) / uint32_t(CHUNK);
    const uint32_t each = split_chunks(chunks, splits);
    if (each == 0 || (splits > 1 && partials == nullptr) ||
        uint64_t(splits) * pairs * uint64_t(padded(n)) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        in_gl ag{static_cast<bf16*>(const_cast<void*>(a)), nullptr, nullptr, size_t(a_rows),
                 size_t(k)};
        weight_gl bg{static_cast<bf16*>(const_cast<void*>(b)), nullptr, size_t(experts),
                     size_t(expert_rows), size_t(k)};
        out_gl cg{static_cast<bf16*>(c), nullptr, nullptr, size_t(pairs), size_t(n)};
        partial_gl pg{static_cast<float*>(partials), nullptr, nullptr, size_t(splits),
                      size_t(pairs * padded(n))};
        idx_gl ig{static_cast<int*>(const_cast<void*>(topk_idx)), nullptr, nullptr, size_t(1),
                  size_t(pairs)};
        gemv_globals G{ag,          bg,        cg,          pg,
                       ig,          int(n),    int(padded(n)), int(chunks),
                       int(each),   int(splits), pair_rows != 0 ? 1 : int(top_k)};
        return launch<true>(G, n, uint32_t(pairs), splits, stream);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

}  // namespace

// Each instance's one-time setup: allow it `GEMV_SMEM_BYTES` of dynamic shared memory on the
// current device (the default limit is 48 KB). Run once per device before any launch
// (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch. The reduce kernel has
// no shared memory; the dense setup only loads it (`cudaFuncGetAttributes`), so a worker's graph
// capture (`CudaWorker::new`, #1048) never loads it mid-capture.
//
// 0 = done; any other value = the cudaError_t of the first call that failed.
extern "C" int scratchy_cuda_gemv_bf16_abt_prepare(void) {
    if (const cudaError_t e = cudaFuncSetAttribute(
            gemv_bf16_ker<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMV_SMEM_BYTES);
        e != cudaSuccess) {
        return static_cast<int>(e);
    }
    cudaFuncAttributes attr;
    return static_cast<int>(cudaFuncGetAttributes(&attr, gemv_reduce_ker));
}

// `scratchy_cuda_gemv_bf16_abt_prepare`'s, for the expert instance.
extern "C" int scratchy_cuda_expert_gemv_bf16_prepare(void) {
    return static_cast<int>(cudaFuncSetAttribute(
        gemv_bf16_ker<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMV_SMEM_BYTES));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value
// = the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `n` must be non-zero, `k` a non-zero multiple of 8, and both at most INT_MAX; all are checked
// before any pointer is touched. `a`, `b` and `c` must be 16-byte-aligned device pointers to `k`,
// `n·k` and `n` bf16, `c` overlapping neither input. One split, so no scratch.
// `scratchy_cuda_gemv_bf16_abt_prepare` must have succeeded on the current device. The launch is
// asynchronous on `stream`.
extern "C" int scratchy_cuda_gemv_bf16_abt(const void* a, const void* b, void* c, uint32_t n,
                                           uint32_t k,
                                           void* stream /* cudaStream_t; null = default */) {
    return dense(a, b, c, nullptr, n, k, 1, stream);
}

// As `scratchy_cuda_gemv_bf16_abt`, its `k` cut into `splits` runs of whole 256-element chunks
// (`step::gemv_splits`): `splits` in 1..=16, at most `⌈k / 256⌉` and cutting the chunks into
// non-empty runs, and `splits · n_pad` at most INT_MAX (`n_pad` = `n` rounded up to 32), checked
// before any pointer is touched. At `splits > 1`, `partials` must be a 16-byte-aligned device
// pointer to `splits · n_pad` f32, overlapping no operand (null is refused), and a second launch
// adds the splits; both are asynchronous on `stream`.
extern "C" int scratchy_cuda_gemv_split_bf16_abt(const void* a, const void* b, void* c,
                                                 void* partials, uint32_t n, uint32_t k,
                                                 uint32_t splits,
                                                 void* stream /* cudaStream_t; null = default */) {
    return dense(a, b, c, partials, n, k, splits, stream);
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value
// = the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `m` non-zero, `top_k` in 1..=32, `m · top_k` at most 65,535 (the grid's z), `n` and `k`
// non-zero multiples of 64, `experts` 128 or 256, `expert_rows` at least `n`, and `a_rows · k`,
// `m · top_k · n` and `experts · expert_rows · k` at most INT_MAX, with `a_rows` `m · top_k`
// when `pair_rows`, else `m`; all are checked before any pointer is touched. `a` is
// `[a_rows, k]` bf16; `b` points at expert 0's block of `[experts, expert_rows, k]` bf16 (only
// the first `n` rows of each are read); `topk_idx` `[m · top_k]` i32, each below `experts`; `c`
// `[m · top_k, n]` bf16. All 16-byte aligned, `c` overlapping no input. One split, so no scratch.
// `scratchy_cuda_expert_gemv_bf16_prepare` must have succeeded on the current device. The launch
// is asynchronous on `stream`.
extern "C" int scratchy_cuda_expert_gemv_bf16(const void* a, const void* b, const void* topk_idx,
                                              void* c, uint32_t m, uint32_t top_k, uint32_t n,
                                              uint32_t k, uint32_t experts, uint32_t expert_rows,
                                              uint32_t pair_rows,
                                              void* stream /* cudaStream_t; null = default */) {
    return expert(a, b, topk_idx, c, nullptr, m, top_k, n, k, experts, expert_rows, pair_rows, 1,
                  stream);
}

// As `scratchy_cuda_expert_gemv_bf16` at `splits` (`step::gemv_splits` of `(n, k)`, the dense
// GEMV's split for the same shape), with `scratchy_cuda_gemv_split_bf16_abt`'s split rule and
// `partials` `splits · m · top_k · n` f32 (`n` is a multiple of 64, so `n_pad = n`).
extern "C" int scratchy_cuda_expert_gemv_split_bf16(
    const void* a, const void* b, const void* topk_idx, void* c, void* partials, uint32_t m,
    uint32_t top_k, uint32_t n, uint32_t k, uint32_t experts, uint32_t expert_rows,
    uint32_t pair_rows, uint32_t splits, void* stream /* cudaStream_t; null = default */) {
    return expert(a, b, topk_idx, c, partials, m, top_k, n, k, experts, expert_rows, pair_rows,
                  splits, stream);
}
