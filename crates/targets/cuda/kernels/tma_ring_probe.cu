// SPDX-License-Identifier: Apache-2.0
//
// Test support only, for `tests/gemv_fp8_ring_probe.rs` (FP8 S8 W2c, /tmp/d-w2c.md §3): no
// `CudaStep` and no lowering use it. It asks whether the fp8 decode GEMV is bound by the rate of
// its TMA operations rather than by its arithmetic: the GEMV's grid (8 warps, 2 CTAs per SM, a
// ring of stages per warp or per 4 warps) streams an e4m3 weight `B [n, k]` through shared memory
// and does no dot product. Lane `l` folds byte `l` of each row's every 256 columns into an f32
// (TK's element convertor, then `base_ops::sum`), and every thread stores its sum to `out`, so no
// load is dead. With every weight byte 1.0 the folds count the bytes read, exactly.
//
// The variants differ only in the bytes one TMA operation carries:
//   0 `vec256_r3`: one `sv_fp8e4m3<256>` (256 B) per row per chunk, 3 rows per warp, the GEMV's
//     `rows_per_warp` at the 27B's gate+up and down, its ring of 4 stages per warp;
//   1 `vec256`: the same at 4 rows per warp (every variant below has 4);
//   2 `pair512`: one plain `st_fp8e4m3<2, 256>` (512 B) per 2 rows per chunk. TK splits a vector
//     longer than 256 elements into several TMA operations (`find_vector_divider`), so 512 bytes in
//     one operation is a 2-row tile;
//   3 `tile4k`: one `st_fp8e4m3<16, 256>` (4 KB) per 4 warps' 16 rows per chunk, 4 stages;
//   4 `page16k`: one `st_fp8e4m3<16, 1024>` (16 KB) per 4 warps per 1,024 columns, 3 stages, so
//     two CTAs still fit an SM.
// A ring shared by 4 warps has an `empty` semaphore per stage: each warp arrives once it has read
// the stage, and the group's first warp waits for all 4 before it refills it.
//
// The consume probe (FP8 S8 W2d, /tmp/d-w2d.md §1): today's load (one `sv_fp8e4m3<256>` per row per
// chunk, a 4-stage ring per warp, 3 or 4 rows a warp), with the consumer's work added one cost at
// a time, up to the GEMV's own dot loop and prologue (`gemv_fp8.cu` at e64f4452):
//   (a) `wait`: the stage handshake and one byte per row per lane, as `vec256`;
//   (b) `bytes`: every byte of each row, as two 32-bit shared reads a lane (TK's `warp::load` of
//       the same 256 bytes TMA-loaded as an `sv_int<64>`), folded by `scratchy::ops::wrap_add`;
//   (c) `widen`: the GEMV's widen of each row (`scratchy::ops::sv_fp8_to_rv_fl`, its body at the
//       base), summed by `warp::add`;
//   (d) `dot`: (c) with the bf16 activation row resident in shared memory, read each chunk
//       (`warp::load`), and the GEMV's `rv_fma<fma_acc>` in place of the add: its dot loop;
//   (e) `full`: (d) with `gemv_fp8.cu`'s `quantize_row` prologue (copied below): its structure.
// Each runs at exactly 2 or 3 CTAs per SM: the dynamic shared memory is padded so a third (or a
// fourth) CTA cannot fit, and `__launch_bounds__` lets the registers allow it. The 3-CTA form
// holds a 32-chunk row (`k` at most 8,192), so the 27B's down (17,408) runs at 2 only.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>
#include <type_traits>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

constexpr int WARPS = 8;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;
constexpr int CHUNK = 256;
// `gemv_fp8.cu`'s `GEMV_SMEM_BYTES`: every variant takes at least this much dynamic shared memory,
// so the SM holds the GEMV's 2 CTAs and no more, whatever its ring needs.
constexpr int GEMV_SMEM_BYTES = 68608;

using vec_box = sv_fp8e4m3<CHUNK>;
using pair_box = st_fp8e4m3<2, CHUNK, false>;
using tile_box = st_fp8e4m3<16, CHUNK>;
using page_box = st_fp8e4m3<16, 4 * CHUNK>;
static_assert(sizeof(vec_box) == 256 && sizeof(pair_box) == 512, "256 and 512 bytes an operation");
static_assert(sizeof(tile_box) == 4096 && sizeof(page_box) == 16384, "4 and 16 KB an operation");
// `B [n, k]` e4m3 with every variant's tensor map; `out [ctas, 256]` f32, no map.
using weight_gl = gl<fp8e4m3, 1, 1, -1, -1, vec_box, pair_box, tile_box, page_box>;
using out_gl = gl<float, 1, 1, -1, -1>;

struct probe_globals {
    weight_gl b;
    out_gl out;
    int n;
    int k;
};

// A variant: its box, the rows one box holds, the warps that share a ring, each warp's rows, the
// ring's stages, and the columns one box covers.
template <typename BOX, int BOX_ROWS, int GROUP, int ROWS, int STAGES, int K_STEP>
struct variant {
    using box = BOX;
    static constexpr int box_rows = BOX_ROWS;
    static constexpr int group = GROUP;
    static constexpr int rows = ROWS;
    static constexpr int stages = STAGES;
    static constexpr int k_step = K_STEP;
    // The boxes of one stage, and the rings of a CTA.
    static constexpr int ops = GROUP * ROWS / BOX_ROWS;
    static constexpr int rings = WARPS / GROUP;
    static constexpr int ring_bytes = rings * STAGES * ops * int(sizeof(BOX)) + 1024;
    static constexpr int smem = ring_bytes > GEMV_SMEM_BYTES ? ring_bytes : GEMV_SMEM_BYTES;
    static_assert(GROUP * ROWS % BOX_ROWS == 0 && WARPS % GROUP == 0, "whole boxes, whole groups");
    static_assert(2 * smem <= kittens::MAX_SHARED_MEMORY, "two CTAs per SM, as the GEMV");
};

using vec256_r3 = variant<vec_box, 1, 1, 3, 4, CHUNK>;
using vec256 = variant<vec_box, 1, 1, 4, 4, CHUNK>;
using pair512 = variant<pair_box, 2, 1, 4, 4, CHUNK>;
using tile4k = variant<tile_box, 16, 4, 4, 4, CHUNK>;
using page16k = variant<page_box, 16, 4, 4, 3, 4 * CHUNK>;

template <typename V>
__global__ __launch_bounds__(NUM_THREADS, 2) void tma_ring_probe_ker(
    const __grid_constant__ probe_globals g) {
    using box = typename V::box;
    extern __shared__ int __shm[];
    shared_allocator<1024> al((int*)&__shm[0]);
    box(&ring)[V::rings][V::stages][V::ops] = al.allocate<box, V::rings, V::stages, V::ops>();
    __shared__ kittens::semaphore full[V::rings][V::stages];
    __shared__ kittens::semaphore empty[V::rings][V::stages];
    const int w = kittens::warpid();
    const int lane = warp::laneid();
    const int grp = w / V::group;
    const int wg = w % V::group;
    const bool producer = wg == 0;
    // The ring's first row, and this warp's rows (a vector variant's last block may hold fewer).
    const int row0 = (blockIdx.x * V::rings + grp) * V::group * V::rows;
    const int rows = max(0, min(V::rows, g.n - (row0 + wg * V::rows)));
    const int chunks = g.k / V::k_step;
    if (producer && lane == 0) {
        for (int s = 0; s < V::stages; s++) {
            init_semaphore(full[grp][s], 0, 1);
            init_semaphore(empty[grp][s], V::group, 0);
        }
    }
    // The semaphores are initialised before any thread waits on them.
    __syncthreads();
    // Chunk `j` of the ring's rows into stage `j % stages`.
    const auto issue = [&](int j) {
        const int s = j % V::stages;
        if constexpr (V::box_rows == 1) {
            warp::tma::expect_bytes(full[grp][s], uint32_t(rows * sizeof(box)));
#pragma unroll
            for (int r = 0; r < V::rows; r++) {
                if (r < rows) {
                    warp::tma::load_async(ring[grp][s][r], g.b, {0, 0, row0 + r, j}, full[grp][s]);
                }
            }
        } else {
            warp::tma::expect_bytes(full[grp][s], uint32_t(V::ops * sizeof(box)));
#pragma unroll
            for (int o = 0; o < V::ops; o++) {
                warp::tma::load_async(ring[grp][s][o], g.b, {0, 0, row0 / V::box_rows + o, j},
                                      full[grp][s]);
            }
        }
    };
    float acc = 0.0f;
    if (rows > 0) {
        if (producer) {
            for (int j = 0; j < min(V::stages, chunks); j++) {
                issue(j);
            }
        }
        for (int j = 0; j < chunks; j++) {
            const int s = j % V::stages;
            wait(full[grp][s], (j / V::stages) % 2);
#pragma unroll
            for (int r = 0; r < V::rows; r++) {
                if (r < rows) {
                    const int rr = wg * V::rows + r;
                    const box& bx = ring[grp][s][rr / V::box_rows];
#pragma unroll
                    for (int q = 0; q < V::k_step / CHUNK; q++) {
                        fp8e4m3 v;
                        if constexpr (V::box_rows == 1) {
                            v = bx[q * CHUNK + lane];
                        } else {
                            v = bx[{rr % V::box_rows, q * CHUNK + lane}];
                        }
                        acc = base_ops::sum::op<float>(
                            acc, base_types::convertor<float, fp8e4m3>::convert(v));
                    }
                }
            }
            // Every lane has read the stage before its warp says so, or refills it.
            __syncwarp();
            if constexpr (V::group > 1) {
                if (lane == 0) {
                    arrive(empty[grp][s]);
                }
            }
            if (producer && j + V::stages < chunks) {
                if constexpr (V::group > 1) {
                    wait(empty[grp][s], (j / V::stages) % 2);
                }
                issue(j + V::stages);
            }
        }
    }
    g.out[{0, 0, int(blockIdx.x), int(threadIdx.x)}] = acc;
}

// Variant `V` over `G`'s weight: the shape rule, then its shared memory and the launch.
template <typename V>
int launch(probe_globals G, uint32_t n, uint32_t k, void* stream) {
    if (k % uint32_t(V::k_step) != 0 || (V::box_rows > 1 && n % uint32_t(WARPS * V::rows) != 0)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const cudaError_t set = cudaFuncSetAttribute(
        tma_ring_probe_ker<V>, cudaFuncAttributeMaxDynamicSharedMemorySize, V::smem);
    if (set != cudaSuccess) {
        return static_cast<int>(set);
    }
    void* args[] = {&G};
    const uint32_t block_rows = uint32_t(WARPS * V::rows);
    const dim3 grid((n + block_rows - 1) / block_rows), block(NUM_THREADS);
    return static_cast<int>(cudaLaunchKernel(tma_ring_probe_ker<V>, grid, block, args, V::smem,
                                             static_cast<cudaStream_t>(stream)));
}

// ── The consume probe ─────────────────────────────────────────────

using row_sv = sv_bf<CHUNK>;
using int_box = sv_int<CHUNK / 4>;
using chunk_rv = rv_fl<CHUNK>;
using int_rv = rv<int, CHUNK / 4>;
static_assert(sizeof(int_box) == sizeof(vec_box), "the same 256 bytes, read as 64 words");
// `a [1, k]` bf16 with its vector map; `B` as 32-bit words `[n, k / 4]`; `out [ctas, 256]` int.
using in_gl = gl<bf16, 1, 1, -1, -1, row_sv>;
using fp8_gl = gl<fp8e4m3, 1, 1, -1, -1, vec_box>;
using word_gl = gl<int, 1, 1, -1, -1, int_box>;
using count_gl = gl<int, 1, 1, -1, -1>;

struct consume_globals {
    fp8_gl b;
    word_gl words;
    in_gl a;
    count_gl out;
    int n;
    int k;
};

constexpr int STAGES = 4;
// The dynamic shared memory of a CTA: 80 KB, so a third CTA cannot fit an SM's 228 KB (2 CTAs),
// or 60 KB, so a fourth cannot (3 CTAs).
template <int CTAS>
constexpr int pad_bytes = CTAS == 2 ? 81920 : 61440;
template <int CTAS>
constexpr int row_chunks = CTAS == 2 ? 68 : 32;

template <int C, int ROWS, int CTAS>
struct consume {
    static constexpr int rc = row_chunks<CTAS>;
    static constexpr int used = rc * int(sizeof(row_sv)) +
                                WARPS * STAGES * ROWS * int(sizeof(vec_box)) + 1024;
    static constexpr int smem = used > pad_bytes<CTAS> ? used : pad_bytes<CTAS>;
    // An SM's 228 KB less 1 KB the runtime reserves per CTA and ~1 KB of static shared memory.
    static_assert(CTAS * (smem + 2048) <= 228 * 1024 && (CTAS + 1) * smem > 228 * 1024,
                  "exactly CTAS CTAs fit an SM by shared memory");
};

// `gemv_fp8.cu`'s `quantize_row` (e64f4452) on a one-split row, every thread of the CTA calling it:
// the row's TMA load into `row`; with `QUANT`, its amax, scale and in-place quantization, else
// only the wait (variant (d)).
template <bool QUANT, int RC>
__device__ inline float probe_row(const in_gl& a, int chunks, row_sv (&row)[RC],
                                  kittens::semaphore& row_arrived, sv_fl<16>& amax) {
    const int w = kittens::warpid();
    if (w == 0) {
        if (warp::laneid() == 0) {
            init_semaphore(row_arrived, 0, 1);
        }
        __syncwarp();
        warp::tma::expect_bytes(row_arrived, uint32_t(chunks * sizeof(row_sv)));
        for (int c = 0; c < chunks; c++) {
            warp::tma::load_async(row[c], a, {0, 0, 0, c}, row_arrived);
        }
    }
    __syncthreads();
    wait(row_arrived, 0);
    if constexpr (!QUANT) {
        return 1.0f;
    } else {
        chunk_rv x, m;
        warp::zero(m);
        for (int c = w; c < chunks; c += WARPS) {
            warp::load(x, row[c]);
            warp::abs(x, x);
            warp::max(m, m, x);
        }
        const float part = warp::max(m);
        if (warp::laneid() == 0) {
            amax[w] = part;
            amax[w + WARPS] = part;
        }
        __syncthreads();
        rv_fl<16> all;
        warp::zero(all);
        warp::load(all, amax);
        const float scale = scratchy::ops::fp8_scale::op<float>(warp::max(all));
        for (int c = w; c < chunks; c += WARPS) {
            warp::load(x, row[c]);
            warp::bin_op<scratchy::ops::fp8_quant_pre>(x, x, scale);
            warp::unary_op<scratchy::ops::fp8_round>(x, x);
            warp::store(row[c], x);
        }
        __syncthreads();
        return scale;
    }
}

template <int C, int ROWS, int CTAS>
__global__ __launch_bounds__(NUM_THREADS, CTAS) void tma_consume_probe_ker(
    const __grid_constant__ consume_globals g) {
    using P = consume<C, ROWS, CTAS>;
    // (b) loads the same 256 bytes as 64 words.
    using box = std::conditional_t<C == 1, int_box, vec_box>;
    extern __shared__ int __shm[];
    shared_allocator<1024> al((int*)&__shm[0]);
    row_sv(&row)[P::rc] = al.allocate<row_sv, P::rc>();
    box(&ring)[WARPS][STAGES][ROWS] = al.allocate<box, WARPS, STAGES, ROWS>();
    __shared__ kittens::semaphore row_arrived;
    __shared__ kittens::semaphore full[WARPS][STAGES];
    __shared__ sv_fl<16> amax;
    const int w = kittens::warpid();
    const int lane = warp::laneid();
    const int row0 = (blockIdx.x * WARPS + w) * ROWS;
    const int rows = max(0, min(ROWS, g.n - row0));
    const int chunks = g.k / CHUNK;
    // Chunk `j` of the warp's rows into stage `j % STAGES`.
    const auto issue = [&](int j) {
        const int s = j % STAGES;
        warp::tma::expect_bytes(full[w][s], uint32_t(rows * sizeof(vec_box)));
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            if (r < rows) {
                if constexpr (C == 1) {
                    warp::tma::load_async(ring[w][s][r], g.words, {0, 0, row0 + r, j},
                                          full[w][s]);
                } else {
                    warp::tma::load_async(ring[w][s][r], g.b, {0, 0, row0 + r, j}, full[w][s]);
                }
            }
        }
    };
    if (rows > 0) {
        if (lane == 0) {
            for (int s = 0; s < STAGES; s++) {
                init_semaphore(full[w][s], 0, 1);
            }
        }
        __syncwarp();
        for (int j = 0; j < min(STAGES, chunks); j++) {
            issue(j);
        }
    }
    if constexpr (C >= 3) {
        probe_row<C == 4, P::rc>(g.a, chunks, row, row_arrived, amax);
    }
    float count = 0.0f;
    int_rv words;
    warp::zero(words);
    chunk_rv acc[ROWS];
    chunk_rv x, y;
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        warp::zero(acc[r]);
    }
    if (rows > 0) {
        for (int j = 0; j < chunks; j++) {
            const int s = j % STAGES;
            if constexpr (C >= 3) {
                warp::load(x, row[j]);
            }
            wait(full[w][s], (j / STAGES) % 2);
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                if (r < rows) {
                    if constexpr (C == 0) {
                        count = base_ops::sum::op<float>(
                            count, base_types::convertor<float, fp8e4m3>::convert(ring[w][s][r][lane]));
                    } else if constexpr (C == 1) {
                        int_rv v;
                        warp::load(v, ring[w][s][r]);
                        warp::bin_op<scratchy::ops::wrap_add>(words, words, v);
                    } else if constexpr (C == 2) {
                        scratchy::ops::sv_fp8_to_rv_fl(y, ring[w][s][r]);
                        warp::add(acc[r], acc[r], y);
                    } else {
                        scratchy::ops::sv_fp8_to_rv_fl(y, ring[w][s][r]);
                        scratchy::ops::rv_fma<scratchy::ops::fma_acc>(acc[r], y, x, acc[r]);
                    }
                }
            }
            // Every lane has read the stage into registers before its next load overwrites it.
            __syncwarp();
            if (j + STAGES < chunks) {
                issue(j + STAGES);
            }
        }
    }
    int folded = 0;
    if constexpr (C == 0) {
        folded = base_types::convertor<int, float>::convert(count);
    } else if constexpr (C == 1) {
        folded = scratchy::ops::wrap_add::op<int>(words[0][0], words[1][0]);
    } else {
        float dots = 0.0f;
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            dots = base_ops::sum::op<float>(dots, warp::sum(acc[r]));
        }
        folded = base_types::convertor<int, float>::convert(dots);
    }
    g.out[{0, 0, int(blockIdx.x), int(threadIdx.x)}] = folded;
}

// Consume variant `C` at `ROWS` rows a warp and `CTAS` CTAs per SM: the shape rule, the attribute
// and the launch.
template <int C, int ROWS, int CTAS>
int launch_consume(const consume_globals& G, uint32_t n, uint32_t k, void* stream) {
    using P = consume<C, ROWS, CTAS>;
    if (C >= 3 && k > uint32_t(P::rc * CHUNK)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const cudaError_t set = cudaFuncSetAttribute(
        tma_consume_probe_ker<C, ROWS, CTAS>, cudaFuncAttributeMaxDynamicSharedMemorySize, P::smem);
    if (set != cudaSuccess) {
        return static_cast<int>(set);
    }
    consume_globals args_g = G;
    void* args[] = {&args_g};
    const uint32_t block_rows = uint32_t(WARPS * ROWS);
    const dim3 grid((n + block_rows - 1) / block_rows), block(NUM_THREADS);
    return static_cast<int>(cudaLaunchKernel(tma_consume_probe_ker<C, ROWS, CTAS>, grid, block,
                                             args, P::smem, static_cast<cudaStream_t>(stream)));
}

// The CTAs per SM the runtime gives consume variant `C` at `ROWS` and `CTAS`, into `blocks`.
template <int C, int ROWS, int CTAS>
int occupancy_consume(int* blocks) {
    using P = consume<C, ROWS, CTAS>;
    const cudaError_t set = cudaFuncSetAttribute(
        tma_consume_probe_ker<C, ROWS, CTAS>, cudaFuncAttributeMaxDynamicSharedMemorySize, P::smem);
    if (set != cudaSuccess) {
        return static_cast<int>(set);
    }
    return static_cast<int>(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        blocks, tma_consume_probe_ker<C, ROWS, CTAS>, NUM_THREADS, P::smem));
}

// The consume instance at run-time `c`, `rows` and `ctas` (already checked): `f` called with a
// `std::integral_constant` each for `C`, `ROWS` and `CTAS`.
template <typename F>
int consume_dispatch(uint32_t c, uint32_t rows, uint32_t ctas, const F& f) {
    using std::integral_constant;
    const auto at = [&](auto cc) {
        if (rows == 3) {
            return ctas == 2 ? f(cc, integral_constant<int, 3>{}, integral_constant<int, 2>{})
                             : f(cc, integral_constant<int, 3>{}, integral_constant<int, 3>{});
        }
        return ctas == 2 ? f(cc, integral_constant<int, 4>{}, integral_constant<int, 2>{})
                         : f(cc, integral_constant<int, 4>{}, integral_constant<int, 3>{});
    };
    switch (c) {
        case 0:
            return at(integral_constant<int, 0>{});
        case 1:
            return at(integral_constant<int, 1>{});
        case 2:
            return at(integral_constant<int, 2>{});
        case 3:
            return at(integral_constant<int, 3>{});
        default:
            return at(integral_constant<int, 4>{});
    }
}

// The CTAs per SM the runtime gives ring variant `V`, into `blocks`.
template <typename V>
int occupancy_ring(int* blocks) {
    const cudaError_t set = cudaFuncSetAttribute(
        tma_ring_probe_ker<V>, cudaFuncAttributeMaxDynamicSharedMemorySize, V::smem);
    if (set != cudaSuccess) {
        return static_cast<int>(set);
    }
    return static_cast<int>(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        blocks, tma_ring_probe_ker<V>, NUM_THREADS, V::smem));
}

}  // namespace

// Test support for `tests/gemv_fp8_ring_probe.rs` only: stream `b [n, k]` e4m3 through variant
// `variant` (0 `vec256_r3`, 1 `vec256`, 2 `pair512`, 3 `tile4k`, 4 `page16k`) and store each
// thread's fold into `out` (`⌈n / (8 · rows)⌉ · 256` f32, `rows` 3 for variant 0 and 4 for the
// others; `16 · n + 256` f32 always suffice). `n` and `k` non-zero and at most INT_MAX / 2, `k` a
// multiple of the variant's columns a box (256, or 1,024 for variant 4), `n` a multiple of 32 for
// variants 2 to 4; anything else returns SCRATCHY_CUDA_ERR_SHAPE before any pointer is touched.
// `b` 16-byte aligned. It sets its instance's shared-memory attribute on every call (no worker
// launches it). 0 = launched; asynchronous on `stream`.
extern "C" int scratchy_cuda_tma_ring_probe(const void* b, void* out, uint32_t n, uint32_t k,
                                            uint32_t variant,
                                            void* stream /* cudaStream_t; null = default */) {
    if (n == 0 || k == 0 || n > uint32_t(INT_MAX / 2) || k > uint32_t(INT_MAX / 2) || variant > 4) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const uint32_t rows = variant == 0 ? 3 : 4;
        const uint32_t ctas = (n + uint32_t(WARPS) * rows - 1) / (uint32_t(WARPS) * rows);
        weight_gl bg{static_cast<fp8e4m3*>(const_cast<void*>(b)), nullptr, nullptr, size_t(n),
                     size_t(k)};
        out_gl og{static_cast<float*>(out), nullptr, nullptr, size_t(ctas), size_t(NUM_THREADS)};
        const probe_globals G{bg, og, int(n), int(k)};
        switch (variant) {
            case 0:
                return launch<vec256_r3>(G, n, k, stream);
            case 1:
                return launch<vec256>(G, n, k, stream);
            case 2:
                return launch<pair512>(G, n, k, stream);
            case 3:
                return launch<tile4k>(G, n, k, stream);
            default:
                return launch<page16k>(G, n, k, stream);
        }
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// Test support for `tests/gemv_fp8_ring_probe.rs` only (FP8 S8 W2d): the consume probe. Stream
// `b [n, k]` e4m3 through consume variant `consume` (0 (a) wait, 1 (b) bytes, 2 (c) widen, 3 (d)
// dot, 4 (e) full) at `rows` (3 or 4) rows a warp and `ctas` (2 or 3) CTAs per SM, with `a [1, k]`
// bf16 the activation row of (d) and (e), and store each thread's fold into `out`
// (`⌈n / (8 · rows)⌉ · 256` int; `11 · n + 256` always suffice). The folds: (a) the bytes read as
// e4m3 values, (b) the 32-bit words read, added as int (wrapping), (c) to (e) the row sums each
// lane holds, added over its rows. `n` and `k` non-zero and at most INT_MAX / 2, `k` a multiple of
// 256, at most 8,192 for (d) and (e) at 3 CTAs and 17,408 otherwise; anything else returns
// SCRATCHY_CUDA_ERR_SHAPE before any pointer is touched. `a` and `b` 16-byte aligned. It sets its
// instance's shared-memory attribute on every call. 0 = launched; asynchronous on `stream`.
extern "C" int scratchy_cuda_tma_consume_probe(const void* b, const void* a, void* out, uint32_t n,
                                               uint32_t k, uint32_t consume, uint32_t rows,
                                               uint32_t ctas,
                                               void* stream /* cudaStream_t; null = default */) {
    if (n == 0 || k == 0 || n > uint32_t(INT_MAX / 2) || k % uint32_t(CHUNK) != 0 ||
        k > uint32_t(68 * CHUNK) || consume > 4 || (rows != 3 && rows != 4) ||
        (ctas != 2 && ctas != 3)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const uint32_t ctas_n = (n + uint32_t(WARPS) * rows - 1) / (uint32_t(WARPS) * rows);
        fp8_gl bg{static_cast<fp8e4m3*>(const_cast<void*>(b)), nullptr, nullptr, size_t(n),
                  size_t(k)};
        word_gl wg{static_cast<int*>(const_cast<void*>(b)), nullptr, nullptr, size_t(n),
                   size_t(k / 4)};
        in_gl ag{static_cast<bf16*>(const_cast<void*>(a)), nullptr, nullptr, size_t(1), size_t(k)};
        count_gl og{static_cast<int*>(out), nullptr, nullptr, size_t(ctas_n), size_t(NUM_THREADS)};
        const consume_globals G{bg, wg, ag, og, int(n), int(k)};
        return consume_dispatch(consume, rows, ctas, [&](auto c, auto r, auto t) {
            return launch_consume<decltype(c)::value, decltype(r)::value, decltype(t)::value>(
                G, n, k, stream);
        });
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// Test support for `tests/gemv_fp8_ring_probe.rs` only: the CTAs per SM the runtime gives a probe
// instance, into `*blocks` (a host pointer): ring variant `variant` (0 to 4) when `consume` is 5,
// else consume variant `consume` at `rows` and `ctas` as `scratchy_cuda_tma_consume_probe`'s.
// 0 = done; SCRATCHY_CUDA_ERR_SHAPE for an argument out of range; else the `cudaError_t`.
extern "C" int scratchy_cuda_tma_probe_occupancy(uint32_t variant, uint32_t consume, uint32_t rows,
                                                 uint32_t ctas, int* blocks) {
    if (blocks == nullptr || variant > 4 || consume > 5 ||
        (consume < 5 && ((rows != 3 && rows != 4) || (ctas != 2 && ctas != 3)))) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    if (consume < 5) {
        return consume_dispatch(consume, rows, ctas, [&](auto c, auto r, auto t) {
            return occupancy_consume<decltype(c)::value, decltype(r)::value, decltype(t)::value>(
                blocks);
        });
    }
    switch (variant) {
        case 0:
            return occupancy_ring<vec256_r3>(blocks);
        case 1:
            return occupancy_ring<vec256>(blocks);
        case 2:
            return occupancy_ring<pair512>(blocks);
        case 3:
            return occupancy_ring<tile4k>(blocks);
        default:
            return occupancy_ring<page16k>(blocks);
    }
}

// ── The ceiling probe's pure stream in the tensor-core GEMV's geometry (/tmp/d-cap.md §3 P-A) ──
//
// `gemv_fp8_tc.cu`'s grid and stream-K runs (`CTAS` CTAs; CTA `c` takes units
// `[⌊c · units / CTAS⌋, ⌊(c + 1) · units / CTAS⌋)` of `tiles × k_steps`), one producer warp (warp
// 0 of the second warpgroup) and one consumer warpgroup that only waits each stage, reads one byte
// per lane (TK's `st` accessor, then TK's element convertor and `base_ops::sum`) and releases it.
// The variants differ in the box and its stage:
//   0 A1 `b64`: B `st_fp8e4m3<128, 64>` (8 KB, 64-byte rows), 12 stages: the TC kernel's B box;
//   1 A2 `b64+a`: A1 plus an `st_fp8e4m3<64, 64>` A box from a `[1, k]` map per stage (63 of its
//     64 rows TMA's zero fill): the TC kernel's stage as it runs;
//   2 A3 `b128`: B `st_fp8e4m3<128, 128>` (16 KB, 128-byte rows), 8 stages;
//   3 A4 `b128+a`: A3 plus an `st_fp8e4m3<64, 128>` A box, 8 stages;
//   4 A5 `b256`: B `st_fp8e4m3<64, 256>` (16 KB, 256-byte rows), 8 stages;
//   5 A6 `b64x2cta`: A1 at 264 CTAs, 2 per SM, 6 stages each.
// With every weight byte 1.0 each consumer thread folds one 1.0 per unit, so the folds add up to
// exactly `128 · units`.

namespace {

template <int ROWS, int K_STEP, bool A, int STAGES_, int CTAS, int MIN_BLOCKS>
struct stream_variant {
    using b_box = st_fp8e4m3<ROWS, K_STEP>;
    using a_box = st_fp8e4m3<64, K_STEP>;
    static constexpr int rows = ROWS;
    static constexpr int k_step = K_STEP;
    static constexpr bool a = A;
    static constexpr int stages = STAGES_;
    static constexpr int ctas = CTAS;
    static constexpr int min_blocks = MIN_BLOCKS;
    static constexpr int smem =
        STAGES_ * int(sizeof(b_box) + (A ? sizeof(a_box) : 0)) + 1024;
    static_assert(MIN_BLOCKS * smem <= kittens::MAX_SHARED_MEMORY, "the CTAs per SM fit");
};

using a1_b64 = stream_variant<128, 64, false, 12, 132, 1>;
using a2_b64a = stream_variant<128, 64, true, 12, 132, 1>;
using a3_b128 = stream_variant<128, 128, false, 8, 132, 1>;
using a4_b128a = stream_variant<128, 128, true, 8, 132, 1>;
using a5_b256 = stream_variant<64, 256, false, 8, 132, 1>;
using a6_b64x2 = stream_variant<128, 64, false, 6, 264, 2>;

using stream_b_gl = gl<fp8e4m3, 1, 1, -1, -1, typename a1_b64::b_box, typename a3_b128::b_box,
                       typename a5_b256::b_box>;
using stream_a_gl = gl<fp8e4m3, 1, 1, -1, -1, typename a1_b64::a_box, typename a3_b128::a_box>;

struct stream_globals {
    stream_b_gl b;
    stream_a_gl a;
    out_gl out;
    int tiles;
    int k_steps;
    int ctas;
};

constexpr int STREAM_THREADS = 2 * kittens::WARPGROUP_THREADS;

__device__ inline long long stream_run_start(int c, long long units, int ctas) {
    return units * c / ctas;
}

template <typename V>
__global__ __launch_bounds__(STREAM_THREADS, V::min_blocks) void tc_stream_probe_ker(
    const __grid_constant__ stream_globals g) {
    using b_box = typename V::b_box;
    using a_box = typename V::a_box;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    b_box(&b_s)[V::stages] = al.allocate<b_box, V::stages>();
    __shared__ kittens::semaphore arrived[V::stages], done[V::stages];
    const long long units = (long long)g.tiles * g.k_steps;
    const int c = blockIdx.x;
    const long long u0 = stream_run_start(c, units, g.ctas);
    const int len = int(stream_run_start(c + 1, units, g.ctas) - u0);
    if (threadIdx.x == 0) {
        for (int s = 0; s < V::stages; s++) {
            init_semaphore(arrived[s], 0, 1);
            init_semaphore(done[s], 1, 0);
        }
    }
    __syncthreads();
    float acc = 0.0f;
    if (warpgroup::groupid() == 1) {
        if (warpgroup::warpid() == 0) {
            if constexpr (V::a) {
                a_box(&a_s)[V::stages] = al.allocate<a_box, V::stages>();
                for (int i = 0; i < len; i++) {
                    const long long u = u0 + i;
                    const int t = int(u / g.k_steps), j = int(u % g.k_steps), s = i % V::stages;
                    if (i >= V::stages) {
                        wait(done[s], ((i - V::stages) / V::stages) % 2);
                    }
                    warp::tma::expect_bytes(arrived[s], sizeof(a_box) + sizeof(b_box));
                    warp::tma::load_async(a_s[s], g.a, {0, 0, 0, j}, arrived[s]);
                    warp::tma::load_async(b_s[s], g.b, {0, 0, t, j}, arrived[s]);
                }
            } else {
                for (int i = 0; i < len; i++) {
                    const long long u = u0 + i;
                    const int t = int(u / g.k_steps), j = int(u % g.k_steps), s = i % V::stages;
                    if (i >= V::stages) {
                        wait(done[s], ((i - V::stages) / V::stages) % 2);
                    }
                    warp::tma::expect_bytes(arrived[s], sizeof(b_box));
                    warp::tma::load_async(b_s[s], g.b, {0, 0, t, j}, arrived[s]);
                }
            }
        }
    } else {
        // Thread `x` of the consumer warpgroup reads byte `(x % rows, x / rows)` of each stage.
        const int x = int(threadIdx.x);
        for (int i = 0; i < len; i++) {
            const int s = i % V::stages;
            wait(arrived[s], (i / V::stages) % 2);
            const fp8e4m3 v = b_s[s][{x % V::rows, x / V::rows}];
            acc = base_ops::sum::op<float>(acc, base_types::convertor<float, fp8e4m3>::convert(v));
            // Every thread has read the stage before it is released.
            warpgroup::sync(1);
            if (warpgroup::laneid() == 0) {
                arrive(done[s], 1);
            }
        }
    }
    g.out[{0, 0, c, int(threadIdx.x)}] = acc;
}

template <typename V>
int launch_stream(stream_globals G, uint32_t n, uint32_t k, void* stream) {
    if (n % uint32_t(V::rows) != 0 || k % uint32_t(V::k_step) != 0) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const cudaError_t set = cudaFuncSetAttribute(
        tc_stream_probe_ker<V>, cudaFuncAttributeMaxDynamicSharedMemorySize, V::smem);
    if (set != cudaSuccess) {
        return static_cast<int>(set);
    }
    G.tiles = int(n / uint32_t(V::rows));
    G.k_steps = int(k / uint32_t(V::k_step));
    const long long units = (long long)G.tiles * G.k_steps;
    G.ctas = units < V::ctas ? int(units) : V::ctas;
    void* args[] = {&G};
    return static_cast<int>(cudaLaunchKernel(tc_stream_probe_ker<V>, dim3(G.ctas),
                                             dim3(STREAM_THREADS), args, V::smem,
                                             static_cast<cudaStream_t>(stream)));
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = refused, nothing launched; SCRATCHY_CUDA_ERR_HOST = a C++
// exception was caught; any other value = the cudaError_t of the attribute call or the launch.
//
// The ceiling probe's pure stream, variant `variant` (0 to 5: A1 to A6 above) over `b [n, k]`
// e4m3, its A box (A2, A4) from `a [1, k]` e4m3; `out [264, 256]` f32 receives every thread's fold.
// `n` a non-zero multiple of the variant's box rows (128, or 64 for A5) and `k` of its K step (64,
// 128 or 256), both at most INT_MAX / 2. It sets its instance's shared-memory attribute on every
// call (no worker launches it); asynchronous on `stream`.
extern "C" int scratchy_cuda_tc_stream_probe(const void* b, const void* a, void* out, uint32_t n,
                                             uint32_t k, uint32_t variant,
                                             void* stream /* cudaStream_t; null = default */) {
    if (n == 0 || k == 0 || n > uint32_t(INT_MAX / 2) || k > uint32_t(INT_MAX / 2) || variant > 5) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        stream_globals G{
            stream_b_gl{static_cast<fp8e4m3*>(const_cast<void*>(b)), nullptr, nullptr, size_t(n),
                        size_t(k)},
            stream_a_gl{static_cast<fp8e4m3*>(const_cast<void*>(a)), nullptr, nullptr, size_t(1),
                        size_t(k)},
            out_gl{static_cast<float*>(out), nullptr, nullptr, size_t(264),
                   size_t(STREAM_THREADS)},
            0, 0, 0};
        switch (variant) {
            case 0: return launch_stream<a1_b64>(G, n, k, stream);
            case 1: return launch_stream<a2_b64a>(G, n, k, stream);
            case 2: return launch_stream<a3_b128>(G, n, k, stream);
            case 3: return launch_stream<a4_b128a>(G, n, k, stream);
            case 4: return launch_stream<a5_b256>(G, n, k, stream);
            default: return launch_stream<a6_b64x2>(G, n, k, stream);
        }
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
