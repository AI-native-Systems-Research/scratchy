// SPDX-License-Identifier: Apache-2.0
//
// Causal grouped-query attention of one prefill bucket over the paged e4m3 KV cache, for the H100
// at head_dim 512 (Gemma 4's global layers; FP8 S5): `attn_prefill_d512_bf16.cu`, which stays as
// its test-only reference, with K and V read as e4m3 and widened to bf16 in shared memory before
// the same math. The cache is one layer's `[num_blocks, kv_heads, 64, 512]` e4m3, one byte an
// element, at a fixed scale of 1.0 (`kv_append_fp8.cu`); Q and O are the tape's token-major
// `[rows, heads · 512]` bf16.
//
// What differs from the bf16 kernel, and why:
//   - K and V land as two `st_fp8e4m3<64, 256>` halves each (16 KiB a half, half the TMA bytes).
//     Warpgroup c widens half c into its own bf16 half W[c], 64 columns at a time:
//     `warpgroup::load` of an `rt_fp8e4m3<16, 64>` through a `subtile<64, 64>` view (TK's 8-bit
//     `ldmatrix` path), `warp::copy` to `rt_bf<16, 64>` (exact: every e4m3 value is a bf16 value),
//     `warpgroup::store` back through W[c]'s view. One bf16 pair for both K and V keeps the dynamic
//     shared memory at today's 197,632 bytes (separate K and V pairs would not fit).
//   - The order per tile: K half c into W[c], `warpgroup::mma_fence()` (TK's `wgmma.fence` +
//     `fence.proxy.async.shared::cta`: the widen writes through the generic proxy, WGMMA reads
//     through the async one), the consumers' barrier 1; both warpgroups' Q·Kᵀ over W[0] and W[1],
//     its wait, `k_done`, and barrier 1 again (the other warpgroup has stopped reading W[c] as K);
//     then V half c into W[c], the fence, the warpgroup's own named barrier c + 4 (only c reads it
//     for P·V), and P·V. The next tile's K widen into W[c] is safe: only warpgroup c read it last.
//     Barriers 4 and 5 also guard the O store after the loop; the uses never overlap. So no
//     inline PTX.
//   - The V rows of slots at or past kv_len (an unwritten e4m3 byte may be NaN) are zeroed in
//     registers during the widen, `warp::lower_fill` with +0, for the bf16 kernel's shared-memory
//     loop: the same +0 rows. K needs nothing, as there: `right_fill` overwrites those scores.
//   - After the widen the math is the bf16 kernel's, line for line, on bf16 tiles, so its output
//     is the bf16 kernel's on the widened cache, bit for bit (`tests/attention_fp8kv.rs`).
//
// The bf16 kernel's description of the column split, the masks, the online softmax, the window
// and the O store over Q holds unchanged (`attn_prefill_d512_bf16.cu`, itself `attn_prefill_bf16.cu`
// adapted from ThunderKittens `kernels/attention/mha_h100/mha_h100.cu` at 67845f5f, MIT, see
// `thunderkittens/LICENSE`).
//
// Shared memory: Q 2 halves (64 KiB), e4m3 K and V 2 halves each (64 KiB), the bf16 pair W (64
// KiB), plus 1,024 bytes of alignment slack = 197,632 bytes of dynamic shared memory, set once by
// `scratchy_cuda_attn_prefill_d512_fp8kv_prepare` (in `PREPARE`). The 5 semaphores are static.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cmath>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::ATTN_D512_BF16_HEAD_DIM`, as two halves of the D = 256 kernel's tile width.
constexpr int HEAD_DIM = 512;
constexpr int HALF_DIM = 256;
constexpr int HALVES = HEAD_DIM / HALF_DIM;
// Query rows per CTA (both consumer warpgroups on the same rows), and K/V rows per tile.
constexpr int QO_HEIGHT = 64;
constexpr int KV_HEIGHT = 64;
constexpr int CONSUMER_WARPGROUPS = 2;
static_assert(CONSUMER_WARPGROUPS == HALVES, "one output half per consumer warpgroup");
constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + 1;
constexpr int NUM_WARPS = NUM_WARPGROUPS * kittens::WARPGROUP_WARPS;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;
// `step::ATTN_PREFILL_BF16_ROWS`: the step's rows are a multiple, so a CTA pair covers each 128.
constexpr uint32_t STEP_ROWS = 128;
static_assert(STEP_ROWS % QO_HEIGHT == 0, "whole CTAs per step tile");
// The named barrier over the two consumer warpgroups (ids 4 and 5 are each warpgroup's own, before
// its P·V on its widened V half and at the O store; 0 is `__syncthreads`).
constexpr int CONSUMERS_BARRIER = 1;
using consumers = kittens::group<CONSUMER_WARPGROUPS * kittens::WARPGROUP_WARPS>;
// Token slots per KV block (`step::CUDA_KV_BLOCK_SIZE`): one block is one K (or V) tile.
constexpr uint32_t BLOCK_SIZE = 64;
static_assert(KV_HEIGHT == int(BLOCK_SIZE), "one paged (block, kv head) half is one TMA box");
// Entries of a sequence's block table (`step::CUDA_MAX_BLOCKS_PER_SEQ`): the columns of its view.
constexpr uint32_t MAX_BLOCKS_PER_SEQ = 128;

using half_tile = st_bf<QO_HEIGHT, HALF_DIM>;
static_assert(KV_HEIGHT == QO_HEIGHT, "Q, O and the widened K and V halves are one tile type");
// A K or V half as cached.
using kv8_half = st_fp8e4m3<KV_HEIGHT, HALF_DIM>;
// Columns per widen step: one `rt_fp8e4m3<16, 64>` and one `rt_bf<16, 64>`, 24 registers a thread.
constexpr int WIDEN_COLS = 64;
constexpr int WIDEN_CHUNKS = HALF_DIM / WIDEN_COLS;

// `axis = 1`: a Q (or O) half's rows are 64 consecutive tokens of one head of the token-major
// `[1, rows, heads, 512]` view, its columns the half's 256.
using qo_gl = gl<bf16, 1, -1, -1, HEAD_DIM, tma::descriptor<half_tile, 1>>;
using kv_gl = gl<fp8e4m3, -1, -1, KV_HEIGHT, HEAD_DIM, kv8_half>;
// An index region, `[1, entries]` int with no TMA type: read element by element.
using index_gl = gl<int, 1, 1, -1, -1>;

struct attn_globals {
    qo_gl q;
    kv_gl k;
    kv_gl v;
    qo_gl o;
    // `[1, MAX_BLOCKS_PER_SEQ]`.
    index_gl block_table;
    // `[1, 2]`: `[P, L]`.
    index_gl sequence;
    // `scale · log2(e)`.
    float scale_log2;
    // `q_heads / kv_heads`.
    int hr;
    // The sliding window `W`, a multiple of 64; 0 for full causal attention.
    int window;
};

constexpr size_t SMEM_TILES = 2 * HALVES * sizeof(half_tile) + 2 * HALVES * sizeof(kv8_half);
static_assert(SMEM_TILES == 196608,
              "2 Q and 2 widened bf16 halves of [64, 256], 2 e4m3 K and V halves");
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int ATTN_SMEM_BYTES = int(SMEM_TILES) + 1024;
static_assert(ATTN_SMEM_BYTES == 197632, "the D = 256 prefill's dynamic shared memory");
static_assert(ATTN_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

// This warpgroup's widen of the e4m3 half `src` into the bf16 half `dst`, `WIDEN_COLS` columns at a
// time; rows at or past `zero_from` (a tile's first unwritten slot; 64 for none) are +0. Warp w of
// the warpgroup holds rows 16w .. 16w + 16 of each chunk (`warpgroup::load`'s split), so its fill
// starts at `zero_from − 16w`, the whole chunk when that is not above 0.
__device__ inline void widen_kv(half_tile& dst, kv8_half& src, int zero_from) {
#pragma unroll
    for (int c = 0; c < WIDEN_CHUNKS; c++) {
        rt_fp8e4m3<16, WIDEN_COLS> q;
        rt_bf<16, WIDEN_COLS> x;
        warpgroup::load(q, src.subtile<KV_HEIGHT, WIDEN_COLS>({0, c}));
        warp::copy(x, q);
        warp::lower_fill(x, x, zero_from - 16 * warpgroup::warpid(),
                         kittens::base_types::constants<bf16>::zero());
        auto d = dst.subtile<KV_HEIGHT, WIDEN_COLS>({0, c});
        warpgroup::store(d, x);
    }
}

template <bool WINDOWED>
__global__ __launch_bounds__(NUM_THREADS, 1) void attn_prefill_d512_fp8kv_ker(
    const __grid_constant__ attn_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    const int warpid = kittens::warpid(), warpgroupid = warpid / kittens::WARPGROUP_WARPS;

    half_tile(&q_smem)[HALVES] = al.allocate<half_tile, HALVES>();
    kv8_half(&k_smem)[HALVES] = al.allocate<kv8_half, HALVES>();
    kv8_half(&v_smem)[HALVES] = al.allocate<kv8_half, HALVES>();
    // Half c widened by warpgroup c: K for both warpgroups' Q·Kᵀ, then V for its own P·V.
    half_tile(&kv_bf)[HALVES] = al.allocate<half_tile, HALVES>();

    const int q_head = blockIdx.y;
    const int kv_head = q_head / g.hr;
    // The CTA's query-row tile, in units of QO_HEIGHT rows.
    const int row_tile = blockIdx.x;
    const int prefix_len = g.sequence[{0, 0, 0, 0}];
    const int kv_len = prefix_len + g.sequence[{0, 0, 0, 1}];
    // The last key any of this CTA's rows may attend to: key 0 is allowed for every row, so the
    // loop reads at least one tile.
    const int row_last = row_tile * QO_HEIGHT + QO_HEIGHT - 1;
    const int key_last = min(prefix_len + row_last, kv_len - 1);
    const int kv_iters = key_last / KV_HEIGHT + 1;
    // The first tile the loop reads: 0 for full causal attention; with a window, the tile of the
    // CTA's lowest window floor (its first row's).
    int kv_first = 0;
    if constexpr (WINDOWED) {
        kv_first = max(0, prefix_len + row_tile * QO_HEIGHT - (g.window - 1)) / KV_HEIGHT;
    }

    __shared__ kittens::semaphore q_arrived, k_arrived, v_arrived, k_done, v_done;
    if (threadIdx.x == 0) {
        init_semaphore(q_arrived, 0, 1);
        init_semaphore(k_arrived, 0, 1);
        init_semaphore(v_arrived, 0, 1);
        init_semaphore(k_done, CONSUMER_WARPGROUPS, 0);
        init_semaphore(v_done, CONSUMER_WARPGROUPS, 0);
        tma::expect_bytes(q_arrived, sizeof(q_smem));
        for (int c = 0; c < HALVES; c++) {
            // Token tile `row_tile` of head `q_head`, half c, as `[1, rows, heads, 512]`.
            coord<half_tile> at = {0, row_tile, q_head, c};
            tma::load_async<1, cache_policy::NORMAL>(q_smem[c], g.q, at, q_arrived);
        }
    }
    __syncthreads();

    if (warpgroupid == NUM_WARPGROUPS - 1) {
        // The producer: one warp loads the loop's `t`-th tile, K once both consumers have read
        // tile t − 1's K, V once both have read tile t − 1's V.
        warpgroup::decrease_registers<24>();
        if (warpid == NUM_WARPS - 4) {
            for (int j = kv_first; j < kv_iters; j++) {
                const int t = j - kv_first;
                const int block = g.block_table[{0, 0, 0, j}];
                if (t >= 1) {
                    wait(k_done, (t - 1) % 2);
                }
                warp::tma::expect_bytes(k_arrived, sizeof(k_smem));
                for (int c = 0; c < HALVES; c++) {
                    coord<kv8_half> at = {block, kv_head, 0, c};
                    warp::tma::load_async(k_smem[c], g.k, at, k_arrived);
                }
                if (t >= 1) {
                    wait(v_done, (t - 1) % 2);
                }
                warp::tma::expect_bytes(v_arrived, sizeof(v_smem));
                for (int c = 0; c < HALVES; c++) {
                    coord<kv8_half> at = {block, kv_head, 0, c};
                    warp::tma::load_async(v_smem[c], g.v, at, v_arrived);
                }
            }
        }
    } else {
        warpgroup::increase_registers<240>();

        rt_fl<16, KV_HEIGHT> att_block;
        rt_bf<16, KV_HEIGHT> att_block_mma;
        // This warpgroup's half of the output columns.
        rt_fl<16, HALF_DIM> o_reg;
        // `max_vec_last` holds the max before a tile, then (after the rescale) the tile's rescale.
        col_vec<rt_fl<16, KV_HEIGHT>> max_vec, norm_vec, max_vec_last, max_vec_scaled;

        warp::neg_infty(max_vec);
        warp::zero(norm_vec);
        warp::zero(o_reg);

        // This warp's 16 query rows, as absolute positions: row `row0 + r` is at `P + row0 + r`.
        // Both warpgroups' warp w hold the same rows.
        const int row0 = row_tile * QO_HEIGHT + warpgroup::warpid() * 16;
        constexpr float NEG_INF = kittens::base_types::constants<float>::neg_infty();
        const int c = warpgroupid;

        wait(q_arrived, 0);

        for (int j = kv_first; j < kv_iters; j++) {
            const int t = j - kv_first;
            const int phase = t % 2;
            wait(k_arrived, phase);
            // Half c of K widened into this warpgroup's bf16 half, whose last reader (its own
            // previous P·V) it has waited on; both halves visible to the tensor cores.
            widen_kv(kv_bf[c], k_smem[c], KV_HEIGHT);
            warpgroup::mma_fence();
            consumers::sync(CONSUMERS_BARRIER);
            // S over the 512 lanes: the low halves, then the high halves accumulated onto them.
            warpgroup::mm_ABt(att_block, q_smem[0], kv_bf[0]);
            warpgroup::mma_ABt(att_block, q_smem[1], kv_bf[1]);

            // The running max before this tile, unscaled (the rescale below reads it).
            warp::copy(max_vec_last, max_vec);

            // The WGMMAs write `att_block` asynchronously: wait before reading it. After the wait
            // this warpgroup has read K: release it to the producer.
            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) {
                arrive(k_done, 1);
            }
            // Both warpgroups past their Q·Kᵀ: each bf16 half is free for its own V.
            consumers::sync(CONSUMERS_BARRIER);

            // The D = 256 kernel's masks and online softmax, line for line.
            const int k0 = j * KV_HEIGHT;
            warp::tril(att_block, att_block, prefix_len + row0 - k0, NEG_INF);
            warp::right_fill(att_block, att_block, kv_len - k0, NEG_INF);
            if constexpr (WINDOWED) {
                warp::triu(att_block, att_block, prefix_len + row0 - k0 - (g.window - 1), NEG_INF);
            }

            warp::row_max(max_vec, att_block, max_vec);
            warp::mul(att_block, att_block, g.scale_log2);
            if constexpr (WINDOWED) {
                // A register vector, so `unary_op`, not the tile map `unary_map`.
                warp::unary_op<scratchy::ops::neg_inf_to_zero>(max_vec_scaled, max_vec);
                warp::mul(max_vec_scaled, max_vec_scaled, g.scale_log2);
            } else {
                warp::mul(max_vec_scaled, max_vec, g.scale_log2);
            }
            warp::sub_row(att_block, att_block, max_vec_scaled);
            warp::exp2(att_block, att_block);
            // `attn_prefill_bf16.cu` gives the reason for `rescale_exact1`.
            scratchy::ops::rv_ternary_map<scratchy::ops::rescale_exact1>(
                max_vec_last, max_vec_last, max_vec, g.scale_log2);
            warp::mul(norm_vec, norm_vec, max_vec_last);
            warp::row_sum(norm_vec, att_block, norm_vec);
            warp::copy(att_block_mma, att_block);
            warp::mul_row(o_reg, o_reg, max_vec_last);

            wait(v_arrived, phase);
            // Half c of V widened over K; only this warpgroup reads it. A partial last tile's rows
            // of the slots at or past kv_len, which no append wrote, are +0 (0 · NaN = NaN, #791).
            widen_kv(kv_bf[c], v_smem[c], kv_len - k0);
            warpgroup::mma_fence();
            warpgroup::sync(c + 4);
            warpgroup::mma_AB(o_reg, att_block_mma, kv_bf[c]);
            // Wait for the WGMMA to finish reading V before releasing it to the producer.
            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) {
                arrive(v_done, 1);
            }
        }

        if constexpr (WINDOWED) {
            warp::unary_op<scratchy::ops::zero_to_one>(norm_vec, norm_vec);
        }
        warp::div_row(o_reg, o_reg, norm_vec);
        // Both warpgroups are past their last read of both Q tiles: tile c is this half's O.
        consumers::sync(CONSUMERS_BARRIER);
        warpgroup::store(q_smem[c], o_reg);
        warpgroup::sync(c + 4);
        if (warpgroup::warpid() == 0) {
            coord<half_tile> at = {0, row_tile, q_head, c};
            warp::tma::store_async<1, cache_policy::NORMAL>(g.o, q_smem[c], at);
        }
        warp::tma::store_async_wait();
    }
}

}  // namespace

// The kernel's one-time setup: allow both instantiations `ATTN_SMEM_BYTES` of dynamic shared
// memory on the current device. Run once per device before any launch (`kernels::prepare_all`).
//
// 0 = done; any other value = the cudaError_t of setting the attribute, the unwindowed one's first.
extern "C" int scratchy_cuda_attn_prefill_d512_fp8kv_prepare(void) {
    const cudaError_t causal =
        cudaFuncSetAttribute(attn_prefill_d512_fp8kv_ker<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, ATTN_SMEM_BYTES);
    if (causal != cudaSuccess) {
        return static_cast<int>(causal);
    }
    return static_cast<int>(cudaFuncSetAttribute(attn_prefill_d512_fp8kv_ker<true>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 ATTN_SMEM_BYTES));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value
// = the cudaError_t cudaLaunchKernel returned for THIS launch.
//
// Refused before any pointer is touched: `head_dim` not 512; `rows` zero or not a multiple of 128
// (`step::ATTN_PREFILL_BF16_ROWS`); `q_heads` or `kv_heads` zero, or `q_heads` not a multiple of
// `kv_heads`; `block_size` not 64; `num_blocks` zero; `scale` not finite or not above 0; `window`
// not a multiple of 64; a dimension or the window above INT_MAX.
//
// `q` and `out` must be 16-byte-aligned device pointers to `rows · q_heads · 512` bf16, `out`
// overlapping no input; `k_cache` and `v_cache` to one layer's `num_blocks · kv_heads · 64 · 512`
// e4m3 bytes; `block_table` and `sequence` as `scratchy_cuda_attn_prefill_bf16`'s.
// `scratchy_cuda_attn_prefill_d512_fp8kv_prepare` must have succeeded on the current device. The
// launch is asynchronous on `stream`.
extern "C" int scratchy_cuda_attn_prefill_d512_fp8kv(
    const void* q, const void* k_cache, const void* v_cache, const void* block_table,
    const void* sequence, void* out, uint32_t rows, uint32_t q_heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t block_size, uint32_t num_blocks, float scale, uint32_t window,
    void* stream /* cudaStream_t; null = default */) {
    if (head_dim != uint32_t(HEAD_DIM) || rows == 0 || rows % STEP_ROWS != 0 || q_heads == 0 ||
        kv_heads == 0 || q_heads % kv_heads != 0 || block_size != BLOCK_SIZE || num_blocks == 0 ||
        !std::isfinite(scale) || !(scale > 0.0f) || window % BLOCK_SIZE != 0 ||
        rows > uint32_t(INT_MAX) || q_heads > uint32_t(INT_MAX) ||
        num_blocks > uint32_t(INT_MAX) || window > uint32_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        bf16* qp = static_cast<bf16*>(const_cast<void*>(q));
        bf16* op = static_cast<bf16*>(out);
        fp8e4m3* kp = static_cast<fp8e4m3*>(const_cast<void*>(k_cache));
        fp8e4m3* vp = static_cast<fp8e4m3*>(const_cast<void*>(v_cache));
        qo_gl qg{qp, nullptr, size_t(rows), size_t(q_heads), nullptr};
        qo_gl og{op, nullptr, size_t(rows), size_t(q_heads), nullptr};
        kv_gl kg{kp, size_t(num_blocks), size_t(kv_heads), nullptr, nullptr};
        kv_gl vg{vp, size_t(num_blocks), size_t(kv_heads), nullptr, nullptr};
        index_gl tg{static_cast<int*>(const_cast<void*>(block_table)), nullptr, nullptr, size_t(1),
                    size_t(MAX_BLOCKS_PER_SEQ)};
        index_gl sg{static_cast<int*>(const_cast<void*>(sequence)), nullptr, nullptr, size_t(1),
                    size_t(2)};
        attn_globals G{qg, kg, vg, og, tg, sg, scale * 1.44269504089f, int(q_heads / kv_heads),
                       int(window)};
        void* args[] = {&G};
        const dim3 grid(rows / uint32_t(QO_HEIGHT), q_heads, 1), block(NUM_THREADS);
        auto* ker =
            window == 0 ? attn_prefill_d512_fp8kv_ker<false> : attn_prefill_d512_fp8kv_ker<true>;
        return static_cast<int>(cudaLaunchKernel(ker, grid, block, args, ATTN_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
