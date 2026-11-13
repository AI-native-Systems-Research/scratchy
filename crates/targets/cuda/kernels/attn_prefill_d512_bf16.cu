// SPDX-License-Identifier: Apache-2.0
//
// Causal grouped-query attention of one prefill bucket over the paged bf16 KV cache, for the H100
// at head_dim 512 (Gemma 4's global layers). Q and O are the tape's token-major [rows, heads · 512]
// bf16; K and V are read from one layer's paged cache [num_blocks, kv_heads, 64, 512] bf16 through
// the sequence's block table. The math, the masks and the online softmax are
// `attn_prefill_bf16.cu`'s, line for line:
//
//   o[i, h, :] = Σ_j softmax_j(scale · q[i, h, :] · k[j, h/hr, :]) · v[j, h/hr, :]
//
// over the allowed keys j ≤ P + i and j < kv_len = P + L (and j > P + i − W under a window W).
//
// No upstream ThunderKittens kernel exists at D = 512 (`kernels/attention/mha_h100/mha_h100.cu` at
// 67845f5f, MIT, instantiates D = 64 and 128); this one is `attn_prefill_bf16.cu` (itself adapted
// from it) with a column split. What differs from that kernel, and why:
//   - At D = 512 one warp's `rt_fl<16, 512>` accumulator is 256 f32 a thread, past the register
//     file, and two stages of [64, 512] K and V do not fit shared memory. So the CTA's two
//     consumer warpgroups own the SAME 64 query rows (grid `(rows / 64, q_heads)`, two CTAs per
//     128-row step), and warpgroup c keeps only output columns [256c, 256c + 256) in its
//     `rt_fl<16, 256>`, the D = 256 kernel's accumulator, under the same
//     `increase_registers<240>` / `decrease_registers<24>` split.
//   - Every shared tile is the D = 256 kernel's `st_bf<64, 256>`, swizzled; a 512-wide row is two
//     (`[0]` = lanes 0..255, `[1]` = 256..511), TMA boxes at column 0 and 256, never past a column
//     edge. Q is two tiles, K and V one stage of two tiles each.
//   - Scores: each warpgroup computes the whole `S = Q·Kᵀ` over 512 as `mm_ABt(att, q[0], k[0])`
//     then `mma_ABt(att, q[1], k[1])`, then `mma_async_wait`. Both warpgroups issue the same
//     WGMMAs on the same shared tiles in the same order, so their `S`, row max and row sum are the
//     same bits, and P·V needs no merge: warpgroup c runs `mma_AB(o, att_bf, v[c])` on its own V
//     half. `Q·Kᵀ` is computed twice per CTA (a later speed issue).
//   - One K/V stage, with K and V on their own semaphores and their own releases: the producer
//     loads K of tile t + 1 as soon as both warpgroups have waited on their scores of tile t
//     (`k_done`), while they run the softmax and P·V on V of tile t, and V of tile t + 1 once both
//     have finished that P·V (`v_done`).
//   - A partial last tile's V rows [kv_len − k0, 64) are zeroed in both halves by all 256
//     consumer threads, then each warpgroup's `warpgroup::mma_fence()` (TK's
//     `wgmma.fence` + `fence.proxy.async.shared::cta`, `ops/group/mma/warpgroup.cuh:76`) makes
//     the writes visible to the tensor cores' async proxy, and the consumers meet at named barrier
//     1 before either P·V. That is the D = 256 kernel's rule (#791), its fence spelled as TK's.
//   - The O store: both warpgroups read both Q tiles until their last tile's score wait, so the
//     consumers meet at barrier 1 after the loop before warpgroup c stores its half over Q tile
//     c (the D = 256 kernel aliases O onto Q the same way, without the barrier, as each of its
//     warpgroups reads only its own Q tile), and warp 0 of it TMA-stores the box at column 256c.
//   - The window (none on Gemma's global layers, which launch `W = 0`): the D = 256 kernel's
//     `WINDOWED` instantiation, `triu`, the loop's first tile, `neg_inf_to_zero` on the max and
//     `zero_to_one` on the sum, line for line.
//
// Shared memory: 2 Q + 2 K + 2 V tiles of [64, 256] bf16 = 196,608 bytes, plus 1,024 bytes of
// alignment slack = 197,632 bytes of dynamic shared memory (the D = 256 prefill's figure), set
// once by `scratchy_cuda_attn_prefill_d512_bf16_prepare` (in `PREPARE`). The 5 semaphores are
// static.

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
constexpr int CONSUMER_THREADS = CONSUMER_WARPGROUPS * kittens::WARPGROUP_THREADS;
// The named barrier over the two consumer warpgroups (ids 4 and 5 are each warpgroup's own, at
// the O store; 0 is `__syncthreads`).
constexpr int CONSUMERS_BARRIER = 1;
using consumers = kittens::group<CONSUMER_WARPGROUPS * kittens::WARPGROUP_WARPS>;
// Token slots per KV block (`step::CUDA_KV_BLOCK_SIZE`): one block is one K (or V) tile.
constexpr uint32_t BLOCK_SIZE = 64;
static_assert(KV_HEIGHT == int(BLOCK_SIZE), "one paged (block, kv head) half is one TMA box");
// Entries of a sequence's block table (`step::CUDA_MAX_BLOCKS_PER_SEQ`): the columns of its view.
constexpr uint32_t MAX_BLOCKS_PER_SEQ = 128;

using half_tile = st_bf<QO_HEIGHT, HALF_DIM>;
static_assert(KV_HEIGHT == QO_HEIGHT, "Q, K, V and O halves are one tile type");

// `axis = 1`: a Q (or O) half's rows are 64 consecutive tokens of one head of the token-major
// `[1, rows, heads, 512]` view, its columns the half's 256.
using qo_gl = gl<bf16, 1, -1, -1, HEAD_DIM, tma::descriptor<half_tile, 1>>;
using kv_gl = gl<bf16, -1, -1, KV_HEIGHT, HEAD_DIM, half_tile>;
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

constexpr size_t SMEM_TILES = 3 * HALVES * sizeof(half_tile);
static_assert(SMEM_TILES == 196608, "2 Q + 2 K + 2 V tiles of [64, 256] bf16");
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int ATTN_SMEM_BYTES = int(SMEM_TILES) + 1024;
static_assert(ATTN_SMEM_BYTES == 197632, "the D = 256 prefill's dynamic shared memory");
static_assert(ATTN_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

template <bool WINDOWED>
__global__ __launch_bounds__(NUM_THREADS, 1) void attn_prefill_d512_bf16_ker(
    const __grid_constant__ attn_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    const int warpid = kittens::warpid(), warpgroupid = warpid / kittens::WARPGROUP_WARPS;

    half_tile(&q_smem)[HALVES] = al.allocate<half_tile, HALVES>();
    half_tile(&k_smem)[HALVES] = al.allocate<half_tile, HALVES>();
    half_tile(&v_smem)[HALVES] = al.allocate<half_tile, HALVES>();

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
                    coord<half_tile> at = {block, kv_head, 0, c};
                    warp::tma::load_async(k_smem[c], g.k, at, k_arrived);
                }
                if (t >= 1) {
                    wait(v_done, (t - 1) % 2);
                }
                warp::tma::expect_bytes(v_arrived, sizeof(v_smem));
                for (int c = 0; c < HALVES; c++) {
                    coord<half_tile> at = {block, kv_head, 0, c};
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
            // S over the 512 lanes: the low halves, then the high halves accumulated onto them.
            warpgroup::mm_ABt(att_block, q_smem[0], k_smem[0]);
            warpgroup::mma_ABt(att_block, q_smem[1], k_smem[1]);

            // The running max before this tile, unscaled (the rescale below reads it).
            warp::copy(max_vec_last, max_vec);

            // The WGMMAs write `att_block` asynchronously: wait before reading it. After the wait
            // this warpgroup has read K: release it to the producer.
            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) {
                arrive(k_done, 1);
            }

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
            // A partial last tile: zero the V rows of the slots at or past kv_len, in both halves,
            // before either warpgroup's P·V reads them (0 · NaN = NaN). `valid` is the same for
            // the whole CTA, so every consumer thread takes this branch or none does.
            const int valid = kv_len - k0;
            if (valid < KV_HEIGHT) {
                const int tid = warpgroupid * kittens::WARPGROUP_THREADS + warpgroup::laneid();
                for (int e = tid; e < (KV_HEIGHT - valid) * HEAD_DIM; e += CONSUMER_THREADS) {
                    const int col = e % HEAD_DIM;
                    v_smem[col / HALF_DIM][{valid + e / HEAD_DIM, col % HALF_DIM}] =
                        kittens::base_types::constants<bf16>::zero();
                }
                // Each warpgroup fences its own threads' writes to the async proxy, then both
                // meet before either P·V.
                warpgroup::mma_fence();
                consumers::sync(CONSUMERS_BARRIER);
            }
            warpgroup::mma_AB(o_reg, att_block_mma, v_smem[c]);
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
extern "C" int scratchy_cuda_attn_prefill_d512_bf16_prepare(void) {
    const cudaError_t causal =
        cudaFuncSetAttribute(attn_prefill_d512_bf16_ker<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, ATTN_SMEM_BYTES);
    if (causal != cudaSuccess) {
        return static_cast<int>(causal);
    }
    return static_cast<int>(cudaFuncSetAttribute(attn_prefill_d512_bf16_ker<true>,
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
// bf16; `block_table` and `sequence` as `scratchy_cuda_attn_prefill_bf16`'s.
// `scratchy_cuda_attn_prefill_d512_bf16_prepare` must have succeeded on the current device. The
// launch is asynchronous on `stream`.
extern "C" int scratchy_cuda_attn_prefill_d512_bf16(
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
        bf16* kp = static_cast<bf16*>(const_cast<void*>(k_cache));
        bf16* vp = static_cast<bf16*>(const_cast<void*>(v_cache));
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
            window == 0 ? attn_prefill_d512_bf16_ker<false> : attn_prefill_d512_bf16_ker<true>;
        return static_cast<int>(cudaLaunchKernel(ker, grid, block, args, ATTN_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
