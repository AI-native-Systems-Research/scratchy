// SPDX-License-Identifier: Apache-2.0
//
// Causal grouped-query attention of one prefill bucket over the paged e4m3 KV cache, for the H100
// at head_dim 256 (FP8 S5): `attn_prefill_bf16.cu`, which stays as its test-only reference, with
// K and V read as e4m3 and widened to bf16 in shared memory before the same math. The cache is one
// layer's `[num_blocks, kv_heads, 64, 256]` e4m3, one byte an element, at a fixed scale of 1.0
// (`kv_append_fp8.cu`); Q and O are the tape's token-major `[rows, heads · 256]` bf16.
//
// What differs from the bf16 kernel, and why:
//   - The two stages hold `st_fp8e4m3<64, 256>` K and V tiles (16 KiB each, half the TMA bytes).
//     Both consumer warpgroups read every stage, so they widen it together: warpgroup w widens
//     64-column chunks w and w + 2 of K into ONE shared `st_bf<64, 256>` K (`warpgroup::load` of
//     an `rt_fp8e4m3<16, 64>` through a `subtile<64, 64>` view, TK's 8-bit `ldmatrix` path;
//     `warp::copy` to `rt_bf<16, 64>`, exact; `warpgroup::store` back through the bf16 tile's
//     view), and later V the same way into one shared bf16 V. One bf16 K and one bf16 V for both
//     stages keep the dynamic shared memory at today's 197,632 bytes.
//   - Each widened tile is written by `stmatrix` (the generic proxy) and read by WGMMA (the async
//     proxy): `warpgroup::mma_fence()` (TK's `wgmma.fence` + `fence.proxy.async.shared::cta`),
//     then the consumers' named barrier 1, before the WGMMAs that read it. So no inline PTX, and
//     two consumer barriers per tile (the bf16 kernel has one, on a partial tile only).
//   - Why one bf16 K and one bf16 V suffice: a warpgroup reaching tile t + 1's K widen has passed
//     tile t's V barrier, which the other reaches only after its tile t Q·Kᵀ wait; reaching tile
//     t + 1's V widen, both have passed tile t + 1's K barrier, so both finished tile t's P·V.
//   - The V rows of slots at or past kv_len (an unwritten e4m3 byte may be NaN) are zeroed in
//     registers during the widen, `warp::lower_fill` with +0, for the bf16 kernel's shared-memory
//     loop: the same +0 rows. K needs nothing, as there: `right_fill` overwrites those scores.
//   - After the widen the math is the bf16 kernel's, line for line, on bf16 tiles, so its output
//     is the bf16 kernel's on the widened cache, bit for bit (`tests/attention_fp8kv.rs`).
//
// The bf16 kernel's description of the grid, the masks, the online softmax, the window, the loop
// bounds and the O store holds unchanged (`attn_prefill_bf16.cu`, adapted from ThunderKittens
// `kernels/attention/mha_h100/mha_h100.cu` at 67845f5f, MIT, see `thunderkittens/LICENSE`).
//
// Shared memory: 2 q tiles (64 KiB), 2 stages of e4m3 K and V (64 KiB), the bf16 K and V (64
// KiB), plus 1,024 bytes of alignment slack = 197,632 bytes of dynamic shared memory, set once by
// `scratchy_cuda_attn_prefill_fp8kv_prepare` (in `PREPARE`). The 7 semaphores are static.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cmath>
#include <cstdint>
#include <type_traits>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

constexpr int HEAD_DIM = 256;
// Query rows per consumer warpgroup, and K/V rows per tile: the KV block size.
constexpr int QO_HEIGHT = 64;
constexpr int KV_HEIGHT = 64;
constexpr int STAGES = 2;
constexpr int CONSUMER_WARPGROUPS = 2;
constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + 1;
constexpr int NUM_WARPS = NUM_WARPGROUPS * kittens::WARPGROUP_WARPS;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;
// Query rows per CTA (`step::ATTN_PREFILL_BF16_ROWS`).
constexpr int CTA_ROWS = CONSUMER_WARPGROUPS * QO_HEIGHT;
// The named barrier over the two consumer warpgroups (ids 4 and 5 are each warpgroup's own, at
// the O store; 0 is `__syncthreads`).
constexpr int CONSUMERS_BARRIER = 1;
using consumers = kittens::group<CONSUMER_WARPGROUPS * kittens::WARPGROUP_WARPS>;
// Token slots per KV block (`step::CUDA_KV_BLOCK_SIZE`): one block is one K (or V) tile.
constexpr uint32_t BLOCK_SIZE = 64;
static_assert(KV_HEIGHT == int(BLOCK_SIZE), "one paged (block, kv head) tile is one TMA box");
// Entries of a sequence's block table (`step::CUDA_MAX_BLOCKS_PER_SEQ`): the columns of its view.
constexpr uint32_t MAX_BLOCKS_PER_SEQ = 128;

using q_tile = st_bf<QO_HEIGHT, HEAD_DIM>;
// A K or V tile as cached, and as widened for the WGMMAs.
using kv8_tile = st_fp8e4m3<KV_HEIGHT, HEAD_DIM>;
using kv_tile = st_bf<KV_HEIGHT, HEAD_DIM>;
using o_tile = st_bf<QO_HEIGHT, HEAD_DIM>;
// Columns per widen step: one `rt_fp8e4m3<16, 64>` and one `rt_bf<16, 64>`, 24 registers a thread.
// Warpgroup w widens chunks w and w + 2.
constexpr int WIDEN_COLS = 64;
constexpr int WIDEN_CHUNKS = HEAD_DIM / WIDEN_COLS;

// `axis = 1`: a Q (or O) tile's rows are 64 consecutive tokens of one head of the token-major
// `[1, rows, heads, 256]` view.
using qo_gl = gl<bf16, 1, -1, -1, HEAD_DIM, tma::descriptor<q_tile, 1>>;
using kv_gl = gl<fp8e4m3, -1, -1, KV_HEIGHT, HEAD_DIM, kv8_tile>;
// An index region, `[1, entries]` int with no TMA type: read element by element. Its u32 values
// are all below 2^31 (block ids below `num_blocks` ≤ INT_MAX, and `P + L` ≤ 8192), so they read
// as the same ints.
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

constexpr size_t SMEM_TILES =
    CONSUMER_WARPGROUPS * sizeof(q_tile) + STAGES * 2 * sizeof(kv8_tile) + 2 * sizeof(kv_tile);
static_assert(SMEM_TILES == 196608, "2 q, 2 stages of e4m3 K and V, a bf16 K and V of [64, 256]");
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int ATTN_SMEM_BYTES = int(SMEM_TILES) + 1024;
static_assert(ATTN_SMEM_BYTES == 197632, "the bf16 prefill's dynamic shared memory");
static_assert(ATTN_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

// This warpgroup's chunks of the widen of the e4m3 tile `src` into the bf16 tile `dst`, `WIDEN_COLS`
// columns each, from chunk `first` in steps of `CONSUMER_WARPGROUPS`; rows at or past `zero_from`
// (a tile's first unwritten slot; 64 for none) are +0. Warp w of the warpgroup holds rows 16w ..
// 16w + 16 of each chunk (`warpgroup::load`'s split), so its fill starts at `zero_from − 16w`, the
// whole chunk when that is not above 0.
__device__ inline void widen_kv(kv_tile& dst, kv8_tile& src, int first, int zero_from) {
#pragma unroll
    for (int i = 0; i < WIDEN_CHUNKS / CONSUMER_WARPGROUPS; i++) {
        const int c = first + i * CONSUMER_WARPGROUPS;
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
__global__ __launch_bounds__(NUM_THREADS, 1) void attn_prefill_fp8kv_ker(
    const __grid_constant__ attn_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    const int warpid = kittens::warpid(), warpgroupid = warpid / kittens::WARPGROUP_WARPS;

    q_tile(&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
    kv8_tile(&k_smem)[STAGES] = al.allocate<kv8_tile, STAGES>();
    kv8_tile(&v_smem)[STAGES] = al.allocate<kv8_tile, STAGES>();
    // The widened K and V, shared by both consumer warpgroups and both stages.
    kv_tile(&k_bf)[1] = al.allocate<kv_tile, 1>();
    kv_tile(&v_bf)[1] = al.allocate<kv_tile, 1>();
    auto(*o_smem) = reinterpret_cast<o_tile(*)>(q_smem);

    const int q_head = blockIdx.y;
    const int kv_head = q_head / g.hr;
    // The CTA's first query-row tile, in units of QO_HEIGHT rows.
    const int row_tile0 = blockIdx.x * CONSUMER_WARPGROUPS;
    const int prefix_len = g.sequence[{0, 0, 0, 0}];
    const int kv_len = prefix_len + g.sequence[{0, 0, 0, 1}];
    // The last key any of this CTA's rows may attend to, and so the tiles the loop reads: key 0
    // is allowed for every row, so this is at least one tile.
    const int row_last = blockIdx.x * CTA_ROWS + CTA_ROWS - 1;
    const int key_last = min(prefix_len + row_last, kv_len - 1);
    const int kv_iters = key_last / KV_HEIGHT + 1;
    // The first tile the loop reads: 0 for full causal attention; with a window, the tile of the
    // CTA's lowest window floor (its first row's). It may be at or past `kv_iters` for a CTA of
    // padding rows far past the last key, whose loop is then empty.
    int kv_first = 0;
    if constexpr (WINDOWED) {
        const int row_first = blockIdx.x * CTA_ROWS;
        kv_first = max(0, prefix_len + row_first - (g.window - 1)) / KV_HEIGHT;
    }

    __shared__ kittens::semaphore q_arrived, k_smem_arrived[STAGES], v_smem_arrived[STAGES],
        compute_done[STAGES];
    if (threadIdx.x == 0) {
        init_semaphore(q_arrived, 0, 1);
        for (int j = 0; j < STAGES; j++) {
            init_semaphore(k_smem_arrived[j], 0, 1);
            init_semaphore(v_smem_arrived[j], 0, 1);
            init_semaphore(compute_done[j], CONSUMER_WARPGROUPS, 0);
        }
        tma::expect_bytes(q_arrived, sizeof(q_smem));
        for (int wg = 0; wg < CONSUMER_WARPGROUPS; wg++) {
            // Token tile `row_tile0 + wg` of head `q_head`, as `[1, rows, heads, 256]`.
            coord<q_tile> at = {0, row_tile0 + wg, q_head, 0};
            tma::load_async<1, cache_policy::NORMAL>(q_smem[wg], g.q, at, q_arrived);
        }
    }
    __syncthreads();

    if (warpgroupid == NUM_WARPGROUPS - 1) {
        // The producer: one warp loads KV tile `j`, the loop's `t`-th, into stage `t % STAGES`,
        // after the consumers released that stage's previous tile, the loop's `t - STAGES`-th.
        warpgroup::decrease_registers<24>();
        if (warpid == NUM_WARPS - 4) {
            for (int j = kv_first; j < kv_iters; j++) {
                const int t = j - kv_first;
                const int stage = t % STAGES;
                if (t >= STAGES) {
                    wait(compute_done[stage], ((t - STAGES) / STAGES) % 2);
                }
                const int block = g.block_table[{0, 0, 0, j}];
                coord<kv8_tile> at = {block, kv_head, 0, 0};
                warp::tma::expect_bytes(k_smem_arrived[stage], sizeof(kv8_tile));
                warp::tma::load_async(k_smem[stage], g.k, at, k_smem_arrived[stage]);
                warp::tma::expect_bytes(v_smem_arrived[stage], sizeof(kv8_tile));
                warp::tma::load_async(v_smem[stage], g.v, at, v_smem_arrived[stage]);
            }
        }
    } else {
        warpgroup::increase_registers<240>();

        rt_fl<16, KV_HEIGHT> att_block;
        rt_bf<16, KV_HEIGHT> att_block_mma;
        rt_fl<16, HEAD_DIM> o_reg;
        // `max_vec_last` holds the max before a tile, then (after the rescale) the tile's rescale.
        col_vec<rt_fl<16, KV_HEIGHT>> max_vec, norm_vec, max_vec_last, max_vec_scaled;

        warp::neg_infty(max_vec);
        warp::zero(norm_vec);
        warp::zero(o_reg);

        // This warp's 16 query rows, as absolute positions: row `row0 + r` is at `P + row0 + r`.
        const int row0 = (row_tile0 + warpgroupid) * QO_HEIGHT + warpgroup::warpid() * 16;
        constexpr float NEG_INF = kittens::base_types::constants<float>::neg_infty();

        wait(q_arrived, 0);

        for (int j = kv_first; j < kv_iters; j++) {
            // The loop's `t`-th tile, in the stage and phase the producer loaded it with.
            const int t = j - kv_first;
            const int stage = t % STAGES, phase = (t / STAGES) % 2;
            wait(k_smem_arrived[stage], phase);
            // This warpgroup's half of K, widened; both halves visible to the tensor cores.
            widen_kv(k_bf[0], k_smem[stage], warpgroupid, KV_HEIGHT);
            warpgroup::mma_fence();
            consumers::sync(CONSUMERS_BARRIER);
            warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_bf[0]);

            // The running max before this tile, unscaled (the rescale below reads it).
            warp::copy(max_vec_last, max_vec);

            // The WGMMA writes `att_block` asynchronously: wait before reading it.
            warpgroup::mma_async_wait();

            // Keys k0 .. k0 + 64 of tile j. Causal: keep key ≤ P + row (`tril` keeps col ≤ row +
            // diagonal, with the tile's row and col from 0). Length: drop key ≥ kv_len.
            const int k0 = j * KV_HEIGHT;
            warp::tril(att_block, att_block, prefix_len + row0 - k0, NEG_INF);
            warp::right_fill(att_block, att_block, kv_len - k0, NEG_INF);
            // Window: keep key ≥ P + row − (W − 1) (`triu` keeps col ≥ row + diagonal).
            if constexpr (WINDOWED) {
                warp::triu(att_block, att_block, prefix_len + row0 - k0 - (g.window - 1), NEG_INF);
            }

            warp::row_max(max_vec, att_block, max_vec);
            warp::mul(att_block, att_block, g.scale_log2);
            if constexpr (WINDOWED) {
                // A tile wholly outside a row's window leaves its max −inf: subtract 0 instead, so
                // its p is exactly 0, never `exp2(−inf − (−inf)) = NaN`. The select is on the
                // unscaled max, before its own `mul`, so the subtraction below still reads that
                // product, as the unwindowed kernel's does. A register vector, so `unary_op`, not
                // the tile map `unary_map`.
                warp::unary_op<scratchy::ops::neg_inf_to_zero>(max_vec_scaled, max_vec);
                warp::mul(max_vec_scaled, max_vec_scaled, g.scale_log2);
            } else {
                warp::mul(max_vec_scaled, max_vec, g.scale_log2);
            }
            warp::sub_row(att_block, att_block, max_vec_scaled);
            warp::exp2(att_block, att_block);
            // The rescale of the running sum and output, before this tile is accumulated into
            // them: exp2((m_old − m_new) · scale_log2), and EXACTLY 1 for a row whose max this
            // tile left unchanged. That makes a tile masked out for a row (every score −inf, every
            // p exactly 0) an exact no-op on it, whatever the tile count: how many tiles a CTA
            // reads depends on where its rows sit in the bucket (`kv_iters`), so the same row of a
            // sequence reads more or fewer masked trailing tiles in one forward than in a chunk of
            // it. Upstream's `exp2(m_old·c − m_new·c)` is not exactly 1 there: nvcc contracts the
            // product and the difference into `fma(m_old, c, −round(m_new·c))`, which is the
            // rounding error of `m·c`, not 0, so each masked tile scaled the row by 1 ± an f32 ulp.
            scratchy::ops::rv_ternary_map<scratchy::ops::rescale_exact1>(
                max_vec_last, max_vec_last, max_vec, g.scale_log2);
            warp::mul(norm_vec, norm_vec, max_vec_last);
            warp::row_sum(norm_vec, att_block, norm_vec);
            warp::copy(att_block_mma, att_block);
            warp::mul_row(o_reg, o_reg, max_vec_last);

            wait(v_smem_arrived[stage], phase);
            // This warpgroup's half of V, widened. A partial last tile's rows of the slots at or
            // past kv_len, which no append wrote, are +0 (0 · NaN = NaN, #791).
            widen_kv(v_bf[0], v_smem[stage], warpgroupid, kv_len - k0);
            warpgroup::mma_fence();
            consumers::sync(CONSUMERS_BARRIER);
            warpgroup::mma_AB(o_reg, att_block_mma, v_bf[0]);
            // Wait for the WGMMA to finish reading the stage before releasing it to the producer.
            warpgroup::mma_async_wait();

            if (warpgroup::laneid() == 0) {
                arrive(compute_done[stage], 1);
            }
        }

        // A windowed padding row with no key in its window has a sum of 0 and an output of 0:
        // divide it by 1, not 0.
        if constexpr (WINDOWED) {
            warp::unary_op<scratchy::ops::zero_to_one>(norm_vec, norm_vec);
        }
        warp::div_row(o_reg, o_reg, norm_vec);
        // O over this warpgroup's own q tile, which no later load reuses.
        warpgroup::store(o_smem[warpgroupid], o_reg);
        warpgroup::sync(warpgroupid + 4);
        if (warpgroup::warpid() == 0) {
            coord<o_tile> at = {0, row_tile0 + warpgroupid, q_head, 0};
            warp::tma::store_async<1, cache_policy::NORMAL>(g.o, o_smem[warpgroupid], at);
        }
        warp::tma::store_async_wait();
    }
}

}  // namespace

// The kernel's one-time setup: allow both instantiations `ATTN_SMEM_BYTES` of dynamic shared
// memory on the current device (the default limit is 48 KB). Run once per device before any launch
// (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch.
//
// 0 = done; any other value = the cudaError_t of setting the attribute, the unwindowed one's first.
extern "C" int scratchy_cuda_attn_prefill_fp8kv_prepare(void) {
    const cudaError_t causal = cudaFuncSetAttribute(
        attn_prefill_fp8kv_ker<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, ATTN_SMEM_BYTES);
    if (causal != cudaSuccess) {
        return static_cast<int>(causal);
    }
    return static_cast<int>(cudaFuncSetAttribute(
        attn_prefill_fp8kv_ker<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, ATTN_SMEM_BYTES));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value
// = the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// Refused before any pointer is touched: `head_dim` not 256; `rows` zero or not a multiple of 128;
// `q_heads` or `kv_heads` zero, or `q_heads` not a multiple of `kv_heads`; `block_size` not 64;
// `num_blocks` zero; `scale` not finite or not above 0; `window` not a multiple of 64
// (`step::ATTN_WINDOW_MULTIPLE`; 0, full causal attention, is one); a dimension or the window
// above INT_MAX.
//
// `q` and `out` must be 16-byte-aligned device pointers to `rows · q_heads · 256` bf16, `out`
// overlapping no input; `k_cache` and `v_cache` to one layer's `num_blocks · kv_heads · 64 · 256`
// e4m3 bytes; `block_table` to the sequence's block ids, `ceil((P + L) / 64)` of them each below
// `num_blocks`; `sequence` to `[P, L]` u32 with `1 ≤ L ≤ rows` and `P + L ≤ 64 · (entries)`.
// `scratchy_cuda_attn_prefill_fp8kv_prepare` must have succeeded on the current device. The launch
// is asynchronous on `stream`; the caller owns it and synchronizes.
extern "C" int scratchy_cuda_attn_prefill_fp8kv(const void* q, const void* k_cache,
                                                const void* v_cache, const void* block_table,
                                                const void* sequence, void* out, uint32_t rows,
                                                uint32_t q_heads, uint32_t kv_heads,
                                                uint32_t head_dim, uint32_t block_size,
                                                uint32_t num_blocks, float scale, uint32_t window,
                                                void* stream /* cudaStream_t; null = default */) {
    if (head_dim != uint32_t(HEAD_DIM) || rows == 0 || rows % uint32_t(CTA_ROWS) != 0 ||
        q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 || block_size != BLOCK_SIZE ||
        num_blocks == 0 || !std::isfinite(scale) || !(scale > 0.0f) ||
        window % BLOCK_SIZE != 0 || rows > uint32_t(INT_MAX) || q_heads > uint32_t(INT_MAX) ||
        num_blocks > uint32_t(INT_MAX) || window > uint32_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        bf16* qp = static_cast<bf16*>(const_cast<void*>(q));
        bf16* op = static_cast<bf16*>(out);
        fp8e4m3* kp = static_cast<fp8e4m3*>(const_cast<void*>(k_cache));
        fp8e4m3* vp = static_cast<fp8e4m3*>(const_cast<void*>(v_cache));
        // Building each `gl` encodes its TMA tensor map into host memory; nothing is allocated on
        // the device. Q and O as `[1, rows, heads, 256]`; K and V as `[num_blocks, kv_heads, 64,
        // 256]`, whose bound on the block axis makes TMA zero-fill, never read past, a bad block.
        qo_gl qg{qp, nullptr, size_t(rows), size_t(q_heads), nullptr};
        qo_gl og{op, nullptr, size_t(rows), size_t(q_heads), nullptr};
        kv_gl kg{kp, size_t(num_blocks), size_t(kv_heads), nullptr, nullptr};
        kv_gl vg{vp, size_t(num_blocks), size_t(kv_heads), nullptr, nullptr};
        // The index regions as `[1, entries]` int views: the block table's `MAX_BLOCKS_PER_SEQ`
        // entries, the sequence's two.
        index_gl tg{static_cast<int*>(const_cast<void*>(block_table)), nullptr, nullptr, size_t(1),
                    size_t(MAX_BLOCKS_PER_SEQ)};
        index_gl sg{static_cast<int*>(const_cast<void*>(sequence)), nullptr, nullptr, size_t(1),
                    size_t(2)};
        attn_globals G{qg, kg, vg, og, tg, sg, scale * 1.44269504089f, int(q_heads / kv_heads),
                       int(window)};
        void* args[] = {&G};
        const dim3 grid(rows / uint32_t(CTA_ROWS), q_heads, 1), block(NUM_THREADS);
        auto* ker = window == 0 ? attn_prefill_fp8kv_ker<false> : attn_prefill_fp8kv_ker<true>;
        return static_cast<int>(cudaLaunchKernel(ker, grid, block, args,
                                                 ATTN_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
