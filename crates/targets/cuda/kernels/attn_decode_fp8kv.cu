// SPDX-License-Identifier: Apache-2.0
//
// Grouped-query decode attention of one query row over the paged e4m3 KV cache, for the H100 at
// head_dim 256 (FP8 S5): `attn_decode_bf16.cu`, which stays as its test-only reference, with K and
// V read as e4m3 and widened to bf16 in shared memory before the same math. The cache is one
// layer's `[num_blocks, kv_heads, 64, 256]` e4m3, one byte an element, at a fixed scale of 1.0
// (`kv_append_fp8.cu`); Q and O are the tape's `[1, q_heads · 256]` bf16.
//
// What differs from the bf16 kernel, and why:
//   - Each warpgroup's K and V tiles are `st_fp8e4m3<64, 256>` (16 KiB each, half the TMA bytes).
//     After a tile's semaphore, the owning warpgroup widens it into ONE `st_bf<64, 256>` of its
//     own, 64 columns at a time: `warpgroup::load` of an `rt_fp8e4m3<16, 64>` through a
//     `subtile<64, 64>` view (TK's 8-bit `ldmatrix` path), `warp::copy` to `rt_bf<16, 64>` (exact:
//     every e4m3 value is a bf16 value), `warpgroup::store` back through the bf16 tile's view.
//     K widens into the buffer, Q·Kᵀ reads it and is waited on; V then widens into the same
//     buffer for P·V. One bf16 buffer per warpgroup, not two, keeps the dynamic shared memory at
//     today's 173,056 bytes (two would be 237,568, over the H100's 227 KiB).
//   - The widened tile is written by `stmatrix` (the generic proxy) and read by WGMMA (the async
//     proxy): `warpgroup::mma_fence()` (TK's `wgmma.fence` + `fence.proxy.async.shared::cta`),
//     then the warpgroup's own named barrier, before each WGMMA. So no inline PTX.
//   - The V rows of slots at or past kv_len (an unwritten e4m3 byte may be NaN) are zeroed in
//     registers during the widen, `warp::lower_fill` with +0, for the bf16 kernel's shared-memory
//     loop: the same +0 rows. K needs nothing, as there: `right_fill` overwrites those scores.
//   - After the widen the math is the bf16 kernel's, line for line, on a bf16 tile, so its output
//     is the bf16 kernel's on the widened cache, bit for bit (`tests/attention_fp8kv.rs`).
//
// The bf16 kernel's description of the grid, the stacked query tile, the online softmax, the
// merge, the window and the fixed order of additions holds unchanged (`attn_decode_bf16.cu`).
//
// Shared memory: the Q tile (32 KiB, also the merge tiles), per warpgroup an e4m3 K and V tile and
// one bf16 tile (128 KiB), the bf16 output tile (8 KiB), plus 1,024 bytes of alignment slack =
// 173,056 bytes of dynamic shared memory, set once by `scratchy_cuda_attn_decode_fp8kv_prepare`
// (in `PREPARE`). The semaphores and the merge's six `sv_fl<16>` are static.

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

// `step::ATTN_DECODE_BF16_HEAD_DIM`.
constexpr int HEAD_DIM = 256;
// Query rows of the stacked tile, and K/V rows per tile: the KV block size.
constexpr int QO_HEIGHT = 64;
constexpr int KV_HEIGHT = 64;
// `step::ATTN_DECODE_BF16_MAX_GROUP`: the rows of a consumer warpgroup's warp 0, which are the ones
// merged and stored.
constexpr int MAX_GROUP = 16;
constexpr int CONSUMER_WARPGROUPS = 2;
constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + 1;
constexpr int NUM_WARPS = NUM_WARPGROUPS * kittens::WARPGROUP_WARPS;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;
// The named barrier over the two consumer warpgroups (ids 4 and 5 are each warpgroup's own, before
// each WGMMA on its widened tile; 0 is `__syncthreads`).
constexpr int CONSUMERS_BARRIER = 1;
using consumers = kittens::group<CONSUMER_WARPGROUPS * kittens::WARPGROUP_WARPS>;
// Token slots per KV block (`step::CUDA_KV_BLOCK_SIZE`): one block is one K (or V) tile.
constexpr uint32_t BLOCK_SIZE = 64;
static_assert(KV_HEIGHT == int(BLOCK_SIZE), "one paged (block, kv head) tile is one TMA box");
// Entries of a sequence's block table (`step::CUDA_MAX_BLOCKS_PER_SEQ`): the columns of its view.
constexpr uint32_t MAX_BLOCKS_PER_SEQ = 128;
// log2(e): the softmax runs in base 2 with the scale folded in.
constexpr float LOG2E = 1.44269504089f;

using q_tile = st_bf<QO_HEIGHT, HEAD_DIM>;
// A K or V tile as cached, and as widened for the WGMMAs.
using kv8_tile = st_fp8e4m3<KV_HEIGHT, HEAD_DIM>;
using kv_tile = st_bf<KV_HEIGHT, HEAD_DIM>;
// Columns per widen step: one `rt_fp8e4m3<16, 64>` and one `rt_bf<16, 64>`, 24 registers a thread.
constexpr int WIDEN_COLS = 64;
constexpr int WIDEN_CHUNKS = HEAD_DIM / WIDEN_COLS;
// One warpgroup's warp-0 rows of the unnormalised output, for the merge. Swizzled, as the f32
// `warp::store`/`warp::load` between a shared and a register tile always address one
// (`shared_to_register.cuh`, the 32-bit branch), so the shared tile's own `operator[]`, which
// `mul_row` and `div_row` index through, agrees with them.
using merge_tile = st_fl<MAX_GROUP, HEAD_DIM>;
// The merged output rows, plain, so its rows are contiguous for the bulk store.
using out_tile = st_bf<MAX_GROUP, HEAD_DIM, false>;
static_assert(CONSUMER_WARPGROUPS * sizeof(merge_tile) == sizeof(q_tile),
              "the two merge tiles fill the Q tile exactly");

// q as `[q_heads, 256]`, the decode row viewed head-major: a box's rows are query heads.
using q_gl = gl<bf16, 1, 1, -1, HEAD_DIM, q_tile>;
using kv_gl = gl<fp8e4m3, -1, -1, KV_HEIGHT, HEAD_DIM, kv8_tile>;
// The output row `[1, q_heads · 256]`, no TMA type: written by bulk copy only.
using out_gl = gl<bf16, 1, 1, -1, -1>;
// An index region, `[1, entries]` int with no TMA type: read element by element. Its u32 values
// are all below 2^31 (block ids below `num_blocks` ≤ INT_MAX, and `P + L` ≤ 8192), so they read
// as the same ints.
using index_gl = gl<int, 1, 1, -1, -1>;

struct decode_globals {
    q_gl q;
    kv_gl k;
    kv_gl v;
    out_gl out;
    // `[1, MAX_BLOCKS_PER_SEQ]`.
    index_gl block_table;
    // `[1, 2]`: `[P, L]`.
    index_gl sequence;
    // `scale · log2(e)`.
    float scale_log2;
    // `q_heads / kv_heads`, at most `MAX_GROUP`.
    int hr;
    // The sliding window `W`, a multiple of 64; 0 for every key.
    int window;
};

constexpr size_t SMEM_TILES =
    sizeof(q_tile) + CONSUMER_WARPGROUPS * (2 * sizeof(kv8_tile) + sizeof(kv_tile)) +
    sizeof(out_tile);
static_assert(SMEM_TILES == 172032,
              "q, per warpgroup e4m3 K and V and one bf16 tile of [64, 256], and [16, 256] bf16");
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int DECODE_SMEM_BYTES = int(SMEM_TILES) + 1024;
static_assert(DECODE_SMEM_BYTES == 173056, "the bf16 decode's dynamic shared memory");
static_assert(DECODE_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

// This warpgroup's widen of the e4m3 tile `src` into the bf16 tile `dst`, `WIDEN_COLS` columns at
// a time; rows at or past `zero_from` (a tile's first unwritten slot; 64 for none) are +0. Warp w
// of the warpgroup holds rows 16w .. 16w + 16 of each chunk (`warpgroup::load`'s split), so its
// fill starts at `zero_from − 16w`, the whole chunk when that is not above 0.
__device__ inline void widen_kv(kv_tile& dst, kv8_tile& src, int zero_from) {
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
__global__ __launch_bounds__(NUM_THREADS, 1) void attn_decode_fp8kv_ker(
    const __grid_constant__ decode_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    const int warpid = kittens::warpid(), warpgroupid = warpid / kittens::WARPGROUP_WARPS;

    q_tile(&q_smem)[1] = al.allocate<q_tile, 1>();
    kv8_tile(&k_smem)[CONSUMER_WARPGROUPS] = al.allocate<kv8_tile, CONSUMER_WARPGROUPS>();
    kv8_tile(&v_smem)[CONSUMER_WARPGROUPS] = al.allocate<kv8_tile, CONSUMER_WARPGROUPS>();
    // Each warpgroup's widened K, then V.
    kv_tile(&kv_bf)[CONSUMER_WARPGROUPS] = al.allocate<kv_tile, CONSUMER_WARPGROUPS>();
    out_tile(&o_bf)[1] = al.allocate<out_tile, 1>();
    // The merge's two output tiles over the Q tile, written only after both warpgroups' last
    // read of Q.
    auto(*o_part) = reinterpret_cast<merge_tile(*)>(q_smem);

    // Each warpgroup's warp-0 running max and sum, and the merge's rescales.
    __shared__ sv_fl<MAX_GROUP> m_part[CONSUMER_WARPGROUPS];
    __shared__ sv_fl<MAX_GROUP> l_part[CONSUMER_WARPGROUPS];
    __shared__ sv_fl<MAX_GROUP> r_part[CONSUMER_WARPGROUPS];

    const int kv_head = blockIdx.x;
    const int kv_len = g.sequence[{0, 0, 0, 0}] + g.sequence[{0, 0, 0, 1}];
    const int kv_iters = (kv_len + KV_HEIGHT - 1) / KV_HEIGHT;
    // The window's first key and the tile holding it: 0 and 0 without a window.
    [[maybe_unused]] int key_first = 0;
    int kv_first = 0;
    if constexpr (WINDOWED) {
        key_first = max(0, kv_len - g.window);
        kv_first = key_first / KV_HEIGHT;
    }

    __shared__ kittens::semaphore q_arrived, k_smem_arrived[CONSUMER_WARPGROUPS],
        v_smem_arrived[CONSUMER_WARPGROUPS], compute_done[CONSUMER_WARPGROUPS];
    if (threadIdx.x == 0) {
        init_semaphore(q_arrived, 0, 1);
        for (int w = 0; w < CONSUMER_WARPGROUPS; w++) {
            init_semaphore(k_smem_arrived[w], 0, 1);
            init_semaphore(v_smem_arrived[w], 0, 1);
            init_semaphore(compute_done[w], 1, 0);
        }
        tma::expect_bytes(q_arrived, sizeof(q_tile));
        // Rows `g·hr .. g·hr + 64` of `[q_heads, 256]`, in element units.
        coord<> at = {0, 0, kv_head * g.hr, 0};
        tma::load_async(q_smem[0], g.q, at, q_arrived);
    }
    __syncthreads();

    if (warpgroupid == NUM_WARPGROUPS - 1) {
        // The producer: one warp loads KV tile `j`, the loop's `t`-th, into warpgroup `t % 2`'s
        // buffer, after that warpgroup released its previous tile, the loop's `t - 2`-th.
        warpgroup::decrease_registers<24>();
        if (warpid == NUM_WARPS - 4) {
            for (int j = kv_first; j < kv_iters; j++) {
                const int t = j - kv_first;
                const int w = t % CONSUMER_WARPGROUPS;
                if (t >= CONSUMER_WARPGROUPS) {
                    wait(compute_done[w], ((t - CONSUMER_WARPGROUPS) / CONSUMER_WARPGROUPS) % 2);
                }
                const int block = g.block_table[{0, 0, 0, j}];
                coord<kv8_tile> at = {block, kv_head, 0, 0};
                warp::tma::expect_bytes(k_smem_arrived[w], sizeof(kv8_tile));
                warp::tma::load_async(k_smem[w], g.k, at, k_smem_arrived[w]);
                warp::tma::expect_bytes(v_smem_arrived[w], sizeof(kv8_tile));
                warp::tma::load_async(v_smem[w], g.v, at, v_smem_arrived[w]);
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
        constexpr float NEG_INF = kittens::base_types::constants<float>::neg_infty();

        wait(q_arrived, 0);

        const int w = warpgroupid;
        for (int j = kv_first + w; j < kv_iters; j += CONSUMER_WARPGROUPS) {
            // The loop's `t`-th tile, in the phase the producer loaded it with.
            const int t = j - kv_first;
            const int phase = (t / CONSUMER_WARPGROUPS) % 2;
            wait(k_smem_arrived[w], phase);
            // K widened into this warpgroup's bf16 tile, whose last reader (the previous tile's
            // P·V) this warpgroup has waited on; made visible to the tensor cores.
            widen_kv(kv_bf[w], k_smem[w], KV_HEIGHT);
            warpgroup::mma_fence();
            warpgroup::sync(w + 4);
            warpgroup::mm_ABt(att_block, q_smem[0], kv_bf[w]);

            // The running max before this tile, unscaled (the rescale below reads it).
            warp::copy(max_vec_last, max_vec);

            // The WGMMA writes `att_block` asynchronously: wait before reading it.
            warpgroup::mma_async_wait();

            // Keys k0 .. k0 + 64 of tile j: drop key ≥ kv_len. No causal mask, as the one query
            // is the last key's position. Window: drop key < lo (`left_fill` returns at once for
            // `lo − k0 ≤ 0`, every tile but j0).
            const int k0 = j * KV_HEIGHT;
            warp::right_fill(att_block, att_block, kv_len - k0, NEG_INF);
            if constexpr (WINDOWED) {
                warp::left_fill(att_block, att_block, key_first - k0, NEG_INF);
            }

            warp::row_max(max_vec, att_block, max_vec);
            warp::mul(att_block, att_block, g.scale_log2);
            warp::mul(max_vec_scaled, max_vec, g.scale_log2);
            warp::sub_row(att_block, att_block, max_vec_scaled);
            warp::exp2(att_block, att_block);
            // The rescale of the running sum and output before this tile is accumulated into
            // them: exp2((m_old − m_new) · scale_log2), and EXACTLY 1 for a row whose max this
            // tile left unchanged (`attn_prefill_bf16.cu` gives the reason).
            scratchy::ops::rv_ternary_map<scratchy::ops::rescale_exact1>(
                max_vec_last, max_vec_last, max_vec, g.scale_log2);
            warp::mul(norm_vec, norm_vec, max_vec_last);
            warp::row_sum(norm_vec, att_block, norm_vec);
            warp::copy(att_block_mma, att_block);
            warp::mul_row(o_reg, o_reg, max_vec_last);

            wait(v_smem_arrived[w], phase);
            // V widened over K, which Q·Kᵀ finished reading at its wait above. A partial last
            // tile's rows of the slots at or past kv_len, which no append wrote, are +0 (0 · NaN =
            // NaN, #791).
            widen_kv(kv_bf[w], v_smem[w], kv_len - k0);
            warpgroup::mma_fence();
            warpgroup::sync(w + 4);
            warpgroup::mma_AB(o_reg, att_block_mma, kv_bf[w]);
            // Wait for the WGMMA to finish reading the buffer before releasing it to the
            // producer.
            warpgroup::mma_async_wait();

            if (warpgroup::laneid() == 0) {
                arrive(compute_done[w], 1);
            }
        }

        // Both warpgroups are past their last read of Q: its memory is the merge's.
        consumers::sync(CONSUMERS_BARRIER);
        if (warpgroup::warpid() == 0) {
            warp::store(o_part[w], o_reg);
            warp::store(m_part[w], max_vec);
            warp::store(l_part[w], norm_vec);
        }
        consumers::sync(CONSUMERS_BARRIER);

        if (w == 0 && warpgroup::warpid() == 0) {
            // The merge, in warpgroup order: this warp's own `max_vec` and `norm_vec` are side 0,
            // the loaded ones side 1. `max_vec_scaled` becomes `M`, `max_vec_last` side 1's max,
            // then its rescale, and `att`'s col-vectors are not used again.
            col_vec<rt_fl<16, KV_HEIGHT>> l1;
            warp::load(max_vec_last, m_part[1]);
            warp::load(l1, l_part[1]);
            warp::max(max_vec_scaled, max_vec, max_vec_last);
            scratchy::ops::rv_ternary_map<scratchy::ops::rescale_exact1>(
                max_vec, max_vec, max_vec_scaled, g.scale_log2);
            scratchy::ops::rv_ternary_map<scratchy::ops::rescale_exact1>(
                max_vec_last, max_vec_last, max_vec_scaled, g.scale_log2);
            warp::mul(norm_vec, norm_vec, max_vec);
            warp::mul(l1, l1, max_vec_last);
            warp::add(norm_vec, norm_vec, l1);
            warp::store(r_part[0], max_vec);
            warp::store(r_part[1], max_vec_last);
            warp::store(l_part[0], norm_vec);
            __syncwarp();
            // `mul_row`/`div_row` (`row_map`) index each shared tile through its swizzled
            // `operator[]`, and `add` (`bin_map`) through its raw `data[i]`, so a lane reads words
            // other lanes just wrote: a warp barrier between each hand-off. Only this warp is here,
            // so it is `__syncwarp`, never a group or named barrier.
            warp::mul_row(o_part[0], o_part[0], r_part[0]);
            warp::mul_row(o_part[1], o_part[1], r_part[1]);
            __syncwarp();
            warp::add(o_part[0], o_part[0], o_part[1]);
            __syncwarp();
            warp::div_row(o_part[0], o_part[0], l_part[0]);
            __syncwarp();
            warp::load(o_reg, o_part[0]);
            warp::store(o_bf[0], o_reg);
            __syncwarp();
            // Rows `0 .. hr`: kv head g's query heads, `hr · 512` contiguous bytes of `out`.
            const uint32_t bytes = uint32_t(g.hr) * HEAD_DIM * sizeof(bf16);
            warp::tma::store_async(g.out[{0, 0, 0, kv_head * g.hr * HEAD_DIM}], o_bf[0][0], bytes);
            warp::tma::store_async_wait();
        }
    }
}

}  // namespace

// The kernel's one-time setup: allow both instantiations `DECODE_SMEM_BYTES` of dynamic shared
// memory on the current device (the default limit is 48 KB). Run once per device before any launch
// (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch.
//
// 0 = done; any other value = the cudaError_t of setting the attribute, the unwindowed one's first.
extern "C" int scratchy_cuda_attn_decode_fp8kv_prepare(void) {
    const cudaError_t full = cudaFuncSetAttribute(
        attn_decode_fp8kv_ker<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, DECODE_SMEM_BYTES);
    if (full != cudaSuccess) {
        return static_cast<int>(full);
    }
    return static_cast<int>(cudaFuncSetAttribute(
        attn_decode_fp8kv_ker<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, DECODE_SMEM_BYTES));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value =
// the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// Refused before any pointer is touched: `head_dim` not 256; `q_heads` or `kv_heads` zero, or
// `q_heads` not a multiple of `kv_heads`; `q_heads / kv_heads` above 16
// (`step::ATTN_DECODE_BF16_MAX_GROUP`); `block_size` not 64; `num_blocks` zero; `scale` not finite
// or not above 0; `window` not a multiple of 64 (`step::ATTN_WINDOW_MULTIPLE`; 0, every key, is
// one); a dimension or the window above INT_MAX.
//
// `q` and `out` must be 16-byte-aligned device pointers to `q_heads · 256` bf16 (one row), `out`
// overlapping no input; `k_cache` and `v_cache` to one layer's `num_blocks · kv_heads · 64 · 256`
// e4m3 bytes; `sequence` to `[P, L]` u32 with `L ≥ 1`; `block_table` to the sequence's block ids,
// `ceil((P + L) / 64)` of them, each below `num_blocks`.
// `scratchy_cuda_attn_decode_fp8kv_prepare` must have succeeded on the current device. The launch
// is asynchronous on `stream`; the caller owns it and synchronizes.
extern "C" int scratchy_cuda_attn_decode_fp8kv(const void* q, const void* k_cache,
                                               const void* v_cache, const void* block_table,
                                               const void* sequence, void* out, uint32_t q_heads,
                                               uint32_t kv_heads, uint32_t head_dim,
                                               uint32_t block_size, uint32_t num_blocks,
                                               float scale, uint32_t window,
                                               void* stream /* cudaStream_t; null = default */) {
    if (head_dim != uint32_t(HEAD_DIM) || q_heads == 0 || kv_heads == 0 ||
        q_heads % kv_heads != 0 || q_heads / kv_heads > uint32_t(MAX_GROUP) ||
        block_size != BLOCK_SIZE || num_blocks == 0 || !std::isfinite(scale) || !(scale > 0.0f) ||
        window % BLOCK_SIZE != 0 || q_heads > uint32_t(INT_MAX) || num_blocks > uint32_t(INT_MAX) ||
        window > uint32_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        bf16* qp = static_cast<bf16*>(const_cast<void*>(q));
        bf16* op = static_cast<bf16*>(out);
        fp8e4m3* kp = static_cast<fp8e4m3*>(const_cast<void*>(k_cache));
        fp8e4m3* vp = static_cast<fp8e4m3*>(const_cast<void*>(v_cache));
        // Building each `gl` with a TMA type encodes its tensor map into host memory; nothing is
        // allocated on the device. Q as `[q_heads, 256]`; K and V as `[num_blocks, kv_heads, 64,
        // 256]`, whose bound on the block axis makes TMA zero-fill, never read past, a bad block;
        // `out` as `[1, q_heads · 256]`, with no tensor map.
        q_gl qg{qp, nullptr, nullptr, size_t(q_heads), nullptr};
        kv_gl kg{kp, size_t(num_blocks), size_t(kv_heads), nullptr, nullptr};
        kv_gl vg{vp, size_t(num_blocks), size_t(kv_heads), nullptr, nullptr};
        out_gl og{op, nullptr, nullptr, size_t(1), size_t(q_heads) * HEAD_DIM};
        // The index regions as `[1, entries]` int views: the block table's `MAX_BLOCKS_PER_SEQ`
        // entries, the sequence's two.
        index_gl tg{static_cast<int*>(const_cast<void*>(block_table)), nullptr, nullptr, size_t(1),
                    size_t(MAX_BLOCKS_PER_SEQ)};
        index_gl sg{static_cast<int*>(const_cast<void*>(sequence)), nullptr, nullptr, size_t(1),
                    size_t(2)};
        decode_globals G{qg, kg, vg, og, tg, sg, scale * LOG2E, int(q_heads / kv_heads),
                         int(window)};
        void* args[] = {&G};
        const dim3 grid(kv_heads), block(NUM_THREADS);
        auto* ker = window == 0 ? attn_decode_fp8kv_ker<false> : attn_decode_fp8kv_ker<true>;
        return static_cast<int>(cudaLaunchKernel(ker, grid, block, args,
                                                 DECODE_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
