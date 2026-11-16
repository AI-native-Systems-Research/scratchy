// SPDX-License-Identifier: Apache-2.0
//
// Grouped-query decode attention of one query row over the paged bf16 KV cache, for the H100 at
// head_dim 512 (Gemma 4's global layers). Q and O are the tape's `[1, q_heads · 512]` bf16; K and
// V are read from one layer's paged cache `[num_blocks, kv_heads, 64, 512]` bf16 through the
// sequence's block table.
//
//   o[h, :] = Σ_j softmax_j(scale · q[h, :] · k[j, h/hr, :]) · v[j, h/hr, :],   j < kv_len
//
// `attn_decode_bf16.cu`'s grid and stacked query tile, with `attn_prefill_d512_bf16.cu`'s column
// split. What differs from the D = 256 decode, and why:
//   - Grid `(kv_heads)`: one CTA per kv head g, its `hr` query heads stacked as rows 0 .. hr of a
//     Q of two `st_bf<64, 256>` halves, TMA-loaded at the element-unit row `g · hr` of q viewed as
//     `[q_heads, 512]` (rows past `q_heads` read zeros, TK-1's R1 gate). Rows `hr .. 63` are other
//     heads' queries or zeros; their outputs are discarded. `hr ≤ 16`.
//   - The column split, with NO even/odd tile split: both consumer warpgroups take EVERY KV tile
//     in ascending order, each keeping output columns [256c, 256c + 256). Their scores, max and
//     sum are the same bits (the same WGMMAs on the same tiles, as the D = 512 prefill), so there
//     is no merge. The D = 256 decode gives tiles alternately to its two warpgroups and merges
//     them, which changes the order of additions past 64 keys (#859); this one walks the tiles in
//     the prefill's order with the prefill's WGMMA shapes. Its row may therefore equal the
//     prefill's bit for bit at any kv_len: expected, not promised (the test prints it).
//   - The pipeline is the D = 512 prefill's: one K/V stage, K released after both warpgroups'
//     score wait (`k_done`), V after both warpgroups' P·V (`v_done`).
//   - No causal mask (the one query is at `kv_len − 1`); `right_fill` drops the slots at or past
//     kv_len; a partial last tile's V rows are zeroed in both halves by all 256 consumer threads,
//     fenced with `warpgroup::mma_fence()` (TK's `fence.proxy.async.shared::cta`), and met at
//     named barrier 1 before either P·V, as the D = 512 prefill.
//   - The window: `left_fill` of the keys below `lo = max(0, kv_len − W)`, the loop from the tile
//     holding `lo`, as the D = 256 decode. The first tile holds `lo`, an allowed key, so no tile a
//     warpgroup reads is wholly masked and no `neg_inf_to_zero` is needed. Gemma's global layers
//     launch `W = 0`.
//   - The store: warp 0 of warpgroup c divides its 16 rows by their sum in registers (as the
//     prefill does), stores them into its own plain `st_bf<16, 256, false>` (`warp::store`, the
//     one rounding), and after a `__syncwarp` bulk-copies each of rows `0 .. hr` (query head
//     `g · hr + r`, its 512 bytes of half c, at `out[(g·hr + r)·512 + 256c]`): a half-row is
//     contiguous in `out`, a half of `hr` rows is not (the other half sits between them), so it is
//     `hr` copies of 512 bytes, not one of `hr · 512`. No tensor map over `out`; rows `hr .. 15`
//     are never written.
//
// Shared memory: Q 2 tiles (64 KiB), K and V 2 tiles each (128 KiB), the two output halves
// (16 KiB), plus 1,024 bytes of alignment slack = 214,016 bytes of dynamic shared memory, set once
// by `scratchy_cuda_attn_decode_d512_bf16_prepare` (in `PREPARE`). The semaphores are static.

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

// `step::ATTN_D512_BF16_HEAD_DIM`, as two halves of the D = 256 kernels' tile width.
constexpr int HEAD_DIM = 512;
constexpr int HALF_DIM = 256;
constexpr int HALVES = HEAD_DIM / HALF_DIM;
// Query rows of the stacked tile, and K/V rows per tile: the KV block size.
constexpr int QO_HEIGHT = 64;
constexpr int KV_HEIGHT = 64;
// `step::ATTN_DECODE_BF16_MAX_GROUP`: the rows of a consumer warpgroup's warp 0, the ones stored.
constexpr int MAX_GROUP = 16;
constexpr int CONSUMER_WARPGROUPS = 2;
static_assert(CONSUMER_WARPGROUPS == HALVES, "one output half per consumer warpgroup");
constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + 1;
constexpr int NUM_WARPS = NUM_WARPGROUPS * kittens::WARPGROUP_WARPS;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;
constexpr int CONSUMER_THREADS = CONSUMER_WARPGROUPS * kittens::WARPGROUP_THREADS;
// The named barrier over the two consumer warpgroups; 0 is `__syncthreads`.
constexpr int CONSUMERS_BARRIER = 1;
using consumers = kittens::group<CONSUMER_WARPGROUPS * kittens::WARPGROUP_WARPS>;
// Token slots per KV block (`step::CUDA_KV_BLOCK_SIZE`): one block is one K (or V) tile.
constexpr uint32_t BLOCK_SIZE = 64;
static_assert(KV_HEIGHT == int(BLOCK_SIZE), "one paged (block, kv head) half is one TMA box");
// Entries of a sequence's block table (`step::CUDA_MAX_BLOCKS_PER_SEQ`): the columns of its view.
constexpr uint32_t MAX_BLOCKS_PER_SEQ = 128;
// log2(e): the softmax runs in base 2 with the scale folded in.
constexpr float LOG2E = 1.44269504089f;

using half_tile = st_bf<QO_HEIGHT, HALF_DIM>;
static_assert(KV_HEIGHT == QO_HEIGHT, "Q, K and V halves are one tile type");
// One warpgroup's stored rows, plain, so each row is contiguous for its bulk copy.
using out_tile = st_bf<MAX_GROUP, HALF_DIM, false>;

// q as `[q_heads, 512]`, the decode row viewed head-major: a box's rows are query heads.
using q_gl = gl<bf16, 1, 1, -1, HEAD_DIM, half_tile>;
using kv_gl = gl<bf16, -1, -1, KV_HEIGHT, HEAD_DIM, half_tile>;
// The output row `[1, q_heads · 512]`, no TMA type: written by bulk copy only.
using out_gl = gl<bf16, 1, 1, -1, -1>;
// An index region, `[1, entries]` int with no TMA type: read element by element.
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

constexpr size_t SMEM_TILES = 3 * HALVES * sizeof(half_tile) + HALVES * sizeof(out_tile);
static_assert(SMEM_TILES == 212992, "2 Q + 2 K + 2 V tiles of [64, 256], 2 of [16, 256] bf16");
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int DECODE_SMEM_BYTES = int(SMEM_TILES) + 1024;
static_assert(DECODE_SMEM_BYTES == 214016, "#923's decode budget");
static_assert(DECODE_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

template <bool WINDOWED>
__global__ __launch_bounds__(NUM_THREADS, 1) void attn_decode_d512_bf16_ker(
    const __grid_constant__ decode_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    const int warpid = kittens::warpid(), warpgroupid = warpid / kittens::WARPGROUP_WARPS;

    half_tile(&q_smem)[HALVES] = al.allocate<half_tile, HALVES>();
    half_tile(&k_smem)[HALVES] = al.allocate<half_tile, HALVES>();
    half_tile(&v_smem)[HALVES] = al.allocate<half_tile, HALVES>();
    out_tile(&o_bf)[HALVES] = al.allocate<out_tile, HALVES>();

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

    __shared__ kittens::semaphore q_arrived, k_arrived, v_arrived, k_done, v_done;
    if (threadIdx.x == 0) {
        init_semaphore(q_arrived, 0, 1);
        init_semaphore(k_arrived, 0, 1);
        init_semaphore(v_arrived, 0, 1);
        init_semaphore(k_done, CONSUMER_WARPGROUPS, 0);
        init_semaphore(v_done, CONSUMER_WARPGROUPS, 0);
        tma::expect_bytes(q_arrived, sizeof(q_smem));
        for (int c = 0; c < HALVES; c++) {
            // Rows `g·hr .. g·hr + 64` of `[q_heads, 512]`, columns of half c, in element units.
            coord<> at = {0, 0, kv_head * g.hr, c * HALF_DIM};
            tma::load_async(q_smem[c], g.q, at, q_arrived);
        }
    }
    __syncthreads();

    if (warpgroupid == NUM_WARPGROUPS - 1) {
        // The producer: the D = 512 prefill's, K once both consumers read tile t − 1's K, V once
        // both read its V.
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
        constexpr float NEG_INF = kittens::base_types::constants<float>::neg_infty();
        const int c = warpgroupid;

        wait(q_arrived, 0);

        for (int j = kv_first; j < kv_iters; j++) {
            const int t = j - kv_first;
            const int phase = t % 2;
            wait(k_arrived, phase);
            warpgroup::mm_ABt(att_block, q_smem[0], k_smem[0]);
            warpgroup::mma_ABt(att_block, q_smem[1], k_smem[1]);

            warp::copy(max_vec_last, max_vec);

            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) {
                arrive(k_done, 1);
            }

            // Drop key ≥ kv_len; no causal mask. Window: drop key < lo (`left_fill` returns at
            // once for `lo − k0 ≤ 0`, every tile but the first).
            const int k0 = j * KV_HEIGHT;
            warp::right_fill(att_block, att_block, kv_len - k0, NEG_INF);
            if constexpr (WINDOWED) {
                warp::left_fill(att_block, att_block, key_first - k0, NEG_INF);
            }

            // The D = 256 kernels' online softmax, line for line.
            warp::row_max(max_vec, att_block, max_vec);
            warp::mul(att_block, att_block, g.scale_log2);
            warp::mul(max_vec_scaled, max_vec, g.scale_log2);
            warp::sub_row(att_block, att_block, max_vec_scaled);
            warp::exp2(att_block, att_block);
            scratchy::ops::rv_ternary_map<scratchy::ops::rescale_exact1>(
                max_vec_last, max_vec_last, max_vec, g.scale_log2);
            warp::mul(norm_vec, norm_vec, max_vec_last);
            warp::row_sum(norm_vec, att_block, norm_vec);
            warp::copy(att_block_mma, att_block);
            warp::mul_row(o_reg, o_reg, max_vec_last);

            wait(v_arrived, phase);
            // A partial last tile: zero the V rows of the slots at or past kv_len in both halves
            // before either warpgroup's P·V. `valid` is the same for the whole CTA.
            const int valid = kv_len - k0;
            if (valid < KV_HEIGHT) {
                const int tid = warpgroupid * kittens::WARPGROUP_THREADS + warpgroup::laneid();
                for (int e = tid; e < (KV_HEIGHT - valid) * HEAD_DIM; e += CONSUMER_THREADS) {
                    const int col = e % HEAD_DIM;
                    v_smem[col / HALF_DIM][{valid + e / HEAD_DIM, col % HALF_DIM}] =
                        kittens::base_types::constants<bf16>::zero();
                }
                warpgroup::mma_fence();
                consumers::sync(CONSUMERS_BARRIER);
            }
            warpgroup::mma_AB(o_reg, att_block_mma, v_smem[c]);
            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) {
                arrive(v_done, 1);
            }
        }

        if (warpgroup::warpid() == 0) {
            // Rows 0 .. 16 of half c: divided in registers, rounded once into the plain tile.
            warp::div_row(o_reg, o_reg, norm_vec);
            warp::store(o_bf[c], o_reg);
            // The register-to-shared store and the bulk copy's reads split the tile across lanes
            // differently: a warp barrier between them. Only this warp is here, so `__syncwarp`.
            __syncwarp();
            const uint32_t bytes = HALF_DIM * sizeof(bf16);
            for (int r = 0; r < g.hr; r++) {
                const int at = (kv_head * g.hr + r) * HEAD_DIM + c * HALF_DIM;
                warp::tma::store_async(g.out[{0, 0, 0, at}], o_bf[c][r * HALF_DIM], bytes);
            }
            warp::tma::store_async_wait();
        }
    }
}

}  // namespace

// The kernel's one-time setup: allow both instantiations `DECODE_SMEM_BYTES` of dynamic shared
// memory on the current device. Run once per device before any launch (`kernels::prepare_all`).
//
// 0 = done; any other value = the cudaError_t of setting the attribute, the unwindowed one's first.
extern "C" int scratchy_cuda_attn_decode_d512_bf16_prepare(void) {
    const cudaError_t full =
        cudaFuncSetAttribute(attn_decode_d512_bf16_ker<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, DECODE_SMEM_BYTES);
    if (full != cudaSuccess) {
        return static_cast<int>(full);
    }
    return static_cast<int>(cudaFuncSetAttribute(attn_decode_d512_bf16_ker<true>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 DECODE_SMEM_BYTES));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value =
// the cudaError_t cudaLaunchKernel returned for THIS launch.
//
// Refused before any pointer is touched: `head_dim` not 512; `q_heads` or `kv_heads` zero, or
// `q_heads` not a multiple of `kv_heads`; `q_heads / kv_heads` above 16
// (`step::ATTN_DECODE_BF16_MAX_GROUP`); `block_size` not 64; `num_blocks` zero; `scale` not finite
// or not above 0; `window` not a multiple of 64; a dimension or the window above INT_MAX.
//
// `q` and `out` must be 16-byte-aligned device pointers to `q_heads · 512` bf16 (one row), `out`
// overlapping no input; `k_cache` and `v_cache` to one layer's `num_blocks · kv_heads · 64 · 512`
// bf16; `sequence` and `block_table` as `scratchy_cuda_attn_decode_bf16`'s.
// `scratchy_cuda_attn_decode_d512_bf16_prepare` must have succeeded on the current device. The
// launch is asynchronous on `stream`.
extern "C" int scratchy_cuda_attn_decode_d512_bf16(
    const void* q, const void* k_cache, const void* v_cache, const void* block_table,
    const void* sequence, void* out, uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
    uint32_t block_size, uint32_t num_blocks, float scale, uint32_t window,
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
        bf16* kp = static_cast<bf16*>(const_cast<void*>(k_cache));
        bf16* vp = static_cast<bf16*>(const_cast<void*>(v_cache));
        q_gl qg{qp, nullptr, nullptr, size_t(q_heads), nullptr};
        kv_gl kg{kp, size_t(num_blocks), size_t(kv_heads), nullptr, nullptr};
        kv_gl vg{vp, size_t(num_blocks), size_t(kv_heads), nullptr, nullptr};
        out_gl og{op, nullptr, nullptr, size_t(1), size_t(q_heads) * HEAD_DIM};
        index_gl tg{static_cast<int*>(const_cast<void*>(block_table)), nullptr, nullptr, size_t(1),
                    size_t(MAX_BLOCKS_PER_SEQ)};
        index_gl sg{static_cast<int*>(const_cast<void*>(sequence)), nullptr, nullptr, size_t(1),
                    size_t(2)};
        decode_globals G{qg, kg, vg, og, tg, sg, scale * LOG2E, int(q_heads / kv_heads),
                         int(window)};
        void* args[] = {&G};
        const dim3 grid(kv_heads), block(NUM_THREADS);
        auto* ker =
            window == 0 ? attn_decode_d512_bf16_ker<false> : attn_decode_d512_bf16_ker<true>;
        return static_cast<int>(cudaLaunchKernel(ker, grid, block, args, DECODE_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
