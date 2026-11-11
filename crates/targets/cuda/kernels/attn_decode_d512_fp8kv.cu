// SPDX-License-Identifier: Apache-2.0
//
// Grouped-query decode attention of one query row over the paged e4m3 KV cache, for the H100 at
// head_dim 512 (Gemma 4's global layers; FP8 S5): `attn_decode_d512_bf16.cu`, which stays as its
// test-only reference, with K and V read as e4m3 and widened to bf16 in shared memory before the
// same math. The cache is one layer's `[num_blocks, kv_heads, 64, 512]` e4m3, one byte an element,
// at a fixed scale of 1.0 (`kv_append_fp8.cu`); Q and O are the tape's `[1, q_heads · 512]` bf16.
//
// What differs from the bf16 kernel, and why:
//   - K and V land as two `st_fp8e4m3<64, 256>` halves each (16 KiB a half, half the TMA bytes).
//     Warpgroup c widens half c into its own bf16 half W[c], 64 columns at a time:
//     `warpgroup::load` of an `rt_fp8e4m3<16, 64>` through a `subtile<64, 64>` view (TK's 8-bit
//     `ldmatrix` path), `warp::copy` to `rt_bf<16, 64>` (exact: every e4m3 value is a bf16 value),
//     `warpgroup::store` back through W[c]'s view. One bf16 pair for both K and V keeps the dynamic
//     shared memory at today's 214,016 bytes (separate K and V pairs would be 262,144).
//   - The order per tile: K half c into W[c], `warpgroup::mma_fence()` (TK's `wgmma.fence` +
//     `fence.proxy.async.shared::cta`: the widen writes through the generic proxy, WGMMA reads
//     through the async one), the consumers' barrier 1; both warpgroups' Q·Kᵀ over W[0] and W[1],
//     its wait, `k_done`, the softmax, `v_arrived`, and barrier 1 again (the other warpgroup has
//     stopped reading W[c] as K); then V half c into W[c], the fence, the warpgroup's own named
//     barrier c + 4 (only c reads it for P·V), and P·V. The next tile's K widen into W[c] is safe:
//     only warpgroup c read it last. So no inline PTX.
//   - The V rows of slots at or past kv_len (an unwritten e4m3 byte may be NaN) are zeroed in
//     registers during the widen, `warp::lower_fill` with +0, for the bf16 kernel's shared-memory
//     loop: the same +0 rows. K needs nothing, as there: `right_fill` overwrites those scores.
//   - After the widen the math is the bf16 kernel's, line for line, on bf16 tiles, so its output
//     is the bf16 kernel's on the widened cache, bit for bit (`tests/attention_fp8kv.rs`).
//
// The bf16 kernel's description of the grid, the stacked query tile, the column split with no
// merge, the window and the store holds unchanged (`attn_decode_d512_bf16.cu`).
//
// The split over keys (FP8 S8 W6b): `attn_decode_fp8kv.cu`'s, on this kernel. With more than one
// used split, warpgroup c stores its unnormalised half of the rows (`o_split`, its half's 256
// columns) and warpgroup 0 the rows' max and sum (`ml_split`; both warpgroups hold the same, from
// the same scores), for `attn_decode_merge.cu`.
//
// Shared memory: Q 2 halves (64 KiB), e4m3 K and V 2 halves each (64 KiB), the bf16 pair W (64 KiB),
// the two output halves (16 KiB), plus 1,024 bytes of alignment slack = 214,016 bytes of dynamic
// shared memory, set once by `scratchy_cuda_attn_decode_d512_fp8kv_prepare` (in `PREPARE`). The
// semaphores are static.

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
// The named barrier over the two consumer warpgroups (ids 4 and 5 are each warpgroup's own, before
// its P·V on its widened V half; 0 is `__syncthreads`).
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
static_assert(KV_HEIGHT == QO_HEIGHT, "Q, O and the widened K and V halves are one tile type");
// A K or V half as cached.
using kv8_half = st_fp8e4m3<KV_HEIGHT, HALF_DIM>;
// Columns per widen step: one `rt_fp8e4m3<16, 64>` and one `rt_bf<16, 64>`, 24 registers a thread.
constexpr int WIDEN_COLS = 64;
constexpr int WIDEN_CHUNKS = HALF_DIM / WIDEN_COLS;
// One warpgroup's stored rows, plain, so each row is contiguous for its bulk copy.
using out_tile = st_bf<MAX_GROUP, HALF_DIM, false>;

// q as `[q_heads, 512]`, the decode row viewed head-major: a box's rows are query heads.
using q_gl = gl<bf16, 1, 1, -1, HEAD_DIM, half_tile>;
using kv_gl = gl<fp8e4m3, -1, -1, KV_HEIGHT, HEAD_DIM, kv8_half>;
// The output row `[1, q_heads · 512]`, no TMA type: written by bulk copy only.
using out_gl = gl<bf16, 1, 1, -1, -1>;
// The split partials (`[S, kv_heads, 16, 512]`, `[S, kv_heads, 2, 16]` f32), no TMA type.
using split_gl = gl<float, -1, -1, MAX_GROUP, HEAD_DIM>;
using ml_gl = gl<float, -1, -1, 2, MAX_GROUP>;
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
    // The split: `S` and the fewest tiles a split takes, and its partials (unused at one split).
    int splits;
    int min_tiles;
    split_gl o_split;
    ml_gl ml_split;
};

constexpr size_t SMEM_TILES =
    2 * HALVES * sizeof(half_tile) + 2 * HALVES * sizeof(kv8_half) + HALVES * sizeof(out_tile);
static_assert(SMEM_TILES == 212992,
              "2 Q and 2 widened bf16 halves of [64, 256], 2 e4m3 K and V halves, 2 of [16, 256]");
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int DECODE_SMEM_BYTES = int(SMEM_TILES) + 1024;
static_assert(DECODE_SMEM_BYTES == 214016, "#923's decode budget, the bf16 decode's");
static_assert(DECODE_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

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
__global__ __launch_bounds__(NUM_THREADS, 1) void attn_decode_d512_fp8kv_ker(
    const __grid_constant__ decode_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    const int warpid = kittens::warpid(), warpgroupid = warpid / kittens::WARPGROUP_WARPS;

    half_tile(&q_smem)[HALVES] = al.allocate<half_tile, HALVES>();
    kv8_half(&k_smem)[HALVES] = al.allocate<kv8_half, HALVES>();
    kv8_half(&v_smem)[HALVES] = al.allocate<kv8_half, HALVES>();
    // Half c widened by warpgroup c: K for both warpgroups' Q·Kᵀ, then V for its own P·V.
    half_tile(&kv_bf)[HALVES] = al.allocate<half_tile, HALVES>();
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
    // This CTA's split: tiles `j_lo .. j_hi` of the window's (`attn_decode_fp8kv.cu`'s rule). A
    // split with no tiles leaves before any shared state, every thread of it alike.
    const int split = blockIdx.y;
    const int n_tiles = kv_iters - kv_first;
    const int per = max(g.min_tiles, (n_tiles + g.splits - 1) / g.splits);
    const int used = (n_tiles + per - 1) / per;
    if (split >= used) {
        return;
    }
    const int j_lo = kv_first + split * per;
    const int j_hi = min(kv_iters, j_lo + per);

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
            for (int j = j_lo; j < j_hi; j++) {
                const int t = j - j_lo;
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
        constexpr float NEG_INF = kittens::base_types::constants<float>::neg_infty();
        const int c = warpgroupid;

        wait(q_arrived, 0);

        for (int j = j_lo; j < j_hi; j++) {
            const int t = j - j_lo;
            const int phase = t % 2;
            wait(k_arrived, phase);
            // Half c of K widened into this warpgroup's bf16 half, whose last reader (its own
            // previous P·V) it has waited on; both halves visible to the tensor cores.
            widen_kv(kv_bf[c], k_smem[c], KV_HEIGHT);
            warpgroup::mma_fence();
            consumers::sync(CONSUMERS_BARRIER);
            warpgroup::mm_ABt(att_block, q_smem[0], kv_bf[0]);
            warpgroup::mma_ABt(att_block, q_smem[1], kv_bf[1]);

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
            // The running sum `l · r + Σ p`, as ONE fused multiply-add: in the bf16 kernel nvcc
            // contracts its `mul` then `row_sum` into one FFMA per packed row, and in this kernel's
            // code shape it does not (two roundings: the 8–99 bits of the H100 pod at 134c057d).
            // So the sum is formed alone (`row_sum` with no accumulator) and fused explicitly
            // (`fma_acc`, `fmaf(r, l, Σ p)`): the bf16 kernel's bits, by construction.
            col_vec<rt_fl<16, KV_HEIGHT>> tile_sum;
            warp::row_sum(tile_sum, att_block);
            scratchy::ops::rv_fma<scratchy::ops::fma_acc>(norm_vec, max_vec_last, norm_vec,
                                                          tile_sum);
            warp::copy(att_block_mma, att_block);
            warp::mul_row(o_reg, o_reg, max_vec_last);

            wait(v_arrived, phase);
            // Half c of V widened over K; only this warpgroup reads it. A partial last tile's rows
            // of the slots at or past kv_len, which no append wrote, are +0 (0 · NaN = NaN, #791).
            // Both warpgroups past this tile's Q·Kᵀ (each waited on it before its softmax), so W[c]
            // is free for V. The barrier sits here, not between the score wait and the softmax:
            // the code from the score wait to `wait(v_arrived)` is the bf16 kernel's, token for
            // token, so nvcc and ptxas compile the softmax as they do there.
            consumers::sync(CONSUMERS_BARRIER);
            widen_kv(kv_bf[c], v_smem[c], kv_len - k0);
            warpgroup::mma_fence();
            warpgroup::sync(c + 4);
            warpgroup::mma_AB(o_reg, att_block_mma, kv_bf[c]);
            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) {
                arrive(v_done, 1);
            }
        }

        if (warpgroup::warpid() == 0 && used > 1) {
            // A split's partials: half c of its unnormalised rows; the rows' max and sum once.
            warp::store(g.o_split, o_reg, {split, kv_head, 0, c});
            if (c == 0) {
                warp::store(g.ml_split, max_vec, {split, kv_head, 0, 0});
                warp::store(g.ml_split, norm_vec, {split, kv_head, 1, 0});
            }
        } else if (warpgroup::warpid() == 0) {
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
extern "C" int scratchy_cuda_attn_decode_d512_fp8kv_prepare(void) {
    const cudaError_t full =
        cudaFuncSetAttribute(attn_decode_d512_fp8kv_ker<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, DECODE_SMEM_BYTES);
    if (full != cudaSuccess) {
        return static_cast<int>(full);
    }
    return static_cast<int>(cudaFuncSetAttribute(attn_decode_d512_fp8kv_ker<true>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 DECODE_SMEM_BYTES));
}

// The launch at `splits` × `min_tiles` (one split: the unsplit kernel), `o_split` and `ml_split`
// its partials (unused, and may be null, at one split). Its shape checks are the callers'. Can
// throw (a tensor-map encode).
static int launch_decode(const void* q, const void* k_cache, const void* v_cache,
                         const void* block_table, const void* sequence, void* out,
                         uint32_t q_heads, uint32_t kv_heads, uint32_t num_blocks, float scale,
                         uint32_t window, uint32_t splits, uint32_t min_tiles, void* o_split,
                         void* ml_split, void* stream) {
    bf16* qp = static_cast<bf16*>(const_cast<void*>(q));
    bf16* op = static_cast<bf16*>(out);
    fp8e4m3* kp = static_cast<fp8e4m3*>(const_cast<void*>(k_cache));
    fp8e4m3* vp = static_cast<fp8e4m3*>(const_cast<void*>(v_cache));
    q_gl qg{qp, nullptr, nullptr, size_t(q_heads), nullptr};
    kv_gl kg{kp, size_t(num_blocks), size_t(kv_heads), nullptr, nullptr};
    kv_gl vg{vp, size_t(num_blocks), size_t(kv_heads), nullptr, nullptr};
    out_gl og{op, nullptr, nullptr, size_t(1), size_t(q_heads) * HEAD_DIM};
    index_gl tg{static_cast<int*>(const_cast<void*>(block_table)), nullptr, nullptr, size_t(1),
                size_t(MAX_BLOCKS_PER_SEQ)};
    index_gl sg{static_cast<int*>(const_cast<void*>(sequence)), nullptr, nullptr, size_t(1),
                size_t(2)};
    // The partials' views (no tensor map), over `out` when a launch of one split has none.
    split_gl osg{static_cast<float*>(o_split != nullptr ? o_split : out), size_t(splits),
                 size_t(kv_heads), nullptr, nullptr};
    ml_gl mlg{static_cast<float*>(ml_split != nullptr ? ml_split : out), size_t(splits),
              size_t(kv_heads), nullptr, nullptr};
    decode_globals G{qg,          kg,  vg,         og,  tg,  sg, scale * LOG2E, int(q_heads / kv_heads),
                     int(window), int(splits), int(min_tiles), osg, mlg};
    void* args[] = {&G};
    const dim3 grid{kv_heads, splits}, block{uint32_t(NUM_THREADS)};
    auto* ker = window == 0 ? attn_decode_d512_fp8kv_ker<false> : attn_decode_d512_fp8kv_ker<true>;
    return static_cast<int>(
        cudaLaunchKernel(ker, grid, block, args, DECODE_SMEM_BYTES, static_cast<cudaStream_t>(stream)));
}

// Whether the decode's shape is refused (the list below `scratchy_cuda_attn_decode_d512_fp8kv`'s).
static bool decode_refused(uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                           uint32_t block_size, uint32_t num_blocks, float scale, uint32_t window) {
    return head_dim != uint32_t(HEAD_DIM) || q_heads == 0 || kv_heads == 0 ||
           q_heads % kv_heads != 0 || q_heads / kv_heads > uint32_t(MAX_GROUP) ||
           block_size != BLOCK_SIZE || num_blocks == 0 || !std::isfinite(scale) || !(scale > 0.0f) ||
           window % BLOCK_SIZE != 0 || q_heads > uint32_t(INT_MAX) || num_blocks > uint32_t(INT_MAX) ||
           window > uint32_t(INT_MAX);
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
// e4m3 bytes; `sequence` and `block_table` as `scratchy_cuda_attn_decode_bf16`'s.
// `scratchy_cuda_attn_decode_d512_fp8kv_prepare` must have succeeded on the current device. The
// launch is asynchronous on `stream`.
extern "C" int scratchy_cuda_attn_decode_d512_fp8kv(
    const void* q, const void* k_cache, const void* v_cache, const void* block_table,
    const void* sequence, void* out, uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
    uint32_t block_size, uint32_t num_blocks, float scale, uint32_t window,
    void* stream /* cudaStream_t; null = default */) {
    if (decode_refused(q_heads, kv_heads, head_dim, block_size, num_blocks, scale, window)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        return launch_decode(q, k_cache, v_cache, block_table, sequence, out, q_heads, kv_heads,
                             num_blocks, scale, window, 1, 1, nullptr, nullptr, stream);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// The split decode (W6b): `scratchy_cuda_attn_decode_d512_fp8kv` on a grid of `kv_heads × splits`,
// as `scratchy_cuda_attn_decode_fp8kv_split` (its partials `[splits, kv_heads, 16, 512]` and
// `[splits, kv_heads, 2, 16]` f32), with its refusals.
extern "C" int scratchy_cuda_attn_decode_d512_fp8kv_split(
    const void* q, const void* k_cache, const void* v_cache, const void* block_table,
    const void* sequence, void* out, uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
    uint32_t block_size, uint32_t num_blocks, float scale, uint32_t window, uint32_t splits,
    uint32_t min_tiles, void* o_split, void* ml_split,
    void* stream /* cudaStream_t; null = default */) {
    if (decode_refused(q_heads, kv_heads, head_dim, block_size, num_blocks, scale, window) ||
        splits == 0 || splits > 64 || min_tiles == 0 || min_tiles > uint32_t(MAX_BLOCKS_PER_SEQ) ||
        (splits > 1 && (o_split == nullptr || ml_split == nullptr))) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        return launch_decode(q, k_cache, v_cache, block_table, sequence, out, q_heads, kv_heads,
                             num_blocks, scale, window, splits, min_tiles, o_split, ml_split,
                             stream);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
