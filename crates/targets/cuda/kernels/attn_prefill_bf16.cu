// SPDX-License-Identifier: Apache-2.0
//
// Causal grouped-query attention of one prefill bucket over the paged bf16 KV cache, for the H100
// at head_dim 256. Q and O are the tape's token-major [rows, heads · 256] bf16; K and V are read
// from one layer's paged cache [num_blocks, kv_heads, 64, 256] bf16 through the sequence's block
// table. The softmax runs online in f32 (flash attention): the full score matrix is never stored.
//
//   o[i, h, :] = Σ_j softmax_j(scale · q[i, h, :] · k[j, h/hr, :]) · v[j, h/hr, :]
//
// over the allowed keys j ≤ P + i and j < kv_len = P + L, where `P` (the cached prefix length)
// and `L` (this forward's real rows) are read from the `sequence` region, and `hr = q_heads /
// kv_heads` is the GQA group: query head h reads kv head h / hr.
//
// Adapted from ThunderKittens `kernels/attention/mha_h100/mha_h100.cu` `fwd_attend_ker` at
// 67845f5f (MIT, see `thunderkittens/LICENSE`). Kept from upstream: the producer/consumer
// warpgroup structure; the semaphores `k_smem_arrived` / `v_smem_arrived` / `compute_done` with
// `tma::expect_bytes` before every load; `warpgroup::mm_ABt(att, q_smem, k_smem)` then
// `mma_async_wait`; the online softmax in f32 with `exp2` and the scale folded into log2(e); the
// scores copied to bf16 and `warpgroup::mma_AB(o, att_bf, v_smem)` then `mma_async_wait`;
// `div_row` by the running sum; the O store through `o_smem` aliased onto `q_smem`. What differs,
// and why:
//   - D = 256, so `kv_height` is 64 (not 128) and there are 2 consumer warpgroups (not 3): at
//     D = 256 a 128-row K/V tile is 64 KiB, and three 64-row query tiles plus two stages of K and
//     V do not fit 227 KiB. 2 · 32 KiB (q) + 2 · 32 KiB (K) + 2 · 32 KiB (V) = 192 KiB.
//   - Registers: `increase_registers<240>` / `decrease_registers<24>` (2 · 128 · 240 + 128 · 24 =
//     64,512 ≤ 65,536), so a consumer holds its `rt_fl<16, 256>` accumulator (128 f32) without
//     spilling.
//   - The scale is a launch argument (`scale · log2(e)`), not the hard-coded `1/sqrt(128)`.
//   - The logsumexp output `L` (for a backward pass) and its `l_smem` are dropped.
//   - The causal mask is element-wise on every tile (`warp::tril` keeps key ≤ P + row, then
//     `warp::right_fill` drops keys ≥ kv_len), not upstream's 16x16 tile-diagonal `make_causal`,
//     which assumes query and key tiles start together; here the queries start at P, which need
//     not be a multiple of anything.
//   - The masks alone do NOT make unwritten cache slots harmless. When kv_len is not a multiple
//     of 64, the last tile holds slots at or past kv_len that no append wrote (the pool is never
//     cleared, so they hold anything, NaN included). Their scores are masked to −inf, so their
//     p is 0, but the P·V WGMMA still multiplies p by their V rows, and 0 · NaN = NaN. So the
//     consumers zero those V rows [kv_len − k0, 64) in shared memory before either P·V: after
//     `v_smem_arrived`, all 256 consumer threads write the zeros (through the swizzled tile's
//     `operator[]`), fence them to the async proxy the tensor cores read through
//     (`fence.proxy.async.shared::cta`), and meet at named barrier 1 before either warpgroup's
//     `mma_AB`. The stage was released to the producer only after both warpgroups' previous
//     `mma_async_wait`, so nothing reads it while it is zeroed. K needs nothing: its scores are
//     overwritten with −inf by `right_fill` before `row_max`.
//   - K and V come from the paged cache: KV tile j is block `block_table[j]`, one TMA box of
//     `[64, 256]` at `{block, kv_head, 0, 0}` (block_size = kv_height = 64, so one box fills the
//     tile). The producer reads the block id from global memory before each load.
//   - The index regions (`block_table`, `sequence`) are TK `gl<int>` views, read element by
//     element; the online softmax's rescale is the `rescale_exact1` op of `scratchy_ops.cuh`.
//   - Q and O are token-major `[1, rows, heads, 256]`: their `gl`s use TMA `axis = 1`, so a tile's
//     64 rows run along the token dimension at a fixed head (`types/global/tma.cuh`,
//     `coord::unit_coord`), and no transpose is needed.
//   - The KV loop bound is the CTA's last allowed key: `ceil((min(P + row_last, kv_len − 1) + 1) /
//     64)` tiles. The producer loads every tile the loop reads, so the pipeline never waits on a
//     load it did not issue; the upstream preload (stage 0 at entry) is folded into the producer's
//     loop.
//   - Upstream's harness, causal template flag and the backward kernels are dropped, and so is
//     its `warp::add(att_block, att_block, 0.f)` before the bf16 copy, which changes no value.
//
// Shared memory: 196,608 bytes of tiles plus 1,024 bytes of alignment slack (the swizzled TMA
// tiles need 1024-byte alignment) = 197,632 bytes of dynamic shared memory, set once by
// `scratchy_cuda_attn_prefill_bf16_prepare` (in `PREPARE`, as the GEMM's). The 7 semaphores (q, and
// K, V and compute-done per stage) are static shared memory, in what is left of 227 KiB.

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
constexpr int CONSUMER_THREADS = CONSUMER_WARPGROUPS * kittens::WARPGROUP_WARPS *
                                 kittens::WARP_THREADS;
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
using k_tile = st_bf<KV_HEIGHT, HEAD_DIM>;
using v_tile = st_bf<KV_HEIGHT, HEAD_DIM>;
using o_tile = st_bf<QO_HEIGHT, HEAD_DIM>;

// `axis = 1`: a Q (or O) tile's rows are 64 consecutive tokens of one head of the token-major
// `[1, rows, heads, 256]` view.
using qo_gl = gl<bf16, 1, -1, -1, HEAD_DIM, tma::descriptor<q_tile, 1>>;
using kv_gl = gl<bf16, -1, -1, KV_HEIGHT, HEAD_DIM, k_tile>;
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
};

constexpr size_t SMEM_TILES =
    CONSUMER_WARPGROUPS * sizeof(q_tile) + STAGES * sizeof(k_tile) + STAGES * sizeof(v_tile);
static_assert(SMEM_TILES == 196608, "2 q + 2 K + 2 V tiles of [64, 256] bf16");
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int ATTN_SMEM_BYTES = int(SMEM_TILES) + 1024;
static_assert(ATTN_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

__global__ __launch_bounds__(NUM_THREADS, 1) void attn_prefill_bf16_ker(
    const __grid_constant__ attn_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    const int warpid = kittens::warpid(), warpgroupid = warpid / kittens::WARPGROUP_WARPS;

    q_tile(&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
    k_tile(&k_smem)[STAGES] = al.allocate<k_tile, STAGES>();
    v_tile(&v_smem)[STAGES] = al.allocate<v_tile, STAGES>();
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
        // The producer: one warp loads KV tile `j` into stage `j % STAGES`, after the consumers
        // released that stage's previous tile `j - STAGES`.
        warpgroup::decrease_registers<24>();
        if (warpid == NUM_WARPS - 4) {
            for (int j = 0; j < kv_iters; j++) {
                const int stage = j % STAGES;
                if (j >= STAGES) {
                    wait(compute_done[stage], ((j - STAGES) / STAGES) % 2);
                }
                const int block = g.block_table[{0, 0, 0, j}];
                coord<k_tile> at = {block, kv_head, 0, 0};
                warp::tma::expect_bytes(k_smem_arrived[stage], sizeof(k_tile));
                warp::tma::load_async(k_smem[stage], g.k, at, k_smem_arrived[stage]);
                warp::tma::expect_bytes(v_smem_arrived[stage], sizeof(v_tile));
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

        for (int j = 0; j < kv_iters; j++) {
            const int stage = j % STAGES, phase = (j / STAGES) % 2;
            wait(k_smem_arrived[stage], phase);
            warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[stage]);

            // The running max before this tile, unscaled (the rescale below reads it).
            warp::copy(max_vec_last, max_vec);

            // The WGMMA writes `att_block` asynchronously: wait before reading it.
            warpgroup::mma_async_wait();

            // Keys k0 .. k0 + 64 of tile j. Causal: keep key ≤ P + row (`tril` keeps col ≤ row +
            // diagonal, with the tile's row and col from 0). Length: drop key ≥ kv_len.
            const int k0 = j * KV_HEIGHT;
            warp::tril(att_block, att_block, prefix_len + row0 - k0, NEG_INF);
            warp::right_fill(att_block, att_block, kv_len - k0, NEG_INF);

            warp::row_max(max_vec, att_block, max_vec);
            warp::mul(att_block, att_block, g.scale_log2);
            warp::mul(max_vec_scaled, max_vec, g.scale_log2);
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
            // A partial last tile: zero the V rows of the slots at or past kv_len, which no append
            // wrote, before either warpgroup's P·V reads them (0 · NaN = NaN). `valid` is the
            // same for the whole CTA, so every consumer thread takes this branch or none does.
            const int valid = kv_len - k0;
            if (valid < KV_HEIGHT) {
                const int t = warpgroupid * kittens::WARPGROUP_THREADS + warpgroup::laneid();
                for (int e = t; e < (KV_HEIGHT - valid) * HEAD_DIM; e += CONSUMER_THREADS) {
                    v_smem[stage][{valid + e / HEAD_DIM, e % HEAD_DIM}] =
                        kittens::base_types::constants<bf16>::zero();
                }
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                consumers::sync(CONSUMERS_BARRIER);
            }
            warpgroup::mma_AB(o_reg, att_block_mma, v_smem[stage]);
            // Wait for the WGMMA to finish reading the stage before releasing it to the producer.
            warpgroup::mma_async_wait();

            if (warpgroup::laneid() == 0) {
                arrive(compute_done[stage], 1);
            }
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

// The kernel's one-time setup: allow it `ATTN_SMEM_BYTES` of dynamic shared memory on the current
// device (the default limit is 48 KB). Run once per device before any launch
// (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch.
//
// 0 = done; any other value = the cudaError_t of setting the attribute.
extern "C" int scratchy_cuda_attn_prefill_bf16_prepare(void) {
    return static_cast<int>(cudaFuncSetAttribute(
        attn_prefill_bf16_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, ATTN_SMEM_BYTES));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value
// = the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// Refused before any pointer is touched: `head_dim` not 256; `rows` zero or not a multiple of 128;
// `q_heads` or `kv_heads` zero, or `q_heads` not a multiple of `kv_heads`; `block_size` not 64;
// `num_blocks` zero; `scale` not finite or not above 0; a dimension above INT_MAX.
//
// `q` and `out` must be 16-byte-aligned device pointers to `rows · q_heads · 256` bf16, `out`
// overlapping no input; `k_cache` and `v_cache` to one layer's `num_blocks · kv_heads · 64 · 256`
// bf16; `block_table` to the sequence's block ids, `ceil((P + L) / 64)` of them each below
// `num_blocks`; `sequence` to `[P, L]` u32 with `1 ≤ L ≤ rows` and `P + L ≤ 64 · (entries)`.
// `scratchy_cuda_attn_prefill_bf16_prepare` must have succeeded on the current device. The launch
// is asynchronous on `stream`; the caller owns it and synchronizes.
extern "C" int scratchy_cuda_attn_prefill_bf16(const void* q, const void* k_cache,
                                               const void* v_cache, const void* block_table,
                                               const void* sequence, void* out, uint32_t rows,
                                               uint32_t q_heads, uint32_t kv_heads,
                                               uint32_t head_dim, uint32_t block_size,
                                               uint32_t num_blocks, float scale,
                                               void* stream /* cudaStream_t; null = default */) {
    if (head_dim != uint32_t(HEAD_DIM) || rows == 0 || rows % uint32_t(CTA_ROWS) != 0 ||
        q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 || block_size != BLOCK_SIZE ||
        num_blocks == 0 || !std::isfinite(scale) || !(scale > 0.0f) ||
        rows > uint32_t(INT_MAX) || q_heads > uint32_t(INT_MAX) || num_blocks > uint32_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        bf16* qp = static_cast<bf16*>(const_cast<void*>(q));
        bf16* op = static_cast<bf16*>(out);
        bf16* kp = static_cast<bf16*>(const_cast<void*>(k_cache));
        bf16* vp = static_cast<bf16*>(const_cast<void*>(v_cache));
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
        attn_globals G{qg, kg, vg, og, tg, sg, scale * 1.44269504089f, int(q_heads / kv_heads)};
        void* args[] = {&G};
        const dim3 grid(rows / uint32_t(CTA_ROWS), q_heads, 1), block(NUM_THREADS);
        return static_cast<int>(cudaLaunchKernel(attn_prefill_bf16_ker, grid, block, args,
                                                 ATTN_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
