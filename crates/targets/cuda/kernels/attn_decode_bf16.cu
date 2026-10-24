// SPDX-License-Identifier: Apache-2.0
//
// Grouped-query decode attention of one query row over the paged bf16 KV cache, for the H100 at
// head_dim 256: the one-row bucket's attention (a decode step's one new token). Q and O are the
// tape's `[1, q_heads · 256]` bf16; K and V are read from one layer's paged cache
// `[num_blocks, kv_heads, 64, 256]` bf16 through the sequence's block table.
//
//   o[h, :] = Σ_j softmax_j(scale · q[h, :] · k[j, h/hr, :]) · v[j, h/hr, :],   j < kv_len
//
// with `kv_len = P + L` read from the tape's `sequence` region `[P, L]` (L = 1 for a decode, the
// new token's key already appended by its layer's `RopeAppend`), and `hr = q_heads / kv_heads` the
// GQA group: query head h reads kv head h / hr.
//
// ThunderKittens' paged-KV tile attention (PLAN-tk100 §2 S5, §3.11): the machinery of
// `attn_prefill_bf16.cu` (adapted from TK's `mha_h100.cu` at 67845f5f, MIT), with the GQA group
// stacked as the rows of one query tile. What differs from the prefill, and why:
//   - Grid `(kv_heads)`: one CTA per kv head g. Its `hr` query heads `g·hr .. g·hr + hr` are
//     contiguous in the decode row, so with q viewed as `[q_heads, 256]` they are rows `g·hr ..
//     g·hr + hr` of it. The producer TMA-loads ONE swizzled `st_bf<64, 256>` box there, at an
//     element-unit row coordinate (`coord<>`, whose `unit_coord` is itself: a tile-unit `coord<st>`
//     would scale the row by 64). Rows `hr .. 63` of the box are the next kv heads' queries, or
//     zeros past `q_heads` (a swizzled box past the array's rows reads zeros, TK-1's R1 gate,
//     `tests/tma_bounds.rs`); their outputs are computed and discarded. Both consumer warpgroups
//     read the same Q tile, and warp 0 of each holds rows 0..15, so `hr ≤ 16`
//     (`step::ATTN_DECODE_BF16_MAX_GROUP`).
//   - KV tile j (one 64-slot block, `{block_table[j], g, 0, 0}`, the prefill's box) goes to
//     consumer warpgroup `j % 2`, which owns a 1-stage K and V buffer: the two warpgroups work on
//     alternate tiles at once. The producer loads tile j after warpgroup `j % 2` released tile
//     `j − 2` (`compute_done[j % 2]`, one arrival).
//   - No causal mask: the one query is at position `kv_len − 1`, so every key below kv_len is
//     allowed. `warp::right_fill` overwrites the scores of slots at or past kv_len with −inf
//     before `row_max` (they may be NaN: the pool is never cleared). A partial last tile's V rows
//     `[kv_len − k0, 64)` are zeroed in shared memory before P·V (0 · NaN = NaN, #791), fenced to
//     the async proxy the tensor cores read through, and met at the warpgroup's own named barrier:
//     only the warpgroup that owns the tile reads it.
//   - The online softmax is the prefill's, line for line, and so is where the max key's `p` is
//     exactly 1: the score is scaled by its own `warp::mul` and the scaled max subtracted by its
//     own `warp::sub_row`, so for the max key both operands of the subtraction are the same
//     product of the same two floats, the difference is 0 and `exp2(0) = 1` (the plain-CUDA
//     kernel rounded `dot · c` with `__fmul_rn` for the same reason). The rescale goes through
//     `rescale_exact1` (exactly 1 for an unchanged max), and `p` is rounded to bf16 once before
//     the `mma_AB`, as the prefill rounds it: the one numerical change from the plain-CUDA
//     kernel, which kept `p` in f32. Each warpgroup's `(m, l, o)` covers its own tiles.
//   - The merge: after both warpgroups' last tile (consumer barrier), warp 0 of each writes its
//     16 rows' `m` and `l` to `sv_fl<16>` and its unnormalised `o` to an `st_fl<16, 256>` over the
//     Q tile, which no one reads any more (the prefill's O store aliases its own Q tile
//     the same way). After a second barrier, warp 0 of warpgroup 0 merges in warpgroup order 0,
//     then 1: `M = max(m₀, m₁)`, `r_w = rescale_exact1(m_w, M)`, `L = l₀·r₀ + l₁·r₁`, `O = o₀·r₀ +
//     o₁·r₁` (TK's shared-tile `mul_row` and `add`), then `div_row(O, O, L)`.
//   - A warpgroup that saw no key (warpgroup 1 when kv_len ≤ 64) has `m = −inf`, `l = 0` and
//     `o = 0`. Warpgroup 0 always saw key 0 (`kv_len ≥ 1`: `forward` checks `L ≥ 1`), so `M` is
//     finite and `r₁ = exp2((−inf − M)·c) = exp2(−inf) = +0` exactly: its side adds `0·0 = +0` to
//     `L` and to `O`, exactly nothing, and `r₀ = 1` exactly. No `exp2(−inf − −inf)` is ever
//     formed, so no select on `l` is needed.
//   - The store: `O` into a plain `st_bf<16, 256>` (`warp::store`, one round-to-nearest-even),
//     then rows `0 .. hr` bulk-stored as the `hr · 512` contiguous bytes of kv head g's query
//     heads in `out` (a plain tile's rows are contiguous). No tensor map is built over `out`, and
//     rows `hr .. 15` are never written.
//
// The order of every addition is a function of kv_len alone: tile j always goes to warpgroup
// `j % 2`, each warpgroup takes its tiles in ascending j, and the merge is in warpgroup order. It
// does not depend on which physical blocks the table names, so a shuffled block table gives the
// same bits, and so does a repeat. No atomics.
//
// What changes in the guarantee: the plain-CUDA kernel never loaded a slot at or past kv_len.
// This one loads whole 64-slot blocks and neutralises the unwritten part before use, scores by
// `right_fill` and V rows by zeroing: the prefill's discipline. A block id at or past
// `num_blocks` (which `forward` refuses) reads a zero-filled box, never memory past the pool.
//
// Cost: only `hr` of the 64 tensor-core rows are useful (4 of 64 for qwen3.5). Decode is bound by
// reading K and V, so the wasted rows cost issue slots, not bandwidth. One CTA per kv head; a
// cross-CTA split over keys ("flash-decoding") stays the deferred speed issue.
//
// Shared memory: the Q tile (32 KiB, its first 32 KiB reused for the two `st_fl<16, 256>` merge
// tiles), K and V per warpgroup (128 KiB), the bf16 output tile (8 KiB), plus 1,024 bytes of
// alignment slack = 173,056 bytes of dynamic shared memory, set once by
// `scratchy_cuda_attn_decode_bf16_prepare` (in `PREPARE`, as the prefill's). The semaphores and the
// merge's six `sv_fl<16>` are static.

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
// The named barrier over the two consumer warpgroups (ids 4 and 5 are each warpgroup's own, at a
// partial V tile; 0 is `__syncthreads`).
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
using k_tile = st_bf<KV_HEIGHT, HEAD_DIM>;
using v_tile = st_bf<KV_HEIGHT, HEAD_DIM>;
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
using kv_gl = gl<bf16, -1, -1, KV_HEIGHT, HEAD_DIM, k_tile>;
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
};

constexpr size_t SMEM_TILES = sizeof(q_tile) + CONSUMER_WARPGROUPS * sizeof(k_tile) +
                              CONSUMER_WARPGROUPS * sizeof(v_tile) + sizeof(out_tile);
static_assert(SMEM_TILES == 172032, "q + 2 K + 2 V tiles of [64, 256] bf16, and [16, 256] bf16");
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int DECODE_SMEM_BYTES = int(SMEM_TILES) + 1024;
static_assert(DECODE_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

__global__ __launch_bounds__(NUM_THREADS, 1) void attn_decode_bf16_ker(
    const __grid_constant__ decode_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    const int warpid = kittens::warpid(), warpgroupid = warpid / kittens::WARPGROUP_WARPS;

    q_tile(&q_smem)[1] = al.allocate<q_tile, 1>();
    k_tile(&k_smem)[CONSUMER_WARPGROUPS] = al.allocate<k_tile, CONSUMER_WARPGROUPS>();
    v_tile(&v_smem)[CONSUMER_WARPGROUPS] = al.allocate<v_tile, CONSUMER_WARPGROUPS>();
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
        // The producer: one warp loads KV tile `j` into warpgroup `j % 2`'s buffer, after that
        // warpgroup released its previous tile `j - 2`.
        warpgroup::decrease_registers<24>();
        if (warpid == NUM_WARPS - 4) {
            for (int j = 0; j < kv_iters; j++) {
                const int w = j % CONSUMER_WARPGROUPS;
                if (j >= CONSUMER_WARPGROUPS) {
                    wait(compute_done[w], ((j - CONSUMER_WARPGROUPS) / CONSUMER_WARPGROUPS) % 2);
                }
                const int block = g.block_table[{0, 0, 0, j}];
                coord<k_tile> at = {block, kv_head, 0, 0};
                warp::tma::expect_bytes(k_smem_arrived[w], sizeof(k_tile));
                warp::tma::load_async(k_smem[w], g.k, at, k_smem_arrived[w]);
                warp::tma::expect_bytes(v_smem_arrived[w], sizeof(v_tile));
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
        for (int j = w; j < kv_iters; j += CONSUMER_WARPGROUPS) {
            const int phase = (j / CONSUMER_WARPGROUPS) % 2;
            wait(k_smem_arrived[w], phase);
            warpgroup::mm_ABt(att_block, q_smem[0], k_smem[w]);

            // The running max before this tile, unscaled (the rescale below reads it).
            warp::copy(max_vec_last, max_vec);

            // The WGMMA writes `att_block` asynchronously: wait before reading it.
            warpgroup::mma_async_wait();

            // Keys k0 .. k0 + 64 of tile j: drop key ≥ kv_len. No causal mask, as the one query
            // is the last key's position.
            const int k0 = j * KV_HEIGHT;
            warp::right_fill(att_block, att_block, kv_len - k0, NEG_INF);

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
            // A partial last tile: zero the V rows of the slots at or past kv_len, which no append
            // wrote, before this warpgroup's P·V reads them (0 · NaN = NaN). `valid` is the same
            // for the whole warpgroup, the only one that reads this tile, so every one of its
            // threads takes this branch or none does.
            const int valid = kv_len - k0;
            if (valid < KV_HEIGHT) {
                for (int e = warpgroup::laneid(); e < (KV_HEIGHT - valid) * HEAD_DIM;
                     e += kittens::WARPGROUP_THREADS) {
                    v_smem[w][{valid + e / HEAD_DIM, e % HEAD_DIM}] =
                        kittens::base_types::constants<bf16>::zero();
                }
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                warpgroup::sync(w + 4);
            }
            warpgroup::mma_AB(o_reg, att_block_mma, v_smem[w]);
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

// The kernel's one-time setup: allow it `DECODE_SMEM_BYTES` of dynamic shared memory on the
// current device (the default limit is 48 KB). Run once per device before any launch
// (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch.
//
// 0 = done; any other value = the cudaError_t of setting the attribute.
extern "C" int scratchy_cuda_attn_decode_bf16_prepare(void) {
    return static_cast<int>(cudaFuncSetAttribute(
        attn_decode_bf16_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, DECODE_SMEM_BYTES));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value =
// the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// Refused before any pointer is touched: `head_dim` not 256; `q_heads` or `kv_heads` zero, or
// `q_heads` not a multiple of `kv_heads`; `q_heads / kv_heads` above 16
// (`step::ATTN_DECODE_BF16_MAX_GROUP`); `block_size` not 64; `num_blocks` zero; `scale` not finite
// or not above 0; a dimension above INT_MAX.
//
// `q` and `out` must be 16-byte-aligned device pointers to `q_heads · 256` bf16 (one row), `out`
// overlapping no input; `k_cache` and `v_cache` to one layer's `num_blocks · kv_heads · 64 · 256`
// bf16; `sequence` to `[P, L]` u32 with `L ≥ 1`; `block_table` to the sequence's block ids,
// `ceil((P + L) / 64)` of them, each below `num_blocks`.
// `scratchy_cuda_attn_decode_bf16_prepare` must have succeeded on the current device. The launch
// is asynchronous on `stream`; the caller owns it and synchronizes.
extern "C" int scratchy_cuda_attn_decode_bf16(const void* q, const void* k_cache,
                                              const void* v_cache, const void* block_table,
                                              const void* sequence, void* out, uint32_t q_heads,
                                              uint32_t kv_heads, uint32_t head_dim,
                                              uint32_t block_size, uint32_t num_blocks,
                                              float scale,
                                              void* stream /* cudaStream_t; null = default */) {
    if (head_dim != uint32_t(HEAD_DIM) || q_heads == 0 || kv_heads == 0 ||
        q_heads % kv_heads != 0 || q_heads / kv_heads > uint32_t(MAX_GROUP) ||
        block_size != BLOCK_SIZE || num_blocks == 0 || !std::isfinite(scale) || !(scale > 0.0f) ||
        q_heads > uint32_t(INT_MAX) || num_blocks > uint32_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        bf16* qp = static_cast<bf16*>(const_cast<void*>(q));
        bf16* op = static_cast<bf16*>(out);
        bf16* kp = static_cast<bf16*>(const_cast<void*>(k_cache));
        bf16* vp = static_cast<bf16*>(const_cast<void*>(v_cache));
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
        decode_globals G{qg, kg, vg, og, tg, sg, scale * LOG2E, int(q_heads / kv_heads)};
        void* args[] = {&G};
        const dim3 grid(kv_heads), block(NUM_THREADS);
        return static_cast<int>(cudaLaunchKernel(attn_decode_bf16_ker, grid, block, args,
                                                 DECODE_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
