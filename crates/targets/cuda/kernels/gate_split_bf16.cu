// SPDX-License-Identifier: Apache-2.0
//
// bf16 attention output-gate split for the H100 (qwen3.5's `attn_output_gate`): q_proj is twice as
// wide as the query, each head's 2·head_dim outputs laid out [query(head_dim) | gate(head_dim)].
// For head h and lane d, over qg [rows, 2·half_cols] and q, gate [rows, half_cols]:
//
//     q[r, h·hd + d]    = qg[r, 2h·hd + d]
//     gate[r, h·hd + d] = qg[r, 2h·hd + hd + d]
//
// Pure copies, so bit-exact. Metal's counterpart is `shaders/gate_split.metal`; transformers'
// `Qwen3_5Attention` is `torch.chunk(q_proj(x).view(..., head_dim*2), 2, dim=-1)`.
//
// ThunderKittens' 16-row tile, copy only (PLAN-tk100 §2 S2): one CTA of one warp per block of 16
// rows. For each head h, lane 0 TMA-loads the plain shared tiles `st_bf<16, 256, false>` at qg's
// column tiles 2h (the query) and 2h + 1 (the gate), `tma::expect_bytes` on the stage's
// semaphore, and TMA-stores them to q and gate at column tile h. The bytes never leave shared
// memory as anything else, and never pass through registers. Every tile is plain, never
// swizzled: a TMA box past the arrays' rows (the last block's padding) reads zeros and stores
// nothing (TK-1's R1 gate, `tests/tma_bounds.rs`), so the outputs' tails are never written. Each
// tile is one whole 256-wide head, so no column tile is partial. A 2-stage ring: the next head's
// loads are in flight while this head's stores drain.
//
// Shared memory: 2 stages × 2 tiles × 8 KiB, plus 1,024 bytes of alignment slack = 33,792 bytes of
// dynamic shared memory, under the 48 KiB default, so no one-time setup.

#include "kittens.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// The launcher's return codes that are not a `cudaError_t` (which is non-negative).
// `src/kernels.rs` mirrors them.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::GATE_SPLIT_BF16_HEAD_DIM`: the one head dim the kernel is instantiated for, one tile wide.
constexpr int HEAD_DIM = 256;
constexpr int ROWS = 16;
constexpr int STAGES = 2;
constexpr int NUM_THREADS = kittens::WARP_THREADS;

using tile = st_bf<ROWS, HEAD_DIM, false>;
using row_gl = gl<bf16, 1, 1, -1, -1, tile>;

struct split_globals {
    row_gl qg;
    row_gl q;
    row_gl gate;
    int heads;
};

constexpr size_t SMEM_TILES = STAGES * 2 * sizeof(tile);
static_assert(SMEM_TILES == 32768, "2 stages x 2 tiles x 8 KiB");
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int SPLIT_SMEM_BYTES = int(SMEM_TILES) + 1024;
static_assert(SPLIT_SMEM_BYTES <= 48 * 1024, "under the default dynamic shared memory limit");

__global__ __launch_bounds__(NUM_THREADS, 1) void gate_split_bf16_ker(const __grid_constant__ split_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    tile(&q_smem)[STAGES] = al.allocate<tile, STAGES>();
    tile(&g_smem)[STAGES] = al.allocate<tile, STAGES>();
    __shared__ kittens::semaphore arrived[STAGES];

    const int row_block = blockIdx.x;
    if (warp::laneid() == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_semaphore(arrived[s], 0, 1);
        }
    }
    __syncwarp();

    // Head `h`'s query and gate tiles into stage `h % 2`.
    auto issue = [&](int h) {
        const int stage = h % STAGES;
        warp::tma::expect_bytes(arrived[stage], 2 * sizeof(tile));
        warp::tma::load_async(q_smem[stage], g.qg, coord<tile>{0, 0, row_block, 2 * h},
                              arrived[stage]);
        warp::tma::load_async(g_smem[stage], g.qg, coord<tile>{0, 0, row_block, 2 * h + 1},
                              arrived[stage]);
    };
    issue(0);
    for (int h = 0; h < g.heads; h++) {
        const int stage = h % STAGES;
        // The other stage's previous stores must have read its tiles before the next loads
        // overwrite them.
        warp::tma::store_async_read_wait();
        __syncwarp();
        if (h + 1 < g.heads) {
            issue(h + 1);
        }
        wait(arrived[stage], (h / STAGES) % 2);
        coord<tile> at = {0, 0, row_block, h};
        warp::tma::store_async(g.q, q_smem[stage], at);
        warp::tma::store_async(g.gate, g_smem[stage], at);
    }
    warp::tma::store_async_wait();
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value =
// the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// Refused before any pointer is touched: `rows` or `half_cols` zero; `head_dim` not 256
// (`step::GATE_SPLIT_BF16_HEAD_DIM`, the one head dim the kernel is instantiated for);
// `half_cols` not a multiple of `head_dim`; `rows · 2·half_cols` above INT_MAX. `qg`, `q` and
// `gate` must be 16-byte-aligned device pointers to `rows·2·half_cols`, `rows·half_cols` and
// `rows·half_cols` bf16, no two overlapping. Its dynamic shared memory is under the 48 KiB
// default, so no one-time setup. The launch is asynchronous on `stream`; the caller owns it and
// synchronizes.
extern "C" int scratchy_cuda_gate_split_bf16(const void* qg, void* q, void* gate, uint32_t rows,
                                             uint32_t half_cols, uint32_t head_dim,
                                             void* stream /* cudaStream_t; null = default */) {
    if (rows == 0 || half_cols == 0 || head_dim != uint32_t(HEAD_DIM) ||
        half_cols % head_dim != 0 ||
        uint64_t(rows) * 2 * uint64_t(half_cols) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        bf16* qgp = static_cast<bf16*>(const_cast<void*>(qg));
        bf16* qp = static_cast<bf16*>(q);
        bf16* gp = static_cast<bf16*>(gate);
        // Building each `gl` encodes its tensor map into host memory; nothing is allocated on
        // the device. Every row stride is a multiple of 512 bytes.
        row_gl qgg{qgp, nullptr, nullptr, size_t(rows), size_t(2) * half_cols};
        row_gl qgl{qp, nullptr, nullptr, size_t(rows), size_t(half_cols)};
        row_gl ggl{gp, nullptr, nullptr, size_t(rows), size_t(half_cols)};
        split_globals G{qgg, qgl, ggl, int(half_cols / head_dim)};
        void* args[] = {&G};
        const dim3 grid((rows + uint32_t(ROWS) - 1) / uint32_t(ROWS)), block(NUM_THREADS);
        return static_cast<int>(cudaLaunchKernel(gate_split_bf16_ker, grid, block, args,
                                                 SPLIT_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
