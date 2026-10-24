// SPDX-License-Identifier: Apache-2.0
//
// Test support only, for `tests/tma_bounds.rs`: no `CudaStep` and no lowering use it. It records
// what ThunderKittens' TMA tile load and store do with a 64×64 bf16 box that runs past the edge of
// its array, for a swizzled `st_bf<64, 64>` and a plain `st_bf<64, 64, false>`: the facts the
// GEMV's and the narrow GEMM's choice of tiles rests on (`gemv_bf16.cu`). They are the GKE probe's
// R1 cases (c8d80fda), re-run here so a change to TK or to the nvcc flags re-checks them.
//
// - load: TMA-load a 64×64 all-NaN `poison` array into the tile and wait, so every element the
//   next load does not write keeps a NaN no generic-proxy write put there; TMA-load the box at
//   tile coordinates `(tile_row, tile_col)` of `src` into it; then TMA-store the tile into the
//   64×64 `dump` array, an exact fit.
// - store: TMA-load a 64×64 `pattern` array (an exact fit) into the tile, then TMA-store it at
//   tile coordinates `(tile_row, tile_col)` of `dst`.
// One warp; its lane 0 issues every copy. One 8 KiB tile plus 1,024 bytes of alignment slack of
// dynamic shared memory, under the default 48 KB limit, so no one-time setup.

#include "kittens.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

constexpr int BOX = 64;
// The arrays' columns are a multiple of this, so every row stride is a multiple of 16 bytes.
constexpr uint32_t VEC = 8;

using swizzled_tile = st_bf<BOX, BOX>;
using plain_tile = st_bf<BOX, BOX, false>;

// A `[rows, cols]` bf16 array with the tensor map of tile type `ST`.
template <typename ST>
using box_gl = gl<bf16, 1, 1, -1, -1, ST>;

template <typename ST>
struct load_globals {
    box_gl<ST> src;
    box_gl<ST> poison;
    box_gl<ST> dump;
    int tile_row;
    int tile_col;
};

template <typename ST>
struct store_globals {
    box_gl<ST> pattern;
    box_gl<ST> dst;
    int tile_row;
    int tile_col;
};

constexpr int PROBE_SMEM_BYTES = int(sizeof(swizzled_tile)) + 1024;
static_assert(sizeof(swizzled_tile) == sizeof(plain_tile), "one tile either way");

template <typename ST>
__global__ void tma_box_probe_load_ker(const __grid_constant__ load_globals<ST> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    ST& t = al.template allocate<ST>();
    __shared__ kittens::semaphore poisoned, loaded;
    if (warp::laneid() == 0) {
        init_semaphore(poisoned, 0, 1);
        init_semaphore(loaded, 0, 1);
    }
    __syncwarp();
    warp::tma::expect_bytes(poisoned, sizeof(ST));
    warp::tma::load_async(t, g.poison, coord<ST>{0, 0, 0, 0}, poisoned);
    wait(poisoned, 0);
    warp::tma::expect_bytes(loaded, sizeof(ST));
    warp::tma::load_async(t, g.src, coord<ST>{0, 0, g.tile_row, g.tile_col}, loaded);
    wait(loaded, 0);
    warp::tma::store_async(g.dump, t, coord<ST>{0, 0, 0, 0});
    warp::tma::store_async_wait();
}

template <typename ST>
__global__ void tma_box_probe_store_ker(const __grid_constant__ store_globals<ST> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    ST& t = al.template allocate<ST>();
    __shared__ kittens::semaphore loaded;
    if (warp::laneid() == 0) {
        init_semaphore(loaded, 0, 1);
    }
    __syncwarp();
    warp::tma::expect_bytes(loaded, sizeof(ST));
    warp::tma::load_async(t, g.pattern, coord<ST>{0, 0, 0, 0}, loaded);
    wait(loaded, 0);
    warp::tma::store_async(g.dst, t, coord<ST>{0, 0, g.tile_row, g.tile_col});
    warp::tma::store_async_wait();
}

template <typename ST>
cudaError_t launch_load(bf16* src, bf16* poison, bf16* dump, uint32_t rows, uint32_t cols,
                        uint32_t tile_row, uint32_t tile_col, cudaStream_t stream) {
    box_gl<ST> sg{src, nullptr, nullptr, size_t(rows), size_t(cols)};
    box_gl<ST> pg{poison, nullptr, nullptr, size_t(BOX), size_t(BOX)};
    box_gl<ST> dg{dump, nullptr, nullptr, size_t(BOX), size_t(BOX)};
    load_globals<ST> G{sg, pg, dg, int(tile_row), int(tile_col)};
    void* args[] = {&G};
    return cudaLaunchKernel(tma_box_probe_load_ker<ST>, dim3(1), dim3(kittens::WARP_THREADS), args,
                            PROBE_SMEM_BYTES, stream);
}

template <typename ST>
cudaError_t launch_store(bf16* pattern, bf16* dst, uint32_t rows, uint32_t cols,
                         uint32_t tile_row, uint32_t tile_col, cudaStream_t stream) {
    box_gl<ST> pg{pattern, nullptr, nullptr, size_t(BOX), size_t(BOX)};
    box_gl<ST> dg{dst, nullptr, nullptr, size_t(rows), size_t(cols)};
    store_globals<ST> G{pg, dg, int(tile_row), int(tile_col)};
    void* args[] = {&G};
    return cudaLaunchKernel(tma_box_probe_store_ker<ST>, dim3(1), dim3(kittens::WARP_THREADS),
                            args, PROBE_SMEM_BYTES, stream);
}

// Refused: `rows` or `cols` zero, `cols` not a multiple of 8, a dimension or tile coordinate at
// or above INT_MAX / 64.
bool refused(uint32_t rows, uint32_t cols, uint32_t tile_row, uint32_t tile_col) {
    constexpr uint32_t LIMIT = uint32_t(INT_MAX) / uint32_t(BOX);
    return rows == 0 || cols == 0 || cols % VEC != 0 || rows >= LIMIT || cols >= LIMIT ||
           tile_row >= LIMIT || tile_col >= LIMIT;
}

}  // namespace

// The load case: `src` `[rows, cols]` bf16, `poison` and `dump` `[64, 64]` bf16, all 16-byte
// aligned device pointers; `swizzled` non-zero for the swizzled tile. 0 = launched;
// SCRATCHY_CUDA_ERR_SHAPE = refused, nothing launched; SCRATCHY_CUDA_ERR_HOST = a C++ exception
// was caught (e.g. tensor-map encode); any other value = the launch's cudaError_t. Asynchronous on
// `stream`.
extern "C" int scratchy_cuda_tma_box_probe_load(const void* src, const void* poison, void* dump,
                                                uint32_t rows, uint32_t cols, uint32_t tile_row,
                                                uint32_t tile_col, uint32_t swizzled,
                                                void* stream /* cudaStream_t; null = default */) {
    if (refused(rows, cols, tile_row, tile_col)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        bf16* sp = static_cast<bf16*>(const_cast<void*>(src));
        bf16* pp = static_cast<bf16*>(const_cast<void*>(poison));
        bf16* dp = static_cast<bf16*>(dump);
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        const cudaError_t e =
            swizzled != 0
                ? launch_load<swizzled_tile>(sp, pp, dp, rows, cols, tile_row, tile_col, s)
                : launch_load<plain_tile>(sp, pp, dp, rows, cols, tile_row, tile_col, s);
        return static_cast<int>(e);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// The store case: `pattern` `[64, 64]` bf16, `dst` `[rows, cols]` bf16, both 16-byte aligned
// device pointers; `swizzled` non-zero for the swizzled tile. The return codes are the load
// case's. Asynchronous on `stream`.
extern "C" int scratchy_cuda_tma_box_probe_store(const void* pattern, void* dst, uint32_t rows,
                                                 uint32_t cols, uint32_t tile_row,
                                                 uint32_t tile_col, uint32_t swizzled,
                                                 void* stream /* cudaStream_t; null = default */) {
    if (refused(rows, cols, tile_row, tile_col)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        bf16* pp = static_cast<bf16*>(const_cast<void*>(pattern));
        bf16* dp = static_cast<bf16*>(dst);
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        const cudaError_t e =
            swizzled != 0 ? launch_store<swizzled_tile>(pp, dp, rows, cols, tile_row, tile_col, s)
                          : launch_store<plain_tile>(pp, dp, rows, cols, tile_row, tile_col, s);
        return static_cast<int>(e);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
