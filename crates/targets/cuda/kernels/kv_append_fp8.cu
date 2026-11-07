// SPDX-License-Identifier: Apache-2.0
//
// The paged KV cache's write into an e4m3 cache for the H100 (FP8 S5): for each token row t of the
// forward whose slot `slot_mapping[t]` is not u32::MAX, row t of K and of V ([rows, kv_heads ·
// head_dim] bf16, row-major) is written into one layer's K and V cache at that slot, each element
// as `e4m3_satfinite(x)`: round to nearest even, clamped to ±448, NaN to 0x7F. The scale is a fixed
// per-tensor 1.0, vLLM's `--kv-cache-dtype fp8` for a checkpoint without KV scales
// (`scaled_vec_conversion<uint8_t, __nv_bfloat16>` at scale 1.0). Every e4m3 value is a bf16 value,
// so the attention kernels' widen on read (`attn_*_fp8kv.cu`) is exact.
//
// `kv_append_bf16.cu` with one conversion added, which stays as the test-only reference of the
// bf16 cache. The cache is [num_blocks, kv_heads, block_size, head_dim] e4m3 (`PagedKvLayout`, in
// `scratchy-subtile`), one byte an element; a padding row's slot (u32::MAX) and a block at or past
// `num_blocks` write nothing, as there. What differs, and why:
//   - After the bulk load, the warp converts each `sv_bf<head_dim>` into an `sv_fp8e4m3<head_dim>`
//     with TK's converting shared-vector copy (`warp::copy`, `ops/group/shared/vec/conversions.cuh`):
//     each lane converts `head_dim / 32` elements through `base_types::convertor<fp8e4m3, bf16>`,
//     cuda_fp8.hpp's `__nv_fp8_e4m3(__nv_bfloat16)`, the satfinite e4m3 rounding. bf16 to f32 is
//     exact, so this is vLLM's bf16 → float → satfinite conversion bit for bit.
//   - A `__syncwarp`, then the vector TMA store of the e4m3 vectors (`quant_rows_fp8.cu`'s store
//     order; TK's store fences the async proxy itself). An e4m3 vector of 256 is one 256-byte box,
//     of 512 two (`sv_tma_dim1` is 256 for both), each row 256-byte aligned.
//
// Static shared memory: 4 warps × (2 bf16 vectors + 2 e4m3 vectors) × head_dim, plus a semaphore
// each (12 KiB at 512), so its one-time setup (`_prepare`, in `PREPARE`) only loads it.

#include "kittens.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// The launcher's return codes that are not a `cudaError_t` (which is non-negative).
// `src/kernels.rs` mirrors them.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::KV_APPEND_BF16_HEAD_DIMS`: the head dims the kernel is instantiated for.
constexpr int HEAD_DIM_SLIDING = 256;
constexpr int HEAD_DIM_GLOBAL = 512;
// Token slots per block, the attention kernel's K/V tile height (`step::CUDA_KV_BLOCK_SIZE`).
constexpr uint32_t BLOCK_SIZE = 64;
// A padding row's slot.
constexpr uint32_t NO_SLOT = 0xFFFFFFFFu;
// The most blocks a u32 slot can address: a larger `num_blocks` changes no write.
constexpr uint32_t MAX_SLOT_BLOCKS = (NO_SLOT / BLOCK_SIZE) + 1;
constexpr int WARPS = 4;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;

using row_gl = gl<bf16, 1, 1, -1, -1>;
using slot_gl = gl<int, 1, 1, 1, -1>;
// A row-head as loaded, and as cached.
template <int HEAD_DIM>
using head_vec = sv_bf<HEAD_DIM>;
template <int HEAD_DIM>
using cache_vec = sv_fp8e4m3<HEAD_DIM>;
template <int HEAD_DIM>
using cache_gl = gl<fp8e4m3, -1, -1, int(BLOCK_SIZE), HEAD_DIM, cache_vec<HEAD_DIM>>;

template <int HEAD_DIM>
struct kv_globals {
    row_gl k;
    row_gl v;
    slot_gl slot_mapping;
    cache_gl<HEAD_DIM> k_cache;
    cache_gl<HEAD_DIM> v_cache;
    // `rows · kv_heads`, one warp per pair.
    int pairs;
    int kv_heads;
    uint32_t num_blocks;
};

template <int HEAD_DIM>
__global__ __launch_bounds__(NUM_THREADS, 1) void kv_append_fp8_ker(
    const __grid_constant__ kv_globals<HEAD_DIM> g) {
    __shared__ sv_bf<HEAD_DIM> k_buf[WARPS];
    __shared__ sv_bf<HEAD_DIM> v_buf[WARPS];
    __shared__ sv_fp8e4m3<HEAD_DIM> k_q[WARPS];
    __shared__ sv_fp8e4m3<HEAD_DIM> v_q[WARPS];
    __shared__ kittens::semaphore arrived[WARPS];

    const int w = kittens::warpid();
    const int pair = blockIdx.x * WARPS + w;
    if (pair >= g.pairs) {
        return;
    }
    const int t = pair / g.kv_heads, h = pair % g.kv_heads;
    const uint32_t slot = uint32_t(g.slot_mapping[{0, 0, 0, t}]);
    if (slot == NO_SLOT) {
        return;
    }
    const uint32_t block = slot / BLOCK_SIZE, off = slot % BLOCK_SIZE;
    if (block >= g.num_blocks) {
        return;
    }
    if (warp::laneid() == 0) {
        init_semaphore(arrived[w], 0, 1);
    }
    __syncwarp();
    constexpr uint32_t bytes = HEAD_DIM * sizeof(bf16);
    warp::tma::expect_bytes(arrived[w], 2 * bytes);
    warp::tma::load_async(k_buf[w][0], g.k[{0, 0, t, h * HEAD_DIM}], bytes, arrived[w]);
    warp::tma::load_async(v_buf[w][0], g.v[{0, 0, t, h * HEAD_DIM}], bytes, arrived[w]);
    wait(arrived[w], 0);
    // Each element to e4m3, satfinite; then every lane's writes before lane 0's stores read them.
    warp::copy(k_q[w], k_buf[w]);
    warp::copy(v_q[w], v_buf[w]);
    __syncwarp();
    const coord<cache_vec<HEAD_DIM>> at = {int(block), h, int(off), 0};
    warp::tma::store_async(g.k_cache, k_q[w], at);
    warp::tma::store_async(g.v_cache, v_q[w], at);
    warp::tma::store_async_wait();
}

// Launch the `HEAD_DIM` instantiation over the launcher's checked shape.
template <int HEAD_DIM>
int launch_kv_append(bf16* kp, bf16* vp, int* sp, fp8e4m3* kc, fp8e4m3* vc, uint32_t rows,
                     uint32_t kv_heads, uint32_t num_blocks, cudaStream_t stream) {
    const uint64_t cols = uint64_t(kv_heads) * uint64_t(HEAD_DIM);
    // Only the caches carry a TMA type: building them encodes their vector tensor maps into
    // host memory, nothing on the device. Their dims are ints, so the block count is the
    // most a u32 slot can address, which changes no write.
    const size_t blocks = size_t(num_blocks < MAX_SLOT_BLOCKS ? num_blocks : MAX_SLOT_BLOCKS);
    row_gl kg{kp, nullptr, nullptr, size_t(rows), size_t(cols)};
    row_gl vg{vp, nullptr, nullptr, size_t(rows), size_t(cols)};
    slot_gl sg{sp, nullptr, nullptr, nullptr, size_t(rows)};
    cache_gl<HEAD_DIM> kcg{kc, blocks, size_t(kv_heads), nullptr, nullptr};
    cache_gl<HEAD_DIM> vcg{vc, blocks, size_t(kv_heads), nullptr, nullptr};
    const int pairs = int(uint64_t(rows) * kv_heads);
    kv_globals<HEAD_DIM> G{kg, vg, sg, kcg, vcg, pairs, int(kv_heads), num_blocks};
    void* args[] = {&G};
    const dim3 grid((uint32_t(pairs) + uint32_t(WARPS) - 1) / uint32_t(WARPS)), block(NUM_THREADS);
    return static_cast<int>(
        cudaLaunchKernel(kv_append_fp8_ker<HEAD_DIM>, grid, block, args, 0, stream));
}

}  // namespace

// The kernel's one-time setup, run once per device by `kernels::prepare_all` (`CudaDevice::open`):
// query its attributes (`cudaFuncGetAttributes`), which loads its module, so a worker's graph
// capture (`CudaWorker::new`) never loads a kernel mid-capture. It changes nothing on the device.
//
// 0 = done; any other value = the cudaError_t of the first query that failed.
extern "C" int scratchy_cuda_kv_append_fp8_prepare(void) {
    cudaFuncAttributes attr;
    if (const cudaError_t e = cudaFuncGetAttributes(&attr, kv_append_fp8_ker<HEAD_DIM_SLIDING>); e != cudaSuccess) {
        return static_cast<int>(e);
    }
    if (const cudaError_t e = cudaFuncGetAttributes(&attr, kv_append_fp8_ker<HEAD_DIM_GLOBAL>); e != cudaSuccess) {
        return static_cast<int>(e);
    }
    return 0;
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value =
// the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// Refused before any pointer is touched: `rows` or `kv_heads` zero; `head_dim` not 256 or 512
// (`step::KV_APPEND_BF16_HEAD_DIMS`, the head dims the kernel is instantiated for); `block_size`
// not 64; `rows · kv_heads · head_dim` above INT_MAX. `k` and `v` must be 16-byte-aligned device
// pointers to `rows · kv_heads · head_dim` bf16, `slot_mapping` to `rows` u32, and `k_cache`,
// `v_cache` to `num_blocks · kv_heads · 64 · head_dim` e4m3 bytes each, overlapping no input.
// `num_blocks` 0 writes nothing, so nothing is launched. Static shared memory only, so no
// one-time setup. The launch is asynchronous on `stream`; the caller owns it and synchronizes.
extern "C" int scratchy_cuda_kv_append_fp8(const void* k, const void* v, const void* slot_mapping,
                                           void* k_cache, void* v_cache, uint32_t rows,
                                           uint32_t kv_heads, uint32_t head_dim,
                                           uint32_t block_size, uint32_t num_blocks,
                                           void* stream /* cudaStream_t; null = default stream */) {
    const uint64_t cols = uint64_t(kv_heads) * uint64_t(head_dim);
    if (rows == 0 || kv_heads == 0 ||
        (head_dim != uint32_t(HEAD_DIM_SLIDING) && head_dim != uint32_t(HEAD_DIM_GLOBAL)) ||
        block_size != BLOCK_SIZE || uint64_t(rows) * cols > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    if (num_blocks == 0) {
        return 0;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        bf16* kp = static_cast<bf16*>(const_cast<void*>(k));
        bf16* vp = static_cast<bf16*>(const_cast<void*>(v));
        int* sp = static_cast<int*>(const_cast<void*>(slot_mapping));
        fp8e4m3* kc = static_cast<fp8e4m3*>(k_cache);
        fp8e4m3* vc = static_cast<fp8e4m3*>(v_cache);
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        return head_dim == uint32_t(HEAD_DIM_GLOBAL)
                   ? launch_kv_append<HEAD_DIM_GLOBAL>(kp, vp, sp, kc, vc, rows, kv_heads,
                                                       num_blocks, s)
                   : launch_kv_append<HEAD_DIM_SLIDING>(kp, vp, sp, kc, vc, rows, kv_heads,
                                                        num_blocks, s);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
