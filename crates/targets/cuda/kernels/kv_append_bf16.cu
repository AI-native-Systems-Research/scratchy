// SPDX-License-Identifier: Apache-2.0
//
// The paged KV cache's write for the H100: for each token row t of the forward whose slot
// `slot_mapping[t]` is not u32::MAX, row t of K and of V ([rows, kv_heads · head_dim] bf16,
// row-major) are copied bit for bit into one layer's K and V cache at that slot.
//
// Each cache is [num_blocks, kv_heads, block_size, head_dim] bf16 (`PagedKvLayout`, in
// `scratchy-subtile`): slot s is token slot `s % block_size` of physical block `s / block_size`,
// and kv head h of it is the cache's row `(block, h, s % block_size)`. A padding row's slot is
// u32::MAX and writes nothing, as metal's `shaders/rope.metal` `rope_append_bf16_specialized`
// skips one; a block at or past `num_blocks` writes nothing either (the host refused it already:
// this is defence in depth). It runs after `rope_bf16.cu` has rotated K into its own buffer (CUDA's
// `RopeAppend` step, two launches on one stream), so it copies the rotated K; V is copied as
// projected. Metal's counterpart of this copy is `shaders/kv_append.metal`.
//
// ThunderKittens' bulk copy, then a vector TMA store (PLAN-tk100 §2 S3). One warp per (token row
// t, kv head h) pair, 4 warps per CTA. Every lane reads the slot from the `gl<int>` slot view
// (`gl::operator[]`; u32::MAX reads as −1, so it is taken back to u32). For a slot that writes,
// lane 0 bulk-loads the 512-byte row-head of K and of V into two `sv_bf<256>` (`tma::load_async`,
// descriptor-free, `tma::expect_bytes` on the warp's semaphore), the warp waits, and lane 0
// stores each with the cache's vector tensor map at `{block, h, off, 0}` (`tma::store_async`; a
// 256-element bf16 vector is one 512-byte box, never swizzled). Bytes are never widened, so the
// copy is bit-exact, and every slot not mapped stays untouched.
//
// At head_dim 512 (Gemma 4's global layers, #885's 13b-3) a row-head is 1,024 bytes: the bulk load
// takes it whole into one `sv_bf<512>`, and TK's vector store writes it as two 256-element boxes
// at columns 0 and 256 (`ops/thread/memory/vec/tma.cuh` `store_async`: a TMA box dimension is at
// most 256 elements, `sv_tma_dim1`), each through the same tensor map. The kernel is instantiated
// once per head dim (`step::KV_APPEND_BF16_HEAD_DIMS`); the 256 instantiation is the code it was.
//
// Static shared memory: 4 warps × 2 vectors × 2 · head_dim bytes plus a semaphore each (8 KiB at
// 512), so no one-time setup.

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
template <int HEAD_DIM>
using head_vec = sv_bf<HEAD_DIM>;
template <int HEAD_DIM>
using cache_gl = gl<bf16, -1, -1, int(BLOCK_SIZE), HEAD_DIM, head_vec<HEAD_DIM>>;

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
__global__ __launch_bounds__(NUM_THREADS, 1) void kv_append_bf16_ker(
    const __grid_constant__ kv_globals<HEAD_DIM> g) {
    __shared__ sv_bf<HEAD_DIM> k_buf[WARPS];
    __shared__ sv_bf<HEAD_DIM> v_buf[WARPS];
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
    const coord<head_vec<HEAD_DIM>> at = {int(block), h, int(off), 0};
    warp::tma::store_async(g.k_cache, k_buf[w], at);
    warp::tma::store_async(g.v_cache, v_buf[w], at);
    warp::tma::store_async_wait();
}

// Launch the `HEAD_DIM` instantiation over the launcher's checked shape.
template <int HEAD_DIM>
int launch_kv_append(bf16* kp, bf16* vp, int* sp, bf16* kc, bf16* vc, uint32_t rows,
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
        cudaLaunchKernel(kv_append_bf16_ker<HEAD_DIM>, grid, block, args, 0, stream));
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value =
// the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// Refused before any pointer is touched: `rows` or `kv_heads` zero; `head_dim` not 256 or 512
// (`step::KV_APPEND_BF16_HEAD_DIMS`, the head dims the kernel is instantiated for); `block_size`
// not 64; `rows · kv_heads · head_dim` above INT_MAX. `k` and `v` must be 16-byte-aligned device
// pointers to `rows · kv_heads · head_dim` bf16, `slot_mapping` to `rows` u32, and `k_cache`,
// `v_cache` to `num_blocks · kv_heads · 64 · head_dim` bf16 each, overlapping no input.
// `num_blocks` 0 writes nothing, so nothing is launched. Static shared memory only, so no
// one-time setup. The launch is asynchronous on `stream`; the caller owns it and synchronizes.
extern "C" int scratchy_cuda_kv_append_bf16(const void* k, const void* v, const void* slot_mapping,
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
        bf16* kc = static_cast<bf16*>(k_cache);
        bf16* vc = static_cast<bf16*>(v_cache);
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
