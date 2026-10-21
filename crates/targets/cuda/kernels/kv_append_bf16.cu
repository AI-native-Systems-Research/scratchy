// SPDX-License-Identifier: Apache-2.0
//
// The paged KV cache's write for the H100: for each token row t of the forward whose slot
// `slot_mapping[t]` is not u32::MAX, row t of K and of V ([rows, kv_heads · head_dim] bf16,
// row-major) are copied bit for bit into one layer's K and V cache at that slot.
//
// Each cache is [num_blocks, kv_heads, block_size, head_dim] bf16 (`PagedKvLayout`, in
// `scratchy-subtile`): slot s is token slot `s % block_size` of physical block `s / block_size`,
// and kv head h of it starts at element `((block · kv_heads + h) · block_size + s % block_size) ·
// head_dim`. A padding row's slot is u32::MAX and writes nothing, as metal's
// `shaders/rope.metal` `rope_append_bf16_specialized` skips one; a block at or past `num_blocks`
// writes nothing either (the host refused it already: this is defence in depth).
//
// Plain CUDA, not ThunderKittens: a scatter copy of 16-byte vectors, not tile-shaped. It runs after
// `rope_bf16.cu` has rotated K into its own buffer (CUDA's `RopeAppend` step, two launches on one
// stream), so it copies the rotated K; V is copied as projected. Metal's counterpart of this copy
// is `shaders/kv_append.metal`.
//
// One thread per 16-byte vector (8 lanes) of one row of one head, copying that vector of K and of
// V. No atomics, no shared memory: the same input gives the same bits.

#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

// The launcher's return code that is not a `cudaError_t` (which is non-negative). `src/kernels.rs`
// mirrors it.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)

namespace {

// bf16 lanes per 16-byte vector: every load and store is one `uint4`.
constexpr uint32_t VEC = 8;
constexpr int THREADS = 256;
// Token slots per block, the attention kernel's K/V tile height (`step::CUDA_KV_BLOCK_SIZE`).
constexpr uint32_t BLOCK_SIZE = 64;
// A padding row's slot.
constexpr uint32_t NO_SLOT = 0xFFFFFFFFu;

// `vecs` = rows · cols / 8: one thread per vector of K (and the same vector of V). `row_vecs` =
// cols / 8, `head_vecs` = head_dim / 8.
__global__ void __launch_bounds__(THREADS)
kv_append_bf16_kernel(const uint4* __restrict__ k, const uint4* __restrict__ v,
                      const uint32_t* __restrict__ slot_mapping, uint4* __restrict__ k_cache,
                      uint4* __restrict__ v_cache, int vecs, int row_vecs, int head_vecs,
                      uint32_t kv_heads, uint32_t num_blocks) {
    const int g = blockIdx.x * THREADS + threadIdx.x;
    if (g >= vecs) {
        return;
    }
    const int t = g / row_vecs;
    const uint32_t slot = slot_mapping[t];
    if (slot == NO_SLOT) {
        return;
    }
    const uint32_t block = slot / BLOCK_SIZE, off = slot % BLOCK_SIZE;
    if (block >= num_blocks) {
        return;
    }
    const int in_row = g % row_vecs;
    const uint32_t h = uint32_t(in_row / head_vecs), d = uint32_t(in_row % head_vecs);
    // In 16-byte vectors: ((block · kv_heads + h) · BLOCK_SIZE + off) · head_dim / 8 + d.
    const size_t at =
        ((size_t(block) * kv_heads + h) * BLOCK_SIZE + off) * size_t(head_vecs) + d;
    k_cache[at] = k[g];
    v_cache[at] = v[g];
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched; any other value = the
// cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// Refused before any pointer is touched: `rows`, `kv_heads` or `head_dim` zero; `head_dim` not a
// multiple of 8; `block_size` not 64; `rows · kv_heads · head_dim` above INT_MAX. `k` and `v` must
// be 16-byte-aligned device pointers to `rows · kv_heads · head_dim` bf16, `slot_mapping` to `rows`
// u32, and `k_cache`, `v_cache` to `num_blocks · kv_heads · 64 · head_dim` bf16 each, overlapping
// no input. No shared memory, so no one-time setup. The launch is asynchronous on `stream`; the
// caller owns it and synchronizes.
extern "C" int scratchy_cuda_kv_append_bf16(const void* k, const void* v, const void* slot_mapping,
                                            void* k_cache, void* v_cache, uint32_t rows,
                                            uint32_t kv_heads, uint32_t head_dim,
                                            uint32_t block_size, uint32_t num_blocks,
                                            void* stream /* cudaStream_t; null = default stream */) {
    const uint64_t cols = uint64_t(kv_heads) * uint64_t(head_dim);
    if (rows == 0 || kv_heads == 0 || head_dim == 0 || head_dim % VEC != 0 ||
        block_size != BLOCK_SIZE || uint64_t(rows) * cols > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const uint4* kp = static_cast<const uint4*>(k);
    const uint4* vp = static_cast<const uint4*>(v);
    const uint32_t* sp = static_cast<const uint32_t*>(slot_mapping);
    uint4* kc = static_cast<uint4*>(k_cache);
    uint4* vc = static_cast<uint4*>(v_cache);
    int vecs = static_cast<int>(uint64_t(rows) * cols / VEC);
    int row_vecs = static_cast<int>(cols / VEC);
    int head_vecs = static_cast<int>(head_dim / VEC);
    void* args[] = {&kp, &vp, &sp, &kc, &vc, &vecs, &row_vecs, &head_vecs, &kv_heads, &num_blocks};
    const dim3 grid((vecs + THREADS - 1) / THREADS), block(THREADS);
    return static_cast<int>(cudaLaunchKernel(kv_append_bf16_kernel, grid, block, args, 0,
                                             static_cast<cudaStream_t>(stream)));
}
