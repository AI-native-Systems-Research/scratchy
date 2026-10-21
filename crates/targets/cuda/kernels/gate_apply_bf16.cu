// SPDX-License-Identifier: Apache-2.0
//
// bf16 attention output gate for the H100 (qwen3.5's `attn_output_gate`): out = attn · σ(gate) =
// attn / (1 + e^(−gate)), over [rows, cols], all row-major bf16 of the same shape (no broadcast).
// The math is f32 and each output is rounded once to bf16 (round-to-nearest-even). Metal's
// counterpart is `shaders/gate_apply.metal` (`a · 1/(1+exp(−g))` in f32); transformers'
// `Qwen3_5Attention` is `attn_output * torch.sigmoid(gate)`.
//
// Plain CUDA, not ThunderKittens: a per-element map. One 16-byte vector of 8 bf16 per thread, 256
// threads per block; the last block's threads past the end do nothing.
//
// Built with `--use_fast_math`: `__expf` and the division are the approximate ones, and subnormals
// flush to zero. For gate ≤ −88, `__expf(−gate)` overflows to +inf (or the denominator passes
// 2^126, where the fast division returns 0), so the output is a signed zero, not NaN.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

// The launcher's return code that is not a `cudaError_t` (which is non-negative). `src/kernels.rs`
// mirrors it.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)

namespace {

// bf16 values per 16-byte vector: every load and store is one `uint4`.
constexpr int VEC = 8;
constexpr int THREADS = 256;

__device__ __forceinline__ void load8(const __nv_bfloat16* p, float (&f)[VEC]) {
    const uint4 raw = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < VEC / 2; ++i) {
        const float2 v = __bfloat1622float2(h[i]);
        f[2 * i] = v.x;
        f[2 * i + 1] = v.y;
    }
}

__device__ __forceinline__ void store8(__nv_bfloat16* p, const float (&f)[VEC]) {
    uint4 raw;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < VEC / 2; ++i) {
        h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    }
    *reinterpret_cast<uint4*>(p) = raw;
}

__global__ void __launch_bounds__(THREADS)
gate_apply_bf16_kernel(const __nv_bfloat16* __restrict__ attn,
                       const __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ out,
                       int vecs) {
    const int v = blockIdx.x * THREADS + threadIdx.x;
    if (v >= vecs) {
        return;
    }
    const size_t at = size_t(v) * VEC;
    float a[VEC], g[VEC];
    load8(attn + at, a);
    load8(gate + at, g);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        a[i] = a[i] / (1.0f + __expf(-g[i]));
    }
    store8(out + at, a);
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched; any other value = the
// cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `rows` and `cols` must be non-zero, `cols` a multiple of 8, and `rows · cols` at most INT_MAX;
// all three are checked before any pointer is touched. `attn`, `gate` and `out` must be
// 16-byte-aligned device pointers to `rows·cols` bf16 each, `out` overlapping neither input. No
// shared memory, so no one-time setup. The launch is asynchronous on `stream`; the caller owns it
// and synchronizes.
extern "C" int scratchy_cuda_gate_apply_bf16(const void* attn, const void* gate, void* out,
                                             uint32_t rows, uint32_t cols,
                                             void* stream /* cudaStream_t; null = default */) {
    if (rows == 0 || cols == 0 || cols % VEC != 0 ||
        uint64_t(rows) * uint64_t(cols) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const __nv_bfloat16* ap = static_cast<const __nv_bfloat16*>(attn);
    const __nv_bfloat16* gp = static_cast<const __nv_bfloat16*>(gate);
    __nv_bfloat16* op = static_cast<__nv_bfloat16*>(out);
    int vecs = static_cast<int>(uint64_t(rows) * uint64_t(cols) / VEC);
    void* args[] = {&ap, &gp, &op, &vecs};
    const dim3 grid((vecs + THREADS - 1) / THREADS), block(THREADS);
    return static_cast<int>(cudaLaunchKernel(gate_apply_bf16_kernel, grid, block, args, 0,
                                             static_cast<cudaStream_t>(stream)));
}
