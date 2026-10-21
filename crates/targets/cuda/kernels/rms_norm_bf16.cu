// SPDX-License-Identifier: Apache-2.0
//
// bf16 RMSNorm for the H100: y[r, c] = x[r, c] / sqrt(mean(x[r, :]²) + eps) · (gain_offset + g[c]),
// with x and y [rows, cols] and g [cols], all row-major bf16. All math is f32; each output is
// rounded to bf16 once (round-to-nearest-even). `gain_offset` is the gain convention's constant:
// 0 for a stored gain (llama, qwen3), 1 for a zero-centred one (qwen3.5, gemma). One kernel serves
// both; the macro never folds the offset into the weights.
//
// Plain CUDA, not ThunderKittens: the op is a row reduction followed by a per-element map, with no
// matrix multiply, so register/shared tiles, WGMMA and TMA bring nothing. Metal's counterpart is
// `shaders/rmsnorm.metal` (f32 sum of squares, `rms = sqrt(sum/N + eps)`, `(x / rms) · w`).
//
// One 256-thread block per row, in two passes over the row:
//   1. each thread sums x² over its 16-byte vectors (a fixed stride), each warp reduces its 32
//      partials with an xor butterfly, lane 0 of each warp stores the warp's sum in shared memory,
//      and after one __syncthreads every thread adds the 8 warp sums in warp order. The order of
//      every addition is fixed by the thread layout alone, with no atomics, so the same input gives
//      the same bits on every run.
//   2. each thread re-reads its vectors of x and of the gain and writes y.
//
// Built with `--use_fast_math` (build.rs): rsqrtf and the division are the approximate ones, and
// subnormals flush to zero. The tests' bound (2⁻⁸ relative plus 2⁻²⁴ absolute) allows for both.

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
// Threads per block, one block per row.
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

// The 8 bf16 at `p` (16-byte aligned), widened to f32. Widening is exact.
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

// `f`, each rounded once to bf16 (round-to-nearest-even), stored as one vector at `p`.
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
rms_norm_bf16_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ gain,
                     __nv_bfloat16* __restrict__ y, int cols, float eps, float gain_offset) {
    __shared__ float warp_sums[WARPS];
    const size_t row = blockIdx.x;
    const __nv_bfloat16* xr = x + row * size_t(cols);
    __nv_bfloat16* yr = y + row * size_t(cols);
    // `cols` is a multiple of VEC (the launcher's rule), so a row is whole vectors: no tail.
    const int vecs = cols / VEC;

    float sum = 0.0f;
    for (int v = threadIdx.x; v < vecs; v += THREADS) {
        float f[VEC];
        load8(xr + v * VEC, f);
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            sum = fmaf(f[i], f[i], sum);
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    if (threadIdx.x % 32 == 0) {
        warp_sums[threadIdx.x / 32] = sum;
    }
    __syncthreads();
    // Every thread adds the warp sums in the same order, so every thread holds the same `inv`.
    // Nothing writes `warp_sums` after the barrier, so no second one is needed.
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
        total += warp_sums[w];
    }
    const float inv = rsqrtf(total / float(cols) + eps);

    for (int v = threadIdx.x; v < vecs; v += THREADS) {
        float f[VEC], g[VEC];
        load8(xr + v * VEC, f);
        load8(gain + v * VEC, g);
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            f[i] = f[i] * inv * (g[i] + gain_offset);
        }
        store8(yr + v * VEC, f);
    }
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched; any other value = the
// cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `rows` and `cols` must be non-zero, `cols` a multiple of 8 (every row then starts 16-byte aligned
// and is whole vectors), and `rows · cols` at most INT_MAX; all three are checked before any
// pointer is touched. `x`, `gain` and `y` must be 16-byte-aligned device pointers to `rows·cols`,
// `cols` and `rows·cols` bf16, `y` overlapping neither input. 32 bytes of static shared memory,
// so no one-time setup. The launch is asynchronous on `stream`; the caller owns it and
// synchronizes.
extern "C" int scratchy_cuda_rms_norm_bf16(const void* x, const void* gain, void* y, uint32_t rows,
                                           uint32_t cols, float eps, float gain_offset,
                                           void* stream /* cudaStream_t; null = default stream */) {
    if (rows == 0 || cols == 0 || cols % VEC != 0 ||
        uint64_t(rows) * uint64_t(cols) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const __nv_bfloat16* xp = static_cast<const __nv_bfloat16*>(x);
    const __nv_bfloat16* gp = static_cast<const __nv_bfloat16*>(gain);
    __nv_bfloat16* yp = static_cast<__nv_bfloat16*>(y);
    int icols = static_cast<int>(cols);
    void* args[] = {&xp, &gp, &yp, &icols, &eps, &gain_offset};
    const dim3 grid(rows), block(THREADS);
    return static_cast<int>(cudaLaunchKernel(rms_norm_bf16_kernel, grid, block, args, 0,
                                             static_cast<cudaStream_t>(stream)));
}
