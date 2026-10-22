// SPDX-License-Identifier: Apache-2.0
//
// bf16 GEMV for the H100: c[n] = Σ_k a[k] · B[n, k], one activation row `a [1, k]` against a
// row-major `B [n, k]` weight, all bf16, accumulated in f32 and rounded once to bf16
// (round-to-nearest-even). It is the `m = 1` case of `gemm_bf16.cu`'s `C = A · Bᵀ`: a decode
// step's every projection and its lm_head.
//
// Plain CUDA, not ThunderKittens: with one row, a WGMMA tile (64 rows at the least) would waste at
// least 63/64 of its work. The op is memory-bound: it reads each weight once and does two flops per
// weight read, so its time is the time to read `B` (`n · k` bf16).
//
// One warp per output column `j`, 8 warps per 256-thread CTA, grid `ceil(n / 8)`. Lane `l` takes
// the 16-byte vectors (8 bf16) `l, l + 32, …` of `k / 8` along `B`'s row `j`, which is contiguous
// in `k`, so each warp step reads 512 coalesced bytes of `B`. Each lane sums its products in
// ascending `k` with `fmaf` into one f32; then a fixed xor butterfly (offsets 16, 8, 4, 2, 1)
// reduces the warp, and lane 0 rounds once and stores `c[j]`. `a` (at most a few thousand bf16) is
// read through L1, not staged. No atomics and a fixed order, so the same input gives the same bits
// on every run. No shared memory and no one-time setup.
//
// No cross-CTA split of `k`: the decode GEMMs of qwen3.5-0.8b have `n` from 512 to 248,320, enough
// columns to fill 132 SMs, and a split would need a scratch buffer and a second launch.
//
// Built with `--use_fast_math` (build.rs): no approximate function is used here; flush-to-zero of
// subnormal products is covered by the GEMM bound's absolute term.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

// The launcher's return code that is not a `cudaError_t` (which is non-negative). `src/kernels.rs`
// mirrors it.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)

namespace {

// bf16 values per 16-byte vector: `step::GEMV_BF16_MULTIPLE`.
constexpr int VEC = 8;
constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;

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

__global__ void __launch_bounds__(THREADS)
gemv_bf16_kernel(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
                 __nv_bfloat16* __restrict__ c, int n, int k) {
    const int lane = threadIdx.x % 32;
    const int col = blockIdx.x * WARPS + threadIdx.x / 32;
    if (col >= n) {
        return;
    }
    const __nv_bfloat16* br = b + size_t(col) * size_t(k);
    const int vecs = k / VEC;
    float acc = 0.0f;
    for (int v = lane; v < vecs; v += 32) {
        float x[VEC], w[VEC];
        load8(a + size_t(v) * VEC, x);
        load8(br + size_t(v) * VEC, w);
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            acc = fmaf(x[i], w[i], acc);
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        c[col] = __float2bfloat16_rn(acc);
    }
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched; any other value = the
// cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `n` must be non-zero, `k` a non-zero multiple of 8, and both at most INT_MAX; all are checked
// before any pointer is touched. `a`, `b` and `c` must be 16-byte-aligned device pointers to `k`,
// `n·k` and `n` bf16, `c` overlapping neither input. No shared memory, so no one-time setup. The
// launch is asynchronous on `stream`.
extern "C" int scratchy_cuda_gemv_bf16_abt(const void* a, const void* b, void* c, uint32_t n,
                                           uint32_t k,
                                           void* stream /* cudaStream_t; null = default */) {
    if (n == 0 || n > uint32_t(INT_MAX) || k == 0 || k % VEC != 0 || k > uint32_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const __nv_bfloat16* ap = static_cast<const __nv_bfloat16*>(a);
    const __nv_bfloat16* bp = static_cast<const __nv_bfloat16*>(b);
    __nv_bfloat16* cp = static_cast<__nv_bfloat16*>(c);
    int in = static_cast<int>(n), ik = static_cast<int>(k);
    void* args[] = {&ap, &bp, &cp, &in, &ik};
    const dim3 grid((n + WARPS - 1) / WARPS), block(THREADS);
    return static_cast<int>(cudaLaunchKernel(gemv_bf16_kernel, grid, block, args, 0,
                                             static_cast<cudaStream_t>(stream)));
}
