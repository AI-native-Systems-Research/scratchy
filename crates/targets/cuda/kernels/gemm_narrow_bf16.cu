// SPDX-License-Identifier: Apache-2.0
//
// Narrow bf16 GEMM for the H100: C[m, n] = A[m, k] · B[n, k]ᵀ, all row-major bf16, accumulated in
// f32 and rounded once to bf16 (round-to-nearest-even), for `n` of at most 64 output columns. The
// gated delta net's `a` and `b` projections are `[hidden, linear_num_value_heads]`: 16 columns in
// qwen3.5-0.8b, 32 in 9b, which the tiled GEMM (`gemm_bf16.cu`, 256-column tiles) refuses.
//
// Plain CUDA, not ThunderKittens: 16 or 32 output columns is below the width at which a WGMMA tile
// does useful work, and the op is bound by reading A once (`m · k` bf16), not by math.
//
// One warp per output row, 8 warps per 256-thread CTA. The warp's 32 lanes split `k` into 16-byte
// vectors of 8 bf16 (lane `l` takes vectors `l, l + 32, …`), and each lane keeps `N` f32 partial
// sums, one per output column, in registers (`N` is a template parameter: 8, 16, …, 64). Each
// partial is the lane's products summed in ascending `k` with `fmaf`. Then a fixed-order xor
// butterfly (offsets 16, 8, 4, 2, 1) reduces each column across the warp, every lane ending with
// the same bits, and lanes `0 .. N/8` each store one 16-byte vector of 8 columns. The order of
// every addition is fixed by `k` and the lane layout alone, with no atomics, so the same input
// gives the same bits on every run. `B` is read through L1/L2, not staged: every warp of the
// grid reads all of it (9b's `[32, 4096]` bf16 is 256 KiB).
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

// bf16 values per 16-byte vector.
constexpr int VEC = 8;
constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
// `step::GEMM_NARROW_BF16_MAX_N`.
constexpr uint32_t MAX_N = 64;

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

template <int N>
__global__ void __launch_bounds__(THREADS)
gemm_narrow_bf16_kernel(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
                        __nv_bfloat16* __restrict__ c, int m, int k) {
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * WARPS + threadIdx.x / 32;
    if (row >= m) {
        return;
    }
    const __nv_bfloat16* ar = a + size_t(row) * size_t(k);
    const int vecs = k / VEC;
    float acc[N];
#pragma unroll
    for (int j = 0; j < N; ++j) {
        acc[j] = 0.0f;
    }
    for (int v = lane; v < vecs; v += 32) {
        float x[VEC];
        load8(ar + size_t(v) * VEC, x);
#pragma unroll
        for (int j = 0; j < N; ++j) {
            float w[VEC];
            load8(b + size_t(j) * size_t(k) + size_t(v) * VEC, w);
#pragma unroll
            for (int i = 0; i < VEC; ++i) {
                acc[j] = fmaf(x[i], w[i], acc[j]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < N; ++j) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], offset);
        }
    }
    // Every lane holds every column's sum; lane `g` stores columns `8g .. 8g + 8`.
    __nv_bfloat16* cr = c + size_t(row) * N;
#pragma unroll
    for (int g = 0; g < N / VEC; ++g) {
        if (lane == g) {
            uint4 raw;
            __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
            for (int i = 0; i < VEC / 2; ++i) {
                h[i] = __floats2bfloat162_rn(acc[g * VEC + 2 * i], acc[g * VEC + 2 * i + 1]);
            }
            *reinterpret_cast<uint4*>(cr + g * VEC) = raw;
        }
    }
}

template <int N>
cudaError_t launch(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, int m, int k,
                   cudaStream_t stream) {
    void* args[] = {&a, &b, &c, &m, &k};
    const dim3 grid((m + WARPS - 1) / WARPS), block(THREADS);
    return cudaLaunchKernel(gemm_narrow_bf16_kernel<N>, grid, block, args, 0, stream);
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched; any other value = the
// cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `m` must be non-zero, `n` a non-zero multiple of 8 and at most 64, `k` a non-zero multiple of 8,
// and `m` and `k` at most INT_MAX; all are checked before any pointer is touched. `a`, `b` and `c`
// must be 16-byte-aligned device pointers to `m·k`, `n·k` and `m·n` bf16, `c` overlapping neither
// input. No shared memory, so no one-time setup. The launch is asynchronous on `stream`.
extern "C" int scratchy_cuda_gemm_narrow_bf16_abt(const void* a, const void* b, void* c,
                                                  uint32_t m, uint32_t n, uint32_t k,
                                                  void* stream /* cudaStream_t; null = default */) {
    if (m == 0 || m > uint32_t(INT_MAX) || n == 0 || n % VEC != 0 || n > MAX_N || k == 0 ||
        k % VEC != 0 || k > uint32_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const __nv_bfloat16* ap = static_cast<const __nv_bfloat16*>(a);
    const __nv_bfloat16* bp = static_cast<const __nv_bfloat16*>(b);
    __nv_bfloat16* cp = static_cast<__nv_bfloat16*>(c);
    const int im = static_cast<int>(m), ik = static_cast<int>(k);
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    cudaError_t e;
    switch (n) {
        case 8: e = launch<8>(ap, bp, cp, im, ik, s); break;
        case 16: e = launch<16>(ap, bp, cp, im, ik, s); break;
        case 24: e = launch<24>(ap, bp, cp, im, ik, s); break;
        case 32: e = launch<32>(ap, bp, cp, im, ik, s); break;
        case 40: e = launch<40>(ap, bp, cp, im, ik, s); break;
        case 48: e = launch<48>(ap, bp, cp, im, ik, s); break;
        case 56: e = launch<56>(ap, bp, cp, im, ik, s); break;
        default: e = launch<64>(ap, bp, cp, im, ik, s); break;
    }
    return static_cast<int>(e);
}
