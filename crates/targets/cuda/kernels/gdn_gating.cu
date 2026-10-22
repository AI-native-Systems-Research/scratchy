// SPDX-License-Identifier: Apache-2.0
//
// Gated-DeltaNet gating for the H100, one prefill chunk of one sequence: launch 2 of the four a
// `GatedDeltaNet` step makes (`src/step.rs`). For each real token `t < q_len` and value head
// `h < nv`, from `a` and `b` `[rows, nv]` (bf16) and the bundle's f32 `a_log [nv]` and `dt_bias
// [nv]`:
//
//     g[t, h]    = −exp(a_log[h]) · softplus(a[t, h] + dt_bias[h])     (the log of the decay α)
//     beta[t, h] = 1 / (1 + exp(−b[t, h]))                              (β = sigmoid(b))
//
// with `softplus(x) = log1p(exp(x))` for `x ≤ 20`, else `x` (metal's `gdn_gating.metal`, its
// threshold 20). Both are f32, into the step's scratch.
//
// Plain CUDA, not ThunderKittens: an elementwise map.
//
// One thread per `(t, h)` of the `q_len · nv` real ones, 256 threads per block, as many blocks as
// cover `rows · nv`; a thread at or past `q_len · nv` returns before reading anything, so padding
// rows are never read nor written.
//
// Built with `--use_fast_math` (build.rs): `expf` is `__expf` and the division approximate.
// `log1pf` has no fast-math intrinsic and stays the library function. The tests' bound
// (2⁻¹⁶·|ref| + 2⁻²² for `g` and for `β`) allows for both.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

// The launcher's return code that is not a `cudaError_t` (which is non-negative). `src/kernels.rs`
// mirrors it.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)

namespace {

constexpr int THREADS = 256;
// The softplus's linear threshold (metal's, and torch's default).
constexpr float SOFTPLUS_THRESHOLD = 20.0f;

__global__ void __launch_bounds__(THREADS)
gdn_gating_kernel(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
                  const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                  const uint32_t* __restrict__ sequence, float* __restrict__ g,
                  float* __restrict__ beta, int nv) {
    const int i = blockIdx.x * THREADS + threadIdx.x;
    const int q_len = int(sequence[1]);
    if (i >= q_len * nv) {
        return;
    }
    const int h = i % nv;
    const float av = __bfloat162float(a[i]) + dt_bias[h];
    const float sp = av <= SOFTPLUS_THRESHOLD ? log1pf(expf(av)) : av;
    g[i] = -expf(a_log[h]) * sp;
    const float bv = __bfloat162float(b[i]);
    beta[i] = 1.0f / (1.0f + expf(-bv));
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched; any other value = the
// cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `rows` and `nv` must be non-zero and `rows · nv` at most INT_MAX; checked before any pointer is
// touched. `a` and `b` are `[rows, nv]` bf16, `a_log` and `dt_bias` `[nv]` f32, `g` and `beta`
// `[rows, nv]` f32, `sequence` `[slot, q_len, start, 0]` u32 with `1 ≤ q_len ≤ rows`; `g` and
// `beta` overlap nothing else. No shared memory, so no one-time setup. The launch is asynchronous
// on `stream`.
extern "C" int scratchy_cuda_gdn_gating(const void* a, const void* b, const void* a_log,
                                        const void* dt_bias, const void* sequence, void* g,
                                        void* beta, uint32_t rows, uint32_t nv,
                                        void* stream /* cudaStream_t; null = default stream */) {
    if (rows == 0 || nv == 0 || uint64_t(rows) * uint64_t(nv) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const __nv_bfloat16* ap = static_cast<const __nv_bfloat16*>(a);
    const __nv_bfloat16* bp = static_cast<const __nv_bfloat16*>(b);
    const float* lp = static_cast<const float*>(a_log);
    const float* dp = static_cast<const float*>(dt_bias);
    const uint32_t* qp = static_cast<const uint32_t*>(sequence);
    float* gp = static_cast<float*>(g);
    float* bt = static_cast<float*>(beta);
    int inv = static_cast<int>(nv);
    void* args[] = {&ap, &bp, &lp, &dp, &qp, &gp, &bt, &inv};
    const int n = static_cast<int>(rows * nv);
    const dim3 grid((n + THREADS - 1) / THREADS), block(THREADS);
    return static_cast<int>(cudaLaunchKernel(gdn_gating_kernel, grid, block, args, 0,
                                             static_cast<cudaStream_t>(stream)));
}
