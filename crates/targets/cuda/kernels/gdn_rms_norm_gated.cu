// SPDX-License-Identifier: Apache-2.0
//
// Gated-DeltaNet gated RMSNorm for the H100, one prefill chunk of one sequence: launch 4 of the
// four a `GatedDeltaNet` step makes (`src/step.rs`). Per token `t` and value head `h`, over the
// head's `head_v` lanes of the scan's f32 `o [rows, value_dim]` and the gate `z [rows, value_dim]`
// (bf16), with the bundle's f32 gain `w [head_v]`:
//
//     out[t, h·head_v + i] = o[i] / sqrt(mean(o²) + eps) · w[i] · SiLU(z[i])     (bf16)
//
// in f32, rounded once (metal's `gdn_rms_norm_gated.metal`). The gate is SiLU, not sigmoid, and
// the gain is plain, not the zero-centred `1 + w` of qwen3.5's other norms.
//
// Plain CUDA, not ThunderKittens: a 128-lane row reduction and a map.
//
// One warp per `(t, h)` row, 8 warps per 256-thread CTA, grid over `rows · nv` rows. Lane `l`
// holds lanes `l, l + 32, …` of the row (`head_v / 32` of them, at most 8); the sum of squares is
// each lane's in ascending order, then a fixed xor butterfly (16, 8, 4, 2, 1), so every lane holds
// the same bits and the result does not depend on the launch shape.
//
// PADDING ROWS (`t ≥ q_len`) ARE WRITTEN `+0.0` WITHOUT READING `o` OR `z`: the scan wrote no `o`
// there and `z`'s padding rows may hold NaN, which `0 · NaN` would carry. So the step's whole
// output is defined, and nothing of a padding row reaches it.
//
// Built with `--use_fast_math` (build.rs): `rsqrtf`, `expf` and the division are the approximate
// ones; the row kernels' bound (2⁻⁸·|ref| + 2⁻²⁴) allows for them.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

// The launcher's return code that is not a `cudaError_t` (which is non-negative). `src/kernels.rs`
// mirrors it.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)

namespace {

constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
// `step::GDN_HEAD_V_MAX` over 32 lanes.
constexpr int MAX_PER_LANE = 8;
// `step::GDN_HEAD_V_MULTIPLE`.
constexpr uint32_t HEAD_V_MULTIPLE = 32;

__global__ void __launch_bounds__(THREADS)
gdn_rms_norm_gated_kernel(const float* __restrict__ o, const __nv_bfloat16* __restrict__ z,
                          const float* __restrict__ w, const uint32_t* __restrict__ sequence,
                          __nv_bfloat16* __restrict__ out, int rows, int nv, int head_v,
                          float eps) {
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * WARPS + threadIdx.x / 32;  // (t, h) as t · nv + h
    if (row >= rows * nv) {
        return;
    }
    const int t = row / nv;
    const int per_lane = head_v / 32;
    const size_t base = size_t(row) * head_v;  // `[rows, nv · head_v]`: row-major (t, h, i)
    if (t >= int(sequence[1])) {
        // A padding row: written, never read.
#pragma unroll
        for (int j = 0; j < MAX_PER_LANE; ++j) {
            if (j < per_lane) {
                out[base + j * 32 + lane] = __float2bfloat16_rn(0.0f);
            }
        }
        return;
    }
    float v[MAX_PER_LANE];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < MAX_PER_LANE; ++j) {
        v[j] = 0.0f;
        if (j < per_lane) {
            v[j] = o[base + j * 32 + lane];
            ss = fmaf(v[j], v[j], ss);
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        ss += __shfl_xor_sync(0xffffffffu, ss, offset);
    }
    const float inv = rsqrtf(ss / float(head_v) + eps);
#pragma unroll
    for (int j = 0; j < MAX_PER_LANE; ++j) {
        if (j < per_lane) {
            const int i = j * 32 + lane;
            const float zi = __bfloat162float(z[base + i]);
            const float silu = zi / (1.0f + expf(-zi));
            out[base + i] = __float2bfloat16_rn(v[j] * inv * w[i] * silu);
        }
    }
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched; any other value = the
// cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `rows` and `nv` must be non-zero, `head_v` a non-zero multiple of 32 and at most 256,
// `rows · nv · head_v` at most INT_MAX, and `eps` finite and above 0; all are checked before any
// pointer is touched. `o` is `[rows, nv · head_v]` f32, `z` and `out` the same shape in bf16, `w`
// `[head_v]` f32, `sequence` `[slot, q_len, start, 0]` u32 with `q_len ≤ rows`; `out` overlaps no
// input. No shared memory, so no one-time setup. The launch is asynchronous on `stream`.
extern "C" int scratchy_cuda_gdn_rms_norm_gated(const void* o, const void* z, const void* w,
                                                const void* sequence, void* out, uint32_t rows,
                                                uint32_t nv, uint32_t head_v, float eps,
                                                void* stream /* cudaStream_t; null = default */) {
    if (rows == 0 || nv == 0 || head_v == 0 || head_v % HEAD_V_MULTIPLE != 0 ||
        head_v > uint32_t(MAX_PER_LANE * 32) ||
        uint64_t(rows) * uint64_t(nv) * uint64_t(head_v) > uint64_t(INT_MAX) ||
        !std::isfinite(eps) || !(eps > 0.0f)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const float* opp = static_cast<const float*>(o);
    const __nv_bfloat16* zp = static_cast<const __nv_bfloat16*>(z);
    const float* wp = static_cast<const float*>(w);
    const uint32_t* qp = static_cast<const uint32_t*>(sequence);
    __nv_bfloat16* outp = static_cast<__nv_bfloat16*>(out);
    int ir = static_cast<int>(rows), inv = static_cast<int>(nv), ihv = static_cast<int>(head_v);
    void* args[] = {&opp, &zp, &wp, &qp, &outp, &ir, &inv, &ihv, &eps};
    const int n = static_cast<int>(rows * nv);
    const dim3 grid((n + WARPS - 1) / WARPS), block(THREADS);
    return static_cast<int>(cudaLaunchKernel(gdn_rms_norm_gated_kernel, grid, block, args, 0,
                                             static_cast<cudaStream_t>(stream)));
}
