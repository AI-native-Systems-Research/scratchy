// SPDX-License-Identifier: Apache-2.0
// Shared block-wide sum reduction for the SDPA kernels (sdpa_naive_kernels.cu
// and sdpa_flash_decode_kernels.cu). Header-only (#included, not a compiled
// unit — no build.rs entry needed).
#pragma once

// Block-wide reduction of a per-thread value into thread 0, using warp shuffles
// + one shared-memory hop across warps. `red` must hold at least
// `blockDim.x / 32` floats. The full sum is valid ONLY in lane 0 of warp 0
// (i.e. threadIdx.x == 0); every other thread's return value is undefined and
// the caller must broadcast the result via shared memory. blockDim.x must be a
// multiple of 32 and <= 1024 (so num_warps <= 32 fits in `red`).
__device__ __forceinline__ float block_reduce_sum(float v, float* red) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float total = 0.0f;
    if (warp == 0) {
        float w = (lane < num_warps) ? red[lane] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) w += __shfl_down_sync(0xffffffffu, w, o);
        total = w;  // valid in lane 0 of warp 0 only
    }
    return total;
}
