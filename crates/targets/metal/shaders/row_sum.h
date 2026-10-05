// SPDX-License-Identifier: Apache-2.0
//
// A norm's row sum — ONE definition, for every RMSNorm kernel: the sum of a threadgroup's `tg_size`
// per-thread partials (a power of two, at most 1024), pairwise with the stride halving — the tree
// those kernels reduced in threadgroup memory, barrier by barrier. The strides whose partners sit in
// another simdgroup still go through `scratch`; the rest go through simdgroup shuffles, which pair
// the same lanes. Same sums in the same order, so the same bits — in 4 barriers where the tree took
// log2(tg_size) + 1. Threadgroup-uniform; every thread gets the total.

#pragma once
#include <metal_stdlib>
using namespace metal;

inline float row_sum(float partial, uint tid, uint tg_size, threadgroup float* scratch) {
    scratch[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint stride = tg_size / 2;
    for (; stride > 32; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 32) {
        float v = scratch[tid];
        if (stride == 32) {
            v += scratch[tid + 32];
            stride = 16;
        }
        for (; stride > 0; stride >>= 1) {
            const float p = simd_shuffle_down(v, ushort(stride));
            if (tid < stride) {
                v += p;
            }
        }
        if (tid == 0) {
            scratch[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}
