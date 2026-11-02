// SPDX-License-Identifier: Apache-2.0
//
// A norm's row quantized for the W4A8 GEMM — the norm-side half of the `affine_w4a8_quant`
// pre-pass (`quantized_qmm_nax.metal`'s `w4a8_quant_impl`), for the quantizing RMSNorm kernels
// (`rmsnorm_w4a8q_*`, `fused_add_rmsnorm_w4a8q_*`): the threadgroup that normed row `m` holds the
// row it stored, rounded through `T`, in threadgroup memory, and quantizes it from there into the
// pre-pass's scratch layout — int8 `xq[M][K]`, then float2 `(a, a * sum)` per (row, 64-chunk) past
// the `M` rows. One lane octet (8 lanes x 8 elements) a 64-chunk, the pre-pass's: the same values,
// the same arithmetic in the same order, so the same bytes.

#pragma once
#include <metal_stdlib>
using namespace metal;

template <typename T>
METAL_FUNC void norm_w4a8_quant_row(
    const threadgroup T* row, device uchar* scratch, uint m, uint K, uint M, uint tid,
    uint threads) {
    const uint KC = K / 64;
    const uint l8 = tid % 8;
    for (uint c = tid / 8; c < KC; c += threads / 8) {
        const threadgroup T* p = row + c * 64 + l8 * 8;
        const float4 v0 = float4(float(p[0]), float(p[1]), float(p[2]), float(p[3]));
        const float4 v1 = float4(float(p[4]), float(p[5]), float(p[6]), float(p[7]));
        float amax = max(max(max(abs(v0.x), abs(v0.y)), max(abs(v0.z), abs(v0.w))),
                         max(max(abs(v1.x), abs(v1.y)), max(abs(v1.z), abs(v1.w))));
        amax = max(amax, simd_shuffle_xor(amax, 1));
        amax = max(amax, simd_shuffle_xor(amax, 2));
        amax = max(amax, simd_shuffle_xor(amax, 4));
        // The pre-pass compiles without fast math (`quantized_qmm_nax` reaches MPP), a norm with
        // it: its divides are the correctly rounded ones, spelled out.
        const float inv = amax > 0.0f ? precise::divide(127.0f, amax) : 0.0f;
        const float a = precise::divide(amax, 127.0f);
        const float4 q0 = rint(v0 * inv), q1 = rint(v1 * inv);
        float sum = (q0.x + q0.y) + (q0.z + q0.w) + (q1.x + q1.y) + (q1.z + q1.w);
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        sum += simd_shuffle_xor(sum, 4);
        device uint2* xq = (device uint2*)(scratch + int64_t(m) * K + c * 64);
        xq[l8] = uint2(as_type<uint>(char4(q0)), as_type<uint>(char4(q1)));
        if (l8 == 0) {
            ((device float2*)(scratch + int64_t(M) * K))[int64_t(m) * KC + c] = float2(a, a * sum);
        }
    }
}
