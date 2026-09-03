// SPDX-License-Identifier: Apache-2.0
//
// Probe kernel for the open MPP `matmul2d` cooperative-tensor layout
// bug. Writes per-(lane, idx) (row, col, is_valid) for each cooperative
// tensor (A=16x16 bf16, B=16x32 bf16, C=16x32 float) so we can compare
// MPP's actual layout to MLX's `BaseNAXFrag` assumption.

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include <metal_tensor>
#include "metal_nax.h"   // NAXTile + tile_matmad_nax — the kernel's real MMA path

using namespace metal;

constexpr constant int CAP_MAX = 32;

[[kernel]] void nax_probe(
    device int32_t* out [[buffer(0)]],
    uint simd_lid [[thread_index_in_simdgroup]])
{
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16,
        /*transpose_a=*/false, /*transpose_b=*/true, /*relaxed_precision=*/false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    auto ct_a = gemm_op.template
        get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_b = gemm_op.template
        get_right_input_cooperative_tensor<bfloat, bfloat, float>();

    using tA_type = metal::tensor<device bfloat, metal::dextents<int, 2>, metal::tensor_inline>;
    using tB_type = metal::tensor<threadgroup bfloat, metal::dextents<int, 2>, metal::tensor_inline>;
    auto ct_c = gemm_op.template
        get_destination_cooperative_tensor<tA_type, tB_type, float>();

#define DUMP_CT(ct_, OP) do { \
    int cap = (int)ct_.get_capacity(); \
    for (int idx = 0; idx < CAP_MAX; idx++) { \
        int base = (OP) * 32 * CAP_MAX * 4 + int(simd_lid) * CAP_MAX * 4 + idx * 4; \
        out[base + 0] = cap; \
        if (idx < cap) { \
            bool v = ct_.is_valid_element(idx); \
            out[base + 1] = v ? 1 : 0; \
            if (v) { \
                auto mdi = ct_.get_multidimensional_index(idx); \
                out[base + 2] = (int)mdi[0]; \
                out[base + 3] = (int)mdi[1]; \
            } else { \
                out[base + 2] = -1; \
                out[base + 3] = -1; \
            } \
        } else { \
            out[base + 1] = -1; \
            out[base + 2] = -1; \
            out[base + 3] = -1; \
        } \
    } \
} while(0)

    DUMP_CT(ct_a, 0);
    DUMP_CT(ct_b, 1);
    DUMP_CT(ct_c, 2);
}

// Discriminating MMA sanity test: A[m, k] = (m+1), B[n, k] = (n+1) for
// all k. With descriptor (M=16, N=32, K=16, transpose_b=true):
//   output[m, n] = sum_{k=0..15} A[m,k] * B[n,k] = 16 * (m+1) * (n+1)
// So output[0, 0] = 16, output[0, 1] = 32, output[5, 3] = 384, etc.
[[kernel]] void nax_ones_mma(
    device float* c_out [[buffer(0)]],  // 16 × 32 float
    threadgroup bfloat* b_ws [[threadgroup(0)]],  // 32 × 16 = 512 bf16
    uint simd_lid [[thread_index_in_simdgroup]])
{
    threadgroup bfloat a_tg[16 * 16];
    if (simd_lid == 0) {
        // A[m, k] = m. Distinct per row.
        for (int m = 0; m < 16; ++m)
            for (int k = 0; k < 16; ++k)
                a_tg[m * 16 + k] = bfloat(float(m));
        // B[n, k] = 1.
        for (int n = 0; n < 32; ++n)
            for (int k = 0; k < 16; ++k)
                b_ws[n * 16 + k] = bfloat(1.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16,
        /*transpose_a=*/false, /*transpose_b=*/true, /*relaxed_precision=*/false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    using tA_extents = metal::extents<int, 16, 16>;
    using tB_extents = metal::extents<int, 32, 16>;
    using tA_type = metal::tensor<threadgroup bfloat, tA_extents, metal::tensor_inline>;
    using tB_type = metal::tensor<threadgroup bfloat, tB_extents, metal::tensor_inline>;

    auto cT = gemm_op.template
        get_destination_cooperative_tensor<tA_type, tB_type, float>();

    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        int16_t idx = static_cast<int16_t>(i);
        if (cT.is_valid_element(idx)) cT[idx] = 0.0f;
    }

    tA_type tA(a_tg, tA_extents{}, metal::array<int, 2>{16, 1});
    tB_type tB(b_ws, tB_extents{}, metal::array<int, 2>{16, 1});

    gemm_op.run(tA, tB, cT);

    // Store cT per-index to c_out (16x32 float, row-major stride 32)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        int16_t idx = static_cast<int16_t>(i);
        if (!cT.is_valid_element(idx)) continue;
        auto mdi = cT.get_multidimensional_index(idx);
        int a0 = static_cast<int>(mdi[0]);
        int a1 = static_cast<int>(mdi[1]);
        // Write to (a0, a1) interpreting coord[0] as M and coord[1] as N.
        // The c_out buffer is 16x32 float row-major.
        if (a0 < 16 && a1 < 32) {
            c_out[a0 * 32 + a1] = cT[idx];
        }
        // Also write to a "shadow" buffer to confirm via swapped axes,
        // if axes are swapped (a0=N, a1=M).
        if (a1 < 16 && a0 < 32) {
            c_out[16 * 32 + a1 * 32 + a0] = cT[idx];
        }
    }
}

// Faithful-path MMA check: mirrors `qmm_t_nax_impl`'s exact register
// path — NAXTile load (threadgroup) → tile_matmad_nax → store — with
// A=B=1.0 over a 32×32×32 (TM=TN=TK=2) tile on ONE simdgroup. Every
// output element must equal K=32 (sum of 32 ones). If this is wrong,
// the bug is in metal_nax.h's MMA on M5, not the quantized loader.
[[kernel]] void nax_frag_mma_check(
    device float* d_out [[buffer(0)]],   // 32×32 float, row-major ld=32
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]])
{
    using namespace mlx::steel;
    constexpr int SZ = 32;
    threadgroup bfloat a_tg[SZ * SZ];
    threadgroup bfloat b_tg[SZ * SZ];
    // Dispatched with 128 threads (4 simdgroups) like mlx's qmm_t_nax;
    // all-ones fill is idempotent across the redundant writers.
    if (simd_lid == 0) {
        for (int i = 0; i < SZ * SZ; ++i) { a_tg[i] = bfloat(1.0f); b_tg[i] = bfloat(1.0f); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    NAXTile<bfloat, 2, 2> Atile;   // [32, 32]
    NAXTile<bfloat, 2, 2> Btile;   // [32, 32]
    NAXTile<float,  2, 2> Dtile;   // [32, 32]
    Dtile.clear();

    Atile.template load<bfloat, SZ, 1>(a_tg);
    Btile.template load<bfloat, SZ, 1>(b_tg);
    tile_matmad_nax(Dtile,
                    Atile, metal::bool_constant<false>{},
                    Btile, metal::bool_constant<true>{});
    // Only simdgroup 0 writes (others compute redundantly) to avoid races.
    if (simd_gid == 0) {
        Dtile.store(d_out, SZ);
    }
}
