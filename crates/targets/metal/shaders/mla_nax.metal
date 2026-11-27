// SPDX-License-Identifier: Apache-2.0
//
// The tiled absorbed attention (`mla.metal`'s `mla_attend_tiles`) on the matrix
// unit (M5+): the same queries, keys, masks and online softmax, its products
// `matmul2d` ops as `attention_nax_paged_wide`'s
// (steel_attention_nax_paged_kernel.h) are, which this kernel's loop follows.
//
// A threadgroup takes BLOCKS blocks of 16 of a sequence's queries (its rows'
// heads, consecutive), each a block of LATENT / WARP_LANES warps.
// A step, warp `wn` of a block scores page group `wn` (KEYS cached rows) over
// the whole query: one op over the lanes both the latent query and the K store
// hold, one over the latent lanes the V store holds, one over the rope lanes.
// The block's scores meet in threadgroup memory, and each warp accumulates its
// WARP_LANES output lanes as ops of 32 lanes — a 32-lane slice never straddles
// the K and V stores (HALF is whole slices) — P converted to their left input
// in registers. Every op reads its operands from device memory itself.
//
// Bindings and dispatch as `mla_attend_tiles`'.
// Constants: `mla.h`'s.

#include <metal_stdlib>
#include "baked.h"
#include "mla.h"
#include "metal_nax.h"
#include "mlx_steel_attn/attn_ops.h"

using namespace metal;
using namespace mlx::steel;

// Unset (another kernel's compile of this file): one op each, never dispatched.
constant constexpr uint MLA_NAX_LANES = MLA_WARP_LANES_SET ? MLA_WARP_LANES : 32u;
constant constexpr uint MLA_NAX_KEYS = MLA_KEYS_SET ? MLA_KEYS : 16u;
// The lanes of one P @ V op.
constant constexpr uint MLA_NAX_SLICE = 32u;
static_assert(!MLA_WARP_LANES_SET
                  || (MLA_LATENT % MLA_WARP_LANES == 0 && MLA_WARP_LANES % MLA_NAX_SLICE == 0
                      && MLA_HALF % MLA_NAX_SLICE == 0 && MLA_ROPE % 16u == 0),
              "whole 32-lane slices a warp, none straddling the stores");
static_assert(!MLA_KEYS_SET || (MLA_KEYS % 16u == 0 && MLA_BLOCK_SIZE % MLA_KEYS == 0),
              "whole ops a page group");

template <typename T>
[[kernel, max_total_threads_per_threadgroup(MLA_TILE_THREADS)]]
void mla_attend_tiles_nax(
    device       T*        out          [[buffer(0)]],
    device const T*        q_latent     [[buffer(1)]],
    device const T*        q            [[buffer(2)]],
    device const uint*     seq_used_k   [[buffer(3)]],
    device const uint*     block_table  [[buffer(4)]],
    device const uint64_t* k_cache      [[buffer(5)]],
    device const uint64_t* v_cache      [[buffer(6)]],
    device const uint*     cu_seqlens_q [[buffer(7)]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid           [[threadgroup_position_in_grid]])
{
    using namespace mpp::tensor_ops;
    using Ext = dextents<int32_t, 2>;
    typedef float AccumType;
    constexpr short kU = 16;
    constexpr int WM = int(MLA_TILE_BLOCKS);
    constexpr int BQ = WM * kU;
    constexpr int WN = int(MLA_TILE_WN);
    constexpr int DW = int(MLA_NAX_LANES);
    constexpr int SL = int(MLA_NAX_SLICE);
    constexpr int NSL = DW / SL;
    constexpr int KQ = int(MLA_NAX_KEYS);
    constexpr short TK = KQ / kU;
    // The query's parts against the row's stores: lanes [0, HALF) of the latent query and the K
    // store; [HALF, LATENT) of the latent query and [0, LATENT - HALF) of the V store; the rope
    // query and the V store's last ROPE lanes.
    constexpr int D_K = int(MLA_HALF);
    constexpr int D_V = int(MLA_LATENT) - int(MLA_HALF);
    constexpr int D_R = int(MLA_ROPE);

    const uint seq = tid.z;
    const uint seq_start = cu_seqlens_q[seq];
    const uint new_q = cu_seqlens_q[seq + 1u] - seq_start;
    const uint kv_len = seq_used_k[seq];
    const uint prefix = kv_len - new_q;
    const uint queries = new_q * MLA_HEADS;
    const uint item0 = tid.x * uint(BQ);
    if (item0 >= queries) {
        return;
    }
    const uint tile_items = min(uint(BQ), queries - item0);

    using stile_t = NAXTile<AccumType, 1, TK>;
    using otile_t = NAXTile<AccumType, 1, SL / kU>;

    const short rb = short(simd_group_id / uint(WN));
    const short wn = short(simd_group_id % uint(WN));
    const short tm = kU * rb;
    const short lim_rows = short(int(tile_items) - int(tm));
    const int rows = max(min(int(lim_rows), int(kU)), 1);
    // A block past the tile's queries still meets every step: it reads the tile's first queries
    // and stores nothing.
    const uint load_item = seq_start * MLA_HEADS + item0 + (lim_rows > 0 ? uint(tm) : 0u);

    const short2 simd_coord = stile_t::NAXFrag_t::get_coord();
    const short sm = simd_coord.y;
    const short sn = simd_coord.x;

    constexpr short kRowsPT = otile_t::kRowsPerThread;
    metal::vec<AccumType, kRowsPT> max_score;
    metal::vec<AccumType, kRowsPT> sum_score{0};
    // The most negative finite f32: a masked score exp2s to 0, never NaN.
    constexpr AccumType neg_inf = -3.40282347e+38f;
    for (short i = 0; i < kRowsPT; ++i) {
        max_score[i] = neg_inf;
    }

    const int pos_block = int(prefix + (item0 + uint(tm)) / MLA_HEADS);
    const int pos_last = int(prefix + (item0 + tile_items - 1u) / MLA_HEADS);
    const int kv_tiles = int((kv_len + uint(KQ) - 1u) / uint(KQ));
    const int kb_lim = min((pos_last + KQ) / KQ, kv_tiles);
    const int num_pages = int((kv_len + MLA_BLOCK_SIZE - 1u) / MLA_BLOCK_SIZE);
    device const uint* blocks = block_table + seq * MLA_MAX_BLOCKS;
    const AccumType scale2 = AccumType(MLA_SCALE) * AccumType(M_LOG2E_F);

    auto rows_at = [&](device const uint64_t* store, int key) -> device T* {
        return mla_rows_at<T>(store, blocks, num_pages, key);
    };

    constexpr auto qk_k_desc = matmul2d_descriptor(
        16, KQ, D_K, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto qk_v_desc = matmul2d_descriptor(
        16, KQ, D_V, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto qk_r_desc = matmul2d_descriptor(
        16, KQ, D_R, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_k_desc, execution_simdgroup> qk_k_op;
    matmul2d<qk_v_desc, execution_simdgroup> qk_v_op;
    matmul2d<qk_r_desc, execution_simdgroup> qk_r_op;
    constexpr auto pv_desc = matmul2d_descriptor(
        16, SL, KQ, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroup> pv_op;

    // Tensor extents bound the rows the ops read: queries past the tile's and V rows past kv_len
    // read as 0.
    tensor<device T, Ext, tensor_inline> ql_rows(
        (device T*)(q_latent + load_item * MLA_LATENT), Ext(int(MLA_LATENT), rows));
    tensor<device T, Ext, tensor_inline> qr_rows(
        (device T*)(q + load_item * MLA_Q_HEAD), Ext(int(MLA_Q_HEAD), rows));
    auto q_k = ql_rows.template slice<D_K, dynamic_extent>(0, 0);
    auto q_v = ql_rows.template slice<D_V, dynamic_extent>(D_K, 0);
    auto q_r = qr_rows.template slice<D_R, dynamic_extent>(int(MLA_NOPE), 0);
    auto store_keys = [&](device T* base) {
        return tensor<device T, Ext, tensor_inline>(base, Ext(int(MLA_HALF), KQ));
    };
    auto v_slice = [&](device T* base, int live) {
        tensor<device T, Ext, tensor_inline> v(base, Ext(int(MLA_HALF), live));
        return v.template slice<SL, dynamic_extent>(0, 0);
    };
    using sqk_t = decltype(q_k);
    using skk_t = decltype(store_keys(nullptr).template slice<D_K, KQ>(0, 0));
    using sqv_t = decltype(q_v);
    using skv_t = decltype(store_keys(nullptr).template slice<D_V, KQ>(0, 0));
    using sqr_t = decltype(q_r);
    using skr_t = decltype(store_keys(nullptr).template slice<D_R, KQ>(0, 0));
    using sv_t = decltype(v_slice(nullptr, 1));
    using pin_t = typename decltype(pv_op)::template cooperative_tensor_left_input_t<
        AccumType, T, AccumType>;
    // A warp's output lanes, four 32-lane slices (WARP_LANES 128), each its own op's destination.
    static_assert(!MLA_WARP_LANES_SET || NSL == 4, "four 32-lane slices a warp");
    auto Oc0 = pv_op.template get_destination_cooperative_tensor<pin_t, sv_t, AccumType>();
    auto Oc1 = pv_op.template get_destination_cooperative_tensor<pin_t, sv_t, AccumType>();
    auto Oc2 = pv_op.template get_destination_cooperative_tensor<pin_t, sv_t, AccumType>();
    auto Oc3 = pv_op.template get_destination_cooperative_tensor<pin_t, sv_t, AccumType>();
    auto each_slice = [&](auto f) {
        f(Oc0, 0);
        f(Oc1, 1);
        f(Oc2, 2);
        f(Oc3, 3);
    };
    each_slice([&](thread auto& oc, short) {
        STEEL_PRAGMA_UNROLL
        for (uint16_t i = 0; i < oc.get_capacity(); ++i) oc[i] = 0;
    });

    constexpr int kPart = 32 * KQ / 2;
    constexpr int kBufs = 2 * WM * WN * kPart * int(sizeof(AccumType)) <= 32768 ? 2 : 1;
    threadgroup AccumType sx[kBufs * WM * WN * kPart];

    for (int kb0 = 0; kb0 < kb_lim; kb0 += WN) {
        // This warp's page group's scores over the whole query, its three parts' ops summed.
        const int g = kb0 + wn;
        AccumType Sg[KQ / 2];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < KQ / 2; i++) Sg[i] = 0;
        if (g < kb_lim) {
            device T* kg = rows_at(k_cache, g * KQ);
            device T* vg = rows_at(v_cache, g * KQ);
            auto ck = qk_k_op.template get_destination_cooperative_tensor<sqk_t, skk_t, AccumType>();
            auto cv = qk_v_op.template get_destination_cooperative_tensor<sqv_t, skv_t, AccumType>();
            auto cr = qk_r_op.template get_destination_cooperative_tensor<sqr_t, skr_t, AccumType>();
            STEEL_PRAGMA_UNROLL
            for (uint16_t i = 0; i < ck.get_capacity(); ++i) {
                ck[i] = 0;
                cv[i] = 0;
                cr[i] = 0;
            }
            auto kt = store_keys(kg);
            auto vt = store_keys(vg);
            auto k_k = kt.template slice<D_K, KQ>(0, 0);
            auto k_v = vt.template slice<D_V, KQ>(0, 0);
            auto k_r = vt.template slice<D_R, KQ>(D_V, 0);
            qk_k_op.run(q_k, k_k, ck);
            qk_v_op.run(q_v, k_v, cv);
            qk_r_op.run(q_r, k_r, cr);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < KQ / 2; i++) Sg[i] = ck[i] + cv[i] + cr[i];
        }
        threadgroup AccumType* part = sx + (kb0 / WN % kBufs) * (WM * WN * kPart);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < KQ / 2; i++) {
            part[(simd_group_id * (KQ / 2) + i) * 32 + simd_lane_id] = Sg[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_UNROLL
        for (short t = 0; t < WN; t++) {
            const int kb = kb0 + t;
            if (kb >= kb_lim) {
                break;
            }
            const int key0 = kb * KQ;
            stile_t S;
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < stile_t::kElemsPerTile; i++) {
                S.elems()[i] = part[((rb * WN + t) * (KQ / 2) + i) * 32 + simd_lane_id];
            }
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < stile_t::kElemsPerTile; i++) {
                S.elems()[i] *= scale2;
            }
            if (key0 + KQ > int(kv_len) || key0 + KQ - 1 > pos_block) {
                STEEL_PRAGMA_UNROLL
                for (short ik = 0; ik < TK; ik++) {
                    thread auto& fg = S.frag_at(0, ik);
                    STEEL_PRAGMA_UNROLL
                    for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
                        const int row = sm + ii * stile_t::kFragRowsJump;
                        const int pos = int(prefix + (item0 + uint(tm + row)) / MLA_HEADS);
                        STEEL_PRAGMA_UNROLL
                        for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
                            const int c = key0 + ik * kU + sn + jj;
                            const auto loc = ii * stile_t::kFragThrCols + jj;
                            fg[loc] = (c >= int(kv_len) || c > pos) ? neg_inf : fg[loc];
                        }
                    }
                }
            }

            metal::vec<AccumType, kRowsPT> new_max;
            metal::vec<AccumType, kRowsPT> factor;
            for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];
            S.template row_reduce<MaxOp>(new_max);
            S.template row_bin_op<ExpSubOp>(new_max);
            for (short i = 0; i < kRowsPT; ++i) {
                factor[i] = fast::exp2(max_score[i] - new_max[i]);
                max_score[i] = new_max[i];
                sum_score[i] = sum_score[i] * factor[i];
            }
            S.template row_reduce<SumOp>(sum_score);
            // O's elements lie as S's: fragments of 16 columns, a lane's 4 of row sm then 4 of
            // row sm + 8.
            each_slice([&](thread auto& oc, short) {
                STEEL_PRAGMA_UNROLL
                for (uint16_t i = 0; i < oc.get_capacity(); ++i) oc[i] *= factor[(i % 8) / 4];
            });

            // O += P V, this warp's lanes 32 at a time from the store that holds them.
            const int live = min(int(kv_len) - key0, KQ);
            auto p = qk_k_op.template get_destination_cooperative_tensor<sqk_t, skk_t, AccumType>();
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < KQ / 2; i++) p[i] = S.elems()[i];
            auto P = pv_op.template get_left_input_cooperative_tensor<AccumType, T, AccumType>(p);
            device T* kg = rows_at(k_cache, key0);
            device T* vg = rows_at(v_cache, key0);
            each_slice([&](thread auto& oc, short s) {
                const int lane = int(wn) * DW + s * SL;
                device T* base = lane < int(MLA_HALF) ? kg + lane : vg + (lane - int(MLA_HALF));
                auto sV = v_slice(base, live);
                pv_op.run(P, sV, oc);
            });
        }
        if (kBufs == 1) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    metal::vec<AccumType, kRowsPT> rcp;
    for (short i = 0; i < kRowsPT; ++i) {
        rcp[i] = 1.f / sum_score[i];
    }
    if (lim_rows <= 0) {
        return;
    }
    device T* o = out + (seq_start * MLA_HEADS + item0 + uint(tm)) * MLA_LATENT + uint(wn) * uint(DW);
    each_slice([&](thread auto& oc, short s) {
        otile_t Otile;
        STEEL_PRAGMA_UNROLL
        for (uint16_t i = 0; i < oc.get_capacity(); ++i) {
            Otile.elems()[i] = oc[i] * rcp[(i % 8) / 4];
        }
        if (lim_rows < kU) {
            Otile.store_rows(o + s * SL, int(MLA_LATENT), lim_rows);
        } else {
            Otile.store(o + s * SL, int(MLA_LATENT));
        }
    });
}

SCRATCHY_KERNEL(mla_attend_tiles_nax_f16, mla_attend_tiles_nax<half>)
SCRATCHY_KERNEL(mla_attend_tiles_nax_bf16, mla_attend_tiles_nax<bfloat>)
