// SPDX-License-Identifier: Apache-2.0
//
// ⚠️⚠️ SPANS BIT-31 CONTRACT — READ BEFORE TOUCHING THE PAGED KERNELS ⚠️⚠️
// `slot_mapping` / `block_table` / `slots` / `logical_slots` entries carry the
// stored-unrotated flag in BIT 31 (0x80000000) for relocatable (span) blocks
// when rope-on-read is active. ANY use of these values as a memory INDEX MUST
// strip bit 31 first (`& 0x7FFFFFFFu`, mirroring attention's ATTN_BT_MASK).
// Forgetting it indexes ~2^31 elements OUT OF BOUNDS and silently corrupts the
// KV cache — but ONLY when spans are active, so it passes ordinary tests. This
// exact omission broke spans here once. Authoritative definition + rationale:
// crates/serving/worker/src/gpu_worker.rs (`slot |= 0x8000_0000`). Enforced by
// crates/targets/metal/tests/kv_index_bit31_mask_test.rs.
//
//! TurboQuant paged-KV kernels bound by the tape's TurboQuant ops; the codebook
//! math (norm, signs, WHT butterfly, nearest-centroid, bit packing) is a port of
//! arozanov's `turboquant_mlx/metal.py`. `dim` threads per threadgroup (dim <=
//! 512, power of two). Each kernel is one template over the cache element type
//! `T`, instantiated as `<name>` (half) and `<name>_bf16` (bfloat).
//!
//! `tq_compress_paged[_bf16]`: one threadgroup per (new KV slot, kv_head) —
//! quantize the pool vector into packed uint32 codes + an f32 norm in the
//! packed store, optionally writing the lossy dequant back into the pool.
//!
//! Attention reads the packed store itself (attention.metal): decode through
//! `attention_via_cache_v2`'s TurboQuant mode, prefill through the rotated-
//! domain image `tq_stage_rotated` stages.
//!
//! OFFSET. The compress codes each vector MINUS its additive offset
//! (`turboquant_offset.h`); every reader restores it exactly.

#include <metal_stdlib>
#include "megakernel/mk_common.h"
#include "turboquant_offset.h"
using namespace metal;

// Unnormalized Walsh-Hadamard transform of the `dim` floats in `shared`, one
// element per thread. Threadgroup-uniform (every thread runs every barrier).
inline void tq_wht_tg(threadgroup float* shared, uint dim, uint elem) {
    for (uint h = 1; h < dim; h *= 2) {
        uint blk = elem / (2 * h), off = elem % (2 * h);
        if (off < h) { uint j = blk * 2 * h + off; float a = shared[j], b = shared[j + h]; shared[j] = a + b; shared[j + h] = a - b; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// tq_compress_paged: the production wiring kernel. For each (new KV slot,
// kv_head), read the vector IN PLACE from the paged pool (chunk-table
// addressing, identical to the attention kernels), remove its offset, quantize
// it to packed codes + f32 norm written to the PACKED STORE (the canonical
// ~4.6x-smaller cache), then dequant, restore the offset and write the lossy
// vector back into the pool so the existing attention reads TurboQuant'd KV
// (dequant-to-buffer; the pool IS the buffer). One threadgroup per
// (slot, kv_head); dim threads. Dispatched post-forward over the new slots of
// one layer's K (and again for V).
// Body shared by the dispatch kernel and the megakernel adapter. KVP / UP / FP are the pool-page,
// packed-store and norm-store pointer types. MK = barrier-uniform form: no early return, every
// access predicated on `live`, every thread runs every barrier.
template <typename T, bool MK, typename KVP, typename UP, typename FP, typename CU, typename CF>
METAL_FUNC void tq_compress_body(
    device const uint64_t* chunk_table, device const uint* slots, device const float* signs,
    device const float* boundaries, device const float* centroids, UP packed_store, FP norms_store,
    CU dim, CU bits, CU vals_per_word, CU packed_dim, CU n_centroids, CF scale,
    CU num_kv_heads, CU block_size, CU blocks_per_chunk, device const uint* logical_slots,
    CU do_writeback, device const T* offset_bias, device const T* cos_sin,
    device const uint* positions, CU offset_mode, CU rot_dim, CU pair_off,
    uint3 tg, uint3 tid, bool live,
    threadgroup float* shared, threadgroup float* ns, threadgroup uint* idx_shared)
{
    uint slot_i       = tg.x;
    uint kv_head      = tg.y;
    uint elem         = tid.x;
    uint slot_raw     = (MK && !live) ? 0xFFFFFFFFu : slots[slot_i];
    if (!MK && slot_raw == 0xFFFFFFFFu) return;              // padding slot — no token
    if (MK) live = live && slot_raw != 0xFFFFFFFFu;
    uint slot         = slot_raw & 0x7FFFFFFFu;
    uint logical_slot = live ? (logical_slots[slot_i] & 0x7FFFFFFFu) : 0u;
    uint block   = slot / block_size;
    uint tok     = slot % block_size;
    uint chunk   = (blocks_per_chunk == 0u) ? 0u : block / blocks_per_chunk;
    uint bic     = (blocks_per_chunk == 0u) ? block : block % blocks_per_chunk;
    uint kv_blk_stride  = num_kv_heads * block_size * dim;
    uint kv_head_stride = block_size * dim;
    KVP vec = live ? (KVP)chunk_table[chunk] + bic * kv_blk_stride + kv_head * kv_head_stride + tok * dim
                   : (KVP)nullptr;
    const float off = live ? tq_offset<T>(offset_mode, offset_bias + kv_head * dim, cos_sin, rot_dim,
                                          pair_off, offset_mode == 2u ? positions[slot_i] : 0u,
                                          (slot_raw & 0x80000000u) != 0u, elem)
                           : 0.0f;

    shared[elem] = live ? (float)vec[elem] - off : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ns[elem] = shared[elem] * shared[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = dim / 2; stride > 0; stride >>= 1) {
        if (elem < stride) ns[elem] += ns[elem + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float vec_norm = sqrt(ns[0]);
    float safe_norm = max(vec_norm, 1e-8f);
    shared[elem] = live ? (shared[elem] / safe_norm) * signs[elem] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_wht_tg(shared, dim, elem);
    float scaled = shared[elem];
    uint idx = 0;
    if (live) for (uint b = 0; b < n_centroids - 1; b++) if (scaled > boundaries[b]) idx++;

    idx_shared[elem] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint store_base = (logical_slot * num_kv_heads + kv_head) * packed_dim;
    uint word_idx = elem / vals_per_word, pos_in_word = elem % vals_per_word;
    if (live && pos_in_word == 0 && word_idx < packed_dim) {
        uint word = 0;
        for (uint i = 0; i < vals_per_word && (word_idx * vals_per_word + i) < dim; i++)
            word |= (idx_shared[word_idx * vals_per_word + i] & ((1u << bits) - 1u)) << (i * bits);
        packed_store[store_base + word_idx] = word;
    }
    if (live && elem == 0) norms_store[logical_slot * num_kv_heads + kv_head] = vec_norm;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    shared[elem] = live ? centroids[idx] * scale : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_wht_tg(shared, dim, elem);
    if (live && do_writeback != 0u) vec[elem] = (T)(shared[elem] * scale * signs[elem] * vec_norm + off);
}

// Megakernel adapter: the runtime scalars the dispatch kernel reads from its `constant` bindings
// (Inline bindings, 7..23) are read from the same inline buffer by address.
template <typename T>
MK_FUNC void mk_tq_compress_paged(thread const MkStep& s, MkLane l, threadgroup uchar* tg) {
    #define TQ_U(i) (*(const device uint*)s.addr[i])
    threadgroup uchar* base = mk_region(s, l, tg);
    tq_compress_body<T, true, mk_ptr<T>, mk_ptr<uint>, mk_ptr<float>, const device uint&, const device float&>(
        (device const uint64_t*)s.addr[0], (device const uint*)s.addr[1], (device const float*)s.addr[2],
        (device const float*)s.addr[3], (device const float*)s.addr[4], (mk_ptr<uint>)s.addr[5],
        (mk_ptr<float>)s.addr[6], TQ_U(7), TQ_U(8), TQ_U(9), TQ_U(10), TQ_U(11),
        *(const device float*)s.addr[12], TQ_U(13), TQ_U(14), TQ_U(15), (device const uint*)s.addr[16],
        TQ_U(17), (device const T*)s.addr[18], (device const T*)s.addr[19], (device const uint*)s.addr[20],
        TQ_U(21), TQ_U(22), TQ_U(23), l.tg_pos, l.tid3, l.live,
        (threadgroup float*)base, (threadgroup float*)(base + 2048), (threadgroup uint*)(base + 4096));
    #undef TQ_U
}

#ifndef MK_BODIES_ONLY
template <typename T>
[[kernel]] void tq_compress_paged(
    device const uint64_t* chunk_table [[buffer(0)]],
    device const uint*  slots        [[buffer(1)]],
    device const float* signs        [[buffer(2)]],
    device const float* boundaries   [[buffer(3)]],
    device const float* centroids    [[buffer(4)]],
    device       uint*  packed_store [[buffer(5)]],
    device       float* norms_store  [[buffer(6)]],
    constant uint&  dim              [[buffer(7)]],
    constant uint&  bits             [[buffer(8)]],
    constant uint&  vals_per_word    [[buffer(9)]],
    constant uint&  packed_dim       [[buffer(10)]],
    constant uint&  n_centroids      [[buffer(11)]],
    constant float& scale            [[buffer(12)]],
    constant uint&  num_kv_heads     [[buffer(13)]],
    constant uint&  block_size       [[buffer(14)]],
    constant uint&  blocks_per_chunk [[buffer(15)]],
    device const uint*  logical_slots [[buffer(16)]],
    constant uint&  do_writeback     [[buffer(17)]],
    device const T*     offset_bias  [[buffer(18)]],
    device const T*     cos_sin      [[buffer(19)]],
    device const uint*  positions    [[buffer(20)]],
    constant uint&  offset_mode      [[buffer(21)]],
    constant uint&  rot_dim          [[buffer(22)]],
    constant uint&  pair_off         [[buffer(23)]],
    uint3 tg  [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    threadgroup float shared[512];
    threadgroup float ns[512];
    threadgroup uint idx_shared[512];
    tq_compress_body<T, false, device T*, device uint*, device float*, constant uint&, constant float&>(
        chunk_table, slots, signs, boundaries, centroids, packed_store, norms_store, dim, bits,
        vals_per_word, packed_dim, n_centroids, scale, num_kv_heads, block_size, blocks_per_chunk,
        logical_slots, do_writeback, offset_bias, cos_sin, positions, offset_mode, rot_dim, pair_off,
        tg, tid, true, shared, ns, idx_shared);
}

#define TQ_INSTANTIATE(fn, name, T)                                            \
    template [[host_name(name)]] [[kernel]] decltype(fn<T>) fn<T>;

TQ_INSTANTIATE(tq_compress_paged, "tq_compress_paged", half)
TQ_INSTANTIATE(tq_compress_paged, "tq_compress_paged_bf16", bfloat)
#else
// shared[512] + ns[512] float + idx_shared[512] uint per virtual threadgroup.
MK_ADAPTER(tq_compress_paged, 6144, (mk_tq_compress_paged<half>), MK_NO_CONSTS)
MK_ADAPTER(tq_compress_paged_bf16, 6144, (mk_tq_compress_paged<bfloat>), MK_NO_CONSTS)
#endif
