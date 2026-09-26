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
template <typename T>
[[kernel]] void tq_compress_paged(
    device const uint64_t* chunk_table [[buffer(0)]],  // per-(tensor) chunk-addr table
    device const uint*  slots        [[buffer(1)]],    // [n_slots] physical KV slots written this fwd
    device const float* signs        [[buffer(2)]],    // [dim]
    device const float* boundaries   [[buffer(3)]],    // [n_centroids-1]
    device const float* centroids    [[buffer(4)]],    // [n_centroids]
    device       uint*  packed_store [[buffer(5)]],    // [max_slots, num_kv_heads, packed_dim]
    device       float* norms_store  [[buffer(6)]],    // [max_slots, num_kv_heads]
    constant uint&  dim              [[buffer(7)]],
    constant uint&  bits             [[buffer(8)]],
    constant uint&  vals_per_word    [[buffer(9)]],
    constant uint&  packed_dim       [[buffer(10)]],
    constant uint&  n_centroids      [[buffer(11)]],
    constant float& scale            [[buffer(12)]],   // 1/sqrt(dim)
    constant uint&  num_kv_heads     [[buffer(13)]],
    constant uint&  block_size       [[buffer(14)]],
    constant uint&  blocks_per_chunk [[buffer(15)]],
    device const uint*  logical_slots [[buffer(16)]],  // [n_slots] LOGICAL slot for the packed-store index (= slots for in-place)
    constant uint&  do_writeback     [[buffer(17)]],   // 1 = lossy in-place dequant; 0 = leave raw K (gemma4 global runs pre-attention)
    device const T*     offset_bias  [[buffer(18)]],   // [num_kv_heads * dim] (offset_mode != 0)
    device const T*     cos_sin      [[buffer(19)]],   // [max_pos, rot_dim]   (offset_mode == 2)
    device const uint*  positions    [[buffer(20)]],   // [n_slots]            (offset_mode == 2)
    constant uint&  offset_mode      [[buffer(21)]],
    constant uint&  rot_dim          [[buffer(22)]],
    constant uint&  pair_off         [[buffer(23)]],
    uint3 tg  [[threadgroup_position_in_grid]],         // x=slot index, y=kv_head
    uint3 tid [[thread_position_in_threadgroup]])
{
    uint slot_i       = tg.x;
    uint kv_head      = tg.y;
    uint elem         = tid.x;
    // Spans: slot_mapping carries the stored-unrotated flag in bit 31, plus a
    // 0xFFFFFFFF padding sentinel. Skip padding (uniform across the threadgroup
    // — tg.x is shared, so no barrier divergence) and strip bit 31 before using
    // as an index, mirroring attention's ATTN_BT_MASK. The unrotated-ness is
    // handled by rope-on-read in attention; here it only selects the offset.
    uint slot_raw     = slots[slot_i];
    if (slot_raw == 0xFFFFFFFFu) return;                     // padding slot — no token
    uint slot         = slot_raw & 0x7FFFFFFFu;              // window slot: pool addressing
    uint logical_slot = logical_slots[slot_i] & 0x7FFFFFFFu; // logical slot: packed-store index
    uint block   = slot / block_size;
    uint tok     = slot % block_size;
    uint chunk   = (blocks_per_chunk == 0u) ? 0u : block / blocks_per_chunk;
    uint bic     = (blocks_per_chunk == 0u) ? block : block % blocks_per_chunk;
    uint kv_blk_stride  = num_kv_heads * block_size * dim;
    uint kv_head_stride = block_size * dim;
    device T* vec = (device T*)chunk_table[chunk]
        + bic * kv_blk_stride + kv_head * kv_head_stride + tok * dim;
    const float off = tq_offset<T>(offset_mode, offset_bias + kv_head * dim, cos_sin, rot_dim,
                                   pair_off, offset_mode == 2u ? positions[slot_i] : 0u,
                                   (slot_raw & 0x80000000u) != 0u, elem);

    // ── quantize the in-place vector, offset removed ──
    threadgroup float shared[512];
    shared[elem] = (float)vec[elem] - off;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float ns[512];
    ns[elem] = shared[elem] * shared[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = dim / 2; stride > 0; stride >>= 1) {
        if (elem < stride) ns[elem] += ns[elem + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float vec_norm = sqrt(ns[0]);
    float safe_norm = max(vec_norm, 1e-8f);
    shared[elem] = (shared[elem] / safe_norm) * signs[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_wht_tg(shared, dim, elem);
    float scaled = shared[elem];
    uint idx = 0;
    for (uint b = 0; b < n_centroids - 1; b++) if (scaled > boundaries[b]) idx++;

    // ── pack codes + norm into the packed store ──
    threadgroup uint idx_shared[512];
    idx_shared[elem] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Index the persistent store by the PHYSICAL slot (token cache position),
    // so codes persist per token across steps — not by the dispatch index.
    uint store_base = (logical_slot * num_kv_heads + kv_head) * packed_dim;
    uint word_idx = elem / vals_per_word, pos_in_word = elem % vals_per_word;
    if (pos_in_word == 0 && word_idx < packed_dim) {
        uint word = 0;
        for (uint i = 0; i < vals_per_word && (word_idx * vals_per_word + i) < dim; i++)
            word |= (idx_shared[word_idx * vals_per_word + i] & ((1u << bits) - 1u)) << (i * bits);
        packed_store[store_base + word_idx] = word;
    }
    if (elem == 0) norms_store[logical_slot * num_kv_heads + kv_head] = vec_norm;

    // ── dequant the codes back into the pool, offset restored ──
    threadgroup_barrier(mem_flags::mem_threadgroup);
    shared[elem] = centroids[idx] * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_wht_tg(shared, dim, elem);
    if (do_writeback != 0u) vec[elem] = (T)(shared[elem] * scale * signs[elem] * vec_norm + off);
}

#define TQ_INSTANTIATE(fn, name, T)                                            \
    template [[host_name(name)]] [[kernel]] decltype(fn<T>) fn<T>;

TQ_INSTANTIATE(tq_compress_paged, "tq_compress_paged", half)
TQ_INSTANTIATE(tq_compress_paged, "tq_compress_paged_bf16", bfloat)
