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
#include "baked.h"
#include "turboquant_offset.h"
using namespace metal;

// `TqCompressConstants`, compiled in.
SCRATCHY_CONSTANT(uint,  TQ_DIM,              0);
SCRATCHY_CONSTANT(uint,  TQ_BITS,             1);
SCRATCHY_CONSTANT(uint,  TQ_VALS_PER_WORD,    2);
SCRATCHY_CONSTANT(uint,  TQ_PACKED_DIM,       3);
SCRATCHY_CONSTANT(uint,  TQ_CENTROIDS,        4);
SCRATCHY_CONSTANT(float, TQ_SCALE,            5);  // 1/sqrt(TQ_DIM)
SCRATCHY_CONSTANT(uint,  TQ_NUM_KV_HEADS,     6);
SCRATCHY_CONSTANT(uint,  TQ_BLOCK_SIZE,       7);
SCRATCHY_CONSTANT(uint,  TQ_BLOCKS_PER_CHUNK, 8);
// 1 = lossy in-place dequant; 0 = leave the raw vector (attention reads the step's new keys raw).
SCRATCHY_CONSTANT(uint,  TQ_WRITEBACK,        9);
SCRATCHY_CONSTANT(uint,  TQ_OFFSET_MODE,      10);
SCRATCHY_CONSTANT(uint,  TQ_ROT_DIM,          11);
SCRATCHY_CONSTANT(uint,  TQ_PAIR_OFF,         12);

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
    device const float* signs        [[buffer(2)]],    // [TQ_DIM]
    device const float* boundaries   [[buffer(3)]],    // [TQ_CENTROIDS-1]
    device const float* centroids    [[buffer(4)]],    // [TQ_CENTROIDS]
    device       uint*  packed_store [[buffer(5)]],    // [max_slots, TQ_NUM_KV_HEADS, TQ_PACKED_DIM]
    device       float* norms_store  [[buffer(6)]],    // [max_slots, TQ_NUM_KV_HEADS]
    device const uint*  logical_slots [[buffer(16)]],  // [n_slots] LOGICAL slot for the packed-store index (= slots for in-place)
    device const T*     offset_bias  [[buffer(18)]],   // [TQ_NUM_KV_HEADS * TQ_DIM] (TQ_OFFSET_MODE != 0)
    device const T*     cos_sin      [[buffer(19)]],   // [max_pos, TQ_ROT_DIM]      (TQ_OFFSET_MODE == 2)
    device const uint*  positions    [[buffer(20)]],   // [n_slots]                  (TQ_OFFSET_MODE == 2)
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
    uint block   = slot / TQ_BLOCK_SIZE;
    uint tok     = slot % TQ_BLOCK_SIZE;
    uint chunk   = (TQ_BLOCKS_PER_CHUNK == 0u) ? 0u : block / TQ_BLOCKS_PER_CHUNK;
    uint bic     = (TQ_BLOCKS_PER_CHUNK == 0u) ? block : block % TQ_BLOCKS_PER_CHUNK;
    uint kv_blk_stride  = TQ_NUM_KV_HEADS * TQ_BLOCK_SIZE * TQ_DIM;
    uint kv_head_stride = TQ_BLOCK_SIZE * TQ_DIM;
    device T* vec = (device T*)chunk_table[chunk]
        + bic * kv_blk_stride + kv_head * kv_head_stride + tok * TQ_DIM;
    const float off = tq_offset<T>(TQ_OFFSET_MODE, offset_bias + kv_head * TQ_DIM, cos_sin, TQ_ROT_DIM,
                                   TQ_PAIR_OFF, TQ_OFFSET_MODE == 2u ? positions[slot_i] : 0u,
                                   (slot_raw & 0x80000000u) != 0u, elem);

    // ── quantize the in-place vector, offset removed ──
    threadgroup float shared[512];
    shared[elem] = (float)vec[elem] - off;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float ns[512];
    ns[elem] = shared[elem] * shared[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = TQ_DIM / 2; stride > 0; stride >>= 1) {
        if (elem < stride) ns[elem] += ns[elem + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float vec_norm = sqrt(ns[0]);
    float safe_norm = max(vec_norm, 1e-8f);
    shared[elem] = (shared[elem] / safe_norm) * signs[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tq_wht_tg(shared, TQ_DIM, elem);
    float scaled = shared[elem];
    uint idx = 0;
    for (uint b = 0; b < TQ_CENTROIDS - 1; b++) if (scaled > boundaries[b]) idx++;

    // ── pack codes + norm into the packed store ──
    threadgroup uint idx_shared[512];
    idx_shared[elem] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Index the persistent store by the PHYSICAL slot (token cache position),
    // so codes persist per token across steps — not by the dispatch index.
    uint store_base = (logical_slot * TQ_NUM_KV_HEADS + kv_head) * TQ_PACKED_DIM;
    uint word_idx = elem / TQ_VALS_PER_WORD, pos_in_word = elem % TQ_VALS_PER_WORD;
    if (pos_in_word == 0 && word_idx < TQ_PACKED_DIM) {
        uint word = 0;
        for (uint i = 0; i < TQ_VALS_PER_WORD && (word_idx * TQ_VALS_PER_WORD + i) < TQ_DIM; i++)
            word |= (idx_shared[word_idx * TQ_VALS_PER_WORD + i] & ((1u << TQ_BITS) - 1u)) << (i * TQ_BITS);
        packed_store[store_base + word_idx] = word;
    }
    if (elem == 0) norms_store[logical_slot * TQ_NUM_KV_HEADS + kv_head] = vec_norm;

    // ── dequant the codes back into the pool, offset restored ──
    if (TQ_WRITEBACK != 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[elem] = centroids[idx] * TQ_SCALE;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tq_wht_tg(shared, TQ_DIM, elem);
        vec[elem] = (T)(shared[elem] * TQ_SCALE * signs[elem] * vec_norm + off);
    }
}

SCRATCHY_KERNEL(tq_compress_paged, tq_compress_paged<half>)
SCRATCHY_KERNEL(tq_compress_paged_bf16, tq_compress_paged<bfloat>)
