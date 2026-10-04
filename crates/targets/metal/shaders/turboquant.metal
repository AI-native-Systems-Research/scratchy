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
//! TurboQuant's standalone paged-KV encode. The encode itself is
//! `turboquant_encode.h`'s, which the tape's KV writers run as they write
//! (rope.metal, `MetalFusion::KvEncoded`). `dim` threads per threadgroup (dim <=
//! 512, power of two), one template over the cache element type `T`,
//! instantiated as `<name>` (half) and `<name>_bf16` (bfloat).
//!
//! `tq_compress_paged[_bf16]`: one threadgroup per (KV slot, kv_head) —
//! quantize the pool vector into packed uint32 codes + an f32 norm in the
//! packed store, optionally writing the lossy dequant back into the pool. No
//! tape binds it: the kernel tests fill packed stores with it, and hold the
//! writers' encode to it bit for bit.
//!
//! Attention reads the packed store itself (attention.metal): decode through
//! `attention_via_cache_v2`'s TurboQuant mode, prefill through the rotated-
//! domain image `tq_stage_rotated` stages.
//!
//! OFFSET. The compress codes each vector MINUS its additive offset
//! (`turboquant_offset.h`); every reader restores it exactly.

#include <metal_stdlib>
#include "baked.h"
#include "turboquant_encode.h"
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

// tq_compress_paged: for each (KV slot, kv_head), read the vector IN PLACE from
// the paged pool (chunk-table addressing, identical to the attention kernels),
// remove its offset, quantize it to packed codes + f32 norm written to the
// PACKED STORE (the canonical ~4.6x-smaller cache), then (TQ_WRITEBACK) dequant,
// restore the offset and write the lossy vector back into the pool. One
// threadgroup per (slot, kv_head); dim threads.
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
    // Index the persistent store by the PHYSICAL slot (token cache position),
    // so codes persist per token across steps — not by the dispatch index.
    threadgroup float scratch[tq_encode_floats(1, TQ_DIM)];
    threadgroup uint codes[TQ_DIM];
    const uint store = logical_slot * TQ_NUM_KV_HEADS + kv_head;
    const TqEncoded e = tq_encode<1>({(float)vec[elem] - off}, elem, TQ_DIM, TQ_BITS, TQ_VALS_PER_WORD,
                                     TQ_PACKED_DIM, TQ_CENTROIDS, signs, boundaries,
                                     {packed_store + store * TQ_PACKED_DIM}, {norms_store + store},
                                     scratch, codes)[0];

    // ── dequant the codes back into the pool, offset restored ──
    if (TQ_WRITEBACK != 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scratch[elem] = centroids[e.code] * TQ_SCALE;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tq_wht_tg(scratch, TQ_DIM, elem);
        vec[elem] = (T)(scratch[elem] * TQ_SCALE * signs[elem] * e.norm + off);
    }
}

SCRATCHY_KERNEL(tq_compress_paged, tq_compress_paged<half>)
SCRATCHY_KERNEL(tq_compress_paged_bf16, tq_compress_paged<bfloat>)
