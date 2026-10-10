// SPDX-License-Identifier: Apache-2.0
//! The paged-cache writer of an attention with no positional rotation
//! (Nemotron-H; `MetalStep::KvAppend`): K and V go to the per-layer KV
//! cache exactly as projected. The rope writer (`rope.metal`) minus the
//! rotation — same cache layout, chunk addressing and padding sentinel.
//!
//! Constants (compiled in; `KvAppendConstants`):
//!   0 = HEAD_DIM, 1 = NUM_KV_HEADS, 2 = BLOCK_SIZE (paged KV page size),
//!   3 = BLOCKS_PER_CHUNK (reactive KV pool: buffers 3/4 are per-layer
//!       chunk-address TABLES; a physical block derefs
//!       `table[block / BLOCKS_PER_CHUNK]`, then `block % BLOCKS_PER_CHUNK`).
//!
//! Bindings:
//!   buffer(0) = k            [bucket_m, NUM_KV_HEADS * HEAD_DIM]
//!   buffer(1) = v            [bucket_m, NUM_KV_HEADS * HEAD_DIM]
//!   buffer(2) = slot_mapping [bucket_m] — global cache slot per token
//!   buffer(3) = kv_cache_k   chunk table of [blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
//!   buffer(4) = kv_cache_v   same shape as kv_cache_k
//!
//! Dispatch: threadgroups (bucket_m, NUM_KV_HEADS, 1), threads (HEAD_DIM, 1, 1);
//! thread `d` of (token, kv_head) copies element `d` of K and of V.

#include <metal_stdlib>
#include "baked.h"
using namespace metal;

SCRATCHY_CONSTANT(uint, KVA_HEAD_DIM, 0);
SCRATCHY_CONSTANT(uint, KVA_NUM_KV_HEADS, 1);
SCRATCHY_CONSTANT(uint, KVA_BLOCK_SIZE, 2);
SCRATCHY_CONSTANT(uint, KVA_BLOCKS_PER_CHUNK, 3);

template <typename T>
[[kernel]] void kv_append(
    device const T* k                 [[buffer(0)]],
    device const T* v                 [[buffer(1)]],
    device const uint* slot_mapping   [[buffer(2)]],
    device const uint64_t* kv_cache_k [[buffer(3)]],
    device const uint64_t* kv_cache_v [[buffer(4)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid    [[thread_position_in_threadgroup]])
{
    const uint t       = tg_pos.x;
    const uint kv_head = tg_pos.y;
    const uint d       = tid.x;
    if (kv_head >= KVA_NUM_KV_HEADS || d >= KVA_HEAD_DIM) return;

    // Sentinel `0xFFFFFFFF` marks padding lanes: no cache write, so a
    // padding row's projection cannot land in slot 0. Bit 31 (the spans
    // "stored unrotated" flag) is stripped before addressing — an
    // un-rotated K is stored the same either way.
    const uint slot_raw = slot_mapping[t];
    if (slot_raw == 0xFFFFFFFFu) return;
    const uint slot = slot_raw & 0x7FFFFFFFu;
    const uint block_id     = slot / KVA_BLOCK_SIZE;
    const uint block_offset = slot % KVA_BLOCK_SIZE;
    const uint chunk        = block_id / KVA_BLOCKS_PER_CHUNK;
    const uint blk_in_chunk = block_id % KVA_BLOCKS_PER_CHUNK;
    const uint dst = blk_in_chunk * (KVA_NUM_KV_HEADS * KVA_BLOCK_SIZE * KVA_HEAD_DIM)
                   + kv_head * (KVA_BLOCK_SIZE * KVA_HEAD_DIM)
                   + block_offset * KVA_HEAD_DIM
                   + d;
    const uint src = t * (KVA_NUM_KV_HEADS * KVA_HEAD_DIM) + kv_head * KVA_HEAD_DIM + d;
    ((device T*)kv_cache_k[chunk])[dst] = k[src];
    ((device T*)kv_cache_v[chunk])[dst] = v[src];
}

SCRATCHY_KERNEL(kv_append_f16, kv_append<half>)
SCRATCHY_KERNEL(kv_append_bf16, kv_append<bfloat>)
