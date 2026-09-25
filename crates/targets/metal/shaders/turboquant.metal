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
//! 512, power of two).
//!
//! `tq_compress_paged[_bf16]`: one threadgroup per (new KV slot, kv_head) —
//! quantize the pool vector into packed uint32 codes + an f32 norm in the
//! packed store, optionally writing the lossy dequant back into the pool.
//!
//! `tq_dequant_blocktable[_bf16]`: one threadgroup per (block, kv_head, seq) —
//! dequant the active context from the packed store into the fp16/bf16 scratch.

#include <metal_stdlib>
using namespace metal;

// tq_compress_paged: the production wiring kernel. For each (new KV slot,
// kv_head), read the fp16 vector IN PLACE from the paged pool (chunk-table
// addressing, identical to the attention kernels), quantize it to packed 3-bit
// codes + f32 norm written to the PACKED STORE (the canonical 4.6x-smaller
// cache), then dequant and write the lossy fp16 back into the pool so the
// existing attention reads TurboQuant'd KV (dequant-to-buffer; the pool IS the
// buffer). One threadgroup per (slot, kv_head); dim threads. Dispatched
// post-forward over the new slots of one layer's K (and again for V).
kernel void tq_compress_paged(
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
    // handled by rope-on-read in attention; here we just need the real cache
    // index of whatever K rope_append left in the pool.
    uint slot_raw     = slots[slot_i];
    if (slot_raw == 0xFFFFFFFFu) return;                     // padding slot — no token
    uint slot         = slot_raw & 0x7FFFFFFFu;              // window slot: fp16 pool addressing
    uint logical_slot = logical_slots[slot_i] & 0x7FFFFFFFu; // logical slot: packed-store index
    uint block   = slot / block_size;
    uint tok     = slot % block_size;
    uint chunk   = (blocks_per_chunk == 0u) ? 0u : block / blocks_per_chunk;
    uint bic     = (blocks_per_chunk == 0u) ? block : block % blocks_per_chunk;
    uint kv_blk_stride  = num_kv_heads * block_size * dim;
    uint kv_head_stride = block_size * dim;
    device half* vec = (device half*)chunk_table[chunk]
        + bic * kv_blk_stride + kv_head * kv_head_stride + tok * dim;

    // ── quantize the in-place fp16 vector ──
    threadgroup float shared[512];
    shared[elem] = (float)vec[elem];
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
    uint h = 1;
    while (h < dim) {
        uint blk = elem / (2 * h), off = elem % (2 * h);
        if (off < h) { uint j = blk * 2 * h + off; float a = shared[j], b = shared[j + h]; shared[j] = a + b; shared[j + h] = a - b; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }
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

    // ── dequant the codes back into the pool (lossy fp16 attention reads) ──
    threadgroup_barrier(mem_flags::mem_threadgroup);
    shared[elem] = centroids[idx] * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    h = 1;
    while (h < dim) {
        uint blk = elem / (2 * h), off = elem % (2 * h);
        if (off < h) { uint j = blk * 2 * h + off; float a = shared[j], b = shared[j + h]; shared[j] = a + b; shared[j + h] = a - b; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }
    if (do_writeback != 0u) vec[elem] = (half)(shared[elem] * scale * signs[elem] * vec_norm);
}

// bf16 twin of tq_compress_paged (the pool is bf16 for Llama/Qwen).
kernel void tq_compress_paged_bf16(
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
    // handled by rope-on-read in attention; here we just need the real cache
    // index of whatever K rope_append left in the pool.
    uint slot_raw     = slots[slot_i];
    if (slot_raw == 0xFFFFFFFFu) return;                     // padding slot — no token
    uint slot         = slot_raw & 0x7FFFFFFFu;              // window slot: fp16 pool addressing
    uint logical_slot = logical_slots[slot_i] & 0x7FFFFFFFu; // logical slot: packed-store index
    uint block   = slot / block_size;
    uint tok     = slot % block_size;
    uint chunk   = (blocks_per_chunk == 0u) ? 0u : block / blocks_per_chunk;
    uint bic     = (blocks_per_chunk == 0u) ? block : block % blocks_per_chunk;
    uint kv_blk_stride  = num_kv_heads * block_size * dim;
    uint kv_head_stride = block_size * dim;
    device bfloat* vec = (device bfloat*)chunk_table[chunk]
        + bic * kv_blk_stride + kv_head * kv_head_stride + tok * dim;

    // ── quantize the in-place fp16 vector ──
    threadgroup float shared[512];
    shared[elem] = (float)vec[elem];
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
    uint h = 1;
    while (h < dim) {
        uint blk = elem / (2 * h), off = elem % (2 * h);
        if (off < h) { uint j = blk * 2 * h + off; float a = shared[j], b = shared[j + h]; shared[j] = a + b; shared[j + h] = a - b; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }
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

    // ── dequant the codes back into the pool (lossy fp16 attention reads) ──
    threadgroup_barrier(mem_flags::mem_threadgroup);
    shared[elem] = centroids[idx] * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    h = 1;
    while (h < dim) {
        uint blk = elem / (2 * h), off = elem % (2 * h);
        if (off < h) { uint j = blk * 2 * h + off; float a = shared[j], b = shared[j + h]; shared[j] = a + b; shared[j + h] = a - b; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }
    if (do_writeback != 0u) vec[elem] = (bfloat)(shared[elem] * scale * signs[elem] * vec_norm);
}


// tq_dequant_blocktable: the FULL-CONTEXT dequant for the pool-capacity design.
// Block-table-driven (like the paged attention kernel) so a STATIC grid can
// cover a runtime-varying context: grid = (max_blocks, num_kv_heads, num_seqs);
// each threadgroup dequants ONE active block's tokens from the packed store
// into the fp16 scratch (same physical slots), early-exiting for padding seqs /
// blocks past seqused_k. dim threads; loops block_size tokens per block.
kernel void tq_dequant_blocktable(
    device const uint64_t* chunk_table [[buffer(0)]],
    device const uint*  block_table  [[buffer(1)]],
    device const uint*  seqused_k    [[buffer(2)]],
    device const float* signs        [[buffer(3)]],
    device const float* centroids    [[buffer(4)]],
    device const uint*  packed_store [[buffer(5)]],
    device const float* norms_store  [[buffer(6)]],
    constant uint&  dim              [[buffer(7)]],
    constant uint&  bits             [[buffer(8)]],
    constant uint&  vals_per_word    [[buffer(9)]],
    constant uint&  packed_dim       [[buffer(10)]],
    constant float& scale            [[buffer(11)]],
    constant uint&  num_kv_heads     [[buffer(12)]],
    constant uint&  block_size       [[buffer(13)]],
    constant uint&  blocks_per_chunk [[buffer(14)]],
    uint3 tg  [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    uint block_in_seq = tg.x;
    uint kv_head      = tg.y;
    uint seq          = tg.z;
    uint elem         = tid.x;
    uint max_blocks   = tpg.x;  // grid.x = block_table row width (worker-dispatched; grid.z = num_seqs)
    uint used = seqused_k[seq];
    uint tok_base = block_in_seq * block_size;
    if (tok_base >= used) return;
    // Spans: block_table carries the stored-unrotated flag in bit 31; strip it
    // before pool/packed indexing (mirrors attention's ATTN_BT_MASK). re-rope
    // is handled in attention, not here.
    uint phys_block = block_table[seq * max_blocks + block_in_seq] & 0x7FFFFFFFu;
    uint n_tok = min(block_size, used - tok_base);

    uint kv_blk_stride  = num_kv_heads * block_size * dim;
    uint kv_head_stride = block_size * dim;
    uint chunk = (blocks_per_chunk == 0u) ? 0u : phys_block / blocks_per_chunk;
    uint bic   = (blocks_per_chunk == 0u) ? phys_block : phys_block % blocks_per_chunk;
    device half* vecbase = (device half*)chunk_table[chunk]
        + bic * kv_blk_stride + kv_head * kv_head_stride;

    uint bit_mask = (1u << bits) - 1u;
    uint word_idx = elem / vals_per_word, pos_in_word = elem % vals_per_word;
    threadgroup float shared[512];
    for (uint t = 0; t < n_tok; t++) {
        uint phys_slot = phys_block * block_size + t;
        uint store_base = (phys_slot * num_kv_heads + kv_head) * packed_dim;
        uint word = packed_store[store_base + word_idx];
        uint idx = (word >> (pos_in_word * bits)) & bit_mask;
        float vec_norm = norms_store[phys_slot * num_kv_heads + kv_head];
        // SKIP un-quantized slots. The per-layer dequant runs BEFORE rope and
        // covers the whole active context, including the CURRENT chunk's brand-
        // new tokens — which are NOT yet in the packed store, so their norm is
        // still 0 (zero-filled at provision). Writing here (even a 0) would
        // RACE/overwrite rope's real new-token K on a multi-chunk (continuation)
        // prefill -> lost context + Inf from the compress's 0/0 normalization ->
        // garbage output past ~4096 tokens. Leave these slots untouched; rope
        // owns them. `continue` is threadgroup-uniform (vec_norm is per-token,
        // identical across the `elem` threads) so the WHT barriers stay
        // balanced. A non-finite norm (never-written slot) is skipped too.
        if (vec_norm == 0.0f || !isfinite(vec_norm)) { continue; }
        shared[elem] = centroids[idx] * scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint h = 1;
        while (h < dim) {
            uint blk = elem / (2 * h), off = elem % (2 * h);
            if (off < h) { uint j = blk * 2 * h + off; float a = shared[j], b = shared[j + h]; shared[j] = a + b; shared[j + h] = a - b; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            h *= 2;
        }
        vecbase[t * dim + elem] = (half)(shared[elem] * scale * signs[elem] * vec_norm);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// bf16 twin.
kernel void tq_dequant_blocktable_bf16(
    device const uint64_t* chunk_table [[buffer(0)]],
    device const uint*  block_table  [[buffer(1)]],
    device const uint*  seqused_k    [[buffer(2)]],
    device const float* signs        [[buffer(3)]],
    device const float* centroids    [[buffer(4)]],
    device const uint*  packed_store [[buffer(5)]],
    device const float* norms_store  [[buffer(6)]],
    constant uint&  dim              [[buffer(7)]],
    constant uint&  bits             [[buffer(8)]],
    constant uint&  vals_per_word    [[buffer(9)]],
    constant uint&  packed_dim       [[buffer(10)]],
    constant float& scale            [[buffer(11)]],
    constant uint&  num_kv_heads     [[buffer(12)]],
    constant uint&  block_size       [[buffer(13)]],
    constant uint&  blocks_per_chunk [[buffer(14)]],
    uint3 tg  [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    uint block_in_seq = tg.x;
    uint kv_head      = tg.y;
    uint seq          = tg.z;
    uint elem         = tid.x;
    uint max_blocks   = tpg.x;  // grid.x = block_table row width (worker-dispatched; grid.z = num_seqs)
    uint used = seqused_k[seq];
    uint tok_base = block_in_seq * block_size;
    if (tok_base >= used) return;
    // Spans: block_table carries the stored-unrotated flag in bit 31; strip it
    // before pool/packed indexing (mirrors attention's ATTN_BT_MASK). re-rope
    // is handled in attention, not here.
    uint phys_block = block_table[seq * max_blocks + block_in_seq] & 0x7FFFFFFFu;
    uint n_tok = min(block_size, used - tok_base);

    uint kv_blk_stride  = num_kv_heads * block_size * dim;
    uint kv_head_stride = block_size * dim;
    uint chunk = (blocks_per_chunk == 0u) ? 0u : phys_block / blocks_per_chunk;
    uint bic   = (blocks_per_chunk == 0u) ? phys_block : phys_block % blocks_per_chunk;
    device bfloat* vecbase = (device bfloat*)chunk_table[chunk]
        + bic * kv_blk_stride + kv_head * kv_head_stride;

    uint bit_mask = (1u << bits) - 1u;
    uint word_idx = elem / vals_per_word, pos_in_word = elem % vals_per_word;
    threadgroup float shared[512];
    for (uint t = 0; t < n_tok; t++) {
        uint phys_slot = phys_block * block_size + t;
        uint store_base = (phys_slot * num_kv_heads + kv_head) * packed_dim;
        uint word = packed_store[store_base + word_idx];
        uint idx = (word >> (pos_in_word * bits)) & bit_mask;
        float vec_norm = norms_store[phys_slot * num_kv_heads + kv_head];
        // SKIP un-quantized slots. The per-layer dequant runs BEFORE rope and
        // covers the whole active context, including the CURRENT chunk's brand-
        // new tokens — which are NOT yet in the packed store, so their norm is
        // still 0 (zero-filled at provision). Writing here (even a 0) would
        // RACE/overwrite rope's real new-token K on a multi-chunk (continuation)
        // prefill -> lost context + Inf from the compress's 0/0 normalization ->
        // garbage output past ~4096 tokens. Leave these slots untouched; rope
        // owns them. `continue` is threadgroup-uniform (vec_norm is per-token,
        // identical across the `elem` threads) so the WHT barriers stay
        // balanced. A non-finite norm (never-written slot) is skipped too.
        if (vec_norm == 0.0f || !isfinite(vec_norm)) { continue; }
        shared[elem] = centroids[idx] * scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint h = 1;
        while (h < dim) {
            uint blk = elem / (2 * h), off = elem % (2 * h);
            if (off < h) { uint j = blk * 2 * h + off; float a = shared[j], b = shared[j + h]; shared[j] = a + b; shared[j + h] = a - b; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            h *= 2;
        }
        vecbase[t * dim + elem] = (bfloat)(shared[elem] * scale * signs[elem] * vec_norm);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
