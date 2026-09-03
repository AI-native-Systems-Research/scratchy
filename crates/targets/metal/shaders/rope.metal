// SPDX-License-Identifier: Apache-2.0
//! Rotary Position Embedding (RoPE) Metal shaders
//!
//! Implements both NeoX-style and GPT-J-style rotary embeddings:
//! - NeoX: pairs element i with i + half_dim
//! - GPT-J (interleaved): pairs element 2i with 2i+1
//!
//! Algorithm:
//! For each pair (x, y):
//!   x' = x * cos(θ) - y * sin(θ)
//!   y' = y * cos(θ) + x * sin(θ)
//!
//! Where θ is position-dependent and pre-computed in cos_sin_cache.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// NeoX-style RoPE (standard Llama, GPT-NeoX)
// ---------------------------------------------------------------------------

/// Apply rotary embedding to a single token's query/key vectors.
/// NeoX style: pairs element i with i + half_dim.
///
/// @param query: [num_heads, head_size] - query vector for this token
/// @param key: [num_kv_heads, head_size] - key vector for this token (nullable)
/// @param cos_sin_cache: [rot_dim] - concatenated [cos; sin] for this position
/// @param num_heads: number of query heads
/// @param num_kv_heads: number of key heads
/// @param rot_dim: rotary dimension (typically head_size or head_size/2)
/// @param head_size: size of each head
kernel void rope_neox_f16(
    device half* query [[buffer(0)]],
    device half* key [[buffer(1)]],
    constant half* cos_sin_cache [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& num_kv_heads [[buffer(4)]],
    constant uint& rot_dim [[buffer(5)]],
    constant uint& head_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint embed_dim = rot_dim / 2;
    constant half* cos_ptr = cos_sin_cache;
    constant half* sin_ptr = cos_sin_cache + embed_dim;
    
    // Apply to query heads
    const uint nq = num_heads * embed_dim;
    if (tid < nq) {
        const uint head_idx = tid / embed_dim;
        const uint rot_offset = tid % embed_dim;
        
        const uint x_index = rot_offset;
        const uint y_index = embed_dim + rot_offset;
        
        const half cos_val = cos_ptr[x_index];
        const half sin_val = sin_ptr[x_index];
        
        device half* head_ptr = query + head_idx * head_size;
        const half x = head_ptr[x_index];
        const half y = head_ptr[y_index];
        
        head_ptr[x_index] = x * cos_val - y * sin_val;
        head_ptr[y_index] = y * cos_val + x * sin_val;
    }
    
    // Apply to key heads (if present)
    if (key != nullptr) {
        const uint nk = num_kv_heads * embed_dim;
        if (tid < nk) {
            const uint head_idx = tid / embed_dim;
            const uint rot_offset = tid % embed_dim;
            
            const uint x_index = rot_offset;
            const uint y_index = embed_dim + rot_offset;
            
            const half cos_val = cos_ptr[x_index];
            const half sin_val = sin_ptr[x_index];
            
            device half* head_ptr = key + head_idx * head_size;
            const half x = head_ptr[x_index];
            const half y = head_ptr[y_index];
            
            head_ptr[x_index] = x * cos_val - y * sin_val;
            head_ptr[y_index] = y * cos_val + x * sin_val;
        }
    }
}

/// BFloat16 variant of NeoX-style RoPE
kernel void rope_neox_bf16(
    device bfloat* query [[buffer(0)]],
    device bfloat* key [[buffer(1)]],
    constant bfloat* cos_sin_cache [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& num_kv_heads [[buffer(4)]],
    constant uint& rot_dim [[buffer(5)]],
    constant uint& head_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint embed_dim = rot_dim / 2;
    constant bfloat* cos_ptr = cos_sin_cache;
    constant bfloat* sin_ptr = cos_sin_cache + embed_dim;
    
    // Apply to query heads
    const uint nq = num_heads * embed_dim;
    if (tid < nq) {
        const uint head_idx = tid / embed_dim;
        const uint rot_offset = tid % embed_dim;
        
        const uint x_index = rot_offset;
        const uint y_index = embed_dim + rot_offset;
        
        const bfloat cos_val = cos_ptr[x_index];
        const bfloat sin_val = sin_ptr[x_index];
        
        device bfloat* head_ptr = query + head_idx * head_size;
        const bfloat x = head_ptr[x_index];
        const bfloat y = head_ptr[y_index];
        
        head_ptr[x_index] = x * cos_val - y * sin_val;
        head_ptr[y_index] = y * cos_val + x * sin_val;
    }
    
    // Apply to key heads (if present)
    if (key != nullptr) {
        const uint nk = num_kv_heads * embed_dim;
        if (tid < nk) {
            const uint head_idx = tid / embed_dim;
            const uint rot_offset = tid % embed_dim;
            
            const uint x_index = rot_offset;
            const uint y_index = embed_dim + rot_offset;
            
            const bfloat cos_val = cos_ptr[x_index];
            const bfloat sin_val = sin_ptr[x_index];
            
            device bfloat* head_ptr = key + head_idx * head_size;
            const bfloat x = head_ptr[x_index];
            const bfloat y = head_ptr[y_index];
            
            head_ptr[x_index] = x * cos_val - y * sin_val;
            head_ptr[y_index] = y * cos_val + x * sin_val;
        }
    }
}

// ---------------------------------------------------------------------------
// Interleaved RoPE (GPT-J style, Cohere CommandR)
// ---------------------------------------------------------------------------

/// Apply rotary embedding with interleaved pairing.
/// GPT-J style: pairs element 2i with 2i+1.
///
/// Used by Cohere's CommandR family.
kernel void rope_interleaved_f16(
    device half* query [[buffer(0)]],
    device half* key [[buffer(1)]],
    constant half* cos_sin_cache [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& num_kv_heads [[buffer(4)]],
    constant uint& rot_dim [[buffer(5)]],
    constant uint& head_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint embed_dim = rot_dim / 2;
    constant half* cos_ptr = cos_sin_cache;
    constant half* sin_ptr = cos_sin_cache + embed_dim;
    
    // Apply to query heads
    const uint nq = num_heads * embed_dim;
    if (tid < nq) {
        const uint head_idx = tid / embed_dim;
        const uint rot_offset = tid % embed_dim;
        
        const uint x_index = 2 * rot_offset;
        const uint y_index = 2 * rot_offset + 1;
        
        const half cos_val = cos_ptr[rot_offset];
        const half sin_val = sin_ptr[rot_offset];
        
        device half* head_ptr = query + head_idx * head_size;
        const half x = head_ptr[x_index];
        const half y = head_ptr[y_index];
        
        head_ptr[x_index] = x * cos_val - y * sin_val;
        head_ptr[y_index] = y * cos_val + x * sin_val;
    }
    
    // Apply to key heads (if present)
    if (key != nullptr) {
        const uint nk = num_kv_heads * embed_dim;
        if (tid < nk) {
            const uint head_idx = tid / embed_dim;
            const uint rot_offset = tid % embed_dim;
            
            const uint x_index = 2 * rot_offset;
            const uint y_index = 2 * rot_offset + 1;
            
            const half cos_val = cos_ptr[rot_offset];
            const half sin_val = sin_ptr[rot_offset];
            
            device half* head_ptr = key + head_idx * head_size;
            const half x = head_ptr[x_index];
            const half y = head_ptr[y_index];
            
            head_ptr[x_index] = x * cos_val - y * sin_val;
            head_ptr[y_index] = y * cos_val + x * sin_val;
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 5.G.3: rope_append_f16_specialized — paged-cache RoPE writer
//
// In-place NeoX-style RoPE on Q and K, plus paged write of (rotated K,
// un-rotated V) into the per-layer KV cache. Mirrors
// `Instruction::RopeAppend` (CUDA) / `KernelId::RopeAppend` (Metal).
//
// Function constants (must match
// `scratchy-target-metal::interpreter::metal::pipelines::constants_for`):
//   0 = HEAD_DIM
//   1 = NUM_Q_HEADS
//   2 = NUM_KV_HEADS
//   3 = ROT_DIM     (typically == HEAD_DIM; partial-rope models pass < HEAD_DIM)
//   4 = BLOCK_SIZE  (paged KV cache page size)
//
// Bindings (must match `interpreter::metal::lowering::lower_one` for
// `Instruction::RopeAppend`):
//   buffer(0) = q_inout       [bucket_m, NUM_Q_HEADS  * HEAD_DIM]   in/out
//   buffer(1) = k_inout       [bucket_m, NUM_KV_HEADS * HEAD_DIM]   in/out
//   buffer(2) = v_inout       [bucket_m, NUM_KV_HEADS * HEAD_DIM]   in/out (un-rotated; cache copy only)
//   buffer(3) = cos_sin       [max_pos, ROT_DIM] — row = [cos[half] | sin[half]]
//   buffer(4) = positions     [bucket_m]
//   buffer(5) = slot_mapping  [bucket_m] — global cache slot per token
//   buffer(6) = kv_cache_k    [num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
//   buffer(7) = kv_cache_v    same shape as kv_cache_k
//
// Dispatch (set by `interpreter::metal::lowering::lower_one`):
//   threadgroups: (bucket_m, NUM_Q_HEADS, 1)
//   threads_per_threadgroup: (HEAD_DIM, 1, 1)
//
// Per (token, q_head) threadgroup:
//   - Threads with `d < ROT_DIM/2` rotate the (d, d+half) pair of
//     `q_inout[token, q_head, :]`.
//   - Threadgroups whose q_head owns a kv_head (i.e. `q_head %
//     group_ratio == 0` where `group_ratio = NUM_Q_HEADS / NUM_KV_HEADS`)
//     additionally:
//       a. Rotate `k_inout[token, kv_head, :]` (same pair shape).
//       b. Copy the rotated K and un-rotated V into the cache slot.
//   - Other q_heads do Q only.
//
// No threadgroup_barrier needed: each thread reads-then-writes its own
// (d, d+half) pair before any other thread touches the same indices,
// and the K/V paged write happens after the K rotation in the same
// thread (sequential dependency).
// ---------------------------------------------------------------------------

constant uint ROPE_HEAD_DIM     [[function_constant(0)]];
constant uint ROPE_NUM_Q_HEADS  [[function_constant(1)]];
constant uint ROPE_NUM_KV_HEADS [[function_constant(2)]];
constant uint ROPE_ROT_DIM      [[function_constant(3)]];
constant uint ROPE_BLOCK_SIZE   [[function_constant(4)]];
// Reactive (chunked) KV pool: buffers 6/7 are per-layer chunk-address
// TABLES (device uint64 gpuAddresses), not the cache buffers. A
// physical block id derefs `table[block_id / BLOCKS_PER_CHUNK]` then
// addresses with `block_id % BLOCKS_PER_CHUNK`. See
// scratchy-target-metal's `BLOCKS_PER_CHUNK`.
constant uint ROPE_BLOCKS_PER_CHUNK [[function_constant(5)]];
// Rotation pairing offset: lane d < ROT_DIM/2 rotates the pair
// (d, d + PAIR_OFF). Standard NeoX (full + HF partial rope) passes
// ROT_DIM/2; Gemma4 proportional rope passes HEAD_DIM/2 (mlx
// `ProportionalRoPE` rotates the first ROT_DIM/2 lanes of EACH head
// half — pairs span the full head, not the rot window).
constant uint ROPE_PAIR_OFF [[function_constant(6)]];

// Norm-prologue fn-consts (rope_append_normed_* only; the plain
// rope_append_* kernels never reference them).
constant float ROPE_NORM_EPS      [[function_constant(7)]];
constant float ROPE_NORM_W_OFFSET [[function_constant(8)]];

// Rope-on-read (spans / position-independent KV): when set, K for blocks
// flagged unrotated is stored WITHOUT rotation (Q still rotates; K-norm
// preserved) so attention can re-rope it to any reuse position on read.
// Optional — default-off via is_function_constant_defined so non-spans
// rope_append is byte-identical (the gate + the flag buffer fold away).
// The K-rotation skip is the ONLY change; Q-rotation, K rmsnorm, and the
// V path are untouched. The "stored unrotated" flag rides in
// slot_mapping bit 31 (set by the worker) — free, since slot_mapping[t]
// is loaded for the paged write anyway; no separate flag buffer.
constant uint ROPE_ROPE_ON_READ [[function_constant(9)]];
constant bool ROPE_ROR_DEFINED = is_function_constant_defined(ROPE_ROPE_ON_READ);
constant uint ROPE_ROR = ROPE_ROR_DEFINED ? ROPE_ROPE_ON_READ : 0u;

kernel void rope_append_f16_specialized(
    device       half* q_inout      [[buffer(0)]],
    device       half* k_inout      [[buffer(1)]],
    device       half* v_inout      [[buffer(2)]],
    device const half* cos_sin      [[buffer(3)]],
    device const uint* positions    [[buffer(4)]],
    device const uint* slot_mapping [[buffer(5)]],
    device const uint64_t* kv_cache_k [[buffer(6)]],
    device const uint64_t* kv_cache_v [[buffer(7)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid    [[thread_position_in_threadgroup]])
{
    const uint t        = tg_pos.x;
    const uint q_head   = tg_pos.y;
    const uint d        = tid.x;
    const uint head_dim = ROPE_HEAD_DIM;
    const uint rot_dim  = ROPE_ROT_DIM;
    const uint half_dim = rot_dim / 2;
    const uint num_q    = ROPE_NUM_Q_HEADS;
    const uint num_kv   = ROPE_NUM_KV_HEADS;
    const uint block_sz = ROPE_BLOCK_SIZE;
    const uint group_r  = num_q / num_kv;

    if (q_head >= num_q || d >= head_dim) return;

    // Spans: store this block's K UNROTATED so attention can re-rope it to
    // any reuse position. Uniform per TG (slot/block shared across d).
    // Folds out (+ skips the flag load) when ROPE_ROR is unset.
    // Spans: slot_mapping bit 31 = this slot's block is stored unrotated
    // (set by the worker). Free — slot_mapping[t] is loaded for the write
    // anyway. (Padding slots are 0xFFFFFFFF → bit 31 set → skip is a
    // harmless no-op, the cache write is skipped on the sentinel below.)
    const bool skip_k_rot = (ROPE_ROR != 0u) && ((slot_mapping[t] & 0x80000000u) != 0u);

    const uint pos = positions[t];
    device const half* cos_row = cos_sin + pos * rot_dim;
    device const half* sin_row = cos_sin + pos * rot_dim + half_dim;

    // ── Q rotation (in-place) ────────────────────────────────────────
    const uint q_dim = num_q * head_dim;
    const uint pair_off = ROPE_PAIR_OFF;
    device half* q_row = q_inout + t * q_dim + q_head * head_dim;
    if (d < half_dim) {
        const float c  = float(cos_row[d]);
        const float s  = float(sin_row[d]);
        const float x0 = float(q_row[d]);
        const float x1 = float(q_row[pair_off + d]);
        q_row[d]            = half(x0 * c - x1 * s);
        q_row[pair_off + d] = half(x1 * c + x0 * s);
    }

    // ── K/V rotation + paged write (only owning q_head per kv_head) ─
    if (q_head % group_r != 0) return;
    const uint kv_head = q_head / group_r;
    const uint kv_dim  = num_kv * head_dim;
    device half* k_row = k_inout + t * kv_dim + kv_head * head_dim;
    device half* v_row = v_inout + t * kv_dim + kv_head * head_dim;

    // K rotation (in-place), unless this block is stored unrotated (spans).
    if (d < half_dim && !skip_k_rot) {
        const float c  = float(cos_row[d]);
        const float s  = float(sin_row[d]);
        const float x0 = float(k_row[d]);
        const float x1 = float(k_row[pair_off + d]);
        k_row[d]            = half(x0 * c - x1 * s);
        k_row[pair_off + d] = half(x1 * c + x0 * s);
    }
    // Fence the K writes — the paged write below has thread `d` read
    // `k_row[d]`, which (for d ≥ half_dim) was written by thread
    // `d - half_dim`. Without the barrier the paged write may see the
    // pre-rotation half.
    threadgroup_barrier(mem_flags::mem_device);

    // Paged write: kv_cache layout [num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM].
    // Sentinel `0xFFFFFFFF` marks padding lanes (write_slot_mapping in
    // pool.rs fills padding with u32::MAX) — skip the cache write so
    // padding's K_proj(token 0) does not corrupt slot 0.
    const uint slot         = slot_mapping[t];
    if (slot == 0xFFFFFFFFu) return;
    // Spans: strip the bit-31 unrotated flag before addressing (identity
    // when ROPE_ROR is off — the worker only sets bit 31 then).
    const uint phys_slot    = (ROPE_ROR != 0u) ? (slot & 0x7FFFFFFFu) : slot;
    const uint block_id     = phys_slot / block_sz;
    const uint block_offset = phys_slot % block_sz;
    const uint kv_blk_stride  = num_kv * block_sz * head_dim;
    const uint kv_head_stride = block_sz * head_dim;
    const uint kv_tok_stride  = head_dim;
    // Chunked KV: deref the chunk that backs this physical block, then
    // address with the block index WITHIN that chunk.
    const uint chunk        = block_id / ROPE_BLOCKS_PER_CHUNK;
    const uint blk_in_chunk = block_id % ROPE_BLOCKS_PER_CHUNK;
    device half* k_dst = (device half*)kv_cache_k[chunk]
        + blk_in_chunk * kv_blk_stride
        + kv_head      * kv_head_stride
        + block_offset * kv_tok_stride;
    device half* v_dst = (device half*)kv_cache_v[chunk]
        + blk_in_chunk * kv_blk_stride
        + kv_head      * kv_head_stride
        + block_offset * kv_tok_stride;

    // Each thread copies one element of K (rotated, post-write above)
    // and V (un-rotated). For partial-rope models (rot_dim < head_dim),
    // the tail [rot_dim, head_dim) of k_row is unrotated and copied
    // through unchanged.
    k_dst[d] = k_row[d];
    v_dst[d] = v_row[d];
}

/// BF16 specialized variant — same dispatch shape, function constants,
/// rotation math, and paged-cache layout as the f16 variant. Bindings
/// switch to `device bfloat*`; the cos_sin cache must be uploaded as
/// bf16 too (`upload_via_gpuweights` honors the dtype passed in).
kernel void rope_append_bf16_specialized(
    device       bfloat* q_inout      [[buffer(0)]],
    device       bfloat* k_inout      [[buffer(1)]],
    device       bfloat* v_inout      [[buffer(2)]],
    device const bfloat* cos_sin      [[buffer(3)]],
    device const uint*   positions    [[buffer(4)]],
    device const uint*   slot_mapping [[buffer(5)]],
    device const uint64_t* kv_cache_k [[buffer(6)]],
    device const uint64_t* kv_cache_v [[buffer(7)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid    [[thread_position_in_threadgroup]])
{
    const uint t        = tg_pos.x;
    const uint q_head   = tg_pos.y;
    const uint d        = tid.x;
    const uint head_dim = ROPE_HEAD_DIM;
    const uint rot_dim  = ROPE_ROT_DIM;
    const uint half_dim = rot_dim / 2;
    const uint num_q    = ROPE_NUM_Q_HEADS;
    const uint num_kv   = ROPE_NUM_KV_HEADS;
    const uint block_sz = ROPE_BLOCK_SIZE;
    const uint group_r  = num_q / num_kv;

    if (q_head >= num_q || d >= head_dim) return;

    // Spans: store this block's K UNROTATED (see the f16 sibling).
    // Spans: slot_mapping bit 31 = this slot's block is stored unrotated
    // (set by the worker). Free — slot_mapping[t] is loaded for the write
    // anyway. (Padding slots are 0xFFFFFFFF → bit 31 set → skip is a
    // harmless no-op, the cache write is skipped on the sentinel below.)
    const bool skip_k_rot = (ROPE_ROR != 0u) && ((slot_mapping[t] & 0x80000000u) != 0u);

    const uint pos = positions[t];
    device const bfloat* cos_row = cos_sin + pos * rot_dim;
    device const bfloat* sin_row = cos_sin + pos * rot_dim + half_dim;

    const uint q_dim = num_q * head_dim;
    const uint pair_off = ROPE_PAIR_OFF;
    device bfloat* q_row = q_inout + t * q_dim + q_head * head_dim;
    if (d < half_dim) {
        const float c  = float(cos_row[d]);
        const float s  = float(sin_row[d]);
        const float x0 = float(q_row[d]);
        const float x1 = float(q_row[pair_off + d]);
        q_row[d]            = bfloat(x0 * c - x1 * s);
        q_row[pair_off + d] = bfloat(x1 * c + x0 * s);
    }

    if (q_head % group_r != 0) return;
    const uint kv_head = q_head / group_r;
    const uint kv_dim  = num_kv * head_dim;
    device bfloat* k_row = k_inout + t * kv_dim + kv_head * head_dim;
    device bfloat* v_row = v_inout + t * kv_dim + kv_head * head_dim;

    // K rotation (in-place), unless this block is stored unrotated (spans).
    if (d < half_dim && !skip_k_rot) {
        const float c  = float(cos_row[d]);
        const float s  = float(sin_row[d]);
        const float x0 = float(k_row[d]);
        const float x1 = float(k_row[pair_off + d]);
        k_row[d]            = bfloat(x0 * c - x1 * s);
        k_row[pair_off + d] = bfloat(x1 * c + x0 * s);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Sentinel `0xFFFFFFFF` marks padding lanes (write_slot_mapping in
    // pool.rs fills padding with u32::MAX) — skip the cache write so
    // padding's K_proj(token 0) does not corrupt slot 0.
    const uint slot         = slot_mapping[t];
    if (slot == 0xFFFFFFFFu) return;
    // Spans: strip the bit-31 unrotated flag before addressing (identity
    // when ROPE_ROR is off — the worker only sets bit 31 then).
    const uint phys_slot    = (ROPE_ROR != 0u) ? (slot & 0x7FFFFFFFu) : slot;
    const uint block_id     = phys_slot / block_sz;
    const uint block_offset = phys_slot % block_sz;
    const uint kv_blk_stride  = num_kv * block_sz * head_dim;
    const uint kv_head_stride = block_sz * head_dim;
    const uint kv_tok_stride  = head_dim;
    // Chunked KV: deref the chunk that backs this physical block, then
    // address with the block index WITHIN that chunk.
    const uint chunk        = block_id / ROPE_BLOCKS_PER_CHUNK;
    const uint blk_in_chunk = block_id % ROPE_BLOCKS_PER_CHUNK;
    device bfloat* k_dst = (device bfloat*)kv_cache_k[chunk]
        + blk_in_chunk * kv_blk_stride
        + kv_head      * kv_head_stride
        + block_offset * kv_tok_stride;
    device bfloat* v_dst = (device bfloat*)kv_cache_v[chunk]
        + blk_in_chunk * kv_blk_stride
        + kv_head      * kv_head_stride
        + block_offset * kv_tok_stride;

    k_dst[d] = k_row[d];
    v_dst[d] = v_row[d];
}

/// BFloat16 variant of interleaved RoPE
kernel void rope_interleaved_bf16(
    device bfloat* query [[buffer(0)]],
    device bfloat* key [[buffer(1)]],
    constant bfloat* cos_sin_cache [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& num_kv_heads [[buffer(4)]],
    constant uint& rot_dim [[buffer(5)]],
    constant uint& head_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint embed_dim = rot_dim / 2;
    constant bfloat* cos_ptr = cos_sin_cache;
    constant bfloat* sin_ptr = cos_sin_cache + embed_dim;
    
    // Apply to query heads
    const uint nq = num_heads * embed_dim;
    if (tid < nq) {
        const uint head_idx = tid / embed_dim;
        const uint rot_offset = tid % embed_dim;
        
        const uint x_index = 2 * rot_offset;
        const uint y_index = 2 * rot_offset + 1;
        
        const bfloat cos_val = cos_ptr[rot_offset];
        const bfloat sin_val = sin_ptr[rot_offset];
        
        device bfloat* head_ptr = query + head_idx * head_size;
        const bfloat x = head_ptr[x_index];
        const bfloat y = head_ptr[y_index];
        
        head_ptr[x_index] = x * cos_val - y * sin_val;
        head_ptr[y_index] = y * cos_val + x * sin_val;
    }
    
    // Apply to key heads (if present)
    if (key != nullptr) {
        const uint nk = num_kv_heads * embed_dim;
        if (tid < nk) {
            const uint head_idx = tid / embed_dim;
            const uint rot_offset = tid % embed_dim;
            
            const uint x_index = 2 * rot_offset;
            const uint y_index = 2 * rot_offset + 1;
            
            const bfloat cos_val = cos_ptr[rot_offset];
            const bfloat sin_val = sin_ptr[rot_offset];
            
            device bfloat* head_ptr = key + head_idx * head_size;
            const bfloat x = head_ptr[x_index];
            const bfloat y = head_ptr[y_index];
            
            head_ptr[x_index] = x * cos_val - y * sin_val;
            head_ptr[y_index] = y * cos_val + x * sin_val;
        }
    }
}


// ---------------------------------------------------------------------------
// rope_append_normed_* — Gemma4 per-head norm prologue + RoPE + paged write
//
// Fuses the per-layer chain
//   q = rmsnorm(q_raw, q_gains)        (per head, over HEAD_DIM)
//   k = rmsnorm(k_raw, k_gains)
//   v = rmsnorm_unit(v_raw)
//   (q', k', v') = rope_append(q, k, v, ...)
// into the rope dispatch. Mirrors `Instruction::RopeAppendNormed`.
//
// BIT-EXACTNESS CONTRACT: the standalone `rmsnorm_specialized_impl` /
// `rmsnorm_unit_impl` run with tg_size = 256 (THREADS_PER_GROUP) — a
// strided `i += 256` accumulation and a 256-wide tree. This kernel
// runs HEAD_DIM threads (256 sliding / 512 global), so the prologue
// REPLICATES the 256-thread pattern exactly (threads d >= 256 idle
// through the reduction) — identical f32 summation order, identical
// rms, and normed values are rounded to T_act in threadgroup memory
// exactly where the unfused chain rounded to memory. The rotation
// then matches `rope_append_*_specialized` verbatim.
//
// OUTPUT CONTRACT (differs from the unfused chain ON PURPOSE):
//   - q' is written for ALL lanes (rotated pairs + pass-through of
//     unrotated lanes) to the q buffer (aliased to q_raw storage).
//   - K and V go ONLY to the paged cache. The k'/v' arena tiles are
//     dead on Gemma4 (every attention impl reads K/V from the cache)
//     and on global layers k_raw and v_raw are THE SAME buffer
//     (k_eq_v), so arena writeback would self-conflict.
//
// Function constants: ROPE_* 0..6 as rope_append + 7 = ROPE_NORM_EPS,
// 8 = ROPE_NORM_W_OFFSET (Gemma4 stores full gains -> 0.0).
//
// Bindings (must match `interpreter::metal::lowering` for
// `Instruction::RopeAppendNormed`):
//   buffer(0) = q_inout  (raw in; normed+rotated out, in place)
//   buffer(1) = k_in     (raw; read-only)
//   buffer(2) = v_in     (raw; read-only — k_in == v_in on k_eq_v)
//   buffer(3) = cos_sin, 4 = positions, 5 = slot_mapping,
//   buffer(6/7) = kv chunk tables, 8 = q_gains, 9 = k_gains.
//
// Dispatch: threadgroups (M, NUM_Q_HEADS, 1) x (HEAD_DIM, 1, 1).
// ---------------------------------------------------------------------------

/// 256-thread-replica per-head RMS: exact clone of the standalone
/// `rmsnorm_specialized_impl` reduction (tg_size = 256), regardless of
/// this kernel's actual threadgroup width. ALL threads of the TG must
/// call this (threadgroup barriers inside).
template <typename T_act>
inline float rope_norm_rms_256(
    device const T_act* row,
    uint n,
    float eps,
    threadgroup float* scratch,
    uint d)
{
    float local_sum = 0.0f;
    if (d < 256u) {
        for (uint i = d; i < n; i += 256u) {
            float val = float(row[i]);
            local_sum += val * val;
        }
        scratch[d] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1) {
        if (d < stride) {
            scratch[d] += scratch[d + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return sqrt(scratch[0] / float(n) + eps);
}

template <typename T_act, typename T_scale>
[[kernel]] void rope_append_normed_impl(
    device       T_act* q_inout      [[buffer(0)]],
    device const T_act* k_in         [[buffer(1)]],
    device const T_act* v_in         [[buffer(2)]],
    device const T_act* cos_sin      [[buffer(3)]],
    device const uint*  positions    [[buffer(4)]],
    device const uint*  slot_mapping [[buffer(5)]],
    device const uint64_t* kv_cache_k [[buffer(6)]],
    device const uint64_t* kv_cache_v [[buffer(7)]],
    device const T_scale* q_gains    [[buffer(8)]],
    device const T_scale* k_gains    [[buffer(9)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid    [[thread_position_in_threadgroup]])
{
    const uint t        = tg_pos.x;
    const uint q_head   = tg_pos.y;
    const uint d        = tid.x;
    const uint head_dim = ROPE_HEAD_DIM;
    const uint rot_dim  = ROPE_ROT_DIM;
    const uint half_dim = rot_dim / 2;
    const uint num_q    = ROPE_NUM_Q_HEADS;
    const uint num_kv   = ROPE_NUM_KV_HEADS;
    const uint block_sz = ROPE_BLOCK_SIZE;
    const uint group_r  = num_q / num_kv;
    const uint pair_off = ROPE_PAIR_OFF;

    if (q_head >= num_q || d >= head_dim) return;

    // Spans: store this block's K NORMED-but-UNROTATED (skip ONLY the K
    // rotation; rmsnorm + Q-rotation + V are preserved) so attention can
    // re-rope it on read. Uniform per TG. Folds out when ROPE_ROR unset.
    // Spans: slot_mapping bit 31 = this slot's block is stored unrotated
    // (set by the worker). Free — slot_mapping[t] is loaded for the write
    // anyway. (Padding slots are 0xFFFFFFFF → bit 31 set → skip is a
    // harmless no-op, the cache write is skipped on the sentinel below.)
    const bool skip_k_rot = (ROPE_ROR != 0u) && ((slot_mapping[t] & 0x80000000u) != 0u);

    // 512 = max head_dim this kernel serves (Gemma4 global). The
    // lowering asserts head_dim <= 512.
    threadgroup float scratch[256];
    threadgroup T_act q_tg[512];
    threadgroup T_act k_tg[512];

    const uint pos = positions[t];
    device const T_act* cos_row = cos_sin + pos * rot_dim;
    device const T_act* sin_row = cos_sin + pos * rot_dim + half_dim;

    // ── Q: per-head rmsnorm into TG memory, then rotate ──────────────
    const uint q_dim = num_q * head_dim;
    device T_act* q_row = q_inout + t * q_dim + q_head * head_dim;
    {
        const float rms = rope_norm_rms_256(q_row, head_dim, ROPE_NORM_EPS, scratch, d);
        const float w   = float(q_gains[d]) + ROPE_NORM_W_OFFSET;
        q_tg[d] = T_act((float(q_row[d]) / rms) * w);
    }
    // All raw-q reads complete before any q_inout write below.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d < half_dim) {
        const float c  = float(cos_row[d]);
        const float s  = float(sin_row[d]);
        const float x0 = float(q_tg[d]);
        const float x1 = float(q_tg[pair_off + d]);
        q_row[d]            = T_act(x0 * c - x1 * s);
        q_row[pair_off + d] = T_act(x1 * c + x0 * s);
    } else if (d < pair_off || d >= pair_off + half_dim) {
        // Lanes outside every rotation pair pass the normed value
        // through (proportional rope: lanes [half, pair_off) and
        // [pair_off + half, head_dim)).
        q_row[d] = q_tg[d];
    }

    // ── K/V: owning q_head only (uniform per-TG branch) ─────────────
    if (q_head % group_r != 0) return;
    const uint kv_head = q_head / group_r;
    const uint kv_dim  = num_kv * head_dim;
    device const T_act* k_row = k_in + t * kv_dim + kv_head * head_dim;
    device const T_act* v_row = v_in + t * kv_dim + kv_head * head_dim;

    {
        const float rms = rope_norm_rms_256(k_row, head_dim, ROPE_NORM_EPS, scratch, d);
        const float w   = float(k_gains[d]) + ROPE_NORM_W_OFFSET;
        k_tg[d] = T_act((float(k_row[d]) / rms) * w);
    }
    // REQUIRED: the V reduction below overwrites `scratch` — without
    // this barrier a fast thread clobbers scratch[0] while slower
    // threads are still reading it as rms_k for their k_tg lane
    // (found as a nondeterministic per-token K-cache divergence vs
    // the unfused chain, first manifesting mid-prompt at prefill).
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // V unit-norm (no gains, no offset — mirrors rmsnorm_unit_impl).
    // Its internal barriers also order the k_tg fill above before the
    // rotation below.
    T_act v_final;
    {
        const float rms = rope_norm_rms_256(v_row, head_dim, ROPE_NORM_EPS, scratch, d);
        v_final = T_act(float(v_row[d]) / rms);
    }
    // K rotation in TG memory (each pair touched by one thread), unless
    // this block is stored unrotated (spans): k_tg keeps the NORMED K.
    if (d < half_dim && !skip_k_rot) {
        const float c  = float(cos_row[d]);
        const float s  = float(sin_row[d]);
        const float x0 = float(k_tg[d]);
        const float x1 = float(k_tg[pair_off + d]);
        k_tg[d]            = T_act(x0 * c - x1 * s);
        k_tg[pair_off + d] = T_act(x1 * c + x0 * s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Paged write — identical addressing to rope_append_*_specialized.
    const uint slot = slot_mapping[t];
    if (slot == 0xFFFFFFFFu) return;
    // Spans: strip the bit-31 unrotated flag before addressing (identity
    // when ROPE_ROR is off — the worker only sets bit 31 then).
    const uint phys_slot    = (ROPE_ROR != 0u) ? (slot & 0x7FFFFFFFu) : slot;
    const uint block_id     = phys_slot / block_sz;
    const uint block_offset = phys_slot % block_sz;
    const uint kv_blk_stride  = num_kv * block_sz * head_dim;
    const uint kv_head_stride = block_sz * head_dim;
    const uint kv_tok_stride  = head_dim;
    const uint chunk        = block_id / ROPE_BLOCKS_PER_CHUNK;
    const uint blk_in_chunk = block_id % ROPE_BLOCKS_PER_CHUNK;
    device T_act* k_dst = (device T_act*)kv_cache_k[chunk]
        + blk_in_chunk * kv_blk_stride
        + kv_head      * kv_head_stride
        + block_offset * kv_tok_stride;
    device T_act* v_dst = (device T_act*)kv_cache_v[chunk]
        + blk_in_chunk * kv_blk_stride
        + kv_head      * kv_head_stride
        + block_offset * kv_tok_stride;

    k_dst[d] = k_tg[d];
    v_dst[d] = v_final;
}

#define INST_ROPE_APPEND_NORMED(act_tag, act_type, scale_tag, scale_type)   \
  template [[host_name("rope_append_normed_" #act_tag "_s_" #scale_tag     \
                       "_specialized")]]                                    \
  [[kernel]] decltype(rope_append_normed_impl<act_type, scale_type>)       \
      rope_append_normed_impl<act_type, scale_type>;

INST_ROPE_APPEND_NORMED(f16,  half,   f16,  half)
INST_ROPE_APPEND_NORMED(bf16, bfloat, f16,  half)
INST_ROPE_APPEND_NORMED(bf16, bfloat, bf16, bfloat)
INST_ROPE_APPEND_NORMED(f16,  half,   bf16, bfloat)
