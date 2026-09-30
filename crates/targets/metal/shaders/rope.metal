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
#include "megakernel/mk_common.h"
using namespace metal;

#ifndef MK_BODIES_ONLY
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

#endif // MK_BODIES_ONLY
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

// 0 HEAD_DIM, 1 NUM_Q_HEADS, 2 NUM_KV_HEADS, 3 ROT_DIM, 4 BLOCK_SIZE.
// 5 BLOCKS_PER_CHUNK — reactive (chunked) KV pool: buffers 6/7 are per-layer chunk-address
// TABLES (device uint64 gpuAddresses), not the cache buffers. A physical block id derefs
// `table[block_id / BLOCKS_PER_CHUNK]` then addresses with `block_id % BLOCKS_PER_CHUNK`. See
// scratchy-target-metal's `BLOCKS_PER_CHUNK`.
// 6 PAIR_OFF — rotation pairing offset: lane d < ROT_DIM/2 rotates the pair (d, d + PAIR_OFF).
// Standard NeoX (full + HF partial rope) passes ROT_DIM/2; Gemma4 proportional rope passes
// HEAD_DIM/2 (mlx `ProportionalRoPE` rotates the first ROT_DIM/2 lanes of EACH head half —
// pairs span the full head, not the rot window).
// 7 NORM_EPS, 8 NORM_W_OFFSET — the norm prologue (rope_append_normed_* only; the plain
// rope_append_* kernels never reference them).
// 9 ROPE_ON_READ — rope-on-read (spans / position-independent KV): when set, K for blocks
// flagged unrotated is stored WITHOUT rotation (Q still rotates; K-norm preserved) so attention
// can re-rope it to any reuse position on read. Optional — default-off via
// is_function_constant_defined so non-spans rope_append is byte-identical (the gate + the flag
// buffer fold away). The K-rotation skip is the ONLY change; Q-rotation, K rmsnorm, and the V
// path are untouched. The "stored unrotated" flag rides in slot_mapping bit 31 (set by the
// worker) — free, since slot_mapping[t] is loaded for the paged write anyway; no separate flag
// buffer.
#define ROPE_CONSTS(X)                                                                          \
  X(uint, head_dim, ROPE_HEAD_DIM, 0) X(uint, num_q, ROPE_NUM_Q_HEADS, 1)                       \
  X(uint, num_kv, ROPE_NUM_KV_HEADS, 2) X(uint, rot_dim, ROPE_ROT_DIM, 3)                       \
  X(uint, block_size, ROPE_BLOCK_SIZE, 4) X(uint, blocks_per_chunk, ROPE_BLOCKS_PER_CHUNK, 5)   \
  X(uint, pair_off, ROPE_PAIR_OFF, 6) X(float, norm_eps, ROPE_NORM_EPS, 7)                      \
  X(float, norm_w_offset, ROPE_NORM_W_OFFSET, 8) X(uint, ror_fc, ROPE_ROPE_ON_READ, 9)
#ifndef MK_BODIES_ONLY
ROPE_CONSTS(MK_FC_DECLARE)
constant bool ROPE_ROR_DEFINED = is_function_constant_defined(ROPE_ROPE_ON_READ);
constant uint ROPE_ROR = ROPE_ROR_DEFINED ? ROPE_ROPE_ON_READ : 0u;
// The dispatch kernels' constants, as the bodies read them; `ror()` = ROPE_ON_READ, 0 if unset.
struct RopeFc {
    ROPE_CONSTS(MK_FC_ACCESSOR)
    static METAL_FUNC uint ror() { return ROPE_ROR; }
};
#endif // MK_BODIES_ONLY
// A megakernel step's constants: its generated policy spells an unset constant as 0, which is
// already `ror()`.
template <typename G>
struct RopeMk {
    ROPE_CONSTS(MK_FC_FORWARD)
    static METAL_FUNC uint ror() { return G::ror_fc(); }
};

// Body shared by both dispatch kernels and the megakernel adapter. `MK`: the barrier-uniform
// form — every lane runs the K barrier; a lane of no real (virtual threadgroup, q_head, d) or of a
// q_head owning no kv_head writes nothing. `QP`: the arena rows' pointer type, `KP` the KV
// pages' (device-coherent in the megakernel).
template <typename T, typename C, bool MK, typename QP, typename KP>
METAL_FUNC void rope_append_body(
    QP q_inout, QP k_inout, QP v_inout, device const T* cos_sin, device const uint* positions,
    device const uint* slot_mapping, device const uint64_t* kv_cache_k,
    device const uint64_t* kv_cache_v, uint3 tg_pos, uint3 tid, bool live)
{
    const uint t        = tg_pos.x;
    const uint q_head   = tg_pos.y;
    const uint d        = tid.x;
    const uint head_dim = C::head_dim();
    const uint rot_dim  = C::rot_dim();
    const uint half_dim = rot_dim / 2;
    const uint num_q    = C::num_q();
    const uint num_kv   = C::num_kv();
    const uint block_sz = C::block_size();
    const uint group_r  = num_q / num_kv;

    if constexpr (MK) {
        live = live && q_head < num_q && d < head_dim;
    } else {
        if (q_head >= num_q || d >= head_dim) return;
    }

    // Spans: store this block's K UNROTATED so attention can re-rope it to
    // any reuse position. Uniform per TG (slot/block shared across d).
    // Folds out (+ skips the flag load) when ROPE_ROR is unset.
    // Spans: slot_mapping bit 31 = this slot's block is stored unrotated
    // (set by the worker). Free — slot_mapping[t] is loaded for the write
    // anyway. (Padding slots are 0xFFFFFFFF → bit 31 set → skip is a
    // harmless no-op, the cache write is skipped on the sentinel below.)
    const bool skip_k_rot = (C::ror() != 0u) && ((slot_mapping[t] & 0x80000000u) != 0u);

    const uint pos = positions[t];
    device const T* cos_row = cos_sin + pos * rot_dim;
    device const T* sin_row = cos_sin + pos * rot_dim + half_dim;

    // ── Q rotation (in-place) ────────────────────────────────────────
    const uint q_dim = num_q * head_dim;
    const uint pair_off = C::pair_off();
    QP q_row = q_inout + t * q_dim + q_head * head_dim;
    if ((!MK || live) && d < half_dim) {
        const float c  = float(cos_row[d]);
        const float s  = float(sin_row[d]);
        const float x0 = float(q_row[d]);
        const float x1 = float(q_row[pair_off + d]);
        q_row[d]            = T(x0 * c - x1 * s);
        q_row[pair_off + d] = T(x1 * c + x0 * s);
    }

    // ── K/V rotation + paged write (only owning q_head per kv_head) ─
    const bool own = live && q_head % group_r == 0;
    if constexpr (!MK) {
        if (q_head % group_r != 0) return;
    }
    const uint kv_head = q_head / group_r;
    const uint kv_dim  = num_kv * head_dim;
    QP k_row = k_inout + t * kv_dim + kv_head * head_dim;
    QP v_row = v_inout + t * kv_dim + kv_head * head_dim;

    // K rotation (in-place), unless this block is stored unrotated (spans).
    if ((!MK || own) && d < half_dim && !skip_k_rot) {
        const float c  = float(cos_row[d]);
        const float s  = float(sin_row[d]);
        const float x0 = float(k_row[d]);
        const float x1 = float(k_row[pair_off + d]);
        k_row[d]            = T(x0 * c - x1 * s);
        k_row[pair_off + d] = T(x1 * c + x0 * s);
    }
    // Fence the K writes — the paged write below has thread `d` read
    // `k_row[d]`, which (for d ≥ half_dim) was written by thread
    // `d - half_dim`. Without the barrier the paged write may see the
    // pre-rotation half.
    threadgroup_barrier(mem_flags::mem_device);
    if (MK && !own) return;

    // Paged write: kv_cache layout [num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM].
    // Sentinel `0xFFFFFFFF` marks padding lanes (write_slot_mapping in
    // pool.rs fills padding with u32::MAX) — skip the cache write so
    // padding's K_proj(token 0) does not corrupt slot 0.
    const uint slot         = slot_mapping[t];
    if (slot == 0xFFFFFFFFu) return;
    // Spans: strip the bit-31 unrotated flag before addressing (identity
    // when ROPE_ROR is off — the worker only sets bit 31 then).
    const uint phys_slot    = (C::ror() != 0u) ? (slot & 0x7FFFFFFFu) : slot;
    const uint block_id     = phys_slot / block_sz;
    const uint block_offset = phys_slot % block_sz;
    const uint kv_blk_stride  = num_kv * block_sz * head_dim;
    const uint kv_head_stride = block_sz * head_dim;
    const uint kv_tok_stride  = head_dim;
    // Chunked KV: deref the chunk that backs this physical block, then
    // address with the block index WITHIN that chunk.
    const uint chunk        = block_id / C::blocks_per_chunk();
    const uint blk_in_chunk = block_id % C::blocks_per_chunk();
    KP k_dst = (KP)kv_cache_k[chunk]
        + blk_in_chunk * kv_blk_stride
        + kv_head      * kv_head_stride
        + block_offset * kv_tok_stride;
    KP v_dst = (KP)kv_cache_v[chunk]
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

// Megakernel adapter: q / k / v (0-2) and the KV pages (via 6 / 7) device-coherent.
template <typename T, typename C>
MK_FUNC void mk_rope_append(thread const MkStep& s, MkLane l, threadgroup uchar*) {
    rope_append_body<T, C, true, mk_ptr<T>, mk_ptr<T>>(
        (mk_ptr<T>)s.addr[0], (mk_ptr<T>)s.addr[1], (mk_ptr<T>)s.addr[2],
        (device const T*)s.addr[3], (device const uint*)s.addr[4], (device const uint*)s.addr[5],
        (device const uint64_t*)s.addr[6], (device const uint64_t*)s.addr[7], l.tg_pos, l.tid3,
        l.live);
}

// The f16 and bf16 kernels: same dispatch shape, function constants, rotation math and
// paged-cache layout; the cos_sin cache is uploaded in the activation dtype
// (`upload_via_gpuweights` honors the dtype passed in).
#ifndef MK_BODIES_ONLY
#define INST_ROPE_APPEND(tag, T)                                                          \
kernel void rope_append_##tag##_specialized(                                             \
    device       T* q_inout      [[buffer(0)]],                                           \
    device       T* k_inout      [[buffer(1)]],                                           \
    device       T* v_inout      [[buffer(2)]],                                           \
    device const T* cos_sin      [[buffer(3)]],                                           \
    device const uint* positions    [[buffer(4)]],                                        \
    device const uint* slot_mapping [[buffer(5)]],                                        \
    device const uint64_t* kv_cache_k [[buffer(6)]],                                      \
    device const uint64_t* kv_cache_v [[buffer(7)]],                                      \
    uint3 tg_pos [[threadgroup_position_in_grid]],                                        \
    uint3 tid    [[thread_position_in_threadgroup]])                                      \
{                                                                                         \
    rope_append_body<T, RopeFc, false, device T*, device T*>(                             \
        q_inout, k_inout, v_inout, cos_sin, positions, slot_mapping, kv_cache_k,          \
        kv_cache_v, tg_pos, tid, true);                                                   \
}
#else
#define INST_ROPE_APPEND(tag, T)                                                            \
  MK_ADAPTER(rope_append_##tag##_specialized, 0, 0xc7, (mk_rope_append<T, RopeMk<MK_C>>), \
             ROPE_CONSTS)
#endif
INST_ROPE_APPEND(f16, half)
INST_ROPE_APPEND(bf16, bfloat)

#ifndef MK_BODIES_ONLY
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


#endif // MK_BODIES_ONLY

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
template <typename T_act, typename RP>
inline float rope_norm_rms_256(
    RP row,
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
        // The standalone 256-thread kernel writes shared_sum[tid] = 0
        // for every tid >= n (its strided loop is empty), so its tree
        // reduction sums exact zeros in the tail lanes. This kernel
        // dispatches only HEAD_DIM threads, so on head_dim < 256 the
        // lanes [n, 256) would otherwise hold uninitialized threadgroup
        // memory through the whole reduction — garbage rms, NaN output.
        // Zero them with the threads that exist (n == the threadgroup
        // width at every call site), keeping the summed values — and
        // therefore the f32 arithmetic — bit-identical to the
        // standalone kernel's.
        for (uint s = n + d; s < 256u; s += n) {
            scratch[s] = 0.0f;
        }
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

// Body shared by the dispatch kernels and the megakernel adapter. `MK`: the barrier-uniform
// form — every lane runs every norm barrier; a lane of no real (virtual threadgroup, q_head, d)
// or of a q_head owning no kv_head writes nothing (its norms reduce rows no one stores). `QP` /
// `IP`: the arena pointer types, `KP` the KV pages' (device-coherent in the megakernel);
// `scratch` / `q_tg` / `k_tg` the kernel's threadgroup arrays.
template <typename T_act, typename T_scale, typename C, bool MK, typename QP, typename IP,
          typename KP>
METAL_FUNC void rope_append_normed_body(
    QP q_inout, IP k_in, IP v_in, device const T_act* cos_sin, device const uint* positions,
    device const uint* slot_mapping, device const uint64_t* kv_cache_k,
    device const uint64_t* kv_cache_v, device const T_scale* q_gains,
    device const T_scale* k_gains, uint3 tg_pos, uint3 tid, bool live,
    threadgroup float* scratch, threadgroup T_act* q_tg, threadgroup T_act* k_tg)
{
    const uint t        = tg_pos.x;
    const uint q_head   = tg_pos.y;
    const uint d        = tid.x;
    const uint head_dim = C::head_dim();
    const uint rot_dim  = C::rot_dim();
    const uint half_dim = rot_dim / 2;
    const uint num_q    = C::num_q();
    const uint num_kv   = C::num_kv();
    const uint block_sz = C::block_size();
    const uint group_r  = num_q / num_kv;
    const uint pair_off = C::pair_off();

    if constexpr (MK) {
        live = live && q_head < num_q && d < head_dim;
    } else {
        if (q_head >= num_q || d >= head_dim) return;
    }

    // Spans: store this block's K NORMED-but-UNROTATED (skip ONLY the K
    // rotation; rmsnorm + Q-rotation + V are preserved) so attention can
    // re-rope it on read. Uniform per TG. Folds out when ROPE_ROR unset.
    // Spans: slot_mapping bit 31 = this slot's block is stored unrotated
    // (set by the worker). Free — slot_mapping[t] is loaded for the write
    // anyway. (Padding slots are 0xFFFFFFFF → bit 31 set → skip is a
    // harmless no-op, the cache write is skipped on the sentinel below.)
    const bool skip_k_rot = (C::ror() != 0u) && ((slot_mapping[t] & 0x80000000u) != 0u);

    const uint pos = positions[t];
    device const T_act* cos_row = cos_sin + pos * rot_dim;
    device const T_act* sin_row = cos_sin + pos * rot_dim + half_dim;

    // ── Q: per-head rmsnorm into TG memory, then rotate ──────────────
    const uint q_dim = num_q * head_dim;
    QP q_row = q_inout + t * q_dim + q_head * head_dim;
    {
        const float rms = rope_norm_rms_256<T_act>(q_row, head_dim, C::norm_eps(), scratch, d);
        const float w   = float(q_gains[d]) + C::norm_w_offset();
        q_tg[d] = T_act((float(q_row[d]) / rms) * w);
    }
    // All raw-q reads complete before any q_inout write below.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((!MK || live) && d < half_dim) {
        const float c  = float(cos_row[d]);
        const float s  = float(sin_row[d]);
        const float x0 = float(q_tg[d]);
        const float x1 = float(q_tg[pair_off + d]);
        q_row[d]            = T_act(x0 * c - x1 * s);
        q_row[pair_off + d] = T_act(x1 * c + x0 * s);
    } else if ((!MK || live) && (d < pair_off || d >= pair_off + half_dim)) {
        // Lanes outside every rotation pair pass the normed value
        // through (proportional rope: lanes [half, pair_off) and
        // [pair_off + half, head_dim)).
        q_row[d] = q_tg[d];
    }

    // ── K/V: owning q_head only (uniform per-TG branch) ─────────────
    const bool own = live && q_head % group_r == 0;
    if constexpr (!MK) {
        if (q_head % group_r != 0) return;
    }
    const uint kv_head = q_head / group_r;
    const uint kv_dim  = num_kv * head_dim;
    IP k_row = k_in + t * kv_dim + kv_head * head_dim;
    IP v_row = v_in + t * kv_dim + kv_head * head_dim;

    {
        const float rms = rope_norm_rms_256<T_act>(k_row, head_dim, C::norm_eps(), scratch, d);
        const float w   = float(k_gains[d]) + C::norm_w_offset();
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
        const float rms = rope_norm_rms_256<T_act>(v_row, head_dim, C::norm_eps(), scratch, d);
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
    if (MK && !own) return;

    // Paged write — identical addressing to rope_append_*_specialized.
    const uint slot = slot_mapping[t];
    if (slot == 0xFFFFFFFFu) return;
    // Spans: strip the bit-31 unrotated flag before addressing (identity
    // when ROPE_ROR is off — the worker only sets bit 31 then).
    const uint phys_slot    = (C::ror() != 0u) ? (slot & 0x7FFFFFFFu) : slot;
    const uint block_id     = phys_slot / block_sz;
    const uint block_offset = phys_slot % block_sz;
    const uint kv_blk_stride  = num_kv * block_sz * head_dim;
    const uint kv_head_stride = block_sz * head_dim;
    const uint kv_tok_stride  = head_dim;
    const uint chunk        = block_id / C::blocks_per_chunk();
    const uint blk_in_chunk = block_id % C::blocks_per_chunk();
    KP k_dst = (KP)kv_cache_k[chunk]
        + blk_in_chunk * kv_blk_stride
        + kv_head      * kv_head_stride
        + block_offset * kv_tok_stride;
    KP v_dst = (KP)kv_cache_v[chunk]
        + blk_in_chunk * kv_blk_stride
        + kv_head      * kv_head_stride
        + block_offset * kv_tok_stride;

    k_dst[d] = k_tg[d];
    v_dst[d] = v_final;
}

// Megakernel adapter: q (0), k (1), v (2) and the KV pages (via 6 / 7) device-coherent; one
// virtual threadgroup's scratch[256] float + q_tg[512] + k_tg[512] T_act in its region.
template <typename T_act, typename T_scale, typename C>
MK_FUNC void mk_rope_append_normed(thread const MkStep& s, MkLane l, threadgroup uchar* tg) {
    threadgroup uchar* region = mk_region(s, l, tg);
    rope_append_normed_body<T_act, T_scale, C, true, mk_ptr<T_act>, mk_cptr<T_act>,
                            mk_ptr<T_act>>(
        (mk_ptr<T_act>)s.addr[0], (mk_cptr<T_act>)s.addr[1], (mk_cptr<T_act>)s.addr[2],
        (device const T_act*)s.addr[3], (device const uint*)s.addr[4],
        (device const uint*)s.addr[5], (device const uint64_t*)s.addr[6],
        (device const uint64_t*)s.addr[7], (device const T_scale*)s.addr[8],
        (device const T_scale*)s.addr[9], l.tg_pos, l.tid3, l.live, (threadgroup float*)region,
        (threadgroup T_act*)(region + 1024), (threadgroup T_act*)(region + 1024 + 512 * sizeof(T_act)));
}

#ifndef MK_BODIES_ONLY
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
    // 512 = max head_dim this kernel serves (Gemma4 global). The
    // lowering asserts head_dim <= 512.
    threadgroup float scratch[256];
    threadgroup T_act q_tg[512];
    threadgroup T_act k_tg[512];
    rope_append_normed_body<T_act, T_scale, RopeFc, false, device T_act*, device const T_act*,
                            device T_act*>(
        q_inout, k_in, v_in, cos_sin, positions, slot_mapping, kv_cache_k, kv_cache_v, q_gains,
        k_gains, tg_pos, tid, true, scratch, q_tg, k_tg);
}

#define INST_ROPE_APPEND_NORMED(act_tag, act_type, scale_tag, scale_type)   \
  template [[host_name("rope_append_normed_" #act_tag "_s_" #scale_tag     \
                       "_specialized")]]                                    \
  [[kernel]] decltype(rope_append_normed_impl<act_type, scale_type>)       \
      rope_append_normed_impl<act_type, scale_type>;
#else
// scratch[256] float + q_tg[512] + k_tg[512] of the activation type.
#define INST_ROPE_APPEND_NORMED(act_tag, act_type, scale_tag, scale_type)                        \
  MK_ADAPTER(rope_append_normed_##act_tag##_s_##scale_tag##_specialized, 3072, 0xc7,          \
             (mk_rope_append_normed<act_type, scale_type, RopeMk<MK_C>>), ROPE_CONSTS)
#endif

INST_ROPE_APPEND_NORMED(f16,  half,   f16,  half)
INST_ROPE_APPEND_NORMED(bf16, bfloat, f16,  half)
INST_ROPE_APPEND_NORMED(bf16, bfloat, bf16, bfloat)
INST_ROPE_APPEND_NORMED(f16,  half,   bf16, bfloat)
