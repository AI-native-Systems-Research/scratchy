// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project
//
// On-GPU token sampler for the metal backend — a VERBATIM port of the CUDA
// fused top-k / top-p / min-p sampler in
// `crates/targets/cuda/csrc/sampling_kernels.cu`:
//   * `apply_penalties_kernel`      (repetition / frequency / presence)
//   * `sample_top_k_top_p_core`     (softmax → radix-select → compact → bitonic
//                                    sort → top-p cutoff → categorical sample)
// plus f16/bf16 → f32 row-gather cast kernels (mirror cuda's `cast_to_f32_kernel`).
//
// Design (matches cuda's f32 "slow" sampling path): the worker gathers each
// sampling request's logits row into a compact f32 scratch buffer via the
// `cast_rows_*` kernel, applies penalties in f32, then samples in f32 — so only
// the CAST kernel is dtype-specialized (f16/bf16); penalties + sampling are
// f32-only. The per-request `uniform_random` is drawn host-side from the request
// RNG (top-k/top-p uses a single uniform per request), so no on-GPU Philox is
// needed (see the cuda gumbel/philox path, which THIS task does not port).
//
// Dispatch (every kernel here): threadgroups = (nrows, 1, 1),
// threadsPerThreadgroup = (256, 1, 1). One threadgroup per request row; the 256
// threads cooperate over the vocab axis. The two-level reductions (simd shuffle
// then threadgroup) mirror cuda's warp+block reductions exactly; Apple GPUs have
// a 32-wide simdgroup with linear lane assignment, so `tid % 32` / `tid / 32`
// are the simd lane / simd group — matching cuda's `threadIdx.x % 32` / `/ 32`.

#include <metal_stdlib>
using namespace metal;

#define SAMPLING_BLOCK_SIZE 256
#define MAX_CANDIDATES 1024
#define WARP_SIZE 32
#define NUM_WARPS (SAMPLING_BLOCK_SIZE / WARP_SIZE)

// The block reductions split the threadgroup into whole 32-lane simdgroups
// (warp_id = tid / WARP_SIZE, lane = tid % WARP_SIZE) and park one partial per
// simdgroup in a NUM_WARPS-slot buffer. That math is only valid if the block is
// an exact multiple of the simdgroup width — guard it at compile time. (The
// matching runtime fact "this device's simdgroup IS 32 wide" is not a
// compile-time constant, so `SamplerKernels::new` checks the pipeline's
// threadExecutionWidth == WARP_SIZE at load.)
static_assert(SAMPLING_BLOCK_SIZE % WARP_SIZE == 0,
              "SAMPLING_BLOCK_SIZE must be a whole number of WARP_SIZE-lane "
              "simdgroups (block_reduce_* warp_buf indexing)");

// ---------------------------------------------------------------------------
// Warp-level (simd) reductions — port of cuda's `__shfl_xor_sync` loops.
// ---------------------------------------------------------------------------

inline float warp_reduce_sum(float val) {
    for (ushort offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += simd_shuffle_xor(val, offset);
    }
    return val;
}

inline float warp_reduce_max(float val) {
    for (ushort offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = max(val, simd_shuffle_xor(val, offset));
    }
    return val;
}

inline int warp_reduce_sum_int(int val) {
    for (ushort offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += simd_shuffle_xor(val, offset);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Block-level reductions (all threads get the result). One-for-one port of the
// cuda `block_reduce_*` — first reduce within each warp, park each warp's
// partial in `warp_buf`, then have warp 0 reduce the partials.
// ---------------------------------------------------------------------------

inline float block_reduce_sum(float val, threadgroup float* warp_buf, uint tid) {
    uint warp_id = tid / WARP_SIZE;
    uint lane_id = tid % WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane_id == 0) warp_buf[warp_id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < WARP_SIZE) {
        float v = (tid < NUM_WARPS) ? warp_buf[tid] : 0.0f;
        v = warp_reduce_sum(v);
        if (tid == 0) warp_buf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result = warp_buf[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

inline float block_reduce_max(float val, threadgroup float* warp_buf, uint tid) {
    uint warp_id = tid / WARP_SIZE;
    uint lane_id = tid % WARP_SIZE;

    val = warp_reduce_max(val);
    if (lane_id == 0) warp_buf[warp_id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < WARP_SIZE) {
        float v = (tid < NUM_WARPS) ? warp_buf[tid] : -INFINITY;
        v = warp_reduce_max(v);
        if (tid == 0) warp_buf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result = warp_buf[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

inline int block_reduce_sum_int(int val, threadgroup int* warp_buf, uint tid) {
    uint warp_id = tid / WARP_SIZE;
    uint lane_id = tid % WARP_SIZE;

    val = warp_reduce_sum_int(val);
    if (lane_id == 0) warp_buf[warp_id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < WARP_SIZE) {
        int v = (tid < NUM_WARPS) ? warp_buf[tid] : 0;
        v = warp_reduce_sum_int(v);
        if (tid == 0) warp_buf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int result = warp_buf[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

// ---------------------------------------------------------------------------
// Next power of 2 — port of cuda `next_pow2`.
// ---------------------------------------------------------------------------

inline int next_pow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

// ---------------------------------------------------------------------------
// Row-gather cast: for row r (= threadgroup), copy logits[row_indices[r], :]
// converted to f32 into out[r, :]. Mirrors cuda `cast_to_f32_kernel`, but with
// a per-row source index (the sample position in the [total_n, vocab] logits).
//
// Bindings:
//   buffer(0) = out          [nrows, vocab]   float   (write)
//   buffer(1) = logits       [total_n, vocab] half/bfloat (read)
//   buffer(2) = row_indices  [nrows]          uint    (sample row per output)
//   buffer(3) = vocab        constant uint
// ---------------------------------------------------------------------------

kernel void cast_rows_f16_to_f32(
    device       float* out          [[buffer(0)]],
    device const half*  logits       [[buffer(1)]],
    device const uint*  row_indices  [[buffer(2)]],
    constant     uint&  vocab        [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]])
{
    uint src_row = row_indices[tg_id];
    device const half* in = logits + (size_t)src_row * vocab;
    device float* dst = out + (size_t)tg_id * vocab;
    for (uint i = tid; i < vocab; i += SAMPLING_BLOCK_SIZE) {
        dst[i] = float(in[i]);
    }
}

kernel void cast_rows_bf16_to_f32(
    device       float*  out         [[buffer(0)]],
    device const bfloat* logits      [[buffer(1)]],
    device const uint*   row_indices [[buffer(2)]],
    constant     uint&   vocab       [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]])
{
    uint src_row = row_indices[tg_id];
    device const bfloat* in = logits + (size_t)src_row * vocab;
    device float* dst = out + (size_t)tg_id * vocab;
    for (uint i = tid; i < vocab; i += SAMPLING_BLOCK_SIZE) {
        dst[i] = float(in[i]);
    }
}

// ---------------------------------------------------------------------------
// apply_penalties — repetition / frequency / presence, one threadgroup per row.
// VERBATIM port of cuda `apply_penalties_kernel` (operates on the f32 scratch).
//
// For each token v:
//   count = occurrences in output_token_ids[row] + prompt_token_ids[row]
//   if count > 0:
//     logit = logit > 0 ? logit / rep : logit * rep
//     logit -= freq * count + pres
// Token id == vocab_size is padding (never matches a real vocab index).
//
// Bindings:
//   buffer(0) = logits            [nrows, vocab]        float (in/out)
//   buffer(1) = output_token_ids  [nrows, max_out]      int  (padded w/ vocab)
//   buffer(2) = prompt_token_ids  [nrows, max_prompt]   int  (padded w/ vocab)
//   buffer(3) = rep_penalties     [nrows]               float
//   buffer(4) = freq_penalties    [nrows]               float
//   buffer(5) = pres_penalties    [nrows]               float
//   buffer(6) = vocab             constant uint
//   buffer(7) = max_output_len    constant uint
//   buffer(8) = max_prompt_len    constant uint
// ---------------------------------------------------------------------------

kernel void apply_penalties(
    device       float* logits            [[buffer(0)]],
    device const int*   output_token_ids  [[buffer(1)]],
    device const int*   prompt_token_ids  [[buffer(2)]],
    device const float* rep_penalties     [[buffer(3)]],
    device const float* freq_penalties    [[buffer(4)]],
    device const float* pres_penalties    [[buffer(5)]],
    constant     uint&  vocab             [[buffer(6)]],
    constant     uint&  max_output_len    [[buffer(7)]],
    constant     uint&  max_prompt_len    [[buffer(8)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]])
{
    float rep_pen  = rep_penalties[tg_id];
    float freq_pen = freq_penalties[tg_id];
    float pres_pen = pres_penalties[tg_id];

    device float* row            = logits + (size_t)tg_id * vocab;
    device const int* out_ids    = output_token_ids + (size_t)tg_id * max_output_len;
    device const int* prompt_ids = prompt_token_ids + (size_t)tg_id * max_prompt_len;

    for (uint v = tid; v < vocab; v += SAMPLING_BLOCK_SIZE) {
        int count = 0;
        for (uint j = 0; j < max_output_len; j++) {
            if (out_ids[j] == (int)v) count++;
        }
        for (uint j = 0; j < max_prompt_len; j++) {
            if (prompt_ids[j] == (int)v) count++;
        }

        if (count > 0) {
            float logit = row[v];
            if (logit > 0.0f) {
                logit /= rep_pen;
            } else {
                logit *= rep_pen;
            }
            logit -= freq_pen * (float)count + pres_pen;
            row[v] = logit;
        }
    }
}

// ---------------------------------------------------------------------------
// sample_top_k_top_p — VERBATIM port of cuda `sample_top_k_top_p_core`, batched
// (one threadgroup per row). Reads the f32 scratch (post-penalties).
//
// Bindings:
//   buffer(0) = output        [nrows]          uint  (write: sampled token id)
//   buffer(1) = logits        [nrows, vocab]    float (read)
//   buffer(2) = temperatures  [nrows]           float
//   buffer(3) = top_ks        [nrows]           int
//   buffer(4) = top_ps        [nrows]           float
//   buffer(5) = min_ps        [nrows]           float
//   buffer(6) = uniforms      [nrows]           float  (host-drawn per row)
//   buffer(7) = vocab         constant uint
// ---------------------------------------------------------------------------

kernel void sample_top_k_top_p(
    device       uint*  output        [[buffer(0)]],
    device const float* logits_all    [[buffer(1)]],
    device const float* temperatures  [[buffer(2)]],
    device const int*   top_ks        [[buffer(3)]],
    device const float* top_ps        [[buffer(4)]],
    device const float* min_ps        [[buffer(5)]],
    device const float* uniforms      [[buffer(6)]],
    constant     uint&  vocab_size    [[buffer(7)]],
#ifdef SCRATCHY_SAMPLER_TELEMETRY
    // Sampler telemetry (all writes gated by telem_on): spill the sorted top-K
    // (token_id, prob) into topk_probs/topk_indices [row*telem_k + i], and
    // [max_prob, entropy_nats] into stats_out[row*2 + {0,1}]. When telem_on is 0
    // these are bound to tiny dummies and never written. Compiled in only under
    // the `sampler-telemetry` feature (build.rs -DSCRATCHY_SAMPLER_TELEMETRY).
    device       float* topk_probs    [[buffer(8)]],
    device       uint*  topk_indices  [[buffer(9)]],
    device       float* stats_out     [[buffer(10)]],
    constant     uint&  telem_on      [[buffer(11)]],
    constant     uint&  telem_k       [[buffer(12)]],
#endif
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]])
{
    threadgroup float s_warp_buf[NUM_WARPS];
    threadgroup int   s_warp_buf_int[NUM_WARPS];
    threadgroup float s_probs[MAX_CANDIDATES];
    threadgroup uint  s_indices[MAX_CANDIDATES];
    threadgroup atomic_int s_num_candidates;

    device const float* logits = logits_all + (size_t)tg_id * vocab_size;
    int vsize = (int)vocab_size;
    float temperature = temperatures[tg_id];
    int   top_k       = top_ks[tg_id];
    float top_p       = top_ps[tg_id];
    float min_p       = min_ps[tg_id];
    float uniform_random = uniforms[tg_id];

    float inv_temp = 1.0f / temperature;

    // ===== Phase 1: Find max logit (1 pass) =====
    float local_max = -INFINITY;
    for (int i = tid; i < vsize; i += SAMPLING_BLOCK_SIZE) {
        float val = logits[i] * inv_temp;
        local_max = max(local_max, val);
    }
    float max_logit = block_reduce_max(local_max, s_warp_buf, tid);

    // ===== Phase 2: Compute sum_exp for softmax denominator (1 pass) =====
    float local_sum = 0.0f;
    for (int i = tid; i < vsize; i += SAMPLING_BLOCK_SIZE) {
        float val = logits[i] * inv_temp;
        local_sum += exp(val - max_logit);
    }
    float sum_exp = block_reduce_sum(local_sum, s_warp_buf, tid);
    float inv_sum_exp = 1.0f / sum_exp;

    // prob(i) = exp(logit(i)/T - max_logit) * inv_sum_exp

#ifdef SCRATCHY_SAMPLER_TELEMETRY
    // ===== Sampler telemetry: full-vocab Shannon entropy (gated) =====
    // One extra strided pass over the same softmax. telem_on is a uniform, so
    // every thread takes the branch together (block_reduce is a collective).
    float entropy = 0.0f;
    if (telem_on != 0u) {
        float local_ent = 0.0f;
        for (int i = tid; i < vsize; i += SAMPLING_BLOCK_SIZE) {
            float val = logits[i] * inv_temp;
            float p = exp(val - max_logit) * inv_sum_exp;
            if (p > 0.0f) {
                local_ent -= p * log(p);
            }
        }
        entropy = block_reduce_sum(local_ent, s_warp_buf, tid);
    }
#endif

    // ===== Phase 3: Radix select for top-K threshold (32 passes) =====
    int effective_k = (top_k > 0)
        ? min(top_k, vsize)
        : min(MAX_CANDIDATES, vsize);

    uint threshold_bits = 0;

    for (int bit = 31; bit >= 0; bit--) {
        uint candidate = threshold_bits | (1u << bit);

        int local_count = 0;
        for (int i = tid; i < vsize; i += SAMPLING_BLOCK_SIZE) {
            float val = logits[i] * inv_temp;
            float prob = exp(val - max_logit) * inv_sum_exp;
            uint bits = as_type<uint>(prob);
            if (bits >= candidate) local_count++;
        }

        int total_count = block_reduce_sum_int(local_count, s_warp_buf_int, tid);
        if (total_count >= effective_k) {
            threshold_bits = candidate;
        }
    }

    float threshold_prob = as_type<float>(threshold_bits);

    // ===== Phase 3b: Apply min_p threshold =====
    if (min_p > 0.0f) {
        // max_prob = exp(0) * inv_sum_exp = inv_sum_exp.
        float max_prob = inv_sum_exp;
        float min_p_threshold = min_p * max_prob;
        threshold_prob = max(threshold_prob, min_p_threshold);
    }

    // ===== Phase 4: Two-phase compaction to threadgroup memory =====
    uint threshold_bits_u = as_type<uint>(threshold_prob);
    int cap = min(effective_k, MAX_CANDIDATES);

    if (tid == 0) atomic_store_explicit(&s_num_candidates, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4a: tokens strictly above threshold (prob bits > threshold_bits).
    for (int i = tid; i < vsize; i += SAMPLING_BLOCK_SIZE) {
        float val = logits[i] * inv_temp;
        float prob = exp(val - max_logit) * inv_sum_exp;
        uint bits = as_type<uint>(prob);
        if (bits > threshold_bits_u) {
            int pos = atomic_fetch_add_explicit(&s_num_candidates, 1, memory_order_relaxed);
            if (pos < cap) {
                s_probs[pos] = prob;
                s_indices[pos] = (uint)i;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int strict_count = min(atomic_load_explicit(&s_num_candidates, memory_order_relaxed), cap);

    // Phase 4b: tokens at threshold (fill remaining slots).
    if (strict_count < cap) {
        for (int i = tid; i < vsize; i += SAMPLING_BLOCK_SIZE) {
            float val = logits[i] * inv_temp;
            float prob = exp(val - max_logit) * inv_sum_exp;
            uint bits = as_type<uint>(prob);
            if (bits == threshold_bits_u) {
                int pos = atomic_fetch_add_explicit(&s_num_candidates, 1, memory_order_relaxed);
                if (pos < cap) {
                    s_probs[pos] = prob;
                    s_indices[pos] = (uint)i;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int num_candidates = min(atomic_load_explicit(&s_num_candidates, memory_order_relaxed), cap);
    if (num_candidates <= 0) {
        // Fallback: should not happen, but return argmax token.
        if (tid == 0) {
            float best = -INFINITY;
            uint best_idx = 0;
            for (int i = 0; i < vsize; i++) {
                float val = logits[i];
                if (val > best) { best = val; best_idx = (uint)i; }
            }
            output[tg_id] = best_idx;
        }
        return;
    }

    // ===== Phase 5: Bitonic sort in threadgroup memory (descending by prob) =====
    int n_padded = next_pow2(num_candidates);

    // Pad with zeros.
    for (int i = tid + num_candidates; i < n_padded; i += SAMPLING_BLOCK_SIZE) {
        s_probs[i] = 0.0f;
        s_indices[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort: descending order.
    for (int k = 2; k <= n_padded; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < n_padded; i += SAMPLING_BLOCK_SIZE) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool swap_if_less = ((i & k) == 0);
                    float pi = s_probs[i];
                    float pj = s_probs[ixj];
                    if (swap_if_less ? (pi < pj) : (pi > pj)) {
                        s_probs[i] = pj;
                        s_probs[ixj] = pi;
                        uint tmp = s_indices[i];
                        s_indices[i] = s_indices[ixj];
                        s_indices[ixj] = tmp;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // ===== Phase 6: Top-p cutoff + renormalize + sample (thread 0) =====
    if (tid == 0) {
        // Find top-p cutoff: first index where cumulative sum > top_p.
        float cumsum = 0.0f;
        int cutoff = num_candidates;
        for (int i = 0; i < num_candidates; i++) {
            cumsum += s_probs[i];
            if (cumsum > top_p) {
                cutoff = i + 1;  // inclusive: keep this token
                break;
            }
        }

        // Re-normalize surviving probs and sample.
        float total = 0.0f;
        for (int i = 0; i < cutoff; i++) {
            total += s_probs[i];
        }

        float target = uniform_random * total;
        cumsum = 0.0f;
        uint sampled = s_indices[cutoff - 1];  // fallback: last survivor
        for (int i = 0; i < cutoff; i++) {
            cumsum += s_probs[i];
            if (cumsum >= target) {
                sampled = s_indices[i];
                break;
            }
        }

        output[tg_id] = sampled;

#ifdef SCRATCHY_SAMPLER_TELEMETRY
        // Spill the sorted top-K + confidence + entropy for the live "soul"
        // panel. s_probs/s_indices are the descending-sorted raw softmax probs
        // (top-p renorm above touched only locals, not these arrays).
        if (telem_on != 0u) {
            uint base = tg_id * telem_k;
            for (uint i = 0; i < telem_k; i++) {
                bool valid = (int)i < num_candidates;
                topk_probs[base + i] = valid ? s_probs[i] : 0.0f;
                topk_indices[base + i] = valid ? s_indices[i] : 0u;
            }
            stats_out[tg_id * 2u + 0u] = inv_sum_exp;  // max_prob
            stats_out[tg_id * 2u + 1u] = entropy;
        }
#endif
    }
}
