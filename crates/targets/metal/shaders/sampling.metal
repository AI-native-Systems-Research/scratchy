// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project
//
// On-GPU token sampler for the metal backend — a port of the CUDA fused
// top-k / top-p / min-p sampler in `crates/targets/cuda/csrc/sampling_kernels.cu`
// (`cast_to_f32_kernel`, `apply_penalties_kernel`, `sample_top_k_top_p_core`),
// restructured for the shape Apple GPUs actually see at chat batch sizes.
//
// The cuda kernel (and this port's first revision) ran one 256-thread
// threadgroup per request row: every full-vocab pass — softmax, 32 radix
// passes, compaction — executed on ONE threadgroup, i.e. one core of the GPU.
// At batch 1 that serializes ~34 vocab passes through 1/N of the machine
// (measured: 4.2 ms per sampled token on a 262k vocab; see
// `tests/sampling_bench.rs`). This revision keeps the cuda kernel's
// ARITHMETIC — the same softmax, the same bit-serial radix threshold, the
// same tie-aware compaction, sort, top-p cutoff and categorical draw — but
// splits each full-vocab pass across many threadgroups (one per vocab slice)
// and replaces the 32 bit-serial passes with a 4-round byte-histogram descent
// that provably selects the same threshold: the bit-serial greedy keeps
// threshold bit b iff count(bits >= candidate) >= k, which over
// monotonically-ordered nonneg-float bits equals picking, per byte from the
// top, the largest b whose suffix count reaches k. Cross-slice decisions run
// in tiny one-threadgroup-per-row kernels; all inter-kernel state lives in a
// per-row `row_state` block so every dispatch shares one binding table and
// the whole pipeline rides one command buffer with Device barriers.
//
// Pipeline (host dispatch order):
//   cast_rows_{f16,bf16}_to_f32  (row × slice)  logits row → f32 scratch
//   apply_penalties              (row × slice)  in-place on the f32 scratch
//   sample_softmax_reduce        (row × slice)  per-slice (max, sum) partials
//   sample_stats_pick            (row)          merge partials → row max/sum
//   sample_softmax_materialize   (row × slice)  probs, as bits, in place
//   sample_histogram_pass        (row × slice)  byte histogram of the round's
//                                              survivors (round in row_state)
//   sample_threshold_pick        (row)          ×4: pick the round's byte,
//                                              advance the round; the last
//                                              round applies min-p
//   sample_count_compact         (row × slice)  per-slice strict/tied counts +
//                                              strict candidates to staging
//   sample_quota_pick            (row)          tie quotas per slice,
//                                              in index order
//   sample_compact_tied          (row × slice)  tied candidates to staging
//   sample_finalize              (row)          gather, bitonic sort desc,
//                                              top-p cutoff, categorical draw
//
// Design (matches cuda's f32 "slow" sampling path): only the CAST kernel is
// dtype-specialized (f16/bf16); penalties + sampling are f32-only. The
// per-request `uniform_random` is drawn host-side from the request RNG, so no
// on-GPU Philox is needed.
//
// Sliced kernels use threadgroups = (nrows * nslices, 1, 1),
// threadsPerThreadgroup = (SAMPLING_BLOCK_SIZE, 1, 1); row = tg / nslices,
// slice = tg % nslices, and each threadgroup strides its contiguous slice of
// the vocab axis. `nslices` follows the step's row count (host-chosen); the
// vocab is the model's, compiled in (`SAMPLER_VOCAB`).
//
// SHARED BINDING TABLE (one table serves every kernel; each kernel reads
// only the slots named in its comment):
//    0  f32 scratch / prob bits   [nrows, vocab]
//    1  logits                     [total_n, vocab]   (cast in)
//    2  row_indices                [nrows] uint
//    3  output_token_ids           [nrows, max_out]   (penalties)
//    4  prompt_token_ids           [nrows, max_prompt]
//    5  rep_penalties              [nrows] f32
//    6  freq_penalties             [nrows] f32
//    7  pres_penalties             [nrows] f32
//    8  row_state                  [nrows, 16] uint   (layout below)
//    9  partials                   [nrows, nslices, 3] uint
//   10  hist                       [nrows, nslices, 256] uint
//   11  counts                     [nrows, nslices, 4] uint
//   12  staging                    [nrows, nslices, 2*MAX_CANDIDATES] uint
//   13  output                     [nrows] uint
//   14  consts                     (nslices, nrows, max_out, max_prompt)
//   15  telemetry topk_probs       (sampler-telemetry only)
//   16  telemetry topk_indices
//   17  telemetry stats_out
//   18  telemetry consts           (telem_on, telem_k)
//
// row_state[row] layout (u32 words; host writes 0..8, GPU the rest):
//    0  temperature bits     1  max_logit bits      2  sum_exp bits
//    3  top_k (raw)          4  top_p bits          5  min_p bits
//    6  uniform bits         7  threshold bits (descent accumulator)
//    8  cap = effective k    9  strict_total       10  tied_total
//   11  round (descent rounds done)
//   12  num_candidates      13  survived (count strictly above the prefix)
//   14  pad                 15  pad

#include <metal_stdlib>
#include "baked.h"
using namespace metal;

SCRATCHY_CONSTANT(uint, SAMPLER_VOCAB, 0);
// The telemetry spill (`sample_softmax_materialize` / `sample_finalize`): slot 1, compiled in.
#define SAMPLER_TELEMETRY SCRATCHY_CONSTANT_1

#define SAMPLING_BLOCK_SIZE 256
#define MAX_CANDIDATES 1024
#define WARP_SIZE 32
#define NUM_WARPS (SAMPLING_BLOCK_SIZE / WARP_SIZE)
#define HIST_BUCKETS 256
#define ROW_STATE_LEN 16

// The block reductions split the threadgroup into whole 32-lane simdgroups
// (warp_id = tid / WARP_SIZE, lane_id = tid % WARP_SIZE) and park one partial
// per simdgroup in a NUM_WARPS-slot buffer. That math is only valid if the
// block is an exact multiple of the simdgroup width — guard it at compile
// time. (The matching runtime fact "this device's simdgroup IS 32 wide" is
// not a compile-time constant, so `SamplerKernels::new` checks the pipeline's
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

inline uint warp_reduce_sum_u32(uint val) {
    for (ushort offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += simd_shuffle_xor(val, offset);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Block-level reductions (all threads get the result) — port of the cuda
// `block_reduce_*`: reduce within each warp, park each warp's partial in
// `warp_buf`, then warp 0 reduces the partials.
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

inline uint block_reduce_sum_u32(uint val, threadgroup uint* warp_buf, uint tid) {
    uint warp_id = tid / WARP_SIZE;
    uint lane_id = tid % WARP_SIZE;

    val = warp_reduce_sum_u32(val);
    if (lane_id == 0) warp_buf[warp_id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < WARP_SIZE) {
        uint v = (tid < NUM_WARPS) ? warp_buf[tid] : 0u;
        v = warp_reduce_sum_u32(v);
        if (tid == 0) warp_buf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint result = warp_buf[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

// ---------------------------------------------------------------------------
// Slice bounds: slice s of a vocab axis owns [s*vocab/nslices, (s+1)*…).
// 64-bit intermediates: vocab * slice can exceed 2^32 for large vocabs.
// ---------------------------------------------------------------------------

inline uint2 slice_bounds(uint vocab, uint nslices, uint slice) {
    return uint2((uint)((uint64_t)slice * vocab / nslices),
                 (uint)((uint64_t)(slice + 1) * vocab / nslices));
}

// ---------------------------------------------------------------------------
// Row-gather cast: for row r, slice s, copy logits[row_indices[r], s's slice]
// converted to f32 into out[r, slice]. `nslices` threadgroups per row so the
// copy uses the whole GPU instead of one core per row. Mirrors cuda
// `cast_to_f32_kernel` with a per-row source index.
// ---------------------------------------------------------------------------

template <typename T>
[[kernel]] void cast_rows(
    device       float* out          [[buffer(0)]],
    device const T*     logits       [[buffer(1)]],
    device const uint*  row_indices  [[buffer(2)]],
    constant     uint*  consts       [[buffer(3)]],
    uint tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint vocab = SAMPLER_VOCAB;
    uint nslices = consts[0];
    uint row = tgp / nslices;
    uint slice = tgp % nslices;
    uint src_row = row_indices[row];
    device const T* in = logits + (size_t)src_row * vocab;
    device float* dst = out + (size_t)row * vocab;
    uint2 b = slice_bounds(vocab, nslices, slice);
    for (uint i = b.x + tid; i < b.y; i += SAMPLING_BLOCK_SIZE) {
        dst[i] = float(in[i]);
    }
}

SCRATCHY_KERNEL(cast_rows_f16_to_f32, cast_rows<half>)
SCRATCHY_KERNEL(cast_rows_bf16_to_f32, cast_rows<bfloat>)
// Identity row-gather for f32 logits — the parity harness's metal side feeds
// f32 rows directly (the model's own cast is one of the f16/bf16 kernels
// above; this one exists so the same pipeline can be tested on host data).
SCRATCHY_KERNEL(cast_rows_f32_to_f32, cast_rows<float>)

// ---------------------------------------------------------------------------
// apply_penalties — repetition / frequency / presence, sliced over the vocab
// axis. Semantics unchanged from the cuda `apply_penalties_kernel`:
//   count = occurrences in output_token_ids[row] + prompt_token_ids[row]
//   if count > 0: logit = logit > 0 ? logit / rep : logit * rep;
//                 logit -= freq * count + pres
// Token id == vocab_size is padding (never matches a real vocab index).
// ---------------------------------------------------------------------------

#if SCRATCHY_COMPILES(apply_penalties)
kernel void apply_penalties(
    device       float* logits            [[buffer(0)]],
    device const int*   output_token_ids  [[buffer(1)]],
    device const int*   prompt_token_ids  [[buffer(2)]],
    device const float* rep_penalties     [[buffer(3)]],
    device const float* freq_penalties    [[buffer(4)]],
    device const float* pres_penalties    [[buffer(5)]],
    constant     uint*  consts            [[buffer(6)]],
    uint tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint vocab = SAMPLER_VOCAB;
    uint nslices = consts[0];
    uint max_output_len = consts[2];
    uint max_prompt_len = consts[3];
    uint row = tgp / nslices;
    uint slice = tgp % nslices;

    float rep_pen  = rep_penalties[row];
    float freq_pen = freq_penalties[row];
    float pres_pen = pres_penalties[row];

    device float* rowp          = logits + (size_t)row * vocab;
    device const int* out_ids    = output_token_ids + (size_t)row * max_output_len;
    device const int* prompt_ids = prompt_token_ids + (size_t)row * max_prompt_len;

    uint2 b = slice_bounds(vocab, nslices, slice);
    for (uint v = b.x + tid; v < b.y; v += SAMPLING_BLOCK_SIZE) {
        int count = 0;
        for (uint j = 0; j < max_output_len; j++) {
            if (out_ids[j] == (int)v) count++;
        }
        for (uint j = 0; j < max_prompt_len; j++) {
            if (prompt_ids[j] == (int)v) count++;
        }

        if (count > 0) {
            float logit = rowp[v];
            if (logit > 0.0f) {
                logit /= rep_pen;
            } else {
                logit *= rep_pen;
            }
            logit -= freq_pen * (float)count + pres_pen;
            rowp[v] = logit;
        }
    }
}
#endif

// ---------------------------------------------------------------------------
// sample_softmax_reduce — per (row, slice): the slice's max(logit/T) and the
// partial sum exp(v - slice_max). `sample_stats_pick` rescales each partial
// against the row max, so the row sum equals the single-threadgroup value up
// to reduction order (the parity harness's robustness margins exist for
// exactly this class of few-ULP difference).
// ---------------------------------------------------------------------------

#if SCRATCHY_COMPILES(sample_softmax_reduce)
kernel void sample_softmax_reduce(
    device       float* logits_all [[buffer(0)]],
    device       uint*  partials   [[buffer(1)]],
    device const uint*  row_state  [[buffer(2)]],
    constant     uint*  consts     [[buffer(3)]],
    uint tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint vocab = SAMPLER_VOCAB;
    uint nslices = consts[0];
    uint row = tgp / nslices;
    uint slice = tgp % nslices;
    device const uint* state = row_state + (size_t)row * ROW_STATE_LEN;
    device float* rowp = logits_all + (size_t)row * vocab;
    uint2 b = slice_bounds(vocab, nslices, slice);

    threadgroup float s_warp_buf[NUM_WARPS];
    float inv_temp = 1.0f / as_type<float>(state[0]);

    float local_max = -INFINITY;
    for (uint i = b.x + tid; i < b.y; i += SAMPLING_BLOCK_SIZE) {
        local_max = max(local_max, rowp[i] * inv_temp);
    }
    float slice_max = block_reduce_max(local_max, s_warp_buf, tid);

    float local_sum = 0.0f;
    for (uint i = b.x + tid; i < b.y; i += SAMPLING_BLOCK_SIZE) {
        local_sum += exp(rowp[i] * inv_temp - slice_max);
    }
    float slice_sum = block_reduce_sum(local_sum, s_warp_buf, tid);

    if (tid == 0) {
        device uint* p = partials + ((size_t)row * nslices + slice) * 3;
        p[0] = as_type<uint>(slice_max);
        p[1] = as_type<uint>(slice_sum);
    }
}
#endif

// ---------------------------------------------------------------------------
// sample_stats_pick — ONE threadgroup per row: merge the slice partials into
// the row's max and sum (rescaling each slice's partial sum by
// exp(slice_max - row_max)), then reset the descent state. 256 threads ≥
// nslices in every configuration the host builds; extra threads contribute
// -inf / 0.
// ---------------------------------------------------------------------------

#if SCRATCHY_COMPILES(sample_stats_pick)
kernel void sample_stats_pick(
    device const uint*  partials  [[buffer(0)]],
    device       uint*  row_state [[buffer(1)]],
    constant     uint*  consts    [[buffer(2)]],
    uint tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint nslices = consts[0];
    uint nrows = consts[1];
    uint row = min(tgp, nrows - 1);
    device uint* state = row_state + (size_t)row * ROW_STATE_LEN;
    device const uint* p = partials + (size_t)row * nslices * 3;

    float my_max = -INFINITY;
    float my_sum = 0.0f;
    if (tid < nslices) {
        my_max = as_type<float>(p[tid * 3 + 0]);
        my_sum = as_type<float>(p[tid * 3 + 1]);
    }

    threadgroup float s_warp_buf[NUM_WARPS];
    float row_max = block_reduce_max(my_max, s_warp_buf, tid);

    float rescaled = (my_max == -INFINITY) ? 0.0f : my_sum * exp(my_max - row_max);
    float row_sum = block_reduce_sum(rescaled, s_warp_buf, tid);

    if (tid == 0) {
        state[1] = as_type<uint>(row_max);
        state[2] = as_type<uint>(row_sum);
        state[7] = 0;   // threshold accumulator
        state[11] = 0;  // round
        state[13] = 0;  // survivors above the prefix
    }
}
#endif

// ---------------------------------------------------------------------------
// sample_softmax_materialize — per (row, slice): overwrite the f32 scratch
// with the softmax PROBABILITY BITS — the exact values the cuda radix loop
// compared on every one of its 32 passes, computed here once:
//   bits(i) = as_uint(exp(v(i) - max) * inv_sum)
// ---------------------------------------------------------------------------

#if SCRATCHY_COMPILES(sample_softmax_materialize)
kernel void sample_softmax_materialize(
    device       float* logits_all [[buffer(0)]],
    device       uint*  partials   [[buffer(1)]],
    device const uint*  row_state  [[buffer(2)]],
    constant     uint*  consts     [[buffer(3)]],
    uint tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint vocab = SAMPLER_VOCAB;
    uint nslices = consts[0];
    uint row = tgp / nslices;
    uint slice = tgp % nslices;
    device const uint* state = row_state + (size_t)row * ROW_STATE_LEN;
    device float* rowp = logits_all + (size_t)row * vocab;
    uint2 b = slice_bounds(vocab, nslices, slice);

    float inv_temp = 1.0f / as_type<float>(state[0]);
    float max_logit = as_type<float>(state[1]);
    float inv_sum_exp = 1.0f / as_type<float>(state[2]);
    device uint* bits = (device uint*)(rowp);

#if SAMPLER_TELEMETRY
    // Entropy partial over this slice (merged by sample_finalize).
    float local_ent = 0.0f;
#endif
    for (uint i = b.x + tid; i < b.y; i += SAMPLING_BLOCK_SIZE) {
        float val = rowp[i] * inv_temp;
        float prob = exp(val - max_logit) * inv_sum_exp;
        bits[i] = as_type<uint>(prob);
#if SAMPLER_TELEMETRY
        if (prob > 0.0f) {
            local_ent -= prob * log(prob);
        }
#endif
    }
#if SAMPLER_TELEMETRY
    threadgroup float s_warp_buf[NUM_WARPS];
    float ent = block_reduce_sum(local_ent, s_warp_buf, tid);
    if (tid == 0) {
        partials[((size_t)row * nslices + slice) * 3 + 2] = as_type<uint>(ent);
    }
#endif
}
#endif

// ---------------------------------------------------------------------------
// sample_histogram_pass — per (row, slice): a 256-bucket histogram of byte
// (3 - round) of the prob bits, over the round's SURVIVORS — elements whose
// already-fixed high bytes equal the threshold's high bytes so far (round and
// threshold both live in row_state, so this kernel's bindings never change
// across the four rounds; only the data does).
// ---------------------------------------------------------------------------

#if SCRATCHY_COMPILES(sample_histogram_pass)
kernel void sample_histogram_pass(
    device const uint*  prob_bits [[buffer(0)]],
    device       uint*  hist      [[buffer(1)]],
    device const uint*  row_state [[buffer(2)]],
    constant     uint*  consts    [[buffer(3)]],
    uint tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint vocab = SAMPLER_VOCAB;
    uint nslices = consts[0];
    uint row = tgp / nslices;
    uint slice = tgp % nslices;
    device const uint* state = row_state + (size_t)row * ROW_STATE_LEN;
    uint round = state[11];  // 0..3; byte (3 - round) is histogrammed next
    device const uint* rowp = prob_bits + (size_t)row * vocab;
    uint2 b = slice_bounds(vocab, nslices, slice);

    // The fixed high bytes: `round` bytes above the histogrammed byte are
    // fixed; the mask keeps them and zeroes the rest.
    uint shift = (3 - round) * 8;
    uint mask = (shift + 8 >= 32) ? 0u : (~0u << (shift + 8));
    uint prefix = state[7] & mask;

    threadgroup atomic_uint local_hist[HIST_BUCKETS];
    for (uint bk = tid; bk < HIST_BUCKETS; bk += SAMPLING_BLOCK_SIZE) {
        atomic_store_explicit(local_hist + bk, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = b.x + tid; i < b.y; i += SAMPLING_BLOCK_SIZE) {
        uint x = rowp[i];
        if ((x & mask) == prefix) {
            uint bucket = (x >> shift) & 0xFFu;
            atomic_fetch_add_explicit(local_hist + bucket, 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device uint* dst = hist + ((size_t)row * nslices + slice) * HIST_BUCKETS;
    for (uint bk = tid; bk < HIST_BUCKETS; bk += SAMPLING_BLOCK_SIZE) {
        dst[bk] = atomic_load_explicit(local_hist + bk, memory_order_relaxed);
    }
    (void)dst;
}
#endif

// ---------------------------------------------------------------------------
// sample_threshold_pick — ONE threadgroup per row, dispatched once per byte
// round (after each histogram pass). Merges the slices' histograms, picks
// this round's byte — the largest b whose suffix count plus the survivors
// above the prefix reaches k, exactly the bit-serial greedy predicate
// evaluated a byte at a time — and advances the round. The fourth round also
// applies the min-p floor over the completed threshold.
// ---------------------------------------------------------------------------

#if SCRATCHY_COMPILES(sample_threshold_pick)
kernel void sample_threshold_pick(
    device const uint*  hist      [[buffer(0)]],
    device       uint*  row_state [[buffer(1)]],
    constant     uint*  consts    [[buffer(2)]],
    uint tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint nslices = consts[0];
    uint nrows = consts[1];
    uint row = min(tgp, nrows - 1);
    device uint* state = row_state + (size_t)row * ROW_STATE_LEN;
    device const uint* h = hist + (size_t)row * nslices * HIST_BUCKETS;

    // Merge: thread tid owns bucket tid, sums it across slices.
    threadgroup uint merged[HIST_BUCKETS];
    if (tid < HIST_BUCKETS) {
        uint acc = 0;
        for (uint s = 0; s < nslices; s++) {
            acc += h[(size_t)s * HIST_BUCKETS + tid];
        }
        merged[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint round = state[11];
        uint k = state[8];             // effective k (host-written)
        uint threshold = state[7];
        uint survived = state[13];
        uint shift = (3 - round) * 8;

        // Largest byte b such that (elements with byte >= b) + survived >= k.
        // The loop always breaks: at b = 0 the count is every prefix-matching
        // element, and the previous round guaranteed that count (plus
        // survived) reaches k.
        uint pick = 0;
        uint suffix = 0;  // count of prefix-matching elements with byte > pick
        uint run = 0;
        for (int bb = 255; bb >= 0; bb--) {
            run += merged[bb];
            if ((uint64_t)run + survived >= (uint64_t)k) {
                pick = (uint)bb;
                suffix = run - merged[bb];
                break;
            }
        }
        threshold |= pick << shift;
        survived += suffix;
        state[7] = threshold;
        state[13] = survived;
        state[11] = round + 1;

        if (round == 3) {
            // The threshold is complete; apply the min-p floor (cuda phase 3b).
            float threshold_prob = as_type<float>(threshold);
            float min_p = as_type<float>(state[5]);
            if (min_p > 0.0f) {
                float inv_sum_exp = 1.0f / as_type<float>(state[2]);
                float max_prob = inv_sum_exp;  // exp(0) * inv_sum_exp
                threshold_prob = max(threshold_prob, min_p * max_prob);
            }
            state[7] = as_type<uint>(threshold_prob);
        }
    }
}
#endif

// ---------------------------------------------------------------------------
// sample_count_compact — per (row, slice): count this slice's strict
// (bits > threshold) and tied (bits == threshold) elements exactly, and
// append the strict candidates to the slice's staging run. The staging cap
// matches the cuda kernel's `pos < cap` drop.
// ---------------------------------------------------------------------------

#if SCRATCHY_COMPILES(sample_count_compact)
kernel void sample_count_compact(
    device const uint*  prob_bits [[buffer(0)]],
    device       uint*  counts    [[buffer(1)]],
    device       uint*  staging   [[buffer(2)]],
    device const uint*  row_state [[buffer(3)]],
    constant     uint*  consts    [[buffer(4)]],
    uint tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint vocab = SAMPLER_VOCAB;
    uint nslices = consts[0];
    uint row = tgp / nslices;
    uint slice = tgp % nslices;
    device const uint* state = row_state + (size_t)row * ROW_STATE_LEN;
    device const uint* rowp = prob_bits + (size_t)row * vocab;
    uint2 b = slice_bounds(vocab, nslices, slice);
    uint threshold = state[7];

    device uint* c = counts + ((size_t)row * nslices + slice) * 4;
    device uint* stage = staging
        + ((size_t)row * nslices + slice) * 2 * MAX_CANDIDATES;

    threadgroup uint s_warp_buf[NUM_WARPS];
    threadgroup atomic_uint s_next;
    if (tid == 0) atomic_store_explicit(&s_next, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint my_strict = 0;
    uint my_tied = 0;
    for (uint i = b.x + tid; i < b.y; i += SAMPLING_BLOCK_SIZE) {
        uint x = rowp[i];
        if (x > threshold) {
            my_strict++;
            uint pos = atomic_fetch_add_explicit(&s_next, 1u, memory_order_relaxed);
            if (pos < MAX_CANDIDATES) {
                stage[2 * pos + 0] = x;
                stage[2 * pos + 1] = i;
            }
        } else if (x == threshold) {
            my_tied++;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint strict_total = block_reduce_sum_u32(my_strict, s_warp_buf, tid);
    uint tied_total = block_reduce_sum_u32(my_tied, s_warp_buf, tid);
    if (tid == 0) {
        c[0] = strict_total;
        c[1] = tied_total;
        c[2] = min(atomic_load_explicit(&s_next, memory_order_relaxed),
                   (uint)MAX_CANDIDATES);  // this slice's staged strict run
    }
}
#endif

// ---------------------------------------------------------------------------
// sample_quota_pick — ONE threadgroup per row: total the slices' strict
// counts, compute each slice's share of the tie slots (slices in index
// order — the cuda kernel's two-phase fill order), and freeze
// num_candidates.
// ---------------------------------------------------------------------------

#if SCRATCHY_COMPILES(sample_quota_pick)
kernel void sample_quota_pick(
    device       uint* counts    [[buffer(0)]],
    device       uint* row_state [[buffer(1)]],
    constant     uint* consts    [[buffer(2)]],
    uint tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint nslices = consts[0];
    uint nrows = consts[1];
    uint row = min(tgp, nrows - 1);
    device uint* state = row_state + (size_t)row * ROW_STATE_LEN;
    device const uint* c = counts + (size_t)row * nslices * 4;

    if (tid == 0) {
        uint cap = state[8];
        uint strict_total = 0;
        uint tied_total = 0;
        for (uint s = 0; s < nslices; s++) {
            strict_total += c[s * 4 + 0];
            tied_total += c[s * 4 + 1];
        }
        uint strict_kept = min(strict_total, cap);
        uint tie_slots = cap - strict_kept;
        uint tied_avail = min(tied_total, tie_slots);

        // Per-slice tie quotas, slices in index order (counts[s, 3]).
        uint remaining = tied_avail;
        for (uint s = 0; s < nslices; s++) {
            uint take = min(c[s * 4 + 1], remaining);
            counts[(size_t)row * nslices * 4 + s * 4 + 3] = take;
            remaining -= take;
        }
        state[9] = strict_kept;
        state[10] = tied_avail;
        state[12] = strict_kept + tied_avail;
    }
}
#endif

// ---------------------------------------------------------------------------
// sample_compact_tied — per (row, slice): append this slice's tied
// candidates after its strict run, up to the slice's quota.
// ---------------------------------------------------------------------------

#if SCRATCHY_COMPILES(sample_compact_tied)
kernel void sample_compact_tied(
    device const uint*  prob_bits [[buffer(0)]],
    device       uint*  counts    [[buffer(1)]],
    device       uint*  staging   [[buffer(2)]],
    device const uint*  row_state [[buffer(3)]],
    constant     uint*  consts    [[buffer(4)]],
    uint tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint vocab = SAMPLER_VOCAB;
    uint nslices = consts[0];
    uint row = tgp / nslices;
    uint slice = tgp % nslices;
    device const uint* state = row_state + (size_t)row * ROW_STATE_LEN;
    device const uint* rowp = prob_bits + (size_t)row * vocab;
    uint2 b = slice_bounds(vocab, nslices, slice);
    uint threshold = state[7];

    device uint* c = counts + ((size_t)row * nslices + slice) * 4;
    device uint* stage = staging
        + ((size_t)row * nslices + slice) * 2 * MAX_CANDIDATES;
    uint strict_staged = c[2];
    uint quota = c[3];

    threadgroup atomic_uint s_next;
    if (tid == 0) atomic_store_explicit(&s_next, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = b.x + tid; i < b.y; i += SAMPLING_BLOCK_SIZE) {
        if (rowp[i] == threshold) {
            uint pos = atomic_fetch_add_explicit(&s_next, 1u, memory_order_relaxed);
            if (pos < quota) {
                stage[2 * (strict_staged + pos) + 0] = threshold;
                stage[2 * (strict_staged + pos) + 1] = i;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint taken = min(atomic_load_explicit(&s_next, memory_order_relaxed), quota);
        c[2] = strict_staged + taken;  // the slice's full staging run length
    }
}
#endif

// ---------------------------------------------------------------------------
// sample_finalize — ONE threadgroup per row: gather the slices' staging runs
// (strict pairs then tied pairs, slices in order — the cuda kernel's
// two-phase fill), then the cuda tail: bitonic sort descending by prob,
// top-p cutoff, renormalize, categorical draw.
//
// num_candidates >= 1 always: the threshold is at most the largest prob bits
// value in the row (the min-p floor is min_p * max_prob <= max_prob), so at
// least the argmax element is strict or tied. The fallback keeps the cuda
// kernel's shape anyway.
// ---------------------------------------------------------------------------

#if SCRATCHY_COMPILES(sample_finalize)
kernel void sample_finalize(
    device const uint*  staging   [[buffer(0)]],
    device const uint*  counts    [[buffer(1)]],
    device const uint*  row_state [[buffer(2)]],
    device const uint*  prob_bits [[buffer(3)]],
    device       uint*  output    [[buffer(4)]],
#if SAMPLER_TELEMETRY
    device const uint*  partials     [[buffer(5)]],
    device       float* topk_probs   [[buffer(6)]],
    device       uint*  topk_indices [[buffer(7)]],
    device       float* stats_out    [[buffer(8)]],
    constant     uint*  telem_consts [[buffer(9)]],
#endif
    constant     uint*  consts    [[buffer(10)]],
    uint tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint vocab = SAMPLER_VOCAB;
    uint nslices = consts[0];
    uint nrows = consts[1];
    uint row = min(tgp, nrows - 1);
    device const uint* state = row_state + (size_t)row * ROW_STATE_LEN;
    device const uint* c = counts + (size_t)row * nslices * 4;

    threadgroup float s_probs[MAX_CANDIDATES];
    threadgroup uint  s_indices[MAX_CANDIDATES];

    // Gather the slices' runs in order.
    threadgroup atomic_uint s_cursor;
    if (tid == 0) atomic_store_explicit(&s_cursor, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0; s < nslices; s++) {
        uint len = c[s * 4 + 2];
        device const uint* stage =
            staging + ((size_t)row * nslices + s) * 2 * MAX_CANDIDATES;
        for (uint j = tid; j < len; j += SAMPLING_BLOCK_SIZE) {
            uint pos = atomic_fetch_add_explicit(&s_cursor, 1u, memory_order_relaxed);
            if (pos < MAX_CANDIDATES) {
                s_probs[pos] = as_type<float>(stage[2 * j + 0]);
                s_indices[pos] = stage[2 * j + 1];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    int num_candidates = (int)min(
        atomic_load_explicit(&s_cursor, memory_order_relaxed),
        (uint)MAX_CANDIDATES);

    float top_p = as_type<float>(state[4]);
    float uniform_random = as_type<float>(state[6]);

    if (num_candidates <= 0) {
        // Fallback (matches the cuda kernel's): argmax over the row.
        if (tid == 0) {
            uint best_bits = 0;
            uint best_idx = 0;
            device const uint* rowp = prob_bits + (size_t)row * vocab;
            for (uint i = 0; i < vocab; i++) {
                if (rowp[i] > best_bits) {
                    best_bits = rowp[i];
                    best_idx = i;
                }
            }
            output[row] = best_idx;
        }
        return;
    }

    // ===== Bitonic sort, descending by prob (cuda phase 5) =====
    int n_padded = 1;
    while (n_padded < num_candidates) n_padded <<= 1;
    for (int i = tid + num_candidates; i < n_padded; i += SAMPLING_BLOCK_SIZE) {
        s_probs[i] = 0.0f;
        s_indices[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

    // ===== Top-p cutoff + renormalize + sample (cuda phase 6, thread 0) =====
    if (tid == 0) {
        float cumsum = 0.0f;
        int cutoff = num_candidates;
        for (int i = 0; i < num_candidates; i++) {
            cumsum += s_probs[i];
            if (cumsum > top_p) {
                cutoff = i + 1;
                break;
            }
        }

        float total = 0.0f;
        for (int i = 0; i < cutoff; i++) {
            total += s_probs[i];
        }

        float target = uniform_random * total;
        cumsum = 0.0f;
        uint sampled = s_indices[cutoff - 1];
        for (int i = 0; i < cutoff; i++) {
            cumsum += s_probs[i];
            if (cumsum >= target) {
                sampled = s_indices[i];
                break;
            }
        }
        output[row] = sampled;

#if SAMPLER_TELEMETRY
        uint telem_on = telem_consts[0];
        if (telem_on != 0u) {
            uint telem_k = telem_consts[1];
            uint base = row * telem_k;
            for (uint i = 0; i < telem_k; i++) {
                bool valid = (int)i < num_candidates;
                topk_probs[base + i] = valid ? s_probs[i] : 0.0f;
                topk_indices[base + i] = valid ? s_indices[i] : 0u;
            }
            float inv_sum_exp = 1.0f / as_type<float>(state[2]);
            stats_out[row * 2u + 0u] = inv_sum_exp;  // max_prob
            // Entropy: merge the materialize pass's per-slice partials.
            float ent = 0.0f;
            for (uint s = 0; s < nslices; s++) {
                ent += as_type<float>(partials[((size_t)row * nslices + s) * 3 + 2]);
            }
            stats_out[row * 2u + 1u] = ent;
        }
#endif
    }
}
#endif
