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
// and replaces the 32 bit-serial passes with a 3-round radix descent over
// 11/11/10-bit digits that provably selects the same threshold: the
// bit-serial greedy keeps threshold bit b iff count(bits >= candidate) >= k,
// which over monotonically-ordered nonneg-float bits equals picking, per digit
// from the top, the largest d whose suffix count reaches k. Every cross-slice
// decision (the row's softmax stats, each round's digit, the tie quotas) is
// made by the next pass's threadgroups themselves from what the pass before
// left in device memory, so the pipeline is six dispatches riding one command
// buffer with Device barriers; all inter-kernel state lives in a per-row
// `row_state` block.
//
// Pipeline (host dispatch order):
//   sample_softmax_reduce_{f16,bf16}
//                                (row × slice)  logits row, penalized → f32
//                                              scratch; per-slice (max, sum)
//                                              partials
//   sample_softmax_materialize   (row × slice)  merge partials → row max/sum;
//                                              probs, as bits, in place;
//                                              round 0's histogram
//   sample_descent_round_{1,2}   (row × slice)  pick the round before;
//                                              this round's histogram
//   sample_count_compact         (row × slice)  pick the last round, min-p;
//                                              per-slice strict/tied counts,
//                                              strict and tied candidates
//   sample_finalize              (row)          tie quotas, gather, bitonic
//                                              sort desc, top-p cutoff,
//                                              categorical draw
//
// Design (matches cuda's f32 "slow" sampling path): only the reduce is
// dtype-specialized (f16/bf16 logits); penalties + sampling are f32-only. The
// per-request `uniform_random` is drawn host-side from the request RNG, so no
// on-GPU Philox is needed.
//
// Sliced kernels use threadgroups = (nrows * nslices, 1, 1),
// threadsPerThreadgroup = (SAMPLING_BLOCK_SIZE, 1, 1); row = tg / nslices,
// slice = tg % nslices, and each threadgroup strides its contiguous slice of
// the vocab axis. `nslices` follows the step's row count (host-chosen, at
// most WARP_SIZE); the vocab is the model's, compiled in (`SAMPLER_VOCAB`).
//
// Each kernel binds its buffers at contiguous slots, in signature order:
//   f32 scratch / prob bits   [nrows, vocab]
//   logits                    [total_n, vocab]   (the reduce reads them)
//   row_indices               [nrows] uint
//   output / prompt token ids [nrows, max_out / max_prompt] (penalties)
//   rep / freq / pres         [nrows] f32
//   row_state                 [nrows, 16] uint   (layout below)
//   partials                  [nrows, nslices, 3] uint
//   hist                      [nrows, DESCENT_ROUNDS, DESCENT_BUCKETS] uint
//   counts                    [nrows, nslices, 4] uint
//   staging                   [nrows, nslices, 4*MAX_CANDIDATES] uint
//   tokens                    [total_n] uint     (the step's argmax; each
//                                                 job's draw overwrites its row)
//   consts                    (nslices, nrows, max_out, max_prompt)
//   telemetry spill           (sampler-telemetry only)
//
// row_state[row] layout (u32 words; host writes 0..8, GPU the rest):
//    0  temperature bits     1  max_logit bits      2  sum_exp bits
//    3  top_k (raw)          4  top_p bits          5  min_p bits
//    6  uniform bits         7  threshold bits (after min-p)
//    8  cap = effective k    9..12  (threshold, survivors) before rounds 1, 2
//   13..15  pad

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
// The radix descent: three digits of the prob bits from the top — bits 31..21,
// 20..10 and 9..0. Round r histograms its digit over the round's survivors
// (elements whose already-fixed high digits equal the threshold's so far) per
// slice in threadgroup memory and adds the slice's counts into the row's
// histogram `hist` [nrows, DESCENT_ROUNDS, DESCENT_BUCKETS], zeroed by the
// reduce; the next pass's threadgroups each pick the round's digit from it.
// row_state[DESCENT_STATE + 2 r - 2 ..] holds (threshold, survivors) before
// round r's pick, for the pass that picks it (round 0 starts from zero).
// ---------------------------------------------------------------------------

#define DESCENT_ROUNDS 3
#define DESCENT_BUCKETS 2048  // 2^11, the widest digit
#define DESCENT_PER_THREAD (DESCENT_BUCKETS / SAMPLING_BLOCK_SIZE)
#define DESCENT_STATE 9

// Round r's digit is bits [DIGIT_SHIFT[r], DIGIT_SHIFT[r - 1]) (32 for r = 0).
constant uint DIGIT_SHIFT[DESCENT_ROUNDS] = {21, 10, 0};

inline uint digit_mask(uint round) {
    return (1u << ((round == 0 ? 32u : DIGIT_SHIFT[round - 1]) - DIGIT_SHIFT[round])) - 1u;
}

// The fixed high digits before round r.
inline uint prefix_mask(uint round) {
    return round == 0 ? 0u : ~0u << DIGIT_SHIFT[round - 1];
}

inline device uint* descent_hist(device uint* hist, uint row, uint round) {
    return hist + ((size_t)row * DESCENT_ROUNDS + round) * DESCENT_BUCKETS;
}

inline uint2 descent_state(device const uint* state, uint round) {
    return round == 0 ? uint2(0u, 0u)
                      : uint2(state[DESCENT_STATE + 2 * round - 2],
                              state[DESCENT_STATE + 2 * round - 1]);
}

// A slice's histogram of the round's digits, in threadgroup memory.
inline void descent_clear(threadgroup atomic_uint* local, uint tid) {
    for (uint j = 0; j < DESCENT_PER_THREAD; j++) {
        atomic_store_explicit(local + j * SAMPLING_BLOCK_SIZE + tid, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Add the slice's histogram into the row's.
inline void descent_flush(threadgroup atomic_uint* local, device uint* row_hist, uint tid) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < DESCENT_PER_THREAD; j++) {
        uint b = j * SAMPLING_BLOCK_SIZE + tid;
        uint n = atomic_load_explicit(local + b, memory_order_relaxed);
        if (n != 0) {
            atomic_fetch_add_explicit((device atomic_uint*)(row_hist + b), n,
                                      memory_order_relaxed);
        }
    }
}

// Pick the round's digit from the row's histogram `h` (thread tid owns digits
// [tid * DESCENT_PER_THREAD, ...)): the largest d whose suffix count (the
// round's survivors with digit >= d) plus the survivors above the prefix
// reaches k — exactly the bit-serial greedy predicate evaluated a digit at a
// time. The counts are non-increasing in d, so exactly one digit is the
// largest that reaches k (round 0 counts every element, and each pick leaves
// the next round's survivors reaching k); its thread publishes it. `prior`
// and the result are (threshold, survivors above the prefix).
inline uint2 descent_pick(device const uint* h, uint round, uint k, uint2 prior,
                          threadgroup uint* warp_buf, threadgroup uint2* pick, uint tid) {
    uint count[DESCENT_PER_THREAD];
    uint mine = 0;
    for (uint j = 0; j < DESCENT_PER_THREAD; j++) {
        count[j] = h[tid * DESCENT_PER_THREAD + j];
        mine += count[j];
    }
    uint warp_id = tid / WARP_SIZE;
    uint warp_total = simd_sum(mine);
    if (tid % WARP_SIZE == 0) warp_buf[warp_id] = warp_total;
    if (tid == 0) *pick = prior;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint greater = warp_total - simd_prefix_inclusive_sum(mine);
    for (uint w = warp_id + 1; w < NUM_WARPS; w++) {
        greater += warp_buf[w];
    }
    for (int j = DESCENT_PER_THREAD - 1; j >= 0; j--) {
        uint at_least = greater + count[j];
        if ((uint64_t)at_least + prior.y >= (uint64_t)k
            && (uint64_t)greater + prior.y < (uint64_t)k) {
            uint digit = tid * DESCENT_PER_THREAD + (uint)j;
            *pick = uint2(prior.x | (digit << DIGIT_SHIFT[round]), prior.y + greater);
        }
        greater = at_least;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint2 result = *pick;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

// ---------------------------------------------------------------------------
// sample_softmax_reduce — per (row, slice): gather the row's logits
// (logits[row_indices[r]], mirrors cuda `cast_to_f32_kernel` with a per-row
// source index), apply the repetition / frequency / presence penalties
// (cuda `apply_penalties_kernel`), write the f32 result to the scratch, and
// reduce the slice's max(logit/T) and partial sum exp(v - slice_max).
// `sample_softmax_materialize` rescales each partial against the row max, so
// the row sum equals the single-threadgroup value up to reduction order (the
// parity harness's robustness margins exist for exactly this class of
// few-ULP difference). It also zeroes the row's descent histograms.
// Penalties, unchanged from cuda:
//   count = occurrences in output_token_ids[row] + prompt_token_ids[row]
//   if count > 0: logit = logit > 0 ? logit / rep : logit * rep;
//                 logit -= freq * count + pres
// Token id == vocab_size is padding (never matches a real vocab index); a
// step without penalties has empty histories (consts[2], consts[3] = 0).
// ---------------------------------------------------------------------------

template <typename T>
[[kernel]] void sample_softmax_reduce(
    device       float* scratch           [[buffer(0)]],
    device const T*     logits            [[buffer(1)]],
    device const uint*  row_indices       [[buffer(2)]],
    device const int*   output_token_ids  [[buffer(3)]],
    device const int*   prompt_token_ids  [[buffer(4)]],
    device const float* rep_penalties     [[buffer(5)]],
    device const float* freq_penalties    [[buffer(6)]],
    device const float* pres_penalties    [[buffer(7)]],
    device const uint*  row_state         [[buffer(8)]],
    device       uint*  partials          [[buffer(9)]],
    constant     uint*  consts            [[buffer(10)]],
    device       uint*  hist              [[buffer(11)]],
    uint tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint vocab = SAMPLER_VOCAB;
    uint nslices = consts[0];
    uint max_output_len = consts[2];
    uint max_prompt_len = consts[3];
    uint row = tgp / nslices;
    uint slice = tgp % nslices;
    device const uint* state = row_state + (size_t)row * ROW_STATE_LEN;
    device const T* in = logits + (size_t)row_indices[row] * vocab;
    device float* rowp = scratch + (size_t)row * vocab;
    device const int* out_ids    = output_token_ids + (size_t)row * max_output_len;
    device const int* prompt_ids = prompt_token_ids + (size_t)row * max_prompt_len;
    float rep_pen  = rep_penalties[row];
    float freq_pen = freq_penalties[row];
    float pres_pen = pres_penalties[row];
    uint2 b = slice_bounds(vocab, nslices, slice);
    for (uint i = slice * SAMPLING_BLOCK_SIZE + tid; i < DESCENT_ROUNDS * DESCENT_BUCKETS;
         i += nslices * SAMPLING_BLOCK_SIZE) {
        descent_hist(hist, row, 0)[i] = 0;
    }

    threadgroup float s_warp_buf[NUM_WARPS];
    float inv_temp = 1.0f / as_type<float>(state[0]);

    float local_max = -INFINITY;
    for (uint v = b.x + tid; v < b.y; v += SAMPLING_BLOCK_SIZE) {
        float logit = float(in[v]);
        int count = 0;
        for (uint j = 0; j < max_output_len; j++) {
            if (out_ids[j] == (int)v) count++;
        }
        for (uint j = 0; j < max_prompt_len; j++) {
            if (prompt_ids[j] == (int)v) count++;
        }
        if (count > 0) {
            if (logit > 0.0f) {
                logit /= rep_pen;
            } else {
                logit *= rep_pen;
            }
            logit -= freq_pen * (float)count + pres_pen;
        }
        rowp[v] = logit;
        local_max = max(local_max, logit * inv_temp);
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

SCRATCHY_KERNEL(sample_softmax_reduce_f16, sample_softmax_reduce<half>)
SCRATCHY_KERNEL(sample_softmax_reduce_bf16, sample_softmax_reduce<bfloat>)
// f32 logits — the parity harness's metal side feeds f32 rows directly (the
// model's own logits are f16/bf16); the same pipeline then runs on host data.
SCRATCHY_KERNEL(sample_softmax_reduce_f32, sample_softmax_reduce<float>)

// ---------------------------------------------------------------------------
// sample_softmax_materialize — per (row, slice): merge the slice partials into
// the row's max and sum (each slice's partial sum rescaled by
// exp(slice_max - row_max); every threadgroup merges them itself — 256
// threads >= nslices, the extra threads contributing -inf / 0), then
// overwrite the f32 scratch with the softmax PROBABILITY BITS — the exact
// values the cuda radix loop compared on every one of its 32 passes,
// computed here once:
//   bits(i) = as_uint(exp(v(i) - max) * inv_sum)
// — and histogram the descent's round-0 digit. Slice 0 records the row's max
// and sum.
// ---------------------------------------------------------------------------

#if SCRATCHY_COMPILES(sample_softmax_materialize)
kernel void sample_softmax_materialize(
    device       float* logits_all [[buffer(0)]],
    device       uint*  partials   [[buffer(1)]],
    device       uint*  row_state  [[buffer(2)]],
    device       uint*  hist       [[buffer(3)]],
    constant     uint*  consts     [[buffer(4)]],
    uint tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint vocab = SAMPLER_VOCAB;
    uint nslices = consts[0];
    uint row = tgp / nslices;
    uint slice = tgp % nslices;
    device uint* state = row_state + (size_t)row * ROW_STATE_LEN;
    device float* rowp = logits_all + (size_t)row * vocab;
    device const uint* p = partials + (size_t)row * nslices * 3;
    uint2 b = slice_bounds(vocab, nslices, slice);

    float my_max = -INFINITY;
    float my_sum = 0.0f;
    if (tid < nslices) {
        my_max = as_type<float>(p[tid * 3 + 0]);
        my_sum = as_type<float>(p[tid * 3 + 1]);
    }
    threadgroup float s_warp_buf[NUM_WARPS];
    float max_logit = block_reduce_max(my_max, s_warp_buf, tid);
    float rescaled = (my_max == -INFINITY) ? 0.0f : my_sum * exp(my_max - max_logit);
    float sum_exp = block_reduce_sum(rescaled, s_warp_buf, tid);
    if (slice == 0 && tid == 0) {
        state[1] = as_type<uint>(max_logit);
        state[2] = as_type<uint>(sum_exp);
    }

    float inv_temp = 1.0f / as_type<float>(state[0]);
    float inv_sum_exp = 1.0f / sum_exp;
    device uint* bits = (device uint*)(rowp);
    threadgroup atomic_uint local_hist[DESCENT_BUCKETS];
    descent_clear(local_hist, tid);

#if SAMPLER_TELEMETRY
    // Entropy partial over this slice (merged by sample_finalize).
    float local_ent = 0.0f;
#endif
    for (uint i = b.x + tid; i < b.y; i += SAMPLING_BLOCK_SIZE) {
        float val = rowp[i] * inv_temp;
        float prob = exp(val - max_logit) * inv_sum_exp;
        bits[i] = as_type<uint>(prob);
        atomic_fetch_add_explicit(local_hist + (as_type<uint>(prob) >> DIGIT_SHIFT[0]), 1u,
                                  memory_order_relaxed);
#if SAMPLER_TELEMETRY
        if (prob > 0.0f) {
            local_ent -= prob * log(prob);
        }
#endif
    }
    descent_flush(local_hist, descent_hist(hist, row, 0), tid);
#if SAMPLER_TELEMETRY
    float ent = block_reduce_sum(local_ent, s_warp_buf, tid);
    if (tid == 0) {
        partials[((size_t)row * nslices + slice) * 3 + 2] = as_type<uint>(ent);
    }
#endif
}
#endif

// ---------------------------------------------------------------------------
// sample_descent_round_{1,2} — per (row, slice): first the pick of the
// previous round (`descent_pick`, every threadgroup for itself; slice 0
// records it for the next pass), then the histogram of round ROUND's digit
// over its survivors. Round 0's histogram rides `sample_softmax_materialize`,
// the last round's pick `sample_count_compact`.
// ---------------------------------------------------------------------------

template <uint ROUND>
[[kernel]] void sample_descent_round(
    device const uint*  prob_bits [[buffer(0)]],
    device       uint*  hist      [[buffer(1)]],
    device       uint*  row_state [[buffer(2)]],
    constant     uint*  consts    [[buffer(3)]],
    uint tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint vocab = SAMPLER_VOCAB;
    uint nslices = consts[0];
    uint row = tgp / nslices;
    uint slice = tgp % nslices;
    device uint* state = row_state + (size_t)row * ROW_STATE_LEN;
    device const uint* rowp = prob_bits + (size_t)row * vocab;
    uint2 b = slice_bounds(vocab, nslices, slice);

    threadgroup uint s_warp_buf[NUM_WARPS];
    threadgroup uint2 s_pick;
    uint2 picked = descent_pick(descent_hist(hist, row, ROUND - 1), ROUND - 1, state[8],
                                descent_state(state, ROUND - 1), s_warp_buf, &s_pick, tid);
    if (slice == 0 && tid == 0) {
        state[DESCENT_STATE + 2 * ROUND - 2] = picked.x;
        state[DESCENT_STATE + 2 * ROUND - 1] = picked.y;
    }

    uint mask = prefix_mask(ROUND);
    uint prefix = picked.x & mask;
    threadgroup atomic_uint local_hist[DESCENT_BUCKETS];
    descent_clear(local_hist, tid);
    for (uint i = b.x + tid; i < b.y; i += SAMPLING_BLOCK_SIZE) {
        uint x = rowp[i];
        if ((x & mask) == prefix) {
            uint digit = (x >> DIGIT_SHIFT[ROUND]) & digit_mask(ROUND);
            atomic_fetch_add_explicit(local_hist + digit, 1u, memory_order_relaxed);
        }
    }
    descent_flush(local_hist, descent_hist(hist, row, ROUND), tid);
}

SCRATCHY_KERNEL(sample_descent_round_1, sample_descent_round<1>)
SCRATCHY_KERNEL(sample_descent_round_2, sample_descent_round<2>)

// ---------------------------------------------------------------------------
// sample_count_compact — per (row, slice): pick the last round (completing
// the threshold), count this slice's strict (bits > threshold) and tied
// (bits == threshold) elements exactly, and append the strict and the tied
// candidates to the slice's two staging runs. The staging cap matches the
// cuda kernel's `pos < cap` drop.
// ---------------------------------------------------------------------------

#if SCRATCHY_COMPILES(sample_count_compact)
kernel void sample_count_compact(
    device const uint*  prob_bits [[buffer(0)]],
    device       uint*  hist      [[buffer(1)]],
    device       uint*  counts    [[buffer(2)]],
    device       uint*  staging   [[buffer(3)]],
    device       uint*  row_state [[buffer(4)]],
    constant     uint*  consts    [[buffer(5)]],
    uint tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint vocab = SAMPLER_VOCAB;
    uint nslices = consts[0];
    uint row = tgp / nslices;
    uint slice = tgp % nslices;
    device uint* state = row_state + (size_t)row * ROW_STATE_LEN;
    device const uint* rowp = prob_bits + (size_t)row * vocab;
    uint2 b = slice_bounds(vocab, nslices, slice);

    // The threshold is complete after the last round's pick; apply the min-p
    // floor (cuda phase 3b).
    constexpr uint last = DESCENT_ROUNDS - 1;
    threadgroup uint s_warp_buf[NUM_WARPS];
    threadgroup uint2 s_pick;
    uint2 picked = descent_pick(descent_hist(hist, row, last), last, state[8],
                                descent_state(state, last), s_warp_buf, &s_pick, tid);
    float threshold_prob = as_type<float>(picked.x);
    float min_p = as_type<float>(state[5]);
    if (min_p > 0.0f) {
        float inv_sum_exp = 1.0f / as_type<float>(state[2]);
        float max_prob = inv_sum_exp;  // exp(0) * inv_sum_exp
        threshold_prob = max(threshold_prob, min_p * max_prob);
    }
    uint threshold = as_type<uint>(threshold_prob);
    if (slice == 0 && tid == 0) state[7] = threshold;

    device uint* c = counts + ((size_t)row * nslices + slice) * 4;
    device uint* stage = staging
        + ((size_t)row * nslices + slice) * 4 * MAX_CANDIDATES;
    device uint* stage_tied = stage + 2 * MAX_CANDIDATES;

    threadgroup atomic_uint s_next[2];
    if (tid < 2) atomic_store_explicit(s_next + tid, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint my_strict = 0;
    uint my_tied = 0;
    for (uint i = b.x + tid; i < b.y; i += SAMPLING_BLOCK_SIZE) {
        uint x = rowp[i];
        if (x > threshold) {
            my_strict++;
            uint pos = atomic_fetch_add_explicit(s_next, 1u, memory_order_relaxed);
            if (pos < MAX_CANDIDATES) {
                stage[2 * pos + 0] = x;
                stage[2 * pos + 1] = i;
            }
        } else if (x == threshold) {
            my_tied++;
            uint pos = atomic_fetch_add_explicit(s_next + 1, 1u, memory_order_relaxed);
            if (pos < MAX_CANDIDATES) {
                stage_tied[2 * pos + 0] = x;
                stage_tied[2 * pos + 1] = i;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint strict_total = block_reduce_sum_u32(my_strict, s_warp_buf, tid);
    uint tied_total = block_reduce_sum_u32(my_tied, s_warp_buf, tid);
    if (tid == 0) {
        c[0] = strict_total;
        c[1] = tied_total;
        // This slice's staged strict and tied runs.
        c[2] = min(atomic_load_explicit(s_next, memory_order_relaxed), (uint)MAX_CANDIDATES);
        c[3] = min(atomic_load_explicit(s_next + 1, memory_order_relaxed), (uint)MAX_CANDIDATES);
    }
}
#endif

// ---------------------------------------------------------------------------
// sample_finalize — ONE threadgroup per row: each slice's share of the tie
// slots, then gather the slices' staging runs (strict pairs then the
// quota's tied pairs, slices in order — the cuda kernel's two-phase fill),
// then the cuda tail: bitonic sort descending by prob, top-p cutoff,
// renormalize, categorical draw. The token lands over the step's argmax
// for the row (`tokens[row_indices[row]]`), where the next step reads it.
//
// num_candidates >= 1 always: the threshold is at most the largest prob bits
// value in the row (the min-p floor is min_p * max_prob <= max_prob), so at
// least the argmax element is strict or tied. The fallback keeps the cuda
// kernel's shape anyway.
// ---------------------------------------------------------------------------

#if SCRATCHY_COMPILES(sample_finalize)
kernel void sample_finalize(
    device const uint*  staging     [[buffer(0)]],
    device const uint*  counts      [[buffer(1)]],
    device const uint*  row_state   [[buffer(2)]],
    device const uint*  prob_bits   [[buffer(3)]],
    device const uint*  row_indices [[buffer(4)]],
    device       uint*  tokens      [[buffer(5)]],
    constant     uint*  consts      [[buffer(6)]],
#if SAMPLER_TELEMETRY
    device const uint*  partials     [[buffer(7)]],
    device       float* topk_probs   [[buffer(8)]],
    device       uint*  topk_indices [[buffer(9)]],
    device       float* stats_out    [[buffer(10)]],
    constant     uint*  telem_consts [[buffer(11)]],
#endif
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

    // Tie quotas, slices in index order: the greedy fill
    //   take_s = min(tied_s, remaining); remaining -= take_s
    // in closed form, take_s = min(tied_s, avail - min(tied before s, avail)).
    // One lane a slice (the host cuts a row into at most WARP_SIZE slices).
    threadgroup uint s_quota[WARP_SIZE];
    if (tid < WARP_SIZE) {
        uint strict = tid < nslices ? c[tid * 4 + 0] : 0u;
        uint tied = tid < nslices ? c[tid * 4 + 1] : 0u;
        uint cap = state[8];
        uint strict_kept = min(simd_sum(strict), cap);
        uint avail = min(simd_sum(tied), cap - strict_kept);
        uint before = simd_prefix_exclusive_sum(tied);
        s_quota[tid] = min(tied, avail - min(before, avail));
    }

    // Gather the slices' runs in order.
    threadgroup atomic_uint s_cursor;
    if (tid == 0) atomic_store_explicit(&s_cursor, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0; s < nslices; s++) {
        uint strict = c[s * 4 + 2];
        uint len = strict + min(c[s * 4 + 3], s_quota[s]);
        device const uint* stage =
            staging + ((size_t)row * nslices + s) * 4 * MAX_CANDIDATES;
        for (uint j = tid; j < len; j += SAMPLING_BLOCK_SIZE) {
            uint at = j < strict ? j : MAX_CANDIDATES + j - strict;
            uint pos = atomic_fetch_add_explicit(&s_cursor, 1u, memory_order_relaxed);
            if (pos < MAX_CANDIDATES) {
                s_probs[pos] = as_type<float>(stage[2 * at + 0]);
                s_indices[pos] = stage[2 * at + 1];
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
            tokens[row_indices[row]] = best_idx;
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
        tokens[row_indices[row]] = sampled;

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
