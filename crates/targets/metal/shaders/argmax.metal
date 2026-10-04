// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

#include <metal_stdlib>
#include "baked.h"
using namespace metal;

// ---------------------------------------------------------------------------
// argmax — greedy sampler.
//
// Input:  logits[batch, ARGMAX_VOCAB]   (half / bfloat)
// Output: token  [batch]                (uint)
//
// Dispatch: threadgroups (batch, 1, 1), threads_per_threadgroup
// (TG_SIZE, 1, 1) where TG_SIZE is a power of 2 ≤ 1024. One
// threadgroup per batch row; threads in a group cooperate over the
// vocab axis. Each thread keeps a (max_val, max_idx) running pair
// over its strided slice; a power-of-two reduction in threadgroup
// memory yields the per-row argmax.
//
// Tie-break: on equal-max values the smaller index wins, matching
// the numpy / torch `argmax` convention. Keeps results
// deterministic across thread schedule changes (e.g. different
// threadgroup sizes for the same input).
//
// Vocab can exceed `TG_SIZE` (50K-150K is typical for LLMs); each
// thread may walk many elements. Reduction overhead is bounded by
// `log2(TG_SIZE)` barriers.
//
// Bindings:
//   buffer(0) = logits  [batch, ARGMAX_VOCAB]
//   buffer(1) = output  [batch]   uint
//   buffer(2) = batch   constant uint
//   buffer(3) = next_in [batch]   uint  (dual write only)
// ---------------------------------------------------------------------------

SCRATCHY_CONSTANT(uint, ARGMAX_VOCAB, 0);

// The argmax of row `gid`, in thread 0.
template <typename T>
inline uint argmax_row(
    device const T* logits,
    uint gid,
    uint tid,
    uint tg,
    threadgroup float* shared_max,
    threadgroup uint* shared_idx)
{
    // Per-thread reduction over a strided slice of the vocab axis.
    float local_max = -INFINITY;
    uint  local_idx = 0;

    device const T* row = logits + uint(gid) * ARGMAX_VOCAB;
    for (uint i = tid; i < ARGMAX_VOCAB; i += tg) {
        float v = float(row[i]);
        // Tie-break on smaller index — needed to make per-thread
        // intermediate state agree with the final reduction's
        // tie-break before any cross-thread compare happens.
        if (v > local_max || (v == local_max && i < local_idx)) {
            local_max = v;
            local_idx = i;
        }
    }

    shared_max[tid] = local_max;
    shared_idx[tid] = local_idx;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction. Pairwise compare; on tie, keep the smaller
    // index. `tg` is power-of-two by contract (caller picks 256 /
    // 512 / 1024 as appropriate).
    for (uint stride = tg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a  = shared_max[tid];
            float b  = shared_max[tid + stride];
            uint  ai = shared_idx[tid];
            uint  bi = shared_idx[tid + stride];
            if (b > a || (b == a && bi < ai)) {
                shared_max[tid] = b;
                shared_idx[tid] = bi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return shared_idx[0];
}

template <typename T>
[[kernel]] void argmax(
    device const T*  logits [[buffer(0)]],
    device       uint* output [[buffer(1)]],
    constant     uint& batch  [[buffer(2)]],
    uint  gid [[threadgroup_position_in_grid]],
    uint  tid [[thread_position_in_threadgroup]],
    uint  tg  [[threads_per_threadgroup]])
{
    if (gid >= batch) return;
    threadgroup float shared_max[1024];
    threadgroup uint  shared_idx[1024];
    uint v = argmax_row(logits, gid, tid, tg, shared_max, shared_idx);
    if (tid == 0) {
        output[gid] = v;
    }
}

// ---------------------------------------------------------------------------
// argmax dual write — Phase 6 primitive
//
// Same reduction as `argmax`, but writes the per-row argmax to TWO
// destination buffers. Phase 6 uses this to feed the next K-step
// iter's `input_ids` on the GPU side without a host roundtrip:
//   * `output`    — the host-visible per-iter draft buffer
//                    (read by the caller after the CB finishes)
//   * `next_in`   — the worker's `runtime.input_ids` (read by the
//                    NEXT forward dispatch in the same CB)
//
// The next forward's embed/gather kernel reads input_ids[0..num_reqs]
// from `runtime.input_ids`; since we encode argmax → next forward in
// the same compute encoder, Metal's intra-encoder data dependency on
// `next_in` serializes them correctly. No barrier needed when the
// next forward's first dispatch declares `next_in` as a read; the
// driver hazards-tracks the write→read on the same buffer.
// ---------------------------------------------------------------------------
template <typename T>
[[kernel]] void argmax_dual_write(
    device const T*  logits  [[buffer(0)]],
    device       uint* output  [[buffer(1)]],
    constant     uint& batch   [[buffer(2)]],
    device       uint* next_in [[buffer(3)]],
    uint  gid [[threadgroup_position_in_grid]],
    uint  tid [[thread_position_in_threadgroup]],
    uint  tg  [[threads_per_threadgroup]])
{
    if (gid >= batch) return;
    threadgroup float shared_max[1024];
    threadgroup uint  shared_idx[1024];
    uint v = argmax_row(logits, gid, tid, tg, shared_max, shared_idx);
    if (tid == 0) {
        output[gid]  = v;
        next_in[gid] = v;
    }
}

SCRATCHY_KERNEL(argmax_f16, argmax<half>)
SCRATCHY_KERNEL(argmax_bf16, argmax<bfloat>)
SCRATCHY_KERNEL(argmax_f16_dual_write, argmax_dual_write<half>)
SCRATCHY_KERNEL(argmax_bf16_dual_write, argmax_dual_write<bfloat>)
