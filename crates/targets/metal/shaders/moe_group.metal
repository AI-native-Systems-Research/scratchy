// SPDX-License-Identifier: Apache-2.0
//
// MoE grouped-GEMM support: sort the (token,expert) rows by expert and
// lay them out PADDED (each expert's run rounded up to a multiple of
// MG_BM — 64 for the grouped GEMMs, 1 for the sorted gathered matvec
// path) so the grouped expert GEMM maps each 64-row output tile to
// exactly one expert. This is the host half of mlx's grouped MoE path
// (mlx-lm SwitchGLU `_gather_sort`; mlx `gather_qmm` with sorted indices)
// — implemented as a counting sort (experts are 0..num_experts-1, so a
// histogram + prefix-sum + scatter is the natural sort), since the
// expert ids form a tiny key space.
//
// Pipeline (per MoE layer — grouped prefill and sorted decode both):
//   1. moe_group_offsets : histogram counts + padded exclusive-scan
//                          offsets (1 threadgroup; tiny key space).
//   2. moe_group_init    : sentinel-fill indices_pad + zero the fill
//                          counters (so trailing unused tiles skip).
//   3. moe_group_scatter : place each real (token,expert) row at
//                          offset[e]+running, recording pos[i],
//                          indices_pad[pos], and the gathered x_pad row.
//   ... the grouped GEMMs (gate/up/down) over the padded layout, or the
//       gathered matvecs over the sorted rows ...
//   4. moe_weighted_sum reads each pair's down row through pos.
//
// All index buffers are u32. `MG_M` = number of (token,expert) pairs
// (= bucket_m * top_k). `MG_NUM_EXPERTS`: the model's experts (gemma4 128,
// Qwen3.5/3.6 256), compiled in — it sizes the histogram.

#include <metal_stdlib>
#include "baked.h"

using namespace metal;

// ── moe_group_offsets ──────────────────────────────────────────────
// Single threadgroup. Histograms `topk_inds[MG_M]` into threadgroup
// memory (no cross-tg device atomics → device `count` needs no prior
// zeroing), then thread 0 runs the padded exclusive scan:
//   offset[e] = Σ_{e'<e} ceil(count[e']/BM)*BM ,  total = Σ ceil(..)*BM
// `count`/`offset` are [MG_NUM_EXPERTS] u32; `total` is [1] u32 (the
// padded row count Mpad, ≤ MG_M + (BM-1)*MG_NUM_EXPERTS).
SCRATCHY_CONSTANT_OPTIONAL(int, MG_M, 0);
SCRATCHY_CONSTANT_OPTIONAL(int, MG_NUM_EXPERTS, 1);

// Pad each expert's run to a multiple of MG_BM: 64 = the NAX grouped GEMM's
// m-tile (also a multiple of the steel grouped GEMM's BM=32, so the same
// padded layout drives either kernel); 1 = no padding, the layout the
// gathered matvec path reads sorted (same-expert pairs adjacent, so their
// repeat slab reads hit cache).
SCRATCHY_CONSTANT_OPTIONAL(int, MG_BM, 6);
// What `moe_group_init` sentinel-fills the dead rows of `indices_pad` with:
// the grouped GEMMs skip a tile whose expert is `MG_NUM_EXPERTS`; the
// gathered matvecs have no such guard, so their bake fills 0 (a real
// expert's slab — garbage compute on rows nothing reads).
SCRATCHY_CONSTANT_OPTIONAL(int, MG_SENTINEL, 7);

#if SCRATCHY_COMPILES(moe_group_offsets)
kernel void moe_group_offsets(
    const device uint* topk_inds [[buffer(0)]],
    device uint*       count     [[buffer(1)]],
    device uint*       offset    [[buffer(2)]],
    device uint*       total     [[buffer(3)]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tgsize  [[threads_per_threadgroup]]) {
  // One bin per expert: the bake compiles this kernel at its block's expert count.
  threadgroup atomic_uint local[MG_NUM_EXPERTS];
  const uint E = uint(MG_NUM_EXPERTS);
  for (uint i = tid; i < E; i += tgsize) {
    atomic_store_explicit(&local[i], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < uint(MG_M); i += tgsize) {
    uint e = topk_inds[i];
    if (e < E) {
      atomic_fetch_add_explicit(&local[e], 1u, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < E; i += tgsize) {
    count[i] = atomic_load_explicit(&local[i], memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint acc = 0;
    for (uint e = 0; e < E; ++e) {
      offset[e] = acc;
      uint c = count[e];
      acc += ((c + uint(MG_BM) - 1u) / uint(MG_BM)) * uint(MG_BM);
    }
    total[0] = acc;
  }
}
#endif

// ── moe_group_init ─────────────────────────────────────────────────
// Sentinel-fills `indices_pad[Mpad_max]` with MG_NUM_EXPERTS (an
// invalid expert → the GEMM skips that tile) and zeroes `fill[E]`.
// `MG_MPAD_MAX` = MG_M + (BM-1)*MG_NUM_EXPERTS (worst-case padded rows;
// the static dispatch upper bound). One thread per padded row.
SCRATCHY_CONSTANT_OPTIONAL(int, MG_MPAD_MAX, 2);

#if SCRATCHY_COMPILES(moe_group_init)
kernel void moe_group_init(
    device uint* indices_pad [[buffer(0)]],
    device uint* fill        [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < uint(MG_MPAD_MAX)) {
    indices_pad[gid] = uint(MG_SENTINEL);
  }
  if (gid < uint(MG_NUM_EXPERTS)) {
    fill[gid] = 0u;
  }
}
#endif

// ── moe_group_scatter ──────────────────────────────────────────────
// For each real (token,expert) pair i in [0, MG_M): compute its padded
// row p = offset[e] + atomic_inc(fill[e]); record pos[i]=p,
// indices_pad[p]=e, and gather the token's x row into x_pad[p].
// x is [bucket_m, K]; the token of pair i is i / MG_TOP_K. One
// threadgroup row-block per pair (grid.y = MG_M), threads cover K.
SCRATCHY_CONSTANT_OPTIONAL(int, MG_TOP_K, 3);
SCRATCHY_CONSTANT_OPTIONAL(int, MG_K, 4);

template <typename T>
kernel void moe_group_scatter(
    const device uint* topk_inds [[buffer(0)]],
    const device uint* offset    [[buffer(1)]],
    const device T*    x         [[buffer(2)]],
    device atomic_uint* fill     [[buffer(3)]],
    device uint*       pos       [[buffer(4)]],
    device uint*       indices_pad [[buffer(5)]],
    device T*          x_pad     [[buffer(6)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tgsz [[threads_per_threadgroup]]) {
  const uint i = tgid.y;
  if (i >= uint(MG_M)) return;
  threadgroup uint p_shared;
  // Thread 0 claims the padded slot for this pair (one atomic per pair).
  if (tid.x == 0) {
    uint e = topk_inds[i];
    uint p = (e < uint(MG_NUM_EXPERTS))
        ? offset[e] + atomic_fetch_add_explicit(&fill[e], 1u, memory_order_relaxed)
        : 0u;
    pos[i] = p;
    indices_pad[p] = e;
    p_shared = p;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint p = p_shared;
  const uint token = i / uint(MG_TOP_K);
  const device T* x_row = x + size_t(token) * uint(MG_K);
  device T* xp_row = x_pad + size_t(p) * uint(MG_K);
  for (uint d = tid.x; d < uint(MG_K); d += tgsz.x) {
    xp_row[d] = x_row[d];
  }
}

#define INST_MG_SCATTER(tag, type) \
  SCRATCHY_KERNEL(moe_group_scatter_##tag, moe_group_scatter<type>)

INST_MG_SCATTER(float16, half)
INST_MG_SCATTER(bfloat16, bfloat)
INST_MG_SCATTER(float32, float)

// ── moe_group_scatter_q8 ───────────────────────────────────────────
// `moe_group_scatter` for the W4A8 grouped GEMM: places each pair's
// token row already quantized (`affine_w4a8_quant` layout: int8 xq[rows][K]
// then float2 qa[rows][K/64]) at its padded slot, so the tokens are
// quantized once, not once per expert copy. x holds the bucket's
// MG_M / MG_TOP_K token rows; x_pad holds MG_MPAD_MAX padded rows.
#if SCRATCHY_COMPILES(moe_group_scatter_q8)
kernel void moe_group_scatter_q8(
    const device uint* topk_inds [[buffer(0)]],
    const device uint* offset    [[buffer(1)]],
    const device uchar* x        [[buffer(2)]],
    device atomic_uint* fill     [[buffer(3)]],
    device uint*       pos       [[buffer(4)]],
    device uint*       indices_pad [[buffer(5)]],
    device uchar*      x_pad     [[buffer(6)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tgsz [[threads_per_threadgroup]]) {
  const uint i = tgid.y;
  if (i >= uint(MG_M)) return;
  threadgroup uint p_shared;
  if (tid.x == 0) {
    uint e = topk_inds[i];
    uint p = (e < uint(MG_NUM_EXPERTS))
        ? offset[e] + atomic_fetch_add_explicit(&fill[e], 1u, memory_order_relaxed)
        : 0u;
    pos[i] = p;
    indices_pad[p] = e;
    p_shared = p;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint p = p_shared;
  const uint token = i / uint(MG_TOP_K);
  const uint K = uint(MG_K), KC = K / 64, rows = uint(MG_M) / uint(MG_TOP_K);
  const device uint4* src = (const device uint4*)(x + size_t(token) * K);
  device uint4* dst = (device uint4*)(x_pad + size_t(p) * K);
  for (uint d = tid.x; d < K / 16; d += tgsz.x) {
    dst[d] = src[d];
  }
  const device float2* qa = (const device float2*)(x + size_t(rows) * K) + size_t(token) * KC;
  device float2* qa_pad = (device float2*)(x_pad + size_t(uint(MG_MPAD_MAX)) * K) + size_t(p) * KC;
  for (uint c = tid.x; c < KC; c += tgsz.x) {
    qa_pad[c] = qa[c];
  }
}
#endif
