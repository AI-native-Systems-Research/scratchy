// SPDX-License-Identifier: Apache-2.0
//
// MoE grouped-GEMM prefill support: sort the (token,expert) rows by
// expert and lay them out PADDED (each expert's run rounded up to a
// multiple of BM=32) so the grouped expert GEMM
// (`affine_gather_qmm_t_kernel`) maps each 32-row output tile to exactly
// one expert. This is the host half of mlx's grouped MoE path
// (mlx-lm SwitchGLU `_gather_sort`; mlx `gather_qmm` with sorted indices)
// — implemented as a counting sort (experts are 0..num_experts-1, so a
// histogram + prefix-sum + scatter is the natural sort), since the
// expert ids form a tiny key space.
//
// Pipeline (per MoE layer, prefill only):
//   1. moe_group_offsets : histogram counts + padded exclusive-scan
//                          offsets (1 threadgroup; tiny key space).
//   2. moe_group_init    : sentinel-fill indices_pad + zero the fill
//                          counters (so trailing unused tiles skip).
//   3. moe_group_scatter : place each real (token,expert) row at
//                          offset[e]+running, recording pos[i],
//                          indices_pad[pos], and the gathered x_pad row.
//   ... grouped GEMM (gate/up/down) over the padded layout ...
//   4. take_along_axis(pos) unsorts the down output back to token order.
//
// All index buffers are u32. `MG_M` = number of (token,expert) pairs
// (= bucket_m * top_k). `MG_NUM_EXPERTS` ≤ 128 (gemma4 = 128).

#include <metal_stdlib>

using namespace metal;

// ── moe_group_offsets ──────────────────────────────────────────────
// Single threadgroup. Histograms `topk_inds[MG_M]` into threadgroup
// memory (no cross-tg device atomics → device `count` needs no prior
// zeroing), then thread 0 runs the padded exclusive scan:
//   offset[e] = Σ_{e'<e} ceil(count[e']/BM)*BM ,  total = Σ ceil(..)*BM
// `count`/`offset` are [MG_NUM_EXPERTS] u32; `total` is [1] u32 (the
// padded row count Mpad, ≤ MG_M + (BM-1)*MG_NUM_EXPERTS).
constant int MG_M           [[function_constant(0)]];
constant int MG_NUM_EXPERTS [[function_constant(1)]];

// Pad each expert's run to a multiple of 64 = the NAX grouped GEMM's
// m-tile (BM=64). 64 is also a multiple of the steel grouped GEMM's
// BM=32, so the same padded layout drives either kernel.
constant int MG_BM = 64;
constant int MG_MAX_EXPERTS = 128;

kernel void moe_group_offsets(
    const device uint* topk_inds [[buffer(0)]],
    device uint*       count     [[buffer(1)]],
    device uint*       offset    [[buffer(2)]],
    device uint*       total     [[buffer(3)]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tgsize  [[threads_per_threadgroup]]) {
  threadgroup atomic_uint local[MG_MAX_EXPERTS];
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

// ── moe_group_init ─────────────────────────────────────────────────
// Sentinel-fills `indices_pad[Mpad_max]` with MG_NUM_EXPERTS (an
// invalid expert → the GEMM skips that tile) and zeroes `fill[E]`.
// `MG_MPAD_MAX` = MG_M + (BM-1)*MG_NUM_EXPERTS (worst-case padded rows;
// the static dispatch upper bound). One thread per padded row.
constant int MG_MPAD_MAX [[function_constant(2)]];

kernel void moe_group_init(
    device uint* indices_pad [[buffer(0)]],
    device uint* fill        [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < uint(MG_MPAD_MAX)) {
    indices_pad[gid] = uint(MG_NUM_EXPERTS);  // sentinel
  }
  if (gid < uint(MG_NUM_EXPERTS)) {
    fill[gid] = 0u;
  }
}

// ── moe_group_scatter ──────────────────────────────────────────────
// For each real (token,expert) pair i in [0, MG_M): compute its padded
// row p = offset[e] + atomic_inc(fill[e]); record pos[i]=p,
// indices_pad[p]=e, and gather the token's x row into x_pad[p].
// x is [bucket_m, K]; the token of pair i is i / MG_TOP_K. One
// threadgroup row-block per pair (grid.y = MG_M), threads cover K.
constant int MG_TOP_K [[function_constant(3)]];
constant int MG_K     [[function_constant(4)]];

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

#define INST_MG_SCATTER(tag, type)                                       \
  template [[host_name("moe_group_scatter_" #tag)]]                      \
  [[kernel]] void moe_group_scatter<type>(                               \
      const device uint* topk_inds [[buffer(0)]],                        \
      const device uint* offset    [[buffer(1)]],                        \
      const device type* x         [[buffer(2)]],                        \
      device atomic_uint* fill     [[buffer(3)]],                        \
      device uint*       pos       [[buffer(4)]],                        \
      device uint*       indices_pad [[buffer(5)]],                      \
      device type*       x_pad     [[buffer(6)]],                        \
      uint2 tid  [[thread_position_in_threadgroup]],                     \
      uint2 tgid [[threadgroup_position_in_grid]],                       \
      uint2 tgsz [[threads_per_threadgroup]]);

INST_MG_SCATTER(float16, half)
INST_MG_SCATTER(bfloat16, bfloat)
INST_MG_SCATTER(float32, float)

// ── moe_group_gather (un-scatter) ──────────────────────────────────
// Restores token order after the grouped GEMM:
//   out[i, :] = src[pos[i], :]   for i in [0, MG_M)
// (src = the padded down-projection [Mpad, MG_W]; out = down_out
// [bucket_m*top_k, MG_W] = the matvec path's layout, so the existing
// moe_weighted_sum reduces it unchanged). One threadgroup row-block per
// pair (grid.y = MG_M, m-scaled to actual pairs); threads cover MG_W.
constant int MG_W [[function_constant(5)]];

template <typename T>
kernel void moe_group_gather(
    const device T*    src [[buffer(0)]],
    const device uint* pos [[buffer(1)]],
    device T*          out [[buffer(2)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tgsz [[threads_per_threadgroup]]) {
  const uint i = tgid.y;
  const uint p = pos[i];
  const device T* src_row = src + size_t(p) * uint(MG_W);
  device T* out_row = out + size_t(i) * uint(MG_W);
  for (uint d = tid.x; d < uint(MG_W); d += tgsz.x) {
    out_row[d] = src_row[d];
  }
}

#define INST_MG_GATHER(tag, type)                                        \
  template [[host_name("moe_group_gather_" #tag)]]                       \
  [[kernel]] void moe_group_gather<type>(                                \
      const device type* src [[buffer(0)]],                              \
      const device uint* pos [[buffer(1)]],                              \
      device type*       out [[buffer(2)]],                              \
      uint2 tid  [[thread_position_in_threadgroup]],                     \
      uint2 tgid [[threadgroup_position_in_grid]],                       \
      uint2 tgsz [[threads_per_threadgroup]]);

INST_MG_GATHER(float16, half)
INST_MG_GATHER(bfloat16, bfloat)
INST_MG_GATHER(float32, float)
