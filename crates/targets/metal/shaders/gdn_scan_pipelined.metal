// SPDX-License-Identifier: Apache-2.0
//
// Gated-DeltaNet prefill scan: the software-pipelined block-staged kernel.
//
// The design is omlx Kernel P's (omlx/custom_kernels/qwen35_prefill/gdn.py,
// `gated_delta_pipelined`, Apache-2.0) — the proven prefill mapping for this
// recurrence — under this crate's kernel contracts. Where Kernel P receives
// q/k/v pre-split, pre-normalized and with g/beta precomputed on the host,
// this kernel reads the SAME buffers `gdn_scan_simd` does (f32 `conv_out` in
// the [q|k|v] interleaved layout, raw a/b, A_log/dt_bias weights) and
// computes the L2 norms and the gating inline. Per token the math is exactly
// `gdn_scan_simd`'s — the SAME eps-inside-sqrt norms, fused-stable softplus
// and fma/recurrence order — with the two head_k dots reduced by an xor
// butterfly over the row's 8 lanes, the summation order simd_sum builds, so
// the two kernels agree to f32 rounding (the parity test pins this).
//
// Mapping (Kernel P): 8 lanes own a value row, each holding head_k/8 state
// channels in four float4 granules at columns seg, seg+8, seg+16, seg+24; a
// thread owns two rows, rg and rg+16 of its threadgroup's 32, so a 128-thread
// threadgroup covers 32 value rows and every k/q fragment a lane reads serves
// both (value_dim/32 threadgroups). The block's k/q rows (the key head's
// head_k channels), the 32-row v slice and the raw a/b scalars are staged
// cooperatively into threadgroup memory and prefetched into registers one
// block ahead, so the recurrence's per-token reads hit registers or
// threadgroup memory, never device memory. Once per block, a token's norms
// (each lane's k_inv, and q scaled by its q_inv in place) and its gating
// (decay, 1 + exp(-b)) are formed once for the threadgroup rather than by
// every lane of every row.
//
// Every value is formed by the expression, and in the statement order, a
// thread owning one row and forming its own norms and gating uses, so the
// outputs do not depend on the mapping, bit for bit. The order is load-bearing
// — under fast math the shader compiler contracts by it (k_inv read after the
// k/q fragments; delta·k_inv as ((v - p)·k_inv)/(1 + exp(-b))) — and a row's 8
// lanes need not agree on a butterfly total to the last bit, so each lane keeps
// the norms its own butterfly formed.
//
// `conv_out` and `o` are f32; `a`/`b`/`dt_bias` are `T`; `ssm_state` is f32,
// laid out as `gdn_scan_varlen` lays a slot out: its state entry
// [HV, head_v, head_k], then — when the model drafts — two record areas.
// This kernel runs only a plain step (`scratchy_layers::gdn_state::GdnStep::
// is_plain`: `gdn_step[seq]` starts from the slot or from zero and carries no
// drafts), so it reads and writes the entry alone; a step that replays or
// records runs `gdn_scan_simd`.
//
// Baked constants (every compile sets every one; slots 0–5 are the scan's):
//   GDN_PIPE_NUM_K_HEADS (H), GDN_PIPE_NUM_V_HEADS (HV),
//   GDN_PIPE_HEAD_K (K), GDN_PIPE_HEAD_V (head_v), GDN_PIPE_SCALE,
//   GDN_PIPE_DRAFTS (the model's drafts: the slot stride),
//   GDN_PIPE_TB (tokens per block; multiple of 4).
// Requires K == 128 (8 lanes × four float4s) and head_v % 32 == 0.
//
// Dispatch: grid (head_v*HV/32, 1, num_seqs); threads (128, 1, 1). The
// lowering routes a plain PREFILL here and keeps decode on `gdn_scan_simd`
// (whose 4-simdgroup groups are the right shape for one token).

#include <metal_stdlib>
#include "baked.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, GDN_PIPE_NUM_K_HEADS, 0);
SCRATCHY_CONSTANT(uint, GDN_PIPE_NUM_V_HEADS, 1);
SCRATCHY_CONSTANT(uint, GDN_PIPE_HEAD_K, 2);
SCRATCHY_CONSTANT(uint, GDN_PIPE_HEAD_V, 3);
SCRATCHY_CONSTANT(float, GDN_PIPE_SCALE, 4);
SCRATCHY_CONSTANT(uint, GDN_PIPE_DRAFTS, 5);
SCRATCHY_CONSTANT(uint, GDN_PIPE_TB, 6);

template <typename T>
[[kernel]] void gdn_scan_pipelined(
    device       float* o             [[buffer(0)]],
    const device float* conv_out      [[buffer(1)]],
    const device T*     a             [[buffer(2)]],
    const device T*     b             [[buffer(3)]],
    device       float* ssm_state     [[buffer(4)]],
    const device int*   cu_seqlens    [[buffer(5)]],
    const device int*   state_indices [[buffer(6)]],
    const device uint*  gdn_step      [[buffer(7)]],
    const device float* a_log         [[buffer(8)]],
    const device T*     dt_bias       [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpig [[thread_position_in_threadgroup]])
{
  constexpr uint TB = GDN_PIPE_TB;
  constexpr uint NT = 128;
  constexpr uint KR = 8;   // lanes per value row (head_k == 8*KR channels)
  constexpr uint RS = NT / KR;  // a thread's second row sits RS rows below its first
  constexpr uint DB = 2 * RS;   // value rows per threadgroup
  const uint H = GDN_PIPE_NUM_K_HEADS;
  const uint HV = GDN_PIPE_NUM_V_HEADS;
  const uint K = GDN_PIPE_HEAD_K;
  const uint Vd = GDN_PIPE_HEAD_V;
  const uint key_dim = H * K;
  const uint value_dim = HV * Vd;
  const uint conv_dim = 2u * key_dim + value_dim;

  const uint row0 = tgid.x * DB;
  const uint tid = tpig.x;
  const uint rg = tid / KR;   // first value row within the threadgroup
  const uint seg = tid % KR;  // lane within the row
  const uint i_n = tgid.z;    // sequence
  const uint vr = row0 + rg;  // global value row = i_hv*Vd + i_v (the second: vr + RS)
  const uint i_hv = vr / Vd;
  const uint i_v = vr % Vd;
  const uint i_h = i_hv / (HV / H);
  if (vr >= value_dim) {
    return;
  }

  const int bos = cu_seqlens[i_n];
  const int eos = cu_seqlens[i_n + 1];
  const int seq_len = eos - bos;
  const int slot = state_indices[i_n];
  if (seq_len <= 0 || slot < 0) {
    return;
  }
  const bool fresh = (gdn_step[i_n] & 0xffu) == 1u;

  // Per-head scalars, off the recurrence path.
  const float neg_a = -exp(a_log[i_hv]);
  const float dtb = float(dt_bias[i_hv]);
  const float scale = GDN_PIPE_SCALE;

  // The slot's stride, as `gdn_scan_varlen` lays it out: its entry, then
  // (with drafts) two record areas of `GDN_PIPE_DRAFTS + 1` rows.
  const uint entry_len = HV * Vd * K;
  const uint area_len = (GDN_PIPE_DRAFTS + 1u) * (conv_dim + 2u * HV);
  const uint slot_len = entry_len + (GDN_PIPE_DRAFTS == 0u ? 0u : 2u * area_len);
  device float* state_row0 = ssm_state + uint(slot) * slot_len + (i_hv * Vd + i_v) * K;
  device float* state_row1 = state_row0 + RS * K;
  // State fragments in registers, a row each: channels 4*(seg + 8*i) .. +3, i = 0..3.
  float4 st0[4], st1[4];
  {
    const device float4* S0 = (const device float4*)(state_row0 + 4u * seg);
    const device float4* S1 = (const device float4*)(state_row1 + 4u * seg);
    for (uint i = 0; i < 4; i++) {
      st0[i] = fresh ? 0.0f : S0[8u * i];
    }
    for (uint i = 0; i < 4; i++) {
      st1[i] = fresh ? 0.0f : S1[8u * i];
    }
  }

  // ── Block staging ──────────────────────────────────────────────────
  // k/q: this key head's head_k channels per token (4 float4s per lane).
  // v: the threadgroup's 32 value rows. a/b: per token, RAW. The row's 8
  // lanes sit in one simdgroup (rg*KR+seg stays within a 32-thread
  // boundary), so the butterflies never cross simdgroups.
  threadgroup float4 k_s[TB][KR][4];
  threadgroup float4 q_s[TB][KR][4];  // raw q, then q·q_inv (the block's prep)
  threadgroup float v_s[TB][DB];
  threadgroup float g_s[TB];
  threadgroup float b_s[TB];
  // The block's prep: per token each lane's k_inv, decay and 1 + exp(-b).
  threadgroup float kinv_s[TB][KR];
  threadgroup float dec_s[TB];
  threadgroup float den_s[TB];

  const device float* qk_base = conv_out + uint(bos) * conv_dim;
  const uint qk_off = i_h * K;  // q's offset in the row; k's is +key_dim
  // v's section starts at 2*key_dim; a value row's offset within it is the
  // GLOBAL row vr (row0 + rg), already spanning heads.
  const device float* v_base = qk_base + 2u * key_dim;

  // Register prefetch of one block: the thread's share of the k and q
  // float4s (TB*KR*4 each — each lane owns four, channels 4*(lane+8*i)),
  // of the v slice (TB*DB floats), and one raw a/b pair — the cooperative
  // loads Kernel P's GDN_PIPE_FETCH performs.
  constexpr uint NKQ = (TB * KR * 4 + NT - 1) / NT;
  constexpr uint NV = (TB * DB + NT - 1) / NT;
  float4 pk[NKQ], pq[NKQ];
  float pv[NV];
  float pg = 0.0f, pb = 0.0f;

#define GDN_PIPE_FETCH(T0N, TTN)                                          \
  {                                                                       \
    const int ttn = (TTN);                                                \
    _Pragma("unroll")                                                     \
    for (uint j = 0; j < NKQ; j++) {                                      \
      const uint p = tid + j * NT;                                        \
      if (p < uint(ttn) * KR * 4) {                                       \
        const uint t = p / (KR * 4);                                      \
        const uint lane = (p % (KR * 4)) / 4;                             \
        const uint i = p % 4;                                             \
        const device float4* kf = (const device float4*)(                 \
            qk_base + (uint(T0N) + t) * conv_dim + key_dim + qk_off +     \
            4u * (lane + 8u * i));                                        \
        pk[j] = *kf;                                                      \
      }                                                                   \
    }                                                                     \
    _Pragma("unroll")                                                     \
    for (uint j = 0; j < NKQ; j++) {                                      \
      const uint p = tid + j * NT;                                        \
      if (p < uint(ttn) * KR * 4) {                                       \
        const uint t = p / (KR * 4);                                      \
        const uint lane = (p % (KR * 4)) / 4;                             \
        const uint i = p % 4;                                             \
        const device float4* qf = (const device float4*)(                 \
            qk_base + (uint(T0N) + t) * conv_dim + qk_off +               \
            4u * (lane + 8u * i));                                        \
        pq[j] = *qf;                                                      \
      }                                                                   \
    }                                                                     \
    _Pragma("unroll")                                                     \
    for (uint j = 0; j < NV; j++) {                                       \
      const uint p = tid + j * NT;                                        \
      if (p < uint(ttn) * DB) {                                           \
        const uint t = p / DB, r = p % DB;                                \
        pv[j] = v_base[(uint(T0N) + t) * conv_dim + row0 + r];            \
      }                                                                   \
    }                                                                     \
    if (tid < uint(ttn)) {                                                \
      pg = float(a[(uint(bos) + uint(T0N) + tid) * HV + i_hv]);           \
      pb = float(b[(uint(bos) + uint(T0N) + tid) * HV + i_hv]);           \
    }                                                                     \
  }

  int t0 = 0;
  while (t0 < seq_len) {
    const uint tt = uint(min((int)TB, seq_len - t0));
    GDN_PIPE_FETCH(t0, (int)tt)
    // Store the prefetch into threadgroup memory (cooperative).
    _Pragma("unroll")
    for (uint j = 0; j < NKQ; j++) {
      const uint p = tid + j * NT;
      if (p < tt * KR * 4) {
        const uint t = p / (KR * 4);
        const uint lane = (p % (KR * 4)) / 4;
        const uint i = p % 4;
        k_s[t][lane][i] = pk[j];
        q_s[t][lane][i] = pq[j];
      }
    }
    _Pragma("unroll")
    for (uint j = 0; j < NV; j++) {
      const uint p = tid + j * NT;
      if (p < tt * DB) {
        const uint t = p / DB, r = p % DB;
        v_s[t][r] = pv[j];
      }
    }
    if (tid < tt) {
      g_s[tid] = pg;
      b_s[tid] = pb;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Prefetch the NEXT block into registers while this one runs.
    if (t0 + (int)TB < seq_len) {
      const int tn = t0 + (int)TB;
      GDN_PIPE_FETCH(tn, seq_len - tn)
    }

    // ── The block's prep: a token's norms by 8 lanes (the channel split and
    // butterfly the rows' lanes would use), q scaled in place, and its gating.
    for (uint w = tid; w < tt * KR; w += NT) {
      const uint t = w / KR, l = w % KR;
      float4 kc[4], qc[4];
      _Pragma("unroll")
      for (uint i = 0; i < 4; i++) {
        kc[i] = k_s[t][l][i];
        qc[i] = q_s[t][l][i];
      }
      float q_sq = 0.0f, k_sq = 0.0f;
      _Pragma("unroll")
      for (uint i = 0; i < 4; i++) {
        q_sq += dot(qc[i], qc[i]);
        k_sq += dot(kc[i], kc[i]);
      }
      float qs = q_sq, ks = k_sq;
      _Pragma("unroll")
      for (uint m = 4; m >= 1; m /= 2) {
        qs += simd_shuffle_xor(qs, m);
        ks += simd_shuffle_xor(ks, m);
      }
      const float q_inv = rsqrt(qs + 1e-6f) * scale;
      const float k_inv = rsqrt(ks + 1e-6f);
      _Pragma("unroll")
      for (uint i = 0; i < 4; i++) {
        q_s[t][l][i] = qc[i] * q_inv;
      }
      kinv_s[t][l] = k_inv;
    }
    if (tid < tt) {
      const float av = g_s[tid] + dtb;
      const float sp = av <= 20.0f ? log(1.0f + exp(av)) : av;
      dec_s[tid] = exp(sp * neg_a);
      den_s[tid] = exp(-b_s[tid]) + 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── The steps. Each lane reads its own k/q fragments (threadgroup
    // memory; lane seg touches columns 4*seg..4*seg+3 only) and runs both
    // rows' recurrences. The butterflies reduce across a row's 8 lanes
    // (xor 4, 2, 1).
    for (uint t = 0; t < tt; t++) {
      const float decay = dec_s[t];
      const float den = den_s[t];
      float4 kc[4], qn[4];
      _Pragma("unroll")
      for (uint i = 0; i < 4; i++) {
        kc[i] = k_s[t][seg][i];
        qn[i] = q_s[t][seg][i];
      }
      const float k_inv = kinv_s[t][seg];

      // Decay, then the S·k partial dots.
      float p0, p1;
      {
        float2 a2 = 0.0f;
        _Pragma("unroll")
        for (uint i = 0; i < 4; i++) {
          st0[i] = st0[i] * decay;
          a2 += float2(dot(st0[i].xy, kc[i].xy * k_inv), dot(st0[i].zw, kc[i].zw * k_inv));
        }
        p0 = a2.x + a2.y;
      }
      {
        float2 a2 = 0.0f;
        _Pragma("unroll")
        for (uint i = 0; i < 4; i++) {
          st1[i] = st1[i] * decay;
          a2 += float2(dot(st1[i].xy, kc[i].xy * k_inv), dot(st1[i].zw, kc[i].zw * k_inv));
        }
        p1 = a2.x + a2.y;
      }
      _Pragma("unroll")
      for (uint m = 4; m >= 1; m /= 2) {
        p0 += simd_shuffle_xor(p0, m);
        p1 += simd_shuffle_xor(p1, m);
      }

      // S += k·delta; o partial = S·q.
      float out0, out1;
      {
        const float dk = ((v_s[t][rg] - p0) * k_inv) / den;
        float2 o2 = 0.0f;
        _Pragma("unroll")
        for (uint i = 0; i < 4; i++) {
          st0[i] = fma(kc[i], float4(dk), st0[i]);
          o2 += float2(dot(st0[i].xy, qn[i].xy), dot(st0[i].zw, qn[i].zw));
        }
        out0 = o2.x + o2.y;
      }
      {
        const float dk = ((v_s[t][rg + RS] - p1) * k_inv) / den;
        float2 o2 = 0.0f;
        _Pragma("unroll")
        for (uint i = 0; i < 4; i++) {
          st1[i] = fma(kc[i], float4(dk), st1[i]);
          o2 += float2(dot(st1[i].xy, qn[i].xy), dot(st1[i].zw, qn[i].zw));
        }
        out1 = o2.x + o2.y;
      }
      _Pragma("unroll")
      for (uint m = 4; m >= 1; m /= 2) {
        out0 += simd_shuffle_xor(out0, m);
        out1 += simd_shuffle_xor(out1, m);
      }
      if (seg == 0u) {
        o[(uint(bos) + t0 + t) * value_dim + vr] = out0;
        o[(uint(bos) + t0 + t) * value_dim + vr + RS] = out1;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    t0 += (int)TB;
  }
#undef GDN_PIPE_FETCH

  {
    device float4* S0 = (device float4*)(state_row0 + 4u * seg);
    device float4* S1 = (device float4*)(state_row1 + 4u * seg);
    for (uint i = 0; i < 4; i++) {
      S0[8u * i] = st0[i];
    }
    for (uint i = 0; i < 4; i++) {
      S1[8u * i] = st1[i];
    }
  }
}

#define INST_GDN_SCAN_PIPELINED(dtype_tag, mtl_type) \
  SCRATCHY_KERNEL(gdn_scan_pipelined_##dtype_tag, gdn_scan_pipelined<mtl_type>)

INST_GDN_SCAN_PIPELINED(f16,  half)
INST_GDN_SCAN_PIPELINED(bf16, bfloat)
INST_GDN_SCAN_PIPELINED(f32,  float)
