// SPDX-License-Identifier: Apache-2.0
//
// Gated-DeltaNet recurrent gated delta-rule scan, varlen + stateful (Qwen3.5).
//
// Faithful port of CUDA `fused_recurrent_gdn_fwd_kernel`
// (gdn_recurrent_kernels.cu) and `cpu_golden::gdn_recurrent` — the SAME
// one-thread-per-(sequence, value-head, value-dim) mapping with serial head_k
// loops and `b_h[K]` register state. This is NOT the mlx 32-lane simd kernel
// (which requires head_k % 32 == 0 and pre-normalizes q/k in Python); the CUDA
// mapping is the canonical one and handles any head_k. Per token, per head:
//
//   q,k ← L2-normalize over head_k (eps INSIDE sqrt);  q ← q·scale
//   S   ← S·exp(g)                          (decay; g is the raw log-decay)
//   u   ← beta·(v − S·k);   S ← S + u⊗k
//   o   ← S·q
//
// q/k/v are read as column-slices of `conv_out` (layout
// [q:key_dim | k:key_dim | v:value_dim]), so the CUDA `gdn_conv_split` kernel
// is unnecessary. Only `b_h[K]` lives in registers (q/k are re-read from
// memory — cheaper register pressure than CUDA's three K-arrays, identical
// math). GVA: the key head is `i_hv / (HV/H)`.
//
// `conv_out` is model dtype (`T`); `g`/`beta`/`ssm_state`/`o` are f32.
// State layout (cuda-symmetric): ssm_state[entries, HV, head_v, head_k], row =
//   ((entry*HV + i_hv)*head_v + i_v)*head_k. A slot's entries (`scratchy_layers::gdn_state`):
//   two states — bit 16 of `gdn_step[seq]` names the newest — then, with drafts, the record
//   entry: two areas (half an entry each) of draft rows, each its `conv_out` row (`T`, padded to
//   4 bytes) then its `g` and `beta` (f32 [HV] each). `gdn_step[seq]` is
//   `scratchy_layers::gdn_state::GdnStep::encode`; the entries a step reads and writes follow
//   `GdnStep::{base, after}`:
//   - it starts from its base entry — the newest state, or (start = checkpoint r) the state the
//     previous verify step started from, then replays that step's rows 0..=r from the record
//     area of bit 17, through the same loop body that first computed them (no output);
//   - a verify step (drafts) leaves its start state in its base entry, records its draft rows
//     in the other area and writes its last row's state to the other entry; any other step
//     writes its base entry in place.
//
// Baked constants:
//   GDN_SCAN_NUM_K_HEADS (H), GDN_SCAN_NUM_V_HEADS (HV),
//   GDN_SCAN_HEAD_K (K), GDN_SCAN_HEAD_V (head_v), GDN_SCAN_SCALE (1/sqrt(K)).
//
// Dispatch: grid (ceil(head_v/tg), HV, num_seqs); thread = (value-dim, head, seq).

#include <metal_stdlib>
#include "baked.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, GDN_SCAN_NUM_K_HEADS, 0);
SCRATCHY_CONSTANT(uint, GDN_SCAN_NUM_V_HEADS, 1);
SCRATCHY_CONSTANT(uint, GDN_SCAN_HEAD_K, 2);
SCRATCHY_CONSTANT(uint, GDN_SCAN_HEAD_V, 3);
SCRATCHY_CONSTANT(float, GDN_SCAN_SCALE, 4);

// Matches CUDA `MAX_HEAD_K_DIM` (gdn_recurrent_kernels.cu): register state row.
constant constexpr uint GDN_SCAN_KMAX = 128;

template <typename T>
[[kernel]] void gdn_scan_varlen(
    device       float* o             [[buffer(0)]],
    const device T*     conv_out      [[buffer(1)]],
    const device float* g             [[buffer(2)]],
    const device float* beta          [[buffer(3)]],
    device       float* ssm_state     [[buffer(4)]],
    const device int*   cu_seqlens    [[buffer(5)]],
    const device int*   state_indices [[buffer(6)]],
    const device uint*  gdn_step      [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpig [[thread_position_in_grid]])
{
  uint H = GDN_SCAN_NUM_K_HEADS;
  uint HV = GDN_SCAN_NUM_V_HEADS;
  uint K = GDN_SCAN_HEAD_K;
  uint Vd = GDN_SCAN_HEAD_V;
  float scale = GDN_SCAN_SCALE;

  uint key_dim = H * K;
  uint value_dim = HV * Vd;
  uint conv_dim = 2u * key_dim + value_dim;

  uint i_n = tgid.z;   // sequence (seq_axis = Z at lowering time)
  uint i_hv = tgid.y;  // value head
  uint i_v = tpig.x;   // value dim
  if (i_v >= Vd || i_hv >= HV) {
    return;
  }
  uint i_h = i_hv / (HV / H);  // GVA: grouped key head

  int bos = cu_seqlens[i_n];
  int eos = cu_seqlens[i_n + 1];
  int seq_len = eos - bos;
  if (seq_len <= 0) {
    return;
  }
  int entry = state_indices[i_n];
  if (entry < 0) {
    return;
  }
  uint code = gdn_step[i_n];
  uint start = code & 0xffu;
  uint drafts = (code >> 8) & 0xffu;
  uint tip = (code >> 16) & 1u;
  uint area = (code >> 17) & 1u;
  uint entry_len = HV * Vd * K;

  device float* slot = ssm_state + uint(entry) * entry_len;
  uint head_row = (i_hv * Vd + i_v) * K;
  uint base = start >= 2u ? 1u - tip : tip;
  device float* base_row = slot + base * entry_len + head_row;

  // Draft records: a row's `conv_out` row, then its decay and beta.
  uint rec_conv = (conv_dim * uint(sizeof(T)) + 3u) & ~3u;
  uint rec_bytes = rec_conv + 8u * HV;
  device uchar* records = (device uchar*)(slot + 2u * entry_len);
  device uchar* kept = records + area * (entry_len * 2u);
  device uchar* draft_rows = records + (1u - area) * (entry_len * 2u);
  int replayed = start >= 2u ? int(start - 1u) : 0;

  float b_h[GDN_SCAN_KMAX];
  for (uint ki = 0; ki < K; ki++) {
    b_h[ki] = start == 1u ? 0.0f : base_row[ki];
  }

  // The previous verify step's kept rows (replayed: no output), then this step's.
  for (int i_t = -replayed; i_t < seq_len; i_t++) {
    if (i_t == 0 && drafts > 0u && start != 0u) {
      // A verify step keeps its start state in its base entry (a slot start's is already there).
      for (uint ki = 0; ki < K; ki++) {
        base_row[ki] = b_h[ki];
      }
    }
    uint t = uint(bos + i_t);
    device const uchar* rec = kept + uint(i_t + replayed) * rec_bytes;
    const device T* row = i_t < 0 ? (const device T*)rec : conv_out + t * conv_dim;
    const device float* rec_gb = (const device float*)(rec + rec_conv);
    float g_t = i_t < 0 ? rec_gb[i_hv] : g[t * HV + i_hv];
    float beta_t = i_t < 0 ? rec_gb[HV + i_hv] : beta[t * HV + i_hv];
    // q from key-head i_h; k from key-head i_h (offset key_dim); v from value-head i_hv.
    const device T* q_ptr = row + i_h * K;
    const device T* k_ptr = row + key_dim + i_h * K;

    float q_sq = 0.0f, k_sq = 0.0f;
    for (uint ki = 0; ki < K; ki++) {
      float qf = float(q_ptr[ki]);
      float kf = float(k_ptr[ki]);
      q_sq += qf * qf;
      k_sq += kf * kf;
    }
    float q_inv = rsqrt(q_sq + 1e-6f);
    float k_inv = rsqrt(k_sq + 1e-6f);

    float decay = exp(g_t);
    for (uint ki = 0; ki < K; ki++) {
      b_h[ki] *= decay;
    }

    float b_v = float(row[2u * key_dim + i_hv * Vd + i_v]);
    float dot_hk = 0.0f;
    for (uint ki = 0; ki < K; ki++) {
      dot_hk += b_h[ki] * (float(k_ptr[ki]) * k_inv);
    }
    b_v = (b_v - dot_hk) * beta_t;

    float b_o = 0.0f;
    for (uint ki = 0; ki < K; ki++) {
      float kn = float(k_ptr[ki]) * k_inv;
      float qn = float(q_ptr[ki]) * q_inv * scale;
      b_h[ki] += b_v * kn;
      b_o += b_h[ki] * qn;
    }
    if (i_t < 0) {
      continue;
    }
    o[t * value_dim + i_hv * Vd + i_v] = b_o;

    // A draft row, for the replay that keeps it: its conv row split over the sequence's
    // threads, its decay and beta by value-dim 0.
    if (uint(i_t) < drafts) {
      device uchar* dst = draft_rows + uint(i_t) * rec_bytes;
      device T* dst_row = (device T*)dst;
      for (uint j = i_hv * Vd + i_v; j < conv_dim; j += value_dim) {
        dst_row[j] = conv_out[t * conv_dim + j];
      }
      if (i_v == 0u) {
        device float* gb = (device float*)(dst + rec_conv);
        gb[i_hv] = g_t;
        gb[HV + i_hv] = beta_t;
      }
    }
  }

  // The last row's state: a verify step's to the other entry, any other step's in place.
  device float* out_row = slot + (drafts > 0u ? 1u - base : base) * entry_len + head_row;
  for (uint ki = 0; ki < K; ki++) {
    out_row[ki] = b_h[ki];
  }
}

// ---------------------------------------------------------------------------
// gdn_scan_simd — the gating and the scan of a decode step, mlx-lm's `gated_delta_step` mapping.
//
// A simdgroup per (sequence, value head, value dim), its 32 lanes splitting head_k (head_k % 32 ==
// 0): each lane holds head_k / 32 state elements, and every dot over head_k is a simd_sum — four
// per token (q's and k's L2 norms, S·k, S·q) instead of the per-thread serial loops above. Every
// lane computes its head's g and beta as `gdn_gating` does. `o` (f32) feeds `gdn_rms_norm_gated`.
// Its state entries, draft records and replay are `gdn_scan_varlen`'s (the file header), records
// as `gdn_scan_varlen_f32` lays them out (the lowering's instantiation: f32 conv rows), so either
// kernel replays the other's records.
//
// Baked constants: the scan's 0-4.
// Dispatch: grid (1, HV * head_v / 4, num_seqs), threads (32, 4, 1): simdgroup s of threadgroup y
// takes value head y / (head_v / 4), value dim (y % (head_v / 4)) * 4 + s.
// ---------------------------------------------------------------------------

template <typename T>
[[kernel]] void gdn_scan_simd(
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
    uint3 tgid     [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  lane     [[thread_index_in_simdgroup]])
{
  const uint H = GDN_SCAN_NUM_K_HEADS;
  const uint HV = GDN_SCAN_NUM_V_HEADS;
  const uint K = GDN_SCAN_HEAD_K;
  const uint Vd = GDN_SCAN_HEAD_V;
  const uint key_dim = H * K;
  const uint value_dim = HV * Vd;
  const uint conv_dim = 2u * key_dim + value_dim;
  // At least one: the batch bake compiles every kernel of the file at every head_k the others take.
  constexpr uint NPT = GDN_SCAN_HEAD_K >= 32u ? GDN_SCAN_HEAD_K / 32u : 1u;

  const uint groups = Vd / 4u;
  const uint i_hv = tgid.y / groups;
  const uint i_v = (tgid.y % groups) * 4u + simd_gid;
  const uint i_n = tgid.z;
  const uint i_h = i_hv / (HV / H);

  const int bos = cu_seqlens[i_n];
  const int seq_len = cu_seqlens[i_n + 1] - bos;
  const int entry = state_indices[i_n];
  if (seq_len <= 0 || entry < 0 || i_v >= Vd) {
    return;
  }
  const uint code = gdn_step[i_n];
  const uint start = code & 0xffu;
  const uint drafts = (code >> 8) & 0xffu;
  const uint tip = (code >> 16) & 1u;
  const uint area = (code >> 17) & 1u;
  const uint entry_len = HV * Vd * K;

  device float* slot = ssm_state + uint(entry) * entry_len;
  const uint head_row = (i_hv * Vd + i_v) * K + lane * NPT;
  const uint base = start >= 2u ? 1u - tip : tip;
  device float* base_row = slot + base * entry_len + head_row;

  // Draft records: a row's f32 `conv_out` row, then its decay and beta.
  const uint rec_conv = conv_dim * 4u;
  const uint rec_bytes = rec_conv + 8u * HV;
  device uchar* records = (device uchar*)(slot + 2u * entry_len);
  device uchar* kept = records + area * (entry_len * 2u);
  device uchar* draft_rows = records + (1u - area) * (entry_len * 2u);
  const int replayed = start >= 2u ? int(start - 1u) : 0;

  float st[NPT];
  for (uint i = 0; i < NPT; i++) {
    st[i] = start == 1u ? 0.0f : base_row[i];
  }
  const float neg_a = -exp(float(a_log[i_hv]));

  // The previous verify step's kept rows (replayed: no output), then this step's.
  for (int i_t = -replayed; i_t < seq_len; i_t++) {
    if (i_t == 0 && drafts > 0u && start != 0u) {
      // A verify step keeps its start state in its base entry (a slot start's is already there).
      for (uint i = 0; i < NPT; i++) {
        base_row[i] = st[i];
      }
    }
    const uint t = uint(bos + i_t);
    device const uchar* rec = kept + uint(i_t + replayed) * rec_bytes;
    const device float* row = i_t < 0 ? (const device float*)rec : conv_out + t * conv_dim;
    const device float* rec_gb = (const device float*)(rec + rec_conv);
    // gdn_gating, or the recorded row's.
    float g_t, beta;
    if (i_t < 0) {
      g_t = rec_gb[i_hv];
      beta = rec_gb[HV + i_hv];
    } else {
      const float av = float(a[t * HV + i_hv]) + float(dt_bias[i_hv]);
      const float sp = av <= 20.0f ? log(1.0f + exp(av)) : av;
      g_t = neg_a * sp;
      beta = 1.0f / (1.0f + exp(-float(b[t * HV + i_hv])));
    }
    const float decay = exp(g_t);

    const device float* q_ptr = row + i_h * K + lane * NPT;
    const device float* k_ptr = row + key_dim + i_h * K + lane * NPT;
    float q[NPT], k[NPT];
    float q_sq = 0.0f, k_sq = 0.0f;
    for (uint i = 0; i < NPT; i++) {
      q[i] = q_ptr[i];
      k[i] = k_ptr[i];
      q_sq += q[i] * q[i];
      k_sq += k[i] * k[i];
    }
    const float q_inv = rsqrt(simd_sum(q_sq) + 1e-6f) * GDN_SCAN_SCALE;
    const float k_inv = rsqrt(simd_sum(k_sq) + 1e-6f);

    float kv = 0.0f;
    for (uint i = 0; i < NPT; i++) {
      st[i] *= decay;
      kv += st[i] * (k[i] * k_inv);
    }
    kv = simd_sum(kv);
    const float v = row[2u * key_dim + i_hv * Vd + i_v];
    const float delta = (v - kv) * beta;
    float out = 0.0f;
    for (uint i = 0; i < NPT; i++) {
      st[i] += (k[i] * k_inv) * delta;
      out += st[i] * (q[i] * q_inv);
    }
    out = simd_sum(out);
    if (i_t < 0) {
      continue;
    }
    if (lane == 0u) {
      o[t * value_dim + i_hv * Vd + i_v] = out;
    }

    // A draft row, for the replay that keeps it: its conv row split over the sequence's lanes,
    // its decay and beta by value-dim 0's lane 0.
    if (uint(i_t) < drafts) {
      device uchar* dst = draft_rows + uint(i_t) * rec_bytes;
      device float* dst_row = (device float*)dst;
      for (uint j = (i_hv * Vd + i_v) * 32u + lane; j < conv_dim; j += value_dim * 32u) {
        dst_row[j] = conv_out[t * conv_dim + j];
      }
      if (i_v == 0u && lane == 0u) {
        device float* gb = (device float*)(dst + rec_conv);
        gb[i_hv] = g_t;
        gb[HV + i_hv] = beta;
      }
    }
  }

  // The last row's state: a verify step's to the other entry, any other step's in place.
  device float* out_row = slot + (drafts > 0u ? 1u - base : base) * entry_len + head_row;
  for (uint i = 0; i < NPT; i++) {
    out_row[i] = st[i];
  }
}

#define INST_GDN_SCAN_VARLEN(dtype_tag, mtl_type) \
  SCRATCHY_KERNEL(gdn_scan_varlen_##dtype_tag, gdn_scan_varlen<mtl_type>) \
  SCRATCHY_KERNEL(gdn_scan_simd_##dtype_tag, gdn_scan_simd<mtl_type>)

INST_GDN_SCAN_VARLEN(f16,  half)
INST_GDN_SCAN_VARLEN(bf16, bfloat)
INST_GDN_SCAN_VARLEN(f32,  float)
