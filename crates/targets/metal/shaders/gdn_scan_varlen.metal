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
// `conv_out` is model dtype (`T`); `g`/`beta`/`o` and the lane records are f32; `ssm_state` is
// `half`, cast at its loads and stores (the arithmetic f32).
// State layout (cuda-symmetric): `state_indices[seq]` is the sequence's `GdnSeqSlots`
//   (`gdn_state.h`), its slot the one its state lives in; a slot is its state
//   entry ssm[HV, head_v, head_k] (row ((i_hv*head_v + i_v)*head_k)). When the model drafts, the
//   sequence's draft lane (the step code's bits 17-31) is two record areas of `GDN_SCAN_DRAFTS + 1`
//   rows in `ssm_lanes` (`scratchy_layers::gdn_state::GdnStateDims::ssm_lane_len`), each row its
//   f32 conv row, then its decay and beta (f32 [HV] each).
//   `gdn_step[seq]` is `scratchy_layers::gdn_state::GdnStep::encode`:
//   - a step starts from the slot's state, zero (fresh), or (start = checkpoint r) the slot's state
//     replayed through the previous verify step's rows 0..=r from the lane's record area its bit 16
//     names, through the same loop body that first computed them (no output), its state rounded
//     through the entry's dtype after the last replayed row — where the plain run's store and its
//     next step's load sat, so it leaves the same state;
//   - a verify step (drafts) leaves the state it starts from in the slot and records each of its
//     rows in the lane's other area; any other step writes its last row's state to the slot.
//   - a slot start restoring a snapshot (the record's source) reads its start state from the
//     snapshot slot's entry; a save stores the state after its row into its snapshot slot's entry
//     too, as the loop passes the row (the step's own rows: a replayed one is not counted).
//
// Mamba-2 (Nemotron-H, `GDN_SCAN_MAMBA`) runs the same mapping over the same state layout — its
// heads the value heads, its groups the key heads, a head's `[head_dim, state]` matrix the
// `[head_v, head_k]` one — with mlx-lm's selective-scan recurrence (`ssm.py`) in place of the delta
// rule. `conv_out` is `[x:value_dim | B:key_dim | C:key_dim]`; per token, per head h in group g:
//
//   S  ← exp(dt·A)·S + (x·dt) ⊗ B_g          (no L2 norm, no beta, no scale)
//   y  ← S·C_g + D_h·x
//
// `g` carries dt·A and `beta` carries dt (`gdn_gating`'s Mamba-2 form); `D` is buffer 8 (f32).
//
// Baked constants:
//   GDN_SCAN_NUM_K_HEADS (H), GDN_SCAN_NUM_V_HEADS (HV),
//   GDN_SCAN_HEAD_K (K), GDN_SCAN_HEAD_V (head_v), GDN_SCAN_SCALE (1/sqrt(K); Mamba-2: unread),
//   GDN_SCAN_MAMBA (optional: the Mamba-2 recurrence), GDN_SCAN_A_STRIDE / GDN_SCAN_A_OFFSET
//   (optional, `gdn_scan_simd` only: `a`'s row stride and first column — Mamba-2's `dt` slice of
//   the in-projection row; unset: `a` is `[T, HV]`), GDN_SCAN_DRAFTS (the drafts each sequence
//   of a verify step carries: `SPEC_DRAFTS`; 0: `ssm_lanes` unread).
//
// Dispatch: grid (ceil(head_v/tg), HV, num_seqs); thread = (value-dim, head, seq).

#include <metal_stdlib>
#include "baked.h"
#include "gdn_state.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, GDN_SCAN_NUM_K_HEADS, 0);
SCRATCHY_CONSTANT(uint, GDN_SCAN_NUM_V_HEADS, 1);
SCRATCHY_CONSTANT(uint, GDN_SCAN_HEAD_K, 2);
SCRATCHY_CONSTANT(uint, GDN_SCAN_HEAD_V, 3);
SCRATCHY_CONSTANT(float, GDN_SCAN_SCALE, 4);
SCRATCHY_CONSTANT_OPTIONAL(bool, GDN_SCAN_MAMBA, 5);
SCRATCHY_CONSTANT_OPTIONAL(uint, GDN_SCAN_A_STRIDE, 6);
SCRATCHY_CONSTANT_OPTIONAL(uint, GDN_SCAN_A_OFFSET, 7);
SCRATCHY_CONSTANT(uint, GDN_SCAN_DRAFTS, 8);
// `gdn_scan_simd` beside the pipelined scan (`route_gdn`, a geometry it covers): the pipelined scan
// runs every plain sequence of more than one row, so this scan skips them, and every save it takes
// — a plain step's, of one row — is of its last row: stored after its row loop, which stays the
// plain one. Without it (a Mamba-2 model, a geometry the pipelined scan does not cover) this scan
// runs every sequence, saving mid-step in stretches.
SCRATCHY_CONSTANT_OPTIONAL(bool, GDN_SCAN_PLAIN_ROWS_PIPELINED, 9);

// The columns of q / k / v (Mamba-2: C / B / x) in a `conv_out` row.
constant constexpr uint GDN_SCAN_KEY_DIM = GDN_SCAN_NUM_K_HEADS * GDN_SCAN_HEAD_K;
constant constexpr uint GDN_SCAN_VALUE_DIM = GDN_SCAN_NUM_V_HEADS * GDN_SCAN_HEAD_V;
constant constexpr uint GDN_SCAN_COL_Q = GDN_SCAN_MAMBA ? GDN_SCAN_VALUE_DIM + GDN_SCAN_KEY_DIM : 0u;
constant constexpr uint GDN_SCAN_COL_K = GDN_SCAN_MAMBA ? GDN_SCAN_VALUE_DIM : GDN_SCAN_KEY_DIM;
constant constexpr uint GDN_SCAN_COL_V = GDN_SCAN_MAMBA ? 0u : 2u * GDN_SCAN_KEY_DIM;

// Matches CUDA `MAX_HEAD_K_DIM` (gdn_recurrent_kernels.cu): register state row.
constant constexpr uint GDN_SCAN_KMAX = 128;

template <typename T>
[[kernel]] void gdn_scan_varlen(
    device       float* o             [[buffer(0)]],
    const device T*     conv_out      [[buffer(1)]],
    const device float* g             [[buffer(2)]],
    const device float* beta          [[buffer(3)]],
    device       half*  ssm_state     [[buffer(4)]],
    const device int*   cu_seqlens    [[buffer(5)]],
    const device GdnSeqSlots* state_indices [[buffer(6)]],
    const device uint*  gdn_step      [[buffer(7)]],
    const device float* d_skip        [[buffer(8)]],
    device       float* ssm_lanes     [[buffer(9)]],
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
  const GdnSeqSlots seq_slots = state_indices[i_n];
  int slot_ix = seq_slots.slot;
  if (slot_ix < 0) {
    return;
  }
  // A model without drafts (`GDN_SCAN_DRAFTS` 0) neither replays nor records: its scan compiles to
  // the plain one.
  uint code = gdn_step[i_n];
  uint start = code & 0xffu;
  uint drafts = GDN_SCAN_DRAFTS == 0u ? 0u : (code >> 8) & 0xffu;
  uint area = (code >> 16) & 1u;

  // The slot's state entry, and the lane's two record areas of `GDN_SCAN_DRAFTS + 1` rows, each
  // row its f32 conv row, then its decay and beta.
  uint entry_len = HV * Vd * K;
  uint rec_len = conv_dim + 2u * HV;
  uint area_len = (GDN_SCAN_DRAFTS + 1u) * rec_len;
  device half* state_row = ssm_state + uint(slot_ix) * entry_len + (i_hv * Vd + i_v) * K;
  device float* lane = ssm_lanes + (code >> 17) * 2u * area_len;
  const device float* kept = lane + area * area_len;
  device float* recorded = lane + (1u - area) * area_len;
  int replayed = GDN_SCAN_DRAFTS > 0u && start >= 2u ? int(start - 1u) : 0;
  // The start state's row: the slot's, or a restore's snapshot's.
  const device half* start_row =
      ssm_state + uint(gdn_start_slot(seq_slots, start)) * entry_len + (i_hv * Vd + i_v) * K;

  float b_h[GDN_SCAN_KMAX];
  for (uint ki = 0; ki < K; ki++) {
    b_h[ki] = start == 1u ? 0.0f : float(start_row[ki]);
  }

  // The previous verify step's kept rows (replayed: no output), then this step's: in
  // stretches, a save's store after each but the last (`gdn_state.h`).
  const int2 saves_through = gdn_saves_through(seq_slots);
  int i_t = -replayed;
  for (uint stretch = 0; stretch <= GDN_SAVES; stretch++) {
    for (const int stop = gdn_stretch_end(saves_through, stretch, seq_len); i_t < stop; i_t++) {
      if (i_t == 0 && replayed > 0) {
        // The plain run of the replayed rows left their state in the slot and its next step
        // loaded it back — the f16 entry's round-trip; the replay holds the rows in registers,
        // so round the state through the entry's dtype to leave the same state.
        for (uint ki = 0; ki < K; ki++) {
          b_h[ki] = float(half(b_h[ki]));
        }
      }
      if (i_t == 0 && drafts > 0u && start != 0u) {
        // A verify step leaves the state it starts from in the slot (a slot start's is already
        // there).
        for (uint ki = 0; ki < K; ki++) {
          state_row[ki] = half(b_h[ki]);
        }
      }
      uint t = uint(bos + i_t);
      const device float* rec = kept + uint(i_t + replayed) * rec_len;
      const device T* live = conv_out + t * conv_dim;
      auto x = [&](uint j) -> float { return i_t < 0 ? rec[j] : float(live[j]); };
      float g_t = i_t < 0 ? rec[conv_dim + i_hv] : g[t * HV + i_hv];
      float beta_t = i_t < 0 ? rec[conv_dim + HV + i_hv] : beta[t * HV + i_hv];
      // A verify row, for the replay that keeps it: its conv row split over the sequence's
      // threads, its decay and beta by value-dim 0.
      if (drafts > 0u && uint(i_t) <= drafts) {
        device float* dst = recorded + uint(i_t) * rec_len;
        for (uint j = i_hv * Vd + i_v; j < conv_dim; j += value_dim) {
          dst[j] = float(live[j]);
        }
        if (i_v == 0u) {
          dst[conv_dim + i_hv] = g_t;
          dst[conv_dim + HV + i_hv] = beta_t;
        }
      }
      // q from key-head i_h; k from key-head i_h (offset key_dim); v from value-head i_hv.
      uint q_at = GDN_SCAN_COL_Q + i_h * K;
      uint k_at = GDN_SCAN_COL_K + i_h * K;
      if (GDN_SCAN_MAMBA) {
        const float xv = x(GDN_SCAN_COL_V + i_hv * Vd + i_v);
        const float decay = exp(g_t);
        const float x_dt = xv * beta_t;
        float y = 0.0f;
        for (uint ki = 0; ki < K; ki++) {
          b_h[ki] = decay * b_h[ki] + x_dt * x(k_at + ki);
          y += b_h[ki] * x(q_at + ki);
        }
        if (i_t >= 0) {
          o[t * value_dim + i_hv * Vd + i_v] = y + d_skip[i_hv] * xv;
        }
        continue;
      }

      float q_sq = 0.0f, k_sq = 0.0f;
      for (uint ki = 0; ki < K; ki++) {
        float qf = x(q_at + ki);
        float kf = x(k_at + ki);
        q_sq += qf * qf;
        k_sq += kf * kf;
      }
      float q_inv = rsqrt(q_sq + 1e-6f);
      float k_inv = rsqrt(k_sq + 1e-6f);

      float decay = exp(g_t);
      for (uint ki = 0; ki < K; ki++) {
        b_h[ki] *= decay;
      }

      float b_v = x(GDN_SCAN_COL_V + i_hv * Vd + i_v);
      float dot_hk = 0.0f;
      for (uint ki = 0; ki < K; ki++) {
        dot_hk += b_h[ki] * (x(k_at + ki) * k_inv);
      }
      b_v = (b_v - dot_hk) * beta_t;

      float b_o = 0.0f;
      for (uint ki = 0; ki < K; ki++) {
        float kn = x(k_at + ki) * k_inv;
        float qn = x(q_at + ki) * q_inv * scale;
        b_h[ki] += b_v * kn;
        b_o += b_h[ki] * qn;
      }
      if (i_t < 0) {
        continue;
      }
      o[t * value_dim + i_hv * Vd + i_v] = b_o;
    }
    const int sv = gdn_stretch_save(saves_through, stretch, seq_len);
    if (sv >= 0) {
      device half* snap =
          ssm_state + uint(state_indices[i_n].saves[sv].y) * entry_len + (i_hv * Vd + i_v) * K;
      for (uint ki = 0; ki < K; ki++) {
        snap[ki] = half(b_h[ki]);
        // The snapshot is the entry's f16 — a restore loads exactly this. The cold run carries
        // on in registers, so round them through the same store: a prefix-cache hit resumes
        // from what a miss computed, bit for bit.
        b_h[ki] = float(snap[ki]);
      }
    }
  }

  // Any step but a verify step leaves its last row's state in the slot.
  if (drafts == 0u) {
    for (uint ki = 0; ki < K; ki++) {
      state_row[ki] = half(b_h[ki]);
    }
  }
}

// ---------------------------------------------------------------------------
// gdn_scan_simd — the gating and the scan of a decode step, mlx-lm's `gated_delta_step` mapping.
//
// A simdgroup per (sequence, value head, value dim), its 32 lanes splitting head_k (head_k % 32 ==
// 0): each lane holds head_k / 32 state elements, and every dot over head_k is a simd_sum — four
// per token (q's and k's L2 norms, S·k, S·q) instead of the per-thread serial loops above. Every
// lane computes its head's g and beta as `gdn_gating` does. `o` (f32) feeds `gdn_rms_norm_gated`.
// Its state entries, draft lanes, replay and snapshots are `gdn_scan_varlen`'s (the file header),
// records as `gdn_scan_varlen_f32` lays them out (the lowering's instantiation: f32 conv rows), so
// either kernel replays the other's records. `ssm_lanes` is buffer 11.
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
    device       half*  ssm_state     [[buffer(4)]],
    const device int*   cu_seqlens    [[buffer(5)]],
    const device GdnSeqSlots* state_indices [[buffer(6)]],
    const device uint*  gdn_step      [[buffer(7)]],
    const device float* a_log         [[buffer(8)]],
    const device T*     dt_bias       [[buffer(9)]],
    const device float* d_skip        [[buffer(10)]],
    device       float* ssm_lanes     [[buffer(11)]],
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
  const GdnSeqSlots seq_slots = state_indices[i_n];
  const int slot_ix = seq_slots.slot;
  if (seq_len <= 0 || slot_ix < 0 || i_v >= Vd) {
    return;
  }
  // A model without drafts (`GDN_SCAN_DRAFTS` 0) neither replays nor records: its scan compiles to
  // the plain one.
  const uint code = gdn_step[i_n];
  if (GDN_SCAN_PLAIN_ROWS_PIPELINED && gdn_step_plain(code) && seq_len > 1) {
    return;
  }
  const uint start = code & 0xffu;
  const uint drafts = GDN_SCAN_DRAFTS == 0u ? 0u : (code >> 8) & 0xffu;
  const uint area = (code >> 16) & 1u;

  // The slot and the lane, as `gdn_scan_varlen` lays them out.
  const uint entry_len = HV * Vd * K;
  const uint rec_len = conv_dim + 2u * HV;
  const uint area_len = (GDN_SCAN_DRAFTS + 1u) * rec_len;
  // This lane's part of a state row in slot `s`'s entry.
  auto row_in = [&](int s) {
    return ssm_state + uint(s) * entry_len + (i_hv * Vd + i_v) * K + lane * NPT;
  };
  device half* state_row = row_in(slot_ix);
  device float* draft_lane = ssm_lanes + (code >> 17) * 2u * area_len;
  const device float* kept = draft_lane + area * area_len;
  device float* recorded = draft_lane + (1u - area) * area_len;
  const int replayed = GDN_SCAN_DRAFTS > 0u && start >= 2u ? int(start - 1u) : 0;

  // The start state: the slot's, or a restore's snapshot's.
  const device half* start_row = row_in(gdn_start_slot(seq_slots, start));
  float st[NPT];
  for (uint i = 0; i < NPT; i++) {
    st[i] = start == 1u ? 0.0f : float(start_row[i]);
  }
  const float neg_a = -exp(float(a_log[i_hv]));

  // One row of the scan: a kept row the previous verify step recorded (`i_t < 0`, replayed: no
  // output), or this step's row `i_t`.
  auto scan_row = [&](int i_t) {
    if (i_t == 0 && replayed > 0) {
      // The plain run of the replayed rows left their state in the slot and its next step
      // loaded it back — the f16 entry's round-trip; the replay holds the rows in registers,
      // so round the state through the entry's dtype to leave the same state.
      for (uint i = 0; i < NPT; i++) {
        st[i] = float(half(st[i]));
      }
    }
    if (i_t == 0 && drafts > 0u && start != 0u) {
      // A verify step leaves the state it starts from in the slot (a slot start's is already
      // there).
      for (uint i = 0; i < NPT; i++) {
        state_row[i] = half(st[i]);
      }
    }
    const uint t = uint(bos + i_t);
    const device float* rec = kept + uint(i_t + replayed) * rec_len;
    const device float* row = i_t < 0 ? rec : conv_out + t * conv_dim;
    // gdn_gating, or the recorded row's. Mamba-2's beta is its dt.
    float g_t, beta;
    if (i_t < 0) {
      g_t = rec[conv_dim + i_hv];
      beta = rec[conv_dim + HV + i_hv];
    } else {
      const uint a_ix = GDN_SCAN_A_STRIDE_SET ? t * GDN_SCAN_A_STRIDE + GDN_SCAN_A_OFFSET + i_hv
                                              : t * HV + i_hv;
      const float av = float(a[a_ix]) + float(dt_bias[i_hv]);
      const float sp = av <= 20.0f ? log(1.0f + exp(av)) : av;
      g_t = neg_a * sp;
      beta = GDN_SCAN_MAMBA ? sp : 1.0f / (1.0f + exp(-float(b[t * HV + i_hv])));
    }
    const float decay = exp(g_t);
    // A verify row, for the replay that keeps it: its conv row split over the sequence's lanes,
    // its decay and beta by value-dim 0's lane 0.
    if (drafts > 0u && uint(i_t) <= drafts) {
      device float* dst = recorded + uint(i_t) * rec_len;
      for (uint j = (i_hv * Vd + i_v) * 32u + lane; j < conv_dim; j += value_dim * 32u) {
        dst[j] = conv_out[t * conv_dim + j];
      }
      if (i_v == 0u && lane == 0u) {
        dst[conv_dim + i_hv] = g_t;
        dst[conv_dim + HV + i_hv] = beta;
      }
    }

    const device float* q_ptr = row + GDN_SCAN_COL_Q + i_h * K + lane * NPT;
    const device float* k_ptr = row + GDN_SCAN_COL_K + i_h * K + lane * NPT;
    if (GDN_SCAN_MAMBA) {
      const float x = row[GDN_SCAN_COL_V + i_hv * Vd + i_v];
      const float x_dt = x * beta;
      float y = 0.0f;
      for (uint i = 0; i < NPT; i++) {
        st[i] = decay * st[i] + x_dt * k_ptr[i];
        y += st[i] * q_ptr[i];
      }
      y = simd_sum(y);
      if (i_t >= 0 && lane == 0u) {
        o[t * value_dim + i_hv * Vd + i_v] = y + d_skip[i_hv] * x;
      }
      return;
    }
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
    const float v = row[GDN_SCAN_COL_V + i_hv * Vd + i_v];
    const float delta = (v - kv) * beta;
    float out = 0.0f;
    for (uint i = 0; i < NPT; i++) {
      st[i] += (k[i] * k_inv) * delta;
      out += st[i] * (q[i] * q_inv);
    }
    out = simd_sum(out);
    if (i_t >= 0 && lane == 0u) {
      o[t * value_dim + i_hv * Vd + i_v] = out;
    }
  };

  // The previous verify step's kept rows (replayed: no output), then this step's.
  const int2 saves_through = gdn_saves_through(seq_slots);
  if (GDN_SCAN_PLAIN_ROWS_PIPELINED) {
    // The plain step's loop, a save — of a plain step's one row — stored after it.
    for (int i_t = -replayed; i_t < seq_len; i_t++) {
      scan_row(i_t);
    }
    for (uint j = 0; j < GDN_SAVES; j++) {
      if (saves_through[j] == seq_len) {
        device half* snap = row_in(seq_slots.saves[j].y);
        for (uint i = 0; i < NPT; i++) {
          snap[i] = half(st[i]);
        }
      }
    }
  } else {
    // In stretches, a save's store after each but the last (`gdn_state.h`).
    int i_t = -replayed;
    for (uint stretch = 0; stretch <= GDN_SAVES; stretch++) {
      for (const int stop = gdn_stretch_end(saves_through, stretch, seq_len); i_t < stop; i_t++) {
        scan_row(i_t);
      }
      const int sv = gdn_stretch_save(saves_through, stretch, seq_len);
      if (sv >= 0) {
        device half* snap = row_in(seq_slots.saves[sv].y);
        for (uint i = 0; i < NPT; i++) {
          snap[i] = half(st[i]);
          // The snapshot is the entry's f16 — a restore loads exactly this. The cold run
          // carries on in registers, so round them through the same store: a prefix-cache
          // hit resumes from what a miss computed, bit for bit.
          st[i] = float(snap[i]);
        }
      }
    }
  }

  // Any step but a verify step leaves its last row's state in the slot.
  if (drafts == 0u) {
    for (uint i = 0; i < NPT; i++) {
      state_row[i] = half(st[i]);
    }
  }
}

#define INST_GDN_SCAN_VARLEN(dtype_tag, mtl_type) \
  SCRATCHY_KERNEL(gdn_scan_varlen_##dtype_tag, gdn_scan_varlen<mtl_type>) \
  SCRATCHY_KERNEL(gdn_scan_simd_##dtype_tag, gdn_scan_simd<mtl_type>)

INST_GDN_SCAN_VARLEN(f16,  half)
INST_GDN_SCAN_VARLEN(bf16, bfloat)
INST_GDN_SCAN_VARLEN(f32,  float)
