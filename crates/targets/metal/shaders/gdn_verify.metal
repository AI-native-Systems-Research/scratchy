// SPDX-License-Identifier: Apache-2.0
//
// gdn_verify — a Gated-DeltaNet layer's step of a few rows a sequence in one command: a verify
// step's last token and drafts, or a short plain step. `gdn_conv1d_varlen`, `gdn_scan_simd` and
// `gdn_rms_norm_gated` run such a step as three, a barrier apart, through device scratch; this
// is their statements in their order (Splash's `verify_gdn_fused`, Inco AI, Apache-2.0: the
// phases hand each other their operands in threadgroup memory). Every row's conv, its q/k norms
// and its gates are computed once for the threadgroup, where the simd scan recomputes them in
// each of its value dims' simdgroups and reads them back from device scratch.
//
// A threadgroup per (sequence, key head, value head), 1024 threads — `gdn_decode`'s one a
// (sequence, key head), split by the value heads it serves so the machine fills (a key head's
// value heads are independent: each channel's conv and each value dim's scan state are its own).
// Every sub-threadgroup computes its key head's q and k conv into its own threadgroup memory
// (the norms and the scan read them), each from their seed; each value channel's state is
// persisted by its owner alone, and the q and k channels' by `gdn_verify_qk_state`, the command
// after, a barrier apart: a threadgroup reading a q or k seed would otherwise race the one
// overwriting it, since no ordering holds between threadgroups (#914 made every verify step's
// greedy output vary run to run).
//   1. the conv, a thread a channel (the head's q and k, and this threadgroup's value head's v),
//      walking the sequence's rows: each row's output into threadgroup memory and a verify row's
//      conv row into the lane's records; a value channel's verify rows' windows into its draft
//      lane's checkpoints and its ring after the last row into the slot — `gdn_conv1d_varlen`'s
//      seed, checkpoints, saves and ring;
//   2. each row's q and k norms (a simdgroup a row) and each row's value head's gates, a verify
//      row's into the records; a replayed row's k norm, from its record;
//   3. the scan, a simdgroup per value dim at a time, BATCH state rows in flight, lanes splitting
//      head_k: the replayed rows from their records, then the step's rows, their outputs into
//      threadgroup memory — `gdn_scan_simd`'s start, replay, rounding, start-state store, saves
//      and last-row store;
//   4. each row's value head's gated RMSNorm, the sum of squares in the halvings
//      `gdn_rms_norm_gated` reduces it in, into the model-dtype `out` rows.
// (The shader compiler contracts each kernel's arithmetic its own way under fast math, so this
// and the three agree to f32 rounding, not to the bit.)
//
// Its state contract is theirs (`gdn_state.h`, `scratchy_layers::gdn_state`): the slot's conv
// ring (f32) and ssm entry (`half`, arithmetic f32), a verify step's draft lane — `k + 1` conv
// checkpoints and two record areas of `k + 1` rows — and the step code's start, drafts, record
// area and lane. A sequence's rows are at most GDN_VER_ROWS (the bucket's, at most 8).
//
// Baked constants: 0-4 the scan's (num_k_heads, num_v_heads, head_k, head_v, scale), 5 the conv
// kernel width, 6 the norm's eps, 7 the rows a sequence (GDN_VER_ROWS), 8 the model's drafts
// (`SPEC_DRAFTS`, the lanes' layout; 0: the lanes, buffers 14 and 15, unread).
// Dispatch: grid (1, num_k_heads * R, num_seqs) — R = num_v_heads / num_k_heads value heads a
// key head serves, one threadgroup each — threads (1024, 1, 1); then `gdn_verify_qk_state`, grid
// (1, num_k_heads, num_seqs), threads (1024, 1, 1), its bindings the same.

#include <metal_stdlib>
#include "baked.h"
#include "gdn_state.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, GDN_VER_NUM_K_HEADS, 0);
SCRATCHY_CONSTANT(uint, GDN_VER_NUM_V_HEADS, 1);
SCRATCHY_CONSTANT(uint, GDN_VER_HEAD_K, 2);
SCRATCHY_CONSTANT(uint, GDN_VER_HEAD_V, 3);
SCRATCHY_CONSTANT(float, GDN_VER_SCALE, 4);
SCRATCHY_CONSTANT(uint, GDN_VER_CONV_KERNEL, 5);
SCRATCHY_CONSTANT(float, GDN_VER_EPS, 6);
SCRATCHY_CONSTANT(uint, GDN_VER_ROWS, 7);
SCRATCHY_CONSTANT(uint, GDN_VER_DRAFTS, 8);

constant constexpr uint H = GDN_VER_NUM_K_HEADS;
constant constexpr uint HV = GDN_VER_NUM_V_HEADS;
constant constexpr uint K = GDN_VER_HEAD_K;
constant constexpr uint Vd = GDN_VER_HEAD_V;
constant constexpr uint ROWS = GDN_VER_ROWS;
// Value heads a key head serves; a threadgroup serves one of them.
constant constexpr uint R = HV / H;
constant constexpr uint KEY_DIM = H * K;
constant constexpr uint VALUE_DIM = HV * Vd;
constant constexpr uint CONV_DIM = 2u * KEY_DIM + VALUE_DIM;
// A threadgroup's conv channels: the head's q and k, and its value head's v.
constant constexpr uint CHANNELS = 2u * K + Vd;
constant constexpr uint THREADS = 1024u;
constant constexpr uint SIMDGROUPS = THREADS / 32u;
constant constexpr uint NPT = K / 32u;
constant constexpr uint STATE_LEN = GDN_VER_CONV_KERNEL - 1u;
// A slot's ssm elements, as the scan lays them out.
constant constexpr uint SSM_SLOT_LEN = HV * Vd * K;
// A lane's record row (the conv row, then each value head's g and beta) and record area.
constant constexpr uint REC_LEN = CONV_DIM + 2u * HV;
constant constexpr uint AREA_LEN = (GDN_VER_DRAFTS + 1u) * REC_LEN;
// The rows a step replays: the previous verify step's kept rows, at most its drafts and token.
constant constexpr uint REPLAY_MAX = GDN_VER_DRAFTS + 1u;
// The value head this threadgroup serves (fixed): its value dims and gates.
constant constexpr uint RV = Vd;
// Value dims a simdgroup scans, and how many it holds the state rows of at once.
constant constexpr uint DIMS = RV / SIMDGROUPS;
constant constexpr uint BATCH = 4u;

static_assert(K % 32u == 0u && K / 32u > 0u, "lanes split head_k");
static_assert(HV % H == 0u, "a key head serves whole value heads");
static_assert(CHANNELS <= THREADS, "a thread a conv channel");
static_assert(RV % SIMDGROUPS == 0u && DIMS % BATCH == 0u, "simdgroups share the value dims evenly");
static_assert(Vd == 128u, "a value head's norm halves 128 dims into one simdgroup");
static_assert(ROWS >= 1u && ROWS <= 8u && ROWS <= SIMDGROUPS, "a simdgroup a row's norm");
static_assert(REPLAY_MAX <= 8u && ROWS + REPLAY_MAX <= SIMDGROUPS, "a simdgroup a row's k norm");
static_assert(STATE_LEN < 8u, "the conv window lives in registers");
// The rows' conv and outputs, their norms and gates: within a threadgroup's 32 KB.
static_assert(ROWS * (CHANNELS + RV) * 4u + (2u * ROWS + REPLAY_MAX + 3u * ROWS) * 4u <= 32768u,
              "the rows' operands fit threadgroup memory");

// The global conv channel of threadgroup channel `c` for key head `h`, value head `i`: the q and
// k channels are the head's own, this threadgroup's value channels its value head's.
inline uint gdn_ver_channel(uint h, uint i, uint c) {
  return c < 2u * K ? (c < K ? h * K + c : KEY_DIM + h * K + c - K)
                    : 2u * KEY_DIM + (h * R + i) * Vd + c - 2u * K;
}

// The conv state channel `c` of sequence `n` leaves after its step's `rows` rows from `t0` on,
// walked from its `seed`: a verify row's window into the draft lane's checkpoint, each save's
// window into its snapshot, and the ring after the last row into the slot (`gdn_state.h`).
template <typename T>
inline void gdn_ver_persist_conv(thread const float* seed, uint c, uint rows, uint t0, uint drafts,
                                 int slot, int2 saves_through, GdnSeqSlots seq_slots,
                                 device float* lane_ptr, device float* conv_state,
                                 const device T* qkv) {
  auto ring = [&](int s) { return conv_state + (uint(s) * CONV_DIM + c) * STATE_LEN; };
  float window[8];
  for (uint ki = 0; ki < STATE_LEN; ki++) {
    window[ki] = seed[ki];
  }
  for (uint t = 0; t < rows; t++) {
    for (uint ki = 0; ki + 1 < STATE_LEN; ki++) {
      window[ki] = window[ki + 1];
    }
    if (STATE_LEN > 0u) {
      window[STATE_LEN - 1u] = float(qkv[(t0 + t) * CONV_DIM + c]);
    }
    if (drafts > 0u && t <= drafts) {
      for (uint ki = 0; ki < STATE_LEN; ki++) {
        lane_ptr[t * CONV_DIM * STATE_LEN + ki] = window[ki];
      }
    }
    for (uint sv = 0; sv < GDN_SAVES; sv++) {
      if (gdn_saves_after(saves_through, sv, int(t))) {
        device float* snap = ring(seq_slots.saves[sv].y);
        for (uint ki = 0; ki < STATE_LEN; ki++) {
          snap[ki] = window[ki];
        }
      }
    }
  }
  // The ring: the last STATE_LEN inputs of `[seed ++ x]`.
  device float* state = ring(slot);
  for (uint ki = 0; ki < STATE_LEN; ki++) {
    const int p = int(rows) + int(ki) - int(STATE_LEN);
    state[ki] = p < 0 ? seed[rows + ki] : float(qkv[(t0 + uint(p)) * CONV_DIM + c]);
  }
}

template <typename T>
[[kernel]] void gdn_verify(
    device       T*     out           [[buffer(0)]],
    const device T*     qkv           [[buffer(1)]],
    const device T*     z             [[buffer(2)]],
    const device T*     a             [[buffer(3)]],
    const device T*     b             [[buffer(4)]],
    const device T*     conv_w        [[buffer(5)]],
    device       float* conv_state    [[buffer(6)]],
    device       half*  ssm_state     [[buffer(7)]],
    const device int*   cu_seqlens    [[buffer(8)]],
    const device GdnSeqSlots* state_indices [[buffer(9)]],
    const device uint*  gdn_step      [[buffer(10)]],
    const device float* a_log         [[buffer(11)]],
    const device T*     dt_bias       [[buffer(12)]],
    const device float* norm_w        [[buffer(13)]],
    device       float* conv_lanes    [[buffer(14)]],
    device       float* ssm_lanes     [[buffer(15)]],
    uint3 tgid     [[threadgroup_position_in_grid]],
    uint  tid      [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  lane     [[thread_index_in_simdgroup]])
{
  threadgroup float conv[ROWS][CHANNELS];
  threadgroup float o[ROWS][RV];
  threadgroup float q_inv[ROWS];
  threadgroup float k_inv[ROWS];
  threadgroup float k_inv_replay[REPLAY_MAX];
  threadgroup float decay[ROWS];
  threadgroup float beta[ROWS];
  threadgroup float ss[ROWS];

  const uint n = tgid.z;
  const uint h = tgid.y / R;
  const uint i = tgid.y % R;
  const uint i_hv = h * R + i;
  const int bos = cu_seqlens[n];
  const int seq_len = cu_seqlens[n + 1] - bos;
  const GdnSeqSlots seq_slots = state_indices[n];
  const int slot = seq_slots.slot;
  if (seq_len <= 0 || slot < 0) {
    return;
  }
  // A bucket's rows bound every sequence's (`cu_seqlens` spans at most the bucket), so this is
  // `seq_len`; the min keeps the threadgroup arrays' row index in bounds regardless.
  const uint rows = min(uint(seq_len), ROWS);
  const uint code = gdn_step[n];
  const uint start = code & 0xffu;
  const bool fresh = start == 1u;
  // A model without drafts neither replays nor records.
  const uint drafts = GDN_VER_DRAFTS == 0u ? 0u : (code >> 8) & 0xffu;
  const bool replays = GDN_VER_DRAFTS > 0u && start >= 2u;
  const uint replayed = replays ? start - 1u : 0u;
  const uint area = (code >> 16) & 1u;
  const uint lane_ix = code >> 17;
  const int2 saves_through = gdn_saves_through(seq_slots);
  // The lane's record areas: the previous verify step's kept rows, and this step's.
  device float* draft_lane = ssm_lanes + lane_ix * 2u * AREA_LEN;
  const device float* kept = draft_lane + area * AREA_LEN;
  device float* recorded = draft_lane + (1u - area) * AREA_LEN;
  const uint t0 = uint(bos);

  // 1. The conv, a thread a channel.
  if (tid < CHANNELS) {
    const uint c = gdn_ver_channel(h, i, tid);
    // A v channel is its owner's alone: it records its rows and persists its state. A q or k
    // channel is every sub-threadgroup's (their norms and scans read its conv): the first records
    // its rows, and `gdn_verify_qk_state` persists its state once every one has read its seed.
    const bool owns = tid >= 2u * K;
    const bool records = owns || i == 0u;
    float wlocal[8], window[8], seed[8];
    for (uint ki = 0; ki <= STATE_LEN; ki++) {
      wlocal[ki] = float(conv_w[c * GDN_VER_CONV_KERNEL + ki]);
    }
    // This channel's window in slot `s`, and in the lane's checkpoint at row `r`.
    auto ring = [&](int s) { return conv_state + (uint(s) * CONV_DIM + c) * STATE_LEN; };
    device float* lane_ptr = conv_lanes + (lane_ix * (GDN_VER_DRAFTS + 1u) * CONV_DIM + c) * STATE_LEN;
    const device float* start_ptr = replays ? lane_ptr + (start - 2u) * CONV_DIM * STATE_LEN
                                            : ring(gdn_start_slot(seq_slots, start));
    // The seed stays in registers: a replay's lives in the checkpoints this step overwrites.
    for (uint ki = 0; ki < STATE_LEN; ki++) {
      seed[ki] = fresh ? 0.0f : start_ptr[ki];
      window[ki] = seed[ki];
    }
    for (uint t = 0; t < rows; t++) {
      const float x_t = float(qkv[(t0 + t) * CONV_DIM + c]);
      float sum = x_t * wlocal[STATE_LEN];
      for (uint ki = 0; ki < STATE_LEN; ki++) {
        sum += window[ki] * wlocal[ki];
      }
      const float y = sum / (1.0f + exp(-sum));  // SiLU
      conv[t][tid] = y;
      for (uint ki = 0; ki + 1 < STATE_LEN; ki++) {
        window[ki] = window[ki + 1];
      }
      if (STATE_LEN > 0u) {
        window[STATE_LEN - 1u] = x_t;
      }
      if (records && drafts > 0u && t <= drafts) {
        recorded[t * REC_LEN + c] = y;
      }
    }
    if (owns) {
      gdn_ver_persist_conv<T>(seed, c, rows, t0, drafts, slot, saves_through, seq_slots,
                              lane_ptr, conv_state, qkv);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2. Each row's q and k norms, a simdgroup a row (each lane its NPT elements, as the scan's);
  //    a replayed row's k norm from its record; each row's value heads' gates.
  if (simd_gid < rows) {
    const uint t = simd_gid;
    float q_sq = 0.0f, k_sq = 0.0f;
    for (uint i = 0; i < NPT; i++) {
      const float q = conv[t][lane * NPT + i];
      const float k = conv[t][K + lane * NPT + i];
      q_sq += q * q;
      k_sq += k * k;
    }
    q_sq = simd_sum(q_sq);
    k_sq = simd_sum(k_sq);
    if (lane == 0u) {
      q_inv[t] = rsqrt(q_sq + 1e-6f) * GDN_VER_SCALE;
      k_inv[t] = rsqrt(k_sq + 1e-6f);
    }
  } else if (simd_gid >= ROWS && simd_gid - ROWS < replayed) {
    const uint r = simd_gid - ROWS;
    const device float* rec = kept + r * REC_LEN;
    float k_sq = 0.0f;
    for (uint i = 0; i < NPT; i++) {
      const float k = rec[KEY_DIM + h * K + lane * NPT + i];
      k_sq += k * k;
    }
    k_sq = simd_sum(k_sq);
    if (lane == 0u) {
      k_inv_replay[r] = rsqrt(k_sq + 1e-6f);
    }
  }
  if (tid < rows) {
    const uint t = tid;
    const uint row = t0 + t;
    const float neg_a = -exp(float(a_log[i_hv]));
    const float av = float(a[row * HV + i_hv]) + float(dt_bias[i_hv]);
    const float sp = av <= 20.0f ? log(1.0f + exp(av)) : av;
    const float g_t = neg_a * sp;
    const float bt = 1.0f / (1.0f + exp(-float(b[row * HV + i_hv])));
    decay[t] = exp(g_t);
    beta[t] = bt;
    if (drafts > 0u && t <= drafts) {
      recorded[t * REC_LEN + CONV_DIM + i_hv] = g_t;
      recorded[t * REC_LEN + CONV_DIM + HV + i_hv] = bt;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 3. The scan: simdgroup s takes this value head's dims s, s + 32, ..., BATCH state rows in
  //    flight.
  const uint start_slot = uint(gdn_start_slot(seq_slots, start));
  for (uint j0 = 0; j0 < DIMS; j0 += BATCH) {
    float st[BATCH][NPT];
    uint dv[BATCH];
    for (uint j = 0; j < BATCH; j++) {
      const uint d = (j0 + j) * SIMDGROUPS + simd_gid;
      dv[j] = d;
      const device half* row = ssm_state + start_slot * SSM_SLOT_LEN + (i_hv * Vd + dv[j]) * K;
      for (uint i = 0; i < NPT; i++) {
        st[j][i] = fresh ? 0.0f : float(row[lane * NPT + i]);
      }
    }
    // The previous verify step's kept rows, from their records: no output.
    for (uint r = 0; r < replayed; r++) {
      const device float* rec = kept + r * REC_LEN;
      float k[NPT];
      for (uint i = 0; i < NPT; i++) {
        k[i] = rec[KEY_DIM + h * K + lane * NPT + i] * k_inv_replay[r];
      }
      float kv[BATCH];
      for (uint j = 0; j < BATCH; j++) {
        const float dc = exp(rec[CONV_DIM + i_hv]);
        kv[j] = 0.0f;
        for (uint i = 0; i < NPT; i++) {
          st[j][i] *= dc;
          kv[j] += st[j][i] * k[i];
        }
      }
      for (uint j = 0; j < BATCH; j++) {
        kv[j] = simd_sum(kv[j]);
      }
      for (uint j = 0; j < BATCH; j++) {
        const float v = rec[2u * KEY_DIM + i_hv * Vd + dv[j]];
        const float delta = (v - kv[j]) * rec[CONV_DIM + HV + i_hv];
        for (uint i = 0; i < NPT; i++) {
          st[j][i] += k[i] * delta;
        }
      }
    }
    if (replayed > 0u) {
      // The replayed rows' state rounds through the entry's dtype, as the plain run that kept
      // them would have left it in the slot.
      for (uint j = 0; j < BATCH; j++) {
        for (uint i = 0; i < NPT; i++) {
          st[j][i] = float(half(st[j][i]));
        }
      }
    }
    if (drafts > 0u && start != 0u) {
      // A verify step leaves the state it starts from in the slot (a slot start's is there).
      for (uint j = 0; j < BATCH; j++) {
        device half* row = ssm_state + uint(slot) * SSM_SLOT_LEN + (i_hv * Vd + dv[j]) * K;
        for (uint i = 0; i < NPT; i++) {
          row[lane * NPT + i] = half(st[j][i]);
        }
      }
    }
    for (uint t = 0; t < rows; t++) {
      float k[NPT], q[NPT];
      for (uint i = 0; i < NPT; i++) {
        k[i] = conv[t][K + lane * NPT + i] * k_inv[t];
        q[i] = conv[t][lane * NPT + i] * q_inv[t];
      }
      float kv[BATCH];
      for (uint j = 0; j < BATCH; j++) {
        const float dc = decay[t];
        kv[j] = 0.0f;
        for (uint i = 0; i < NPT; i++) {
          st[j][i] *= dc;
          kv[j] += st[j][i] * k[i];
        }
      }
      for (uint j = 0; j < BATCH; j++) {
        kv[j] = simd_sum(kv[j]);
      }
      float od[BATCH];
      for (uint j = 0; j < BATCH; j++) {
        const uint d = (j0 + j) * SIMDGROUPS + simd_gid;
        const float delta = (conv[t][2u * K + d] - kv[j]) * beta[t];
        od[j] = 0.0f;
        for (uint i = 0; i < NPT; i++) {
          st[j][i] += k[i] * delta;
          od[j] += st[j][i] * q[i];
        }
      }
      for (uint j = 0; j < BATCH; j++) {
        od[j] = simd_sum(od[j]);
      }
      if (lane == 0u) {
        for (uint j = 0; j < BATCH; j++) {
          o[t][(j0 + j) * SIMDGROUPS + simd_gid] = od[j];
        }
      }
      for (uint sv = 0; sv < GDN_SAVES; sv++) {
        if (gdn_saves_after(saves_through, sv, int(t))) {
          for (uint j = 0; j < BATCH; j++) {
            device half* snap = ssm_state + uint(seq_slots.saves[sv].y) * SSM_SLOT_LEN +
                                (i_hv * Vd + dv[j]) * K;
            for (uint i = 0; i < NPT; i++) {
              snap[lane * NPT + i] = half(st[j][i]);
              st[j][i] = float(snap[lane * NPT + i]);
            }
          }
        }
      }
    }
    // Any step but a verify step leaves its last row's state in the slot.
    if (drafts == 0u) {
      for (uint j = 0; j < BATCH; j++) {
        device half* row = ssm_state + uint(slot) * SSM_SLOT_LEN + (i_hv * Vd + dv[j]) * K;
        for (uint i = 0; i < NPT; i++) {
          row[lane * NPT + i] = half(st[j][i]);
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 4. Each row's value head's sums of squares, a simdgroup a row: the norm kernel's halvings
  //    of the head's 128 dims (64, then 32), then the simdgroup's shuffles down.
  if (simd_gid < rows) {
    const uint t = simd_gid;
    const uint e = lane;
    const float s0 = o[t][e] * o[t][e], s1 = o[t][e + 32u] * o[t][e + 32u];
    const float s2 = o[t][e + 64u] * o[t][e + 64u], s3 = o[t][e + 96u] * o[t][e + 96u];
    float p = (s0 + s2) + (s1 + s3);
    for (uint s = 16u; s > 0u; s >>= 1) {
      p += simd_shuffle_down(p, s);
    }
    if (lane == 0u) {
      ss[t] = p;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint x = tid; x < rows * RV; x += THREADS) {
    const uint t = x / RV;
    const uint j = x % RV;
    const float inv = rsqrt(ss[t] / float(Vd) + GDN_VER_EPS);
    const uint row = (t0 + t) * VALUE_DIM + i_hv * Vd + j;
    const float zi = float(z[row]);
    const float silu_z = zi / (1.0f + exp(-zi));
    out[row] = T(o[t][j] * inv * float(norm_w[j]) * silu_z);
  }
}

// gdn_verify_qk_state — the conv state a gdn_verify step's q and k channels leave, the command
// after it: every one of its threadgroups has read their seed by then. A threadgroup a (sequence,
// key head), a thread a q or k channel of it; gdn_verify's bindings, of which it reads the rows,
// the slots, the step codes and the conv state and lanes.
template <typename T>
[[kernel]] void gdn_verify_qk_state(
    device       T*     out           [[buffer(0)]],
    const device T*     qkv           [[buffer(1)]],
    const device T*     z             [[buffer(2)]],
    const device T*     a             [[buffer(3)]],
    const device T*     b             [[buffer(4)]],
    const device T*     conv_w        [[buffer(5)]],
    device       float* conv_state    [[buffer(6)]],
    device       half*  ssm_state     [[buffer(7)]],
    const device int*   cu_seqlens    [[buffer(8)]],
    const device GdnSeqSlots* state_indices [[buffer(9)]],
    const device uint*  gdn_step      [[buffer(10)]],
    const device float* a_log         [[buffer(11)]],
    const device T*     dt_bias       [[buffer(12)]],
    const device float* norm_w        [[buffer(13)]],
    device       float* conv_lanes    [[buffer(14)]],
    device       float* ssm_lanes     [[buffer(15)]],
    uint3 tgid     [[threadgroup_position_in_grid]],
    uint  tid      [[thread_index_in_threadgroup]])
{
  const uint n = tgid.z;
  const uint h = tgid.y;
  const int bos = cu_seqlens[n];
  const int seq_len = cu_seqlens[n + 1] - bos;
  const GdnSeqSlots seq_slots = state_indices[n];
  const int slot = seq_slots.slot;
  if (seq_len <= 0 || slot < 0 || tid >= 2u * K) {
    return;
  }
  const uint rows = min(uint(seq_len), ROWS);
  const uint code = gdn_step[n];
  const uint start = code & 0xffu;
  const bool fresh = start == 1u;
  const uint drafts = GDN_VER_DRAFTS == 0u ? 0u : (code >> 8) & 0xffu;
  const bool replays = GDN_VER_DRAFTS > 0u && start >= 2u;
  const uint lane_ix = code >> 17;
  const uint c = gdn_ver_channel(h, 0u, tid);
  // The seed gdn_verify read: the slot's ring, a restore's snapshot, or a replay's checkpoint.
  device float* lane_ptr = conv_lanes + (lane_ix * (GDN_VER_DRAFTS + 1u) * CONV_DIM + c) * STATE_LEN;
  const device float* start_ptr =
      replays ? lane_ptr + (start - 2u) * CONV_DIM * STATE_LEN
              : conv_state + (uint(gdn_start_slot(seq_slots, start)) * CONV_DIM + c) * STATE_LEN;
  float seed[8];
  for (uint ki = 0; ki < STATE_LEN; ki++) {
    seed[ki] = fresh ? 0.0f : start_ptr[ki];
  }
  gdn_ver_persist_conv<T>(seed, c, rows, uint(bos), drafts, slot, gdn_saves_through(seq_slots),
                          seq_slots, lane_ptr, conv_state, qkv);
}

#define INST_GDN_VERIFY(dtype_tag, mtl_type) \
  SCRATCHY_KERNEL(gdn_verify_##dtype_tag, gdn_verify<mtl_type>) \
  SCRATCHY_KERNEL(gdn_verify_qk_state_##dtype_tag, gdn_verify_qk_state<mtl_type>)

INST_GDN_VERIFY(f16,  half)
INST_GDN_VERIFY(bf16, bfloat)
INST_GDN_VERIFY(f32,  float)
