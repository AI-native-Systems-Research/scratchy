// SPDX-License-Identifier: Apache-2.0
//
// gdn_decode — a Gated-DeltaNet layer's decode token in one command: the causal conv, the gating
// and the delta-rule scan, the gated RMSNorm. `gdn_conv1d_varlen`, `gdn_scan_simd` and
// `gdn_rms_norm_gated` run it as three, a barrier apart; this is their statements in their order.
// (The shader compiler contracts and reassociates each kernel's arithmetic its own way under fast
// math, so the two agree to f32 rounding, not to the bit.)
//
// A threadgroup per (sequence, key head) holds everything that head's token touches:
//   - its conv channels — the head's q and k, and the v of the value heads that share the key
//     head — a thread apiece. A channel's conv state is read and rewritten by the thread that owns
//     it, and no other threadgroup reads it; a threadgroup per value head would share q/k state
//     between threadgroups that cannot order their reads before its rewrite.
//   - the scan of those value heads, a simdgroup per value dim at a time, lanes splitting head_k
//     as in `gdn_scan_simd`;
//   - the norm of each of those value heads, its sum of squares reduced in the tree
//     `gdn_rms_norm_gated`'s threadgroup reduces it in (head_v ≤ its 256 threads, so each of
//     their partials is one square, and its halvings at strides ≥ head_v add zeros) — the same
//     additions in the same pairs, so the same sum.
//
// One token a sequence: the one-row decode bucket. q/k/v come straight from the model-dtype
// `qkv` row, the output goes to the model-dtype `out` row; no scratch.
//
// The state is laid out as `gdn_conv1d_varlen` and `gdn_scan_varlen` lay a slot out: with drafts,
// a conv slot is its entry and `GDN_DEC_DRAFTS + 1` checkpoints, an ssm slot its entry and two
// record areas. This kernel runs only a plain step (`scratchy_layers::gdn_state::GdnStep::
// is_plain`: `gdn_step[seq]` starts from the slot or from zero and carries no drafts), so it reads
// and writes the entries alone; a step that replays runs the three commands.
//
// Baked constants: 0-5 the scan's (num_k_heads, num_v_heads, head_k, head_v, scale, the model's
// drafts), 6 the conv kernel width, 7 the norm's eps.
// Dispatch: grid (1, num_k_heads, num_seqs), threads (1024, 1, 1).

#include <metal_stdlib>
#include "baked.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, GDN_DEC_NUM_K_HEADS, 0);
SCRATCHY_CONSTANT(uint, GDN_DEC_NUM_V_HEADS, 1);
SCRATCHY_CONSTANT(uint, GDN_DEC_HEAD_K, 2);
SCRATCHY_CONSTANT(uint, GDN_DEC_HEAD_V, 3);
SCRATCHY_CONSTANT(float, GDN_DEC_SCALE, 4);
SCRATCHY_CONSTANT(uint, GDN_DEC_DRAFTS, 5);
SCRATCHY_CONSTANT(uint, GDN_DEC_CONV_KERNEL, 6);
SCRATCHY_CONSTANT(float, GDN_DEC_EPS, 7);

constant constexpr uint H = GDN_DEC_NUM_K_HEADS;
constant constexpr uint HV = GDN_DEC_NUM_V_HEADS;
constant constexpr uint K = GDN_DEC_HEAD_K;
constant constexpr uint Vd = GDN_DEC_HEAD_V;
// Value heads a key head serves, and their value dims.
constant constexpr uint R = HV / H;
constant constexpr uint RV = R * Vd;
constant constexpr uint KEY_DIM = H * K;
constant constexpr uint VALUE_DIM = HV * Vd;
constant constexpr uint CONV_DIM = 2u * KEY_DIM + VALUE_DIM;
// A threadgroup's conv channels: q, k, then the value heads' v.
constant constexpr uint CHANNELS = 2u * K + RV;
constant constexpr uint THREADS = 1024u;
constant constexpr uint SIMDGROUPS = THREADS / 32u;
constant constexpr uint NPT = K / 32u;
constant constexpr uint STATE_LEN = GDN_DEC_CONV_KERNEL - 1u;
// A slot's conv entries and ssm floats, as the conv and the scan lay them out.
constant constexpr uint CONV_ENTRIES = GDN_DEC_DRAFTS == 0u ? 1u : GDN_DEC_DRAFTS + 2u;
constant constexpr uint SSM_SLOT_LEN =
    HV * Vd * K + (GDN_DEC_DRAFTS == 0u ? 0u : 2u * (GDN_DEC_DRAFTS + 1u) * (CONV_DIM + 2u * HV));
// Value dims a simdgroup scans, and how many it holds the state rows of at once.
constant constexpr uint DIMS = RV / SIMDGROUPS;
constant constexpr uint BATCH = 4u;
// The norm kernel's threadgroup, whose halvings the reduction below repeats.
constant constexpr uint NORM_THREADS = 256u;

static_assert(K % 32u == 0u && K / 32u > 0u, "lanes split head_k");
static_assert(HV % H == 0u, "a key head serves whole value heads");
static_assert(CHANNELS <= THREADS, "a thread a conv channel");
static_assert(RV % SIMDGROUPS == 0u, "simdgroups share the value dims evenly");
static_assert(Vd % 32u == 0u && Vd <= NORM_THREADS, "a value head's first 32 dims are one simdgroup");
static_assert(STATE_LEN < 8u, "the conv window lives in registers");

template <typename T>
[[kernel]] void gdn_decode(
    device       T*     out           [[buffer(0)]],
    const device T*     qkv           [[buffer(1)]],
    const device T*     z             [[buffer(2)]],
    const device T*     a             [[buffer(3)]],
    const device T*     b             [[buffer(4)]],
    const device T*     conv_w        [[buffer(5)]],
    device       float* conv_state    [[buffer(6)]],
    device       float* ssm_state     [[buffer(7)]],
    const device int*   cu_seqlens    [[buffer(8)]],
    const device int*   state_indices [[buffer(9)]],
    const device uint*  gdn_step      [[buffer(10)]],
    const device float* a_log         [[buffer(11)]],
    const device T*     dt_bias       [[buffer(12)]],
    const device float* norm_w        [[buffer(13)]],
    uint3 tgid     [[threadgroup_position_in_grid]],
    uint  tid      [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  lane     [[thread_index_in_simdgroup]])
{
  threadgroup float conv[CHANNELS];
  threadgroup float o[RV];
  threadgroup float sq[RV];
  threadgroup float ss[R];

  const uint h = tgid.y;
  const uint n = tgid.z;
  const int bos = cu_seqlens[n];
  const int slot = state_indices[n];
  if (cu_seqlens[n + 1] - bos <= 0 || slot < 0) {
    return;
  }
  const bool fresh = (gdn_step[n] & 0xffu) == 1u;
  device float* slot_ssm = ssm_state + uint(slot) * SSM_SLOT_LEN;
  const uint t = uint(bos);

  // The conv, a thread a channel: q, k of key head h, then v of its value heads.
  if (tid < CHANNELS) {
    const uint c = tid < 2u * K ? (tid < K ? h * K + tid : KEY_DIM + h * K + tid - K)
                                : 2u * KEY_DIM + h * RV + tid - 2u * K;
    float wlocal[8], window[8];
    for (uint ki = 0; ki <= STATE_LEN; ki++) {
      wlocal[ki] = float(conv_w[c * GDN_DEC_CONV_KERNEL + ki]);
    }
    device float* state = conv_state + (uint(slot) * CONV_ENTRIES * CONV_DIM + c) * STATE_LEN;
    for (uint ki = 0; ki < STATE_LEN; ki++) {
      window[ki] = fresh ? 0.0f : state[ki];
    }
    const float x_t = float(qkv[t * CONV_DIM + c]);
    float sum = x_t * wlocal[STATE_LEN];
    for (uint ki = 0; ki < STATE_LEN; ki++) {
      sum += window[ki] * wlocal[ki];
    }
    conv[tid] = sum / (1.0f + exp(-sum));  // SiLU
    for (uint ki = 0; ki + 1 < STATE_LEN; ki++) {
      window[ki] = window[ki + 1];
    }
    if (STATE_LEN > 0u) {
      window[STATE_LEN - 1u] = x_t;
    }
    for (uint ki = 0; ki < STATE_LEN; ki++) {
      state[ki] = window[ki];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The scan: simdgroup s takes value dims s, s + 32, ..., BATCH state rows in flight.
  for (uint j0 = 0; j0 < DIMS; j0 += BATCH) {
    float st[BATCH][NPT];
    for (uint j = 0; j < BATCH && j0 + j < DIMS; j++) {
      const uint d = (j0 + j) * SIMDGROUPS + simd_gid;
      const uint i_hv = h * R + d / Vd;
      device const float* row = slot_ssm + (i_hv * Vd + d % Vd) * K;
      for (uint i = 0; i < NPT; i++) {
        st[j][i] = fresh ? 0.0f : row[lane * NPT + i];
      }
    }
    for (uint j = 0; j < BATCH && j0 + j < DIMS; j++) {
      const uint d = (j0 + j) * SIMDGROUPS + simd_gid;
      const uint i_hv = h * R + d / Vd;
      const float neg_a = -exp(float(a_log[i_hv]));
      const float av = float(a[t * HV + i_hv]) + float(dt_bias[i_hv]);
      const float sp = av <= 20.0f ? log(1.0f + exp(av)) : av;
      const float decay = exp(neg_a * sp);
      const float beta = 1.0f / (1.0f + exp(-float(b[t * HV + i_hv])));

      float q[NPT], k[NPT];
      float q_sq = 0.0f, k_sq = 0.0f;
      for (uint i = 0; i < NPT; i++) {
        q[i] = conv[lane * NPT + i];
        k[i] = conv[K + lane * NPT + i];
        q_sq += q[i] * q[i];
        k_sq += k[i] * k[i];
      }
      const float q_inv = rsqrt(simd_sum(q_sq) + 1e-6f) * GDN_DEC_SCALE;
      const float k_inv = rsqrt(simd_sum(k_sq) + 1e-6f);

      float kv = 0.0f;
      for (uint i = 0; i < NPT; i++) {
        st[j][i] *= decay;
        kv += st[j][i] * (k[i] * k_inv);
      }
      kv = simd_sum(kv);
      const float v = conv[2u * K + d];
      const float delta = (v - kv) * beta;
      float od = 0.0f;
      for (uint i = 0; i < NPT; i++) {
        st[j][i] += (k[i] * k_inv) * delta;
        od += st[j][i] * (q[i] * q_inv);
      }
      od = simd_sum(od);
      if (lane == 0u) {
        o[d] = od;
        sq[d] = od * od;
      }
      device float* row = slot_ssm + (i_hv * Vd + d % Vd) * K;
      for (uint i = 0; i < NPT; i++) {
        row[lane * NPT + i] = st[j][i];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Each value head's sum of squares: the norm kernel's halvings, through threadgroup memory
  // while they cross simdgroups, then within the head's first simdgroup.
  const uint head = tid / Vd;
  const uint e = tid % Vd;
  for (uint s = NORM_THREADS / 2u; s >= 32u; s >>= 1) {
    if (s < Vd) {
      if (head < R && e < s) {
        sq[tid] += sq[tid + s];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (head < R && e < 32u) {
    float p = sq[tid];
    for (uint s = 16u; s > 0u; s >>= 1) {
      p += simd_shuffle_down(p, s);
    }
    if (e == 0u) {
      ss[head] = p;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < RV) {
    const float inv = rsqrt(ss[head] / float(Vd) + GDN_DEC_EPS);
    const uint row = t * VALUE_DIM + h * RV + tid;
    const float zi = float(z[row]);
    const float silu_z = zi / (1.0f + exp(-zi));
    out[row] = T(o[tid] * inv * float(norm_w[e]) * silu_z);
  }
}

#define INST_GDN_DECODE(dtype_tag, mtl_type) \
  SCRATCHY_KERNEL(gdn_decode_##dtype_tag, gdn_decode<mtl_type>)

INST_GDN_DECODE(f16,  half)
INST_GDN_DECODE(bf16, bfloat)
INST_GDN_DECODE(f32,  float)
