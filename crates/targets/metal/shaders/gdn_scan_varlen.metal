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
// State layout (cuda-symmetric): ssm_state[num_slots, HV, head_v, head_k],
//   row = ((slot*HV + i_hv)*head_v + i_v)*head_k. is_fresh → S starts at 0.
//
// Function constants:
//   GDN_SCAN_NUM_K_HEADS (H), GDN_SCAN_NUM_V_HEADS (HV),
//   GDN_SCAN_HEAD_K (K), GDN_SCAN_HEAD_V (head_v), GDN_SCAN_SCALE (1/sqrt(K)).
//
// Dispatch: grid (ceil(head_v/tg), HV, num_seqs); thread = (value-dim, head, seq).

#include <metal_stdlib>

using namespace metal;

constant uint  GDN_SCAN_NUM_K_HEADS [[function_constant(0)]];
constant uint  GDN_SCAN_NUM_V_HEADS [[function_constant(1)]];
constant uint  GDN_SCAN_HEAD_K      [[function_constant(2)]];
constant uint  GDN_SCAN_HEAD_V      [[function_constant(3)]];
constant float GDN_SCAN_SCALE       [[function_constant(4)]];

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
    const device uint*  is_fresh      [[buffer(7)]],
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
  int slot = state_indices[i_n];
  if (slot < 0) {
    return;
  }
  bool fresh = is_fresh[i_n] != 0u;

  device float* state_row =
      ssm_state + ((uint(slot) * HV + i_hv) * Vd + i_v) * K;
  float b_h[GDN_SCAN_KMAX];
  for (uint ki = 0; ki < K; ki++) {
    b_h[ki] = fresh ? 0.0f : state_row[ki];
  }

  for (int i_t = 0; i_t < seq_len; i_t++) {
    uint t = uint(bos + i_t);
    // q from key-head i_h; k from key-head i_h (offset key_dim); v from value-head i_hv.
    const device T* q_ptr = conv_out + t * conv_dim + i_h * K;
    const device T* k_ptr = conv_out + t * conv_dim + key_dim + i_h * K;

    float q_sq = 0.0f, k_sq = 0.0f;
    for (uint ki = 0; ki < K; ki++) {
      float qf = float(q_ptr[ki]);
      float kf = float(k_ptr[ki]);
      q_sq += qf * qf;
      k_sq += kf * kf;
    }
    float q_inv = rsqrt(q_sq + 1e-6f);
    float k_inv = rsqrt(k_sq + 1e-6f);

    float decay = exp(g[t * HV + i_hv]);
    for (uint ki = 0; ki < K; ki++) {
      b_h[ki] *= decay;
    }

    float b_v = float(conv_out[t * conv_dim + 2u * key_dim + i_hv * Vd + i_v]);
    float dot_hk = 0.0f;
    for (uint ki = 0; ki < K; ki++) {
      dot_hk += b_h[ki] * (float(k_ptr[ki]) * k_inv);
    }
    b_v = (b_v - dot_hk) * beta[t * HV + i_hv];

    float b_o = 0.0f;
    for (uint ki = 0; ki < K; ki++) {
      float kn = float(k_ptr[ki]) * k_inv;
      float qn = float(q_ptr[ki]) * q_inv * scale;
      b_h[ki] += b_v * kn;
      b_o += b_h[ki] * qn;
    }
    o[t * value_dim + i_hv * Vd + i_v] = b_o;
  }

  for (uint ki = 0; ki < K; ki++) {
    state_row[ki] = b_h[ki];
  }
}

#define INST_GDN_SCAN_VARLEN(dtype_tag, mtl_type)                            \
  template [[host_name("gdn_scan_varlen_" #dtype_tag)]] [[kernel]] void      \
  gdn_scan_varlen<mtl_type>(                                                 \
      device       float*    o             [[buffer(0)]],                    \
      const device mtl_type* conv_out      [[buffer(1)]],                    \
      const device float*    g             [[buffer(2)]],                    \
      const device float*    beta          [[buffer(3)]],                    \
      device       float*    ssm_state     [[buffer(4)]],                    \
      const device int*      cu_seqlens    [[buffer(5)]],                    \
      const device int*      state_indices [[buffer(6)]],                    \
      const device uint*     is_fresh      [[buffer(7)]],                    \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint3 tpig [[thread_position_in_grid]]);

INST_GDN_SCAN_VARLEN(f16,  half)
INST_GDN_SCAN_VARLEN(bf16, bfloat)
INST_GDN_SCAN_VARLEN(f32,  float)
