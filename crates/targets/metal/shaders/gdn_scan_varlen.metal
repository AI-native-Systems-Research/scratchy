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
#include "megakernel/mk_common.h"

using namespace metal;

#define GDN_SCAN_CONSTS(X)                                                                \
  X(uint, num_k_heads, GDN_SCAN_NUM_K_HEADS, 0) X(uint, num_v_heads, GDN_SCAN_NUM_V_HEADS, 1) \
  X(uint, head_k, GDN_SCAN_HEAD_K, 2) X(uint, head_v, GDN_SCAN_HEAD_V, 3)                  \
  X(float, scale, GDN_SCAN_SCALE, 4)
#ifndef MK_BODIES_ONLY
GDN_SCAN_CONSTS(MK_FC_DECLARE)
struct GdnScanFc {
  GDN_SCAN_CONSTS(MK_FC_ACCESSOR)
};
#endif

// Matches CUDA `MAX_HEAD_K_DIM` (gdn_recurrent_kernels.cu): register state row.
constant constexpr uint GDN_SCAN_KMAX = 128;

// Body shared by the dispatch kernels and the megakernel adapter: the thread at `tgid` / `tpig`.
template <typename T, typename C, typename OP, typename CP, typename FP>
METAL_FUNC void gdn_scan_varlen_body(
    OP o, CP conv_out, FP g, FP beta, device float* ssm_state, const device int* cu_seqlens,
    const device int* state_indices, const device uint* is_fresh, uint3 tgid, uint3 tpig)
{
  uint H = C::num_k_heads();
  uint HV = C::num_v_heads();
  uint K = C::head_k();
  uint Vd = C::head_v();
  float scale = C::scale();

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
    CP q_ptr = conv_out + t * conv_dim + i_h * K;
    CP k_ptr = conv_out + t * conv_dim + key_dim + i_h * K;

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

// Megakernel adapter: the layer's recurrent state is touched by this step alone, one thread per
// (value-dim, head, sequence).
template <typename T, typename C>
MK_FUNC void mk_gdn_scan_varlen(thread const MkStep& s, MkLane l, threadgroup uchar*) {
  if (!l.live) return;
  gdn_scan_varlen_body<T, C>((mk_ptr<float>)s.addr[0], (mk_cptr<T>)s.addr[1],
                             (mk_cptr<float>)s.addr[2], (mk_cptr<float>)s.addr[3],
                             (device float*)s.addr[4], (const device int*)s.addr[5],
                             (const device int*)s.addr[6], (const device uint*)s.addr[7],
                             l.tg_pos, mk_thread_in_grid(l));
}

#ifndef MK_BODIES_ONLY
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
  gdn_scan_varlen_body<T, GdnScanFc>(o, conv_out, g, beta, ssm_state, cu_seqlens, state_indices,
                                     is_fresh, tgid, tpig);
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
#else
#define INST_GDN_SCAN_VARLEN(dtype_tag, mtl_type)                                        \
  MK_ADAPTER(gdn_scan_varlen_##dtype_tag, 0, (mk_gdn_scan_varlen<mtl_type, MK_C>),       \
             GDN_SCAN_CONSTS)
#endif

INST_GDN_SCAN_VARLEN(f16,  half)
INST_GDN_SCAN_VARLEN(bf16, bfloat)
INST_GDN_SCAN_VARLEN(f32,  float)
