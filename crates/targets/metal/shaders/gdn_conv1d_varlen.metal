// SPDX-License-Identifier: Apache-2.0
//
// Gated-DeltaNet causal depthwise conv1d, varlen + stateful (Qwen3.5).
//
// ONE unified kernel subsuming both CUDA kernels (gdn_conv1d_kernels.cu):
// `causal_conv1d_update_kernel` (decode, seqlen==1) and
// `causal_conv1d_prefill_kernel` (prefill, seqlen large). One thread owns one
// (sequence, channel) pair and walks that sequence's tokens *sequentially*,
// keeping the causal window in registers — so there is NO cross-thread race on
// `conv_state` (the CUDA prefill kernel races: it writes state while sibling
// token-threads read it; here each thread reads old state once at the start and
// writes new state once at the end).
//
// The ring update is the CORRECT general shift (matches the decode kernel for
// seqlen==1 and the prefill kernel for seqlen>=state_len): the new state is the
// last `state_len` inputs of `[old_state ++ x_seq]`. This makes the multi-step
// continuity property hold — forward(T) then forward(1) == forward(T+1) — which
// the CUDA prefill kernel's "keep old state[ki]" branch silently breaks for
// chunks shorter than the kernel.
//
//   out[t,c] = SiLU( Σ_{j} w[c,j] · x_pad[t-(K-1)+j, c] )
//
// left-padded from the per-sequence `conv_state` ring (or zero when is_fresh).
//
// State layout (cuda-symmetric): conv_state[num_slots, conv_dim, state_len],
//   state_len = kernel-1, base = (slot*conv_dim + d)*state_len, oldest-first.
//
// `x`/`w` are model dtype (`T`), f32-accumulated; `conv_out` + `conv_state`
// are f32. Per-channel (depthwise) so GVA / head grouping does not apply here.
//
// Function constants:
//   GDN_CONV_DIM    — conv_dim (= 2*key_dim + value_dim)
//   GDN_CONV_KERNEL — kernel_size (<= GDN_CONV_KMAX)
//
// Dispatch: grid (num_seqs, ceil(conv_dim/tg), 1); one thread per (seq, channel).

#include <metal_stdlib>
#include "megakernel/mk_common.h"

using namespace metal;

#define GDN_CONV_CONSTS(X) X(uint, conv_dim, GDN_CONV_DIM, 0) X(uint, kernel_size, GDN_CONV_KERNEL, 1)
#ifndef MK_BODIES_ONLY
GDN_CONV_CONSTS(MK_FC_DECLARE)
struct GdnConvFc {
  GDN_CONV_CONSTS(MK_FC_ACCESSOR)
};
#endif

// Upper bound for the register window/weights (kernel-1 and kernel). GDN conv
// kernels are tiny (Qwen3.5 uses 4); 8 is a safe compile-time ceiling.
constant constexpr uint GDN_CONV_KMAX = 8;

// Body shared by the dispatch kernels and the megakernel adapter: the thread at `tgid` / `tpig`.
template <typename T, typename C, typename OP, typename XP>
METAL_FUNC void gdn_conv1d_varlen_body(
    OP conv_out, XP x, const device T* w, device float* conv_state,
    const device int* cu_seqlens, const device int* state_indices, const device uint* is_fresh,
    uint3 tgid, uint3 tpig)
{
  uint conv_dim = C::conv_dim();
  uint kernel_size = C::kernel_size();
  uint state_len = kernel_size - 1;

  uint seq = tgid.x;
  uint d = tpig.y;  // grid.y spans channel blocks; thread = channel
  if (d >= conv_dim) {
    return;
  }

  int base = cu_seqlens[seq];
  int seqlen = cu_seqlens[seq + 1] - base;
  if (seqlen <= 0) {
    return;
  }
  int slot = state_indices[seq];
  bool fresh = is_fresh[seq] != 0u;

  // Load conv weights for this channel.
  float wlocal[GDN_CONV_KMAX];
  for (uint ki = 0; ki < kernel_size; ki++) {
    wlocal[ki] = float(w[d * kernel_size + ki]);
  }

  // Seed the causal window from per-sequence state (zero when fresh).
  device float* state_ptr = conv_state + (uint(slot) * conv_dim + d) * state_len;
  float window[GDN_CONV_KMAX];
  for (uint ki = 0; ki < state_len; ki++) {
    window[ki] = fresh ? 0.0f : state_ptr[ki];
  }

  // Walk the sequence's tokens; the window slides through registers.
  for (int t = 0; t < seqlen; t++) {
    float x_t = float(x[(uint(base + t)) * conv_dim + d]);
    float sum = x_t * wlocal[state_len];
    for (uint ki = 0; ki < state_len; ki++) {
      sum += window[ki] * wlocal[ki];
    }
    conv_out[(uint(base + t)) * conv_dim + d] = sum / (1.0f + exp(-sum));  // SiLU
    // Shift window left, insert the new input.
    for (uint ki = 0; ki + 1 < state_len; ki++) {
      window[ki] = window[ki + 1];
    }
    if (state_len > 0) {
      window[state_len - 1] = x_t;
    }
  }

  // Persist the last `state_len` inputs as the new ring (correct general shift).
  for (uint ki = 0; ki < state_len; ki++) {
    state_ptr[ki] = window[ki];
  }
}

// Megakernel adapter: conv_out (0) and x (1) device-coherent; the layer's conv state is touched by
// this step alone, one thread per (sequence, channel). Compiled alone: inlined into the megakernel,
// the four-tap sum's products fuse into its adds differently than in the dispatch kernel.
template <typename T, typename C>
MK_FUNC_ALONE void mk_gdn_conv1d_varlen(thread const MkStep& s, MkLane l, threadgroup uchar*) {
  if (!l.live) return;
  gdn_conv1d_varlen_body<T, C>((mk_ptr<float>)s.addr[0], (mk_cptr<T>)s.addr[1],
                               (const device T*)s.addr[2], (device float*)s.addr[3],
                               (const device int*)s.addr[4], (const device int*)s.addr[5],
                               (const device uint*)s.addr[6], l.tg_pos, mk_thread_in_grid(l));
}

#ifndef MK_BODIES_ONLY
template <typename T>
[[kernel]] void gdn_conv1d_varlen(
    device       float* conv_out      [[buffer(0)]],
    const device T*     x             [[buffer(1)]],
    const device T*     w             [[buffer(2)]],
    device       float* conv_state    [[buffer(3)]],
    const device int*   cu_seqlens    [[buffer(4)]],
    const device int*   state_indices [[buffer(5)]],
    const device uint*  is_fresh      [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpig [[thread_position_in_grid]])
{
  gdn_conv1d_varlen_body<T, GdnConvFc>(conv_out, x, w, conv_state, cu_seqlens, state_indices,
                                       is_fresh, tgid, tpig);
}

#define INST_GDN_CONV1D_VARLEN(dtype_tag, mtl_type)                          \
  template [[host_name("gdn_conv1d_varlen_" #dtype_tag)]] [[kernel]] void    \
  gdn_conv1d_varlen<mtl_type>(                                               \
      device       float*    conv_out      [[buffer(0)]],                    \
      const device mtl_type* x             [[buffer(1)]],                    \
      const device mtl_type* w             [[buffer(2)]],                    \
      device       float*    conv_state    [[buffer(3)]],                    \
      const device int*      cu_seqlens    [[buffer(4)]],                    \
      const device int*      state_indices [[buffer(5)]],                    \
      const device uint*     is_fresh      [[buffer(6)]],                    \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint3 tpig [[thread_position_in_grid]]);
#else
#define INST_GDN_CONV1D_VARLEN(dtype_tag, mtl_type)                                       \
  MK_ADAPTER(gdn_conv1d_varlen_##dtype_tag, 0, 0x3, (mk_gdn_conv1d_varlen<mtl_type, MK_C>), \
             GDN_CONV_CONSTS)
#endif

INST_GDN_CONV1D_VARLEN(f16,  half)
INST_GDN_CONV1D_VARLEN(bf16, bfloat)
INST_GDN_CONV1D_VARLEN(f32,  float)
