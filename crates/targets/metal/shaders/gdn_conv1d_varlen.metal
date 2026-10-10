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
// left-padded from the per-sequence `conv_state` ring (or zero when fresh).
//
// State layout (cuda-symmetric): conv_state[entries, conv_dim, state_len], state_len =
//   kernel-1, base = (entry*conv_dim + d)*state_len, oldest-first. `state_indices[seq]` is the
//   sequence's slot, `gdn_step[seq]` `scratchy_layers::gdn_state::GdnStep::encode`: a slot's
//   entry is followed by its checkpoints, one after each of a verify step's rows (the model's
//   drafts + 1), so slot `s` is entry `s` without drafts and `s * (GDN_CONV_DRAFTS + 2)` with.
//
// `x`/`w` are model dtype (`T`), f32-accumulated; `conv_out` + `conv_state`
// are f32. Per-channel (depthwise) so GVA / head grouping does not apply here.
//
// Baked constants:
//   GDN_CONV_DIM    — conv_dim (= 2*key_dim + value_dim)
//   GDN_CONV_KERNEL — kernel_size (<= GDN_CONV_KMAX)
//   GDN_CONV_DRAFTS — the drafts each sequence of a verify step carries (`SPEC_DRAFTS`)
//
// Dispatch: grid (num_seqs, ceil(conv_dim/tg), 1); one thread per (seq, channel).

#include <metal_stdlib>
#include "baked.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, GDN_CONV_DIM, 0);
SCRATCHY_CONSTANT(uint, GDN_CONV_KERNEL, 1);
SCRATCHY_CONSTANT(uint, GDN_CONV_DRAFTS, 2);

// Upper bound for the register window/weights (kernel-1 and kernel). GDN conv
// kernels are tiny (Qwen3.5 uses 4); 8 is a safe compile-time ceiling.
constant constexpr uint GDN_CONV_KMAX = 8;

template <typename T>
[[kernel]] void gdn_conv1d_varlen(
    device       float* conv_out      [[buffer(0)]],
    const device T*     x             [[buffer(1)]],
    const device T*     w             [[buffer(2)]],
    device       float* conv_state    [[buffer(3)]],
    const device int*   cu_seqlens    [[buffer(4)]],
    const device int*   state_indices [[buffer(5)]],
    const device uint*  gdn_step      [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpig [[thread_position_in_grid]])
{
  uint conv_dim = GDN_CONV_DIM;
  uint kernel_size = GDN_CONV_KERNEL;
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
  uint code = gdn_step[seq];
  uint start = code & 0xffu;
  // A model without drafts (`GDN_CONV_DRAFTS` 0) neither replays nor checkpoints.
  uint drafts = GDN_CONV_DRAFTS == 0u ? 0u : (code >> 8) & 0xffu;
  uint entry = uint(slot) * (GDN_CONV_DRAFTS == 0u ? 1u : GDN_CONV_DRAFTS + 2u);
  // A verify step checkpoints the window after each of its rows.
  uint checkpoint_rows = drafts == 0u ? 0u : drafts + 1u;
  uint entry_len = conv_dim * state_len;

  // Load conv weights for this channel.
  float wlocal[GDN_CONV_KMAX];
  for (uint ki = 0; ki < kernel_size; ki++) {
    wlocal[ki] = float(w[d * kernel_size + ki]);
  }

  // Seed the causal window from the start entry (zero when fresh).
  device float* state_ptr = conv_state + (entry * conv_dim + d) * state_len;
  const device float* start_ptr =
      state_ptr + (GDN_CONV_DRAFTS > 0u && start >= 2u ? (start - 1u) * entry_len : 0u);
  float window[GDN_CONV_KMAX];
  for (uint ki = 0; ki < state_len; ki++) {
    window[ki] = start == 1u ? 0.0f : start_ptr[ki];
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
    for (uint ki = 0; uint(t) < checkpoint_rows && ki < state_len; ki++) {
      state_ptr[(uint(t) + 1u) * entry_len + ki] = window[ki];
    }
  }

  // Persist the last `state_len` inputs as the new ring (correct general shift).
  for (uint ki = 0; ki < state_len; ki++) {
    state_ptr[ki] = window[ki];
  }
}

#define INST_GDN_CONV1D_VARLEN(dtype_tag, mtl_type) \
  SCRATCHY_KERNEL(gdn_conv1d_varlen_##dtype_tag, gdn_conv1d_varlen<mtl_type>)

INST_GDN_CONV1D_VARLEN(f16,  half)
INST_GDN_CONV1D_VARLEN(bf16, bfloat)
INST_GDN_CONV1D_VARLEN(f32,  float)
