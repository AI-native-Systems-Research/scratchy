// SPDX-License-Identifier: Apache-2.0
//
// Gated-DeltaNet causal depthwise conv1d, varlen + stateful (Qwen3.5).
//
// ONE unified kernel subsuming both CUDA kernels (gdn_conv1d_kernels.cu):
// `causal_conv1d_update_kernel` (decode, seqlen==1) and
// `causal_conv1d_prefill_kernel` (prefill, seqlen large). One thread owns one
// (sequence, channel, GDN_CONV_BLOCK-token block) and walks its block's tokens
// sequentially, keeping the causal window in registers — so there is NO
// cross-thread race on `conv_state` (the CUDA prefill kernel races: it writes
// state while sibling token-threads read it; here only the block running a
// sequence's last row writes state, once, and only the first block reads the
// seed). Blocks past a sequence's end early out: the grid's Z is the whole
// step's rows over GDN_CONV_BLOCK, always at least any one sequence's blocks.
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
// left-padded from the per-sequence `conv_state` ring (or zero when fresh). A
// block past the sequence's head seeds its window from `x` directly — the seed
// and the rows before a block are one stream, `[seed ++ x]`, its window at row
// t0 the last `state_len` entries before t0.
//
// State layout (cuda-symmetric): conv_state[slots, conv_dim, state_len], state_len =
//   kernel-1, base = (slot*conv_dim + d)*state_len, oldest-first. `state_indices[seq]` is the
//   sequence's `GdnSeqSlots` (`gdn_state.h`), `gdn_step[seq]`
//   `scratchy_layers::gdn_state::GdnStep::encode`. A verify step checkpoints the window after each
//   of its rows (the model's drafts + 1) in its sequence's draft lane (the code's bits 17-31),
//   `conv_lanes[lanes * (GDN_CONV_DRAFTS + 1), conv_dim, state_len]`; the step after one starts
//   from the lane's checkpoint at its last kept row. A slot start reads its window from the
//   record's source slot when it has one (a prefix-cache restore); a save stores the window after
//   its row into its snapshot slot.
//
// `x`/`w` are model dtype (`T`), f32-accumulated; `conv_out` + `conv_state`
// are f32. Per-channel (depthwise) so GVA / head grouping does not apply here.
//
// Mamba-2 (Nemotron-H) runs the same conv over the `xBC` columns of its whole in-projection row
// `z | xBC | dt`, plus a per-channel bias (`use_conv_bias`): the optional constants read `x` as a
// column slice of a wider row and add the bias ahead of the taps.
//
// Baked constants:
//   GDN_CONV_DIM       — conv_dim (GDN: 2*key_dim + value_dim; Mamba-2: d_inner + 2*n_groups*state)
//   GDN_CONV_KERNEL    — kernel_size (<= GDN_CONV_KMAX)
//   GDN_CONV_IN_STRIDE — optional: `x`'s row stride (unset: conv_dim, the qkv row)
//   GDN_CONV_IN_OFFSET — optional: the first conv channel's column in that row (unset: 0)
//   GDN_CONV_HAS_BIAS  — optional: add `bias[c]` (buffer 7; unset: no bias, buffer 7 unread)
//   GDN_CONV_DRAFTS    — the drafts each sequence of a verify step carries (`SPEC_DRAFTS`); 0:
//                        `conv_lanes` (buffer 8) unread
//   GDN_CONV_BLOCK     — the tokens a thread owns (the grid's Z is the step's rows over it)
//
// Dispatch: grid (num_seqs[set via seq_axis=X], ceil(conv_dim/tg),
// ceil(rows/GDN_CONV_BLOCK) [scaled by num_tokens]); one thread per (seq, channel, block).

#include <metal_stdlib>
#include "baked.h"
#include "gdn_state.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, GDN_CONV_DIM, 0);
SCRATCHY_CONSTANT(uint, GDN_CONV_KERNEL, 1);
SCRATCHY_CONSTANT_OPTIONAL(uint, GDN_CONV_IN_STRIDE, 2);
SCRATCHY_CONSTANT_OPTIONAL(uint, GDN_CONV_IN_OFFSET, 3);
SCRATCHY_CONSTANT_OPTIONAL(bool, GDN_CONV_HAS_BIAS, 4);
SCRATCHY_CONSTANT(uint, GDN_CONV_DRAFTS, 5);
SCRATCHY_CONSTANT(uint, GDN_CONV_BLOCK, 6);

constant constexpr uint GDN_CONV_X_STRIDE = GDN_CONV_IN_STRIDE_SET ? GDN_CONV_IN_STRIDE : GDN_CONV_DIM;

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
    const device GdnSeqSlots* state_indices [[buffer(5)]],
    const device uint*  gdn_step      [[buffer(6)]],
    const device T*     bias          [[buffer(7)]],
    device       float* conv_lanes    [[buffer(8)]],
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
  const GdnSeqSlots seq_slots = state_indices[seq];
  uint code = gdn_step[seq];
  uint start = code & 0xffu;
  // A model without drafts (`GDN_CONV_DRAFTS` 0) neither replays nor checkpoints.
  uint drafts = GDN_CONV_DRAFTS == 0u ? 0u : (code >> 8) & 0xffu;
  bool replays = GDN_CONV_DRAFTS > 0u && start >= 2u;
  // A verify step checkpoints the window after each of its rows.
  uint checkpoint_rows = drafts == 0u ? 0u : drafts + 1u;
  uint entry_len = conv_dim * state_len;
  // This channel's window in slot `s`.
  auto ring = [&](int s) { return conv_state + (uint(s) * conv_dim + d) * state_len; };
  const int2 saves_through = gdn_saves_through(seq_slots);

  // Load conv weights for this channel.
  float wlocal[GDN_CONV_KMAX];
  for (uint ki = 0; ki < kernel_size; ki++) {
    wlocal[ki] = float(w[d * kernel_size + ki]);
  }

  // The seed the sequence's first block walks in from: its slot, a restore's snapshot, or the
  // lane's checkpoint at the last kept row (zero when fresh).
  device float* state_ptr = ring(seq_slots.slot);
  device float* lane_ptr =
      conv_lanes + ((code >> 17) * (GDN_CONV_DRAFTS + 1u) * conv_dim + d) * state_len;
  const device float* start_ptr =
      replays ? lane_ptr + (start - 2u) * entry_len : ring(gdn_start_slot(seq_slots, start));

  // This thread's block; blocks past the sequence's end have nothing to do.
  int t0 = int(tgid.z) * int(GDN_CONV_BLOCK);
  if (t0 >= seqlen) {
    return;
  }
  const int t_end = metal::min(t0 + int(GDN_CONV_BLOCK), seqlen);

  // Seed the causal window: the last `state_len` entries of `[seed ++ x]` before row t0 — the
  // seed's own entries while t0 still sits inside it, `x` rows past it.
  float window[GDN_CONV_KMAX];
  for (uint ki = 0; ki < state_len; ki++) {
    uint p = uint(t0) + ki;
    window[ki] = p < state_len
                     ? (start == 1u ? 0.0f : start_ptr[p])
                     : float(x[(uint(base) + p - state_len) * GDN_CONV_X_STRIDE
                               + GDN_CONV_IN_OFFSET + d]);
  }

  // Walk the block's tokens; the window slides through registers, a save storing its snapshot
  // after the row it runs through (`gdn_state.h`).
  const float b = GDN_CONV_HAS_BIAS ? float(bias[d]) : 0.0f;
  for (int t = t0; t < t_end; t++) {
    float x_t = float(x[(uint(base + t)) * GDN_CONV_X_STRIDE + GDN_CONV_IN_OFFSET + d]);
    float sum = b + x_t * wlocal[state_len];
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
      lane_ptr[uint(t) * entry_len + ki] = window[ki];
    }
    for (uint j = 0; j < GDN_SAVES; j++) {
      if (gdn_saves_after(saves_through, j, t)) {
        device float* snap = ring(seq_slots.saves[j].y);
        for (uint ki = 0; ki < state_len; ki++) {
          snap[ki] = window[ki];
        }
      }
    }
  }

  // The block that ran the sequence's last row persists the last `state_len` inputs as the new
  // ring (correct general shift).
  if (t_end == seqlen) {
    for (uint ki = 0; ki < state_len; ki++) {
      state_ptr[ki] = window[ki];
    }
  }
}

#define INST_GDN_CONV1D_VARLEN(dtype_tag, mtl_type) \
  SCRATCHY_KERNEL(gdn_conv1d_varlen_##dtype_tag, gdn_conv1d_varlen<mtl_type>)

INST_GDN_CONV1D_VARLEN(f16,  half)
INST_GDN_CONV1D_VARLEN(bf16, bfloat)
INST_GDN_CONV1D_VARLEN(f32,  float)
