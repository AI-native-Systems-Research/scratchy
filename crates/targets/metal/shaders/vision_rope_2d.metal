// SPDX-License-Identifier: Apache-2.0
//
// Vision 2D rotary position embedding (Qwen3.5-VL / Qwen3-VL ViT).
//
// Faithful port of mlx-vlm `qwen3_vl/vision.py::apply_rotary_pos_emb_vision`
// (GPT-NeoX `rotate_half`). Applied independently to q and k (MHA — every head
// rotates; NO GQA grouping, NO paged-KV write — unlike the text rope kernel).
//
// Inputs (one image's patch tokens flattened, contiguous [L, H, D]):
//   x     : model dtype, the q OR k tensor, row-major [L, H, D]
//   freqs : f32, the per-token rotary table [L, half] where half = D/2
//
// Math, for output element (t, h, d):
//   half = D/2
//   f    = freqs[t*half + (d % half)]              // cos/sin tiled by concat
//   rh   = (d < half) ? -x[t,h,d+half] : x[t,h,d-half]   // rotate_half
//   out  = x[t,h,d]*cos(f) + rh*sin(f)
//
// Verified == mlx-vlm golden (rope_block0_q): max_abs_err 9.5e-7, cosine 1.0.
//
// Function constants:
//   VR_HEAD_DIM  (D), VR_NUM_HEADS (H), VR_N_ELEMS (L*H*D, dispatch guard).
//
// Dispatch: 1 thread per output element, flat grid of ceil(N/tg) threadgroups.

#include <metal_stdlib>

using namespace metal;

constant uint VR_HEAD_DIM  [[function_constant(0)]];
constant uint VR_NUM_HEADS [[function_constant(1)]];
constant uint VR_N_ELEMS   [[function_constant(2)]];

template <typename T>
[[kernel]] void vision_rope_2d(
    device       T*     out   [[buffer(0)]],
    const device T*     x     [[buffer(1)]],
    const device float* freqs [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  uint D = VR_HEAD_DIM;
  uint H = VR_NUM_HEADS;
  if (gid >= VR_N_ELEMS) {
    return;
  }
  uint hd = D / 2u;        // half rotary dim (`half` shadows the MSL type)
  uint d = gid % D;        // dim within the head
  uint row = gid / D;      // = t*H + h
  uint t = row / H;        // token index (for the per-token freqs row)

  float f = freqs[t * hd + (d % hd)];
  float c = cos(f);
  float s = sin(f);

  uint base = gid - d;     // start of this (token, head)'s D-vector
  float xv = float(x[gid]);
  // rotate_half: first half pairs with -(second half), second half with +(first half)
  float partner = (d < hd) ? -float(x[base + d + hd]) : float(x[base + d - hd]);

  out[gid] = T(xv * c + partner * s);
}

#define INST_VISION_ROPE_2D(dtype_tag, mtl_type)                              \
  template [[host_name("vision_rope_2d_" #dtype_tag)]] [[kernel]] void        \
  vision_rope_2d<mtl_type>(                                                   \
      device       mtl_type* out   [[buffer(0)]],                            \
      const device mtl_type* x     [[buffer(1)]],                            \
      const device float*    freqs [[buffer(2)]],                            \
      uint gid [[thread_position_in_grid]]);

INST_VISION_ROPE_2D(f16,  half)
INST_VISION_ROPE_2D(bf16, bfloat)
INST_VISION_ROPE_2D(f32,  float)

// --------------------------------------------------------------------------
// Interleaved (GPT-J / "traditional") variant — MoonViT (LocateAnything).
//
// Faithful port of mlx-vlm `locateanything/vision.py::apply_rope`
// (`view_as_complex` on adjacent pairs, complex multiply by freqs_cis).
//
// Math, for the complex pair i = d/2 of output element (t, h, d):
//   f         = freqs[t*half + i]                  // one angle per PAIR
//   out[2i]   = x[2i]*cos(f) - x[2i+1]*sin(f)
//   out[2i+1] = x[2i]*sin(f) + x[2i+1]*cos(f)
//
// The host freqs table carries the LocateAnything interleaved x/y layout
// (angle[2j] = x·f_j, angle[2j+1] = y·f_j, f_j = theta^(-4j/D)); the kernel
// is layout-agnostic — it just rotates pair i by freqs[t*half + i].
//
// Verified == mlx-vlm golden (rope_block0_q): max_abs_err 4.8e-7, cosine 1.0
// (numpy transcription in tools/vision_parity, LocateAnything fixtures).

template <typename T>
[[kernel]] void vision_rope_2d_interleaved(
    device       T*     out   [[buffer(0)]],
    const device T*     x     [[buffer(1)]],
    const device float* freqs [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  uint D = VR_HEAD_DIM;
  uint H = VR_NUM_HEADS;
  if (gid >= VR_N_ELEMS) {
    return;
  }
  uint hd = D / 2u;
  uint d = gid % D;        // dim within the head
  uint row = gid / D;      // = t*H + h
  uint t = row / H;        // token index (for the per-token freqs row)

  float f = freqs[t * hd + d / 2u];
  float c = cos(f);
  float s = sin(f);

  float xv = float(x[gid]);
  // adjacent-pair rotation: even lane pairs with -(odd), odd with +(even);
  // both lanes then share the same xv*c + partner*s form.
  bool even = (d & 1u) == 0u;
  float partner = even ? -float(x[gid + 1u]) : float(x[gid - 1u]);

  out[gid] = T(xv * c + partner * s);
}

#define INST_VISION_ROPE_2D_INTERLEAVED(dtype_tag, mtl_type)                  \
  template [[host_name("vision_rope_2d_interleaved_" #dtype_tag)]]            \
  [[kernel]] void vision_rope_2d_interleaved<mtl_type>(                       \
      device       mtl_type* out   [[buffer(0)]],                            \
      const device mtl_type* x     [[buffer(1)]],                            \
      const device float*    freqs [[buffer(2)]],                            \
      uint gid [[thread_position_in_grid]]);

INST_VISION_ROPE_2D_INTERLEAVED(f16,  half)
INST_VISION_ROPE_2D_INTERLEAVED(bf16, bfloat)
INST_VISION_ROPE_2D_INTERLEAVED(f32,  float)
