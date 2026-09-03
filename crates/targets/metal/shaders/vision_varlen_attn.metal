// SPDX-License-Identifier: Apache-2.0
//
// Vision bidirectional varlen attention (Qwen3.5-VL / Qwen3-VL ViT).
//
// Cacheless, NON-causal, per-segment SDPA over `cu_seqlens` — the vision
// analogue of the text decoder's attention, but with NO paged KV-cache, NO
// causal mask, NO GQA, and an arbitrary head_dim (function-constant). This is
// net-new: every wired metal attention kernel is paged-cache + causal, and
// head_dim 72 (Qwen3.5-VL) / 80 (Qwen2.5-VL) are absent from the steel head-dim
// instantiations and fail `attention_via_cache_v2`'s head_dim%32==0.
//
// Faithful to mlx-vlm `qwen3_vl/vision.py::Attention` →
// `base.ensure_fused_sdpa`: per segment s and head h, for each query token i in
// [bos,eos):
//     out[i] = sum_{j in [bos,eos)} softmax_j( SCALE * dot(q[i], k[j]) ) * v[j]
// SCALE = head_dim**-0.5. (mlx pads head_dim 72->80 only to hit its fused
// kernel, then slices back to 72 — mathematically a no-op; we compute 72
// directly.) Bidirectional: a query attends every key in its OWN segment only.
//
// Layout: q/k/v/out are model dtype, row-major [L, H, D] (token-major):
//   element (token t, head h, dim d) at ((t*H + h)*D + d).
// `cu_seqlens` is int [NUM_SEGS+1], segment s spans tokens [cu[s], cu[s+1]).
//
// Function constants:
//   VA_HEAD_DIM (D), VA_NUM_HEADS (H), VA_NUM_SEGS, VA_N_TOKENS (L, guard),
//   VA_SCALE (head_dim**-0.5).
//
// Dispatch: 1 thread per (query token, head); flat grid of L*H threads.
// MVP (not FA2): online-softmax single pass, head_dim in registers.

#include <metal_stdlib>

using namespace metal;

constant uint  VA_HEAD_DIM  [[function_constant(0)]];
constant uint  VA_NUM_HEADS [[function_constant(1)]];
constant uint  VA_NUM_SEGS  [[function_constant(2)]];
constant uint  VA_N_TOKENS  [[function_constant(3)]];
constant float VA_SCALE     [[function_constant(4)]];

// Register accumulator bound (head_dim <= 128, matches the gdn scan).
constant constexpr uint VA_DMAX = 128;

template <typename T>
[[kernel]] void vision_varlen_attn(
    device       T*   out        [[buffer(0)]],
    const device T*   q          [[buffer(1)]],
    const device T*   k          [[buffer(2)]],
    const device T*   v          [[buffer(3)]],
    const device int* cu_seqlens [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  uint D = VA_HEAD_DIM;
  uint H = VA_NUM_HEADS;
  uint i = gid / H;        // query token
  uint h = gid % H;        // head
  if (i >= VA_N_TOKENS) {
    return;
  }

  // Locate the segment containing query token i (NUM_SEGS is small).
  int bos = -1, eos = -1;
  for (uint s = 0; s < VA_NUM_SEGS; s++) {
    int b = cu_seqlens[s];
    int e = cu_seqlens[s + 1];
    if (int(i) >= b && int(i) < e) {
      bos = b;
      eos = e;
      break;
    }
  }
  if (bos < 0 || eos <= bos) {
    return;
  }

  const device T* q_ptr = q + (uint(i) * H + h) * D;

  // Online (streaming) softmax over the segment's keys.
  float acc[VA_DMAX];
  for (uint d = 0; d < D; d++) {
    acc[d] = 0.0f;
  }
  float m = -INFINITY;
  float l = 0.0f;

  for (int j = bos; j < eos; j++) {
    const device T* k_ptr = k + (uint(j) * H + h) * D;
    float score = 0.0f;
    for (uint d = 0; d < D; d++) {
      score += float(q_ptr[d]) * float(k_ptr[d]);
    }
    score *= VA_SCALE;

    float m_new = max(m, score);
    float corr = exp(m - m_new);    // 0 on the first key (m=-inf → exp(-inf)=0)
    float p = exp(score - m_new);
    l = l * corr + p;

    const device T* v_ptr = v + (uint(j) * H + h) * D;
    for (uint d = 0; d < D; d++) {
      acc[d] = acc[d] * corr + p * float(v_ptr[d]);
    }
    m = m_new;
  }

  device T* out_ptr = out + (uint(i) * H + h) * D;
  float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
  for (uint d = 0; d < D; d++) {
    out_ptr[d] = T(acc[d] * inv);
  }
}

#define INST_VISION_VARLEN_ATTN(dtype_tag, mtl_type)                          \
  template [[host_name("vision_varlen_attn_" #dtype_tag)]] [[kernel]] void    \
  vision_varlen_attn<mtl_type>(                                               \
      device       mtl_type* out        [[buffer(0)]],                       \
      const device mtl_type* q          [[buffer(1)]],                       \
      const device mtl_type* k          [[buffer(2)]],                       \
      const device mtl_type* v          [[buffer(3)]],                       \
      const device int*      cu_seqlens [[buffer(4)]],                       \
      uint gid [[thread_position_in_grid]]);

INST_VISION_VARLEN_ATTN(f16,  half)
INST_VISION_VARLEN_ATTN(bf16, bfloat)
INST_VISION_VARLEN_ATTN(f32,  float)
