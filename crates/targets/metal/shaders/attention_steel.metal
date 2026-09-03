// SPDX-License-Identifier: Apache-2.0
//
// Faithful port of MLX's `steel_attention` (tiled FlashAttention-2)
// for scratchy-target-metal's prefill path. Source: mlx/backend/metal/kernels/
// steel/attn/kernels/steel_attention.h + supporting headers (vendored
// verbatim under mlx_steel_attn/). Replaces the prior
// `attention_prefill_sdpa_v2_paged_*` which mistakenly ported MLX's
// `sdpa_vector` (decode-only kernel) for the prefill path. See
// `project_metal_attention_port_gap.md`.
//
// Tile sizes mirror MLX's M4 (non-NAX) dispatch
// (`scaled_dot_product_attention.cpp:166 sdpa_full_self_attention_metal`):
//   WM=4, WN=1; BQ=32; BK = BD < 128 ? 32 : 16; BD = head_dim.
//
// Llama-3.2-3B: head_dim=128 → BK=16 (32 Q tile × 16 K tile).
//
// PHASE 1 (this file): contiguous-K/V variant only — the same shape
// as MLX's reference kernel. Validates the port end-to-end against
// MLX-equivalent inputs.
//
// PHASE 2 (next commit): paged-K/V-cache adapter. Wraps `loader_k` /
// `loader_v` with block-table-indexed gather so prefill that reads
// already-cached prefix tokens (chunked prefill, prefix-cache hits,
// multi-turn continuation) works with the steel kernel. The contiguous
// variant stays for the no-prefix path.

#include "mlx_steel_attn/steel_attention_kernel.h"

// Llama-3.2-3B shape: BQ=32, BK=16, BD=128, WM=4, WN=1.
// MaskType=float (matches MLX's float mask path). do_causal is set
// via function constant 301 at pipeline-creation time.
template [[host_name("attention_steel_f16_bq32_bk16_bd128_wm4_wn1")]]
[[kernel]] decltype(attention<half, 32, 16, 128, 4, 1, float, float>)
    attention<half, 32, 16, 128, 4, 1, float, float>;

template [[host_name("attention_steel_bf16_bq32_bk16_bd128_wm4_wn1")]]
[[kernel]] decltype(attention<bfloat, 32, 16, 128, 4, 1, float, float>)
    attention<bfloat, 32, 16, 128, 4, 1, float, float>;
