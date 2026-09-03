// SPDX-License-Identifier: Apache-2.0
//
// Paged-K/V-cache variant of MLX's steel_attention prefill kernel.
// Replaces `attention_prefill_sdpa_v2_paged_*` (which was a port of
// the wrong MLX source — sdpa_vector is decode-only).
//
// Algorithm is byte-for-byte MLX's `steel_attention` (FA-2 tiled);
// only the K/V load path is rewritten to use `PagedKVBlockLoader`
// over the scratchy-target-metal paged cache. See
// `mlx_steel_attn/steel_attention_paged_kernel.h`.

#include "mlx_steel_attn/steel_attention_paged_kernel.h"

// Tile shape: BQ=32, BK=16, WM=4, WN=1. BLOCK_SIZE=16 matches the
// scratchy-target-metal paged-cache block size so one MLX kb iter = one paged
// block. BD (head_dim) is the only axis that varies across supported
// arches.
//
// Single source of truth for the (dtype, BD) instantiation list
// lives in `scratchy-target-metal/build.rs::STEEL_PAGED_HEAD_DIMS`.
// build.rs writes:
//   - `attention_steel_paged_instantiations.h` (this include) — the
//     `INST_STEEL_PAGED(tag, type, bd)` expansion lines.
//   - `steel_paged_kernels_generated.rs` — the Rust dispatcher's
//     symbol table, mirroring the same list.
//
// Add a HEAD_DIM in build.rs and both sides pick it up. Adding it
// here would link to `steel_paged_symbol()` returning `None` for the
// new dim and the dispatcher would silently fall through to SDPA;
// adding it in Rust without here would call `library.get_function()`
// for an absent symbol and panic at pipeline-build time.
#define INST_STEEL_PAGED(dt_tag, dt_type, bk, bd)                               \
  template [[host_name(                                                         \
      "attention_steel_paged_" #dt_tag "_bq32_bk" #bk "_bd" #bd "_wm4_wn1_bs16" \
  )]] [[kernel]]                                                                \
  decltype(attention_paged<dt_type, 32, bk, bd, 4, 1, 16, float>)               \
      attention_paged<dt_type, 32, bk, bd, 4, 1, 16, float>;                    \
  template [[host_name("rope_once_steel_" #dt_tag "_bd" #bd "_bs16")]]          \
  [[kernel]]                                                                    \
  decltype(rope_once_steel_kernel<dt_type, bd, 16>)                             \
      rope_once_steel_kernel<dt_type, bd, 16>;

#include "attention_steel_paged_instantiations.h"
