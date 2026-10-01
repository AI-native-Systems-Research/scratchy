// SPDX-License-Identifier: Apache-2.0
// THE DECODE MEGAKERNEL'S BODIES: the translation unit the macro-generated kernels complete.
//
// `megakernel_bake.rs` compiles the bucket-1 decode forward into SEGMENT kernels
// (`[[kernel]] void mk_seg_<n>(...)`), each a run of the tape's steps in which no threadgroup
// waits on another, as straight-line calls to the adapters below — every constant a literal, work
// split statically over the launch's `MK_P` threadgroups — one launch each, a launch boundary
// wherever the dataflow crosses threadgroups. build.rs inlines this file's local includes into one
// self-contained text (`mk_bodies.metal`); the bake appends the tape's generated kernels and
// compiles the library at build time.
//
// build.rs also preprocesses this file with `-DMK_ENUMERATE`: each `MK_ADAPTER` prints a marker,
// and the markers become `MK_ADAPTERS`, the adapters the bake may call.
#define MK_BODIES_ONLY 1
#include "megakernel/mk_common.h"

#define MK_LIB embed
#include "embed.metal"
#undef MK_LIB
#define MK_LIB rmsnorm
#include "rmsnorm.metal"
#undef MK_LIB
#define MK_LIB vision_layernorm
#include "vision_layernorm.metal"
#undef MK_LIB
#define MK_LIB fused_add_rmsnorm
#include "fused_add_rmsnorm.metal"
#undef MK_LIB
#define MK_LIB quantized_qmv
#include "quantized_qmv.metal"
#undef MK_LIB
#define MK_LIB silu_mul
#include "silu_mul.metal"
#undef MK_LIB
#define MK_LIB fused_gate_up_silu_mul
#include "fused_gate_up_silu_mul.metal"
#undef MK_LIB
#define MK_LIB gate_split
#include "gate_split.metal"
#undef MK_LIB
#define MK_LIB gate_apply
#include "gate_apply.metal"
#undef MK_LIB
#define MK_LIB gate_scale
#include "gate_scale.metal"
#undef MK_LIB
#define MK_LIB gdn_conv1d_varlen
#include "gdn_conv1d_varlen.metal"
#undef MK_LIB
#define MK_LIB gdn_gating
#include "gdn_gating.metal"
#undef MK_LIB
#define MK_LIB gdn_scan_varlen
#include "gdn_scan_varlen.metal"
#undef MK_LIB
#define MK_LIB gdn_rms_norm_gated
#include "gdn_rms_norm_gated.metal"
#undef MK_LIB
#define MK_LIB elementwise
#include "elementwise.metal"
#undef MK_LIB
#define MK_LIB quantized_dequantize
#include "quantized_dequantize.metal"
#undef MK_LIB
#define MK_LIB rope
#include "rope.metal"
#undef MK_LIB
#define MK_LIB turboquant
#include "turboquant.metal"
#undef MK_LIB
#define MK_LIB gemm
#include "gemm.metal"
#undef MK_LIB
#define MK_LIB argpartition
#include "argpartition.metal"
#undef MK_LIB
#define MK_LIB slice_trailing_cols
#include "slice_trailing_cols.metal"
#undef MK_LIB
#define MK_LIB take_along_axis
#include "take_along_axis.metal"
#undef MK_LIB
#define MK_LIB softmax
#include "softmax.metal"
#undef MK_LIB
#define MK_LIB moe_per_expert_scale
#include "moe_per_expert_scale.metal"
#undef MK_LIB
#define MK_LIB moe_weighted_sum
#include "moe_weighted_sum.metal"
#undef MK_LIB
#define MK_LIB attention
#include "attention.metal"
#undef MK_LIB

#ifndef MK_ENUMERATE
// The threadgroups every launch runs — the GPU's cores, known at load. The generated work split
// reads it as `mk_p`: item `i` of a spread step at segment cursor `c` runs on threadgroup
// `(c + i) mod mk_p`, a pinned unit on the threadgroup its segment's split names.
constant uint MK_P [[function_constant(4096)]];

// The first work item of a spread step on threadgroup `idx` of `p`, the segment cursor at `c`.
METAL_FUNC uint mk_first(uint idx, uint p, uint c) { return (idx + p - c % p) % p; }

// The work items of a grid the load sizes (`vpi` virtual threadgroups per item).
METAL_FUNC uint mk_items(uint3 grid, uint vpi) { return (grid.x * grid.y * grid.z + vpi - 1u) / vpi; }
#endif // MK_ENUMERATE
