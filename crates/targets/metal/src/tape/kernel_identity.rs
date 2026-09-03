// SPDX-License-Identifier: Apache-2.0
//! Type-level kernel identity.
//!
//! Today `KernelId` (an enum
//! tag) + `library: &'static str` + `function: &'static str` are three
//! independent fields on `LoweredCommand`. The lowering arm emits all
//! three by hand; getting the function symbol's `_bq32_` token wrong
//! while leaving the matching `KernelId` correct silently selects the
//! wrong specialization.
//!
//! The [`MetalKernel`] trait bundles the three identifiers into one
//! ZST per (kernel, dtype) combination. `LoweredCommand::for_kernel`
//! reads all three from the ZST so the lowering arm can only emit a
//! coherent trio.
//!
//! Phase 4 lands the attention family (the kernels where bug #8's
//! sibling — the slot-99 ATTN_PAGED_DEBUG_MODE omission — fired).
//! Dynamic-symbol kernels (AffineQmv/QmmT/Synth*) keep the
//! `&'static str` pattern for now — their symbol is composed at
//! lowering time and doesn't fit a `const &'static str` field. Phase
//! 6 (const generics) lifts that restriction.
//!
//! Kernels not yet typed:
//!   - AffineQmv{Quad,Fast,_} (symbol depends on dtype × bucket × group_size)
//!   - AffineQmmT / Nax / SplitK (same)
//!   - SplitKReduceSum (dtype-only — could be typed; left for follow-up)
//!   - AffineEmbed (same)
//!   - SiluMul (same)
//!   - Embed (same — static after dtype pick; left for follow-up)
//!   - RmsNorm / FusedAddRmsNorm (depends on scale dtype)
//!   - FusedGateUpSiluMul (decode vs prefill branch)
//!   - GatherLastToken / ScatterFirstToLastRow (same)
//!   - RopeAppend (dtype-only — left for follow-up)
//!   - FusedQkvRopeCache (dtype-only — left for follow-up)
//!   - Synth* (compiler-emitted symbol — never static)

use crate::tape::constants::ConstantValue;

use crate::tape::kernel_bindings::AttentionPrefillPagedBindingSet;
use crate::tape::kernel_constants::AttentionPrefillPagedConstants;
use crate::tape::lowered::{Binding, KernelId};

/// One-of identifier for a single kernel pipeline + its parameter
/// shape. Each `MetalKernel` impl wires the static
/// `(library, function, KernelId)` trio that the worker walks at
/// dispatch time, plus the per-call typed `Constants` and `BindingSet`
/// types it accepts.
pub trait MetalKernel {
    /// Per-call function-constants struct (Phase 2). The trait's
    /// `for_kernel` constructor calls `.into()` to lower to the
    /// existing `Vec<ConstantValue>` wire format.
    type Constants: Into<Vec<ConstantValue>>;

    /// Per-call binding-set struct (Phase 3).
    type BindingSet: Into<Vec<Binding>>;

    /// `.metallib` file (matches the keys
    /// `SpecializedPipelineCache::with_standard_shaders` registers).
    const LIBRARY: &'static str;

    /// MSL `kernel void` symbol the pipeline binds.
    const FUNCTION: &'static str;

    /// Coarse `KernelId` tag (carried alongside the LIBRARY/FUNCTION
    /// pair for dispatch-timing labels and debug formatting).
    const KERNEL_ID: KernelId;
}

// ── AttentionPrefillSdpaPaged variants ─────────────────────────────
//
// `attention_steel_paged` does NOT have a typed ZST per (dtype, BD)
// combo: BD (head_dim) is part of the symbol name, and emitting one
// `pub struct ...` + `MetalKernel` impl per combination would
// duplicate the head-dim list in N places. Instead, the lowering arm
// looks up the symbol at dispatch time via
// `crate::steel_paged::steel_paged_symbol(dtype, head_dim)`,
// whose match arms are codegen'd from
// `scratchy-target-metal/build.rs::STEEL_PAGED_HEAD_DIMS` — the same
// list `attention_steel_paged_instantiations.h` is emitted from. See
// `lowering.rs::AttentionPrefillPaged`.
//
// SDPA fallback below stays typed: its symbol is fully static (one
// per dtype, no per-combo template instantiation).

/// `attention_prefill_sdpa_v2_paged_bf16_specialized` (the
/// known-correct sdpa_vector port, kept as fallback behind
/// `SCRATCHY_METAL_STEEL_ATTN=0`).
pub struct AttentionSdpaPagedBf16;

impl MetalKernel for AttentionSdpaPagedBf16 {
    type Constants = AttentionPrefillPagedConstants;
    type BindingSet = AttentionPrefillPagedBindingSet;
    const LIBRARY: &'static str = "attention";
    const FUNCTION: &'static str = "attention_prefill_sdpa_v2_paged_bf16_specialized";
    const KERNEL_ID: KernelId = KernelId::AttentionPrefillSdpaPaged;
}

/// `attention_prefill_sdpa_v2_paged_f16_specialized`.
pub struct AttentionSdpaPagedF16;

impl MetalKernel for AttentionSdpaPagedF16 {
    type Constants = AttentionPrefillPagedConstants;
    type BindingSet = AttentionPrefillPagedBindingSet;
    const LIBRARY: &'static str = "attention";
    const FUNCTION: &'static str = "attention_prefill_sdpa_v2_paged_f16_specialized";
    const KERNEL_ID: KernelId = KernelId::AttentionPrefillSdpaPaged;
}
