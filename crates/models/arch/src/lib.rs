// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused_imports)] // the per-model emit brings ops traits into scope conditionally
#![allow(non_camel_case_types)] // emitted variant names mirror model stems (Llama_3_2_1b)
#![allow(non_snake_case)] // emitted fused-weight field names (mlp_gate_proj__fused__mlp_up_proj)
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Model architectures — ONE crate (the per-arch crates collapsed).
//! `build.rs` drives the forward pipeline per arch (from `dsl/<arch>.py`
//! against `configs/<arch>`) into `$OUT_DIR/<mod>.rs`; each is wrapped in a
//! `pub mod <arch>` here. Runtime discovery is via the emitted
//! `inventory::submit!` (see `scratchy_forward_compiler`), so a consumer just
//! `extern crate scratchy_models as _;` to force-link the whole crate — no
//! per-arch force-links needed now that it's one crate.

// Real HF org/repo ids resolved once against huggingface.co at BUILD time
// (`hf_registry_build.rs`), scoped to exactly the architectures this binary
// was compiled to run. `scr model names` completes from this — no runtime
// network call, no local hf-hub cache read.
include!(concat!(env!("OUT_DIR"), "/hf_registry.rs"));

/// The compiled-in HF model registry — see [`COMPILED_HF_REGISTRY`].
pub fn compiled_hf_registry() -> &'static [&'static str] {
    COMPILED_HF_REGISTRY
}

#[cfg(all(feature = "arch-commandr", any(feature = "cuda", feature = "metal")))]
#[path = "arch/commandr.rs"]
pub mod commandr;

#[cfg(all(feature = "arch-deepseek-v2", feature = "cuda"))]
#[path = "arch/deepseek_v2.rs"]
pub mod deepseek_v2;

#[cfg(all(feature = "arch-deepseek-v3", feature = "cuda"))]
#[path = "arch/deepseek_v3.rs"]
pub mod deepseek_v3;

#[cfg(all(feature = "arch-deepseek-v3-flat", feature = "cuda"))]
#[path = "arch/deepseek_v3_flat.rs"]
pub mod deepseek_v3_flat;

#[cfg(all(feature = "arch-gemma2", any(feature = "cuda", feature = "metal")))]
#[path = "arch/gemma2.rs"]
pub mod gemma2;

#[cfg(all(feature = "arch-gemma3", any(feature = "cuda", feature = "metal")))]
#[path = "arch/gemma3.rs"]
pub mod gemma3;

#[cfg(all(feature = "arch-gemma3-mm", feature = "cuda"))]
#[path = "arch/gemma3_mm.rs"]
pub mod gemma3_mm;

#[cfg(all(feature = "arch-gemma4", any(feature = "cuda", feature = "metal")))]
#[path = "arch/gemma4.rs"]
pub mod gemma4;

#[cfg(all(feature = "arch-gemma4-moe", any(feature = "cuda", feature = "metal")))]
#[path = "arch/gemma4_moe.rs"]
pub mod gemma4_moe;

#[cfg(all(
    feature = "arch-granite",
    any(feature = "cuda", feature = "metal", feature = "spyre")
))]
#[path = "arch/granite.rs"]
pub mod granite;

#[cfg(all(
    feature = "arch-llama",
    any(feature = "cuda", feature = "metal", feature = "spyre")
))]
#[path = "arch/llama.rs"]
pub mod llama;

#[cfg(all(
    feature = "arch-locateanything",
    any(feature = "cuda", feature = "metal")
))]
#[path = "arch/locateanything.rs"]
pub mod locateanything;

#[cfg(all(feature = "arch-mistral", any(feature = "cuda", feature = "metal")))]
#[path = "arch/mistral.rs"]
pub mod mistral;

#[cfg(all(feature = "arch-mixtral", any(feature = "cuda", feature = "metal")))]
#[path = "arch/mixtral.rs"]
pub mod mixtral;

#[cfg(all(feature = "arch-modernbert", any(feature = "cuda", feature = "metal")))]
#[path = "arch/modernbert.rs"]
pub mod modernbert;

#[cfg(all(feature = "arch-phi3", any(feature = "cuda", feature = "metal")))]
#[path = "arch/phi3.rs"]
pub mod phi3;

#[cfg(all(feature = "arch-qwen2", any(feature = "cuda", feature = "metal")))]
#[path = "arch/qwen2.rs"]
pub mod qwen2;

#[cfg(all(feature = "arch-qwen2-5-vl", any(feature = "cuda", feature = "metal")))]
#[path = "arch/qwen2_5_vl.rs"]
pub mod qwen2_5_vl;

#[cfg(all(feature = "arch-qwen2-moe", any(feature = "cuda", feature = "metal")))]
#[path = "arch/qwen2_moe.rs"]
pub mod qwen2_moe;

#[cfg(all(feature = "arch-qwen2-vl", any(feature = "cuda", feature = "metal")))]
#[path = "arch/qwen2_vl.rs"]
pub mod qwen2_vl;

#[cfg(all(feature = "arch-qwen3", any(feature = "cuda", feature = "metal")))]
#[path = "arch/qwen3.rs"]
pub mod qwen3;

#[cfg(all(feature = "arch-qwen3-5", any(feature = "cuda", feature = "metal")))]
#[path = "arch/qwen3_5.rs"]
pub mod qwen3_5;

#[cfg(all(feature = "arch-qwen3-5-moe", any(feature = "cuda", feature = "metal")))]
#[path = "arch/qwen3_5_moe.rs"]
pub mod qwen3_5_moe;

#[cfg(all(feature = "arch-qwen3-5-vl", any(feature = "cuda", feature = "metal")))]
#[path = "arch/qwen3_5_vl.rs"]
pub mod qwen3_5_vl;

#[cfg(all(feature = "arch-qwen3-moe", any(feature = "cuda", feature = "metal")))]
#[path = "arch/qwen3_moe.rs"]
pub mod qwen3_moe;

// ── Force-link the vision arch modules (consolidation-regression guard) ──
// A vision arch's ONLY inventory rows are `ScratchyMmRegistration` submits; its
// text backbone loads via the sibling text arch (e.g. qwen2-vl → qwen2), so
// nothing else in the binary references the vision module. Now that every arch
// is consolidated into THIS crate, `extern crate scratchy_models as _` pulls
// the crate root but the linker still dead-strips any arch-module object that
// nothing references — silently emptying the MM inventory, so `serve` drops
// every image and a VL model answers as if it were text-only. Referencing each
// vision arch's `PROCESSOR` from a `#[used]` anchor forces the module object
// into the final link, so its `inventory::submit!`s actually register.
#[cfg(all(feature = "arch-qwen2-vl", any(feature = "cuda", feature = "metal")))]
#[used]
static _FORCE_MM_QWEN2_VL: &scratchy_vision::MmMetadata = &qwen2_vl::PROCESSOR;

#[cfg(all(feature = "arch-qwen2-5-vl", any(feature = "cuda", feature = "metal")))]
#[used]
static _FORCE_MM_QWEN2_5_VL: &scratchy_vision::MmMetadata = &qwen2_5_vl::PROCESSOR;

#[cfg(all(feature = "arch-qwen3-5-vl", any(feature = "cuda", feature = "metal")))]
#[used]
static _FORCE_MM_QWEN3_5_VL: &scratchy_vision::MmMetadata = &qwen3_5_vl::PROCESSOR;

#[cfg(all(feature = "arch-gemma3-mm", feature = "cuda"))]
#[used]
static _FORCE_MM_GEMMA3_MM: &scratchy_vision::MmMetadata = &gemma3_mm::PROCESSOR;

#[cfg(all(
    feature = "arch-locateanything",
    any(feature = "cuda", feature = "metal")
))]
#[used]
static _FORCE_MM_LOCATEANYTHING: &scratchy_vision::MmMetadata = &locateanything::PROCESSOR;
