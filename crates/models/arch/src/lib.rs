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

// The tokenizers of the models this binary compiles, baked by the build script from the Hub-cache
// checkpoints of those models (`scratchy_core_model::baked_tokenizer`).
include!(concat!(env!("OUT_DIR"), "/baked_tokenizers.rs"));

#[cfg(test)]
mod baked_tokenizer_tables {
    use scratchy_core_model::baked_tokenizer::BakedTokenizer;

    /// Checkpoints whose `tokenizer.json`s bake the same tables (fine-tunes of one base) share one
    /// embedded copy of them: two baked tokenizers with equal tables point at the same bytes.
    #[test]
    fn equal_tables_are_embedded_once() {
        let baked: Vec<_> =
            scratchy_forward_compiler::inventory::iter::<BakedTokenizer>().collect();
        for (i, a) in baked.iter().enumerate() {
            for b in &baked[i + 1..] {
                assert!(
                    a.tables.0 != b.tables.0 || std::ptr::eq(a.tables, b.tables),
                    "{} and {} embed the same {}-byte tables twice",
                    a.digest,
                    b.digest,
                    a.tables.0.len(),
                );
            }
        }
    }
}

// Every baked metal kernel the arches below name (`crate::__metal_bake::K_<key>`), each once,
// beside the metallibs holding them — baked by the build script after it emits every arch.
#[cfg(baked_kernels)]
mod __metal_bake {
    include!(concat!(env!("OUT_DIR"), "/metal_bake.rs"));
}

#[cfg(compiled_arch = "commandr")]
#[path = "arch/commandr.rs"]
pub mod commandr;

#[cfg(compiled_arch = "gemma2")]
#[path = "arch/gemma2.rs"]
pub mod gemma2;

#[cfg(compiled_arch = "gemma3")]
#[path = "arch/gemma3.rs"]
pub mod gemma3;

#[cfg(compiled_arch = "gemma4")]
#[path = "arch/gemma4.rs"]
pub mod gemma4;

#[cfg(compiled_arch = "gemma4-moe")]
#[path = "arch/gemma4_moe.rs"]
pub mod gemma4_moe;

#[cfg(compiled_arch = "glm4-moe")]
#[path = "arch/glm4_moe.rs"]
pub mod glm4_moe;

#[cfg(compiled_arch = "gpt-oss")]
#[path = "arch/gpt_oss.rs"]
pub mod gpt_oss;

#[cfg(compiled_arch = "granite")]
#[path = "arch/granite.rs"]
pub mod granite;

#[cfg(compiled_arch = "llama")]
#[path = "arch/llama.rs"]
pub mod llama;

#[cfg(compiled_arch = "locateanything")]
#[path = "arch/locateanything.rs"]
pub mod locateanything;

#[cfg(compiled_arch = "mistral")]
#[path = "arch/mistral.rs"]
pub mod mistral;

#[cfg(compiled_arch = "mixtral")]
#[path = "arch/mixtral.rs"]
pub mod mixtral;

#[cfg(compiled_arch = "modernbert")]
#[path = "arch/modernbert.rs"]
pub mod modernbert;

#[cfg(compiled_arch = "nemotron-h")]
#[path = "arch/nemotron_h.rs"]
pub mod nemotron_h;

#[cfg(compiled_arch = "nemotron-h-latent")]
#[path = "arch/nemotron_h_latent.rs"]
pub mod nemotron_h_latent;

#[cfg(compiled_arch = "nemotron-h-mtp")]
#[path = "arch/nemotron_h_mtp.rs"]
pub mod nemotron_h_mtp;

#[cfg(compiled_arch = "phi3")]
#[path = "arch/phi3.rs"]
pub mod phi3;

#[cfg(compiled_arch = "qwen2")]
#[path = "arch/qwen2.rs"]
pub mod qwen2;

#[cfg(compiled_arch = "qwen2-5-vl")]
#[path = "arch/qwen2_5_vl.rs"]
pub mod qwen2_5_vl;

#[cfg(compiled_arch = "qwen2-moe")]
#[path = "arch/qwen2_moe.rs"]
pub mod qwen2_moe;

#[cfg(compiled_arch = "qwen2-vl")]
#[path = "arch/qwen2_vl.rs"]
pub mod qwen2_vl;

#[cfg(compiled_arch = "qwen3")]
#[path = "arch/qwen3.rs"]
pub mod qwen3;

#[cfg(compiled_arch = "qwen3-5")]
#[path = "arch/qwen3_5.rs"]
pub mod qwen3_5;

#[cfg(compiled_arch = "qwen3-5-moe")]
#[path = "arch/qwen3_5_moe.rs"]
pub mod qwen3_5_moe;

#[cfg(compiled_arch = "qwen3-5-dense-mtp")]
#[path = "arch/qwen3_5_dense_mtp.rs"]
pub mod qwen3_5_dense_mtp;
#[cfg(compiled_arch = "qwen3-5-mtp")]
#[path = "arch/qwen3_5_mtp.rs"]
pub mod qwen3_5_mtp;

#[cfg(compiled_arch = "gemma4-assistant")]
#[path = "arch/gemma4_assistant.rs"]
pub mod gemma4_assistant;

#[cfg(compiled_arch = "dflash2")]
#[path = "arch/dflash2.rs"]
pub mod dflash2;

#[cfg(compiled_arch = "qwen3-5-vl")]
#[path = "arch/qwen3_5_vl.rs"]
pub mod qwen3_5_vl;

#[cfg(compiled_arch = "qwen3-moe")]
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
#[cfg(compiled_arch = "qwen2-vl")]
#[used]
static _FORCE_MM_QWEN2_VL: &scratchy_vision::MmMetadata = &qwen2_vl::PROCESSOR;

#[cfg(compiled_arch = "qwen2-5-vl")]
#[used]
static _FORCE_MM_QWEN2_5_VL: &scratchy_vision::MmMetadata = &qwen2_5_vl::PROCESSOR;

#[cfg(compiled_arch = "qwen3-5-vl")]
#[used]
static _FORCE_MM_QWEN3_5_VL: &scratchy_vision::MmMetadata = &qwen3_5_vl::PROCESSOR;

#[cfg(compiled_arch = "locateanything")]
#[used]
static _FORCE_MM_LOCATEANYTHING: &scratchy_vision::MmMetadata = &locateanything::PROCESSOR;
