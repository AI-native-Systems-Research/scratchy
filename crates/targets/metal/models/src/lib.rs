// SPDX-License-Identifier: Apache-2.0
// Generated code: the emitted modules mirror model stems and fused weight names, and bring the
// loader traits into scope whether or not a given model reads them.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused_imports)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! The selected models, compiled for metal: one `pub mod <arch>` per arch the scope selects
//! (`scratchy-models`' features), each holding its models' baked tapes, pools and weights. The
//! build script runs every selected arch through the shared pipeline with metal's emitter into
//! `$OUT_DIR/<mod>.rs`, which `arch/<mod>.rs` includes. Runtime discovery is via the emitted
//! `inventory::submit!` (see `scratchy_forward_compiler`), so a consumer
//! `extern crate scratchy_target_metal_models as _;` to force-link it.

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
mod __metal_bake {
    include!(concat!(env!("OUT_DIR"), "/metal_bake.rs"));
}

// Every model this build bakes a metal tape for, beside its buckets — listed by the build script
// off the modules it emitted.
include!(concat!(env!("OUT_DIR"), "/metal_models.rs"));

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

#[cfg(compiled_arch = "gemma4_moe")]
#[path = "arch/gemma4_moe.rs"]
pub mod gemma4_moe;

#[cfg(compiled_arch = "glm4_moe")]
#[path = "arch/glm4_moe.rs"]
pub mod glm4_moe;

#[cfg(compiled_arch = "gpt_oss")]
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

#[cfg(compiled_arch = "nemotron_h")]
#[path = "arch/nemotron_h.rs"]
pub mod nemotron_h;

#[cfg(compiled_arch = "nemotron_h_latent")]
#[path = "arch/nemotron_h_latent.rs"]
pub mod nemotron_h_latent;

#[cfg(compiled_arch = "nemotron_h_mtp")]
#[path = "arch/nemotron_h_mtp.rs"]
pub mod nemotron_h_mtp;

#[cfg(compiled_arch = "phi3")]
#[path = "arch/phi3.rs"]
pub mod phi3;

#[cfg(compiled_arch = "qwen2")]
#[path = "arch/qwen2.rs"]
pub mod qwen2;

#[cfg(compiled_arch = "qwen2_5_vl")]
#[path = "arch/qwen2_5_vl.rs"]
pub mod qwen2_5_vl;

#[cfg(compiled_arch = "qwen2_moe")]
#[path = "arch/qwen2_moe.rs"]
pub mod qwen2_moe;

#[cfg(compiled_arch = "qwen2_vl")]
#[path = "arch/qwen2_vl.rs"]
pub mod qwen2_vl;

#[cfg(compiled_arch = "qwen3")]
#[path = "arch/qwen3.rs"]
pub mod qwen3;

#[cfg(compiled_arch = "qwen3_5")]
#[path = "arch/qwen3_5.rs"]
pub mod qwen3_5;

#[cfg(compiled_arch = "qwen3_5_moe")]
#[path = "arch/qwen3_5_moe.rs"]
pub mod qwen3_5_moe;

#[cfg(compiled_arch = "qwen3_5_dense_mtp")]
#[path = "arch/qwen3_5_dense_mtp.rs"]
pub mod qwen3_5_dense_mtp;
#[cfg(compiled_arch = "qwen3_5_mtp")]
#[path = "arch/qwen3_5_mtp.rs"]
pub mod qwen3_5_mtp;

#[cfg(compiled_arch = "gemma4_assistant")]
#[path = "arch/gemma4_assistant.rs"]
pub mod gemma4_assistant;

#[cfg(compiled_arch = "dflash2")]
#[path = "arch/dflash2.rs"]
pub mod dflash2;

#[cfg(compiled_arch = "qwen3_5_vl")]
#[path = "arch/qwen3_5_vl.rs"]
pub mod qwen3_5_vl;

#[cfg(compiled_arch = "qwen3_moe")]
#[path = "arch/qwen3_moe.rs"]
pub mod qwen3_moe;

// ── Force-link the vision arch modules (consolidation-regression guard) ──
// A vision arch's ONLY inventory rows are `ScratchyMmRegistration` submits; its
// text backbone loads via the sibling text arch (e.g. qwen2-vl → qwen2), so
// nothing else in the binary references the vision module. Now that every arch
// is consolidated into THIS crate, `extern crate scratchy_target_metal_models as _` pulls
// the crate root but the linker still dead-strips any arch-module object that
// nothing references — silently emptying the MM inventory, so `serve` drops
// every image and a VL model answers as if it were text-only. Referencing each
// vision arch's `PROCESSOR` from a `#[used]` anchor forces the module object
// into the final link, so its `inventory::submit!`s actually register.
#[cfg(compiled_arch = "qwen2_vl")]
#[used]
static _FORCE_MM_QWEN2_VL: &scratchy_vision::MmMetadata = &qwen2_vl::PROCESSOR;

#[cfg(compiled_arch = "qwen2_5_vl")]
#[used]
static _FORCE_MM_QWEN2_5_VL: &scratchy_vision::MmMetadata = &qwen2_5_vl::PROCESSOR;

#[cfg(compiled_arch = "qwen3_5_vl")]
#[used]
static _FORCE_MM_QWEN3_5_VL: &scratchy_vision::MmMetadata = &qwen3_5_vl::PROCESSOR;

#[cfg(compiled_arch = "locateanything")]
#[used]
static _FORCE_MM_LOCATEANYTHING: &scratchy_vision::MmMetadata = &locateanything::PROCESSOR;
