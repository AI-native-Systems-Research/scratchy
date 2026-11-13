// SPDX-License-Identifier: Apache-2.0
//! The models: every arch's DSL carrier (`dsl/<arch>.py`) and configs (`configs/<arch>/`), and
//! the model SCOPE — which models, heads and quant presets a build compiles — as this crate's
//! features. The build script publishes the scope as build metadata (`links`); each target's
//! models crate reads it back and compiles that selection with the target's emitter
//! (`scratchy_models_build`). This crate compiles no model itself.

// Real HF org/repo ids resolved once against huggingface.co at BUILD time
// (`hf_registry_build.rs`), scoped to exactly the architectures this binary
// was compiled to run. `scr model names` completes from this — no runtime
// network call, no local hf-hub cache read.
include!(concat!(env!("OUT_DIR"), "/hf_registry.rs"));

/// The compiled-in HF model registry — see [`COMPILED_HF_REGISTRY`].
pub fn compiled_hf_registry() -> &'static [&'static str] {
    COMPILED_HF_REGISTRY
}
