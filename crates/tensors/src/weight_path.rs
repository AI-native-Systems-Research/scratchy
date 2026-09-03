// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral safetensors weight-name path builders.
//!
//! Pure `String`-building helpers shared by the forward-compiler codegen and the
//! per-target layered-load helpers. They live in the universal leaf crate so
//! both `targets/{cuda,metal}` and the compiler can name them without a
//! dependency cycle (metal cannot depend on the compiler).

/// Multimodal-aware decoder layered key. `root` is the per-arch
/// `<prefix>.layers` template (e.g. `"model.layers"` for text-only and
/// Qwen-style VL, `"language_model.model.layers"` for Gemma3-MM-style
/// arches that nest the text decoder under `language_model.<...>`).
/// Mirrors [`vision_block_weight_path`] for the decoder side.
#[inline]
pub fn layer_weight_path_with_root(root: &str, layer: u32, suffix: &str) -> String {
    format!("{root}.{layer}.{suffix}")
}

/// Vision-tower analogue: `<root>.<layer>.<suffix>` where `root` is
/// the per-arch concatenation `<default_root>.<layered_subpath>`
/// derived from `vision_safetensors_layout` (e.g. `visual.blocks` for
/// Qwen2-VL / Qwen2.5-VL or `vision_tower.vision_model.encoder.layers`
/// for SigLIP-style encoders like Gemma3-MM). Used by the
/// `load_layered_*_vision` helpers when the codegen emits a
/// `#[vision_forward]` body — the per-block prefix differs from the
/// decoder's `model.layers.<L>.` convention.
#[inline]
pub fn vision_block_weight_path(root: &str, layer: u32, suffix: &str) -> String {
    format!("{root}.{layer}.{suffix}")
}
