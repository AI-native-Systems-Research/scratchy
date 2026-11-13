// SPDX-License-Identifier: Apache-2.0
//! Shared quantization presets + GGUF (llama.cpp) format support.
//!
//! Presets: the `#[forward]` macro discovers `presets/*.json` by walking
//! up from the consuming crate's manifest dir to the workspace root, then
//! into `crates/quantization/presets/`.
//!
//! GGUF format-level code lives in the [`gguf`] module. The pure MLX-affine
//! int4 dequant math lives in [`affine`].

pub mod affine;
pub mod ggml_quant;
pub mod gguf;
pub mod layers_quant;
mod presets;

pub use presets::enabled_presets;

pub use affine::affine_dequant_b4_to_dtype;
pub use ggml_quant::{GgmlDType, GgmlStorage};
pub use layers_quant::{AffineQuantEmbedding, AffineQuantLinear, Nvfp4Linear};

// Crate-root re-exports so the `#[macro_export]`ed `register!` macro's
// `$crate::…` paths resolve at the crate root where the macro lands.
pub use gguf::inventory;
pub use gguf::{GgufArchSpec, GgufDefault};
