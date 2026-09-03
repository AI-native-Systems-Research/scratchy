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

/// Preset names (matching `quantizations.json` entries verbatim) whose
/// same-named Cargo feature is active on THIS compilation of the crate.
/// Baked in at compile time via `cfg!` — the only caller is
/// `crates/models/arch/scratchy-forwards.rs` (scratchy-models' build
/// script), which depends on this crate as a build-dependency specifically
/// so it can call this instead of reading its own now-nonexistent
/// `CARGO_FEATURE_QUANT_*` (quant scope lives here now, not on
/// scratchy-models — see the `[features]` comment in this crate's
/// Cargo.toml for why). Feature name == preset name exactly (no `quant-`
/// prefix), so this is a straight cfg-gated list, not a name mapping.
pub fn enabled_presets() -> Vec<&'static str> {
    macro_rules! preset {
        ($v:ident, $name:literal) => {
            if cfg!(feature = $name) {
                $v.push($name);
            }
        };
    }
    let mut v = Vec::new();
    preset!(v, "awq-gemm");
    preset!(v, "bnb-nf4-dq");
    preset!(v, "ct-int4-sym");
    preset!(v, "fp8-block-128x128");
    preset!(v, "fp8-dynamic-per-channel");
    preset!(v, "fp8-dynamic-per-tensor");
    preset!(v, "fp8-static-per-tensor");
    preset!(v, "gemma4-moe-ct-int4-sym");
    preset!(v, "ggml");
    preset!(v, "gptq-sym");
    preset!(v, "gptq-sym-desc_act");
    preset!(v, "mlx-affine-b4-g128");
    preset!(v, "mlx-affine-b4-g128-qembed");
    preset!(v, "mlx-affine-b4-g32");
    preset!(v, "mlx-affine-b4-g64");
    preset!(v, "mlx-affine-b4-g64-gate8-qembed");
    preset!(v, "mlx-affine-b4-g64-mlp8-router8");
    preset!(v, "mlx-affine-b4-g64-qembed");
    preset!(v, "nvfp4");
    v
}

pub use affine::affine_dequant_b4_to_dtype;
pub use ggml_quant::{GgmlDType, GgmlStorage};
pub use layers_quant::{AffineQuantEmbedding, AffineQuantLinear, Nvfp4Linear};

// Crate-root re-exports so the `#[macro_export]`ed `register!` macro's
// `$crate::…` paths resolve at the crate root where the macro lands.
pub use gguf::inventory;
pub use gguf::{GgufArchSpec, GgufDefault};
