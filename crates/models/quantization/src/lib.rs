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

pub use presets::{Preset, enabled_presets};

pub use affine::affine_dequant_b4_to_dtype;
pub use ggml_quant::{GgmlDType, GgmlStorage};
pub use layers_quant::{AffineQuantEmbedding, AffineQuantLinear, Nvfp4Linear};

// Crate-root re-exports so the `#[macro_export]`ed `register!` macro's
// `$crate::…` paths resolve at the crate root where the macro lands.
pub use gguf::inventory;
pub use gguf::{GgufArchSpec, GgufDefault};

impl Preset {
    /// The preset a name states (a `quantizations.json` entry, a `presets/<name>.json` file, a
    /// build's published scope): the one place a preset is read by its name.
    pub fn from_name(name: &str) -> Option<Preset> {
        Preset::ALL.iter().copied().find(|p| p.name() == name)
    }
}

#[cfg(test)]
mod preset_tests {
    use super::Preset;

    #[test]
    fn every_preset_is_found_by_its_name_and_the_names_are_sorted_and_distinct() {
        for &p in Preset::ALL {
            assert_eq!(Preset::from_name(p.name()), Some(p), "{p:?}");
        }
        let names: Vec<&str> = Preset::ALL.iter().map(|p| p.name()).collect();
        assert!(names.windows(2).all(|w| w[0] < w[1]), "{names:?}");
        assert_eq!(Preset::from_name("no-such-preset"), None);
    }

    #[test]
    fn every_preset_has_its_json_file() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("presets");
        for &p in Preset::ALL {
            assert!(dir.join(format!("{}.json", p.name())).is_file(), "{p:?}");
        }
    }
}
