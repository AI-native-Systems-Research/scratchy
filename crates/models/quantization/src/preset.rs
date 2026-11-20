// SPDX-License-Identifier: Apache-2.0
//! The typed quant presets: [`Preset`] and its [`Scheme`], expanded from the one list
//! (`presets.rs`'s `with_presets!`). A target states the presets it lowers as a const of these,
//! derived from their schemes ([`Preset::of_schemes`]), so no target matches a preset by its name.

/// The family of kernels that read a preset's weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Scheme {
    /// AWQ int4 (Marlin).
    Awq,
    /// bitsandbytes NF4.
    Bnb4,
    /// compressed-tensors int4.
    CompressedTensorsInt4,
    /// FP8 (e4m3) weights, block- or channel- or tensor-scaled.
    Fp8,
    /// GGUF (llama.cpp) quant types.
    Ggml,
    /// GPTQ int4 (Marlin).
    Gptq,
    /// MLX affine codes with per-group scales and biases.
    MlxAffine,
    /// NVFP4 (E2M1 with FP8 scales).
    Nvfp4,
}

impl Scheme {
    /// The Hugging Face Hub's library or quantization tag for checkpoints of this scheme.
    pub const fn hub_tag(self) -> &'static str {
        match self {
            Scheme::Awq => "awq",
            Scheme::Bnb4 => "bitsandbytes",
            Scheme::CompressedTensorsInt4 => "compressed-tensors",
            Scheme::Fp8 => "fp8",
            Scheme::Ggml => "gguf",
            Scheme::Gptq => "gptq",
            Scheme::MlxAffine => "mlx",
            Scheme::Nvfp4 => "nvfp4",
        }
    }
}

macro_rules! define_presets {
    ($($variant:ident $name:literal $scheme:ident,)*) => {
        /// A quant preset (`quant/<preset>`).
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum Preset {
            $($variant,)*
        }

        impl Preset {
            /// Every preset, in name order.
            pub const ALL: &'static [Preset] = &[$(Preset::$variant,)*];

            /// Its name: its `presets/<name>.json` file and its Cargo feature.
            pub const fn name(self) -> &'static str {
                match self {
                    $(Preset::$variant => $name,)*
                }
            }

            /// The family of kernels that read its weights.
            pub const fn scheme(self) -> Scheme {
                match self {
                    $(Preset::$variant => Scheme::$scheme,)*
                }
            }

            /// Whether its same-named Cargo feature is active on THIS compilation of the crate.
            pub const fn enabled(self) -> bool {
                match self {
                    $(Preset::$variant => cfg!(feature = $name),)*
                }
            }
        }
    };
}

with_presets!(define_presets);

impl Preset {
    /// The preset a name states (a `quantizations.json` entry, a build's published scope): the
    /// one place a preset is read by its name.
    pub fn from_name(name: &str) -> Option<Preset> {
        Preset::ALL.iter().copied().find(|p| p.name() == name)
    }

    /// How many presets have a scheme in `schemes`: [`Self::of_schemes`]' length.
    pub const fn count_of(schemes: &[Scheme]) -> usize {
        let mut n = 0;
        let mut i = 0;
        while i < Preset::ALL.len() {
            if Preset::ALL[i].in_schemes(schemes) {
                n += 1;
            }
            i += 1;
        }
        n
    }

    /// The presets whose scheme is in `schemes`, in name order; `N` is
    /// [`Self::count_of`]`(schemes)`, checked at compile time.
    pub const fn of_schemes<const N: usize>(schemes: &[Scheme]) -> [Preset; N] {
        assert!(
            N == Preset::count_of(schemes),
            "N is not the presets of these schemes"
        );
        let mut out = [Preset::ALL[0]; N];
        let (mut i, mut n) = (0, 0);
        while i < Preset::ALL.len() {
            if Preset::ALL[i].in_schemes(schemes) {
                out[n] = Preset::ALL[i];
                n += 1;
            }
            i += 1;
        }
        out
    }

    const fn in_schemes(self, schemes: &[Scheme]) -> bool {
        let mut j = 0;
        while j < schemes.len() {
            if self.scheme() as u8 == schemes[j] as u8 {
                return true;
            }
            j += 1;
        }
        false
    }
}

/// The presets whose same-named Cargo feature is active on THIS compilation of the crate, in
/// name order.
pub fn enabled_presets() -> Vec<Preset> {
    Preset::ALL
        .iter()
        .copied()
        .filter(|p| p.enabled())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Preset, Scheme};

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

    /// A scheme set picks exactly the presets of those schemes, in name order.
    #[test]
    fn of_schemes_is_the_presets_of_those_schemes() {
        const S: &[Scheme] = &[Scheme::Fp8, Scheme::Nvfp4];
        const P: [Preset; Preset::count_of(S)] = Preset::of_schemes(S);
        let want: Vec<Preset> = (Preset::ALL.iter().copied())
            .filter(|p| S.contains(&p.scheme()))
            .collect();
        assert_eq!(P.as_slice(), want.as_slice());
    }
}
