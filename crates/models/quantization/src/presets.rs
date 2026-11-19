// SPDX-License-Identifier: Apache-2.0
// Shared by the library (`lib.rs`) and the build script (`build.rs`), both through `include!`:
// a build script is compiled with its package's features, so the same `cfg!` list answers in both.

/// Every quant preset, once: its [`Preset`] value and its name, which is its `presets/<name>.json`
/// file, its `quantizations.json` entries and its same-named Cargo feature (no `quant-` prefix).
macro_rules! presets {
    ($($variant:ident $name:literal,)*) => {
        /// A quant preset (`quant/<preset>`). A target states the presets it lowers as a const
        /// slice of these, so no target matches a preset by its name.
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

            /// Whether its same-named Cargo feature is active on THIS compilation of the crate.
            pub const fn enabled(self) -> bool {
                match self {
                    $(Preset::$variant => cfg!(feature = $name),)*
                }
            }
        }
    };
}

presets! {
    AwqGemm "awq-gemm",
    BnbNf4Dq "bnb-nf4-dq",
    CtInt4Sym "ct-int4-sym",
    Fp8Block128x128 "fp8-block-128x128",
    Fp8DynamicPerChannel "fp8-dynamic-per-channel",
    Fp8DynamicPerTensor "fp8-dynamic-per-tensor",
    Fp8StaticPerTensor "fp8-static-per-tensor",
    Gemma4MoeCtInt4Sym "gemma4-moe-ct-int4-sym",
    Ggml "ggml",
    GptqSym "gptq-sym",
    GptqSymDescAct "gptq-sym-desc_act",
    MlxAffineB2G64Attn4Router8Qembed "mlx-affine-b2-g64-attn4-router8-qembed",
    MlxAffineB2G64Qembed "mlx-affine-b2-g64-qembed",
    MlxAffineB3G64Attn4Router8Qembed "mlx-affine-b3-g64-attn4-router8-qembed",
    MlxAffineB3G64Qembed "mlx-affine-b3-g64-qembed",
    MlxAffineB4G128 "mlx-affine-b4-g128",
    MlxAffineB4G128Qembed "mlx-affine-b4-g128-qembed",
    MlxAffineB4G32 "mlx-affine-b4-g32",
    MlxAffineB4G64 "mlx-affine-b4-g64",
    MlxAffineB4G64Mlp8Router8 "mlx-affine-b4-g64-mlp8-router8",
    MlxAffineB4G64Qembed "mlx-affine-b4-g64-qembed",
    Nvfp4 "nvfp4",
}

/// The names of the presets whose same-named Cargo feature is active on THIS compilation of the
/// crate, in name order.
pub fn enabled_presets() -> Vec<&'static str> {
    (Preset::ALL.iter())
        .filter(|p| p.enabled())
        .map(|p| p.name())
        .collect()
}
