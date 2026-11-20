// SPDX-License-Identifier: Apache-2.0
// THE ONE LIST of quant presets. Shared by the library (`preset.rs` expands it into `Preset`) and
// the build script (`build.rs` expands it into the enabled names it publishes), both through
// `include!`: a build script is compiled with its package's features, so the same `cfg!` answers
// in both.

/// Every quant preset, once, handed to `$expand`: its `Preset` variant, its name (its
/// `presets/<name>.json` file, its `quantizations.json` entries and its same-named Cargo feature,
/// no `quant-` prefix) and its `Scheme`, the family of kernels that read its weights.
macro_rules! with_presets {
    ($expand:ident) => {
        $expand! {
            AwqGemm "awq-gemm" Awq,
            BnbNf4Dq "bnb-nf4-dq" Bnb4,
            CtInt4Sym "ct-int4-sym" CompressedTensorsInt4,
            Fp8Block128x128 "fp8-block-128x128" Fp8,
            Fp8DynamicPerChannel "fp8-dynamic-per-channel" Fp8,
            Fp8DynamicPerTensor "fp8-dynamic-per-tensor" Fp8,
            Fp8StaticPerTensor "fp8-static-per-tensor" Fp8,
            Gemma4MoeCtInt4Sym "gemma4-moe-ct-int4-sym" CompressedTensorsInt4,
            Ggml "ggml" Ggml,
            GptqSym "gptq-sym" Gptq,
            GptqSymDescAct "gptq-sym-desc_act" Gptq,
            MlxAffineB2G64Attn4Router8Qembed "mlx-affine-b2-g64-attn4-router8-qembed" MlxAffine,
            MlxAffineB2G64Qembed "mlx-affine-b2-g64-qembed" MlxAffine,
            MlxAffineB3G64Attn4Router8Qembed "mlx-affine-b3-g64-attn4-router8-qembed" MlxAffine,
            MlxAffineB3G64Qembed "mlx-affine-b3-g64-qembed" MlxAffine,
            MlxAffineB4G128 "mlx-affine-b4-g128" MlxAffine,
            MlxAffineB4G128Qembed "mlx-affine-b4-g128-qembed" MlxAffine,
            MlxAffineB4G32 "mlx-affine-b4-g32" MlxAffine,
            MlxAffineB4G64 "mlx-affine-b4-g64" MlxAffine,
            MlxAffineB4G64Mlp8Router8 "mlx-affine-b4-g64-mlp8-router8" MlxAffine,
            MlxAffineB4G64Qembed "mlx-affine-b4-g64-qembed" MlxAffine,
            Nvfp4 "nvfp4" Nvfp4,
        }
    };
}
