// SPDX-License-Identifier: Apache-2.0
// Shared by the library (`lib.rs`) and the build script (`build.rs`), both through `include!`:
// a build script is compiled with its package's features, so the same `cfg!` list answers in both.

/// Preset names (matching `quantizations.json` entries verbatim) whose same-named Cargo feature is
/// active on THIS compilation of the crate. Feature name == preset name exactly (no `quant-`
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
    preset!(v, "mlx-affine-b2-g64-attn4-router8-qembed");
    preset!(v, "mlx-affine-b2-g64-qembed");
    preset!(v, "mlx-affine-b3-g64-attn4-router8-qembed");
    preset!(v, "mlx-affine-b3-g64-qembed");
    preset!(v, "mlx-affine-b4-g128");
    preset!(v, "mlx-affine-b4-g128-qembed");
    preset!(v, "mlx-affine-b4-g32");
    preset!(v, "mlx-affine-b4-g64");
    preset!(v, "mlx-affine-b4-g64-mlp8-router8");
    preset!(v, "mlx-affine-b4-g64-qembed");
    preset!(v, "nvfp4");
    v
}
