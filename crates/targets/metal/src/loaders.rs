// SPDX-License-Identifier: Apache-2.0
//! Metal layered-load helper surface.
//!
//! The forward-compiler macro emits `crate::__gpu::load_layered_*` per layered
//! accessor. The fully-neutral loaders (every per-item load is a neutral free
//! fn) are shared from `scratchy_layers::loaders`; this module supplies the
//! metal-divergent ones — the `RmsNorm` keep-dtype load, the direct-write
//! `load_dense_concat_packed`, and the metal-only MLX-affine / NVFP4 quant
//! loaders — calling the metal `*Ops` upload traits in `crate::layers`.

use anyhow::Result;
use scratchy_tensors::{layer_weight_path_with_root, vision_block_weight_path};

use crate::layers::{LinearLayer, MetalLinearLayerOps, RmsNorm};
use crate::layers::{LinearLayerOps, RmsNormOps};
use crate::weights::GpuWeights;

// Fully-neutral layered loaders shared from scratchy-layers (both backends).
pub use scratchy_layers::loaders::{
    load_layered_embedding, load_layered_layer_norm, load_layered_layer_norm_vision,
    load_layered_linear_dense, load_layered_linear_dense_vision,
};

/// Build a `Vec<&str>` of fully-qualified weight paths for a
/// concat-style accessor at one specific layer. The returned `paths`
/// owns the `String`s the `&str` borrows point into; the caller
/// must keep `paths` alive for the duration of the borrow.
#[inline]
fn concat_paths_for_layer(root: &str, layer: u32, suffixes: &[&str]) -> Vec<String> {
    suffixes
        .iter()
        .map(|s| layer_weight_path_with_root(root, layer, s))
        .collect()
}

#[inline]
fn as_str_refs(paths: &[String]) -> Vec<&str> {
    paths.iter().map(|s| s.as_str()).collect()
}

/// Metal-divergent layered RMSNorm load: `RmsNorm::load` routes through
/// `take_keep_dtype` so the on-disk gain dtype reaches the device verbatim
/// (the cuda variant load-time-casts to bf16), so it cannot be neutral.
pub fn load_layered_rms_norm(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffix: &str,
    eps: f32,
) -> Result<Vec<RmsNorm>> {
    (0..n_layers)
        .map(|layer| RmsNorm::load(gw, &layer_weight_path_with_root(root, layer, suffix), eps))
        .collect()
}

/// Vision-tower analogue of [`load_layered_rms_norm`]:
/// `visual.blocks.<L>.<suffix>` per-block prefix instead of
/// `model.layers.<L>.<suffix>`. Used by Qwen2.5-VL's `norm1` / `norm2`
/// per-block RMSNorms (Qwen2-VL uses LayerNorm-with-bias via the
/// `MeanSubRmsNormBiasAdd` matcher chain instead).
pub fn load_layered_rms_norm_vision(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffix: &str,
    eps: f32,
) -> Result<Vec<RmsNorm>> {
    (0..n_layers)
        .map(|layer| RmsNorm::load(gw, &vision_block_weight_path(root, layer, suffix), eps))
        .collect()
}

/// Stream-free CPU-concat then one direct-write pack per layer via
/// [`LinearLayer::load_dense_concat_packed`]. The metal pack streams each source
/// tensor's bytes straight into a pre-allocated `MTLBuffer`, so it's
/// metal-divergent and stays per-target.
pub fn load_layered_linear_dense_concat_packed(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffixes: &[&str],
) -> Result<Vec<LinearLayer>> {
    (0..n_layers)
        .map(|layer| {
            let paths = concat_paths_for_layer(root, layer, suffixes);
            let refs = as_str_refs(&paths);
            LinearLayer::load_dense_concat_packed(gw, &refs)
        })
        .collect()
}

/// MLX-affine int4 layered linear load (Metal-only). Reads the
/// `<root>.<layer>.<suffix>.{weight,scales,biases,bias?}` triple per
/// decoder layer. Produces AffineQuant LinearLayers — forward-time
/// qmv path consumer (P3+).
pub fn load_layered_linear_affine_quant(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffix: &str,
    group_size: u32,
    bits: u32,
    in_features: u32,
) -> Result<Vec<LinearLayer>> {
    (0..n_layers)
        .map(|layer| {
            LinearLayer::load_affine_quant(
                gw,
                &layer_weight_path_with_root(root, layer, suffix),
                group_size,
                bits,
                in_features,
            )
        })
        .collect()
}

/// NVFP4 int4 layered forward-time load (Metal-only). One
/// `LinearLayer::Nvfp4` per decoder layer, packed E2M1 weight + folded
/// F16 scales kept on device for the `nvfp4_qmv` / `nvfp4_qmm_t`
/// dispatchers `MetalNvfp4QmmImpl` emits. Mirrors
/// [`load_layered_linear_affine_quant`].
pub fn load_layered_linear_nvfp4_quant(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffix: &str,
    group_size: u32,
) -> Result<Vec<LinearLayer>> {
    (0..n_layers)
        .map(|layer| {
            LinearLayer::load_nvfp4_quant(
                gw,
                &layer_weight_path_with_root(root, layer, suffix),
                group_size,
            )
        })
        .collect()
}

/// Fused-concat sibling of [`load_layered_linear_nvfp4_quant`]. One
/// `LinearLayer::Nvfp4` per layer, each the byte-concat of the per-layer
/// NVFP4 prefixes (gate / up, or q / k / v).
pub fn load_layered_linear_nvfp4_quant_concat(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffixes: &[&str],
    group_size: u32,
) -> Result<Vec<LinearLayer>> {
    (0..n_layers)
        .map(|layer| {
            let paths = concat_paths_for_layer(root, layer, suffixes);
            let refs = as_str_refs(&paths);
            LinearLayer::load_nvfp4_quant_concat(gw, &refs, group_size)
        })
        .collect()
}

/// MLX-affine int4 layered dequant-as-dense load (Metal-only). INT4
/// P2 slow-reference path: CPU-dequantize each per-layer affine
/// triple into a BF16 Dense Linear. Mirrors
/// [`load_layered_linear_affine_quant`] but the macro emits this
/// (not the AffineQuant variant) so the forward path stays on the
/// existing dense Gemm impls.
pub fn load_layered_linear_affine_dequant_as_dense(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffix: &str,
    group_size: u32,
    bits: u32,
) -> Result<Vec<LinearLayer>> {
    (0..n_layers)
        .map(|layer| {
            LinearLayer::load_affine_dequant_as_dense(
                gw,
                &layer_weight_path_with_root(root, layer, suffix),
                group_size,
                bits,
            )
        })
        .collect()
}

/// Fused-concat sibling of [`load_layered_linear_affine_dequant_as_dense`].
/// One Dense per layer, each containing the byte-concat of the per-layer
/// affine prefixes (gate / up, or q / k / v).
pub fn load_layered_linear_affine_dequant_concat_as_dense(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffixes: &[&str],
    group_size: u32,
    bits: u32,
    in_features: u32,
) -> Result<Vec<LinearLayer>> {
    (0..n_layers)
        .map(|layer| {
            let paths = concat_paths_for_layer(root, layer, suffixes);
            let refs = as_str_refs(&paths);
            LinearLayer::load_affine_dequant_concat_as_dense(
                gw,
                &refs,
                group_size,
                bits,
                in_features,
            )
        })
        .collect()
}

/// Per-layer mixed-bit sibling of [`load_layered_linear_affine_quant`].
/// MLX-native dynamic quant (OptiQ) ships different numbered layers at
/// different bit-widths; `bits_per_layer[l]` is layer `l`'s on-disk width
/// (4 or 8) at the shared `group_size`. The uniform helper stays the fast
/// path — this is only emitted for a heterogeneous layered group.
pub fn load_layered_linear_affine_quant_mixed(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffix: &str,
    group_size: u32,
    bits_per_layer: &[u32],
    in_features: u32,
) -> Result<Vec<LinearLayer>> {
    (0..n_layers)
        .map(|layer| {
            LinearLayer::load_affine_quant(
                gw,
                &layer_weight_path_with_root(root, layer, suffix),
                group_size,
                bits_per_layer[layer as usize],
                in_features,
            )
        })
        .collect()
}

/// Per-layer mixed-bit sibling of
/// [`load_layered_linear_affine_dequant_concat_as_dense`].
pub fn load_layered_linear_affine_dequant_concat_as_dense_mixed(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffixes: &[&str],
    group_size: u32,
    bits_per_layer: &[u32],
    in_features: u32,
) -> Result<Vec<LinearLayer>> {
    (0..n_layers)
        .map(|layer| {
            let paths = concat_paths_for_layer(root, layer, suffixes);
            let refs = as_str_refs(&paths);
            LinearLayer::load_affine_dequant_concat_as_dense(
                gw,
                &refs,
                group_size,
                bits_per_layer[layer as usize],
                in_features,
            )
        })
        .collect()
}
