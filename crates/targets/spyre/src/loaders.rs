// SPDX-License-Identifier: Apache-2.0
//! Spyre layered-load helper surface.
//!
//! The forward-compiler macro emits `crate::__gpu::load_layered_*` per layered
//! accessor. The fully-neutral loaders (every per-item load is a neutral free
//! fn) are shared from `scratchy_layers::loaders`; this module supplies the
//! spyre-divergent ones — the `RmsNorm` keep-dtype load, the direct-write
//! `load_dense_concat_packed`, and the spyre quant loaders (MLX-affine / NVFP4)
//! — calling the spyre `*Ops` load traits in `crate::layers`.
//!
//! Spyre is a HOST backend (fp16), so these helpers are allocator-generic over
//! `A: DeviceAllocator` (the host `SpyreAllocator` is the concrete `A`), exactly
//! like the neutral `scratchy_layers::loaders`. RmsNorm mirrors metal's
//! keep-dtype choice (spyre is fp16 like metal). The quant loaders are
//! kernel-bound and not yet wired on the host path; their `*Ops` constructors
//! `anyhow::bail!` at runtime, so the signatures type-check (every model
//! variant's load body compiles) while a bf16 checkpoint never calls them.

use anyhow::Result;
use scratchy_tensors::{layer_weight_path_with_root, vision_block_weight_path};

use crate::layers::{LinearLayer, LinearLayerOps, RmsNorm, RmsNormOps};
use crate::weights::GpuWeights;

// Fully-neutral layered loaders shared from scratchy-layers (every backend).
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

/// Spyre-divergent layered RMSNorm load: `RmsNorm::load` routes through
/// `take_keep_dtype` so the on-disk gain dtype reaches the host verbatim
/// (mirroring metal — spyre is fp16; the cuda variant load-time-casts to bf16),
/// so it cannot be neutral.
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

/// Layered FP8 (per-tensor / per-channel) linear load — spyre's own.
///
/// ⛔ NOT SHARED WITH CUDA, AND NOT A DUPLICATE OF IT. Cuda's `Fp8LoadOps::load`
/// is bound to its device: it takes `weights.stream()`, forces the scale to f32
/// because the CUTLASS epilogue requires `float*`, and will ONLINE-QUANTIZE a
/// bf16 checkpoint with a device kernel. None of that is meaningful on a host
/// backend that stages bytes and retiles them for the PT array.
///
/// ⛔⛔ AND IT DOES NOT DEQUANT. The fp8 code IS the device format (SEN143_FP8,
/// same bit layout), so weight and scale are taken VERBATIM — `take_keep_dtype`,
/// not `take`, which would widen the fp8 byte to the target dtype at load and
/// silently turn a W8A8 model into fp16 inference with none of the bandwidth
/// win it was quantized for.
///
/// `input_scale` / `bias` are optional in the checkpoint (dynamic-activation
/// models ship neither), so a miss is `None`, not an error.
pub fn load_layered_fp8_linear<A: scratchy_tensors::DeviceAllocator>(
    gw: &mut scratchy_layers::weights::GpuWeights<A>,
    n_layers: u32,
    root: &str,
    suffix: &str,
    output_dtype: scratchy_tensors::DType,
) -> Result<Vec<scratchy_layers::layers::Fp8AnyLinear>> {
    (0..n_layers)
        .map(|layer| {
            let prefix = layer_weight_path_with_root(root, layer, suffix);
            let weight = gw.take_keep_dtype(&format!("{prefix}.weight"))?;
            let weight_scale = gw.take_keep_dtype(&format!("{prefix}.weight_scale"))?;
            let input_scale = gw.take_keep_dtype(&format!("{prefix}.input_scale")).ok();
            let bias = gw.take_keep_dtype(&format!("{prefix}.bias")).ok();
            Ok(scratchy_layers::layers::Fp8AnyLinear::Std(
                scratchy_layers::layers::Fp8Linear {
                    weight,
                    weight_scale,
                    input_scale,
                    bias,
                    output_dtype,
                },
            ))
        })
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
/// [`LinearLayer::load_dense_concat_packed`]. The spyre pack plans the
/// destination buffer then byte-copies each source tensor in, so it's kept
/// per-target (sibling of metal's direct-write pack).
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

/// MLX-affine int4 layered linear load. Reads the
/// `<root>.<layer>.<suffix>.{weight,scales,biases,bias?}` triple per
/// decoder layer. Produces AffineQuant LinearLayers — the host qmv path
/// is not yet wired, so `load_affine_quant` `bail!`s; the signature exists
/// so every model variant's load body type-checks.
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

/// NVFP4 int4 layered forward-time load. One `LinearLayer::Nvfp4` per decoder
/// layer. Mirrors [`load_layered_linear_affine_quant`]; the host NVFP4 dispatch
/// is not yet wired, so `load_nvfp4_quant` `bail!`s.
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

/// MLX-affine int4 layered dequant-as-dense load. INT4 slow-reference path:
/// CPU-dequantize each per-layer affine triple into a dense Linear. Mirrors
/// [`load_layered_linear_affine_quant`] but the macro emits this (not the
/// AffineQuant variant) so the forward path stays on the dense gemm impls;
/// the host dequant is not yet wired, so `load_affine_dequant_as_dense`
/// `bail!`s.
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
