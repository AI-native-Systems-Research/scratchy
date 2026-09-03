// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral layered-load helpers.
//!
//! Each folds the per-accessor `(0..N).map(|layer| Type::load(gw, &path, …))`
//! loop the codegen used to inline into a single fn call. Only the loaders whose
//! every per-item load is a **neutral** free fn live here (generic over
//! `A: DeviceAllocator`); both `targets/{cuda,metal}` re-export them at their
//! crate root so the forward-compiler macro's emitted `crate::__gpu::load_layered_*`
//! resolves under either backend.
//!
//! Genuinely backend-divergent loaders (the RmsNorm keep-dtype split, the
//! direct-write `load_dense_concat_packed`, cuda's stream/sharded/quant
//! variants, metal's MLX-affine variants) stay in the target crates.

use anyhow::Result;
use scratchy_tensors::DeviceAllocator;
use scratchy_tensors::{layer_weight_path_with_root, vision_block_weight_path};

use crate::layers::{Embedding, LayerNorm, LinearLayer};
use crate::layers::{
    embedding_load, layer_norm_load, linear_layer_load_dense, linear_layer_load_dense_or_ggml,
};
use crate::weights::GpuWeights;

pub fn load_layered_embedding<A: DeviceAllocator>(
    gw: &mut GpuWeights<A>,
    n_layers: u32,
    root: &str,
    suffix: &str,
) -> Result<Vec<Embedding>> {
    (0..n_layers)
        .map(|layer| embedding_load(gw, &layer_weight_path_with_root(root, layer, suffix)))
        .collect()
}

/// Layered LayerNorm load — pulls `<prefix>.weight` AND optional
/// `<prefix>.bias` together. Used by the `MeanSubRmsNormBiasAddImpl`
/// 4-tile fusion (encoder models like ModernBERT). The `bias` field
/// is `Option<GpuTensor>`; the eval path consumes `Some(bias)` when
/// the fusion fires (the matcher only claims the pattern when a
/// `bias_add` tile is downstream of the rmsnorm, so the loader must
/// have produced the bias — see `MeanSubRmsNormBiasAdd` instr).
pub fn load_layered_layer_norm<A: DeviceAllocator>(
    gw: &mut GpuWeights<A>,
    n_layers: u32,
    root: &str,
    suffix: &str,
    eps: f32,
) -> Result<Vec<LayerNorm>> {
    (0..n_layers)
        .map(|layer| layer_norm_load(gw, &layer_weight_path_with_root(root, layer, suffix), eps))
        .collect()
}

/// Vision-tower analogue of [`load_layered_layer_norm`]:
/// `visual.blocks.<L>.<suffix>` per-block prefix. Routed by the
/// codegen when the `LayerNorm`-typed accessor sits on the vision-
/// prelude (Qwen2-VL norm1 / norm2). Returns the same `LayerNorm`
/// wrapper (weight + optional bias + eps) so the
/// `MeanSubRmsNormBiasAddImpl` eval path is shared with the text
/// encoders.
pub fn load_layered_layer_norm_vision<A: DeviceAllocator>(
    gw: &mut GpuWeights<A>,
    n_layers: u32,
    root: &str,
    suffix: &str,
    eps: f32,
) -> Result<Vec<LayerNorm>> {
    (0..n_layers)
        .map(|layer| layer_norm_load(gw, &vision_block_weight_path(root, layer, suffix), eps))
        .collect()
}

/// Vision-tower analogue of [`load_layered_linear_dense`]:
/// `visual.blocks.<L>.<suffix>` per-block prefix instead of
/// `model.layers.<L>.<suffix>`.
pub fn load_layered_linear_dense_vision<A: DeviceAllocator>(
    gw: &mut GpuWeights<A>,
    n_layers: u32,
    root: &str,
    suffix: &str,
) -> Result<Vec<LinearLayer>> {
    (0..n_layers)
        .map(|layer| linear_layer_load_dense(gw, &vision_block_weight_path(root, layer, suffix)))
        .collect()
}

pub fn load_layered_linear_dense<A: DeviceAllocator>(
    gw: &mut GpuWeights<A>,
    n_layers: u32,
    root: &str,
    suffix: &str,
) -> Result<Vec<LinearLayer>> {
    // `load_dense_or_ggml`: tries `take_quantized_linear` first
    // (for `StorageFormat::Ggml` weights), falls back to dense.
    // Transparent on safetensors models since the GGUF map is empty.
    (0..n_layers)
        .map(|layer| {
            linear_layer_load_dense_or_ggml(gw, &layer_weight_path_with_root(root, layer, suffix))
        })
        .collect()
}
