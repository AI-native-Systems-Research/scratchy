// SPDX-License-Identifier: Apache-2.0
//! Metal layer runtime surface.
//!
//! Re-exports the backend-neutral layer TYPE DEFINITIONS from `scratchy-layers`
//! and provides the metal `*Ops` load traits the `#[forward]`-emitted metal
//! `load` body calls as `crate::__gpu::layers::…`. The neutral load logic is
//! shared from `scratchy_layers::layers` (both backends delegate to it); only
//! the `RmsNorm` keep-dtype load and the direct-write `load_dense_concat_packed`
//! are metal-specific. Quant load Ops live in `crate::layers_quant`, re-exported
//! here so the macro reaches them under the `layers::` path too.

use anyhow::Result;
use scratchy_tensors::GpuTensor;

use crate::weights::GpuWeights;
use crate::weights_metal::MetalWeightsExt;

// Neutral layer types + the shared neutral load free fns.
pub use scratchy_layers::layers::*;
// Quant layer types the metal macro arm names.
pub use scratchy_quantizations::layers_quant::{
    AffineQuantEmbedding, AffineQuantLinear, Nvfp4Linear,
};
// Metal quant load Ops (the macro imports these from `crate::__gpu::layers::…`).
pub use crate::layers_quant::{
    MetalAffineEmbedOps, MetalAffineQuantOps, MetalEmbeddingOps, MetalLinearLayerOps, MetalNvfp4Ops,
};

/// Metal dense-load surface for [`LinearLayer`]. Neutral loads delegate to the
/// shared `scratchy_layers::layers` fns; `load_dense_concat_packed` uses the
/// metal direct-write pack — pre-allocate the destination `MTLBuffer` and
/// stream each source tensor's bytes straight in, eliding the mmap → heap Vec →
/// buffer double memcpy (the dominant chunk of metal init engine time).
pub trait LinearLayerOps: Sized {
    fn load_dense(weights: &mut GpuWeights, prefix: &str) -> Result<Self>;
    fn load_raw(weights: &mut GpuWeights, key: &str) -> Result<Self>;
    fn load_dense_or_ggml(weights: &mut GpuWeights, prefix: &str) -> Result<Self>;
    fn load_dense_concat_packed(weights: &mut GpuWeights, prefixes: &[&str]) -> Result<Self>;
}

impl LinearLayerOps for LinearLayer {
    fn load_dense(weights: &mut GpuWeights, prefix: &str) -> Result<Self> {
        scratchy_layers::layers::linear_layer_load_dense(weights, prefix)
    }

    fn load_raw(weights: &mut GpuWeights, key: &str) -> Result<Self> {
        scratchy_layers::layers::linear_layer_load_raw(weights, key)
    }

    fn load_dense_or_ggml(weights: &mut GpuWeights, prefix: &str) -> Result<Self> {
        scratchy_layers::layers::linear_layer_load_dense_or_ggml(weights, prefix)
    }

    fn load_dense_concat_packed(weights: &mut GpuWeights, prefixes: &[&str]) -> Result<Self> {
        let (total_out, hidden, dtype, total_bytes) =
            scratchy_layers::layers::concat_packed_plan(weights, prefixes)?;
        let dst = weights
            .metal_allocator()
            .alloc_uninit(total_bytes)
            .map_err(|e| anyhow::anyhow!("alloc_uninit({total_bytes}) failed: {e}"))?;
        let mut offset = 0usize;
        for p in prefixes {
            let weight_name = format!("{p}.weight");
            let (written, _shape, dt) =
                unsafe { weights.take_into_metal(&weight_name, dst.add(offset))? };
            anyhow::ensure!(
                dt == dtype,
                "load_dense_concat_packed: `{}` cpu dtype drift {:?} vs {:?}",
                weight_name,
                dt,
                dtype,
            );
            offset += written;
        }
        anyhow::ensure!(
            offset == total_bytes,
            "load_dense_concat_packed: wrote {offset} bytes, expected {total_bytes}",
        );
        let packed_weight = unsafe { GpuTensor::new(dst, &[total_out, hidden], dtype) };
        let packed_bias = scratchy_layers::layers::concat_packed_bias(weights, prefixes, dtype)?;
        Ok(LinearLayer::Dense(Linear::new(packed_weight, packed_bias)))
    }
}

/// Metal load surface for [`Embedding`] (dense table load).
pub trait EmbeddingOps: Sized {
    fn load(weights: &mut GpuWeights, prefix: &str) -> Result<Self>;
}
impl EmbeddingOps for Embedding {
    fn load(weights: &mut GpuWeights, prefix: &str) -> Result<Self> {
        scratchy_layers::layers::embedding_load(weights, prefix)
    }
}

/// Metal load surface for [`RmsNorm`].
pub trait RmsNormOps: Sized {
    fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self>;
}
impl RmsNormOps for RmsNorm {
    /// Metal: route through `take_keep_dtype` so the on-disk gain dtype (F16 on
    /// every sampled mlx-community / Llama-3.x checkpoint) reaches the device
    /// verbatim — the `<T_act, T_scale>` rmsnorm kernel casts the gain to the
    /// activation dtype in registers (vs cuda's load-time cast to bf16).
    fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        // The `_s_<scale>_` rmsnorm kernel binds the gain through a
        // `device const T_scale*` whose dtype is fixed at compile time from
        // `W::SCALE_DTYPE` — NOT the on-disk dtype. mlx-community 4bit Llama
        // ships F16 gains (matches the F16 default), but a standard HF bf16
        // checkpoint (e.g. unsloth/Llama-3.2-1B-Instruct) ships BF16 gains;
        // reading those as F16 mis-scales every gain → garbage output. When
        // the model pinned a scale dtype, `take_as_dtype` converts only on a
        // genuine mismatch (the mlx zero-copy path is untouched). cuda/spyre
        // never pin it — they cast the gain at load themselves.
        let weight = match weights.rmsnorm_scale_dtype() {
            Some(dt) => weights
                .take_as_dtype(&weight_name, dt)
                .or_else(|_| weights.take_as_dtype(prefix, dt))?,
            None => weights
                .take_keep_dtype(&weight_name)
                .or_else(|_| weights.take_keep_dtype(prefix))?,
        };
        Ok(RmsNorm::new(weight, eps))
    }
}

/// Metal load surface for [`LayerNorm`].
pub trait LayerNormOps: Sized {
    fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self>;
}
impl LayerNormOps for LayerNorm {
    fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self> {
        scratchy_layers::layers::layer_norm_load(weights, prefix, eps)
    }
}

/// Metal load surface for [`GatedDeltaNetLayer`].
pub trait GatedDeltaNetOps: Sized {
    fn load(weights: &mut GpuWeights, prefix: &str) -> Result<Self>;
}
impl GatedDeltaNetOps for GatedDeltaNetLayer {
    fn load(weights: &mut GpuWeights, prefix: &str) -> Result<Self> {
        scratchy_layers::layers::gated_delta_net_load(weights, prefix)
    }
}
