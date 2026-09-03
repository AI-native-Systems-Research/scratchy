//! Layered-load helper subroutines that fold each per-accessor
//! `let <base>: Vec<T> = (0..N).map(|layer| Type::load(gw,
//! &layer_weight_path_with_root(root, layer, suffix), …)).collect::<Result<Vec<_>>>()?;`
//! block emitted by `emit_layered_load_body` into a single fn-call
//! at the call site.
//!
//! Each accessor used to expand to ~4 lines of source plus the
//! per-iteration `format!`/`load`/return path. Routing through a
//! one-line helper collapses every per-canonical accessor to a
//! single line of expanded source. With ~9 layered accessors per
//! canonical and 57 canonicals in the llama crate, this trims
//! roughly 6-8k lines off the `load_with` section.
//!
//! The helpers are intentionally narrow — one per `FieldLoad` arm —
//! so the codegen stays a 1:1 mapping rather than re-deriving any
//! load shape at call time.

#[cfg(feature = "cuda")]
use crate::DType;
#[cfg(feature = "cuda")]
use crate::layers::{Bnb4bitLinear, Fp8AnyLinear, Fp8BlockLinear, Fp8Linear, MarlinLinear};
use anyhow::Result;
#[cfg(feature = "cuda")]
use scratchy_tensors::LoadStream;
// `Embedding` is named only by the cuda-only vocab-parallel sharded loader.
#[cfg(feature = "cuda")]
use crate::layers::Embedding;
use crate::layers::{LinearLayer, RmsNorm};
// Runtime extension traits hosting the `load*` methods relocated out of the
// neutral type defs (now in `scratchy-layers`). Both backends call
// `RmsNorm::load` (keep-dtype divergent, kept per-target) + `LinearLayer`'s
// dense/affine/nvfp4/packed loaders; the vocab-parallel `Embedding::load_sharded`
// is cuda-only. The cuda-only quant load traits are imported below.
#[cfg(feature = "cuda")]
use crate::layers::EmbeddingOps;
use crate::layers::{LinearLayerOps, RmsNormOps};
// Quant-load traits — only the `#[cfg(cuda)]` loader fns call these.
#[cfg(feature = "cuda")]
use crate::layers_quant::MarlinFormat;
#[cfg(feature = "cuda")]
use crate::layers_quant::{Bnb4bitLoadOps, Fp8BlockLoadOps, Fp8LoadOps, MarlinLoadOps};
#[cfg(feature = "cuda")]
use crate::tensor::GpuTensor;
use crate::weights::GpuWeights;

// The `model.layers.<L>.<suffix>` / `visual.blocks.<L>.<suffix>` path builders
// are neutral string helpers that stay in the compiler; this crate deps it.
use scratchy_forward_compiler::{layer_weight_path_with_root, vision_block_weight_path};

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

/// Vocab-parallel layered Embedding load. Mirrors
/// [`load_layered_embedding`] but slices each layer's embedding
/// table along dim 0 (`vocab_size`) per `(rank, world)`. No
/// layered Embedding accessor exists in any current arch — this
/// helper is here for symmetry with the other `_sharded` helpers
/// and so the codegen macro can route every layered FieldLoad
/// through a `_sharded` variant uniformly. Per Python vLLM's
/// `VocabParallelEmbedding`.
#[cfg(feature = "cuda")]
pub fn load_layered_embedding_sharded(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffix: &str,
    rank: usize,
    world: usize,
) -> Result<Vec<Embedding>> {
    (0..n_layers)
        .map(|layer| {
            Embedding::load_sharded(
                gw,
                &layer_weight_path_with_root(root, layer, suffix),
                rank,
                world,
            )
        })
        .collect()
}

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

/// Tensor-parallel layered dense Linear load. See
/// [`scratchy_target_cuda::layers::Linear::load_sharded`] for the per-
/// dim bias semantics. Used by codegen at tp>1: column-parallel
/// (q/k/v/gate/up) → `dim = 0`; row-parallel (o/down) → `dim = 1`.
#[cfg(feature = "cuda")]
pub fn load_layered_linear_dense_sharded(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffix: &str,
    dim: usize,
    rank: usize,
    world: usize,
) -> Result<Vec<LinearLayer>> {
    (0..n_layers)
        .map(|layer| {
            LinearLayer::load_dense_sharded(
                gw,
                &layer_weight_path_with_root(root, layer, suffix),
                dim,
                rank,
                world,
            )
        })
        .collect()
}

#[cfg(feature = "cuda")]
pub fn load_layered_linear_dense_concat(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffixes: &[&str],
    stream: LoadStream,
) -> Result<Vec<LinearLayer>> {
    (0..n_layers)
        .map(|layer| {
            let paths = concat_paths_for_layer(root, layer, suffixes);
            let refs = as_str_refs(&paths);
            LinearLayer::load_dense_concat_or_ggml(gw, &refs, stream)
        })
        .collect()
}

/// Vision-tower analogue of [`load_layered_linear_dense_concat`]:
/// `visual.blocks.<L>.<suffix>` per-block prefix instead of
/// `model.layers.<L>.<suffix>`. Used by the
/// [`crate::FusedGateUpSiluMul`] accessor when the body lives in a
/// `#[vision_forward]` (the SwiGLU MLP fusion concatenates
/// `mlp.gate_proj` + `mlp.up_proj` weights at load time so the fused
/// kernel runs against a single packed `[K, 2*N]` linear).
#[cfg(feature = "cuda")]
pub fn load_layered_linear_dense_concat_vision(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffixes: &[&str],
    stream: LoadStream,
) -> Result<Vec<LinearLayer>> {
    (0..n_layers)
        .map(|layer| {
            let paths: Vec<String> = suffixes
                .iter()
                .map(|s| vision_block_weight_path(root, layer, s))
                .collect();
            let refs = as_str_refs(&paths);
            LinearLayer::load_dense_concat_or_ggml(gw, &refs, stream)
        })
        .collect()
}

/// Tensor-parallel column-parallel concat (no `dim` arg — fused
/// QKV / gate_up are always column-parallel; see
/// [`scratchy_target_cuda::layers::LinearLayer::load_dense_concat_sharded`]).
#[cfg(feature = "cuda")]
pub fn load_layered_linear_dense_concat_sharded(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffixes: &[&str],
    stream: LoadStream,
    rank: usize,
    world: usize,
) -> Result<Vec<LinearLayer>> {
    (0..n_layers)
        .map(|layer| {
            let paths = concat_paths_for_layer(root, layer, suffixes);
            let refs = as_str_refs(&paths);
            LinearLayer::load_dense_concat_sharded(gw, &refs, stream, rank, world)
        })
        .collect()
}

/// Stream-free counterpart to [`load_layered_linear_dense_concat`] —
/// CPU-concat then one allocator call per layer via
/// [`LinearLayer::load_dense_concat_packed`]. Reachable under either
/// backend; metal callers route here.
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

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn load_layered_marlin_linear(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffix: &str,
    storage: MarlinFormat,
    workspace: GpuTensor,
    device_id: i32,
) -> Result<Vec<MarlinLinear>> {
    (0..n_layers)
        .map(|layer| {
            MarlinLinear::load(
                gw,
                &layer_weight_path_with_root(root, layer, suffix),
                storage,
                workspace,
                device_id,
            )
        })
        .collect()
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn load_layered_marlin_linear_concat(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffixes: &[&str],
    storage: MarlinFormat,
    workspace: GpuTensor,
    device_id: i32,
) -> Result<Vec<MarlinLinear>> {
    (0..n_layers)
        .map(|layer| {
            let paths = concat_paths_for_layer(root, layer, suffixes);
            let refs = as_str_refs(&paths);
            MarlinLinear::load_concat(gw, &refs, storage, workspace, device_id)
        })
        .collect()
}

#[cfg(feature = "cuda")]
pub fn load_layered_fp8_linear(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffix: &str,
    output_dtype: DType,
) -> Result<Vec<Fp8AnyLinear>> {
    (0..n_layers)
        .map(|layer| {
            Fp8Linear::load(
                gw,
                &layer_weight_path_with_root(root, layer, suffix),
                output_dtype,
            )
            .map(Fp8AnyLinear::Std)
        })
        .collect()
}

#[cfg(feature = "cuda")]
pub fn load_layered_fp8_linear_concat(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffixes: &[&str],
    output_dtype: DType,
) -> Result<Vec<Fp8AnyLinear>> {
    (0..n_layers)
        .map(|layer| {
            let paths = concat_paths_for_layer(root, layer, suffixes);
            let refs = as_str_refs(&paths);
            Fp8Linear::load_concat(gw, &refs, output_dtype).map(Fp8AnyLinear::Std)
        })
        .collect()
}

#[cfg(feature = "cuda")]
pub fn load_layered_fp8_block_linear(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffix: &str,
    output_dtype: DType,
) -> Result<Vec<Fp8AnyLinear>> {
    (0..n_layers)
        .map(|layer| {
            Fp8BlockLinear::load(
                gw,
                &layer_weight_path_with_root(root, layer, suffix),
                output_dtype,
            )
            .map(Fp8AnyLinear::Block)
        })
        .collect()
}

#[cfg(feature = "cuda")]
pub fn load_layered_fp8_block_linear_concat(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffixes: &[&str],
    output_dtype: DType,
) -> Result<Vec<Fp8AnyLinear>> {
    (0..n_layers)
        .map(|layer| {
            let paths = concat_paths_for_layer(root, layer, suffixes);
            let refs = as_str_refs(&paths);
            Fp8BlockLinear::load_concat(gw, &refs, output_dtype).map(Fp8AnyLinear::Block)
        })
        .collect()
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn load_layered_bnb4(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffix: &str,
    code_gpu: GpuTensor,
    dequant_scratch: GpuTensor,
    out_features: usize,
    in_features: usize,
    blocksize: usize,
) -> Result<Vec<Bnb4bitLinear>> {
    (0..n_layers)
        .map(|layer| {
            Bnb4bitLinear::load(
                gw,
                &layer_weight_path_with_root(root, layer, suffix),
                code_gpu,
                dequant_scratch,
                out_features,
                in_features,
                blocksize,
            )
        })
        .collect()
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn load_layered_bnb4_concat(
    gw: &mut GpuWeights,
    n_layers: u32,
    root: &str,
    suffixes: &[&str],
    code_gpu: GpuTensor,
    dequant_scratch: GpuTensor,
    out_features_per_shard: &[usize],
    in_features: usize,
    blocksize: usize,
) -> Result<Vec<Bnb4bitLinear>> {
    (0..n_layers)
        .map(|layer| {
            let paths = concat_paths_for_layer(root, layer, suffixes);
            let refs = as_str_refs(&paths);
            Bnb4bitLinear::load_concat(
                gw,
                &refs,
                code_gpu,
                dequant_scratch,
                out_features_per_shard,
                in_features,
                blocksize,
            )
        })
        .collect()
}
