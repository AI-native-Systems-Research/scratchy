// SPDX-License-Identifier: Apache-2.0
//! Spyre layer runtime surface.
//!
//! Re-exports the backend-neutral layer TYPE DEFINITIONS from `scratchy-layers`
//! and the quant payload TYPES from `scratchy-quantizations`, then provides the
//! spyre `*Ops` load traits the `#[forward]`-emitted spyre `load` body calls as
//! `crate::__gpu::layers::…`.
//!
//! Spyre is a HOST backend (fp16) — loading a weight is "read the bytes out of
//! `GpuWeights<SpyreAllocator>` and construct the neutral struct". Every dense
//! load is therefore allocator-generic: it delegates to the shared neutral free
//! fns in `scratchy_layers::layers` (the same ones cuda/metal share) or builds
//! the packed buffer through the allocator-generic `take_cpu` /
//! `alloc_packed_from_host` host pipeline — no GPU kernels, no stream.
//!
//! The quant constructors (MLX-affine int4, NVFP4) are kernel-bound and cannot
//! be done allocator-generically yet, so they `anyhow::bail!` with a clear
//! "not yet implemented" runtime error. A bf16/fp16 checkpoint never calls them,
//! but the macro emits the call site for every model variant so they must
//! type-check — hence `bail!` (recoverable), never `unimplemented!()`.

use anyhow::Result;
use scratchy_tensors::{DType, LoadStream};

use crate::weights::GpuWeights;

// Neutral layer TYPE DEFINITIONS + the shared neutral load free fns. The spyre
// `crate::__gpu::layers::*` field-type surface (LinearLayer, Embedding, RmsNorm,
// LayerNorm, GatedDeltaNetLayer, GpuTensor handles, …) resolves through this
// glob — it formerly lived inline in `lib.rs`, moved here so the whole layer
// surface is in one place, peer to targets/metal's `layers.rs`.
pub use scratchy_layers::layers::*;

// The quant payload TYPES live in `scratchy-quantizations` (a neutral leaf,
// NOT a `targets/` crate). Re-export them under `crate::__gpu::layers::…` so the
// macro-emitted `AffineQuant*` / `Nvfp4Linear` constructor + field paths keep
// resolving against the same type the cuda/metal arms use.
pub use scratchy_quantizations::{AffineQuantEmbedding, AffineQuantLinear, Nvfp4Linear};

// ---------------------------------------------------------------------------
// LinearLayer — dense load Ops (real) + quant constructors (bail)
// ---------------------------------------------------------------------------

/// Spyre load surface for [`LinearLayer`]. Mirrors the cuda `LinearLayerOps`
/// trait NAMES + SIGNATURES exactly (the macro calls these by path-syntax), but
/// the stream parameter is the host `LoadStream` (= `()`) rather than a
/// `CUstream`. All dense loads are allocator-generic host reads; the quant
/// constructors are `bail!` stubs (kernel-bound, not yet implemented).
pub trait LinearLayerOps: Sized {
    fn load_dense(weights: &mut GpuWeights, prefix: &str) -> Result<Self>;
    fn load_raw(weights: &mut GpuWeights, key: &str) -> Result<Self>;
    fn load_dense_or_ggml(weights: &mut GpuWeights, prefix: &str) -> Result<Self>;
    fn load_dense_concat_or_ggml(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        stream: LoadStream,
    ) -> Result<Self>;
    fn load_dense_sharded(
        weights: &mut GpuWeights,
        prefix: &str,
        dim: usize,
        rank: usize,
        world: usize,
    ) -> Result<Self>;
    fn load_dense_concat_packed(weights: &mut GpuWeights, prefixes: &[&str]) -> Result<Self>;
    fn load_dense_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        stream: LoadStream,
    ) -> Result<Self>;
    fn load_dense_concat_sharded(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        stream: LoadStream,
        rank: usize,
        world: usize,
    ) -> Result<Self>;

    // Quant constructors — kernel-bound, not yet implemented on spyre. A
    // bf16/fp16 checkpoint never reaches these, but the macro emits the call
    // site for every model variant so they must type-check.
    fn load_affine_quant(
        weights: &mut GpuWeights,
        prefix: &str,
        group_size: u32,
        bits: u32,
        expected_in_features: u32,
    ) -> Result<Self>;
    fn load_affine_dequant_as_dense(
        weights: &mut GpuWeights,
        prefix: &str,
        group_size: u32,
        bits: u32,
    ) -> Result<Self>;
    fn load_affine_dequant_concat_as_dense(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        group_size: u32,
        bits: u32,
        expected_in_features: u32,
    ) -> Result<Self>;
    fn load_nvfp4_quant(weights: &mut GpuWeights, prefix: &str, group_size: u32) -> Result<Self>;
    fn load_nvfp4_quant_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        group_size: u32,
    ) -> Result<Self>;
}

impl LinearLayerOps for LinearLayer {
    /// Load a single dense bf16/fp16 linear by safetensors prefix
    /// (`<prefix>.weight` + optional `<prefix>.bias`).
    fn load_dense(weights: &mut GpuWeights, prefix: &str) -> Result<Self> {
        scratchy_layers::layers::linear_layer_load_dense(weights, prefix)
    }

    /// Load an `nn.Parameter`-style weight by its verbatim safetensors key
    /// (no `.weight`/`.bias` suffix). Transposes the matmul-natural `[K, N]`
    /// on-disk layout to the gemm `[N, K]` convention. No bias.
    fn load_raw(weights: &mut GpuWeights, key: &str) -> Result<Self> {
        scratchy_layers::layers::linear_layer_load_raw(weights, key)
    }

    /// Load a linear that may be dense (safetensors) or GGUF-quantized — tries
    /// the quantized map first, falls back to dense. Harmlessly equivalent to
    /// `load_dense` on safetensors models.
    fn load_dense_or_ggml(weights: &mut GpuWeights, prefix: &str) -> Result<Self> {
        scratchy_layers::layers::linear_layer_load_dense_or_ggml(weights, prefix)
    }

    /// Concat-load variant of `load_dense_or_ggml`. Spyre runs the dense
    /// host-concat path; GGUF byte-packing is out of scope for the fp16 host
    /// backend, so this delegates straight to the dense concat (which errors
    /// cleanly if a source is quantized).
    fn load_dense_concat_or_ggml(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        stream: LoadStream,
    ) -> Result<Self> {
        Self::load_dense_concat(weights, prefixes, stream)
    }

    /// Tensor-parallel sharded dense load. `dim == 0` is column-parallel
    /// (q/k/v, gate/up, lm_head, embed): the weight + bias slice along dim 0.
    /// `dim == 1` is row-parallel (o/down): the weight slices along dim 1 and
    /// the bias stays full-size on rank 0 only. `world == 1` degrades to the
    /// unsharded `load_dense` behavior.
    fn load_dense_sharded(
        weights: &mut GpuWeights,
        prefix: &str,
        dim: usize,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        assert!(dim < 2, "load_dense_sharded: dim must be 0 or 1, got {dim}");
        if world == 1 {
            return Self::load_dense(weights, prefix);
        }
        let weight_name = format!("{prefix}.weight");
        let bias_name = format!("{prefix}.bias");
        let weight = weights.take_shard(&weight_name, dim, rank, world)?;
        let bias = if weights.contains(&bias_name) {
            if dim == 0 {
                // Column-parallel: bias shards along dim 0 too.
                Some(weights.take_shard(&bias_name, 0, rank, world)?)
            } else if rank == 0 {
                // Row-parallel: bias is replicated full-size on rank 0 only,
                // added once after the cross-rank reduce.
                Some(weights.take(&bias_name)?)
            } else {
                None
            }
        } else {
            None
        };
        Ok(Self::Dense(Linear::new(weight, bias)))
    }

    /// Load several dense linears and concat along the out-feature dim (dim 0)
    /// into one packed `LinearLayer::Dense`, via the allocator-generic host
    /// pipeline: CPU-concat each source's bytes, then one `alloc_packed_from_host`.
    /// Biases are all-or-none across the source set.
    fn load_dense_concat_packed(weights: &mut GpuWeights, prefixes: &[&str]) -> Result<Self> {
        let (total_out, hidden, dtype, total_bytes) =
            scratchy_layers::layers::concat_packed_plan(weights, prefixes)?;
        let mut packed: Vec<u8> = Vec::with_capacity(total_bytes);
        for p in prefixes {
            let weight_name = format!("{p}.weight");
            let (data, _shape, dt) = weights.take_cpu(&weight_name)?;
            anyhow::ensure!(
                dt == dtype,
                "load_dense_concat_packed: `{}` cpu dtype drift {:?} vs {:?}",
                weight_name,
                dt,
                dtype,
            );
            packed.extend_from_slice(&data);
        }
        let packed_weight = weights.alloc_packed_from_host(&packed, &[total_out, hidden], dtype)?;
        let packed_bias = scratchy_layers::layers::concat_packed_bias(weights, prefixes, dtype)?;
        Ok(Self::Dense(Linear::new(packed_weight, packed_bias)))
    }

    /// Stream-free host concat. Spyre has no async DMA — the `stream` arg is
    /// `()` and ignored. Identical packed `[sum(out), hidden]` layout the cuda
    /// stream path produces, just built through one CPU memcpy per source.
    fn load_dense_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        _stream: LoadStream,
    ) -> Result<Self> {
        Self::load_dense_concat_packed(weights, prefixes)
    }

    /// Tensor-parallel sharded variant of `load_dense_concat`. Fused QKV /
    /// gate_up are always column-parallel (`ShardDim0`): each source slices
    /// along dim 0 to `[out_i / world, in]` then concatenates. Biases follow
    /// the same column-parallel rule. `world == 1` degrades to the unsharded
    /// concat.
    fn load_dense_concat_sharded(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        _stream: LoadStream,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        if prefixes.is_empty() {
            anyhow::bail!("load_dense_concat_sharded: empty prefix list");
        }
        if world == 1 {
            return Self::load_dense_concat_packed(weights, prefixes);
        }

        // Resolve uniform in_features (hidden) + dtype from CPU metadata, then
        // shard each source along dim 0 and concat its bytes.
        let mut packed: Vec<u8> = Vec::new();
        let mut per_rank_outs: Vec<usize> = Vec::with_capacity(prefixes.len());
        let mut hidden: Option<usize> = None;
        let mut dtype: Option<DType> = None;
        for p in prefixes {
            let weight_name = format!("{p}.weight");
            let (shape, dt) = weights
                .tensor_info(&weight_name)
                .ok_or_else(|| anyhow::anyhow!("weight not found: {weight_name}"))?;
            anyhow::ensure!(
                shape.len() == 2,
                "load_dense_concat_sharded: `{weight_name}` has rank {}, expected 2",
                shape.len(),
            );
            let this_hidden = shape[1];
            match hidden {
                None => hidden = Some(this_hidden),
                Some(h) => anyhow::ensure!(
                    h == this_hidden,
                    "load_dense_concat_sharded: `{}` in_features {} != {}",
                    p,
                    this_hidden,
                    h,
                ),
            }
            match dtype {
                None => dtype = Some(dt),
                Some(d) => anyhow::ensure!(
                    d == dt,
                    "load_dense_concat_sharded: `{}` dtype {:?} != {:?}",
                    p,
                    dt,
                    d,
                ),
            }
        }
        let dtype = dtype.expect("non-empty prefixes");
        let hidden = hidden.expect("non-empty prefixes");

        for p in prefixes {
            let weight_name = format!("{p}.weight");
            let shard = weights.take_shard(&weight_name, 0, rank, world)?;
            per_rank_outs.push(shard.dim(0));
            // SAFETY: dim-0 shard is a contiguous row range; copy its bytes
            // into the packed host buffer, then drop the shard alloc.
            let shard_bytes = shard.dim(0) * hidden * dtype.size_bytes();
            let slice = unsafe { std::slice::from_raw_parts(shard.raw_ptr(), shard_bytes) };
            packed.extend_from_slice(slice);
        }
        let total_out: usize = per_rank_outs.iter().sum();
        let packed_weight = weights.alloc_packed_from_host(&packed, &[total_out, hidden], dtype)?;

        // Biases: column-parallel concat shards each bias along dim 0. All-or-none.
        let bias_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.bias")).collect();
        let any_bias = bias_names.iter().any(|n| weights.contains(n));
        let all_bias = bias_names.iter().all(|n| weights.contains(n));
        anyhow::ensure!(
            !any_bias || all_bias,
            "load_dense_concat_sharded: inconsistent biases across prefixes {:?}",
            prefixes,
        );
        let packed_bias = if all_bias {
            let mut bias_bytes: Vec<u8> = Vec::new();
            let mut total_bias_elems: usize = 0;
            for bn in &bias_names {
                let shard = weights.take_shard(bn, 0, rank, world)?;
                let elems = shard.dim(0);
                let nbytes = elems * dtype.size_bytes();
                let slice = unsafe { std::slice::from_raw_parts(shard.raw_ptr(), nbytes) };
                bias_bytes.extend_from_slice(slice);
                total_bias_elems += elems;
            }
            Some(weights.alloc_packed_from_host(&bias_bytes, &[total_bias_elems], dtype)?)
        } else {
            None
        };

        Ok(Self::Dense(Linear::new(packed_weight, packed_bias)))
    }

    // ── Quant constructors — kernel-bound, not yet implemented on spyre ──

    fn load_affine_quant(
        _weights: &mut GpuWeights,
        _prefix: &str,
        _group_size: u32,
        _bits: u32,
        _expected_in_features: u32,
    ) -> Result<Self> {
        anyhow::bail!("scratchy-target-spyre: load_affine_quant load not yet implemented")
    }

    fn load_affine_dequant_as_dense(
        _weights: &mut GpuWeights,
        _prefix: &str,
        _group_size: u32,
        _bits: u32,
    ) -> Result<Self> {
        anyhow::bail!(
            "scratchy-target-spyre: load_affine_dequant_as_dense load not yet implemented"
        )
    }

    fn load_affine_dequant_concat_as_dense(
        _weights: &mut GpuWeights,
        _prefixes: &[&str],
        _group_size: u32,
        _bits: u32,
        _expected_in_features: u32,
    ) -> Result<Self> {
        anyhow::bail!(
            "scratchy-target-spyre: load_affine_dequant_concat_as_dense load not yet implemented"
        )
    }

    fn load_nvfp4_quant(
        _weights: &mut GpuWeights,
        _prefix: &str,
        _group_size: u32,
    ) -> Result<Self> {
        anyhow::bail!("scratchy-target-spyre: load_nvfp4_quant load not yet implemented")
    }

    fn load_nvfp4_quant_concat(
        _weights: &mut GpuWeights,
        _prefixes: &[&str],
        _group_size: u32,
    ) -> Result<Self> {
        anyhow::bail!("scratchy-target-spyre: load_nvfp4_quant_concat load not yet implemented")
    }
}

// ---------------------------------------------------------------------------
// Embedding — dense load Ops (real) + affine-dequant constructor (bail)
// ---------------------------------------------------------------------------

/// Spyre load surface for [`Embedding`] (dense table load + vocab-parallel
/// shard). `load_affine_dequant` is the kernel-bound quant path — `bail!` for
/// now (a bf16/fp16 checkpoint never reaches it).
pub trait EmbeddingOps: Sized {
    fn load(weights: &mut GpuWeights, prefix: &str) -> Result<Self>;
    fn load_sharded(
        weights: &mut GpuWeights,
        prefix: &str,
        rank: usize,
        world: usize,
    ) -> Result<Self>;
    fn load_affine_dequant(
        weights: &mut GpuWeights,
        prefix: &str,
        group_size: u32,
        bits: u32,
    ) -> Result<Self>;
}

impl EmbeddingOps for Embedding {
    /// Load the dense embedding table by prefix (`<prefix>.weight`).
    fn load(weights: &mut GpuWeights, prefix: &str) -> Result<Self> {
        scratchy_layers::layers::embedding_load(weights, prefix)
    }

    /// Vocab-parallel sharded load — slices the table along dim 0 (`vocab_size`)
    /// so each rank holds `[vocab_size / world, hidden_size]`. `world == 1`
    /// degrades to the same shape `load` produces.
    fn load_sharded(
        weights: &mut GpuWeights,
        prefix: &str,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        if world == 1 {
            return Self::load(weights, prefix);
        }
        let weight_name = format!("{prefix}.weight");
        let weight = weights.take_shard(&weight_name, 0, rank, world)?;
        Ok(Self::new(weight))
    }

    fn load_affine_dequant(
        _weights: &mut GpuWeights,
        _prefix: &str,
        _group_size: u32,
        _bits: u32,
    ) -> Result<Self> {
        anyhow::bail!("scratchy-target-spyre: load_affine_dequant load not yet implemented")
    }
}

// ---------------------------------------------------------------------------
// RmsNorm — load (keep on-disk dtype, mirroring metal)
// ---------------------------------------------------------------------------

/// Spyre load surface for [`RmsNorm`].
pub trait RmsNormOps: Sized {
    fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self>;
}

impl RmsNormOps for RmsNorm {
    /// Route through `take_keep_dtype` so the on-disk gain dtype (F16 on every
    /// sampled mlx-community / Llama-3.x checkpoint) reaches the host verbatim
    /// — spyre is fp16 like metal, so it mirrors metal's keep-dtype choice
    /// rather than cuda's load-time bf16 cast. Falls back to the bare `<prefix>`
    /// key (Gemma's per-layer scalar gain ships with no `.weight` suffix).
    fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        let weight = weights
            .take_keep_dtype(&weight_name)
            .or_else(|_| weights.take_keep_dtype(prefix))?;
        Ok(RmsNorm::new(weight, eps))
    }
}

// ---------------------------------------------------------------------------
// LayerNorm — load
// ---------------------------------------------------------------------------

/// Spyre load surface for [`LayerNorm`].
pub trait LayerNormOps: Sized {
    fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self>;
}

impl LayerNormOps for LayerNorm {
    /// Load `<prefix>.weight` + optional `<prefix>.bias` as 1D tensors.
    fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self> {
        scratchy_layers::layers::layer_norm_load(weights, prefix, eps)
    }
}

// ---------------------------------------------------------------------------
// GatedDeltaNetLayer — load
// ---------------------------------------------------------------------------

/// Spyre load surface for [`GatedDeltaNetLayer`].
pub trait GatedDeltaNetOps: Sized {
    fn load(weights: &mut GpuWeights, prefix: &str) -> Result<Self>;
}

impl GatedDeltaNetOps for GatedDeltaNetLayer {
    /// Load the linear-attn bundle. Tensors keep their on-disk dtype except
    /// `norm.weight` (upcast to f32) — same neutral logic the cuda/metal arms
    /// share.
    fn load(weights: &mut GpuWeights, prefix: &str) -> Result<Self> {
        scratchy_layers::layers::gated_delta_net_load(weights, prefix)
    }
}
