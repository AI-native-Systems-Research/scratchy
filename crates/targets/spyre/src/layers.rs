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
use scratchy_tensors::{DType, GpuTensor, LoadStream};

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
        weights: &mut GpuWeights,
        prefix: &str,
        group_size: u32,
        bits: u32,
    ) -> Result<Self> {
        let weight = affine_dequant_b4(weights, prefix, group_size, bits, DType::BF16)?;
        let bias_name = format!("{prefix}.bias");
        let bias = if weights.contains(&bias_name) {
            Some(weights.take(&bias_name)?)
        } else {
            None
        };
        Ok(Self::Dense(Linear::new(weight, bias)))
    }

    fn load_affine_dequant_concat_as_dense(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        group_size: u32,
        bits: u32,
        expected_in_features: u32,
    ) -> Result<Self> {
        let weight = affine_dequant_b4_concat(
            weights,
            prefixes,
            group_size,
            bits,
            DType::BF16,
            expected_in_features,
        )?;
        Ok(Self::Dense(Linear::new(weight, None)))
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

// ── MLX-affine int4 → dense host dequant (the P2 slow-reference path) ───────
//
// The same three helpers metal's `layers_quant.rs` carries, over the SAME
// frozen `WeightSource` seam: spyre's `GpuWeights` implements it (neutral, in
// scratchy-layers), so the reads and the `alloc_packed_from_host` writes are
// allocator-generic with no GPU kernel anywhere. The dequant arithmetic itself
// is the shared neutral `affine_dequant_b4_to_dtype` (scratchy-quantizations).
//
// ⛔ THE FORWARD-TIME KERNEL PATH (`load_affine_quant`) STAYS A `bail!` — this
// is the DEQUANT-AT-LOAD path only, which the macro emits for `mlx-affine-*`
// presets. A dequantized-then-staged weight is fp16 bandwidth, not int4 — the
// slow reference, exactly as on metal.

/// CPU-dequantize one MLX-affine int4 prefix's `{weight,scales,biases}` triple
/// to `(dense [N,K] bytes, n, k)` — the shared validation + math half.
fn affine_dequant_b4_bytes(
    weights: &mut GpuWeights,
    prefix: &str,
    group_size: u32,
    bits: u32,
    dtype_out: DType,
) -> Result<(Vec<u8>, usize, usize)> {
    anyhow::ensure!(
        bits == 4 || bits == 8,
        "affine_dequant_b4: only bits=4 or bits=8 supported, got bits={bits}"
    );
    anyhow::ensure!(
        matches!(dtype_out, DType::F16 | DType::BF16),
        "affine_dequant_b4: dtype_out must be F16 or BF16, got {dtype_out}"
    );

    let (w_bytes, w_shape, w_dtype) = weights.take_cpu(&format!("{prefix}.weight"))?;
    let (s_bytes, s_shape, s_dtype) = weights.take_cpu(&format!("{prefix}.scales"))?;
    let (b_bytes, b_shape, b_dtype) = weights.take_cpu(&format!("{prefix}.biases"))?;

    anyhow::ensure!(
        w_dtype == DType::U32,
        "affine_dequant_b4: `{prefix}.weight` dtype is {w_dtype} (expected U32)",
    );
    // mlx-community 4bit repos ship scales/biases as either F16 (older
    // Llama / Mixtral) or BF16 (newer Qwen3-MoE); the dequant loop decodes
    // the right bit pattern from the actual dtype.
    anyhow::ensure!(
        matches!(s_dtype, DType::F16 | DType::BF16),
        "affine_dequant_b4: `{prefix}.scales` dtype is {s_dtype} (expected F16 or BF16)",
    );
    anyhow::ensure!(
        b_dtype == s_dtype,
        "affine_dequant_b4: `{prefix}.biases` dtype is {b_dtype} (expected same as scales {s_dtype})",
    );
    anyhow::ensure!(
        w_shape.len() == 2,
        "affine_dequant_b4: packed weight shape rank {} (expected 2)",
        w_shape.len(),
    );

    // pack_factor = 32 / bits (8 for bits=4 U32-packed nibbles; 4 for bits=8
    // U32-packed bytes). The packed weight is `[N, K / pack_factor]`.
    let pack_factor = (32 / bits) as usize;
    let n = w_shape[0];
    let k = w_shape[1] * pack_factor;
    anyhow::ensure!(
        k.is_multiple_of(group_size as usize),
        "affine_dequant_b4: K={k} not divisible by group_size={group_size}"
    );
    anyhow::ensure!(
        s_shape == [n, k / group_size as usize],
        "affine_dequant_b4: scales shape {:?} != [{n}, {}]",
        s_shape,
        k / group_size as usize,
    );
    anyhow::ensure!(
        b_shape == s_shape,
        "affine_dequant_b4: biases shape {:?} != scales shape {:?}",
        b_shape,
        s_shape,
    );

    let out_bytes = scratchy_quantizations::affine_dequant_b4_to_dtype(
        &w_bytes, &s_bytes, &b_bytes, n, k, group_size, bits, s_dtype, dtype_out,
    )?;
    Ok((out_bytes, n, k))
}

/// Dequantize one MLX-affine int4 prefix into a dense `[N, K]` host tensor.
fn affine_dequant_b4(
    weights: &mut GpuWeights,
    prefix: &str,
    group_size: u32,
    bits: u32,
    dtype_out: DType,
) -> Result<GpuTensor> {
    let (out_bytes, n, k) = affine_dequant_b4_bytes(weights, prefix, group_size, bits, dtype_out)?;
    weights.alloc_packed_from_host(&out_bytes, &[n, k], dtype_out)
}

/// Concat sibling of [`affine_dequant_b4`]: CPU-dequantizes each prefix's
/// affine triple, byte-concats the `[N, K]` results along dim 0 into one
/// packed buffer. All sources must share `K`, `group_size`, `bits`, `dtype_out`.
fn affine_dequant_b4_concat(
    weights: &mut GpuWeights,
    prefixes: &[&str],
    group_size: u32,
    bits: u32,
    dtype_out: DType,
    expected_in_features: u32,
) -> Result<GpuTensor> {
    anyhow::ensure!(
        !prefixes.is_empty(),
        "affine_dequant_b4_concat: empty prefix list"
    );
    let elem_size = match dtype_out {
        DType::F16 | DType::BF16 => 2,
        _ => anyhow::bail!(
            "affine_dequant_b4_concat: dtype_out must be F16 or BF16, got {dtype_out}"
        ),
    };

    let mut packed: Vec<u8> = Vec::new();
    let mut total_n: usize = 0;
    let mut k_shared: Option<usize> = None;
    for prefix in prefixes {
        let (bytes, n, k) = affine_dequant_b4_bytes(weights, prefix, group_size, bits, dtype_out)?;
        // Alignment check: reject a checkpoint whose packed `.weight` implies
        // a K that disagrees with the compiled preset's declared in_features
        // (i.e. it was quantized at different bits than this build expects).
        anyhow::ensure!(
            k == expected_in_features as usize,
            "affine dequant checkpoint/preset mismatch at `{prefix}`: packed `.weight` implies \
             in_features={k} at bits={bits} (pack_factor={}); this build expects \
             in_features={expected_in_features}. The checkpoint is quantized at a different \
             bit-width than this build's preset.",
            32 / bits,
        );
        if let Some(prev_k) = k_shared {
            anyhow::ensure!(
                prev_k == k,
                "affine_dequant_b4_concat: in_features mismatch across prefixes \
                 ({prev_k} vs {k} at `{prefix}`)"
            );
        } else {
            k_shared = Some(k);
        }
        total_n += n;
        anyhow::ensure!(
            bytes.len() == n * k * elem_size,
            "affine_dequant_b4_concat: `{prefix}` produced {} bytes; expected {}",
            bytes.len(),
            n * k * elem_size,
        );
        packed.extend_from_slice(&bytes);
    }
    let k = k_shared.expect("affine_dequant_b4_concat: prefixes non-empty above");
    weights.alloc_packed_from_host(&packed, &[total_n, k], dtype_out)
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
        weights: &mut GpuWeights,
        prefix: &str,
        group_size: u32,
        bits: u32,
    ) -> Result<Self> {
        let weight = affine_dequant_b4(weights, prefix, group_size, bits, DType::BF16)?;
        Ok(Self::new(weight))
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
