// SPDX-License-Identifier: Apache-2.0
//! Model layers using `GpuTensor`.
//!
//! These are minimal, inference-only layer types. Weights are stored as
//! `GpuTensor` (raw GPU pointers). Forward passes use cuBLAS GEMM from
//! the `GpuDevice` and fused CUDA kernels.

// The backend-neutral layer type DEFINITIONS + pure accessors live in
// `scratchy-layers`; re-export them here (glob) so the macro-emitted
// `::scratchy_target_cuda::layers::*` paths and the in-crate re-exports keep
// resolving. The kernel-calling `forward` passes and the `GpuWeights`-bound
// `load_*` constructors stay here as per-type extension traits (`*Ops`),
// declared with the same per-method `#[cfg]` they had as inherent methods.
pub use scratchy_layers::*;

// The quant payload TYPES (`AffineQuantLinear` / `Nvfp4Linear` /
// `AffineQuantEmbedding`) live in `scratchy-quantizations` (below the target
// crates). Re-export them under the historic `crate::layers::*` path so the
// macro-emitted `::scratchy_target_cuda::layers::AffineQuant*` /
// `::scratchy_target_cuda::layers::Nvfp4Linear` constructor + field paths keep
// resolving against the same type.
pub use scratchy_quantizations::{AffineQuantEmbedding, AffineQuantLinear, Nvfp4Linear};

// Always-available imports: `GpuTensor` lives in the unconditional
// `crate::tensor` module (pure metadata, no CUDA calls).
use crate::tensor::GpuTensor;
#[cfg(feature = "cuda")]
use crate::weights::CudaWeightsExt;
// The cuda `GpuWeights` alias (`<A = CudaAllocator>`) intentionally shadows the
// generic `GpuWeights` the `pub use scratchy_layers::*` glob above re-exports —
// the `*Ops` loaders below need the concrete cuda allocator. That's the whole
// point of the override, so silence `hidden_glob_reexports`.
#[cfg(feature = "cuda")]
#[allow(hidden_glob_reexports)]
use crate::weights::GpuWeights;
#[cfg(feature = "cuda")]
use anyhow::Result;

#[cfg(feature = "cuda")]
use crate::alloc::{CachingAllocator, OwnedTensor};
#[cfg(feature = "cuda")]
use crate::cublas::CublasHandle;
#[cfg(feature = "cuda")]
use crate::tensor::TensorView;

#[cfg(feature = "nccl")]
use crate::nccl::NcclGroup;
#[cfg(feature = "nccl")]
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Packed-source fallback (Phi-3 family): on a missing `.weight`, detect
// the conventional packed parent (`qkv_proj` for q/k/v; `gate_up_proj` for
// gate/up) and ask `GpuWeights` to synthesize the per-slice virtual
// entries before the take. Idempotent — if the sibling has already been
// synthesized (by a prior call for q_proj, say), the packed parent is
// already gone and this is a no-op.
// ---------------------------------------------------------------------------

/// Concat-load when every source weight is in the GGUF dense map
/// (FP16/F32 GGUFs land all weights here). `take()` already checks
/// `gguf_dense` first and returns the GpuTensor; we then D2D-copy
/// each [rows, cols] slab into a packed [sum(rows), cols] buffer.
/// Bias follows the same path.
#[cfg(feature = "cuda")]
#[cfg(feature = "cuda")]
fn load_gguf_dense_concat(
    weights: &mut GpuWeights,
    prefixes: &[&str],
    stream: cudarc::driver::sys::CUstream,
) -> Result<LinearLayer> {
    debug_assert!(!prefixes.is_empty());
    let mut parts: Vec<GpuTensor> = Vec::with_capacity(prefixes.len());
    for p in prefixes {
        let name = format!("{p}.weight");
        let t = weights.take(&name)?;
        if t.ndim() != 2 {
            anyhow::bail!(
                "load_gguf_dense_concat: `{name}` has rank {}, expected 2",
                t.ndim()
            );
        }
        parts.push(t);
    }
    let cols = parts[0].dim(1);
    let dtype = parts[0].dtype();
    for (i, t) in parts.iter().enumerate() {
        if t.dim(1) != cols {
            anyhow::bail!(
                "load_gguf_dense_concat: `{}` in_features {} != {cols}",
                prefixes[i],
                t.dim(1)
            );
        }
        if t.dtype() != dtype {
            anyhow::bail!(
                "load_gguf_dense_concat: `{}` dtype {:?} != {dtype:?}",
                prefixes[i],
                t.dtype()
            );
        }
    }
    let total_rows: usize = parts.iter().map(|t| t.dim(0)).sum();
    let elem = dtype.size_bytes();
    let total_bytes = total_rows * cols * elem;
    let dst = unsafe { crate::driver::mem_alloc(total_bytes)? };
    weights.record_alloc(dst, total_bytes);
    let mut row_off: usize = 0;
    for t in &parts {
        let part_rows = t.dim(0);
        let part_bytes = part_rows * cols * elem;
        unsafe {
            crate::driver::memcpy_dtod_async(
                (dst as *mut u8).add(row_off * cols * elem),
                t.raw_ptr(),
                part_bytes,
                stream,
            )?;
        }
        row_off += part_rows;
    }
    let weight = unsafe { GpuTensor::new(dst, &[total_rows, cols], dtype) };

    // Bias: either all branches ship one or none.
    let bias_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.bias")).collect();
    let any_bias = weights.contains(&bias_names[0]);
    let bias = if any_bias {
        let mut bias_parts: Vec<GpuTensor> = Vec::with_capacity(prefixes.len());
        for n in &bias_names {
            if !weights.contains(n) {
                anyhow::bail!(
                    "load_gguf_dense_concat: inconsistent bias — `{}` exists but `{n}` is missing",
                    bias_names[0]
                );
            }
            bias_parts.push(weights.take(n)?);
        }
        let bias_dtype = bias_parts[0].dtype();
        let bias_total_rows: usize = bias_parts.iter().map(|t| t.dim(0)).sum();
        let bias_elem = bias_dtype.size_bytes();
        let bias_bytes = bias_total_rows * bias_elem;
        let bdst = unsafe { crate::driver::mem_alloc(bias_bytes)? };
        weights.record_alloc(bdst, bias_bytes);
        let mut bias_off: usize = 0;
        for bt in &bias_parts {
            let n = bt.dim(0);
            unsafe {
                crate::driver::memcpy_dtod_async(
                    (bdst as *mut u8).add(bias_off * bias_elem),
                    bt.raw_ptr(),
                    n * bias_elem,
                    stream,
                )?;
            }
            bias_off += n;
        }
        Some(unsafe { GpuTensor::new(bdst, &[bias_total_rows], bias_dtype) })
    } else {
        None
    };

    Ok(LinearLayer::Dense(Linear { weight, bias }))
}

/// Narrow a replicated full-size 1D bias down to the rank's shard.
///
/// GGUF biases ship 1D and land in `gguf_dense` full-sized on every
/// rank (the shard-kind rule `Replicate`s ndim<2 tensors). For a
/// column-parallel Linear at tp>1, each rank's weight produces
/// `[tokens, out_full/world]` and the bias must match that second dim
/// — otherwise `bias_add_inplace` reads the wrong slice. This returns
/// a view into the same GPU allocation offset by `rank * (out_full/world)`
/// elements.
///
/// `per_rank_out` is the weight's per-rank out-feature count (storage
/// row count for quantized, or `dim(0)` for dense).
#[cfg(feature = "cuda")]
fn per_rank_bias_slice(
    full: crate::tensor::GpuTensor,
    rank: usize,
    world: usize,
    per_rank_out: usize,
) -> crate::tensor::GpuTensor {
    debug_assert_eq!(full.ndim(), 1, "GGUF bias must be 1D");
    debug_assert_eq!(
        full.dim(0),
        per_rank_out * world,
        "bias length must equal per-rank out × world"
    );
    let start = rank * per_rank_out;
    full.narrow_dim0(start, per_rank_out)
}

#[cfg(feature = "cuda")]
fn try_synthesize_packed_slice(weights: &mut GpuWeights, prefix: &str) -> Result<()> {
    scratchy_layers::layers::try_synthesize_packed_slice(weights, prefix)
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

/// Runtime extension methods on the neutral [`Linear`] type (kernel-calling
/// `forward`, `GpuWeights`-bound `load*`). Declared here, with the impl in the
/// same module, so the inherent-impl orphan rule is satisfied; each method
/// keeps the exact `#[cfg]` it had as an inherent method.
#[cfg(feature = "cuda")]
pub trait LinearOps {
    #[cfg(feature = "cuda")]
    fn load(weights: &mut GpuWeights, prefix: &str) -> Result<Self>
    where
        Self: Sized;

    #[cfg(feature = "cuda")]
    fn load_sharded(
        weights: &mut GpuWeights,
        prefix: &str,
        dim: usize,
        rank: usize,
        world: usize,
    ) -> Result<Self>
    where
        Self: Sized;

    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
    ) -> OwnedTensor;
}

#[cfg(feature = "cuda")]
impl LinearOps for Linear {
    /// Load from `GpuWeights` by prefix (e.g. "model.layers.0.self_attn.q_proj").
    ///
    /// Packed-source fallback: some checkpoints (Phi-3 family) ship
    /// `self_attn.qkv_proj.weight` and `mlp.gate_up_proj.weight` in place
    /// of the per-slice `q_proj`/`k_proj`/`v_proj` and `gate_proj`/`up_proj`
    /// tensors the DSL body references. When `{prefix}.weight` is missing,
    /// this function detects the packed-source convention by `prefix`'s
    /// last path segment and asks `GpuWeights` to synthesize virtual
    /// per-slice entries (even row-wise split) before the take. The
    /// split is recoverable — later sibling calls reuse the synthesized
    /// entries, so the packed tensor is walked once.
    #[cfg(feature = "cuda")]
    fn load(weights: &mut GpuWeights, prefix: &str) -> Result<Self> {
        scratchy_layers::layers::linear_load(weights, prefix)
    }

    /// Load a tensor-parallel slice of `{prefix}.weight` (and bias)
    /// per Python vLLM's `ColumnParallelLinear` (`dim = 0`,
    /// column-parallel) and `RowParallelLinear` (`dim = 1`,
    /// row-parallel) conventions.
    ///
    /// - **`dim == 0` (column-parallel: `q_proj`, `k_proj`, `v_proj`,
    ///   `gate_proj`, `up_proj`, `lm_head`, `embed_tokens`).** The
    ///   weight is sliced along its output dim (dim 0 of the
    ///   `[out, in]` matrix). The bias, if present, is also sliced
    ///   along dim 0 — each rank holds its own contiguous chunk.
    ///   Mirrors Python `ColumnParallelLinear.weight_loader` →
    ///   `loaded_weight.narrow(output_dim=0, ...)`.
    ///
    /// - **`dim == 1` (row-parallel: `o_proj`, `down_proj`).** The
    ///   weight is sliced along its input dim (dim 1 of the
    ///   `[out, in]` matrix). The bias is **replicated full-size on
    ///   rank 0 only**, `None` on other ranks. The forward path adds
    ///   bias before the cross-rank `AllReduce`-sum, which produces
    ///   exactly one bias contribution to the residual stream —
    ///   mirrors Python `RowParallelLinear.forward` line 1543:
    ///   `bias_ = None if (self.tp_rank > 0 ...) else self.bias`.
    ///
    /// `world == 1` is supported and degrades to the unsharded
    /// `Self::load` semantics (bias is loaded on rank 0, which is
    /// the only rank). `dim` must be 0 or 1 — anything else panics.
    #[cfg(feature = "cuda")]
    fn load_sharded(
        weights: &mut GpuWeights,
        prefix: &str,
        dim: usize,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        assert!(
            dim < 2,
            "Linear::load_sharded: dim must be 0 or 1, got {dim}"
        );
        let weight_name = format!("{prefix}.weight");
        let bias_name = format!("{prefix}.bias");

        if !weights.contains(&weight_name) {
            try_synthesize_packed_slice(weights, prefix)?;
        }

        let weight = weights.take_shard(&weight_name, dim, rank, world)?;
        let bias = if weights.contains(&bias_name) {
            if dim == 0 {
                // Column-parallel: bias shards along dim 0 too.
                Some(weights.take_shard(&bias_name, 0, rank, world)?)
            } else {
                // Row-parallel: bias is replicated full-size on rank 0,
                // None on other ranks. The bias entry stays unconsumed
                // in `weights` on rank > 0 — its mmap page is freed
                // when `GpuWeights` drops.
                if rank == 0 {
                    Some(weights.take(&bias_name)?)
                } else {
                    None
                }
            }
        } else {
            None
        };
        Ok(Self::new(weight, bias))
    }

    /// Forward: y = x @ W^T (+ bias)
    ///
    /// `x`: `[num_tokens, in_features]`
    /// Returns: `[num_tokens, out_features]` as OwnedTensor from caching allocator.
    ///
    /// # Safety
    /// All tensors must be valid GPU memory. cuBLAS handle must be on the correct stream.
    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
    ) -> OwnedTensor {
        debug_assert_eq!(x.ndim(), 2);
        debug_assert_eq!(x.dim(1), self.weight.dim(1), "Linear: input dim mismatch");

        if let Some(bias) = self.bias {
            cublas.gemm_bias(*x, self.weight, bias, alloc)
        } else {
            cublas.gemm(*x, self.weight, alloc)
        }
    }
}

// ---------------------------------------------------------------------------
// MarlinLinear (INT4 quantized via Marlin GEMM) — forward
// ---------------------------------------------------------------------------

/// Forward-only extension trait on the neutral [`MarlinLinear`].
#[cfg(feature = "cuda")]
pub trait MarlinOps {
    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        x: TensorView<'_>,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor;
}

#[cfg(feature = "cuda")]
impl MarlinOps for MarlinLinear {
    /// Forward: y = marlin_gemm(x, qweight, scales, zeros)
    ///
    /// `x`: `[num_tokens, size_k]` (F16 or BF16)
    /// Returns: `[num_tokens, size_n]`
    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        x: TensorView<'_>,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        let size_m = x.dim(0);
        debug_assert_eq!(
            x.dim(1),
            self.size_k,
            "MarlinLinear: input dim {} != size_k {}",
            x.dim(1),
            self.size_k
        );
        // Don't pass bias to Marlin kernel (would need permutation).
        // Instead, add bias after the GEMM with a simple broadcast add.
        let out = crate::kernels::marlin_gemm(
            *x,
            self.qweight,
            self.scales,
            self.zeros,
            self.g_idx,
            self.g_idx_sort_indices,
            None,
            self.workspace,
            size_m,
            self.size_n,
            self.size_k,
            self.num_groups,
            self.group_size,
            self.has_act_order,
            self.has_zp,
            self.b_type_id,
            self.device_id,
            alloc,
            stream,
        );

        if let Some(bias) = self.bias {
            crate::kernels::bias_add_inplace(out.as_gpu_tensor(), bias, stream);
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Bnb4bitLinear (BitsAndBytes NF4/FP4 4-bit quantized) — forward
// ---------------------------------------------------------------------------

/// Forward-only extension trait on the neutral [`Bnb4bitLinear`].
#[cfg(feature = "cuda")]
pub trait Bnb4bitOps {
    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor;
}

#[cfg(feature = "cuda")]
impl Bnb4bitOps for Bnb4bitLinear {
    /// Forward: dequantize → cuBLAS GEMM.
    ///
    /// `x`: `[num_tokens, in_features]`
    /// Returns: `[num_tokens, out_features]`
    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        // BNB stores weights in [out_features, in_features] order (same as original W).
        // Dequant produces the flat W data, reshape as [out, in], then cuBLAS does x @ W^T.
        let weight_view = self
            .dequant_scratch
            .reshape(&[self.out_features, self.in_features]);

        // Dequantize into scratch buffer (reused across layers).
        crate::kernels::dequantize_bnb4bit(
            self.packed_weight,
            self.absmax,
            self.code,
            weight_view,
            self.blocksize,
            stream,
        );

        // cuBLAS GEMM: x @ weight_view^T
        if let Some(bias) = self.bias {
            cublas.gemm_bias(*x, weight_view, bias, alloc)
        } else {
            cublas.gemm(*x, weight_view, alloc)
        }
    }
}

// ---------------------------------------------------------------------------
// GgmlLinear (GGML quantized via llama.cpp kernels) — forward
// ---------------------------------------------------------------------------

/// Forward-only extension trait on the neutral [`GgmlLinear`].
#[cfg(feature = "cuda")]
pub trait GgmlOps {
    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        x: TensorView<'_>,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor;
}

#[cfg(feature = "cuda")]
impl GgmlOps for GgmlLinear {
    /// Forward: y = ggml_matmul(weight, x) + bias
    ///
    /// Activations must be f32 (GGML kernels operate on f32).
    /// Output is f32 `[num_tokens, out_features]`.
    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        x: TensorView<'_>,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        let input_dtype = x.dtype();

        // GGML kernels require f32 activations — cast if needed.
        let cast_buf = if input_dtype != crate::dtype::DType::F32 {
            Some(crate::kernels::cast_logits_to_f32(*x, alloc, stream))
        } else {
            None
        };
        let x_f32 = if let Some(ref cast) = cast_buf {
            cast.as_gpu_tensor()
        } else {
            *x
        };

        let out_f32 = crate::ggml::ggml_matmul(&self.storage, x_f32, alloc, stream);
        drop(cast_buf);

        // ggml_matmul output is F32; bias was cast to model_dtype
        // (typically BF16) at GGUF load time so the dense path can
        // consume it directly. Cast back to F32 here when dtypes
        // disagree — `bias_add_inplace` reinterprets the bias
        // pointer as `out.dtype()` so a BF16 source against F32 out
        // would be silent garbage. The temp F32 buffer must
        // outlive the (async) `bias_add_inplace` kernel: hoist its
        // binding to function scope so the caching allocator can't
        // hand the same pointer to a downstream `cast_from_f32`
        // launch on the same stream.
        let _bias_keepalive = if let Some(bias) = self.bias {
            if bias.dtype() == crate::dtype::DType::F32 {
                crate::kernels::bias_add_inplace(out_f32.as_gpu_tensor(), bias, stream);
                None
            } else {
                let bias_f32 = crate::kernels::cast_logits_to_f32(bias, alloc, stream);
                crate::kernels::bias_add_inplace(
                    out_f32.as_gpu_tensor(),
                    bias_f32.as_gpu_tensor(),
                    stream,
                );
                Some(bias_f32)
            }
        } else {
            None
        };

        // Cast back to original dtype if we converted to f32.
        if input_dtype != crate::dtype::DType::F32 {
            let out_f32_gpu = out_f32.as_gpu_tensor();
            let result = crate::kernels::cast_from_f32(out_f32_gpu, input_dtype, alloc, stream);
            drop(out_f32);
            drop(_bias_keepalive);
            result
        } else {
            drop(_bias_keepalive);
            out_f32
        }
    }
}

// ---------------------------------------------------------------------------
// Fp8Linear (FP8 E4M3 quantized via cublasLt FP8 GEMM) — forward
// ---------------------------------------------------------------------------

/// Forward-only extension trait on the neutral [`Fp8Linear`].
#[cfg(feature = "cuda")]
pub trait Fp8Ops {
    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        x: TensorView<'_>,
        _cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor;
}

#[cfg(feature = "cuda")]
impl Fp8Ops for Fp8Linear {
    /// Forward: quantize activations → CUTLASS FP8 GEMM → output in output_dtype.
    ///
    /// Uses fused CUTLASS `cutlass_scaled_mm` (single kernel launch) with per-row
    /// activation scales and per-tensor weight scale in the epilogue.
    /// Matches Python vLLM's `cutlass_scaled_mm` exactly.
    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        x: TensorView<'_>,
        _cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        debug_assert_eq!(x.ndim(), 2);
        debug_assert_eq!(
            x.dim(1),
            self.weight.dim(1),
            "Fp8Linear: input dim mismatch"
        );

        if let Some(ref input_scale) = self.input_scale {
            // Static activation quantization: scalar input_scale + scalar weight_scale.
            // Quantize with pre-calibrated scale, then fused CUTLASS GEMM.
            let a_scale = input_scale.as_ptr::<f32>();
            let x_fp8 = crate::kernels::scaled_fp8_quant_static(*x, a_scale, alloc, stream);
            // input_scale is [1] (scalar) — CUTLASS handles scalar a_scale correctly.
            let result = if let Some(bias) = self.bias {
                crate::kernels::cutlass_scaled_mm_with_bias(
                    x_fp8.as_gpu_tensor(),
                    self.weight,
                    *input_scale,
                    self.weight_scale,
                    bias,
                    self.output_dtype,
                    alloc,
                    stream,
                )
            } else {
                crate::kernels::cutlass_scaled_mm(
                    x_fp8.as_gpu_tensor(),
                    self.weight,
                    *input_scale,
                    self.weight_scale,
                    self.output_dtype,
                    alloc,
                    stream,
                )
            };
            drop(x_fp8);
            result
        } else {
            // Dynamic per-token activation quantization.
            // Uses CUTLASS cutlass_scaled_mm with fused per-row scale_a epilogue —
            // single kernel launch, exactly matching Python vLLM.
            //
            // 1. Quantize activations: BF16 → FP8 + per-token scales [M]
            // 2. CUTLASS FP8 GEMM with per-token a_scales + per-tensor b_scale
            //    fused into the epilogue. ONE kernel launch.
            let (x_fp8, x_scales) = crate::kernels::scaled_fp8_quant_dynamic(*x, alloc, stream);
            let result = if let Some(bias) = self.bias {
                crate::kernels::cutlass_scaled_mm_with_bias(
                    x_fp8.as_gpu_tensor(),
                    self.weight,
                    x_scales.as_gpu_tensor(),
                    self.weight_scale,
                    bias,
                    self.output_dtype,
                    alloc,
                    stream,
                )
            } else {
                crate::kernels::cutlass_scaled_mm(
                    x_fp8.as_gpu_tensor(),
                    self.weight,
                    x_scales.as_gpu_tensor(),
                    self.weight_scale,
                    self.output_dtype,
                    alloc,
                    stream,
                )
            };
            drop(x_fp8);
            drop(x_scales);
            result
        }
    }
}

// ---------------------------------------------------------------------------
// LinearLayer — runtime extension trait (forward + GpuWeights-bound load_*)
// ---------------------------------------------------------------------------

/// Runtime extension methods on the neutral [`LinearLayer`] enum (the
/// kernel-calling `forward` and the `GpuWeights`-bound `load_*`
/// constructors). The enum definition + its pure accessors live in
/// `scratchy-layers`; each method below keeps the exact `#[cfg]` it had as
/// an inherent method.
#[cfg(feature = "cuda")]
pub trait LinearLayerOps {
    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor;

    #[cfg(feature = "cuda")]
    fn load_dense(weights: &mut GpuWeights, prefix: &str) -> Result<Self>
    where
        Self: Sized;

    #[cfg(feature = "cuda")]
    fn load_raw(weights: &mut GpuWeights, key: &str) -> Result<Self>
    where
        Self: Sized;

    #[cfg(feature = "cuda")]
    fn load_dense_or_ggml(weights: &mut GpuWeights, prefix: &str) -> Result<Self>
    where
        Self: Sized;

    #[cfg(feature = "cuda")]
    fn load_dense_concat_or_ggml(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self>
    where
        Self: Sized;

    #[cfg(feature = "cuda")]
    fn load_dense_sharded(
        weights: &mut GpuWeights,
        prefix: &str,
        dim: usize,
        rank: usize,
        world: usize,
    ) -> Result<Self>
    where
        Self: Sized;

    #[cfg(feature = "cuda")]
    fn load_dense_concat_packed(weights: &mut GpuWeights, prefixes: &[&str]) -> Result<Self>
    where
        Self: Sized;

    #[cfg(feature = "cuda")]
    fn load_dense_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self>
    where
        Self: Sized;

    #[cfg(feature = "cuda")]
    fn load_dense_concat_sharded(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        stream: cudarc::driver::sys::CUstream,
        rank: usize,
        world: usize,
    ) -> Result<Self>
    where
        Self: Sized;
}

#[cfg(feature = "cuda")]
impl LinearLayerOps for LinearLayer {
    /// Forward: y = x @ W^T (dense) or quantized GEMM variant.
    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        match self {
            Self::Dense(l) => l.forward(x, cublas, alloc),
            Self::Marlin(l) => l.forward(x, alloc, stream),
            Self::Ggml(l) => l.forward(x, alloc, stream),
            Self::GgmlConcat(branches) => {
                // Heterogeneous-dtype gate/up packing for Q4_K_M-style
                // mixed quants. Compute each branch separately, then
                // concat along dim 1 into the same `[tokens, sum(out)]`
                // layout the homogeneous packed `Ggml` arm produces —
                // so downstream `silu_and_mul_fused(out, intermediate)`
                // sees identical bytes regardless of which arm
                // produced them.
                debug_assert!(!branches.is_empty(), "GgmlConcat with no branches");
                let num_tokens = x.dim(0);
                let total_out: usize = branches.iter().map(|b| b.out_features()).sum();
                // Output dtype follows the input (the per-branch
                // forwards cast back to the input dtype if needed).
                let out_dtype = x.dtype();
                let packed = alloc.alloc_tensor(&[num_tokens, total_out], out_dtype);
                let row_stride_bytes = total_out * out_dtype.size_bytes();
                let elem = out_dtype.size_bytes();
                let mut col_offset_elems: usize = 0;
                // Hold every per-branch `part` alive until the loop
                // exits so the CachingAllocator can't recycle a
                // dropped branch's GPU buffer for the NEXT branch's
                // matmul allocation while D2D copies from the dropped
                // branch are still queued on the stream — the same
                // use-after-free pattern.
                let mut keepalive: Vec<OwnedTensor> = Vec::with_capacity(branches.len());
                for branch in branches {
                    let part = branch.forward(x, alloc, stream);
                    let part_view = part.as_gpu_tensor();
                    let part_out = branch.out_features();
                    let part_row_bytes = part_out * elem;
                    // Strided per-row D2D copy of `part` columns into
                    // `packed[:, col_offset:col_offset + part_out]`.
                    for row in 0..num_tokens {
                        let src = (part_view.raw_ptr() as *const u8).add(row * part_row_bytes);
                        let dst = (packed.as_mut_ptr() as *mut u8)
                            .add(row * row_stride_bytes + col_offset_elems * elem);
                        crate::driver::memcpy_dtod_async(dst, src, part_row_bytes, stream)
                            .expect("dtod copy in GgmlConcat::forward");
                    }
                    col_offset_elems += part_out;
                    keepalive.push(part);
                }
                drop(keepalive);
                packed
            }
            Self::Bnb4bit(l) => l.forward(x, cublas, alloc, stream),
            Self::Fp8(l) => l.forward(x, cublas, alloc, stream),
            Self::Fp8Block(l) => l.forward(x, cublas, alloc, stream),
            Self::AffineQuant(_) => {
                unreachable!("AffineQuant forward is metal-only; cuda never constructs it")
            }
            Self::Nvfp4(_) => {
                unreachable!("Nvfp4 forward is metal-only; cuda never constructs it")
            }
        }
    }

    /// Load a single dense bf16/fp16 linear layer by safetensors prefix
    /// (e.g. `"model.lm_head"` → reads `"model.lm_head.weight"` and an
    /// optional `".bias"`).
    #[cfg(feature = "cuda")]
    fn load_dense(weights: &mut GpuWeights, prefix: &str) -> Result<Self> {
        scratchy_layers::layers::linear_layer_load_dense(weights, prefix)
    }

    /// Load an `nn.Parameter`-style weight by its verbatim safetensors
    /// key (no `.weight` / `.bias` suffix). Used for matmul-natural
    /// `[K, N]` parameters that ship outside the `nn.Linear`
    /// convention — Gemma3 MM projector's
    /// `multi_modal_projector.mm_input_projection_weight` is the
    /// motivating example. No bias.
    ///
    /// HF stores such params as `[K, N]` (the matmul-natural shape, since
    /// they're consumed via `act @ param`). scratchy's gemm expects
    /// `[N, K]` (PyTorch nn.Linear convention). We transpose at CPU side
    /// before upload so the runtime weight matches the gemm convention.
    /// Shape inference declares the matmul-natural `[K, N]` shape (same
    /// way nn.Linear weights are declared), keeping the DSL surface
    /// uniform.
    #[cfg(feature = "cuda")]
    fn load_raw(weights: &mut GpuWeights, key: &str) -> Result<Self> {
        scratchy_layers::layers::linear_layer_load_raw(weights, key)
    }

    /// Load a linear layer that may be either dense (safetensors) or
    /// GGUF-quantized. Tries `take_quantized_linear` first; falls back
    /// to `load_dense` when the prefix isn't quantized in the
    /// backing `GpuWeights`. Used by codegen for any
    /// `StorageFormat::Ggml` weight — and harmlessly equivalent to
    /// `load_dense` on safetensors models (where
    /// `take_quantized_linear` always returns `None`).
    #[cfg(feature = "cuda")]
    fn load_dense_or_ggml(weights: &mut GpuWeights, prefix: &str) -> Result<Self> {
        scratchy_layers::layers::linear_layer_load_dense_or_ggml(weights, prefix)
    }

    /// Concat-load variant of `load_dense_or_ggml`. If every prefix
    /// is GGUF-quantized in the backing `GpuWeights`, byte-stacks
    /// the per-prefix `GgmlStorage` blobs into one packed
    /// `[sum(out_features), in_features]` quantized tensor and
    /// returns `Self::Ggml`. Otherwise falls back to
    /// `load_dense_concat`.
    ///
    /// The byte-stack works because each row of a row-major
    /// quantized matrix is a whole number of GGML blocks (block
    /// boundaries never straddle rows), so `cat[gate.bytes,
    /// up.bytes]` produces a valid `[gate.nrows + up.nrows, ncols]`
    /// blob with the same dtype. Refuses if dtype or `ncols` differ
    /// across prefixes — neither is expected in practice (GGUF
    /// quantizers ship gate/up with identical layouts) but the
    /// guard prevents silent corruption if an arch ever pairs
    /// different quants.
    #[cfg(feature = "cuda")]
    fn load_dense_concat_or_ggml(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        if prefixes.is_empty() {
            anyhow::bail!("load_dense_concat_or_ggml: empty prefix list");
        }
        // Probe: are ALL prefixes GGUF-quantized? (Either every
        // prefix or none — mixed would mean the GGUF loader took
        // some weights as quantized and others as dense, which it
        // doesn't do today.)
        let weight_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.weight")).collect();
        let all_gguf = weight_names
            .iter()
            .all(|n| weights.contains_quantized_linear(n));

        if !all_gguf {
            // FP16/F32 GGUFs land all weights in `gguf_dense` (none
            // in `quantized`); `load_dense_concat` reads via
            // `tensor_info` + `take_into` which only see the
            // safetensors mmap map. Detect that case and assemble
            // the packed Dense linear via `take()` (which DOES check
            // gguf_dense first) + per-row D2D copy. Bias comes from
            // the same path.
            let all_gguf_dense = weight_names.iter().all(|n| weights.gguf_dense_contains(n));
            if all_gguf_dense {
                return load_gguf_dense_concat(weights, prefixes, stream);
            }
            return Self::load_dense_concat(weights, prefixes, stream);
        }

        // Take all storages.
        let mut storages: Vec<crate::ggml::GgmlStorage> = Vec::with_capacity(prefixes.len());
        for name in &weight_names {
            let s = weights
                .take_quantized_linear(name)
                .ok_or_else(|| anyhow::anyhow!("load_dense_concat_or_ggml: missing {name}"))?;
            storages.push(s);
        }

        // Validate ncols (in_features) — must be uniform across all
        // branches; mismatched dtype is OK and triggers the
        // `GgmlConcat` runtime-concat path below.
        let first = storages[0];
        for s in &storages[1..] {
            if s.ncols != first.ncols {
                anyhow::bail!(
                    "load_dense_concat_or_ggml: ncols mismatch ({} vs {})",
                    first.ncols,
                    s.ncols
                );
            }
        }
        // Bias collection: GGUF biases are dequantized into
        // `gguf_dense` at load time. If every prefix has a sibling
        // `.bias`, collect them; if any has and others don't, hard-
        // error (Qwen2's biased q/k/v is all-or-none). Bias-less
        // GGUFs (Llama gate/up, Llama q/k/v) keep `bias = None` per
        // branch and the fast-path stays the same.
        let bias_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.bias")).collect();
        let any_bias = bias_names.iter().any(|n| weights.contains(n));
        let all_bias = bias_names.iter().all(|n| weights.contains(n));
        if any_bias && !all_bias {
            anyhow::bail!(
                "load_dense_concat_or_ggml: inconsistent biases across prefixes {:?} \
                 — biased Qwen-style q/k/v requires every prefix to ship a `.bias`",
                prefixes
            );
        }
        let mut biases: Vec<Option<GpuTensor>> = vec![None; prefixes.len()];
        if all_bias {
            for (i, n) in bias_names.iter().enumerate() {
                biases[i] = Some(weights.take(n)?);
            }
        }

        // Always use the per-branch `GgmlConcat` path — never allocate
        // a byte-packed copy of the source weights.
        //
        // The per-prefix `GgmlStorage` values share the underlying
        // GPU buffers in `GpuWeights.quantized` (`take_quantized_linear`
        // is non-destructive). Routing the fused load through a
        // byte-packed copy would duplicate those bytes — fatal at
        // commandR-35B scale and fundamentally wrong as a "fix" for
        // any sharing problem. `GgmlConcat::forward` produces the
        // same packed `[num_tokens, sum(out)]` activation the
        // downstream `fused_qkv_rope_cache` / `silu_and_mul` kernels
        // consume; only the activation is materialized, the weights
        // stay in their original quantized buffers.
        //
        // Each branch keeps its own bias (applied inside
        // `GgmlLinear::forward` before `GgmlConcat::forward` packs the
        // outputs), matching the heterogeneous-dtype path's contract.
        let branches: Vec<GgmlLinear> = storages
            .into_iter()
            .zip(biases)
            .map(|(s, bias)| GgmlLinear { storage: s, bias })
            .collect();
        Ok(Self::GgmlConcat(branches))
    }

    /// Tensor-parallel sharded dense linear load. See
    /// [`Linear::load_sharded`] for the full bias/dim semantics —
    /// this is the `LinearLayer` wrapper. Used by codegen at tp>1
    /// when [`crate::scratchy_forward_compiler_macro::tp_lowering::shard_kind_for_weight_path`]
    /// reports the prefix's last segment is column-parallel
    /// (`dim = 0`) or row-parallel (`dim = 1`). Pass `(rank, world)`
    /// from the runtime; `world == 1` short-circuits to the same
    /// behavior as `load_dense` plus shard-kind-aware bias rules.
    #[cfg(feature = "cuda")]
    fn load_dense_sharded(
        weights: &mut GpuWeights,
        prefix: &str,
        dim: usize,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        // GGUF fast-path. The GGUF loader pre-shards quantized linears
        // at file-read time per `gguf_shard_kind_for_hf_name`:
        //   ShardDim0 → rows split, bias is 1D so always `Replicate`.
        //   ShardDim1 → per-row column slice, bias is 1D `Replicate`.
        // Both cases land the quantized storage in `weights.quantized`
        // with per-rank shape and the optional bias in `gguf_dense`
        // with full shape. The safetensors `Linear::load_sharded` path
        // would error with "weight not found" on either — those maps
        // are only consulted by `take_quantized_linear` / `take`.
        let weight_name = format!("{prefix}.weight");
        if let Some(storage) = weights.take_quantized_linear(&weight_name) {
            let bias_name = format!("{prefix}.bias");
            // Bias rules per-dim at tp > 1:
            //   dim=0 (column-parallel): bias shards along dim 0 too.
            //     The GGUF loader replicates 1D tensors (`gguf_shard_kind_for_hf_name`
            //     returns `Replicate` for ndim<2), so each rank currently holds
            //     the full `[out_full]` bias. Narrow it to the per-rank slice
            //     here — matches Python vLLM's `ColumnParallelLinear` weight
            //     loader + `Qwen2`'s biased q/k/v/o at tp>1.
            //   dim=1 (row-parallel): bias is added once per output element,
            //     after the NCCL all-reduce. Python vLLM's `RowParallelLinear`
            //     adds it only on rank 0; adding on every rank would sum
            //     `world * bias`. Keep on rank 0, drop on others.
            let bias = if weights.contains(&bias_name) {
                match dim {
                    1 if rank != 0 => None,
                    0 if world > 1 => weights
                        .take(&bias_name)
                        .ok()
                        .map(|b| per_rank_bias_slice(b, rank, world, storage.nrows)),
                    _ => weights.take(&bias_name).ok(),
                }
            } else {
                None
            };
            return Ok(Self::Ggml(Box::new(GgmlLinear { storage, bias })));
        }
        if let Some(w) = weights.take_gguf_dense(&weight_name) {
            // F16/F32 GGUF: linear weight lives in `gguf_dense`, already
            // per-rank sharded. Same bias rule as the quantized branch.
            let bias_name = format!("{prefix}.bias");
            let bias = if weights.contains(&bias_name) {
                match dim {
                    1 if rank != 0 => None,
                    0 if world > 1 => weights
                        .take(&bias_name)
                        .ok()
                        .map(|b| per_rank_bias_slice(b, rank, world, w.dim(0))),
                    _ => weights.take(&bias_name).ok(),
                }
            } else {
                None
            };
            return Ok(Self::Dense(Linear::new(w, bias)));
        }
        Ok(Self::Dense(Linear::load_sharded(
            weights, prefix, dim, rank, world,
        )?))
    }

    /// Backend-neutral, stream-free counterpart to [`load_dense_concat`].
    ///
    /// Reads each source's CPU bytes via `take_cpu`, concatenates them
    /// along dim 0 into a single packed buffer, and uploads via the
    /// active [`DeviceAllocator`]. No precast pipeline / async DMA —
    /// callers pay one CPU memcpy per prefix in exchange for not
    /// needing a `CUstream`. Acceptable on the load-once path; metal
    /// has no DMA either way (`StorageModeShared` is unified memory).
    ///
    /// All sources must share `in_features` (dim 1) and dtype. If any
    /// source has a bias, every source must — biases concat in the
    /// same order. Refuses any source with quantized storage (callers
    /// route GGUF through `load_dense_concat_or_ggml` under cuda; metal
    /// never sees quantized variants per the macro's quant-skip).
    ///
    /// [`load_dense_concat`]: Self::load_dense_concat
    #[cfg(feature = "cuda")]
    fn load_dense_concat_packed(weights: &mut GpuWeights, prefixes: &[&str]) -> Result<Self> {
        let (total_out, hidden, dtype, total_bytes) =
            scratchy_layers::layers::concat_packed_plan(weights, prefixes)?;

        // Direct-write packing: pre-allocate the destination MTLBuffer
        // and stream each source tensor's bytes straight into it. The
        // older path went mmap → heap Vec → arena MTLBuffer (two
        // memcpies, each ~100 MB/layer for Llama-3.2-3B's gate_up
        // pack — the dominant chunk of init engine time after the
        // zero-copy weight load fix).
        //
        // CUDA uses `take_cpu` + heap `Vec` because its async H2D copy needs
        // the bytes pinned, and the existing batched pinned-pool pipeline
        // already handles the staging — there's no "second memcpy" to elide.
        let packed_weight = {
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
            weights.alloc_packed_from_host(&packed, &[total_out, hidden], dtype)?
        };

        let packed_bias = scratchy_layers::layers::concat_packed_bias(weights, prefixes, dtype)?;

        Ok(Self::Dense(Linear::new(packed_weight, packed_bias)))
    }

    /// Load several dense linear layers and concatenate along the
    /// out-feature dim (dim 0 of the weight matrix), returning one
    /// packed `LinearLayer::Dense`.
    ///
    /// Streams each source weight directly from CPU-safetensors into
    /// the packed GPU buffer (no D2D copy, no intermediate allocation).
    /// Used by scratchy-forward-compiler's fused accessors — e.g. `FusedQkvRopeCacheImpl`
    /// expects one packed `[q_size + 2*kv_size, hidden]` weight covering
    /// the three source q/k/v projections.
    ///
    /// If any source weight has a bias, all of them must — the biases
    /// are concatenated in the same order as the weights. Otherwise
    /// the returned layer has no bias.
    #[cfg(feature = "cuda")]
    fn load_dense_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        if prefixes.is_empty() {
            anyhow::bail!("load_dense_concat: empty prefix list");
        }

        // Packed-source fallback: if any source .weight is missing, it
        // may live under a packed parent (qkv_proj / gate_up_proj).
        // Mirrors the fallback in `Linear::load`.
        for p in prefixes {
            if !weights.contains(&format!("{p}.weight")) {
                try_synthesize_packed_slice(weights, p)?;
            }
        }

        // First pass: resolve shapes/dtype from CPU-side metadata.
        let mut shapes_dtypes: Vec<(Vec<usize>, crate::dtype::DType)> =
            Vec::with_capacity(prefixes.len());
        for p in prefixes {
            let weight_name = format!("{p}.weight");
            let (shape, dtype) = weights
                .tensor_info(&weight_name)
                .ok_or_else(|| anyhow::anyhow!("weight not found: {weight_name}"))?;
            if shape.len() != 2 {
                anyhow::bail!(
                    "load_dense_concat: `{weight_name}` has rank {}, expected 2",
                    shape.len(),
                );
            }
            shapes_dtypes.push((shape.to_vec(), dtype));
        }

        // All sources must share the same in-features (dim 1) and dtype.
        let hidden = shapes_dtypes[0].0[1];
        let dtype = shapes_dtypes[0].1;
        for (i, (shape, dt)) in shapes_dtypes.iter().enumerate() {
            if shape[1] != hidden {
                anyhow::bail!(
                    "load_dense_concat: `{}` has in_features {}, expected {}",
                    prefixes[i],
                    shape[1],
                    hidden,
                );
            }
            if *dt != dtype {
                anyhow::bail!(
                    "load_dense_concat: `{}` has dtype {:?}, expected {:?}",
                    prefixes[i],
                    dt,
                    dtype,
                );
            }
        }

        let total_out: usize = shapes_dtypes.iter().map(|(s, _)| s[0]).sum();
        let elem = dtype.size_bytes();
        let total_bytes = total_out * hidden * elem;

        // One contiguous GPU buffer; stream each source into its offset.
        let ptr = unsafe { crate::driver::mem_alloc(total_bytes)? };
        weights.record_alloc(ptr, total_bytes);
        let mut offset_bytes: usize = 0;
        for (i, p) in prefixes.iter().enumerate() {
            let weight_name = format!("{p}.weight");
            let bytes = shapes_dtypes[i].0[0] * hidden * elem;
            unsafe {
                weights.take_into(&weight_name, ptr.add(offset_bytes), stream)?;
            }
            offset_bytes += bytes;
        }
        let packed_weight = unsafe { GpuTensor::new(ptr, &[total_out, hidden], dtype) };

        // Biases: either all-or-none across the source set.
        let bias_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.bias")).collect();
        let packed_bias = if weights.contains(&bias_names[0]) {
            for bn in &bias_names {
                if !weights.contains(bn) {
                    anyhow::bail!(
                        "load_dense_concat: inconsistent bias — `{}` exists but `{bn}` is missing",
                        bias_names[0],
                    );
                }
            }
            let mut per_bias_bytes: Vec<usize> = Vec::with_capacity(bias_names.len());
            let mut total_bias_bytes = 0usize;
            for bn in &bias_names {
                let (bshape, bdt) = weights
                    .tensor_info(bn)
                    .ok_or_else(|| anyhow::anyhow!("bias metadata missing: {bn}"))?;
                if bdt != dtype {
                    anyhow::bail!(
                        "load_dense_concat: bias `{bn}` dtype {:?} != weight dtype {:?}",
                        bdt,
                        dtype,
                    );
                }
                let b = bshape.iter().product::<usize>() * elem;
                per_bias_bytes.push(b);
                total_bias_bytes += b;
            }
            let bptr = unsafe { crate::driver::mem_alloc(total_bias_bytes)? };
            weights.record_alloc(bptr, total_bias_bytes);
            let mut boff = 0usize;
            for (i, bn) in bias_names.iter().enumerate() {
                unsafe {
                    weights.take_into(bn, bptr.add(boff), stream)?;
                }
                boff += per_bias_bytes[i];
            }
            Some(unsafe { GpuTensor::new(bptr, &[total_bias_bytes / elem], dtype) })
        } else {
            None
        };

        Ok(Self::Dense(Linear::new(packed_weight, packed_bias)))
    }

    /// Tensor-parallel sharded variant of [`Self::load_dense_concat`].
    /// Used by codegen at tp>1 for the fused QKV / gate_up
    /// projections, which are always column-parallel (`ShardDim0`)
    /// — no row-parallel concat exists in any current arch.
    ///
    /// Each source weight is sliced along its output dim (`dim 0` of
    /// the `[out, in]` matrix) to `[out / world, in]`, then concatenated
    /// along that same dim 0 into one packed `[(sum out_i) / world, in]`
    /// GPU buffer. Biases follow the column-parallel rule (sliced
    /// along dim 0 too) — matches Python `MergedColumnParallelLinear`
    /// / `QKVParallelLinear` weight loaders.
    ///
    /// `world == 1` short-circuits to byte-equivalent behavior with
    /// `Self::load_dense_concat`. Each source's `out` dim must be
    /// divisible by `world` — the macro's outer-loop fanout already
    /// `skip`s indivisible (variant, tp) tuples (per the activation
    /// commit `889c44b2f`), so this is a runtime invariant the
    /// compile-time set guarantees, not a per-call assertion.
    #[cfg(feature = "cuda")]
    fn load_dense_concat_sharded(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        stream: cudarc::driver::sys::CUstream,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        if prefixes.is_empty() {
            anyhow::bail!("load_dense_concat_sharded: empty prefix list");
        }

        // GGUF fast-path. The GGUF loader pre-shards every source at
        // file-read time (fused QKV and gate_up sources are all
        // ShardDim0 → per-rank rows). Each per-prefix tensor is already
        // `[out_i / world, in]`; we just collect them without touching
        // the bytes. Mirrors the tp=1 `load_dense_concat_or_ggml` path.
        let weight_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.weight")).collect();
        let all_gguf_quantized = weight_names
            .iter()
            .all(|n| weights.contains_quantized_linear(n));
        if all_gguf_quantized {
            // Prefer the zero-copy `GgmlConcat` path — the storages
            // share the underlying GPU buffers in `GpuWeights.quantized`
            // (take is non-destructive). No byte-pack, no extra H2D,
            // no dequant. Concat is always column-parallel (ShardDim0),
            // so each branch's bias (if present) must be narrowed to
            // the per-rank slice matching `storage.nrows` — Qwen2's
            // biased q/k/v at tp>1 is the canonical example. GGUF 1D
            // biases are `Replicate` on-disk (full length on every rank),
            // so we compute the rank slice here against the replicated
            // allocation.
            let mut storages: Vec<crate::ggml::GgmlStorage> = Vec::with_capacity(prefixes.len());
            for name in &weight_names {
                let s = weights
                    .take_quantized_linear(name)
                    .ok_or_else(|| anyhow::anyhow!("load_dense_concat_sharded: missing {name}"))?;
                storages.push(s);
            }
            let first_ncols = storages[0].ncols;
            for s in &storages[1..] {
                if s.ncols != first_ncols {
                    anyhow::bail!(
                        "load_dense_concat_sharded: ncols mismatch ({} vs {})",
                        first_ncols,
                        s.ncols
                    );
                }
            }
            let bias_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.bias")).collect();
            let any_bias = bias_names.iter().any(|n| weights.contains(n));
            let all_bias = bias_names.iter().all(|n| weights.contains(n));
            if any_bias && !all_bias {
                anyhow::bail!(
                    "load_dense_concat_sharded: inconsistent biases across prefixes {:?}",
                    prefixes
                );
            }
            let mut biases: Vec<Option<GpuTensor>> = vec![None; prefixes.len()];
            if all_bias {
                for (i, n) in bias_names.iter().enumerate() {
                    let full = weights.take(n)?;
                    biases[i] = Some(if world > 1 {
                        per_rank_bias_slice(full, rank, world, storages[i].nrows)
                    } else {
                        full
                    });
                }
            }
            let _ = stream;
            let branches: Vec<GgmlLinear> = storages
                .into_iter()
                .zip(biases)
                .map(|(s, bias)| GgmlLinear { storage: s, bias })
                .collect();
            return Ok(Self::GgmlConcat(branches));
        }
        let all_gguf_dense = weight_names.iter().all(|n| weights.gguf_dense_contains(n));
        if all_gguf_dense {
            // F16/F32 GGUF: tensors are in `gguf_dense` already per-rank
            // sharded (ShardDim0). `load_gguf_dense_concat` packs them
            // row-wise into one `[sum(rows_per_rank), cols]` buffer —
            // the contract it already has, just with per-rank inputs
            // instead of unsharded ones.
            let _ = (rank, world);
            return load_gguf_dense_concat(weights, prefixes, stream);
        }

        // Same packed-source fallback as the unsharded path.
        for p in prefixes {
            if !weights.contains(&format!("{p}.weight")) {
                try_synthesize_packed_slice(weights, p)?;
            }
        }

        // Resolve unsharded shapes/dtype from CPU-side metadata,
        // then divide each source's out dim by `world` to get the
        // per-rank packed shape.
        let mut shapes_dtypes: Vec<(Vec<usize>, crate::dtype::DType)> =
            Vec::with_capacity(prefixes.len());
        for p in prefixes {
            let weight_name = format!("{p}.weight");
            let (shape, dtype) = weights
                .tensor_info(&weight_name)
                .ok_or_else(|| anyhow::anyhow!("weight not found: {weight_name}"))?;
            if shape.len() != 2 {
                anyhow::bail!(
                    "load_dense_concat_sharded: `{weight_name}` has rank {}, expected 2",
                    shape.len(),
                );
            }
            shapes_dtypes.push((shape.to_vec(), dtype));
        }

        let hidden = shapes_dtypes[0].0[1];
        let dtype = shapes_dtypes[0].1;
        for (i, (shape, dt)) in shapes_dtypes.iter().enumerate() {
            if shape[1] != hidden {
                anyhow::bail!(
                    "load_dense_concat_sharded: `{}` has in_features {}, expected {}",
                    prefixes[i],
                    shape[1],
                    hidden,
                );
            }
            if *dt != dtype {
                anyhow::bail!(
                    "load_dense_concat_sharded: `{}` has dtype {:?}, expected {:?}",
                    prefixes[i],
                    dt,
                    dtype,
                );
            }
        }

        let elem = dtype.size_bytes();
        // Per-rank packed shape: each source's out dim / world.
        let per_rank_outs: Vec<usize> = shapes_dtypes.iter().map(|(s, _)| s[0] / world).collect();
        let total_out: usize = per_rank_outs.iter().sum();
        let total_bytes = total_out * hidden * elem;

        let ptr = unsafe { crate::driver::mem_alloc(total_bytes)? };
        weights.record_alloc(ptr, total_bytes);
        let mut offset_bytes: usize = 0;
        for (i, p) in prefixes.iter().enumerate() {
            let weight_name = format!("{p}.weight");
            let bytes = per_rank_outs[i] * hidden * elem;
            unsafe {
                weights.take_shard_into(
                    &weight_name,
                    0,
                    rank,
                    world,
                    ptr.add(offset_bytes),
                    stream,
                )?;
            }
            offset_bytes += bytes;
        }
        let packed_weight = unsafe { GpuTensor::new(ptr, &[total_out, hidden], dtype) };

        // Biases: column-parallel concat shards each bias along dim 0
        // — same rule as the weight. Either all sources have a bias
        // or none of them do (matches the unsharded contract).
        let bias_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.bias")).collect();
        let packed_bias = if weights.contains(&bias_names[0]) {
            for bn in &bias_names {
                if !weights.contains(bn) {
                    anyhow::bail!(
                        "load_dense_concat_sharded: inconsistent bias — `{}` exists but `{bn}` is missing",
                        bias_names[0],
                    );
                }
            }
            // Per-rank bias bytes mirror per-rank weight outs (bias
            // shape is `[out_i]` unsharded → `[out_i / world]` per rank).
            let mut per_bias_bytes: Vec<usize> = Vec::with_capacity(bias_names.len());
            let mut total_bias_bytes = 0usize;
            for (i, bn) in bias_names.iter().enumerate() {
                let (bshape, bdt) = weights
                    .tensor_info(bn)
                    .ok_or_else(|| anyhow::anyhow!("bias metadata missing: {bn}"))?;
                if bdt != dtype {
                    anyhow::bail!(
                        "load_dense_concat_sharded: bias `{bn}` dtype {:?} != weight dtype {:?}",
                        bdt,
                        dtype,
                    );
                }
                // Ignore unsharded `bshape` here — per-rank bytes are
                // derived from the matched weight's per-rank out dim
                // (already validated as `out_i / world` above).
                let _ = bshape;
                let b = per_rank_outs[i] * elem;
                per_bias_bytes.push(b);
                total_bias_bytes += b;
            }
            let bptr = unsafe { crate::driver::mem_alloc(total_bias_bytes)? };
            weights.record_alloc(bptr, total_bias_bytes);
            let mut boff = 0usize;
            for (i, bn) in bias_names.iter().enumerate() {
                unsafe {
                    weights.take_shard_into(bn, 0, rank, world, bptr.add(boff), stream)?;
                }
                boff += per_bias_bytes[i];
            }
            Some(unsafe { GpuTensor::new(bptr, &[total_bias_bytes / elem], dtype) })
        } else {
            None
        };

        Ok(Self::Dense(Linear::new(packed_weight, packed_bias)))
    }
}

// ---------------------------------------------------------------------------
// Fp8BlockLinear (FP8 E4M3 with per-block scales, e.g. DeepSeek-V3) — forward
// ---------------------------------------------------------------------------

/// Forward-only extension trait on the neutral [`Fp8BlockLinear`].
#[cfg(feature = "cuda")]
pub trait Fp8BlockOps {
    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor;
}

#[cfg(feature = "cuda")]
impl Fp8BlockOps for Fp8BlockLinear {
    /// Forward: dequant FP8 → BF16 per block, then cuBLAS GEMM.
    ///
    /// Current implementation: CPU-side dequant-then-GEMM for correctness.
    /// TODO: CUTLASS block-scaled FP8 GEMM for perf parity.
    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        debug_assert_eq!(x.ndim(), 2);
        let k = self.weight.dim(1);
        debug_assert_eq!(x.dim(1), k, "Fp8BlockLinear: input dim mismatch");

        // Dequantize FP8 weight to BF16/F16 using per-block scales.
        let dequant_weight = crate::kernels::fp8_block_dequant(
            self.weight,
            self.weight_scale_inv,
            self.block_size,
            self.output_dtype,
            alloc,
            stream,
        );

        // Standard GEMM: x @ dequant_weight^T
        // Use as_gpu_tensor() to borrow — dequant_weight drops after GEMM,
        // returning the buffer to the caching allocator.
        let out = cublas.gemm(*x, dequant_weight.as_gpu_tensor(), alloc);
        drop(dequant_weight);

        if let Some(bias) = self.bias {
            crate::kernels::bias_add_inplace(out.as_gpu_tensor(), bias, stream);
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Fp8AnyLinear (per-tensor / per-channel `Fp8Linear` ⨁ blockwise `Fp8BlockLinear`) — forward
// ---------------------------------------------------------------------------

/// Forward-only extension trait on the neutral [`Fp8AnyLinear`] wrapper.
#[cfg(feature = "cuda")]
pub trait Fp8AnyOps {
    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor;
}

#[cfg(feature = "cuda")]
impl Fp8AnyOps for Fp8AnyLinear {
    /// Forward: dispatches to whichever inner FP8 layer is wrapped.
    /// Both variants take the same arguments and return `OwnedTensor`.
    ///
    /// # Safety
    /// All tensors must be valid GPU memory; `cublas` / `alloc` /
    /// `stream` must be live. Inner `forward`s carry the same
    /// invariants.
    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        match self {
            Self::Std(l) => unsafe { l.forward(x, cublas, alloc, stream) },
            Self::Block(l) => unsafe { l.forward(x, cublas, alloc, stream) },
        }
    }
}

// ---------------------------------------------------------------------------
// Embedding — runtime extension trait (load* + forward)
// ---------------------------------------------------------------------------

/// Runtime extension methods on the neutral [`Embedding`] type. The struct +
/// its pure accessors live in `scratchy-layers`; each method keeps the exact
/// `#[cfg]` it had as an inherent method.
#[cfg(feature = "cuda")]
pub trait EmbeddingOps {
    #[cfg(feature = "cuda")]
    fn load(weights: &mut GpuWeights, prefix: &str) -> Result<Self>
    where
        Self: Sized;

    #[cfg(feature = "cuda")]
    fn load_sharded(
        weights: &mut GpuWeights,
        prefix: &str,
        rank: usize,
        world: usize,
    ) -> Result<Self>
    where
        Self: Sized;

    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        input_ids: TensorView<'_>,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> GpuTensor;
}

#[cfg(feature = "cuda")]
impl EmbeddingOps for Embedding {
    /// Load from `GpuWeights` by prefix.
    #[cfg(feature = "cuda")]
    fn load(weights: &mut GpuWeights, prefix: &str) -> Result<Self> {
        scratchy_layers::layers::embedding_load(weights, prefix)
    }

    /// Vocab-parallel sharded load — slices the embedding table
    /// along dim 0 (`vocab_size`) so each rank holds
    /// `[vocab_size / world, hidden_size]`. Mirrors Python vLLM's
    /// `VocabParallelEmbedding` (the same scheme `ParallelLMHead`
    /// inherits, which is what makes `tie_weights` self-consistent
    /// at tp>1 — both sharded slices come from the same dim-0 cut).
    /// `world == 1` degrades to the same shape `Self::load` produces.
    #[cfg(feature = "cuda")]
    fn load_sharded(
        weights: &mut GpuWeights,
        prefix: &str,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        // GGUF fast-path: `GgufGpuWeights::load` pre-shards `embed_tokens`
        // as ShardDim0 at file-read time (see `gguf_shard_kind_for_hf_name`
        // in `scratchy-target-cuda/src/ggml.rs`), so the per-rank tensor is
        // already in `gguf_dense` with shape `[vocab/world, hidden]`.
        // Take it as-is; the safetensors `take_shard` path below would
        // miss because the tensor was never inserted into the CPU
        // `tensors` map.
        if let Some(t) = weights.take_gguf_dense(&weight_name) {
            return Ok(Self::new(t));
        }
        let weight = weights.take_shard(&weight_name, 0, rank, world)?;
        Ok(Self::new(weight))
    }

    /// Forward: gather embedding rows by token IDs.
    ///
    /// `input_ids`: `[num_tokens]` (U32 on GPU)
    /// Returns: `[num_tokens, hidden_size]` allocated from arena.
    ///
    /// # Safety
    /// All tensors must be valid GPU memory. Stream must be valid.
    /// This currently uses a simple gather kernel (TODO: implement via CUDA kernel).
    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        _input_ids: TensorView<'_>,
        _alloc: &mut CachingAllocator,
        _stream: cudarc::driver::sys::CUstream,
    ) -> GpuTensor {
        // TODO: implement embedding gather kernel.
        // For now, return a placeholder. The kernel is trivial:
        // one thread per (token, dim) reads weight[input_ids[token], dim].
        todo!("embedding gather kernel not yet implemented")
    }
}

// ---------------------------------------------------------------------------
// RmsNorm — load (NUMERIC: take_keep_dtype vs take cfg split)
// ---------------------------------------------------------------------------

/// Load extension trait on the neutral [`RmsNorm`] type. The struct + pure
/// accessors live in `scratchy-layers`.
#[cfg(feature = "cuda")]
pub trait RmsNormOps {
    #[cfg(feature = "cuda")]
    fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self>
    where
        Self: Sized;
}

#[cfg(feature = "cuda")]
impl RmsNormOps for RmsNorm {
    /// Load from `GpuWeights` by prefix.
    ///
    /// Metal: route through `take_keep_dtype` so the on-disk gain
    /// dtype (F16 on every sampled mlx-community / Llama-3.x
    /// checkpoint) reaches the device verbatim. The
    /// `<T_act, T_scale>` rmsnorm kernel (`shaders/rmsnorm.metal`,
    /// picked by `interpreter::metal::lowering::rmsnorm_kernel_static_name`)
    /// casts the gain to the activation dtype in registers. Mirrors
    /// the P10b in-register cast applied to affine quant scales; the
    /// pre-P10c bf16-stack `take()` truncated the F16 gain to BF16 at
    /// load (10→7 bit mantissa).
    ///
    /// CUDA: keep the existing `take()` path — `rms_norm_bf16` /
    /// `rms_norm_f16` (`kernels.rs:26-46`) take a single typed
    /// weight pointer, so the loader cast is what makes the
    /// dtype-matched dispatcher (`rms_norm_with_offset`) work today.
    /// Extending the kernel signature is its own thread; out of P10c
    /// scope.
    #[cfg(feature = "cuda")]
    fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self> {
        // Standard convention: `<prefix>.weight`. Fallback: the bare
        // `<prefix>` key — Gemma4's per-layer `layer_scalar` is a bare
        // [1] tensor with no `.weight` suffix on disk (it loads through
        // the RmsNorm accessor kind as a 1-element gain).
        let weight_name = format!("{prefix}.weight");
        let weight = weights
            .take(&weight_name)
            .or_else(|_| weights.take(prefix))?;
        Ok(Self::new(weight, eps))
    }
}

// ---------------------------------------------------------------------------
// LayerNorm — load
// ---------------------------------------------------------------------------

/// Load extension trait on the neutral [`LayerNorm`] type. The struct + pure
/// accessors live in `scratchy-layers`.
#[cfg(feature = "cuda")]
pub trait LayerNormOps {
    #[cfg(feature = "cuda")]
    fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self>
    where
        Self: Sized;
}

#[cfg(feature = "cuda")]
impl LayerNormOps for LayerNorm {
    /// Load from `GpuWeights` by prefix. Loads bias if present.
    #[cfg(feature = "cuda")]
    fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self> {
        scratchy_layers::layers::layer_norm_load(weights, prefix, eps)
    }
}

// ---------------------------------------------------------------------------
// Tensor-parallel layer wrappers
// ---------------------------------------------------------------------------

/// Column-parallel linear: shards output dim (dim=0 of weight).
///
/// After forward, optionally all-gathers output across ranks to reconstruct
/// the full output (used for lm_head). For most uses (QKV, gate_up), no
/// all-gather is needed because the downstream layer consumes the shard.
pub struct ColumnParallelLinear {
    pub inner: LinearLayer,
    pub gather_output: bool,
    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
}

#[cfg(feature = "cuda")]
impl ColumnParallelLinear {
    pub fn new(inner: LinearLayer, gather_output: bool) -> Self {
        Self {
            inner,
            gather_output,
            #[cfg(feature = "nccl")]
            tp_group: None,
        }
    }

    /// Forward: y = x @ W_shard^T, optionally all-gather.
    pub unsafe fn forward(
        &self,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        let out = self.inner.forward(x, cublas, alloc, stream);

        #[cfg(feature = "nccl")]
        if self.gather_output
            && let Some(ref group) = self.tp_group
        {
            let gathered = group.all_gather(out.as_gpu_tensor(), alloc);
            drop(out);
            return gathered;
        }

        out
    }

    pub fn out_features(&self) -> usize {
        self.inner.out_features()
    }

    pub fn in_features(&self) -> usize {
        self.inner.in_features()
    }

    #[cfg(feature = "nccl")]
    pub fn set_tp_group(&mut self, group: Arc<NcclGroup>) {
        self.tp_group = Some(group);
    }
}

/// Row-parallel linear: shards input dim (dim=1 of weight).
///
/// After forward, all-reduces output across ranks (each rank computed a
/// partial sum). Bias is added AFTER the all-reduce.
pub struct RowParallelLinear {
    pub inner: LinearLayer,
    /// Bias added after all-reduce (not inside the GEMM).
    pub bias: Option<GpuTensor>,
    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
}

#[cfg(feature = "cuda")]
impl RowParallelLinear {
    pub fn new(inner: LinearLayer, bias: Option<GpuTensor>) -> Self {
        Self {
            inner,
            bias,
            #[cfg(feature = "nccl")]
            tp_group: None,
        }
    }

    /// Forward: y = all_reduce(x @ W_shard^T) + bias.
    pub unsafe fn forward(
        &self,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        let out = self.inner.forward(x, cublas, alloc, stream);

        #[cfg(feature = "nccl")]
        if let Some(ref group) = self.tp_group {
            group
                .all_reduce_inplace(out.as_gpu_tensor())
                .expect("all_reduce failed");
        }

        if let Some(bias) = self.bias {
            crate::kernels::bias_add_inplace(out.as_gpu_tensor(), bias, stream);
        }

        out
    }

    pub fn out_features(&self) -> usize {
        self.inner.out_features()
    }

    #[cfg(feature = "nccl")]
    pub fn set_tp_group(&mut self, group: Arc<NcclGroup>) {
        self.tp_group = Some(group);
    }
}

/// Vocab-parallel embedding: shards vocab rows across ranks.
///
/// Each rank holds rows `[rank * shard_size .. (rank+1) * shard_size]`.
/// Tokens outside the local range produce zeros. All-reduce sums partial
/// results to reconstruct the full embedding.
pub struct VocabParallelEmbedding {
    pub inner: Embedding,
    /// Global vocab start offset for this rank's shard.
    pub vocab_start: usize,
    /// Global vocab end offset (exclusive) for this rank's shard.
    pub vocab_end: usize,
    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
}

#[cfg(feature = "cuda")]
impl VocabParallelEmbedding {
    pub fn new(inner: Embedding, vocab_start: usize, vocab_end: usize) -> Self {
        Self {
            inner,
            vocab_start,
            vocab_end,
            #[cfg(feature = "nccl")]
            tp_group: None,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }

    #[cfg(feature = "nccl")]
    pub fn set_tp_group(&mut self, group: Arc<NcclGroup>) {
        self.tp_group = Some(group);
    }
}

// ---------------------------------------------------------------------------
// CohereLayerNorm — load (cuda-only)
// ---------------------------------------------------------------------------

/// Load extension trait on the neutral [`CohereLayerNorm`]. The struct + pure
/// `new`/`hidden_size` live in `scratchy-layers`. The load was inside a
/// `#[cfg(cuda)] impl` block, so its net cfg is cuda.
#[cfg(feature = "cuda")]
pub trait CohereLayerNormOps {
    #[cfg(feature = "cuda")]
    fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self>
    where
        Self: Sized;
}

#[cfg(feature = "cuda")]
impl CohereLayerNormOps for CohereLayerNorm {
    /// Load from `GpuWeights` by prefix.
    #[cfg(feature = "cuda")]
    fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        let weight = weights.take(&weight_name)?;
        Ok(Self::new(weight, eps))
    }
}

// ---------------------------------------------------------------------------
// LayerNormBias — load (cuda-only)
// ---------------------------------------------------------------------------

/// Load extension trait on the neutral [`LayerNormBias`]. The struct + pure
/// `new` live in `scratchy-layers`. The load was inside a `#[cfg(cuda)] impl`
/// block, so its net cfg is cuda.
#[cfg(feature = "cuda")]
pub trait LayerNormBiasOps {
    #[cfg(feature = "cuda")]
    fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self>
    where
        Self: Sized;
}

#[cfg(feature = "cuda")]
impl LayerNormBiasOps for LayerNormBias {
    /// Load from `GpuWeights` by prefix — looks up `<prefix>.weight`
    /// and `<prefix>.bias` as 1D tensors.
    #[cfg(feature = "cuda")]
    fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self> {
        let weight = weights.take(&format!("{prefix}.weight"))?;
        let bias = weights.take(&format!("{prefix}.bias"))?;
        Ok(Self::new(weight, bias, eps))
    }
}

// ---------------------------------------------------------------------------
// GatedDeltaNetLayer — load (NUMERIC: take_keep_dtype ×3 + take_as_f32)
// ---------------------------------------------------------------------------

/// Load extension trait on the neutral [`GatedDeltaNetLayer`] bundle. The
/// struct + pure `new` live in `scratchy-layers`.
#[cfg(feature = "cuda")]
pub trait GatedDeltaNetOps {
    #[cfg(feature = "cuda")]
    fn load(weights: &mut GpuWeights, prefix: &str) -> Result<Self>
    where
        Self: Sized;
}

#[cfg(feature = "cuda")]
impl GatedDeltaNetOps for GatedDeltaNetLayer {
    /// Load from `GpuWeights`. `prefix` is the linear-attn prefix for this
    /// layer (e.g. `model.language_model.layers.0.linear_attn`). Tensors are
    /// taken in their on-disk dtype (`take_keep_dtype`) — `A_log`/`dt_bias` are
    /// commonly fp32 in mamba-family checkpoints and the cuda eval casts the
    /// rest to f32 at use.
    #[cfg(feature = "cuda")]
    fn load(weights: &mut GpuWeights, prefix: &str) -> Result<Self> {
        scratchy_layers::layers::gated_delta_net_load(weights, prefix)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
