// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral quant payload TYPE DEFINITIONS for the MLX-affine INT4 and
//! NVFP4 linears + the affine-quantized embedding.
//!
//! These are pure data (`GpuTensor` / primitives) — the `LinearLayer`
//! enum holds them behind `AffineQuant` / `Nvfp4` variants, so they must live
//! below it in a backend-neutral crate. Their `load()` constructors (against
//! the `WeightSource` seam) and the `e4m3_to_f32` decode stay in
//! `scratchy-target-metal`, which owns the runtime; this crate names no backend.

use scratchy_tensors::GpuTensor;

// ---------------------------------------------------------------------------
// AffineQuantLinear (MLX-native int4 affine)
// ---------------------------------------------------------------------------

/// MLX-native affine INT4 quantized linear layer (Metal backend only).
///
/// Storage layout matches `mlx-community/*-4bit` checkpoints exactly:
/// weights packed `[N, K / pack_factor]` U32 (`pack_factor = 32 / bits =
/// 8` for bits=4); per-group affine offset (`scales`, `biases`) stored
/// `[N, K / group_size]` F16. Activation dtype is bf16 or f16 per
/// `torch_dtype`. The kernel reads scales/biases as `T_scale = half`
/// and casts to float in registers.
///
/// "biases" here is MLX's per-group affine offset, NOT the linear-layer
/// bias. The optional fp linear-layer bias (when models like Phi-3 / some
/// Qwen2 variants ship it) lives in `linear_bias` separately.
///
/// No `forward()` method on this type — Metal forwards go through the
/// macro-emitted `Instruction<W>` stream and dispatch via the worker
/// resolver, not direct method calls.
pub struct AffineQuantLinear {
    /// Packed 4-bit weights, shape `[N, K / pack_factor]`, dtype `U32`.
    pub weight: GpuTensor,
    /// Per-group scales, shape `[N, K / group_size]`, dtype `F16`.
    pub scales: GpuTensor,
    /// Per-group affine offsets ("biases" in MLX terminology — NOT the
    /// linear-layer bias). Shape `[N, K / group_size]`, dtype `F16`.
    pub affine_biases: GpuTensor,
    /// Optional fp linear-layer bias `[N]` (when present in the
    /// safetensors as `<prefix>.bias`; absent on Llama-3.2 family).
    pub linear_bias: Option<GpuTensor>,
    pub in_features: usize,
    pub out_features: usize,
    pub group_size: u32,
    pub bits: u32,
}

impl AffineQuantLinear {
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }
}

// ---------------------------------------------------------------------------
// Nvfp4Linear — NVIDIA ModelOpt NVFP4 4-bit linear
// ---------------------------------------------------------------------------

/// NVIDIA ModelOpt NVFP4 4-bit quantized linear (Metal-only).
///
/// On disk: `.weight` is `uint8 [N, K/2]` (two packed E2M1 codes per byte,
/// low nibble = even element), `.weight_scale` is `float8_e4m3 [N, K/16]`
/// (per-block scale), `.weight_scale_2` is an `f32` per-tensor global scale.
///
/// At load we fold the two-level scale into a single per-group `F16` tensor
/// `scales[n,g] = e4m3_to_f32(weight_scale[n,g]) * weight_scale_2`, so
/// the `nvfp4_qmv` / `nvfp4_qmm_t` kernels reconstruct exactly
/// `w = E2M1_signed[code] * scales[n, k/16]` (matching Python vLLM's
/// `dequantize_to_dtype`). Weight-only dequant — activations stay bf16/f16.
/// No per-group bias (NVFP4 is symmetric); `linear_bias` is the optional
/// fp linear-layer bias, distinct from any quantization offset.
///
/// As with `AffineQuantLinear`, there's no `forward()` here — Metal forwards
/// go through the macro-emitted `Instruction` stream and the worker resolver.
pub struct Nvfp4Linear {
    /// Packed E2M1 weights, shape `[N, K/2]`, dtype `U8`.
    pub weight: GpuTensor,
    /// Per-group folded scales `[N, K/group_size]`, dtype `F16` —
    /// `e4m3_to_f32(weight_scale) * weight_scale_2`.
    pub scales: GpuTensor,
    /// Optional fp linear-layer bias `[N]` (`<prefix>.bias`).
    pub linear_bias: Option<GpuTensor>,
    pub in_features: usize,
    pub out_features: usize,
    /// NVFP4 block size — 16 for every ModelOpt checkpoint.
    pub group_size: u32,
    /// Always 4 (NVFP4 is a 4-bit format).
    pub bits: u32,
}

impl Nvfp4Linear {
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }
}

// ---------------------------------------------------------------------------
// AffineQuantEmbedding — MLX-affine int4 quantized token embedding
// ---------------------------------------------------------------------------

/// MLX-affine int4 quantized token-embedding table. Mirrors
/// `AffineQuantLinear` (the int4 Linear) but for an embedding's
/// `[vocab_size, hidden_size]` layout. Stored as packed U32 weights
/// (`[vocab, hidden / pack_factor]`, `pack_factor = 32 / bits = 8` for
/// bits=4) plus per-group `scales` / `affine_biases`
/// (`[vocab, hidden / group_size]` F16). MLX terminology: "biases" is
/// the per-group affine offset, NOT a linear-layer bias — embeddings
/// have no fp bias term at all.
///
/// Used by P6's `Instruction::AffineEmbed`: forward-time gather +
/// dequant via `affine_embed_<dtype>_gs_<gs>_b_4`. The macro emits this
/// type instead of `Embedding` when `model.embed_tokens` carries
/// `(weight=U32, scales, biases)` safetensors keys. Tied lm_head reuses
/// the same buffer triple (GpuTensor is Copy under metal — it's a thin
/// pointer wrapper) via `LinearLayer::AffineQuant`.
pub struct AffineQuantEmbedding {
    /// Packed 4-bit weights, shape `[vocab_size, hidden_size / pack_factor]`,
    /// dtype `U32`.
    pub weight: GpuTensor,
    /// Per-group scales, shape `[vocab_size, hidden_size / group_size]`,
    /// dtype `F16`.
    pub scales: GpuTensor,
    /// Per-group affine offsets, shape `[vocab_size, hidden_size / group_size]`,
    /// dtype `F16`. NOT a linear-layer bias — MLX-terminology naming.
    pub affine_biases: GpuTensor,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub group_size: u32,
    pub bits: u32,
}

impl AffineQuantEmbedding {
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}
