// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral model layer TYPE DEFINITIONS + pure accessors.
//!
//! Weights are stored as `GpuTensor` (raw GPU pointers / metadata only). The
//! struct/enum definitions and their *pure* shape/accessor methods live here so
//! both `targets/{cuda,metal}` can name them without a circular dependency. The
//! kernel-calling `forward` passes live in the target crates as `*Ops`
//! extension traits; the **backend-neutral** `GpuWeights`-bound `load_*` logic
//! (which names only neutral `GpuWeights` access methods) lives here as shared
//! free fns both targets delegate to — only genuinely backend-divergent loads
//! (cuda-stream sharded/concat, metal direct-write packing, the RmsNorm
//! keep-dtype split) stay in the target crates.

use anyhow::Result;
use scratchy_tensors::{DType, DeviceAllocator, GpuTensor};

use crate::weights::GpuWeights;
use scratchy_quantizations::ggml_quant::GgmlStorage;
use scratchy_quantizations::layers_quant::{AffineQuantLinear, Nvfp4Linear};

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

/// Dense linear layer: y = x @ W^T + b
///
/// Weight is stored in `[out_features, in_features]` layout (NOT pre-transposed).
/// cuBLAS GEMM handles the transpose internally via `CUBLAS_OP_T`, which is
/// more efficient than a separate transpose copy.
pub struct Linear {
    pub weight: GpuTensor,       // [out_features, in_features]
    pub bias: Option<GpuTensor>, // [out_features]
}

impl Linear {
    /// Create from explicit weight and bias tensors.
    pub fn new(weight: GpuTensor, bias: Option<GpuTensor>) -> Self {
        debug_assert_eq!(weight.ndim(), 2);
        if let Some(ref b) = bias {
            debug_assert_eq!(b.ndim(), 1);
            debug_assert_eq!(b.dim(0), weight.dim(0));
        }
        Self { weight, bias }
    }

    pub fn out_features(&self) -> usize {
        self.weight.dim(0)
    }

    pub fn in_features(&self) -> usize {
        self.weight.dim(1)
    }
}

// ---------------------------------------------------------------------------
// MarlinLinear (INT4 quantized via Marlin GEMM)
// ---------------------------------------------------------------------------

/// Quantized linear layer using Marlin INT4×FP16→FP16 GEMM.
///
/// Weights are repacked to Marlin tiled format at load time.
/// Supports both AWQ (has zero points) and GPTQ (symmetric or with zeros).
pub struct MarlinLinear {
    /// Marlin-tiled packed INT4 weights.
    pub qweight: GpuTensor,
    /// Per-group scales `[num_groups, size_n]`, permuted for Marlin.
    pub scales: GpuTensor,
    /// Packed zero points (AWQ) or None (GPTQ symmetric).
    pub zeros: Option<GpuTensor>,
    /// Group index for act_order (desc_act) or None.
    pub g_idx: Option<GpuTensor>,
    /// Sort indices for act_order or None.
    pub g_idx_sort_indices: Option<GpuTensor>,
    /// `[num_sms]` i32 workspace for Marlin barrier sync.
    pub workspace: GpuTensor,
    /// Input features (unquantized K dimension).
    pub size_k: usize,
    /// Output features (N dimension).
    pub size_n: usize,
    /// Quantization group size.
    pub group_size: usize,
    /// Number of groups.
    pub num_groups: usize,
    /// Whether this layer has zero points.
    pub has_zp: bool,
    /// Whether act_order (desc_act) is enabled.
    pub has_act_order: bool,
    /// Marlin b_type_id: 0 = GPTQ (uint4b8), 1 = AWQ (uint4).
    pub b_type_id: i32,
    /// Device ID for the Marlin kernel.
    pub device_id: i32,
    /// Optional bias `[size_n]` — added after Marlin GEMM.
    pub bias: Option<GpuTensor>,
}

impl MarlinLinear {
    pub fn out_features(&self) -> usize {
        self.size_n
    }

    pub fn in_features(&self) -> usize {
        self.size_k
    }
}

// ---------------------------------------------------------------------------
// Bnb4bitLinear (BitsAndBytes NF4/FP4 4-bit quantized)
// ---------------------------------------------------------------------------

/// BitsAndBytes 4-bit quantized linear layer.
///
/// Forward: dequantize packed NF4/FP4 weights to BF16 scratch buffer, then cuBLAS GEMM.
/// Matches Python `bitsandbytes.matmul_4bit` behavior.
pub struct Bnb4bitLinear {
    /// Packed NF4/FP4 nibbles `[num_packed_bytes]` U8 (2 values per byte).
    pub packed_weight: GpuTensor,
    /// Per-block scale factors `[num_blocks]` F32.
    pub absmax: GpuTensor,
    /// NF4 or FP4 lookup table `[16]` F32 on GPU.
    pub code: GpuTensor,
    /// Dequantization scratch buffer `[out_features, in_features]` BF16 on GPU.
    /// Shared across layers — the caller allocates once and passes to all layers.
    pub dequant_scratch: GpuTensor,
    /// Original output features (rows).
    pub out_features: usize,
    /// Original input features (cols).
    pub in_features: usize,
    /// Block size for quantization (typically 64).
    pub blocksize: usize,
    /// Optional bias `[out_features]`.
    pub bias: Option<GpuTensor>,
}

impl Bnb4bitLinear {
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }
}

// ---------------------------------------------------------------------------
// GgmlLinear (GGML quantized via llama.cpp kernels)
// ---------------------------------------------------------------------------

/// Quantized linear layer holding raw GGML-quantized bytes on GPU.
///
/// Forward dispatches to the appropriate dequant-matvec kernel based on GGML
/// dtype and batch size:
/// - BS=1: `dequantize_mul_mat_vec` (fused dequant + dot product)
/// - BS>1: quantize activations to Q8_1, then integer dot products
pub struct GgmlLinear {
    pub storage: GgmlStorage,
    pub bias: Option<GpuTensor>,
}

impl GgmlLinear {
    pub fn out_features(&self) -> usize {
        self.storage.nrows
    }

    pub fn in_features(&self) -> usize {
        self.storage.ncols
    }
}

// ---------------------------------------------------------------------------
// Fp8Linear (FP8 E4M3 quantized via cublasLt FP8 GEMM)
// ---------------------------------------------------------------------------

/// FP8 (E4M3) quantized linear layer.
///
/// Weights stored as FP8 `[out_features, in_features]` with a per-tensor
/// f32 weight scale on GPU. Activations are dynamically quantized to FP8
/// per-token at runtime (or statically if `input_scale` is provided).
///
/// Forward: quantize(x) → FP8 GEMM → BF16 output.
/// Matches Python vLLM's `Fp8LinearMethod`.
pub struct Fp8Linear {
    /// FP8 E4M3 weights `[out_features, in_features]`.
    pub weight: GpuTensor,
    /// Per-tensor weight scale: single f32 scalar on GPU.
    pub weight_scale: GpuTensor,
    /// Pre-calibrated input scale (static activation quantization).
    /// If `None`, uses dynamic per-token quantization.
    pub input_scale: Option<GpuTensor>,
    /// Optional bias `[out_features]` in output dtype (BF16/F16).
    pub bias: Option<GpuTensor>,
    /// Output dtype (BF16 or F16) — determines GEMM output and bias dtype.
    pub output_dtype: DType,
}

impl Fp8Linear {
    pub fn out_features(&self) -> usize {
        self.weight.dim(0)
    }

    pub fn in_features(&self) -> usize {
        self.weight.dim(1)
    }
}

// ---------------------------------------------------------------------------
// LinearLayer (enum dispatch: Dense, Marlin, Ggml, Bnb4bit, Fp8, AffineQuant)
// ---------------------------------------------------------------------------

/// Unified linear layer — dense (cuBLAS), Marlin INT4, GGML quantized,
/// BNB 4-bit, FP8, or MLX-affine INT4 (Metal-only).
///
/// Models use this everywhere they currently use `Linear`. The factory decides
/// at load time which variant to create based on weight format.
pub enum LinearLayer {
    Dense(Linear),
    Marlin(Box<MarlinLinear>),
    Ggml(Box<GgmlLinear>),
    /// Heterogeneous-dtype concat of GGUF linears. Used by
    /// `load_dense_concat_or_ggml` when the source tensors don't
    /// share a `GgmlDType` (e.g. a Q4_K_M Llama-3.2-1B has gate_proj
    /// at Q4_K but up_proj at Q6_K). Forward computes each branch
    /// separately and concatenates outputs along dim 1, producing
    /// the same `[tokens, sum(out_features)]` layout the
    /// homogeneous packed `GgmlLinear` produces — so downstream
    /// `silu_and_mul_fused` (which splits at `intermediate_size`)
    /// works without per-variant kernel changes.
    GgmlConcat(Vec<GgmlLinear>),
    Bnb4bit(Box<Bnb4bitLinear>),
    Fp8(Box<Fp8Linear>),
    Fp8Block(Box<Fp8BlockLinear>),
    /// MLX-native int4 affine quantization (Metal-only). The CUDA
    /// stack uses Marlin/AWQ/GPTQ/Bnb/Fp8 instead — affine is exclusive
    /// to the Metal backend, so this variant is never constructed on cuda
    /// (its loader is metal-gated in targets/cuda). The variant itself
    /// is unconditional here so this enum stays backend-neutral.
    AffineQuant(Box<AffineQuantLinear>),
    /// NVIDIA ModelOpt NVFP4 4-bit quantization (Metal-only). Like
    /// `AffineQuant` it's dequant-on-read; the difference is the decode
    /// (E2M1 LUT, no per-group bias) — see `Nvfp4Linear`.
    Nvfp4(Box<Nvfp4Linear>),
}

impl LinearLayer {
    /// Access the raw dense weight tensor. Panics if quantized —
    /// CUTLASS standalone GEMM only works with dense bf16 weights.
    /// Quant variants are unreachable under metal (the macro only
    /// emits Dense `LinearLayer`s on that path), so the panic arms
    /// matter only on cuda.
    pub fn dense_weight(&self) -> GpuTensor {
        match self {
            Self::Dense(l) => l.weight,
            Self::Marlin(_) => panic!("dense_weight() called on Marlin LinearLayer"),
            Self::Ggml(s) => {
                eprintln!(
                    "[dense_weight] called on Ggml LinearLayer (storage dtype={:?} \
                     shape=[{}, {}]). Call site backtrace:\n{}",
                    s.storage.dtype,
                    s.storage.nrows,
                    s.storage.ncols,
                    std::backtrace::Backtrace::force_capture()
                );
                panic!("dense_weight() called on Ggml LinearLayer");
            }
            Self::GgmlConcat(branches) => {
                eprintln!(
                    "[dense_weight] called on GgmlConcat LinearLayer ({} branches: {:?}). \
                     Call site backtrace:\n{}",
                    branches.len(),
                    branches
                        .iter()
                        .map(|b| (b.storage.dtype, b.storage.nrows, b.storage.ncols))
                        .collect::<Vec<_>>(),
                    std::backtrace::Backtrace::force_capture()
                );
                panic!("dense_weight() called on GgmlConcat LinearLayer");
            }
            Self::Bnb4bit(_) => panic!("dense_weight() called on Bnb4bit LinearLayer"),
            Self::Fp8(_) => panic!("dense_weight() called on Fp8 LinearLayer"),
            Self::Fp8Block(_) => panic!("dense_weight() called on Fp8Block LinearLayer"),
            Self::AffineQuant(_) => panic!(
                "dense_weight() called on AffineQuant LinearLayer — \
                 use affine_weight() / affine_scales() / affine_biases() instead"
            ),
            Self::Nvfp4(_) => panic!(
                "dense_weight() called on Nvfp4 LinearLayer — \
                 use nvfp4_weight() / nvfp4_scales() instead"
            ),
        }
    }

    /// Access the bias tensor from a dense layer. Returns the bias
    /// `GpuTensor` or `None` if the layer has no bias. Panics on
    /// quantized variants — solver only supports dense bf16.
    pub fn dense_bias(&self) -> Option<GpuTensor> {
        match self {
            Self::Dense(l) => l.bias,
            _ => panic!("dense_bias() called on quantized LinearLayer"),
        }
    }

    /// Cheap copy for dense layers (GpuTensor metadata only, no weight copy).
    /// Panics on quantized variants — solver only supports dense bf16.
    pub fn shallow_clone(&self) -> Self {
        match self {
            Self::Dense(l) => Self::Dense(Linear::new(l.weight, l.bias)),
            _ => panic!("shallow_clone() called on quantized LinearLayer"),
        }
    }

    pub fn out_features(&self) -> usize {
        match self {
            Self::Dense(l) => l.out_features(),
            Self::Marlin(l) => l.out_features(),
            Self::Ggml(l) => l.out_features(),
            Self::GgmlConcat(branches) => branches.iter().map(|b| b.out_features()).sum(),
            Self::Bnb4bit(l) => l.out_features(),
            Self::Fp8(l) => l.out_features(),
            Self::Fp8Block(l) => l.out_features(),
            Self::AffineQuant(l) => l.out_features(),
            Self::Nvfp4(l) => l.out_features(),
        }
    }

    pub fn in_features(&self) -> usize {
        match self {
            Self::Dense(l) => l.in_features(),
            Self::Marlin(l) => l.in_features(),
            Self::Ggml(l) => l.in_features(),
            Self::GgmlConcat(branches) => branches[0].in_features(),
            Self::Bnb4bit(l) => l.in_features(),
            Self::Fp8(l) => l.in_features(),
            Self::Fp8Block(l) => l.in_features(),
            Self::AffineQuant(l) => l.in_features(),
            Self::Nvfp4(l) => l.in_features(),
        }
    }

    /// Access the affine-quantized packed weight tensor (`[N, K / pack_factor]`
    /// U32). Panics on every other LinearLayer arm — affine accessors
    /// are meaningful only on Affine layers (Metal builds).
    pub fn affine_weight(&self) -> GpuTensor {
        match self {
            Self::AffineQuant(l) => l.weight,
            _ => panic!("affine_weight() called on non-AffineQuant LinearLayer"),
        }
    }

    /// Per-group scales (`[N, K / group_size]` F16). See `affine_weight`.
    pub fn affine_scales(&self) -> GpuTensor {
        match self {
            Self::AffineQuant(l) => l.scales,
            _ => panic!("affine_scales() called on non-AffineQuant LinearLayer"),
        }
    }

    /// Per-group affine offsets (MLX-terminology "biases" — NOT the
    /// linear-layer bias). `[N, K / group_size]` F16. See `affine_weight`.
    pub fn affine_biases(&self) -> GpuTensor {
        match self {
            Self::AffineQuant(l) => l.affine_biases,
            _ => panic!("affine_biases() called on non-AffineQuant LinearLayer"),
        }
    }

    /// Optional fp linear-layer bias on an affine-quantized layer
    /// (distinct from the per-group affine offsets). `[N]` in the
    /// model's activation dtype. Most mlx-community 4bit checkpoints
    /// don't have one (Llama-3.2 family has neither attention_bias nor
    /// mlp_bias); some Phi-3 / Qwen2 variants do.
    pub fn affine_linear_bias(&self) -> Option<GpuTensor> {
        match self {
            Self::AffineQuant(l) => l.linear_bias,
            _ => panic!("affine_linear_bias() called on non-AffineQuant LinearLayer"),
        }
    }

    pub fn affine_group_size(&self) -> u32 {
        match self {
            Self::AffineQuant(l) => l.group_size,
            _ => panic!("affine_group_size() called on non-AffineQuant LinearLayer"),
        }
    }

    pub fn affine_bits(&self) -> u32 {
        match self {
            Self::AffineQuant(l) => l.bits,
            _ => panic!("affine_bits() called on non-AffineQuant LinearLayer"),
        }
    }

    /// Access the NVFP4 packed weight tensor (`[N, K/2]` U8, two E2M1
    /// codes per byte). Panics on every other arm — like the `affine_*`
    /// accessors, meaningful only on Nvfp4 layers.
    pub fn nvfp4_weight(&self) -> GpuTensor {
        match self {
            Self::Nvfp4(l) => l.weight,
            _ => panic!("nvfp4_weight() called on non-Nvfp4 LinearLayer"),
        }
    }

    /// Per-group folded scales (`[N, K/group_size]` F16). See `nvfp4_weight`.
    pub fn nvfp4_scales(&self) -> GpuTensor {
        match self {
            Self::Nvfp4(l) => l.scales,
            _ => panic!("nvfp4_scales() called on non-Nvfp4 LinearLayer"),
        }
    }

    /// Optional fp linear-layer bias on an NVFP4 layer. `[N]` in the
    /// model's activation dtype. See `nvfp4_weight`.
    pub fn nvfp4_linear_bias(&self) -> Option<GpuTensor> {
        match self {
            Self::Nvfp4(l) => l.linear_bias,
            _ => panic!("nvfp4_linear_bias() called on non-Nvfp4 LinearLayer"),
        }
    }

    pub fn nvfp4_group_size(&self) -> u32 {
        match self {
            Self::Nvfp4(l) => l.group_size,
            _ => panic!("nvfp4_group_size() called on non-Nvfp4 LinearLayer"),
        }
    }

    pub fn nvfp4_bits(&self) -> u32 {
        match self {
            Self::Nvfp4(l) => l.bits,
            _ => panic!("nvfp4_bits() called on non-Nvfp4 LinearLayer"),
        }
    }
}

impl From<Linear> for LinearLayer {
    fn from(l: Linear) -> Self {
        Self::Dense(l)
    }
}

impl From<MarlinLinear> for LinearLayer {
    fn from(l: MarlinLinear) -> Self {
        Self::Marlin(Box::new(l))
    }
}

impl From<GgmlLinear> for LinearLayer {
    fn from(l: GgmlLinear) -> Self {
        Self::Ggml(Box::new(l))
    }
}

impl From<Fp8Linear> for LinearLayer {
    fn from(l: Fp8Linear) -> Self {
        Self::Fp8(Box::new(l))
    }
}

impl From<Fp8BlockLinear> for LinearLayer {
    fn from(l: Fp8BlockLinear) -> Self {
        Self::Fp8Block(Box::new(l))
    }
}

// ---------------------------------------------------------------------------
// Fp8BlockLinear (FP8 E4M3 with per-block scales, e.g. DeepSeek-V3)
// ---------------------------------------------------------------------------

/// FP8 block-quantized linear layer with per-block weight scales.
///
/// Weights are stored as FP8 `[out_features, in_features]` with per-block
/// scales `[ceil(N/block_n), ceil(K/block_k)]`. Forward dequantizes to BF16
/// then uses standard cuBLAS GEMM.
///
/// PERF GAP: Python uses CUTLASS block-scaled FP8 GEMM (one fused kernel)
/// or deep_gemm (Hopper). We dequant + cuBLAS which adds an extra memory
/// round-trip. Functionally correct, but slower for block-quantized models
/// like DeepSeek-V3. To close: port CUTLASS block-scaled kernel from
/// `vllm/csrc/quantization/cutlass_w8a8/`.
///
/// Matches Python vLLM's `Fp8LinearMethod` with `weight_block_size`.
pub struct Fp8BlockLinear {
    /// FP8 E4M3 weights `[out_features, in_features]`.
    pub weight: GpuTensor,
    /// Per-block weight scale inverse: `[ceil(N/block_n), ceil(K/block_k)]` f32.
    pub weight_scale_inv: GpuTensor,
    /// Block quantization block size `[block_n, block_k]`.
    pub block_size: [usize; 2],
    /// Optional bias `[out_features]`.
    pub bias: Option<GpuTensor>,
    /// Output dtype (BF16 or F16).
    pub output_dtype: DType,
}

impl Fp8BlockLinear {
    pub fn out_features(&self) -> usize {
        self.weight.dim(0)
    }

    pub fn in_features(&self) -> usize {
        self.weight.dim(1)
    }
}

// ---------------------------------------------------------------------------
// Fp8AnyLinear (per-tensor / per-channel `Fp8Linear` ⨁ blockwise `Fp8BlockLinear`)
// ---------------------------------------------------------------------------

/// Storage-uniform wrapper around the two FP8 linear variants. Lets
/// the codegen pick a single `weight_fn` Rust type for FP8 GEMM Impls
/// — every claim of an FP8 Impl in a model's Weights struct holds an
/// `Fp8AnyLinear`, regardless of whether that specific tile's
/// `quantization_config` declared per-tensor / per-channel scales
/// (`Std`) or per-block scales (`Block`).
///
/// `forward` matches both inner `forward`s' signature, so the
/// per-arch interpreter arm calls `(weight_fn)(wm, layer).forward(x,
/// cublas, alloc, stream)` without caring which variant is inside.
pub enum Fp8AnyLinear {
    Std(Fp8Linear),
    Block(Fp8BlockLinear),
}

impl Fp8AnyLinear {
    pub fn out_features(&self) -> usize {
        match self {
            Self::Std(l) => l.out_features(),
            Self::Block(l) => l.out_features(),
        }
    }

    pub fn in_features(&self) -> usize {
        match self {
            Self::Std(l) => l.in_features(),
            Self::Block(l) => l.in_features(),
        }
    }
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/// Token embedding lookup table.
///
/// Weight shape: `[vocab_size, hidden_size]`.
/// Forward gathers rows by token IDs.
pub struct Embedding {
    pub weight: GpuTensor, // [vocab_size, hidden_size]
}

impl Embedding {
    pub fn new(weight: GpuTensor) -> Self {
        debug_assert_eq!(weight.ndim(), 2);
        Self { weight }
    }

    pub fn vocab_size(&self) -> usize {
        self.weight.dim(0)
    }

    pub fn hidden_size(&self) -> usize {
        self.weight.dim(1)
    }
}

// ---------------------------------------------------------------------------
// RmsNorm
// ---------------------------------------------------------------------------

/// Root Mean Square Layer Normalization.
///
/// `y = x / sqrt(mean(x^2) + eps) * weight`
///
/// Weight shape: `[hidden_size]`.
pub struct RmsNorm {
    pub weight: GpuTensor, // [hidden_size]
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(weight: GpuTensor, eps: f32) -> Self {
        debug_assert_eq!(weight.ndim(), 1);
        Self { weight, eps }
    }

    pub fn hidden_size(&self) -> usize {
        self.weight.dim(0)
    }
}

/// Standard LayerNorm with weight and optional bias.
///
/// Computes: `(x - mean) / sqrt(var + eps) * weight + bias`
pub struct LayerNorm {
    pub weight: GpuTensor,       // [hidden_size]
    pub bias: Option<GpuTensor>, // [hidden_size] or None
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(weight: GpuTensor, bias: Option<GpuTensor>, eps: f32) -> Self {
        debug_assert_eq!(weight.ndim(), 1);
        if let Some(ref b) = bias {
            debug_assert_eq!(b.ndim(), 1);
            debug_assert_eq!(b.dim(0), weight.dim(0));
        }
        Self { weight, bias, eps }
    }

    pub fn hidden_size(&self) -> usize {
        self.weight.dim(0)
    }
}

// ---------------------------------------------------------------------------
// CohereLayerNorm
// ---------------------------------------------------------------------------

/// Cohere LayerNorm: full LayerNorm with mean subtraction, weight only (no bias).
///
/// `y = weight * (x - mean(x)) / sqrt(var(x) + eps)`
///
/// Used by Command R (CohereForCausalLM).
/// Weight shape: `[hidden_size]`.
pub struct CohereLayerNorm {
    pub weight: GpuTensor, // [hidden_size]
    pub eps: f32,
}

impl CohereLayerNorm {
    pub fn new(weight: GpuTensor, eps: f32) -> Self {
        debug_assert_eq!(weight.ndim(), 1);
        Self { weight, eps }
    }

    pub fn hidden_size(&self) -> usize {
        self.weight.dim(0)
    }
}

// ---------------------------------------------------------------------------
// LayerNormBias — standard PyTorch nn.LayerNorm
// ---------------------------------------------------------------------------

/// Standard `nn.LayerNorm` with both affine weight AND bias.
///
/// `y = weight * (x - mean(x)) / sqrt(var(x) + eps) + bias`
///
/// Used by every CLIP/ViT-style vision tower (Qwen2-VL, Qwen2.5-VL,
/// SigLIP, BERT, …) where the layer-norm sites carry both gamma and
/// beta. Sibling of [`CohereLayerNorm`] (which is bias-free).
pub struct LayerNormBias {
    pub weight: GpuTensor, // [hidden]
    pub bias: GpuTensor,   // [hidden]
    pub eps: f32,
}

impl LayerNormBias {
    pub fn new(weight: GpuTensor, bias: GpuTensor, eps: f32) -> Self {
        debug_assert_eq!(weight.ndim(), 1);
        debug_assert_eq!(bias.ndim(), 1);
        debug_assert_eq!(weight.dim(0), bias.dim(0));
        Self { weight, bias, eps }
    }
}

// ---------------------------------------------------------------------------
// GatedDeltaNetLayer
// ---------------------------------------------------------------------------

/// Per-layer weight bundle for a Gated-DeltaNet (linear-attention) layer of a
/// hybrid model (Qwen3.5 / Qwen3-Next). The four non-projection weights the
/// `Instruction::GatedDeltaNet` eval consumes; the in/out projections are plain
/// `gemm`s in the DSL (so quantization applies to them normally) and are NOT
/// part of this bundle.
///
/// Tensors are kept in their on-disk dtype here; the cuda eval casts them to
/// f32 at use (the `gdn_*` kernels are f32). This mirrors the
/// `DeepSeekV2MoELayer` typed-bundle pattern: one DSL weight accessor
/// (`linear_attn[layer]`) resolves to one of these via `required_weights`.
pub struct GatedDeltaNetLayer {
    /// Causal depthwise conv1d weight. HF on-disk shape `[conv_dim, 1, kernel]`
    /// (the singleton dim is dropped at use → `[conv_dim, kernel]`).
    pub conv1d: GpuTensor,
    /// Per-value-head log-decay base `A_log`, shape `[num_v_heads]`.
    pub a_log: GpuTensor,
    /// Per-value-head softplus bias `dt_bias`, shape `[num_v_heads]`.
    pub dt_bias: GpuTensor,
    /// Gated-RMSNorm weight, shape `[head_v_dim]` (norm is per value-head).
    pub norm: GpuTensor,
}

impl GatedDeltaNetLayer {
    pub fn new(conv1d: GpuTensor, a_log: GpuTensor, dt_bias: GpuTensor, norm: GpuTensor) -> Self {
        Self {
            conv1d,
            a_log,
            dt_bias,
            norm,
        }
    }
}

// ---------------------------------------------------------------------------
// Neutral load helpers
// ---------------------------------------------------------------------------
//
// Backend-agnostic weight-load logic shared by both targets' `*Ops` trait
// impls. They name only the neutral `GpuWeights<A>` access methods, so the
// load logic lives once here instead of being copied into each target crate.
// Genuinely-divergent loads (cuda-stream sharded/concat, metal direct-write
// packing, the RmsNorm keep-dtype split) stay in the target crates.

/// Packed-source fallback: when `{prefix}.weight` is missing, detect the
/// packed-source convention (`qkv_proj` / `gate_up_proj`) by `prefix`'s last
/// segment and synthesize even row-wise per-slice entries before the take.
pub fn try_synthesize_packed_slice<A: DeviceAllocator>(
    weights: &mut GpuWeights<A>,
    prefix: &str,
) -> Result<()> {
    let (parent, suffix) = match prefix.rsplit_once('.') {
        Some(split) => split,
        None => return Ok(()),
    };
    let (packed_suffix, targets): (&str, &[&str]) = match suffix {
        "q_proj" | "k_proj" | "v_proj" => ("qkv_proj", &["q_proj", "k_proj", "v_proj"]),
        "gate_proj" | "up_proj" => ("gate_up_proj", &["gate_proj", "up_proj"]),
        _ => return Ok(()),
    };
    let packed_prefix = format!("{parent}.{packed_suffix}");
    weights.synthesize_packed_row_split(&packed_prefix, targets)?;
    Ok(())
}

/// Load a dense [`Linear`] by prefix (with packed-source fallback + bias).
pub fn linear_load<A: DeviceAllocator>(
    weights: &mut GpuWeights<A>,
    prefix: &str,
) -> Result<Linear> {
    let weight_name = format!("{prefix}.weight");
    let bias_name = format!("{prefix}.bias");

    if !weights.contains(&weight_name) {
        try_synthesize_packed_slice(weights, prefix)?;
    }

    let weight = weights.take(&weight_name)?;
    let bias = if weights.contains(&bias_name) {
        Some(weights.take(&bias_name)?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

/// `LinearLayer::Dense` via [`linear_load`].
pub fn linear_layer_load_dense<A: DeviceAllocator>(
    weights: &mut GpuWeights<A>,
    prefix: &str,
) -> Result<LinearLayer> {
    Ok(LinearLayer::Dense(linear_load(weights, prefix)?))
}

/// Load an `nn.Parameter`-style weight by its verbatim safetensors key (no
/// `.weight`/`.bias` suffix). Transposes the matmul-natural `[K, N]` on-disk
/// layout to the gemm `[N, K]` convention. No bias.
pub fn linear_layer_load_raw<A: DeviceAllocator>(
    weights: &mut GpuWeights<A>,
    key: &str,
) -> Result<LinearLayer> {
    weights.transpose_2d_in_place(key)?;
    let weight = weights.take(key)?;
    Ok(LinearLayer::Dense(Linear::new(weight, None)))
}

/// Load a linear that may be dense (safetensors) or GGUF-quantized. Tries
/// `take_quantized_linear` first; falls back to [`linear_layer_load_dense`].
pub fn linear_layer_load_dense_or_ggml<A: DeviceAllocator>(
    weights: &mut GpuWeights<A>,
    prefix: &str,
) -> Result<LinearLayer> {
    let weight_name = format!("{prefix}.weight");
    if let Some(storage) = weights.take_quantized_linear(&weight_name) {
        // Optional bias — GGUF rarely ships bias on linears, but qwen-style
        // checkpoints can. `take` falls back to gguf_dense automatically.
        let bias_name = format!("{prefix}.bias");
        let bias = weights.take(&bias_name).ok();
        return Ok(LinearLayer::Ggml(Box::new(GgmlLinear { storage, bias })));
    }
    linear_layer_load_dense(weights, prefix)
}

/// Load an [`Embedding`] table by prefix.
pub fn embedding_load<A: DeviceAllocator>(
    weights: &mut GpuWeights<A>,
    prefix: &str,
) -> Result<Embedding> {
    let weight_name = format!("{prefix}.weight");
    let weight = weights.take(&weight_name)?;
    Ok(Embedding::new(weight))
}

/// Load a [`LayerNorm`] by prefix (bias if present).
pub fn layer_norm_load<A: DeviceAllocator>(
    weights: &mut GpuWeights<A>,
    prefix: &str,
    eps: f32,
) -> Result<LayerNorm> {
    let weight = weights.take(&format!("{prefix}.weight"))?;
    let bias_name = format!("{prefix}.bias");
    let bias = if weights.contains(&bias_name) {
        Some(weights.take(&bias_name)?)
    } else {
        None
    };
    Ok(LayerNorm::new(weight, bias, eps))
}

/// Load a [`GatedDeltaNetLayer`] by linear-attn prefix. `A_log` and
/// `norm.weight` are upcast to f32 because the `gdn_gating` / `gdn_rms_norm_
/// gated` kernels bind them as `float*` (Python `.float()`s A_log at use). A
/// bf16-only repack (mlx OptiQ ships the WHOLE checkpoint bf16, no fp32) would
/// otherwise have the kernel read 2-byte bf16 as 4-byte f32 → garbage A_log →
/// `exp()` overflow → NaN gate → NaN'd GDN state. Older fp32-A_log repacks
/// (e.g. the uniform 4bit checkpoint) round-trip through f32 unchanged.
/// `dt_bias` keeps its on-disk dtype — its kernel param is model-dtype `T*`.
pub fn gated_delta_net_load<A: DeviceAllocator>(
    weights: &mut GpuWeights<A>,
    prefix: &str,
) -> Result<GatedDeltaNetLayer> {
    let conv1d = weights.take_keep_dtype(&format!("{prefix}.conv1d.weight"))?;
    let a_log = weights.take_as_f32(&format!("{prefix}.A_log"))?;
    let dt_bias = weights.take_keep_dtype(&format!("{prefix}.dt_bias"))?;
    let norm = weights.take_as_f32(&format!("{prefix}.norm.weight"))?;
    Ok(GatedDeltaNetLayer::new(conv1d, a_log, dt_bias, norm))
}

/// Validate a `load_dense_concat_packed` prefix set: synthesize missing
/// per-slice entries, then check every `{p}.weight` is rank-2 with a uniform
/// `in_features` (`hidden`) and `dtype`. Returns
/// `(total_out_features, hidden, dtype, total_bytes)` for the packed buffer.
/// The neutral preamble shared by both backends' direct-pack loaders (only the
/// per-tensor byte-copy into the packed buffer differs: cuda pinned `take_cpu`
/// + `alloc_packed_from_host`, metal direct-write via `take_into_metal`).
pub fn concat_packed_plan<A: DeviceAllocator>(
    weights: &mut GpuWeights<A>,
    prefixes: &[&str],
) -> Result<(usize, usize, DType, usize)> {
    if prefixes.is_empty() {
        anyhow::bail!("load_dense_concat_packed: empty prefix list");
    }
    for p in prefixes {
        if !weights.contains(&format!("{p}.weight")) {
            try_synthesize_packed_slice(weights, p)?;
        }
    }
    let mut shapes_dtypes: Vec<(Vec<usize>, DType)> = Vec::with_capacity(prefixes.len());
    for p in prefixes {
        let weight_name = format!("{p}.weight");
        let (shape, dtype) = weights
            .tensor_info(&weight_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {weight_name}"))?;
        anyhow::ensure!(
            shape.len() == 2,
            "load_dense_concat_packed: `{weight_name}` has rank {}, expected 2",
            shape.len(),
        );
        shapes_dtypes.push((shape.to_vec(), dtype));
    }
    let hidden = shapes_dtypes[0].0[1];
    let dtype = shapes_dtypes[0].1;
    for (i, (shape, dt)) in shapes_dtypes.iter().enumerate() {
        anyhow::ensure!(
            shape[1] == hidden,
            "load_dense_concat_packed: `{}` has in_features {}, expected {}",
            prefixes[i],
            shape[1],
            hidden,
        );
        anyhow::ensure!(
            *dt == dtype,
            "load_dense_concat_packed: `{}` has dtype {:?}, expected {:?}",
            prefixes[i],
            dt,
            dtype,
        );
    }
    let total_out: usize = shapes_dtypes.iter().map(|(s, _)| s[0]).sum();
    let total_bytes = total_out * hidden * dtype.size_bytes();
    Ok((total_out, hidden, dtype, total_bytes))
}

/// Load + pack the all-or-none biases for a `load_dense_concat_packed` set.
/// Returns `None` when no prefix carries a bias; errors if some-but-not-all do.
pub fn concat_packed_bias<A: DeviceAllocator>(
    weights: &mut GpuWeights<A>,
    prefixes: &[&str],
    dtype: DType,
) -> Result<Option<GpuTensor>> {
    let bias_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.bias")).collect();
    let any_bias = bias_names.iter().any(|n| weights.contains(n));
    let all_bias = bias_names.iter().all(|n| weights.contains(n));
    anyhow::ensure!(
        !any_bias || all_bias,
        "load_dense_concat_packed: inconsistent biases across prefixes {:?}",
        prefixes,
    );
    if !all_bias {
        return Ok(None);
    }
    let mut bias_bytes: Vec<u8> = Vec::new();
    let mut total_bias_elems: usize = 0;
    for bn in &bias_names {
        let (data, shape, bdt) = weights.take_cpu(bn)?;
        anyhow::ensure!(
            bdt == dtype,
            "load_dense_concat_packed: bias `{bn}` dtype {:?} != weight dtype {:?}",
            bdt,
            dtype,
        );
        total_bias_elems += shape.iter().product::<usize>();
        bias_bytes.extend_from_slice(&data);
    }
    Ok(Some(weights.alloc_packed_from_host(
        &bias_bytes,
        &[total_bias_elems],
        dtype,
    )?))
}
