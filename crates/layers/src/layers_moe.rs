// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral MoE (Mixture of Experts) layer TYPE DEFINITIONS.
//!
//! Weights are stored as `GpuTensor` (raw GPU pointers / metadata only). The
//! struct/enum definitions and the pure `MoeRouting` borrow-bundle + the pure
//! `select_moe_block_m` heuristic live here so both `targets/{metal,spyre}` can
//! name them without a circular dependency. The `GpuWeights`-bound `load_*`
//! constructors live in the target crates as per-type `*Ops` extension traits —
//! this crate names no backend.

use scratchy_tensors::GpuTensor;

use crate::{Fp8BlockLinear, GgmlLinear, Linear};
use scratchy_quantizations::ggml_quant::GgmlStorage;

/// Dynamic BLOCK_M selection for fused MoE GEMM tiling.
///
/// Matches Python vLLM's `get_default_config` heuristic: select the smallest
/// tile size that covers the expected tokens-per-expert, reducing wasted
/// compute on zero-padded rows during decode.
pub fn select_moe_block_m(num_tokens: usize, top_k: usize, num_experts: usize) -> usize {
    let tokens_per_expert = (num_tokens * top_k) / num_experts;
    if tokens_per_expert <= 16 {
        16
    } else if tokens_per_expert <= 32 {
        32
    } else if tokens_per_expert <= 64 {
        64
    } else {
        128
    }
}

/// Routing-stage configuration shared by every fused-MoE layer (BF16, FP8
/// FP8 block, GGML). The families differ only in how they consume
/// the resulting `(topk_weights, topk_ids)` — the *selection* logic is uniform.
///
/// The three branches mirror Python vLLM's expert-selection paths:
/// * `e_score_correction_bias = None` → `topk_softmax` (Mixtral / Qwen MoE / DSv2).
/// * `Some(bias)` with `n_expert_group == 0` → flat sigmoid+bias top-k.
/// * `Some(bias)` with `n_expert_group > 0` and `topk_group > 0` → grouped
///   `noaux_tc` (DeepSeek V3 / Kimi K2). `routed_scaling_factor` is folded into
///   the unbiased sigmoid weights inside `topk_noaux_tc`.
pub struct MoeRouting<'a> {
    pub top_k: usize,
    pub renormalize: bool,
    pub e_score_correction_bias: Option<&'a GpuTensor>,
    pub n_expert_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f64,
}

// ---------------------------------------------------------------------------
// AffineFusedMoELayer (metal: MLX-affine int4 experts + dense BF16 router)
// ---------------------------------------------------------------------------

/// MLX-affine int4 fused MoE layer (Metal-only at runtime).
///
/// Mirrors the binding shape consumed by `lower_moe_step` in
/// `scratchy-forward-compiler/src/interpreter/metal/lowering.rs`:
/// * `router_gate` — dense `[num_experts, hidden_size]` BF16/F16 — fed
///   to the routing Gemm.
/// * `expert_{gate,up}_{w,scales,biases}` — stacked per-expert affine
///   tensors with `out_features = intermediate_size`,
///   `in_features = hidden_size`. `w` is packed U32
///   `[E, intermediate, ceil(hidden*bits/32)]`; `scales` and `biases`
///   are F16 `[E, intermediate, hidden / group_size]`.
/// * `expert_down_{w,scales,biases}` — same triple but with
///   `out_features = hidden_size`, `in_features = intermediate_size`.
/// * An ungated bank ([`ExpertBits::Ungated`], Nemotron-H's `switch_mlp.fc1` /
///   `fc2`) has no gate: its `fc1` is the up triple, its `fc2` the down.
///
/// The packed width follows the ceil law (bits=3 packs 8 elements per
/// 3 bytes — a continuous bitstream, not `K / (32 / bits)`). Reading
/// order in `affine_gather_qmv` shaders is `[E, N, K]`
/// — matches the per-expert flatten the loader produces.
pub struct AffineFusedMoELayer {
    /// Dense router projection `[num_experts, hidden_size]`,
    /// BF16/F16 — Mixtral / Qwen-MoE / Qwen3-MoE all keep the small
    /// router as fp (the mlx-community 4bit checkpoint quantize filter
    /// skips linears with out_features below the 1k threshold).
    pub router_gate: GpuTensor,

    /// `None` exactly for an ungated bank.
    pub expert_gate_w: Option<GpuTensor>,
    pub expert_gate_scales: Option<GpuTensor>,
    pub expert_gate_biases: Option<GpuTensor>,

    pub expert_up_w: GpuTensor,
    pub expert_up_scales: GpuTensor,
    pub expert_up_biases: GpuTensor,

    pub expert_down_w: GpuTensor,
    pub expert_down_scales: GpuTensor,
    pub expert_down_biases: GpuTensor,

    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub renormalize: bool,
    pub group_size: u32,
    pub bits: ExpertBits,
}

/// Each routed-expert projection's MLX-affine code width — and with it the bank's form, which
/// decides the projections it has: a gated bank's three (`act(gate) * up`, then down), or an
/// ungated bank's two (`fc2(act(fc1))`, mlx's `SwitchMLP`). Mixed-bit checkpoints (OptiQ) set
/// the projections of one layer apart (Nemotron-3-Super layer 8: `fc1` 4-bit, `fc2` 2-bit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertBits {
    Gated { gate: u32, up: u32, down: u32 },
    Ungated { fc1: u32, fc2: u32 },
}

impl ExpertBits {
    /// The gate projection's width; `None` for an ungated bank, which has none.
    pub const fn gate(self) -> Option<u32> {
        match self {
            Self::Gated { gate, .. } => Some(gate),
            Self::Ungated { .. } => None,
        }
    }

    /// The width of the projection the activation reads (a gated bank's up, an ungated `fc1`).
    pub const fn up(self) -> u32 {
        match self {
            Self::Gated { up, .. } => up,
            Self::Ungated { fc1, .. } => fc1,
        }
    }

    /// The out-projection's width (a gated bank's down, an ungated `fc2`).
    pub const fn down(self) -> u32 {
        match self {
            Self::Gated { down, .. } => down,
            Self::Ungated { fc2, .. } => fc2,
        }
    }
}

// ---------------------------------------------------------------------------
// Gemma-4 MoE (gemma_moe op): separate router + router-less SwitchGLU experts
// ---------------------------------------------------------------------------

/// Gemma-4-26B-A4B router weight bundle (base `router`, Metal-only at runtime).
///
/// Three on-disk leaves under `...layers.N.router`:
/// * `router.proj` — `[num_experts, hidden]` **8-bit** MLX-affine; dequantized
///   to a dense `gate` tensor at load (the router GEMM stays on the dense
///   BF16/F16 path, like `AffineFusedMoELayer::router_gate`).
/// * `router.per_expert_scale` — `[num_experts]` bf16, gather-multiplied into
///   the softmaxed top-k weights (the NEW `moe_per_expert_scale` step).
/// * `router.scale` — `[hidden]` bf16 RMSNorm gain applied to the router input
///   (the post-attention residual) before the router GEMM. Plain `router.scale`;
///   the `hidden^-0.5` temperature is folded into the softmax, not the gain.
pub struct GemmaRouterLayer {
    /// Dense `[num_experts, hidden]` router projection, dequantized from the
    /// 8-bit affine `router.proj` at load. Read by the router `Gemm` step.
    pub gate: GpuTensor,
    /// `[num_experts]` bf16 per-expert score scale. Read by the
    /// `moe_per_expert_scale` gather-multiply after the softmax.
    pub per_expert_scale: GpuTensor,
    /// `[hidden]` bf16 RMSNorm gain on the router input.
    pub scale: GpuTensor,
    pub num_experts: usize,
    pub hidden_size: usize,
}

/// Gemma-4-26B-A4B SwitchGLU expert weight bundle (base `experts.switch_glu`,
/// Metal-only at runtime). Same stacked-per-expert MLX-affine int4 layout as
/// [`AffineFusedMoELayer`]'s routed experts, but with NO router gate (the router
/// is the separate [`GemmaRouterLayer`] bundle) and `gate/up/down_proj` naming
/// directly under `...layers.N.experts.switch_glu` (NOT `switch_mlp`).
///
/// `gate/up`: `[num_experts, moe_inter, hidden / div]`; `down`:
/// `[num_experts, hidden, moe_inter / div]`. 4-bit, group_size 64.
pub struct SwitchGluExpertsLayer {
    pub expert_gate_w: GpuTensor,
    pub expert_gate_scales: GpuTensor,
    pub expert_gate_biases: GpuTensor,

    pub expert_up_w: GpuTensor,
    pub expert_up_scales: GpuTensor,
    pub expert_up_biases: GpuTensor,

    pub expert_down_w: GpuTensor,
    pub expert_down_scales: GpuTensor,
    pub expert_down_biases: GpuTensor,

    pub num_experts: usize,
    pub top_k: usize,
    pub moe_intermediate_size: usize,
    pub hidden_size: usize,
    pub group_size: u32,
    pub bits: u32,

    /// Native INT4 (Marlin) expert storage. No in-tree forward reads these;
    /// every loader sets them to `None` / `0`.
    pub marlin_w1: Option<GpuTensor>,
    pub marlin_w2: Option<GpuTensor>,
    pub marlin_w1_scales: Option<GpuTensor>,
    pub marlin_w2_scales: Option<GpuTensor>,
    pub marlin_workspace: Option<GpuTensor>,
    pub marlin_b_type_id: i32,
}

// ---------------------------------------------------------------------------
// GptOssMoELayer (gptoss_moe op): biased router + SwiGLU-OAI experts with
// per-expert LINEAR biases
// ---------------------------------------------------------------------------

/// gpt-oss (20b / 120b) fused MoE weight bundle (base `mlp`, Metal-only at
/// runtime).
///
/// Distinct from [`AffineFusedMoELayer`] — which it otherwise mirrors in its
/// expert storage — in three ways the gpt-oss forward needs:
/// * the router is linear WITH bias: `router_gate` `[E, hidden]` (dequantized
///   from the affine `{prefix}.router` at load, at the router's own bit width)
///   plus `router_bias` F32 `[E]` added to the logits before top-k, with the
///   BIASED values read back as the gathered scores;
/// * the experts are SwiGLU-OAI (`y = (clamp(u, ±7) + 1) · min(g, 7) ·
///   sigmoid(1.702 · g)`) with per-expert LINEAR biases — the checkpoint ships
///   dense `mlp.experts.{gate,up,down}_proj.bias`, concatenated here into
///   `gate_up_linear_bias` `[E, 2*inter]` (gate rows then up rows) and
///   `down_linear_bias` `[E, hidden]`, added inside the gather kernels;
/// * the expert stacks arrive as SINGLE pre-stacked 2-D tensors
///   `[(E·N), K/div]` on disk, which the loader restacks (zero-copy view)
///   into the `[E, N, K/div]` reading order the shaders walk.
///
/// `top_k` picks; the post-gather normalizer is a softmax over the top-4
/// (no renorm), matching HF's `GptOssSparseMoeBlock`.
pub struct GptOssMoELayer {
    /// Dense router projection `[num_experts, hidden_size]`, dequantized
    /// from the affine `{prefix}.router` at load.
    pub router_gate: GpuTensor,
    /// F32 `[num_experts]` router linear bias (converted from the
    /// checkpoint's BF16 at load). Added to the logits before top-k AND
    /// read back as the gathered score.
    pub router_bias: GpuTensor,

    pub expert_gate_w: GpuTensor,
    pub expert_gate_scales: GpuTensor,
    pub expert_gate_biases: GpuTensor,

    pub expert_up_w: GpuTensor,
    pub expert_up_scales: GpuTensor,
    pub expert_up_biases: GpuTensor,

    pub expert_down_w: GpuTensor,
    pub expert_down_scales: GpuTensor,
    pub expert_down_biases: GpuTensor,

    /// Dense `[E, 2*intermediate_size]` per-expert gate/up LINEAR biases,
    /// gate rows then up rows — added by the gate/up gather kernel.
    pub gate_up_linear_bias: GpuTensor,
    /// Dense `[E, hidden_size]` per-expert down-proj LINEAR biases — added
    /// by the down/combine gather kernel.
    pub down_linear_bias: GpuTensor,

    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub group_size: u32,
    pub bits: u32,
}

// ---------------------------------------------------------------------------
// FusedMoELayer (enum: Affine for metal)
// ---------------------------------------------------------------------------

/// Unified fused-MoE layer. Mirrors the [`crate::LinearLayer`]
/// precedent: one public enum that holds the per-backend storage. The
/// macro-emitted `WeightAccessors::fused_moe_at` returns a reference
/// to this type, and the per-target Impls match on the variant they
/// expect (the Metal `MetalFusedMoeImpl` fires on MLX-affine int4
/// checkpoints).
pub enum FusedMoELayer {
    /// MLX-affine int4 MoE (Metal-only). Per-expert (W, scales,
    /// biases) triples + dense fp router; lowered to a 10-step ICB
    /// decomposition by `lower_moe_step`.
    Affine(Box<AffineFusedMoELayer>),
}

// ---------------------------------------------------------------------------
// AffineSharedFusedMoELayer (metal: MLX-affine int4 Qwen-MoE family)
// ---------------------------------------------------------------------------

/// MLX-affine int4 fused MoE + shared expert layer (Metal-only at runtime).
///
/// Same stacked-per-expert affine layout as [`AffineFusedMoELayer`]
/// (Qwen naming: `gate_proj/up_proj/down_proj`), plus optional shared
/// expert tail used by Qwen2-MoE / Qwen3-MoE.
///
/// `shared_*` fields are `None` when `shared_intermediate_size == 0`
/// (Qwen3-MoE-30B-A3B-Instruct ships shared_inter=0). The Metal
/// lowering pass conditions its shared-expert tail on the same flag,
/// so an absent shared expert is structurally consistent — the
/// lowered tape doesn't reference these slots.
pub struct AffineSharedFusedMoELayer {
    /// Routed experts + dense router. Same shape as
    /// [`AffineFusedMoELayer`].
    pub routed: AffineFusedMoELayer,

    // ── Shared expert (Option, all-or-none) ───────────────────────
    /// Shared expert fused gate+up `[2*shared_inter, hidden]`, packed
    /// U32 weights. Lowering's `MoeSharedGateUpW` resolves to this.
    pub shared_gate_up_w: Option<GpuTensor>,
    /// Shared expert gate+up scales `[2*shared_inter, hidden / gs]`,
    /// F16. Lowering's `MoeSharedGateUpS` resolves to this.
    pub shared_gate_up_scales: Option<GpuTensor>,
    /// Shared expert gate+up affine biases. F16. `MoeSharedGateUpB`.
    pub shared_gate_up_biases: Option<GpuTensor>,

    /// Shared expert down `[hidden, shared_inter]` packed U32.
    /// `MoeSharedDownW`.
    pub shared_down_w: Option<GpuTensor>,
    pub shared_down_scales: Option<GpuTensor>,
    pub shared_down_biases: Option<GpuTensor>,

    /// Shared expert sigmoid gate `[1, hidden]` dense F16/BF16.
    /// `MoeSharedExpertGate`.
    pub shared_expert_gate: Option<GpuTensor>,

    /// `[num_experts]` F32 `e_score_correction_bias` for sigmoid routing
    /// (GLM-4 / DeepSeek-V3 `noaux_tc` at `n_group = 1`): the BIASED
    /// sigmoids order the top-k picks, the UNBIASED sigmoids are the
    /// scores the experts scale by. `None` on every softmax-routed MoE
    /// (Qwen, Mixtral).
    pub e_score_correction_bias: Option<GpuTensor>,

    pub shared_intermediate_size: usize,
}

// ---------------------------------------------------------------------------
// SharedFusedMoELayer (enum: Affine for metal)
// ---------------------------------------------------------------------------

pub enum SharedFusedMoELayer {
    Affine(Box<AffineSharedFusedMoELayer>),
}

// ---------------------------------------------------------------------------
// DeepSeekV2MoELayer
// ---------------------------------------------------------------------------

/// MoE layer for DeepSeek V2 / V3. Two differences from `SharedFusedMoELayer`:
/// 1. Shared expert uses a **plain add** (no sigmoid gate).
/// 2. Routed output is multiplied by `routed_scaling_factor` before the add.
///
/// Forward: `output = routed_scaling_factor * moe(x) + shared_expert(x)`
pub struct DeepSeekV2MoELayer {
    pub moe: FusedMoELayer,
    /// Shared expert fused gate+up: `[2 * shared_inter, hidden_size]`.
    pub shared_gate_up: Linear,
    /// Shared expert down: `[hidden_size, shared_inter]`.
    pub shared_down: Linear,
    pub shared_intermediate_size: usize,
    pub routed_scaling_factor: f32,
}

// ---------------------------------------------------------------------------
// DeepSeekV2Fp8BlockMoELayer
// ---------------------------------------------------------------------------

/// FP8 block-quantized analog of [`DeepSeekV2MoELayer`]. Used by DeepSeek-V3
/// and Kimi K2 official checkpoints (FP8 E4M3 with 128×128 weight-block scales).
///
/// Forward shape is identical to `DeepSeekV2MoELayer::forward` — the only
/// change is the storage of the routed experts (`Fp8BlockFusedMoELayer`) and
/// the shared expert (`Fp8BlockLinear`).
pub struct DeepSeekV2Fp8BlockMoELayer {
    pub moe: Fp8BlockFusedMoELayer,
    /// Shared expert fused gate+up: concatenated `Fp8BlockLinear`
    /// `[2 * shared_inter, hidden_size]`.
    pub shared_gate_up: Fp8BlockLinear,
    /// Shared expert down: `Fp8BlockLinear` `[hidden_size, shared_inter]`.
    pub shared_down: Fp8BlockLinear,
    pub shared_intermediate_size: usize,
    pub routed_scaling_factor: f32,
}

// ---------------------------------------------------------------------------
// Fp8BlockFusedMoELayer (FP8 block-quantized MoE)
// ---------------------------------------------------------------------------

/// Fused MoE layer with FP8 E4M3 block-quantized weights.
///
/// Stacked FP8 E4M3 routed experts with per-block weight scales.
pub struct Fp8BlockFusedMoELayer {
    pub gate: Linear,
    /// Stacked gate+up weights: `[E, 2*inter, hidden]` FP8 E4M3.
    pub w1: GpuTensor,
    /// Stacked down weights: `[E, hidden, inter]` FP8 E4M3.
    pub w2: GpuTensor,
    /// Block scales for w1: `[E, ceil(2*inter/bn), ceil(hidden/bk)]` f32.
    pub w1_scale_inv: GpuTensor,
    /// Block scales for w2: `[E, ceil(hidden/bn), ceil(inter/bk)]` f32.
    pub w2_scale_inv: GpuTensor,
    /// Quantization block size `[block_n, block_k]`.
    pub block_size: [usize; 2],
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub renormalize: bool,
    /// `[num_experts]` F32 e_score_correction_bias for sigmoid routing
    /// (DeepSeek V3 / Kimi K2). `None` ⇒ softmax routing.
    pub e_score_correction_bias: Option<GpuTensor>,
    pub n_expert_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f64,
}

// ---------------------------------------------------------------------------
// GgmlFusedMoELayer (GGML quantized MoE)
// ---------------------------------------------------------------------------

/// Fused Mixture of Experts layer with GGML-quantized expert weights.
///
/// Uses the `indexed_moe_forward` kernels which handle expert routing
/// internally via an index array: `blockIdx.y = batch_idx`, `blockIdx.z = topk_idx`,
/// expert looked up from `indices[batch * topk + topk_idx]`.
///
/// Forward pass:
/// 1. Gate → router logits (dense matmul)
/// 2. topk_softmax → topk_weights, topk_ids
/// 3. Quantize hidden_states to Q8_1
/// 4. indexed_moe_forward(w1, q8_input, indices) → [batch*topk, 2*inter] f32
/// 5. silu_and_mul → [batch*topk, inter] f32
/// 6. Quantize activated to Q8_1
/// 7. indexed_moe_forward(w2, q8_activated, indices) → [batch*topk, hidden] f32
/// 8. Scale by topk_weights, sum across topk → [batch, hidden] f32
pub struct GgmlFusedMoELayer {
    /// Gate projection: `[num_experts, hidden_size]` dense.
    /// llama.cpp converters store the router gate as F32 unconditionally
    /// (even in Q4_K_M GGUFs), so the scratchy GGUF loader puts it in
    /// `gguf_dense` already cast to model dtype — a plain dense `Linear`
    /// is the right type. (FP8/AWQ/etc. router gates are similarly dense.)
    pub gate: Linear,
    /// Stacked gate+up weights: `[num_experts, 2*intermediate_size, hidden_size]` quantized.
    /// GgmlStorage with nrows = num_experts * 2 * intermediate_size, ncols = hidden_size.
    pub w1: GgmlStorage,
    /// Stacked down weights: `[num_experts, hidden_size, intermediate_size]` quantized.
    /// GgmlStorage with nrows = num_experts * hidden_size, ncols = intermediate_size.
    pub w2: GgmlStorage,
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub renormalize: bool,
    /// `[num_experts]` F32 e_score_correction_bias for sigmoid routing
    /// (DeepSeek V3 / Kimi K2). `None` ⇒ softmax routing.
    pub e_score_correction_bias: Option<GpuTensor>,
    pub n_expert_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f64,
}

// ---------------------------------------------------------------------------
// DeepSeekV2GgmlMoELayer (GGML-quantized DeepSeek MoE)
// ---------------------------------------------------------------------------

/// GGUF/GGML-quantized analog of [`DeepSeekV2MoELayer`]. Used for any
/// DeepSeek-family GGUF (V2-Lite, Moonlight V3-flat, K2 GGUFs).
///
/// Forward shape mirrors `DeepSeekV2MoELayer::forward` — only the
/// expert storage changes (`GgmlFusedMoELayer` for routed experts;
/// `GgmlLinear` for shared gate_up + down).
pub struct DeepSeekV2GgmlMoELayer {
    pub moe: GgmlFusedMoELayer,
    /// Shared expert fused gate+up: concatenated `GgmlLinear`
    /// `[2 * shared_inter, hidden_size]`.
    pub shared_gate_up: GgmlLinear,
    /// Shared expert down: `GgmlLinear` `[hidden_size, shared_inter]`.
    pub shared_down: GgmlLinear,
    pub shared_intermediate_size: usize,
    pub routed_scaling_factor: f32,
}
