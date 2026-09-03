// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral MoE (Mixture of Experts) layer TYPE DEFINITIONS.
//!
//! Weights are stored as `GpuTensor` (raw GPU pointers / metadata only). The
//! struct/enum definitions and the pure `MoeRouting` borrow-bundle + the pure
//! `select_moe_block_m` heuristic live here so both `targets/{cuda,metal}` can
//! name them without a circular dependency. The kernel-calling `forward` passes
//! and the `GpuWeights`-bound `load_*` constructors live in `targets/cuda` as
//! per-type `*Ops` extension traits — this crate names no backend.

use scratchy_tensors::GpuTensor;

use crate::{Fp8BlockLinear, GgmlLinear, Linear, LinearLayer};
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

/// Routing-stage configuration shared by every fused-MoE layer (BF16, FP8 scalar,
/// FP8 block, Marlin, GGML). The four families differ only in how they consume
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
// DenseFusedMoELayer (cuda Dense MoE: Mixtral / Qwen2-MoE / Qwen3-MoE BF16)
// ---------------------------------------------------------------------------

/// Cuda-side Dense Mixture of Experts layer. The variant of the public
/// [`FusedMoELayer`] enum that holds stacked BF16/F16 expert weights
/// fed through `kernels::fused_moe_gemm`.
///
/// Weights are stored as stacked `[num_experts, dim, hidden]` tensors.
/// Forward matches Python vLLM's `fused_experts_impl` exactly.
pub struct DenseFusedMoELayer {
    /// Gate projection: `[hidden_size, num_experts]`.
    pub gate: Linear,
    /// Stacked gate+up weights: `[num_experts, 2*intermediate_size, hidden_size]`.
    pub w1: GpuTensor,
    /// Stacked down weights: `[num_experts, hidden_size, intermediate_size]`.
    pub w2: GpuTensor,
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub renormalize: bool,
    /// `[num_experts]` F32 e_score_correction_bias for sigmoid routing (DeepSeek V3 / Kimi K2).
    /// When `Some`, uses sigmoid top-k with bias instead of softmax top-k.
    pub e_score_correction_bias: Option<GpuTensor>,
    /// Number of expert groups for `noaux_tc` grouped routing (DeepSeek V3).
    /// 0 = no group selection (flat top-k).
    pub n_expert_group: usize,
    /// Number of groups to select in grouped routing. 0 = disabled.
    pub topk_group: usize,
    /// Scaling factor applied to routing weights (default 1.0).
    pub routed_scaling_factor: f64,
}

// ---------------------------------------------------------------------------
// AffineFusedMoELayer (metal: MLX-affine int4 experts + dense BF16 router)
// ---------------------------------------------------------------------------

/// MLX-affine int4 fused MoE layer (Metal-only at runtime).
///
/// Mirrors the binding shape consumed by `lower_metal_moe` in
/// `scratchy-forward-compiler/src/interpreter/metal/lowering.rs`:
/// * `router_gate` — dense `[num_experts, hidden_size]` BF16/F16 — fed
///   to the routing Gemm.
/// * `expert_{gate,up}_{w,scales,biases}` — stacked per-expert affine
///   tensors with `out_features = intermediate_size`,
///   `in_features = hidden_size`. `w` is packed U32
///   `[E, intermediate, hidden / pack_factor]`; `scales` and `biases`
///   are F16 `[E, intermediate, hidden / group_size]`.
/// * `expert_down_{w,scales,biases}` — same triple but with
///   `out_features = hidden_size`, `in_features = intermediate_size`.
///
/// `pack_factor = 32 / bits = 8` for bits=4 (the only configuration
/// we ship). Reading order in `affine_gather_qmv` shaders is `[E, N, K]`
/// — matches the per-expert flatten the loader produces.
pub struct AffineFusedMoELayer {
    /// Dense router projection `[num_experts, hidden_size]`,
    /// BF16/F16 — Mixtral / Qwen-MoE / Qwen3-MoE all keep the small
    /// router as fp (the mlx-community 4bit checkpoint quantize filter
    /// skips linears with out_features below the 1k threshold).
    pub router_gate: GpuTensor,

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
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub renormalize: bool,
    pub group_size: u32,
    pub bits: u32,
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

    /// Native INT4 (Marlin) expert path — the cuda `gemma_moe_forward` uses
    /// these instead of the dequant'd bf16 `expert_*_w` stacks when present
    /// (cyankiwi `gemma-4-26B-A4B-it-AWQ-4bit`, compressed-tensors INT4). Keeps
    /// experts INT4 in VRAM (~0.4 GiB) instead of bf16 (~43 GiB). `w1` is the
    /// fused gate+up stacked Marlin weight `[E, hidden*2*moe_inter/8]`; `w2` is
    /// down `[E, moe_inter*hidden/8]`; scales are `[E, num_groups, N]`; the
    /// workspace is the Marlin barrier-lock buffer. `None` on the bf16 / MLX /
    /// metal paths. `marlin_b_type_id` = 0 (kU4B8, compressed-tensors sym).
    pub marlin_w1: Option<GpuTensor>,
    pub marlin_w2: Option<GpuTensor>,
    pub marlin_w1_scales: Option<GpuTensor>,
    pub marlin_w2_scales: Option<GpuTensor>,
    pub marlin_workspace: Option<GpuTensor>,
    pub marlin_b_type_id: i32,
}

// ---------------------------------------------------------------------------
// FusedMoELayer (enum: Dense for cuda, Affine for metal)
// ---------------------------------------------------------------------------

/// Unified fused-MoE layer. Mirrors the [`crate::LinearLayer`]
/// precedent: one public enum that holds the per-backend storage. The
/// macro-emitted `WeightAccessors::fused_moe_at` returns a reference
/// to this type, and the per-target Impls match on the variant they
/// expect (the cuda `FusedMoeRefImpl` only fires on Dense MoE
/// checkpoints; the Metal `MetalFusedMoeImpl` only fires on
/// MLX-affine int4 checkpoints).
pub enum FusedMoELayer {
    /// Cuda Dense BF16/F16 MoE — Mixtral / Qwen-MoE / DeepSeek-MoE
    /// reference path. Stacked `[E, 2*inter, hidden]` gate+up plus
    /// `[E, hidden, inter]` down, fed through
    /// `kernels::fused_moe_gemm`.
    Dense(Box<DenseFusedMoELayer>),
    /// MLX-affine int4 MoE (Metal-only). Per-expert (W, scales,
    /// biases) triples + dense fp router; lowered to a 10-step ICB
    /// decomposition by `lower_metal_moe`.
    Affine(Box<AffineFusedMoELayer>),
}

// ---------------------------------------------------------------------------
// DenseSharedFusedMoELayer (cuda BF16 Qwen2/3 MoE with shared expert)
// ---------------------------------------------------------------------------

/// Cuda-side Dense MoE layer with optional shared expert (Qwen2 MoE,
/// Qwen3 MoE). The variant of the public [`SharedFusedMoELayer`] enum
/// holding the stacked dense expert weights + dense shared expert.
///
/// The shared expert runs in parallel with the MoE routing:
/// ```text
/// output = moe(hidden_states) + shared_expert_gate(hidden_states).sigmoid() * shared_expert(hidden_states)
/// ```
pub struct DenseSharedFusedMoELayer {
    pub moe: FusedMoELayer,
    /// Shared expert: fused gate+up projection `[2*intermediate, hidden]`.
    pub shared_gate_up: Option<Linear>,
    /// Shared expert: down projection `[hidden, intermediate]`.
    pub shared_down: Option<Linear>,
    /// Shared expert gate: `[1, hidden]` — sigmoid gate for shared expert output.
    pub shared_expert_gate: Option<Linear>,
    pub intermediate_size: usize,
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

    pub shared_intermediate_size: usize,
}

// ---------------------------------------------------------------------------
// SharedFusedMoELayer (enum: Dense for cuda, Affine for metal)
// ---------------------------------------------------------------------------

pub enum SharedFusedMoELayer {
    Dense(Box<DenseSharedFusedMoELayer>),
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
// Fp8FusedMoELayer (FP8 E4M3 quantized MoE)
// ---------------------------------------------------------------------------

/// Fused Mixture of Experts layer with FP8 E4M3 quantized weights.
///
/// Forward pass matches Python vLLM's `fused_experts_impl` with `use_fp8_w8a8=True`:
/// 1. Gate → router logits (dense BF16)
/// 2. topk_softmax → topk weights + ids
/// 3. scaled_fp8_quant_dynamic(hidden) → FP8 input + per-token scales
/// 4. moe_align_block_size
/// 5. fused_moe_fp8_gemm (GEMM 1: gate+up)
/// 6. silu_and_mul activation
/// 7. scaled_fp8_quant_dynamic(activated) → FP8 act + per-token scales
/// 8. fused_moe_fp8_gemm (GEMM 2: down, with routing weight)
/// 9. moe_sum → reduced output
pub struct Fp8FusedMoELayer {
    /// Gate projection: `[hidden_size, num_experts]` — always dense BF16.
    pub gate: Linear,
    /// Stacked gate+up weights: `[num_experts, 2*intermediate_size, hidden_size]` FP8 E4M3.
    pub w1: GpuTensor,
    /// Stacked down weights: `[num_experts, hidden_size, intermediate_size]` FP8 E4M3.
    pub w2: GpuTensor,
    /// Per-expert w1 scales: `[num_experts]` f32.
    pub w1_scale: GpuTensor,
    /// Per-expert w2 scales: `[num_experts]` f32.
    pub w2_scale: GpuTensor,
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub renormalize: bool,
    /// `[num_experts]` F32 e_score_correction_bias for sigmoid routing (DeepSeek V3 / Kimi K2).
    /// `None` ⇒ softmax routing (Mixtral / Qwen MoE).
    pub e_score_correction_bias: Option<GpuTensor>,
    /// 0 ⇒ flat top-k. >0 with `topk_group` >0 ⇒ noaux_tc grouped routing.
    pub n_expert_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f64,
}

// ---------------------------------------------------------------------------
// Fp8BlockFusedMoELayer (FP8 block-quantized MoE)
// ---------------------------------------------------------------------------

/// Fused MoE layer with FP8 E4M3 block-quantized weights.
///
/// Same pipeline as `Fp8FusedMoELayer` but uses per-block weight scales
/// applied during the FP8→BF16 dequant step in the CUDA kernel.
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
    /// See [`Fp8FusedMoELayer::e_score_correction_bias`].
    pub e_score_correction_bias: Option<GpuTensor>,
    pub n_expert_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f64,
}

// ---------------------------------------------------------------------------
// Fp8SharedFusedMoELayer (Qwen2/3 MoE with FP8 experts)
// ---------------------------------------------------------------------------

/// FP8 MoE layer with optional shared expert.
pub struct Fp8SharedFusedMoELayer {
    pub moe: Fp8FusedMoELayer,
    /// Shared expert: fused gate+up projection.
    pub shared_gate_up: Option<Linear>,
    /// Shared expert: down projection.
    pub shared_down: Option<Linear>,
    /// Shared expert gate: `[1, hidden]` — sigmoid gate for shared expert output.
    pub shared_expert_gate: Option<Linear>,
    pub intermediate_size: usize,
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
    /// See [`Fp8FusedMoELayer::e_score_correction_bias`].
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

// ---------------------------------------------------------------------------
// MarlinFusedMoELayer (AWQ/GPTQ INT4 quantized MoE)
// ---------------------------------------------------------------------------

/// Fused Mixture of Experts layer using Marlin INT4 quantized weights.
///
/// Uses the `marlin_moe_gemm` kernel which fuses expert routing + Marlin
/// INT4 GEMM + topk weight multiply into a single kernel per pass.
///
/// Forward matches Python vLLM's `_fused_marlin_moe` two-pass pattern.
pub struct MarlinFusedMoELayer {
    /// Gate projection: `[hidden_size, num_experts]` — always dense.
    pub gate: Linear,
    /// Stacked gate+up weights: `[E, K/tile, 2N*tile]` Marlin-packed.
    pub w1: GpuTensor,
    /// Stacked down weights: `[E, N/tile, K*tile]` Marlin-packed.
    pub w2: GpuTensor,
    /// Scales for w1: `[E, num_groups_w1, 2*intermediate_size]`.
    pub w1_scales: GpuTensor,
    /// Scales for w2: `[E, num_groups_w2, hidden_size]`.
    pub w2_scales: GpuTensor,
    /// Zero points for w1 (AWQ only): `[E, num_groups_w1, 2*intermediate/8]`.
    pub w1_zeros: Option<GpuTensor>,
    /// Zero points for w2 (AWQ only): `[E, num_groups_w2, hidden/8]`.
    pub w2_zeros: Option<GpuTensor>,
    /// Workspace for barrier synchronization: `[sms * 4]` i32.
    pub workspace: GpuTensor,
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub group_size: usize,
    pub has_zp: bool,
    /// 1 = kU4 (AWQ), 0 = kU4B8 (GPTQ).
    pub b_type_id: i32,
    pub renormalize: bool,
    /// See [`Fp8FusedMoELayer::e_score_correction_bias`].
    pub e_score_correction_bias: Option<GpuTensor>,
    pub n_expert_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f64,
}

// ---------------------------------------------------------------------------
// MarlinSharedFusedMoELayer
// ---------------------------------------------------------------------------

/// MoE layer with optional shared expert, using Marlin quantized weights.
pub struct MarlinSharedFusedMoELayer {
    pub moe: MarlinFusedMoELayer,
    /// Shared expert: fused gate+up (Marlin-quantized or dense Linear).
    pub shared_gate_up: Option<LinearLayer>,
    /// Shared expert: down projection.
    pub shared_down: Option<LinearLayer>,
    /// Shared expert gate: `[1, hidden]` — sigmoid gate for shared expert output.
    pub shared_expert_gate: Option<Linear>,
    pub intermediate_size: usize,
}
