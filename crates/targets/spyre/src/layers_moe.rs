// SPDX-License-Identifier: Apache-2.0
// The MoE loader `*Ops` mirror cuda/metal's multi-arg `load` signatures.
#![allow(clippy::too_many_arguments)]
//! Spyre (KTIR) MoE (Mixture of Experts) load surface.
//!
//! Spyre is a host fp16 backend with no MoE GEMM / topk kernels yet, so every
//! MoE *load* here is a clean runtime `bail!` — a bf16 (non-MoE) checkpoint
//! never reaches these methods, but the `#[forward]` macro's shared
//! `Weights::load` body names them by exact path for every model variant, so
//! they must compile with the cuda/metal trait + method *signatures* verbatim.
//!
//! The MoE type *definitions* (bare field bags) + the pure `MoeRouting` borrow
//! bundle + `select_moe_block_m` live in the cfg-free `scratchy-layers` crate;
//! re-exported here via the glob below so `crate::__gpu::layers_moe::<T>` paths
//! resolve. The kernel-bound `load_*` runtime methods are per-type `*Ops`
//! extension traits, matching `scratchy-target-{cuda,metal}/src/layers_moe.rs`.
//!
//! Trait + method names/signatures are kept identical to the cuda/metal peers
//! (the macro emits `crate::__gpu::layers_moe::<Type>::<method>(...)` by exact
//! path) — the only divergence is the body, which bails instead of allocating
//! and stacking expert weights.

pub use scratchy_layers::layers_moe::*;

// ---------------------------------------------------------------------------
// FusedMoELayer (enum: Dense for cuda, Affine for metal — neither built here)
// ---------------------------------------------------------------------------

pub trait FusedMoEOps {
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        intermediate_size: usize,
        hidden_size: usize,
        stream: crate::LoadStream,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn load_affine(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        intermediate_size: usize,
        hidden_size: usize,
        group_size: u32,
        bits: u32,
        gate_bits: u32,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
}

impl FusedMoEOps for FusedMoELayer {
    /// Macro-emitted load entry for Dense BF16 MoE checkpoints (spyre).
    #[allow(clippy::too_many_arguments)]
    fn load(
        _gw: &mut crate::weights::GpuWeights,
        _prefix: &str,
        _num_experts: usize,
        _top_k: usize,
        _intermediate_size: usize,
        _hidden_size: usize,
        _stream: crate::LoadStream,
    ) -> anyhow::Result<Self> {
        anyhow::bail!("scratchy-target-spyre: FusedMoELayer load not yet implemented")
    }

    /// Macro-emitted load entry for MLX-affine int4 MoE checkpoints (spyre).
    #[allow(clippy::too_many_arguments)]
    fn load_affine(
        _gw: &mut crate::weights::GpuWeights,
        _prefix: &str,
        _num_experts: usize,
        _top_k: usize,
        _intermediate_size: usize,
        _hidden_size: usize,
        _group_size: u32,
        _bits: u32,
        _gate_bits: u32,
    ) -> anyhow::Result<Self> {
        anyhow::bail!("scratchy-target-spyre: FusedMoELayer affine load not yet implemented")
    }
}

// ---------------------------------------------------------------------------
// SharedFusedMoELayer (enum: Dense for cuda, Affine for metal)
// ---------------------------------------------------------------------------

pub trait SharedFusedMoEOps {
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        shared_expert_intermediate_size: usize,
        hidden_size: usize,
        stream: crate::LoadStream,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn load_affine(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        shared_expert_intermediate_size: usize,
        hidden_size: usize,
        group_size: u32,
        bits: u32,
        gate_bits: u32,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
}

impl SharedFusedMoEOps for SharedFusedMoELayer {
    /// Macro-emitted load entry for Dense BF16 Qwen-MoE checkpoints (spyre).
    #[allow(clippy::too_many_arguments)]
    fn load(
        _gw: &mut crate::weights::GpuWeights,
        _prefix: &str,
        _num_experts: usize,
        _top_k: usize,
        _moe_intermediate_size: usize,
        _shared_expert_intermediate_size: usize,
        _hidden_size: usize,
        _stream: crate::LoadStream,
    ) -> anyhow::Result<Self> {
        anyhow::bail!("scratchy-target-spyre: SharedFusedMoELayer load not yet implemented")
    }

    /// Macro-emitted load entry for MLX-affine int4 Qwen-MoE checkpoints (spyre).
    #[allow(clippy::too_many_arguments)]
    fn load_affine(
        _gw: &mut crate::weights::GpuWeights,
        _prefix: &str,
        _num_experts: usize,
        _top_k: usize,
        _moe_intermediate_size: usize,
        _shared_expert_intermediate_size: usize,
        _hidden_size: usize,
        _group_size: u32,
        _bits: u32,
        _gate_bits: u32,
    ) -> anyhow::Result<Self> {
        anyhow::bail!("scratchy-target-spyre: SharedFusedMoELayer affine load not yet implemented")
    }
}

// ---------------------------------------------------------------------------
// GemmaRouterLayer (Gemma-4 MoE router bundle)
// ---------------------------------------------------------------------------

pub trait Gemma4RouterOps {
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        hidden_size: usize,
        group_size: u32,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
}

impl Gemma4RouterOps for GemmaRouterLayer {
    /// Macro-emitted load entry for the Gemma-4 router bundle (spyre).
    fn load(
        _gw: &mut crate::weights::GpuWeights,
        _prefix: &str,
        _num_experts: usize,
        _hidden_size: usize,
        _group_size: u32,
    ) -> anyhow::Result<Self> {
        anyhow::bail!("scratchy-target-spyre: GemmaRouterLayer load not yet implemented")
    }
}

// ---------------------------------------------------------------------------
// SwitchGluExpertsLayer (Gemma-4 fused routed experts)
// ---------------------------------------------------------------------------

pub trait SwitchGluExpertsOps {
    #[allow(clippy::too_many_arguments)]
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        hidden_size: usize,
        group_size: u32,
        bits: u32,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
}

impl SwitchGluExpertsOps for SwitchGluExpertsLayer {
    /// Macro-emitted load entry for the Gemma-4 fused routed experts (spyre).
    #[allow(clippy::too_many_arguments)]
    fn load(
        _gw: &mut crate::weights::GpuWeights,
        _prefix: &str,
        _num_experts: usize,
        _top_k: usize,
        _moe_intermediate_size: usize,
        _hidden_size: usize,
        _group_size: u32,
        _bits: u32,
    ) -> anyhow::Result<Self> {
        anyhow::bail!("scratchy-target-spyre: SwitchGluExpertsLayer load not yet implemented")
    }
}

// ---------------------------------------------------------------------------
// DeepSeekV2MoELayer (BF16 routed + shared experts)
// ---------------------------------------------------------------------------

pub trait DeepSeekV2MoEOps {
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        n_routed_experts: usize,
        n_shared_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        hidden_size: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        use_sigmoid: bool,
        n_expert_group: usize,
        topk_group: usize,
        stream: crate::LoadStream,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
}

impl DeepSeekV2MoEOps for DeepSeekV2MoELayer {
    /// Macro-emitted load entry for DeepSeek-V2/V3-family BF16 MoE (spyre).
    #[allow(clippy::too_many_arguments)]
    fn load(
        _gw: &mut crate::weights::GpuWeights,
        _prefix: &str,
        _n_routed_experts: usize,
        _n_shared_experts: usize,
        _top_k: usize,
        _moe_intermediate_size: usize,
        _hidden_size: usize,
        _norm_topk_prob: bool,
        _routed_scaling_factor: f32,
        _use_sigmoid: bool,
        _n_expert_group: usize,
        _topk_group: usize,
        _stream: crate::LoadStream,
    ) -> anyhow::Result<Self> {
        anyhow::bail!("scratchy-target-spyre: DeepSeekV2MoELayer load not yet implemented")
    }
}

// ---------------------------------------------------------------------------
// DeepSeekV2Fp8BlockMoELayer (FP8 E4M3 block-quantized routed + shared)
// ---------------------------------------------------------------------------

pub trait DeepSeekV2Fp8BlockMoEOps {
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        n_routed_experts: usize,
        n_shared_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        hidden_size: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        use_sigmoid: bool,
        n_expert_group: usize,
        topk_group: usize,
        output_dtype: crate::dtype::DType,
        stream: crate::LoadStream,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
}

impl DeepSeekV2Fp8BlockMoEOps for DeepSeekV2Fp8BlockMoELayer {
    /// Macro-emitted load entry for DeepSeek-V3/Kimi-K2 FP8-block MoE (spyre).
    #[allow(clippy::too_many_arguments)]
    fn load(
        _gw: &mut crate::weights::GpuWeights,
        _prefix: &str,
        _n_routed_experts: usize,
        _n_shared_experts: usize,
        _top_k: usize,
        _moe_intermediate_size: usize,
        _hidden_size: usize,
        _norm_topk_prob: bool,
        _routed_scaling_factor: f32,
        _use_sigmoid: bool,
        _n_expert_group: usize,
        _topk_group: usize,
        _output_dtype: crate::dtype::DType,
        _stream: crate::LoadStream,
    ) -> anyhow::Result<Self> {
        anyhow::bail!("scratchy-target-spyre: DeepSeekV2Fp8BlockMoELayer load not yet implemented")
    }
}

// ---------------------------------------------------------------------------
// DeepSeekV2GgmlMoELayer (GGUF/GGML-quantized routed + shared experts)
// ---------------------------------------------------------------------------

pub trait DeepSeekV2GgmlMoEOps {
    fn load_gguf(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        n_routed_experts: usize,
        n_shared_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        hidden_size: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        use_sigmoid: bool,
        n_expert_group: usize,
        topk_group: usize,
        stream: crate::LoadStream,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
}

impl DeepSeekV2GgmlMoEOps for DeepSeekV2GgmlMoELayer {
    /// Macro-emitted load entry for DeepSeek-family GGUF/GGML MoE (spyre).
    #[allow(clippy::too_many_arguments)]
    fn load_gguf(
        _gw: &mut crate::weights::GpuWeights,
        _prefix: &str,
        _n_routed_experts: usize,
        _n_shared_experts: usize,
        _top_k: usize,
        _moe_intermediate_size: usize,
        _hidden_size: usize,
        _norm_topk_prob: bool,
        _routed_scaling_factor: f32,
        _use_sigmoid: bool,
        _n_expert_group: usize,
        _topk_group: usize,
        _stream: crate::LoadStream,
    ) -> anyhow::Result<Self> {
        anyhow::bail!("scratchy-target-spyre: DeepSeekV2GgmlMoELayer load not yet implemented")
    }
}
