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
    ///
    /// Two on-disk layouts, picked by probing `{prefix}.proj.weight` — the SAME
    /// detection cuda's loader uses:
    ///
    /// * **fp8-dynamic-per-channel** (RedHatAI `gemma-4-26B-A4B-it-FP8-dynamic`):
    ///   `router.proj.weight` is a DENSE BF16 `[E, hidden]` tensor (the router is
    ///   never quantized on this checkpoint) plus `per_expert_scale` `[E]` and
    ///   `scale` (the RMSNorm gain) `[hidden]`, all BF16. Everything is taken
    ///   VERBATIM (`take_keep_dtype`) — the staging path narrows bf16→f16 and the
    ///   launch binds it like every other gemm weight.
    /// * **MLX-affine 8-bit** (mlx-community 4bit): `router.proj` is an affine
    ///   `{weight, scales, biases}` triple, dequantized to dense `[E, hidden]`
    ///   via the shared `affine_dequant_b4_to_dtype` (the same numerics metal's
    ///   `take_affine_dequant_b4` uses) and packed from host.
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        hidden_size: usize,
        group_size: u32,
    ) -> anyhow::Result<Self> {
        // fp8 checkpoint: dense bf16 gate, kept verbatim (the staging path narrows).
        if gw.contains(&format!("{prefix}.proj.weight")) {
            let gate = gw.take_keep_dtype(&format!("{prefix}.proj.weight"))?;
            gemma_router_gate_check(&gate, num_experts, hidden_size, "fp8")?;
            let per_expert_scale = gw.take_keep_dtype(&format!("{prefix}.per_expert_scale"))?;
            let scale = gw.take_keep_dtype(&format!("{prefix}.scale"))?;
            return Ok(GemmaRouterLayer {
                gate,
                per_expert_scale,
                scale,
                num_experts,
                hidden_size,
            });
        }
        // MLX-affine 8-bit: dequant {prefix}.proj to dense [E, hidden] bf16.
        let (w_bytes, w_dt, _) = gw.take_to_cpu_bytes(&format!("{prefix}.proj.weight"))?;
        let (s_bytes, s_dt, _) = gw.take_to_cpu_bytes(&format!("{prefix}.proj.scales"))?;
        let (b_bytes, _, _) = gw.take_to_cpu_bytes(&format!("{prefix}.proj.biases"))?;
        anyhow::ensure!(
            matches!(
                w_dt,
                scratchy_tensors::DType::U32 | scratchy_tensors::DType::U8
            ),
            "GemmaRouterLayer (spyre): {prefix}.proj.weight must be packed U32/U8, got {w_dt:?}"
        );
        anyhow::ensure!(
            matches!(
                s_dt,
                scratchy_tensors::DType::F16 | scratchy_tensors::DType::BF16
            ),
            "GemmaRouterLayer (spyre): {prefix}.proj.scales must be F16/BF16, got {s_dt:?}"
        );
        let dense = scratchy_quantizations::affine::affine_dequant_b4_to_dtype(
            &w_bytes,
            &s_bytes,
            &b_bytes,
            num_experts,
            hidden_size,
            group_size,
            8,
            s_dt,
            scratchy_tensors::DType::BF16,
        )
        .map_err(|e| {
            anyhow::anyhow!("GemmaRouterLayer (spyre): {prefix}.proj dequant failed: {e}")
        })?;
        let gate = gw.alloc_packed_from_host(
            &dense,
            &[num_experts, hidden_size],
            scratchy_tensors::DType::BF16,
        )?;
        let per_expert_scale = gw.take_keep_dtype(&format!("{prefix}.per_expert_scale"))?;
        let scale = gw.take_keep_dtype(&format!("{prefix}.scale"))?;
        Ok(GemmaRouterLayer {
            gate,
            per_expert_scale,
            scale,
            num_experts,
            hidden_size,
        })
    }
}

/// Shape guard for the router's dense-gate layout: `[E, hidden]` exactly, or
/// the manifest's staged extents disagree with the buffer and staging would
/// silently misindex. (The affine path derives its shape from the same
/// `(num_experts, hidden_size)` pair, so it cannot disagree.)
fn gemma_router_gate_check(
    gate: &scratchy_tensors::GpuTensor,
    num_experts: usize,
    hidden_size: usize,
    layout: &str,
) -> anyhow::Result<()> {
    let shape = gate.shape();
    anyhow::ensure!(
        shape == [num_experts as u32, hidden_size as u32],
        "GemmaRouterLayer (spyre, {layout}): router.proj is {shape:?}, expected \
         [{num_experts}, {hidden_size}] — the checkpoint and the config disagree"
    );
    Ok(())
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
    ///
    /// fp8-dynamic-per-channel checkpoint (RedHatAI): PER-EXPERT files
    /// `{experts_base}.{e}.{gate,up,down}_proj.weight` F8_E4M3 `[out, in]` +
    /// `.weight_scale` BF16 `[out, 1]`, where `experts_base` is the prefix with
    /// `.switch_glu` stripped (the fp8 checkpoint has no switch_glu level —
    /// cuda's loader strips the same suffix for the HF layout). The per-expert
    /// files are STACKED at load into the expert-major `[E*out, in]` /
    /// `[E*out, 1]` buffers the BundleTensor sources declared: expert codes stay
    /// fp8-verbatim (1 byte; `matmulfp8` reads them 1-byte — the W8A8 win), and
    /// the per-channel scales stay bf16-verbatim. `biases` are unused on this
    /// path (fp8 per-channel has none) — one shared `[1]` placeholder, the same
    /// convention cuda's fused/marlin arms use for unused fields.
    ///
    /// Any other layout (MLX-affine pre-stacked, HF fused bf16,
    /// compressed-tensors INT4) bails for now — the fp8 checkpoint is this
    /// track's target and a partial dequant path here would be untestable.
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
    ) -> anyhow::Result<Self> {
        let experts_base = prefix.strip_suffix(".switch_glu").unwrap_or(prefix);
        let fp8 = format!("{experts_base}.0.gate_proj.weight");
        if !gw.contains(&fp8) {
            anyhow::bail!(
                "SwitchGluExpertsLayer (spyre): no per-expert fp8 tensors at \
                 `{experts_base}.{{e}}.{{gate,up,down}}_proj.weight` — this spyre loader \
                 covers the fp8-dynamic-per-channel checkpoint only (got group_size={group_size} \
                 bits={bits}); the MLX-affine / HF-fused / INT4 layouts are not ported"
            );
        }
        // Per-proj extents: gate/up are [moe_inter, hidden], down is
        // [hidden, moe_inter] — the on-disk [out, in] of every expert file.
        let mut stack = |proj: &str,
                         out: usize,
                         inp: usize|
         -> anyhow::Result<(
            scratchy_tensors::GpuTensor,
            scratchy_tensors::GpuTensor,
        )> {
            let per_w = out * inp;
            let mut w = Vec::with_capacity(num_experts * per_w);
            let mut s = Vec::with_capacity(num_experts * out * 2);
            for e in 0..num_experts {
                let (wb, wdt, wsh) =
                    gw.take_to_cpu_bytes(&format!("{experts_base}.{e}.{proj}.weight"))?;
                let (sb, sdt, ssh) =
                    gw.take_to_cpu_bytes(&format!("{experts_base}.{e}.{proj}.weight_scale"))?;
                anyhow::ensure!(
                    wdt == scratchy_tensors::DType::Fp8E4m3 && wsh == [out, inp],
                    "SwitchGluExpertsLayer (spyre): {experts_base}.{e}.{proj}.weight is \
                     {wdt:?} {wsh:?}, expected Fp8E4m3 [{out}, {inp}]"
                );
                anyhow::ensure!(
                    sdt == scratchy_tensors::DType::BF16 && ssh == [out, 1],
                    "SwitchGluExpertsLayer (spyre): {experts_base}.{e}.{proj}.weight_scale is \
                     {sdt:?} {ssh:?}, expected BF16 [{out}, 1] (per-channel dynamic)"
                );
                w.extend_from_slice(&wb);
                s.extend_from_slice(&sb);
            }
            let wt = gw.alloc_packed_from_host(
                &w,
                &[num_experts * out, inp],
                scratchy_tensors::DType::Fp8E4m3,
            )?;
            let st = gw.alloc_packed_from_host(
                &s,
                &[num_experts * out, 1],
                scratchy_tensors::DType::BF16,
            )?;
            Ok((wt, st))
        };
        let (expert_gate_w, expert_gate_scales) =
            stack("gate_proj", moe_intermediate_size, hidden_size)?;
        let (expert_up_w, expert_up_scales) = stack("up_proj", moe_intermediate_size, hidden_size)?;
        let (expert_down_w, expert_down_scales) =
            stack("down_proj", hidden_size, moe_intermediate_size)?;
        // fp8 per-channel has no biases: one shared [1] placeholder per the
        // cuda fused-arm convention (never read on this path).
        let placeholder =
            gw.alloc_packed_from_host(&[0u8, 0], &[1], scratchy_tensors::DType::BF16)?;
        Ok(SwitchGluExpertsLayer {
            expert_gate_w,
            expert_gate_scales,
            expert_gate_biases: placeholder.clone(),
            expert_up_w,
            expert_up_scales,
            expert_up_biases: placeholder.clone(),
            expert_down_w,
            expert_down_scales,
            expert_down_biases: placeholder,
            num_experts,
            top_k,
            moe_intermediate_size,
            hidden_size,
            group_size,
            bits,
            marlin_w1: None,
            marlin_w2: None,
            marlin_w1_scales: None,
            marlin_w2_scales: None,
            marlin_workspace: None,
            marlin_b_type_id: 0,
        })
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
