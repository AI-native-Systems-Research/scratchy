// SPDX-License-Identifier: Apache-2.0
//! Metal MoE (Mixture of Experts) runtime layers using `GpuTensor`.
//!
//! Metal serves MLX-affine int4 expert weights (Mixtral / Qwen-MoE family);
//! the routed forward pass is dispatched via the lowered dispatch tape
//! (`Instruction::MetalFusedMoe`), so this file only owns the *load* surface
//! the forward-compiler macro emits as `crate::__gpu::layers_moe::…`.
//!
//! The MoE type *definitions* (bare field bags) + the pure `MoeRouting` borrow
//! bundle + `select_moe_block_m` live in the cfg-free `scratchy-layers` crate;
//! re-exported here via the glob below so `crate::__gpu::layers_moe::<T>`
//! paths resolve. The kernel-calling `load_*` runtime methods stay in this
//! file as per-type `*Ops` extension traits.

pub use scratchy_layers::layers_moe::*;

// `take_into_metal` / `metal_allocator` / `take_affine_dequant_b4` are on the
// metal weight-load extension trait; `take` / `take_keep_dtype` / `contains`
// / `tensor_info` / `target_dtype` are inherent on the neutral `GpuWeights`.
use crate::weights_metal::MetalWeightsExt;

// ---------------------------------------------------------------------------
// AffineFusedMoELayer (metal: MLX-affine experts + dense BF16 router)
// ---------------------------------------------------------------------------

pub trait AffineFusedMoEOps {
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        intermediate_size: usize,
        hidden_size: usize,
        group_size: u32,
        bits: ExpertBits,
        gate_bits: u32,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn load_with_naming(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        intermediate_size: usize,
        hidden_size: usize,
        group_size: u32,
        bits: ExpertBits,
        gate_bits: u32,
        naming: MoeExpertNaming,
        renormalize: bool,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
}

impl AffineFusedMoEOps for AffineFusedMoELayer {
    /// Load a Mixtral-style MLX-affine MoE checkpoint. Expert
    /// weights follow HF's `experts.{e}.{w1,w2,w3}.{weight,scales,biases}`
    /// naming (w1 = gate_proj, w3 = up_proj, w2 = down_proj). Router
    /// lives at `{prefix}.gate.weight` as a dense `[E, hidden]` tensor.
    ///
    /// Per-expert tensors are stacked into one MTLBuffer per
    /// `(weight_kind, tensor_role)` pair — packed at expert-major
    /// stride so the `affine_gather_qmv` kernel can resolve
    /// `expert_offset = e * out * in/8` (for U32 weights) or
    /// `e * out * in/gs` (for F16 scales/biases) with a single load.
    #[allow(clippy::too_many_arguments)]
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        intermediate_size: usize,
        hidden_size: usize,
        group_size: u32,
        bits: ExpertBits,
        gate_bits: u32,
    ) -> anyhow::Result<Self> {
        Self::load_with_naming(
            gw,
            prefix,
            num_experts,
            top_k,
            intermediate_size,
            hidden_size,
            group_size,
            bits,
            gate_bits,
            MoeExpertNaming::Mixtral,
            true,
        )
    }

    /// Load with explicit per-expert naming convention.
    ///
    /// Each projection is read through [`load_expert_projection`]: its
    /// pre-stacked `{prefix}.switch_mlp.{proj}.*` tensors (mlx-community
    /// Qwen3-MoE-4bit and newer, Nemotron-H), else the per-expert
    /// `{prefix}.experts.{e}.{proj}.*` files stacked at load (Mixtral-4bit,
    /// older Qwen-MoE 4bit). A gated bank's projections are named by
    /// `naming`; an ungated one's are mlx `SwitchMLP`'s own `fc1` / `fc2`.
    ///
    /// Router gate (`{prefix}.gate`) is always quantized in
    /// mlx-community 4bit MoE checkpoints; we dequantize it at load
    /// time via `take_affine_dequant_b4` so the routing Gemm step
    /// stays on the dense BF16/F16 path (no quantized GEMM kernel
    /// for `[E, hidden]` is needed).
    #[allow(clippy::too_many_arguments)]
    fn load_with_naming(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        intermediate_size: usize,
        hidden_size: usize,
        group_size: u32,
        bits: ExpertBits,
        gate_bits: u32,
        naming: MoeExpertNaming,
        renormalize: bool,
    ) -> anyhow::Result<Self> {
        use crate::dtype::DType;

        // Router gate: quantized in mlx-community 4bit MoE
        // checkpoints (verified on Qwen3-30B-A3B-4bit and
        // Mixtral-8x7B-Instruct-v0.1-4bit). Dequant to a typed dense
        // tensor so the routing Gemm step in `lower_moe_step`
        // (Step 1) reads it as a normal `[E, hidden]` BF16/F16
        // weight — same shape contract as the cuda Dense path.
        // GLM-4.5 and Nemotron-H keep their router DENSE (`{prefix}.gate.weight`
        // bf16, no scales/biases) — taken as-is. The discriminator is the
        // SCALES SIBLING, not the weight key: a quantized router ALSO
        // carries `{prefix}.gate.weight` (the packed bitstream), so
        // probing the weight alone would hand the routing gemm raw
        // nibbles as logits.
        // Choose dequant dtype from the model's target_dtype (BF16
        // on every metal build path today).
        let router_dtype = gw.target_dtype().unwrap_or(DType::BF16);
        anyhow::ensure!(
            matches!(router_dtype, DType::BF16 | DType::F16),
            "AffineFusedMoELayer: router dequant target must be BF16 or F16, got {router_dtype}"
        );
        let router_is_dense = !gw.contains(&format!("{prefix}.gate.scales"));
        let router_gate = if router_is_dense {
            gw.take(&format!("{prefix}.gate.weight"))?
        } else {
            gw.take_affine_dequant_b4(
                &format!("{prefix}.gate"),
                group_size,
                gate_bits,
                router_dtype,
            )?
        };

        let (gate, up, down) = match bits {
            ExpertBits::Gated { .. } => naming.proj_names(),
            ExpertBits::Ungated { .. } => ("", "fc1", "fc2"),
        };
        let (infix, n) = (Some("switch_mlp"), num_experts);
        let (inter, hidden) = (intermediate_size, hidden_size);
        let in_proj = |gw: &mut crate::weights::GpuWeights, proj, bits| {
            load_expert_projection(gw, prefix, infix, proj, n, inter, hidden, group_size, bits)
        };
        let gate = bits.gate().map(|b| in_proj(gw, gate, b)).transpose()?;
        let [up_w, up_s, up_b] = in_proj(gw, up, bits.up())?;
        let [down_w, down_s, down_b] = load_expert_projection(
            gw,
            prefix,
            infix,
            down,
            n,
            hidden,
            inter,
            group_size,
            bits.down(),
        )?;
        let [gate_w, gate_s, gate_b] = gate.map_or([None; 3], |t| t.map(Some));

        Ok(AffineFusedMoELayer {
            router_gate,
            expert_gate_w: gate_w,
            expert_gate_scales: gate_s,
            expert_gate_biases: gate_b,
            expert_up_w: up_w,
            expert_up_scales: up_s,
            expert_up_biases: up_b,
            expert_down_w: down_w,
            expert_down_scales: down_s,
            expert_down_biases: down_b,
            num_experts,
            top_k,
            intermediate_size,
            hidden_size,
            renormalize,
            group_size,
            bits,
        })
    }
}

/// Per-expert tensor naming convention of a GATED bank. Mixtral safetensors keep the
/// pre-Llama-2 `w1/w2/w3` names while every other HF MoE checkpoint
/// (Qwen2-MoE, Qwen3-MoE, DeepSeek-MoE) uses `gate_proj/up_proj/down_proj`.
#[derive(Clone, Copy, Debug)]
pub enum MoeExpertNaming {
    /// `w1` = gate, `w3` = up, `w2` = down.
    Mixtral,
    /// `gate_proj`, `up_proj`, `down_proj`.
    Qwen,
    /// Gemma-4 SwitchGLU experts: pre-stacked `[E, out, in]` tensors live
    /// directly under `{prefix}.gate_proj/.up_proj/.down_proj` (no
    /// `switch_mlp` / `switch_glu` infix in the leaf name — the
    /// `experts.switch_glu` infix is already part of the bundle prefix).
    SwitchGlu,
}

impl MoeExpertNaming {
    fn proj_names(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Self::Mixtral => ("w1", "w3", "w2"),
            Self::Qwen | Self::SwitchGlu => ("gate_proj", "up_proj", "down_proj"),
        }
    }
}

/// One routed-expert projection's MLX-affine `[weight, scales, biases]`, expert-major
/// `[E, n_out, k_in / div]` with the weight packed `bits` wide (the ceil(k·bits/32) law, exact:
/// `k_in` is a multiple of the group size).
///
/// The pre-stacked tensors `{prefix}[.{infix}].{proj}.*` are taken zero-copy, their shapes
/// verified by [`restack_expert_tensor`] (which also views gpt-oss's 2-D `[(E·N), K/div]` export
/// as the 3-D stack); without them, each expert's own `{prefix}.experts.{e}.{proj}.*` is stacked
/// into one buffer per tensor. Experts are WEIGHTS: they go to the weight pool, not the wired
/// activation arena (`alloc_uninit` once pinned mixtral's 23.62 GiB of experts as "scratch").
#[allow(clippy::too_many_arguments)]
fn load_expert_projection(
    gw: &mut crate::weights::GpuWeights,
    prefix: &str,
    stacked_infix: Option<&str>,
    proj: &str,
    num_experts: usize,
    n_out: usize,
    k_in: usize,
    group_size: u32,
    bits: u32,
) -> anyhow::Result<[crate::tensor::GpuTensor; 3]> {
    use crate::dtype::DType;
    anyhow::ensure!(
        matches!(bits, 2 | 3 | 4 | 8),
        "affine MoE `{prefix}` {proj}: only bits ∈ {{2, 3, 4, 8}} supported (got {bits})"
    );
    anyhow::ensure!(
        k_in.is_multiple_of(group_size as usize),
        "affine MoE `{prefix}` {proj}: in-width {k_in} not divisible by group_size={group_size}"
    );
    let packed = crate::layers_quant::affine_packed_cols(k_in, bits);
    let groups = k_in / group_size as usize;
    let stacked = |kind: &str| match stacked_infix {
        Some(infix) => format!("{prefix}.{infix}.{proj}.{kind}"),
        None => format!("{prefix}.{proj}.{kind}"),
    };
    if gw.contains(&stacked("weight")) {
        let at = |t, kind: &str, cols| {
            restack_expert_tensor(t, &stacked(kind), num_experts, n_out, cols)
        };
        let w = at(gw.take(&stacked("weight"))?, "weight", packed)?;
        let s = at(gw.take_keep_dtype(&stacked("scales"))?, "scales", groups)?;
        let b = at(gw.take_keep_dtype(&stacked("biases"))?, "biases", groups)?;
        return Ok([w, s, b]);
    }
    // Per-expert naming: mlx `SwitchMLP`'s own `fc1`/`fc2`, but a conversion that keeps nvidia's
    // per-expert names spells them `up_proj`/`down_proj` (Nemotron-3 Super's OptiQ MTP sidecar —
    // its main shards' pre-stacked `switch_mlp.*` tensors use fc1/fc2, and the two layouts never
    // mix in one file). Probe expert 0 for whichever this store holds.
    let proj = if gw.contains(&format!("{prefix}.experts.0.{proj}.scales")) {
        proj
    } else {
        match proj {
            "fc1" => "up_proj",
            "fc2" => "down_proj",
            _ => proj,
        }
    };
    let name = |e: usize, kind: &str| format!("{prefix}.experts.{e}.{proj}.{kind}");
    let (_, scales_dtype) = gw
        .tensor_info(&name(0, "scales"))
        .ok_or_else(|| anyhow::anyhow!("affine MoE: weight not found: {}", name(0, "scales")))?;
    let mut stack = |kind: &str, cols: usize, dtype: DType| -> anyhow::Result<_> {
        let per_expert = n_out * cols * dtype.size_bytes();
        let bytes = num_experts * per_expert;
        let ptr = gw
            .metal_allocator()
            .alloc_uninit_weights(bytes)
            .map_err(|e| anyhow::anyhow!("alloc_uninit_weights({bytes}) failed: {e}"))?;
        for e in 0..num_experts {
            // SAFETY: `ptr` holds `num_experts` slabs of `per_expert` bytes, expert `e`'s
            // tensor exactly one of them.
            unsafe { gw.take_into_metal(&name(e, kind), ptr.add(e * per_expert))? };
        }
        // SAFETY: the buffer was sized for exactly this shape and filled above.
        Ok(unsafe { crate::tensor::GpuTensor::new(ptr, &[num_experts, n_out, cols], dtype) })
    };
    Ok([
        stack("weight", packed, DType::U32)?,
        stack("scales", groups, scales_dtype)?,
        stack("biases", groups, scales_dtype)?,
    ])
}

// ---------------------------------------------------------------------------
// Gemma-4 MoE loaders: GemmaRouterLayer + SwitchGluExpertsLayer (Metal-only)
// ---------------------------------------------------------------------------

/// Metal load extension trait on the neutral [`GemmaRouterLayer`].
pub trait Gemma4RouterOps {
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        hidden_size: usize,
        group_size: u32,
        bits: u32,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
}

impl Gemma4RouterOps for GemmaRouterLayer {
    /// Load the Gemma-4 router bundle from `{prefix}` (= `...layers.N.router`):
    /// * `{prefix}.proj` → dequantized affine `[E, hidden]` dense gate, at the
    ///   `bits` the caller resolved (8 for every checkpoint shipped so far).
    /// * `{prefix}.per_expert_scale` → `[E]` bf16, kept verbatim.
    /// * `{prefix}.scale` → `[hidden]` bf16 RMSNorm gain, kept verbatim.
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        hidden_size: usize,
        group_size: u32,
        bits: u32,
    ) -> anyhow::Result<Self> {
        use crate::dtype::DType;

        let router_dtype = gw.target_dtype().unwrap_or(DType::BF16);
        anyhow::ensure!(
            matches!(router_dtype, DType::BF16 | DType::F16),
            "GemmaRouterLayer: router dequant target must be BF16 or F16, got {router_dtype}"
        );
        // router.proj is MLX-affine on disk at its OWN width — 8-bit on every
        // Gemma-4-MoE checkpoint so far, while the rest of the model is 4-bit.
        // The width arrives from the macro (resolved through `affine_role_bits`
        // from the arch's declared `bits_overrides`) rather than being baked
        // here, so the fingerprint's bit map and this call cannot disagree (#202).
        // Dequant at load so the routing GEMM stays on the dense path.
        let gate =
            gw.take_affine_dequant_b4(&format!("{prefix}.proj"), group_size, bits, router_dtype)?;
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

/// Metal load extension trait on the neutral [`SwitchGluExpertsLayer`].
pub trait SwitchGluExpertsOps {
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
    /// Load the router-less SwitchGLU expert stack from `{prefix}`
    /// (= `...layers.N.experts.switch_glu`) through [`load_expert_projection`]
    /// with no stacked infix (the leaves are `{prefix}.{gate,up,down}_proj.*`).
    /// NO router gate is loaded.
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
        anyhow::ensure!(
            bits == 4 || bits == 8,
            "SwitchGluExpertsLayer: only bits ∈ {{4, 8}} supported (got {bits})"
        );
        let (n, inter, hidden) = (num_experts, moe_intermediate_size, hidden_size);
        let proj = |gw: &mut crate::weights::GpuWeights, name, n_out, k_in| {
            load_expert_projection(gw, prefix, None, name, n, n_out, k_in, group_size, bits)
        };
        let [expert_gate_w, expert_gate_scales, expert_gate_biases] =
            proj(gw, "gate_proj", inter, hidden)?;
        let [expert_up_w, expert_up_scales, expert_up_biases] = proj(gw, "up_proj", inter, hidden)?;
        let [expert_down_w, expert_down_scales, expert_down_biases] =
            proj(gw, "down_proj", hidden, inter)?;
        Ok(SwitchGluExpertsLayer {
            expert_gate_w,
            expert_gate_scales,
            expert_gate_biases,
            expert_up_w,
            expert_up_scales,
            expert_up_biases,
            expert_down_w,
            expert_down_scales,
            expert_down_biases,
            num_experts,
            top_k,
            moe_intermediate_size,
            hidden_size,
            group_size,
            bits,
            // Native INT4 (Marlin) expert path is cuda-only; metal dequants.
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
// FusedMoELayer (enum: Affine for metal)
// ---------------------------------------------------------------------------

pub trait FusedMoEOps {
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        intermediate_size: usize,
        hidden_size: usize,
        stream: crate::CUstream,
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
        bits: ExpertBits,
        gate_bits: u32,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
}

impl FusedMoEOps for FusedMoELayer {
    /// Metal load stub for Dense MoE checkpoints. Bails at load time
    /// — Dense BF16 MoE on Metal would need a per-expert dense-matmul
    /// path we haven't ported. Affine int4 checkpoints go through
    /// [`Self::load_affine`] instead. The stub exists so model crates
    /// whose `model.bounds` advertise Dense MoE storage (e.g. the
    /// 1-layer Mixtral test config) still satisfy macro expansion
    /// under `--features metal` — same surface the pre-§2 `bail!`
    /// stub provided.
    #[allow(clippy::too_many_arguments)]
    fn load(
        _gw: &mut crate::weights::GpuWeights,
        _prefix: &str,
        _num_experts: usize,
        _top_k: usize,
        _intermediate_size: usize,
        _hidden_size: usize,
        _stream: crate::CUstream,
    ) -> anyhow::Result<Self> {
        anyhow::bail!(
            "FusedMoELayer: Dense BF16 MoE not supported on metal — use a Mixtral / \
             Qwen-MoE / DeepSeek-MoE checkpoint with MLX-affine int4 expert weights instead"
        )
    }

    /// Macro-emitted load entry for MLX-affine int4 MoE checkpoints
    /// (metal). The macro detects `StorageFormat::Affine` on the MoE
    /// source weights and emits this call site with `group_size` and
    /// `bits` baked in from the model's `quantization_config`.
    #[allow(clippy::too_many_arguments)]
    fn load_affine(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        intermediate_size: usize,
        hidden_size: usize,
        group_size: u32,
        bits: ExpertBits,
        gate_bits: u32,
    ) -> anyhow::Result<Self> {
        Ok(Self::Affine(Box::new(AffineFusedMoELayer::load(
            gw,
            prefix,
            num_experts,
            top_k,
            intermediate_size,
            hidden_size,
            group_size,
            bits,
            gate_bits,
        )?)))
    }
}

// ---------------------------------------------------------------------------
// AffineSharedFusedMoELayer (metal: MLX-affine int4 Qwen-MoE family)
// ---------------------------------------------------------------------------

pub trait AffineSharedFusedMoEOps {
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        shared_expert_intermediate_size: usize,
        hidden_size: usize,
        group_size: u32,
        bits: ExpertBits,
        gate_bits: u32,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
}

impl AffineSharedFusedMoEOps for AffineSharedFusedMoELayer {
    /// Load a Qwen-MoE-style MLX-affine int4 checkpoint. Expert
    /// weights use HF's `gate_proj/up_proj/down_proj` naming; shared
    /// expert lives at `{prefix}.shared_expert.{gate,up,down}_proj`
    /// with a sigmoid gate at `{prefix}.shared_expert_gate.weight`.
    #[allow(clippy::too_many_arguments)]
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        shared_expert_intermediate_size: usize,
        hidden_size: usize,
        group_size: u32,
        bits: ExpertBits,
        gate_bits: u32,
    ) -> anyhow::Result<Self> {
        use crate::dtype::DType;
        use crate::tensor::GpuTensor;

        // Qwen-MoE family always renormalizes (`norm_topk_prob = true`
        // is the family default; Qwen3-MoE configs make it explicit).
        let routed = AffineFusedMoELayer::load_with_naming(
            gw,
            prefix,
            num_experts,
            top_k,
            moe_intermediate_size,
            hidden_size,
            group_size,
            bits,
            gate_bits,
            MoeExpertNaming::Qwen,
            true,
        )?;

        // Shared expert tail — only present when shared_inter > 0.
        let (
            shared_gate_up_w,
            shared_gate_up_scales,
            shared_gate_up_biases,
            shared_down_w,
            shared_down_scales,
            shared_down_biases,
            shared_expert_gate,
        ) = if shared_expert_intermediate_size > 0 {
            // The internal shared tail is quantized as the routed bank (Qwen's uniform
            // checkpoints); a DSL-declared shared expert loads through its own accessors.
            let bits = bits.up();
            anyhow::ensure!(
                matches!(bits, 2 | 3 | 4 | 8),
                "AffineSharedFusedMoELayer: only bits ∈ {{2, 3, 4, 8}} supported (got {bits})"
            );
            let shared_inter = shared_expert_intermediate_size;
            // Two cases for shared gate+up: either fused on-disk
            // (`shared_expert.gate_up_proj`) or split (the more common
            // Qwen-MoE shape with separate gate_proj/up_proj). The
            // affine_gather_qmv lowering wants a single packed
            // `[2*shared_inter, ceil(hidden*bits/32)]` triple, so we
            // stack split-on-disk variants by allocating and copying
            // gate then up into adjacent halves.
            let has_fused = gw.contains(&format!("{prefix}.shared_expert.gate_up_proj.weight"));

            // Sample scales dtype the same way as the routed path.
            let probe_name = if has_fused {
                format!("{prefix}.shared_expert.gate_up_proj.scales")
            } else {
                format!("{prefix}.shared_expert.gate_proj.scales")
            };
            let (_, scales_dtype) = gw.tensor_info(&probe_name).ok_or_else(|| {
                anyhow::anyhow!("affine MoE shared: weight not found: {probe_name}")
            })?;
            let elem_w = DType::U32.size_bytes();
            let elem_sb = scales_dtype.size_bytes();
            let gs = group_size as usize;

            // Allocate destination buffers. Packed cols follow the
            // ceil(in*bits/32) law (exact: in is a multiple of group_size).
            let packed_hidden = crate::layers_quant::affine_packed_cols(hidden_size, bits);
            let packed_shared_inter = crate::layers_quant::affine_packed_cols(shared_inter, bits);
            let two_si = 2 * shared_inter;
            let gu_w_bytes = two_si * packed_hidden * elem_w;
            let gu_sb_bytes = two_si * (hidden_size / gs) * elem_sb;
            let down_w_bytes = hidden_size * packed_shared_inter * elem_w;
            let down_sb_bytes = hidden_size * (shared_inter / gs) * elem_sb;

            // Shared-expert weights: same pool rule as the routed
            // experts above.
            let alloc = |bytes: usize| -> anyhow::Result<*mut u8> {
                gw.metal_allocator()
                    .alloc_uninit_weights(bytes)
                    .map_err(|e| anyhow::anyhow!("alloc_uninit_weights({bytes}) failed: {e}"))
            };

            let gu_w_ptr = alloc(gu_w_bytes)?;
            let gu_s_ptr = alloc(gu_sb_bytes)?;
            let gu_b_ptr = alloc(gu_sb_bytes)?;
            let down_w_ptr = alloc(down_w_bytes)?;
            let down_s_ptr = alloc(down_sb_bytes)?;
            let down_b_ptr = alloc(down_sb_bytes)?;

            if has_fused {
                unsafe {
                    let _ = gw.take_into_metal(
                        &format!("{prefix}.shared_expert.gate_up_proj.weight"),
                        gu_w_ptr,
                    )?;
                    let _ = gw.take_into_metal(
                        &format!("{prefix}.shared_expert.gate_up_proj.scales"),
                        gu_s_ptr,
                    )?;
                    let _ = gw.take_into_metal(
                        &format!("{prefix}.shared_expert.gate_up_proj.biases"),
                        gu_b_ptr,
                    )?;
                }
            } else {
                // Split → stack into [gate_half | up_half].
                let half_w_bytes = shared_inter * packed_hidden * elem_w;
                let half_sb_bytes = shared_inter * (hidden_size / gs) * elem_sb;
                unsafe {
                    let _ = gw.take_into_metal(
                        &format!("{prefix}.shared_expert.gate_proj.weight"),
                        gu_w_ptr,
                    )?;
                    let _ = gw.take_into_metal(
                        &format!("{prefix}.shared_expert.up_proj.weight"),
                        gu_w_ptr.add(half_w_bytes),
                    )?;
                    let _ = gw.take_into_metal(
                        &format!("{prefix}.shared_expert.gate_proj.scales"),
                        gu_s_ptr,
                    )?;
                    let _ = gw.take_into_metal(
                        &format!("{prefix}.shared_expert.up_proj.scales"),
                        gu_s_ptr.add(half_sb_bytes),
                    )?;
                    let _ = gw.take_into_metal(
                        &format!("{prefix}.shared_expert.gate_proj.biases"),
                        gu_b_ptr,
                    )?;
                    let _ = gw.take_into_metal(
                        &format!("{prefix}.shared_expert.up_proj.biases"),
                        gu_b_ptr.add(half_sb_bytes),
                    )?;
                }
            }

            unsafe {
                let _ = gw.take_into_metal(
                    &format!("{prefix}.shared_expert.down_proj.weight"),
                    down_w_ptr,
                )?;
                let _ = gw.take_into_metal(
                    &format!("{prefix}.shared_expert.down_proj.scales"),
                    down_s_ptr,
                )?;
                let _ = gw.take_into_metal(
                    &format!("{prefix}.shared_expert.down_proj.biases"),
                    down_b_ptr,
                )?;
            }

            let gu_w = unsafe { GpuTensor::new(gu_w_ptr, &[two_si, packed_hidden], DType::U32) };
            let gu_s =
                unsafe { GpuTensor::new(gu_s_ptr, &[two_si, hidden_size / gs], scales_dtype) };
            let gu_b =
                unsafe { GpuTensor::new(gu_b_ptr, &[two_si, hidden_size / gs], scales_dtype) };
            let d_w = unsafe {
                GpuTensor::new(down_w_ptr, &[hidden_size, packed_shared_inter], DType::U32)
            };
            let d_s = unsafe {
                GpuTensor::new(down_s_ptr, &[hidden_size, shared_inter / gs], scales_dtype)
            };
            let d_b = unsafe {
                GpuTensor::new(down_b_ptr, &[hidden_size, shared_inter / gs], scales_dtype)
            };
            let sgate = gw.take(&format!("{prefix}.shared_expert_gate.weight"))?;

            (
                Some(gu_w),
                Some(gu_s),
                Some(gu_b),
                Some(d_w),
                Some(d_s),
                Some(d_b),
                Some(sgate),
            )
        } else {
            (None, None, None, None, None, None, None)
        };

        // GLM-4.5's F32 `e_score_correction_bias` (`{prefix}.gate.e_score_correction_bias`) —
        // present on every sigmoid-routed MoE (noaux_tc), absent on every softmax-routed one
        // (Qwen, Mixtral). Probed, never assumed. `take_keep_dtype`, NOT `take`: the routing
        // kernels read it as `const device float*` (`moe_route.metal` buffer 4,
        // `quantized_qmv.metal` buffer 13), so the F32 bytes must reach the device un-cast —
        // `take` would truncate them to the model's target dtype (BF16) and the kernels would
        // reassemble half-pairs as F32 garbage. The cuda loader casts this tensor to F32 for
        // the same reason (`weights_quant.rs`: "vLLM always casts this to F32 before the
        // routing kernel; we match"); every noaux_tc checkpoint ships it F32 on disk.
        let e_score_correction_bias = match gw
            .contains(&format!("{prefix}.gate.e_score_correction_bias"))
        {
            true => Some(gw.take_keep_dtype(&format!("{prefix}.gate.e_score_correction_bias"))?),
            false => None,
        };

        Ok(AffineSharedFusedMoELayer {
            routed,
            shared_gate_up_w,
            shared_gate_up_scales,
            shared_gate_up_biases,
            shared_down_w,
            shared_down_scales,
            shared_down_biases,
            shared_expert_gate,
            e_score_correction_bias,
            shared_intermediate_size: shared_expert_intermediate_size,
        })
    }
}

// ---------------------------------------------------------------------------
// GptOssMoELayer (gpt-oss 20b/120b: biased router + SwiGLU-OAI experts)
// ---------------------------------------------------------------------------

/// Metal load extension trait on the neutral [`GptOssMoELayer`].
pub trait GptOssMoEOps {
    #[allow(clippy::too_many_arguments)]
    fn load(
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

/// A pre-stacked expert tensor, verified `[e, out, packed]`. The gpt-oss
/// checkpoint ships each one 2-D `[(E·N), K/div]` (mlx-lm's converter
/// flattens the expert axis); the
/// shaders walk `[E, N, K/div]`. Row-major, those are the same bytes — so
/// the restack is a metadata-only reshape, guarded by a REAL shape assert
/// (a silent mismatch here is fluent garbage, per the `AffineInt4::mint`
/// law). A genuinely 3-D checkpoint (the Qwen3-MoE convention) passes
/// through with its dims verified.
fn restack_expert_tensor(
    t: crate::tensor::GpuTensor,
    name: &str,
    e: usize,
    out: usize,
    packed: usize,
) -> anyhow::Result<crate::tensor::GpuTensor> {
    let expected = e * out * packed;
    anyhow::ensure!(
        t.numel() == expected,
        "affine MoE: `{name}` holds {} elements (expected {e}·{out}·{packed} = {expected})",
        t.numel(),
    );
    if t.ndim() == 3 {
        anyhow::ensure!(
            t.dim(0) == e && t.dim(1) == out && t.dim(2) == packed,
            "affine MoE: `{name}` shape {:?} (expected [{e}, {out}, {packed}])",
            t.shape(),
        );
        Ok(t)
    } else {
        anyhow::ensure!(
            t.ndim() == 2 && t.dim(0) == e * out && t.dim(1) == packed,
            "affine MoE: `{name}` shape {:?} (expected 2-D [{}, {packed}] — the \
             pre-stacked gpt-oss layout — or 3-D [{e}, {out}, {packed}])",
            t.shape(),
            e * out,
        );
        Ok(t.reshape(&[e, out, packed]))
    }
}

impl GptOssMoEOps for GptOssMoELayer {
    /// Load the gpt-oss MoE bundle from `{prefix}` (= `...layers.N.mlp`):
    /// * `{prefix}.router` → dequantized affine `[E, hidden]` dense gate at
    ///   `gate_bits` — the jesusoctavioas 2-bit checkpoints quantize the
    ///   router at the preset's own width, so `gate_bits` is the live path.
    ///   A dense `{prefix}.router.weight` (no scales sibling) is taken as-is.
    /// * `{prefix}.router.bias` → `[E]`, converted to F32: the routing
    ///   kernels read `const device float*` (buffer 4 of the route command,
    ///   buffer 13 of the gathered expert kernels) — the same ABI as GLM's
    ///   F32 `e_score_correction_bias`, but gpt-oss ships BF16 so the
    ///   conversion happens here rather than on disk.
    /// * `{prefix}.experts.{gate,up,down}_proj.*` → the pre-stacked affine
    ///   experts (Qwen proj names + the `experts` infix), each restacked
    ///   2-D→3-D by [`restack_expert_tensor`].
    /// * `{prefix}.experts.{gate,up}_proj.bias` → the per-expert LINEAR
    ///   biases, interleaved per expert (expert e's `inter` gate rows then
    ///   its `inter` up rows) into one `[E, 2·inter]` buffer;
    ///   `{prefix}.experts.down_proj.bias` → `[E, hidden]`.
    #[allow(clippy::too_many_arguments)]
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        intermediate_size: usize,
        hidden_size: usize,
        group_size: u32,
        bits: u32,
        gate_bits: u32,
    ) -> anyhow::Result<Self> {
        use crate::dtype::DType;

        anyhow::ensure!(
            bits == 2 || bits == 3 || bits == 4 || bits == 8,
            "GptOssMoELayer: only bits ∈ {{2, 3, 4, 8}} supported (got {bits})"
        );
        anyhow::ensure!(
            hidden_size.is_multiple_of(group_size as usize),
            "GptOssMoELayer: hidden_size={hidden_size} not divisible by group_size={group_size}"
        );
        anyhow::ensure!(
            intermediate_size.is_multiple_of(group_size as usize),
            "GptOssMoELayer: intermediate_size={intermediate_size} not divisible by \
             group_size={group_size}"
        );

        // Router gate: quantized on the 2-bit checkpoints (uniform affine
        // incl. the router); a dense export (no scales sibling) is taken
        // as-is. Same scales-sibling discriminator as `AffineFusedMoELayer`
        // — probing `.weight` alone would hand the routing gemm raw nibbles.
        let router_dtype = gw.target_dtype().unwrap_or(DType::BF16);
        anyhow::ensure!(
            matches!(router_dtype, DType::BF16 | DType::F16),
            "GptOssMoELayer: router dequant target must be BF16 or F16, got {router_dtype}"
        );
        let router_is_dense = !gw.contains(&format!("{prefix}.router.scales"));
        let router_gate = if router_is_dense {
            gw.take(&format!("{prefix}.router.weight"))?
        } else {
            gw.take_affine_dequant_b4(
                &format!("{prefix}.router"),
                group_size,
                gate_bits,
                router_dtype,
            )?
        };

        // Router bias: F32 `[E]` on the device, converted from the
        // checkpoint's BF16/F16/F32 — `take_to_cpu_f32` handles every
        // on-disk float width, and `alloc_packed_from_host` re-uploads as
        // F32. The kernels' `const device float*` ABI is unchanged from
        // GLM's (see the e_score_correction_bias note above).
        let router_bias_f32 = gw.take_to_cpu_f32(&format!("{prefix}.router.bias"))?;
        anyhow::ensure!(
            router_bias_f32.len() == num_experts,
            "GptOssMoELayer: `{prefix}.router.bias` holds {} elements (expected {num_experts})",
            router_bias_f32.len(),
        );
        let mut router_bias_bytes = Vec::with_capacity(num_experts * 4);
        for v in &router_bias_f32 {
            router_bias_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let router_bias =
            gw.alloc_packed_from_host(&router_bias_bytes, &[num_experts], DType::F32)?;

        // Experts: the pre-stacked layout under the `experts` infix with
        // Qwen proj names — exactly `mlp.experts.{gate,up,down}_proj.*`.
        let (n, inter, hidden) = (num_experts, intermediate_size, hidden_size);
        let proj = |gw: &mut crate::weights::GpuWeights, name, n_out, k_in| {
            load_expert_projection(
                gw,
                prefix,
                Some("experts"),
                name,
                n,
                n_out,
                k_in,
                group_size,
                bits,
            )
        };
        let [expert_gate_w, expert_gate_scales, expert_gate_biases] =
            proj(gw, "gate_proj", inter, hidden)?;
        let [expert_up_w, expert_up_scales, expert_up_biases] = proj(gw, "up_proj", inter, hidden)?;
        let [expert_down_w, expert_down_scales, expert_down_biases] =
            proj(gw, "down_proj", hidden, inter)?;

        // Per-expert gate/up LINEAR biases → one `[E, 2·inter]` buffer,
        // expert e's gate rows then its up rows. `take_cpu` hands back
        // target-dtype bytes (floats only are cast), so the 2-byte element
        // interleave is dtype-agnostic between F16 and BF16.
        let (g_bytes, g_shape, g_dt) = gw.take_cpu(&format!("{prefix}.experts.gate_proj.bias"))?;
        let (u_bytes, u_shape, u_dt) = gw.take_cpu(&format!("{prefix}.experts.up_proj.bias"))?;
        anyhow::ensure!(
            g_dt == u_dt && matches!(g_dt, DType::F16 | DType::BF16),
            "GptOssMoELayer: expert linear biases must be F16/BF16 (got gate {g_dt}, up {u_dt})"
        );
        let elems = |shape: &[usize]| shape.iter().product::<usize>();
        anyhow::ensure!(
            elems(&g_shape) == num_experts * intermediate_size,
            "GptOssMoELayer: `{prefix}.experts.gate_proj.bias` holds {} elements (expected \
             {num_experts}·{intermediate_size})",
            elems(&g_shape),
        );
        anyhow::ensure!(
            elems(&u_shape) == num_experts * intermediate_size,
            "GptOssMoELayer: `{prefix}.experts.up_proj.bias` holds {} elements (expected \
             {num_experts}·{intermediate_size})",
            elems(&u_shape),
        );
        let esz = g_dt.size_bytes();
        let per_expert = intermediate_size * esz;
        let mut gate_up_bytes = Vec::with_capacity(num_experts * 2 * intermediate_size * esz);
        for e in 0..num_experts {
            let at = e * per_expert;
            gate_up_bytes.extend_from_slice(&g_bytes[at..at + per_expert]);
            gate_up_bytes.extend_from_slice(&u_bytes[at..at + per_expert]);
        }
        let gate_up_linear_bias =
            gw.alloc_packed_from_host(&gate_up_bytes, &[num_experts, 2 * intermediate_size], g_dt)?;

        // Down-proj LINEAR bias → `[E, hidden]`, same bytes expert-major.
        let (d_bytes, d_shape, d_dt) = gw.take_cpu(&format!("{prefix}.experts.down_proj.bias"))?;
        anyhow::ensure!(
            d_dt == g_dt,
            "GptOssMoELayer: down bias dtype {d_dt} differs from gate/up {g_dt}"
        );
        anyhow::ensure!(
            elems(&d_shape) == num_experts * hidden_size,
            "GptOssMoELayer: `{prefix}.experts.down_proj.bias` holds {} elements (expected \
             {num_experts}·{hidden_size})",
            elems(&d_shape),
        );
        let down_linear_bias =
            gw.alloc_packed_from_host(&d_bytes, &[num_experts, hidden_size], d_dt)?;

        Ok(GptOssMoELayer {
            router_gate,
            router_bias,
            expert_gate_w,
            expert_gate_scales,
            expert_gate_biases,
            expert_up_w,
            expert_up_scales,
            expert_up_biases,
            expert_down_w,
            expert_down_scales,
            expert_down_biases,
            gate_up_linear_bias,
            down_linear_bias,
            num_experts,
            top_k,
            intermediate_size,
            hidden_size,
            group_size,
            bits,
        })
    }
}

// ---------------------------------------------------------------------------
// SharedFusedMoELayer (enum: Affine for metal)
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
        stream: crate::CUstream,
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
        bits: ExpertBits,
        gate_bits: u32,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
}

impl SharedFusedMoEOps for SharedFusedMoELayer {
    /// Metal load stub — same rationale as
    /// [`FusedMoEOps::load`] (the metal Dense MoE branch).
    #[allow(clippy::too_many_arguments)]
    fn load(
        _gw: &mut crate::weights::GpuWeights,
        _prefix: &str,
        _num_experts: usize,
        _top_k: usize,
        _moe_intermediate_size: usize,
        _shared_expert_intermediate_size: usize,
        _hidden_size: usize,
        _stream: crate::CUstream,
    ) -> anyhow::Result<Self> {
        anyhow::bail!(
            "SharedFusedMoELayer: Dense BF16 Qwen-MoE not supported on metal — use \
             an MLX-affine int4 checkpoint instead"
        )
    }

    /// Macro-emitted load entry for MLX-affine int4 Qwen-MoE
    /// checkpoints (metal).
    #[allow(clippy::too_many_arguments)]
    fn load_affine(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        shared_expert_intermediate_size: usize,
        hidden_size: usize,
        group_size: u32,
        bits: ExpertBits,
        gate_bits: u32,
    ) -> anyhow::Result<Self> {
        Ok(Self::Affine(Box::new(AffineSharedFusedMoELayer::load(
            gw,
            prefix,
            num_experts,
            top_k,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            hidden_size,
            group_size,
            bits,
            gate_bits,
        )?)))
    }
}

// ---------------------------------------------------------------------------
// DeepSeekV2MoELayer
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
        stream: crate::CUstream,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
}

impl DeepSeekV2MoEOps for DeepSeekV2MoELayer {
    /// Metal stub — see [`FusedMoEOps::load`].
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
        _stream: crate::CUstream,
    ) -> anyhow::Result<Self> {
        anyhow::bail!(
            "DeepSeekV2MoELayer not supported on metal: port MoE GEMM + topk kernels first"
        )
    }
}
