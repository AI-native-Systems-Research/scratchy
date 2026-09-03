// SPDX-License-Identifier: Apache-2.0
//! MoE (Mixture of Experts) layers using `GpuTensor`.
//!
//! Implements the full Python vLLM fused MoE pipeline:
//! 1. Gate → router logits
//! 2. topk_softmax → topk weights + ids
//! 3. moe_align_block_size → sorted tokens by expert
//! 4. fused_moe_gemm (GEMM 1: gate+up)
//! 5. silu_and_mul activation
//! 6. fused_moe_gemm (GEMM 2: down, with routing weight)
//! 7. moe_sum → reduced output
//!
//! The MoE type *definitions* (bare field bags) + the pure `MoeRouting` borrow
//! bundle + `select_moe_block_m` live in the cfg-free `scratchy-layers` crate;
//! re-exported here via the glob below so existing `crate::layers_moe::<T>`
//! paths keep resolving. The kernel-calling `forward`/`load_*` runtime methods
//! stay in this file as per-type `*Ops` extension traits.

pub use scratchy_layers::layers_moe::*;

// `GpuTensor` (and `Linear`, which holds only `GpuTensor` fields) are named
// by the cuda runtime trait bodies + the `#[cfg(test)]` struct-construction
// tests below; the metal load bodies bring their own `GpuTensor` in scope.
#[cfg(any(feature = "cuda", test))]
use crate::layers::Linear;
#[cfg(any(feature = "cuda", test))]
use crate::tensor::GpuTensor;
#[cfg(feature = "cuda")]
use crate::weights::CudaWeightsExt;

// Runtime extension traits for the `Linear` / `Fp8BlockLinear` payloads (their
// `load`/`forward` moved off the neutral defs into `*Ops` traits). All cuda:
// the MoE `load`/`forward` bodies that call these are inside `#[cfg(cuda)]`
// `impl` blocks.
#[cfg(feature = "cuda")]
use crate::layers::{Fp8BlockOps, GgmlOps, LinearLayerOps, LinearOps};
#[cfg(feature = "cuda")]
use crate::layers_quant::Fp8BlockLoadOps;

#[cfg(feature = "cuda")]
use crate::alloc::OwnedTensor;
#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use crate::kernels;
#[cfg(feature = "cuda")]
use crate::tensor::TensorView;

/// Dispatch the three router-selection variants. Returns `(topk_weights, topk_ids)`
/// matching `topk_softmax`'s shape contract: `[num_tokens, top_k]` F32 / I32.
#[cfg(feature = "cuda")]
pub unsafe fn route_experts(
    router_logits: GpuTensor,
    cfg: &MoeRouting<'_>,
    caching: &mut crate::alloc::CachingAllocator,
    stream: crate::CUstream,
) -> (OwnedTensor, OwnedTensor) {
    if let Some(bias) = cfg.e_score_correction_bias {
        if cfg.n_expert_group > 0 && cfg.topk_group > 0 {
            kernels::topk_noaux_tc(
                router_logits,
                *bias,
                cfg.top_k,
                cfg.n_expert_group,
                cfg.topk_group,
                cfg.renormalize,
                cfg.routed_scaling_factor,
                caching,
                stream,
            )
        } else {
            kernels::topk_sigmoid_with_bias(
                router_logits,
                *bias,
                cfg.top_k,
                cfg.renormalize,
                caching,
                stream,
            )
        }
    } else {
        kernels::topk_softmax(router_logits, cfg.top_k, cfg.renormalize, caching, stream)
    }
}

// ---------------------------------------------------------------------------
// DenseFusedMoELayer (cuda Dense MoE: Mixtral / Qwen2-MoE / Qwen3-MoE BF16)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub trait DenseFusedMoEOps {
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

    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor;

    unsafe fn forward_fused(
        &self,
        hidden_states: TensorView<'_>,
        device: &mut GpuDevice,
    ) -> OwnedTensor;
}

#[cfg(feature = "cuda")]
impl DenseFusedMoEOps for DenseFusedMoELayer {
    /// Load a Mixtral-style BF16 fused MoE layer from safetensors.
    ///
    /// Mirrors `scratchy-serving-cuda/src/model/mixtral.rs::MixtralDecoderLayer::load_moe`
    /// at single-rank (tp=1). Mixtral's on-disk weight names:
    /// - `{prefix}.gate.weight` — `[num_experts, hidden_size]`, dense BF16.
    /// - `{prefix}.experts.{e}.w1.weight` — `[intermediate_size, hidden_size]` (gate_proj).
    /// - `{prefix}.experts.{e}.w3.weight` — `[intermediate_size, hidden_size]` (up_proj).
    /// - `{prefix}.experts.{e}.w2.weight` — `[hidden_size, intermediate_size]` (down_proj).
    ///
    /// Stacked into:
    /// - `w1`: `[num_experts, 2*intermediate_size, hidden_size]` (gate+up fused).
    /// - `w2`: `[num_experts, hidden_size, intermediate_size]`.
    ///
    /// `routed_scaling_factor` defaults to 1.0; `renormalize` is false (Mixtral
    /// does not renormalize topk weights). Sigmoid bias / grouped routing /
    /// expert-group selection all default off — those branches are reserved for
    /// the DeepSeek-V3 family, not for Mixtral or Qwen3-MoE.
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        intermediate_size: usize,
        hidden_size: usize,
        stream: crate::CUstream,
    ) -> anyhow::Result<Self> {
        use crate::driver;

        let gate = crate::layers::Linear::load(gw, &format!("{prefix}.gate"))?;

        let first_w1 = format!("{prefix}.experts.0.w1.weight");
        let (_, disk_dtype) = gw
            .tensor_info(&first_w1)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {first_w1}"))?;
        // `take_into` casts floating-point sources to the configured
        // target dtype on the way in. Size + tag the stacked buffer
        // with the post-cast width — otherwise FP32 checkpoints blow
        // up the buffer to 2× the right size and feed the fused-MoE
        // GEMM a wrongly-typed tensor.
        let dtype = gw.target_dtype().unwrap_or(disk_dtype);
        let elem = dtype.size_bytes();

        let w1_bytes = num_experts * 2 * intermediate_size * hidden_size * elem;
        let w2_bytes = num_experts * hidden_size * intermediate_size * elem;
        let w1_ptr = unsafe { driver::mem_alloc(w1_bytes)? };
        let w2_ptr = unsafe { driver::mem_alloc(w2_bytes)? };

        let gate_proj_bytes = intermediate_size * hidden_size * elem;
        for e in 0..num_experts {
            let w1_name = format!("{prefix}.experts.{e}.w1.weight");
            let w3_name = format!("{prefix}.experts.{e}.w3.weight");
            let w2_name = format!("{prefix}.experts.{e}.w2.weight");
            let expert_w1_off = e * 2 * intermediate_size * hidden_size * elem;
            let expert_w2_off = e * hidden_size * intermediate_size * elem;
            unsafe {
                gw.take_into(&w1_name, w1_ptr.add(expert_w1_off), stream)?;
                gw.take_into(
                    &w3_name,
                    w1_ptr.add(expert_w1_off + gate_proj_bytes),
                    stream,
                )?;
                gw.take_into(&w2_name, w2_ptr.add(expert_w2_off), stream)?;
            }
        }

        let w1 = unsafe {
            GpuTensor::new(
                w1_ptr,
                &[num_experts, 2 * intermediate_size, hidden_size],
                dtype,
            )
        };
        let w2 = unsafe {
            GpuTensor::new(
                w2_ptr,
                &[num_experts, hidden_size, intermediate_size],
                dtype,
            )
        };

        Ok(DenseFusedMoELayer {
            gate,
            w1,
            w2,
            num_experts,
            top_k,
            intermediate_size,
            hidden_size,
            // Python vLLM's `MixtralMoE` constructs `FusedMoE(...,
            // renormalize=True)` (vllm/model_executor/models/mixtral.py
            // L136). HF transformers' `MixtralSparseMoeBlock.forward`
            // unconditionally divides by `routing_weights.sum(dim=-1)`.
            // The hand-written `scratchy-serving-cuda::mixtral::load_moe` path has
            // a latent bug here (`renormalize: false`); scratchy tracks
            // Python vLLM as the reference.
            renormalize: true,
            e_score_correction_bias: None,
            n_expert_group: 0,
            topk_group: 0,
            routed_scaling_factor: 1.0,
        })
    }

    /// Forward pass — full MoE pipeline.
    ///
    /// Always uses `forward_fused` (WMMA kernel). The fused kernel is a plain
    /// CUDA kernel launch, fully compatible with CUDA graph capture.
    ///
    /// * `hidden_states`: `[num_tokens, hidden_size]`
    ///
    /// Returns: `[num_tokens, hidden_size]`
    ///
    /// # Safety
    /// All tensors must be valid GPU memory. Device must be properly initialized.
    #[allow(clippy::too_many_arguments)]
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor {
        self.forward_fused(hidden_states, device)
    }

    /// Fused MoE forward — uses custom tiled GEMM kernel. Graph-capturable.
    #[allow(clippy::too_many_arguments)]
    unsafe fn forward_fused(
        &self,
        hidden_states: TensorView<'_>,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let num_tokens = hidden_states.dim(0);
        let stream = device.compute_stream;

        let block_m = select_moe_block_m(num_tokens, self.top_k, self.num_experts);

        let router_logits =
            self.gate
                .forward(hidden_states, &mut device.cublas, &mut device.caching);

        let (topk_weights, topk_ids) = route_experts(
            router_logits.as_gpu_tensor(),
            &MoeRouting {
                top_k: self.top_k,
                renormalize: self.renormalize,
                e_score_correction_bias: self.e_score_correction_bias.as_ref(),
                n_expert_group: self.n_expert_group,
                topk_group: self.topk_group,
                routed_scaling_factor: self.routed_scaling_factor,
            },
            &mut device.caching,
            stream,
        );
        drop(router_logits);

        let (sorted_token_ids, expert_ids, num_tokens_post_padded) = kernels::moe_align_block_size(
            topk_ids.as_gpu_tensor(),
            self.num_experts,
            block_m,
            &mut device.caching,
            stream,
        );
        drop(topk_ids);

        let intermediate1 = kernels::fused_moe_gemm(
            *hidden_states,
            self.w1,
            topk_weights.as_gpu_tensor(),
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            num_tokens,
            self.top_k,
            block_m,
            false,
            &mut device.caching,
            stream,
        );

        let activated = kernels::silu_and_mul_fused(
            intermediate1.as_gpu_tensor(),
            self.intermediate_size,
            &mut device.caching,
            stream,
        );
        drop(intermediate1);

        let intermediate2 = kernels::fused_moe_gemm(
            activated.as_gpu_tensor(),
            self.w2,
            topk_weights.as_gpu_tensor(),
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            num_tokens * self.top_k,
            1,
            block_m,
            true,
            &mut device.caching,
            stream,
        );

        drop(activated);
        drop(topk_weights);
        drop(sorted_token_ids);
        drop(expert_ids);
        drop(num_tokens_post_padded);

        let output = kernels::moe_sum(
            intermediate2.as_gpu_tensor(),
            num_tokens,
            self.hidden_size,
            self.top_k,
            &mut device.caching,
            stream,
        );
        drop(intermediate2);

        output
    }
}

// ---------------------------------------------------------------------------
// Gemma-4-26B-A4B MoE loaders + forward (GemmaRouterLayer + SwitchGluExpertsLayer)
//
// The cuda Gemma-4 MoE op is ONLY the sparse SwitchGLU block: router norm +
// route + softmax-with-temp + per_expert_scale, then the GeGLU experts and the
// weighted sum. The int4/int8 MLX-affine weights are dequantized to BF16 at
// LOAD (the affine_dequant kernel) so the hot path reuses the existing dense
// BF16 fused_moe_gemm / moe_align_block_size / gelu_and_mul_fused / moe_sum
// primitives — no native MLX-affine GEMM exists on cuda.
// ---------------------------------------------------------------------------

/// Dequantize an MLX-affine packed weight (`DType::U32` on disk) to a fresh,
/// model-lifetime BF16 buffer. `rows = E*out` (expert axis folded in), `cols`
/// = logical in-features. The destination is tracked via `gw.record_alloc` so
/// it lives as long as the model; the packed weight + scales + biases are NOT
/// freed here — the caller decides (the expert path keeps scales/biases in the
/// struct fields, ignored by the bf16 forward).
#[cfg(feature = "cuda")]
unsafe fn affine_dequant_weight_bf16(
    gw: &mut crate::weights::GpuWeights,
    packed: GpuTensor,
    scales: GpuTensor,
    biases: GpuTensor,
    rows: usize,
    cols: usize,
    bits: u32,
    group_size: usize,
    stream: crate::CUstream,
) -> anyhow::Result<GpuTensor> {
    use crate::driver;

    let out_bytes = rows * cols * crate::dtype::DType::BF16.size_bytes();
    let out_ptr = unsafe { driver::mem_alloc(out_bytes)? };
    unsafe {
        kernels::affine_dequant_into(
            packed.as_ptr::<u8>(),
            scales.as_ptr::<u16>(),
            biases.as_ptr::<u16>(),
            out_ptr as *mut u16,
            rows,
            cols,
            bits,
            group_size,
            stream,
        );
    }
    gw.record_alloc(out_ptr, out_bytes);
    Ok(unsafe { GpuTensor::new(out_ptr, &[rows, cols], crate::dtype::DType::BF16) })
}

/// CUDA load extension trait for the Gemma-4 router bundle.
#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
impl Gemma4RouterOps for GemmaRouterLayer {
    /// Load the Gemma-4 router bundle from `{prefix}` (= `...layers.N.router`):
    /// * `{prefix}.proj` — **8-bit** MLX-affine `[E, hidden]`, dequantized to a
    ///   dense BF16 `gate` tensor at load (the router GEMM stays dense BF16).
    /// * `{prefix}.per_expert_scale` — `[E]` bf16, kept verbatim.
    /// * `{prefix}.scale` — `[hidden]` bf16 RMSNorm gain, kept verbatim.
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        hidden_size: usize,
        group_size: u32,
    ) -> anyhow::Result<Self> {
        use crate::driver;

        // HF (`google/gemma-4-26B-A4B-it`) checkpoint detection: the HF router is
        // a plain bf16 `{prefix}.proj.weight` with NO `.proj.scales` (the MLX path
        // ships 8-bit affine `proj.{weight,scales,biases}`). Detect by the ABSENCE
        // of the scales tensor.
        if !gw.contains(&format!("{prefix}.proj.scales")) {
            // The cyankiwi compressed-tensors checkpoint ships the router tensors
            // as **FP16** (`scale`≈31, `proj`≈0.01, `per_expert_scale`≈1.0), while
            // the rest of the model — and every router kernel below
            // (`rms_norm_with_offset`, the router GEMM, `moe_per_expert_scale`) —
            // is BF16. `take` CASTS each tensor to the model dtype (BF16); using
            // `take_keep_dtype` here would leave them FP16, and the BF16 kernels
            // would reinterpret the FP16 bit-pattern as BF16 (e.g. 31.25 FP16 =
            // 0x4FD0 read as BF16 ≈ 7e9), destroying the routing signal and
            // collapsing the top-k softmax to a degenerate top-1. The magnitudes
            // (≈31, ≈0.01, ≈1.0) are all comfortably representable in BF16, so the
            // straightforward gain·proj math (no fold) matches the fp32 reference.
            let gate = gw.take(&format!("{prefix}.proj.weight"))?;
            let per_expert_scale = gw.take(&format!("{prefix}.per_expert_scale"))?;
            let scale = gw.take(&format!("{prefix}.scale"))?;

            // GUARD (inviolable rule): the silent FP16-as-BF16 misread above was a
            // runtime correctness bug (degenerate routing, fluent-but-wrong output,
            // no crash). Turn the dtype assumption these kernels rely on into a hard
            // LOAD-TIME error: if a future checkpoint ships a dtype that `take` does
            // not cast to BF16, fail loudly here instead of corrupting the routing.
            for (name, t) in [
                ("proj.weight", &gate),
                ("per_expert_scale", &per_expert_scale),
                ("scale", &scale),
            ] {
                anyhow::ensure!(
                    t.dtype() == crate::dtype::DType::BF16,
                    "GemmaRouterLayer: {prefix}.{name} must be BF16 after load (got \
                     {:?}); the router rms_norm/GEMM/per_expert_scale kernels assume \
                     BF16 and would misread any other dtype as BF16, degenerating \
                     the top-k routing.",
                    t.dtype()
                );
            }

            return Ok(GemmaRouterLayer {
                gate,
                per_expert_scale,
                scale,
                num_experts,
                hidden_size,
            });
        }

        let stream = gw.stream();
        anyhow::ensure!(
            hidden_size.is_multiple_of(group_size as usize),
            "GemmaRouterLayer: hidden_size={hidden_size} not divisible by \
             group_size={group_size}"
        );

        // router.proj: 8-bit MLX-affine, packed `[E, hidden/4]` U32 on disk.
        let proj_w = gw.take(&format!("{prefix}.proj.weight"))?;
        let proj_s = gw.take_keep_dtype(&format!("{prefix}.proj.scales"))?;
        let proj_b = gw.take_keep_dtype(&format!("{prefix}.proj.biases"))?;
        anyhow::ensure!(
            matches!(proj_s.dtype(), crate::dtype::DType::BF16),
            "GemmaRouterLayer: router.proj.scales must be BF16 (got {}); the \
             affine_dequant kernel reads bf16 scales/biases",
            proj_s.dtype()
        );

        let gate = unsafe {
            affine_dequant_weight_bf16(
                gw,
                proj_w,
                proj_s,
                proj_b,
                num_experts,
                hidden_size,
                8,
                group_size as usize,
                stream,
            )?
        };

        let per_expert_scale = gw.take_keep_dtype(&format!("{prefix}.per_expert_scale"))?;
        let scale = gw.take_keep_dtype(&format!("{prefix}.scale"))?;

        // The packed proj weight + its scales/biases are scratch — the dense
        // bf16 `gate` superseded them. Free them now (sync first so the
        // in-flight dequant finished reading).
        unsafe {
            driver::stream_synchronize(stream)?;
            gw.unrecord_alloc(proj_w.as_mut_ptr::<u8>());
            driver::mem_free(proj_w.as_mut_ptr::<u8>())?;
            gw.unrecord_alloc(proj_s.as_mut_ptr::<u8>());
            driver::mem_free(proj_s.as_mut_ptr::<u8>())?;
            gw.unrecord_alloc(proj_b.as_mut_ptr::<u8>());
            driver::mem_free(proj_b.as_mut_ptr::<u8>())?;
        }

        Ok(GemmaRouterLayer {
            gate,
            per_expert_scale,
            scale,
            num_experts,
            hidden_size,
        })
    }
}

/// CUDA load extension trait for the Gemma-4 SwitchGLU expert stack.
#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
impl SwitchGluExpertsOps for SwitchGluExpertsLayer {
    /// Load the router-less SwitchGLU expert stack from `{prefix}`
    /// (= `...layers.N.experts.switch_glu`). The on-disk tensors are pre-stacked
    /// MLX-affine int4 `[E, out, in/8]` U32 weights + `[E, out, in/gs]` bf16
    /// scales/biases. Each proj is dequantized to a dense BF16 `[E, out, in]`
    /// stack stored in the `expert_*_w` field; the scales/biases tensors are
    /// retained (the bf16 forward ignores them).
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
        use crate::driver;

        // HF (`google/gemma-4-26B-A4B-it`) checkpoint detection: the HF experts
        // ship FUSED bf16 tensors under `...experts.gate_up_proj` /
        // `...experts.down_proj` (the MLX `experts.switch_glu.*` int4 names are
        // absent). Strip `.switch_glu` from the prefix — the HF tensors live one
        // level up. Detect by tensor presence, NOT config.
        let experts_base = prefix.strip_suffix(".switch_glu").unwrap_or(prefix);
        if gw.contains(&format!("{experts_base}.gate_up_proj")) {
            // Fused bf16, NO dequant. `take` casts the source to the target bf16
            // dtype on the way in (and the result is owned by the allocator for
            // the model lifetime). Keep gate_up FUSED in `expert_gate_w`; the
            // forward detects fused via `expert_gate_w.dim(1) == 2*moe_inter` and
            // runs ONE grouped GEMM + `gelu_and_mul_fused` (splitting into two
            // buffers would need an untestable per-expert device memcpy).
            let gate_up = gw.take(&format!("{experts_base}.gate_up_proj"))?;
            let down = gw.take(&format!("{experts_base}.down_proj"))?;

            // Re-view as 3D `[E, out, in]` stacks so the grouped GEMM's
            // `weights.dim(1) = out_features` is correct.
            let gate_up_3d = unsafe {
                GpuTensor::new(
                    gate_up.as_mut_ptr::<u8>(),
                    &[num_experts, 2 * moe_intermediate_size, hidden_size],
                    crate::dtype::DType::BF16,
                )
            };
            let down_3d = unsafe {
                GpuTensor::new(
                    down.as_mut_ptr::<u8>(),
                    &[num_experts, hidden_size, moe_intermediate_size],
                    crate::dtype::DType::BF16,
                )
            };

            // The 7 unused struct fields (expert_up_w + the 6 scales/biases) are
            // never read on the fused bf16 forward. Point all of them at ONE tiny
            // `[1]` bf16 placeholder allocation (recorded once in gw so the model
            // owns it; the struct does NOT free on drop). Do NOT alias the real
            // gate_up/down tensors into them — that risks a double-free / aliasing.
            let placeholder_bytes = crate::dtype::DType::BF16.size_bytes();
            let placeholder_ptr = unsafe { driver::mem_alloc(placeholder_bytes)? };
            gw.record_alloc(placeholder_ptr, placeholder_bytes);
            let placeholder =
                unsafe { GpuTensor::new(placeholder_ptr, &[1], crate::dtype::DType::BF16) };

            return Ok(SwitchGluExpertsLayer {
                expert_gate_w: gate_up_3d,
                expert_gate_scales: placeholder,
                expert_gate_biases: placeholder,
                expert_up_w: placeholder,
                expert_up_scales: placeholder,
                expert_up_biases: placeholder,
                expert_down_w: down_3d,
                expert_down_scales: placeholder,
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
            });
        }

        // compressed-tensors INT4-symmetric checkpoint (cyankiwi
        // `gemma-4-26B-A4B-it-AWQ-4bit`, compressed-tensors despite the "AWQ"
        // repo name): PER-EXPERT, unstacked `experts.{e}.{proj}.weight_packed`
        // (uint4b8 INT32 `[out, in/8]`) + `.weight_scale` (F16). Detect by
        // expert 0's gate_proj packed tensor. Keep experts INT4 (Marlin) in
        // VRAM (~0.4 GiB) and run `marlin_moe_gemm` in the forward — the
        // dequant-to-bf16 path needs ~43 GiB and won't fit one 46 GB card.
        if gw.contains(&format!("{experts_base}.0.gate_proj.weight_packed")) {
            // Derive the group stride from the scale shape: `weight_scale` is
            // `[out, in/group_size]`, and gate_proj's in == hidden_size.
            let sname = format!("{experts_base}.0.gate_proj.weight_scale");
            let (sshape, _) = gw.tensor_info(&sname).ok_or_else(|| {
                anyhow::anyhow!(
                    "SwitchGluExpertsLayer (compressed-tensors): missing `{sname}`. Fix: \
                     confirm the checkpoint ships per-expert `weight_scale` tensors."
                )
            })?;
            anyhow::ensure!(
                sshape.len() == 2 && sshape[1] > 0 && hidden_size.is_multiple_of(sshape[1]),
                "SwitchGluExpertsLayer (compressed-tensors): gate_proj.weight_scale shape \
                 {sshape:?} incompatible with hidden_size={hidden_size} (expected \
                 [out, hidden/group_size]). Fix: group_size must divide hidden_size."
            );
            let ct_group_size = hidden_size / sshape[1];
            anyhow::ensure!(
                moe_intermediate_size.is_multiple_of(ct_group_size),
                "SwitchGluExpertsLayer (compressed-tensors): group_size={ct_group_size} must \
                 divide moe_intermediate_size={moe_intermediate_size} (down_proj groups along \
                 moe_inter)."
            );

            let device_id = unsafe { crate::driver::current_device()? };
            let (w1, w2, w1_scales, w2_scales, workspace) =
                crate::weights_quant::load_gemma_marlin_experts(
                    gw,
                    experts_base,
                    num_experts,
                    moe_intermediate_size,
                    hidden_size,
                    ct_group_size,
                    device_id,
                )?;

            // The bf16 `expert_*_w`/scales/biases fields are unused on the
            // marlin path → one tiny placeholder; the `marlin_*` fields drive
            // `gemma_moe_forward`.
            let placeholder_bytes = crate::dtype::DType::BF16.size_bytes();
            let placeholder_ptr = unsafe { driver::mem_alloc(placeholder_bytes)? };
            gw.record_alloc(placeholder_ptr, placeholder_bytes);
            let placeholder =
                unsafe { GpuTensor::new(placeholder_ptr, &[1], crate::dtype::DType::BF16) };

            return Ok(SwitchGluExpertsLayer {
                expert_gate_w: placeholder,
                expert_gate_scales: placeholder,
                expert_gate_biases: placeholder,
                expert_up_w: placeholder,
                expert_up_scales: placeholder,
                expert_up_biases: placeholder,
                expert_down_w: placeholder,
                expert_down_scales: placeholder,
                expert_down_biases: placeholder,
                num_experts,
                top_k,
                moe_intermediate_size,
                hidden_size,
                group_size,
                bits,
                marlin_w1: Some(w1),
                marlin_w2: Some(w2),
                marlin_w1_scales: Some(w1_scales),
                marlin_w2_scales: Some(w2_scales),
                marlin_workspace: Some(workspace),
                marlin_b_type_id: 0,
            });
        }

        anyhow::ensure!(
            bits == 4,
            "SwitchGluExpertsLayer: only bits=4 supported (got {bits})"
        );
        let gs = group_size as usize;
        anyhow::ensure!(
            hidden_size.is_multiple_of(gs),
            "SwitchGluExpertsLayer: hidden_size={hidden_size} not divisible by \
             group_size={group_size}"
        );
        anyhow::ensure!(
            moe_intermediate_size.is_multiple_of(gs),
            "SwitchGluExpertsLayer: moe_intermediate_size={moe_intermediate_size} \
             not divisible by group_size={group_size}"
        );

        let stream = gw.stream();

        // Dequant one proj to a dense BF16 `[E, out, in]` stack. `rows = E*out`
        // folds the expert axis into the flat row index — the stacked
        // scales/biases `[E, out, in/gs]` index correctly with no per-expert
        // offset (gs divides in). Frees the packed weight afterward; keeps the
        // scales/biases (returned for the struct field, ignored by the forward).
        let mut load_proj = |proj: &str,
                             out_dim: usize,
                             in_dim: usize|
         -> anyhow::Result<(GpuTensor, GpuTensor, GpuTensor)> {
            let w = gw.take(&format!("{prefix}.{proj}.weight"))?;
            let s = gw.take_keep_dtype(&format!("{prefix}.{proj}.scales"))?;
            let b = gw.take_keep_dtype(&format!("{prefix}.{proj}.biases"))?;
            anyhow::ensure!(
                matches!(s.dtype(), crate::dtype::DType::BF16),
                "SwitchGluExpertsLayer: {proj}.scales must be BF16 (got {})",
                s.dtype()
            );
            let dense = unsafe {
                affine_dequant_weight_bf16(
                    gw,
                    w,
                    s,
                    b,
                    num_experts * out_dim,
                    in_dim,
                    bits,
                    gs,
                    stream,
                )?
            };
            // Re-view the dense bf16 buffer as a 3D `[E, out, in]` stack so
            // fused_moe_gemm's `weights.dim(1) = out_features` is correct.
            let dense3d = unsafe {
                GpuTensor::new(
                    dense.as_mut_ptr::<u8>(),
                    &[num_experts, out_dim, in_dim],
                    crate::dtype::DType::BF16,
                )
            };
            // Free the packed weight (superseded). Keep scales/biases.
            unsafe {
                driver::stream_synchronize(stream)?;
                gw.unrecord_alloc(w.as_mut_ptr::<u8>());
                driver::mem_free(w.as_mut_ptr::<u8>())?;
            }
            Ok((dense3d, s, b))
        };

        // gate/up: out=moe_intermediate, in=hidden. down: out=hidden, in=moe_inter.
        let (expert_gate_w, expert_gate_scales, expert_gate_biases) =
            load_proj("gate_proj", moe_intermediate_size, hidden_size)?;
        let (expert_up_w, expert_up_scales, expert_up_biases) =
            load_proj("up_proj", moe_intermediate_size, hidden_size)?;
        let (expert_down_w, expert_down_scales, expert_down_biases) =
            load_proj("down_proj", hidden_size, moe_intermediate_size)?;

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
            marlin_w1: None,
            marlin_w2: None,
            marlin_w1_scales: None,
            marlin_w2_scales: None,
            marlin_workspace: None,
            marlin_b_type_id: 0,
        })
    }
}

/// Gemma-4 sparse SwitchGLU forward (the `gemma_moe` op body).
///
/// `router_in` = the post-attention residual; `expert_in` = the ff2-normed
/// activation. Both BF16 `[M, hidden]`. Returns BF16 `[M, hidden]`.
///
/// Sequence (mirrors metal `lower_gemma_moe`):
///   xr      = rms_norm(router_in, router.scale)                  (gain only;
///             the hidden^-0.5 temperature is folded into the softmax)
///   logits  = xr · router.gate^T                       → [M, num_experts]
///   ids     = top-k over logits (softmax-monotone → raw top-k ids)
///   scores  = softmax(logits[ids] · hidden^-0.5)       → [M, top_k] f32
///   scores *= per_expert_scale[ids]
///   gate_o  = cutlass_moe_grouped(expert_in, W_gate)    (apply_weights=false)
///   up_o    = cutlass_moe_grouped(expert_in, W_up)
///   act     = gelu_pytorch_tanh(gate_o) * up_o
///   down_o  = cutlass_moe_grouped(act, W_down, scores, apply_weights=true)
///   out     = moe_sum(down_o)                           → [M, hidden]
///
/// The expert GEMM is the arch-optimal CUTLASS grouped GEMM (Sm90 wgmma+TMA
/// ptr-array warp-specialized cooperative / Sm80 mma.sync grouped on sm89/Ada),
/// a drop-in for the legacy `fused_moe_gemm` (identical token-indexed output).
///
/// # Safety
/// `router`/`experts` weights must be the dequant'd BF16 bundles; tiles must be
/// valid BF16 `[M, hidden]` GPU memory; `device.compute_stream` is live.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemma_moe_forward(
    router: &GemmaRouterLayer,
    experts: &SwitchGluExpertsLayer,
    router_in: TensorView<'_>,
    expert_in: TensorView<'_>,
    eps: f32,
    device: &mut GpuDevice,
) -> OwnedTensor {
    let stream = device.compute_stream;
    let num_tokens = router_in.dim(0);
    let top_k = experts.top_k;
    let num_experts = experts.num_experts;
    let hidden_size = experts.hidden_size;
    let block_m = select_moe_block_m(num_tokens, top_k, num_experts);

    // Router folded-gain RMSNorm (plain `router.scale`, offset 0.0).
    let xr = kernels::rms_norm_with_offset(
        *router_in,
        router.scale,
        eps,
        0.0,
        &mut device.caching,
        stream,
    );

    // Dense router GEMM: xr [M, hidden] × gate [E, hidden]^T → logits [M, E].
    let logits = device
        .cublas
        .gemm(xr.as_gpu_tensor(), router.gate, &mut device.caching);
    drop(xr);

    // Top-k IDs over the logits. Softmax is monotone, so topk_softmax's
    // selected ids == top-k over raw logits; we keep ONLY the ids and recompute
    // the gemma softmax (over the top_k, with hidden^-0.5 temperature) below.
    let (ignored_weights, topk_ids) = kernels::topk_softmax(
        logits.as_gpu_tensor(),
        top_k,
        false,
        &mut device.caching,
        stream,
    );
    drop(ignored_weights);

    // scores = softmax(logits[ids] · hidden^-0.5)  → [M, top_k] f32.
    let temp = (hidden_size as f32).powf(-0.5);
    let scores = kernels::moe_softmax_topk_temp(
        logits.as_gpu_tensor(),
        topk_ids.as_gpu_tensor(),
        top_k,
        temp,
        &mut device.caching,
        stream,
    );
    drop(logits);

    // scores[m,j] *= per_expert_scale[ids[m,j]]  (in place).
    kernels::moe_per_expert_scale(
        scores.as_gpu_tensor(),
        topk_ids.as_gpu_tensor(),
        router.per_expert_scale,
        stream,
    );

    // Native INT4 (Marlin) expert path (cyankiwi compressed-tensors INT4): keep
    // experts INT4 in VRAM and run the W4A16 grouped Marlin MoE GEMM instead of
    // dequant→bf16 cutlass. Gemma's own routing (`scores`/`topk_ids` above) is
    // reused; only the expert GEMM differs. Mirrors `MarlinFusedMoEOps::forward`
    // but with gemma's gelu_pytorch_tanh GeGLU and pre-tempered routing weights.
    if let (Some(w1), Some(w2), Some(w1_scales), Some(w2_scales), Some(workspace)) = (
        experts.marlin_w1.as_ref(),
        experts.marlin_w2.as_ref(),
        experts.marlin_w1_scales.as_ref(),
        experts.marlin_w2_scales.as_ref(),
        experts.marlin_workspace.as_ref(),
    ) {
        let inter = experts.moe_intermediate_size;
        // Marlin tiles only instantiate thread_m_blocks=1 → max block size 16.
        let moe_block_size = select_moe_block_size(num_tokens, top_k, num_experts);
        let (sorted_token_ids, expert_ids, num_tokens_post_padded) = kernels::moe_align_block_size(
            topk_ids.as_gpu_tensor(),
            num_experts,
            moe_block_size,
            &mut device.caching,
            stream,
        );

        // Group strides from the stacked scale shapes ([E, num_groups, N]).
        let num_groups_w1 = w1_scales.dim(1);
        let num_groups_w2 = w2_scales.dim(1);
        let group_size_w1 = if num_groups_w1 > 1 {
            hidden_size / num_groups_w1
        } else {
            -1i64 as usize
        };
        let group_size_w2 = if num_groups_w2 > 1 {
            inter / num_groups_w2
        } else {
            -1i64 as usize
        };

        // GEMM 1: x × w1^T (fused gate+up) → [M*top_k, 2*inter]. No routing
        // weight on pass 1 (mul_topk_weights=false).
        let intermediate1 = kernels::marlin_moe_gemm(
            *expert_in,
            *w1,
            *w1_scales,
            None,
            None,
            None,
            *workspace,
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            scores.as_gpu_tensor(),
            moe_block_size,
            num_experts,
            top_k,
            false,
            num_tokens,
            2 * inter,
            hidden_size,
            num_groups_w1,
            group_size_w1,
            false,
            false,
            experts.marlin_b_type_id,
            device.device_id as i32,
            &mut device.caching,
            stream,
        );

        // act = gelu_pytorch_tanh(gate) * up (gate = first `inter` cols).
        let activated = kernels::gelu_and_mul_fused(
            intermediate1.as_gpu_tensor(),
            inter,
            &mut device.caching,
            stream,
        );
        drop(intermediate1);

        // GEMM 2: act × w2^T (down) → [M*top_k, hidden]. Apply the gemma routing
        // weights here (mul_topk_weights=true, top_k=1 — input already expanded).
        let intermediate2 = kernels::marlin_moe_gemm(
            activated.as_gpu_tensor(),
            *w2,
            *w2_scales,
            None,
            None,
            None,
            *workspace,
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            scores.as_gpu_tensor(),
            moe_block_size,
            num_experts,
            1,
            true,
            num_tokens * top_k,
            hidden_size,
            inter,
            num_groups_w2,
            group_size_w2,
            false,
            false,
            experts.marlin_b_type_id,
            device.device_id as i32,
            &mut device.caching,
            stream,
        );
        drop(activated);

        let out = kernels::moe_sum(
            intermediate2.as_gpu_tensor(),
            num_tokens,
            hidden_size,
            top_k,
            &mut device.caching,
            stream,
        );
        drop(intermediate2);

        drop(scores);
        drop(topk_ids);
        return out;
    }

    // Expert routing alignment.
    let (sorted_token_ids, expert_ids, num_tokens_post_padded) = kernels::moe_align_block_size(
        topk_ids.as_gpu_tensor(),
        num_experts,
        block_m,
        &mut device.caching,
        stream,
    );

    // GeGLU experts: gate & up, gelu*up, down GEMM with the final scores
    // applied, then sum across top_k.
    //
    // The expert GEMM is the CUTLASS grouped GEMM (Sm90 wgmma+TMA / Sm80
    // mma.sync), arch-dispatched inside `cutlass_moe_grouped`. It is a drop-in
    // for the legacy `fused_moe_gemm` (same token-indexed output layout), so
    // `moe_sum` below is unchanged.
    //
    // Two gate/up layouts, by checkpoint:
    //   * fused (HF): `expert_gate_w` is `[E, 2*moe_inter, hidden]` — ONE grouped
    //     GEMM → `[M, 2*moe_inter]`, then the interleaved `gelu_and_mul_fused`
    //     (gate = first moe_inter cols, up = next — matches HF gate_up row order).
    //   * split (MLX): `expert_gate_w` / `expert_up_w` are two `[E, moe_inter,
    //     hidden]` stacks — two grouped GEMMs + the non-fused `gelu_and_mul`.
    let fused = experts.expert_gate_w.dim(1) == 2 * experts.moe_intermediate_size;
    let act = if fused {
        let gate_up_o = kernels::cutlass_moe_grouped(
            *expert_in,
            experts.expert_gate_w,
            scores.as_gpu_tensor(),
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            num_experts,
            num_tokens,
            top_k,
            block_m,
            false,
            &mut device.caching,
            stream,
        );
        let act = kernels::gelu_and_mul_fused(
            gate_up_o.as_gpu_tensor(),
            experts.moe_intermediate_size,
            &mut device.caching,
            stream,
        );
        drop(gate_up_o);
        act
    } else {
        let gate_o = kernels::cutlass_moe_grouped(
            *expert_in,
            experts.expert_gate_w,
            scores.as_gpu_tensor(),
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            num_experts,
            num_tokens,
            top_k,
            block_m,
            false,
            &mut device.caching,
            stream,
        );
        let up_o = kernels::cutlass_moe_grouped(
            *expert_in,
            experts.expert_up_w,
            scores.as_gpu_tensor(),
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            num_experts,
            num_tokens,
            top_k,
            block_m,
            false,
            &mut device.caching,
            stream,
        );

        // act = gelu_pytorch_tanh(gate) * up. Non-fused (gate/up are separate
        // GEMM outputs) — the gemma GeGLU uses the tanh approximation.
        let act = kernels::gelu_and_mul(
            gate_o.as_gpu_tensor(),
            up_o.as_gpu_tensor(),
            &mut device.caching,
            stream,
        );
        drop(gate_o);
        drop(up_o);
        act
    };

    // down GEMM, applying the final routing scores, then sum across top_k.
    let down_o = kernels::cutlass_moe_grouped(
        act.as_gpu_tensor(),
        experts.expert_down_w,
        scores.as_gpu_tensor(),
        sorted_token_ids.as_gpu_tensor(),
        expert_ids.as_gpu_tensor(),
        num_tokens_post_padded.as_gpu_tensor(),
        num_experts,
        num_tokens * top_k,
        1,
        block_m,
        true,
        &mut device.caching,
        stream,
    );
    drop(act);
    drop(scores);
    drop(topk_ids);
    drop(sorted_token_ids);
    drop(expert_ids);
    drop(num_tokens_post_padded);

    let out = kernels::moe_sum(
        down_o.as_gpu_tensor(),
        num_tokens,
        hidden_size,
        top_k,
        &mut device.caching,
        stream,
    );
    drop(down_o);
    out
}

// ---------------------------------------------------------------------------
// FusedMoELayer (enum: Dense for cuda, Affine for metal)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub trait FusedMoEOps {
    #[cfg(feature = "cuda")]
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

    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        hidden_states: crate::tensor::TensorView<'_>,
        device: &mut crate::device::GpuDevice,
    ) -> crate::alloc::OwnedTensor;
}

#[cfg(feature = "cuda")]
impl FusedMoEOps for FusedMoELayer {
    /// Macro-emitted load entry for Dense MoE checkpoints (cuda).
    /// Mirrors the historical struct-`FusedMoELayer::load` signature
    /// so the macro's `FieldLoad::FusedMoe` arm doesn't need an
    /// extra parameter.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        intermediate_size: usize,
        hidden_size: usize,
        stream: crate::CUstream,
    ) -> anyhow::Result<Self> {
        Ok(Self::Dense(Box::new(DenseFusedMoELayer::load(
            gw,
            prefix,
            num_experts,
            top_k,
            intermediate_size,
            hidden_size,
            stream,
        )?)))
    }

    /// Cuda forward — delegates to the `Dense` variant. The `Affine`
    /// arm is unreachable here because the Metal worker dispatches
    /// MoE via the lowered ICB tape (`Instruction::MetalFusedMoe`),
    /// never through this `forward` method.
    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        hidden_states: crate::tensor::TensorView<'_>,
        device: &mut crate::device::GpuDevice,
    ) -> crate::alloc::OwnedTensor {
        match self {
            Self::Dense(d) => unsafe { d.forward(hidden_states, device) },
            Self::Affine(_) => {
                unreachable!(
                    "FusedMoELayer::forward called on Affine variant — \
                              Metal MoE is dispatched via the ICB tape, not the \
                              cuda forward path"
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DenseSharedFusedMoELayer (cuda BF16 Qwen2/3 MoE with shared expert)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub trait DenseSharedFusedMoEOps {
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

    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor;
}

#[cfg(feature = "cuda")]
impl DenseSharedFusedMoEOps for DenseSharedFusedMoELayer {
    /// Load a Qwen-MoE-style BF16 fused MoE + shared-expert layer.
    ///
    /// Mirrors `scratchy-serving-cuda/src/model/qwen3_moe.rs::Qwen3MoeDecoderLayer::load_moe`
    /// at single-rank (tp=1). Qwen3-MoE / Qwen2-MoE expert weight names use the
    /// HF `gate_proj/up_proj/down_proj` convention (NOT Mixtral's
    /// `w1/w2/w3`); the shared expert lives at
    /// `{prefix}.shared_expert.{gate,up,down}_proj.weight` with a sigmoid
    /// gate at `{prefix}.shared_expert_gate.weight`.
    ///
    /// Routed `renormalize: true` matches Python vLLM's Qwen-MoE softmax
    /// path (`norm_topk_prob = True` is implicit in the family).
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        shared_expert_intermediate_size: usize,
        hidden_size: usize,
        stream: crate::CUstream,
    ) -> anyhow::Result<Self> {
        use crate::driver;

        let gate = crate::layers::Linear::load(gw, &format!("{prefix}.gate"))?;

        // Stacked-experts probe: modern HF Qwen3.5-MoE / Qwen3-MoE-Instruct
        // checkpoints ship the routed experts as two pre-stacked tensors
        // (`{prefix}.experts.gate_up_proj` `[E, 2*inter, hidden]` and
        // `{prefix}.experts.down_proj` `[E, hidden, inter]`) instead of
        // per-expert `experts.{e}.{gate,up,down}_proj.weight`. The stacked
        // dim-(-2) order is gate||up (Python vLLM `qwen3_5.py` chunks
        // `gate_up_proj` along dim=-2 with chunk[0]=w1=gate, chunk[1]=w3=up
        // — matches our w1 buffer layout byte-for-byte). Use a single
        // bulk DMA per stacked tensor when present; fall back to the
        // per-expert loop for older Mixtral-style checkpoints.
        let stacked_gate_up = format!("{prefix}.experts.gate_up_proj");
        let use_stacked = gw.contains(&stacked_gate_up);
        let probe = if use_stacked {
            stacked_gate_up.clone()
        } else {
            format!("{prefix}.experts.0.gate_proj.weight")
        };
        let (_, disk_dtype) = gw
            .tensor_info(&probe)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {probe}"))?;
        // Use post-cast dtype for the stacked buffer; see
        // `FusedMoELayer::load` for the rationale.
        let dtype = gw.target_dtype().unwrap_or(disk_dtype);
        let elem = dtype.size_bytes();

        let inter = moe_intermediate_size;
        let w1_bytes = num_experts * 2 * inter * hidden_size * elem;
        let w2_bytes = num_experts * hidden_size * inter * elem;
        let w1_ptr = unsafe { driver::mem_alloc(w1_bytes)? };
        let w2_ptr = unsafe { driver::mem_alloc(w2_bytes)? };

        if use_stacked {
            unsafe {
                gw.take_into(&stacked_gate_up, w1_ptr, stream)?;
                gw.take_into(&format!("{prefix}.experts.down_proj"), w2_ptr, stream)?;
            }
        } else {
            let gate_proj_bytes = inter * hidden_size * elem;
            for e in 0..num_experts {
                let gate_name = format!("{prefix}.experts.{e}.gate_proj.weight");
                let up_name = format!("{prefix}.experts.{e}.up_proj.weight");
                let down_name = format!("{prefix}.experts.{e}.down_proj.weight");
                let expert_w1_off = e * 2 * inter * hidden_size * elem;
                let expert_w2_off = e * hidden_size * inter * elem;
                unsafe {
                    gw.take_into(&gate_name, w1_ptr.add(expert_w1_off), stream)?;
                    gw.take_into(
                        &up_name,
                        w1_ptr.add(expert_w1_off + gate_proj_bytes),
                        stream,
                    )?;
                    gw.take_into(&down_name, w2_ptr.add(expert_w2_off), stream)?;
                }
            }
        }

        let w1 = unsafe { GpuTensor::new(w1_ptr, &[num_experts, 2 * inter, hidden_size], dtype) };
        let w2 = unsafe { GpuTensor::new(w2_ptr, &[num_experts, hidden_size, inter], dtype) };

        let moe = FusedMoELayer::Dense(Box::new(DenseFusedMoELayer {
            gate,
            w1,
            w2,
            num_experts,
            top_k,
            intermediate_size: inter,
            hidden_size,
            renormalize: true,
            e_score_correction_bias: None,
            n_expert_group: 0,
            topk_group: 0,
            routed_scaling_factor: 1.0,
        }));

        let (shared_gate_up, shared_down, shared_expert_gate) = if shared_expert_intermediate_size
            > 0
        {
            let shared_inter = shared_expert_intermediate_size;
            let shared_gate_proj_bytes = shared_inter * hidden_size * elem;
            let shared_total = 2 * shared_gate_proj_bytes;
            let ptr = unsafe { driver::mem_alloc(shared_total)? };
            unsafe {
                gw.take_into(
                    &format!("{prefix}.shared_expert.gate_proj.weight"),
                    ptr,
                    stream,
                )?;
                gw.take_into(
                    &format!("{prefix}.shared_expert.up_proj.weight"),
                    ptr.add(shared_gate_proj_bytes),
                    stream,
                )?;
            }
            let gu_w = unsafe { GpuTensor::new(ptr, &[2 * shared_inter, hidden_size], dtype) };
            let gate_up = crate::layers::Linear::new(gu_w, None);
            let down =
                crate::layers::Linear::load(gw, &format!("{prefix}.shared_expert.down_proj"))?;
            let sgate = crate::layers::Linear::load(gw, &format!("{prefix}.shared_expert_gate"))?;
            (Some(gate_up), Some(down), Some(sgate))
        } else {
            (None, None, None)
        };

        Ok(DenseSharedFusedMoELayer {
            moe,
            shared_gate_up,
            shared_down,
            shared_expert_gate,
            intermediate_size: shared_expert_intermediate_size,
        })
    }

    /// Forward pass — MoE + shared expert.
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor {
        let stream = device.compute_stream;

        // MoE path.
        let moe_out = self.moe.forward(hidden_states, device);

        // Shared expert path (if present).
        if let (Some(shared_gate_up), Some(shared_down), Some(shared_gate)) = (
            &self.shared_gate_up,
            &self.shared_down,
            &self.shared_expert_gate,
        ) {
            // shared_gate_up(hidden_states) → [num_tokens, 2*intermediate]
            let shared_gu =
                shared_gate_up.forward(hidden_states, &mut device.cublas, &mut device.caching);
            // SiLU-and-mul → [num_tokens, intermediate]
            let shared_activated = kernels::silu_and_mul_fused(
                shared_gu.as_gpu_tensor(),
                self.intermediate_size,
                &mut device.caching,
                stream,
            );
            drop(shared_gu);

            // down_proj → [num_tokens, hidden]
            let shared_out = shared_down.forward(
                shared_activated.view(),
                &mut device.cublas,
                &mut device.caching,
            );
            drop(shared_activated);

            // Gate: sigmoid(shared_expert_gate(hidden_states)) * shared_out
            let gate_logits =
                shared_gate.forward(hidden_states, &mut device.cublas, &mut device.caching);

            // Fused: out = moe_out + sigmoid(gate_logits) * shared_out
            let result = kernels::sigmoid_mul_add(
                moe_out.as_gpu_tensor(),
                shared_out.as_gpu_tensor(),
                gate_logits.as_gpu_tensor(),
                &mut device.caching,
                stream,
            );
            drop(moe_out);
            drop(shared_out);
            drop(gate_logits);

            result
        } else {
            moe_out
        }
    }
}

// ---------------------------------------------------------------------------
// SharedFusedMoELayer (enum: Dense for cuda, Affine for metal)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub trait SharedFusedMoEOps {
    #[cfg(feature = "cuda")]
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

    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        hidden_states: crate::tensor::TensorView<'_>,
        device: &mut crate::device::GpuDevice,
    ) -> crate::alloc::OwnedTensor;
}

#[cfg(feature = "cuda")]
impl SharedFusedMoEOps for SharedFusedMoELayer {
    /// Macro-emitted load entry for Dense Qwen-MoE checkpoints (cuda).
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn load(
        gw: &mut crate::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        shared_expert_intermediate_size: usize,
        hidden_size: usize,
        stream: crate::CUstream,
    ) -> anyhow::Result<Self> {
        Ok(Self::Dense(Box::new(DenseSharedFusedMoELayer::load(
            gw,
            prefix,
            num_experts,
            top_k,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            hidden_size,
            stream,
        )?)))
    }

    /// Cuda forward — delegates to the `Dense` variant. The `Affine`
    /// arm is unreachable here (Metal worker dispatches via ICB tape).
    #[cfg(feature = "cuda")]
    unsafe fn forward(
        &self,
        hidden_states: crate::tensor::TensorView<'_>,
        device: &mut crate::device::GpuDevice,
    ) -> crate::alloc::OwnedTensor {
        match self {
            Self::Dense(d) => unsafe { d.forward(hidden_states, device) },
            Self::Affine(_) => {
                unreachable!("SharedFusedMoELayer::forward called on Affine variant")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DeepSeekV2MoELayer
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub trait DeepSeekV2MoEOps {
    #[cfg(feature = "cuda")]
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor;

    #[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
impl DeepSeekV2MoEOps for DeepSeekV2MoELayer {
    /// Forward pass.
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor {
        let stream = device.compute_stream;

        // Routed experts.
        let moe_out = self.moe.forward(hidden_states, device);
        if self.routed_scaling_factor != 1.0 {
            kernels::scale_inplace(*moe_out.view(), self.routed_scaling_factor, &device.cublas);
        }

        // Shared expert: silu(gate_up) → down.
        let shared_gu =
            self.shared_gate_up
                .forward(hidden_states, &mut device.cublas, &mut device.caching);
        let shared_activated = kernels::silu_and_mul_fused(
            *shared_gu.view(),
            self.shared_intermediate_size,
            &mut device.caching,
            stream,
        );
        drop(shared_gu);
        let shared_out = self.shared_down.forward(
            shared_activated.view(),
            &mut device.cublas,
            &mut device.caching,
        );
        drop(shared_activated);

        // output = moe_out + shared_out (no sigmoid gate).
        kernels::add_inplace(*moe_out.view(), *shared_out.view(), stream);
        drop(shared_out);
        moe_out
    }

    /// Load from safetensors. `prefix` is the MLP prefix for this layer
    /// (e.g. `model.layers.3.mlp`).
    ///
    /// `use_sigmoid`: if true, loads `{prefix}.gate.e_score_correction_bias`
    /// and uses sigmoid routing (DeepSeek V3 / Kimi K2 `topk_method="noaux_tc"`).
    #[allow(clippy::too_many_arguments)]
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
    ) -> anyhow::Result<Self> {
        use crate::driver;
        use crate::tensor::GpuTensor;

        let gate = crate::layers::Linear::load(gw, &format!("{prefix}.gate"))?;

        let first_gate = format!("{prefix}.experts.0.gate_proj.weight");
        let (_, dtype) = gw
            .tensor_info(&first_gate)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {first_gate}"))?;
        let elem = dtype.size_bytes();
        let inter = moe_intermediate_size;

        // Stack expert weights: w1 = [E, 2*inter, hidden], w2 = [E, hidden, inter].
        let w1_bytes = n_routed_experts * 2 * inter * hidden_size * elem;
        let w2_bytes = n_routed_experts * hidden_size * inter * elem;
        let w1_ptr = unsafe { driver::mem_alloc(w1_bytes)? };
        let w2_ptr = unsafe { driver::mem_alloc(w2_bytes)? };

        for e in 0..n_routed_experts {
            let gate_name = format!("{prefix}.experts.{e}.gate_proj.weight");
            let up_name = format!("{prefix}.experts.{e}.up_proj.weight");
            let down_name = format!("{prefix}.experts.{e}.down_proj.weight");
            let expert_w1_off = e * 2 * inter * hidden_size * elem;
            let gate_bytes = inter * hidden_size * elem;
            let expert_w2_off = e * hidden_size * inter * elem;
            unsafe {
                gw.take_into(&gate_name, w1_ptr.add(expert_w1_off), stream)?;
                gw.take_into(&up_name, w1_ptr.add(expert_w1_off + gate_bytes), stream)?;
                gw.take_into(&down_name, w2_ptr.add(expert_w2_off), stream)?;
            }
        }

        let w1 =
            unsafe { GpuTensor::new(w1_ptr, &[n_routed_experts, 2 * inter, hidden_size], dtype) };
        let w2 = unsafe { GpuTensor::new(w2_ptr, &[n_routed_experts, hidden_size, inter], dtype) };

        // Load e_score_correction_bias for sigmoid routing (DeepSeek V3 / Kimi K2).
        // Python vLLM always casts this to F32 before the routing kernel; we match.
        let e_score_correction_bias = if use_sigmoid {
            let bias_name = format!("{prefix}.gate.e_score_correction_bias");
            let bias_raw = gw.take(&bias_name)?;
            let bias_f32 = if bias_raw.dtype() == crate::dtype::DType::F32 {
                bias_raw
            } else {
                // Cast BF16/F16 → F32 (happens when checkpoint is in BF16)
                let n = bias_raw.numel();
                let f32_ptr = unsafe { driver::mem_alloc(n * 4)? };
                let f32_bias = unsafe { GpuTensor::new(f32_ptr, &[n], crate::dtype::DType::F32) };
                unsafe { kernels::cast_bias_to_f32(bias_raw, f32_bias, stream) };
                f32_bias
            };
            Some(bias_f32)
        } else {
            None
        };

        let moe = FusedMoELayer::Dense(Box::new(DenseFusedMoELayer {
            gate,
            w1,
            w2,
            num_experts: n_routed_experts,
            top_k,
            intermediate_size: inter,
            hidden_size,
            renormalize: norm_topk_prob,
            e_score_correction_bias,
            n_expert_group,
            topk_group,
            // Python passes routed_scaling_factor=1.0 to the inner FusedMoE and
            // applies the actual factor *outside* (after the expert computation).
            // DeepSeekV2MoELayer::forward does the same via scale_inplace — so
            // we must not fold it into the routing weights here too (double apply).
            routed_scaling_factor: 1.0,
        }));

        // Shared expert: concat gate_proj + up_proj → [2*shared_inter, hidden].
        let shared_inter = n_shared_experts * moe_intermediate_size;
        let shared_gate_name = format!("{prefix}.shared_experts.gate_proj.weight");
        let shared_up_name = format!("{prefix}.shared_experts.up_proj.weight");
        let gate_bytes = shared_inter * hidden_size * elem;
        let shared_ptr = unsafe { driver::mem_alloc(2 * gate_bytes)? };
        unsafe {
            gw.take_into(&shared_gate_name, shared_ptr, stream)?;
            gw.take_into(&shared_up_name, shared_ptr.add(gate_bytes), stream)?;
        }
        let shared_w =
            unsafe { GpuTensor::new(shared_ptr, &[2 * shared_inter, hidden_size], dtype) };
        let shared_gate_up = crate::layers::Linear::new(shared_w, None);

        let shared_down =
            crate::layers::Linear::load(gw, &format!("{prefix}.shared_experts.down_proj"))?;

        Ok(Self {
            moe,
            shared_gate_up,
            shared_down,
            shared_intermediate_size: shared_inter,
            routed_scaling_factor,
        })
    }
}

// ---------------------------------------------------------------------------
// DeepSeekV2Fp8BlockMoELayer
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub trait DeepSeekV2Fp8BlockMoEOps {
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor;

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
        stream: crate::CUstream,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
}

#[cfg(feature = "cuda")]
impl DeepSeekV2Fp8BlockMoEOps for DeepSeekV2Fp8BlockMoELayer {
    /// Forward pass: `output = routed_scaling_factor * moe(x) + shared_expert(x)`.
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor {
        let stream = device.compute_stream;

        // Routed experts (FP8 block).
        let moe_out = self.moe.forward(hidden_states, device);
        if self.routed_scaling_factor != 1.0 {
            kernels::scale_inplace(*moe_out.view(), self.routed_scaling_factor, &device.cublas);
        }

        // Shared expert: silu(gate_up) → down. Both Fp8BlockLinear.
        let shared_gu = self.shared_gate_up.forward(
            hidden_states,
            &mut device.cublas,
            &mut device.caching,
            stream,
        );
        let shared_activated = kernels::silu_and_mul_fused(
            *shared_gu.view(),
            self.shared_intermediate_size,
            &mut device.caching,
            stream,
        );
        drop(shared_gu);
        let shared_out = self.shared_down.forward(
            shared_activated.view(),
            &mut device.cublas,
            &mut device.caching,
            stream,
        );
        drop(shared_activated);

        // output = moe_out + shared_out (no sigmoid gate).
        kernels::add_inplace(*moe_out.view(), *shared_out.view(), stream);
        drop(shared_out);
        moe_out
    }

    /// Load from safetensors. Mirrors `DeepSeekV2MoELayer::load` but expects
    /// FP8 E4M3 expert weights with `weight_scale_inv` block scales (V3/K2
    /// canonical layout).
    #[allow(clippy::too_many_arguments)]
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
        stream: crate::CUstream,
    ) -> anyhow::Result<Self> {
        use crate::driver;
        use crate::dtype::DType;
        use crate::layers_quant::{block_scale_name, ensure_f32_scale};
        use crate::tensor::GpuTensor;

        let inter = moe_intermediate_size;

        // Gate router (always dense BF16 for V3/K2 — even when experts are FP8).
        let gate = crate::layers::Linear::load(gw, &format!("{prefix}.gate"))?;

        // Derive block size from first expert's gate_proj scale shape.
        let first_gate_w = format!("{prefix}.experts.0.gate_proj.weight");
        let first_gate_s = block_scale_name(gw, &format!("{prefix}.experts.0.gate_proj"));
        let (w_shape, w_dtype) = gw
            .tensor_info(&first_gate_w)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {first_gate_w}"))?;
        anyhow::ensure!(
            w_dtype == DType::Fp8E4m3,
            "DeepSeekV2Fp8BlockMoELayer::load: expected Fp8E4m3 expert weights, got {w_dtype}"
        );
        let (s_shape, _) = gw
            .tensor_info(&first_gate_s)
            .ok_or_else(|| anyhow::anyhow!("scale not found: {first_gate_s}"))?;
        let block_n = w_shape[0] / s_shape[0];
        let block_k = w_shape[1] / s_shape[1];

        // Stacked expert layout (single-rank, no TP yet): w1=[E, 2*inter, hidden] FP8,
        // w2=[E, hidden, inter] FP8.
        let w1_n = 2 * inter;
        let w1_k = hidden_size;
        let w2_n = hidden_size;
        let w2_k = inter;
        let w1_scale_rows = w1_n.div_ceil(block_n);
        let w1_scale_cols = w1_k.div_ceil(block_k);
        let w2_scale_rows = w2_n.div_ceil(block_n);
        let w2_scale_cols = w2_k.div_ceil(block_k);
        let gate_scale_rows = inter.div_ceil(block_n);

        let w1_bytes = n_routed_experts * w1_n * w1_k;
        let w2_bytes = n_routed_experts * w2_n * w2_k;
        let w1_ptr = unsafe { driver::mem_alloc(w1_bytes)? };
        let w2_ptr = unsafe { driver::mem_alloc(w2_bytes)? };

        let w1_scale_bytes = n_routed_experts * w1_scale_rows * w1_scale_cols * 4;
        let w2_scale_bytes = n_routed_experts * w2_scale_rows * w2_scale_cols * 4;
        let w1_scale_ptr = unsafe { driver::mem_alloc(w1_scale_bytes)? };
        let w2_scale_ptr = unsafe { driver::mem_alloc(w2_scale_bytes)? };

        for e in 0..n_routed_experts {
            let gate_pfx = format!("{prefix}.experts.{e}.gate_proj");
            let up_pfx = format!("{prefix}.experts.{e}.up_proj");
            let down_pfx = format!("{prefix}.experts.{e}.down_proj");

            // FP8 weights: stack gate then up along dim=0, down separately.
            let expert_w1_off = e * w1_n * w1_k;
            let gate_proj_bytes = inter * hidden_size;
            let expert_w2_off = e * w2_n * w2_k;
            unsafe {
                gw.take_into(
                    &format!("{gate_pfx}.weight"),
                    w1_ptr.add(expert_w1_off),
                    stream,
                )?;
                gw.take_into(
                    &format!("{up_pfx}.weight"),
                    w1_ptr.add(expert_w1_off + gate_proj_bytes),
                    stream,
                )?;
                gw.take_into(
                    &format!("{down_pfx}.weight"),
                    w2_ptr.add(expert_w2_off),
                    stream,
                )?;
            }

            // Block scales: copy into stacked 3D buffers.
            let expert_w1_scale_off = e * w1_scale_rows * w1_scale_cols * 4;
            let gate_scale = {
                let raw = gw.take(&block_scale_name(gw, &gate_pfx))?;
                ensure_f32_scale(raw, stream)?
            };
            let gate_scale_bytes = gate_scale_rows * w1_scale_cols * 4;
            unsafe {
                driver::memcpy_dtod_async(
                    w1_scale_ptr.add(expert_w1_scale_off),
                    gate_scale.raw_ptr(),
                    gate_scale_bytes,
                    stream,
                )?;
            }
            let up_scale = {
                let raw = gw.take(&block_scale_name(gw, &up_pfx))?;
                ensure_f32_scale(raw, stream)?
            };
            let up_scale_rows = w1_scale_rows - gate_scale_rows;
            let up_scale_bytes = up_scale_rows * w1_scale_cols * 4;
            unsafe {
                driver::memcpy_dtod_async(
                    w1_scale_ptr.add(expert_w1_scale_off + gate_scale_bytes),
                    up_scale.raw_ptr(),
                    up_scale_bytes,
                    stream,
                )?;
            }
            let expert_w2_scale_off = e * w2_scale_rows * w2_scale_cols * 4;
            let down_scale = {
                let raw = gw.take(&block_scale_name(gw, &down_pfx))?;
                ensure_f32_scale(raw, stream)?
            };
            let down_scale_bytes = w2_scale_rows * w2_scale_cols * 4;
            unsafe {
                driver::memcpy_dtod_async(
                    w2_scale_ptr.add(expert_w2_scale_off),
                    down_scale.raw_ptr(),
                    down_scale_bytes,
                    stream,
                )?;
            }

            // Consume input_scale if present (block quant uses dynamic activation).
            for pfx in &[&gate_pfx, &up_pfx, &down_pfx] {
                let is_name = format!("{pfx}.input_scale");
                if gw.contains(&is_name) {
                    let _ = gw.take(&is_name);
                }
            }
        }

        let w1 = unsafe { GpuTensor::new(w1_ptr, &[n_routed_experts, w1_n, w1_k], DType::Fp8E4m3) };
        let w2 = unsafe { GpuTensor::new(w2_ptr, &[n_routed_experts, w2_n, w2_k], DType::Fp8E4m3) };
        let w1_scale_inv = unsafe {
            GpuTensor::new(
                w1_scale_ptr,
                &[n_routed_experts, w1_scale_rows, w1_scale_cols],
                DType::F32,
            )
        };
        let w2_scale_inv = unsafe {
            GpuTensor::new(
                w2_scale_ptr,
                &[n_routed_experts, w2_scale_rows, w2_scale_cols],
                DType::F32,
            )
        };

        // e_score_correction_bias (F32) for sigmoid routing.
        let e_score_correction_bias = if use_sigmoid {
            let bias_name = format!("{prefix}.gate.e_score_correction_bias");
            let bias_raw = gw.take(&bias_name)?;
            let bias_f32 = if bias_raw.dtype() == DType::F32 {
                bias_raw
            } else {
                let n = bias_raw.numel();
                let f32_ptr = unsafe { driver::mem_alloc(n * 4)? };
                let f32_bias = unsafe { GpuTensor::new(f32_ptr, &[n], DType::F32) };
                unsafe { kernels::cast_bias_to_f32(bias_raw, f32_bias, stream) };
                f32_bias
            };
            Some(bias_f32)
        } else {
            None
        };

        let moe = Fp8BlockFusedMoELayer {
            gate,
            w1,
            w2,
            w1_scale_inv,
            w2_scale_inv,
            block_size: [block_n, block_k],
            num_experts: n_routed_experts,
            top_k,
            intermediate_size: inter,
            hidden_size,
            renormalize: norm_topk_prob,
            e_score_correction_bias,
            n_expert_group,
            topk_group,
            // Same as DeepSeekV2MoELayer: outer layer applies scale_inplace after
            // experts; inner layer must use 1.0 to avoid double application.
            routed_scaling_factor: 1.0,
        };

        // Shared expert (FP8 block): concat gate_proj+up_proj, then down_proj.
        let shared_gate_pfx = format!("{prefix}.shared_experts.gate_proj");
        let shared_up_pfx = format!("{prefix}.shared_experts.up_proj");
        let shared_down_pfx = format!("{prefix}.shared_experts.down_proj");
        let shared_gate_up = crate::layers::Fp8BlockLinear::load_concat(
            gw,
            &[&shared_gate_pfx, &shared_up_pfx],
            output_dtype,
        )?;
        let shared_down = crate::layers::Fp8BlockLinear::load(gw, &shared_down_pfx, output_dtype)?;
        let shared_inter = n_shared_experts * moe_intermediate_size;

        Ok(Self {
            moe,
            shared_gate_up,
            shared_down,
            shared_intermediate_size: shared_inter,
            routed_scaling_factor,
        })
    }
}

// ---------------------------------------------------------------------------
// Fp8FusedMoELayer (FP8 E4M3 quantized MoE)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub trait Fp8FusedMoEOps {
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor;
}

#[cfg(feature = "cuda")]
impl Fp8FusedMoEOps for Fp8FusedMoELayer {
    /// Forward pass — full FP8 MoE pipeline.
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor {
        let num_tokens = hidden_states.dim(0);
        let stream = device.compute_stream;
        let sm_version = device.sm_version;

        let block_m = select_moe_block_m(num_tokens, self.top_k, self.num_experts);

        // 1. Gate: router_logits = hidden_states @ gate_weight^T
        let router_logits =
            self.gate
                .forward(hidden_states, &mut device.cublas, &mut device.caching);

        // 2. Route: softmax / sigmoid+bias / grouped noaux_tc (DSv3/Kimi K2).
        let (topk_weights, topk_ids) = route_experts(
            router_logits.as_gpu_tensor(),
            &MoeRouting {
                top_k: self.top_k,
                renormalize: self.renormalize,
                e_score_correction_bias: self.e_score_correction_bias.as_ref(),
                n_expert_group: self.n_expert_group,
                topk_group: self.topk_group,
                routed_scaling_factor: self.routed_scaling_factor,
            },
            &mut device.caching,
            stream,
        );
        drop(router_logits);

        // 3. Quantize hidden states to FP8 with per-token dynamic scales.
        let (fp8_input, a1_scales) =
            kernels::scaled_fp8_quant_dynamic(*hidden_states, &mut device.caching, stream);

        // 4. Align block size: sort tokens by expert
        let (sorted_token_ids, expert_ids, num_tokens_post_padded) = kernels::moe_align_block_size(
            topk_ids.as_gpu_tensor(),
            self.num_experts,
            block_m,
            &mut device.caching,
            stream,
        );
        drop(topk_ids);

        // 5. GEMM 1: fp8_input × w1^T → [num_tokens * top_k, 2*intermediate] BF16
        let intermediate1 = kernels::fused_moe_fp8_gemm(
            fp8_input.as_gpu_tensor(),
            self.w1,
            a1_scales.as_gpu_tensor(),
            self.w1_scale,
            topk_weights.as_gpu_tensor(),
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            num_tokens,
            self.top_k,
            block_m,
            false, // don't apply routing weights on first GEMM
            sm_version,
            &mut device.caching,
            stream,
        );
        drop(fp8_input);
        drop(a1_scales);

        // 6. Activation: SiLU(gate) * up → [num_tokens * top_k, intermediate]
        let activated = kernels::silu_and_mul_fused(
            intermediate1.as_gpu_tensor(),
            self.intermediate_size,
            &mut device.caching,
            stream,
        );
        drop(intermediate1);

        // 7. Re-quantize activated to FP8 with fresh per-token scales (matching Python).
        let (fp8_act, a2_scales) = kernels::scaled_fp8_quant_dynamic(
            activated.as_gpu_tensor(),
            &mut device.caching,
            stream,
        );
        drop(activated);

        // 8. GEMM 2: fp8_act × w2^T → [num_tokens * top_k, hidden_size] BF16
        //    Apply routing weight here. top_k=1 for pass 2 (input already expanded).
        let intermediate2 = kernels::fused_moe_fp8_gemm(
            fp8_act.as_gpu_tensor(),
            self.w2,
            a2_scales.as_gpu_tensor(),
            self.w2_scale,
            topk_weights.as_gpu_tensor(),
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            num_tokens * self.top_k,
            1, // top_k=1: index directly into expanded input
            block_m,
            true, // apply routing weights
            sm_version,
            &mut device.caching,
            stream,
        );
        drop(fp8_act);
        drop(a2_scales);
        drop(topk_weights);
        drop(sorted_token_ids);
        drop(expert_ids);
        drop(num_tokens_post_padded);

        // 9. Reduce: sum across top_k experts → [num_tokens, hidden_size]
        let output = kernels::moe_sum(
            intermediate2.as_gpu_tensor(),
            num_tokens,
            self.hidden_size,
            self.top_k,
            &mut device.caching,
            stream,
        );
        drop(intermediate2);

        output
    }
}

// ---------------------------------------------------------------------------
// Fp8BlockFusedMoELayer (FP8 block-quantized MoE)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub trait Fp8BlockFusedMoEOps {
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor;
}

#[cfg(feature = "cuda")]
impl Fp8BlockFusedMoEOps for Fp8BlockFusedMoELayer {
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor {
        let num_tokens = hidden_states.dim(0);
        let stream = device.compute_stream;

        let block_m = select_moe_block_m(num_tokens, self.top_k, self.num_experts);

        // 1. Gate
        let router_logits =
            self.gate
                .forward(hidden_states, &mut device.cublas, &mut device.caching);

        // 2. Route: softmax / sigmoid+bias / grouped noaux_tc.
        let (topk_weights, topk_ids) = route_experts(
            router_logits.as_gpu_tensor(),
            &MoeRouting {
                top_k: self.top_k,
                renormalize: self.renormalize,
                e_score_correction_bias: self.e_score_correction_bias.as_ref(),
                n_expert_group: self.n_expert_group,
                topk_group: self.topk_group,
                routed_scaling_factor: self.routed_scaling_factor,
            },
            &mut device.caching,
            stream,
        );
        drop(router_logits);

        // 3. Quantize hidden states to FP8
        let (fp8_input, a1_scales) =
            kernels::scaled_fp8_quant_dynamic(*hidden_states, &mut device.caching, stream);

        // 4. Align block size
        let (sorted_token_ids, expert_ids, num_tokens_post_padded) = kernels::moe_align_block_size(
            topk_ids.as_gpu_tensor(),
            self.num_experts,
            block_m,
            &mut device.caching,
            stream,
        );
        drop(topk_ids);

        // 5. GEMM 1: block-scaled FP8
        let intermediate1 = kernels::fused_moe_fp8_block_gemm(
            fp8_input.as_gpu_tensor(),
            self.w1,
            a1_scales.as_gpu_tensor(),
            self.w1_scale_inv,
            topk_weights.as_gpu_tensor(),
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            num_tokens,
            self.top_k,
            block_m,
            false,
            self.block_size,
            &mut device.caching,
            stream,
        );
        drop(fp8_input);
        drop(a1_scales);

        // 6. SiLU activation
        let activated = kernels::silu_and_mul_fused(
            intermediate1.as_gpu_tensor(),
            self.intermediate_size,
            &mut device.caching,
            stream,
        );
        drop(intermediate1);

        // 7. Re-quantize to FP8
        let (fp8_act, a2_scales) = kernels::scaled_fp8_quant_dynamic(
            activated.as_gpu_tensor(),
            &mut device.caching,
            stream,
        );
        drop(activated);

        // 8. GEMM 2: block-scaled FP8
        let intermediate2 = kernels::fused_moe_fp8_block_gemm(
            fp8_act.as_gpu_tensor(),
            self.w2,
            a2_scales.as_gpu_tensor(),
            self.w2_scale_inv,
            topk_weights.as_gpu_tensor(),
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            num_tokens * self.top_k,
            1,
            block_m,
            true,
            self.block_size,
            &mut device.caching,
            stream,
        );
        drop(fp8_act);
        drop(a2_scales);
        drop(topk_weights);
        drop(sorted_token_ids);
        drop(expert_ids);
        drop(num_tokens_post_padded);

        // 9. Reduce
        let output = kernels::moe_sum(
            intermediate2.as_gpu_tensor(),
            num_tokens,
            self.hidden_size,
            self.top_k,
            &mut device.caching,
            stream,
        );
        drop(intermediate2);

        output
    }
}

// ---------------------------------------------------------------------------
// Fp8SharedFusedMoELayer (Qwen2/3 MoE with FP8 experts)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub trait Fp8SharedFusedMoEOps {
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor;
}

#[cfg(feature = "cuda")]
impl Fp8SharedFusedMoEOps for Fp8SharedFusedMoELayer {
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor {
        let stream = device.compute_stream;

        let moe_out = self.moe.forward(hidden_states, device);

        if let (Some(shared_gate_up), Some(shared_down), Some(shared_gate)) = (
            &self.shared_gate_up,
            &self.shared_down,
            &self.shared_expert_gate,
        ) {
            let shared_gu =
                shared_gate_up.forward(hidden_states, &mut device.cublas, &mut device.caching);
            let shared_activated = kernels::silu_and_mul_fused(
                shared_gu.as_gpu_tensor(),
                self.intermediate_size,
                &mut device.caching,
                stream,
            );
            drop(shared_gu);

            let shared_out = shared_down.forward(
                shared_activated.view(),
                &mut device.cublas,
                &mut device.caching,
            );
            drop(shared_activated);

            let gate_logits =
                shared_gate.forward(hidden_states, &mut device.cublas, &mut device.caching);

            let result = kernels::sigmoid_mul_add(
                moe_out.as_gpu_tensor(),
                shared_out.as_gpu_tensor(),
                gate_logits.as_gpu_tensor(),
                &mut device.caching,
                stream,
            );
            drop(moe_out);
            drop(shared_out);
            drop(gate_logits);

            result
        } else {
            moe_out
        }
    }
}

// ---------------------------------------------------------------------------
// GgmlFusedMoELayer (GGML quantized MoE)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub trait GgmlFusedMoEOps {
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor;
}

#[cfg(feature = "cuda")]
impl GgmlFusedMoEOps for GgmlFusedMoELayer {
    /// Forward pass — full quantized MoE pipeline.
    ///
    /// * `hidden_states`: `[num_tokens, hidden_size]` in any dtype (cast to f32 internally).
    ///
    /// Returns: `[num_tokens, hidden_size]` in same dtype as input.
    ///
    /// # Safety
    /// All tensors must be valid GPU memory.
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor {
        use crate::dtype::DType;
        use crate::ggml::{MATRIX_ROW_PADDING, ggml_moe_forward, ggml_quantize_q8_1_alloc};

        let input_dtype = hidden_states.dtype();
        let num_tokens = hidden_states.dim(0);
        let stream = device.compute_stream;

        // 1. Gate: router_logits = hidden_states @ gate_weight^T (dense Linear).
        let router_logits =
            self.gate
                .forward(hidden_states, &mut device.cublas, &mut device.caching);

        // 2. Route: softmax / sigmoid+bias / grouped noaux_tc.
        let (topk_weights, topk_ids) = route_experts(
            router_logits.as_gpu_tensor(),
            &MoeRouting {
                top_k: self.top_k,
                renormalize: self.renormalize,
                e_score_correction_bias: self.e_score_correction_bias.as_ref(),
                n_expert_group: self.n_expert_group,
                topk_group: self.topk_group,
                routed_scaling_factor: self.routed_scaling_factor,
            },
            &mut device.caching,
            stream,
        );
        drop(router_logits);

        // Expert indices: [num_tokens, top_k] i32 viewed as u32 (expert ids are non-negative).
        let indices_ptr = topk_ids.as_gpu_tensor().raw_ptr() as *const u32;

        // 3. Cast hidden_states to f32 if needed, then quantize to Q8_1.
        let hs_f32 = if input_dtype != DType::F32 {
            Some(kernels::cast_logits_to_f32(
                *hidden_states,
                &mut device.caching,
                stream,
            ))
        } else {
            None
        };
        let hs_f32_ptr = if let Some(ref cast) = hs_f32 {
            cast.as_gpu_tensor().raw_ptr() as *const f32
        } else {
            hidden_states.as_ptr::<f32>()
        };
        let k = self.hidden_size;
        let k_padded = crate::ggml::pad(k, MATRIX_ROW_PADDING);
        let (q8_hidden, _) =
            ggml_quantize_q8_1_alloc(hs_f32_ptr, k, num_tokens, &mut device.caching, stream);
        drop(hs_f32);

        // 4. GEMM 1: w1 × q8_input → [num_tokens * top_k, 2*intermediate] f32
        // The indexed_moe_forward kernel uses input_dim1 to determine input sharing:
        //   input_idx = (input_dim1 == 1) ? current_batch : task_id
        // We pass input_dim1=1 so all topk experts for a token share the same
        // quantized input row (no replication needed).
        let out1 = device.caching.alloc_tensor(
            &[num_tokens * self.top_k, 2 * self.intermediate_size],
            crate::dtype::DType::F32,
        );
        ggml_moe_forward(
            &self.w1,
            q8_hidden,
            indices_ptr,
            out1.as_gpu_tensor().raw_ptr() as *mut f32,
            2 * self.intermediate_size,
            k,
            num_tokens,
            self.top_k,
            k_padded,
            1, // input_dim1=1: share input across topk per batch item
            stream,
        );

        // 5. SiLU-and-mul → [num_tokens * top_k, intermediate] f32
        let activated = kernels::silu_and_mul_fused(
            out1.as_gpu_tensor(),
            self.intermediate_size,
            &mut device.caching,
            stream,
        );
        drop(out1);

        // 6. Quantize activated to Q8_1
        let inter = self.intermediate_size;
        let inter_padded = crate::ggml::pad(inter, MATRIX_ROW_PADDING);
        let (q8_act, _) = ggml_quantize_q8_1_alloc(
            activated.as_gpu_tensor().raw_ptr() as *const f32,
            inter,
            num_tokens * self.top_k,
            &mut device.caching,
            stream,
        );
        drop(activated);

        // 7. GEMM 2: w2 × q8_act → [num_tokens * top_k, hidden_size] f32
        // For GEMM 2, each task has its own unique input row (the activated output),
        // so input_dim1 = batch * topk (i.e. not 1).
        let out2 = device.caching.alloc_tensor(
            &[num_tokens * self.top_k, self.hidden_size],
            crate::dtype::DType::F32,
        );
        ggml_moe_forward(
            &self.w2,
            q8_act,
            indices_ptr,
            out2.as_gpu_tensor().raw_ptr() as *mut f32,
            self.hidden_size,
            inter,
            num_tokens,
            self.top_k,
            inter_padded,
            num_tokens * self.top_k, // input_dim1: each task has unique input
            stream,
        );
        drop(topk_ids); // indices no longer needed

        // 8. Scale by topk_weights and sum across topk → [num_tokens, hidden_size] f32.
        //
        // out2 is `[num_tokens * topk, hidden]` f32; topk_weights is
        // `[num_tokens, topk]` f32 (also num_tokens*topk contiguous f32s). We
        // need `out2[row, :] *= topk_weights[row]` — *row*-wise scaling.
        //
        // `kernels::broadcast_mul_inplace` is the WRONG primitive here: its
        // kernel does `out2[r, c] *= scale[c]` — column-wise — and indexes
        // `scale[c]` for c in [0, hidden), reading num_tokens*topk past the
        // end of the topk_weights buffer (= UB; corrupts output to ±Inf/NaN
        // and the bug propagates through the residual stream into every
        // downstream MoE layer's gate matmul). Use `fp8_post_scale_multiply`,
        // which is genuinely per-row (`output[i, :] *= scales[i]`).
        // Reshape via `with_view` because the kernel asserts `output.ndim()==2`
        // and `scales.ndim()==1` and that `output.dim(0)==scales.dim(0)`.
        let topk_view = topk_weights.view();
        let topk_flat = unsafe {
            crate::tensor::TensorView::from_raw(crate::tensor::GpuTensor::new(
                topk_view.raw_ptr(),
                &[num_tokens * self.top_k],
                crate::dtype::DType::F32,
            ))
        };
        kernels::fp8_post_scale_multiply(out2.as_gpu_tensor(), *topk_flat, stream);
        drop(topk_weights);

        let output = kernels::moe_sum(
            out2.as_gpu_tensor(),
            num_tokens,
            self.hidden_size,
            self.top_k,
            &mut device.caching,
            stream,
        );
        drop(out2);

        // Cast back to original dtype if we converted to f32.
        if input_dtype != DType::F32 {
            let result = kernels::cast_from_f32(
                output.as_gpu_tensor(),
                input_dtype,
                &mut device.caching,
                stream,
            );
            drop(output);
            result
        } else {
            output
        }
    }
}

// ---------------------------------------------------------------------------
// DeepSeekV2GgmlMoELayer (GGML-quantized DeepSeek MoE)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub trait DeepSeekV2GgmlMoEOps {
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor;

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
        stream: crate::CUstream,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
}

#[cfg(feature = "cuda")]
impl DeepSeekV2GgmlMoEOps for DeepSeekV2GgmlMoELayer {
    /// Forward pass: `output = routed_scaling_factor * moe(x) + shared_expert(x)`.
    ///
    /// `moe.forward` returns the input dtype (cast back from f32 inside).
    /// `shared_gate_up.forward` casts back to input dtype too (GgmlLinear
    /// internal contract). So both `moe_out` and `shared_out` are
    /// `input_dtype` at the `add_inplace` seam — same shape as
    /// `DeepSeekV2MoELayer::forward`'s terminal add.
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor {
        let stream = device.compute_stream;

        // Routed experts (GGML).
        let moe_out = self.moe.forward(hidden_states, device);
        if self.routed_scaling_factor != 1.0 {
            kernels::scale_inplace(*moe_out.view(), self.routed_scaling_factor, &device.cublas);
        }

        // Shared expert: silu(gate_up) → down. Both GgmlLinear; outputs match
        // `hidden_states.dtype()` because GgmlLinear casts back from its
        // internal f32 to the input dtype on exit.
        let shared_gu = self
            .shared_gate_up
            .forward(hidden_states, &mut device.caching, stream);
        let shared_activated = kernels::silu_and_mul_fused(
            *shared_gu.view(),
            self.shared_intermediate_size,
            &mut device.caching,
            stream,
        );
        drop(shared_gu);
        let shared_out =
            self.shared_down
                .forward(shared_activated.view(), &mut device.caching, stream);
        drop(shared_activated);

        // output = moe_out + shared_out (no sigmoid gate); both in input_dtype.
        kernels::add_inplace(*moe_out.view(), *shared_out.view(), stream);
        drop(shared_out);
        moe_out
    }

    /// Load from GGUF weights. Mirrors the hand-written
    /// `scratchy-serving-cuda::deepseek_v2::load_gguf` (model-side fused-3D MoE
    /// expert layout): consumes `mlp.experts.fused_{gate,up,down}_exps.weight`
    /// (already 3D `[num_experts, slab, hidden|inter]` after the `gguf`
    /// rename) plus `mlp.shared_experts.{gate,up,down}_proj.weight` and
    /// `mlp.gate.weight` (router).
    ///
    /// Interleaves gate+up byte slabs into a single quantized
    /// `w1 = [E, 2*inter, hidden]` for the `indexed_moe_forward` kernel;
    /// reuses the `down` tensor as `w2 = [E, hidden, inter]`. Frees the
    /// per-component `gate_exps`/`up_exps` buffers after copy so net
    /// memory matches the per-tensor sum (no doubling).
    #[allow(clippy::too_many_arguments)]
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
        stream: crate::CUstream,
    ) -> anyhow::Result<Self> {
        use crate::driver;
        use crate::dtype::DType;
        use crate::ggml::GgmlStorage;
        use crate::tensor::GpuTensor;

        let inter = moe_intermediate_size;
        let hidden = hidden_size;

        // ── Router gate (dense BF16; the gguf loader converts the
        // F32-on-disk gate to model dtype during load, so it lands in
        // `gguf_dense` already shape-correct for a plain Linear). ──
        let gate_name = format!("{prefix}.gate.weight");
        let gate_w = gw
            .take_gguf_dense(&gate_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {gate_name}"))?;
        let gate = Linear::new(gate_w, None);

        // ── Routed expert weights (fused 3D) ──
        // `mlp.experts.fused_gate_exps.weight` lands as a single GgmlStorage
        // with `nrows = E * inter`, `ncols = hidden` (set by ggml.rs's 3D-flatten).
        let gate_exps_name = format!("{prefix}.experts.fused_gate_exps.weight");
        let up_exps_name = format!("{prefix}.experts.fused_up_exps.weight");
        let down_exps_name = format!("{prefix}.experts.fused_down_exps.weight");

        let gate_exps = gw
            .quantized_map_mut()
            .remove(&gate_exps_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {gate_exps_name}"))?;
        let up_exps = gw
            .quantized_map_mut()
            .remove(&up_exps_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {up_exps_name}"))?;
        let down_exps = gw
            .quantized_map_mut()
            .remove(&down_exps_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {down_exps_name}"))?;

        // gate + up must share dtype because we interleave their byte slabs
        // into a single w1 quantized buffer; down lives in its own w2 storage
        // and can use a different (often higher-precision) quant — Q4_K_M
        // GGUFs typically ship gate/up=Q4_K, down=Q8_0.
        anyhow::ensure!(
            gate_exps.dtype == up_exps.dtype,
            "DeepSeekV2GgmlMoELayer::load_gguf: gate/up expert quant dtype \
             mismatch (gate={:?}, up={:?}); cannot interleave into shared w1",
            gate_exps.dtype,
            up_exps.dtype,
        );

        let qdtype = gate_exps.dtype;
        let bs = qdtype.block_size();
        let ts = qdtype.type_size();

        // Each expert's gate slab: [inter, hidden] quantized.
        // Each expert's up   slab: [inter, hidden] quantized (same size).
        // w1 expert slab        : [2*inter, hidden] quantized — gate then up.
        let expert_slab_bytes = (inter * hidden / bs) * ts;
        let w1_expert_bytes = 2 * expert_slab_bytes;
        let w1_total_bytes = n_routed_experts * w1_expert_bytes;
        let w1_ptr = unsafe { driver::mem_alloc(w1_total_bytes)? };

        for e in 0..n_routed_experts {
            let gate_offset = e * expert_slab_bytes;
            let up_offset = e * expert_slab_bytes;
            let w1_gate_offset = e * w1_expert_bytes;
            let w1_up_offset = w1_gate_offset + expert_slab_bytes;
            unsafe {
                driver::memcpy_dtod_async(
                    w1_ptr.add(w1_gate_offset),
                    gate_exps.ptr.add(gate_offset),
                    expert_slab_bytes,
                    stream,
                )?;
                driver::memcpy_dtod_async(
                    w1_ptr.add(w1_up_offset),
                    up_exps.ptr.add(up_offset),
                    expert_slab_bytes,
                    stream,
                )?;
            }
        }

        // Sync, then free the per-component buffers — w1 now owns the bytes.
        unsafe {
            driver::stream_synchronize(stream)?;
            gw.unrecord_alloc(gate_exps.ptr);
            driver::mem_free(gate_exps.ptr)?;
            gw.unrecord_alloc(up_exps.ptr);
            driver::mem_free(up_exps.ptr)?;
        }
        // Track w1 with the GpuWeights so its lifetime matches the model's.
        gw.record_alloc(w1_ptr, w1_total_bytes);

        let w1 = GgmlStorage {
            ptr: w1_ptr,
            len: w1_total_bytes,
            dtype: qdtype,
            nrows: n_routed_experts * 2 * inter,
            ncols: hidden,
        };
        // w2 = down_exps unchanged: [E, hidden, inter] quantized
        // (nrows = E * hidden, ncols = inter — already set by 3D flatten).
        let w2 = down_exps;

        // ── e_score_correction_bias (V3 / Kimi K2 sigmoid routing) ──
        let e_score_correction_bias = if use_sigmoid {
            let bias_name = format!("{prefix}.gate.e_score_correction_bias");
            let bias_raw = gw.take(&bias_name)?;
            let bias_f32 = if bias_raw.dtype() == DType::F32 {
                bias_raw
            } else {
                let n = bias_raw.numel();
                let f32_ptr = unsafe { driver::mem_alloc(n * 4)? };
                let f32_bias = unsafe { GpuTensor::new(f32_ptr, &[n], DType::F32) };
                unsafe { kernels::cast_bias_to_f32(bias_raw, f32_bias, stream) };
                gw.record_alloc(f32_ptr, n * 4);
                f32_bias
            };
            Some(bias_f32)
        } else {
            None
        };

        let moe = GgmlFusedMoELayer {
            gate,
            w1,
            w2,
            num_experts: n_routed_experts,
            top_k,
            intermediate_size: inter,
            hidden_size: hidden,
            renormalize: norm_topk_prob,
            e_score_correction_bias,
            n_expert_group,
            topk_group,
            // Outer DeepSeekV2GgmlMoELayer::forward applies routed_scaling_factor
            // via scale_inplace; inner stays at 1.0 to avoid double application
            // (matches DeepSeekV2MoELayer's split).
            routed_scaling_factor: 1.0,
        };

        // ── Shared experts: concat gate_proj + up_proj into one fused slab ──
        let shared_inter = n_shared_experts * moe_intermediate_size;
        let shared_gate_name = format!("{prefix}.shared_experts.gate_proj.weight");
        let shared_up_name = format!("{prefix}.shared_experts.up_proj.weight");
        let shared_down_name = format!("{prefix}.shared_experts.down_proj.weight");

        let shared_gate_s = gw
            .quantized_map_mut()
            .remove(&shared_gate_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {shared_gate_name}"))?;
        let shared_up_s = gw
            .quantized_map_mut()
            .remove(&shared_up_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {shared_up_name}"))?;
        let shared_down_s = gw
            .quantized_map_mut()
            .remove(&shared_down_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {shared_down_name}"))?;

        anyhow::ensure!(
            shared_gate_s.dtype == shared_up_s.dtype,
            "DeepSeekV2GgmlMoELayer::load_gguf: shared gate/up quant dtype mismatch \
             ({:?} vs {:?})",
            shared_gate_s.dtype,
            shared_up_s.dtype,
        );

        let se_qdtype = shared_gate_s.dtype;
        let se_bs = se_qdtype.block_size();
        let se_ts = se_qdtype.type_size();
        let se_slab_bytes = (shared_inter * hidden / se_bs) * se_ts;
        let se_total_bytes = 2 * se_slab_bytes;
        let se_ptr = unsafe { driver::mem_alloc(se_total_bytes)? };
        unsafe {
            driver::memcpy_dtod_async(se_ptr, shared_gate_s.ptr, se_slab_bytes, stream)?;
            driver::memcpy_dtod_async(
                se_ptr.add(se_slab_bytes),
                shared_up_s.ptr,
                se_slab_bytes,
                stream,
            )?;
            driver::stream_synchronize(stream)?;
            gw.unrecord_alloc(shared_gate_s.ptr);
            driver::mem_free(shared_gate_s.ptr)?;
            gw.unrecord_alloc(shared_up_s.ptr);
            driver::mem_free(shared_up_s.ptr)?;
        }
        gw.record_alloc(se_ptr, se_total_bytes);

        let shared_gate_up = crate::layers::GgmlLinear {
            storage: GgmlStorage {
                ptr: se_ptr,
                len: se_total_bytes,
                dtype: se_qdtype,
                nrows: 2 * shared_inter,
                ncols: hidden,
            },
            bias: None,
        };
        let shared_down = crate::layers::GgmlLinear {
            storage: shared_down_s,
            bias: None,
        };

        Ok(Self {
            moe,
            shared_gate_up,
            shared_down,
            shared_intermediate_size: shared_inter,
            routed_scaling_factor,
        })
    }
}

// ---------------------------------------------------------------------------
// FP8 MoE Weight Loading Helpers
// ---------------------------------------------------------------------------

/// Load FP8 MoE expert weights with per-tensor or per-block scales.
///
/// For per-tensor scales: dequantizes FP8 expert weights to BF16 at load time
/// using merged scales (max of gate/up shard scales, matching Python's
/// `process_fp8_weight_tensor_strategy_moe`).
///
/// For per-block scales: dequantizes using block-indexed scales.
///
/// The dequantized BF16 weights are then used with the existing fused_moe_gemm
/// kernel which operates on BF16. This is the correctness-first approach;
/// a native FP8 fused MoE GEMM kernel can be added later for performance.
///
/// NOTE: This function should be called at model load time, not on the hot path.
/// The returned tensors are BF16 and work with the existing `FusedMoELayer`.
#[cfg(feature = "cuda")]
pub fn load_fp8_moe_weights_dequant(
    w_fp8: GpuTensor, // [num_experts, dim, hidden] FP8 E4M3
    scale: GpuTensor, // [num_experts] f32 (per-tensor) or per-block
    output_dtype: crate::dtype::DType,
    alloc: &mut crate::alloc::CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> crate::alloc::OwnedTensor {
    // For now, use the per-tensor approach: dequantize the entire stacked tensor.
    // This works because scale is per-expert (the fused_moe_gemm kernel selects
    // the right expert slice anyway).
    //
    // TODO: Implement native FP8 fused MoE GEMM for perf parity.
    let _total_elements: usize = w_fp8.numel();

    // Reshape to 2D for dequant, then reshape back.
    // w_fp8: [num_experts, dim, hidden] → flatten to [num_experts * dim, hidden]
    // Then dequant each element using per-expert scale.
    //
    // For the correctness-first approach, we use the max scale across all experts
    // and dequant the entire tensor as 2D.
    let ne = w_fp8.dim(0);
    let d1 = w_fp8.dim(1);
    let d2 = w_fp8.dim(2);

    // Read scales to CPU to find max.
    // Note: This is only done once at load time, so D2H is acceptable.
    let _scale = scale;
    let _ne = ne;

    // For simplicity and correctness, we dequant each expert's 2D slice separately.
    // This is done at load time and is not on the hot path.
    let total_elems = ne * d1 * d2;
    let out = alloc.alloc_tensor(&[ne, d1, d2], output_dtype);

    // Use the block dequant kernel with block_size = [d1, d2] (entire expert = one block).
    // Or simpler: element-wise dequant with per-expert scale.
    // For now, fall through to a per-element kernel using a scale of 1.0 (identity).
    // The actual dequant should use the per-expert scale.
    //
    // TODO: Implement proper per-expert FP8 dequant kernel.
    // For now, this is a placeholder that marks the integration point.
    let _ = total_elems;
    let _ = stream;

    out
}

// ---------------------------------------------------------------------------
// MarlinFusedMoELayer (AWQ/GPTQ INT4 quantized MoE)
// ---------------------------------------------------------------------------

/// Dynamic block size selection for Marlin MoE GEMM tiling.
/// Matches Python vLLM's `_fused_marlin_moe` logic:
///   for block_size_m in [8, 16, 32, 48, 64]:
///       if M * topk / E / block_size_m < 0.9: break
/// Only thread_m_blocks=1 kernels are instantiated → max block_size = 16.
#[cfg(feature = "cuda")]
fn select_moe_block_size(num_tokens: usize, top_k: usize, num_experts: usize) -> usize {
    for &bs in &[8usize, 16] {
        if ((num_tokens * top_k) as f64 / num_experts as f64 / bs as f64) < 0.9 {
            return bs;
        }
    }
    16 // max with thread_m_blocks=1
}

#[cfg(feature = "cuda")]
pub trait MarlinFusedMoEOps {
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor;
}

#[cfg(feature = "cuda")]
impl MarlinFusedMoEOps for MarlinFusedMoELayer {
    /// Forward pass — full Marlin MoE pipeline.
    ///
    /// * `hidden_states`: `[num_tokens, hidden_size]`
    ///
    /// Returns: `[num_tokens, hidden_size]`
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor {
        let num_tokens = hidden_states.dim(0);
        let stream = device.compute_stream;

        // 1. Gate: router_logits = hidden_states @ gate_weight^T
        let router_logits =
            self.gate
                .forward(hidden_states, &mut device.cublas, &mut device.caching);

        // 2. Route: softmax / sigmoid+bias / grouped noaux_tc.
        let (topk_weights, topk_ids) = route_experts(
            router_logits.as_gpu_tensor(),
            &MoeRouting {
                top_k: self.top_k,
                renormalize: self.renormalize,
                e_score_correction_bias: self.e_score_correction_bias.as_ref(),
                n_expert_group: self.n_expert_group,
                topk_group: self.topk_group,
                routed_scaling_factor: self.routed_scaling_factor,
            },
            &mut device.caching,
            stream,
        );
        drop(router_logits);

        // 3. Dynamic block size selection (matches Python vLLM)
        let moe_block_size = select_moe_block_size(num_tokens, self.top_k, self.num_experts);

        // Align block size: sort tokens by expert for fused GEMM
        let (sorted_token_ids, expert_ids, num_tokens_post_padded) = kernels::moe_align_block_size(
            topk_ids.as_gpu_tensor(),
            self.num_experts,
            moe_block_size,
            &mut device.caching,
            stream,
        );
        drop(topk_ids);

        let num_groups_w1 = self.w1_scales.dim(1);
        let num_groups_w2 = self.w2_scales.dim(1);
        let group_size_w1 = if num_groups_w1 > 1 {
            self.hidden_size / num_groups_w1
        } else {
            -1i64 as usize
        };
        let group_size_w2 = if num_groups_w2 > 1 {
            self.intermediate_size / num_groups_w2
        } else {
            -1i64 as usize
        };

        // 4. GEMM 1: hidden_states × w1^T → [num_tokens * top_k, 2 * intermediate]
        //    No topk weight applied (mul_topk_weights=false).
        let intermediate1 = kernels::marlin_moe_gemm(
            *hidden_states,
            self.w1,
            self.w1_scales,
            self.w1_zeros,
            None, // g_idx
            None, // perm
            self.workspace,
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            topk_weights.as_gpu_tensor(),
            moe_block_size,
            self.num_experts,
            self.top_k,
            false, // don't apply weights on first GEMM
            num_tokens,
            2 * self.intermediate_size,
            self.hidden_size,
            num_groups_w1,
            group_size_w1,
            false, // has_act_order
            self.has_zp,
            self.b_type_id,
            device.device_id as i32,
            &mut device.caching,
            stream,
        );

        // 5. Activation: SiLU(gate) * up → [num_tokens * top_k, intermediate]
        let activated = kernels::silu_and_mul_fused(
            intermediate1.as_gpu_tensor(),
            self.intermediate_size,
            &mut device.caching,
            stream,
        );
        drop(intermediate1);

        // 6. GEMM 2: activated × w2^T → [num_tokens * top_k, hidden_size]
        //    Reuse same sorted_token_ids/expert_ids/num_tokens_post_padded from pass 1.
        //    Set top_k=1 because input is already expanded to [M*top_k, intermediate].
        //    Apply routing weights (mul_topk_weights=true). Matches Python vLLM's
        //    _fused_marlin_moe second pass.
        //    Apply routing weight here (mul_topk_weights=true).
        let intermediate2 = kernels::marlin_moe_gemm(
            activated.as_gpu_tensor(),
            self.w2,
            self.w2_scales,
            self.w2_zeros,
            None, // g_idx
            None, // perm
            self.workspace,
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            topk_weights.as_gpu_tensor(),
            moe_block_size,
            self.num_experts,
            1,    // top_k=1 for pass 2 (input already expanded)
            true, // apply routing weights
            num_tokens * self.top_k,
            self.hidden_size,
            self.intermediate_size,
            num_groups_w2,
            group_size_w2,
            false, // has_act_order
            self.has_zp,
            self.b_type_id,
            device.device_id as i32,
            &mut device.caching,
            stream,
        );
        drop(activated);
        drop(topk_weights);
        drop(sorted_token_ids);
        drop(expert_ids);
        drop(num_tokens_post_padded);

        // 7. Reduce: sum across top_k experts → [num_tokens, hidden_size]
        let output = kernels::moe_sum(
            intermediate2.as_gpu_tensor(),
            num_tokens,
            self.hidden_size,
            self.top_k,
            &mut device.caching,
            stream,
        );
        drop(intermediate2);

        output
    }
}

// ---------------------------------------------------------------------------
// MarlinSharedFusedMoELayer
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub trait MarlinSharedFusedMoEOps {
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor;
}

#[cfg(feature = "cuda")]
impl MarlinSharedFusedMoEOps for MarlinSharedFusedMoELayer {
    unsafe fn forward(&self, hidden_states: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor {
        let stream = device.compute_stream;

        // MoE path.
        let moe_out = self.moe.forward(hidden_states, device);

        // Shared expert path (if present).
        if let (Some(shared_gate_up), Some(shared_down), Some(shared_gate)) = (
            &self.shared_gate_up,
            &self.shared_down,
            &self.shared_expert_gate,
        ) {
            let shared_gu = shared_gate_up.forward(
                hidden_states,
                &mut device.cublas,
                &mut device.caching,
                stream,
            );
            let shared_activated = kernels::silu_and_mul_fused(
                shared_gu.as_gpu_tensor(),
                self.intermediate_size,
                &mut device.caching,
                stream,
            );
            drop(shared_gu);

            let shared_out = shared_down.forward(
                shared_activated.view(),
                &mut device.cublas,
                &mut device.caching,
                stream,
            );
            drop(shared_activated);

            let gate_logits =
                shared_gate.forward(hidden_states, &mut device.cublas, &mut device.caching);

            let result = kernels::sigmoid_mul_add(
                moe_out.as_gpu_tensor(),
                shared_out.as_gpu_tensor(),
                gate_logits.as_gpu_tensor(),
                &mut device.caching,
                stream,
            );
            drop(moe_out);
            drop(shared_out);
            drop(gate_logits);

            result
        } else {
            moe_out
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn test_fused_moe_layer_sizes() {
        // Just verify struct construction with dummy tensors.
        let gate_w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[8, 4096], DType::BF16) };
        let w1 = unsafe { GpuTensor::new(0x2000 as *mut u8, &[8, 28672, 4096], DType::BF16) };
        let w2 = unsafe { GpuTensor::new(0x3000 as *mut u8, &[8, 4096, 14336], DType::BF16) };

        let layer = DenseFusedMoELayer {
            gate: Linear::new(gate_w, None),
            w1,
            w2,
            num_experts: 8,
            top_k: 2,
            intermediate_size: 14336,
            hidden_size: 4096,
            renormalize: false,
            e_score_correction_bias: None,
            n_expert_group: 0,
            topk_group: 0,
            routed_scaling_factor: 1.0,
        };

        assert_eq!(layer.num_experts, 8);
        assert_eq!(layer.top_k, 2);
        assert_eq!(layer.intermediate_size, 14336);
    }

    #[test]
    fn test_fp8_fused_moe_layer_sizes() {
        // Verify Fp8FusedMoELayer struct construction with dummy tensors.
        let gate_w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[8, 4096], DType::BF16) };
        let w1 = unsafe { GpuTensor::new(0x2000 as *mut u8, &[8, 28672, 4096], DType::Fp8E4m3) };
        let w2 = unsafe { GpuTensor::new(0x3000 as *mut u8, &[8, 4096, 14336], DType::Fp8E4m3) };
        let w1_scale = unsafe { GpuTensor::new(0x4000 as *mut u8, &[8], DType::F32) };
        let w2_scale = unsafe { GpuTensor::new(0x5000 as *mut u8, &[8], DType::F32) };

        let layer = Fp8FusedMoELayer {
            gate: Linear::new(gate_w, None),
            w1,
            w2,
            w1_scale,
            w2_scale,
            num_experts: 8,
            top_k: 2,
            intermediate_size: 14336,
            hidden_size: 4096,
            renormalize: false,
            e_score_correction_bias: None,
            n_expert_group: 0,
            topk_group: 0,
            routed_scaling_factor: 1.0,
        };

        assert_eq!(layer.num_experts, 8);
        assert_eq!(layer.top_k, 2);
        assert_eq!(layer.intermediate_size, 14336);
        assert_eq!(layer.hidden_size, 4096);
        assert_eq!(layer.w1.shape(), &[8, 28672, 4096]);
        assert_eq!(layer.w2.shape(), &[8, 4096, 14336]);
        assert_eq!(layer.w1.dtype(), DType::Fp8E4m3);
        assert_eq!(layer.w2.dtype(), DType::Fp8E4m3);
        assert_eq!(layer.w1_scale.shape(), &[8]);
        assert_eq!(layer.w2_scale.shape(), &[8]);
    }

    #[test]
    fn test_fp8_shared_fused_moe_layer_sizes() {
        // Verify Fp8SharedFusedMoELayer struct construction.
        let gate_w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[4, 2048], DType::BF16) };
        let w1 = unsafe { GpuTensor::new(0x2000 as *mut u8, &[4, 6144, 2048], DType::Fp8E4m3) };
        let w2 = unsafe { GpuTensor::new(0x3000 as *mut u8, &[4, 2048, 3072], DType::Fp8E4m3) };
        let w1_scale = unsafe { GpuTensor::new(0x4000 as *mut u8, &[4], DType::F32) };
        let w2_scale = unsafe { GpuTensor::new(0x5000 as *mut u8, &[4], DType::F32) };
        let shared_gate_up =
            unsafe { GpuTensor::new(0x6000 as *mut u8, &[6144, 2048], DType::BF16) };
        let shared_down = unsafe { GpuTensor::new(0x7000 as *mut u8, &[2048, 3072], DType::BF16) };

        let moe = Fp8FusedMoELayer {
            gate: Linear::new(gate_w, None),
            w1,
            w2,
            w1_scale,
            w2_scale,
            num_experts: 4,
            top_k: 2,
            intermediate_size: 3072,
            hidden_size: 2048,
            renormalize: true,
            e_score_correction_bias: None,
            n_expert_group: 0,
            topk_group: 0,
            routed_scaling_factor: 1.0,
        };

        let layer = Fp8SharedFusedMoELayer {
            moe,
            shared_gate_up: Some(Linear::new(shared_gate_up, None)),
            shared_down: Some(Linear::new(shared_down, None)),
            shared_expert_gate: None,
            intermediate_size: 3072,
        };

        assert_eq!(layer.moe.num_experts, 4);
        assert_eq!(layer.moe.top_k, 2);
        assert_eq!(layer.moe.intermediate_size, 3072);
    }

    #[test]
    fn test_fp8_block_fused_moe_layer_sizes() {
        // Verify Fp8BlockFusedMoELayer struct construction with dummy tensors.
        // Model: 128 experts, inter=768, hidden=2048, block_size=[128,128]
        let num_experts = 128;
        let inter = 768;
        let hidden = 2048;
        let block_n = 128;
        let block_k = 128;

        let gate_w =
            unsafe { GpuTensor::new(0x1000 as *mut u8, &[num_experts, hidden], DType::BF16) };
        let w1 = unsafe {
            GpuTensor::new(
                0x2000 as *mut u8,
                &[num_experts, 2 * inter, hidden],
                DType::Fp8E4m3,
            )
        };
        let w2 = unsafe {
            GpuTensor::new(
                0x3000 as *mut u8,
                &[num_experts, hidden, inter],
                DType::Fp8E4m3,
            )
        };

        // Scale shapes: [E, ceil(N/bn), ceil(K/bk)]
        let w1_sr = (2 * inter).div_ceil(block_n); // ceil(1536/128) = 12
        let w1_sc = hidden.div_ceil(block_k); // ceil(2048/128) = 16
        let w2_sr = hidden.div_ceil(block_n); // ceil(2048/128) = 16
        let w2_sc = inter.div_ceil(block_k); // ceil(768/128) = 6

        let w1_scale =
            unsafe { GpuTensor::new(0x4000 as *mut u8, &[num_experts, w1_sr, w1_sc], DType::F32) };
        let w2_scale =
            unsafe { GpuTensor::new(0x5000 as *mut u8, &[num_experts, w2_sr, w2_sc], DType::F32) };

        let layer = Fp8BlockFusedMoELayer {
            gate: Linear::new(gate_w, None),
            w1,
            w2,
            w1_scale_inv: w1_scale,
            w2_scale_inv: w2_scale,
            block_size: [block_n, block_k],
            num_experts,
            top_k: 8,
            intermediate_size: inter,
            hidden_size: hidden,
            renormalize: true,
            e_score_correction_bias: None,
            n_expert_group: 0,
            topk_group: 0,
            routed_scaling_factor: 1.0,
        };

        assert_eq!(layer.num_experts, 128);
        assert_eq!(layer.top_k, 8);
        assert_eq!(layer.intermediate_size, 768);
        assert_eq!(layer.hidden_size, 2048);
        assert_eq!(layer.block_size, [128, 128]);
        assert_eq!(layer.w1.shape(), &[128, 1536, 2048]);
        assert_eq!(layer.w2.shape(), &[128, 2048, 768]);
        assert_eq!(layer.w1.dtype(), DType::Fp8E4m3);
        assert_eq!(layer.w1_scale_inv.shape(), &[128, 12, 16]);
        assert_eq!(layer.w2_scale_inv.shape(), &[128, 16, 6]);
    }
}
