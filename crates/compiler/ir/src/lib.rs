// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral universal interpreter instruction set.
//!
//! [`Instruction`] is a closed enum over every kernel-call shape any
//! solver-picked `Implementation` produces. It is pure data — every
//! variant carries only scalar / `&'static str` fields — so it compiles
//! under any feature combination and names no backend. The match in
//! `Instruction::eval` (the CUDA interpreter) and the metal worker both
//! live in the `scratchy-forward-compiler` crate, which adds the
//! runtime methods on top of this neutral definition via extension
//! traits.
//!
//! [`WeightAccessors`] is the per-arch weight-accessor dispatch trait.
//! Its default-method return types name only the neutral layer/tensor
//! type defs in `scratchy-layers` / `scratchy-tensors`; the per-arch
//! `impl`s are emitted by `scratchy-forward-compiler-macro`.

// `MAX_DIMS` is needed by the `Instruction::Reshape` variant which is
// ungated, so this import stays at the module level. `MAX_DIMS` is not
// root-exported from `scratchy-tensors`, so use the module path.
use scratchy_tensors::tensor::MAX_DIMS;
// `MetalDtype` / `ScaleDtype` back the `CanonicalParams::METAL_DTYPE` /
// `SCALE_DTYPE` consts. They relocated to the cfg-free
// `scratchy-tensors` core so this cfg-free crate can name them without a
// backend feature.
use scratchy_tensors::{MetalDtype, ScaleDtype};
// The `GpuTensor`, `AffineQuantEmbedding` (in `scratchy-quantizations`), and
// `scratchy_layers` layer structs are referenced exclusively through
// fully-qualified paths (`::scratchy_tensors::tensor::GpuTensor`,
// `::scratchy_quantizations::AffineQuantEmbedding`,
// `::scratchy_layers::LinearLayer`, etc.) in the `WeightAccessors`
// trait return types, so no `use` import is needed here. The
// `Instruction` enum itself references only `u32` / scalar fields plus
// `MAX_DIMS`.

/// Per-arch weight-accessor dispatch trait.
///
/// Each per-`L` method matches on `(bucket, op_idx)` against the
/// closed set of weight-consuming positions the proc-macro emitted
/// for this arch. Replaces the `WtFn<W, L>` / `CosSinFn<W>` fn-
/// pointer fields previously stored on `Instruction<W>` variants
/// AND the parallel `WeightAccessorIdx` array.
///
/// Why `(bucket, op_idx)`: the `Instruction` variant determines the
/// *kind* of weight (RmsNorm always consumes an RmsNorm). What
/// varies per call site is *which named field* on `Weights` the per-
/// arch impl returns. The bucket id + position-in-bucket fully
/// identify the call site; the layer disambiguates the slice.
///
/// Per-arch impls are macro-generated — see
/// `scratchy-forward-compiler-macro::codegen::emit_weight_accessors_impl`.
///
/// Sibling methods cover every weight kind: `rms_norm_at`,
/// `embedding_at`, `linear_at`, `layer_norm_at`, `marlin_at`,
/// `bnb4_at`, `fp8_at`, the four MoE getters, and `cos_sin_at`.
/// Default bodies all `unreachable!()` so per-arch impls only
/// override the methods their generated tape actually exercises.
/// Each method matches on `(bucket, op_idx, slot)` to resolve the
/// named weight at this tape call site. `slot` disambiguates
/// multiple accessors of the same kind at the same position
/// (e.g. `FusedQkvQkNormRopeCache` consumes 3 LinearLayers — q_proj,
/// k_proj, v_proj — all at the same `op_idx`; their `slot`s are
/// 0, 1, 2). For variants with one accessor per kind, `slot` is
/// always 0.
///
/// Default bodies all `unreachable!()` so per-arch impls only
/// override the methods their generated tape actually calls.
///
/// Cross-backend: both the cuda eval body and the metal worker call
/// into the same per-arch impl emitted by
/// `scratchy-forward-compiler-macro::codegen::emit_weight_accessors_impl`.
/// Gated on `any(cuda, metal)` because the return types reference
/// `scratchy_layers::*` structs which only compile under
/// one of those features.
pub trait WeightAccessors {
    fn rms_norm_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::scratchy_layers::RmsNorm {
        unreachable!("rms_norm_at not implemented for this arch")
    }
    fn embedding_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::scratchy_layers::Embedding {
        unreachable!("embedding_at not implemented for this arch")
    }
    fn linear_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::scratchy_layers::LinearLayer {
        unreachable!("linear_at not implemented for this arch")
    }
    fn layer_norm_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::scratchy_layers::LayerNorm {
        unreachable!("layer_norm_at not implemented for this arch")
    }
    fn marlin_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::scratchy_layers::MarlinLinear {
        unreachable!("marlin_at not implemented for this arch")
    }
    fn bnb4_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::scratchy_layers::Bnb4bitLinear {
        unreachable!("bnb4_at not implemented for this arch")
    }
    fn fp8_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::scratchy_layers::Fp8AnyLinear {
        unreachable!("fp8_at not implemented for this arch")
    }
    fn deepseek_moe_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::scratchy_layers::layers_moe::DeepSeekV2MoELayer {
        unreachable!("deepseek_moe_at not implemented for this arch")
    }
    fn gated_delta_net_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::scratchy_layers::GatedDeltaNetLayer {
        unreachable!("gated_delta_net_at not implemented for this arch")
    }
    fn deepseek_moe_fp8_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::scratchy_layers::layers_moe::DeepSeekV2Fp8BlockMoELayer {
        unreachable!("deepseek_moe_fp8_at not implemented for this arch")
    }
    fn deepseek_moe_ggml_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::scratchy_layers::layers_moe::DeepSeekV2GgmlMoELayer {
        unreachable!("deepseek_moe_ggml_at not implemented for this arch")
    }
    fn fused_moe_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::scratchy_layers::layers_moe::FusedMoELayer {
        unreachable!("fused_moe_at not implemented for this arch")
    }
    fn shared_fused_moe_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::scratchy_layers::layers_moe::SharedFusedMoELayer {
        unreachable!("shared_fused_moe_at not implemented for this arch")
    }
    /// Metal-only: Gemma-4 router bundle (`GemmaMoe` op, base `router`).
    fn gemma_router_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::scratchy_layers::layers_moe::GemmaRouterLayer {
        unreachable!("gemma_router_at not implemented for this arch")
    }
    /// Metal-only: Gemma-4 SwitchGLU experts bundle (`GemmaMoe` op, base
    /// `experts.switch_glu`).
    fn gemma_switch_glu_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::scratchy_layers::layers_moe::SwitchGluExpertsLayer {
        unreachable!("gemma_switch_glu_at not implemented for this arch")
    }
    fn cos_sin_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> ::scratchy_tensors::tensor::GpuTensor {
        unreachable!("cos_sin_at not implemented for this arch")
    }
    /// Spans rope-on-read: the cos/sin table the paged attention kernels
    /// use to re-rope cached unrotated K on read. Resolved by LAYER
    /// CLASS, not by op operand (attention ops carry no cos_sin slot):
    /// `is_global` selects the global/full-attention `rotary` cache vs
    /// the sliding/local `rotary_local` cache (dual-rotary arches like
    /// Gemma4). Only called when `W::ROPE_ON_READ` and the attention
    /// lowering binds `WeightBundleKind::RopeOnReadCosSin`; the default
    /// is unreachable (non-spans arches never bind it).
    fn rope_on_read_cos_sin(
        &self,
        _is_global: bool,
        _layer: u32,
    ) -> ::scratchy_tensors::tensor::GpuTensor {
        unreachable!("rope_on_read_cos_sin not implemented for this arch")
    }
    /// Metal-only: MLX-affine int4 quantized embedding. Resolves the
    /// `AffineQuantEmbedding` field on `Weights` for the
    /// `Instruction::AffineEmbed` lowering.
    fn affine_quant_embedding_at(
        &self,
        _bucket: u32,
        _op_idx: u32,
        _slot: u32,
        _layer: u32,
    ) -> &::scratchy_quantizations::AffineQuantEmbedding {
        unreachable!("affine_quant_embedding_at not implemented for this arch")
    }
}

/// Universal opcode set. Tuple variants throughout — keeps each
/// row in a per-canonical static slice on a single line of cargo
/// expand. Field order per variant matches the per-Impl
/// `OpcodeShape::fields` order in `impl_lib.rs`.
///
/// No longer generic over `W` — every variant lost its `WtFn<W, L>` /
/// `CosSinFn<W>` field when fan_out moved weight resolution to the
/// per-arch `WeightAccessors` trait at the tape level. `W` lives only
/// on `eval<W: CanonicalParams>` and `run<W>` / `run_backbone<W>` as
/// method-level generics, parameterizing the interpreter's access to
/// per-canonical model constants without polluting every static row.
#[allow(clippy::type_complexity)]
#[derive(Debug)]
pub enum Instruction {
    Embed(u32),
    /// `RmsNorm(in_slot, out_slot, layer, hidden_size, m_multiplier)`.
    /// Weight accessor lives at the tape level — see [`Tape`].
    ///
    /// `hidden_size` is the row width the kernel reduces over;
    /// `m_multiplier` is the number of normalization rows per input
    /// token. For the standard residual-stream RmsNorm:
    /// `(hidden_size = W::HIDDEN_SIZE, m_multiplier = 1)`. For
    /// per-head `q_norm` (Qwen3): `(W::HEAD_DIM, num_q_heads)`;
    /// for `k_norm`: `(W::HEAD_DIM, num_kv_heads)`. The metal kernel
    /// dispatches `bucket_m * m_multiplier` threadgroups, each
    /// reducing `hidden_size` elements. Pre-Qwen3 the lowering
    /// hardcoded `(W::Q_SIZE, 1)` — correct for Llama (Q_SIZE ==
    /// hidden) but wrong for Qwen3 (Q_SIZE != hidden, q_norm has
    /// per-head reduction). The cuda interpreter reads tile shape
    /// directly and ignores both fields.
    RmsNorm(u32, u32, u32, u32, u32),
    /// CohereLayerNorm-flavored norm (subtracts mean before scaling).
    /// Claimed from the `(mean, sub, rmsnorm)` math trio in the DSL.
    /// Weight is typed `RmsNorm` because the DSL author writes
    /// `rmsnorm(centered, w)`; the wrapper struct is structurally
    /// identical to the now-retired `CohereLayerNorm` (one `weight`
    /// + one `eps`), and the kernel reads only those two fields.
    MeanSubRmsNorm(u32, u32, u32),
    /// Torch-style LayerNorm with bias. Claimed from the 4-tile pattern
    /// `(mean, sub, rmsnorm, bias_add)` in the DSL — extends the
    /// `MeanSubRmsNorm` math trio with a learned bias addition. Weight
    /// is typed `LayerNorm` (not `RmsNorm`) so the loader auto-pulls
    /// `<prefix>.weight` AND `<prefix>.bias` together; the matcher's
    /// `required_weights` returns one accessor whose source is the
    /// rmsnorm's weight ref. The DSL bias-weight ref (e.g.
    /// `attn_norm.bias[layer]`) is structural-only — same trick the
    /// dense `(Gemm, BiasAdd)` fusion plays via `LinearLayer`.
    /// Used by encoder models like ModernBERT and vision towers like
    /// Qwen2-VL.
    MeanSubRmsNormBiasAdd(u32, u32, u32),
    /// `Reshape(in_slot, out_slot, dims_lit, dims_nt_pow, dims_div_lit, ndim)`.
    /// Output axis i is computed as
    /// `(dims_lit[i] * num_tokens^dims_nt_pow[i]) / dims_div_lit[i]`.
    /// `dims_div_lit` is `1` for every dim by default (G.5.f.a opens
    /// the divisor for DSL-authored arithmetic — the merger's
    /// `[num_tokens / vision_merge_factor, vision_merge_hidden]`
    /// is the first consumer, decomposing as
    /// `dims_div_lit = [vision_merge_factor, 1]`).
    Reshape(
        u32,
        u32,
        [u32; MAX_DIMS],
        [u8; MAX_DIMS],
        [u32; MAX_DIMS],
        u8,
    ),
    Add(u32, u32),
    /// Tensor-parallel all-reduce-sum on the slot in place. Inserted
    /// by the lowering pass after every gemm whose weight is
    /// row-parallel (`ShardDim1`) and after the vocab-parallel embed.
    /// At tp=1 the lowering pass emits zero of these.
    #[cfg(feature = "nccl")]
    AllReduce(u32),
    /// Tensor-parallel all-gather along the last dim: `(in_slot, out_slot)`.
    /// Inserted by the lowering pass after the lm_head Gemm at tp>1
    /// (lm_head is vocab-parallel `ShardDim0`).
    #[cfg(feature = "nccl")]
    AllGather(u32, u32),
    /// Multimodal embed splice — D2D-copy projected vision-encoder
    /// rows into the placeholder positions of the post-embed hidden
    /// states in-place. Always inserted by `tp_lowering::insert_mm_splices`
    /// after every `Instruction::Embed` (after the vocab-parallel
    /// `Instruction::AllReduce` at tp>1, so the splice runs on the
    /// fully-reduced embedding and its D2D overwrite is NOT summed
    /// across ranks). At runtime the op is a no-op when
    /// `ForwardCtx::embed_patches` is empty (text-only batches) —
    /// one extra slot check per forward pass, cost negligible.
    SpliceMmEmbeds(u32),
    ScalarMul(u32, u32, f32),
    TanhSoftCap(u32, u32),
    /// Unit-gain RMSNorm (no learnable scale — mlx `RMSNormNoScale`,
    /// Gemma4 `v_norm`). `(in_slot, out_slot, hidden_size,
    /// m_multiplier)`: input treated as `[M * m_multiplier,
    /// hidden_size]` rows, each normalized by rsqrt(mean(x²)+eps)
    /// with gain ≡ 1. The per-head width is baked per attention
    /// class by `RmsNormUnitImpl::fan_out`.
    RmsNormUnit(u32, u32, u32, u32),
    /// Multiply by a loaded `[1]`-shaped weight (Gemma4
    /// `layer_scalar[layer]`). `(in_slot, out_slot, layer)`; the
    /// weight resolves through the tape-level RmsNorm-kind accessor.
    ScalarWeightMul(u32, u32, u32),
    /// Gemma4 post-FFN tail fused into one kernel:
    /// `out = (rmsnorm(delta, gains) + residual) * layer_scalar`
    /// (the DSL's `rmsnorm(down) → add(.., hidden) → scalar_weight_mul`
    /// chain — norm-THEN-add, the mirror image of `FusedAddRmsNorm`).
    /// `(delta_slot, residual_slot, out_slot, layer, hidden_size)`.
    /// Weights resolve through RmsNorm-kind accessors: sub-slot 0 =
    /// post-FFN norm gains `[hidden]`, sub-slot 1 = `layer_scalar [1]`.
    NormAddScalarMul(u32, u32, u32, u32, u32),
    /// Gemma4 pre-attention tail fused into the rope dispatch:
    /// per-head `rmsnorm(q, q_gains)` / `rmsnorm(k, k_gains)` /
    /// `rmsnorm_unit(v)` prologues + NeoX RoPE + paged KV write.
    /// Field layout mirrors [`RopeAppend`](Instruction::RopeAppend):
    /// `(q_slot, k_slot, v_slot, q_out_slot, k_out_slot, v_out_slot,
    /// layer, interleaved, is_global)` — in-slots are the RAW
    /// projection outputs (pre-norm). Q is normed+rotated in place;
    /// K/V go to the cache ONLY (their arena tiles are dead on
    /// Gemma4, and k_eq_v global layers share one raw buffer).
    /// Weights: RmsNorm-kind sub-slots 0/1 = q/k gains; CosSin
    /// auto-injected.
    RopeAppendNormed(u32, u32, u32, u32, u32, u32, u32, bool, bool),
    /// `FusedAddRmsNorm(delta_slot, residual_slot, layer, hidden_size, m_multiplier)`.
    /// See [`RmsNorm`](Instruction::RmsNorm) for the field semantics.
    /// FusedAddRmsNorm is always on the residual stream so
    /// `m_multiplier = 1` and `hidden_size = W::HIDDEN_SIZE`.
    FusedAddRmsNorm(u32, u32, u32, u32, u32),
    FusedAddRmsNormWithOffset(u32, u32, u32, f32),
    /// `ScalarOffsetRmsNorm(in_slot, out_slot, layer, offset, hidden_size, m_multiplier)`.
    /// `rmsnorm(x, weight + offset)` — the Gemma `weight + 1.0`
    /// zero-centered form. `hidden_size` / `m_multiplier` mirror
    /// [`RmsNorm`](Instruction::RmsNorm): residual-stream norms are
    /// `(HIDDEN_SIZE, 1)`; per-head q/k norms are `(head_dim,
    /// num_q/kv_heads)`. The cuda eval reads the tile shape and ignores
    /// both; the metal lowering bakes them into the kernel fn-consts.
    ScalarOffsetRmsNorm(u32, u32, u32, f32, u32, u32),
    /// Norm→Gemm fusion: `cutlass_gemm(rms_norm(in), gemm_w)`. The
    /// CUTLASS tile is bucket-pickable per `CUTLASS_TILE_ZOO` entry.
    /// Reuses existing `kernels::rms_norm` + `cutlass::cutlass_gemm`
    /// — no new .cu file. Matches body norms whose only downstream
    /// consumer is a single dense Gemm.
    CutlassFusedRmsNormGemm(u32, u32, u32, u32, u32, u32, u32, u32),
    /// Norm→Gemm fusion for CohereLayerNorm-flavored norms.
    /// Claimed from the `(mean, sub, rmsnorm, gemm)` 4-tile pattern
    /// where the Gemm is the sole consumer of the rmsnorm output.
    /// Runs `cohere_layer_norm` then `cutlass_gemm` sequentially.
    CutlassFusedMeanSubRmsNormGemm(u32, u32, u32, u32, u32, u32, u32, u32),
    /// (Add, RmsNorm, Gemm) 3-tile fusion. Claim shape captures
    /// the lm_head canonical pattern `x = x + delta; logits =
    /// lm_head(rmsnorm(x))`. Runtime: `fused_add_rms_norm_inplace`
    /// then `cutlass_gemm`. The Add's residual update is exposed as
    /// a TensorView aliasing the residual upstream OwnedTensor (same
    /// alias semantics as `FusedAddRmsNorm`); the Gemm output is a
    /// fresh OwnedTensor.
    CutlassFusedAddRmsNormGemm(u32, u32, u32, u32, u32, u32, u32, u32, u32),
    Gemm(u32, u32, u32, u32, u32),
    /// cuBLAS-side peer to `CutlassGemmAdd`. cuBLAS GEMM produces a
    /// delta; `add_inplace` then folds it into the residual buffer.
    /// Output is the residual upstream's OwnedTensor (aliased via
    /// the codegen prelude); no `out_slot` payload.
    FusedCublasGemmAdd(u32, u32, u32, u32, u32),
    FusedGemmBias(u32, u32, u32),
    /// Metal-only per-row bias broadcast add. Emitted by
    /// `MetalBiasAddImpl` when the synth-pre-attn megakernel doesn't
    /// claim the biased QKV chain (today: M ≥ 2 prefill, where the
    /// cost CSV picks unfused).
    /// CUDA's analogue is `FusedGemmBias`, which folds the bias into
    /// cuBLAS's gemm_bias epilog. On Metal the singleton path
    /// dispatches a separate `KernelId::BiasAdd` after the AffineQmm
    /// / Gemm; once the synth path absorbs biases the solver will
    /// prefer that over this singleton on decode workloads.
    ///
    /// Fields: `(in_slot, out_slot, layer, n, is_affine)`. Bias is
    /// resolved through the tape-level `WeightAccessors::linear_at`
    /// at the macro-emitted `(bucket, op_idx, slot=0, layer)` site;
    /// `is_affine` tells the worker whether to pull `LinearLayer::Dense
    /// .bias` or `LinearLayer::AffineQuant.linear_bias`. `n` is the
    /// bias's broadcast dimension — baked into the specialized
    /// pipeline as `function_constant(0)`.
    MetalBiasAdd(u32, u32, u32, u32, bool),
    FusedGateUpSiluMul(u32, u32, u32),
    FusedGateUpGeluMul(u32, u32, u32),
    FusedQkvRopeCache(u32, u32, u32, bool, bool),
    FusedQkvQkNormRopeCache(u32, u32, u32, f32, f32),
    FusedQkvRopePrefill(u32, u32, u32, u32, u32, bool, bool),
    AttentionViaCache(u32, u32, u32, bool),
    AttentionPrefillContiguous(u32, u32, u32, u32, bool),
    /// Prefill attention reading K/V from the paged KV cache via
    /// block_table indirection. `(q_slot, out_slot, layer, interleaved)`.
    /// Drops the (k_slot, v_slot) pair that
    /// [`Instruction::AttentionPrefillContiguous`] carries — the
    /// upstream `RopeAppend` already wrote rotated K + raw V into the
    /// per-layer paged cache slots, and the kernel reads them through
    /// `block_table` with the K-axis covering the FULL `seqused_k[seq]`
    /// (prefix + new). The per-Q causal mask shifts by
    /// `(seqused_k[seq] - new_q_for_seq)` so prior cached prefix
    /// contributes to attention. Required for chunked prefill, prefix
    /// caching, mixed prefill+decode batches, and multi-turn chat —
    /// scenarios `AttentionPrefillContiguous` cannot handle because
    /// its K-axis is bounded by `cu_seqlens_q` (new tokens only).
    /// Currently emitted only by the metal adapter; cuda continues to
    /// route prefill through `flash_attn_contiguous`.
    AttentionPrefillPaged(u32, u32, u32, bool),
    /// Bidirectional / encoder attention. Reads contiguous Q/K/V from
    /// the upstream tile slots; calls `flash_attn_contiguous` with
    /// `is_causal=false` and a null cos_sin pointer (RoPE applied
    /// separately upstream). No KV cache, no per-layer cos_sin —
    /// the encoder DSL form is `attention(q, k, v)` (3 args). Q/K/V
    /// must already be 3D `[T, heads, head_dim]` from the upstream
    /// projection chain. Output is reshaped to `[T, Q_SIZE]`.
    EncoderAttention(u32, u32, u32, u32),
    SlidingAttentionViaCache(u32, u32, u32, bool),
    SlidingAttentionPrefillContiguous(u32, u32, u32, u32, bool),
    /// Sliding-window variant of [`Instruction::AttentionPrefillPaged`]
    /// — same operands `(q_slot, out_slot, layer, interleaved)` and the
    /// same paged-cache read path; the metal lowering arm additionally
    /// bakes `W::SLIDING_WINDOW` into the kernel's `ATTN_WINDOW`
    /// function constant and never routes to the steel kernel (steel
    /// has no window support). Emitted only by the metal adapter
    /// (`metal::attention::fan_out`, sliding+multihead arm); cuda
    /// routes sliding prefill through
    /// [`Instruction::SlidingAttentionPrefillContiguous`].
    SlidingAttentionPrefillPaged(u32, u32, u32, bool),
    /// Vision-tower variable-length attention: `(q_slot, k_slot,
    /// v_slot, out_slot, cu_seqlens_kind)`. The `cu_seqlens_kind`
    /// u8 discriminant selects which `(cu_seqlens, max_seqlen)` pair
    /// the kernel reads:
    ///
    /// - `0` (Default): `ForwardCtx::cu_seqlens_q` + `max_seqlen_q`.
    ///   Qwen2-VL's single-cu-seqlens path; the host wrapper
    ///   populates these from the per-batch concatenated boundaries.
    /// - `1` (Full): `ForwardCtx::vision_cu_seqlens_full` +
    ///   `vision_max_seqlen_full`. Qwen2.5-VL's full-frame layers
    ///   (`fullatt_block_indexes = [7, 15, 23, 31]`).
    /// - `2` (Window): `ForwardCtx::vision_cu_seqlens_window` +
    ///   `vision_max_seqlen_window`. Qwen2.5-VL's windowed-attn
    ///   layers (the other 28 of 32).
    ///
    /// Bidirectional (no causal mask), no softcap, no fused rope —
    /// rope is applied upstream by `Instruction::VisionRope`. Output
    /// reshaped to rank-2 `[L, vision_num_heads * vision_head_dim]`
    /// to match the FUF's q-shape contract.
    VarlenAttention(u32, u32, u32, u32, u8),
    /// Vision 2D RoPE pair-rotation: `(q_in, k_in, q_out, k_out)`.
    /// Reads cos / sin from `ForwardCtx::vision_rope_cos` /
    /// `vision_rope_sin`. In-place on q / k buffers; the output slots
    /// reinsert the same `OwnedTensor`s after mutation. Handles the
    /// rank-2 → rank-3 reshape internally; the kernel
    /// `vision_rope_apply` requires rank-3 `[L, H, D]`.
    VisionRope(u32, u32, u32, u32),
    /// Quick-GELU activation: `(in_slot, out_slot)`. In-place
    /// elementwise mutation, take-owned + kernel + reinsert. Same
    /// shape as `TanhSoftCap` / `ScalarMul` consume-pattern.
    QuickGelu(u32, u32),
    /// GELU tanh-approximation activation: `(in_slot, out_slot)`.
    /// In-place mutation. Mirrors `QuickGelu` / `GeluErf`; matches
    /// PyTorch `nn.GELU(approximate="tanh")`. Used by SigLIP /
    /// Gemma3-MM vision MLP.
    Gelu(u32, u32),
    /// Vision learned positional embedding lookup: `(out_slot,
    /// weight_fn)`. Reads `ctx.fwd.vision_position_ids` (rank-1 u32
    /// view of length `num_tokens`) and gathers rows from the
    /// per-arch positional table via the same
    /// `kernels::embedding_gather_masked` kernel
    /// `Instruction::Embed` calls (with `vocab_offset = 0` /
    /// `vocab_per_rank = num_positions` so the mask never trips).
    /// Used by SigLIP / Gemma3-MM. Distinct from `Embed` because the
    /// table dim 0 is `vision_num_positions` (not `vocab_size`); the
    /// `OpKind::PosEmbed` shape sig anchors on the vision bound names
    /// so the macro emits a `Weights` field of the right type.
    PosEmbed(u32),
    /// Materialize the vision-prelude `pixels` extern as a tile:
    /// `(out_slot)`. Reads `ctx.fwd.pixels` (the rank-2
    /// `[num_tokens, vision_in_features]` view the
    /// `vision_forward` host wrapper writes onto `ForwardCtx`
    /// before invoking the vision interpreter), allocates a fresh
    /// `OwnedTensor` of the same shape/dtype, D2D-copies the
    /// pixels view into it, and publishes the result at `out_slot`.
    ///
    /// The copy is what lets the consume-pattern vision Impls
    /// (`QuickGelu` / `GeluErf` / `VisionRope`) downstream of
    /// `pixels` read it as a tile-table `Owned` entry — `take_owned`
    /// requires `Owned`, and a borrowed `External` wrapper around
    /// `ctx.fwd.pixels` would panic on the first such consumer.
    /// In G.5.f's real encoder body the first op on pixels is a
    /// non-consuming `gemm` (`patch_embed`), so the copy is paid
    /// once per encoder invocation regardless. Synthesized
    /// exclusively by `vision_lowering::materialize_pixels` —
    /// never appears in any DSL.
    LoadPixels(u32),
    /// Materialize the Qwen3.5-VL `pos_embeds` extern as a tile:
    /// `(out_slot)`. The exact sibling of [`Self::LoadPixels`] — reads
    /// `ctx.fwd.pos_embeds` (the rank-2 `[num_tokens, vision_embed_dim]`
    /// host-interpolated learned positional embedding the
    /// `vision_forward` wrapper writes onto `ForwardCtx`), D2D-copies it
    /// into a fresh `OwnedTensor`, and publishes it at `out_slot` so the
    /// downstream `add(pos_embeds, hidden_states)` reads it as an `Owned`
    /// tile. Synthesized exclusively by
    /// `vision_lowering::materialize_pos_embeds` — never appears in any DSL.
    LoadPosEmbeds(u32),
    /// Erf-form GELU activation: same shape as `QuickGelu`. Distinct
    /// numerics (`0.5 * x * (1 + erf(x / sqrt(2)))`).
    GeluErf(u32, u32),
    /// Row-permutation gather: `(in_slot, out_slot, indices_kind)`.
    /// Reads the source rank-2 tile from `in_slot`, the rank-1 u32
    /// indices buffer from `ForwardCtx::vision_window_index` (kind=0)
    /// or `ForwardCtx::vision_reverse_indices` (kind=1), and writes a
    /// fresh `OwnedTensor` of the same shape as the source to
    /// `out_slot`. Output row `i` = source row `indices[i]`. Used by
    /// Qwen2.5-VL's window-attention dispatch — tokens, cos, sin are
    /// gather-permuted into window order on encoder entry, and the
    /// merger output is permuted back to natural order at exit.
    EmbeddingGather(u32, u32, u8),
    /// 2-D non-overlapping average pool: `(in_slot, out_slot)`. Reads
    /// the source rank-2 tile `[L = ph², e]` from `in_slot`, walks the
    /// flat row index as a `(row, col)` pair on a `ph × ph` grid (with
    /// `ph = W::VISION_PATCH_GRID_SIDE`), and averages each k×k cell
    /// (`k = W::VISION_POOL_KERNEL`) into one output row. Output is a
    /// fresh `OwnedTensor` of shape `[(ph/k)², e]`. Used by Gemma3-MM's
    /// SigLIP→text projector. Stride == kernel (non-overlapping).
    AvgPool2d(u32, u32),
    /// CLIP-class CLS-token strip: `(in_slot, out_slot)`. Reads the
    /// rank-2 tile `[L, e]` from `in_slot` (where the CLS row has been
    /// run through the encoder at row 0), copies rows `1..L` into a
    /// fresh `OwnedTensor` of shape `[L - 1, e]`, and publishes that
    /// at `out_slot`. No baked-in constant — input dims are read off
    /// the source tile at runtime, so a misconfigured `vision_in_seq_len
    /// = vision_num_positions - 1` invariant in the variant config
    /// surfaces as a downstream shape mismatch (caught at expansion
    /// time by the bound-resolved sig), not as a kernel panic. Used by
    /// LLaVA-1.5 family with `vision_feature_select_strategy =
    /// "default"`.
    StripCls(u32, u32),
    FlashInferAttentionDecode(u32, u32, u32, u32, bool),
    FlashInferAttentionPrefill(u32, u32, u32, u32, u32, u32, bool),
    /// Hopper-native FA3 paged decode. Selected by the solver when
    /// `cuda_arch >= 90` and the cost CSV calibrated FA3 below FI for
    /// this `(num_tokens, sk_bucket)` cell. No softcap field — the
    /// vendored FA3 build excludes softcap variants
    /// (`FLASHATTENTION_DISABLE_SOFTCAP`).
    /// Args: (in_slot, out_slot, layer, head_dim).
    #[cfg(fa3_built)]
    FlashAttention3Decode(u32, u32, u32, u32),
    /// `(q_slot, k_slot, v_slot, q_out, k_out, v_out, layer,
    /// interleaved, is_global, kv_offsets)`. `is_global` selects the attention
    /// geometry class on hybrid sliding/global arches (Gemma4): the
    /// metal lowering reads `GLOBAL_HEAD_DIM/NUM_GLOBAL_KV_HEADS/
    /// GLOBAL_ROT_DIM` when true, the base consts when false.
    /// Identical on uniform models (GLOBAL_* default to base).
    /// `kv_offsets`: the additive offset the K and V it writes carry — see
    /// [`KvOffsets`].
    RopeAppend(u32, u32, u32, u32, u32, u32, u32, bool, bool, KvOffsets),
    MlaSplit(u32, u32, u32),
    MlaAttention(u32, u32, u32, u32, u32),
    /// Gated-DeltaNet linear attention (Qwen3.5 / Qwen3-Next). Args:
    /// `(qkv_slot, z_slot, a_slot, b_slot, out_slot, layer)`. Reads the
    /// `linear_attn[layer]` weight bundle (`GatedDeltaNetLayer`) via
    /// `gated_delta_net_at` and the ambient conv/ssm state from
    /// `ForwardCtx::{gdn_state, gdn_state_indices}` at `layer`. Output
    /// `[T, value_dim]`.
    GatedDeltaNet(u32, u32, u32, u32, u32, u32),
    /// Qwen3.5 attention output-gate split. Args `(qg_slot, q_slot, gate_slot)`.
    /// Deinterleaves the doubled `q_proj` output (per head `[query | gate]`)
    /// into two `[T, num_heads*head_dim]` owned tiles.
    GateSplit(u32, u32, u32),
    /// Qwen3.5 attention output gate. Args `(attn_slot, gate_slot, out_slot)`.
    /// `out = attn * sigmoid(gate)`.
    GateApply(u32, u32, u32),
    /// Qwen3.5-MoE shared-expert combine. Args `(routed_slot, shared_slot,
    /// gate_slot, out_slot)`. `out = routed + shared_y * sigmoid(g)`; `g` is
    /// `[T, 1]`, row-broadcast across the hidden axis.
    GateScale(u32, u32, u32, u32),
    DeepSeekMoe(u32, u32, u32),
    DeepSeekMoeFp8Block(u32, u32, u32),
    DeepSeekMoeGgml(u32, u32, u32),
    /// Mixtral-style BF16 fused MoE (no shared expert). Top-k routing
    /// via softmax; experts are stacked `[E, 2*inter, hidden]` /
    /// `[E, hidden, inter]` and dispatched through the fused-MoE GEMM.
    FusedMoe(u32, u32, u32),
    /// Qwen-MoE-style BF16 fused MoE PLUS shared expert with sigmoid
    /// gate. Routed experts use the same fused-MoE pipeline as
    /// `FusedMoe` (with `renormalize=true`); the shared expert is a
    /// SwiGLU MLP gated by `sigmoid(shared_expert_gate(x))`.
    SharedFusedMoe(u32, u32, u32),
    /// Metal-only Mixtral-style fused MoE: carries the full structural
    /// shape baked at macro-expansion time from the model config, so
    /// the Metal lowering pass + worker can specialize pipelines
    /// (`function_constant`s for moe_weighted_sum, affine_gather_qmv),
    /// stamp `Binding::Inline` u32s (top_k, axis_size), and size the
    /// per-bucket MoE scratch buffer — all without a runtime
    /// shape-resolution detour.
    ///
    /// Tuple fields: `(in_slot, out_slot, layer, num_experts, top_k,
    /// moe_intermediate_size, hidden_size, group_size, bits)`.
    /// `group_size` + `bits` come from the per-expert weight
    /// `StorageFormat::Affine`; for the on-disk mlx-community 4bit
    /// MoE checkpoints those are `64` and `4` respectively.
    ///
    /// Softmax order is fixed by variant identity: Mixtral does
    /// `topk → softmax(scores)`, no renorm flag. (See `SharedFusedMoe`
    /// for the Qwen `softmax → topk → take_along_axis` order.)
    /// CUDA-target macros keep emitting [`Instruction::FusedMoe`];
    /// the Metal-target `MetalFusedMoeImpl` (in
    /// `scratchy-forward-compiler-macro::metal::moe`) emits this variant when
    /// `profile.backend == Backend::Metal` and the per-expert weight
    /// storage is `Affine`.
    MetalFusedMoe(u32, u32, u32, u32, u32, u32, u32, u32, u32),
    /// Metal-only Qwen-MoE-style fused MoE + optional shared expert.
    /// Same rationale as [`Instruction::MetalFusedMoe`]: macro-baked
    /// shape for the Metal lowering arm.
    ///
    /// Tuple fields: `(in_slot, out_slot, layer, num_experts, top_k,
    /// moe_intermediate_size, hidden_size, shared_intermediate_size,
    /// group_size, bits, norm_topk_prob)`.
    /// `shared_intermediate_size = 0` means no shared expert — the
    /// lowering arm skips the shared-expert tail entirely
    /// (modern Qwen3-MoE-30B-A3B-Instruct ships this). Qwen1.5-MoE /
    /// Qwen2-MoE ship non-zero. `norm_topk_prob = true` triggers the
    /// post-`take_along_axis` renorm of gathered scores (Qwen3-MoE's
    /// `norm_topk_prob` config flag).
    ///
    /// Softmax order is fixed by variant identity: Qwen-MoE does
    /// `softmax → topk → take_along_axis(scores)`.
    MetalSharedFusedMoe(u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, bool),
    /// Metal-only Gemma-4 fused MoE block (the `gemma_moe` op). Routes off
    /// the post-attention residual (`router_in`, input 0) and runs the
    /// selected GeGLU experts on the pre-FF2-normed input (`expert_in`,
    /// input 1), summing by the routed weights. Distinct from
    /// [`Instruction::MetalSharedFusedMoe`] in three ways the lowering needs:
    /// a leading RMSNorm of `router_in` by the router gain, a softmax
    /// TEMPERATURE of `hidden_size^-0.5` applied to the gathered top-k scores
    /// BEFORE the softmax, a per-expert-scale gather-multiply after it, and
    /// GeGLU (not SiLU) activation. No shared expert, no renorm-by-sum.
    ///
    /// Tuple fields: `(router_in, expert_in, out, layer, num_experts, top_k,
    /// moe_intermediate_size, hidden_size, group_size, bits)`. `group_size`
    /// / `bits` are the EXPERT quant (4 / 64); the router.proj 8-bit dequant
    /// is handled at load inside `GemmaRouterLayer`.
    GemmaMoe(u32, u32, u32, u32, u32, u32, u32, u32, u32, u32),
    CutlassGemm(u32, u32, u32, u32, u32, u32, u32, u32),
    CutlassGemmSplitK(u32, u32, u32, u32, u32, u32, u32, u32, u32),
    CutlassGemmAdd(u32, u32, u32, u32, u32, u32, u32, u32),
    CutlassGemv(u32, u32, u32, u32, u32),
    CutlassFusedGemmBias(u32, u32, u32, u32, u32, u32, u32, u32),
    CutlassFusedGateUpSiluMul(u32, u32, u32, u32, u32, u32),
    CutlassFusedGateUpGeluMul(u32, u32, u32, u32, u32, u32, u32, u32),
    CutlassFusedQkvRopeCache(u32, u32, u32, bool, u32, u32, u32, u32, u32),
    CutlassFusedQkvRopePrefill(u32, u32, u32, u32, u32, bool, u32, u32, u32, u32, u32),
    MarlinGemm(u32, u32, u32),
    MarlinFusedGateUpSiluMul(u32, u32, u32),
    MarlinFusedGateUpGeluMul(u32, u32, u32),
    MarlinFusedQkvRopeCache(u32, u32, u32),
    MarlinFusedQkvRopePrefill(u32, u32, u32, u32, u32),
    Bnb4Gemm(u32, u32, u32),
    Bnb4FusedGateUpSiluMul(u32, u32, u32),
    Bnb4FusedGateUpGeluMul(u32, u32, u32),
    Bnb4FusedQkvRopeCache(u32, u32, u32),
    Bnb4FusedQkvRopePrefill(u32, u32, u32, u32, u32),
    GgmlGemm(u32, u32, u32),
    GgmlFusedGateUpSiluMul(u32, u32, u32),
    GgmlFusedGateUpGeluMul(u32, u32, u32),
    GgmlFusedQkvRopeCache(u32, u32, u32, bool),
    GgmlFusedQkvRopePrefill(u32, u32, u32, u32, u32),
    Fp8Gemm(u32, u32, u32),
    Fp8FusedGemmBias(u32, u32, u32),
    Fp8FusedGateUpSiluMul(u32, u32, u32),
    Fp8FusedGateUpGeluMul(u32, u32, u32),
    Fp8FusedQkvRopeCache(u32, u32, u32),
    Fp8FusedQkvRopePrefill(u32, u32, u32, u32, u32),
    /// MLX-affine int4 matmul (transpose=true). Metal-only. The
    /// `LinearLayer` resolved through `WeightAccessors::linear_at`
    /// must be `AffineQuant` — the metal worker reads the quant
    /// accessors (packed weight, scales, per-group affine biases,
    /// optional fp linear bias) through the `WeightTensor::Affine*`
    /// arms.
    ///
    /// Tuple fields: `(in_slot, out_slot, layer, n, k, group_size,
    /// bits, vector_limit)`. Weight comes from
    /// `WeightAccessors::linear_at(bucket, op_idx, slot=0, layer)`.
    /// `vector_limit` is the matvec/matmul boundary from
    /// `get_qmv_batch_limit(K, N, arch_gen)` (mirrors MLX
    /// `quantized.cpp:84`); the macro bakes it at codegen time so
    /// `lower_one` can compare against `bucket_m` without reaching
    /// for the target profile. M < limit → qmv (decode-matvec);
    /// M ≥ limit → qmm_t (prefill-matmul, SplitK heuristic deferred
    /// to C3).
    ///
    /// CUDA eval is `unreachable!` — emit only on the metal forward.
    AffineQmm(u32, u32, u32, u32, u32, u32, u32, u32),
    /// NVIDIA ModelOpt NVFP4 int4 matmul (transpose=true). Metal-only.
    /// Identical shape and dispatch to `AffineQmm` — the only difference
    /// is the in-shader weight decode (E2M1 LUT, no per-group bias). The
    /// `LinearLayer` resolved through `WeightAccessors::linear_at` must be
    /// `Nvfp4`; the metal worker reads the quant accessors (packed E2M1
    /// weight, folded per-group scales, optional fp linear bias) through
    /// the `WeightTensor::Nvfp4*` arms.
    ///
    /// Tuple fields: `(in_slot, out_slot, layer, n, k, group_size, bits,
    /// vector_limit)` — `group_size` is 16, `bits` is 4. `vector_limit` is
    /// the matvec/matmul boundary baked at codegen time (same source as
    /// `AffineQmm`). M < limit → nvfp4_qmv (decode); M ≥ limit →
    /// nvfp4_qmm_t (prefill).
    ///
    /// CUDA eval is `unreachable!` — emit only on the metal forward.
    Nvfp4Qmm(u32, u32, u32, u32, u32, u32, u32, u32),
    /// Compiler-synthesized pre-attention chunk megakernel. Metal-only.
    /// Combines (Add → RmsNorm → 3×AffineQmv → RoPE → paged KV-cache
    /// write) into one dispatch. The kernel itself is generated at
    /// macro-expansion time by
    /// `scratchy-forward-compiler-macro/src/fuse_pass.rs` stitching MK primitive
    /// calls; the symbol name carried here resolves at runtime against
    /// a per-arch source-compiled library registered into the
    /// `SpecializedPipelineCache` at worker-pool init.
    ///
    /// Tuple fields: `(residual_slot, delta_slot, q_out_slot, layer,
    /// group_size, bits, kernel_symbol, has_linear_bias)`. Q/K/V
    /// LinearLayer accessors and the RmsNorm/CosSin pair resolve
    /// through `WeightAccessors::{linear_at, rms_norm_at, cos_sin_at}`
    /// at slots 0..3 of the same `(bucket, op_idx)` — avoiding the
    /// load-time packed-concat infrastructure that Phase 0's revert
    /// dropped.
    ///
    /// CUDA eval is `unreachable!`.
    SynthPreAttn(
        u32, // 0: residual_in_slot
        u32, // 1: delta_slot
        u32, // 2: q_out_slot
        /// 3: `residual_out_slot` — distinct arena slot for the updated
        /// residual (`residual_in + delta`). NOT written in place: the
        /// kernel is tiled across threadgroups and reads the whole
        /// `residual_in` for the rmsnorm sum, so an in-place write would
        /// race cross-threadgroup. The coloring keeps it distinct from
        /// `residual_in_slot`; downstream reads this. For the layer-0
        /// `_init` variant (no residual add) the `fan_out` sets it equal
        /// to `residual_in_slot` and the kernel leaves it untouched.
        u32,
        u32,          // 4: layer
        u32,          // 5: group_size
        u32,          // 6: bits
        &'static str, // 7: kernel_symbol
        /// 8: `has_linear_bias`. When `true`, the lowering arm appends 3
        /// extra `Binding::Weight { which: AffineLinearBias, .. }`
        /// entries for Q/K/V at buffers 18/19/20 and picks the
        /// `_bias` synth kernel symbol (set by `fan_out`). Qwen2/2.5
        /// quantized chains land here; Llama stays `false`.
        bool,
    ),
    /// Compiler-synthesized MLP pre-down chunk megakernel. Metal-only.
    /// Combines (FusedAddRmsNorm → AffineQmv gate → AffineQmv up →
    /// SiluMul) into one dispatch, leaving the down-projection as a
    /// standalone `AffineQmm` consumer of the synthesized output. The
    /// kernel itself is generated at macro-expansion time by
    /// `scratchy-forward-compiler-macro::fuse_pass::synthesize_mlp_pre_down_chunk`;
    /// the symbol name carried here resolves at runtime against a
    /// per-arch source-compiled library registered into the
    /// `SpecializedPipelineCache` at worker-pool init.
    ///
    /// Tuple fields: `(residual_in_slot, delta_slot, silu_mul_out_slot,
    /// residual_out_slot, layer, group_size, bits, kernel_symbol)`.
    /// Gate/up `LinearLayer`s and the RmsNorm resolve through
    /// `WeightAccessors::{linear_at, rms_norm_at}` at slots 0/1/2 of the
    /// same `(bucket, op_idx)`. The standalone `AffineQmm` for down_proj
    /// follows directly in the instruction stream as before.
    ///
    /// `residual_out_slot` is the distinct arena slot the fusion writes
    /// the updated residual (`residual_in + delta`) into — NOT in place,
    /// because the kernel is tiled across threadgroups and reads the
    /// whole `residual_in` for the rmsnorm sum (an in-place write would
    /// race cross-threadgroup). The coloring assigns it a slot distinct
    /// from `residual_in_slot` (fused-subgraph inputs stay live to the
    /// subgraph's end); downstream reads `residual_out_slot`.
    ///
    /// CUDA eval is `unreachable!`.
    SynthMlpPreDown(u32, u32, u32, u32, u32, u32, u32, &'static str),
    /// Fused elementwise `silu(gate) * up` for the decomposed q-MLP
    /// path (plan P12 branch (i)). The macro emits this after a pair
    /// of `AffineQmm` GEMMs when both gate_proj and up_proj are
    /// MLX-affine quantized — `MetalFusedGateUpSiluMulImpl::fan_out`
    /// produces (AffineQmm, AffineQmm, SiluMul) in that case rather
    /// than a single fused `FusedGateUpSiluMul` (which assumes Dense
    /// storage).
    ///
    /// Tuple fields: `(gate_slot, up_slot, out_slot, width)`. Both
    /// inputs are `[M, width]` in the activation dtype; output is the
    /// same shape. `width` is the per-row element count, baked at
    /// macro time from the claim's solved gate/up Gemm N — NOT from a
    /// model-level bound, because one model can carry SwiGLU blocks of
    /// different widths (e.g. Qwen3.5-MoE's 512-wide shared expert
    /// next to a dense `intermediate_size`-wide MLP in hybrid
    /// configs). CUDA eval is `unreachable!` — metal-only (CUDA's
    /// q-MLP routes through Marlin/Bnb/etc).
    SiluMul(u32, u32, u32, u32),
    /// GELU (tanh approx) sibling of [`Instruction::SiluMul`] for the
    /// decomposed GeGLU q-MLP path (Gemma2/3/4:
    /// `gelu_pytorch_tanh(gate) * up`). Same tuple fields
    /// `(gate_slot, up_slot, out_slot)`, same `[M, intermediate_size]`
    /// shapes. Metal-only.
    GeluMul(u32, u32, u32),
    /// Fused gate+up GEMM + SiluMul for large-M prefill (M ≥ 8).
    /// Metal-only; emitted by `MetalSynthGateUpSiluMulImpl`.
    /// Fields: `(x_norm_slot, out_slot, layer, group_size, bits,
    /// kernel_symbol)`. Gate/up LinearLayer accessors resolve through
    /// `WeightAccessors::linear_at` at slots 0/1 of the same
    /// `(bucket, op_idx)`.
    SynthGateUpSiluMul(u32, u32, u32, u32, u32, &'static str),
    /// MLX-affine int4 quantized embedding lookup (Metal-only).
    /// Replaces `Instruction::Embed` when `model.embed_tokens` ships
    /// as a quantized triple `(weight=U32, scales, biases)` — i.e.
    /// every `mlx-community/*-4bit` checkpoint.
    ///
    /// Faithful port of MLX's `nn.QuantizedEmbedding.__call__`
    /// (`python/mlx/nn/layers/quantized.py:144`), fused into one
    /// dispatch via `affine_embed_<dtype>_gs_<gs>_b_4` in
    /// `quantized_dequantize.metal`. Without this lift the embedding
    /// would CPU-dequantize at load (P2 deviation), burning ~2 GB of
    /// arena on Llama-3.2-1B.
    ///
    /// Tuple fields: `(out_slot, group_size, bits)`. The
    /// `AffineQuantEmbedding` weight resolves through a dedicated
    /// `WeightAccessors::affine_quant_embedding_at` accessor (mirrors
    /// `embedding_at` for dense embeds). The hidden_size rides
    /// through `W::Q_SIZE` (function constant baked at lower time).
    /// Metal-only — `AffineQuantEmbedding` is cfg-gated to the metal
    /// backend (mirrors `LinearLayer::AffineQuant`).
    AffineEmbed(u32, u32, u32),
    /// Re-run the next `body_len` instructions `count` times, advancing the LAYER by
    /// `layer_stride` each time.
    ///
    /// ⛔ THE STRIDE IS NOT ALWAYS 1. A body that spans ONE layer advances the layer by one per
    /// iteration, and that was the only case until gemma-4: its layers run `SSSSSG` — five
    /// sliding-window then one global — so the repeating unit is a SIX-layer cell, and iteration
    /// `i` of it starts at layer `6i`. With the stride pinned at 1 the cell could not be rolled
    /// at all; the body is periodic, the layer numbering simply is not.
    ///
    /// Tuple fields: `(count, body_len, layer_stride)`.
    Loop(u32, u32, u32),
    /// `tiles[dst] = Some(View(src))`.
    Alias(u32, u32),
    /// Drop the OwnedTensor at `slot`.
    Free(u32),
}

impl Copy for Instruction {}
impl Clone for Instruction {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

/// How a projection's learned bias is stored — which accessor reads it: a
/// dense `LinearLayer`'s `.bias`, or an MLX-affine layer's `linear_bias`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BiasStorage {
    Dense,
    Affine,
}

/// The additive offset one KV operand carries when its writer caches it.
///
/// A per-vector-norm KV codec (TurboQuant) quantizes each cached vector
/// relative to its own L2 norm, so its error scales with that norm. An operand
/// that is `projection(x) + bias` has a norm the bias can dominate — Qwen2's
/// `k_proj` bias is 7–137x the input-dependent part on its outlier layers — and
/// the error then swamps the part that tells one key from another. The offset
/// is known exactly, so the codec removes it before quantizing and restores it
/// after; the writer declares it so the codec cannot miss it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvOffset {
    /// A projection, or a norm of one: no additive offset.
    Centered,
    /// The producing projection's learned bias. K's is rotated with the key
    /// (RoPE is linear), so at position `t` the offset is `R_t · b`.
    LinearBias(BiasStorage),
}

/// [`KvOffset`] of the K and V operands of one KV writer. No default: whoever
/// emits a writer classifies each operand by its producer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KvOffsets {
    pub k: KvOffset,
    pub v: KvOffset,
}

/// Pack the three routed-expert projection bit-widths (gate / up / down)
/// into the single `bits` field of [`Instruction::MetalSharedFusedMoe`].
///
/// MLX dynamic/mixed quant (Qwen3.5 OptiQ) gives each routed-expert
/// projection its OWN bit-width, and they can differ WITHIN one layer
/// (e.g. layer 1: gate=4, up=8, down=8) — so the metal MoE dispatch needs
/// all three to pick each projection's `affine_gather_qmv_..._b_{bits}`
/// symbol. Each width is 4 or 8 (fits a byte). Uniform / non-OptiQ
/// checkpoints pack the same width three times, so decode is transparent.
#[inline]
pub const fn pack_moe_expert_bits(gate: u32, up: u32, down: u32) -> u32 {
    (gate & 0xff) | ((up & 0xff) << 8) | ((down & 0xff) << 16)
}

/// Inverse of [`pack_moe_expert_bits`] → `(gate, up, down)`.
#[inline]
pub const fn unpack_moe_expert_bits(packed: u32) -> (u32, u32, u32) {
    (packed & 0xff, (packed >> 8) & 0xff, (packed >> 16) & 0xff)
}

/// Per-canonical model parameters. Implemented by each canonical's
/// `Weights` so the universal `Instruction::eval` body can read
/// model constants without storing them on every variant instance.
/// Defaults to 0 / 0.0 / -1 for fields the canonical doesn't use.
pub trait CanonicalParams: WeightAccessors {
    const HEAD_DIM: u32;
    const NUM_Q_HEADS: u32;
    const NUM_KV_HEADS: u32;
    const Q_SIZE: usize;
    const KV_SIZE: usize;
    const INTERMEDIATE_SIZE: usize;
    const ATTN_SCALE: f32;
    const ATTN_SOFTCAP: f32;
    const SLIDING_WINDOW: i32;
    const KV_LORA_RANK: usize;
    const QK_NOPE_HEAD_DIM: usize;
    const QK_ROPE_HEAD_DIM: usize;
    const V_HEAD_DIM: usize;
    const FINAL_LOGIT_SOFTCAPPING: f32;
    /// `qk_nope_head_dim + qk_rope_head_dim` (MlaAttention).
    const QK_HEAD_DIM: usize;
    /// MlaAttention scale: `1/sqrt(qk_head_dim)` w/ YaRN correction.
    const MLA_ATTN_SCALE: f32;
    /// MRoPE (Qwen2-VL / Qwen2.5-VL) section split `[T, H, W]` (rotary
    /// pairs assigned to time / height / width axes; sum equals
    /// `head_dim/2`). `None` for every text-only arch — the rope kernel
    /// takes the legacy 1D-positions fast path. `Some([a,b,c])` selects
    /// the MRoPE path: kernel reads three position values per token
    /// (positions tensor shape `[3, n_tokens]`) and dispatches each
    /// rotary pair through the section that owns it. Default `None` so
    /// existing arches need no override; Qwen2-VL sets it via the
    /// proc-macro emit path. Plan: `~/.claude/plans/distributed-mapping-map.md`.
    const MROPE_SECTION: Option<[u32; 3]> = None;
    /// TurboQuant KV-cache codebook bit-width. Conservative default that
    /// survives outlier-heavy KV (Qwen-class massive activations); arches
    /// validated coherent at 3-bit override this for ~4.7x compression
    /// (Llama family). The metal RuntimeFactory and the lowering pass both
    /// read it (env `SCRATCHY_TQ_BITS` overrides for experimentation), so the
    /// codebook and the baked kernel constants always agree for a given arch.
    const TQ_KV_BITS: u32 = 4;
    /// Vision-tower attention head count. Vision encoders run plain
    /// MHA (`num_kv_heads == num_heads`); only one head dim is needed.
    /// Default 0 for text-only arches that never produce
    /// `OpKind::VarlenAttention` / `OpKind::VisionRope` tiles, so the
    /// matching `Instruction` variants stay registered but unreachable.
    /// Set by `#[vision_forward]` from the vision config.
    const VISION_NUM_HEADS: u32 = 0;

    /// Metal-only: list of compiler-synthesized kernel sources (per
    /// `scratchy-forward-compiler-macro::fuse_pass`). Each entry is
    /// `(symbol_name, precompiled .metallib bytes)`. The proc-macro
    /// AOT-compiles synthesized MSL via `xcrun metal -c` +
    /// `xcrun metallib` at macro-expansion time and embeds the
    /// resulting bytes as `&'static [u8]`. The MetalWorkerPool
    /// registers each via
    /// `SpecializedPipelineCache::register_metallib_library`
    /// (`newLibraryWithData`) — same path used by every hand-written
    /// shader, NOT `newLibraryWithSource`. Default empty — the macro
    /// overrides this per Metal arch with the actual synthesized
    /// metallibs from the FUF analysis.
    fn synthesized_kernel_metallibs() -> &'static [(&'static str, &'static [u8])] {
        &[]
    }
    /// Vision-tower attention head dimension. Same defaults / set-by
    /// rule as [`Self::VISION_NUM_HEADS`].
    const VISION_HEAD_DIM: u32 = 0;
    /// `vision_num_heads * vision_head_dim` — the rank-2 last-dim of
    /// q/k/v at the FUF level (the kernel internally reshapes to
    /// rank-3). Same defaults / set-by rule as
    /// [`Self::VISION_NUM_HEADS`].
    const VISION_Q_SIZE: usize = 0;
    /// Vision patch-embed input width = `in_chans * temporal_patch_size
    /// * patch_size²` (Qwen3.5-VL: 3·2·16² = 1536). The rank-2 last-dim
    /// of the `pixels` extern; the metal `LoadPixels` arm uses it to
    /// size the copy-into-arena dispatch. Same defaults / set-by rule as
    /// [`Self::VISION_NUM_HEADS`] (0 on non-vision arches).
    const VISION_IN_FEATURES: usize = 0;
    /// Vision 2D-RoPE pairing convention: `false` = GPT-NeoX
    /// rotate_half (every Qwen-VL tower), `true` = adjacent-pair
    /// GPT-J style (MoonViT / LocateAnything). Set from the optional
    /// `vision_rope_style` config key ("interleaved_xy"); selects the
    /// `vision_rope_2d[_interleaved]` kernel entry point at lowering.
    const VISION_ROPE_INTERLEAVED: bool = false;
    /// Vision-tower softmax scale: `1 / sqrt(vision_head_dim)`. Same
    /// defaults / set-by rule as [`Self::VISION_NUM_HEADS`].
    const VISION_ATTN_SCALE: f32 = 0.0;
    /// Patch-grid side length (square): `vision_image_size / vision_patch_size`.
    /// Used by `Instruction::AvgPool2d` to convert flat row index
    /// `[L = ph * ph]` → `(row, col)` and walk the k² source cells per
    /// output. Defaults to 0 for arches that never emit `OpKind::AvgPool2d`.
    const VISION_PATCH_GRID_SIDE: u32 = 0;
    /// Stride / kernel size of the post-encoder average pool (Gemma3-MM
    /// projector: k=4 over a 64×64 patch grid → 16×16 = 256 tokens). The
    /// pool is non-overlapping, so stride == kernel. Defaults to 0.
    const VISION_POOL_KERNEL: u32 = 0;

    /// RmsNorm epsilon — read from `rms_norm_eps` in the model
    /// config at macro-expand time. The metal `rmsnorm_*_specialized`
    /// kernels consume this via `[[function_constant]]` baked into
    /// the compiled pipeline; cuda's interpreter still reads it
    /// from `RmsNorm.eps` on the loaded layer struct (same value,
    /// same source). Default is the value Llama / Qwen / Phi
    /// canonically use; per-canonical macro impls override.
    const RMS_NORM_EPS: f32 = 1e-5;

    /// Zero-centered RMSNorm gain offset (Gemma / Qwen3.5): the effective
    /// gain applied by the standard `RmsNorm` / `FusedAddRmsNorm` metal
    /// kernels is `weight + NORM_WEIGHT_OFFSET`. `1.0` for `(1 + weight)`
    /// arches whose norm gains are stored zero-centered (Gemma2/3,
    /// Qwen3.5 — input/post/final layernorms + per-head q/k norm), `0.0`
    /// for plain RMSNorm (Llama / Qwen2 / Qwen3). The GDN gated RMSNorm
    /// (`gdn_rms_norm_gated`) is a separate kernel and is NOT affected —
    /// its gain is stored as the true value. Per-canonical macro impls
    /// override from the config's `norm_weight_offset` field.
    const NORM_WEIGHT_OFFSET: f32 = 0.0;

    /// Paged-KV-cache block stride (the `block_size` function
    /// constant `attention_via_cache_*_specialized` and
    /// `rope_append_*_specialized` consume). Backend-fixed at 16
    /// (vLLM's default); per-canonical override only if a model
    /// chooses a different paging size. This is the SLIDING-class block
    /// size on hybrid arches (see [`Self::GLOBAL_BLOCK_SIZE`]).
    const BLOCK_SIZE: u32 = 16;

    /// Paged-KV-cache block stride for the GLOBAL (full-attention) layer
    /// class on hybrid sliding/global arches. vLLM's group-shared KV
    /// layout PAGE-UNIFIES the two classes: the full class has a smaller
    /// per-block byte footprint (`num_global_kv_heads · head_dim` vs the
    /// sliding class's `num_kv_heads · head_dim`), so its block_size is
    /// scaled UP so both classes' blocks occupy the same bytes and can
    /// share one physical KV tensor. Gemma4: sliding `8·256 = 2048` vs
    /// global `2·512 = 1024` per token → global block_size `2 × 16 = 32`
    /// makes both blocks `32768` elems. Defaults to `BLOCK_SIZE` →
    /// uniform-geometry models compile unchanged. The metal lowering
    /// feeds this to the full-attention + full-layer KV-write arms.
    const GLOBAL_BLOCK_SIZE: u32 = Self::BLOCK_SIZE;

    // ── Gated-DeltaNet (Qwen3.5 / Qwen3-Next linear attention) ──────
    // Per-layer GDN dims the `Instruction::GatedDeltaNet` eval reads as
    // `W::GDN_*`. All default 0 so non-hybrid arches need no override;
    // the proc-macro emits overrides for GDN arches from the
    // `linear_num_key_heads` / `linear_num_value_heads` /
    // `linear_key_head_dim` / `linear_value_head_dim` /
    // `linear_conv_kernel_dim` config keys. `GDN_CONV_DIM` =
    // `2·(num_k_heads·head_k_dim) + num_v_heads·head_v_dim`.
    const GDN_NUM_K_HEADS: u32 = 0;
    const GDN_NUM_V_HEADS: u32 = 0;
    const GDN_HEAD_K_DIM: u32 = 0;
    const GDN_HEAD_V_DIM: u32 = 0;
    const GDN_CONV_KERNEL: u32 = 0;
    const GDN_CONV_DIM: usize = 0;

    /// Block-table row stride (in u32s), equal to
    /// `ceil(MAX_SEQ_LEN / BLOCK_SIZE)`. Baked into
    /// `attention_via_cache_*_specialized` so the kernel can index
    /// `block_table[seq * MAX_BLOCKS_PER_SEQ + logical_block]`
    /// without a runtime divide. Default sized for ~2k tokens; per-
    /// canonical macro impls override for longer-context models.
    const MAX_BLOCKS_PER_SEQ: u32 = 128;

    /// Q-axis tile size for the cuda contiguous-prefill kernel — the
    /// kernel processes this many query tokens per threadgroup.
    /// Metal post-Phase B always emits `Instruction::AttentionPrefillPaged`
    /// (1 Q per TG via `sdpa_vector` port) so this constant is
    /// cuda-only; backend-fixed and tuning requires kernel co-evolution.
    const PREFILL_TILE_Q: u32 = 16;

    /// Partial-rope rotation dim — for models where only the first
    /// `ROT_DIM` of `HEAD_DIM` get rotary applied (Qwen2-VL, GPT-J).
    /// Default equals `HEAD_DIM` (full rope, the common case).
    const ROT_DIM: u32 = Self::HEAD_DIM;

    /// Per-layer-class attention geometry for hybrid sliding/global
    /// architectures whose two classes differ in dims (Gemma4: sliding
    /// layers `head_dim 256 × 8 kv heads`, global layers `head_dim 512
    /// × 1 kv head`). The DSL's `attention()` (global) vs
    /// `sliding_attention()` (local) tiles lower to distinct
    /// Instruction variants, so the metal lowering feeds the GLOBAL_*
    /// family to the full-attention arms and the base `HEAD_DIM` /
    /// `NUM_KV_HEADS` to the sliding arms. Defaults equal the base
    /// values — uniform-geometry models compile unchanged. Sourced
    /// from config `global_head_dim` / `num_global_key_value_heads`.
    const GLOBAL_HEAD_DIM: u32 = Self::HEAD_DIM;
    /// See [`Self::GLOBAL_HEAD_DIM`].
    const NUM_GLOBAL_KV_HEADS: u32 = Self::NUM_KV_HEADS;
    /// Partial-rope rotation dim for the GLOBAL class (Gemma4:
    /// proportional rope rotates 128 of 512). Defaults to the global
    /// head_dim (full rope).
    const GLOBAL_ROT_DIM: u32 = Self::GLOBAL_HEAD_DIM;
    /// `num_q_heads * global_head_dim` — the global-class Q/attn-out
    /// row width (Gemma4: 16×512 = 8192 vs sliding 16×256 = 4096).
    const GLOBAL_Q_SIZE: usize = Self::Q_SIZE;
    /// Gemma4 "proportional" partial rope on the GLOBAL class (config
    /// `global_partial_rotary_factor`): rotation pairs span the FULL
    /// head's halves — lane `i < GLOBAL_ROT_DIM/2` pairs with `i +
    /// GLOBAL_HEAD_DIM/2` (mlx `ProportionalRoPE`), NOT `i +
    /// GLOBAL_ROT_DIM/2` like standard HF partial rotary (Qwen3.5).
    /// The metal RopeAppend arm keys the kernel's pairing offset on
    /// this. False everywhere else.
    const ROPE_PROPORTIONAL: bool = false;

    /// Spans / position-independent KV caching capability. When true,
    /// `BlockKind::Relocatable` (span) blocks are stored with K
    /// UNROTATED by `rope_append`, and the paged attention kernels
    /// re-rope cached K to each reader's position on read — letting one
    /// cached span be reused at any position (zero-copy block sharing).
    /// A compile-time CAPABILITY: when set, the metal lowering binds the
    /// per-block unrotated-flag buffer + cos_sin into the attention /
    /// rope_append kernels and emits the rope-on-read function constants.
    /// Non-span requests carry an all-zero flag buffer → the per-block
    /// rotation branch is never taken (parity). When false (default),
    /// the lowering is byte-identical to the no-spans path (the rope-on-
    /// read constants/buffers are omitted and dead-eliminate in-shader).
    /// Set true only for models/deployments that opt into spans.
    const ROPE_ON_READ: bool = false;

    /// Element dtype the metal backend should run this canonical in.
    /// Picks between the `_f16_specialized` / `_bf16_specialized`
    /// shader symbols and matching MPS GEMM data type. Default
    /// `MetalDtype::Bf16` matches every modern HF Llama / Qwen /
    /// Phi / Mistral checkpoint (`torch_dtype: bfloat16` on disk)
    /// and the cuda backend's native dtype. Per-canonical macro
    /// overrides set this from the model config's `torch_dtype`.
    ///
    /// Pre-bf16-rollout default was `F16`; that path lost exponent
    /// range over deep layer chains and produced incoherent outputs
    /// on Llama-3.x. The default is `Bf16` now; arches that ship
    /// fp16 on disk (rare) override.
    const METAL_DTYPE: MetalDtype = MetalDtype::Bf16;

    /// `hidden_size` (residual-stream width) — distinct from
    /// `Q_SIZE = num_q_heads * head_dim`. On Llama / Qwen2.5 these
    /// happen to be equal because each Q head is `hidden_size /
    /// num_heads`. On Qwen3 (and any GQA / different head_dim
    /// arch) they differ: e.g. Qwen3-30B-A3B has hidden=2048,
    /// num_q=32, head_dim=128 → Q_SIZE=4096. The metal interpreter
    /// previously baked `W::Q_SIZE` as `AFFINE_EMBED_HIDDEN_SIZE`
    /// and `RMSNORM_HIDDEN_SIZE`, which broke Qwen3 silently. The
    /// macro emits this const from `model.bounds["hidden_size"]`.
    /// Default value is invalid (0) so unset arches surface as
    /// compile-time-detectable bad values rather than silent
    /// corruption.
    const HIDDEN_SIZE: usize = 0;

    /// Vocabulary size — the lm_head Gemm's output dim. Used by the
    /// generic `Instruction::Gemm` arm to identify the lm_head call
    /// site (so it can apply the last-token-per-seq narrow before the
    /// GEMM). The previous heuristic `n > INTERMEDIATE_SIZE` mis-fires
    /// on MoE arches whose `INTERMEDIATE_SIZE` is `moe_intermediate_size`
    /// (Qwen3.5-MoE: 512), making q_proj / linear_attn.in_proj_qkv
    /// (n=8192) look like an lm_head and narrowing their input to one
    /// token at multi-token prefill → garbage activations → NaN logits.
    /// `n == VOCAB_SIZE` is the unambiguous test. Default 0 so non-MoE
    /// arches still hit the legacy `INTERMEDIATE_SIZE` fallback.
    const VOCAB_SIZE: usize = 0;

    /// On-disk storage dtype for `*.scales` / `*.biases` tensors on
    /// mlx-affine-b4 checkpoints. The affine quant Metal kernels
    /// (`affine_qmv`, `affine_qmm_t`, `affine_qvm`, `affine_gather_qmv`,
    /// `affine_embed`) read these as `T_scale` and cast to `T_act` in-
    /// register. MLX templates `T_scale ∈ {half, bfloat16_t}` natively
    /// (`mlx/.../quantized.h INSTANTIATE_QUANTIZED_FUNCTIONS`);
    /// scratchy-target-metal mirrors that surface, picking the right symbol
    /// based on this const.
    ///
    /// `mlx-community` convention varies by arch family (probed across
    /// the cached HF snapshots): Llama-3.x / Qwen2.5 / SmolLM → F16,
    /// Qwen3 family → BF16. Default `F16` matches the
    /// majority-of-checkpoints convention; per-arch macro overrides
    /// flip it to BF16 (Qwen3 family). Sourced from the manifest's
    /// `scale_dtype` field on the matched `mlx-affine-*` preset.
    const SCALE_DTYPE: ScaleDtype = ScaleDtype::F16;

    // ── Backend-capability flags ────────────────────────────────
    //
    // Each `HAS_*` flag tells `BackendCompat`
    // whether the arch's DSL emits a tile that requires a particular
    // backend-side `Impl`. The macro derives these by walking the
    // classified `Program` at expansion time — they're never
    // hand-set per canonical, and never read at runtime.
    //
    // Flags exist to make a missing-Impl-on-backend bug a compile
    // error instead of silent garbage output. See
    // `crates/scratchy-forward-compiler/src/backend_compat.rs`.

    /// True iff the arch's DSL body emits `bias_add(...)` tile(s)
    /// (Qwen-family QKV biases, ModernBERT-style LayerNorm post-add,
    /// etc.). Set by the macro via DSL inspection; never written
    /// by hand.
    ///
    /// On CUDA the `(Gemm, BiasAdd)` and `(AffineQmm, BiasAdd)`
    /// fusions absorb every shipping pattern; on Metal the matching
    /// fused Impls (`MetalBiasAddImpl::matches() → None`,
    /// `FusedGemmBiasImpl` Affine-storage gate, missing
    /// `FusedAffineQkvRopeCacheWithBias`) leave bias_add tiles
    /// unclaimed, so a `true` value blocks
    /// `BackendCompat<Metal>` at compile time.
    const HAS_BIAS_ADD: bool = false;

    /// True iff the arch's DSL emits `OpKind::Moe` (router softmax,
    /// top-k, experts gather, and SwitchGLU bundled into one tile).
    /// MoE-on-Metal isn't landed yet — the router prereqs (softmax,
    /// argpartition, take_along_axis) are ported, but `gather_qmm_rhs`
    /// and the pure-ICB SwitchGLU decomposition are open. MoE arches block
    /// `BackendCompat<Metal>` until those land.
    const HAS_MOE: bool = false;
}
