// SPDX-License-Identifier: Apache-2.0
//! Hashable view of an [`Instruction`] for backbone introspection
//! (`scr model info`).
//!
//! `Instruction` lives behind `cfg(feature = "cuda")`. None of that
//! is useful when the consumer wants to:
//! - tell two backbones apart for fusion-hunting
//! - hash a backbone for equivalence-class grouping
//! - print a backbone as ASCII without needing a live `&Weights`
//!
//! [`Instruction::normalize`] runs once per row to produce a
//! [`NormalizedStep`]: the variant tag as `&'static str`, plus a
//! flat field list with semantic kinds (slot / layer / const /
//! kernel-class). The kinds and counts of weight slots a variant
//! consumes — historically encoded as `WtFn<W, L>` / `CosSinFn<W>`
//! variant fields, now resolved at the tape level via
//! [`crate::WeightAccessors`] — collapse here to
//! `LayerKind(L_name)` / `RopeCosSin` markers.
//!
//! The `match` is closed (no `_` arm). Adding a new
//! [`Instruction`] variant fails the build until normalize is
//! taught the new arm — same closed-emitter invariant the
//! interpreter `eval` already enforces.

use crate::Instruction;

/// One backbone step in non-generic form. Equivalent to one row of a
/// `BACKBONE_M_<…>` / `LM_HEAD_M_<…>` static slice with the W
/// dependence projected out.
///
/// `Hash` / `Eq` ignore nothing — every produced field participates.
/// Consumers that want a coarser equivalence (e.g. `body` —
/// alpha-renamed slots, dropped loop count) build that as a
/// transformation on top of `Vec<NormalizedStep>`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NormalizedStep {
    /// PascalCase variant ident — e.g. `"FusedAddRmsNorm"`.
    pub kind: &'static str,
    /// Field values in declaration order (matches the
    /// [`Instruction`] tuple-variant order).
    pub fields: Vec<NormalizedField>,
}

/// Kinded view of a single [`Instruction`] field.
///
/// `f32` consts stored as `u32` bits so this enum stays `Eq + Hash`
/// — the renderer reconstructs via `f32::from_bits`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NormalizedField {
    /// Tile-table slot index. Subject to alpha-renaming under the
    /// `body` equivalence class.
    Slot(u32),
    /// Layer index. Inside a `Loop` body these are *baseline* values
    /// (the per-iter index is added by `InterpreterCtx::layer_offset`
    /// at dispatch).
    Layer(u32),
    /// Plain integer const — Cutlass tile dims, FlashInfer head_dim,
    /// Reshape ndim, etc.
    ConstU32(u32),
    /// `f32` const stored as bit pattern for `Eq + Hash`.
    ConstF32Bits(u32),
    /// Boolean flag — `interleaved`, `biased`,
    /// `use_logits_soft_cap`, …
    ConstBool(bool),
    /// The additive offset a KV writer's K and V carry.
    KvOffsets(crate::KvOffsets),
    /// Fixed-length integer array, truncated to its valid prefix
    /// (`Reshape::dims_lit[..ndim]`).
    ConstU32Array(Vec<u32>),
    /// Same, but the underlying field type was `[u8; N]`.
    ConstU8Array(Vec<u8>),
    /// One weight slot's kernel-class name (`L`'s type name) —
    /// e.g. `"RmsNorm"`, `"LinearLayer"`, `"MarlinLinear"`,
    /// `"Bnb4bitLinear"`, `"Fp8AnyLinear"`, `"CohereLayerNorm"`,
    /// `"Embedding"`, `"DeepSeekV2MoELayer"`. Replaces the historical
    /// `WtFn<W, L>` field projection — multi-accessor variants emit
    /// one `LayerKind` per slot in declaration order.
    LayerKind(&'static str),
    /// Static weight matrix shape for a GEMM-class step, captured at
    /// codegen time from the FUF's `eval_shape`. `n` is output dim
    /// (LinearLayer's out_features), `k` is reduction dim
    /// (in_features). M is workload-dependent — the dump consumer
    /// derives it from the bucket's `m_min..m_max_excl`.
    ///
    /// Emitted by simple-GEMM Instruction variants only (one matmul,
    /// one weight) — `Gemm`, `CutlassGemm`, `CutlassGemmSplitK`,
    /// `CutlassGemmAdd`, `CutlassGemv`. Fused QKV / GateUp variants
    /// have multiple participating shapes and don't emit this.
    WeightShape { n: u32, k: u32 },
    /// Rotary cos/sin table accessor marker. Carries no other
    /// identity at this layer — the per-arch `WeightAccessors`
    /// `cos_sin_at` resolves it from the tape position; different
    /// rope tables come from per-variant `CanonicalParams` consts,
    /// not from this slice.
    RopeCosSin,
    /// `Loop(count, body_len)` — count of iterations.
    LoopCount(u32),
    /// `Loop(count, body_len)` — number of subsequent rows that form
    /// the body.
    LoopBodyLen(u32),
    /// `Loop(.., layer_stride)` — how far the LAYER advances per iteration.
    ///
    /// ⛔ IT HAS TO BE HERE. This is the instrument the roll is PROVEN with: the rolled emission
    /// is expanded and compared row-for-row against the un-rolled one. A stride left out of the
    /// normalized form is a difference the proof cannot see, so a wrong stride would compare
    /// equal and ship.
    LoopLayerStride(u32),
}

/// Local extension trait carrying [`Instruction::normalize`]. Lives
/// here because `Instruction` moved to the foreign `scratchy-ir`
/// crate — an inherent `impl` would be an orphan-rule violation
/// (E0116). A local-trait-for-foreign-type impl is orphan-legal.
pub trait InstructionInfo {
    /// Project this row to a [`NormalizedStep`]. Pure data — no
    /// `&W` needed, no GPU work, no allocator beyond the field
    /// `Vec`.
    ///
    /// The match is closed (no catch-all). Every `Instruction`
    /// variant has exactly one arm.
    #[must_use]
    fn normalize(&self) -> NormalizedStep;
}

impl InstructionInfo for Instruction {
    fn normalize(&self) -> NormalizedStep {
        use NormalizedField as F;
        let (kind, fields): (&'static str, Vec<NormalizedField>) = match *self {
            Instruction::Embed(out_slot) => {
                ("Embed", vec![F::Slot(out_slot), F::LayerKind("Embedding")])
            }
            Instruction::RmsNorm(in_slot, out_slot, layer, _hidden_size, _m_mult) => (
                "RmsNorm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("RmsNorm"),
                ],
            ),
            Instruction::MeanSubRmsNorm(in_slot, out_slot, layer) => (
                "MeanSubRmsNorm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("RmsNorm"),
                ],
            ),
            Instruction::MeanSubRmsNormBiasAdd(in_slot, out_slot, layer) => (
                "MeanSubRmsNormBiasAdd",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LayerNorm"),
                ],
            ),
            Instruction::Reshape(in_slot, out_slot, dims_lit, dims_nt_pow, dims_div_lit, ndim) => {
                let n = ndim as usize;
                (
                    "Reshape",
                    vec![
                        F::Slot(in_slot),
                        F::Slot(out_slot),
                        F::ConstU32Array(dims_lit[..n].to_vec()),
                        F::ConstU8Array(dims_nt_pow[..n].to_vec()),
                        F::ConstU32Array(dims_div_lit[..n].to_vec()),
                        F::ConstU32(u32::from(ndim)),
                    ],
                )
            }
            Instruction::Add(delta_slot, residual_slot) => {
                ("Add", vec![F::Slot(delta_slot), F::Slot(residual_slot)])
            }
            #[cfg(feature = "nccl")]
            Instruction::AllReduce(slot) => ("AllReduce", vec![F::Slot(slot)]),
            #[cfg(feature = "nccl")]
            Instruction::AllGather(in_slot, out_slot) => {
                ("AllGather", vec![F::Slot(in_slot), F::Slot(out_slot)])
            }
            Instruction::ScalarMul(in_slot, out_slot, scale) => (
                "ScalarMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::ConstF32Bits(scale.to_bits()),
                ],
            ),
            Instruction::TanhSoftCap(in_slot, out_slot) => {
                ("TanhSoftCap", vec![F::Slot(in_slot), F::Slot(out_slot)])
            }
            Instruction::RmsNormUnit(in_slot, out_slot, hidden, m_mult) => (
                "RmsNormUnit",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::ConstU32(hidden),
                    F::ConstU32(m_mult),
                ],
            ),
            Instruction::ScalarWeightMul(in_slot, out_slot, layer) => (
                "ScalarWeightMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("RmsNorm"),
                ],
            ),
            Instruction::RopeAppendNormed(
                q_slot,
                k_slot,
                v_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                _interleaved,
                is_global,
            ) => (
                "RopeAppendNormed",
                vec![
                    F::Slot(q_slot),
                    F::Slot(k_slot),
                    F::Slot(v_slot),
                    F::Slot(q_out_slot),
                    F::Slot(k_out_slot),
                    F::Slot(v_out_slot),
                    F::Layer(layer),
                    F::ConstU32(is_global as u32),
                    F::LayerKind("RmsNorm"),
                ],
            ),
            Instruction::NormAddScalarMul(delta_slot, residual_slot, out_slot, layer, hidden) => (
                "NormAddScalarMul",
                vec![
                    F::Slot(delta_slot),
                    F::Slot(residual_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::ConstU32(hidden),
                    F::LayerKind("RmsNorm"),
                ],
            ),
            Instruction::FusedAddRmsNorm(in_slot, out_slot, layer, _hidden_size, _m_mult) => (
                "FusedAddRmsNorm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("RmsNorm"),
                ],
            ),
            Instruction::FusedAddRmsNormWithOffset(in_slot, out_slot, layer, offset) => (
                "FusedAddRmsNormWithOffset",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::ConstF32Bits(offset.to_bits()),
                    F::LayerKind("RmsNorm"),
                ],
            ),
            Instruction::ScalarOffsetRmsNorm(
                in_slot,
                out_slot,
                layer,
                offset,
                hidden_size,
                m_multiplier,
            ) => (
                "ScalarOffsetRmsNorm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::ConstF32Bits(offset.to_bits()),
                    F::ConstU32(hidden_size),
                    F::ConstU32(m_multiplier),
                    F::LayerKind("RmsNorm"),
                ],
            ),
            Instruction::CutlassFusedRmsNormGemm(
                in_slot,
                out_slot,
                layer,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => (
                "CutlassFusedRmsNormGemm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("RmsNorm"),
                    F::LayerKind("LinearLayer"),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                    F::WeightShape { n, k },
                ],
            ),
            Instruction::CutlassFusedMeanSubRmsNormGemm(
                in_slot,
                out_slot,
                layer,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => (
                "CutlassFusedMeanSubRmsNormGemm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("RmsNorm"),
                    F::LayerKind("LinearLayer"),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                    F::WeightShape { n, k },
                ],
            ),
            Instruction::CutlassFusedAddRmsNormGemm(
                delta_slot,
                residual_slot,
                out_slot,
                layer,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => (
                "CutlassFusedAddRmsNormGemm",
                vec![
                    F::Slot(delta_slot),
                    F::Slot(residual_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("RmsNorm"),
                    F::LayerKind("LinearLayer"),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                    F::WeightShape { n, k },
                ],
            ),
            Instruction::Gemm(in_slot, out_slot, layer, n, k) => (
                "Gemm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::WeightShape { n, k },
                ],
            ),
            Instruction::FusedCublasGemmAdd(in_slot, residual_slot, layer, n, k) => (
                "FusedCublasGemmAdd",
                vec![
                    F::Slot(in_slot),
                    F::Slot(residual_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::WeightShape { n, k },
                ],
            ),
            Instruction::FusedGemmBias(in_slot, out_slot, layer) => (
                "FusedGemmBias",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::MetalBiasAdd(in_slot, out_slot, layer, _n, _is_affine) => (
                "MetalBiasAdd",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::FusedGateUpSiluMul(in_slot, out_slot, layer) => (
                "FusedGateUpSiluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::FusedGateUpGeluMul(in_slot, out_slot, layer) => (
                "FusedGateUpGeluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::FusedQkvRopeCache(in_slot, out_slot, layer, biased, interleaved) => (
                "FusedQkvRopeCache",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::RopeCosSin,
                    F::ConstBool(biased),
                    F::ConstBool(interleaved),
                ],
            ),
            Instruction::FusedQkvQkNormRopeCache(in_slot, out_slot, layer, q_offset, k_offset) => (
                "FusedQkvQkNormRopeCache",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::LayerKind("LinearLayer"),
                    F::LayerKind("LinearLayer"),
                    F::LayerKind("RmsNorm"),
                    F::LayerKind("RmsNorm"),
                    F::RopeCosSin,
                    F::ConstF32Bits(q_offset.to_bits()),
                    F::ConstF32Bits(k_offset.to_bits()),
                ],
            ),
            Instruction::FusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                biased,
                interleaved,
            ) => (
                "FusedQkvRopePrefill",
                vec![
                    F::Slot(in_slot),
                    F::Slot(q_out_slot),
                    F::Slot(k_out_slot),
                    F::Slot(v_out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::RopeCosSin,
                    F::ConstBool(biased),
                    F::ConstBool(interleaved),
                ],
            ),
            Instruction::AttentionViaCache(in_slot, out_slot, layer, interleaved) => (
                "AttentionViaCache",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::RopeCosSin,
                    F::ConstBool(interleaved),
                ],
            ),
            Instruction::AttentionPrefillContiguous(
                q_slot,
                k_slot,
                v_slot,
                out_slot,
                interleaved,
            ) => (
                "AttentionPrefillContiguous",
                vec![
                    F::Slot(q_slot),
                    F::Slot(k_slot),
                    F::Slot(v_slot),
                    F::Slot(out_slot),
                    F::ConstBool(interleaved),
                ],
            ),
            Instruction::AttentionPrefillPaged(q_slot, out_slot, layer, interleaved) => (
                "AttentionPrefillPaged",
                vec![
                    F::Slot(q_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::ConstBool(interleaved),
                ],
            ),
            Instruction::SlidingAttentionPrefillPaged(q_slot, out_slot, layer, interleaved) => (
                "SlidingAttentionPrefillPaged",
                vec![
                    F::Slot(q_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::ConstBool(interleaved),
                ],
            ),
            Instruction::EncoderAttention(q_slot, k_slot, v_slot, out_slot) => (
                "EncoderAttention",
                vec![
                    F::Slot(q_slot),
                    F::Slot(k_slot),
                    F::Slot(v_slot),
                    F::Slot(out_slot),
                ],
            ),
            Instruction::SlidingAttentionViaCache(in_slot, out_slot, layer, interleaved) => (
                "SlidingAttentionViaCache",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::RopeCosSin,
                    F::ConstBool(interleaved),
                ],
            ),
            Instruction::SlidingAttentionPrefillContiguous(
                q_slot,
                k_slot,
                v_slot,
                out_slot,
                interleaved,
            ) => (
                "SlidingAttentionPrefillContiguous",
                vec![
                    F::Slot(q_slot),
                    F::Slot(k_slot),
                    F::Slot(v_slot),
                    F::Slot(out_slot),
                    F::ConstBool(interleaved),
                ],
            ),
            Instruction::VarlenAttention(q_slot, k_slot, v_slot, out_slot, cu_seqlens_kind) => (
                "VarlenAttention",
                vec![
                    F::Slot(q_slot),
                    F::Slot(k_slot),
                    F::Slot(v_slot),
                    F::Slot(out_slot),
                    F::ConstU32(u32::from(cu_seqlens_kind)),
                ],
            ),
            Instruction::VisionRope(q_slot, k_slot, q_out_slot, k_out_slot) => (
                "VisionRope",
                vec![
                    F::Slot(q_slot),
                    F::Slot(k_slot),
                    F::Slot(q_out_slot),
                    F::Slot(k_out_slot),
                ],
            ),
            Instruction::QuickGelu(in_slot, out_slot) => {
                ("QuickGelu", vec![F::Slot(in_slot), F::Slot(out_slot)])
            }
            Instruction::GeluErf(in_slot, out_slot) => {
                ("GeluErf", vec![F::Slot(in_slot), F::Slot(out_slot)])
            }
            Instruction::Gelu(in_slot, out_slot) => {
                ("Gelu", vec![F::Slot(in_slot), F::Slot(out_slot)])
            }
            Instruction::PosEmbed(out_slot) => (
                "PosEmbed",
                vec![F::Slot(out_slot), F::LayerKind("Embedding")],
            ),
            Instruction::LoadPixels(out_slot) => ("LoadPixels", vec![F::Slot(out_slot)]),
            Instruction::LoadPosEmbeds(out_slot) => ("LoadPosEmbeds", vec![F::Slot(out_slot)]),
            Instruction::EmbeddingGather(in_slot, out_slot, indices_kind) => (
                "EmbeddingGather",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::ConstU32(u32::from(indices_kind)),
                ],
            ),
            Instruction::AvgPool2d(in_slot, out_slot) => {
                ("AvgPool2d", vec![F::Slot(in_slot), F::Slot(out_slot)])
            }
            Instruction::StripCls(in_slot, out_slot) => {
                ("StripCls", vec![F::Slot(in_slot), F::Slot(out_slot)])
            }
            Instruction::FlashInferAttentionDecode(
                in_slot,
                out_slot,
                layer,
                head_dim,
                use_logits_soft_cap,
            ) => (
                "FlashInferAttentionDecode",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::RopeCosSin,
                    F::ConstU32(head_dim),
                    F::ConstBool(use_logits_soft_cap),
                ],
            ),
            #[cfg(fa3_built)]
            Instruction::FlashAttention3Decode(in_slot, out_slot, layer, head_dim) => (
                "FlashAttention3Decode",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::ConstU32(head_dim),
                ],
            ),
            Instruction::FlashInferAttentionPrefill(
                q_slot,
                k_slot,
                v_slot,
                out_slot,
                layer,
                head_dim,
                use_logits_soft_cap,
            ) => (
                "FlashInferAttentionPrefill",
                vec![
                    F::Slot(q_slot),
                    F::Slot(k_slot),
                    F::Slot(v_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::ConstU32(head_dim),
                    F::ConstBool(use_logits_soft_cap),
                ],
            ),
            Instruction::RopeAppend(
                q_slot,
                k_slot,
                v_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                interleaved,
                is_global,
                kv_offsets,
            ) => (
                "RopeAppend",
                vec![
                    F::Slot(q_slot),
                    F::Slot(k_slot),
                    F::Slot(v_slot),
                    F::Slot(q_out_slot),
                    F::Slot(k_out_slot),
                    F::Slot(v_out_slot),
                    F::Layer(layer),
                    F::RopeCosSin,
                    F::ConstBool(interleaved),
                    F::ConstBool(is_global),
                    F::KvOffsets(kv_offsets),
                ],
            ),
            Instruction::MlaSplit(in_slot, kv_latent_slot, k_pe_slot) => (
                "MlaSplit",
                vec![
                    F::Slot(in_slot),
                    F::Slot(kv_latent_slot),
                    F::Slot(k_pe_slot),
                ],
            ),
            Instruction::MlaAttention(q_slot, kv_b_slot, k_pe_slot, out_slot, layer) => (
                "MlaAttention",
                vec![
                    F::Slot(q_slot),
                    F::Slot(kv_b_slot),
                    F::Slot(k_pe_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::RopeCosSin,
                ],
            ),
            Instruction::GatedDeltaNet(qkv_slot, z_slot, a_slot, b_slot, out_slot, layer) => (
                "GatedDeltaNet",
                vec![
                    F::Slot(qkv_slot),
                    F::Slot(z_slot),
                    F::Slot(a_slot),
                    F::Slot(b_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                ],
            ),
            Instruction::GateSplit(qg_slot, q_slot, gate_slot) => (
                "GateSplit",
                vec![F::Slot(qg_slot), F::Slot(q_slot), F::Slot(gate_slot)],
            ),
            Instruction::GateApply(attn_slot, gate_slot, out_slot) => (
                "GateApply",
                vec![F::Slot(attn_slot), F::Slot(gate_slot), F::Slot(out_slot)],
            ),
            Instruction::GateScale(routed_slot, shared_slot, gate_slot, out_slot) => (
                "GateScale",
                vec![
                    F::Slot(routed_slot),
                    F::Slot(shared_slot),
                    F::Slot(gate_slot),
                    F::Slot(out_slot),
                ],
            ),
            Instruction::DeepSeekMoe(in_slot, out_slot, layer) => (
                "DeepSeekMoe",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("DeepSeekV2MoELayer"),
                ],
            ),
            Instruction::DeepSeekMoeFp8Block(in_slot, out_slot, layer) => (
                "DeepSeekMoeFp8Block",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("DeepSeekV2Fp8BlockMoELayer"),
                ],
            ),
            Instruction::DeepSeekMoeGgml(in_slot, out_slot, layer) => (
                "DeepSeekMoeGgml",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("DeepSeekV2GgmlMoELayer"),
                ],
            ),
            Instruction::FusedMoe(in_slot, out_slot, layer) => (
                "FusedMoe",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("FusedMoELayer"),
                ],
            ),
            Instruction::SharedFusedMoe(in_slot, out_slot, layer) => (
                "SharedFusedMoe",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("SharedFusedMoELayer"),
                ],
            ),
            Instruction::MetalFusedMoe(
                in_slot,
                out_slot,
                layer,
                num_experts,
                top_k,
                moe_inter,
                hidden,
                gs,
                bits,
            ) => (
                "MetalFusedMoe",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("FusedMoELayer"),
                    F::ConstU32(num_experts),
                    F::ConstU32(top_k),
                    F::ConstU32(moe_inter),
                    F::ConstU32(hidden),
                    F::ConstU32(gs),
                    F::ConstU32(bits),
                ],
            ),
            Instruction::MetalSharedFusedMoe(
                in_slot,
                out_slot,
                layer,
                num_experts,
                top_k,
                moe_inter,
                hidden,
                shared_inter,
                gs,
                bits,
                norm_topk_prob,
            ) => (
                "MetalSharedFusedMoe",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("SharedFusedMoELayer"),
                    F::ConstU32(num_experts),
                    F::ConstU32(top_k),
                    F::ConstU32(moe_inter),
                    F::ConstU32(hidden),
                    F::ConstU32(shared_inter),
                    F::ConstU32(gs),
                    F::ConstU32(bits),
                    F::ConstBool(norm_topk_prob),
                ],
            ),
            Instruction::GemmaMoe(
                router_in,
                expert_in,
                out_slot,
                layer,
                num_experts,
                top_k,
                moe_inter,
                hidden,
                gs,
                bits,
            ) => (
                "GemmaMoe",
                vec![
                    F::Slot(router_in),
                    F::Slot(expert_in),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("GemmaRouterLayer"),
                    F::LayerKind("SwitchGluExpertsLayer"),
                    F::ConstU32(num_experts),
                    F::ConstU32(top_k),
                    F::ConstU32(moe_inter),
                    F::ConstU32(hidden),
                    F::ConstU32(gs),
                    F::ConstU32(bits),
                ],
            ),
            Instruction::CutlassGemm(in_slot, out_slot, layer, tile_m, tile_n, stages, n, k) => (
                "CutlassGemm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                    F::WeightShape { n, k },
                ],
            ),
            Instruction::CutlassGemmSplitK(
                in_slot,
                out_slot,
                layer,
                tile_m,
                tile_n,
                stages,
                split_k,
                n,
                k,
            ) => (
                "CutlassGemmSplitK",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                    F::ConstU32(split_k),
                    F::WeightShape { n, k },
                ],
            ),
            Instruction::CutlassGemmAdd(
                in_slot,
                residual_slot,
                layer,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => (
                "CutlassGemmAdd",
                vec![
                    F::Slot(in_slot),
                    F::Slot(residual_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                    F::WeightShape { n, k },
                ],
            ),
            Instruction::CutlassGemv(in_slot, out_slot, layer, n, k) => (
                "CutlassGemv",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::WeightShape { n, k },
                ],
            ),
            Instruction::CutlassFusedGemmBias(
                in_slot,
                out_slot,
                layer,
                tile_m,
                tile_n,
                stages,
                n,
                k,
            ) => (
                "CutlassFusedGemmBias",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                    F::WeightShape { n, k },
                ],
            ),
            Instruction::CutlassFusedGateUpSiluMul(
                in_slot,
                out_slot,
                layer,
                tile_m,
                tile_n,
                stages,
            ) => (
                "CutlassFusedGateUpSiluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                ],
            ),
            Instruction::CutlassFusedGateUpGeluMul(
                in_slot,
                out_slot,
                layer,
                tile_m,
                tile_n,
                stages,
                packed_n,
                k,
            ) => (
                "CutlassFusedGateUpGeluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                    F::WeightShape { n: packed_n, k },
                ],
            ),
            Instruction::CutlassFusedQkvRopeCache(
                in_slot,
                out_slot,
                layer,
                interleaved,
                tile_m,
                tile_n,
                stages,
                packed_n,
                k,
            ) => (
                "CutlassFusedQkvRopeCache",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::RopeCosSin,
                    F::ConstBool(interleaved),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                    F::WeightShape { n: packed_n, k },
                ],
            ),
            Instruction::CutlassFusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
                interleaved,
                tile_m,
                tile_n,
                stages,
                packed_n,
                k,
            ) => (
                "CutlassFusedQkvRopePrefill",
                vec![
                    F::Slot(in_slot),
                    F::Slot(q_out_slot),
                    F::Slot(k_out_slot),
                    F::Slot(v_out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::RopeCosSin,
                    F::ConstBool(interleaved),
                    F::ConstU32(tile_m),
                    F::ConstU32(tile_n),
                    F::ConstU32(stages),
                    F::WeightShape { n: packed_n, k },
                ],
            ),
            Instruction::MarlinGemm(in_slot, out_slot, layer) => (
                "MarlinGemm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("MarlinLinear"),
                ],
            ),
            Instruction::MarlinFusedGateUpSiluMul(in_slot, out_slot, layer) => (
                "MarlinFusedGateUpSiluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("MarlinLinear"),
                ],
            ),
            Instruction::MarlinFusedGateUpGeluMul(in_slot, out_slot, layer) => (
                "MarlinFusedGateUpGeluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("MarlinLinear"),
                ],
            ),
            Instruction::MarlinFusedQkvRopeCache(in_slot, out_slot, layer) => (
                "MarlinFusedQkvRopeCache",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("MarlinLinear"),
                    F::RopeCosSin,
                ],
            ),
            Instruction::MarlinFusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
            ) => (
                "MarlinFusedQkvRopePrefill",
                vec![
                    F::Slot(in_slot),
                    F::Slot(q_out_slot),
                    F::Slot(k_out_slot),
                    F::Slot(v_out_slot),
                    F::Layer(layer),
                    F::LayerKind("MarlinLinear"),
                    F::RopeCosSin,
                ],
            ),
            Instruction::GgmlGemm(in_slot, out_slot, layer) => (
                "GgmlGemm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::GgmlFusedGateUpSiluMul(in_slot, out_slot, layer) => (
                "GgmlFusedGateUpSiluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::GgmlFusedGateUpGeluMul(in_slot, out_slot, layer) => (
                "GgmlFusedGateUpGeluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::GgmlFusedQkvRopeCache(in_slot, out_slot, layer, _i) => (
                "GgmlFusedQkvRopeCache",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::RopeCosSin,
                ],
            ),
            Instruction::GgmlFusedQkvRopePrefill(in_slot, q_out, k_out, v_out, layer) => (
                "GgmlFusedQkvRopePrefill",
                vec![
                    F::Slot(in_slot),
                    F::Slot(q_out),
                    F::Slot(k_out),
                    F::Slot(v_out),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::RopeCosSin,
                ],
            ),
            Instruction::Bnb4Gemm(in_slot, out_slot, layer) => (
                "Bnb4Gemm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("Bnb4bitLinear"),
                ],
            ),
            Instruction::Bnb4FusedGateUpSiluMul(in_slot, out_slot, layer) => (
                "Bnb4FusedGateUpSiluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("Bnb4bitLinear"),
                ],
            ),
            Instruction::Bnb4FusedGateUpGeluMul(in_slot, out_slot, layer) => (
                "Bnb4FusedGateUpGeluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("Bnb4bitLinear"),
                ],
            ),
            Instruction::Bnb4FusedQkvRopeCache(in_slot, out_slot, layer) => (
                "Bnb4FusedQkvRopeCache",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("Bnb4bitLinear"),
                    F::RopeCosSin,
                ],
            ),
            Instruction::Bnb4FusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
            ) => (
                "Bnb4FusedQkvRopePrefill",
                vec![
                    F::Slot(in_slot),
                    F::Slot(q_out_slot),
                    F::Slot(k_out_slot),
                    F::Slot(v_out_slot),
                    F::Layer(layer),
                    F::LayerKind("Bnb4bitLinear"),
                    F::RopeCosSin,
                ],
            ),
            Instruction::Fp8Gemm(in_slot, out_slot, layer) => (
                "Fp8Gemm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("Fp8AnyLinear"),
                ],
            ),
            Instruction::Fp8FusedGemmBias(in_slot, out_slot, layer) => (
                "Fp8FusedGemmBias",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("Fp8AnyLinear"),
                ],
            ),
            Instruction::Fp8FusedGateUpSiluMul(in_slot, out_slot, layer) => (
                "Fp8FusedGateUpSiluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("Fp8AnyLinear"),
                ],
            ),
            Instruction::Fp8FusedGateUpGeluMul(in_slot, out_slot, layer) => (
                "Fp8FusedGateUpGeluMul",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("Fp8AnyLinear"),
                ],
            ),
            Instruction::Fp8FusedQkvRopeCache(in_slot, out_slot, layer) => (
                "Fp8FusedQkvRopeCache",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("Fp8AnyLinear"),
                    F::RopeCosSin,
                ],
            ),
            Instruction::Fp8FusedQkvRopePrefill(
                in_slot,
                q_out_slot,
                k_out_slot,
                v_out_slot,
                layer,
            ) => (
                "Fp8FusedQkvRopePrefill",
                vec![
                    F::Slot(in_slot),
                    F::Slot(q_out_slot),
                    F::Slot(k_out_slot),
                    F::Slot(v_out_slot),
                    F::Layer(layer),
                    F::LayerKind("Fp8AnyLinear"),
                    F::RopeCosSin,
                ],
            ),
            // MLX-affine int4 family. `AffineQmm` is the metal-only
            // quantized matmul; CUDA `eval` is `unreachable!`, but the
            // variant is unconditional on `Instruction<W>` so info.rs
            // must cover it under either backend. `SiluMul` is the
            // P3-P4 C4a fused activation that pairs with two
            // `AffineQmm`s on the q-MLP decomposed branch.
            Instruction::AffineQmm(
                in_slot,
                out_slot,
                layer,
                n,
                k,
                _group_size,
                _bits,
                _vector_limit,
            ) => (
                "AffineQmm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::WeightShape { n, k },
                ],
            ),
            // NVFP4 int4 family — same metal-only shape as `AffineQmm`.
            Instruction::Nvfp4Qmm(
                in_slot,
                out_slot,
                layer,
                n,
                k,
                _group_size,
                _bits,
                _vector_limit,
            ) => (
                "Nvfp4Qmm",
                vec![
                    F::Slot(in_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                    F::WeightShape { n, k },
                ],
            ),
            Instruction::SynthPreAttn(
                residual_slot,
                delta_slot,
                out_slot,
                residual_out_slot,
                layer,
                _group_size,
                _bits,
                _symbol,
                _has_linear_bias,
            ) => (
                "SynthPreAttn",
                vec![
                    F::Slot(residual_slot),
                    F::Slot(delta_slot),
                    F::Slot(out_slot),
                    F::Slot(residual_out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::SynthMlpPreDown(
                residual_slot,
                delta_slot,
                out_slot,
                residual_out_slot,
                layer,
                _group_size,
                _bits,
                _symbol,
            ) => (
                "SynthMlpPreDown",
                vec![
                    F::Slot(residual_slot),
                    F::Slot(delta_slot),
                    F::Slot(out_slot),
                    F::Slot(residual_out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::SynthGateUpSiluMul(
                x_norm_slot,
                out_slot,
                layer,
                _group_size,
                _bits,
                _symbol,
            ) => (
                "SynthGateUpSiluMul",
                vec![
                    F::Slot(x_norm_slot),
                    F::Slot(out_slot),
                    F::Layer(layer),
                    F::LayerKind("LinearLayer"),
                ],
            ),
            Instruction::SiluMul(gate_slot, up_slot, out_slot, width) => (
                "SiluMul",
                vec![
                    F::Slot(gate_slot),
                    F::Slot(up_slot),
                    F::Slot(out_slot),
                    F::ConstU32(width),
                ],
            ),
            Instruction::GeluMul(gate_slot, up_slot, out_slot) => (
                "GeluMul",
                vec![F::Slot(gate_slot), F::Slot(up_slot), F::Slot(out_slot)],
            ),
            // Fused gather+dequant for MLX `*-4bit` checkpoints (P6,
            // metal-only at runtime; the neutral normalize projection
            // covers it on every backend).
            Instruction::AffineEmbed(out_slot, _group_size, _bits) => (
                "AffineEmbed",
                vec![F::Slot(out_slot), F::LayerKind("AffineQuantEmbedding")],
            ),
            Instruction::Loop(count, body_len, layer_stride) => (
                "Loop",
                vec![
                    F::LoopCount(count),
                    F::LoopBodyLen(body_len),
                    F::LoopLayerStride(layer_stride),
                ],
            ),
            Instruction::Alias(dst, src) => ("Alias", vec![F::Slot(dst), F::Slot(src)]),
            Instruction::Free(slot) => ("Free", vec![F::Slot(slot)]),
            Instruction::SpliceMmEmbeds(slot) => ("SpliceMmEmbeds", vec![F::Slot(slot)]),
        };
        NormalizedStep { kind, fields }
    }
}

/// Walk a `&[Instruction]` slice and collect [`NormalizedStep`]s
/// in source order. `Loop` rows stay in-line — consumers that want
/// to render them as scoped blocks scan for `kind == "Loop"` and
/// take the next `LoopBodyLen` rows.
#[must_use]
pub fn normalize_slice(slice: &[Instruction]) -> Vec<NormalizedStep> {
    slice.iter().map(InstructionInfo::normalize).collect()
}

/// One bucket's worth of dumped backbone + lm_head, with the
/// workload-bucket bounds the macro emitted alongside the slice.
#[derive(Clone, Debug)]
pub struct BucketDump {
    pub m_min: u64,
    pub m_max_excl: u64,
    pub sk_min: u64,
    pub sk_max_excl: u64,
    pub backbone: Vec<NormalizedStep>,
    pub lm_head: Vec<NormalizedStep>,
}

/// Per-variant backbone dump. Built by macro-emitted `dump()` fns
/// that walk the variant's `FORWARD_TABLE` and call
/// [`normalize_slice`] on each entry's backbone + lm_head.
#[derive(Clone, Debug)]
pub struct VariantDump {
    pub variant_stem: &'static str,
    /// Tensor-parallel world size this variant was compiled for. At
    /// `--features nccl` the macro emits one `VariantDump` per (model,
    /// tp) tuple in `{1, 2, 4, 8}`; at default `--features cuda` only
    /// tp=1 is emitted. `scr model info` displays this on each row
    /// and includes it in its substring filter as `tp=N`.
    pub tp_world_size: u8,
    pub buckets: Vec<BucketDump>,
}

/// Per-arch entry in the backbone-dump registry. The `dump_all` fn
/// walks every variant in this arch's compiled set and returns a
/// flat list. One [`BackboneDumpRegistration`] per `#[forward]` —
/// inventory-collected, mirroring the existing
/// `ScratchyArchRegistration` pattern.
pub struct BackboneDumpRegistration {
    pub arch_name: &'static str,
    pub dump_all: fn() -> Vec<VariantDump>,
}

inventory::collect!(BackboneDumpRegistration);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_normalizes() {
        let i: Instruction = Instruction::Embed(7);
        let n = i.normalize();
        assert_eq!(n.kind, "Embed");
        assert_eq!(
            n.fields,
            vec![
                NormalizedField::Slot(7),
                NormalizedField::LayerKind("Embedding"),
            ]
        );
    }

    #[test]
    fn fused_add_rmsnorm_carries_layer_and_kernel_class() {
        let i: Instruction = Instruction::FusedAddRmsNorm(3, 4, 12, 2048, 1);
        let n = i.normalize();
        assert_eq!(n.kind, "FusedAddRmsNorm");
        assert_eq!(
            n.fields,
            vec![
                NormalizedField::Slot(3),
                NormalizedField::Slot(4),
                NormalizedField::Layer(12),
                NormalizedField::LayerKind("RmsNorm"),
            ]
        );
    }

    #[test]
    fn cutlass_gemm_add_keeps_tile_consts() {
        let i: Instruction = Instruction::CutlassGemmAdd(5, 6, 0, 128, 128, 3, 4096, 11008);
        let n = i.normalize();
        assert_eq!(n.kind, "CutlassGemmAdd");
        assert_eq!(
            n.fields,
            vec![
                NormalizedField::Slot(5),
                NormalizedField::Slot(6),
                NormalizedField::Layer(0),
                NormalizedField::LayerKind("LinearLayer"),
                NormalizedField::ConstU32(128),
                NormalizedField::ConstU32(128),
                NormalizedField::ConstU32(3),
                NormalizedField::WeightShape { n: 4096, k: 11008 },
            ]
        );
    }

    #[test]
    fn loop_count_and_body_len_are_distinguishable() {
        let i: Instruction = Instruction::Loop(32, 9, 1);
        let n = i.normalize();
        assert_eq!(n.kind, "Loop");
        assert_eq!(
            n.fields,
            vec![
                NormalizedField::LoopCount(32),
                NormalizedField::LoopBodyLen(9),
                NormalizedField::LoopLayerStride(1),
            ]
        );
    }

    #[test]
    fn flashinfer_decode_keeps_head_dim_and_softcap_flag() {
        let i: Instruction = Instruction::FlashInferAttentionDecode(0, 1, 0, 128, true);
        let n = i.normalize();
        assert_eq!(n.kind, "FlashInferAttentionDecode");
        assert_eq!(
            n.fields,
            vec![
                NormalizedField::Slot(0),
                NormalizedField::Slot(1),
                NormalizedField::Layer(0),
                NormalizedField::RopeCosSin,
                NormalizedField::ConstU32(128),
                NormalizedField::ConstBool(true),
            ]
        );
    }
}
