// SPDX-License-Identifier: Apache-2.0
//! THE opcode-shape table — every `Instruction` variant the shared-tape
//! bridge emits, as ONE const-data site — instruction-selection
//! metadata lives as tables. During the transition the impls
//! keep their own `opcode_shape()` copies; `ArchOpcodes`' agreement
//! panic enforces that both sources stay identical, so drift is a
//! BUILD error — when the impls die, this module is the single source.

use crate::weight_vocab::OpcodeShape;

/// Shape for a bridge-emitted variant. `None` = not a bridge variant
/// (the caller refuses loudly).
pub(crate) fn bridge_shape(name: &str) -> Option<OpcodeShape> {
    match name {
        "Add" => Some(OpcodeShape::new(
            "Add",
            vec![
                ("delta_slot", syn::parse_quote!(u32)),
                ("residual_slot", syn::parse_quote!(u32)),
            ],
        )),
        "AffineEmbed" => Some(OpcodeShape::new(
            "AffineEmbed",
            vec![
                ("out_slot", syn::parse_quote!(u32)),
                ("group_size", syn::parse_quote!(u32)),
                ("bits", syn::parse_quote!(u32)),
            ],
        )),
        "AffineQmm" => Some(OpcodeShape::new(
            "AffineQmm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
                ("group_size", syn::parse_quote!(u32)),
                ("bits", syn::parse_quote!(u32)),
                ("vector_limit", syn::parse_quote!(u32)),
            ],
        )),
        "AttentionPrefillPaged" => Some(OpcodeShape::new(
            "AttentionPrefillPaged",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("interleaved", syn::parse_quote!(bool)),
            ],
        )),
        "AttentionViaCache" => Some(OpcodeShape::new(
            "AttentionViaCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("interleaved", syn::parse_quote!(bool)),
            ],
        )),
        "Embed" => Some(OpcodeShape::new(
            "Embed",
            vec![("out_slot", syn::parse_quote!(u32))],
        )),
        "EncoderAttention" => Some(OpcodeShape::new(
            "EncoderAttention",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("k_slot", syn::parse_quote!(u32)),
                ("v_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
            ],
        )),
        "FusedAddRmsNorm" => Some(OpcodeShape::new(
            "FusedAddRmsNorm",
            vec![
                ("delta_slot", syn::parse_quote!(u32)),
                ("residual_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("hidden_size", syn::parse_quote!(u32)),
                ("m_multiplier", syn::parse_quote!(u32)),
            ],
        )),
        "FusedAddRmsNormWithOffset" => Some(OpcodeShape::new(
            "FusedAddRmsNormWithOffset",
            vec![
                ("delta_slot", syn::parse_quote!(u32)),
                ("residual_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("offset", syn::parse_quote!(f32)),
            ],
        )),
        "FusedGateUpGeluMul" => Some(OpcodeShape::new(
            "FusedGateUpGeluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )),
        "FusedGateUpSiluMul" => Some(OpcodeShape::new(
            "FusedGateUpSiluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )),
        "GateApply" => Some(OpcodeShape::new(
            "GateApply",
            vec![
                ("attn_slot", syn::parse_quote!(u32)),
                ("gate_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
            ],
        )),
        "GateScale" => Some(OpcodeShape::new(
            "GateScale",
            vec![
                ("routed_slot", syn::parse_quote!(u32)),
                ("shared_slot", syn::parse_quote!(u32)),
                ("gate_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
            ],
        )),
        "GateSplit" => Some(OpcodeShape::new(
            "GateSplit",
            vec![
                ("qg_slot", syn::parse_quote!(u32)),
                ("q_slot", syn::parse_quote!(u32)),
                ("gate_slot", syn::parse_quote!(u32)),
            ],
        )),
        "GatedDeltaNet" => Some(OpcodeShape::new(
            "GatedDeltaNet",
            vec![
                ("qkv_slot", syn::parse_quote!(u32)),
                ("z_slot", syn::parse_quote!(u32)),
                ("a_slot", syn::parse_quote!(u32)),
                ("b_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )),
        "GeluErf" => Some(OpcodeShape::new(
            "GeluErf",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
            ],
        )),
        "GeluMul" => Some(OpcodeShape::new(
            "GeluMul",
            vec![
                ("gate_slot", syn::parse_quote!(u32)),
                ("up_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
            ],
        )),
        "Gemm" => Some(OpcodeShape::new(
            "Gemm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )),
        "GemmaMoe" => Some(OpcodeShape::new(
            "GemmaMoe",
            vec![
                ("router_in", syn::parse_quote!(u32)),
                ("expert_in", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("num_experts", syn::parse_quote!(u32)),
                ("top_k", syn::parse_quote!(u32)),
                ("moe_intermediate_size", syn::parse_quote!(u32)),
                ("hidden_size", syn::parse_quote!(u32)),
                ("group_size", syn::parse_quote!(u32)),
                ("bits", syn::parse_quote!(u32)),
            ],
        )),
        "LoadPixels" => Some(OpcodeShape::new(
            "LoadPixels",
            vec![("out_slot", syn::parse_quote!(u32))],
        )),
        "MeanSubRmsNorm" => Some(OpcodeShape::new(
            "MeanSubRmsNorm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )),
        "MeanSubRmsNormBiasAdd" => Some(OpcodeShape::new(
            "MeanSubRmsNormBiasAdd",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )),
        "MetalBiasAdd" => Some(OpcodeShape::new(
            "MetalBiasAdd",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("n", syn::parse_quote!(u32)),
                ("is_affine", syn::parse_quote!(bool)),
            ],
        )),
        "MetalFusedMoe" => Some(OpcodeShape::new(
            "MetalFusedMoe",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("num_experts", syn::parse_quote!(u32)),
                ("top_k", syn::parse_quote!(u32)),
                ("moe_intermediate_size", syn::parse_quote!(u32)),
                ("hidden_size", syn::parse_quote!(u32)),
                ("group_size", syn::parse_quote!(u32)),
                ("bits", syn::parse_quote!(u32)),
            ],
        )),
        "MetalSharedFusedMoe" => Some(OpcodeShape::new(
            "MetalSharedFusedMoe",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("num_experts", syn::parse_quote!(u32)),
                ("top_k", syn::parse_quote!(u32)),
                ("moe_intermediate_size", syn::parse_quote!(u32)),
                ("hidden_size", syn::parse_quote!(u32)),
                ("shared_intermediate_size", syn::parse_quote!(u32)),
                ("group_size", syn::parse_quote!(u32)),
                ("bits", syn::parse_quote!(u32)),
                ("norm_topk_prob", syn::parse_quote!(bool)),
            ],
        )),
        "NormAddScalarMul" => Some(OpcodeShape::new(
            "NormAddScalarMul",
            vec![
                ("delta_slot", syn::parse_quote!(u32)),
                ("residual_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("hidden_size", syn::parse_quote!(u32)),
            ],
        )),
        "QuickGelu" => Some(OpcodeShape::new(
            "QuickGelu",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
            ],
        )),
        "Reshape" => Some(OpcodeShape::new(
            "Reshape",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                (
                    "dims_lit",
                    syn::parse_quote!([u32; crate::__gpu::tensor::MAX_DIMS]),
                ),
                (
                    "dims_nt_pow",
                    syn::parse_quote!([u8; crate::__gpu::tensor::MAX_DIMS]),
                ),
                (
                    "dims_div_lit",
                    syn::parse_quote!([u32; crate::__gpu::tensor::MAX_DIMS]),
                ),
                ("ndim", syn::parse_quote!(u8)),
            ],
        )),
        "RmsNorm" => Some(OpcodeShape::new(
            "RmsNorm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("hidden_size", syn::parse_quote!(u32)),
                ("m_multiplier", syn::parse_quote!(u32)),
            ],
        )),
        "RmsNormUnit" => Some(OpcodeShape::new(
            "RmsNormUnit",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("hidden_size", syn::parse_quote!(u32)),
                ("m_multiplier", syn::parse_quote!(u32)),
            ],
        )),
        "RopeAppend" => Some(OpcodeShape::new(
            "RopeAppend",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("k_slot", syn::parse_quote!(u32)),
                ("v_slot", syn::parse_quote!(u32)),
                ("q_out_slot", syn::parse_quote!(u32)),
                ("k_out_slot", syn::parse_quote!(u32)),
                ("v_out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("interleaved", syn::parse_quote!(bool)),
                ("is_global", syn::parse_quote!(bool)),
                (
                    "kv_offsets",
                    syn::parse_quote!(::scratchy_forward_compiler::KvOffsets),
                ),
            ],
        )),
        "RopeAppendNormed" => Some(OpcodeShape::new(
            "RopeAppendNormed",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("k_slot", syn::parse_quote!(u32)),
                ("v_slot", syn::parse_quote!(u32)),
                ("q_out_slot", syn::parse_quote!(u32)),
                ("k_out_slot", syn::parse_quote!(u32)),
                ("v_out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("interleaved", syn::parse_quote!(bool)),
                ("is_global", syn::parse_quote!(bool)),
            ],
        )),
        "ScalarMul" => Some(OpcodeShape::new(
            "ScalarMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("scale", syn::parse_quote!(f32)),
            ],
        )),
        "ScalarOffsetRmsNorm" => Some(OpcodeShape::new(
            "ScalarOffsetRmsNorm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("offset", syn::parse_quote!(f32)),
            ],
        )),
        "ScalarWeightMul" => Some(OpcodeShape::new(
            "ScalarWeightMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        )),
        "SiluMul" => Some(OpcodeShape::new(
            "SiluMul",
            vec![
                ("gate_slot", syn::parse_quote!(u32)),
                ("up_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("width", syn::parse_quote!(u32)),
            ],
        )),
        "SlidingAttentionPrefillPaged" => Some(OpcodeShape::new(
            "SlidingAttentionPrefillPaged",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("interleaved", syn::parse_quote!(bool)),
            ],
        )),
        "SlidingAttentionViaCache" => Some(OpcodeShape::new(
            "SlidingAttentionViaCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("interleaved", syn::parse_quote!(bool)),
            ],
        )),
        "SpliceMmEmbeds" => Some(OpcodeShape::new(
            "SpliceMmEmbeds",
            vec![("slot", syn::parse_quote!(u32))],
        )),
        "TanhSoftCap" => Some(OpcodeShape::new(
            "TanhSoftCap",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
            ],
        )),
        "VarlenAttention" => Some(OpcodeShape::new(
            "VarlenAttention",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("k_slot", syn::parse_quote!(u32)),
                ("v_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("cu_seqlens_kind", syn::parse_quote!(u8)),
            ],
        )),
        "VisionRope" => Some(OpcodeShape::new(
            "VisionRope",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("k_slot", syn::parse_quote!(u32)),
                ("q_out_slot", syn::parse_quote!(u32)),
                ("k_out_slot", syn::parse_quote!(u32)),
            ],
        )),
        _ => None,
    }
}

/// Every bridge-emitted variant name, for whole-table registration.
pub(crate) const BRIDGE_VARIANTS: &[&str] = &[
    "Add",
    "AffineEmbed",
    "AffineQmm",
    "AttentionPrefillPaged",
    "AttentionViaCache",
    "Embed",
    "EncoderAttention",
    "FusedAddRmsNorm",
    "FusedAddRmsNormWithOffset",
    "FusedGateUpGeluMul",
    "FusedGateUpSiluMul",
    "GateApply",
    "GateScale",
    "GateSplit",
    "GatedDeltaNet",
    "GeluErf",
    "GeluMul",
    "Gemm",
    "GemmaMoe",
    "LoadPixels",
    "MeanSubRmsNorm",
    "MeanSubRmsNormBiasAdd",
    "MetalBiasAdd",
    "MetalFusedMoe",
    "MetalSharedFusedMoe",
    "NormAddScalarMul",
    "QuickGelu",
    "Reshape",
    "RmsNorm",
    "RmsNormUnit",
    "RopeAppend",
    "RopeAppendNormed",
    "ScalarMul",
    "ScalarOffsetRmsNorm",
    "ScalarWeightMul",
    "SiluMul",
    "SlidingAttentionPrefillPaged",
    "SlidingAttentionViaCache",
    "SpliceMmEmbeds",
    "TanhSoftCap",
    "VarlenAttention",
    "VisionRope",
];
