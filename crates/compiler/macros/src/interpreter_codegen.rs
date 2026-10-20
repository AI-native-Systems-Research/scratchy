// SPDX-License-Identifier: Apache-2.0
//! Host-interpreter codegen.
//!
//! Replaces the per-tile inlined `let X = kernel_call(...);` body
//! the old `emit_subgraph` produced. The new shape per arch:
//!
//! - One Rust enum codegened from the FUF the solver actually
//!   solved for that arch. Variants are exactly the kernel calls
//!   the picked Impls produce, plus `Free` (drop-pass).
//! - One `static FORWARD_M_<N>: &[<Arch>Op] = &[…];` per (variant
//!   × workload-point). Each element is a per-arch enum value.
//! - One per-arch interpreter — `for op in slice { match op { … }
//!   }` — closed and exhaustive over the per-arch enum.
//!
//! No universal opcode enum. No central registry. No string-keyed
//! opcode lookup. No `_` catch-all arm. No `unsafe { transmute }`
//! or `from_wire_unchecked`. The compiler enforces exhaustiveness
//! over the enum the macro just minted from this arch's solved FUF.
//!
//! # Megakernel concerns are out of scope
//!
//! Megakernel codegen is a separate code generator. When it lands,
//! it gets its own translator from per-arch enums to its own wire
//! format. This module does not produce `[i32; 32]` packed rows,
//! does not match KVM tp_throughput opcode numbering, does not
//! emit `Noop` padding. Those are megakernel concerns.

use proc_macro2::TokenStream;
use quote::quote;

use scratchy_forward_compiler::Instruction;

use crate::weight_vocab::WeightSlot;

// ── Instruction helpers ──────────────────────────────────────────
//
// `Instruction` (from scratchy-forward-compiler) is the universal typed
// tape row. These helpers render an `Instruction` back to TokenStream
// form for the static-slice emit.

/// Render one [`Instruction`] as a tuple-style variant constructor:
/// `Embed(7)`, `RmsNorm(0, 1, 2)`, etc. Each numeric field renders
/// as an unsuffixed integer literal; bool fields render as
/// `true`/`false`; `f32` fields render as `f32::from_bits(<bits>)`
/// because direct float literals can't represent NaN / Inf reliably.
/// Array fields render as bracketed literals.
///
/// The variant ident matches the Rust enum, and the per-arch
/// codegen prelude does `use ::scratchy_forward_compiler::Instruction::*;` so
/// each row reads as `Variant(...)` without the path prefix.
pub fn instruction_to_tokens(inst: &Instruction) -> TokenStream {
    use scratchy_forward_compiler::Instruction as I;
    let lit_u32 = |v: u32| -> TokenStream {
        let lit = proc_macro2::Literal::u32_unsuffixed(v);
        quote! { #lit }
    };
    let lit_u8 = |v: u8| -> TokenStream {
        let lit = proc_macro2::Literal::u8_unsuffixed(v);
        quote! { #lit }
    };
    let lit_bool = |v: bool| -> TokenStream {
        if v {
            quote! { true }
        } else {
            quote! { false }
        }
    };
    let lit_f32 = |v: f32| -> TokenStream {
        let bits = v.to_bits();
        let blit = proc_macro2::Literal::u32_unsuffixed(bits);
        quote! { f32::from_bits(#blit) }
    };
    let lit_arr_u32 = |arr: &[u32]| -> TokenStream {
        let elems = arr.iter().map(|&x| lit_u32(x));
        quote! { [ #(#elems),* ] }
    };
    let lit_arr_u8 = |arr: &[u8]| -> TokenStream {
        let elems = arr.iter().map(|&x| lit_u8(x));
        quote! { [ #(#elems),* ] }
    };
    match *inst {
        I::Embed(a) => {
            let a = lit_u32(a);
            quote! { Embed(#a) }
        }
        I::RmsNorm(a, b, c, d, e) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            quote! { RmsNorm(#a, #b, #c, #d, #e) }
        }
        I::MeanSubRmsNorm(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { MeanSubRmsNorm(#a, #b, #c) }
        }
        I::MeanSubRmsNormBiasAdd(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { MeanSubRmsNormBiasAdd(#a, #b, #c) }
        }
        I::Reshape(a, b, c, d, e, f) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_arr_u32(&c);
            let d = lit_arr_u8(&d);
            let e = lit_arr_u32(&e);
            let f = lit_u8(f);
            quote! { Reshape(#a, #b, #c, #d, #e, #f) }
        }
        I::Add(a, b) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            quote! { Add(#a, #b) }
        }
        I::SpliceMmEmbeds(a) => {
            let a = lit_u32(a);
            quote! { SpliceMmEmbeds(#a) }
        }
        I::ScalarMul(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_f32(c);
            quote! { ScalarMul(#a, #b, #c) }
        }
        I::TanhSoftCap(a, b) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            quote! { TanhSoftCap(#a, #b) }
        }
        I::RmsNormUnit(a, b, c, d) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            quote! { RmsNormUnit(#a, #b, #c, #d) }
        }
        I::ScalarWeightMul(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { ScalarWeightMul(#a, #b, #c) }
        }
        I::NormAddScalarMul(a, b, c, d, e) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            quote! { NormAddScalarMul(#a, #b, #c, #d, #e) }
        }
        I::FusedAddRmsNorm(a, b, c, d, e) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            quote! { FusedAddRmsNorm(#a, #b, #c, #d, #e) }
        }
        I::FusedAddRmsNormWithOffset(a, b, c, d) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_f32(d);
            quote! { FusedAddRmsNormWithOffset(#a, #b, #c, #d) }
        }
        I::ScalarOffsetRmsNorm(a, b, c, d, e, f) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_f32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            quote! { ScalarOffsetRmsNorm(#a, #b, #c, #d, #e, #f) }
        }
        I::CutlassFusedRmsNormGemm(a, b, c, d, e, f, g, h) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            let g = lit_u32(g);
            let h = lit_u32(h);
            quote! { CutlassFusedRmsNormGemm(#a, #b, #c, #d, #e, #f, #g, #h) }
        }
        I::CutlassFusedMeanSubRmsNormGemm(a, b, c, d, e, f, g, h) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            let g = lit_u32(g);
            let h = lit_u32(h);
            quote! { CutlassFusedMeanSubRmsNormGemm(#a, #b, #c, #d, #e, #f, #g, #h) }
        }
        I::CutlassFusedAddRmsNormGemm(a, b, c, d, e, f, g, h, i) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            let g = lit_u32(g);
            let h = lit_u32(h);
            let i = lit_u32(i);
            quote! { CutlassFusedAddRmsNormGemm(#a, #b, #c, #d, #e, #f, #g, #h, #i) }
        }
        I::Gemm(a, b, c, d, e) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            quote! { Gemm(#a, #b, #c, #d, #e) }
        }
        I::FusedCublasGemmAdd(a, b, c, d, e) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            quote! { FusedCublasGemmAdd(#a, #b, #c, #d, #e) }
        }
        I::FusedGemmBias(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { FusedGemmBias(#a, #b, #c) }
        }
        I::FusedGateUpSiluMul(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { FusedGateUpSiluMul(#a, #b, #c) }
        }
        I::FusedGateUpGeluMul(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { FusedGateUpGeluMul(#a, #b, #c) }
        }
        I::FusedQkvRopeCache(a, b, c, d, e) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_bool(d);
            let e = lit_bool(e);
            quote! { FusedQkvRopeCache(#a, #b, #c, #d, #e) }
        }
        I::FusedQkvQkNormRopeCache(a, b, c, d, e) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_f32(d);
            let e = lit_f32(e);
            quote! { FusedQkvQkNormRopeCache(#a, #b, #c, #d, #e) }
        }
        I::FusedQkvRopePrefill(a, b, c, d, e, f, g) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_bool(f);
            let g = lit_bool(g);
            quote! { FusedQkvRopePrefill(#a, #b, #c, #d, #e, #f, #g) }
        }
        I::AttentionViaCache(a, b, c, d) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_bool(d);
            quote! { AttentionViaCache(#a, #b, #c, #d) }
        }
        I::AttentionPrefillContiguous(a, b, c, d, e) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_bool(e);
            quote! { AttentionPrefillContiguous(#a, #b, #c, #d, #e) }
        }
        I::EncoderAttention(a, b, c, d) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            quote! { EncoderAttention(#a, #b, #c, #d) }
        }
        I::SlidingAttentionViaCache(a, b, c, d) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_bool(d);
            quote! { SlidingAttentionViaCache(#a, #b, #c, #d) }
        }
        I::SlidingAttentionPrefillContiguous(a, b, c, d, e) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_bool(e);
            quote! { SlidingAttentionPrefillContiguous(#a, #b, #c, #d, #e) }
        }
        I::SlidingAttentionPrefillPaged(a, b, c, d) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_bool(d);
            quote! { SlidingAttentionPrefillPaged(#a, #b, #c, #d) }
        }
        I::VarlenAttention(a, b, c, d, e) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u8(e);
            quote! { VarlenAttention(#a, #b, #c, #d, #e) }
        }
        I::VisionRope(a, b, c, d) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            quote! { VisionRope(#a, #b, #c, #d) }
        }
        I::QuickGelu(a, b) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            quote! { QuickGelu(#a, #b) }
        }
        I::Gelu(a, b) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            quote! { Gelu(#a, #b) }
        }
        I::PosEmbed(a) => {
            let a = lit_u32(a);
            quote! { PosEmbed(#a) }
        }
        I::LoadRows(a, source) => {
            let a = lit_u32(a);
            let source = rows_extern_tokens(source);
            quote! { LoadRows(#a, #source) }
        }
        I::GeluErf(a, b) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            quote! { GeluErf(#a, #b) }
        }
        I::EmbeddingGather(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u8(c);
            quote! { EmbeddingGather(#a, #b, #c) }
        }
        I::AvgPool2d(a, b) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            quote! { AvgPool2d(#a, #b) }
        }
        I::StripCls(a, b) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            quote! { StripCls(#a, #b) }
        }
        I::FlashInferAttentionDecode(a, b, c, d, e) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_bool(e);
            quote! { FlashInferAttentionDecode(#a, #b, #c, #d, #e) }
        }
        I::FlashInferAttentionPrefill(a, b, c, d, e, f, g) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            let g = lit_bool(g);
            quote! { FlashInferAttentionPrefill(#a, #b, #c, #d, #e, #f, #g) }
        }
        I::RopeAppend(a, b, c, d, e, f, g, h, i, j) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            let g = lit_u32(g);
            let h = lit_bool(h);
            let i = lit_bool(i);
            let j = kv_offsets_tokens(j);
            quote! { RopeAppend(#a, #b, #c, #d, #e, #f, #g, #h, #i, #j) }
        }
        I::RopeAppendNormed(a, b, c, d, e, f, g, h, i) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            let g = lit_u32(g);
            quote! { RopeAppendNormed(#a, #b, #c, #d, #e, #f, #g, #h, #i) }
        }
        I::MlaSplit(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { MlaSplit(#a, #b, #c) }
        }
        I::MlaAttention(a, b, c, d, e) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            quote! { MlaAttention(#a, #b, #c, #d, #e) }
        }
        I::GatedDeltaNet(a, b, c, d, e, f) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            quote! { GatedDeltaNet(#a, #b, #c, #d, #e, #f) }
        }
        I::GateSplit(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { GateSplit(#a, #b, #c) }
        }
        I::GateApply(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { GateApply(#a, #b, #c) }
        }
        I::Concat(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { Concat(#a, #b, #c) }
        }
        I::GateScale(a, b, c, d) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            quote! { GateScale(#a, #b, #c, #d) }
        }
        I::DeepSeekMoe(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { DeepSeekMoe(#a, #b, #c) }
        }
        I::DeepSeekMoeFp8Block(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { DeepSeekMoeFp8Block(#a, #b, #c) }
        }
        I::DeepSeekMoeGgml(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { DeepSeekMoeGgml(#a, #b, #c) }
        }
        I::FusedMoe(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { FusedMoe(#a, #b, #c) }
        }
        I::SharedFusedMoe(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { SharedFusedMoe(#a, #b, #c) }
        }
        I::MetalFusedMoe(a, b, c, d, e, f, g, h, i) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            let g = lit_u32(g);
            let h = lit_u32(h);
            let i = lit_u32(i);
            quote! { MetalFusedMoe(#a, #b, #c, #d, #e, #f, #g, #h, #i) }
        }
        I::MetalSharedFusedMoe(a, b, c, d, e, f, g, h, i, j, k) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            let g = lit_u32(g);
            let h = lit_u32(h);
            let i = lit_u32(i);
            let j = lit_u32(j);
            let k = if k {
                quote! { true }
            } else {
                quote! { false }
            };
            quote! { MetalSharedFusedMoe(#a, #b, #c, #d, #e, #f, #g, #h, #i, #j, #k) }
        }
        I::GemmaMoe(a, b, c, d, e, f, g, h, i, j) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            let g = lit_u32(g);
            let h = lit_u32(h);
            let i = lit_u32(i);
            let j = lit_u32(j);
            quote! { GemmaMoe(#a, #b, #c, #d, #e, #f, #g, #h, #i, #j) }
        }
        I::CutlassGemm(a, b, c, d, e, f, g, h) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            let g = lit_u32(g);
            let h = lit_u32(h);
            quote! { CutlassGemm(#a, #b, #c, #d, #e, #f, #g, #h) }
        }
        I::CutlassGemmSplitK(a, b, c, d, e, f, g, h, i) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            let g = lit_u32(g);
            let h = lit_u32(h);
            let i = lit_u32(i);
            quote! { CutlassGemmSplitK(#a, #b, #c, #d, #e, #f, #g, #h, #i) }
        }
        I::CutlassGemmAdd(a, b, c, d, e, f, g, h) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            let g = lit_u32(g);
            let h = lit_u32(h);
            quote! { CutlassGemmAdd(#a, #b, #c, #d, #e, #f, #g, #h) }
        }
        I::CutlassGemv(a, b, c, d, e) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            quote! { CutlassGemv(#a, #b, #c, #d, #e) }
        }
        I::CutlassFusedGemmBias(a, b, c, d, e, f, g, h) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            let g = lit_u32(g);
            let h = lit_u32(h);
            quote! { CutlassFusedGemmBias(#a, #b, #c, #d, #e, #f, #g, #h) }
        }
        I::CutlassFusedGateUpSiluMul(a, b, c, d, e, f) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            quote! { CutlassFusedGateUpSiluMul(#a, #b, #c, #d, #e, #f) }
        }
        I::CutlassFusedGateUpGeluMul(a, b, c, d, e, f, g, h) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            let g = lit_u32(g);
            let h = lit_u32(h);
            quote! { CutlassFusedGateUpGeluMul(#a, #b, #c, #d, #e, #f, #g, #h) }
        }
        I::CutlassFusedQkvRopeCache(a, b, c, d, e, f, g, h, i) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_bool(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            let g = lit_u32(g);
            let h = lit_u32(h);
            let i = lit_u32(i);
            quote! { CutlassFusedQkvRopeCache(#a, #b, #c, #d, #e, #f, #g, #h, #i) }
        }
        I::CutlassFusedQkvRopePrefill(a, b, c, d, e, f, g, h, i, j, k) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_bool(f);
            let g = lit_u32(g);
            let h = lit_u32(h);
            let i = lit_u32(i);
            let j = lit_u32(j);
            let k = lit_u32(k);
            quote! { CutlassFusedQkvRopePrefill(#a, #b, #c, #d, #e, #f, #g, #h, #i, #j, #k) }
        }
        I::MarlinGemm(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { MarlinGemm(#a, #b, #c) }
        }
        I::MarlinFusedGateUpSiluMul(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { MarlinFusedGateUpSiluMul(#a, #b, #c) }
        }
        I::MarlinFusedGateUpGeluMul(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { MarlinFusedGateUpGeluMul(#a, #b, #c) }
        }
        I::MarlinFusedQkvRopeCache(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { MarlinFusedQkvRopeCache(#a, #b, #c) }
        }
        I::MarlinFusedQkvRopePrefill(a, b, c, d, e) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            quote! { MarlinFusedQkvRopePrefill(#a, #b, #c, #d, #e) }
        }
        I::Bnb4Gemm(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { Bnb4Gemm(#a, #b, #c) }
        }
        I::Bnb4FusedGateUpSiluMul(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { Bnb4FusedGateUpSiluMul(#a, #b, #c) }
        }
        I::Bnb4FusedGateUpGeluMul(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { Bnb4FusedGateUpGeluMul(#a, #b, #c) }
        }
        I::Bnb4FusedQkvRopeCache(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { Bnb4FusedQkvRopeCache(#a, #b, #c) }
        }
        I::Bnb4FusedQkvRopePrefill(a, b, c, d, e) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            quote! { Bnb4FusedQkvRopePrefill(#a, #b, #c, #d, #e) }
        }
        I::GgmlGemm(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { GgmlGemm(#a, #b, #c) }
        }
        I::GgmlFusedGateUpSiluMul(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { GgmlFusedGateUpSiluMul(#a, #b, #c) }
        }
        I::GgmlFusedGateUpGeluMul(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { GgmlFusedGateUpGeluMul(#a, #b, #c) }
        }
        I::GgmlFusedQkvRopeCache(a, b, c, d) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_bool(d);
            quote! { GgmlFusedQkvRopeCache(#a, #b, #c, #d) }
        }
        I::GgmlFusedQkvRopePrefill(a, b, c, d, e) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            quote! { GgmlFusedQkvRopePrefill(#a, #b, #c, #d, #e) }
        }
        I::Fp8Gemm(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { Fp8Gemm(#a, #b, #c) }
        }
        I::Fp8FusedGemmBias(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { Fp8FusedGemmBias(#a, #b, #c) }
        }
        I::Fp8FusedGateUpSiluMul(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { Fp8FusedGateUpSiluMul(#a, #b, #c) }
        }
        I::Fp8FusedGateUpGeluMul(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { Fp8FusedGateUpGeluMul(#a, #b, #c) }
        }
        I::Fp8FusedQkvRopeCache(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { Fp8FusedQkvRopeCache(#a, #b, #c) }
        }
        I::Fp8FusedQkvRopePrefill(a, b, c, d, e) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            quote! { Fp8FusedQkvRopePrefill(#a, #b, #c, #d, #e) }
        }
        I::Loop(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { Loop(#a, #b, #c) }
        }
        I::Alias(a, b) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            quote! { Alias(#a, #b) }
        }
        I::Free(a) => {
            let a = lit_u32(a);
            quote! { Free(#a) }
        }
        // Metal-only variants — only emitted on metal builds by the
        // per-arch lowering. The macro-static slice emission walks
        // them through this same path; emit field-for-field literals.
        I::MetalBiasAdd(a, b, c, d, e) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            quote! { MetalBiasAdd(#a, #b, #c, #d, #e) }
        }
        I::AttentionPrefillPaged(a, b, c, d) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { AttentionPrefillPaged(#a, #b, #c, #d) }
        }
        I::AffineQmm(a, b, c, d, e, f, g, h) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            let g = lit_u32(g);
            let h = lit_u32(h);
            quote! { AffineQmm(#a, #b, #c, #d, #e, #f, #g, #h) }
        }
        I::Nvfp4Qmm(a, b, c, d, e, f, g, h) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            let g = lit_u32(g);
            let h = lit_u32(h);
            quote! { Nvfp4Qmm(#a, #b, #c, #d, #e, #f, #g, #h) }
        }
        I::SynthPreAttn(a, b, c, d, e, f, g, h, i) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            let g = lit_u32(g);
            quote! { SynthPreAttn(#a, #b, #c, #d, #e, #f, #g, #h, #i) }
        }
        I::SynthMlpPreDown(a, b, c, d, e, f, g, h) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            let f = lit_u32(f);
            let g = lit_u32(g);
            quote! { SynthMlpPreDown(#a, #b, #c, #d, #e, #f, #g, #h) }
        }
        I::SiluMul(a, b, c, w) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let w = lit_u32(w);
            quote! { SiluMul(#a, #b, #c, #w) }
        }
        I::GeluMul(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { GeluMul(#a, #b, #c) }
        }
        I::SynthGateUpSiluMul(a, b, c, d, e, f) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            let e = lit_u32(e);
            quote! { SynthGateUpSiluMul(#a, #b, #c, #d, #e, #f) }
        }
        I::AffineEmbed(a, b, c) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            quote! { AffineEmbed(#a, #b, #c) }
        }
    }
}

/// A [`RowsExtern`](scratchy_forward_compiler::RowsExtern) as a const expression.
fn rows_extern_tokens(source: scratchy_forward_compiler::RowsExtern) -> TokenStream {
    use scratchy_forward_compiler::RowsExtern;
    match source {
        RowsExtern::Pixels => quote! { ::scratchy_forward_compiler::RowsExtern::Pixels },
        RowsExtern::PosEmbeds => quote! { ::scratchy_forward_compiler::RowsExtern::PosEmbeds },
        RowsExtern::TargetHidden => {
            quote! { ::scratchy_forward_compiler::RowsExtern::TargetHidden }
        }
    }
}

/// A [`KvOffsets`](scratchy_forward_compiler::KvOffsets) as a const expression.
fn kv_offsets_tokens(o: scratchy_forward_compiler::KvOffsets) -> TokenStream {
    use scratchy_forward_compiler::{BiasStorage, KvOffset};
    let one = |x: KvOffset| match x {
        KvOffset::Centered => quote! { ::scratchy_forward_compiler::KvOffset::Centered },
        KvOffset::LinearBias(BiasStorage::Dense) => quote! {
            ::scratchy_forward_compiler::KvOffset::LinearBias(
                ::scratchy_forward_compiler::BiasStorage::Dense)
        },
        KvOffset::LinearBias(BiasStorage::Affine) => quote! {
            ::scratchy_forward_compiler::KvOffset::LinearBias(
                ::scratchy_forward_compiler::BiasStorage::Affine)
        },
    };
    let (k, v) = (one(o.k), one(o.v));
    quote! { ::scratchy_forward_compiler::KvOffsets { k: #k, v: #v } }
}

// ── Per-bucket lowering ──────────────────────────────────────────

/// Output of lowering one (variant × workload-point). The codegen
/// stitches these into the per-bucket forward fn body.
pub struct LoweredBucket {
    /// Op instructions in execution order, including `Free` rows
    /// emitted by the drop pass at the same scheduling points the
    /// old codegen would have emitted `drop()` statements.
    pub instances: Vec<Instruction>,
    /// Per-instruction list of weight slots consumed by that op
    /// position. Same length as `instances`. The per-arch
    /// `WeightAccessors` impl walks `(bucket, op_idx, slot)` against
    /// this parallel array to emit match arms — `op_idx` is the index
    /// into `instances`.
    pub weight_slots: Vec<Vec<WeightSlot>>,
    /// One `barrier_before` flag per entry in `instances`. `true`
    /// means a concurrency-aware backend (today: metal MTL4
    /// encoder) must serialize the dispatched instance against
    /// every prior instance in the bucket. Computed at macro time —
    /// runtime never re-derives. Read only by metal's codegen — under
    /// `spyre` the tape path writes this field but never reads it, and
    /// that is EXPECTED.
    #[cfg_attr(feature = "spyre", expect(dead_code))]
    pub barriers: Vec<bool>,
}

/// Emit one per-bucket
/// `static <ident>: &[__I] = &[…];` where `__I` is the per-canonical
/// alias for `::scratchy_forward_compiler::Instruction`. Each row is
/// `<Variant>(v0, v1, …)` — tuple-style construction matching the
/// variant declaration order in `Instruction`; the variant + field
/// arity comes directly from the typed `Instruction` value.
pub fn emit_bucket_static_slice(
    static_ident: &syn::Ident,
    instances: &[Instruction],
) -> TokenStream {
    let elements = instances.iter().map(instruction_to_tokens);
    quote! {
        static #static_ident: &[__I] = &[ #(#elements),* ];
    }
}
