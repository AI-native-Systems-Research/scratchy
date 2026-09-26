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

use std::collections::{BTreeMap, HashMap, HashSet};

use proc_macro2::TokenStream;
use quote::quote;

use scratchy_forward_compiler::Instruction;

use crate::assignment::{Assignment, SubgraphId};
use crate::classified::{ExternKind, Program, UnrollIndex};
use crate::config::ModelParams;
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::{ImplementationLibrary, MatchInfo};
use crate::schedule::Loop;
use crate::shape::Shape;
use crate::weight_vocab::{OpcodeShape, SlotMap, WeightSlot};

// ── Instruction helpers ──────────────────────────────────────────
//
// `Instruction` (from scratchy-forward-compiler) is the universal typed
// fan_out output. The codegen walker calls `Implementation::fan_out`
// to get a `Vec<Instruction>`, and `Implementation::required_weights`
// alongside to get the parallel `Vec<WeightSlot>` for the per-arch
// `WeightAccessors` impl.
//
// These helpers render an `Instruction` back to TokenStream form for
// the static-slice emit, and extract scalar fields by position for
// the loop-detection / loop-compression passes.

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
        #[cfg(feature = "nccl")]
        I::AllReduce(a) => {
            let a = lit_u32(a);
            quote! { AllReduce(#a) }
        }
        #[cfg(feature = "nccl")]
        I::AllGather(a, b) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            quote! { AllGather(#a, #b) }
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
        I::LoadPixels(a) => {
            let a = lit_u32(a);
            quote! { LoadPixels(#a) }
        }
        I::LoadPosEmbeds(a) => {
            let a = lit_u32(a);
            quote! { LoadPosEmbeds(#a) }
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
        #[cfg(fa3_built)]
        I::FlashAttention3Decode(a, b, c, d) => {
            let a = lit_u32(a);
            let b = lit_u32(b);
            let c = lit_u32(c);
            let d = lit_u32(d);
            quote! { FlashAttention3Decode(#a, #b, #c, #d) }
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

/// Variant ident as it appears on the Rust enum and in
/// `OpcodeShape::name`. Used by the loop-detection pass to look up
/// per-variant iter-index field positions.
pub fn instruction_variant_name(inst: &Instruction) -> &'static str {
    use scratchy_forward_compiler::Instruction as I;
    match inst {
        I::Embed(..) => "Embed",
        I::RmsNorm(..) => "RmsNorm",
        I::MeanSubRmsNorm(..) => "MeanSubRmsNorm",
        I::MeanSubRmsNormBiasAdd(..) => "MeanSubRmsNormBiasAdd",
        I::Reshape(..) => "Reshape",
        I::Add(..) => "Add",
        #[cfg(feature = "nccl")]
        I::AllReduce(..) => "AllReduce",
        #[cfg(feature = "nccl")]
        I::AllGather(..) => "AllGather",
        I::SpliceMmEmbeds(..) => "SpliceMmEmbeds",
        I::ScalarMul(..) => "ScalarMul",
        I::TanhSoftCap(..) => "TanhSoftCap",
        I::RmsNormUnit(..) => "RmsNormUnit",
        I::ScalarWeightMul(..) => "ScalarWeightMul",
        I::NormAddScalarMul(..) => "NormAddScalarMul",
        I::FusedAddRmsNorm(..) => "FusedAddRmsNorm",
        I::FusedAddRmsNormWithOffset(..) => "FusedAddRmsNormWithOffset",
        I::ScalarOffsetRmsNorm(..) => "ScalarOffsetRmsNorm",
        I::CutlassFusedRmsNormGemm(..) => "CutlassFusedRmsNormGemm",
        I::CutlassFusedMeanSubRmsNormGemm(..) => "CutlassFusedMeanSubRmsNormGemm",
        I::CutlassFusedAddRmsNormGemm(..) => "CutlassFusedAddRmsNormGemm",
        I::Gemm(..) => "Gemm",
        I::FusedCublasGemmAdd(..) => "FusedCublasGemmAdd",
        I::FusedGemmBias(..) => "FusedGemmBias",
        I::FusedGateUpSiluMul(..) => "FusedGateUpSiluMul",
        I::FusedGateUpGeluMul(..) => "FusedGateUpGeluMul",
        I::FusedQkvRopeCache(..) => "FusedQkvRopeCache",
        I::FusedQkvQkNormRopeCache(..) => "FusedQkvQkNormRopeCache",
        I::FusedQkvRopePrefill(..) => "FusedQkvRopePrefill",
        I::AttentionViaCache(..) => "AttentionViaCache",
        I::AttentionPrefillContiguous(..) => "AttentionPrefillContiguous",
        I::EncoderAttention(..) => "EncoderAttention",
        I::SlidingAttentionViaCache(..) => "SlidingAttentionViaCache",
        I::SlidingAttentionPrefillContiguous(..) => "SlidingAttentionPrefillContiguous",
        I::SlidingAttentionPrefillPaged(..) => "SlidingAttentionPrefillPaged",
        I::VarlenAttention(..) => "VarlenAttention",
        I::VisionRope(..) => "VisionRope",
        I::QuickGelu(..) => "QuickGelu",
        I::Gelu(..) => "Gelu",
        I::PosEmbed(..) => "PosEmbed",
        I::LoadPixels(..) => "LoadPixels",
        I::LoadPosEmbeds(..) => "LoadPosEmbeds",
        I::GeluErf(..) => "GeluErf",
        I::EmbeddingGather(..) => "EmbeddingGather",
        I::AvgPool2d(..) => "AvgPool2d",
        I::StripCls(..) => "StripCls",
        I::FlashInferAttentionDecode(..) => "FlashInferAttentionDecode",
        #[cfg(fa3_built)]
        I::FlashAttention3Decode(..) => "FlashAttention3Decode",
        I::FlashInferAttentionPrefill(..) => "FlashInferAttentionPrefill",
        I::RopeAppend(..) => "RopeAppend",
        I::RopeAppendNormed(..) => "RopeAppendNormed",
        I::MlaSplit(..) => "MlaSplit",
        I::MlaAttention(..) => "MlaAttention",
        I::GatedDeltaNet(..) => "GatedDeltaNet",
        I::GateSplit(..) => "GateSplit",
        I::GateApply(..) => "GateApply",
        I::GateScale(..) => "GateScale",
        I::DeepSeekMoe(..) => "DeepSeekMoe",
        I::DeepSeekMoeFp8Block(..) => "DeepSeekMoeFp8Block",
        I::DeepSeekMoeGgml(..) => "DeepSeekMoeGgml",
        I::FusedMoe(..) => "FusedMoe",
        I::SharedFusedMoe(..) => "SharedFusedMoe",
        I::MetalFusedMoe(..) => "MetalFusedMoe",
        I::MetalSharedFusedMoe(..) => "MetalSharedFusedMoe",
        I::GemmaMoe(..) => "GemmaMoe",
        I::CutlassGemm(..) => "CutlassGemm",
        I::CutlassGemmSplitK(..) => "CutlassGemmSplitK",
        I::CutlassGemmAdd(..) => "CutlassGemmAdd",
        I::CutlassGemv(..) => "CutlassGemv",
        I::CutlassFusedGemmBias(..) => "CutlassFusedGemmBias",
        I::CutlassFusedGateUpSiluMul(..) => "CutlassFusedGateUpSiluMul",
        I::CutlassFusedGateUpGeluMul(..) => "CutlassFusedGateUpGeluMul",
        I::CutlassFusedQkvRopeCache(..) => "CutlassFusedQkvRopeCache",
        I::CutlassFusedQkvRopePrefill(..) => "CutlassFusedQkvRopePrefill",
        I::MarlinGemm(..) => "MarlinGemm",
        I::MarlinFusedGateUpSiluMul(..) => "MarlinFusedGateUpSiluMul",
        I::MarlinFusedGateUpGeluMul(..) => "MarlinFusedGateUpGeluMul",
        I::MarlinFusedQkvRopeCache(..) => "MarlinFusedQkvRopeCache",
        I::MarlinFusedQkvRopePrefill(..) => "MarlinFusedQkvRopePrefill",
        I::Bnb4Gemm(..) => "Bnb4Gemm",
        I::Bnb4FusedGateUpSiluMul(..) => "Bnb4FusedGateUpSiluMul",
        I::Bnb4FusedGateUpGeluMul(..) => "Bnb4FusedGateUpGeluMul",
        I::Bnb4FusedQkvRopeCache(..) => "Bnb4FusedQkvRopeCache",
        I::Bnb4FusedQkvRopePrefill(..) => "Bnb4FusedQkvRopePrefill",
        I::GgmlGemm(..) => "GgmlGemm",
        I::GgmlFusedGateUpSiluMul(..) => "GgmlFusedGateUpSiluMul",
        I::GgmlFusedGateUpGeluMul(..) => "GgmlFusedGateUpGeluMul",
        I::GgmlFusedQkvRopeCache(..) => "GgmlFusedQkvRopeCache",
        I::GgmlFusedQkvRopePrefill(..) => "GgmlFusedQkvRopePrefill",
        I::Fp8Gemm(..) => "Fp8Gemm",
        I::Fp8FusedGemmBias(..) => "Fp8FusedGemmBias",
        I::Fp8FusedGateUpSiluMul(..) => "Fp8FusedGateUpSiluMul",
        I::Fp8FusedGateUpGeluMul(..) => "Fp8FusedGateUpGeluMul",
        I::Fp8FusedQkvRopeCache(..) => "Fp8FusedQkvRopeCache",
        I::Fp8FusedQkvRopePrefill(..) => "Fp8FusedQkvRopePrefill",
        I::Loop(..) => "Loop",
        I::Alias(..) => "Alias",
        I::Free(..) => "Free",
        I::MetalBiasAdd(..) => "MetalBiasAdd",
        I::AttentionPrefillPaged(..) => "AttentionPrefillPaged",
        I::AffineQmm(..) => "AffineQmm",
        I::Nvfp4Qmm(..) => "Nvfp4Qmm",
        I::SynthPreAttn(..) => "SynthPreAttn",
        I::SynthMlpPreDown(..) => "SynthMlpPreDown",
        I::SiluMul(..) => "SiluMul",
        I::GeluMul(..) => "GeluMul",
        I::SynthGateUpSiluMul(..) => "SynthGateUpSiluMul",
        I::AffineEmbed(..) => "AffineEmbed",
    }
}

/// Extract the field at position `idx` as a `u64` for loop-detection
/// purposes (which only ever needs to compare scalar layer-style
/// fields). Returns `None` for fields that aren't a single scalar
/// integer (arrays, bools, floats — none of which a sane iter-index
/// would ever be), and for out-of-range indices.
pub fn instruction_field_at(inst: &Instruction, idx: usize) -> Option<u64> {
    use scratchy_forward_compiler::Instruction as I;
    let u = |v: u32| Some(v as u64);
    let u8v = |v: u8| Some(v as u64);
    match *inst {
        I::Embed(a) => match idx {
            0 => u(a),
            _ => None,
        },
        I::RmsNorm(a, b, c, d, e) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            _ => None,
        },
        I::MeanSubRmsNorm(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::MeanSubRmsNormBiasAdd(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::Reshape(a, b, _, _, _, f) => match idx {
            0 => u(a),
            1 => u(b),
            5 => u8v(f),
            _ => None,
        },
        I::Add(a, b) => match idx {
            0 => u(a),
            1 => u(b),
            _ => None,
        },
        #[cfg(feature = "nccl")]
        I::AllReduce(a) => match idx {
            0 => u(a),
            _ => None,
        },
        #[cfg(feature = "nccl")]
        I::AllGather(a, b) => match idx {
            0 => u(a),
            1 => u(b),
            _ => None,
        },
        I::SpliceMmEmbeds(a) => match idx {
            0 => u(a),
            _ => None,
        },
        I::ScalarMul(a, b, _) => match idx {
            0 => u(a),
            1 => u(b),
            _ => None,
        },
        I::TanhSoftCap(a, b) => match idx {
            0 => u(a),
            1 => u(b),
            _ => None,
        },
        I::RmsNormUnit(a, b, c, d) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            _ => None,
        },
        I::NormAddScalarMul(a, b, c, d, e) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            _ => None,
        },
        I::ScalarWeightMul(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::FusedAddRmsNorm(a, b, c, d, e) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            _ => None,
        },
        I::FusedAddRmsNormWithOffset(a, b, c, _) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::ScalarOffsetRmsNorm(a, b, c, _, _, _) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::CutlassFusedRmsNormGemm(a, b, c, d, e, f, g, h) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            5 => u(f),
            6 => u(g),
            7 => u(h),
            _ => None,
        },
        I::CutlassFusedMeanSubRmsNormGemm(a, b, c, d, e, f, g, h) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            5 => u(f),
            6 => u(g),
            7 => u(h),
            _ => None,
        },
        I::CutlassFusedAddRmsNormGemm(a, b, c, d, e, f, g, h, i) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            5 => u(f),
            6 => u(g),
            7 => u(h),
            8 => u(i),
            _ => None,
        },
        I::Gemm(a, b, c, d, e) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            _ => None,
        },
        I::FusedCublasGemmAdd(a, b, c, d, e) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            _ => None,
        },
        I::FusedGemmBias(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::FusedGateUpSiluMul(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::FusedGateUpGeluMul(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::FusedQkvRopeCache(a, b, c, _, _) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::FusedQkvQkNormRopeCache(a, b, c, _, _) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::FusedQkvRopePrefill(a, b, c, d, e, _, _) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            _ => None,
        },
        I::AttentionViaCache(a, b, c, _) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::AttentionPrefillContiguous(a, b, c, d, _) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            _ => None,
        },
        I::EncoderAttention(a, b, c, d) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            _ => None,
        },
        I::SlidingAttentionViaCache(a, b, c, _) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::SlidingAttentionPrefillPaged(a, b, c, _) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::SlidingAttentionPrefillContiguous(a, b, c, d, _) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            _ => None,
        },
        I::VarlenAttention(a, b, c, d, e) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u8v(e),
            _ => None,
        },
        I::VisionRope(a, b, c, d) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            _ => None,
        },
        I::QuickGelu(a, b) => match idx {
            0 => u(a),
            1 => u(b),
            _ => None,
        },
        I::Gelu(a, b) => match idx {
            0 => u(a),
            1 => u(b),
            _ => None,
        },
        I::PosEmbed(a) => match idx {
            0 => u(a),
            _ => None,
        },
        I::LoadPixels(a) => match idx {
            0 => u(a),
            _ => None,
        },
        I::LoadPosEmbeds(a) => match idx {
            0 => u(a),
            _ => None,
        },
        I::GeluErf(a, b) => match idx {
            0 => u(a),
            1 => u(b),
            _ => None,
        },
        I::EmbeddingGather(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u8v(c),
            _ => None,
        },
        I::AvgPool2d(a, b) => match idx {
            0 => u(a),
            1 => u(b),
            _ => None,
        },
        I::StripCls(a, b) => match idx {
            0 => u(a),
            1 => u(b),
            _ => None,
        },
        I::FlashInferAttentionDecode(a, b, c, d, _) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            _ => None,
        },
        #[cfg(fa3_built)]
        I::FlashAttention3Decode(a, b, c, d) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            _ => None,
        },
        I::FlashInferAttentionPrefill(a, b, c, d, e, f, _) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            5 => u(f),
            _ => None,
        },
        I::RopeAppendNormed(a, b, c, d, e, f, g, _, _) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            5 => u(f),
            6 => u(g),
            _ => None,
        },
        I::RopeAppend(a, b, c, d, e, f, g, _, _, _) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            5 => u(f),
            6 => u(g),
            _ => None,
        },
        I::MlaSplit(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::MlaAttention(a, b, c, d, e) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            _ => None,
        },
        I::GatedDeltaNet(a, b, c, d, e, f) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            5 => u(f),
            _ => None,
        },
        I::GateSplit(a, b, c) | I::GateApply(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::GateScale(a, b, c, d) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            _ => None,
        },
        I::DeepSeekMoe(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::DeepSeekMoeFp8Block(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::DeepSeekMoeGgml(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::FusedMoe(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::SharedFusedMoe(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::MetalFusedMoe(a, b, c, d, e, f, g, h, i) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            5 => u(f),
            6 => u(g),
            7 => u(h),
            8 => u(i),
            _ => None,
        },
        I::MetalSharedFusedMoe(a, b, c, d, e, f, g, h, i, j, k) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            5 => u(f),
            6 => u(g),
            7 => u(h),
            8 => u(i),
            9 => u(j),
            10 => Some(if k { 1 } else { 0 }),
            _ => None,
        },
        I::GemmaMoe(a, b, c, d, e, f, g, h, i, j) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            5 => u(f),
            6 => u(g),
            7 => u(h),
            8 => u(i),
            9 => u(j),
            _ => None,
        },
        I::CutlassGemm(a, b, c, d, e, f, g, h) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            5 => u(f),
            6 => u(g),
            7 => u(h),
            _ => None,
        },
        I::CutlassGemmSplitK(a, b, c, d, e, f, g, h, i) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            5 => u(f),
            6 => u(g),
            7 => u(h),
            8 => u(i),
            _ => None,
        },
        I::CutlassGemmAdd(a, b, c, d, e, f, g, h) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            5 => u(f),
            6 => u(g),
            7 => u(h),
            _ => None,
        },
        I::CutlassGemv(a, b, c, d, e) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            _ => None,
        },
        I::CutlassFusedGemmBias(a, b, c, d, e, f, g, h) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            5 => u(f),
            6 => u(g),
            7 => u(h),
            _ => None,
        },
        I::CutlassFusedGateUpSiluMul(a, b, c, d, e, f) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            5 => u(f),
            _ => None,
        },
        I::CutlassFusedGateUpGeluMul(a, b, c, d, e, f, g, h) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            5 => u(f),
            6 => u(g),
            7 => u(h),
            _ => None,
        },
        I::CutlassFusedQkvRopeCache(a, b, c, _, e, f, g, h, i) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            4 => u(e),
            5 => u(f),
            6 => u(g),
            7 => u(h),
            8 => u(i),
            _ => None,
        },
        I::CutlassFusedQkvRopePrefill(a, b, c, d, e, _, g, h, i, j, k) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            6 => u(g),
            7 => u(h),
            8 => u(i),
            9 => u(j),
            10 => u(k),
            _ => None,
        },
        I::MarlinGemm(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::MarlinFusedGateUpSiluMul(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::MarlinFusedGateUpGeluMul(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::MarlinFusedQkvRopeCache(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::MarlinFusedQkvRopePrefill(a, b, c, d, e) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            _ => None,
        },
        I::Bnb4Gemm(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::Bnb4FusedGateUpSiluMul(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::Bnb4FusedGateUpGeluMul(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::Bnb4FusedQkvRopeCache(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::Bnb4FusedQkvRopePrefill(a, b, c, d, e) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            _ => None,
        },
        I::GgmlGemm(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::GgmlFusedGateUpSiluMul(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::GgmlFusedGateUpGeluMul(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::GgmlFusedQkvRopeCache(a, b, c, _) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::GgmlFusedQkvRopePrefill(a, b, c, d, e) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            _ => None,
        },
        I::Fp8Gemm(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::Fp8FusedGemmBias(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::Fp8FusedGateUpSiluMul(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::Fp8FusedGateUpGeluMul(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::Fp8FusedQkvRopeCache(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::Fp8FusedQkvRopePrefill(a, b, c, d, e) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            _ => None,
        },
        I::Loop(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::Alias(a, b) => match idx {
            0 => u(a),
            1 => u(b),
            _ => None,
        },
        I::Free(a) => match idx {
            0 => u(a),
            _ => None,
        },
        I::MetalBiasAdd(a, b, c, d, _e) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            _ => None,
        },
        I::AttentionPrefillPaged(a, b, c, _d) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::AffineQmm(a, b, c, d, e, f, g, h) | I::Nvfp4Qmm(a, b, c, d, e, f, g, h) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            5 => u(f),
            6 => u(g),
            7 => u(h),
            _ => None,
        },
        I::SynthPreAttn(a, b, c, d, e, f, g, _h, _i) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            5 => u(f),
            6 => u(g),
            _ => None,
        },
        I::SynthMlpPreDown(a, b, c, d, e, f, g, _h) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            5 => u(f),
            6 => u(g),
            _ => None,
        },
        I::SiluMul(a, b, c, _w) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::GeluMul(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
        I::SynthGateUpSiluMul(a, b, c, d, e, _f) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            3 => u(d),
            4 => u(e),
            _ => None,
        },
        I::AffineEmbed(a, b, c) => match idx {
            0 => u(a),
            1 => u(b),
            2 => u(c),
            _ => None,
        },
    }
}

/// Replace the field at position `idx` (interpreted as u32) with
/// `new_val`. Used by `apply_loop_compression` to set per-row
/// baselines for the iter-index field. Panics if `idx` is invalid
/// for this variant or if the field at that position is not u32.
pub fn instruction_with_field_set(inst: Instruction, idx: usize, new_val: u32) -> Instruction {
    use scratchy_forward_compiler::Instruction as I;
    let n = new_val;
    match inst {
        I::Embed(_a) => match idx {
            0 => I::Embed(n),
            _ => panic!("Embed: bad idx {idx}"),
        },
        I::RmsNorm(a, b, c, d, e) => match idx {
            0 => I::RmsNorm(n, b, c, d, e),
            1 => I::RmsNorm(a, n, c, d, e),
            2 => I::RmsNorm(a, b, n, d, e),
            3 => I::RmsNorm(a, b, c, n, e),
            4 => I::RmsNorm(a, b, c, d, n),
            _ => panic!("RmsNorm: bad idx {idx}"),
        },
        I::MeanSubRmsNorm(a, b, c) => match idx {
            0 => I::MeanSubRmsNorm(n, b, c),
            1 => I::MeanSubRmsNorm(a, n, c),
            2 => I::MeanSubRmsNorm(a, b, n),
            _ => panic!("MeanSubRmsNorm: bad idx {idx}"),
        },
        I::MeanSubRmsNormBiasAdd(a, b, c) => match idx {
            0 => I::MeanSubRmsNormBiasAdd(n, b, c),
            1 => I::MeanSubRmsNormBiasAdd(a, n, c),
            2 => I::MeanSubRmsNormBiasAdd(a, b, n),
            _ => panic!("MeanSubRmsNormBiasAdd: bad idx {idx}"),
        },
        I::Reshape(a, b, c, d, e, f) => match idx {
            0 => I::Reshape(n, b, c, d, e, f),
            1 => I::Reshape(a, n, c, d, e, f),
            _ => panic!("Reshape: bad idx {idx}"),
        },
        I::Add(a, b) => match idx {
            0 => I::Add(n, b),
            1 => I::Add(a, n),
            _ => panic!("Add: bad idx {idx}"),
        },
        #[cfg(feature = "nccl")]
        I::AllReduce(_a) => match idx {
            0 => I::AllReduce(n),
            _ => panic!("AllReduce: bad idx {idx}"),
        },
        #[cfg(feature = "nccl")]
        I::AllGather(a, b) => match idx {
            0 => I::AllGather(n, b),
            1 => I::AllGather(a, n),
            _ => panic!("AllGather: bad idx {idx}"),
        },
        I::SpliceMmEmbeds(_a) => match idx {
            0 => I::SpliceMmEmbeds(n),
            _ => panic!("SpliceMmEmbeds: bad idx {idx}"),
        },
        I::ScalarMul(a, b, c) => match idx {
            0 => I::ScalarMul(n, b, c),
            1 => I::ScalarMul(a, n, c),
            _ => panic!("ScalarMul: bad idx {idx}"),
        },
        I::TanhSoftCap(a, b) => match idx {
            0 => I::TanhSoftCap(n, b),
            1 => I::TanhSoftCap(a, n),
            _ => panic!("TanhSoftCap: bad idx {idx}"),
        },
        I::RmsNormUnit(a, b, c, d) => match idx {
            0 => I::RmsNormUnit(n, b, c, d),
            1 => I::RmsNormUnit(a, n, c, d),
            2 => I::RmsNormUnit(a, b, n, d),
            3 => I::RmsNormUnit(a, b, c, n),
            _ => panic!("RmsNormUnit: bad idx {idx}"),
        },
        I::ScalarWeightMul(a, b, c) => match idx {
            0 => I::ScalarWeightMul(n, b, c),
            1 => I::ScalarWeightMul(a, n, c),
            2 => I::ScalarWeightMul(a, b, n),
            _ => panic!("ScalarWeightMul: bad idx {idx}"),
        },
        I::NormAddScalarMul(a, b, c, d, e) => match idx {
            0 => I::NormAddScalarMul(n, b, c, d, e),
            1 => I::NormAddScalarMul(a, n, c, d, e),
            2 => I::NormAddScalarMul(a, b, n, d, e),
            3 => I::NormAddScalarMul(a, b, c, n, e),
            4 => I::NormAddScalarMul(a, b, c, d, n),
            _ => panic!("NormAddScalarMul: bad idx {idx}"),
        },
        I::FusedAddRmsNorm(a, b, c, d, e) => match idx {
            0 => I::FusedAddRmsNorm(n, b, c, d, e),
            1 => I::FusedAddRmsNorm(a, n, c, d, e),
            2 => I::FusedAddRmsNorm(a, b, n, d, e),
            3 => I::FusedAddRmsNorm(a, b, c, n, e),
            4 => I::FusedAddRmsNorm(a, b, c, d, n),
            _ => panic!("FusedAddRmsNorm: bad idx {idx}"),
        },
        I::FusedAddRmsNormWithOffset(a, b, c, d) => match idx {
            0 => I::FusedAddRmsNormWithOffset(n, b, c, d),
            1 => I::FusedAddRmsNormWithOffset(a, n, c, d),
            2 => I::FusedAddRmsNormWithOffset(a, b, n, d),
            _ => panic!("FusedAddRmsNormWithOffset: bad idx {idx}"),
        },
        I::ScalarOffsetRmsNorm(a, b, c, d, e, f) => match idx {
            0 => I::ScalarOffsetRmsNorm(n, b, c, d, e, f),
            1 => I::ScalarOffsetRmsNorm(a, n, c, d, e, f),
            2 => I::ScalarOffsetRmsNorm(a, b, n, d, e, f),
            _ => panic!("ScalarOffsetRmsNorm: bad idx {idx}"),
        },
        I::CutlassFusedRmsNormGemm(a, b, c, d, e, f, g, h) => match idx {
            0 => I::CutlassFusedRmsNormGemm(n, b, c, d, e, f, g, h),
            1 => I::CutlassFusedRmsNormGemm(a, n, c, d, e, f, g, h),
            2 => I::CutlassFusedRmsNormGemm(a, b, n, d, e, f, g, h),
            3 => I::CutlassFusedRmsNormGemm(a, b, c, n, e, f, g, h),
            4 => I::CutlassFusedRmsNormGemm(a, b, c, d, n, f, g, h),
            5 => I::CutlassFusedRmsNormGemm(a, b, c, d, e, n, g, h),
            6 => I::CutlassFusedRmsNormGemm(a, b, c, d, e, f, n, h),
            7 => I::CutlassFusedRmsNormGemm(a, b, c, d, e, f, g, n),
            _ => panic!("CutlassFusedRmsNormGemm: bad idx {idx}"),
        },
        I::CutlassFusedMeanSubRmsNormGemm(a, b, c, d, e, f, g, h) => match idx {
            0 => I::CutlassFusedMeanSubRmsNormGemm(n, b, c, d, e, f, g, h),
            1 => I::CutlassFusedMeanSubRmsNormGemm(a, n, c, d, e, f, g, h),
            2 => I::CutlassFusedMeanSubRmsNormGemm(a, b, n, d, e, f, g, h),
            3 => I::CutlassFusedMeanSubRmsNormGemm(a, b, c, n, e, f, g, h),
            4 => I::CutlassFusedMeanSubRmsNormGemm(a, b, c, d, n, f, g, h),
            5 => I::CutlassFusedMeanSubRmsNormGemm(a, b, c, d, e, n, g, h),
            6 => I::CutlassFusedMeanSubRmsNormGemm(a, b, c, d, e, f, n, h),
            7 => I::CutlassFusedMeanSubRmsNormGemm(a, b, c, d, e, f, g, n),
            _ => panic!("CutlassFusedMeanSubRmsNormGemm: bad idx {idx}"),
        },
        I::CutlassFusedAddRmsNormGemm(a, b, c, d, e, f, g, h, i) => match idx {
            0 => I::CutlassFusedAddRmsNormGemm(n, b, c, d, e, f, g, h, i),
            1 => I::CutlassFusedAddRmsNormGemm(a, n, c, d, e, f, g, h, i),
            2 => I::CutlassFusedAddRmsNormGemm(a, b, n, d, e, f, g, h, i),
            3 => I::CutlassFusedAddRmsNormGemm(a, b, c, n, e, f, g, h, i),
            4 => I::CutlassFusedAddRmsNormGemm(a, b, c, d, n, f, g, h, i),
            5 => I::CutlassFusedAddRmsNormGemm(a, b, c, d, e, n, g, h, i),
            6 => I::CutlassFusedAddRmsNormGemm(a, b, c, d, e, f, n, h, i),
            7 => I::CutlassFusedAddRmsNormGemm(a, b, c, d, e, f, g, n, i),
            8 => I::CutlassFusedAddRmsNormGemm(a, b, c, d, e, f, g, h, n),
            _ => panic!("CutlassFusedAddRmsNormGemm: bad idx {idx}"),
        },
        I::Gemm(a, b, c, d, e) => match idx {
            0 => I::Gemm(n, b, c, d, e),
            1 => I::Gemm(a, n, c, d, e),
            2 => I::Gemm(a, b, n, d, e),
            3 => I::Gemm(a, b, c, n, e),
            4 => I::Gemm(a, b, c, d, n),
            _ => panic!("Gemm: bad idx {idx}"),
        },
        I::FusedCublasGemmAdd(a, b, c, d, e) => match idx {
            0 => I::FusedCublasGemmAdd(n, b, c, d, e),
            1 => I::FusedCublasGemmAdd(a, n, c, d, e),
            2 => I::FusedCublasGemmAdd(a, b, n, d, e),
            3 => I::FusedCublasGemmAdd(a, b, c, n, e),
            4 => I::FusedCublasGemmAdd(a, b, c, d, n),
            _ => panic!("FusedCublasGemmAdd: bad idx {idx}"),
        },
        I::FusedGemmBias(a, b, c) => match idx {
            0 => I::FusedGemmBias(n, b, c),
            1 => I::FusedGemmBias(a, n, c),
            2 => I::FusedGemmBias(a, b, n),
            _ => panic!("FusedGemmBias: bad idx {idx}"),
        },
        I::FusedGateUpSiluMul(a, b, c) => match idx {
            0 => I::FusedGateUpSiluMul(n, b, c),
            1 => I::FusedGateUpSiluMul(a, n, c),
            2 => I::FusedGateUpSiluMul(a, b, n),
            _ => panic!("FusedGateUpSiluMul: bad idx {idx}"),
        },
        I::FusedGateUpGeluMul(a, b, c) => match idx {
            0 => I::FusedGateUpGeluMul(n, b, c),
            1 => I::FusedGateUpGeluMul(a, n, c),
            2 => I::FusedGateUpGeluMul(a, b, n),
            _ => panic!("FusedGateUpGeluMul: bad idx {idx}"),
        },
        I::FusedQkvRopeCache(a, b, c, d, e) => match idx {
            0 => I::FusedQkvRopeCache(n, b, c, d, e),
            1 => I::FusedQkvRopeCache(a, n, c, d, e),
            2 => I::FusedQkvRopeCache(a, b, n, d, e),
            _ => panic!("FusedQkvRopeCache: bad idx {idx}"),
        },
        I::FusedQkvQkNormRopeCache(a, b, c, d, e) => match idx {
            0 => I::FusedQkvQkNormRopeCache(n, b, c, d, e),
            1 => I::FusedQkvQkNormRopeCache(a, n, c, d, e),
            2 => I::FusedQkvQkNormRopeCache(a, b, n, d, e),
            _ => panic!("FusedQkvQkNormRopeCache: bad idx {idx}"),
        },
        I::FusedQkvRopePrefill(a, b, c, d, e, f, g) => match idx {
            0 => I::FusedQkvRopePrefill(n, b, c, d, e, f, g),
            1 => I::FusedQkvRopePrefill(a, n, c, d, e, f, g),
            2 => I::FusedQkvRopePrefill(a, b, n, d, e, f, g),
            3 => I::FusedQkvRopePrefill(a, b, c, n, e, f, g),
            4 => I::FusedQkvRopePrefill(a, b, c, d, n, f, g),
            _ => panic!("FusedQkvRopePrefill: bad idx {idx}"),
        },
        I::AttentionViaCache(a, b, c, d) => match idx {
            0 => I::AttentionViaCache(n, b, c, d),
            1 => I::AttentionViaCache(a, n, c, d),
            2 => I::AttentionViaCache(a, b, n, d),
            _ => panic!("AttentionViaCache: bad idx {idx}"),
        },
        I::AttentionPrefillContiguous(a, b, c, d, e) => match idx {
            0 => I::AttentionPrefillContiguous(n, b, c, d, e),
            1 => I::AttentionPrefillContiguous(a, n, c, d, e),
            2 => I::AttentionPrefillContiguous(a, b, n, d, e),
            3 => I::AttentionPrefillContiguous(a, b, c, n, e),
            _ => panic!("AttentionPrefillContiguous: bad idx {idx}"),
        },
        I::EncoderAttention(a, b, c, d) => match idx {
            0 => I::EncoderAttention(n, b, c, d),
            1 => I::EncoderAttention(a, n, c, d),
            2 => I::EncoderAttention(a, b, n, d),
            3 => I::EncoderAttention(a, b, c, n),
            _ => panic!("EncoderAttention: bad idx {idx}"),
        },
        I::SlidingAttentionViaCache(a, b, c, d) => match idx {
            0 => I::SlidingAttentionViaCache(n, b, c, d),
            1 => I::SlidingAttentionViaCache(a, n, c, d),
            2 => I::SlidingAttentionViaCache(a, b, n, d),
            _ => panic!("SlidingAttentionViaCache: bad idx {idx}"),
        },
        I::SlidingAttentionPrefillContiguous(a, b, c, d, e) => match idx {
            0 => I::SlidingAttentionPrefillContiguous(n, b, c, d, e),
            1 => I::SlidingAttentionPrefillContiguous(a, n, c, d, e),
            2 => I::SlidingAttentionPrefillContiguous(a, b, n, d, e),
            3 => I::SlidingAttentionPrefillContiguous(a, b, c, n, e),
            _ => panic!("SlidingAttentionPrefillContiguous: bad idx {idx}"),
        },
        I::SlidingAttentionPrefillPaged(a, b, c, d) => match idx {
            0 => I::SlidingAttentionPrefillPaged(n, b, c, d),
            1 => I::SlidingAttentionPrefillPaged(a, n, c, d),
            2 => I::SlidingAttentionPrefillPaged(a, b, n, d),
            _ => panic!("SlidingAttentionPrefillPaged: bad idx {idx}"),
        },
        I::VarlenAttention(a, b, c, d, e) => match idx {
            0 => I::VarlenAttention(n, b, c, d, e),
            1 => I::VarlenAttention(a, n, c, d, e),
            2 => I::VarlenAttention(a, b, n, d, e),
            3 => I::VarlenAttention(a, b, c, n, e),
            _ => panic!("VarlenAttention: bad idx {idx}"),
        },
        I::VisionRope(a, b, c, d) => match idx {
            0 => I::VisionRope(n, b, c, d),
            1 => I::VisionRope(a, n, c, d),
            2 => I::VisionRope(a, b, n, d),
            3 => I::VisionRope(a, b, c, n),
            _ => panic!("VisionRope: bad idx {idx}"),
        },
        I::QuickGelu(a, b) => match idx {
            0 => I::QuickGelu(n, b),
            1 => I::QuickGelu(a, n),
            _ => panic!("QuickGelu: bad idx {idx}"),
        },
        I::Gelu(a, b) => match idx {
            0 => I::Gelu(n, b),
            1 => I::Gelu(a, n),
            _ => panic!("Gelu: bad idx {idx}"),
        },
        I::PosEmbed(_a) => match idx {
            0 => I::PosEmbed(n),
            _ => panic!("PosEmbed: bad idx {idx}"),
        },
        I::LoadPixels(_a) => match idx {
            0 => I::LoadPixels(n),
            _ => panic!("LoadPixels: bad idx {idx}"),
        },
        I::LoadPosEmbeds(_a) => match idx {
            0 => I::LoadPosEmbeds(n),
            _ => panic!("LoadPosEmbeds: bad idx {idx}"),
        },
        I::GeluErf(a, b) => match idx {
            0 => I::GeluErf(n, b),
            1 => I::GeluErf(a, n),
            _ => panic!("GeluErf: bad idx {idx}"),
        },
        I::EmbeddingGather(a, b, c) => match idx {
            0 => I::EmbeddingGather(n, b, c),
            1 => I::EmbeddingGather(a, n, c),
            _ => panic!("EmbeddingGather: bad idx {idx}"),
        },
        I::AvgPool2d(a, b) => match idx {
            0 => I::AvgPool2d(n, b),
            1 => I::AvgPool2d(a, n),
            _ => panic!("AvgPool2d: bad idx {idx}"),
        },
        I::StripCls(a, b) => match idx {
            0 => I::StripCls(n, b),
            1 => I::StripCls(a, n),
            _ => panic!("StripCls: bad idx {idx}"),
        },
        I::FlashInferAttentionDecode(a, b, c, d, e) => match idx {
            0 => I::FlashInferAttentionDecode(n, b, c, d, e),
            1 => I::FlashInferAttentionDecode(a, n, c, d, e),
            2 => I::FlashInferAttentionDecode(a, b, n, d, e),
            3 => I::FlashInferAttentionDecode(a, b, c, n, e),
            _ => panic!("FlashInferAttentionDecode: bad idx {idx}"),
        },
        #[cfg(fa3_built)]
        I::FlashAttention3Decode(a, b, c, d) => match idx {
            0 => I::FlashAttention3Decode(n, b, c, d),
            1 => I::FlashAttention3Decode(a, n, c, d),
            2 => I::FlashAttention3Decode(a, b, n, d),
            3 => I::FlashAttention3Decode(a, b, c, n),
            _ => panic!("FlashAttention3Decode: bad idx {idx}"),
        },
        I::FlashInferAttentionPrefill(a, b, c, d, e, f, g) => match idx {
            0 => I::FlashInferAttentionPrefill(n, b, c, d, e, f, g),
            1 => I::FlashInferAttentionPrefill(a, n, c, d, e, f, g),
            2 => I::FlashInferAttentionPrefill(a, b, n, d, e, f, g),
            3 => I::FlashInferAttentionPrefill(a, b, c, n, e, f, g),
            4 => I::FlashInferAttentionPrefill(a, b, c, d, n, f, g),
            5 => I::FlashInferAttentionPrefill(a, b, c, d, e, n, g),
            _ => panic!("FlashInferAttentionPrefill: bad idx {idx}"),
        },
        I::RopeAppendNormed(a, b, c, d, e, f, g, h, i) => match idx {
            0 => I::RopeAppendNormed(n, b, c, d, e, f, g, h, i),
            1 => I::RopeAppendNormed(a, n, c, d, e, f, g, h, i),
            2 => I::RopeAppendNormed(a, b, n, d, e, f, g, h, i),
            3 => I::RopeAppendNormed(a, b, c, n, e, f, g, h, i),
            4 => I::RopeAppendNormed(a, b, c, d, n, f, g, h, i),
            5 => I::RopeAppendNormed(a, b, c, d, e, n, g, h, i),
            6 => I::RopeAppendNormed(a, b, c, d, e, f, n, h, i),
            _ => panic!("RopeAppendNormed: bad idx {idx}"),
        },
        I::RopeAppend(a, b, c, d, e, f, g, h, i, j) => match idx {
            0 => I::RopeAppend(n, b, c, d, e, f, g, h, i, j),
            1 => I::RopeAppend(a, n, c, d, e, f, g, h, i, j),
            2 => I::RopeAppend(a, b, n, d, e, f, g, h, i, j),
            3 => I::RopeAppend(a, b, c, n, e, f, g, h, i, j),
            4 => I::RopeAppend(a, b, c, d, n, f, g, h, i, j),
            5 => I::RopeAppend(a, b, c, d, e, n, g, h, i, j),
            6 => I::RopeAppend(a, b, c, d, e, f, n, h, i, j),
            _ => panic!("RopeAppend: bad idx {idx}"),
        },
        I::MlaSplit(a, b, c) => match idx {
            0 => I::MlaSplit(n, b, c),
            1 => I::MlaSplit(a, n, c),
            2 => I::MlaSplit(a, b, n),
            _ => panic!("MlaSplit: bad idx {idx}"),
        },
        I::MlaAttention(a, b, c, d, e) => match idx {
            0 => I::MlaAttention(n, b, c, d, e),
            1 => I::MlaAttention(a, n, c, d, e),
            2 => I::MlaAttention(a, b, n, d, e),
            3 => I::MlaAttention(a, b, c, n, e),
            4 => I::MlaAttention(a, b, c, d, n),
            _ => panic!("MlaAttention: bad idx {idx}"),
        },
        I::GatedDeltaNet(a, b, c, d, e, f) => match idx {
            0 => I::GatedDeltaNet(n, b, c, d, e, f),
            1 => I::GatedDeltaNet(a, n, c, d, e, f),
            2 => I::GatedDeltaNet(a, b, n, d, e, f),
            3 => I::GatedDeltaNet(a, b, c, n, e, f),
            4 => I::GatedDeltaNet(a, b, c, d, n, f),
            5 => I::GatedDeltaNet(a, b, c, d, e, n),
            _ => panic!("GatedDeltaNet: bad idx {idx}"),
        },
        I::GateSplit(a, b, c) => match idx {
            0 => I::GateSplit(n, b, c),
            1 => I::GateSplit(a, n, c),
            2 => I::GateSplit(a, b, n),
            _ => panic!("GateSplit: bad idx {idx}"),
        },
        I::GateApply(a, b, c) => match idx {
            0 => I::GateApply(n, b, c),
            1 => I::GateApply(a, n, c),
            2 => I::GateApply(a, b, n),
            _ => panic!("GateApply: bad idx {idx}"),
        },
        I::GateScale(a, b, c, d) => match idx {
            0 => I::GateScale(n, b, c, d),
            1 => I::GateScale(a, n, c, d),
            2 => I::GateScale(a, b, n, d),
            3 => I::GateScale(a, b, c, n),
            _ => panic!("GateScale: bad idx {idx}"),
        },
        I::DeepSeekMoe(a, b, c) => match idx {
            0 => I::DeepSeekMoe(n, b, c),
            1 => I::DeepSeekMoe(a, n, c),
            2 => I::DeepSeekMoe(a, b, n),
            _ => panic!("DeepSeekMoe: bad idx {idx}"),
        },
        I::DeepSeekMoeFp8Block(a, b, c) => match idx {
            0 => I::DeepSeekMoeFp8Block(n, b, c),
            1 => I::DeepSeekMoeFp8Block(a, n, c),
            2 => I::DeepSeekMoeFp8Block(a, b, n),
            _ => panic!("DeepSeekMoeFp8Block: bad idx {idx}"),
        },
        I::DeepSeekMoeGgml(a, b, c) => match idx {
            0 => I::DeepSeekMoeGgml(n, b, c),
            1 => I::DeepSeekMoeGgml(a, n, c),
            2 => I::DeepSeekMoeGgml(a, b, n),
            _ => panic!("DeepSeekMoeGgml: bad idx {idx}"),
        },
        I::FusedMoe(a, b, c) => match idx {
            0 => I::FusedMoe(n, b, c),
            1 => I::FusedMoe(a, n, c),
            2 => I::FusedMoe(a, b, n),
            _ => panic!("FusedMoe: bad idx {idx}"),
        },
        I::MetalFusedMoe(a, b, c, d, e, f, g, h, i) => match idx {
            0 => I::MetalFusedMoe(n, b, c, d, e, f, g, h, i),
            1 => I::MetalFusedMoe(a, n, c, d, e, f, g, h, i),
            2 => I::MetalFusedMoe(a, b, n, d, e, f, g, h, i),
            3 => I::MetalFusedMoe(a, b, c, n, e, f, g, h, i),
            4 => I::MetalFusedMoe(a, b, c, d, n, f, g, h, i),
            5 => I::MetalFusedMoe(a, b, c, d, e, n, g, h, i),
            6 => I::MetalFusedMoe(a, b, c, d, e, f, n, h, i),
            7 => I::MetalFusedMoe(a, b, c, d, e, f, g, n, i),
            8 => I::MetalFusedMoe(a, b, c, d, e, f, g, h, n),
            _ => panic!("MetalFusedMoe: bad idx {idx}"),
        },
        I::MetalSharedFusedMoe(a, b, c, d, e, f, g, h, i, j, k) => match idx {
            0 => I::MetalSharedFusedMoe(n, b, c, d, e, f, g, h, i, j, k),
            1 => I::MetalSharedFusedMoe(a, n, c, d, e, f, g, h, i, j, k),
            2 => I::MetalSharedFusedMoe(a, b, n, d, e, f, g, h, i, j, k),
            3 => I::MetalSharedFusedMoe(a, b, c, n, e, f, g, h, i, j, k),
            4 => I::MetalSharedFusedMoe(a, b, c, d, n, f, g, h, i, j, k),
            5 => I::MetalSharedFusedMoe(a, b, c, d, e, n, g, h, i, j, k),
            6 => I::MetalSharedFusedMoe(a, b, c, d, e, f, n, h, i, j, k),
            7 => I::MetalSharedFusedMoe(a, b, c, d, e, f, g, n, i, j, k),
            8 => I::MetalSharedFusedMoe(a, b, c, d, e, f, g, h, n, j, k),
            9 => I::MetalSharedFusedMoe(a, b, c, d, e, f, g, h, i, n, k),
            10 => I::MetalSharedFusedMoe(a, b, c, d, e, f, g, h, i, j, n != 0),
            _ => panic!("MetalSharedFusedMoe: bad idx {idx}"),
        },
        I::GemmaMoe(a, b, c, d, e, f, g, h, i, j) => match idx {
            0 => I::GemmaMoe(n, b, c, d, e, f, g, h, i, j),
            1 => I::GemmaMoe(a, n, c, d, e, f, g, h, i, j),
            2 => I::GemmaMoe(a, b, n, d, e, f, g, h, i, j),
            3 => I::GemmaMoe(a, b, c, n, e, f, g, h, i, j),
            4 => I::GemmaMoe(a, b, c, d, n, f, g, h, i, j),
            5 => I::GemmaMoe(a, b, c, d, e, n, g, h, i, j),
            6 => I::GemmaMoe(a, b, c, d, e, f, n, h, i, j),
            7 => I::GemmaMoe(a, b, c, d, e, f, g, n, i, j),
            8 => I::GemmaMoe(a, b, c, d, e, f, g, h, n, j),
            9 => I::GemmaMoe(a, b, c, d, e, f, g, h, i, n),
            _ => panic!("GemmaMoe: bad idx {idx}"),
        },
        I::SharedFusedMoe(a, b, c) => match idx {
            0 => I::SharedFusedMoe(n, b, c),
            1 => I::SharedFusedMoe(a, n, c),
            2 => I::SharedFusedMoe(a, b, n),
            _ => panic!("SharedFusedMoe: bad idx {idx}"),
        },
        I::CutlassGemm(a, b, c, d, e, f, g, h) => match idx {
            0 => I::CutlassGemm(n, b, c, d, e, f, g, h),
            1 => I::CutlassGemm(a, n, c, d, e, f, g, h),
            2 => I::CutlassGemm(a, b, n, d, e, f, g, h),
            3 => I::CutlassGemm(a, b, c, n, e, f, g, h),
            4 => I::CutlassGemm(a, b, c, d, n, f, g, h),
            5 => I::CutlassGemm(a, b, c, d, e, n, g, h),
            6 => I::CutlassGemm(a, b, c, d, e, f, n, h),
            7 => I::CutlassGemm(a, b, c, d, e, f, g, n),
            _ => panic!("CutlassGemm: bad idx {idx}"),
        },
        I::CutlassGemmSplitK(a, b, c, d, e, f, g, h, i) => match idx {
            0 => I::CutlassGemmSplitK(n, b, c, d, e, f, g, h, i),
            1 => I::CutlassGemmSplitK(a, n, c, d, e, f, g, h, i),
            2 => I::CutlassGemmSplitK(a, b, n, d, e, f, g, h, i),
            3 => I::CutlassGemmSplitK(a, b, c, n, e, f, g, h, i),
            4 => I::CutlassGemmSplitK(a, b, c, d, n, f, g, h, i),
            5 => I::CutlassGemmSplitK(a, b, c, d, e, n, g, h, i),
            6 => I::CutlassGemmSplitK(a, b, c, d, e, f, n, h, i),
            7 => I::CutlassGemmSplitK(a, b, c, d, e, f, g, n, i),
            8 => I::CutlassGemmSplitK(a, b, c, d, e, f, g, h, n),
            _ => panic!("CutlassGemmSplitK: bad idx {idx}"),
        },
        I::CutlassGemmAdd(a, b, c, d, e, f, g, h) => match idx {
            0 => I::CutlassGemmAdd(n, b, c, d, e, f, g, h),
            1 => I::CutlassGemmAdd(a, n, c, d, e, f, g, h),
            2 => I::CutlassGemmAdd(a, b, n, d, e, f, g, h),
            3 => I::CutlassGemmAdd(a, b, c, n, e, f, g, h),
            4 => I::CutlassGemmAdd(a, b, c, d, n, f, g, h),
            5 => I::CutlassGemmAdd(a, b, c, d, e, n, g, h),
            6 => I::CutlassGemmAdd(a, b, c, d, e, f, n, h),
            7 => I::CutlassGemmAdd(a, b, c, d, e, f, g, n),
            _ => panic!("CutlassGemmAdd: bad idx {idx}"),
        },
        I::CutlassGemv(a, b, c, d, e) => match idx {
            0 => I::CutlassGemv(n, b, c, d, e),
            1 => I::CutlassGemv(a, n, c, d, e),
            2 => I::CutlassGemv(a, b, n, d, e),
            3 => I::CutlassGemv(a, b, c, n, e),
            4 => I::CutlassGemv(a, b, c, d, n),
            _ => panic!("CutlassGemv: bad idx {idx}"),
        },
        I::CutlassFusedGemmBias(a, b, c, d, e, f, g, h) => match idx {
            0 => I::CutlassFusedGemmBias(n, b, c, d, e, f, g, h),
            1 => I::CutlassFusedGemmBias(a, n, c, d, e, f, g, h),
            2 => I::CutlassFusedGemmBias(a, b, n, d, e, f, g, h),
            3 => I::CutlassFusedGemmBias(a, b, c, n, e, f, g, h),
            4 => I::CutlassFusedGemmBias(a, b, c, d, n, f, g, h),
            5 => I::CutlassFusedGemmBias(a, b, c, d, e, n, g, h),
            6 => I::CutlassFusedGemmBias(a, b, c, d, e, f, n, h),
            7 => I::CutlassFusedGemmBias(a, b, c, d, e, f, g, n),
            _ => panic!("CutlassFusedGemmBias: bad idx {idx}"),
        },
        I::CutlassFusedGateUpSiluMul(a, b, c, d, e, f) => match idx {
            0 => I::CutlassFusedGateUpSiluMul(n, b, c, d, e, f),
            1 => I::CutlassFusedGateUpSiluMul(a, n, c, d, e, f),
            2 => I::CutlassFusedGateUpSiluMul(a, b, n, d, e, f),
            3 => I::CutlassFusedGateUpSiluMul(a, b, c, n, e, f),
            4 => I::CutlassFusedGateUpSiluMul(a, b, c, d, n, f),
            5 => I::CutlassFusedGateUpSiluMul(a, b, c, d, e, n),
            _ => panic!("CutlassFusedGateUpSiluMul: bad idx {idx}"),
        },
        I::CutlassFusedGateUpGeluMul(a, b, c, d, e, f, g, h) => match idx {
            0 => I::CutlassFusedGateUpGeluMul(n, b, c, d, e, f, g, h),
            1 => I::CutlassFusedGateUpGeluMul(a, n, c, d, e, f, g, h),
            2 => I::CutlassFusedGateUpGeluMul(a, b, n, d, e, f, g, h),
            3 => I::CutlassFusedGateUpGeluMul(a, b, c, n, e, f, g, h),
            4 => I::CutlassFusedGateUpGeluMul(a, b, c, d, n, f, g, h),
            5 => I::CutlassFusedGateUpGeluMul(a, b, c, d, e, n, g, h),
            6 => I::CutlassFusedGateUpGeluMul(a, b, c, d, e, f, n, h),
            7 => I::CutlassFusedGateUpGeluMul(a, b, c, d, e, f, g, n),
            _ => panic!("CutlassFusedGateUpGeluMul: bad idx {idx}"),
        },
        I::CutlassFusedQkvRopeCache(a, b, c, d, e, f, g, h, i) => match idx {
            0 => I::CutlassFusedQkvRopeCache(n, b, c, d, e, f, g, h, i),
            1 => I::CutlassFusedQkvRopeCache(a, n, c, d, e, f, g, h, i),
            2 => I::CutlassFusedQkvRopeCache(a, b, n, d, e, f, g, h, i),
            4 => I::CutlassFusedQkvRopeCache(a, b, c, d, n, f, g, h, i),
            5 => I::CutlassFusedQkvRopeCache(a, b, c, d, e, n, g, h, i),
            6 => I::CutlassFusedQkvRopeCache(a, b, c, d, e, f, n, h, i),
            7 => I::CutlassFusedQkvRopeCache(a, b, c, d, e, f, g, n, i),
            8 => I::CutlassFusedQkvRopeCache(a, b, c, d, e, f, g, h, n),
            _ => panic!("CutlassFusedQkvRopeCache: bad idx {idx}"),
        },
        I::CutlassFusedQkvRopePrefill(a, b, c, d, e, f, g, h, i, j, k) => match idx {
            0 => I::CutlassFusedQkvRopePrefill(n, b, c, d, e, f, g, h, i, j, k),
            1 => I::CutlassFusedQkvRopePrefill(a, n, c, d, e, f, g, h, i, j, k),
            2 => I::CutlassFusedQkvRopePrefill(a, b, n, d, e, f, g, h, i, j, k),
            3 => I::CutlassFusedQkvRopePrefill(a, b, c, n, e, f, g, h, i, j, k),
            4 => I::CutlassFusedQkvRopePrefill(a, b, c, d, n, f, g, h, i, j, k),
            6 => I::CutlassFusedQkvRopePrefill(a, b, c, d, e, f, n, h, i, j, k),
            7 => I::CutlassFusedQkvRopePrefill(a, b, c, d, e, f, g, n, i, j, k),
            8 => I::CutlassFusedQkvRopePrefill(a, b, c, d, e, f, g, h, n, j, k),
            9 => I::CutlassFusedQkvRopePrefill(a, b, c, d, e, f, g, h, i, n, k),
            10 => I::CutlassFusedQkvRopePrefill(a, b, c, d, e, f, g, h, i, j, n),
            _ => panic!("CutlassFusedQkvRopePrefill: bad idx {idx}"),
        },
        I::MarlinGemm(a, b, c) => match idx {
            0 => I::MarlinGemm(n, b, c),
            1 => I::MarlinGemm(a, n, c),
            2 => I::MarlinGemm(a, b, n),
            _ => panic!("MarlinGemm: bad idx {idx}"),
        },
        I::MarlinFusedGateUpSiluMul(a, b, c) => match idx {
            0 => I::MarlinFusedGateUpSiluMul(n, b, c),
            1 => I::MarlinFusedGateUpSiluMul(a, n, c),
            2 => I::MarlinFusedGateUpSiluMul(a, b, n),
            _ => panic!("MarlinFusedGateUpSiluMul: bad idx {idx}"),
        },
        I::MarlinFusedGateUpGeluMul(a, b, c) => match idx {
            0 => I::MarlinFusedGateUpGeluMul(n, b, c),
            1 => I::MarlinFusedGateUpGeluMul(a, n, c),
            2 => I::MarlinFusedGateUpGeluMul(a, b, n),
            _ => panic!("MarlinFusedGateUpGeluMul: bad idx {idx}"),
        },
        I::MarlinFusedQkvRopeCache(a, b, c) => match idx {
            0 => I::MarlinFusedQkvRopeCache(n, b, c),
            1 => I::MarlinFusedQkvRopeCache(a, n, c),
            2 => I::MarlinFusedQkvRopeCache(a, b, n),
            _ => panic!("MarlinFusedQkvRopeCache: bad idx {idx}"),
        },
        I::MarlinFusedQkvRopePrefill(a, b, c, d, e) => match idx {
            0 => I::MarlinFusedQkvRopePrefill(n, b, c, d, e),
            1 => I::MarlinFusedQkvRopePrefill(a, n, c, d, e),
            2 => I::MarlinFusedQkvRopePrefill(a, b, n, d, e),
            3 => I::MarlinFusedQkvRopePrefill(a, b, c, n, e),
            4 => I::MarlinFusedQkvRopePrefill(a, b, c, d, n),
            _ => panic!("MarlinFusedQkvRopePrefill: bad idx {idx}"),
        },
        I::Bnb4Gemm(a, b, c) => match idx {
            0 => I::Bnb4Gemm(n, b, c),
            1 => I::Bnb4Gemm(a, n, c),
            2 => I::Bnb4Gemm(a, b, n),
            _ => panic!("Bnb4Gemm: bad idx {idx}"),
        },
        I::Bnb4FusedGateUpSiluMul(a, b, c) => match idx {
            0 => I::Bnb4FusedGateUpSiluMul(n, b, c),
            1 => I::Bnb4FusedGateUpSiluMul(a, n, c),
            2 => I::Bnb4FusedGateUpSiluMul(a, b, n),
            _ => panic!("Bnb4FusedGateUpSiluMul: bad idx {idx}"),
        },
        I::Bnb4FusedGateUpGeluMul(a, b, c) => match idx {
            0 => I::Bnb4FusedGateUpGeluMul(n, b, c),
            1 => I::Bnb4FusedGateUpGeluMul(a, n, c),
            2 => I::Bnb4FusedGateUpGeluMul(a, b, n),
            _ => panic!("Bnb4FusedGateUpGeluMul: bad idx {idx}"),
        },
        I::Bnb4FusedQkvRopeCache(a, b, c) => match idx {
            0 => I::Bnb4FusedQkvRopeCache(n, b, c),
            1 => I::Bnb4FusedQkvRopeCache(a, n, c),
            2 => I::Bnb4FusedQkvRopeCache(a, b, n),
            _ => panic!("Bnb4FusedQkvRopeCache: bad idx {idx}"),
        },
        I::Bnb4FusedQkvRopePrefill(a, b, c, d, e) => match idx {
            0 => I::Bnb4FusedQkvRopePrefill(n, b, c, d, e),
            1 => I::Bnb4FusedQkvRopePrefill(a, n, c, d, e),
            2 => I::Bnb4FusedQkvRopePrefill(a, b, n, d, e),
            3 => I::Bnb4FusedQkvRopePrefill(a, b, c, n, e),
            4 => I::Bnb4FusedQkvRopePrefill(a, b, c, d, n),
            _ => panic!("Bnb4FusedQkvRopePrefill: bad idx {idx}"),
        },
        I::GgmlGemm(a, b, c) => match idx {
            0 => I::GgmlGemm(n, b, c),
            1 => I::GgmlGemm(a, n, c),
            2 => I::GgmlGemm(a, b, n),
            _ => panic!("GgmlGemm: bad idx {idx}"),
        },
        I::GgmlFusedGateUpSiluMul(a, b, c) => match idx {
            0 => I::GgmlFusedGateUpSiluMul(n, b, c),
            1 => I::GgmlFusedGateUpSiluMul(a, n, c),
            2 => I::GgmlFusedGateUpSiluMul(a, b, n),
            _ => panic!("GgmlFusedGateUpSiluMul: bad idx {idx}"),
        },
        I::GgmlFusedGateUpGeluMul(a, b, c) => match idx {
            0 => I::GgmlFusedGateUpGeluMul(n, b, c),
            1 => I::GgmlFusedGateUpGeluMul(a, n, c),
            2 => I::GgmlFusedGateUpGeluMul(a, b, n),
            _ => panic!("GgmlFusedGateUpGeluMul: bad idx {idx}"),
        },
        I::GgmlFusedQkvRopeCache(a, b, c, d) => match idx {
            0 => I::GgmlFusedQkvRopeCache(n, b, c, d),
            1 => I::GgmlFusedQkvRopeCache(a, n, c, d),
            2 => I::GgmlFusedQkvRopeCache(a, b, n, d),
            _ => panic!("GgmlFusedQkvRopeCache: bad idx {idx}"),
        },
        I::GgmlFusedQkvRopePrefill(a, b, c, d, e) => match idx {
            0 => I::GgmlFusedQkvRopePrefill(n, b, c, d, e),
            1 => I::GgmlFusedQkvRopePrefill(a, n, c, d, e),
            2 => I::GgmlFusedQkvRopePrefill(a, b, n, d, e),
            3 => I::GgmlFusedQkvRopePrefill(a, b, c, n, e),
            4 => I::GgmlFusedQkvRopePrefill(a, b, c, d, n),
            _ => panic!("GgmlFusedQkvRopePrefill: bad idx {idx}"),
        },
        I::Fp8Gemm(a, b, c) => match idx {
            0 => I::Fp8Gemm(n, b, c),
            1 => I::Fp8Gemm(a, n, c),
            2 => I::Fp8Gemm(a, b, n),
            _ => panic!("Fp8Gemm: bad idx {idx}"),
        },
        I::Fp8FusedGemmBias(a, b, c) => match idx {
            0 => I::Fp8FusedGemmBias(n, b, c),
            1 => I::Fp8FusedGemmBias(a, n, c),
            2 => I::Fp8FusedGemmBias(a, b, n),
            _ => panic!("Fp8FusedGemmBias: bad idx {idx}"),
        },
        I::Fp8FusedGateUpSiluMul(a, b, c) => match idx {
            0 => I::Fp8FusedGateUpSiluMul(n, b, c),
            1 => I::Fp8FusedGateUpSiluMul(a, n, c),
            2 => I::Fp8FusedGateUpSiluMul(a, b, n),
            _ => panic!("Fp8FusedGateUpSiluMul: bad idx {idx}"),
        },
        I::Fp8FusedGateUpGeluMul(a, b, c) => match idx {
            0 => I::Fp8FusedGateUpGeluMul(n, b, c),
            1 => I::Fp8FusedGateUpGeluMul(a, n, c),
            2 => I::Fp8FusedGateUpGeluMul(a, b, n),
            _ => panic!("Fp8FusedGateUpGeluMul: bad idx {idx}"),
        },
        I::Fp8FusedQkvRopeCache(a, b, c) => match idx {
            0 => I::Fp8FusedQkvRopeCache(n, b, c),
            1 => I::Fp8FusedQkvRopeCache(a, n, c),
            2 => I::Fp8FusedQkvRopeCache(a, b, n),
            _ => panic!("Fp8FusedQkvRopeCache: bad idx {idx}"),
        },
        I::Fp8FusedQkvRopePrefill(a, b, c, d, e) => match idx {
            0 => I::Fp8FusedQkvRopePrefill(n, b, c, d, e),
            1 => I::Fp8FusedQkvRopePrefill(a, n, c, d, e),
            2 => I::Fp8FusedQkvRopePrefill(a, b, n, d, e),
            3 => I::Fp8FusedQkvRopePrefill(a, b, c, n, e),
            4 => I::Fp8FusedQkvRopePrefill(a, b, c, d, n),
            _ => panic!("Fp8FusedQkvRopePrefill: bad idx {idx}"),
        },
        I::Loop(a, b, c) => match idx {
            0 => I::Loop(n, b, c),
            1 => I::Loop(a, n, c),
            2 => I::Loop(a, b, n),
            _ => panic!("Loop: bad idx {idx}"),
        },
        I::Alias(a, b) => match idx {
            0 => I::Alias(n, b),
            1 => I::Alias(a, n),
            _ => panic!("Alias: bad idx {idx}"),
        },
        I::Free(_a) => match idx {
            0 => I::Free(n),
            _ => panic!("Free: bad idx {idx}"),
        },
        I::MetalBiasAdd(a, b, c, d, e) => match idx {
            0 => I::MetalBiasAdd(n, b, c, d, e),
            1 => I::MetalBiasAdd(a, n, c, d, e),
            2 => I::MetalBiasAdd(a, b, n, d, e),
            3 => I::MetalBiasAdd(a, b, c, n, e),
            _ => panic!("MetalBiasAdd: bad idx {idx}"),
        },
        I::AttentionPrefillPaged(a, b, c, d) => match idx {
            0 => I::AttentionPrefillPaged(n, b, c, d),
            1 => I::AttentionPrefillPaged(a, n, c, d),
            2 => I::AttentionPrefillPaged(a, b, n, d),
            _ => panic!("AttentionPrefillPaged: bad idx {idx}"),
        },
        I::AffineQmm(a, b, c, d, e, f, g, h) => match idx {
            0 => I::AffineQmm(n, b, c, d, e, f, g, h),
            1 => I::AffineQmm(a, n, c, d, e, f, g, h),
            2 => I::AffineQmm(a, b, n, d, e, f, g, h),
            3 => I::AffineQmm(a, b, c, n, e, f, g, h),
            4 => I::AffineQmm(a, b, c, d, n, f, g, h),
            5 => I::AffineQmm(a, b, c, d, e, n, g, h),
            6 => I::AffineQmm(a, b, c, d, e, f, n, h),
            7 => I::AffineQmm(a, b, c, d, e, f, g, n),
            _ => panic!("AffineQmm: bad idx {idx}"),
        },
        I::Nvfp4Qmm(a, b, c, d, e, f, g, h) => match idx {
            0 => I::Nvfp4Qmm(n, b, c, d, e, f, g, h),
            1 => I::Nvfp4Qmm(a, n, c, d, e, f, g, h),
            2 => I::Nvfp4Qmm(a, b, n, d, e, f, g, h),
            3 => I::Nvfp4Qmm(a, b, c, n, e, f, g, h),
            4 => I::Nvfp4Qmm(a, b, c, d, n, f, g, h),
            5 => I::Nvfp4Qmm(a, b, c, d, e, n, g, h),
            6 => I::Nvfp4Qmm(a, b, c, d, e, f, n, h),
            7 => I::Nvfp4Qmm(a, b, c, d, e, f, g, n),
            _ => panic!("Nvfp4Qmm: bad idx {idx}"),
        },
        I::SynthPreAttn(a, b, c, d, e, f, g, h, i) => match idx {
            0 => I::SynthPreAttn(n, b, c, d, e, f, g, h, i),
            1 => I::SynthPreAttn(a, n, c, d, e, f, g, h, i),
            2 => I::SynthPreAttn(a, b, n, d, e, f, g, h, i),
            3 => I::SynthPreAttn(a, b, c, n, e, f, g, h, i),
            4 => I::SynthPreAttn(a, b, c, d, n, f, g, h, i),
            5 => I::SynthPreAttn(a, b, c, d, e, n, g, h, i),
            6 => I::SynthPreAttn(a, b, c, d, e, f, n, h, i),
            _ => panic!("SynthPreAttn: bad idx {idx}"),
        },
        I::SynthMlpPreDown(a, b, c, d, e, f, g, h) => match idx {
            0 => I::SynthMlpPreDown(n, b, c, d, e, f, g, h),
            1 => I::SynthMlpPreDown(a, n, c, d, e, f, g, h),
            2 => I::SynthMlpPreDown(a, b, n, d, e, f, g, h),
            3 => I::SynthMlpPreDown(a, b, c, n, e, f, g, h),
            4 => I::SynthMlpPreDown(a, b, c, d, n, f, g, h),
            5 => I::SynthMlpPreDown(a, b, c, d, e, n, g, h),
            6 => I::SynthMlpPreDown(a, b, c, d, e, f, n, h),
            _ => panic!("SynthMlpPreDown: bad idx {idx}"),
        },
        I::SiluMul(a, b, c, w) => match idx {
            0 => I::SiluMul(n, b, c, w),
            1 => I::SiluMul(a, n, c, w),
            2 => I::SiluMul(a, b, n, w),
            _ => panic!("SiluMul: bad idx {idx}"),
        },
        I::GeluMul(a, b, c) => match idx {
            0 => I::GeluMul(n, b, c),
            1 => I::GeluMul(a, n, c),
            2 => I::GeluMul(a, b, n),
            _ => panic!("GeluMul: bad idx {idx}"),
        },
        I::SynthGateUpSiluMul(a, b, c, d, e, f) => match idx {
            0 => I::SynthGateUpSiluMul(n, b, c, d, e, f),
            1 => I::SynthGateUpSiluMul(a, n, c, d, e, f),
            2 => I::SynthGateUpSiluMul(a, b, n, d, e, f),
            3 => I::SynthGateUpSiluMul(a, b, c, n, e, f),
            4 => I::SynthGateUpSiluMul(a, b, c, d, n, f),
            _ => panic!("SynthGateUpSiluMul: bad idx {idx}"),
        },
        I::AffineEmbed(a, b, c) => match idx {
            0 => I::AffineEmbed(n, b, c),
            1 => I::AffineEmbed(a, n, c),
            2 => I::AffineEmbed(a, b, n),
            _ => panic!("AffineEmbed: bad idx {idx}"),
        },
    }
}

// ── Slot allocation ──────────────────────────────────────────────

// ⛔ INSTRUCTION-SELECTION ONLY. A tape-scheduled target reads its loop off the shared
// tape; this searches the ISel stream for one because that stream has no tape. Gated so
// metal/spyre do not COMPILE it, rather than compiling it and allowing it to be dead.
#[cfg(not(any(feature = "metal", feature = "spyre")))]
/// One slot per (tile, output_slot). Used when no liveness
/// information is available; the colored variant
/// [`colored_slot_map`] is what `lower_bucket` actually picks for
/// emission.
///
/// Its only caller is this module's own tests — `lower_bucket` takes the coloured map.
#[cfg(test)]
pub fn build_slot_map(fuf: &Fuf) -> SlotMap {
    let mut sm = SlotMap::new();
    for node in &fuf.nodes {
        let n_outputs = node.outputs.len().max(1) as u8;
        for slot in 0..n_outputs {
            sm.insert(node.id, slot);
        }
    }
    sm
}

/// Build a colored slot allocation via linear-scan register
/// allocation on the solved FUF. Two non-overlapping live ranges
/// share the same slot index — so layer L's `q` register and layer
/// L+1's `q` register collapse to one slot, and the per-layer body
/// emitted by [`lower_bucket`] becomes byte-identical across every
/// layer iteration. That's the precondition the layer-loop
/// detection relies on.
///
/// Constraints honored:
/// 1. **Shape partitioning.** Each color carries the shape of the
///    tiles it holds. A freed color goes back to *its shape's* free
///    pool; a tile of a different shape can never reuse it. This is
///    load-bearing for in-place mutation: `cutlass_gemm_add` writes
///    `[M, N]` into the residual buffer, so the buffer must have
///    been allocated for `[M, N]`. If a `[M, num_kv_heads, head_dim]`
///    K-tile and a `[M, hidden]` residual tile share a slot because
///    their lifetimes don't overlap, the K-tile's smaller buffer
///    survives into the residual op and the in-place write goes OOB.
/// 2. **Live-range overlap (within a shape).** Two same-shape slots
///    co-live iff one's def position ≤ the other's last-use position
///    and vice versa. Co-live slots within a shape get distinct
///    colors.
/// 3. **Same-shape aliasing collapses.** When `output_alias`
///    declares dst aliases owner AND `dst.shape == owner.shape`,
///    the dst is the same physical buffer (in-place mutation
///    semantics: the kernel mutates owner's buffer and downstream
///    consumers read it). The allocator pins dst to owner's color
///    — same `__tiles` slot, no `View` entry, no `Op::Alias` row.
/// 4. **Different-shape aliasing keeps a View.** `Reshape`-style
///    aliases (dst shape ≠ owner shape) point at the owner's
///    storage with new metadata; they need their own slot to hold
///    a `View`/`Reshaped` entry. Shape partitioning already places
///    them in a different free pool from the owner, so the runtime
///    `View(ref_slot=owner_slot)` indirection is non-trivial.
/// 5. **Consume.** `consumes_input_tiles` declares a slot whose
///    `OwnedTensor` migrates into the consumer's output. The
///    consumed slot's last-use is the consume site — past that it's
///    dead and its color is freed.
/// 6. **Protected slots** (the per-bucket fn's return tile, and the
///    backbone-output slot for `forward_backbone`) never have their
///    color reused — they must stay alive past the slice's end so
///    the per-bucket fn can `take_owned` them.
#[allow(clippy::too_many_arguments)]
pub fn colored_slot_map(
    fuf: &Fuf,
    sfuf: &Assignment,
    loop_ir: &Loop,
    lib: &ImplementationLibrary,
    skip_subgraph: Option<SubgraphId>,
    protected: &HashSet<(TileId, u8)>,
) -> SlotMap {
    use std::collections::BTreeSet;

    // Walk subgraphs in execution order; assign each one a position.
    let mut order: HashMap<SubgraphId, usize> = HashMap::new();
    let mut order_arr: Vec<SubgraphId> = Vec::new();
    for wave in &loop_ir.waves {
        for (sg, _) in &wave.subgraphs {
            if Some(*sg) == skip_subgraph {
                continue;
            }
            order.insert(*sg, order_arr.len());
            order_arr.push(*sg);
        }
    }

    // Alias: dst -> src (immediate). The slice's runtime View::ref_slot
    // points at the *flattened* owner, so we resolve chains for the
    // last-use extension below; for the color-distinction constraint
    // we use the resolved owner since that's what the View holds.
    let mut alias_to_owner: HashMap<(TileId, u8), (TileId, u8)> = HashMap::new();
    for &sg in &order_arr {
        let imp_id = sfuf
            .impl_of(sg)
            .expect("every scheduled subgraph has an Impl");
        let imp = lib.get(imp_id);
        let claimed = sfuf.tiles_in_subgraph(sg);
        let aliases = crate::alias_rules::output_alias(imp.alias_rule(), &claimed, fuf);
        for (dst, src_opt) in aliases {
            if let Some(src) = src_opt {
                alias_to_owner.insert(dst, src);
            }
        }
    }
    let resolve = |start: (TileId, u8)| -> (TileId, u8) {
        let mut cur = start;
        let mut seen = HashSet::new();
        while seen.insert(cur) {
            match alias_to_owner.get(&cur) {
                Some(up) => cur = *up,
                None => break,
            }
        }
        cur
    };

    // Consume: a slot whose `OwnedTensor` migrates into the consumer.
    // After the consume site the source slot is empty — its color is
    // freeable past consumer's position, regardless of any cross-
    // subgraph reads (the drop pass already excludes consumed slots).
    let mut consumed: HashSet<(TileId, u8)> = HashSet::new();
    for &sg in &order_arr {
        let imp_id = sfuf.impl_of(sg).expect("every subgraph has an Impl");
        let imp = lib.get(imp_id);
        let claimed = sfuf.tiles_in_subgraph(sg);
        for upstream in imp.consumes_input_tiles(&claimed, fuf) {
            consumed.insert(upstream);
        }
    }

    // Per-tile sub-positions. Each tile gets a unique flat index by
    // walking subgraphs in execution order, then tiles within each
    // subgraph in claim order. Coarser per-subgraph positions would
    // collapse every tile in a subgraph to the same `dp`, so a
    // producer tile whose only reader sits in the SAME subgraph's
    // claim ends up freed at its own def — and the next tile in the
    // subgraph reuses its color. Fine for single-kernel claims (the
    // producer's output is never materialized in the arena), but
    // wrong for storage-polymorphic impls whose `fan_out` emits
    // multiple kernels per subgraph (MetalFusedGateUpSiluMulImpl's
    // affine path: AffineQmm gate, AffineQmm up, SiluMul). Per-tile
    // sub-positions let the within-subgraph read walk below record
    // the consumer's sub-position as the producer's `last_use`, so
    // gate's slot stays distinct from up's slot across the SiluMul
    // read.
    //
    // Layer-template byte-equivalence: per-tile positions increment
    // monotonically. Each layer body's tiles occupy the same
    // relative offset range, so per-layer color assignments stay
    // byte-equivalent (the linear-scan reg allocation runs against
    // the same free-pool state at the same relative positions).
    let mut tile_position: HashMap<TileId, usize> = HashMap::new();
    let mut next_pos: usize = 0;
    for &sg in &order_arr {
        for tile in sfuf.tiles_in_subgraph(sg) {
            tile_position.insert(tile, next_pos);
            next_pos += 1;
        }
    }

    // Last use per OWNER (resolving alias chains): the latest tile-
    // position that reads this owner's storage, directly or via a
    // View. Within-subgraph reads count too (the impl's fan_out may
    // emit a kernel chain whose intermediate outputs hit the arena).
    // Last use of a non-owner (a View slot itself) is computed
    // separately below.
    // A fused subgraph is ONE kernel (a fusion) whose claimed tiles
    // execute CONCURRENTLY across threadgroups — there is no intra-subgraph
    // ordering between a tile that READS a boundary input and a tile
    // that WRITES (in-place) that same storage. So a boundary input
    // read by ANY tile of the subgraph stays live until the subgraph
    // COMPLETES (its last per-tile position), not just until the
    // reading tile's position. Charging the reading tile's position
    // lets the linear scan free the input early and an in-place output
    // of the SAME subgraph (e.g. SynthMlpPreDown / SynthPreAttn residual
    // `Add`) reuse its slot — a cross-threadgroup RAW/WAR race that only
    // manifests on hardware that doesn't run the tiles in position order
    // (M5 / gen-17). Single-tile subgraphs are unaffected (subgraph-last
    // == the tile's own position). Genuinely-safe element-wise in-place
    // (the standalone `Add`) is untouched: it's pinned via an explicit
    // `output_alias`, a separate mechanism from this implicit reuse.
    let mut owner_last_use: HashMap<(TileId, u8), usize> = HashMap::new();
    for &sg in &order_arr {
        let tiles = sfuf.tiles_in_subgraph(sg);
        let sg_last = tiles.iter().map(|t| tile_position[t]).max().unwrap_or(0);
        for tile in &tiles {
            for input in &fuf.get(*tile).inputs {
                if let FufInput::Tile { id, slot } = input {
                    let owner = resolve((*id, *slot));
                    owner_last_use
                        .entry(owner)
                        .and_modify(|p| *p = (*p).max(sg_last))
                        .or_insert(sg_last);
                }
            }
        }
    }
    // View slots' last_use: when is the View itself read? A View is
    // read whenever its dst slot appears as a Tile input to some
    // downstream tile. Same walk but without alias resolution.
    let mut view_last_use: HashMap<(TileId, u8), usize> = HashMap::new();
    for &sg in &order_arr {
        for tile in sfuf.tiles_in_subgraph(sg) {
            let consumer_pos = tile_position[&tile];
            for input in &fuf.get(tile).inputs {
                if let FufInput::Tile { id, slot } = input
                    && alias_to_owner.contains_key(&(*id, *slot))
                {
                    view_last_use
                        .entry((*id, *slot))
                        .and_modify(|p| *p = (*p).max(consumer_pos))
                        .or_insert(consumer_pos);
                }
            }
        }
    }

    // Collect every (tile, output_slot) pair, sorted by per-tile
    // sub-position.
    let mut def_pos: HashMap<TileId, usize> = HashMap::new();
    for (&tile, &pos) in &tile_position {
        def_pos.insert(tile, pos);
    }
    let mut pairs: Vec<(usize, TileId, u8)> = Vec::new();
    for &sg in &order_arr {
        for tile in sfuf.tiles_in_subgraph(sg) {
            let n_out = fuf.get(tile).outputs.len().max(1) as u8;
            for slot in 0..n_out {
                pairs.push((tile_position[&tile], tile, slot));
            }
        }
    }
    pairs.sort_by_key(|&(p, t, s)| (p, t, s));

    // Linear-scan, partitioned by shape. Each color is born tagged
    // with the shape of the tile that minted it; a freed color
    // returns to *that* shape's pool. A tile of a different shape
    // never reuses it.
    //
    // Aliases: an `output_alias` declaration whose dst and resolved
    // owner share a shape pins the dst to the owner's color. These
    // are in-place mutation aliases (CutlassGemmAdd's add output is
    // the residual buffer; FusedAddRmsNorm's outputs are the
    // mutated upstream buffers). They literally share storage, so
    // they're the same `__tiles` slot — no `View` entry, no
    // `Op::Alias` row, no separate active entry.
    //
    // Aliases whose dst and owner have *different* shapes are
    // metadata-only (Reshape). They get their own slot in their
    // own shape pool; a `View { ref_slot: owner_slot }` entry
    // populated by `Op::Alias(dst_slot, owner_slot)` indirects to
    // the owner's storage.
    let mut active: Vec<(usize, u32, (TileId, u8))> = Vec::new();
    let mut free_colors_by_shape: HashMap<Shape, BTreeSet<u32>> = HashMap::new();
    let mut color_shape: HashMap<u32, Shape> = HashMap::new();
    let mut next_color: u32 = 0;
    let mut sm = SlotMap::new();

    for (dp, tile, slot) in pairs {
        // Owners that are DIRECT inputs to the tile being defined at this
        // position, WHEN that tile is a non-pointwise op (output element
        // depends on input elements other than its own index). Such an op
        // reads across its input while writing its output — a tiled gemm
        // streams `input` from device across threadgroups (see
        // `gemm_bf16_specialized`), vision RoPE reads the rotate-half
        // partner, varlen attention reads every key/value in the segment.
        // If the output reused an input's color (implicit in-place) one
        // threadgroup's write clobbers a region another threadgroup still
        // has to read → cross-threadgroup RAW/WAR corruption (gemm
        // overflows to NaN; rope/attn silently wrong). So an input stays
        // live THROUGH such a tile's def and must not be freed for that
        // tile's own output.
        //
        // POINTWISE ops (Add/Mul/Gelu/…, and per-row RmsNorm with one
        // threadgroup per row) are left untouched: `out[i]` reads only
        // `in[i]`, so implicit in-place chain-collapse is safe and stays
        // (preserves the elementwise color-chain coalescing + byte-
        // equivalence across layers). Genuinely-safe element-wise in-place
        // is also pinned explicitly via `output_alias` (collapses below).
        //
        // Op set: `Gemm` is shared with text, but text never coalesces it
        // in-place (fused QKV / down-proj outputs are a DIFFERENT shape
        // than their input, so the shape-partitioned pool can't reuse it);
        // the exclusion is a no-op there and only bites Qwen3.5-VL's
        // separate same-shape q/k/v gemms. `VisionRope` / `VarlenAttention`
        // are vision-only opcodes (text uses `RopeAppend` / `Attention`).
        let reads_across_input = matches!(
            fuf.get(tile).op,
            crate::classified::OpKind::Gemm
                | crate::classified::OpKind::VisionRope
                | crate::classified::OpKind::VarlenAttention
        );
        let cur_inputs: std::collections::HashSet<(TileId, u8)> = if reads_across_input {
            fuf.get(tile)
                .inputs
                .iter()
                .filter_map(|input| match input {
                    FufInput::Tile { id, slot } => Some(resolve((*id, *slot))),
                    _ => None,
                })
                .collect()
        } else {
            std::collections::HashSet::new()
        };

        // Free expired colors before allocating. `lu <= dp` is the
        // standard "use kills before def" semantics: a slot whose
        // last reader is the subgraph at `dp` dies after that
        // read — so when we're allocating outputs at `dp`, its
        // color is reusable. Without `<=`, layer 0's body would
        // differ from layer 1's because the embed slot wouldn't
        // be freed in time for layer 0's add output to take its
        // color, breaking byte-equivalence across layers. Exception:
        // a color last-read by THIS tile (its own input) is held one
        // position longer (see `cur_inputs` above).
        active.retain(|&(lu, color, ts)| {
            if lu <= dp && !cur_inputs.contains(&ts) {
                let s = color_shape[&color].clone();
                free_colors_by_shape.entry(s).or_default().insert(color);
                false
            } else {
                true
            }
        });

        let tile_shape = fuf.get(tile).outputs[slot as usize].clone();
        let is_alias = alias_to_owner.contains_key(&(tile, slot));

        // Alias collapse: dst pins to owner's color. By construction
        // an alias preserves bytes (Reshape is metadata-only on the
        // arena, just changes the logical shape interpretation), so
        // sharing the owner's color is safe even when shapes differ
        // (`[T, 4096]` vs `[T, num_q, head_dim]` for Qwen3's per-head
        // q_norm path).
        //
        // Pre-2026-05-17 this branch had a `owner_shape == tile_shape`
        // gate. That worked for Llama (no reshape between gemm and
        // rmsnorm) but broke Qwen3-family per-head q_norm/k_norm: the
        // reshaped view got its own arena slot, never written by the
        // upstream gemm, so RmsNorm read zeros → cascading garbage
        // through attention/MoE. The MLX-vs-metal parity bisect at
        // dispatch 8 (q_norm) showed input ptr != q_proj output ptr;
        // see `project-metal-bf16-scales` memory.
        if is_alias {
            let owner = resolve((tile, slot));
            let owner_color = sm.of(owner.0, owner.1);
            sm.insert_at(tile, slot, owner_color);
            continue;
        }

        // Compute lu_self for the active entry.
        let lu_self = if is_alias {
            // Different-shape alias (Reshape view): dies after its
            // last reader.
            *view_last_use.get(&(tile, slot)).unwrap_or(&dp)
        } else {
            // Owner: dies at its own last_use, OR at consume site
            // (whichever is later — consume IS a use).
            let from_reads = *owner_last_use.get(&(tile, slot)).unwrap_or(&dp);
            // Protected slots stay alive forever. Consumed slots
            // (in-place consume pattern) end at the consume site,
            // which `owner_last_use` already records as a read —
            // so `from_reads` is correct for both consumed and
            // non-consumed cases. Only protection is special.
            if protected.contains(&(tile, slot)) {
                usize::MAX
            } else {
                let _ = &consumed; // documenting reliance on the walk above
                from_reads
            }
        };

        // Pick from this shape's free pool; mint a new color if empty.
        let pool = free_colors_by_shape.entry(tile_shape.clone()).or_default();
        let color = if let Some(&c) = pool.iter().next() {
            pool.remove(&c);
            c
        } else {
            let c = next_color;
            next_color += 1;
            color_shape.insert(c, tile_shape.clone());
            c
        };

        sm.insert_at(tile, slot, color);
        active.push((lu_self, color, (tile, slot)));
    }

    sm
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
    /// position. Same length as `instances`. Each entry is the
    /// converted `Implementation::required_weights()` output for the
    /// fan_out call that produced the corresponding instruction (the
    /// vector is broadcast across all instructions a single fan_out
    /// call returned). The per-arch `WeightAccessors` impl walks
    /// `(bucket, op_idx, slot)` against this parallel array to emit
    /// match arms — `op_idx` is the index into `instances`.
    pub weight_slots: Vec<Vec<WeightSlot>>,
    /// Total size of the runtime tile table for this bucket.
    /// Tape-path only: the ISel stream carries its slot table elsewhere.
    #[cfg(any(feature = "metal", feature = "spyre"))]
    pub num_slots: u32,
    /// Slot index whose `Owned` entry is the bucket fn's return
    /// value. Tape-path only, same reason as `num_slots`.
    #[cfg(any(feature = "metal", feature = "spyre"))]
    pub final_slot: u32,
    /// One `barrier_before` flag per entry in `instances`. `true`
    /// means a concurrency-aware backend (today: metal MTL4
    /// encoder) must serialize the dispatched instance against
    /// every prior instance in the bucket. Computed at macro time
    /// from the FUF dependency graph + per-`Implementation` KV-
    /// layer-IO declarations — runtime never re-derives.
    pub barriers: Vec<bool>,
}

/// Variant shapes the macro accumulates across every bucket of one
/// arch. Used to drive shape-agreement checking + per-variant
/// iter-index field discovery for `apply_loop_compression`.
///
/// Bodies for each variant live in `scratchy_forward_compiler::Instruction::eval`;
/// nothing per-arch needs the body here.
#[derive(Default)]
pub struct ArchOpcodes {
    /// Variant ident → shape. First insert wins; later inserts of
    /// the same variant ident must agree on shape (codegen panics
    /// on mismatch).
    by_name: BTreeMap<String, OpcodeShape>,
}

impl ArchOpcodes {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a variant shape. Panics if the same variant ident is
    /// registered with a structurally different shape.
    pub fn register(&mut self, shape: OpcodeShape) {
        let key = shape.name.to_string();
        if let Some(existing_shape) = self.by_name.get(&key) {
            assert_shapes_agree(existing_shape, &shape);
            return;
        }
        self.by_name.insert(key, shape);
    }

    /// Snapshot of registered variant shapes keyed by variant ident
    /// string. Includes the universal `Alias` / `Free` / `Loop`
    /// variants. Consumed by [`emit_bucket_static_slice`] to
    /// type-check positional field values against the registered
    /// shape per bucket.
    pub fn shapes_by_name(&self) -> BTreeMap<String, OpcodeShape> {
        let mut out: BTreeMap<String, OpcodeShape> = self.by_name.clone();
        out.insert("Alias".to_string(), alias_variant_shape());
        out.insert("Free".to_string(), free_variant_shape());
        out.insert("Loop".to_string(), loop_variant_shape());
        out
    }

    /// Iterate (variant_name, shape). Used by
    /// `apply_loop_compression` to build the per-variant layer-field
    /// position map.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &OpcodeShape)> {
        self.by_name.iter()
    }
}

// ⛔ INSTRUCTION-SELECTION ONLY. A tape-scheduled target reads its loop off the shared
// tape; this searches the ISel stream for one because that stream has no tape. Gated so
// metal/spyre do not COMPILE it, rather than compiling it and allowing it to be dead.
#[cfg(not(any(feature = "metal", feature = "spyre")))]
/// Find the largest contiguous run of instances that can be
/// described as N copies of a P-instruction body, optionally
/// allowing a per-variant ITERATION-INDEX field to step linearly
/// (by exactly 1) between copies. Returns `Some((start, period,
/// num_iters))` on success, `None` when no such run exists.
///
/// This is generic loop detection — there's no semantic notion of
/// "layer" in here. The caller passes
/// `iter_index_field_per_variant`: for any variant where one of
/// its fields' value forms `c, c+1, c+2, …` across consecutive
/// candidate iterations, that field's index is in the map and
/// gets compared modulo iter offset; every other field is
/// compared byte-exactly. Variants not in the map have all fields
/// compared byte-exactly.
///
/// Algorithm: O((n × max_period) × (avg_iters × period_check_cost)).
/// `period_check_cost` is `O(P)` of pre-hashed fingerprint
/// equality + integer-add equality for iter-index fields.
fn detect_repeating_run(
    instances: &[Instruction],
    iter_index_field_per_variant: &std::collections::HashMap<String, usize>,
) -> Option<(usize, usize, u32)> {
    let n = instances.len();
    if n < 2 {
        return None;
    }

    // Precompute per-instance: a fingerprint covering only the
    // byte-exact-compared parts (variant ident + every field that
    // ISN'T the iter-index). Pre-hashed once so the inner pattern-
    // match loop is integer compare instead of repeated rendering
    // (the n³ blow-up). Fields are read out positionally via the
    // variant_name + field_at helpers; for the iter-index field
    // position we record the value separately for the +1 step check.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let fp_and_iter: Vec<(u64, Option<u32>)> = instances
        .iter()
        .map(|inst| {
            let var_name = instruction_variant_name(inst);
            let iter_field = iter_index_field_per_variant.get(var_name).copied();
            let mut hasher = DefaultHasher::new();
            var_name.hash(&mut hasher);
            // Iterate over all fields up to a generous bound; field_at
            // returns None past the variant's arity, terminating the
            // walk for that variant. Fields the helper can't represent
            // as u64 (arrays, floats, bools) hash as a marker sentinel
            // so two instances with structurally-different non-scalar
            // fields can still collide on hash but that's fine — a
            // hash-only filter is followed by no further check today
            // since the only known non-scalar fields (Reshape arrays,
            // f32 epsilons, bool causal flags) are stable per impl
            // and don't iterate.
            for i in 0..16 {
                if Some(i) == iter_field {
                    continue;
                }
                match instruction_field_at(inst, i) {
                    Some(v) => v.hash(&mut hasher),
                    None => break,
                }
            }
            let iter_val = iter_field
                .and_then(|i| instruction_field_at(inst, i))
                .map(|v| v as u32);
            (hasher.finish(), iter_val)
        })
        .collect();

    let mut best: Option<(usize, usize, u32, usize)> = None; // (start, period, iters, span)

    for start in 0..n {
        let max_period = (n - start) / 2;
        for period in 1..=max_period {
            // Quick reject: fingerprint of [start..start+P] must
            // equal fingerprint of [start+P..start+2P].
            let block_a = &fp_and_iter[start..start + period];
            let block_b = &fp_and_iter[start + period..start + 2 * period];
            if !blocks_match(block_a, block_b, 1) {
                continue;
            }
            // Confirmed at least 2 iterations. Try extending.
            let mut iters = 2u32;
            loop {
                let next_start = start + iters as usize * period;
                if next_start + period > n {
                    break;
                }
                let block_n = &fp_and_iter[next_start..next_start + period];
                if !blocks_match(block_a, block_n, iters) {
                    break;
                }
                iters += 1;
            }
            let span = period * iters as usize;
            let cand = (start, period, iters, span);
            if best.is_none_or(|b| cand.3 > b.3) {
                best = Some(cand);
            }
        }
    }

    best.map(|(s, p, n, _)| (s, p, n))
}

// ⛔ INSTRUCTION-SELECTION ONLY. A tape-scheduled target reads its loop off the shared
// tape; this searches the ISel stream for one because that stream has no tape. Gated so
// metal/spyre do not COMPILE it, rather than compiling it and allowing it to be dead.
#[cfg(not(any(feature = "metal", feature = "spyre")))]
/// Two blocks of pre-hashed (fingerprint, iter_index_value) pairs
/// match iff the fingerprints are equal pairwise AND the
/// iter-index values, when present, satisfy `cand = base +
/// iter_offset`.
fn blocks_match(
    base: &[(u64, Option<u32>)],
    cand: &[(u64, Option<u32>)],
    iter_offset: u32,
) -> bool {
    if base.len() != cand.len() {
        return false;
    }
    for (b, c) in base.iter().zip(cand.iter()) {
        if b.0 != c.0 {
            return false;
        }
        match (b.1, c.1) {
            (Some(bv), Some(cv)) if cv == bv + iter_offset => {}
            (None, None) => {}
            _ => return false,
        }
    }
    true
}

// ⛔ INSTRUCTION-SELECTION ONLY. A tape-scheduled target reads its loop off the shared
// tape; this searches the ISel stream for one because that stream has no tape. Gated so
// metal/spyre do not COMPILE it, rather than compiling it and allowing it to be dead.
#[cfg(not(any(feature = "metal", feature = "spyre")))]
/// Apply loop compression to `lowered.instances` in place. When
/// [`detect_repeating_run`] finds a contiguous run, replace it
/// with one `Op::Loop` row plus a single iteration's body. The
/// iteration-index field on each body row is set to that row's
/// **per-row baseline** — the value the field had in iter 0 at
/// that row's position — instead of zeroed. The interpreter passes
/// the runtime iteration counter as `__layer`, and arm bodies
/// compute `let layer: u32 = __layer + layer;` so each row's
/// effective index is `__l + baseline`.
///
/// Per-row baselines are load-bearing because the body period can
/// span a layer boundary: in Llama, the body's trailing
/// `FusedAddRmsNorm(input_layernorm)` IS the *next* layer's
/// input_ln (fused with the residual add); its iter-0 layer is 1,
/// not 0, while the body's other rows have iter-0 layer 0.
/// Zeroing all rows to 0 collapses every iteration's input_ln to
/// `__l` — i.e., uses layer N's weights when computing layer N+1's
/// input ln. Numerical drift accumulates and decode turns to
/// garbage after a few tokens.
///
/// `iter_index_field_name` is the field name a variant uses to
/// carry its iteration-index value. Today's only such name is
/// `"layer"` (transformer-layer index); the function takes the
/// name as a parameter so the same code works for any future
/// loop construct (e.g., per-head, per-block) that adds a
/// different convention.
/// Where the layer loop comes from.
///
/// ⛔ THESE ARE NOT INTERCHANGEABLE, WHICH IS WHY THEY ARE NAMED. `Shared` means this stream was
/// lowered from the `SubtileTape`, so the loop is the ONE answer `find_layer_loop` gave, mapped
/// through the emitter's step->instruction spans. `LocalSearch` means the stream has no tape
/// correspondence (the instruction-selection path), so the only available answer is a search of
/// the stream itself. An `Option` here would have let a tape-lowered stream silently fall back
/// to the second search — the exact duplication being removed.
pub enum LoopSource {
    #[cfg(any(feature = "metal", feature = "spyre"))]
    Shared(Option<(usize, usize, u32)>),
    LocalSearch,
}

// ⛔ INSTRUCTION-SELECTION ONLY. A tape-scheduled target reads its loop off the shared
// tape; this searches the ISel stream for one because that stream has no tape. Gated so
// metal/spyre do not COMPILE it, rather than compiling it and allowing it to be dead.
#[cfg(not(any(feature = "metal", feature = "spyre")))]
/// `LoopSource::Shared` carries `(start, period, iters)` in `lowered.instances` positions, from
/// [`scratchy_subtile::subtile_tape::find_layer_loop`] — the ONE search, run on the shared
/// `SubtileTape` before any target lowered.
///
/// ⛔ THE LOCAL SEARCH IS A CROSS-CHECK, NOT AN ALTERNATIVE. `detect_repeating_run` still runs,
/// and a disagreement with the shared answer is a build failure naming the model. Two searches
/// over two representations is exactly the duplication being removed; keeping one as an
/// assertion is how the removal is PROVEN rather than asserted.
pub fn apply_loop_compression(
    arch_opcodes: &ArchOpcodes,
    lowered: &mut LoweredBucket,
    iter_index_field_name: &str,
    loop_source: LoopSource,
    // Read only by the tape-path `Shared` arm below; the ISel `LocalSearch`
    // arm has no use for it. Underscored like `_protected` in `lower_bucket`
    // so the signature stays symmetric across targets.
    _what: &str,
) {
    use std::collections::HashMap;

    let mut iter_idx: HashMap<String, usize> = HashMap::new();
    for (name, shape) in arch_opcodes.iter() {
        for (i, (fname, _ty)) in shape.fields.iter().enumerate() {
            if fname == iter_index_field_name {
                iter_idx.insert(name.clone(), i);
                break;
            }
        }
    }

    let local = detect_repeating_run(&lowered.instances, &iter_idx);
    let chosen = match loop_source {
        #[cfg(any(feature = "metal", feature = "spyre"))]
        LoopSource::Shared(shared) => {
            // ⛔ THE STRUCTURE MUST AGREE; THE PHASE NEED NOT, AND THAT IS NOT A FUDGE.
            // `period` and `iters` say WHAT the repeating body is — a disagreement there means
            // the step->instruction span map is wrong and the two targets would run different
            // bodies. `start` says WHERE the run was cut, and the shared re-roll deliberately
            // rotates that cut to the layer boundary with the fewest carried slots (see
            // `reroll_subtile_tape`'s phase rotation), so it legitimately differs from a search
            // that just maximises span. Taking the shared phase is the POINT; that the rewrite
            // preserves semantics at that phase is proven by `verify_loop_compression`, which
            // replays the compressed stream against the original.
            let shape = |r: &Option<(usize, usize, u32)>| r.map(|(_, p, i)| (p, i));
            assert!(
                shape(&shared) == shape(&local),
                "[one-reroll] {_what}: the shared tape's layer loop {shared:?} and the \
                 instruction-stream search {local:?} describe DIFFERENT bodies (period, iters). \
                 They are the same repeating run in the same units, so this means the \
                 step->instruction span map is wrong."
            );
            shared
        }
        LoopSource::LocalSearch => local,
    };
    let Some((start, period, iters)) = chosen else {
        return;
    };

    let span_end = start + period * iters as usize;
    let mut new_instances: Vec<Instruction> = Vec::new();
    let mut new_weight_slots: Vec<Vec<WeightSlot>> = Vec::new();
    new_instances.extend_from_slice(&lowered.instances[..start]);
    new_weight_slots.extend_from_slice(&lowered.weight_slots[..start]);
    new_instances.push(loop_instance(iters, period as u32));
    new_weight_slots.push(Vec::new());
    for off in 0..period {
        let inst = lowered.instances[start + off];
        let var_name = instruction_variant_name(&inst);
        let new_inst = if let Some(&fi) = iter_idx.get(var_name) {
            // Preserve the iter-0 baseline per row — the arm
            // computes `layer = __layer + baseline` at dispatch.
            let baseline = instruction_field_at(&inst, fi).unwrap_or(0) as u32;
            instruction_with_field_set(inst, fi, baseline)
        } else {
            inst
        };
        new_instances.push(new_inst);
        new_weight_slots.push(lowered.weight_slots[start + off].clone());
    }
    new_instances.extend_from_slice(&lowered.instances[span_end..]);
    new_weight_slots.extend_from_slice(&lowered.weight_slots[span_end..]);
    // Compress barriers in lockstep with instances. The body is
    // byte-equivalent across iterations (that's the precondition
    // for loop compression to apply at all), so per-iteration
    // barrier flags also repeat — keep iter-0's body slice. The
    // Loop row inserted ahead of the body gets `false` (no
    // dispatch). Body row 0's flag covers the boundary between
    // the last pre-loop instance (iter 0 case) AND the last
    // body instance of the previous iteration (iter N>0 case);
    // both transitions have the same hazard footprint when the
    // body is byte-equivalent, so the saved flag is correct.
    if !lowered.barriers.is_empty() {
        let mut new_barriers: Vec<bool> = Vec::new();
        new_barriers.extend_from_slice(&lowered.barriers[..start]);
        new_barriers.push(false);
        new_barriers.extend_from_slice(&lowered.barriers[start..start + period]);
        new_barriers.extend_from_slice(&lowered.barriers[span_end..]);
        lowered.barriers = new_barriers;
    }
    lowered.instances = new_instances;
    lowered.weight_slots = new_weight_slots;
}

/// Emit one per-bucket
/// `static <ident>: &[__I] = &[…];` where `__I` is the per-canonical
/// alias for `::scratchy_forward_compiler::Instruction`. Each row is
/// `<Variant>(v0, v1, …)` — tuple-style construction matching the
/// variant declaration order in `Instruction`. The variant + field
/// arity comes directly from the typed `Instruction` value, so the
/// `OpcodeShape` arity check is no longer load-bearing here; we keep
/// `shapes_by_name` in the signature for caller-side
/// shape-registration plumbing but the per-row body trusts the
/// constructor.
pub fn emit_bucket_static_slice(
    static_ident: &syn::Ident,
    _shapes_by_name: &BTreeMap<String, OpcodeShape>,
    instances: &[Instruction],
) -> TokenStream {
    let elements = instances.iter().map(instruction_to_tokens);
    quote! {
        static #static_ident: &[__I] = &[ #(#elements),* ];
    }
}

// ── Helpers ──────────────────────────────────────────────────────

fn assert_shapes_agree(a: &OpcodeShape, b: &OpcodeShape) {
    assert_eq!(
        a.name, b.name,
        "OpcodeShape registration: variant ident mismatch ({} vs {})",
        a.name, b.name
    );
    assert_eq!(
        a.fields.len(),
        b.fields.len(),
        "OpcodeShape `{}`: field count mismatch ({} vs {})",
        a.name,
        a.fields.len(),
        b.fields.len()
    );
    for ((an, at), (bn, bt)) in a.fields.iter().zip(b.fields.iter()) {
        assert_eq!(
            an, bn,
            "OpcodeShape `{}`: field name mismatch ({} vs {})",
            a.name, an, bn
        );
        let a_ty = quote! { #at }.to_string();
        let b_ty = quote! { #bt }.to_string();
        assert_eq!(
            a_ty, b_ty,
            "OpcodeShape `{}` field `{}`: type mismatch ({} vs {})",
            a.name, an, a_ty, b_ty
        );
    }
}

/// The universal `Free` variant codegen always emits. Not Impl-
/// driven; emitted at the drop-pass-determined scheduling points
/// to clear a tile slot.
pub fn free_variant_shape() -> OpcodeShape {
    OpcodeShape::new("Free", vec![("slot", syn::parse_quote!(u32))])
}

/// The universal `Alias` variant codegen emits at the start of
/// every per-bucket slice — one row per zero-copy `View` aliasing
/// pair the lowering surfaced via `output_alias`. The interpreter's
/// arm sets `__tiles[dst] = Some(view(src))`, the same setup the
/// per-bucket fn used to do as a separate prelude. Folding aliases
/// into the slice means there's no per-fn prelude duplication
/// between forward and forward_backbone.
pub fn alias_variant_shape() -> OpcodeShape {
    OpcodeShape::new(
        "Alias",
        vec![
            ("dst", syn::parse_quote!(u32)),
            ("src", syn::parse_quote!(u32)),
        ],
    )
}

/// Construct an `Alias(dst, src)` instance for the alias prelude.
pub fn alias_instance(dst: u32, src: u32) -> Instruction {
    Instruction::Alias(dst, src)
}

/// The universal `Loop` variant the layer-template detection
/// emits when a subsequence of the slice repeats N times. The
/// interpreter sees `Op::Loop(count, body_len)` and runs the
/// next `body_len` ops `count` times, threading the iteration
/// index through as `__layer`. Compresses a 40-layer transformer
/// body from 40·body rows to body+1 rows.
pub fn loop_variant_shape() -> OpcodeShape {
    OpcodeShape::new(
        "Loop",
        vec![
            ("count", syn::parse_quote!(u32)),
            ("body_len", syn::parse_quote!(u32)),
        ],
    )
}

// ⛔ INSTRUCTION-SELECTION ONLY. A tape-scheduled target reads its loop off the shared
// tape; this searches the ISel stream for one because that stream has no tape. Gated so
// metal/spyre do not COMPILE it, rather than compiling it and allowing it to be dead.
#[cfg(not(any(feature = "metal", feature = "spyre")))]
/// Construct a `Loop(count, body_len, layer_stride)` instance the layer-template
/// detection prepends in front of a repeating sub-sequence of the
/// slice.
///
/// The instruction-selection path detects a run of layer-sized bodies, so its stride is one
/// layer per iteration; a wider cell is metal's tape path (see `from_tape::class_rolled_items`).
pub fn loop_instance(count: u32, body_len: u32) -> Instruction {
    Instruction::Loop(count, body_len, 1)
}

// ── Bucket lowering driver ───────────────────────────────────────

/// Rewrite every `MetalSharedFusedMoe`'s `bits` field to carry the three
/// routed-expert projection widths (gate/up/down), packed via
/// [`scratchy_forward_compiler::pack_moe_expert_bits`]. `fan_out` sees only
/// `program` (not the per-module quant map), so the single width it bakes is
/// the `mlp` bundle edge's — the router gate's. This pass runs where the
/// per-variant `model` is in scope and resolves each projection's true
/// on-disk width from the per-module map. No-op for non-affine / uniform
/// checkpoints (all three resolve to the same width).
pub(crate) fn repack_moe_expert_bits(
    emits: Vec<Instruction>,
    program: &Program,
    model: &ModelParams,
) -> Vec<Instruction> {
    use Instruction as I;
    let Some(qc) = model.quantization.as_ref() else {
        return emits;
    };
    emits
        .into_iter()
        .map(|instr| match instr {
            I::MetalSharedFusedMoe(a, b, layer, d, e, f, g, h, i, bits, k) => {
                let idx = Some(UnrollIndex(layer as u64));
                let resolve = |proj: &str| {
                    crate::quantization::affine_role_bits(
                        program,
                        &qc.method,
                        &format!("mlp.switch_mlp.{proj}"),
                        idx,
                    )
                    .unwrap_or(bits)
                };
                let packed = scratchy_forward_compiler::pack_moe_expert_bits(
                    resolve("gate_proj"),
                    resolve("up_proj"),
                    resolve("down_proj"),
                );
                I::MetalSharedFusedMoe(a, b, layer, d, e, f, g, h, i, packed, k)
            }
            other => other,
        })
        .collect()
}

/// Lower one (variant × workload-point) into [`LoweredBucket`].
/// Walks the same wave/loop the old codegen did, calls each picked
/// Impl's `fan_out`, interleaves `Free` instances at drop-pass
/// scheduling points, and registers each Impl's `OpcodeShape` into
/// `arch_opcodes` for shape-checking + iter-index discovery.
///
/// `final_tile` is the `(TileId, output_slot)` whose slot index will
/// be exposed as `LoweredBucket.final_slot`. The full forward passes
/// `(fuf.last(), 0)`; the backbone-only forward passes the input of
/// the skipped terminal subgraph. The drop pass is told to protect
/// this slot via the caller's `protected` set, since the per-bucket
/// fn `take_owned`s it as the return value.
#[allow(clippy::too_many_arguments)]
pub fn lower_bucket(
    fuf: &Fuf,
    sfuf: &Assignment,
    loop_ir: &Loop,
    program: &Program,
    model: &ModelParams,
    lib: &ImplementationLibrary,
    bounds: &BTreeMap<String, u64>,
    skip_subgraph: Option<SubgraphId>,
    // `protected` is consumed by the *caller's* `colored_slot_map`
    // call (which also produces `slots`), not by the lowering walk
    // itself — without an explicit Free pass, `lower_bucket` only
    // emits Alias rows + per-Impl fan_out output, neither of which
    // needs to know which slots are pinned beyond slice end. Kept
    // in the signature so calls stay symmetric with `colored_slot_map`.
    _protected: &HashSet<(TileId, u8)>,
    arch_opcodes: &mut ArchOpcodes,
    // Feeds `final_slot`, which is tape-path only — see the field's gate.
    _final_tile: (TileId, u8),
    slots: &SlotMap,
) -> LoweredBucket {
    #[cfg(any(feature = "metal", feature = "spyre"))]
    let num_slots = slots.total();

    // Aliases: every Impl's output_alias declares which of its
    // outputs borrow from an upstream owner.
    let mut alias_to_owner: HashMap<(TileId, u8), (TileId, u8)> = HashMap::new();
    for wave in &loop_ir.waves {
        for (sg, imp_id) in &wave.subgraphs {
            if Some(*sg) == skip_subgraph {
                continue;
            }
            let claimed = sfuf.tiles_in_subgraph(*sg);
            let imp = lib.get(*imp_id);
            let aliases = crate::alias_rules::output_alias(imp.alias_rule(), &claimed, fuf);
            for (dst, src_opt) in aliases {
                if let Some(src) = src_opt {
                    alias_to_owner.insert(dst, src);
                }
            }
        }
    }
    let resolve_owner = |start: (TileId, u8)| -> (TileId, u8) {
        let mut cur = start;
        let mut seen = HashSet::new();
        while seen.insert(cur) {
            match alias_to_owner.get(&cur) {
                Some(up) => cur = *up,
                None => break,
            }
        }
        cur
    };
    let mut aliases: Vec<(u32, u32)> = alias_to_owner
        .iter()
        .map(|(&dst, _)| {
            let owner = resolve_owner(dst);
            (slots.of(dst.0, dst.1), slots.of(owner.0, owner.1))
        })
        .filter(|(d, s)| d != s)
        .collect();
    aliases.sort();
    aliases.dedup();

    // Prepend Alias rows so the slice is self-contained: running it
    // sets up the zero-copy views, runs the kernels, and frees on its
    // own — no per-bucket fn alias prelude.
    //
    // Note: with `colored_slot_map`, the slot map already collapses
    // dead slots' colors into the free pool the moment they expire.
    // The next writer to that color overwrites the slot's
    // `Some(OwnedTensor)` — Rust drops the old tensor at the
    // overwrite, returning its GPU memory to the caching allocator
    // automatically. Explicit `Op::Free` rows would be redundant
    // (the next write does the same drop) and actively harmful for
    // the layer-template invariant (a Free emitted in layer L but
    // not layer L+1 — because in layer L the color isn't reused
    // before exit, but in layer L+1 it is — would break body byte-
    // equivalence). So we don't emit Free here at all.
    let mut instances: Vec<Instruction> =
        aliases.iter().map(|&(d, s)| alias_instance(d, s)).collect();
    // Alias rows are metadata only — they don't dispatch on any
    // backend — so they get `barrier_before = false`.
    let mut barriers: Vec<bool> = vec![false; instances.len()];
    // Alias rows consume no weights — empty parallel entries.
    let mut weight_slots: Vec<Vec<WeightSlot>> = (0..instances.len()).map(|_| Vec::new()).collect();

    // Hazard-analysis state, shared across waves (one logical MTL4
    // encoder per bucket; barriers flush all pending sets).
    let mut pending_writes: HashSet<u32> = HashSet::new();
    let mut pending_reads: HashSet<u32> = HashSet::new();
    let mut pending_kv_writes: HashSet<u32> = HashSet::new();
    let mut pending_kv_reads: HashSet<u32> = HashSet::new();
    let mut first_dispatch = true;

    for wave in &loop_ir.waves {
        for (sg, imp_id) in &wave.subgraphs {
            if Some(*sg) == skip_subgraph {
                continue;
            }
            let claimed = sfuf.tiles_in_subgraph(*sg);
            if std::env::var_os("M2_RANK_DEBUG").is_some() {
                eprintln!(
                    "[isel-order] sg={} imp={} tiles={:?}",
                    sg.0,
                    lib.get(*imp_id).name(),
                    claimed.iter().map(|t| t.0).collect::<Vec<_>>()
                );
            }
            let imp = lib.get(*imp_id);
            let m = MatchInfo {
                claimed_tiles: claimed.clone(),
                boundary_inputs: collect_boundary_inputs(fuf, &claimed),
                boundary_outputs: claimed.clone(),
            };
            let emits = match imp.fan_out(&m, fuf, program, bounds, slots) {
                Some(e) => e,
                // A claim-only impl (its real lowering happens elsewhere, e.g. the
                // KTIR bundle) declares it emits no host `Instruction`; skip it
                // here. Any other `None` is an unmigrated-Impl bug (panic below).
                None if !imp.emits_host_instruction() => continue,
                None => panic!(
                    "Impl `{name}` (id {id}) has no fan_out — unmigrated to host \
                     interpreter IR. Override `opcode_shape` + `fan_out` on \
                     `{name}`, and ensure the matching `Instruction<W>` variant \
                     exists in `scratchy_forward_compiler::instr`.",
                    name = imp.name(),
                    id = imp_id.0,
                ),
            };
            // OptiQ mixed-bit routed experts: `MetalSharedFusedMoeImpl::fan_out`
            // bakes ONE dispatch bit-width taken from the `mlp` bundle edge,
            // which the storage annotator resolves to the router gate's width
            // (the first `.mlp.*` per-module key). But each routed-expert
            // projection (gate/up/down) has its OWN on-disk width and they
            // differ per layer (Qwen3.5 OptiQ: 35 layers 4-bit, 3 mixed,
            // 2 all-8), so re-resolve all three from the per-variant `model`'s
            // per-module map — in scope here, unlike inside `fan_out` — and
            // pack them into the `bits` field. Uniform checkpoints resolve
            // three identical widths, leaving the instruction unchanged.
            let emits = repack_moe_expert_bits(emits, program, model);
            // Per-Impl weight kinds + base names. Same value for every
            // emitted Instruction in this fan_out (multi-chunk Impls
            // like TkGemmAdd that emit N rows for one logical kernel
            // share the same weight, by design).
            let accs = imp.required_weights(&claimed, fuf, program);
            let slots_for_emit = weight_accessors_to_slots(&accs);
            // Eval bodies live in `scratchy_forward_compiler::Instruction::eval`
            // — register only the shape, used for static-slice
            // emission and `apply_loop_compression`'s per-variant
            // iter-index field discovery. `extra_opcode_shapes`
            // covers storage-polymorphic impls that fan out a
            // multi-variant mix (e.g. metal int4's decomposed q-MLP
            // emits `AffineQmm`/`SiluMul` from the same Impl whose
            // primary `opcode_shape` is `FusedGateUpSiluMul`).
            arch_opcodes.register(imp.opcode_shape());
            for extra in imp.extra_opcode_shapes() {
                arch_opcodes.register(extra);
            }

            // Per-subgraph dataflow signature, sourced from the
            // exact same primitives `colored_slot_map` uses:
            // claimed-tile outputs (alias-resolved) for writes,
            // FufInput::Tile boundary edges (alias-resolved) for
            // reads, plus the impl's `kv_layer_io` declaration for
            // the runtime-ambient KV cache.
            let mut sg_writes: Vec<u32> = Vec::new();
            let mut sg_writes_set: HashSet<u32> = HashSet::new();
            for &t in &claimed {
                let n_out = fuf.get(t).outputs.len().max(1) as u8;
                for s in 0..n_out {
                    let owner = resolve_owner((t, s));
                    let slot = slots.of(owner.0, owner.1);
                    if sg_writes_set.insert(slot) {
                        sg_writes.push(slot);
                    }
                }
            }
            let claimed_set: HashSet<TileId> = claimed.iter().copied().collect();
            let mut sg_reads: Vec<u32> = Vec::new();
            let mut sg_reads_set: HashSet<u32> = HashSet::new();
            for &t in &claimed {
                for input in &fuf.get(t).inputs {
                    if let crate::fuf::FufInput::Tile { id, slot } = input
                        && !claimed_set.contains(id)
                    {
                        let owner = resolve_owner((*id, *slot));
                        let s = slots.of(owner.0, owner.1);
                        if sg_reads_set.insert(s) {
                            sg_reads.push(s);
                        }
                    }
                }
            }
            let (kv_w, kv_r) = crate::alias_rules::kv_layer_io(imp.kv_rule(), &claimed, fuf);

            // Metadata-only emits (Reshape/Alias/Free) lower to ZERO
            // device dispatches — every metal `lower_one` returns an
            // empty command Vec for them, so a `true` flag computed
            // here is silently DROPPED by the runtime tape assembly
            // (flags are only pushed for n_cmds >= 1). Letting such a
            // row "fire" a barrier and CLEAR the pending sets opens a
            // lost-barrier hole: the next real dispatch sees an
            // analytically-flushed state that no runtime barrier ever
            // flushed. (Found on Gemma4: the q-head Reshape between
            // v_proj's write and rmsnorm_unit's read of the same slot
            // cleared the pending write → the unit norm raced the NAX
            // matmul.) Metadata rows are excluded from hazard
            // bookkeeping entirely; their aliasing semantics are
            // already covered by `resolve_owner`, which maps every
            // downstream reader to the owning slot.
            let metadata_only = !emits.is_empty()
                && emits.iter().all(|e| {
                    matches!(
                        e,
                        scratchy_forward_compiler::Instruction::Reshape(..)
                            | scratchy_forward_compiler::Instruction::Alias(..)
                            | scratchy_forward_compiler::Instruction::Free(..)
                    )
                });

            // Hazard check against pending sets. RAW (my reads ∩
            // pending writes) + WAW (my writes ∩ pending writes) +
            // WAR (my writes ∩ pending reads) + KV-layer
            // equivalents.
            let arena_conflict = sg_reads.iter().any(|s| pending_writes.contains(s))
                || sg_writes.iter().any(|s| pending_writes.contains(s))
                || sg_writes.iter().any(|s| pending_reads.contains(s));
            let kv_conflict = kv_r
                .map(|l| pending_kv_writes.contains(&l))
                .unwrap_or(false)
                || kv_w
                    .map(|l| pending_kv_writes.contains(&l) || pending_kv_reads.contains(&l))
                    .unwrap_or(false);
            let need_barrier = !metadata_only && !first_dispatch && (arena_conflict || kv_conflict);
            if need_barrier {
                pending_writes.clear();
                pending_reads.clear();
                pending_kv_writes.clear();
                pending_kv_reads.clear();
            }
            // Per-emit barrier flag. The first emit of an Impl's
            // fan_out picks up the hazard flag we computed; any
            // subsequent emits (e.g. AffineQmmTSplitK's qmm_t →
            // reduce pair sharing scratch) are conservatively
            // serialized — Impls that need internal concurrency
            // can refine this later. Metadata-only rows always get
            // `false` (the runtime drops their flags anyway).
            for (i, _emit) in emits.iter().enumerate() {
                barriers.push(if metadata_only {
                    false
                } else if i == 0 {
                    need_barrier
                } else {
                    true
                });
            }
            // Update pending sets after recording the flag —
            // metadata-only rows contribute nothing (no dispatch =
            // no device-memory access to track).
            if !metadata_only {
                pending_writes.extend(sg_writes.iter().copied());
                pending_reads.extend(sg_reads.iter().copied());
                if let Some(l) = kv_w {
                    pending_kv_writes.insert(l);
                }
                if let Some(l) = kv_r {
                    pending_kv_reads.insert(l);
                }
                if !emits.is_empty() {
                    first_dispatch = false;
                }
            }
            // Rotary cos_sin cache is keyed off the *claim*, not a
            // weight ref — `RotaryLocal` is an `Extern` input on the
            // rope-consuming tile, so it never appears in
            // `required_weights`. Inject a `WeightSlot` of kind
            // `CosSin` for any emitted Instruction whose `eval` body
            // calls `wm.cos_sin_at(...)`. Base ident picks
            // `rotary_local` if any tile in the claim references the
            // local rotary extern, else `rotary` — matching the field
            // names emitted on the per-arch `Weights` struct.
            let rotary_base = rotary_base_for_claim(fuf, &claimed);
            // Distribute `slots_for_emit` across this Impl's emits in
            // declaration order — each emit takes the next
            // `instruction_weight_count(inst)` accessors. Cloning the
            // whole vec to every emit (the post-lift bug) routes slot
            // 0 to the wrong accessor on every non-leading emit and
            // produces garbage output. The Impl is responsible for
            // returning `required_weights` in the same order its
            // `fan_out` lays out its emits.
            let mut slots_cursor = 0usize;
            let imp_name = imp.name();
            for inst in emits {
                let n_w = instruction_weight_count(&inst);
                let slots_this_vec: Vec<WeightSlot> = slots_for_emit
                    .iter()
                    .skip(slots_cursor)
                    .take(n_w)
                    .cloned()
                    .collect();
                // Compile-time coverage check: the Impl's
                // `required_weights` MUST supply at least
                // `instruction_weight_count(inst)` accessors per emit,
                // in the same order the runtime Instruction will resolve
                // them through `WeightAccessors::*_at`. Silent
                // under-supply here surfaces at worker-init time as
                // `WeightAccessors::<method>: no match for ...` — a
                // cryptic panic the user can't act on. Fail at macro
                // expansion instead so the offending Impl is named
                // directly. `slots_for_emit` length total is
                // double-checked after the loop.
                assert_eq!(
                    slots_this_vec.len(),
                    n_w,
                    "Impl `{}` `required_weights` returned too few accessors for \
                     Instruction `{}` at op_idx={} in this claim: \
                     instruction_weight_count() = {}, but only {} slot(s) remain. \
                     The Impl's `required_weights` MUST yield exactly the kinds \
                     and count the emitted Instruction will resolve through \
                     `WeightAccessors::*_at` at runtime — in the same order as \
                     the emits.",
                    imp_name,
                    instruction_variant_name(&inst),
                    instances.len(),
                    n_w,
                    slots_this_vec.len(),
                );
                let mut slots_this = slots_this_vec;
                slots_cursor += n_w;
                if instruction_consumes_rotary(&inst) {
                    slots_this.push(WeightSlot {
                        kind: crate::weight_vocab::WeightKind::CosSin,
                        base: rotary_base.to_string(),
                    });
                }
                instances.push(inst);
                weight_slots.push(slots_this);
            }
            // Compile-time check #2: the Impl must not over-supply its
            // `required_weights` either — any trailing accessors past
            // the cumulative `instruction_weight_count` would be silently
            // dropped, hiding a real authoring bug (e.g. an Impl that
            // declared an unused LinearLayer accessor and then refactored
            // its emits to one fewer Instruction).
            assert_eq!(
                slots_cursor,
                slots_for_emit.len(),
                "Impl `{}` `required_weights` returned {} accessors but its \
                 emits only consume {}. Either trim `required_weights` or \
                 split the extra accessor onto an additional Instruction emit.",
                imp_name,
                slots_for_emit.len(),
                slots_cursor,
            );
        }
    }

    debug_assert_eq!(barriers.len(), instances.len());
    #[cfg(any(feature = "metal", feature = "spyre"))]
    let final_slot = slots.of(_final_tile.0, _final_tile.1);

    LoweredBucket {
        barriers,
        instances,
        weight_slots,
        #[cfg(any(feature = "metal", feature = "spyre"))]
        num_slots,
        #[cfg(any(feature = "metal", feature = "spyre"))]
        final_slot,
    }
}

/// Convert the per-Impl `required_weights` output into the
/// `Vec<WeightSlot>` parallel-array entry used by the
/// `WeightAccessors` walker. Each `WeightAccessor` has a
/// `<base>_<layer>` name (or just `<base>` for arch-wide accessors)
/// and a Rust type that selects the `WeightKind` variant. The
/// `_<layer>` suffix is stripped — the per-arch impl receives `layer`
/// as a runtime arg.
pub fn weight_accessors_to_slots(
    accessors: &[crate::weight_vocab::WeightAccessor],
) -> Vec<WeightSlot> {
    use crate::weight_vocab::WeightKind;
    accessors
        .iter()
        .map(|acc| {
            let (base, _layer) = crate::codegen::split_base_layer(&acc.name.to_string());
            let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
            // Rust-type → WeightKind. The string is the rendered
            // TokenStream (whitespace varies but the path tokens are
            // stable). Match on the trailing PascalCase ident.
            let ts_str = acc.rust_type.to_string();
            let kind = if ts_str.ends_with("RmsNorm") {
                WeightKind::RmsNorm
            } else if ts_str.ends_with("AffineQuantEmbedding") {
                // Order matters: `AffineQuantEmbedding` also ends with
                // `Embedding`, so the affine variant must be checked
                // first to avoid mis-classification into the dense
                // `Embedding` kind (which routes via `embedding_at` and
                // surfaces as `&Embedding` vs `&AffineQuantEmbedding`
                // type mismatch on `mlx-community/*-4bit` checkpoints).
                WeightKind::AffineQuantEmbedding
            } else if ts_str.ends_with("Embedding") {
                WeightKind::Embedding
            } else if ts_str.ends_with("LinearLayer") {
                WeightKind::Linear
            } else if ts_str.ends_with("LayerNorm") {
                WeightKind::LayerNorm
            } else if ts_str.ends_with("MarlinLinear") {
                WeightKind::Marlin
            } else if ts_str.ends_with("Bnb4bitLinear") {
                WeightKind::Bnb4
            } else if ts_str.ends_with("Fp8AnyLinear") {
                WeightKind::Fp8
            } else if ts_str.ends_with("DeepSeekV2MoELayer") {
                WeightKind::DeepSeekMoe
            } else if ts_str.ends_with("DeepSeekV2Fp8BlockMoELayer") {
                WeightKind::DeepSeekMoeFp8
            } else if ts_str.ends_with("DeepSeekV2GgmlMoELayer") {
                WeightKind::DeepSeekMoeGgml
            } else if ts_str.ends_with("SharedFusedMoELayer") {
                // Order matters: "SharedFusedMoELayer" also ends with
                // "FusedMoELayer", so the shared variant must be
                // checked first to avoid mis-classification into the
                // `FusedMoe` kind (which routes via `fused_moe_at` and
                // panics with a `&FusedMoELayer` vs `&SharedFusedMoELayer`
                // type mismatch on Qwen2-MoE / Qwen3-MoE).
                WeightKind::SharedFusedMoe
            } else if ts_str.ends_with("FusedMoELayer") {
                WeightKind::FusedMoe
            } else if ts_str.ends_with("GemmaRouterLayer") {
                WeightKind::GemmaRouter
            } else if ts_str.ends_with("SwitchGluExpertsLayer") {
                WeightKind::GemmaSwitchGlu
            } else if ts_str.ends_with("GatedDeltaNetLayer") {
                WeightKind::GatedDeltaNet
            } else if ts_str.ends_with("GpuTensor") {
                WeightKind::CosSin
            } else {
                panic!(
                    "weight_accessors_to_slots: unknown WeightAccessor rust_type `{}` \
                     for accessor `{}`",
                    ts_str, acc.name,
                );
            };
            WeightSlot {
                kind,
                base: base_ident.to_string(),
            }
        })
        .collect()
}

/// True for any [`Instruction`] variant whose `eval` body calls
/// `wm.cos_sin_at(...)`. The walker uses this to inject a `CosSin`
/// `WeightSlot` parallel-array entry for each such emit, since
/// rotary cache is sourced from an `Extern` input rather than a
/// `Weight` ref and so never appears in `required_weights`.
///
/// Keep in sync with the runtime arms in
/// `scratchy_forward_compiler::Instruction::eval` — any new variant that calls
/// `wm.cos_sin_at` needs a matching arm here.
/// How many `WeightSlot` entries (excluding the auto-injected CosSin
/// for rotary-consuming ops) each emitted `Instruction` variant
/// consumes from the Impl's `required_weights` list. Used by
/// `lower_bucket` to distribute the per-Impl accessor list across a
/// multi-Instruction `fan_out` (e.g. metal's affine-decomposed
/// gate/up/silu_mul: `[gate-Linear, up-Linear]` → AffineQmm(gate)
/// gets the gate accessor, AffineQmm(up) gets the up accessor,
/// SiluMul gets none). Cloning `slots_for_emit` whole to every emit
/// would route slot 0 to *the wrong accessor* on every non-leading
/// emit and produce garbage output post-lift.
///
/// Default (`0`) is correct for any Instruction that doesn't pull a
/// per-layer/per-op weight at codegen time — control-flow rows
/// (`Loop`, `Reshape`, `Alias`, `Free`), elementwise rows
/// (`Add`, `Mul`, `Silu`, `SiluMul`), bias-add rows (the bias is
/// resolved against the upstream Linear's `AffineLinearBias` field),
/// runtime-only attention reads, etc. Variants that explicitly
/// declare a typed weight accessor return ≥1.
pub fn instruction_weight_count(inst: &Instruction) -> usize {
    use Instruction as I;
    match inst {
        // Single LinearLayer / quant-sibling per instance — dense Gemm
        // + every cutlass tile-zoo variant emits one `LinearLayer`
        // accessor; the GEMM-then-bias / GEMM-then-add fusions wrap
        // that single accessor in a richer epilogue. Quant flavors
        // (Marlin / Bnb4 / Ggml / Fp8) hit the same eval shape via
        // their own typed accessor (`marlin_at`, `bnb4_at`,
        // `ggml_at`, `fp8_at`).
        I::AffineQmm(..)
        | I::Nvfp4Qmm(..)
        | I::Gemm(..)
        | I::CutlassGemm(..)
        | I::CutlassGemv(..)
        | I::CutlassGemmSplitK(..)
        | I::CutlassGemmAdd(..)
        | I::CutlassFusedGemmBias(..)
        | I::FusedCublasGemmAdd(..)
        | I::FusedGemmBias(..)
        | I::Fp8Gemm(..)
        | I::Fp8FusedGemmBias(..)
        | I::MarlinGemm(..)
        | I::GgmlGemm(..)
        | I::Bnb4Gemm(..) => 1,
        // Singleton rmsnorm flavors. `ScalarOffsetRmsNorm` is Gemma2/
        // Gemma3-style zero-init `(1 + w)`; `MeanSubRmsNorm` and
        // `MeanSubRmsNormBiasAdd` are Cohere-style mean-subtract +
        // rms; `FusedAddRmsNormWithOffset` is the Gemma-style
        // residual-stream variant that folds the +offset into the
        // norm. Each resolves one weight accessor at slot 0.
        I::ScalarOffsetRmsNorm(..)
        | I::MeanSubRmsNorm(..)
        | I::MeanSubRmsNormBiasAdd(..)
        | I::FusedAddRmsNormWithOffset(..) => 1,
        // CUTLASS fused (RmsNorm-flavor + Gemm) — the norm weight and
        // the GEMM weight ride as separate accessors because the
        // CUTLASS epilogue still expects a contiguous `[M, K]` post-
        // norm input. RmsNorm at slot 0 + Linear at slot 0.
        I::CutlassFusedRmsNormGemm(..)
        | I::CutlassFusedMeanSubRmsNormGemm(..)
        | I::CutlassFusedAddRmsNormGemm(..) => 2,
        // Vision positional embedding (LLaVA / Qwen2-VL family).
        // Resolves one `Embedding` accessor at slot 0.
        I::PosEmbed(..) => 1,
        // Singleton bias-add (Qwen2-style biased QKV when the synth
        // megakernel doesn't claim the chain). Resolves the upstream
        // Gemm's `LinearLayer::affine_linear_bias` via its own
        // standalone accessor — `MetalBiasAddImpl::required_weights`
        // returns one `LinearLayer` accessor pointing at the same
        // weight ref the upstream Gemm holds.
        I::MetalBiasAdd(..) => 1,
        // Single typed embedding.
        I::Embed(..) => 1,
        #[cfg(feature = "metal")]
        I::AffineEmbed(..) => 1,
        // Single RmsNorm.
        I::RmsNorm(..) | I::FusedAddRmsNorm(..) => 1,
        // ScalarWeightMul consumes its layer_scalar[layer] accessor
        // (RmsNorm-kind [1] vector) through the same tape locator path.
        I::ScalarWeightMul(..) => 1,
        // NormAddScalarMul: post-FFN norm gains [hidden] + the
        // layer_scalar [1] — both RmsNorm-kind, sub-slots 0/1.
        I::NormAddScalarMul(..) => 2,
        // RopeAppendNormed: q_norm + k_norm gains (RmsNorm-kind,
        // sub-slots 0/1). CosSin is auto-injected by the rotary
        // check, not counted here (mirrors FusedQkvQkNormRopeCache).
        I::RopeAppendNormed(..) => 2,
        // Megakernels — same accessor inventory as their unfused
        // chains (RmsNorm + 3 LinearLayer / RmsNorm + 2 LinearLayer /
        // 2 LinearLayer). CosSin is auto-injected on top by the
        // rotary check, not counted here.
        I::SynthPreAttn(..) => 4,
        I::SynthMlpPreDown(..) => 3,
        #[cfg(feature = "metal")]
        I::SynthGateUpSiluMul(..) => 2,
        // Fused QKV+RoPE family (cuda). The dense + cutlass + quant
        // (marlin / bnb4 / ggml / fp8) variants all run a SINGLE
        // packed `[q | k | v]` GEMM through one accessor — the loader
        // concats the three projection weights at load time and
        // `required_weights` returns one `LinearLayer` (or quant
        // sibling) covering the whole packed weight. The runtime
        // eval reads exactly `linear_at(bucket, op_idx, 0, layer)`.
        I::FusedQkvRopeCache(..)
        | I::FusedQkvRopePrefill(..)
        | I::CutlassFusedQkvRopeCache(..)
        | I::CutlassFusedQkvRopePrefill(..)
        | I::MarlinFusedQkvRopeCache(..)
        | I::MarlinFusedQkvRopePrefill(..)
        | I::Bnb4FusedQkvRopeCache(..)
        | I::Bnb4FusedQkvRopePrefill(..)
        | I::GgmlFusedQkvRopeCache(..)
        | I::GgmlFusedQkvRopePrefill(..)
        | I::Fp8FusedQkvRopeCache(..)
        | I::Fp8FusedQkvRopePrefill(..) => 1,
        // Qwen3-style fused QK-norm path can't fold the three
        // projection weights together — `qk_norm_rope_inplace` needs
        // contiguous `[T, num_heads, head_dim]` Q/K, which a packed
        // cuBLAS output doesn't deliver — so this Impl emits three
        // separate `LinearLayer` accessors plus two `RmsNorm`
        // accessors (q_norm / k_norm). CosSin is auto-injected by
        // the rotary check, not counted here.
        I::FusedQkvQkNormRopeCache(..) => 5,
        // Dense + quant-flavored fused Gate/Up+SiluMul: a single
        // packed `[gate|up]` LinearLayer accessor (loader concats the
        // two source weights at load time). The Affine path decomposes
        // into separate `AffineQmm` + `AffineQmm` + `SiluMul` emits via
        // `affine_decomposed_fan_out` and never reaches this arm.
        I::FusedGateUpSiluMul(..)
        | I::FusedGateUpGeluMul(..)
        | I::CutlassFusedGateUpSiluMul(..)
        | I::CutlassFusedGateUpGeluMul(..)
        | I::MarlinFusedGateUpSiluMul(..)
        | I::MarlinFusedGateUpGeluMul(..)
        | I::Bnb4FusedGateUpSiluMul(..)
        | I::Bnb4FusedGateUpGeluMul(..)
        | I::GgmlFusedGateUpSiluMul(..)
        | I::GgmlFusedGateUpGeluMul(..)
        | I::Fp8FusedGateUpSiluMul(..)
        | I::Fp8FusedGateUpGeluMul(..) => 1,
        // MoE Instructions consume one full MoE layer accessor
        // (`FusedMoELayer` / `SharedFusedMoELayer` / `DeepSeekV2MoELayer`
        // / Fp8 / Ggml siblings). The Impl's `required_weights`
        // returns exactly one accessor per Instruction; counting it
        // here makes the macro register a (bucket, op_idx, 0) →
        // wm.<base>(layer) arm on the appropriate accessor method.
        // Without this slot the per-arch `WeightAccessors` impl falls
        // back to the trait's `unreachable!()` default and panics at
        // first MoE forward.
        I::FusedMoe(..) | I::MetalFusedMoe(..) => 1,
        I::SharedFusedMoe(..) | I::MetalSharedFusedMoe(..) => 1,
        // GemmaMoe consumes TWO weight bundles: the `router` bundle
        // (GemmaRouterLayer, slot 0 of its kind) and the
        // `experts.switch_glu` bundle (SwitchGluExpertsLayer, slot 0 of
        // its kind). `required_weights` returns exactly these two.
        I::GemmaMoe(..) => 2,
        I::DeepSeekMoe(..) | I::DeepSeekMoeFp8Block(..) | I::DeepSeekMoeGgml(..) => 1,
        // GatedDeltaNet consumes one `linear_attn[layer]` weight bundle
        // (GatedDeltaNetLayer). The ambient conv/ssm state is read from
        // `ForwardCtx`, not a codegen-time weight.
        I::GatedDeltaNet(..) => 1,
        // Everything else: no codegen-time weight, or weight resolved
        // via a different path (MetalBiasAdd through the upstream
        // Linear's `AffineLinearBias` field, attention reads through
        // runtime KV cache, etc).
        _ => 0,
    }
}

pub fn instruction_consumes_rotary(inst: &Instruction) -> bool {
    use Instruction as I;
    matches!(
        inst,
        I::FusedQkvRopeCache(..)
            | I::FusedQkvQkNormRopeCache(..)
            | I::FusedQkvRopePrefill(..)
            | I::AttentionViaCache(..)
            | I::SlidingAttentionViaCache(..)
            | I::FlashInferAttentionDecode(..)
            | I::RopeAppend(..)
            | I::RopeAppendNormed(..)
            | I::CutlassFusedQkvRopeCache(..)
            | I::CutlassFusedQkvRopePrefill(..)
            | I::MarlinFusedQkvRopeCache(..)
            | I::MarlinFusedQkvRopePrefill(..)
            | I::GgmlFusedQkvRopeCache(..)
            | I::GgmlFusedQkvRopePrefill(..)
            | I::Bnb4FusedQkvRopeCache(..)
            | I::Bnb4FusedQkvRopePrefill(..)
            | I::Fp8FusedQkvRopeCache(..)
            | I::Fp8FusedQkvRopePrefill(..)
            | I::MlaAttention(..)
            // Metal compiler-synth megakernel: fuses RMSNorm + QKV
            // GEMMs + RoPE + paged KV-write into one dispatch; the
            // RoPE step inside reads the per-arch rotary cos/sin
            // cache, so the macro must inject a `CosSin` slot for
            // this op.
            | I::SynthPreAttn(..)
    )
}

/// Pick the rotary field on the per-arch `Weights` struct that this
/// claim consumes. Mirrors the legacy `rotary_cos_sin_tokens` helper:
/// if any claimed tile carries an `ExternKind::RotaryLocal` input,
/// the claim's rope op pulls from `wm.rotary_local`; else from
/// `wm.rotary`. Both fields are emitted by `codegen.rs` per arch.
pub fn rotary_base_for_claim(fuf: &Fuf, claimed: &[TileId]) -> syn::Ident {
    let uses_local = claimed.iter().any(|&tid| {
        fuf.get(tid).inputs.iter().any(|i| {
            matches!(
                i,
                FufInput::Extern {
                    kind: ExternKind::RotaryLocal,
                    ..
                }
            )
        })
    });
    let name = if uses_local { "rotary_local" } else { "rotary" };
    syn::Ident::new(name, proc_macro2::Span::call_site())
}

pub fn collect_boundary_inputs(fuf: &Fuf, claimed: &[TileId]) -> Vec<TileId> {
    let claimed_set: HashSet<TileId> = claimed.iter().copied().collect();
    let mut seen: HashSet<TileId> = HashSet::new();
    let mut out = Vec::new();
    for &t in claimed {
        for input in &fuf.get(t).inputs {
            if let FufInput::Tile { id, .. } = input
                && !claimed_set.contains(id)
                && seen.insert(*id)
            {
                out.push(*id);
            }
        }
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(all(test, any(feature = "cuda", feature = "spyre")))]
mod tests {
    use super::*;
    use crate::fuf::FufNode;
    use crate::shape::Dim;
    use quote::format_ident;

    /// Slot allocation packs (tile, output_slot) pairs in
    /// topological order. Test pins the order so emitted
    /// constructors and runtime indexing line up.
    #[test]
    fn slot_map_packs_fuf_outputs_densely() {
        let f = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: crate::classified::OpKind::Add,
                    inputs: vec![],
                    outputs: vec![vec![Dim::Lit(1)]],
                },
                FufNode {
                    id: TileId(1),
                    op: crate::classified::OpKind::Reshape,
                    inputs: vec![],
                    outputs: vec![vec![Dim::Lit(1)], vec![Dim::Lit(1)], vec![Dim::Lit(1)]],
                },
                FufNode {
                    id: TileId(2),
                    op: crate::classified::OpKind::Add,
                    inputs: vec![],
                    outputs: vec![vec![Dim::Lit(1)]],
                },
            ],
        };
        let sm = build_slot_map(&f);
        assert_eq!(sm.total(), 5);
        assert_eq!(sm.of(TileId(0), 0), 0);
        assert_eq!(sm.of(TileId(1), 0), 1);
        assert_eq!(sm.of(TileId(1), 1), 2);
        assert_eq!(sm.of(TileId(1), 2), 3);
        assert_eq!(sm.of(TileId(2), 0), 4);
    }

    // ── Coloring invariants ─────────────────────────────────────
    //
    // These tests construct synthetic FUFs + Assignments + Loops +
    // a tiny test-only ImplementationLibrary, run `colored_slot_map`,
    // and assert structural properties of the resulting SlotMap.
    // They're independent of any real arch's DSL: the point is to
    // pin invariants the colorer must maintain regardless of what
    // gets lowered through it.

    use crate::assignment::ImplId;
    use crate::assignment::{Assignment, SubgraphId};
    use crate::classified::OpKind;
    use crate::fuf::FufInput;
    use crate::impl_lib::{
        CostCtx, Handoff, ImplementationLibrary, LaunchKind, Layout, MatchInfo, Resources,
    };
    use crate::schedule::{Loop, Wave};
    use crate::target::TargetProfile;
    use crate::weight_vocab::WeightAccessor;
    use std::collections::HashSet;

    /// A test-only Impl that claims exactly one tile per subgraph and
    /// reports a configurable alias rule / `consumes_input_tiles`.
    /// Lets the synthetic-FUF tests below exercise alias and consume
    /// constraints on the colorer without dragging in any real
    /// kernel-emitting Impl.
    #[derive(Debug)]
    struct StubImpl {
        name: &'static str,
        rule: crate::alias_rules::AliasRule,
        consumes: Vec<(TileId, u8)>,
    }

    impl crate::impl_lib::Implementation for StubImpl {
        fn name(&self) -> &'static str {
            self.name
        }
        fn target_compatible(&self, _profile: &TargetProfile) -> bool {
            true
        }
        fn matches(
            &self,
            _fuf: &Fuf,
            _seed: TileId,
            _profile: &TargetProfile,
        ) -> Option<MatchInfo> {
            None
        }
        fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
            0.0
        }
        fn resources(&self, _m: &MatchInfo) -> Resources {
            Resources::ZERO
        }
        fn launch_kind(&self) -> LaunchKind {
            LaunchKind::HostCallback
        }
        fn supported_input_handoffs(&self) -> &[Handoff] {
            &[]
        }
        fn supported_output_handoffs(&self) -> &[Handoff] {
            &[]
        }
        fn input_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
            vec![]
        }
        fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
            vec![]
        }
        fn required_weights(
            &self,
            _claimed: &[TileId],
            _fuf: &Fuf,
            _program: &crate::classified::Program,
        ) -> Vec<WeightAccessor> {
            vec![]
        }
        fn alias_rule(&self) -> crate::alias_rules::AliasRule {
            self.rule
        }
        fn consumes_input_tiles(&self, _claimed: &[TileId], _fuf: &Fuf) -> Vec<(TileId, u8)> {
            self.consumes.clone()
        }
    }

    /// Build a single-output Add tile with the given upstream Tile
    /// inputs (each as `(producer_tile, output_slot)`).
    fn add_tile(id: u32, inputs: &[(TileId, u8)]) -> FufNode {
        FufNode {
            id: TileId(id),
            op: OpKind::Add,
            inputs: inputs
                .iter()
                .map(|&(id, slot)| FufInput::Tile { id, slot })
                .collect(),
            outputs: vec![vec![Dim::Lit(1)]],
        }
    }

    /// Build a single-tile per-subgraph Assignment with one
    /// Impl per subgraph, in tile-id order.
    fn linear_assignment(tiles: &[TileId], impls: &[ImplId]) -> Assignment {
        assert_eq!(tiles.len(), impls.len());
        let mut a = Assignment::default();
        for (i, &t) in tiles.iter().enumerate() {
            let sg = SubgraphId(i as u32);
            a.cover.insert(t, sg);
            a.impls.insert(sg, impls[i]);
        }
        a
    }

    /// Linear schedule: one wave per subgraph, in id order.
    fn linear_loop(n_subgraphs: u32) -> Loop {
        let waves = (0..n_subgraphs)
            .map(|i| Wave {
                subgraphs: vec![(SubgraphId(i), ImplId(0))], // ImplId here is unused by colorer
            })
            .collect();
        Loop { waves }
    }

    /// 3-node chain `t0 → t1 → t2`, no aliases. With `lu <= dp`
    /// kill-before-def semantics, every tile lands at the same slot:
    /// t0's color is freed at pos-1 (t1's def), t1 reuses it; same
    /// at pos-2. The kernel call's read of slot 0 happens before
    /// the overwrite (`__tiles[0] = Some(new)` drops the old
    /// `OwnedTensor` only after the kernel was already queued
    /// against its view), so 1 color is correct here.
    #[test]
    fn coloring_chain_collapses_to_one_color() {
        let f = Fuf {
            nodes: vec![
                add_tile(0, &[]),
                add_tile(1, &[(TileId(0), 0)]),
                add_tile(2, &[(TileId(1), 0)]),
            ],
        };
        let mut lib = ImplementationLibrary::new();
        let id_plain = lib.push(Box::new(StubImpl {
            name: "stub",
            rule: crate::alias_rules::AliasRule::OwnAll,
            consumes: vec![],
        }));
        let sfuf = linear_assignment(&[TileId(0), TileId(1), TileId(2)], &[id_plain; 3]);
        let lp = linear_loop(3);
        let protected: HashSet<(TileId, u8)> = HashSet::new();
        let sm = colored_slot_map(&f, &sfuf, &lp, &lib, None, &protected);

        assert_eq!(sm.total(), 1, "serial chain coalesces to one slot");
        assert_eq!(sm.of(TileId(0), 0), sm.of(TileId(1), 0));
        assert_eq!(sm.of(TileId(1), 0), sm.of(TileId(2), 0));
    }

    /// A diamond `t0 → t1, t0 → t2, then t3 = f(t1, t2)` — t1 and
    /// t2 must coexist (both read t0; both written before t3 reads
    /// them) so they get distinct colors. Pins the must-not-collapse
    /// half of the colorer's job.
    #[test]
    fn coloring_diamond_keeps_concurrent_outputs_distinct() {
        let f = Fuf {
            nodes: vec![
                add_tile(0, &[]),
                add_tile(1, &[(TileId(0), 0)]),
                add_tile(2, &[(TileId(0), 0)]),
                add_tile(3, &[(TileId(1), 0), (TileId(2), 0)]),
            ],
        };
        let mut lib = ImplementationLibrary::new();
        let id_plain = lib.push(Box::new(StubImpl {
            name: "stub",
            rule: crate::alias_rules::AliasRule::OwnAll,
            consumes: vec![],
        }));
        let sfuf = linear_assignment(
            &[TileId(0), TileId(1), TileId(2), TileId(3)],
            &[id_plain; 4],
        );
        let lp = linear_loop(4);
        let protected: HashSet<(TileId, u8)> = HashSet::new();
        let sm = colored_slot_map(&f, &sfuf, &lp, &lib, None, &protected);

        // t1 and t2 are co-live at the moment t2 is being written
        // (t1 was just written, t2 is being written, both must
        // remain in __tiles for t3 to read). Distinct colors.
        assert_ne!(sm.of(TileId(1), 0), sm.of(TileId(2), 0));
    }

    /// A `protected` slot's color must NEVER appear on any other
    /// tile — the per-bucket fn's `take_owned(final_slot)` runs at
    /// the very end and would corrupt other tiles if their colors
    /// collided with `final_slot`. The colorer keeps protected
    /// colors out of the free pool forever; here that means t0's
    /// color is unique even though t0 has no readers past pos-1.
    /// (t1 and t2 may still share a color with each other — that's
    /// fine.)
    #[test]
    fn coloring_protected_slot_color_is_unique() {
        let f = Fuf {
            nodes: vec![
                add_tile(0, &[]),
                add_tile(1, &[(TileId(0), 0)]),
                add_tile(2, &[(TileId(1), 0)]),
            ],
        };
        let mut lib = ImplementationLibrary::new();
        let id_plain = lib.push(Box::new(StubImpl {
            name: "stub",
            rule: crate::alias_rules::AliasRule::OwnAll,
            consumes: vec![],
        }));
        let sfuf = linear_assignment(&[TileId(0), TileId(1), TileId(2)], &[id_plain; 3]);
        let lp = linear_loop(3);
        let mut protected: HashSet<(TileId, u8)> = HashSet::new();
        protected.insert((TileId(0), 0));
        let sm = colored_slot_map(&f, &sfuf, &lp, &lib, None, &protected);

        let c_protected = sm.of(TileId(0), 0);
        let c_t1 = sm.of(TileId(1), 0);
        let c_t2 = sm.of(TileId(2), 0);
        assert_ne!(
            c_protected, c_t1,
            "protected color must not be reused by t1"
        );
        assert_ne!(
            c_protected, c_t2,
            "protected color must not be reused by t2"
        );
    }

    /// Same-shape alias collapses: dst and source share the slot.
    /// `output_alias` says t1 aliases t0's storage; t0 and t1 have
    /// the same shape (both `[1]` here, modeling an in-place mutator
    /// like `cutlass_gemm_add` whose output IS the residual buffer
    /// post-mutation). The runtime never has two `__tiles` entries
    /// for one buffer — so the colorer must place them at the same
    /// color, no `View` indirection, no `Op::Alias` row.
    #[test]
    fn coloring_same_shape_alias_collapses_to_source() {
        let f = Fuf {
            nodes: vec![
                add_tile(0, &[]),               // source, shape [1]
                add_tile(1, &[(TileId(0), 0)]), // alias-dst, shape [1]
                add_tile(2, &[(TileId(1), 0)]), // reads via the alias
            ],
        };
        let mut lib = ImplementationLibrary::new();
        let id_plain = lib.push(Box::new(StubImpl {
            name: "stub",
            rule: crate::alias_rules::AliasRule::OwnAll,
            consumes: vec![],
        }));
        let id_alias = lib.push(Box::new(StubImpl {
            name: "stub_alias",
            rule: crate::alias_rules::AliasRule::AliasToInput(0),
            consumes: vec![],
        }));
        let sfuf = linear_assignment(
            &[TileId(0), TileId(1), TileId(2)],
            &[id_plain, id_alias, id_plain],
        );
        let lp = linear_loop(3);
        let protected: HashSet<(TileId, u8)> = HashSet::new();
        let sm = colored_slot_map(&f, &sfuf, &lp, &lib, None, &protected);

        assert_eq!(
            sm.of(TileId(0), 0),
            sm.of(TileId(1), 0),
            "same-shape alias dst must share color with source (in-place mutation)",
        );
    }

    /// `Instruction::AllReduce` (the row-parallel TP communicator)
    /// is one-tile in-place same-shape — its output IS the input
    /// buffer post-mutation, just like `cutlass_gemm_add` /
    /// `fused_add_rms_norm`. Coloring must place the AllReduce
    /// output at the same slot as its input, no `View` indirection.
    /// At runtime this lets `NcclGroup::all_reduce_inplace` operate
    /// directly on the tile slot the gemm output landed in, with no
    /// alias-row preamble in the bucket slice.
    ///
    /// Shape is the realistic gemm-output shape `[N, H]` with H>1
    /// rather than `[1]` — `coloring_disjoint_lifetimes_dont_share_
    /// across_shapes` is the load-bearing bug-driven test for shape
    /// partitioning, and pinning AllReduce on a non-trivial shape
    /// keeps both invariants in scope when something here regresses.
    #[test]
    fn coloring_allreduce_collapses_to_input_slot() {
        let f = Fuf {
            nodes: vec![
                // gemm output (row-parallel weight, e.g. o_proj or
                // down_proj). Shape [N=4, H=16] stands in for the
                // post-attention or post-MLP residual stream.
                FufNode {
                    id: TileId(0),
                    op: OpKind::Gemm,
                    inputs: vec![],
                    outputs: vec![vec![Dim::Lit(4), Dim::Lit(16)]],
                },
                // AllReduce in-place: output aliases gemm output,
                // same shape. This is the shape the lowering pass
                // (task #5) emits at every ShardDim1 weight at tp>1.
                FufNode {
                    id: TileId(1),
                    op: OpKind::Add, // OpKind::AllReduce will land with task #5
                    inputs: vec![FufInput::Tile {
                        id: TileId(0),
                        slot: 0,
                    }],
                    outputs: vec![vec![Dim::Lit(4), Dim::Lit(16)]],
                },
                // Downstream consumer (e.g. residual-add) reads via
                // the alias.
                FufNode {
                    id: TileId(2),
                    op: OpKind::Add,
                    inputs: vec![FufInput::Tile {
                        id: TileId(1),
                        slot: 0,
                    }],
                    outputs: vec![vec![Dim::Lit(4), Dim::Lit(16)]],
                },
            ],
        };
        let mut lib = ImplementationLibrary::new();
        let id_plain = lib.push(Box::new(StubImpl {
            name: "stub",
            rule: crate::alias_rules::AliasRule::OwnAll,
            consumes: vec![],
        }));
        let id_all_reduce = lib.push(Box::new(StubImpl {
            name: "all_reduce",
            // Same-shape in-place: dst slot 0 aliases gemm output.
            rule: crate::alias_rules::AliasRule::AliasToInput(0),
            consumes: vec![],
        }));
        let sfuf = linear_assignment(
            &[TileId(0), TileId(1), TileId(2)],
            &[id_plain, id_all_reduce, id_plain],
        );
        let lp = linear_loop(3);
        let protected: HashSet<(TileId, u8)> = HashSet::new();
        let sm = colored_slot_map(&f, &sfuf, &lp, &lib, None, &protected);

        assert_eq!(
            sm.of(TileId(0), 0),
            sm.of(TileId(1), 0),
            "AllReduce dst must share slot with input — in-place \
             same-shape, no View row, NCCL all_reduce_inplace \
             operates on the gemm output tile directly",
        );
    }

    /// Different-shape alias keeps its own slot: dst and source have
    /// distinct colors so the runtime can hold a `View { ref_slot:
    /// source_color }` entry at the dst slot. Models `Reshape` —
    /// dst metadata differs (rank or dims) but storage is shared via
    /// indirection. Source's color stays in its own shape pool; dst
    /// gets a fresh color in *its* shape pool. They can never collide.
    #[test]
    fn coloring_different_shape_alias_keeps_own_slot() {
        let f = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Add,
                    inputs: vec![],
                    outputs: vec![vec![Dim::Lit(6)]], // [6]
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Reshape,
                    inputs: vec![FufInput::Tile {
                        id: TileId(0),
                        slot: 0,
                    }],
                    outputs: vec![vec![Dim::Lit(2), Dim::Lit(3)]], // [2,3]
                },
                add_tile(2, &[(TileId(1), 0)]),
            ],
        };
        let mut lib = ImplementationLibrary::new();
        let id_plain = lib.push(Box::new(StubImpl {
            name: "stub",
            rule: crate::alias_rules::AliasRule::OwnAll,
            consumes: vec![],
        }));
        let id_alias = lib.push(Box::new(StubImpl {
            name: "stub_reshape",
            rule: crate::alias_rules::AliasRule::AliasToInput(0),
            consumes: vec![],
        }));
        let sfuf = linear_assignment(
            &[TileId(0), TileId(1), TileId(2)],
            &[id_plain, id_alias, id_plain],
        );
        let lp = linear_loop(3);
        let protected: HashSet<(TileId, u8)> = HashSet::new();
        let sm = colored_slot_map(&f, &sfuf, &lp, &lib, None, &protected);

        assert_ne!(
            sm.of(TileId(0), 0),
            sm.of(TileId(1), 0),
            "different-shape alias dst needs its own slot for the View entry",
        );
    }

    /// Shape-partitioned reuse: a `[1]` tile and an `[N]` tile cannot
    /// share a slot even when their lifetimes are disjoint. This is
    /// the load-bearing invariant for in-place mutations whose
    /// downstream op writes more bytes than the prior occupant's
    /// allocation. Without shape partitioning, a `[41, 8, 128]` K
    /// tile (84 KB) would hand its slot to a `[41, 3072]` residual
    /// tile (252 KB), and the next `cutlass_gemm_add` would write
    /// past the buffer's end — observed as the K=8192 N=5120 cublas
    /// panic on Llama 3.2 3B before this fix.
    #[test]
    fn coloring_disjoint_lifetimes_dont_share_across_shapes() {
        let f = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Add,
                    inputs: vec![],
                    outputs: vec![vec![Dim::Lit(8)]],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Add,
                    inputs: vec![FufInput::Tile {
                        id: TileId(0),
                        slot: 0,
                    }],
                    outputs: vec![vec![Dim::Lit(3072)]],
                },
            ],
        };
        let mut lib = ImplementationLibrary::new();
        let id_plain = lib.push(Box::new(StubImpl {
            name: "stub",
            rule: crate::alias_rules::AliasRule::OwnAll,
            consumes: vec![],
        }));
        let sfuf = linear_assignment(&[TileId(0), TileId(1)], &[id_plain; 2]);
        let lp = linear_loop(2);
        let protected: HashSet<(TileId, u8)> = HashSet::new();
        let sm = colored_slot_map(&f, &sfuf, &lp, &lib, None, &protected);

        assert_ne!(
            sm.of(TileId(0), 0),
            sm.of(TileId(1), 0),
            "tiles of different shape never share a color, even with disjoint lifetimes",
        );
    }

    // ── Loop-detection invariants ───────────────────────────────
    //
    // The loop-detection logic is variant-agnostic; tests use real
    // `Instruction` variants but exercise the algorithm against
    // its data shape (variant tag + per-position field comparison).

    /// Three byte-identical `Free(0)` rows → period 1, count 3.
    #[test]
    fn loop_detection_finds_simplest_run() {
        let v = vec![
            Instruction::Free(0),
            Instruction::Free(0),
            Instruction::Free(0),
        ];
        let map = std::collections::HashMap::new();
        let r = detect_repeating_run(&v, &map);
        assert_eq!(r, Some((0, 1, 3)));
    }

    /// A prefix + a 3×2 repeating block + a suffix: the detector
    /// returns the largest span `(start=2, period=2, iters=3)`.
    /// Pins the "find the largest contiguous repeating run" bit;
    /// the prefix/suffix residues stay where they are.
    #[test]
    fn loop_detection_picks_largest_span_amid_residue() {
        // Use Free(7) for prefix (so it differs from the body's Free),
        // Free(0) + Add(0,0) for the repeating body, Free(9) for suffix.
        let v = vec![
            Instruction::Free(7),
            Instruction::Free(7),
            Instruction::Free(0),
            Instruction::Add(0, 0),
            Instruction::Free(0),
            Instruction::Add(0, 0),
            Instruction::Free(0),
            Instruction::Add(0, 0),
            Instruction::Free(9),
        ];
        let map = std::collections::HashMap::new();
        let r = detect_repeating_run(&v, &map);
        assert_eq!(r, Some((2, 2, 3)));
    }

    /// A run where one variant carries an iter-index field is
    /// detected only when the iter-index field steps by exactly 1
    /// between iterations. The other field values must be byte-
    /// equal across iterations. This pins the `iter_offset` check
    /// in `blocks_match`. Use `RmsNorm(in_slot, out_slot, layer)`
    /// with iter-index field 2 (layer).
    #[test]
    fn loop_detection_handles_iter_index_field() {
        let mut map = std::collections::HashMap::new();
        map.insert("RmsNorm".to_string(), 2usize);
        let v = vec![
            Instruction::RmsNorm(0, 1, 0, 2048, 1),
            Instruction::RmsNorm(0, 1, 1, 2048, 1),
            Instruction::RmsNorm(0, 1, 2, 2048, 1),
        ];
        let r = detect_repeating_run(&v, &map);
        assert_eq!(r, Some((0, 1, 3)));
    }

    /// The iter-index check is strict: stepping by 2 (or any
    /// non-1 offset) does NOT count as a loop. The detector
    /// returns None.
    #[test]
    fn loop_detection_rejects_non_unit_iter_step() {
        let mut map = std::collections::HashMap::new();
        map.insert("RmsNorm".to_string(), 2usize);
        let v = vec![
            Instruction::RmsNorm(0, 1, 0, 2048, 1),
            Instruction::RmsNorm(0, 1, 2, 2048, 1),
            Instruction::RmsNorm(0, 1, 4, 2048, 1),
        ];
        let r = detect_repeating_run(&v, &map);
        assert_eq!(r, None);
    }

    /// A non-iter field varies between rows → not a loop. Locks
    /// the "every non-iter field must be byte-equal" rule.
    /// `RmsNorm(in_slot, out_slot, layer)`: vary `in_slot` (field 0)
    /// while iter-index field is layer (field 2).
    #[test]
    fn loop_detection_rejects_non_iter_field_drift() {
        let mut map = std::collections::HashMap::new();
        map.insert("RmsNorm".to_string(), 2usize);
        let v = vec![
            Instruction::RmsNorm(0, 1, 0, 2048, 1),
            Instruction::RmsNorm(7, 1, 1, 2048, 1),
        ];
        let r = detect_repeating_run(&v, &map);
        assert_eq!(r, None);
    }

    /// `apply_loop_compression`: when a run is detected,
    /// `instances` becomes prefix + Op::Loop + one-iteration body
    /// (with each iter-index field set to that row's iter-0
    /// baseline) + suffix. Locks the rewrite shape end-to-end.
    #[test]
    fn loop_compression_emits_loop_and_keeps_baseline() {
        let mut arch_opcodes = ArchOpcodes::new();
        arch_opcodes.register(OpcodeShape::new(
            "RmsNorm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        ));
        let mut lb = LoweredBucket {
            instances: vec![
                Instruction::RmsNorm(0, 1, 0, 2048, 1),
                Instruction::RmsNorm(0, 1, 1, 2048, 1),
                Instruction::RmsNorm(0, 1, 2, 2048, 1),
            ],
            barriers: vec![false; 3],
            weight_slots: vec![Vec::new(); 3],
            num_slots: 1,
            final_slot: 0,
        };
        apply_loop_compression(&arch_opcodes, &mut lb, "layer");
        assert_eq!(lb.instances.len(), 2, "Loop + 1 body row");
        assert_eq!(instruction_variant_name(&lb.instances[0]), "Loop");
        // Loop fields: count, body_len. body_len = 1.
        assert_eq!(instruction_field_at(&lb.instances[0], 0), Some(3));
        assert_eq!(instruction_field_at(&lb.instances[0], 1), Some(1));
        assert_eq!(instruction_variant_name(&lb.instances[1]), "RmsNorm");
        // Body's `layer` field carries the iter-0 baseline (0).
        assert_eq!(instruction_field_at(&lb.instances[1], 2), Some(0));
    }

    /// Per-row baseline preservation: when the body period has rows
    /// whose iter-0 layer values differ (e.g., row A starts at 0,
    /// row B starts at 1 — Llama's body has `input_layernorm` at
    /// position 6 with baseline = 1 because it logically belongs to
    /// the *next* layer), `apply_loop_compression` must keep each
    /// row's baseline. Zeroing all rows to 0 silently uses layer N's
    /// weights for layer N+1's input ln — accumulating drift that
    /// turns decode into garbage after a few tokens.
    #[test]
    fn loop_compression_preserves_per_row_baseline() {
        let mut arch_opcodes = ArchOpcodes::new();
        arch_opcodes.register(OpcodeShape::new(
            "RmsNorm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        ));
        arch_opcodes.register(OpcodeShape::new(
            "FusedAddRmsNorm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
            ],
        ));
        let mut lb = LoweredBucket {
            instances: vec![
                // iter 0: RmsNorm@0, FusedAddRmsNorm@1
                Instruction::RmsNorm(0, 1, 0, 2048, 1),
                Instruction::FusedAddRmsNorm(0, 1, 1, 2048, 1),
                // iter 1
                Instruction::RmsNorm(0, 1, 1, 2048, 1),
                Instruction::FusedAddRmsNorm(0, 1, 2, 2048, 1),
                // iter 2
                Instruction::RmsNorm(0, 1, 2, 2048, 1),
                Instruction::FusedAddRmsNorm(0, 1, 3, 2048, 1),
            ],
            barriers: vec![false; 6],
            weight_slots: vec![Vec::new(); 6],
            num_slots: 1,
            final_slot: 0,
        };
        apply_loop_compression(&arch_opcodes, &mut lb, "layer");
        assert_eq!(lb.instances.len(), 3, "Loop + 2 body rows");
        assert_eq!(instruction_variant_name(&lb.instances[0]), "Loop");
        assert_eq!(instruction_variant_name(&lb.instances[1]), "RmsNorm");
        assert_eq!(
            instruction_field_at(&lb.instances[1], 2),
            Some(0),
            "row A's baseline is 0",
        );
        assert_eq!(
            instruction_variant_name(&lb.instances[2]),
            "FusedAddRmsNorm"
        );
        assert_eq!(
            instruction_field_at(&lb.instances[2], 2),
            Some(1),
            "row B's baseline is 1 — must be preserved, not zeroed, so the arm \
             can compute `__layer + 1` for the iter-N execution",
        );
    }

    /// No loop in the IR → `apply_loop_compression` is a no-op.
    /// Important property: the pass must not corrupt instances
    /// when no run is present.
    #[test]
    fn loop_compression_is_noop_without_runs() {
        let mut arch_opcodes = ArchOpcodes::new();
        arch_opcodes.register(OpcodeShape::new(
            "Free",
            vec![("slot", syn::parse_quote!(u32))],
        ));
        let original = vec![Instruction::Free(0)];
        let mut lb = LoweredBucket {
            instances: original.clone(),
            barriers: vec![false; original.len()],
            weight_slots: vec![Vec::new(); original.len()],
            num_slots: 1,
            final_slot: 0,
        };
        apply_loop_compression(&arch_opcodes, &mut lb, "layer");
        assert_eq!(lb.instances.len(), 1);
        assert_eq!(instruction_variant_name(&lb.instances[0]), "Free");
    }
}

// The metal/spyre roll proof: the rolled emission must expand to the un-rolled one.
#[cfg(any(feature = "metal", feature = "spyre"))]
/// Expand a loop-compressed instruction stream back to its full
/// form and byte-compare against the pre-compression original —
/// the compression's own correctness oracle, run at expansion
/// (`layer = baseline + iteration`, mirroring the interpreter's
/// `__layer + layer` arm computation). Returns the first mismatch.
pub fn verify_loop_compression(
    original: &[Instruction],
    compressed: &[Instruction],
    arch_opcodes: &ArchOpcodes,
    iter_index_field_name: &str,
) -> Result<(), String> {
    let mut iter_idx: std::collections::HashMap<String, usize> = Default::default();
    for (name, shape) in arch_opcodes.iter() {
        for (i, (fname, _ty)) in shape.fields.iter().enumerate() {
            if fname == iter_index_field_name {
                iter_idx.insert(name.clone(), i);
                break;
            }
        }
    }
    // ⛔ RECURSIVE, BECAUSE LOOPS NEST. gemma-4 rolls an outer six-layer cell (`SSSSSG`) with an
    // inner loop over the five sliding layers, so the layer an inner row lands on is the SUM of
    // every enclosing loop's `iteration * stride`. A flat walk would expand the outer body once
    // and re-prove the inner loop against itself.
    fn expand_into(
        rows: &[Instruction],
        base: u32,
        iter_idx: &std::collections::HashMap<String, usize>,
        out: &mut Vec<Instruction>,
    ) {
        let mut k = 0usize;
        while k < rows.len() {
            let inst = rows[k];
            if let Instruction::Loop(iters, period, layer_stride) = inst {
                let body = &rows[k + 1..k + 1 + period as usize];
                for it in 0..iters {
                    expand_into(body, base + it * layer_stride, iter_idx, out);
                }
                k += 1 + period as usize;
            } else {
                let name = instruction_variant_name(&inst);
                out.push(match iter_idx.get(name) {
                    Some(&fi) => {
                        let at = instruction_field_at(&inst, fi).unwrap_or(0) as u32;
                        instruction_with_field_set(inst, fi, at + base)
                    }
                    None => inst,
                });
                k += 1;
            }
        }
    }
    let mut expanded: Vec<Instruction> = Vec::new();
    expand_into(compressed, 0, &iter_idx, &mut expanded);
    if expanded.len() != original.len() {
        return Err(format!(
            "expanded len {} != original {}",
            expanded.len(),
            original.len()
        ));
    }
    for (i, (a, b)) in original.iter().zip(expanded.iter()).enumerate() {
        if format!("{a:?}") != format!("{b:?}") {
            return Err(format!("row {i}: original={a:?} expanded={b:?}"));
        }
    }
    Ok(())
}

// The metal/spyre roll proof: the rolled emission must expand to the un-rolled one.
#[cfg(any(feature = "metal", feature = "spyre"))]
/// Same oracle for the barrier stream: expand the compressed flags
/// (Loop row = `false`, body slice replicated per iteration) and
/// compare against the pre-compression flags.
pub fn verify_loop_compression_barriers(
    original: &[bool],
    compressed_instances: &[Instruction],
    compressed_barriers: &[bool],
) -> Result<(), String> {
    // ⛔ RECURSIVE, LIKE THE INSTRUCTION ORACLE. A nested body's rows would otherwise be copied
    // verbatim per outer iteration WITHOUT expanding the inner loop, so the flag stream comes out
    // short — measured on gemma-4 as `barrier expanded len 375 != original 662`.
    fn expand_into(rows: &[Instruction], flags: &[bool], base: usize, out: &mut Vec<bool>) {
        let mut k = 0usize;
        while k < rows.len() {
            if let Instruction::Loop(iters, period, _) = rows[k] {
                let body = k + 1..k + 1 + period as usize;
                for _ in 0..iters {
                    expand_into(&rows[body.clone()], flags, base + body.start, out);
                }
                k = body.end;
            } else {
                out.push(flags[base + k]);
                k += 1;
            }
        }
    }
    let mut expanded: Vec<bool> = Vec::new();
    expand_into(compressed_instances, compressed_barriers, 0, &mut expanded);
    if expanded.len() != original.len() {
        return Err(format!(
            "barrier expanded len {} != original {}",
            expanded.len(),
            original.len()
        ));
    }
    for (i, (a, b)) in original.iter().zip(expanded.iter()).enumerate() {
        if a != b {
            return Err(format!("barrier row {i}: original={a} expanded={b}"));
        }
    }
    Ok(())
}
