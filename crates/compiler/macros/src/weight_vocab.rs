// SPDX-License-Identifier: Apache-2.0
//! The slot and weight VOCABULARY the shared tape path speaks.
//!
//! These types and helpers describe WHERE a weight lives and WHICH slot a
//! value occupies. They are consumed by `to_wavefront` (the shared front
//! end) and `metal_from_subtile` (metal's bridge) — the code that REPLACED
//! instruction selection — as well as by codegen and the selection library
//! itself.
//!
//! They lived in `impl_lib` for historical reasons: that module grew up as
//! the instruction-selection library and accumulated this vocabulary along
//! the way. Keeping them there means a build that performs no instruction
//! selection still has to compile all of it, which is why `impl_lib` cannot
//! simply be feature-gated off the metal path, even though instruction
//! selection itself never runs under metal or spyre.

use std::collections::BTreeMap;

use proc_macro2::TokenStream;

use crate::classified::{ExternKind, OpKind, UnrollIndex, WeightId};
use crate::fuf::{Fuf, FufInput, FufNode, TileId};
use crate::quantization::StorageFormat;
use crate::shape::{Dim, Shape};

/// Evaluate a `Dim` to a concrete integer using `bounds`. Free
/// function so non-`CostCtx` callers (e.g. `fan_out`, which only has
/// `fuf + bounds`) can reuse it without round-tripping through a
/// CostCtx instance.
pub fn eval_dim_with(dim: &Dim, bounds: &BTreeMap<String, u64>) -> Option<u64> {
    match dim {
        Dim::Lit(n) => Some(*n),
        Dim::Bound(name) => bounds.get(name).copied(),
        Dim::Mul(cs) => cs
            .iter()
            .map(|c| eval_dim_with(c, bounds))
            .try_fold(1u64, |acc, v| v.map(|x| acc.saturating_mul(x))),
        Dim::Div(num, den) => {
            let n = eval_dim_with(num, bounds)?;
            let d = eval_dim_with(den, bounds)?;
            n.checked_div(d)
        }
        Dim::Var(_) => None,
    }
}

pub fn eval_shape_with(shape: &Shape, bounds: &BTreeMap<String, u64>) -> Option<Vec<u64>> {
    shape.iter().map(|d| eval_dim_with(d, bounds)).collect()
}
/// A method the emitted `WeightBundle` trait must expose, as
/// declared by an [`Implementation`]. Codegen aggregates these
/// declarations across every subgraph of an SFUF and emits one
/// trait method per unique accessor.
///
/// Why this lives on the Impl: a fusion impl that claims multiple
/// Gemm tiles may want the user to pre-concatenate the weights
/// into one packed buffer, so its `emit_call` can issue a single
/// cuBLAS call. Declaring a fused accessor (e.g. `mlp_gate_up_0`
/// returning one `LinearLayer`) is how the impl expresses that
/// contract to the user, without the compiler knowing anything
/// about "gate" or "up" specifically.
#[derive(Clone)]
pub struct WeightAccessor {
    /// Trait method name. Two impls that declare the same name
    /// must agree on `rust_type`; a mismatch is a hard error at
    /// trait-emission time.
    pub name: syn::Ident,
    /// Return type, as emitted Rust tokens (e.g.
    /// `crate::__gpu::layers::LinearLayer`).
    pub rust_type: TokenStream,
    /// DSL weights that feed this accessor. One pair for a simple
    /// accessor, multiple for a fused one. Used for dedup and
    /// documentation; the user is responsible for providing a
    /// weight object with the declared `rust_type` that stands in
    /// for the listed sources.
    pub source_weights: Vec<(WeightId, Option<UnrollIndex>)>,
}
/// Walk `claimed` tiles' inputs for the first
/// `FufInput::Extern { kind: ExternKind::KvCache, index: Some(layer) }`
/// and return the layer index as a u32. KV-touching Impls
/// (`RopeAppend`, fused QKV+cache, `SynthPreAttn`,
/// `AttentionViaCache`, paged prefill attention) declare a
/// [`crate::alias_rules::KvRule`] via [`Implementation::kv_rule`] —
/// the same extern is already on the tile's input list, the Impl
/// just declares which direction (write vs read) the layer flows.
pub fn kv_cache_extern_layer(claimed_tiles: &[TileId], fuf: &Fuf) -> Option<u32> {
    for &t in claimed_tiles {
        for input in &fuf.get(t).inputs {
            if let FufInput::Extern {
                kind: ExternKind::KvCache,
                index: Some(layer),
            } = input
            {
                return Some(layer.0 as u32);
            }
        }
    }
    None
}
/// Variant declaration an Impl contributes to its arch's
/// macro-emitted opcode enum.
///
/// The codegen, processing one arch's solved FUF, collects
/// `OpcodeShape`s from the Impls the solver picked for that arch.
/// From the union of shapes it emits one Rust enum per arch:
///
/// ```ignore
/// enum LlamaOp {
///     AttnNorm { layer: u32, in_slot: u32, out_slot: u32 },
///     QkvRopeAppend { layer: u32, in_slot: u32, out_q_slot: u32 },
///     // … only the variants Llama's picked Impls declared
///     Free { slot: u32 },  // injected by codegen, not by any Impl
/// }
/// ```
///
/// `Free` is added unconditionally by codegen for the drop pass.
/// Every Impl-driven variant is named here, by exactly one Impl.
/// Two Impls declaring the same variant ident must agree on field
/// shape — codegen panics on mismatch.
#[derive(Clone, Debug)]
pub struct OpcodeShape {
    /// PascalCase ident the per-arch enum uses for this variant.
    pub name: syn::Ident,
    /// Ordered field declarations. The codegen renders them as
    /// `name: type` inside the variant's struct-style payload.
    pub fields: Vec<(syn::Ident, syn::Type)>,
}
impl OpcodeShape {
    /// Build from a variant name and a list of (field_ident, type)
    /// pairs. Used inside Impls' `opcode_shape()` overrides.
    pub fn new(name: &str, fields: Vec<(&str, syn::Type)>) -> Self {
        let name_ident = syn::Ident::new(name, proc_macro2::Span::call_site());
        let fields = fields
            .into_iter()
            .map(|(f, ty)| {
                let f_ident = syn::Ident::new(f, proc_macro2::Span::call_site());
                (f_ident, ty)
            })
            .collect();
        Self {
            name: name_ident,
            fields,
        }
    }

    /// Sentinel shape returned by the trait default. The codegen
    /// checks for this and panics with the Impl's `name()` so a
    /// newly-added Impl can't silently bypass migration.
    pub(crate) fn unmigrated(impl_name: &str) -> Self {
        // The variant ident here is never actually used — codegen
        // detects the unmigrated state via `fan_out → None` long
        // before it would consume the shape. Pick a placeholder that
        // wouldn't collide with a real variant name by accident.
        let _ = impl_name;
        Self {
            name: syn::Ident::new("__Unmigrated", proc_macro2::Span::call_site()),
            fields: Vec::new(),
        }
    }
}
pub use scratchy_subtile::handoff::WeightKind;

pub use scratchy_subtile::handoff::WeightSlot;

pub use scratchy_subtile::handoff::SlotMap;

/// Softmax scale for one attention call. Priority order:
///   1. Granite's `attention_multiplier` — a direct override (no
///      transform); the HF config already stores the final scale.
///   2. Gemma2's `query_pre_attn_scalar` — convention is
///      `scale = query_pre_attn_scalar.powf(-0.5)`.
///   3. Fallback `1 / sqrt(head_dim)` for Llama / Qwen2 / Qwen3.
pub(crate) fn attention_scale_for(model: &crate::config::ModelParams) -> f32 {
    if let Some(s) = model.scalars.get("attention_multiplier") {
        return *s as f32;
    }
    match model.scalars.get("query_pre_attn_scalar") {
        Some(q) => (*q as f32).powf(-0.5),
        None => {
            let head_dim_f = *model
                .bounds
                .get("head_dim")
                .expect("attention emit: head_dim missing from model config")
                as f32;
            1.0 / head_dim_f.sqrt()
        }
    }
}
/// The [`StorageFormat`] of the first weight input of `node`, or
/// `None` if the node has no weight inputs. Every Gemm tile in the
/// FUF carries exactly one `FufInput::Weight`; non-Gemm tiles return
/// `None`. Dense and quant-aware impls alike read this to decide
/// whether the kernel they emit (cuBLAS / cutlass vs. Marlin) can
/// legally consume the weight's storage.
pub(crate) fn weight_storage_of(node: &FufNode) -> Option<&StorageFormat> {
    node.inputs.iter().find_map(|i| match i {
        FufInput::Weight { storage, .. } => Some(storage),
        _ => None,
    })
}
/// Return the `(TileId, slot)` of a node's first `FufInput::Tile`
/// input. For Gemm this identifies the activation (the weight input
/// is a `FufInput::Weight`).
pub(crate) fn first_tile_input(node: &crate::fuf::FufNode) -> Option<(TileId, u8)> {
    node.inputs.iter().find_map(|i| match i {
        FufInput::Tile { id, slot } => Some((*id, *slot)),
        _ => None,
    })
}
/// True for either rope-append flavor — the NeoX-style `RopeAppend`
/// or the Cohere-style `RopeAppendInterleaved`. Used by the QKV+rope
/// fusion matchers and the singleton fallback so a single impl claims
/// both flavors and dispatches to the right kernel at emit time.
pub(crate) fn is_rope_append_op(op: OpKind) -> bool {
    matches!(op, OpKind::RopeAppend | OpKind::RopeAppendInterleaved)
}
/// Static (N, K) of a Gemm tile from its FUF node + variable bounds.
/// `N` is the tile's output last dim (out_features); `K` is the
/// input's last dim (in_features). Available to `fan_out` impls so
/// they can bake weight shape into the emitted `Instruction` for the
/// runtime shape-assert that guards against loader/codegen drift.
pub(crate) fn gemm_nk_from_fuf(
    fuf: &Fuf,
    node: &crate::fuf::FufNode,
    bounds: &BTreeMap<String, u64>,
) -> Option<(u32, u32)> {
    let out_shape = node
        .outputs
        .first()
        .and_then(|s| eval_shape_with(s, bounds))?;
    if out_shape.len() != 2 {
        return None;
    }
    let n = *out_shape.last()? as u32;
    let k = node.inputs.iter().find_map(|inp| match inp {
        FufInput::Tile { id, slot } => {
            let up = fuf.get(*id);
            up.outputs
                .get(*slot as usize)
                .and_then(|s| eval_shape_with(s, bounds))
                .and_then(|v| v.last().copied())
        }
        _ => None,
    })? as u32;
    Some((n, k))
}

// ── ScalarMul tile helpers ───────────────────────────────────────
// Vocabulary, not selection: `alias_rules` calls these on the tape path to
// decide whether a scalar multiply is an identity pass-through. They lived on
// `ScalarMulImpl` only because that impl was their first caller.
/// Extract the FUF's scalar literal from a single-tile claim.
/// Returns `f32::NAN` for malformed claims (matcher invariants
/// guarantee one Scalar input, so this only fires if the matcher
/// is bypassed).
pub(crate) fn scalar_mul_scale_of(claimed_tiles: &[TileId], fuf: &Fuf) -> f32 {
    let tile = match claimed_tiles.first() {
        Some(&t) => t,
        None => return f32::NAN,
    };
    for inp in &fuf.get(tile).inputs {
        if let FufInput::Scalar(v) = inp {
            return *v as f32;
        }
    }
    f32::NAN
}
/// Whether the matched `x * scale` is mathematically `x` and
/// therefore reducible to an alias of the source tile.
pub(crate) fn scalar_mul_is_unity_passthrough(claimed_tiles: &[TileId], fuf: &Fuf) -> bool {
    scalar_mul_scale_of(claimed_tiles, fuf) == 1.0
}
/// `(src_tile, src_slot)` of the Tile input. Helpers shared by
/// `output_alias` / `consumes_input_tiles` / `fan_out`.
pub(crate) fn scalar_mul_tile_input(claimed_tiles: &[TileId], fuf: &Fuf) -> (TileId, u8) {
    let tile = claimed_tiles[0];
    let node = fuf.get(tile);
    node.inputs
        .iter()
        .find_map(|i| match i {
            FufInput::Tile { id, slot } => Some((*id, *slot)),
            _ => None,
        })
        .expect("ScalarMul has a Tile input")
}
