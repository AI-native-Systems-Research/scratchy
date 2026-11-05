// SPDX-License-Identifier: Apache-2.0
//! The slot and weight VOCABULARY the shared tape path speaks.
//!
//! These types and helpers describe WHERE a weight lives and WHICH slot a
//! value occupies. They are consumed by `to_wavefront` (the shared front
//! end), `metal_from_subtile` (metal's bridge) and codegen.

use std::collections::BTreeMap;

use proc_macro2::TokenStream;

use crate::classified::{UnrollIndex, WeightId};
use crate::fuf::{Fuf, FufInput, FufNode};
use crate::quantization::StorageFormat;
use crate::shape::{Dim, Shape};

/// Evaluate a `Dim` to a concrete integer using `bounds`.
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
/// One field of the emitted `Weights` struct, as derived from the
/// shared tape's weight bindings (`weight_bindings::from_tape`). A
/// fused accessor (e.g. `mlp_gate_up_0` returning one packed
/// `LinearLayer`) lists every DSL weight it stands in for.
#[derive(Clone)]
pub struct WeightAccessor {
    /// Accessor (field) name.
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
