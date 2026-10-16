// SPDX-License-Identifier: Apache-2.0
//! Weight-field naming convention.
//!
//! Single helper: turn a `(WeightId, optional layer index)` pair
//! into the per-arch `Weights` struct field ident the loader and
//! every Impl's `fan_out` use to refer to that weight. Owns the
//! convention `<dotted_path>_<layer>` for layered accessors and
//! `<dotted_path>` for arch-wide ones — `split_base_layer` in
//! `codegen.rs` is the inverse.

use quote::format_ident;

use crate::classified::{Program, UnrollIndex, WeightId};

/// The field name for a weight instance. `self_attn.q_proj` at
/// layer 3 → `self_attn_q_proj_3`; unindexed `embed_tokens` →
/// `embed_tokens`. Shared between codegen (which emits the
/// `Weights` struct) and Impls' `fan_out` (which reference the
/// fields the codegen emitted).
pub fn weight_field_name(
    program: &Program,
    id: WeightId,
    index: Option<UnrollIndex>,
) -> syn::Ident {
    format_ident!("{}", weight_field_stem(&program.weights, id, index))
}

/// [`weight_field_name`]'s spelling as a plain `String` — for code that must not hold a token (the
/// spyre bake's parallel rungs: a `proc_macro2::Ident` cannot cross threads).
pub fn weight_field_stem(
    weights: &crate::classified::WeightTable,
    id: WeightId,
    index: Option<UnrollIndex>,
) -> String {
    let stem = weights.path(id).join("_");
    match index {
        Some(i) => format!("{stem}_{}", i.0),
        None => stem,
    }
}
