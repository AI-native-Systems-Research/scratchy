// SPDX-License-Identifier: Apache-2.0
//! Instruction-selection metadata as declared data: the output-alias
//! and KV-layer-I/O rules every `Implementation` used to encode as an
//! overridden method body. An impl now DECLARES which rule applies
//! (`Implementation::alias_rule` / `Implementation::kv_rule`, both
//! one-line data declarations); the semantics of each rule live here,
//! once, and the consumption sites (`colored_slot_map`, `lower_bucket`
//! aliasing, the hazard walk) funnel through [`output_alias`] /
//! [`kv_layer_io`]. There is no per-impl override surface left — the
//! trait has no aliasing/KV *method* to override, only the rule
//! declaration.
//!
//! Alias vs consume (the distinction the old method docs carried): an
//! alias entry `(dst, Some(src))` says dst's output is a `TensorView`
//! borrowing src's storage — the drop pass must keep src's ultimate
//! owner alive until every consumer of dst is done, and
//! `colored_slot_map` collapses the two onto one color. `(dst, None)`
//! says dst owns a fresh buffer. An output absent from the list is
//! UNTRACKED (paged-cache views owned by the kv_cache pool, or
//! threadgroup-memory intermediates that never escape to a runtime
//! slot — declaring those would have the slot allocator size arena
//! buffers for ghost slots no kernel binds).

use crate::classified::OpKind;
use crate::fuf::{Fuf, FufInput, TileId};
use crate::weight_vocab::{first_tile_input, is_rope_append_op, kv_cache_extern_layer};
use crate::weight_vocab::{scalar_mul_is_unity_passthrough, scalar_mul_tile_input};

/// One entry per tracked output: `((tile, slot), alias_src)` — see
/// the module docs for alias vs own vs untracked.
pub type OutputAliases = Vec<((TileId, u8), Option<(TileId, u8)>)>;

/// Selector for [`AliasRule::SoleOutput`]: which claimed tile carries
/// the subgraph's single externally-visible output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpSel {
    /// Either rope-append flavor (plain or interleaved).
    RopeAny,
    ScalarWeightMul,
    Mul,
}

impl OpSel {
    fn matches(self, op: OpKind) -> bool {
        match self {
            OpSel::RopeAny => is_rope_append_op(op),
            OpSel::ScalarWeightMul => op == OpKind::ScalarWeightMul,
            OpSel::Mul => op == OpKind::Mul,
        }
    }
}

/// The output-alias rule an `Implementation` declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AliasRule {
    /// Default: every claimed tile's outputs are fresh owned buffers.
    OwnAll,
    /// The single claimed tile's output 0 aliases its tile input at
    /// the given position (in-place kernels: Add mutates the residual
    /// at input 1; Reshape/AllReduce/MmEmbedSplice mutate or view
    /// input 0).
    AliasToInput(usize),
    /// ScalarMul: `x * 1.0` is a structural identity — output aliases
    /// the source tile (combined with empty fan_out / consumes this
    /// removes the kernel call entirely). Any other scale owns its
    /// output (the in-place path moves the upstream's storage).
    ScalarMulUnityOrOwn,
    /// RopeAppend (unfused): all three outputs (q', k', v') are 3D
    /// `TensorView` reshapes over the upstream tile storage.
    RopeTripleAlias,
    /// RopeAppendNormed: rope outputs alias the RAW projection
    /// buffers upstream of the folded per-input norms (q is normed +
    /// rotated in place; k'/v' tiles are dead — attention reads the
    /// cache — but the alias keeps lifetimes conservative). Each rope
    /// input peels an optional flatten-back Reshape, then the norm,
    /// to reach the projection tile.
    RopeTripleAliasPeeled,
    /// Fused Add+RmsNorm: the kernel mutates the delta buffer in
    /// place to produce the rmsnorm output and the residual buffer to
    /// produce the updated residual — `(rmsnorm, 0)` aliases the
    /// Add's input 0, `(add, 0)` aliases the Add's input 1.
    NormDeltaResidual,
    /// Cutlass fused AddRmsNorm+Gemm: only the Add output aliases the
    /// residual (input 1); the Gemm output is a fresh buffer.
    AddAliasResidualIn1,
    /// Gemm+Add fusions: the Add's output aliases its non-Gemm
    /// (residual) tile input — `add_inplace` mutates it directly.
    GemmAddResidual,
    /// Exactly one externally-visible output: `(sel-tile, 0)`, owned.
    /// Everything else in the claim is untracked (cache views /
    /// threadgroup intermediates).
    SoleOutput(OpSel),
}

/// The complete alias-rule vocabulary — the const table of every
/// rule any impl may declare, independent of which backend feature
/// is compiled in (the metal-only rules stay in the table under a
/// pure-cuda build: the vocabulary is target-neutral data, only the
/// declaring impls are feature-gated).
pub const ALL_ALIAS_RULES: &[AliasRule] = &[
    AliasRule::OwnAll,
    AliasRule::AliasToInput(0),
    AliasRule::AliasToInput(1),
    AliasRule::ScalarMulUnityOrOwn,
    AliasRule::RopeTripleAlias,
    AliasRule::RopeTripleAliasPeeled,
    AliasRule::NormDeltaResidual,
    AliasRule::AddAliasResidualIn1,
    AliasRule::GemmAddResidual,
    AliasRule::SoleOutput(OpSel::RopeAny),
    AliasRule::SoleOutput(OpSel::ScalarWeightMul),
    AliasRule::SoleOutput(OpSel::Mul),
];

/// The KV-cache hazard I/O rule an `Implementation` declares.
/// `Writes`/`Reads` resolve the per-layer index off the claim's
/// `ExternKind::KvCache` input. Without the write declaration on the
/// rope/pre-attn impls the hazard analyzer never fences the
/// KV-writing dispatch against the attention that reads the same
/// layer — Metal then runs them concurrently and produces garbage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvRule {
    NoIo,
    Writes,
    Reads,
}

/// The complete KV-rule vocabulary (see [`ALL_ALIAS_RULES`]).
pub const ALL_KV_RULES: &[KvRule] = &[KvRule::NoIo, KvRule::Writes, KvRule::Reads];

/// Peel an optional single-consumer flatten-back `Reshape` between a
/// rope input and its norm (Gemma4 q/k norms run per-head on a
/// `[m, heads, head_dim]` view, then a Reshape flattens back for the
/// rope tile; the global-class k path has NO such reshape). Returns
/// `(norm_tile, reshape_tile_if_any)`.
pub(crate) fn peel_reshape(fuf: &Fuf, t: TileId) -> (TileId, Option<TileId>) {
    let node = fuf.get(t);
    if node.op == OpKind::Reshape
        && let Some(inner) = node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } => Some(*id),
            _ => None,
        })
    {
        return (inner, Some(t));
    }
    (t, None)
}

fn tile_input_at(fuf: &Fuf, t: TileId, idx: usize) -> Option<(TileId, u8)> {
    match fuf.get(t).inputs.get(idx) {
        Some(FufInput::Tile { id, slot }) => Some((*id, *slot)),
        _ => None,
    }
}

fn find_op(claimed: &[TileId], fuf: &Fuf, op: OpKind) -> TileId {
    *claimed
        .iter()
        .find(|t| fuf.get(**t).op == op)
        .unwrap_or_else(|| panic!("alias rule: claim contains {op:?}"))
}

fn own_all(claimed: &[TileId], fuf: &Fuf) -> OutputAliases {
    claimed
        .iter()
        .flat_map(|&t| {
            let n = fuf.get(t).outputs.len().max(1);
            (0..n as u8).map(move |s| ((t, s), None))
        })
        .collect()
}

/// Interpret `rule` over a claim — THE `output_alias` of every impl.
pub fn output_alias(rule: AliasRule, claimed: &[TileId], fuf: &Fuf) -> OutputAliases {
    match rule {
        AliasRule::OwnAll => own_all(claimed, fuf),
        AliasRule::AliasToInput(idx) => {
            let t = claimed[0];
            vec![((t, 0), tile_input_at(fuf, t, idx))]
        }
        AliasRule::ScalarMulUnityOrOwn => {
            if scalar_mul_is_unity_passthrough(claimed, fuf) {
                let src = scalar_mul_tile_input(claimed, fuf);
                vec![((claimed[0], 0), Some(src))]
            } else {
                own_all(claimed, fuf)
            }
        }
        AliasRule::RopeTripleAlias => {
            let t = claimed[0];
            vec![
                ((t, 0), tile_input_at(fuf, t, 0)),
                ((t, 1), tile_input_at(fuf, t, 1)),
                ((t, 2), tile_input_at(fuf, t, 2)),
            ]
        }
        AliasRule::RopeTripleAliasPeeled => {
            let rope_id = *claimed
                .iter()
                .find(|t| matches!(fuf.get(**t).op, OpKind::RopeAppend))
                .expect("RopeAppendNormed claim contains RopeAppend");
            let upstream = |idx: usize| -> Option<(TileId, u8)> {
                match fuf.get(rope_id).inputs.get(idx) {
                    Some(FufInput::Tile { id, .. }) => {
                        let (norm_t, _) = peel_reshape(fuf, *id);
                        first_tile_input(fuf.get(norm_t))
                    }
                    _ => None,
                }
            };
            vec![
                ((rope_id, 0), upstream(0)),
                ((rope_id, 1), upstream(1)),
                ((rope_id, 2), upstream(2)),
            ]
        }
        AliasRule::NormDeltaResidual => {
            // The residual-stream Add is the one whose inputs are all
            // tiles (a WithOffset claim also carries scalar gain
            // Adds; a plain claim has exactly one Add and it
            // qualifies).
            let add_id = *claimed
                .iter()
                .find(|t| {
                    let n = fuf.get(**t);
                    n.op == OpKind::Add
                        && n.inputs.iter().all(|i| matches!(i, FufInput::Tile { .. }))
                })
                .expect("fused add+rmsnorm claim contains a residual-stream Add");
            let rmsnorm_id = find_op(claimed, fuf, OpKind::RmsNorm);
            let delta_src =
                tile_input_at(fuf, add_id, 0).expect("residual Add input 0 (delta) must be a Tile");
            let residual_src = tile_input_at(fuf, add_id, 1)
                .expect("residual Add input 1 (residual) must be a Tile");
            vec![
                ((rmsnorm_id, 0), Some(delta_src)),
                ((add_id, 0), Some(residual_src)),
            ]
        }
        AliasRule::AddAliasResidualIn1 => {
            let add_id = find_op(claimed, fuf, OpKind::Add);
            let residual_src = tile_input_at(fuf, add_id, 1)
                .expect("fused norm+gemm: Add input 1 (residual) must be a Tile");
            vec![((add_id, 0), Some(residual_src))]
        }
        AliasRule::GemmAddResidual => {
            let add_id = find_op(claimed, fuf, OpKind::Add);
            let gemm_id = find_op(claimed, fuf, OpKind::Gemm);
            let residual_src = fuf.get(add_id).inputs.iter().find_map(|i| match i {
                FufInput::Tile { id, slot } if *id != gemm_id => Some((*id, *slot)),
                _ => None,
            });
            vec![((add_id, 0), residual_src)]
        }
        AliasRule::SoleOutput(sel) => {
            let t = *claimed
                .iter()
                .find(|t| sel.matches(fuf.get(**t).op))
                .unwrap_or_else(|| panic!("alias rule: claim contains {sel:?}"));
            vec![((t, 0), None)]
        }
    }
}

/// Interpret `rule` over a claim — THE `kv_layer_io` of every impl.
pub fn kv_layer_io(rule: KvRule, claimed: &[TileId], fuf: &Fuf) -> (Option<u32>, Option<u32>) {
    match rule {
        KvRule::NoIo => (None, None),
        KvRule::Writes => (kv_cache_extern_layer(claimed, fuf), None),
        KvRule::Reads => (None, kv_cache_extern_layer(claimed, fuf)),
    }
}
