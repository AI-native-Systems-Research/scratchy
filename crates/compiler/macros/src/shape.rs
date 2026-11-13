// SPDX-License-Identifier: Apache-2.0
//! The shapes every later pass reads: each local's, weight's and nested value's, as the carrier's
//! typed program states them ([`Inferred`], the trace of `typed_shapes`), and what the passes
//! compute from them.
//!
//! A [`Dim`] is symbolic over a compact vocabulary: `Dim::Lit(n)`, a concrete integer;
//! `Dim::Bound(name)`, a named config bound like `"hidden_size"`; `Dim::Mul`/`Dim::Add`/`Dim::Div`
//! over those. rustc's type checker holds every shape law when the macro crate compiles (the
//! signatures and carriers of `typed_shapes`); nothing here checks one.

use crate::classified::{Expr, ExternKind, LocalId, OpKind, Program, Stmt};
pub use crate::dim::{Dim, Shape};
pub use crate::typed_shapes::Shapes as Inferred;

// ── Extern param shapes ──────────────────────────────────────────

pub fn extern_shape(kind: ExternKind) -> Shape {
    match kind {
        ExternKind::InputIds => vec![Dim::Bound("num_tokens".into())],
        ExternKind::Positions => vec![Dim::Bound("num_tokens".into())],
        // The following externs have opaque shapes — shape
        // inference doesn't need to model them to derive weight
        // shapes for the Llama/Qwen2 body. The solver/codegen
        // consumes them via their ExternKind tag rather than via
        // shape.
        ExternKind::Rotary => vec![],
        ExternKind::RotaryLocal => vec![],
        ExternKind::BlockTable => vec![],
        ExternKind::KvCache => vec![],
        // Vision externs. `Pixels` has the per-row patch shape the
        // encoder's first GEMM reads (`patch_embed_proj`); `Cos`/`Sin`
        // carry the per-row half-
        // dim RoPE tables. The remaining vision externs are opaque
        // (varlen index / per-image grid / scalar) — same role as
        // the decoder's `BlockTable`/`KvCache`. Bound names anchor
        // to vision-config fields populated by per-arch crates in
        // G.5; until then they're symbolic placeholders that resolve
        // only when a real vision config is loaded.
        ExternKind::Pixels => vec![
            Dim::Bound("num_tokens".into()),
            Dim::Bound("vision_in_features".into()),
        ],
        ExternKind::Cos => vec![
            Dim::Bound("num_tokens".into()),
            Dim::Bound("vision_rope_half_dim".into()),
        ],
        ExternKind::Sin => vec![
            Dim::Bound("num_tokens".into()),
            Dim::Bound("vision_rope_half_dim".into()),
        ],
        ExternKind::CuSeqlens => vec![],
        ExternKind::GridThw => vec![],
        ExternKind::MaxSeqlen => vec![],
        // Qwen2.5-VL split externs — all opaque (rank-1 cu_seqlens, scalar
        // max_seqlens, rank-1 permutations). Same role as their non-split
        // siblings.
        ExternKind::CuSeqlensFull => vec![],
        ExternKind::CuSeqlensWindow => vec![],
        ExternKind::MaxSeqlenFull => vec![],
        ExternKind::MaxSeqlenWindow => vec![],
        ExternKind::WindowIndex => vec![],
        ExternKind::ReverseIndices => vec![],
        // SigLIP-style positional-embedding indices, shape
        // `[num_tokens]`. Built host-side as `[0..N, 0..N, ...]`
        // (N = `vision_num_positions`) per image; consumed by
        // `OpKind::PosEmbed` via `kernels::embedding_gather_masked`.
        ExternKind::PositionIds => vec![Dim::Bound("num_tokens".into())],
        // Qwen3.5-VL host-interpolated positional embedding, added to
        // the patch-embed output, so its inner dim is the residual
        // width `vision_embed_dim` (= vision_num_heads * vision_head_dim).
        ExternKind::PosEmbeds => vec![
            Dim::Bound("num_tokens".into()),
            Dim::Bound("vision_embed_dim".into()),
        ],
        // A target model's final hidden states: one row per token of the target's residual width.
        ExternKind::TargetHidden => vec![
            Dim::Bound("num_tokens".into()),
            Dim::Bound("backbone_hidden_size".into()),
        ],
    }
}

/// `d`'s value under `bounds`; `None` if it names a bound `bounds` lacks.
pub fn eval_closed_dim(d: &Dim, bounds: &std::collections::BTreeMap<String, u64>) -> Option<u64> {
    match d {
        Dim::Lit(n) => Some(*n),
        Dim::Bound(name) => bounds.get(name).copied(),
        Dim::Mul(factors) => factors
            .iter()
            .try_fold(1u64, |acc, f| eval_closed_dim(f, bounds).map(|v| acc * v)),
        Dim::Add(terms) => terms
            .iter()
            .try_fold(0u64, |acc, t| eval_closed_dim(t, bounds).map(|v| acc + v)),
        Dim::Div(num, den) | Dim::FloorDiv(num, den) => {
            let n = eval_closed_dim(num, bounds)?;
            let d = eval_closed_dim(den, bounds)?;
            // Caller's responsibility to keep the dividend exactly
            // divisible — the merger reshape's `total_l /
            // spatial_merge_size**2` always is. Truncating divide is
            // fine for the bounds-equivalence check (mismatched
            // remainders are caught by the unify path, not here).
            n.checked_div(d)
        }
    }
}

/// Erase every regrouping that is the identity for one model: a `reshape` of a local whose source
/// and target have the same per-axis extents under `bounds` (a split or merge by a factor of 1, as
/// a single global KV head makes) becomes an alias of its source, which binds the source's tile
/// and pushes no node. Keyed only on the evaluated extents. The typed program keeps the
/// regrouping, so rustc still checks it whatever the factor; this is the per-model view of it.
pub fn erase_identity_regroupings(
    program: &mut Program,
    inferred: &Inferred,
    bounds: &std::collections::BTreeMap<String, u64>,
) {
    // The row count is the forward's, not the config's: any one nonzero count shows whether an
    // axis's extent is the other's, since both sides scale by the same rows.
    let mut bounds = bounds.clone();
    bounds.insert("num_tokens".to_string(), 7919);
    let extents = |id: &LocalId| -> Option<Vec<u64>> {
        (inferred.locals.get(id)?.iter())
            .map(|d| eval_closed_dim(d, &bounds))
            .collect()
    };
    fn walk(stmts: &mut [Stmt], identity: &dyn Fn(LocalId, LocalId) -> bool) {
        for s in stmts {
            match s {
                Stmt::Assign { target, value } => {
                    if let Expr::Call {
                        op: OpKind::Reshape,
                        args,
                    } = value
                        && let [Expr::Local(source)] = args.as_slice()
                        && identity(*source, *target)
                    {
                        *value = Expr::Local(*source);
                    }
                }
                Stmt::AssignTuple { .. } => {}
                Stmt::For { body, .. } => walk(body, identity),
                Stmt::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    walk(then_body, identity);
                    walk(else_body, identity);
                }
            }
        }
    }
    let identity = |source: LocalId, target: LocalId| matches!((extents(&source), extents(&target)), (Some(a), Some(b)) if a == b);
    walk(&mut program.statements, &identity);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dim::canonical_mul;

    fn bound(name: &str) -> Dim {
        Dim::Bound(name.into())
    }

    fn heads_layout_attn() -> Dim {
        canonical_mul(vec![bound("head_dim"), bound("num_attention_heads")])
    }

    #[test]
    fn canonical_mul_sorts_and_flattens() {
        let d = canonical_mul(vec![
            bound("head_dim"),
            Dim::Mul(vec![bound("num_attention_heads"), Dim::Lit(1)]),
        ]);
        assert_eq!(d, heads_layout_attn());
    }

    #[test]
    fn vision_op_names_round_trip_through_as_str() {
        // OpKind::as_str must be total over every variant; these
        // are the new names landed in G.2.
        assert_eq!(OpKind::VarlenAttention.as_str(), "varlen_attention");
        assert_eq!(OpKind::VisionRope.as_str(), "vision_rope");
        assert_eq!(OpKind::QuickGelu.as_str(), "quick_gelu");
        assert_eq!(OpKind::GeluErf.as_str(), "gelu_erf");
        // G.6.4: row-permutation gather added for Qwen2.5-VL window
        // attention dispatch. Round-trip required so DSL bodies that
        // write `embedding_gather(...)` parse cleanly.
        assert_eq!(OpKind::EmbeddingGather.as_str(), "embedding_gather");
        assert_eq!(
            OpKind::from_name("embedding_gather"),
            Some(OpKind::EmbeddingGather)
        );
    }

    #[test]
    fn avg_pool_2d_round_trips_through_op_name() {
        assert_eq!(OpKind::AvgPool2d.as_str(), "avg_pool_2d");
        assert_eq!(OpKind::from_name("avg_pool_2d"), Some(OpKind::AvgPool2d));
    }

    #[test]
    fn quick_gelu_and_gelu_erf_are_distinct_from_tanh_gelu() {
        // Numerics-bearing distinction: the three GELU OpKinds must
        // be PartialEq-distinct so consumer Impls can match exactly
        // one of them. (Eq is derived; this test is a structural
        // promise that future refactors can't accidentally collapse
        // them onto a single variant.)
        assert_ne!(OpKind::Gelu, OpKind::QuickGelu);
        assert_ne!(OpKind::Gelu, OpKind::GeluErf);
        assert_ne!(OpKind::QuickGelu, OpKind::GeluErf);
    }

    #[test]
    fn vision_op_names_resolve_via_from_name() {
        // G.4 gate: every name `OpKind::as_str` returns must round-
        // trip through `from_name`, otherwise classify rejects DSL
        // bodies that wrote the op (parse-then-reject violation).
        // Locks in the four arms added in G.4.
        assert_eq!(
            OpKind::from_name("varlen_attention"),
            Some(OpKind::VarlenAttention)
        );
        assert_eq!(OpKind::from_name("vision_rope"), Some(OpKind::VisionRope));
        assert_eq!(OpKind::from_name("quick_gelu"), Some(OpKind::QuickGelu));
        assert_eq!(OpKind::from_name("gelu_erf"), Some(OpKind::GeluErf));
        assert_eq!(OpKind::from_name("relu2"), Some(OpKind::Relu2));
        assert_eq!(OpKind::Relu2.as_str(), "relu2");
    }

    #[test]
    fn qwen2_5_vl_split_externs_classify_under_vision_prelude() {
        // G.6.3: the four cu_seqlens/max_seqlen split externs and the
        // window_index / reverse_indices permutation externs must
        // classify under the vision prelude. Decoder prelude must
        // continue to NOT see them — pin both directions so a future
        // edit that leaks them into the decoder set trips here.
        use crate::classified::{ExternKind, Prelude};
        let names = [
            ("cu_seqlens_full", ExternKind::CuSeqlensFull),
            ("cu_seqlens_window", ExternKind::CuSeqlensWindow),
            ("max_seqlen_full", ExternKind::MaxSeqlenFull),
            ("max_seqlen_window", ExternKind::MaxSeqlenWindow),
            ("window_index", ExternKind::WindowIndex),
            ("reverse_indices", ExternKind::ReverseIndices),
        ];
        for (name, kind) in names {
            assert_eq!(
                ExternKind::from_name_for(name, Prelude::Vision),
                Some(kind),
                "vision prelude must resolve {name}",
            );
            assert_eq!(
                ExternKind::from_name_for(name, Prelude::Decoder),
                None,
                "decoder prelude must NOT resolve {name}",
            );
            // All six new externs are opaque (rank-0): no dimensions
            // for a vision body that names them.
            assert!(
                extern_shape(kind).is_empty(),
                "{name} extern must be opaque (rank-0)",
            );
        }
    }

    #[test]
    fn load_rows_is_lowering_only_op_kind() {
        // Contract: `LoadRows` is produced exclusively by the
        // `rows_lowering::materialize_rows` pass between `fuf::unroll` and the solver. It must
        // NOT be reachable via `from_name` (no parse-then-reject — a user that wrote
        // `load_rows(...)` in a body classifies as an unknown op rather than an OpKind that any
        // consumer is unprepared for). The materialize pass writes `outputs[0]` directly when
        // constructing the node.
        for source in scratchy_forward_compiler::RowsExtern::ALL {
            let load = OpKind::LoadRows(source);
            assert_eq!(load.as_str(), "load_rows");
            assert_eq!(OpKind::from_name("load_rows"), None);
        }
    }
}
