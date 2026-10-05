// SPDX-License-Identifier: Apache-2.0
//! Host-staged rows → tile FUF lowering pass.
//!
//! The DSL exposes runtime row tensors — `pixels` and `pos_embeds` in a `#[vision_forward]`
//! body, `target_hidden` in an MTP head's `#[forward]` — as `ExternKind`s, so a body reads them
//! directly (`out = quick_gelu(pixels)`) without wrapping the buffer in an explicit op call.
//! Every Impl consumes its inputs as `FufInput::Tile`, so the extern → tile transition happens
//! exactly once, up front, rather than being hand-unrolled into every per-Impl
//! `consumes_input_tiles` / `fan_out`.
//!
//! [`materialize_rows`] runs between [`crate::fuf::unroll`] and the solver. For each
//! [`RowsExtern`] the body reads it synthesizes one `OpKind::LoadRows` node with no FUF inputs
//! and the rank-2 output shape [`crate::shape::extern_shape`] gives that extern, and rewrites
//! every FUF input that read the extern to read the new node's slot 0.
//!
//! Mirrors the role `EmbedRefImpl` plays for `input_ids`: an Impl that reads an ambient
//! `ForwardCtx` field rather than a tile dataflow. `Embed` can carry that read itself because its
//! second input is a weight (the embed_tokens table); a row extern has no weight to anchor on,
//! so the materialization is its own OpKind.

use crate::classified::{ExternKind, OpKind};
use crate::fuf::{Fuf, FufInput, FufNode, TileId};
use crate::shape::extern_shape;
use scratchy_forward_compiler::RowsExtern;

/// Synthesize one `LoadRows` tile per row extern the FUF reads and rewire every
/// `FufInput::Extern` of that kind to read it. Sources the body never reads get no tile.
pub(crate) fn materialize_rows(fuf: &mut Fuf) {
    for source in RowsExtern::ALL {
        let kind = ExternKind::of_rows(source);
        let reads = |i: &FufInput| matches!(i, FufInput::Extern { kind: k, .. } if *k == kind);
        if !fuf.nodes.iter().any(|node| node.inputs.iter().any(reads)) {
            continue;
        }

        let load_id = TileId(fuf.nodes.len() as u32);
        fuf.nodes.push(FufNode {
            id: load_id,
            op: OpKind::LoadRows(source),
            inputs: Vec::new(),
            outputs: vec![extern_shape(kind)],
        });

        for node in &mut fuf.nodes {
            if node.id == load_id {
                continue;
            }
            for input in &mut node.inputs {
                if reads(input) {
                    *input = FufInput::Tile {
                        id: load_id,
                        slot: 0,
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Dim;

    fn bound(name: &str) -> Dim {
        Dim::Bound(name.into())
    }

    fn reads(kind: ExternKind) -> FufInput {
        FufInput::Extern { kind, index: None }
    }

    /// Synthetic: one node consuming `pixels` directly. After the
    /// pass the FUF has one extra `LoadRows(Pixels)` tile and the original
    /// node reads from it instead of from the extern.
    fn fuf_with_one_pixels_consumer() -> Fuf {
        let pixels_shape = vec![bound("num_tokens"), bound("vision_in_features")];
        Fuf {
            nodes: vec![FufNode {
                id: TileId(0),
                op: OpKind::QuickGelu,
                inputs: vec![reads(ExternKind::Pixels)],
                outputs: vec![pixels_shape],
            }],
        }
    }

    #[test]
    fn materialize_rows_inserts_one_load_rows_tile() {
        let mut fuf = fuf_with_one_pixels_consumer();
        materialize_rows(&mut fuf);
        assert_eq!(fuf.nodes.len(), 2, "one extra LoadRows tile");

        let load = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::LoadRows(RowsExtern::Pixels))
            .expect("LoadRows(Pixels) tile inserted");
        assert!(load.inputs.is_empty(), "LoadRows has no FUF inputs");
        assert_eq!(
            load.outputs,
            vec![vec![bound("num_tokens"), bound("vision_in_features")]],
            "LoadRows output shape matches extern_shape(Pixels)",
        );
    }

    #[test]
    fn materialize_rows_rewires_consumers_to_tile_input() {
        let mut fuf = fuf_with_one_pixels_consumer();
        materialize_rows(&mut fuf);

        let load_id = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::LoadRows(RowsExtern::Pixels))
            .expect("LoadRows(Pixels) tile inserted")
            .id;
        let consumer = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::QuickGelu)
            .expect("original consumer present");
        match consumer.inputs.first() {
            Some(FufInput::Tile { id, slot }) => {
                assert_eq!(*id, load_id, "consumer reads LoadRows");
                assert_eq!(*slot, 0, "consumer reads slot 0");
            }
            other => panic!(
                "consumer's first input must now be a Tile (got {other:?}) \
                 — pass failed to rewire pixels-extern"
            ),
        }
    }

    #[test]
    fn materialize_rows_is_noop_without_row_externs() {
        // A body that reads no row extern — the pass must not synthesize a LoadRows tile that
        // would never have a runtime-populated value.
        let mut fuf = Fuf {
            nodes: vec![FufNode {
                id: TileId(0),
                op: OpKind::Add,
                inputs: vec![FufInput::Scalar(0.0), FufInput::Scalar(0.0)],
                outputs: vec![vec![]],
            }],
        };
        let before = fuf.nodes.len();
        materialize_rows(&mut fuf);
        assert_eq!(fuf.nodes.len(), before, "no LoadRows when no consumer");
        assert!(
            fuf.nodes
                .iter()
                .all(|n| !matches!(n.op, OpKind::LoadRows(_))),
            "no LoadRows tile must be synthesized",
        );
    }

    #[test]
    fn materialize_rows_shares_one_tile_across_multiple_consumers() {
        // Multiple consumers must share a single LoadRows tile — ctx.fwd.pixels is a single
        // buffer, and synthesizing one tile per reference would multiply both the runtime tile-
        // table footprint and the solver's per-Impl cost.
        let pixels_shape = vec![bound("num_tokens"), bound("vision_in_features")];
        let mut fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::QuickGelu,
                    inputs: vec![reads(ExternKind::Pixels)],
                    outputs: vec![pixels_shape.clone()],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::GeluErf,
                    inputs: vec![reads(ExternKind::Pixels)],
                    outputs: vec![pixels_shape],
                },
            ],
        };
        materialize_rows(&mut fuf);

        let load = OpKind::LoadRows(RowsExtern::Pixels);
        let load_count = fuf.nodes.iter().filter(|n| n.op == load).count();
        assert_eq!(load_count, 1, "one LoadRows tile shared across consumers");
        assert_eq!(fuf.nodes.len(), 3, "two original consumers + one LoadRows");

        let load_id = fuf.nodes.iter().find(|n| n.op == load).unwrap().id;
        for consumer in fuf.nodes.iter().filter(|n| n.op != load) {
            match consumer.inputs.first() {
                Some(FufInput::Tile { id, slot }) => {
                    assert_eq!(
                        *id, load_id,
                        "consumer {:?} reads shared LoadRows",
                        consumer.id
                    );
                    assert_eq!(*slot, 0);
                }
                other => panic!(
                    "consumer {:?} retained Extern input after pass: {:?}",
                    consumer.id, other,
                ),
            }
        }
    }

    #[test]
    fn materialize_rows_preserves_other_externs() {
        // CuSeqlens / Cos / Sin / GridThw / MaxSeqlen stay as FufInput::Extern — only row
        // externs get materialized into a tile. The others are consumed via ForwardCtx
        // (cu_seqlens_q / max_seqlen_q / vision_rope_cos / sin) at runtime and don't have any
        // tile pattern.
        let q_shape = vec![bound("num_tokens"), bound("h_d")];
        let mut fuf = Fuf {
            nodes: vec![FufNode {
                id: TileId(0),
                op: OpKind::VarlenAttention,
                inputs: vec![
                    reads(ExternKind::Pixels),
                    reads(ExternKind::Pixels),
                    reads(ExternKind::Pixels),
                    reads(ExternKind::CuSeqlens),
                    reads(ExternKind::MaxSeqlen),
                ],
                outputs: vec![q_shape],
            }],
        };
        materialize_rows(&mut fuf);

        let consumer = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::VarlenAttention)
            .unwrap();
        assert!(
            matches!(consumer.inputs[0], FufInput::Tile { .. }),
            "Pixels rewired to Tile",
        );
        assert!(
            matches!(
                consumer.inputs[3],
                FufInput::Extern {
                    kind: ExternKind::CuSeqlens,
                    ..
                },
            ),
            "CuSeqlens preserved as Extern",
        );
        assert!(
            matches!(
                consumer.inputs[4],
                FufInput::Extern {
                    kind: ExternKind::MaxSeqlen,
                    ..
                },
            ),
            "MaxSeqlen preserved as Extern",
        );
    }

    /// Each source gets its own tile, shaped by its own extern: an MTP head reads its target's
    /// hidden rows (`[num_tokens, hidden_size]`) beside the token embedding.
    #[test]
    fn materialize_rows_gives_each_source_its_own_tile() {
        let mut fuf = Fuf {
            nodes: vec![FufNode {
                id: TileId(0),
                op: OpKind::Add,
                inputs: vec![
                    reads(ExternKind::TargetHidden),
                    reads(ExternKind::PosEmbeds),
                ],
                outputs: vec![vec![bound("num_tokens"), bound("hidden_size")]],
            }],
        };
        materialize_rows(&mut fuf);

        let tile_of = |source| {
            let node = fuf
                .nodes
                .iter()
                .find(|n| n.op == OpKind::LoadRows(source))
                .unwrap_or_else(|| panic!("LoadRows({source:?}) tile inserted"));
            (node.id, node.outputs[0].clone())
        };
        let (hidden, hidden_shape) = tile_of(RowsExtern::TargetHidden);
        let (pos, _) = tile_of(RowsExtern::PosEmbeds);
        assert_eq!(
            hidden_shape,
            vec![bound("num_tokens"), bound("hidden_size")]
        );
        let consumer = &fuf.nodes[0].inputs;
        assert!(matches!(consumer[0], FufInput::Tile { id, slot: 0 } if id == hidden));
        assert!(matches!(consumer[1], FufInput::Tile { id, slot: 0 } if id == pos));
    }
}
