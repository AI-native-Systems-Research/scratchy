// SPDX-License-Identifier: Apache-2.0
//! Vision-prelude FUF lowering pass.
//!
//! The DSL exposes `pixels` (and friends) as `ExternKind`s so a
//! `#[vision_forward]` body can write `out = quick_gelu(pixels)`
//! without first wrapping the patches buffer in an explicit
//! op call. Every existing vision Impl
//! (`VarlenAttentionImpl` / `VisionRopeImpl` / `QuickGeluImpl` /
//! `GeluErfImpl`) consumes its first input as `FufInput::Tile`, so
//! the extern → tile transition has to happen exactly once,
//! up-front, rather than being hand-unrolled into every per-Impl
//! `consumes_input_tiles` / `fan_out`.
//!
//! [`materialize_pixels`] runs between [`crate::fuf::unroll`] and
//! the solver under `Prelude::Vision`. It synthesizes a single
//! `OpKind::LoadPixels` node with no FUF inputs and a rank-2 output
//! shape `[num_tokens, vision_in_features]` — the same shape
//! [`crate::shape::extern_shape`] returns for `ExternKind::Pixels`.
//! Every other FUF input that previously read the pixels extern is
//! rewritten to read the new node's slot 0.
//!
//! Mirrors the role `EmbedRefImpl` plays for `input_ids` on the
//! decoder side: an Impl that reads from an ambient `ForwardCtx`
//! field rather than a tile dataflow. The text-side analog is
//! special-cased inside the Impl because `Embed`'s second input is
//! a weight (the embed_tokens table), and `Embed`'s output is the
//! seed every other text-side Impl already expects as `Tile`. On
//! the vision side there's no weight to anchor on — the pixels
//! buffer is a pure runtime extern — so the materialization needs
//! its own OpKind + Impl pair.

use crate::classified::{ExternKind, OpKind};
use crate::fuf::{Fuf, FufInput, FufNode, TileId};
use crate::shape::extern_shape;

/// Synthesize a single `LoadPixels` tile and rewire every
/// `FufInput::Extern { kind: Pixels, .. }` to read it. No-op when
/// the FUF carries no Pixels-extern reference (text-only bodies in
/// vision-prelude mode are not the production path, but this pass
/// is total over them).
pub(crate) fn materialize_pixels(fuf: &mut Fuf) {
    let needs_load = fuf.nodes.iter().any(|node| {
        node.inputs.iter().any(|i| {
            matches!(
                i,
                FufInput::Extern {
                    kind: ExternKind::Pixels,
                    ..
                }
            )
        })
    });
    if !needs_load {
        return;
    }

    let load_id = TileId(fuf.nodes.len() as u32);
    fuf.nodes.push(FufNode {
        id: load_id,
        op: OpKind::LoadPixels,
        inputs: Vec::new(),
        outputs: vec![extern_shape(ExternKind::Pixels)],
    });

    for node in &mut fuf.nodes {
        if node.id == load_id {
            continue;
        }
        for input in &mut node.inputs {
            if let FufInput::Extern {
                kind: ExternKind::Pixels,
                ..
            } = input
            {
                *input = FufInput::Tile {
                    id: load_id,
                    slot: 0,
                };
            }
        }
    }
}

/// Synthesize a single `LoadPosEmbeds` tile and rewire every
/// `FufInput::Extern { kind: PosEmbeds, .. }` to read it. The exact
/// sibling of [`materialize_pixels`] for Qwen3.5-VL's host-interpolated
/// learned positional embedding (consumed by `add(pos_embeds, …)` after
/// patch_embed). No-op when the FUF carries no PosEmbeds-extern
/// reference (any tower without a learned pos-embed).
pub(crate) fn materialize_pos_embeds(fuf: &mut Fuf) {
    let needs_load = fuf.nodes.iter().any(|node| {
        node.inputs.iter().any(|i| {
            matches!(
                i,
                FufInput::Extern {
                    kind: ExternKind::PosEmbeds,
                    ..
                }
            )
        })
    });
    if !needs_load {
        return;
    }

    let load_id = TileId(fuf.nodes.len() as u32);
    fuf.nodes.push(FufNode {
        id: load_id,
        op: OpKind::LoadPosEmbeds,
        inputs: Vec::new(),
        outputs: vec![extern_shape(ExternKind::PosEmbeds)],
    });

    for node in &mut fuf.nodes {
        if node.id == load_id {
            continue;
        }
        for input in &mut node.inputs {
            if let FufInput::Extern {
                kind: ExternKind::PosEmbeds,
                ..
            } = input
            {
                *input = FufInput::Tile {
                    id: load_id,
                    slot: 0,
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classified::OpKind;
    use crate::fuf::{Fuf, FufInput, FufNode, TileId};
    use crate::shape::Dim;

    fn bound(name: &str) -> Dim {
        Dim::Bound(name.into())
    }

    /// Synthetic: one node consuming `pixels` directly. After the
    /// pass the FUF has one extra `LoadPixels` tile and the original
    /// node reads from it instead of from the extern.
    fn fuf_with_one_pixels_consumer() -> Fuf {
        let pixels_shape = vec![bound("num_tokens"), bound("vision_in_features")];
        Fuf {
            nodes: vec![FufNode {
                id: TileId(0),
                op: OpKind::QuickGelu,
                inputs: vec![FufInput::Extern {
                    kind: ExternKind::Pixels,
                    index: None,
                }],
                outputs: vec![pixels_shape],
            }],
        }
    }

    #[test]
    fn materialize_pixels_inserts_one_load_pixels_tile() {
        let mut fuf = fuf_with_one_pixels_consumer();
        materialize_pixels(&mut fuf);
        assert_eq!(fuf.nodes.len(), 2, "one extra LoadPixels tile");

        let load = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::LoadPixels)
            .expect("LoadPixels tile inserted");
        assert!(load.inputs.is_empty(), "LoadPixels has no FUF inputs");
        assert_eq!(
            load.outputs,
            vec![vec![bound("num_tokens"), bound("vision_in_features")]],
            "LoadPixels output shape matches extern_shape(Pixels)",
        );
    }

    #[test]
    fn materialize_pixels_rewires_consumers_to_tile_input() {
        let mut fuf = fuf_with_one_pixels_consumer();
        materialize_pixels(&mut fuf);

        let load_id = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::LoadPixels)
            .expect("LoadPixels tile inserted")
            .id;
        let consumer = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::QuickGelu)
            .expect("original consumer present");
        match consumer.inputs.first() {
            Some(FufInput::Tile { id, slot }) => {
                assert_eq!(*id, load_id, "consumer reads LoadPixels");
                assert_eq!(*slot, 0, "consumer reads slot 0");
            }
            other => panic!(
                "consumer's first input must now be a Tile (got {other:?}) \
                 — pass failed to rewire pixels-extern"
            ),
        }
    }

    #[test]
    fn materialize_pixels_is_noop_without_pixels_extern() {
        // Text-only body translated through the vision prelude (no
        // pixels reference) — pass must not synthesize a LoadPixels
        // tile that would never have a runtime-populated value.
        let mut fuf = Fuf {
            nodes: vec![FufNode {
                id: TileId(0),
                op: OpKind::Add,
                inputs: vec![FufInput::Scalar(0.0), FufInput::Scalar(0.0)],
                outputs: vec![vec![]],
            }],
        };
        let before = fuf.nodes.len();
        materialize_pixels(&mut fuf);
        assert_eq!(fuf.nodes.len(), before, "no LoadPixels when no consumer");
        assert!(
            fuf.nodes.iter().all(|n| n.op != OpKind::LoadPixels),
            "no LoadPixels tile must be synthesized",
        );
    }

    #[test]
    fn materialize_pixels_shares_one_tile_across_multiple_consumers() {
        // Multiple consumers must share a single LoadPixels tile —
        // ctx.fwd.pixels is a single buffer, and synthesizing one
        // tile per reference would multiply both the runtime tile-
        // table footprint and the solver's per-Impl cost.
        let pixels_shape = vec![bound("num_tokens"), bound("vision_in_features")];
        let mut fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::QuickGelu,
                    inputs: vec![FufInput::Extern {
                        kind: ExternKind::Pixels,
                        index: None,
                    }],
                    outputs: vec![pixels_shape.clone()],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::GeluErf,
                    inputs: vec![FufInput::Extern {
                        kind: ExternKind::Pixels,
                        index: None,
                    }],
                    outputs: vec![pixels_shape],
                },
            ],
        };
        materialize_pixels(&mut fuf);

        let load_count = fuf
            .nodes
            .iter()
            .filter(|n| n.op == OpKind::LoadPixels)
            .count();
        assert_eq!(load_count, 1, "one LoadPixels tile shared across consumers");
        assert_eq!(
            fuf.nodes.len(),
            3,
            "two original consumers + one LoadPixels"
        );

        let load_id = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::LoadPixels)
            .unwrap()
            .id;
        for consumer in fuf.nodes.iter().filter(|n| n.op != OpKind::LoadPixels) {
            match consumer.inputs.first() {
                Some(FufInput::Tile { id, slot }) => {
                    assert_eq!(
                        *id, load_id,
                        "consumer {:?} reads shared LoadPixels",
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
    fn materialize_pixels_preserves_non_pixels_externs() {
        // CuSeqlens / Cos / Sin / GridThw / MaxSeqlen stay as
        // FufInput::Extern — only Pixels gets materialized into a
        // tile. The other externs are consumed via ForwardCtx
        // (cu_seqlens_q / max_seqlen_q / vision_rope_cos / sin) at
        // runtime and don't have any tile pattern.
        let q_shape = vec![bound("num_tokens"), bound("h_d")];
        let mut fuf = Fuf {
            nodes: vec![FufNode {
                id: TileId(0),
                op: OpKind::VarlenAttention,
                inputs: vec![
                    FufInput::Extern {
                        kind: ExternKind::Pixels,
                        index: None,
                    },
                    FufInput::Extern {
                        kind: ExternKind::Pixels,
                        index: None,
                    },
                    FufInput::Extern {
                        kind: ExternKind::Pixels,
                        index: None,
                    },
                    FufInput::Extern {
                        kind: ExternKind::CuSeqlens,
                        index: None,
                    },
                    FufInput::Extern {
                        kind: ExternKind::MaxSeqlen,
                        index: None,
                    },
                ],
                outputs: vec![q_shape],
            }],
        };
        materialize_pixels(&mut fuf);

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
}
