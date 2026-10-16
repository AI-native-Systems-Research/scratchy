// SPDX-License-Identifier: Apache-2.0
//! #554: THE CHUNK LADDER IS ADMITTED BY EVERY ATTENTION CLASS IN THE GRAPH OR BY NONE.
//!
//! The wiring under test is the graph→ladder path — [`admitted_chunk_ladder`],
//! [`admitted_chunk`], [`chunk_swept_per_request`] over a `SubtileIR` minted by
//! `lower_region`, the same front door a model's forward comes through — not
//! `ChunkScratch::of_chunk`'s own refusal arithmetic (the child, proven where it
//! lives). Before #554 the ladder was admitted off the graph's FIRST `AttnDecode`
//! alone, so a hybrid-attention model (gemma-4: sliding hd=256 layers plus
//! every-sixth global hd=512 ones) minted a 64-page chunk the global class's door
//! then refused at bake. The conjunction is pinned from both sides:
//!
//!   * a TWO-CLASS graph — gemma-4 26b's own sliding `(16, 8, 256)` beside its
//!     global `(16, 2, 512)` — mints NO rung, whichever class the graph states
//!     first (the first-class read was the defect), and narrows a requested rung
//!     to `None` the same way;
//!   * a UNIFORM graph — granite-8b's `(32, 8, 128)`, the chunk-admitting control —
//!     keeps exactly the one pool's own ladder, byte for byte.
//!
//! No device, no emulator session, no `SCRATCHY_PLAN_ONLY_BAKE`: these are pure
//! facts of the graph, so the file runs at default features.

use ktir_superdsc::sdsc_abstract::{ChunkScratch, PagedKvPool};
use scratchy_subtile::lower::{InputRef, LoweringInput, OpDesc};
use scratchy_subtile::model_geometry::{HeadDim, KvHeads, ModelAttnGeometry, QueryHeads};
use scratchy_subtile::subtile_ir::{
    AttnMask, GainConvention, NeoX, SourceShape, SubOp, SubtileIR, SubtileId, SubtileNode,
    TensorId, TensorRegion, TensorShape, lower_region,
};
use scratchy_target_spyre::lower_subtile_tape_to_ktir::{
    admitted_chunk, admitted_chunk_ladder, chunk_swept_per_request,
};

/// One geometry through the config path's own mint — the GQA proof spent where the
/// model spends it.
fn geom(nqh: u32, nkvh: u32, hd: u32) -> ModelAttnGeometry {
    ModelAttnGeometry::mint(QueryHeads::new(nqh), KvHeads::new(nkvh), HeadDim::new(hd))
        .unwrap_or_else(|| panic!("({nqh}, {nkvh}, {hd}) is a GQA-divisible geometry"))
}

/// One `AttnDecode` per geometry, in the order given — the graph shape the ladder is
/// asked of. Each attention reads its own q/kv sources at the widths its geometry's
/// projections state (`q_width` / `kv_width`), pre-populated caches, nothing else in
/// the region: the ladder has only the classes to read.
fn attn_region(geoms: &[ModelAttnGeometry]) -> SubtileIR<NeoX> {
    let mut sources = Vec::new();
    let mut ops = Vec::new();
    for g in geoms {
        let base = sources.len();
        // [q, prefix_k, prefix_v, new_k, new_v] — the attention's five, per class.
        sources.push(SourceShape {
            rows: 1,
            cols: g.q_width(),
        });
        sources.push(SourceShape {
            rows: 64,
            cols: g.kv_width(),
        });
        sources.push(SourceShape {
            rows: 64,
            cols: g.kv_width(),
        });
        sources.push(SourceShape {
            rows: 1,
            cols: g.kv_width(),
        });
        sources.push(SourceShape {
            rows: 1,
            cols: g.kv_width(),
        });
        ops.push(OpDesc {
            op: SubOp::attn_decode(*g, 1.0, 64, AttnMask::Causal),
            m: 1,
            inputs: (0..5).map(|i| InputRef::Ext(base + i)).collect(),
        });
    }
    let result = ops.len() - 1;
    lower_region(
        &LoweringInput {
            sources,
            ops,
            result,
            exported: None,
        },
        std::num::NonZeroU32::new(8192).expect("8192 != 0"),
    )
}

/// gemma-4 26b's own two classes — sliding `(16, 8, 256)` and every-sixth global
/// `(16, 2, 512)` — in one graph, the exact shape that hit #554. No rung may be
/// minted, whichever class the graph states FIRST: the first-class read was the
/// defect, so both orders must refuse.
#[test]
fn a_hybrid_graph_mints_no_chunk_ladder_in_either_order() {
    let sliding = geom(16, 8, 256);
    let global = geom(16, 2, 512);
    for (first, second) in [(sliding, global), (global, sliding)] {
        let ir = attn_region(&[first, second]);
        let ladder = admitted_chunk_ladder(&ir, 1);
        assert!(
            ladder.is_empty(),
            "a graph mixing {first:?} with {second:?} must keep the per-page fold, \
             not mint rungs {ladder:?}"
        );
        // The same conjunction narrows a requested rung — the bundle's own door — and
        // keys its sweep: a width one class refuses is not a chunk of this graph's.
        assert_eq!(admitted_chunk(&ir, Some(64)), None);
        assert_eq!(chunk_swept_per_request(&ir, 64, 1), None);
    }
}

/// granite-8b's single class `(32, 8, 128)`, stated twice the way a rolled layer body
/// states one class per attention — the chunk-admitting control. Its ladder is exactly
/// the one pool's own `ladder_for`, byte for byte: the conjunction must leave a uniform
/// graph's emission alone.
#[test]
fn a_uniform_graph_keeps_its_own_ladder() {
    let g = geom(32, 8, 128);
    let ir = attn_region(&[g, g]);
    assert_eq!(
        admitted_chunk_ladder(&ir, 1),
        ChunkScratch::ladder_for(PagedKvPool::new(8, 128), 1),
        "a uniform graph's ladder is its one pool's own"
    );
    // And the narrowing and sweep keys answer as the single pool always did.
    assert_eq!(admitted_chunk(&ir, Some(2)), Some(2));
    assert_eq!(
        chunk_swept_per_request(&ir, 2, 1),
        Some(2 * PagedKvPool::PAGE_SLOTS as u32)
    );
}

/// A graph with no attention node has nothing to chunk — the conjunction's empty-pool
/// arm, which keeps the ladder empty exactly as the first-class read's `None` did.
#[test]
fn a_graph_without_attention_has_no_ladder() {
    let tensors = vec![
        TensorShape { rows: 7, cols: 512 },
        TensorShape { rows: 1, cols: 512 },
        TensorShape { rows: 7, cols: 512 },
    ];
    let whole = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let (act, gain, out) = (whole(0), whole(1), whole(2));
    let ir: SubtileIR = SubtileIR {
        tensors,
        num_sources: 2,
        nodes: vec![SubtileNode {
            id: SubtileId::from_index(0),
            op: SubOp::RmsNorm {
                eps: 1e-5,
                gain: GainConvention::Scale,
            },
            inputs: vec![act, gain],
            output: out,
        }],
        result: TensorId::from_index(2),
        op_output: Vec::new(),
    };
    assert!(admitted_chunk_ladder(&ir, 1).is_empty());
    assert_eq!(admitted_chunk(&ir, Some(2)), None);
}
