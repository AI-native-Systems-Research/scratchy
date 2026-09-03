// SPDX-License-Identifier: Apache-2.0
//! Standalone verification for `subtile_tape_to_tile_ir::lower_tape_to_tile_ir` — builds a REAL
//! `SubtileIR`+`SubtileTape` from real fixtures and runs the pass end-to-end. Bypasses the
//! bit-rotted `#[cfg(test)]` module in `lower_subtile_tape_to_superdsc.rs` (pre-existing, unrelated
//! signature drift — `cargo test -p scratchy-subtile` fails to even COMPILE independent of this file;
//! confirmed via `git stash` before adding this pass).
//!
//!   cargo run -p scratchy-subtile --features superdsc --example verify_subtile_tape_to_tile_ir

use scratchy_subtile::subtile_ir::{ValidatedGraph, lower_region};
use scratchy_subtile::subtile_tape::lower_dag_to_tape;
use scratchy_target_spyre::subtile_tape_to_tile_ir::lower_tape_to_tile_ir;
use scratchy_target_spyre::tile_op::TileOpKind;
use std::num::NonZeroU32;

fn main() {
    verify_add_only();
    verify_gemm_m1();
    verify_rope_pure_syntax();
    verify_attn_decode_pure_syntax();
    println!(
        "OK: add_only (Elementwise), gemm_m1 (MatmulTile), rope + attn (undecomposed) all lower + tile end-to-end"
    );
}

/// FOURTH real `SubOp` kind: `one_layer_input` (a full production-shaped decode layer) contains ONE
/// `SubOp::AttnDecode` node among its RmsNorm/Gemm/RopeRotate/RopeAppend/SiluMul/Add nodes. Proves the
/// AttnDecode arm — same pure-syntax convention as RoPE — actually runs on a real, full-layer tape.
fn verify_attn_decode_pure_syntax() {
    let input = scratchy_subtile::fixtures::one_layer_input();
    let nb = NonZeroU32::new(8192).unwrap(); // wide enough to keep every op untiled
    let rg = lower_region(&input, nb);
    let valid = ValidatedGraph::new(&rg).expect("one_layer_input fixture validates");
    let tape = lower_dag_to_tape(&valid);
    let tiled_computes =
        lower_tape_to_tile_ir(&tape, &rg).expect("every node in one_layer_input is mapped");

    let mut found = 0;
    for (i, tc) in tiled_computes.iter().enumerate() {
        // AttnDecode's node id is opaque outside the crate (SubtileId is sealed), so identify it by
        // its TileOp's operand count: AttnDecode's inputs are [q', prefix_k, prefix_v, new_k, v] = 5,
        // +1 output = 6 — distinct from every other node in this layer (RmsNorm=2in+1, Gemm handled
        // separately as Matmul, RopeRotate/Append=3or6in+1, SiluMul=2in+1, Add=2in+1).
        if let [op] = tc.tile_ops.as_slice()
            && matches!(op.kind, TileOpKind::PointwiseOrReduce { n_operands: 6 })
        {
            found += 1;
            println!("compute[{i}]: AttnDecode-shaped TileOp (n_operands=6) found");
            let tiled = op
                .tile(
                    scratchy_subtile::superdsc_opspec::MaxCores::<32>,
                    |dims, n| {
                        let _ = (dims, n);
                        Default::default()
                    },
                    "out",
                )
                .expect("tiler accepts the lowered AttnDecode TileOp");
            println!(
                "  -> TiledOp cores_used={} time_tile={:?}",
                tiled.plan.cores_used().get(),
                tiled.time_tile.map(|t| t.count())
            );
        }
    }
    assert_eq!(
        found, 1,
        "exactly one AttnDecode node (n_operands=6) in one_layer_input"
    );
}

/// THIRD real `SubOp` kind: `rope_rotate_only_input` lowers to a `SubOp::RopeRotate` node — as ONE
/// undecomposed `TileOp` reflecting the node's OWN `[rows,cols]` shape and input count, NOT split into
/// its constituent primitive ops (rotate-matmul + xc + rs + add). That decomposition belongs to a
/// LATER stage (an explicit IR-to-IR transform, or TileIR→SdscOp itself) — lowering only translates,
/// it does not decompose or schedule. Proves rows=1 (this fixture, decode-shaped) hits the exact same
/// arm any row count would.
fn verify_rope_pure_syntax() {
    let input = scratchy_subtile::fixtures::rope_rotate_only_input(); // rows=1, heads=32, head_dim=64
    let nb = NonZeroU32::new(4096).unwrap(); // wide enough to keep this untiled
    let rg = lower_region(&input, nb);
    let valid = ValidatedGraph::new(&rg).expect("rope_rotate_only fixture validates");
    let tape = lower_dag_to_tape(&valid);
    let tiled_computes =
        lower_tape_to_tile_ir(&tape, &rg).expect("RopeRotate is a mapped SubOp kind");
    assert_eq!(
        tiled_computes.len(),
        1,
        "rope_rotate_only_input is a single RopeRotate node"
    );
    let ops = &tiled_computes[0].tile_ops;
    assert_eq!(ops.len(), 1, "one node -> one TileOp, undecomposed");
    let tile_op = &ops[0];
    assert!(
        matches!(
            tile_op.kind,
            TileOpKind::PointwiseOrReduce { n_operands: 4 }
        ),
        "3 declared inputs (x, cos, sin) + 1 output"
    );
    let mb_dim = tile_op
        .dims
        .iter()
        .find(|d| d.name == "mb")
        .expect("mb present");
    assert_eq!(
        mb_dim.size, 1,
        "rope_rotate_only_input is rows=1 by construction"
    );

    let tiled = tile_op
        .tile(
            scratchy_subtile::superdsc_opspec::MaxCores::<32>,
            |dims, n| {
                let _ = (dims, n);
                Default::default()
            },
            "out",
        )
        .expect("tiler accepts the lowered RopeRotate TileOp");
    println!(
        "rope: TileOp -> TiledOp cores_used={} time_tile={:?}",
        tiled.plan.cores_used().get(),
        tiled.time_tile.map(|t| t.count())
    );
}

/// SECOND real `SubOp` kind, beyond the elementwise case below: `gemm_m1_only_input` lowers to a
/// `SubOp::MatmulTile` node — proving `node_to_tile_ops`'s `Matmul` arm (the `in`(K) extent read off
/// the first input's column region, not hardcoded) actually runs on a real tape, not just compiles.
fn verify_gemm_m1() {
    let input = scratchy_subtile::fixtures::gemm_m1_only_input(); // m=1, k=2048, n=2048, dense weight
    let nb = NonZeroU32::new(2048).unwrap(); // >= both k and n: one untiled MatmulTile, no K-chunk combine
    let rg = lower_region(&input, nb);
    let valid = ValidatedGraph::new(&rg).expect("gemm_m1 fixture validates");
    let tape = lower_dag_to_tape(&valid);
    assert!(!tape.instrs().is_empty(), "tape has at least one Compute");

    let tiled_computes =
        lower_tape_to_tile_ir(&tape, &rg).expect("MatmulTile is a mapped SubOp kind");
    assert!(
        !tiled_computes.is_empty(),
        "at least one Compute produced a TiledCompute"
    );

    for (i, tc) in tiled_computes.iter().enumerate() {
        assert_eq!(
            tc.tile_ops.len(),
            1,
            "a plain MatmulTile node is exactly one TileOp"
        );
        let tile_op = &tc.tile_ops[0];
        assert!(
            matches!(tile_op.kind, TileOpKind::Matmul),
            "gemm lowers to TileOpKind::Matmul"
        );
        let mb_dim = tile_op
            .dims
            .iter()
            .find(|d| d.name == "mb")
            .expect("mb dim present");
        let in_dim = tile_op
            .dims
            .iter()
            .find(|d| d.name == "in")
            .expect("in (K) dim present");
        let out_dim = tile_op
            .dims
            .iter()
            .find(|d| d.name == "out")
            .expect("out (N) dim present");
        assert_eq!(mb_dim.size, 1, "gemm_m1_only_input is m=1 by construction");
        assert_eq!(in_dim.size, 2048, "K extent matches the fixture");
        assert_eq!(out_dim.size, 2048, "N extent matches the fixture");

        let tiled = tile_op
            .tile(
                scratchy_subtile::superdsc_opspec::MaxCores::<32>,
                |dims, n| {
                    let _ = (dims, n);
                    Default::default()
                },
                "out",
            )
            .expect("tiler accepts the lowered matmul TileOp");
        println!(
            "gemm compute[{i}]: TileOp -> TiledOp cores_used={} time_tile={:?}",
            tiled.plan.cores_used().get(),
            tiled.time_tile.map(|t| t.count())
        );
    }
}

fn verify_add_only() {
    // add_only_input: two [1,2048] sources -> Elementwise(Add) -> [1,2048]. Known by construction
    // (fixtures.rs), asserted against directly — SubtileId is deliberately sealed (`pub(crate)`, no
    // out-of-crate reader, unlike TensorId), so this example does NOT index back into `rg.nodes`; it
    // checks the LOWERING'S OUTPUT against the fixture's own known shape instead.
    let input = scratchy_subtile::fixtures::add_only_input();
    let (expect_rows, expect_cols) = (1u32, 2048u32);
    let nb = NonZeroU32::new(expect_cols).unwrap(); // untiled: exactly one Elementwise(Add) node
    let rg = lower_region(&input, nb);
    let valid = ValidatedGraph::new(&rg).expect("add_only fixture validates");
    let tape = lower_dag_to_tape(&valid);
    assert!(!tape.instrs().is_empty(), "tape has at least one Compute");

    let tiled_computes = lower_tape_to_tile_ir(&tape, &rg).expect("Add is a mapped SubOp kind");
    assert!(
        !tiled_computes.is_empty(),
        "at least one Compute produced a TiledCompute"
    );
    assert_eq!(
        tiled_computes.len(),
        1,
        "add_only_input is a single untiled op"
    );

    for (i, tc) in tiled_computes.iter().enumerate() {
        assert_eq!(
            tc.tile_ops.len(),
            1,
            "a plain Elementwise node is exactly one TileOp"
        );
        let tile_op = &tc.tile_ops[0];
        let mb_dim = tile_op
            .dims
            .iter()
            .find(|d| d.name == "mb")
            .expect("mb dim present");
        let out_dim = tile_op
            .dims
            .iter()
            .find(|d| d.name == "out")
            .expect("out dim present");
        // The TileOp's extents match the fixture's KNOWN shape — proving the lowering reads shape
        // off the node (rows==1 hits the SAME arm any rows would), not a hardcoded constant.
        assert_eq!(
            mb_dim.size, expect_rows,
            "mb extent matches the fixture's row count"
        );
        assert_eq!(
            out_dim.size, expect_cols,
            "out extent matches the fixture's col count"
        );
        assert!(
            matches!(
                tile_op.kind,
                TileOpKind::PointwiseOrReduce { n_operands: 3 }
            ),
            "binary elementwise = 2 inputs + 1 output"
        );

        // Actually RUN the tiler on the lowered TileOp — the full pipeline
        // (SubtileTape -> TileOp [TileIR] -> tiler -> TiledOp) end-to-end.
        let tiled = tile_op
            .tile(
                scratchy_subtile::superdsc_opspec::MaxCores::<32>,
                |dims, n| {
                    // A trivial 1-core splitter (this fixture's mb=1,out=2048 fits without splitting) —
                    // real ops use `distribute_cores`; this example only proves the plumbing runs.
                    let _ = (dims, n);
                    Default::default()
                },
                "out",
            )
            .expect("tiler accepts the lowered TileOp");
        println!(
            "compute[{i}]: TileOp -> TiledOp cores_used={} time_tile={:?}",
            tiled.plan.cores_used().get(),
            tiled.time_tile.map(|t| t.count())
        );
    }

    println!(
        "OK: {} Compute instruction(s) lowered to TileIR and tiled successfully",
        tiled_computes.len()
    );
}
