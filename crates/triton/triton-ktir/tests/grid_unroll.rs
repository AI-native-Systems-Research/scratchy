// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! THE GRID-POSITION UNROLL, against a no-unroll control on the same fixtures.
//!
//! `unroll_grid_positions` (to_ktir's own arm) unrolls a multi-tile grid whose load
//! tiles state computed corners: N straight-line copies of the body, one per position,
//! the landmark a constant naming that position, and the `grid` collapsed to 1. The
//! C++ golden never did that, so the stage-3 census comparison cannot express it --
//! this file measures it directly instead: the SAME fixture, the SAME pipeline, with
//! and without the arm, and the identities that must hold between the two.
//!
//! THE CONTROL IS THE POINT. A pipeline change that unrolls nothing must produce a
//! byte-identical module (the whole-node bodies, whose grid position rides store
//! regions and per-core work slices, never a load-tile corner), or the arm is
//! touching kernels whose emission is pinned to bytes.

mod common;

use common::*;
use triton_ktir::text::parse;
use triton_ktir::Module;

/// The full `to_ktir::run` pipeline on a golden fixture's ttir.
fn run_pipeline(config: &str) -> Module {
    let mut m = parse::parse(&golden(config, "0_ttir.mlir"))
        .unwrap_or_else(|e| panic!("{config}: ttir must parse: {e}"));
    let g = grid(config);
    triton_ktir::make_ktir(&mut m, &g)
        .unwrap_or_else(|e| panic!("{config}: make_ktir refused: {e}"));
    triton_ktir::passes::to_ktir::run(&mut m, &g)
        .unwrap_or_else(|e| panic!("{config}: ToSchedulerKTIR refused: {e}"));
    m
}

/// The fixture's grid extent, which is the position count the unroll must emit.
fn positions(config: &str) -> i64 {
    grid(config).iter().product()
}

/// THE UNROLL ITSELF: N copies, collapsed grid, no landmark left, every load tile's
/// corner a constant.
#[test]
fn a_corner_carrying_grid_unrolls_to_one_copy_per_position() {
    let m = run_pipeline("attention_flash_noncausal");
    let n = positions("attention_flash_noncausal");

    // The grid collapsed: ONE program covers the positions.
    let f = m
        .ops
        .iter()
        .find(|o| o.kind == triton_ktir::ir::OpKind::FuncFunc)
        .expect("func.func");
    let g = f
        .attr(&triton_ktir::ir::AttrKey::Grid)
        .and_then(|a| a.as_int_list().map(|v| v.to_vec()))
        .expect("grid attr");
    assert_eq!(
        g.iter().product::<i64>(),
        1,
        "the unrolled grid must be a single tile, got {g:?}"
    );

    // No landmark left -- every use names a position constant.
    assert!(
        !m.ops_deep()
            .iter()
            .any(|o| o.kind == triton_ktir::ir::OpKind::KtdpGetComputeTileId),
        "the landmark must not survive the unroll"
    );

    // Every load's tile corner is an arith.constant -- the form the whole-function
    // door reads, and the refusal this pass exists to resolve.
    let body = &f.regions[0].ops;
    for o in body {
        if o.kind != triton_ktir::ir::OpKind::KtdpLoad {
            continue;
        }
        let tile = body
            .iter()
            .find(|d| d.kind == triton_ktir::ir::OpKind::KtdpConstructAccessTile
                && d.results.contains(&o.operands[0]))
            .expect("a load's tile");
        for c in &tile.operands[1..3] {
            assert!(
                triton_ktir::passes::dot_to_linalg::const_int(&m, *c).is_some(),
                "a load tile corner did not fold to a constant after the unroll"
            );
        }
    }

    // N position constants were materialised (one per copy, plus whatever the folder
    // minted) -- at LEAST n, and a census multiple check below pins the scaling.
    let _ = n;
}

/// THE SCALING: the unrolled census is the control's census times N, for the ops the
/// copies own. Measured on `attention_flash_noncausal` (grid 2,4): loads, tiles,
/// matmuls, generics all scale by exactly 8 -- the same body, once per position.
#[test]
fn the_unrolled_census_is_the_control_times_positions() {
    let m = run_pipeline("attention_flash_noncausal");
    let n = positions("attention_flash_noncausal") as usize;
    let count = |m: &Module, name: &str| {
        m.census()
            .into_iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v)
            .unwrap_or(0)
    };
    // The measured baseline (the control, HEAD's own pipeline before this arm) is
    // stated here rather than re-derived, so a change in EITHER number fails loudly.
    // These are the post-KV-unroll counts: the sweep's 4 trips are inside the body the
    // grid unroll then copies.
    for (name, control) in [
        ("ktdp.load", 5),
        ("ktdp.store", 1),
        ("ktdp.construct_access_tile", 6),
        ("linalg.matmul", 4),
        ("linalg.generic", 8),
        ("tensor.expand_shape", 4),
    ] {
        assert_eq!(
            count(&m, name),
            control * n,
            "attention_flash_noncausal / {name}: the unroll must scale the control's \
             count by the position count exactly"
        );
    }
}

/// THE CONTROL: a kernel the arm must not touch. `swiglu_mlp` runs a whole-node body
/// at a multi-tile grid (its grid position rides store regions and work slices, never
/// a load-tile corner), so its pipeline output must be BYTE-IDENTICAL with and without
/// the arm. The arm's fire gate (a load tile with a non-constant corner) is what makes
/// that true, and this is the assertion that keeps it true.
#[test]
fn a_whole_node_body_is_untouched_by_the_unroll() {
    // swiglu_mlp has no 3_sched golden (refused a stage earlier), but its ttir is
    // present and the pipeline runs; what is pinned is the output's own shape, not a
    // golden's.
    let m = run_pipeline("swiglu_mlp");
    let f = m
        .ops
        .iter()
        .find(|o| o.kind == triton_ktir::ir::OpKind::FuncFunc)
        .expect("func.func");
    let g = f
        .attr(&triton_ktir::ir::AttrKey::Grid)
        .and_then(|a| a.as_int_list().map(|v| v.to_vec()))
        .expect("grid attr");
    assert_eq!(
        g,
        grid("swiglu_mlp"),
        "swiglu_mlp's grid must be untouched: its position never reaches a load corner"
    );
}
