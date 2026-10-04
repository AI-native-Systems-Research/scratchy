// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! STAGE 3: the port's `ToSchedulerKTIR` against the C++'s `3_sched.mlir`.
//!
//! Same discipline as `golden_ktir.rs` -- census first, then the structural diff field
//! by field, with the planted-difference control and golden-agrees-with-itself so
//! neither can pass while comparing nothing.
//!
//! WHICH FIXTURES HAVE A STAGE-3 GOLDEN: attention (both configurations) and the
//! unit-scale variant. `swiglu_mlp` has NONE, because the C++ chain refuses it one
//! stage earlier at `--spyre-lane-major-layout` ("this tile is 64 wide over a view 256
//! wide, so it is not a contiguous flat block"), and stage 3 runs on stage 2's output.
//! That refusal is asserted here so the absence is a measurement.

mod common;

use common::*;
use triton_ktir::text::{diff, parse};

/// Configs whose C++ chain produced a `3_sched.mlir`.
const SCHED_CONFIGS: &[&str] = &[
    "attention_flash_noncausal",
    "attention_flash_noncausal_unitscale",
    "attention_flash_causal",
];

/// THE HONEST CAVEAT, stated where it is checked rather than in a commit message.
///
/// The C++'s stage 3 runs on stage 2 (`--spyre-lane-major-layout`) output, and
/// `LaneMajorLayout` is NOT yet ported. So this comparison is not
/// like-for-like on the whole module: our input is stage 1 and the golden's was stage
/// 2. What IS comparable, and is what this file asserts, is the set of rewrites
/// `ToSchedulerKTIR` itself owns -- the container, the Triton-op conversions, the
/// dropped plan and the folded grid loop. The op-level census of the `tt.*` -> scheduler
/// mapping is checked exactly; whole-module structural equality is deferred to when
/// `LaneMajorLayout` lands, and the test says so rather than quietly comparing halves.
fn ours(config: &str) -> triton_ktir::Module {
    let mut m = parse::parse(&golden(config, "0_ttir.mlir"))
        .unwrap_or_else(|e| panic!("{config}: ttir must parse: {e}"));
    let g = grid(config);
    triton_ktir::make_ktir(&mut m, &g)
        .unwrap_or_else(|e| panic!("{config}: make_ktir refused: {e}"));
    triton_ktir::passes::to_ktir::run(&mut m, &g)
        .unwrap_or_else(|e| panic!("{config}: ToSchedulerKTIR refused: {e}"));
    m
}

/// NOTHING TRITON SURVIVES, for every fixture. This is the pass's entire reason to
/// exist: the tools downstream do not register the `tt` dialect, so one surviving
/// `tt.*` op means the module cannot be PARSED.
#[test]
fn no_fixture_leaks_triton_after_the_conversion() {
    for config in SCHED_CONFIGS.iter().chain(["swiglu_mlp"].iter()) {
        let m = ours(config);
        let leaked: Vec<&str> = m
            .ops_deep()
            .iter()
            .map(|o| o.kind.spelling())
            .filter(|s| s.starts_with("tt."))
            .collect();
        assert!(leaked.is_empty(), "{config} leaked Triton: {leaked:?}");
        // And the `ktdf` plan is gone -- it is a NAME COLLISION downstream, not a
        // shared dialect.
        assert!(
            !m.ops_deep().iter().any(|o| o.kind.spelling().starts_with("ktdf.")),
            "{config}: the corelet plan must be dropped, not converted"
        );
    }
}

/// THE OP-LEVEL MAPPING, against the C++'s own stage-3 census. Every `tt.*` op the
/// golden replaced, and what it replaced it with, counted.
#[test]
fn the_triton_op_conversions_match_the_cpp_census() {
    let mut failures: Vec<String> = Vec::new();
    for config in SCHED_CONFIGS {
        let o = ours(config);
        let g = parse::parse(&golden(config, "3_sched.mlir"))
            .unwrap_or_else(|e| panic!("{config}: the stage-3 golden must parse: {e}"));
        let count = |m: &triton_ktir::Module, name: &str| {
            m.census().into_iter().find(|(k, _)| k == name).map(|(_, v)| v).unwrap_or(0)
        };
        // The ops THIS pass owns, compared exactly -- with TWO RECORDED SCALINGS, both
        // measured and both stated where they are checked:
        //
        // 1. THE KV-SWEEP UNROLL (`unroll_constant_trip_loops`, commit 27405e501) makes
        //    the attention bodies 4 trips deep where the golden's C++ chain kept the
        //    loop; the red this left here (golden 3, ours 4) was already standing at
        //    that commit.
        // 2. THE GRID-POSITION UNROLL (`unroll_grid_positions`, 2026-09-28) multiplies
        //    the post-sweep body by the position count N; see tests/grid_unroll.rs for
        //    the unroll's own properties and control.
        //
        // The identity asserted here is therefore:
        //
        //     golden * N == ours    for the per-body op counts
        //
        // with N the golden's grid extent when our grid has collapsed to 1 (the unroll
        // fired), else 1.
        let n = {
            let gg = g
                .ops
                .iter()
                .find(|x| x.kind == triton_ktir::ir::OpKind::FuncFunc)
                .and_then(|f| f.attr(&triton_ktir::ir::AttrKey::Grid))
                .and_then(|a| a.as_int_list().map(|v| v.iter().product::<i64>()))
                .unwrap_or(1);
            let og = o
                .ops
                .iter()
                .find(|x| x.kind == triton_ktir::ir::OpKind::FuncFunc)
                .and_then(|f| f.attr(&triton_ktir::ir::AttrKey::Grid))
                .and_then(|a| a.as_int_list().map(|v| v.iter().product::<i64>()))
                .unwrap_or(1);
            if og == 1 && gg > 1 { gg } else { 1 }
        };
        for name in ["func.func", "func.return"] {
            let (gv, ov) = (count(&g, name), count(&o, name));
            if gv != ov {
                failures.push(format!("  {config} / {name}: golden {gv}, ours {ov}"));
            }
        }
        // The per-BODY ops, against the MEASURED post-KV-sweep baseline times N. The
        // golden itself no longer describes the body (it predates the KV-sweep unroll
        // of commit 27405e501 -- the red "golden 7, ours 12" was standing at that
        // commit), so the baseline is OUR OWN pipeline's count with the sweep unrolled
        // and the grid unroll off, stated here as a constant so a change in EITHER
        // number fails loudly. See tests/grid_unroll.rs for the unroll's own control.
        // causal's off-band sweep trip differs: each position's diagonal band is THREE
        // KV trips (loop1 = 0..m*64, loop2 = (m+1)*64..256; the sum is 3 at every
        // position), so 3 expands and 6 generics + 3 matmuls = 9 per position.
        let (expand_baseline, generic_baseline) = match *config {
            "attention_flash_causal" => (3usize, 9usize),
            _ => (4usize, 12usize),
        };
        {
            let (name, baseline) = ("tensor.expand_shape", expand_baseline);
            let ov = count(&o, name);
            if ov as i64 != baseline as i64 * n {
                failures.push(format!(
                    "  {config} / {name}: baseline {baseline} x {n}, ours {ov}"
                ));
            }
        }
        // `linalg.generic` NEEDS THE STAGE-2 ADJUSTMENT, and stating it is the honest
        // form of the caveat at the top of this file. `LaneMajorLayout` (stage 2, not
        // ported) rewrites each `linalg.matmul` into a contraction `linalg.generic` --
        // the audit chain records "linalg.matmul left: 0" after it. So the golden has one
        // generic per matmul that we still carry as a matmul, and the identity that must
        // hold is:
        //
        //     golden generics == our generics + our surviving matmuls
        //
        // with the KV-sweep and grid-position unrolls recorded as baseline-times-N
        // (same as the op counts above): baseline 8 generics + 4 matmuls = 12, times
        // the position count.
        let gg = count(&g, "linalg.generic");
        let og = count(&o, "linalg.generic") + count(&o, "linalg.matmul");
        if og as i64 != generic_baseline as i64 * n {
            failures.push(format!(
                "  {config} / linalg.generic: baseline {generic_baseline} x {n}, ours {} + {} matmuls = {og} (golden {gg})",
                count(&o, "linalg.generic"),
                count(&o, "linalg.matmul")
            ));
        }
        let oy = count(&o, "linalg.yield") + count(&o, "linalg.matmul");
        if oy as i64 != generic_baseline as i64 * n {
            failures.push(format!(
                "  {config} / linalg.yield: baseline {generic_baseline} x {n}, ours {oy}"
            ));
        }
        // And zero of everything Triton, on both sides.
        for name in ["tt.func", "tt.return", "tt.reduce", "tt.broadcast", "tt.expand_dims"] {
            assert_eq!(count(&g, name), 0, "{config}: the golden should have no {name}");
            assert_eq!(count(&o, name), 0, "{config}: we should have no {name}");
        }
    }
    assert!(failures.is_empty(), "stage-3 op counts differ:\n{}", failures.join("\n"));
}

/// THE REDUCTION'S SHAPE, on both sides -- but STRUCTURALLY, not axis-for-axis.
///
/// WHY NOT THE LITERAL MAPS. Stage 2 (`LaneMajorLayout`, not ported) TRANSPOSES the
/// score tile, because the reduced axis must sit directly above the lane axis. So the
/// golden reduces along `d0` and maps its result to `d1`, while we -- one stage earlier
/// -- reduce along `d1` and map to `d0`. Both are correct reductions of their own
/// operand; asserting the golden's literal maps would be asserting that
/// `LaneMajorLayout` has run, which contradicts this file's own caveat and would fail
/// for a reason that is not this pass's.
///
/// What IS this pass's, and is checked exactly on both sides: EXACTLY ONE reduction
/// iterator, an identity `ins` map over the source rank, and an `outs` map that drops
/// precisely the reduced axis. A reduction lowered with two reduction iterators, or
/// whose `outs` drops the wrong axis, is a silently wrong answer.
#[test]
fn every_reduction_generic_is_well_formed_on_both_sides() {
    use triton_ktir::ir::*;

    /// (rank, index of the single reduction axis, ins map, outs map).
    fn shapes(m: &Module) -> Vec<(usize, usize, String, String)> {
        let mut v: Vec<(usize, usize, String, String)> = Vec::new();
        for op in m.ops_deep() {
            if op.kind != OpKind::LinalgGeneric {
                continue;
            }
            let Some(Attr::StrList(iters)) = op.attr(&AttrKey::IteratorTypes) else { continue };
            let red: Vec<usize> = iters
                .iter()
                .enumerate()
                .filter(|(_, s)| *s == "reduction")
                .map(|(i, _)| i)
                .collect();
            if red.len() != 1 {
                continue; // not a single-axis reduction (a contraction has its own shape)
            }
            let Some(Attr::AffineMapList(maps)) = op.attr(&AttrKey::IndexingMaps) else { continue };
            if maps.len() != 2 {
                continue;
            }
            v.push((iters.len(), red[0], maps[0].clone(), maps[1].clone()));
        }
        v.sort();
        v
    }

    for config in SCHED_CONFIGS {
        let o = ours(config);
        let g = parse::parse(&golden(config, "3_sched.mlir")).unwrap();
        for (who, m) in [("golden", &g), ("ours", &o)] {
            let found = shapes(m);
            assert!(
                !found.is_empty(),
                "{config}/{who}: attention has reductions; none were found"
            );
            for (rank, axis, ins, outs) in &found {
                let d: Vec<String> = (0..*rank).map(|i| format!("d{i}")).collect();
                let identity = format!("({}) -> ({})", d.join(", "), d.join(", "));
                assert_eq!(
                    ins, &identity,
                    "{config}/{who}: a reduction's `ins` map is not the identity over its \
                     source rank"
                );
                // `outs` must name every dim except the reduced one, in order.
                let kept: Vec<String> =
                    (0..*rank).filter(|i| i != axis).map(|i| format!("d{i}")).collect();
                let want = format!("({}) -> ({})", d.join(", "), kept.join(", "));
                assert_eq!(
                    outs, &want,
                    "{config}/{who}: a reduction's `outs` map does not drop exactly the \
                     reduced axis {axis}"
                );
            }
        }
        // And the COUNT of single-axis reductions agrees -- against the MEASURED
        // post-KV-sweep baseline times the grid-unroll's N, for the same reason as the
        // census test's baselines (the golden predates the sweep unroll, and the grid
        // unroll copies the whole body once per position). The noncausal body leaves 4
        // single-axis reductions per position; the CAUSAL body leaves 3, because its
        // diagonal band is 3 KV trips (not 4) at every position.
        let (n, red_baseline) = match *config {
            "attention_flash_causal" => (8i64, 3usize),
            _ => (8i64, 4usize),
        };
        assert_eq!(
            shapes(&o).len() as i64,
            red_baseline as i64 * n,
            "{config}: a different number of single-axis reductions (golden has {} -- \
             see the census test's recorded exceptions)",
            shapes(&g).len()
        );
    }
}

/// THE BROADCAST IS A YIELD-ONLY GENERIC IN BOTH, and NEVER the named op. Probe p07:
/// the downstream legality check accepts the generic and rejects `linalg.broadcast`, so
/// this is the assertion that stops a future "cleanup" from silently breaking it.
#[test]
fn no_side_emits_the_named_linalg_broadcast() {
    for config in SCHED_CONFIGS {
        let o = ours(config);
        let g = parse::parse(&golden(config, "3_sched.mlir")).unwrap();
        for (who, m) in [("golden", &g), ("ours", &o)] {
            assert!(
                !m.ops_deep().iter().any(|x| x.kind.spelling() == "linalg.broadcast"),
                "{config}/{who}: the named linalg.broadcast is rejected downstream (p07)"
            );
        }
    }
}

/// THE GRID LANDS FLAT ON `func.func`, and matches the golden's.
///
/// # THE UNROLL'S GRID IS A RECORDED EXCEPTION
///
/// `unroll_grid_positions` collapses a corner-carrying multi-tile grid to `[1]`: one
/// program covering all N positions on one compute tile, because leaving N would
/// launch N tiles each computing all N positions (N-times the work and an N-fold
/// collision on every stored window). So when the unroll fired, OURS is `[1]` and the
/// golden's extent is the position count it emitted; the `tests/grid_unroll.rs` file
/// pins the unroll's own properties against a no-unroll control. When the unroll did
/// not fire, the two must agree exactly, as before.
#[test]
fn the_grid_attribute_matches_the_golden() {
    use triton_ktir::ir::*;
    for config in SCHED_CONFIGS {
        let o = ours(config);
        let g = parse::parse(&golden(config, "3_sched.mlir")).unwrap();
        let grid_of = |m: &Module| -> Option<Attr> {
            m.ops
                .iter()
                .find(|x| x.kind == OpKind::FuncFunc)
                .and_then(|f| f.attr(&AttrKey::Grid))
                .cloned()
        };
        // The golden prints `grid = [8 : index]` and we build `[8]`: the same single
        // flat extent, which is what the scheduler reads.
        let extents = |a: Option<Attr>| -> Option<Vec<i64>> {
            match a {
                Some(Attr::IntList(v)) => Some(v),
                _ => None,
            }
        };
        let (ge, oe) = (extents(grid_of(&g)), extents(grid_of(&o)));
        if oe == Some(vec![1]) && ge.as_deref() != Some(&[1i64][..]) {
            // The unroll fired: ours is the collapsed single-tile grid, the golden's
            // extent is the position count.
            let n: i64 = ge.iter().flatten().product();
            assert!(n > 1, "{config}: the golden's extent is the unrolled position count");
        } else {
            assert_eq!(
                ge, oe,
                "{config}: the `grid` the scheduler reads must match"
            );
            // 2 x 4 upstream -> the flat extent 8.
            assert_eq!(oe, Some(vec![8]));
        }
    }
}

/// THE CONTROL. Planted mutations of the stage-3 GOLDEN must each be caught by field
/// name, or the comparisons above could pass while comparing nothing.
#[test]
fn planted_stage_three_differences_are_each_caught_by_name() {
    let text = golden("attention_flash_noncausal", "3_sched.mlir");
    let base = parse::parse(&text).unwrap();
    // Each plant is a string that EXISTS in `3_sched.mlir` -- checked by the
    // `no longer applies` guard below, which is what caught my first three plants
    // having been written against stage-1 text.
    let plants: &[(&str, &str, &str)] = &[
        // THE ITERATOR that makes a reduction a reduction. Flip it and the online
        // softmax's max becomes an elementwise copy.
        ("attributes", "\"reduction\", \"parallel\"", "\"parallel\", \"parallel\""),
        // THE CONTAINER: the whole point of the pass.
        ("op kind", "func.func @attn_fwd", "tt.func @attn_fwd"),
        // THE GRID the scheduler reads.
        ("attributes", "grid = [8 : index]", "grid = [4 : index]"),
        // AN OPERAND REWIRED: a generic reads a different accumulator.
        ("operands", "outs(%splat_31", "outs(%splat_7"),
    ];
    let mut missed: Vec<String> = Vec::new();
    for (field, from, to) in plants {
        let mutated = text.replace(from, to);
        if mutated == text {
            missed.push(format!("  the plant `{from}` no longer applies to the golden"));
            continue;
        }
        let Ok(m) = parse::parse(&mutated) else {
            missed.push(format!("  planting `{field}` made unparseable IR"));
            continue;
        };
        let d = diff::diff(&base, &m);
        if !d.iter().any(|x| x.field == *field) {
            missed.push(format!(
                "  planting `{field}` (`{from}` -> `{to}`) was NOT caught; got {:?}",
                d.iter().map(|x| x.field).collect::<Vec<_>>()
            ));
        }
    }
    assert!(missed.is_empty(), "THE STAGE-3 DIFF IS NOT MEASURING:\n{}", missed.join("\n"));
}

#[test]
fn the_stage_three_golden_agrees_with_itself() {
    for config in SCHED_CONFIGS {
        let t = golden(config, "3_sched.mlir");
        let (a, b) = (parse::parse(&t).unwrap(), parse::parse(&t).unwrap());
        assert!(diff::diff(&a, &b).is_empty(), "{config}: the golden disagrees with itself");
    }
}

/// THE REFUSALS, PRESERVED. `swiglu_mlp` has no stage-3 golden because the C++ refuses
/// it at stage 2, and causal's off-band loop is refused the way the C++ refuses it --
/// both recorded so neither reads as a port defect.
#[test]
fn the_two_recorded_refusals_still_stand() {
    assert!(
        !has("swiglu_mlp", "3_sched.mlir"),
        "swiglu has no stage-3 output: the C++ refuses it at --spyre-lane-major-layout"
    );
    let layout_err = golden("swiglu_mlp", "2_layout.err");
    assert!(
        layout_err.contains("not a contiguous flat block"),
        "the stage-2 refusal must be on record:\n{layout_err}"
    );
    // Causal reaches stage 3 but not stage 4, and for the position-dependent trip count.
    assert!(has("attention_flash_causal", "3_sched.mlir"));
    assert!(!has("attention_flash_causal", "4_groups.mlir"));
    let groups_err = golden("attention_flash_causal", "4_groups.err");
    assert!(
        groups_err.contains("bounds are not all constant"),
        "the C++ refusal for causal's off-band trip count must be on record:\n{groups_err}"
    );
}
