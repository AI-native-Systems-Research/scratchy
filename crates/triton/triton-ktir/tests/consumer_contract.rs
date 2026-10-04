// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! CAN scratchy's `ktir_to_superdsc` WALK WHAT WE EMIT? Measured per fixture.
//!
//! The KTIR -> SuperDSC leg is theirs now, so "our KTIR matches the C++'s KTIR" is no
//! longer the finishing condition -- "their walker can consume it" is. This file is
//! that measurement, and it is a REGRESSION GATE in the direction that matters: the
//! per-fixture gap counts are pinned, so the day a pass lands that closes one, this
//! test fails and forces the count to be updated with evidence rather than the
//! closure being assumed.
//!
//! See `src/consumer.rs` for where each gap's premise is read out of their tree.

mod common;

use common::*;
use triton_ktir::consumer;
use triton_ktir::text::parse;

const CONFIGS: &[&str] = &[
    "attention_flash_noncausal",
    "attention_flash_noncausal_unitscale",
    "attention_flash_causal",
    "swiglu_mlp",
];

fn ours(config: &str) -> triton_ktir::Module {
    let mut m = parse::parse(&golden(config, "0_ttir.mlir"))
        .unwrap_or_else(|e| panic!("{config}: ttir must parse: {e}"));
    triton_ktir::make_ktir(&mut m, &grid(config))
        .unwrap_or_else(|e| panic!("{config}: make_ktir refused: {e}"));
    m
}

/// THE HEADLINE: our `make_ktir` output is NOT consumable by their walker, and the
/// reason is gap 1 -- `tt.*` ops have no variant in `ktir_core::OpKind` at all.
///
/// This is asserted rather than merely reported because it reorders the remaining
/// work: `ToSchedulerKTIR` is the pass that removes those ops, so it is a
/// PREREQUISITE for emitting `ktir_core` values, not an independent item to be
/// scheduled after "wire up the boundary".
#[test]
fn stage_one_ktir_is_not_yet_consumable_and_gap_one_is_why() {
    for config in CONFIGS {
        let m = ours(config);
        let gaps = consumer::summary(&m);
        assert!(
            !gaps.is_empty(),
            "{config}: if this is now empty, ToSchedulerKTIR has landed and this test \
             should become the opposite assertion"
        );
        let tt = gaps
            .iter()
            .find(|(g, _)| g == "gap1-tt-not-in-vocabulary")
            .map(|(_, n)| *n)
            .unwrap_or(0);
        assert!(
            tt > 0,
            "{config}: gap 1 must be the blocker -- got {gaps:?}"
        );
    }
}

/// THE PER-FIXTURE COUNTS, pinned. Printed on failure so the new numbers are in the
/// output and can be pasted in with the commit that changed them.
#[test]
fn the_gap_counts_per_fixture_are_what_they_are() {
    // MEASURED, not predicted -- my first attempt at this table was wrong in four
    // places and two of those were bugs in the CHECK rather than real gaps (see
    // `consumer.rs`'s notes on `WALKED_OPS` and on `get_compute_tile_id`). Regenerate
    // with `cargo run --offline --example brk`.
    let expected: &[(&str, &str, usize)] = &[
        ("attention_flash_noncausal", "gap1-grid-loop-not-folded", 1),
        ("attention_flash_noncausal", "gap1-not-walked", 13),
        ("attention_flash_noncausal", "gap1-regions-have-no-block-arguments", 4),
        ("attention_flash_noncausal", "gap1-tt-not-in-vocabulary", 12),
        ("attention_flash_noncausal", "gap2-reduce-combiner-in-a-region", 2),
        ("attention_flash_noncausal", "gap3-loop-carried-count", 1),
        ("attention_flash_noncausal_unitscale", "gap1-grid-loop-not-folded", 1),
        ("attention_flash_noncausal_unitscale", "gap1-not-walked", 13),
        ("attention_flash_noncausal_unitscale", "gap1-regions-have-no-block-arguments", 4),
        ("attention_flash_noncausal_unitscale", "gap1-tt-not-in-vocabulary", 12),
        ("attention_flash_noncausal_unitscale", "gap2-reduce-combiner-in-a-region", 2),
        ("attention_flash_noncausal_unitscale", "gap3-loop-carried-count", 1),
        ("attention_flash_causal", "gap1-grid-loop-not-folded", 1),
        ("attention_flash_causal", "gap1-not-walked", 16),
        ("attention_flash_causal", "gap1-regions-have-no-block-arguments", 7),
        ("attention_flash_causal", "gap1-tt-not-in-vocabulary", 20),
        ("attention_flash_causal", "gap2-reduce-combiner-in-a-region", 4),
        ("attention_flash_causal", "gap3-loop-carried-count", 2),
        // THE POSITION-DEPENDENT TRIP COUNT, and the C++ refuses the same loop.
        ("attention_flash_causal", "gap3-non-constant-bound", 3),
        // swiglu is the CLOSEST to consumable: no reduction, no transposing tile, and
        // its K-loop carries exactly ONE accumulator with constant bounds.
        ("swiglu_mlp", "gap1-grid-loop-not-folded", 1),
        ("swiglu_mlp", "gap1-not-walked", 14),
        ("swiglu_mlp", "gap1-regions-have-no-block-arguments", 2),
        ("swiglu_mlp", "gap1-tt-not-in-vocabulary", 2),
    ];

    let mut actual: Vec<(String, String, usize)> = Vec::new();
    for config in CONFIGS {
        let m = ours(config);
        for (gap, n) in consumer::summary(&m) {
            actual.push((config.to_string(), gap, n));
        }
    }

    let mut wrong: Vec<String> = Vec::new();
    for (cfg, gap, want) in expected {
        let got = actual
            .iter()
            .find(|(c, g, _)| c == cfg && g == gap)
            .map(|(_, _, n)| *n)
            .unwrap_or(0);
        if got != *want {
            wrong.push(format!("  {cfg} / {gap}: expected {want}, got {got}"));
        }
    }
    if !wrong.is_empty() {
        let all: Vec<String> =
            actual.iter().map(|(c, g, n)| format!("        (\"{c}\", \"{g}\", {n}),")).collect();
        panic!(
            "the consumer gaps changed. If a pass closed one, that is PROGRESS -- paste \
             the new table in with the commit:\n{}\n\n    THE ACTUAL TABLE:\n{}",
            wrong.join("\n"),
            all.join("\n")
        );
    }
}

/// SWIGLU IS THE CLOSEST TO CONSUMABLE, and saying which fixture is nearest is how
/// the next slice gets chosen by evidence rather than by taste.
///
/// Its only gaps are the `tt.func`/`tt.return` wrapper -- which becomes their
/// `IRFunction` + `func.return` rather than needing a pass -- so it clears gaps 2, 3
/// and 4 outright: no reduction, no transposing tile, and a K-loop with constant
/// bounds carrying exactly ONE accumulator.
#[test]
fn swiglu_clears_the_reduce_loop_and_transpose_gaps_outright() {
    let m = ours("swiglu_mlp");
    let found = consumer::check(&m);
    for gap in [
        "gap2-reduce-combiner-in-a-region",
        "gap3-loop-carried-count",
        "gap3-non-constant-bound",
        "gap4-transpose-dropped-silently",
    ] {
        assert!(
            !found.iter().any(|i| i.gap == gap),
            "swiglu should clear {gap}: {:?}",
            found.iter().filter(|i| i.gap == gap).collect::<Vec<_>>()
        );
    }
    // What remains for swiglu is ALL gap 1 -- the function wrapper, the pointer
    // casts, `arith.index_cast`, the KTDF plan and the unfolded grid loop. Every one
    // of those is `ToSchedulerKTIR`/`CarriedValuesToMemory` territory or dissolves
    // into their `IRFunction` boundary; NONE needs a change on their side. That is
    // what makes swiglu the shortest path to their walker, and it is worth asserting
    // rather than eyeballing.
    let gaps: std::collections::BTreeSet<&str> = found.iter().map(|i| i.gap).collect();
    assert!(
        gaps.iter().all(|g| g.starts_with("gap1-")),
        "swiglu's remaining gaps should all be gap 1: {gaps:?}"
    );
}

/// CAUSAL'S OFF-BAND TRIP COUNT IS A HOLE IN BOTH IMPLEMENTATIONS, and this test is
/// the record of that -- with the C++'s OWN refusal as the corroborating evidence, so
/// it cannot be mistaken for a defect this port introduced.
#[test]
fn the_causal_off_band_trip_count_is_refused_by_the_cpp_too() {
    let m = ours("attention_flash_causal");
    let found = consumer::check(&m);
    let nonconst: Vec<&consumer::Incompatibility> =
        found.iter().filter(|i| i.gap == "gap3-non-constant-bound").collect();
    assert!(
        !nonconst.is_empty(),
        "the position-dependent trip count must be reported"
    );
    // The C++ chain refuses the same loop at stage 4. That golden is the corroboration.
    let cpp = golden("attention_flash_causal", "4_groups.err");
    assert!(
        cpp.contains("bounds are not all constant"),
        "the C++ refusal must be on record beside ours, or this reads as our bug:\n{cpp}"
    );
    assert!(
        !has("attention_flash_causal", "4_groups.mlir"),
        "and the C++ produced no stage-4 output for causal"
    );
}

/// AFTER `ToSchedulerKTIR`, gap 1 IS CLOSED -- and gap 6 is what remains.
///
/// This is the state that matters now: the Triton-free module we would actually hand
/// over. Every `tt.*` op is gone, so nothing is unrepresentable any more; what stops the
/// handover is that their walker has no `linalg.generic` arm, which is the form both the
/// reduction and the broadcast take. See `consumer.rs`'s gap 6 for the three ways out.
#[test]
fn after_the_conversion_gap_one_closes_and_gap_six_is_the_blocker() {
    for config in CONFIGS {
        let mut m = parse::parse(&golden(config, "0_ttir.mlir")).unwrap();
        let g = grid(config);
        triton_ktir::make_ktir(&mut m, &g).unwrap();
        triton_ktir::passes::to_ktir::run(&mut m, &g).unwrap();

        let gaps = consumer::summary(&m);
        let get = |name: &str| {
            gaps.iter().find(|(x, _)| x == name).map(|(_, n)| *n).unwrap_or(0)
        };
        // GAP 1 IS CLOSED: nothing Triton, and no region block arguments outside the
        // function (their regions are bare op lists).
        assert_eq!(
            get("gap1-tt-not-in-vocabulary"),
            0,
            "{config}: ToSchedulerKTIR must leave nothing unrepresentable: {gaps:?}"
        );
        assert_eq!(get("gap1-grid-loop-not-folded"), 0, "{config}: the grid loop is folded");
        // AND GAP 6 IS WHAT IS LEFT -- for the fixtures that HAVE a reduction or a
        // broadcast, which is the three attention configurations. swiglu has neither, so
        // it emits no `linalg.generic` at all and clears gap 6 outright: after this pass
        // its whole remainder is the mechanical part of building their values (block
        // arguments becoming `IterVar`/`IterArgs`, and the `arith.index_cast` chain
        // GridIndexChains rebuilds). That makes swiglu the one fixture whose handover is
        // blocked by nothing needing an owner decision.
        if *config == "swiglu_mlp" {
            assert_eq!(
                get("gap6-linalg-generic-has-no-walker-arm"),
                0,
                "swiglu has no reduction and no broadcast, so no linalg.generic: {gaps:?}"
            );
            assert_eq!(
                get("gap2-reduce-combiner-in-a-region"),
                0,
                "and no reduce combiner either: {gaps:?}"
            );
        } else {
            assert!(
                get("gap6-linalg-generic-has-no-walker-arm") > 0,
                "{config}: gap 6 must be the blocker after the conversion: {gaps:?}"
            );
        }
    }
}

/// THE RUNTIME-BROADCAST ANSWER, pinned.
///
/// A reduced value reaching the lanes again is `linalg.broadcast` + `tensor.empty` in
/// scratchy's producer -- NOT `tensor.splat`, which is scalar-and-constant-only. And
/// their walker has no arm for either, so their own RMSNorm is not yet walkable by their
/// own lowering.
///
/// This is a fact about THEIR tree, so the test guards the two lists rather than running
/// anything: if `linalg.broadcast` ever appears in `WALKED_OPS`, the arm landed and gap 6
/// has moved.
#[test]
fn the_runtime_broadcast_form_is_the_named_op_and_is_not_yet_walkable() {
    for op in consumer::THEIR_PRODUCER_EMITS_BUT_WALKER_LACKS {
        assert!(
            !consumer::WALKED_OPS.contains(op),
            "`{op}` is now walked -- their walker grew an arm and gap 6 has moved; \
             re-measure rather than trusting this list"
        );
    }
    // And the form is NOT tensor.splat, which their walker DOES handle but only for a
    // constant scalar fill.
    assert!(consumer::WALKED_OPS.contains(&"tensor.splat"));
    assert!(!consumer::THEIR_PRODUCER_EMITS_BUT_WALKER_LACKS.contains(&"tensor.splat"));
}
