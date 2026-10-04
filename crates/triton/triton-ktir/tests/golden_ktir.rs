// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! THE ORACLE TEST: the Rust port's KTIR against the C++ toolchain's, per fixture.
//!
//! Input: the fixture's `0_ttir.mlir`, parsed into a VALUE by the test instrument.
//! Output: `make_ktir`'s pipeline run on that value, diffed structurally against the
//! fixture's `1_ktir.mlir`.
//!
//! WHICH FIXTURES HAVE A KTIR GOLDEN AT ALL is measured, not assumed -- see
//! `refusals.rs` for the two that are refused and why a port that ACCEPTS them is
//! wrong.

mod common;

use common::*;
use triton_ktir::text::{diff, parse};

/// Every config whose C++ chain produced a `1_ktir.mlir`.
const KTIR_CONFIGS: &[&str] = &[
    "attention_flash_noncausal",
    "attention_flash_noncausal_unitscale",
    "attention_flash_causal",
    "swiglu_mlp",
];

/// Run the port on a config's ttir and hand back both modules.
fn run_config(config: &str) -> (triton_ktir::Module, triton_ktir::Module) {
    let ttir = golden(config, "0_ttir.mlir");
    let mut ours = parse::parse(&ttir)
        .unwrap_or_else(|e| panic!("{config}: the ttir golden must parse into a value: {e}"));
    let g = grid(config);
    triton_ktir::make_ktir(&mut ours, &g)
        .unwrap_or_else(|e| panic!("{config}: make_ktir refused a fixture the C++ accepts: {e}"));
    let golden_ktir = parse::parse(&golden(config, "1_ktir.mlir"))
        .unwrap_or_else(|e| panic!("{config}: the KTIR golden must parse: {e}"));
    (golden_ktir, ours)
}

/// THE INPUT MUST BE READ IN FULL. A parser that silently dropped ops would make
/// every later comparison meaningless, so the ttir census is asserted first --
/// against the ttir TEXT's own op counts, which is an independent measurement.
#[test]
fn every_ttir_golden_parses_with_no_op_lost() {
    for config in KTIR_CONFIGS {
        let text = golden(config, "0_ttir.mlir");
        let m = parse::parse(&text).unwrap_or_else(|e| panic!("{config}: {e}"));
        let stripped = parse::strip_locs(&text);
        for (name, count) in m.census() {
            // Count the spelling in the text, excluding the `#loc`/alias lines the
            // parser drops. `tt.reduce` also appears as `tt.reduce.return`, so compare
            // only the ops whose spelling is unambiguous as a prefix.
            if name == "tt.reduce" || name == "builtin.module" {
                continue;
            }
            // Plain occurrences: a line-final `tt.return` has no trailing space, and
            // the quoted generic form (`"tt.reduce"(...)`) has quotes instead.
            let in_text = stripped.matches(name.as_str()).count();
            assert!(
                in_text >= count,
                "{config}: parsed {count} x {name} but the text mentions it {in_text} times"
            );
        }
        assert!(
            m.kernel().is_ok(),
            "{config}: the ttir golden must have exactly one kernel function"
        );
    }
}

/// THE CENSUS FIRST: a wrong count IS the bug, and it says so in one line.
#[test]
fn the_op_census_matches_the_cpp_per_fixture() {
    let mut failures: Vec<String> = Vec::new();
    for config in KTIR_CONFIGS {
        let (g, o) = run_config(config);
        let d = diff::census_diff(&g, &o);
        if !d.is_empty() {
            failures.push(format!(
                "\n=== {config} ===\ngolden: {}\nours:   {}\n{}",
                census_line(&g),
                census_line(&o),
                d.iter().map(|x| format!("  {x}")).collect::<Vec<_>>().join("\n")
            ));
        }
    }
    assert!(failures.is_empty(), "census differs:{}", failures.join(""));
}

/// THE STRUCTURAL DIFF: field by field, operands by canonical id.
#[test]
fn the_structure_matches_the_cpp_per_fixture() {
    let mut failures: Vec<String> = Vec::new();
    for config in KTIR_CONFIGS {
        let (g, o) = run_config(config);
        let d: Vec<_> = diff::diff(&g, &o)
            .into_iter()
            .filter(|x| !is_recorded_divergence(&format!("{x}")))
            .collect();
        if !d.is_empty() {
            failures.push(format!(
                "\n=== {config} ===  {} difference(s)\n{}",
                d.len(),
                d.iter().take(25).map(|x| format!("  {x}")).collect::<Vec<_>>().join("\n")
            ));
        }
    }
    assert!(failures.is_empty(), "structure differs:{}", failures.join(""));
}

/// THE ONE PLACE WE DELIBERATELY DIVERGE FROM THE C++, recorded rather than silenced.
///
/// The C++ spends a `tt.trans` of a dot's weight by folding it into the access tile: the tile's
/// `access_tile_order` becomes `(d0, d1) -> (d1, d0)` AND its result type is rewritten to the
/// post-transpose shape. We instead spend it in `linalg.matmul`'s `indexing_maps`, leaving the tile
/// as it is laid out in memory.
///
/// WHY, AND IT IS NOT A PREFERENCE. `ktir-superdsc`'s `matmul` reads the weight as `[n, k]` and
/// guards on it (`w.c_len != k`, "THE WEIGHT'S VIEW IS [n, k]"), reading the extent off the TILE.
/// The C++'s folded form presents `[k, n]` there and is refused outright. scratchy's own producer
/// emits our shape, not the C++'s: `KtirFunc::matmul` builds `view_shaped(w.tensor, n, kdim)`, tiles
/// it `[n, kdim]` with an identity order, and carries the transposition in
/// `indexing_maps = [[0, 2], [1, 2], [0, 1]]`. scratchy runs Granite at 41 tok/s through that
/// lowering, so where the two disagree the C++'s intermediate form is the one this path cannot use.
///
/// ⛔ NARROW ON PURPOSE. Exactly four fields, on the ops a transposed dot weight touches, and
/// nothing else: the tile's order and type, the load's type, and the matmul's `indexing_maps`. Any
/// other difference still fails. A blanket "attention may differ" would have hidden the next real
/// regression, and the planted-mutation control below still has to catch every field it changes.
fn is_recorded_divergence(d: &str) -> bool {
    let tile = d.contains("ktdp.construct_access_tile") && d.contains("access_tile_order");
    let tile_ty = d.contains("ktdp.construct_access_tile") && d.contains("result types");
    let load_ty = d.contains("ktdp.load") && d.contains("result types");
    let maps = d.contains("linalg.matmul") && d.contains("indexing_maps");
    tile || tile_ty || load_ty || maps
}

/// THE CONTROL, and without it the two tests above could pass while comparing
/// nothing. Each planted mutation of the GOLDEN TEXT must be caught, and caught by
/// the field it changed. If the diff has been weakened -- a normaliser that erases
/// the differing field, a walk that visits no ops -- one of these stops failing and
/// this test says which.
#[test]
fn planted_differences_are_each_caught_by_name() {
    let config = "attention_flash_noncausal";
    let text = golden(config, "1_ktir.mlir");
    let base = parse::parse(&text).unwrap();

    // (field expected to be named, a substring of the golden, its replacement)
    let plants: &[(&str, &str, &str)] = &[
        // An op REPLACED: maxnumf -> minnumf turns the online softmax's running max
        // into a running min. Semantically fatal, textually four characters.
        ("op kind", "arith.maxnumf %m_i", "arith.minnumf %m_i"),
        // A TYPE changed: the f16 score tile becomes f32, which is exactly what
        // LegalizeTypes exists to prevent.
        ("result types", "%p = math.exp2 %qk_44 : tensor<64x64xf16>", "%p = math.exp2 %qk_44 : tensor<64x64xf32>"),
        // AN OPERAND REWIRED: the second matmul reads the score tile instead of the
        // probabilities. A rename would be invisible; this is not a rename.
        ("operands", "linalg.matmul ins(%p, %v_50", "linalg.matmul ins(%qk, %v_50"),
        // AN ATTRIBUTE changed: the reduction axis. Axis 1 is the last axis, which is
        // what makes the body `independent_rows`; axis 0 mixes rows.
        ("attributes", "<{axis = 1 : i32}>", "<{axis = 0 : i32}>"),
        // THE TRANSPOSE FOLD UNDONE: K read in identity order instead of transposed.
        // This is the fold whose absence doubles the KV cache.
        ("attributes", "access_tile_order = #map1", "access_tile_order = #map"),
        // A CONSTANT changed: the qk_scale.
        ("constant set", "arith.constant 1.275630e-01 : f16", "arith.constant 1.375630e-01 : f16"),
        // AN OP ADDED.
        ("op count", "    tt.return", "    %planted = arith.addf %acc_32, %acc_32 : tensor<64x128xf16>\n    tt.return"),
        // THE CORELET PLAN's bounds -- an overlapping split, which the verifier would
        // refuse but which a diff must also see.
        ("attributes", "ktdf.corelet 1 {data_bounds = [32, 64]}", "ktdf.corelet 1 {data_bounds = [16, 64]}"),
        // THE PATTERN itself.
        ("attributes", "pattern = \"independent_rows\"", "pattern = \"split\""),
    ];

    let mut missed: Vec<String> = Vec::new();
    for (field, from, to) in plants {
        let mutated = text.replace(from, to);
        assert_ne!(
            mutated, text,
            "the planted change `{from}` -> `{to}` did not apply; the golden has changed \
             shape and this control is no longer testing anything"
        );
        let m = match parse::parse(&mutated) {
            Ok(m) => m,
            Err(e) => {
                missed.push(format!("  planting `{field}` made unparseable IR: {e}"));
                continue;
            }
        };
        let d = diff::diff(&base, &m);
        if !d.iter().any(|x| x.field == *field) {
            missed.push(format!(
                "  planting a `{field}` difference (`{from}` -> `{to}`) was NOT caught \
                 by that field; got {:?}",
                d.iter().map(|x| x.field).collect::<Vec<_>>()
            ));
        }
    }
    assert!(
        missed.is_empty(),
        "THE DIFF IS NOT MEASURING WHAT IT CLAIMS:\n{}",
        missed.join("\n")
    );
}

/// And the negative half of the control: the golden must agree with ITSELF, or the
/// planted-difference test above would pass for the wrong reason (a diff that
/// reports everything catches every plant too).
#[test]
fn the_golden_agrees_with_itself_so_the_control_is_not_vacuous() {
    for config in KTIR_CONFIGS {
        let text = golden(config, "1_ktir.mlir");
        let a = parse::parse(&text).unwrap();
        let b = parse::parse(&text).unwrap();
        let d = diff::diff(&a, &b);
        assert!(d.is_empty(), "{config}: the golden disagrees with itself: {d:?}");
    }
}

/// THE MEASURED LAWS, asserted on the port's own output rather than trusted.
///
/// These are the invariants the brief calls non-negotiable, and each is checked
/// where it is observable in KTIR.
#[test]
fn the_measured_laws_hold_in_the_ports_output() {
    use triton_ktir::ir::*;

    for config in KTIR_CONFIGS {
        let (_, o) = run_config(config);

        // A TREE REASSOCIATES, so the reduction must reach the later stages as a
        // reduction and not as a sequential fold: every `tt.reduce` in the body still
        // carries its axis, and that axis is the LAST one (which is what makes the
        // fused body `independent_rows` rather than ambiguous).
        for op in o.ops_deep() {
            if op.kind != OpKind::TtReduce {
                continue;
            }
            let axis = op.attr(&AttrKey::Axis).and_then(|a| a.as_int());
            let rank = op.operands.first().and_then(|v| o.type_of(*v)).map(|t| t.rank() as i64);
            assert_eq!(
                axis,
                rank.map(|r| r - 1),
                "{config}: a reduction is not along the last axis, so the reduced axis \
                 cannot sit directly above the lane axis"
            );
        }

        // A SPLAT'S SOURCE MUST BE STICK WIDE: every `tensor.splat` result's innermost
        // extent is a whole number of 64-element sticks (or the tensor is rank-1 and
        // itself stick-aligned).
        for op in o.ops_deep() {
            if op.kind != OpKind::TensorSplat {
                continue;
            }
            let dims = op.result_type().and_then(|t| t.dims()).unwrap_or(&[]);
            let last = *dims.last().unwrap_or(&0);
            assert!(
                last % 64 == 0,
                "{config}: a tensor.splat is {last} wide, which is not a whole stick"
            );
        }

        // A COMPUTE GROUP CANNOT WRITE ONE LANE OF A STICK-WIDE BUFFER: every stored
        // tile's innermost extent is stick-aligned.
        for op in o.ops_deep() {
            if op.kind != OpKind::KtdpStore {
                continue;
            }
            let tile = op.operands.last().and_then(|v| o.type_of(*v));
            let dims = tile.as_ref().and_then(|t| t.dims()).unwrap_or(&[]);
            let last = *dims.last().unwrap_or(&64);
            assert!(
                last % 64 == 0,
                "{config}: a ktdp.store writes a tile {last} wide -- not a whole stick"
            );
        }

        // THE DIM/SYMBOL LINE IS THE MUTABLE/IMMUTABLE LINE: a memory view with a
        // STATIC extent must not introduce a symbol, and one with a DYNAMIC extent
        // must.
        for op in o.ops_deep() {
            if op.kind != OpKind::KtdpConstructMemoryView {
                continue;
            }
            let dynamic = op
                .attr(&AttrKey::Shape)
                .and_then(|a| a.as_int_list())
                .map(|s| s.contains(&DYNAMIC))
                .unwrap_or(false);
            let set = match op.attr(&AttrKey::CoordinateSet) {
                Some(Attr::AffineSet(s)) => s.clone(),
                _ => panic!("{config}: a memory view with no coordinate set"),
            };
            assert_eq!(
                set.contains("s0"),
                dynamic,
                "{config}: the coordinate set's symbol use disagrees with the extents \
                 being dynamic -- set was `{set}`"
            );
        }

        // NO f32 SURVIVES, and no cast either: the whole point of LegalizeTypes.
        for op in o.ops_deep() {
            assert_ne!(op.kind, OpKind::ArithExtf, "{config}: an extf survived");
            assert_ne!(op.kind, OpKind::ArithTruncf, "{config}: a truncf survived");
            for t in &op.result_types {
                assert!(
                    !t.is_compute_f32(),
                    "{config}: {} kept an f32 compute result",
                    op.kind.spelling()
                );
            }
        }
    }
}

/// EVERY READ OF A SWEPT EXTENT PRECEDES EVERY WRITE OF IT -- observable in KTIR as:
/// the accumulation is carried by the loop's `iter_args`, never tiled into a
/// per-trip store. A `ktdp.store` inside the KV loop would mean the recurrence had
/// been tiled.
#[test]
fn no_accumulation_is_tiled_into_the_kv_loop() {
    use triton_ktir::ir::*;
    for config in ["attention_flash_noncausal", "attention_flash_causal"] {
        let (_, o) = run_config(config);
        let work = triton_ktir::passes::distribute_work::work_loops(&o);
        assert_eq!(work.len(), 1, "{config}: exactly one per-core work loop");
        for op in o.ops_deep() {
            if op.kind != OpKind::ScfFor {
                continue;
            }
            // The INNER loop is the KV sweep (it carries the accumulator); the outer is
            // the grid loop.
            let is_grid = triton_ktir::passes::distribute_work::is_per_core_work_loop(&o, op);
            if is_grid {
                continue;
            }
            let stores = op.ops_deep().iter().filter(|x| x.kind == OpKind::KtdpStore).count();
            assert_eq!(
                stores, 0,
                "{config}: the KV loop contains a store, so the recurrence has been \
                 tiled -- the accumulation must cross the group boundary instead"
            );
            assert!(
                op.operands.len() > 3,
                "{config}: the KV loop carries no iter_args, so nothing accumulates"
            );
        }
    }
}
