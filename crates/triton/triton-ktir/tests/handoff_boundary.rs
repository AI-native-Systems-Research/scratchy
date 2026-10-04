// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! WHAT `ktir_superdsc::emit::lower_ktir_to_superdsc::regions` READS OFF WHAT WE HAND OVER.
//!
//! That reader is the very first thing past [`triton_ktir::handoff::lower`], and its
//! contract is narrow and unnegotiable: for parameter `i` it takes the ONE
//! `ktdp.construct_memory_view` whose address operand is that parameter, found by a
//! NON-RECURSIVE `find` over `IRFunction::operations` -- the function's TOP-LEVEL op list.
//! A parameter with no top-level view is a parameter whose buffer width nothing states, and
//! it refuses.
//!
//! So "every parameter declares its width at the top level" is a property of the program WE
//! EMIT, and this file measures it directly rather than through their crate -- an assertion
//! here names the pass that broke it, where the same failure seen through `drive` names only
//! the reader.
//!
//! MEASURED, on the tree before the commit that added this file: `rope_q32` at grid 4,32
//! declared ZERO of its four parameters, and `attention_flash_noncausal` at grid 4,4 declared
//! four of five. The two have OPPOSITE causes and each has its own control here.

use ktir_core::arena::Arena;
use ktir_core::ir::{IRFunction, Ssa};
use ktir_core::opkind::OpKind;
use std::path::PathBuf;

/// One experiment-1 ttir golden. These are the twelve Granite configurations the
/// `triton-ktir-superdsc` driver runs; `tests/goldens/ktir` holds a different, smaller set
/// and neither of the two fixtures this file is about is in it.
///
/// THE ROPE TTIR GOLDENS ARE THE HEAD-MAJOR KERNEL, AND EVERY ROPE NUMBER BELOW IS THAT
/// PROGRAM'S. `test/fixtures/rope.py` deltas 7 and 8 made the fixture token-major with
/// `[N_TOK * H, HEAD_DIM]` tables, launched at grid `[256]`; `rope_q32.ttir.mlir` here is still
/// `offs_m = off_h * N_TOK + start_m * BLOCK_M` at grid `[4, 32]`. So "rope_q32 is 128 work items
/// over 32 cores, 4 trips, 24 windows, 8 stores" describes THIS FILE'S INPUT and no longer the
/// shipped fixture, which is 256 items over 32 cores at 8 trips.
///
/// THAT IS STILL A VALID TEST OF WHAT THIS FILE MEASURES -- the handoff's unroll machinery, for
/// which a 4-trip loop is a perfectly good case, and the golden is a FROZEN input rather than a
/// claim about the fixture. It is recorded because a reader would otherwise take the numbers for
/// the current kernel's. Regenerating these two goldens needs the experiment-1 harness
/// (`test/experiment1/run_experiment1.py`, real Triton + torch) and `test/experiment1/**` is not
/// this crate's to write; it is the follow-up that retires this note.
fn ttir(config: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.join("test-experiment1/ktir").join(format!("{config}.ttir.mlir")))
        .expect("the crate sits one level under crates/triton");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("MISSING TTIR {}: {e}", path.display()))
}

/// The module the handoff sees: parsed, `make_ktir`, `to_ktir`. Panics rather than returning
/// an error, because a fixture that cannot reach the handoff is measuring nothing.
fn converted(config: &str, grid: &[i64]) -> triton_ktir::Module {
    let mut m = triton_ktir::text::parse::parse(&ttir(config))
        .unwrap_or_else(|e| panic!("{config}: ttir must parse: {e}"));
    triton_ktir::make_ktir(&mut m, grid)
        .unwrap_or_else(|e| panic!("{config}: make_ktir refused: {e}"));
    triton_ktir::passes::to_ktir::run(&mut m, grid)
        .unwrap_or_else(|e| panic!("{config}: to_ktir refused: {e}"));
    m
}

/// The twelve configurations `triton-ktir-superdsc`'s driver runs, with the grid each is
/// launched at -- the same values `test/experiment1/index.json` states.
const ALL_TWELVE: &[(&str, &[i64])] = &[
    ("attention_flash_noncausal", &[4, 4]),
    ("attention_flash_causal", &[4, 4]),
    ("swiglu_mlp_small", &[1]),
    ("swiglu_mlp_granite", &[1]),
    ("swiglu_mlp_tiled_k", &[1]),
    ("embedding_granite_bm128_control", &[2]),
    ("embedding_granite", &[4]),
    ("rmsnorm_granite", &[1]),
    ("rope_q32", &[4, 32]),
    ("rope_kv8", &[4, 8]),
    ("decoder_layer_one", &[1]),
    ("decoder_two_layers", &[1]),
];

/// `regions()`'s own search, reproduced: parameter -> its top-level view's `shape`, or
/// `None` when nothing at the top level addresses it.
fn declared_widths(f: &IRFunction<'_>) -> Vec<Option<Vec<i64>>> {
    f.arguments
        .iter()
        .map(|(ptr, _)| view_of(f.operations, *ptr))
        .collect()
}

/// How many `ktdp.construct_access_tile` windows sit inside a PER-CORE WORK LOOP's body.
///
/// The work loop is recognised the way `distribute_work::is_per_core_work_loop` recognises it and
/// the way their `regions()` would have to: its lower bound (operand 0) is the value a
/// `ktdp.get_compute_tile_id` defines. That is the landmark contract, and it is what separates a
/// loop OUR passes minted from a `tl.range` the kernel source wrote.
fn windows_under_work_loop(f: &IRFunction<'_>) -> usize {
    let landmark: Vec<Ssa> = f
        .ops_deep()
        .into_iter()
        .filter(|o| o.op_type == OpKind::KtdpGetComputeTileId)
        .filter_map(|o| o.result)
        .collect();
    fn count(
        ops: &[ktir_core::ir::Operation<'_>],
        landmark: &[Ssa],
        inside: bool,
        n: &mut usize,
    ) {
        for o in ops {
            if inside && o.op_type == OpKind::KtdpConstructAccessTile {
                *n += 1;
            }
            let is_work = o.op_type == OpKind::ScfFor
                && o.operands.first().is_some_and(|v| landmark.contains(v));
            for r in o.regions {
                count(r, landmark, inside || is_work, n);
            }
        }
    }
    let mut n = 0;
    count(f.operations, &landmark, false, &mut n);
    n
}

/// `param_tiles` / `param_tiles_deep`'s own search, reproduced: for parameter `ptr`, how many
/// `ktdp.construct_access_tile` WINDOWS over its view(s) are visible at the function's TOP
/// LEVEL, and how many exist at all.
///
/// Their `regions()` refuses when the two disagree, and it is `deep > top` rather than
/// `top == 0` -- a parameter windowed above a loop AND again inside it has a top-level tile and
/// still hides windows from every reader there. Mirrored here so a failure names the pass that
/// buried them instead of naming their reader.
fn window_counts(f: &IRFunction<'_>) -> Vec<(usize, usize)> {
    fn views_over(ops: &[&ktir_core::ir::Operation<'_>], ptr: Ssa) -> Vec<Ssa> {
        ops.iter()
            .filter(|o| {
                o.op_type == OpKind::KtdpConstructMemoryView && o.operands.first() == Some(&ptr)
            })
            .filter_map(|o| o.result)
            .collect()
    }
    fn tiles_over(ops: &[&ktir_core::ir::Operation<'_>], views: &[Ssa]) -> usize {
        ops.iter()
            .filter(|o| {
                o.op_type == OpKind::KtdpConstructAccessTile
                    && o.operands.first().is_some_and(|v| views.contains(v))
            })
            .count()
    }
    let top: Vec<&ktir_core::ir::Operation<'_>> = f.operations.iter().collect();
    let deep: Vec<&ktir_core::ir::Operation<'_>> = f.ops_deep();
    f.arguments
        .iter()
        .map(|(ptr, _)| {
            (
                tiles_over(&top, &views_over(&top, *ptr)),
                tiles_over(&deep, &views_over(&deep, *ptr)),
            )
        })
        .collect()
}

fn view_of(ops: &[ktir_core::ir::Operation<'_>], ptr: Ssa) -> Option<Vec<i64>> {
    ops.iter()
        .find(|o| o.op_type == OpKind::KtdpConstructMemoryView && o.operands.first() == Some(&ptr))
        .and_then(|o| {
            o.attributes.iter().find_map(|(k, v)| match (k, v) {
                (ktir_core::attrkey::AttrKey::Shape, ktir_core::ir::Attr::IntList(s)) => {
                    Some(s.to_vec())
                }
                _ => None,
            })
        })
}

//===----------------------------------------------------------------------===//
// Defect A: the work loop is not where a buffer's width is declared
//===----------------------------------------------------------------------===//

/// THE SAME KERNEL AT TWO GRIDS DECLARES THE SAME BUFFERS.
///
/// `rope` at grid 4,8 is 32 work items over 32 compute tiles, so `to_ktir`'s
/// `fold_grid_work_loop` removes the work loop and the body -- views included -- is
/// straight-line. At grid 4,32 it is 128 work items over 32 tiles, the loop is genuinely
/// multi-trip and STAYS, and `distribute_work` had swept every op after the program-id
/// anchor into it.
///
/// A view's only operand is a function parameter and its shape/strides/space are constant
/// attributes, so it is loop-invariant BY CONSTRUCTION: which side of the work loop it sits
/// on is not a fact about the program. The trip count must therefore not change which
/// parameters state a width, and it must not change WHAT width they state -- the second half
/// is the control that a hoist did not mint a synthetic view of invented extent.
#[test]
fn the_trip_count_does_not_change_which_buffers_state_a_width() {
    let a = Arena::global();
    // desc_x, desc_cos, desc_sin, desc_o. The x/o extent is heads*seq: 32*256 = 8192 rows at
    // q32, 8*256 = 2048 at kv8. cos/sin are shared across heads, so [256, 64] at both.
    let cases: &[(&str, &[i64], &[&[i64]])] = &[
        ("rope_kv8", &[4, 8], &[&[2048, 128], &[256, 64], &[256, 64], &[2048, 128]]),
        ("rope_q32", &[4, 32], &[&[8192, 128], &[256, 64], &[256, 64], &[8192, 128]]),
    ];
    for (config, grid, want) in cases {
        let m = converted(config, grid);
        let f = triton_ktir::passes::to_ktir_emit::lower(&m, a)
            .unwrap_or_else(|e| panic!("{config}: the handoff refused: {e}"));
        let got = declared_widths(&f);
        let want: Vec<Option<Vec<i64>>> = want.iter().map(|s| Some(s.to_vec())).collect();
        assert_eq!(
            got, want,
            "{config} at grid {grid:?}: every parameter must state its width at the TOP LEVEL, \
             which is the only place their `regions()` looks. `None` here is a parameter whose \
             view is buried in the work loop's region."
        );
    }
}

/// THE WORK LOOP IS GONE, AND NOT BY DROPPING WORK.
///
/// The control for the width test above, in the direction that matters. `to_ktir` has two ways
/// to leave no `scf.for` behind: FOLD it, which is exact only when it can run at most one trip,
/// or UNROLL it, which is exact whenever the trip count divides evenly. rope_q32 at grid 4,32 is
/// 128 work items over 32 cores -- folding it would drop 96 of them -- so the loop it hands over
/// must be gone by the SECOND route, and the body must appear once per trip.
///
/// This asserts the shape; `the_unrolled_trips_address_four_distinct_work_items` asserts the
/// four copies are four different work items, and
/// `the_multi_trip_work_loop_is_unrolled_and_every_trip_states_its_own_windows` asserts the
/// window count multiplied with them.
#[test]
fn the_multi_trip_work_loop_is_unrolled_not_folded() {
    let m = converted("rope_q32", &[4, 32]);
    let func = m.kernel().expect("one kernel");
    let loops = func.regions[0]
        .ops
        .iter()
        .filter(|o| triton_ktir::passes::distribute_work::is_per_core_work_loop(&m, o))
        .count();
    assert_eq!(
        loops, 0,
        "the work loop must not reach the handoff: every reader in their `regions()` walks the \
         top level, so a loop there hides the whole body from all of them"
    );
    // FOUR trips, so four `ktdp.store` pairs: the kernel stores twice per work item.
    let stores = func.regions[0]
        .ops
        .iter()
        .filter(|o| o.kind == triton_ktir::ir::OpKind::KtdpStore)
        .count();
    assert_eq!(
        stores, 8,
        "rope_fwd stores twice per work item and rope_q32 runs 4 trips per core: 8 stores. \
         {stores} says the loop was folded (2) rather than unrolled, and 96 of the 128 work \
         items were dropped"
    );
    // A launch contract is recorded ONLY for the fold, which invents a trip for an
    // out-of-range tile. An even unroll invents nothing, so there is nothing to promise.
    assert!(
        func.attr(&triton_ktir::ir::AttrKey::FoldedGridLoop).is_none(),
        "an evenly unrolled loop has no zero-trip tile, so it must not record the fold's \
         launch obligation"
    );
}

/// THE PLANTED DEFECT: put the views back inside a region and the guard says WHICH failure it is.
///
/// The hoist and the unroll together make the "view exists but is nested" branch of
/// `handoff::lower`'s guard unreachable from any of the twelve fixtures, so it is exercised by
/// re-nesting them on purpose. Without this the guard would have one arm that has never run, and
/// the two arms are the whole point of it -- a nested view and an absent one need opposite fixes
/// and must not report the same way.
///
/// The region is planted rather than borrowed: rope_q32's own work loop is unrolled away by the
/// time the handoff sees it, and a test that needed a surviving loop would be measuring the
/// unroll instead of the guard.
#[test]
fn a_view_pushed_into_a_region_is_refused_as_nested_not_as_absent() {
    let a = Arena::global();
    let mut m = converted("rope_q32", &[4, 32]);
    let func = m.kernel().expect("one kernel");
    // The landmark is the loop-shape lower bound `is_per_core_work_loop` looks for, so the
    // planted region is the same shape the pass would have produced.
    let (li, core_id) = func.regions[0]
        .ops
        .iter()
        .enumerate()
        .find_map(|(i, o)| {
            (o.kind == triton_ktir::ir::OpKind::KtdpGetComputeTileId).then(|| (i, o.result()))
        })
        .expect("the work-distribution landmark survives the unroll");
    let core_id = core_id.expect("the landmark defines a value");
    let at: Vec<usize> = (li..func.regions[0].ops.len())
        .rev()
        .filter(|i| {
            func.regions[0].ops[*i].kind == triton_ktir::ir::OpKind::KtdpConstructMemoryView
        })
        .collect();
    let block = &mut m.kernel_mut().expect("one kernel").regions[0].ops;
    let views: Vec<triton_ktir::ir::Op> = at.into_iter().map(|i| block.remove(i)).collect();
    assert_eq!(views.len(), 4, "the four views the hoist lifted");
    let mut body: Vec<triton_ktir::ir::Op> = views;
    body.reverse();
    block.insert(
        li + 1,
        triton_ktir::ir::Op::new(triton_ktir::ir::OpKind::ScfFor)
            .with_operands([core_id, core_id, core_id])
            .with_region(triton_ktir::ir::Region { args: vec![], ops: body }),
    );

    let e = triton_ktir::passes::to_ktir_emit::lower(&m, a)
        .expect_err("a view their reader cannot see is not a width they can read");
    let msg = e.to_string();
    assert!(
        msg.contains("only inside a nested region") && msg.contains("parameter 0 (%desc_x)"),
        "the refusal must say the view is NESTED and name the parameter -- got {msg:?}"
    );
    assert!(
        !msg.contains("addressed NOWHERE"),
        "a nested view must not be reported as an absent one -- got {msg:?}"
    );
}

//===----------------------------------------------------------------------===//
// Defect A2: a WINDOW stated inside the work loop
//===----------------------------------------------------------------------===//

/// NO PARAMETER'S WINDOW IS BURIED IN THE WORK LOOP.
///
/// The view guard above is only half of what `regions()` reads off a parameter. The other half
/// is the `ktdp.construct_access_tile` WINDOW -- the tile the program actually takes -- read by
/// the same non-recursive walk of `IRFunction::operations`. Nested, the reader's
/// `unwrap_or((0, 0, rows, cols))` says "the window is the whole buffer", which for a tiled
/// program is the one answer it never means; their guard turns that silence into a refusal.
///
/// MEASURED, before the commit that added this test: `rope_q32` at grid 4,32 hid 6 of its 6
/// windows (2 of 2 on parameter 0) inside the surviving work loop's region, and `regions()`
/// refused it by name at the `bake_py` stage `regions`.
///
/// ⛔ THE WORK LOOP ONLY, AND THE OTHER KIND IS RECORDED SEPARATELY BELOW. A window can be
/// nested by either of two loops and they are not the same defect. The work loop is one WE
/// mint -- `distribute_work` wraps the whole body in `for pid = ktdp.get_compute_tile_id() to
/// work_items step num_cores` -- so a window inside it is our pass's doing and this asserts it
/// never happens. A `tl.range` in the kernel's own source is the PROGRAM's loop, and taking it
/// apart is a different piece of work; those are listed exactly, not skipped.
#[test]
fn no_parameters_window_is_buried_in_the_work_loop() {
    let a = Arena::global();
    let mut buried: Vec<String> = Vec::new();
    for (config, grid) in ALL_TWELVE {
        // `attention_flash_noncausal` is refused a stage earlier for Defect B (a parameter
        // nothing addresses), which has its own test below. Windowing is not measurable on a
        // function that cannot be built.
        if *config == "attention_flash_noncausal" {
            continue;
        }
        let m = converted(config, grid);
        let f = triton_ktir::passes::to_ktir_emit::lower(&m, a)
            .unwrap_or_else(|e| panic!("{config}: the handoff refused: {e}"));
        let n = windows_under_work_loop(&f);
        if n > 0 {
            buried.push(format!("{config} at grid {grid:?}: {n} window(s)"));
        }
    }
    assert!(
        buried.is_empty(),
        "a window inside the per-core work loop is a window their `regions()` cannot see, and it \
         falls back to the WHOLE view instead of saying so. `fold_grid_work_loop` folds a \
         single-trip loop and unrolls a multi-trip one, so no window should be under one at \
         all:\n  {}",
        buried.join("\n  ")
    );
}

/// THE PROGRAM'S OWN `tl.range` LOOPS STILL BURY THEIR WINDOWS, AND EXACTLY THESE DO.
///
/// ⛔ RECORDED, NOT SKIPPED, AND NARROW: the exact parameter list per configuration, so a
/// change that buries one more window fails here with its name in it and a change that fixes one
/// fails here too and has to say so. A blanket "these configurations may nest" would have hidden
/// the work-loop defect the test above exists for.
///
/// WHY IT IS A DIFFERENT PIECE OF WORK. These are K-blocking and flash-attention loops the
/// kernel SOURCE writes (`for k in tl.range(0, D_MODEL, BLOCK_K)`, `swiglu_mlp.py`; the KV-block
/// loop in `attention_flash*.py`), so the window really does move per trip and there is no
/// smaller tile that stands for the loop. Their `regions()` names the two legal answers -- one
/// node per trip, or hoisted windows -- and for a genuine K-blocked matmul the answer is their
/// `WorkPlan::time_tile_for_lx`, which is the only minter of `TimeTile` and is theirs to run,
/// not ours to pre-empt by unrolling a 200-trip loop. `bake_py`'s `swiglu_mlp_flat` and
/// `swiglu_mlp_granite_flat` configurations exist to measure exactly that and are not in the
/// driver's twelve.
///
/// The work loop is not like this: its trip count is the grid over the core count, it is ours
/// rather than the program's, and 4 copies is the whole of it.
#[test]
fn the_source_loops_that_still_bury_windows_are_exactly_these() {
    let a = Arena::global();
    // (config, grid, the parameters whose windows sit inside a `tl.range` body)
    let recorded: &[(&str, &[i64], &[usize])] = &[
        // The flash KV-block loop: K, V and the mask are read per block. THE CAUSAL CONFIG IS NO
        // LONGER BURIED: its off-band trip count is grid-position-dependent, so the first trip
        // unroll left the sweep standing and the grid unroll's per-copy constant folding plus the
        // SECOND trip-unroll arm (and its zero-trip deletion) now straight-line it per copy.
        ("attention_flash_causal", &[4, 4], &[]),
        // `for n in tl.range(0, D_FF, BLOCK_N)` -- 4 trips at D_FF=256 / BLOCK_N=64. NOT
        // BURIED ANY MORE: the KV-sweep unroll (27405e501) straight-lines these too, so
        // the recorded state moved from [0,1,2,3] to [] and this row is now a CONTROL.
        ("swiglu_mlp_small", &[1], &[]),
        ("swiglu_mlp_granite", &[1], &[]),
        ("swiglu_mlp_tiled_k", &[1], &[]),
        // The decoder's MLP block: the gate/up/down weights. UNBURIED with the swiglu
        // rows above (same KV-sweep unroll debt, 27405e501) -- the recorded state moved.
        ("decoder_layer_one", &[1], &[]),
        // Two layers, so the same three weights twice.
        ("decoder_two_layers", &[1], &[]),
        // AND THE CONTROLS: these have no source loop left, so nothing is buried.
        ("embedding_granite_bm128_control", &[2], &[]),
        ("embedding_granite", &[4], &[]),
        ("rmsnorm_granite", &[1], &[]),
        ("rope_kv8", &[4, 8], &[]),
        // THE ONE THIS COMMIT CHANGED: rope_q32's four parameters were all buried by the WORK
        // loop, and after the unroll none of them is buried by anything.
        ("rope_q32", &[4, 32], &[]),
    ];
    for (config, grid, want) in recorded {
        let m = converted(config, grid);
        let f = triton_ktir::passes::to_ktir_emit::lower(&m, a)
            .unwrap_or_else(|e| panic!("{config}: the handoff refused: {e}"));
        let got: Vec<usize> = window_counts(&f)
            .into_iter()
            .enumerate()
            .filter(|(_, (top, deep))| deep > top)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            got,
            want.to_vec(),
            "{config} at grid {grid:?}: the parameters whose windows are nested changed. This \
             list is the recorded state of the SOURCE-loop gap, so a difference is either a \
             regression or that gap moving -- say which."
        );
    }
}

/// THE WINDOWS ARE THE PROGRAM'S OWN, NOT ONE STANDING FOR FOUR.
///
/// The control for the test above, and the one that separates the two legal answers the guard
/// names. `deep == top` is reachable two ways: hoist each window out of the loop (which cannot
/// be done here -- see below), or unroll the loop so each trip states its own. A "fix" that
/// moved ONE window out and dropped the other three trips' would satisfy `deep == top` and
/// silently compute a quarter of the program.
///
/// WHY HOISTING IS NOT AVAILABLE FOR A WINDOW, read off the emission rather than argued: a
/// window's row offset is `(pid / 4) * 256 + (pid % 4) * 64`, a function of the loop's INDUCTION
/// VARIABLE (`arith.remui`/`arith.divui` of it, in `distribute_work`'s axis bridges). A view is
/// loop-invariant by construction and hoists; a window is not, and the only way to state all of
/// them above the loop is to have one copy per trip. So the count must MULTIPLY by the trip
/// count: 6 windows in the body become 4 x 6 = 24 at the top level.
#[test]
fn the_multi_trip_work_loop_is_unrolled_and_every_trip_states_its_own_windows() {
    let a = Arena::global();
    // rope_kv8 is 4*8 = 32 work items over 32 cores: ONE trip, the loop folds, 6 windows.
    // rope_q32 is 4*32 = 128 over 32: FOUR trips, and 24 windows is the only count that has
    // not dropped a work item.
    for (config, grid, want) in [("rope_kv8", &[4, 8], 6usize), ("rope_q32", &[4, 32], 24)] {
        let m = converted(config, grid);
        let f = triton_ktir::passes::to_ktir_emit::lower(&m, a)
            .unwrap_or_else(|e| panic!("{config}: the handoff refused: {e}"));
        let n = f
            .operations
            .iter()
            .filter(|o| o.op_type == OpKind::KtdpConstructAccessTile)
            .count();
        assert_eq!(
            n, want,
            "{config} at grid {grid:?} must state {want} top-level windows -- one set per trip. \
             {} says a trip's windows were dropped rather than emitted.",
            if n < want { "fewer" } else { "more" }
        );
        assert!(
            !f.ops_deep().iter().any(|o| o.op_type == OpKind::ScfFor),
            "{config}: the handed-over function must be straight-line -- their producer emits no \
             `scf.for` at all and every reader in `regions()` walks the top level only"
        );
    }
}

/// AND THE FOUR TRIPS ARE FOUR DIFFERENT WORK ITEMS.
///
/// 24 windows could be one trip emitted four times, which passes the test above and computes
/// work item `core_id` four times over. The loop was
/// `for pid = ktdp.get_compute_tile_id() to 128 step 32`, so the unrolled trips must address
/// `core_id`, `core_id + 32`, `core_id + 64` and `core_id + 96` -- the landmark itself, plus one
/// `arith.addi` against each of the three offsets. With `core_id` in `[0, 32)` those cover
/// `0..128` exactly once, which is what the loop did.
#[test]
fn the_unrolled_trips_address_four_distinct_work_items() {
    let a = Arena::global();
    let m = converted("rope_q32", &[4, 32]);
    let f = triton_ktir::passes::to_ktir_emit::lower(&m, a).expect("rope_q32 must lower");
    let consts: std::collections::HashMap<Ssa, i64> = f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::ArithConstant)
        .filter_map(|o| {
            let v = o.attributes.iter().find_map(|(k, a)| match (k, a) {
                (ktir_core::attrkey::AttrKey::Value, ktir_core::ir::Attr::Int(i)) => Some(*i),
                _ => None,
            })?;
            Some((o.result?, v))
        })
        .collect();
    let landmark: Vec<Ssa> = f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::KtdpGetComputeTileId)
        .filter_map(|o| o.result)
        .collect();
    assert_eq!(landmark.len(), 1, "one work-distribution landmark, shared by every trip");
    let mut offsets: Vec<i64> = f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::ArithAddi)
        .filter(|o| o.operands.first() == Some(&landmark[0]))
        .filter_map(|o| consts.get(o.operands.get(1)?).copied())
        .collect();
    offsets.sort_unstable();
    assert_eq!(
        offsets,
        vec![32, 64, 96],
        "trip t addresses `core_id + t * num_cores`; anything else recomputes one work item or \
         skips one. Got the offsets {offsets:?} against the landmark."
    );
}

//===----------------------------------------------------------------------===//
// Defect B: a parameter nothing addresses
//===----------------------------------------------------------------------===//

/// A PARAMETER NOTHING ADDRESSES IS REFUSED BY NAME, NOT GIVEN AN INVENTED WIDTH.
///
/// `attn_fwd` takes `desc_mask` as parameter 4 in both fixtures. The causal kernel loads it
/// (`tt.make_tensor_descriptor` + `tt.descriptor_load`, attention_flash_causal.ttir.mlir:52
/// and :111) so it gets a [64, 64] view; the noncausal one names it in the signature and
/// NOWHERE else, so no pass has anything to build a view from and none should invent one.
///
/// That parameter still occupies a binding slot -- `node_for` numbers `BufferId`s by
/// parameter position -- so dropping it here would silently renumber every buffer after it.
/// The refusal is the honest answer and the owner's call is which of the two fixes lands;
/// see the report on this commit.
#[test]
fn a_parameter_nothing_addresses_is_refused_naming_it_and_the_grid() {
    let a = Arena::global();
    let m = converted("attention_flash_noncausal", &[4, 4]);
    let e = triton_ktir::passes::to_ktir_emit::lower(&m, a)
        .expect_err("a parameter with no view must not reach their reader");
    let msg = e.to_string();
    // `grid [16]` was the pre-unroll spelling; the grid-position unroll collapses the
    // launch to one tile, so the refusal now names the collapsed grid. Both spellings
    // name the same kernel and the same unaddressed parameter.
    for needle in ["parameter 4", "desc_mask", "grid [1]", "construct_memory_view"] {
        assert!(
            msg.contains(needle),
            "the refusal must name {needle:?} -- got {msg:?}"
        );
    }
}

/// THE DISCRIMINATING CONTROL: the same kernel at the same grid, with the mask LIVE, lowers.
/// Without this, a refusal keyed on the kernel or on the parameter count would pass the test
/// above.
///
/// The grid-position unroll appends per-position CLONE arguments (the Q/O windows at
/// nonzero corners), so the width list is the five originals followed by the clones'
/// widths; every clone states the width of the parameter it clones. The five ORIGINALS
/// still lead, in order, which is what `node_for`'s positional binding slots read.
#[test]
fn the_causal_sibling_addresses_its_mask_and_lowers() {
    let a = Arena::global();
    let m = converted("attention_flash_causal", &[4, 4]);
    let f = triton_ktir::passes::to_ktir_emit::lower(&m, a)
        .unwrap_or_else(|e| panic!("attention_flash_causal must lower: {e}"));
    let widths = declared_widths(&f);
    assert!(
        widths.len() >= 5,
        "the five original parameters must still bind, got {}",
        widths.len()
    );
    assert_eq!(
        widths[..5],
        vec![
            Some(vec![1024, 128]),
            Some(vec![512, 128]),
            Some(vec![512, 128]),
            Some(vec![1024, 128]),
            Some(vec![64, 64]),
        ],
        "all five parameters, mask included, state a width"
    );
    assert!(
        widths[5..].iter().all(|w| w.is_some()),
        "every unroll clone argument must state a width too"
    );
}

/// THE UNROLL'S BLAST RADIUS, MEASURED RATHER THAN ARGUED.
///
/// `fold_grid_work_loop` handles every work loop, and of the twelve configurations the driver
/// runs exactly ONE has a loop that can run more than one trip -- so exactly one takes the
/// unroll arm and the other eleven's emissions are byte-unchanged. Asserting it here means a
/// future fixture that starts going multi-trip shows up as a failure with its name in it rather
/// than as an unexplained golden diff.
///
/// The measurement is the unroll's own fingerprint: every configuration hands over a
/// straight-line function, and rope_q32 is the only one whose op count is a MULTIPLE of the
/// per-trip body rather than one copy of it.
#[test]
fn exactly_one_of_the_twelve_is_multi_trip_and_none_hands_over_a_loop() {
    let mut with_loop: Vec<&str> = Vec::new();
    let mut unrolled: Vec<&str> = Vec::new();
    for (config, grid) in ALL_TWELVE {
        let m = converted(config, grid);
        let func = m.kernel().expect("one kernel");
        if func
            .regions[0]
            .ops
            .iter()
            .any(|o| triton_ktir::passes::distribute_work::is_per_core_work_loop(&m, o))
        {
            with_loop.push(config);
        }
        // The unroll is the only thing that mints an `arith.addi` against the landmark.
        let landmark: Vec<triton_ktir::ir::Ssa> = func.regions[0]
            .ops
            .iter()
            .filter(|o| o.kind == triton_ktir::ir::OpKind::KtdpGetComputeTileId)
            .filter_map(|o| o.result())
            .collect();
        if func.regions[0].ops.iter().any(|o| {
            o.kind == triton_ktir::ir::OpKind::ArithAddi
                && o.operands.first().is_some_and(|v| landmark.contains(v))
        }) {
            unrolled.push(config);
        }
    }
    assert!(
        with_loop.is_empty(),
        "no configuration may hand over a per-core work loop; {with_loop:?} still do"
    );
    assert_eq!(
        unrolled,
        vec!["rope_q32"],
        "only a grid with more work items than the 32 compute cores is multi-trip, and rope_q32 \
         is the one: 128 over 32. Any other name here is a configuration whose emission just \
         changed shape."
    );
}

/// EVERY OTHER CONFIGURATION THE DRIVER RUNS STILL DECLARES ALL OF ITS PARAMETERS.
///
/// The regression gate in the direction that matters: ten of the twelve already got past
/// `regions()` before this commit, and neither fix may cost one of them.
#[test]
fn the_other_ten_configurations_declare_every_parameter() {
    let a = Arena::global();
    let cases: &[(&str, &[i64])] = &[
        ("attention_flash_causal", &[4, 4]),
        ("swiglu_mlp_small", &[1]),
        ("swiglu_mlp_granite", &[1]),
        ("swiglu_mlp_tiled_k", &[1]),
        ("embedding_granite_bm128_control", &[2]),
        ("embedding_granite", &[4]),
        ("rmsnorm_granite", &[1]),
        ("rope_kv8", &[4, 8]),
        ("decoder_layer_one", &[1]),
        ("decoder_two_layers", &[1]),
    ];
    for (config, grid) in cases {
        let m = converted(config, grid);
        let f = triton_ktir::passes::to_ktir_emit::lower(&m, a)
            .unwrap_or_else(|e| panic!("{config}: the handoff refused: {e}"));
        let widths = declared_widths(&f);
        let missing: Vec<usize> = widths
            .iter()
            .enumerate()
            .filter(|(_, w)| w.is_none())
            .map(|(i, _)| i)
            .collect();
        assert!(
            missing.is_empty(),
            "{config} at grid {grid:?}: parameters {missing:?} of {} state no top-level view",
            widths.len()
        );
    }
}

//===----------------------------------------------------------------------===//
// THE DTYPE AND THE SHAPE, which the C++ text carries only in a result TYPE
//===----------------------------------------------------------------------===//

/// EVERY OP WITH AN ELEMENT DTYPE MUST STATE IT AS AN ATTRIBUTE, and every op with a shaped
/// result must state its shape as one.
///
/// ⛔ THIS IS A CLOSED SILENT WRONG ANSWER, NOT A STYLE PREFERENCE. The C++ KTIR text carries
/// the element dtype only in the result type -- `ktdp.construct_memory_view ... :
/// memref<64x4096xf16>` -- and an access tile's window extents only in its result type
/// (`-> !ktdp.access_tile<64x4096xindex>`). A port that emits result types therefore looks
/// complete and is not. THEIR OWN PRODUCER sets both forms:
/// `scratchy:crates/targets/spyre/src/lower_subtile_tape_to_ktir.rs:2304-2310` builds
/// `KtdpConstructMemoryView` with `AttrKey::Dtype` AND an `IrType::MemRef`, and does the same for
/// `TensorSplat` (:2585) and `TensorEmpty` (:2724).
///
/// NOTHING IN THE DXP PATH READS IT. `ktir-superdsc`'s `lower_ktir_to_superdsc` never reads
/// `AttrKey::Dtype` as a required attribute, and `dxp_standalone` compiles and executes no
/// arithmetic, so seven configurations reached SpyreCode with both attributes missing. The KTIR
/// EXECUTOR is the first consumer that reads them, and it reads them three ways -- two hard
/// refusals (`ktir-emulator:src/dialects/ktdp.rs:92` for the view's dtype, `:188` for the access
/// tile's shape) and ⚠️ ONE SILENT GUESS: a tensor `arith.constant` with no dtype attribute is
/// read as `DType::F16` (`src/dialects/arith.rs:554`). For an all-f16 Granite kernel that guess is
/// right for the wrong reason; an f32 constant tensor would have been read as f16 and no stage
/// would have said so.
#[test]
fn every_shaped_result_states_its_dtype_and_shape_as_attributes() {
    use ktir_core::ir::Attr;
    use ktir_core::irtype::IrType as KIrType;

    let a = Arena::global();
    // Three fixtures, chosen for the op kinds they exercise rather than for coverage: rmsnorm has
    // the views + access tiles + a reduce, swiglu has the matmuls, embedding has the INDIRECT
    // access tile (whose parent view is the one a gather addresses).
    for (config, grid) in [
        ("rmsnorm_granite", &[1i64][..]),
        ("swiglu_mlp_small", &[1][..]),
        ("embedding_granite", &[4][..]),
    ] {
        let m = converted(config, grid);
        let f = triton_ktir::passes::to_ktir_emit::lower(&m, a)
            .unwrap_or_else(|e| panic!("{config}: the handoff refused: {e}"));

        let mut checked_dtype = 0usize;
        let mut checked_shape = 0usize;
        walk_ops(f.operations, &mut |op| {
            let Some(t) = op.result_type.as_ref() else { return };
            // The element dtype, where the type has one.
            let want_elem = match t {
                KIrType::Tensor { elem, .. } | KIrType::MemRef { elem, .. } => Some(*elem),
                _ => None,
            };
            if let Some(elem) = want_elem {
                match op.attr(ktir_core::attrkey::AttrKey::Dtype) {
                    Some(Attr::Dtype(d)) => {
                        assert_eq!(
                            *d, elem,
                            "{config}: `{:?}` states dtype {d:?} as an attribute and {elem:?} in \
                             its result type. TWO WRITERS FOR ONE FACT is worse than one, because \
                             a consumer reading the attribute and one reading the type now \
                             disagree.",
                            op.op_type
                        );
                        checked_dtype += 1;
                    }
                    other => panic!(
                        "{config}: `{:?}` has result type {t:?} but its `Dtype` attribute is \
                         {other:?}. `ktir-emulator` REFUSES a memory view without it \
                         (dialects/ktdp.rs:92) and silently GUESSES f16 for a constant tensor \
                         without it (dialects/arith.rs:554).",
                        op.op_type
                    ),
                }
            }
            // The shape, for every shaped result type.
            if let Some(dims) = result_dims(t) {
                match op.attr(ktir_core::attrkey::AttrKey::Shape) {
                    Some(Attr::IntList(got)) => {
                        assert_eq!(
                            got.to_vec(),
                            dims,
                            "{config}: `{:?}` states shape {got:?} as an attribute and {dims:?} in \
                             its result type",
                            op.op_type
                        );
                        checked_shape += 1;
                    }
                    // `sizes_dyn` is the runtime-extent form; the emulator accepts either.
                    _ if op.attr(ktir_core::attrkey::AttrKey::SizesDyn).is_some() => {}
                    other => panic!(
                        "{config}: `{:?}` has shaped result type {t:?} but its `Shape` attribute \
                         is {other:?} and it has no `sizes_dyn` either. `ktir-emulator` REFUSES an \
                         access tile without it (dialects/ktdp.rs:188).",
                        op.op_type
                    ),
                }
            }
        });
        // ⛔ A LOOP THAT VISITED NOTHING PASSES. These counts are the guard against that -- and
        // they are lower bounds, not pins, because the number of views a configuration emits is
        // the unroller's business and pinning it here would make this test fail for the wrong
        // reason every time that changes.
        assert!(
            checked_dtype >= 3,
            "{config}: only {checked_dtype} op(s) with an element dtype were checked; a walk that \
             visits nothing asserts nothing"
        );
        assert!(
            checked_shape >= 3,
            "{config}: only {checked_shape} shaped result(s) were checked"
        );
    }
}

/// ⛔ AND IT IS NOT A BLANKET STAMP. THIS IS THE CONTROL.
///
/// The trap this tree has already fallen into once is the ONE UNIFORM STAMP: an attribute applied
/// to every op regardless of whether the op has the fact, which looks like provenance and carries
/// none. So the emission is asserted to be DERIVED -- an op whose result type has no element type
/// (`index`, the type every `arith.constant` address and every `ktdp.construct_access_tile` result
/// carries) must have NO `Dtype` attribute, and an op with no result type at all must have neither.
///
/// If this test ever fails, the emission became unconditional and the attribute stopped meaning
/// anything.
#[test]
fn the_dtype_attribute_is_derived_and_not_stamped_on_everything() {
    use ktir_core::ir::Attr;
    use ktir_core::irtype::IrType as KIrType;

    let a = Arena::global();
    let m = converted("rmsnorm_granite", &[1]);
    let f = triton_ktir::passes::to_ktir_emit::lower(&m, a).expect("rmsnorm_granite must lower");

    let mut index_typed = 0usize;
    let mut untyped = 0usize;
    walk_ops(f.operations, &mut |op| {
        match op.result_type.as_ref() {
            // An `index`/access-tile result has no ELEMENT type, so it must carry no dtype.
            Some(KIrType::Index) | Some(KIrType::AccessTile { .. }) => {
                index_typed += 1;
                assert!(
                    !matches!(op.attr(ktir_core::attrkey::AttrKey::Dtype), Some(Attr::Dtype(_))),
                    "`{:?}` has no element type and must carry no `Dtype` attribute: an attribute \
                     stamped on every op is provenance that means nothing",
                    op.op_type
                );
            }
            None => {
                untyped += 1;
                assert!(
                    !matches!(op.attr(ktir_core::attrkey::AttrKey::Dtype), Some(Attr::Dtype(_))),
                    "`{:?}` has no result type at all and must carry no `Dtype` attribute",
                    op.op_type
                );
            }
            _ => {}
        }
    });
    assert!(
        index_typed >= 3 && untyped >= 1,
        "the control saw {index_typed} index/access-tile result(s) and {untyped} untyped op(s); \
         with too few of either it is not discriminating between derived and stamped"
    );
}

/// Their `IrType`'s dims, where it has them.
fn result_dims(t: &ktir_core::irtype::IrType<'_>) -> Option<Vec<i64>> {
    use ktir_core::irtype::IrType as K;
    match t {
        K::Tensor { dims, .. } | K::MemRef { dims, .. } | K::AccessTile { dims } => {
            Some(dims.to_vec())
        }
        _ => None,
    }
}

/// Every op, recursing into regions. A top-level-only walk is how a fact buried in a work loop's
/// body goes unmeasured -- the defect this whole file exists to catch in another form.
fn walk_ops<'a>(
    ops: &'a [ktir_core::ir::Operation<'a>],
    f: &mut impl FnMut(&'a ktir_core::ir::Operation<'a>),
) {
    for op in ops {
        f(op);
        for r in op.regions {
            walk_ops(r, f);
        }
    }
}

/// ⭐ THE GATHER'S SUBSCRIPT, AS THE EXECUTOR WILL READ IT — AND THE WRONG READING NAMED.
///
/// `to_ktir_emit::indirect_access_tile` crosses four disagreements at once (its doc table). This
/// pins the result of that crossing on the one fixture that produces the op, because THREE of the
/// four have a silently-wrong neighbour:
///
///   * drop the captures and state no subscript — the executor's legacy identity path addresses
///     the index view by the bare enumeration point, so grid item 0 agrees and items 1..3 each
///     re-read ids[0..64]. 192 of 256 tokens get the wrong embedding and every shape checks out.
///   * swap the two symbols — the row offset lands on the COLUMN subscript. Still well-formed.
///   * leave the captures-first domain in place — `Dim(0)`/`Dim(1)` then mean the captured
///     scalars, and `eval` at a 2-point enumeration reads the point where the capture belongs.
///
/// So the assertions below are structural on purpose: not "a subscript exists" but WHICH one.
#[test]
fn the_gathers_subscript_crosses_into_the_executors_domain() {
    use ktir_core::affine::AffineExpr as E;
    use ktir_core::attrkey::AttrKey;
    use ktir_core::ir::Attr;

    let m = converted("embedding_granite", &[4]);
    let a = Arena::global();
    let f = triton_ktir::passes::to_ktir_emit::lower(&m, a).expect("embedding_granite hands over");
    let tile = f
        .operations
        .iter()
        .find(|o| o.op_type == ktir_core::opkind::OpKind::KtdpConstructIndirectAccessTile)
        .expect("the gather reaches the handoff");

    // Their layout: operands[0] primary, operands[1..] INDEX VIEWS ONLY. The captured scalars
    // (`%row_off`, `%c0`) are no longer operands, so two is the whole list.
    assert_eq!(
        tile.operands.len(),
        2,
        "the handed-off gather must carry `(primary, index_view)` and nothing else; the executor \
         reads every operand past the first as a MemRef, which is what `index_view 0 is Index(0), \
         expected MemRef` was"
    );

    match tile.attr(AttrKey::DimKinds) {
        Some(Attr::StrList(k)) => assert_eq!(
            *k,
            ["indirect", "direct_sub"],
            "the executor matches `dim_kinds` as STRINGS; the C++ boolean list [\"true\", \
             \"false\"] falls through its unknown-kind arm"
        ),
        other => panic!("dim_kinds is {other:?}"),
    }
    match tile.attr(AttrKey::DimData) {
        Some(Attr::IntList(d)) => assert_eq!(*d, [0, 0], "one index view, so view index 0"),
        other => panic!("dim_data is {other:?}"),
    }
    match tile.attr(AttrKey::IntermediateVars) {
        Some(Attr::Ssas(v)) => assert_eq!(
            v.len(),
            2,
            "the two captured scalars -- the block's row offset and c_y -- must arrive as \
             `intermediate_vars`, which is the only thing the subscript's symbols can name"
        ),
        other => panic!("intermediate_vars is {other:?}"),
    }

    let Some(Attr::AffineMapList(subs)) = tile.attr(AttrKey::DimSubs) else {
        panic!(
            "the gather carries no `dim_subs`. Without it the executor takes the legacy \
             identity-subscript path and 192 of 256 tokens read the wrong row -- silently"
        );
    };
    assert_eq!(subs.len(), 2, "one subscript map per output dim");
    for (d, map) in subs.iter().enumerate() {
        assert_eq!(
            (map.num_dims, map.num_syms),
            (2, 2),
            "dim {d}'s subscript must be over the ENUMERATION point (2 dims) with the captures as \
             SYMBOLS (2); a captures-first 4-dim domain evaluates the wrong operand at every point"
        );
        assert_eq!(map.exprs.len(), 1, "dim {d} takes one subscript expression");
    }
    // `s0 + d0` for the indirect row and `s1 + d1` for the direct column -- NOT the other way
    // round, which would put the block's row offset on the column.
    let want = |sym: usize, dim: usize| {
        (
            E::Add(a.expr(E::Sym(sym)), a.expr(E::Dim(dim))),
            E::Add(a.expr(E::Dim(dim)), a.expr(E::Sym(sym))),
        )
    };
    for (d, sym, dim) in [(0usize, 0usize, 0usize), (1, 1, 1)] {
        let (lr, rl) = want(sym, dim);
        let got = subs[d].exprs[0];
        assert!(
            got == lr || got == rl,
            "dim {d}'s subscript is {got:?}; it must be s{sym} + d{dim} -- the capture the C++ map \
             puts at domain position {sym} plus enumeration variable {dim}. Anything else gathers \
             a well-formed WRONG element"
        );
    }
}
