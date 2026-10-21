// SPDX-License-Identifier: Apache-2.0

//! ⛔⛔ THE UNROLLED GRID'S POSITION COUNT IS `n × trips`, NOT `n` — AND THE TWIN THAT
//! MEASURES IT IS A CONFIGURATION PAIR, NOT A HAND-BUILT MODULE.
//!
//! # WHAT THIS MEASURES
//!
//! When a launch asks for more grid positions than there are compute tiles,
//! `fold_grid_work_loop`'s unroll arm regrids `grid` to `[num_cores]` and leaves
//! `trips = work_items / num_cores` straight-line copies per core in the body. The
//! launch then carries `n * trips` positions' worth of windows under an `n`-tile grid —
//! and `unroll_grid_positions` tells a parameter's PARTITIONED windows from its SWEPT
//! ones by counting its tiles against the position count. A count that forgets the
//! trip factor classifies every partitioned parameter as swept: NOTHING materializes,
//! the door's `r_cover` unions the store windows no clone wrote into a whole-view
//! cover, and the refusal chain ends at the epilogue divide's extent check — a
//! correct-looking kernel refused for a reason three passes upstream.
//!
//! The fix travels as `spyre.unrolled_grid_loop = [num_cores, work_items]`, set by the
//! arm that created the trips and read where the count is taken, so the two passes
//! cannot disagree about what a copy is. This file pins the OUTCOME of that contract:
//!
//! * `attention_banded_fold` — `(1, 8, 4)` = 32 positions, exactly the 32-core fold
//!   bound. One trip per core; the row BAKED before the fix and must bake after it,
//!   because the fix multiplies a count by `trips = 1` there.
//! * `attention_banded_batch` — `(2, 8, 4)` = 64 positions over the same kernel, TWO
//!   trips per core. Before the fix this row REFUSED (MEASURED: the RealDiv extent
//!   refusal above); after it, it bakes at EXACTLY TWICE the fold row's op count —
//!   the same per-position body, one window per position per trip, every partitioned
//!   parameter materialized.
//!
//! # WHY THE RATIO AND NOT TWO ABSOLUTE COUNTS
//!
//! The twin's rows differ only in `NUM_SEQS` and the grid, so their emissions differ
//! only by the trip factor: the batch row is the fold row's op list with each
//! position's ops appearing once per trip. Pinning `batch == 2 × fold` states exactly
//! that invariant and nothing else — an unrelated emission change that moves both rows
//! together stays green, while the two ways the count can go wrong (a factor that
//! drops, leaving the batch row refused; a factor that doubles the FOLD row too,
//! leaving 4×) both fail. The absolute counts are printed in the panic messages so a
//! failure names its numbers.
//!
//! # WHY THE CHAIN IS DRIVEN FROM THE CONFIGURATION TABLE
//!
//! Both rows come from `cases::case`, the same statement `bake_py` reads — a test that
//! restated the fixture, the constexprs, or the grid would be a second statement of a
//! configuration, and a configuration restated in two places drifts in one of them.
//! The chain is the whole `KTIR_WHOLE=1` leg set, value-level, minus the directory
//! write: source → ttir → KTIR → `to_ktir`'s passes → their node → their layout →
//! `emit_whole`. A refusal at any stage fails the test with that stage's message.

/// Drive one row of the twin through the whole-function door — every leg `bake_py`
/// runs under `KTIR_WHOLE=1`, minus the bake directory. `Err` carries the first
/// refusing stage's message.
fn drive(name: &str) -> Result<Vec<String>, String> {
    let (kspec, grid, door) = triton_ktir_superdsc::cases::case(name)
        .unwrap_or_else(|| panic!("`{name}` is not a configuration"));
    // This test drives the DOOR, so a doorless row cannot run here by construction; the twin
    // rows it names both have `Program::Attn`.
    let program = door.unwrap_or_else(|| panic!("`{name}` has no door kind to drive"));
    let src = std::fs::read_to_string(&kspec.file).map_err(|e| format!("read: {e}"))?;
    let mut tt =
        triton_frontend::codegen::compile(&src, &kspec, triton_frontend::target::Target::spyre())
            .map_err(|e| format!("codegen: {e}"))?;
    triton_frontend::opt::make_ttir(&mut tt).map_err(|e| format!("make_ttir: {e}"))?;
    let mut m = triton_ktir::from_ttir::convert(&tt).map_err(|e| format!("from_ttir: {e}"))?;
    triton_ktir::make_ktir(&mut m, &grid).map_err(|e| format!("make_ktir: {e}"))?;
    triton_ktir::passes::to_ktir::run(&mut m, &grid).map_err(|e| format!("to_ktir: {e}"))?;
    let node = triton_ktir_superdsc::node_for(&m, program).map_err(|e| e.message)?;
    let layout = triton_ktir_superdsc::layout::for_node(&node).map_err(|e| e.message)?;
    let ops = triton_ktir_superdsc::emit_whole(&node, Some(&layout)).map_err(|e| e.message)?;
    Ok(ops.into_iter().map(|o| o.op_name.clone()).collect())
}

/// THE CONTROL: the fold row — 32 positions, one per core — bakes through the
/// whole-function door. This row was door-legal before the trip contract existed
/// (`attn_paged_decode_batch_fold` proved the fold path materializes every
/// partitioned parameter), so a failure here means the fix broke the fold path, not
/// that the twin found a new wall.
#[test]
fn the_fold_row_bakes_one_trip_per_core() {
    let ops = drive("attention_banded_fold").expect("the 32-position fold row bakes (trips = 1)");
    assert!(
        !ops.is_empty(),
        "the fold row baked but emitted no ops — a bake over nothing"
    );
}

/// THE WALL, LIFTED: the batch row — the same kernel at 64 positions, two trips per
/// core — bakes, and its op count is EXACTLY TWICE the fold row's. Before the trip
/// contract this row refused at the epilogue divide's extent check (the partitioned
/// parameters' `n × trips` tiles read as swept, nothing materialized, and the door's
/// whole-view cover met the per-position `[1, 64]` divide operand); a return to that
/// state fails here. So does a count that over-multiplies — if the fold row ever
/// carried the factor too, the ratio would read 1, not 2.
#[test]
fn the_batch_row_bakes_at_exactly_the_trip_multiple_of_the_fold_row() {
    let fold = drive("attention_banded_fold").expect("the fold row bakes");
    let batch = drive("attention_banded_batch").expect(
        "the 64-position batch row bakes — two trips per core, the partitioned \
         parameters materialized at n × trips tiles",
    );
    assert_eq!(
        batch.len(),
        2 * fold.len(),
        "the batch row is the fold row's body with each position's ops appearing once \
         per trip: 64 positions over 32 cores is exactly two trips, so the op counts \
         must be in ratio 2:1 (fold = {}, batch = {})",
        fold.len(),
        batch.len()
    );
}
