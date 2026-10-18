// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ A STICK-MAJOR ELEMENTWISE OP SPLITS ITS STICKS FIRST — the work-division law
//! [`EwDivision`](ktir_superdsc::work::EwDivision), proven exhaustively and pinned at the shipped doors.
//!
//! A stick-major rank-2 `[mb, out]` operand is laid out `[out/64][mb][64]`, so a core holding all rows of a
//! run of sticks reads one contiguous block, while the rows-first cut [`distribute_cores`] makes hands each
//! core `cols/64` strided one-row bursts. MEASURED in isolation (#549's sweep, the real `mulsilu_o461`
//! descriptor and placement, one CB, 32 cores): `[256, 8192]` mul 249.8 µs (50.4 GB/s) rows-first →
//! 121.8 µs (103.3 GB/s) sticks-first; `[256, 2048]` 50.0 → 30.3 µs.
//!
//! Kani is blocked for this crate (#618), so THIS is the proof: every `(rows, cols)` an fp16 elementwise op
//! of the models in scope can take is enumerated, and each plan is checked against the properties the
//! sealed `WorkPlan` and the card rely on.
//!
//! [`distribute_cores`]: ktir_superdsc::work::distribute_cores

use ktir_superdsc::emit::{In, assemble_pointwise_broadcast_off};
use ktir_superdsc::sdsc_abstract::{BlockCols, RowBlockedTag, RowCount, StickLayout, Stk};
use ktir_superdsc::superdsc_opspec::{Df, ItDim, MAX_CORES, MaxCores, WorkPlan};
use ktir_superdsc::work::{
    EwDivision, core_to_wk_slice, distribute_cores, distribute_cores_sticks_first,
};

const STICK: u32 = 64;

fn dims(rows: u32, cols: u32) -> Vec<ItDim> {
    let dim = |name: &'static str, size: u32, is_stick: bool| ItDim {
        name,
        size,
        is_reduction: false,
        is_stick,
        df: Df::Fp16,
    };
    vec![
        dim("mb", rows, false),
        dim("out", cols, true),
        dim("y", 1, false),
    ]
}

/// Every row count an fp16 elementwise op takes in the shipped ladders (prefill rungs, decode batch
/// rungs, the prefix rungs), and every 64-multiple width up to granite-8b's 12800-wide MLP.
fn grid() -> impl Iterator<Item = (u32, u32)> {
    let rows = (2u32..=64).chain([71, 80, 88, 96, 112, 128, 192, 256, 384, 512]);
    rows.flat_map(|r| (1u32..=200).map(move |s| (r, s * STICK)))
}

/// ⭐⭐⭐ THE PROOF. For every `(rows, cols)` of the grid the sticks-first plan:
///   * is a valid `WorkPlan` (the sealed proof: product ≤ 32, every core a whole number of sticks);
///   * splits only `mb` and `out`, and gives each split dim an even division;
///   * covers every element of the `[rows, cols]` tensor EXACTLY ONCE — the cores' work-slice corners,
///     read through the same `core_to_wk_slice` the emitter writes `coreIdToWkSlice_` from, tile it;
///   * uses AT LEAST AS MANY cores as the rows-first plan.
#[test]
fn the_sticks_first_plan_tiles_every_shape_on_at_least_as_many_cores() {
    let mut cases = 0usize;
    let mut wider = 0usize;
    for (rows, cols) in grid() {
        let d = dims(rows, cols);
        let plan = WorkPlan::divide(&d, MaxCores::<MAX_CORES>, distribute_cores_sticks_first)
            .unwrap_or_else(|e| panic!("[{rows}, {cols}]: sticks-first plan refused: {e}"));
        let rows_first = WorkPlan::divide(&d, MaxCores::<MAX_CORES>, distribute_cores)
            .unwrap_or_else(|e| panic!("[{rows}, {cols}]: rows-first plan refused: {e}"));
        let cores = plan.cores_used().get();
        assert!(cores <= MAX_CORES, "[{rows}, {cols}]: {cores} cores");
        assert!(
            cores >= rows_first.cores_used().get(),
            "[{rows}, {cols}]: sticks-first uses {cores} cores, rows-first {}",
            rows_first.cores_used().get()
        );
        if cores > rows_first.cores_used().get() {
            wider += 1;
        }
        assert_eq!(plan.split_of("y"), 1, "[{rows}, {cols}]: y split");
        let (sm, so) = (plan.split_of("mb"), plan.split_of("out"));
        assert_eq!(rows % sm, 0, "[{rows}, {cols}]: mb split {sm}");
        assert_eq!(
            (cols / STICK) % so,
            0,
            "[{rows}, {cols}]: out split {so} not whole sticks"
        );
        assert_eq!(sm * so, cores, "[{rows}, {cols}]: splits vs cores");
        // Exact cover: each core's (mb, out) corner block, at its per-core extents.
        let (pm, po) = (plan.per_core_extent("mb"), plan.per_core_extent("out"));
        assert_eq!(pm * sm, rows);
        assert_eq!(po * so, cols);
        let slices = core_to_wk_slice(&plan, &["mb", "out", "y"]);
        assert_eq!(slices.len() as u32, cores);
        let mut seen = std::collections::BTreeSet::new();
        for s in slices.values() {
            let key = (
                serde_json::to_value(s["mb"]).unwrap().as_u64().unwrap(),
                serde_json::to_value(s["out"]).unwrap().as_u64().unwrap(),
            );
            assert!(
                key.0 < sm as u64 && key.1 < so as u64,
                "[{rows}, {cols}]: slice {key:?}"
            );
            assert!(
                seen.insert(key),
                "[{rows}, {cols}]: two cores own block {key:?}"
            );
        }
        assert_eq!(seen.len() as u32, sm * so, "[{rows}, {cols}]: blocks owned");
        cases += 1;
    }
    assert!(cases > 10_000, "the grid shrank to {cases} cases");
    assert!(
        wider > 0,
        "no shape gained cores — the case the stick-split search exists for is not reached"
    );
}

/// The stick axis takes its split FIRST: every shape whose stick count has a divisor filling the
/// machine runs out-only, and the measured shapes land exactly where #549's sweep put them.
#[test]
fn the_measured_shapes_split_their_sticks_first() {
    let split = |rows: u32, cols: u32| {
        let p = WorkPlan::divide(
            &dims(rows, cols),
            MaxCores::<MAX_CORES>,
            distribute_cores_sticks_first,
        )
        .unwrap();
        (p.split_of("mb"), p.split_of("out"))
    };
    // granite-2b's MLP intermediate (mulsilu / silu), 128 sticks: out:32.
    assert_eq!(split(256, 8192), (1, 32));
    assert_eq!(split(512, 8192), (1, 32));
    // the 2048-wide residual stream, 32 sticks: out:32.
    assert_eq!(split(256, 2048), (1, 32));
    // granite-8b: hidden 4096 (64 sticks) out:32; the MLP's second column chunk 72 sticks — no divisor in
    // 25..=32, so the search takes out:8 × mb:4 rather than strand cores at out:24.
    assert_eq!(split(512, 4096), (1, 32));
    assert_eq!(split(512, 4608), (4, 8));
    // a 4-request decode rung on the 2b MLP: out:32, where rows-first made mb:4 × out:8.
    assert_eq!(split(4, 8192), (1, 32));
    // Rows-first, for contrast: the division it replaces.
    let rows_first =
        WorkPlan::divide(&dims(256, 8192), MaxCores::<MAX_CORES>, distribute_cores).unwrap();
    assert_eq!(
        (rows_first.split_of("mb"), rows_first.split_of("out")),
        (32, 1)
    );
}

/// The law's resolution: sticks first only for a stick-major op touching no core-local pin, and a
/// concrete division is its own answer.
#[test]
fn the_law_resolves_on_the_layout_and_the_pins() {
    assert_eq!(
        EwDivision::Law.resolve(true, false),
        EwDivision::SticksFirst
    );
    assert_eq!(EwDivision::Law.resolve(true, true), EwDivision::RowsFirst);
    assert_eq!(EwDivision::Law.resolve(false, false), EwDivision::RowsFirst);
    assert_eq!(
        EwDivision::RowsFirst.resolve(true, false),
        EwDivision::RowsFirst
    );
    assert_eq!(
        EwDivision::SticksFirst.resolve(false, true),
        EwDivision::SticksFirst
    );
}

fn rb(name: &str, rows: u32, cols: u32) -> Stk<RowBlockedTag> {
    Stk::<RowBlockedTag>::new(name, StickLayout::row_blocked(rows as usize, cols as usize)).unwrap()
}

/// `(mb, out)` splits the descriptor the shipped pointwise door emits, read off `numWkSlicesPerDim_`.
fn emitted_split(rows: u32, cols: u32, rank3_flat_out: bool) -> (u64, u64) {
    let a = rb("t_a", rows, cols);
    let b = rb("t_b", rows, cols);
    let o = rb("t_o", rows, cols);
    let mut sid = 0i64;
    let op = if rank3_flat_out {
        // A column-sliced output is not a whole-block shift, so the door keeps it rank-3 flat — the
        // stick-major predicate does not hold and the op stays rows-first.
        assemble_pointwise_broadcast_off(
            "mul_flat",
            "multiply",
            RowCount::of_token_rows(rows),
            BlockCols::of_feature_cols(cols),
            &[In::full(&a).ew(), In::full(&b).ew()],
            &o,
            ktir_superdsc::addr::col_of(rows, 2 * cols, cols, Df::Fp16),
            &mut sid,
            None,
        )
    } else {
        assemble_pointwise_broadcast_off(
            "mul_sm",
            "multiply",
            RowCount::of_token_rows(rows),
            BlockCols::of_feature_cols(cols),
            &[In::full(&a).ew(), In::full(&b).ew()],
            &o,
            ktir_superdsc::addr::col_of(rows, cols, 0, Df::Fp16),
            &mut sid,
            None,
        )
    };
    let v = serde_json::to_value(op.op.as_ref().expect("a descriptor")).unwrap();
    let s = &v["numWkSlicesPerDim_"];
    (
        s["mb"].as_u64().unwrap_or(1),
        s["out"].as_u64().unwrap_or(1),
    )
}

/// ⭐ THE WIRING, through the door every dense elementwise op is assembled at: a stick-major rank-2 op
/// (granite-2b's mulsilu, `[256, 8192]`) emits out:32; the same op with a rank-3 flat output stays
/// rows-first; and a one-row op (decode) is untouched — rows-first and sticks-first agree there.
#[test]
fn the_pointwise_door_splits_a_stick_major_op_by_its_sticks() {
    assert_eq!(
        emitted_split(256, 8192, false),
        (1, 32),
        "stick-major: out first"
    );
    assert_eq!(emitted_split(256, 8192, true), (32, 1), "flat: rows first");
    assert_eq!(emitted_split(1, 8192, false), (1, 32), "one row: unchanged");
}
