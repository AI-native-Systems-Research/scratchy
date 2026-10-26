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

use ktir_superdsc::emit::{In, PointwiseFacts, assemble_pointwise_broadcast_off};
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
        // ⭐ THE TIE-BREAK: of every legal (out, mb) pair, the plan uses the MOST cores, and among the pairs
        // that tie on cores it takes the WIDEST out — the most contiguous sticks per core, which is the
        // win #549's (e) cells measured. Checked against the whole candidate set, not against the code's
        // own search.
        let sticks = cols / STICK;
        let best = (1..=sticks.min(MAX_CORES))
            .filter(|c| sticks.is_multiple_of(*c))
            .flat_map(|c| {
                (1..=(MAX_CORES / c).min(rows))
                    .filter(move |m| rows.is_multiple_of(*m))
                    .map(move |m| (c * m, c))
            })
            .max()
            .unwrap();
        assert_eq!(
            (cores, so),
            best,
            "[{rows}, {cols}]: (cores, out) = ({cores}, {so}); the most cores, widest out on a tie, is {best:?}"
        );
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
    // granite-8b: hidden 4096 (64 sticks) out:32; a 72-stick shape has no divisor in 25..=32, so the
    // search takes out:8 × mb:4 rather than strand cores at out:24. (8b's own 72-stick op — its second MLP
    // column chunk — is a non-block window, so the door presents it flat and it stays rows-first; see the
    // wiring test. The law's arithmetic is pinned here regardless.)
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

/// The law's resolution: sticks first only for a stick-major op touching no core-local pin, sprays
/// first for a row-split (y-major) op, and a concrete division is its own answer.
#[test]
fn the_law_resolves_on_the_layout_and_the_pins() {
    assert_eq!(
        EwDivision::Law.resolve(true, false, false),
        EwDivision::SticksFirst
    );
    assert_eq!(
        EwDivision::Law.resolve(true, true, false),
        EwDivision::RowsFirst
    );
    assert_eq!(
        EwDivision::Law.resolve(false, false, false),
        EwDivision::RowsFirst
    );
    assert_eq!(
        EwDivision::RowsFirst.resolve(true, false, false),
        EwDivision::RowsFirst
    );
    assert_eq!(
        EwDivision::SticksFirst.resolve(false, true, false),
        EwDivision::SticksFirst
    );
    // ⭐ THE ROW-SPLIT FACT OUTRANKS THE OTHERS: a y-major op's tensors walk ["y", "mb", "out"],
    // so only a split over the leading `y` gives a core a contiguous slab — the prefill mask add
    // measured 620 of 672 sticks never written under the rows-first split (see
    // `distribute_cores_sprays_first`'s doc).
    assert_eq!(
        EwDivision::Law.resolve(false, false, true),
        EwDivision::SpraysFirst
    );
    assert_eq!(
        EwDivision::Law.resolve(true, false, true),
        EwDivision::SpraysFirst
    );
    assert_eq!(
        EwDivision::SpraysFirst.resolve(false, false, true),
        EwDivision::SpraysFirst
    );
}

fn rb(name: &str, rows: RowCount, cols: BlockCols) -> Stk<RowBlockedTag> {
    Stk::<RowBlockedTag>::new(
        name,
        StickLayout::row_blocked(rows.get() as usize, cols.get() as usize),
    )
    .unwrap()
}

/// `(mb, out)` splits of the descriptor the shipped pointwise door emits for a `[rows, cols]` multiply
/// whose output is the column window starting at `c_start` of a `full_cols`-wide tensor, read off
/// `numWkSlicesPerDim_`.
///
/// The handles declare the FULL `[rows, full_cols]` storage — the window rides the offset, and a
/// window-shaped handle under a full-tensor offset is exactly the past-its-own-tensor window the
/// operand mint refuses (#731's law: the column-chunk bodies declare the region's storage width
/// too, never the piece's).
fn emitted_split(rows: u32, cols: u32, full_cols: u32, c_start: u32) -> (u64, u64) {
    let a = rb(
        "t_a",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(full_cols),
    );
    let b = rb(
        "t_b",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(full_cols),
    );
    let o = rb(
        "t_o",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(full_cols),
    );
    let mut sid = 0i64;
    let op = assemble_pointwise_broadcast_off(
        PointwiseFacts {
            op_name: "mul",
            op_func: "multiply",
            rows: RowCount::of_token_rows(rows),
            cols: BlockCols::of_feature_cols(cols),
            inputs: &[In::full(&a).ew(), In::full(&b).ew()],
            o: &o,
            out_offset: ktir_superdsc::addr::col_of(rows, full_cols, c_start, Df::Fp16).into(),
        },
        &mut sid,
        None,
    );
    let v = serde_json::to_value(op.op.as_ref().expect("a descriptor")).unwrap();
    let s = &v["numWkSlicesPerDim_"];
    (
        s["mb"].as_u64().unwrap_or(1),
        s["out"].as_u64().unwrap_or(1),
    )
}

/// [`emitted_split`] with every operand and the output at a [`ktir_superdsc::addr::ColWindow`] — the corner
/// the column-chunk bodies (`silumul`, `elementwise`, `scalarmul_scaled`) state since #622 —
/// returning the `(mb, out)` split and the output's `layoutDimOrder_`. The handles declare the full
/// `[rows, full_cols]` storage, as `emitted_split`'s and the chunk bodies' do.
fn emitted_window_split(
    rows: u32,
    cols: u32,
    full_cols: u32,
    c_start: u32,
) -> ((u64, u64), serde_json::Value) {
    let a = rb(
        "t_a",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(full_cols),
    );
    let b = rb(
        "t_b",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(full_cols),
    );
    let o = rb(
        "t_o",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(full_cols),
    );
    let w = ktir_superdsc::addr::col_window_of(rows, full_cols, c_start, Df::Fp16)
        .expect("a whole-stick corner");
    let mut sid = 0i64;
    let op = assemble_pointwise_broadcast_off(
        PointwiseFacts {
            op_name: "mul",
            op_func: "multiply",
            rows: RowCount::of_token_rows(rows),
            cols: BlockCols::of_feature_cols(cols),
            inputs: &[In::col_window(&a, w).ew(), In::col_window(&b, w).ew()],
            o: &o,
            out_offset: w.into(),
        },
        &mut sid,
        None,
    );
    let v = serde_json::to_value(op.op.as_ref().expect("a descriptor")).unwrap();
    let s = &v["numWkSlicesPerDim_"];
    let key = v["dscs_"][0]
        .as_object()
        .and_then(|m| m.keys().next().cloned())
        .expect("a dsc");
    (
        (
            s["mb"].as_u64().unwrap_or(1),
            s["out"].as_u64().unwrap_or(1),
        ),
        v["dscs_"][0][&key]["primaryDsInfo_"]["OUTPUT"]["layoutDimOrder_"].clone(),
    )
}

/// ⭐ THE WIRING, through the door every dense elementwise op is assembled at:
///   * a stick-major rank-2 op (granite-2b's mulsilu, `[256, 8192]` whole) emits out:32;
///   * a column window that is a WHOLE-BLOCK shift (columns `[8192, 16384)` of a 16384-wide tensor — its
///     corner is `rows·cols` elements in) is still stick-major, so it takes the law too;
///   * a column window that is NOT a whole-block shift — granite-3.1-8b's second MLP chunk, 4608 wide at
///     column 8192 of 12800 — stated as a bare offset is presented rank-3 flat by the door, and stays
///     rows-first (the control: an offset is never taken for a window by its value), while stated as a
///     `ColWindow` (#622) it is a stick-major tensor of its own and takes the law — see
///     `a_whole_stick_column_window_takes_the_law`;
///   * a one-row op (decode) is untouched: rows-first and sticks-first agree there.
#[test]
fn the_pointwise_door_splits_a_stick_major_op_by_its_sticks() {
    assert_eq!(
        emitted_split(256, 8192, 8192, 0),
        (1, 32),
        "stick-major: out first"
    );
    assert_eq!(
        emitted_split(256, 8192, 16384, 8192),
        (1, 32),
        "a whole-block column window is stick-major"
    );
    assert_eq!(
        emitted_split(256, 4608, 12800, 8192),
        (32, 1),
        "a non-block column window stated as a bare offset is flat: rows first"
    );
    assert_eq!(
        emitted_split(1, 8192, 8192, 0),
        (1, 32),
        "one row: unchanged"
    );
}

/// ⭐⭐⭐ #622: A WHOLE-STICK COLUMN WINDOW TAKES THE LAW. granite-3.1-8b's second MLP piece, `[m, 4608]` at
/// column 8192 of the 12800-wide intermediate, stated as a [`ktir_superdsc::addr::ColWindow`], is
/// presented rank-2 stick-major (`layoutDimOrder_ [mb, out]`) and divided by the law: its 72 sticks give
/// `out:8 × mb:4` at the prefill rungs and at a 4-request decode rung (the pinned 72-stick tie-break). Its
/// first piece (column 0, a whole-block shift) is unchanged, and so is a one-row piece.
#[test]
fn a_whole_stick_column_window_takes_the_law() {
    let rank2 = serde_json::json!(["mb", "out"]);
    for rows in [256u32, 512] {
        let (split, order) = emitted_window_split(rows, 4608, 12800, 8192);
        assert_eq!(order, rank2, "[{rows}, 4608] @ 8192: presented stick-major");
        assert_eq!(
            split,
            (4, 8),
            "[{rows}, 4608] @ 8192: 72 sticks, out:8 × mb:4"
        );
        let (first, order0) = emitted_window_split(rows, 8192, 12800, 0);
        assert_eq!(order0, rank2, "[{rows}, 8192] @ 0: stick-major as before");
        assert_eq!(first, (1, 32), "[{rows}, 8192] @ 0: out:32 as before");
    }
    // A batched-decode rung: 4 rows cannot take more than a 4-way mb split, so the stick axis takes 8.
    let (split, order) = emitted_window_split(4, 4608, 12800, 8192);
    assert_eq!(order, rank2, "[4, 4608] @ 8192: presented stick-major");
    assert_eq!(split, (4, 8), "[4, 4608] @ 8192: out:8 × mb:4");
    // One row: `stickmajor` needs rows > 1, so a decode piece keeps its rank-3 form and its division.
    let (one, order1) = emitted_window_split(1, 4608, 12800, 8192);
    assert_eq!(
        order1,
        serde_json::json!(["mb", "out", "y"]),
        "one row: rank-3 as before"
    );
    assert_eq!(
        one,
        emitted_split(1, 4608, 12800, 8192),
        "one row: unchanged"
    );
}
