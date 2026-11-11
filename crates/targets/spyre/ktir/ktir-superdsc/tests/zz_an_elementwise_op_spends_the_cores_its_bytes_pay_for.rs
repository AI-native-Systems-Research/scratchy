// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ AN ELEMENTWISE OP SPENDS THE CORES ITS BYTES PAY FOR — [`EwDivision::budget`], applied to
//! the two divisions the elementwise law chooses between, and to nothing else. The budget is read
//! from the op's own [`TileOp`] (its swept dims and format state the bytes), never re-derived from
//! a second spelling of the shape.
//!
//! MEASURED on gen1 (the card's cycle counter, 200 launches, N = 32 chained descriptors): a descriptor's
//! fixed cost grows with the cores it is spread over while its byte cost is flat for a few KiB a core.
//! A `[1, 2048]` add costs 0.51 µs on 4–8 cores against 0.80 µs on 32; 32 cores start to win at 24–32 KiB
//! of output (96 KiB moved by a binary op). Gen1's `BYTES_PER_CORE` = 3 KiB is that crossover.
//!
//! The law holds of a BANDED operand no less than a whole one: a row band's stride safety is the
//! door's pitch statement (the back-gap law,
//! `zz_a_row_band_states_its_parent_pitch_as_a_back_gap`), never the budget's, so nothing about a
//! band moves this spend.

use ktir_superdsc::device_gen::{DeviceGen, Gen1};
use ktir_superdsc::ir::island::tile_op::{TileOp, TileOpKind};
use ktir_superdsc::superdsc_opspec::{CoreBudget, Df, ItDim, WorkPlan};
use ktir_superdsc::work::{EwDivision, distribute_cores, distribute_cores_sticks_first};

/// A BINARY op's declaration: `n_operands` = the two inputs plus the output, so its bytes are
/// `3 · rows · cols · 2`.
fn op(rows: u32, cols: u32) -> TileOp {
    let dim = |name: &'static str, size: u32, is_stick: bool| ItDim {
        name,
        size,
        is_reduction: false,
        is_stick,
        df: Df::Fp16,
    };
    TileOp::new(
        TileOpKind::PointwiseOrReduce { n_operands: 3 },
        vec![
            dim("mb", rows, false),
            dim("out", cols, true),
            dim("y", 1, false),
        ],
        Df::Fp16,
    )
}

/// The budget at each measured decode shape, at gen1.
#[test]
fn the_budget_at_each_measured_shape() {
    let at = |rows, cols| EwDivision::Law.budget(&op(rows, cols)).unwrap().get();
    assert_eq!(
        at(1, 512),
        1,
        "[1, 512]: 3 KiB, one core (measured best: 1-2)"
    );
    assert_eq!(at(1, 1024), 2, "[1, 1024]: 6 KiB (measured best: 2-4)");
    assert_eq!(
        at(1, 2048),
        4,
        "[1, 2048]: 12 KiB (measured best: 4, 0.51 µs vs 0.80 on 32)"
    );
    assert_eq!(at(1, 4096), 8, "[1, 4096]: 24 KiB (measured best: 8)");
    assert_eq!(at(1, 8192), 16, "[1, 8192]: 48 KiB — 16 KiB of output");
    assert_eq!(
        at(8, 2048),
        32,
        "[8, 2048]: 96 KiB — 32 KiB of output, where 32 cores win"
    );
    assert_eq!(at(512, 2048), 32, "prefill: every core");
    assert_eq!(at(1, 64), 1, "384 bytes: never zero cores");
    assert_eq!(CoreBudget::whole().get(), <Gen1 as DeviceGen>::CORES);
}

/// The two free divisions spend by bytes; a stated division keeps every core.
#[test]
fn a_stated_division_ignores_the_byte_law() {
    let small = op(1, 2048);
    assert_eq!(EwDivision::RowsFirst.budget(&small).unwrap().get(), 4);
    assert_eq!(EwDivision::SticksFirst.budget(&small).unwrap().get(), 4);
    assert_eq!(EwDivision::Law.budget(&small).unwrap().get(), 4);
    assert_eq!(
        EwDivision::Groups(8).budget(&small).unwrap().get(),
        <Gen1 as DeviceGen>::CORES,
        "one row group a core is a correctness law of the walk's LX plan, not a cost choice"
    );
}

/// Under the byte budget each free division still produces a valid plan — whole sticks a core, the
/// product within the budget — at every decode and prefill width the ladders take, and the plan uses
/// exactly `min(budget, what the old whole-machine split used)` cores' worth or fewer.
#[test]
fn every_budgeted_plan_is_a_valid_plan() {
    let mut cases = 0;
    for rows in [1u32, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
        for sticks in 1u32..=200 {
            let cols = sticks * 64;
            let d = op(rows, cols);
            for (div, splitter) in [
                (
                    EwDivision::RowsFirst,
                    distribute_cores as fn(&[ItDim], u32) -> _,
                ),
                (EwDivision::SticksFirst, distribute_cores_sticks_first),
            ] {
                let budget = div.budget(&d).unwrap();
                let plan = WorkPlan::divide(&d.dims, budget, splitter)
                    .unwrap_or_else(|e| panic!("[{rows}, {cols}] {div:?}: {e}"));
                let whole = WorkPlan::divide(&d.dims, CoreBudget::whole(), splitter).unwrap();
                assert!(
                    plan.cores_used().get() <= budget.get(),
                    "[{rows}, {cols}] {div:?}"
                );
                assert!(
                    plan.cores_used().get() <= whole.cores_used().get(),
                    "[{rows}, {cols}] {div:?}: the budget never spends more than the whole machine did"
                );
                cases += 1;
            }
        }
    }
    assert!(cases > 3_000, "{cases}");
}

/// Wherever the bytes fill the machine the budget IS the whole machine, so those ops' plans — and so
/// their emitted descriptors — are byte-identical to the whole-machine split.
#[test]
fn an_op_whose_bytes_fill_the_machine_is_unchanged() {
    for rows in [8u32, 16, 32, 64, 128, 256, 512] {
        for sticks in [32u32, 64, 72, 128, 200] {
            let (cols, d) = (sticks * 64, op(rows, sticks * 64));
            assert_eq!(
                EwDivision::Law.budget(&d).unwrap().get(),
                32,
                "[{rows}, {cols}]"
            );
            for splitter in [
                distribute_cores as fn(&[ItDim], u32) -> _,
                distribute_cores_sticks_first,
            ] {
                let a = WorkPlan::divide(&d.dims, EwDivision::Law.budget(&d).unwrap(), splitter)
                    .unwrap();
                let b = WorkPlan::divide(&d.dims, CoreBudget::whole(), splitter).unwrap();
                assert_eq!(a.splits(), b.splits(), "[{rows}, {cols}]");
            }
        }
    }
}
