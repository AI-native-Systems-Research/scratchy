// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ AN ELEMENTWISE OP SPENDS THE CORES ITS BYTES PAY FOR — [`CoreBudget::for_bytes`], applied by
//! [`EwDivision::budget`] to the two divisions the elementwise law chooses between, and to nothing else.
//!
//! MEASURED on gen1 (the card's cycle counter, 200 launches, N = 32 chained descriptors): a descriptor's
//! fixed cost grows with the cores it is spread over while its byte cost is flat for a few KiB a core.
//! A `[1, 2048]` add costs 0.51 µs on 4–8 cores against 0.80 µs on 32; 32 cores start to win at 24–32 KiB
//! of output (96 KiB moved by a binary op). Gen1's `BYTES_PER_CORE` = 3 KiB is that crossover.

use ktir_superdsc::device_gen::{DeviceGen, Gen1};
use ktir_superdsc::superdsc_opspec::{CoreBudget, Df, ItDim, WorkPlan};
use ktir_superdsc::work::{EwDivision, distribute_cores, distribute_cores_sticks_first};

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

/// A binary fp16 op's bytes: two inputs and the output, `[rows, cols]` each.
fn binary_bytes(rows: u32, cols: u32) -> u64 {
    3 * rows as u64 * cols as u64 * 2
}

/// The budget at each measured decode shape, at gen1.
#[test]
fn the_budget_at_each_measured_shape() {
    let at = |rows, cols| CoreBudget::<Gen1>::for_bytes(binary_bytes(rows, cols)).get();
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
    assert_eq!(
        CoreBudget::<Gen1>::for_bytes(0).get(),
        1,
        "never zero cores"
    );
    assert_eq!(
        CoreBudget::<Gen1>::whole().get(),
        <Gen1 as DeviceGen>::CORES
    );
}

/// The law is stated over the generation, not over 32: a 16-core generation's budget clamps at 16,
/// and a cheaper core spends more of them for the same bytes.
#[test]
fn the_law_is_the_generations() {
    struct Narrow;
    // Every fact but the three the law reads is Gen1's — the law reads only these.
    impl DeviceGen for Narrow {
        const DEVICE_OPS: [ktir_core::ddl_binds::DeviceOpRow;
            ktir_core::ddl_binds::DEVICE_OP_COUNT] = <Gen1 as DeviceGen>::DEVICE_OPS;
        const CORES: u32 = 16;
        const CORELETS: u32 = <Gen1 as DeviceGen>::CORELETS;
        const LX_CAPACITY_BYTES: u64 = <Gen1 as DeviceGen>::LX_CAPACITY_BYTES;
        const MAX_SIDS: u32 = <Gen1 as DeviceGen>::MAX_SIDS;
        const RDMA_MESSAGE_CAP_BYTES: u64 = <Gen1 as DeviceGen>::RDMA_MESSAGE_CAP_BYTES;
        const NUM_SEGMENTS: usize = <Gen1 as DeviceGen>::NUM_SEGMENTS;
        const MAX_SEGMENT_BYTES: u64 = <Gen1 as DeviceGen>::MAX_SEGMENT_BYTES;
        const FLIT_BYTES: u64 = <Gen1 as DeviceGen>::FLIT_BYTES;
        const BYTES_PER_CORE: u64 = 1024;
        const UTIL_FLOOR_CORES: u32 = 4;
        const DL16TOFP32_PERMUTATION: [u8; 64] = <Gen1 as DeviceGen>::DL16TOFP32_PERMUTATION;
    }
    let at = |rows, cols| CoreBudget::<Narrow>::for_bytes(binary_bytes(rows, cols)).get();
    assert_eq!(at(1, 512), 3, "3 KiB at 1 KiB a core");
    assert_eq!(at(1, 2048), 12);
    assert_eq!(at(8, 2048), 16, "clamped at the generation's cores");
    assert_eq!(CoreBudget::<Narrow>::whole().get(), 16);
    // And the split product is held to the budget's count, whatever the splitter asks for.
    let plan = WorkPlan::divide(
        &dims(1, 2048),
        CoreBudget::<Narrow>::whole(),
        distribute_cores,
    )
    .expect("a 32-stick op on 16 cores");
    assert_eq!(plan.cores_used().get(), 16);
}

/// The two free divisions spend by bytes; a stated division keeps every core.
#[test]
fn a_stated_division_ignores_the_byte_law() {
    let small = binary_bytes(1, 2048);
    assert_eq!(EwDivision::RowsFirst.budget::<Gen1>(small).get(), 4);
    assert_eq!(EwDivision::SticksFirst.budget::<Gen1>(small).get(), 4);
    assert_eq!(EwDivision::Law.budget::<Gen1>(small).get(), 4);
    assert_eq!(
        EwDivision::Groups(8).budget::<Gen1>(small).get(),
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
            let d = dims(rows, cols);
            for (div, splitter) in [
                (
                    EwDivision::RowsFirst,
                    distribute_cores as fn(&[ItDim], u32) -> _,
                ),
                (EwDivision::SticksFirst, distribute_cores_sticks_first),
            ] {
                let budget = div.budget::<Gen1>(binary_bytes(rows, cols));
                let plan = WorkPlan::divide(&d, budget, splitter)
                    .unwrap_or_else(|e| panic!("[{rows}, {cols}] {div:?}: {e}"));
                let whole = WorkPlan::divide(&d, CoreBudget::<Gen1>::whole(), splitter).unwrap();
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
            let (cols, d) = (sticks * 64, dims(rows, sticks * 64));
            let bytes = binary_bytes(rows, cols);
            assert_eq!(
                CoreBudget::<Gen1>::for_bytes(bytes).get(),
                32,
                "[{rows}, {cols}]"
            );
            for splitter in [
                distribute_cores as fn(&[ItDim], u32) -> _,
                distribute_cores_sticks_first,
            ] {
                let a =
                    WorkPlan::divide(&d, CoreBudget::<Gen1>::for_bytes(bytes), splitter).unwrap();
                let b = WorkPlan::divide(&d, CoreBudget::<Gen1>::whole(), splitter).unwrap();
                assert_eq!(a.splits(), b.splits(), "[{rows}, {cols}]");
            }
        }
    }
}
