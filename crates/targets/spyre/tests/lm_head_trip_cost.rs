// SPDX-License-Identifier: Apache-2.0
//! How many TRIPS the vocab-wide lm-head costs as the row count grows.
//!
//! The bundle fingerprint counts EmittedOps, not trips: a time-tiled op is one op and `time`
//! launches. The lm-head is the one op in the decode suffix that always time-tiles, so its trip
//! count — not its op count — is what a batched decode step actually pays for the tail.

use scratchy_subtile::sdsc_abstract::{KernelTag, MatK, MatM, MatN, MatY, QueryRowCount, Stk};
use scratchy_target_spyre::ir::bridge::tiled_op_sdsc_op::{
    SharedKernelBmmForm, assemble_matmul_off,
};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::{concrete_trips, rb};

/// granite-3.1-2b's tail: hidden 2048 -> the padded 51200-wide logits placement.
fn lm_head_trips(m: u32) -> usize {
    let (n, k) = (51200u32, 2048u32);
    let mut sym = 0i64;
    let e = assemble_matmul_off(
        "lmhead",
        MatM::of_token_rows(m),
        MatN::of_out_features(n),
        MatK::of_in_features(k),
        MatY::unbatched(),
        // Token rows, never a request batch: the row-kind boundary yields the proven walk.
        SharedKernelBmmForm::of_attn_rows(false, QueryRowCount::of_mq(1)),
        &rb("a", m, k),
        scratchy_subtile::addr::DevOff::ZERO,
        &Stk::<KernelTag>::kernel(k as usize, n as usize, "w"),
        scratchy_subtile::addr::DevOff::ZERO,
        &rb("o", m, n),
        scratchy_subtile::addr::DevOff::ZERO,
        &mut sym,
        None,
    );
    concrete_trips(&e).len()
}

#[test]
fn report_lm_head_trip_growth() {
    let mut rows = Vec::new();
    for m in [1u32, 2, 4, 8, 16, 32] {
        rows.push((m, lm_head_trips(m)));
    }
    let base = rows[0].1;
    for (m, t) in &rows {
        println!(
            "m={m:2}  trips={t:5}  x{:.1} vs m=1",
            *t as f64 / base as f64
        );
    }
    // Not an assertion about a good number — a tripwire on SUPERLINEAR growth. Trips scale with the
    // output volume, so 8 rows costing ~8x is the shape of the thing. Well past that means each row
    // is dragging the whole weight through again, which is the opposite of why we batch.
    let (m8, t8) = *rows.last().unwrap();
    assert!(
        t8 <= base * (m8 as usize) * 2,
        "lm-head trips grew {t8} at m={m8} from {base} at m=1 — more than 2x the row count, so the \
         tail is re-streaming the weight per row"
    );
}
