// SPDX-License-Identifier: Apache-2.0
//! granite-3.1-8b's fp8 `down_proj` TIME-TILES and granite-3.1-2b's does not.
//!
//! This is a code path the working model never enters. 2b's `K=8192` reduction fits the scratchpad in
//! one trip; 8b's `K=12800` does not, so it is emitted as multiple trips whose output base advances by
//! an affine stride. That op also only bakes at all because the LX budget was corrected to charge fp8
//! operands one byte instead of two -- before that the tiler refused it outright -- so nothing about
//! it has ever executed on hardware.
//!
//! Recorded as a test rather than a comment because it is the sharpest remaining difference between the
//! model that works and the model that degrades, and because if a future change makes 2b time-tile too,
//! that is a fact worth failing on rather than discovering downstream.

use scratchy_subtile::sdsc_abstract::{MatK, MatM, MatN, MatY, QueryRowCount};
use scratchy_subtile::superdsc_opspec::{Df, Fp16};
use scratchy_target_spyre::ir::bridge::tiled_op_sdsc_op::{
    SharedKernelBmmForm, matmul_opspec_off_operands,
};

fn time_tiled(m: u32, n: u32, k: u32) -> bool {
    matmul_opspec_off_operands::<Fp16>(
        MatM::of_token_rows(m),
        MatN::of_out_features(n),
        MatK::of_in_features(k),
        MatY::unbatched(),
        // Token rows, never a request batch: the row-kind boundary yields the proven walk.
        SharedKernelBmmForm::of_attn_rows(false, QueryRowCount::of_mq(1)),
        "a",
        "w",
        "o",
        0,
        0,
        0,
        Df::Fp8,
    )
    .expect("the fp8 down_proj must be emittable at both shapes")
    .time_tile
    .is_some()
}

#[test]
fn only_the_8b_fp8_down_proj_needs_more_than_one_trip() {
    assert!(
        !time_tiled(31, 2048, 8192),
        "granite-3.1-2b: K=8192 fits the scratchpad in one trip"
    );
    assert!(
        time_tiled(31, 4096, 12800),
        "granite-3.1-8b: K=12800 does not, so it is multi-trip"
    );
}

/// The one-byte operand accounting changes how the same op is TILED, which is what made it emittable
/// when the tiler had previously refused it. Both formats emit today (the refusal was at the older,
/// coarser out_per_time), but fp16 needs strictly more trips for the same work — the residency fix is
/// load-bearing, not cosmetic.
#[test]
fn charging_fp8_operands_two_bytes_costs_the_8b_down_proj_extra_trips() {
    let (m, n, k) = (31u32, 4096u32, 12800u32);
    let trips = |df| {
        matmul_opspec_off_operands::<Fp16>(
            MatM::of_token_rows(m),
            MatN::of_out_features(n),
            MatK::of_in_features(k),
            MatY::unbatched(),
            SharedKernelBmmForm::of_attn_rows(false, QueryRowCount::of_mq(1)),
            "a",
            "w",
            "o",
            0,
            0,
            0,
            df,
        )
        .expect("emittable")
        .time_tile
        .map(|t| t.count())
        .unwrap_or(1)
    };
    let (fp8, fp16) = (trips(Df::Fp8), trips(Df::Fp16));
    assert!(
        fp8 < fp16,
        "one-byte operands must need FEWER trips: fp8={fp8} fp16={fp16}"
    );
}
