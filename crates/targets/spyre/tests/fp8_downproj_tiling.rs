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

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{
    SharedKernelBmmForm, matmul_opspec_off_operands,
};
use scratchy_subtile::sdsc_abstract::{MatK, MatM, MatN, MatY, QueryRowCount};
use scratchy_subtile::superdsc_opspec::{Df, Fp16, WorkPlan};

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
/// when the tiler had previously refused it. The SAME plan with its operands charged two bytes an element
/// needs strictly more trips (or does not fit at all) — the residency fix is load-bearing, not cosmetic.
/// (fp16 itself no longer takes the frontend's trips: an unbatched fp16 matmul is handed to dxp whole,
/// `MatmulStaging::Dxp`, so the two-byte charge is evaluated on the fp8 op's own plan.)
#[test]
fn charging_fp8_operands_two_bytes_costs_the_8b_down_proj_extra_trips() {
    let (m, n, k) = (31u32, 4096u32, 12800u32);
    let op = matmul_opspec_off_operands::<Fp16>(
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
        Df::Fp8,
    )
    .expect("emittable");
    let fp8 = op.time_tile.map(|t| t.count()).unwrap_or(1);
    // The per-core resident set (activation slab + weight slab + output) at two bytes an element.
    let two_bytes = |p: &WorkPlan, out_per_time: u32| {
        let (mb, kc, opt) = (
            u64::from(p.per_core_extent("mb")),
            u64::from(p.per_core_extent("in")),
            u64::from(out_per_time),
        );
        2 * (mb * kc + kc * opt + mb * opt)
    };
    match op.iter.time_tile_for_lx(two_bytes, "out") {
        Ok(tt) => {
            let fp16_charge = tt.map(|t| t.count()).unwrap_or(1);
            assert!(
                fp8 < fp16_charge,
                "one-byte operands must need FEWER trips: fp8={fp8}, charged two bytes={fp16_charge}"
            );
        }
        // Charged two bytes the same plan does not fit at any trip count — the refusal the fix removed.
        Err(_) => assert!(fp8 > 1, "the fp8 down_proj is multi-trip at K=12800"),
    }
}
