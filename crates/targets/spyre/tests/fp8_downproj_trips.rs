// SPDX-License-Identifier: Apache-2.0
//! The 8b `down_proj` shape's TRIPS must tile its output exactly — disjoint, and covering it.
//!
//! ⭐ An unbatched fp16 matmul is now handed to dxp whole (`MatmulStaging::Dxp`), so the dense
//! `down_proj` is ONE descriptor and has no trips. The advance below is the one every matmul that KEEPS
//! the frontend's LX trips runs, checked on the shipped one that does at this K=12800 contraction: the
//! fp8 `down_proj` (`MatmulStaging::FrontendTrips` for fp8 operands), emitted exactly as the fp8
//! lowering emits it (`ktir_matmul_fp8`'s `fq_mm`). A `y`-batched fp16 matmul at this contraction is no
//! longer a form the build admits — 200 sticks under a `y` batch is the shape measured incoherent on
//! granite-3.1-8b — and that refusal is pinned here too, so the old fixture cannot quietly return.
//! A wrong advance writes trips on top of each other or past the tensor, and the symptom would be a
//! partially-correct output every layer — output that stays fluent and drifts.
//!
//! The check is on the ADDRESSES the emitter bakes, via `concrete_trips`, not on a re-derivation of
//! them: every trip's OUTPUT start is read back out of its dataspace.
//!
//! VERDICT: the advance is CORRECT. The output steps by `out_per_time * m * 2` bytes per trip, the
//! kernel by its own device stride, and the activation not at all — it carries no `out` dim. Recorded
//! so this path is not re-suspected: it is the only op that is multi-trip at granite-3.1-8b and
//! single-trip at 2b, which makes it an obvious thing to blame for 8b-only misbehaviour, and it is not
//! at fault.
//!
//! (A first version of this test read the minimum address across the core map, which silently picked
//! a DIFFERENT dataspace and reported every trip starting at 0. Keying on the output's `ldsIdx_` is
//! what makes it a gate rather than a false alarm.)

use ktir_superdsc::emit::{EmittedOp, emit_sdsc_tiled, rb};
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{
    SharedKernelBmmForm, assemble_matmul, matmul_opspec_off_operands, try_assemble_matmul_seeded,
};
use scratchy_subtile::sdsc_abstract::{KernelTag, MatK, MatM, MatN, MatY, QueryRowCount, Stk};
use scratchy_subtile::superdsc_opspec::{Df, Fp16, Role, SdscFoldSet};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::concrete_trips;

/// The fp16 `down_proj` at `m × n × k`, unbatched — handed to dxp whole (`MatmulStaging::Dxp`).
fn fp16_down_proj(m: u32, n: u32, k: u32) -> EmittedOp {
    assemble_matmul(
        "dp",
        m,
        n,
        k,
        1,
        &rb("a", m, k),
        &Stk::<KernelTag>::kernel(k as usize, n as usize, "w"),
        &rb("o", m, n),
        None,
    )
}

/// The fp8 `down_proj` at `m × n × k`, built the way the fp8 lowering builds its `fq_mm`: the fp16
/// geometry with both operands' LX residency charged one byte (`operand_df`), the non-output operands
/// then stamped `Df::Fp8` on the wire, and the op emitted on its own plan's cores. fp8 keeps the
/// frontend's LX trips (`MatmulStaging::of`), so this is the shipped matmul whose advance runs.
fn fp8_down_proj(m: u32, n: u32, k: u32) -> EmittedOp {
    let mut spec = matmul_opspec_off_operands::<Fp16>(
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
    .expect("the fp8 down_proj is emittable");
    for arg in &mut spec.args {
        if !matches!(arg.view().role, Role::Output) {
            arg.set_df(Df::Fp8);
        }
    }
    let folds = SdscFoldSet::new(spec.iter.cores_used());
    emit_sdsc_tiled("dp", &spec, &folds, &mut 0, None).expect("the fp8 down_proj emits")
}

/// Core-0 start address of each trip's OUTPUT dataspace, in elements.
fn trip_output_starts(e: &EmittedOp) -> Vec<i64> {
    concrete_trips(e)
        .iter()
        .filter_map(|op| {
            op.dscs_.iter().find_map(|dm| {
                dm.iter().find_map(|(_, d)| {
                    d.scheduleTree_
                        .iter()
                        .find(|a| a.ldsIdx_ == 2) // arg 2 is the OUTPUT by the OpSpec's type invariant
                        .and_then(|a| {
                            a.startAddressCoreCorelet_
                                .data_
                                .iter()
                                .min_by_key(|(k, _)| k.to_string())
                                .and_then(|(_, v)| v.trim().parse::<i64>().ok())
                        })
                })
            })
        })
        .collect()
}

/// The trips must ADVANCE — a stride of zero writes every trip at the same place, which is the
/// silently-wrong output the emitter's own comment warns about at "design risk #4".
#[test]
fn the_8b_down_proj_trips_advance_and_do_not_alias() {
    // fp16 and unbatched it is dxp-staged, one descriptor: nothing to advance.
    assert_eq!(
        trip_output_starts(&fp16_down_proj(31, 4096, 12800)).len(),
        1
    );
    // The same K=12800 contraction at two kernel batches, the shape this test drove before the
    // one-stick refusal: 200 sticks under a `y` batch. It is a build error now, naming why.
    let refused = try_assemble_matmul_seeded(
        "dp",
        31,
        4096,
        12800,
        2,
        &rb("a", 31, 12800),
        &Stk::<KernelTag>::kernel(12800, 4096, "w"),
        &rb("o", 31, 4096),
        &mut 0,
        None,
    )
    .err()
    .expect("a y-batched multi-stick contraction must be refused");
    assert!(
        refused.contains(
            "a y-batched (batch 2) shared-kernel matmul contracts 12800 elements, 200 sticks"
        ) && refused.contains("card-validated only at a one-stick contraction"),
        "the refusal names the shape and the reason: {refused}"
    );
    // The fp8 down_proj keeps the frontend's trips, and it is the shipped one at this contraction.
    let starts = trip_output_starts(&fp8_down_proj(31, 4096, 12800));
    assert!(
        starts.len() > 1,
        "this shape must be multi-trip, got {} trip(s)",
        starts.len()
    );
    let mut seen = std::collections::BTreeSet::new();
    for (i, s) in starts.iter().enumerate() {
        assert!(
            seen.insert(*s),
            "trip {i} starts at {s}, where an earlier trip already writes"
        );
    }
    // Strictly increasing by a uniform step: an affine advance, not a scatter.
    let step = starts[1] - starts[0];
    assert!(
        step > 0,
        "the per-trip advance must be positive, got {step}"
    );
    for w in starts.windows(2) {
        assert_eq!(
            w[1] - w[0],
            step,
            "the advance must be uniform across trips: {starts:?}"
        );
    }
}

/// The 2b shape is single-trip, so it exercises none of this — stated so the contrast is a test and
/// not a comment that can drift.
#[test]
fn the_2b_down_proj_is_a_single_trip() {
    assert_eq!(trip_output_starts(&fp16_down_proj(31, 2048, 8192)).len(), 1);
    assert_eq!(trip_output_starts(&fp8_down_proj(31, 2048, 8192)).len(), 1);
}
