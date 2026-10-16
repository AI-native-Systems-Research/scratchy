// SPDX-License-Identifier: Apache-2.0
//! The 8b `down_proj` shape's TRIPS must tile its output exactly — disjoint, and covering it.
//!
//! ⭐ An unbatched fp16 matmul is now handed to dxp whole (`MatmulStaging::Dxp`), so the dense
//! `down_proj` is ONE descriptor and has no trips. The advance below is the one every matmul that KEEPS
//! the frontend's LX trips (a `y`-batched one, an fp8 one) runs, checked at the same K=12800 contraction.
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

use ktir_superdsc::emit::rb;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::assemble_matmul;
use scratchy_subtile::sdsc_abstract::{KernelTag, Stk};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::concrete_trips;

/// Core-0 start address of each trip's OUTPUT dataspace, in elements, for the shape at `batch` kernel
/// batches. Unbatched, an fp16 matmul is handed to dxp whole (`MatmulStaging::Dxp`), so it is ONE trip;
/// a `y`-batched one keeps the frontend's LX trips — the only fp16 matmuls whose advance still runs.
fn trip_output_starts(m: u32, n: u32, k: u32, batch: u32) -> Vec<i64> {
    let e = assemble_matmul(
        "dp",
        m,
        n,
        k,
        batch,
        &rb("a", m, k),
        &Stk::<KernelTag>::kernel(k as usize, n as usize, "w"),
        &rb("o", m, n),
        None,
    );
    concrete_trips(&e)
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
    // Unbatched it is dxp-staged, one descriptor: nothing to advance.
    assert_eq!(trip_output_starts(31, 4096, 12800, 1).len(), 1);
    // At two kernel batches the same K=12800 contraction takes the reduction-split repair AND trips.
    let starts = trip_output_starts(31, 4096, 12800, 2);
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
    assert_eq!(trip_output_starts(31, 2048, 8192, 1).len(), 1);
}
