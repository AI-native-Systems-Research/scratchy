// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ AN UNBATCHED fp16 MATMUL IS HANDED TO dxp WHOLE — [`MatmulStaging`]: one descriptor over the cost
//! model's per-core slab, no frontend LX trips, at every row count. torch-spyre's form, and on card the
//! frontend's trips (each re-staging the same activation slab) cost granite-3.1-8b ~9 % of its prompt and
//! ~7–11 % of its decode step.
//!
//! [`MatmulStaging`]: ktir_superdsc::superdsc_opspec::MatmulStaging

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::matmul_opspec;
use ktir_superdsc::superdsc_opspec::{Df, MatmulStaging};

/// The law itself: only an unbatched fp16 matmul is dxp-staged.
#[test]
fn only_an_unbatched_fp16_matmul_is_staged_by_dxp() {
    assert_eq!(MatmulStaging::of(1, Df::Fp16), MatmulStaging::Dxp);
    // A `y`-batched matmul keeps the card-proven frontend trips.
    for batch in [2, 4, 8, 32] {
        assert_eq!(
            MatmulStaging::of(batch, Df::Fp16),
            MatmulStaging::FrontendTrips,
            "batch {batch}"
        );
    }
    // fp8's untiled form is unmeasured on card, so fp8 keeps the trips.
    assert_eq!(MatmulStaging::of(1, Df::Fp8), MatmulStaging::FrontendTrips);
}

/// granite-3.1-8b's projections — which the frontend cut into 14–16 LX trips at 256 prompt rows and 2–7 at
/// one decode row — each emit ONE untiled op under the cost model's own split.
#[test]
fn granite_8b_projections_emit_one_untiled_op_at_every_row_count() {
    // (name, n, k): gate/up padded to full occupancy, down, o_proj / q, k / v.
    let projections = [
        ("gate", 14336, 4096),
        ("down", 4096, 12800),
        ("o", 4096, 4096),
        ("kv", 1024, 4096),
    ];
    for m in [1, 8, 128, 256] {
        for (name, n, k) in projections {
            let op = matmul_opspec(m, n, k, 1, "a", "w", "o")
                .unwrap_or_else(|e| panic!("{name} at {m} rows: {e}"));
            assert!(
                op.time_tile.is_none(),
                "{name} [{m}x{k}]·[{k}x{n}] is time-tiled into {:?} trips — an unbatched fp16 matmul is \
                 handed to dxp whole",
                op.time_tile.map(|t| t.count())
            );
            assert!(
                op.tiled_symbols.is_empty(),
                "{name} at {m} rows declares tiled symbols"
            );
            // The whole extent is one op: the plan divides the full `out` across its cores, untrimmed.
            assert_eq!(
                op.iter.per_core_extent("out") * op.iter.split_of("out"),
                n,
                "{name} at {m} rows: the op must cover all {n} output columns"
            );
        }
    }
}
