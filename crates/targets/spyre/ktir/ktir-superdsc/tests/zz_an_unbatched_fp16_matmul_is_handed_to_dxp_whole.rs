// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ AN UNBATCHED fp16 MATMUL IS HANDED TO dxp WHOLE — [`MatmulStaging`]: one descriptor over the cost
//! model's per-core slab, no frontend LX trips, at every row count. torch-spyre's form, and on card the
//! frontend's trips (each re-staging the same activation slab) cost granite-3.1-8b ~9 % of its prompt and
//! ~7–11 % of its decode step.
//!
//! ⛔ AND ONLY INSIDE THE EAR ENVELOPE ([`WHOLE_DXP_EAR_KERNEL_SLAB_LIMIT`]): past it dxp refuses the
//! whole-dxp descriptor (`MutableAddrSplitting.cpp:814`, the gemma-4 31b lm_head's measured 176.2
//! MB/core), and the matmul takes the frontend's trips — the staging the flip-everything probe baked
//! `RC=0` and ran coherent on card for the same shape.
//!
//! [`MatmulStaging`]: ktir_superdsc::superdsc_opspec::MatmulStaging
//! [`WHOLE_DXP_EAR_KERNEL_SLAB_LIMIT`]: ktir_superdsc::superdsc_opspec::WHOLE_DXP_EAR_KERNEL_SLAB_LIMIT

use std::collections::BTreeMap;

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::matmul_opspec;
use ktir_superdsc::superdsc_opspec::{Df, ItDim, KernelSlab, MatmulStaging, MaxCores, WorkPlan};

/// A matmul-shaped plan with a STATED division — the law test's vehicle. The dims are
/// `matmul_dims`' own {mb, out, in} at batch 1, and the splits are injected so a slab can be
/// placed on either side of the envelope without leaning on the cost model's choice.
fn slab_plan(m: u32, n: u32, k: u32, splits: &[(&'static str, u32)]) -> WorkPlan {
    let dims = [
        ItDim {
            name: "mb",
            size: m,
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16,
        },
        ItDim {
            name: "out",
            size: n,
            is_reduction: false,
            is_stick: true,
            df: Df::Fp16,
        },
        ItDim {
            name: "in",
            size: k,
            is_reduction: true,
            is_stick: true,
            df: Df::Fp16,
        },
    ];
    let stated: BTreeMap<&'static str, u32> = splits.iter().copied().collect();
    WorkPlan::divide(&dims, MaxCores::<32>, move |_, _| stated.clone()).unwrap()
}

/// The law itself: only an unbatched fp16 matmul INSIDE the EAR envelope is dxp-staged.
#[test]
fn only_an_unbatched_fp16_matmul_inside_the_ear_envelope_is_staged_by_dxp() {
    // Inside the envelope: granite-8b's o_proj class of slab (2048×8192×2 B = 33.6 MB/core).
    let inside = slab_plan(64, 8192, 4096, &[("in", 2)]);
    assert_eq!(
        MatmulStaging::of(1, Df::Fp16, KernelSlab::of_plan(&inside, Df::Fp16)),
        MatmulStaging::Dxp
    );
    // Past the envelope: a stated division whose slab is the 31b lm_head's measured 176.2
    // MB/core — the slab dxp refused with the EAR overflow. (The cost model's own division of
    // that two-row form — `mb`×2 × `out`×16, `in` whole — is pinned by the wiring test below;
    // this one states different factors to the same product, so the law does not lean on the
    // cost model's choice.)
    let over = slab_plan(2, 262144, 5376, &[("in", 2), ("out", 8)]);
    assert_eq!(
        KernelSlab::of_plan(&over, Df::Fp16).bytes(),
        2688 * 32768 * 2,
        "the 31b lm_head's slab is the measured 176.2 MB/core — if this moved, the envelope's \
         boundary moved with it"
    );
    assert_eq!(
        MatmulStaging::of(1, Df::Fp16, KernelSlab::of_plan(&over, Df::Fp16)),
        MatmulStaging::FrontendTrips,
        "an unbatched fp16 matmul past the EAR limit must take the frontend's trips — dxp \
         refuses the whole-dxp descriptor there (MutableAddrSplitting.cpp:814)"
    );
    // A `y`-batched matmul keeps the card-proven frontend trips, envelope or not.
    for batch in [2, 4, 8, 32] {
        assert_eq!(
            MatmulStaging::of(batch, Df::Fp16, KernelSlab::of_plan(&inside, Df::Fp16)),
            MatmulStaging::FrontendTrips,
            "batch {batch}"
        );
    }
    // fp8's untiled form is unmeasured on card, so fp8 keeps the trips.
    assert_eq!(
        MatmulStaging::of(1, Df::Fp8, KernelSlab::of_plan(&inside, Df::Fp8)),
        MatmulStaging::FrontendTrips
    );
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

/// The gemma-4 lm_heads STRADDLE the whole-dxp EAR envelope, at the same 262144-wide vocabulary.
/// The cost model divides the vocabulary `out`×16 and spends its other core factor on `in` at one
/// row (`in`×2) or on `mb` itself at two (`mb`×2, leaving `in` WHOLE — the per-core KERNEL slab
/// doubles with it): the 31b's two-row form is past the envelope (the slab dxp refused at bake,
/// `MutableAddrSplitting.cpp:814`), its one-row form and the 12b's and 26b-A4B's forms at both row
/// counts are inside it (the card-proven whole-dxp forms). The failing descriptor ran at mb=2.
#[test]
fn gemma4_lm_heads_split_at_the_whole_dxp_ear_envelope() {
    // (model, rows, hidden = K, expected per-core kernel slab, expected frontend trips)
    let heads = [
        // 31b: mb=1 divides in×2 × out×16 ⇒ 2688×16384×2 = 88.1 MB/core (inside the envelope);
        // mb=2 divides mb×2 × out×16 with `in` whole ⇒ 5376×16384×2 = 176.2 MB/core — the
        // refused slab (the failing bundle's input was 2×5376).
        ("gemma-4-31b", 1u32, 5376u32, 2688u64 * 16384 * 2, false),
        ("gemma-4-31b", 2, 5376, 5376 * 16384 * 2, true),
        // 12b: 1920×16384×2 = 62.9 MB at mb=1; 3840×16384×2 = 125.8 MB at mb=2 — the largest
        // slab any of its whole-dxp forms routes, card-proven coherent.
        ("gemma-4-12b", 1, 3840, 1920 * 16384 * 2, false),
        ("gemma-4-12b", 2, 3840, 3840 * 16384 * 2, false),
        // 26b-A4B: 1408×16384×2 = 46.1 MB at mb=1; 2816×16384×2 = 92.3 MB at mb=2 — bakes green
        // whole-dxp at both.
        ("gemma-4-26b-a4b", 1, 2816, 1408 * 16384 * 2, false),
        ("gemma-4-26b-a4b", 2, 2816, 2816 * 16384 * 2, false),
    ];
    for (model, m, k, want_slab, want_trips) in heads {
        let op = matmul_opspec(m, 262144, k, 1, "a", "w", "o")
            .unwrap_or_else(|e| panic!("{model} at {m} rows: {e}"));
        let slab = KernelSlab::of_plan(&op.iter, Df::Fp16).bytes();
        assert_eq!(
            slab, want_slab,
            "{model} at {m} rows: the per-core kernel slab is {} B, not the measured {} B — the \
             cost model's division of the lm_head moved, and the envelope's boundary with it",
            slab, want_slab
        );
        assert_eq!(
            op.time_tile.is_some(),
            want_trips,
            "{model} lm_head [{m}x{k}]·[{k}x262144] at {slab} B/core — {}",
            if want_trips {
                "must take the frontend's LX-fit trips (past the EAR envelope)"
            } else {
                "stays whole-dxp (inside the EAR envelope)"
            }
        );
        // Either staging covers the whole vocabulary: the time tile divides `out` in TIME,
        // never trimming the spatial division.
        assert_eq!(
            op.iter.per_core_extent("out") * op.iter.split_of("out"),
            262144,
            "{model} at {m} rows: the op must cover all 262144 output columns"
        );
    }
}
