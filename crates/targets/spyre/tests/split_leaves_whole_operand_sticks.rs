// SPDX-License-Identifier: Apache-2.0
//! EVERY CORE GETS WHOLE OPERAND STICKS, ON BOTH STICK AXES — read off the EMITTED
//! `numWkSlicesPerDim_`, at the extents that measured the fault.
//!
//! dxp SCHEDULES a per-core tile, and a tile whose `out` (N) or `in` (K) slice is not a whole stick of
//! the OPERAND's format has no schedule: `DtException: There must be at least one valid candidate.`
//! (`L3DlOpsScheduler.cpp:1375`). The failure is a BUILD failure of the whole group, so one bad split
//! refuses ops that are individually fine — which is why the law belongs on the split, not on a retry.
//!
//! ⛔ THE `out` HALF WAS ALREADY ENFORCED AND THE `in` HALF WAS NOT, and that asymmetry is what this
//! pins. gemma-4's own one-op-per-dxp-compile table (branch `worktree-spyre-gemma4`, `a1cde93f`):
//!
//!   | k     | n     | in | out | k/in | /128 | n/out | /128 | dxp     |
//!   | 3840  | 15360 |  1 |  30 | 3840 | 30   |  512  |  4   | ok      |
//!   | 15360 |  3840 |  5 |   6 | 3072 | 24   |  640  |  5   | ok      |
//!   | 3840  | 15360 |  4 |   8 |  960 | 7.5  | 1920  | 15   | refused |
//!
//! ⭐⭐⭐ THE BASIS IS PER **AXIS**, NOT PER OP, and that asymmetry is the whole subtlety:
//!   * `in` (K) is the fp8 ACTIVATION's stick — 128 lanes. A per-core slice of 960 is 7.5 of them.
//!   * `out` (N) is the packed fp8 KERNEL's stick — 64 lanes, the same as fp16.
//! ⛔ AND GRANITE-8b IS THE WITNESS FOR THE N SIDE, not an argument: its working emission splits
//! n=12800 into 1600 per core, which is 25 fp16 sticks and only 12.5 fp8 ones. That bundle bakes and
//! decodes coherently, so charging `out` at 128 would reject a division the card demonstrably runs.
//! Measuring both axes at one basis is therefore wrong in BOTH directions — too strict on N, too lax
//! on K, and the too-lax direction is the one that reaches dxp as a DtException.
//!
//! ⛔ AND GRANITE MUST NOT MOVE: both granite stems' emit fingerprints are byte-identical with and
//! without the law (verified by `sdsc_emit_fingerprint.sh` when it landed), because their extents
//! already divide evenly. A future change that makes granite's splits depend on this is a regression
//! this file cannot see — the fingerprint diff is the check for that.

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::matmul_opspec_off_operands;
use scratchy_subtile::sdsc_abstract::{MatK, MatM, MatN, MatY};
use scratchy_subtile::superdsc_opspec::Df;

/// The split the emitter chose for one matmul, as `{axis: cores}`.
fn splits(m: u32, n: u32, k: u32, df: Df) -> std::collections::BTreeMap<String, u32> {
    let spec = matmul_opspec_off_operands::<scratchy_subtile::superdsc_opspec::Fp16>(
        MatM::of_token_rows(m),
        MatN::of_out_features(n),
        MatK::of_in_features(k),
        MatY::unbatched(),
        SharedKernelBmmForm::of_kv_cache_write(),
        "t_a",
        "t_w",
        "t_o",
        0,
        0,
        0,
        df,
    )
    .expect("a plain 2-D matmul opspec");
    spec.iter
        .splits()
        .iter()
        .map(|(k, v)| ((*k).to_string(), *v))
        .collect()
}

/// THE LAW, at gemma-4's extents and granite's, in both operand formats: whatever the cost model picks,
/// every core's `out` and `in` slice is a whole number of the OPERAND's sticks.
#[test]
fn every_split_leaves_whole_operand_sticks_on_both_stick_axes() {
    // (m, n, k) — gemma-4's projections (hidden 3840, intermediate 15360) and granite's (2048/8192,
    // 4096/12800), at prefill row counts where the cost model is free to K-split.
    for &(m, n, k) in &[
        (21u32, 15360u32, 3840u32), // gemma-4 gate/up — the pair that measured `in=4` => 7.5 sticks
        (21, 3840, 15360),          // gemma-4 down
        (21, 4096, 3840),           // gemma-4 q (the 32-core refusal in the original table)
        (21, 2048, 3840),
        (31, 12800, 4096), // granite-8b gate/up
        (31, 4096, 12800), // granite-8b down
        (31, 8192, 2048),  // granite-2b gate/up
        (31, 2048, 8192),  // granite-2b down
        (1, 12800, 4096),  // decode
    ] {
        for df in [Df::Fp8, Df::Fp16] {
            let s = splits(m, n, k, df);
            // `in` at the OPERAND's basis, `out` at the KERNEL's — see the module note.
            for (axis, extent, lanes) in [
                ("out", n, Df::Fp16.elems_per_stick()),
                ("in", k, df.elems_per_stick()),
            ] {
                let split = s.get(axis).copied().unwrap_or(1).max(1);
                assert_eq!(
                    extent % split,
                    0,
                    "m={m} n={n} k={k} {df:?}: {axis} split {split} does not divide {extent} — the \
                     remainder core gets a short slice, which is the same sub-stick tile dxp refuses"
                );
                assert_eq!(
                    (extent / split) % lanes,
                    0,
                    "m={m} n={n} k={k} {df:?}: {axis} split {split} leaves {} per core, which is not \
                     whole {lanes}-lane sticks (DtException L3DlOpsScheduler:1375)",
                    extent / split,
                );
            }
        }
    }
}

/// AND THE FAULT IS EXPRESSIBLE, so the test above is not vacuous: `in = 4` on gemma-4's k=3840 is
/// exactly 7.5 fp8 sticks per core. If the splitter ever proposes it again, the assertions fire.
#[test]
fn the_refused_division_really_is_sub_stick() {
    let (k, split, lanes) = (3840u32, 4u32, Df::Fp8.elems_per_stick());
    assert_eq!(
        k % split,
        0,
        "4 divides 3840 evenly — the extent is not the problem"
    );
    assert_ne!(
        (k / split) % lanes,
        0,
        "3840/4 = 960 must NOT be whole 128-lane sticks (it is 7.5), or this table entry has drifted"
    );
}
