// SPDX-License-Identifier: Apache-2.0
//! ⛔ A `y`-BATCHED SHARED-KERNEL MATMUL CONTRACTS ONE STICK, OR THE BUILD REFUSES IT.
//!
//! MEASURED on granite-3.1-8b (head dim 128): the score leg `y`-batched over `in = 128`, two sticks, had
//! every per-core address correct and decoded incoherently from the second token. The card-validated
//! batched form has one stick on the contraction. So the shared emission refuses any other, at
//! `matmul_opspec_split`, the one raw door every shared-kernel matmul passes. A wider contraction is split
//! by slab or runs unbatched.

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::matmul::matmul_opspec;

#[test]
fn a_two_stick_contraction_under_a_y_batch_is_refused_naming_why() {
    // The 8b score leg's shape: gqa = 4 batched heads, one row, the whole head dim (128 = 2 sticks).
    let err = matmul_opspec(1, 512, 128, 4, "q", "kt", "sc")
        .expect_err("a y-batched two-stick contraction must be refused");
    assert!(err.contains("2 sticks"), "{err}");
    assert!(err.contains("incoherent"), "{err}");
}

#[test]
fn a_one_stick_contraction_under_a_y_batch_and_any_contraction_unbatched_still_build() {
    // The validated batched form: one stick of contraction (the cache write, the query re-layout).
    assert!(matmul_opspec(1, 64, 64, 4, "q", "w", "o").is_ok());
    // Unbatched, the whole head dim contracts in one op (the per-head and query-group scores).
    assert!(matmul_opspec(4, 512, 128, 1, "q", "kt", "sc").is_ok());
    assert!(matmul_opspec(32, 2048, 128, 1, "q", "kt", "sc").is_ok());
}
