// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ A ONE-ROW MATMUL FILLS EVERY CORE — the decode projections' exact work division, pinned at
//! the door the dense lowering builds them through ([`matmul_opspec`]), for every granite projection.
//!
//! At one row a matmul is a pure weight stream, and the card reaches its ~149 GB/s HBM read ceiling only
//! when all 32 cores read. The out-only split stranded two granite-8b families: gate/up (200 output
//! sticks — no divisor in 26..=32, so 25 cores at 126 GB/s) and k/v (16 sticks, 16 cores at 89 GB/s).
//! [`one_row_core_fill`] fills the rest with a whole-stick reduction factor. MEASURED in the 8b fp16
//! decode on card (RB per-CB device time): gate/up 857 → 730 µs, k/v 97 → 59 µs, the step −14 ms at
//! 123 and at 10,086 tokens of context.
//!
//! The triples below are the COMPLETE split of each op (every axis, `in` included), so a change that
//! moves any one of them — the fill, the cost model, the stick basis — fails here by name.
//!
//! [`matmul_opspec`]: ktir_superdsc::ir::bridge::tiled_op_sdsc_op::matmul_opspec
//! [`one_row_core_fill`]: ktir_superdsc::work::one_row_core_fill

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::matmul_opspec;

/// `(mb, out, in)` cores of the one-row matmul `[1, k] · [k, n]`, read off the plan the door built.
fn split_triple(n: u32, k: u32) -> (u32, u32, u32) {
    let spec = matmul_opspec(1, n, k, 1, "t_a", "t_w", "t_o").expect("a one-row projection");
    let s = &spec.iter;
    (s.split_of("mb"), s.split_of("out"), s.split_of("in"))
}

/// granite-3.1-8b fp16 decode: every projection on all 32 cores, at its exact split.
#[test]
fn every_granite_8b_decode_projection_runs_on_all_32_cores() {
    // (name, k = contraction, n = output width as the device holds it, the exact (mb, out, in))
    for (name, k, n, want) in [
        ("q_proj", 4096u32, 4096u32, (1u32, 32u32, 1u32)),
        ("k_proj", 4096, 1024, (1, 16, 2)),
        ("v_proj", 4096, 1024, (1, 16, 2)),
        ("o_proj", 4096, 4096, (1, 32, 1)),
        ("gate_proj", 4096, 12800, (1, 8, 4)),
        ("up_proj", 4096, 12800, (1, 8, 4)),
        ("down_proj", 12800, 4096, (1, 32, 1)),
        // The lm_head's vocab 49155 is padded to 800 sticks (= 32 x 25) by `DeviceWidth`.
        ("lm_head", 4096, 51200, (1, 32, 1)),
    ] {
        let got = split_triple(n, k);
        assert_eq!(
            got, want,
            "granite-8b {name} [{k} -> {n}]: (mb, out, in) = {got:?}, want {want:?}"
        );
        assert_eq!(got.0 * got.1 * got.2, 32, "{name}: every core reads");
    }
}

/// granite-3.1-2b fp16 decode: k/v is 512 wide (8 sticks), which the out-only split put on a quarter
/// of the machine (MEASURED 84.0 GB/s at out8 vs 144.9 at in4·out8, one projection per launch).
#[test]
fn every_granite_2b_decode_projection_runs_on_all_32_cores() {
    for (name, k, n, want) in [
        ("q_proj", 2048u32, 2048u32, (1u32, 32u32, 1u32)),
        ("k_proj", 2048, 512, (1, 8, 4)),
        ("v_proj", 2048, 512, (1, 8, 4)),
        ("o_proj", 2048, 2048, (1, 32, 1)),
        ("gate_proj", 2048, 8192, (1, 32, 1)),
        ("up_proj", 2048, 8192, (1, 32, 1)),
        ("down_proj", 8192, 2048, (1, 32, 1)),
    ] {
        let got = split_triple(n, k);
        assert_eq!(
            got, want,
            "granite-2b {name} [{k} -> {n}]: (mb, out, in) = {got:?}, want {want:?}"
        );
    }
}

/// The fp8 W8A8 contraction (`fq_mm`) takes the SAME door at the same `::<Fp16>` geometry with 1-byte
/// operands, so the fill reaches it unchanged — and there it ALSO removes a time tile: gate/up at
/// out25 ran as 2 LX-fit trips (MEASURED 425.8 µs, 123.1 GB/s) and in4·out8 runs as 1 (359.8 µs, 145.7).
#[test]
fn every_granite_8b_fp8_decode_contraction_runs_on_all_32_cores() {
    use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{
        SharedKernelBmmForm, matmul_opspec_off_operands,
    };
    use ktir_superdsc::sdsc_abstract::{MatK, MatM, MatN, MatY, QueryRowCount};
    use ktir_superdsc::superdsc_opspec::Fp16;
    for (name, k, n, want, trips) in [
        ("k_proj", 4096u32, 1024u32, (1u32, 16u32, 2u32), 1u32),
        ("gate_proj", 4096, 12800, (1, 8, 4), 1),
        ("down_proj", 12800, 4096, (1, 32, 1), 1),
        ("q_proj", 4096, 4096, (1, 32, 1), 1),
    ] {
        let spec = matmul_opspec_off_operands::<Fp16>(
            MatM::of_token_rows(1),
            MatN::of_out_features(n),
            MatK::of_in_features(k),
            MatY::unbatched(),
            SharedKernelBmmForm::of_attn_rows(false, QueryRowCount::of_mq(1)),
            "t_a",
            "t_w",
            "t_o",
            0,
            0,
            0,
            Df::Fp8,
        )
        .expect("a one-row fp8 contraction");
        let s = &spec.iter;
        let got = (s.split_of("mb"), s.split_of("out"), s.split_of("in"));
        assert_eq!(
            got, want,
            "granite-8b fp8 {name} [{k} -> {n}]: (mb, out, in)"
        );
        let t = spec.time_tile.as_ref().map(|t| t.count()).unwrap_or(1);
        assert_eq!(t, trips, "granite-8b fp8 {name}: LX-fit trips");
    }
}

/// A prompt chunk keeps the cost model's split: the fill is the ONE-ROW law, and a K-split at 256 rows
/// MEASURED 2.1-2.6x slower than the cost model's mb·out split (one projection per launch: gate
/// 834 → 1775 µs, down 738 → 1832 µs), because every reduction core then re-streams the activation.
#[test]
fn a_prompt_chunk_keeps_the_cost_models_split() {
    let spec =
        matmul_opspec(256, 12800, 4096, 1, "t_a", "t_w", "t_o").expect("a prompt projection");
    assert_eq!(
        spec.iter.split_of("in"),
        1,
        "gate at m=256 stays unsplit on `in`"
    );
    assert_eq!(
        (spec.iter.split_of("mb"), spec.iter.split_of("out")),
        (4, 8),
        "gate at m=256 keeps the cost model's mb4·out8"
    );
}
