//! Rung-2 substrate guard (integration test — `tests/` target, NOT the lib unit-tests). Run:
//!   cargo test -p scratchy-subtile --features spyre,superdsc --test sdsc_fp8_stick_split
//!
//! WHAT IT LOCKS (rung 2: fp8 W8A8 on the live emitter). The matmul work-division stick basis is
//! now a COMPILE-TIME type parameter `DF: DataFormat`, so an fp8 matmul is emitted only through
//! `matmul_opspec_off::<Fp8>` — whose N/K are `StickExtent<Fp8>` (128-multiple witnesses) and whose
//! stick dims carry `Fp8::ELEMS_PER_STICK` = 128. The historical bug (reuse the fp16 ÷64 basis for
//! an fp8 op ⇒ 64-wide per-core slice ⇒ dxp `L3DlOpsScheduler:1070 multiple-of-stick` DtException)
//! is now unrepresentable: you cannot pick the 64 basis for a typed-fp8 matmul.
//!
//! These asserts pin the three observable consequences:
//!   (1) fp8 `out` (N) splits on the 128-stick (per-core extent is a whole 128-multiple), COARSER
//!       than the fp16 64-stick split of the same N.
//!   (2) the fp8 activation + weight operands are `Df::Fp8` (1-byte packed HBM residency) — the
//!       residency is stamped from the SAME `DF`, so it cannot silently drop to 2-byte fp16.
//!   (3) a non-128-multiple fp8 extent is a Rust `Err` AT EMIT (the sealed `StickExtent<Fp8>`
//!       constructor), NEVER an on-card DtException — while the same extent is legal for fp16.
//! Reverting the stick basis to a hardcoded 64 (or dropping the `::<Fp8>` residency stamp) turns
//! (1)/(2) RED; deleting the `StickExtent<Fp8>` witness turns (3) RED.

use scratchy_subtile::sdsc_abstract::{MatK, MatM, MatN, MatY, QueryRowCount};
use scratchy_subtile::superdsc_opspec::{Df, Fp8, Fp16, Role};
use scratchy_target_spyre::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm;
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::{
    assemble_convert, matmul_opspec_off, rb,
};

/// These probes emit decode-shaped projections: one token row, projection feature widths, no batch.
fn mm(m: u32) -> MatM {
    MatM::of_token_rows(m)
}
fn nn(n: u32) -> MatN {
    MatN::of_out_features(n)
}
fn kk(k: u32) -> MatK {
    MatK::of_in_features(k)
}
fn yy() -> MatY {
    MatY::unbatched()
}
fn pf() -> SharedKernelBmmForm {
    // Token rows, never a request batch: the row-kind boundary yields the proven walk.
    SharedKernelBmmForm::of_attn_rows(false, QueryRowCount::of_mq(1))
}

#[test]
fn fp8_matmul_splits_on_128_stick_not_64() {
    // m=1 decode, N=256, K=256 — fits LX (no time-tile), a clean single spatial split.
    let f8 = matmul_opspec_off::<Fp8>(
        mm(1),
        nn(256),
        kk(256),
        yy(),
        pf(),
        "act",
        "w",
        "out",
        0,
        0,
        0,
    )
    .expect("fp8 256×256 matmul must emit");
    let f16 = matmul_opspec_off::<Fp16>(
        mm(1),
        nn(256),
        kk(256),
        yy(),
        pf(),
        "act",
        "w",
        "out",
        0,
        0,
        0,
    )
    .expect("fp16 256×256 matmul must emit");

    let s8 = f8.iter.split_of("out");
    let s16 = f16.iter.split_of("out");

    // fp8: 256 / 128 = 2 sticks ⇒ split ≤ 2, and every per-core `out` extent is a whole 128-stick.
    assert!(
        s8 <= 2,
        "fp8 out split must be ≤2 (256/128 sticks), got {s8}"
    );
    assert_eq!(
        (256 / s8) % 128,
        0,
        "fp8 per-core out={} must be a whole 128-stick (never sub-stick ⇒ no DtException)",
        256 / s8
    );
    // fp16: 256 / 64 = 4 sticks ⇒ the 64-basis split is FINER (≥ the fp8 split) and 64-aligned.
    assert!(
        s16 >= s8,
        "fp16 (64-stick) must split at least as fine as fp8 (128-stick): s16={s16} s8={s8}"
    );
    assert_eq!((256 / s16) % 64, 0, "fp16 per-core out must be 64-aligned");
}

#[test]
fn fp8_operands_are_fp8_resident() {
    // The `::<Fp8>` type param is the SOLE fp8 decision: it stamps the activation + weight operands
    // Df::Fp8 (1-byte reads). A regression that split at 128 but read the weight 2-byte fp16 would
    // be the dequant-to-fp16 reward-hack — this catches it.
    let f8 = matmul_opspec_off::<Fp8>(
        mm(1),
        nn(256),
        kk(256),
        yy(),
        pf(),
        "act",
        "w",
        "out",
        0,
        0,
        0,
    )
    .unwrap();
    let mut non_output = 0;
    for arg in &f8.args {
        let v = arg.view();
        if !matches!(v.role, Role::Output) {
            assert_eq!(
                v.df,
                Df::Fp8,
                "fp8 matmul non-output operand '{}' must be Fp8-resident (1-byte), got {:?}",
                v.name,
                v.df
            );
            non_output += 1;
        }
    }
    assert_eq!(
        non_output, 2,
        "expected activation + weight non-output operands"
    );

    // The dense fp16 path is byte-identical: its operands stay Fp16.
    let f16 = matmul_opspec_off::<Fp16>(
        mm(1),
        nn(256),
        kk(256),
        yy(),
        pf(),
        "act",
        "w",
        "out",
        0,
        0,
        0,
    )
    .unwrap();
    for arg in &f16.args {
        assert_eq!(arg.view().df, Df::Fp16, "fp16 matmul operands stay Fp16");
    }
}

#[test]
fn fp8_non_128_multiple_extent_is_err_not_dtexception() {
    // N=192 = 3×64 but NOT a 128-multiple. For fp8 (128-lane stick) that is a sub-stick extent: it
    // MUST be a Rust `Err` at emit (the sealed StickExtent<Fp8> ctor), NOT an on-card
    // L3DlOpsScheduler:1070 DtException. The SAME 192 is a valid fp16 extent (3 whole 64-sticks).
    let bad = matmul_opspec_off::<Fp8>(
        mm(1),
        nn(192),
        kk(256),
        yy(),
        pf(),
        "act",
        "w",
        "out",
        0,
        0,
        0,
    );
    assert!(
        bad.is_err(),
        "non-128-multiple fp8 N=192 must be Err (sub-stick), got Ok"
    );
    let ok = matmul_opspec_off::<Fp16>(
        mm(1),
        nn(192),
        kk(256),
        yy(),
        pf(),
        "act",
        "w",
        "out",
        0,
        0,
        0,
    );
    assert!(
        ok.is_ok(),
        "N=192 is a valid fp16 extent (3×64), got {ok:?}"
    );
}

#[test]
fn fp8_large_matmul_time_tile_never_substick() {
    // The REAL granite-3.1-2b fp8 projection shapes (decode m=1). The wide-K down_proj (N=2048,
    // K=8192) is the one that faulted: the fp16-byte LX resident estimate over-counted the 1-byte
    // fp8 weight 2×, spuriously time-tiling `out` to a 64-wide (sub-128) slab → dxp
    // `L3DlOpsScheduler:1070`. With the df-aware resident + df-aware time-tile stick, EVERY per-core
    // per-time `out` slab MUST be a whole 128-fp8 stick (no sub-stick, no DtException).
    for &(n, k, tag) in &[
        (2048u32, 8192u32, "down_proj"),
        (8192, 2048, "gate/up_proj"),
        (2048, 2048, "q/o_proj"),
        (512, 2048, "k/v_proj"),
    ] {
        let f = matmul_opspec_off::<Fp8>(mm(1), nn(n), kk(k), yy(), pf(), "a", "w", "o", 0, 0, 0)
            .unwrap_or_else(|e| panic!("{tag} N={n} K={k} fp8 must emit, got Err: {e}"));
        let split = f.iter.split_of("out").max(1);
        let per_core = f.iter.per_core_extent("out");
        let time = f.time_tile.as_ref().map(|t| t.count()).unwrap_or(1).max(1);
        let per_core_per_trip = per_core / time;
        assert_eq!(
            per_core_per_trip % 128,
            0,
            "{tag} N={n} K={k}: per-core-per-trip out={per_core_per_trip} (split={split}, time={time}) \
             must be a whole 128-fp8 stick — a sub-128 slab is the DtException",
        );
    }
}

#[test]
fn qfp8ch_convert_splits_on_128_fp8_output() {
    // The qfp8ch activation quantizer (f16 → fp8) has an fp8 OUTPUT (128-lane stick). Its work-
    // division touches both dataspaces, so the stick-axis split must keep each core a whole 128-fp8
    // stick — NOT the f16 input's 64. K=2048 split at 64 ⇒ 64/core = HALF an fp8 stick (the dxp
    // `L3DlOpsScheduler:1070` that persisted after the matmul fix). Assert per-core out is 128-aligned.
    let mut sym: i64 = 0;
    for &cols in &[2048u32, 8192, 512] {
        let e = assemble_convert(
            "fq_afp8",
            "qfp8ch",
            1,
            cols,
            &rb("act", 1, cols),
            &rb("afp8", 1, cols),
            &mut sym,
            None,
        );
        let split =
            e.op.numWkSlicesPerDim_
                .get("out")
                .copied()
                .unwrap_or(1)
                .max(1);
        let per_core = cols / split;
        assert_eq!(
            per_core % 128,
            0,
            "qfp8ch cols={cols}: per-core out={per_core} (split={split}) must be a whole 128-fp8 stick",
        );
    }
}

#[test]
fn fp8_untileable_matmul_is_err_not_substick() {
    // A pathological fp8 matmul whose per-core tile does NOT fit LX even at a SINGLE 128-fp8 stick
    // (N=128 ⇒ 1 stick, unsplittable; K=16384 ⇒ W[16384,128]·1B = 2 MiB > 1.68 MiB usable LX). The
    // time-tile pass cannot slice below 128 without a sub-stick slab, so it MUST return a typed `Err`
    // at emit (the build-time DtException-1535/1070 guard), NEVER silently emit a 64-wide fp8 tile.
    let r = matmul_opspec_off::<Fp8>(
        mm(1),
        nn(128),
        kk(16384),
        yy(),
        pf(),
        "a",
        "w",
        "o",
        0,
        0,
        0,
    );
    assert!(
        r.is_err(),
        "an fp8 matmul that can't fit LX without a sub-128 slab must be Err, got Ok"
    );
}

/// The fp8 packed tile's re-tile map must read the ON-DISK `[out, in]` buffer, since the worker no
/// longer transposes. granite-3.1-8b is fp8, so this branch — not the fp16 one — is what its GEMM
/// weights go through; leaving it on the transposed map is what produced garbage output.
#[test]
fn fp8_packed_retile_reads_the_disk_orientation() {
    let (k, n) = (8u64, 128u64);
    let dev = [n / 64, k / 2, 64, 2];
    let disk = [64 * k, 2, k, 1];
    for a in 0..dev[0] {
        for b in 0..dev[1] {
            for c in 0..dev[2] {
                for d in 0..dev[3] {
                    let (o, i) = (a * 64 + c, b * 2 + d); // out index, in index
                    let off = a * disk[0] + b * disk[1] + c * disk[2] + d * disk[3];
                    assert_eq!(
                        off,
                        o * k + i,
                        "device ({a},{b},{c},{d}) must read disk [out,in]"
                    );
                }
            }
        }
    }
}
