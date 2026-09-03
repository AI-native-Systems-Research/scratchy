// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "superdsc")]
//! THE DECODE LADDER AS TYPES — documentation of the law `PaddedMq::of_bundle` / `Rung<MQ>` carry.
//!
//! The load-bearing enforcement is the TYPE, not these tests: a non-ladder decode width cannot name
//! a `Rung` at all (`Rung::<31>::baked()` is a `cargo build` E0080 naming the ladder — verified by
//! a compile probe when the type landed), the pad of a ladder width is `Rung::<MQ>::PAD` computed by
//! the compiler, and a width and its pad travel fused in one `PaddedMq` with no constructor that
//! accepts them separately. What CAN be exercised at runtime is the parse boundary itself — the one
//! `match` from a runtime width to the consts — and that is what these pin.

use scratchy_subtile::sdsc_abstract::{PaddedMq, PagedKvPool, Rung};

/// Every width the ladder bakes crosses the boundary, and comes out with the pad the pad law gives:
/// one stick (64) at every rung — the single-window new-block form the card has accepted.
#[test]
fn every_baked_rung_crosses_the_boundary_with_a_one_stick_pad() {
    for w in std::iter::once(1).chain(PagedKvPool::BATCH_RUNGS) {
        let width = PaddedMq::of_bundle(w, true)
            .unwrap_or_else(|| panic!("rung {w} is baked — the dispatch must name it"));
        assert_eq!(width.mq().get(), w, "the width is the rung's own");
        assert_eq!(
            width.pad().rows().row_axis_extent(),
            64,
            "rung {w}: every baked rung pads to one stick of rows"
        );
    }
}

/// A decode width the ladder does not bake is REFUSED at the boundary — `None`, which the emitter
/// turns into a loud bake error naming the width. Padding it silently is the bug this closes: the
/// pad law would accept any integer, but a decode bundle at an unbaked width describes a launch no
/// rung selection can ever pick.
#[test]
fn an_unbaked_decode_width_is_refused() {
    for w in [0u32, 3, 5, 6, 7, 9, 15, 17, 31, 33, 64, 96] {
        assert!(
            PaddedMq::of_bundle(w, true).is_none(),
            "decode width {w} is not a baked rung and must not cross the boundary"
        );
    }
}

/// A prefill chunk's width is a runtime quantity and takes the runtime arm of the SAME law — same
/// arithmetic, no rung: 31 pads to 64, 96 to 128, 65 to 128. The chunk arm cannot mint a `Rung`
/// (there is no value→const conversion outside the dispatch's literal arms), which is what keeps
/// the wide prefill range from multiplying the const-generic instantiations.
#[test]
fn a_prefill_width_takes_the_runtime_law() {
    for (w, pad) in [(31u32, 64u32), (47, 64), (64, 64), (65, 128), (96, 128)] {
        let width = PaddedMq::of_bundle(w, false).expect("a prompt chunk width always parses");
        assert_eq!(width.mq().get(), w);
        assert_eq!(
            width.pad().rows().row_axis_extent(),
            pad,
            "prefill width {w}"
        );
    }
}

/// The const side agrees with the boundary: `Rung::<MQ>::PAD` and `SUB_BLOCKS` are the compiler's
/// answers, and they are the same numbers the minted value carries. (`SUB_BLOCKS == 1` is not just
/// checked here — it is a compile-time assert of the type, so a ladder widened past one stick stops
/// building before any test runs.)
#[test]
fn the_const_arithmetic_is_the_minted_value() {
    assert_eq!(Rung::<1>::PAD, 64);
    assert_eq!(Rung::<32>::PAD, 64);
    assert_eq!(Rung::<8>::SUB_BLOCKS, 1);
    assert_eq!(Rung::<8>::WINDOW.index(), 0);
    let width = PaddedMq::of_rung(Rung::<8>::baked());
    assert_eq!(width.mq().get(), 8);
    assert_eq!(width.pad().rows().row_axis_extent(), Rung::<8>::PAD);
}
