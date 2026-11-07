// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "spyre")]
//! THE DECODE LADDER AS TYPES — documentation of the law `Rung<MQ>` and the ladder walk carry.
//!
//! The load-bearing enforcement is the TYPE, not these tests: a non-ladder decode width cannot name
//! a `Rung` at all (`Rung::<3>::baked()` is a `cargo build` E0080 naming the ladder, pinned by the
//! `compile_fail,E0080` doctest on `each_batch_rung_widest_first`), the pad of a ladder width is
//! `Rung::<MQ>::PAD` computed by the compiler, and a width and its pad travel fused in one
//! `PaddedMq` with no constructor that accepts them separately. A decode width is never parsed from
//! a number: the bake walks the ladder as types. What CAN be exercised at runtime is what the walk
//! hands out, and that is what these pin.

use scratchy_subtile::sdsc_abstract::{PaddedMq, PagedKvPool, Rung, each_batch_rung_widest_first};

/// The walk hands out every batch rung, widest first, each with the pad the pad law gives: one
/// stick (64) at every rung — the single-window new-block form the card has accepted. The solo
/// decode width, 1, is its own rung and pads the same.
#[test]
fn the_walk_hands_out_every_batch_rung_widest_first_with_a_one_stick_pad() {
    let mut walked = Vec::new();
    each_batch_rung_widest_first(|width| walked.push(width));
    let widths: Vec<u32> = walked.iter().map(|w| w.mq().get()).collect();
    let mut ladder = PagedKvPool::BATCH_RUNGS.to_vec();
    ladder.reverse();
    assert_eq!(widths, ladder, "every batch rung, widest first");
    for width in walked
        .into_iter()
        .chain([PaddedMq::of_rung(Rung::<1>::baked())])
    {
        assert_eq!(
            width.pad().rows().row_axis_extent(),
            64,
            "rung {}: every baked rung pads to one stick of rows",
            width.mq().get()
        );
    }
}

/// A prefill chunk's width is a runtime quantity and takes the runtime arm of the SAME law — same
/// arithmetic, no rung: 31 pads to 64, 96 to 128, 65 to 128. The chunk arm cannot mint a `Rung`
/// (there is no value→const conversion at all), which is what keeps the wide prefill range from
/// multiplying the const-generic instantiations.
#[test]
fn a_prefill_width_takes_the_runtime_law() {
    for (w, pad) in [(31u32, 64u32), (47, 64), (64, 64), (65, 128), (96, 128)] {
        let width = PaddedMq::of_chunk(w);
        assert_eq!(width.mq().get(), w);
        assert_eq!(
            width.pad().rows().row_axis_extent(),
            pad,
            "prefill width {w}"
        );
    }
}

/// The const side agrees with the minted value: `Rung::<MQ>::PAD` and `SUB_BLOCKS` are the compiler's
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
