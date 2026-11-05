// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE LANDING FRAME IS THE RESHAPE LAW — [`LandingFrame::of`] over the domain grid, its
//! expected outcome derived from the LAW stated here, not from the emitter.
//!
//! The law (the reference's own `kv.index_select(0, page_ids).reshape(...)`):
//! `index_select` keeps the entry axis whole, so the gathered value is
//! `[entries, slab]` and the reshape's `to` must re-frame exactly those bytes:
//!
//! * `total = entries · slab` (a reshape moves no bytes);
//! * the frame's last dim `w` is whole fp16 sticks (a plane of the folded cache
//!   is whole sticks, stick dim last) — anything else is
//!   [`FrameError::NotWholeSticks`];
//! * each entry's plane (`slab` elements) is a whole number of the frame's
//!   `[rows, w]` blocks — a frame whose blocks cross an entry is
//!   [`FrameError::CrossesEntry`];
//! * rank ≥ 3 states its blocks: `Blocks`, and the landing registers them;
//! * rank 2 states none: the landing is the frame's own bytes when there is one
//!   entry or one stick (`Flat`), and one whole-stick permute away otherwise
//!   (`Restickify`).

use ktir_superdsc::superdsc_opspec::{
    FrameError, LandingFrame, PagePlane, StickExtent, Fp16, GatherEntries,
};
use std::num::NonZeroU32;

const STICK: u64 = 64;

fn plane(sticks: u32) -> PagePlane {
    PagePlane::of_sticks(nz(sticks))
}

fn nz(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).expect("nonzero")
}

/// The expected outcome of `LandingFrame::of`, derived from the law above.
#[derive(Debug)]
enum Expected {
    Blocks { blocks: u32, rows: u32 },
    Flat { rows: u32 },
    Restickify { t_rows: u32 },
    NotWholeSticks,
    CrossesEntry,
}

fn expect(to: &[i64], sticks: u32, entries: u32) -> Expected {
    let slab = sticks as u64 * STICK;
    let total = entries as u64 * slab;
    let w = to.last().copied().unwrap_or(0) as u64;
    if w == 0 || !w.is_multiple_of(STICK) {
        return Expected::NotWholeSticks;
    }
    if to.len() < 3 {
        // Rank 2 `[T, w]`: the block is the row — each entry's plane is
        // `slab / w` one-row blocks, so `w` must divide the plane.
        if !slab.is_multiple_of(w) {
            return Expected::CrossesEntry;
        }
        let t_rows = total / w;
        if entries > 1 && w > STICK {
            Expected::Restickify {
                t_rows: t_rows as u32,
            }
        } else {
            Expected::Flat {
                rows: (slab / w) as u32,
            }
        }
    } else {
        // Rank ≥ 3: the frame's own `[..., blocks, rows, w]`; each entry's
        // plane must tile into whole blocks and the stated total must agree.
        let rows = to[to.len() - 2] as u64;
        let block = rows * w;
        if rows == 0 || block == 0 || !slab.is_multiple_of(block) {
            return Expected::CrossesEntry;
        }
        let blocks_per_entry = slab / block;
        if total != blocks_per_entry * entries as u64 * block {
            return Expected::CrossesEntry;
        }
        Expected::Blocks {
            blocks: blocks_per_entry as u32,
            rows: rows as u32,
        }
    }
}

#[test]
fn the_landing_frame_is_the_reshape_law_over_the_domain_grid() {
    for &sticks in &[1u32, 2, 128, 256] {
        for &entries in &[1u32, 2, 33] {
            for &w in &[64i64, 128, 192, 256, 1024] {
                let slab = sticks as u64 * STICK;
                // rank 2: [T, w] with the law's own T (and, where the plane
                // has a divisor, a WRONG T to cross the entry)
                let frames: Vec<Vec<i64>> = vec![
                    vec![(entries as u64 * slab / w as u64) as i64, w],
                    vec![entries as i64, (slab / w as u64) as i64, w],
                    vec![entries as i64, 2, (slab / (w as u64 * 2)) as i64, w],
                ];
                for to in frames {
                    let exp = expect(&to, sticks, entries);
                    let got = LandingFrame::of(&to, plane(sticks), nz(entries));
                    match (exp, got) {
                        (Expected::NotWholeSticks, Err(FrameError::NotWholeSticks { .. })) => {}
                        (Expected::CrossesEntry, Err(FrameError::CrossesEntry { .. })) => {}
                        (
                            Expected::Blocks { blocks, rows },
                            Ok(LandingFrame::Blocks {
                                blocks: b,
                                rows: r,
                                w: gw,
                                ..
                            }),
                        ) => {
                            assert_eq!(b.get(), blocks, "{to:?} sticks {sticks} entries {entries}");
                            assert_eq!(r.get(), rows, "{to:?} sticks {stick_} entries {entries}", stick_ = sticks);
                            assert_eq!(gw.elems(), w as u32);
                        }
                        (
                            Expected::Flat { rows },
                            Ok(LandingFrame::Flat { rows: r, w: gw, .. }),
                        ) => {
                            assert_eq!(r.get(), rows, "{to:?} sticks {sticks} entries {entries}");
                            assert_eq!(gw.elems(), w as u32);
                        }
                        (
                            Expected::Restickify { t_rows },
                            Ok(LandingFrame::Restickify { t_rows: t, rows: per_entry, w: gw, .. }),
                        ) => {
                            assert_eq!(t.get(), t_rows, "{to:?} sticks {sticks} entries {entries}");
                            // the stored per-entry rows are the division the law guarantees exact
                            assert_eq!(
                                per_entry.get(),
                                (t_rows / entries) as u32,
                                "{to:?} sticks {sticks} entries {entries}"
                            );
                            assert_eq!(gw.elems(), w as u32);
                        }
                        (exp, got) => panic!(
                            "{to:?} sticks {sticks} entries {entries}: expected {exp:?}, got {got:?}"
                        ),
                    }
                }
            }
        }
    }
}

/// ⭐ A DIM NO LANDING CAN HOLD is its own named refusal — a negative dim, or a stated extent
/// whose product overflows — never folded into the two shape refusals.
#[test]
fn a_dim_no_landing_holds_refuses_by_name() {
    let err = LandingFrame::of(&[-4, 128], plane(2), nz(4)).expect_err("-4 rows");
    assert!(matches!(err, FrameError::NotStatable { dim: -4 }), "{err:?}");
    let huge = i64::MAX;
    let err = LandingFrame::of(&[huge, huge, 128], plane(2), nz(2))
        .expect_err("a product that overflows");
    assert!(matches!(err, FrameError::NotStatable { .. }), "{err:?}");
}

/// The width the frame lands: whole sticks by construction — a sub-stick width
/// has no `StickExtent`, so a frame stating one refuses by name.
#[test]
fn a_sub_stick_width_refuses_by_name() {
    let err = LandingFrame::of(&[4, 96], plane(2), nz(4)).expect_err("96 is not whole sticks");
    assert!(
        matches!(err, FrameError::NotWholeSticks { width: 96, stick: 64 }),
        "{err:?}"
    );
    let err = LandingFrame::of(&[2, 3, 130], plane(2), nz(2)).expect_err("130 is not whole sticks");
    assert!(matches!(err, FrameError::NotWholeSticks { width: 130, .. }), "{err:?}");
}

/// A frame whose blocks cross a gathered entry refuses by name: `index_select`
/// keeps the entry axis whole, so each entry's plane must be a whole number of
/// the frame's blocks.
#[test]
fn a_frame_that_crosses_an_entry_refuses_by_name() {
    // rank 2: a 3-stick plane (192 elements) read 256 wide — 256 does not
    // divide 192, so the frame crosses the entry.
    let err = LandingFrame::of(&[3, 256], plane(3), nz(4)).expect_err("256 does not divide the 192-element plane");
    assert!(matches!(err, FrameError::CrossesEntry { .. }), "{err:?}");
    // rank 3: rows that do not tile the plane
    let err = LandingFrame::of(&[4, 5, 64], plane(2), nz(4)).expect_err("5 rows do not tile 8192");
    assert!(matches!(err, FrameError::CrossesEntry { .. }), "{err:?}");
    // rank 3: a stated total that disagrees with the entries it re-frames
    let err = LandingFrame::of(&[8, 2, 64], plane(2), nz(4)).expect_err("8 frames of 4 entries");
    assert!(matches!(err, FrameError::CrossesEntry { .. }), "{err:?}");
}

/// ⭐ THE CHUNK ITERATOR'S OWN LAW — every total entry count from 1 to 200
/// chunks into runs of at most one index stick (32 entries), summing back to
/// the total, with contiguous bases in whole index sticks.
#[test]
fn the_chunk_iterator_covers_one_to_two_hundred() {
    let stick = 32u32; // one SenUint32 stick of entries — ENTRIES_PER_OP
    for n in 1..=200u32 {
        let mut sum = 0u32;
        let mut base = 0u32;
        let legs = GatherEntries::chunks(nz(n)).collect::<Vec<_>>();
        assert!(!legs.is_empty(), "n = {n}");
        for (i, (b, run)) in legs.into_iter().enumerate() {
            assert!(run.get() >= 1 && run.get() <= stick, "n = {n} leg {i}");
            assert_eq!(b.entries(), base, "n = {n} leg {i}: bases are contiguous");
            assert!(
                b.entries().is_multiple_of(stick),
                "n = {n} leg {i}: a base is a whole index stick in"
            );
            base += run.get();
            sum += run.get();
        }
        assert_eq!(sum, n, "n = {n}");
    }
}

/// A `GatherEntries` past one index stick is unwritable: the type's only
/// constructor is the chunk iterator, and every run it yields is ≤ the stick.
/// `StickExtent` is the same law for widths (the reuse this build stands on).
#[test]
fn the_extents_are_whole_by_construction() {
    assert_eq!(
        StickExtent::<Fp16>::new(64).unwrap().stick_count(),
        1,
        "the existing StickExtent is the width type"
    );
    assert!(StickExtent::<Fp16>::new(65).is_err(), "sub-stick widths do not exist");
}
