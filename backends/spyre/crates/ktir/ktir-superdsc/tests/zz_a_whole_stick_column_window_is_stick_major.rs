// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ A WHOLE-STICK COLUMN WINDOW OF A STICK-MAJOR TENSOR IS A STICK-MAJOR TENSOR IN ITS OWN RIGHT (#622).
//!
//! A stick-major `[rows, full]` tensor is laid out `[full/64][rows][64]`: plane `p` holds columns
//! `64p .. 64p+64` of every row, `rows·64` elements long. So a `[rows, w]` window starting at a whole-stick
//! column `c` is planes `c/64 .. (c+w)/64` — each a whole `[rows, 64]` plane — with the PARENT's plane
//! stride, `rows·64`. That stride depends on `rows` alone, so a `[rows, w]` view addressed stick-major from
//! the window's corner reaches exactly the parent's bytes. This is what lets the elementwise door present
//! granite-3.1-8b's second MLP column piece (`[m, 4608]` at column 8192 of 12800) rank-2 stick-major, so the
//! work-division law reaches it.
//!
//! Kani is blocked for this crate (#618), so THIS is the proof, checked against TWO authorities that the
//! emitter does not compute: [`Nest`] (the ported dxp `SpyreTensorLayout` law, `addr::Nest::new`) for the
//! parent, and the `StickLayout` the emitter's own `per_core_addr` reads a `[rows, w]` view through.

use ktir_superdsc::addr::{Feat, Idx, Nest, Row, col_of, col_window_of};
use ktir_superdsc::sdsc_abstract::{StickKind, StickLayout};
use ktir_superdsc::superdsc_opspec::Df;

const STICK: u32 = 64;

/// The element offset of `(r, c)` in the stick-major tensor `nest` declares, read off the dxp law.
fn parent_off(nest: &Nest, r: u32, c: u32) -> i64 {
    nest.view().at(Idx::<Row>::n(r)).at(Idx::<Feat>::n(c)).off()
}

/// ⭐⭐⭐ THE IDENTITY: the window's `[rows, w]` stick-major view, based at the window's corner (`col_of`, the
/// address the door already emits), reaches the parent's `(r, c + j)`.
///
/// THE GRID (≈ 0.5 M windows × 12 address checks ≈ 6 M; a floor of 400 k windows is asserted):
///   * rows: every count 2..=64 (prefill tails, batched-decode rungs) and 71, 128, 256, 512 — 67 values;
///   * parent width: every whole-stick width to 64 sticks (4096), and 128, 200, 240, 336, 512 sticks —
///     granite-2b 8192, granite-8b 12800, gemma-4-12b 15360, gemma-4-31b 21504, granite-4.1-30b 32768;
///   * corner: EVERY whole stick of a parent to 64 sticks, from stick 0; on a wider parent the first two,
///     the last, the middle, and every tiler block boundary (multiples of 128 sticks = 8192 columns) ± 1;
///   * width at each corner: one stick, two, half the rest, and the rest — so a window ending at the
///     parent's last column and a one-stick window at the parent's last stick are always in the set.
///
/// Each window is checked at its first, middle and last plane, at the first and last lane of the first and
/// last row. Inside one plane both sides are `+ r·64 + j % 64` by the law's own construction, so the plane
/// corners are where the two could disagree; every lane of the shipped window is checked by
/// `the_granite_8b_remainder_piece_is_its_parents_bytes`. The parent [`Nest`] is built once per
/// `(rows, width)` and reused for every corner and width.
#[test]
fn every_whole_stick_window_addresses_its_parent_stick_major() {
    let rows_set: Vec<u32> = (2u32..=64).chain([71, 128, 256, 512]).collect();
    let fulls: Vec<u32> = (2u32..=64).chain([128, 200, 240, 336, 512]).collect();
    let corners_of = |full_sticks: u32| -> Vec<u32> {
        if full_sticks <= 64 {
            return (0..full_sticks).collect();
        }
        let mut cs = vec![0, 1, full_sticks / 2, full_sticks - 1];
        for b in (128..full_sticks).step_by(128) {
            cs.extend([b - 1, b, b + 1]);
        }
        cs.retain(|&c| c < full_sticks);
        cs.sort_unstable();
        cs.dedup();
        cs
    };
    let mut cases = 0u64;
    for &rows in &rows_set {
        for &full_sticks in &fulls {
            let full = full_sticks * STICK;
            let nest = Nest::new(&["row", "feat"], &[rows, full], Df::Fp16);
            for c_stick in corners_of(full_sticks) {
                let c = c_stick * STICK;
                let base = col_of(rows, full, c, Df::Fp16).into_raw_elems() as i64;
                // The window's corner is a whole number of the parent's planes.
                assert_eq!(
                    base,
                    i64::from(c_stick) * i64::from(rows) * i64::from(STICK),
                    "[{rows}, {full}] corner {c}: base {base}"
                );
                let rest = full_sticks - c_stick;
                let mut widths = vec![1, rest, rest.div_ceil(2)];
                if rest >= 2 {
                    widths.push(2);
                }
                widths.sort_unstable();
                widths.dedup();
                for w_stick in widths {
                    let w = w_stick * STICK;
                    let view = StickLayout::row_blocked(rows as usize, w as usize);
                    assert_eq!(view.kind, StickKind::RowBlocked, "[{rows}, {w}] view kind");
                    for p in [0, w_stick / 2, w_stick - 1] {
                        for r in [0, rows - 1] {
                            for j in [p * STICK, p * STICK + STICK - 1] {
                                let window = base + view.dev_off(r as usize, j as usize) as i64;
                                let parent = parent_off(&nest, r, c + j);
                                assert_eq!(
                                    window, parent,
                                    "[{rows}, {full}] window c={c} w={w}: (r={r}, j={j}) window \
                                     {window} parent {parent}"
                                );
                            }
                        }
                    }
                    cases += 1;
                }
            }
        }
    }
    assert!(cases > 400_000, "the grid shrank to {cases} windows");
}

/// ⛔ THE SHIPPED WINDOW, every lane: granite-3.1-8b's second MLP piece, `[m, 4608]` at column 8192 of the
/// 12800-wide intermediate, at both prefill rungs and a batched-decode rung — the case #622 is about,
/// checked exhaustively rather than sampled.
#[test]
fn the_granite_8b_remainder_piece_is_its_parents_bytes() {
    let (full, c, w) = (12800u32, 8192u32, 4608u32);
    for rows in [4u32, 256, 512] {
        let nest = Nest::new(&["row", "feat"], &[rows, full], Df::Fp16);
        let base = col_of(rows, full, c, Df::Fp16).into_raw_elems() as i64;
        let view = StickLayout::row_blocked(rows as usize, w as usize);
        for r in 0..rows {
            for j in 0..w {
                assert_eq!(
                    base + view.dev_off(r as usize, j as usize) as i64,
                    parent_off(&nest, r, c + j),
                    "rows {rows}: (r={r}, j={j})"
                );
            }
        }
    }
}

/// ⭐ THE TYPE'S DOOR: [`col_window_of`] mints a window at a whole-stick corner — at the same element
/// [`col_of`] gives — and refuses a sub-stick one, so a [`ktir_superdsc::addr::ColWindow`] is a whole-stick
/// window by construction.
#[test]
fn a_col_window_is_minted_only_at_a_whole_stick() {
    let (rows, full) = (512u32, 12800u32);
    let w = col_window_of(rows, full, 8192, Df::Fp16).expect("a whole-stick corner");
    assert_eq!(w.dev(), col_of(rows, full, 8192, Df::Fp16));
    assert!(
        w.is_stick_major_for(rows),
        "same rows: the window's planes are the view's"
    );
    assert!(
        !w.is_stick_major_for(rows / 2),
        "a view with other rows strides its planes differently from the parent"
    );
    for c in [1, 32, 8191, 8193] {
        assert!(
            col_window_of(rows, full, c, Df::Fp16).is_err(),
            "corner {c} is not a whole stick and must be refused"
        );
    }
}

/// ⛔ A SUB-STICK CORNER IS NOT A STICK-MAJOR WINDOW: its first column sits mid-plane, so the `[rows, w]`
/// view's plane `p` would straddle two of the parent's planes. Pinned so the door can only admit a corner
/// that is a whole stick.
#[test]
fn a_sub_stick_corner_is_not_a_window_of_whole_planes() {
    let (rows, full) = (256u32, 12800u32);
    let nest = Nest::new(&["row", "feat"], &[rows, full], Df::Fp16);
    let view = StickLayout::row_blocked(rows as usize, 128);
    // c = 32: the window's lane 32 is the parent's column 64, the first lane of the NEXT plane.
    let c = 32u32;
    let base = parent_off(&nest, 0, c);
    let window = base + view.dev_off(0, 32) as i64;
    assert_ne!(
        window,
        parent_off(&nest, 0, c + 32),
        "a sub-stick corner addresses within one plane where the parent has crossed to the next"
    );
}
