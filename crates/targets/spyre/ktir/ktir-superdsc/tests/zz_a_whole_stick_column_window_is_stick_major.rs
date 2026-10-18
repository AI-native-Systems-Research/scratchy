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

use ktir_superdsc::addr::{Feat, Idx, Nest, Row, col_of};
use ktir_superdsc::sdsc_abstract::{StickKind, StickLayout};
use ktir_superdsc::superdsc_opspec::Df;

const STICK: u32 = 64;

/// The element offset of `(r, c)` in a stick-major `[rows, cols]` tensor, read off the dxp law.
fn parent_off(rows: u32, full: u32, r: u32, c: u32) -> i64 {
    let nest = Nest::new(&["row", "feat"], &[rows, full], Df::Fp16);
    let v = nest.view().at(Idx::<Row>::n(r)).at(Idx::<Feat>::n(c));
    v.off()
}

/// ⭐⭐⭐ THE IDENTITY. For every row count a column-chunked elementwise op takes, every 64-multiple parent
/// width to 32768, every whole-stick corner and every whole-stick window width that fits: the window's
/// `[rows, w]` stick-major view, based at the window's corner (`col_of`, the address the door already
/// emits), reaches the parent's `(r, c + j)` for every `(r, j)` of the window.
///
/// Checked at every plane boundary and both ends of each row rather than every lane — inside one plane the
/// offset is `+ j % 64` on both sides by construction of the law, so the plane corners and the last lane are
/// where the two could disagree.
#[test]
fn every_whole_stick_window_addresses_its_parent_stick_major() {
    let rows_set: Vec<u32> = (2u32..=64).chain([71, 128, 256, 512]).collect();
    let mut cases = 0u64;
    for &rows in &rows_set {
        for full_sticks in 2u32..=512 {
            let full = full_sticks * STICK;
            // Corners and widths are sampled on a stride that still hits every remainder class of the
            // shipped widths (8192-block pieces of 12800, 15360, 21504) — the corners the front end makes.
            for c_stick in (1..full_sticks).step_by(if full_sticks > 64 { 7 } else { 1 }) {
                let c = c_stick * STICK;
                let base = col_of(rows, full, c, Df::Fp16).into_raw_elems() as i64;
                // The window's corner is a whole number of the parent's planes.
                assert_eq!(
                    base,
                    i64::from(c_stick) * i64::from(rows) * i64::from(STICK),
                    "[{rows}, {full}] corner {c}: base {base}"
                );
                for w_stick in
                    (1..=full_sticks - c_stick).step_by(if full_sticks > 64 { 5 } else { 1 })
                {
                    let w = w_stick * STICK;
                    let view = StickLayout::row_blocked(rows as usize, w as usize);
                    assert_eq!(view.kind, StickKind::RowBlocked, "[{rows}, {w}] view kind");
                    for r in [0, rows / 2, rows - 1] {
                        for j in (0..w_stick)
                            .flat_map(|p| [p * STICK, p * STICK + STICK - 1])
                            .filter(|&j| j < w)
                        {
                            let window = base + view.dev_off(r as usize, j as usize) as i64;
                            let parent = parent_off(rows, full, r, c + j);
                            assert_eq!(
                                window, parent,
                                "[{rows}, {full}] window c={c} w={w}: (r={r}, j={j}) window {window} \
                                 parent {parent}"
                            );
                        }
                    }
                    cases += 1;
                }
            }
        }
    }
    assert!(cases > 100_000, "the grid shrank to {cases} windows");
}

/// ⛔ THE SHIPPED WINDOW, every lane: granite-3.1-8b's second MLP piece, `[m, 4608]` at column 8192 of the
/// 12800-wide intermediate, at both prefill rungs and a batched-decode rung — the case #622 is about,
/// checked exhaustively rather than sampled.
#[test]
fn the_granite_8b_remainder_piece_is_its_parents_bytes() {
    let (full, c, w) = (12800u32, 8192u32, 4608u32);
    for rows in [4u32, 256, 512] {
        let base = col_of(rows, full, c, Df::Fp16).into_raw_elems() as i64;
        let view = StickLayout::row_blocked(rows as usize, w as usize);
        for r in 0..rows {
            for j in 0..w {
                assert_eq!(
                    base + view.dev_off(r as usize, j as usize) as i64,
                    parent_off(rows, full, r, c + j),
                    "rows {rows}: (r={r}, j={j})"
                );
            }
        }
    }
}

/// ⛔ A SUB-STICK CORNER IS NOT A STICK-MAJOR WINDOW: its first column sits mid-plane, so the `[rows, w]`
/// view's plane `p` would straddle two of the parent's planes. Pinned so the door can only admit a corner
/// that is a whole stick.
#[test]
fn a_sub_stick_corner_is_not_a_window_of_whole_planes() {
    let (rows, full) = (256u32, 12800u32);
    let view = StickLayout::row_blocked(rows as usize, 128);
    // c = 32: the window's lane 32 is the parent's column 64, the first lane of the NEXT plane.
    let c = 32u32;
    let base = parent_off(rows, full, 0, c);
    let window = base + view.dev_off(0, 32) as i64;
    assert_ne!(
        window,
        parent_off(rows, full, 0, c + 32),
        "a sub-stick corner addresses within one plane where the parent has crossed to the next"
    );
}
