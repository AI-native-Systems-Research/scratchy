// SPDX-License-Identifier: Apache-2.0
//! ⭐ A REDUCE OPERAND WINDOW PAST ITS TENSOR REFUSES AT THE MINT — the reduce family's own
//! doors, on top of the `Win` type's law tests
//! (`zz_a_window_past_its_tensor_refuses_at_the_mint.rs`).
//!
//! The family's two mints ([`win_reduce_data`]/[`win_reduce_accum`]) state the DEVICE ACCESS
//! extents from the op's own typed dims — the data at the STICK-PADDED swept width (the pad lanes
//! ride the reduction, masked; minting at the logical width would under-bound the read the
//! descriptor itself bills), the accum at ONE STICK PER ROW (`Scale::RedStick`) — and check them
//! against whichever authority the caller holds: a [`BundleLayout`] (both doors, exactly as the
//! bake-time guard resolves the name), or the handle's own declared shape when no layout exists.
//! And the opspec layer itself refuses a `Win` whose extents disagree with the op's own iteration
//! dims — the pairing-by-convention seam the raw `(name, u32)` form left open.

use ktir_superdsc::addr::DevOff;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{
    reduce_opspec_off, win_reduce_accum, win_reduce_data,
};
use ktir_superdsc::place::PlaceId;
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
use ktir_superdsc::sdsc_abstract::{BlockCols, RowBlockedTag, RowCount, StickLayout, Stk};
use ktir_superdsc::superdsc_opspec::OpFunc;

const ROWS: u32 = 8;

fn rows(n: u32) -> RowCount {
    RowCount::of_head_major_rows(n)
}

fn cols(n: u32) -> BlockCols {
    BlockCols::of_feature_cols(n)
}

#[test]
fn the_data_mint_pads_to_the_swept_stick_width_and_refuses_past_the_handle() {
    let stk =
        Stk::<RowBlockedTag>::new("t9", StickLayout::row_blocked(8, 128)).expect("RowBlocked");
    // A stick-multiple width over the whole handle — LEGAL, and the window's columns are the
    // width itself (no pad).
    let ok = win_reduce_data(
        None,
        &stk,
        DevOff::ZERO,
        rows(ROWS),
        cols(128),
        "test: whole",
    )
    .expect("the whole block");
    assert_eq!((ok.rows(), ok.cols()), (ROWS, 128));
    // 192 pads to itself but the handle is 128 wide: 8·192 elements over an 8·128 handle — the
    // #717 class (an extent from one law over a shape from another), refused at the mint.
    let err = win_reduce_data(
        None,
        &stk,
        DevOff::ZERO,
        rows(ROWS),
        cols(192),
        "test: past end",
    )
    .expect_err("the window outruns its handle");
    assert!(err.contains("test: past end"), "{err}");
    assert!(
        err.contains("runs past the end its handle declares"),
        "{err}"
    );
    // 130 is NOT a stick multiple: the device walks the PADDED 192 — the refusal must name the
    // padded extent, not the logical width, or it would under-bound the read.
    let err = win_reduce_data(
        None,
        &stk,
        DevOff::ZERO,
        rows(ROWS),
        cols(130),
        "test: pad lanes",
    )
    .expect_err("the padded extent outruns the handle");
    assert!(err.contains("test: pad lanes"), "{err}");
    assert!(err.contains("8×192"), "names the padded extent: {err}");
}

#[test]
fn the_accum_mint_is_one_stick_per_row_whatever_the_swept_width() {
    let stk =
        Stk::<RowBlockedTag>::new("t13", StickLayout::row_blocked(8, 64)).expect("RowBlocked");
    let ok = win_reduce_accum(None, &stk, DevOff::ZERO, rows(ROWS), "test: accum")
        .expect("one stick per row");
    assert_eq!((ok.rows(), ok.cols()), (ROWS, 64));
    // A row count past the buffer's rows — the row law and the handle disagree.
    let err = win_reduce_accum(None, &stk, DevOff::ZERO, rows(ROWS + 1), "test: rows past")
        .expect_err("the accum outruns its rows");
    assert!(err.contains("test: rows past"), "{err}");
    assert!(
        err.contains("runs past the end its handle declares"),
        "{err}"
    );
}

#[test]
fn the_opspec_refuses_a_win_whose_extents_disagree_with_the_ops_own_dims() {
    let data_stk =
        Stk::<RowBlockedTag>::new("t_sc", StickLayout::row_blocked(8, 1024)).expect("RowBlocked");
    let accum_stk =
        Stk::<RowBlockedTag>::new("t_bmax", StickLayout::row_blocked(8, 64)).expect("RowBlocked");
    // A data window minted at 512 (a caller stating the extents from a different law than the
    // op's own dims): the op sweeps a 256-wide reduce — 64 padded columns of disagreement.
    let data = win_reduce_data(
        None,
        &data_stk,
        DevOff::ZERO,
        rows(ROWS),
        cols(512),
        "test: mismatch",
    )
    .expect("the window itself is in bounds");
    let accum =
        win_reduce_accum(None, &accum_stk, DevOff::ZERO, rows(ROWS), "test: mismatch").unwrap();
    let err = reduce_opspec_off(OpFunc::Max, ROWS, 256, data, accum)
        .expect_err("the window and the iteration dims disagree");
    assert!(
        err.contains("the data operand's window is 8×512 but the op sweeps 8×256"),
        "{err}"
    );
    assert!(err.contains("win_reduce_data"), "names the remedy: {err}");
    // The accum's law too: a window at two sticks per row is not the RedStick shape.
    let data2 = win_reduce_data(
        None,
        &data_stk,
        DevOff::ZERO,
        rows(ROWS),
        cols(256),
        "test: accum law",
    )
    .unwrap();
    let wide_accum_stk =
        Stk::<RowBlockedTag>::new("t_bmax", StickLayout::row_blocked(8, 128)).expect("RowBlocked");
    let wide_accum = win_reduce_data(
        None,
        &wide_accum_stk,
        DevOff::ZERO,
        rows(ROWS),
        cols(128),
        "test: accum law",
    )
    .expect("in bounds, but the wrong shape for an accum");
    let err = reduce_opspec_off(OpFunc::Max, ROWS, 256, data2, wide_accum)
        .expect_err("an accum window of two sticks per row is not the seeded convention");
    assert!(err.contains("ONE stick"), "{err}");
    assert!(err.contains("win_reduce_accum"), "names the remedy: {err}");
}

#[test]
fn a_layout_in_hand_consults_the_placement_footprint() {
    // A real placement far smaller than the handle's declared shape: the mint must refuse through
    // the SAME arm the bake-time guard would take, at expansion time rather than at bake.
    let stk =
        Stk::<RowBlockedTag>::new("t9", StickLayout::row_blocked(8, 2048)).expect("RowBlocked");
    let mut layout = BundleLayout::default();
    let name = PlaceId::Act(9).to_string();
    layout
        .ids
        .borrow_mut()
        .insert(name.clone(), PlaceId::Act(9));
    layout.placements.insert(
        9,
        TensorPlacement {
            tid: 9,
            role: SegRole::Activation,
            segment: 3,
            bank: 0,
            offset: 0,
            size: 1024,
            cover: ktir_superdsc::placement::BindCover::Full,
        },
    );
    let err = win_reduce_data(
        Some(&layout),
        &stk,
        DevOff::ZERO,
        rows(ROWS),
        cols(2048),
        "test: past placement",
    )
    .expect_err("the placement cannot hold the swept block");
    assert!(err.contains("test: past placement"), "{err}");
    assert!(err.contains("exceeds its placement footprint"), "{err}");
    assert!(err.contains("SAME layout authority"), "{err}");
}
