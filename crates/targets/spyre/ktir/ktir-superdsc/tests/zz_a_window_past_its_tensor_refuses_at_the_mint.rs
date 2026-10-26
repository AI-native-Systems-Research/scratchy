// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ A WINDOW PAST ITS TENSOR REFUSES AT THE MINT — the #717 class, unrepresentable by type.
//!
//! Until the `Win` type, an operand's offset and its name travelled separately: the builder took
//! `(&Stk, DevOff)`, split them into `name` + raw `u32` at the opspec boundary, re-paired them by
//! convention inside the descriptor layer, and the only extent check came at bake
//! (`resolve_seg_base`), reconstructing the window from the built descriptor's own dims. The card
//! walked through that seam twice in one week — the windowed matmul that read 1/8 past its own
//! tensor (caught only because two raw values happened to disagree), and the pad law that swept
//! 512 columns out of a 64-column window. Both EXTENT-side; both would have been a mint `Err`.
//!
//! These tests pin the mint law itself, per authority:
//! - [`View::win`] — the flat element bound against the nest (the layout-free door);
//! - [`View::win_at`] — the per-axis bound (a window that fits the FLAT count but straddles a row
//!   boundary reads another row's bytes as this one's elements: legal bytes, wrong data, the
//!   "clean bake, no fault" shape — only the per-axis check catches it);
//! - [`BundleLayout::win`] — the placement bound: the declared shape, the footprint (resolved by
//!   name exactly as the bake-time guard resolves it), and the disagreement between the two
//!   (a placement smaller than its tensor's declared shape is itself an `Err`).

use ktir_superdsc::addr::{DevOff, Feat, Idx, Nest, Row};
use ktir_superdsc::place::{PlaceId, SynthRole};
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
use ktir_superdsc::sdsc_abstract::{RowBlockedTag, StickLayout, Stk};
use ktir_superdsc::superdsc_opspec::Df;

const ROWS: u32 = 8;
const COLS: u32 = 2048;
/// The whole `[ROWS, COLS]` tensor, in elements.
const ELEMS: u64 = ROWS as u64 * COLS as u64;

fn a_stk() -> Stk<RowBlockedTag> {
    Stk::new("t7", StickLayout::row_blocked(ROWS as usize, COLS as usize)).unwrap()
}

fn a_nest() -> Nest {
    Nest::new(&["row", "feat"], &[ROWS, COLS], Df::Fp16)
}

/// A layout placing `t{tid}` as an activation of `size` bytes.
fn layout_placing(tid: u32, size: u64) -> BundleLayout {
    let mut layout = BundleLayout::default();
    let name = PlaceId::Act(tid).to_string();
    layout
        .ids
        .borrow_mut()
        .insert(name.clone(), PlaceId::Act(tid));
    layout.placements.insert(
        tid,
        TensorPlacement {
            tid,
            role: SegRole::Activation,
            segment: 3,
            bank: 0,
            offset: 0,
            size,
            cover: ktir_superdsc::placement::BindCover::Full,
        },
    );
    layout
}

#[test]
fn the_shape_mint_refuses_a_window_past_the_nest() {
    let stk = a_stk();
    let n = a_nest();
    // The whole tensor from the whole-tensor corner — exactly to the end, LEGAL.
    let ok = n
        .view()
        .win(&stk, DevOff::ZERO, ROWS, COLS, "test: whole")
        .unwrap();
    assert_eq!(ok.off(), DevOff::ZERO);
    assert_eq!(ok.rows(), ROWS);
    assert_eq!(ok.cols(), COLS);
    assert_eq!(ok.elems(), ELEMS);
    assert_eq!(ok.name(), "t7");
    // One column more runs past the tensor's end — the #717 class, refused at the mint.
    let err = n
        .view()
        .win(&stk, DevOff::ZERO, ROWS, COLS + 1, "test: past end")
        .unwrap_err();
    assert!(err.contains("test: past end"), "{err}");
    assert!(err.contains("past the end"), "{err}");
    assert!(err.contains(&ELEMS.to_string()), "{err}");
}

#[test]
fn the_strict_mint_refuses_a_window_past_either_axis_alone() {
    let stk = a_stk();
    let n = a_nest();
    // Columns past the column axis: corner (0, 1984) sweeping 128 — the FLAT count fits
    // (15872 + 128 = 16000 ≤ 16384 elements), so only the per-axis bound catches the straddle.
    let err = n
        .view()
        .at(Idx::<Feat>::n(1984))
        .win_at(&stk, 1, 128, "test: cols past")
        .unwrap_err();
    assert!(err.contains("test: cols past"), "{err}");
    assert!(err.contains("columns [1984, 2112)"), "{err}");
    // Rows past the row axis: corner (7, 0) sweeping 2 rows — flat fits (448 + 128 = 576), the
    // row axis does not (7 + 2 > 8).
    let err = n
        .view()
        .at(Idx::<Row>::n(ROWS - 1))
        .win_at(&stk, 2, 64, "test: rows past")
        .unwrap_err();
    assert!(err.contains("test: rows past"), "{err}");
    assert!(err.contains("rows [7, 9)"), "{err}");
    // At both axes' boundaries — corner (7, 1984), window (1, 64) — LEGAL, and the minted offset
    // is the view's own device offset.
    let view = n
        .view()
        .at(Idx::<Row>::n(ROWS - 1))
        .at(Idx::<Feat>::n(1984));
    let ok = view.win_at(&stk, 1, 64, "test: boundary").unwrap();
    assert_eq!(ok.off(), view.dev());
    assert_eq!(ok.rows(), 1);
    assert_eq!(ok.cols(), 64);
}

#[test]
fn the_placement_mint_refuses_a_window_past_the_tensor_it_names() {
    let stk = a_stk();
    // 32768 B = exactly the handle's declared shape (8×2048 fp16).
    let layout = layout_placing(7, 2 * ELEMS);
    let ok = layout
        .win(&stk, DevOff::ZERO, ROWS, COLS, "test: whole")
        .unwrap();
    assert_eq!(ok.elems(), ELEMS);
    // Past the handle's declared shape — refused before the placement is even consulted.
    let err = layout
        .win(&stk, DevOff::ZERO, ROWS, COLS + 1, "test: past end")
        .unwrap_err();
    assert!(err.contains("test: past end"), "{err}");
    assert!(err.contains("past the end"), "{err}");
}

#[test]
fn the_placement_mint_refuses_a_window_past_the_placement() {
    let stk = a_stk();
    // A placement far smaller than the window: the window fits the handle's declared shape
    // (16384 elements) but its 32768 B exceed the 1024 B placement — the aliasing refusal.
    let layout = layout_placing(7, 1024);
    let err = layout
        .win(&stk, DevOff::ZERO, ROWS, COLS, "test: past placement")
        .unwrap_err();
    assert!(err.contains("test: past placement"), "{err}");
    assert!(err.contains("exceeds its placement footprint"), "{err}");
    assert!(err.contains("SAME layout authority"), "{err}");
}

#[test]
fn a_placement_smaller_than_the_declared_shape_is_itself_a_refusal() {
    let stk = a_stk();
    let layout = layout_placing(7, 1024);
    // The window itself FITS the placement (128 B ≤ 1024 B) — the DISAGREEMENT between the
    // placement and the handle's declared shape is the defect, and it refuses anyway.
    let err = layout
        .win(&stk, DevOff::ZERO, 1, 64, "test: small window")
        .unwrap_err();
    assert!(err.contains("test: small window"), "{err}");
    assert!(err.contains("disagree about this tensor"), "{err}");
}

#[test]
fn an_undeclared_name_refuses_rather_than_panics() {
    let stk = Stk::<RowBlockedTag>::new("t404", StickLayout::row_blocked(8, 2048)).unwrap();
    let layout = BundleLayout::default();
    let err = layout
        .win(&stk, DevOff::ZERO, 1, 64, "test: undeclared")
        .unwrap_err();
    assert!(err.contains("test: undeclared"), "{err}");
    assert!(err.contains("never declared"), "{err}");
}

#[test]
fn a_declared_synth_window_is_bounded_by_its_declaration() {
    let layout = BundleLayout::default();
    let id = PlaceId::Synth {
        of: 7,
        role: SynthRole::Silu,
    };
    layout.synth(id, &[ROWS, COLS]);
    let stk = Stk::<RowBlockedTag>::new(
        id.to_string(),
        StickLayout::row_blocked(ROWS as usize, COLS as usize),
    )
    .unwrap();
    // The whole declared intermediate — legal.
    layout
        .win(&stk, DevOff::ZERO, ROWS, COLS, "test: synth whole")
        .unwrap();
    // A declaration SMALLER than the handle's shape: a whole-tensor sweep exceeds the
    // declaration's footprint —
    let short = PlaceId::Synth {
        of: 7,
        role: SynthRole::Xc,
    };
    layout.synth(short, &[ROWS, COLS / 2]);
    let big_stk = Stk::<RowBlockedTag>::new(
        short.to_string(),
        StickLayout::row_blocked(ROWS as usize, COLS as usize),
    )
    .unwrap();
    let err = layout
        .win(&big_stk, DevOff::ZERO, ROWS, COLS, "test: synth short")
        .unwrap_err();
    assert!(err.contains("test: synth short"), "{err}");
    assert!(
        err.contains("exceeds its declared synth footprint"),
        "{err}"
    );
    // — and a window that FITS the declaration still refuses on the disagreement itself.
    let err = layout
        .win(&big_stk, DevOff::ZERO, 1, 64, "test: synth disagreement")
        .unwrap_err();
    assert!(err.contains("test: synth disagreement"), "{err}");
    assert!(err.contains("disagree about this tensor"), "{err}");
}

#[test]
fn a_core_local_lx_window_refuses_an_offset_and_op_local_is_bounded() {
    let layout = BundleLayout::default();
    // CORE-LOCAL: every core's slice starts AT the pin, so a view into it has no offset to take.
    let core_id = PlaceId::Synth {
        of: 7,
        role: SynthRole::Qs,
    };
    layout.synth_lx_core_local(core_id, 256);
    let core_stk =
        Stk::<RowBlockedTag>::new(core_id.to_string(), StickLayout::row_blocked(1, 128)).unwrap();
    layout
        .win(&core_stk, DevOff::ZERO, 1, 128, "test: core-local at pin")
        .unwrap();
    let err = layout
        .win(
            &core_stk,
            DevOff::from_view_step(64),
            1,
            64,
            "test: core-local offset",
        )
        .unwrap_err();
    assert!(err.contains("test: core-local offset"), "{err}");
    assert!(err.contains("no offset to take"), "{err}");
    // OP-LOCAL: a pin that does not cover the handle's declared shape is the disagreement.
    let op_id = PlaceId::Synth {
        of: 7,
        role: SynthRole::KRep,
    };
    layout.synth_lx_op_local(op_id, &[1, 128]);
    let op_stk =
        Stk::<RowBlockedTag>::new(op_id.to_string(), StickLayout::row_blocked(1, 128)).unwrap();
    layout
        .win(&op_stk, DevOff::ZERO, 1, 128, "test: op-local fits")
        .unwrap();
    let big =
        Stk::<RowBlockedTag>::new(op_id.to_string(), StickLayout::row_blocked(1, 1024)).unwrap();
    let err = layout
        .win(&big, DevOff::ZERO, 1, 1024, "test: op-local past pin")
        .unwrap_err();
    assert!(err.contains("test: op-local past pin"), "{err}");
    assert!(err.contains("pinned-LX footprint"), "{err}");
}
