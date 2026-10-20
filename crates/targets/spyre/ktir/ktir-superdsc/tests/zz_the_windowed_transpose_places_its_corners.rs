// SPDX-License-Identifier: Apache-2.0
//! THE WINDOWED TRANSPOSE — the grid-trip form a paged Kᵀ plane needs, pinned at the door.
//!
//! # THE SHAPE
//!
//! An unrolled trip of the Kᵀ store lowers ONE page part per trip: it loads `[page, hd]` rows of a
//! head's `[cap, hd]` K-plane slice and stores the transposed `[hd, page]` into the matching
//! COLUMNS of the `[hd, cap]` Kᵀ scratch. The door
//! (`lower_ktir_to_superdsc::transpose`) takes the pointwise arms' two bases for it:
//!
//! * a program whose access tiles TILE its view keeps the door's original whole-tensor form —
//!   guards verbatim, zero corners, byte-identical;
//! * a single window at a corner (the grid trip) emits the WINDOW at `rc_of` corners over each
//!   region's OWN view, with the physical row count declared on the input when its row window is
//!   partial — the torch-spyre `arg.device_size` law — and rewritten into `N_`, the residency
//!   extent the card's stick-blocked reconstruction reads (the gathered-leg and windowed-matmul
//!   measurements: pinning `maxDimSizes_` was inert, rewriting `N_` fixed the stride).
//!
//! # THE CONTROL TABLE
//!
//! Same discipline as every door here: a test that only pins "it emits" is satisfied by a door
//! that emits ALWAYS. Each refusal below moves exactly one variable from the accepting case:
//!
//! | case | moved | verdict | what it pins |
//! |---|---|---|---|
//! | (a) | — | emits, corners 4096/8192, `N_.y_` = the VIEW's 256 | the placement itself |
//! | (b) | the store's column window (64 → 128) | refused: "TRANSPOSED" | the correspondence law |
//! | (c) | the window's row extent (64 → 32) | refused: window stick law | the historically fatal shape, per window |
//! | (d) | the corners (64/64 → 32/32) | refused: "stick-aligned" | each side's stick-axis corner |
//! | (e) | the input's cover ((64,128) → (0,128)) | refused: "neither" | one rectangle per side |
//! | (f) | — (whole views) | emits, corners 0 | the folded form is unchanged |
//!
//! (a)→(f) is the byte-identity anchor: the whole-view program that always worked emits exactly as
//! it did before the windowed arm existed (zero offsets, no declarations), which is what keeps the
//! new arm from silently re-addressing every existing bundle.
//!
//! # THE ARITHMETIC THE CORNERS PIN
//!
//! Input `[256, 128]` sticked on its last dim (hd): device nest `[128/64, 256, 64]`, so the corner
//! `(row 64, col 0)` sits at `64·64 = 4096` elements. Output `[128, 256]` sticked on its last dim
//! (cap): nest `[256/64, 128, 64]`, so the corner `(row 0, col 64)` sits at `1·(128·64) = 8192`.
//! One `rc_of` over ONE view cannot produce both — the swap keeps the two nests distinct — which
//! is why the door derives each side's corner over that side's own view.

use ktir_superdsc::emit::lower_ktir_to_superdsc::{Region, transpose};

/// A window stated per axis as `(start, len)` — the two facts the door reads on each side.
struct Win {
    rows: (u32, u32),
    cols: (u32, u32),
}

/// A parameter's region with ONE window — `Win`'s rectangle — and covers equal to that window,
/// the shape `regions()` reports for an unrolled trip whose single access tile is the window.
fn windowed(tid: u32, v_rows: u32, v_cols: u32, win: Win, is_out: bool) -> Region {
    let (r_start, r_len) = win.rows;
    let (c_start, c_len) = win.cols;
    Region {
        tid,
        v_rows,
        v_cols,
        r_start,
        c_start,
        r_len,
        c_len,
        r_cover: (r_start, r_start + r_len),
        c_cover: (c_start, c_start + c_len),
        is_out,
        is_fp8: false,
    }
}

/// A whole-view region — the standalone door's own form.
fn whole(tid: u32, rows: u32, cols: u32, is_out: bool) -> Region {
    windowed(
        tid,
        rows,
        cols,
        Win {
            rows: (0, rows),
            cols: (0, cols),
        },
        is_out,
    )
}

/// The door over a region list, at the whole-tensor layout the standalone door takes.
fn lowered(r: &[Region]) -> Result<usize, String> {
    let mut sid = -1i64;
    match transpose("tr", r, &mut sid, None) {
        Ok(ops) => Ok(ops.len()),
        Err(e) => Err(e.message),
    }
}

/// (a) THE PLACEMENT ITSELF — the page-part window emits with both corners placed and the view's
/// row count in `N_`.
///
/// The physical-row declaration is the one fact this test can read off the emitted descriptor two
/// ways: the input binding's `offset_elems` is the corner the emission placed, and `N_.y_` carries
/// the VIEW's 256 rather than the tile's 64 — the residency the card reconstructs the stick-group
/// stride from. A door that dropped either fact still emits a well-formed descriptor that
/// misaddresses on card, which is exactly what these assertions exist to catch at bake.
#[test]
fn a_page_part_transpose_places_both_corners_and_declares_the_view() {
    // Input t7: `[256, 128]` view, window rows [64, 128), columns whole.
    let a = windowed(
        7,
        256,
        128,
        Win {
            rows: (64, 64),
            cols: (0, 128),
        },
        false,
    );
    // Output t9: `[128, 256]` view (the transpose), window rows whole, columns [64, 128).
    let out = windowed(
        9,
        128,
        256,
        Win {
            rows: (0, 128),
            cols: (64, 64),
        },
        true,
    );
    let mut sid = -1i64;
    let ops = transpose("tr", &[a, out], &mut sid, None)
        .unwrap_or_else(|e| panic!("the page-part window must lower: {}", e.message));
    assert_eq!(ops.len(), 1, "one relayout descriptor per window");
    let op = &ops[0];

    // The corners, on the bindings — see the module doc for the arithmetic.
    let in_off = op
        .arg_bindings
        .iter()
        .find(|b| b.buffer == "t7")
        .expect("the input binding");
    let out_off = op
        .arg_bindings
        .iter()
        .find(|b| b.buffer == "t9")
        .expect("the output binding");
    assert_eq!(
        in_off.offset_elems, 4096,
        "the input's corner is `rc_of(256, 128, 64, 0)` = row 64 of a stick-on-last nest = 64·64"
    );
    assert_eq!(
        out_off.offset_elems, 8192,
        "the output's corner is `rc_of(128, 256, 0, 64)` = stick group 1 of a stick-on-last nest \
         = 1·(128·64)"
    );

    // The `N_` rewrite: `y_` carries the VIEW's 256, not the tile's 64 — the card's stick-group
    // stride is reconstructed from `N_`, and the tile's would invent 64-row groups in a 256-row
    // allocation.
    let dsc = op
        .op
        .as_ref()
        .expect("a relayout carries a SuperDSC descriptor");
    let body = dsc
        .dscs_
        .first()
        .expect("one dsc")
        .values()
        .next()
        .expect("one op");
    assert_eq!(
        body.N_.y_, 256,
        "N_.y_ carries the input view's row count — the residency the card reconstructs from"
    );
    assert_eq!(
        body.N_.out_, 128,
        "N_.out_ is the hd axis, full-width in this form, so the plan's own extent is already the \
         view's"
    );

    // And the descriptor is still a TRANSPOSITION: the two roles' dim orders are swapped, the
    // output's stick is the single `y` axis (the property `SNComputeLowering.cpp:74` is a
    // function of — see `zz_the_transpose_door_carries_a_two_dim_output_stick`).
    let (inp, outp) = (
        &body.primaryDsInfo_["INPUT"],
        &body.primaryDsInfo_["OUTPUT"],
    );
    assert_eq!(
        (
            inp.layoutDimOrder_.as_slice(),
            outp.layoutDimOrder_.as_slice()
        ),
        (["y", "out"].as_slice(), ["out", "y"].as_slice()),
        "the two roles' dim orders must be SWAPPED — that swap IS the transposition"
    );
}

/// (f) THE BYTE-IDENTITY ANCHOR — whole views keep the folded form: zero corners, `N_` from the
/// plan (the views ARE the tiles), one descriptor.
#[test]
fn a_whole_view_transpose_keeps_the_folded_form() {
    let r = [whole(7, 256, 128, false), whole(9, 128, 256, true)];
    let mut sid = -1i64;
    let ops = transpose("tr", &r, &mut sid, None)
        .unwrap_or_else(|e| panic!("the whole-view form always lowered: {}", e.message));
    assert_eq!(ops.len(), 1, "one relayout descriptor");
    let op = &ops[0];
    for b in &op.arg_bindings {
        assert_eq!(
            b.offset_elems, 0,
            "the folded form is base-addressed — {} carries a corner it cannot place",
            b.buffer
        );
    }
    let dsc = op
        .op
        .as_ref()
        .expect("a relayout carries a SuperDSC descriptor");
    let body = dsc
        .dscs_
        .first()
        .expect("one dsc")
        .values()
        .next()
        .expect("one op");
    assert_eq!(
        (body.N_.y_, body.N_.out_),
        (256, 128),
        "at whole views the plan's extents ARE the views' — no rewrite, no declaration"
    );
}

/// (b) THE CORRESPONDENCE LAW — a store over any rectangle but the transposed window is refused.
/// One variable: the output's column window moves from [64, 128) to [128, 192).
#[test]
fn a_store_over_any_rectangle_but_the_transposed_window_refuses() {
    let a = windowed(
        7,
        256,
        128,
        Win {
            rows: (64, 64),
            cols: (0, 128),
        },
        false,
    );
    let out = windowed(
        9,
        128,
        256,
        Win {
            rows: (0, 128),
            cols: (128, 64),
        },
        true,
    );
    let e = lowered(&[a, out]).expect_err("a store over a different rectangle must refuse");
    assert!(
        e.contains("TRANSPOSED"),
        "the refusal must name the correspondence law — the output window is the input's, \
         transposed; got: {e}"
    );
}

/// (c) THE WINDOW STICK LAW — the historically fatal shape, per window. One variable: the window's
/// row extent drops from 64 to 32 (with the output's column extent matching, so it is the extents
/// that moved and not the correspondence).
#[test]
fn a_sub_stick_window_extent_refuses_by_name() {
    let a = windowed(
        7,
        256,
        128,
        Win {
            rows: (64, 32),
            cols: (0, 128),
        },
        false,
    );
    let out = windowed(
        9,
        128,
        256,
        Win {
            rows: (0, 128),
            cols: (64, 32),
        },
        true,
    );
    let e = lowered(&[a, out]).expect_err("a sub-stick window extent must refuse");
    assert!(
        e.contains("window extents") && e.contains("SEN169_FP16"),
        "the refusal must name the window stick law and the 64 rule; got: {e}"
    );
}

/// (d) THE CORNER ALIGNMENT — each side's stick-axis corner must be a whole stick. One variable:
/// both corners move from 64 to 32 (extents and correspondence held).
#[test]
fn a_sub_stick_corner_refuses_by_name() {
    let a = windowed(
        7,
        256,
        128,
        Win {
            rows: (32, 64),
            cols: (0, 128),
        },
        false,
    );
    let out = windowed(
        9,
        128,
        256,
        Win {
            rows: (0, 128),
            cols: (32, 64),
        },
        true,
    );
    let e = lowered(&[a, out]).expect_err("a sub-stick corner must refuse");
    assert!(
        e.contains("stick-aligned"),
        "the refusal must name the corner alignment law; got: {e}"
    );
}

/// (e) ONE RECTANGLE PER SIDE — a cover that is neither the whole view nor the single stated
/// window is several disjoint rectangles, and is refused rather than folded. One variable: the
/// input's row cover widens from the window (64, 128) to (0, 128) — a second tile at rows [0, 64).
#[test]
fn disjoint_windows_refuse_rather_than_fold() {
    let mut a = windowed(
        7,
        256,
        128,
        Win {
            rows: (64, 64),
            cols: (0, 128),
        },
        false,
    );
    a.r_cover = (0, 128);
    let out = windowed(
        9,
        128,
        256,
        Win {
            rows: (0, 128),
            cols: (64, 64),
        },
        true,
    );
    let e = lowered(&[a, out]).expect_err("a disjoint cover must refuse");
    assert!(
        e.contains("neither"),
        "the refusal must name the two bases and say this program states several rectangles; \
         got: {e}"
    );
}
