// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ A RESTICKIFY OR CONVERT WINDOW PAST ITS TENSOR REFUSES AT THE MINT — the fail-closed
//! half of the Win lockdown's third family door, after the pointwise (#734) and reduce (#736)
//! mints.
//!
//! [`assemble_restickify_kt_2d`] re-lays one `[slots, feats]` tile — ONE CONTIGUOUS run of
//! `slots·feats` elements at each corner — and [`assemble_convert`]/[`assemble_convert_col`]
//! sweep a `[rows, cols]` window at a nest-derived corner. Both families' mint
//! (`mint_full_window`) bounds that window against BOTH doors:
//! 1. THE DECLARED SHAPE — the handle's own `rows × cols`;
//! 2. THE PLACEMENT FOOTPRINT — resolved by name against the same `align128` granule the
//!    bake-time guard reads.
//!
//! The class is #717's (`add_o1137` reading `t1131` past its footprint): a restickify whose
//! Kᵀ tile runs past the `[nkvh, hd, mq_pad]` scratch, or a convert whose slot window runs
//! past `cl_reach`/`k_pad`, bakes clean and reads or writes the next intermediate's bytes —
//! fluent, wrong, no fault. Every refusal below has a control one variable away, and each
//! control BUILDS.
//!
//! ⭐ AND THE fp8 SIDE'S OWN LAW RIDES HERE TOO: the convert's output handle carries the
//! OUTPUT's df (`rb_df`), because the placement door prices bytes at the handle's word length —
//! an fp16-annotated fp8 buffer overstates every window 2× and refuses the last slot, whose
//! window ends exactly at the buffer's own end by the `cl_reach = (pairs−1)·inn + inn_pad` law.

use ktir_superdsc::addr::DevOff;
use ktir_superdsc::emit::{
    EmittedOp, assemble_convert, assemble_convert_col, assemble_restickify_kt_2d, rb_df,
    try_assemble_restickify_transpose_2d,
};
use ktir_superdsc::place::PlaceId;
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
use ktir_superdsc::sdsc_abstract::{
    BlockCols, KtTileFeats, KtTileSlots, Lanes, RowBlockedTag, RowCount, RowWindow, StickLayout,
    Stk,
};
use ktir_superdsc::superdsc_opspec::Df;

fn rbl(name: &str, rows: RowCount, cols: BlockCols) -> Stk<RowBlockedTag> {
    Stk::<RowBlockedTag>::new(
        name,
        StickLayout::row_blocked(rows.get() as usize, cols.get() as usize),
    )
    .unwrap()
}

/// A layout placing `t{tid}` as an activation of `size` bytes — the same rig
/// [`zz_a_pointwise_windows_refuse_at_the_mint`] drives the family mints with.
fn layout_placing(entries: &[(u32, u64)]) -> BundleLayout {
    let mut layout = BundleLayout::default();
    for &(tid, size) in entries {
        layout
            .ids
            .borrow_mut()
            .insert(PlaceId::Act(tid).to_string(), PlaceId::Act(tid));
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
    }
    layout
}

/// The panic text out of a refused build — the assemblers panic with `format!`ed refusal
/// messages, so the refusal is observable as the payload.
fn refused(r: Result<EmittedOp, Box<dyn std::any::Any + Send>>) -> String {
    let payload = match r {
        Ok(_) => panic!("the build was accepted — the refusal under test did not fire"),
        Err(p) => p,
    };
    if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else {
        panic!("a panic payload that is not a string")
    }
}

/// The decoder's own `64 × 64` Kᵀ tile, at the corners given — the restickify door under test.
fn kt_2d(
    in_: &Stk<RowBlockedTag>,
    in_off: DevOff,
    o: &Stk<RowBlockedTag>,
    out_off: DevOff,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let mut sym = 0i64;
    assemble_restickify_kt_2d(
        "rk_test",
        KtTileSlots::of_row_window(RowWindow::ROWS),
        KtTileFeats::of_head_dim(64),
        in_,
        in_off,
        o,
        out_off,
        &mut sym,
        layout,
    )
}

/// ⭐ THE Kᵀ TILE'S FLAT RUN PAST ITS INPUT — the tile is one contiguous `4096`-element run at
/// the corner, and a corner that walks past the roped new-K's end reads the next tensor's bytes
/// as keys. The control is the same op at corner 0 over the TRUE `[64, 64]` buffers: it builds.
#[test]
fn a_kt_tile_past_its_input_handle_refuses() {
    let (k_nat, k_t) = (
        rbl(
            "k_nat",
            RowCount::of_token_rows(64),
            BlockCols::of_feature_cols(64),
        ),
        rbl(
            "k_t",
            RowCount::of_token_rows(64),
            BlockCols::of_feature_cols(64),
        ),
    );
    let built = kt_2d(&k_nat, DevOff::ZERO, &k_t, DevOff::ZERO, None);
    assert!(built.op.is_some(), "the true handles build — the control");

    let past = ktir_superdsc::addr::col_of(1, 8192, 4096, Df::Fp16);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        kt_2d(&k_nat, past, &k_t, DevOff::ZERO, None)
    }));
    let msg = refused(r);
    assert!(
        msg.contains("operand 'k_nat'")
            && msg.contains("[4096, 8192)")
            && msg.contains("runs past the end its handle declares"),
        "the refusal names the tensor, its flat run and the bound: {msg}"
    );
}

/// ⭐ THE SAME TILE PAST ITS OUTPUT — the Kᵀ scratch side. A tile written past the
/// `[nkvh, hd, mq_pad]` scratch lands in the next intermediate's bytes; the score matmul then
/// reads keys nobody restickified. The control is the same op at corner 0: it builds.
#[test]
fn a_kt_tile_past_its_output_handle_refuses() {
    let (k_nat, k_t) = (
        rbl(
            "k_nat",
            RowCount::of_token_rows(64),
            BlockCols::of_feature_cols(64),
        ),
        rbl(
            "k_t",
            RowCount::of_token_rows(64),
            BlockCols::of_feature_cols(64),
        ),
    );
    let past = ktir_superdsc::addr::col_of(1, 8192, 4096, Df::Fp16);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        kt_2d(&k_nat, DevOff::ZERO, &k_t, past, None)
    }));
    let msg = refused(r);
    assert!(
        msg.contains("output 'k_t'") && msg.contains("[4096, 8192)"),
        "the refusal names the output and its flat run: {msg}"
    );
}

/// ⭐ THE PLACEMENT DOOR — the handle states the truth and the placement does not. The tile's
/// run fits the declared `[64, 64]` exactly; the LAYOUT holds a placement half that size, which
/// is the "512 reserved against 2048 declared" class: a clean bake whose overrun is another
/// tensor's bytes. The control is the same op against the placement the handle states: it builds.
#[test]
fn a_kt_tile_past_its_placement_refuses() {
    let (k_nat, k_t) = (
        rbl(
            "t7",
            RowCount::of_token_rows(64),
            BlockCols::of_feature_cols(64),
        ),
        rbl(
            "t8",
            RowCount::of_token_rows(64),
            BlockCols::of_feature_cols(64),
        ),
    );
    let small = layout_placing(&[(7, 4096), (8, 8192)]);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        kt_2d(&k_nat, DevOff::ZERO, &k_t, DevOff::ZERO, Some(&small))
    }));
    let msg = refused(r);
    assert!(
        msg.contains("operand 't7'") && msg.contains("exceeds its placement footprint"),
        "the refusal names the placement it would overrun: {msg}"
    );

    let true_layout = layout_placing(&[(7, 8192), (8, 8192)]);
    let built = kt_2d(&k_nat, DevOff::ZERO, &k_t, DevOff::ZERO, Some(&true_layout));
    assert!(
        built.op.is_some(),
        "the true placement builds — the control"
    );
}

/// ⭐ THE DECODE ARM'S OWN ARITHMETIC, AS THE CONTROL — the per-slot convert at `m == 1`, routed
/// through the column door with `col_of` corners exactly as `ktir_matmul_fp8` emits it. At
/// `pairs = 2, inn = 64, inn_pad = 128`: `cl_reach = 192`, `k_pad = 256`, and slot 1's windows
/// end EXACTLY at both buffers' ends — the last-slot fit the zero-fill law guarantees. It builds.
/// This is the emission every decode bundle now carries, pinned green.
#[test]
fn the_last_slots_convert_fits_exactly_through_the_column_door() {
    let (m, inn, inn_pad, pairs) = (1u32, 64u32, 128u32, 2u32);
    let (cl_reach, k_pad) = ((pairs - 1) * inn + inn_pad, pairs * inn_pad);
    let (cl, afp8) = (
        rbl(
            "cl",
            RowCount::of_token_rows(m),
            BlockCols::of_feature_cols(cl_reach),
        ),
        rb_df(
            "afp8",
            RowCount::of_token_rows(m),
            BlockCols::of_feature_cols(k_pad),
            Df::Fp8,
        ),
    );
    let mut sym = 0i64;
    for s in 0..pairs {
        let op = assemble_convert_col(
            &format!("fq_p{s}_op"),
            "qfp8ch",
            m,
            inn_pad,
            &cl,
            ktir_superdsc::addr::col_of(m, cl_reach, s * inn, Df::Fp16),
            &afp8,
            ktir_superdsc::addr::col_of(m, k_pad, s * inn_pad, Df::Fp8),
            &mut sym,
            None,
        );
        assert!(
            op.op.is_some(),
            "slot {s}'s convert builds — the exact-fit control"
        );
    }
}

/// ⭐ THE PIECE HANDLE — the lie the m=1 arm used to carry. Slot 1's input window is
/// `[64, 192)` against `cl_reach = 192`; a handle declaring only the PIECE `[1, inn_pad]` = 128
/// elements refuses, which is what routing the decode arm through piece handles at raw offsets
/// could never state. The control is the same slot over the full-buffer handle: it builds
/// (pinned above).
#[test]
fn a_piece_convert_handle_refuses_the_last_slot() {
    let (m, inn, inn_pad, pairs) = (1u32, 64u32, 128u32, 2u32);
    let cl_reach = (pairs - 1) * inn + inn_pad;
    let piece = rbl(
        "cl",
        RowCount::of_token_rows(m),
        BlockCols::of_feature_cols(inn_pad),
    );
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut sym = 0i64;
        assemble_convert_col(
            "fq_p1",
            "qfp8ch",
            m,
            inn_pad,
            &piece,
            ktir_superdsc::addr::col_of(m, cl_reach, inn, Df::Fp16),
            &rb_df(
                "afp8",
                RowCount::of_token_rows(m),
                BlockCols::of_feature_cols(pairs * inn_pad),
                Df::Fp8,
            ),
            ktir_superdsc::addr::col_of(m, pairs * inn_pad, inn_pad, Df::Fp8),
            &mut sym,
            None,
        )
    }));
    let msg = refused(r);
    assert!(
        msg.contains("operand 'cl'") && msg.contains("[64, 192)"),
        "the refusal names the piece handle and the window that crosses it: {msg}"
    );
}

/// ⭐ THE fp8 SIDE'S df LAW — the output handle carries the OUTPUT's own df, because the
/// placement door prices bytes at the handle's word length. Slot 1's output window is
/// `[128, 256)` of a 256-element fp8 buffer placed at its fp8 truth (256 B): the df-TRUE handle
/// builds (256 B), and the fp16-ANNOTATED one — same elements, twice the bytes (512 B) —
/// refuses, which is the overstatement `rb_df` exists to prevent. The input's own window
/// (`[64, 192)` of 192 fp16 elements = 384 B) fits its placement exactly, so the output's
/// annotation is the only variable.
#[test]
fn an_fp16_annotated_fp8_output_overstates_its_window_and_refuses() {
    let (m, inn, inn_pad, pairs) = (1u32, 64u32, 128u32, 2u32);
    let (cl_reach, k_pad) = ((pairs - 1) * inn + inn_pad, pairs * inn_pad);
    let layout = layout_placing(&[(8, cl_reach as u64 * 2), (9, k_pad as u64)]);
    let (cl, afp8_true, afp8_annotated) = (
        rbl(
            "t8",
            RowCount::of_token_rows(m),
            BlockCols::of_feature_cols(cl_reach),
        ),
        rb_df(
            "t9",
            RowCount::of_token_rows(m),
            BlockCols::of_feature_cols(k_pad),
            Df::Fp8,
        ),
        rbl(
            "t9",
            RowCount::of_token_rows(m),
            BlockCols::of_feature_cols(k_pad),
        ),
    );
    let mut sym = 0i64;
    let built = assemble_convert_col(
        "fq_df",
        "qfp8ch",
        m,
        inn_pad,
        &cl,
        ktir_superdsc::addr::col_of(m, cl_reach, inn, Df::Fp16),
        &afp8_true,
        ktir_superdsc::addr::col_of(m, k_pad, inn_pad, Df::Fp8),
        &mut sym,
        Some(&layout),
    );
    assert!(
        built.op.is_some(),
        "the df-true handle builds — the control"
    );

    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut sym = 0i64;
        assemble_convert_col(
            "fq_df",
            "qfp8ch",
            m,
            inn_pad,
            &cl,
            ktir_superdsc::addr::col_of(m, cl_reach, inn, Df::Fp16),
            &afp8_annotated,
            ktir_superdsc::addr::col_of(m, k_pad, inn_pad, Df::Fp8),
            &mut sym,
            Some(&layout),
        )
    }));
    let msg = refused(r);
    assert!(
        msg.contains("output 't9'") && msg.contains("exceeds its placement footprint"),
        "the refusal names the fp8 output and the footprint the fp16 annotation overstates: {msg}"
    );
}

/// ⭐ THE WHOLE-TENSOR CONVERT DOOR — the same law at corner 0: a `[1, 128]` convert over a
/// 64-element input refuses on the handle door. The control is the true `[1, 128]` pair: it
/// builds.
#[test]
fn a_whole_tensor_convert_past_its_handle_refuses() {
    let (act, afp8) = (
        rbl(
            "act",
            RowCount::of_broadcast_row(),
            BlockCols::of_feature_cols(128),
        ),
        rb_df(
            "afp8",
            RowCount::of_broadcast_row(),
            BlockCols::of_feature_cols(128),
            Df::Fp8,
        ),
    );
    let mut sym = 0i64;
    let built = assemble_convert("fq", "qfp8ch", 1, 128, &act, &afp8, &mut sym, None);
    assert!(built.op.is_some(), "the true handles build — the control");

    let small = rbl(
        "act",
        RowCount::of_broadcast_row(),
        BlockCols::of_one_stick(Lanes::FP16),
    );
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut sym = 0i64;
        assemble_convert("fq", "qfp8ch", 1, 128, &small, &afp8, &mut sym, None)
    }));
    let msg = refused(r);
    assert!(
        msg.contains("operand 'act'") && msg.contains("runs past the end its handle declares"),
        "the refusal names the input and the bound: {msg}"
    );
}

/// ⭐ THE TRANSPOSE DOOR — whole-buffer windows both sides; a handle smaller than the stated
/// `[rows, cols]` is a reservation that cannot hold the relayout ("512 reserved against 2048
/// declared"). This door is fallible, so the refusal is the `Err` itself. The control is the
/// true pair: it emits.
#[test]
fn a_transpose_over_a_handle_smaller_than_its_view_refuses() {
    let mut sym = 0i64;
    let built = try_assemble_restickify_transpose_2d(
        "tr",
        128,
        128,
        &rbl(
            "t_in",
            RowCount::of_token_rows(128),
            BlockCols::of_feature_cols(128),
        ),
        &rbl(
            "t_out",
            RowCount::of_token_rows(128),
            BlockCols::of_feature_cols(128),
        ),
        &mut sym,
        None,
    );
    assert!(built.is_ok(), "the true handles emit — the control");

    let mut sym = 0i64;
    let err = match try_assemble_restickify_transpose_2d(
        "tr",
        128,
        128,
        &rbl(
            "t_in",
            RowCount::of_token_rows(64),
            BlockCols::of_feature_cols(64),
        ),
        &rbl(
            "t_out",
            RowCount::of_token_rows(128),
            BlockCols::of_feature_cols(128),
        ),
        &mut sym,
        None,
    ) {
        Ok(_) => panic!("a [128, 128] transpose over a [64, 64] handle must refuse at the mint"),
        Err(e) => e,
    };
    assert!(
        err.contains("operand 't_in'") && err.contains("runs past the end its handle declares"),
        "the refusal names the input and the bound: {err}"
    );
}
