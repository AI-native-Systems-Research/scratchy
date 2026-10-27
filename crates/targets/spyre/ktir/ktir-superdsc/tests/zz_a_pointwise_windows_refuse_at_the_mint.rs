// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ A POINTWISE WINDOW PAST ITS TENSOR REFUSES AT THE MINT — the family door's
//! fail-closed half, the #717 class (`add_o1137` reading `t1131` past its footprint at the
//! width-512 prefix-capable continuation).
//!
//! [`zz_a_window_past_its_tensor_refuses_at_the_mint`] pins the SAME law for the `Win` type's
//! own doors; THIS file pins it for the pointwise family's operand and output mints
//! (`mint_ew_operands` / `mint_ew_output`), which every elementwise/scalarmul/gather/convert
//! builder passes through. A test set that only accepts is satisfied by a predicate that
//! accepts everything, so every refusal below has a control one variable away from it — the
//! handle's dims, the placement's size, the broadcast's offset — and each control BUILDS.
//!
//! The mint's two doors:
//! 1. THE DECLARED SHAPE — the swept window against the handle's own `rows × cols`;
//! 2. THE PLACEMENT FOOTPRINT — resolved by name against the same `align128` granule the
//!    bake-time guard reads, so a mint-pass and a bake-refusal cannot disagree.
//!
//! And the RedStick broadcast law's two bounds: the out-broadcast (col/col_at) forms' DEVICE
//! access is the stick RUN containing the offset (the L3 EBR rounds the reader down to the
//! stick base — rmsnorm's `rinv` half-stick lane select), while the VALUE each row reads sits
//! at the offset itself, stated as its own bound. A sub-stick offset can neither overstate
//! the access (refusing a legal lane select) nor hide a scalar past the tensor's end.

use ktir_superdsc::addr::DevOff;
use ktir_superdsc::emit::{EmittedOp, In, PointwiseFacts, assemble_pointwise_broadcast_off};
use ktir_superdsc::place::PlaceId;
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
use ktir_superdsc::sdsc_abstract::{BlockCols, Lanes, RowBlockedTag, RowCount, StickLayout, Stk};

fn rb(name: &str, rows: RowCount, cols: BlockCols) -> Stk<RowBlockedTag> {
    Stk::<RowBlockedTag>::new(
        name,
        StickLayout::row_blocked(rows.get() as usize, cols.get() as usize),
    )
    .unwrap()
}

/// `[rows, cols]` add over the two handles and the output — the family's own door, at the
/// residual-stream shape the fp8 quant chain really emits.
fn add(
    rows: u32,
    cols: u32,
    a: &Stk<RowBlockedTag>,
    b: &Stk<RowBlockedTag>,
    o: &Stk<RowBlockedTag>,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let mut sym = 0i64;
    assemble_pointwise_broadcast_off(
        PointwiseFacts {
            op_name: "add_test",
            op_func: "add",
            rows: RowCount::of_token_rows(rows),
            cols: BlockCols::of_feature_cols(cols),
            inputs: &[In::full(a).ew(), In::full(b).ew()],
            o,
            out_offset: DevOff::ZERO.into(),
        },
        &mut sym,
        layout,
    )
}

/// A layout placing `t{tid}` as an activation of `size` bytes — the same rig
/// [`zz_a_window_past_its_tensor_refuses_at_the_mint`] drives the `Win` doors with.
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

/// ⭐ THE #717 CLASS, OPERAND SIDE — the `1×64` annotation lie under a `[31, 2048]` sweep.
/// The control is the same op over the TRUE `[31, 2048]` handle: it builds.
#[test]
fn an_operand_window_past_its_handle_refuses() {
    let true_h = rb(
        "t_true",
        RowCount::of_token_rows(31),
        BlockCols::of_feature_cols(2048),
    );
    let o = rb(
        "t_o",
        RowCount::of_token_rows(31),
        BlockCols::of_feature_cols(2048),
    );
    let built = add(
        31,
        2048,
        &true_h,
        &rb(
            "t_true2",
            RowCount::of_token_rows(31),
            BlockCols::of_feature_cols(2048),
        ),
        &o,
        None,
    );
    assert!(built.op.is_some(), "the true handle builds — the control");

    let lie = rb(
        "t_lie",
        RowCount::of_broadcast_row(),
        BlockCols::of_one_stick(Lanes::FP16),
    );
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        add(31, 2048, &lie, &true_h, &o, None)
    }));
    let msg = refused(r);
    assert!(
        msg.contains("operand 't_lie'")
            && msg.contains("[0, 63488)")
            && msg.contains("runs past the end its handle declares"),
        "the refusal names the tensor, its window and the bound: {msg}"
    );
}

/// ⭐ THE #717 CLASS, OUTPUT SIDE — the write window is checked by the same law (the output
/// mint runs FIRST in the assembler, so a lying output refuses even with true inputs).
#[test]
fn an_output_window_past_its_handle_refuses() {
    let a = rb(
        "t_a",
        RowCount::of_token_rows(31),
        BlockCols::of_feature_cols(2048),
    );
    let b = rb(
        "t_b",
        RowCount::of_token_rows(31),
        BlockCols::of_feature_cols(2048),
    );
    let small = rb(
        "t_small",
        RowCount::of_broadcast_row(),
        BlockCols::of_one_stick(Lanes::FP16),
    );
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        add(31, 2048, &a, &b, &small, None)
    }));
    let msg = refused(r);
    assert!(
        msg.contains("output 't_small'") && msg.contains("runs past the end its handle declares"),
        "the refusal names the output and the bound: {msg}"
    );
}

/// ⭐ #717'S OWN CARD NUMBERS — the placement door. The handle declares the TRUE
/// `[512, 768]` (393216 elements, 786432 B at Fp16) and the window fits it exactly; the
/// PLACEMENT the layout holds is the pod's 279552 B validity shape. The mint refuses at
/// build naming the footprint — the refusal the card answered with a DMA past the segment.
/// The control is the same op at the pod's demanded 786432 B: it builds.
#[test]
fn an_operand_window_past_its_placement_refuses() {
    let (rows, cols) = (512u32, 768u32);
    let mask = rb(
        "t1131",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(cols),
    );
    let a = rb(
        "t8",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(cols),
    );
    let o = rb(
        "t9",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(cols),
    );
    let small = layout_placing(&[(1131, 279552), (8, 786432), (9, 786432)]);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        add(rows, cols, &mask, &a, &o, Some(&small))
    }));
    let msg = refused(r);
    assert!(
        msg.contains("operand 't1131'")
            && msg.contains("786432B")
            && msg.contains("exceeds its placement footprint 279552B"),
        "the refusal names the window's bytes and the footprint it exceeds: {msg}"
    );
    let covering = layout_placing(&[(1131, 786432), (8, 786432), (9, 786432)]);
    let built = add(rows, cols, &mask, &a, &o, Some(&covering));
    assert!(
        built.op.is_some(),
        "a covering placement builds — the control"
    );
}

/// ⭐ THE REDSTICK VALUE BOUND — a per-row scalar broadcast whose LAST row's value runs past
/// the handle's end. The stick-run window fits (the doors see `[0, 1984)` of a `[31, 64]`
/// column), so only the value bound catches it — the scalar past the tensor's end that a
/// stick-granular door alone would hide.
#[test]
fn a_broadcast_last_row_scalar_past_the_end_refuses() {
    let rows = 31u32;
    let col = rb(
        "t_col",
        RowCount::of_token_rows(rows),
        BlockCols::of_one_stick(Lanes::FP16),
    );
    let a = rb(
        "t_a",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(2048),
    );
    let o = rb(
        "t_o",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(2048),
    );
    let build = |b: In<'_, RowBlockedTag>| {
        let mut sym = 0i64;
        assemble_pointwise_broadcast_off(
            PointwiseFacts {
                op_name: "mul_test",
                op_func: "multiply",
                rows: RowCount::of_token_rows(rows),
                cols: BlockCols::of_feature_cols(2048),
                inputs: &[In::full(&a).ew(), b.ew()],
                o: &o,
                out_offset: DevOff::ZERO.into(),
            },
            &mut sym,
            None,
        )
    };
    // The column at its base: last row's scalar at (rows-1)·64 = 1920 < 1984 — builds.
    let built = build(In::col(&col));
    assert!(
        built.op.is_some(),
        "the column at its base builds — the control"
    );
    // The same column shifted a whole row past its end: the last row's scalar sits at
    // 1984 + 1920 = 3904, past the 1984 elements the handle declares.
    let past = In::col_at(&col, DevOff::from_view_step(rows * 64));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| build(past)));
    let msg = refused(r);
    assert!(
        msg.contains("the broadcast read's last row scalar (element 3904) runs past"),
        "the refusal names the last row's scalar position: {msg}"
    );
}

/// ⭐ THE HALF-STICK LANE SELECT STAYS LEGAL — rmsnorm's `rinv` law, the ACCEPTING control
/// for the sub-stick offset: the value bound holds (8 + 30·64 = 1928 < 1984) and the doors
/// see the stick RUN from the base, so the EBR's stick-granular read neither overstates the
/// access nor hides the scalar. A mint that refused this would be refusing the vendor's own
/// lane select.
#[test]
fn a_sub_stick_lane_select_stays_legal() {
    let rows = 31u32;
    let col = rb(
        "t_rinv",
        RowCount::of_token_rows(rows),
        BlockCols::of_one_stick(Lanes::FP16),
    );
    let a = rb(
        "t_x",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(2048),
    );
    let o = rb(
        "t_o",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(2048),
    );
    let mut sym = 0i64;
    let built = assemble_pointwise_broadcast_off(
        PointwiseFacts {
            op_name: "mul_test",
            op_func: "multiply",
            rows: RowCount::of_token_rows(rows),
            cols: BlockCols::of_feature_cols(2048),
            inputs: &[
                In::full(&a).ew(),
                In::col_at(&col, DevOff::from_view_step(8)).ew(),
            ],
            o: &o,
            out_offset: DevOff::ZERO.into(),
        },
        &mut sym,
        None,
    );
    assert!(
        built.op.is_some(),
        "the +8 lane select builds — the rinv law"
    );
}

/// ⭐ THE PITCH LAW — a per-row scalar broadcast reads ONE STICK PER ROW; a column wider
/// than the stick has no stated bound and is refused rather than guessed.
#[test]
fn a_wider_per_row_broadcast_refuses() {
    let rows = 31u32;
    let wide_col = rb(
        "t_widecol",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(128),
    );
    let a = rb(
        "t_a",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(2048),
    );
    let o = rb(
        "t_o",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(2048),
    );
    let mut sym = 0i64;
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assemble_pointwise_broadcast_off(
            PointwiseFacts {
                op_name: "mul_test",
                op_func: "multiply",
                rows: RowCount::of_token_rows(rows),
                cols: BlockCols::of_feature_cols(2048),
                inputs: &[In::full(&a).ew(), In::col(&wide_col).ew()],
                o: &o,
                out_offset: DevOff::ZERO.into(),
            },
            &mut sym,
            None,
        )
    }));
    let msg = refused(r);
    assert!(
        msg.contains("ONE STICK PER ROW"),
        "the refusal states the one-stick-per-row law: {msg}"
    );
}

/// ⭐ THE MINT NEVER ALLOCATES — a name no authority declared (not an Act placement, not a
/// synth) is refused, not silently bumped: the bump-allocator defect `synth` exists to kill.
/// Every OTHER name in the op is declared, so the refusal isolates the ghost.
#[test]
fn an_undeclared_name_refuses_rather_than_bumps() {
    let (rows, cols) = (31u32, 2048u32);
    let ghost = rb(
        "t4104",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(cols),
    );
    let a = rb(
        "t8",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(cols),
    );
    let o = rb(
        "t9",
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(cols),
    );
    let layout = layout_placing(&[(8, 126976), (9, 126976)]);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        add(rows, cols, &ghost, &a, &o, Some(&layout))
    }));
    let msg = refused(r);
    assert!(
        msg.contains("'t4104'") && msg.contains("was never declared to the layout"),
        "the refusal demands the declaration: {msg}"
    );
}
