// SPDX-License-Identifier: Apache-2.0

//! ⭐⭐⭐⭐⭐ THE REDUCE FAMILY's FRAME — ITS RESPelled PRESENTATION, AS TESTS.
//!
//! A row reduce writes ONE VALUE PER DATA ROW: the accum is `Scale::RedStick` (one whole stick
//! per row, value k at stick k), so the reduce's result — however the program SPELLS it — is
//! the data's row-sweep vector in the accumulator's frame. The online softmax states that
//! vector as a rank-2 `[256, 4]` tile (the torch spelling of 1024 per-row values over
//! `[256, 4, 64]` scores), and the walk's intermediate mint respells it to `[1024, 1]`
//! ([`respelled_frame_rows`]) — the frame the accum writes, the family's pointwise ops read,
//! and the placement reserves. MEASURED (wall 11): the verbatim mint placed the reduce's
//! output at `[256, 4]` — a 256-stick footprint under a 1024-stick accum — and the door
//! refused correctly.
//!
//! What these tests pin:
//!
//! * THE POSITIVE: a `[256, 4]` reduce over `[256, 4, 64]` data (loaded flat `[1024, 64]`,
//!   reshaped — the real program's own spelling) lowers with EVERY op in the flat frame —
//!   the reduce's iteration space at 1024 rows, the family member (`exp`) at 1024 rows, the
//!   two-tensor family add (`%113`'s own shape) at 1024 rows.
//! * ⛔ CONTROL (the door's law stands): a reduce whose result numel is NOT the data's row
//!   sweep (an axis-0 reduce's `[4, 64]`) keeps its stated shape and the door refuses it by
//!   name — the respelling is not a license for non-row-reduce forms.
//! * ⛔ CONTROL (no family input, no respell): an op whose result matches NO family input
//!   keeps its stated shape and its OWN door refuses the extents — never silently re-framed.
//! * ⛔ CONTROL (a WIDE input's rows are not a frame fact): a `[32, 64]` result over a
//!   `[2048, 64]` table has numel == the table's rows — the gather-copy trap — and must keep
//!   its STATED shape (the refusal names `[32, 64]` as the output), because a wide input's
//!   row count frames nothing but a reduce.
//!
//! Every program is built from the op kinds the walk really reads, so the door sees a real
//! program and not a stub.

use ktir_core::arena::Arena;
use ktir_core::attrkey::AttrKey;
use ktir_core::dtypes::DType;
use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
use ktir_core::irtype::IrType;
use ktir_core::opkind::OpKind;
use ktir_superdsc::emit::whole_function::lower_function;
use ktir_superdsc::ktir_node::{BufferId, Elementwise, KtirNode, Program};
use ktir_superdsc::place::act_name;
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};

/// The flat row sweep of the scores: `[256, 4, 64]` data, last axis one stick, read flat.
const FLAT_ROWS: i64 = 1024;
/// The scores' own width — one stick, so the flat frame is byte-true.
const COLS: i64 = 64;
/// The family's rank-2 spelling of the 1024 per-row values.
const FAMILY: (i64, i64) = (256, 4);

/// The online-softmax slice the wall measured: scores loaded flat `[1024, 64]`, reshaped to
/// the program's `[256, 4, 64]`, reduced (sum, splat initial) to a `[256, 4]` intermediate,
/// `exp`'d (a family member reading the one-column frame), added to the reduce itself
/// (`%113`'s own two-tensor shape), the sum stored to a `[1024, 1]` output.
///
/// `result` overrides the intermediate results' stated type, so the controls can state the
/// wrong shape for the same program skeleton.
fn softmax_slice(result: (i64, i64)) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };
    let ty = |r: i64, c: i64| {
        Some(IrType::Tensor {
            dims: a.ints(vec![r, c]),
            elem: DType::F16,
        })
    };

    // Parameters: %0 = scores, %1 = out.
    let (p_q, p_out) = (Ssa(0), Ssa(1));
    let mut next = 2u32;
    let mut ssa = || {
        let s = Ssa(next);
        next += 1;
        s
    };

    let (v_q, v_o) = (ssa(), ssa());
    let mut ops = vec![
        shape(
            Operation::new(a, Some(v_q), OpKind::KtdpConstructMemoryView, &[p_q]),
            FLAT_ROWS,
            COLS,
        ),
        shape(
            Operation::new(a, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
            FLAT_ROWS,
            1,
        ),
    ];

    // The scores load: one whole-view tile, flat `[1024, 64]`.
    let (c_zero, c_zero2) = (ssa(), ssa());
    ops.push(
        Operation::new(a, Some(c_zero), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(0),
        ),
    );
    ops.push(
        Operation::new(a, Some(c_zero2), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(0),
        ),
    );
    let (t_q, l_q, rs) = (ssa(), ssa(), ssa());
    ops.push(shape(
        Operation::new(
            a,
            Some(t_q),
            OpKind::KtdpConstructAccessTile,
            &[v_q, c_zero, c_zero2],
        ),
        FLAT_ROWS,
        COLS,
    ));
    let mut load = shape(
        Operation::new(a, Some(l_q), OpKind::KtdpLoad, &[t_q]),
        FLAT_ROWS,
        COLS,
    );
    load.result_type = ty(FLAT_ROWS, COLS);
    ops.push(load);
    // The program's own spelling of the scores — numel-preserving, so the walk traverses it
    // and the reduce's data region stays the flat `[1024, 64]` frame. The reshape states its
    // own result type: the #741 re-row law reads both frames off it.
    let mut reshape = Operation::new(a, Some(rs), OpKind::TensorReshape, &[l_q])
        .with_attr(
            a,
            AttrKey::Shape,
            Attr::IntList(a.ints(vec![FAMILY.0, FAMILY.1, COLS])),
        )
        .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
        .with_attr(
            a,
            AttrKey::TargetShape,
            Attr::IntList(a.ints(vec![FAMILY.0, FAMILY.1, COLS])),
        );
    reshape.result_type = Some(IrType::Tensor {
        dims: a.ints(vec![FAMILY.0, FAMILY.1, COLS]),
        elem: DType::F16,
    });
    ops.push(reshape);

    // The reduce: sum over the stick axis, splat zero initial, `[256, 4]` result.
    let (c_zero_f, splat, red) = (ssa(), ssa(), ssa());
    ops.push(
        Operation::new(a, Some(c_zero_f), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Float(0.0),
        ),
    );
    ops.push(shape(
        Operation::new(a, Some(splat), OpKind::TensorSplat, &[c_zero_f]),
        FAMILY.0,
        FAMILY.1,
    ));
    let mut reduce = shape(
        Operation::new(a, Some(red), OpKind::LinalgReduce, &[rs, splat]),
        FAMILY.0,
        FAMILY.1,
    )
    .with_attr(a, AttrKey::ReduceFn, Attr::Op(OpKind::ArithAddf))
    .with_attr(a, AttrKey::Dimensions, Attr::IntList(a.ints(vec![2])));
    reduce.result_type = ty(result.0, result.1);
    ops.push(reduce);

    // A family member: `exp` of the reduce's result, the same rank-2 spelling.
    let exp_res = ssa();
    let mut exp = shape(
        Operation::new(a, Some(exp_res), OpKind::MathExp, &[red]),
        FAMILY.0,
        FAMILY.1,
    );
    exp.result_type = ty(result.0, result.1);
    ops.push(exp);

    // The two-tensor family add (`%113`'s own shape) and the store of its result.
    let add = ssa();
    let mut add_op = shape(
        Operation::new(a, Some(add), OpKind::ArithAddf, &[red, exp_res]),
        FAMILY.0,
        FAMILY.1,
    );
    add_op.result_type = ty(result.0, result.1);
    ops.push(add_op);
    let t_o = ssa();
    ops.push(shape(
        Operation::new(
            a,
            Some(t_o),
            OpKind::KtdpConstructAccessTile,
            &[v_o, c_zero, c_zero2],
        ),
        FLAT_ROWS,
        1,
    ));
    ops.push(Operation::new(a, None, OpKind::KtdpStore, &[add, t_o]));

    KtirNode {
        func: IRFunction {
            name: "softmax_slice",
            arguments: a.args(vec![(p_q, IrType::Index), (p_out, IrType::Index)]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        },
        program: Program::Elementwise(Elementwise::Add),
        bindings: vec![BufferId::new(401), BufferId::new(402)],
        mask: None,
        step_inputs: Vec::new(),
        node_out_tid: None,
        minted: Default::default(),
    }
}

/// A fully-placed layout for the two parameters — `resolve_seg_base` panics on a name that is
/// in neither `ids` nor the synth map, so every operand the emission names must be placed.
fn layout_for(k: &KtirNode) -> BundleLayout {
    let mut l = BundleLayout::default();
    for (i, b) in k.bindings.iter().enumerate() {
        let tid = b.get();
        l.ids
            .borrow_mut()
            .insert(act_name(tid), ktir_superdsc::place::PlaceId::Act(tid));
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: 3,
                bank: 0,
                offset: (i as u64) * 0x100000,
                size: (FLAT_ROWS * COLS * 2) as u64,
                cover: ktir_superdsc::placement::BindCover::Full,
            },
        );
    }
    l
}

fn lowered(k: &KtirNode) -> Result<Vec<ktir_superdsc::emit::EmittedOp>, String> {
    let layout = layout_for(k);
    let mut sid = 0i64;
    lower_function(k, Some(&layout), &mut sid, None).map_err(|e| e.message)
}

/// The `dscs_[0]` entry of an emitted op — the descriptor the assertions read.
fn dsc(op: &ktir_superdsc::emit::EmittedOp) -> &ktir_superdsc::wire::Dsc {
    op.dsc()
        .dscs_
        .first()
        .and_then(|m| m.values().next())
        .expect("the op has a descriptor")
}

/// ⭐⭐⭐ THE POSITIVE: the `[256, 4]` family lowers in the accumulator's `[1024, 1]` frame.
///
/// The reduce's iteration space is the FLAT rows (1024, not the spelling's 256), the family
/// `exp` and the two-tensor family add sit at the same 1024 rows, and every op emits — the
/// frame is one law the reduce, the family members, the placement, and the consumers all
/// read. Before the respelling the door refused this exact program ("ONE value per row —
/// `[1024, 1]`").
#[test]
fn the_rank_two_reduce_family_lowers_in_the_accumulator_frame() {
    let k = softmax_slice(FAMILY);
    let ops = lowered(&k).expect("the respelled family lowers through the door");
    let [sum, exp, add] = &ops[..] else {
        panic!(
            "the slice is three ops (reduce, exp, add), got {:?}",
            ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
        );
    };
    for (op, what) in [
        (sum, "the reduce"),
        (exp, "the family exp"),
        (add, "the family add"),
    ] {
        assert_eq!(
            dsc(op).N_.mb_,
            FLAT_ROWS,
            "{what} iterates the FLAT row sweep, not the spelling's rows"
        );
    }
    // The reduce's own width is the stick axis it reduces — one stick, unpadded.
    assert_eq!(
        dsc(sum).N_.out_,
        COLS,
        "the reduce sweeps the 64-wide stick axis"
    );
}

/// ⛔ THE DOOR's LAW STANDS: a reduce whose result numel is NOT the data's row sweep keeps
/// its stated shape and the door refuses it by name. An axis-0 reduce's `[4, 64]` result
/// (numel 256 ≠ 1024) is not the row-reduce form, and the respelling is not a license for it.
#[test]
fn a_reduce_whose_result_numel_is_not_the_row_sweep_refuses_at_the_door() {
    let k = softmax_slice((4, 64));
    let e = match lowered(&k) {
        Err(e) => e,
        Ok(ops) => panic!(
            "a non-row-sweep reduce result must refuse, not lower {} op(s)",
            ops.len()
        ),
    };
    assert!(
        e.contains("ONE value per row"),
        "the refusal names the one-value-per-row law, got: {e}"
    );
    assert!(
        e.contains("[1024, 1]"),
        "the refusal names the accumulator frame, got: {e}"
    );
    assert!(
        e.contains("[4, 64]"),
        "the refusal names the stated shape it refused, got: {e}"
    );
}

/// ⛔ THE NUMEL GUARD: a one-column input does not respell a result whose numel DIFFERS. An
/// `exp` over a `[32, 1]` load stating a `[32, 64]` result (a broadcast's shape — numel 2048,
/// not 32) keeps its stated shape and its OWN door refuses the extents. Drop the numel guard
/// and this exp would be minted `[32, 1]` and LOWER — the expected refusal is the pin.
#[test]
fn a_one_column_input_does_not_respell_a_wider_result() {
    let k = unmatched_exp_family();
    let e = match lowered(&k) {
        Err(e) => e,
        Ok(ops) => panic!(
            "the extents guard must refuse a [32, 64] result over a [32, 1] input, not lower \
             {} op(s)",
            ops.len()
        ),
    };
    assert!(
        e.contains("[32, 64]"),
        "the refusal names the stated [32, 64] output, got: {e}"
    );
}

/// ⛔ A WIDE INPUT's ROWS ARE NOT A FRAME FACT: a `[32, 64]` result over a `[2048, 64]` table
/// has numel == the table's rows (the gather-copy trap — a materialized copy's result is
/// `[entries, width]` while the table's rows can coincide with its numel), and it must keep
/// its STATED shape: the refusal names `[32, 64]` as the output, never a re-framed
/// `[2048, 1]`. Only a reduce frames its data's row sweep; only a one-column input frames its
/// family.
#[test]
fn a_wide_inputs_rows_do_not_frame_a_non_reduce_result() {
    let k = wide_table_add();
    let e = match lowered(&k) {
        Err(e) => e,
        Ok(ops) => panic!(
            "the extents guard must refuse a [32, 64] result over a [2048, 64] table, not \
             lower {} op(s)",
            ops.len()
        ),
    };
    assert!(
        e.contains("[32, 64]"),
        "the refusal names the STATED output shape (the mint kept it verbatim), got: {e}"
    );
    assert!(
        !e.contains("[2048, 1]"),
        "a wide input's rows must never re-frame a non-reduce result, got: {e}"
    );
}

/// The numel-guard control's program: `exp` of a `[32, 1]` load stating a `[32, 64]` result —
/// a one-column input at 32 rows, a result of numel 2048, so no family fact applies.
fn unmatched_exp_family() -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };
    let (p_x, p_out) = (Ssa(0), Ssa(1));
    let mut next = 2u32;
    let mut ssa = || {
        let s = Ssa(next);
        next += 1;
        s
    };
    let (v_x, v_o) = (ssa(), ssa());
    let mut ops = vec![
        shape(
            Operation::new(a, Some(v_x), OpKind::KtdpConstructMemoryView, &[p_x]),
            32,
            1,
        ),
        shape(
            Operation::new(a, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
            32,
            64,
        ),
    ];
    let c_zero = ssa();
    ops.push(
        Operation::new(a, Some(c_zero), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(0),
        ),
    );
    let (t_x, l_x) = (ssa(), ssa());
    ops.push(shape(
        Operation::new(
            a,
            Some(t_x),
            OpKind::KtdpConstructAccessTile,
            &[v_x, c_zero, c_zero],
        ),
        32,
        1,
    ));
    ops.push(shape(
        Operation::new(a, Some(l_x), OpKind::KtdpLoad, &[t_x]),
        32,
        1,
    ));
    let exp_res = ssa();
    let mut exp = shape(
        Operation::new(a, Some(exp_res), OpKind::MathExp, &[l_x]),
        32,
        64,
    );
    exp.result_type = Some(IrType::Tensor {
        dims: a.ints(vec![32, 64]),
        elem: DType::F16,
    });
    ops.push(exp);
    let t_o = ssa();
    ops.push(shape(
        Operation::new(
            a,
            Some(t_o),
            OpKind::KtdpConstructAccessTile,
            &[v_o, c_zero, c_zero],
        ),
        32,
        64,
    ));
    ops.push(Operation::new(a, None, OpKind::KtdpStore, &[exp_res, t_o]));
    KtirNode {
        func: IRFunction {
            name: "unmatched_exp",
            arguments: a.args(vec![(p_x, IrType::Index), (p_out, IrType::Index)]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        },
        program: Program::Elementwise(Elementwise::Exp),
        bindings: vec![BufferId::new(401), BufferId::new(402)],
        mask: None,
        step_inputs: Vec::new(),
        node_out_tid: None,
        minted: Default::default(),
    }
}

/// The wide-table control's program: `add` of a `[2048, 64]` table load and a `[32, 64]`
/// load, stating a `[32, 64]` result — numel 2048 equals the table's rows, the gather-copy
/// trap the ⛔ paragraph of [`respelled_frame_rows`]'s doc records.
fn wide_table_add() -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };
    let (p_tab, p_x, p_out) = (Ssa(0), Ssa(1), Ssa(2));
    let mut next = 3u32;
    let mut ssa = || {
        let s = Ssa(next);
        next += 1;
        s
    };
    let (v_tab, v_x, v_o) = (ssa(), ssa(), ssa());
    let mut ops = vec![
        shape(
            Operation::new(a, Some(v_tab), OpKind::KtdpConstructMemoryView, &[p_tab]),
            2048,
            64,
        ),
        shape(
            Operation::new(a, Some(v_x), OpKind::KtdpConstructMemoryView, &[p_x]),
            32,
            64,
        ),
        shape(
            Operation::new(a, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
            32,
            64,
        ),
    ];
    let c_zero = ssa();
    ops.push(
        Operation::new(a, Some(c_zero), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(0),
        ),
    );
    let mut loads = vec![];
    for (v, rows) in [(v_tab, 2048), (v_x, 32)] {
        let (t, l) = (ssa(), ssa());
        ops.push(shape(
            Operation::new(
                a,
                Some(t),
                OpKind::KtdpConstructAccessTile,
                &[v, c_zero, c_zero],
            ),
            rows,
            64,
        ));
        ops.push(shape(
            Operation::new(a, Some(l), OpKind::KtdpLoad, &[t]),
            rows,
            64,
        ));
        loads.push(l);
    }
    let add = ssa();
    let mut add_op = shape(
        Operation::new(a, Some(add), OpKind::ArithAddf, &[loads[0], loads[1]]),
        32,
        64,
    );
    add_op.result_type = Some(IrType::Tensor {
        dims: a.ints(vec![32, 64]),
        elem: DType::F16,
    });
    ops.push(add_op);
    let t_o = ssa();
    ops.push(shape(
        Operation::new(
            a,
            Some(t_o),
            OpKind::KtdpConstructAccessTile,
            &[v_o, c_zero, c_zero],
        ),
        32,
        64,
    ));
    ops.push(Operation::new(a, None, OpKind::KtdpStore, &[add, t_o]));
    KtirNode {
        func: IRFunction {
            name: "wide_table_add",
            arguments: a.args(vec![
                (p_tab, IrType::Index),
                (p_x, IrType::Index),
                (p_out, IrType::Index),
            ]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        },
        program: Program::Elementwise(Elementwise::Add),
        bindings: vec![BufferId::new(401), BufferId::new(402), BufferId::new(403)],
        mask: None,
        step_inputs: Vec::new(),
        node_out_tid: None,
        minted: Default::default(),
    }
}
