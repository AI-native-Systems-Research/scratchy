// SPDX-License-Identifier: Apache-2.0

//! ⭐⭐⭐⭐⭐ WALL 13's ISLAND — THE WALK THREADS THE DATA'S OWN `df`, pinned at the WALK level.
//!
//! The door commit (a48cd051e) made every door df-aware but the walk still handed each region
//! `Df::Fp16`; this is the island that closes that: a parameter's `df` is read off its own
//! `ktdp.construct_memory_view`'s `Dtype` (`regions`), an intermediate's off its result type
//! (`synth_df`, 4 B/elem at f32), a dtype CAST is a first-class CONVERT (`dl16tofp32` /
//! `fp32todl16`), and an elementwise op at f32 states IEEE_FP32 on every operand.
//!
//! # WHAT THESE PINS PIN
//!
//! * THE AUDITOR'S WATCH-ITEM, THE HEADLINE: the `m`/`l` merge's own shape — `[32, 8, 4]` reduced
//!   over dims `[0]` to `[8, 4]`, 32 survivor lanes — REFUSED at the fp16 basis for the whole
//!   campaign. At the data's own F32 it now LOWERS: 32 lanes are EXACTLY one whole fp32 stick,
//!   `sumnonstick`, `IEEE_FP32` on every operand. The refusal DISAPPEARED at the f32 data; the
//!   f16 spelling of the same program keeps the refusal (pinned in
//!   `zz_the_leading_axis_reduce_lowers_nonstick`, the no-drift control).
//! * THE CAST PINS: `arith.extf` f16→f32 lowers as ONE `dl16tofp32` convert (input SEN169_FP16,
//!   output IEEE_FP32 — the two `primaryDsInfo` sticks a convert needs); `arith.truncf` f32→f16
//!   as `fp32todl16`.
//! * ⛔ THE MIXED-df PIN (the auditor's law): an elementwise op whose input views are f16 and
//!   whose output view is f32 REFUSES BY NAME — a mixed-df TTIR op never emits as ONE op, so
//!   the DD2 mixed-op refusal is unreachable by our emission. The CAST is the only legal
//!   mixed-df shape, and it emits as a convert with each side at its own df.
//! * ⛔ THE CAST INVENTORY'S REFUSALS: a MISTYPED `extf` (stating f32→f16 — extf widens) and a
//!   FOREIGN pair (`convertf` at f16→fp8) refuse BY NAME, the closed inventory.
//! * ⛔ THE FP32 ALLOWLIST, both laws named (never conflated): `tanh` at f32 refuses as
//!   HARDWARE-ABSENT (the DDL's own FEST comment); `rsqrt` at f32 refuses as outside the
//!   ISLAND's allowlist (the vendor binds it at fp32; this island has not tested it).
//! * THE FOOTPRINT PIN: an f32 INTERMEDIATE is reserved at 4 B/elem (`synth_df`'s law) — the
//!   minted `[8, 64]` f32 add result reserves 2048 B, not 1024.
//!
//! The ddc leg (`matchDdl2Dsc` for real) and the numerics close at the BAKE on card — the same
//! closure law every reduce pin records.

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

/// One f32 element's bytes — the footprint law's multiplier.
const F32_BYTES: u64 = 4;

/// The merge's own geometry at B=1: `[256, 4]` flat, `[32, 8, 4]` spelled, dims `[0]`, result
/// `[8, 4]` — 32 survivor lanes, EXACTLY one whole fp32 stick.
type Merge = ((i64, i64), &'static [i64], &'static [i64], (i64, i64));
const MERGE: Merge = ((256, 4), &[32, 8, 4], &[0], (8, 4));

/// The leading-axis merge program at a STATED element type — `leading_axis_case`'s own spelling
/// (the wall-12 file's builder), with the dtype a parameter so the f32 island and the f16 control
/// are the SAME program shape and only the data's own format moves.
fn merge_case(
    flat: (i64, i64),
    reshape: &[i64],
    dims: &[i64],
    result: (i64, i64),
    reduce_fn: OpKind,
    dt: DType,
) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(dt))
    };
    let ty = |r: i64, c: i64| {
        Some(IrType::Tensor {
            dims: a.ints(vec![r, c]),
            elem: dt,
        })
    };

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
            flat.0,
            flat.1,
        ),
        shape(
            Operation::new(a, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
            result.0,
            result.1,
        ),
    ];

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
        flat.0,
        flat.1,
    ));
    let mut load = shape(
        Operation::new(a, Some(l_q), OpKind::KtdpLoad, &[t_q]),
        flat.0,
        flat.1,
    );
    load.result_type = ty(flat.0, flat.1);
    ops.push(load);
    let mut reshape_op = Operation::new(a, Some(rs), OpKind::TensorReshape, &[l_q])
        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(reshape.to_vec())))
        .with_attr(a, AttrKey::Dtype, Attr::Dtype(dt))
        .with_attr(
            a,
            AttrKey::TargetShape,
            Attr::IntList(a.ints(reshape.to_vec())),
        );
    reshape_op.result_type = Some(IrType::Tensor {
        dims: a.ints(reshape.to_vec()),
        elem: dt,
    });
    ops.push(reshape_op);

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
        result.0,
        result.1,
    ));
    let mut reduce = shape(
        Operation::new(a, Some(red), OpKind::LinalgReduce, &[rs, splat]),
        result.0,
        result.1,
    )
    .with_attr(a, AttrKey::ReduceFn, Attr::Op(reduce_fn))
    .with_attr(a, AttrKey::Dimensions, Attr::IntList(a.ints(dims.to_vec())));
    reduce.result_type = ty(result.0, result.1);
    ops.push(reduce);

    let t_o = ssa();
    ops.push(shape(
        Operation::new(
            a,
            Some(t_o),
            OpKind::KtdpConstructAccessTile,
            &[v_o, c_zero, c_zero2],
        ),
        result.0,
        result.1,
    ));
    ops.push(Operation::new(a, None, OpKind::KtdpStore, &[red, t_o]));

    KtirNode {
        func: IRFunction {
            name: "merge_case",
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

/// A three-view unary program — `out = kind(load(x))` — the cast and unary-allowlist cases' own
/// spelling: one input view, one output view, a load, the op, a store. `in_dt`/`out_dt` state the
/// two sides' element types, so a cast program and a same-dtype unary are the SAME shape.
fn unary_case(kind: OpKind, rows: i64, cols: i64, in_dt: DType, out_dt: DType) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64, dt: DType| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(dt))
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
            rows,
            cols,
            in_dt,
        ),
        shape(
            Operation::new(a, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
            rows,
            cols,
            out_dt,
        ),
    ];
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
    let (t_x, l_x) = (ssa(), ssa());
    ops.push(shape(
        Operation::new(
            a,
            Some(t_x),
            OpKind::KtdpConstructAccessTile,
            &[v_x, c_zero, c_zero2],
        ),
        rows,
        cols,
        in_dt,
    ));
    let mut load = shape(
        Operation::new(a, Some(l_x), OpKind::KtdpLoad, &[t_x]),
        rows,
        cols,
        in_dt,
    );
    load.result_type = Some(IrType::Tensor {
        dims: a.ints(vec![rows, cols]),
        elem: in_dt,
    });
    ops.push(load);

    let r = ssa();
    let mut op = shape(Operation::new(a, Some(r), kind, &[l_x]), rows, cols, out_dt);
    op.result_type = Some(IrType::Tensor {
        dims: a.ints(vec![rows, cols]),
        elem: out_dt,
    });
    ops.push(op);

    let t_o = ssa();
    ops.push(shape(
        Operation::new(
            a,
            Some(t_o),
            OpKind::KtdpConstructAccessTile,
            &[v_o, c_zero, c_zero2],
        ),
        rows,
        cols,
        out_dt,
    ));
    ops.push(Operation::new(a, None, OpKind::KtdpStore, &[r, t_o]));

    KtirNode {
        func: IRFunction {
            name: "unary_case",
            arguments: a.args(vec![(p_x, IrType::Index), (p_out, IrType::Index)]),
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

/// A two-view binary program — `out = kind(load(x), load(y))` — the elementwise and mixed-df
/// cases' own spelling. `x_dt`/`y_dt`/`out_dt` state each side independently, so the mixed-df
/// refusal is stated by the PROGRAM, not patched into the walk.
fn binary_case(
    kind: OpKind,
    rows: i64,
    cols: i64,
    x_dt: DType,
    y_dt: DType,
    out_dt: DType,
) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64, dt: DType| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(dt))
    };

    let (p_x, p_y, p_out) = (Ssa(0), Ssa(1), Ssa(2));
    let mut next = 3u32;
    let mut ssa = || {
        let s = Ssa(next);
        next += 1;
        s
    };

    let (v_x, v_y, v_o) = (ssa(), ssa(), ssa());
    let mut ops = vec![
        shape(
            Operation::new(a, Some(v_x), OpKind::KtdpConstructMemoryView, &[p_x]),
            rows,
            cols,
            x_dt,
        ),
        shape(
            Operation::new(a, Some(v_y), OpKind::KtdpConstructMemoryView, &[p_y]),
            rows,
            cols,
            y_dt,
        ),
        shape(
            Operation::new(a, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
            rows,
            cols,
            out_dt,
        ),
    ];
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
    let mut loads = Vec::new();
    for (ptr, view, dt) in [(p_x, v_x, x_dt), (p_y, v_y, y_dt)] {
        let (t, l) = (ssa(), ssa());
        ops.push(shape(
            Operation::new(
                a,
                Some(t),
                OpKind::KtdpConstructAccessTile,
                &[view, c_zero, c_zero2],
            ),
            rows,
            cols,
            dt,
        ));
        let mut load = shape(
            Operation::new(a, Some(l), OpKind::KtdpLoad, &[t]),
            rows,
            cols,
            dt,
        );
        load.result_type = Some(IrType::Tensor {
            dims: a.ints(vec![rows, cols]),
            elem: dt,
        });
        ops.push(load);
        loads.push(l);
        let _ = ptr;
    }

    let r = ssa();
    let mut op = shape(
        Operation::new(a, Some(r), kind, &[loads[0], loads[1]]),
        rows,
        cols,
        out_dt,
    );
    op.result_type = Some(IrType::Tensor {
        dims: a.ints(vec![rows, cols]),
        elem: out_dt,
    });
    ops.push(op);

    let t_o = ssa();
    ops.push(shape(
        Operation::new(
            a,
            Some(t_o),
            OpKind::KtdpConstructAccessTile,
            &[v_o, c_zero, c_zero2],
        ),
        rows,
        cols,
        out_dt,
    ));
    ops.push(Operation::new(a, None, OpKind::KtdpStore, &[r, t_o]));

    KtirNode {
        func: IRFunction {
            name: "binary_case",
            arguments: a.args(vec![
                (p_x, IrType::Index),
                (p_y, IrType::Index),
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

/// A fully-placed layout for the parameters, each at its OWN bytes-per-element — the placement
/// is the second authority the window mint checks, so an f32 view must reserve 4 B/elem.
fn layout_for(k: &KtirNode, numel: i64, bytes_per_elem: u64) -> BundleLayout {
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
                size: (numel as u64) * bytes_per_elem,
                cover: ktir_superdsc::placement::BindCover::Full,
            },
        );
    }
    l
}

fn lowered(
    k: &KtirNode,
    numel: i64,
    bytes_per_elem: u64,
) -> Result<Vec<ktir_superdsc::emit::EmittedOp>, String> {
    let layout = layout_for(k, numel, bytes_per_elem);
    let mut sid = 0i64;
    lower_function(k, Some(&layout), &mut sid, None).map_err(|e| e.message)
}

fn op_json(op: &ktir_superdsc::emit::EmittedOp) -> serde_json::Value {
    serde_json::to_value(&op.op).expect("the op serializes")
}

/// The single `dscs_[0]` body of an emitted op.
fn body(j: &serde_json::Value) -> &serde_json::Value {
    j["dscs_"][0]
        .as_object()
        .expect("the op has a dsc body")
        .values()
        .next()
        .expect("the body has an entry")
}

/// ⭐⭐⭐ THE HEADLINE (the auditor's watch-item): the `m`/`l` merge's own shape at the data's
/// own F32 — `[32, 8, 4]` reduced over dims `[0]` to `[8, 4]`, the 32 survivor lanes that
/// REFUSED at the fp16 basis for the whole campaign — LOWERS through the walk as ONE fp32
/// nonstick descriptor. The refusal DISAPPEARED at the f32 data: 32 lanes are EXACTLY one whole
/// fp32 stick, and every operand states IEEE_FP32.
#[test]
fn the_m_merge_lowers_at_f32_through_the_walk() {
    let ((fr, fc), reshape, dims, result) = MERGE;
    let k = merge_case(
        (fr, fc),
        reshape,
        dims,
        result,
        OpKind::ArithAddf,
        DType::F32,
    );
    let ops = lowered(&k, fr * fc, F32_BYTES).unwrap_or_else(|e| {
        panic!("the f32 m-merge geometry must lower through the walk, refused: {e}")
    });
    let [red] = &ops[..] else {
        panic!(
            "the merge is ONE op, got {:?}",
            ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
        )
    };
    let j = op_json(red);
    let c = &body(&j)["computeOp_"][0];
    assert_eq!(
        c["opFuncName"], "sumnonstick",
        "the fp32 family's wire name"
    );
    assert_eq!(
        c["exUnit"], "sfp",
        "RCUDD1A has no fp32 PE — fp32 lives on SFP"
    );
    // 32 survivor lanes = EXACTLY one whole fp32 stick.
    assert_eq!(body(&j)["N_"]["y_"].as_u64(), Some(32));
    // Every operand states the data's own format — the walk threaded it from the VIEW's Dtype.
    for lds in body(&j)["labeledDs_"]
        .as_array()
        .expect("the labeledDs list")
    {
        assert_eq!(
            lds["dataFormat_"], "IEEE_FP32",
            "an f32 merge operand's dataFormat, walked off its own view"
        );
        assert_eq!(lds["wordLength"].as_u64(), Some(4));
    }
}

/// ⭐ THE EXT CAST: `arith.extf` f16→f32 over `[8, 64]` lowers as ONE `dl16tofp32` convert —
/// the input at SEN169_FP16 (wordLength 2), the output at IEEE_FP32 (wordLength 4), the TWO
/// distinct sticks a convert needs (its input and output have different sticks, which is why it
/// is never a pointwise op).
#[test]
fn an_extf_lowers_as_a_first_class_convert() {
    let k = unary_case(OpKind::ArithExtf, 8, 64, DType::F16, DType::F32);
    let ops = lowered(&k, 512, F32_BYTES).unwrap_or_else(|e| {
        panic!("the f16→f32 widening cast must lower as a convert, refused: {e}")
    });
    let [cvt] = &ops[..] else {
        panic!("the cast is ONE op, got {} ops", ops.len())
    };
    assert!(
        cvt.op_name.starts_with("dl16tofp32_o"),
        "the convert's own op name, got {}",
        cvt.op_name
    );
    let j = op_json(cvt);
    let c = &body(&j)["computeOp_"][0];
    assert_eq!(c["opFuncName"], "dl16tofp32", "the vendor's widening bind");
    // Each side at its OWN stick: the input a 64-lane SEN169 word, the output a 32-lane IEEE one.
    let lds = body(&j)["labeledDs_"]
        .as_array()
        .expect("the labeledDs list");
    let formats: Vec<&str> = lds
        .iter()
        .map(|e| e["dataFormat_"].as_str().expect("a format"))
        .collect();
    assert!(
        formats.contains(&"SEN169_FP16"),
        "the input side keeps its f16 word (got {formats:?})"
    );
    assert!(
        formats.contains(&"IEEE_FP32"),
        "the output side states the f32 word (got {formats:?})"
    );
    for e in lds {
        let wl = e["wordLength"].as_u64().expect("a word length");
        assert!(
            wl == 2 || wl == 4,
            "each side at its own word length, got {wl}"
        );
    }
}

/// ⭐ THE TRUNC TWIN: `arith.truncf` f32→f16 lowers as `fp32todl16` — the narrowing bind.
#[test]
fn a_truncf_lowers_as_the_narrowing_convert() {
    let k = unary_case(OpKind::ArithTruncf, 8, 64, DType::F32, DType::F16);
    let ops = lowered(&k, 512, F32_BYTES)
        .unwrap_or_else(|e| panic!("the f32→f16 narrowing cast must lower, refused: {e}"));
    let [cvt] = &ops[..] else {
        panic!("the cast is ONE op, got {} ops", ops.len())
    };
    assert!(
        cvt.op_name.starts_with("fp32todl16_o"),
        "the narrowing convert's own op name, got {}",
        cvt.op_name
    );
    let j = op_json(cvt);
    assert_eq!(
        body(&j)["computeOp_"][0]["opFuncName"],
        "fp32todl16",
        "the vendor's narrowing bind"
    );
}

/// ⛔ THE MIXED-df PIN (the auditor's law): an `arith.addf` whose input views are f16 and whose
/// output view is f32 REFUSES BY NAME — a mixed-df TTIR op never emits as ONE op. The legal
/// spelling is the cast chain (`extf` then `addf` at f32), which the two tests above pin; this
/// pin is what makes the DD2 mixed-op refusal unreachable by our emission.
#[test]
fn a_mixed_df_elementwise_refuses_by_name() {
    let k = binary_case(OpKind::ArithAddf, 8, 64, DType::F16, DType::F16, DType::F32);
    let e = match lowered(&k, 512, F32_BYTES) {
        Err(e) => e,
        Ok(ops) => panic!(
            "a mixed-df elementwise op must refuse, not lower {} op(s)",
            ops.len()
        ),
    };
    assert!(
        e.contains("MIXED-df"),
        "the refusal names the mixed-df law, got: {e}"
    );
    assert!(
        e.contains("arith.extf") || e.contains("arith.truncf"),
        "the refusal names the legal spelling (the explicit cast), got: {e}"
    );
}

/// ⛔ THE MISTYPED CAST: an `arith.extf` stating f32→f16 (extf WIDENS; this program truncates)
/// refuses BY NAME as mistyped — its value's bytes would re-interpret at the wrong width no
/// matter which convert emitted them.
#[test]
fn a_mistyped_extf_refuses_by_name() {
    let k = unary_case(OpKind::ArithExtf, 8, 64, DType::F32, DType::F16);
    let e = match lowered(&k, 512, F32_BYTES) {
        Err(e) => e,
        Ok(ops) => panic!("a mistyped extf must refuse, not lower {} op(s)", ops.len()),
    };
    assert!(
        e.contains("WIDENING"),
        "the refusal names the direction law, got: {e}"
    );
}

/// ⛔ THE FOREIGN PAIR: `arith.convertf` at f16→fp8 refuses BY NAME — the closed inventory
/// vendor-binds only the fp16↔fp32 pair, and routing any other transition through a
/// nearest-neighbor opfunc would silently re-interpret the bytes.
#[test]
fn a_foreign_pair_convert_refuses_by_name() {
    let k = unary_case(OpKind::ArithConvertf, 8, 64, DType::F16, DType::Fp8E4m3);
    let e = match lowered(&k, 512, 2) {
        Err(e) => e,
        Ok(ops) => panic!(
            "a foreign-pair convert must refuse, not lower {} op(s)",
            ops.len()
        ),
    };
    assert!(
        e.contains("closed convert inventory"),
        "the refusal names the closed inventory, got: {e}"
    );
}

/// ⛔ THE FP32 ALLOWLIST, HARDWARE GROUP: `math.tanh` at f32 refuses citing the HARDWARE (the
/// DDL's own FEST comment) — never "not in the allowlist", which is the other group's law.
#[test]
fn tanh_at_f32_refuses_citing_the_hardware() {
    let k = unary_case(OpKind::MathTanh, 8, 64, DType::F32, DType::F32);
    let e = match lowered(&k, 512, F32_BYTES) {
        Err(e) => e,
        Ok(ops) => panic!(
            "tanh at f32 must refuse as hardware-absent, not lower {} op(s)",
            ops.len()
        ),
    };
    assert!(
        e.contains("HARDWARE-ABSENT"),
        "the refusal names the hardware law, got: {e}"
    );
    assert!(
        e.contains("FEST"),
        "the refusal cites the DDL's own comment, got: {e}"
    );
}

/// ⛔ THE FP32 ALLOWLIST, ISLAND GROUP: `math.rsqrt` at f32 refuses as OUTSIDE THE ISLAND's
/// allowlist — the vendor DOES bind it at fp32 (unary_parallel.ddl), so the refusal must say
/// the island is narrow, never that the hardware lacks it. Two laws, two messages, never
/// conflated.
#[test]
fn rsqrt_at_f32_refuses_citing_the_island_allowlist() {
    let k = unary_case(OpKind::MathRsqrt, 8, 64, DType::F32, DType::F32);
    let e = match lowered(&k, 512, F32_BYTES) {
        Err(e) => e,
        Ok(ops) => panic!(
            "rsqrt at f32 must refuse as outside the island's allowlist, not lower {} op(s)",
            ops.len()
        ),
    };
    assert!(
        e.contains("not in the fp32 island's allowlist"),
        "the refusal names the ISLAND's allowlist, got: {e}"
    );
    assert!(
        e.contains("vendor DOES bind"),
        "the refusal states the vendor's own bind honestly, got: {e}"
    );
}

/// The footprint pin's own spelling: `out16 = truncf(load32(x) + load32(y))` — TWO f32 views, an
/// f16 view, two f32 loads, an f32 `addf` whose result is consumed (never stored → the MINTED
/// intermediate this test prices), the `truncf`, the store. The addf's result is the island's
/// whole subject in one value: minted at the data's own F32, 4 B/elem.
fn f32_add_then_trunc_case(rows: i64, cols: i64) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64, dt: DType| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(dt))
    };
    let ty = |r: i64, c: i64, dt: DType| {
        Some(IrType::Tensor {
            dims: a.ints(vec![r, c]),
            elem: dt,
        })
    };

    let (p_x, p_y, p_out) = (Ssa(0), Ssa(1), Ssa(2));
    let mut next = 3u32;
    let mut ssa = || {
        let s = Ssa(next);
        next += 1;
        s
    };

    let (v_x, v_y, v_o) = (ssa(), ssa(), ssa());
    let mut ops = vec![
        shape(
            Operation::new(a, Some(v_x), OpKind::KtdpConstructMemoryView, &[p_x]),
            rows,
            cols,
            DType::F32,
        ),
        shape(
            Operation::new(a, Some(v_y), OpKind::KtdpConstructMemoryView, &[p_y]),
            rows,
            cols,
            DType::F32,
        ),
        shape(
            Operation::new(a, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
            rows,
            cols,
            DType::F16,
        ),
    ];
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
    let mut loads = Vec::new();
    for view in [v_x, v_y] {
        let (t, l) = (ssa(), ssa());
        ops.push(shape(
            Operation::new(
                a,
                Some(t),
                OpKind::KtdpConstructAccessTile,
                &[view, c_zero, c_zero2],
            ),
            rows,
            cols,
            DType::F32,
        ));
        let mut load = shape(
            Operation::new(a, Some(l), OpKind::KtdpLoad, &[t]),
            rows,
            cols,
            DType::F32,
        );
        load.result_type = ty(rows, cols, DType::F32);
        ops.push(load);
        loads.push(l);
    }

    // The f32 addf — its result is consumed by the truncf, never stored: the MINTED intermediate.
    let add = ssa();
    let mut addf = shape(
        Operation::new(a, Some(add), OpKind::ArithAddf, &[loads[0], loads[1]]),
        rows,
        cols,
        DType::F32,
    );
    addf.result_type = ty(rows, cols, DType::F32);
    ops.push(addf);

    // The narrowing cast off the intermediate.
    let tr = ssa();
    let mut trunc = shape(
        Operation::new(a, Some(tr), OpKind::ArithTruncf, &[add]),
        rows,
        cols,
        DType::F16,
    );
    trunc.result_type = ty(rows, cols, DType::F16);
    ops.push(trunc);

    let t_o = ssa();
    ops.push(shape(
        Operation::new(
            a,
            Some(t_o),
            OpKind::KtdpConstructAccessTile,
            &[v_o, c_zero, c_zero2],
        ),
        rows,
        cols,
        DType::F16,
    ));
    ops.push(Operation::new(a, None, OpKind::KtdpStore, &[tr, t_o]));

    KtirNode {
        func: IRFunction {
            name: "f32_add_then_trunc",
            arguments: a.args(vec![
                (p_x, IrType::Index),
                (p_y, IrType::Index),
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

/// ⭐ THE FOOTPRINT PIN (the auditor's condition 6): an f32 INTERMEDIATE is reserved at
/// 4 B/elem. The program stores a `truncf` of an f32 `addf` result, so the addf's result is a
/// minted intermediate: its `[8, 64]` view at f32 reserves 2048 B, not the 1024 an fp16
/// annotation would state — the reservation is read straight out of the layout's own synth
/// sizes, the allocator `resolve_seg_base` guards every access against. The chain also lowers
/// end to end (the f32 `addf` then the `fp32todl16` convert), so the mint and the convert door
/// compose on one value.
#[test]
fn an_f32_intermediate_is_reserved_at_four_bytes_per_element() {
    let (rows, cols) = (8i64, 64i64);
    let k = f32_add_then_trunc_case(rows, cols);
    let layout = layout_for(&k, rows * cols, F32_BYTES);
    let mut sid = 0i64;
    let ops = lower_function(&k, Some(&layout), &mut sid, None)
        .unwrap_or_else(|e| panic!("the f32 chain must lower, refused: {}", e.message));
    // TWO ops: the f32 addf and the fp32todl16 convert off its intermediate.
    let names: Vec<&str> = ops.iter().map(|o| o.op_name.as_str()).collect();
    assert_eq!(
        ops.len(),
        2,
        "the addf and the convert, nothing else (got {names:?})"
    );
    assert!(
        names.iter().any(|n| n.starts_with("add_o")),
        "the f32 addf lowered (got {names:?})"
    );
    assert!(
        names.iter().any(|n| n.starts_with("fp32todl16_o")),
        "the convert off the intermediate lowered (got {names:?})"
    );
    // THE FOOTPRINT: the minted intermediate (the addf's result, [8, 64] at f32) reserved
    // 8·64·4 = 2048 B — the synth allocator's own sizes map, the authority every access is
    // guarded against. An fp16 annotation would have declared 1024 B and every past-the-half
    // access would alias the next intermediate.
    let sizes = layout.synth.borrow();
    let want = (rows as u64) * (cols as u64) * F32_BYTES;
    let f32_mints: Vec<(&String, &u64)> = sizes.sizes.iter().filter(|&(_, &b)| b == want).collect();
    assert!(
        !f32_mints.is_empty(),
        "an [8, 64] f32 intermediate reserved {want} B (4 B/elem) — got {:?}",
        sizes.sizes
    );
    assert!(
        !sizes.sizes.values().any(|&b| b == want / 2),
        "no [8, 64] mint was priced at the fp16 1024 B — the footprint is the data's own df"
    );
}
