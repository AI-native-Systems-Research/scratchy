// SPDX-License-Identifier: Apache-2.0

//! ⭐⭐⭐⭐⭐ WALL 12 — THE LEADING-AXIS REDUCE, AS TESTS.
//!
//! A reduce whose `Dimensions` state a LEADING PREFIX (and not the whole rank) is the DDL's
//! NONSTICK form (`summeanmaxexx2.ddl:39/:45`): the survivors stride 1 in row-major flat
//! (`reduced · survivor_cols + survivor`), so the accumulator is the SURVIVOR VECTOR, not one
//! stick per row. The door classifies the block off the op's own `Dimensions`/`Shape`
//! (`reduce_axis_block`) and the geometry off the op's own numels (`nonstick_geometry`) —
//! never the walk's threaded frame (wall 11's law).
//!
//! # THE ORACLE, AND WHAT IT DOES AND DOES NOT LICENSE
//!
//! `ddc/ddl_templates/test/sdsc_maxnonstick.json` is the only descriptor anywhere that names a
//! nonstick opFunc. It is a SENULATOR structural smoke (`i_ = 1`, `target_ "senulator"`) — no
//! accumulation semantics — so this is NOT byte parity and no test below claims one. What it
//! pins — read out of IT and out of the DDL, not out of our own emission:
//!
//! ```text
//!   computeOp_      exUnit "sfp", opFuncName "maxnonstick",
//!                   inputLabeledDs [data-idx0, accum-idx1] (TWO), outputLabeledDs [accum-idx1]
//!                   (ONE), interimLabeledDs [] (EMPTY — no exx2-style interim tensor)
//!   layouts         layoutDimOrder_ ["out","i","mb","x"], stickDimOrder_ ["out"] (the SURVIVOR
//!                   is the stick), stickSize_ [64]
//!   scales          data scale_ [1,1,1,1]; accum scale_ [1,-1,1,1] — the −1 (RedNonStick) on
//!                   the REDUCED dim, the survivor stick Active
//!   division        numWkSlicesPerDim_ {"mb": 8} — the SURVIVOR splits, the reduced dim
//!                   ("i") UNSPLIT; ringless (no genericpartialreduction node)
//!   masking         coordinateMasking_ {} — no SAMV mask (a whole-stick survivor has no pad)
//! ```
//!
//! # THE DOCUMENTED DIVERGENCES (asserted or stated, not hidden)
//!
//! • DIM NAMES — this crate's reduce vocabulary names the REDUCED dim `out` (the DDL's own
//!   `%reduce_dim` "OUT dimension"); the FIXTURE's `out` is its SURVIVOR stick. The names never
//!   reach matchDdl2Dsc (it binds by layout and opFuncName, `ddl_conversion.cpp:2117-2122` /
//!   `:2360-2460`); the ROLES are what bind, pinned here by the scales.
//! • CONSTANTINFO — the fixture ships the FALSY empty object, but it is the senulator form; this
//!   emission mirrors the card-proven STICK form's `scaling_factor` at 1.0 (the DDL gates the
//!   scaling transfer on the MEAN branch only, `:133-137`, so sum/max never read it — the same
//!   `OpInfo` the stick form ships, PR #813's five card-proven `tt.reduce` maxima). The bake on
//!   card arbitrates; no test here claims either side.
//! • The fixture's 4-dim layout (`["out","i","mb","x"]`) folds to this builder's 2-dim
//!   `["out","y"]` — the minimal form at a geometry with no instance dims, the same folding the
//!   topk pair's 3-dim form makes of the fixtures' 4-dim one.
//!
//! # WHAT THE DOOR-LEVEL TESTS PIN
//!
//! * THE POSITIVE: `[1024, 64]` data reshaped `[32, 32, 64]`, reduced over dims `[0]` to a
//!   `[32, 64]` survivor vector (2048 lanes = 32 sticks), LOWERS as ONE nonstick descriptor —
//!   sum AND max spellings — with the fixture's structural contract.
//! * ⛔ CONTROL (wall 13): the online-softmax `m`/`l` merge shape (`[32, 8, 4] → [8, 4]`, 32
//!   survivor lanes = half a stick) REFUSES by name — SUB-STICK, our side's certain fact only.
//! * ⛔ CONTROL (no-drift, refusal): a MIDDLE-axis reduce (`[32, 32, 64]` reducing dims `[1]`)
//!   keeps the row-reduce body's own refusal VERBATIM — "ONE value per row".
//! * ⛔ CONTROL (no-drift, lowering): a DEGENERATE middle reduce (`[8, 1, 64]` reducing dims
//!   `[1]` to `[8, 1]`) still LOWERS through the row-reduce path as the STICK form (`sum`,
//!   never `sumnonstick`) — byte-identical to before wall 12.
//!
//! The ddc leg (`matchDdl2Dsc` for real) closes at the BAKE on card — `cargo test` cannot run
//! ddc; the same closure law the topk pair's pins record.

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

/// A leading-axis reduce program: `out = reduce(reshape(load(flat)), dims)` — the real
/// walk's own spelling (a flat load, a numel-preserving reshape, a reduce over a stated dim
/// list, a store of the result). `flat` is the load frame; `reshape` the program's spelling;
/// `dims` the reduce's own `Dimensions`; `result` the reduce's stated result (also the store's
/// destination); `reduce_fn` the combiner.
fn leading_axis_case(
    flat: (i64, i64),
    reshape: &[i64],
    dims: &[i64],
    result: (i64, i64),
    reduce_fn: OpKind,
) -> KtirNode {
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

    // Parameters: %0 = data, %1 = out.
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

    // The flat load, then the program's own numel-preserving spelling of it.
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
        .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
        .with_attr(
            a,
            AttrKey::TargetShape,
            Attr::IntList(a.ints(reshape.to_vec())),
        );
    reshape_op.result_type = Some(IrType::Tensor {
        dims: a.ints(reshape.to_vec()),
        elem: DType::F16,
    });
    ops.push(reshape_op);

    // The reduce over the stated dims, splat-zero initial, `result` stated.
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

    // The store of the result.
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
            name: "leading_axis_reduce",
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

/// A fully-placed layout for the two parameters (the wall-11 pin's own helper, sized for this
/// program's flat frame).
fn layout_for(k: &KtirNode, flat_numel: i64) -> BundleLayout {
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
                size: (flat_numel * 2) as u64,
                cover: ktir_superdsc::placement::BindCover::Full,
            },
        );
    }
    l
}

fn lowered(k: &KtirNode, flat_numel: i64) -> Result<Vec<ktir_superdsc::emit::EmittedOp>, String> {
    let layout = layout_for(k, flat_numel);
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

/// The whole emitted op as JSON — the wire-level fields (`numWkSlicesPerDim_`, the computeOp
/// row, the labeledDs scales) live across both levels, same as the topk pair's pins.
fn op_json(op: &ktir_superdsc::emit::EmittedOp) -> serde_json::Value {
    serde_json::to_value(&op.op).expect("the op serializes")
}

/// The single computeOp row of the emitted nonstick descriptor.
fn compute_op(j: &serde_json::Value) -> &serde_json::Value {
    let body = j["dscs_"][0]
        .as_object()
        .expect("the op has a dsc body")
        .values()
        .next()
        .expect("the body has an entry");
    &body["computeOp_"][0]
}

/// The wall-12 positive geometry: `[1024, 64]` flat, `[32, 32, 64]` spelled, reduced over
/// dims `[0]` to the `[32, 64]` survivor vector — 65536 elements in, 2048 out, 2048 = 32
/// whole sticks.
type Pos = ((i64, i64), &'static [i64], &'static [i64], (i64, i64));
const POS: Pos = ((1024, 64), &[32, 32, 64], &[0], (32, 64));

/// ⭐ THE POSITIVE (sum): the leading-axis reduce LOWERS as ONE nonstick descriptor carrying
/// the fixture's structural contract — the wire name, the sfp unit, the seeded two-in/one-out
/// arity with an EMPTY interim list, the reduced `out` UNSPLIT with the survivor `y` split by
/// stick count, and the −1 accum scale on the REDUCED dim with the survivor stick Active.
#[test]
fn the_leading_axis_sum_lowers_nonstick() {
    let ((fr, fc), reshape, dims, result) = POS;
    let k = leading_axis_case((fr, fc), reshape, dims, result, OpKind::ArithAddf);
    let ops = lowered(&k, fr * fc).unwrap_or_else(|e| {
        panic!("the leading-axis sum must lower through the door, refused: {e}")
    });
    let [red] = &ops[..] else {
        panic!(
            "the slice is ONE op (the reduce), got {:?}",
            ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
        );
    };
    // The iteration space: the reduced dim at its full extent, the survivor at its whole-stick
    // extent — the door derived both from the op's own numels, never the flat frame's rows.
    assert_eq!(dsc(red).N_.out_, 32, "the reduced extent");
    assert_eq!(dsc(red).N_.y_, 2048, "the survivor lanes (32 whole sticks)");
    let j = op_json(red);
    let c = compute_op(&j);
    assert_eq!(c["opFuncName"], "sumnonstick", "the DDL's own wire name");
    assert_eq!(c["exUnit"], "sfp", "the typed surface's ex unit");
    assert_eq!(
        c["inputLabeledDs"].as_array().map(Vec::len),
        Some(2),
        "the seeded convention: data + accumulator"
    );
    assert_eq!(
        c["outputLabeledDs"].as_array().map(Vec::len),
        Some(1),
        "the accumulator is the sole output"
    );
    assert!(
        c["interimLabeledDs"]
            .as_array()
            .is_none_or(|a| a.is_empty()),
        "the nonstick binds carry NO interim tensor (the fixture's interimLabeledDs [] verbatim)"
    );
    // The division: the reduced dim NEVER splits (the accumulation axis — a split would need
    // the psum ring the fixture proves the vendor's own nonstick emission does not bind); the
    // cores take the survivor's sticks.
    assert_eq!(
        j["numWkSlicesPerDim_"]["out"], 1,
        "the reduced dim stays whole per core"
    );
    let y_split = j["numWkSlicesPerDim_"]["y"].as_u64().unwrap_or_else(|| {
        panic!(
            "the survivor split is stated, got {}",
            j["numWkSlicesPerDim_"]["y"]
        )
    });
    assert!(y_split >= 1, "the survivor split is at least one");
    // The scales — the fixture's own law: data all-Active, the accum −1 (RedNonStick) on the
    // REDUCED dim with the survivor stick Active. Read off the labeledDs entries.
    let body = j["dscs_"][0].as_object().unwrap().values().next().unwrap();
    let lds = body["labeledDs_"].as_array().expect("the labeledDs list");
    let scales: Vec<Vec<i64>> = lds
        .iter()
        .map(|e| {
            e["scale_"]
                .as_array()
                .expect("a labeledDs carries scale_")
                .iter()
                .map(|s| s.as_i64().expect("a scale is an integer"))
                .collect()
        })
        .collect();
    assert!(
        scales.contains(&vec![1, 1]),
        "the data operand is all-Active (got {scales:?})"
    );
    assert!(
        scales.contains(&vec![-1, 1]),
        "the accumulator carries RedNonStick (-1) on the REDUCED dim and Active on the survivor \
         stick (got {scales:?})"
    );
    // No SAMV mask — a whole-stick survivor with an exact product has no pad lane anywhere.
    let masking = serde_json::to_string(&body["coordinateMasking_"]).unwrap_or_default();
    assert!(
        masking == "null" || masking == "{}",
        "the nonstick form binds no coordinate mask (got {masking})"
    );
}

/// ⭐ THE POSITIVE (max): the same geometry under `arith.maxnumf` lowers as `maxnonstick` —
/// the DDL's own twin bind, never the stick form's name.
#[test]
fn the_leading_axis_max_lowers_nonstick() {
    let ((fr, fc), reshape, dims, result) = POS;
    let k = leading_axis_case((fr, fc), reshape, dims, result, OpKind::ArithMaxnumf);
    let ops = lowered(&k, fr * fc)
        .unwrap_or_else(|e| panic!("the leading-axis max must lower, refused: {e}"));
    let [red] = &ops[..] else {
        panic!("the slice is ONE op, got {} ops", ops.len())
    };
    let j = op_json(red);
    let c = compute_op(&j);
    assert_eq!(c["opFuncName"], "maxnonstick", "the max twin's wire name");
}

/// ⛔ WALL 13's OWN REFUSAL, PINNED: the online-softmax `m`/`l` merge shape — `[32, 8, 4]`
/// reduced over dims `[0]` to `[8, 4]`, 32 survivor lanes = HALF a stick — refuses BY NAME,
/// claiming only our side's certain fact (one `is_stick` ItDim; a compound stick is wall-13
/// investigation, not cited by the refusal). The flat frame is `[256, 4]` — the reshape's own
/// flat presentation, so the re-row law passes and the DOOR (not the reshape) is what refuses:
/// the geometry law reads the op's numels, which are frame-invariant (1024 in, 32 out,
/// whatever frame the walk threaded — the wall-11 law).
#[test]
fn a_sub_stick_survivor_refuses_by_name() {
    let k = leading_axis_case((256, 4), &[32, 8, 4], &[0], (8, 4), OpKind::ArithAddf);
    let e = match lowered(&k, 1024) {
        Err(e) => e,
        Ok(ops) => panic!(
            "a sub-stick survivor vector must refuse, not lower {} op(s)",
            ops.len()
        ),
    };
    assert!(
        e.contains("SUB-STICK"),
        "the refusal names the sub-stick law, got: {e}"
    );
    assert!(
        e.contains("32 survivor lanes"),
        "the refusal names the survivor width it refused, got: {e}"
    );
    assert!(
        e.contains("[8, 4]"),
        "the refusal names the stated shape it refused, got: {e}"
    );
}

/// ⭐ THE BATCH BOUNDARY (wall 13's other side): the SAME merge shape family one batch wider —
/// `[16, 16, 4]` reduced over dims `[0]` to `[16, 4]`, 64 survivor lanes = exactly ONE whole
/// stick — LOWERS through the wall-12 arm. This is the B=2 decode merge (survivors
/// `B·NKVH·GQA = 64`; J = `32/B` = 16), the smallest survivor set the arm accepts, pinned beside
/// the 32-lane B=1 refusal above so the boundary is a test failure on either side of it: the
/// sub-stick refusal is wall 13's, and everything from one stick up was wall 12's to lower.
#[test]
fn a_one_stick_survivor_set_lowers_at_the_batch_boundary() {
    let k = leading_axis_case((256, 4), &[16, 16, 4], &[0], (16, 4), OpKind::ArithAddf);
    let ops = lowered(&k, 1024).unwrap_or_else(|e| {
        panic!("the one-stick survivor merge (B=2's m/l shape) must lower, refused: {e}")
    });
    let [red] = &ops[..] else {
        panic!("the slice is ONE op, got {} ops", ops.len())
    };
    // The geometry the door derived from the op's own numels: 1024 in, 64 out, reduced 16 —
    // the flat frame is the SAME `[256, 4]` the refused B=1 case threads, so only the numels
    // moved, which is the wall-11 frame-invariance law this boundary pins from the lowering side.
    assert_eq!(dsc(red).N_.out_, 16, "the reduced extent (J = 32/B at B=2)");
    assert_eq!(dsc(red).N_.y_, 64, "the survivor lanes (one whole stick)");
    let j = op_json(red);
    let c = compute_op(&j);
    assert_eq!(
        c["opFuncName"], "sumnonstick",
        "the one-stick survivor set takes the nonstick form, not a new one"
    );
    let body = j["dscs_"][0].as_object().unwrap().values().next().unwrap();
    let masking = serde_json::to_string(&body["coordinateMasking_"]).unwrap_or_default();
    assert!(
        masking == "null" || masking == "{}",
        "an exact one-stick product has no pad lane (got {masking})"
    );
}

/// ⛔ NO-DRIFT (refusal): a MIDDLE-axis reduce (`[32, 32, 64]` reducing dims `[1]`, a wide
/// output) keeps the row-reduce body's own refusal VERBATIM — the leading-prefix arm did not
/// move a byte of the path every other block takes.
#[test]
fn a_middle_axis_reduce_keeps_the_row_reduce_refusal() {
    let k = leading_axis_case((1024, 64), &[32, 32, 64], &[1], (32, 64), OpKind::ArithAddf);
    let e = match lowered(&k, 65536) {
        Err(e) => e,
        Ok(ops) => panic!(
            "a middle-axis reduce with a wide output must keep the row-reduce refusal, not \
             lower {} op(s)",
            ops.len()
        ),
    };
    assert!(
        e.contains("ONE value per row"),
        "the row-reduce body's own refusal, verbatim, got: {e}"
    );
}

/// ⛔ NO-DRIFT (lowering): a DEGENERATE middle reduce (`[8, 1, 64]` reducing dims `[1]` to
/// `[8, 1]` — the op removes nothing along a middle axis) still LOWERS through the row-reduce
/// path as the STICK form — `sum`, never `sumnonstick`. Hard-refusing Middle outright would
/// have regressed exactly this shape; the fall-through is the no-drift law made testable.
#[test]
fn a_degenerate_middle_reduce_still_lowers_through_the_row_reduce_path() {
    let k = leading_axis_case((8, 64), &[8, 1, 64], &[1], (8, 1), OpKind::ArithAddf);
    let ops = lowered(&k, 512)
        .unwrap_or_else(|e| panic!("the degenerate middle reduce still lowers, refused: {e}"));
    let [red] = &ops[..] else {
        panic!("the slice is ONE op, got {} ops", ops.len())
    };
    let j = op_json(red);
    let c = compute_op(&j);
    assert_eq!(
        c["opFuncName"], "sum",
        "the row-reduce path emits the STICK form — the no-drift law"
    );
}
