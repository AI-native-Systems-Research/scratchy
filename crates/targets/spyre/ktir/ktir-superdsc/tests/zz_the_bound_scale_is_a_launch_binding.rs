// SPDX-License-Identifier: Apache-2.0

//! ⭐⭐⭐⭐⭐ RUNG 3 — THE BOUND-SCALE SCALARMUL, THE DOOR NO TEST EVER OPENED.
//!
//! Issue 201 item 9's own list: `bound_scale_of`, `BoundScalarMul` and `scalarmul_bound` had
//! no tests in this repo. The mechanism is attention's `ATTN_SCALE` with the multiplier BOUND
//! per launch instead of baked: a `tensor.splat` over a `ktdp.load` of a `[1,1]` view of a
//! function ARGUMENT, multiplied into a tensor by an `arith.mulf`.
//!
//! What these tests pin:
//!
//! * THE DISCRIMINATOR (`bound_scale_of`): the splat must read a BOTH-AXES-DEGENERATE `[1,1]`
//!   view of a PARAMETER. A `[1, cols]` row view — the mb-broadcast defect shape — must fall
//!   through to `None` (fail closed), because emitting a scalarmul for it would address one
//!   value where the program means a row.
//! * THE EMISSION (`scalarmul_bound` through `lower_function`): the descriptor names the
//!   SCALE OPERAND by the parameter's own binding tid, so the value the card multiplies by is
//!   whatever the launch bound at `const:t<tid>` — read at RUN TIME, not baked. The control is
//!   `scalarmul_at`: a baked scale resolves its operand through the REGISTRY SLOT's reserved
//!   tid instead, which is a different number by construction (the reserved block is far above
//!   any parameter tid), so the two arms cannot pass each other's assertions.
//!
//! Every program is built from the op kinds the walk really reads, so the door sees a real
//! program and not a stub.

use ktir_core::arena::Arena;
use ktir_core::attrkey::AttrKey;
use ktir_core::dtypes::DType;
use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
use ktir_core::irtype::IrType;
use ktir_core::opkind::OpKind;
use ktir_superdsc::emit::whole_function::{bound_scale_of, lower_function};
use ktir_superdsc::ktir_node::{BufferId, KtirNode, Program};
use ktir_superdsc::place::act_name;
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};

/// The shape of the scale parameter's view — the one variable of the discriminator tests.
#[derive(Clone, Copy, PartialEq)]
enum ScaleView {
    /// The rung-3 chain: a `[1,1]` fp16 view of an argument.
    Degenerate,
    /// A `[1, 64]` ROW view of the same argument — the mb-broadcast defect shape, which must
    /// NOT be read as a bound scalar.
    Row,
}

/// One `y = x * scale_arg` program with the scale's view shape as the only variable:
///
/// * `ScaleView::Degenerate` — the chain `to_ktir`'s `expand_splat_of_scalar_argument` emits;
/// * `ScaleView::Row` — the same chain with the view widened to `[1, 64]`.
fn scale_program(view: ScaleView) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };
    let (rows, cols) = match view {
        ScaleView::Degenerate => (1, 1),
        ScaleView::Row => (1, 64),
    };

    // Parameters: %0 = x, %1 = scale, %2 = out.
    let (p_x, p_s, p_out) = (Ssa(0), Ssa(1), Ssa(2));
    let zero = Ssa(3);
    // x's chain.
    let (v_x, t_x, val_x) = (Ssa(4), Ssa(5), Ssa(6));
    // The scale's chain: view, [1,1] access tile, load, splat.
    let (v_s, t_s, val_s, splat) = (Ssa(7), Ssa(8), Ssa(9), Ssa(10));
    // The mulf and the store.
    let (mul, v_o, t_o) = (Ssa(11), Ssa(12), Ssa(13));

    let ops = vec![
        Operation::new(a, Some(zero), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(0),
        ),
        // x: [8, 64] — the tensor the scale multiplies.
        shape(
            Operation::new(a, Some(v_x), OpKind::KtdpConstructMemoryView, &[p_x]),
            8,
            64,
        ),
        shape(
            Operation::new(
                a,
                Some(t_x),
                OpKind::KtdpConstructAccessTile,
                &[v_x, zero, zero],
            ),
            8,
            64,
        ),
        shape(
            Operation::new(a, Some(val_x), OpKind::KtdpLoad, &[t_x]),
            8,
            64,
        ),
        // The scale's chain — the ONE variable is this view's shape.
        shape(
            Operation::new(a, Some(v_s), OpKind::KtdpConstructMemoryView, &[p_s]),
            rows,
            cols,
        ),
        shape(
            Operation::new(
                a,
                Some(t_s),
                OpKind::KtdpConstructAccessTile,
                &[v_s, zero, zero],
            ),
            rows,
            cols,
        ),
        shape(
            Operation::new(a, Some(val_s), OpKind::KtdpLoad, &[t_s]),
            rows,
            cols,
        ),
        shape(
            Operation::new(a, Some(splat), OpKind::TensorSplat, &[val_s]),
            8,
            64,
        ),
        // y = x · scale.
        shape(
            Operation::new(a, Some(mul), OpKind::ArithMulf, &[val_x, splat]),
            8,
            64,
        ),
        // out: [8, 64].
        shape(
            Operation::new(a, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
            8,
            64,
        ),
        shape(
            Operation::new(
                a,
                Some(t_o),
                OpKind::KtdpConstructAccessTile,
                &[v_o, zero, zero],
            ),
            8,
            64,
        ),
        Operation::new(a, None, OpKind::KtdpStore, &[mul, t_o]),
    ];

    KtirNode {
        func: IRFunction {
            name: "bound_scale",
            arguments: a.args(vec![
                (p_x, IrType::Index),
                (p_s, IrType::Index),
                (p_out, IrType::Index),
            ]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        },
        program: Program::ScalarMul,
        bindings: vec![BufferId::new(301), BufferId::new(302), BufferId::new(303)],
        mask: None,
        node_out_tid: None,
    }
}

/// The layout every test here places its parameters with: distinct segments per parameter, the
/// scale at its own segment so a scale-operand assertion discriminates WHICH parameter an op
/// reads, not a coincidence of two equal offsets.
fn layout_for(k: &KtirNode) -> BundleLayout {
    let mut l = BundleLayout::default();
    for (i, b) in k.bindings.iter().enumerate() {
        let tid = b.get();
        l.ids
            .borrow_mut()
            .insert(act_name(tid), ktir_superdsc::place::PlaceId::Act(tid));
        let bytes: u64 = match i {
            // x [8,64] f16, scale [1,1] f16, out [8,64] f16.
            0 | 2 => 8 * 64 * 2,
            _ => 2,
        };
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: 3,
                bank: 0,
                offset: (i as u64) * 0x10000,
                size: bytes,
            },
        );
    }
    l
}

/// The `dscs_[0]` entry of an emitted op — the descriptor the assertions read, asserted on the
/// TYPED structs so the assertions name the fields the card acts on with no second parser to
/// disagree with the first.
fn dsc(op: &ktir_superdsc::emit::EmittedOp) -> &ktir_superdsc::wire::Dsc {
    op.dsc()
        .dscs_
        .first()
        .and_then(|m| m.values().next())
        .expect("the scalarmul has a descriptor")
}

/// Every declared operand's device address, in `labeledDs_` order. The scalarmul's operand
/// list is `[x, scale, out]` (positional `Tensor0/1/2`); the ADDRESS each resolves to is the
/// fact the card acts on: the descriptor names operands only positionally, so which address
/// appears is which buffer the op reads, and under `layout_for` the three parameters sit at
/// three DISTINCT placements.
/// placements — so which address appears is which buffer the op reads.
fn operand_addresses(d: &ktir_superdsc::wire::Dsc) -> Vec<u64> {
    d.labeledDs_
        .iter()
        .filter_map(|l| {
            d.scheduleTree_
                .iter()
                .find(|n| n.nodeType_ == "allocate" && n.ldsIdx_ == l.ldsIdx_)
                .and_then(|n| {
                    n.startAddressCoreCorelet_
                        .data_
                        .get("[0, 0, 0]")
                        .and_then(|v| v.parse().ok())
                })
        })
        .collect()
}

/// The scale parameter's device address under `layout_for` (parameter 1, at seg3 + 0x10000).
const SCALE_ADDR: u64 = 0xC_0000_0000 + 0x10000;

/// ⭐⭐⭐ THE DISCRIMINATOR: a splat over a `[1,1]` load of a PARAMETER is a bound scale, naming
/// the parameter's own binding. The positive — the whole point of rung 3.
#[test]
fn a_degenerate_splat_of_a_loaded_parameter_is_a_bound_scale() {
    let k = scale_program(ScaleView::Degenerate);
    let mulf = k
        .func
        .operations
        .iter()
        .find(|o| o.op_type == OpKind::ArithMulf)
        .expect("the program states its mulf");
    let b = bound_scale_of(&k, mulf)
        .expect("the discriminator decides")
        .expect("a [1,1] splat over a parameter load IS a bound scale");
    assert_eq!(
        b.tid, 302,
        "the bound scale names the scale PARAMETER's own binding tid"
    );
    assert_eq!(
        b.tensor,
        Ssa(6),
        "the tensor side is the mulf's other operand — x's load"
    );
}

/// ⛔ CONTROL — THE BOTH-AXES-DEGENERATE TEST: a `[1, 64]` ROW view is the mb-broadcast defect
/// shape. Reading it as a bound scalar would emit a scalarmul addressing ONE value where the
/// program means a row of 64 — the silent wrong answer the shape guard exists to refuse. The
/// discriminator must answer `None`, and the walk must then refuse the program by name rather
/// than lower it as the nearest thing.
#[test]
fn a_row_view_is_not_a_bound_scale_and_the_walk_refuses_it() {
    let k = scale_program(ScaleView::Row);
    let mulf = k
        .func
        .operations
        .iter()
        .find(|o| o.op_type == OpKind::ArithMulf)
        .expect("the program states its mulf");
    let b = bound_scale_of(&k, mulf).expect("the discriminator decides");
    assert!(
        b.is_none(),
        "a [1,64] row view is the mb-broadcast defect shape — emitting a scalarmul for it would          address one value where the program means a row"
    );
    // And the whole-function walk, with no other reading available, refuses by name.
    let layout = layout_for(&k);
    let mut sid = 0i64;
    let e = match lower_function(&k, Some(&layout), &mut sid) {
        Err(e) => e.message,
        Ok(_) => panic!("a row-vector scale must be refused, not lowered as a scalarmul"),
    };
    assert!(
        e.contains("is neither a load of a parameter nor an intermediate"),
        "the refusal names the unresolvable splat operand, got: {e}"
    );
}

/// ⭐⭐⭐ THE EMISSION: `scalarmul_bound` names the SCALE OPERAND by the parameter's binding
/// tid, so the card reads whatever the launch bound at `const:t302` — RUN TIME, not bake time.
/// Proven by ADDRESS: the scale operand's `labeledDs_` entry resolves to the scale parameter's
/// own placement, and the multiply is a `multiply` opFunc over the tensor and that operand.
#[test]
fn the_bound_scale_operand_is_the_parameters_own_placement() {
    let k = scale_program(ScaleView::Degenerate);
    let layout = layout_for(&k);
    let mut sid = 0i64;
    let ops = lower_function(&k, Some(&layout), &mut sid)
        .expect("the bound-scale program lowers through the whole-function door");
    let [op] = &ops[..] else {
        panic!(
            "one mulf is one scalarmul, got {:?}",
            ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
        );
    };
    let d = dsc(op);
    // The op is a `multiply`.
    assert!(
        d.computeOp_.iter().any(|c| c.opFuncName == "mul"),
        "a scalarmul is a `mul` opFunc (`multiply` is the OpFunc KEY, `mul` its dxp spelling): {}",
        op.op_name
    );
    // ⭐ THE SCALE OPERAND IS THE PARAMETER'S OWN ADDRESS. A descriptor that baked the value
    // instead would carry NO operand at the scale parameter's placement at all — the registry
    // slot's tid is a different number, and a baked `const:` ride would need a different
    // placement.
    let inputs = operand_addresses(d);
    assert!(
        inputs.contains(&SCALE_ADDR),
        "the scale operand reads the scale parameter at {SCALE_ADDR:#x} — got operand addresses \
         {inputs:#x?}, which is a BAKED multiply with no bound operand: {}",
        op.op_name
    );
    assert!(
        inputs.contains(&(0xC_0000_0000)),
        "the tensor operand reads x at its own placement, got {inputs:#x?}"
    );
}

/// ⛔ CONTROL — THE BAKED ARM NAMES A DIFFERENT OPERAND: `scalarmul_at`'s scale resolves
/// through the REGISTRY SLOT's reserved tid (`SCALARMUL_SCALE_BASE - idx`, far above any
/// parameter tid), placed by the reserved-scale machinery and NOT at the scale parameter's
/// placement. The two arms therefore emit descriptors whose scale operands sit at different
/// addresses — the difference that makes the bound test above non-vacuous. Asserted on the
/// SAME shape through the one API that exercises `scalarmul_at` with a registry: a splat of a
/// CONSTANT, whose scale operand must NOT be at the scale parameter's address (it is at the
/// registry slot's, or the program refuses for want of a slot — both are "not at the
/// parameter's", and the positive control below pins which).
#[test]
fn a_baked_scale_does_not_read_the_scale_parameter() {
    let k = scale_program(ScaleView::Degenerate);
    let layout = layout_for(&k);
    let mut sid = 0i64;
    // The SAME program, but with the splat's operand replaced by a float CONSTANT and the
    // registry holding that value: the `scalarmul_at` arm.
    let a: &'static Arena = Arena::global();
    let ops = {
        // Rebuild the program with a constant-backed splat: reuse the builder by hand.
        let k = {
            let mut k = scale_program(ScaleView::Degenerate);
            // The scale chain stays (the parameter is still bound and placed); only the SPLAT's
            // operand changes to a fresh float constant.
            let const_ssa = Ssa(20);
            let mut ops: Vec<Operation<'static>> = Vec::new();
            for o in k.func.operations.iter() {
                if o.op_type == OpKind::TensorSplat {
                    ops.push(
                        Operation::new(
                            a,
                            Some(o.result.unwrap()),
                            OpKind::TensorSplat,
                            &[const_ssa],
                        )
                        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![8, 64])))
                        .with_attr(
                            a,
                            AttrKey::Dtype,
                            Attr::Dtype(DType::F16),
                        ),
                    );
                } else {
                    ops.push(*o);
                }
            }
            ops.insert(
                0,
                Operation::new(a, Some(const_ssa), OpKind::ArithConstant, &[]).with_attr(
                    a,
                    AttrKey::Value,
                    Attr::Float(0.125),
                ),
            );
            k.func.operations = a.ops(ops);
            k
        };
        let mut layout = layout;
        // THE REGISTRY: one slot holding the baked value, exactly as `compute_bundle_layout`
        // collects it, AND the slot's reserved-tid placement, exactly as the real bundle
        // builder declares it (`lower_subtile_tape_to_superdsc`: a 2-B Activation at the top of
        // its own segment) — without either, `scalarmul_at` refuses, which is itself the
        // fail-closed behavior, but the emission is what this control compares.
        layout.scalarmul_scales = vec![0.125];
        let slot_tid = ktir_superdsc::reserved_tids::scalarmul_scale_tid(0);
        layout.ids.borrow_mut().insert(
            act_name(slot_tid),
            ktir_superdsc::place::PlaceId::Act(slot_tid),
        );
        layout.placements.insert(
            slot_tid,
            TensorPlacement {
                tid: slot_tid,
                role: SegRole::Activation,
                segment: 2,
                bank: 0,
                offset: 0,
                size: 2,
            },
        );
        lower_function(&k, Some(&layout), &mut sid)
            .expect("the baked-scale program lowers through the registry arm")
    };
    let [op] = &ops[..] else {
        panic!("one mulf is one scalarmul");
    };
    let d = dsc(op);
    let inputs = operand_addresses(d);
    assert!(
        !inputs.contains(&SCALE_ADDR),
        "the BAKED arm's scale operand is the registry slot's placement, NOT the scale \
         parameter's {SCALE_ADDR:#x} — got {inputs:#x?}, which means the bound arm was taken"
    );
}
