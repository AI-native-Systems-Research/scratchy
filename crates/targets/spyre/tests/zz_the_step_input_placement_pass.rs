// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE STEP-INPUT PLACEMENT PASS — pinned as the GENERAL law it is.
//!
//! [`place_step_inputs`] is the producer half of the host-staged per-step inputs: a spliced
//! program can bind a parameter no graph tensor names (an index table, a validity mask), and
//! the pass places exactly the bindings the program STATES — minting a seg-ACTIVATION
//! placement sized from the program's OWN `construct_memory_view` extent and dtype, checking
//! every stated tid against the one numbering law
//! [`step_input_tid`](ktir_superdsc::ktir_node::step_input_tid)`(n_graph_tensors, slot)`.
//!
//! ## What this roster pins
//!
//! **THE HONESTY CHECK** (`a_non_attention_program_with_step_inputs_places_them`): the pass's
//! design constraint is that it has ZERO kernel vocabulary — it asks WHICH bindings are
//! host-staged (the producer stated it), WHAT the kernel's own view says, and WHICH slot each
//! took. The proof is a NON-ATTENTION program (an elementwise mul with an unstored add, the
//! `minting_program` miniature's own shape) stating a per-step input: it must flow through
//! the pass and the whole-function door with zero ladder changes. A second kernel family
//! stating per-step inputs is a row here, not a new arm in the pass.
//!
//! **THE REFUSALS**: a hand-picked tid (one that disagrees with the numbering law) is refused
//! by name — one collision away from silently aliasing a graph tensor's placement; two
//! programs disagreeing about one buffer's extent are refused — they would share bytes while
//! disagreeing about how many.
//!
//! **THE PRESERVED DEFAULT**: a binding past the graph that no program STATES is still
//! refused by the door, by name. The pass places only what is stated; it never opens the
//! door for a binding nobody promised to fill.

use ktir_core::arena::Arena;
use ktir_core::attrkey::AttrKey;
use ktir_core::dtypes::DType;
use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
use ktir_core::irtype::IrType;
use ktir_core::opkind::OpKind;
use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::emit::whole_function::lower_function;
use ktir_superdsc::ktir_node::{BufferId, Elementwise, KtirNode, Program, StepInput};
use ktir_superdsc::place::{PlaceId, act_name};
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::place_step_inputs;

/// A NON-ATTENTION program stating one host-staged step input — the `minting_program`
/// miniature's own shape (two loads, an UNSTORED `arith.addf` the terminal consumes, one
/// stored `arith.mulf`), with a third parameter bound past the graph: `step`, a buffer the
/// runtime stages every step. `slot: Some(_)` states it; `None` leaves it unstated (the
/// preserved-default row below).
///
/// The step parameter's view carries its OWN extent and dtype — the two facts the pass sizes
/// the placement from — so each test states them independently of the graph operands.
#[allow(clippy::too_many_arguments)]
fn step_program(
    name: &'static str,
    a: u32,
    out: u32,
    step: u32,
    slot: Option<u32>,
    step_dtype: DType,
    step_rows: i64,
    step_cols: i64,
) -> KtirNode {
    let arena: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64, d: DType| {
        op.with_attr(arena, AttrKey::Shape, Attr::IntList(arena.ints(vec![r, c])))
            .with_attr(arena, AttrKey::Dtype, Attr::Dtype(d))
    };
    let ty = |r: i64, c: i64| {
        Some(IrType::Tensor {
            dims: arena.ints(vec![r, c]),
            elem: DType::F16,
        })
    };
    // Parameters: a, step, out — the operand prefix, output LAST (the splice's own
    // `default_bindings` law).
    let (p_a, p_step, p_out) = (Ssa(0), Ssa(1), Ssa(2));
    let zero = Ssa(3);
    let (v_a, t_a, val_a) = (Ssa(4), Ssa(5), Ssa(6));
    let (v_s, t_s, val_s) = (Ssa(7), Ssa(8), Ssa(9));
    let (sum, v_o, t_o, prod) = (Ssa(10), Ssa(11), Ssa(12), Ssa(13));
    let ops = vec![
        Operation::new(arena, Some(zero), OpKind::ArithConstant, &[]).with_attr(
            arena,
            AttrKey::Value,
            Attr::Int(0),
        ),
        shape(
            Operation::new(arena, Some(v_a), OpKind::KtdpConstructMemoryView, &[p_a]),
            8,
            64,
            DType::F16,
        ),
        shape(
            Operation::new(
                arena,
                Some(t_a),
                OpKind::KtdpConstructAccessTile,
                &[v_a, zero, zero],
            ),
            8,
            64,
            DType::F16,
        ),
        shape(
            Operation::new(arena, Some(val_a), OpKind::KtdpLoad, &[t_a]),
            8,
            64,
            DType::F16,
        ),
        shape(
            Operation::new(arena, Some(v_s), OpKind::KtdpConstructMemoryView, &[p_step]),
            step_rows,
            step_cols,
            step_dtype,
        ),
        shape(
            Operation::new(
                arena,
                Some(t_s),
                OpKind::KtdpConstructAccessTile,
                &[v_s, zero, zero],
            ),
            step_rows,
            step_cols,
            step_dtype,
        ),
        shape(
            Operation::new(arena, Some(val_s), OpKind::KtdpLoad, &[t_s]),
            step_rows,
            step_cols,
            step_dtype,
        ),
        // ⭐ THE INTERMEDIATE: `a + step` is consumed by the mul below and stored NOWHERE —
        // the mint shape the whole-function walk really reads.
        {
            let mut op = shape(
                Operation::new(arena, Some(sum), OpKind::ArithAddf, &[val_a, val_s]),
                8,
                64,
                DType::F16,
            );
            op.result_type = ty(8, 64);
            op
        },
        shape(
            Operation::new(arena, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
            8,
            64,
            DType::F16,
        ),
        shape(
            Operation::new(
                arena,
                Some(t_o),
                OpKind::KtdpConstructAccessTile,
                &[v_o, zero, zero],
            ),
            8,
            64,
            DType::F16,
        ),
        shape(
            Operation::new(arena, Some(prod), OpKind::ArithMulf, &[sum, val_s]),
            8,
            64,
            DType::F16,
        ),
        Operation::new(arena, None, OpKind::KtdpStore, &[prod, t_o]),
    ];
    KtirNode {
        func: IRFunction {
            name,
            arguments: arena.args(vec![
                (p_a, IrType::Index),
                (p_step, IrType::Index),
                (p_out, IrType::Index),
            ]),
            operations: arena.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        },
        program: Program::Elementwise(Elementwise::Mul),
        bindings: vec![BufferId::new(a), BufferId::new(step), BufferId::new(out)],
        mask: None,
        step_inputs: match slot {
            Some(slot) => vec![StepInput {
                buffer: BufferId::new(step),
                slot,
            }],
            None => Vec::new(),
        },
        node_out_tid: None,
        minted: Default::default(),
    }
}

/// The bundle the graph walk would have built: `n` graph tensors `t0..t(n-1)`, each placed in
/// the Activation segment with its identity row — the same two facts `compute_bundle_layout`
/// states for every tensor it places.
fn graph_layout(n: u32) -> BundleLayout {
    let mut l = BundleLayout::default();
    for tid in 0..n {
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: SegRole::Activation.segment(),
                bank: 0,
                offset: u64::from(tid) * 0x10000,
                size: 8 * 64 * 2,
                cover: ktir_superdsc::placement::BindCover::Full,
            },
        );
    }
    l.segment_bytes[SegRole::Activation.segment()] = u64::from(n) * 0x10000;
    l
}

/// One op list carrying `node`, in the pass's own input shape.
fn carrying(node: KtirNode) -> Vec<EmittedOp> {
    let mut e = EmittedOp::bare(node.func.name.to_string());
    e.ktir = Some(node);
    vec![e]
}

/// ⭐⭐⭐ THE HONESTY CHECK — a NON-ATTENTION program stating per-step inputs flows through the
/// pass and the door with zero ladder changes.
///
/// The program is an elementwise mul (the `minting_program` miniature's shape) with a third
/// parameter bound past the graph at the numbering law's slot 0. The door REFUSES it before
/// the pass — by name, exactly the t1412 refusal that blocked the batched decode build — and
/// accepts it after: the pass placed the stated binding from the program's own view, recorded
/// the discovery row (tid → slot), and wrote the identity row the door's `resolve_seg_base`
/// resolves through. Nothing in the pass asked what the input IS.
#[test]
fn a_non_attention_program_with_step_inputs_places_them() {
    // Six graph tensors: the operands bind t0 and t5; slot 0 of a 6-tensor graph is t6.
    let n = 6u32;
    let node = step_program("mul_s0", 0, 5, 6, Some(0), DType::F16, 8, 64);
    let mut layout = graph_layout(n);

    // BEFORE: the door refuses the unplaced binding by name — the pass has not run, and the
    // walk declares nothing itself.
    let mut syms = 0i64;
    let refused = match lower_function(&node, Some(&layout), &mut syms, None) {
        Ok(_) => panic!("the door refuses a binding no step is promised to fill"),
        Err(e) => e,
    };
    assert!(
        refused.message.contains("t6") && refused.message.contains("has no placement"),
        "the pre-pass refusal names the binding and its defect: {}",
        refused.message
    );

    // THE PASS: one op list, the graph's own tensor count.
    let ops = carrying(node.clone());
    place_step_inputs(&mut layout, &[&ops], n)
        .expect("a stated step input places through the general law");

    // The placement: Activation segment, sized from the program's OWN view (8×64 fp16 =
    // 1024 B), packed at the segment's aligned end — never a re-derivation.
    let p = layout
        .placements
        .get(&6)
        .unwrap_or_else(|| panic!("the stated step input t6 was placed"));
    assert_eq!(p.role, SegRole::Activation);
    assert_eq!(p.segment, SegRole::Activation.segment());
    assert_eq!(
        p.size,
        8 * 64 * 2,
        "the extent and dtype are the view's own"
    );
    assert_eq!(
        p.offset,
        u64::from(n) * 0x10000,
        "packed at the segment end"
    );
    // The discovery row: tid → (slot, element format) — the slot the runtime addresses the input
    // by, and the format the bake maps to the staging channel. Both are the program's own
    // statements; neither says what the input IS.
    assert_eq!(
        layout.step_inputs.get(&6),
        Some(&ktir_superdsc::placement::StepInputRow {
            slot: 0,
            df: ktir_superdsc::superdsc_opspec::Df::Fp16,
        }),
        "the discovery table records the slot and the view's own element format"
    );
    // The identity row: without it `resolve_seg_base` treats the parameter as an undeclared
    // synthetic.
    assert_eq!(
        layout.ids.borrow().get(&act_name(6)),
        Some(&PlaceId::Act(6)),
        "the placement carries its identity row"
    );

    // AFTER: the door lowers the same program over the same layout — the mint above every
    // placement, the stated binding resolved through its placement.
    let mut syms = 0i64;
    lower_function(&node, Some(&layout), &mut syms, None)
        .unwrap_or_else(|e| panic!("the door accepts the placed program: {}", e.message));

    // IDEMPOTENT: every layer of a graph states the same input; the second sighting agrees.
    place_step_inputs(&mut layout, &[&ops], n)
        .expect("the second layer's identical statement agrees");
    assert_eq!(
        layout.placements.len(),
        (n as usize) + 1,
        "the second sighting placed nothing new"
    );
}

/// ⭐ THE SIZING LAW IS THE PROGRAM'S OWN — an i32 view of 32 entries places at
/// SenUint32's word length (4 B), 128 B total. The element width comes from the view's own
/// dtype through `df_of_dtype`, never from a re-derivation keyed on what the input is.
#[test]
fn the_placement_sizes_from_the_views_own_dtype() {
    let n = 2u32;
    let node = step_program("mul_s1", 0, 1, 2, Some(0), DType::I32, 1, 32);
    let mut layout = graph_layout(n);
    let ops = carrying(node);
    place_step_inputs(&mut layout, &[&ops], n)
        .expect("an i32 step input places through the same law");
    let p = layout
        .placements
        .get(&2)
        .unwrap_or_else(|| panic!("the stated step input t2 was placed"));
    assert_eq!(
        p.size, 128,
        "1×32 i32 at SenUint32's 4 B word is 128 B — an fp16 default here would under-reserve \
         the buffer the program reads"
    );
    // And the discovery row carries the SAME element format — the bake maps it to the raw-u32
    // staging channel, so a row that said fp16 would stage an index table through the f16
    // narrowing and corrupt every entry.
    assert_eq!(
        layout.step_inputs.get(&2),
        Some(&ktir_superdsc::placement::StepInputRow {
            slot: 0,
            df: ktir_superdsc::superdsc_opspec::Df::SenUint32,
        }),
        "the discovery row states the view's own element format"
    );
}

/// ⛔ TWO PROGRAMS DISAGREEING ABOUT ONE BUFFER'S ELEMENT FORMAT ARE REFUSED — same slot, same
/// extent, different dtype. The two would share one buffer while the bake maps them to
/// DIFFERENT staging channels, so the f16 narrowing would rewrite an index table's entries.
/// Same refusal family as the extent disagreement: stated twice, both statements must agree.
#[test]
fn two_programs_disagreeing_about_one_buffers_format_are_refused() {
    let n = 6u32;
    // Both state slot 0 (t6) at the SAME byte extent (8×64 fp16 = 8×32 i32 = 1024 B); only the
    // element format differs.
    let first = step_program("mul_s6", 0, 5, 6, Some(0), DType::F16, 8, 64);
    let second = step_program("mul_s7", 1, 4, 6, Some(0), DType::I32, 8, 32);
    let ops = [carrying(first), carrying(second)];
    let lists: Vec<&[EmittedOp]> = ops.iter().map(|v| v.as_slice()).collect();
    let mut layout = graph_layout(n);
    let err = place_step_inputs(&mut layout, &lists, n)
        .expect_err("two element formats for one buffer is a refusal, not a repair");
    assert!(
        err.0.contains("two element formats"),
        "the refusal names the disagreement: {}",
        err.0
    );
}

/// ⛔ A HAND-PICKED TID IS REFUSED BY NAME. The numbering law states slot 0 of a 6-tensor
/// graph as t6; a producer that states slot 0 at t8 (or anything else) is one collision away
/// from silently aliasing a graph tensor's placement, and the pass refuses rather than
/// repairing — the `gather_entry_ceiling` shape.
#[test]
fn a_hand_picked_tid_is_refused_by_the_numbering_law() {
    let n = 6u32;
    // Slot 0 stated at t8: the law says t6.
    let node = step_program("mul_s2", 0, 5, 8, Some(0), DType::F16, 8, 64);
    let mut layout = graph_layout(n);
    let ops = carrying(node);
    let err = place_step_inputs(&mut layout, &[&ops], n)
        .expect_err("a binding numbered off the law is refused, never repaired");
    assert!(
        err.0.contains("numbering law") && err.0.contains("t8"),
        "the refusal names the binding and the law: {}",
        err.0
    );
}

/// ⛔ TWO PROGRAMS DISAGREEING ABOUT ONE BUFFER'S EXTENT ARE REFUSED. Every layer of a graph
/// states the same inputs, so the second sighting must AGREE — same segment, same role, same
/// size. A second program stating a different view for one tid would share bytes with the
/// first while disagreeing about how many there are.
#[test]
fn two_programs_disagreeing_about_one_buffer_are_refused() {
    let n = 6u32;
    // Both state slot 0 (t6); the second states half the extent.
    let first = step_program("mul_s3", 0, 5, 6, Some(0), DType::F16, 8, 64);
    let second = step_program("mul_s4", 1, 4, 6, Some(0), DType::F16, 4, 64);
    let ops = [carrying(first), carrying(second)];
    let lists: Vec<&[EmittedOp]> = ops.iter().map(|v| v.as_slice()).collect();
    let mut layout = graph_layout(n);
    let err = place_step_inputs(&mut layout, &lists, n)
        .expect_err("two extents for one buffer is a refusal, not a repair");
    assert!(
        err.0.contains("disagree about one buffer's extent"),
        "the refusal names the disagreement: {}",
        err.0
    );
}

/// ⛔ THE PRESERVED DEFAULT — a binding past the graph that NO program states is still
/// refused by the door, by name. The pass places only what is stated; it is not a door
/// opener for parameters nobody promised to fill (the `09603fd88` behavior, intact).
#[test]
fn an_unstated_binding_is_still_refused_by_the_door() {
    let n = 6u32;
    // The step parameter binds t6 with a view, but the node states NO step input.
    let node = step_program("mul_s5", 0, 5, 6, None, DType::F16, 8, 64);
    let mut layout = graph_layout(n);
    let ops = carrying(node.clone());
    // The pass is a no-op: nothing is stated, nothing is placed.
    place_step_inputs(&mut layout, &[&ops], n)
        .expect("an unstated graph places nothing and refuses nothing");
    assert!(
        !layout.placements.contains_key(&6) && layout.step_inputs.is_empty(),
        "the pass placed nothing it was not told to place"
    );
    // The door still refuses, by name.
    let mut syms = 0i64;
    let refused = match lower_function(&node, Some(&layout), &mut syms, None) {
        Ok(_) => panic!("an unstated binding is still nobody's promise to fill"),
        Err(e) => e,
    };
    assert!(
        refused.message.contains("t6") && refused.message.contains("has no placement"),
        "the refusal names the binding and its defect: {}",
        refused.message
    );
}
