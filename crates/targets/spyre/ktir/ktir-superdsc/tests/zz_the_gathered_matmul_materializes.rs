// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! ⭐⭐⭐⭐⭐ RUNG 4 — THE GATHERED MATMUL'S **B** MATERIALIZES, AND THESE ARE ITS PROOFS.
//!
//! The whole-function door's gather guard refuses a gathered operand on anything but a
//! `ScalarMul` — fail-closed, because every other assembler would read the table at its base and
//! ignore the ids: the right shape and the wrong rows, from a clean bake. RUNG 4's exception is a
//! `linalg.matmul` whose B is the gathered table, and the materialization is
//! `whole_function::gathered_matmul_materializes`: a KERNEL-less gathered `identity` copy into a
//! minted intermediate, the matmul reading the intermediate as plain-B `[k, n]`.
//!
//! # WHY THAT SHAPE AND NOT AN INDEX ON THE MATMUL ITSELF
//!
//! A matmul operand goes to `Stk::<KernelTag>::kernel(k, n)`, and a KERNEL-stick operand makes
//! the dxp reuse explorer run (`L3DlOpsScheduler.cpp:1550` gates it on `isReuse`;
//! `hasDimensionReuse` `:303-321` is `primaryDsInfo_.size() > 1 && count(KERNEL)`), which
//! demands an LX allocate node for every HBM-pinned labeledDs (`calculateFlopPerByte`
//! `:2334`/`:2337`) — a build-time `Err` from this crate's own gather-on-KERNEL guard. The
//! vendor's own gathered fixture (`dxp/test/test_gather_1core`) and IBM's paged attention both
//! spell the way around it: the gather rides a KERNEL-less elementwise copy (`identity` /
//! `AddZero`), and the matmul reads the copy's result.
//!
//! # WHAT THESE TESTS PIN
//!
//! * the POSITIVE: the door emits the materialized copy (an `identity` opFunc carrying an
//!   indirect-access index, over the TABLE's tid) and then the matmul, whose B is the minted
//!   intermediate and NOT the table;
//! * CONTROL (the wrong-rows trap): the copy declares the INDEX operand and points its value
//!   operand at the TABLE's tid — the two facts whose absence is the silent direct read the guard
//!   exists to refuse, and the matmul does not name the table at all;
//! * CONTROL (fail-closed survival): a gather whose table is the matmul's **A** still refuses,
//!   with the original guard message — the exception did not swallow the fail-closed rule.

use ktir_core::affine::{AffineExpr, AffineMap};
use ktir_core::arena::Arena;
use ktir_core::attrkey::AttrKey;
use ktir_core::dtypes::DType;
use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
use ktir_core::irtype::IrType;
use ktir_core::opkind::OpKind;
use ktir_superdsc::emit::whole_function::lower_function;
use ktir_superdsc::ktir_node::{BufferId, KtirNode, Program};
use ktir_superdsc::place::{PlaceId, act_name};
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};

/// `[m=8, k=64] · [64, 64] -> [8, 64]` over a `[128, 64]` V table gathered by a `[64]` i32 index —
/// the rung-4 fixture's own geometry (M=8, K=64, V=128, HEAD_DIM=64), the smallest shape every
/// door admits (the `tl.dot` K floor, the matmul's whole-stick K, the gather's index-vector and
/// f16-column floors).
const M: i64 = 8;
const K: i64 = 64;
const V: i64 = 128;
const HEAD: i64 = 64;

/// Which matmul slot the gathered load feeds — the swap for the fail-closed control.
#[derive(Clone, Copy)]
enum GatheredSlot {
    /// The fixture's own shape: the gathered load is the matmul's B (operand 1).
    B,
    /// The gathered load feeds the matmul's **A** (operand 0), with B a plain load of the table —
    /// a gather this arm must NOT materialize. The walk's fail-closed refusal owns it.
    A,
}

/// One rung-4 program with a zero `outs`:
///
/// * `GatheredSlot::B` — `o = matmul(p, gather(v, ids))`, the fixture's own shape;
/// * `GatheredSlot::A` — `o = matmul(gather(v, ids), p)`, the same gather feeding the A slot with
///   B a plain load of the table.
///
/// Built from the op kinds `gather_of`, `region_for_operand` and `lower_function` actually read —
// `construct_memory_view`, `construct_access_tile`, `construct_indirect_access_tile`,
// `arith.constant`, `ktdp.load`, `linalg.matmul`, `ktdp.store` — so the door sees a real program
// and not a stub. The ONLY difference between the two cases is which matmul slot consumes the
// gathered load.
fn paged_program(slot: GatheredSlot) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };

    // Parameters: %0 = P, %1 = V table, %2 = ids, %3 = output.
    let (p_p, p_v, p_ids, p_out) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3));
    // The index constant 0 and the A chain.
    let (zero, v_a, t_a, val_a) = (Ssa(4), Ssa(5), Ssa(6), Ssa(7));
    // The V chain — the table's PLAIN view, for the non-gathered proof that V is a real tensor.
    let (v_v, t_v, val_v) = (Ssa(8), Ssa(9), Ssa(10));
    // The ids chain.
    let (v_i, t_i, val_i) = (Ssa(11), Ssa(12), Ssa(13));
    // The INDIRECT tile over the (possibly swapped) table view, and its load.
    let (ind, prod) = (Ssa(14), Ssa(15));
    // The output view/tile, the zero outs, and the matmul result.
    let (v_o, t_o, outs, matmul) = (Ssa(16), Ssa(17), Ssa(18), Ssa(19));

    // Which matmul slot consumes the gathered load — the control's one variable. Both slots are
    // `[64, 64]` here (K = HEAD_DIM = 64, attention's own square), so the swap changes NO extent
    // and no tid: only the POSITION of the gathered value.
    let (a_operand, b_operand) = match slot {
        GatheredSlot::B => (val_a, prod),
        GatheredSlot::A => (prod, val_a),
    };

    // The indirect dim's subscript — `Sym(0) + Dim(0)`, the one form `gather_subscripts` builds
    // and the one `gathers_of` folds into an entry base. Interned before the ops vec because its
    // children are borrows.
    let sub0 = a.expr(AffineExpr::Add(
        a.expr(AffineExpr::Sym(0)),
        a.expr(AffineExpr::Dim(0)),
    ));
    let ops = vec![
        Operation::new(a, Some(zero), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(0),
        ),
        // P: [8, 64]
        shape(
            Operation::new(a, Some(v_a), OpKind::KtdpConstructMemoryView, &[p_p]),
            M,
            K,
        ),
        shape(
            Operation::new(
                a,
                Some(t_a),
                OpKind::KtdpConstructAccessTile,
                &[v_a, zero, zero],
            ),
            M,
            K,
        ),
        shape(
            Operation::new(a, Some(val_a), OpKind::KtdpLoad, &[t_a]),
            M,
            K,
        ),
        // V: [128, 64] — the table.
        shape(
            Operation::new(a, Some(v_v), OpKind::KtdpConstructMemoryView, &[p_v]),
            V,
            HEAD,
        ),
        shape(
            Operation::new(
                a,
                Some(t_v),
                OpKind::KtdpConstructAccessTile,
                &[v_v, zero, zero],
            ),
            V,
            HEAD,
        ),
        shape(
            Operation::new(a, Some(val_v), OpKind::KtdpLoad, &[t_v]),
            V,
            HEAD,
        ),
        // ids: [64] i32 — the index vector.
        Operation::new(a, Some(v_i), OpKind::KtdpConstructMemoryView, &[p_ids])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![K])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::I32)),
        shape(
            Operation::new(
                a,
                Some(t_i),
                OpKind::KtdpConstructAccessTile,
                &[v_i, zero, zero],
            ),
            K,
            1,
        ),
        shape(
            Operation::new(a, Some(val_i), OpKind::KtdpLoad, &[t_i]),
            K,
            1,
        ),
        // THE INDIRECT TILE: data view first, index view second — `gather_of`'s own reading. Its
        // `Shape` is the GATHERED tile `[64, 64]`, over the V table in BOTH cases (the control
        // swaps the MATMUL SLOT, not the table — same table, same ids, same extents, only the
        // position of the gathered value changes). The anchor triple is what a REAL program
        // carries after `to_ktir_emit`'s re-base: dim 0 indirect with the `Sym(0) + Dim(0)`
        // subscript `gather_subscripts` builds, `Sym(0)` named by `intermediate_vars[0]` = the
        // zero constant — a one-block gather anchored at index word 0.
        shape(
            Operation::new(
                a,
                Some(ind),
                OpKind::KtdpConstructIndirectAccessTile,
                &[v_v, v_i],
            ),
            K,
            HEAD,
        )
        .with_attr(
            a,
            AttrKey::DimKinds,
            Attr::StrList(a.names(vec!["indirect", "direct_sub"])),
        )
        .with_attr(a, AttrKey::DimData, Attr::IntList(a.ints(vec![0, 0])))
        .with_attr(
            a,
            AttrKey::IntermediateVars,
            Attr::Ssas(a.ssa(vec![zero, zero])),
        )
        .with_attr(
            a,
            AttrKey::DimSubs,
            Attr::AffineMapList(a.maps(vec![
                AffineMap {
                    num_dims: 2,
                    num_syms: 2,
                    exprs: std::slice::from_ref(sub0),
                },
                AffineMap {
                    num_dims: 2,
                    num_syms: 2,
                    exprs: a.exprs(vec![AffineExpr::Dim(1)]),
                },
            ])),
        ),
        shape(
            Operation::new(a, Some(prod), OpKind::KtdpLoad, &[ind]),
            K,
            HEAD,
        ),
        // OUT: [8, 64]
        shape(
            Operation::new(a, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
            M,
            HEAD,
        ),
        shape(
            Operation::new(
                a,
                Some(t_o),
                OpKind::KtdpConstructAccessTile,
                &[v_o, zero, zero],
            ),
            M,
            HEAD,
        ),
        // The zero outs.
        shape(
            Operation::new(a, Some(outs), OpKind::TensorEmpty, &[]),
            M,
            HEAD,
        ),
        // THE CONTRACTION — plain-B, NO indexing maps: the gathered rows contracted where they lie.
        shape(
            Operation::new(
                a,
                Some(matmul),
                OpKind::LinalgMatmul,
                &[a_operand, b_operand, outs],
            ),
            M,
            HEAD,
        ),
        Operation::new(a, None, OpKind::KtdpStore, &[matmul, t_o]),
    ];

    KtirNode {
        func: IRFunction {
            name: "paged_vmatmul_fwd",
            arguments: a.args(vec![
                (p_p, IrType::Index),
                (p_v, IrType::Index),
                (p_ids, IrType::Index),
                (p_out, IrType::Index),
            ]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        },
        program: Program::Matmul,
        bindings: vec![
            BufferId::new(201),
            BufferId::new(202),
            BufferId::new(203),
            BufferId::new(204),
        ],
        mask: None,
        node_out_tid: None,
    }
}

/// A fully-placed layout for the four parameters — `resolve_seg_base` panics on a name that is in
/// neither `ids` nor the synth map, so every operand the emission names must be placed.
///
/// The table (t202) is placed ALONE in the Activation segment, at a NON-ZERO offset — so its
/// device address (`SEGMENT_OFFSETS[3] + 0x8000`) is shared by no other operand in the test, and
/// an assertion on that number is an assertion on WHICH buffer an op reads, not on a coincidence
/// of two zero offsets.
fn layout_for(k: &KtirNode) -> BundleLayout {
    let mut l = BundleLayout::default();
    for (i, b) in k.bindings.iter().enumerate() {
        let tid = b.get();
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        // Bytes per parameter: P [8,64] f16, V [128,64] f16, ids [64] i32, OUT [8,64] f16.
        let bytes: u64 = match i {
            0 => (M as u64) * (K as u64) * 2,
            1 => (V as u64) * (HEAD as u64) * 2,
            2 => (K as u64) * 4,
            _ => (M as u64) * (HEAD as u64) * 2,
        };
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: 3,
                bank: 0,
                // Distinct offsets inside seg3, so no two parameters share a base.
                offset: (i as u64) * 0x10000,
                size: bytes,
            },
        );
    }
    l
}

/// The emitted ops, descriptors and all — asserted on the TYPED structs rather than re-parsed
/// JSON, so the assertions name the fields the card acts on (`labeledDs_`, `scheduleTree_`) with
/// no second parser to disagree with the first.
fn emitted(k: &KtirNode) -> Result<Vec<ktir_superdsc::emit::EmittedOp>, String> {
    let layout = layout_for(k);
    let mut sid = 0i64;
    lower_function(k, Some(&layout), &mut sid).map_err(|e| e.message)
}

/// The device address the `dsType_`-tagged operand of `op` starts at — the alloc's per-core
/// `data_` value at core 0, joined to `labeledDs_` by `ldsIdx_`. This is the fact the card acts
/// on: two descriptors reading the same buffer name it through the same number, and the
/// descriptor names operands only positionally (`Tensor0/1/2`), so the address is the
/// discriminator.
fn alloc_address(op: &ktir_superdsc::emit::EmittedOp, tag: &str) -> Option<u64> {
    let dsc = op.dsc();
    let dsc = dsc.dscs_.first()?.values().next()?;
    let idx = dsc.labeledDs_.iter().find(|ds| ds.dsType_ == tag)?.ldsIdx_;
    let node = dsc
        .scheduleTree_
        .iter()
        .find(|n| n.ldsIdx_ == idx && n.nodeType_ == "allocate")?;
    node.startAddressCoreCorelet_
        .data_
        .get("[0, 0, 0]")?
        .parse()
        .ok()
}

/// The `value_tensor` alloc's base address in a gathered descriptor — the gather's SOURCE buffer.
fn value_tensor_address(op: &ktir_superdsc::emit::EmittedOp) -> Option<u64> {
    let dsc = op.dsc();
    let dsc = dsc.dscs_.first()?.values().next()?;
    let node = dsc
        .scheduleTree_
        .iter()
        .find(|n| n.nodeType_ == "allocate" && n.indirectAllocType_ == "value_tensor")?;
    node.startAddressCoreCorelet_
        .data_
        .get("[0, 0, 0]")?
        .parse()
        .ok()
}

/// The table t202's device address under [`layout_for`]: `SEGMENT_OFFSETS[3] + 0x10000` (parameter
/// 1's slot). The only buffer in the test at this address.
const TABLE_ADDR: u64 = 0xC_0000_0000 + 0x10000;

/// ⭐⭐⭐ THE POSITIVE: the door emits the materialized copy legs AND the matmul, in that order.
///
/// The copy is the KERNEL-less identity the vendor's own gathered fixture spells, split into one
/// op per index stick (`CopyDims::ENTRIES_PER_OP` = 32, so a 64-entry gather is TWO legs —
/// `t205_g0`/`t205_g1`, row-window writes via the leg's `first_row`), with the index operand the
/// direct-read trap would drop; and the matmul reads the MINTED INTERMEDIATE — proven by address,
/// because the descriptor names operands only positionally: the table sits at `TABLE_ADDR`
/// (seg3 + 0x10000) and the intermediate in the synth segment (seg0), so "the matmul's KERNEL
/// operand is not at `TABLE_ADDR`" is exactly "the matmul does not read the table".
#[test]
fn the_gathered_matmul_emits_the_copy_then_the_matmul() {
    let ops = emitted(&paged_program(GatheredSlot::B))
        .expect("the rung-4 program lowers through the whole-function door");
    // 64 entries / 32 per op = 2 copy legs, then the contraction.
    assert_eq!(
        ops.len(),
        3,
        "a gathered matmul emits two materialized copy legs (one per 32-entry index stick) and \
         the contraction -- got {:?}",
        ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
    );
    let (copy0, copy1, mm) = (&ops[0], &ops[1], &ops[2]);
    for copy in [copy0, copy1] {
        let dsc = copy.dsc();
        let dsc = dsc
            .dscs_
            .first()
            .and_then(|m| m.values().next())
            .expect("each copy leg has a descriptor");
        // The copy is the vendor's own gathered fixture shape: an `identity` opFunc with the
        // indirect-access index attached, and the value/index alloc pair.
        let identity = dsc
            .computeOp_
            .iter()
            .any(|c| c.opFuncName == "identity" && !c.indirectAccessIndexLabeledDs.is_empty());
        assert!(
            identity,
            "each copy leg is an `identity` carrying the indirect-access index (the wrong-rows \
             trap's whole point): {:?}",
            copy.op_name
        );
        let has_pair = dsc.scheduleTree_.iter().any(|n| {
            n.indirectAllocType_ == "value_tensor" && n.relatedIndirectAccessAlloc_.is_some()
        }) && dsc
            .scheduleTree_
            .iter()
            .any(|n| n.indirectAllocType_ == "index_tensor");
        assert!(
            has_pair,
            "each copy leg's value/index alloc pair is the gather itself: {:?}",
            copy.op_name
        );
        // The copy's VALUE alloc IS the TABLE, at the table's own address.
        assert_eq!(
            value_tensor_address(copy),
            Some(TABLE_ADDR),
            "each copy leg gathers from the table at {TABLE_ADDR:#x}: {:?}",
            copy.op_name
        );
    }
    // The contraction is a matmul.
    let mm_dsc = mm
        .dsc()
        .dscs_
        .first()
        .and_then(|m| m.values().next())
        .expect("the matmul has a descriptor");
    assert!(
        mm_dsc
            .computeOp_
            .iter()
            .any(|c| c.opFuncName == "batchmatmul"),
        "the last op is the contraction: {:?}",
        mm.op_name
    );
    // ⭐ THE MATMUL READS THE INTERMEDIATE, NOT THE TABLE — by address. A direct-read descriptor
    // (the trap this guard exists to refuse) would put the KERNEL operand at `TABLE_ADDR`.
    let b_addr = alloc_address(mm, "KERNEL");
    assert!(
        b_addr.is_some(),
        "the matmul declares a KERNEL (B) operand: {:?}",
        mm.op_name
    );
    assert_ne!(
        b_addr,
        Some(TABLE_ADDR),
        "the matmul's B is at {b_addr:#?}, NOT the table's {TABLE_ADDR:#x} -- a B at the table's \
         address is the direct-read trap (right shape, wrong rows): {:?}",
        mm.op_name
    );
}

/// ⛔ CONTROL — A GATHER THAT FEEDS THE MATMUL'S **A** STILL REFUSES, with the fail-closed guard's
/// own message. The exception must not have swallowed the rule.
///
/// ⭐ THE CONTROL'S OWN TRAP, WHICH THE FIRST DRAFT FELL INTO: the first control swapped the
/// TABLE the gather reads while leaving the gathered load in the matmul's B slot — which is a
/// DIFFERENT (and perfectly lowerable) program, `p @ gather(p, ids)`, and the door materialized
/// it correctly. The control that isolates the guard swaps the SLOT: same table, same ids, same
/// extents, the gathered value at operand 0 and a plain load at operand 1. That is also exactly
/// the shape a TID-comparing guard would have mis-lowered (both operands are `[64, 64]` at
/// K=64=HEAD_DIM, attention's own square), which is why the emission checks VALUE IDENTITY
/// (`in_values[1] == gathered_result`) and not tids.
#[test]
fn a_gather_over_the_matmuls_a_is_still_refused_by_the_guard() {
    let e = match emitted(&paged_program(GatheredSlot::A)) {
        Err(e) => e,
        Ok(_) => panic!("a gather over A is not the matmul's B -- the door must refuse it"),
    };
    assert!(
        e.contains("declares an index operand only for `Program::ScalarMul`"),
        "the ORIGINAL fail-closed refusal must fire -- got {e}"
    );
}
