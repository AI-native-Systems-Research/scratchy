// SPDX-License-Identifier: Apache-2.0

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
//! * THE A SLOT (restated for the walk the WIP left behind): a gathered read materializes for
//!   ANY consumer — the copy is the gather, and what reads its result reads an ordinary
//!   activation — so a gather feeding the matmul's **A** materializes exactly as the B-slot one
//!   does, on a shape-valid fixture (every extent square at K), and the matmul reads the copy.
//!   The old control pinned the dead law (a ScalarMul-only refusal for the A slot) and is
//!   replaced, not weakened: the fail-closed rule survives as its own case below — a gathered
//!   load with TWO consumers is refused by name, because the materialized copy stands for
//!   exactly one;
//! * THE POOLED TABLE (b1): a gathered source whose layout declares a `gather_reach` is a POOL
//!   SLICE — its placement covers one entry while its index names values across the whole pool.
//!   The door checks the read's ENTRY UNIT against the reach's own (the unit the host's
//!   `GatherEntry` composition counts in — a descriptor striding at any other width reads every
//!   entry at the wrong address), and the VALUES' bound is the ceiling the stage-side
//!   composition makes unconstructible (pinned in `zz_the_gather_index_reproduces_the_host_page_
//!   address`). The one-entry placement the swept-window law refuses is admitted under a reach,
//!   a reach whose entry unit DISAGREES with the read is refused by name, and the same placement
//!   with NO reach still refuses: the reach widens the indexed read and nothing else.

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

/// Which matmul slot the gathered load feeds.
#[derive(Clone, Copy)]
enum GatheredSlot {
    /// The fixture's own shape: the gathered load is the matmul's B (operand 1), `o = matmul(p,
    /// gather(v, ids))` with P `[8, 64]`.
    B,
    /// The gathered load feeds the matmul's **A** (operand 0) with the plain operand `[64, 64]`
    /// and the result `[64, 64]` — every extent square at K, so the fixture is SHAPE-VALID in
    /// this arrangement and the test pins the materialization itself, not a shape refusal.
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

    // Which matmul slot consumes the gathered load — the one variable. The B case is the
    // fixture's own `[8, 64] · [64, 64] -> [8, 64]`; the A case keeps every extent square at K
    // (`[64, 64] · [64, 64] -> [64, 64]`), because the gathered load `[K, HEAD]` is the
    // contraction's A there and the plain operand must carry K = 64 as its own K extent — the
    // first draft left P at `[8, 64]` and the control died at the matmul's own K-extent check,
    // which pins nothing about the gather.
    let (p_rows, p_cols, out_rows, out_cols) = match slot {
        GatheredSlot::B => (M, K, M, HEAD),
        GatheredSlot::A => (K, K, K, K),
    };

    // The swap itself: same table, same ids, only the POSITION of the gathered value changes.
    // That is also exactly the shape a TID-comparing guard would have mis-lowered (both operands
    // resolve to the same tids), which is why the emission checks VALUE IDENTITY
    // (`in_values[gi] == g.load`) and not tids.
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
        // P: the plain operand — [8, 64] as the B case's A, [64, 64] as the A case's B.
        shape(
            Operation::new(a, Some(v_a), OpKind::KtdpConstructMemoryView, &[p_p]),
            p_rows,
            p_cols,
        ),
        shape(
            Operation::new(
                a,
                Some(t_a),
                OpKind::KtdpConstructAccessTile,
                &[v_a, zero, zero],
            ),
            p_rows,
            p_cols,
        ),
        shape(
            Operation::new(a, Some(val_a), OpKind::KtdpLoad, &[t_a]),
            p_rows,
            p_cols,
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
        {
            // The gathered load's OWN type — the flat-law check's second authority. Every SSA in a
            // real TTIR function carries one (the splice states it); the fixture states it too, so
            // the check under test reads a real program and not a stub.
            let mut op = shape(
                Operation::new(a, Some(prod), OpKind::KtdpLoad, &[ind]),
                K,
                HEAD,
            );
            op.result_type = Some(IrType::Tensor {
                dims: a.ints(vec![K, HEAD]),
                elem: DType::F16,
            });
            op
        },
        // OUT: [8, 64] in the B case, [64, 64] in the A case.
        shape(
            Operation::new(a, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
            out_rows,
            out_cols,
        ),
        shape(
            Operation::new(
                a,
                Some(t_o),
                OpKind::KtdpConstructAccessTile,
                &[v_o, zero, zero],
            ),
            out_rows,
            out_cols,
        ),
        // The zero outs.
        shape(
            Operation::new(a, Some(outs), OpKind::TensorEmpty, &[]),
            out_rows,
            out_cols,
        ),
        // THE CONTRACTION — plain-B, NO indexing maps: the gathered rows contracted where they lie.
        shape(
            Operation::new(
                a,
                Some(matmul),
                OpKind::LinalgMatmul,
                &[a_operand, b_operand, outs],
            ),
            out_rows,
            out_cols,
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
        step_inputs: Vec::new(),
        node_out_tid: None,
        minted: Default::default(),
    }
}

/// A fully-placed layout for the four parameters — `resolve_seg_base` panics on a name that is in
/// neither `ids` nor the synth map, so every operand the emission names must be placed.
///
/// The table (t202) is placed ALONE in the Activation segment, at a NON-ZERO offset — so its
/// device address (`SEGMENT_OFFSETS[3] + 0x8000`) is shared by no other operand in the test, and
/// an assertion on that number is an assertion on WHICH buffer an op reads, not on a coincidence
/// of two zero offsets.
fn layout_for(k: &KtirNode, slot: GatheredSlot) -> BundleLayout {
    let mut l = BundleLayout::default();
    for (i, b) in k.bindings.iter().enumerate() {
        let tid = b.get();
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        // Bytes per parameter: P (the plain operand, slot-shaped), V [128,64] f16, ids [64] i32,
        // OUT (the result, slot-shaped).
        let bytes: u64 = match (slot, i) {
            (_, 1) => (V as u64) * (HEAD as u64) * 2,
            (_, 2) => (K as u64) * 4,
            // P: [8, 64] in the B case, [64, 64] in the A case.
            (GatheredSlot::B, 0) => (M as u64) * (K as u64) * 2,
            (GatheredSlot::A, 0) => (K as u64) * (K as u64) * 2,
            // OUT: [8, 64] in the B case, [64, 64] in the A case.
            (GatheredSlot::B, _) => (M as u64) * (HEAD as u64) * 2,
            (GatheredSlot::A, _) => (K as u64) * (K as u64) * 2,
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
                cover: ktir_superdsc::placement::BindCover::Full,
            },
        );
    }
    l
}

/// The emitted ops, descriptors and all — asserted on the TYPED structs rather than re-parsed
/// JSON, so the assertions name the fields the card acts on (`labeledDs_`, `scheduleTree_`) with
/// no second parser to disagree with the first.
fn emitted(
    k: &KtirNode,
    slot: GatheredSlot,
) -> Result<Vec<ktir_superdsc::emit::EmittedOp>, String> {
    let layout = layout_for(k, slot);
    let mut sid = 0i64;
    lower_function(k, Some(&layout), &mut sid, None).map_err(|e| e.message)
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
    let ops = emitted(&paged_program(GatheredSlot::B), GatheredSlot::B)
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

/// ⭐⭐⭐ THE A SLOT, RESTATED FOR THE LAW THE WALK NOW RUNS: a gathered read materializes for ANY
/// consumer — the copy is the gather, and what reads its result reads an ordinary activation —
/// so a gather feeding the matmul's **A** emits the same two copy legs and a contraction whose
/// A is the minted intermediate, NOT the table.
///
/// ⛔ THE OLD CONTROL PINNED A DEAD LAW, AND THIS IS ITS REPLACEMENT, NOT A WEAKENING. The old
/// test expected the walk's ScalarMul-only refusal to fire for the A slot; the walk it inherited
/// (38b5c73c9) materializes every gathered read instead, so the refusal never fires and the
/// fixture died at the matmul's own K-extent check — P was left `[8, 64]`, which is not a valid
/// B against a `[64, 64]` A. This fixture is SHAPE-VALID in the A arrangement (every extent
/// square at K), so what it pins is the materialization itself. The fail-closed rule survives
/// as its own case in the test below this one.
///
/// The A operand is proven by ADDRESS, the same discriminator the B-slot positive uses: the
/// table sits at `TABLE_ADDR` (seg3 + 0x10000) and the intermediate in the synth segment, so
/// "the matmul's INPUT operand is not at `TABLE_ADDR`" is exactly "the matmul does not read the
/// table".
#[test]
fn an_a_side_gathered_operand_materializes_and_the_matmul_reads_the_copy() {
    let ops = emitted(&paged_program(GatheredSlot::A), GatheredSlot::A)
        .expect("an A-side gathered operand materializes like any other gathered read");
    assert_eq!(
        ops.len(),
        3,
        "the A-slot gather emits the same two materialized copy legs and the contraction -- got \
         {:?}",
        ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
    );
    let (copy0, _copy1, mm) = (&ops[0], &ops[1], &ops[2]);
    // The copy legs are the gather itself — the same identity-with-index shape as the B case.
    let dsc = copy0
        .dsc()
        .dscs_
        .first()
        .and_then(|m| m.values().next())
        .expect("each copy leg has a descriptor");
    assert!(
        dsc.computeOp_
            .iter()
            .any(|c| c.opFuncName == "identity" && !c.indirectAccessIndexLabeledDs.is_empty()),
        "the A-slot copy leg is an `identity` carrying the indirect-access index: {:?}",
        copy0.op_name
    );
    assert_eq!(
        value_tensor_address(copy0),
        Some(TABLE_ADDR),
        "the A-slot copy leg gathers from the table at {TABLE_ADDR:#x}: {:?}",
        copy0.op_name
    );
    // ⭐ THE MATMUL READS THE INTERMEDIATE IN ITS **A** SLOT, NOT THE TABLE — by address. A
    // direct-read descriptor (the trap the guard exists to refuse) would put the INPUT operand
    // at `TABLE_ADDR`.
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
    let a_addr = alloc_address(mm, "INPUT");
    assert!(
        a_addr.is_some(),
        "the matmul declares an INPUT (A) operand: {:?}",
        mm.op_name
    );
    assert_ne!(
        a_addr,
        Some(TABLE_ADDR),
        "the matmul's A is at {a_addr:#?}, NOT the table's {TABLE_ADDR:#x} -- an A at the table's \
         address is the direct-read trap (right shape, wrong rows): {:?}",
        mm.op_name
    );
}

/// ⛔ THE FAIL-CLOSED RULE SURVIVES THE MATERIALIZE-ANY-GATHERED-READ LAW, AS ITS OWN CASE: a
/// gathered load with **TWO** consumers is refused by name, because the materialized copy this
/// door emits stands for exactly one — a second consumer needs its own node, and silently
/// sharing the copy would read the first consumer's rows. This is the refusal half the old
/// A-slot control stood for, restated for the law that replaced it.
#[test]
fn a_gathered_load_with_two_consumers_is_refused_by_name() {
    // The B fixture with ONE extra op: the gathered load `prod` feeds BOTH the matmul and a
    // second store of its own — two consumers of one gathered read.
    let mut k = paged_program(GatheredSlot::B);
    let a: &'static Arena = Arena::global();
    let zero = Ssa(4);
    // A second output parameter for the gathered rows themselves.
    let (p_g, v_g, t_g) = (Ssa(20), Ssa(21), Ssa(22));
    let store_g = Ssa(23);
    k.func.arguments = a.args(vec![
        (Ssa(0), IrType::Index),
        (Ssa(1), IrType::Index),
        (Ssa(2), IrType::Index),
        (Ssa(3), IrType::Index),
        (p_g, IrType::Index),
    ]);
    k.bindings.push(BufferId::new(205));
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };
    // `prod` (Ssa 15) is the gathered load; store it raw beside the contraction.
    let extra = vec![
        shape(
            Operation::new(a, Some(v_g), OpKind::KtdpConstructMemoryView, &[p_g]),
            K,
            HEAD,
        ),
        shape(
            Operation::new(
                a,
                Some(t_g),
                OpKind::KtdpConstructAccessTile,
                &[v_g, zero, zero],
            ),
            K,
            HEAD,
        ),
        Operation::new(a, Some(store_g), OpKind::KtdpStore, &[Ssa(15), t_g]),
    ];
    let mut ops = k.func.operations.to_vec();
    ops.extend(extra);
    k.func.operations = a.ops(ops);
    let mut l = layout_for(&k, GatheredSlot::B);
    let tid = 205;
    l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
    l.placements.insert(
        tid,
        TensorPlacement {
            tid,
            role: SegRole::Activation,
            segment: 3,
            bank: 0,
            offset: 4 * 0x10000,
            size: (K as u64) * (HEAD as u64) * 2,
            cover: ktir_superdsc::placement::BindCover::Full,
        },
    );
    let mut sid = 0i64;
    let e = match lower_function(&k, Some(&l), &mut sid, None) {
        Err(e) => e.message,
        Ok(_) => panic!("a gathered load with two consumers must refuse, not share one copy"),
    };
    assert!(
        e.contains("2 consumer(s)"),
        "the refusal names the consumer count and the law (one copy stands for exactly one \
         consumer) -- got {e}"
    );
}

// ── THE BOTH-GATHERED CONTRACTION (the MoE honesty row, at a non-attention geometry) ────────
//
// A MoE-style expert contraction gathers BOTH operands — token rows gathered by one index
// vector over one table, expert weights gathered by another index vector over a second table —
// and the law (a) series must not care WHICH side a gather feeds: each gathered input
// materializes through its OWN table and its OWN index buffer, and the contraction reads both
// minted intermediates. This fixture spells that shape with the file's own constants (no
// attention geometry: no batch axis, no pool, both tables placed WHOLE):
//
//   out = matmul(gather(table_a, ids_a), gather(table_b, ids_b))
//
// with `gather(table_a, ids_a)` sweeping `[M, K] = [8, 64]` rows of a `[128, 64]` table and
// `gather(table_b, ids_b)` sweeping `[K, HEAD] = [64, 64]` rows of a second `[128, 64]` table.
// The A-side sweep is 8 entries (ONE copy leg at 32 entries per index stick) and the B-side
// sweep is 64 entries (TWO legs), so the expected emission is 1 + 2 copy legs and the
// contraction: FOUR ops, each copy carrying its own table's address.

/// The both-gathered program: five parameters (table_a, ids_a, table_b, ids_b, out — the
/// operand prefix, output LAST, the splice's own binding law), two independent gather chains
/// built the same way [`paged_program`] builds its one chain, and one `linalg.matmul` over both
/// gathered loads.
fn both_gathered_program() -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };

    // Parameters: %0 = table_a, %1 = ids_a, %2 = table_b, %3 = ids_b, %4 = out.
    let (p_ta, p_ia, p_tb, p_ib, p_out) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3), Ssa(4));
    let zero = Ssa(5);
    // The A chain: table_a's plain view/tile/load, ids_a's view/tile/load, the indirect tile
    // [M, K] and its gathered load.
    let (v_ta, t_ta, val_ta, v_ia, t_ia, val_ia, ind_a, load_a) = (
        Ssa(6),
        Ssa(7),
        Ssa(8),
        Ssa(9),
        Ssa(10),
        Ssa(11),
        Ssa(12),
        Ssa(13),
    );
    // The B chain: the same five-op shape over table_b/ids_b, sweeping [K, HEAD].
    let (v_tb, t_tb, val_tb, v_ib, t_ib, val_ib, ind_b, load_b) = (
        Ssa(14),
        Ssa(15),
        Ssa(16),
        Ssa(17),
        Ssa(18),
        Ssa(19),
        Ssa(20),
        Ssa(21),
    );
    // The output view/tile, the zero outs, and the contraction.
    let (v_o, t_o, outs, matmul) = (Ssa(22), Ssa(23), Ssa(24), Ssa(25));

    // The indirect dim's subscript — `Sym(0) + Dim(0)`, the one form `gather_subscripts` builds.
    let sub0 = a.expr(AffineExpr::Add(
        a.expr(AffineExpr::Sym(0)),
        a.expr(AffineExpr::Dim(0)),
    ));
    // One gather chain: the table's plain view/tile/load (the "this is a real tensor" proof),
    // the ids view/tile/load, then the indirect tile over (table view, ids view) and its load —
    // `gather_of`'s own reading, data view first, index view second.
    let gather_chain = |p_table: Ssa,
                        p_ids: Ssa,
                        (v_t, t_t, val_t, v_i, t_i, val_i, ind, load): (
        Ssa,
        Ssa,
        Ssa,
        Ssa,
        Ssa,
        Ssa,
        Ssa,
        Ssa,
    ),
                        rows: i64,
                        width: i64,
                        entries: i64,
                        ops: &mut Vec<Operation<'static>>| {
        // The table: [V, HEAD] in both chains.
        ops.push(shape(
            Operation::new(a, Some(v_t), OpKind::KtdpConstructMemoryView, &[p_table]),
            V,
            HEAD,
        ));
        ops.push(shape(
            Operation::new(
                a,
                Some(t_t),
                OpKind::KtdpConstructAccessTile,
                &[v_t, zero, zero],
            ),
            V,
            HEAD,
        ));
        ops.push(shape(
            Operation::new(a, Some(val_t), OpKind::KtdpLoad, &[t_t]),
            V,
            HEAD,
        ));
        // The ids: [entries] i32.
        ops.push(
            Operation::new(a, Some(v_i), OpKind::KtdpConstructMemoryView, &[p_ids])
                .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![entries])))
                .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::I32)),
        );
        ops.push(shape(
            Operation::new(
                a,
                Some(t_i),
                OpKind::KtdpConstructAccessTile,
                &[v_i, zero, zero],
            ),
            entries,
            1,
        ));
        ops.push(shape(
            Operation::new(a, Some(val_i), OpKind::KtdpLoad, &[t_i]),
            entries,
            1,
        ));
        // The indirect tile: Shape [rows, width], the anchor triple `gathers_of` folds.
        ops.push(
            shape(
                Operation::new(
                    a,
                    Some(ind),
                    OpKind::KtdpConstructIndirectAccessTile,
                    &[v_t, v_i],
                ),
                rows,
                width,
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
        );
        // The gathered load — its OWN type stated, the flat-law check's second authority.
        let mut op = shape(
            Operation::new(a, Some(load), OpKind::KtdpLoad, &[ind]),
            rows,
            width,
        );
        op.result_type = Some(IrType::Tensor {
            dims: a.ints(vec![rows, width]),
            elem: DType::F16,
        });
        ops.push(op);
    };

    let mut ops = vec![
        Operation::new(a, Some(zero), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(0),
        ),
    ];
    gather_chain(
        p_ta,
        p_ia,
        (v_ta, t_ta, val_ta, v_ia, t_ia, val_ia, ind_a, load_a),
        M,
        K,
        M,
        &mut ops,
    );
    gather_chain(
        p_tb,
        p_ib,
        (v_tb, t_tb, val_tb, v_ib, t_ib, val_ib, ind_b, load_b),
        K,
        HEAD,
        K,
        &mut ops,
    );
    // OUT: [M, HEAD] = [8, 64].
    ops.push(shape(
        Operation::new(a, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
        M,
        HEAD,
    ));
    ops.push(shape(
        Operation::new(
            a,
            Some(t_o),
            OpKind::KtdpConstructAccessTile,
            &[v_o, zero, zero],
        ),
        M,
        HEAD,
    ));
    ops.push(shape(
        Operation::new(a, Some(outs), OpKind::TensorEmpty, &[]),
        M,
        HEAD,
    ));
    ops.push(shape(
        Operation::new(
            a,
            Some(matmul),
            OpKind::LinalgMatmul,
            &[load_a, load_b, outs],
        ),
        M,
        HEAD,
    ));
    ops.push(Operation::new(a, None, OpKind::KtdpStore, &[matmul, t_o]));

    KtirNode {
        func: IRFunction {
            name: "both_gathered_matmul_fwd",
            arguments: a.args(vec![
                (p_ta, IrType::Index),
                (p_ia, IrType::Index),
                (p_tb, IrType::Index),
                (p_ib, IrType::Index),
                (p_out, IrType::Index),
            ]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        },
        program: Program::Matmul,
        bindings: vec![
            BufferId::new(301),
            BufferId::new(302),
            BufferId::new(303),
            BufferId::new(304),
            BufferId::new(305),
        ],
        mask: None,
        step_inputs: Vec::new(),
        node_out_tid: None,
        minted: Default::default(),
    }
}

/// [`layout_for`]'s shape for the five parameters: both tables placed WHOLE at distinct
/// non-zero seg3 offsets (table_a at 0x10000, table_b at 0x30000), so each table's device
/// address is shared by no other operand and an assertion on an address is an assertion on
/// WHICH table an op reads.
fn both_gathered_layout(k: &KtirNode) -> BundleLayout {
    let mut l = BundleLayout::default();
    for (i, b) in k.bindings.iter().enumerate() {
        let tid = b.get();
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        let bytes: u64 = match i {
            // table_a and table_b: [V, HEAD] f16 each.
            0 | 2 => (V as u64) * (HEAD as u64) * 2,
            // ids_a [M] i32, ids_b [K] i32.
            1 => (M as u64) * 4,
            3 => (K as u64) * 4,
            // out: [M, HEAD] f16.
            _ => (M as u64) * (HEAD as u64) * 2,
        };
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: 3,
                bank: 0,
                offset: ((i + 1) as u64) * 0x10000,
                size: bytes,
                cover: ktir_superdsc::placement::BindCover::Full,
            },
        );
    }
    l
}

/// table_a's device address under [`both_gathered_layout`]: seg3 + 0x10000 (parameter 0's
/// slot). The only buffer in the test at this address.
const TABLE_A_ADDR: u64 = 0xC_0000_0000 + 0x10000;

/// table_b's device address: seg3 + 0x30000 (parameter 2's slot). The only buffer at this one.
const TABLE_B_ADDR: u64 = 0xC_0000_0000 + 0x30000;

/// ⭐⭐⭐ THE BOTH-GATHERED HONESTY ROW: a contraction whose EVERY input is gathered materializes
/// EACH input through its OWN table and index buffer — the A-side sweep is 8 entries (ONE copy
/// leg), the B-side sweep is 64 entries (TWO legs), and the contraction reads BOTH minted
/// intermediates, proven by address against BOTH tables. The MoE shape is the motivating case
/// (token rows × expert weights, both gathered); the law is stated over inputs, so this row is
/// the one that fails if any "which side is gathered" rule survives in the door.
#[test]
fn a_both_gathered_matmul_materializes_each_input_through_its_own_table() {
    let k = both_gathered_program();
    let ops = emitted_with(&k, &both_gathered_layout(&k)).expect(
        "a contraction with both operands gathered materializes each through its own table",
    );
    assert_eq!(
        ops.len(),
        4,
        "the both-gathered contraction emits ONE A-side copy leg (8 entries, one index stick), \
         TWO B-side copy legs (64 entries) and the contraction -- got {:?}",
        ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
    );
    let (copy_a, copy_b0, copy_b1, mm) = (&ops[0], &ops[1], &ops[2], &ops[3]);
    // The A-side copy gathers from table_a — its OWN table, not table_b's.
    assert_eq!(
        value_tensor_address(copy_a),
        Some(TABLE_A_ADDR),
        "the A-side copy leg gathers from table_a at {TABLE_A_ADDR:#x}: {:?}",
        copy_a.op_name
    );
    // Each B-side copy gathers from table_b.
    for copy in [copy_b0, copy_b1] {
        assert_eq!(
            value_tensor_address(copy),
            Some(TABLE_B_ADDR),
            "each B-side copy leg gathers from table_b at {TABLE_B_ADDR:#x}: {:?}",
            copy.op_name
        );
    }
    // Every copy is the gather itself: identity with the indirect-access index attached.
    for copy in [copy_a, copy_b0, copy_b1] {
        let dsc = copy
            .dsc()
            .dscs_
            .first()
            .and_then(|m| m.values().next())
            .expect("each copy leg has a descriptor");
        assert!(
            dsc.computeOp_
                .iter()
                .any(|c| c.opFuncName == "identity" && !c.indirectAccessIndexLabeledDs.is_empty()),
            "each copy leg is an `identity` carrying the indirect-access index: {:?}",
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
    // ⭐ THE MATMUL READS NEITHER TABLE, IN EITHER SLOT — by address. A direct-read descriptor
    // (the trap this guard exists to refuse) would put INPUT at table_a's address or KERNEL at
    // table_b's; a cross-wired materialization would swap them.
    for tag in ["INPUT", "KERNEL"] {
        let addr = alloc_address(mm, tag);
        assert!(
            addr.is_some(),
            "the matmul declares its {tag} operand: {:?}",
            mm.op_name
        );
        assert_ne!(
            addr,
            Some(TABLE_A_ADDR),
            "the matmul's {tag} is at {addr:#?}, NOT table_a's {TABLE_A_ADDR:#x} -- an operand at \
             the table's address is the direct-read trap: {:?}",
            mm.op_name
        );
        assert_ne!(
            addr,
            Some(TABLE_B_ADDR),
            "the matmul's {tag} is at {addr:#?}, NOT table_b's {TABLE_B_ADDR:#x} -- an operand at \
             the table's address is the direct-read trap: {:?}",
            mm.op_name
        );
    }
}

/// ⭐⭐⭐ THE SOURCE-EXTENT FAMILY'S OWN ROW: a repeat gather — a sweep whose EVERY LEG honestly
/// EXCEEDS its source, one row read many entries' worth of times — admits under a declared
/// `GatherReach::of_source_extent` (ceiling = the source's own row count, unit = its row
/// width), and REFUSES, fail-closed, without one. The mint runs PER LEG (one op per 32-entry
/// index stick), so the over-sweep the law bounds is the LEG's: the fixture is the A-slot
/// program (a 64-entry gather = two 32-row legs) with the table's placement shrunk to 16 of
/// its 128 rows (2,048 B) — each leg sweeps 32 rows × 64 elements (4,096 B) past a source that
/// holds 16, and the reach states the bound the placement alone cannot: the values name rows
/// of the 16 placed rows. The window lands at `min(32, 16) = 16` rows = 2,048 B = the
/// placement. One variable away, the same placement with NO reach is the refusal — the
/// fail-closed half, pinned at THIS geometry (the pooled control below pins it at the slice
/// geometry).
#[test]
fn a_repeat_gather_over_a_whole_placed_source_admits_under_a_declared_source_extent_reach() {
    let k = paged_program(GatheredSlot::A);
    // The table placed at 16 rows — every 32-row leg over-sweeps it.
    let shrink = |l: &mut BundleLayout| {
        let tid = k.bindings[1].get();
        l.placements
            .get_mut(&tid)
            .expect("the table is placed")
            .size = 16 * HEAD as u64 * 2;
    };
    // ⛔ FAIL-CLOSED FIRST — the same placement with NO reach refuses, naming the sweep, the
    // placement and the missing declaration.
    let mut bare = layout_for(&k, GatheredSlot::A);
    shrink(&mut bare);
    let e = match emitted_with(&k, &bare) {
        Err(e) => e,
        Ok(_) => panic!("an over-sweep with no declared reach must refuse, fail-closed"),
    };
    assert!(
        e.contains("exceeds its placement footprint"),
        "the refusal names the placement overrun -- got {e}"
    );
    assert!(
        e.contains("Declare the reach"),
        "the refusal names the missing declaration -- got {e}"
    );
    // ⭐ THE ADMISSION — under the declared reach, the same emission the fitting sweep gets.
    // The reach: the source's OWN extent — ceiling 16 rows, unit the 64-element row.
    let reach = ktir_superdsc::sdsc_abstract::GatherReach::of_source_extent(16, HEAD as u64)
        .expect("16 rows at a 64-element unit is a legal source extent");
    assert_eq!(
        reach.entry_elems(),
        HEAD as u64,
        "the source-extent unit is the row width this read strides at"
    );
    assert_eq!(
        reach.max_entry().get(),
        16,
        "the source-extent ceiling is the source's own row count (values < 16)"
    );
    let mut l = layout_for(&k, GatheredSlot::A);
    shrink(&mut l);
    l.gather_reach.insert(k.bindings[1].get(), reach);
    let ops = emitted_with(&k, &l).expect(
        "a repeat gather admits under a declared source-extent reach: the values name rows of \
         the placed source",
    );
    assert_eq!(
        ops.len(),
        3,
        "the repeat gather emits the same two copy legs and the contraction -- got {:?}",
        ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
    );
    for copy in &ops[..2] {
        assert_eq!(
            value_tensor_address(copy),
            Some(TABLE_ADDR),
            "each copy leg gathers from the table's own address: {:?}",
            copy.op_name
        );
    }
    let mm = &ops[2];
    let a_addr = alloc_address(mm, "INPUT");
    assert_ne!(
        a_addr,
        Some(TABLE_ADDR),
        "the matmul's A is the minted intermediate, NOT the table: {:?}",
        mm.op_name
    );
}

// ── THE POOLED TABLE (b1) ──────────────────────────────────────────────────────────────────
//
// The fixture is `paged_program(GatheredSlot::B)` unchanged; only the LAYOUT changes. The table
// (t202) is placed at ONE ENTRY's footprint (`[1, HEAD]` = 128 B) instead of the whole
// `[V, HEAD]`, and the bundle declares the table's `gather_reach` as the TYPED pair — the entry
// ceiling and the entry unit — minted through the pool's own door:
//
//   * the SWEPT window is 64 × 64 = 4096 elements = 8192 B — 64× the one-entry placement, which
//     the swept-window law (the placement footprint) refuses;
//   * the reach's entry unit must be the width the read itself strides at (HEAD = 64), which a
//     pool at hd=1 states (a stick block is `hd · 64`) and a pool at hd=2 does not — that
//     disagreement is the door's refusal, and the test that pins it FAILS WITHOUT the check.

/// The pool the fixture's table models, at head dim `hd`: one reach minted through the pool's
/// own door (`GatherReach::of_plane`), with a 4-page budget over a 128-entry page (four whole
/// planes — a page smaller than one of its own planes is not a pool, and the door refuses it).
/// At `hd = 1` the pool's stick block IS the fixture's HEAD (64 elements); at `hd = 2` it is
/// twice that, which is exactly the disagreement the door must refuse.
fn fixture_reach(hd: usize) -> Option<ktir_superdsc::sdsc_abstract::GatherReach> {
    let pool = ktir_superdsc::sdsc_abstract::PagedKvPool::new(8, hd as u32);
    let page_stride = 128 * pool.stick_block_bytes();
    ktir_superdsc::sdsc_abstract::GatherReach::of_plane(4, page_stride, pool)
}

/// [`layout_for`] with the table (t202) placed at ONE ENTRY's footprint and a declared
/// `gather_reach` — the pooled-table statement: a slice placement plus the typed bound on the
/// index values that address the pool behind it.
fn pooled_layout_for(
    k: &KtirNode,
    reach: ktir_superdsc::sdsc_abstract::GatherReach,
) -> BundleLayout {
    let mut l = layout_for(k, GatheredSlot::B);
    let tid = k.bindings[1].get();
    l.placements
        .get_mut(&tid)
        .expect("the table is placed")
        .size = HEAD as u64 * 2;
    l.gather_reach.insert(tid, reach);
    l
}

/// [`emitted`] under a caller-built layout — the pooled cases' door, so the reach and the
/// placement it bounds are stated by the test and not re-derived.
fn emitted_with(
    k: &KtirNode,
    layout: &BundleLayout,
) -> Result<Vec<ktir_superdsc::emit::EmittedOp>, String> {
    let mut sid = 0i64;
    lower_function(k, Some(layout), &mut sid, None).map_err(|e| e.message)
}

/// ⭐ THE POSITIVE: a one-entry placement the swept-window law refuses is ADMITTED under a
/// declared reach — the door checks the read's entry unit against the reach's own (64 = 64) and
/// the mint checks ONE entry's coverage against the slice (64 × 2 B = 128 B ≤ 128 B). The
/// emission is the rung-4 shape itself: two copy legs and the contraction, the legs still
/// gathering FROM the table's own address (the slice placement's base — the pool's base).
#[test]
fn a_pooled_table_gathers_past_its_one_slice_placement() {
    let k = paged_program(GatheredSlot::B);
    let reach = fixture_reach(1).expect("hd=1 states the fixture's own entry unit");
    assert_eq!(
        reach.entry_elems(),
        HEAD as u64,
        "the hd=1 pool's stick block is the fixture's HEAD — the unit the read strides at"
    );
    let ops = emitted_with(&k, &pooled_layout_for(&k, reach))
        .expect("a pooled table's gather is bounded by its reach, not its slice placement");
    assert_eq!(
        ops.len(),
        3,
        "the pooled gather emits the same two copy legs and the contraction -- got {:?}",
        ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
    );
    for copy in &ops[..2] {
        assert_eq!(
            value_tensor_address(copy),
            Some(TABLE_ADDR),
            "each copy leg gathers from the table's own (slice) placement base: {:?}",
            copy.op_name
        );
    }
}

/// ⛔ THE REFUSAL, ON THE FACT THE DOOR ALONE CAN SEE: the read's ENTRY UNIT. This node gathers
/// 64-element rows, but the declared pool's entries are 128-element stick blocks (an hd=2 pool)
/// — two units for one address. dxp turns a value into an address as `idx * skip_addr + base`,
/// and the host composes those values in the pool's own unit, so a descriptor striding at 64
/// reads every entry at half its address — from a clean bake. The refusal names the op, the
/// table, both widths and the ceiling.
///
/// ⭐ THIS IS THE TEST THAT FAILS WITHOUT THE CHECK: with no unit law (the state this commit
/// fixes), the hd=2 reach is admitted and the emission succeeds — the very defect the audit
/// named, a bound that looks like a law and pins nothing.
#[test]
fn a_pooled_gather_whose_entry_unit_disagrees_with_the_pool_is_refused_by_name() {
    let k = paged_program(GatheredSlot::B);
    let reach = fixture_reach(2).expect("hd=2 states a wider entry unit");
    assert_eq!(
        reach.entry_elems(),
        2 * HEAD as u64,
        "the hd=2 pool's stick block is twice the fixture's HEAD -- the disagreement under test"
    );
    let e = match emitted_with(&k, &pooled_layout_for(&k, reach)) {
        Err(e) => e,
        Ok(_) => panic!(
            "a read striding 64-element entries over a pool of 128-element stick blocks must \
             refuse -- two units for one address"
        ),
    };
    assert!(
        e.contains("strides one index entry at 64"),
        "the refusal names the read's own unit -- got {e}"
    );
    assert!(
        e.contains("128-element stick blocks"),
        "the refusal names the pool's own unit -- got {e}"
    );
}

/// ⛔ CONTROL — THE REACH WIDENS THE INDEXED READ AND NOTHING ELSE: the SAME one-entry
/// placement with NO declared reach still refuses at the placement footprint. Without this, the
/// positive above proves nothing — a door that stopped checking placements would pass it too.
#[test]
fn a_one_slice_table_without_a_reach_still_refuses_at_the_placement() {
    let k = paged_program(GatheredSlot::B);
    let reach = fixture_reach(1).expect("hd=1 states the fixture's own entry unit");
    let mut l = pooled_layout_for(&k, reach);
    let tid = k.bindings[1].get();
    l.gather_reach.remove(&tid);
    let e = match emitted_with(&k, &l) {
        Err(e) => e,
        Ok(_) => panic!("a swept window of 8192B past a 128B placement must refuse"),
    };
    assert!(
        e.contains("exceeds its placement footprint"),
        "the un-reached table keeps the swept-window law -- got {e}"
    );
}

// ── THE LEG-REACH PIN (defect C, RE-POINTED at the carrier form) ───────────────────────────
//
// The gathered fact's two halves: the STAGING half (the bundle's layout reserves an index tensor)
// and the ADDRESSING half (`EmittedOp::kv_gathered`, which the runtime's page-base composition
// asks — `nonfold_page_delta` and the composed chunk index). Defect C was the ONE-SHOT attention
// carriers minting the addressing half FALSE while reading the fill-staged region form, so the
// runtime composed the page base on top of the operand's own pool base and attention read one page
// past its keys. The cure states the fact at the CARRIER's mint (`EmittedOp::paged_region`, the
// one door) and stamps it in the walk onto every op lowered from that carrier — so the DOOR no
// longer stamps anything, and this pin holds the ONE-STAMPING-SITE law at the door the corpus
// drives, in all three cases:
//
//   * a POOL-PLANE reach (`GatherReach::of_plane`)      ⇒ the door's ops carry `false` — the
//     fact is the carrier's to state, and a door stamp would be a SECOND writer for the same
//     fact on the same ops (the parked general-door class; no non-carrier pool program exists
//     in the tree's models today);
//   * NO reach (a sweep that fits its placement)        ⇒ every op carries `false`;
//   * a SOURCE-EXTENT reach (`of_source_extent`)        ⇒ every op carries `false`.
//
// The walk's region stamp is pinned where it lives, in the spyre crate's own corpus — this file
// can only see the door.
#[test]
fn the_leg_reach_law_three_cases() {
    // ⭐ CASE 1 — POOL-PLANE: the pooled-table statement's own shape (a one-entry slice placement
    // under an `of_plane` reach), and the door must NOT stamp it — the carrier mint is the one
    // writer. A `true` here would be a second stamping site for the addressing fact, the desync
    // the one-door law exists to prevent.
    let k = paged_program(GatheredSlot::B);
    let reach = fixture_reach(1).expect("hd=1 states the fixture's own entry unit");
    let ops = emitted_with(&k, &pooled_layout_for(&k, reach))
        .expect("a pooled table's gather is bounded by its reach, not its slice placement");
    assert_eq!(
        ops.len(),
        3,
        "the pooled gather emits two copy legs and the contraction -- got {:?}",
        ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
    );
    for o in &ops {
        assert!(
            !o.kv_gathered,
            "the door does not stamp the gathered bit -- the carrier mint \
             (EmittedOp::paged_region) is the ONE writer: {:?}",
            o.op_name
        );
    }

    // ⛔ CASE 2 — NO REACH: the whole-placed table whose sweep fits, the embedding's own
    // card-proven shape. Nothing here addresses a pool, so every op must carry `false`.
    let ops = emitted_with(&k, &layout_for(&k, GatheredSlot::B))
        .expect("a fitting sweep keeps the swept-window law, byte for byte");
    assert_eq!(
        ops.len(),
        3,
        "the fitting sweep emits the same two copy legs and the contraction -- got {:?}",
        ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
    );
    for o in &ops {
        assert!(
            !o.kv_gathered,
            "an op with no declared reach carries no gathered bit: {:?}",
            o.op_name
        );
    }

    // ⛔ CASE 3 — SOURCE-EXTENT: the whole-placed-source statement's own admitted arm (the table
    // shrunk to 16 rows under an `of_source_extent(16, HEAD)` reach). The presence of a reach is
    // NOT the pool fact — the entries are rows of the placement itself, not stick-blocks of a
    // pool — so every op must carry `false`. A presence-keyed stamp would fail exactly here.
    let k = paged_program(GatheredSlot::A);
    let shrink = |l: &mut BundleLayout| {
        let tid = k.bindings[1].get();
        l.placements
            .get_mut(&tid)
            .expect("the table is placed")
            .size = 16 * HEAD as u64 * 2;
    };
    let reach = ktir_superdsc::sdsc_abstract::GatherReach::of_source_extent(16, HEAD as u64)
        .expect("16 rows at a 64-element unit is a legal source extent");
    let mut l = layout_for(&k, GatheredSlot::A);
    shrink(&mut l);
    l.gather_reach.insert(k.bindings[1].get(), reach);
    let ops = emitted_with(&k, &l).expect(
        "a repeat gather admits under a declared source-extent reach: the values name rows of \
         the placed source",
    );
    assert_eq!(
        ops.len(),
        3,
        "the repeat gather emits the same two copy legs and the contraction -- got {:?}",
        ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
    );
    for o in &ops {
        assert!(
            !o.kv_gathered,
            "a SOURCE-EXTENT reach's op carries no gathered bit -- the family, not the presence, \
             is the fact: {:?}",
            o.op_name
        );
    }
}
