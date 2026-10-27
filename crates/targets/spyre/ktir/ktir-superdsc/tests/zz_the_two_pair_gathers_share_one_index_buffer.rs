// SPDX-License-Identifier: Apache-2.0

//! ⭐⭐⭐ THE TWO-PAIR GATHER — the batched paged-decode shape no single-pair
//! reading admits.
//!
//! The sibling file (`zz_the_two_trip_gathers_join_by_load_identity.rs`) covers
//! the SAME-PAIR multi-tile shape: an unrolled sweep's trips, every tile over
//! the same two parameters. The batched paged-decode kernel
//! (`crates/targets/spyre/kernels/batched_decode.py`) states a DIFFERENT
//! multi-tile shape: TWO tables — Kᵀ and V — gathered through ONE ids buffer,
//! each gathered load feeding its own contraction. That program spans two
//! (index, value) pairs in one function, which `gathers_of` used to refuse
//! ("gathers over different buffers need one node per gather").
//!
//! The relaxation's safety argument, pinned here: the one-index-per-descriptor
//! law is a PER-OP fact, and the walk joins each op to its own tile by the
//! identity of the gathered LOAD. So the program below must lower with
//! each contraction's materialized copy gathering from ITS OWN table —
//! proven by address, because the descriptor names operands only positionally
//! and two tables at two addresses are the discriminator a name cannot be.
//!
//! Every program here is built from the op kinds the walk really reads, so the
//! door sees a real program and not a stub.

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

/// The two-pair program at the sibling file's own geometry (`M=8, K=64,
/// HEAD=64`, 64-entry gathers): TWO tables gathered through ONE ids buffer,
/// each gathered load the B of its own contraction, the two contractions
/// summed and stored — the batched paged-decode kernel's gather structure at
/// the smallest shape that lowers.
fn two_pair_program() -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };

    // Parameters: %0 = P, %1 = Kᵀ table, %2 = V table, %3 = ids, %4 = out.
    let (p_p, p_kt, p_v, p_ids, p_out) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3), Ssa(4));
    let zero = Ssa(5);
    // The A chain (P's plain load), both tables' views, the ids view.
    let (v_a, t_a, val_a) = (Ssa(6), Ssa(7), Ssa(8));
    let (v_kt, v_v, v_i) = (Ssa(9), Ssa(10), Ssa(11));
    // Each table's indirect tile and its load.
    let (ind_kt, load_kt, ind_v, load_v) = (Ssa(12), Ssa(13), Ssa(14), Ssa(15));
    // The output view/tile, the zero outs, the two matmuls and their addf.
    let (v_o, t_o, outs, mm_kt, mm_v, addf) =
        (Ssa(16), Ssa(17), Ssa(18), Ssa(19), Ssa(20), Ssa(21));

    // The `Sym(1) + Dim(0)` subscript `gather_subscripts` builds. Both tiles
    // anchor at entry 0 (`intermediate_vars[1]` is the shared `zero`), which is
    // the kernel's own shape: one chunk's Kᵀ and V gathers read the SAME ids
    // window. Interned before the ops vec because its children are borrows.
    let sub = a.expr(AffineExpr::Add(
        a.expr(AffineExpr::Sym(1)),
        a.expr(AffineExpr::Dim(0)),
    ));

    // One table's indirect tile over the ids view, anchored at entry 0.
    let indirect = |result: Ssa, table_view: Ssa| {
        shape(
            Operation::new(
                a,
                Some(result),
                OpKind::KtdpConstructIndirectAccessTile,
                &[table_view, v_i],
            ),
            64,
            64,
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
                    exprs: std::slice::from_ref(sub),
                },
                AffineMap {
                    num_dims: 2,
                    num_syms: 2,
                    exprs: a.exprs(vec![AffineExpr::Dim(1)]),
                },
            ])),
        )
    };

    // The matmuls' results are UNSTORED intermediates, so each states a 2-D
    // tensor type — the footprint the walk's minted buffer is sized from.
    let ty = |r: i64, c: i64| {
        Some(IrType::Tensor {
            dims: a.ints(vec![r, c]),
            elem: DType::F16,
        })
    };

    let ops = vec![
        Operation::new(a, Some(zero), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(0),
        ),
        // P: [8, 64] — the shared activation, both contractions' A.
        shape(
            Operation::new(a, Some(v_a), OpKind::KtdpConstructMemoryView, &[p_p]),
            8,
            64,
        ),
        shape(
            Operation::new(
                a,
                Some(t_a),
                OpKind::KtdpConstructAccessTile,
                &[v_a, zero, zero],
            ),
            8,
            64,
        ),
        shape(
            Operation::new(a, Some(val_a), OpKind::KtdpLoad, &[t_a]),
            8,
            64,
        ),
        // Kᵀ: [128, 64] — the FIRST gathered table.
        shape(
            Operation::new(a, Some(v_kt), OpKind::KtdpConstructMemoryView, &[p_kt]),
            128,
            64,
        ),
        // V: [128, 64] — the SECOND gathered table, a different parameter.
        shape(
            Operation::new(a, Some(v_v), OpKind::KtdpConstructMemoryView, &[p_v]),
            128,
            64,
        ),
        // ids: [64] i32 — the ONE index buffer both gathers read.
        Operation::new(a, Some(v_i), OpKind::KtdpConstructMemoryView, &[p_ids])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![64])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::I32)),
        // Each table's tile and load. The gathered loads state their OWN types too — the
        // flat-law check's second authority (every SSA in a real TTIR function carries one).
        indirect(ind_kt, v_kt),
        {
            let mut op = shape(
                Operation::new(a, Some(load_kt), OpKind::KtdpLoad, &[ind_kt]),
                64,
                64,
            );
            op.result_type = ty(64, 64);
            op
        },
        indirect(ind_v, v_v),
        {
            let mut op = shape(
                Operation::new(a, Some(load_v), OpKind::KtdpLoad, &[ind_v]),
                64,
                64,
            );
            op.result_type = ty(64, 64);
            op
        },
        // OUT: [8, 64].
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
        shape(
            Operation::new(a, Some(outs), OpKind::TensorEmpty, &[]),
            8,
            64,
        ),
        // The two contractions: each a [8,64]·[64,64] matmul over ITS OWN
        // gathered B, summed.
        {
            let mut op = shape(
                Operation::new(
                    a,
                    Some(mm_kt),
                    OpKind::LinalgMatmul,
                    &[val_a, load_kt, outs],
                ),
                8,
                64,
            );
            op.result_type = ty(8, 64);
            op
        },
        {
            let mut op = shape(
                Operation::new(a, Some(mm_v), OpKind::LinalgMatmul, &[val_a, load_v, outs]),
                8,
                64,
            );
            op.result_type = ty(8, 64);
            op
        },
        shape(
            Operation::new(a, Some(addf), OpKind::ArithAddf, &[mm_kt, mm_v]),
            8,
            64,
        ),
        Operation::new(a, None, OpKind::KtdpStore, &[addf, t_o]),
    ];

    KtirNode {
        func: IRFunction {
            name: "batched_two_table",
            arguments: a.args(vec![
                (p_p, IrType::Index),
                (p_kt, IrType::Index),
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

/// The fully-placed layout for the five parameters — the sibling files' own
/// placement law: every table at a non-zero offset so an address assertion
/// discriminates WHICH buffer an op reads.
fn layout_for(k: &KtirNode) -> BundleLayout {
    let mut l = BundleLayout::default();
    for (i, b) in k.bindings.iter().enumerate() {
        let tid = b.get();
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        // Bytes: P [8,64] f16, Kᵀ [128,64] f16, V [128,64] f16, ids [64] i32, OUT [8,64] f16.
        let bytes: u64 = match i {
            0 => 8 * 64 * 2,
            1 => 128 * 64 * 2,
            2 => 128 * 64 * 2,
            3 => 64 * 4,
            _ => 8 * 64 * 2,
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
                cover: ktir_superdsc::placement::BindCover::Full,
            },
        );
    }
    l
}

fn emitted(k: &KtirNode) -> Result<Vec<ktir_superdsc::emit::EmittedOp>, String> {
    let layout = layout_for(k);
    let mut sid = 0i64;
    lower_function(k, Some(&layout), &mut sid, None).map_err(|e| e.message)
}

/// The `index_tensor` alloc's run-base word of an emitted gathered op — `None`
/// when the op declares no index alloc. The sibling file's own reader.
fn index_run_base(op: &ktir_superdsc::emit::EmittedOp) -> Option<u64> {
    let dsc = op.dsc();
    let dsc = dsc.dscs_.first()?.values().next()?;
    let node = dsc
        .scheduleTree_
        .iter()
        .find(|n| n.nodeType_ == "allocate" && n.indirectAllocType_ == "index_tensor")?;
    node.startAddressCoreCorelet_
        .data_
        .get("[0, 0, 0]")?
        .parse()
        .ok()
}

/// The `value_tensor` alloc's base address in a gathered descriptor — the
/// gather's SOURCE buffer. The rung-4 file's own reader.
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

/// The device address the `dsType_`-tagged operand of `op` starts at. The
/// rung-4 file's own reader.
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

/// Kᵀ (parameter 1) and V (parameter 2) under [`layout_for`]: seg3 + their
/// parameter slots. The only buffers in the test at these addresses.
const KT_ADDR: u64 = 0xC_0000_0000 + 0x10000;
const V_ADDR: u64 = 0xC_0000_0000 + 0x20000;
/// The ONE ids buffer (parameter 3), where every index alloc must run.
const IDS_ADDR: u64 = 0xC_0000_0000 + 0x30000;

/// ⭐⭐⭐ THE POSITIVE: the two-pair program LOWERS, and each contraction's
/// materialized copy gathers from ITS OWN table.
///
/// The walk emits in program order — the Kᵀ contraction's copy legs first, then
/// that contraction, then the V pair — so the first two gathered legs must carry
/// Kᵀ's address and the last two V's. A join that matched tiles by enumeration
/// position alone cannot fail this test; a join that hands the SECOND
/// contraction the FIRST table's gather — the half-materialization the pair
/// refusal used to guard against — emits V's legs at Kᵀ's address, and the
/// address is the discriminator a positional operand name cannot be.
#[test]
fn two_tables_through_one_ids_each_matmul_gathers_its_own() {
    let ops = emitted(&two_pair_program())
        .expect("the two-pair program lowers through the whole-function door");
    let copies: Vec<&ktir_superdsc::emit::EmittedOp> =
        ops.iter().filter(|o| index_run_base(o).is_some()).collect();
    assert_eq!(
        copies.len(),
        4,
        "two gathers of 64 entries = four gathered copy legs (one per 32-entry index \
         stick) -- got {} out of {} ops: {:?}",
        copies.len(),
        ops.len(),
        ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
    );
    // ⭐ THE PAIR FACT, BY ADDRESS: the first two legs gather Kᵀ, the last two V.
    for (i, copy) in copies.iter().enumerate() {
        let want = if i < 2 { KT_ADDR } else { V_ADDR };
        assert_eq!(
            value_tensor_address(copy),
            Some(want),
            "copy leg {i} gathers from {} -- each contraction's copy reads its OWN \
             table: {:?}",
            if i < 2 { "Kᵀ" } else { "V" },
            copy.op_name
        );
    }
    // ⭐ AND THE ONE IDS BUFFER: every leg's index alloc runs at the shared ids
    // address — two tables, one index buffer, four readers of the same one. Each
    // gather's SECOND leg steps one 32-entry index stick (128 B) past the first,
    // exactly as the two-trip file's own arithmetic states.
    for (i, copy) in copies.iter().enumerate() {
        assert_eq!(
            index_run_base(copy),
            Some(IDS_ADDR + (i as u64 % 2) * 128),
            "every copy leg's index alloc runs at the ONE ids buffer (leg {i} at \
             stick {}): {:?}",
            i % 2,
            copy.op_name
        );
    }
    // ⭐ EACH CONTRACTION READS ITS OWN MINTED INTERMEDIATE — not either table
    // (the direct-read trap), and not the SAME intermediate (the
    // half-materialization: both contractions onto one gather would mint one).
    let matmuls: Vec<&ktir_superdsc::emit::EmittedOp> = ops
        .iter()
        .filter(|o| {
            o.dsc()
                .dscs_
                .first()
                .and_then(|m| m.values().next())
                .is_some_and(|d| d.computeOp_.iter().any(|c| c.opFuncName == "batchmatmul"))
        })
        .collect();
    assert_eq!(
        matmuls.len(),
        2,
        "the two contractions emit -- got {} batchmatmuls out of {} ops: {:?}",
        matmuls.len(),
        ops.len(),
        ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
    );
    let b_addrs: Vec<Option<u64>> = matmuls
        .iter()
        .map(|mm| alloc_address(mm, "KERNEL"))
        .collect();
    for (i, b) in b_addrs.iter().enumerate() {
        assert!(b.is_some(), "contraction {i} declares a KERNEL (B) operand");
        assert_ne!(
            *b,
            Some(KT_ADDR),
            "contraction {i} reads Kᵀ directly at its base -- the direct-read trap"
        );
        assert_ne!(
            *b,
            Some(V_ADDR),
            "contraction {i} reads V directly at its base -- the direct-read trap"
        );
    }
    assert_ne!(
        b_addrs[0], b_addrs[1],
        "the two contractions read TWO DISTINCT minted intermediates -- the same \
         address means one gather's copy stood for both tables"
    );
}
