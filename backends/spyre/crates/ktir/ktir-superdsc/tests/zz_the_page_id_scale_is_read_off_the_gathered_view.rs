// SPDX-License-Identifier: Apache-2.0

//! ⭐ THE PAGE ID'S SCALE — minted off the VALUE VIEW's own `Strides`, at the one
//! place the program states them.
//!
//! The device's gather addresses entry `i` at `i · skip_addr`, and `skip_addr` is
//! DERIVED from the value alloc's per-dim capacities — the entry's own size, always
//! (`GatherIndexConversion.cpp:393-424`, `dsc2.cpp:3900-3908`). A view whose entries
//! are spaced WIDER than their own extent — a unit-major pool, where one growth unit
//! holds both planes — therefore carries its spacing in the INDEX: the id the host
//! stages is the scheduler page id times `stride[0] ÷ entry`. `page_index_scale` is
//! that one reading, and these fixtures pin its three laws: a strided view mints
//! `stride/entry`; a dense view mints 1; two gathers through one page table that
//! disagree are refused by name.
//!
//! Every program is built from the op kinds `gathers_of` really reads, so the mint
//! sees a real gather and not a stub.

use ktir_core::affine::{AffineExpr, AffineMap};
use ktir_core::arena::Arena;
use ktir_core::attrkey::AttrKey;
use ktir_core::dtypes::DType;
use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
use ktir_core::irtype::IrType;
use ktir_core::opkind::OpKind;
use ktir_superdsc::emit::whole_function::page_index_scale;
use ktir_superdsc::ktir_node::{BufferId, KtirNode, Program};

/// One gathered program over ONE page table: 4 page ids gathering 64-element (one
/// stick) slabs of f16, the value view stating `strides = [stride0, 1]` — the
/// compiled form of the kernel's own
/// `make_tensor_descriptor(..., strides=[stride0, 1])`. `index_binding` names which
/// parameter the page table is, so a fixture can point the mint at a buffer no
/// gather reads.
fn gathered_program(stride0: Option<i64>) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };

    // Parameters: %0 = the gathered table, %1 = the page table, %2 = a third buffer
    // no gather reads (the empty-set fixture's own page table).
    let (p_val, p_ids, p_other) = (Ssa(0), Ssa(1), Ssa(2));
    let (v_val, v_ids, ind, load, zero) = (Ssa(3), Ssa(4), Ssa(5), Ssa(6), Ssa(7));

    // The page-table subscript's own form: a bare `Dim(0)` — the run of entries
    // starts at index word 0. Interned before the ops vec because its children are
    // borrows.
    let dim0 = a.expr(AffineExpr::Dim(0));
    let dim1 = a.expr(AffineExpr::Dim(1));

    let ops = vec![
        Operation::new(a, Some(zero), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(0),
        ),
        // The gathered table's view: [4, 64] f16, its entries spaced `stride0`
        // elements apart — the unit-major pool's own descriptor. `None` states no
        // strides at all: the refusal fixture's own view.
        match stride0 {
            Some(stride0) => shape(
                Operation::new(a, Some(v_val), OpKind::KtdpConstructMemoryView, &[p_val]),
                4,
                64,
            )
            .with_attr(
                a,
                AttrKey::Strides,
                Attr::IntList(a.ints(vec![stride0, 1])),
            ),
            None => shape(
                Operation::new(a, Some(v_val), OpKind::KtdpConstructMemoryView, &[p_val]),
                4,
                64,
            ),
        },
        // The page table's view: [4] i32 — raw scheduler page ids.
        Operation::new(a, Some(v_ids), OpKind::KtdpConstructMemoryView, &[p_ids])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![4])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::I32)),
        // The gather: one tile over the two views, its load the one reader.
        shape(
            Operation::new(
                a,
                Some(ind),
                OpKind::KtdpConstructIndirectAccessTile,
                &[v_val, v_ids],
            ),
            4,
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
                    exprs: std::slice::from_ref(dim0),
                },
                AffineMap {
                    num_dims: 2,
                    num_syms: 2,
                    exprs: std::slice::from_ref(dim1),
                },
            ])),
        ),
        shape(
            Operation::new(a, Some(load), OpKind::KtdpLoad, &[ind]),
            4,
            64,
        ),
    ];

    let bindings = vec![BufferId::new(401), BufferId::new(402), BufferId::new(403)];
    KtirNode {
        func: IRFunction {
            name: "page_gather_s0",
            arguments: a.args(vec![
                (p_val, IrType::Index),
                (p_ids, IrType::Index),
                (p_other, IrType::Index),
            ]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        },
        program: Program::Matmul,
        bindings,
        mask: None,
        node_out_tid: None,
        answer_rows: None,
        minted: Default::default(),
        lx_plan: Default::default(),
    }
}

/// ⭐ FIXTURE (i): a strided view mints `stride[0] ÷ entry`. The unit-major pool's
/// own geometry — 64-element entries spaced 128 apart — mints 2, and a wider
/// stride the same law mints more of.
#[test]
fn a_strided_view_mints_stride_over_entry() {
    let k = gathered_program(Some(128));
    let page_table = k.bindings[1].get();
    assert_eq!(
        page_index_scale(&k, page_table).expect("the strided view's scale is minted"),
        2,
        "entries of 64 elements spaced 128 apart: page id p addresses p·2 entries"
    );
    let k = gathered_program(Some(256));
    let page_table = k.bindings[1].get();
    assert_eq!(
        page_index_scale(&k, page_table).expect("the wider view's scale is minted"),
        4,
        "the law is over the view, not over one geometry: 256/64 mints 4"
    );
}

/// ⭐ FIXTURE (ii): a DENSE view — entries packed, `stride[0] == entry` — mints 1.
/// The staged ids, the emitted descriptors and the bytes on the wire are unchanged
/// for every dense-gathered program.
#[test]
fn a_dense_view_mints_one() {
    let k = gathered_program(Some(64));
    let page_table = k.bindings[1].get();
    assert_eq!(
        page_index_scale(&k, page_table).expect("the dense view's scale is minted"),
        1,
        "a packed view's ids are staged raw: scale 1, byte-identical wire"
    );
}

/// ⭐ FIXTURE (iii), the empty set: a page id no gather reads AS an index through
/// the page table reaches only tables staged in page units by their own laws, so
/// the scale is 1 — not an error, not a branch.
#[test]
fn a_page_table_no_gather_reads_mints_one() {
    let k = gathered_program(Some(128));
    let other = k.bindings[2].get();
    assert_eq!(
        page_index_scale(&k, other).expect("the unread buffer's scale is minted"),
        1,
        "no gather through this buffer: its ids carry no stride of this view's"
    );
}

/// ⛔ A view that states NO strides names no scale — the host cannot stage its ids,
/// so the mint refuses by name rather than guessing packed.
#[test]
fn a_view_without_strides_is_refused_by_name() {
    let k = gathered_program(None);
    let page_table = k.bindings[1].get();
    let e = page_index_scale(&k, page_table)
        .expect_err("a view with no strides names no id scale");
    assert!(
        e.message.contains("states no strides"),
        "the refusal names the missing strides, got: {}",
        e.message
    );
}

/// ⛔ A stride that is not a whole number of entries stages no integer id — refused
/// by name at the one statement.
#[test]
fn a_stride_between_entries_is_refused_by_name() {
    let k = gathered_program(Some(65));
    let page_table = k.bindings[1].get();
    let e = page_index_scale(&k, page_table)
        .expect_err("a stride of 65 over 64-element entries is not a whole scale");
    assert!(
        e.message.contains("not a whole number of entries"),
        "the refusal names the fractional scale, got: {}",
        e.message
    );
}

/// ⛔ TWO gathers through ONE page table that state different scales read two
/// layouts through one buffer — refused by name at the one statement, rather than
/// staged for one of them.
#[test]
fn two_gathers_with_different_scales_are_refused_by_name() {
    let k = gathered_program(Some(128));
    // A second gather through the SAME page table over a view of HALF the entry
    // size — same 128-element stride, 32-element entries — mints 4 where the first
    // mints 2.
    let a: &'static Arena = Arena::global();
    let mut k2 = k;
    let (v_half, ind2, load2) = (Ssa(8), Ssa(9), Ssa(10));
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };
    let dim0 = a.expr(AffineExpr::Dim(0));
    let dim1 = a.expr(AffineExpr::Dim(1));
    let zero = Ssa(7);
    let extra = vec![
        shape(
            Operation::new(
                a,
                Some(v_half),
                OpKind::KtdpConstructMemoryView,
                &[Ssa(0)],
            ),
            4,
            32,
        )
        .with_attr(
            a,
            AttrKey::Strides,
            Attr::IntList(a.ints(vec![128, 1])),
        ),
        shape(
            Operation::new(
                a,
                Some(ind2),
                OpKind::KtdpConstructIndirectAccessTile,
                &[v_half, Ssa(4)],
            ),
            4,
            32,
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
                    exprs: std::slice::from_ref(dim0),
                },
                AffineMap {
                    num_dims: 2,
                    num_syms: 2,
                    exprs: std::slice::from_ref(dim1),
                },
            ])),
        ),
        shape(
            Operation::new(a, Some(load2), OpKind::KtdpLoad, &[ind2]),
            4,
            32,
        ),
    ];
    let mut ops = k2.func.operations.to_vec();
    ops.extend(extra);
    k2.func.operations = a.ops(ops);
    let page_table = k2.bindings[1].get();
    let e = page_index_scale(&k2, page_table)
        .expect_err("two gathers through one page table cannot state two scales");
    assert!(
        e.message.contains("different id scales"),
        "the refusal names the disagreement, got: {}",
        e.message
    );
}
