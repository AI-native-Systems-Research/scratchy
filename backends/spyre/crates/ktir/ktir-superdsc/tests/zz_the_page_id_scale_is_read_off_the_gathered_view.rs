// SPDX-License-Identifier: Apache-2.0

//! ⭐ THE PAGE ID'S SCALE — minted off EVERY op that reads the page table, at the one
//! place the program states them.
//!
//! The device's gather addresses entry `i` at `i · skip_addr`, and `skip_addr` is
//! DERIVED from the value alloc's per-dim capacities — the entry's own size, always
//! (`GatherIndexConversion.cpp:393-424`, `dsc2.cpp:3900-3908`). A view whose entries
//! are spaced WIDER than their own extent — a unit-major pool, where one growth unit
//! holds both planes — therefore carries its spacing in the INDEX: the id the host
//! stages is the scheduler page id times `stride[0] ÷ entry`. `page_index_scale` is
//! that one reading, and these fixtures pin its laws on BOTH arms: a strided value
//! view mints `stride/entry`; a dense value view mints 1; a page id LOADED out of
//! the table (the chained index, the decode kernel's own spelling) mints 1 off its
//! run table's PACKEDNESS, and a run table spaced wider is refused by name; two
//! readers of one page table that disagree are refused by name; and a page table
//! read by NOTHING answers `None` — a fact the fold over the bundle turns into a
//! bake error, never a default.
//!
//! Every program is built from the op kinds `gathers_of` and `chained_index_of`
//! really read, so the mint sees a real gather and not a stub.

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
        Some(2),
        "entries of 64 elements spaced 128 apart: page id p addresses p·2 entries"
    );
    let k = gathered_program(Some(256));
    let page_table = k.bindings[1].get();
    assert_eq!(
        page_index_scale(&k, page_table).expect("the wider view's scale is minted"),
        Some(4),
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
        Some(1),
        "a packed view's ids are staged raw: scale 1, byte-identical wire"
    );
}

/// ⭐ FIXTURE (iii), the unread table: a page table no op reads states no law at
/// all — `None`, which the fold over the bundle's programs turns into a bake error
/// naming the table. Never a silent default of 1.
#[test]
fn a_page_table_no_op_reads_answers_none() {
    let k = gathered_program(Some(128));
    let other = k.bindings[2].get();
    assert_eq!(
        page_index_scale(&k, other).expect("the unread buffer's answer is minted"),
        None,
        "no reader of this buffer: its ids carry no law, and the fold refuses the \
         bundle rather than defaulting"
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

/// ⭐ One CHAINED program — the sequence decode kernel's own spelling, built from the op kinds
/// `chained_index_of` really reads: the page id is LOADED out of the page table (a direct
/// access tile at a constant entry, its scalar extracted), scaled by the run table's word
/// pitch (`page · 8`), and consumed as the position where a gather's run of index words
/// starts inside the RUN TABLE. Parameters: `%0` = the gathered value table, `%1` = the run
/// table (`kv_row_pool`), `%2` = the page table.
///
/// `run_strides`/`run_shape` state the run table's own view — `[1]` over `[32]` is the
/// kernel's flat pool view (`[num_pages · num_kv_heads]` at element stride 1); anything else
/// is the refusal fixture's own spacing. `direct_scale` adds a SECOND gather indexed straight
/// through the page table over a value view of that entry scale, for the disagreement
/// fixture.
fn chained_program(
    run_strides: Option<Vec<i64>>,
    run_shape: Vec<i64>,
    direct_scale: Option<i64>,
) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };
    let ishape = |op: Operation<'static>, dims: Vec<i64>| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims)))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::I32))
    };

    // Parameters: %0 = gathered table, %1 = run table, %2 = page table.
    let (p_val, p_run, p_page) = (Ssa(0), Ssa(1), Ssa(2));
    let (v_k, v_run, v_page, zero, eight) = (Ssa(3), Ssa(4), Ssa(5), Ssa(6), Ssa(7));
    let (pt_tile, ld, ext, mul, cast, ind, gld) = (Ssa(8), Ssa(9), Ssa(10), Ssa(11), Ssa(12), Ssa(13), Ssa(14));

    let dim0 = a.expr(AffineExpr::Dim(0));
    let dim1 = a.expr(AffineExpr::Dim(1));

    let mut ops = vec![
        Operation::new(a, Some(zero), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(0),
        ),
        Operation::new(a, Some(eight), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(8),
        ),
        // The gathered table's view: [8, 64] f16, dense — the chained arm never reads it.
        shape(
            Operation::new(a, Some(v_k), OpKind::KtdpConstructMemoryView, &[p_val]),
            8,
            64,
        )
        .with_attr(a, AttrKey::Strides, Attr::IntList(a.ints(vec![64, 1]))),
        // The RUN TABLE's view — the fixture's own stated spacing.
        match run_strides {
            Some(s) => ishape(
                Operation::new(a, Some(v_run), OpKind::KtdpConstructMemoryView, &[p_run]),
                run_shape,
            )
            .with_attr(a, AttrKey::Strides, Attr::IntList(a.ints(s))),
            None => ishape(
                Operation::new(a, Some(v_run), OpKind::KtdpConstructMemoryView, &[p_run]),
                run_shape,
            ),
        },
        // The page table's view: [4] i32 — raw scheduler page ids.
        ishape(
            Operation::new(a, Some(v_page), OpKind::KtdpConstructMemoryView, &[p_page]),
            vec![4],
        )
        .with_attr(a, AttrKey::Strides, Attr::IntList(a.ints(vec![1]))),
        // The chained read: page = the scalar at entry 0 of the page table's view.
        ishape(
            Operation::new(a, Some(pt_tile), OpKind::KtdpConstructAccessTile, &[v_page, zero]),
            vec![1],
        ),
        ishape(
            Operation::new(a, Some(ld), OpKind::KtdpLoad, &[pt_tile]),
            vec![1],
        ),
        Operation::new(a, Some(ext), OpKind::TensorExtract, &[ld, zero]),
        // The run table's own word pitch: page · 8 (num_kv_heads).
        Operation::new(a, Some(mul), OpKind::ArithMuli, &[ext, eight]),
        Operation::new(a, Some(cast), OpKind::ArithIndexCast, &[mul]),
        // The gather: 8 rows of the value table, its run of index words starting at the
        // loaded position inside the run table.
        shape(
            Operation::new(
                a,
                Some(ind),
                OpKind::KtdpConstructIndirectAccessTile,
                &[v_k, v_run],
            ),
            8,
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
            Attr::Ssas(a.ssa(vec![cast, zero])),
        )
        .with_attr(
            a,
            AttrKey::DimSubs,
            Attr::AffineMapList(a.maps(vec![
                AffineMap {
                    num_dims: 2,
                    num_syms: 2,
                    exprs: std::slice::from_ref(a.expr(AffineExpr::Add(
                        a.expr(AffineExpr::Sym(0)),
                        dim0,
                    ))),
                },
                AffineMap {
                    num_dims: 2,
                    num_syms: 2,
                    exprs: std::slice::from_ref(a.expr(AffineExpr::Add(
                        a.expr(AffineExpr::Sym(1)),
                        dim1,
                    ))),
                },
            ])),
        ),
        shape(
            Operation::new(a, Some(gld), OpKind::KtdpLoad, &[ind]),
            8,
            64,
        ),
    ];
    // The disagreement fixture's own second reader: a gather indexed STRAIGHT through the
    // page table, over a value view of the stated entry scale.
    if let Some(scale) = direct_scale {
        let (v_direct, ind2, load2) = (Ssa(15), Ssa(16), Ssa(17));
        ops.extend([
            shape(
                Operation::new(a, Some(v_direct), OpKind::KtdpConstructMemoryView, &[p_val]),
                4,
                64,
            )
            .with_attr(
                a,
                AttrKey::Strides,
                Attr::IntList(a.ints(vec![64 * scale, 1])),
            ),
            shape(
                Operation::new(
                    a,
                    Some(ind2),
                    OpKind::KtdpConstructIndirectAccessTile,
                    &[v_direct, v_page],
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
                Operation::new(a, Some(load2), OpKind::KtdpLoad, &[ind2]),
                4,
                64,
            ),
        ]);
    }

    let bindings = vec![BufferId::new(401), BufferId::new(402), BufferId::new(403)];
    KtirNode {
        func: IRFunction {
            name: "chained_gather_s0",
            arguments: a.args(vec![
                (p_val, IrType::Index),
                (p_run, IrType::Index),
                (p_page, IrType::Index),
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

/// ⭐⭐ FIXTURE (iv), THE LOAD ARM — the decode kernel's own shape: the page id is read out of
/// the table and consumed as the position of one RUN of the gather's index words in the run
/// table. The host stages a run table ONE RUN PER POSITION, so the id is raw BY THAT LAW —
/// `Some(1)`, read off the run view's packedness, not an empty-set default. This is the
/// fixture that keeps a decode bundle's scale 1 a STATED fact.
#[test]
fn a_chained_run_table_mints_one_off_its_packedness() {
    // The kernel's flat pool view: [pages · num_kv_heads] at element stride 1 — packed.
    let k = chained_program(Some(vec![1]), vec![32], None);
    let page_table = k.bindings[2].get();
    assert_eq!(
        page_index_scale(&k, page_table).expect("the chained run table's scale is minted"),
        Some(1),
        "the run table is staged one run per position: the loaded page id is raw, by law"
    );
    // The 2-D spelling of the same packed law: [positions, width] at stride [width, 1].
    let k = chained_program(Some(vec![8, 1]), vec![4, 8], None);
    let page_table = k.bindings[2].get();
    assert_eq!(
        page_index_scale(&k, page_table).expect("the 2-D run table's scale is minted"),
        Some(1),
        "rows at their own width are packed whatever rank the view states"
    );
}

/// ⛔ A run table spaced WIDER than its own runs names a layout this lowering stages no table
/// for — the id would have to carry a scale the run copy's one-run-per-position entry cannot
/// express. Refused by name at the one statement, never silently scaled.
#[test]
fn a_run_table_spaced_wider_than_its_runs_is_refused_by_name() {
    let k = chained_program(Some(vec![16, 1]), vec![4, 8], None);
    let page_table = k.bindings[2].get();
    let e = page_index_scale(&k, page_table)
        .expect_err("runs of 8 spaced 16 apart are not a staged run table");
    assert!(
        e.message.contains("not packed at their own width"),
        "the refusal names the spacing, got: {}",
        e.message
    );
    // And a view that states NO strides names no packedness at all.
    let k = chained_program(None, vec![32], None);
    let page_table = k.bindings[2].get();
    let e = page_index_scale(&k, page_table)
        .expect_err("a run view with no strides names no packedness");
    assert!(
        e.message.contains("states no strides"),
        "the refusal names the missing strides, got: {}",
        e.message
    );
}

/// ⛔ ONE TABLE, ONE LAW, ACROSS THE ARMS: a program whose page table is read BOTH ways — a
/// chained run table minting 1 and a direct gather minting 2 — reads two layouts through one
/// buffer. Refused by name at the one statement.
#[test]
fn a_chained_reader_beside_a_disagreeing_gather_is_refused_by_name() {
    let k = chained_program(Some(vec![1]), vec![32], Some(2));
    let page_table = k.bindings[2].get();
    let e = page_index_scale(&k, page_table)
        .expect_err("one page table cannot carry a chained law and a gather law that disagree");
    assert!(
        e.message.contains("different id scales"),
        "the refusal names the disagreement, got: {}",
        e.message
    );
}
