// SPDX-License-Identifier: Apache-2.0

//! ⭐ A FLOAT8 PAGE LANDS WIDENED BY ITS OWN GATHER — a gathered page read through its one
//! `arith.extf` to f16 is ONE device op: `fp8todl16` whose input is the pool read through the index,
//! as torch-spyre lowers a pointwise op over an indexed read (`spyre_kernel.py:1527-1553`); dxp's
//! scheduler makes the op's one paged input's transfer indirect whatever the op computes
//! (`L3DlOpsScheduler.cpp` `convertTransferDirectToIndirect`). Its fixtures:
//!
//! * the match: the float8 read lowers to the 16-bit read's descriptors, each page gather leg one
//!   `fp8todl16` at the 16-bit gather's own extents and division, its input the paged value;
//! * the near-misses, refused by name: a float8 page read with no widen, and one widened past f16 —
//!   the chain does not close, and the walk never reads float8 bytes as its 16-bit table;
//! * the neighbour it must not steal from: the 16-bit page's gather stays an `identity`.
//!
//! Every program is built from the op kinds the walk really reads, over a cache allocated
//! row-outermost at the page's own format exactly as the splice allocates the paged KV
//! (`allocate_row_outermost`). The `zz_` prefix keeps this door at the END of the crate's test list.

use ktir_core::affine::{AffineExpr, AffineMap};
use ktir_core::arena::Arena;
use ktir_core::attrkey::AttrKey;
use ktir_core::dtypes::DType;
use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
use ktir_core::irtype::IrType;
use ktir_core::opkind::OpKind;
use ktir_superdsc::collective::TpWorld;
use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::emit::whole_function::lower_function;
use ktir_superdsc::ktir_node::{BufferId, KtirNode, Program};
use ktir_superdsc::place::{PlaceId, act_name};
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
use ktir_superdsc::superdsc_opspec::{DataFormat, Fp8, Fp16, OpFunc};

/// One page's slab: 8192 elements — read as one `[64, 128]` block, whose 128-wide row is one whole
/// float8 stick and two 16-bit ones.
const SLAB: i64 = 8192;

/// The page's one row width.
const W: i64 = 128;

/// What the contraction reads of the gathered page.
#[derive(Clone, Copy)]
enum Read {
    /// The page as it is: a 16-bit page's read, or a float8 page read with no widen.
    Direct,
    /// The page's one `arith.extf` to this type.
    Widened(DType),
}

/// One page of a `page`-typed cache, gathered and read as the `[64, 128]` frame by an `[8, 64]`
/// activation's contraction, the result stored.
fn program(page: DType, read: Read) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let (entries, m) = (1i64, 8i64);
    let t_rows = entries * SLAB / W;
    let typed = |op: Operation<'static>, r: i64, c: i64, dt: DType| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(dt))
    };
    let shape = |op: Operation<'static>, r: i64, c: i64| typed(op, r, c, DType::F16);
    let ty = |r: i64, c: i64, elem: DType| {
        Some(IrType::Tensor {
            dims: a.ints(vec![r, c]),
            elem,
        })
    };

    // Parameters: %0 = A, %1 = the cache, %2 = ids, %3 = out.
    let (p_a, p_cache, p_ids, p_out) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3));
    let zero = Ssa(4);
    let (v_a, t_a, val_a) = (Ssa(5), Ssa(6), Ssa(7));
    let (v_c, v_i) = (Ssa(8), Ssa(9));
    let (ind, load, reshaped) = (Ssa(10), Ssa(11), Ssa(12));
    let (v_o, t_o, outs, mm) = (Ssa(13), Ssa(14), Ssa(15), Ssa(16));
    let widened = Ssa(17);

    // The `Sym(1) + Dim(0)` subscript `gather_subscripts` builds, anchored at 0.
    let sub = a.expr(AffineExpr::Add(
        a.expr(AffineExpr::Sym(1)),
        a.expr(AffineExpr::Dim(0)),
    ));

    let mut ops = vec![
        Operation::new(a, Some(zero), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(0),
        ),
        // A: [M, T].
        shape(
            Operation::new(a, Some(v_a), OpKind::KtdpConstructMemoryView, &[p_a]),
            m,
            t_rows,
        ),
        shape(
            Operation::new(
                a,
                Some(t_a),
                OpKind::KtdpConstructAccessTile,
                &[v_a, zero, zero],
            ),
            m,
            t_rows,
        ),
        shape(
            Operation::new(a, Some(val_a), OpKind::KtdpLoad, &[t_a]),
            m,
            t_rows,
        ),
        // The cache: [8, SLAB] at the page's format — eight pages of one row each.
        typed(
            Operation::new(a, Some(v_c), OpKind::KtdpConstructMemoryView, &[p_cache]),
            8,
            SLAB,
            page,
        ),
        // ids: [entries] i32.
        Operation::new(a, Some(v_i), OpKind::KtdpConstructMemoryView, &[p_ids])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![entries])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::I32)),
        // The gather's tile: `entries` pages of SLAB elements, anchored at 0.
        typed(
            Operation::new(
                a,
                Some(ind),
                OpKind::KtdpConstructIndirectAccessTile,
                &[v_c, v_i],
            ),
            entries,
            SLAB,
            page,
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
        ),
        {
            let mut op = typed(
                Operation::new(a, Some(load), OpKind::KtdpLoad, &[ind]),
                entries,
                SLAB,
                page,
            );
            op.result_type = ty(entries, SLAB, page);
            op
        },
        // The frame: the gathered `[entries, SLAB]` read as `[T, w]`.
        {
            let mut op = typed(
                Operation::new(a, Some(reshaped), OpKind::TensorReshape, &[load]),
                t_rows,
                W,
                page,
            );
            op.result_type = ty(t_rows, W, page);
            op
        },
    ];
    let rhs = match read {
        Read::Direct => reshaped,
        Read::Widened(to) => {
            let mut op = typed(
                Operation::new(a, Some(widened), OpKind::ArithExtf, &[reshaped]),
                t_rows,
                W,
                to,
            );
            op.result_type = ty(t_rows, W, to);
            ops.push(op);
            widened
        }
    };
    ops.extend([
        // OUT: [M, w].
        shape(
            Operation::new(a, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
            m,
            W,
        ),
        shape(
            Operation::new(
                a,
                Some(t_o),
                OpKind::KtdpConstructAccessTile,
                &[v_o, zero, zero],
            ),
            m,
            W,
        ),
        shape(
            Operation::new(a, Some(outs), OpKind::TensorEmpty, &[]),
            m,
            W,
        ),
        {
            let mut op = shape(
                Operation::new(a, Some(mm), OpKind::LinalgMatmul, &[val_a, rhs, outs]),
                m,
                W,
            );
            op.result_type = ty(m, W, DType::F16);
            op
        },
        Operation::new(a, None, OpKind::KtdpStore, &[mm, t_o]),
    ]);

    KtirNode {
        func: IRFunction {
            name: "float8_page_read",
            arguments: a.args(vec![
                (p_a, IrType::Index),
                (p_cache, IrType::Index),
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
        answer_rows: None,
        minted: Default::default(),
        lx_plan: Default::default(),
    }
}

/// The fully-placed layout: the four parameters at non-zero offsets, the cache allocated
/// row-outermost at the page's format `P`, as the splice allocates the paged KV.
fn layout_for<P: DataFormat>(k: &KtirNode) -> BundleLayout {
    let mut l = BundleLayout::default();
    for (i, b) in k.bindings.iter().enumerate() {
        let tid = b.get();
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        // Bytes: A [M, T] f16, cache [8, SLAB] at `P`, ids [entries] i32, OUT [M, w] f16.
        let bytes: u64 = match i {
            0 => 8 * 64 * 2,
            1 => 8 * SLAB as u64 * u64::from(P::WORD_LENGTH),
            2 => 4 * 4,
            _ => 8 * W as u64 * 2,
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
    l.allocate_row_outermost::<P>(k.bindings[1].get())
        .expect("the cache allocates row-outermost");
    l
}

fn emitted<P: DataFormat>(k: &KtirNode) -> Result<Vec<EmittedOp<TpWorld<1>>>, String> {
    let layout = layout_for::<P>(k);
    let mut sid = 0i64;
    lower_function(
        k,
        Some(&layout),
        &mut sid,
        None,
        &mut ktir_superdsc::emit::ktir_matmul_fp8::QuantizedActivations::default(),
    )
    .map_err(|e| e.message)
}

/// An op's descriptor, as JSON.
fn json(op: &EmittedOp<TpWorld<1>>) -> serde_json::Value {
    serde_json::to_value(op.op.as_ref().expect("a descriptor")).expect("json")
}

/// The op's one dsc.
fn dsc(op: &EmittedOp<TpWorld<1>>) -> serde_json::Value {
    json(op)["dscs_"][0]
        .as_object()
        .and_then(|m| m.values().next())
        .expect("one dsc")
        .clone()
}

/// An op's kind: its name without the buffer number it ends in.
fn kind(op: &EmittedOp<TpWorld<1>>) -> &str {
    op.op_name.trim_end_matches(|c: char| c.is_ascii_digit())
}

#[test]
fn a_float8_page_read_through_its_one_widen_lands_by_one_fp8todl16_a_leg() {
    let at16 = emitted::<Fp16>(&program(DType::F16, Read::Direct))
        .unwrap_or_else(|e| panic!("the 16-bit read lowers: {e}"));
    let at8 = emitted::<Fp8>(&program(DType::Fp8E4m3, Read::Widened(DType::F16)))
        .unwrap_or_else(|e| panic!("the float8 read lowers: {e}"));
    assert_eq!(
        at8.iter().map(kind).collect::<Vec<_>>(),
        at16.iter().map(kind).collect::<Vec<_>>(),
        "the float8 read is the 16-bit read, op for op — no float8 local, no separate widen"
    );
    let legs: Vec<_> = at8
        .iter()
        .zip(&at16)
        .filter(|(o, _)| o.op_name.starts_with("gather"))
        .collect();
    assert!(!legs.is_empty(), "the page is gathered");
    for (o8, o16) in legs {
        let (d8, d16) = (dsc(o8), dsc(o16));
        assert_eq!(
            d8["computeOp_"][0]["opFuncName"],
            OpFunc::Fp8ToDl16.name(),
            "{}: the float8 page's leg is the widen",
            o8.op_name
        );
        assert_eq!(
            d16["computeOp_"][0]["opFuncName"],
            OpFunc::Identity.name(),
            "{}: the 16-bit page's leg stays a copy",
            o16.op_name
        );
        assert_eq!(
            d8["N_"], d16["N_"],
            "the leg's extents are the 16-bit gather's"
        );
        assert_eq!(
            json(o8)["numWkSlicesPerDim_"],
            json(o16)["numWkSlicesPerDim_"],
            "and its division"
        );
        let value = d8["scheduleTree_"]
            .as_array()
            .expect("a tree")
            .iter()
            .find(|n| n["nodeType_"] == "allocate")
            .expect("an allocate")["indirectAllocType_"]
            .clone();
        assert_eq!(
            value,
            serde_json::json!("value_tensor"),
            "the widen's input is the pool read through the index"
        );
    }
}

/// The near-misses: the chain does not close, so no fold — the page arm refuses it by name.
#[test]
fn a_float8_page_read_with_no_widen_or_past_f16_is_refused_by_name() {
    for read in [Read::Direct, Read::Widened(DType::F32)] {
        let Err(err) = emitted::<Fp8>(&program(DType::Fp8E4m3, read)) else {
            panic!("a float8 page not read through one widen to f16 does not lower");
        };
        assert!(
            err.contains("is read through its widen to f16 alone"),
            "refused by name: {err}"
        );
    }
}
