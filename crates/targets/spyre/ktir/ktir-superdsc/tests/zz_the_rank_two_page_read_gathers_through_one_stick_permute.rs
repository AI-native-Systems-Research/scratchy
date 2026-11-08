// SPDX-License-Identifier: Apache-2.0

//! ⭐ A RANK-2 `[T, w]` READ OF SEVERAL GATHERED PLANES WIDER THAN ONE STICK — the frame the one
//! `index_select` gather lowers, not refuses.
//!
//! The card frame of a `[T, w]` tensor is ONE stick column over all `T` rows, while the gather
//! lands each entry's blocks stick-major one after another — the same bytes exactly when there is
//! one block or one stick. So:
//!
//! * SEVERAL entries, `w` wider than one stick: the entries land as one-row blocks — row-major
//!   `[T, w]`, the same bytes the op reads — and ONE whole stick permute restickifies them into
//!   the frame's own column. This is the restickify the consumer's reshape used to perform (the
//!   deleted `page_to_sticks` arm), stated at the landing that needs it.
//! * ONE entry (however wide), or `w` one stick wide (however many entries): the landing IS the
//!   frame and NO permute follows — the near-misses that must not steal the permute.
//!
//! Every program here is built from the op kinds the walk really reads, over a cache allocated
//! row-outermost exactly as the splice allocates the paged KV (`allocate_row_outermost`), so the
//! door sees a real program and not a stub.
//!
//! The `zz_` prefix keeps this door at the END of the crate's test list, after the splice's
//! golden legs — the same convention the other whole-function doors carry — so a failure here
//! reads as "the door" and not as the first thing anyone runs.

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

/// One page's slab: 8192 elements (128 sticks) — wide enough that `w` of 128 is two sticks and
/// the rank-2 frame's stick column is not the per-block landing, and that the consuming
/// contraction's K (the frame's rows) is whole sticks.
const SLAB: i64 = 8192;

/// The gathered program at `entries` pages read as the rank-2 `[T, w]` frame: an `[M, T]`
/// activation contracted against the gathered `[T, w]`, the result stored — the shape a batched
/// consumer of a rank-2 read takes.
fn program(entries: i64, w: i64, m: i64) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let t_rows = entries * SLAB / w;
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };

    // Parameters: %0 = A, %1 = the cache, %2 = ids, %3 = out.
    let (p_a, p_cache, p_ids, p_out) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3));
    let zero = Ssa(4);
    // A's view/tile/load, the cache and ids views, the indirect tile and its load.
    let (v_a, t_a, val_a) = (Ssa(5), Ssa(6), Ssa(7));
    let (v_c, v_i) = (Ssa(8), Ssa(9));
    let (ind, load, reshaped) = (Ssa(10), Ssa(11), Ssa(12));
    // The output view/tile, the empty out, the matmul, the store.
    let (v_o, t_o, outs, mm) = (Ssa(13), Ssa(14), Ssa(15), Ssa(16));

    // The `Sym(1) + Dim(0)` subscript `gather_subscripts` builds, anchored at 0.
    let sub = a.expr(AffineExpr::Add(
        a.expr(AffineExpr::Sym(1)),
        a.expr(AffineExpr::Dim(0)),
    ));

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
        // The cache: [8, SLAB] — eight pages of one row each, row-outermost in the layout below.
        shape(
            Operation::new(a, Some(v_c), OpKind::KtdpConstructMemoryView, &[p_cache]),
            8,
            SLAB,
        ),
        // ids: [entries] i32.
        Operation::new(a, Some(v_i), OpKind::KtdpConstructMemoryView, &[p_ids])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![entries])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::I32)),
        // The gather's tile: `entries` pages of SLAB elements, anchored at 0.
        shape(
            Operation::new(
                a,
                Some(ind),
                OpKind::KtdpConstructIndirectAccessTile,
                &[v_c, v_i],
            ),
            entries,
            SLAB,
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
            let mut op = shape(
                Operation::new(a, Some(load), OpKind::KtdpLoad, &[ind]),
                entries,
                SLAB,
            );
            op.result_type = ty(entries, SLAB);
            op
        },
        // ⭐ THE RANK-2 FRAME — the gathered `[entries, SLAB]` read as `[T, w]`.
        {
            let mut op = shape(
                Operation::new(a, Some(reshaped), OpKind::TensorReshape, &[load]),
                t_rows,
                w,
            );
            op.result_type = ty(t_rows, w);
            op
        },
        // OUT: [M, w].
        shape(
            Operation::new(a, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
            m,
            w,
        ),
        shape(
            Operation::new(
                a,
                Some(t_o),
                OpKind::KtdpConstructAccessTile,
                &[v_o, zero, zero],
            ),
            m,
            w,
        ),
        shape(
            Operation::new(a, Some(outs), OpKind::TensorEmpty, &[]),
            m,
            w,
        ),
        {
            let mut op = shape(
                Operation::new(a, Some(mm), OpKind::LinalgMatmul, &[val_a, reshaped, outs]),
                m,
                w,
            );
            op.result_type = ty(m, w);
            op
        },
        Operation::new(a, None, OpKind::KtdpStore, &[mm, t_o]),
    ];

    KtirNode {
        func: IRFunction {
            name: "rank_two_page_read",
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
        minted: Default::default(),
        lx_plan: Default::default(),
    }
}

/// The fully-placed layout: the four parameters at non-zero offsets, the cache allocated
/// row-outermost exactly as the splice allocates the paged KV (`allocate_paged_cache`).
fn layout_for(k: &KtirNode) -> BundleLayout {
    let mut l = BundleLayout::default();
    for (i, b) in k.bindings.iter().enumerate() {
        let tid = b.get();
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        // Bytes: A [M, T] f16, cache [8, SLAB] f16, ids [entries] i32, OUT [M, w] f16.
        let bytes: u64 = match i {
            0 => 8 * 512 * 2,
            1 => 8 * SLAB as u64 * 2,
            2 => 4 * 4,
            _ => 8 * 128 * 2,
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
    l.allocate_row_outermost(k.bindings[1].get())
        .expect("the cache allocates row-outermost");
    l
}

fn emitted(
    k: &KtirNode,
) -> Result<Vec<ktir_superdsc::emit::EmittedOp<ktir_superdsc::collective::TpWorld<1>>>, String> {
    let layout = layout_for(k);
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

/// The one gather op's `N_` — its iteration extents `[mb: entries, y: blocks, x: rows, out: cols]`.
fn gather_n(
    op: &ktir_superdsc::emit::EmittedOp<ktir_superdsc::collective::TpWorld<1>>,
) -> serde_json::Value {
    let json = serde_json::to_value(op.op.as_ref().expect("a descriptor")).expect("json");
    json["dscs_"][0]
        .as_object()
        .and_then(|m| m.values().next())
        .expect("one dsc")["N_"]
        .clone()
}

#[test]
fn several_multi_stick_entries_land_row_major_and_one_permute_restickifies() {
    // 4 pages of 8192, read as the rank-2 `[256, 128]` frame — two sticks wide, four entries: the
    // per-block landing is not the frame's one stick column.
    let k = program(4, 128, 8);
    let ops = emitted(&k).unwrap_or_else(|e| panic!("the rank-2 read lowers: {e}"));
    let gathers: Vec<_> = ops
        .iter()
        .filter(|e| e.op_name.starts_with("gather"))
        .collect();
    assert_eq!(
        gathers.len(),
        1,
        "one gather op: {:?}",
        ops.iter().map(|e| &e.op_name).collect::<Vec<_>>()
    );
    assert_eq!(
        gather_n(gathers[0]),
        serde_json::json!({"name_": "n", "mb_": 4, "y_": 64, "x_": 1, "out_": 128}),
        "the entries land as ONE-ROW blocks — `slab / w` of them an entry, each a row's own bytes"
    );
    let permutes: Vec<_> = ops
        .iter()
        .filter(|e| e.op_name.starts_with("permute"))
        .collect();
    assert_eq!(
        permutes.len(),
        1,
        "one whole stick permute restickifies the landing into the frame's own column: {:?}",
        ops.iter().map(|e| &e.op_name).collect::<Vec<_>>()
    );
    // The permute's destination is the buffer the contraction reads — the frame's own column, not
    // the row-major scratch the gather wrote.
    let permute_dst = permutes[0]
        .arg_bindings
        .iter()
        .find(|a| !a.is_input)
        .expect("the permute writes a buffer")
        .buffer
        .clone();
    let mm = ops
        .iter()
        .find(|e| e.op_name.starts_with("matmul") || e.op_name.starts_with("mm"))
        .expect("the contraction lowers");
    assert!(
        mm.arg_bindings.iter().any(|a| a.buffer == permute_dst),
        "the contraction reads the restickified frame {}",
        permute_dst
    );
}

#[test]
fn one_entry_lands_its_whole_plane_with_no_permute() {
    // 1 page of 8192, read as the rank-2 `[64, 128]` frame: one block, so the per-block landing IS
    // the frame's one stick column — the near-miss that must not steal the permute.
    let k = program(1, 128, 8);
    let ops = emitted(&k).unwrap_or_else(|e| panic!("the one-entry read lowers: {e}"));
    assert_eq!(
        ops.iter()
            .filter(|e| e.op_name.starts_with("gather"))
            .count(),
        1
    );
    assert_eq!(
        gather_n(
            ops.iter()
                .find(|e| e.op_name.starts_with("gather"))
                .unwrap()
        ),
        serde_json::json!({"name_": "n", "mb_": 1, "y_": 1, "x_": 64, "out_": 128}),
        "the one entry's whole plane is one `[slab / w, w]` block"
    );
    assert_eq!(
        ops.iter()
            .filter(|e| e.op_name.starts_with("permute"))
            .count(),
        0,
        "one block: the landing is already the frame — no restickify"
    );
}

#[test]
fn one_stick_wide_entries_land_flat_with_no_permute() {
    // 4 pages of 8192, read as the rank-2 `[512, 64]` frame: one stick wide, so stick-major and
    // row-major are the same bytes — the other near-miss.
    let k = program(4, 64, 8);
    let ops = emitted(&k).unwrap_or_else(|e| panic!("the one-stick read lowers: {e}"));
    assert_eq!(
        ops.iter()
            .filter(|e| e.op_name.starts_with("gather"))
            .count(),
        1
    );
    assert_eq!(
        gather_n(
            ops.iter()
                .find(|e| e.op_name.starts_with("gather"))
                .unwrap()
        ),
        serde_json::json!({"name_": "n", "mb_": 4, "y_": 1, "x_": 128, "out_": 64}),
        "each entry's plane is one `[slab / w, w]` block, one stick wide"
    );
    assert_eq!(
        ops.iter()
            .filter(|e| e.op_name.starts_with("permute"))
            .count(),
        0,
        "one stick: the landing is already the frame — no restickify"
    );
}
