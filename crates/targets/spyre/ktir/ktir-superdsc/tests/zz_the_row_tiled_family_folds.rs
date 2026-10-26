// SPDX-License-Identifier: Apache-2.0

//! ⭐⭐⭐⭐⭐ THE ROW-TILED FAMILY FOLD — ITS SEALS, AS TESTS.
//!
//! A row-blocked pointwise program (the kernels' own `EW_LX_ELEMS` law: `silumul` always at
//! prefill, elementwise and scalarmul above the budget) states one load/chain/store family per
//! `[BLOCK_M, N]` block. The walk lowers per-op at the LOAD'S OWN tile, so every block past the
//! first refuses at the row-corner law — while the FIRST block's descriptor already spans the
//! whole view (its out region's `r_cover` is the hull of every block's store, and `node_rows`
//! folds that into the row count). [`row_tiled_family`] proves the blocks are ONE op over
//! windows that TILE THE VIEW EXACTLY and returns the non-leader terminals to skip; the leader
//! emits the fold — one op, `time` factor 1.
//!
//! What these tests pin, per the fold's ruling:
//!
//! * THE POSITIVE: blocks that tile the view fold to ONE descriptor spanning the WHOLE view
//!   (`N_.mb_` is the view's rows, not block 0's height) at `time == 1`.
//! * ⛔ A GAP refuses by name — never a partial merge (the leader's descriptor spans the whole
//!   view, so a gap would compute rows the program never states).
//! * ⛔ AN OVERLAP refuses by name — same law, the mirrored failure.
//! * A COLUMN-CHUNKED pair (several windows at the SAME rows) is NOT a row family: the fold
//!   stays out of it, and the walk lowers each block as its own op at its own column corner
//!   (#722's per-operand column windows — the legalization this pair is the witness for).
//! * THE BARE SILU folds too: its terminal is the longhand's own `arith.divf` (the stored
//!   `silu(x)`, no `up` multiply), one input per block — the chain reader's second terminal
//!   form, not a per-op divide family.
//!
//! Every program is built from the op kinds the walk really reads, so the door sees a real
//! program and not a stub.

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

/// One binary-add program over parameters `a`, `b`, `out` — each viewed `[view.0, view.1]` —
/// stating one load/add/store family per block in `blocks`.
///
/// A block is `(r_start, r_len, c_start, c_len)`: its own access tiles at that row/column
/// corner, its own loads, its own `arith.addf`, its own store. The ONLY variables between the
/// tests are the view's extent and the blocks' windows — the program shape is identical, so a
/// refusal can only be about the windows.
fn add_family(view: (i64, i64), blocks: &[(i64, i64, i64, i64)]) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };

    // Parameters: %0 = a, %1 = b, %2 = out. Every Ssa below is minted in order after them.
    let (p_a, p_b, p_out) = (Ssa(0), Ssa(1), Ssa(2));
    let mut next = 3u32;
    let mut ssa = || {
        let s = Ssa(next);
        next += 1;
        s
    };

    // One view per parameter — the blocks' tiles all reference these, and the view states the
    // parameter's WHOLE extent, which is what `Region::v_rows` and the partition seals read.
    let (v_a, v_b, v_o) = (ssa(), ssa(), ssa());
    let mut ops = vec![
        shape(
            Operation::new(a, Some(v_a), OpKind::KtdpConstructMemoryView, &[p_a]),
            view.0,
            view.1,
        ),
        shape(
            Operation::new(a, Some(v_b), OpKind::KtdpConstructMemoryView, &[p_b]),
            view.0,
            view.1,
        ),
        shape(
            Operation::new(a, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
            view.0,
            view.1,
        ),
    ];

    for &(r_start, r_len, c_start, c_len) in blocks {
        let (c_rs, c_cs) = (ssa(), ssa());
        ops.push(
            Operation::new(a, Some(c_rs), OpKind::ArithConstant, &[]).with_attr(
                a,
                AttrKey::Value,
                Attr::Int(r_start),
            ),
        );
        ops.push(
            Operation::new(a, Some(c_cs), OpKind::ArithConstant, &[]).with_attr(
                a,
                AttrKey::Value,
                Attr::Int(c_start),
            ),
        );
        let (t_a, l_a) = (ssa(), ssa());
        ops.push(shape(
            Operation::new(
                a,
                Some(t_a),
                OpKind::KtdpConstructAccessTile,
                &[v_a, c_rs, c_cs],
            ),
            r_len,
            c_len,
        ));
        ops.push(shape(
            Operation::new(a, Some(l_a), OpKind::KtdpLoad, &[t_a]),
            r_len,
            c_len,
        ));
        let (t_b, l_b) = (ssa(), ssa());
        ops.push(shape(
            Operation::new(
                a,
                Some(t_b),
                OpKind::KtdpConstructAccessTile,
                &[v_b, c_rs, c_cs],
            ),
            r_len,
            c_len,
        ));
        ops.push(shape(
            Operation::new(a, Some(l_b), OpKind::KtdpLoad, &[t_b]),
            r_len,
            c_len,
        ));
        let add = ssa();
        ops.push(shape(
            Operation::new(a, Some(add), OpKind::ArithAddf, &[l_a, l_b]),
            r_len,
            c_len,
        ));
        let t_o = ssa();
        ops.push(shape(
            Operation::new(
                a,
                Some(t_o),
                OpKind::KtdpConstructAccessTile,
                &[v_o, c_rs, c_cs],
            ),
            r_len,
            c_len,
        ));
        ops.push(Operation::new(a, None, OpKind::KtdpStore, &[add, t_o]));
    }

    KtirNode {
        func: IRFunction {
            name: "row_fold_add",
            arguments: a.args(vec![
                (p_a, IrType::Index),
                (p_b, IrType::Index),
                (p_out, IrType::Index),
            ]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        },
        program: Program::Elementwise(Elementwise::Add),
        bindings: vec![BufferId::new(401), BufferId::new(402), BufferId::new(403)],
        mask: None,
        step_inputs: Vec::new(),
        node_out_tid: None,
        minted: Default::default(),
    }
}

/// One BARE-SILU program over parameters `x`, `out` — each viewed `[view.0, view.1]` — stating
/// one five-op longhand family per block in `blocks` (`elementwise.py`'s `silu_fwd` form: the
/// divide's result stored directly, no `up` multiply). The block law is `add_family`'s; only the
/// per-block body differs, so a refusal can only be about the windows or the chain form.
fn silu_family(view: (i64, i64), blocks: &[(i64, i64, i64, i64)]) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };

    // Parameters: %0 = x, %1 = out.
    let (p_x, p_out) = (Ssa(0), Ssa(1));
    let mut next = 2u32;
    let mut ssa = || {
        let s = Ssa(next);
        next += 1;
        s
    };

    let (v_x, v_o) = (ssa(), ssa());
    let mut ops = vec![
        shape(
            Operation::new(a, Some(v_x), OpKind::KtdpConstructMemoryView, &[p_x]),
            view.0,
            view.1,
        ),
        shape(
            Operation::new(a, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
            view.0,
            view.1,
        ),
    ];

    for &(r_start, r_len, c_start, c_len) in blocks {
        let (c_rs, c_cs) = (ssa(), ssa());
        ops.push(
            Operation::new(a, Some(c_rs), OpKind::ArithConstant, &[]).with_attr(
                a,
                AttrKey::Value,
                Attr::Int(r_start),
            ),
        );
        ops.push(
            Operation::new(a, Some(c_cs), OpKind::ArithConstant, &[]).with_attr(
                a,
                AttrKey::Value,
                Attr::Int(c_start),
            ),
        );
        let (t_x, l_x) = (ssa(), ssa());
        ops.push(shape(
            Operation::new(
                a,
                Some(t_x),
                OpKind::KtdpConstructAccessTile,
                &[v_x, c_rs, c_cs],
            ),
            r_len,
            c_len,
        ));
        ops.push(shape(
            Operation::new(a, Some(l_x), OpKind::KtdpLoad, &[t_x]),
            r_len,
            c_len,
        ));
        // The five-op longhand, exactly as the recognizer reads it: negate → exp → `1 + e^-x`
        // (a splatted 1.0 — the value the reader checks) → the divide whose NUMERATOR is the
        // gate. The divide is the terminal: its result is what the store writes.
        let (neg, exp) = (ssa(), ssa());
        ops.push(shape(
            Operation::new(a, Some(neg), OpKind::ArithNegf, &[l_x]),
            r_len,
            c_len,
        ));
        ops.push(shape(
            Operation::new(a, Some(exp), OpKind::MathExp, &[neg]),
            r_len,
            c_len,
        ));
        let (one, splat, add) = (ssa(), ssa(), ssa());
        ops.push(
            Operation::new(a, Some(one), OpKind::ArithConstant, &[]).with_attr(
                a,
                AttrKey::Value,
                Attr::Float(1.0),
            ),
        );
        ops.push(shape(
            Operation::new(a, Some(splat), OpKind::TensorSplat, &[one]),
            r_len,
            c_len,
        ));
        ops.push(shape(
            Operation::new(a, Some(add), OpKind::ArithAddf, &[exp, splat]),
            r_len,
            c_len,
        ));
        let div = ssa();
        ops.push(shape(
            Operation::new(a, Some(div), OpKind::ArithDivf, &[l_x, add]),
            r_len,
            c_len,
        ));
        let t_o = ssa();
        ops.push(shape(
            Operation::new(
                a,
                Some(t_o),
                OpKind::KtdpConstructAccessTile,
                &[v_o, c_rs, c_cs],
            ),
            r_len,
            c_len,
        ));
        ops.push(Operation::new(a, None, OpKind::KtdpStore, &[div, t_o]));
    }

    KtirNode {
        func: IRFunction {
            name: "row_fold_silu",
            arguments: a.args(vec![(p_x, IrType::Index), (p_out, IrType::Index)]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        },
        program: Program::Elementwise(Elementwise::Silu),
        bindings: vec![BufferId::new(401), BufferId::new(402)],
        mask: None,
        step_inputs: Vec::new(),
        node_out_tid: None,
        minted: Default::default(),
    }
}

/// The layout every test here places its parameters with: distinct segments per parameter, so
/// an operand-address assertion discriminates WHICH parameter an op reads.
fn layout_for(k: &KtirNode, view: (i64, i64)) -> BundleLayout {
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
                offset: (i as u64) * 0x10000,
                size: (view.0 * view.1 * 2) as u64,
                cover: ktir_superdsc::placement::BindCover::Full,
            },
        );
    }
    l
}

fn lowered(k: &KtirNode, view: (i64, i64)) -> Result<Vec<ktir_superdsc::emit::EmittedOp>, String> {
    let layout = layout_for(k, view);
    let mut sid = 0i64;
    lower_function(k, Some(&layout), &mut sid, None).map_err(|e| e.message)
}

/// The `dscs_[0]` entry of an emitted op — the descriptor the assertions read.
fn dsc(op: &ktir_superdsc::emit::EmittedOp) -> &ktir_superdsc::wire::Dsc {
    op.dsc()
        .dscs_
        .first()
        .and_then(|m| m.values().next())
        .expect("the add has a descriptor")
}

/// ⭐⭐⭐ THE POSITIVE: blocks that tile the view fold to ONE whole-view descriptor.
///
/// Two `[32, 128]` blocks over a `[64, 128]` view — the row-blocked shape whose second block
/// the walk refused at the row-corner law before the fold. The fold must answer ONE op whose
/// iteration space spans the VIEW (mb = 64, not block 0's 32) at `time == 1`: the ruling's
/// "the folded op goes to dxp whole, time factor 1".
#[test]
fn row_blocks_that_tile_the_view_fold_into_one_whole_view_descriptor() {
    let view = (64, 128);
    let k = add_family(view, &[(0, 32, 0, 128), (32, 32, 0, 128)]);
    let ops = lowered(&k, view).expect("a tiling row-blocked family folds");
    let [op] = &ops[..] else {
        panic!(
            "two row blocks that tile the view are ONE folded op, got {:?}",
            ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
        );
    };
    assert_eq!(
        op.op_name, "add_o403",
        "the folded op is the family's one add"
    );
    assert_eq!(
        op.time, 1,
        "the fold emits no tiling of its own — time factor 1"
    );
    let d = dsc(op);
    assert_eq!(
        (d.N_.mb_, d.N_.out_),
        (64, 128),
        "the folded descriptor spans the WHOLE view — the leader's r_cover is the hull of every \
         block's store, and node_rows folds that into the row count. A descriptor cut to block \
         0's height would leave every later block's rows unwritten on the card"
    );
}

/// ⛔ THE GAP: a family whose windows LEAVE ROWS UNSTATED refuses by name, never a partial
/// merge. Blocks at rows `0..32` and `40..64` of a 64-row view leave `32..40` to no store —
/// and the leader's one descriptor spans all 64 rows, so folding would compute rows the
/// program never states.
#[test]
fn a_gap_between_row_windows_refuses_by_name() {
    let view = (64, 128);
    let k = add_family(view, &[(0, 32, 0, 128), (40, 24, 0, 128)]);
    let e = match lowered(&k, view) {
        Err(e) => e,
        Ok(ops) => panic!(
            "a gapped family must refuse, not fold {} op(s) over rows no store states",
            ops.len()
        ),
    };
    assert!(
        e.contains("must TILE the view exactly"),
        "the refusal names the tiling law, got: {e}"
    );
    assert!(
        e.contains("row 32 was due"),
        "the refusal names WHERE the gap is (row 32 was due, 40 arrived), got: {e}"
    );
}

/// ⛔ THE OVERLAP: blocks at rows `0..40` and `32..64` state rows `32..40` TWICE. The fold
/// refuses rather than merge — one folded descriptor computes each row once, so an overlapping
/// family is not the program the producer stated.
#[test]
fn an_overlap_between_row_windows_refuses_by_name() {
    let view = (64, 128);
    let k = add_family(view, &[(0, 40, 0, 128), (32, 32, 0, 128)]);
    let e = match lowered(&k, view) {
        Err(e) => e,
        Ok(ops) => panic!(
            "an overlapping family must refuse, not fold {} op(s) over rows stated twice",
            ops.len()
        ),
    };
    assert!(
        e.contains("must TILE the view exactly"),
        "the refusal names the tiling law, got: {e}"
    );
    assert!(
        e.contains("row 40 was due"),
        "the refusal names WHERE the overlap is (row 40 was due, 32 arrived), got: {e}"
    );
}

/// ⭐ A COLUMN-CHUNKED PAIR IS NOT A ROW FAMILY. Two `[64, 64]` windows at the SAME rows and
/// different columns are the column-chunked shape, and the fold must stay out of it — the
/// blocks tile the COLUMN axis, which the row fold does not own. The walk lowers each block
/// AS ITS OWN OP at its own column corner — #722's per-operand column windows legalized
/// exactly this pair, which the pre-#722 walk refused at its one-function column-corner law —
/// so the answer is TWO ops, never the fold's one.
#[test]
fn a_column_chunked_pair_is_not_a_row_family() {
    let view = (64, 128);
    let k = add_family(view, &[(0, 64, 0, 64), (0, 64, 64, 64)]);
    let ops = lowered(&k, view).expect(
        "a column-chunked pair lowers as two ops at their own column corners — #722's \
         per-operand column windows legalized exactly this pair",
    );
    let [op0, op1] = &ops[..] else {
        panic!(
            "the fold must stay out of a column family — the blocks tile the COLUMN axis, \
             which the row fold does not own, so two blocks are TWO ops. Got {}",
            ops.len()
        );
    };
    // EACH OP IS THE FAMILY'S OWN ADD at its own chunk: the iteration space is the block's
    // own `[64, 64]` window, never the view's 128 cols — that whole-view descriptor is the
    // FOLD's emission, and the fold must not engage here.
    for op in [op0, op1] {
        assert_eq!(op.op_name, "add_o403", "each op is the family's one add");
        assert_eq!(op.time, 1, "a per-op lowering emits no tiling of its own");
        let d = dsc(op);
        assert_eq!(
            (d.N_.mb_, d.N_.out_),
            (64, 64),
            "each op's iteration space is its OWN `[64, 64]` chunk — the view's 128 cols would \
             be the fold's one whole-view descriptor"
        );
    }
    // EACH AT ITS OWN CORNER: every tensor the two descriptors share — `a`, `b`, and `out` —
    // starts one CHUNK higher in op 1, the same constant on every core. That constant is the
    // column corner 64 in the `[64, 128]` stick-major buffer's bytes: 64 rows · 64 elements ·
    // 2 B = 8192, the second stick-group's base. The same corner on both ops would state the
    // same addresses; the fold's one descriptor would state them once.
    fn per_tensor_addresses(
        op: &ktir_superdsc::emit::EmittedOp,
    ) -> std::collections::BTreeMap<String, std::collections::BTreeMap<String, u64>> {
        let mut out = std::collections::BTreeMap::new();
        for n in &dsc(op).scheduleTree_ {
            if n.nodeType_ != "allocate" {
                continue;
            }
            let cores = n
                .startAddressCoreCorelet_
                .data_
                .iter()
                .map(|(core, addr)| {
                    (
                        core.clone(),
                        addr.parse::<u64>()
                            .expect("an address is an integer string"),
                    )
                })
                .collect();
            out.insert(n.name_.clone(), cores);
        }
        out
    }
    let a0 = per_tensor_addresses(op0);
    let a1 = per_tensor_addresses(op1);
    assert_eq!(
        a0.len(),
        3,
        "the pair's descriptors allocate the family's three tensors"
    );
    for (name, cores0) in &a0 {
        let cores1 = a1
            .get(name)
            .unwrap_or_else(|| panic!("tensor {name} is allocated by both ops"));
        for (core, addr0) in cores0 {
            let addr1 = cores1[core];
            assert_eq!(
                addr1 - addr0,
                8192,
                "tensor {name} core {core}: op 1's chunk starts one corner higher — 64 rows · \
                 64-col corner · 2 B, the second stick-group's base in the `[64, 128]` \
                 stick-major buffer. The same corner on both ops would state the same address"
            );
        }
    }
}

/// ⭐ THE BARE SILU FOLDS AS THE CHAIN'S UNARY FORM. Two `[32, 128]` blocks of the five-op
/// longhand, each storing its own `silu(x)` — the terminal is the DIVIDE, not a multiply, and
/// the family must fold to ONE `("silu", 1)` descriptor spanning the view. This pins the path
/// no splice golden covers (their elementwise shapes are single-window): the fold's leader
/// classifies through the chain reader (`RowFoldKind::BareSilu`, one input — the gate), never
/// as a per-op `Elementwise(RealDiv)` family over the divide's own two operands.
#[test]
fn row_blocks_of_a_bare_silu_fold_into_one_whole_view_descriptor() {
    let view = (64, 128);
    let k = silu_family(view, &[(0, 32, 0, 128), (32, 32, 0, 128)]);
    let ops = lowered(&k, view).expect("a tiling bare-silu family folds");
    let [op] = &ops[..] else {
        panic!(
            "two row blocks of one bare silu are ONE folded op, got {:?}",
            ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
        );
    };
    assert_eq!(
        op.op_name, "silu_o402",
        "the folded op is the family's one silu"
    );
    assert_eq!(
        op.time, 1,
        "the fold emits no tiling of its own — time factor 1"
    );
    let d = dsc(op);
    assert_eq!(
        (d.N_.mb_, d.N_.out_),
        (64, 128),
        "the folded descriptor spans the WHOLE view, exactly as the binary family's does"
    );
}

/// One SUBTRACTION-BY-CONSTANT program over parameters `a`, `out` — each viewed
/// `[view.0, view.1]` — stating one load/subtract/store family per block in `blocks`. A block
/// is `(r_start, r_len, c_start, c_len, const_first)`: the `const_first` flag states WHICH
/// input the one shared splatted constant occupies in that block's `arith.subf` — the ORDER
/// fact, non-commutative, carried per block so the mixing test can state two functions under
/// one family's stores.
fn sub_family(view: (i64, i64), blocks: &[(i64, i64, i64, i64, bool)]) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };

    let (p_a, p_out) = (Ssa(0), Ssa(1));
    let mut next = 2u32;
    let mut ssa = || {
        let s = Ssa(next);
        next += 1;
        s
    };

    let (v_a, v_o) = (ssa(), ssa());
    let mut ops = vec![
        shape(
            Operation::new(a, Some(v_a), OpKind::KtdpConstructMemoryView, &[p_a]),
            view.0,
            view.1,
        ),
        shape(
            Operation::new(a, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
            view.0,
            view.1,
        ),
    ];

    // ONE shared constant, splatted once — the family's stated subtrahend (the f16 floor, the
    // online softmax's running-max initial, a constant no graph payload names).
    let (cst, splat) = (ssa(), ssa());
    ops.push(
        Operation::new(a, Some(cst), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Float(-65504.0),
        ),
    );
    ops.push(shape(
        Operation::new(a, Some(splat), OpKind::TensorSplat, &[cst]),
        1,
        1,
    ));

    for &(r_start, r_len, c_start, c_len, const_first) in blocks {
        let (c_rs, c_cs) = (ssa(), ssa());
        ops.push(
            Operation::new(a, Some(c_rs), OpKind::ArithConstant, &[]).with_attr(
                a,
                AttrKey::Value,
                Attr::Int(r_start),
            ),
        );
        ops.push(
            Operation::new(a, Some(c_cs), OpKind::ArithConstant, &[]).with_attr(
                a,
                AttrKey::Value,
                Attr::Int(c_start),
            ),
        );
        let (t_a, l_a) = (ssa(), ssa());
        ops.push(shape(
            Operation::new(
                a,
                Some(t_a),
                OpKind::KtdpConstructAccessTile,
                &[v_a, c_rs, c_cs],
            ),
            r_len,
            c_len,
        ));
        ops.push(shape(
            Operation::new(a, Some(l_a), OpKind::KtdpLoad, &[t_a]),
            r_len,
            c_len,
        ));
        let sub = ssa();
        let operands = if const_first {
            &[splat, l_a]
        } else {
            &[l_a, splat]
        };
        ops.push(shape(
            Operation::new(a, Some(sub), OpKind::ArithSubf, operands),
            r_len,
            c_len,
        ));
        let t_o = ssa();
        ops.push(shape(
            Operation::new(
                a,
                Some(t_o),
                OpKind::KtdpConstructAccessTile,
                &[v_o, c_rs, c_cs],
            ),
            r_len,
            c_len,
        ));
        ops.push(Operation::new(a, None, OpKind::KtdpStore, &[sub, t_o]));
    }

    KtirNode {
        func: IRFunction {
            name: "row_fold_sub",
            arguments: a.args(vec![(p_a, IrType::Index), (p_out, IrType::Index)]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        },
        program: Program::Elementwise(Elementwise::Sub),
        bindings: vec![BufferId::new(401), BufferId::new(402)],
        mask: None,
        step_inputs: Vec::new(),
        node_out_tid: None,
        minted: Default::default(),
    }
}

/// The subtraction family's layout: `layout_for`'s placements plus the scale registry the
/// subtraction's own scalar form resolves through — the door looks the stated constant up BY
/// VALUE, so the layout that baked the program must carry it (and its `[1,1]` slot placement).
fn sub_lowered(
    k: &KtirNode,
    view: (i64, i64),
) -> Result<Vec<ktir_superdsc::emit::EmittedOp>, String> {
    let mut layout = layout_for(k, view);
    layout.scalarmul_scales = vec![-65504.0];
    layout.synth(
        ktir_superdsc::place::PlaceId::Act(ktir_superdsc::reserved_tids::scalarmul_scale_tid(0)),
        &[1, 1],
    );
    let mut sid = 0i64;
    lower_function(k, Some(&layout), &mut sid, None).map_err(|e| e.message)
}

/// ⭐ THE SUBTRACTION FAMILY FOLDS TOO — one ORDER throughout. Two `tile − C` blocks that tile
/// the view fold to ONE whole-view descriptor, exactly the binary family's fold: the kind is
/// `RowFoldKind::ScalarSub`, one input per block (the tile — the constant rides the op), and
/// the leader lowers through the subtraction's own scalar form.
#[test]
fn row_blocks_of_a_subtraction_fold_into_one_whole_view_descriptor() {
    let view = (64, 128);
    let k = sub_family(view, &[(0, 32, 0, 128, false), (32, 32, 0, 128, false)]);
    let ops = sub_lowered(&k, view).expect("a tiling subtraction family folds");
    let [op] = &ops[..] else {
        panic!(
            "two row blocks of one subtraction are ONE folded op, got {:?}",
            ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
        );
    };
    assert_eq!(
        op.op_name, "scalarmul_o402",
        "the folded op is the family's one subtraction"
    );
    assert_eq!(
        op.time, 1,
        "the fold emits no tiling of its own — time factor 1"
    );
    let d = dsc(op);
    assert_eq!(
        (d.N_.mb_, d.N_.out_),
        (64, 128),
        "the folded descriptor spans the WHOLE view, exactly as the binary family's does"
    );
}

/// ⛔ A FAMILY MIXING THE TWO ORDERS OF A SUBTRACTION IS REFUSED. `C − tile` and `tile − C`
/// are DIFFERENT functions, so the ORDER rides `RowFoldKind::ScalarSub` itself: one block
/// stating each order is two kinds under one family's stores, and the fold refuses rather
/// than merge blocks that compute different functions per block — the same law that refuses
/// a family whose blocks clamped at different bounds.
#[test]
fn a_family_mixing_the_two_orders_of_a_subtraction_is_refused() {
    let view = (64, 128);
    let k = sub_family(view, &[(0, 32, 0, 128, false), (32, 32, 0, 128, true)]);
    let e = match sub_lowered(&k, view) {
        Err(e) => e,
        Ok(ops) => panic!(
            "a family mixing `tile − C` and `C − tile` must refuse, not fold {} op(s) over two \
             different functions",
            ops.len()
        ),
    };
    assert!(
        e.contains("not one op family"),
        "the refusal names the kind mismatch, got: {e}"
    );
    assert!(
        e.contains("const_first"),
        "the refusal's kinds carry the ORDER fact, so the two sides are named as the different \
         functions they are, got: {e}"
    );
}
