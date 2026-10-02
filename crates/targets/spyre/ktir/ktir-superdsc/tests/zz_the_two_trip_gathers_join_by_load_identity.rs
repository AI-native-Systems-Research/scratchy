// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! ⭐⭐⭐⭐⭐ ISSUE 201 ITEM 9 — THE MULTI-TILE FACTS NO FIXTURE EVER STATED.
//!
//! The rung-4 file (`zz_the_gathered_matmul_materializes.rs`) covers ONE tile starting at
//! entry 0 — the single-block shape. The multi-block shape — an unrolled KV sweep spliced into
//! straight-line trips, every trip carrying its own `ktdp.construct_indirect_access_tile` over
//! the SAME two parameters — is the shape `gathers_of` exists for, and none of its facts had a
//! test in this repo:
//!
//! * TWO TRIPS WITH DIFFERENT START ENTRIES PRODUCE DIFFERENT INDEX OFFSETS, and swapping them
//!   swaps the offsets (the issue's own example of a control that comes out different). A join
//!   that matched tiles by enumeration position or by TID — instead of by the identity of each
//!   trip's own LOAD — hands every trip the first tile's index window, and trip 1 gathers
//!   trip 0's rows from a clean bake. The two programs below are IDENTICAL except which trip
//!   carries anchor 32 and which carries 64, so any join that cannot tell them apart by SSA
//!   identity emits the same descriptors for both — and the assertion on the two index offsets
//!   catches that by coming out equal when it must come out swapped.
//! * THE DEVICE-WIDTH RESERVATION (item 4) AND THE WINDOWED PAD DROP (item 5): a windowed
//!   whole-function matmul whose logical `n` already meets the util floor reserves AND emits
//!   at `n` rounded UP to a whole 64-stick — never at the 8×-padded `for_output` width, and
//!   never at a raw sub-stick `n` that the per-core stick guard would refuse.
//!
//! Every program here is built from the op kinds the walk really reads, so the door sees a real
//! program and not a stub.

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

/// Two gathered trips over the same `[128, 64]` V table and `[128]` ids — the multi-block
/// paged-attention shape at its smallest, `M=8, K=64, V=128, HEAD=64` (the rung-4 fixture's own
/// geometry, so the two files' programs differ only in the number of tiles).
///
/// Trip 0 gathers 64 entries at anchor `a0`; trip 1 gathers 64 entries at anchor `a1`. The
/// anchors are 32-entry sticks apart (64 and 96 here — BLOCK_N=64 trips, anchor = trip · 64),
/// because `EntryBase::of_entries` refuses an unaligned one and the CUT-arm leg stride test in
/// the vendor-pin file owns that refusal. THE ONLY DIFFERENCE between `Swapped::No` and
/// `Swapped::Yes` is which trip carries which anchor.
#[derive(Clone, Copy)]
enum Swapped {
    /// Trip 0 at entry 64, trip 1 at entry 128 — the program's own order.
    No,
    /// Trip 0 at entry 128, trip 1 at entry 64 — the same two trips, anchors exchanged.
    Yes,
}

fn two_trip_program(swapped: Swapped) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };

    // Parameters: %0 = P, %1 = V table, %2 = ids, %3 = out.
    let (p_p, p_v, p_ids, p_out) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3));
    let zero = Ssa(4);
    // The A chain (P's plain load), the V table's plain view (each trip's data operand), the
    // ids view (each trip's index operand).
    let (v_a, t_a, val_a) = (Ssa(5), Ssa(6), Ssa(7));
    let (v_v, v_i) = (Ssa(8), Ssa(9));
    // Trip b's indirect tile and its load.
    let (ind0, load0, ind1, load1) = (Ssa(10), Ssa(11), Ssa(12), Ssa(13));
    // The output view/tile, the zero outs, the two matmuls and their addf, the store.
    let (v_o, t_o, outs, mm0, mm1, addf) = (Ssa(14), Ssa(15), Ssa(16), Ssa(17), Ssa(18), Ssa(19));

    // The ids buffer covers the FARTHEST window: anchor 128 + 64 entries = 192 words.
    let (anchor0, anchor1) = match swapped {
        Swapped::No => (64, 128),
        Swapped::Yes => (128, 64),
    };

    // The `Sym(1) + Dim(0)` subscript `gather_subscripts` builds — `Sym(1)` named by each
    // tile's `intermediate_vars[1]`, a PER-TRIP constant, so the anchor is the vars entry and
    // not the map itself. Interned before the ops vec because their children are borrows.
    let sub = a.expr(AffineExpr::Add(
        a.expr(AffineExpr::Sym(1)),
        a.expr(AffineExpr::Dim(0)),
    ));

    // One trip's indirect tile over the SAME two parameter views, anchored at `anchor`.
    let indirect = |result: Ssa, anchor: i64| {
        shape(
            Operation::new(
                a,
                Some(result),
                OpKind::KtdpConstructIndirectAccessTile,
                &[v_v, v_i],
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
            // The anchor constant, stated per trip: trip b's tile names ITS OWN anchor.
            Attr::Ssas(a.ssa(vec![zero, Ssa((30 + anchor).try_into().unwrap())])),
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
    // The per-trip anchor constants, so each tile's `intermediate_vars[0]` resolves to its own.
    let anchor_const = |anchor: i64| {
        Operation::new(
            a,
            Some(Ssa((30 + anchor).try_into().unwrap())),
            OpKind::ArithConstant,
            &[],
        )
        .with_attr(a, AttrKey::Value, Attr::Int(anchor))
    };

    // The matmuls' results are UNSTORED intermediates, so each states a 2-D tensor type — the
    // footprint the walk's minted buffer is sized from.
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
        anchor_const(anchor0),
        anchor_const(anchor1),
        // P: [8, 64] — the shared activation.
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
        // V: [128, 64] — the table BOTH trips gather.
        shape(
            Operation::new(a, Some(v_v), OpKind::KtdpConstructMemoryView, &[p_v]),
            128,
            64,
        ),
        // ids: [192] i32 — the whole sweep: trip 0 reads words 64–128, trip 1 words 128–192.
        Operation::new(a, Some(v_i), OpKind::KtdpConstructMemoryView, &[p_ids])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![192])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::I32)),
        // Trip 0's tile and load.
        indirect(ind0, anchor0),
        shape(
            Operation::new(a, Some(load0), OpKind::KtdpLoad, &[ind0]),
            64,
            64,
        ),
        // Trip 1's tile and load.
        indirect(ind1, anchor1),
        shape(
            Operation::new(a, Some(load1), OpKind::KtdpLoad, &[ind1]),
            64,
            64,
        ),
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
        // The contraction: each trip a [8,64]·[64,64] matmul over its OWN gathered B, summed.
        {
            let mut op = shape(
                Operation::new(a, Some(mm0), OpKind::LinalgMatmul, &[val_a, load0, outs]),
                8,
                64,
            );
            op.result_type = ty(8, 64);
            op
        },
        {
            let mut op = shape(
                Operation::new(a, Some(mm1), OpKind::LinalgMatmul, &[val_a, load1, outs]),
                8,
                64,
            );
            op.result_type = ty(8, 64);
            op
        },
        shape(
            Operation::new(a, Some(addf), OpKind::ArithAddf, &[mm0, mm1]),
            8,
            64,
        ),
        Operation::new(a, None, OpKind::KtdpStore, &[addf, t_o]),
    ];

    KtirNode {
        func: IRFunction {
            name: "paged_two_trip",
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

/// The fully-placed layout for the four parameters — the rung-4 file's own placement law: the
/// table at a non-zero offset so an address assertion discriminates WHICH buffer an op reads.
fn layout_for(k: &KtirNode) -> BundleLayout {
    let mut l = BundleLayout::default();
    for (i, b) in k.bindings.iter().enumerate() {
        let tid = b.get();
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        // Bytes: P [8,64] f16, V [128,64] f16, ids [128] i32, OUT [8,64] f16.
        let bytes: u64 = match i {
            0 => 8 * 64 * 2,
            1 => 128 * 64 * 2,
            2 => 192 * 4,
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
            },
        );
    }
    l
}

fn emitted(k: &KtirNode) -> Result<Vec<ktir_superdsc::emit::EmittedOp>, String> {
    let layout = layout_for(k);
    let mut sid = 0i64;
    lower_function(k, Some(&layout), &mut sid).map_err(|e| e.message)
}

/// The `index_tensor` alloc's run-base word (`startAddressCoreCorelet_.data_["[0, 0, 0]"]`,
/// in 4-B index words) of an emitted gathered op — THE fact that differs between two trips with
/// different start entries. `None` when the op declares no index alloc.
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

/// ⭐⭐⭐ TWO TRIPS WITH DIFFERENT START ENTRIES GATHER FROM DIFFERENT INDEX WORDS, and the
/// identity join — not the enumeration order — decides which is which.
///
/// Each trip's `ktdp.construct_indirect_access_tile` names its OWN anchor constant, so trip 0's
/// index alloc runs at anchor words and trip 1's at its own. The CONTROL is the swap: the same
/// two trips with the anchors exchanged must exchange the two run bases. A join that matched
/// tiles by enumeration position alone cannot fail this test (both programs walk their tiles in
/// the same order); a join that hands every trip the FIRST tile's index window cannot either
/// (both programs' first tiles agree) — but a join that reads the WRONG trip's anchor swaps the
/// two run bases in one program and not the other, and the paired assertions below catch
/// exactly that disagreement.
#[test]
fn two_trips_with_different_starts_gather_from_different_index_words() {
    let ops = emitted(&two_trip_program(Swapped::No))
        .expect("the two-trip program lowers through the whole-function door");
    // 4 materialized copy legs (two trips × two 32-entry sticks each) + 2 matmuls + 1 addf.
    let copies: Vec<&ktir_superdsc::emit::EmittedOp> =
        ops.iter().filter(|o| index_run_base(o).is_some()).collect();
    assert_eq!(
        copies.len(),
        4,
        "two trips of 64 entries = four gathered copy legs, one per 32-entry index stick -- \
         got {} ops with an index alloc out of {} total: {:?}",
        copies.len(),
        ops.len(),
        ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
    );
    // The copy legs emit in walk order: trip 0's two sticks, then trip 1's two. The run base is
    // an ADDRESS: the ids parameter's placement (seg3 + param-2's 0x20000 offset) plus the
    // anchor in 4-B words — trip 0's first leg at word 64, its second one stick (32 words) on,
    // trip 1's two legs at words 128 and 160.
    const SEG3: u64 = 0xC_0000_0000;
    const IDS_OFFSET: u64 = 2 * 0x10000;
    let ids_base = SEG3 + IDS_OFFSET;
    let bases: Vec<u64> = copies.iter().filter_map(|o| index_run_base(o)).collect();
    assert_eq!(
        bases,
        (0..4).map(|i| ids_base + 256 + i * 128).collect::<Vec<_>>(),
        "trip 0 (anchor entry 64) runs its two index sticks at words 64 and 96 (byte 256/384 of \
         the ids buffer), trip 1 (anchor entry 128) at 512/640 -- got {bases:?} for op order {:?}",
        copies.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
    );
}

/// ⛔ CONTROL — THE SAME TWO TRIPS WITH THE ANCHORS EXCHANGED EXCHANGE THE RUN BASES. This is
/// the issue's own discriminating control: "swapping them swaps the offsets". A join that
/// cannot tell the two trips apart by SSA identity — one that reads the FIRST tile's anchor for
/// every trip, which is what a TID join or an enumeration-index join does — emits THE SAME run
/// bases for both programs, and this assertion catches that by demanding the exchange.
#[test]
fn swapping_the_trips_swaps_the_index_offsets() {
    let straight =
        emitted(&two_trip_program(Swapped::No)).expect("the straight two-trip program lowers");
    let swapped =
        emitted(&two_trip_program(Swapped::Yes)).expect("the swapped two-trip program lowers");
    let bases = |ops: &Vec<ktir_superdsc::emit::EmittedOp>| -> Vec<u64> {
        ops.iter().filter_map(index_run_base).collect()
    };
    let (b, s) = (bases(&straight), bases(&swapped));
    assert_ne!(
        b, s,
        "the two programs differ only in which trip carries which anchor -- a join that emits \
         the same index offsets for both has matched the trips by something other than their \
         own loads (straight {b:?}, swapped {s:?})"
    );
    // The walk emits ops in PROGRAM order, so the swapped program's first two legs are the trip
    // anchored at 128 and its last two the trip anchored at 64: the two HALVES exchange, not
    // the whole list reversed.
    assert_eq!(
        b,
        vec![s[2], s[3], s[0], s[1]],
        "exchanging the anchors must exchange the two trips' index run bases (straight {b:?}, \
         swapped {s:?})"
    );
}

// ── THE DEVICE-WIDTH RESERVATION AND THE WINDOWED PAD DROP (items 4 and 5) ─────────────────

use ktir_superdsc::work::{DeviceWidth, UTIL_FLOOR_CORES};

/// ⭐⭐⭐ `for_matmul` IS THE ONE WIDTH DECISION, AND `windowed` DROPS ONLY THE UTIL-FLOOR BUMP
/// (item 5's own numbers): an unaligned logical `n` whose `CoreSplit` already meets the floor
/// still rounds UP to a whole 64-stick — never to the raw sub-stick `n` the per-core stick
/// guard refuses, and never to the 8×-padded `for_output` width that over-reserved granite's
/// tiled_k intermediate.
#[test]
fn the_windowed_drop_keeps_the_stick_rounding_and_drops_only_the_floor_bump() {
    // m=64, n=100, k=2048: `CoreSplit::plan(64, 100)` splits onto 32 cores (≥ the floor of 8),
    // so the windowed drop fires — and the width is 100 rounded UP to a whole stick (128), not
    // the raw 100 (sub-stick: the emitter's per-core stick guard refuses it) and not
    // `for_output`'s padded 512 (macs = 64·128·2048 ≥ 2²⁰ bumps 2 sticks to the floor, and
    // `bump_sticks_to_splittable`'s ≤1/8 pad test sends 128 → 512).
    let m = 64;
    let n = 100;
    let k = 2048;
    assert!(
        ktir_superdsc::work::CoreSplit::plan(m, n).ncores() >= UTIL_FLOOR_CORES,
        "the control's premise: {m}×{n} already splits onto ≥{UTIL_FLOOR_CORES} cores, so the \
         windowed drop fires"
    );
    assert_eq!(
        DeviceWidth::for_matmul(m, n, k, true).get(),
        128,
        "windowed: the stick rounding SURVIVES the drop (100 -> 128), and only the floor bump \
         is dropped (not 512)"
    );
    // The control: the SAME shape staged (not windowed) takes the full `for_output` rule and
    // pads to the splittable width — the two readings differ, so the test discriminates them.
    assert_eq!(
        DeviceWidth::for_matmul(m, n, k, false).get(),
        DeviceWidth::for_output(m, n, k).get(),
        "staged: the full rule applies, and the over-reservation the issue measured (granite \
         tiled_k's 8×) is exactly this branch"
    );
    assert_ne!(
        DeviceWidth::for_matmul(m, n, k, false).get(),
        DeviceWidth::for_matmul(m, n, k, true).get(),
        "the control comes out different: windowed and staged disagree on this shape, so the \
         windowed assertion above is not vacuous"
    );
}

/// ⭐ THE BUMP SURVIVES WHEN THE LOGICAL WIDTH DOES NOT MEET THE FLOOR: a windowed matmul whose
/// `CoreSplit` strands below the util floor still takes `for_output`'s bump — the drop is
/// conditional on the floor, not unconditional on `windowed`.
#[test]
fn the_windowed_drop_does_not_fire_below_the_util_floor() {
    // m=64, n=64, k=2048 — the granite tiled_k shape the issue measured: one output stick, so
    // `CoreSplit::plan(64, 64)` uses 64 rows over 1 stick = 32 cores... by ROWS. The floor is
    // met, so this would drop; instead take the shape that does NOT meet it: m=1 (decode) with
    // n=64: one row, one stick, `plan(1, 64)` = 1 core < 8, so the bump must fire even windowed.
    let m = 1;
    let n = 64;
    let k = 2048;
    assert!(
        ktir_superdsc::work::CoreSplit::plan(m, n).ncores() < UTIL_FLOOR_CORES,
        "the premise: a 1×64 output strands on one core, below the floor"
    );
    assert_eq!(
        DeviceWidth::for_matmul(m, n, k, true).get(),
        DeviceWidth::for_output(m, n, k).get(),
        "below the floor, windowed takes the SAME bumped width as staged — the drop exists to \
         skip a pad the caller's windows cannot honor, not to strand the gemm"
    );
}
