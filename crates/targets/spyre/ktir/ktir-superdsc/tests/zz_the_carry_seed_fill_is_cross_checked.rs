// SPDX-License-Identifier: Apache-2.0

//! ⭐⭐⭐⭐⭐ THE CARRY SEED'S CROSS-CHECK — the sentinel pair's door, driven through the real
//! walk.
//!
//! The paged fold's max carry is seeded by the KERNEL's own constant (`paged_fold.py`'s
//! `tl.full([BATCH * MQ, 1], -65504.0, tl.float16)`), while the validity lanes the HOST stages
//! carry [`ATTN_SEED_FLOOR`] — the finite f16 floor. Those are TWO independent spellings of ONE
//! value, and the pair is load-bearing: a masked lane's score saturates at the floor, so its `p`
//! is exactly 0 only while `m` sits above the seed the lanes name (the seed-safety law whose
//! edge the row-sweep readings guard). A kernel whose stated seed drifted from the lanes'
//! sentinel changes the expunge margins in the dangerous direction — fluent output, never a
//! fault — so the whole-function walk's FILL arm refuses the desync, naming both values.
//!
//! Every program here is built from the op kinds the walk really reads (the `zz_the_bound_scale`
//! construction idiom), bound to the carry's own reserved tid, and lowered through
//! [`lower_function`] — the same arm the real seed program rides. What is pinned:
//!
//! * THE REFUSAL — a fill of the max carry at a drifted value is refused naming BOTH the stated
//!   value and the floor, before any registry or placement is consulted;
//! * THE PASS — the fill at the floor lowers through the fill arm exactly as the real seed does
//!   (registry slot + slot-tid placement, the fail-closed prerequisites the emission needs);
//! * THE SCOPE — the door is the MAX carry's alone: the Sum carry's fill (the additive identity,
//!   a value only the kernel spells — no second spelling exists to desync from) lowers
//!   unrefused at any registered value.

use ktir_core::arena::Arena;
use ktir_core::attrkey::AttrKey;
use ktir_core::dtypes::DType;
use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
use ktir_core::irtype::IrType;
use ktir_core::opkind::OpKind;
use ktir_superdsc::emit::whole_function::lower_function;
use ktir_superdsc::ktir_node::{BufferId, KtirNode, Program};
use ktir_superdsc::place::act_name;
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
use ktir_superdsc::reserved_tids::{PagedFoldCarry, scalarmul_scale_tid};
use ktir_superdsc::sdsc_abstract::ATTN_SEED_FLOOR;
use ktir_superdsc::superdsc_opspec::Df;

/// The seed program's own shape — `[BATCH * MQ, 1]` f16, the frame `paged_fold.py`'s
/// `paged_fold_seed_hm` stores its `tl.full` into: one param (the carry being filled), a view,
/// a `[0, 0]` access tile, a splatted float constant, and the store. `fill` is the constant the
/// program states — the one variable of the door's tests; `tid` is the buffer the param binds.
fn seed_program(fill: f64, tid: u32) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };

    // Parameter: %0 = the carry being filled.
    let p_m = Ssa(0);
    let zero = Ssa(1);
    // The chain: view, [0, 0] access tile, the constant, its splat, the store.
    let (v_m, t_m) = (Ssa(2), Ssa(3));
    let (cst, splat) = (Ssa(4), Ssa(5));

    let ops = vec![
        Operation::new(a, Some(zero), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(0),
        ),
        shape(
            Operation::new(a, Some(v_m), OpKind::KtdpConstructMemoryView, &[p_m]),
            32,
            1,
        ),
        shape(
            Operation::new(
                a,
                Some(t_m),
                OpKind::KtdpConstructAccessTile,
                &[v_m, zero, zero],
            ),
            32,
            1,
        ),
        Operation::new(a, Some(cst), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Float(fill),
        ),
        shape(
            Operation::new(a, Some(splat), OpKind::TensorSplat, &[cst]),
            32,
            1,
        ),
        Operation::new(a, None, OpKind::KtdpStore, &[splat, t_m]),
    ];

    KtirNode {
        func: IRFunction {
            name: "seed_fill_probe",
            arguments: a.args(vec![(p_m, IrType::Index)]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        },
        program: Program::Attn,
        bindings: vec![BufferId::new(tid)],
        mask: None,
        step_inputs: Vec::new(),
        node_out_tid: None,
        minted: Default::default(),
    }
}

/// The layout the fill arm needs: the carry's own placement (a whole-stick-padded window, the
/// extent `place_paged_fold_carry` reserves for the real carry), the registry holding the fill's
/// value, and the registry slot's own reserved-tid placement — the fail-closed prerequisites
/// `scalarfill_at` resolves, exactly as the bound-scale test declares them.
fn layout_for(k: &KtirNode, scales: Vec<f32>) -> BundleLayout {
    let mut l = BundleLayout::default();
    let tid = k.bindings[0].get();
    l.ids
        .borrow_mut()
        .insert(act_name(tid), ktir_superdsc::place::PlaceId::Act(tid));
    // [32, 1] f16 padded to a whole stick, the placement the fill's width law reads.
    l.placements.insert(
        tid,
        TensorPlacement {
            tid,
            role: SegRole::Activation,
            segment: 3,
            bank: 0,
            offset: 0,
            size: 32 * 64 * 2,
            cover: ktir_superdsc::placement::BindCover::Full,
        },
    );
    l.scalarmul_scales = scales;
    l.scalarmul_scales_df = vec![Df::Fp16];
    let slot_tid = scalarmul_scale_tid(0);
    l.ids.borrow_mut().insert(
        act_name(slot_tid),
        ktir_superdsc::place::PlaceId::Act(slot_tid),
    );
    l.placements.insert(
        slot_tid,
        TensorPlacement {
            tid: slot_tid,
            role: SegRole::Activation,
            segment: 2,
            bank: 0,
            offset: 0,
            size: 2,
            cover: ktir_superdsc::placement::BindCover::Full,
        },
    );
    l
}

/// ⛔ THE REFUSAL — a seed fill that drifted from the floor is the desync the door exists to
/// catch, and it fires BEFORE any registry or placement is consulted: the pair's disagreement
/// is knowable from the program and the const alone.
#[test]
fn a_drifted_seed_fill_is_refused_naming_both_values() {
    let k = seed_program(-60000.0, PagedFoldCarry::Max.tid());
    let mut sid = 0i64;
    let Err(e) = lower_function(&k, None, &mut sid, None) else {
        panic!("a max-carry fill at a drifted value must refuse")
    };
    assert!(
        e.message.contains("-60000") && e.message.contains("-65504"),
        "names the stated value AND the floor: {}",
        e.message
    );
    assert!(
        e.message.contains("seed_fill_probe"),
        "names the program: {}",
        e.message
    );
}

/// THE PASS — the fill at the floor lowers through the fill arm exactly as the real seed
/// program does: one op, the identity broadcast of the registry const over the carry's window.
#[test]
fn the_floor_seed_lowers_through_the_fill_arm() {
    let k = seed_program(-65504.0, PagedFoldCarry::Max.tid());
    let mut sid = 0i64;
    let ops = lower_function(
        &k,
        Some(&layout_for(&k, vec![ATTN_SEED_FLOOR.to_f32_const()])),
        &mut sid,
        None,
    )
    .expect("the floor-valued seed fill lowers through the registry arm");
    assert!(
        !ops.is_empty(),
        "the fill lowers as a descriptor, not a silent skip"
    );
}

/// ⛔ THE SCOPE CONTROL — the door is the MAX carry's alone. The Sum carry seeds at the
/// additive identity 0, a value only the kernel spells: no second spelling exists to desync
/// from, so its fill is not the door's business and lowers unrefused at any registered value.
#[test]
fn the_door_is_the_max_carrys_alone() {
    let k = seed_program(-60000.0, PagedFoldCarry::Sum.tid());
    let mut sid = 0i64;
    let ops = lower_function(&k, Some(&layout_for(&k, vec![-60000.0])), &mut sid, None)
        .expect("a non-Max carry's fill is not the seed door's business");
    assert!(
        !ops.is_empty(),
        "the Sum carry's fill still lowers — the door did not fire"
    );
}
