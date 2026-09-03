// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "superdsc")]
//! MODEL PARAMS DRIVE ADDRESSING — the property, not an instance of it.
//!
//! Every addressing bug this week was a stride written by hand from the one model that had ever run:
//! `DOWNPROJ_B = 16` is 2b's `K/512`; `b*stick*hd` is the V block stride only at one stick per head;
//! `mq_pad*stick` is the slab stride only when padded and real rows coincide. All correct for 2b, all
//! silently wrong for 8b, none reachable by a test.
//!
//! These lock the property that makes that class unwritable: the SAME expression, instantiated at both
//! head_dims, must produce each model's correct address. A test that passes at one head_dim and needs a
//! branch at the other has reintroduced the special case.

use scratchy_subtile::addr::{Idx, Rows, Shape, Slot};
use scratchy_subtile::superdsc_opspec::Df;

/// Slab count is DERIVED from (head_dim, format) -- never a constant, and never branched on. At one
/// stick per head it degenerates to 1 and every slab term vanishes, which is exactly why hd=64 models
/// never exercised them: same path, collapsed arithmetic.
#[test]
fn slab_count_is_derived_at_every_head_dim() {
    assert_eq!(Shape::<64, 32, 8, 128>::slabs(Df::Fp16).get(), 1);
    assert_eq!(Shape::<128, 32, 8, 128>::slabs(Df::Fp16).get(), 2);
    assert_eq!(Shape::<256, 32, 8, 128>::slabs(Df::Fp16).get(), 4);
    // fp8 has twice the lanes, so the SAME head_dim has half the slabs — a property of the pair.
    assert_eq!(Shape::<128, 32, 8, 128>::slabs(Df::Fp8).get(), 1);
}

/// THE V CACHE BUG, as a property. `vc` is `[cap, hd]` per head, stick-major on the feature axis, so a
/// 64-slot block advances by one stick of SLOTS. That is independent of head_dim -- which is precisely
/// what the hand-written `b*stick*hd` got wrong, by scaling a slot stride by a feature extent.
#[test]
fn v_cache_block_stride_does_not_depend_on_head_dim() {
    let a = Shape::<64, 32, 8, 128>::v_cache(Df::Fp16)
        .view()
        .at(Idx::<Slot>::n(64))
        .dev();
    let b = Shape::<128, 32, 8, 128>::v_cache(Df::Fp16)
        .view()
        .at(Idx::<Slot>::n(64))
        .dev();
    assert_eq!(
        a.into_raw_elems(),
        b.into_raw_elems(),
        "a slot stride cannot scale with head_dim"
    );
    assert_eq!(a.into_raw_elems(), 64 * 64, "one stick of slots");
}

/// GQA grouping is derived too. `DOWNPROJ_B = 16` was the same mistake in a different place: a ratio
/// between two model params, frozen as a literal from one model.
#[test]
fn gqa_group_is_derived_from_the_head_counts() {
    assert_eq!(Shape::<64, 32, 8, 128>::gqa().get(), 4);
    assert_eq!(Shape::<128, 32, 8, 128>::gqa().get(), 4);
    assert_eq!(Shape::<128, 32, 32, 128>::gqa().get(), 1);
}

/// The token stream at both head_dims: head `h` is a COORDINATE, so its address comes out of the nest
/// rather than an `h * mq * hd` a caller wrote. At mq>1 and hd>one stick those two disagree, which is
/// the replicate-copy bug.
#[test]
fn head_base_comes_from_the_nest_at_both_head_dims() {
    for (hd, nest) in [
        (
            64u32,
            Shape::<64, 32, 8, 128>::kv_stream(Rows::new(31), Df::Fp16),
        ),
        (
            128,
            Shape::<128, 32, 8, 128>::kv_stream(Rows::new(31), Df::Fp16),
        ),
    ] {
        let slabs = hd / 64;
        for kvh in 0..8u32 {
            let got = nest
                .view()
                .at(Idx::<scratchy_subtile::addr::Head>::n(kvh))
                .dev()
                .into_raw_elems();
            assert_eq!(
                got,
                (kvh * slabs) * 31 * 64,
                "kv head {kvh} at hd={hd} is packed by the REAL row count"
            );
        }
    }
}

/// The precondition the emitter's three fast paths rest on, stated as a property. Each collapses a
/// per-head loop into one whole-tensor op by treating `[rows, heads*head_dim]` as `[heads*rows,
/// head_dim]`. That identity holds ONLY when a head is exactly one stick, which is why those paths
/// were gated on `head_dim == 64` — not arbitrarily, but the gate said the number instead of the
/// reason, so nothing connected it to the format's lane count.
#[test]
fn head_major_collapse_is_valid_exactly_at_one_stick_per_head() {
    type S = Shape<0, 0, 0, 0>;
    assert!((64 == Df::Fp16.elems_per_stick()), "one fp16 stick");
    assert!(
        !(128 == Df::Fp16.elems_per_stick()),
        "two sticks — the collapse is not byte-identical"
    );
    assert!(!(32 == Df::Fp16.elems_per_stick()), "half a stick");
    // fp8 has 128 lanes, so the SAME head_dim flips the answer — which a `== 64` test could not do.
    assert!((128 == Df::Fp8.elems_per_stick()));
    assert!(!(64 == Df::Fp8.elems_per_stick()));
}

/// And the identity itself, exhaustively, so the predicate above is anchored to the address law rather
/// than to my reading of it.
#[test]
fn the_collapse_identity_holds_exactly_where_the_predicate_says() {
    let (mq, heads, stk) = (31usize, 32usize, 64usize);
    for hd in [64usize, 128] {
        let same = (0..heads).all(|h| {
            (0..mq).all(|r| {
                (0..hd).all(|d| {
                    let stream = ((h * hd + d) / stk) * (mq * stk) + r * stk + d % stk;
                    let major = (d / stk) * (heads * mq * stk) + (h * mq + r) * stk + d % stk;
                    stream == major
                })
            })
        });
        assert_eq!(
            same,
            (hd as u32 == Df::Fp16.elems_per_stick()),
            "the predicate must agree with the addresses at head_dim {hd}"
        );
    }
}

/// `Nest::block(i)` — several device tensors are `n` independent blocks of one shape (`vc` is nqh
/// separate `[cap, hd]`, `new_kt` is nkvh separate `[hd, mq_pad]`). Describing them as one rank-3 nest
/// gives the outer axis a stride of ONE STICK instead of the block's footprint, which is the mistake
/// that produced `b*stick*hd` for the V cache. The block index must step by `elems()`, at every
/// head_dim, and the coordinate inside must be unaffected by which block it is in.
#[test]
fn block_index_steps_by_the_blocks_own_footprint() {
    use scratchy_subtile::addr::{Nest, Slot};
    for hd in [64u32, 128] {
        let cap = 128u32;
        let blk = Nest::new(&["slot", "feat"], &[cap, hd], Df::Fp16);
        assert_eq!(blk.elems(), (cap * hd) as i64);
        for i in 0..4u32 {
            let base = blk.block(i).dev().into_raw_elems();
            assert_eq!(base, i * cap * hd, "block {i} at hd={hd}");
            // slot window 64 inside block i — one stick of SLOTS, independent of head_dim.
            let win = blk.block(i).at(Idx::<Slot>::n(64)).dev().into_raw_elems();
            assert_eq!(
                win - base,
                64 * 64,
                "a slot window cannot scale with head_dim"
            );
        }
    }
}

/// A nest assigns device slots by POSITION, not by NAME: axis 0 takes the row slot, the last axis is
/// the stick, anything between is outer. That made the names decorative — `["head","row","feat"]` reads
/// as head-major and addresses head as the ROW, giving it a stride of one lane instead of `mq` sticks.
///
/// I wrote exactly that twice while converting this code, for the online-softmax buffers and for
/// `new_kt`, and only a 22-of-23-bundle bake diff caught it. A misleading order must now fail loudly at
/// construction rather than produce a plausible wrong address.
#[test]
#[should_panic(expected = "`row` is at index 1")]
fn a_nest_that_names_row_out_of_the_row_slot_is_rejected() {
    let _ = scratchy_subtile::addr::Nest::new(&["head", "row", "feat"], &[32, 31, 128], Df::Fp16);
}

/// The orders the emitter actually uses must all still be accepted — including the transposed K caches,
/// whose stick axis is `slot` rather than `feat`.
#[test]
fn the_orders_the_emitter_uses_are_all_legal() {
    use scratchy_subtile::addr::Nest;
    let _ = Nest::new(&["row", "feat"], &[31, 4096], Df::Fp16);
    let _ = Nest::new(&["row", "head", "feat"], &[31, 32, 128], Df::Fp16);
    let _ = Nest::new(&["slot", "head", "feat"], &[128, 32, 128], Df::Fp16);
    let _ = Nest::new(&["feat", "slot"], &[128, 128], Df::Fp16);
    let _ = Nest::new(&["feat", "head", "slot"], &[128, 8, 128], Df::Fp16);
}
