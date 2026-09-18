// SPDX-License-Identifier: Apache-2.0
//! THE Kᵀ RESTICKIFY'S TILE AND THE RUNTIME'S PER-STEP SHIFT ARE ONE QUANTITY — pinned off the
//! EMITTED descriptor's own `N_`, not off a re-derivation of the extents that were handed in.
//!
//! A decode step appends ONE slot, so exactly one of a page's `PAGE_SLOTS / STK` stick-blocks has a
//! stale Kᵀ. [`KtTileSlots::of_write_slab`] is the tile that covers that block; the runtime moves the
//! op onto the live one by shifting its KV-segment base (`fold_plan::slab_delta`) — ONE number, for
//! an op that reads natural K and writes Kᵀ through that one base.
//!
//! ⛔⛔⛔ SO THE TILE AND THE SHIFT MUST BE THE SAME SIZE, AND THEY ARE COMPUTED IN DIFFERENT PLACES.
//! The tile is what the emitter declares (`N_`); the shift is what the pool's address law says a
//! stick-block costs (`PagedKvPool::slab_shift_elems`). A shift SMALLER than the tile re-transposes
//! ground already covered and leaves the block's tail stale; a shift LARGER skips slots outright, and
//! either way the miss is a partially-stale Kᵀ — the shape that reads back as fluent wrong output
//! rather than as a crash. Nothing in the type system relates them, so it is asserted here.
//!
//! ⭐ AND THE ONE-SHIFT PRECONDITION IS ASSERTED AS AN EQUATION, NOT AS `hd == 64`. The two planes
//! stick on opposite axes, so a stick-block is the same distance in both only where `hd == STK`. That
//! is `slab_shift_elems`'s answer, and above it the incremental form must be REFUSED (`None`) so the
//! emitter falls back to the whole-page tile — which is what the `hd > 64` cases below pin.

#![cfg(feature = "spyre")]

use ktir_superdsc::emit::assemble_restickify_kt_2d;
use scratchy_subtile::addr::DevOff;
use scratchy_subtile::sdsc_abstract::{
    KtTileFeats, KtTileSlots, KvHead, PagedKvPool, POOL_STICK,
};
use std::num::NonZeroU32;

const NKVH: u32 = 8;

fn head0(nkvh: u32) -> KvHead {
    KvHead::new(0, NonZeroU32::new(nkvh).unwrap()).expect("head 0 is in range")
}

/// The emitted restickify's declared iteration extents, read out of the descriptor: `(out_, y_)` =
/// (feature width, slot width).
fn declared_extents(slots: KtTileSlots, hd: u32) -> (u64, u64) {
    let mut sym = 0i64;
    let e = assemble_restickify_kt_2d(
        "kt_probe",
        slots,
        KtTileFeats::of_head_dim(hd),
        "t_kc",
        DevOff::ZERO,
        "t_kct",
        DevOff::ZERO,
        &mut sym,
        None,
    );
    let v = serde_json::to_value(&e.op).unwrap();
    let n = &v["dscs_"][0]["kt_probe"]["N_"];
    (
        n["out_"].as_u64().expect("declared feature extent"),
        n["y_"].as_u64().expect("declared slot extent"),
    )
}

/// ⭐ THE TWO DOORS EMIT DIFFERENT DESCRIPTORS, and the slab one is EMITTABLE AT ALL.
///
/// That second half is not rhetorical: `restickify_kt_opspec_2d` puts both extents through
/// `StickExtent`, which refuses anything that is not a whole 64-element fp16 stick — and a sub-stick
/// Kᵀ extent is exactly what deleted the generic op walk this crate used to have. A 64-slot tile is
/// one whole stick, so it passes; this pins that it does, rather than assuming it.
#[test]
fn the_slab_door_declares_one_stick_block_and_the_page_door_declares_a_page() {
    let hd = POOL_STICK;
    assert_eq!(
        declared_extents(KtTileSlots::of_write_slab(), hd),
        (hd as u64, POOL_STICK as u64),
        "the slab door must declare ONE stick-block of slots"
    );
    assert_eq!(
        declared_extents(KtTileSlots::of_page(), hd),
        (hd as u64, PagedKvPool::PAGE_SLOTS as u64),
        "the page door must still declare the whole page"
    );
    // The whole point of the door, as a ratio: a decode step re-transposes a quarter of what it did.
    let (_, slab) = declared_extents(KtTileSlots::of_write_slab(), hd);
    let (_, page) = declared_extents(KtTileSlots::of_page(), hd);
    assert_eq!(
        page / slab,
        PagedKvPool::PAGE_SLOTS as u64 / POOL_STICK as u64,
        "a page is this many stick-blocks, and only one of them is ever stale"
    );
}

/// ⛔⛔⛔ THE RUNTIME'S PER-STEP SHIFT IS EXACTLY THE TILE THE EMITTER DECLARED.
///
/// `slab_shift_elems` is what `fold_plan::slab_delta` multiplies the block index by; `out_ * y_` is
/// what the op covers. Equal means step `p+STK` starts precisely where step `p`'s tile ended.
#[test]
fn the_shift_is_exactly_the_tile() {
    let hd = POOL_STICK;
    let pool = PagedKvPool::new(NKVH as usize, hd as usize);
    let shift = pool
        .slab_shift_elems(head0(NKVH))
        .expect("at hd == STK one shift serves both planes");
    let (out, y) = declared_extents(KtTileSlots::of_write_slab(), hd);
    assert_eq!(
        shift as u64,
        out * y,
        "the segment shift ({shift} elems) must equal the tile the op covers ({out} x {y}) — a \
         smaller shift leaves each block's tail stale, a larger one skips slots"
    );
}

/// ⛔ ABOVE ONE STICK THE INCREMENTAL FORM MUST BE REFUSED, so the emitter keeps the page tile.
///
/// Knat advances `STK` slots of `STK` elements each, INDEPENDENT of the head dim; Kᵀ advances one
/// slot-stick group of `hd * STK`. Those agree only at `hd == STK`, and a single segment shift cannot
/// serve two different strides — so `None` here is the emitter's instruction to fall back, and its
/// absence would be a silently half-shifted transpose at every head dim but one.
#[test]
fn no_single_shift_serves_both_planes_above_one_stick() {
    for hd in [POOL_STICK * 2, POOL_STICK * 4] {
        let pool = PagedKvPool::new(NKVH as usize, hd as usize);
        assert_eq!(
            pool.slab_shift_elems(head0(NKVH)),
            None,
            "hd={hd}: the two planes' stick-block strides differ, so the incremental form has no \
             single shift and must be refused"
        );
    }
    // And the refusal is about the RELATION, not about being large: at hd == STK it is granted.
    assert!(
        PagedKvPool::new(NKVH as usize, POOL_STICK as usize)
            .slab_shift_elems(head0(NKVH))
            .is_some()
    );
}

/// The grant does not depend on which kv head asks — a stick-block is a stick-block in every block.
#[test]
fn every_kv_head_gets_the_same_shift() {
    let pool = PagedKvPool::new(NKVH as usize, POOL_STICK as usize);
    let n = NonZeroU32::new(NKVH).unwrap();
    let first = pool.slab_shift_elems(head0(NKVH));
    for h in 0..NKVH {
        let head = KvHead::new(h, n).expect("in range");
        assert_eq!(
            pool.slab_shift_elems(head),
            first,
            "kv head {h} must take the same shift — a fused Slab group shares ONE stride"
        );
    }
}
