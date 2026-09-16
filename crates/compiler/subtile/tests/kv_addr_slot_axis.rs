// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "spyre")]
//! THE POOL'S SLOT AXIS, AT ANY HEAD DIM — what the deleted request dimension turned into.
//!
//! ⛔⛔⛔ THIS FILE REPLACES `kv_addr_request_dim.rs`, WHICH HAD BEEN BREAKING THE WHOLE TEST BUILD.
//! That file was written against `RequestInPage` and `PagedKvPool::request_stride`, both deleted by
//! `cc4d5254` ("a page holds SLOTS, not requests") — so `cargo test -p scratchy-subtile --tests` has
//! failed to COMPILE ever since, and every "tests green" in that window came from single-target
//! `--test <name>` runs, which build only their own target. A stale test file is not a dormant test; it
//! is a hole in every other test in the crate.
//!
//! ⭐ AND THE DELETION IS THE POINT. The pool coordinate is `(plane, kvh, slot, feat)` with the slot
//! ABSOLUTE: there is no request below the host, and request identity lives only in the host mask. The
//! old file's headline finding — "a request IS a page of slots, so the fold needs a wider sweep and not
//! a launch per request" — is no longer a property to test, because the alternative it argued against
//! cannot be written down any more. What survives translation is the SLOT and FEATURE geometry, which is
//! where the two plane families part company above head_dim 64.

use scratchy_subtile::sdsc_abstract::{FeatIdx, KvCoord, KvHead, KvPlane, KvSlot, PagedKvPool};
use std::num::NonZeroU32;

const NKVH: usize = 8;

fn kvh(i: u32) -> KvHead {
    KvHead::new(i, NonZeroU32::new(NKVH as u32).unwrap()).expect("kv head in range")
}

/// THE THREE PLANES DISAGREE ONLY IN THEIR INTRA-BLOCK ORDER, and that is the whole reason Kᵀ exists.
///
/// Kᵀ is `[hd, PAGE_SLOTS]` sticked on the SLOT axis, so one slot costs one lane and one feature costs a
/// whole stick. Natural K and V are `[PAGE_SLOTS, hd]` sticked on the FEATURE axis, so it is exactly the
/// other way round. Pinned at every head dim, because this is the transposition the score leg depends on.
#[test]
fn a_slot_costs_a_lane_on_kt_and_a_stick_on_v() {
    for hd in [64usize, 80, 128, 256] {
        let p = PagedKvPool::new(NKVH, hd);
        let stick = 64u32;
        for h in [0u32, 3, 7] {
            let kt = KvCoord::block(KvPlane::Kt, kvh(h));
            let v = KvCoord::block(KvPlane::V, kvh(h));
            let knat = KvCoord::block(KvPlane::Knat, kvh(h));
            // One slot on.
            assert_eq!(
                p.addr(kt.at_slot(KvSlot::new(1))) - p.addr(kt),
                1,
                "hd={hd}: Kᵀ slot is a lane"
            );
            assert_eq!(
                p.addr(v.at_slot(KvSlot::new(1))) - p.addr(v),
                stick,
                "hd={hd}: V slot is a stick"
            );
            assert_eq!(
                p.addr(knat.at_slot(KvSlot::new(1))) - p.addr(knat),
                stick,
                "hd={hd}: natural K slot is a stick"
            );
            // One feature on — the mirror image.
            assert_eq!(
                p.addr(kt.at_feat(FeatIdx::new(1))) - p.addr(kt),
                stick,
                "hd={hd}: Kᵀ feat is a stick"
            );
            assert_eq!(
                p.addr(v.at_feat(FeatIdx::new(1))) - p.addr(v),
                1,
                "hd={hd}: V feat is a lane"
            );
        }
    }
}

/// ⚠️⚠️ ABOVE head_dim 64 THE TWO PLANE FAMILIES PART, AND THIS IS THE MEASUREMENT THAT SAYS SO.
///
/// The old file established this as "at hd=128 only Kᵀ stays contiguous across requests". With the
/// request gone the same fact is a statement about the SLOT axis: how far is a full page of slots, versus
/// the plane's own footprint?
///
/// Kᵀ sticks on the slot axis, so `PAGE_SLOTS` slots is `(PAGE_SLOTS/64)*(hd*64)` = `hd*PAGE_SLOTS` —
/// the plane's entire footprint, at ANY head dim. The score leg's kernel is therefore one contiguous
/// run whatever the model.
///
/// Natural K and V stick on the FEATURE axis, so `PAGE_SLOTS` slots is `PAGE_SLOTS*64`, INDEPENDENT of
/// head_dim, while the footprint grows with it. ⭐ At hd == 64 those two coincide by arithmetic accident
/// — which is exactly why hd=64 could never see this — and at hd == 128 the footprint is twice a page of
/// slots. The value leg cannot simply sweep wider.
#[test]
fn only_kt_spans_its_whole_plane_with_slots_at_any_head_dim() {
    for hd in [64usize, 128, 256] {
        let p = PagedKvPool::new(NKVH, hd);
        let footprint = p.plane_block_elems() as u32;
        for plane in [KvPlane::Kt, KvPlane::V, KvPlane::Knat] {
            let base = KvCoord::block(plane, kvh(0));
            let a_page_of_slots =
                p.addr(base.at_slot(KvSlot::new(PagedKvPool::PAGE_SLOTS as u32))) - p.addr(base);
            let spans = a_page_of_slots == footprint;
            match (hd, plane) {
                // The slot axis carries the stick, so it holds at every head dim.
                (_, KvPlane::Kt) => assert!(
                    spans,
                    "hd={hd}: Kᵀ's page of slots ({a_page_of_slots}) must be its whole plane ({footprint})"
                ),
                // At one stick the feature term vanishes and everything coincides.
                (64, _) => assert!(
                    spans,
                    "hd=64 plane {plane:?}: expected the accidental coincidence"
                ),
                (_, _) => assert!(
                    !spans,
                    "hd={hd} plane {plane:?}: a page of slots is {a_page_of_slots} and the plane is \
                     {footprint} — if these are equal the feature-stick term has gone missing again"
                ),
            }
        }
    }
}

/// ⛔⛔⛔ NO TWO SLOTS OF A PLANE MAY SHARE AN ADDRESS, and every slot must land inside the plane.
///
/// ⚠️ I FIRST WROTE THIS AS "the slot axis is linear" AND IT FAILED, CORRECTLY: on Kᵀ at hd=64, slot 64
/// is 4096 away, not 64. The slot axis is STICK-BLOCKED, so it is piecewise — within a stick one slot is
/// one lane, and crossing a stick boundary jumps a whole stick plane. Linearity was my assumption, not
/// the layout's law.
///
/// The right property is also the one that matters, and it needs no address law at all: the map from slot
/// to address must be INJECTIVE and land inside the plane's footprint. Two slots sharing an address is
/// one row's keys silently overwriting another's — the exact shape of the batch defect — and it would be
/// invisible to any test that checks a stride. Asserting injectivity instead of a formula also keeps this
/// file from restating the arithmetic it is supposed to be checking, which is how the guard ends up being
/// a copy of the thing it guards.
#[test]
fn no_two_slots_of_a_plane_can_alias() {
    for hd in [64usize, 80, 128, 256] {
        let p = PagedKvPool::new(NKVH, hd);
        let footprint = p.plane_block_elems() as u32;
        for plane in [KvPlane::Kt, KvPlane::V, KvPlane::Knat] {
            for h in [0u32, 5, 7] {
                let base = KvCoord::block(plane, kvh(h));
                let mut seen: std::collections::BTreeMap<u32, u32> = Default::default();
                for s in 0..PagedKvPool::PLANE_SLOTS as u32 {
                    let a = p.addr(base.at_slot(KvSlot::new(s)));
                    let off = a - p.addr(base);
                    assert!(
                        off < footprint,
                        "hd={hd} plane {plane:?} kvh {h} slot {s}: offset {off} escapes the plane's \
                         {footprint} elements — it lands in another kv head's keys"
                    );
                    if let Some(prev) = seen.insert(a, s) {
                        panic!(
                            "hd={hd} plane {plane:?} kvh {h}: slots {prev} and {s} share address {a} — \
                             one row's keys would overwrite another's"
                        );
                    }
                }
                assert_eq!(
                    seen.len(),
                    PagedKvPool::PLANE_SLOTS,
                    "hd={hd} plane {plane:?}: every slot must have its own address"
                );
            }
        }
    }
}

/// AND THE SAME FOR THE FULL (slot, feat) GRID — the pair is what an op actually names, and a collision
/// between two DIFFERENT pairs is the head_dim>64 failure mode: at one stick the feature term is 0, so
/// every feature aliases onto feature 0 and nothing notices until hd exceeds a stick.
#[test]
fn no_two_slot_feature_pairs_can_alias() {
    for hd in [64usize, 128, 256] {
        let p = PagedKvPool::new(NKVH, hd);
        for plane in [KvPlane::Kt, KvPlane::V, KvPlane::Knat] {
            let base = KvCoord::block(plane, kvh(2));
            let mut seen: std::collections::BTreeMap<u32, (u32, u32)> = Default::default();
            for s in 0..PagedKvPool::PLANE_SLOTS as u32 {
                for f in 0..hd as u32 {
                    let a = p.addr(base.at_slot(KvSlot::new(s)).at_feat(FeatIdx::new(f)));
                    if let Some(prev) = seen.insert(a, (s, f)) {
                        panic!(
                            "hd={hd} plane {plane:?}: (slot {}, feat {}) and (slot {s}, feat {f}) both \
                             address {a}",
                            prev.0, prev.1
                        );
                    }
                }
            }
            assert_eq!(
                seen.len(),
                PagedKvPool::PLANE_SLOTS * hd,
                "hd={hd} plane {plane:?}: the (slot, feat) grid must be a bijection onto the plane"
            );
        }
    }
}

/// ⛔ A KV HEAD PAST `nkvh` HAS NO COORDINATE. The old file made this point about a request past a page's
/// rows; with the request gone, the head is the axis whose overrun would land silently on another head's
/// keys. `KvHead::new` is the only door and it refuses.
#[test]
fn a_kv_head_past_nkvh_cannot_be_named() {
    let n = NonZeroU32::new(NKVH as u32).unwrap();
    assert!(KvHead::new(NKVH as u32, n).is_none());
    assert!(KvHead::new(NKVH as u32 + 1, n).is_none());
    assert!(KvHead::new(NKVH as u32 - 1, n).is_some());
}
