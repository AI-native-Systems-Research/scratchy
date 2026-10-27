// SPDX-License-Identifier: Apache-2.0
//! ⭐ A PAGE PART NEVER SHARES A GROUP WITH ANOTHER PART — the group law for a prompt window wider
//! than a page, AS A TYPE.
//!
//! A group is one launch, and a launch shifts the KV segment base ONCE, to the page holding the group's
//! cursor plus its part. A 512-row chunk's cache write is two parts aimed at two pages, so a group
//! holding ops of both would write one part's rows onto the other's page — a cache write nothing
//! downstream can detect. The guard for that used to be a runtime offset comparison in the fusion walk
//! PLUS an `assert!` over every op of every group in `launch_index`, re-reading the owner op's field —
//! two derivations that could only be proven to agree by running them. It is a TYPE now
//! ([`PagePart`], minted only by `PagedKvPool::write_parts`, compared by [`Trip::fusable_with`],
//! carried to the launch off the walk's own trip) and the assert is deleted. These pin what the type
//! has to deliver: the cut's own parts, the walk's break over them for every kind, and the shipped
//! emission's launch index.

use ktir_superdsc::emit::lower_ktir_to_superdsc::{KvCacheWrite, kv_cache_write};
use ktir_superdsc::page_part::Slots;
use ktir_superdsc::score_scale::NewK;
use ktir_superdsc::sdsc_abstract::{PagedKvPool, QueryRowCount, WriteRowStart};

/// One page, in slots — the cut's own unit.
const PAGE: u32 = PagedKvPool::PAGE_SLOTS as u32;
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::{
    FoldGrouping, GroupKind, Trip, TripRequest, group_ranges, launch_index,
};

/// EVERY kind breaks at a part change — the cache write's `Slot` parts and the Kᵀ re-transpose's
/// `Pure` parts alike — while one part's own run still fuses.
#[test]
fn no_kind_fuses_across_page_parts() {
    // The parts of a 512-row window, off the ONE cutting law: a part is what the cut says it is,
    // and this test cannot spell one from a number either.
    let cut = PagedKvPool::new(1, 64).write_parts(
        WriteRowStart::at_cursor(),
        QueryRowCount::of_mq(512),
        false,
    );
    for kind in [
        GroupKind::Pure,
        GroupKind::Slot {
            req: TripRequest(0),
        },
        GroupKind::Slab,
        GroupKind::Gathered,
    ] {
        let trips = vec![
            Trip::at_part(kind, TripRequest(0), cut[0].page),
            Trip::at_part(kind, TripRequest(0), cut[0].page),
            Trip::at_part(kind, TripRequest(0), cut[1].page),
            Trip::at_part(kind, TripRequest(0), cut[1].page),
        ];
        assert_eq!(
            group_ranges(&trips, 512),
            vec![0..2, 2..4],
            "{kind:?}: two page parts fused into one launch — one page shift cannot address both"
        );
    }
    // `Trip::new` is the cursor-page trip, and the cut's part 0 IS the cursor page (its row 0), so
    // a part-0 op still fuses with an untagged neighbour.
    let trips = vec![
        Trip::new(GroupKind::Pure, TripRequest(0)),
        Trip::at_part(GroupKind::Pure, TripRequest(0), cut[0].page),
    ];
    assert_eq!(group_ranges(&trips, 512), vec![0..2]);
}

/// THE SHIPPED EMISSION'S LAUNCH INDEX: a 512-row chunk's cache write is one group per part and its Kᵀ
/// re-transpose one group per part, each carrying its part's offset, in part order — at granite-2b
/// (one slab) and granite-8b (two).
#[test]
fn a_two_page_chunks_write_is_one_launch_per_part() {
    for hd in [64u32, 128] {
        let mut sym = 0i64;
        let ops = kv_cache_write(
            KvCacheWrite {
                t: ktir_superdsc::place::PlaceId::Act(7),
                mq: QueryRowCount::of_mq(512),
                pool: PagedKvPool::new(8, hd),
                rows_are_requests: false,
                // The door's own main-call spelling: the window starts at the cursor's row 0.
                row_start: ktir_superdsc::sdsc_abstract::WriteRowStart::at_cursor(),
                new_k: NewK::reader_stated("t_newk"),
                new_v: "t_newv",
                kc: "t_kc",
                vc: "t_vc",
                kct: "t_kct",
                ident: "t_ident",
            },
            &mut sym,
            None,
        )
        .unwrap_or_else(|e| panic!("hd={hd}: the cache write lowers: {}", e.message));
        let idx = launch_index(&ops, FoldGrouping::Split);
        let shape: Vec<(bool, Slots)> = idx
            .iter()
            .map(|g| (g.slot_write(), g.part.slots()))
            .collect();
        assert_eq!(
            shape,
            vec![
                (true, Slots::of(0)),
                (true, Slots::of(PAGE)),
                (false, Slots::of(0)),
                (false, Slots::of(PAGE)),
            ],
            "hd={hd}: cache write part 0, part 1, then the Kᵀ re-transpose's two parts"
        );
        // And a one-page chunk keeps the launches it always had: no part-1 group at all.
        let one = kv_cache_write(
            KvCacheWrite {
                t: ktir_superdsc::place::PlaceId::Act(7),
                mq: QueryRowCount::of_mq(256),
                pool: PagedKvPool::new(8, hd),
                rows_are_requests: false,
                // The door's own main-call spelling: the window starts at the cursor's row 0.
                row_start: ktir_superdsc::sdsc_abstract::WriteRowStart::at_cursor(),
                new_k: NewK::reader_stated("t_newk"),
                new_v: "t_newv",
                kc: "t_kc",
                vc: "t_vc",
                kct: "t_kct",
                ident: "t_ident",
            },
            &mut sym,
            None,
        )
        .unwrap_or_else(|e| panic!("hd={hd}: the cache write lowers: {}", e.message));
        assert!(
            launch_index(&one, FoldGrouping::Split)
                .iter()
                .all(|g| g.part.is_cursor()),
            "hd={hd}: a one-page chunk has no page part"
        );
    }
}
