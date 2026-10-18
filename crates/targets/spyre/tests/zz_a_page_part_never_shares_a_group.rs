// SPDX-License-Identifier: Apache-2.0
//! ⭐ A PAGE PART NEVER SHARES A GROUP WITH ANOTHER PART — the group law for a prompt window wider
//! than a page.
//!
//! A group is one launch, and a launch shifts the KV segment base ONCE, to the page holding the group's
//! cursor plus its slot offset. A 512-row chunk's cache write is two parts aimed at two pages
//! (`EmittedOp::kv_slot_offset` 0 and 256), so a group holding ops of both would write one part's rows
//! onto the other's page — a cache write nothing downstream can detect. [`Trip::fusable_with`] checks
//! the offset beside the request, for every kind; these pin it on bare trips and on the shipped
//! emission's launch index.

use ktir_superdsc::emit::lower_ktir_to_superdsc::{KvCacheWrite, kv_cache_write};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::{
    FoldGrouping, GroupKind, Trip, TripRequest, group_ranges, launch_index,
};

/// EVERY kind breaks at an offset change — the cache write's `Slot` parts and the Kᵀ re-transpose's
/// `Pure` parts alike — while one part's own run still fuses.
#[test]
fn no_kind_fuses_across_page_parts() {
    for kind in [
        GroupKind::Pure,
        GroupKind::Slot { req: 0 },
        GroupKind::Slab,
        GroupKind::Gathered,
    ] {
        let trips = vec![
            Trip::at_slot_offset(kind, TripRequest(0), 0),
            Trip::at_slot_offset(kind, TripRequest(0), 0),
            Trip::at_slot_offset(kind, TripRequest(0), 256),
            Trip::at_slot_offset(kind, TripRequest(0), 256),
        ];
        assert_eq!(
            group_ranges(&trips, 512),
            vec![0..2, 2..4],
            "{kind:?}: two page parts fused into one launch — one page shift cannot address both"
        );
    }
    // `Trip::new` is the offset-0 trip, so a part-0 op still fuses with an untagged neighbour.
    let trips = vec![
        Trip::new(GroupKind::Pure, TripRequest(0)),
        Trip::at_slot_offset(GroupKind::Pure, TripRequest(0), 0),
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
                t: 7,
                mq: 512,
                nkvh: 8,
                hd,
                rows_are_requests: false,
                new_k: "t_newk",
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
        let shape: Vec<(bool, u32)> = idx
            .iter()
            .map(|g| (g.slot_write(), g.slot_offset))
            .collect();
        assert_eq!(
            shape,
            vec![(true, 0), (true, 256), (false, 0), (false, 256)],
            "hd={hd}: cache write part 0, part 1, then the Kᵀ re-transpose's two parts"
        );
        // And a one-page chunk keeps the launches it always had: no part-1 group at all.
        let one = kv_cache_write(
            KvCacheWrite {
                t: 7,
                mq: 256,
                nkvh: 8,
                hd,
                rows_are_requests: false,
                new_k: "t_newk",
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
                .all(|g| g.slot_offset == 0),
            "hd={hd}: a one-page chunk has no page part"
        );
    }
}
