//! A FUSION GROUP IS ONE REQUEST'S — the law that ends the batched-decode whack-a-mole.
//!
//! A group is one launch. A launch resolves ONE request's page table and write cursor, and shifts the
//! KV segment base ONCE. So a group holding trips of two requests hands the second request the
//! FIRST one's KV: no crash, no shape error, fluent output, wrong tokens. That symptom — a batch of 8
//! answering two or three good tokens and then decaying — is what these tests pin.
//!
//! Before [`Trip`], the group walk saw only [`GroupKind`], and only the `Slot` variant happened to
//! carry a request. Every other kind fused across requests, and the attention matmuls (which READ
//! the KV pages) plus the per-page Kᵀ re-transpose are all `Pure` — so at the shipped g=512 they
//! fused straight across all 8 requests of a batch and every row read row 0's page. Six earlier bugs
//! of the same shape were fixed one site at a time; this is the type that removes the possibility.
//!
//! Kani proves the same property for ALL trip sequences (`no_group_spans_two_requests`); these tests
//! are the fast concrete twin — and they are what a mutation check flips red in seconds.

use scratchy_target_spyre::lower_subtile_tape_to_superdsc::{
    GroupKind, Trip, TripRequest, group_ranges, group_ranges_cover_ok, group_spans_two_requests,
};

fn pure(r: u32) -> Trip {
    Trip::new(GroupKind::Pure, TripRequest(r))
}

/// THE BUG: eight requests' worth of `Pure` attention work in a row, at the shipped group size.
///
/// Fusing on kind alone made this ONE group — one launch, one page shift, so all eight rows read
/// request 0's KV. Now it must be eight groups, one per request. (Mutation check: drop the `req`
/// test from `Trip::fusable_with` and this drops to 1 group.)
#[test]
fn pure_run_breaks_at_every_request_change() {
    let trips: Vec<Trip> = (0..8u32).map(pure).collect();
    let groups = group_ranges(&trips, 512);
    assert_eq!(
        groups.len(),
        8,
        "a Pure run spanning 8 requests fused into {} group(s) — every row but the first would read \
         the wrong request's KV page",
        groups.len()
    );
    for (r, g) in groups.iter().enumerate() {
        assert_eq!(*g, r..r + 1);
    }
}

/// The flip side, and the reason the fix is a request test and not "make attention unfusable": ONE
/// request's own consecutive work still fuses to the launch budget. Decode is launch-bound; a batch
/// of 8 ran ~20x slower when its cache writes were made singletons, so over-breaking is not free.
#[test]
fn one_requests_run_still_fuses_to_g() {
    let trips: Vec<Trip> = (0..8).map(|_| pure(3)).collect();
    assert_eq!(group_ranges(&trips, 512), vec![0..8]);
    assert_eq!(group_ranges(&trips, 3), vec![0..3, 3..6, 6..8]);
}

/// EVERY kind, not just the ones that existed when the rule was written. A kind that fuses without
/// consulting the request is the bug; the request test sits above the kind test so a new variant
/// gets it by default rather than having to remember.
#[test]
fn no_kind_fuses_across_requests() {
    for kind in [
        GroupKind::Pure,
        GroupKind::Slab,
        GroupKind::PageFold,
        GroupKind::SlotSolo,
    ] {
        let trips = vec![
            Trip::new(kind, TripRequest(0)),
            Trip::new(kind, TripRequest(1)),
        ];
        let groups = group_ranges(&trips, 512);
        assert_eq!(
            groups.len(),
            2,
            "{kind:?} fused two requests into one launch — one page shift cannot address both"
        );
        assert!(!group_spans_two_requests(&trips, 512));
    }
    // `Slot` carries its request in the kind as well; both paths must agree.
    let trips = vec![
        Trip::new(GroupKind::Slot { req: 0 }, TripRequest(0)),
        Trip::new(GroupKind::Slot { req: 1 }, TripRequest(1)),
    ];
    assert_eq!(group_ranges(&trips, 512).len(), 2);
}

/// A `PageFold` reclassified to `Pure` for the fused-body variant KEEPS ITS REQUEST — the whole
/// point of `Trip`. Bug #4 was the fused body losing the fold's pass count; losing its request the
/// same way would be bug #4 again wearing a different hat.
#[test]
fn reclassified_fold_keeps_its_request() {
    let trips = vec![
        Trip::new(GroupKind::Pure, TripRequest(0)), // was PageFold, now Pure, still request 0
        Trip::new(GroupKind::Pure, TripRequest(1)),
    ];
    assert!(!group_spans_two_requests(&trips, 512));
    assert_eq!(group_ranges(&trips, 512).len(), 2);
}

/// An UNBATCHED bundle must be byte-identical to what it was: every trip is request 0, so the
/// request test never fires and the partition is exactly the old one. This is why `TripRequest(0)`
/// deliberately means both "request 0" and "untagged".
#[test]
fn unbatched_partition_is_unchanged() {
    let kinds = [
        GroupKind::Pure,
        GroupKind::Pure,
        GroupKind::Slot { req: 0 },
        GroupKind::Slot { req: 0 },
        GroupKind::SlotSolo,
        GroupKind::Slab,
        GroupKind::Slab,
        GroupKind::Pure,
    ];
    let trips: Vec<Trip> = kinds
        .iter()
        .map(|k| Trip::new(*k, TripRequest(0)))
        .collect();
    assert_eq!(
        group_ranges(&trips, 512),
        vec![0..2, 2..4, 4..5, 5..7, 7..8],
        "grouping of an unbatched body changed — every baked bundle's fingerprint would move"
    );
}

/// `HostKv` and the `skip` entries it replaces stay singletons whatever the requests are: the shim's
/// per-entry `oi += skip` has to land on exactly them.
#[test]
fn hostkv_skip_entries_stay_singletons() {
    let trips = vec![
        Trip::new(GroupKind::HostKv { skip: 2 }, TripRequest(1)),
        pure(1),
        pure(1),
        pure(1),
    ];
    assert_eq!(group_ranges(&trips, 512), vec![0..1, 1..2, 2..3, 3..4]);
}

/// The partition must still EXACTLY tile the trip sequence — a gap drops compute, an overlap
/// double-runs it — for mixed kinds AND mixed requests, at every granularity.
#[test]
fn partition_still_covers_exactly_with_mixed_requests() {
    let trips = vec![
        pure(0),
        pure(1),
        Trip::new(GroupKind::Slot { req: 1 }, TripRequest(1)),
        Trip::new(GroupKind::Slot { req: 2 }, TripRequest(2)),
        Trip::new(GroupKind::SlotSolo, TripRequest(2)),
        Trip::new(GroupKind::HostKv { skip: 1 }, TripRequest(2)),
        pure(2),
        Trip::new(GroupKind::Slab, TripRequest(0)),
        Trip::new(GroupKind::PageFold, TripRequest(0)),
    ];
    for g in 1..=8 {
        let (covered, ok) = group_ranges_cover_ok(&trips, g);
        assert!(
            ok && covered == trips.len(),
            "g={g}: covered={covered} ok={ok}"
        );
        // the Vec walk and the scalar twin must agree, else the Kani proof proves the wrong walk
        let groups = group_ranges(&trips, g);
        assert_eq!(groups.first().unwrap().start, 0);
        assert_eq!(groups.last().unwrap().end, trips.len());
        for w in groups.windows(2) {
            assert_eq!(w[0].end, w[1].start);
        }
        assert!(!group_spans_two_requests(&trips, g));
    }
}
