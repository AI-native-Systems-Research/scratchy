//! THE LAUNCH-GROUP PARTITION — which trips share one launch, and how long a run may grow.
//!
//! A group is one launch. What a launch can supply is decided at the bake and carried by the trip's
//! KIND: a `Pure` run needs nothing per trip, so it fuses up to the launch budget; a `PageFold` body
//! is relaunched per page of the resident prefix, so what its trips share is a page base and a
//! scratch state; a `Gathered` run reads through an index whose entries are absolute; a `TpConsume`
//! has the CB pipeline's send and wait placed in front of exactly its launch. Two kinds that need
//! different things from a launch must never share one — and the two laws that are correctness, not
//! cost, are pinned here:
//!
//! * a **gathered** `PageFold` run is ONE group at any length (the launch loop is group-major, so
//!   cutting it makes every pass of the second group read the page the first group's copy left
//!   behind — a silent wrong-answer bug that shipped);
//! * a consume is ONE launch per consume (each group of a chunked run would get its own send and a
//!   wait for a peer send that never comes).

use ktir_superdsc::collective::{CollectiveKind, TpConsume, TpWorld, WindowCut, ranks};
use ktir_superdsc::place::PlaceId;
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::{
    GroupKind, Trip, group_ranges, group_ranges_cover_ok,
};

/// A launch-grouping law is the same in every world; the fixtures run one card's.
type W1 = TpWorld<1>;

fn pure() -> Trip<W1> {
    Trip::new(GroupKind::Pure)
}

fn consume(stick: u32) -> TpConsume<W1> {
    TpConsume {
        kind: CollectiveKind::AllReduce,
        operand: PlaceId::Act(7),
        fragment: ktir_superdsc::collective::Fragment::baked(stick, 1, 64),
        plays_at: ranks::<W1>().next().unwrap(),
        cut: WindowCut::of_collective(0),
    }
}

/// One request's own consecutive work still fuses to the launch budget: decode is launch-bound, so
/// over-breaking is not free. `g == 1` reproduces the per-op partition exactly.
#[test]
fn a_pure_run_fuses_to_g() {
    let trips: Vec<Trip<W1>> = (0..8).map(|_| pure()).collect();
    assert_eq!(group_ranges(&trips, 512), vec![0..8]);
    assert_eq!(group_ranges(&trips, 3), vec![0..3, 3..6, 6..8]);
    assert_eq!(
        group_ranges(&trips, 0),
        vec![0..1, 1..2, 2..3, 3..4, 4..5, 5..6, 6..7, 7..8]
    );
}

/// ⭐ EVERY KIND, NOT JUST THE ONES THAT EXISTED WHEN A RULE WAS WRITTEN. A launch supplies what a
/// kind's group reads — a page base, an absolute index, a send/wait placement — so two trips of
/// different kinds never share one launch. (Mutation check: widen any `fusable_kind_with` arm and
/// this drops to 1 group for that pair.)
#[test]
fn no_two_kinds_share_a_launch() {
    let kinds = [
        GroupKind::<W1>::Pure,
        GroupKind::PageFold { gathered: true },
        GroupKind::PageFold { gathered: false },
        GroupKind::Gathered,
        GroupKind::TpConsume(consume(3)),
    ];
    for (i, a) in kinds.iter().enumerate() {
        for b in &kinds[i + 1..] {
            let trips = vec![Trip::new(*a), Trip::new(*b)];
            let groups = group_ranges(&trips, 512);
            assert_eq!(
                groups.len(),
                2,
                "{a:?} + {b:?} fused into one launch — a launch cannot supply what both kinds read"
            );
        }
    }
    // AND THE TWO FOLDS DO NOT FUSE WITH EACH OTHER. A group takes its SIZE CAP from its FIRST trip,
    // so a mixed run would hand one half the other's cap — silently, and in the direction that loses
    // the gathered fold's exemption.
    let mixed = vec![
        Trip::<W1>::new(GroupKind::PageFold { gathered: true }),
        Trip::new(GroupKind::PageFold { gathered: false }),
    ];
    assert_eq!(group_ranges(&mixed, 512), vec![0..1, 1..2]);
}

/// A consume's own descriptor ops are ONE launch — and two consumes never share one, because the CB
/// pipeline places this rank's send and the wait for the peer's in front of exactly one consume
/// launch: a neighbour fused in would run before its operand's exchange, and a consume split across
/// two groups would send twice. And the run is never chunked by the group size.
#[test]
fn one_consume_is_one_launch_whatever_the_group_size() {
    let same = |s: u32| Trip::new(GroupKind::TpConsume(consume(s)));
    // SAME consume: one group at any length and any g.
    let trips: Vec<Trip<W1>> = (0..40).map(|_| same(3)).collect();
    for g in [1usize, 2, 8, 512] {
        assert_eq!(
            group_ranges(&trips, g),
            vec![0..40],
            "g={g}: a consume's ops must be ONE group — a chunked run sends twice and waits for a \
             second peer send that never comes"
        );
    }
    // DIFFERENT consumes: never one launch.
    let two = vec![same(3), same(4)];
    assert_eq!(group_ranges(&two, 512), vec![0..1, 1..2]);
    // A consume never fuses with its pure neighbours.
    let mixed = vec![pure(), same(3), pure()];
    assert_eq!(group_ranges(&mixed, 512), vec![0..1, 1..2, 2..3]);
}

/// The partition must EXACTLY tile the trip sequence — a gap drops compute, an overlap double-runs
/// it — for mixed kinds at every granularity. The scalar twin must agree with the Vec walk, else
/// the Kani proof proves a walk the emitter does not take.
#[test]
fn the_partition_covers_exactly_with_mixed_kinds() {
    let trips = vec![
        pure(),
        pure(),
        Trip::new(GroupKind::PageFold { gathered: true }),
        Trip::new(GroupKind::PageFold { gathered: true }),
        Trip::new(GroupKind::PageFold { gathered: false }),
        Trip::new(GroupKind::Gathered),
        pure(),
        Trip::new(GroupKind::TpConsume(consume(3))),
    ];
    for g in 1..=8 {
        let (covered, ok) = group_ranges_cover_ok(&trips, g);
        assert!(
            ok && covered == trips.len(),
            "g={g}: covered={covered} ok={ok}"
        );
        let groups = group_ranges(&trips, g);
        assert_eq!(groups.first().unwrap().start, 0);
        assert_eq!(groups.last().unwrap().end, trips.len());
        for w in groups.windows(2) {
            assert_eq!(w[0].end, w[1].start);
        }
    }
}

/// ⭐⭐⭐⭐⭐ A **GATHERED** `PageFold` RUN IS ONE GROUP AT ANY LENGTH — the guard for a silent
/// wrong-answer bug that shipped, not a tidiness check.
///
/// A group is the unit of the `reps` relaunch and the launch loop is group-major
/// (`superdsc_exec::launch_ops_inner`: `for op { for rep { … } }`). So a fold cut into groups A and B
/// runs A(pass 0..n) then B(pass 0..n) — and a GATHERED fold's passes communicate through a scratch
/// they REWRITE, so every pass of B reads the page A left behind on its LAST pass. With a single
/// pass that is invisible; it breaks at the first page crossing, at every batch width.
///
/// MEASURED on granite-3.1-2b fp8, 420-token probe: first divergence at absolute slot 256, offset 0
/// into page 1, identical at width 2 and width 8, and `SCRATCHY_SDSC_PEROP_SYNC=1` does not move it
/// (the order is semantically wrong, not racy). `GroupSize`'s own doc records the same effect from
/// the other side — `solo_diff` 8,8,8,8,8 at `g = 128` against 7,6,7 at `g = 512`, i.e. the
/// partition changed the answer. A bigger `g` only moves which contexts are wrong; not cutting the
/// run is the fix.
#[test]
fn a_gathered_page_fold_run_is_never_chunked_by_the_group_size() {
    let fold = Trip::new(GroupKind::PageFold { gathered: true });
    // Longer than every `g` tried, so a cap that still applied would show up as more than one group.
    let trips: Vec<Trip<W1>> = (0..40).map(|_| fold).collect();
    for g in [1usize, 2, 3, 8, 128, 512] {
        assert_eq!(
            group_ranges(&trips, g),
            vec![0..40],
            "g={g}: a gathered PageFold run must be ONE group — cutting it makes every pass of the \
             second group read the gathered page the first group left at its LAST pass"
        );
        let (covered, ok) = group_ranges_cover_ok(&trips, g);
        assert!(
            ok && covered == trips.len(),
            "g={g}: the scalar twin must accept the unchunked run too, or the Kani proof proves a \
             walk the emitter does not take (covered={covered} ok={ok})"
        );
    }
    // ⛔ AND ONLY THE GATHERED FOLD IS EXEMPT — a `Pure` run of the same length still chunks, which is
    // what keeps the launch budget meaningful for the rest of the body.
    let pures: Vec<Trip<W1>> = (0..40).map(|_| pure()).collect();
    assert_eq!(group_ranges(&pures, 8).len(), 5);
}

/// ⭐⭐⭐⭐⭐ AND AN **UNGATHERED** `PageFold` RUN **IS** CHUNKED — the other half of the same law,
/// and a MEASURED 8b regression when it was not.
///
/// The exemption above was applied to both folds with the justification that "the ungathered fold
/// loses nothing by it: fewer groups is fewer launches". The measurements below were taken when
/// `RedHatAI/granite-3.1-8b-instruct-FP8-dynamic` (hd = 128) had NO gather, because
/// `PageScratch::of_pass` refused two slabs — a refusal since removed, so that model's fold is now
/// GATHERED and takes the exemption. The rule is unchanged and so is its evidence: what these
/// numbers pin is that the UNGATHERED fold (every prompt-chunk bundle, and any future ungathered
/// decode) must keep the cap. On the 420-token `c` probe at width 8, each binary against its OWN
/// `--max-num-seqs 1` run, N = 3:
/// ```text
///   origin/main (cap applied)  own_bad 8,7,7  first divergence 87-118 chars in (term 14-19)
///   cap lifted for both folds  own_bad 8,8,8  first divergence 1-4 chars in (term 0): " 11111111…"
///   this rule (gathered only)  own_bad 7,8,8  first divergence 87-118 chars in (term 14-19)
/// ```
/// Width 8 at hd=128 is broken in all three — that predates the fold work — but the lifted cap turns
/// a prefix that is coherent for fifteen terms into garbage from the first token, and restoring the
/// cap puts the per-row divergence offsets back on main's values row for row.
///
/// (Mutation check: make `run_may_be_chunked` ignore the payload again and this test drops to 1
/// group.)
#[test]
fn an_ungathered_page_fold_run_is_chunked_like_any_other() {
    let fold = Trip::new(GroupKind::PageFold { gathered: false });
    let trips: Vec<Trip<W1>> = (0..40).map(|_| fold).collect();
    assert_eq!(
        group_ranges(&trips, 8),
        vec![0..8, 8..16, 16..24, 24..32, 32..40],
        "an UNGATHERED fold run must chunk at g — its passes rebase the KV segment instead of \
         rewriting a shared scratch, so it never needed the exemption, and taking it is a measured \
         granite-3.1-8b fp8 regression"
    );
    for g in [1usize, 2, 3, 8, 128, 512] {
        let (covered, ok) = group_ranges_cover_ok(&trips, g);
        assert!(
            ok && covered == trips.len(),
            "g={g}: the scalar twin must agree with the chunked walk too (covered={covered} ok={ok})"
        );
    }
}
