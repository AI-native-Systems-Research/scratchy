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
//!
//! # ⭐⭐⭐⭐⭐ AND THE GATHER IS THE LEVER ON THE `4 + 2*mq` SPLIT ITSELF — READ THIS BEFORE COSTING IT
//!
//! The per-request split is NOT `SCRATCHY_SUPERDSC_GROUP_SIZE`, and it cannot be tuned away.
//! [`Trip::fusable_with`] tests `self.req == other.req` FIRST and UNCONDITIONALLY, so the group count
//! is `4 + 2*mq` whatever the size knob says: two of the per-request groups are the **KV cache write**
//! and the **Kᵗ re-transpose**, and they are per-request because each one needs a per-request PAGE
//! BASE, which a launch can supply exactly once.
//!
//! ⛔ SO THE GATHER IS NOT ONLY THE SCORE/VALUE READ. An indirect address is precisely what lets an op
//! stop needing a per-request page base: `addr = idx * skip_addr + base_addr` puts the page in the
//! INDEX ENTRY, and an op whose page comes from its index has nothing left that makes it request-shaped
//! — so it may fuse across requests and the two remaining per-request groups collapse. Giving the cache
//! write and the Kᵗ re-transpose their own index is step (1) of the batch-decode plan, and it is the
//! SAME mechanism `assemble_gather_copy` already emits for the fold.
//!
//! That is why the gather is worth more than the fold pass it was built for. The launch count is the
//! whole cost of batched decode (`PoolSplit`: ~88 µs per launch whatever it contains, 27.6 launches in
//! a bs=8 layer, 95 ms of a 119 ms step), and `4 + 2*mq` is what the gather can move to `4`.
//! ⛔ NOT BUILT YET, and nothing here asserts it — this is the priority note, recorded where the next
//! session reads the group law rather than in a commit message.
//!
//! # MEASURED, so the next session does not re-derive it: the fold gather's own two numbers
//!
//! Read off the REAL rung-2 batched-decode descriptors on the card pod
//! (`/tmp/superdsc-dump/05f19f853f28683b/group_1/sdsc_{0,1}.json`, granite-3.1-2b, `nkvh=8 hd=64
//! mq=2 nb=2`), because the fence that fires in that group names nothing and the first thing to rule
//! out is an extent that disagrees with its placement:
//!
//! | quantity | `attn_gkt_o734` | `attn_gv_o734` |
//! |---|---|---|
//! | `N_` (`mb x out x y`) | 2048 x 64 x 1 = **262,144 B** | same |
//! | per-core `ss_` | 64 x 64 x 1 = 8,192 B x 32 cores | same |
//! | declared footprint (`GatherScratch::footprint_dims` -> `synth_footprint_bytes`) | 32 x 4096 x 2 = **262,144 B** | same |
//! | output (`Tensor2`) per-core span, seg0 | [8,335,104, 8,597,248) | [8,597,248, 8,859,392) |
//! | gathered source (`Tensor0`) per-core span, seg2 | [524,288, 786,432) (`kct`) | [262,144, 524,288) (`vc`) |
//!
//! ⭐ **THEY AGREE, EXACTLY AND IN BOTH DIRECTIONS.** The declared output extent IS the reserved
//! footprint (`sub_rows * POOL_STICK == rows * cols`, which is now one value — `CopyDims`), the two
//! scratches TILE seg0 without overlap, and the two gathered sources tile their own placements in seg2.
//! `bake_layout` grows seg0 to the synth high-water and `audit_layout_addresses` proves
//! `offset + size <= segment_bytes[0]`, so no baked address in either op leaves its tensor.
//!
//! ## The SHIFTED WINDOW, considered and ELIMINATED — no card run spent
//!
//! `DevAddr::shifted` moves a segment's base up by `off[seg]` and SHRINKS its size by the same amount,
//! and the launch applies `off[SEG_INTERMEDIATE] += delta.intermediate` / `off[SEG_MASK] += delta.mask`
//! per rep. Every build-time guard compares a baked offset against `segment_bytes[seg]`, never against
//! `segment_bytes[seg] - shift`, so a tensor at the TOP of a shifted segment would reach `shift` bytes
//! past its window — and a region overrun is exactly what reaches the card as `0xa35e BusFence` with no
//! address (`control_block_wire.rs:239`, `scheduler.rs:2996`). The gather puts two 256 KB synths at the
//! top of seg0, which is the first time that segment has had anything there.
//!
//! ⛔ **BUT seg0 IS NEVER SHIFTED.** `assemble_attn` sets `let per_req = false` UNCONDITIONALLY, so
//! `block_rows` is always `BlockRows::WholeBatch`, so the manifest declares
//! `FoldRowRegime::WholeBatch`, whose `int_rep_stride_bytes` is **0 by derivation** — and
//! `fold_plan::fold_delta`'s intermediate term is `req * int_rep_stride_bytes`. Every bundle in the
//! tree, gathered or not, shifts seg0 by zero. The two gather scratches are therefore read and written
//! at exactly their baked addresses, which `audit_layout_addresses` already proves are inside
//! `segment_bytes[0]`.
//!
//! That leaves only seg3, which the mask shift does move — and the gather's INDEX lives there
//! (`SegRole::Activation` IS `SEG_MASK`). Its placement is `pmbytes = pm_blocks * rep_stride_bytes` and
//! the shift is `rep * rep_stride_bytes`, so the read stays inside the placement exactly while
//! `reps <= pm_blocks`, which the launch already REFUSES to exceed ("fold: N pass(es) REFUSED"). So no
//! shifted window explains the fence either.
//!
//! ## ⭐ WHAT THE DESCRIPTOR DOES SAY, AND IT IS THE NEXT THING TO SETTLE
//!
//! The index operand (`Tensor1` of both copies) is declared `layoutDimOrder_ ["mb"]`,
//! `maxDimSizes_ [2048]`, `stickSize_ [32]`, `SENUINT32` — **2048 entry slots, one per `mb` POSITION**
//! — and its 32 per-core start addresses step **256 B = 64 entries**, because `mb` is split 32 ways and
//! a core owns 64 positions. The HOST stages `gather_index_table`, which writes `scratch.rows()` = **32**
//! live entries at indices `0..32` and leaves the rest of the pass block at `pad` (page 0, block 0 — a
//! REAL address, stated as such at that function).
//!
//! So core `c` is handed the index at entry `c * 64`, while the entry the host wrote for it is at entry
//! `c`. Under "one entry per `mb` position" the table would have to be written at stride `page` (entry
//! for scratch row `r` at index `r * 64`); under "one entry per PAGE of positions" the declared extent
//! and the 256 B per-core step are both 64x too large. One of those two is wrong and the descriptor
//! cannot say which.
//!
//! ⛔ THIS IS CONSISTENT WITH THE ZERO-ENTRY ABLATION LEAVING THE FENCE BIT-IDENTICAL: zeroing the 32
//! live entries changes only what 1 of 32 cores reads, and the other 31 already read `pad`. It is a
//! FLUENT wrong-address class, not an out-of-bounds one, so it is probably not the fence — but it IS a
//! real defect and the ablation does not clear it. Settle it against the C++ AUTHORITY on the pod
//! (`/project_src/deeptools`: `dsc2.cpp:4001-4008` for the declared extent,
//! `ConvertData_gather_idx` for which slot a position's entry is read from), not by reading our own
//! declaration back.

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
        GroupKind::PageFold { gathered: true },
        GroupKind::PageFold { gathered: false },
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
        Trip::new(GroupKind::PageFold { gathered: true }, TripRequest(0)),
        Trip::new(GroupKind::PageFold { gathered: false }, TripRequest(0)),
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

/// ⭐⭐⭐⭐⭐ A **GATHERED** `PageFold` RUN IS ONE GROUP AT ANY LENGTH — the guard for a silent
/// wrong-answer bug that shipped, not a tidiness check.
///
/// A group is the unit of the `reps` relaunch and the launch loop is group-major
/// (`superdsc_exec::launch_ops_inner`: `for op { for rep { … } }`). So a fold cut into groups A and B
/// runs A(pass 0..n) then B(pass 0..n) — and a GATHERED fold's passes communicate through a scratch they
/// REWRITE, so every pass of B reads the page A left behind on its LAST pass. With a single pass that is
/// invisible; it breaks at the first page crossing, at every batch width.
///
/// MEASURED on granite-3.1-2b fp8, 420-token probe: first divergence at absolute slot 256, offset 0 into
/// page 1, identical at width 2 and width 8, and `SCRATCHY_SDSC_PEROP_SYNC=1` does not move it (the order
/// is semantically wrong, not racy). `GroupSize`'s own doc records the same effect from the other side —
/// `solo_diff` 8,8,8,8,8 at `g = 128` against 7,6,7 at `g = 512`, i.e. the partition changed the answer.
/// A bigger `g` only moves which contexts are wrong; not cutting the run is the fix.
#[test]
fn a_gathered_page_fold_run_is_never_chunked_by_the_group_size() {
    let fold = |r: u32| Trip::new(GroupKind::PageFold { gathered: true }, TripRequest(r));
    // Longer than every `g` tried, so a cap that still applied would show up as more than one group.
    let trips: Vec<Trip> = (0..40).map(|_| fold(0)).collect();
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
    // ⛔ AND THE REQUEST TEST STILL WINS. Exemption from the SIZE cap is not exemption from the rule
    // that one launch resolves one request's page table.
    let mixed = vec![fold(0), fold(0), fold(1), fold(1)];
    assert_eq!(group_ranges(&mixed, 512), vec![0..2, 2..4]);
    // ⛔ AND ONLY THE GATHERED FOLD IS EXEMPT — a `Pure` run of the same length still chunks, which is
    // what keeps the launch budget meaningful for the rest of the body.
    let pures: Vec<Trip> = (0..40).map(|_| pure(0)).collect();
    assert_eq!(group_ranges(&pures, 8).len(), 5);
}

/// ⭐⭐⭐⭐⭐ AND AN **UNGATHERED** `PageFold` RUN **IS** CHUNKED — the other half of the same law, and a
/// MEASURED 8b regression when it was not.
///
/// The exemption above was applied to both folds with the justification that "the ungathered fold loses
/// nothing by it: fewer groups is fewer launches". The measurements below were taken when
/// `RedHatAI/granite-3.1-8b-instruct-FP8-dynamic` (hd = 128) had NO gather, because
/// `PageScratch::of_pass` refused two slabs — a refusal since removed, so that model's fold is now
/// GATHERED and takes the exemption. The rule is unchanged and so is its evidence: what these numbers
/// pin is that the UNGATHERED fold (every prompt-chunk bundle, and any future ungathered decode) must
/// keep the cap. On the 420-token `c` probe at width 8, each binary against its OWN `--max-num-seqs 1`
/// run, N = 3:
/// ```text
///   origin/main (cap applied)  own_bad 8,7,7  first divergence 87-118 chars in (term 14-19)
///   cap lifted for both folds  own_bad 8,8,8  first divergence 1-4 chars in (term 0): " 11111111…"
///   this rule (gathered only)  own_bad 7,8,8  first divergence 87-118 chars in (term 14-19)
/// ```
/// Width 8 at hd=128 is broken in all three — that predates the fold work — but the lifted cap turns a
/// prefix that is coherent for fifteen terms into garbage from the first token, and restoring the cap
/// puts the per-row divergence offsets back on main's values row for row.
///
/// (Mutation check: make `run_may_be_chunked` ignore the payload again and this test drops to 1 group.)
#[test]
fn an_ungathered_page_fold_run_is_chunked_like_any_other() {
    let fold = Trip::new(GroupKind::PageFold { gathered: false }, TripRequest(0));
    let trips: Vec<Trip> = (0..40).map(|_| fold).collect();
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
        assert!(!group_spans_two_requests(&trips, g));
    }
    // ⛔ AND THE TWO FOLDS DO NOT FUSE WITH EACH OTHER. A group takes its cap from its FIRST trip, so a
    // mixed run would hand one half the other's cap — silently, and in the direction that loses the
    // gathered fold's exemption.
    let mixed = vec![
        Trip::new(GroupKind::PageFold { gathered: true }, TripRequest(0)),
        Trip::new(GroupKind::PageFold { gathered: false }, TripRequest(0)),
    ];
    assert_eq!(group_ranges(&mixed, 512), vec![0..1, 1..2]);
}
