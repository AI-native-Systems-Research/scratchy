// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE GATHER GRANULARITY WAS THE WHOLE PROBLEM — proven against [`PagedKvPool::addr`], the one
//! address law, at hd=64 AND hd=128.
//!
//! This file exists because four days went into deriving constraints that IBM's own reference does not
//! have. `spyre-inference`'s `page_attn_head_major_decode_kernel` gathers `k_pages[kv_rows]` — an ENTIRE
//! PAGE by ONE index value — inside a per-page online-softmax loop, and its own probe runs that at
//! `head_size=128`. We gathered a 64-slot WINDOW OF ONE PLANE.
//!
//! ## The three claims, and the contrast that explains every refusal
//!
//! 1. **A WHOLE PAGE PLANE IS A BIJECTION ONTO `[0, nkvh*hd*PAGE_SLOTS)` — for ALL THREE PLANES, at BOTH
//!    head dims.** No hole, no collision. So ONE index entry names a request's whole page at any head dim
//!    that is a whole number of sticks, and `skip_addr` is that run.
//!
//! 2. **AT PAGE GRANULARITY EVEN Kᵗ IS CONTIGUOUS**, so the "two planes number their blocks differently
//!    above one stick" obstacle — which is REAL at window granularity and which cost a full session to
//!    derive — simply does not arise. Deleting the resident Kᵗ plane is a memory win; it was never a
//!    precondition for gathering at hd=128.
//!
//! 3. **THE OLD GRANULARITY IS WHAT FAILED.** A 64-slot WINDOW of `Knat`/`V` is one contiguous run only
//!    at `hd == POOL_STICK`; at hd=128 it is `nslab` runs `PAGE_SLOTS*POOL_STICK` apart. That single fact
//!    is the sole origin of `of_fold_pass`'s `hd > POOL_STICK` refusal.
//!
//! ⛔ AND THE ENTRY COUNT COLLAPSES WITH IT. A page-granular pass needs one INDEX STICK PER REQUEST,
//! never `nkvh * windows * mq` entries in one op — which is 64/128/256 at width 8 against dxp's ONE
//! 32-word IBR stick, the measured cause of the rung-8 corruption. A request's own run is
//! `nkvh * PAGE_SLOTS / POOL_STICK` (32 at nkvh=8, at EVERY head dim), which is exactly the stick.
//!
//! ## ⛔⛔⛔ CLAIM 1 IS ABOUT THE ROW, NOT ABOUT THE PIN — AND CONFLATING THEM COST A CARD REFUSAL
//!
//! "One index entry names a request's whole page" is the sentence that does not survive the card. A whole
//! page plane IS a bijection (claim 1 stands, and it is what makes the DESTINATION one contiguous row per
//! request), but PINNING a whole plane on the gather dim is refused at bake:
//! ```text
//! sbf-ddc: DtException: Unable to map graph within architecture constraints:
//!   The initial chunk parameters must fit in LX for SuperDSC: 78_attn_gkt_o734_s0
//!   L3DlOpsScheduler.cpp:1534   (gated on isDoubleBuffering)
//! ```
//! The pin is the op's UNIT OF WORK DIVISION, not just a size: `gather_copy_cores` may only split the
//! gather dim into WHOLE entries, so a plane-sized pin over a plane-sized `mb` leaves ONE entry, hence
//! ONE CORE, and dxp measures that core's double-buffered 256 KB in + 256 KB out against LX. The pin is
//! therefore ONE STICK BLOCK ([`PagePlaneExtent::entry_elems`]) and a request's row is filled by its own
//! `blocks()` entries — the vendor's `[num_blocks, INT32_ELEMS_PER_STICK]` layout, one stick per request.

use ktir_superdsc::sdsc_abstract::{
    CopyDims, FeatIdx, GatherEntry, GatherScratch, KvCoord, KvHead, KvPlane, KvSlot, POOL_STICK,
    PagePlaneExtent, PageScratch, PagedKvPool, QueryRowCount, gather_entries_per_page,
};
use std::collections::BTreeSet;
use std::num::NonZeroU32;

/// ⭐⭐⭐⭐⭐ THE KV-HEAD SWEEP THAT DID NOT EXIST. Every `of_pass` call site in this suite passed 8, and
/// `PagePlaneExtent::blocks` is `nkvh * PAGE_SLOTS / POOL_STICK` — 32 at nkvh=8, which is ALSO
/// `CopyDims::ENTRIES_PER_OP` and ALSO `PageScratch::ENTRIES_PER_PASS_MAX`. Three unrelated quantities
/// with one value, never separated by a test, and the one that moves is this axis.
const NKVH_SWEEP: [usize; 4] = [4, 8, 16, 32];

/// And the head-dim sweep, which must include a dim where `hd != POOL_STICK` (the pin) and one where
/// `nslab > 2` (the fold's slab loops).
const HD_SWEEP: [usize; 3] = [64, 128, 256];

/// Every address of one page's plane, over every kv head, slot and feature.
fn plane_addrs(pool: &PagedKvPool, plane: KvPlane) -> Vec<u32> {
    let nkvh = NonZeroU32::new(pool.nkvh as u32).expect("nkvh > 0");
    let mut out = Vec::new();
    for h in 0..pool.nkvh as u32 {
        let kvh = KvHead::new(h, nkvh).expect("kv head in range");
        for slot in 0..PagedKvPool::PAGE_SLOTS as u32 {
            for feat in 0..pool.hd as u32 {
                out.push(
                    pool.addr(
                        KvCoord::block(plane, kvh)
                            .at_slot(KvSlot::new(slot))
                            .at_feat(FeatIdx::new(feat)),
                    ),
                );
            }
        }
    }
    out
}

/// CLAIM 1 + 2: all three planes tile `[0, nkvh*hd*PAGE_SLOTS)` exactly, at both head dims.
#[test]
fn a_whole_page_plane_is_a_hole_free_bijection_for_every_plane_at_both_head_dims() {
    for hd in [64usize, 128] {
        let pool = PagedKvPool::new(8, hd);
        let want = 8u64 * hd as u64 * PagedKvPool::PAGE_SLOTS as u64;

        let extent = PagePlaneExtent::of_pool(pool).expect("a stick-multiple head dim is admitted");
        assert_eq!(
            extent.elems(),
            want,
            "hd={hd}: PagePlaneExtent::elems must be the whole page plane"
        );

        for plane in [KvPlane::Knat, KvPlane::V, KvPlane::Kt] {
            let addrs = plane_addrs(&pool, plane);
            assert_eq!(
                addrs.len() as u64,
                want,
                "hd={hd} {plane:?}: coordinate count"
            );

            let distinct: BTreeSet<u32> = addrs.iter().copied().collect();
            assert_eq!(
                distinct.len() as u64,
                want,
                "hd={hd} {plane:?}: two coordinates COLLIDE — one would overwrite the other"
            );
            assert_eq!(
                (
                    *distinct.iter().next().unwrap(),
                    *distinct.iter().next_back().unwrap()
                ),
                (0, (want - 1) as u32),
                "hd={hd} {plane:?}: the plane is not the contiguous range [0, {want}) — a HOLE means a \
                 page-granular gather would move the wrong elements for every slab above the first"
            );
        }
    }
}

/// The pin dxp derives `skip_addr` from, over the WHOLE `nkvh` x `hd` grid.
///
/// ⭐ THE EVENNESS IS NOT HERE ANY MORE, AND ITS ABSENCE IS THE POINT. `PagePlaneExtent::pin_is_even` was
/// a runtime `bool` that `PageScratch::of_pass` branched on, and its test sat inside an
/// hd-multiple-of-64 loop — so it asserted its own premise and could never fail. The pin is `hd` sticks
/// and `of_pool` admits only whole multiples of `POOL_STICK`, so "the pin is even" is a statement about
/// the STICK, and it now lives where a statement about a constant belongs: a module-level
/// `const _: () = assert!(POOL_STICK.is_multiple_of(2), ..)` in `sdsc_abstract`, which is a
/// required-const context and therefore actually evaluates.
#[test]
fn the_pin_is_one_stick_block_at_every_head_dim_and_kv_head_count() {
    for nkvh in NKVH_SWEEP {
        for hd in HD_SWEEP {
            let extent = PagePlaneExtent::of_pool(PagedKvPool::new(nkvh, hd)).expect("admitted");
            assert_eq!(
                extent.entry_sub_rows(),
                extent.entry_elems() / POOL_STICK as u64,
                "nkvh={nkvh} hd={hd}: the pinned position count is ONE STICK BLOCK in sticks"
            );
            // ⛔ THE PIN IS `hd` SUB-ROWS, WHICH EQUALS `POOL_STICK` AT hd=64 AND NOWHERE ELSE. That
            // coincidence is one of the four 32s/64s in this door; stating the pin against `hd` rather
            // than against the stick is what keeps it from being read as the stick.
            assert_eq!(
                extent.skip_addr_sticks(),
                hd as u64,
                "nkvh={nkvh} hd={hd}: skip_addr is `hd` sticks — a quantity that MOVES with the head \
                 dim, not the constant it coincides with at hd=64"
            );
            // ⛔ AND IT IS **NOT** THE PLANE. This inequality is the card refusal in one line: a
            // plane-sized pin leaves one entry, therefore one core, therefore a double-buffered chunk of
            // the whole op against LX.
            assert!(
                extent.entry_sub_rows() < extent.sub_rows(),
                "nkvh={nkvh} hd={hd}: the pin must be strictly smaller than the plane — a plane-sized \
                 pin is REFUSED at bake (L3DlOpsScheduler.cpp:1534) because it admits no core split"
            );
            assert_eq!(
                extent.sub_rows(),
                extent.entry_sub_rows() * extent.blocks(),
                "nkvh={nkvh} hd={hd}: a request's row is a whole number of pinned entries, or the last \
                 entry covers a partial block — which is an address"
            );
        }
    }
}

/// ⭐⭐⭐⭐⭐ THE FOUR COINCIDING 32s, SEPARATED — the test that exists because NOTHING swept `nkvh`.
///
/// At the one geometry every call site used (nkvh=8, hd=64) these are all 32 and all unrelated:
///
/// | quantity | what it is | `nkvh*hd` dependence |
/// |---|---|---|
/// | `CopyDims::ENTRIES_PER_OP` | the int32 IBR stick WIDTH | none — the index's dtype |
/// | `PageScratch::ENTRIES_PER_PASS_MAX` | the widest batch rung | none — the ladder |
/// | `PagePlaneExtent::blocks` | entries one request's plane needs | `nkvh * PAGE_SLOTS / POOL_STICK` |
/// | `PagePlaneExtent::lx_entries_per_op` | entries whose per-core chunk fits LX | `~1/hd` |
///
/// Only the third and fourth move, and they move on DIFFERENT axes — which is why a sweep of `hd` alone
/// could not have found the conflation either.
#[test]
fn the_four_coinciding_thirty_twos_are_four_different_quantities() {
    // The two constants do not move at all.
    assert_eq!(CopyDims::ENTRIES_PER_OP, 32);
    assert_eq!(PageScratch::ENTRIES_PER_PASS_MAX, 32);
    for nkvh in NKVH_SWEEP {
        for hd in HD_SWEEP {
            let plane = PagePlaneExtent::of_pool(PagedKvPool::new(nkvh, hd)).expect("admitted");
            // `blocks` moves with `nkvh` ONLY — `hd` cancels.
            assert_eq!(
                plane.blocks(),
                nkvh as u64 * PagedKvPool::PAGE_SLOTS as u64 / POOL_STICK as u64,
                "nkvh={nkvh} hd={hd}: a request's entry count is nkvh * PAGE_SLOTS / stick"
            );
            assert_eq!(
                plane.blocks() == 32,
                nkvh == 8,
                "blocks() is 32 iff nkvh == 8"
            );
            // The LX ceiling moves with `hd` ONLY, and it is far above the IBR stick at every real
            // geometry — so the CUT is the IBR stick's, which is what the emitter must be derived from.
            assert_eq!(
                plane.lx_entries_per_op(),
                (ktir_superdsc::superdsc_opspec::USABLE_LX_BYTES / (4 * plane.entry_bytes()))
                    * ktir_superdsc::superdsc_opspec::MAX_CORES as u64,
                "nkvh={nkvh} hd={hd}: the LX ceiling is a PER-CORE budget times the cores"
            );
            assert!(
                plane.lx_entries_per_op() > CopyDims::ENTRIES_PER_OP as u64,
                "nkvh={nkvh} hd={hd}: LX admits {} entries, so the binding ceiling is the IBR stick — \
                 if this ever fails the cut changes and `entries_per_op` already takes the min",
                plane.lx_entries_per_op()
            );
        }
    }
    // And the LX term DOES bind somewhere, so the `min` is not decoration: one entry of a 4096-wide head
    // is 512 KB, whose double-buffered source+destination pair is 2 MB against a 1.6 MB LX.
    let huge = PagePlaneExtent::of_pool(PagedKvPool::new(8, 4096)).expect("stick multiple");
    assert_eq!(
        huge.lx_entries_per_op(),
        0,
        "hd=4096: one entry's own chunk does not fit LX, which `of_pass` must refuse so the emitter \
         fails the BUILD rather than dropping the gather"
    );
    assert_eq!(
        PageScratch::of_pass(PagedKvPool::new(8, 4096), QueryRowCount::of_mq(8)),
        None,
        "hd=4096: an inexpressible entry is a refusal here and a build-time panic at the door"
    );
}

/// CLAIM 3, the contrast: the OLD granularity. A 64-slot WINDOW of `Knat`/`V` is one contiguous run only
/// at `hd == POOL_STICK`. This is the refusal's actual cause, pinned so it cannot be re-derived as a
/// property of the gather or of the hardware.
#[test]
fn a_sixty_four_slot_window_is_contiguous_only_at_one_stick_which_is_why_the_old_door_refused() {
    let nkvh = NonZeroU32::new(8).unwrap();
    for hd in [64usize, 128] {
        let pool = PagedKvPool::new(8, hd);
        let kvh = KvHead::new(0, nkvh).unwrap();
        // One 64-slot window of V, every feature: the run the OLD gather tried to name with one entry.
        let mut addrs: Vec<u32> = Vec::new();
        for slot in 0..POOL_STICK {
            for feat in 0..hd as u32 {
                addrs.push(
                    pool.addr(
                        KvCoord::block(KvPlane::V, kvh)
                            .at_slot(KvSlot::new(slot))
                            .at_feat(FeatIdx::new(feat)),
                    ),
                );
            }
        }
        let distinct: BTreeSet<u32> = addrs.iter().copied().collect();
        let span = *distinct.iter().next_back().unwrap() - *distinct.iter().next().unwrap() + 1;
        let contiguous = span as usize == distinct.len();
        assert_eq!(
            contiguous,
            hd == POOL_STICK as usize,
            "hd={hd}: a 64-slot V window is contiguous iff hd is ONE STICK (span {span}, count {}) — \
             this asymmetry, not the hardware, is what made the gather hd==64-only",
            distinct.len()
        );
    }
}

/// ⭐⭐⭐⭐⭐ THE SCRATCH IS THE **SAME FUNCTION** AS THE POOL, not a second derivation of it.
///
/// A page-granular row is a byte-for-byte copy of the page plane, so element `(request, kvh, slot, feat)`
/// of the scratch must be `request * cols + PagedKvPool::addr(...)`. Pinning it against the real `addr` is
/// what stops the scratch's head stride and the pool's from drifting apart — the defect class that has
/// produced a wrong answer in this neighbourhood more than once ("one quantity computed twice").
#[test]
fn the_scratch_head_corner_is_the_pools_own_addr_for_every_plane_and_head_dim() {
    for nkvh_n in NKVH_SWEEP {
        for hd in HD_SWEEP {
            let pool = PagedKvPool::new(nkvh_n, hd);
            let nkvh = NonZeroU32::new(nkvh_n as u32).unwrap();
            let scratch = PageScratch::of_pass(pool, QueryRowCount::of_mq(8)).expect("admitted");
            assert_eq!(
                scratch.cols(),
                nkvh_n as u64 * hd as u64 * PagedKvPool::PAGE_SLOTS as u64
            );

            for plane in [KvPlane::Knat, KvPlane::V, KvPlane::Kt] {
                for r in 0..scratch.mq() {
                    for h in 0..nkvh_n as u32 {
                        let kvh = KvHead::new(h, nkvh).unwrap();
                        let pool_corner = pool.addr(KvCoord::block(plane, kvh)) as u64;
                        assert_eq!(
                            scratch.head_off(r, h).expect("in range"),
                            r as u64 * scratch.cols() + pool_corner,
                            "nkvh={nkvh_n} hd={hd} {plane:?} request {r} head {h}: the scratch's head \
                             corner disagrees with PagedKvPool::addr — the matmul would read another \
                             head's slots"
                        );
                        // ⭐ AND THE SLAB REACHES IT THROUGH THE SAME LAW. The gathered fold's legs are
                        // one op per (head, request, SLAB), and each op's base is this corner plus the
                        // pool's own `at_feat` — never a slab term added here.
                        for s in 0..(hd as u32 / POOL_STICK) {
                            let coord = KvCoord::block(plane, kvh).at_feat(FeatIdx::of_slab(s));
                            assert_eq!(
                                scratch.coord_off(r, &pool, coord).expect("in range"),
                                r as u64 * scratch.cols() + pool.addr(coord) as u64,
                                "nkvh={nkvh_n} hd={hd} {plane:?} r={r} h={h} slab={s}: the gathered \
                                 slab base is the pool's own address plus this request's row"
                            );
                        }
                    }
                }
            }
            assert_eq!(
                scratch.head_off(8, 0),
                None,
                "nkvh={nkvh_n} hd={hd}: past the batch must refuse"
            );
            assert_eq!(
                scratch.head_off(0, nkvh_n as u32),
                None,
                "nkvh={nkvh_n} hd={hd}: past the head count must refuse"
            );
        }
    }
}

/// ⭐⭐⭐⭐⭐ THE SAME IDENTITY AT **BOTH** HEAD DIMS, asked of the PLANE rather than the scratch — so the
/// derivation a future widening needs survives the door's refusal instead of being deleted with it.
///
/// The plane's head corner is the pool's own `addr` for every plane and head at hd=128 as well as hd=64.
/// That is what makes the page-granular COPY expressible at two slabs; what is refused
/// ([`PageScratch::of_pass`]) is the INDEX serving both legs from one table there, on card evidence.
#[test]
fn the_plane_head_corner_is_the_pools_own_addr_at_both_head_dims() {
    let nkvh = NonZeroU32::new(8).unwrap();
    for hd in [64usize, 128] {
        let pool = PagedKvPool::new(8, hd);
        let plane_extent = PagePlaneExtent::of_pool(pool).expect("a stick-multiple head dim");
        assert_eq!(
            plane_extent.slabs(),
            hd as u32 / POOL_STICK,
            "hd={hd}: the slab count IS hd/stick, and it is the one quantity the door reads"
        );
        for plane in [KvPlane::Knat, KvPlane::V, KvPlane::Kt] {
            for h in 0..8u32 {
                let kvh = KvHead::new(h, nkvh).unwrap();
                assert_eq!(
                    plane_extent.head_off(h).expect("in range"),
                    pool.addr(KvCoord::block(plane, kvh)) as u64,
                    "hd={hd} {plane:?} head {h}: the plane's head corner disagrees with \
                     PagedKvPool::addr"
                );
            }
        }
        assert_eq!(
            plane_extent.head_off(8),
            None,
            "hd={hd}: past the head count must refuse"
        );
    }
}

/// THE ENTRY COUNT, which is the measured cause of the rung-8 corruption — swept over the WHOLE
/// `nkvh` x `hd` x `mq` grid, because every part of it used to be checked at one point of that grid.
///
/// ⭐ AND THE CUT IS THE INVARIANT, NOT THE OP COUNT. A request's run is `blocks()` entries and an op may
/// name `entries_per_op()` of them; the ops of one request are contiguous index STICKS and contiguous
/// destination entries, and every op's index fits the one stick dxp loads. That holds at nkvh=32/hd=256
/// (16 ops per request) exactly as at nkvh=8 (one), which is what the old `blocks() > ENTRIES_PER_OP`
/// refusal replaced with "no gather at all".
#[test]
fn every_copy_op_names_at_most_one_index_stick_at_every_geometry() {
    for nkvh in NKVH_SWEEP {
        for hd in HD_SWEEP {
            for mq in [1u32, 2, 4, 8, 16, 32] {
                let pool = PagedKvPool::new(nkvh, hd);
                let scratch = PageScratch::of_pass(pool, QueryRowCount::of_mq(mq))
                    .unwrap_or_else(|| panic!("nkvh={nkvh} hd={hd} mq={mq} must be admitted"));
                let at = format!("nkvh={nkvh} hd={hd} mq={mq}");
                assert_eq!(scratch.rows(), mq, "{at}: a row is a REQUEST");
                assert!(
                    scratch.rows() <= PageScratch::ENTRIES_PER_PASS_MAX,
                    "{at}: rows exceed the widest batch rung"
                );
                assert_eq!(scratch.entry_page(), scratch.plane().entry_sub_rows());
                assert_eq!(scratch.sub_rows(), mq as u64 * scratch.plane().sub_rows());
                let (cut, per_row) = (scratch.entries_per_op(), scratch.entries_per_row() as u32);
                assert!(
                    cut > 0 && cut <= CopyDims::ENTRIES_PER_OP,
                    "{at}: the cut is a positive number of entries no wider than the IBR stick"
                );
                assert_eq!(
                    scratch.ops_per_row(),
                    per_row.div_ceil(cut),
                    "{at}: ops per request is the run divided by the cut"
                );
                assert_eq!(
                    scratch.ops_per_row() == 1,
                    nkvh <= 8,
                    "{at}: one op per request iff the run fits one IBR stick — nkvh=8 is the boundary \
                     and `hd` has nothing to do with it"
                );
                let ops: Vec<_> = scratch.copies().collect();
                assert_eq!(
                    ops.len(),
                    mq as usize * scratch.ops_per_row() as usize,
                    "{at}: one copy op per (request, cut)"
                );
                // ⭐ EVERY OP, MEASURED FROM ITS OWN DECLARED EXTENTS (`mb / page`) — the division dxp
                // itself performs — and not from the scratch's row count, which is what the two sides
                // used to derive separately.
                let mut covered: Vec<u32> = Vec::new();
                for (i, cp) in ops.iter().enumerate() {
                    let (r, j) = (
                        i as u32 / scratch.ops_per_row(),
                        i as u32 % scratch.ops_per_row(),
                    );
                    let entries = cp.dims().entries().expect("page divides mb");
                    assert!(
                        cp.dims().fits_one_index_stick(),
                        "{at} r={r} j={j}: an op's index must fit the ONE stick dxp loads for it"
                    );
                    assert_eq!(
                        entries,
                        (per_row - j * cut).min(cut),
                        "{at} r={r} j={j}: the declared entry count is this op's own run"
                    );
                    // ⛔ ONE INDEX STICK PER OP. Two ops sharing a stick would give the second a base of
                    // `r*32 + cut` words, which is not a stick, and its 32-word load would straddle both.
                    assert_eq!(
                        cp.index_base().entries(),
                        (r * scratch.ops_per_row() + j) * CopyDims::ENTRIES_PER_OP,
                        "{at} r={r} j={j}: op (r,j)'s entries live at index stick r*ops + j"
                    );
                    // ⛔ AND THE DESTINATION IS **NOT** THE INDEX BASE. Entries advance one whole stick
                    // per op while the destination advances only the entries the op covers; deriving one
                    // from the other put every op's rows `blocks()`× off.
                    assert_eq!(
                        cp.dest_entry(),
                        r * per_row + j * cut,
                        "{at} r={r} j={j}: the destination entry is r*blocks + j*cut"
                    );
                    covered.extend(cp.dest_entry()..cp.dest_entry() + entries);
                }
                // ⭐ AND THE OPS TILE THE WHOLE SCRATCH EXACTLY ONCE — no gap (an unwritten run is read
                // as keys) and no overlap (one op's blocks over another's rows).
                let want: Vec<u32> = (0..mq * per_row).collect();
                assert_eq!(
                    covered, want,
                    "{at}: the pass's ops must tile destination entries [0, mq*blocks) in order, once"
                );
            }
            assert_eq!(
                PageScratch::of_pass(PagedKvPool::new(nkvh, hd), QueryRowCount::of_mq(33)),
                None,
                "nkvh={nkvh} hd={hd}: a rung wider than the ladder must refuse here"
            );
        }
    }
}

/// ⭐⭐⭐⭐⭐ **THE PAGE-GRANULAR DOOR ADMITS EVERY STICK-MULTIPLE HEAD DIM; THE WINDOW-GRANULAR ONE STILL
/// CANNOT.** The asymmetry is the whole reason page granularity was adopted, and it is now unconditional.
///
/// ⛔ THIS TEST HAS BEEN BOTH WAYS ROUND, AND THE HISTORY IS THE LESSON. It first asserted that page
/// granularity admits hd=128 (green, on the derivations above); then, after granite-3.1-8b came back
/// wrong from its first generated token with the gather live, it asserted the REFUSAL instead — also
/// green, and pinning as "correct" a door that did nothing at the only geometry anyone cared about. What
/// changed is neither assertion: the two `nslab`-blind defects in the FOLD's own legs were found and
/// fixed (a value leg that wrote only feature slab 0, a score leg that contracted two sticks under a
/// `y`-batch). See `PageScratch::of_pass`.
///
/// ⭐ SO THE DOOR IS TOTAL ON THE GEOMETRY, and a refusal at the emitter is a BUILD failure rather than a
/// silent ungathered bundle — which is what made "green at 2b, nothing at 8b" shippable.
#[test]
fn the_page_granular_door_admits_every_stick_multiple_head_dim() {
    let mq = QueryRowCount::of_mq(8);
    for nkvh in NKVH_SWEEP {
        for hd in HD_SWEEP {
            let pool = PagedKvPool::new(nkvh, hd);
            assert!(
                PageScratch::of_pass(pool, mq).is_some(),
                "nkvh={nkvh} hd={hd}: the page-granular door must admit every geometry a model can \
                 present — a refusal here is a bundle that silently drops the gather"
            );
            // The WINDOW-granular door is the contrast, and it is unchanged: a 64-slot window of V is
            // `nslab` runs apart above one stick, so its flat copy genuinely cannot express hd=128.
            assert_eq!(
                GatherScratch::admits(pool, mq),
                hd == POOL_STICK as usize,
                "nkvh={nkvh} hd={hd}: the window-granular door admits iff hd is one stick"
            );
            assert!(
                PagePlaneExtent::of_pool(pool).is_some(),
                "nkvh={nkvh} hd={hd}: the PLANE is expressible at every stick-multiple head dim"
            );
        }
    }
    // A head dim that is not a whole number of sticks has no stick-block entry at all, so it is the one
    // geometry the plane law itself refuses.
    assert_eq!(
        PagePlaneExtent::of_pool(PagedKvPool::new(8, 96)),
        None,
        "hd=96: a part-stick head dim has no stick-block entry, so the PLANE refuses it"
    );
}

/// The `hd`-cancellation the entry law rests on, over the whole grid — and the ONE quantity that does not
/// cancel, stated beside it so the two cannot be read as the same fact.
#[test]
fn the_entry_count_cancels_the_head_dim_but_the_entry_size_does_not() {
    for nkvh in NKVH_SWEEP {
        for hd in HD_SWEEP {
            let plane =
                PagePlaneExtent::of_pool(PagedKvPool::new(nkvh, hd)).expect("stick multiple");
            assert_eq!(
                plane.blocks(),
                nkvh as u64 * PagedKvPool::PAGE_SLOTS as u64 / POOL_STICK as u64,
                "nkvh={nkvh} hd={hd}: a request's entry count is nkvh * PAGE_SLOTS / stick — `hd` CANCELS"
            );
            assert_eq!(
                plane.entry_elems(),
                hd as u64 * POOL_STICK as u64,
                "nkvh={nkvh} hd={hd}: one entry is one stick BLOCK, the quantity that DOES move with hd"
            );
            assert_eq!(
                plane.slabs(),
                hd as u32 / POOL_STICK,
                "nkvh={nkvh} hd={hd}: the slab count IS hd/stick — the fold's legs loop it; the gather \
                 no longer reads it at all"
            );
        }
    }
}

/// ⭐⭐⭐⭐⭐ THE ENTRY REPRODUCES THE HOST'S PAGE ADDRESS EXACTLY — the identity, not a second derivation.
///
/// `host addr = phys * page_stride_bytes` and `card addr = idx * skip_addr + base`, with `skip_addr` ONE
/// STICK BLOCK, so entry `i` of physical page `phys` must satisfy
/// `entry * stick_block_bytes == phys * page_stride_bytes + i * stick_block_bytes`. Asserted rather than
/// reasoned about, because an entry off by a factor is a REAL address in another page: clean bake, no
/// fault, another request's keys.
///
/// ⛔ AND THE `i` TERM IS THE HALF THAT WENT MISSING. A page-granular entry with NO block term (`phys *
/// planes_per_page`) is what the plane-sized pin needed; against the stick-block `skip_addr` the card
/// actually derives, that entry is `nkvh * PAGE_SLOTS / POOL_STICK` = 32× SHORT — so for every physical
/// page below 32 it lands *inside page 0*, which is "correct for one page, wrong at the second" exactly.
#[test]
fn the_page_entry_reproduces_the_hosts_page_address() {
    const LAYERS: u64 = 40;
    const PLANES: u64 = 3;
    for hd in [64usize, 128] {
        let pool = PagedKvPool::new(8, hd);
        let plane = PagePlaneExtent::of_pool(pool).expect("admitted");
        let page_stride = LAYERS * PLANES * plane.bytes();
        let block_bytes = pool.stick_block_bytes();

        let per_page =
            gather_entries_per_page(page_stride, pool).expect("a whole number of stick blocks");
        assert_eq!(
            per_page.get() as u64,
            page_stride / block_bytes,
            "hd={hd}: the factor is stick-blocks-per-page"
        );

        for phys in [0i64, 1, 7, 128] {
            for i in [0u32, 1, 31] {
                let e = GatherEntry::of_page_block(per_page, phys, i).expect("in range");
                assert_eq!(
                    e.as_i32() as u64 * block_bytes,
                    phys as u64 * page_stride + i as u64 * block_bytes,
                    "hd={hd} phys={phys} i={i}: the entry does not reproduce the host's address"
                );
            }
        }
        assert_eq!(
            GatherEntry::of_page_block(per_page, -1, 0),
            None,
            "hd={hd}: a negative page must refuse"
        );
        assert_eq!(
            gather_entries_per_page(block_bytes + 1, pool),
            None,
            "hd={hd}: a page that is not a whole number of stick blocks must refuse, not round"
        );
    }
}

/// ⛔⛔⛔ THE FACTOR A PLANE-SIZED PIN WOULD HAVE NEEDED IS 32× SMALLER THAN THE ONE THE CARD USES, and
/// both are plain counts — so this is the substitution that has to stay unrepresentable.
///
/// The plane factor type (`PlanesPerPage`) and its entry (`PageEntry`) are DELETED, because the pin they
/// existed for is refused at bake. This test is what stops them coming back by accident: it states the
/// ratio, and therefore states that an entry built in plane units against the real `skip_addr` reaches
/// only `1/ratio` of the pool — i.e. page `phys/ratio`, a real address, clean bake, wrong keys.
#[test]
fn a_plane_unit_entry_would_reach_only_one_thirty_second_of_the_pool() {
    const LAYERS: u64 = 40;
    const PLANES: u64 = 3;
    for hd in [64usize, 128] {
        let pool = PagedKvPool::new(8, hd);
        let plane = PagePlaneExtent::of_pool(pool).expect("admitted");
        let page_stride = LAYERS * PLANES * plane.bytes();

        let sticks = gather_entries_per_page(page_stride, pool).expect("whole stick blocks");
        let planes = page_stride / plane.bytes();
        let ratio = plane.blocks();
        assert_eq!(
            ratio,
            8 * PagedKvPool::PAGE_SLOTS as u64 / POOL_STICK as u64,
            "hd={hd}: the ratio is nkvh*PAGE_SLOTS/POOL_STICK, and `hd` cancels out of it"
        );
        assert_eq!(
            sticks.get() as u64,
            planes * ratio,
            "hd={hd}: the stick-block factor is exactly `ratio` times the plane factor"
        );
        assert!(
            ratio > 1,
            "hd={hd}: the substitution is silent, not a no-op"
        );
    }
}
