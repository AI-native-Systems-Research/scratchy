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

/// The pin dxp derives `skip_addr` from, and SEN1P5+'s `DT_CHECK(skip_addr_sticks % 2 == 0)` — an odd pin
/// is a device ABORT, not a refusal, so it is asserted rather than assumed.
#[test]
fn the_pin_is_a_whole_even_number_of_sticks_at_both_head_dims() {
    for hd in [64usize, 128] {
        let extent = PagePlaneExtent::of_pool(PagedKvPool::new(8, hd)).expect("admitted");
        assert_eq!(
            extent.entry_sub_rows(),
            extent.entry_elems() / POOL_STICK as u64,
            "hd={hd}: the pinned position count is ONE STICK BLOCK in sticks"
        );
        // ⛔ AND IT IS **NOT** THE PLANE. This inequality is the card refusal in one line: a plane-sized
        // pin leaves one entry, therefore one core, therefore a double-buffered 512 KB chunk against LX.
        assert!(
            extent.entry_sub_rows() < extent.sub_rows(),
            "hd={hd}: the pin must be strictly smaller than the plane — a plane-sized pin is REFUSED at \
             bake (L3DlOpsScheduler.cpp:1534) because it admits no core split"
        );
        assert_eq!(
            extent.sub_rows(),
            extent.entry_sub_rows() * extent.blocks(),
            "hd={hd}: a request's row is a whole number of pinned entries, or the last entry covers a \
             partial block — which is an address"
        );
        assert!(
            extent.pin_is_even(),
            "hd={hd}: skip_addr is {} sticks, which is ODD — deeptools ABORTS on it",
            extent.skip_addr_sticks()
        );
    }
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
    for hd in [64usize, 128] {
        let pool = PagedKvPool::new(8, hd);
        let nkvh = NonZeroU32::new(8).unwrap();
        let scratch = PageScratch::of_pass(pool, QueryRowCount::of_mq(8)).expect("admitted");
        assert_eq!(
            scratch.cols(),
            8 * hd as u64 * PagedKvPool::PAGE_SLOTS as u64
        );

        for plane in [KvPlane::Knat, KvPlane::V, KvPlane::Kt] {
            for r in 0..scratch.mq() {
                for h in 0..8u32 {
                    let kvh = KvHead::new(h, nkvh).unwrap();
                    let pool_corner = pool.addr(KvCoord::block(plane, kvh)) as u64;
                    assert_eq!(
                        scratch.head_off(r, h).expect("in range"),
                        r as u64 * scratch.cols() + pool_corner,
                        "hd={hd} {plane:?} request {r} head {h}: the scratch's head corner disagrees \
                         with PagedKvPool::addr — the matmul would read another head's slots"
                    );
                }
            }
        }
        assert_eq!(
            scratch.head_off(8, 0),
            None,
            "hd={hd}: past the batch must refuse"
        );
        assert_eq!(
            scratch.head_off(0, 8),
            None,
            "hd={hd}: past the head count must refuse"
        );
    }
}

/// THE ENTRY COUNT, which is the measured cause of the rung-8 corruption. Page granularity makes it the
/// REQUEST count, so dxp's one-stick IBR cannot be overflowed by any rung the ladder admits.
#[test]
fn entries_per_pass_is_the_request_count_and_never_exceeds_one_index_stick() {
    for hd in [64usize, 128] {
        for mq in [1u32, 2, 4, 8, 16, 32] {
            let scratch = PageScratch::of_pass(PagedKvPool::new(8, hd), QueryRowCount::of_mq(mq))
                .unwrap_or_else(|| panic!("hd={hd} mq={mq} must be admitted"));
            assert_eq!(scratch.rows(), mq, "hd={hd} mq={mq}: a row is a REQUEST");
            assert!(
                scratch.rows() <= PageScratch::ENTRIES_PER_PASS_MAX,
                "hd={hd} mq={mq}: entries exceed ONE index stick, which WRAPS silently on card"
            );
            assert_eq!(scratch.entry_page(), scratch.plane().entry_sub_rows());
            assert_eq!(scratch.sub_rows(), mq as u64 * scratch.plane().sub_rows());
            // ⭐ ONE OP PER REQUEST, AND ITS INDEX IS EXACTLY ONE STICK. Both halves asserted from the
            // op's OWN declared extents (`mb / page`), because that division is what dxp performs — not
            // from the scratch's row count, which is what the two sides used to derive separately.
            let ops: Vec<_> = scratch.copies().collect();
            assert_eq!(
                ops.len(),
                mq as usize,
                "hd={hd} mq={mq}: one copy op per request"
            );
            for (r, cp) in ops.iter().enumerate() {
                assert_eq!(
                    cp.dims().entries(),
                    Some(scratch.entries_per_row() as u32),
                    "hd={hd} mq={mq} r={r}: the declared entry count is the plane's blocks"
                );
                assert!(
                    cp.dims().fits_one_index_stick(),
                    "hd={hd} mq={mq} r={r}: an op's index must fit the ONE stick dxp loads for it"
                );
                assert_eq!(
                    cp.index_base().entries(),
                    r as u32 * CopyDims::ENTRIES_PER_OP,
                    "hd={hd} mq={mq} r={r}: request r's entries live at index stick r"
                );
                // ⛔ AND THE DESTINATION IS **NOT** THE INDEX BASE. Entries advance one stick per op
                // while the destination advances a whole page plane; deriving one from the other put
                // every op's rows `blocks()`× off.
                assert_eq!(
                    cp.dest_entry(),
                    r as u32 * scratch.entries_per_row() as u32,
                    "hd={hd} mq={mq} r={r}: the destination entry is r * blocks, not r"
                );
            }
            assert!(
                scratch.entries_per_row() <= CopyDims::ENTRIES_PER_OP as u64,
                "hd={hd} mq={mq}: a request's own run exceeds one index stick, which WRAPS on card"
            );
        }
        assert_eq!(
            PageScratch::of_pass(PagedKvPool::new(8, hd), QueryRowCount::of_mq(33)),
            None,
            "hd={hd}: a rung wider than one index stick must refuse here, not wrap on the card"
        );
    }
}

/// ⭐ THE REGRESSION THIS WHOLE CHANGE EXISTS FOR, as a direct contrast in one place: the OLD
/// window-granular door refuses hd=128; the page-granular door admits it.
#[test]
fn page_granularity_admits_the_head_dim_the_window_granular_door_refused() {
    let mq = QueryRowCount::of_mq(8);
    for hd in [64usize, 128] {
        let pool = PagedKvPool::new(8, hd);
        assert_eq!(
            GatherScratch::admits(pool, mq),
            hd == POOL_STICK as usize,
            "hd={hd}: the window-granular door admits iff hd is one stick"
        );
        assert!(
            PageScratch::of_pass(pool, mq).is_some(),
            "hd={hd}: the PAGE-granular door must admit every stick-multiple head dim — this is the \
             entire point of the granularity change"
        );
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
