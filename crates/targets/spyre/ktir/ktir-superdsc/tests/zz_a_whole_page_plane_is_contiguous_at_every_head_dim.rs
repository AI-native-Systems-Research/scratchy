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
    FeatIdx, KvCoord, KvHead, KvPlane, KvSlot, POOL_STICK, PagePlaneExtent, PagedKvPool,
    gather_entries_per_page, gather_entry_ceiling,
};
use std::collections::BTreeSet;
use std::num::NonZeroU32;

/// ⭐⭐⭐⭐⭐ THE KV-HEAD SWEEP THAT DID NOT EXIST. Every `of_pass` call site in this suite passed 8, and
/// `PagePlaneExtent::blocks` is `nkvh * PAGE_SLOTS / POOL_STICK` — 32 at nkvh=8, which is ALSO
/// `CopyDims::ENTRIES_PER_OP`. Two unrelated quantities with one value, never separated by a test, and
/// the one that moves is this axis.
const NKVH_SWEEP: [usize; 4] = [4, 8, 16, 32];

/// And the head-dim sweep, which must include a dim where `hd != POOL_STICK` (the pin) and one where
/// `nslab > 2` (the fold's slab loops).
const HD_SWEEP: [usize; 3] = [64, 128, 256];

/// Every address of one page's plane, over every kv head, slot and feature.
fn plane_addrs(pool: &PagedKvPool, plane: KvPlane) -> Vec<u32> {
    let nkvh = NonZeroU32::new(pool.nkvh()).expect("nkvh > 0");
    let mut out = Vec::new();
    for h in 0..pool.nkvh() {
        let kvh = KvHead::new(h, nkvh).expect("kv head in range");
        for slot in 0..PagedKvPool::PAGE_SLOTS as u32 {
            for feat in 0..pool.hd() {
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
        let pool = PagedKvPool::new(8, hd as u32);
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
/// a runtime `bool` the (since-deleted) page scratch door branched on, and its test sat inside an
/// hd-multiple-of-64 loop — so it asserted its own premise and could never fail. The pin is `hd` sticks
/// and `of_pool` admits only whole multiples of `POOL_STICK`, so "the pin is even" is a statement about
/// the STICK, and it now lives where a statement about a constant belongs: a module-level
/// `const _: () = assert!(POOL_STICK.is_multiple_of(2), ..)` in `sdsc_abstract`, which is a
/// required-const context and therefore actually evaluates.
#[test]
fn the_pin_is_one_stick_block_at_every_head_dim_and_kv_head_count() {
    for nkvh in NKVH_SWEEP {
        for hd in HD_SWEEP {
            let pool = PagedKvPool::new(nkvh as u32, hd as u32);
            let extent = PagePlaneExtent::of_pool(pool).expect("admitted");
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

/// CLAIM 3, the contrast: the OLD granularity. A 64-slot WINDOW of `Knat`/`V` is one contiguous run only
/// at `hd == POOL_STICK`. This is the refusal's actual cause, pinned so it cannot be re-derived as a
/// property of the gather or of the hardware.
#[test]
fn a_sixty_four_slot_window_is_contiguous_only_at_one_stick_which_is_why_the_old_door_refused() {
    let nkvh = NonZeroU32::new(8).unwrap();
    for hd in [64usize, 128] {
        let pool = PagedKvPool::new(8, hd as u32);
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

/// ⭐⭐⭐⭐⭐ THE PLANE'S HEAD CORNER IS THE POOL'S OWN `addr` AT **BOTH** HEAD DIMS — for every plane and
/// head at hd=128 as well as hd=64, which is what makes the per-slab block gather expressible at two
/// slabs.
#[test]
fn the_plane_head_corner_is_the_pools_own_addr_at_both_head_dims() {
    let nkvh = NonZeroU32::new(8).unwrap();
    for hd in [64usize, 128] {
        let pool = PagedKvPool::new(8, hd as u32);
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

/// The `hd`-cancellation the entry law rests on, over the whole grid — and the ONE quantity that does not
/// cancel, stated beside it so the two cannot be read as the same fact.
#[test]
fn the_entry_count_cancels_the_head_dim_but_the_entry_size_does_not() {
    for nkvh in NKVH_SWEEP {
        for hd in HD_SWEEP {
            let pool = PagedKvPool::new(nkvh as u32, hd as u32);
            let plane = PagePlaneExtent::of_pool(pool).expect("stick multiple");
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

/// ⛔⛔⛔ THE FACTOR A PLANE-SIZED PIN WOULD HAVE NEEDED IS `nkvh · W4 · SLABS`× SMALLER THAN THE ONE THE
/// CARD USES, and both are plain counts — so this is the substitution that has to stay unrepresentable.
///
/// The plane factor type (`PlanesPerPage`) and its entry (`PageEntry`) are DELETED, because the pin they
/// existed for is refused at bake. This test is what stops them coming back by accident: it states the
/// ratio, and therefore states that an entry built in plane units against the real `skip_addr` reaches
/// only `1/ratio` of the pool — i.e. page `phys/ratio`, a real address, clean bake, wrong keys.
///
/// ⭐ THE ENTRY IS ONE SLAB BLOCK (`POOL_STICK²` at every head dim — `GatherEntry::of_slab_block`'s
/// unit), so `hd` does NOT cancel: one page plane is `nkvh · W4 · SLABS` entries — 32 at hd 64, **64 at
/// hd 128** — which is exactly why the entries-per-page factor is a bundle fact and never a constant.
#[test]
fn a_plane_unit_entry_would_reach_only_a_slab_fraction_of_the_pool() {
    const LAYERS: u64 = 40;
    const PLANES: u64 = 3;
    for hd in [64usize, 128] {
        let pool = PagedKvPool::new(8, hd as u32);
        let plane = PagePlaneExtent::of_pool(pool).expect("admitted");
        let page_stride = LAYERS * PLANES * plane.bytes();

        let entries = gather_entries_per_page(page_stride, pool).expect("whole slab blocks");
        let planes = page_stride / plane.bytes();
        let ratio = plane.elems() / pool.slab_block_elems() as u64;
        let slabs = pool.window_slabs().expect("a stick-multiple head dim") as u64;
        assert_eq!(
            ratio,
            8 * PagedKvPool::WINDOWS_PER_PAGE as u64 * slabs,
            "hd={hd}: one page plane is nkvh·W4·SLABS slab blocks — `hd` rides SLABS"
        );
        assert_eq!(
            entries.get() as u64,
            planes * ratio,
            "hd={hd}: the slab-block factor is exactly `ratio` times the plane factor"
        );
        assert!(
            ratio > 1,
            "hd={hd}: the substitution is silent, not a no-op"
        );
    }
}

/// ⭐⭐⭐⭐⭐ THE HOST'S ENTRY REPRODUCES THE HOST'S PAGE ADDRESS — `GatherEntry::of_slab_block`, the ONE
/// composition the id tables are built from, against `page_base_bytes`' own law (`phys ·
/// page_stride`) plus the slab block's place inside the page through `PagedKvPool::addr`. dxp turns
/// the value into an address as `idx · skip_addr + base` with `skip_addr` one slab block, so an entry
/// whose value times the slab block is not the host's byte address reads another block — a clean
/// bake, wrong keys. Swept over both planes, both head dims, and every (kv head, window, slab).
#[test]
fn the_slab_entry_reproduces_the_hosts_page_address() {
    use ktir_superdsc::sdsc_abstract::GatherEntry;
    const LAYERS: u64 = 40;
    const PLANES: u64 = 3;
    for hd in [64usize, 128] {
        let pool = PagedKvPool::new(8, hd as u32);
        let plane = PagePlaneExtent::of_pool(pool).expect("admitted");
        let page_stride = LAYERS * PLANES * plane.bytes();
        let block_bytes = pool.slab_block_bytes();
        let block_elems = pool.slab_block_elems() as u64;
        // The pool's own ceiling — 256 pages of budget, past every page the sweep below names.
        let ceiling =
            gather_entry_ceiling(256, page_stride, pool).expect("a whole number of slab blocks");
        assert_eq!(
            ceiling.per_page().get() as u64,
            page_stride / block_bytes,
            "hd={hd}: the factor is slab-blocks-per-page"
        );
        let nkvh = NonZeroU32::new(pool.nkvh()).unwrap();
        let slabs = pool.window_slabs().expect("a stick-multiple head dim");
        for plane in [KvPlane::Kt, KvPlane::V] {
            for phys in [0i64, 1, 7, 128] {
                for h in 0..pool.nkvh() {
                    for w in 0..pool.windows_per_page() {
                        for s in 0..slabs {
                            let e = GatherEntry::of_slab_block(plane, pool, ceiling, phys, h, w, s)
                                .expect("in range");
                            // The slab block's first element inside one page plane, by the pool's
                            // own address law — window w's first slot, slab s's first feature.
                            let kvh = KvHead::new(h, nkvh).unwrap();
                            let corner = pool.addr(
                                KvCoord::block(plane, kvh)
                                    .at_slot(KvSlot::new(w * POOL_STICK))
                                    .at_feat(FeatIdx::new(s * POOL_STICK)),
                            ) as u64;
                            assert_eq!(
                                corner % block_elems,
                                0,
                                "hd={hd} {plane:?} h={h} w={w} s={s}: the block corner is whole blocks"
                            );
                            assert_eq!(
                                e.as_i32() as u64 * block_bytes,
                                phys as u64 * page_stride + corner * 2,
                                "hd={hd} {plane:?} phys={phys} h={h} w={w} s={s}: the entry does not \
                                 reproduce the host's address"
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(
            GatherEntry::of_slab_block(KvPlane::Kt, pool, ceiling, -1, 0, 0, 0),
            None,
            "hd={hd}: a negative page must refuse"
        );
        assert_eq!(
            GatherEntry::of_slab_block(KvPlane::Kt, pool, ceiling, 256, 0, 0, 0),
            None,
            "hd={hd}: a page the pool's budget does not hold must be unconstructible"
        );
        assert_eq!(
            GatherEntry::of_slab_block(KvPlane::V, pool, ceiling, 0, 0, 0, slabs),
            None,
            "hd={hd}: a slab past the head dim must be unconstructible"
        );
        assert_eq!(
            GatherEntry::of_slab_block(KvPlane::Knat, pool, ceiling, 0, 0, 0, 0),
            None,
            "hd={hd}: no gathered reader names the natural-K plane"
        );
        assert_eq!(
            gather_entries_per_page(block_bytes + 1, pool),
            None,
            "hd={hd}: a page that is not a whole number of slab blocks must refuse, not round"
        );
    }
}

/// ⭐ THE PLANE IS EXPRESSIBLE AT EVERY STICK-MULTIPLE HEAD DIM, and a part-stick head dim is the one
/// geometry the plane law refuses — the admission every gathered reader of the pool rides.
#[test]
fn the_plane_admits_every_stick_multiple_head_dim() {
    for nkvh in NKVH_SWEEP {
        for hd in HD_SWEEP {
            let pool = PagedKvPool::new(nkvh as u32, hd as u32);
            assert!(
                PagePlaneExtent::of_pool(pool).is_some(),
                "nkvh={nkvh} hd={hd}: the PLANE is expressible at every stick-multiple head dim"
            );
            assert_eq!(
                pool.window_slabs(),
                Some(hd as u32 / POOL_STICK),
                "nkvh={nkvh} hd={hd}: a window spans hd/64 slabs"
            );
        }
    }
    assert_eq!(
        PagePlaneExtent::of_pool(PagedKvPool::new(8, 96)),
        None,
        "hd=96: a part-stick head dim has no slab-block entry, so the PLANE refuses it"
    );
}
