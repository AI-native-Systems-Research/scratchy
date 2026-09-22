// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ WHAT A GATHER AT `hd > POOL_STICK` REALLY COSTS — derived from
//! [`PagedKvPool::addr`], the one law, NOT from the prose in
//! [`GatherScratch`]'s note and NOT from arithmetic spelled at a call site.
//!
//! ## The three claims this file settles, in order
//!
//! 1. **Kᵗ needs nothing.** A 64-slot Kᵗ window is ONE contiguous `hd * POOL_STICK` run at every head
//!    dim, and its offset is an exact multiple of that run — so the SHIPPED entry unit
//!    ([`PagedKvPool::stick_block_elems`], the unit [`gather_entries_per_page`] divides a page by) names
//!    a Kᵗ window at hd=128 exactly as it does at hd=64. The refusal was never about Kᵗ.
//!
//! 2. **V's window stride is `POOL_STICK * POOL_STICK`, not `hd * POOL_STICK`.** V is `[cap, hd]`
//!    stick-major on `hd`, so its SLOT stride is ONE STICK (64 elements) — a 64-slot window advances
//!    `64 * 64 = 4096` elements, whatever `hd` is. At hd=128 that is HALF a stick block, so the shipped
//!    entry unit cannot name V's odd windows at all: `addr = idx * skip_addr + base` has no fractional
//!    `idx`. **A `skip_addr` of one stick block is therefore refuted at hd=128, and the hypothesis that
//!    one entry can carry `window + 2 * slab` in stick-block units with it is refuted with it.** The
//!    entry unit that works for BOTH planes is `POOL_STICK * POOL_STICK`, which is the stick block at
//!    hd=64 and half of it at hd=128.
//!
//! 3. **And that is not enough, because the two planes then number the same blocks DIFFERENTLY.** In
//!    `POOL_STICK * POOL_STICK` units, Kᵗ's (window `b`, slab `s`) is `b * nslab + s` inside a kv head
//!    while V's is `s * blocks_per_page + b`. Those are the SAME SET of blocks — so the gather moves
//!    exactly the same bytes — in a DIFFERENT ORDER, and they coincide for every coordinate **iff
//!    `nslab == 1`**. One index table serves both copies at hd=64 by that coincidence and cannot at
//!    hd=128: the two legs need two tables (or two sections of one), and a section base derived from the
//!    live window COUNT is the emitter/host disagreement already recorded on
//!    [`GatherScratch::row_of`]. That — not a second `skip_addr` — is the real price.
//!
//! ⛔ NO PART OF THIS FILE SPELLS AN ADDRESS. Every offset comes from `pool.addr(KvCoord…)`, so if the
//! pool's layout moves, this file moves with it or fails.

use ktir_superdsc::addr::Gqa;
use ktir_superdsc::sdsc_abstract::{
    FeatIdx, GatherScratch, KvCoord, KvHead, KvPlane, KvSlot, POOL_STICK, PagedKvPool, QueryHead,
    QueryRowCount, SlotCount, SlotWindow,
};
use std::num::NonZeroU32;

/// granite-3.1-2b (hd=64, the one geometry the gather has run at) and granite-3.1-8b fp8 (hd=128, the
/// 70-config slice the refusal excludes). `nkvh` is 8 on both.
const NKVH: usize = 8;
const HD_2B: usize = 64;
const HD_8B: usize = 128;

/// The elements one 64-slot window occupies in ONE feature slab — `POOL_STICK * POOL_STICK`, and the
/// only thing it is read off is the stick itself.
const SLAB_WINDOW_ELEMS: u32 = POOL_STICK * POOL_STICK;

fn head(kvh: u32) -> KvHead {
    // Head `kvh` of an 8-head model, through the one door that bounds it.
    KvHead::of_query(
        QueryHead::all(NonZeroU32::new(NKVH as u32).unwrap())
            .nth(kvh as usize)
            .expect("kvh < nkvh"),
        Gqa::new(1),
    )
}

/// The plane-relative element offset of (kv head, 64-slot window, feature slab)'s CORNER, from the
/// pool's own address law.
fn corner(pool: &PagedKvPool, plane: KvPlane, kvh: u32, window: u32, slab: u32) -> u32 {
    pool.addr(
        KvCoord::block(plane, head(kvh))
            .at_slot(KvSlot::new(window * POOL_STICK))
            .at_feat(FeatIdx::of_slab(slab)),
    )
}

/// ⭐⭐⭐ CLAIM 1 — A Kᵗ WINDOW IS ONE CONTIGUOUS STICK BLOCK AT EVERY HEAD DIM, and its offset is an
/// exact multiple of one. So the shipped entry unit already names it.
#[test]
fn a_kt_window_is_one_contiguous_stick_block_at_every_head_dim() {
    for hd in [HD_2B, HD_8B] {
        let pool = PagedKvPool::new(NKVH, hd);
        let unit = pool.stick_block_elems() as u32;
        assert_eq!(
            unit,
            hd as u32 * POOL_STICK,
            "hd={hd}: the stick block is `hd * POOL_STICK` by definition"
        );
        for kvh in 0..NKVH as u32 {
            for b in 0..(PagedKvPool::PAGE_SLOTS as u32 / POOL_STICK) {
                let base = corner(&pool, KvPlane::Kt, kvh, b, 0);
                // CONTIGUOUS: every (feature, slot) of the window lands inside `[base, base + unit)`,
                // and the window's elements are exactly that range — checked by walking the law.
                let mut seen = vec![false; unit as usize];
                for d in 0..hd as u32 {
                    for j in 0..POOL_STICK {
                        let at = pool.addr(
                            KvCoord::block(KvPlane::Kt, head(kvh))
                                .at_slot(KvSlot::new(b * POOL_STICK + j))
                                .at_feat(FeatIdx::new(d)),
                        );
                        let rel = at.checked_sub(base).expect("inside the window");
                        assert!(
                            rel < unit,
                            "hd={hd} kvh={kvh} b={b} d={d} j={j}: a Kᵗ window must be ONE run of \
                             {unit} elements; this element is {rel} past its corner"
                        );
                        // And the run's internal order is the score kernel's own: `d * 64 + j`.
                        assert_eq!(
                            rel,
                            d * POOL_STICK + j,
                            "hd={hd}: a Kᵗ window's internal order must be `feat * POOL_STICK + slot` \
                             — that is what `Stk::kernel(hd, 64)` reads"
                        );
                        seen[rel as usize] = true;
                    }
                }
                assert!(
                    seen.into_iter().all(|x| x),
                    "hd={hd} kvh={kvh} b={b}: the window must TILE its run with no hole"
                );
                // ⭐ AND THE CORNER IS AN EXACT MULTIPLE OF THE UNIT — so one integer names it.
                assert_eq!(
                    base % unit,
                    0,
                    "hd={hd} kvh={kvh} b={b}: Kᵗ window corner {base} is not a whole number of \
                     {unit}-element blocks, so no index entry can name it"
                );
            }
        }
    }
}

/// ⭐⭐⭐⭐⭐ CLAIM 2 — THE V WINDOW STRIDE IS `POOL_STICK * POOL_STICK`, AND AT hd=128 THE SHIPPED
/// STICK-BLOCK ENTRY CANNOT NAME HALF THE WINDOWS.
///
/// ⛔ THIS IS THE ARITHMETIC THAT WAS ASSUMED AND IS WRONG. The hypothesis under test was that V's slab
/// stride (`PLANE_SLOTS * POOL_STICK` = 16384) is an exact multiple of the stick block (8192), so one
/// entry in stick-block units could carry `window + 2 * slab`. The slab stride is indeed a multiple —
/// but the WINDOW stride is 4096, half a stick block, so window 1 has no stick-block number at all.
#[test]
fn the_v_window_stride_is_one_slab_window_and_refutes_a_stick_block_entry_at_hd_128() {
    for hd in [HD_2B, HD_8B] {
        let pool = PagedKvPool::new(NKVH, hd);
        let nslab = hd as u32 / POOL_STICK;
        let unit = pool.stick_block_elems() as u32;
        let head_block = pool.plane_block_elems() as u32;
        let blocks_per_page = PagedKvPool::PAGE_SLOTS as u32 / POOL_STICK;

        // The slab stride, measured: `PLANE_SLOTS * POOL_STICK`.
        if nslab > 1 {
            assert_eq!(
                corner(&pool, KvPlane::V, 0, 0, 1) - corner(&pool, KvPlane::V, 0, 0, 0),
                PagedKvPool::PLANE_SLOTS as u32 * POOL_STICK,
                "hd={hd}: V's feature-slab stride is the plane's slots times one stick"
            );
        }
        // ⛔ AND THE WINDOW STRIDE IS `POOL_STICK * POOL_STICK`, NOT `hd * POOL_STICK`. This is the
        // number the hypothesis got wrong, and it does not depend on `hd` at all.
        assert_eq!(
            corner(&pool, KvPlane::V, 0, 1, 0) - corner(&pool, KvPlane::V, 0, 0, 0),
            SLAB_WINDOW_ELEMS,
            "hd={hd}: V is stick-major on `hd`, so its SLOT stride is one stick and a 64-slot window \
             advances {SLAB_WINDOW_ELEMS} elements — `hd * POOL_STICK` is the Kᵗ number"
        );

        // A (window, slab) run of V is CONTIGUOUS and exactly `SLAB_WINDOW_ELEMS` long, in the value
        // kernel's own internal order (`slot * 64 + feat % 64`) — so the flat copy is legal per slab.
        for kvh in 0..NKVH as u32 {
            for s in 0..nslab {
                for b in 0..blocks_per_page {
                    let base = corner(&pool, KvPlane::V, kvh, b, s);
                    for j in 0..POOL_STICK {
                        for f in 0..POOL_STICK {
                            let at = pool.addr(
                                KvCoord::block(KvPlane::V, head(kvh))
                                    .at_slot(KvSlot::new(b * POOL_STICK + j))
                                    .at_feat(FeatIdx::new(s * POOL_STICK + f)),
                            );
                            assert_eq!(
                                at - base,
                                j * POOL_STICK + f,
                                "hd={hd} kvh={kvh} b={b} s={s}: a V (window, slab) run must be \
                                 contiguous in `slot * POOL_STICK + feat % POOL_STICK` order — that \
                                 is what `Stk::kernel(64, hd)` reads at slab `s`"
                            );
                        }
                    }
                    // ⭐ EXPRESSIBLE IN `POOL_STICK * POOL_STICK` UNITS AT EVERY HEAD DIM …
                    assert_eq!(
                        base % SLAB_WINDOW_ELEMS,
                        0,
                        "hd={hd}: every V (window, slab) corner is a whole number of \
                         {SLAB_WINDOW_ELEMS}-element blocks"
                    );
                    // … AND **NOT** IN STICK-BLOCK UNITS ONCE `hd` EXCEEDS ONE STICK.
                    if nslab > 1 && b % 2 == 1 {
                        assert_ne!(
                            base % unit,
                            0,
                            "hd={hd} kvh={kvh} b={b} s={s}: V window corner {base} IS a multiple of \
                             the {unit}-element stick block — the premise that it is not is what \
                             makes the finer entry necessary"
                        );
                    }
                }
            }
        }
        // The whole head block is a whole number of slab windows, so an entry reaches any cell.
        assert_eq!(
            head_block % SLAB_WINDOW_ELEMS,
            0,
            "hd={hd}: a kv head's block must be a whole number of slab windows"
        );
        assert_eq!(
            head_block / SLAB_WINDOW_ELEMS,
            blocks_per_page * nslab,
            "hd={hd}: a kv head holds `blocks_per_page * nslab` slab windows"
        );
    }
}

/// ⭐⭐⭐⭐⭐ CLAIM 3 — IN THE ONE UNIT THAT WORKS FOR BOTH PLANES, THE TWO PLANES NUMBER THE SAME
/// BLOCKS DIFFERENTLY, AND THEY COINCIDE EXACTLY AT `nslab == 1`.
///
/// This is why "an index entry per (window, slab)" is not the whole change. The entry VALUES diverge,
/// so one table cannot serve both copies — and the SET of values is identical, which is the reassuring
/// half: the gather still moves exactly the same blocks, in a different order per plane.
#[test]
fn the_two_planes_block_numbering_coincides_exactly_at_one_slab() {
    for hd in [HD_2B, HD_8B] {
        let pool = PagedKvPool::new(NKVH, hd);
        let nslab = hd as u32 / POOL_STICK;
        let blocks_per_page = PagedKvPool::PAGE_SLOTS as u32 / POOL_STICK;
        let mut agree = true;
        for kvh in 0..NKVH as u32 {
            let mut kt_numbers = vec![];
            let mut v_numbers = vec![];
            for b in 0..blocks_per_page {
                for s in 0..nslab {
                    let kt = corner(&pool, KvPlane::Kt, kvh, b, s) / SLAB_WINDOW_ELEMS;
                    let v = corner(&pool, KvPlane::V, kvh, b, s) / SLAB_WINDOW_ELEMS;
                    // The two laws, stated — and derived from `addr`, not asserted onto it.
                    let head_first = kvh * blocks_per_page * nslab;
                    assert_eq!(
                        kt,
                        head_first + b * nslab + s,
                        "hd={hd}: Kᵗ numbers a (window, slab) `b * nslab + s` inside its head"
                    );
                    assert_eq!(
                        v,
                        head_first + s * blocks_per_page + b,
                        "hd={hd}: V numbers a (window, slab) `s * blocks_per_page + b` inside its head"
                    );
                    agree &= kt == v;
                    kt_numbers.push(kt);
                    v_numbers.push(v);
                }
            }
            kt_numbers.sort_unstable();
            v_numbers.sort_unstable();
            assert_eq!(
                kt_numbers, v_numbers,
                "hd={hd} kvh={kvh}: the two planes must gather the SAME SET of blocks — only the \
                 order may differ"
            );
        }
        assert_eq!(
            agree,
            nslab == 1,
            "hd={hd} nslab={nslab}: the two planes' block numbering coincides for every coordinate \
             IFF there is one slab. At nslab>1 one index table cannot serve both copies."
        );
    }
}

/// ⭐ AND THE SHIPPED hd=64 EMISSION IS EXACTLY THE `POOL_STICK * POOL_STICK` UNIT ALREADY — so
/// adopting the finer entry costs the proven geometry NOTHING.
///
/// `GatherScratch::entry_page` is `cols / POOL_STICK` = `hd` sub-rows today; at hd=64 that IS
/// `POOL_STICK` sub-rows, and `stick_block_elems` IS `POOL_STICK * POOL_STICK`. Stated as a test so the
/// claim "hd=64 is byte-identical under the finer unit" is checked rather than argued.
#[test]
fn the_shipped_entry_unit_at_hd_64_is_already_one_slab_window() {
    let pool = PagedKvPool::new(NKVH, HD_2B);
    assert_eq!(pool.stick_block_elems() as u32, SLAB_WINDOW_ELEMS);
    let scratch = GatherScratch::of_fold_pass(
        pool,
        SlotWindow::count_in(SlotCount::new(PagedKvPool::PAGE_SLOTS as u32)),
        QueryRowCount::of_mq(8),
    )
    .expect("hd=64 admits the flat copy");
    assert_eq!(
        scratch.entry_page(),
        POOL_STICK,
        "at hd=64 the shipped pin is already ONE STICK of sub-rows"
    );
    assert_eq!(
        scratch.cols(),
        SLAB_WINDOW_ELEMS,
        "and a scratch row is already exactly one slab window"
    );
    // `block_in_page` is the shipped law, and it is both planes' law at one slab.
    for kvh in 0..NKVH as u32 {
        for b in 0..(PagedKvPool::PAGE_SLOTS as u32 / POOL_STICK) {
            let want = corner(&pool, KvPlane::Kt, kvh, b, 0) / SLAB_WINDOW_ELEMS;
            assert_eq!(
                scratch.block_in_page(kvh, b),
                Some(want),
                "the shipped `block_in_page` must be the pool's own Kᵗ block number"
            );
            assert_eq!(
                corner(&pool, KvPlane::V, kvh, b, 0) / SLAB_WINDOW_ELEMS,
                want,
                "and V's at one slab, which is why one table serves both copies today"
            );
        }
    }
}
