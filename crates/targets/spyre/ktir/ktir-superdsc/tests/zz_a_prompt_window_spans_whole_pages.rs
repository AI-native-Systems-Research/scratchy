// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ A PROMPT WINDOW MAY SPAN WHOLE PAGES — the pool laws a 512-row prefill chunk rests on.
//!
//! One page used to be the widest window: a chunk's cache write was ONE `[mq, hd]` copy, resolved to
//! ONE page, so a padded write past the page end landed on the next kv head's keys. The write is now
//! cut into one part per page, each aimed at its own page through the request's block table, so the
//! window may span `PREFILL_WINDOW_PAGES` WHOLE pages — never part of one. These pin the three host
//! laws that keep a part from ever straddling a page, and the map that gives every written page a host
//! block of its own.

use ktir_superdsc::sdsc_abstract::{
    BlockTable, ChunkKvExtent, KvHistory, KvSlot, PagedKvPool, PoolPages, PoolPartition, PoolRows,
    SlotCount,
};

const PAGE: u32 = PagedKvPool::PAGE_SLOTS as u32;
const WINDOW: u32 = PagedKvPool::PREFILL_CHUNK_SLOTS as u32;

/// THE MULTI-PAGE LAW ITSELF: the widest window is a whole number of pages, and that number is
/// `PREFILL_WINDOW_PAGES` — two, at the shipped 512.
#[test]
fn the_widest_window_is_a_whole_number_of_pages() {
    assert_eq!(PagedKvPool::PREFILL_CHUNK_SLOTS, 512);
    assert_eq!(PagedKvPool::PREFILL_WINDOW_PAGES, 2);
    assert_eq!(
        PagedKvPool::PREFILL_WINDOW_PAGES * PagedKvPool::PAGE_SLOTS,
        PagedKvPool::PREFILL_CHUNK_SLOTS,
        "a window wider than a page is a whole number of pages"
    );
}

/// ⭐ THE ROOM: a chunk starting at a page boundary may fill the whole window; one starting mid-page
/// stays inside its page, because a part addresses the page its first row falls in.
#[test]
fn a_page_aligned_chunk_may_fill_the_window_and_a_mid_page_one_its_page() {
    for page in 0..4u32 {
        assert_eq!(
            PagedKvPool::chunk_room(KvSlot::new(page * PAGE)).get(),
            WINDOW,
            "a chunk at page {page}'s start may fill both pages of its window"
        );
        for off in [1u32, 64, 160, 255] {
            assert_eq!(
                PagedKvPool::chunk_room(KvSlot::new(page * PAGE + off)).get(),
                PAGE - off,
                "a chunk at offset {off} of page {page} stays inside that page"
            );
        }
    }
}

/// ⭐ THE STEP-BACK: a padded window wider than a page starts at a page boundary — `want`'s own page
/// start — and one that fits a page keeps the old law (the last offset whose window still fits).
#[test]
fn a_multi_page_padded_window_steps_back_to_its_page_start() {
    let place = |want: u32, padded: u32| {
        PagedKvPool::chunk_write_start(KvSlot::new(want), SlotCount::new(padded))
            .slot()
            .get()
    };
    // Aligned: stays.
    assert_eq!(place(512, WINDOW), 512);
    assert_eq!(place(0, WINDOW), 0);
    // Mid-page: back to the page start, never forward, never to another page.
    assert_eq!(place(600, WINDOW), 512);
    assert_eq!(place(767, WINDOW), 512);
    assert_eq!(place(257, 384), 256);
    // A one-page padded window is unchanged — the last start whose window still fits the page.
    assert_eq!(place(192, 96), 160);
    assert_eq!(place(100, 96), 100);
    assert_eq!(place(300, 256), 256);
    // And the chunk the step-back places always has the whole window of room.
    for want in [512u32, 600, 767, 1000] {
        let at = PagedKvPool::chunk_write_start(KvSlot::new(want), SlotCount::new(WINDOW));
        assert_eq!(
            PagedKvPool::chunk_room(at.slot()).get(),
            WINDOW,
            "want={want}"
        );
    }
}

/// ⭐ THE WORKER'S CHUNK LOOP AT THE SHIPPED WIDTH: every continuation chunk starts at a page boundary,
/// every mid-prompt chunk is exactly the bundle's 512 rows (so `ChunkKvExtent::exact` holds — padding
/// only ever trails the prompt), and the history ends at the prompt.
#[test]
fn a_prompt_runs_in_whole_windows_from_page_boundaries() {
    for total in (1usize..=1200).chain([4096, 10_300, 17_500]) {
        let mut hist = KvHistory::default();
        let (mut tok, mut chunk_start, mut first) = (0usize, 0usize, true);
        while tok < total {
            let padded = SlotCount::new(if first { 0 } else { WINDOW });
            let placed = PagedKvPool::chunk_write_start(KvSlot::new(chunk_start as u32), padded);
            let back = chunk_start - placed.slot().get() as usize;
            chunk_start -= back;
            tok -= back;
            assert!(
                first || chunk_start.is_multiple_of(PAGE as usize),
                "total={total}: a continuation chunk at {chunk_start} is not page-aligned"
            );
            let room = PagedKvPool::chunk_room(placed.slot()).get() as usize;
            let chunk_len = (total - tok).min(WINDOW as usize).min(room);
            if tok + chunk_len < total {
                assert!(
                    ChunkKvExtent::exact(chunk_len as u32, WINDOW).is_some(),
                    "total={total}: a mid-prompt chunk of {chunk_len} rows on the {WINDOW}-row bundle"
                );
            }
            hist.record_chunk(placed.wrote(SlotCount::new(chunk_len as u32)));
            chunk_start += chunk_len;
            tok += chunk_len;
            first = false;
        }
        assert_eq!(hist.end().get() as usize, total, "total={total}");
        assert_eq!(hist.runs().len(), 1, "total={total}: one contiguous run");
    }
}

fn partition() -> PoolPartition {
    PoolPartition::of_pool(PoolPages::of_pool(136).unwrap(), PoolRows::WIDEST)
        .expect("136 pages fund it")
}

fn phys(t: &BlockTable) -> Vec<u32> {
    t.pages().iter().map(|p| p.0).collect()
}

/// ⭐ EVERY PAGE A CHUNK WRITES IS A HOST BLOCK. A chunk at 512 of a request holding `[0, 512)` writes
/// pages 2 and 3; `map_row` backs only page 2 (the history's next slot) and would leave page 3 out of
/// the map entirely — so part 1's real rows would have no page. `map_chunk` backs both.
#[test]
fn a_two_page_chunk_maps_both_of_its_pages_to_host_blocks() {
    let part = partition();
    let host = [10usize, 11, 12, 13, 14];
    let hist = KvHistory::contiguous(512);
    let full = PagedKvPool::chunk_write_start(KvSlot::new(512), SlotCount::new(WINDOW))
        .wrote(SlotCount::new(WINDOW));
    let map = BlockTable::map_chunk(&hist, &host, full, part).expect("mappable");
    assert_eq!(
        phys(&map),
        vec![10, 11, 12, 13],
        "pages 0-1 held, pages 2-3 written"
    );
    // The pages holding keys get the same host blocks `map_row` gives them — the gather table, built
    // through `map_row`, names what this map installed.
    let row = BlockTable::map_row(
        &hist,
        &host,
        ktir_superdsc::sdsc_abstract::RowPages::holding(SlotCount::new(513)),
        part,
    )
    .expect("mappable");
    assert_eq!(phys(&row), vec![10, 11, 12]);
    assert_eq!(phys(&map)[..3], phys(&row)[..]);
}

/// ⛔ A SHORT LAST CHUNK'S MAP STOPS AT ITS LAST REAL ROW — so the part holding only padding past it
/// finds no page (the launch skips it) rather than a page the request never asked for.
#[test]
fn a_short_last_chunk_maps_no_page_past_its_real_rows() {
    let part = partition();
    let host = [10usize, 11, 12, 13, 14];
    let hist = KvHistory::contiguous(512);
    let tail = PagedKvPool::chunk_write_start(KvSlot::new(512), SlotCount::new(WINDOW))
        .wrote(SlotCount::new(88));
    let map = BlockTable::map_chunk(&hist, &host, tail, part).expect("mappable");
    assert_eq!(
        phys(&map),
        vec![10, 11, 12],
        "88 rows from 512 end in page 2; page 3 is not this request's"
    );
    // A step-back chunk (stepped from 600 to 512) writes the same pages as an aligned one.
    let hist = KvHistory::contiguous(600);
    let back = PagedKvPool::chunk_write_start(KvSlot::new(600), SlotCount::new(WINDOW))
        .wrote(SlotCount::new(WINDOW));
    let map = BlockTable::map_chunk(&hist, &host, back, part).expect("mappable");
    assert_eq!(phys(&map), vec![10, 11, 12, 13]);
}
