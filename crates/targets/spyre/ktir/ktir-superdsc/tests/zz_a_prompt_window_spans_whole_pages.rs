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
    BlockTable, ChunkKvExtent, KvHistory, KvSlot, NewBlockForm, PagedKvPool, PoolPages,
    PoolPartition, PoolRows, SlotCount,
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

/// ⭐ THE PREFIX LADDER IS THE PAGE AND THE WINDOW: a ladder reaching the window bakes a one-page prefix
/// rung beside the window one, and a ladder that tops out within a page bakes its top alone — so every
/// model whose causal block stays sub-blocked (its ladder stops at 128) bakes exactly what it did.
#[test]
fn the_prefix_ladder_is_the_page_and_the_window() {
    let granite = [7u32, 63, 128, 192, 256, 384, 512];
    assert_eq!(PagedKvPool::prefix_rungs(&granite), vec![PAGE, WINDOW]);
    assert_eq!(PagedKvPool::prefix_rungs(&[7, 63, 128]), vec![128]);
    assert_eq!(PagedKvPool::prefix_rungs(&[7, 128, 256]), vec![PAGE]);
    // A window ladder whose page rung did not lower bakes the window alone — still a valid ladder.
    assert_eq!(PagedKvPool::prefix_rungs(&[7, 128, 384, 512]), vec![WINDOW]);
    assert!(PagedKvPool::prefix_rungs(&[]).is_empty());
}

/// ⛔ THE BAKED PREFIX LADDER MUST TOP OUT AT THE PREFIX-FREE LADDER's TOP, or a mid-prompt continuation
/// chunk (which runs a whole window wide) has no bundle; and a prefix ladder that baked nothing at all
/// refuses every continuation chunk.
#[test]
fn a_prefix_ladder_short_of_the_ladder_top_refuses() {
    let granite = [7u32, 128, 256, 384, 512];
    assert!(PagedKvPool::prefix_ladder_tops_the_ladder(&granite, &[PAGE, WINDOW]).is_ok());
    assert!(PagedKvPool::prefix_ladder_tops_the_ladder(&granite, &[WINDOW]).is_ok());
    let short = PagedKvPool::prefix_ladder_tops_the_ladder(&granite, &[PAGE]).unwrap_err();
    assert!(short.contains("tops out at 256"), "{short}");
    let none = PagedKvPool::prefix_ladder_tops_the_ladder(&granite, &[]).unwrap_err();
    assert!(none.contains("no prefix-capable"), "{none}");
    // No prefix-free ladder ⇒ nothing to top.
    assert!(PagedKvPool::prefix_ladder_tops_the_ladder(&[], &[]).is_ok());
}

/// ⭐ A CONTINUATION CHUNK RUNS ON THE NARROWEST RUNG THAT FINISHES THE PROMPT AS PLACED — and a narrow
/// rung that would need a second chunk (because its placement re-processes overlap) loses to the window.
#[test]
fn a_continuation_chunk_takes_the_narrowest_rung_that_finishes_the_prompt() {
    let widths = [PAGE, WINDOW];
    let pick = |want: u32, left: u32| {
        widths[PagedKvPool::prefix_rung_for(&widths, KvSlot::new(want), left).unwrap()]
    };
    // A page-aligned tail of at most a page: the page rung.
    assert_eq!(pick(1536, 52), PAGE);
    assert_eq!(pick(1536, 256), PAGE);
    // One row more than a page: the window.
    assert_eq!(pick(1536, 257), WINDOW);
    // A mid-prompt chunk (more than a window left): the window, the ladder's top.
    assert_eq!(pick(1024, 9000), WINDOW);
    // A cached prefix ending mid-page: 200 rows at slot 100 fit their page as placed — the page rung.
    assert_eq!(pick(100, 156), PAGE);
    // 240 rows at slot 300 are 284 from slot 256 on the page rung (it steps back 44) — not one chunk; on
    // the window rung (back to 256 as well) they are one. The window.
    assert_eq!(pick(300, 240), WINDOW);
    // A one-rung ladder always answers its rung; an empty one answers nothing.
    assert_eq!(
        PagedKvPool::prefix_rung_for(&[WINDOW], KvSlot::new(1536), 52),
        Some(0)
    );
    assert_eq!(PagedKvPool::prefix_rung_for(&[], KvSlot::new(1536), 52), None);
}

/// ⭐⭐ THE WORKER'S CHUNK LOOP ON THE PREFIX LADDER, transcribed: every continuation chunk chooses its rung
/// ONCE (`prefix_rung_for`), places its padded write at THAT rung's width, takes its rows within that width
/// and the room, and proves its K/V extent against that same width. Over every prompt length up to 1200,
/// long prompts, and prefix-cache entry at every offset of the first page: every mid-prompt chunk is exact
/// on its rung, a tail of at most a page runs on the page rung, and the history is one contiguous run that
/// ends at the prompt.
#[test]
fn a_prompt_runs_on_the_prefix_ladder_with_one_rung_per_chunk() {
    let widths = [PAGE, WINDOW];
    let mut page_tails = 0usize;
    let starts = (0u32..PAGE).step_by(17).chain([0, 1, 255]);
    for cached in starts {
        for total in (1usize..=1200).step_by(7).chain([2048, 4096, 10_300]) {
            let mut hist = KvHistory::contiguous(cached as usize);
            let (mut tok, mut chunk_start) = (cached as usize, cached as usize);
            let tok_end = cached as usize + total;
            let mut first = cached == 0;
            while tok < tok_end {
                // ONE choice, carried to the cap, the padded write and the extent.
                let rung_m = if first {
                    WINDOW
                } else {
                    let i = PagedKvPool::prefix_rung_for(
                        &widths,
                        KvSlot::new(chunk_start as u32),
                        (tok_end - tok) as u32,
                    )
                    .expect("a two-rung ladder always answers");
                    widths[i]
                };
                let padded = SlotCount::new(if first { 0 } else { rung_m });
                let placed =
                    PagedKvPool::chunk_write_start(KvSlot::new(chunk_start as u32), padded);
                let back = chunk_start - placed.slot().get() as usize;
                chunk_start -= back;
                tok -= back;
                let room = PagedKvPool::chunk_room(placed.slot()).get() as usize;
                let chunk_len = (tok_end - tok).min(rung_m as usize).min(room);
                let is_final = tok + chunk_len == tok_end;
                if !is_final {
                    assert!(
                        ChunkKvExtent::exact(chunk_len as u32, rung_m).is_some(),
                        "cached={cached} total={total}: a mid-prompt chunk of {chunk_len} rows on the \
                         {rung_m}-row rung writes rows the history never counts"
                    );
                } else if !first && rung_m == PAGE {
                    page_tails += 1;
                }
                if !first {
                    assert!(
                        chunk_len as u32 <= rung_m,
                        "cached={cached} total={total}: {chunk_len} rows on a {rung_m}-row rung"
                    );
                }
                hist.record_chunk(placed.wrote(SlotCount::new(chunk_len as u32)));
                chunk_start += chunk_len;
                tok += chunk_len;
                first = false;
            }
            assert_eq!(
                hist.end().get() as usize,
                tok_end,
                "cached={cached} total={total}"
            );
            assert_eq!(
                hist.runs().len(),
                1,
                "cached={cached} total={total}: one contiguous run"
            );
        }
    }
    assert!(
        page_tails > 0,
        "no prompt's last continuation chunk took the page rung — the test does not reach the case it is for"
    );
}

/// ⭐ THE ~2k-TOKEN PROMPT THAT MEASURED THE TAIL COST, by its own chunk sequence: 512 prefix-free, three
/// 512-row continuations, then a 52-row tail — which now runs on the PAGE rung, the one this ladder adds.
#[test]
fn the_measured_two_k_prompt_ends_on_the_page_rung() {
    let widths = [PAGE, WINDOW];
    let total = 2100u32;
    let mut at = WINDOW;
    let mut rungs = Vec::new();
    while at < total {
        let i = PagedKvPool::prefix_rung_for(&widths, KvSlot::new(at), total - at).unwrap();
        rungs.push(widths[i]);
        at += (total - at).min(widths[i]);
    }
    assert_eq!(rungs, vec![WINDOW, WINDOW, WINDOW, PAGE]);
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

/// ⛔⛔⛔ A CAPPED LADDER REFUSES. A chunk-form model (granite-2b's head dim) whose baked ladder stops
/// at 256 fails the bake, naming why: a ceiling that never asked for the window (the resident-cache cap
/// that hid the 512 rungs), or the rungs that did not lower and their reasons. The ladder that reaches
/// the window passes, and so does a sub-blocked model's own shorter ladder.
#[test]
fn a_chunk_form_ladder_baked_short_of_the_window_refuses() {
    let chunk_form = NewBlockForm::of(WINDOW, 64, false) == NewBlockForm::Chunk;
    assert!(
        chunk_form,
        "granite-2b's causal block is chunk-formed at the window"
    );
    let to_256 = [7u32, 63, 128, 192, 256];

    let capped = PagedKvPool::prefill_ladder_reaches_the_window(true, 256, &to_256, &[])
        .expect_err("a ladder capped at 256 must refuse");
    assert!(
        capped.contains("256") && capped.contains("ceiling was 256"),
        "names the cap: {capped}"
    );

    let refused = PagedKvPool::prefill_ladder_reaches_the_window(
        true,
        WINDOW,
        &to_256,
        &[(384, "LX overflow".into()), (512, "LX overflow".into())],
    )
    .expect_err("a ladder whose top rungs did not lower must refuse");
    assert!(
        refused.contains("rung m=384 did not lower: LX overflow")
            && refused.contains("rung m=512 did not lower: LX overflow"),
        "names each refusing rung and its reason: {refused}"
    );

    let full = [7u32, 128, 256, 384, 512];
    assert_eq!(
        PagedKvPool::prefill_ladder_reaches_the_window(true, WINDOW, &full, &[]),
        Ok(())
    );
    // A model whose causal block stays sub-blocked bakes its own 128-row ladder by design.
    assert_eq!(
        PagedKvPool::prefill_ladder_reaches_the_window(false, 128, &[7, 63, 128], &[]),
        Ok(())
    );
}
