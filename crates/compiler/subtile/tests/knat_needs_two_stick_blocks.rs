// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "spyre")]
//! ⭐⭐⭐⭐⭐ HOW MANY STICK-BLOCKS OF NATURAL K A PREFILL CHUNK CAN TOUCH — the number the Knat plane has
//! to be, and the reason it does not have to be a whole page.
//!
//! The Kᵀ re-transpose reads natural K and writes Kᵀ through ONE segment shift, and that shift moves the
//! op by whole 64-slot stick blocks (`bundle::SlabShift`). So a prefill chunk's re-transpose can be a
//! BAKED number of block-sized ops — one per block its padded window can touch — only if that number is
//! the same for every chunk start the host can produce. This file measures it, from the real
//! [`PagedKvPool::chunk_room`] / [`PagedKvPool::chunk_write_start`] laws rather than from a hand-derived
//! start sequence.
//!
//! ⛔ AND IT IS NOT OBVIOUS, WHICH IS WHY IT IS MEASURED. `PREFILL_CHUNK_SLOTS` (96) does NOT divide
//! `PAGE_SLOTS` (256), so starts are not block-aligned: they land on page offsets 0, 96 and then a
//! STEPPED-BACK 160. A 96-slot window at offset 32-within-a-block spans three blocks if the offset were
//! larger than 32 — `32 + 96 = 128` is exactly two blocks, and one slot more would be three. The whole
//! plan rests on that inequality holding for every reachable start, so it is asserted rather than
//! assumed.

use scratchy_subtile::sdsc_abstract::{KvSlot, PagedKvPool, SlotCount};

const STK: u32 = scratchy_subtile::sdsc_abstract::POOL_STICK;

/// Every chunk start the host can actually place inside one page, by walking the real laws: clip the
/// chunk to the room its page has left, step the start back if the PADDED window would leave the page.
fn reachable_starts_in_a_page() -> Vec<u32> {
    let page = PagedKvPool::PAGE_SLOTS as u32;
    let padded = SlotCount::new(PagedKvPool::PREFILL_CHUNK_SLOTS as u32);
    let mut starts = Vec::new();
    let mut want = 0u32;
    while want < page {
        let placed = PagedKvPool::chunk_write_start(KvSlot::new(want), padded).slot().get();
        starts.push(placed);
        // The chunk advances by the room its page has left, clipped to a whole chunk — the host's own law.
        let room = PagedKvPool::chunk_room(KvSlot::new(want)).get();
        let advance = room.min(PagedKvPool::PREFILL_CHUNK_SLOTS as u32);
        if advance == 0 {
            break;
        }
        want += advance;
    }
    starts
}

/// ⭐ THE MEASUREMENT: a padded window never spans more than TWO stick blocks.
#[test]
fn a_prefill_chunks_padded_window_spans_at_most_two_stick_blocks() {
    let padded = PagedKvPool::PREFILL_CHUNK_SLOTS as u32;
    let starts = reachable_starts_in_a_page();
    assert!(
        !starts.is_empty(),
        "the walk produced no chunk starts — the laws changed shape"
    );
    for &start in &starts {
        let first = start / STK;
        let last = (start + padded - 1) / STK;
        let blocks = last - first + 1;
        assert!(
            blocks <= 2,
            "a chunk at page offset {start} spans {blocks} stick blocks ({first}..={last}) — the \
             re-transpose's baked op count is 2, so a third block would leave that block's Kᵀ stale \
             while its slots read as valid: fluent wrong output, not a fault"
        );
    }
}

/// ⛔⛔⛔ EVERY BLOCK THE EMITTER BAKES STAYS INSIDE THE PAGE — asserted against
/// [`PagedKvPool::PREFILL_CHUNK_BLOCKS`] ITSELF, not against a hand-written 2.
///
/// This is the assertion that would have caught a real bug and did not. `PREFILL_CHUNK_BLOCKS` was briefly
/// `(SLOTS + STK - 1).div_ceil(STK)` — the ceiling applied twice — which is 3, and at the stepped-back
/// start 160 the third block's index is `2 + 2 = 4` in a page of four blocks numbered 0..3: the op writes
/// onto the NEXT KV HEAD's keys, which is the `WRITE_SLACK` corruption (correct first token, then fluent
/// garbage past a chunk boundary). It reached the card; the short-chunk prompts did not expose it and only
/// the descriptor dump did (`attn_kctpost0b2_o734`).
///
/// The previous version of this test checked `first + 1`, i.e. it hard-coded the two it was supposed to be
/// guarding. Reading the count from the constant the emitter loops over is what makes it a guard.
#[test]
fn every_baked_block_stays_inside_the_page() {
    let blocks_per_page = PagedKvPool::PAGE_SLOTS as u32 / STK;
    let baked = PagedKvPool::PREFILL_CHUNK_BLOCKS as u32;
    for start in reachable_starts_in_a_page() {
        let first = start / STK;
        assert!(
            first + baked <= blocks_per_page,
            "a chunk at page offset {start} has the re-transpose bake blocks {first}..{} of a page with \
             only {blocks_per_page} — the overrun lands on the next kv head's keys",
            first + baked - 1
        );
    }
}

/// And the baked count is exactly the span the windows need — neither short (a stale block) nor long (the
/// overrun above).
#[test]
fn the_baked_block_count_matches_the_measured_span() {
    let padded = PagedKvPool::PREFILL_CHUNK_SLOTS as u32;
    let worst = reachable_starts_in_a_page()
        .into_iter()
        .map(|s| {
            let first = s / STK;
            (s + padded - 1) / STK - first + 1
        })
        .max()
        .expect("at least one start");
    assert_eq!(
        PagedKvPool::PREFILL_CHUNK_BLOCKS as u32, worst,
        "the emitter bakes {} block op(s) per kv head but the widest reachable window spans {worst}",
        PagedKvPool::PREFILL_CHUNK_BLOCKS
    );
}

/// The bound as the INEQUALITY it actually is, so a future ceiling change fails here with the reason
/// rather than somewhere downstream: the worst in-block offset plus a padded window must fit two blocks.
#[test]
fn the_bound_is_the_worst_in_block_offset_plus_the_window() {
    let padded = PagedKvPool::PREFILL_CHUNK_SLOTS as u32;
    let worst = reachable_starts_in_a_page()
        .into_iter()
        .map(|s| s % STK)
        .max()
        .expect("at least one start");
    assert!(
        worst + padded <= 2 * STK,
        "the worst chunk start sits {worst} slots into its stick block and the padded window is \
         {padded}, which needs {} slots — more than the two blocks ({}) the re-transpose bakes. Either \
         the chunk ceiling or the block count has to change together.",
        worst + padded,
        2 * STK
    );
}
