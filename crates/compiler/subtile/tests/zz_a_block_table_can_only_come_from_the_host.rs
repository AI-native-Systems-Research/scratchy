// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "spyre")]

//! ⭐ THE PAGES A REQUEST HOLDS COME FROM THE HOST, AND THE CACHEABLE PREFIX STOPS AT THE FIRST HOLE.
//!
//! Two properties, both of which used to be false and each of which cost a session:
//!
//! * A request's page map was filled by a free-list search inside the worker — a SECOND allocator over
//!   the pool the scheduler already owns and refcounts. That is what made a prefix-cache HIT meaningless
//!   (the scheduler said "these blocks hold that prefix"; the worker held it nowhere) and it is now
//!   unwritable, because [`BlockTable`] has no constructor that does not take host ids.
//! * A batched step appends every row at ONE slot, so a request shorter than its batch-mates carries a
//!   masked hole and its keys stop being token-addressed there. Only the leading run may be hashed by a
//!   token-indexed prefix cache.

use scratchy_subtile::sdsc_abstract::{
    BatchSlot, BlockTable, KvHistory, PagedKvPool, PoolPages, PoolPartition, PoolRows, RowPages,
    SlotCount,
};

const PAGE: u32 = PagedKvPool::PAGE_SLOTS as u32;

fn part(pool: usize) -> PoolPartition {
    PoolPartition::of_pool(PoolPages::of_pool(pool).expect("pages"), PoolRows::WIDEST)
        .expect("the pool funds the reserve")
}

// ⛔ THERE IS NO `row0()` HELPER ANY MORE, AND NO `KvRow` TO RETURN. `map_row` took the request's pool row so
// it could draw the batched write page from that row's reserve; the host allocates that page now, so the map
// is a function of the request's HISTORY and the HOST'S LIST alone.

/// ⛔ THERE IS NO CONSTRUCTOR THAT DOES NOT TAKE THE HOST'S IDS. `map_row` is the only way to obtain a
/// `BlockTable`, and it refuses an id the host could not legally have allocated.
#[test]
fn a_block_id_the_host_could_not_have_allocated_is_refused_not_clamped() {
    let p = part(136);
    let hist = KvHistory::contiguous(PAGE as usize);
    let want = RowPages::holding(SlotCount::new(PAGE));
    assert!(
        BlockTable::map_row(&hist, &[7], want, p).is_some(),
        "an id inside the host range is the ordinary case"
    );
    // The host is told it owns `host_blocks()` pages; anything at or above that is the reserve, which is
    // the worker's and which the host must never name.
    let reserved = p.host_blocks() as usize;
    assert!(
        BlockTable::map_row(&hist, &[reserved], want, p).is_none(),
        "id {reserved} is a RESERVED page (scratch/hole): refusing names the disagreement between the \
         count reported to the host and the partition, and a clamp would put a request's keys on the page \
         another row writes its hole into"
    );
    assert!(
        PoolPages::of_pool(0).is_none(),
        "a pool with no page is not a pool"
    );
}

/// ⭐ TWO REQUESTS HOLDING THE SAME PAGE IS WHAT A CACHE HIT IS — and under the old page STRIPE
/// (`physical = row * pages_per_row + lp`) it was not a state the pool could represent, which is the
/// entire reason prefix caching was off.
#[test]
fn two_requests_can_hold_the_same_page_which_is_what_a_cache_hit_is() {
    let p = part(136);
    let hist = KvHistory::contiguous(3 * PAGE as usize);
    let want = RowPages::holding(SlotCount::new(3 * PAGE));
    let a = BlockTable::map_row(&hist, &[3, 4, 5], want, p).expect("mappable");
    let b = BlockTable::map_row(&hist, &[3, 9, 10], want, p).expect("mappable");
    assert_eq!(
        a.pages()[0],
        b.pages()[0],
        "the shared prefix page IS shared"
    );
    assert_ne!(a.pages()[1], b.pages()[1], "and the tails are not");
}

/// Coverage is asked between two PAGE COUNTS, and the slot→page division has exactly one home.
#[test]
fn coverage_is_pages_times_page_slots() {
    let p = part(136);
    let hist = KvHistory::contiguous(2 * PAGE as usize);
    let t = BlockTable::map_row(
        &hist,
        &[0, 1],
        RowPages::holding(SlotCount::new(2 * PAGE)),
        p,
    )
    .expect("mappable");
    assert!(t.covers(RowPages::holding(SlotCount::new(2 * PAGE))));
    assert!(
        !t.covers(RowPages::holding(SlotCount::new(2 * PAGE + 1))),
        "one slot more needs a third page — the worker must REFUSE, not draw one"
    );
    // A request that has written nothing still writes into its first page on this forward.
    assert_eq!(RowPages::holding(SlotCount::new(0)).get(), 1);
    assert_eq!(RowPages::holding(SlotCount::new(PAGE)).get(), 1);
    assert_eq!(RowPages::holding(SlotCount::new(PAGE + 1)).get(), 2);
    // The mask's ceiling is expressed in the same type, so it cannot be compared with a slot count.
    assert!(
        RowPages::holding(SlotCount::new(16 * PAGE))
            .within_mask_reach()
            .is_some()
    );
    assert!(
        RowPages::holding(SlotCount::new(16 * PAGE + 1))
            .within_mask_reach()
            .is_none()
    );
}

/// ⛔⛔ THE MAP IS DERIVED PER BIND AND STORED NOWHERE, so it cannot drift from the host's list. The
/// scheduler ships that list IN FULL every step (`InputBatch::update_blocks` documents it) — a stored,
/// incrementally-appended copy doubled it and pointed every page past the first copy at another request's
/// keys.
#[test]
fn the_map_is_a_function_of_the_hosts_list_so_it_cannot_drift() {
    let p = part(136);
    let hist = KvHistory::contiguous(2 * PAGE as usize);
    let want = RowPages::holding(SlotCount::new(2 * PAGE));
    let first = BlockTable::map_row(&hist, &[0, 1], want, p).expect("mappable");
    // The next step ships the SAME two blocks plus one more — the full table, not a delta.
    let grown = BlockTable::map_row(&hist, &[0, 1, 7], want, p).expect("mappable");
    assert_eq!(
        grown.held().get(),
        2,
        "the map has as many pages as the SLOTS need, not as many as the host has listed — appending \
         would have made it 5"
    );
    assert_eq!(
        first.pages(),
        grown.pages(),
        "and the pages holding keys did not move"
    );
    // A resume: entirely different blocks, and none of the old ones may survive.
    let resumed = BlockTable::map_row(&hist, &[12, 13], want, p).expect("mappable");
    assert_eq!(
        resumed.pages()[0].0,
        12,
        "a resume replaces — same operation, no separate API"
    );
}

#[test]
fn the_cacheable_prefix_is_the_first_run_and_stops_at_the_first_hole() {
    // A 450-token prompt, prefilled contiguously: every token is at its own slot, so all of it is
    // token-addressed and hashable.
    let mut h = KvHistory::contiguous(450);
    assert_eq!(h.cacheable_prefix().get(), 450);

    // Then it shares a batched step with a much longer request. There is no way to NAME a write slot
    // here — `BatchSlot` has only `of` (the max over the live set) and `solo`, which is the lock that
    // stops a caller reaching for a token count — so the batch is built the way the worker builds it.
    let long = KvHistory::contiguous(900);
    let at = BatchSlot::of([&long, &h].into_iter());
    assert_eq!(at.get(), 900, "the batch appends past every live history");
    h.record(at, SlotCount::new(1));
    assert_eq!(h.end().get(), 901, "its keys now reach slot 901");
    assert_eq!(
        h.cacheable_prefix().get(),
        450,
        "but only the first 450 tokens are at their own positions — a block hashed past that would \
         promise a later request the keys of the wrong tokens"
    );
}

#[test]
fn a_prefill_records_one_run_so_a_whole_prompt_stays_cacheable() {
    // The chunked prefill path: every chunk records where it actually wrote, so a prompt that took
    // several chunks is still ONE run, and the whole prompt is cacheable.
    let mut h = KvHistory::contiguous(0);
    let mut at = 0u32;
    while at < 700 {
        let n = 96u32.min(700 - at);
        let start = PagedKvPool::chunk_write_start(
            scratchy_subtile::sdsc_abstract::KvSlot::new(at),
            SlotCount::new(96),
        );
        h.record_chunk(start.wrote(SlotCount::new(n)));
        at = h.end().get();
    }
    assert_eq!(h.runs().len(), 1, "a contiguous prefill is a single run");
    assert_eq!(
        h.cacheable_prefix().get(),
        h.end().get(),
        "so everything it wrote is token-addressed and may be cached"
    );
}

/// ⛔⛔⛔ A HOST LIST THAT STOPS SHORT IS **REFUSED** NOW — IT USED TO BE SERVED FROM A PRIVATE RESERVE.
///
/// 🛑 THIS TEST WAS THE POSITIVE CONTROL FOR `BlockTable::hole_pages_used`, and both the counter and the arm it
/// measured are deleted. Keeping the test's SHAPE and inverting its expectation is deliberate: the same input
/// that used to produce a working map plus a counter tick must now produce `None`, and that is exactly the
/// behavioural change the deletion makes. A test that was only removed would leave nothing saying so.
///
/// Why the control mattered, kept because the lesson outlives the counter: the worker printed `[hole-pages]`
/// only when the counter MOVED, so "the reserve is no longer load-bearing" and "the instrument is dead" were
/// the same silence — and that had already fooled me once, when the counter printed from `Worker::shutdown`
/// which `scr batch` never calls. This test asserted the counter could move BEFORE any zero from it was
/// believed. The zeros it then licensed: 0 firings across 30 gate runs on granite-3.1-2b (hd=64) and
/// granite-3.1-8b (hd=128), and 0/20 on a repeat of the axis that had been firing intermittently.
#[test]
fn a_host_list_that_stops_short_of_the_write_page_is_refused_not_privately_served() {
    let p = part(136);
    // Keys reaching into a third page while the host granted ONE block: the write page has no host block
    // behind it. This was "the batched hole", served from the row's reserved run.
    let hist = KvHistory::contiguous(2 * PAGE as usize + 5);
    let want = RowPages::holding(SlotCount::new(2 * PAGE + 5));
    assert!(
        BlockTable::map_row(&hist, &[3], want, p).is_none(),
        "a short host list must REFUSE: serving it from a private page is correct output bought with a \
         per-request row below the host, which is the thing this branch exists to remove"
    );

    // And the ordinary case still maps: a host list covering every page the row holds keys in.
    let full: Vec<usize> = (0..want.get() as usize).collect();
    assert!(
        BlockTable::map_row(&hist, &full, want, p).is_some(),
        "a fully funded row maps — the refusal above is about the host stopping short, nothing else"
    );
}
