// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "spyre")]

//! ⭐⭐⭐ REPLAY OF THE PARTIAL-CACHE-HIT SEQUENCE, THROUGH THE REAL PAGE MAP.
//!
//! MEASURED on granite-3.1-8b fp8, 2026-08-13, ragged 6-probe batch with prefix caching ON:
//! three requests took PARTIAL hits (`1/3`, `3/5`, `5/7` blocks) and four answers差 from the
//! caching-OFF control — while a FULL `8/8` hit (four requests sharing a 2048-token prefix) was correct
//! and byte-identical to its control. So the defect, if it is in the host's page map, is in the shape a
//! partial hit produces: some pages the host allocated, then the request's OWN continuation prefill, then
//! a batched decode whose write slot is set by a much deeper request.
//!
//! This walks that exact sequence and asserts the ONE property the map must have: **every logical page
//! that holds keys resolves to a distinct host block, in ascending order, and no page that holds keys
//! ever resolves to the shared scratch page.** A page aliased to scratch is read fully-masked and never
//! written, so putting a page with real keys there loses them silently.

use scratchy_subtile::sdsc_abstract::{
    BatchSlot, BlockTable, ChunkStart, KvHistory, KvSlot, PagedKvPool, PoolPages, PoolPartition,
    PoolRows, RowPages, SlotCount,
};

const PAGE: u32 = PagedKvPool::PAGE_SLOTS as u32;

fn partition() -> PoolPartition {
    // The measured pool: 136 pages, 32 launch rows → 71 host blocks, page 71 the scratch, 72.. the holes.
    PoolPartition::of_pool(PoolPages::of_pool(136).unwrap(), PoolRows::WIDEST)
        .expect("136 pages fund it")
}

/// Every page holding keys must map to a distinct host block; nothing holding keys may land on scratch.
fn check(map: &BlockTable, hist: &KvHistory, want: RowPages, part: PoolPartition, label: &str) {
    let mut seen_host = Vec::new();
    for page in want.pages() {
        let phys = map.pages()[page.get() as usize];
        if hist.touches(page) {
            assert_ne!(
                phys,
                part.scratch(),
                "{label}: logical page {} HOLDS KEYS and was aliased to the shared scratch page — those \
                 keys are read fully-masked and never written back",
                page.get()
            );
            assert!(
                !seen_host.contains(&phys),
                "{label}: logical page {} shares physical page {:?} with an earlier page that also holds \
                 keys — one of them is reading the other's slots",
                page.get(),
                phys
            );
            seen_host.push(phys);
        }
    }
}

#[test]
fn a_partial_hit_then_a_continuation_then_a_deep_batched_decode() {
    let part = partition();

    // ── 1. ADMITTED WITH A 3-BLOCK HIT. 884-token prompt, 768 tokens already resident in another
    //       request's blocks: the host's list starts with those three, then this request's own.
    let host: Vec<usize> = vec![10, 11, 12, 30, 31, 32, 33, 34];
    let mut hist = KvHistory::contiguous(768);
    let want = RowPages::holding(SlotCount::new(769));
    let map = BlockTable::map_row(&hist, &host, want, part).expect("mappable");
    assert_eq!(
        map.pages().iter().map(|p| p.0).collect::<Vec<_>>(),
        vec![10, 11, 12, 30],
        "the three cached blocks at pages 0-2, and the page the continuation will write at page 3"
    );
    check(&map, &hist, want, part, "at the first continuation bind");

    // ── 2. THE CONTINUATION PREFILL, chunk by chunk through the real placement function.
    while hist.end().get() < 884 {
        let at = BatchSlot::solo(&hist);
        let want = RowPages::holding(SlotCount::new(at.get() + 1));
        let map = BlockTable::map_row(&hist, &host, want, part).expect("mappable");
        check(&map, &hist, want, part, "mid-continuation");
        // The page this chunk writes into must be a host block, never scratch.
        let write_page = want.write_page().expect("non-empty");
        assert_ne!(
            map.pages()[write_page.get() as usize],
            part.scratch(),
            "the chunk about to write logical page {} was pointed at the scratch page",
            write_page.get()
        );
        let n = 96u32.min(884 - hist.end().get());
        let start =
            PagedKvPool::chunk_write_start(KvSlot::new(hist.end().get()), SlotCount::new(96));
        hist.record_chunk(start.wrote(SlotCount::new(n)));
    }
    assert_eq!(hist.end().get(), 884, "the prompt is resident");
    assert_eq!(
        hist.cacheable_prefix().get(),
        884,
        "and all of it is contiguous, so all of it may be hashed"
    );

    // ── 3. A BATCHED DECODE WHOSE WRITE SLOT COMES FROM A 1840-SLOT REQUEST.
    let deep = KvHistory::contiguous(1840);
    let at = BatchSlot::of([&deep, &hist].into_iter());
    assert_eq!(at.get(), 1840);
    let want = RowPages::holding(SlotCount::new(at.get() + 1));
    assert_eq!(want.get(), 8, "eight logical pages to reach slot 1840");
    let map = BlockTable::map_row(&hist, &host, want, part).expect("mappable");
    check(&map, &hist, want, part, "at the batched decode bind");
    let phys: Vec<u32> = map.pages().iter().map(|p| p.0).collect();
    assert_eq!(
        &phys[..4],
        &[10, 11, 12, 30],
        "the pages holding keys keep their host blocks"
    );
    assert_eq!(
        &phys[4..7],
        &[part.scratch().0; 3],
        "the fully-masked hole pages alias the one scratch page"
    );
    assert_ne!(
        phys[7],
        part.scratch().0,
        "but the page the write slot lands in must have storage of its own"
    );

    // ── 4. AND THE KEY IT WRITES THERE IS REAL — so the NEXT step must still resolve page 7 to the same
    //       physical page, or that key is lost.
    hist.record(at, SlotCount::new(1));
    let at2 = BatchSlot::of([&deep, &hist].into_iter());
    let want2 = RowPages::holding(SlotCount::new(at2.get() + 1));
    let map2 = BlockTable::map_row(&hist, &host, want2, part).expect("mappable");
    assert_eq!(
        map2.pages()[7],
        map.pages()[7],
        "page 7 moved between steps — the key written last step is now at a page nobody reads"
    );
    check(&map2, &hist, want2, part, "after the first batched write");
}

/// ⛔ THE DEEPEST ROW HAS NO HOLE AT ALL: every page holds keys, so every page must be a host block.
#[test]
fn the_deepest_row_maps_entirely_to_its_own_host_blocks() {
    let part = partition();
    let hist = KvHistory::contiguous(1840);
    let host: Vec<usize> = (10..18).collect();
    let at = BatchSlot::solo(&hist);
    let want = RowPages::holding(SlotCount::new(at.get() + 1));
    let map = BlockTable::map_row(&hist, &host, want, part).expect("mappable");
    assert_eq!(
        map.pages().iter().map(|p| p.0).collect::<Vec<_>>(),
        (10..18).collect::<Vec<_>>(),
        "the writer of a shared prefix must see exactly its own blocks, in order, with no scratch and no \
         hole page — anything else means the request that OWNS the keys stopped reading them"
    );
    check(&map, &hist, want, part, "deepest row");
}

/// ⛔⛔ AND A PREFILL BIND MUST NEVER ALIAS THE PAGE IT IS ABOUT TO WRITE. A cold prompt's history is
/// EMPTY at its first bind, so "which pages hold keys" answers `none` — the write page is the only thing
/// standing between that chunk and the scratch page.
#[test]
fn a_cold_prompts_first_chunk_writes_a_real_page() {
    let part = partition();
    let hist = KvHistory::contiguous(0);
    let host: Vec<usize> = vec![40, 41];
    let want = RowPages::holding(SlotCount::new(1));
    let map = BlockTable::map_row(&hist, &host, want, part).expect("mappable");
    assert_eq!(want.get(), 1, "one page holds slot 0");
    assert_eq!(
        map.pages()[0].0,
        40,
        "the first chunk of a cold prompt must write the host's first block, not the scratch page"
    );
    let _ = PAGE;
}
