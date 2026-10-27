// SPDX-License-Identifier: Apache-2.0
//! The split-KV chunk over a decode BATCH (`BatchChunk`): the ladder the host and the bake share,
//! and the host law its mask and index are staged by.
//!
//! (The emitter-side half of this file — the batch chunk's op counts, mask epilogue shapes and
//! placement refusals, minted through the deleted hand-written attention emitter — went with it;
//! when the live spliced kernels need the same pins, mint them from the splice, not from a copy of
//! the old emitter.)

use ktir_superdsc::sdsc_abstract::{BatchChunk, ChunkScratch, KvHistory, PagedKvPool};

fn pool_2b() -> PagedKvPool {
    PagedKvPool::new(8, 64)
}

/// ⭐ THE BATCH LADDER IS `B × J` PAGES AND ITS SWEPT KEY IS `J` PAGES — and the host's inverse of that
/// key is the chunk the bake attended, at every rung of every batch width.
#[test]
fn the_batch_ladder_round_trips_through_its_swept_key() {
    let pool = pool_2b();
    assert_eq!(
        ChunkScratch::ladder_for(pool, 1),
        ChunkScratch::LADDER.to_vec(),
        "one request keeps the solo ladder"
    );
    for seqs in PagedKvPool::BATCH_RUNGS {
        let ladder = ChunkScratch::ladder_for(pool, seqs);
        assert!(
            ladder
                .iter()
                .all(|&t| t.is_multiple_of(seqs) && t / seqs >= 2 && t <= ChunkScratch::PAGES_MAX),
            "seqs={seqs}: {ladder:?}"
        );
        assert!(
            ladder.windows(2).all(|w| w[0] < w[1]),
            "seqs={seqs}: ascending"
        );
        for &t in &ladder {
            let chunk = ChunkScratch::of_chunk(pool, t).expect("admitted");
            let swept = chunk.swept_per_request(seqs);
            assert_eq!(swept, (t / seqs) * 256, "seqs={seqs} t={t}");
            assert_eq!(
                ChunkScratch::of_batch(pool, seqs, swept),
                Some(chunk),
                "seqs={seqs} t={t}: the host's chunk is the bake's"
            );
        }
    }
    // The canonical batched cell — conc-8 at ~1.05k-token prompts, 256 out — crosses 5 → 6 pages per
    // request; both are exact rungs, so no request pads a page.
    assert!(ChunkScratch::ladder_for(pool, 8).contains(&40));
    assert!(ChunkScratch::ladder_for(pool, 8).contains(&48));
}

/// ⭐⭐⭐ EACH ROW VALIDATES ON ITS OWN REQUEST'S HISTORY ALONE — the live batched mask
/// (`spliced_batched_mask_f16`, the bytes `batched_decode_hm` and the batched fold add at the score
/// tile). Row `(p·W4 + w)·PG + (b·GQA + g)` belongs to request `b / NKVH`; lane `cc` is valid exactly
/// where that request's own history holds slot `p·PAGE_SLOTS + w·64 + cc`, or the batch's one shared
/// write slot. A row reading a batch-mate's keys is the cross-request leak this pins shut.
#[test]
fn the_batch_mask_validates_each_row_on_its_own_requests_history_alone() {
    use ktir_superdsc::sdsc_abstract::{BatchSlot, SplicedBatchRegion, spliced_batched_mask_f16};
    use std::num::NonZeroU32;
    let pool = pool_2b();
    let (seqs, gqa) = (4u32, 4u32);
    let ktb = seqs * pool.nkvh();
    let pg = ktb * gqa;
    let w4 = pool.windows_per_page();
    let pages = 2u32;
    let block = u64::from(w4) * u64::from(pg) * 64 * 2;
    let region = SplicedBatchRegion::of_mask_extent(
        block * u64::from(pages),
        pool,
        NonZeroU32::new(gqa).expect("gqa is nonzero"),
        NonZeroU32::new(seqs).expect("a batch is nonzero"),
    )
    .expect("two whole page blocks is a region this family states");
    // Request 0: 10 keys. Request 1: a whole page. Request 2: 5 keys, a hole, then one key a batch
    // appended at its shared write slot (300, past a 300-key batch-mate). Request 3: the padding row,
    // live 0's history.
    let hists = vec![
        KvHistory::contiguous(10),
        KvHistory::contiguous(256),
        {
            let mut h = KvHistory::contiguous(5);
            let deep = [KvHistory::contiguous(300)];
            h.record(
                BatchSlot::of(deep.iter()),
                ktir_superdsc::sdsc_abstract::SlotCount::new(1),
            );
            assert!(h.contains(300) && !h.contains(5) && !h.contains(299));
            h
        },
        KvHistory::contiguous(10),
    ];
    let write = BatchSlot::of(hists.iter());
    let mask = spliced_batched_mask_f16(region, &hists, write).expect("one history per launch row");
    let rows = (pages * w4 * pg) as usize;
    assert_eq!(mask.rows(), rows);
    let m = mask.wire_bytes();
    assert_eq!(m.len(), rows * 64 * 2);
    for r in 0..rows {
        let p = r / (w4 * pg) as usize;
        let rem = r % (w4 * pg) as usize;
        let (w, flat) = (rem / pg as usize, rem % pg as usize);
        let request = flat / gqa as usize / pool.nkvh() as usize;
        let mut lanes = 0u32;
        for cc in 0..64usize {
            let slot = p * PagedKvPool::PAGE_SLOTS + w * 64 + cc;
            let e = (r * 64 + cc) * 2;
            let valid = half::f16::from_le_bytes([m[e], m[e + 1]]).to_f32() == 0.0;
            let want = hists[request].contains(slot) || slot == write.get() as usize;
            assert_eq!(
                valid, want,
                "row {r} (request {request}) at slot {slot}: validity is its OWN history's"
            );
            lanes += u32::from(want);
        }
        assert_eq!(
            mask.window_valid(r, 0).get(),
            lanes,
            "row {r}: the count must be the bytes' own validity, minted by the loop that wrote them"
        );
    }
    assert!(
        spliced_batched_mask_f16(region, &hists[..3], write).is_none(),
        "a history count that is not the launch's row count refuses"
    );
}

/// ⭐⭐⭐ THE SWEEP READING over the live batched mask: a row whose whole sweep is empty is a request
/// that holds none of its own slots — the never-prefilled desync the fold's `l` would normalize,
/// because the row's `m` never rises past the sentinel seed to expunge it. A healthy batch sweeps one
/// on every row; a starved request's rows are named.
#[test]
fn the_batch_mask_names_an_empty_sweep_row() {
    use ktir_superdsc::sdsc_abstract::{BatchSlot, SplicedBatchRegion, spliced_batched_mask_f16};
    use std::num::NonZeroU32;
    let pool = pool_2b();
    let (seqs, gqa) = (4u32, 4u32);
    let pg = seqs * pool.nkvh() * gqa;
    let w4 = pool.windows_per_page();
    let block = u64::from(w4) * u64::from(pg) * 64 * 2;
    let region = SplicedBatchRegion::of_mask_extent(
        block,
        pool,
        NonZeroU32::new(gqa).expect("gqa is nonzero"),
        NonZeroU32::new(seqs).expect("a batch is nonzero"),
    )
    .expect("one whole page block is a region this family states");
    let rows = (w4 * pg) as usize;
    // The empty-sweep reading: a row whose every window counts zero valid lanes.
    let empty_row = |mask: &ktir_superdsc::sdsc_abstract::ValidityMask| {
        (0..pg as usize).find(|&row| {
            (0..w4 as usize).all(|w| mask.window_valid(w * pg as usize + row, 0).get() == 0)
        })
    };
    let healthy = vec![
        KvHistory::contiguous(10),
        KvHistory::contiguous(256),
        KvHistory::contiguous(5),
        KvHistory::contiguous(10),
    ];
    // A write slot past the page — so the sweep law reads the histories alone.
    let past = [KvHistory::contiguous(PagedKvPool::PAGE_SLOTS)];
    let write = BatchSlot::of(past.iter());
    let m = spliced_batched_mask_f16(region, &healthy, write).expect("one history per row");
    assert_eq!(m.rows(), rows);
    assert_eq!(
        empty_row(&m),
        None,
        "a batch whose every request holds a slot must sweep one on every row"
    );
    // Request 1's history is EMPTY, so every one of request 1's flat rows sweeps nothing, and the
    // first of them is named.
    let starved = vec![
        KvHistory::contiguous(10),
        KvHistory::contiguous(0),
        KvHistory::contiguous(5),
        KvHistory::contiguous(10),
    ];
    let m = spliced_batched_mask_f16(region, &starved, write).expect("one history per row");
    let row = empty_row(&m).expect("request 1's rows sweep nothing");
    assert_eq!(
        row / gqa as usize / pool.nkvh() as usize,
        1,
        "the named row {row} must be request 1's"
    );
}

/// ⭐ THE PAGE LIST LAYS EACH REQUEST's OWN MAP AT `J` PAGES, padding a short map with ITS OWN last page
/// — never another request's — and refuses a map past `J` (a body the selector should not have chosen).
#[test]
fn the_batch_page_list_pads_each_request_with_its_own_last_page() {
    let chunk = ChunkScratch::of_chunk(pool_2b(), 6).expect("admitted");
    let bc = BatchChunk::of(chunk, 2, 4).expect("three pages per request");
    assert_eq!(bc.pages_per_request(), 3);
    assert_eq!(
        bc.page_list(&[vec![7, 9], vec![4, 5, 6]]),
        Some(vec![7, 9, 9, 4, 5, 6])
    );
    assert_eq!(bc.page_list(&[vec![7, 9, 1, 2], vec![4]]), None);
    assert_eq!(bc.page_list(&[vec![7]]), None);
    assert_eq!(bc.page_list(&[vec![], vec![4]]), None);
    assert!(
        BatchChunk::of(chunk, 4, 4).is_none(),
        "6 pages is no whole number per request at 4"
    );
}
