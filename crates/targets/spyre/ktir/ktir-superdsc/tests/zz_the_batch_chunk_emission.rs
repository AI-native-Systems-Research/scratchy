// SPDX-License-Identifier: Apache-2.0
//! The split-KV chunk over a decode BATCH (`BatchChunk`): the ladder the host and the bake share,
//! and the host law its mask and index are staged by.
//!
//! (The emitter-side half of this file — the batch chunk's op counts, mask epilogue shapes and
//! placement refusals, minted through the deleted hand-written attention emitter — went with it;
//! when the live spliced kernels need the same pins, mint them from the splice, not from a copy of
//! the old emitter.)

use ktir_superdsc::sdsc_abstract::{
    BatchChunk, ChunkScratch, KvHistory, PagedKvPool, StickLayout, batch_chunk_empty_sweep_row,
    batch_chunk_mask_f16,
};

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

#[test]
fn the_batch_mask_validates_each_row_on_its_own_requests_history_alone() {
    let (seqs, j) = (4u32, 3u32);
    let chunk = ChunkScratch::of_chunk(pool_2b(), seqs * j).expect("admitted");
    let bc = BatchChunk::of(chunk, seqs, 4).expect("a whole number of pages per request");
    assert_eq!(bc.mask_rows(), 4 * seqs);
    assert_eq!(bc.mask_cols(), seqs * j * 256);
    let layout = StickLayout::row_blocked(bc.mask_rows() as usize, bc.mask_cols() as usize);
    for row in [0, 1, 5, 15] {
        for slot in [0, 63, 64, 700, bc.mask_cols() - 1] {
            assert_eq!(
                bc.mask_elem(row, slot),
                Some(layout.dev_off(row as usize, slot as usize)),
                "({row}, {slot})"
            );
        }
    }
    assert_eq!(bc.mask_elem(bc.mask_rows(), 0), None);
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
                ktir_superdsc::sdsc_abstract::BatchSlot::of(deep.iter()),
                ktir_superdsc::sdsc_abstract::SlotCount::new(1),
            );
            assert!(h.contains(300) && !h.contains(5) && !h.contains(299));
            h
        },
        KvHistory::contiguous(10),
    ];
    let mask = batch_chunk_mask_f16(bc, &hists, -100.0).expect("one history per request row");
    let m = mask.wire_bytes();
    assert_eq!(m.len(), bc.mask_elems() * 2);
    let valid = |row: u32, slot: u32| {
        let e = bc.mask_elem(row, slot).expect("in the block");
        half::f16::from_le_bytes([m[e * 2], m[e * 2 + 1]]).to_f32() == 0.0
    };
    let per = j * 256;
    // ⭐ THE TALLIES ARE THE BYTES' OWN VALIDITY — derived independently of the minter's runs
    // walk, at the minter's own (row, 64-lane slab) pitch; every count must agree.
    let slabs = bc.mask_cols() as usize / 64;
    let mut tallies = vec![0u32; bc.mask_rows() as usize * slabs];
    for row in 0..bc.mask_rows() {
        let r = row % seqs;
        for slot in 0..bc.mask_cols() {
            let (owner, local) = bc.slot_of(slot).expect("inside the chunk");
            let want = owner == r && hists[r as usize].contains(local as usize);
            assert_eq!(
                valid(row, slot),
                want,
                "row {row} (request {r}) at slot {slot} (request {owner}, local {local})"
            );
            assert!(local < per);
            if want {
                tallies[row as usize * slabs + slot as usize / 64] += 1;
            }
        }
    }
    for (flat, &t) in tallies.iter().enumerate() {
        assert_eq!(
            mask.window_valid(flat / slabs, flat % slabs).get(),
            t,
            "flat window-row {flat}: the count must be the bytes' own validity, minted by the \
             same runs walk that wrote them"
        );
    }
    assert!(batch_chunk_mask_f16(bc, &hists[..3], -100.0).is_none());
}

/// ⭐⭐⭐ THE SWEEP READING at the chunk family's own door: a row whose whole slab sweep is empty
/// is a request that holds none of its own run's slots — the never-prefilled desync the fold's
/// `l` would normalize, because the row's `m` never rises past the sentinel seed to expunge it.
/// A healthy batch sweeps one on every row; the named row maps to its request by the chunk's
/// own law door.
#[test]
fn the_chunk_family_names_an_empty_sweep_row() {
    let (seqs, j) = (4u32, 3u32);
    let chunk = ChunkScratch::of_chunk(pool_2b(), seqs * j).expect("admitted");
    let bc = BatchChunk::of(chunk, seqs, 4).expect("a whole number of pages per request");
    let healthy = vec![
        KvHistory::contiguous(10),
        KvHistory::contiguous(256),
        KvHistory::contiguous(5),
        KvHistory::contiguous(10),
    ];
    let m = batch_chunk_mask_f16(bc, &healthy, -100.0).expect("one history per request row");
    assert_eq!(
        batch_chunk_empty_sweep_row(&m),
        None,
        "a batch whose every request holds a slot must sweep one on every row"
    );
    // Request 1's history is EMPTY — no run, no shared slot — so every row `i % B == 1`
    // sweeps nothing, and the first of them is named.
    let starved = vec![
        KvHistory::contiguous(10),
        KvHistory::contiguous(0),
        KvHistory::contiguous(5),
        KvHistory::contiguous(10),
    ];
    let m = batch_chunk_mask_f16(bc, &starved, -100.0).expect("one history per request row");
    let row = batch_chunk_empty_sweep_row(&m).expect("request 1's rows sweep nothing");
    assert_eq!(
        bc.row_request(row),
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
