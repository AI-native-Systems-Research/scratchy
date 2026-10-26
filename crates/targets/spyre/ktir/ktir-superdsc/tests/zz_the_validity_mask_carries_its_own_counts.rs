// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ A VALIDITY MASK CARRIES ITS OWN VALID-LANE COUNTS — the fold's `l`-invariant, as a
//! type. The online-softmax denominator is a STRICT running sum over exactly the lanes each row
//! marks valid, so the count is not a diagnostic: it is the number the type exists to hold, minted
//! by the same loop that writes the row's bytes (never re-derived from a staged buffer).
//!
//! THE SEED-SAFETY LAW, pinned first: the carry seeds `m` at the mask's own −65504 sentinel, so an
//! ALL-masked FIRST window would leave every lane at `exp2(0) = 1` — `l += 64`, `acc += ΣV`, the
//! mask doing nothing. `real: NonZeroU32` makes the law a THEOREM (`pos = start + min(t, real−1)
//! ≥ start ≥ 0`, so window (0, 0) holds ≥ 1 valid lane for every query row) — this test pins the
//! theorem's conclusion across the chunk family's whole (start, real) range, at granite-3.1-2b's
//! geometry (8 kv heads, hd 64).
//!
//! The batch family's rows are PRE-REPLICATED xGQA, so its counts are checked as facts (equal
//! within each group, equal to the row's own zero-lane count) rather than as a theorem — its
//! coverage is a placement fact, and a row past the history's end with the shared write slot
//! inside it is VALID-INVALID-VALID, not a prefix.

use std::num::NonZeroU32;

use ktir_superdsc::sdsc_abstract::{
    BatchSlot, ChunkRows, CopyDims, KvHistory, PagedKvPool, RungWidth, SplicedAttnRegion,
    SplicedDecodeBatch, ValidKeys, spliced_attn_mask_f16, spliced_decode_mask_f16,
};

/// granite-3.1-2b's pool: 8 kv heads at head dim 64.
fn pool() -> PagedKvPool {
    PagedKvPool::new(8, 64)
}

/// A region of `pages` page blocks at a rung of `mq` rows — the extent the bake's mask placement
/// states for it (`W4 · mq` rows of 64 f16 lanes per page block, the padded pitch law).
fn region(pages: u32, mq: u32) -> SplicedAttnRegion {
    let mq = RungWidth::of_baked_rows(mq).expect("a test rung is nonzero");
    let w4 = pool()
        .windows_per_page()
        .expect("hd 64 divides a page into whole windows");
    let block = u64::from(w4) * mq.count() as u64 * 64 * 2;
    SplicedAttnRegion::of_mask_extent(
        block * pages as u64,
        pool(),
        NonZeroU32::new(4).expect("4 is nonzero"),
        mq,
    )
    .expect("a whole number of page blocks is a region")
}

/// The f16 lanes of one mask row, LE.
fn row_lanes(bytes: &[u8], r: usize) -> Vec<half::f16> {
    bytes[r * 128..(r + 1) * 128]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| half::f16::from_le_bytes(*c))
        .collect()
}

#[test]
fn a_count_past_the_lanes_of_a_row_refuses_at_the_mint() {
    assert!(ValidKeys::of(64).is_some());
    assert!(ValidKeys::of(65).is_none());
}

#[test]
fn the_chunk_family_first_window_always_holds_a_valid_lane() {
    let mq = NonZeroU32::new(4).expect("4 is nonzero");
    let rows = ChunkRows::chunk(
        RungWidth::of_baked_rows(4)
            .expect("a test rung is nonzero")
            .rows(),
    );
    // Window boundaries of the sweep: page edges, window edges, and one past them, so the clamp
    // (`min(t, real−1)`) and the `real == mq` identity are both exercised.
    for start in [0usize, 1, 63, 64, 65, 191, 192, 193, 255, 256, 300] {
        for real in 1..=4u32 {
            let real = NonZeroU32::new(real).expect("the sweep is nonzero");
            let mask = spliced_attn_mask_f16(region(3, 4), rows, start, real)
                .expect("real within the rung is a chunk this region states");
            let least = mask
                .first_window_valid_keys(mq)
                .expect("the region states first-window rows");
            assert!(
                least.get() >= 1,
                "start {start} real {real}: the first window holds {} valid lane(s) — the \
                 carry's −65504 seed met an all-masked window, and the fold's l would count \
                 lanes the mask meant to silence",
                least.get()
            );
        }
    }
}

#[test]
fn a_chunk_rows_count_is_its_own_prefix_of_zero_lanes() {
    let region = region(2, 4);
    let mq = 4usize;
    let w4 = region.w4() as usize;
    assert!(w4 >= 2, "this geometry must state a second window to pin");
    let rows = ChunkRows::chunk(
        RungWidth::of_baked_rows(4)
            .expect("a test rung is nonzero")
            .rows(),
    );
    // start 5, real 3 of a 4-row rung: pos(t) = 5 + min(t, 2) = [5, 6, 7, 7] — the pad-row clamp
    // sends row 3 to the last real token's position.
    let mask = spliced_attn_mask_f16(region, rows, 5, NonZeroU32::new(3).expect("3 is nonzero"))
        .expect("real 3 of a 4-row rung is a chunk this region states");
    let bytes = mask.wire_bytes();
    assert_eq!(mask.rows(), 2 * w4 * mq);
    assert_eq!(bytes.len(), mask.rows() * 64 * 2);
    // Window (0, 0): slot_base 0, so the count is pos+1 — INCLUDING the +1 at pos == slot_base
    // (lane 0 is valid; the count is not the difference but the difference plus one).
    for (t, want) in [6u32, 7, 8, 8].into_iter().enumerate() {
        assert_eq!(
            mask.window_valid(t, 0).get(),
            want,
            "window (0,0) row {t}: the causal count is pos+1, clamped to the row's lanes"
        );
    }
    // Window (0, 1): slot_base 64, every pos < 64 — an ALL-MASKED window, legal ONLY past the
    // first (the carry's m has been raised by then), and its count is ZERO.
    for t in 0..mq {
        assert_eq!(
            mask.window_valid(mq + t, 0).get(),
            0,
            "window (0,1) row {t}: a window past every position counts zero valid lanes"
        );
    }
    // EVERY row: its bytes are its count's own prefix — `count` zero lanes, then −65504 to the
    // end. The count and the bytes are one fact; this is the byte-side half of that statement.
    for r in 0..mask.rows() {
        let k = mask.window_valid(r, 0).get() as usize;
        let lanes = row_lanes(bytes, r);
        for (c, v) in lanes.iter().enumerate() {
            let want = if c < k {
                half::f16::from_f32(0.0)
            } else {
                half::f16::MIN
            };
            assert_eq!(*v, want, "row {r} lane {c}: bytes and count disagree");
        }
    }
}

#[test]
fn the_batch_family_counts_its_pre_replicated_rows() {
    let pool = pool();
    let (nkvh, gqa, seqs) = (8u32, 4u32, 4u32);
    let w4 = pool
        .windows_per_page()
        .expect("hd 64 divides a page into whole windows");
    assert!(
        w4 >= 4,
        "this geometry must state the window holding a mid-page write slot"
    );
    let ktb = CopyDims::ENTRIES_PER_OP * nkvh;
    let per_chunk = u64::from(w4) * u64::from(ktb) * 4;
    let batch = SplicedDecodeBatch::of_ids_extent(
        per_chunk * 2,
        pool,
        NonZeroU32::new(gqa).expect("gqa is nonzero"),
        NonZeroU32::new(seqs).expect("a batch is nonzero"),
    )
    .expect("two whole chunk tables at this width is a batch this family states");
    // Ends 70, 200, 9, 0 with the shared write slot at 200: a row spanning slots 192..255 holds
    // invalid-valid-invalid for the shortest history — the count is a COUNT, not a prefix.
    let hists = vec![
        KvHistory::contiguous(70),
        KvHistory::contiguous(200),
        KvHistory::contiguous(9),
        KvHistory::contiguous(0),
    ];
    let write = BatchSlot::of(hists.iter());
    let mask = spliced_decode_mask_f16(batch, &hists, write)
        .expect("one history per launch row is a batch this family states");
    let rows = 2 * w4 as usize * ktb as usize * gqa as usize;
    assert_eq!(mask.rows(), rows);
    assert_eq!(mask.wire_bytes().len(), rows * 64 * 2);
    let bytes = mask.wire_bytes();
    let mut non_prefix = 0usize;
    for r in 0..rows {
        let lanes = row_lanes(bytes, r);
        // The g-lanes are pre-replicated: every run of `gqa` consecutive rows is ONE sequence's
        // validity stated `gqa` times, so their counts are equal.
        if r % gqa as usize != 0 {
            assert_eq!(
                mask.window_valid(r, 0).get(),
                mask.window_valid(r - r % gqa as usize, 0).get(),
                "row {r}: a pre-replicated g-lane must carry its group's own count"
            );
        }
        // The count is the row's own zero-lane count — minted by the same loop, agreeing by
        // construction, and NOT a prefix: the write slot lands inside masked runs.
        let zeros = lanes
            .iter()
            .filter(|&&v| v == half::f16::from_f32(0.0))
            .count();
        assert_eq!(
            zeros,
            mask.window_valid(r, 0).get() as usize,
            "row {r}: bytes and count disagree"
        );
        let k = mask.window_valid(r, 0).get() as usize;
        let prefix = lanes.iter().take(k).all(|&v| v == half::f16::from_f32(0.0));
        if k > 0 && !prefix {
            non_prefix += 1;
        }
    }
    assert!(
        non_prefix > 0,
        "the write slot at 200 inside a history ending at 70 must produce at least one \
         valid-invalid-valid row — a count that were only a prefix would not have pinned it"
    );
}
