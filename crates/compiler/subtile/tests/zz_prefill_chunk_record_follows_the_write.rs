#![cfg(feature = "superdsc")]
//! ⛔⛔⛔ A PREFILL CHUNK'S RECORD MUST FOLLOW ITS WRITE ACROSS A PAGE BOUNDARY.
//!
//! THE SEAM THIS SPANS: `PagedKvPool::chunk_write_start` decides WHERE a chunk writes and may step it
//! BACK so the chunk's padded window stays inside one page; `KvHistory` decides WHICH SLOTS THE MASK
//! CALLS VALID. Every per-module test of either side passed while they disagreed, because the disagreement
//! is a value neither side owns: the chunk wrote `[placed, placed + len)` and the history recorded
//! `[own_end, own_end + len)`. Those are the same range for every chunk whose padded window already fits
//! its page, so short prompts could not see it.
//!
//! MEASURED on granite-3.1-8b fp8 at `live=1` (no batch at all), 2026-08-12:
//!   * 450-token prompt: two step-backs of 32 ⇒ history ended at 514, position at 450.
//!   * 731-token prompt: three step-backs of 32 ⇒ history ended at 827, position at 731.
//! The extra slots are marked valid and hold whatever the pool held before, so a softmax attends
//! real-looking keys the request never said — fluent output that drifts, worse the longer the prompt,
//! and non-monotone in length because whether a chunk straddles a page depends on `total % PAGE_SLOTS`.
//!
//! WHAT THIS ASSERTS is the one value that is only right if both sides agree: after replaying the
//! worker's chunk loop through the REAL placement functions, the history must end EXACTLY at the token
//! count, and must call the last token's slot valid and the one past it not.

use scratchy_subtile::sdsc_abstract::{BatchSlot, KvHistory, KvSlot, PagedKvPool, SlotCount};

/// Replay `crates/serving/worker/src/spyre_worker.rs`'s prefill chunk loop for a `total`-token prompt
/// served in `cap_m`-wide chunks, and return the history it builds.
///
/// The arithmetic here is the worker's, line for line — `off` is the token offset, `chunk_start` the KV
/// write slot, and the two move together because stepping back moves both.
fn replay_prefill(total: usize, cap_m: usize) -> KvHistory {
    let mut hist = KvHistory::default();
    let mut off = 0usize;
    let mut chunk_start = 0usize;
    let mut first = true;
    while off < total {
        // A continuation chunk runs on the prefix bundle, so its cache write covers that bundle's BAKED
        // row count; the first chunk runs on an exact rung and pads nothing.
        let padded = SlotCount::new(if first { 0 } else { cap_m as u32 });
        let placed = PagedKvPool::chunk_write_start(KvSlot::new(chunk_start as u32), padded);
        if placed.slot().get() as usize != chunk_start {
            let back = chunk_start - placed.slot().get() as usize;
            chunk_start = placed.slot().get() as usize;
            off -= back;
        }
        let room = PagedKvPool::chunk_room(placed.slot()).get() as usize;
        let chunk_len = (total - off).min(cap_m).min(room);
        assert!(
            chunk_len > 0,
            "a chunk of no tokens would spin forever (total={total} cap_m={cap_m})"
        );
        hist.record_chunk(placed.wrote(SlotCount::new(chunk_len as u32)));
        chunk_start += chunk_len;
        off += chunk_len;
        first = false;
    }
    hist
}

#[test]
fn a_prompts_history_ends_exactly_at_its_token_count() {
    // 96 is the width the 8b's prefix bundle actually bakes and the width that produced the measured
    // 514-vs-450; 64 and 128 are the neighbouring rungs, and 256 divides the page so it never steps back
    // (the case that hid the defect — it must stay right too).
    for cap_m in [64usize, 96, 128, 256] {
        for total in 1usize..=2200 {
            let hist = replay_prefill(total, cap_m);
            assert_eq!(
                hist.end().get() as usize,
                total,
                "cap_m={cap_m} total={total}: history ends at {} but the prompt is {total} tokens — \
                 {} slot(s) are marked valid that no chunk wrote",
                hist.end().get(),
                hist.end().get() as usize - total,
            );
        }
    }
}

#[test]
fn b_the_slot_past_the_last_token_is_not_valid() {
    for cap_m in [64usize, 96, 128, 256] {
        // The lengths that cross a page boundary are the ones that step back; 260..280 and 510..530
        // straddle the first and second boundaries at PAGE_SLOTS == 256.
        for total in (1usize..=40)
            .chain(250..290)
            .chain(500..540)
            .chain(720..760)
        {
            let hist = replay_prefill(total, cap_m);
            assert!(
                hist.contains(total - 1),
                "cap_m={cap_m} total={total}: the last token's slot is not in the history"
            );
            assert!(
                !hist.contains(total),
                "cap_m={cap_m} total={total}: slot {total} is valid, but the prompt ends at {}",
                total - 1
            );
        }
    }
}

/// ⛔ THE WITNESS THAT THE THREE TESTS ABOVE ARE NOT VACUOUS.
///
/// Replays the SAME loop with the recording the worker used until 2026-08-12 — `record` at the request's
/// own end, which clamps forward and therefore ignores the step-back — and pins the two numbers measured
/// on the card. If a future change makes this test fail, the old defect is no longer reachable and this
/// test should be deleted; if it fails because the numbers moved, the placement arithmetic changed and
/// the invariant tests above are what to trust.
#[test]
fn d_recording_at_the_requests_own_end_over_records_exactly_as_measured() {
    fn replay_with_the_old_recording(total: usize, cap_m: usize) -> KvHistory {
        let mut hist = KvHistory::default();
        let (mut off, mut chunk_start, mut first) = (0usize, 0usize, true);
        while off < total {
            let padded = SlotCount::new(if first { 0 } else { cap_m as u32 });
            let placed = PagedKvPool::chunk_write_start(KvSlot::new(chunk_start as u32), padded);
            if placed.slot().get() as usize != chunk_start {
                let back = chunk_start - placed.slot().get() as usize;
                chunk_start = placed.slot().get() as usize;
                off -= back;
            }
            let room = PagedKvPool::chunk_room(placed.slot()).get() as usize;
            let chunk_len = (total - off).min(cap_m).min(room);
            // ⛔ THE DEFECT: the slot comes from the history, not from `placed`.
            hist.record(BatchSlot::solo(&hist), SlotCount::new(chunk_len as u32));
            chunk_start += chunk_len;
            off += chunk_len;
            first = false;
        }
        hist
    }

    // The two prompts read off the card by `[kcache-batch]`, at the 8b's baked prefix width of 96.
    assert_eq!(
        replay_with_the_old_recording(450, 96).end().get(),
        514,
        "the measured 450 -> 514"
    );
    assert_eq!(
        replay_with_the_old_recording(731, 96).end().get(),
        827,
        "the measured 731 -> 827"
    );
    // And the fix must not merely shrink the excess — it must remove it.
    assert_eq!(replay_prefill(450, 96).end().get(), 450);
    assert_eq!(replay_prefill(731, 96).end().get(), 731);
}

#[test]
fn c_a_contiguous_prompt_is_one_run_however_many_times_it_stepped_back() {
    // Stepping back re-writes slots the request already holds, so it must not fragment the history: a
    // hole would be masked valid by `col < p` reasoning elsewhere, and a fragmented representation of a
    // contiguous history breaks the uniqueness `KvHistory` documents.
    for cap_m in [64usize, 96, 128] {
        for total in [96usize, 256, 257, 450, 512, 731, 800, 1143, 1839] {
            let hist = replay_prefill(total, cap_m);
            assert_eq!(
                hist.runs().len(),
                1,
                "cap_m={cap_m} total={total}: a contiguous prefill produced {} runs — {:?}",
                hist.runs().len(),
                hist.runs(),
            );
        }
    }
}
