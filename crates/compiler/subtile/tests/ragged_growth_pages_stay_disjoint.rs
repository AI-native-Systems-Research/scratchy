//! ⭐ DO TWO ROWS EVER SHARE A PHYSICAL PAGE WHEN A RAGGED BATCH GROWS? — on the host, in milliseconds.
//!
//! The card says the defect is a function of exactly two things: **the row's own depth** and **the batch
//! being ragged**. Row position is not one of them — rotating the probe order leaves every row's output
//! BYTE-IDENTICAL. So the mechanism has to be something that differs for the deep row between a uniform and
//! a ragged batch, and the deep row's OWN history, page count and mask are identical in both. Only the other
//! rows differ, which points at shared state: a physical page held by two rows at once.
//!
//! That is the one story that fits all four facts. In a UNIFORM batch every row already holds `want` pages
//! from prefill, so `ensure_pages` returns early and draws nothing. In a RAGGED batch the shallow rows must
//! GROW to cover the shared write slot — and every live row then writes its new token at that ONE slot. If a
//! grown row is handed a page the deep row already holds, the two writes land on the same cell and one
//! silently overwrites the other's newest key, which is precisely "the deep row lost its recent context and
//! continued its own filler".
//!
//! This file is the allocator's side of that question, reproducing `ensure_pages`' draw rule exactly:
//! *the lowest physical page that no live request holds and this request does not already hold*, with the
//! held set recomputed before each request (as the worker does — it is inside the per-request loop).
//!
//! ⛔ WHY IT IS WORTH PINNING EVEN THOUGH THE WORKER LOOKS RIGHT. The `held_pages` snapshot is passed in as a
//! `&HashSet<usize>` beside the request, so *the caller* decides how fresh it is. Hoisting that one line out
//! of the loop — a change that looks like a harmless CSE, and which the original indentation made it look
//! like someone already had — makes every grown row draw the same pages. A test is the only thing that
//! notices, because the result is a fluent wrong answer with no fault.
#![cfg(feature = "spyre")]

use scratchy_subtile::sdsc_abstract::PagedKvPool;
use std::collections::HashSet;

/// `ensure_pages`' draw rule, verbatim: the lowest page no live request holds and this one does not.
fn draw(pool_pages: usize, held: &HashSet<usize>, own: &[usize], want: usize) -> Vec<usize> {
    let mut pages = own.to_vec();
    while pages.len() < want {
        let phys = (0..pool_pages)
            .find(|p| !held.contains(p) && !pages.contains(p))
            .expect("pool exhausted");
        pages.push(phys);
    }
    pages
}

/// Grow a ragged batch the way the worker does, and assert no two rows ever share a page.
fn grow_ragged(pool_pages: usize, initial: &[usize], want: usize) -> Vec<Vec<usize>> {
    // PREFILL: each row holds exactly the pages its own prompt needed, drawn in row order.
    let mut rows: Vec<Vec<usize>> = Vec::new();
    for &n in initial {
        let held: HashSet<usize> = rows.iter().flatten().copied().collect();
        rows.push(draw(pool_pages, &held, &[], n));
    }
    // DECODE: every row is grown to cover the SHARED write slot — the max over live ends, which is why a
    // shallow row grows at all. The held set is recomputed before each row, as the worker's loop does.
    for r in 0..rows.len() {
        let held: HashSet<usize> = rows
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != r)
            .flat_map(|(_, p)| p.iter().copied())
            .collect();
        let own = rows[r].clone();
        rows[r] = draw(pool_pages, &held, &own, want);
    }
    rows
}

fn assert_pairwise_disjoint(rows: &[Vec<usize>], what: &str) {
    for (a, pa) in rows.iter().enumerate() {
        let uniq: HashSet<usize> = pa.iter().copied().collect();
        assert_eq!(
            uniq.len(),
            pa.len(),
            "{what}: row {a} holds a page twice: {pa:?}"
        );
        for (b, pb) in rows.iter().enumerate().skip(a + 1) {
            let shared: Vec<usize> = pa.iter().filter(|p| pb.contains(p)).copied().collect();
            assert!(
                shared.is_empty(),
                "{what}: rows {a} and {b} SHARE page(s) {shared:?} — both write their new token at the \
                 batch's ONE shared slot, so one silently overwrites the other's newest key.\n  row {a}: \
                 {pa:?}\n  row {b}: {pb:?}"
            );
        }
    }
}

#[test]
fn a_ragged_batch_grown_to_the_shared_slot_keeps_every_rows_pages_disjoint() {
    // The profile the card fails on: 220 / 698 / 1268 / 1982 tokens = 1 / 3 / 5 / 8 pages, grown to 8.
    let want = 8usize;
    let rows = grow_ragged(136, &[1, 3, 5, 8], want);
    assert_pairwise_disjoint(&rows, "ragged 1/3/5/8 -> 8");
    for (r, pages) in rows.iter().enumerate() {
        assert_eq!(
            pages.len(),
            want,
            "row {r} must cover the shared slot: {pages:?}"
        );
    }
}

#[test]
fn the_uniform_batch_that_passes_on_card_draws_nothing_at_all() {
    // Every row already holds `want`, so `ensure_pages` returns early — the reason a uniform batch cannot
    // exercise the growth path, and therefore cannot see any bug that lives in it.
    let want = 8usize;
    let rows = grow_ragged(136, &[8, 8, 8, 8], want);
    assert_pairwise_disjoint(&rows, "uniform 8/8/8/8");
    assert_eq!(
        rows,
        vec![
            (0..8).collect::<Vec<_>>(),
            (8..16).collect::<Vec<_>>(),
            (16..24).collect::<Vec<_>>(),
            (24..32).collect::<Vec<_>>(),
        ],
        "a uniform batch's prefill draw is already final"
    );
}

#[test]
fn every_raggedness_up_to_the_ceiling_stays_disjoint() {
    // Not just the one profile: sweep the shapes a 4-wide batch can take, to the mask's page ceiling.
    let max = PagedKvPool::MAX_PAGES_PER_ROW as usize;
    for a in 1..=max {
        for b in 1..=max {
            for c in 1..=max {
                for d in 1..=max {
                    let initial = [a, b, c, d];
                    let want = *initial.iter().max().unwrap();
                    let rows = grow_ragged(4 * max + 8, &initial, want);
                    assert_pairwise_disjoint(&rows, &format!("ragged {initial:?} -> {want}"));
                }
            }
        }
    }
}

#[test]
fn hoisting_the_held_set_out_of_the_loop_makes_two_rows_share_a_page() {
    // ⭐ THE NEGATIVE CONTROL: the bug this file exists to catch, committed on purpose, so a passing run
    // above means the rule HOLDS rather than that the test cannot fail. Here `held` is snapshotted ONCE
    // before the growth loop instead of per row — the one-line change the old indentation implied.
    let pool = 136usize;
    let initial = [1usize, 3, 5, 8];
    let want = 8usize;
    let mut rows: Vec<Vec<usize>> = Vec::new();
    for &n in &initial {
        let held: HashSet<usize> = rows.iter().flatten().copied().collect();
        rows.push(draw(pool, &held, &[], n));
    }
    let stale: HashSet<usize> = rows.iter().flatten().copied().collect(); // <-- hoisted
    for r in 0..rows.len() {
        let own = rows[r].clone();
        rows[r] = draw(pool, &stale, &own, want);
    }
    let mut shared_any = false;
    for (a, pa) in rows.iter().enumerate() {
        for pb in rows.iter().skip(a + 1) {
            if pa.iter().any(|p| pb.contains(p)) {
                shared_any = true;
            }
        }
    }
    assert!(
        shared_any,
        "the negative control must FAIL to be a control — a stale held set has to produce a shared page, \
         or the assertions above prove nothing: {rows:?}"
    );
}
