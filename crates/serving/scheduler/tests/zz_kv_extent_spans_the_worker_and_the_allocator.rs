// SPDX-License-Identifier: Apache-2.0

//! ⭐⭐⭐ THE ONE TEST THAT SPANS THE WORKER/SCHEDULER SEAM FOR KV EXTENT.
//!
//! Two facts have to hold TOGETHER for prefix caching to be correct on a backend whose keys do not sit
//! at their token positions (the spyre paged pool, which appends a whole batch AT ONE SLOT per step and
//! therefore leaves a masked hole in every request shorter than its batch-mates):
//!
//! 1. blocks are allocated for the SLOT SPAN — or the worker writes past the last block it was given;
//! 2. only the leading TOKEN-ADDRESSED run is hashed into the cache — or a later request is handed a
//!    block whose keys belong to different tokens than its hash claims.
//!
//! ⛔ EACH ONE ALONE LOOKS FINE AND IS A WRONG-OUTPUT BUG. That is why this is one test over both, with
//! the SAME extent driving both assertions: `~360 green per-module tests` once coexisted with no program
//! able to cross this kind of seam ([[per-module-tests-cannot-see-the-seam]]), because each side was
//! tested against its own idea of the other.
//!
//! The numbers are the measured shape of the real defect: a 450-token prompt whose history reached 514
//! slots (`52551252`). At a 256-token block that is 2 pages of tokens and 3 pages of slots.

use scratchy_core_common::kv::{CacheableTokens, KvExtent, KvSlotSpan};
use scratchy_core_common::{Request, SamplingParams};
use scratchy_serving_scheduler::scheduler::core::{KVCacheManagerOps, SimpleBlockTracker};

const BLOCK: usize = 256;

/// The tracker as the spyre pool declares it, with a step reach already set — the state every batched
/// decode step allocates in.
fn batched_tracker(step_reach: u32) -> SimpleBlockTracker {
    let mut t = SimpleBlockTracker::with_caching(64, BLOCK).sharing_one_write_slot();
    t.set_step_reach(KvSlotSpan::new(step_reach));
    t
}

fn request_of(id: &str, tokens: usize, extent: Option<KvExtent>) -> Request {
    let mut r = Request::new(
        id.into(),
        (0..tokens as u32).collect(),
        SamplingParams {
            max_tokens: Some(8),
            ..Default::default()
        },
        1.0,
        0,
        0,
        None,
    );
    r.kv_extent = extent;
    r
}

/// A request whose keys span MORE slots than it has tokens gets blocks for the SLOTS, and hashes for
/// the TOKEN-ADDRESSED PREFIX ONLY. Both from one `KvExtent`.
#[test]
fn a_hole_bearing_request_is_allocated_by_slots_and_cached_by_its_contiguous_prefix() {
    let mut t = SimpleBlockTracker::with_caching(64, BLOCK);

    // 450 tokens, but the batch's shared write slot pushed its history to 514 slots. Its first
    // contiguous run — the part where token `n` is at slot `n` — is the 450-token prompt.
    let extent = KvExtent::new(KvSlotSpan::new(514), CacheableTokens::new(450));
    let req = request_of("hole", 450, Some(extent));

    let blocks = t
        .allocate_slots(&req, 450, 0, &[])
        .expect("64 blocks is plenty");
    let n = blocks[0].len();

    // 1. ALLOCATED BY SLOTS: 514 slots + the 450 this step writes... the rule is
    //    `max(span + new, computed + new)`, and the span is what makes it 3 blocks rather than 2. A
    //    token-count allocation gives ceil(450/256) = 2, and the worker's write at slot 514 then lands
    //    in a page it was never given.
    assert_eq!(
        n, 4,
        "blocks must cover the SLOT span (514) plus this step's tokens (450) = 964 slots → 4 blocks; \
         got {n}, which is what a token-count allocation would give"
    );

    // 2. CACHED BY THE CONTIGUOUS PREFIX: 450 cacheable tokens = ONE full 256-block. The second block
    //    holds tokens 256..450 and is not full, and nothing past 450 may be hashed at all.
    let mut same_prefix = request_of("reuse", 450, None);
    same_prefix.kv_extent = None;
    let (hit_tokens, _ids) = t.get_computed_blocks(&same_prefix);
    assert_eq!(
        hit_tokens, 256,
        "exactly the full blocks of the CACHEABLE prefix may be hittable — one 256-token block"
    );
}

/// ⛔ THE CONTROL: with no extent reported — every backend whose KV is token-addressed — both rules are
/// exactly the arithmetic they always were. A regression here is a change to cuda/metal behaviour.
#[test]
fn a_backend_that_reports_nothing_is_allocated_and_cached_by_its_token_count() {
    let mut t = SimpleBlockTracker::with_caching(64, BLOCK);
    let req = request_of("plain", 450, None);

    let blocks = t.allocate_slots(&req, 450, 0, &[]).expect("plenty");
    assert_eq!(
        blocks[0].len(),
        2,
        "450 tokens = ceil(450/256) = 2 blocks, unchanged"
    );

    let (hit_tokens, _) = t.get_computed_blocks(&request_of("plain2", 450, None));
    assert_eq!(hit_tokens, 256, "one full block hashed, unchanged");
}

/// ⛔ AND THE HALF-FIX FAILS LOUDLY: a span reported WITHOUT capping the cache would hash a block that
/// covers the hole. Expressed as the property rather than the mechanism — nothing past the cacheable
/// prefix may ever become a cache hit, however many slots the request occupies.
#[test]
fn no_block_past_the_cacheable_prefix_can_ever_be_hit() {
    let mut t = SimpleBlockTracker::with_caching(64, BLOCK);

    // 600 tokens; the hole starts at 300, so only ONE full block (0..256) is token-addressed even
    // though the request holds 3 pages of keys.
    let extent = KvExtent::new(KvSlotSpan::new(900), CacheableTokens::new(300));
    let req = request_of("late-hole", 600, Some(extent));
    t.allocate_slots(&req, 600, 0, &[]).expect("plenty");

    // A later request with the SAME 600 tokens must hit only the first block. Hitting two would mean
    // block 1 was hashed as "tokens 256..512" while it actually holds keys from past the hole.
    let (hit_tokens, _) = t.get_computed_blocks(&request_of("later", 600, None));
    assert_eq!(
        hit_tokens, 256,
        "the cache may promise only the token-addressed prefix; {hit_tokens} tokens hit means a block \
         covering the hole was hashed"
    );
}

/// ⛔⛔⛔ THE CARD'S SHAPE, ON THE HOST — WHICH SIDE OF THE SEAM DROPS THE BLOCKS?
///
/// MEASURED on granite-3.1-2b, 4 of 20 identical runs (the firing is intermittent, so one quiet run
/// proves nothing):
/// ```text
/// [step-reach] asked=798 applied=798 pool_reach=Some(697)
/// [hole-page] logical page 2 of 3 has no host block: history ends at slot 441,
///             write page Some(2), host granted 2 block(s), row KvRow(0)
/// ```
/// The reach WAS applied — `asked == applied == 798` — yet the worker saw 2 blocks, which is exactly
/// `ceil(441/256)`, the request's own token count. So either the tracker computes `needed` from the reach
/// and hands back fewer, or everything downstream of it truncates. This test pins the tracker's half: with
/// a deep step reach, a shallow request must HOLD `ceil(reach/BLOCK)` blocks.
///
/// If this PASSES, the tracker is innocent and the loss is downstream (scheduler output → `InputBatch`
/// → the worker's `host` slice) — which is where to look next, instead of at the arithmetic again.
#[test]
fn a_shallow_request_under_a_deep_step_reach_holds_blocks_for_the_reach() {
    // ⛔ `sharing_one_write_slot()` IS THE POINT: the tracker used to infer this from `set_kv_pool_reach`
    // having been called, and this very test reproduced the card's failure on the host when that inference
    // was removed — `[step-reach] asked=798 applied=0`. The capability is DECLARED now.
    let mut t = SimpleBlockTracker::with_caching(64, BLOCK).sharing_one_write_slot();
    t.set_kv_pool_reach(KvSlotSpan::new(697));
    t.set_step_reach(KvSlotSpan::new(798));

    // The card's row: 441 slots of its own history, in a step whose shared write slot reaches 798.
    let extent = KvExtent::new(KvSlotSpan::new(441), CacheableTokens::new(441));
    let req = request_of("shallow", 441, Some(extent));
    let blocks = t
        .allocate_slots(&req, 1, 0, &[])
        .expect("64 blocks is plenty");

    // ⭐ 3 = its own 2 pages (441 slots + 1 appended) + ONE for the shared write page.
    //
    // ⛔ THIS ASSERTED **4** FIRST — `ceil(798/256)`, i.e. allocating the masked gap between this row's keys
    // and the write page as though it held keys. It does not: `map_row` aliases every fully-masked page to one
    // scratch page. Over-allocating it starved the block pool and cost 20% more prefill chunks, which on a
    // prefill that is not chunk-width invariant moves generated text.
    assert_eq!(
        blocks[0].len(),
        3,
        "own pages (2) + the write page (1); 2 blocks is the token-count answer that leaves the worker \
         serving the write page from its own reserve, and 4 pays for masked pages nobody reads"
    );
    assert_eq!(
        t.get_blocks("shallow")[0].len(),
        3,
        "and the table the engine will SHIP must hold them too — `allocate_slots` returning 3 while the \
         stored table holds fewer is precisely how the worker ends up short"
    );

    // ⛔ THE CONTROL, IN THE SAME TEST: a tracker that did NOT declare the capability must be untouched by
    // any reach. This is cuda and metal, and a regression here is a change to their allocation.
    let mut plain = SimpleBlockTracker::with_caching(64, BLOCK);
    plain.set_kv_pool_reach(KvSlotSpan::new(697));
    plain.set_step_reach(KvSlotSpan::new(798));
    let same = request_of("shallow", 441, Some(extent));
    let blocks = plain.allocate_slots(&same, 1, 0, &[]).expect("plenty");
    assert_eq!(
        blocks[0].len(),
        2,
        "without the declaration the reach means nothing: 441 slots + 1 = 2 blocks, exactly as before"
    );
}

/// ⛔⛔⛔ THE REPORT IS ONE FINALIZED STEP OLD, AND AT A PAGE BOUNDARY THAT IS ONE PAGE SHORT.
///
/// MEASURED ON CARD at the SHIPPED, capped width 8 — not at an experimental width — granite-3.1-2b fp8,
/// ragged probe, **5 of 5 trials**, and 1 of 3 on the shallower ragged probe:
/// ```text
/// cannot map 513 slot(s) = 3 page(s): the host granted 2 block(s)
/// cannot map 769 slot(s) = 4 page(s): the host granted 3 block(s)
/// ```
/// Both are `k*256 + 1` slots against exactly `k` blocks. The engine's async loop finalizes step `n-1`,
/// schedules step `n+1`, then waits for step `n` (`gpu_in_flight < 2`), so the extent the allocator reads
/// describes the step BEFORE the one about to run — and a batched step advances the shared write slot by
/// exactly one slot, which is one PAGE whenever it lands on a boundary.
///
/// ⭐ WHY IT NEEDS A RAGGED BATCH, WHICH IS WHAT MADE IT LOOK LIKE A PAGE-CROSSING BUG: `blocks_this_step`
/// maxes the span against the request's own token count, and that term is the scheduler's own bookkeeping
/// and never stale. A row with no masked hole has as many tokens as slots, so the token term covers the
/// missing page and the staleness is invisible. Only a row carrying a hole — fewer tokens than slots — has
/// nothing to fall back on. `--max-num-seqs 1` cannot produce a hole at all.
///
/// ⛔ AND THE `+1` WRITE-PAGE ARM CANNOT RESCUE IT, which is why nothing else caught this: `step_reach` is
/// derived from the same reported span, so it ages by the same slot and its page count matches `own`
/// exactly when `own` is the one that is short.
#[test]
fn an_extent_one_step_in_flight_is_allocated_for_the_slot_the_launch_will_write() {
    // ⭐ THE TWO MEASURED FACTS, AND NOTHING DERIVED BY THE CODE UNDER TEST. At the failing step the last
    // FINALIZED report said 511 slots and exactly one step was on the card, so the row's keys really end at
    // 512 and the launch is about to write slot 512.
    const REPORTED_SLOTS: u32 = 511;
    const STEPS_IN_FLIGHT: u32 = 1;
    let reported = KvExtent::new(KvSlotSpan::new(REPORTED_SLOTS), CacheableTokens::new(250));
    let mut req = request_of("ragged", 250, Some(reported));
    req.num_output_placeholders = STEPS_IN_FLIGHT;

    // ⭐ AND THE EXPECTED PAGE COUNT IS THE WORKER'S OWN RULE, NOT A CONSTANT THIS TEST PICKED:
    // `RowPages::holding(batch_slot + 1)` is `ceil((end + 1) / PAGE_SLOTS)`.
    let current_end = (REPORTED_SLOTS + STEPS_IN_FLIGHT) as usize;
    let launch_needs = (current_end + 1).div_ceil(BLOCK);
    assert_eq!(
        current_end % BLOCK,
        0,
        "the reproduction is the page BOUNDARY: off it, `own` rounds up to the same page anyway"
    );

    // ⭐ OVER **BOTH** REACHES, AND THAT IS THE POINT. The step reach is `end + the row's append`, so the
    // aged-out scheduler produced `511 + 1` and the corrected one produces `512 + 1`. The per-request
    // allocation must reach the launch's own slot either way: the reach is a pool-wide upper bound, and this
    // row's own pages are not its to get right. Pinning only the corrected reach would pass on the defect —
    // the write-page arm covers the missing page when the reach alone is fresh.
    for reach in [REPORTED_SLOTS + 1, (current_end + 1) as u32] {
        let mut t = batched_tracker(reach);
        let blocks = t
            .allocate_slots(&req, 1, 0, &[])
            .expect("64 blocks is plenty");
        assert_eq!(
            blocks[0].len(),
            launch_needs,
            "the launch writes slot {current_end} and maps {launch_needs} pages; at step reach {reach} the \
             allocation gave {} — {} is the count the card refused",
            blocks[0].len(),
            launch_needs - 1
        );
        assert_eq!(
            t.get_blocks("ragged")[0].len(),
            launch_needs,
            "and the table the engine SHIPS must hold them — a grant the stored table does not carry is \
             exactly how the worker ends up short"
        );
    }

    // ⛔ THE CONTROL — THE SYNCHRONOUS PATH IS UNTOUCHED. With nothing in flight the report is already
    // current, and the same true state must give the same allocation. A change here is a change to every
    // non-async run.
    let sync_report = KvExtent::new(
        KvSlotSpan::new(current_end as u32),
        CacheableTokens::new(250),
    );
    let sync_req = request_of("sync", 250, Some(sync_report));
    assert_eq!(sync_req.num_output_placeholders, 0, "nothing in flight");
    let mut sync_t = batched_tracker((current_end + 1) as u32);
    assert_eq!(
        sync_t.allocate_slots(&sync_req, 1, 0, &[]).expect("plenty")[0].len(),
        launch_needs,
        "an up-to-date report and an aged one describing the SAME state must allocate the same"
    );

    // ⛔ AND THE TOKEN-ADDRESSED CONTROL: cuda and metal report no extent at all, so nothing here can reach
    // them. A regression on this line is a change to their allocation.
    let mut plain = SimpleBlockTracker::with_caching(64, BLOCK);
    plain.set_step_reach(KvSlotSpan::new((current_end + 1) as u32));
    let mut fresh = request_of("plain-async", 300, None);
    fresh.num_output_placeholders = 1;
    assert_eq!(
        plain.allocate_slots(&fresh, 300, 0, &[]).expect("plenty")[0].len(),
        300usize.div_ceil(BLOCK),
        "no extent and no shared write slot: the token count rules, exactly as before"
    );
}

/// ⛔⛔⛔ A REQUEST ADMITTED BEFORE THE WORKER'S FIRST REPORT — THE ARM THAT SKIPPED THE REACH.
///
/// `allocate_slots` had a nested match: with a pool reach it used the reach, and WITHOUT one it fell back to
/// sizing by token count — silently ignoring the step reach the scheduler had just computed and the tracker
/// had just stored. Every request admitted before the first report went down that arm, which is exactly the
/// chunked-prefill phase where the shared write slot runs ahead of the shallow rows.
///
/// MEASURED, granite-3.1-2b:
/// ```text
/// [step-reach] asked=698 applied=698 pool_reach=None
/// [hole-page] logical page 2 of 3 has no host block: history ends at slot 441, host granted 2 block(s)
/// ```
/// `2 == ceil(441/256)` — the token-count answer, from the arm that had no reach in it.
#[test]
fn a_new_request_is_allocated_for_the_step_reach_before_any_pool_reach_exists() {
    let mut t = SimpleBlockTracker::with_caching(64, BLOCK).sharing_one_write_slot();
    // NO `set_kv_pool_reach` — this is the state on every step up to the worker's first report.
    t.set_step_reach(KvSlotSpan::new(698));

    // A request nothing has run: no extent, 441 tokens, joining a batch whose write slot reaches 698.
    let req = request_of("fresh", 441, None);
    let blocks = t
        .allocate_slots(&req, 441, 0, &[])
        .expect("64 blocks is plenty");
    assert_eq!(
        blocks[0].len(),
        3,
        "the step reaches slot 698, so 3 blocks (768 slots) put the write page inside; 2 is ceil(441/256), \
         the token-count answer that left the worker serving page 2 from its own reserve"
    );

    // The control again: undeclared, so the reach is inert and this is the cuda/metal answer.
    let mut plain = SimpleBlockTracker::with_caching(64, BLOCK);
    plain.set_step_reach(KvSlotSpan::new(698));
    let same = request_of("fresh", 441, None);
    assert_eq!(
        plain.allocate_slots(&same, 441, 0, &[]).expect("plenty")[0].len(),
        2,
        "441 tokens = 2 blocks on a token-addressed backend, untouched"
    );
}
