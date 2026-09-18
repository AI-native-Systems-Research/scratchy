// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! THE TWO NUMBERS A WORKER OWES THE SCHEDULER ABOUT ITS KV, and the block size it addresses it by.
//!
//! The scheduler allocates KV blocks and hashes cached prefixes by TOKEN POSITION: block `b` holds
//! tokens `[b*block_size, (b+1)*block_size)`, and a cache hit hands another request that block on the
//! promise that it holds those tokens' keys. Both claims are the WORKER's to honour, and on a backend
//! where a request's keys do not sit at its token positions neither one is automatic:
//!
//! * a pool whose batched write appends EVERY row of a batch at ONE slot per step leaves a masked HOLE
//!   in any request shorter than its batch-mates, so its keys spread over MORE slots than it has tokens
//!   ([`KvExtent::span_now`]);
//! * and past that hole, token `t` no longer lives at slot `t`, so the block covering it is not the
//!   block the scheduler would hash it under — only the request's LEADING CONTIGUOUS run is
//!   token-addressed and therefore cacheable ([`KvExtent::cacheable_tokens`]).
//!
//! The PROPERTY is what this module is about, not any one backend: a worker reports these two numbers or
//! it reports nothing, and reporting nothing means "my keys are at their token positions" — which is what
//! every budget-sized paged GPU pool means. (The backend that needs it today is the IBM Spyre paged pool,
//! whose collapsed cache write is what makes one slot serve a whole batch.)
//!
//! ⛔ THESE TRAVEL TOGETHER, AS ONE VALUE. Sizing the allocation without capping the cache gives a
//! later request a block hash pointing at keys for the wrong tokens; capping the cache without sizing
//! the allocation makes the worker write past the blocks it was given. Both failures are fluent wrong
//! output with no fault raised, and both are invisible at the call site that has only one of the two
//! numbers — so there is one type carrying both, and no way to obtain either alone.

use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

/// HOW MANY TOKENS ONE KV BLOCK HOLDS — the scheduler's block size, which is also the worker's page,
/// because `slot = block * block_size + offset` is the scheduler's arithmetic and the worker's pool is
/// what it addresses.
///
/// 🛑 IT WAS A LITERAL IN THE ENGINE AND A CONST IN THE BACKEND. `effective_block_size` hard-coded 64
/// under a per-backend `#[cfg]`, describing a pool shape that had since been rebuilt with a different
/// page. Two constants for one quantity, four crates apart, with a comment as the only thing asserting
/// they matched. While the worker ignored the scheduler's block ids that disagreement was inert; the
/// moment a block id becomes a page number it is silent KV corruption. So the value comes FROM the
/// backend that addresses it, and the engine holds no copy.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Serialize, Deserialize)]
pub struct KvBlockTokens(NonZeroUsize);

impl KvBlockTokens {
    /// The block size a backend declares. `None` for zero, which would make every block index a
    /// division by zero rather than a small pool.
    pub fn new(tokens: usize) -> Option<KvBlockTokens> {
        NonZeroUsize::new(tokens).map(KvBlockTokens)
    }

    pub fn get(self) -> usize {
        self.0.get()
    }
}

/// A TOKEN COUNT WHOSE KEYS SIT AT ITS TOKEN POSITIONS — the prefix of a request that a token-indexed
/// block hash may legally name.
///
/// Distinct from a plain token count because that is exactly the substitution the type exists to
/// refuse: `num_tokens()` is what the scheduler caches by default and is WRONG here for every request
/// that has shared a batched step.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Serialize, Deserialize)]
pub struct CacheableTokens(u32);

impl CacheableTokens {
    /// Nothing may be cached — the safe value, and what every backend that does not report gets.
    pub const NONE: CacheableTokens = CacheableTokens(0);

    pub fn new(tokens: u32) -> CacheableTokens {
        CacheableTokens(tokens)
    }

    pub fn get(self) -> u32 {
        self.0
    }
}

/// ⛔⛔⛔ HOW A BACKEND ADDRESSES ITS KV — DECLARED AT CONSTRUCTION, NEVER INFERRED FROM A REPORT.
///
/// 🛑 IT WAS INFERRED, AND THAT IS A LATE ANSWER TO AN EARLY QUESTION. `SimpleBlockTracker` decided
/// "does this backend share one write slot?" by asking whether the worker had ever sent a pool reach —
/// which first arrives with the worker's FIRST REPORT, i.e. after a step has already been scheduled and
/// allocated. MEASURED on granite-3.1-2b with the decision printed:
/// ```text
/// [step-reach] asked=517 applied=0 pool_reach=None
/// ```
/// The scheduler computed the right reach and the tracker threw it away, for every allocation before the
/// first report — which is exactly when a prompt is being chunked and the shared write slot is running
/// ahead of the shallow rows. The hole pages that kept the spyre-specific KV structures alive came from
/// there.
///
/// A capability is a property of the backend, known before it runs; a report is a measurement of what it
/// did. Reading the second as the first is what let a default of "token-addressed" apply to a backend
/// that is not.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum KvAddressing {
    /// Key `n` lives at slot `n`: cuda, metal, and every backend whose pool is indexed by token. The
    /// default, so a backend that says nothing keeps the arithmetic it always had.
    #[default]
    ByToken,
    /// A batched step appends EVERY row at one shared slot, so a request shorter than its batch-mates has
    /// keys past its own token count and the host must allocate for the step's reach, not the request's
    /// length. The spyre paged pool.
    OneSharedWriteSlot,
}

impl KvAddressing {
    /// Does a step's reach mean anything here? The one question the allocator asks of it.
    pub fn shares_one_write_slot(self) -> bool {
        matches!(self, KvAddressing::OneSharedWriteSlot)
    }
}

/// ⭐⭐⭐ SLOTS APPENDED SINCE THE REPORT WAS MADE — how far a [`KvExtent`] has aged.
///
/// 🛑 **A REPORT IS A MEASUREMENT OF A PAST STEP, AND UNDER ASYNC SCHEDULING THE STEP BEING SCHEDULED IS
/// NOT THE NEXT ONE.** The engine's async loop finalizes step `n-1`, then schedules step `n+1`, then waits
/// for step `n`'s result (`gpu_in_flight < 2`, `crates/serving/api/src/engine.rs`). So the extent the
/// allocator reads is one *finalized* step behind the step it is allocating for, and on a backend that
/// appends every row at ONE shared slot each step, "one step behind" is "one slot short".
///
/// ⛔ MEASURED ON CARD, granite-3.1-2b fp8, `scr batch` at the SHIPPED width 8, ragged probe, 5 of 5 trials:
/// ```text
/// cannot map 513 slot(s) = 3 page(s): the host granted 2 block(s)
/// cannot map 769 slot(s) = 4 page(s): the host granted 3 block(s)
/// ```
/// Both are `k*256 + 1` slots against exactly `k` blocks: the shared write slot had reached a page
/// boundary, the launch needed the page past it, and the allocation had been sized from a span one slot
/// shallower — so `own` came out `k` and the write-page arm did not fire either, because `step_reach` is
/// derived from the same aged span. ⛔ IT FIRES ONLY ON RAGGED BATCHES because a row with no hole has
/// `own_tokens == its slots`, and that term (the scheduler's own bookkeeping, never stale) covers the
/// missing page; a row carrying a masked hole has fewer tokens than slots and nothing else to fall back
/// on. `--max-num-seqs 1` cannot fire it at all — no shared slot, no hole.
///
/// ⭐ THE COUNT IS THE SCHEDULER'S OWN `num_output_placeholders`: one per step scheduled and not yet
/// finalized, times the tokens that step appends. It is decremented in the same `update_from_output` that
/// applies the extent (`set_kv_extent` first, `append_output_tokens` after), so the pair is always
/// consistent — the span is as of the last FINALIZED step, and this counts the steps scheduled since.
/// [`InflightSlots::NONE`] on the synchronous path and on every token-addressed backend, which makes the
/// arithmetic there byte-identical to what it was.
///
/// ⛔ IT IS A REQUIRED ARGUMENT, NOT A DEFAULT, AND THAT IS THE GUARD. `KvExtent` has no `span()` any
/// more: the only way to get a slot span out of a report is to say how far it has aged, so "used a stale
/// span" stopped being a thing a call site can express. The bug was not a wrong formula anywhere — it was
/// three call sites reading a value whose age nobody had to name.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Serialize, Deserialize)]
pub struct InflightSlots(u32);

impl InflightSlots {
    /// Nothing is in flight — the synchronous path, and a request nothing has scheduled yet.
    pub const NONE: InflightSlots = InflightSlots(0);

    pub fn new(slots: u32) -> InflightSlots {
        InflightSlots(slots)
    }

    pub fn get(self) -> u32 {
        self.0
    }
}

/// HOW MANY KV SLOTS A REQUEST OCCUPIES — one past its last key, which is what the pool must hold
/// blocks for.
///
/// `>= ` its token count, and strictly greater once it has shared a step with a longer request.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Serialize, Deserialize)]
pub struct KvSlotSpan(u32);

impl KvSlotSpan {
    pub fn new(slots: u32) -> KvSlotSpan {
        KvSlotSpan(slots)
    }

    pub fn get(self) -> u32 {
        self.0
    }
    /// ⭐⭐⭐ SLOTS A REQUEST OCCUPIES THIS STEP — the ONE arithmetic every allocation decision goes through.
    ///
    /// 🛑 THIS WAS TWO FUNCTIONS WITH THE SAME BODY: `blocks_for_new_request` (self = the pool reach) and
    /// `KvExtent::blocks_needed` (self = the request's own span). Identical arithmetic reached from two ends is
    /// the shape that lets a fix land on one caller and miss the other — the defect family
    /// `why-locks-missed-the-prefill-bug` names, where newtypes cannot help because both spellings are already
    /// the same type. One function now, with every span that could bound the request as an argument.
    ///
    /// * `self` — where this request's own keys already end (`NONE` for a request being admitted).
    /// * `reach` — the deepest slot ANY row will have written by the **END** of this step. A batched step
    ///   appends every row at one shared slot, so a request shorter than its batch-mates writes past its own
    ///   tokens; unless the host allocated to `reach`, that page belongs to nobody and the worker serves it
    ///   from a reserve of its own — the whole reason a per-request KV row still exists below the host.
    /// * `writing` — slots THIS request appends this step.
    /// * `own_tokens` — the caller's own token arithmetic, so the answer is never less than the tokens need.
    ///
    /// ⛔⛔⛔ `reach` IS POST-APPEND, AND THAT DISTINCTION IS THE WHOLE BUG IT CLOSES. Measured on the 2b gate
    /// while `reach` was a PRE-append max: the hole arm still fired on the caching-OFF and permuted runs,
    /// because the shared write slot ends at `deepest_row_end + THAT ROW'S append`, while each request was
    /// allocated `its own end + ITS OWN append`. A deep row taking a full prefill chunk while a shallow row
    /// appends one decode token leaves the shallow row exactly one page short. So the three terms are maxed —
    /// `writing` is NOT added to `reach`, because `reach` already contains the deepest row's append.
    ///
    /// `reach` is [`KvSlotSpan::NONE`] on every token-addressed backend (cuda/metal), and the arithmetic is then
    /// exactly what it always was for them.
    pub fn slots_this_step(self, reach: KvSlotSpan, own_tokens: usize, writing: usize) -> usize {
        (self.get() as usize + writing)
            .max(reach.get() as usize)
            .max(own_tokens)
    }

    /// ⭐⭐⭐ BLOCKS THE HOST MUST OWN THIS STEP — its own pages, PLUS ONE for the shared write page.
    ///
    /// ⛔⛔⛔ NOT `ceil(reach / block)`, AND THAT MISTAKE COST 20% MORE PREFILL CHUNKS. The pages between a
    /// row's own keys and the batch's write page hold NO KEY: they are read fully-masked, and `BlockTable::
    /// map_row` aliases every one of them to a SINGLE SCRATCH PAGE. So the host never needs storage for them.
    /// What it needs is the pages that hold keys, plus the one page the launch writes into.
    ///
    /// Sizing by `ceil(reach/block)` allocated the masked middle as if it were real: with `own = 441` (2 pages)
    /// and a reach of 1594, SEVEN blocks instead of THREE, per row, every step. MEASURED consequence on the
    /// gate's `long` axis — chunks went 120 → 144 (+20%) because the fatter allocation starved the block pool
    /// and the scheduler chunked prefills harder. That matters beyond throughput: chunk width changes how many
    /// passes reduce the same prefix, and this prefill is NOT chunk-width invariant
    /// (`prefix-caching-changes-output-because-prefill-is-not-chunk-width-invariant`), so an allocation change
    /// moved GENERATED TEXT on granite-8b's near-tied probes.
    ///
    /// ⭐ AND IT RETIRES THE COST ARGUMENT FOR KEEPING THE WORKER'S RESERVE. The crossover test
    /// (`zz_the_hole_reserve_is_cheaper_than_host_allocation`) measured host allocation at 24-45 pages against
    /// the reserve's 9, which is what "the reserve is the cheaper design" rested on — but it measured
    /// `ceil(reach/block)`, the wrong rule. At own+1 the host pays ONE page per row for the write, which is
    /// what the reserve was paying for it.
    pub fn blocks_this_step(
        self,
        block: KvBlockTokens,
        reach: KvSlotSpan,
        own_tokens: usize,
        writing: usize,
    ) -> usize {
        // The pages this row actually holds keys in, by its own arithmetic — the reach plays no part.
        let own = (self.get() as usize + writing)
            .max(own_tokens)
            .div_ceil(block.get());
        // The page the launch writes into this step. Past this row's own pages exactly when the batch's write
        // slot is deeper than this row's keys reach — and then it costs ONE page, never the gap.
        let write_page_blocks = (reach.get() as usize).div_ceil(block.get());
        if write_page_blocks > own {
            own + 1
        } else {
            own
        }
    }

    /// A backend that addresses KV by token has no shared write slot at all. Named rather than `new(0)` so a
    /// call site reads as that fact instead of as a magic zero.
    pub const NONE: KvSlotSpan = KvSlotSpan(0);
}

/// WHAT ONE REQUEST'S KV ACTUALLY LOOKS LIKE TO THE WORKER, reported back every step so the
/// scheduler's next allocation and its next cache insertion are both sized by the truth.
///
/// Constructed only by [`KvExtent::new`], which takes both numbers, so a backend cannot report the
/// span it needs and leave the cacheable prefix to a default — see the module docs for why either one
/// alone is a wrong-output bug.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct KvExtent {
    span: KvSlotSpan,
    cacheable: CacheableTokens,
}

impl KvExtent {
    pub fn new(span: KvSlotSpan, cacheable: CacheableTokens) -> KvExtent {
        KvExtent { span, cacheable }
    }

    /// ⭐ SLOTS TO HOLD BLOCKS FOR, **AS OF NOW** — the span the worker reported, plus the slots appended
    /// by the steps scheduled since that report.
    ///
    /// 🛑 IT WAS `span()`, AND EVERY CALLER READ IT AS CURRENT. See [`InflightSlots`] for the card
    /// measurement: a report one finalized step behind is one slot shallow, and one slot is one PAGE
    /// whenever the shared write slot lands on a page boundary. Naming the age is what makes the three
    /// readers of this value agree about which step they are talking about.
    pub fn span_now(self, inflight: InflightSlots) -> KvSlotSpan {
        KvSlotSpan(self.span.0.saturating_add(inflight.0))
    }

    /// Leading tokens whose keys are at their token positions, and therefore the most a token-indexed
    /// prefix cache may claim.
    pub fn cacheable_tokens(self) -> CacheableTokens {
        self.cacheable
    }

    /// ⭐ BLOCKS THIS REQUEST NEEDS — its own span and the step's reach, through the one arithmetic.
    ///
    /// 🛑 IT WAS `cdiv(num_computed + num_new, block_size)` SPELLED AT THREE CALL SITES, and adding the span to
    /// one of them would have left the others sizing by tokens. It delegates to
    /// [`KvSlotSpan::blocks_this_step`] now, so the request-side and admission-side rules cannot drift apart.
    ///
    /// ⛔ `inflight` IS NOT OPTIONAL: the span this sizes from is [`Self::span_now`], because a request
    /// allocated from a span one step stale is a request one PAGE short at every page boundary.
    pub fn blocks_needed(
        self,
        block: KvBlockTokens,
        tokens_through_this_step: usize,
        writing: usize,
        reach: KvSlotSpan,
        inflight: InflightSlots,
    ) -> usize {
        self.span_now(inflight)
            .blocks_this_step(block, reach, tokens_through_this_step, writing)
    }

    /// ⭐ FULL BLOCKS A TOKEN-INDEXED CACHE MAY HASH — the caching rule, in the same one place.
    ///
    /// Capped by the cacheable prefix AND by whole blocks: a block hashed over anything past the first
    /// hole promises a later request the keys of tokens that are not there.
    pub fn hashable_full_blocks(self, block: KvBlockTokens, tokens_computed: usize) -> usize {
        std::cmp::min(tokens_computed, self.cacheable.get() as usize) / block.get()
    }
}

/// ⭐⭐⭐ THE ALLOCATION IS A CONSERVATION LAW, AND NEWTYPES CANNOT EXPRESS IT.
///
/// `slots_this_step` is reached from two ends — a RUNNING request through its own `KvExtent`, and a request
/// being ADMITTED through the pool reach — and both ends already speak `KvSlotSpan` and `KvBlockTokens`, so
/// no type can separate them. That is the family `why-locks-missed-the-prefill-bug` names: not two
/// quantities sharing a machine type, but ONE quantity computed at two call sites. What must hold is a
/// relation between the answer and every input, over all of them at once, which is a proof obligation.
#[cfg(kani)]
mod step_reach_proofs {
    use super::*;

    /// Bounded to keep the arithmetic in a range the checker decides quickly; the properties are ordering and
    /// divisibility facts, and neither has a magnitude threshold.
    fn inputs() -> (KvSlotSpan, KvSlotSpan, KvBlockTokens, usize, usize) {
        let span: u32 = kani::any();
        let reach: u32 = kani::any();
        let block: usize = kani::any();
        let own: usize = kani::any();
        let writing: usize = kani::any();
        kani::assume(span <= 4096 && reach <= 4096);
        kani::assume(block >= 1 && block <= 256);
        kani::assume(own <= 4096 && writing <= 256);
        (
            KvSlotSpan::new(span),
            KvSlotSpan::new(reach),
            KvBlockTokens::new(block).expect("assumed >= 1"),
            own,
            writing,
        )
    }

    /// ⛔ THE HOLE PAGE, MADE UNREACHABLE. A batched step appends every row at the shared write slot, so the
    /// blocks the host allocated must cover the whole step's `reach` — not just this request's own tokens.
    /// When they did not, the page the launch wrote belonged to nobody and the worker served it from a reserve
    /// of its own, which is the entire reason a per-request KV row still exists below the host.
    ///
    /// ⛔ `reach` IS POST-APPEND, so what must be covered is `reach`, NOT `reach + writing`. The first version
    /// of this proof asserted the latter and passed — because `slots_this_step` then ADDED `writing` to the
    /// max, which is a different (over-)allocation that still left the real hole open: the deepest row's
    /// append is not this row's append. The proof agreed with the code and both were wrong about the card.
    #[kani::proof]
    fn the_allocation_covers_every_slot_the_launch_writes() {
        let (span, reach, block, own, writing) = inputs();
        let blocks = span.blocks_this_step(block, reach, own, writing);
        let covered = blocks * block.get();

        // 1. This row's own keys and this step's append are inside its own pages.
        assert!(
            covered >= span.get() as usize + writing,
            "this row's own append is inside"
        );
        assert!(covered >= own, "never fewer blocks than the tokens need");

        // 2. AND THE PAGE THE LAUNCH WRITES INTO IS OWNED. Stated as a PAGE COUNT, not a slot bound: the
        //    masked pages between this row's keys and the write page are aliased to one scratch page, so
        //    `covered >= reach` would be the wrong (and much more expensive) property — it was the old one,
        //    and it cost 20% more prefill chunks.
        let own_pages = (span.get() as usize + writing)
            .max(own)
            .div_ceil(block.get());
        let write_page = (reach.get() as usize).div_ceil(block.get());
        if write_page > own_pages {
            assert!(
                blocks == own_pages + 1,
                "one page for the write, never the masked gap"
            );
            assert!(
                blocks * block.get() >= own_pages * block.get(),
                "and it is additional"
            );
        } else {
            assert!(
                blocks == own_pages,
                "the write page is already one of this row's own"
            );
        }
    }

    /// ⛔ THE COST PROPERTY, PROVED — because it is what makes the host able to own the write page at all.
    /// One extra page per row, whatever the depth of the batch. The old rule grew with the DEEPEST row, which
    /// is what made "the worker's reserve is cheaper" true.
    #[kani::proof]
    fn a_deep_batch_costs_one_extra_page_not_the_gap() {
        let (span, reach, block, own, writing) = inputs();
        let own_only = span.blocks_this_step(block, KvSlotSpan::NONE, own, writing);
        let with_reach = span.blocks_this_step(block, reach, own, writing);
        assert!(
            with_reach <= own_only + 1,
            "however deep the batch, a row pays at most ONE page more than its own keys need"
        );
        assert!(with_reach >= own_only, "and never fewer");
    }

    /// ⛔ NEITHER CALL SITE CAN LEAVE THE WRITE PAGE UNOWNED — the property that replaced an equality I had
    /// no business asserting.
    ///
    /// 🛑 THIS PROOF FIRST SAID `running == admitted` under one reach, AND IT PASSED — because the old rule
    /// maxed every term into a single slot count, so a running row's own keys and a fresh request's absence of
    /// them collapsed to the same number. They are not the same: a row holding 1000 slots of keys needs four
    /// more pages than a request holding none, and the corrected rule says so. The "conservation law" was an
    /// artifact of the over-allocating arithmetic, and keeping it would have blocked the fix that made
    /// allocation cheap. AN EQUALITY THAT ONLY HOLDS BECAUSE BOTH SIDES ARE TOO LARGE IS NOT AN INVARIANT.
    ///
    /// What must hold, and does, for BOTH arms: if the step's write page is past a request's own pages, the
    /// request is given exactly one more page for it. That is what closes the window on admission — the arm
    /// that used to be sized by token count alone.
    #[kani::proof]
    fn neither_arm_leaves_the_write_page_unowned() {
        let (span, reach, block, own, writing) = inputs();
        for held in [span, KvSlotSpan::NONE] {
            let blocks = held.blocks_this_step(block, reach, own, writing);
            let own_pages = (held.get() as usize + writing)
                .max(own)
                .div_ceil(block.get());
            let write_page = (reach.get() as usize).div_ceil(block.get());
            assert_eq!(
                blocks,
                if write_page > own_pages {
                    own_pages + 1
                } else {
                    own_pages
                },
                "own pages, plus one for the write page when the step reaches past them"
            );
            assert!(
                blocks * block.get() >= own,
                "and never fewer than the tokens need"
            );
        }
    }

    /// A DEEPER STEP NEVER ALLOCATES LESS. The scheduler computes one reach for the whole step and every
    /// request reads it, so the only safe direction for a wrong reach is over — this pins that direction.
    #[kani::proof]
    fn a_deeper_reach_never_shrinks_the_allocation() {
        let (span, reach, block, own, writing) = inputs();
        let deeper: u32 = kani::any();
        kani::assume(deeper >= reach.get() && deeper <= 4096);
        assert!(
            span.blocks_this_step(block, KvSlotSpan::new(deeper), own, writing)
                >= span.blocks_this_step(block, reach, own, writing)
        );
    }

    /// A TOKEN-ADDRESSED BACKEND IS UNTOUCHED. cuda and metal report no pool reach, so they pass
    /// [`KvSlotSpan::NONE`] and must get byte-identical arithmetic to what they had before the reach existed.
    #[kani::proof]
    fn no_reach_is_the_arithmetic_that_was_there_before() {
        let (span, _reach, _block, own, writing) = inputs();
        let slots = span.slots_this_step(KvSlotSpan::NONE, own, writing);
        assert_eq!(slots, (span.get() as usize + writing).max(own));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_block_size_is_not_a_block_size() {
        assert!(KvBlockTokens::new(0).is_none());
        assert_eq!(KvBlockTokens::new(256).map(|b| b.get()), Some(256));
    }

    #[test]
    fn an_extent_carries_both_numbers_or_neither() {
        // The point of the test is the SIGNATURE: there is no `KvExtent::of_span`, so a caller that
        // knows only the span cannot construct one. If that constructor ever appears, this comment is
        // the record of why it must not.
        let e = KvExtent::new(KvSlotSpan::new(514), CacheableTokens::new(450));
        assert_eq!(e.span_now(InflightSlots::NONE).get(), 514);
        assert_eq!(e.cacheable_tokens().get(), 450);
    }

    /// ⛔⛔⛔ THE CARD'S REFUSAL, AS ARITHMETIC: a report one finalized step behind is one PAGE short at
    /// exactly the page boundaries — and only for a row carrying a masked hole.
    ///
    /// MEASURED, granite-3.1-2b fp8 at the shipped width 8, ragged probe, 5 of 5 trials:
    /// `cannot map 513 slot(s) = 3 page(s): the host granted 2 block(s)`. The launch's demand (3) is the
    /// number to reproduce; `span_now` is what makes the allocation agree with it.
    #[test]
    fn a_span_one_step_stale_is_a_page_short_at_the_page_boundary() {
        let block = KvBlockTokens::new(256).expect("256 > 0");
        // The card's state. The last FINALIZED report said 511 slots; one step is on the card, so the row's
        // keys really end at 512 and the launch is about to write slot 512 — logical page 2. The row has a
        // hole, so its own token count (250) is far short of its slots and cannot cover for the span.
        let reported = KvExtent::new(KvSlotSpan::new(511), CacheableTokens::new(250));
        let inflight = InflightSlots::new(1);
        // `reach` is `end + the row's append`, derived from the same report — so it ages identically, which
        // is why the write-page arm did not rescue this on the card.
        let stale_reach = KvSlotSpan::new(reported.span_now(InflightSlots::NONE).get() + 1);
        let fresh_reach = KvSlotSpan::new(reported.span_now(inflight).get() + 1);

        assert_eq!(
            reported.blocks_needed(block, 250, 1, stale_reach, InflightSlots::NONE),
            2,
            "the defect: 2 blocks for a launch that maps 3 pages — `cannot map 513 slot(s) = 3 page(s): \
             the host granted 2 block(s)`"
        );
        assert_eq!(
            reported.blocks_needed(block, 250, 1, fresh_reach, inflight),
            3,
            "aged by the one step in flight, the allocation reaches the page the launch writes"
        );
        // ⭐ AND IT DOES NOT DEPEND ON THE REACH BEING FRESH TOO: this row's own pages are its own
        // arithmetic, not the pool-wide bound's.
        assert_eq!(
            reported.blocks_needed(block, 250, 1, stale_reach, inflight),
            3,
            "the aged span alone is enough — the reach is an upper bound, not this row's own page count"
        );

        // ⛔ THE CONTROL, AND IT IS WHY THIS ONLY EVER FIRED ON RAGGED BATCHES: give the same row a token
        // count that matches its slots (no hole) and the stale span costs nothing, because
        // `own_tokens` — the scheduler's own bookkeeping, never stale — already covers the page.
        assert_eq!(
            reported.blocks_needed(block, 513, 1, stale_reach, InflightSlots::NONE),
            3,
            "a hole-free row is sized by its tokens, so staleness in the span is invisible there — which is \
             why this read as a page-crossing bug until the ragged probes separated the two"
        );
    }
}
