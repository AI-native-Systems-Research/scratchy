// SPDX-License-Identifier: Apache-2.0
//! THE `Worker` TRAIT — the engine's view of a Spyre device.
//!
//! Scheduling, batching, cache sizing and the step loop. Target-neutral in SHAPE (metal's
//! equivalent is `gpu_worker`'s own `impl Worker`, which is larger) and model-neutral in CONTENT:
//! every model-specific decision was resolved at bake and reaches here as generated data.

// ⭐ THE POOL/SLOT VOCABULARY IS CARD-ONLY, so its imports carry the same cfg as the blocks that
// use it. `PagedKvPool` is the exception — it sizes the resident cache on both paths.
#[cfg(feature = "spyre-hw")]
use std::collections::HashMap;

use scratchy_serving_engine::executor::ModelRunnerOutput;
use scratchy_serving_scheduler::scheduler::output::SchedulerOutput;
use scratchy_subtile::sdsc_abstract::PagedKvPool;
#[cfg(feature = "spyre-hw")]
use scratchy_subtile::sdsc_abstract::{RowPages, SlotCount, SlotMap, SlotMapError};
#[cfg(feature = "spyre-hw")]
use scratchy_target_spyre::manifest::argmax;
#[cfg(feature = "spyre-hw")]
use scratchy_target_spyre::sdsc_runner::{IntRepStride, MaskRepStride};

#[cfg(feature = "spyre-hw")]
use crate::error::ExecutorError;
use crate::error::ExecutorResult;
use crate::spyre_forward::*;
#[cfg(feature = "spyre-hw")]
use crate::spyre_pool::*;
use crate::spyre_types::*;
use crate::spyre_worker::*;
use crate::worker::Worker;

/// ⛔⛔⛔ THE WIDEST BATCH THAT PRODUCES CORRECT TEXT — a CONTAINMENT for an open bug, NOT a fix for it.
///
/// The ladder bakes and loads rungs at `[2, 4, 8, 16, 32]` and every one of them RUNS. Widths above 8
/// also produce plausible, fluent, *wrong* output: rows collapse into a repeated token, and the pool's
/// throughput counters look fine while they do it.
///
/// 📊 RE-MEASURED 2026-09-17 ON CARD, FIVE INDEPENDENT TRIALS PER WIDTH, against the only gate that has
/// ever survived review here: **every row compared against its OWN `--max-num-seqs 1` output over the
/// identical probe file**. granite-3.1-2b fp8, `scr batch --no-prefix-caching`, one distinct subject per
/// row and a DISJOINT prompt prefix per row (longest shared prefix <= 2 chars, so no prefix-cache hit can
/// alias one row onto another). Rows whose text differs from their own solo text, one figure per trial:
/// ```text
///   admitted width          8            9           12               16                32
///   rung selected           8           16           16               16                32
///   short ctx, 8 tok    0,0,0,1,0   5,5,5,5,5   12,12,12,12,11   14,15,14,15,15   31,31,31,31,31
///   2-page prompt       0,1,1,0,0   5,6,6,5,5   11,10,11,8,12    14,14,16,11,16     REFUSED x5
///   ragged, 1 page      3,4,2,1,3   5,7,6,6,6   12,11,12,12,12   15,15,15,15,15   31,31,31,31,31
///   420-tok generation  1,1,1,1,1       —            —           16,15,ERR,15,ERR       —
/// ```
/// ⭐⭐ AND THE TWO KINDS OF DIVERGENCE ARE NOT ONE KIND. At width 8 a diverging row is fluent, on-subject
/// English that branched tens of characters in (`' Athens. The Eiffel Tower is in Paris, not New Orleans'`
/// against solo's `'… not in Greece'`) and the HARD oracle — is the first word this row's own capital —
/// matches solo EXACTLY in every trial of every probe. From width 9 up the diverging rows lose the answer
/// itself: `' R  (uri,  Rome.'`, `' C,  (uri,'`, `' Os.<U+FFFD>,  (1)'`. A repetition detector scores most
/// of those ZERO, which is why every earlier detector under-counted — 8-of-16 where it is really 15.
/// ⭐ ONE COUNT NEEDS NO JUDGEMENT AT ALL: the ragged probe emits 1184 tokens solo and 666-873 at width 16,
/// 2368 solo and 1919-2148 at width 32. A fifth to a third of the output vanishes into early EOS.
///
/// ⭐⭐⭐ THE BOUNDARY IS STILL THE RUNG, NOT THE LIVE COUNT. 8 live on rung 8 is clean; 9 live — which
/// promotes to the 16-row rung — already loses 5 of 9, in 5 trials out of 5. `live=16` fills rung 16 with
/// ZERO padding rows and still loses 14-16 of 16, so it is not the padding-row aliasing this file
/// documents at the cache-write site either. **Rung 16's own bundle is wrong**, and rungs <= 8 are right.
///
/// ⛔ AND IT IS A RACE, RE-CONFIRMED BY REPETITION: the COUNT is stable per width while the MEMBERSHIP
/// moves between otherwise identical trials — width 9's losers were `{0,1,4,5,8}` three times, then
/// `{1,2,3,5,6}`, then `{0,1,2,5,7}`. So it is neither a row range nor a stride nor a deterministic map.
///
/// ⛔⛔⛔ AND HERE IS THE TRAP THAT ALMOST RAISED THIS NUMBER. `scr batch --max-num-seqs 16` runs CLEAN —
/// 16 of 16 rows on their own distinct subjects, solo-diff 0,0,1 over three trials — because THIS CONSTANT
/// CAPS IT: the run logs `Capping max_num_seqs 16 → 8` and executes two sequential 8-wide batches. A clean
/// 16-REQUEST run is evidence FOR the cap, not against it.
/// ⛔ AND THE POOL LINE IS NOT THE WIDTH. `At most 16 concurrent request (launch slots…)` prints on a
/// CAPPED run too — the pool is cut from the REQUESTED `max_num_seqs`, before this cap is applied (257
/// pages for a run that admits 8), so that line cannot tell a wide run from a capped one. The two things
/// that can: the ABSENCE of the `Capping` line, and the step time — the same 16-row probe reports avg ITL
/// 65.9 ms capped to 8 against 105.9 ms genuinely 16 wide.
///
/// ⛔ AND THE THROUGHPUT IT WOULD BUY IS 8%, NOT 1.9x — AND NEGATIVE PAST ONE PAGE. Per token, from the
/// step time at each width on the same probe and binary:
/// ```text
///   1 resident page, 8 output tok    width 8: 57.0 ms/step = 7.13 ms/tok   width 16: 105.9 = 6.62  (+7.7%)
///   2 resident pages, 420 output tok width 8: 87.5 ms/step = 10.94         width 16: 196.5 = 12.28 (-12%)
/// ```
/// The 420-token case in wall clock, no model of startup at all: 3360 tokens in 45.4 s (5/5 trials) at
/// width 8 against 6720 in 92.5 s at width 16. The step nearly DOUBLES for twice the rows. The recorded
/// "bs=16 = 3.44 ms/token off a 55.0 ms step" cannot be reproduced at any context length here, and 55.0 ms
/// is within noise of the width-EIGHT step on the same short probe (57.0) — consistent with that figure
/// having been an 8-wide step divided by 16 rows.
/// ⛔ NOT the fusion knob: `SCRATCHY_SUPERDSC_GROUP_SIZE=512` against the default 128 gives avg ITL 87.5 vs
/// 87.5 ms and wall 45.5 vs 45.4 s on the 420-token probe. Bit-for-bit the same speed, so a group-size
/// difference between measurements cannot explain the gap either.
///
/// ⛔ TWO SEPARATE, LOUD DEFECTS FOUND ALONGSIDE — both fail closed, so neither is this silent one:
/// * ✅ **FIXED.** `superdsc paged: cannot map N slot(s) = P page(s): the host granted P-1 block(s)` fired
///   whenever a RAGGED batch's shared write slot landed on a page boundary — at the shipped width 8 (5 of 5
///   trials on the 1-4-page ragged probe, 1 of 3 on the 1-page one), and it killed 2 of 5 long-generation
///   trials at width 16. Every firing was `k*256 + 1` slots against exactly `k` blocks. The message blamed
///   `ReqState::kv_extent`; the report was right and the ALLOCATION was short, because the async loop
///   schedules step `n+1` while step `n` is on the card and the extent it sizes from is from step `n-1`
///   ([`scratchy_core_common::InflightSlots`]). ⛔ RAGGED-ONLY because a hole-free row's own token count
///   covers the missing page; that is why it read as a page-crossing bug for a whole round.
/// * At width 32 a 3-page prompt hits the pmask guard — `needs 96 mask block(s) (3 page(s) x 32 row(s))
///   but the rung's baked pmask holds 64` — so every request fails. The guard is correct; the width is not
///   usable for multi-page contexts even if the silent corruption were fixed.
///
/// ⛔ RULED OUT ON CARD, so nobody re-runs these:
/// * **Pool page supply.** Re-run at `SUPERDSC_POOL_PAGES=128` — 16 pages per row instead of 1 —
///   still exactly 8 degenerate. Page allocation is not the mechanism.
/// * **Per-rung pmask capacity.** The guard exists, returns `Err`, and correctly does not fire: every
///   rung allows `pages*width <= 64` and a short prompt needs one page, so 32 <= 64.
/// * **Mask pass stride.** Derived from the shared `PrefixMaskShape` law at the rung's own width, not
///   spelled a second time here.
/// * **The row-capacity wiring bug** fixed alongside this — that one made width >= 2 refuse outright
///   rather than corrupt, so it was never this.
///
/// ⛔ AND THE INSTRUMENT THE COMMENTS PROMISE DOES NOT EXIST. `SCRATCHY_KCACHE_PROBE` is described in
/// two comments (`spyre_exec.rs`, `spyre_forward.rs`) as reading back layer-0 Kᵀ per row with its block
/// table and valid columns — there is no implementation and no env check. Diagnosing this needs that
/// readback written first; guessing without it is what produced two wrong diagnoses in one session.
///
/// ▶️ THE NEXT STEP, NAMED PRECISELY. `attn.rs` already says how: *"the wide rungs are still the ones to
/// distrust first … verify by DIFFING `layoutDimOrder_` / `maxDimSizes_` / `numWkSlicesPerDim_` per
/// rung, not addresses — the previous attempt compared only start addresses, found them a strict
/// subset, and shipped complete noise."* That diff is a LOCAL check, no card needed, and it now has a
/// specific question to answer: what differs in those three fields between the rung-8 and rung-16
/// emissions of the score and value legs.
///
/// ⛔ IT MUST BE THE SHIPPED OP, THOUGH. An attempt at this diff built the score leg with
/// `MatY::of_requests`, which puts the BATCH on `y` (measured: `mq=8` gives `y=8, mb=1`), while shipped
/// attention uses `MatY::of_gqa_group` — `y` is the GQA GROUP (4 here) and `mb` carries the batch. The
/// improvised harness therefore measured a different op and was deleted rather than kept as a
/// misleading green test. Reproduce `assemble_attn`'s own construction (`of_gqa_group` plus the
/// `OperandPlacement` the shipped legs pass) and vary only `mq`.
///
/// ⭐ WHY CAPPING COSTS NOTHING MEASURABLE. Width 8 is already the throughput peak — see the wall-clock
/// comparison above, where width 16 is slower in aggregate as well as wrong. The widths this forbids were
/// never worth having: they trade correctness for no speed.
///
/// ⛔ AND CAPPING IS NOT VALIDATING. This makes the corruption UNREACHABLE instead of SILENT; it does
/// not explain it. The root cause is open, the rungs stay baked and loaded so the day it is fixed the
/// only change is this number, and nothing here should be read as evidence that width 8 is *proven* —
/// only that it is the widest width measured clean. It now IS measured against a solo oracle rather than
/// a signature detector (39 of 40 rows bit-identical to solo on the short probe, and every hard-oracle
/// answer identical), which is a stronger claim than this comment could previously make; a detector whose
/// predecessor was a literal `"(0"` match reported a fix that was not there, because the degenerate token
/// differs per run.
/// ⭐⭐⭐ STILL 8, AND THE WIDTH-8 COLUMN IS NOW MEASURED ON A PROBE THAT CROSSES A PAGE. The note above
/// asked for the per-rung descriptor diff; it was done, and it found the gather's index: one op's index
/// is ONE 128-byte stick while the pass declared `nkvh * nb * mq` entries, so above 32 the per-core IBR
/// offset wrapped. Page granularity makes an entry a REQUEST, so no rung the ladder admits can overflow
/// it.
///
/// MEASURED, `RedHatAI/granite-3.1-2b-instruct-FP8-dynamic` (hd=64, the geometry the gather runs on),
/// 420-token generations that cross a page boundary, width 8 against its OWN `--max-num-seqs 1` run of
/// the identical file, N=3:
/// ```text
///   this branch     own_ok 7/8   degen 0   1 row differs from solo   ITL 62.4-63.8 ms
///   origin/main     own_ok 7/8   degen 0   0 rows differ            ITL 87.1-88.6 ms
/// ```
/// The 8th row is `c_05_seq1185`, whose OWN SOLO degenerates on both trees — the model fails that
/// prompt, so it is excluded rather than counted. Six of the seven remaining rows are
/// character-identical to solo through all 420 characters in every trial; the seventh diverges at
/// in-page offset 178, NOT at a page boundary, and identically in all three trials.
///
/// ⛔ SO WIDTH 8 IS COHERENT, NOT BIT-EQUAL, AND THAT IS A DIFFERENT MECHANISM. A batched row is not
/// expected to equal its bs=1 output token for token while `distribute_cores` divides the projections by
/// ROW COUNT: at m=1 `q_proj` splits `out=32`, at m=8 `mb=8, out=4`. Measured with eight IDENTICAL
/// prompts — same history, mask, write slot and hole as solo — the first batched step's top-2 logit gap
/// still moves 8.156250 / 7.937500 / 7.812500 at m=1 / 2,4 / 8, deterministically per rung and
/// identically across slots. `origin/main`, with no gather in the bundle at all, shows the same residual
/// (1-2 rows at N=6). It is therefore not the gather, not the KV and not a race, and it is not closed by
/// this change.
///
/// ⛔ AND 16 IS STILL NOT CLEAN, FOR A THIRD REASON. Measured with this cap temporarily at 32, on the
/// SHORT probe (one sample per rung, which is why only the boundary and not the numbers is carried
/// forward): rungs 2/4/8 matched solo, rung 16 put 8 of 16 rows wrong at ITL 71.1 ms and rung 32 put 24
/// of 32 wrong at 297.2 ms. The crisp boundary is the PER-CORE ROW FOLD: `mq * GQA` is 32 row-slots at
/// mq=8 — exactly the core count, one slot each — and 64 at mq=16, where each core must walk TWO
/// (`zz_diff_the_rung_descriptors::doubling_the_rung_doubles_the_per_core_row_fold` measures the factor
/// going 1 → 2 at precisely that width). That is the next thing to distrust, and it is a different
/// mechanism from anything the gather touches.
///
/// ⭐ AND 16/32 ARE NOT EVEN A THROUGHPUT WIN, so the cap still forbids nothing worth having: 32 was
/// 297 ms/step against rung 8's 47 ms for four times the rows — SLOWER in aggregate as well as wrong.
#[cfg(feature = "spyre-hw")]
const CORRECT_BATCH_WIDTH: usize = 8;

/// CONTINUOUS BATCHING (#45) execute_model for the sendnn PAGED session. Consumes
/// the scheduler's batched `SchedulerOutput` like the cuda/metal gpu_worker:
///   • PREFILL (scheduled_new_reqs + cached chunks with n>1): each request's chunk
///     runs as its OWN multi-row `predict_perrow` over THAT request's scheduler
///     blocks (rows = consecutive prompt positions, causal self-attention).
///   • DECODE (cached reqs with n==1): ALL decode requests are PACKED into the mq
///     query rows of ONE `predict_perrow` (sub-batched at mq when B>mq) — the CB
///     win. Each row attends only its own request's blocks (PROVEN: per-row
///     block_table honored; rows don't cross-contaminate).
impl Worker for SpyreWorker {
    /// THE CONTEXT THE PREFIX MASK CAN ADDRESS — reported so the ENGINE rejects an over-long prompt at
    /// admission instead of the worker killing the whole batch mid-forward.
    ///
    /// 🛑 THIS IS THE HOOK I SHOULD HAVE USED FIRST, AND METAL/CUDA ALREADY DO (`gpu_worker.rs:2509`). I
    /// enforced `MAX_PAGES_PER_ROW` inside `ensure_pages`, which runs INSIDE a forward — so refusing ONE
    /// over-deep request failed the whole step with an `Executor error` and took the other three requests
    /// with it. Measured: 4 of 4 requests errored where 3 of 4 had answered. A per-request limit enforced
    /// per-BATCH is a batch-fatal limit.
    ///
    /// The engine's own doc for this method says exactly what is wanted: it caps `max_model_len` to this
    /// value "so an over-long prompt is rejected, not silently truncated by the kernel". One request gets a
    /// clean rejection; the rest of the batch is untouched.
    ///
    /// The number is the mask's, not the pool's: the prefix mask costs `pages * width^2` blocks against a
    /// baked segment (`PagedKvPool::decode_mask_bytes`), so the pages a row may hold is bounded by
    /// `MAX_PAGES_PER_ROW` even though the pool holds far more.
    fn kv_max_addressable_tokens(&self) -> Option<usize> {
        Some(PagedKvPool::MAX_PAGES_PER_ROW as usize * PagedKvPool::PAGE_SLOTS)
    }

    fn init_device(&mut self) -> ExecutorResult<()> {
        Ok(())
    }

    fn load_model(&mut self) -> ExecutorResult<()> {
        let loaded = self.load_inner()?;
        // CB CRUX PROBE (#45): env-gated standalone on-card de-risk. Runs the
        // per-row-block_table experiment against the prepared static-paged session,
        // prints YES/NO, and exits (it does NOT touch the production serving path).

        self.model = Some(loaded);
        Ok(())
    }

    fn initialize_cache(
        &mut self,
        num_gpu_blocks: usize,
        _num_cpu_blocks: usize,
    ) -> ExecutorResult<()> {
        let _ = num_gpu_blocks;
        Ok(())
    }

    /// THE KV BYTE BUDGET, as the pool was actually sized — not a constant.
    ///
    /// 🛑 THIS RETURNED A FLAT `8 GiB`, and the comment justifying it ("Host RAM; the KTIR KV cache is
    /// small (host-grown per request)") described the KTIR emulator path, not the on-card paged pool this
    /// backend has served from since. The engine sizes the scheduler's block allocator from this and then
    /// takes `num_gpu_blocks = min(that, kv_cache_num_blocks_override)` — so while the pool was ALSO
    /// sized from an 8 GiB constant the two agreed by coincidence, and the moment the pool's budget
    /// became the card's real capacity the stale constant here would have capped a larger pool at ~273
    /// pages and silently shrunk every request's context.
    ///
    /// So it reports [`Loaded::kv_budget_bytes`] — the one number the pool was cut from. The KTIR fallback
    /// keeps the old constant, because on that path there is no on-card pool and the comment above is
    /// actually true.
    fn determine_available_memory(&mut self) -> ExecutorResult<usize> {
        if let Some(b) = self.model.as_ref().and_then(|m| m.kv_budget_bytes) {
            return Ok(b as usize);
        }
        // No paged pool was sized (KTIR emulator / non-paged session): the KV lives on the host and grows
        // per request, so this really is just a host-RAM allowance.
        Ok(8usize << 30)
    }

    /// ⭐ THE POOL'S PAGE COUNT, so every block id the scheduler hands out is a page this pool has.
    ///
    /// `None` — what this returned — let the scheduler size its allocator from a memory estimate while
    /// the pool was sized from a byte budget: two counts for one page space, harmless only because the
    /// worker ignored the ids. It does not ignore them any more (`install_host_blocks`), so a block id
    /// past the pool would be an address outside the KV segment. The engine's own doc for this hook says
    /// exactly this: "the on-card pool is exactly `nblk` blocks, so the scheduler's block allocator must
    /// not exceed it".
    fn kv_cache_num_blocks_override(&self) -> Option<usize> {
        #[cfg(feature = "spyre-hw")]
        {
            // ⭐ THE HOST RANGE, NOT THE WHOLE POOL. The pages above it back the batched write's hole
            // (`PoolPartition`) and the host must never name one — that separation is the reason a page
            // the worker writes for a masked hole can never be a page the scheduler hands to another
            // request, which is the collision a shared free list made possible.
            return self.model.as_ref().and_then(|m| match &m.session {
                SendnnSession::SuperDsc(sb) if sb.decode.is_paged() => {
                    pool_partition(&sb).ok().map(|p| p.host_blocks() as usize)
                }
                _ => None,
            });
        }
        #[cfg(not(feature = "spyre-hw"))]
        None
    }

    /// THE WIDEST CONCURRENCY ONE BATCHED DECODE CAN EXPRESS — the pool's row count, which is a rung of the
    /// baked `BATCH_RUNGS` ladder by construction ([`PoolRows`]).
    ///
    /// It is a CAP and not a refusal, which is also what cuda does — see the pool-sizing site in
    /// `load_inner` for why the refusal this replaced was resting on a premise that no longer exists.
    /// A capped run stays on the batched path; without the cap the scheduler admits past the ladder and
    /// every request takes a launch of its own.
    ///
    /// ⛔ CORRECTNESS AT A GIVEN WIDTH IS THE GATE'S QUESTION, NOT THIS COMMENT'S:
    /// Only a batch gate can judge a batch — one that screens every probe SOLO first, uses ragged
    /// lengths, and diffs caching-on/off/reversed. An ad-hoc concurrent probe cannot, which is how
    /// uniform-length prose continuations once got read as corruption.
    fn max_num_seqs_override(&self) -> Option<usize> {
        #[cfg(feature = "spyre-hw")]
        {
            return self.model.as_ref().and_then(|m| match &m.session {
                SendnnSession::SuperDsc(sb) if sb.decode.is_paged() => {
                    Some((sb.pool_rows.get().get() as usize).min(CORRECT_BATCH_WIDTH))
                }
                _ => None,
            });
        }
        #[cfg(not(feature = "spyre-hw"))]
        None
    }

    /// ⭐⭐⭐ THIS BACKEND APPENDS EVERY ROW AT ONE SHARED SLOT — declared, so the scheduler allocates for it
    /// from the FIRST step rather than from the first report.
    ///
    /// Only on the PAGED superdsc path: that is the one whose batched step collapses the request axis into a
    /// single launch writing every row at the same slot. Every other session here is token-addressed, and
    /// saying otherwise would make the scheduler over-allocate for no reason.
    fn kv_addressing(&self) -> scratchy_core_common::KvAddressing {
        #[cfg(feature = "spyre-hw")]
        {
            if let Some(m) = self.model.as_ref()
                && matches!(&m.session, SendnnSession::SuperDsc(sb) if sb.decode.is_paged())
            {
                return scratchy_core_common::KvAddressing::OneSharedWriteSlot;
            }
        }
        scratchy_core_common::KvAddressing::ByToken
    }

    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
        let Self {
            model,
            requests,
            input_batch,
            ..
        } = self;
        let model = model.as_mut().ok_or_else(|| werr("model not loaded"))?;
        // ONE derivation of the pool's owner split for the whole step: the same value the block count
        // reported to the scheduler came from, so a derived page map can never name a page the host may
        // also hand out.
        #[cfg(feature = "spyre-hw")]
        let part = {
            let SendnnSession::SuperDsc(sb) = &model.session;
            pool_partition(&sb)?
        };

        // ⛔⛔⛔ FINISHED FIRST, AND THE ORDER IS A CAPACITY INVARIANT, NOT TIDINESS.
        //
        // This ran at the END of the step. A step that FINISHES one request and ADMITS another then needed
        // both KV rows at once, because dropping the request IS the release of its row — so a run at
        // `--max-num-seqs N` could hold N+1 rows for the length of one step. That was invisible while the
        // pool was cut for the widest baked rung (32 rows for any N); the moment the reserve was charged
        // against the admitted width (`9d65317a`, 4 rows at `--max-num-seqs 4`) it refused 5 of 6 requests
        // on the ragged gate:
        //     superdsc paged: every one of the 4 KV row(s) is held by a live request
        // ⇒ A capacity change did not cause this; it EXPOSED an off-by-one the oversized pool had been
        // paying for. And the probe that caught it is the one with MORE REQUESTS THAN ROWS — the clean
        // 4-probe (4 requests, 4 rows, never 5 live) passes either way.
        //
        // ⭐ THE COMMON BACKEND ALREADY HAD THE ORDER RIGHT: `gpu_worker.rs` is literally
        // "── 1. Lifecycle: drop finished requests" then "── 2. Lifecycle: add newly scheduled requests".
        // Third time this session that the metal path had the answer written down.
        //
        // `finished_req_ids` are requests that finished in the PREVIOUS step, which is why cleaning them up
        // before this step's admissions is correct and not a race.
        for fid in &scheduler_output.finished_req_ids {
            // Dropping the request IS the release of its KV ROW: `rows_held` reads the live requests, so
            // nothing has to be handed back here and no path can forget to. Its PAGES are the scheduler's
            // to free — it refcounts them, which is what keeps a cached prefix alive after the request
            // that wrote it is gone.
            requests.remove(fid);
            #[cfg(feature = "spyre-hw")]
            input_batch.remove_request(fid);
        }

        let mut order: Vec<String> = Vec::new();
        let mut sampled: Vec<Option<u32>> = Vec::new();

        // New requests = (chunked) prefill: register the prompt + empty KV cache.
        for nr in &scheduler_output.scheduled_new_reqs {
            let n = scheduler_output
                .num_scheduled_tokens
                .get(&nr.req_id)
                .copied()
                .unwrap_or(0);
            let start = nr.num_computed_tokens as usize;
            let prompt = nr.prompt_token_ids.clone().unwrap_or_default();
            #[cfg(not(feature = "spyre-hw"))]
            let nlayers = model.decode[0].layers.len();
            requests.insert(
                nr.req_id.clone(),
                ReqState {
                    // A request enters holding `start` contiguous slots (0 for a cold prompt, the
                    // prefix-cache hit otherwise): it has not shared a step with anyone yet, so its
                    // slots and its tokens are still the same set.
                    kv_hist: scratchy_subtile::sdsc_abstract::KvHistory::contiguous(start),
                    #[cfg(not(feature = "spyre-hw"))]
                    kv_k: vec![Vec::new(); nlayers],
                    #[cfg(not(feature = "spyre-hw"))]
                    kv_v: vec![Vec::new(); nlayers],
                },
            );
            // ⭐ THE HOST'S TABLE GOES INTO `InputBatch`, the engine's own per-request store — including,
            // on a prefix-cache hit, the blocks that already hold this request's first `start` tokens.
            // ⛔ ONE KV CACHE GROUP: `block_ids` is per group and every group of a non-hybrid model gets
            // the same ids ("In the common single-group case, all groups share the same block IDs"), so
            // group 0 is the list. A hybrid model would need a group per attention class, which this
            // backend does not bake.
            input_batch.add_request(
                nr.req_id.clone(),
                // ⛔ NO CHUNK. This argument is "the tokens to feed this step" and it drives
                // `prepare_inputs`, which is the cuda/metal step builder — this worker builds its own
                // chunks from the prompt below, so handing it one here would populate a field nothing on
                // this path reads. The PROMPT goes in through `set_prompt`, which is a different thing.
                &[],
                nr.block_ids.first().cloned().unwrap_or_default(),
                nr.num_computed_tokens,
            );
            // ⭐ AND THE TOKENS THEMSELVES, in the same store as the block table. `ReqState` used to hold
            // `tokens` + `prompt_len`; the length is now derived from this one value.
            input_batch.set_prompt(&nr.req_id, &prompt);
            // Only the paged (card) install below reads it.
            #[cfg(feature = "spyre-hw")]
            let host_now: Vec<usize> = input_batch
                .block_table(&nr.req_id)
                .map(|t| t.to_vec())
                .unwrap_or_default();
            let req = requests.get_mut(&nr.req_id).expect("just inserted");
            // PAGED: install the blocks the SCHEDULER allocated for this request — including, on a
            // prefix-cache hit, the blocks that already hold its first `start` tokens' keys. That
            // sharing is the whole point: the request does not recompute them and does not copy them,
            // because its block table names the same physical pages the earlier request wrote.
            //
            // ⛔ ONE KV CACHE GROUP. `block_ids` is per group and every group of a non-hybrid model
            // gets the same ids (`kv_cache_manager`: "In the common single-group case, all groups share
            // the same block IDs"), so group 0 is the list. A hybrid model would need a group per
            // attention class, which this backend does not bake.
            #[cfg(feature = "spyre-hw")]
            install_host_blocks(
                &mut model.session,
                req,
                req.kv_span_slots(n as u32),
                PageMapCtx {
                    host: host_now.as_slice(),
                    part,
                },
            )?;
            let s = run_request_step(
                model,
                req,
                &nr.req_id,
                input_batch,
                start,
                n,
                #[cfg(feature = "spyre-hw")]
                host_now.as_slice(),
                #[cfg(feature = "spyre-hw")]
                part,
            )?;
            order.push(nr.req_id.clone());
            sampled.push(s);
        }

        // Cached requests = decode (n=1) or a prefill-continuation chunk.
        let cr = &scheduler_output.scheduled_cached_reqs;
        // ⭐ INSTALL THE STEP'S BLOCK GRANT FOR EVERY CACHED REQUEST, BEFORE ANY OF THEM FORWARD, and
        // in the scheduler's own indexing — `new_block_ids[i]` belongs to `req_ids[i]`, and a resumed
        // request's list REPLACES its old one rather than extending it.
        //
        // Once per request per step, ahead of the batched/per-request split, because the grant is a fact
        // about the STEP and not about which path a request takes through it. The batched path then only
        // has to check that the grant reaches the shared write slot — it can no longer draw a page to
        // make that true, which is what let the worker's idea of a request's pages diverge from the
        // scheduler's.
        #[cfg(feature = "spyre-hw")]
        for (i, id) in cr.req_ids.iter().enumerate() {
            let ids: &[usize] = cr
                .new_block_ids
                .get(i)
                .and_then(|g| g.as_ref())
                .and_then(|groups| groups.first())
                .map_or(&[], |g| g.as_slice());
            let resumed = cr.resumed_req_ids.contains(id);
            // ⭐ INTO THE ONE STORE, REPLACING. `InputBatch::update_blocks` is the backend-neutral
            // per-request block table cuda and metal already use, and its own doc carries the rule: the
            // scheduler ships the FULL table every step, so this REPLACES rather than appends. That is
            // also exactly what a RESUMED request needs — its old blocks were freed and re-allocated.
            if !ids.is_empty() {
                input_batch.update_blocks(id, ids.to_vec());
            }
            // This request's table, read back RIGHT AFTER its own update — the fresh value, not a
            // snapshot taken before the loop (which is what made the worker refuse a page the scheduler
            // had already granted).
            let host_now: Vec<usize> = input_batch
                .block_table(id)
                .map(|t| t.to_vec())
                .unwrap_or_default();
            let computed = cr.num_computed_tokens.get(i).copied().unwrap_or(0) as usize;
            let Some(req) = requests.get_mut(id) else {
                continue;
            };
            // A RESUME IS A RESET: the blocks this request held were freed and re-allocated, so its old
            // history describes keys it no longer owns.
            if resumed {
                // BOTH HALVES, ADJACENTLY: the slots this request owns (its `KvHistory`) and the token
                // count the store holds. A resume that reset one and not the other would leave a request
                // attending slots it no longer owns, or asking for a token it has not been given.
                req.reset_for_resume(computed);
                input_batch.set_tokens_in_pool(id, computed);
            }
            let span = req.kv_span_slots(0);
            install_host_blocks(
                &mut model.session,
                req,
                span,
                PageMapCtx {
                    host: host_now.as_slice(),
                    part,
                },
            )?;
        }
        // ⭐ THE HOST'S TABLES FOR THIS STEP, READ OUT OF `InputBatch` ONCE — **AFTER** THE LOOP ABOVE HAS
        // WRITTEN THIS STEP'S GRANT INTO IT.
        //
        // 🛑 IT USED TO BE READ BEFORE THAT LOOP, WHICH MADE IT STALE BY EXACTLY ONE STEP. Measured on a
        // 2599-token prompt (one request, prefix caching OFF): the step that finishes the prompt needs 11
        // pages, the scheduler had allocated 11, and the worker refused with `cannot map 11 page(s) … the
        // host granted 8 block(s)` — 8 being the PREVIOUS step's table. The step then produced no token and
        // the next one asked for a token that did not exist (`token index 2599 out of range`).
        //
        // ⛔ THIS IS THE "SECOND STORE" HAZARD IN ITS SMALLEST FORM: a snapshot of a store is a copy, and a
        // copy read before the store is updated is a different value. Reading it AFTER is what makes it a
        // read rather than a store — it must be taken downstream of every `update_blocks` for the step.
        #[cfg(feature = "spyre-hw")]
        let hosts: HashMap<String, Vec<usize>> = cr
            .req_ids
            .iter()
            .chain(
                scheduler_output
                    .scheduled_new_reqs
                    .iter()
                    .map(|nr| &nr.req_id),
            )
            .filter_map(|id| {
                input_batch
                    .block_table(id)
                    .map(|t| (id.clone(), t.to_vec()))
            })
            .collect();
        // ── BATCHED DECODE ────────────────────────────────────────────────────────────────────────
        // Every cached request scheduled for exactly ONE token is a decode step, and those are the
        // whole reason the weight set gets streamed over and over: each used to be its own forward.
        // Run them together on the smallest rung that holds them and the weights move once.
        //
        // Through `run_prefill_batch` — the SAME path a prompt chunk takes, at B rows instead of
        // N. Decode is not a separate implementation of attention; what a batch needs and a prompt
        // does not is a page map per request and a validity row per request, and both are already
        // carried by the bundle. What the caller supplies is the new-block law: a prompt's rows are
        // consecutive positions and attend each other causally, a batch's rows are separate requests
        // and must attend only themselves.
        //
        // A continuation chunk (n > 1) is not a decode step and keeps the per-request path below, as
        // does every request when no ladder was baked or the live count exceeds the widest rung.
        #[cfg(feature = "spyre-hw")]
        let mut batched: std::collections::HashSet<String> = std::collections::HashSet::new();
        #[cfg(feature = "spyre-hw")]
        {
            let decode_ids: Vec<(String, usize)> = cr
                .req_ids
                .iter()
                .enumerate()
                .filter(|(i, id)| {
                    scheduler_output
                        .num_scheduled_tokens
                        .get(*id)
                        .copied()
                        .unwrap_or(0)
                        == 1
                        && cr.num_computed_tokens.get(*i).is_some()
                })
                .map(|(i, id)| (id.clone(), cr.num_computed_tokens[i] as usize))
                .collect();
            // ORDER IS BY KV ROW, NOT BY LENGTH — and the row a request occupies in the launch is the
            // row it owns for its whole life (`ReqState::row`). The sort therefore happens AFTER
            // `ensure_pages`, which is what claims the row.
            //
            // WAS descending length, so that `starts[0]` would be the maximum resident length: the
            // runtime reads the rung and the number of pages to fold out of the `seq_pos` the forward
            // is launched with, and taking that from an arbitrary row sweeps only that row's history
            // while a longer row silently stops attending its older pages. That requirement has not
            // gone away; it is now met by passing the maximum explicitly instead of by arranging for
            // row 0 to hold it. Which is strictly better: the launch position stops being a fact about
            // whichever request happened to sort first.
            //
            // The reason it has to change is that a per-step order makes a row a per-step position, and
            // then a request's pages move whenever another request finishes. One launch can address B
            // rows only if row r's KV base is `base + r*stride`, so the row has to be an identity.
            let decode_ids = decode_ids;
            // ⛔⛔⛔ THE "CONSECUTIVE ROWS" REQUIREMENT IS GONE, AND WITH IT THE ONE-AT-A-TIME FALLBACK.
            //
            // This used to read every live request's pool row, prove with `AffineRows::of` that the set was
            // `base..base+live`, and fall back to decoding ONE REQUEST AT A TIME when it was not — because a
            // fused cache write with the request baked into the block addressed pool row `kv_rows[0] + i`, so
            // a request finishing in the MIDDLE left a gap that made slot `i` write somebody else's row.
            //
            // `kv_rows` left the device layer at `5e319159` (the fused write takes a launch slot and a page
            // list), and the per-request row that survived host-side existed only to name the reserved pages
            // behind a batched write. The host allocates that page now, so there is no row to be consecutive,
            // no witness to construct, and no fragmented-pool case to degrade for. A batch of any composition
            // runs on one rung.
            let rung = if decode_ids.len() > 1 {
                {
                    // The count the rung is selected FROM is the LIVE one; everything the launch
                    // then binds is sized by the rung's own baked width.
                    let SendnnSession::SuperDsc(sb) = &model.session;
                    sb.decode_rung_for(scratchy_subtile::sdsc_abstract::LiveRows::of_scheduled(
                        decode_ids.len(),
                    ))
                }
            } else {
                None
            };
            if let Some(ri) = rung {
                // Grow pages BEFORE the forward, for every request: one crossing into a new page
                // needs it bound when its own row writes, not after.
                //
                // TO THE SHARED WRITE SLOT, not to each request's own length: the batch appends at one
                // slot for all of them, so a short request needs the page that slot lands in even
                // though its own keys stop earlier.
                //
                // ⛔ OVER EVERY LIVE REQUEST, NOT JUST THE ONES DECODING THIS STEP — which is what makes
                // the collapsed cache write SAFE, not merely faster. A rung is the smallest baked width
                // that HOLDS the decode count, so a step with 5 decoding requests runs the 8-row rung
                // and rows 5..7 are PADDING: they replicate the last real token and, with the request
                // baked into the block, they write it into `base + p` — a pool row a live request that
                // is still PREFILLING holds. Taking the slot past every live history puts that stray
                // write in the victim's masked HOLE: it will never write slot V itself (its next slot is
                // a later maximum), so `KvHistory::contains` is false there forever and the key is never
                // read. Over the decoding requests alone, V could land inside a longer prefilling
                // request's real prompt, and the corruption is a whole block of requests at once —
                // MEASURED, five of eight garbled in one trial of four.
                //
                // The pages are not per-request space: every row of a page holds a different request,
                // so this grows the pool by pages the longest request needed anyway.
                let batch_slot = scratchy_subtile::sdsc_abstract::BatchSlot::of(
                    requests.values().map(|r| &r.kv_hist),
                );
                for (id, _) in &decode_ids {
                    let req = requests
                        .get_mut(id)
                        .ok_or_else(|| werr(format!("unknown cached req {id}")))?;
                    // ⛔ A CHECK NOW, NOT A GROWTH. This used to draw pages from the worker's own free
                    // list until every row reached the shared write slot; the grant was installed above
                    // and the only question left is whether it REACHES that slot. It does when the
                    // scheduler sized the request's blocks from the span this worker reported
                    // (`ReqState::kv_extent`), and the refusal names the two disagreeing when it does not.
                    install_host_blocks(
                        &mut model.session,
                        req,
                        // ONE PAST THE SHARED WRITE SLOT: what this launch will have occupied once every
                        // row has written its token there.
                        SlotCount::new(batch_slot.get().saturating_add(1)),
                        PageMapCtx {
                            host: hosts.get(id).map_or(&[], |v| v.as_slice()),
                            part,
                        },
                    )?;
                }
                // ⛔ THE BATCH IS NO LONGER SORTED INTO ROW ORDER, BECAUSE THERE IS NO ROW ORDER. This sorted
                // `decode_ids` by pool row so that launch slot `i` lined up with row `base + i` for the fused
                // cache write. Each launch slot now binds its OWN block table (`set_block_table(slot, write
                // slot, pages)`), so the launch order carries no meaning about where KV lives — which is the
                // property the gate's permutation axis has been asserting all along.
                let sh = Shared {
                    embed_tokens: &model.embed_tokens,
                    hidden: model.hidden,
                    head_dim: model.head_dim,
                    kv_dim: model.kv_dim,
                    vocab: model.vocab,
                    rope_theta: model.rope_theta,
                };
                let SendnnSession::SuperDsc(sb) = &mut model.session;
                // The pool's owner split, once for this launch — the same derivation the host's block
                // count came from, so a derived page map cannot name a block the host may also hand out.
                let part = pool_partition(&sb)?;
                let DecodeRung {
                    seqs,
                    mask_cap,
                    fold_rows,
                    sess,
                    logits,
                } = &mut sb.decode_rungs[ri];
                let seqs = *seqs;
                let fold_rows = *fold_rows;
                // ⛔⛔⛔ "DOES THIS BUNDLE GATHER" IS NOT ASKED HERE ANY MORE, AND THAT IS THE FIX.
                // It was `let gathers_kv = *gathers_kv` off `DecodeRung` — which was already the
                // SECOND home for this question (the first was `model.decode.last()`, the m=1 bucket,
                // which places no index and answered `false` for every rung that does: `0xa35e
                // RAS::PCI::BusFence` from two gather copies resolving `idx * skip_addr + base` over a
                // tensor no step staged, a fence with no `vars` raised on senlib's own monitor thread).
                //
                // Both homes answered about a BUNDLE, and the question is about a BODY: a rung's bundle
                // holds a whole sk_bucket ladder, and which body runs is the selector's answer from the
                // live context length. So it is asked BELOW — once, of the session, after the page maps
                // are installed (the selector reads `fold_requests` off them) and from the same write
                // slot the launch is given. See `SuperDscSession::step_body`.
                // The pool geometry the entry factor needs. `head_dim` decides the entry SIZE
                // (`hd * 64` elements), so this is not interchangeable with any other pool value.
                let pool = scratchy_subtile::sdsc_abstract::PagedKvPool::new(
                    model.kv_dim / model.head_dim,
                    model.head_dim,
                );
                // WHICH RUNG IS ACTUALLY RUNNING, and how much of it is padding. A rung that failed
                // to build at startup is logged and skipped, and the selection then lands on the next
                // WIDER one — 8 live requests running the 16-row bundle is twice the rows for the
                // same work, which no change to the bundle itself would ever show up as. Reported
                // once per (live, rung) pair so it names the mismatch without repeating every step.
                {
                    use std::sync::OnceLock;
                    static SEEN: OnceLock<
                        std::sync::Mutex<std::collections::HashSet<(usize, usize)>>,
                    > = OnceLock::new();
                    let seen = SEEN
                        .get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
                    if let Ok(mut g) = seen.lock() {
                        if g.insert((decode_ids.len(), seqs.count())) {
                            // The one place the two widths are COMPARED on purpose: reporting how
                            // much of the rung is padding.
                            if seqs.count() == decode_ids.len() {
                                // ⛔ `debug!`, NOT `eprintln!`. This fires on EVERY DECODE STEP —
                                // once per generated token, straight to stderr, where no
                                // `RUST_LOG` can turn it off and it interleaves untimestamped with
                                // the tracing output anyone is actually reading.
                                tracing::debug!(
                                    "[spyre-worker] decode batch: {} live on the {}-row rung (exact)",
                                    decode_ids.len(),
                                    seqs.count()
                                );
                            } else {
                                tracing::debug!(
                                    "[spyre-worker] decode batch: {} live but running the {}-row rung \
                                     — {} padded row(s), {:.0}% of the work is padding. The exact rung is \
                                     missing; look for 'decode batch rung seqs={} ... SKIPPED' at startup.",
                                    decode_ids.len(),
                                    seqs.count(),
                                    seqs.count() - decode_ids.len(),
                                    100.0 * (seqs.count() - decode_ids.len()) as f64
                                        / seqs.count() as f64,
                                    decode_ids.len()
                                );
                            }
                        }
                    }
                }
                let logits = &*logits;
                // THE MASK'S PER-FOLD-PASS STRIDE. A pass reads one request's page, so the mask is
                // one `[nqh*seqs, page_slots]` block per pass and the runtime steps it by a whole
                // block. Declared here because this is where the row count and the head count are
                // both known; idempotent, so restating it each step costs nothing and cannot drift
                // from the rung actually running.
                let nqh = sh.hidden / sh.head_dim.max(1);
                // ROW-BATCHED FOLD (emitter: `per_request` at mq>1 with rows_are_requests). A pass is
                // baked for ONE request's `nqh` rows, so a mask BLOCK is `nqh` rows deep, not
                // `nqh*seqs`, and each pass must additionally be rebased onto its own request's row
                // block in the intermediate segment. Both numbers are the worker's to declare: the
                // runtime cannot see what row count the bundle was baked at.
                //
                // A batch of ONE runs the unbatched bundle, where the fold spans the (single) request
                // and neither shift applies — 0 restores exactly the pre-batching behaviour.
                // ⭐ FROM THE SHARED SHAPE. A rep step is ONE BLOCK by definition, so taking it from
                // `PrefixMaskShape` is the same law the emitter reads its slabs with and the staging
                // writes its bytes with — one derivation, not a third one spelled out here. Declaring
                // it by hand is how the stride and the layout drift, and a mask that steps wrong reads
                // another pass's validity rows.
                let mask_shape = {
                    use scratchy_subtile::sdsc_abstract::PrefixMaskShape;
                    const PER_PAGE: u32 =
                        scratchy_subtile::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32;
                    // The RUNG's width, not the live count: the emitter baked `nqh * seqs` mask
                    // rows and the fold steps by that block whatever is live.
                    PrefixMaskShape::<{ scratchy_subtile::sdsc_abstract::POOL_STICK }, PER_PAGE>::new(nqh as u32, seqs)
                    .ok_or_else(|| werr("decode batch: prefix mask has zero heads".to_string()))?
                };
                sess.set_mask_stride(MaskRepStride::of(mask_shape))
                    .map_err(|e| werr(format!("decode batch: set_mask_stride: {e}")))?;

                // THE INTERMEDIATE-SEGMENT REBASE STRIDE — the distance between two requests' row
                // blocks in every intermediate the fold touches (`qs`, `sc`, the softmax state,
                // `out`) — DERIVED from the regime this rung's own bundle declares, not decided
                // here. The stride is a fact about how the fold's passes were BAKED (`attn.rs`'s
                // `BlockRows`, riding the manifest as `kv_fold_rows_per_request`): a whole-batch
                // bundle's every pass sweeps all `nqh*mq` shared-buffer rows and the mask silences
                // the rows the pass does not own, so NO pass may be rebased and the stride is 0 —
                // regardless of whether the launch also collapses to one pass per page
                // (`fold_plan::collapsed` = a request axis on the op AND an affine pool). A
                // per-request bundle's passes carry one request's `nqh` rows each and must be
                // rebased onto their own block.
                //
                // ⛔ THE CONSTANT THIS DERIVATION REPLACES hand-mirrored the emitter's `per_req`
                // across two crates and justified its value as "the fold really is COLLAPSED here,
                // tested on card". The VALUE was right, the reasoning was not: these bundles
                // declare no request axis (`kv_batched_requests` false), so `collapsed` is false
                // and the fold runs `pages × requests` passes — each of which STILL spans the whole
                // batch's rows. The card test (stride = nqh*64*2 baked onto this bundle → word
                // salad on previously-working rows while the deep row still EOS'd) is exactly what
                // rebasing whole-batch passes does; it says nothing about collapse.
                sess.set_int_stride(IntRepStride::of(fold_rows, nqh as u32))
                    .map_err(|e| werr(format!("decode batch: set_int_stride: {e}")))?;
                // WHICH LIVE REQUEST EACH LAUNCH SLOT HOLDS — DENSELY: the live list is already in KV-row
                // order, and live `i` runs in slot `i`.
                //
                // 🛑 THE SLOT IS NOT THE KV ROW. Binding it that way is a bug that shipped, and it is
                // the one this whole `SlotMap` exists to make unexpressible: the rung is chosen from how
                // many requests are LIVE (`decode_rung_for`), so putting a request in the slot numbered
                // after its row makes the launch width depend on WHICH rows they hold instead — four
                // requests holding rows 4..7 select the four-wide rung and then cannot be addressed in
                // it. It surfaced as an intermittent hard failure at bs=8, only in the runs where a
                // narrower rung ran first, which is why it took a proof over symbolic rows to pin.
                //
                // Nothing requires them to be equal: a slot's KV comes from the block table installed AT
                // that slot (`fold_plan::page_base_bytes` looks up `block_tables[slot][page]`), so row
                // 7's pages bound at slot 0 address row 7's KV exactly. The row still has to be an
                // IDENTITY — a request's pages must not move when the batch's composition changes — and
                // it is, because `PoolSplit` cut the pool for the widest rung and `ensure_pages` draws
                // only from the request's own run. That is the property; equal numbering was never it.
                //
                // Slots past the live count are padding and reuse live 0's map, so no slot addresses an
                // uninstalled table. That re-writes one token at one slot a second time with the same
                // key and value — idempotent, which is why padding can point at real pages instead of
                // somewhere invented for it.
                // The launch is as wide as the RUNG — a zero-row rung is unrepresentable, so the
                // old runtime refusal for it has no arm left.
                let width = seqs;
                let slots = SlotMap::of_live(decode_ids.len(), width).map_err(|e| {
                    werr(match e {
                        SlotMapError::NoLiveRequest => {
                            "decode batch: no live request to lay into the launch".to_string()
                        }
                        SlotMapError::TooWide { live, width } => format!(
                            "decode batch: {live} live request(s) into a {width}-row rung. Rung \
                             selection takes the smallest rung that holds the live count, so this \
                             means no baked rung is wide enough — bake one, or admit fewer"
                        ),
                    })
                })?;
                // SLOT -> REQUEST, said out loud once per step. ⭐ THE INSTRUMENT THAT SEPARATES A SLOT
                // DEFECT FROM A REQUEST DEFECT, and it is the one this hunt lacked: at hd=128 width 4,
                // EXACTLY 2 of 4 requests collapse every trial but WHICH 2 varies
                // ({0,1}, {0,3}, {0,2}, {2,3}). A fixed COUNT with a varying IDENTITY is what a per-SLOT
                // defect looks like once admission order decides who sits where — and without this line the
                // two readings are indistinguishable, which is how four separate mechanisms got proposed for
                // it from single trials. `SLOT_TRACE` because a per-step line is too much for a normal run.
                // ONE WRITE SLOT FOR THE WHOLE LAUNCH. Every request's page map is bound at the same
                // slot, because a launch resolves exactly one slot shift for every trip inside it: per
                // request slots are per-request LAUNCHES, and a launch costs ~93 us of device time
                // whatever it carries — 8 x 40 of them per step is about 26 ms of a 111 ms bs=8 step.
                //
                // The slot is past every live request's own keys (`BatchSlot::of`), so no request
                // overwrites its own history, and the shorter ones gain a masked hole rather than a
                // wrong write. The hole is not wasted pool space: the pages spanning it are resident for
                // the batch's longest request anyway and the request dimension lives INSIDE the page, so
                // a short request is not paying for pages it skipped — only for masked columns, in a
                // mask that is already materialised at full page width every step.
                // THE SAME SLOT `ensure_pages` ABOVE SIZED THE POOL FOR, over the same set: every LIVE
                // request, decoding or not, so a padded row's write lands in a hole nobody reads. See
                // that call for why the set is all of them and not just this step's decoders.
                let write_slot = scratchy_subtile::sdsc_abstract::BatchSlot::of(
                    requests.values().map(|r| &r.kv_hist),
                );
                let mut max_pages = 1usize;
                // ⭐⭐⭐⭐⭐ THE GATHER'S INDEX TABLE, COLLECTED IN THE SAME WALK THAT INSTALLS THE PAGE
                // MAPS — because it is the same data, and any second walk is a second chance to
                // disagree about which page a row holds.
                //
                // ⛔ THE ROW ORDER IS THE SLOT ORDER, NOT THE REQUEST ORDER, for exactly the reason
                // the `set_block_table` note below gives: slot `i` IS row `row0 + i` for the fold, and
                // that includes the padding slots, whose map is live 0's. A table keyed by request
                // would send every row past the first to another row's pages.
                let mut page_maps: Vec<Vec<i64>> = Vec::with_capacity(width.count());
                // ⛔ THE AFFINE-ROWS WITNESS WAS UNWRAPPED HERE, and the refusal it produced
                // ("batched path reached without the affine-rows witness") had no way to fire — the rung was
                // `Some` only when the witness was. It is all gone with the rows themselves.
                for (slot, src) in slots.slots() {
                    // WHOSE PAGE MAP THIS SLOT INSTALLS. A live slot installs its own request's; a
                    // padding slot installs LIVE 0's: every slot's table must exist (the collapsed
                    // cachewr resolves a table for the whole rung, so slot `i` resolves one whether
                    // or not anyone lives there), and the pool's pages are one shared set per row
                    // anyway.
                    // ⛔ THE BORROW IS NOT FREE. The on-card paged cachewr address has NO row term
                    // inside a page — a pool cell is (plane, kv-head, slot-in-page, feature) — so
                    // the padding slot's one cache write lands on the SAME physical cell as live
                    // 0's, and the shim elides the pipeline barrier between consecutive slot-write
                    // groups on the premise the writes are disjoint: the two writes are UNORDERED.
                    // That is why the padding row's WHOLE FORWARD is live 0's — token, rotation,
                    // new-block column and prefix history (`LaunchRows`, laid in `run_prefill_batch`
                    // from the live rows below). Identical bytes are the one payload that commutes,
                    // so live 0's newest key survives whichever write lands last; and the bytes are
                    // the forward's OUTPUT, so a row that matched live 0's input but read a different
                    // key set would still write different K/V from layer 1 on.
                    let owner = match src.live() {
                        Some(i) => i,
                        None => 0,
                    };
                    let (id, _start) = &decode_ids[owner];
                    // ⭐ THE ONE BIT THE PAGE TRACE COULD NOT SAY: is this slot LIVE or PADDING, and whose
                    // request is it serving? Without it, "row 3 was bound with row 0's pages" is either the
                    // harmless padding page-map borrow or a live request reading another's history, and those
                    // two have the same page-trace line. Measured 2026-08-11: rows 1 and 3 of the ragged probe
                    // both emit ROW 0's continuation, and row 3's table at the first batched step IS row 0's.
                    let req = requests
                        .get(id)
                        .ok_or_else(|| werr(format!("unknown cached req {id}")))?;
                    // ⭐ SLOT `i` IS ROW `row0 + i`, INCLUDING THE PADDING SLOTS. That is not a choice
                    // here, it is what the baked addressing does for the FOLD: slot `i`'s pass reads
                    // `row0 + i`. Binding a padding slot to live 0's ROW instead (which is what
                    // `SlotMap` points its map at) made the installed rows read `row0, row0+1, …, row0`
                    // — no single stride, so `LaunchPages::of` refused `Affine` and the fold silently
                    // kept its `× requests` passes, with "not faster" as the only symptom.
                    //
                    // Sound because the rows ARE consecutive (checked before the rung was chosen). The
                    // CACHE WRITE is the one place the row term does NOT reach — the paged cachewr's
                    // pool cell is (plane, kv-head, slot-in-page, feature), no row inside a page — so
                    // the padding slot's write shares live 0's cell and must carry live 0's bytes (the
                    // write mirror above). The PAGES stay live 0's — every row of the pool has the
                    // same pages, so there is nothing per-request left in them.
                    // THE MAP, DERIVED FOR THIS LAUNCH: the host's blocks at the logical pages this
                    // request holds keys in, every fully-masked page aliased to the pool's scratch page,
                    // and the page the SHARED write slot lands in backed by this row's reserved hole page
                    // when the host's blocks stop short of it — which is the ordinary case for a row
                    // shorter than its batch-mates, and what the host cannot allocate for because the slot
                    // depends on the deepest LIVE request.
                    let want =
                        RowPages::holding(SlotCount::new(write_slot.get().saturating_add(1)));
                    let map = req.page_map(
                        PageMapCtx { host: hosts.get(id).map_or(&[], |v| v.as_slice()), part },
                        want,
                    ).ok_or_else(|| {
                        werr(format!(
                            "superdsc paged: request {id} holds keys in more pages than the {} host \
                             block(s) it was granted — the scheduler allocates its own pages plus ONE for \
                             the batch's shared write page, so that allocation and this launch disagree",
                            hosts.get(id).map_or(0, |v| v.len()),
                        ))
                    })?;
                    bind_launch_slot(sess, slot, write_slot, &map)?;
                    max_pages = max_pages.max(map.held().get() as usize);
                    // The SAME `BlockTable` the launch just installed — `as_i64()` is the one untyping
                    // in the path, and this reads it at the same call site rather than re-deriving the
                    // map from `req.page_map` a second time.
                    page_maps.push(map.as_i64());
                }
                // ⭐⭐⭐⭐⭐ **WHICH BODY THIS STEP WILL RUN**, and the two facts that are its alone. Asked
                // of the session, from the write slot — and the launch below is handed that same slot as
                // `start`, so the body staged for and the body launched are one body by construction.
                //
                // ⛔ IT IS ASKED HERE AND NOT EARLIER: the selector's fused/split decision turns on
                // `fold_requests`, which the runtime INFERS from the page maps the loop above just
                // installed. Asked before them it would answer for an unbatched step and hand back the
                // fold-FUSED body, whose fold runs once — the (request 0, page 0) pass — leaving every
                // other request attending no resident prefix.
                //
                // ⛔ AND THE TWO FACTS ARE PER-BODY, WHICH IS WHY THEY NO LONGER LIVE ON `DecodeRung`.
                // `swept` was the manifest's — one number per BATCH WIDTH, i.e. the CEILING body's,
                // while the launched body is whichever ladder rung the live context picked (`nb ∈
                // {1,2,4}`); `gathers_kv` was the bundle's `KV_BLOCK_INDEX_TID` placement, which is
                // present if ANY body of the bundle gathers. Three defects of this exact shape have now
                // been fixed here, so the fields are DELETED and this is the only door.
                let launch_pos = write_slot.get() as usize;
                let step = sess.step_body(launch_pos).map_err(|e| {
                    werr(format!(
                        "decode batch: the session could not say which body this step runs: {e}"
                    ))
                })?;
                let swept = step.swept();
                let gathers_kv = step.gathers();
                let bake_gathers_kv = gathers_kv.get();
                // THE LIVE ROWS ONLY, built by the slot map's own walk — one token and one
                // `BatchRow` per LIVE slot, in slot order, and nothing invented for the padding
                // slots. `run_prefill_batch` widens this to the rung (`LaunchRows`), where a padding
                // slot runs LIVE 0 ENTIRE — same token, rotation, new-block column and prefix
                // history — so the cache write it cannot help making carries live 0's exact bytes
                // into the racing same-cell write (see the page-map note above). Its logits are
                // never read. The launch geometry is still the RUNG's: `seqs` rows are bound and
                // run, live or padding.
                let mut toks: Vec<usize> = Vec::with_capacity(slots.live());
                let rows_in = slots.live_rows(|live| {
                    let (id, start) = &decode_ids[live];
                    let req = requests
                        .get(id)
                        .ok_or_else(|| werr(format!("unknown cached req {id}")))?;
                    let tok = input_batch.token_at(id, *start).ok_or_else(|| {
                        werr(format!("decode batch: req {id} has no token at {start}"))
                    })? as usize;
                    toks.push(tok);
                    // The row's ROTARY position is still its own token count — sharing a write slot
                    // moves where the key is stored, not where the request is in its conversation.
                    Ok::<_, ExecutorError>(scratchy_subtile::sdsc_abstract::BatchRow {
                        // `*start` is THIS REQUEST's own token index (it came from `decode_ids[live]` and
                        // was just used as `input_batch.token_at(id, *start)`), which is why it is a position and not
                        // the `write_slot` above. Those were both bare integers in this scope until now.
                        rope_pos: scratchy_subtile::sdsc_abstract::SeqPos::new(*start as u32),
                        hist: req.kv_hist.clone(),
                    })
                })?;
                // THE LAUNCH POSITION IS THE WRITE SLOT, and now there is only one of it. The runtime
                // picks the body rung and how many resident pages to fold from this number, so it has to
                // cover every row's history — which the shared slot does by construction, being past all
                // of them. It used to be `max(starts)`: the maximum of the requests' own positions, which
                // covered the longest row but described no single write.
                // HOW MANY BLOCKS THE MASK WILL HAVE — `max_pages`, because the staging blocks by a
                // page and covers the live pages. Declared HERE and not earlier precisely so it comes
                // from the same number the staging uses; deriving it from the session's `cap` instead
                // was a fourth derivation of the mask's geometry, which is the bug class this whole
                // shape exists to close.
                //
                // The runtime refuses a fold needing more passes than this: past the last block it
                // reads unstaged zeros, and zero is VALID in an additive mask.
                // ⭐ A BLOCK PER (ROW, PAGE) PASS. The fold takes `pages * rows` passes — `fold_plan::reps`
                // — because the emitted op carries no row axis, so each pass must find its own mask block
                // or the runtime refuses (it did: "1 pass(es) but the staged mask has fewer blocks",
                // abandoning every fold launch, which read as a 6.3 ms decode step emitting one repeated
                // token). This is not a request concept: a pass is a pass, and `fold_pass` maps rep ->
                // (row, page) in exactly this order.
                // `width` is the RUNG's row count — the rows the launch binds, padding included, which is
                // exactly what `fold_plan::reps` multiplies by (`block_tables.len()`).
                //
                // ⭐⭐ FROM THE WRITE SLOT, NOT FROM `max_pages`. `max_pages` is the biggest page LIST any
                // live request holds; the shim's fold divides by `ceil(seq_pos / kv_page_slots)` where
                // `seq_pos` is the launch position — i.e. this same `write_slot`. Those two agree only
                // because `ensure_pages` grows every row to the shared slot, a fact stated in neither of
                // them. `FoldPages::covering(write_slot, ..)` IS the shim's ceiling, computed once on the
                // Rust side, so the mask's block count and the fold's pass count are one number.
                // ⛔ `resident_before`, NOT `covering`: the fold walks the RESIDENT prefix, and the page
                // this step's write lands in is the body's. `covering` includes that page, so at every
                // exact page multiple this staged one block more than the launch walks and the fold
                // refused (`walks 8 page(s) but the host STAGED the mask for 9` at start=2048).
                let fold_pages = scratchy_subtile::sdsc_abstract::FoldPages::resident_before(
                    write_slot,
                    scratchy_subtile::sdsc_abstract::SlotCount::new(
                        scratchy_subtile::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32,
                    ),
                )
                .ok_or_else(|| werr("decode batch: a row holds no pages".to_string()))?;
                // ⭐⭐⭐ AND WHAT ONE BLOCK DESCRIBES — a BAKE fact, from the same `gathers_kv` the index
                // table below is built for. A gathered bundle's fold pass IS a page and serves every row
                // (`MaskBlockForm::PerPage`), so the block count, the fill's block index and
                // `fold_plan::reps` are one decision reaching all three. Staging the wrong form is silent
                // both ways: `PerRowPage` against a collapsed fold leaves `mq-1` of every `mq` rows at −∞,
                // and `PerPage` against an uncollapsed one gives every pass row 0's validity.
                let fold_grid = fold_pages.grid(
                    width,
                    if bake_gathers_kv {
                        scratchy_subtile::sdsc_abstract::MaskBlockForm::PerPage
                    } else {
                        scratchy_subtile::sdsc_abstract::MaskBlockForm::PerRowPage
                    },
                );
                // ⛔⛔⛔ THE BUNDLE'S OWN PMASK MUST HOLD THIS GRID'S BLOCKS. `pmask = [nqh*mq, cap]`, so the
                // rung baked room for `cap / COLS` blocks while this step needs `pages * width`. The
                // capacity was already computed at load — for an `eprintln!` — and thrown away, so
                // exceeding it wrote past the placement and the fold read bytes nobody staged. In an
                // ADDITIVE mask those read as ZERO, which means VALID, so the tail passes attend whatever
                // the pool holds and the affected rows answer confidently and wrongly. Refusing names the
                // launch instead.
                // `None` = the layout was unreadable, so this keeps the prior behaviour rather than
                // refusing on a number we do not have.
                if let Some(cap) = mask_cap {
                    let need = scratchy_subtile::sdsc_abstract::MaskBlocks::of(fold_grid);
                    if !need.fits(*cap) {
                        return Err(werr(format!(
                            "decode batch: this step needs {} mask block(s) ({} page(s) x {} row(s)) but \
                             the rung's baked pmask holds {} — the fold would read mask bytes the host \
                             never staged, which are ZERO and therefore VALID, so those rows would attend \
                             keys that are not theirs. Admit fewer requests, serve a shorter context, or \
                             bake a rung whose pmask covers pages x width.",
                            need.get(),
                            fold_grid.pages().get(),
                            fold_grid.rows().get(),
                            cap.get(),
                        )));
                    }
                }
                // ⭐ THE BLOCK COUNT AND THE PAGE COUNT, DECLARED TOGETHER, and the value that comes back is
                // what the prefix mask is blocked by (`run_prefill_batch` takes it and can be blocked by
                // nothing else). The shim derives the same ceiling independently
                // (`(seq_pos + kv_page_slots - 1) / kv_page_slots`, twice), so declaring the host's
                // `FoldPages` lets `fold_plan::reps` refuse on a disagreement instead of blocking the mask by
                // one number while indexing passes by another — which sends every row but row 0 to another
                // row's page, silently, since an unstaged additive-mask byte reads as VALID.
                //
                // ⭐⭐ AND IT IS DECLARED OVER `write_slot`: EVERY LIVE REQUEST, decoding or not. That is the
                // population the batch appends at (a request still PREFILLING can be the deepest one), it is
                // the population `ensure_pages` grew the pool for, and it is what the shim's own ceiling is
                // computed from — so it is the number the fold walks, and therefore the number the mask must
                // be blocked by. The decoding rows alone are a smaller set and would answer smaller.
                let fold = fold_grid
                    .declare_to(sess)
                    .map_err(|e| werr(format!("decode batch: declare fold: {e}")))?;
                // ⭐ THE CAPACITY QUESTION, ANSWERED BY THE RUNTIME INSTEAD OF BY ME. The hd=128 deep-batch
                // failure tracks the FOLD PASS COUNT, not depth: 16 passes work (width 2 x 8 pages), 24 work
                // (4 x 6), 32 FAIL (4 x 8). `blocks * stride` is what the host stages into pmask, and pmask's
                // placement is a fixed reservation in the bundle — so if the product crosses it, the top
                // passes read bytes nobody staged, which an additive mask reads as VALID.
                //
                // ⛔ I DERIVED THIS TWICE AND GOT IT WRONG ONCE BY A FACTOR OF 2 (called the hypothesis
                // "refuted" on the strength of it). `rows = nqh*mq` and `block = rows*COLS*2` give 65536 B at
                // nqh=32/mq=4, i.e. 48 blocks inside a 3,145,728 B placement — but 24 blocks would be EXACTLY
                // that placement at a 131,072 B stride, and 24-works/32-fails is exactly what the card shows.
                // So print the real numbers and stop guessing.
                // ⛔ AND THE MASK MUST DESCRIBE EVERY SLOT ANY REQUEST HOLDS. The fold cannot attend a
                // slot the mask does not cover, and the mask covers `blocks * cols` = `max_pages` pages.
                // A request's keys reach to the SHARED write slot, which is the batch's MAXIMUM and not
                // its own length — so this bound moves with the batch, not with the request, and nothing
                // tied the two together.
                //
                // The failure mode is the one that looks like a broken kernel: a request whose history
                // falls outside the covered window attends its prompt and its newest token and nothing
                // between, so its distribution collapses onto a single repeated token. Saying so beats
                // discovering it as word salad.
                //
                // ⚠️ This is the LOOSE bound. The tight one is `active_cap * passes`, since a pass sweeps
                // `active_cap` columns and not a whole page — but the rung's `active_cap` is not exposed
                // to the worker, and guarding with a number that is merely nearby would be worse than
                // not guarding. Plumb it through `DecodeRung` and tighten this.
                // ⭐⭐⭐ TIGHTENED: the rung's OWN swept extent, not `PAGE_SLOTS`. A pass sweeps
                // `active_cap` columns of its page, so the covered window is `swept * pages` — and the
                // emitter bakes bodies at 64/128/256 (measured). With `PAGE_SLOTS` here the guard passed at
                // 1536 covered slots while a 128-column body actually covered 768, and the deepest row's
                // tail was described by no swept column. `SweptCols` now rides in from the manifest, which
                // is what the "Plumb it through `DecodeRung` and tighten this" note above asked for.
                if let Some(missed) = scratchy_subtile::sdsc_abstract::first_unswept_slot(
                    write_slot.slot(),
                    scratchy_subtile::sdsc_abstract::SlotCount::new(swept.get()),
                    max_pages as u32,
                ) {
                    return Err(werr(format!(
                        "decode batch: the fold sweeps {} slot(s) ({} swept column(s) per pass x \
                         {max_pages} page(s)) but the batch's write slot is {} — slot {} onward is swept by \
                         no pass, so a request would answer from its prompt and its newest token alone. \
                         Grow the pool's pages, admit fewer requests, or run a rung with a wider sweep.",
                        swept.get() as usize * max_pages,
                        swept.get(),
                        write_slot.get(),
                        missed.get()
                    )));
                }
                // ⭐⭐⭐⭐⭐ THE INDEX TABLE, BUILT ONLY IF THE BODY THIS STEP RUNS GATHERS — and the entry
                // factor is the SESSION's, not a constant.
                //
                // ⛔ THE ENTRY IS NOT A PAGE NUMBER. dxp computes `addr = idx * skip_addr + base` with
                // `skip_addr` one stick block (`hd * 64` elements — 4096 at hd=64, 8192 at hd=128), while
                // a PAGE spans every layer. So the factor is `page_stride_bytes / stick_block_bytes`,
                // read off this session's own `page_stride_bytes` — the very number
                // `fold_plan::page_base_bytes` multiplies — which makes the two addressings an identity
                // rather than two derivations to keep in step
                // (`zz_the_gather_index_reproduces_the_host_page_address`).
                //
                // ⛔ AND A REFUSAL, NEVER A ZERO TABLE, when the factor does not exist: an entry of 0 is
                // block 0, a REAL address (row 0's first page), so a fallback would give the whole batch
                // row 0's keys — fluent and wrong, which is the corruption this path exists to remove.
                let kv_blocks: Option<Vec<scratchy_subtile::sdsc_abstract::GatherEntry>> =
                    if bake_gathers_kv {
                        let per_page = sess.gather_entries_per_page(pool).ok_or_else(|| {
                        werr(
                            "decode batch: this bundle emits a GATHERED KV read, but a physical page \
                             is not a whole number of STICK BLOCKS — no integer index can name a page \
                             boundary, so every row would gather from inside the previous page"
                                .to_string(),
                        )
                    })?;
                        // ⭐⭐⭐ THE SCRATCH'S OWN SHAPE — and at page granularity it takes ONE extent, the
                        // rung's width, because a row is a REQUEST.
                        //
                        // ⛔ THE WINDOW COUNT IS GONE FROM THIS CALL, WHICH IS THE POINT. It used to be
                        // `SlotWindow::count_in(swept)` here and the EMITTER's own body rung there — two
                        // different numbers by construction (the host reads the CEILING body's `swept`, the
                        // emitter its interior rung), and a table built for a different `nb` wrote real block
                        // numbers into rows the copy does not read while leaving the rows it does read at
                        // zero: block 0, a REAL address. With `mq` alone both sides read the same rung.
                        let scratch = scratchy_subtile::sdsc_abstract::PageScratch::of_pass(
                        pool,
                        scratchy_subtile::sdsc_abstract::QueryRowCount::of_mq(seqs.get()),
                    )
                    .ok_or_else(|| {
                        werr(
                            "decode batch: this bundle emits a GATHERED KV read, but its geometry \
                             admits no page-granular copy — the head dim is not a whole number of \
                             sticks, the rung is wider than one index stick, or the plane's footprint \
                             does not fit the descriptor's u32 extents. The bake and the host disagree \
                             about whether this bundle gathers."
                                .to_string(),
                        )
                    })?;
                        // ⭐⭐⭐ THE PASS PITCH IS THE MASK'S BLOCK STRIDE, and it is the same `mask_shape` the
                        // launch already declared to the session (`set_mask_stride`). The index is an
                        // ACTIVATION, so it lives in the segment the fold shifts for the MASK — one shift, two
                        // tensors, therefore one pitch. Re-deriving it here from a page count is exactly the
                        // "same number, two derivations" defect the mask's own grid types exist to close.
                        // ⭐ THE MASK'S OWN NUMBER, ASKED OF THE MASK. This was
                        // `usize::try_from(mask_shape.rep_stride_bytes() / 4)` — the `/ 4` spelled at the
                        // launch, i.e. the second derivation `PassStride` exists to remove.
                        let pass_stride = mask_shape.pass_stride();
                        // ⛔ THE PASS PITCH IS NOT THE SUSPECT IT LOOKED LIKE, AND THAT IS MEASURED.
                        // Pinning every pass's entries to logical page 0 CHANGED the card's answer
                        // (granite-3.1-2b fp8, width 8), so pass `p >= 1` demonstrably reads its OWN
                        // index block: the index does ride the mask's per-pass segment shift, and the
                        // page-crossing failure was never here. It was the launch GROUPING — see
                        // `GroupKind::run_may_be_chunked`.
                        let table = scratchy_subtile::sdsc_abstract::page_gather_index_table(
                            scratch,
                            &page_maps,
                            max_pages,
                            per_page,
                            pass_stride,
                        )
                        .ok_or_else(|| {
                            werr(format!(
                                "decode batch: could not build the gather's index table from {} \
                             installed page map(s) at {max_pages} page(s) — a launch row holds \
                             fewer pages than the fold sweeps, an entry overflows int32, or the \
                             pass block ({} entries) is smaller than one pass needs \
                             ({} entries)",
                                page_maps.len(),
                                pass_stride.get(),
                                scratch.rows(),
                            ))
                        })?;
                        // ── GATHER DIAG (`SCRATCHY_GATHER_DIAG`) ── the index table is the one operand the
                        // fence could name and does not: `RAS::PCI::BusFence` (0xa35e) carries NO `vars` in
                        // `aiuras.json`, so the card states no address. Print the table's own extremes and the
                        // pool they must land inside, as ONE string (a per-line `eprintln!` comes back shredded
                        // mid-number), so an out-of-pool entry is a host-side fact before the launch.
                        if std::env::var_os("SCRATCHY_GATHER_DIAG").is_some() {
                            let (lo, hi) = table.iter().fold((i32::MAX, i32::MIN), |(l, h), e| {
                                (l.min(e.as_i32()), h.max(e.as_i32()))
                            });
                            let max_phys = page_maps.iter().flatten().copied().max().unwrap_or(-1);
                            // ⛔ THE UNIT IS THE STICK BLOCK, AND IT IS THE POOL'S OWN NUMBER. `per_page`
                            // is `page_stride_bytes / stick_block_bytes` and `skip_addr` is one stick
                            // block, so `entry * stick_block_bytes` IS the byte the card computes —
                            // converting with any other unit would mis-report the reach by `nkvh *
                            // PAGE_SLOTS / POOL_STICK` (32× at the shipped 2b geometry). A diagnostic
                            // that mis-reports an address is worse than none: this is the one print that
                            // says whether an entry lands inside the pool.
                            let stick = pool.stick_block_bytes() as i64;
                            eprint!(
                                "{}",
                                format_args!(
                                    "[gather] mq={} entries={} per_row={} pass_stride={} max_pages={} \
                                 per_page={} scratch_rows={} idx_lo={lo} idx_hi={hi} \
                                 max_phys={max_phys} stick_block_bytes={stick} \
                                 reach_bytes={} page_stride_bytes={} page_maps={} \
                                 mask_rep_stride={} table_bytes={}\n",
                                    seqs.get(),
                                    table.len(),
                                    scratch.entries_per_row(),
                                    pass_stride.get(),
                                    max_pages,
                                    per_page.get(),
                                    scratch.rows(),
                                    hi as i64 * stick,
                                    per_page.get() * stick,
                                    page_maps.len(),
                                    mask_shape.rep_stride_bytes(),
                                    table.len() * 4,
                                )
                            );
                        }
                        Some(table)
                    } else {
                        None
                    };
                let rows = run_prefill_batch(
                    Some((&rows_in, fold)),
                    kv_blocks.as_deref(),
                    &sh,
                    sess,
                    // A batched rung sweeps the whole pool, so it runs on the widest decode
                    // bucket. (The sendnn path bakes exactly one, so this is that one.)
                    model.decode.last().expect("at least one decode bucket"),
                    &toks,
                    seqs,
                    // ⭐ THE RUNG'S OWN ANSWER, riding with the rung's own `seqs` for the same reason:
                    // the meta above is borrowed for tensor ids and describes neither.
                    gathers_kv,
                    launch_pos,
                    // Every row is a different request and every row is sampled — the tail runs
                    // unfolded, so the logits come back de-interleaved, one contiguous row each.
                    LogitsWanted::PerRequest(logits),
                )?;
                let logits = rows
                    .ok_or_else(|| werr("decode batch: the rung produced no logits".to_string()))?;
                // ── KCACHE PROBE (SCRATCHY_KCACHE_PROBE), BATCHED STEP ── the batched counterpart of the
                // solo `[kcache]` probe: after the fold's forward (cache write included), read back the
                // layer-0 resident Kᵀ plane and print, per live request row, its launch slot, its pool
                // row, its block table and the actual key columns its prefix mask marks valid — so "the
                // keys under this request's mask are ANOTHER request's" is a fact the log states rather
                // than a hypothesis. Same `read_tensor` readback the solo probe uses, through the OWNER
                // decode session: every rung aliases its seg2 onto `sb.decode`'s pool region, so the
                // owner's placement names the very pool this forward just wrote, and the owner is the one
                // session whose seg2 host shadow spans a whole page on the paged path. `read_tensor`
                // converts exactly the baked placement, which covers PHYSICAL PAGE 0 of the pool — pages
                // past it are named in the output as unreadable rather than silently skipped, so a run
                // states what it could not see. Costs one whole-segment D2H per step; env-gated off.
                let vocab = model.vocab;
                // A REQUEST'S LOGITS COME BACK IN THE SLOT THE LAUNCH PUT IT IN. Ask the same `SlotMap`
                // that bound the page maps rather than recomputing the position here — one value decides
                // where a request's KV is written and where its distribution is read, so the two cannot
                // drift apart.
                for (di, (id, start)) in decode_ids.iter().enumerate() {
                    let slot = slots
                        .slot_of(di)
                        .ok_or_else(|| {
                            werr(format!(
                                "decode batch: request {id} was not laid into the {}-row launch",
                                seqs.count()
                            ))
                        })?
                        .index();
                    let row = logits
                        .get(slot * vocab..(slot + 1) * vocab)
                        .ok_or_else(|| werr(format!(
                            "decode batch: {} logit(s) for {} request(s) x {vocab} vocab — the rung's \
                             lm-head tail does not produce one per request",
                            logits.len(),
                            decode_ids.len()
                        )))?;
                    let tok = argmax(row) as u32;
                    // ⭐⭐ THE TOP-2 GAP, PER REQUEST, PER STEP — the instrument this path has been missing.
                    //
                    // Batched output is not reproducible run to run at an IDENTICAL chunk count (measured:
                    // one probe gave 'Tokyo','Tokyoto','Tokyoto' on three identical runs, chunks=32 each,
                    // while SOLO is bit-stable), and the divergence enters at DECODE STEP 2 — the first step
                    // whose mask must cover a hole. A SAMPLED TOKEN cannot tell a 1e-4 perturbation on a
                    // near-tied argmax from a request reading another's keys: both look like one wrong word.
                    // The GAP can. Two runs of the same input, diffed:
                    //   * same argmax, gap identical to the last digit  => that step is reproducible
                    //   * same argmax, gap differs by ~1e-3            => numeric noise, and the flip is a
                    //     coin toss on a near-tie (report the gap, not the word)
                    //   * different argmax with a LARGE gap            => a real KV/mask fault, not noise
                    // Print-only and env-gated: it changes nothing the card computes.
                    let req = requests.get_mut(id).expect("bound above");
                    input_batch.push_generated(id, tok);
                    input_batch.set_tokens_in_pool(id, start + 1);
                    // The launch wrote this request's key AT THE SHARED SLOT, so that is what its
                    // history records — and the next step's mask, built from the history, then says
                    // valid on exactly the slots the card wrote and −∞ on the hole.
                    req.record_kv(
                        write_slot,
                        scratchy_subtile::sdsc_abstract::SlotCount::new(1),
                    );
                    order.push(id.clone());
                    sampled.push(Some(tok));
                    batched.insert(id.clone());
                }
            }
        }
        for (i, req_id) in cr.req_ids.iter().enumerate() {
            #[cfg(feature = "spyre-hw")]
            if batched.contains(req_id) {
                continue;
            }
            let n = scheduler_output
                .num_scheduled_tokens
                .get(req_id)
                .copied()
                .unwrap_or(0);
            let start = cr.num_computed_tokens.get(i).copied().unwrap_or(0) as usize;
            let req = requests
                .get_mut(req_id)
                .ok_or_else(|| werr(format!("unknown cached req {req_id}")))?;
            // The step's grant was installed above; this checks it reaches what this forward writes.
            // Sized by the KV SPAN, not the token count — a request that has been in a batch holds a
            // hole, so its keys reach further than `start` and the page it writes into is the one past
            // its span.
            #[cfg(feature = "spyre-hw")]
            install_host_blocks(
                &mut model.session,
                req,
                req.kv_span_slots(n as u32),
                PageMapCtx {
                    host: hosts.get(req_id).map_or(&[], |v| v.as_slice()),
                    part,
                },
            )?;
            let s = run_request_step(
                model,
                req,
                req_id,
                input_batch,
                start,
                n,
                #[cfg(feature = "spyre-hw")]
                hosts.get(req_id).map_or(&[], |v| v.as_slice()),
                #[cfg(feature = "spyre-hw")]
                part,
            )?;
            order.push(req_id.clone());
            sampled.push(s);
        }

        // ⛔⛔⛔ THE HOLE-PAGE COUNTER IS GONE, AND ITS ANSWER IS THE REASON THIS FILE SHRANK.
        //
        // It counted pages served from a row's reserved HOLE run — the last use of a per-request row below the
        // host. It read 0 across 30 gate runs on granite-3.1-2b (hd=64) and granite-3.1-8b (hd=128), and
        // 0/20 on a repeat of the axis that had been firing, once the scheduler allocated `own pages + 1`
        // (`4ef84be5`). There is no reserve left to count from: `map_row` REFUSES instead, so the condition
        // this measured is now a hard error rather than a silent fallback.
        //
        // ⛔ TWO LESSONS FROM IT, BOTH ABOUT INSTRUMENTS RATHER THAN THIS CODE. It lived in `Worker::shutdown`
        // first, which `scr batch` never calls, so it reported the zero I was hoping for. And it printed only
        // when it MOVED, so "the reserve is dead" and "the instrument is dead" were the same silence — closed
        // with a positive control (`the_hole_counter_moves_when_the_hosts_blocks_stop_short_of_the_write_page`)
        // before any zero from it was believed.
        // One sampled token per request, in processing order; intermediate
        // prefill chunks (None) carry no token.
        let tokens: Vec<u32> = sampled.iter().map(|s| s.unwrap_or(0)).collect();
        let mut out = ModelRunnerOutput::from_ordered(order, tokens);
        for (idx, s) in sampled.iter().enumerate() {
            if s.is_none() {
                out.sampled_token_ids[idx].clear();
            }
        }
        // ⭐ REPORT WHAT THIS STEP DID TO EVERY LIVE REQUEST'S KV — the span its blocks must cover and
        // the prefix a token-indexed cache may claim. Over ALL live requests, not just the ones that
        // forwarded: a batched step advances the shared write slot past every live history, so a request
        // that only PADDED this step still had its span moved and needs the allocation to follow.
        #[cfg(feature = "spyre-hw")]
        {
            // THE DEEPEST ANY LIVE REQUEST WILL REACH, over the same live set the next step's
            // `BatchSlot::of` runs over — but bounded by each one's PROMPT rather than by the history it
            // happens to have written so far, so an in-step prefill chunk cannot move it past what was
            // reported. One derivation, read here and used by the scheduler; nothing re-derives it from
            // the batch composition.
            // ⛔ THE PROMPT COMES FROM `InputBatch`, per request, at both reads. It used to be a
            // `ReqState` field; reading it from the one store here is what keeps "how long is this
            // prompt" from having two answers.
            let prompt_of = |id: &String| SlotCount::new(input_batch.prompt_len(id) as u32);
            let pool_reach = requests
                .iter()
                .map(|(id, r)| r.kv_reach(prompt_of(id)))
                .max()
                .unwrap_or(SlotCount::new(0));
            out.kv_extent = requests
                .iter()
                .map(|(id, r)| (id.clone(), r.kv_extent(pool_reach, prompt_of(id))))
                .collect();
            // ⭐ AND THE SAME REACH, POOL-WIDE, FOR THE REQUESTS THAT HAVE NO ENTRY ABOVE.
            //
            // `kv_extent` is keyed by request id, so it can only size a request this worker has already
            // run. A request being ADMITTED next step is sized by its token count alone — and if it joins a
            // batch whose shared write slot is already deep, its write page is past every block the host
            // gave it. That is precisely the case the worker has been covering from its own reserved hole
            // run (`PoolPartition::hole`), which is the last reason a per-request KV row exists.
            //
            // Reporting the reach lets the HOST allocate that page instead. It is the same number the spans
            // above are maxed with, so there is nothing new to keep in step — one derivation, two readers.
            out.kv_pool_reach = Some(scratchy_core_common::KvSlotSpan::new(pool_reach.get()));
        }
        Ok(out)
    }

    fn shutdown(&mut self) {
        self.is_shutdown = true;
    }

    fn rank(&self) -> usize {
        self.config.rank
    }

    fn local_rank(&self) -> usize {
        self.config.local_rank
    }

    fn is_driver_worker(&self) -> bool {
        self.config.is_driver_worker
    }
}
