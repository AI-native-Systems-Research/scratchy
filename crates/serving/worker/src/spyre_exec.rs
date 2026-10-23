// SPDX-License-Identifier: Apache-2.0
//! THE `Worker` TRAIT — the engine's view of a Spyre device.
//!
//! Scheduling, batching, cache sizing and the step loop. Target-neutral in SHAPE (metal's
//! equivalent is `gpu_worker`'s own `impl Worker`, which is larger) and model-neutral in CONTENT:
//! every model-specific decision was resolved at bake and reaches here as generated data.

// ⭐ THE POOL/SLOT VOCABULARY IS CARD-ONLY, so its imports carry the same cfg as the blocks that
// use it. `PagedKvPool` used to be the exception — the context hook read its consts on BOTH paths;
// that hook reads the reach off the loaded bundle now, so even that import is card-only again.
#[cfg(feature = "spyre-hw")]
use std::collections::HashMap;

use scratchy_serving_engine::executor::ModelRunnerOutput;
use scratchy_serving_scheduler::scheduler::output::SchedulerOutput;
#[cfg(feature = "spyre-hw")]
use scratchy_subtile::sdsc_abstract::PagedKvPool;
#[cfg(feature = "spyre-hw")]
use scratchy_subtile::sdsc_abstract::{RowPages, SlotCount, SlotMap, SlotMapError};
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

/// One scheduled token of a cached request is a DECODE step only when the request's
/// prompt is already fully computed — the token this step runs sits at a GENERATED
/// position (`num_computed >= prompt_len`). A 1-token slice of an UNFINISHED prompt is
/// a prefill continuation and must take the per-request path: `run_request_step` runs
/// it through the token-addressed m=1 forward, which writes its K/V at its own token
/// position and samples nothing. Sent to batched decode instead, it would
/// `push_generated` a token the prompt never contained and `record_kv` it at the
/// SHARED batch slot — a bogus sample, plus a hole between the request's history and
/// its token positions that the batched chunk loop (whose write slot IS its token
/// position) can no longer write correctly.
// The `test` arm makes the fn exist under `spyre-emu`, which compiles this
// module via the `spyre` feature but has no `spyre-hw`; CI runs the contract
// test in that shape (the spyre-triton job).
#[cfg(any(test, feature = "spyre-hw"))]
pub(crate) fn solo_step_is_decode(
    num_scheduled: usize,
    num_computed: usize,
    prompt_len: usize,
) -> bool {
    num_scheduled == 1 && num_computed >= prompt_len
}

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
    /// The Spyre worker applies no grammar.
    fn set_stop_token_ids(&mut self, _stop_token_ids: &[u32]) {}

    /// THE CONTEXT THE POOL CAN ADDRESS, so the ENGINE rejects an over-long prompt at admission
    /// instead of the worker killing the whole batch mid-forward.
    ///
    /// 🛑 THIS IS THE HOOK I SHOULD HAVE USED FIRST, AND METAL/CUDA ALREADY DO (`gpu_worker.rs:2509`). I
    /// enforced the page ceiling inside `ensure_pages`, which runs INSIDE a forward — so refusing ONE
    /// over-deep request failed the whole step with an `Executor error` and took the other three requests
    /// with it. Measured: 4 of 4 requests errored where 3 of 4 had answered. A per-request limit enforced
    /// per-BATCH is a batch-fatal limit.
    ///
    /// The engine's own doc for this method says exactly what is wanted: it caps `max_model_len` to this
    /// value "so an over-long prompt is rejected, not silently truncated by the kernel". One request gets a
    /// clean rejection; the rest of the batch is untouched.
    ///
    /// ⭐ THE NUMBER IS `min(WHAT THE POOL FUNDS, WHAT THE MASK BAKED)` — both read from the load, never
    /// a const. The affordable term is the MEMORY law: the pool's own budget over its page stride across
    /// the launch rows (`served_context`, decided at pool sizing), so the served context moves with the
    /// card, the KV dtype's page stride and the row count the way metal's does with available GPU memory.
    /// The mask term is the DEVICE law and the safety: every rung's pmask is baked at
    /// `MAX_SEGMENT_BYTES / page_bytes` blocks, which covers every pool any card can budget for — so it
    /// can only bind if the worker's page arithmetic ever disagrees with the bake's, and then it is the
    /// difference between a clean admission refusal and the batch-fatal one above.
    fn kv_max_addressable_tokens(&self) -> Option<usize> {
        #[cfg(feature = "spyre-hw")]
        {
            let model = self.model.as_ref()?;
            let SendnnSession::SuperDsc(sb) = &model.session;
            let mask = sb.mask_reach()?;
            let mask_tokens = mask.get() as usize * PagedKvPool::PAGE_SLOTS;
            return Some(model.served_context.min(mask_tokens));
        }
        // The emulator grows its KV on the host and sizes no pool, so `served_context`
        // stays `usize::MAX` there — a cap that never binds, which the engine treats
        // exactly like the absence of one.
        #[cfg(not(feature = "spyre-hw"))]
        self.model.as_ref().map(|m| m.served_context)
    }

    /// 🛑 THE DECODE READS ONE BLOCK TABLE, SO HYBRID SWA KV MUST STAY OFF — the trait's own
    /// contract (`worker.rs`: "a worker whose decode reads a single block table ... must run on
    /// the uniform single-pool layout"). CUDA overrides this for the same reason.
    ///
    /// ⛔⭐⭐ THE GEMMA-4-12B fp8 GARBAGE-OUTPUT DEFECT, FOUND (2026-10-05). `SpyreWorker`
    /// inherited the trait default `true`, so the engine handed gemma-4 the HYBRID scheduler
    /// config (card log: "Hybrid SWA KV enabled: 84 shared blocks, 6 groups (5 sliding)") while
    /// this backend bakes exactly ONE KV group — `spyre_exec.rs`'s own install path reads only
    /// group 0's table (`.and_then(|groups| groups.first())`) and its comment admits "A hybrid
    /// model would need a group per attention class, which this backend does not bake".
    ///
    /// The corruption is structural, not subtle:
    /// * Page-size unification (`compute_hybrid_kv_layout`) scales the FULL group's block size
    ///   4× (1024 tokens/block for gemma-4-12b: sliding 8×256 vs global 1×512) while the spyre
    ///   pool's page is `PagedKvPool::PAGE_SLOTS` = 256 slots and the worker maps each block id
    ///   to ONE page — so from token 256 on, a request's full-attention KV lands on pages its
    ///   truncated table never names, and it attends another request's (or stale) keys: fluent
    ///   multilingual gibberish at full speed, exactly the observed 8.3 tok/s symptom.
    /// * The 5 SLIDING groups' tables are DISCARDED outright — the sliding-window layers
    ///   attend through the full group's block ids, which the allocator freed out-of-window
    ///   (null-padded) blocks for.
    /// * Every non-hybrid model (granite) is unaffected — the same binary serves it coherently.
    /// * The emulator is unaffected — the paged install is `spyre-hw`-gated, so emu runs never
    ///   read a scheduler block table at all.
    ///
    /// Returning `false` keeps the scheduler on the UNIFORM single-group pool, where one block
    /// table per request, 256 slots per block, IS the layout this backend bakes. The cost is the
    /// documented uniform-pool tradeoff (sliding layers keep full-context KV rather than
    /// window-evicting) — memory, not correctness, and the pool's byte-budget sizing already
    /// accounts for it.
    fn supports_hybrid_swa_kv(&self) -> bool {
        false
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

    /// THE POOL'S OWN PRICE OF ONE SCHEDULER BLOCK — the page stride, prorated to `block_size`
    /// tokens.
    ///
    /// The engine sizes the scheduler's block allocator as `budget / kv_block_bytes`, then clamps
    /// DOWN to [`Self::kv_cache_num_blocks_override`]. Without this hook it divided by its own
    /// dense-row estimate (`2 × kv_heads × head_dim × elem`, no Kᵀ plane), which overcounts a spyre
    /// page by 4/3 — and because the clamp only ever lowers, an UNDERestimated block count sailed
    /// under the override unheard, leaving the scheduler naming fewer pages than the pool has (the
    /// 2b card run: 204 blocks against a 544-page host range). With the real stride the computed
    /// count is `budget / stride`, which is ≥ the host range by construction (the pool is cut from
    /// that same division), so the clamp fires and lands exactly on the host range: the two counts
    /// are one count.
    ///
    /// `None` on the paths that never size a paged pool — the engine's estimate is all there is to
    /// divide by there.
    fn kv_block_bytes(&self, block_size: usize) -> Option<usize> {
        self.model
            .as_ref()
            .and_then(|m| m.pool_stride_bytes)
            .map(|stride| {
                // `PLANE_SLOTS` is how many tokens one page holds, so the stride prorated to
                // `block_size` tokens is the block cost at ANY block size the engine picks. The
                // path is fully qualified because the pool/slot vocabulary's imports are
                // card-only (see this file's header) and the emu tier reaches here too, with
                // `None` — no pool was ever sized.
                let plane_slots = scratchy_subtile::sdsc_abstract::PagedKvPool::PLANE_SLOTS as u64;
                (stride / plane_slots * block_size as u64) as usize
            })
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
            // ⛔ ONE KV CACHE GROUP: `block_ids` is per group, and the scheduler's tracker gives a
            // non-hybrid model one (`SimpleBlockTracker::allocate_slots`), so group 0 is the list. A
            // hybrid model gets a table per attention class (`allocate_slots_hybrid`), which this
            // backend does not bake.
            // ⭐ AND THE TOKENS THEMSELVES, in the same store as the block table. `ReqState` used to hold
            // `tokens` + `prompt_len`; the length is now derived from this one value.
            input_batch.add_request(
                nr.req_id.clone(),
                &prompt,
                nr.block_ids.first().cloned().unwrap_or_default(),
                nr.num_computed_tokens,
            );
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
            // ⛔ ONE KV CACHE GROUP, as for `input_batch.add_request` above: group 0 is the list.
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
        // Every cached request scheduled for exactly ONE GENERATED token is a decode step, and those
        // are the whole reason the weight set gets streamed over and over: each used to be its own
        // forward. Run them together on the smallest rung that holds them and the weights move once.
        // A 1-token slice of an UNFINISHED prompt is not one of them (`solo_step_is_decode`): it is
        // a prefill continuation and stays on the per-request path below.
        //
        // Through `run_prefill_batch` — the SAME path a prompt chunk takes, at B rows instead of
        // N. Decode is not a separate implementation of attention; what a batch needs and a prompt
        // does not is a page map per request and a validity row per request, and both are already
        // carried by the bundle. What the caller supplies is the new-block law: a prompt's rows are
        // consecutive positions and attend each other causally, a batch's rows are separate requests
        // and must attend only themselves.
        //
        // ⛔⛔⛔ THE BATCHED DECODE STEP IS DELETED — no host pick exists under hw.
        //
        // This block used to gather every live decode request into one fold launch
        // on the widest rung, read the whole `[rows, vocab]` logits block back, and
        // argmax each row ON THE HOST — the batched sampler seat. The batched rungs
        // carry no device argmax (the mint is one-request), so under the no-host-pick
        // law the fold has no pick and cannot run: every request takes the per-request
        // loop below, each through its own minted m=1 rung — the same machinery a
        // solo `scr chat` uses, one request at a time. The throughput cost at
        // concurrency is the ordered, named regression; the per-request CB replay
        // (#603) is the path that restores batched throughput.
        //
        // A continuation chunk (n > 1) is not a decode step and keeps the per-request path below, as
        // does every request when no ladder was baked or the live count exceeds the widest rung.
        #[cfg(feature = "spyre-hw")]
        if cr.req_ids.len() > 1 {
            // ONCE per serve session, not per step — a per-step line at c8 is log
            // flooding, its own defect.
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[spyre-worker] batched decode is host-pick-free: {} live request(s) run \
                     per-request through the minted m=1 rungs (the batched fold's pick was a \
                     host argmax — deleted); throughput at concurrency is the named regression",
                    cr.req_ids.len()
                );
            }
        }
        for (i, req_id) in cr.req_ids.iter().enumerate() {
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

#[cfg(test)]
mod tests {
    use super::solo_step_is_decode;

    #[test]
    fn a_one_token_slice_of_an_unfinished_prompt_is_not_a_decode_step() {
        // Mid-prompt: this step computes the prompt's 2nd token — a continuation.
        assert!(!solo_step_is_decode(1, 1, 500));
        // The prompt's LAST token is still a prompt token, not a generated one —
        // the exact 1-token remainder a ragged budget leaves, and the case the
        // shared-slot write would corrupt.
        assert!(!solo_step_is_decode(1, 499, 500));
        // First generated position: the prompt is fully computed.
        assert!(solo_step_is_decode(1, 500, 500));
        // Mid-generation.
        assert!(solo_step_is_decode(1, 600, 500));
        // A multi-token slice is a continuation wave, never a decode step.
        assert!(!solo_step_is_decode(96, 96, 500));
    }
}
