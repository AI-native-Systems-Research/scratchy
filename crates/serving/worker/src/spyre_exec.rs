// SPDX-License-Identifier: Apache-2.0
//! THE `Worker` TRAIT — the engine's view of a Spyre device.
//!
//! Scheduling, batching, cache sizing and the step loop. Target-neutral in SHAPE (metal's
//! equivalent is `gpu_worker`'s own `impl Worker`, which is larger) and model-neutral in CONTENT:
//! every model-specific decision was resolved at bake and reaches here as generated data.

#[cfg(feature = "spyre-hw")]
use std::collections::HashMap;

use scratchy_serving_engine::executor::ModelRunnerOutput;
use scratchy_serving_scheduler::scheduler::output::SchedulerOutput;

use crate::error::ExecutorResult;
use crate::spyre_forward::*;
use crate::spyre_types::*;
use crate::spyre_worker::*;
use crate::worker::Worker;

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
#[cfg(feature = "spyre-hw")]
impl SpyreWorker {
    /// ⭐ #32 (c4) — THE DRAIN'S ONE POP: collect one in-flight step and own its whole
    /// bookkeeping. The same body runs for every caller — the knock (`drain_until`), the
    /// in-call drain of a schedule that could not defer, and the entry-drain that empties the
    /// queue before the next step builds its launch — so there is one collect path, not one
    /// per driver. The step's cell is filled here (with the step's own error if the collect
    /// failed) so a parked resolver can never hang.
    fn drain_one(
        infl: InFlightDecode,
        model: &mut Loaded,
        requests: &mut HashMap<String, ReqState>,
        input_batch: &mut scratchy_serving_engine::input_batch::InputBatch,
    ) -> ExecutorResult<Vec<u32>> {
        let InFlightDecode {
            cell,
            sub,
            rung,
            rows,
        } = infl;
        // Only the submitting rung's session can finish the read — each rung owns its own
        // executor and stream, and the W=1 seat is the owner session itself.
        let SendnnSession::SuperDsc(sb) = &mut model.session;
        let sess = sb.decode_seat(rung).sess;
        let toks = match collect_decode_tokens(sess, &sub) {
            Ok(t) => t,
            Err(e) => {
                let msg = e.to_string();
                cell.fill(Err(scratchy_serving_engine::error::EngineError::Executor(
                    msg.clone(),
                )));
                return Err(werr(msg));
            }
        };
        // A REQUEST'S TOKEN COMES BACK IN THE SLOT THE LAUNCH PUT IT IN — the slot was resolved
        // at submit and rides in `rows`, so the drain cannot re-derive (and mis-derive) it.
        // The sentinel check already happened at the decode law (`collect_decode_tokens`);
        // this guard is the slice's own bound.
        let n_req = rows.len();
        let mut out = Vec::with_capacity(n_req);
        for (id, start, slot) in rows {
            let tok = toks.get(slot).copied().ok_or_else(|| {
                werr(format!(
                    "decode batch: {} token(s) for {n_req} request(s) — the rung's sampler did \
                     not pick one per launch slot",
                    toks.len()
                ))
            })?;
            requests
                .get_mut(&id)
                .ok_or_else(|| werr(format!("unknown in-flight req {id}")))?;
            input_batch.push_generated(&id, tok);
            input_batch.set_tokens_in_pool(&id, start + 1);
            // ⭐ THE KV HISTORY WAS RECORDED AT SUBMIT (c4b) — the write slot is a position fact
            // the host knew when it launched, and the next step's geometry reads it before this
            // drain on a stable seating. The drain owns the PICK; the history was never its fact.
            out.push(tok);
        }
        cell.fill(Ok(out.clone()));
        Ok(out)
    }
}

impl Worker for SpyreWorker {
    /// The Spyre worker applies no grammar.
    fn set_stop_token_ids(&mut self, _stop_token_ids: &[u32]) {}

    /// THE CONTEXT THE POOL CAN ADDRESS, so the ENGINE rejects an over-long prompt at admission
    /// instead of the worker killing the whole batch mid-forward — what the run serves
    /// ([`Loaded::served_context`]): the declared context, which the baked pool holds by the load's
    /// own check. The emulator sizes no pool and reports `usize::MAX`, a cap that never binds.
    fn kv_max_addressable_tokens(&self) -> Option<usize> {
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

    /// THE KV POOL'S BYTES — on the card, the KV segment the bake placed ([`Loaded::kv_budget_bytes`]),
    /// which the engine divides by [`Self::kv_block_bytes`] and [`Self::kv_cache_num_blocks_override`]
    /// then holds to the pool's own page count. The emulator grows nothing on the device, so its number
    /// is a host-RAM allowance.
    fn determine_available_memory(&mut self) -> ExecutorResult<usize> {
        if let Some(b) = self.model.as_ref().and_then(|m| m.kv_budget_bytes) {
            return Ok(b as usize);
        }
        Ok(8usize << 30)
    }

    /// ONE SCHEDULER BLOCK'S BYTES — a pool page ([`Loaded::pool_stride_bytes`]) prorated to
    /// `block_size` tokens of its `PAGE_SLOTS`. `None` on the emulator.
    fn kv_block_bytes(&self, block_size: usize) -> Option<usize> {
        self.model
            .as_ref()
            .and_then(|m| m.pool_stride_bytes)
            .map(|page| {
                let slots = scratchy_target_spyre::paged_step::PAGE_SLOTS as u64;
                (page / slots * block_size as u64) as usize
            })
    }

    /// ⭐ THE POOL'S PAGE COUNT, so every block id the scheduler hands out is a page this pool has —
    /// every page but the null page a padded row writes (`paged_step::page_of_block`). The pool is the
    /// bake's: every bundle places it, at the decode wiring's `PagedStep::pages`.
    fn kv_cache_num_blocks_override(&self) -> Option<usize> {
        self.model
            .as_ref()
            .and_then(|m| m.decode.first())
            .and_then(|b| {
                #[cfg(feature = "spyre-hw")]
                let paged = b.wiring.paged;
                #[cfg(not(feature = "spyre-hw"))]
                let paged = b.paged;
                paged
            })
            .map(|p| p.host_blocks() as usize)
    }

    /// THE WIDEST CONCURRENCY ONE DECODE LAUNCH CAN EXPRESS — the widest baked decode rung. A CAP, not a
    /// refusal (cuda's rule): a capped run stays on the batched path, where without the cap the
    /// scheduler admits past the ladder and the step refuses.
    fn max_num_seqs_override(&self) -> Option<usize> {
        #[cfg(feature = "spyre-hw")]
        {
            self.model.as_ref().map(|m| match &m.session {
                SendnnSession::SuperDsc(sb) => sb
                    .decode_rungs
                    .iter()
                    .map(|r| r.seqs.count())
                    .max()
                    .unwrap_or(1),
            })
        }
        #[cfg(not(feature = "spyre-hw"))]
        None
    }

    /// Every row writes its key at its own token position, on every tier.
    fn kv_addressing(&self) -> scratchy_core_common::KvAddressing {
        scratchy_core_common::KvAddressing::ByToken
    }

    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
        // ⭐ #32 (c4b) — THE ENTRY-DRAIN IS GONE, AND THE SEED LAW IS WHY. It existed because a
        // decode step's launch read `input_batch.token_at` — step j's pick — to seed step j+1's
        // embed head, so the queue had to be empty before every launch. The device writes the
        // seeds itself (each step's ChainStage decomposes its pick into the persistent sticks),
        // so a step on an UNCHANGED seating binds nothing and needs nothing drained: it submits
        // while the previous step is still in flight — depth 2. A CHURNED seating (below, at the
        // rung) still drains in-call, through the same pop the knock takes, because a re-seated
        // row's seed is the previous step's pick and only the drain holds it. Deferred steps that
        // neither churn nor block are drained by their own resolver's knock — the engine resolves
        // a step before the schedule two later can clean up the requests it named, so no drain
        // races the finished-request cleanup.
        let Self {
            model,
            requests,
            input_batch,
            ..
        } = self;
        let model = model.as_mut().ok_or_else(|| werr("model not loaded"))?;
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
            requests.insert(
                nr.req_id.clone(),
                ReqState {
                    // ⭐ THE DRAW, DECIDED ONCE AT ADMISSION and staged per step by every seat the
                    // request's rows run through — the prompt-final chunk's, the solo decode loop's,
                    // the batch's. Cached steps never see sampling params again, so this is the one
                    // door. A request this path cannot serve honestly (a knob with no card leg) is
                    // refused here, naming the knob — on both tiers, which mint the same draw.
                    draw: SampledDraw::of(&nr.req_id, nr.sampling_params.as_ref()).map_err(werr)?,
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
            let req = requests.get_mut(&nr.req_id).expect("just inserted");
            let s = run_request_step(model, req, &nr.req_id, input_batch, start, n)?;
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
            let computed = cr.num_computed_tokens.get(i).copied().unwrap_or(0) as usize;
            if !requests.contains_key(id) {
                continue;
            }
            // A RESUME IS A RESET: the blocks this request held were freed and re-allocated, and the
            // store's token count is what the scheduler says it now holds. Its pages are read out of
            // `InputBatch` at each forward, so nothing else here remembers the old ones.
            if resumed {
                input_batch.set_tokens_in_pool(id, computed);
            }
        }
        // ── BATCHED DECODE ────────────────────────────────────────────────────────────────────────
        // Every cached request scheduled for exactly ONE GENERATED token is a decode step, and those
        // are the whole reason the weight set gets streamed over and over: each used to be its own
        // forward. Run them together on the smallest rung that holds them and the weights move once.
        // A 1-token slice of an UNFINISHED prompt is not one of them (`solo_step_is_decode`): it is
        // a prefill continuation and stays on the per-request path below.
        //
        // Through `run_card_launch` — the SAME path a prompt chunk takes. One request decodes on the
        // W=1 owner (spyre-inference's `page_attn_head_major_decode`), several on a ladder rung
        // (`batched_decode_head_major`); the paged step each stages is built from the requests' own
        // block tables and positions.
        //
        // A continuation chunk (n > 1) is not a decode step and keeps the per-request path below;
        // a DECODE step never does — one request is the W=1 OWNER here, and a live count past the
        // widest rung refuses below.
        #[cfg(feature = "spyre-hw")]
        let mut batched: std::collections::HashSet<String> = std::collections::HashSet::new();
        #[cfg(feature = "spyre-hw")]
        {
            let decode_ids: Vec<(String, usize)> = cr
                .req_ids
                .iter()
                .enumerate()
                .filter(|(i, id)| {
                    cr.num_computed_tokens.get(*i).is_some()
                        && solo_step_is_decode(
                            scheduler_output
                                .num_scheduled_tokens
                                .get(*id)
                                .copied()
                                .unwrap_or(0),
                            cr.num_computed_tokens[*i] as usize,
                            input_batch.prompt_len(id),
                        )
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
            // ⭐ ONE DECODE PATH: the rung is selected for EVERY decode step, one request
            // included — a single request is a batch of 1 on the OWNER session (the W=1
            // rung, by the loader's `rn <= 1 → continue`), through the same submit, defer,
            // drain, seat walk and collect. `None` (live past the widest rung) refuses
            // below: there is no one-forward-per-request decode arm.
            let rung = if decode_ids.is_empty() {
                None
            } else {
                {
                    // The count the rung is selected FROM is the LIVE one; everything the launch
                    // then binds is sized by the rung's own baked width.
                    let SendnnSession::SuperDsc(sb) = &model.session;
                    sb.decode_rung_for(
                        scratchy_subtile::sdsc_abstract::LiveRows::of_scheduled(decode_ids.len()),
                        decode_ids
                            .iter()
                            .map(|(_, start)| *start)
                            .max()
                            .unwrap_or(0),
                    )
                }
            };
            // ⭐⭐⭐ #32 (c4b) — THE SEATING DISSOLVES whenever a decoder runs OUTSIDE the batched
            // arm (`rung` is `None` with decoders live — the live count ran past the widest baked
            // rung; a single request is the OWNER and always seats). Two things must happen, and both are the seed law's own:
            // the per-request path reads the token POOL (`token_at` = the previous pick), so the
            // queue drains first — the same pop, never a second path; and the stored seating is
            // DISCARDED, because a solo interlude advances a request's pick on another session
            // while the rung's sticks still hold its pick from the last BATCHED step — a later
            // batched step that compared equal against the stale seating would read a STALE seed,
            // silently. `None` makes the next batched step churn: drain (a no-op here) and seed
            // from the pool. A step with NO decoders (pure prefill) dissolves nothing — no
            // decoder's pick advanced, so the sticks still hold every seat's own seed.
            #[cfg(feature = "spyre-hw")]
            if rung.is_none() && !decode_ids.is_empty() {
                self.seating = None;
                while let Some(infl) = self.in_flight.pop_front() {
                    Self::drain_one(infl, model, requests, input_batch)?;
                }
            }
            if let Some(ri) = rung {
                // ⭐⭐⭐ #32 (c4b) — THE SEED LAW'S CHURN TEST, taken before the rung's session is
                // borrowed (the drain needs `&mut model.session`). The seating is the rung plus
                // which request sits in each launch slot, dense in `decode_ids` order; the pad set
                // is decided by the same pair (rung width minus live count), so the pair IS the
                // seating. IDENTICAL to the previous step's ⇒ the card's own ChainStage writes are
                // every seat's seed and this step binds none — it submits with the previous step
                // still in flight (depth 2). Any difference — a join, a leave, a rung switch, a
                // changed pad count — re-seats rows onto sticks holding the PREVIOUS occupant's
                // pick (or none at all, on a fresh session), so those steps DRAIN FIRST, through
                // the same pop the knock takes, and seed from the collected picks: a fresh row's
                // `token_at` is its prompt's final token, a continuing row's is the pick the drain
                // just wrote into the pool. Churn is a SCHEDULER fact, known before submit — the
                // host never waits on a card result to learn it.
                let seating_now: Vec<String> =
                    decode_ids.iter().map(|(id, _)| id.clone()).collect();
                let churned = match &self.seating {
                    None => true,
                    Some((last_rung, last_ids)) => {
                        *last_rung != ri || last_ids.as_slice() != seating_now.as_slice()
                    }
                };
                self.seating = Some((ri, seating_now));
                if churned {
                    while let Some(infl) = self.in_flight.pop_front() {
                        Self::drain_one(infl, model, requests, input_batch)?;
                    }
                }
                // ⭐ STAGED AFTER ANY DRAIN: the churn drain above takes `&mut model` (the
                // session it collects through), so the shared immutable facts ride below it.
                let sh = Shared {
                    embed_tokens: &model.embed_tokens,
                    hidden: model.hidden,
                    vocab: model.vocab,
                };
                let SendnnSession::SuperDsc(sb) = &mut model.session;
                let DecodeSeat { seqs, sess } = sb.decode_seat(ri);
                tracing::debug!(
                    "[spyre-worker] decode: {} live on the {}-row rung",
                    decode_ids.len(),
                    seqs.count()
                );
                // ⭐ EACH LIVE REQUEST'S PAGES AND POSITION, in slot order (live `i` runs in slot
                // `i`): the scheduler's block table as pool pages, and the position of the one token
                // this step decodes — its token count before the step. The paged step's tables are
                // built from exactly these; a padding slot writes the null slot and is read by
                // nobody.
                let blocks: Vec<Vec<u32>> = decode_ids
                    .iter()
                    .map(|(id, _)| {
                        input_batch
                            .block_table(id)
                            .map(|t| {
                                t.iter()
                                    .map(|&b| {
                                        scratchy_target_spyre::paged_step::page_of_block(b as u32)
                                    })
                                    .collect()
                            })
                            .unwrap_or_default()
                    })
                    .collect();
                // ⭐⭐⭐ THE TOKEN IS READ ONLY ON A CHURNED SEATING. A stable step's seeds are the
                // card's own ChainStage writes — `token_at` here would be the PREVIOUS step's
                // pick, uncollected, and reading it is exactly the fence the seed law deletes. On
                // churn the drain above has already collected, so `token_at(id, start)` answers:
                // a FRESH row's prompt-final token, or a continuing row's just-collected pick.
                let mut toks: Vec<usize> = Vec::with_capacity(decode_ids.len());
                // ⭐ THE LIVE ROWS' DRAWS, one per live slot, in slot order, so a law can never
                // describe another request's row.
                let mut draws: Vec<Option<&SampledDraw>> = Vec::with_capacity(decode_ids.len());
                for (id, start) in &decode_ids {
                    let req = requests
                        .get(id)
                        .ok_or_else(|| werr(format!("unknown cached req {id}")))?;
                    draws.push(req.draw.as_ref());
                    if churned {
                        let tok = input_batch.token_at(id, *start).ok_or_else(|| {
                            werr(format!("decode: req {id} has no token at {start}"))
                        })? as usize;
                        toks.push(tok);
                    }
                }
                let seqs_in: Vec<scratchy_target_spyre::paged_step::BatchSeq<'_>> = decode_ids
                    .iter()
                    .zip(&blocks)
                    .map(
                        |((_, start), b)| scratchy_target_spyre::paged_step::BatchSeq {
                            blocks: b,
                            pos: *start,
                        },
                    )
                    .collect();
                // The W=1 seat is one sequence's decode (`page_attn_head_major_decode`); a ladder rung
                // is a decode batch (`batched_decode_head_major`).
                let rows_in = match ri {
                    RungId::Owner => LaunchRows::Sequence {
                        blocks: &blocks[0],
                        rows: scratchy_target_spyre::paged_step::StepRows {
                            start: decode_ids[0].1,
                            real: 1,
                        },
                    },
                    RungId::Ladder(_) => LaunchRows::Batch { seqs: &seqs_in },
                };
                // ⭐ THE ONE DECODE PATH, SUBMITTED: every row is a different request and every row
                // is sampled ON THE CARD by the rung's minted sampler; the idx stick's raw D2H is
                // enqueued and comes back through the SAME collect door whether this schedule
                // defers it to the drain or collects it right here.
                let sub = match run_card_launch(
                    rows_in,
                    &sh,
                    sess,
                    // The one-request decode's wiring: every rung is the same tape at its own rows,
                    // binding the same tensor ids.
                    model.decode.last().expect("at least one decode bucket"),
                    seqs,
                    SamplerLaws {
                        seeds: if churned {
                            scratchy_subtile::sdsc_abstract::SeedSource::Host(&toks)
                        } else {
                            scratchy_subtile::sdsc_abstract::SeedSource::OnCard
                        },
                        want_logits: LogitsWanted::PerRequestDeferred,
                        laws: SeatLaws::Batch { draws: &draws },
                    },
                )? {
                    BatchOut::Deferred(sub) => sub,
                    BatchOut::Done(_) => {
                        return Err(werr(
                            "decode: the rung finished in-call — the deferred arm is the only \
                             decode path"
                                .to_string(),
                        ));
                    }
                };
                // ⭐ #32 (c4) — THE STEP IS SUBMITTED; its collect is QUEUED, not run. Everything
                // the drain needs rides in `InFlightDecode` — per live row, in `decode_ids` order,
                // the request, its position and the launch slot its token lands in.
                let laid: Vec<(String, usize, usize)> = decode_ids
                    .iter()
                    .enumerate()
                    .map(|(slot, (id, start))| (id.clone(), *start, slot))
                    .collect();
                // A step defers only when it is THE WHOLE STEP — a pure decode, on a driver whose
                // engine can drain it. A mixed schedule (a prefill chunk ran, or a per-request
                // forward will run after this) collects in-call through the SAME pop, because its
                // output has to carry the other rows' tokens with it.
                let pure = scheduler_output.scheduled_new_reqs.is_empty()
                    && cr
                        .req_ids
                        .iter()
                        .all(|id| decode_ids.iter().any(|(d, _)| d == id));
                let cell = std::sync::Arc::new(scratchy_serving_engine::executor::DrainCell::new());
                self.in_flight.push_back(InFlightDecode {
                    cell: cell.clone(),
                    sub,
                    rung: ri,
                    rows: laid,
                });
                if pure && self.drain_door.is_some() {
                    // THE DEFERRED OUTPUT: knock, THEN wait — in that order, because a resolver
                    // that waited first would block the one thread that can fill the cell.
                    let door = self.drain_door.clone().expect("checked above");
                    let waiting = cell.clone();
                    let resolver = move || -> Result<Vec<u32>, String> {
                        door.knock(waiting.clone()).map_err(|e| e.to_string())?;
                        waiting.wait().map_err(|e| e.to_string())
                    };
                    let req_ids: Vec<String> =
                        decode_ids.iter().map(|(id, _)| id.clone()).collect();
                    return Ok(ModelRunnerOutput::deferred(req_ids, Box::new(resolver)));
                }
                // NOT DEFERRABLE — drain it right here, through the same pop the knock takes.
                let infl = self.in_flight.pop_back().expect("queued just above");
                let rows = Self::drain_one(infl, model, requests, input_batch)?;
                // A REQUEST'S TOKEN COMES BACK IN THE SLOT THE LAUNCH PUT IT IN — live `i` in slot
                // `i`. The pool bookkeeping (push_generated, set_tokens_in_pool) already ran in the
                // drain; this loop only carries the step's answer into the output.
                for (slot, (id, _start)) in decode_ids.iter().enumerate() {
                    let tok = rows.get(slot).copied().ok_or_else(|| {
                        werr(format!(
                            "decode: {} token(s) for {} request(s) — the rung's sampler did not \
                             pick one per launch slot",
                            rows.len(),
                            decode_ids.len()
                        ))
                    })?;
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
            // ⛔ REAL BATCHED DECODE ONLY. A decode step reaches this per-request loop only
            // when the batched arm could not seat it — the live count ran past the widest
            // baked rung (the W=1 case is the OWNER and always seats). The one-forward-
            // per-request decode arm is DELETED, so say so rather than serve unpipelined.
            #[cfg(feature = "spyre-hw")]
            if solo_step_is_decode(n, start, input_batch.prompt_len(req_id)) {
                return Err(werr(format!(
                    "decode step for {req_id} ({start} computed) fell to the per-request loop — \
                     the live decode count ran past the widest baked rung, and the solo decode \
                     arm is removed. Bake a wider rung or admit fewer requests. Real batched \
                     decode only."
                )));
            }
            let req = requests
                .get_mut(req_id)
                .ok_or_else(|| werr(format!("unknown cached req {req_id}")))?;
            let s = run_request_step(model, req, req_id, input_batch, start, n)?;
            order.push(req_id.clone());
            sampled.push(s);
        }

        // One sampled token per request, in processing order; intermediate
        // prefill chunks (None) carry no token.
        let tokens: Vec<u32> = sampled.iter().map(|s| s.unwrap_or(0)).collect();
        let mut out = ModelRunnerOutput::from_ordered(order, tokens);
        for (idx, s) in sampled.iter().enumerate() {
            if s.is_none() {
                out.sampled_token_ids[idx].clear();
            }
        }
        Ok(out)
    }

    fn shutdown(&mut self) {
        // ⭐ #32 (c4) — NO DEFERRED STEP OUTLIVES THE WORKER: drain the queue so a resolver
        // still holding a cell gets an answer rather than a hang. Teardown reaches the drain
        // thread as a MESSAGE (`PipelineMsg::Shutdown`), which is why this runs there and not
        // on the client's drop.
        #[cfg(feature = "spyre-hw")]
        while let Some(infl) = self.in_flight.pop_front() {
            match self.model.as_mut() {
                Some(model) => {
                    let _ = Self::drain_one(infl, model, &mut self.requests, &mut self.input_batch);
                }
                None => infl
                    .cell
                    .fill(Err(scratchy_serving_engine::error::EngineError::Executor(
                        "shutdown: the model was dropped before this step landed".into(),
                    ))),
            }
        }
        self.is_shutdown = true;
    }

    /// ⭐ #32 (c4) — THE DOOR LANDS: this worker's steps may now hand their collect to the
    /// engine's drain thread. Every step still runs the same submit → defer → drain; the door
    /// only decides WHERE the drain runs.
    #[cfg(feature = "spyre-hw")]
    fn set_drain_door(&mut self, door: scratchy_serving_engine::core_client::DrainDoor) {
        self.drain_door = Some(door);
    }

    /// ⭐ #32 (c4) — A KNOCK: drain in-flight steps until the named cell lands. The knock is a
    /// deferred output's resolver asking — it runs on the engine thread while THIS thread owns
    /// the queue, which is why the answer is a cell and not a return value.
    #[cfg(feature = "spyre-hw")]
    fn drain_until(&mut self, cell: &std::sync::Arc<scratchy_serving_engine::executor::DrainCell>) {
        while !cell.is_filled() {
            let Some(infl) = self.in_flight.pop_front() else {
                // A verdict computed over nothing must FAIL, never invent tokens.
                cell.fill(Err(scratchy_serving_engine::error::EngineError::Executor(
                    "drain: the in-flight queue emptied before this step landed".into(),
                )));
                return;
            };
            let Some(model) = self.model.as_mut() else {
                let e = scratchy_serving_engine::error::EngineError::Executor(
                    "drain: the model is not loaded".into(),
                );
                infl.cell
                    .fill(Err(scratchy_serving_engine::error::EngineError::Executor(
                        "drain: the model is not loaded".into(),
                    )));
                cell.fill(Err(e));
                return;
            };
            if Self::drain_one(infl, model, &mut self.requests, &mut self.input_batch).is_err() {
                // The step's own cell carries its error to its resolver. A failed collect
                // leaves the stream sticky-dead, so every later step fails the same way —
                // stop rather than hand each one the same dead fence.
                break;
            }
        }
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
