// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Persistent `InputBatch` that maintains pre-allocated buffers across engine
//! steps, eliminating redundant allocations on the decode hot path.
//!
//! Port of the Python V1 `InputBatch` concept: delta-update a dense array of
//! per-request slots instead of rebuilding all model inputs from scratch every
//! step.

use std::collections::HashMap;

use scratchy_core_model::AttentionMetadata;
use scratchy_serving_scheduler::scheduler::output::SchedulerOutput;

// ---------------------------------------------------------------------------
// The per-slot conservation law
// ---------------------------------------------------------------------------

/// EVERY PER-SLOT VEC, NAMED ONCE — and this list is the only place a slot field may be introduced.
///
/// ⛔⛔⛔ WHY THIS IS A MACRO AND NOT SIX HAND-WRITTEN LINES ×3. `InputBatch` stores a slot as one
/// element in each of six parallel `Vec`s, so "slot `i` exists" is really the claim *all six have a
/// length greater than `i`, describing the same request*. Nothing in the type system said so: the push,
/// the swap and the pop were three hand-maintained lists of the same names, and adding another
/// field meant editing all three correctly or getting a store where one field is off by one slot — i.e.
/// one request reading another's block table, silently, with no length ever disagreeing about anything
/// the compiler can see.
///
/// That is the CONSERVATION-LAW family of defect, the one a newtype cannot catch because every field
/// here is a *different* quantity and each is already correctly typed. The law is not about any one
/// field's type; it is that they move TOGETHER.
///
/// So this list generates all three motions, plus a [`NewSlot`] whose struct literal the compiler
/// refuses to accept until every field has a value. Adding a per-slot field is now: add one line here,
/// and the build fails at `add_request_hybrid` until the new field is supplied. Forgetting the swap or
/// the pop is no longer expressible.
///
/// ⭐ The Struct-of-Arrays LAYOUT IS PRESERVED DELIBERATELY: `fast_path_info` hands the cuda graph path
/// `&[Vec<usize>]` and `&[usize]` directly, which an array-of-structs cannot do without allocating on
/// the decode hot path. The guard is about the motions, not the layout.
macro_rules! per_slot_fields {
    // `$(#[$m:meta])*` and not `#[doc = $d:expr]`: a two-line doc comment is TWO `#[doc]` attributes,
    // so the single-attribute form rejects any field documented in more than one line — which is every
    // field worth documenting.
    ( $( $(#[$m:meta])* $field:ident : $ty:ty ),+ $(,)? ) => {
        /// One slot's complete state, as ONE value — the argument to `InputBatch::push_slot`.
        ///
        /// Every field is required by the struct literal, which is the whole point: a new per-slot field
        /// cannot be added without every construction site being made to name it.
        struct NewSlot {
            $( $(#[$m])* $field: $ty, )+
        }

        impl InputBatch {
            /// Append one slot to EVERY per-slot vec. The only way a slot comes into existence.
            fn push_slot(&mut self, s: NewSlot) {
                $( self.$field.push(s.$field); )+
            }

            /// Exchange two slots in EVERY per-slot vec. Callers must still fix `req_id_to_slot`.
            fn swap_slots(&mut self, a: usize, b: usize) {
                $( self.$field.swap(a, b); )+
            }

            /// Drop the last slot from EVERY per-slot vec. The only way a slot ceases to exist.
            fn pop_slot(&mut self) {
                $( self.$field.pop(); )+
            }

            /// The lengths of every per-slot vec, for the test that they are all equal. A test rather
            /// than a runtime check on purpose: with all three motions generated from one list, an
            /// inequality is now only reachable by hand-editing one of the vecs, and the test is what
            /// notices if someone does.
            #[cfg(test)]
            fn per_slot_lens(&self) -> Vec<(&'static str, usize)> {
                vec![ $( (stringify!($field), self.$field.len()) ),+ ]
            }
        }
    };
}

per_slot_fields! {
    /// Request id — the slot's identity, and what `req_id_to_slot` maps to its index.
    req_ids: String,
    /// How many of this request's tokens are in the KV block pool. Every step
    /// [`InputBatch::prepare_inputs`] builds sets it to the scheduler's `num_computed_tokens`, and
    /// [`InputBatch::commit_step`] advances it past the step — what the cuda graph fast path and spyre
    /// read in between.
    tokens_in_pool: usize,
    /// Block table (full/global KV-cache group block ids), REPLACED wholesale each step.
    block_tables: Vec<usize>,
    /// Per-SLIDING-GROUP block tables (gemma4 SWA); empty on non-SWA requests.
    sliding_groups: Vec<Vec<usize>>,
    /// This request's PROMPT, whole. `prompt_len` is derived from it and never stored.
    prompt: Vec<u32>,
    /// Everything sampled since, in order. `prompt ++ generated` is the token history.
    generated: Vec<u32>,
    /// Tokens sampled by steps still on the device: they follow `prompt ++ generated`, and a step
    /// that reads one takes it from the device ([`PendingInput`]) until [`InputBatch::resolve`].
    in_flight: usize,
    /// Whether this request's latest step opened ([`InputBatch::open_step`]) and has yet to commit.
    step: StepState,
}

/// Where a request's latest step stands. A step opens before the worker runs it and closes when it
/// commits; one still open when the next opens never committed — it failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StepState {
    Committed,
    Open,
}

/// A step scheduled behind one that failed: the engine queued it on tokens the failed step never
/// produced, so it cannot run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailedPredecessor {
    pub req_id: String,
}

impl std::fmt::Display for FailedPredecessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: the step this one was scheduled behind failed",
            self.req_id
        )
    }
}

// ---------------------------------------------------------------------------
// InputBatch
// ---------------------------------------------------------------------------

/// Persistent per-request state that survives across engine steps, and the builder of each step's
/// flat model inputs.
///
/// Requests occupy dense slots (0..num_active). Finished requests are swap-removed with the last slot
/// to keep the array compact without gaps.
///
/// ⛔⛔ A SLOT IS NOT A SCHEDULED REQUEST. The scheduler leaves running requests out of a step — when
/// the token budget runs out, when async scheduling knows an in-flight step will finish one at
/// `max_tokens`, when it preempts one — and every such request keeps its slot (its history is needed
/// again). So WHAT a step runs is never read off the slots: [`Self::prepare_inputs`] walks the
/// [`SchedulerOutput`] and takes each request's chunk from its history at the scheduler's
/// `num_computed_tokens`, for exactly its `num_scheduled_tokens`. The step's token count is the
/// scheduler's `total_num_scheduled_tokens` by construction, and that is bounded by the budget the
/// engine sized to the largest resident bucket.
pub struct InputBatch {
    // --- Slot management ---
    /// req_id → slot index.
    req_id_to_slot: HashMap<String, usize>,
    /// Ordered req_ids, length == num_active.
    req_ids: Vec<String>,

    // --- Per-slot persistent state (indexed by slot) ---
    /// Tokens of each request already written to the KV block pool.
    tokens_in_pool: Vec<usize>,
    /// Block table per request (full/global KV-cache group block IDs).
    block_tables: Vec<Vec<usize>>,
    /// Per-request, per-SLIDING-GROUP block tables (vLLM group-shared layout):
    /// `sliding_groups[req][sliding_group]`. Empty inner vec on non-SWA reqs.
    sliding_groups: Vec<Vec<Vec<usize>>>,
    /// THE PROMPT, whole, per slot. Every length about it is derived from this — there is no
    /// `prompt_len` vec, because a stored length is a second store of one quantity.
    prompt: Vec<Vec<u32>>,
    /// Everything sampled since, in order, per slot. `prompt ++ generated` IS the token history — what
    /// every step reads its chunk from, and the samplers their seed position (and metal its penalty
    /// histories) — and [`Self::token_at`] is the one place the boundary arithmetic lives.
    generated: Vec<Vec<u32>>,
    /// Per slot, the tokens sampled by steps still on the device, which follow `generated`.
    in_flight: Vec<usize>,
    /// Per slot, whether the latest step opened and has yet to commit.
    step: Vec<StepState>,

    // --- Reusable per-step buffers (cleared + refilled each step) ---
    flat_token_ids: Vec<u32>,
    flat_positions: Vec<u32>,
    /// `(slot, num_computed_tokens, num_scheduled_tokens)` of this step's requests, in slot order.
    step_rows: Vec<(usize, usize, usize)>,

    // --- Reusable output buffers (swapped out in prepare_inputs, swapped back via reclaim) ---
    req_inputs_buf: Vec<ReqSlice>,
    query_start_loc_buf: Vec<usize>,
    q_lens_buf: Vec<usize>,
    seq_lens_buf: Vec<usize>,
    block_ids_buf: Vec<Vec<usize>>,
    tokens_before_buf: Vec<usize>,
    is_prefill_buf: Vec<bool>,
    req_ids_buf: Vec<String>,
}

impl Default for InputBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl InputBatch {
    /// Create an empty InputBatch.
    pub fn new() -> Self {
        Self {
            req_id_to_slot: HashMap::new(),
            req_ids: Vec::new(),
            tokens_in_pool: Vec::new(),
            block_tables: Vec::new(),
            sliding_groups: Vec::new(),
            prompt: Vec::new(),
            generated: Vec::new(),
            in_flight: Vec::new(),
            step: Vec::new(),
            flat_token_ids: Vec::new(),
            flat_positions: Vec::new(),
            step_rows: Vec::new(),
            req_inputs_buf: Vec::new(),
            query_start_loc_buf: Vec::new(),
            q_lens_buf: Vec::new(),
            seq_lens_buf: Vec::new(),
            block_ids_buf: Vec::new(),
            tokens_before_buf: Vec::new(),
            is_prefill_buf: Vec::new(),
            req_ids_buf: Vec::new(),
        }
    }

    /// Number of active requests.
    pub fn num_active(&self) -> usize {
        self.req_ids.len()
    }

    /// Apply one step's request lifecycle from the scheduler: drop the finished, admit the new (with
    /// their WHOLE prompt), and replace the block tables of the cached. Ported from Python
    /// `GPUModelRunner._update_states`.
    ///
    /// Preempted requests are NOT dropped: the scheduler resumes one by scheduling it again from its
    /// `num_computed_tokens` (0, or its prefix-cache hit) with its complete block tables, and its chunk
    /// is read from the history kept here. Until then it is simply not scheduled, so not run.
    pub fn update_states(&mut self, sched: &SchedulerOutput) {
        self.remove_finished(&sched.finished_req_ids);
        for new_req in &sched.scheduled_new_reqs {
            // Group 0 = full (global layers); groups 1.. = the sliding groups (gemma4 SWA).
            let (full, sliding) = new_req
                .block_ids
                .split_first()
                .map_or((Vec::new(), Vec::new()), |(f, s)| (f.clone(), s.to_vec()));
            self.add_request_hybrid(
                new_req.req_id.clone(),
                new_req.prompt_token_ids.as_deref().unwrap_or(&[]),
                full,
                sliding,
                new_req.num_computed_tokens,
            );
        }
        let cached = &sched.scheduled_cached_reqs;
        for (req_id, groups) in cached.req_ids.iter().zip(&cached.new_block_ids) {
            if let Some((full, sliding)) = groups.as_deref().and_then(<[_]>::split_first) {
                self.update_blocks_hybrid(req_id, full.clone(), sliding.to_vec());
            }
        }
    }

    /// Add a new request (single KV-cache group; non-SWA models). `prompt` is the WHOLE prompt.
    pub fn add_request(
        &mut self,
        req_id: String,
        prompt: &[u32],
        block_ids: Vec<usize>,
        num_computed_tokens: u32,
    ) {
        self.add_request_hybrid(req_id, prompt, block_ids, Vec::new(), num_computed_tokens);
    }

    /// Add a new request with its SLIDING KV-cache group block tables (gemma4
    /// SWA): `sliding_groups[s]` is sliding group `s`'s blocks for this request.
    /// Empty is identical to [`Self::add_request`]. `prompt` is the WHOLE prompt; a request already
    /// holding a slot under this id is replaced, never duplicated.
    pub fn add_request_hybrid(
        &mut self,
        req_id: String,
        prompt: &[u32],
        block_ids: Vec<usize>,
        sliding_groups: Vec<Vec<usize>>,
        num_computed_tokens: u32,
    ) {
        self.remove_request(&req_id);
        let slot = self.req_ids.len();
        self.req_id_to_slot.insert(req_id.clone(), slot);
        // ⛔ ONE LITERAL, EVERY FIELD, NO ORDER TO GET WRONG. Nothing here pushes to an individual vec:
        // `push_slot` is generated from `per_slot_fields!`, so a field added to that list stops this
        // literal from compiling until it is given a value.
        self.push_slot(NewSlot {
            req_ids: req_id,
            // When prefix caching supplies computed tokens, their KV is already in the pool.
            tokens_in_pool: num_computed_tokens as usize,
            block_tables: block_ids,
            sliding_groups,
            prompt: prompt.to_vec(),
            generated: Vec::new(),
            in_flight: 0,
            step: StepState::Committed,
        });
    }

    /// Open a step for every request `sched` runs tokens for — before the worker runs any of it. A
    /// request whose last step opened and never committed was scheduled behind a step that failed:
    /// refused, naming it, with no slot changed.
    pub fn open_step(&mut self, sched: &SchedulerOutput) -> Result<(), FailedPredecessor> {
        let slots: Vec<usize> = (sched.num_scheduled_tokens.iter())
            .filter(|&(_, &n)| n > 0)
            .filter_map(|(req_id, _)| self.req_id_to_slot.get(req_id).copied())
            .collect();
        if let Some(&slot) = slots.iter().find(|&&s| self.step[s] == StepState::Open) {
            let req_id = self.req_ids[slot].clone();
            return Err(FailedPredecessor { req_id });
        }
        for slot in slots {
            self.step[slot] = StepState::Open;
        }
        Ok(())
    }

    /// Remove a finished request (swap-remove to keep dense packing).
    pub fn remove_request(&mut self, req_id: &str) {
        let Some(slot) = self.req_id_to_slot.remove(req_id) else {
            return;
        };
        let last = self.req_ids.len() - 1;
        if slot != last {
            // Swap the last slot into the removed slot's position, in every per-slot vec at once.
            self.swap_slots(slot, last);
            // Then the ONE piece of slot state that is not per-slot-indexed: the id → slot map.
            let moved_req_id = self.req_ids[slot].clone();
            self.req_id_to_slot.insert(moved_req_id, slot);
        }
        // Drop the last slot (now the removed request) from every per-slot vec.
        self.pop_slot();
    }

    /// Remove all finished requests.
    pub fn remove_finished(&mut self, finished_req_ids: &std::collections::HashSet<String>) {
        for req_id in finished_req_ids {
            self.remove_request(req_id);
        }
    }

    /// Update block table for a cached request that got new blocks (single
    /// KV-cache group; non-SWA models).
    pub fn update_blocks(&mut self, req_id: &str, new_block_ids: Vec<usize>) {
        self.update_blocks_hybrid(req_id, new_block_ids, Vec::new());
    }

    /// Update block table for a cached request that got new blocks. The
    /// scheduler ships the FULL table each step, so this REPLACES (not
    /// appends). `sliding_new[s]` is sliding group `s`'s full table (empty on
    /// non-SWA models, leaving the slot's sliding tables untouched).
    pub fn update_blocks_hybrid(
        &mut self,
        req_id: &str,
        new_block_ids: Vec<usize>,
        sliding_new: Vec<Vec<usize>>,
    ) {
        if let Some(&slot) = self.req_id_to_slot.get(req_id) {
            self.block_tables[slot] = new_block_ids;
            if !sliding_new.is_empty() {
                self.sliding_groups[slot] = sliding_new;
            }
        }
    }

    /// Get the block table for a request.
    pub fn block_table(&self, req_id: &str) -> Option<&[usize]> {
        self.req_id_to_slot
            .get(req_id)
            .map(|&slot| self.block_tables[slot].as_slice())
    }

    /// Get the stored SLIDING KV-cache group block tables for a request
    /// (gemma4 SWA). Empty on uniform models.
    pub fn sliding_groups(&self, req_id: &str) -> Option<&[Vec<usize>]> {
        self.req_id_to_slot
            .get(req_id)
            .map(|&slot| self.sliding_groups[slot].as_slice())
    }

    /// SET how many of this request's tokens are in the KV pool — for a backend that advances it from
    /// inside its own forward rather than through [`Self::commit_step`].
    ///
    /// ⛔ THE SPYRE PATH DOES NOT CALL `commit_step`: it chunks a prefill itself and knows how far it got
    /// only after the chunk loop, so the count is written here and this is the ONE store of it. Without
    /// this setter the field kept its admission value forever on that path — a number in a shared store
    /// that was quietly wrong for one backend, which is how a second store starts.
    pub fn set_tokens_in_pool(&mut self, req_id: &str, n: usize) {
        if let Some(&slot) = self.req_id_to_slot.get(req_id) {
            self.tokens_in_pool[slot] = n;
        }
    }

    /// Get tokens-in-pool count for a request.
    pub fn tokens_in_pool_for(&self, req_id: &str) -> usize {
        self.req_id_to_slot
            .get(req_id)
            .map(|&slot| self.tokens_in_pool[slot])
            .unwrap_or(0)
    }

    /// Check if a request is known to this batch.
    pub fn contains(&self, req_id: &str) -> bool {
        self.req_id_to_slot.contains_key(req_id)
    }

    /// Lightweight query for the greedy graph fast path.
    ///
    /// Returns `(req_ids, block_tables, tokens_in_pool)` without building
    /// full `PreparedInputs`. This avoids the cost of `prepare_inputs` when
    /// the GPU self-updates all metadata.
    pub fn fast_path_info(&self) -> (&[String], &[Vec<usize>], &[usize]) {
        (&self.req_ids, &self.block_tables, &self.tokens_in_pool)
    }

    /// Whether this step is EVERY slot decoding exactly one token — the only step a replay over all
    /// slots ([`Self::fast_path_info`]) runs exactly. A slot the scheduler left out, a chunk, or a
    /// speculative verify makes it false.
    pub fn schedules_one_token_per_slot(&self, sched: &SchedulerOutput) -> bool {
        sched.num_scheduled_tokens.len() == self.req_ids.len()
            && self
                .req_ids
                .iter()
                .all(|id| sched.num_scheduled_tokens.get(id) == Some(&1))
    }

    /// Token counts for the super-fast graph path's `PendingCommit`.
    ///
    /// The super-fast path is always a pure decode batch (all q_len=1), so each
    /// request contributes exactly 1 input token. This MUST return `vec![1; n]`,
    /// NOT `tokens_in_pool` — using cumulative `tokens_in_pool` would cause
    /// exponential growth when later passed to `commit_step` as `input_token_count`.
    pub fn fast_path_token_counts(&self) -> Vec<usize> {
        vec![1; self.req_ids.len()]
    }

    /// Build this step's flat model inputs FROM THE SCHEDULE. Ported from Python
    /// `GPUModelRunner._prepare_inputs`.
    ///
    /// Every request the scheduler scheduled — and only those — is packed, in slot order. Its chunk is
    /// its history `prompt ++ generated` from the scheduler's `num_computed_tokens`, for its
    /// `num_scheduled_tokens` minus its draft tokens, followed by the drafts; positions run from
    /// `num_computed_tokens`. Prefill, chunked-prefill continuation, decode, speculative verify and a
    /// resume after preemption are all this one slice — there is no per-slot mode to fall out of step
    /// with the scheduler.
    ///
    /// # Panics
    /// If a scheduled request holds no slot, or its history is shorter than the chunk the scheduler
    /// scheduled — either means this batch lost track of a request the scheduler still runs.
    pub fn prepare_inputs(&mut self, sched: &SchedulerOutput) -> PreparedInputs {
        self.flat_token_ids.clear();
        self.flat_positions.clear();

        let cached = &sched.scheduled_cached_reqs;
        let scheduled = sched
            .scheduled_new_reqs
            .iter()
            .map(|r| (&r.req_id, r.num_computed_tokens))
            .chain(
                cached
                    .req_ids
                    .iter()
                    .zip(cached.num_computed_tokens.iter().copied()),
            );
        let mut rows = std::mem::take(&mut self.step_rows);
        rows.clear();
        for (req_id, num_computed) in scheduled {
            let slot = *self
                .req_id_to_slot
                .get(req_id)
                .unwrap_or_else(|| panic!("scheduled request {req_id} holds no InputBatch slot"));
            rows.push((
                slot,
                num_computed as usize,
                sched.num_scheduled_tokens[req_id],
            ));
        }
        rows.sort_unstable_by_key(|&(slot, _, _)| slot);
        let num_reqs = rows.len();

        // Reuse pre-allocated buffers (retain capacity across steps).
        let mut req_inputs = std::mem::take(&mut self.req_inputs_buf);
        req_inputs.clear();
        let mut query_start_loc = std::mem::take(&mut self.query_start_loc_buf);
        query_start_loc.clear();
        let mut q_lens = std::mem::take(&mut self.q_lens_buf);
        q_lens.clear();
        let mut seq_lens = std::mem::take(&mut self.seq_lens_buf);
        seq_lens.clear();
        let mut batch_block_ids = std::mem::take(&mut self.block_ids_buf);
        batch_block_ids.clear();
        let mut batch_tokens_before = std::mem::take(&mut self.tokens_before_buf);
        batch_tokens_before.clear();
        let mut is_prefill_vec = std::mem::take(&mut self.is_prefill_buf);
        is_prefill_vec.clear();
        let mut batch_req_ids = std::mem::take(&mut self.req_ids_buf);
        batch_req_ids.clear();
        let mut pending = Vec::new();

        let mut offset = 0usize;

        for &(slot, num_computed, num_scheduled) in &rows {
            let req_id = &self.req_ids[slot];
            let drafts = sched
                .scheduled_spec_decode_tokens
                .get(req_id)
                .map_or(&[][..], Vec::as_slice);
            let num_real = num_scheduled - drafts.len();
            let (prompt, generated) = (&self.prompt[slot], &self.generated[slot]);
            let end = num_computed + num_real;
            let token_start = self.flat_token_ids.len();
            // `prompt ++ generated` over `num_computed..end`, split at the prompt boundary; past the
            // known history, the in-flight tokens, which the device fills in. An index past those
            // panics: that chunk was never this batch's to run.
            let known = prompt.len() + generated.len();
            assert!(
                end <= known + self.in_flight[slot],
                "{req_id}: chunk ends at {end}, past its {known} known and {} in-flight tokens",
                self.in_flight[slot]
            );
            let in_prompt = num_computed.min(prompt.len())..end.min(prompt.len());
            let in_generated = num_computed.clamp(prompt.len(), known) - prompt.len()
                ..end.clamp(prompt.len(), known) - prompt.len();
            self.flat_token_ids.extend_from_slice(&prompt[in_prompt]);
            self.flat_token_ids
                .extend_from_slice(&generated[in_generated]);
            for position in num_computed.max(known)..end {
                pending.push(PendingInput {
                    flat_index: self.flat_token_ids.len(),
                    req_id: req_id.clone(),
                    position,
                });
                self.flat_token_ids.push(0);
            }
            self.flat_token_ids.extend_from_slice(drafts);
            self.flat_positions
                .extend(num_computed as u32..(num_computed + num_scheduled) as u32);

            query_start_loc.push(offset);
            q_lens.push(num_scheduled);
            seq_lens.push(num_computed + num_scheduled);
            batch_block_ids.push(self.block_tables[slot].clone());
            batch_tokens_before.push(num_computed);
            is_prefill_vec.push(num_computed < prompt.len());
            batch_req_ids.push(req_id.clone());
            req_inputs.push(ReqSlice {
                req_id: req_id.clone(),
                token_start,
                token_count: num_scheduled,
                spec_token_ids: drafts.to_vec(),
                // A chunk that stops short of the end of the history (a prefill chunk, or a resume
                // re-prefilling what it had generated) samples nothing — Python's discard mask.
                emits_token: end >= known + self.in_flight[slot],
            });
            self.tokens_in_pool[slot] = num_computed;
            offset += num_scheduled;
        }
        query_start_loc.push(offset);

        // Sliding KV-cache GROUPS (gemma4 SWA): transpose the per-request,
        // per-group tables to [sliding_group][req][block] in the same row
        // order as `batch_block_ids`. Empty on non-SWA models.
        let num_sliding_groups = rows
            .iter()
            .map(|&(slot, _, _)| self.sliding_groups[slot].len())
            .max()
            .unwrap_or(0);
        let sliding_groups: Vec<Vec<Vec<usize>>> = (0..num_sliding_groups)
            .map(|g| {
                rows.iter()
                    .map(|&(slot, _, _)| {
                        self.sliding_groups[slot]
                            .get(g)
                            .cloned()
                            .unwrap_or_default()
                    })
                    .collect::<Vec<Vec<usize>>>()
            })
            .collect();
        self.step_rows = rows;

        let attn_meta = AttentionMetadata::new(
            num_reqs,
            offset,
            query_start_loc,
            q_lens,
            seq_lens,
            batch_block_ids,
            batch_tokens_before,
            is_prefill_vec,
            batch_req_ids,
        )
        .with_sliding_groups(sliding_groups);

        PreparedInputs {
            req_inputs,
            flat_token_ids: std::mem::take(&mut self.flat_token_ids),
            flat_positions: std::mem::take(&mut self.flat_positions),
            attn_meta,
            pending,
        }
    }

    /// Commit step results after sampling: advance the pool cursor past the step's KV and append the
    /// emitted tokens to the history.
    ///
    /// `sampled_tokens` is what the step emitted for this request — empty for a chunk that emits
    /// nothing ([`ReqSlice::emits_token`]).
    pub fn commit_step(
        &mut self,
        req_id: &str,
        sampled_tokens: &[u32],
        input_token_count: usize,
        was_spec_decode: bool,
    ) {
        let Some(&slot) = self.req_id_to_slot.get(req_id) else {
            return;
        };
        self.tokens_in_pool[slot] += if was_spec_decode {
            // Spec decode: only accepted tokens get cached (input tokens up to
            // the first rejection). `sampled_tokens.len()` == num_accepted + 1
            // which also equals the number of input tokens correctly in cache
            // (last_token + accepted drafts).
            sampled_tokens.len()
        } else {
            input_token_count
        };
        self.generated[slot].extend_from_slice(sampled_tokens);
        self.step[slot] = StepState::Committed;
    }

    /// Commit a step that emitted one token the host does not have yet: advance the pool cursor past
    /// its `input_token_count` tokens, and count its token in flight until [`Self::resolve`].
    pub fn commit_in_flight(&mut self, req_id: &str, input_token_count: usize) {
        let Some(&slot) = self.req_id_to_slot.get(req_id) else {
            return;
        };
        self.tokens_in_pool[slot] += input_token_count;
        self.in_flight[slot] += 1;
        self.step[slot] = StepState::Committed;
    }

    /// Commit a speculative step whose tokens the host does not have yet — its own and every draft
    /// it keeps, `rows` at most: advance the pool cursor past its `input_token_count` tokens, and
    /// count `rows` in flight until [`Self::resolve_rows`].
    pub fn commit_in_flight_rows(&mut self, req_id: &str, input_token_count: usize, rows: usize) {
        let Some(&slot) = self.req_id_to_slot.get(req_id) else {
            return;
        };
        self.tokens_in_pool[slot] += input_token_count;
        self.in_flight[slot] += rows;
        self.step[slot] = StepState::Committed;
    }

    /// The oldest in-flight speculative step of `req_id`, now on the host: `tokens` of its `rows`.
    /// Append them to the history; the rows it did not keep leave the pool cursor.
    pub fn resolve_rows(&mut self, req_id: &str, tokens: &[u32], rows: usize) {
        let Some(&slot) = self.req_id_to_slot.get(req_id) else {
            return;
        };
        assert!(
            self.in_flight[slot] >= rows && tokens.len() <= rows,
            "{req_id}: resolved {} tokens of {rows} rows with {} in flight",
            tokens.len(),
            self.in_flight[slot]
        );
        self.in_flight[slot] -= rows;
        self.tokens_in_pool[slot] -= rows - tokens.len();
        self.generated[slot].extend_from_slice(tokens);
    }

    /// The oldest in-flight token of `req_id`, now on the host: append it to the history.
    pub fn resolve(&mut self, req_id: &str, token: u32) {
        let Some(&slot) = self.req_id_to_slot.get(req_id) else {
            return;
        };
        assert!(
            self.in_flight[slot] > 0,
            "{req_id}: resolved a token it had no step in flight for"
        );
        self.in_flight[slot] -= 1;
        self.generated[slot].push(token);
    }

    /// Where this request's prompt ends — the position sampling starts at. DERIVED, never stored.
    pub fn prompt_len(&self, req_id: &str) -> usize {
        self.req_id_to_slot
            .get(req_id)
            .map_or(0, |&slot| self.prompt[slot].len())
    }

    /// How many tokens this request has: prompt ++ generated. DERIVED.
    pub fn token_count(&self, req_id: &str) -> usize {
        self.req_id_to_slot.get(req_id).map_or(0, |&slot| {
            self.prompt[slot].len() + self.generated[slot].len()
        })
    }

    /// This request's history as `(prompt, generated)` — empty for a request this batch does not hold.
    pub fn history(&self, req_id: &str) -> (&[u32], &[u32]) {
        self.req_id_to_slot.get(req_id).map_or((&[], &[]), |&slot| {
            (
                self.prompt[slot].as_slice(),
                self.generated[slot].as_slice(),
            )
        })
    }

    /// How many tokens this request has generated, those still on the device included.
    pub fn num_generated(&self, req_id: &str) -> usize {
        self.req_id_to_slot
            .get(req_id)
            .map_or(0, |&slot| self.generated[slot].len() + self.in_flight[slot])
    }

    /// One token by ABSOLUTE position in `prompt ++ generated`.
    ///
    /// The two vecs are one sequence; which of them a position falls in is arithmetic done here rather
    /// than by every caller, because a caller that gets the boundary wrong reads a plausible token from
    /// the wrong half and nothing downstream can tell.
    pub fn token_at(&self, req_id: &str, pos: usize) -> Option<u32> {
        let &slot = self.req_id_to_slot.get(req_id)?;
        let p = &self.prompt[slot];
        match pos.checked_sub(p.len()) {
            None => p.get(pos).copied(),
            Some(off) => self.generated[slot].get(off).copied(),
        }
    }

    /// Append one sampled token to the history — for a backend that samples outside
    /// [`Self::commit_step`].
    pub fn push_generated(&mut self, req_id: &str, token_id: u32) {
        if let Some(&slot) = self.req_id_to_slot.get(req_id) {
            self.generated[slot].push(token_id);
        }
    }

    /// Reclaim reusable buffers from a consumed `PreparedInputs`.
    ///
    /// Call this at the end of `execute_model` to return Vec capacity back to
    /// `InputBatch`, avoiding heap allocations on the next `prepare_inputs`.
    pub fn reclaim_buffers(&mut self, prepared: PreparedInputs) {
        self.flat_token_ids = prepared.flat_token_ids;
        self.flat_positions = prepared.flat_positions;
        self.req_inputs_buf = prepared.req_inputs;
        // Reclaim AttentionMetadata buffers.
        let meta = prepared.attn_meta;
        self.query_start_loc_buf = meta.query_start_loc;
        self.q_lens_buf = meta.q_lens;
        self.seq_lens_buf = meta.seq_lens;
        self.block_ids_buf = meta.block_ids;
        self.tokens_before_buf = meta.tokens_before;
        self.is_prefill_buf = meta.is_prefill;
        self.req_ids_buf = meta.req_ids;
    }
}

// ---------------------------------------------------------------------------
// PreparedInputs — output of prepare_inputs()
// ---------------------------------------------------------------------------

/// Output of `InputBatch::prepare_inputs()`.
///
/// All data is owned so the caller can freely borrow other fields of the
/// parent struct while using these results.
pub struct PreparedInputs {
    /// Per-request slicing info.
    pub req_inputs: Vec<ReqSlice>,
    /// Flat token IDs for all requests (moved from InputBatch).
    pub flat_token_ids: Vec<u32>,
    /// Flat positions for all requests (moved from InputBatch).
    pub flat_positions: Vec<u32>,
    /// Attention metadata (also owns block_ids and tokens_before).
    pub attn_meta: AttentionMetadata,
    /// The input tokens still on the device, in flat order: their `flat_token_ids` entries are 0
    /// until the device writes each from the step that sampled it.
    pub pending: Vec<PendingInput>,
}

/// One input token a step reads before the host has it: sampled by a step still on the device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingInput {
    /// Its index in `flat_token_ids`.
    pub flat_index: usize,
    pub req_id: String,
    /// Its position in the request's `prompt ++ generated`.
    pub position: usize,
}

/// Per-request slice info within the flat tensors.
pub struct ReqSlice {
    /// Request ID.
    pub req_id: String,
    /// Start index in flat_token_ids / flat_positions.
    pub token_start: usize,
    /// Number of tokens for this request — its `num_scheduled_tokens`.
    pub token_count: usize,
    /// Speculative decode draft tokens (empty for normal decode/prefill).
    pub spec_token_ids: Vec<u32>,
    /// Whether this step samples a token for the request: its chunk reaches the end of its history.
    /// False for a chunk of a longer prefill (or of a resume's re-prefill), whose sample is discarded.
    pub emits_token: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use scratchy_serving_scheduler::scheduler::output::NewRequestData;

    use crate::spec_decode::greedy_rejection_sample;

    use super::*;

    // -----------------------------------------------------------------------
    // What the scheduler sends
    // -----------------------------------------------------------------------

    /// The `SchedulerOutput` of a step that runs `rows` — `(req_id, num_computed_tokens,
    /// num_scheduled_tokens)` — as requests the scheduler has scheduled before.
    fn step(rows: &[(&str, u32, usize)]) -> SchedulerOutput {
        let mut s = SchedulerOutput::make_empty();
        for &(id, num_computed, num_scheduled) in rows {
            let c = &mut s.scheduled_cached_reqs;
            c.req_ids.push(id.into());
            c.new_block_ids.push(None);
            c.num_computed_tokens.push(num_computed);
            c.num_output_tokens.push(0);
            s.num_scheduled_tokens.insert(id.into(), num_scheduled);
            s.total_num_scheduled_tokens += num_scheduled;
        }
        s
    }

    /// `s` admitting `req_id` for the first time, scheduling `num_scheduled` of its `prompt` from
    /// `num_computed` (its prefix-cache hit).
    fn admitting(
        mut s: SchedulerOutput,
        req_id: &str,
        prompt: &[u32],
        blocks: Vec<usize>,
        num_computed: u32,
        num_scheduled: usize,
    ) -> SchedulerOutput {
        s.scheduled_new_reqs.push(NewRequestData::new(
            req_id.into(),
            Some(prompt.to_vec()),
            vec![blocks],
            num_computed,
            None,
            None,
            None,
        ));
        s.num_scheduled_tokens.insert(req_id.into(), num_scheduled);
        s.total_num_scheduled_tokens += num_scheduled;
        s
    }

    /// `s` carrying `drafts` as `req_id`'s speculative tokens (counted in its `num_scheduled_tokens`).
    fn drafted(mut s: SchedulerOutput, req_id: &str, drafts: &[u32]) -> SchedulerOutput {
        s.scheduled_spec_decode_tokens
            .insert(req_id.into(), drafts.to_vec());
        s
    }

    /// `s` resuming `req_id` (scheduled in `s`) onto its complete new block table.
    fn resumed(mut s: SchedulerOutput, req_id: &str, blocks: Vec<usize>) -> SchedulerOutput {
        let c = &mut s.scheduled_cached_reqs;
        let i = c.req_ids.iter().position(|r| r == req_id).unwrap();
        c.resumed_req_ids.insert(req_id.into());
        c.new_block_ids[i] = Some(vec![blocks]);
        s
    }

    /// `s` preempting `ids`.
    fn preempting(mut s: SchedulerOutput, ids: &[&str]) -> SchedulerOutput {
        s.preempted_req_ids = Some(ids.iter().map(|&id| id.into()).collect());
        s
    }

    /// `s` finishing `ids`.
    fn finishing(mut s: SchedulerOutput, ids: &[&str]) -> SchedulerOutput {
        s.finished_req_ids = ids.iter().map(|&id| id.into()).collect();
        s
    }

    /// The row `req_id` was packed at.
    fn row(p: &PreparedInputs, req_id: &str) -> usize {
        p.attn_meta
            .req_ids
            .iter()
            .position(|r| r == req_id)
            .unwrap()
    }

    /// Run one step the way a worker does: lifecycle, build, commit what each row emitted
    /// (`sampled(req_id)`), reclaim. Returns what was packed as `(req_id, tokens, positions)`.
    fn run(
        b: &mut InputBatch,
        s: &SchedulerOutput,
        sampled: impl Fn(&str) -> Vec<u32>,
    ) -> Vec<(String, Vec<u32>, Vec<u32>)> {
        b.update_states(s);
        let p = b.prepare_inputs(s);
        let mut packed = Vec::new();
        for r in &p.req_inputs {
            let span = r.token_start..r.token_start + r.token_count;
            packed.push((
                r.req_id.clone(),
                p.flat_token_ids[span.clone()].to_vec(),
                p.flat_positions[span].to_vec(),
            ));
            let out = if r.emits_token {
                sampled(&r.req_id)
            } else {
                Vec::new()
            };
            b.commit_step(&r.req_id, &out, r.token_count, !r.spec_token_ids.is_empty());
        }
        b.reclaim_buffers(p);
        packed
    }

    // -----------------------------------------------------------------------
    // Slot mechanics
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_and_remove_request() {
        let mut batch = InputBatch::new();
        assert_eq!(batch.num_active(), 0);

        batch.add_request("r1".into(), &[10, 20, 30], vec![0, 1], 0);
        assert_eq!(batch.num_active(), 1);
        assert!(batch.contains("r1"));
        assert_eq!(batch.block_table("r1"), Some(&[0, 1][..]));

        batch.add_request("r2".into(), &[40, 50], vec![2], 0);
        assert_eq!(batch.num_active(), 2);

        batch.remove_request("r1");
        assert_eq!(batch.num_active(), 1);
        assert!(!batch.contains("r1"));
        assert!(batch.contains("r2"));
        // r2 should have been swapped into slot 0.
        assert_eq!(batch.block_table("r2"), Some(&[2][..]));

        batch.remove_request("r2");
        assert_eq!(batch.num_active(), 0);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10], vec![0], 0);
        batch.remove_request("nonexistent"); // should not panic
        assert_eq!(batch.num_active(), 1);
    }

    #[test]
    fn test_swap_remove_preserves_mapping() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10], vec![0], 0);
        batch.add_request("r2".into(), &[20], vec![1], 0);
        batch.add_request("r3".into(), &[30], vec![2], 0);

        // Remove r1 (slot 0) — r3 should move to slot 0.
        batch.remove_request("r1");
        assert_eq!(batch.num_active(), 2);
        assert!(batch.contains("r2"));
        assert!(batch.contains("r3"));
        assert_eq!(batch.block_table("r3"), Some(&[2][..]));
        assert_eq!(batch.block_table("r2"), Some(&[1][..]));
    }

    #[test]
    fn test_update_blocks() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10], vec![0], 0);
        assert_eq!(batch.block_table("r1"), Some(&[0][..]));

        batch.update_blocks("r1", vec![0, 1, 2]);
        assert_eq!(batch.block_table("r1"), Some(&[0, 1, 2][..]));
    }

    #[test]
    fn test_remove_finished() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10], vec![0], 0);
        batch.add_request("r2".into(), &[20], vec![1], 0);
        batch.add_request("r3".into(), &[30], vec![2], 0);

        let mut finished = std::collections::HashSet::new();
        finished.insert("r1".to_string());
        finished.insert("r3".to_string());

        batch.remove_finished(&finished);
        assert_eq!(batch.num_active(), 1);
        assert!(batch.contains("r2"));
    }

    /// Removing the LAST active request leaves an empty batch.
    #[test]
    fn test_remove_last_request_leaves_empty_batch() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[1, 2], vec![0], 0);
        batch.remove_request("r1");
        assert_eq!(batch.num_active(), 0);
        assert!(!batch.contains("r1"));
    }

    /// Removing the FIRST of N requests triggers swap-remove; all surviving
    /// requests must remain slot-consistent and individually removable.
    #[test]
    fn test_remove_first_of_many_slot_consistency() {
        let mut batch = InputBatch::new();
        for i in 0u32..5 {
            batch.add_request(format!("r{i}"), &[i], vec![i as usize], 0);
        }
        // Remove r0 (slot 0); r4 should swap into slot 0.
        batch.remove_request("r0");
        assert_eq!(batch.num_active(), 4);
        for i in 1u32..5 {
            let id = format!("r{i}");
            assert!(batch.contains(&id));
            assert_eq!(batch.block_table(&id), Some(&[i as usize][..]));
        }
        for i in 1u32..5 {
            batch.remove_request(&format!("r{i}"));
        }
        assert_eq!(batch.num_active(), 0);
    }

    /// Removing a middle request: slot indices of all other requests must
    /// remain valid (no off-by-one errors from the swap-remove).
    #[test]
    fn test_remove_middle_request_slot_consistency() {
        let mut batch = InputBatch::new();
        batch.add_request("a".into(), &[1], vec![10], 0);
        batch.add_request("b".into(), &[2], vec![20], 0);
        batch.add_request("c".into(), &[3], vec![30], 0);
        batch.add_request("d".into(), &[4], vec![40], 0);

        // Remove "b" (slot 1). "d" (last slot = 3) should fill slot 1.
        batch.remove_request("b");
        assert_eq!(batch.num_active(), 3);
        assert!(!batch.contains("b"));
        assert_eq!(batch.block_table("a"), Some(&[10][..]));
        assert_eq!(batch.block_table("c"), Some(&[30][..]));
        assert_eq!(batch.block_table("d"), Some(&[40][..]));
    }

    /// commit_step must be a no-op for a removed request. This guards against stale slot reuse when
    /// another request gets the same slot index via swap-remove.
    #[test]
    fn test_commit_step_noop_for_removed_request() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30], vec![0], 0);
        batch.add_request("r2".into(), &[40, 50], vec![1], 0);
        run(&mut batch, &step(&[("r1", 0, 3), ("r2", 0, 2)]), |_| {
            vec![9]
        });

        batch.remove_request("r1");

        let tip_before = batch.tokens_in_pool_for("r2");
        batch.commit_step("r1", &[0], 99, false); // r1 no longer in batch
        assert_eq!(
            batch.tokens_in_pool_for("r2"),
            tip_before,
            "stale commit_step must not corrupt surviving requests"
        );
        assert_eq!(batch.token_count("r2"), 3, "nor their history");
    }

    /// A request admitted under an id that already holds a slot REPLACES that slot: no duplicate
    /// slot, no stale history, blocks or pool cursor.
    #[test]
    fn test_readmitted_req_id_replaces_its_slot() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[1, 2, 3], vec![0, 1], 0);
        run(&mut batch, &step(&[("r1", 0, 3)]), |_| vec![99]);
        for n in 3..8 {
            run(&mut batch, &step(&[("r1", n, 1)]), |_| vec![0]);
        }
        assert_eq!(batch.tokens_in_pool_for("r1"), 8);

        batch.add_request("r1".into(), &[10, 20], vec![5], 0);
        assert_eq!(batch.num_active(), 1, "one slot per id");
        assert_eq!(batch.tokens_in_pool_for("r1"), 0);
        assert_eq!(batch.block_table("r1"), Some(&[5][..]));
        assert_eq!(batch.history("r1"), (&[10u32, 20][..], &[][..]));
    }

    // -----------------------------------------------------------------------
    // The step is the schedule
    // -----------------------------------------------------------------------

    /// THE #146 OVERFLOW. A running request the scheduler left out of a full-budget step (the async
    /// max-tokens skip, the budget running out, a preemption) is NOT packed: the step carries exactly
    /// `total_num_scheduled_tokens`, so it can never grow past the budget — the 4096 → 4097
    /// `NoBucketFits` panic.
    #[test]
    fn a_request_the_scheduler_left_out_is_not_packed() {
        let mut batch = InputBatch::new();
        batch.add_request("stale".into(), &[1, 2], vec![0], 0);
        run(&mut batch, &step(&[("stale", 0, 2)]), |_| vec![7]);
        let big: Vec<u32> = (100..104).collect();
        let s = admitting(SchedulerOutput::make_empty(), "big", &big, vec![1], 0, 4);

        batch.update_states(&s);
        let p = batch.prepare_inputs(&s);
        assert_eq!(p.attn_meta.total_tokens, s.total_num_scheduled_tokens);
        assert_eq!(p.attn_meta.req_ids, vec!["big".to_string()]);
        assert_eq!(p.flat_token_ids, big);
        assert_eq!(p.attn_meta.query_start_loc, vec![0, 4]);
        batch.reclaim_buffers(p);

        // The skipped request is untouched: when it is scheduled again it decodes its last token.
        assert_eq!(batch.tokens_in_pool_for("stale"), 2);
        let packed = run(&mut batch, &step(&[("stale", 2, 1)]), |_| vec![8]);
        assert_eq!(packed, vec![("stale".into(), vec![7], vec![2])]);
    }

    /// Every step packs exactly what was scheduled, whatever mix of prefill, chunk and decode — and
    /// the slots it leaves out contribute nothing.
    #[test]
    fn the_packed_token_count_is_the_scheduled_token_count() {
        let mut batch = InputBatch::new();
        for (i, id) in ["a", "b", "c", "d"].iter().enumerate() {
            batch.add_request((*id).into(), &[1; 10], vec![i], 0);
        }
        let schedules = [
            step(&[("a", 0, 10), ("b", 0, 4), ("c", 0, 10)]),
            step(&[("a", 10, 1), ("b", 4, 6), ("d", 0, 3)]),
            step(&[("b", 10, 1), ("d", 3, 7)]),
            step(&[("a", 11, 1), ("c", 10, 1)]),
        ];
        for s in &schedules {
            batch.update_states(s);
            let p = batch.prepare_inputs(s);
            assert_eq!(p.attn_meta.total_tokens, s.total_num_scheduled_tokens);
            assert_eq!(p.flat_token_ids.len(), s.total_num_scheduled_tokens);
            assert_eq!(p.attn_meta.num_reqs, s.num_scheduled_tokens.len());
            for r in &p.req_inputs {
                assert_eq!(r.token_count, s.num_scheduled_tokens[&r.req_id]);
            }
            for r in 0..p.attn_meta.num_reqs {
                let (id, q) = (&p.attn_meta.req_ids[r], p.attn_meta.q_lens[r]);
                let out = if p.req_inputs[r].emits_token {
                    vec![5]
                } else {
                    vec![]
                };
                batch.commit_step(id, &out, q, false);
            }
            batch.reclaim_buffers(p);
        }
    }

    /// Rows come out in SLOT order, whatever order the scheduler listed them in — the order a replay
    /// over every slot (the cuda graph fast path) assumes.
    #[test]
    fn rows_are_in_slot_order() {
        let mut batch = InputBatch::new();
        for (i, id) in ["a", "b", "c"].iter().enumerate() {
            batch.add_request((*id).into(), &[1, 2], vec![i], 0);
        }
        let p = batch.prepare_inputs(&step(&[("c", 0, 2), ("a", 0, 2), ("b", 0, 2)]));
        assert_eq!(p.attn_meta.req_ids, vec!["a", "b", "c"]);
    }

    /// A scheduled request with no slot is a lost request, not an empty row.
    #[test]
    #[should_panic(expected = "holds no InputBatch slot")]
    fn a_scheduled_request_without_a_slot_panics() {
        let mut batch = InputBatch::new();
        let _ = batch.prepare_inputs(&step(&[("ghost", 0, 1)]));
    }

    /// A chunk the history does not hold was never this batch's to run.
    #[test]
    #[should_panic]
    fn a_chunk_past_the_history_panics() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], vec![0], 0);
        let _ = batch.prepare_inputs(&step(&[("r1", 7, 10)]));
    }

    /// `update_states` is the lifecycle: finished requests go, new ones come in with their whole
    /// prompt, cached ones get their block tables replaced — and a preempted one keeps its slot.
    #[test]
    fn update_states_admits_drops_replaces_blocks_and_keeps_the_preempted() {
        let mut batch = InputBatch::new();
        batch.add_request("done".into(), &[1], vec![0], 0);
        batch.add_request("running".into(), &[2, 3], vec![1], 0);
        batch.push_generated("running", 9);
        batch.add_request("preempted".into(), &[4], vec![2], 0);

        let mut s = step(&[("running", 2, 1)]);
        s.scheduled_cached_reqs.new_block_ids[0] = Some(vec![vec![1, 7]]);
        let s = preempting(
            finishing(admitting(s, "new", &[5, 6, 7], vec![3], 1, 2), &["done"]),
            &["preempted"],
        );
        batch.update_states(&s);

        assert!(!batch.contains("done"));
        assert!(
            batch.contains("preempted"),
            "a preempted request keeps its history"
        );
        assert_eq!(batch.block_table("running"), Some(&[1, 7][..]));
        assert_eq!(batch.history("new"), (&[5u32, 6, 7][..], &[][..]));
        assert_eq!(batch.tokens_in_pool_for("new"), 1);
        let p = batch.prepare_inputs(&s);
        assert_eq!(p.attn_meta.req_ids, vec!["running", "new"]);
        assert_eq!(
            p.flat_token_ids,
            &[9, 6, 7],
            "the new request's chunk skips its cached prefix"
        );
    }

    /// Only a step of every slot decoding one token is a step a replay over all slots runs exactly.
    #[test]
    fn schedules_one_token_per_slot_only_when_every_slot_decodes_one() {
        let mut batch = InputBatch::new();
        batch.add_request("a".into(), &[1, 2], vec![0], 0);
        batch.add_request("b".into(), &[1, 2], vec![1], 0);
        assert!(batch.schedules_one_token_per_slot(&step(&[("a", 2, 1), ("b", 2, 1)])));
        assert!(
            !batch.schedules_one_token_per_slot(&step(&[("a", 2, 1)])),
            "a slot left out"
        );
        assert!(
            !batch.schedules_one_token_per_slot(&step(&[("a", 2, 1), ("b", 0, 2)])),
            "a chunk"
        );
        assert!(
            !batch.schedules_one_token_per_slot(&step(&[("a", 2, 1), ("b", 2, 1), ("c", 0, 1)])),
            "a request with no slot"
        );
    }

    // -----------------------------------------------------------------------
    // Prefill, chunked prefill, decode
    // -----------------------------------------------------------------------

    #[test]
    fn test_prepare_inputs_prefill() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30], vec![0, 1], 0);

        let prepared = batch.prepare_inputs(&step(&[("r1", 0, 3)]));

        assert_eq!(prepared.flat_token_ids, &[10, 20, 30]);
        assert_eq!(prepared.flat_positions, &[0, 1, 2]);
        assert_eq!(prepared.req_inputs.len(), 1);
        assert_eq!(prepared.req_inputs[0].token_count, 3);
        assert!(prepared.req_inputs[0].emits_token);
        assert!(prepared.attn_meta.is_prefill[0]);
        assert_eq!(prepared.attn_meta.num_reqs, 1);
        assert_eq!(prepared.attn_meta.total_tokens, 3);
    }

    /// A prefix-cache hit: the chunk is the prompt's suffix past the scheduler's `num_computed`, at
    /// its own positions — and the prompt is still the whole prompt.
    #[test]
    fn test_prepare_inputs_prefill_with_offset() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[1, 2, 3, 4, 5, 10, 20, 30], vec![0], 5);

        let prepared = batch.prepare_inputs(&step(&[("r1", 5, 3)]));

        assert_eq!(prepared.flat_token_ids, &[10, 20, 30]);
        assert_eq!(prepared.flat_positions, &[5, 6, 7]);
        assert_eq!(prepared.attn_meta.tokens_before[0], 5);
        assert_eq!(batch.prompt_len("r1"), 8);
    }

    #[test]
    fn test_prefill_to_decode_transition() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30], vec![0, 1], 0);

        let prepared = batch.prepare_inputs(&step(&[("r1", 0, 3)]));
        assert_eq!(prepared.req_inputs[0].token_count, 3);
        assert!(prepared.attn_meta.is_prefill[0]);
        batch.reclaim_buffers(prepared);
        batch.commit_step("r1", &[99], 3, false);

        let prepared = batch.prepare_inputs(&step(&[("r1", 3, 1)]));
        assert_eq!(prepared.flat_token_ids, &[99]);
        assert_eq!(prepared.req_inputs[0].token_count, 1);
        assert!(!prepared.attn_meta.is_prefill[0]);
        assert_eq!(prepared.attn_meta.tokens_before[0], 3);
    }

    /// A decode step whose input token is still on the device packs a placeholder the device fills,
    /// emits, and leaves the history to the resolve — which appends tokens in the order they came.
    #[test]
    fn an_in_flight_token_is_a_pending_input_until_it_resolves() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20], vec![0], 0);
        let p = batch.prepare_inputs(&step(&[("r1", 0, 2)]));
        assert!(p.pending.is_empty() && p.req_inputs[0].emits_token);
        batch.reclaim_buffers(p);
        batch.commit_in_flight("r1", 2);

        // Position 2 is the prefill's token, still on the device.
        let p = batch.prepare_inputs(&step(&[("r1", 2, 1)]));
        let at_2 = PendingInput {
            flat_index: 0,
            req_id: "r1".into(),
            position: 2,
        };
        assert_eq!(
            (p.flat_token_ids.clone(), p.pending.clone()),
            (vec![0], vec![at_2])
        );
        assert!(p.req_inputs[0].emits_token);
        assert_eq!(p.flat_positions, vec![2]);
        batch.reclaim_buffers(p);
        batch.commit_in_flight("r1", 1);

        // Two in flight; the next step reads only the newer one, at position 3.
        let p = batch.prepare_inputs(&step(&[("r1", 3, 1)]));
        assert_eq!(
            p.pending.iter().map(|x| x.position).collect::<Vec<_>>(),
            vec![3]
        );
        batch.reclaim_buffers(p);
        // An unseeded sampler seeds from this count: in flight or resolved, the same.
        assert_eq!(batch.num_generated("r1"), 2);
        batch.resolve("r1", 30);
        batch.resolve("r1", 40);
        assert_eq!(batch.num_generated("r1"), 2);
        assert_eq!(batch.history("r1"), (&[10u32, 20][..], &[30u32, 40][..]));
        assert_eq!(batch.tokens_in_pool_for("r1"), 3);

        // Resolved, the same step reads its token from the history.
        let p = batch.prepare_inputs(&step(&[("r1", 3, 1)]));
        assert_eq!((p.flat_token_ids.clone(), p.pending.len()), (vec![40], 0));
    }

    /// A chunk past the in-flight tokens too was never this batch's to run.
    #[test]
    #[should_panic(expected = "in-flight tokens")]
    fn a_chunk_past_the_in_flight_tokens_panics() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20], vec![0], 0);
        batch.commit_in_flight("r1", 2);
        let _ = batch.prepare_inputs(&step(&[("r1", 3, 1)]));
    }

    /// The pipelined engine queues a request's decode behind its last prefill chunk, on the
    /// chunk's token. When that chunk fails before it commits (here: a device out-of-memory
    /// surfacing as the step resolves the one before it), the decode is refused at open — the
    /// request named, nothing changed — instead of reaching `prepare_inputs` past every token the
    /// batch will ever have, the panic above. A request whose steps commit opens every time.
    #[test]
    fn a_step_queued_behind_a_failed_one_is_refused_at_open() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30], vec![0], 0);
        batch.add_request("r2".into(), &[40], vec![1], 0);

        // First chunk: opens, runs, commits.
        let first = step(&[("r1", 0, 2)]);
        assert_eq!(batch.open_step(&first), Ok(()));
        let p = batch.prepare_inputs(&first);
        batch.reclaim_buffers(p);
        batch.commit_step("r1", &[], 2, false);

        // Last chunk: opens, then fails before it prepares or commits.
        assert_eq!(batch.open_step(&step(&[("r1", 2, 1)])), Ok(()));

        // The decode the engine queued behind it, on the token it never sampled.
        let decode = step(&[("r1", 3, 1), ("r2", 0, 1)]);
        let refused = FailedPredecessor {
            req_id: "r1".into(),
        };
        assert_eq!(batch.open_step(&decode), Err(refused));
        assert_eq!(batch.tokens_in_pool_for("r1"), 2);

        // The other request, alone, still runs.
        let other = step(&[("r2", 0, 1)]);
        assert_eq!(batch.open_step(&other), Ok(()));
        let p = batch.prepare_inputs(&other);
        batch.reclaim_buffers(p);
        batch.commit_in_flight("r2", 1);
        assert_eq!(batch.open_step(&step(&[("r2", 1, 1)])), Ok(()));
    }

    #[test]
    fn test_decode_multiple_steps() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20], vec![0], 0);

        run(&mut batch, &step(&[("r1", 0, 2)]), |_| vec![30]);
        let packed = run(&mut batch, &step(&[("r1", 2, 1)]), |_| vec![40]);
        assert_eq!(packed, vec![("r1".into(), vec![30], vec![2])]);
        let packed = run(&mut batch, &step(&[("r1", 3, 1)]), |_| vec![50]);
        assert_eq!(packed, vec![("r1".into(), vec![40], vec![3])]);
        assert_eq!(
            batch.history("r1"),
            (&[10u32, 20][..], &[30u32, 40, 50][..])
        );
    }

    #[test]
    fn test_mixed_prefill_and_decode() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20], vec![0], 0);
        run(&mut batch, &step(&[("r1", 0, 2)]), |_| vec![30]);

        let s = admitting(step(&[("r1", 2, 1)]), "r2", &[50, 60, 70], vec![1, 2], 0, 3);
        batch.update_states(&s);
        let prepared = batch.prepare_inputs(&s);
        assert_eq!(prepared.attn_meta.num_reqs, 2);
        assert_eq!(prepared.attn_meta.total_tokens, 4);

        let (r1, r2) = (row(&prepared, "r1"), row(&prepared, "r2"));
        assert_eq!(prepared.req_inputs[r1].token_count, 1);
        assert_eq!(prepared.req_inputs[r2].token_count, 3);
        assert!(!prepared.attn_meta.is_prefill[r1]);
        assert!(prepared.attn_meta.is_prefill[r2]);
    }

    /// A prompt longer than one step's budget runs as chunks, each read from the prompt at the
    /// scheduler's `num_computed`; only the chunk that reaches the end of the prompt emits a token.
    #[test]
    fn test_chunked_prefill_continuation() {
        let mut batch = InputBatch::new();
        let prompt: Vec<u32> = (0..10).collect();
        batch.add_request("r1".into(), &prompt, vec![0, 1, 2], 0);

        let p1 = batch.prepare_inputs(&step(&[("r1", 0, 5)]));
        assert_eq!(p1.flat_token_ids, &[0, 1, 2, 3, 4]);
        assert!(p1.attn_meta.is_prefill[0]);
        assert!(
            !p1.req_inputs[0].emits_token,
            "a mid-prompt chunk samples nothing"
        );
        batch.reclaim_buffers(p1);
        batch.commit_step("r1", &[], 5, false);
        assert_eq!(batch.tokens_in_pool_for("r1"), 5);
        assert_eq!(batch.token_count("r1"), 10, "nothing was appended");

        let p2 = batch.prepare_inputs(&step(&[("r1", 5, 5)]));
        assert_eq!(p2.flat_token_ids, &[5, 6, 7, 8, 9]);
        assert_eq!(p2.flat_positions, &[5, 6, 7, 8, 9]);
        assert!(p2.attn_meta.is_prefill[0]);
        assert!(
            p2.req_inputs[0].emits_token,
            "the final chunk samples the first token"
        );
        assert_eq!(p2.attn_meta.seq_lens[0], 10);
        batch.reclaim_buffers(p2);
        batch.commit_step("r1", &[90], 5, false);

        let pd = batch.prepare_inputs(&step(&[("r1", 10, 1)]));
        assert!(!pd.attn_meta.is_prefill[0]);
        assert_eq!(pd.flat_token_ids, &[90]);
        assert_eq!(pd.attn_meta.q_lens[0], 1);
        assert_eq!(pd.attn_meta.seq_lens[0], 11);
    }

    /// Two chunks of 32 read `prompt[0..32]` then `prompt[32..64]`.
    #[test]
    fn test_chunked_prefill_second_chunk_slice() {
        let mut batch = InputBatch::new();
        let prompt: Vec<u32> = (0..64).collect();
        batch.add_request("r1".into(), &prompt, vec![0, 1, 2, 3], 0);
        let packed = run(&mut batch, &step(&[("r1", 0, 32)]), |_| vec![]);
        assert_eq!(packed[0].1, (0..32).collect::<Vec<u32>>());
        let packed = run(&mut batch, &step(&[("r1", 32, 32)]), |_| vec![64]);
        assert_eq!(packed[0].1, (32..64).collect::<Vec<u32>>());
        assert_eq!(packed[0].2, (32..64).collect::<Vec<u32>>());
    }

    /// seq_lens stays within the scheduler's block allocation across chunks.
    #[test]
    fn test_chunked_prefill_slot_mapping_invariant() {
        let block_size = 4usize;
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[0, 1, 2, 3, 4, 5, 6, 7], vec![0], 0);
        run(&mut batch, &step(&[("r1", 0, 4)]), |_| vec![]);

        let mut s = step(&[("r1", 4, 4)]);
        s.scheduled_cached_reqs.new_block_ids[0] = Some(vec![vec![0, 1]]);
        batch.update_states(&s);
        let p2 = batch.prepare_inputs(&s);
        assert_eq!(p2.attn_meta.seq_lens[0], 8);
        assert!(
            p2.attn_meta.seq_lens[0] <= p2.attn_meta.block_ids[0].len() * block_size,
            "seq_lens must not exceed block capacity"
        );
    }

    #[test]
    fn test_seq_lens_formula_for_prefill() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[1, 2, 3, 4, 5], vec![0, 1], 0);
        let p = batch.prepare_inputs(&step(&[("r1", 0, 5)]));
        assert_eq!(p.attn_meta.seq_lens[0], 5, "seq_lens = 0 + 5 = 5");
        batch.reclaim_buffers(p);

        batch.add_request("r2".into(), &[7, 8, 10, 20, 30], vec![2, 3], 2);
        let p2 = batch.prepare_inputs(&step(&[("r2", 2, 3)]));
        assert_eq!(
            p2.attn_meta.seq_lens[row(&p2, "r2")],
            5,
            "2 (cached) + 3 (tokens)"
        );
    }

    #[test]
    fn test_query_start_loc_consistency() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20], vec![0], 0);
        batch.add_request("r2".into(), &[30, 40, 50], vec![1], 0);

        let prepared = batch.prepare_inputs(&step(&[("r1", 0, 2), ("r2", 0, 3)]));
        let meta = &prepared.attn_meta;

        assert_eq!(meta.query_start_loc.len(), meta.num_reqs + 1);
        assert_eq!(*meta.query_start_loc.last().unwrap(), meta.total_tokens);
        for i in 0..meta.num_reqs {
            assert_eq!(
                meta.query_start_loc[i + 1] - meta.query_start_loc[i],
                meta.q_lens[i]
            );
        }
    }

    /// `seq_lens[i]` = `tokens_before[i]` + `q_lens[i]` in a mixed decode / fresh / cached batch.
    #[test]
    fn test_seq_lens_equals_tokens_before_plus_q_lens() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[1, 2], vec![0], 0);
        run(&mut batch, &step(&[("r1", 0, 2)]), |_| vec![99]);
        batch.add_request("r2".into(), &[10, 20, 30], vec![1, 2], 0);
        batch.add_request("r3".into(), &[1, 2, 3, 4, 40, 50], vec![3, 4], 4);

        let prepared = batch.prepare_inputs(&step(&[("r1", 2, 1), ("r2", 0, 3), ("r3", 4, 2)]));
        let meta = &prepared.attn_meta;
        for i in 0..meta.num_reqs {
            assert_eq!(
                meta.seq_lens[i],
                meta.tokens_before[i] + meta.q_lens[i],
                "req {i}: seq_lens must equal tokens_before + q_lens"
            );
        }
    }

    #[test]
    fn test_tokens_in_pool_tracking() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30], vec![0], 0);
        assert_eq!(batch.tokens_in_pool_for("r1"), 0);

        run(&mut batch, &step(&[("r1", 0, 3)]), |_| vec![40]);
        assert_eq!(batch.tokens_in_pool_for("r1"), 3);
        run(&mut batch, &step(&[("r1", 3, 1)]), |_| vec![50]);
        assert_eq!(batch.tokens_in_pool_for("r1"), 4);
    }

    /// The scheduler's `num_computed_tokens` is the pool cursor of every step it schedules — e.g. a
    /// spans credit advances it WITHOUT an executed step, and the step follows the jump.
    #[test]
    fn the_scheduler_sets_the_pool_cursor_of_every_scheduled_step() {
        let mut batch = InputBatch::new();
        let prompt: Vec<u32> = (0..12).collect();
        batch.add_request("r1".into(), &prompt, vec![0, 1, 2], 0);
        run(&mut batch, &step(&[("r1", 0, 4)]), |_| vec![]);
        // Tokens 4..8 are credited (reused KV); the scheduler resumes the prefill at 8.
        let p = batch.prepare_inputs(&step(&[("r1", 8, 4)]));
        assert_eq!(p.flat_token_ids, &[8, 9, 10, 11]);
        assert_eq!(p.attn_meta.tokens_before[0], 8);
        assert_eq!(batch.tokens_in_pool_for("r1"), 8);
    }

    /// Sliding-group tables follow the packed rows, not the raw slot order.
    #[test]
    fn sliding_groups_follow_the_packed_rows() {
        let mut batch = InputBatch::new();
        batch.add_request_hybrid("a".into(), &[1], vec![0], vec![vec![100]], 0);
        batch.add_request_hybrid("b".into(), &[2], vec![1], vec![vec![200]], 0);

        let prepared = batch.prepare_inputs(&step(&[("b", 0, 1)]));
        assert_eq!(prepared.attn_meta.req_ids, vec!["b".to_string()]);
        assert_eq!(prepared.attn_meta.sliding_groups, vec![vec![vec![200]]]);
    }

    // -----------------------------------------------------------------------
    // Speculative decode
    // -----------------------------------------------------------------------

    #[test]
    fn test_spec_decode_tokens() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10], vec![0], 0);
        run(&mut batch, &step(&[("r1", 0, 1)]), |_| vec![20]);

        let prepared = batch.prepare_inputs(&drafted(step(&[("r1", 1, 3)]), "r1", &[30, 40]));
        // [last_token=20, draft_30, draft_40].
        assert_eq!(prepared.req_inputs[0].token_count, 3);
        assert_eq!(prepared.req_inputs[0].spec_token_ids, vec![30, 40]);
        assert_eq!(prepared.flat_token_ids, &[20, 30, 40]);
        assert_eq!(prepared.flat_positions, &[1, 2, 3]);
        assert!(prepared.req_inputs[0].emits_token);
    }

    #[test]
    fn test_spec_decode_commit() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10], vec![0], 0);
        run(&mut batch, &step(&[("r1", 0, 1)]), |_| vec![20]);
        assert_eq!(batch.tokens_in_pool_for("r1"), 1);

        // Spec decode: 3 tokens input, 2 accepted.
        batch.commit_step("r1", &[30, 40], 3, true);
        // tokens_in_pool increases by sampled.len() (2), not input count (3).
        assert_eq!(batch.tokens_in_pool_for("r1"), 3);
        assert_eq!(batch.history("r1").1, &[20, 30, 40]);
    }

    #[test]
    fn test_spec_decode_multi_token_commit() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30], vec![0], 0);

        run(&mut batch, &step(&[("r1", 0, 3)]), |_| vec![40]);
        assert_eq!(batch.tokens_in_pool_for("r1"), 3);
        run(&mut batch, &step(&[("r1", 3, 1)]), |_| vec![50]);
        assert_eq!(batch.tokens_in_pool_for("r1"), 4);

        // Spec decode: 4 input tokens [50, d0, d1, d2], all 3 accepted + bonus.
        let packed = run(
            &mut batch,
            &drafted(step(&[("r1", 4, 4)]), "r1", &[60, 70, 80]),
            |_| vec![60, 70, 80, 90],
        );
        assert_eq!(packed[0].1, vec![50, 60, 70, 80]);
        assert_eq!(batch.tokens_in_pool_for("r1"), 8);

        // Nothing rejected: the scheduler resumes at 8, on the bonus token.
        let packed = run(&mut batch, &step(&[("r1", 8, 1)]), |_| vec![91]);
        assert_eq!(packed, vec![("r1".into(), vec![90], vec![8])]);
    }

    #[test]
    fn test_spec_decode_partial_accept_commit() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20], vec![0], 0);

        run(&mut batch, &step(&[("r1", 0, 2)]), |_| vec![30]);
        assert_eq!(batch.tokens_in_pool_for("r1"), 2);

        // Spec decode: 4 input tokens [30, d0, d1, d2], only d0 accepted + recovered 50.
        run(
            &mut batch,
            &drafted(step(&[("r1", 2, 4)]), "r1", &[40, 41, 42]),
            |_| vec![40, 50],
        );
        // 30 and d0 are correctly cached.
        assert_eq!(batch.tokens_in_pool_for("r1"), 4);

        // The scheduler rewound its cursor by the 2 rejected drafts: 2 + 4 - 2 = 4.
        let packed = run(&mut batch, &step(&[("r1", 4, 1)]), |_| vec![60]);
        assert_eq!(packed, vec![("r1".into(), vec![50], vec![4])]);
    }

    /// Regression test for the super-fast graph path deferred commit bug.
    ///
    /// Simulates the deferred commit pattern from the worker's super-fast
    /// graph path. Each iteration:
    ///   1. Capture `token_counts` via `fast_path_token_counts()` BEFORE
    ///      resolving the previous step's pending commit.
    ///   2. Resolve the previous step with its captured `token_counts`.
    ///   3. Store the new `token_counts` for the next iteration.
    ///
    /// The old code used `tokens_in_pool` instead of `fast_path_token_counts()`,
    /// causing Fibonacci-like exponential growth of `tokens_in_pool`.
    /// With the fix, `fast_path_token_counts()` returns `[1; n]` and growth
    /// is linear (exactly +1 per step).
    #[test]
    fn test_fast_path_token_counts_linear_growth() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30], vec![0, 1], 0);

        // Step 1: prefill.
        run(&mut batch, &step(&[("r1", 0, 3)]), |_| vec![40]);
        assert_eq!(batch.tokens_in_pool_for("r1"), 3);

        // Step 2: first graph decode (normal path). Deferred, token_count=1.
        let mut pending_tc = vec![1usize];

        // Steps 3..20: super-fast path loop.
        for step in 3..=20 {
            // Capture token_counts BEFORE resolving the previous pending commit.
            let new_pending_tc = batch.fast_path_token_counts();
            assert_eq!(
                new_pending_tc,
                vec![1],
                "fast_path_token_counts must always be [1]"
            );

            // Resolve previous step's deferred commit.
            batch.commit_step("r1", &[40 + step], pending_tc[0], false);

            // tokens_in_pool should be exactly: prompt_len + (step - 2)
            // because we've resolved (step - 2) decode commits so far.
            let expected = 3 + (step - 2) as usize;
            assert_eq!(
                batch.tokens_in_pool_for("r1"),
                expected,
                "step {step}: tokens_in_pool should be {expected} (linear), got {}",
                batch.tokens_in_pool_for("r1")
            );

            pending_tc = new_pending_tc;
        }

        // After 18 super-fast steps, tokens_in_pool should be 3 + 18 = 21.
        assert_eq!(batch.tokens_in_pool_for("r1"), 21);
    }

    // Rejection-sample unit tests live with their owning module:
    //   `scratchy-serving-engine/src/spec_decode/verify.rs`. Local copies removed
    //   in phase 5.5 along with the `pub use` shim.

    #[test]
    fn test_rejection_output_length_invariants() {
        // For K drafts:
        //   - All accept: output length = K + 1
        //   - First reject: output length = 1
        //   - M accepted (0 < M < K): output length = M + 1
        for k in 1..=8 {
            let drafts: Vec<u32> = (1..=k).collect();

            // All accept
            let mut targets: Vec<u32> = (1..=k).collect();
            targets.push(99); // bonus
            let result = greedy_rejection_sample(&targets, &drafts);
            assert_eq!(
                result.accepted_tokens.len(),
                k as usize + 1,
                "all-accept with {k} drafts should produce {0} tokens",
                k + 1
            );
            assert_eq!(result.num_accepted_drafts, k as usize);

            // First reject
            let mut targets: Vec<u32> = vec![0]; // mismatch (draft[0] = 1)
            targets.extend(2..=k);
            targets.push(99);
            let result = greedy_rejection_sample(&targets, &drafts);
            assert_eq!(
                result.accepted_tokens.len(),
                1,
                "first-reject with {k} drafts should produce 1 token"
            );
            assert_eq!(result.num_accepted_drafts, 0);
        }
    }

    #[test]
    fn test_rejection_mixed_batch_simulation() {
        // Simulate a mixed batch: some requests have drafts, some don't.
        struct ReqInput {
            target_ids: Vec<u32>,
            draft_ids: Vec<u32>,
        }

        let batch = [
            // Normal request (no drafts)
            ReqInput {
                target_ids: vec![42],
                draft_ids: vec![],
            },
            // Spec decode, all accept (3 drafts)
            ReqInput {
                target_ids: vec![10, 20, 30, 99],
                draft_ids: vec![10, 20, 30],
            },
            // Spec decode, partial accept (5 drafts, first 2 match)
            ReqInput {
                target_ids: vec![1, 2, 77, 4, 5, 99],
                draft_ids: vec![1, 2, 3, 4, 5],
            },
            // Normal request (no drafts)
            ReqInput {
                target_ids: vec![55],
                draft_ids: vec![],
            },
            // Spec decode, first reject (2 drafts)
            ReqInput {
                target_ids: vec![88, 20, 99],
                draft_ids: vec![10, 20],
            },
        ];

        let results: Vec<_> = batch
            .iter()
            .map(|r| greedy_rejection_sample(&r.target_ids, &r.draft_ids))
            .collect();

        // Normal: 1 token
        assert_eq!(results[0].accepted_tokens, vec![42]);
        assert_eq!(results[0].num_accepted_drafts, 0);

        // All accept: 4 tokens (3 + bonus)
        assert_eq!(results[1].accepted_tokens, vec![10, 20, 30, 99]);
        assert_eq!(results[1].num_accepted_drafts, 3);

        // Partial: 3 tokens (2 accepted + recovered)
        assert_eq!(results[2].accepted_tokens, vec![1, 2, 77]);
        assert_eq!(results[2].num_accepted_drafts, 2);

        // Normal: 1 token
        assert_eq!(results[3].accepted_tokens, vec![55]);
        assert_eq!(results[3].num_accepted_drafts, 0);

        // First reject: 1 token
        assert_eq!(results[4].accepted_tokens, vec![88]);
        assert_eq!(results[4].num_accepted_drafts, 0);
    }

    #[test]
    fn test_rejection_bonus_token_differs_from_drafts() {
        // Verify the bonus token comes from target_ids[K], not from drafts.
        let drafts = vec![10, 20];
        let targets = vec![10, 20, 777]; // bonus = 777
        let result = greedy_rejection_sample(&targets, &drafts);
        assert_eq!(result.accepted_tokens, vec![10, 20, 777]);
        assert_eq!(*result.accepted_tokens.last().unwrap(), 777);
    }

    #[test]
    fn test_rejection_recovered_token_is_target_not_draft() {
        // On rejection at position i, output target_ids[i] (not draft_ids[i]).
        let drafts = vec![10, 20, 30];
        let targets = vec![10, 55, 30, 99]; // reject at position 1: target=55, draft=20
        let result = greedy_rejection_sample(&targets, &drafts);
        assert_eq!(result.accepted_tokens, vec![10, 55]);
        assert_eq!(result.accepted_tokens[1], 55); // recovered token is target, not draft (20)
    }

    #[test]
    fn test_rejection_num_accepted_plus_output_len() {
        // Invariant: accepted_tokens.len() = num_accepted_drafts + 1
        for num_drafts in 1u32..=6 {
            let drafts: Vec<u32> = (1..=num_drafts).collect();

            // Test every possible rejection point
            for reject_at in 0..=num_drafts {
                let mut targets: Vec<u32> = (1..=num_drafts).collect();
                targets.push(99); // bonus
                if reject_at < num_drafts {
                    targets[reject_at as usize] = 0; // force mismatch
                }

                let result = greedy_rejection_sample(&targets, &drafts);
                assert_eq!(
                    result.accepted_tokens.len(),
                    result.num_accepted_drafts + 1,
                    "invariant: len = num_accepted + 1 (drafts={num_drafts}, reject_at={reject_at})"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Preemption / resumption
    //
    // A preempted request keeps its slot and its history and is simply not scheduled. The scheduler
    // resumes it by scheduling it again from its `num_computed_tokens` (0, or a prefix-cache hit)
    // with its complete new block tables; its chunk is then read from `prompt ++ generated` like any
    // other. The root bug this family of tests grew from was a resume that packed MORE tokens than the
    // scheduler allocated blocks for — seq_lens > blocks → slot_mapping = -1 → a GPU fault — which a
    // step built from `num_scheduled_tokens` cannot express.
    // -----------------------------------------------------------------------

    /// Helper: whether `block_ids` cover `num_tokens` at `block_size`.
    fn blocks_cover(num_tokens: usize, block_ids: &[usize], block_size: usize) -> bool {
        block_ids.len() >= num_tokens.div_ceil(block_size)
    }

    /// A request that ran, then was preempted: prefill of `prompt` then `decodes` decode steps, each
    /// sampling `100 + i`.
    fn ran_then_preempted(b: &mut InputBatch, id: &str, prompt: &[u32], decodes: u32) {
        b.add_request(id.into(), prompt, vec![0, 1], 0);
        let n = prompt.len() as u32;
        run(b, &step(&[(id, 0, prompt.len())]), |_| vec![100]);
        for i in 0..decodes {
            run(b, &step(&[(id, n + i, 1)]), move |_| vec![101 + i]);
        }
        b.update_states(&preempting(step(&[]), &[id]));
    }

    /// Preemption keeps the slot and its history, and the request does not run while preempted.
    #[test]
    fn test_preempted_request_keeps_its_slot_and_does_not_run() {
        let mut batch = InputBatch::new();
        batch.add_request("r2".into(), &[40, 50], vec![2], 0);
        ran_then_preempted(&mut batch, "r1", &[10, 20, 30], 1);
        assert!(batch.contains("r1"));
        assert_eq!(batch.history("r1").1, &[100, 101]);

        let p = batch.prepare_inputs(&step(&[("r2", 0, 2)]));
        assert_eq!(p.attn_meta.req_ids, vec!["r2"]);
        assert_eq!(p.attn_meta.total_tokens, 2);
    }

    /// A resume re-prefills the WHOLE history — prompt and everything it had generated — from the
    /// scheduler's cursor, onto the blocks it ships, as a prefill.
    #[test]
    fn test_resumed_request_re_prefills_its_history() {
        let mut batch = InputBatch::new();
        ran_then_preempted(&mut batch, "r1", &[10, 20, 30], 1);

        let s = resumed(step(&[("r1", 0, 5)]), "r1", vec![5, 6]);
        batch.update_states(&s);
        assert_eq!(batch.block_table("r1"), Some(&[5, 6][..]));
        let p = batch.prepare_inputs(&s);
        assert_eq!(p.flat_token_ids, &[10, 20, 30, 100, 101]);
        assert_eq!(p.flat_positions, &[0, 1, 2, 3, 4]);
        assert!(p.attn_meta.is_prefill[0]);
        assert_eq!(p.attn_meta.tokens_before[0], 0);
        assert!(
            p.req_inputs[0].emits_token,
            "the resume caught up: it samples the next token"
        );
    }

    /// A resumed request's pool cursor is the scheduler's (0 from scratch), not what it had before.
    #[test]
    fn test_resumed_tokens_in_pool_is_zero_from_scratch() {
        let mut batch = InputBatch::new();
        ran_then_preempted(&mut batch, "r1", &[10, 20, 30, 40, 50], 2);
        assert_eq!(batch.tokens_in_pool_for("r1"), 7);

        let s = resumed(step(&[("r1", 0, 3)]), "r1", vec![0, 1]);
        batch.update_states(&s);
        let p = batch.prepare_inputs(&s);
        assert_eq!(
            batch.tokens_in_pool_for("r1"),
            0,
            "resumed-from-scratch must have tokens_in_pool=0"
        );
        batch.reclaim_buffers(p);
        batch.commit_step("r1", &[], 3, false);
        assert_eq!(batch.tokens_in_pool_for("r1"), 3, "0 + 3, not 7 + 3");
    }

    /// Resume with a partial prefix cache: the KV of the first `num_computed` tokens is still valid.
    #[test]
    fn test_resumed_with_partial_prefix_cache() {
        let mut batch = InputBatch::new();
        ran_then_preempted(&mut batch, "r1", &[10, 20, 30, 40], 1);

        let s = resumed(step(&[("r1", 2, 4)]), "r1", vec![0, 1]);
        batch.update_states(&s);
        let p = batch.prepare_inputs(&s);
        assert_eq!(p.flat_token_ids, &[30, 40, 100, 101]);
        assert_eq!(p.flat_positions, &[2, 3, 4, 5]);
        assert_eq!(p.attn_meta.tokens_before[0], 2);
    }

    /// The resumed chunk is exactly what the scheduler scheduled — never the whole history — so the
    /// blocks it allocated for that chunk cover it.
    #[test]
    fn test_resumed_chunk_is_the_scheduled_chunk() {
        let block_size = 16;
        let mut batch = InputBatch::new();
        // 10 prompt tokens + 20 generated before preemption = a 30-token history.
        batch.add_request("r1".into(), &(0..10).collect::<Vec<u32>>(), vec![0], 0);
        for t in 10..30 {
            batch.push_generated("r1", t);
        }
        let s = resumed(step(&[("r1", 0, 16)]), "r1", vec![0]);
        batch.update_states(&s);
        let p = batch.prepare_inputs(&s);

        assert_eq!(p.req_inputs[0].token_count, 16);
        assert_eq!(p.flat_token_ids, (0..16).collect::<Vec<u32>>());
        assert_eq!(p.attn_meta.seq_lens[0], 16);
        assert!(blocks_cover(
            p.attn_meta.seq_lens[0],
            &p.attn_meta.block_ids[0],
            block_size
        ));
        assert!(
            !p.req_inputs[0].emits_token,
            "a resume chunk short of the history samples nothing"
        );
    }

    /// A resume that takes two chunks: the second starts where the first stopped, and only it emits.
    #[test]
    fn test_chunked_resumption_two_steps() {
        let block_size = 16;
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &(0..40).collect::<Vec<u32>>(), vec![0], 0);

        let s1 = resumed(step(&[("r1", 0, 16)]), "r1", vec![0]);
        let packed = run(&mut batch, &s1, |_| vec![999]);
        assert_eq!(packed[0].1, (0..16).collect::<Vec<u32>>());
        assert_eq!(batch.tokens_in_pool_for("r1"), 16);
        assert_eq!(
            batch.token_count("r1"),
            40,
            "the first chunk emitted nothing"
        );

        let mut s2 = step(&[("r1", 16, 24)]);
        s2.scheduled_cached_reqs.new_block_ids[0] = Some(vec![vec![0, 1, 2]]);
        batch.update_states(&s2);
        let p2 = batch.prepare_inputs(&s2);
        assert_eq!(p2.flat_token_ids, (16..40).collect::<Vec<u32>>());
        assert_eq!(p2.attn_meta.seq_lens[0], 16 + 24);
        assert!(blocks_cover(40, &p2.attn_meta.block_ids[0], block_size));
        assert!(p2.req_inputs[0].emits_token);
    }

    /// A 1024-token history resumed as a 128-token chunk onto 8 blocks of 16.
    #[test]
    fn test_resumed_chunk_matches_block_capacity() {
        let block_size = 16usize;
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &(0..1024).collect::<Vec<u32>>(), vec![0], 0);
        let s = resumed(step(&[("r1", 0, 128)]), "r1", (0..8).collect());
        batch.update_states(&s);
        let p = batch.prepare_inputs(&s);
        assert_eq!(p.flat_token_ids.len(), 128);
        assert_eq!(p.attn_meta.block_ids[0].len() * block_size, 128);
        assert!(blocks_cover(
            p.attn_meta.seq_lens[0],
            &p.attn_meta.block_ids[0],
            block_size
        ));
    }

    /// Preempted and resumed in the same scheduler step.
    #[test]
    fn test_same_step_preemption_and_resumption() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30], vec![0, 1], 0);
        run(&mut batch, &step(&[("r1", 0, 3)]), |_| vec![99]);
        run(&mut batch, &step(&[("r1", 3, 1)]), |_| vec![100]);
        assert_eq!(batch.tokens_in_pool_for("r1"), 4);

        let s = preempting(resumed(step(&[("r1", 0, 2)]), "r1", vec![5, 6]), &["r1"]);
        batch.update_states(&s);
        let p = batch.prepare_inputs(&s);
        assert_eq!(p.flat_token_ids, &[10, 20]);
        assert!(p.attn_meta.is_prefill[0]);
        assert_eq!(batch.tokens_in_pool_for("r1"), 0);
        assert_eq!(batch.block_table("r1"), Some(&[5, 6][..]));
    }

    /// Preempted, resumed, preempted again, resumed again — the history carries through each cycle.
    #[test]
    fn test_multiple_preemption_cycles() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[1, 2, 3, 4], vec![0, 1], 0);
        run(&mut batch, &step(&[("r1", 0, 4)]), |_| vec![10]);
        batch.update_states(&preempting(step(&[]), &["r1"]));

        // First resumption: a full re-prefill of prompt ++ [10].
        let packed = run(
            &mut batch,
            &resumed(step(&[("r1", 0, 5)]), "r1", vec![2, 3]),
            |_| vec![11],
        );
        assert_eq!(packed[0].1, vec![1, 2, 3, 4, 10]);
        run(&mut batch, &step(&[("r1", 5, 1)]), |_| vec![12]);
        assert_eq!(batch.tokens_in_pool_for("r1"), 6);
        batch.update_states(&preempting(step(&[]), &["r1"]));

        // Second resumption: partial prefix cache, num_computed=6, one token left.
        let s = resumed(step(&[("r1", 6, 1)]), "r1", vec![4, 5]);
        batch.update_states(&s);
        let p = batch.prepare_inputs(&s);
        assert_eq!(p.flat_token_ids, &[12]);
        assert_eq!(p.flat_positions, &[6]);
        assert_eq!(p.attn_meta.seq_lens[0], 7);
    }

    /// The block table a resume ships replaces whatever the request held before.
    #[test]
    fn test_resumed_block_table_overwrite() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20], vec![0, 1], 0);
        run(&mut batch, &step(&[("r1", 0, 2)]), |_| vec![99]);
        batch.update_blocks("r1", vec![0, 1, 2, 3]);
        batch.update_states(&preempting(step(&[]), &["r1"]));

        batch.update_states(&resumed(step(&[("r1", 0, 3)]), "r1", vec![7, 8]));
        assert_eq!(
            batch.block_table("r1"),
            Some(&[7, 8][..]),
            "resumed request must use the new block allocation, not the old one"
        );
    }

    /// After a resume, decode continues from the history's end.
    #[test]
    fn test_resumed_then_normal_decode_progression() {
        let mut batch = InputBatch::new();
        ran_then_preempted(&mut batch, "r1", &[10, 20, 30], 0);

        run(
            &mut batch,
            &resumed(step(&[("r1", 0, 4)]), "r1", vec![0]),
            |_| vec![200],
        );
        assert_eq!(batch.tokens_in_pool_for("r1"), 4);
        let packed = run(&mut batch, &step(&[("r1", 4, 1)]), |_| vec![201]);
        assert_eq!(packed, vec![("r1".into(), vec![200], vec![4])]);
        let packed = run(&mut batch, &step(&[("r1", 5, 1)]), |_| vec![202]);
        assert_eq!(packed, vec![("r1".into(), vec![201], vec![5])]);
    }

    /// A decode, a resume and a brand-new request in one step.
    #[test]
    fn test_mixed_batch_with_preempted_and_new() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20], vec![0], 0);
        run(&mut batch, &step(&[("r1", 0, 2)]), |_| vec![99]);
        batch.add_request("r2".into(), &[30, 40, 50], vec![1, 2], 0);
        run(&mut batch, &step(&[("r2", 0, 3)]), |_| vec![100]);
        batch.update_states(&preempting(step(&[]), &["r2"]));

        let s = admitting(
            resumed(step(&[("r1", 2, 1), ("r2", 0, 2)]), "r2", vec![4]),
            "r3",
            &[60, 70],
            vec![3],
            0,
            2,
        );
        batch.update_states(&s);
        assert_eq!(batch.num_active(), 3);
        let prepared = batch.prepare_inputs(&s);
        assert_eq!(prepared.attn_meta.num_reqs, 3);

        let (r1, r2, r3) = (
            row(&prepared, "r1"),
            row(&prepared, "r2"),
            row(&prepared, "r3"),
        );
        assert!(!prepared.attn_meta.is_prefill[r1], "r1 must be decode");
        assert!(prepared.attn_meta.is_prefill[r2], "r2 must be prefill");
        assert!(prepared.attn_meta.is_prefill[r3], "r3 must be prefill");
        assert_eq!(prepared.req_inputs[r1].token_count, 1);
        assert_eq!(prepared.req_inputs[r2].token_count, 2);
        assert_eq!(prepared.req_inputs[r3].token_count, 2);
        assert_eq!(prepared.attn_meta.seq_lens[r1], 3);
        assert_eq!(prepared.attn_meta.seq_lens[r2], 2);
        assert_eq!(prepared.attn_meta.seq_lens[r3], 2);
        assert!(
            !prepared.req_inputs[r2].emits_token,
            "r2 resumed 2 of its 4 tokens"
        );
    }

    /// Positions of a resumed prefill start at the scheduler's cursor.
    #[test]
    fn test_resumed_prefill_positions() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[10, 20, 30, 40, 50, 60, 70], vec![0, 1], 0);
        let p = batch.prepare_inputs(&resumed(step(&[("r1", 0, 3)]), "r1", vec![0, 1]));
        assert_eq!(p.flat_positions, &[0, 1, 2]);
        batch.reclaim_buffers(p);

        let p2 = batch.prepare_inputs(&resumed(step(&[("r1", 4, 3)]), "r1", vec![0, 1]));
        assert_eq!(
            p2.flat_positions,
            &[4, 5, 6],
            "resumed with num_computed=4: positions must be 4,5,6"
        );
        assert_eq!(p2.flat_token_ids, &[50, 60, 70]);
    }

    /// query_start_loc invariants with a resumed multi-token prefill beside a decode.
    #[test]
    fn test_query_start_loc_with_resumed_prefill() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[1, 2], vec![0], 0);
        run(&mut batch, &step(&[("r1", 0, 2)]), |_| vec![99]);
        batch.add_request("r2".into(), &[10, 20, 30, 40, 50], vec![1, 2, 3], 0);

        let prepared = batch.prepare_inputs(&resumed(
            step(&[("r1", 2, 1), ("r2", 0, 5)]),
            "r2",
            vec![1, 2, 3],
        ));
        let meta = &prepared.attn_meta;
        assert_eq!(meta.query_start_loc.len(), meta.num_reqs + 1);
        assert_eq!(*meta.query_start_loc.last().unwrap(), meta.total_tokens);
        for i in 0..meta.num_reqs {
            assert_eq!(
                meta.query_start_loc[i + 1] - meta.query_start_loc[i],
                meta.q_lens[i]
            );
        }
        assert_eq!(meta.total_tokens, 6);
    }

    /// Every request preempted at once: nothing runs; every one resumes as a prefill.
    #[test]
    fn test_preempt_all_requests_simultaneously() {
        let mut batch = InputBatch::new();
        let ids: Vec<String> = (0..6).map(|i| format!("r{i}")).collect();
        for (i, id) in ids.iter().enumerate() {
            batch.add_request(id.clone(), &[i as u32, i as u32 + 1], vec![i], 0);
        }
        let all: Vec<(&str, u32, usize)> = ids.iter().map(|id| (id.as_str(), 0, 2)).collect();
        run(&mut batch, &step(&all), |_| vec![100]);

        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let s = preempting(step(&[]), &refs);
        batch.update_states(&s);
        assert_eq!(batch.num_active(), 6);
        let p = batch.prepare_inputs(&s);
        assert_eq!(p.attn_meta.num_reqs, 0);
        assert_eq!(p.attn_meta.total_tokens, 0);
        batch.reclaim_buffers(p);

        let resume: Vec<(&str, u32, usize)> = ids.iter().map(|id| (id.as_str(), 0, 3)).collect();
        let mut s = step(&resume);
        for (i, id) in ids.iter().enumerate() {
            s = resumed(s, id, vec![10 + i, 20 + i]);
        }
        batch.update_states(&s);
        let prepared = batch.prepare_inputs(&s);
        assert!(prepared.attn_meta.is_prefill.iter().all(|&p| p));
        assert!(prepared.req_inputs.iter().all(|r| r.emits_token));
        assert_eq!(prepared.attn_meta.total_tokens, 18);
    }

    /// 10 preempt/resume cycles: each resume re-prefills the grown history onto fresh blocks.
    #[test]
    fn test_many_preemption_cycles_stress() {
        let mut batch = InputBatch::new();
        let prompt: Vec<u32> = (1..=8).collect();
        batch.add_request("r1".into(), &prompt, vec![0, 1], 0);
        run(&mut batch, &step(&[("r1", 0, 8)]), |_| vec![99]);

        for cycle in 0u32..10 {
            batch.update_states(&preempting(step(&[]), &["r1"]));
            let history = batch.token_count("r1");
            let blocks: Vec<usize> = vec![cycle as usize * 2, cycle as usize * 2 + 1];
            let s = resumed(step(&[("r1", 0, history)]), "r1", blocks.clone());
            batch.update_states(&s);
            assert_eq!(batch.block_table("r1").unwrap(), blocks.as_slice());
            let p = batch.prepare_inputs(&s);
            assert_eq!(batch.tokens_in_pool_for("r1"), 0, "cycle {cycle}");
            assert_eq!(p.flat_token_ids.len(), history, "cycle {cycle}");
            assert_eq!(p.flat_token_ids[..8], prompt[..], "cycle {cycle}");
            assert!(p.req_inputs[0].emits_token);
            batch.reclaim_buffers(p);
            batch.commit_step("r1", &[200 + cycle], history, false);
            assert_eq!(batch.tokens_in_pool_for("r1"), history);
        }
        assert_eq!(batch.token_count("r1"), 8 + 1 + 10);
    }

    /// A resumed prefill's tokens_before is the scheduler's cursor: 0 from scratch, K with a hit.
    #[test]
    fn test_tokens_before_for_resumed_prefill() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[1, 2, 3], vec![0, 1], 0);
        batch.add_request(
            "r2".into(),
            &(0..10).collect::<Vec<u32>>(),
            vec![2, 3, 4],
            0,
        );

        let s = resumed(
            resumed(step(&[("r1", 0, 3), ("r2", 6, 4)]), "r1", vec![0, 1]),
            "r2",
            vec![2, 3, 4],
        );
        let prepared = batch.prepare_inputs(&s);
        assert_eq!(prepared.attn_meta.tokens_before[row(&prepared, "r1")], 0);
        assert_eq!(prepared.attn_meta.tokens_before[row(&prepared, "r2")], 6);
    }

    /// Preempting a request with speculative-decode history leaves the others untouched.
    #[test]
    fn test_preempt_request_with_spec_decode_history() {
        let mut batch = InputBatch::new();
        batch.add_request("r1".into(), &[1, 2], vec![0], 0);
        batch.add_request("r2".into(), &[3, 4], vec![1], 0);
        run(&mut batch, &step(&[("r1", 0, 2), ("r2", 0, 2)]), |id| {
            vec![if id == "r1" { 10 } else { 20 }]
        });
        run(
            &mut batch,
            &drafted(step(&[("r1", 2, 3)]), "r1", &[11, 12]),
            |_| vec![11, 12, 13],
        );
        assert_eq!(batch.tokens_in_pool_for("r1"), 5);
        run(&mut batch, &step(&[("r2", 2, 1)]), |_| vec![21]);
        assert_eq!(batch.tokens_in_pool_for("r2"), 3);

        batch.update_states(&preempting(step(&[]), &["r1"]));
        assert_eq!(batch.tokens_in_pool_for("r2"), 3);
        let prepared = batch.prepare_inputs(&step(&[("r2", 3, 1)]));
        assert_eq!(prepared.attn_meta.req_ids, vec!["r2"]);
        assert_eq!(prepared.flat_token_ids, &[21]);
        assert!(!prepared.attn_meta.is_prefill[0]);
    }

    /// A resume as a single-token chunk: seq_lens = 0 + 1.
    #[test]
    fn test_resumed_single_token_chunk() {
        let mut batch = InputBatch::new();
        ran_then_preempted(&mut batch, "r1", &[1, 2, 3, 4, 5], 100);
        assert_eq!(batch.tokens_in_pool_for("r1"), 105);

        let p = batch.prepare_inputs(&resumed(step(&[("r1", 0, 1)]), "r1", vec![0]));
        assert_eq!(p.attn_meta.seq_lens[0], 1);
        assert_eq!(p.flat_token_ids, &[1]);
        assert!(p.attn_meta.is_prefill[0]);
        assert!(!p.req_inputs[0].emits_token);
    }

    /// Block coverage: boundary at exactly one block full, one over, block_size 1, empty.
    #[test]
    fn test_block_coverage_boundaries() {
        assert!(blocks_cover(32, &[0, 1], 16));
        assert!(!blocks_cover(33, &[0, 1], 16));
        assert!(blocks_cover(33, &[0, 1, 2], 16));
        assert!(blocks_cover(16, &[0], 16));
        assert!(!blocks_cover(17, &[0], 16));
        assert!(blocks_cover(5, &[0, 1, 2, 3, 4], 1));
        assert!(!blocks_cover(5, &[0, 1, 2, 3], 1));
        assert!(blocks_cover(0, &[], 16));
    }

    // ─────────────────────────────────────────────────────────────────────────────────────────────
    //  The per-slot conservation law, and the token store
    // ─────────────────────────────────────────────────────────────────────────────────────────────

    /// EVERY PER-SLOT VEC HOLDS THE SAME NUMBER OF SLOTS, through any sequence of adds and removes.
    ///
    /// This is the law `per_slot_fields!` exists to make unbreakable: a slot is one element in each of
    /// six parallel vecs, so "slot `i` describes request `i`" is only true while all six have the
    /// same length. With push/swap/pop generated from one field list an inequality is no longer
    /// expressible through the API — this is what notices if someone hand-edits one vec anyway.
    #[test]
    fn every_per_slot_vec_holds_the_same_number_of_slots() {
        let mut b = InputBatch::new();
        let check = |b: &InputBatch, want: usize, when: &str| {
            for (name, len) in b.per_slot_lens() {
                assert_eq!(
                    len, want,
                    "{name} has {len} slots, expected {want} — {when}"
                );
            }
        };
        check(&b, 0, "empty");
        for i in 0..5 {
            b.add_request(format!("r{i}"), &[i as u32], vec![i], 0);
        }
        check(&b, 5, "after 5 adds");
        b.add_request("r2".into(), &[9], vec![9], 0); // re-admission replaces, never duplicates
        check(&b, 5, "after re-admitting an id");
        b.remove_request("r0"); // swap-remove from the FRONT: exercises the swap, not just the pop
        check(&b, 4, "after removing the first slot");
        b.remove_request("r4");
        check(&b, 3, "after removing another");
        b.remove_request("nonexistent"); // must not resize anything
        check(&b, 3, "after removing a request that was never added");
        for id in ["r1", "r2", "r3"] {
            b.remove_request(id);
        }
        check(&b, 0, "after removing everything");
    }

    /// `prompt ++ generated` is ONE sequence and `token_at` is the only place that knows the boundary.
    #[test]
    fn token_at_reads_across_the_prompt_generated_boundary() {
        let mut b = InputBatch::new();
        b.add_request("r".into(), &[10, 11, 12], vec![0], 0);
        assert_eq!(b.prompt_len("r"), 3);
        assert_eq!(b.token_count("r"), 3);

        b.push_generated("r", 90);
        b.commit_step("r", &[91], 1, false);
        assert_eq!(b.token_count("r"), 5, "generation extends the sequence");
        assert_eq!(
            b.prompt_len("r"),
            3,
            "and does NOT move the prompt boundary"
        );

        let seq: Vec<Option<u32>> = (0..6).map(|p| b.token_at("r", p)).collect();
        assert_eq!(
            seq,
            vec![Some(10), Some(11), Some(12), Some(90), Some(91), None],
            "positions 0..5 read straight through the boundary; 5 does not exist"
        );
        assert_eq!(b.history("r"), (&[10u32, 11, 12][..], &[90u32, 91][..]));
    }

    /// A step reads its chunk through the same boundary: a chunk straddling it takes the prompt's
    /// tail and the generated head.
    #[test]
    fn a_chunk_reads_across_the_prompt_generated_boundary() {
        let mut b = InputBatch::new();
        b.add_request("r".into(), &[10, 11, 12], vec![0], 0);
        b.push_generated("r", 90);
        b.push_generated("r", 91);
        let p = b.prepare_inputs(&step(&[("r", 1, 3)]));
        assert_eq!(p.flat_token_ids, &[11, 12, 90]);
        assert!(
            !p.req_inputs[0].emits_token,
            "91 is still ahead of the chunk"
        );
    }

    /// An emitted token is appended to the history; a chunk that emits nothing appends nothing.
    #[test]
    fn commit_step_appends_exactly_what_was_emitted() {
        let mut b = InputBatch::new();
        b.add_request("r".into(), &[1, 2, 3, 4], vec![0], 0);
        b.commit_step("r", &[], 2, false);
        assert_eq!(b.token_count("r"), 4);
        b.commit_step("r", &[7], 2, false);
        assert_eq!(b.history("r").1, &[7]);
        b.commit_step("r", &[8, 9], 3, true);
        assert_eq!(b.history("r").1, &[7, 8, 9]);
    }

    /// The tokens follow their request through a swap-remove — the reason they belong in the struct that
    /// owns the slot mapping rather than in a backend's private map.
    #[test]
    fn the_tokens_follow_their_request_through_a_swap_remove() {
        let mut b = InputBatch::new();
        b.add_request("a".into(), &[10, 11], vec![0], 0);
        b.add_request("victim".into(), &[90], vec![1], 0);
        b.add_request("z".into(), &[70, 71, 72], vec![2], 0);
        b.push_generated("z", 73);

        b.remove_request("victim"); // "z" (last slot) is swapped into slot 1

        assert_eq!(b.prompt_len("a"), 2);
        assert_eq!(b.token_at("a", 1), Some(11));
        assert_eq!(b.token_count("z"), 4);
        assert_eq!(b.token_at("z", 3), Some(73));
        assert_eq!(
            b.prompt_len("z"),
            3,
            "the generated token is not part of the prompt"
        );
        assert_eq!(b.token_at("victim", 0), None);
        assert_eq!(b.token_count("victim"), 0);
    }

    /// An unknown request is not a slot: reads answer nothing and writes do not create one.
    #[test]
    fn the_token_store_has_no_answer_for_a_request_it_does_not_have() {
        let mut b = InputBatch::new();
        assert_eq!(b.token_at("ghost", 0), None);
        assert_eq!(b.token_count("ghost"), 0);
        assert_eq!(b.prompt_len("ghost"), 0);
        assert_eq!(b.history("ghost"), (&[][..], &[][..]));
        b.push_generated("ghost", 4);
        b.commit_step("ghost", &[5], 1, false);
        assert_eq!(b.num_active(), 0);
        assert_eq!(b.token_count("ghost"), 0);
    }
}
