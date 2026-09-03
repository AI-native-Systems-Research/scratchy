// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Proposer trait — what `EngineCore` calls each step to get the
//! per-request draft tokens for speculative decoding.
//!
//! Two impls today:
//!
//!   * [`NgramProposer`] (in [`super::ngram`]) — pure-CPU n-gram lookup
//!     over each request's own token history. Doesn't touch
//!     [`ProposerStepCtx::worker_drafts`].
//!
//!   * [`DraftModelProposer`] — reads pre-computed drafts the worker
//!     produced during `execute_model`, read out of
//!     [`ProposerStepCtx::worker_drafts`]. Doesn't touch token history.
//!     A later change moves the K-step chain itself into this proposer;
//!     until then it's a thin reader.
//!
//! The single-method shape (`propose_for_step`) lets `EngineCore` hold
//! `Option<Box<dyn Proposer>>` and dispatch once per finalize, replacing
//! the prior `Option<NgramProposer> + use_draft_model_proposer: bool`
//! pair.

use std::collections::HashMap;

use super::backend::{ForwardArgmaxRequest, KvPoolHandle, ModelHandle, SpecDecodeBackend};
use super::{DraftModelProposerConfig, NgramProposer};

/// Owned-data seed bundle the worker hands to `DraftModelProposer` so
/// the proposer can run the lockstep prefill + K-step decode chain
/// against the draft model without re-deriving anything from the
/// scheduler state.
///
/// The fields fall into two groups:
///
///   * **Lockstep-prefill inputs** (`input_ids` … `num_tokens`) — owned
///     copies of the verify batch the target ran. Same shape
///     [`ForwardArgmaxRequest`] takes as borrowed slices; the proposer
///     re-issues a `forward_argmax_blocking(DRAFT, DRAFT_KV, …)` over
///     these so the draft's KV mirrors target's.
///
///   * **Per-req state for the K-step chain** (`req_ids` … `block_size`)
///     — what the chain needs to compute each step's `slot_mapping`
///     (= `block_ids[i][pos/bs] * bs + pos%bs`) and to seed each req's
///     starting position (= `tokens_before[i] + (sampled.len() if
///     was_spec_decode else q_lens[i])`).
///
/// The proposer reads `ModelRunnerOutput.sampled_token_ids` for the
/// per-req seed token (the last accepted/bonus token), so we don't
/// duplicate it here.
pub struct DraftSeedInputs {
    pub input_ids: Vec<u32>,
    pub positions: Vec<u32>,
    pub slot_mapping: Vec<u32>,
    pub cu_seqlens_q: Vec<u32>,
    pub seqused_k: Vec<u32>,
    pub block_table: Vec<u32>,
    pub block_table_stride: usize,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub num_tokens: usize,

    pub req_ids: Vec<String>,
    pub block_ids: Vec<Vec<u32>>,
    pub tokens_before: Vec<usize>,
    pub q_lens: Vec<usize>,
    pub was_spec_decode: Vec<bool>,
    pub block_size: usize,

    /// Phase 8 signal: when `true`, the worker has already submitted
    /// the lockstep prefill on its dedicated draft queue (in parallel
    /// with target verify) and awaited its completion before returning
    /// this seed. `DraftModelProposer::propose_lockstep_k` skips its
    /// in-proposer lockstep call. Defaults to `false` for backward
    /// compatibility — the proposer issues the lockstep itself.
    pub async_lockstep_done: bool,

    /// Phase 9 signal: when `Some`, the worker speculatively ran the
    /// K-step draft chain in parallel with target verify on the draft
    /// queue, using draft's lockstep-last argmax per req as the
    /// speculative bonus seed (`speculative_seeds[i]`). The chain's
    /// per-iter argmax IS the `speculative_chain_drafts[req_id]`
    /// output. The proposer checks whether each req's speculative
    /// seed matched target's sampled bonus token; on full agreement
    /// it returns the speculative drafts directly. On any mismatch
    /// it falls back to running the chain on the (correct) bonus
    /// seeds — the speculative work is wasted but stayed hidden
    /// behind target verify so the critical path is unchanged.
    ///
    /// Both vectors are indexed parallel to `req_ids` /
    /// `cu_seqlens_q`. Empty when the backend hasn't wired
    /// speculative chain (CUDA, the in-proposer pre-Phase-9 path).
    pub speculative_seeds: Vec<u32>,
    pub speculative_chain_drafts: Vec<Vec<u32>>,
}

/// Per-step context handed to [`Proposer::propose_for_step`]. Carries
/// the shape both impls need without coupling the trait to either
/// `Scheduler` or `ModelRunnerOutput` directly — `EngineCore` packs the
/// few projections each impl needs into refs/closures here.
pub struct ProposerStepCtx<'a> {
    /// Request IDs scheduled this step (the keys of
    /// `SchedulerOutput::num_scheduled_tokens`).
    pub scheduled_req_ids: &'a [&'a str],
    /// Closure to fetch a request's full token history (prompt +
    /// accepted output so far). Returns `None` if the request is gone
    /// from the scheduler or has no tokens yet — proposer skips it.
    /// Boxed-as-closure rather than a generic so the trait stays
    /// object-safe.
    pub get_all_tokens: &'a dyn Fn(&str) -> Option<Vec<u32>>,
    /// Worker-side drafts (from `ModelRunnerOutput.draft_token_ids`).
    /// `None` when the worker didn't run a draft chain this step.
    /// [`DraftModelProposer`] reads this; [`NgramProposer`] ignores it.
    pub worker_drafts: Option<&'a HashMap<String, Vec<u32>>>,
    /// Mutable handle to the GPU backend, for proposers that need to
    /// issue forward+argmax calls (e.g. [`DraftModelProposer`] runs
    /// the lockstep prefill + K-step chain through it). `None` when
    /// the executor doesn't expose a `SpecDecodeBackend`.
    pub backend: Option<&'a mut dyn SpecDecodeBackend>,
    /// Owned-data seed bundle from the worker. `Some` when the worker
    /// ran a target verify with a draft model loaded; carries the
    /// lockstep-prefill inputs + per-req attn state the K-step chain
    /// needs.
    pub draft_seed: Option<&'a DraftSeedInputs>,
    /// Per-request sampled token IDs from this step's target verify
    /// (= `ModelRunnerOutput.sampled_token_ids`, paired index-wise
    /// with `draft_seed.req_ids`). The K-step chain seeds each req
    /// from this req's last accepted token.
    pub sampled_token_ids: Option<&'a [Vec<u32>]>,
}

/// Single-step speculative-decoding proposer interface.
///
/// Returns the draft tokens for each request scheduled this step. The
/// returned map is consumed by `EngineCore::finalize_step`, which hands
/// each entry to `Scheduler::set_spec_token_ids`. Absent keys / empty
/// vecs skip proposal for that request.
///
/// Takes `&mut` because draft-model proposers issue GPU work via
/// `ctx.backend`; ngram impls treat it as `&self` effectively.
pub trait Proposer {
    fn propose_for_step(&mut self, ctx: &mut ProposerStepCtx<'_>) -> HashMap<String, Vec<u32>>;
}

impl Proposer for NgramProposer {
    fn propose_for_step(&mut self, ctx: &mut ProposerStepCtx<'_>) -> HashMap<String, Vec<u32>> {
        let mut out = HashMap::with_capacity(ctx.scheduled_req_ids.len());
        for &req_id in ctx.scheduled_req_ids {
            let Some(history) = (ctx.get_all_tokens)(req_id) else {
                continue;
            };
            if history.is_empty() {
                continue;
            }
            let drafts = self.propose(&history);
            if !drafts.is_empty() {
                out.insert(req_id.to_string(), drafts);
            }
        }
        out
    }
}

/// Draft-model proposer. Holds the K-step chain that produces draft
/// tokens per request.
///
/// Per step it does two GPU-side things via `ctx.backend`:
///
///   1. **Lockstep prefill** — re-issue the verify-batch forward
///      against the draft model + draft KV pool (`KvPoolHandle(1)` →
///      draft pool, `ModelHandle(1)` → draft model). Argmaxes are
///      discarded; the only side effect we want is the draft's KV
///      mirroring target's at the verify positions.
///   2. **K decode steps** — chain K M=1 forwards on the draft starting
///      from each req's last accepted/bonus token, at the slot just
///      after the verify positions.
///
/// The seed bundle [`DraftSeedInputs`] is filled by the worker during
/// target verify and carried in `ModelRunnerOutput.draft_seed_inputs`.
/// When `ctx.backend` or `ctx.draft_seed` is `None`, propose returns
/// empty (defensive — happens during warmup or on backends without a
/// `SpecDecodeBackend` exposure).
#[derive(Debug)]
pub struct DraftModelProposer {
    config: DraftModelProposerConfig,
}

impl DraftModelProposer {
    pub fn new(config: DraftModelProposerConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &DraftModelProposerConfig {
        &self.config
    }
}

impl DraftModelProposer {
    /// Lockstep + K-iter chain (HEAD's pre-extended shape). Designed
    /// to be parallelizable: the lockstep prefill uses target's verify
    /// INPUTS (not outputs) so it can run concurrently with target
    /// verify on a dedicated draft queue (Phase 8). The K-step iters
    /// then run serially after target verify finishes (they need
    /// target's sampled bonus token as seed).
    ///
    /// When the worker has already kicked off the lockstep prefill
    /// async on `draft_queue` (signalled via
    /// `DraftSeedInputs.async_lockstep_done`), this method skips the
    /// in-proposer lockstep call — the worker has already awaited it
    /// before returning the seed.
    fn propose_lockstep_k(
        &self,
        seed: &DraftSeedInputs,
        sampled: &[Vec<u32>],
        backend: &mut dyn SpecDecodeBackend,
        k: usize,
        num_reqs: usize,
    ) -> HashMap<String, Vec<u32>> {
        const DRAFT_MODEL: ModelHandle = ModelHandle(1);
        const DRAFT_KV: KvPoolHandle = KvPoolHandle(1);

        // ── 1. Lockstep prefill — draft KV mirrors target KV.
        // Skipped if the worker already ran it async on draft_queue
        // (parallel with target verify) and awaited completion.
        if !seed.async_lockstep_done {
            let prefill_req = ForwardArgmaxRequest {
                input_ids: &seed.input_ids,
                positions: &seed.positions,
                slot_mapping: &seed.slot_mapping,
                cu_seqlens_q: &seed.cu_seqlens_q,
                seqused_k: &seed.seqused_k,
                block_table: &seed.block_table,
                span_ids: None,
                // Draft/spec path is single-group (no SWA ring).
                sliding_slot_mappings: &[],
                sliding_block_tables: &[],
                block_table_stride: seed.block_table_stride,
                max_seqlen_q: seed.max_seqlen_q,
                max_seqlen_k: seed.max_seqlen_k,
                num_tokens: seed.num_tokens,
                has_spec_tokens: false,
                last_token_indices: None,
            };
            if backend
                .forward_argmax_blocking(DRAFT_MODEL, DRAFT_KV, &prefill_req)
                .is_err()
            {
                return HashMap::new();
            }
        }

        // ── 2. K decode steps on the draft.
        let mut current_tokens: Vec<u32> = sampled
            .iter()
            .map(|s| *s.last().expect("sampled_token_ids[i] non-empty"))
            .collect();
        let mut current_positions: Vec<usize> = (0..num_reqs)
            .map(|i| {
                let advance = if seed.was_spec_decode[i] {
                    sampled[i].len()
                } else {
                    seed.q_lens[i]
                };
                seed.tokens_before[i] + advance
            })
            .collect();
        let mut drafts: HashMap<String, Vec<u32>> = seed
            .req_ids
            .iter()
            .map(|rid| (rid.clone(), Vec::with_capacity(k)))
            .collect();

        let step_cu_seqlens: Vec<u32> = (0..=(num_reqs as u32)).collect();

        // Builds (positions, slot_mapping, seqused_k, max_seqlen_k) for
        // a chain iter from the per-req `current_positions[i]`. The
        // chain primitive (Phase 6) advances these on-device between
        // iters; the iter-0 values are still computed here from host
        // state. The host-loop fallback recomputes every iter.
        let build_iter = |current_positions: &[usize]| -> (Vec<u32>, Vec<u32>, Vec<u32>, usize) {
            let mut positions = Vec::with_capacity(num_reqs);
            let mut slot_mapping = Vec::with_capacity(num_reqs);
            let mut seqused_k = Vec::with_capacity(num_reqs);
            let mut max_k: usize = 0;
            for (i, &pos) in current_positions.iter().enumerate().take(num_reqs) {
                positions.push(pos as u32);
                let block_idx = pos / seed.block_size;
                let offset = pos % seed.block_size;
                let blocks = &seed.block_ids[i];
                let slot = if block_idx < blocks.len() {
                    (blocks[block_idx] as usize * seed.block_size + offset) as u32
                } else {
                    u32::MAX
                };
                slot_mapping.push(slot);
                let seq_used = pos + 1;
                seqused_k.push(seq_used as u32);
                if seq_used > max_k {
                    max_k = seq_used;
                }
            }
            (positions, slot_mapping, seqused_k, max_k)
        };

        // Iter-0 inputs (always needed: chain primitive consumes them
        // as the chain seed; loop fallback uses them as the first iter
        // before advancing on the host).
        let (iter0_positions, iter0_slot, iter0_seqused_k, iter0_max_k) =
            build_iter(&current_positions);

        // ── 3a. Phase 9 fast-fast path: the worker already ran the
        //       K-step chain speculatively in parallel with target
        //       verify (on the draft queue), seeded by draft's
        //       lockstep last-token argmax. Use the speculative
        //       drafts directly when:
        //         (a) every req's spec_seed matched target's sampled
        //             bonus token,
        //         (b) every req was actually spec-decoded AND all
        //             of its K drafts were accepted (so the spec
        //             chain's iter-0 position == proposer's
        //             current_position),
        //         (c) the speculative chain output has length K
        //             (the worker ran the correctly-sized chain).
        //       Any divergence falls through to the standard chain
        //       path (worker's speculative work was hidden behind
        //       target verify, so the miss is free apart from a
        //       small KV cache transient that gets overwritten by
        //       the corrected chain at the same slots).
        if !seed.speculative_seeds.is_empty()
            && seed.speculative_chain_drafts.len() == k
            && seed.speculative_seeds.len() == num_reqs
        {
            let mut all_hit = true;
            for (i, sampled_i) in sampled.iter().enumerate().take(num_reqs) {
                let was_spec = seed.was_spec_decode[i];
                let accepted_count = sampled_i.len();
                let expected_full_accept = seed.q_lens[i];
                let bonus = *sampled_i.last().expect("sampled non-empty");
                if !was_spec
                    || accepted_count != expected_full_accept
                    || bonus != seed.speculative_seeds[i]
                {
                    all_hit = false;
                    break;
                }
            }
            if all_hit {
                // Speculation correct on every req — return the
                // worker-side speculative drafts directly.
                for iter_drafts in seed.speculative_chain_drafts.iter() {
                    if iter_drafts.len() != num_reqs {
                        return HashMap::new();
                    }
                    for (i, &draft) in iter_drafts.iter().enumerate().take(num_reqs) {
                        drafts
                            .get_mut(&seed.req_ids[i])
                            .expect("inserted above")
                            .push(draft);
                    }
                }
                return drafts;
            }
        }

        // ── 3. Phase 6 fast path: one CB drives K forward + argmax +
        //       chain_advance dispatches. Falls back to the iter-loop
        //       when the backend hasn't implemented the chain
        //       primitive yet (CUDA, etc.).
        {
            let iter0_req = ForwardArgmaxRequest {
                input_ids: &current_tokens,
                positions: &iter0_positions,
                slot_mapping: &iter0_slot,
                cu_seqlens_q: &step_cu_seqlens,
                seqused_k: &iter0_seqused_k,
                block_table: &seed.block_table,
                span_ids: None,
                // Draft/spec path is single-group (no SWA ring).
                sliding_slot_mappings: &[],
                sliding_block_tables: &[],
                block_table_stride: seed.block_table_stride,
                max_seqlen_q: 1,
                max_seqlen_k: iter0_max_k,
                num_tokens: num_reqs,
                has_spec_tokens: false,
                last_token_indices: None,
            };
            match backend.forward_chain_k(DRAFT_MODEL, DRAFT_KV, &iter0_req, seed.block_size, k) {
                Ok(per_iter) => {
                    if per_iter.len() != k {
                        return drafts;
                    }
                    for iter_argmax in per_iter.iter() {
                        if iter_argmax.len() != num_reqs {
                            return drafts;
                        }
                        for (i, &argmax) in iter_argmax.iter().enumerate().take(num_reqs) {
                            drafts
                                .get_mut(&seed.req_ids[i])
                                .expect("inserted above")
                                .push(argmax);
                        }
                    }
                    return drafts;
                }
                Err(crate::spec_decode::BackendError::NotImplemented(_)) => {
                    // Backend hasn't wired the chain primitive — fall
                    // through to the per-iter loop.
                }
                Err(_) => {
                    // Real backend error (kernel dispatch fail, etc.).
                    // Match prior behavior: return whatever drafts we
                    // accumulated (none yet at this point).
                    return drafts;
                }
            }
        }

        // ── 3b. Loop fallback. K host-driven forward+argmax calls,
        //       one commit + wait each. Used by backends (CUDA today)
        //       that haven't wired the chain primitive.
        let mut step_positions = iter0_positions;
        let mut step_slot_mapping = iter0_slot;
        let mut step_seqused_k = iter0_seqused_k;
        let mut step_max_k = iter0_max_k;

        for step in 0..k {
            if step > 0 {
                let (p, s, sk, m) = build_iter(&current_positions);
                step_positions = p;
                step_slot_mapping = s;
                step_seqused_k = sk;
                step_max_k = m;
            }

            let step_req = ForwardArgmaxRequest {
                input_ids: &current_tokens,
                positions: &step_positions,
                slot_mapping: &step_slot_mapping,
                cu_seqlens_q: &step_cu_seqlens,
                seqused_k: &step_seqused_k,
                block_table: &seed.block_table,
                span_ids: None,
                // Draft/spec path is single-group (no SWA ring).
                sliding_slot_mappings: &[],
                sliding_block_tables: &[],
                block_table_stride: seed.block_table_stride,
                max_seqlen_q: 1,
                max_seqlen_k: step_max_k,
                num_tokens: num_reqs,
                has_spec_tokens: false,
                last_token_indices: None,
            };
            let step_argmax =
                match backend.forward_argmax_blocking(DRAFT_MODEL, DRAFT_KV, &step_req) {
                    Ok(v) => v,
                    Err(_) => return drafts,
                };
            if step_argmax.len() != num_reqs {
                return drafts;
            }
            for i in 0..num_reqs {
                let d = step_argmax[i];
                drafts
                    .get_mut(&seed.req_ids[i])
                    .expect("inserted above")
                    .push(d);
                current_tokens[i] = d;
                current_positions[i] += 1;
            }
        }
        drafts
    }
}

impl Proposer for DraftModelProposer {
    fn propose_for_step(&mut self, ctx: &mut ProposerStepCtx<'_>) -> HashMap<String, Vec<u32>> {
        let Some(seed) = ctx.draft_seed else {
            return HashMap::new();
        };
        let Some(sampled) = ctx.sampled_token_ids else {
            return HashMap::new();
        };
        let Some(backend) = ctx.backend.as_deref_mut() else {
            return HashMap::new();
        };
        let k = self.config.num_speculative_tokens;
        if k == 0 {
            return HashMap::new();
        }
        let num_reqs = seed.req_ids.len();
        if num_reqs == 0 || sampled.len() != num_reqs {
            return HashMap::new();
        }

        // Lockstep+K (Phase 8 friendly — lockstep can overlap with
        // target verify on a separate Metal queue).
        self.propose_lockstep_k(seed, sampled, backend, k, num_reqs)
    }
}
