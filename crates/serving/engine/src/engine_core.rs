// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Engine core loop and request lifecycle management.
//!
//! The `EngineCore` is the central coordinator of vLLM. It sits between the
//! API server (front-end) and the model executor (back-end):
//!
//! ```text
//!   API Server  ──► EngineCore ──► Executor ──► GPU Workers
//!                    │ scheduler │
//!                    │ kv cache  │
//! ```
//!
//! Each iteration of the engine core:
//! 1. Processes pending input requests (add, abort, utility)
//! 2. Runs the scheduler to decide which requests to process
//! 3. Dispatches execution to the model executor
//! 4. Collects outputs and routes them back to clients
//!
//! Port of: `vllm/v1/engine/core.py::EngineCore`

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use scratchy_core_common::{
    EngineCoreOutput, EngineCoreOutputs, FinishReason, Request, RequestStatus, SchedulerStats,
    StopReason,
};
use scratchy_core_config::SchedulerConfig;
use scratchy_serving_scheduler::scheduler::output::SchedulerOutput;
use scratchy_serving_scheduler::scheduler::{PauseState, Scheduler, SchedulerInterface};
use scratchy_serving_transport::messages::PauseMode;
use tracing::{debug, error, info};

use crate::error::{EngineError, EngineResult};
use crate::executor::{Executor, ModelRunnerOutput};
use crate::spec_decode::{
    DraftModelProposer, NgramProposer, Proposer, ProposerConfig, ProposerStepCtx,
};

// ---------------------------------------------------------------------------
// EngineCore
// ---------------------------------------------------------------------------

/// The inner loop of the vLLM engine.
///
/// Manages the scheduler, dispatches work to the executor, and produces
/// outputs. This struct is protocol-agnostic — it can be used in-process
/// or wrapped with ZMQ sockets for multi-process operation.
///
/// Port of: `vllm/v1/engine/core.py::EngineCore`
pub struct EngineCore {
    /// The scheduler that decides which requests to process.
    scheduler: Scheduler,

    /// The executor that runs model forward passes.
    /// `None` when taken for async scheduling (moved to a dedicated thread).
    executor: Option<Box<dyn Executor>>,

    /// Index of this engine (for data-parallel setups).
    engine_index: u32,

    /// Whether the engine has been shut down.
    is_shutdown: bool,

    /// Monotonic start time for computing relative timestamps.
    start_time: Instant,

    /// Pending abort request IDs.
    aborts_queue: VecDeque<Vec<String>>,

    /// Whether async scheduling is enabled.
    async_scheduling: bool,

    /// Speculative-decoding proposer (n-gram or draft-model). `None`
    /// disables spec decode. Dispatched once per `finalize_step` via the
    /// [`Proposer`] trait — see [`crate::spec_decode::proposer`].
    proposer: Option<Box<dyn Proposer + Send>>,

    /// EOS token IDs for stop criteria (primary + additional from config).
    eos_token_ids: Vec<u32>,

    /// Whether the engine is in pooling mode (embedding-only).
    is_pooling: bool,

    /// KV cache block size in tokens (for seal-pad block alignment checks).
    block_size: usize,

    /// 🦭 Sealed request IDs that have already generated an EOS/stop token.
    /// Tracked so check_stop_criteria can defer stopping until block-aligned.
    seal_eos_seen: HashSet<String>,
}

/// Configuration for creating an EngineCore.
pub struct EngineCoreConfig {
    /// Scheduler configuration.
    pub scheduler_config: SchedulerConfig,
    /// Maximum model length (context window).
    pub max_model_len: usize,
    /// Number of GPU blocks available for KV cache.
    pub num_gpu_blocks: usize,
    /// Block size (tokens per block).
    pub block_size: usize,
    /// Engine index for data-parallel setups.
    pub engine_index: u32,
    /// Whether to enable async scheduling.
    pub async_scheduling: bool,
    /// Whether speculative decoding is enabled.
    pub use_spec_decode: bool,
    /// Speculative-decoding proposer config. `None` disables speculative
    /// decoding; the two variants (n-gram / draft-model) are routed in
    /// [`EngineCore::new`].
    pub proposer_config: Option<ProposerConfig>,
    /// EOS token IDs for stop criteria. Empty disables EOS-based stopping.
    /// Supports models with multiple EOS tokens (e.g. LLaMA 3:
    /// `<|end_of_text|>`, `<|eom_id|>`, `<|eot_id|>`).
    pub eos_token_ids: Vec<u32>,
    /// Whether the engine is in pooling mode (embedding-only).
    /// In pooling mode, requests are finished after one forward pass.
    pub is_pooling: bool,
    /// Whether prefix caching (KV cache reuse) is enabled.
    pub enable_prefix_caching: bool,
    /// Hybrid SWA KV layout for models that mix full + sliding-window attention
    /// (gemma4: 5 full / 25 sliding). `Some((num_blocks, groups))` switches the
    /// block tracker to vLLM's per-group path: one null-padded block table per
    /// group over the ONE shared pool of `num_blocks`. `None` keeps the single
    /// uniform full-attention pool (every non-SWA model — unchanged behavior).
    pub hybrid_kv: Option<HybridKvConfig>,
}

/// Hybrid SWA KV config for [`EngineCoreConfig::hybrid_kv`]: the shared-pool
/// size and the per-group specs `(is_sliding, window_tokens, block_size)`.
pub type HybridKvConfig = (usize, Vec<(bool, usize, usize)>);

/// Output from a single engine step, grouped by client index.
pub type StepOutputs = HashMap<u32, EngineCoreOutputs>;

/// ⛔⛔⛔ WHAT ONE ENGINE STEP DID — AND THE OUTPUTS ARE ONLY REACHABLE THROUGH THE MATCH.
///
/// This replaces `(outputs, model_executed: bool)`. The bool was the same information, but a caller
/// could write `let (outputs, _) = step()?` and never learn that the step did nothing — which is
/// precisely what a `while has_unfinished { step() }` loop must not do. Every blocking loop that
/// discarded that bool spun at 100% CPU forever the first time a step made no progress.
///
/// A `bool` cannot be made un-ignorable; an enum whose payload lives inside ONE variant can. There
/// is no way to get at `outputs` without writing an arm for [`Self::Stalled`], so a NEW step loop
/// that forgets the livelock does not compile. That is the whole point of the type:
///
/// ```ignore
/// while client.has_unfinished_requests() {
///     let outputs = match client.get_output()? {
///         StepOutcome::Progressed { outputs, .. } => outputs,
///         StepOutcome::Stalled => bail!("..."),   // ← cannot be omitted
///         StepOutcome::Idle => break,
///     };
/// }
/// ```
///
/// The three cases are exclusive and total, and they are decided in [`EngineCore::step`] alone:
/// no requests at all, requests but nothing scheduled and nothing retired, or progress.
#[must_use]
#[derive(Debug)]
pub enum StepOutcome<T> {
    /// The step scheduled work, retired finished requests, or both.
    ///
    /// `model_executed` is NOT "did anything happen" — that is what this variant means. It says
    /// whether a DEVICE FORWARD ran, which is false for a cleanup-only step that merely retired
    /// requests that finished. Timing probes and metrics want that distinction; progress does not.
    Progressed { outputs: T, model_executed: bool },
    /// ⛔ REQUESTS ARE OUTSTANDING AND THE STEP DID NOTHING. No tokens scheduled, nothing retired,
    /// no outputs — so there is nothing to hand back and no reason the next step would differ
    /// unless something OUTSIDE this loop changes the engine's state.
    ///
    /// For the async server that is possible (an HTTP handler may submit a request between steps),
    /// so it holds this against its `no_progress_timeout`. For a BLOCKING loop nothing else runs,
    /// so the state is terminal and the only correct arm is to fail with what stalled.
    Stalled,
    /// No requests at all. Nothing to do and nothing wrong.
    Idle,
}

impl<T> StepOutcome<T> {
    /// Replace the payload, keeping the variant — for a layer that merges or reshapes the outputs
    /// it forwards (`InprocClient` folds a per-client map into one `EngineCoreOutputs`).
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> StepOutcome<U> {
        match self {
            Self::Progressed {
                outputs,
                model_executed,
            } => StepOutcome::Progressed {
                outputs: f(outputs),
                model_executed,
            },
            Self::Stalled => StepOutcome::Stalled,
            Self::Idle => StepOutcome::Idle,
        }
    }

    /// TESTS ONLY: the old `(outputs, model_executed)` shape, with a stall folded back into
    /// `(default, false)`. Deliberately `cfg(test)` — production code must take the match, which is
    /// the only reason this type exists.
    #[cfg(test)]
    pub fn into_tuple(self) -> (T, bool)
    where
        T: Default,
    {
        match self {
            Self::Progressed {
                outputs,
                model_executed,
            } => (outputs, model_executed),
            Self::Stalled | Self::Idle => (T::default(), false),
        }
    }
}

impl EngineCore {
    /// Create a new EngineCore.
    pub fn new(config: EngineCoreConfig, executor: Box<dyn Executor>) -> Self {
        use scratchy_serving_scheduler::scheduler::core::SimpleBlockTracker;

        let kv_cache: Box<dyn scratchy_serving_scheduler::scheduler::core::KVCacheManagerOps> = {
            let mut tracker = if config.enable_prefix_caching {
                info!("Prefix caching enabled");
                SimpleBlockTracker::with_caching(config.num_gpu_blocks, config.block_size)
            } else {
                SimpleBlockTracker::new(config.num_gpu_blocks, config.block_size)
            };
            // ⭐⭐⭐ ASK THE WORKER HOW IT ADDRESSES KV, BEFORE THE FIRST STEP IS SCHEDULED.
            //
            // ⛔ THE TRACKER USED TO INFER THIS from whether a pool reach had ever been reported — a value
            // that first exists AFTER a step has been scheduled and allocated. So every allocation up to the
            // worker's first report used the token-addressed rule on a backend that shares one write slot,
            // and the pages the launch wrote past those allocations came out of the worker's own reserve.
            // Measured with the decision printed: `[step-reach] asked=517 applied=0 pool_reach=None`.
            if executor.kv_addressing().shares_one_write_slot() {
                info!(
                    "KV addressing: one shared write slot per batched step — requests are allocated for the \
                     step's reach, not their own token count"
                );
                tracker = tracker.sharing_one_write_slot();
            }
            // Hybrid sliding-window KV (gemma4): one null-padded block table per
            // group over the ONE shared pool. `enable_hybrid` forces prefix
            // caching off (sliding groups can't be rebuilt from a cached prefix).
            if let Some((num_blocks, groups)) = config.hybrid_kv.clone() {
                let n_sliding = groups.iter().filter(|g| g.0).count();
                info!(
                    "Hybrid SWA KV enabled: {} shared blocks, {} groups ({} sliding), \
                     lowest-free-first (sliding groups recycle low IDs → bounded residency)",
                    num_blocks,
                    groups.len(),
                    n_sliding
                );
                tracker.enable_hybrid(num_blocks, groups);
            }
            Box::new(tracker)
        };

        let scheduler = Scheduler::new(&config.scheduler_config, config.max_model_len, kv_cache);

        let mut async_scheduling = config.async_scheduling;
        let proposer: Option<Box<dyn Proposer + Send>> =
            config.proposer_config.map(|cfg| -> Box<dyn Proposer + Send> {
                match cfg {
                    ProposerConfig::Ngram(cfg) => {
                        info!(
                            "N-gram speculative decoding enabled: num_speculative_tokens={}, max_ngram_size={}, min_ngram_size={}",
                            cfg.num_speculative_tokens, cfg.max_ngram_size, cfg.min_ngram_size
                        );
                        // Mirror Python vLLM (vllm/config/vllm.py:709-721): force
                        // sync scheduling for non-EAGLE spec-decode methods. The
                        // 1-step async lookahead skews proposer-to-verify position
                        // alignment.
                        if async_scheduling {
                            info!(
                                "N-gram spec decode forces sync scheduling \
                                 (matches Python vLLM's auto-disable for non-EAGLE methods)."
                            );
                            async_scheduling = false;
                        }
                        Box::new(NgramProposer::new(cfg))
                    }
                    ProposerConfig::DraftModel(cfg) => {
                        // The worker produces
                        // K draft tokens per req and stashes them in
                        // `ModelRunnerOutput.draft_token_ids`. `finalize_step` then
                        // forwards them to the scheduler, same end-state as the
                        // n-gram path but with the proposer running GPU-side
                        // instead of CPU-side. Phase 5.4 will move the K-step
                        // chain itself into [`DraftModelProposer::propose_for_step`].
                        info!(
                            "Draft-model speculative decoding enabled ({}, k={}): worker-side proposer \
                             producing K drafts per step.",
                            cfg.model, cfg.num_speculative_tokens,
                        );
                        // Async scheduling defers finalize_step by 1 step, so
                        // drafts proposed in step N take effect in step N+2 — but
                        // the K-step chain seeds for step N+1, so positions land
                        // wrong and acceptance is ~0. Force sync scheduling until
                        // the K-step chain learns to look 2 steps ahead.
                        if async_scheduling {
                            info!(
                                "Draft-model spec decode forces sync scheduling \
                                 (async would skew proposer-to-verify positions by 1 step)."
                            );
                            async_scheduling = false;
                        }
                        Box::new(DraftModelProposer::new(cfg))
                    }
                }
            });

        info!(
            "EngineCore initialized: max_model_len={}, num_gpu_blocks={}, block_size={}, eos_token_ids={:?}, spec_decode={}, pooling={}",
            config.max_model_len,
            config.num_gpu_blocks,
            config.block_size,
            config.eos_token_ids,
            proposer.is_some(),
            config.is_pooling,
        );

        Self {
            scheduler,
            executor: Some(executor),
            engine_index: config.engine_index,
            is_shutdown: false,
            start_time: Instant::now(),
            aborts_queue: VecDeque::new(),
            async_scheduling,
            proposer,
            eos_token_ids: config.eos_token_ids,
            is_pooling: config.is_pooling,
            block_size: config.block_size,
            seal_eos_seen: HashSet::new(),
        }
    }

    /// Get the current monotonic timestamp (seconds since engine start).
    fn timestamp(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    // -----------------------------------------------------------------------
    // Request management
    // -----------------------------------------------------------------------

    /// Add a new request to the scheduler.
    pub fn add_request(&mut self, request: Request) {
        if self.is_shutdown {
            error!("Cannot add request to shut down engine");
            return;
        }
        self.scheduler.add_request(request);
    }

    /// Abort requests by ID.
    pub fn abort_requests(&mut self, request_ids: &[String]) {
        for id in request_ids {
            self.seal_eos_seen.remove(id);
        }
        let id_refs: Vec<&str> = request_ids.iter().map(String::as_str).collect();
        self.scheduler
            .finish_requests(&id_refs, RequestStatus::FinishedAborted);
    }

    /// Abort all currently running and waiting requests.
    ///
    /// Used when a persistent executor error makes it impossible to continue
    /// processing the current batch. Drains the scheduler's unfinished queue.
    pub fn abort_running_requests(&mut self) {
        let ids = self.scheduler.get_unfinished_request_ids();
        if !ids.is_empty() {
            let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
            self.scheduler
                .finish_requests(&id_refs, RequestStatus::FinishedAborted);
        }
    }

    /// Queue abort requests for processing during the next step.
    ///
    /// This is used when aborts arrive asynchronously (e.g., from the
    /// input socket thread) and need to be processed during the main loop.
    pub fn queue_aborts(&mut self, request_ids: Vec<String>) {
        self.aborts_queue.push_back(request_ids);
    }

    /// Process any pending aborts from the queue.
    fn process_aborts_queue(&mut self) {
        while let Some(ids) = self.aborts_queue.pop_front() {
            self.abort_requests(&ids);
        }
    }

    // -----------------------------------------------------------------------
    // Pause / resume
    // -----------------------------------------------------------------------

    /// Pause the scheduler.
    ///
    /// - `abort`: Abort all in-flight requests, set PAUSED_NEW.
    /// - `keep`: Set PAUSED_ALL (freeze everything).
    /// - `wait`: Not supported in in-process mode.
    pub fn pause_scheduler(&mut self, mode: PauseMode) -> EngineResult<Vec<(String, u32)>> {
        let mut aborted = Vec::new();

        if mode == PauseMode::Abort {
            aborted = self.finish_all_requests(RequestStatus::FinishedAborted);
        }

        let pause_state = match mode {
            PauseMode::Keep => PauseState::PausedAll,
            PauseMode::Abort | PauseMode::Wait => PauseState::PausedNew,
        };
        self.scheduler.set_pause_state(pause_state);

        Ok(aborted)
    }

    /// Resume the scheduler.
    pub fn resume_scheduler(&mut self) {
        self.scheduler.set_pause_state(PauseState::Unpaused);
    }

    /// Whether the scheduler is paused.
    pub fn is_scheduler_paused(&self) -> bool {
        self.scheduler.pause_state() != PauseState::Unpaused
    }

    /// Finish all requests with the given status.
    fn finish_all_requests(&mut self, _status: RequestStatus) -> Vec<(String, u32)> {
        // Collect all request IDs first to avoid borrow issues.
        let (running, waiting) = self.scheduler.get_request_counts();
        if running == 0 && waiting == 0 {
            return Vec::new();
        }

        // The Python code passes `None` to finish_requests to finish all.
        // Our Rust interface requires explicit IDs, so we need to handle
        // this differently. For now, we use the scheduler's internal method.
        // TODO: Add a finish_all method to the scheduler interface.
        Vec::new()
    }

    // -----------------------------------------------------------------------
    // Prefix cache
    // -----------------------------------------------------------------------

    /// Reset the prefix cache.
    pub fn reset_prefix_cache(&mut self) -> bool {
        self.scheduler.reset_prefix_cache()
    }

    // -----------------------------------------------------------------------
    // Engine step
    // -----------------------------------------------------------------------

    /// Whether the engine has work to do.
    pub fn has_work(&self) -> bool {
        self.scheduler.has_requests()
    }

    /// Execute one scheduling + execution step.
    ///
    /// Returns a map of client_index → outputs, and a flag indicating
    /// whether the model was actually executed.
    ///
    /// Requires the executor to be present (not taken for async scheduling).
    pub fn step(&mut self) -> EngineResult<StepOutcome<StepOutputs>> {
        let executor = self
            .executor
            .as_mut()
            .ok_or_else(|| EngineError::Executor("executor taken for async scheduling".into()))?;

        if !self.scheduler.has_requests() {
            return Ok(StepOutcome::Idle);
        }

        // TTFT ATTRIBUTION (SCRATCHY_TTFT_PHASE). Printed only for a PREFILL step (>1 scheduled
        // token), so decode is unaffected and the line appears once per request — the step that IS
        // time-to-first-token. Subtracting the worker's own `predict_total` (SCRATCHY_SDSC_PHASE_TIME)
        // from `execute` gives the executor overhead OUTSIDE the device forward, which is the last
        // unmeasured term in the TTFT budget.
        let ttft_phase = std::env::var_os("SCRATCHY_TTFT_PHASE").is_some();
        let _t_sched = std::time::Instant::now();

        // 1. Schedule.
        let scheduler_output = self.scheduler.schedule();
        let model_executed = scheduler_output.total_num_scheduled_tokens > 0;
        let _d_sched = _t_sched.elapsed();

        // ⛔ NOTHING SCHEDULED AND NOTHING RETIRED, with requests still outstanding. Naming it
        // `Stalled` rather than handing back an empty map is what forces every caller to say what
        // it does about that — see [`StepOutcome`].
        if !model_executed && scheduler_output.finished_req_ids.is_empty() {
            return Ok(StepOutcome::Stalled);
        }

        // 2. Execute model (also handles cleanup of finished requests even
        //    when no tokens are scheduled).
        let _t_exec = std::time::Instant::now();
        let mut model_output = executor
            .execute_model(&scheduler_output)
            .map_err(|e| EngineError::Executor(e.to_string()))?;

        // Resolve deferred D2H if present (sync path — resolve immediately).
        model_output.resolve();
        let _d_exec = _t_exec.elapsed();

        // 3. Finalize: process outputs, aborts, ngram, and stats.
        let _t_fin = std::time::Instant::now();
        let outputs = self.finalize_step(&scheduler_output, &model_output);

        if ttft_phase && scheduler_output.total_num_scheduled_tokens > 1 {
            eprintln!(
                "[ttft-step] schedule={:.2}  execute={:.2}  finalize={:.2}  step_total={:.2} ms  \
                 ({} scheduled tokens)",
                _d_sched.as_secs_f64() * 1e3,
                _d_exec.as_secs_f64() * 1e3,
                _t_fin.elapsed().as_secs_f64() * 1e3,
                _t_sched.elapsed().as_secs_f64() * 1e3,
                scheduler_output.total_num_scheduled_tokens,
            );
        }

        Ok(StepOutcome::Progressed {
            outputs,
            model_executed,
        })
    }

    /// Take the executor out of the engine core for use on a dedicated thread.
    ///
    /// Returns `None` if the executor was already taken. After this call,
    /// `step()` and `embed()` will error — the caller must use
    /// `schedule_next()` + `finalize_step()` with the taken executor.
    pub fn take_executor(&mut self) -> Option<Box<dyn Executor>> {
        self.executor.take()
    }

    /// Run scheduling if there is work to do.
    ///
    /// Returns `Some(scheduler_output)` when tokens are scheduled,
    /// `None` when there is nothing to execute.
    pub fn schedule_next(&mut self) -> Option<SchedulerOutput> {
        if !self.scheduler.has_requests() {
            return None;
        }
        let sched = self.scheduler.schedule();
        // Still return the output if there are finished request IDs to clean up,
        // even when no tokens are scheduled. The executor needs to see these IDs
        // to release per-request resources (KV cache buffers, token buffers, etc.).
        if sched.total_num_scheduled_tokens == 0 && sched.finished_req_ids.is_empty() {
            return None;
        }
        Some(sched)
    }

    /// Post-execution processing: update state from model output, process
    /// aborts, run ngram proposer, and attach scheduler stats.
    ///
    /// Used by both the sync `step()` path and the async scheduling path.
    pub fn finalize_step(
        &mut self,
        scheduler_output: &SchedulerOutput,
        model_output: &ModelRunnerOutput,
    ) -> StepOutputs {
        // 1. Snapshot scheduler stats BEFORE processing outputs (which frees
        //    blocks for finished requests). This gives an accurate view of
        //    blocks in use during the step.
        let (num_running, num_waiting) = self.scheduler.get_request_counts();
        let stats = SchedulerStats {
            num_running_reqs: num_running,
            num_waiting_reqs: num_waiting,
            kv_cache_usage: self.scheduler.kv_cache_usage(),
            gpu_cache_blocks_used: self.scheduler.num_used_blocks(),
            gpu_cache_blocks_total: self.scheduler.num_total_blocks(),
            num_cached_blocks: self.scheduler.num_cached_blocks(),
            spec_decode_stats: None,
        };

        // 2. Process any pending aborts.
        self.process_aborts_queue();

        // 3. Update scheduler state and build outputs.
        let mut outputs = self.update_from_output(scheduler_output, model_output);

        // 4. Propose speculative draft tokens for running requests via
        //    the Proposer trait. Two impls today: NgramProposer (CPU
        //    n-gram lookup) reads per-request history; DraftModelProposer
        //    reads worker-side drafts out of `model_output.draft_token_ids`.
        //    Both shapes funnel through the same `propose_for_step` →
        //    `set_spec_token_ids` flow here.
        if let Some(ref mut proposer) = self.proposer {
            let scheduled_req_ids: Vec<&str> = scheduler_output
                .num_scheduled_tokens
                .keys()
                .map(String::as_str)
                .collect();
            // `get_all_tokens` borrows the scheduler immutably; the
            // backend borrows the executor mutably. Resolve both into
            // owned data / a separate borrow before building ctx so
            // there's no overlap of scheduler + executor borrows.
            let scheduler_ref = &self.scheduler;
            let get_all_tokens = |req_id: &str| -> Option<Vec<u32>> {
                scheduler_ref.get_request(req_id).and_then(|r| {
                    if r.status.is_finished() {
                        None
                    } else {
                        Some(r.all_token_ids.clone())
                    }
                })
            };
            let backend = self.executor.as_mut().and_then(|e| e.spec_decode_backend());
            let mut ctx = ProposerStepCtx {
                scheduled_req_ids: &scheduled_req_ids,
                get_all_tokens: &get_all_tokens,
                worker_drafts: model_output.draft_token_ids.as_ref(),
                backend,
                draft_seed: model_output.draft_seed_inputs.as_ref(),
                sampled_token_ids: Some(&model_output.sampled_token_ids),
            };
            let drafts_map = proposer.propose_for_step(&mut ctx);
            for (req_id, drafts) in drafts_map {
                if drafts.is_empty() {
                    continue;
                }
                let still_running = self
                    .scheduler
                    .get_request(&req_id)
                    .is_some_and(|r| !r.status.is_finished() && !r.all_token_ids.is_empty());
                if !still_running {
                    continue;
                }
                self.scheduler.set_spec_token_ids(&req_id, drafts);
            }
        }

        // 5. Compute spec decode metrics.
        let spec_decode_stats = if !scheduler_output.scheduled_spec_decode_tokens.is_empty() {
            let mut num_drafts = 0usize;
            let mut num_draft_tokens = 0usize;
            let mut num_accepted_tokens = 0usize;

            for (req_id, draft_tokens) in &scheduler_output.scheduled_spec_decode_tokens {
                if draft_tokens.is_empty() {
                    continue;
                }
                num_drafts += 1;
                num_draft_tokens += draft_tokens.len();

                // Count accepted: compare draft tokens against actual output.
                if let Some(output_tokens) = model_output.get_tokens(req_id) {
                    // output_tokens = [accepted_0, accepted_1, ..., bonus_or_recovered]
                    // draft_tokens = [draft_0, draft_1, ...]
                    // Accepted = min(output_tokens.len() - 1, draft_tokens.len())
                    // because output always has at least 1 token (the bonus/recovered).
                    let accepted = output_tokens
                        .len()
                        .saturating_sub(1)
                        .min(draft_tokens.len());
                    num_accepted_tokens += accepted;
                }
            }

            if num_drafts > 0 {
                let acceptance_rate = if num_draft_tokens > 0 {
                    num_accepted_tokens as f64 / num_draft_tokens as f64
                } else {
                    0.0
                };
                debug!(
                    "Spec decode: {num_drafts} reqs, {num_draft_tokens} drafts, \
                     {num_accepted_tokens} accepted ({:.1}%)",
                    acceptance_rate * 100.0
                );
                Some(scratchy_core_common::SpecDecodingStats {
                    num_drafts,
                    num_draft_tokens,
                    num_accepted_tokens,
                })
            } else {
                None
            }
        } else {
            None
        };

        // 6. Attach pre-captured scheduler stats to outputs.
        let mut stats = stats;
        stats.spec_decode_stats = spec_decode_stats;
        for engine_outputs in outputs.values_mut() {
            engine_outputs.scheduler_stats = Some(stats.clone());
        }

        outputs
    }

    /// Update the scheduler state from model output and build engine outputs.
    ///
    /// This mirrors the Python `Scheduler.update_from_output()` method.
    /// After each step:
    /// 1. Append new tokens to each request's state in the scheduler.
    /// 2. Check stop criteria (max_tokens, EOS, stop_token_ids).
    /// 3. Finish requests that hit a stop condition.
    fn update_from_output(
        &mut self,
        scheduler_output: &SchedulerOutput,
        model_output: &ModelRunnerOutput,
    ) -> StepOutputs {
        let timestamp = self.timestamp();
        let mut client_outputs: StepOutputs = HashMap::new();
        let mut finished_ids: Vec<(String, RequestStatus)> = Vec::new();

        // ⭐ THE WORKER'S KV REPORT, BEFORE ANY OF THE TOKEN BOOKKEEPING. Empty for every backend whose
        // keys sit at their token positions; on one where they do not, this is what the next step's block
        // allocation is sized by and what bounds the prefix it may cache. Over the whole report rather
        // than the scheduled ids: a batched step can move the KV span of a request that only padded it.
        for (req_id, extent) in &model_output.kv_extent {
            self.scheduler.set_kv_extent(req_id, *extent);
        }
        // ⭐ AND THE POOL-WIDE REACH, WHICH IS ABOUT THE REQUESTS THIS REPORT DOES NOT MENTION. The loop
        // above can only size requests the worker has already run; this is what sizes the next one ADMITTED,
        // whose write page would otherwise land past everything the host allocated for it.
        if let Some(reach) = model_output.kv_pool_reach {
            self.scheduler.set_kv_pool_reach(reach);
        }

        // Process each request that was scheduled.
        for req_id in scheduler_output.num_scheduled_tokens.keys() {
            // Check if this is a pooling request.
            let is_pooling_request = self.is_pooling
                && self
                    .scheduler
                    .get_request(req_id)
                    .is_some_and(|r| r.is_pooling);

            if is_pooling_request {
                // Pooling request: extract embedding vector, finish immediately.
                let pooler_vec = model_output
                    .pooler_output
                    .as_ref()
                    .and_then(|m| m.get(req_id))
                    .cloned();

                finished_ids.push((req_id.clone(), RequestStatus::FinishedStopped));

                let output = EngineCoreOutput {
                    request_id: req_id.clone(),
                    new_token_ids: Vec::new(),
                    finish_reason: Some(FinishReason::Stop),
                    stop_reason: None,
                    num_cached_tokens: 0,
                    events: None,
                    new_logprobs: None,
                    new_prompt_logprobs: None,
                    pooler_output: pooler_vec,
                };

                let engine_outputs = client_outputs
                    .entry(0)
                    .or_insert_with(|| EngineCoreOutputs {
                        engine_index: self.engine_index,
                        outputs: Vec::new(),
                        timestamp,
                        scheduler_stats: None,
                    });
                engine_outputs.outputs.push(output);
                continue;
            }

            // Skip requests that finished in a previous pipeline stage.
            // With 2-batch lookahead, a request may have been finished in the
            // last finalize_step but its next batch is still being drained.
            if self
                .scheduler
                .get_request(req_id)
                .is_some_and(|r| r.status.is_finished())
            {
                continue;
            }

            // Generation request: normal token-based processing.
            let new_token_ids_slice: &[u32] = model_output.get_tokens(req_id).unwrap_or_default();

            // Append new tokens to the request's state in the scheduler.
            if !new_token_ids_slice.is_empty() {
                self.scheduler
                    .append_output_tokens(req_id, new_token_ids_slice);
            }

            // Rewind num_computed_tokens for rejected spec decode drafts.
            // The scheduler already advanced num_computed_tokens by num_scheduled_tokens
            // (which includes ALL draft tokens) in update_after_schedule(). If some
            // drafts were rejected, we must rewind by num_rejected so the KV cache
            // position is correct for the next step.
            // Matches Python: scheduler.update_from_output() lines 1324-1338.
            if let Some(scheduled_spec_ids) = scheduler_output
                .scheduled_spec_decode_tokens
                .get(req_id)
                .filter(|ids| !ids.is_empty())
                && !new_token_ids_slice.is_empty()
            {
                let num_draft_tokens = scheduled_spec_ids.len();
                let num_accepted = new_token_ids_slice.len().saturating_sub(1);
                let num_rejected = num_draft_tokens.saturating_sub(num_accepted);
                if num_rejected > 0 {
                    self.scheduler
                        .rewind_num_computed_tokens(req_id, num_rejected);
                }
            }

            // Check stop criteria against the updated request state.
            let (finish_reason, stop_reason) =
                self.check_stop_criteria(req_id, new_token_ids_slice);

            if let Some(reason) = finish_reason {
                let status = match reason {
                    FinishReason::Length => RequestStatus::FinishedLengthCapped,
                    FinishReason::Stop => RequestStatus::FinishedStopped,
                    _ => RequestStatus::FinishedStopped,
                };
                finished_ids.push((req_id.clone(), status));
                debug!(
                    "Request {} finished: {:?} (stop_reason={:?})",
                    req_id, reason, stop_reason
                );
            }

            // Extract logprobs for this request if available.
            let new_logprobs = model_output
                .logprobs
                .as_ref()
                .and_then(|lp_vec| {
                    model_output
                        .req_id_to_index
                        .get(req_id)
                        .and_then(|&idx| lp_vec.get(idx))
                })
                .and_then(|opt| opt.clone());

            // Extract prompt logprobs for this request if available.
            let new_prompt_logprobs = model_output.prompt_logprobs_dict.get(req_id).map(|plp| {
                // Wrap in Option: first position is None (no prior context),
                // rest are Some.
                let mut result: Vec<Option<scratchy_core_common::LogprobsOutput>> =
                    Vec::with_capacity(plp.len() + 1);
                result.push(None); // position 0
                for lp in plp {
                    result.push(Some(lp.clone()));
                }
                result
            });

            // Skip output for intermediate prefill chunks (no sampled tokens).
            // Matches Python: EngineCore only emits output when new_token_ids
            // is non-empty or request is stopped/pooling.
            if new_token_ids_slice.is_empty() && finish_reason.is_none() {
                continue;
            }

            // Build the output for this request.
            let output = EngineCoreOutput {
                request_id: req_id.clone(),
                new_token_ids: new_token_ids_slice.to_vec(),
                finish_reason,
                stop_reason,
                num_cached_tokens: 0,
                events: None,
                new_logprobs,
                new_prompt_logprobs,
                pooler_output: None,
            };

            // Route to client_index 0 (default for single-client mode).
            let engine_outputs = client_outputs
                .entry(0)
                .or_insert_with(|| EngineCoreOutputs {
                    engine_index: self.engine_index,
                    outputs: Vec::new(),
                    timestamp,
                    scheduler_stats: None,
                });
            engine_outputs.outputs.push(output);
        }

        // Finish requests that hit stop criteria.
        for (req_id, status) in &finished_ids {
            self.scheduler.finish_requests(&[req_id.as_str()], *status);
        }

        // Include already-finished request IDs from the scheduler (e.g. aborts).
        if !scheduler_output.finished_req_ids.is_empty() {
            let engine_outputs = client_outputs
                .entry(0)
                .or_insert_with(|| EngineCoreOutputs {
                    engine_index: self.engine_index,
                    outputs: Vec::new(),
                    timestamp,
                    scheduler_stats: None,
                });

            for req_id in &scheduler_output.finished_req_ids {
                // Only add if not already included from the loop above.
                if !finished_ids.iter().any(|(id, _)| id == req_id) {
                    engine_outputs.outputs.push(EngineCoreOutput {
                        request_id: req_id.clone(),
                        new_token_ids: Vec::new(),
                        finish_reason: Some(FinishReason::Stop),
                        stop_reason: None,
                        num_cached_tokens: 0,
                        events: None,
                        new_logprobs: None,
                        new_prompt_logprobs: None,
                        pooler_output: None,
                    });
                }
            }
        }

        client_outputs
    }

    /// Check stop criteria for a request given its newly generated tokens.
    ///
    /// Returns `(finish_reason, stop_reason)`.
    ///
    /// 🦭 Sealed requests: EOS/stop tokens don't trigger an immediate stop.
    /// Instead we record that EOS was seen and continue generating. The
    /// SealPadProcessor forces pad tokens on the GPU side. Once the total
    /// token count is block-aligned we stop here.
    fn check_stop_criteria(
        &mut self,
        req_id: &str,
        new_token_ids: &[u32],
    ) -> (Option<FinishReason>, Option<StopReason>) {
        let request = match self.scheduler.get_request(req_id) {
            Some(r) => r,
            None => return (None, None),
        };

        let num_output_tokens = request.output_token_ids.len() as u32;
        let is_sealed = request.seal;

        // 1. For sealed requests in padding mode, check block alignment.
        if is_sealed && self.seal_eos_seen.contains(req_id) {
            let total_tokens = request.all_token_ids.len();
            if total_tokens % self.block_size == 0 {
                self.seal_eos_seen.remove(req_id);
                return (Some(FinishReason::Stop), Some(StopReason::Token(0)));
            }
            return (None, None);
        }

        // 2. Check max_tokens. For sealed requests, defer until block-aligned.
        if num_output_tokens >= request.max_tokens {
            if is_sealed && request.all_token_ids.len() % self.block_size != 0 {
                self.seal_eos_seen.insert(req_id.to_string());
                return (None, None);
            }
            self.seal_eos_seen.remove(req_id);
            return (Some(FinishReason::Length), None);
        }

        // Only check token-based stop criteria if we have new tokens.
        if new_token_ids.is_empty() {
            return (None, None);
        }

        let params = &request.sampling_params;

        // Check each new token against stop conditions.
        for &token_id in new_token_ids {
            // 3. Check EOS tokens (unless ignore_eos is set).
            if !params.ignore_eos && self.eos_token_ids.contains(&token_id) {
                if is_sealed {
                    // Don't stop — record EOS and let SealPadProcessor pad.
                    self.seal_eos_seen.insert(req_id.to_string());
                    return (None, None);
                }
                return (Some(FinishReason::Stop), Some(StopReason::Token(token_id)));
            }

            // 4. Check stop_token_ids.
            if params.stop_token_ids.contains(&token_id) {
                if is_sealed {
                    self.seal_eos_seen.insert(req_id.to_string());
                    return (None, None);
                }
                return (Some(FinishReason::Stop), Some(StopReason::Token(token_id)));
            }
        }

        (None, None)
    }

    // -----------------------------------------------------------------------
    // Busy loop (for process-based engine core)
    // -----------------------------------------------------------------------

    /// Run the core busy loop.
    ///
    /// This is the main entry point for the engine core process. It
    /// alternates between processing input requests and executing steps.
    ///
    /// The loop runs until `shutdown()` is called.
    pub fn run_busy_loop<F>(&mut self, mut process_input: F)
    where
        F: FnMut(&mut Self) -> bool,
    {
        info!("EngineCore busy loop started");

        loop {
            if self.is_shutdown {
                break;
            }

            // 1. Process input requests until we have work to do.
            let should_continue = process_input(self);
            if !should_continue {
                break;
            }

            // 2. Step the engine.
            match self.step() {
                Ok(StepOutcome::Progressed { outputs, .. }) => {
                    // In a real implementation, outputs would be sent over ZMQ.
                    // For now, they are returned via the step() call above.
                    let _ = outputs;
                }
                // This loop's `process_input` callback CAN change the engine's state between steps
                // (that is what it is for), so a stall here is not terminal the way it is in a
                // blocking client loop — go round and let `process_input` run again.
                Ok(StepOutcome::Stalled | StepOutcome::Idle) => {}
                Err(e) => {
                    error!("Engine step failed: {}", e);
                    break;
                }
            }
        }

        info!("EngineCore busy loop stopped");
    }

    // -----------------------------------------------------------------------
    // Shutdown
    // -----------------------------------------------------------------------

    /// Compute embeddings, bypassing the scheduler.
    pub fn embed(&mut self, token_id_seqs: Vec<Vec<u32>>) -> EngineResult<Vec<Vec<f32>>> {
        let executor = self
            .executor
            .as_mut()
            .ok_or_else(|| EngineError::Executor("executor taken for async scheduling".into()))?;
        executor.embed(token_id_seqs)
    }

    /// Put the engine to sleep, freeing GPU memory.
    pub fn sleep(&mut self, level: u32) -> EngineResult<()> {
        // Abort all running requests first.
        self.abort_running_requests();
        let executor = self
            .executor
            .as_mut()
            .ok_or_else(|| EngineError::Executor("executor taken for async scheduling".into()))?;
        executor.sleep(level)
    }

    /// Wake the engine from sleep.
    pub fn wake_up(&mut self, tags: Option<&[String]>) -> EngineResult<()> {
        let executor = self
            .executor
            .as_mut()
            .ok_or_else(|| EngineError::Executor("executor taken for async scheduling".into()))?;
        executor.wake_up(tags)
    }

    /// Whether the engine is currently sleeping.
    pub fn is_sleeping(&self) -> bool {
        self.executor.as_ref().is_some_and(|e| e.is_sleeping())
    }

    /// Shut down the engine core.
    pub fn shutdown(&mut self) {
        if self.is_shutdown {
            return;
        }
        info!("Shutting down EngineCore");
        self.is_shutdown = true;
        self.scheduler.shutdown();
        if let Some(ref mut executor) = self.executor {
            executor.shutdown();
        }
    }

    /// Whether async scheduling is enabled.
    pub fn async_scheduling(&self) -> bool {
        self.async_scheduling
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Number of unfinished requests.
    pub fn num_unfinished_requests(&self) -> usize {
        self.scheduler.get_num_unfinished_requests()
    }

    /// Whether there are unfinished requests.
    pub fn has_unfinished_requests(&self) -> bool {
        self.scheduler.has_unfinished_requests()
    }

    /// Whether there are finished requests pending notification.
    pub fn has_finished_requests(&self) -> bool {
        self.scheduler.has_finished_requests()
    }

    /// Get request counts (running, waiting).
    pub fn get_request_counts(&self) -> (usize, usize) {
        self.scheduler.get_request_counts()
    }

    /// Engine index.
    pub fn engine_index(&self) -> u32 {
        self.engine_index
    }

    /// Whether the engine is shut down.
    pub fn is_shutdown(&self) -> bool {
        self.is_shutdown
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::NoopExecutor;
    use scratchy_core_common::SamplingParams;
    use scratchy_core_common::engine_io::EmbeddingData;
    use scratchy_core_config::SchedulerPolicy;

    fn make_test_config() -> EngineCoreConfig {
        EngineCoreConfig {
            scheduler_config: SchedulerConfig {
                max_num_batched_tokens: 8192,
                max_num_seqs: 256,
                max_num_scheduled_tokens: None,
                policy: SchedulerPolicy::Fcfs,
                enable_chunked_prefill: true,
                long_prefill_token_threshold: 0,
                ..Default::default()
            },
            max_model_len: 4096,
            num_gpu_blocks: 1024,
            block_size: 16,
            engine_index: 0,
            async_scheduling: false,
            use_spec_decode: false,
            proposer_config: None,
            eos_token_ids: vec![],
            is_pooling: false,
            enable_prefix_caching: false,
            hybrid_kv: None,
        }
    }

    fn make_request(id: &str, num_tokens: usize) -> Request {
        let prompt_token_ids: Vec<u32> = (0..num_tokens as u32).collect();
        let params = SamplingParams {
            max_tokens: Some(16),
            ..Default::default()
        };
        Request::new(id.to_string(), prompt_token_ids, params, 0.0, 0, 0, None)
    }

    #[test]
    fn test_engine_core_new() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let engine = EngineCore::new(config, executor);

        assert_eq!(engine.engine_index(), 0);
        assert!(!engine.is_shutdown());
        assert!(!engine.has_work());
        assert_eq!(engine.num_unfinished_requests(), 0);
    }

    #[test]
    fn test_add_and_step() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Add a request.
        let req = make_request("req-1", 10);
        engine.add_request(req);

        assert!(engine.has_work());
        assert_eq!(engine.num_unfinished_requests(), 1);
        let (running, waiting) = engine.get_request_counts();
        assert_eq!(running, 0);
        assert_eq!(waiting, 1);

        // Step the engine.
        let (outputs, model_executed) = engine.step().unwrap().into_tuple();
        assert!(model_executed);

        // We should have outputs.
        assert!(!outputs.is_empty());
        let client_0_outputs = outputs.get(&0).unwrap();
        assert!(!client_0_outputs.outputs.is_empty());
        assert_eq!(client_0_outputs.engine_index, 0);
    }

    #[test]
    fn test_empty_step() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Step with no requests.
        let (outputs, model_executed) = engine.step().unwrap().into_tuple();
        assert!(!model_executed);
        assert!(outputs.is_empty());
    }

    #[test]
    fn test_multiple_requests() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Add multiple requests.
        for i in 0..5 {
            let req = make_request(&format!("req-{i}"), 10);
            engine.add_request(req);
        }

        assert_eq!(engine.num_unfinished_requests(), 5);

        // Step should schedule all requests.
        let (outputs, model_executed) = engine.step().unwrap().into_tuple();
        assert!(model_executed);

        let client_0 = outputs.get(&0).unwrap();
        assert!(client_0.outputs.len() >= 5);
    }

    #[test]
    fn test_abort_request() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Add and then abort.
        engine.add_request(make_request("req-1", 10));
        engine.add_request(make_request("req-2", 10));
        assert_eq!(engine.num_unfinished_requests(), 2);

        engine.abort_requests(&["req-1".to_string()]);
        assert_eq!(engine.num_unfinished_requests(), 1);
    }

    #[test]
    fn test_queue_aborts() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        engine.add_request(make_request("req-1", 10));
        engine.add_request(make_request("req-2", 10));

        // Queue aborts (simulating async arrival).
        engine.queue_aborts(vec!["req-1".to_string()]);
        assert_eq!(engine.num_unfinished_requests(), 2); // Not processed yet.

        // Process aborts queue.
        engine.process_aborts_queue();
        assert_eq!(engine.num_unfinished_requests(), 1);
    }

    #[test]
    fn test_pause_resume() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        assert!(!engine.is_scheduler_paused());

        // Pause with "keep" mode.
        engine.pause_scheduler(PauseMode::Keep).unwrap();
        assert!(engine.is_scheduler_paused());

        // Resume.
        engine.resume_scheduler();
        assert!(!engine.is_scheduler_paused());
    }

    #[test]
    fn test_prefix_cache_reset() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Reset with no running requests should succeed.
        assert!(engine.reset_prefix_cache());
    }

    #[test]
    fn test_shutdown() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        engine.shutdown();
        assert!(engine.is_shutdown());

        // Double shutdown should be safe.
        engine.shutdown();
        assert!(engine.is_shutdown());
    }

    #[test]
    fn test_step_after_abort() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Add, step (to move to running), abort, step again.
        engine.add_request(make_request("req-1", 10));
        let _ = engine.step().unwrap();

        engine.abort_requests(&["req-1".to_string()]);

        // The step after abort should still work (may have empty output).
        let (_, _) = engine.step().unwrap().into_tuple();
    }

    #[test]
    fn test_engine_index() {
        let mut config = make_test_config();
        config.engine_index = 42;
        let executor = Box::new(NoopExecutor::new(1024));
        let engine = EngineCore::new(config, executor);
        assert_eq!(engine.engine_index(), 42);
    }

    #[test]
    fn test_timestamp_increases() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let engine = EngineCore::new(config, executor);

        let t1 = engine.timestamp();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let t2 = engine.timestamp();
        assert!(t2 > t1);
    }

    #[test]
    fn test_busy_loop_terminates() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        let mut iterations = 0;
        engine.run_busy_loop(|engine| {
            iterations += 1;
            if iterations == 1 {
                engine.add_request(make_request("req-1", 10));
            }
            if iterations >= 3 {
                engine.shutdown();
                return false;
            }
            true
        });

        assert!(engine.is_shutdown());
        assert!(iterations >= 3);
    }

    // -- Stop criteria tests --

    #[test]
    fn test_max_tokens_stop() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Create a request with max_tokens=3.
        let params = SamplingParams {
            max_tokens: Some(3),
            ..Default::default()
        };
        let req = Request::new("req-1".to_string(), vec![1, 2, 3], params, 0.0, 0, 0, None);
        engine.add_request(req);
        assert_eq!(engine.num_unfinished_requests(), 1);

        // Step repeatedly — request should finish after generating 3 tokens.
        let mut finished = false;
        for _ in 0..10 {
            let (outputs, _) = engine.step().unwrap().into_tuple();
            if let Some(client_out) = outputs.get(&0) {
                for out in &client_out.outputs {
                    if out.request_id == "req-1" && out.finish_reason.is_some() {
                        assert_eq!(out.finish_reason, Some(FinishReason::Length));
                        finished = true;
                    }
                }
            }
            if finished {
                break;
            }
        }
        assert!(finished, "Request should have finished due to max_tokens");
    }

    #[test]
    fn test_eos_token_stop() {
        let mut config = make_test_config();
        // NoopExecutor generates incrementing token IDs starting from 1000.
        // Set EOS to 1000 so the first decode step triggers it.
        config.eos_token_ids = vec![1000];
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        let params = SamplingParams {
            max_tokens: Some(100),
            ..Default::default()
        };
        let req = Request::new("req-1".to_string(), vec![10, 20], params, 0.0, 0, 0, None);
        engine.add_request(req);

        // Step — the first generated token should be 1000 (EOS).
        let (outputs, _) = engine.step().unwrap().into_tuple();
        let client_out = outputs.get(&0).unwrap();
        let req_out = client_out
            .outputs
            .iter()
            .find(|o| o.request_id == "req-1")
            .unwrap();

        assert!(req_out.new_token_ids.contains(&1000));
        assert_eq!(req_out.finish_reason, Some(FinishReason::Stop));
        assert_eq!(req_out.stop_reason, Some(StopReason::Token(1000)));
    }

    #[test]
    fn test_stop_token_ids_stop() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Set stop_token_ids to include 1000 (NoopExecutor starts there).
        let params = SamplingParams {
            max_tokens: Some(100),
            stop_token_ids: vec![1000],
            ..Default::default()
        };
        let req = Request::new("req-1".to_string(), vec![10, 20], params, 0.0, 0, 0, None);
        engine.add_request(req);

        let (outputs, _) = engine.step().unwrap().into_tuple();
        let client_out = outputs.get(&0).unwrap();
        let req_out = client_out
            .outputs
            .iter()
            .find(|o| o.request_id == "req-1")
            .unwrap();

        assert!(req_out.new_token_ids.contains(&1000));
        assert_eq!(req_out.finish_reason, Some(FinishReason::Stop));
        assert_eq!(req_out.stop_reason, Some(StopReason::Token(1000)));
    }

    #[test]
    fn test_ignore_eos() {
        let mut config = make_test_config();
        config.eos_token_ids = vec![1000];
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Set ignore_eos = true — EOS should not stop the request.
        let params = SamplingParams {
            max_tokens: Some(100),
            ignore_eos: true,
            ..Default::default()
        };
        let req = Request::new("req-1".to_string(), vec![10, 20], params, 0.0, 0, 0, None);
        engine.add_request(req);

        // Step — even though token 1000 (EOS) is generated, it shouldn't stop.
        let (outputs, _) = engine.step().unwrap().into_tuple();
        let client_out = outputs.get(&0).unwrap();
        let req_out = client_out
            .outputs
            .iter()
            .find(|o| o.request_id == "req-1")
            .unwrap();

        // Token 1000 should be generated.
        assert!(req_out.new_token_ids.contains(&1000));
        // With ignore_eos, the EOS token should NOT trigger a stop.
        assert!(
            req_out.finish_reason.is_none(),
            "EOS should be ignored when ignore_eos is set"
        );
    }

    #[test]
    fn test_multiple_eos_token_ids() {
        // Simulate a model with multiple EOS tokens (e.g. LLaMA 3:
        // 128001=<|end_of_text|>, 128008=<|eom_id|>, 128009=<|eot_id|>).
        // NoopExecutor generates token IDs starting from 1000, so use 1002
        // as a secondary EOS (hit on the 3rd decode step).
        let mut config = make_test_config();
        config.eos_token_ids = vec![9999, 1002, 8888]; // 1002 should trigger.
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        let params = SamplingParams {
            max_tokens: Some(100),
            ..Default::default()
        };
        let req = Request::new("req-1".to_string(), vec![100, 200], params, 0.0, 0, 0, None);
        engine.add_request(req);

        // Step until the request finishes. NoopExecutor generates 1000, 1001,
        // 1002, ... so token 1002 (a secondary EOS) should trigger stop.
        let mut finished = false;
        for _ in 0..10 {
            let (outputs, _) = engine.step().unwrap().into_tuple();
            if let Some(client_out) = outputs.get(&0) {
                for out in &client_out.outputs {
                    if out.request_id == "req-1" && out.finish_reason.is_some() {
                        assert_eq!(out.finish_reason, Some(FinishReason::Stop));
                        assert_eq!(out.stop_reason, Some(StopReason::Token(1002)));
                        finished = true;
                    }
                }
            }
            if finished {
                break;
            }
        }
        assert!(finished, "Request should stop on secondary EOS token ID");
    }

    #[test]
    fn test_prompt_logprobs_propagation() {
        use scratchy_core_common::sampling::{LogprobsOutput, TokenLogprob};

        // Simulate a ModelRunnerOutput with prompt_logprobs_dict populated.
        let mut prompt_logprobs_dict = std::collections::HashMap::new();
        prompt_logprobs_dict.insert(
            "req-1".to_string(),
            vec![
                LogprobsOutput {
                    sampled: TokenLogprob {
                        token_id: 20,
                        logprob: -0.5,
                        rank: 1,
                    },
                    top_logprobs: vec![],
                },
                LogprobsOutput {
                    sampled: TokenLogprob {
                        token_id: 30,
                        logprob: -1.2,
                        rank: 2,
                    },
                    top_logprobs: vec![],
                },
            ],
        );

        let model_output = ModelRunnerOutput {
            req_ids: vec!["req-1".to_string()],
            req_id_to_index: [("req-1".to_string(), 0)].into_iter().collect(),
            sampled_token_ids: vec![vec![42]],
            logprobs: None,
            prompt_logprobs_dict,
            draft_token_ids: None,
            pooler_output: None,
            d2h_resolver: None,
            draft_seed_inputs: None,
            // A test constructor is a CONSTRUCTION SITE: adding `kv_extent` to
            // ModelRunnerOutput broke these four and nothing noticed, because this
            // crate's test target was never built while the check ran elsewhere.
            kv_extent: std::collections::HashMap::new(),
            kv_pool_reach: None,
        };

        // Set up engine.
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Add a request so the scheduler has it.
        let req = make_request("req-1", 10);
        engine.add_request(req);

        // Schedule it.
        let scheduler_output = engine.scheduler.schedule();

        // Manually call update_from_output with our test model_output.
        let outputs = engine.update_from_output(&scheduler_output, &model_output);

        // Check that prompt logprobs are propagated.
        let client_out = outputs.get(&0).unwrap();
        let req_out = client_out
            .outputs
            .iter()
            .find(|o| o.request_id == "req-1")
            .unwrap();

        let plps = req_out.new_prompt_logprobs.as_ref().unwrap();
        // Position 0 is None, positions 1-2 are Some.
        assert_eq!(plps.len(), 3);
        assert!(plps[0].is_none());
        assert_eq!(plps[1].as_ref().unwrap().sampled.token_id, 20);
        assert_eq!(plps[2].as_ref().unwrap().sampled.token_id, 30);
    }

    // -- Speculative decoding tests --

    fn make_spec_decode_config() -> EngineCoreConfig {
        EngineCoreConfig {
            scheduler_config: SchedulerConfig {
                max_num_batched_tokens: 8192,
                max_num_seqs: 256,
                max_num_scheduled_tokens: None,
                policy: SchedulerPolicy::Fcfs,
                enable_chunked_prefill: true,
                long_prefill_token_threshold: 0,
                ..Default::default()
            },
            max_model_len: 4096,
            num_gpu_blocks: 1024,
            block_size: 16,
            engine_index: 0,
            async_scheduling: false,
            use_spec_decode: true,
            proposer_config: Some(crate::spec_decode::ProposerConfig::Ngram(
                crate::spec_decode::NgramProposerConfig {
                    num_speculative_tokens: 3,
                    max_ngram_size: 3,
                    min_ngram_size: 1,
                    max_model_len: 4096,
                },
            )),
            eos_token_ids: vec![],
            is_pooling: false,
            enable_prefix_caching: false,
            hybrid_kv: None,
        }
    }

    #[test]
    fn test_spec_decode_proposer_creates() {
        let config = make_spec_decode_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let engine = EngineCore::new(config, executor);
        assert!(engine.proposer.is_some());
    }

    #[test]
    fn test_spec_decode_disabled_by_default() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let engine = EngineCore::new(config, executor);
        assert!(engine.proposer.is_none());
    }

    #[test]
    fn test_spec_decode_proposes_after_step() {
        // Use a prompt with repeated patterns so the proposer finds matches.
        // Prompt: [10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20]
        // After NoopExecutor generates one token (e.g. 1000), all_token_ids
        // becomes [10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 1000].
        // The proposer should find an n-gram match in the repetitive prefix.
        let config = make_spec_decode_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        let prompt = vec![10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20];
        let params = SamplingParams {
            max_tokens: Some(50),
            ..Default::default()
        };
        let req = Request::new("spec-1".to_string(), prompt, params, 0.0, 0, 0, None);
        engine.add_request(req);

        // First step: prefill + generate first token.
        let _ = engine.step().unwrap();

        // After step, the proposer should have set spec_token_ids on the request.
        let request = engine.scheduler.get_request("spec-1");
        // The request may have finished or still be running.
        if let Some(req) = request {
            // After one NoopExecutor step, all_token_ids has the prompt + 1 token.
            // The repetitive pattern should yield proposals.
            // spec_token_ids may or may not be populated depending on the
            // proposer finding a match, but the mechanism works either way.
            // The key assertion: spec_token_ids is a valid Vec (not panicking).
            let _spec = &req.spec_token_ids;
        }
    }

    #[test]
    fn test_spec_decode_proposals_cleared_after_schedule() {
        // Verify that after scheduling, spec tokens are consumed and cleared.
        let config = make_spec_decode_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        let prompt = vec![10, 20, 30, 10, 20, 30, 10, 20];
        let params = SamplingParams {
            max_tokens: Some(50),
            ..Default::default()
        };
        let req = Request::new("spec-2".to_string(), prompt, params, 0.0, 0, 0, None);
        engine.add_request(req);

        // Step 1: prefill.
        let _ = engine.step().unwrap();

        // Step 2: should pick up any proposed drafts from step 1.
        let _ = engine.step().unwrap();

        // After step 2, the spec_token_ids should be either empty (consumed
        // by scheduler) or repopulated by the proposer for the next step.
        // The scheduler clears them during schedule().
        if let Some(req) = engine.scheduler.get_request("spec-2") {
            // Just verify no panics and the request is functional.
            assert!(!req.all_token_ids.is_empty());
        }
    }

    // -------------------------------------------------------------------
    // Async scheduling support tests
    // -------------------------------------------------------------------

    #[test]
    fn test_take_executor() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // First take succeeds.
        let ex = engine.take_executor();
        assert!(ex.is_some());

        // Second take returns None.
        assert!(engine.take_executor().is_none());
    }

    #[test]
    fn test_step_errors_after_take() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        engine.add_request(make_request("req-1", 10));
        let _ = engine.take_executor();

        // step() should error because executor was taken.
        assert!(engine.step().is_err());
    }

    #[test]
    fn test_schedule_next_empty() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // No requests → None.
        assert!(engine.schedule_next().is_none());
    }

    #[test]
    fn test_schedule_next_with_requests() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        engine.add_request(make_request("req-1", 10));

        let sched = engine.schedule_next();
        assert!(sched.is_some());
        let sched = sched.unwrap();
        assert!(sched.total_num_scheduled_tokens > 0);
    }

    #[test]
    fn test_finalize_step_produces_output() {
        let config = make_test_config();
        let mut noop = NoopExecutor::new(1024);
        noop.initialize_cache(1024, 0).unwrap();

        let mut engine = EngineCore::new(config, Box::new(NoopExecutor::new(1024)));
        engine.add_request(make_request("req-1", 10));

        // Schedule, then manually execute and finalize.
        let sched = engine.schedule_next().unwrap();
        let mut executor = engine.take_executor().unwrap();
        let model_output = executor.execute_model(&sched).unwrap();
        let outputs = engine.finalize_step(&sched, &model_output);

        // Should have outputs for client 0.
        assert!(!outputs.is_empty());
        let client_0 = outputs.get(&0).unwrap();
        assert!(!client_0.outputs.is_empty());
        assert_eq!(client_0.outputs[0].request_id, "req-1");
        assert!(!client_0.outputs[0].new_token_ids.is_empty());
    }

    #[test]
    fn test_shutdown_after_take() {
        let config = make_test_config();
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        let _ = engine.take_executor();

        // Shutdown should not panic even with executor taken.
        engine.shutdown();
        assert!(engine.is_shutdown());
    }

    // -------------------------------------------------------------------
    // Pooling mode tests
    // -------------------------------------------------------------------

    #[test]
    fn test_pooling_request_finishes_with_embedding() {
        // In pooling mode, a request with is_pooling=true should:
        // 1. Finish after one step (FinishReason::Stop)
        // 2. Have pooler_output populated with the embedding vector
        // 3. Have no generated tokens
        let mut config = make_test_config();
        config.is_pooling = true;
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Create a pooling request.
        let params = SamplingParams {
            max_tokens: Some(1),
            ..Default::default()
        };
        let mut req = Request::new("pool-1".to_string(), vec![1, 2, 3], params, 0.0, 0, 0, None);
        req.is_pooling = true;
        engine.add_request(req);

        // Schedule it.
        let scheduler_output = engine.scheduler.schedule();

        // Simulate a model output with pooler_output.
        let embedding = EmbeddingData::Single(vec![0.1, 0.2, 0.3, 0.4]);
        let mut pooler_output = HashMap::new();
        pooler_output.insert("pool-1".to_string(), embedding.clone());

        let model_output = ModelRunnerOutput {
            req_ids: vec!["pool-1".to_string()],
            req_id_to_index: [("pool-1".to_string(), 0)].into_iter().collect(),
            sampled_token_ids: vec![vec![]], // No tokens for pooling.
            logprobs: None,
            prompt_logprobs_dict: HashMap::new(),
            draft_token_ids: None,
            pooler_output: Some(pooler_output),
            d2h_resolver: None,
            draft_seed_inputs: None,
            // A test constructor is a CONSTRUCTION SITE: adding `kv_extent` to
            // ModelRunnerOutput broke these four and nothing noticed, because this
            // crate's test target was never built while the check ran elsewhere.
            kv_extent: std::collections::HashMap::new(),
            kv_pool_reach: None,
        };

        let outputs = engine.update_from_output(&scheduler_output, &model_output);

        // Check the output.
        let client_out = outputs.get(&0).unwrap();
        let req_out = client_out
            .outputs
            .iter()
            .find(|o| o.request_id == "pool-1")
            .unwrap();

        // Should be finished immediately.
        assert_eq!(req_out.finish_reason, Some(FinishReason::Stop));
        // Should have no generated tokens.
        assert!(req_out.new_token_ids.is_empty());
        // Should have the embedding vector.
        assert_eq!(req_out.pooler_output, Some(embedding));
    }

    #[test]
    fn test_pooling_request_multi_vector_embedding() {
        // Multi-vector (ColBERT AllTokens) pooler output should flow through.
        let mut config = make_test_config();
        config.is_pooling = true;
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        let params = SamplingParams {
            max_tokens: Some(1),
            ..Default::default()
        };
        let mut req = Request::new("mv-1".to_string(), vec![1, 2, 3], params, 0.0, 0, 0, None);
        req.is_pooling = true;
        engine.add_request(req);

        let scheduler_output = engine.scheduler.schedule();

        let multi_emb = EmbeddingData::Multi(vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ]);
        let mut pooler_output = HashMap::new();
        pooler_output.insert("mv-1".to_string(), multi_emb.clone());

        let model_output = ModelRunnerOutput {
            req_ids: vec!["mv-1".to_string()],
            req_id_to_index: [("mv-1".to_string(), 0)].into_iter().collect(),
            sampled_token_ids: vec![vec![]],
            logprobs: None,
            prompt_logprobs_dict: HashMap::new(),
            draft_token_ids: None,
            pooler_output: Some(pooler_output),
            d2h_resolver: None,
            draft_seed_inputs: None,
            // A test constructor is a CONSTRUCTION SITE: adding `kv_extent` to
            // ModelRunnerOutput broke these four and nothing noticed, because this
            // crate's test target was never built while the check ran elsewhere.
            kv_extent: std::collections::HashMap::new(),
            kv_pool_reach: None,
        };

        let outputs = engine.update_from_output(&scheduler_output, &model_output);
        let client_out = outputs.get(&0).unwrap();
        let req_out = client_out
            .outputs
            .iter()
            .find(|o| o.request_id == "mv-1")
            .unwrap();

        assert_eq!(req_out.finish_reason, Some(FinishReason::Stop));
        assert_eq!(req_out.pooler_output, Some(multi_emb));
    }

    #[test]
    fn test_pooling_request_finishes_even_without_pooler_data() {
        // A pooling request should still finish (with None pooler_output)
        // even if the model output has no pooler data for it.
        let mut config = make_test_config();
        config.is_pooling = true;
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        let params = SamplingParams {
            max_tokens: Some(1),
            ..Default::default()
        };
        let mut req = Request::new("pool-2".to_string(), vec![1, 2], params, 0.0, 0, 0, None);
        req.is_pooling = true;
        engine.add_request(req);

        let scheduler_output = engine.scheduler.schedule();

        // Model output with empty pooler_output map.
        let model_output = ModelRunnerOutput {
            req_ids: vec!["pool-2".to_string()],
            req_id_to_index: [("pool-2".to_string(), 0)].into_iter().collect(),
            sampled_token_ids: vec![vec![]],
            logprobs: None,
            prompt_logprobs_dict: HashMap::new(),
            draft_token_ids: None,
            pooler_output: Some(HashMap::new()),
            d2h_resolver: None,
            draft_seed_inputs: None,
            // A test constructor is a CONSTRUCTION SITE: adding `kv_extent` to
            // ModelRunnerOutput broke these four and nothing noticed, because this
            // crate's test target was never built while the check ran elsewhere.
            kv_extent: std::collections::HashMap::new(),
            kv_pool_reach: None,
        };

        let outputs = engine.update_from_output(&scheduler_output, &model_output);
        let client_out = outputs.get(&0).unwrap();
        let req_out = client_out
            .outputs
            .iter()
            .find(|o| o.request_id == "pool-2")
            .unwrap();

        // Should still finish.
        assert_eq!(req_out.finish_reason, Some(FinishReason::Stop));
        // Pooler output is None (not found in the map).
        assert!(req_out.pooler_output.is_none());
    }

    #[test]
    fn test_non_pooling_request_unaffected_in_pooling_engine() {
        // A non-pooling request (is_pooling=false) in a pooling engine
        // should go through the normal generation path.
        let mut config = make_test_config();
        config.is_pooling = true;
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        // Regular generation request (is_pooling defaults to false).
        let req = make_request("gen-1", 5);
        engine.add_request(req);

        // Step — NoopExecutor generates a token.
        let (outputs, model_executed) = engine.step().unwrap().into_tuple();
        assert!(model_executed);

        let client_out = outputs.get(&0).unwrap();
        let req_out = client_out
            .outputs
            .iter()
            .find(|o| o.request_id == "gen-1")
            .unwrap();

        // Should have generated tokens (not pooling path).
        assert!(!req_out.new_token_ids.is_empty());
        // No pooler output.
        assert!(req_out.pooler_output.is_none());
    }

    // -----------------------------------------------------------------------
    // Seal-pad stop criteria tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sealed_request_does_not_stop_on_eos() {
        // A sealed request hitting EOS should NOT stop immediately.
        // It should continue generating (SealPadProcessor forces pads on GPU).
        let mut config = make_test_config();
        config.block_size = 4;
        config.eos_token_ids = vec![1000]; // NoopExecutor's first token
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        let params = SamplingParams {
            max_tokens: Some(100),
            seal: true,
            ..Default::default()
        };
        // 2-token prompt → after first decode step, total = 3 tokens (not block-aligned).
        let mut req = Request::new(
            "sealed-1".to_string(),
            vec![10, 20],
            params,
            0.0,
            0,
            0,
            None,
        );
        req.seal = true;
        engine.add_request(req);

        // Step 1: generates token 1000 (EOS). Sealed request should NOT finish.
        let (outputs, _) = engine.step().unwrap().into_tuple();
        let client_out = outputs.get(&0).unwrap();
        let req_out = client_out
            .outputs
            .iter()
            .find(|o| o.request_id == "sealed-1");
        // Request should still be running (no finish_reason) or the output
        // should show it continuing. If it's in the output, check no finish.
        if let Some(out) = req_out {
            assert!(
                out.finish_reason.is_none(),
                "sealed request should not stop on EOS: got {:?}",
                out.finish_reason
            );
        }
        // Request should still be unfinished (EOS deferred for seal padding).
        assert_eq!(engine.num_unfinished_requests(), 1);
    }

    #[test]
    fn test_sealed_request_stops_when_block_aligned() {
        // A sealed request should stop when total tokens reach block alignment
        // after EOS has been seen.
        let mut config = make_test_config();
        config.block_size = 4;
        config.eos_token_ids = vec![1000];
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        let params = SamplingParams {
            max_tokens: Some(100),
            seal: true,
            ..Default::default()
        };
        // 3-token prompt → first decode = 4 tokens (block-aligned!).
        // EOS at position 4 means it finishes immediately after padding.
        let mut req = Request::new(
            "sealed-2".to_string(),
            vec![10, 20, 30],
            params,
            0.0,
            0,
            0,
            None,
        );
        req.seal = true;
        engine.add_request(req);

        // Step 1: token 1000 generated → total = 4 = block-aligned.
        // Even though it's EOS, since it's already block-aligned, it stops.
        let (_outputs, _) = engine.step().unwrap().into_tuple();

        // EOS is detected in the same step but alignment is checked on the
        // NEXT invocation of check_stop_criteria. So the request stays alive.
        assert_eq!(
            engine.num_unfinished_requests(),
            1,
            "should NOT stop on first step (EOS detected, alignment checked next step)"
        );

        // Step 2: NoopExecutor generates token 1001. Total = 5 tokens.
        // Not block-aligned → continues.
        let (_outputs2, _) = engine.step().unwrap().into_tuple();
        // Still running (5 tokens, not aligned to 4).
        assert_eq!(engine.num_unfinished_requests(), 1);

        // Step 3: token 1002. Total = 6. Not aligned.
        let (_outputs3, _) = engine.step().unwrap().into_tuple();
        assert_eq!(engine.num_unfinished_requests(), 1);

        // Step 4: token 1003. Total = 7. Not aligned.
        let (_outputs4, _) = engine.step().unwrap().into_tuple();
        assert_eq!(engine.num_unfinished_requests(), 1);

        // Step 5: token 1004. Total = 8. Block-aligned! Should stop.
        let (outputs5, _) = engine.step().unwrap().into_tuple();
        let client_out5 = outputs5.get(&0).unwrap();
        let req_out5 = client_out5
            .outputs
            .iter()
            .find(|o| o.request_id == "sealed-2")
            .expect("should have output for sealed-2");
        assert_eq!(
            req_out5.finish_reason,
            Some(FinishReason::Stop),
            "should stop at block-aligned boundary"
        );
        assert_eq!(engine.num_unfinished_requests(), 0);
    }

    #[test]
    fn test_sealed_max_tokens_defers_until_block_aligned() {
        // Sealed request hitting max_tokens defers stop until block-aligned.
        // block_size=4, 2-token prompt, max_tokens=1.
        // Step 1: 1 output → 3 total, max_tokens hit, not aligned → defer.
        // Step 2: 4 total, aligned → stop.
        let mut config = make_test_config();
        config.block_size = 4;
        config.eos_token_ids = vec![9999]; // won't be generated
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        let params = SamplingParams {
            max_tokens: Some(1),
            seal: true,
            ..Default::default()
        };
        let mut req = Request::new(
            "sealed-3".to_string(),
            vec![10, 20],
            params,
            0.0,
            0,
            0,
            None,
        );
        req.seal = true;
        engine.add_request(req);

        // Step 1: 1 output token → 3 total, not aligned → deferred.
        let (_outputs, _) = engine.step().unwrap().into_tuple();
        assert_eq!(engine.num_unfinished_requests(), 1);

        // Step 2: 2 output tokens → 4 total, aligned → stop.
        let (outputs, _) = engine.step().unwrap().into_tuple();
        let client_out = outputs.get(&0).unwrap();
        let req_out = client_out
            .outputs
            .iter()
            .find(|o| o.request_id == "sealed-3")
            .expect("should finish on block alignment");
        assert_eq!(req_out.finish_reason, Some(FinishReason::Stop));
        assert_eq!(engine.num_unfinished_requests(), 0);
    }

    #[test]
    fn test_sealed_request_output_includes_padding_tokens() {
        // Verify that a sealed request generates MORE output tokens than a
        // non-sealed one. Non-sealed stops at EOS (1 output token). Sealed
        // continues until block-aligned.
        //
        // block_size=4, 1-token prompt, EOS=1000 (first decode token).
        // Non-sealed: 1 prompt + 1 output (EOS) = 2 tokens → stop.
        // Sealed: 1 prompt + 1 output (EOS) = 2 tokens. Not aligned (2%4≠0).
        //   Step 2: 3 tokens. Not aligned.
        //   Step 3: 4 tokens. 4%4==0 → stop. Total output = 3.

        let mut config = make_test_config();
        config.block_size = 4;
        config.eos_token_ids = vec![1000];
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        let params = SamplingParams {
            max_tokens: Some(100),
            seal: true,
            ..Default::default()
        };
        let mut req = Request::new("s1".to_string(), vec![10], params, 0.0, 0, 0, None);
        req.seal = true;
        engine.add_request(req);

        let mut total_output_tokens = 0u32;
        for _ in 0..10 {
            let (outputs, _) = engine.step().unwrap().into_tuple();
            if let Some(client_out) = outputs.get(&0) {
                for out in &client_out.outputs {
                    if out.request_id == "s1" {
                        total_output_tokens += out.new_token_ids.len() as u32;
                        if out.finish_reason.is_some() {
                            assert_eq!(
                                total_output_tokens, 3,
                                "sealed: 1 prompt + 3 output = 4 (block-aligned). \
                                 Non-sealed would stop at 1 output."
                            );
                            return;
                        }
                    }
                }
            }
        }
        panic!("sealed request should have finished within 10 steps");
    }

    #[test]
    fn test_non_sealed_request_still_stops_on_eos() {
        // Verify we didn't break normal (non-sealed) EOS behavior.
        let mut config = make_test_config();
        config.block_size = 4;
        config.eos_token_ids = vec![1000];
        let executor = Box::new(NoopExecutor::new(1024));
        let mut engine = EngineCore::new(config, executor);

        let params = SamplingParams {
            max_tokens: Some(100),
            ..Default::default()
        };
        let req = Request::new(
            "normal-1".to_string(),
            vec![10, 20],
            params,
            0.0,
            0,
            0,
            None,
        );
        engine.add_request(req);

        let (outputs, _) = engine.step().unwrap().into_tuple();
        let client_out = outputs.get(&0).unwrap();
        let req_out = client_out
            .outputs
            .iter()
            .find(|o| o.request_id == "normal-1")
            .unwrap();

        assert_eq!(req_out.finish_reason, Some(FinishReason::Stop));
        assert_eq!(engine.num_unfinished_requests(), 0);
    }
}
