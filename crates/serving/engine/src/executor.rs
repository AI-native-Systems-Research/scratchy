// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Executor trait defining the interface between the engine core and GPU
//! workers.
//!
//! The executor is responsible for:
//! - Managing worker processes
//! - Executing model forward passes
//! - Managing GPU memory and KV cache initialization
//!
//! Port of: `vllm/v1/executor/abstract.py`

use std::collections::HashMap;

use scratchy_core_common::engine_io::EmbeddingData;

use scratchy_core_common::LogprobsOutput;
#[cfg(test)]
use scratchy_core_common::TokenLogprob;
use scratchy_serving_scheduler::scheduler::output::SchedulerOutput;

use crate::error::EngineResult;

// ---------------------------------------------------------------------------
// ModelRunnerOutput
// ---------------------------------------------------------------------------

/// ⭐ #32 (C) — ONE DEFERRED STEP'S ANSWER CELL: filled by the drain that owns
/// the step ([`crate::core_client::DrainDoor`]'s knock → the executor thread's
/// `drain_knock` pump), waited on by the resolver the deferred output carries.
///
/// `fill` is ONE-WAY and single-shot — the double fill is an assert, because
/// two drains taking one step is a queue-law violation, not a load-bearing
/// race to paper over. `wait` parks until the fill; the resolver that knocks
/// and then waits is the deadlock fix's own order (a resolver that WAITED
/// first would block the one thread that can fill the cell).
///
/// ⭐ THE ANSWER IS EVERY ROW'S, NOT THE FIRST READER'S. One cell serves a
/// whole step's owed rows — a wave of k tails, a decode batch of k requests —
/// and the knock loop's `is_filled` polls the same fill, so the read must not
/// consume: the answer is held behind an [`Arc`] and `wait` hands out shares.
/// A consuming wait made the second row of a multi-row piece park on an
/// emptied cell forever (serve c4: the wave's row 0 took the answer, row 1
/// hung), and made a consumed cell read unfilled to the knock loop, which
/// then drained an empty queue and errored the parked row ("the in-flight
/// queue emptied before this step landed"). One answer, many readers — the
/// type says so; a consuming read is unrepresentable.
#[derive(Clone)]
pub struct DrainCell(std::sync::Arc<DrainCellInner>);

struct DrainCellInner {
    filled: std::sync::Mutex<Option<std::sync::Arc<EngineResult<Vec<u32>>>>>,
    cv: std::sync::Condvar,
}

impl DrainCell {
    pub fn new() -> Self {
        Self(std::sync::Arc::new(DrainCellInner {
            filled: std::sync::Mutex::new(None),
            cv: std::sync::Condvar::new(),
        }))
    }

    /// Fill the cell — once. The double fill panics by design: the drain's
    /// pop is the single authority over which drain owns a step, so a second
    /// fill means two drains took one step and one of them lied.
    pub fn fill(&self, tokens: EngineResult<Vec<u32>>) {
        let mut g = self.0.filled.lock().unwrap();
        assert!(
            g.is_none(),
            "a deferred step's DrainCell was filled twice — two drains took one step"
        );
        *g = Some(std::sync::Arc::new(tokens));
        self.0.cv.notify_all();
    }

    /// Has this cell been filled? The knock loop's own ask — it drains only
    /// while the step it was knocked for is unanswered. A cell stays filled
    /// through every read: the answer is the rows' to share, not the first
    /// reader's to take.
    pub fn is_filled(&self) -> bool {
        self.0.filled.lock().unwrap().is_some()
    }

    /// Park until the fill, then a share of the answer. Every row of the
    /// step waits this one cell, so the share — not the answer itself — is
    /// what a reader takes; the latch stays set for the rows behind it and
    /// for the knock loop's poll. A cell that was filled with an error hands
    /// that error to every row — the resolver surfaces it as the deferred
    /// output's own failure, never as a hang.
    pub fn wait(&self) -> std::sync::Arc<EngineResult<Vec<u32>>> {
        let mut g = self.0.filled.lock().unwrap();
        loop {
            if let Some(t) = g.clone() {
                return t;
            }
            g = self.0.cv.wait(g).unwrap();
        }
    }
}

impl Default for DrainCell {
    fn default() -> Self {
        Self::new()
    }
}

/// Output from one model execution step.
///
/// This is the Rust equivalent of `vllm/v1/outputs.py::ModelRunnerOutput`.
/// Unlike the Python version which uses parallel arrays optimized for GPU
/// tensor operations, the Rust version uses per-request maps for clarity.
///
/// Fields related to torch tensors (pooler_output, etc.) are represented
/// as opaque byte buffers or omitted until Phase 5 (tensor infrastructure).
pub struct ModelRunnerOutput {
    /// Ordered list of request IDs that were processed.
    pub req_ids: Vec<String>,

    /// Request ID to index mapping (for fast lookup).
    pub req_id_to_index: HashMap<String, usize>,

    /// Per-request generated token IDs.
    ///
    /// Each request may produce multiple tokens per step (e.g., speculative
    /// decoding). Inner vec has one entry per generated token.
    pub sampled_token_ids: Vec<Vec<u32>>,

    /// Per-request log-probabilities (if requested).
    ///
    /// Outer vec is indexed by request (same order as `req_ids`).
    /// Inner vec has one entry per generated token position.
    /// `None` if no request in this batch requested logprobs.
    pub logprobs: Option<Vec<Option<Vec<LogprobsOutput>>>>,

    /// Prompt log-probabilities for requests that requested them.
    ///
    /// Maps request ID to per-position logprobs for the prompt tokens.
    pub prompt_logprobs_dict: HashMap<String, Vec<LogprobsOutput>>,

    /// Draft token IDs for speculative decoding.
    ///
    /// Maps request ID to draft token sequences.
    pub draft_token_ids: Option<HashMap<String, Vec<u32>>>,

    /// Owned-data seed bundle for `DraftModelProposer`'s K-step chain.
    /// `Some` when the worker ran a target verify with a draft model
    /// loaded; carries the lockstep-prefill inputs + per-req attn state
    /// the chain needs. The proposer runs in `EngineCore::finalize_step`
    /// against this bundle via `spec_decode_backend()`.
    pub draft_seed_inputs: Option<crate::spec_decode::DraftSeedInputs>,

    /// Pooling output (embedding data) for requests in pooling mode.
    ///
    /// Maps request ID to the embedding data (single or multi-vector).
    /// `None` when the engine is not in pooling mode.
    pub pooler_output: Option<HashMap<String, EmbeddingData>>,

    /// ⭐ WHAT THE WORKER'S KV ACTUALLY LOOKS LIKE, per request — the slots its blocks must cover and
    /// the leading tokens a token-indexed prefix cache may claim ([`scratchy_core_common::KvExtent`]).
    ///
    /// Empty for every backend whose KV sits at its token positions: there the span IS the
    /// token count and the whole computed prefix is cacheable, so the scheduler's own arithmetic is
    /// already right and reporting would be a second source for it.
    ///
    /// Non-empty for a backend whose batched write appends every row of a batch at ONE slot per step: a
    /// request shorter than its batch-mates then spreads its keys over more slots than it has tokens, and
    /// past that hole token `t` is no longer at slot `t` — so the scheduler must allocate by the span and
    /// must not hash a block over anything past the hole. Both numbers are facts about the history the
    /// worker just wrote, which is why they come back from the worker rather than being re-derived from
    /// the batch composition the scheduler chose.
    pub kv_extent: HashMap<String, scratchy_core_common::KvExtent>,
    /// THE POOL-WIDE REACH: the deepest slot ANY live request will occupy on its own account, which is the
    /// slot a batched step's shared write lands at or before.
    ///
    /// ⛔ SEPARATE FROM `kv_extent` BECAUSE IT IS ABOUT REQUESTS THAT HAVE NO ENTRY THERE. `kv_extent` is
    /// keyed by request id, so it can only size a request the worker has already run — and a request being
    /// ADMITTED is sized by its token count alone. If it then joins a batch whose shared write slot is
    /// already deep, its write page is past every block the host gave it, and the worker must serve that
    /// page from a reserve of its own. This number is what lets the HOST own that page instead.
    ///
    /// `None` for every backend whose KV is token-addressed: there the write slot IS the token
    /// count and a new request's own allocation already covers it.
    pub kv_pool_reach: Option<scratchy_core_common::KvSlotSpan>,
    /// HOW MANY BLOCKS OF EACH KV STORE THE WORKER HAS BACKED WITH MEMORY after this step. `Some` only from
    /// a worker that declared [`scratchy_core_common::KvBacking::OnFirstUse`], for the allocator to
    /// spend the prefix cache's free blocks before a block that would map more.
    pub kv_backed: Option<scratchy_core_common::KvBackedBlocks>,
    /// ⛔ THE SNAPSHOT SLOTS THE WORKER COULD NOT READ BACK FROM DISK this step (`SnapshotDiskOps`
    /// fills). No request ran on one: the request a fill is for waits for its step. The scheduler
    /// forgets each, so that request, and the next hit on its prefix, resume cold
    /// ([`Scheduler::forget_failed_fills`]).
    ///
    /// [`Scheduler::forget_failed_fills`]: scratchy_serving_scheduler::scheduler::Scheduler::forget_failed_fills
    pub failed_fills: Vec<usize>,

    /// Deferred D2H resolver. When present, `sampled_token_ids` contains
    /// placeholders. Call `resolve()` to synchronize the D2H transfer and
    /// populate the real token IDs — or learn the step failed on the device.
    ///
    /// Matches Python's `AsyncOutput` pattern: the GPU enqueues a D2H copy
    /// on a transfer stream and returns immediately. The closure waits for
    /// the device and reads the sampled tokens back.
    pub d2h_resolver: Option<D2hResolve>,
}

/// A deferred step's tokens, one per request in `req_ids` order, once its device work is done; `Err`
/// if that work failed.
pub type D2hResolver = Box<dyn FnOnce() -> Result<Vec<u32>, String> + Send>;

/// A deferred speculative step's outputs once its device work is done: each request's tokens, in
/// `req_ids` order — the drafts it kept and the target's next — and the drafts it made for its
/// next step; `Err` if that work failed.
pub type SpecResolver =
    Box<dyn FnOnce() -> Result<(Vec<Vec<u32>>, HashMap<String, Vec<u32>>), String> + Send>;

/// How a deferred output resolves.
pub enum D2hResolve {
    Tokens(D2hResolver),
    Speculative(SpecResolver),
}

impl std::fmt::Debug for ModelRunnerOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelRunnerOutput")
            .field("req_ids", &self.req_ids)
            .field("sampled_token_ids", &self.sampled_token_ids)
            .field("is_deferred", &self.d2h_resolver.is_some())
            .finish()
    }
}

impl ModelRunnerOutput {
    /// Create an empty ModelRunnerOutput.
    pub fn empty() -> Self {
        Self {
            req_ids: Vec::new(),
            req_id_to_index: HashMap::new(),
            sampled_token_ids: Vec::new(),
            logprobs: None,
            prompt_logprobs_dict: HashMap::new(),
            draft_token_ids: None,
            draft_seed_inputs: None,
            pooler_output: None,
            kv_extent: HashMap::new(),
            kv_pool_reach: None,
            kv_backed: None,
            failed_fills: Vec::new(),
            d2h_resolver: None,
        }
    }

    /// Whether this output has a deferred D2H transfer that needs resolving.
    pub fn is_deferred(&self) -> bool {
        self.d2h_resolver.is_some()
    }

    /// Resolve the deferred D2H transfer, populating `sampled_token_ids`.
    ///
    /// Waits for the device and reads the sampled token IDs back. No-op if the output is already resolved. `Err`: the step
    /// failed on the device, and its tokens are not to be used.
    pub fn resolve(&mut self) -> Result<(), String> {
        match self.d2h_resolver.take() {
            None => {}
            Some(D2hResolve::Tokens(resolver)) => {
                let token_ids = resolver()?;
                self.sampled_token_ids = token_ids.into_iter().map(|t| vec![t]).collect();
            }
            Some(D2hResolve::Speculative(resolver)) => {
                let (tokens, drafts) = resolver()?;
                self.sampled_token_ids = tokens;
                self.draft_token_ids = Some(drafts);
            }
        }
        Ok(())
    }

    /// Get the sampled token IDs for a specific request.
    pub fn get_tokens(&self, req_id: &str) -> Option<&[u32]> {
        self.req_id_to_index
            .get(req_id)
            .and_then(|&idx| self.sampled_token_ids.get(idx))
            .map(|v| v.as_slice())
    }

    /// Number of requests in this output.
    pub fn num_requests(&self) -> usize {
        self.req_ids.len()
    }

    /// Whether this output is empty (no requests processed).
    pub fn is_empty(&self) -> bool {
        self.req_ids.is_empty()
    }

    /// Helper: build from a simple request-to-tokens map.
    ///
    /// Useful for testing and for simple executor implementations.
    pub fn from_token_map(token_map: HashMap<String, Vec<u32>>) -> Self {
        let mut req_ids = Vec::with_capacity(token_map.len());
        let mut req_id_to_index = HashMap::with_capacity(token_map.len());
        let mut sampled_token_ids = Vec::with_capacity(token_map.len());
        for (id, tokens) in token_map {
            let idx = req_ids.len();
            req_id_to_index.insert(id.clone(), idx);
            req_ids.push(id);
            sampled_token_ids.push(tokens);
        }
        Self {
            req_ids,
            req_id_to_index,
            sampled_token_ids,
            logprobs: None,
            prompt_logprobs_dict: HashMap::new(),
            draft_token_ids: None,
            draft_seed_inputs: None,
            pooler_output: None,
            kv_extent: HashMap::new(),
            kv_pool_reach: None,
            kv_backed: None,
            failed_fills: Vec::new(),
            d2h_resolver: None,
        }
    }

    /// Build from pre-ordered req_ids and per-request token IDs.
    ///
    /// More efficient than `from_token_map` — avoids the intermediate HashMap
    /// and its String clones. `req_ids` and `token_ids` must have the same length
    /// and be in the same order.
    pub fn from_ordered(req_ids: Vec<String>, token_ids: Vec<u32>) -> Self {
        debug_assert_eq!(req_ids.len(), token_ids.len());
        let mut req_id_to_index = HashMap::with_capacity(req_ids.len());
        let mut sampled_token_ids = Vec::with_capacity(req_ids.len());
        for (idx, id) in req_ids.iter().enumerate() {
            req_id_to_index.insert(id.clone(), idx);
            sampled_token_ids.push(vec![token_ids[idx]]);
        }
        Self {
            req_ids,
            req_id_to_index,
            sampled_token_ids,
            logprobs: None,
            prompt_logprobs_dict: HashMap::new(),
            draft_token_ids: None,
            draft_seed_inputs: None,
            pooler_output: None,
            kv_extent: HashMap::new(),
            kv_pool_reach: None,
            kv_backed: None,
            failed_fills: Vec::new(),
            d2h_resolver: None,
        }
    }

    /// Build a deferred output whose token IDs will be resolved lazily.
    ///
    /// The `resolver` closure is called by `resolve()` to synchronize the
    /// device and read the sampled token IDs back.
    /// Until resolved, `sampled_token_ids` is empty.
    pub fn deferred(req_ids: Vec<String>, resolver: D2hResolver) -> Self {
        let req_id_to_index: HashMap<String, usize> = req_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();
        Self {
            req_ids,
            req_id_to_index,
            sampled_token_ids: Vec::new(),
            logprobs: None,
            prompt_logprobs_dict: HashMap::new(),
            draft_token_ids: None,
            draft_seed_inputs: None,
            pooler_output: None,
            kv_extent: HashMap::new(),
            kv_pool_reach: None,
            kv_backed: None,
            failed_fills: Vec::new(),
            d2h_resolver: Some(D2hResolve::Tokens(resolver)),
        }
    }

    /// A speculative step committed without waiting for it: its tokens and drafts once `resolver`
    /// runs ([`Self::resolve`]).
    pub fn deferred_speculative(req_ids: Vec<String>, resolver: SpecResolver) -> Self {
        Self {
            d2h_resolver: Some(D2hResolve::Speculative(resolver)),
            ..Self::deferred(req_ids, Box::new(|| Ok(Vec::new())))
        }
    }
}

// ---------------------------------------------------------------------------
// Executor trait
// ---------------------------------------------------------------------------

/// Trait defining the executor interface.
///
/// The executor manages worker processes and dispatches model execution
/// to them. The engine core calls into the executor to run model forward
/// passes and collect outputs.
///
/// Port of: `vllm/v1/executor/abstract.py::Executor`
pub trait Executor: Send {
    /// Execute the model with the given scheduler output.
    ///
    /// Returns the model runner output containing generated tokens.
    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> EngineResult<ModelRunnerOutput>;

    /// Get the maximum number of concurrent batches.
    ///
    /// Returns > 1 for pipeline parallelism or async scheduling.
    fn max_concurrent_batches(&self) -> usize {
        1
    }

    /// Whether the executor runs a speculative step scheduled behind one still in flight
    /// ([`crate::worker::Worker::pipelines_speculative_steps`]).
    fn pipelines_speculative_steps(&self) -> bool {
        false
    }

    /// Initialize KV cache on workers with the given configuration.
    fn initialize_cache(
        &mut self,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    ) -> EngineResult<()>;

    /// Determine available GPU memory for KV cache (in bytes).
    ///
    /// Returns one value per worker.
    fn determine_available_memory(&mut self) -> EngineResult<Vec<usize>>;

    /// ⭐⭐⭐ HOW THE WORKER ADDRESSES ITS KV — asked ONCE, when the engine builds the block tracker.
    ///
    /// Forwarded from [`crate::worker::Worker::kv_addressing`]. `ByToken` for token-addressed KV; a paged
    /// pool may share one write slot per batched step, and the scheduler must know that BEFORE it allocates for
    /// the first step. Asked of the executor rather than passed through `EngineCoreConfig` because the
    /// worker is the thing that knows, and a config field is one more place for the answer to be wrong.
    fn kv_addressing(&self) -> scratchy_core_common::KvAddressing {
        scratchy_core_common::KvAddressing::ByToken
    }

    /// Forwarded from [`crate::worker::Worker::hybrid_kv_stores`], asked when the engine builds the block
    /// tracker, beside [`Self::kv_addressing`].
    fn hybrid_kv_stores(&self) -> scratchy_core_common::HybridKvStores {
        scratchy_core_common::HybridKvStores::Shared
    }

    /// Forwarded from [`crate::worker::Worker::kv_backing`], asked when the engine builds the block
    /// tracker, beside [`Self::hybrid_kv_stores`].
    fn kv_backing(&self) -> scratchy_core_common::KvBacking {
        scratchy_core_common::KvBacking::Allocated
    }

    /// ⭐ HOW THE WORKER'S RECURRENT-STATE SNAPSHOTS SHARE THE KV BUDGET — asked once, when the
    /// engine builds the block tracker, for the same reason as [`Self::kv_addressing`].
    ///
    /// Forwarded from [`crate::worker::Worker::snapshot_budget`]. `None` (the default) for every
    /// executor that does not forward it: the tracker then tracks no recurrent state, and
    /// `initialize_core` keeps prefix caching off for a recurrent-hybrid model.
    fn snapshot_budget(&self) -> Option<scratchy_serving_scheduler::scheduler::SnapshotBudget> {
        None
    }

    /// Hand the engine's stop tokens to the worker that samples
    /// ([`crate::worker::Worker::set_stop_token_ids`]): the engine calls it once, as it is built.
    /// `EngineCore` stops a request on the same ids, so the two can't disagree. No default: an
    /// executor that kept the ids from its workers would leave their grammars ending on a guess.
    fn set_stop_token_ids(&mut self, stop_token_ids: &[u32]);

    /// Check whether the executor is healthy.
    ///
    /// Returns `Ok(())` if healthy, or an error describing the problem.
    fn check_health(&self) -> EngineResult<()> {
        Ok(())
    }

    /// Put the executor to sleep to free GPU memory.
    ///
    /// `level` controls how aggressively resources are freed.
    fn sleep(&mut self, _level: u32) -> EngineResult<()> {
        Ok(())
    }

    /// Wake the executor from sleep.
    ///
    /// If `tags` is `None`, wake all resources. Otherwise, only wake
    /// the specified resource tags (e.g., "weights", "kv_cache").
    fn wake_up(&mut self, _tags: Option<&[String]>) -> EngineResult<()> {
        Ok(())
    }

    /// ⭐ #32 (C) — HAND THE WORKER THE DRAIN DOOR. The pipelined client mints
    /// the door at `start_pipeline` (a sender clone onto its executor thread)
    /// and hands it down through here, so a deferred output's resolver can
    /// KNOCK — ask the thread that owns the queue to drain until the step it
    /// is waiting on lands. Default no-op: an executor whose worker never
    /// defers has nothing to knock on.
    fn set_drain_door(&mut self, _door: crate::core_client::DrainDoor) {}

    /// ⭐ RUN THE CALLING THREAD WHERE THIS EXECUTOR'S DEVICES WANT THE THREAD THAT DRIVES THEM:
    /// the first act of the thread that runs the decode loop (the pipelined client's executor
    /// thread, `scr serve`'s). A multi-card worker answers with its cards' NUMA node. Default: nothing
    /// to place.
    fn pin_driving_thread(&self) -> EngineResult<()> {
        Ok(())
    }

    /// ⭐ #32 (C) — A KNOCK'S NON-BLOCKING HALF: drain every step whose landing
    /// has already fired, in submission order, stopping at the first that has
    /// not, and say whether `cell` is filled. The knock's caller runs this in
    /// [`drain_knock`]'s pump — feeding the executor every message that arrives
    /// while the knock waits — so a queued Step's submit is never made to wait
    /// behind the collect of a step already on the card. The pop inside the
    /// worker's drain stays the single authority, so the double-fill assert on
    /// [`DrainCell`] is the loud boundary if two drains ever take one step.
    /// Default: nothing to drain — an executor whose worker never defers has no
    /// knock to answer.
    fn drain_ready(&mut self, _cell: &std::sync::Arc<DrainCell>) -> bool {
        true
    }

    /// Whether the executor is currently sleeping.
    fn is_sleeping(&self) -> bool {
        false
    }

    /// Compute embeddings for the given token ID sequences.
    ///
    /// Returns one embedding vector per input sequence.
    /// Default implementation returns an error.
    fn embed(&mut self, _token_id_seqs: Vec<Vec<u32>>) -> EngineResult<Vec<Vec<f32>>> {
        Err(crate::error::EngineError::Executor(
            "embedding not supported".into(),
        ))
    }

    /// Mutable access to the spec-decode backend, if this executor
    /// exposes one. Used by `EngineCore::finalize_step` to thread a
    /// `&mut dyn SpecDecodeBackend` into `ProposerStepCtx::backend` so
    /// `DraftModelProposer` can run the lockstep prefill + K-step
    /// chain via `forward_argmax_blocking` against the executor's
    /// driver worker.
    ///
    /// Default returns `None` — only single-worker executors that own
    /// a worker (UniProcExecutor, ThreadPoolExecutor's driver
    /// shard) need to override.
    fn spec_decode_backend(&mut self) -> Option<&mut dyn crate::spec_decode::SpecDecodeBackend> {
        None
    }

    /// Shut down the executor and all workers.
    fn shutdown(&mut self);
}

/// ⭐ WHAT THE KNOCK'S PUMP LEARNED FROM THE CHANNEL: a message ran (keep
/// feeding), another knock arrived (it joins the pending set), nothing is
/// waiting (harvest what has landed), or the loop must stop — the teardown
/// arrived, or the channel died, and the caller honors it through its own.
pub enum Pumped {
    /// A message ran — a Step fed the executor, or a side channel answered.
    Ran,
    /// Another knock arrived: the pump queues it with the rest.
    Knock(std::sync::Arc<DrainCell>),
    /// Nothing is waiting on the channel.
    Idle,
    /// The teardown arrived mid-pump.
    Stop,
}

/// ⭐ FEED BEFORE HARVEST — the knock's drain loop, the one statement of the law.
///
/// A knock asks the executor to drain until its cell lands; a Step queued behind
/// it asks to be submitted. The drain's park only ever waits on the device, and
/// the device can only complete MORE work if the queued Step is submitted first
/// — so while a knocked cell is unfilled, every message that arrives RUNS (a
/// Step feeds the card; a later knock joins the pending set; the teardown stops
/// the pump), only LANDED steps are harvested ([`Executor::drain_ready`], in
/// submission order, never parking), and the loop sleeps a slice between
/// attempts instead of parking the executor's one thread. FIFO is otherwise both
/// callers' law; this is its one permitted reordering, safe because a Step's own
/// churn drain fills cells through the same pop the knock takes — the pop in the
/// worker's drain is the single authority over which drain owns a step.
///
/// Returns `true` when every knocked cell is filled, `false` when the teardown
/// arrived (the caller breaks to its own shutdown tail).
pub fn drain_knock(
    executor: &mut dyn Executor,
    knocks: &mut Vec<std::sync::Arc<DrainCell>>,
    mut pump: impl FnMut(&mut dyn Executor) -> Pumped,
) -> bool {
    loop {
        // FEED: everything that arrived runs before anything is harvested.
        match pump(executor) {
            Pumped::Ran => continue,
            Pumped::Knock(cell) => {
                knocks.push(cell);
                continue;
            }
            Pumped::Stop => return false,
            Pumped::Idle => {}
        }
        // HARVEST: the oldest unfilled knock, and only what has landed.
        knocks.retain(|c| !c.is_filled());
        let Some(oldest) = knocks.first().cloned() else {
            return true;
        };
        if executor.drain_ready(&oldest) {
            continue;
        }
        // PARK A SLICE: nothing landed, nothing queued — the device is still
        // running the knocked step. The next round feeds the channel first: the
        // Step that arrives is what keeps the device busy past this landing.
        std::thread::sleep(std::time::Duration::from_micros(250));
    }
}

// ---------------------------------------------------------------------------
// NoopExecutor -- a minimal executor for testing
// ---------------------------------------------------------------------------

/// A no-op executor that generates dummy tokens.
///
/// Used for testing the engine core logic without requiring actual GPU
/// workers.
pub struct NoopExecutor {
    /// Number of GPU blocks to report.
    num_gpu_blocks: usize,
    /// Next token ID to generate.
    next_token_id: u32,
    /// Whether the executor has been shut down.
    is_shutdown: bool,
    /// Reports [`Executor::pipelines_speculative_steps`].
    pipelines: bool,
}

impl NoopExecutor {
    /// Create a new no-op executor.
    pub fn new(num_gpu_blocks: usize) -> Self {
        Self {
            num_gpu_blocks,
            next_token_id: 1000,
            is_shutdown: false,
            pipelines: false,
        }
    }

    /// One that reports running a speculative step behind one still in flight.
    pub fn with_pipelining(self) -> Self {
        Self {
            pipelines: true,
            ..self
        }
    }
}

impl Executor for NoopExecutor {
    /// It samples no grammar.
    fn set_stop_token_ids(&mut self, _stop_token_ids: &[u32]) {}

    fn pipelines_speculative_steps(&self) -> bool {
        self.pipelines
    }

    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> EngineResult<ModelRunnerOutput> {
        let n = scheduler_output.num_scheduled_tokens.len();
        let mut req_ids = Vec::with_capacity(n);
        let mut req_id_to_index = HashMap::with_capacity(n);
        let mut sampled_token_ids = Vec::with_capacity(n);
        for (i, req_id) in scheduler_output.num_scheduled_tokens.keys().enumerate() {
            req_id_to_index.insert(req_id.clone(), i);
            req_ids.push(req_id.clone());
            sampled_token_ids.push(vec![self.next_token_id]);
            self.next_token_id += 1;
        }
        Ok(ModelRunnerOutput {
            req_ids,
            req_id_to_index,
            sampled_token_ids,
            logprobs: None,
            prompt_logprobs_dict: HashMap::new(),
            draft_token_ids: None,
            draft_seed_inputs: None,
            pooler_output: None,
            kv_extent: HashMap::new(),
            kv_pool_reach: None,
            kv_backed: None,
            failed_fills: Vec::new(),
            d2h_resolver: None,
        })
    }

    fn initialize_cache(
        &mut self,
        _num_gpu_blocks: usize,
        _num_cpu_blocks: usize,
    ) -> EngineResult<()> {
        Ok(())
    }

    fn determine_available_memory(&mut self) -> EngineResult<Vec<usize>> {
        Ok(vec![self.num_gpu_blocks * 16 * 1024]) // fake bytes
    }

    fn shutdown(&mut self) {
        self.is_shutdown = true;
    }

    fn is_sleeping(&self) -> bool {
        self.is_shutdown
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drain_cell_answer_is_every_rows() {
        // One cell serves a whole step's owed rows: filled once, every row's
        // wait must hand back the same answer. A consuming wait parked the
        // second row of a multi-row piece on an emptied cell forever (the
        // serve c4 deadlock: a wave's row 0 took the answer, row 1 hung).
        let cell = DrainCell::new();
        assert!(!cell.is_filled());
        cell.fill(Ok(vec![7, 8, 9]));
        assert!(cell.is_filled());
        let first = cell.wait();
        // The latch stays set through the read — the knock loop's poll and
        // the rows behind this one read the same answer, not an emptied cell.
        assert!(cell.is_filled());
        let second = cell.wait();
        assert!(cell.is_filled());
        assert!(matches!(&*first, Ok(v) if v.as_slice() == [7, 8, 9]));
        assert!(matches!(&*second, Ok(v) if v.as_slice() == [7, 8, 9]));
    }

    #[test]
    fn test_drain_cell_error_reaches_every_row() {
        // A cell filled with an error hands that error to every row — the
        // resolver surfaces it as the deferred output's own failure, never
        // as a hang.
        let cell = DrainCell::new();
        cell.fill(Err(crate::error::EngineError::Executor("boom".into())));
        let first = cell.wait();
        let second = cell.wait();
        assert!(first.is_err());
        assert!(second.is_err());
        assert!(cell.is_filled());
    }

    #[test]
    fn test_drain_cell_wait_parks_until_the_fill() {
        // The resolver parks before the drain fills: the fill's notify must
        // wake it, and a row that waits again behind the first reads the
        // same answer (the share, not a take).
        let cell = DrainCell::new();
        let parked = cell.clone();
        let handle = std::thread::spawn(move || {
            let answer = parked.wait();
            match &*answer {
                Ok(v) => Some(v.clone()),
                Err(_) => None,
            }
        });
        assert!(!handle.is_finished());
        std::thread::sleep(std::time::Duration::from_millis(50));
        cell.fill(Ok(vec![42]));
        assert_eq!(handle.join().unwrap(), Some(vec![42]));
        // The answer is still there for the rows behind the parked one.
        assert!(matches!(&*cell.wait(), Ok(v) if v.as_slice() == [42]));
    }

    #[test]
    fn test_model_runner_output_empty() {
        let output = ModelRunnerOutput::empty();
        assert!(output.is_empty());
        assert_eq!(output.num_requests(), 0);
        assert!(output.get_tokens("anything").is_none());
    }

    #[test]
    fn test_model_runner_output_from_token_map() {
        let mut map = HashMap::new();
        map.insert("req-1".to_string(), vec![100, 101]);
        map.insert("req-2".to_string(), vec![200]);

        let output = ModelRunnerOutput::from_token_map(map);
        assert_eq!(output.num_requests(), 2);
        assert!(!output.is_empty());

        let tokens1 = output.get_tokens("req-1").unwrap();
        assert_eq!(tokens1, &[100, 101]);

        let tokens2 = output.get_tokens("req-2").unwrap();
        assert_eq!(tokens2, &[200]);

        assert!(output.get_tokens("req-3").is_none());
    }

    #[test]
    fn test_logprobs_output() {
        let logprob = LogprobsOutput {
            sampled: TokenLogprob {
                token_id: 42,
                logprob: -0.5,
                rank: 1,
            },
            top_logprobs: vec![
                TokenLogprob {
                    token_id: 42,
                    logprob: -0.5,
                    rank: 1,
                },
                TokenLogprob {
                    token_id: 43,
                    logprob: -1.2,
                    rank: 2,
                },
            ],
        };

        assert_eq!(logprob.sampled.token_id, 42);
        assert_eq!(logprob.top_logprobs.len(), 2);
    }

    #[test]
    fn test_noop_executor_basic() {
        let mut executor = NoopExecutor::new(1024);

        // Initialize cache.
        executor.initialize_cache(1024, 0).unwrap();

        // Check available memory.
        let mem = executor.determine_available_memory().unwrap();
        assert_eq!(mem.len(), 1);

        // Create a scheduler output with one request.
        let mut num_scheduled = std::collections::HashMap::new();
        num_scheduled.insert("req-1".to_string(), 100);

        let sched_output = SchedulerOutput {
            num_scheduled_tokens: num_scheduled,
            total_num_scheduled_tokens: 100,
            ..SchedulerOutput::make_empty()
        };

        let output = executor.execute_model(&sched_output).unwrap();
        assert_eq!(output.num_requests(), 1);
        assert_eq!(output.get_tokens("req-1").unwrap(), &[1000]);

        // Second execution generates different token IDs.
        let output2 = executor.execute_model(&sched_output).unwrap();
        assert_eq!(output2.get_tokens("req-1").unwrap(), &[1001]);
    }

    #[test]
    fn test_noop_executor_shutdown() {
        let mut executor = NoopExecutor::new(512);
        assert!(!executor.is_sleeping());
        executor.shutdown();
        assert!(executor.is_sleeping());
    }

    #[test]
    fn test_noop_executor_multiple_requests() {
        let mut executor = NoopExecutor::new(1024);

        let mut num_scheduled = std::collections::HashMap::new();
        num_scheduled.insert("a".to_string(), 50);
        num_scheduled.insert("b".to_string(), 30);
        num_scheduled.insert("c".to_string(), 20);

        let sched_output = SchedulerOutput {
            num_scheduled_tokens: num_scheduled,
            total_num_scheduled_tokens: 100,
            ..SchedulerOutput::make_empty()
        };

        let output = executor.execute_model(&sched_output).unwrap();
        assert_eq!(output.num_requests(), 3);

        // Each request should get a unique token.
        let tokens: std::collections::HashSet<u32> = output
            .req_ids
            .iter()
            .filter_map(|id| output.get_tokens(id))
            .flat_map(|v| v.iter().copied())
            .collect();
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn test_noop_executor_health() {
        let executor = NoopExecutor::new(512);
        assert!(executor.check_health().is_ok());
    }

    #[test]
    fn test_noop_executor_sleep_wake() {
        let mut executor = NoopExecutor::new(512);
        assert!(!executor.is_sleeping());
        executor.sleep(1).unwrap();
        executor.wake_up(None).unwrap();
    }
}
