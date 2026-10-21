// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Scheduler output types, ported from `vllm/v1/core/sched/output.py`.
//!
//! These types describe the result of a single scheduling step and are sent
//! from the scheduler to the model runner (and workers) so they know which
//! requests to process and with how many tokens.

use std::collections::{HashMap, HashSet};

use scratchy_core_common::SamplingParams;
use scratchy_core_common::multimodal::MultimodalData;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// NewRequestData
// ---------------------------------------------------------------------------

/// Data for a request that is being scheduled for the first time.
///
/// Workers cache this data so that subsequent steps only need the diff
/// (see [`CachedRequestData`]).
///
/// Ported from the Python `NewRequestData` dataclass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewRequestData {
    /// Unique request identifier.
    pub req_id: String,

    /// The tokenized prompt. `None` when prompt embeddings are provided
    /// instead.
    pub prompt_token_ids: Option<Vec<u32>>,

    /// Block IDs assigned to this request, one `Vec<usize>` per KV cache
    /// group.
    pub block_ids: Vec<Vec<usize>>,

    /// Number of prompt tokens already computed (e.g. from prefix cache).
    pub num_computed_tokens: u32,

    /// Sampling parameters for this request.
    pub sampling_params: Option<SamplingParams>,

    /// Multimodal data (images) for vision-language models.
    /// Only present on the first scheduling of a request with images.
    pub mm_data: Option<MultimodalData>,

    /// Sparse map of block index → [`BlockKind`] for span-aware hashing/RoPE.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub block_annotations: Option<scratchy_core_common::BlockAnnotations>,

    /// Spans Phase 2: logical block indices whose KV is a reused cache hit
    /// (shared via §a), so the worker can skip rewriting their KV (slot_mapping
    /// = u32::MAX) — the cached block already holds valid (position-independent)
    /// K. `None`/empty for non-span paths. Set post-construction by the
    /// scheduler when the ordinal-reuse branch fires.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reused_block_idxs: Option<Vec<usize>>,
}

impl NewRequestData {
    /// Construct `NewRequestData` from the core fields of a request.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        req_id: String,
        prompt_token_ids: Option<Vec<u32>>,
        block_ids: Vec<Vec<usize>>,
        num_computed_tokens: u32,
        sampling_params: Option<SamplingParams>,
        mm_data: Option<MultimodalData>,
        block_annotations: Option<scratchy_core_common::BlockAnnotations>,
    ) -> Self {
        Self {
            req_id,
            prompt_token_ids,
            block_ids,
            num_computed_tokens,
            sampling_params,
            mm_data,
            block_annotations,
            reused_block_idxs: None,
        }
    }
}

// ---------------------------------------------------------------------------
// CachedRequestData
// ---------------------------------------------------------------------------

/// Data for requests that have been scheduled before.
///
/// Since the full request data is already cached in worker processes, we only
/// send the diff to minimize communication cost.
///
/// Ported from the Python `CachedRequestData` dataclass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedRequestData {
    /// Request IDs in the order they appear in this batch.
    pub req_ids: Vec<String>,

    /// Request IDs that are being resumed (e.g. after preemption). For
    /// resumed requests, `new_block_ids` replaces the existing block IDs
    /// rather than appending.
    pub resumed_req_ids: HashSet<String>,

    /// New token IDs for each request (used with pipeline parallelism).
    pub new_token_ids: Vec<Vec<u32>>,

    /// New block IDs for each request. `None` means no new blocks. Each
    /// entry is a vector of block-ID vectors, one per KV cache group.
    pub new_block_ids: Vec<Option<Vec<Vec<usize>>>>,

    /// Number of computed tokens for each request.
    pub num_computed_tokens: Vec<u32>,

    /// Number of output tokens for each request.
    pub num_output_tokens: Vec<u32>,
}

impl CachedRequestData {
    /// Create an empty `CachedRequestData` (no cached requests).
    pub fn make_empty() -> Self {
        Self {
            req_ids: Vec::new(),
            resumed_req_ids: HashSet::new(),
            new_token_ids: Vec::new(),
            new_block_ids: Vec::new(),
            num_computed_tokens: Vec::new(),
            num_output_tokens: Vec::new(),
        }
    }

    /// Number of cached requests in this batch.
    pub fn num_reqs(&self) -> usize {
        self.req_ids.len()
    }
}

// ---------------------------------------------------------------------------
// SchedulerOutput
// ---------------------------------------------------------------------------

/// The output of a single scheduling step.
///
/// Ported from the Python `SchedulerOutput` dataclass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerOutput {
    /// Requests scheduled for the first time in this step.
    pub scheduled_new_reqs: Vec<NewRequestData>,

    /// Requests that have been scheduled before (cached data).
    pub scheduled_cached_reqs: CachedRequestData,

    /// `req_id -> num_scheduled_tokens`: how many tokens are scheduled for
    /// each request in this step.
    pub num_scheduled_tokens: HashMap<String, usize>,

    /// Total number of tokens scheduled across all requests.
    /// Equal to `sum(num_scheduled_tokens.values())`.
    pub total_num_scheduled_tokens: usize,

    /// `req_id -> spec_token_ids`: speculative decode draft tokens.
    /// Only requests with draft tokens are included.
    pub scheduled_spec_decode_tokens: HashMap<String, Vec<u32>>,

    /// `req_id -> encoder_input_indices`: encoder inputs that need processing
    /// in this step.
    pub scheduled_encoder_inputs: HashMap<String, Vec<usize>>,

    /// Number of common prefix blocks for all requests in each KV cache
    /// group. Used for cascade attention.
    pub num_common_prefix_blocks: Vec<usize>,

    /// Request IDs that finished between the previous and current steps.
    /// Workers use this to free cached states.
    pub finished_req_ids: HashSet<String>,

    /// Multimodal hash strings for encoder outputs to be freed from the
    /// encoder cache.
    pub free_encoder_mm_hashes: Vec<String>,

    /// Request IDs preempted in this step (used by v2 model runner).
    pub preempted_req_ids: Option<HashSet<String>>,

    /// The requests a worker-side draft head runs for once this step's target forward has: their
    /// rows pass through it, and each that produces a token drafts the next step's. Planned by the
    /// engine's proposer after scheduling, not by the scheduler.
    #[serde(default)]
    pub draft_req_ids: HashSet<String>,

    /// This step's head reads nothing back: it is past its drafter's gate, or every request it
    /// plans runs a prefill chunk short of its prompt's end, so gets no token to draft from. Its
    /// head's pass 1 primes KV only — no chain — so the step behind it needs no outcome of it, and
    /// the worker defers it as a plain step. Stamped by the engine when it plans, not by the
    /// scheduler.
    #[serde(default)]
    pub primes_only: bool,

    /// Where this step stands among speculative steps scheduled ahead of their outcomes.
    #[serde(default)]
    pub spec_pipeline: SpecPipeline,

    /// This step's recurrent-state snapshot copies. Empty unless the model carries recurrent
    /// state (Gated-DeltaNet hybrids) and prefix caching is on.
    #[serde(default)]
    pub recurrent_state: RecurrentStateOps,

    /// Which scheduling step this is: what its coming back, or failing, settles about the
    /// prefix-cache blocks registered while it was scheduled.
    #[serde(default)]
    pub step: StepId,
}

/// ⭐ A SCHEDULING STEP, numbered from 1 by the KV cache manager as it starts
/// (`KVCacheManagerOps::new_step_starts`); 0 for an output built by hand. Steps come back, or
/// fail, in this order.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct StepId(u64);

impl StepId {
    /// The step after this one. Only the KV cache manager numbers steps.
    pub(crate) fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

/// A speculative step scheduled ahead (#240): the engine schedules the step after one that drafts
/// before that one is finalized, and the worker picks what depends on its outcome on the device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecPipeline {
    /// The step after this one may be scheduled before this one is finalized: its one request
    /// drafts here, and the engine gave its next step placeholder drafts the device fills.
    pub leads: bool,
    /// Scheduled while the step before it was in flight: its positions assume every draft that
    /// step verifies kept, its tokens and drafts are that step's on the device.
    pub behind: bool,
}

// ---------------------------------------------------------------------------
// Recurrent-state snapshots
// ---------------------------------------------------------------------------

/// One copy of a request's recurrent state between its scan and a snapshot slot.
///
/// A Gated-DeltaNet or Mamba-2 layer (Qwen3.5 / Qwen3.6, Nemotron-H) carries a per-sequence state
/// that a cached KV prefix cannot rebuild: it is the scan over EVERY earlier token. So a
/// prefix-cache hit on such a model is only real where a snapshot of that state exists, stored
/// when an earlier request's scan passed the same prefix. The scheduler owns the snapshots the way
/// it owns KV blocks — it picks the slot and the prefix — and the worker only points the
/// recurrent kernels at them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateSnapshotOp {
    /// The request whose live state is copied to or from the snapshot.
    pub req_id: String,
    /// Index into the worker's snapshot pool.
    pub slot: usize,
    /// The snapshot holds the state after tokens `[0, position)`. Block-aligned; equal to the
    /// row's `num_computed_tokens` for a restore, and within the row's chunk —
    /// `(num_computed_tokens, num_computed_tokens + num_scheduled_tokens]` — for a save. The
    /// worker checks both.
    pub position: u32,
    /// The prefix hash the snapshot is keyed by. The worker tags the slot with it on save and
    /// refuses a restore whose tag differs, so a disagreement between scheduler and worker is an
    /// error rather than another request's state.
    pub key: u64,
}

/// A request with recurrent state resumed short of the end of its KV hit: its positions
/// `[num_computed_tokens, through)` run again only to rebuild that state. Their KV is the cached
/// blocks' — shared with every request that hits them — so the worker does not write it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedKvOp {
    pub req_id: String,
    /// The end of the KV hit, block-aligned: every step of the request leaves the KV of the
    /// positions before it as cached.
    pub through: u32,
}

/// The recurrent-state work of one step (see [`StateSnapshotOp`]).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecurrentStateOps {
    /// The request was just (re)admitted at a prefix-cache hit and this is its first step: its
    /// scan starts from the snapshot, not from its own slot.
    pub restores: Vec<StateSnapshotOp>,
    /// The request's scan writes its state at `position` into the snapshot as it passes it.
    pub saves: Vec<StateSnapshotOp>,
    /// Requests (re)admitted this step whose recompute runs over cached KV.
    pub cached_kv: Vec<CachedKvOp>,
    /// The memory KV and the snapshots hand each other this step (see [`SharedMemoryOps`]).
    pub memory: SharedMemoryOps,
}

impl RecurrentStateOps {
    /// Whether this step does any recurrent-state work.
    pub fn is_empty(&self) -> bool {
        self.restores.is_empty()
            && self.saves.is_empty()
            && self.cached_kv.is_empty()
            && self.memory.is_empty()
    }
}

/// A run of [`SnapshotBudget::kv_granule_blocks`] KV blocks from block `index × that`: what KV
/// maps and releases at a time when it shares its budget with recurrent-state snapshots.
///
/// [`SnapshotBudget::kv_granule_blocks`]: crate::scheduler::SnapshotBudget::kv_granule_blocks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct KvGranule(pub u32);

/// A run of [`SnapshotBudget::unit_slots`] snapshot slots from slot `index × that`: what the
/// snapshots map and release at a time.
///
/// [`SnapshotBudget::unit_slots`]: crate::scheduler::SnapshotBudget::unit_slots
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SnapshotUnit(pub u32);

/// ⭐ THE MEMORY ONE STEP MOVES BETWEEN KV AND THE RECURRENT-STATE SNAPSHOTS, which share one
/// budget: the scheduler decides, the worker maps and releases. The worker releases before it
/// maps — a release makes room for a map in the same step — and only once every step in flight
/// is done, since a step scheduled before this one may still read what is released.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedMemoryOps {
    /// KV granules released: none of their blocks is held, and their cached contents are gone.
    pub release_kv: Vec<KvGranule>,
    /// Snapshot units released: their snapshots are gone.
    pub release_snapshots: Vec<SnapshotUnit>,
    /// KV granules mapped before this step's forward.
    pub map_kv: Vec<KvGranule>,
    /// Snapshot units mapped before this step's forward.
    pub map_snapshots: Vec<SnapshotUnit>,
}

impl SharedMemoryOps {
    pub fn is_empty(&self) -> bool {
        self.release_kv.is_empty()
            && self.release_snapshots.is_empty()
            && self.map_kv.is_empty()
            && self.map_snapshots.is_empty()
    }
}

impl SchedulerOutput {
    /// Whether the step after this one needs this one's outcome to be scheduled: the drafts it
    /// makes (`draft_req_ids`; a step past its gate makes none), or how many of the drafts it
    /// verifies it keeps. A loop that schedules ahead queues nothing behind it.
    pub fn is_speculative(&self) -> bool {
        (!self.draft_req_ids.is_empty() && !self.primes_only)
            || !self.scheduled_spec_decode_tokens.is_empty()
    }

    /// Create an empty `SchedulerOutput` with no scheduled work.
    pub fn make_empty() -> Self {
        Self {
            scheduled_new_reqs: Vec::new(),
            scheduled_cached_reqs: CachedRequestData::make_empty(),
            num_scheduled_tokens: HashMap::new(),
            total_num_scheduled_tokens: 0,
            scheduled_spec_decode_tokens: HashMap::new(),
            scheduled_encoder_inputs: HashMap::new(),
            num_common_prefix_blocks: Vec::new(),
            finished_req_ids: HashSet::new(),
            free_encoder_mm_hashes: Vec::new(),
            preempted_req_ids: None,
            draft_req_ids: HashSet::new(),
            primes_only: false,
            spec_pipeline: SpecPipeline::default(),
            recurrent_state: RecurrentStateOps::default(),
            step: StepId::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cached_request_data_empty() {
        let crd = CachedRequestData::make_empty();
        assert_eq!(crd.num_reqs(), 0);
        assert!(crd.req_ids.is_empty());
        assert!(crd.resumed_req_ids.is_empty());
    }

    #[test]
    fn test_cached_request_data_with_entries() {
        let crd = CachedRequestData {
            req_ids: vec!["r1".into(), "r2".into()],
            resumed_req_ids: HashSet::from(["r2".into()]),
            new_token_ids: vec![vec![10], vec![20]],
            new_block_ids: vec![None, Some(vec![vec![5, 6]])],
            num_computed_tokens: vec![100, 50],
            num_output_tokens: vec![10, 5],
        };
        assert_eq!(crd.num_reqs(), 2);
        assert!(crd.resumed_req_ids.contains("r2"));
    }

    #[test]
    fn test_new_request_data() {
        let nrd = NewRequestData::new(
            "req-1".into(),
            Some(vec![1, 2, 3]),
            vec![vec![0, 1, 2]],
            10,
            None,
            None,
            None,
        );
        assert_eq!(nrd.req_id, "req-1");
        assert_eq!(nrd.prompt_token_ids.as_ref().unwrap().len(), 3);
        assert_eq!(nrd.num_computed_tokens, 10);
    }

    #[test]
    fn test_scheduler_output_empty() {
        let so = SchedulerOutput::make_empty();
        assert!(so.scheduled_new_reqs.is_empty());
        assert_eq!(so.total_num_scheduled_tokens, 0);
        assert!(so.finished_req_ids.is_empty());
        assert!(so.preempted_req_ids.is_none());
    }

    #[test]
    fn test_scheduler_output_with_data() {
        let mut num_scheduled = HashMap::new();
        num_scheduled.insert("r1".into(), 100);
        num_scheduled.insert("r2".into(), 50);

        let so = SchedulerOutput {
            scheduled_new_reqs: vec![NewRequestData::new(
                "r1".into(),
                Some(vec![1, 2, 3]),
                vec![vec![0]],
                0,
                None,
                None,
                None,
            )],
            scheduled_cached_reqs: CachedRequestData {
                req_ids: vec!["r2".into()],
                resumed_req_ids: HashSet::new(),
                new_token_ids: Vec::new(),
                new_block_ids: vec![None],
                num_computed_tokens: vec![50],
                num_output_tokens: vec![10],
            },
            num_scheduled_tokens: num_scheduled,
            total_num_scheduled_tokens: 150,
            scheduled_spec_decode_tokens: HashMap::new(),
            scheduled_encoder_inputs: HashMap::new(),
            num_common_prefix_blocks: vec![0],
            finished_req_ids: HashSet::new(),
            free_encoder_mm_hashes: Vec::new(),
            preempted_req_ids: None,
            draft_req_ids: HashSet::new(),
            primes_only: false,
            spec_pipeline: SpecPipeline::default(),
            recurrent_state: RecurrentStateOps::default(),
            step: StepId::default(),
        };
        assert_eq!(so.total_num_scheduled_tokens, 150);
        assert_eq!(so.scheduled_new_reqs.len(), 1);
        assert_eq!(so.scheduled_cached_reqs.num_reqs(), 1);
    }

    #[test]
    fn test_scheduler_output_is_speculative() {
        let mut so = SchedulerOutput::make_empty();
        assert!(!so.is_speculative());

        // A step that drafts: the step after it needs its drafts.
        so.draft_req_ids.insert("r1".into());
        assert!(so.is_speculative());

        // Past the gate the head primes KV only — nothing drafted, nothing read back.
        so.primes_only = true;
        assert!(!so.is_speculative());

        // A gated step that still verifies drafts decides what follows it: how many it keeps.
        so.scheduled_spec_decode_tokens
            .insert("r1".into(), vec![7, 8]);
        assert!(so.is_speculative());
    }
}
