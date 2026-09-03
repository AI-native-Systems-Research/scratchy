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
}

impl SchedulerOutput {
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
        };
        assert_eq!(so.total_num_scheduled_tokens, 150);
        assert_eq!(so.scheduled_new_reqs.len(), 1);
        assert_eq!(so.scheduled_cached_reqs.num_reqs(), 1);
    }
}
