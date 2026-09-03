// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Request and request-status types, ported from `vllm/v1/request.py`.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::engine_io::FinishReason;
use crate::multimodal::MultimodalData;
use crate::sampling::SamplingParams;

// ---------------------------------------------------------------------------
// BlockKind — span annotation for block hashing
// ---------------------------------------------------------------------------

/// Annotation for a block's hashing behavior in the span-aware cache.
///
/// The `/v1/query/execute` endpoint produces a sparse map of block indices to
/// `BlockKind` values. The block hasher reads these annotations to decide
/// parent-hash chaining.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockKind {
    /// Position-independent block: parent hash is reset to `NONE_HASH`,
    /// making this block cacheable regardless of where it appears in the
    /// sequence. `first_token` is the span's first TOKEN index — used directly
    /// as the block-diagonal attention lower bound, so a span needs NO block
    /// alignment (its partial boundary block is simply part of the span; the
    /// tokenizer never crops/pads to a block multiple).
    Relocatable { first_token: u32 },
    /// Prefix-dependent block: all preceding tokens are folded into the hash,
    /// forcing recomputation when any prior context differs. Also carries its
    /// `first_token` so it forms an exact RIGHT boundary for the preceding
    /// relocatable span (the query token where that span ends).
    Prefixed { first_token: u32 },
}

impl BlockKind {
    /// Whether this block begins/continues a relocatable span.
    pub fn is_relocatable(&self) -> bool {
        matches!(self, BlockKind::Relocatable { .. })
    }
    /// The block's first TOKEN index. Carried by BOTH kinds so per-token span
    /// labeling has exact boundaries (a `Prefixed` annotation ends the preceding
    /// span at the precise query token).
    pub fn first_token(&self) -> u32 {
        match self {
            BlockKind::Relocatable { first_token } | BlockKind::Prefixed { first_token } => {
                *first_token
            }
        }
    }
}

/// Sparse map of block index to [`BlockKind`] for span-aware block hashing.
pub type BlockAnnotations = BTreeMap<usize, BlockKind>;

/// Compute per-block RoPE rotation flags for a single block.
///
/// Returns `(is_relocatable, is_unrotated)`:
/// - `is_relocatable`: true if the block is annotated as [`BlockKind::Relocatable`].
///   Post-attention un-rotation will remove RoPE from this block.
/// - `is_unrotated`: true if the block was written in a prior step (its K is
///   currently stored without RoPE). Pre-attention rotation will apply RoPE.
///
/// `block_idx`: logical block index in the sequence.
/// `block_size`: tokens per block.
/// `seq_len`: total sequence length.
/// `tokens_before`: number of tokens computed before this step.
pub fn compute_block_flags(
    annotations: &BlockAnnotations,
    block_idx: usize,
    _block_size: usize,
    _seq_len: usize,
    _tokens_before: usize,
) -> (bool, bool) {
    let is_relocatable = annotations
        .get(&block_idx)
        .is_some_and(BlockKind::is_relocatable);
    // NOTE: is_unrotated is always false because there is no post-attention
    // inverse-RoPE pass. K values in the cache are always stored WITH RoPE
    // from the QKV projection. Setting is_unrotated=true would cause the
    // attention kernel to double-rotate K on reuse, destroying attention.
    let is_unrotated = false;
    (is_relocatable, is_unrotated)
}

/// Metal spans (rope-on-read) variant of [`compute_block_flags`].
///
/// Returns `(is_relocatable, is_unrotated)` where **`is_unrotated ==
/// is_relocatable`**: under rope-on-read, a `BlockKind::Relocatable`
/// (span) block has its K stored WITHOUT RoPE by `rope_append`, and the
/// paged attention kernels re-rope it to each reader's position on read.
/// So a relocatable block IS an unrotated block — the flag the metal KV
/// pool mirrors to the GPU (`block_is_unrotated`) for the kernels' gate.
///
/// This is DISTINCT from [`compute_block_flags`], which is consumed by
/// the CUDA worker's rotate-on-write design (K stored WITH RoPE,
/// `is_unrotated` always false) and MUST stay byte-identical — hence a
/// separate function rather than a parameter on the shared one. Only the
/// metal worker calls this.
pub fn compute_block_flags_rope_on_read(
    annotations: &BlockAnnotations,
    block_idx: usize,
) -> (bool, bool) {
    let is_relocatable = annotations
        .get(&block_idx)
        .is_some_and(BlockKind::is_relocatable);
    (is_relocatable, is_relocatable)
}

// ---------------------------------------------------------------------------
// Span partition — block-diagonal isolation for Relocatable spans
// ---------------------------------------------------------------------------

/// A contiguous run of logical blocks forming ONE isolated span (`[start, end)`).
///
/// For block-diagonal prefill, a span's tokens attend ONLY their own blocks
/// (SELF-ONLY) — never sibling spans, never any prefix — so the span's K/V is
/// context-independent. That is exactly what the `Relocatable` cache key already
/// promises (parent hash reset to `NONE_HASH`), so isolated-prefilling a span
/// this way makes the existing content-addressed reuse CORRECT under any
/// recombination. (Today a `Relocatable` span is prefilled FULL-CAUSAL yet reused
/// content-addressed → it silently returns K/V computed for the wrong
/// neighbors.) A richer `[shared-prefix + self]`
/// mode is possible later but would require folding that prefix into the
/// content-address — it is NOT self-only and is out of scope here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpanRange {
    /// First logical block index of the span (inclusive).
    pub start: usize,
    /// One past the last logical block of the span (exclusive).
    pub end: usize,
    /// First TOKEN index of the span — the exact block-diagonal attention lower
    /// bound. The span may begin mid-block; that partial boundary block is still
    /// part of the span (no cropping/padding to a block multiple).
    pub first_token: u32,
}

impl SpanRange {
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

/// Partition a sequence's blocks into its independent `Relocatable` spans, from
/// the sparse [`BlockAnnotations`].
///
/// Each `Relocatable` annotation STARTS a span; the span extends through the
/// following (un-annotated) blocks until the next annotated block (of EITHER
/// kind) or `num_blocks`. This mirrors the hash semantics: a `Relocatable` block
/// resets the parent hash (independent of predecessors) and the blocks chaining
/// off it depend only on the span's own start, so the run is collectively
/// independent of sibling spans. Blocks NOT inside any returned span (a leading
/// prefix, a `Prefixed` tail) stay on the normal full-causal path. Annotation
/// keys `>= num_blocks` are ignored.
pub fn relocatable_spans(annotations: &BlockAnnotations, num_blocks: usize) -> Vec<SpanRange> {
    // BTreeMap iterates keys in ascending order, so `keys[i+1]` is the next
    // annotated block — the boundary that ends span `i`.
    let keys: Vec<usize> = annotations
        .keys()
        .copied()
        .filter(|&k| k < num_blocks)
        .collect();
    let mut spans = Vec::new();
    for (i, &start) in keys.iter().enumerate() {
        let Some(kind) = annotations.get(&start) else {
            continue;
        };
        if kind.is_relocatable() {
            let end = keys.get(i + 1).copied().unwrap_or(num_blocks);
            if end > start {
                spans.push(SpanRange {
                    start,
                    end,
                    first_token: kind.first_token(),
                });
            }
        }
    }
    spans
}

/// The isolated (self-only) block-table for a span: just the span's own blocks,
/// sliced from the request's full block-table. Handing this — with a matching
/// `seqused_k = the span's own token count` — to the existing causal attention
/// kernel yields block-diagonal isolation with NO kernel change: the kernel
/// masks over whatever scope it is given, and this scope excludes every sibling
/// span. Bounds-clamped so an out-of-range span can't panic.
pub fn isolated_span_block_table(block_table: &[usize], span: SpanRange) -> &[usize] {
    let end = span.end.min(block_table.len());
    let start = span.start.min(end);
    &block_table[start..end]
}

/// Per-TOKEN span label for block-diagonal attention. The kernel indexes this by
/// RAW TOKEN POSITION (not by block), so a span may begin AND end mid-block — no
/// alignment, no cropping. `0` outside any relocatable span (the shared prefix
/// and the `Prefixed` query/tail — the "attends everything" group). Inside a
/// span, the label is the span's FIRST TOKEN `+ 1`, so the kernel's lower bound
/// `label - 1` is the exact first token.
///
/// Every annotation (relocatable OR prefixed) is a token boundary: a span runs
/// from its first token up to the NEXT annotation's first token. A `Prefixed`
/// boundary (the query) carries label `0` so it merely terminates the preceding
/// span. This is what turns O(N²) into O(N·span) without any block alignment.
pub fn span_ids_per_token(annotations: &BlockAnnotations, num_tokens: usize) -> Vec<u32> {
    let mut ids = vec![0u32; num_tokens];
    // (first_token, label) boundaries, ascending. Relocatable → first_token+1;
    // Prefixed → 0. Before the first boundary (the shared prefix) stays 0.
    let mut bounds: Vec<(usize, u32)> = annotations
        .values()
        .map(|k| {
            let ft = k.first_token();
            (ft as usize, if k.is_relocatable() { ft + 1 } else { 0 })
        })
        .collect();
    bounds.sort_unstable();
    for (i, &(start, label)) in bounds.iter().enumerate() {
        let end = bounds.get(i + 1).map(|&(s, _)| s).unwrap_or(num_tokens);
        for id in ids
            .iter_mut()
            .take(end.min(num_tokens))
            .skip(start.min(num_tokens))
        {
            *id = label;
        }
    }
    ids
}

#[cfg(test)]
mod span_partition_tests {
    use super::*;

    // Test block size; these block-partition tests use block-aligned spans, so a
    // span at block `b` has first_token = b * BS.
    const BS: u32 = 16;
    fn ann(pairs: &[(usize, bool)]) -> BlockAnnotations {
        pairs
            .iter()
            .map(|&(b, reloc)| {
                let first_token = b as u32 * BS;
                let k = if reloc {
                    BlockKind::Relocatable { first_token }
                } else {
                    BlockKind::Prefixed { first_token }
                };
                (b, k)
            })
            .collect()
    }
    fn span(start: usize, end: usize) -> SpanRange {
        SpanRange {
            start,
            end,
            first_token: start as u32 * BS,
        }
    }

    #[test]
    fn no_annotations_yields_no_spans() {
        assert!(relocatable_spans(&BlockAnnotations::new(), 10).is_empty());
    }

    #[test]
    fn three_single_block_spans() {
        // e.g. system = blocks 0..2 (prefix), three 1-block tool spans at 2,3,4.
        let a = ann(&[(2, true), (3, true), (4, true)]);
        assert_eq!(
            relocatable_spans(&a, 5),
            vec![span(2, 3), span(3, 4), span(4, 5)]
        );
    }

    #[test]
    fn multi_block_span_runs_to_next_annotation() {
        // span1 = blocks 1..4 (chains until the next annotation at 4), span2 = 4..6.
        let a = ann(&[(1, true), (4, true)]);
        assert_eq!(relocatable_spans(&a, 6), vec![span(1, 4), span(4, 6)]);
    }

    #[test]
    fn prefixed_block_bounds_a_span_but_is_not_one() {
        // span = 1..3; the Prefixed downstream block at 3 ends it and is NOT a span.
        let a = ann(&[(1, true), (3, false)]);
        assert_eq!(relocatable_spans(&a, 5), vec![span(1, 3)]);
    }

    #[test]
    fn leading_prefix_excluded_trailing_span_runs_to_end() {
        // blocks 0,1 = unannotated shared prefix; one span 2..5.
        let a = ann(&[(2, true)]);
        assert_eq!(relocatable_spans(&a, 5), vec![span(2, 5)]);
    }

    #[test]
    fn annotation_past_num_blocks_is_ignored() {
        let a = ann(&[(2, true), (99, true)]);
        assert_eq!(relocatable_spans(&a, 4), vec![span(2, 4)]);
    }

    #[test]
    fn isolated_block_table_slices_own_blocks_only() {
        let bt = vec![10usize, 11, 12, 13, 14];
        assert_eq!(isolated_span_block_table(&bt, span(1, 3)), &[11, 12]);
        // out-of-range end clamps instead of panicking.
        assert_eq!(isolated_span_block_table(&bt, span(3, 99)), &[13, 14]);
    }

    #[test]
    fn span_ids_per_token_labels_by_first_token() {
        // Two relocatable spans + a prefixed query, keyed by first TOKEN (no block
        // math, so a span may begin mid-block). Block keys are arbitrary.
        let mut a = BlockAnnotations::new();
        a.insert(0, BlockKind::Relocatable { first_token: 3 });
        a.insert(1, BlockKind::Relocatable { first_token: 7 });
        a.insert(2, BlockKind::Prefixed { first_token: 10 });
        // 0..3 prefix(0), 3..7 spanA(label 4), 7..10 spanB(label 8), 10..12 query(0).
        assert_eq!(
            span_ids_per_token(&a, 12),
            vec![0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 0, 0]
        );
    }

    #[test]
    fn span_ids_empty_when_no_spans() {
        assert_eq!(
            span_ids_per_token(&BlockAnnotations::new(), 4),
            vec![0, 0, 0, 0]
        );
    }
}

// ---------------------------------------------------------------------------
// RequestStatus
// ---------------------------------------------------------------------------

/// Status of a request as it moves through the engine pipeline.
///
/// Values after `Preempted` are considered "finished".
/// This mirrors the Python `RequestStatus(IntEnum)` from `vllm/v1/request.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RequestStatus {
    Waiting = 1,
    WaitingForFsm = 2,
    WaitingForRemoteKvs = 3,
    WaitingForStreamingReq = 4,
    Running = 5,
    Preempted = 6,
    // --- everything below is "finished" ---
    FinishedStopped = 7,
    FinishedLengthCapped = 8,
    FinishedAborted = 9,
    FinishedIgnored = 10,
    FinishedError = 11,
}

impl RequestStatus {
    /// Returns `true` if the status represents a terminal (finished) state.
    ///
    /// A request is finished if its discriminant is greater than `Preempted`.
    pub fn is_finished(self) -> bool {
        (self as u8) > (Self::Preempted as u8)
    }

    /// Map a finished status to its corresponding [`FinishReason`].
    ///
    /// Returns `None` for non-finished statuses (with the exception of
    /// `WaitingForStreamingReq`, which maps to `Stop` -- matching the
    /// Python `_FINISHED_REASON_MAP`).
    pub fn get_finished_reason(self) -> Option<FinishReason> {
        match self {
            Self::FinishedStopped => Some(FinishReason::Stop),
            Self::FinishedLengthCapped => Some(FinishReason::Length),
            Self::FinishedAborted => Some(FinishReason::Abort),
            // Ignored requests hit the model length cap, so the reason is Length.
            Self::FinishedIgnored => Some(FinishReason::Length),
            Self::FinishedError => Some(FinishReason::Error),
            Self::WaitingForStreamingReq => Some(FinishReason::Stop),
            _ => None,
        }
    }
}

impl std::fmt::Display for RequestStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Match Python's `__str__` which returns the variant name.
        // Use write_str with a match to avoid Debug formatter overhead.
        f.write_str(match self {
            Self::Waiting => "Waiting",
            Self::WaitingForFsm => "WaitingForFsm",
            Self::WaitingForRemoteKvs => "WaitingForRemoteKvs",
            Self::WaitingForStreamingReq => "WaitingForStreamingReq",
            Self::Running => "Running",
            Self::Preempted => "Preempted",
            Self::FinishedStopped => "FinishedStopped",
            Self::FinishedLengthCapped => "FinishedLengthCapped",
            Self::FinishedAborted => "FinishedAborted",
            Self::FinishedIgnored => "FinishedIgnored",
            Self::FinishedError => "FinishedError",
        })
    }
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// Core scheduler-level representation of an in-flight request.
///
/// Ported from the Python `Request` class in `vllm/v1/request.py`.
/// Heavy Python-only fields (tensors, block-hashers, structured-output FSM
/// state, LoRA requests, multimodal features) are omitted here; they will be
/// added in later phases or handled via trait objects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    /// Unique identifier for this request.
    pub request_id: String,

    /// Index of the front-end client that owns this request.
    pub client_index: u32,

    /// Scheduling priority (lower = higher priority).
    pub priority: i32,

    /// Sampling parameters for generation.
    pub sampling_params: SamplingParams,

    /// Wall-clock arrival time (seconds since epoch).
    pub arrival_time: f64,

    /// Current lifecycle status.
    pub status: RequestStatus,

    /// Maximum number of tokens to generate for this request.
    pub max_tokens: u32,

    /// The tokenized prompt.
    pub prompt_token_ids: Vec<u32>,

    /// Tokens generated so far (output only, excludes prompt).
    pub output_token_ids: Vec<u32>,

    /// Concatenation of `prompt_token_ids` and `output_token_ids`.
    pub all_token_ids: Vec<u32>,

    /// Speculative-decoding draft token IDs (if any).
    pub spec_token_ids: Vec<u32>,

    /// Number of tokens that have been computed (prompt + output KV cached).
    pub num_computed_tokens: u32,

    /// Length of the prompt in tokens.
    pub num_prompt_tokens: u32,

    /// Optional per-request cache salt for prefix-cache isolation.
    pub cache_salt: Option<String>,

    /// Number of prompt tokens served from cache (local + external).
    /// -1 means "not yet known".
    pub num_cached_tokens: i32,

    /// `true` while the request is being prefilled in chunks (not yet
    /// finished prefill).
    pub is_prefill_chunk: bool,

    /// How many times the scheduler has preempted this request.
    pub num_preemptions: u32,

    /// Number of tokens computed remotely (P/D disaggregated serving).
    pub num_external_computed_tokens: u32,

    /// Number of placeholder output tokens reserved for async scheduling.
    pub num_output_placeholders: u32,

    /// Whether this is a pooling (embedding) request rather than generation.
    /// Pooling requests are finished after one forward pass (no decode loop).
    pub is_pooling: bool,

    /// Multimodal data (images) for vision-language models.
    /// Set once at request creation, consumed during the first prefill step.
    #[serde(skip)]
    pub mm_data: Option<MultimodalData>,

    /// Sparse map of block index → [`BlockKind`] for span-aware block hashing.
    /// Only blocks with non-default hashing behavior are present.
    /// `None` means all blocks use normal parent-chained hashing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_annotations: Option<BlockAnnotations>,

    /// 🦭 When true, pad and hash the final partial block on completion so
    /// future requests can get a cache hit on this request's full output.
    #[serde(default)]
    pub seal: bool,

    /// When true, deprioritize this request's cached blocks for eviction
    /// after generation completes. Used for one-shot consumers like inner
    /// generates in a nested generation pattern.
    #[serde(default)]
    pub volatile: bool,

    /// ⭐ WHAT THE WORKER'S KV LOOKS LIKE FOR THIS REQUEST — reported back every step by backends whose
    /// keys do not sit at their token positions, and `None` for the ones where they do (cuda/metal).
    ///
    /// Two things depend on it, and both are wrong-output bugs when it is missing on a backend that
    /// needs it:
    ///
    /// * blocks are allocated for [`KvExtent::span`] rather than the token count, because a pool whose
    ///   batched write appends every row at ONE slot per step spreads a short request's keys over MORE
    ///   slots than it has tokens;
    /// * only [`KvExtent::cacheable_tokens`] may be hashed into the prefix cache, because past that
    ///   request's first hole token `t` no longer lives at slot `t`, and a block hashed over it would
    ///   hand a later request the keys of the wrong tokens.
    ///
    /// ⛔ NOT DERIVED HERE. The scheduler picks the batch and could run the same recurrence the worker
    /// does, and that is exactly the two-sources-for-one-quantity shape this path's defects have all had.
    #[serde(default)]
    pub kv_extent: Option<crate::kv::KvExtent>,
}

impl Request {
    /// Create a new `Request` with the given core fields.
    ///
    /// Initializes all mutable counters to their default state.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_id: String,
        prompt_token_ids: Vec<u32>,
        sampling_params: SamplingParams,
        arrival_time: f64,
        client_index: u32,
        priority: i32,
        cache_salt: Option<String>,
    ) -> Self {
        let num_prompt_tokens = prompt_token_ids.len() as u32;
        let max_tokens = sampling_params.max_tokens.unwrap_or(u32::MAX);

        Self {
            request_id,
            client_index,
            priority,
            sampling_params,
            arrival_time,
            status: RequestStatus::Waiting,
            max_tokens,
            all_token_ids: prompt_token_ids.clone(),
            prompt_token_ids,
            output_token_ids: Vec::new(),
            spec_token_ids: Vec::new(),
            num_computed_tokens: 0,
            num_prompt_tokens,
            cache_salt,
            num_cached_tokens: -1,
            is_prefill_chunk: false,
            num_preemptions: 0,
            num_external_computed_tokens: 0,
            num_output_placeholders: 0,
            is_pooling: false,
            mm_data: None,
            block_annotations: None,
            seal: false,
            volatile: false,
            kv_extent: None,
        }
    }

    /// Append one or more output token IDs, updating both `output_token_ids`
    /// and `all_token_ids`.
    pub fn append_output_token_ids(&mut self, token_ids: &[u32]) {
        self.output_token_ids.extend_from_slice(token_ids);
        self.all_token_ids.extend_from_slice(token_ids);
    }

    /// The total number of tokens (prompt + output) currently tracked.
    pub fn num_tokens(&self) -> usize {
        self.all_token_ids.len()
    }

    /// Total tokens including speculative draft tokens.
    pub fn num_tokens_with_spec(&self) -> usize {
        self.all_token_ids.len() + self.spec_token_ids.len()
    }

    /// The number of output tokens generated so far.
    pub fn num_output_tokens(&self) -> usize {
        self.output_token_ids.len()
    }

    /// Whether this request has reached a terminal state.
    pub fn is_finished(&self) -> bool {
        self.status.is_finished()
    }

    /// The finish reason, if the request is in a terminal state.
    pub fn get_finished_reason(&self) -> Option<FinishReason> {
        self.status.get_finished_reason()
    }
}

// ---------------------------------------------------------------------------
// Ordering (for priority scheduling)
// ---------------------------------------------------------------------------

impl PartialEq for Request {
    fn eq(&self, other: &Self) -> bool {
        self.request_id == other.request_id
    }
}

impl Eq for Request {}

impl PartialOrd for Request {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Ordering used by the scheduler's priority queue.
///
/// Lower priority value => higher scheduling priority.
/// Ties are broken by arrival time (earlier first), then by request ID
/// (lexicographic).
impl Ord for Request {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| {
                self.arrival_time
                    .partial_cmp(&other.arrival_time)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| self.request_id.cmp(&other.request_id))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(id: &str, priority: i32, arrival: f64) -> Request {
        Request::new(
            id.into(),
            vec![1, 2, 3],
            SamplingParams::default(),
            arrival,
            0,
            priority,
            None,
        )
    }

    // -- RequestStatus tests --

    #[test]
    fn test_status_is_finished() {
        assert!(!RequestStatus::Waiting.is_finished());
        assert!(!RequestStatus::WaitingForFsm.is_finished());
        assert!(!RequestStatus::WaitingForRemoteKvs.is_finished());
        assert!(!RequestStatus::WaitingForStreamingReq.is_finished());
        assert!(!RequestStatus::Running.is_finished());
        assert!(!RequestStatus::Preempted.is_finished());

        assert!(RequestStatus::FinishedStopped.is_finished());
        assert!(RequestStatus::FinishedLengthCapped.is_finished());
        assert!(RequestStatus::FinishedAborted.is_finished());
        assert!(RequestStatus::FinishedIgnored.is_finished());
        assert!(RequestStatus::FinishedError.is_finished());
    }

    #[test]
    fn test_status_finish_reasons() {
        assert_eq!(
            RequestStatus::FinishedStopped.get_finished_reason(),
            Some(FinishReason::Stop)
        );
        assert_eq!(
            RequestStatus::FinishedLengthCapped.get_finished_reason(),
            Some(FinishReason::Length)
        );
        assert_eq!(
            RequestStatus::FinishedAborted.get_finished_reason(),
            Some(FinishReason::Abort)
        );
        assert_eq!(
            RequestStatus::FinishedIgnored.get_finished_reason(),
            Some(FinishReason::Length)
        );
        assert_eq!(
            RequestStatus::FinishedError.get_finished_reason(),
            Some(FinishReason::Error)
        );
        assert_eq!(
            RequestStatus::WaitingForStreamingReq.get_finished_reason(),
            Some(FinishReason::Stop)
        );
    }

    #[test]
    fn test_status_non_finished_reason_is_none() {
        assert_eq!(RequestStatus::Waiting.get_finished_reason(), None);
        assert_eq!(RequestStatus::Running.get_finished_reason(), None);
        assert_eq!(RequestStatus::Preempted.get_finished_reason(), None);
    }

    #[test]
    fn test_status_display() {
        assert_eq!(format!("{}", RequestStatus::Waiting), "Waiting");
        assert_eq!(format!("{}", RequestStatus::Running), "Running");
        assert_eq!(
            format!("{}", RequestStatus::FinishedStopped),
            "FinishedStopped"
        );
    }

    #[test]
    fn test_status_repr_values() {
        assert_eq!(RequestStatus::Waiting as u8, 1);
        assert_eq!(RequestStatus::WaitingForFsm as u8, 2);
        assert_eq!(RequestStatus::WaitingForRemoteKvs as u8, 3);
        assert_eq!(RequestStatus::WaitingForStreamingReq as u8, 4);
        assert_eq!(RequestStatus::Running as u8, 5);
        assert_eq!(RequestStatus::Preempted as u8, 6);
        assert_eq!(RequestStatus::FinishedStopped as u8, 7);
        assert_eq!(RequestStatus::FinishedLengthCapped as u8, 8);
        assert_eq!(RequestStatus::FinishedAborted as u8, 9);
        assert_eq!(RequestStatus::FinishedIgnored as u8, 10);
        assert_eq!(RequestStatus::FinishedError as u8, 11);
    }

    // -- Request construction tests --

    #[test]
    fn test_new_request_defaults() {
        let req = make_request("r1", 0, 1.0);
        assert_eq!(req.request_id, "r1");
        assert_eq!(req.status, RequestStatus::Waiting);
        assert_eq!(req.num_prompt_tokens, 3);
        assert_eq!(req.num_computed_tokens, 0);
        assert_eq!(req.num_cached_tokens, -1);
        assert!(!req.is_prefill_chunk);
        assert_eq!(req.num_preemptions, 0);
        assert!(req.output_token_ids.is_empty());
        assert_eq!(req.all_token_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_append_output_tokens() {
        let mut req = make_request("r1", 0, 1.0);
        req.append_output_token_ids(&[10, 11]);
        assert_eq!(req.output_token_ids, vec![10, 11]);
        assert_eq!(req.all_token_ids, vec![1, 2, 3, 10, 11]);
        assert_eq!(req.num_tokens(), 5);
        assert_eq!(req.num_output_tokens(), 2);
    }

    #[test]
    fn test_num_tokens_with_spec() {
        let mut req = make_request("r1", 0, 1.0);
        req.spec_token_ids = vec![99, 100, 101];
        assert_eq!(req.num_tokens_with_spec(), 6); // 3 prompt + 3 spec
    }

    #[test]
    fn test_is_finished() {
        let mut req = make_request("r1", 0, 1.0);
        assert!(!req.is_finished());

        req.status = RequestStatus::FinishedStopped;
        assert!(req.is_finished());
        assert_eq!(req.get_finished_reason(), Some(FinishReason::Stop));
    }

    // -- Ordering tests --

    #[test]
    fn test_ordering_by_priority() {
        let r_high = make_request("a", 0, 1.0); // higher priority (lower value)
        let r_low = make_request("b", 10, 1.0);
        assert!(r_high < r_low);
    }

    #[test]
    fn test_ordering_by_arrival_time() {
        let r_early = make_request("a", 0, 1.0);
        let r_late = make_request("b", 0, 2.0);
        assert!(r_early < r_late);
    }

    #[test]
    fn test_ordering_by_request_id() {
        let r_a = make_request("aaa", 0, 1.0);
        let r_b = make_request("bbb", 0, 1.0);
        assert!(r_a < r_b);
    }

    #[test]
    fn test_equality_by_request_id() {
        let r1 = make_request("r1", 0, 1.0);
        let r2 = make_request("r1", 5, 99.0);
        assert_eq!(r1, r2); // equality is by request_id only
    }

    #[test]
    fn test_sort_requests() {
        let r1 = make_request("c", 0, 3.0);
        let r2 = make_request("a", 0, 1.0);
        let r3 = make_request("b", 0, 2.0);
        let r4 = make_request("d", -1, 10.0); // highest priority

        let mut reqs = [r1, r2, r3, r4];
        reqs.sort();

        let ids: Vec<&str> = reqs.iter().map(|r| r.request_id.as_str()).collect();
        // d has priority -1 (first), then a,b,c ordered by arrival time
        assert_eq!(ids, vec!["d", "a", "b", "c"]);
    }

    // -- Serde tests --

    #[test]
    fn test_request_serde_roundtrip() {
        let req = make_request("r1", 0, 1.0);
        let json = serde_json::to_string(&req).unwrap();
        let req2: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(req2.request_id, "r1");
        assert_eq!(req2.num_prompt_tokens, 3);
        assert_eq!(req2.status, RequestStatus::Waiting);
    }

    #[test]
    fn test_request_status_serde_roundtrip() {
        let status = RequestStatus::FinishedAborted;
        let json = serde_json::to_string(&status).unwrap();
        let status2: RequestStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status, status2);
    }

    // -----------------------------------------------------------------------
    // compute_block_flags tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_block_flags_unannotated_block() {
        // No annotations → never flagged, regardless of position or timing.
        let ann = BTreeMap::new();
        let (is_reloc, is_unrot) = compute_block_flags(&ann, 0, 16, 64, 32);
        assert!(!is_reloc);
        assert!(!is_unrot);
    }

    #[test]
    fn test_block_flags_relocatable_freshly_written() {
        // Relocatable block written THIS step: is_relocatable=true,
        // is_unrotated=false (K has RoPE from QKV projection).
        let mut ann = BTreeMap::new();
        ann.insert(0, BlockKind::Relocatable { first_token: 0 });
        // block 0, block_size=4, seq_len=8, tokens_before=0 (all new)
        let (is_reloc, is_unrot) = compute_block_flags(&ann, 0, 4, 8, 0);
        assert!(is_reloc);
        assert!(!is_unrot); // freshly written → still has RoPE
    }

    #[test]
    fn test_block_flags_relocatable_previously_cached() {
        // ⛔ CACHE STATE DOES NOT MOVE `is_unrotated` — there is no post-attention inverse-RoPE pass, so K
        // in the cache always carries the RoPE the QKV projection gave it, prior step or not. This test
        // asserted the opposite (`is_unrot` on a cached block) and encoded the rotate-on-read design that
        // `compute_block_flags` deliberately does NOT implement; `compute_block_flags_rope_on_read` is where
        // `is_unrotated == is_relocatable` holds, and only metal calls it.
        let mut ann = BTreeMap::new();
        ann.insert(0, BlockKind::Relocatable { first_token: 0 });
        // block 0, block_size=4, seq_len=8, tokens_before=4 (block 0 fully cached)
        let (is_reloc, is_unrot) = compute_block_flags(&ann, 0, 4, 8, 4);
        assert!(is_reloc);
        assert!(
            !is_unrot,
            "cached or fresh, rotate-on-write K is never unrotated"
        );
    }

    #[test]
    fn test_block_flags_prefixed_never_flagged() {
        // Prefixed blocks are NOT Relocatable — they should never be
        // flagged for rotation, regardless of cache state.
        let mut ann = BTreeMap::new();
        ann.insert(2, BlockKind::Prefixed { first_token: 0 });
        let (is_reloc, is_unrot) = compute_block_flags(&ann, 2, 4, 16, 12);
        assert!(!is_reloc);
        assert!(!is_unrot);
    }

    #[test]
    fn test_block_flags_mixed_annotations() {
        // Sequence: [Relocatable, Relocatable, Prefixed]
        // All previously cached (tokens_before covers all).
        let mut ann = BTreeMap::new();
        ann.insert(0, BlockKind::Relocatable { first_token: 0 });
        ann.insert(1, BlockKind::Relocatable { first_token: 0 });
        ann.insert(2, BlockKind::Prefixed { first_token: 0 });

        let block_size = 4;
        let seq_len = 12;
        let tokens_before = 12; // all cached

        // Block 0: Relocatable — flagged relocatable, NEVER unrotated (see the cached-block test above).
        let (r, u) = compute_block_flags(&ann, 0, block_size, seq_len, tokens_before);
        assert!(r);
        assert!(!u);

        // Block 1: Relocatable — same.
        let (r, u) = compute_block_flags(&ann, 1, block_size, seq_len, tokens_before);
        assert!(r);
        assert!(!u);

        // Block 2: Prefixed → not flagged
        let (r, u) = compute_block_flags(&ann, 2, block_size, seq_len, tokens_before);
        assert!(!r);
        assert!(!u);
    }

    #[test]
    fn test_block_flags_partially_written_block() {
        // Block is being written this step (block_end > tokens_before).
        let mut ann = BTreeMap::new();
        ann.insert(1, BlockKind::Relocatable { first_token: 0 });
        // block 1 spans positions 4..8, tokens_before=6 → partially written
        let (is_reloc, is_unrot) = compute_block_flags(&ann, 1, 4, 8, 6);
        assert!(is_reloc);
        assert!(!is_unrot); // not fully cached → freshly written
    }

    // -----------------------------------------------------------------------
    // compute_block_flags_rope_on_read (metal spans) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rope_on_read_flags_relocatable_is_unrotated() {
        // Rope-on-read: a Relocatable block is stored unrotated, so
        // is_unrotated == is_relocatable == true (unlike the CUDA fn).
        let mut ann = BTreeMap::new();
        ann.insert(0, BlockKind::Relocatable { first_token: 0 });
        let (is_reloc, is_unrot) = compute_block_flags_rope_on_read(&ann, 0);
        assert!(is_reloc);
        assert!(is_unrot);
    }

    #[test]
    fn test_rope_on_read_flags_nonspan_blocks_clear() {
        // Unannotated + Prefixed blocks are stored rotated → both flags
        // false (kernel rotation branch never taken for them).
        let mut ann = BTreeMap::new();
        ann.insert(1, BlockKind::Prefixed { first_token: 0 });
        let (r0, u0) = compute_block_flags_rope_on_read(&ann, 0); // unannotated
        assert!(!r0 && !u0);
        let (r1, u1) = compute_block_flags_rope_on_read(&ann, 1); // Prefixed
        assert!(!r1 && !u1);
    }

    #[test]
    fn test_rope_on_read_does_not_change_cuda_fn() {
        // Guard: the CUDA-consumed compute_block_flags still returns
        // is_unrotated=false for a relocatable block (rotate-on-write).
        let mut ann = BTreeMap::new();
        ann.insert(0, BlockKind::Relocatable { first_token: 0 });
        let (_, cuda_unrot) = compute_block_flags(&ann, 0, 4, 8, 0);
        assert!(!cuda_unrot);
        let (_, metal_unrot) = compute_block_flags_rope_on_read(&ann, 0);
        assert!(metal_unrot); // metal path differs by design
    }
}
