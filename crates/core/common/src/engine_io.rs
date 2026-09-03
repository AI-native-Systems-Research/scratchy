// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Engine-core I/O types, ported from `vllm/v1/engine/__init__.py`.
//!
//! These types form the wire protocol between the engine front-end
//! (API server) and the engine core (scheduler + workers).

use serde::{Deserialize, Serialize};

use crate::multimodal::MultimodalData;
use crate::sampling::{LogprobsOutput, SamplingParams};

// ---------------------------------------------------------------------------
// FinishReason
// ---------------------------------------------------------------------------

/// Reason a request finished.
///
/// Int-valued for compact serialization over IPC / ZMQ.
///
/// * `Stop`   -- a stop string or stop token was emitted.
/// * `Length` -- `max_tokens` was consumed or `max_model_len` was reached.
/// * `Abort`  -- aborted by the client.
/// * `Error`  -- a retryable request-level internal error (e.g. KV load
///   failure). Always surfaced as HTTP 500.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum FinishReason {
    Stop = 0,
    Length = 1,
    Abort = 2,
    Error = 3,
}

/// The canonical string representations, matching the OpenAI API.
const FINISH_REASON_STRINGS: [&str; 4] = ["stop", "length", "abort", "error"];

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(FINISH_REASON_STRINGS[*self as usize])
    }
}

// ---------------------------------------------------------------------------
// EngineCoreEventType / EngineCoreEvent
// ---------------------------------------------------------------------------

/// The type of a timestamped engine-core event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum EngineCoreEventType {
    Queued = 1,
    Scheduled = 2,
    Preempted = 3,
}

/// A timestamped engine-core event associated with a request.
///
/// The timestamp is monotonic (within the engine-core process) and is used
/// by the engine front-end to compute inter-event intervals. It should *not*
/// be compared with timestamps from other processes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EngineCoreEvent {
    /// What happened.
    pub event_type: EngineCoreEventType,
    /// Monotonic timestamp (seconds).
    pub timestamp: f64,
}

impl EngineCoreEvent {
    /// Convenience constructor.
    pub fn new(event_type: EngineCoreEventType, timestamp: f64) -> Self {
        Self {
            event_type,
            timestamp,
        }
    }
}

// ---------------------------------------------------------------------------
// StopReason
// ---------------------------------------------------------------------------

/// The reason generation stopped, which may be either a token ID or a string.
///
/// Mirrors the Python `stop_reason: int | str | None` field.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StopReason {
    /// Stopped because a specific token ID was emitted.
    Token(u32),
    /// Stopped because a specific string was emitted.
    String(String),
}

// ---------------------------------------------------------------------------
// EngineCoreRequest
// ---------------------------------------------------------------------------

/// A request submitted to the engine core by the front-end.
///
/// Mirrors the Python `EngineCoreRequest` msgspec struct.
/// Fields that depend on Python-only types (tensors, LoRA, multimodal
/// features, pooling params) are omitted in this initial Rust port and
/// will be added as those subsystems are brought up.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreRequest {
    /// Unique request identifier.
    pub request_id: String,

    /// The tokenized prompt. `None` when prompt embeddings are provided
    /// instead (multimodal / embedding models).
    pub prompt_token_ids: Option<Vec<u32>>,

    /// Sampling parameters (for generative models).
    pub sampling_params: Option<SamplingParams>,

    /// Wall-clock arrival time (seconds since epoch).
    pub arrival_time: f64,

    /// Index of the front-end client, used to route outputs back to the
    /// correct client when the front-end is scaled out.
    pub client_index: u32,

    /// Scheduling priority (lower = higher priority).
    pub priority: i32,

    /// Optional cache salt for prefix-cache isolation.
    pub cache_salt: Option<String>,

    /// In data-parallel mode, the rank this request should be sent to.
    pub data_parallel_rank: Option<u32>,

    /// Whether this is a pooling (embedding) request rather than generation.
    /// Pooling requests are finished after one forward pass (no decode loop).
    #[serde(default)]
    pub is_pooling: bool,

    /// Multimodal data (images) for vision-language models.
    /// Only present for requests with image content. Skipped during
    /// serialization since images are passed in-process only.
    #[serde(skip)]
    pub mm_data: Option<MultimodalData>,

    /// Sparse map of block index → [`BlockKind`] for span-aware block hashing.
    /// Produced by `/v1/query/execute`; `None` for normal requests.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub block_annotations: Option<crate::request::BlockAnnotations>,

    /// 🦭 Pad and cache the final partial block on completion.
    #[serde(default)]
    pub seal: bool,

    /// Deprioritize cached blocks for eviction after completion.
    #[serde(default)]
    pub volatile: bool,
}

// ---------------------------------------------------------------------------
// EngineCoreOutput
// ---------------------------------------------------------------------------

/// Per-request output emitted by the engine core after each scheduler step.
///
/// Embedding data: single vector (standard pooling) or multi-vector (ColBERT).
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingData {
    /// Single embedding vector (Last/Cls/Mean pooling).
    Single(Vec<f32>),
    /// Per-token embedding vectors (AllTokens / ColBERT).
    Multi(Vec<Vec<f32>>),
}

/// Mirrors the Python `EngineCoreOutput` msgspec struct. Fields that depend
/// on tensor types (logprobs, pooling output, routed experts) are omitted
/// in this initial port.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreOutput {
    /// The request this output belongs to.
    pub request_id: String,

    /// Newly generated token IDs since the last output.
    pub new_token_ids: Vec<u32>,

    /// Set when the request has finished; `None` while still running.
    pub finish_reason: Option<FinishReason>,

    /// Why generation stopped (token ID or string), if applicable.
    pub stop_reason: Option<StopReason>,

    /// Number of prompt tokens served from prefix cache (local + external).
    pub num_cached_tokens: u32,

    /// Events (queued, scheduled, preempted) recorded during this step.
    pub events: Option<Vec<EngineCoreEvent>>,

    /// Per-token log-probabilities for the newly generated tokens.
    /// Only populated when the request's `SamplingParams.logprobs` is set.
    /// Skipped during serialization (only used in-process).
    #[serde(skip)]
    pub new_logprobs: Option<Vec<LogprobsOutput>>,

    /// Per-position log-probabilities for the prompt tokens.
    /// Only populated on the prefill step when `SamplingParams.prompt_logprobs`
    /// is set. Position 0 is `None` (no prior context); positions 1..n have
    /// `Some(LogprobsOutput)` where `.sampled` is the actual prompt token.
    /// Skipped during serialization (only used in-process).
    #[serde(skip)]
    pub new_prompt_logprobs: Option<Vec<Option<LogprobsOutput>>>,

    /// Pooling output (embedding) for this request.
    /// Only populated when the engine is in pooling mode.
    /// Skipped during serialization (only used in-process).
    #[serde(skip)]
    pub pooler_output: Option<EmbeddingData>,
}

impl EngineCoreOutput {
    /// Whether the request has finished.
    pub fn finished(&self) -> bool {
        self.finish_reason.is_some()
    }
}

// ---------------------------------------------------------------------------
// SchedulerStats
// ---------------------------------------------------------------------------

/// Scheduler statistics piggybacked on engine-core outputs so the serving
/// layer can update Prometheus gauges without direct access to the engine.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SchedulerStats {
    /// Number of requests currently running.
    pub num_running_reqs: usize,
    /// Number of requests waiting to be scheduled.
    pub num_waiting_reqs: usize,
    /// KV cache usage as a fraction in `[0.0, 1.0]`.
    pub kv_cache_usage: f64,
    /// Number of GPU KV cache blocks currently in use.
    pub gpu_cache_blocks_used: usize,
    /// Total number of GPU KV cache blocks.
    pub gpu_cache_blocks_total: usize,
    /// Number of blocks retained in the prefix cache.
    pub num_cached_blocks: usize,
    /// Speculative decoding metrics (None when spec decode is disabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_decode_stats: Option<SpecDecodingStats>,
}

/// Speculative decoding metrics, matching Python's `SpecDecodingStats`.
///
/// Tracks acceptance rates and draft token counts for monitoring spec decode
/// performance. Updated after each engine step.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SpecDecodingStats {
    /// Number of requests that had draft tokens this step.
    pub num_drafts: usize,
    /// Total draft tokens proposed across all requests.
    pub num_draft_tokens: usize,
    /// Total draft tokens accepted across all requests.
    pub num_accepted_tokens: usize,
}

// ---------------------------------------------------------------------------
// EngineCoreOutputs
// ---------------------------------------------------------------------------

/// A batch of outputs from one engine-core step, possibly spanning multiple
/// requests.
///
/// Mirrors the Python `EngineCoreOutputs` msgspec struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreOutputs {
    /// Index of the engine that produced these outputs (for multi-engine /
    /// data-parallel setups).
    pub engine_index: u32,

    /// Per-request outputs.
    pub outputs: Vec<EngineCoreOutput>,

    /// Monotonic timestamp of when the step completed.
    pub timestamp: f64,

    /// Scheduler statistics from the most recent step.
    #[serde(skip)]
    pub scheduler_stats: Option<SchedulerStats>,
}

impl Default for EngineCoreOutputs {
    fn default() -> Self {
        Self {
            engine_index: 0,
            outputs: Vec::new(),
            timestamp: 0.0,
            scheduler_stats: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- FinishReason tests --

    #[test]
    fn test_finish_reason_display() {
        assert_eq!(format!("{}", FinishReason::Stop), "stop");
        assert_eq!(format!("{}", FinishReason::Length), "length");
        assert_eq!(format!("{}", FinishReason::Abort), "abort");
        assert_eq!(format!("{}", FinishReason::Error), "error");
    }

    #[test]
    fn test_finish_reason_repr() {
        assert_eq!(FinishReason::Stop as u8, 0);
        assert_eq!(FinishReason::Length as u8, 1);
        assert_eq!(FinishReason::Abort as u8, 2);
        assert_eq!(FinishReason::Error as u8, 3);
    }

    #[test]
    fn test_finish_reason_serde_roundtrip() {
        for reason in [
            FinishReason::Stop,
            FinishReason::Length,
            FinishReason::Abort,
            FinishReason::Error,
        ] {
            let json = serde_json::to_string(&reason).unwrap();
            let back: FinishReason = serde_json::from_str(&json).unwrap();
            assert_eq!(reason, back);
        }
    }

    // -- EngineCoreEventType / EngineCoreEvent tests --

    #[test]
    fn test_event_type_repr() {
        assert_eq!(EngineCoreEventType::Queued as u8, 1);
        assert_eq!(EngineCoreEventType::Scheduled as u8, 2);
        assert_eq!(EngineCoreEventType::Preempted as u8, 3);
    }

    #[test]
    fn test_event_new() {
        let evt = EngineCoreEvent::new(EngineCoreEventType::Scheduled, 42.5);
        assert_eq!(evt.event_type, EngineCoreEventType::Scheduled);
        assert_eq!(evt.timestamp, 42.5);
    }

    #[test]
    fn test_event_serde_roundtrip() {
        let evt = EngineCoreEvent::new(EngineCoreEventType::Preempted, 100.0);
        let json = serde_json::to_string(&evt).unwrap();
        let evt2: EngineCoreEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(evt, evt2);
    }

    // -- StopReason tests --

    #[test]
    fn test_stop_reason_token() {
        let sr = StopReason::Token(50256);
        assert_eq!(sr, StopReason::Token(50256));
    }

    #[test]
    fn test_stop_reason_string() {
        let sr = StopReason::String("</s>".into());
        assert_eq!(sr, StopReason::String("</s>".into()));
    }

    #[test]
    fn test_stop_reason_serde_roundtrip() {
        let cases = vec![StopReason::Token(123), StopReason::String("END".into())];
        for sr in cases {
            let json = serde_json::to_string(&sr).unwrap();
            let sr2: StopReason = serde_json::from_str(&json).unwrap();
            assert_eq!(sr, sr2);
        }
    }

    // -- EngineCoreRequest tests --

    #[test]
    fn test_engine_core_request_minimal() {
        let req = EngineCoreRequest {
            request_id: "req-1".into(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            sampling_params: Some(SamplingParams::default()),
            arrival_time: 1000.0,
            client_index: 0,
            priority: 0,
            cache_salt: None,
            data_parallel_rank: None,
            is_pooling: false,
            mm_data: None,
            block_annotations: None,
            seal: false,
            volatile: false,
        };
        assert_eq!(req.request_id, "req-1");
        assert_eq!(req.prompt_token_ids.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_engine_core_request_serde_roundtrip() {
        let req = EngineCoreRequest {
            request_id: "req-2".into(),
            prompt_token_ids: Some(vec![10, 20]),
            sampling_params: Some(SamplingParams {
                temperature: 0.5,
                max_tokens: Some(128),
                ..Default::default()
            }),
            arrival_time: 999.0,
            client_index: 2,
            priority: 5,
            cache_salt: Some("salt-abc".into()),
            data_parallel_rank: Some(1),
            is_pooling: false,
            mm_data: None,
            block_annotations: None,
            seal: false,
            volatile: false,
        };
        let json = serde_json::to_string(&req).unwrap();
        let req2: EngineCoreRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req2.request_id, "req-2");
        assert_eq!(req2.client_index, 2);
        assert_eq!(req2.priority, 5);
        assert_eq!(req2.cache_salt.as_deref(), Some("salt-abc"));
        assert_eq!(req2.data_parallel_rank, Some(1));
    }

    // -- EngineCoreOutput tests --

    #[test]
    fn test_engine_core_output_not_finished() {
        let out = EngineCoreOutput {
            request_id: "r1".into(),
            new_token_ids: vec![42, 43],
            finish_reason: None,
            stop_reason: None,
            num_cached_tokens: 10,
            events: None,
            new_logprobs: None,
            new_prompt_logprobs: None,
            pooler_output: None,
        };
        assert!(!out.finished());
    }

    #[test]
    fn test_engine_core_output_finished() {
        let out = EngineCoreOutput {
            request_id: "r1".into(),
            new_token_ids: vec![42],
            finish_reason: Some(FinishReason::Stop),
            stop_reason: Some(StopReason::Token(50256)),
            num_cached_tokens: 0,
            events: Some(vec![
                EngineCoreEvent::new(EngineCoreEventType::Queued, 1.0),
                EngineCoreEvent::new(EngineCoreEventType::Scheduled, 2.0),
            ]),
            new_logprobs: None,
            new_prompt_logprobs: None,
            pooler_output: None,
        };
        assert!(out.finished());
        assert_eq!(out.finish_reason, Some(FinishReason::Stop));
        assert_eq!(out.events.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_engine_core_output_serde_roundtrip() {
        let out = EngineCoreOutput {
            request_id: "r1".into(),
            new_token_ids: vec![1, 2, 3],
            finish_reason: Some(FinishReason::Length),
            stop_reason: None,
            num_cached_tokens: 5,
            events: None,
            new_logprobs: None,
            new_prompt_logprobs: None,
            pooler_output: None,
        };
        let json = serde_json::to_string(&out).unwrap();
        let out2: EngineCoreOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(out2.request_id, "r1");
        assert_eq!(out2.new_token_ids, vec![1, 2, 3]);
        assert_eq!(out2.finish_reason, Some(FinishReason::Length));
        assert_eq!(out2.num_cached_tokens, 5);
    }

    // -- EngineCoreOutputs tests --

    #[test]
    fn test_engine_core_outputs_default() {
        let outs = EngineCoreOutputs::default();
        assert_eq!(outs.engine_index, 0);
        assert!(outs.outputs.is_empty());
        assert_eq!(outs.timestamp, 0.0);
        assert!(outs.scheduler_stats.is_none());
    }

    #[test]
    fn test_engine_core_outputs_with_data() {
        let outs = EngineCoreOutputs {
            engine_index: 2,
            outputs: vec![
                EngineCoreOutput {
                    request_id: "a".into(),
                    new_token_ids: vec![10],
                    finish_reason: None,
                    stop_reason: None,
                    num_cached_tokens: 0,
                    events: None,
                    new_logprobs: None,
                    new_prompt_logprobs: None,
                    pooler_output: None,
                },
                EngineCoreOutput {
                    request_id: "b".into(),
                    new_token_ids: vec![20],
                    finish_reason: Some(FinishReason::Stop),
                    stop_reason: Some(StopReason::String("END".into())),
                    num_cached_tokens: 3,
                    events: None,
                    new_logprobs: None,
                    new_prompt_logprobs: None,
                    pooler_output: None,
                },
            ],
            timestamp: 1234.5,
            scheduler_stats: None,
        };
        assert_eq!(outs.outputs.len(), 2);
        assert_eq!(outs.engine_index, 2);
        assert_eq!(outs.timestamp, 1234.5);
    }

    // -- SchedulerStats tests --

    #[test]
    fn test_scheduler_stats_default() {
        let stats = SchedulerStats::default();
        assert_eq!(stats.num_running_reqs, 0);
        assert_eq!(stats.num_waiting_reqs, 0);
        assert!((stats.kv_cache_usage - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scheduler_stats_construction() {
        let stats = SchedulerStats {
            num_running_reqs: 5,
            num_waiting_reqs: 3,
            kv_cache_usage: 0.42,
            gpu_cache_blocks_used: 100,
            gpu_cache_blocks_total: 1024,
            num_cached_blocks: 0,
            spec_decode_stats: None,
        };
        assert_eq!(stats.num_running_reqs, 5);
        assert_eq!(stats.num_waiting_reqs, 3);
        assert!((stats.kv_cache_usage - 0.42).abs() < f64::EPSILON);
    }

    #[test]
    fn test_engine_core_outputs_serde_roundtrip() {
        let outs = EngineCoreOutputs {
            engine_index: 1,
            outputs: vec![EngineCoreOutput {
                request_id: "r1".into(),
                new_token_ids: vec![7, 8, 9],
                finish_reason: None,
                stop_reason: None,
                num_cached_tokens: 0,
                events: Some(vec![EngineCoreEvent::new(EngineCoreEventType::Queued, 0.5)]),
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: None,
            }],
            timestamp: 42.0,
            scheduler_stats: None,
        };
        let json = serde_json::to_string(&outs).unwrap();
        let outs2: EngineCoreOutputs = serde_json::from_str(&json).unwrap();
        assert_eq!(outs2.engine_index, 1);
        assert_eq!(outs2.outputs.len(), 1);
        assert_eq!(outs2.outputs[0].new_token_ids, vec![7, 8, 9]);
        assert_eq!(outs2.timestamp, 42.0);
    }

    // -- EmbeddingData tests --

    #[test]
    fn test_embedding_data_single() {
        let data = EmbeddingData::Single(vec![0.1, 0.2, 0.3]);
        match &data {
            EmbeddingData::Single(v) => assert_eq!(v.len(), 3),
            EmbeddingData::Multi(_) => panic!("expected Single"),
        }
    }

    #[test]
    fn test_embedding_data_multi() {
        let data = EmbeddingData::Multi(vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]]);
        match &data {
            EmbeddingData::Multi(rows) => {
                assert_eq!(rows.len(), 3);
                assert_eq!(rows[0], vec![0.1, 0.2]);
            }
            EmbeddingData::Single(_) => panic!("expected Multi"),
        }
    }

    #[test]
    fn test_embedding_data_equality() {
        let a = EmbeddingData::Single(vec![1.0, 2.0]);
        let b = EmbeddingData::Single(vec![1.0, 2.0]);
        let c = EmbeddingData::Multi(vec![vec![1.0, 2.0]]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
