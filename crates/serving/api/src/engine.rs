// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Async engine interface for the serving layer.
//!
//! Provides the bridge between the HTTP server and the engine core client.
//! Handles request lifecycle: validate → tokenize → submit → poll → detokenize → respond.
//!
//! Port of: `vllm/v1/engine/async_llm.py` (subset)

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Marks a client's prompt-cache breakpoint in a chat message's text (the Anthropic path's
/// `cache_control`): taken out of the rendered prompt before it is tokenized, and turned into the
/// prompt length that reaches it (`EngineCoreRequest::cache_breakpoints`). A private-use code point,
/// so no client text means it.
pub(crate) const CACHE_BREAKPOINT: char = '\u{F8FF}';

/// `text` without its [`CACHE_BREAKPOINT`] marks, and the byte offset of each in what is left.
fn take_cache_breakpoints(text: String) -> (String, Vec<usize>) {
    if !text.contains(CACHE_BREAKPOINT) {
        return (text, Vec::new());
    }
    let mut kept = String::with_capacity(text.len());
    let mut marks = Vec::new();
    for c in text.chars() {
        match c {
            CACHE_BREAKPOINT => marks.push(kept.len()),
            c => kept.push(c),
        }
    }
    (kept, marks)
}
use std::panic::AssertUnwindSafe;
use std::task::Poll;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rayon::prelude::*;
use scratchy_core_common::engine_io::EmbeddingData;
#[cfg(feature = "multimodal")]
use scratchy_core_common::multimodal::{ImageData, MultimodalData};
use scratchy_core_common::sampling::GuidedGrammar;
use scratchy_core_common::{
    EngineCoreOutput, EngineCoreRequest, FinishReason, SamplingParams, StopReason,
};
use scratchy_serving_engine::core_client::EngineCoreClient;
use scratchy_serving_engine::engine_core::StepOutcome;
use scratchy_serving_engine::executor::{Executor, ModelRunnerOutput};
use tokio::sync::{Mutex, Notify, mpsc, oneshot, watch};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::chat_template::ChatTemplate;
use crate::detokenizer::IncrementalDetokenizer;
use crate::error::{ServeError, ServeResult};
use crate::protocol;
use crate::reasoning_parser::{ReasoningDelta, ReasoningParser, StreamingReasoningParserState};
use crate::tokenizer::Tokenizer;
use crate::tool_parser::{
    DeltaToolCall, StreamingToolParserState, ToolCallParser, ToolParserDelta,
};

/// Upper bound on `n` (completions per request), matching the OpenAI API's own limit. Without
/// this, a client-supplied `n` feeds `Vec::with_capacity(n)`/`Vec::with_capacity(prompts*n)`
/// directly — an attacker-controlled allocation size (CWE-789).
const MAX_N: u32 = 128;

fn validate_n(n: u32) -> ServeResult<usize> {
    if n > MAX_N {
        return Err(ServeError::Validation(format!(
            "n={n} exceeds the maximum of {MAX_N}"
        )));
    }
    Ok(n.max(1) as usize)
}

// ---------------------------------------------------------------------------
// Parallel detokenization type aliases
// ---------------------------------------------------------------------------

/// A REQUEST'S DETOKENIZER, SHARED RATHER THAN MOVED.
///
/// ⛔ It used to be MOVED OUT of the request state for the parallel phase and put back afterwards,
/// across a RELEASED LOCK — so for the whole of that phase every request in the batch was publicly
/// reachable with `detokenizer == None`, and any waiter that took the lock in that window with
/// `finish_reason` already set removed the state and fell back to `placeholder_text`. That is the
/// `<token_37769><token_3919>…` output: the token ids were always CORRECT, only the text was lost.
/// Worse, if the state was already gone, phase 3's put-back found nothing and DROPPED the
/// detokenizer for good, so that request could never produce text again.
///
/// Shared behind an `Arc`, there is no window and no put-back: phase 1 clones the handle, phase 2
/// mutates through it, and nothing can empty the field. `None` now means exactly one thing — the
/// engine has no tokenizer, a static property fixed at load — so `placeholder_text` is unreachable
/// for any request submitted with one. That is the guard: "detokenizer temporarily absent" stops
/// being a representable state rather than being checked for.
///
/// `std::sync::Mutex`, not the `tokio` one, because phase 2 runs on rayon and cannot await. It is
/// uncontended by construction — one work item per request, and only phase 2 mutates.
type SharedDetok = Arc<std::sync::Mutex<IncrementalDetokenizer>>;

/// Phase 1 work item: (new_token_ids, stop_terminated, finish_reason, detokenizer).
type DetokWork = (Vec<u32>, bool, Option<FinishReason>, SharedDetok);
/// Phase 2 result: (detokenizer_stop_string, delta_text). The detokenizer itself never travels.
type DetokResult = (Option<String>, Option<String>);

// ---------------------------------------------------------------------------
// ControlMessage — commands sent to the step loop from HTTP handlers.
// ---------------------------------------------------------------------------

enum ControlMessage {
    /// Reset the prefix cache. Returns `true` if successful.
    ResetPrefixCache(oneshot::Sender<bool>),
    /// Put the engine to sleep (free GPU memory).
    Sleep(u32, oneshot::Sender<Result<(), String>>),
    /// Wake the engine from sleep.
    WakeUp(Option<Vec<String>>, oneshot::Sender<Result<(), String>>),
    /// Query whether the engine is sleeping.
    IsSleeping(oneshot::Sender<bool>),
    /// Abort specific requests on the scheduler so they stop generating and
    /// free their sequence slots (the API-map removal alone leaves a zombie).
    Abort(Vec<String>),
}

/// Bundled arguments for `step_loop_sync` / `step_loop_async`.
struct StepLoopArgs {
    client: Box<dyn EngineCoreClient + Send>,
    request_rx: mpsc::UnboundedReceiver<EngineCoreRequest>,
    embed_rx: mpsc::UnboundedReceiver<EmbedRequest>,
    control_rx: mpsc::UnboundedReceiver<ControlMessage>,
    requests: Arc<Mutex<HashMap<String, RequestState>>>,
    notify: Arc<Notify>,
    no_progress_timeout: Duration,
}

/// Where the background step loop is in its life.
#[derive(Clone, Debug)]
enum StepLoop {
    /// Not spawned yet.
    Pending,
    Running,
    /// Gone for good, and every request it was serving has been failed.
    Ended(StepLoopEnd),
}

/// How the step loop ended. While anything holds the engine its channels stay open, so either
/// way the engine is dead: nothing will answer a request again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepLoopEnd {
    /// It returned: its channels closed, or its executor thread exited.
    Exited,
    /// It panicked, with this message.
    Panicked(String),
}

impl std::fmt::Display for StepLoopEnd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exited => f.write_str("exited"),
            Self::Panicked(message) => write!(f, "panicked: {message}"),
        }
    }
}

/// What a caught panic said: its payload, when that is the usual string.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|message| message.to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a payload that is not a string".to_string())
}

// ---------------------------------------------------------------------------
// RequestState
// ---------------------------------------------------------------------------

/// Tracks the state of an in-flight request.
struct RequestState {
    /// Accumulated generated token IDs.
    generated_token_ids: Vec<u32>,

    /// Number of prompt tokens.
    num_prompt_tokens: u32,

    /// Number of cached prompt tokens.
    num_cached_tokens: u32,

    /// Finish reason, if completed.
    finish_reason: Option<FinishReason>,

    /// Error message if the request was aborted due to an engine error.
    error: Option<String>,

    /// Stop reason, if applicable.
    stop_reason: Option<StopReason>,

    /// Channel to send streaming output deltas.
    stream_tx: Option<mpsc::UnboundedSender<StreamDelta>>,

    /// Incremental detokenizer for this request. `None` IFF the engine has no tokenizer — see
    /// [`SharedDetok`] for why it is shared rather than moved in and out.
    detokenizer: Option<SharedDetok>,

    /// The response choice it fills, and the prompt it answers (for n>1 / multi-prompt support).
    choice: Choice,

    /// When this request was submitted (for TTFT calculation).
    submit_time: Instant,
    /// When the first output token was received (None until first token).
    first_token_time: Option<Instant>,
    /// When the last output token was received (for ITL calculation).
    last_token_time: Option<Instant>,
    /// Number of inter-token intervals observed (for computing avg ITL).
    itl_count: u32,
    /// Sum of inter-token latencies in seconds (for computing avg ITL).
    itl_sum: f64,
    /// Per-token inter-token latencies in milliseconds (newest last) for the
    /// live ITL distribution — the whole current generation (bounded only by
    /// max_tokens), so a UI can accumulate a session-wide histogram.
    #[cfg(feature = "live-stats")]
    itl_samples: Vec<f32>,

    /// Accumulated per-token log-probabilities (if requested).
    logprobs: Vec<scratchy_core_common::LogprobsOutput>,

    /// Per-prompt-token log-probabilities (if requested). Set once during prefill.
    /// Position 0 is None (no prior context); positions 1..n are Some.
    prompt_logprobs: Option<Vec<Option<scratchy_core_common::LogprobsOutput>>>,

    /// Streaming tool parser state (if tool parsing is active for this request).
    tool_parser_state: Option<Box<dyn StreamingToolParserState + Send>>,

    /// Streaming reasoning parser state (if reasoning parsing is active).
    reasoning_parser_state: Option<Box<dyn StreamingReasoningParserState + Send>>,

    /// Whether reasoning has ended (</think> seen) for this request.
    reasoning_ended: bool,

    /// Whether to include reasoning content in the response.
    include_reasoning: bool,

    /// Accumulated generated text so far (for streaming tool parsing).
    accumulated_text: String,

    /// The generated text past the reasoning so far: what the streaming tool parser reads when a
    /// reasoning parser routes text first.
    accumulated_content: String,

    /// Accumulated generated token IDs (for streaming reasoning parsing).
    accumulated_token_ids: Vec<u32>,

    /// Whether any tool call deltas have been emitted (for setting finish_reason).
    tool_calls_emitted: bool,

    /// The tool calls streamed so far: each call named first, and only the forced tool's.
    tool_calls: ToolCallStream,

    /// Pooling output (embedding data), set when the engine is in pooling mode.
    pooler_output: Option<EmbeddingData>,
}

impl RequestState {
    /// The token counts every delta sent for this request carries, as of now.
    fn stream_usage(&self) -> StreamUsage {
        StreamUsage {
            prompt: self.choice.prompt,
            prompt_tokens: self.num_prompt_tokens,
            cached_tokens: self.num_cached_tokens,
            completion_tokens: self.generated_token_ids.len() as u32,
        }
    }
}

/// Where one engine request's output goes in its response: the choice it fills, and the prompt it
/// answers. A completion request runs `n` choices of each of its prompts, a chat request `n` of
/// its one.
#[derive(Debug, Clone, Copy, Default)]
struct Choice {
    index: u32,
    prompt: PromptIndex,
}

/// Which of a request's prompts a choice answers: its position in a completion request's prompt
/// list (0 for chat, which has one). Its own type, so it cannot be swapped for a choice index.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct PromptIndex(pub u32);

/// Token counts for one request, carried on every [`StreamDelta`].
///
/// A streaming response has to report usage from the delta stream — it never
/// sees a `UsageInfo`. `cached_tokens` is the prefix-cache hit, which the
/// scheduler fixes at admission (before any token is emitted), so the first
/// delta already carries the final value; a consumer should still take the last
/// value it sees rather than assume which delta arrives first.
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamUsage {
    /// Which of the request's prompts this choice answers. The choices of one prompt share its
    /// tokens and its cache hit, so a response counts each once per prompt, not once per choice.
    pub prompt: PromptIndex,
    /// Total prompt tokens for the request.
    pub prompt_tokens: u32,
    /// Prompt tokens served from the prefix cache. Block-aligned, so it
    /// slightly understates logical reuse and never overstates it.
    pub cached_tokens: u32,
    /// Tokens generated so far for this choice, this step's included — the
    /// count the `request finished` log reports. Cumulative, so the last delta
    /// a consumer sees carries the total. Not the sum of the deltas'
    /// `new_token_ids`: a step whose text a tool parser is still buffering
    /// sends no delta at all, and its tokens reach no consumer any other way.
    pub completion_tokens: u32,
}

/// A delta sent to a streaming response.
#[derive(Debug, Clone)]
pub struct StreamDelta {
    /// Choice index in the response (for n>1 support).
    pub index: u32,
    /// New token IDs generated in this step.
    pub new_token_ids: Vec<u32>,
    /// Decoded text for this step (None if no tokenizer).
    pub text: Option<String>,
    /// Finish reason, if the request completed in this step.
    pub finish_reason: Option<FinishReason>,
    /// Stop reason, if applicable.
    pub stop_reason: Option<StopReason>,
    /// Per-token log-probabilities for this step (if requested).
    pub logprobs: Option<Vec<scratchy_core_common::LogprobsOutput>>,
    /// Tool call deltas for streaming tool parsing.
    pub tool_call_deltas: Option<Vec<DeltaToolCall>>,
    /// Reasoning content delta (for streaming reasoning parsing).
    pub reasoning: Option<String>,
    /// Token counts, repeated on every delta — see [`StreamUsage`].
    pub usage: StreamUsage,
}

// ---------------------------------------------------------------------------
// AsyncEngine
// ---------------------------------------------------------------------------

/// An embedding request sent to the step loop.
struct EmbedRequest {
    token_id_seqs: Vec<Vec<u32>>,
    reply: tokio::sync::oneshot::Sender<ServeResult<Vec<Vec<f32>>>>,
}

/// Maximum time the step loop can run with pending requests but no output
/// tokens before aborting all requests. Acts as a safety net for scheduler
/// bugs where requests get stuck without making forward progress.
const DEFAULT_NO_PROGRESS_TIMEOUT: Duration = Duration::from_secs(60);

/// An async wrapper around the engine core client, providing request lifecycle
/// management for the HTTP serving layer.
///
/// Architecture: the engine client lives exclusively in the background step
/// loop (spawned by `spawn_step_loop`). HTTP handlers submit requests via an
/// `mpsc` channel and wait on a `Notify` for results. This eliminates mutex
/// contention between the slow model forward pass and fast request submission.
///
/// This is a simplified version of Python's `AsyncLLM`. It manages:
/// Cheap snapshot of the active request's live timing/counters, for UIs that
/// want to peek under the hood ([`AsyncEngine::live_stats`]).
#[cfg(feature = "live-stats")]
#[derive(Clone, Copy, Debug, Default)]
pub struct LiveStats {
    /// False = prefilling (no token yet), true = decoding.
    pub generating: bool,
    /// Prompt tokens (current context size).
    pub prompt_tokens: u32,
    /// Prompt tokens served from the prefix cache (reuse).
    pub cached_tokens: u32,
    /// Tokens generated so far for this request.
    pub generated_tokens: u32,
    /// Time-to-first-token in seconds (`None` while still prefilling).
    pub ttft_s: Option<f64>,
    /// Decode speed in tokens/sec (`1 / mean inter-token latency`); 0 if <2 tokens.
    pub decode_tok_s: f64,
}

/// - Request submission and tracking
/// - Tokenization and detokenization (when a tokenizer is provided)
/// - Polling the engine for outputs
/// - Routing outputs to the correct response (streaming or non-streaming)
pub struct AsyncEngine {
    /// In-flight request states. Only held briefly for HashMap reads/writes,
    /// never during model execution.
    requests: Arc<Mutex<HashMap<String, RequestState>>>,
    /// Channel to send new `EngineCoreRequest`s to the step loop.
    request_tx: mpsc::UnboundedSender<EngineCoreRequest>,
    /// Channel to send embedding requests to the step loop.
    embed_tx: mpsc::UnboundedSender<EmbedRequest>,
    /// Channel to send control messages (e.g. reset prefix cache) to the step loop.
    control_tx: mpsc::UnboundedSender<ControlMessage>,
    /// The engine client + channel receivers, held until `spawn_step_loop`
    /// moves them into the background task. `None` after the loop starts.
    #[allow(clippy::type_complexity)]
    pending_loop: std::sync::Mutex<
        Option<(
            Box<dyn EngineCoreClient + Send>,
            mpsc::UnboundedReceiver<EngineCoreRequest>,
            mpsc::UnboundedReceiver<EmbedRequest>,
            mpsc::UnboundedReceiver<ControlMessage>,
        )>,
    >,
    model_name: String,
    max_model_len: usize,
    /// Notified after every engine step so request handlers can check results.
    notify: Arc<Notify>,
    /// Optional tokenizer for encoding prompts and decoding outputs.
    tokenizer: Option<Arc<Tokenizer>>,
    /// Optional chat template for formatting chat messages.
    chat_template: Option<Arc<ChatTemplate>>,
    /// Optional tool call parser for extracting structured tool calls from output.
    tool_parser: Option<Arc<dyn ToolCallParser>>,
    /// Optional reasoning parser for extracting <think>...</think> blocks.
    reasoning_parser: Option<Arc<dyn ReasoningParser>>,
    /// Default extra kwargs for chat template rendering (e.g. `enable_thinking`).
    default_chat_template_kwargs: Option<std::collections::HashMap<String, serde_json::Value>>,
    /// Model-recommended sampling defaults from `generation_config.json`
    /// (Python vLLM `generation_config="auto"` parity). Request fields
    /// left unset fall back to these before the neutral defaults.
    generation_defaults: scratchy_core_common::sampling::GenerationDefaults,
    /// Whether async scheduling is enabled (overlap GPU execution with CPU scheduling).
    async_scheduling: bool,
    /// Where the step loop is. `poll_until_done` reads it to see a dead loop, and the server
    /// watches it to shut down when the loop ends, since nothing can then be served.
    step_loop: Arc<watch::Sender<StepLoop>>,
    /// Resolved multimodal processor handle. `None` for text-only.
    /// Built at startup from `scratchy_vision::MmMetadata` declared by
    /// the per-arch crate; carries the placeholder token id, size
    /// policy, tokens-per-image policy, preprocess fn pointer, and
    /// runtime-resolved knobs (smart-resize bounds, image_size,
    /// patch_size). The engine dispatches generically through this
    /// handle — no arch names appear here.
    #[cfg(feature = "multimodal")]
    mm_processor: Option<std::sync::Arc<crate::multimodal::ResolvedMmProcessor>>,
    /// Whether the engine is in pooling mode (embedding requests go through scheduler).
    is_pooling: bool,
    /// Maximum time the step loop can have pending requests with no output
    /// before aborting them. Configurable for tests (default: 60s).
    no_progress_timeout: Duration,
}

impl AsyncEngine {
    /// Create a new `AsyncEngine` wrapping the given client.
    ///
    /// **Important**: call `spawn_step_loop` to start the background engine
    /// loop. Without it, submitted requests will never be processed.
    pub fn new(
        client: Box<dyn EngineCoreClient + Send>,
        model_name: String,
        max_model_len: usize,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let (embed_tx, embed_rx) = mpsc::unbounded_channel();
        let (control_tx, control_rx) = mpsc::unbounded_channel();

        Self {
            requests: Arc::new(Mutex::new(HashMap::new())),
            request_tx: tx,
            embed_tx,
            control_tx,
            pending_loop: std::sync::Mutex::new(Some((client, rx, embed_rx, control_rx))),
            model_name,
            max_model_len,
            notify: Arc::new(Notify::new()),
            tokenizer: None,
            chat_template: None,
            tool_parser: None,
            reasoning_parser: None,
            default_chat_template_kwargs: None,
            generation_defaults: Default::default(),
            async_scheduling: false,
            step_loop: Arc::new(watch::Sender::new(StepLoop::Pending)),
            #[cfg(feature = "multimodal")]
            mm_processor: None,
            is_pooling: false,
            no_progress_timeout: DEFAULT_NO_PROGRESS_TIMEOUT,
        }
    }

    /// Enable or disable pooling mode.
    pub fn set_is_pooling(&mut self, enabled: bool) {
        self.is_pooling = enabled;
    }

    /// Whether the engine is in pooling mode.
    pub fn is_pooling(&self) -> bool {
        self.is_pooling
    }

    /// Reset the prefix cache. Returns `true` if successful, `false` if there
    /// are running requests blocking the reset.
    ///
    /// This sends a control message to the step loop which calls through to the
    /// engine core client's `reset_prefix_cache`.
    pub async fn reset_prefix_cache(&self) -> ServeResult<bool> {
        let (tx, rx) = oneshot::channel();
        self.control_tx
            .send(ControlMessage::ResetPrefixCache(tx))
            .map_err(|_| ServeError::Internal("step loop not running".into()))?;
        rx.await
            .map_err(|_| ServeError::Internal("step loop dropped reply".into()))
    }

    /// Put the engine to sleep, freeing GPU memory.
    pub async fn sleep(&self, level: u32) -> ServeResult<()> {
        let (tx, rx) = oneshot::channel();
        self.control_tx
            .send(ControlMessage::Sleep(level, tx))
            .map_err(|_| ServeError::Internal("step loop not running".into()))?;
        rx.await
            .map_err(|_| ServeError::Internal("step loop dropped reply".into()))?
            .map_err(ServeError::Internal)
    }

    /// Wake the engine from sleep.
    pub async fn wake_up(&self, tags: Option<Vec<String>>) -> ServeResult<()> {
        let (tx, rx) = oneshot::channel();
        self.control_tx
            .send(ControlMessage::WakeUp(tags, tx))
            .map_err(|_| ServeError::Internal("step loop not running".into()))?;
        rx.await
            .map_err(|_| ServeError::Internal("step loop dropped reply".into()))?
            .map_err(ServeError::Internal)
    }

    /// Whether the engine is currently sleeping.
    pub async fn is_sleeping(&self) -> ServeResult<bool> {
        let (tx, rx) = oneshot::channel();
        self.control_tx
            .send(ControlMessage::IsSleeping(tx))
            .map_err(|_| ServeError::Internal("step loop not running".into()))?;
        rx.await
            .map_err(|_| ServeError::Internal("step loop dropped reply".into()))
    }

    /// Override the no-progress watchdog timeout (for testing).
    #[cfg(test)]
    fn set_no_progress_timeout(&mut self, timeout: Duration) {
        self.no_progress_timeout = timeout;
    }

    /// Create a new `AsyncEngine` with a tokenizer for real text processing.
    pub fn with_tokenizer(
        client: Box<dyn EngineCoreClient + Send>,
        model_name: String,
        max_model_len: usize,
        tokenizer: Arc<Tokenizer>,
    ) -> Self {
        let mut engine = Self::new(client, model_name, max_model_len);
        engine.tokenizer = Some(tokenizer);
        engine
    }

    /// Create a new `AsyncEngine` with a tokenizer and chat template.
    pub fn with_tokenizer_and_template(
        client: Box<dyn EngineCoreClient + Send>,
        model_name: String,
        max_model_len: usize,
        tokenizer: Arc<Tokenizer>,
        chat_template: Arc<ChatTemplate>,
    ) -> Self {
        let mut engine = Self::new(client, model_name, max_model_len);
        engine.tokenizer = Some(tokenizer);
        engine.chat_template = Some(chat_template);
        engine
    }

    /// Set the chat template on this engine.
    pub fn set_chat_template(&mut self, template: Arc<ChatTemplate>) {
        self.chat_template = Some(template);
    }

    /// Set the tool call parser on this engine.
    pub fn set_tool_parser(&mut self, parser: Arc<dyn ToolCallParser>) {
        self.tool_parser = Some(parser);
    }

    /// The tool-call parser set on this engine, if any. The `/v1/messages` spans
    /// path uses it to extract `tool_use` blocks from the raw completion output
    /// (the chat path applies it internally).
    pub fn tool_parser(&self) -> Option<Arc<dyn ToolCallParser>> {
        self.tool_parser.clone()
    }

    /// Set the reasoning parser on this engine.
    pub fn set_reasoning_parser(&mut self, parser: Arc<dyn ReasoningParser>) {
        self.reasoning_parser = Some(parser);
    }

    /// Set default extra kwargs for chat template rendering.
    pub fn set_default_chat_template_kwargs(
        &mut self,
        kwargs: std::collections::HashMap<String, serde_json::Value>,
    ) {
        self.default_chat_template_kwargs = Some(kwargs);
    }

    /// Enable or disable async scheduling.
    /// Install model-recommended sampling defaults
    /// (`generation_config.json`). See `GenerationDefaults`.
    pub fn set_generation_defaults(
        &mut self,
        d: scratchy_core_common::sampling::GenerationDefaults,
    ) {
        self.generation_defaults = d;
    }

    pub fn set_async_scheduling(&mut self, enabled: bool) {
        self.async_scheduling = enabled;
    }

    /// Set the resolved multimodal processor (built from a scratchy
    /// inventory lookup at init). `None` clears it (text-only). All
    /// per-arch knobs (placeholder token id, image_size, smart-resize
    /// bounds, tokens-per-image, preprocess fn pointer) are carried by
    /// the resolved handle — the engine never names an arch.
    #[cfg(feature = "multimodal")]
    pub fn set_mm_processor(
        &mut self,
        processor: Option<std::sync::Arc<crate::multimodal::ResolvedMmProcessor>>,
    ) {
        self.mm_processor = processor;
    }

    /// Whether the engine takes image inputs: a multimodal processor is set.
    #[cfg(feature = "multimodal")]
    pub fn takes_images(&self) -> bool {
        self.mm_processor.is_some()
    }

    /// Whether the engine takes image inputs: never, in a build without `multimodal`.
    #[cfg(not(feature = "multimodal"))]
    pub fn takes_images(&self) -> bool {
        false
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Compute embeddings synchronously by sending through the embed channel
    /// and blocking on the reply. Panics if called outside a tokio runtime.
    pub fn embed_sync(&self, token_id_seqs: Vec<Vec<u32>>) -> anyhow::Result<Vec<Vec<f32>>> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.embed_tx
            .send(EmbedRequest {
                token_id_seqs,
                reply: reply_tx,
            })
            .map_err(|_| anyhow::anyhow!("embed channel closed"))?;
        // block_on the oneshot — we're inside spawn_blocking so this is fine
        tokio::runtime::Handle::current()
            .block_on(reply_rx)
            .map_err(|_| anyhow::anyhow!("embed reply channel closed"))?
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Get the maximum model length.
    pub fn max_model_len(&self) -> usize {
        self.max_model_len
    }

    /// A cheap snapshot of the currently-active request's live timing/counters,
    /// read directly from the shared request map. Returns `None` when idle.
    /// The lock is held only for map reads (never during model execution), so
    /// sampling this from a UI ticker is safe. Assumes `max_num_seqs = 1`.
    #[cfg(feature = "live-stats")]
    pub fn live_stats(&self) -> Option<LiveStats> {
        let reqs = self.requests.try_lock().ok()?;
        let rs = reqs.values().next()?;
        let ttft_s = rs
            .first_token_time
            .map(|t| t.duration_since(rs.submit_time).as_secs_f64());
        let decode_tok_s = if rs.itl_count > 0 && rs.itl_sum > 0.0 {
            rs.itl_count as f64 / rs.itl_sum
        } else {
            0.0
        };
        Some(LiveStats {
            generating: rs.first_token_time.is_some(),
            prompt_tokens: rs.num_prompt_tokens,
            cached_tokens: rs.num_cached_tokens,
            generated_tokens: rs.generated_token_ids.len() as u32,
            ttft_s,
            decode_tok_s,
        })
    }

    /// Recent inter-token latencies in milliseconds (newest last) for the live
    /// ITL distribution. Non-blocking; empty when idle or momentarily locked.
    /// Assumes `max_num_seqs = 1`.
    #[cfg(feature = "live-stats")]
    pub fn recent_itls(&self) -> Vec<f32> {
        let Ok(reqs) = self.requests.try_lock() else {
            return Vec::new();
        };
        reqs.values()
            .next()
            .map(|rs| rs.itl_samples.clone())
            .unwrap_or_default()
    }

    /// Whether this engine has a tokenizer configured.
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    /// Get a reference to the tokenizer, if available.
    pub fn tokenizer(&self) -> Option<&Arc<Tokenizer>> {
        self.tokenizer.as_ref()
    }

    /// Decode a single token id to its display string (e.g. `1234 -> " the"`),
    /// for rendering sampler candidates. Uses `decode` (not `id_to_token`) so
    /// byte-level/`▁` space markers are resolved; falls back to `<token_{id}>`.
    #[cfg(feature = "live-stats")]
    pub fn decode_token(&self, id: u32) -> String {
        self.tokenizer
            .as_ref()
            .and_then(|t| t.decode(&[id], false).ok())
            .unwrap_or_else(|| format!("<token_{id}>"))
    }

    /// Get a reference to the chat template, if available.
    pub fn chat_template(&self) -> Option<&Arc<ChatTemplate>> {
        self.chat_template.as_ref()
    }

    // -----------------------------------------------------------------------
    // Request handling
    // -----------------------------------------------------------------------

    /// Add a non-streaming chat completion request.
    ///
    /// Returns the completed response when generation finishes.
    /// Supports `n > 1`: generates `n` independent completions.
    pub async fn chat_completion(
        &self,
        request: protocol::ChatCompletionRequest,
    ) -> ServeResult<protocol::ChatCompletionResponse> {
        if self.is_pooling {
            return Err(ServeError::Validation(
                "server is in pooling mode — chat completions are not supported".to_string(),
            ));
        }
        let base_id = request
            .request_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.model_name.clone());
        let n = validate_n(request.n)?;

        let mut sampling_params = self.build_sampling_params_from_chat(&request)?;

        // Tokenize prompt once.
        let mut ec_request = self.chat_to_engine_request(&base_id, &request, &sampling_params)?;
        let mut prompt_token_ids = ec_request.prompt_token_ids.clone().unwrap_or_default();

        // Truncate prompt tokens from the left (keep the last N).
        truncate_prompt(&mut prompt_token_ids, request.truncate_prompt_tokens)?;
        ec_request.prompt_token_ids = Some(prompt_token_ids.clone());

        let num_prompt_tokens = prompt_token_ids.len() as u32;

        // Admission guard: reject prompts that exceed the context window before
        // they ever reach the scheduler (avoids the never-schedulable busy-spin).
        self.check_prompt_fits(prompt_token_ids.len())?;

        // Resolve max_tokens: None → remaining capacity, Some(v) → min(v, remaining).
        self.resolve_max_tokens(&mut sampling_params, prompt_token_ids.len());

        debug!(
            request_id = %base_id,
            prompt_tokens = num_prompt_tokens,
            max_tokens = ?sampling_params.max_tokens,
            max_model_len = self.max_model_len,
            temperature = sampling_params.temperature,
            "chat completion request"
        );

        // Per-HTTP-request metrics (once, not per child).
        #[cfg(feature = "metrics")]
        {
            let metrics = crate::metrics::VllmMetrics::global();
            metrics.requests_total.inc();
            metrics.prompt_tokens_total.inc_by(num_prompt_tokens as u64);
        }

        // Submit n child requests.
        let mut child_ids = Vec::with_capacity(n);
        for i in 0..n {
            let child_id = if n == 1 {
                base_id.clone()
            } else {
                format!("{base_id}-{i}")
            };

            let mut sp = sampling_params.clone();
            sp.seed = sp.seed.map(|s| s.wrapping_add(i as u64));

            let detokenizer = if sp.detokenize {
                self.tokenizer.as_ref().map(|tok| {
                    IncrementalDetokenizer::new(
                        Arc::clone(tok),
                        &prompt_token_ids,
                        sp.stop.clone(),
                        sp.min_tokens,
                        sp.include_stop_str_in_output,
                        sp.skip_special_tokens,
                    )
                })
            } else {
                None
            };

            let mut ec_req = ec_request.clone();
            ec_req.request_id = child_id.clone();
            ec_req.sampling_params = Some(sp);

            self.submit_request(
                child_id.clone(),
                ec_req,
                num_prompt_tokens,
                None,
                detokenizer,
                Choice {
                    index: i as u32,
                    prompt: PromptIndex(0),
                },
                None, // no streaming tool parser for non-streaming requests
                None,
                None, // no streaming reasoning parser for non-streaming requests
                request.include_reasoning,
            )
            .await?;

            child_ids.push(child_id);
        }

        // Poll all children and build choices.
        let mut choices = Vec::with_capacity(n);
        let mut total_completion_tokens = 0u32;
        // The prompt's cache hit, counted once as the prompt is: every choice runs the same
        // prompt, and each records its own hit on it (#580). The least of them: what the cache
        // held when the request came in, not a sibling's fresh prefill (`ChoiceUsage::totals`).
        let mut cached_tokens: Option<u32> = None;

        for (i, child_id) in child_ids.iter().enumerate() {
            let state = self.poll_until_done(child_id).await?;
            let completion_tokens = state.generated_token_ids.len() as u32;
            total_completion_tokens += completion_tokens;
            cached_tokens = Some(
                cached_tokens.map_or(state.num_cached_tokens, |c| c.min(state.num_cached_tokens)),
            );

            let finish_reason_str = state
                .finish_reason
                .map(|r| r.to_string())
                .unwrap_or_else(|| "stop".to_string());

            // `None` HERE MEANS "the engine has no tokenizer", and nothing else — see [`SharedDetok`].
            // It used to also mean "phase 2 is holding it right now", which is how correct token ids
            // came back as `<token_N>` text.
            let text = match state.detokenizer.as_ref() {
                Some(d) => d
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get_next_output_text(true, false),
                None => placeholder_text(&state.generated_token_ids),
            };
            warn_if_empty_text(child_id, &text, &state.generated_token_ids);

            let chat_logprobs = if state.logprobs.is_empty() {
                None
            } else {
                Some(build_chat_logprobs(
                    &state.logprobs,
                    self.tokenizer.as_deref(),
                ))
            };

            // Extract reasoning content if parser is configured.
            let (reasoning_text, text_for_tools, content_was_none) =
                if let Some(ref rp) = self.reasoning_parser {
                    let extracted = rp.extract_reasoning(&text);
                    let reasoning = if request.include_reasoning {
                        extracted.reasoning
                    } else {
                        None
                    };
                    // Tool extraction runs on the content portion only.
                    let was_none = extracted.content.is_none();
                    let content_for_tools = extracted.content.unwrap_or_default();
                    (reasoning, content_for_tools, was_none)
                } else {
                    (None, text, false)
                };

            // Try tool call extraction if parser is configured and request has tools.
            let (final_content, final_tool_calls, final_finish_reason) = match &self.tool_parser {
                Some(parser) if request.tools.is_some() => {
                    let reply =
                        reply_tool_calls(parser.as_ref(), text_for_tools, &request.tool_choice);
                    let finish = match reply.calls {
                        Some(_) => "tool_calls".to_string(),
                        None => finish_reason_str,
                    };
                    (reply.content, reply.calls, finish)
                }
                _ => (Some(text_for_tools), None, finish_reason_str),
            };

            // If reasoning parser returned content=None (everything was reasoning),
            // preserve None rather than wrapping an empty string.
            let final_content = if content_was_none && final_tool_calls.is_none() {
                match final_content {
                    Some(ref s) if s.is_empty() => None,
                    other => other,
                }
            } else {
                final_content
            };

            // Build prompt logprobs if available.
            let prompt_logprobs_content = state.prompt_logprobs.as_ref().map(|plps| {
                plps.iter()
                    .map(|opt_lp| {
                        opt_lp.as_ref().map(|lp| {
                            let token_str = self
                                .tokenizer
                                .as_ref()
                                .and_then(|tok| tok.decode(&[lp.sampled.token_id], false).ok())
                                .unwrap_or_else(|| format!("<token_{}>", lp.sampled.token_id));
                            let top_logprobs: Vec<protocol::ChatCompletionLogProb> = lp
                                .top_logprobs
                                .iter()
                                .map(|tlp| {
                                    let t = self
                                        .tokenizer
                                        .as_ref()
                                        .and_then(|tok| tok.decode(&[tlp.token_id], false).ok())
                                        .unwrap_or_else(|| format!("<token_{}>", tlp.token_id));
                                    protocol::ChatCompletionLogProb {
                                        token: t,
                                        logprob: tlp.logprob as f64,
                                        bytes: None,
                                    }
                                })
                                .collect();
                            protocol::ChatCompletionLogProbsContent {
                                token: token_str,
                                logprob: lp.sampled.logprob as f64,
                                bytes: None,
                                top_logprobs,
                            }
                        })
                    })
                    .collect()
            });

            choices.push(protocol::ChatCompletionResponseChoice {
                index: i as u32,
                message: protocol::ChatMessage {
                    role: "assistant".to_string(),
                    content: final_content,
                    refusal: None,
                    tool_calls: final_tool_calls,
                    reasoning: reasoning_text,
                },
                logprobs: chat_logprobs,
                finish_reason: Some(final_finish_reason),
                stop_reason: state.stop_reason.map(|sr| match sr {
                    StopReason::Token(id) => serde_json::Value::Number(id.into()),
                    StopReason::String(s) => serde_json::Value::String(s),
                }),
                prompt_logprobs: prompt_logprobs_content,
            });
        }

        let usage = protocol::UsageInfo {
            prompt_tokens: num_prompt_tokens,
            completion_tokens: Some(total_completion_tokens),
            total_tokens: num_prompt_tokens + total_completion_tokens,
            prompt_tokens_details: (cached_tokens.filter(|&c| c > 0)).map(|c| {
                protocol::PromptTokenUsageInfo {
                    cached_tokens: Some(c),
                }
            }),
        };

        Ok(protocol::ChatCompletionResponse::new(model, choices, usage))
    }

    /// Render a chat completion request: apply the chat template and tokenize,
    /// but do not generate any tokens.
    ///
    /// Returns `[conversation, engine_prompts]` matching Python vLLM's format.
    ///
    /// Port of: `POST /v1/chat/completions/render`
    pub fn render_chat_completion(
        &self,
        request: protocol::ChatCompletionRequest,
    ) -> ServeResult<protocol::ChatCompletionRenderResponse> {
        if self.is_pooling {
            return Err(ServeError::Validation(
                "server is in pooling mode — chat completions are not supported".to_string(),
            ));
        }

        // Build conversation: the original messages as JSON values.
        let conversation: Vec<serde_json::Value> = request
            .messages
            .iter()
            .map(|msg| serde_json::to_value(msg).unwrap_or_default())
            .collect();

        // Apply chat template + tokenize to get the rendered prompt.
        let base_id = request
            .request_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let sampling_params = self.build_sampling_params_from_chat(&request)?;
        let ec_request = self.chat_to_engine_request(&base_id, &request, &sampling_params)?;
        let token_ids = ec_request.prompt_token_ids.unwrap_or_default();

        // Decode token IDs back to text for the prompt field.
        let prompt_text = if let Some(tok) = &self.tokenizer {
            tok.decode(&token_ids, true)
                .unwrap_or_else(|_| String::new())
        } else {
            String::new()
        };

        let engine_prompts = vec![protocol::RenderEnginePrompt {
            prompt: serde_json::Value::String(prompt_text),
        }];

        Ok((conversation, engine_prompts))
    }

    /// Add a streaming chat completion request.
    ///
    /// Returns a receiver that yields streaming deltas.
    /// Supports `n > 1`: all child requests share one channel; each delta
    /// carries its `choice_index` so the SSE layer can route correctly.
    pub async fn chat_completion_stream(
        &self,
        request: protocol::ChatCompletionRequest,
    ) -> ServeResult<(String, String, mpsc::UnboundedReceiver<StreamDelta>)> {
        // TTFT ATTRIBUTION (SCRATCHY_TTFT_PHASE). The device side is instrumented down to the
        // microsecond by SCRATCHY_SDSC_PHASE_TIME, but everything BEFORE `submit_time` — chat-template
        // render + tokenize — was invisible, and on SuperDSC that gap is a third of TTFT. This stamps
        // arrival so the `[ttft-host] prep=` line below can price it; the matching
        // `submit_to_first_token=` line is emitted where the first token lands. prep + submit_to_first
        // accounts for everything except HTTP/SSE framing and the client's own handling.
        let _t_arrive = Instant::now();
        let base_id = request
            .request_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.model_name.clone());
        let n = validate_n(request.n)?;

        let mut sampling_params = self.build_sampling_params_from_chat(&request)?;
        let mut ec_request = self.chat_to_engine_request(&base_id, &request, &sampling_params)?;
        let mut prompt_token_ids = ec_request.prompt_token_ids.clone().unwrap_or_default();

        // Truncate prompt tokens from the left (keep the last N).
        truncate_prompt(&mut prompt_token_ids, request.truncate_prompt_tokens)?;
        ec_request.prompt_token_ids = Some(prompt_token_ids.clone());

        let num_prompt_tokens = prompt_token_ids.len() as u32;

        // Admission guard: reject prompts that exceed the context window before
        // they ever reach the scheduler (avoids the never-schedulable busy-spin).
        self.check_prompt_fits(prompt_token_ids.len())?;

        if std::env::var_os("SCRATCHY_TTFT_PHASE").is_some() {
            eprintln!(
                "[ttft-host] prep={:.2} ms  (chat template + tokenize, {num_prompt_tokens} prompt tokens)",
                _t_arrive.elapsed().as_secs_f64() * 1e3,
            );
        }

        // Resolve max_tokens: None → remaining capacity, Some(v) → min(v, remaining).
        self.resolve_max_tokens(&mut sampling_params, prompt_token_ids.len());

        // Per-HTTP-request metrics.
        #[cfg(feature = "metrics")]
        {
            let metrics = crate::metrics::VllmMetrics::global();
            metrics.requests_total.inc();
            metrics.prompt_tokens_total.inc_by(num_prompt_tokens as u64);
        }

        // Determine if tool parsing is active for this request.
        let use_tool_parser = self.tool_parser.is_some()
            && request.tools.is_some()
            && !is_tool_choice_none(&request.tool_choice);

        let forced_tool = ForcedTool::of(&request.tool_choice);

        // All n children share one channel.
        let (tx, rx) = mpsc::unbounded_channel();

        for i in 0..n {
            let child_id = if n == 1 {
                base_id.clone()
            } else {
                format!("{base_id}-{i}")
            };

            let mut sp = sampling_params.clone();
            sp.seed = sp.seed.map(|s| s.wrapping_add(i as u64));

            let detokenizer = if sp.detokenize {
                self.tokenizer.as_ref().map(|tok| {
                    IncrementalDetokenizer::new(
                        Arc::clone(tok),
                        &prompt_token_ids,
                        sp.stop.clone(),
                        sp.min_tokens,
                        sp.include_stop_str_in_output,
                        sp.skip_special_tokens,
                    )
                })
            } else {
                None
            };

            let mut ec_req = ec_request.clone();
            ec_req.request_id = child_id.clone();
            ec_req.sampling_params = Some(sp);

            // Create streaming tool parser state if applicable.
            let tool_state = if use_tool_parser {
                self.tool_parser
                    .as_ref()
                    .map(|p| p.create_streaming_state())
            } else {
                None
            };

            // Create streaming reasoning parser state if applicable.
            let reasoning_state = self
                .reasoning_parser
                .as_ref()
                .map(|p| p.create_streaming_state());

            self.submit_request(
                child_id,
                ec_req,
                num_prompt_tokens,
                Some(tx.clone()),
                detokenizer,
                Choice {
                    index: i as u32,
                    prompt: PromptIndex(0),
                },
                tool_state,
                forced_tool.clone(),
                reasoning_state,
                request.include_reasoning,
            )
            .await?;
        }

        // Drop the original sender so rx closes when all children finish.
        drop(tx);

        Ok((base_id, model, rx))
    }

    /// Add a non-streaming completion request.
    ///
    /// Supports `n > 1` and multi-prompt: generates `num_prompts * n` choices.
    pub async fn completion(
        &self,
        request: protocol::CompletionRequest,
    ) -> ServeResult<protocol::CompletionResponse> {
        if self.is_pooling {
            return Err(ServeError::Validation(
                "server is in pooling mode — text completions are not supported".to_string(),
            ));
        }
        let base_id = request
            .request_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.model_name.clone());
        let n = validate_n(request.n)?;

        let sampling_params = self.build_sampling_params_from_completion(&request)?;

        // Normalize prompt to Vec<Vec<u32>>.
        let mut prompts = self.tokenize_completion_prompts(&request)?;

        // Truncate each prompt from the left (keep the last N tokens).
        for p in &mut prompts {
            truncate_prompt(p, request.truncate_prompt_tokens)?;
        }
        let total = prompts.len() * n;

        // Per-HTTP-request metrics (once).
        let total_prompt_tokens: u32 = prompts.iter().map(|p| p.len() as u32).sum();
        #[cfg(feature = "metrics")]
        {
            let metrics = crate::metrics::VllmMetrics::global();
            metrics.requests_total.inc();
            metrics
                .prompt_tokens_total
                .inc_by(total_prompt_tokens as u64);
        }

        // Submit all child requests.
        let mut child_ids: Vec<(String, u32)> = Vec::with_capacity(total);

        for (p_idx, prompt_ids) in prompts.iter().enumerate() {
            let num_prompt_tokens = prompt_ids.len() as u32;

            for n_idx in 0..n {
                let choice_index = (p_idx * n + n_idx) as u32;
                let child_id = if total == 1 {
                    base_id.clone()
                } else {
                    format!("{base_id}-{choice_index}")
                };

                let mut sp = sampling_params.clone();
                sp.seed = sp.seed.map(|s| s.wrapping_add(n_idx as u64));
                sp.seal = request.seal;
                self.resolve_max_tokens(&mut sp, prompt_ids.len());

                let ec_req = EngineCoreRequest {
                    request_id: child_id.clone(),
                    prompt_token_ids: Some(prompt_ids.clone()),
                    sampling_params: Some(sp.clone()),
                    arrival_time: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64(),
                    client_index: 0,
                    priority: request.priority,
                    cache_salt: request.cache_salt.clone(),
                    data_parallel_rank: None,
                    is_pooling: false,
                    mm_data: None,
                    block_annotations: request.block_annotations.clone(),
                    seal: request.seal,
                    volatile: request.volatile,
                    cache_breakpoints: Vec::new(),
                };

                let detokenizer = self.tokenizer.as_ref().map(|tok| {
                    IncrementalDetokenizer::new(
                        Arc::clone(tok),
                        prompt_ids,
                        sp.stop.clone(),
                        sp.min_tokens,
                        sp.include_stop_str_in_output,
                        sp.skip_special_tokens,
                    )
                });

                self.submit_request(
                    child_id.clone(),
                    ec_req,
                    num_prompt_tokens,
                    None,
                    detokenizer,
                    Choice {
                        index: choice_index,
                        prompt: PromptIndex(p_idx as u32),
                    },
                    None, // no tool parsing for completions
                    None,
                    None, // no reasoning parsing for completions
                    true,
                )
                .await?;

                child_ids.push((child_id, choice_index));
            }
        }

        // Poll all children and build choices.
        let mut choices = Vec::with_capacity(total);
        let mut total_completion_tokens = 0u32;

        for (child_id, choice_index) in &child_ids {
            let state = self.poll_until_done(child_id).await?;
            let completion_tokens = state.generated_token_ids.len() as u32;
            total_completion_tokens += completion_tokens;

            let finish_reason_str = state
                .finish_reason
                .map(|r| r.to_string())
                .unwrap_or_else(|| "stop".to_string());

            // `None` HERE MEANS "the engine has no tokenizer", and nothing else — see [`SharedDetok`].
            // It used to also mean "phase 2 is holding it right now", which is how correct token ids
            // came back as `<token_N>` text.
            let text = match state.detokenizer.as_ref() {
                Some(d) => d
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get_next_output_text(true, false),
                None => placeholder_text(&state.generated_token_ids),
            };
            warn_if_empty_text(child_id, &text, &state.generated_token_ids);

            let completion_logprobs = if state.logprobs.is_empty() {
                None
            } else {
                Some(build_completion_logprobs(
                    &state.logprobs,
                    self.tokenizer.as_deref(),
                ))
            };

            // Build prompt logprobs for completion response if available.
            let prompt_lps = state.prompt_logprobs.as_ref().and_then(|plps| {
                // Convert Option<LogprobsOutput> entries (skipping the None for pos 0)
                // into a flat Vec for build_completion_logprobs.
                let flat: Vec<scratchy_core_common::LogprobsOutput> =
                    plps.iter().filter_map(|opt| opt.clone()).collect();
                if flat.is_empty() {
                    None
                } else {
                    Some(build_completion_logprobs(&flat, self.tokenizer.as_deref()))
                }
            });

            choices.push(protocol::CompletionResponseChoice {
                index: *choice_index,
                text,
                logprobs: completion_logprobs,
                finish_reason: Some(finish_reason_str),
                stop_reason: state.stop_reason.map(|sr| match sr {
                    StopReason::Token(id) => serde_json::Value::Number(id.into()),
                    StopReason::String(s) => serde_json::Value::String(s),
                }),
                prompt_logprobs: prompt_lps,
            });
        }

        let usage = protocol::UsageInfo {
            prompt_tokens: total_prompt_tokens,
            completion_tokens: Some(total_completion_tokens),
            total_tokens: total_prompt_tokens + total_completion_tokens,
            prompt_tokens_details: None,
        };

        Ok(protocol::CompletionResponse::new(model, choices, usage))
    }

    // -----------------------------------------------------------------------
    // Streaming completions
    // -----------------------------------------------------------------------

    /// Add a streaming completion request.
    ///
    /// Returns `(request_id, model, receiver)` — the receiver yields
    /// [`StreamDelta`] chunks exactly like `chat_completion_stream`.
    /// Simpler than chat streaming: no tool parsing, no reasoning parsing.
    pub async fn completion_stream(
        &self,
        request: protocol::CompletionRequest,
    ) -> ServeResult<(String, String, mpsc::UnboundedReceiver<StreamDelta>)> {
        if self.is_pooling {
            return Err(ServeError::Validation(
                "server is in pooling mode — text completions are not supported".to_string(),
            ));
        }
        let base_id = request
            .request_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.model_name.clone());
        let n = validate_n(request.n)?;

        let sampling_params = self.build_sampling_params_from_completion(&request)?;

        // Normalize prompt to Vec<Vec<u32>>.
        let mut prompts = self.tokenize_completion_prompts(&request)?;

        // Truncate each prompt from the left (keep the last N tokens).
        for p in &mut prompts {
            truncate_prompt(p, request.truncate_prompt_tokens)?;
        }
        let total = prompts.len() * n;

        // Per-HTTP-request metrics (once).
        // Per-HTTP-request metrics (once).
        #[cfg(feature = "metrics")]
        {
            let total_prompt_tokens: u32 = prompts.iter().map(|p| p.len() as u32).sum();
            let metrics = crate::metrics::VllmMetrics::global();
            metrics.requests_total.inc();
            metrics
                .prompt_tokens_total
                .inc_by(total_prompt_tokens as u64);
        }

        // All children share one channel.
        let (tx, rx) = mpsc::unbounded_channel();

        for (p_idx, prompt_ids) in prompts.iter().enumerate() {
            let num_prompt_tokens = prompt_ids.len() as u32;

            for n_idx in 0..n {
                let choice_index = (p_idx * n + n_idx) as u32;
                let child_id = if total == 1 {
                    base_id.clone()
                } else {
                    format!("{base_id}-{choice_index}")
                };

                let mut sp = sampling_params.clone();
                sp.seed = sp.seed.map(|s| s.wrapping_add(n_idx as u64));
                sp.seal = request.seal;
                self.resolve_max_tokens(&mut sp, prompt_ids.len());

                let ec_req = EngineCoreRequest {
                    request_id: child_id.clone(),
                    prompt_token_ids: Some(prompt_ids.clone()),
                    sampling_params: Some(sp.clone()),
                    arrival_time: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64(),
                    client_index: 0,
                    priority: request.priority,
                    cache_salt: request.cache_salt.clone(),
                    data_parallel_rank: None,
                    is_pooling: false,
                    mm_data: None,
                    block_annotations: request.block_annotations.clone(),
                    seal: request.seal,
                    volatile: request.volatile,
                    cache_breakpoints: Vec::new(),
                };

                let detokenizer = self.tokenizer.as_ref().map(|tok| {
                    IncrementalDetokenizer::new(
                        Arc::clone(tok),
                        prompt_ids,
                        sp.stop.clone(),
                        sp.min_tokens,
                        sp.include_stop_str_in_output,
                        sp.skip_special_tokens,
                    )
                });

                self.submit_request(
                    child_id,
                    ec_req,
                    num_prompt_tokens,
                    Some(tx.clone()),
                    detokenizer,
                    Choice {
                        index: choice_index,
                        prompt: PromptIndex(p_idx as u32),
                    },
                    None, // no tool parsing for completions
                    None,
                    None, // no reasoning parsing for completions
                    true,
                )
                .await?;
            }
        }

        // Drop the original sender so rx closes when all children finish.
        drop(tx);

        Ok((base_id, model, rx))
    }

    // -----------------------------------------------------------------------
    // Abort
    // -----------------------------------------------------------------------

    /// Abort a request by ID.
    ///
    /// Tells the scheduler to drop the sequence (so it stops generating and
    /// frees its slot) and removes it from request tracking.
    pub async fn abort_request(&self, request_id: &str) {
        let _ = self
            .control_tx
            .send(ControlMessage::Abort(vec![request_id.to_string()]));
        let mut reqs = self.requests.lock().await;
        reqs.remove(request_id);
    }

    // -----------------------------------------------------------------------
    // Embedding
    // -----------------------------------------------------------------------

    /// Process an embedding request.
    ///
    /// In pooling mode, routes through the scheduler (batched, lifecycle-managed).
    /// In default mode, uses the side-channel embed path (sequential).
    pub async fn embeddings(
        &self,
        request: protocol::EmbeddingRequest,
    ) -> ServeResult<protocol::EmbeddingResponse> {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.model_name.clone());

        // Tokenize inputs into token ID sequences.
        let token_id_seqs = self.tokenize_embedding_inputs(&request)?;
        let total_prompt_tokens: u32 = token_id_seqs.iter().map(|s| s.len() as u32).sum();

        let embeddings: Vec<EmbeddingData> = if self.is_pooling {
            // Pooling mode: route each input through the scheduler as a pooling request.
            let mut results = Vec::with_capacity(token_id_seqs.len());
            for token_ids in &token_id_seqs {
                let request_id = Uuid::new_v4().to_string();
                let num_prompt_tokens = token_ids.len() as u32;

                let ec_request = EngineCoreRequest {
                    request_id: request_id.clone(),
                    prompt_token_ids: Some(token_ids.clone()),
                    sampling_params: Some(SamplingParams {
                        max_tokens: Some(1), // Will never be used — pooling finishes in one pass.
                        ..Default::default()
                    }),
                    arrival_time: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64(),
                    client_index: 0,
                    priority: 0,
                    cache_salt: None,
                    data_parallel_rank: None,
                    is_pooling: true,
                    mm_data: None,
                    block_annotations: None,
                    seal: false,
                    volatile: false,
                    cache_breakpoints: Vec::new(),
                };

                self.submit_request(
                    request_id.clone(),
                    ec_request,
                    num_prompt_tokens,
                    None,
                    None,
                    Choice::default(),
                    None,
                    None,
                    None,
                    true,
                )
                .await?;

                // Wait for the request to complete.
                let state = self.poll_until_done(&request_id).await?;

                // Extract the embedding data from the pooler output.
                let emb = state.pooler_output.ok_or_else(|| {
                    ServeError::Internal("pooling request completed without embedding".into())
                })?;
                results.push(emb);
            }
            results
        } else {
            // Default mode: send to step loop via embed channel (bypasses scheduler).
            // Side-channel always returns single-vector embeddings.
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            self.embed_tx
                .send(EmbedRequest {
                    token_id_seqs,
                    reply: reply_tx,
                })
                .map_err(|_| ServeError::Internal("embed channel closed".into()))?;

            let vecs = reply_rx
                .await
                .map_err(|_| ServeError::Internal("embed reply channel closed".into()))??;
            vecs.into_iter().map(EmbeddingData::Single).collect()
        };

        // Build response objects, applying optional dimension truncation.
        let data: Vec<protocol::EmbeddingObject> = embeddings
            .into_iter()
            .enumerate()
            .map(|(i, emb_data)| {
                let embedding = match emb_data {
                    EmbeddingData::Single(mut emb) => {
                        if let Some(dims) = request.dimensions
                            && dims < emb.len()
                        {
                            emb.truncate(dims);
                            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                            if norm > 0.0 {
                                for x in &mut emb {
                                    *x /= norm;
                                }
                            }
                        }
                        serde_json::json!(emb)
                    }
                    EmbeddingData::Multi(mut rows) => {
                        if let Some(dims) = request.dimensions {
                            for row in &mut rows {
                                if dims < row.len() {
                                    row.truncate(dims);
                                    let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                                    if norm > 0.0 {
                                        for x in row.iter_mut() {
                                            *x /= norm;
                                        }
                                    }
                                }
                            }
                        }
                        serde_json::json!(rows)
                    }
                };
                protocol::EmbeddingObject {
                    index: i,
                    object: "embedding".to_string(),
                    embedding,
                }
            })
            .collect();

        Ok(protocol::EmbeddingResponse::new(
            model,
            data,
            protocol::EmbeddingUsage {
                prompt_tokens: total_prompt_tokens,
                total_tokens: total_prompt_tokens,
            },
        ))
    }

    /// Tokenize embedding request inputs into token ID sequences.
    fn tokenize_embedding_inputs(
        &self,
        request: &protocol::EmbeddingRequest,
    ) -> ServeResult<Vec<Vec<u32>>> {
        match &request.input {
            protocol::EmbeddingInput::Single(text) => {
                let ids = self.tokenize_embed_text(text)?;
                Ok(vec![ids])
            }
            protocol::EmbeddingInput::Multiple(items) => {
                let mut seqs = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        protocol::EmbeddingInputItem::Text(text) => {
                            seqs.push(self.tokenize_embed_text(text)?);
                        }
                        protocol::EmbeddingInputItem::TokenIds(ids) => {
                            seqs.push(ids.clone());
                        }
                    }
                }
                Ok(seqs)
            }
        }
    }

    /// Tokenize a single text string for embedding.
    ///
    /// Special tokens are added by default — encoder pooling models
    /// (BERT/ModernBERT family) require [CLS]/[SEP] (or their arch-
    /// specific equivalents) for the pooled hidden state to match
    /// the model's training distribution. Matches Python vLLM's
    /// embedding path, which calls `tokenizer.encode(text,
    /// add_special_tokens=True)` by default for the same reason.
    fn tokenize_embed_text(&self, text: &str) -> ServeResult<Vec<u32>> {
        if let Some(ref tokenizer) = self.tokenizer {
            tokenizer.encode(text, true)
        } else {
            // Fallback: byte-level tokenization (for testing without a tokenizer).
            Ok(text.bytes().map(|b| b as u32).collect())
        }
    }

    // -----------------------------------------------------------------------
    // Engine step loop
    // -----------------------------------------------------------------------

    /// Start the background step loop.
    ///
    /// Takes ownership of the engine client and moves it into a background
    /// task. After this call, the engine client is exclusively owned by the
    /// step loop — all interaction goes through channels and the shared
    /// request map.
    ///
    /// Must be called exactly once. Panics if called twice.
    pub fn spawn_step_loop(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let (client, rx, embed_rx, control_rx) = self
            .pending_loop
            .lock()
            .expect("spawn_step_loop lock poisoned")
            .take()
            .expect("spawn_step_loop called twice");

        // Running BEFORE spawning, so poll_until_done never takes a loop that hasn't started
        // yet for a dead one.
        self.step_loop.send_replace(StepLoop::Running);

        let args = StepLoopArgs {
            client,
            request_rx: rx,
            embed_rx,
            control_rx,
            requests: Arc::clone(&self.requests),
            notify: Arc::clone(&self.notify),
            no_progress_timeout: self.no_progress_timeout,
        };
        let requests = Arc::clone(&self.requests);
        let notify = Arc::clone(&self.notify);
        let state = Arc::clone(&self.step_loop);
        if self.async_scheduling {
            info!("Async scheduling enabled — overlapping GPU execution with CPU scheduling");
            tokio::spawn(Self::run_to_end(
                Self::step_loop_async(args),
                requests,
                notify,
                state,
            ))
        } else {
            tokio::spawn(Self::run_to_end(
                Self::step_loop_sync(args),
                requests,
                notify,
                state,
            ))
        }
    }

    /// Run a step loop to its end, however it ends, then fail every request it was still
    /// serving and say that it ended. A panic in the loop — a detokenizer slicing inside a
    /// character took one down in #576 — unwinds past the loop's own exit path, and a loop
    /// that returns leaves its requests in the map. Either way they waited on a loop that was
    /// gone: a stream on a sender nothing would send on again, and a server whose graceful
    /// shutdown waited on that connection forever (#587).
    async fn run_to_end(
        step_loop: impl Future<Output = ()>,
        requests: Arc<Mutex<HashMap<String, RequestState>>>,
        notify: Arc<Notify>,
        state: Arc<watch::Sender<StepLoop>>,
    ) {
        let end = {
            let mut step_loop = std::pin::pin!(step_loop);
            std::future::poll_fn(|cx| {
                match std::panic::catch_unwind(AssertUnwindSafe(|| step_loop.as_mut().poll(cx))) {
                    Ok(poll) => poll.map(|()| StepLoopEnd::Exited),
                    Err(panic) => Poll::Ready(StepLoopEnd::Panicked(panic_message(&*panic))),
                }
            })
            .await
        }; // The loop drops here, and its engine client with it.
        match &end {
            StepLoopEnd::Exited => info!("The engine step loop exited"),
            StepLoopEnd::Panicked(_) => error!("The engine step loop {end}"),
        }
        Self::fail_requests(
            &mut *requests.lock().await,
            &format!("the engine step loop {end}"),
            |_| true,
        );
        state.send_replace(StepLoop::Ended(end));
        notify.notify_waiters(); // Wake any poll_until_done waiters so they see the dead loop.
    }

    /// How the step loop ended, once it has.
    pub fn step_loop_end(&self) -> Option<StepLoopEnd> {
        match &*self.step_loop.borrow() {
            StepLoop::Ended(end) => Some(end.clone()),
            StepLoop::Pending | StepLoop::Running => None,
        }
    }

    /// Resolves when the step loop ends: the engine is dead then, and can serve nothing again.
    pub async fn step_loop_ended(&self) -> StepLoopEnd {
        let mut state = self.step_loop.subscribe();
        loop {
            if let StepLoop::Ended(end) = &*state.borrow_and_update() {
                return end.clone();
            }
            state
                .changed()
                .await
                .expect("the engine holds its step loop's state");
        }
    }

    /// Handle a control message from the HTTP layer.
    fn handle_control_message(client: &mut Box<dyn EngineCoreClient + Send>, msg: ControlMessage) {
        match msg {
            ControlMessage::ResetPrefixCache(reply) => {
                let result = client.reset_prefix_cache().unwrap_or(false);
                let _ = reply.send(result);
            }
            ControlMessage::Sleep(level, reply) => {
                let result = client.sleep(level).map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            ControlMessage::WakeUp(tags, reply) => {
                let result = client.wake_up(tags.as_deref()).map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            ControlMessage::IsSleeping(reply) => {
                let _ = reply.send(client.is_sleeping());
            }
            ControlMessage::Abort(ids) => {
                client
                    .abort_requests(&ids)
                    .unwrap_or_else(|e| error!("abort_requests failed: {e}"));
            }
        }
    }

    /// Handle a control message in async scheduling mode, routing executor ops
    /// through the executor thread channel.
    async fn handle_control_message_async(
        client: &mut Box<dyn EngineCoreClient + Send>,
        msg: ControlMessage,
        executor_tx: &tokio::sync::mpsc::Sender<ExecutorWork>,
    ) {
        match msg {
            ControlMessage::ResetPrefixCache(reply) => {
                let result = client.reset_prefix_cache().unwrap_or(false);
                let _ = reply.send(result);
            }
            ControlMessage::Sleep(level, reply) => {
                // Abort running requests first via the client.
                client.abort_running_requests();
                let _ = executor_tx.send(ExecutorWork::Sleep(level, reply)).await;
            }
            ControlMessage::WakeUp(tags, reply) => {
                let _ = executor_tx.send(ExecutorWork::WakeUp(tags, reply)).await;
            }
            ControlMessage::IsSleeping(reply) => {
                let _ = executor_tx.send(ExecutorWork::IsSleeping(reply)).await;
            }
            ControlMessage::Abort(ids) => {
                client
                    .abort_requests(&ids)
                    .unwrap_or_else(|e| error!("abort_requests failed: {e}"));
            }
        }
    }

    /// The step loop, one step at a time.
    fn step_loop_sync(args: StepLoopArgs) -> impl Future<Output = ()> + Send + 'static {
        let StepLoopArgs {
            mut client,
            mut request_rx,
            mut embed_rx,
            mut control_rx,
            requests,
            notify,
            no_progress_timeout,
        } = args;
        async move {
            let mut last_progress = Instant::now();
            loop {
                // 0. Drain pending embedding requests (synchronous, bypasses scheduler).
                while let Ok(embed_req) = embed_rx.try_recv() {
                    let result =
                        tokio::task::block_in_place(|| client.embed(embed_req.token_id_seqs));
                    let _ = embed_req.reply.send(result.map_err(ServeError::from));
                }

                // 0b. Drain control messages.
                while let Ok(msg) = control_rx.try_recv() {
                    Self::handle_control_message(&mut client, msg);
                }

                // 1. Drain pending request submissions (non-blocking).
                //    This never blocks on a mutex — the channel is lock-free.
                let mut added = false;
                while let Ok(ec_request) = request_rx.try_recv() {
                    if let Err(e) = client.add_request(ec_request) {
                        error!("Failed to add request: {}", e);
                    }
                    added = true;
                }

                // 2. Check if there's work. If not, wait for a request.
                let has_requests = {
                    let reqs = requests.lock().await;
                    !reqs.is_empty()
                };

                if !has_requests && !added {
                    // Reset watchdog timer when idle (no pending requests).
                    last_progress = Instant::now();
                    // Block until a new request arrives on either channel.
                    tokio::select! {
                        Some(ec_request) = request_rx.recv() => {
                            if let Err(e) = client.add_request(ec_request) {
                                error!("Failed to add request: {}", e);
                            }
                        }
                        Some(embed_req) = embed_rx.recv() => {
                            let result = tokio::task::block_in_place(|| {
                                client.embed(embed_req.token_id_seqs)
                            });
                            let _ = embed_req.reply.send(result.map_err(ServeError::from));
                            // After handling embed, continue loop to check for more work.
                            notify.notify_waiters();
                            continue;
                        }
                        Some(msg) = control_rx.recv() => {
                            Self::handle_control_message(&mut client, msg);
                            continue;
                        }
                        else => break, // All channels closed, engine dropped.
                    }
                }

                // 3. Run one engine step. This is synchronous (model forward)
                //    so we use block_in_place to let tokio schedule other work.
                let step_result = tokio::task::block_in_place(|| client.get_output());

                match step_result {
                    Ok(outcome) => {
                        // ⭐ THE SERVER IS THE ONE LOOP FOR WHICH A STALL IS *NOT* TERMINAL. An
                        // HTTP handler can submit a request between two steps, so "nothing was
                        // schedulable just now" may resolve on its own. So this loop deliberately
                        // treats a stall as a step that did no work — it falls through to the
                        // `no_progress_timeout` watchdog below, which aborts only after a stall has
                        // persisted. The blocking loops in `llm.rs` cannot make this choice, and
                        // [`StepOutcome`] is what forces each loop to say which one it is.
                        let (outputs, model_executed) = match outcome {
                            StepOutcome::Progressed {
                                outputs,
                                model_executed,
                            } => (outputs, model_executed),
                            StepOutcome::Stalled | StepOutcome::Idle => {
                                (scratchy_core_common::EngineCoreOutputs::default(), false)
                            }
                        };

                        // 4. Update scheduler gauges from stats.
                        #[cfg(feature = "metrics")]
                        if let Some(stats) = &outputs.scheduler_stats {
                            let m = crate::metrics::VllmMetrics::global();
                            m.num_requests_running.set(stats.num_running_reqs as f64);
                            m.num_requests_waiting.set(stats.num_waiting_reqs as f64);
                            m.kv_cache_usage_perc.set(stats.kv_cache_usage);
                            m.gpu_cache_blocks_used
                                .set(stats.gpu_cache_blocks_used as i64);
                            m.gpu_cache_blocks_total
                                .set(stats.gpu_cache_blocks_total as i64);
                            m.prefix_cache_blocks.set(stats.num_cached_blocks as i64);
                            if let Some(sd) = &stats.spec_decode_stats {
                                m.spec_decode_num_drafts.inc_by(sd.num_drafts as u64);
                                m.spec_decode_num_draft_tokens
                                    .inc_by(sd.num_draft_tokens as u64);
                                m.spec_decode_num_accepted_tokens
                                    .inc_by(sd.num_accepted_tokens as u64);
                            }
                        }

                        // 5. Route outputs to requests (brief lock).
                        if !outputs.outputs.is_empty() {
                            last_progress = Instant::now();

                            // Phase 1: accumulate tokens, take detokenizers (brief lock).
                            let detok_work = {
                                let mut reqs = requests.lock().await;
                                Self::process_outputs_phase1(&mut reqs, &outputs.outputs)
                            };

                            // Phase 2: parallel detokenize (no lock held).
                            // See the note at the other call site: rayon from async blocks a
                            // tokio worker for the whole parallel phase.
                            let detok_results = tokio::task::block_in_place(|| {
                                Self::parallel_detokenize(detok_work)
                            });

                            // Phase 3: apply detok results, send streams (brief lock).
                            let stop_string_finished = {
                                let mut reqs = requests.lock().await;
                                Self::process_outputs_phase3(
                                    &mut reqs,
                                    outputs.outputs,
                                    detok_results,
                                )
                            };

                            // A stop string finished a request the engine is still decoding —
                            // abort it there, or it runs on to max_tokens under the next
                            // request's steps (#711).
                            if !stop_string_finished.is_empty() {
                                client
                                    .abort_requests(&stop_string_finished)
                                    .unwrap_or_else(|e| error!("abort_requests failed: {e}"));
                            }

                            // Wake waiters only when there are actual outputs.
                            notify.notify_waiters();
                        } else if has_requests && last_progress.elapsed() > no_progress_timeout {
                            // Watchdog: no output tokens for too long with pending
                            // requests — likely a scheduler bug. Abort everything.
                            let reqs_count = {
                                let r = requests.lock().await;
                                r.len()
                            };
                            error!(
                                "No output tokens for {:?} with {} pending requests \
                                 — aborting all (no-progress watchdog)",
                                last_progress.elapsed(),
                                reqs_count,
                            );
                            let err_msg = format!(
                                "No-progress watchdog: no output tokens for {:?}",
                                last_progress.elapsed(),
                            );
                            client.abort_running_requests();
                            Self::fail_requests(&mut *requests.lock().await, &err_msg, |_| true);
                            notify.notify_waiters();
                            last_progress = Instant::now();
                        } else if !model_executed {
                            // No work done — yield to avoid busy-spinning.
                            tokio::task::yield_now().await;
                        }
                    }
                    Err(e) => {
                        error!("Engine step error: {}", e);
                        // Abort all running requests to prevent infinite retry loop.
                        // get_output() internally does schedule + execute + finalize,
                        // and we don't know which requests were in the failed batch,
                        // so abort everything the scheduler currently has.
                        let err_msg = format!("Engine step error: {e}");
                        client.abort_running_requests();
                        Self::fail_requests(&mut *requests.lock().await, &err_msg, |_| true);
                        notify.notify_waiters();
                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    }
                }
            }
        }
    }

    /// Async scheduling step loop: overlaps GPU execution with CPU work.
    ///
    /// The executor is moved to a dedicated OS thread. The tokio task handles
    /// scheduling, output processing, and request drain concurrently with
    /// GPU execution.
    fn step_loop_async(args: StepLoopArgs) -> impl Future<Output = ()> + Send + 'static {
        let StepLoopArgs {
            mut client,
            mut request_rx,
            mut embed_rx,
            mut control_rx,
            requests,
            notify,
            no_progress_timeout,
        } = args;
        // Take the executor out of the client for the dedicated thread.
        let executor = client
            .take_executor()
            .expect("async scheduling requires an executor");

        // Bounded channel with capacity 2: allows up to 2 batches in flight
        // (one executing, one queued) without blocking the tokio thread.
        let (sched_tx, sched_rx) = tokio::sync::mpsc::channel::<ExecutorWork>(2);
        let (model_tx, mut model_rx) = tokio::sync::mpsc::channel::<ExecutorResult>(2);

        // Spawn the executor on a dedicated OS thread, and own both its
        // channel sender and its join handle in one guard moved into the
        // loop below. A bare `spawn` whose handle is dropped detaches the
        // thread: nothing then waits out the teardown it runs at the end
        // (`executor.shutdown()` → the worker's sidecar drain and residency
        // release), so the process exited mid-drain whenever the loop ended
        // — measured as the aligned-weights sidecar being discarded
        // mid-write on every SIGTERM'd launch (#411), the step loop's task
        // finishing while the executor thread was still shutting down.
        let executor_pipe = ExecutorPipeline {
            tx: Some(sched_tx),
            thread: Some(
                std::thread::Builder::new()
                    .name("scratchy-serving-worker".into())
                    .spawn(move || executor_thread_loop(executor, sched_rx, model_tx))
                    .expect("failed to spawn executor thread"),
            ),
        };

        async move {
            // Pre-scheduling with deferred finalization:
            //
            // GPU: ─execute(N)───────────────────────┬─execute(N+1)──
            //                                        │ (pre-queued)
            // CPU: finalize(N-1)→sched(N+1)→wait(N)→send│→finalize(N)→...
            //
            // The scheduler uses `num_output_placeholders` to account for
            // tokens that are in-flight on the GPU but not yet finalized.
            // This allows schedule(N+1) to run before finalize(N) without
            // double-scheduling the same position.
            let mut deferred: Option<(
                Box<scratchy_serving_scheduler::scheduler::output::SchedulerOutput>,
                ModelRunnerOutput,
            )> = None;
            let mut gpu_in_flight: u32 = 0;
            // A step in flight that drafts (the worker's head drafts the next step's tokens) or
            // verifies (the next step's positions depend on how many drafts it keeps): nothing is
            // scheduled behind it until it is finalized.
            let mut awaits_step = false;
            let mut last_progress = Instant::now();

            'step: loop {
                // 0. Drain pending embedding requests.
                while let Ok(embed_req) = embed_rx.try_recv() {
                    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
                    if executor_pipe
                        .tx()
                        .send(ExecutorWork::Embed(embed_req.token_id_seqs, reply_tx))
                        .await
                        .is_err()
                    {
                        let _ = embed_req.reply.send(Err(ServeError::Internal(
                            "executor thread shut down".into(),
                        )));
                        break 'step;
                    }
                    let result = reply_rx
                        .await
                        .unwrap_or(Err(ServeError::Internal("embed reply lost".into())));
                    let _ = embed_req.reply.send(result);
                }

                // 0b. Drain control messages.
                while let Ok(msg) = control_rx.try_recv() {
                    Self::handle_control_message_async(&mut client, msg, executor_pipe.tx()).await;
                }

                // 1. Drain pending request submissions (non-blocking).
                let mut added = false;
                while let Ok(ec_request) = request_rx.try_recv() {
                    if let Err(e) = client.add_request(ec_request) {
                        error!("Failed to add request: {}", e);
                    }
                    added = true;
                }

                // 2. Finalize the previously completed step. The GPU is
                //    already running (or about to run) the next batch, so
                //    this CPU work overlaps with GPU execution.
                if let Some((prev_sched, prev_output)) = deferred.take() {
                    match client.finalize_step(&prev_sched, &prev_output) {
                        Ok(outputs) => {
                            let stop_string_finished = route_step_outputs(&requests, outputs).await;
                            // A stop string finished a request the engine is still decoding —
                            // abort it there, or it runs on to max_tokens under the next
                            // request's steps (#711).
                            if !stop_string_finished.is_empty() {
                                client
                                    .abort_requests(&stop_string_finished)
                                    .unwrap_or_else(|e| error!("abort_requests failed: {e}"));
                            }
                            // A finalized step IS forward progress: prefill
                            // chunks sample no tokens, so gating this reset on
                            // sampled outputs would let the watchdog shoot a
                            // healthy >60s prefill and would date a real stall
                            // from the last sampled token, not stall onset.
                            last_progress = Instant::now();
                        }
                        Err(e) => {
                            error!("finalize_step error: {}", e);
                        }
                    }
                    notify.notify_waiters();
                }

                // 3. Pre-schedule: fill the pipeline up to 2 in-flight
                //    batches. With placeholder tracking, the scheduler
                //    correctly accounts for tokens still on the GPU.
                while gpu_in_flight < 2 && !awaits_step {
                    match client.schedule_next() {
                        Ok(Some(sched)) => {
                            let speculative = sched.is_speculative() && !sched.spec_pipeline.leads;
                            if executor_pipe
                                .tx()
                                .send(ExecutorWork::Execute(Box::new(sched)))
                                .await
                                .is_err()
                            {
                                // The executor thread exited: nothing will run a step again.
                                break 'step;
                            }
                            gpu_in_flight += 1;
                            awaits_step = speculative;
                        }
                        Ok(None) => break, // Nothing to schedule.
                        Err(e) => {
                            error!("schedule_next error: {}", e);
                            break;
                        }
                    }
                }

                // 4. Wait for the oldest GPU result. A deferred one is waited
                //    for here — the batch queued behind it keeps the GPU busy —
                //    and fails its step like an executor error if its device
                //    work did.
                //    The wait blocks its thread for the step's device work, so
                //    it runs on a blocking thread, never this task's worker:
                //    there it held the streams this loop had just woken (in
                //    the worker's LIFO slot, which no other worker steals)
                //    until the loop next yielded, and their chunks went out
                //    in bursts.
                if gpu_in_flight > 0 {
                    let received = match model_rx.recv().await {
                        Some(ExecutorResult::Model(result, sched)) => {
                            let resolved = tokio::task::spawn_blocking(move || {
                                result
                                    .map_err(|e| e.to_string())
                                    .and_then(|mut output| output.resolve().map(|()| output))
                            })
                            .await
                            .unwrap_or_else(|e| Err(format!("resolving the step: {e}")));
                            Some((resolved, sched))
                        }
                        None => None,
                    };
                    match received {
                        Some((Ok(model_output), sched)) => {
                            deferred = Some((sched, model_output));
                            gpu_in_flight -= 1;
                        }
                        Some((Err(e), sched)) => {
                            error!("Executor error: {}", e);
                            // Aborts the step's requests in the engine, too, and any a step behind
                            // it admitted onto the KV it never wrote.
                            let aborted = client.abandon_step(&sched);
                            let err_msg = format!("Executor error: {e}");
                            Self::fail_requests(&mut *requests.lock().await, &err_msg, |id| {
                                aborted.iter().any(|a| a == id)
                            });
                            notify.notify_waiters();
                            gpu_in_flight -= 1;
                        }
                        None => {
                            // Executor thread exited.
                            break;
                        }
                    }
                    // The last step sent has come back once none is in flight, so nothing is left
                    // to wait on. Here, not in step 2's finalize: a failed step is never
                    // finalized, and a wait it set would block step 3 for good (#477). Step 2
                    // still finalizes a returned step before step 3 schedules the next.
                    awaits_step &= gpu_in_flight > 0;

                    // Drain requests that arrived while waiting for GPU.
                    while let Ok(ec_request) = request_rx.try_recv() {
                        if let Err(e) = client.add_request(ec_request) {
                            error!("Failed to add request: {}", e);
                        }
                    }
                } else {
                    // 5. No GPU work — idle-wait for a new request.
                    let has_requests = {
                        let reqs = requests.lock().await;
                        !reqs.is_empty()
                    };

                    if !has_requests && !added {
                        last_progress = Instant::now();
                        tokio::select! {
                            Some(ec_request) = request_rx.recv() => {
                                if let Err(e) = client.add_request(ec_request) {
                                    error!("Failed to add request: {}", e);
                                }
                            }
                            Some(embed_req) = embed_rx.recv() => {
                                let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
                                if executor_pipe
                                    .tx()
                                    .send(ExecutorWork::Embed(
                                        embed_req.token_id_seqs,
                                        reply_tx,
                                    ))
                                    .await
                                    .is_err()
                                {
                                    let _ = embed_req.reply.send(Err(ServeError::Internal(
                                        "executor thread shut down".into(),
                                    )));
                                    break 'step;
                                } else {
                                    let result = reply_rx.await.unwrap_or(Err(
                                        ServeError::Internal("embed reply lost".into()),
                                    ));
                                    let _ = embed_req.reply.send(result);
                                }
                                notify.notify_waiters();
                                continue;
                            }
                            Some(msg) = control_rx.recv() => {
                                Self::handle_control_message_async(&mut client, msg, executor_pipe.tx()).await;
                                continue;
                            }
                            else => break, // All channels closed.
                        }
                    } else if has_requests && last_progress.elapsed() > no_progress_timeout {
                        // Watchdog: no output tokens for too long with pending
                        // requests — likely a scheduler bug. Abort everything.
                        let reqs_count = {
                            let r = requests.lock().await;
                            r.len()
                        };
                        error!(
                            "No output tokens for {:?} with {} pending requests \
                             — aborting all (no-progress watchdog, async)",
                            last_progress.elapsed(),
                            reqs_count,
                        );
                        let err_msg = format!(
                            "No-progress watchdog: no output tokens for {:?}",
                            last_progress.elapsed(),
                        );
                        client.abort_running_requests();
                        Self::fail_requests(&mut *requests.lock().await, &err_msg, |_| true);
                        notify.notify_waiters();
                        last_progress = Instant::now();
                    } else {
                        tokio::task::yield_now().await;
                    }
                }
            }

            // Finalize any remaining deferred output before shutdown.
            if let Some((prev_sched, prev_output)) = deferred.take() {
                if let Ok(outputs) = client.finalize_step(&prev_sched, &prev_output) {
                    let stop_string_finished = route_step_outputs(&requests, outputs).await;
                    // A stop string finished a request the engine is still decoding —
                    // abort it there, or it runs on to max_tokens under the next
                    // request's steps (#711).
                    if !stop_string_finished.is_empty() {
                        client
                            .abort_requests(&stop_string_finished)
                            .unwrap_or_else(|e| error!("abort_requests failed: {e}"));
                    }
                }
                notify.notify_waiters();
            }

            // Shutdown: drop the sender so the executor thread exits.
            let _ = executor_pipe.tx().send(ExecutorWork::Shutdown).await;
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Submit a request to the engine.
    ///
    /// Inserts the request state into the map (brief lock), then sends the
    /// engine request to the step loop via channel (non-blocking).
    ///
    /// Note: `requests_total` and `prompt_tokens_total` metrics are the
    /// caller's responsibility (once per HTTP request). This method only
    /// increments `requests_active` (once per engine-core request).
    #[allow(clippy::too_many_arguments)]
    async fn submit_request(
        &self,
        request_id: String,
        ec_request: EngineCoreRequest,
        num_prompt_tokens: u32,
        stream_tx: Option<mpsc::UnboundedSender<StreamDelta>>,
        detokenizer: Option<IncrementalDetokenizer>,
        choice: Choice,
        tool_parser_state: Option<Box<dyn StreamingToolParserState + Send>>,
        forced_tool: Option<ForcedTool>,
        reasoning_parser_state: Option<Box<dyn StreamingReasoningParserState + Send>>,
        include_reasoning: bool,
    ) -> ServeResult<()> {
        // ⛔ A PROMPT LONGER THAN THE MODEL'S CONTEXT IS REJECTED HERE, ALONE.
        //
        // `max_model_len` is already capped to what the backend's paged attention can ADDRESS
        // (`init_cache` -> `Worker::kv_max_addressable_tokens`; spyre logs
        // "Capping max_model_len 131072 -> 1536"). Nothing compared a request against it, so an over-long
        // prompt was ADMITTED and only discovered deep in the worker — measured on granite-3.1-8b: a
        // 1986-token request in a 1536-token model came back `status=200` with ZERO tokens while the
        // worker's own refusal fired and failed the other THREE requests in the batch with an
        // `Executor error`.
        //
        // ⭐ THE PLACEMENT IS THE POINT, and it took three tries to find it. Inside the forward
        // (`ensure_pages`) the refusal is BATCH-FATAL — one over-deep row errors every row. In the engine's
        // cap it never fires, because capping the config is not the same as validating a request against it.
        // HERE it fails exactly one request, before any batch is formed, and every other request in the
        // file is served normally.
        if num_prompt_tokens as usize > self.max_model_len {
            return Err(crate::error::ServeError::Validation(format!(
                "prompt is {num_prompt_tokens} tokens but this model serves at most {} — the \
                 limit is what the backend's KV memory funds and its prefix mask addresses",
                self.max_model_len
            )));
        }
        #[cfg(feature = "metrics")]
        {
            let metrics = crate::metrics::VllmMetrics::global();
            metrics.requests_active.inc();
        }

        // Insert request state (brief lock, no engine interaction).
        {
            let mut reqs = self.requests.lock().await;
            reqs.insert(
                request_id,
                RequestState {
                    generated_token_ids: Vec::new(),
                    num_prompt_tokens,
                    num_cached_tokens: 0,
                    finish_reason: None,
                    error: None,
                    stop_reason: None,
                    stream_tx,
                    // WRAPPED ONCE, HERE. Callers hand over a plain detokenizer; the sharing is
                    // the state's business, and this is the only place a request's is installed — so
                    // exactly one place could ever get it wrong.
                    detokenizer: detokenizer.map(|d| Arc::new(std::sync::Mutex::new(d))),
                    choice,
                    submit_time: Instant::now(),
                    first_token_time: None,
                    last_token_time: None,
                    itl_count: 0,
                    itl_sum: 0.0,
                    #[cfg(feature = "live-stats")]
                    itl_samples: Vec::new(),
                    logprobs: Vec::new(),
                    prompt_logprobs: None,
                    tool_parser_state,
                    reasoning_parser_state,
                    reasoning_ended: false,
                    include_reasoning,
                    accumulated_text: String::new(),
                    accumulated_content: String::new(),
                    accumulated_token_ids: Vec::new(),
                    tool_calls_emitted: false,
                    tool_calls: ToolCallStream::new(forced_tool),
                    pooler_output: None,
                },
            );
        }

        // Send to the step loop via channel (non-blocking, no mutex).
        self.request_tx
            .send(ec_request)
            .map_err(|_| ServeError::Engine("engine step loop shut down".to_string()))?;

        Ok(())
    }

    /// Poll until a request completes, returning its final state.
    ///
    /// This does NOT call `step()` itself — the background step loop
    /// (spawned by `spawn_step_loop`) drives the engine. This method
    /// just waits for the `Notify` signal after each step and checks
    /// whether the request is done. This allows many concurrent requests
    /// to wait efficiently without fighting over the engine mutex.
    async fn poll_until_done(&self, request_id: &str) -> ServeResult<RequestState> {
        loop {
            // Register interest in notifications BEFORE checking state.
            // This prevents the classic Notify race: if notify_waiters()
            // fires between our check and our await, we still catch it.
            let notified = self.notify.notified();

            // Check if the step loop is still running.
            if !matches!(*self.step_loop.borrow(), StepLoop::Running) {
                // Step loop has exited — check one more time for results,
                // then error out if the request isn't done.
                let mut reqs = self.requests.lock().await;
                if let Some(req_state) = reqs.get(request_id)
                    && req_state.finish_reason.is_some()
                {
                    let state = reqs.remove(request_id).unwrap();
                    if let Some(ref err) = state.error {
                        return Err(ServeError::Internal(err.clone()));
                    }
                    return Ok(state);
                }
                return Err(ServeError::Internal(
                    "step loop exited before request completed".to_string(),
                ));
            }

            // Check if the request is done (brief lock).
            {
                let mut reqs = self.requests.lock().await;
                if let Some(req_state) = reqs.get(request_id)
                    && req_state.finish_reason.is_some()
                {
                    let state = reqs.remove(request_id).unwrap();
                    if let Some(ref err) = state.error {
                        return Err(ServeError::Internal(err.clone()));
                    }
                    return Ok(state);
                } else if !reqs.contains_key(request_id) {
                    return Err(ServeError::RequestNotFound(request_id.to_string()));
                }
            }

            // Wait for the background step loop to produce new outputs.
            notified.await;
        }
    }

    // ------------------------------------------------------------------
    // Three-phase parallel detokenization
    // ------------------------------------------------------------------

    /// Phase 1: Under lock — accumulate tokens, timing, logprobs on each
    /// request and take out the detokenizer for parallel processing.
    ///
    /// Returns one entry per output. `Some(...)` contains the detokenizer
    /// and the data it needs; `None` means the output had no detokenizer
    /// (or the request was not found).
    fn process_outputs_phase1(
        requests: &mut HashMap<String, RequestState>,
        outputs: &[EngineCoreOutput],
    ) -> Vec<Option<DetokWork>> {
        outputs
            .iter()
            .map(|output| {
                let Some(req_state) = requests.get_mut(&output.request_id) else {
                    debug!("Output for unknown request {}, ignoring", output.request_id);
                    return None;
                };
                // Finished here already (a stop string matched; a non-streaming entry waits to be
                // collected): what the steps still in flight for it produced is not its output.
                if req_state.finish_reason.is_some() {
                    return None;
                }

                // Track output tokens.
                #[cfg(feature = "metrics")]
                {
                    let metrics = crate::metrics::VllmMetrics::global();
                    metrics
                        .output_tokens_total
                        .inc_by(output.new_token_ids.len() as u64);
                }

                // --- TTFT / ITL timing ---
                let now = Instant::now();
                if !output.new_token_ids.is_empty() {
                    if req_state.first_token_time.is_none() {
                        req_state.first_token_time = Some(now);
                        if std::env::var_os("SCRATCHY_TTFT_PHASE").is_some() {
                            eprintln!(
                                "[ttft-host] submit_to_first_token={:.2} ms  (queue + prefill forward + sample + detokenize)",
                                now.duration_since(req_state.submit_time).as_secs_f64() * 1e3,
                            );
                        }
                        #[cfg(feature = "metrics")]
                        {
                            let ttft = now.duration_since(req_state.submit_time).as_secs_f64();
                            crate::metrics::VllmMetrics::global()
                                .time_to_first_token_seconds
                                .observe(ttft);
                        }
                    } else if let Some(last) = req_state.last_token_time {
                        let itl = now.duration_since(last).as_secs_f64();
                        #[cfg(feature = "metrics")]
                        crate::metrics::VllmMetrics::global()
                            .inter_token_latency_seconds
                            .observe(itl);
                        req_state.itl_count += 1;
                        req_state.itl_sum += itl;
                        #[cfg(feature = "live-stats")]
                        req_state.itl_samples.push((itl * 1000.0) as f32);
                    }
                    req_state.last_token_time = Some(now);
                }

                // Accumulate tokens and logprobs.
                req_state.generated_token_ids.extend(&output.new_token_ids);
                if let Some(lps) = &output.new_logprobs {
                    req_state.logprobs.extend(lps.iter().cloned());
                }
                if let Some(plps) = &output.new_prompt_logprobs {
                    req_state.prompt_logprobs = Some(plps.clone());
                }

                // Update cached tokens.
                if output.num_cached_tokens > 0 {
                    req_state.num_cached_tokens = output.num_cached_tokens;
                }

                // Store pooler output if present.
                if output.pooler_output.is_some() {
                    req_state.pooler_output = output.pooler_output.clone();
                }

                // SHARE the detokenizer with the parallel phase; do NOT move it out. Taking it here
                // is what left the state publicly detokenizer-less for the whole of phase 2.
                let stop_terminated = output.finish_reason == Some(FinishReason::Stop);
                req_state.detokenizer.clone().map(|detok| {
                    (
                        output.new_token_ids.clone(),
                        stop_terminated,
                        output.finish_reason,
                        detok,
                    )
                })
            })
            .collect()
    }

    /// Phase 2: No lock — run detokenization in parallel using Rayon.
    ///
    /// Each work item runs `detok.update()` + `detok.get_next_output_text()`
    /// concurrently across a thread pool, removing tokenizer decode calls
    /// from the critical path.
    fn parallel_detokenize(work: Vec<Option<DetokWork>>) -> Vec<Option<DetokResult>> {
        work.into_par_iter()
            .map(|item| {
                item.map(|(new_token_ids, stop_terminated, finish_reason, detok)| {
                    // Uncontended: one work item per request, and this is the only mutator. A
                    // poisoned lock hands back its inner value rather than propagating — the
                    // worst case is one stale delta, against losing the whole sequence's text.
                    let mut detok = detok.lock().unwrap_or_else(|e| e.into_inner());
                    let detok_stop = detok.update(&new_token_ids, stop_terminated);
                    let is_finished = finish_reason.is_some() || detok_stop.is_some();
                    let delta_text = detok.get_next_output_text(is_finished, true);
                    (detok_stop, Some(delta_text))
                })
            })
            .collect()
    }

    /// End the requests `pick` selects, which the engine will produce no more output for, with
    /// `err`. A non-streaming request keeps its entry, now finished, for `poll_until_done` to
    /// collect. A streaming one has no such consumer: its stream gets a final `Abort` delta and its
    /// entry goes. Marked and left in place, a stream whose request the engine no longer holds
    /// hangs its client forever, and its entry keeps `has_requests` true — so the no-progress
    /// watchdog re-fires every `no_progress_timeout` without ever clearing it.
    fn fail_requests(
        requests: &mut HashMap<String, RequestState>,
        err: &str,
        pick: impl Fn(&str) -> bool,
    ) {
        let mut ended = Vec::new();
        for (id, req_state) in requests.iter_mut() {
            if req_state.finish_reason.is_some() || !pick(id) {
                continue;
            }
            req_state.finish_reason = Some(FinishReason::Abort);
            req_state.error = Some(err.to_string());
            if let Some(tx) = req_state.stream_tx.take() {
                let _ = tx.send(StreamDelta {
                    index: req_state.choice.index,
                    new_token_ids: Vec::new(),
                    text: None,
                    finish_reason: Some(FinishReason::Abort),
                    stop_reason: None,
                    logprobs: None,
                    tool_call_deltas: None,
                    reasoning: None,
                    usage: req_state.stream_usage(),
                });
                ended.push(id.clone());
            }
        }
        for id in ended {
            requests.remove(&id);
        }
    }

    /// Phase 3: Under lock — put detokenizers back, apply detok results,
    /// send streaming deltas, handle finish/cleanup.
    fn process_outputs_phase3(
        requests: &mut HashMap<String, RequestState>,
        outputs: Vec<EngineCoreOutput>,
        detok_results: Vec<Option<DetokResult>>,
    ) -> Vec<String> {
        // The requests this call finishes by a STOP STRING alone — the engine kept
        // decoding them, because a stop string is not an engine-side criterion
        // (engine_core's `check_stop_criteria` knows EOS, stop tokens and
        // max_tokens only). The caller must abort each one engine-side or it
        // decodes on to max_tokens while the client believes it done: the next
        // request is then admitted under a still-running row, sharing its steps
        // (a 3.3x first-token latency), and its eventual retirement changes the
        // batch's composition under whoever is still decoding (#711).
        let mut stop_string_finished: Vec<String> = Vec::new();
        for (output, detok_result) in outputs.into_iter().zip(detok_results) {
            let Some(req_state) = requests.get_mut(&output.request_id) else {
                // Already logged in Phase 1.
                continue;
            };
            // Finished here already: dropped in Phase 1.
            if req_state.finish_reason.is_some() {
                continue;
            }

            // NOTHING TO PUT BACK — the detokenizer never left. This arm used to reinstall it, and
            // when the state had already been removed it `continue`d above and dropped it forever.
            // No result here means no detokenizer, so the engine has no tokenizer
            // ([`SharedDetok`]): a step's text is its token ids' placeholders, as the
            // non-streaming response's is. Filled here, the one place that knows why the text is
            // missing: a delta's `text: None` means "no text" to every encoder, which is what a
            // reasoning-only or finish delta carries (#601: the chat encoder streamed a reasoning
            // delta's token ids as `<token_N>` content).
            let (detokenizer_stop, delta_text) = detok_result
                .unwrap_or_else(|| (None, Some(placeholder_text(&output.new_token_ids))));

            // Captured before `output` is consumed by the stream deltas below.
            // The engine's own finish for this step means it has already retired
            // the request — only a detokenizer-only finish leaves it running.
            if detokenizer_stop.is_some() && output.finish_reason.is_none() {
                stop_string_finished.push(output.request_id.clone());
            }

            let step_logprobs = output.new_logprobs.clone();

            // Determine finish/stop reason. Detokenizer stop takes priority.
            let is_finished = output.finish_reason.is_some() || detokenizer_stop.is_some();
            let (delta_finish_reason, delta_stop_reason) = if let Some(stop_str) = detokenizer_stop
            {
                (Some(FinishReason::Stop), Some(StopReason::String(stop_str)))
            } else {
                (output.finish_reason, output.stop_reason.clone())
            };

            // Send streaming delta if applicable.
            // Token counts, stamped on every delta this step emits. Read once
            // here so the literals below stay uniform: which of them fires
            // first depends on reasoning/tool-parser state, so there is no single
            // "first delta" to special-case. `num_cached_tokens` is already set
            // for this step (above), and the scheduler fixes it at admission.
            // `generated_token_ids` already holds this step's tokens (above).
            let usage = req_state.stream_usage();

            if let Some(tx) = &req_state.stream_tx {
                // Track accumulated text and token IDs for reasoning/tool parsing.
                if let Some(ref text) = delta_text {
                    req_state.accumulated_text.push_str(text);
                }
                req_state
                    .accumulated_token_ids
                    .extend_from_slice(&output.new_token_ids);

                // Phase 1: Reasoning parsing (runs before tool parsing).
                // If reasoning parser is active and reasoning hasn't ended yet,
                // route through reasoning parser first. Interleaved parsers
                // (Gemma 4's channels open after content too) keep routing
                // every delta; strictly reasoning-first formats stop here once
                // content has been seen.
                let (reasoning_delta, content_text_for_tools) =
                    if let Some(ref mut rp_state) = req_state.reasoning_parser_state {
                        if !req_state.reasoning_ended || rp_state.interleaved() {
                            // The final delta folds in the parser's
                            // end-of-stream flush: text it held back against
                            // a marker that never arrived must not vanish.
                            let rd = if let Some(ref text) = delta_text {
                                let prev_len = req_state.accumulated_token_ids.len()
                                    - output.new_token_ids.len();
                                let previous_ids = &req_state.accumulated_token_ids[..prev_len];
                                let current_ids = &req_state.accumulated_token_ids;
                                let previous_text = &req_state.accumulated_text
                                    [..req_state.accumulated_text.len() - text.len()];
                                let current_text = &req_state.accumulated_text;

                                let rd = rp_state.process_delta(
                                    previous_text,
                                    current_text,
                                    text,
                                    previous_ids,
                                    current_ids,
                                    &output.new_token_ids,
                                );
                                if is_finished {
                                    rd.merge(rp_state.flush())
                                } else {
                                    rd
                                }
                            } else if is_finished {
                                // Finished with no text (e.g. a stop-token-only
                                // final step): still release held-back text.
                                rp_state.flush()
                            } else {
                                ReasoningDelta::None
                            };

                            match rd {
                                ReasoningDelta::Reasoning(r) => {
                                    // Still in reasoning — emit reasoning delta.
                                    let reasoning_out = if req_state.include_reasoning {
                                        Some(r)
                                    } else {
                                        None
                                    };
                                    (reasoning_out, None)
                                }
                                ReasoningDelta::Content(c) => {
                                    // Reasoning ended previously, this is content.
                                    req_state.reasoning_ended = true;
                                    (None, Some(c))
                                }
                                ReasoningDelta::Split { reasoning, content } => {
                                    // Reasoning ends in this delta.
                                    req_state.reasoning_ended = true;
                                    let reasoning_out = if req_state.include_reasoning {
                                        reasoning
                                    } else {
                                        None
                                    };
                                    (reasoning_out, content)
                                }
                                ReasoningDelta::None => (None, None),
                            }
                        } else {
                            // Reasoning already ended — pass text to tool parser or as content.
                            (None, delta_text.clone())
                        }
                    } else {
                        // No reasoning parser — pass text through as-is.
                        (None, delta_text.clone())
                    };

                // Phase 2: Tool parsing on content portion (only after reasoning ends).
                if let Some(ref mut parser_state) = req_state.tool_parser_state {
                    if let Some(ref content_text) = content_text_for_tools {
                        // The tool parser reads the content alone: past the reasoning, which
                        // it would otherwise emit as content (`</think>` included).
                        let previous_len = req_state.accumulated_content.len();
                        req_state.accumulated_content.push_str(content_text);
                        let parser_result = parser_state.process_delta(
                            &req_state.accumulated_content[..previous_len],
                            &req_state.accumulated_content,
                            content_text,
                        );

                        // Whether an arm below already sent this output's finish on a
                        // content delta — the separate finish delta after the match
                        // must not duplicate it, and must fire when no arm did.
                        let mut finish_sent = false;

                        match parser_result {
                            ToolParserDelta::Content(content) => {
                                let attach_finish = is_finished && !req_state.tool_calls_emitted;
                                let delta = StreamDelta {
                                    index: req_state.choice.index,
                                    new_token_ids: output.new_token_ids.clone(),
                                    text: Some(content),
                                    finish_reason: if attach_finish {
                                        delta_finish_reason
                                    } else {
                                        None
                                    },
                                    stop_reason: if attach_finish {
                                        delta_stop_reason.clone()
                                    } else {
                                        None
                                    },
                                    logprobs: step_logprobs.clone(),
                                    tool_call_deltas: None,
                                    reasoning: reasoning_delta,
                                    usage,
                                };
                                let _ = tx.send(delta);
                                finish_sent |= attach_finish;
                            }
                            ToolParserDelta::ToolCalls(tool_deltas) => {
                                // Emit reasoning delta separately if present.
                                if let Some(reasoning) = reasoning_delta {
                                    let delta = StreamDelta {
                                        index: req_state.choice.index,
                                        new_token_ids: vec![],
                                        text: None,
                                        finish_reason: None,
                                        stop_reason: None,
                                        logprobs: None,
                                        tool_call_deltas: None,
                                        reasoning: Some(reasoning),
                                        usage,
                                    };
                                    let _ = tx.send(delta);
                                }
                                let tool_deltas = req_state.tool_calls.admit(tool_deltas);
                                if tool_deltas.is_empty() {
                                    // Held for a name, or dropped — don't emit.
                                } else {
                                    req_state.tool_calls_emitted = true;
                                    let delta = StreamDelta {
                                        index: req_state.choice.index,
                                        new_token_ids: output.new_token_ids.clone(),
                                        text: None,
                                        finish_reason: None,
                                        stop_reason: None,
                                        logprobs: step_logprobs.clone(),
                                        tool_call_deltas: Some(tool_deltas),
                                        reasoning: None,
                                        usage,
                                    };
                                    let _ = tx.send(delta);
                                }
                            }
                            ToolParserDelta::None => {
                                // Buffering in tool parser — but still emit reasoning if present.
                                if let Some(reasoning) = reasoning_delta {
                                    let delta = StreamDelta {
                                        index: req_state.choice.index,
                                        new_token_ids: output.new_token_ids.clone(),
                                        text: None,
                                        finish_reason: None,
                                        stop_reason: None,
                                        logprobs: step_logprobs.clone(),
                                        tool_call_deltas: None,
                                        reasoning: Some(reasoning),
                                        usage,
                                    };
                                    let _ = tx.send(delta);
                                }
                            }
                        }

                        // Send the finish delta separately whenever this output ends
                        // the request and no arm above already carried the finish.
                        // The arms above attach it inline only to a content delta
                        // with no tool calls emitted — a final increment the parser
                        // returns `None` for reaches none of them. That is the
                        // normal EOS ending for a tool-parsed reply: the terminal
                        // token decodes to a control token (gemma 4:
                        // `<end_of_turn>`) the parser strips to empty, so the last
                        // `process_delta` says "buffering" and the request finishes
                        // with no finish-bearing delta ever sent. The stream would
                        // end with no `message_delta`/`message_stop`, and every SSE
                        // client reads the response as truncated (Claude Code:
                        // "Streaming response ended before any complete data was
                        // received").
                        if is_finished && !finish_sent {
                            // The step's own reason, a tool call's included: the stream encoder
                            // makes a tool call's `Stop` `tool_calls`, and a `Length` stays
                            // `length` (the call was cut off).
                            let finish_delta = StreamDelta {
                                index: req_state.choice.index,
                                new_token_ids: vec![],
                                text: None,
                                finish_reason: delta_finish_reason,
                                stop_reason: delta_stop_reason.clone(),
                                logprobs: None,
                                tool_call_deltas: None,
                                reasoning: None,
                                usage,
                            };
                            let _ = tx.send(finish_delta);
                        }
                    } else if is_finished {
                        // No content text but finished — send finish delta, with the step's own
                        // reason (see the finish above).
                        let delta = StreamDelta {
                            index: req_state.choice.index,
                            new_token_ids: output.new_token_ids.clone(),
                            text: None,
                            finish_reason: delta_finish_reason,
                            stop_reason: delta_stop_reason.clone(),
                            logprobs: step_logprobs.clone(),
                            tool_call_deltas: None,
                            reasoning: reasoning_delta,
                            usage,
                        };
                        let _ = tx.send(delta);
                    } else if reasoning_delta.is_some() {
                        // Only reasoning, no content for tool parser yet.
                        let delta = StreamDelta {
                            index: req_state.choice.index,
                            new_token_ids: output.new_token_ids.clone(),
                            text: None,
                            finish_reason: None,
                            stop_reason: None,
                            logprobs: step_logprobs.clone(),
                            tool_call_deltas: None,
                            reasoning: reasoning_delta,
                            usage,
                        };
                        let _ = tx.send(delta);
                    }
                } else {
                    // No tool parsing — send content (or reasoning) directly.
                    let text_out = if req_state.reasoning_parser_state.is_some() {
                        content_text_for_tools
                    } else {
                        delta_text
                    };
                    if text_out.is_some() || reasoning_delta.is_some() || is_finished {
                        let delta = StreamDelta {
                            index: req_state.choice.index,
                            new_token_ids: output.new_token_ids,
                            text: text_out,
                            finish_reason: delta_finish_reason,
                            stop_reason: delta_stop_reason.clone(),
                            logprobs: step_logprobs.clone(),
                            tool_call_deltas: None,
                            reasoning: reasoning_delta,
                            usage,
                        };
                        let _ = tx.send(delta);
                    }
                }
            }

            // Update request state finish/stop reason.
            if is_finished && req_state.finish_reason.is_none() {
                req_state.finish_reason = delta_finish_reason;
                req_state.stop_reason = delta_stop_reason.or(output.stop_reason);
            }

            // Log and clean up when done.
            if is_finished {
                let now = Instant::now();
                let total_latency = now.duration_since(req_state.submit_time).as_secs_f64();
                let ttft_ms = req_state
                    .first_token_time
                    .map(|t| t.duration_since(req_state.submit_time).as_secs_f64() * 1000.0);
                let avg_itl_ms = if req_state.itl_count > 0 {
                    Some(req_state.itl_sum / req_state.itl_count as f64 * 1000.0)
                } else {
                    None
                };
                let prompt_tokens = req_state.num_prompt_tokens;
                let completion_tokens = req_state.generated_token_ids.len() as u32;

                let finish_str = req_state
                    .finish_reason
                    .map(|r| r.to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                let stop_str = req_state
                    .stop_reason
                    .as_ref()
                    .map(|sr| match sr {
                        StopReason::Token(id) => format!("token:{id}"),
                        StopReason::String(s) => format!("string:{s}"),
                    })
                    .unwrap_or_default();

                tracing::info!(
                    request_id = %output.request_id,
                    prompt_tokens = prompt_tokens,
                    completion_tokens = completion_tokens,
                    finish_reason = finish_str,
                    stop_reason = stop_str,
                    latency_ms = format!("{:.1}", total_latency * 1000.0),
                    ttft_ms = ttft_ms.map(|v| format!("{v:.1}")).unwrap_or_else(|| "-".into()),
                    avg_itl_ms = avg_itl_ms.map(|v| format!("{v:.1}")).unwrap_or_else(|| "-".into()),
                    "request finished"
                );

                #[cfg(feature = "metrics")]
                {
                    let metrics = crate::metrics::VllmMetrics::global();
                    metrics.request_latency_seconds.observe(total_latency);
                    metrics.requests_active.dec();
                    metrics.requests_success_total.inc();
                }
                // Drop stream sender; capture whether this was a streaming request.
                let was_streaming = req_state.stream_tx.take().is_some();

                // Streaming requests are cleaned up here — there's no poll_until_done
                // consumer. Non-streaming requests stay for poll_until_done to remove.
                if was_streaming {
                    requests.remove(&output.request_id);
                }
            }
        }
        stop_string_finished
    }

    /// Convert a chat completion request to an engine core request.
    fn chat_to_engine_request(
        &self,
        request_id: &str,
        request: &protocol::ChatCompletionRequest,
        sampling_params: &SamplingParams,
    ) -> ServeResult<EngineCoreRequest> {
        // Build text from chat messages — using chat template if available.
        let text = if let Some(template) = &self.chat_template {
            // Merge default + per-request chat template kwargs.
            let merged_kwargs = merge_chat_template_kwargs(
                self.default_chat_template_kwargs.as_ref(),
                request.chat_template_kwargs.as_ref(),
            );

            template.render_chat(
                &request.messages,
                request.tools.as_deref(),
                merged_kwargs.as_ref(),
                true,
            )?
        } else {
            // Fallback: concatenate messages with newlines.
            let mut text = String::new();
            for msg in &request.messages {
                if let Some(content) = &msg.content
                    && let Some(s) = content.as_str()
                {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(s);
                }
            }
            text
        };
        // Numbered-image-tag normalization (LocateAnything-class
        // arches): the checkpoint's chat template renders image parts
        // as literal `<image-N>` text, which would tokenize to several
        // ordinary ids instead of the single placeholder token the
        // expansion pass scans for. Replace each tag with the marker
        // token's text pre-tokenization (mirrors mlx-vlm's processor
        // regex). Declared per-arch; None for every other family.
        #[cfg(feature = "multimodal")]
        let text = match self
            .mm_processor
            .as_ref()
            .and_then(|p| p.metadata.numbered_image_tag_marker)
        {
            Some(marker) if text.contains("<image-") => {
                scratchy_vision::preprocess::replace_numbered_image_tags(&text, marker)
            }
            _ => text,
        };

        let (text, marks) = take_cache_breakpoints(text);

        // Tokenize the text, or fall back to byte-value IDs.
        let mut cache_breakpoints = Vec::new();
        #[allow(unused_mut)]
        let mut token_ids = if let Some(tok) = &self.tokenizer {
            if text.is_empty() {
                vec![]
            } else {
                let (ids, through) = tok.encode_through(&text, true, &marks)?;
                cache_breakpoints = through;
                ids
            }
        } else if text.is_empty() {
            vec![0] // BOS placeholder
        } else {
            text.as_bytes().iter().map(|&b| b as u32).collect()
        };

        // Extract images from message content arrays and build multimodal data.
        #[cfg(feature = "multimodal")]
        let mm_data = if self.mm_processor.is_some() {
            self.extract_images_from_messages(&request.messages, &mut token_ids)?
        } else {
            None
        };
        #[cfg(not(feature = "multimodal"))]
        let mm_data = None;
        // An image's placeholder expands past the text's token count: no breakpoint stands then.
        if mm_data.is_some() {
            cache_breakpoints.clear();
        }

        Ok(EngineCoreRequest {
            request_id: request_id.to_string(),
            prompt_token_ids: Some(token_ids),
            sampling_params: Some(sampling_params.clone()),
            arrival_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
            client_index: 0,
            priority: request.priority,
            cache_salt: request.cache_salt.clone(),
            data_parallel_rank: None,
            is_pooling: false,
            mm_data,
            block_annotations: None,
            seal: false,
            volatile: false,
            cache_breakpoints,
        })
    }

    /// Normalize a completion prompt to a list of token ID sequences.
    fn tokenize_completion_prompts(
        &self,
        request: &protocol::CompletionRequest,
    ) -> ServeResult<Vec<Vec<u32>>> {
        let add_special = request.add_special_tokens;
        match &request.prompt {
            Some(protocol::CompletionPrompt::Single(text)) => {
                Ok(vec![self.tokenize_text(text, add_special)?])
            }
            Some(protocol::CompletionPrompt::Multiple(texts)) => texts
                .iter()
                .map(|t| self.tokenize_text(t, add_special))
                .collect(),
            Some(protocol::CompletionPrompt::TokenIds(ids)) => Ok(vec![ids.clone()]),
            Some(protocol::CompletionPrompt::MultipleTokenIds(vv)) => Ok(vv.clone()),
            None => Ok(vec![vec![0]]),
        }
    }

    /// Tokenize a text string, using the tokenizer if available.
    fn tokenize_text(&self, text: &str, add_special: bool) -> ServeResult<Vec<u32>> {
        if let Some(tok) = &self.tokenizer {
            if text.is_empty() {
                Ok(vec![])
            } else {
                Ok(tok.encode(text, add_special)?)
            }
        } else if text.is_empty() {
            Ok(vec![0])
        } else {
            Ok(text.as_bytes().iter().map(|&b| b as u32).collect())
        }
    }

    /// Convert a completion request to an engine core request.
    #[cfg(test)]
    fn completion_to_engine_request(
        &self,
        request_id: &str,
        request: &protocol::CompletionRequest,
        sampling_params: &SamplingParams,
    ) -> ServeResult<EngineCoreRequest> {
        let token_ids = match &request.prompt {
            Some(protocol::CompletionPrompt::Single(text)) => {
                if let Some(tok) = &self.tokenizer {
                    tok.encode(text, false)?
                } else {
                    text.as_bytes().iter().map(|&b| b as u32).collect()
                }
            }
            Some(protocol::CompletionPrompt::TokenIds(ids)) => ids.clone(),
            Some(protocol::CompletionPrompt::Multiple(texts)) => {
                let text = texts.first().map(|t| t.as_str()).unwrap_or("");
                if let Some(tok) = &self.tokenizer {
                    tok.encode(text, false)?
                } else {
                    text.as_bytes().iter().map(|&b| b as u32).collect()
                }
            }
            Some(protocol::CompletionPrompt::MultipleTokenIds(id_lists)) => {
                id_lists.first().cloned().unwrap_or_else(|| vec![0])
            }
            None => vec![0],
        };

        Ok(EngineCoreRequest {
            request_id: request_id.to_string(),
            prompt_token_ids: Some(token_ids),
            sampling_params: Some(sampling_params.clone()),
            arrival_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
            client_index: 0,
            priority: request.priority,
            cache_salt: request.cache_salt.clone(),
            data_parallel_rank: None,
            is_pooling: false,
            mm_data: None,
            block_annotations: None,
            seal: false,
            volatile: false,
            cache_breakpoints: Vec::new(),
        })
    }

    /// Extract image URLs from chat message content arrays, decode/preprocess
    /// images, expand image placeholder tokens, and build `MultimodalData`.
    ///
    /// Returns `None` if no images are present.
    #[cfg(feature = "multimodal")]
    fn extract_images_from_messages(
        &self,
        messages: &[protocol::ChatCompletionMessageParam],
        token_ids: &mut Vec<u32>,
    ) -> ServeResult<Option<MultimodalData>> {
        let processor = match &self.mm_processor {
            Some(p) => p.clone(),
            None => return Ok(None),
        };
        let image_token_id = processor.image_token_id;

        let mut images: Vec<ImageData> = Vec::new();

        // Walk messages looking for content arrays with image_url parts.
        for msg in messages {
            if let Some(content) = &msg.content
                && let Some(parts) = content.as_array()
            {
                for part in parts {
                    if part.get("type").and_then(|t| t.as_str()) == Some("image_url")
                        && let Some(image_url_obj) = part.get("image_url")
                    {
                        let url = image_url_obj
                            .get("url")
                            .and_then(|u| u.as_str())
                            .unwrap_or("");
                        if url.is_empty() {
                            continue;
                        }

                        // Decode the image.
                        let img_bytes = if url.starts_with("data:") {
                            scratchy_vision::preprocess::decode_data_uri(url).map_err(|e| {
                                ServeError::Validation(format!("failed to decode data URI: {e}"))
                            })?
                        } else {
                            // For now, only data URIs are supported.
                            // HTTP URL download would require async reqwest.
                            return Err(ServeError::Validation(
                                "HTTP image URLs not yet supported; use base64 data URIs".into(),
                            ));
                        };

                        let dyn_image = scratchy_vision::preprocess::decode_image(&img_bytes)
                            .map_err(|e| {
                                ServeError::Validation(format!("failed to decode image: {e}"))
                            })?;

                        let image_data = processor.preprocess(&dyn_image);
                        images.push(image_data);
                    }
                }
            }
        }

        if images.is_empty() {
            return Ok(None);
        }

        // Per-image expanded placeholder count. The processor's
        // `tokens_for_image` honors per-arch policy: `FromConfig`
        // returns the static count (Gemma3-MM, SigLIP-class),
        // `PerImageGrid` returns the per-image grid product
        // (Qwen2-VL family — varies per aspect ratio).
        let per_image_counts: Vec<usize> = images
            .iter()
            .map(|img| processor.tokens_for_image(img.height, img.width))
            .collect();

        // Expand image placeholders in token IDs per the arch's
        // declared placeholder policy. Returned ranges cover only the
        // positions the vision splice writes into (RepeatMarker: all
        // expanded copies; BoiSoftEoiWrap: just the middle soft tokens).
        let placeholders = scratchy_vision::preprocess::expand_placeholders_by_policy(
            token_ids,
            image_token_id,
            &per_image_counts,
            &processor.metadata.placeholder_policy,
        );

        if placeholders.len() != images.len() {
            debug!(
                "Image count mismatch: {} images but {} placeholders in token sequence",
                images.len(),
                placeholders.len()
            );
        }

        Ok(Some(MultimodalData {
            images,
            image_placeholders: placeholders,
        }))
    }

    /// Admission guard: reject a prompt that cannot fit in the model's context
    /// window.
    ///
    /// `resolve_max_tokens` only clamps the completion budget via
    /// `saturating_sub`, so an over-length prompt silently resolves to a
    /// `max_tokens` of 0 and is admitted into the scheduler's WAITING queue,
    /// where it can never be placed (its KV demand exceeds total capacity) and
    /// the engine busy-spins re-hashing it forever (the "scr launch claude"
    /// 100%-CPU meltdown). Reject up front with a vLLM-parity 400, mirroring
    /// vLLM's `_validate_model_input` length check.
    fn check_prompt_fits(&self, num_prompt_tokens: usize) -> ServeResult<()> {
        if num_prompt_tokens >= self.max_model_len {
            return Err(ServeError::Validation(format!(
                "This model's maximum context length is {} tokens, however you \
                 requested {} input tokens. Please reduce the length of the \
                 messages.",
                self.max_model_len, num_prompt_tokens
            )));
        }
        Ok(())
    }

    /// Resolve `max_tokens` to fit within `max_model_len - prompt_len`.
    ///
    /// - If `None`, sets it to the remaining capacity.
    /// - If `Some(val)`, caps it at the remaining capacity.
    fn resolve_max_tokens(&self, sampling_params: &mut SamplingParams, num_prompt_tokens: usize) {
        let remaining = self.max_model_len.saturating_sub(num_prompt_tokens);
        match sampling_params.max_tokens {
            None => sampling_params.max_tokens = Some(remaining as u32),
            Some(val) => sampling_params.max_tokens = Some(val.min(remaining as u32)),
        }
    }

    /// Build SamplingParams from a chat completion request.
    fn build_sampling_params_from_chat(
        &self,
        request: &protocol::ChatCompletionRequest,
    ) -> ServeResult<SamplingParams> {
        let tools: Vec<&str> = (request.tools.iter().flatten())
            .map(|t| t.function.name.as_str())
            .collect();
        check_tool_choice(&request.tool_choice, &tools)?;
        let stop = match &request.stop {
            Some(protocol::StopCondition::Single(s)) => vec![s.clone()],
            Some(protocol::StopCondition::Multiple(v)) => v.clone(),
            None => vec![],
        };
        // Resolve max_tokens: max_completion_tokens takes priority over max_tokens.
        let max_tokens = request.max_completion_tokens.or(request.max_tokens);

        // Chat API: logprobs is bool, top_logprobs is count.
        let logprobs = if request.logprobs == Some(true) {
            Some(request.top_logprobs.unwrap_or(0) as i32)
        } else {
            None
        };

        // Parse response_format / guided_regex / guided_grammar (mutually exclusive).
        let guided_grammar = resolve_guided_grammar(
            &request.response_format,
            &request.guided_regex,
            &request.guided_grammar,
        )?;

        // Tokenize bad_words strings into token sequences.
        let bad_words_token_ids = self.tokenize_bad_words(&request.bad_words)?;

        // Some tool parsers (Gemma 4's `<|tool_call>` / `<|"|>`) key on real
        // added tokens that `skip_special_tokens` (default true) would strip
        // before the parser ever runs. The same holds for reasoning parsers
        // whose thinking markers are added tokens (Gemma 4's thought
        // channels) — including on tool-less requests, where the reasoning
        // parser is the only consumer. When such a parser is active, keep
        // special tokens in the output so the delimiters survive
        // detokenization. Requests without a special-token-hungry parser and
        // plain-text parsers (Hermes, Granite 3.1, …) are unaffected.
        let has_tools = request.tools.as_ref().is_some_and(|t| !t.is_empty());
        let keep_special_tokens = (has_tools
            && self
                .tool_parser
                .as_ref()
                .is_some_and(|p| p.requires_special_tokens()))
            || self
                .reasoning_parser
                .as_ref()
                .is_some_and(|p| p.requires_special_tokens());

        // Unset request fields fall back to the model's
        // generation_config.json defaults (Python vLLM
        // `generation_config="auto"`), then to neutral.
        let d = &self.generation_defaults;
        Ok(SamplingParams {
            temperature: request.temperature.or(d.temperature).unwrap_or(1.0),
            top_p: request.top_p.or(d.top_p).unwrap_or(1.0),
            top_k: request.top_k.or(d.top_k).unwrap_or(0),
            min_p: request.min_p.or(d.min_p).unwrap_or(0.0),
            max_tokens: max_tokens.or(d.max_new_tokens),
            repetition_penalty: request
                .repetition_penalty
                .or(d.repetition_penalty)
                .unwrap_or(1.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            seed: request.seed.map(|s| s as u64),
            ignore_eos: request.ignore_eos,
            min_tokens: request.min_tokens,
            stop,
            stop_token_ids: request.stop_token_ids.clone(),
            include_stop_str_in_output: request.include_stop_str_in_output,
            skip_special_tokens: request.skip_special_tokens && !keep_special_tokens,
            logprobs,
            prompt_logprobs: request.prompt_logprobs.map(|n| n as i32),
            logit_bias: parse_logit_bias(&request.logit_bias),
            guided_grammar,
            allowed_token_ids: request.allowed_token_ids.clone(),
            bad_words_token_ids,
            ..Default::default()
        })
    }

    /// Build SamplingParams from a completion request.
    fn build_sampling_params_from_completion(
        &self,
        request: &protocol::CompletionRequest,
    ) -> ServeResult<SamplingParams> {
        let stop = match &request.stop {
            Some(protocol::StopCondition::Single(s)) => vec![s.clone()],
            Some(protocol::StopCondition::Multiple(v)) => v.clone(),
            None => vec![],
        };
        // Completion API: logprobs is directly the count.
        let logprobs = request.logprobs.map(|n| n as i32);

        // guided_regex / guided_grammar (completions have no response_format).
        let guided_grammar =
            resolve_guided_grammar(&None, &request.guided_regex, &request.guided_grammar)?;

        // Tokenize bad_words strings into token sequences.
        let bad_words_token_ids = self.tokenize_bad_words(&request.bad_words)?;

        let d = &self.generation_defaults;
        Ok(SamplingParams {
            temperature: request.temperature.or(d.temperature).unwrap_or(1.0),
            top_p: request.top_p.or(d.top_p).unwrap_or(1.0),
            top_k: request.top_k.or(d.top_k).unwrap_or(0),
            min_p: request.min_p.or(d.min_p).unwrap_or(0.0),
            max_tokens: request.max_tokens.or(d.max_new_tokens),
            repetition_penalty: request
                .repetition_penalty
                .or(d.repetition_penalty)
                .unwrap_or(1.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            seed: request.seed.map(|s| s as u64),
            ignore_eos: request.ignore_eos,
            min_tokens: request.min_tokens,
            stop,
            stop_token_ids: request.stop_token_ids.clone(),
            include_stop_str_in_output: request.include_stop_str_in_output,
            skip_special_tokens: request.skip_special_tokens,
            logprobs,
            prompt_logprobs: request.prompt_logprobs.map(|n| n as i32),
            logit_bias: parse_logit_bias(&request.logit_bias),
            guided_grammar,
            allowed_token_ids: request.allowed_token_ids.clone(),
            bad_words_token_ids,
            ..Default::default()
        })
    }

    /// Tokenize bad_words strings into token ID sequences.
    fn tokenize_bad_words(
        &self,
        bad_words: &Option<Vec<String>>,
    ) -> ServeResult<Option<Vec<Vec<u32>>>> {
        let Some(words) = bad_words else {
            return Ok(None);
        };
        if words.is_empty() {
            return Ok(None);
        }
        let Some(ref tokenizer) = self.tokenizer else {
            return Ok(None);
        };
        let mut result = Vec::with_capacity(words.len());
        for word in words {
            let ids = tokenizer.encode(word, false)?;
            if !ids.is_empty() {
                result.push(ids);
            }
        }
        if result.is_empty() {
            Ok(None)
        } else {
            Ok(Some(result))
        }
    }
}

// ---------------------------------------------------------------------------
// Async scheduling types and executor thread
// ---------------------------------------------------------------------------

use scratchy_serving_engine::error::EngineResult;

/// Work item sent from the step loop to the executor thread.
enum ExecutorWork {
    /// Execute a model forward pass.
    Execute(Box<scratchy_serving_scheduler::scheduler::output::SchedulerOutput>),
    /// Compute embeddings (reply goes back via the oneshot).
    Embed(
        Vec<Vec<u32>>,
        tokio::sync::oneshot::Sender<ServeResult<Vec<Vec<f32>>>>,
    ),
    /// Put the executor to sleep.
    Sleep(u32, oneshot::Sender<Result<(), String>>),
    /// Wake the executor from sleep.
    WakeUp(Option<Vec<String>>, oneshot::Sender<Result<(), String>>),
    /// Query whether the executor is sleeping.
    IsSleeping(oneshot::Sender<bool>),
    /// Shut down the executor.
    Shutdown,
}

/// Result sent back from the executor thread to the step loop.
enum ExecutorResult {
    /// Model forward pass result, together with the scheduler output that
    /// produced it (returned so the step loop can finalize without cloning).
    Model(
        EngineResult<ModelRunnerOutput>,
        Box<scratchy_serving_scheduler::scheduler::output::SchedulerOutput>,
    ),
}

/// Owns the async executor thread's channel sender and join handle, so the
/// thread's teardown is waited out wherever the step loop's future ends —
/// completed or dropped mid-await (a SIGTERM ends the process through
/// `serve` returning, which can drop the loop's task before its own tail
/// runs). [`Drop`] closes the channel first — the executor thread's cue to
/// run `executor.shutdown()` (see `executor_thread_loop`) — then joins, so
/// the worker shutdown the thread performs (the metal aligned-sidecar
/// drain, the GPU residency release) completes before whatever dropped this
/// proceeds. The sender is `Option` only so `Drop` can take it; the thread
/// handle likewise. Mirrors the `InprocClient` pipeline's own
/// drop-sender-then-join teardown in `core_client.rs`.
struct ExecutorPipeline {
    tx: Option<tokio::sync::mpsc::Sender<ExecutorWork>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl ExecutorPipeline {
    /// The sender, for as long as the pipeline is alive.
    fn tx(&self) -> &tokio::sync::mpsc::Sender<ExecutorWork> {
        self.tx
            .as_ref()
            .expect("executor pipeline sender already dropped")
    }
}

impl Drop for ExecutorPipeline {
    fn drop(&mut self) {
        // Close the channel before joining: the executor thread exits its
        // work loop on a closed channel (running its teardown), and would
        // otherwise sit in `blocking_recv` forever waiting for this sender.
        drop(self.tx.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The executor thread's main loop.
///
/// Receives work items from the step loop, executes them, and sends results
/// back. Runs on a dedicated OS thread so GPU work doesn't block tokio.
fn executor_thread_loop(
    mut executor: Box<dyn Executor>,
    mut rx: tokio::sync::mpsc::Receiver<ExecutorWork>,
    tx: tokio::sync::mpsc::Sender<ExecutorResult>,
) {
    while let Some(work) = rx.blocking_recv() {
        match work {
            ExecutorWork::Execute(sched) => {
                let result = executor.execute_model(&sched);
                // Send the sched back so the step loop can finalize without cloning.
                if tx
                    .blocking_send(ExecutorResult::Model(result, sched))
                    .is_err()
                {
                    break; // Step loop dropped its receiver.
                }
            }
            ExecutorWork::Embed(seqs, reply) => {
                let result = executor
                    .embed(seqs)
                    .map_err(|e| ServeError::Engine(e.to_string()));
                let _ = reply.send(result);
            }
            ExecutorWork::Sleep(level, reply) => {
                let result = executor.sleep(level).map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            ExecutorWork::WakeUp(tags, reply) => {
                let result = executor.wake_up(tags.as_deref()).map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            ExecutorWork::IsSleeping(reply) => {
                let _ = reply.send(executor.is_sleeping());
            }
            // The teardown itself runs once, below the loop: `Shutdown`
            // breaks out of it, and a channel that closed without `Shutdown`
            // must reach the same call.
            ExecutorWork::Shutdown => break,
        }
    }
    // Shut down on EVERY exit from the loop, not just the explicit
    // `Shutdown` work item. The other exit — the channel closing — is the
    // common way a SIGTERM ends this server: the step loop's task is
    // dropped when the tokio runtime tears down after `serve` returns,
    // racing the loop's own wake-and-break (its tail sends `Shutdown`
    // only if it gets to run first), so the executor thread used to see a
    // closed channel and exit silently, skipping `worker.shutdown()`.
    // Measured consequence: the metal worker's aligned-sidecar drain never
    // ran — the background writer was killed mid-write and its build
    // discarded, so every launch re-paid the cold realign copy (#411) —
    // and the wired GPU residency was left to the kernel's intermittent
    // reclaim instead of the deterministic release this path promises.
    // With the teardown here, the dropping thread's `InprocClient` join
    // (which is what closed the channel) waits out the drain before the
    // process can exit. The sibling `executor_bg_loop` has always run
    // this same tail on its own channel-close path.
    executor.shutdown();
}

/// Route step outputs to request states and update metrics.
///
/// Returns the requests this pass finishes by a stop string alone — the engine
/// is still decoding each of those (a stop string is not an engine-side
/// criterion), and the caller must abort them engine-side (#711).
async fn route_step_outputs(
    requests: &Mutex<HashMap<String, RequestState>>,
    outputs: scratchy_serving_engine::engine_core::StepOutputs,
) -> Vec<String> {
    let mut stop_string_finished: Vec<String> = Vec::new();
    for (_, engine_outputs) in outputs {
        // Update scheduler gauges from stats.
        #[cfg(feature = "metrics")]
        if let Some(stats) = &engine_outputs.scheduler_stats {
            let m = crate::metrics::VllmMetrics::global();
            m.num_requests_running.set(stats.num_running_reqs as f64);
            m.num_requests_waiting.set(stats.num_waiting_reqs as f64);
            m.kv_cache_usage_perc.set(stats.kv_cache_usage);
            m.gpu_cache_blocks_used
                .set(stats.gpu_cache_blocks_used as i64);
            m.gpu_cache_blocks_total
                .set(stats.gpu_cache_blocks_total as i64);
            m.prefix_cache_blocks.set(stats.num_cached_blocks as i64);
            if let Some(sd) = &stats.spec_decode_stats {
                m.spec_decode_num_drafts.inc_by(sd.num_drafts as u64);
                m.spec_decode_num_draft_tokens
                    .inc_by(sd.num_draft_tokens as u64);
                m.spec_decode_num_accepted_tokens
                    .inc_by(sd.num_accepted_tokens as u64);
            }
        }

        if !engine_outputs.outputs.is_empty() {
            // Phase 1: accumulate tokens, take detokenizers (brief lock).
            let detok_work = {
                let mut reqs = requests.lock().await;
                AsyncEngine::process_outputs_phase1(&mut reqs, &engine_outputs.outputs)
            };

            // Phase 2: parallel detokenize (no lock held).
            // RAYON FROM AN ASYNC FN BLOCKS A TOKIO WORKER, and the block scales with batch size —
            // which is what stretched the window this phase used to leave open. `block_in_place`
            // hands the worker's other tasks to a sibling thread for the duration.
            let detok_results =
                tokio::task::block_in_place(|| AsyncEngine::parallel_detokenize(detok_work));

            // Phase 3: apply detok results, send streams (brief lock).
            {
                let mut reqs = requests.lock().await;
                stop_string_finished.extend(AsyncEngine::process_outputs_phase3(
                    &mut reqs,
                    engine_outputs.outputs,
                    detok_results,
                ));
            }
        }
    }
    stop_string_finished
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert API-style logit_bias (string keys, f64 values) to sampler format.
///
/// OpenAI API uses string token IDs as keys; we parse them to u32.
/// Invalid keys are silently ignored.
/// Truncate prompt token IDs from the left, keeping the last N tokens.
///
/// `truncate` values: `None` → no-op, `Some(-1)` → no-op (model max),
/// `Some(n)` where n >= 1 → keep last n tokens.
fn truncate_prompt(tokens: &mut Vec<u32>, truncate: Option<i64>) -> ServeResult<()> {
    let Some(n) = truncate else { return Ok(()) };
    if n == -1 {
        // -1 means "use model's max input length", which is already enforced
        // by resolve_max_tokens — nothing to do here.
        return Ok(());
    }
    if n < 1 {
        return Err(ServeError::Validation(format!(
            "truncate_prompt_tokens must be >= 1 or -1, got {n}"
        )));
    }
    let n = n as usize;
    if n < tokens.len() {
        let start = tokens.len() - n;
        tokens.drain(..start);
    }
    Ok(())
}

fn parse_logit_bias(
    api_bias: &Option<std::collections::HashMap<String, f64>>,
) -> Option<std::collections::HashMap<u32, f32>> {
    let bias = api_bias.as_ref()?;
    if bias.is_empty() {
        return None;
    }
    let parsed: std::collections::HashMap<u32, f32> = bias
        .iter()
        .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|tid| (tid, v as f32)))
        .collect();
    if parsed.is_empty() {
        None
    } else {
        Some(parsed)
    }
}

/// Parse `response_format` from an API request into a `GuidedGrammar`.
///
/// - `type: "text"` → `None` (no constraint)
/// - `type: "json_object"` → `Some(GuidedGrammar::Json)`
/// - `type: "json_schema"` → `Some(GuidedGrammar::JsonSchema { schema })`
/// - `type: "structural_tag"` → `Some(GuidedGrammar::StructuralTag { spec })`
fn parse_response_format(
    rf: &Option<protocol::ResponseFormat>,
) -> ServeResult<Option<GuidedGrammar>> {
    let Some(rf) = rf else {
        return Ok(None);
    };
    match rf {
        protocol::ResponseFormat::LegacyStructuralTag(st) => {
            // Serialize the full object as JSON spec string (Python does the same).
            let spec = serde_json::to_string(st).map_err(|e| {
                ServeError::Validation(format!("failed to serialize structural_tag: {e}"))
            })?;
            Ok(Some(GuidedGrammar::StructuralTag { spec }))
        }
        protocol::ResponseFormat::NewStructuralTag(st) => {
            let spec = serde_json::to_string(st).map_err(|e| {
                ServeError::Validation(format!("failed to serialize structural_tag: {e}"))
            })?;
            Ok(Some(GuidedGrammar::StructuralTag { spec }))
        }
        protocol::ResponseFormat::Standard(rf) => match rf.format_type.as_str() {
            "text" => Ok(None),
            "json_object" => Ok(Some(GuidedGrammar::Json)),
            "json_schema" => {
                let schema = rf
                    .json_schema
                    .as_ref()
                    .and_then(|js| js.json_schema.clone())
                    .ok_or_else(|| {
                        ServeError::Validation(
                            "response_format type 'json_schema' requires json_schema.schema".into(),
                        )
                    })?;
                Ok(Some(GuidedGrammar::JsonSchema { schema }))
            }
            other => Err(ServeError::Validation(format!(
                "unsupported response_format type: {other:?}. Must be 'text', 'json_object', 'json_schema', or 'structural_tag'",
            ))),
        },
    }
}

/// Resolve `response_format`, `guided_regex`, and `guided_grammar` into a single `GuidedGrammar`.
///
/// These are mutually exclusive — returns an error if more than one is set.
fn resolve_guided_grammar(
    response_format: &Option<protocol::ResponseFormat>,
    guided_regex: &Option<String>,
    guided_grammar: &Option<String>,
) -> ServeResult<Option<GuidedGrammar>> {
    let from_rf = parse_response_format(response_format)?;
    let from_regex = guided_regex.as_ref().map(|pattern| GuidedGrammar::Regex {
        pattern: pattern.clone(),
    });
    let from_ebnf = guided_grammar.as_ref().map(|grammar| GuidedGrammar::Ebnf {
        grammar: grammar.clone(),
    });

    let options: Vec<_> = [from_rf, from_regex, from_ebnf]
        .into_iter()
        .flatten()
        .collect();
    match options.len() {
        0 => Ok(None),
        1 => Ok(Some(options.into_iter().next().unwrap())),
        _ => Err(ServeError::Validation(
            "response_format, guided_regex, and guided_grammar are mutually exclusive".into(),
        )),
    }
}

/// Convert engine logprobs to chat completion logprobs format.
fn build_chat_logprobs(
    logprobs: &[scratchy_core_common::LogprobsOutput],
    tokenizer: Option<&Tokenizer>,
) -> protocol::ChatCompletionLogProbs {
    let content: Vec<protocol::ChatCompletionLogProbsContent> = logprobs
        .iter()
        .map(|lp| {
            let token_str = tokenizer
                .and_then(|tok| tok.decode(&[lp.sampled.token_id], false).ok())
                .unwrap_or_else(|| format!("<token_{}>", lp.sampled.token_id));

            let top_logprobs: Vec<protocol::ChatCompletionLogProb> = lp
                .top_logprobs
                .iter()
                .map(|tlp| {
                    let t = tokenizer
                        .and_then(|tok| tok.decode(&[tlp.token_id], false).ok())
                        .unwrap_or_else(|| format!("<token_{}>", tlp.token_id));
                    protocol::ChatCompletionLogProb {
                        token: t,
                        logprob: tlp.logprob as f64,
                        bytes: None,
                    }
                })
                .collect();

            protocol::ChatCompletionLogProbsContent {
                token: token_str,
                logprob: lp.sampled.logprob as f64,
                bytes: None,
                top_logprobs,
            }
        })
        .collect();

    protocol::ChatCompletionLogProbs {
        content: Some(content),
    }
}

/// Convert engine logprobs to completion logprobs format.
fn build_completion_logprobs(
    logprobs: &[scratchy_core_common::LogprobsOutput],
    tokenizer: Option<&Tokenizer>,
) -> protocol::CompletionLogProbs {
    let mut text_offset = Vec::with_capacity(logprobs.len());
    let mut token_logprobs = Vec::with_capacity(logprobs.len());
    let mut tokens = Vec::with_capacity(logprobs.len());
    let mut top_logprobs = Vec::with_capacity(logprobs.len());

    let mut offset = 0u32;
    for lp in logprobs {
        let token_str = tokenizer
            .and_then(|tok| tok.decode(&[lp.sampled.token_id], false).ok())
            .unwrap_or_else(|| format!("<token_{}>", lp.sampled.token_id));

        text_offset.push(offset);
        offset += token_str.len() as u32;

        token_logprobs.push(Some(lp.sampled.logprob as f64));
        tokens.push(token_str);

        if lp.top_logprobs.is_empty() {
            top_logprobs.push(None);
        } else {
            let top: std::collections::HashMap<String, f64> = lp
                .top_logprobs
                .iter()
                .map(|tlp| {
                    let t = tokenizer
                        .and_then(|tok| tok.decode(&[tlp.token_id], false).ok())
                        .unwrap_or_else(|| format!("<token_{}>", tlp.token_id));
                    (t, tlp.logprob as f64)
                })
                .collect();
            top_logprobs.push(Some(top));
        }
    }

    protocol::CompletionLogProbs {
        text_offset,
        token_logprobs,
        tokens,
        top_logprobs,
    }
}

/// ⛔ A `tool_choice` THAT NEEDS A TOOL NAMES ONE THE REQUEST DECLARES. A forced tool missing
/// from `tools` would have every call the model makes dropped, and its reply would come back as
/// text with no error: #767's bug, through a misspelled name. `"required"` with no tools could be
/// met by none. Both are refused, as vLLM's `check_tool_usage` and the upstream APIs refuse them.
/// `tools` names the tools the request declares. Every chat request is checked as its sampling
/// params are built; `/v1/messages`' `tool-spans` path, which builds none, checks its own.
pub(crate) fn check_tool_choice(
    tool_choice: &Option<protocol::ToolChoice>,
    tools: &[&str],
) -> ServeResult<()> {
    match tool_choice {
        Some(protocol::ToolChoice::Function(name)) if !tools.contains(&name.as_str()) => {
            Err(ServeError::Validation(format!(
                "tool_choice forces the tool `{name}`, which `tools` does not declare"
            )))
        }
        Some(protocol::ToolChoice::Required) if tools.is_empty() => Err(ServeError::Validation(
            "tool_choice requires a tool call, and `tools` declares none".to_string(),
        )),
        _ => Ok(()),
    }
}

/// Whether `tool_choice` is `"none"`: the reply is never read for a tool call.
fn is_tool_choice_none(tool_choice: &Option<protocol::ToolChoice>) -> bool {
    matches!(tool_choice, Some(protocol::ToolChoice::None))
}

/// A whole reply as its `tool_choice` lets its tool calls out: read by `parser` unless the choice
/// is `"none"`, and with a [`ForcedTool`] only that tool's calls, the reply then the text around
/// the others. Both whole-reply paths apply it (the chat endpoint's, and `/v1/messages`' under
/// `tool-spans`); the stream applies the same rule call by call ([`ToolCallStream`]).
pub(crate) fn reply_tool_calls(
    parser: &dyn ToolCallParser,
    text: String,
    tool_choice: &Option<protocol::ToolChoice>,
) -> ToolReply {
    if is_tool_choice_none(tool_choice) {
        return ToolReply {
            content: Some(text),
            calls: None,
        };
    }
    let extracted = parser.extract_tool_calls(&text);
    if !extracted.tools_called {
        // No tool call: keep the parser's cleaned content (it may strip control tokens) instead
        // of the raw text.
        return ToolReply {
            content: extracted.content.or(Some(text)),
            calls: None,
        };
    }
    let forced = ForcedTool::of(tool_choice);
    let calls: Vec<_> = (extracted.tool_calls.into_iter())
        .filter(|tc| (forced.as_ref()).is_none_or(|f| f.admits(&tc.function.name)))
        .collect();
    // Every call named a tool other than the forced one: dropped, as the stream drops them, so
    // the reply is the text around them.
    ToolReply {
        content: extracted.content,
        calls: (!calls.is_empty()).then_some(calls),
    }
}

/// A whole reply's text and the tool calls [`reply_tool_calls`] kept, `None` when it kept none.
pub(crate) struct ToolReply {
    pub content: Option<String>,
    pub calls: Option<Vec<protocol::ToolCall>>,
}

/// The one tool a `tool_choice` makes the model call (OpenAI's
/// `{"type": "function", "function": {"name": "get_weather"}}`, Anthropic's
/// `{"type": "tool", "name": …}`: [`protocol::ToolChoice::Function`]), decoded once per request.
/// Both replies drop a call to any other tool, so the reply is the text around it: the stream must
/// decide a call at its first delta, before it can know whether a later call will be kept (#683).
#[derive(Clone, Debug, PartialEq)]
struct ForcedTool(String);

impl ForcedTool {
    /// The tool `tool_choice` forces, if it names one.
    fn of(tool_choice: &Option<protocol::ToolChoice>) -> Option<Self> {
        match tool_choice {
            Some(protocol::ToolChoice::Function(name)) => Some(Self(name.clone())),
            _ => None,
        }
    }

    /// Whether a call to `name` is kept.
    fn admits(&self, name: &str) -> bool {
        self.0 == name
    }
}

/// A tool call's index among the tool parser's deltas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ParserCall(u32);

/// A tool call's index in the stream the client receives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct StreamCall(u32);

/// A request's streamed tool calls as they leave the output path: each call's name first, and
/// with a [`ForcedTool`], only that tool's calls, numbered from 0 in the order they are named (the
/// places they hold in the non-streaming reply's list).
///
/// Every consumer reads a call's first delta as its start (`/v1/messages` opens the `tool_use`
/// block with its name; the forced filter keeps or drops the call on it), but a parser can write
/// a call's arguments before its name: Mistral's pre-v11 format takes the object's keys in the
/// order the model writes them. So a call's deltas are held until its name arrives, and go out
/// right after it; a call that never names its tool sends nothing.
#[derive(Default)]
struct ToolCallStream {
    forced: Option<ForcedTool>,
    kept: HashMap<ParserCall, StreamCall>,
    /// The calls that named a tool the request does not allow.
    dropped: HashSet<ParserCall>,
    /// The deltas of calls whose name has not arrived.
    unnamed: HashMap<ParserCall, Vec<DeltaToolCall>>,
}

impl ToolCallStream {
    fn new(forced: Option<ForcedTool>) -> Self {
        Self {
            forced,
            ..Self::default()
        }
    }

    /// The deltas to send for the parser's `deltas`, renumbered for the stream.
    fn admit(&mut self, deltas: Vec<DeltaToolCall>) -> Vec<DeltaToolCall> {
        let mut out = Vec::new();
        for delta in deltas {
            let call = ParserCall(delta.index);
            if self.dropped.contains(&call) {
                continue;
            }
            if let Some(&StreamCall(index)) = self.kept.get(&call) {
                out.push(DeltaToolCall { index, ..delta });
                continue;
            }
            let Some(name) = delta.function_name.as_deref() else {
                self.unnamed.entry(call).or_default().push(delta);
                continue;
            };
            let held = self.unnamed.remove(&call).unwrap_or_default();
            if self.forced.as_ref().is_some_and(|f| !f.admits(name)) {
                self.dropped.insert(call);
                continue;
            }
            let stream_call = StreamCall(self.kept.len() as u32);
            self.kept.insert(call, stream_call);
            let StreamCall(index) = stream_call;
            out.extend(
                std::iter::once(delta)
                    .chain(held)
                    .map(|d| DeltaToolCall { index, ..d }),
            );
        }
        out
    }
}

/// Generate placeholder text from token IDs (used when no tokenizer is available).
fn placeholder_text(token_ids: &[u32]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for id in token_ids {
        let _ = write!(s, "<token_{id}>");
    }
    s
}

/// ⛔⛔⛔ TOKENS THAT RENDER TO NOTHING ARE A MUTE FAILURE, AND THE IDS ARE THE WHOLE DIAGNOSIS.
///
/// A request can finish with `completion_tokens: 50`, `finish_reason: "length"` and `"text": ""`. That is
/// not a detokenizer fault: `decode_next` returns `""` for any id that decodes to nothing under
/// `skip_special_tokens` (`string.len() > prefix.len()` fails), so fifty SPECIAL ids render as fifty
/// nothings. EOS is excluded by construction — it would have stopped generation at the first token with
/// `finish_reason: "stop"` — so a long empty completion means the sampler is landing on non-EOS special
/// ids, i.e. the logits are junk. That is a BACKEND bug, and the response body cannot express it: the ids
/// never reach the client, so the failure reads as "the server returned nothing".
///
/// So say the ids. Once per finished request, only when text is empty and tokens exist, bounded so a long
/// generation cannot flood the log. Whoever reads this next gets in one line what cost this session many
/// rounds of inference: WHICH ids, and whether they are one id repeated (a stuck argmax) or a spread.
pub(crate) fn warn_if_empty_text(request_id: &str, text: &str, token_ids: &[u32]) {
    if !text.is_empty() || token_ids.is_empty() {
        return;
    }
    const SHOW: usize = 16;
    let head: Vec<u32> = token_ids.iter().take(SHOW).copied().collect();
    let all_same = token_ids.windows(2).all(|w| w[0] == w[1]);
    warn!(
        request_id = %request_id,
        completion_tokens = token_ids.len(),
        first_ids = ?head,
        all_ids_identical = all_same,
        "generated tokens rendered to EMPTY text — every id decoded to nothing under \
         skip_special_tokens, so these are non-EOS SPECIAL ids and the sampler's logits are suspect \
         (a backend problem, not detokenization). `all_ids_identical` distinguishes a stuck argmax from \
         a spread of special ids."
    );
}

/// Merge default and per-request chat template kwargs.
/// Request kwargs override defaults (like Python's `default | request`).
fn merge_chat_template_kwargs(
    defaults: Option<&std::collections::HashMap<String, serde_json::Value>>,
    request: Option<&std::collections::HashMap<String, serde_json::Value>>,
) -> Option<std::collections::HashMap<String, serde_json::Value>> {
    match (defaults, request) {
        (None, None) => None,
        (Some(d), None) => Some(d.clone()),
        (None, Some(r)) => Some(r.clone()),
        (Some(d), Some(r)) => {
            let mut merged = d.clone();
            merged.extend(r.iter().map(|(k, v)| (k.clone(), v.clone())));
            Some(merged)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    /// The breakpoint marks come out of the rendered prompt, each kept as its byte offset in what
    /// is left; a prompt with none is passed through as it is.
    #[test]
    fn take_cache_breakpoints_strips_the_marks_and_keeps_their_offsets() {
        let marked = format!("ab{0}c—d{0}", super::CACHE_BREAKPOINT);
        let (text, marks) = super::take_cache_breakpoints(marked);
        assert_eq!((text.as_str(), marks), ("abc—d", vec![2, 7]));
        let (text, marks) = super::take_cache_breakpoints("plain".into());
        assert_eq!((text.as_str(), marks), ("plain", vec![]));
    }

    use super::*;
    use crate::tokenizer::make_test_tokenizer;
    use scratchy_core_config::{SchedulerConfig, SchedulerPolicy};
    use scratchy_serving_engine::core_client::InprocClient;
    use scratchy_serving_engine::engine_core::EngineCoreConfig;
    use scratchy_serving_engine::executor::NoopExecutor;
    use scratchy_serving_transport::messages::PauseMode;

    /// One engine output through the server's output path: phase 1, the detokenizer, phase 3.
    fn run_output(requests: &mut HashMap<String, RequestState>, output: EngineCoreOutput) {
        let work = AsyncEngine::process_outputs_phase1(requests, std::slice::from_ref(&output));
        let results = AsyncEngine::parallel_detokenize(work);
        AsyncEngine::process_outputs_phase3(requests, vec![output], results);
    }

    pub(super) fn make_test_request_state(
        stream_tx: Option<mpsc::UnboundedSender<StreamDelta>>,
    ) -> RequestState {
        RequestState {
            generated_token_ids: Vec::new(),
            num_prompt_tokens: 5,
            num_cached_tokens: 0,
            finish_reason: None,
            error: None,
            stop_reason: None,
            stream_tx,
            detokenizer: None,
            choice: Choice::default(),
            submit_time: Instant::now(),
            first_token_time: None,
            last_token_time: None,
            itl_count: 0,
            itl_sum: 0.0,
            logprobs: Vec::new(),
            prompt_logprobs: None,
            tool_parser_state: None,
            reasoning_parser_state: None,
            reasoning_ended: false,
            include_reasoning: true,
            accumulated_text: String::new(),
            accumulated_content: String::new(),
            accumulated_token_ids: Vec::new(),
            tool_calls_emitted: false,
            tool_calls: ToolCallStream::default(),
            pooler_output: None,
        }
    }

    fn make_engine_config() -> EngineCoreConfig {
        EngineCoreConfig {
            hybrid_kv: None,
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
        }
    }

    fn make_test_engine() -> AsyncEngine {
        let executor = Box::new(NoopExecutor::new(1024));
        let client = Box::new(InprocClient::new(make_engine_config(), executor));
        AsyncEngine::new(client, "test-model".to_string(), 4096)
    }

    fn make_test_engine_with_tokenizer() -> AsyncEngine {
        let executor = Box::new(NoopExecutor::new(1024));
        let client = Box::new(InprocClient::new(make_engine_config(), executor));
        let tok = Arc::new(make_test_tokenizer());
        AsyncEngine::with_tokenizer(client, "test-model".to_string(), 4096, tok)
    }

    fn make_chat_request() -> protocol::ChatCompletionRequest {
        protocol::ChatCompletionRequest {
            model: Some("test".to_string()),
            messages: vec![protocol::ChatCompletionMessageParam {
                role: "user".to_string(),
                content: Some(serde_json::Value::String("Hello".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning: None,
            }],
            temperature: Some(0.7),
            top_p: Some(0.9),
            n: 1,
            max_tokens: Some(100),
            max_completion_tokens: None,
            stream: false,
            stream_options: None,
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            prompt_logprobs: None,
            seed: Some(42),
            response_format: None,
            tools: None,
            tool_choice: None,
            user: None,
            top_k: None,
            min_p: None,
            repetition_penalty: None,
            min_tokens: 0,
            stop_token_ids: vec![],
            include_stop_str_in_output: false,
            ignore_eos: false,
            skip_special_tokens: true,
            priority: 0,
            cache_salt: None,
            request_id: None,
            guided_regex: None,
            guided_grammar: None,
            allowed_token_ids: None,
            bad_words: None,
            truncate_prompt_tokens: None,
            include_reasoning: true,
            chat_template_kwargs: None,
        }
    }

    #[test]
    fn test_engine_new() {
        let engine = make_test_engine();
        assert_eq!(engine.model_name(), "test-model");
        assert_eq!(engine.max_model_len(), 4096);
        assert!(!engine.has_tokenizer());
    }

    #[test]
    fn test_engine_with_tokenizer() {
        let engine = make_test_engine_with_tokenizer();
        assert_eq!(engine.model_name(), "test-model");
        assert!(engine.has_tokenizer());
    }

    #[test]
    fn test_check_prompt_fits_rejects_over_length() {
        // The "scr launch claude" meltdown guard: a prompt at/over
        // max_model_len must be rejected with a 400 (ServeError::Validation)
        // instead of being admitted into the never-schedulable WAITING queue.
        let engine = make_test_engine(); // max_model_len = 4096
        let err = engine
            .check_prompt_fits(4096)
            .expect_err("prompt == max_model_len must be rejected");
        match err {
            ServeError::Validation(msg) => {
                assert!(
                    msg.contains("maximum context length is 4096"),
                    "unexpected message: {msg}"
                );
                assert!(
                    msg.contains("4096 input tokens"),
                    "unexpected message: {msg}"
                );
            }
            other => panic!("expected ServeError::Validation, got {other:?}"),
        }
        // Well over the limit also rejects.
        assert!(engine.check_prompt_fits(100_000).is_err());
    }

    // The ServeError -> HTTP 400 mapping lives behind the `serve` feature.
    #[cfg(feature = "serve")]
    #[test]
    fn test_check_prompt_fits_validation_maps_to_400() {
        use axum::response::IntoResponse;
        let engine = make_test_engine();
        let err = engine.check_prompt_fits(5000).unwrap_err();
        let resp = err.into_response();
        assert_eq!(resp.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_check_prompt_fits_allows_in_range() {
        let engine = make_test_engine(); // max_model_len = 4096
        // Strictly below the limit is admitted (one decode slot remains).
        assert!(engine.check_prompt_fits(4095).is_ok());
        assert!(engine.check_prompt_fits(0).is_ok());
        assert!(engine.check_prompt_fits(1024).is_ok());
    }

    #[test]
    fn test_chat_to_engine_request_without_tokenizer() {
        let engine = make_test_engine();
        let request = make_chat_request();
        let params = engine.build_sampling_params_from_chat(&request).unwrap();
        let ec_req = engine
            .chat_to_engine_request("test-1", &request, &params)
            .unwrap();
        assert_eq!(ec_req.request_id, "test-1");
        assert!(ec_req.prompt_token_ids.is_some());
        assert!(!ec_req.prompt_token_ids.as_ref().unwrap().is_empty());
        // Without tokenizer, prompt is byte values of "Hello".
        let ids = ec_req.prompt_token_ids.unwrap();
        assert_eq!(ids.len(), 5); // b"Hello" = 5 bytes
    }

    /// A `tool_choice` of `"none"` renders the same prompt as `"auto"`, tools included: most
    /// templates put the tools first, so leaving them out would give a `"none"` request no
    /// prefix in common with its siblings', and it would prefill from scratch. The engine reads
    /// no tool call out of its reply instead.
    #[test]
    fn a_tool_choice_of_none_renders_the_prompt_auto_does() {
        let mut engine = make_test_engine();
        let template = "{% if tools %}TOOLS {% endif %}{{ messages[0]['content'] }}";
        engine.set_chat_template(Arc::new(ChatTemplate::new(template.to_string()).unwrap()));
        let prompt = |tool_choice: Option<protocol::ToolChoice>| {
            let request = protocol::ChatCompletionRequest {
                tools: Some(vec![protocol::ChatCompletionToolsParam {
                    tool_type: "function".to_string(),
                    function: protocol::FunctionDefinition {
                        name: "Bash".to_string(),
                        description: None,
                        parameters: None,
                    },
                }]),
                tool_choice,
                ..make_chat_request()
            };
            let params = engine.build_sampling_params_from_chat(&request).unwrap();
            let ec_req = engine
                .chat_to_engine_request("test-1", &request, &params)
                .unwrap();
            ec_req.prompt_token_ids.unwrap()
        };
        let auto = prompt(Some(protocol::ToolChoice::Auto));
        assert_eq!(auto, b"TOOLS Hello".map(u32::from), "the tools render");
        assert_eq!(prompt(Some(protocol::ToolChoice::None)), auto);
        assert_eq!(prompt(None), auto);
    }

    #[test]
    fn test_chat_to_engine_request_with_tokenizer() {
        let engine = make_test_engine_with_tokenizer();
        let request = make_chat_request();
        let params = engine.build_sampling_params_from_chat(&request).unwrap();
        let ec_req = engine
            .chat_to_engine_request("test-1", &request, &params)
            .unwrap();
        assert_eq!(ec_req.request_id, "test-1");
        let ids = ec_req.prompt_token_ids.unwrap();
        assert!(!ids.is_empty());
        // With tokenizer, "Hello" is properly tokenized.

        let sp = ec_req.sampling_params.unwrap();
        assert!((sp.temperature - 0.7).abs() < 0.01);
        assert!((sp.top_p - 0.9).abs() < 0.01);
        assert_eq!(sp.max_tokens, Some(100));
        assert_eq!(sp.seed, Some(42));
    }

    #[test]
    fn test_completion_to_engine_request_token_ids() {
        let engine = make_test_engine();
        let request = protocol::CompletionRequest {
            model: None,
            prompt: Some(protocol::CompletionPrompt::TokenIds(vec![1, 2, 3, 4])),
            echo: false,
            temperature: None,
            top_p: None,
            n: 1,
            max_tokens: Some(50),
            stream: false,
            stream_options: None,
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            logit_bias: None,
            logprobs: None,
            prompt_logprobs: None,
            suffix: None,
            seed: None,
            user: None,
            top_k: None,
            min_p: None,
            repetition_penalty: None,
            min_tokens: 0,
            stop_token_ids: vec![],
            include_stop_str_in_output: false,
            ignore_eos: false,
            skip_special_tokens: true,
            add_special_tokens: true,
            priority: 0,
            cache_salt: None,
            request_id: None,
            guided_regex: None,
            guided_grammar: None,
            allowed_token_ids: None,
            bad_words: None,
            truncate_prompt_tokens: None,
            block_annotations: None,
            seal: false,
            volatile: false,
        };

        let params = engine
            .build_sampling_params_from_completion(&request)
            .unwrap();
        let ec_req = engine
            .completion_to_engine_request("comp-1", &request, &params)
            .unwrap();
        assert_eq!(ec_req.request_id, "comp-1");
        assert_eq!(ec_req.prompt_token_ids, Some(vec![1, 2, 3, 4]));
        assert_eq!(ec_req.sampling_params.unwrap().max_tokens, Some(50));
    }

    #[test]
    fn test_completion_to_engine_request_with_tokenizer() {
        let engine = make_test_engine_with_tokenizer();
        let request = protocol::CompletionRequest {
            model: None,
            prompt: Some(protocol::CompletionPrompt::Single(
                "Hello world".to_string(),
            )),
            echo: false,
            temperature: None,
            top_p: None,
            n: 1,
            max_tokens: Some(50),
            stream: false,
            stream_options: None,
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            logit_bias: None,
            logprobs: None,
            prompt_logprobs: None,
            suffix: None,
            seed: None,
            user: None,
            top_k: None,
            min_p: None,
            repetition_penalty: None,
            min_tokens: 0,
            stop_token_ids: vec![],
            include_stop_str_in_output: false,
            ignore_eos: false,
            skip_special_tokens: true,
            add_special_tokens: true,
            priority: 0,
            cache_salt: None,
            request_id: None,
            guided_regex: None,
            guided_grammar: None,
            allowed_token_ids: None,
            bad_words: None,
            truncate_prompt_tokens: None,
            block_annotations: None,
            seal: false,
            volatile: false,
        };

        let params = engine
            .build_sampling_params_from_completion(&request)
            .unwrap();
        let ec_req = engine
            .completion_to_engine_request("comp-1", &request, &params)
            .unwrap();
        let ids = ec_req.prompt_token_ids.unwrap();
        assert!(!ids.is_empty());
        // "Hello world" properly tokenized by byte-level BPE.
    }

    #[test]
    fn test_run_output_unknown_request() {
        let mut requests = HashMap::new();
        let output = EngineCoreOutput {
            request_id: "unknown".to_string(),
            new_token_ids: vec![1, 2, 3],
            finish_reason: None,
            stop_reason: None,
            num_cached_tokens: 0,
            events: None,
            new_logprobs: None,
            new_prompt_logprobs: None,
            pooler_output: None,
        };
        // Should not panic.
        run_output(&mut requests, output);
    }

    /// An output for a request finished here already — a stop string matched, its non-streaming
    /// entry waiting to be collected — came from a step still in flight for it: the output path
    /// drops it, the request's tokens as they were.
    #[test]
    fn an_output_for_a_finished_request_is_dropped() {
        let mut requests = HashMap::new();
        let mut state = make_test_request_state(None);
        state.generated_token_ids = vec![10, 11];
        state.finish_reason = Some(FinishReason::Stop);
        requests.insert("req-1".to_string(), state);
        let output = || EngineCoreOutput {
            request_id: "req-1".to_string(),
            new_token_ids: vec![12, 13],
            finish_reason: None,
            stop_reason: None,
            num_cached_tokens: 0,
            events: None,
            new_logprobs: None,
            new_prompt_logprobs: None,
            pooler_output: None,
        };
        let work = AsyncEngine::process_outputs_phase1(&mut requests, &[output()]);
        assert!(work.iter().all(Option::is_none), "nothing to detokenize");
        let results = AsyncEngine::parallel_detokenize(work);
        AsyncEngine::process_outputs_phase3(&mut requests, vec![output()], results);
        assert_eq!(requests["req-1"].generated_token_ids, [10, 11]);
    }

    /// A stop string is not an engine-side criterion, so a request it finishes is still running
    /// engine-side. Phase 3 names every such request so the step loop can abort it there —
    /// without that abort the engine decodes it on to max_tokens under the NEXT request's steps
    /// (#711: the follow-up request came back garbage with a 3.3x first-token latency).
    #[test]
    fn a_stop_string_finish_names_the_request_for_engine_abort() {
        let mut requests = HashMap::new();
        requests.insert("req-1".to_string(), make_test_request_state(None));
        let output = EngineCoreOutput {
            request_id: "req-1".to_string(),
            new_token_ids: vec![12, 13],
            finish_reason: None,
            stop_reason: None,
            num_cached_tokens: 0,
            events: None,
            new_logprobs: None,
            new_prompt_logprobs: None,
            pooler_output: None,
        };
        let detok = Some((Some("Century".to_string()), Some("…Cent".to_string())));
        let to_abort =
            AsyncEngine::process_outputs_phase3(&mut requests, vec![output], vec![detok]);
        assert_eq!(
            to_abort,
            vec!["req-1".to_string()],
            "the engine still holds req-1"
        );
        assert_eq!(requests["req-1"].finish_reason, Some(FinishReason::Stop));
        assert_eq!(
            requests["req-1"].stop_reason,
            Some(StopReason::String("Century".to_string()))
        );
    }

    /// The engine's own finish for the same step means it already retired the request — no abort
    /// may be asked for, or the loop would abort an id the scheduler no longer knows.
    #[test]
    fn an_engine_finish_needs_no_abort() {
        let mut requests = HashMap::new();
        requests.insert("req-1".to_string(), make_test_request_state(None));
        let output = EngineCoreOutput {
            request_id: "req-1".to_string(),
            new_token_ids: vec![12, 13],
            finish_reason: Some(FinishReason::Length),
            stop_reason: None,
            num_cached_tokens: 0,
            events: None,
            new_logprobs: None,
            new_prompt_logprobs: None,
            pooler_output: None,
        };
        let detok = Some((Some("Century".to_string()), Some("…Cent".to_string())));
        let to_abort =
            AsyncEngine::process_outputs_phase3(&mut requests, vec![output], vec![detok]);
        assert!(to_abort.is_empty(), "the engine finished it itself");
    }

    /// No stop string, no engine finish: the request keeps decoding and nothing is named.
    #[test]
    fn a_running_request_names_no_abort() {
        let mut requests = HashMap::new();
        requests.insert("req-1".to_string(), make_test_request_state(None));
        let output = EngineCoreOutput {
            request_id: "req-1".to_string(),
            new_token_ids: vec![12, 13],
            finish_reason: None,
            stop_reason: None,
            num_cached_tokens: 0,
            events: None,
            new_logprobs: None,
            new_prompt_logprobs: None,
            pooler_output: None,
        };
        let detok = Some((None, Some("…".to_string())));
        let to_abort =
            AsyncEngine::process_outputs_phase3(&mut requests, vec![output], vec![detok]);
        assert!(to_abort.is_empty());
        assert_eq!(requests["req-1"].finish_reason, None);
    }

    #[test]
    fn test_run_output_accumulates_tokens() {
        let mut requests = HashMap::new();
        requests.insert("req-1".to_string(), make_test_request_state(None));

        // First output.
        run_output(
            &mut requests,
            EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![10, 11],
                finish_reason: None,
                stop_reason: None,
                num_cached_tokens: 0,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: None,
            },
        );

        // Second output.
        run_output(
            &mut requests,
            EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![12],
                finish_reason: Some(FinishReason::Stop),
                stop_reason: Some(StopReason::Token(50256)),
                num_cached_tokens: 3,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: None,
            },
        );

        let state = requests.get("req-1").unwrap();
        assert_eq!(state.generated_token_ids, vec![10, 11, 12]);
        assert_eq!(state.finish_reason, Some(FinishReason::Stop));
        assert_eq!(state.stop_reason, Some(StopReason::Token(50256)));
        assert_eq!(state.num_cached_tokens, 3);
    }

    #[test]
    fn test_run_output_streaming() {
        let tok = Arc::new(make_test_tokenizer());
        let prompt_ids = tok.encode("Hi", false).unwrap();
        let (first, last) = (
            tok.encode("ab", false).unwrap(),
            tok.encode("c", false).unwrap(),
        );
        let detok =
            IncrementalDetokenizer::new(Arc::clone(&tok), &prompt_ids, vec![], 0, false, false);
        let (tx, mut rx) = mpsc::unbounded_channel();

        let mut requests = HashMap::new();
        {
            let mut state = make_test_request_state(Some(tx));
            state.num_prompt_tokens = prompt_ids.len() as u32;
            state.detokenizer = Some(Arc::new(std::sync::Mutex::new(detok)));
            requests.insert("req-1".to_string(), state);
        }

        run_output(
            &mut requests,
            EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: first.clone(),
                finish_reason: None,
                stop_reason: None,
                num_cached_tokens: 0,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: None,
            },
        );

        let delta = rx.try_recv().unwrap();
        assert_eq!(delta.new_token_ids, first);
        assert!(delta.finish_reason.is_none());
        assert_eq!(delta.text.as_deref(), Some("ab"));

        // Finish.
        run_output(
            &mut requests,
            EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: last.clone(),
                finish_reason: Some(FinishReason::Length),
                stop_reason: None,
                num_cached_tokens: 0,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: None,
            },
        );

        let delta = rx.try_recv().unwrap();
        assert_eq!(delta.new_token_ids, last);
        assert_eq!(delta.text.as_deref(), Some("c"));
        assert_eq!(delta.finish_reason, Some(FinishReason::Length));

        // Finished streaming request should be removed from the map.
        assert!(!requests.contains_key("req-1"));
    }

    #[test]
    fn test_run_output_with_detokenizer() {
        let tok = Arc::new(make_test_tokenizer());
        // Encode a prompt and some continuation.
        let prompt = "Hi";
        let prompt_ids = tok.encode(prompt, false).unwrap();
        let full_text = format!("{prompt} there");
        let full_ids = tok.encode(&full_text, false).unwrap();
        let output_ids = full_ids[prompt_ids.len()..].to_vec();

        let detok =
            IncrementalDetokenizer::new(Arc::clone(&tok), &prompt_ids, vec![], 0, false, false);

        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut requests = HashMap::new();
        {
            let mut state = make_test_request_state(Some(tx));
            state.num_prompt_tokens = prompt_ids.len() as u32;
            state.detokenizer = Some(Arc::new(std::sync::Mutex::new(detok)));
            requests.insert("req-1".to_string(), state);
        }

        run_output(
            &mut requests,
            EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: output_ids,
                finish_reason: Some(FinishReason::Length),
                stop_reason: None,
                num_cached_tokens: 0,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: None,
            },
        );

        let delta = rx.try_recv().unwrap();
        // With detokenizer, we should get real text.
        assert!(delta.text.is_some());
        let text = delta.text.unwrap();
        assert!(
            text.contains("there"),
            "Expected 'there' in delta text, got: {text:?}"
        );
    }

    /// An engine with no tokenizer streams each step's token ids as placeholders, as its
    /// non-streaming response does: the output path fills them, so an encoder reads a delta's
    /// `text: None` as no text.
    #[test]
    fn a_request_with_no_detokenizer_streams_placeholders() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut requests =
            HashMap::from([("req-1".to_string(), make_test_request_state(Some(tx)))]);
        let outputs = vec![EngineCoreOutput {
            request_id: "req-1".to_string(),
            new_token_ids: vec![10, 11],
            finish_reason: None,
            stop_reason: None,
            num_cached_tokens: 0,
            events: None,
            new_logprobs: None,
            new_prompt_logprobs: None,
            pooler_output: None,
        }];
        let work = AsyncEngine::process_outputs_phase1(&mut requests, &outputs);
        let results = AsyncEngine::parallel_detokenize(work);
        AsyncEngine::process_outputs_phase3(&mut requests, outputs, results);
        let delta = rx.try_recv().unwrap();
        assert_eq!(delta.text.as_deref(), Some("<token_10><token_11>"));
    }

    #[test]
    fn test_placeholder_text() {
        assert_eq!(placeholder_text(&[]), "");
        assert_eq!(placeholder_text(&[1, 2]), "<token_1><token_2>");
    }

    #[test]
    fn test_build_sampling_params_with_stop() {
        let engine = make_test_engine();
        let mut request = make_chat_request();
        request.stop = Some(protocol::StopCondition::Multiple(vec![
            "END".to_string(),
            "STOP".to_string(),
        ]));
        request.include_stop_str_in_output = true;

        let params = engine.build_sampling_params_from_chat(&request).unwrap();
        assert_eq!(params.stop, vec!["END", "STOP"]);
        assert!(params.include_stop_str_in_output);
    }

    fn make_completion_request(
        prompt: Option<protocol::CompletionPrompt>,
    ) -> protocol::CompletionRequest {
        protocol::CompletionRequest {
            model: None,
            prompt,
            echo: false,
            temperature: None,
            top_p: None,
            n: 1,
            max_tokens: Some(50),
            stream: false,
            stream_options: None,
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            logit_bias: None,
            logprobs: None,
            prompt_logprobs: None,
            suffix: None,
            seed: None,
            user: None,
            top_k: None,
            min_p: None,
            repetition_penalty: None,
            min_tokens: 0,
            stop_token_ids: vec![],
            include_stop_str_in_output: false,
            ignore_eos: false,
            skip_special_tokens: true,
            add_special_tokens: true,
            priority: 0,
            cache_salt: None,
            request_id: None,
            guided_regex: None,
            guided_grammar: None,
            allowed_token_ids: None,
            bad_words: None,
            truncate_prompt_tokens: None,
            block_annotations: None,
            seal: false,
            volatile: false,
        }
    }

    // -- n>1 and multi-prompt tests --

    #[test]
    fn test_truncate_prompt_none() {
        let mut tokens = vec![1, 2, 3, 4, 5];
        truncate_prompt(&mut tokens, None).unwrap();
        assert_eq!(tokens, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_truncate_prompt_minus_one() {
        let mut tokens = vec![1, 2, 3, 4, 5];
        truncate_prompt(&mut tokens, Some(-1)).unwrap();
        assert_eq!(tokens, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_truncate_prompt_keeps_last_n() {
        let mut tokens = vec![1, 2, 3, 4, 5];
        truncate_prompt(&mut tokens, Some(3)).unwrap();
        assert_eq!(tokens, vec![3, 4, 5]);
    }

    #[test]
    fn test_truncate_prompt_larger_than_len() {
        let mut tokens = vec![1, 2, 3];
        truncate_prompt(&mut tokens, Some(10)).unwrap();
        assert_eq!(tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_truncate_prompt_zero_invalid() {
        let mut tokens = vec![1, 2, 3];
        assert!(truncate_prompt(&mut tokens, Some(0)).is_err());
    }

    #[test]
    fn test_truncate_prompt_negative_invalid() {
        let mut tokens = vec![1, 2, 3];
        assert!(truncate_prompt(&mut tokens, Some(-2)).is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_chat_completion_n1_regression() {
        let engine = Arc::new(make_test_engine());
        engine.spawn_step_loop();

        let request = make_chat_request();
        let response = engine.chat_completion(request).await.unwrap();

        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].index, 0);
        assert!(response.usage.prompt_tokens > 0);
        assert!(response.usage.completion_tokens.unwrap_or(0) > 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_chat_completion_n2() {
        let engine = Arc::new(make_test_engine());
        engine.spawn_step_loop();

        let mut request = make_chat_request();
        request.n = 2;
        let response = engine.chat_completion(request).await.unwrap();

        assert_eq!(response.choices.len(), 2);
        assert_eq!(response.choices[0].index, 0);
        assert_eq!(response.choices[1].index, 1);
        // prompt_tokens counted once (not * n).
        assert!(response.usage.prompt_tokens > 0);
        // completion_tokens is sum of both choices.
        assert!(response.usage.completion_tokens.unwrap_or(0) > 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_completion_n1_regression() {
        let engine = Arc::new(make_test_engine());
        engine.spawn_step_loop();

        let request =
            make_completion_request(Some(protocol::CompletionPrompt::TokenIds(vec![1, 2, 3])));
        let response = engine.completion(request).await.unwrap();

        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].index, 0);
        assert_eq!(response.usage.prompt_tokens, 3);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_completion_n3() {
        let engine = Arc::new(make_test_engine());
        engine.spawn_step_loop();

        let mut request =
            make_completion_request(Some(protocol::CompletionPrompt::TokenIds(vec![1, 2, 3])));
        request.n = 3;
        let response = engine.completion(request).await.unwrap();

        assert_eq!(response.choices.len(), 3);
        assert_eq!(response.choices[0].index, 0);
        assert_eq!(response.choices[1].index, 1);
        assert_eq!(response.choices[2].index, 2);
        // prompt_tokens counted once.
        assert_eq!(response.usage.prompt_tokens, 3);
        // completion_tokens summed across all 3.
        assert!(response.usage.completion_tokens.unwrap_or(0) > 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_completion_multi_prompt_n1() {
        let engine = Arc::new(make_test_engine());
        engine.spawn_step_loop();

        let request =
            make_completion_request(Some(protocol::CompletionPrompt::MultipleTokenIds(vec![
                vec![1, 2],
                vec![3, 4, 5],
            ])));
        let response = engine.completion(request).await.unwrap();

        assert_eq!(response.choices.len(), 2);
        assert_eq!(response.choices[0].index, 0);
        assert_eq!(response.choices[1].index, 1);
        // prompt_tokens = sum of prompt lengths: 2 + 3 = 5.
        assert_eq!(response.usage.prompt_tokens, 5);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_completion_multi_prompt_n2() {
        let engine = Arc::new(make_test_engine());
        engine.spawn_step_loop();

        let mut request =
            make_completion_request(Some(protocol::CompletionPrompt::MultipleTokenIds(vec![
                vec![1, 2],
                vec![3, 4, 5],
            ])));
        request.n = 2;
        let response = engine.completion(request).await.unwrap();

        // 2 prompts * 2 = 4 choices.
        assert_eq!(response.choices.len(), 4);
        assert_eq!(response.choices[0].index, 0);
        assert_eq!(response.choices[1].index, 1);
        assert_eq!(response.choices[2].index, 2);
        assert_eq!(response.choices[3].index, 3);
        // prompt_tokens = sum of unique prompt lengths: 2 + 3 = 5.
        assert_eq!(response.usage.prompt_tokens, 5);
    }

    /// A streamed completion stamps each choice's delta with the prompt it answers, the one
    /// fact the usage chunk counts each prompt once by (#557): choice `i` of `n` per prompt
    /// answers prompt `i / n`, with that prompt's length.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_completion_stream_stamps_each_choice_with_its_prompt() {
        let engine = Arc::new(make_test_engine());
        engine.spawn_step_loop();
        let mut request =
            make_completion_request(Some(protocol::CompletionPrompt::MultipleTokenIds(vec![
                vec![1, 2],
                vec![3, 4, 5],
            ])));
        request.n = 2;
        request.stream = true;
        let (_, _, mut rx) = engine.completion_stream(request).await.unwrap();

        let mut seen = std::collections::BTreeSet::new();
        while let Some(delta) = rx.recv().await {
            let prompt = delta.index / 2;
            assert_eq!(
                delta.usage.prompt,
                PromptIndex(prompt),
                "choice {}",
                delta.index
            );
            assert_eq!(delta.usage.prompt_tokens, [2, 3][prompt as usize]);
            seen.insert(delta.index);
        }
        assert_eq!(seen, (0..4).collect(), "every choice streamed");
    }

    /// The non-streaming chat reply counts the prompt's cache hit once, as it counts the prompt
    /// (#580): with the prompt cached, two choices that each hit it report the one hit, and never
    /// more cached tokens than prompt tokens.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_chat_completion_n2_counts_the_prompts_cache_hit_once() {
        let mut config = make_engine_config();
        config.enable_prefix_caching = true;
        let client = Box::new(InprocClient::new(config, Box::new(NoopExecutor::new(1024))));
        let tok = Arc::new(make_test_tokenizer());
        let engine = Arc::new(AsyncEngine::with_tokenizer(
            client,
            "test-model".to_string(),
            4096,
            tok,
        ));
        engine.spawn_step_loop();
        // Several 16-token blocks long, so a cache hit spans whole blocks.
        let request = |n| {
            let mut request = make_chat_request();
            request.messages[0].content = Some(serde_json::Value::String(
                "a prompt long enough to fill several cache blocks of sixteen tokens".repeat(2),
            ));
            request.n = n;
            request.max_tokens = Some(2);
            request
        };
        let cached = |usage: &protocol::UsageInfo| {
            (usage.prompt_tokens_details.as_ref()).and_then(|d| d.cached_tokens)
        };

        engine.chat_completion(request(1)).await.unwrap();
        let warm = engine.chat_completion(request(1)).await.unwrap().usage;
        let hit = cached(&warm).expect("the prompt is cached");
        assert!(hit > 0 && hit <= warm.prompt_tokens);

        let two = engine.chat_completion(request(2)).await.unwrap().usage;
        assert_eq!(two.prompt_tokens, warm.prompt_tokens);
        assert_eq!(cached(&two), Some(hit), "the one hit, not one per choice");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_chat_completion_stream_n2() {
        let engine = Arc::new(make_test_engine());
        engine.spawn_step_loop();

        let mut request = make_chat_request();
        request.n = 2;
        request.stream = true;

        let (_request_id, _model, mut rx) = engine.chat_completion_stream(request).await.unwrap();

        // Collect all deltas.
        let mut seen_indices = std::collections::HashSet::new();
        while let Some(delta) = rx.recv().await {
            seen_indices.insert(delta.index);
        }

        // Both choice indices should appear.
        assert!(
            seen_indices.contains(&0),
            "Missing index 0 in stream deltas"
        );
        assert!(
            seen_indices.contains(&1),
            "Missing index 1 in stream deltas"
        );
    }

    #[test]
    fn test_tokenize_completion_prompts_single() {
        let engine = make_test_engine();
        let request = make_completion_request(Some(protocol::CompletionPrompt::Single(
            "hello".to_string(),
        )));
        let prompts = engine.tokenize_completion_prompts(&request).unwrap();
        assert_eq!(prompts.len(), 1);
        // Without tokenizer, byte values of "hello".
        assert_eq!(prompts[0].len(), 5);
    }

    #[test]
    fn test_tokenize_completion_prompts_multiple() {
        let engine = make_test_engine();
        let request = make_completion_request(Some(protocol::CompletionPrompt::Multiple(vec![
            "foo".to_string(),
            "bar".to_string(),
        ])));
        let prompts = engine.tokenize_completion_prompts(&request).unwrap();
        assert_eq!(prompts.len(), 2);
    }

    #[test]
    fn test_tokenize_completion_prompts_token_ids() {
        let engine = make_test_engine();
        let request =
            make_completion_request(Some(protocol::CompletionPrompt::TokenIds(vec![10, 20, 30])));
        let prompts = engine.tokenize_completion_prompts(&request).unwrap();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0], vec![10, 20, 30]);
    }

    #[test]
    fn test_tokenize_completion_prompts_none() {
        let engine = make_test_engine();
        let request = make_completion_request(None);
        let prompts = engine.tokenize_completion_prompts(&request).unwrap();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0], vec![0]);
    }

    #[test]
    fn test_stream_delta_has_choice_index() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut requests = HashMap::new();
        {
            let mut state = make_test_request_state(Some(tx));
            state.num_prompt_tokens = 3;
            state.choice.index = 7;
            requests.insert("req-idx".to_string(), state);
        }

        run_output(
            &mut requests,
            EngineCoreOutput {
                request_id: "req-idx".to_string(),
                new_token_ids: vec![42],
                finish_reason: Some(FinishReason::Stop),
                stop_reason: None,
                num_cached_tokens: 0,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: None,
            },
        );

        let delta = rx.try_recv().unwrap();
        assert_eq!(delta.index, 7);
    }

    // ---------------------------------------------------------------
    // parse_response_format tests
    // ---------------------------------------------------------------

    #[test]
    fn test_parse_response_format_none() {
        let result = parse_response_format(&None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_response_format_text() {
        let rf = protocol::ResponseFormat::Standard(protocol::StandardResponseFormat {
            format_type: "text".to_string(),
            json_schema: None,
        });
        let result = parse_response_format(&Some(rf)).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_response_format_json_object() {
        let rf = protocol::ResponseFormat::Standard(protocol::StandardResponseFormat {
            format_type: "json_object".to_string(),
            json_schema: None,
        });
        let result = parse_response_format(&Some(rf)).unwrap();
        assert!(matches!(result, Some(GuidedGrammar::Json)));
    }

    #[test]
    fn test_parse_response_format_json_schema() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"]
        });
        let rf = protocol::ResponseFormat::Standard(protocol::StandardResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: Some(protocol::JsonSchemaResponseFormat {
                name: "test".to_string(),
                description: None,
                json_schema: Some(schema.clone()),
                strict: None,
            }),
        });
        let result = parse_response_format(&Some(rf)).unwrap();
        match result {
            Some(GuidedGrammar::JsonSchema { schema: s }) => {
                assert_eq!(s, schema);
            }
            other => panic!("expected JsonSchema, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_response_format_json_schema_missing_schema() {
        let rf = protocol::ResponseFormat::Standard(protocol::StandardResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: None,
        });
        let result = parse_response_format(&Some(rf));
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_response_format_invalid_type() {
        let rf = protocol::ResponseFormat::Standard(protocol::StandardResponseFormat {
            format_type: "xml".to_string(),
            json_schema: None,
        });
        let result = parse_response_format(&Some(rf));
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // resolve_guided_grammar tests
    // ---------------------------------------------------------------

    #[test]
    fn test_resolve_guided_grammar_none() {
        let result = resolve_guided_grammar(&None, &None, &None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_guided_grammar_regex_only() {
        let result = resolve_guided_grammar(&None, &Some("[0-9]+".to_string()), &None).unwrap();
        match result {
            Some(GuidedGrammar::Regex { pattern }) => assert_eq!(pattern, "[0-9]+"),
            other => panic!("expected Regex, got {other:?}"),
        }
    }

    #[test]
    fn test_resolve_guided_grammar_response_format_only() {
        let rf = protocol::ResponseFormat::Standard(protocol::StandardResponseFormat {
            format_type: "json_object".to_string(),
            json_schema: None,
        });
        let result = resolve_guided_grammar(&Some(rf), &None, &None).unwrap();
        assert!(matches!(result, Some(GuidedGrammar::Json)));
    }

    #[test]
    fn test_resolve_guided_grammar_ebnf_only() {
        let result =
            resolve_guided_grammar(&None, &None, &Some(r#"start: /[0-9]+/"#.to_string())).unwrap();
        match result {
            Some(GuidedGrammar::Ebnf { grammar }) => {
                assert_eq!(grammar, r#"start: /[0-9]+/"#)
            }
            other => panic!("expected Ebnf, got {other:?}"),
        }
    }

    #[test]
    fn test_resolve_guided_grammar_conflict() {
        let rf = protocol::ResponseFormat::Standard(protocol::StandardResponseFormat {
            format_type: "json_object".to_string(),
            json_schema: None,
        });
        let result = resolve_guided_grammar(&Some(rf), &Some("[0-9]+".to_string()), &None);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_guided_grammar_regex_and_ebnf_conflict() {
        let result = resolve_guided_grammar(
            &None,
            &Some("[0-9]+".to_string()),
            &Some("start: /[0-9]+/".to_string()),
        );
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // structural tag response_format tests
    // ---------------------------------------------------------------

    #[test]
    fn test_parse_response_format_legacy_structural_tag() {
        let json_str = r#"{
            "type": "structural_tag",
            "structures": [
                {"begin": "<function=get_weather>", "schema": {"type": "object"}, "end": "</function>"}
            ],
            "triggers": ["<function"]
        }"#;
        let rf: protocol::ResponseFormat = serde_json::from_str(json_str).unwrap();
        let result = parse_response_format(&Some(rf)).unwrap();
        match result {
            Some(GuidedGrammar::StructuralTag { spec }) => {
                // The spec should be a valid JSON string containing the structural tag info
                let parsed: serde_json::Value = serde_json::from_str(&spec).unwrap();
                assert_eq!(parsed["type"], "structural_tag");
                assert!(parsed["structures"].is_array());
                assert!(parsed["triggers"].is_array());
            }
            other => panic!("expected StructuralTag, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_response_format_new_structural_tag() {
        let json_str = r#"{
            "type": "structural_tag",
            "format": {
                "structures": [
                    {"begin": "<fn>", "schema": {"type": "object"}, "end": "</fn>"}
                ],
                "triggers": ["<fn>"]
            }
        }"#;
        let rf: protocol::ResponseFormat = serde_json::from_str(json_str).unwrap();
        let result = parse_response_format(&Some(rf)).unwrap();
        match result {
            Some(GuidedGrammar::StructuralTag { spec }) => {
                let parsed: serde_json::Value = serde_json::from_str(&spec).unwrap();
                assert_eq!(parsed["type"], "structural_tag");
                assert!(parsed["format"].is_object());
            }
            other => panic!("expected StructuralTag, got {other:?}"),
        }
    }

    #[test]
    fn test_resolve_guided_grammar_structural_tag_only() {
        let json_str = r#"{
            "type": "structural_tag",
            "structures": [
                {"begin": "<fn=test>", "schema": {"type": "object"}, "end": "</fn>"}
            ],
            "triggers": ["<fn"]
        }"#;
        let rf: protocol::ResponseFormat = serde_json::from_str(json_str).unwrap();
        let result = resolve_guided_grammar(&Some(rf), &None, &None).unwrap();
        assert!(matches!(result, Some(GuidedGrammar::StructuralTag { .. })));
    }

    #[test]
    fn test_resolve_guided_grammar_structural_tag_with_regex_conflict() {
        let json_str = r#"{
            "type": "structural_tag",
            "structures": [
                {"begin": "<fn=test>", "schema": {"type": "object"}, "end": "</fn>"}
            ],
            "triggers": ["<fn"]
        }"#;
        let rf: protocol::ResponseFormat = serde_json::from_str(json_str).unwrap();
        let result = resolve_guided_grammar(&Some(rf), &Some("[0-9]+".to_string()), &None);
        assert!(result.is_err());
    }

    #[test]
    fn test_protocol_response_format_standard_still_works() {
        // Ensure standard formats still deserialize correctly through the enum
        let json_str = r#"{"type": "json_object"}"#;
        let rf: protocol::ResponseFormat = serde_json::from_str(json_str).unwrap();
        let result = parse_response_format(&Some(rf)).unwrap();
        assert!(matches!(result, Some(GuidedGrammar::Json)));
    }

    #[test]
    fn test_protocol_response_format_json_schema_still_works() {
        // Note: the inner `json_schema` field maps to JsonSchemaResponseFormat.json_schema
        // which expects the key "json_schema" (not "schema") in our struct.
        let json_str = r#"{
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "json_schema": {"type": "object", "properties": {"x": {"type": "integer"}}}
            }
        }"#;
        let rf: protocol::ResponseFormat = serde_json::from_str(json_str).unwrap();
        let result = parse_response_format(&Some(rf)).unwrap();
        assert!(matches!(result, Some(GuidedGrammar::JsonSchema { .. })));
    }

    // ---------------------------------------------------------------
    // tool_choice helper tests
    // ---------------------------------------------------------------

    #[test]
    fn test_is_tool_choice_none_with_none_value() {
        assert!(!is_tool_choice_none(&None));
    }

    #[test]
    fn test_is_tool_choice_none_with_none_string() {
        assert!(is_tool_choice_none(&Some(protocol::ToolChoice::None)));
    }

    #[test]
    fn test_is_tool_choice_none_with_auto() {
        assert!(!is_tool_choice_none(&Some(protocol::ToolChoice::Auto)));
    }

    /// A forced tool `tools` doesn't declare (a misspelled or empty name), and `"required"` with
    /// no tools, are refused as the request is read: a 400, not a reply with every call dropped.
    /// A declared tool, `"auto"` and `"none"` pass, with tools or without.
    #[test]
    fn a_tool_choice_naming_an_undeclared_tool_is_refused() {
        use protocol::ToolChoice;
        let engine = make_test_engine();
        let bash = protocol::ChatCompletionToolsParam {
            tool_type: "function".to_string(),
            function: protocol::FunctionDefinition {
                name: "Bash".to_string(),
                description: None,
                parameters: None,
            },
        };
        let check = |tools: Option<Vec<protocol::ChatCompletionToolsParam>>, choice| {
            let request = protocol::ChatCompletionRequest {
                tools,
                tool_choice: Some(choice),
                ..make_chat_request()
            };
            engine.build_sampling_params_from_chat(&request).err()
        };
        let declared = || Some(vec![bash.clone()]);
        for (tools, choice) in [
            (declared(), ToolChoice::Function("Bsh".into())),
            (declared(), ToolChoice::Function(String::new())),
            (None, ToolChoice::Function("Bash".into())),
            (None, ToolChoice::Required),
            (Some(vec![]), ToolChoice::Required),
        ] {
            let err = check(tools, choice.clone());
            assert!(
                matches!(err, Some(ServeError::Validation(_))),
                "{choice:?}: {err:?}"
            );
        }
        for (tools, choice) in [
            (declared(), ToolChoice::Function("Bash".into())),
            (declared(), ToolChoice::Required),
            (None, ToolChoice::Auto),
            (None, ToolChoice::None),
        ] {
            assert!(check(tools, choice.clone()).is_none(), "{choice:?}");
        }
    }

    /// `tool_choice` forces a tool only when it names one.
    #[test]
    fn a_forced_tool_is_read_from_a_named_function_only() {
        use protocol::ToolChoice;
        let of = |tc: Option<ToolChoice>| ForcedTool::of(&tc);
        assert_eq!(of(None), None);
        assert_eq!(of(Some(ToolChoice::Auto)), None);
        assert_eq!(of(Some(ToolChoice::Required)), None);
        assert_eq!(
            of(Some(ToolChoice::Function("get_weather".into()))),
            Some(ForcedTool("get_weather".to_string()))
        );
    }

    /// What a reply streams for `pieces` (one engine output each, the last finishing it) with
    /// the `parser` tool parser and `forced` as the request's forced tool.
    fn stream_tool_reply(parser: &str, forced: Option<&str>, pieces: &[&str]) -> Vec<StreamDelta> {
        let pieces = (pieces.iter().enumerate())
            .map(|(i, text)| (vec![20 + i as u32], *text))
            .collect();
        let configure = |state: &mut RequestState| {
            state.tool_parser_state = Some(
                crate::tool_parser::get_tool_parser(parser)
                    .unwrap()
                    .create_streaming_state(),
            );
            state.tool_calls = ToolCallStream::new(forced.map(|f| ForcedTool(f.to_string())));
        };
        super::test_support::stream_pieces(configure, pieces, FinishReason::Stop)
    }

    /// The tool-call deltas of `deltas`, in order. A call's id is random per run, so each one
    /// present reads `call`.
    fn tool_calls(deltas: &[StreamDelta]) -> Vec<DeltaToolCall> {
        let calls = deltas
            .iter()
            .flat_map(|d| d.tool_call_deltas.iter().flatten());
        calls
            .map(|d| DeltaToolCall {
                id: d.id.as_ref().map(|_| "call".to_string()),
                ..d.clone()
            })
            .collect()
    }

    /// A whole reply under each `tool_choice`, as both whole-reply paths take it (the chat
    /// endpoint's, and `/v1/messages`' under `tool-spans`, #758): `"none"` reads no call, so the
    /// text comes back as written; a forced tool keeps only its own calls, and the reply is the text
    /// around the rest; anything else keeps every call.
    #[test]
    fn a_whole_reply_keeps_the_calls_its_tool_choice_lets_out() {
        use protocol::ToolChoice;
        let parser = crate::tool_parser::get_tool_parser("hermes").unwrap();
        let text = format!("Let me look.{}", BASH_CALL.concat());
        let reply = |choice: Option<ToolChoice>| {
            let reply = reply_tool_calls(parser.as_ref(), text.clone(), &choice);
            let names = (reply.calls.iter().flatten())
                .map(|tc| tc.function.name.clone())
                .collect::<Vec<_>>();
            (reply.content, names)
        };
        let calls_bash = (Some("Let me look.".to_string()), vec!["Bash".to_string()]);
        assert_eq!(reply(None), calls_bash);
        assert_eq!(reply(Some(ToolChoice::Auto)), calls_bash);
        assert_eq!(reply(Some(ToolChoice::Required)), calls_bash);
        assert_eq!(reply(Some(ToolChoice::Function("Bash".into()))), calls_bash);
        assert_eq!(
            reply(Some(ToolChoice::Function("Read".into()))),
            (Some("Let me look.".to_string()), vec![])
        );
        assert_eq!(reply(Some(ToolChoice::None)), (Some(text.clone()), vec![]));
    }

    /// A call to Bash, written across five tokens.
    const BASH_CALL: [&str; 5] = [
        "<tool_call>\n",
        "{\"name\": \"Bash\", ",
        "\"arguments\": {\"command\": ",
        "\"rm -rf build\"}}",
        "\n</tool_call>",
    ];

    /// A forced tool_choice drops a call to another tool whole: the deltas after its first carry
    /// no name, and none of its arguments may reach the stream (#683). The reply then is the text
    /// around the call — none here — and finishes `Stop`, with nothing a stream encoder would
    /// turn into a tool-call finish.
    #[test]
    fn a_forced_tool_choice_streams_no_part_of_another_tools_call() {
        let unforced = tool_calls(&stream_tool_reply("hermes", None, &BASH_CALL));
        assert_eq!(unforced[0].function_name.as_deref(), Some("Bash"));
        assert!(
            unforced[1..].iter().any(|d| d.function_arguments.is_some()),
            "unforced, the call streams its arguments after its name: {unforced:?}"
        );

        let forced = stream_tool_reply("hermes", Some("Read"), &BASH_CALL);
        assert!(tool_calls(&forced).is_empty(), "{forced:?}");
        assert!(
            forced
                .iter()
                .all(|d| d.text.as_deref().is_none_or(str::is_empty)),
            "no text around the call, and none of the call's own: {forced:?}"
        );
        let finishes: Vec<_> = forced.iter().filter_map(|d| d.finish_reason).collect();
        assert_eq!(finishes, [FinishReason::Stop]);

        let kept = stream_tool_reply("hermes", Some("Bash"), &BASH_CALL);
        assert_eq!(tool_calls(&kept), unforced);
    }

    /// The calls a forced tool_choice keeps are numbered from 0, the places they hold in the
    /// non-streaming response's list, whatever calls before them it dropped.
    #[test]
    fn a_forced_tool_choice_numbers_the_calls_it_keeps_from_zero() {
        let mut pieces = BASH_CALL.to_vec();
        pieces.extend([
            "\n<tool_call>\n",
            "{\"name\": \"Read\", ",
            "\"arguments\": {\"file_path\": \"a.rs\"}}",
            "\n</tool_call>",
        ]);
        let unforced = tool_calls(&stream_tool_reply("hermes", None, &pieces));
        let read_index = unforced
            .iter()
            .find(|d| d.function_name.as_deref() == Some("Read"))
            .unwrap()
            .index;
        assert_ne!(read_index, 0, "the parser numbers Read after Bash");

        let read: Vec<_> = unforced
            .into_iter()
            .filter(|d| d.index == read_index)
            .map(|d| DeltaToolCall { index: 0, ..d })
            .collect();
        assert!(read.iter().any(|d| d.function_arguments.is_some()));
        assert_eq!(
            tool_calls(&stream_tool_reply("hermes", Some("Read"), &pieces)),
            read,
            "Read's deltas, at index 0"
        );
    }

    /// Mistral's pre-v11 format takes a call's keys in the order the model writes them, so its
    /// arguments can stream before its name. The output path holds them until the name arrives:
    /// every consumer reads a call's first delta as its start (`/v1/messages` opened a `tool_use`
    /// block named ""), and a forced tool_choice decides the call on its name (it dropped the
    /// arguments and kept the name).
    #[test]
    fn a_calls_name_streams_before_its_arguments() {
        let pieces = [
            "[TOOL_CALLS]",
            " [{\"arguments\": {\"file_path\": ",
            "\"a.rs\"}",
            ", \"name\": \"Read\"}]",
        ];
        // The parser itself writes the arguments first: without the hold, the stream would too.
        let mut parser = crate::tool_parser::get_tool_parser("mistral")
            .unwrap()
            .create_streaming_state();
        let mut text = String::new();
        let mut parsed = Vec::new();
        for piece in pieces {
            let previous = text.clone();
            text.push_str(piece);
            if let ToolParserDelta::ToolCalls(d) = parser.process_delta(&previous, &text, piece) {
                parsed.extend(d);
            }
        }
        assert!(
            parsed[0].function_name.is_none() && parsed[0].function_arguments.is_some(),
            "{parsed:?}"
        );

        for forced in [None, Some("Read")] {
            let calls = tool_calls(&stream_tool_reply("mistral", forced, &pieces));
            assert_eq!(
                calls[0].function_name.as_deref(),
                Some("Read"),
                "{forced:?}"
            );
            assert!(calls.iter().all(|d| d.index == 0), "{forced:?}");
            let args: String = calls
                .iter()
                .filter_map(|d| d.function_arguments.as_deref())
                .collect();
            let args: serde_json::Value = serde_json::from_str(&args).unwrap();
            assert_eq!(args, serde_json::json!({"file_path": "a.rs"}), "{forced:?}");
        }
        let other = stream_tool_reply("mistral", Some("Bash"), &pieces);
        assert!(tool_calls(&other).is_empty(), "{other:?}");
    }

    // -------------------------------------------------------------------
    // Pooling mode tests
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn test_pooling_mode_rejects_chat_completion() {
        let mut engine = make_test_engine();
        engine.set_is_pooling(true);
        let engine = Arc::new(engine);
        engine.spawn_step_loop();

        let request: protocol::ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "Hello"}]
        }))
        .unwrap();
        let result = engine.chat_completion(request).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("pooling mode"),
            "Error should mention pooling mode, got: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_pooling_mode_rejects_completion() {
        let mut engine = make_test_engine();
        engine.set_is_pooling(true);
        let engine = Arc::new(engine);
        engine.spawn_step_loop();

        let request: protocol::CompletionRequest =
            serde_json::from_value(serde_json::json!({"prompt": "Hello"})).unwrap();
        let result = engine.completion(request).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("pooling mode"),
            "Error should mention pooling mode, got: {err_msg}"
        );
    }

    #[test]
    fn test_run_output_stores_pooler_output() {
        let mut requests = HashMap::new();
        requests.insert("pool-1".to_string(), make_test_request_state(None));

        let embedding = EmbeddingData::Single(vec![0.1, 0.2, 0.3]);
        run_output(
            &mut requests,
            EngineCoreOutput {
                request_id: "pool-1".to_string(),
                new_token_ids: vec![],
                finish_reason: Some(FinishReason::Stop),
                stop_reason: None,
                num_cached_tokens: 0,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: Some(embedding.clone()),
            },
        );

        // Request should be removed (finished + no stream).
        // Check that pooler_output was set before removal by using streaming.
        let mut requests2 = HashMap::new();
        let (tx, _rx) = mpsc::unbounded_channel();
        requests2.insert("pool-2".to_string(), make_test_request_state(Some(tx)));

        run_output(
            &mut requests2,
            EngineCoreOutput {
                request_id: "pool-2".to_string(),
                new_token_ids: vec![],
                finish_reason: None, // Not finished yet, so state stays.
                stop_reason: None,
                num_cached_tokens: 0,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: Some(embedding.clone()),
            },
        );

        let state = requests2.get("pool-2").unwrap();
        assert_eq!(state.pooler_output, Some(embedding));
    }

    #[test]
    fn test_pooling_mode_flag() {
        let mut engine = make_test_engine();
        assert!(!engine.is_pooling());
        engine.set_is_pooling(true);
        assert!(engine.is_pooling());
    }

    // -----------------------------------------------------------------------
    // FailingExecutor — always returns Err from execute_model
    // -----------------------------------------------------------------------

    /// An executor that always fails on `execute_model`.
    /// Used to test the error-abort path in the step loops.
    struct FailingExecutor;

    impl scratchy_serving_engine::executor::Executor for FailingExecutor {
        /// It samples no grammar.
        fn set_stop_token_ids(&mut self, _stop_token_ids: &[u32]) {}

        fn execute_model(
            &mut self,
            _scheduler_output: &scratchy_serving_scheduler::scheduler::output::SchedulerOutput,
        ) -> scratchy_serving_engine::error::EngineResult<
            scratchy_serving_engine::executor::ModelRunnerOutput,
        > {
            Err(scratchy_serving_engine::error::EngineError::Executor(
                "simulated forward pass failure".into(),
            ))
        }

        fn initialize_cache(
            &mut self,
            _num_gpu_blocks: usize,
            _num_cpu_blocks: usize,
        ) -> scratchy_serving_engine::error::EngineResult<()> {
            Ok(())
        }

        fn determine_available_memory(
            &mut self,
        ) -> scratchy_serving_engine::error::EngineResult<Vec<usize>> {
            Ok(vec![1024 * 16 * 1024])
        }

        fn shutdown(&mut self) {}

        fn is_sleeping(&self) -> bool {
            false
        }
    }

    // -----------------------------------------------------------------------
    // PanickingExecutor — panics in execute_model
    // -----------------------------------------------------------------------

    /// An executor whose forward pass panics: the engine dies mid-request, as a detokenizer
    /// slicing inside a character took it down in #576.
    struct PanickingExecutor;

    impl scratchy_serving_engine::executor::Executor for PanickingExecutor {
        /// It samples no grammar.
        fn set_stop_token_ids(&mut self, _stop_token_ids: &[u32]) {}

        fn execute_model(
            &mut self,
            _scheduler_output: &scratchy_serving_scheduler::scheduler::output::SchedulerOutput,
        ) -> scratchy_serving_engine::error::EngineResult<
            scratchy_serving_engine::executor::ModelRunnerOutput,
        > {
            panic!("simulated forward pass panic")
        }

        fn initialize_cache(
            &mut self,
            _num_gpu_blocks: usize,
            _num_cpu_blocks: usize,
        ) -> scratchy_serving_engine::error::EngineResult<()> {
            Ok(())
        }

        fn determine_available_memory(
            &mut self,
        ) -> scratchy_serving_engine::error::EngineResult<Vec<usize>> {
            Ok(vec![1024 * 16 * 1024])
        }

        fn shutdown(&mut self) {}

        fn is_sleeping(&self) -> bool {
            false
        }
    }

    fn make_engine_on(executor: Box<dyn Executor>, async_scheduling: bool) -> AsyncEngine {
        let mut config = make_engine_config();
        config.async_scheduling = async_scheduling;
        let client = Box::new(InprocClient::new(config, executor));
        let mut engine = AsyncEngine::new(client, "test-model".to_string(), 4096);
        engine.set_async_scheduling(async_scheduling);
        engine
    }

    /// A mock client that always stalls (simulates the scheduler never scheduling any requests —
    /// the no-progress bug). Now says so with [`StepOutcome::Stalled`] rather than by handing back
    /// empty outputs and a `false`, which is the same claim the real engine makes.
    struct NoProgressClient;

    impl EngineCoreClient for NoProgressClient {
        fn get_output(
            &mut self,
        ) -> scratchy_serving_engine::error::EngineResult<
            StepOutcome<scratchy_core_common::EngineCoreOutputs>,
        > {
            // Simulate scheduler returning nothing to schedule.
            std::thread::sleep(std::time::Duration::from_millis(10));
            Ok(StepOutcome::Stalled)
        }

        fn add_request(
            &mut self,
            _request: EngineCoreRequest,
        ) -> scratchy_serving_engine::error::EngineResult<()> {
            Ok(())
        }

        fn abort_requests(
            &mut self,
            _request_ids: &[String],
        ) -> scratchy_serving_engine::error::EngineResult<()> {
            Ok(())
        }

        // It never schedules a step, so none can fail.
        fn abandon_step(
            &mut self,
            _sched: &scratchy_serving_scheduler::scheduler::output::SchedulerOutput,
        ) -> Vec<String> {
            Vec::new()
        }

        fn abort_running_requests(&mut self) {}

        fn shutdown(&mut self) -> scratchy_serving_engine::error::EngineResult<()> {
            Ok(())
        }

        fn reset_prefix_cache(&mut self) -> scratchy_serving_engine::error::EngineResult<bool> {
            Ok(true)
        }

        fn pause_scheduler(
            &mut self,
            _mode: PauseMode,
        ) -> scratchy_serving_engine::error::EngineResult<()> {
            Ok(())
        }

        fn resume_scheduler(&mut self) -> scratchy_serving_engine::error::EngineResult<()> {
            Ok(())
        }

        fn is_scheduler_paused(&self) -> bool {
            false
        }

        fn sleep(&mut self, _level: u32) -> scratchy_serving_engine::error::EngineResult<()> {
            Ok(())
        }

        fn wake_up(
            &mut self,
            _tags: Option<&[String]>,
        ) -> scratchy_serving_engine::error::EngineResult<()> {
            Ok(())
        }

        fn is_sleeping(&self) -> bool {
            false
        }
    }

    fn make_no_progress_engine() -> AsyncEngine {
        let client: Box<dyn EngineCoreClient + Send> = Box::new(NoProgressClient);
        let mut engine = AsyncEngine::new(client, "test-model".to_string(), 4096);
        engine.set_no_progress_timeout(Duration::from_secs(2));
        engine
    }

    /// The stop-string abort test's shared id recorders (added ids, aborted ids).
    type IdLog = Arc<std::sync::Mutex<Vec<String>>>;

    /// A client whose every request decodes to text containing the stop word, and which records
    /// every abort it is asked for. The stop word finishes each request API-side while the
    /// engine-side request would decode on to max_tokens: the step loop must abort it there
    /// (#711 — the request after a stop-string finish decoded garbage).
    struct StopStringClient {
        /// The stop word's token ids, ENCODED by the same tokenizer the engine detokenizes with
        /// (the test vocab is the sorted byte-level alphabet, so an id is not its byte value).
        stop_word_tokens: Vec<u32>,
        added: IdLog,
        aborted: IdLog,
        emitted: std::sync::Mutex<std::collections::HashSet<String>>,
    }

    impl StopStringClient {
        fn new(stop_word_tokens: Vec<u32>) -> (Self, IdLog, IdLog) {
            let added = Arc::new(std::sync::Mutex::new(Vec::new()));
            let aborted = Arc::new(std::sync::Mutex::new(Vec::new()));
            (
                Self {
                    stop_word_tokens,
                    added: Arc::clone(&added),
                    aborted: Arc::clone(&aborted),
                    emitted: std::sync::Mutex::new(std::collections::HashSet::new()),
                },
                added,
                aborted,
            )
        }
    }

    impl EngineCoreClient for StopStringClient {
        fn get_output(
            &mut self,
        ) -> scratchy_serving_engine::error::EngineResult<
            StepOutcome<scratchy_core_common::EngineCoreOutputs>,
        > {
            let pending: Vec<String> = {
                let added = self.added.lock().unwrap().clone();
                let mut emitted = self.emitted.lock().unwrap();
                added
                    .into_iter()
                    .filter(|id| emitted.insert(id.clone()))
                    .collect()
            };
            if pending.is_empty() {
                std::thread::sleep(std::time::Duration::from_millis(1));
                return Ok(StepOutcome::Stalled);
            }
            let outputs = pending
                .into_iter()
                .map(|id| EngineCoreOutput {
                    request_id: id,
                    new_token_ids: self.stop_word_tokens.clone(),
                    finish_reason: None,
                    stop_reason: None,
                    num_cached_tokens: 0,
                    events: None,
                    new_logprobs: None,
                    new_prompt_logprobs: None,
                    pooler_output: None,
                })
                .collect();
            Ok(StepOutcome::Progressed {
                outputs: scratchy_core_common::EngineCoreOutputs {
                    outputs,
                    ..Default::default()
                },
                model_executed: true,
            })
        }

        fn add_request(
            &mut self,
            request: EngineCoreRequest,
        ) -> scratchy_serving_engine::error::EngineResult<()> {
            self.added.lock().unwrap().push(request.request_id);
            Ok(())
        }

        fn abort_requests(
            &mut self,
            request_ids: &[String],
        ) -> scratchy_serving_engine::error::EngineResult<()> {
            self.aborted.lock().unwrap().extend_from_slice(request_ids);
            Ok(())
        }

        // As the engine does: a failed step's requests are aborted.
        fn abandon_step(
            &mut self,
            sched: &scratchy_serving_scheduler::scheduler::output::SchedulerOutput,
        ) -> Vec<String> {
            let ids: Vec<String> = sched.num_scheduled_tokens.keys().cloned().collect();
            self.aborted.lock().unwrap().extend_from_slice(&ids);
            ids
        }

        fn abort_running_requests(&mut self) {}

        fn shutdown(&mut self) -> scratchy_serving_engine::error::EngineResult<()> {
            Ok(())
        }

        fn reset_prefix_cache(&mut self) -> scratchy_serving_engine::error::EngineResult<bool> {
            Ok(true)
        }

        fn pause_scheduler(
            &mut self,
            _mode: PauseMode,
        ) -> scratchy_serving_engine::error::EngineResult<()> {
            Ok(())
        }

        fn resume_scheduler(&mut self) -> scratchy_serving_engine::error::EngineResult<()> {
            Ok(())
        }

        fn is_scheduler_paused(&self) -> bool {
            false
        }

        fn sleep(&mut self, _level: u32) -> scratchy_serving_engine::error::EngineResult<()> {
            Ok(())
        }

        fn wake_up(
            &mut self,
            _tags: Option<&[String]>,
        ) -> scratchy_serving_engine::error::EngineResult<()> {
            Ok(())
        }

        fn is_sleeping(&self) -> bool {
            false
        }
    }

    /// Stop → stop: each request finishes by its stop string API-side while the engine would
    /// keep decoding it — the loop must abort each one engine-side, or the request after a
    /// stop-string finish shares its steps with the leftover row (#711).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_stop_string_finish_aborts_the_engine_side_request() {
        let tok = Arc::new(make_test_tokenizer());
        // " the country" — decodes to text matching a `stop: ["country"]` request.
        let stop_word_tokens = tok.encode(" the country", false).unwrap();
        let (client, added, aborted) = StopStringClient::new(stop_word_tokens);
        let client: Box<dyn EngineCoreClient + Send> = Box::new(client);
        let engine = Arc::new(AsyncEngine::with_tokenizer(
            client,
            "test-model".to_string(),
            4096,
            tok,
        ));
        engine.spawn_step_loop();

        let mut request = make_chat_request();
        request.stop = Some(protocol::StopCondition::Multiple(vec![
            "country".to_string(),
        ]));
        request.max_tokens = Some(50);

        let a = tokio::time::timeout(
            Duration::from_secs(10),
            engine.chat_completion(request.clone()),
        )
        .await
        .expect("request A timed out — the stop string never finished it")
        .expect("request A failed");
        assert_eq!(a.choices[0].finish_reason.as_deref(), Some("stop"));
        assert!(
            !a.choices[0]
                .message
                .content
                .as_deref()
                .unwrap_or("")
                .contains("country"),
            "the reply is truncated at the stop string"
        );

        // The request right after the stop-string finish — the one #711 caught decoding
        // garbage. Its own stop string must finish it the same way.
        let b = tokio::time::timeout(Duration::from_secs(10), engine.chat_completion(request))
            .await
            .expect("request B timed out — the stop string never finished it")
            .expect("request B failed");
        assert_eq!(b.choices[0].finish_reason.as_deref(), Some("stop"));

        let added = added.lock().unwrap().clone();
        let aborted = aborted.lock().unwrap().clone();
        assert_eq!(
            added.len(),
            2,
            "both requests reached the engine: {added:?}"
        );
        assert_eq!(
            aborted.len(),
            2,
            "every stop-string-finished request is aborted engine-side — without the abort the \
             engine decodes it on to max_tokens under the next request's steps (#711); \
             aborted {aborted:?} of {added:?}"
        );
    }

    /// Test that the no-progress watchdog aborts requests stuck with no output.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_no_progress_watchdog() {
        let engine = Arc::new(make_no_progress_engine());
        engine.spawn_step_loop();

        let request = make_chat_request();

        // Submit a request. The mock client never produces output tokens,
        // so the watchdog should abort it after ~2 seconds.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            engine.chat_completion(request),
        )
        .await;

        // Must not time out (watchdog should fire within ~2s).
        let inner = result.expect("request timed out — no-progress watchdog did not fire");

        // Must be an error (aborted by watchdog).
        let err = inner.expect_err("expected error from no-progress watchdog");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("No-progress watchdog") || err_msg.contains("step loop exited"),
            "unexpected error: {err_msg}"
        );
    }

    /// A STREAMING request the engine never answers must still end: the watchdog sends its
    /// stream a final `Abort` delta and drops its entry. Marking the entry alone — which nothing
    /// streaming reads — left the client waiting forever and the watchdog re-firing every timeout.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_no_progress_watchdog_ends_streams() {
        let engine = Arc::new(make_no_progress_engine());
        engine.spawn_step_loop();

        let (_, _, mut rx) = engine
            .chat_completion_stream(make_chat_request())
            .await
            .expect("stream starts");
        let last = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            let mut last = None;
            while let Some(delta) = rx.recv().await {
                last = Some(delta);
            }
            last
        })
        .await
        .expect("the stream never ended — the watchdog did not end it");

        assert_eq!(
            last.and_then(|d| d.finish_reason),
            Some(FinishReason::Abort),
            "the stream's last delta says why it ended"
        );
        assert!(
            engine.requests.lock().await.is_empty(),
            "the ended stream's entry must go, or the watchdog re-fires forever"
        );
    }

    /// A step loop that dies — its forward pass panics — ends the requests it was serving. A
    /// dead loop left a stream waiting on a sender nothing would send on again, and a server's
    /// graceful shutdown waiting on that connection forever (#587). The sync loop panics itself;
    /// the async one loses its executor thread.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_a_dead_step_loop_ends_its_requests() {
        for async_scheduling in [false, true] {
            let engine = Arc::new(make_engine_on(
                Box::new(PanickingExecutor),
                async_scheduling,
            ));
            engine.spawn_step_loop();

            let (_, _, mut rx) = engine
                .chat_completion_stream(make_chat_request())
                .await
                .expect("stream starts");
            let last = tokio::time::timeout(Duration::from_secs(10), async {
                let mut last = None;
                while let Some(delta) = rx.recv().await {
                    last = Some(delta);
                }
                last
            })
            .await
            .unwrap_or_else(|_| {
                panic!("async={async_scheduling}: the dead loop left the stream open")
            });
            assert_eq!(
                last.and_then(|d| d.finish_reason),
                Some(FinishReason::Abort),
                "async={async_scheduling}: the stream's last delta says why it ended"
            );

            // And the loop says it ended, which is what shuts a server down.
            let end = tokio::time::timeout(Duration::from_secs(10), engine.step_loop_ended())
                .await
                .unwrap_or_else(|_| panic!("async={async_scheduling}: the end was never said"));
            let expected = if async_scheduling {
                StepLoopEnd::Exited // its executor thread is the one that panicked
            } else {
                StepLoopEnd::Panicked("simulated forward pass panic".to_string())
            };
            assert_eq!(end, expected, "async={async_scheduling}");
            assert_eq!(engine.step_loop_end(), Some(expected));
            assert!(
                engine.requests.lock().await.is_empty(),
                "async={async_scheduling}"
            );
        }
    }

    /// A server whose engine dies ends by itself, with the engine's error. Left up, it answered
    /// every request with a 500 and held its port, and a SIGTERM's graceful shutdown waited
    /// forever on the stream the dead engine had been serving (#587) — here, a real HTTP one.
    #[cfg(feature = "serve")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_a_server_ends_when_its_engine_dies() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let engine = Arc::new(make_engine_on(Box::new(PanickingExecutor), false));
        engine.spawn_step_loop();
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .and_then(|listener| listener.local_addr())
            .expect("a free port")
            .port();
        let state = Arc::new(crate::server::AppState {
            engine,
            config: crate::server::ServerConfig {
                bind_address: format!("127.0.0.1:{port}"),
                ..Default::default()
            },
            is_pooling: false,
            scratchy_core_config: None,
            requests_served: std::sync::atomic::AtomicU64::new(0),
        });

        let stream = async {
            let mut conn = loop {
                match tokio::net::TcpStream::connect(("127.0.0.1", port)).await {
                    Ok(conn) => break conn,
                    Err(_) => tokio::time::sleep(Duration::from_millis(10)).await,
                }
            };
            let body = r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}"#;
            let request = format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            conn.write_all(request.as_bytes())
                .await
                .expect("request sent");
            let mut response = String::new();
            conn.read_to_string(&mut response)
                .await
                .expect("response read");
            response
        };
        let (served, response) = tokio::time::timeout(Duration::from_secs(10), async {
            tokio::join!(crate::server::serve(state), stream)
        })
        .await
        .expect("the server outlived its engine, or its stream did");

        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(response.contains("[DONE]"), "the stream ends: {response}");
        let err = served.expect_err("a dead engine ends its server with an error");
        assert!(
            err.to_string().contains("simulated forward pass panic"),
            "{err}"
        );
    }

    /// Test that a forward pass failure in the **sync** step loop aborts the
    /// request and returns an error instead of hanging forever.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_sync_loop_aborts_on_executor_error() {
        let engine = Arc::new(make_engine_on(Box::new(FailingExecutor), false));
        engine.spawn_step_loop();

        let request = make_chat_request();

        // This should NOT hang — the executor error should abort the request
        // and poll_until_done should return an error.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            engine.chat_completion(request),
        )
        .await;

        // Must not time out.
        let inner = result.expect("request timed out — step loop hang not fixed");

        // Must be an error (not a successful completion).
        let err = inner.expect_err("expected error from failing executor");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("simulated forward pass failure")
                || err_msg.contains("Engine step error")
                || err_msg.contains("step loop exited"),
            "unexpected error: {err_msg}"
        );
    }

    /// Test that a forward pass failure in the **async** step loop aborts the
    /// request and returns an error instead of hanging forever.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_async_loop_aborts_on_executor_error() {
        let engine = Arc::new(make_engine_on(Box::new(FailingExecutor), true));
        engine.spawn_step_loop();

        let request = make_chat_request();

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            engine.chat_completion(request),
        )
        .await;

        let inner = result.expect("request timed out — async step loop hang not fixed");

        let err = inner.expect_err("expected error from failing executor");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("simulated forward pass failure")
                || err_msg.contains("Executor error")
                || err_msg.contains("step loop exited"),
            "unexpected error: {err_msg}"
        );
    }

    /// An executor whose steps resolve as a device's do: each output deferred, its resolve
    /// blocking its thread for a 2 ms device step — by when the step behind it has its output
    /// queued — and then until the request's stream has received every step before it, what a
    /// device step's wait overlaps when the step loop holds no stream behind it, or until a
    /// second passes, a stall it counts.
    struct DeviceStepExecutor {
        inner: NoopExecutor,
        steps: usize,
        streamed: Arc<std::sync::atomic::AtomicUsize>,
        stalls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl scratchy_serving_engine::executor::Executor for DeviceStepExecutor {
        /// It samples no grammar.
        fn set_stop_token_ids(&mut self, _stop_token_ids: &[u32]) {}

        fn execute_model(
            &mut self,
            scheduler_output: &scratchy_serving_scheduler::scheduler::output::SchedulerOutput,
        ) -> scratchy_serving_engine::error::EngineResult<
            scratchy_serving_engine::executor::ModelRunnerOutput,
        > {
            use std::sync::atomic::Ordering::SeqCst;
            let output = self.inner.execute_model(scheduler_output)?;
            let before = self.steps;
            self.steps += 1;
            let (streamed, stalls) = (Arc::clone(&self.streamed), Arc::clone(&self.stalls));
            let tokens = output.sampled_token_ids.into_iter().flatten().collect();
            Ok(
                scratchy_serving_engine::executor::ModelRunnerOutput::deferred(
                    output.req_ids,
                    Box::new(move || {
                        std::thread::sleep(Duration::from_millis(2));
                        let deadline = Instant::now() + Duration::from_secs(1);
                        while streamed.load(SeqCst) < before {
                            if Instant::now() > deadline {
                                stalls.fetch_add(1, SeqCst);
                                break;
                            }
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        Ok(tokens)
                    }),
                ),
            )
        }

        fn initialize_cache(
            &mut self,
            num_gpu_blocks: usize,
            num_cpu_blocks: usize,
        ) -> scratchy_serving_engine::error::EngineResult<()> {
            self.inner.initialize_cache(num_gpu_blocks, num_cpu_blocks)
        }

        fn determine_available_memory(
            &mut self,
        ) -> scratchy_serving_engine::error::EngineResult<Vec<usize>> {
            self.inner.determine_available_memory()
        }

        fn shutdown(&mut self) {
            self.inner.shutdown()
        }

        fn is_sleeping(&self) -> bool {
            self.inner.is_sleeping()
        }
    }

    /// The async step loop waits out a deferred step's device work without holding the streams
    /// it just sent a step's tokens to: each reaches its reader while the next step resolves.
    /// Waited for on the loop's own worker, a reader that step had woken sat in that worker's
    /// LIFO slot, which no other worker steals, until the loop next yielded — its tokens went
    /// out in bursts.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_async_loop_streams_each_step_while_the_next_resolves() {
        use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
        let (streamed, stalls) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let executor = DeviceStepExecutor {
            inner: NoopExecutor::new(1024),
            steps: 0,
            streamed: Arc::clone(&streamed),
            stalls: Arc::clone(&stalls),
        };
        // The scheduler's own flag too: a step is scheduled behind the one in flight, so the
        // loop's wait for its result is ready at once, as a device step's is.
        let mut config = make_engine_config();
        config.async_scheduling = true;
        config.scheduler_config.async_scheduling = Some(true);
        let client = Box::new(InprocClient::new(config, Box::new(executor)));
        let mut engine = AsyncEngine::new(client, "test-model".to_string(), 4096);
        engine.set_async_scheduling(true);
        let engine = Arc::new(engine);
        engine.spawn_step_loop();

        let mut request =
            make_completion_request(Some(protocol::CompletionPrompt::TokenIds(vec![1, 2, 3])));
        request.max_tokens = Some(64);
        request.stream = true;
        let (_, _, mut rx) = engine.completion_stream(request).await.unwrap();
        // The stream's reader is a task of the runtime, as a server connection's is.
        let reader = Arc::clone(&streamed);
        tokio::spawn(async move {
            while rx.recv().await.is_some() {
                reader.fetch_add(1, SeqCst);
            }
        })
        .await
        .unwrap();

        assert!(streamed.load(SeqCst) >= 64, "every token streamed");
        assert_eq!(
            stalls.load(SeqCst),
            0,
            "steps resolved with their stream held"
        );
    }

    /// A failed speculative step (an MTP head plans drafts for every request it admits) is
    /// never finalized, so the async loop must stop waiting on it: the next request is
    /// scheduled — and fails — at once, not left pending for the no-progress watchdog (#477).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_async_loop_schedules_again_after_a_speculative_step_fails() {
        let mut config = make_engine_config();
        config.async_scheduling = true;
        config.use_spec_decode = true;
        config.proposer_config = Some(scratchy_serving_engine::spec_decode::ProposerConfig::Mtp(
            scratchy_serving_engine::spec_decode::MtpProposerConfig {
                num_speculative_tokens: 2,
                max_seqs: 2,
                max_model_len: 4096,
            },
        ));
        let client = Box::new(InprocClient::new(config, Box::new(FailingExecutor)));
        let mut engine = AsyncEngine::new(client, "test-model".to_string(), 4096);
        engine.set_async_scheduling(true);
        let engine = Arc::new(engine);
        engine.spawn_step_loop();

        for attempt in ["first", "second"] {
            let result = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                engine.chat_completion(make_chat_request()),
            )
            .await;
            let inner = result.unwrap_or_else(|_| panic!("{attempt} request was never scheduled"));
            inner.expect_err("expected error from failing executor");
        }
    }

    #[test]
    fn test_render_chat_completion() {
        let engine = make_test_engine_with_tokenizer();
        let request = make_chat_request();
        let (conversation, engine_prompts) = engine.render_chat_completion(request).unwrap();

        // Conversation should contain the original message.
        assert_eq!(conversation.len(), 1);
        assert_eq!(conversation[0]["role"], "user");
        assert_eq!(conversation[0]["content"], "Hello");

        // Engine prompts should have exactly one entry with a non-empty prompt.
        assert_eq!(engine_prompts.len(), 1);
        let prompt = engine_prompts[0].prompt.as_str().unwrap();
        assert!(!prompt.is_empty(), "rendered prompt should not be empty");
    }

    #[test]
    fn test_render_chat_completion_pooling_mode_rejected() {
        let mut engine = make_test_engine();
        engine.set_is_pooling(true);
        let request = make_chat_request();
        let result = engine.render_chat_completion(request);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("pooling mode"));
    }

    #[tokio::test]
    async fn test_reset_prefix_cache_via_control_channel() {
        let engine = Arc::new(make_test_engine());
        engine.spawn_step_loop();

        // No running requests — should succeed.
        let result = engine.reset_prefix_cache().await.unwrap();
        assert!(
            result,
            "reset_prefix_cache should return true with no running requests"
        );
    }

    // -------------------------------------------------------------------
    // Reasoning parser integration tests
    // -------------------------------------------------------------------

    /// Helper: build a minimal vocab for reasoning parser tests.
    pub(super) fn reasoning_test_vocab() -> std::collections::HashMap<String, u32> {
        let mut vocab = std::collections::HashMap::new();
        vocab.insert("<think>".to_string(), 100);
        vocab.insert("</think>".to_string(), 101);
        vocab
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_chat_completion_with_deepseek_reasoning_parser() {
        // NoopExecutor generates placeholder text "<token_1000><token_1001>..."
        // which doesn't contain <think>/<think>, so DeepSeek R1 treats it as
        // "everything is reasoning" (no end token found).
        let vocab = reasoning_test_vocab();
        let parser = crate::reasoning_parser::get_reasoning_parser("deepseek_r1", &vocab).unwrap();

        let mut engine = make_test_engine();
        engine.set_reasoning_parser(parser);
        let engine = Arc::new(engine);
        engine.spawn_step_loop();

        let request = make_chat_request();
        let response = engine.chat_completion(request).await.unwrap();

        assert_eq!(response.choices.len(), 1);
        let msg = &response.choices[0].message;
        // DeepSeek: no </think> found → everything is reasoning, content is None.
        assert!(
            msg.reasoning.is_some(),
            "reasoning should be populated for DeepSeek R1"
        );
        assert!(
            msg.content.is_none(),
            "content should be None when no </think>"
        );
    }

    /// A tool parser that reads one Bash call beside fixed text out of any reply, so a
    /// non-streaming test can see what the response keeps of it (NoopExecutor's text is
    /// placeholders).
    struct OneBashCall;

    impl crate::tool_parser::ToolCallParser for OneBashCall {
        fn extract_tool_calls(&self, _: &str) -> crate::tool_parser::ExtractedToolCallInfo {
            crate::tool_parser::ExtractedToolCallInfo {
                tools_called: true,
                tool_calls: vec![protocol::ToolCall {
                    id: "call_1".to_string(),
                    call_type: "function".to_string(),
                    function: protocol::FunctionCall {
                        name: "Bash".to_string(),
                        arguments: "{\"command\": \"ls\"}".to_string(),
                    },
                }],
                content: Some("Let me look.".to_string()),
            }
        }

        fn create_streaming_state(&self) -> Box<dyn StreamingToolParserState + Send> {
            crate::tool_parser::get_tool_parser("hermes")
                .unwrap()
                .create_streaming_state()
        }
    }

    /// The non-streaming reply drops a call to a tool other than the forced one, as the stream
    /// does, and is the text around it — not the raw output, and not a tool-call finish (#683).
    /// Forcing the tool the model called keeps the call (the control).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_chat_completion_drops_a_call_to_a_tool_not_forced() {
        let mut engine = make_test_engine();
        engine.set_tool_parser(Arc::new(OneBashCall));
        let engine = Arc::new(engine);
        engine.spawn_step_loop();
        let request = |forced: &str| protocol::ChatCompletionRequest {
            tools: Some(vec![protocol::ChatCompletionToolsParam {
                tool_type: "function".to_string(),
                function: protocol::FunctionDefinition {
                    name: forced.to_string(),
                    description: None,
                    parameters: None,
                },
            }]),
            tool_choice: Some(protocol::ToolChoice::Function(forced.to_string())),
            max_tokens: Some(4),
            ..make_chat_request()
        };

        let kept = engine.chat_completion(request("Bash")).await.unwrap();
        let choice = &kept.choices[0];
        let calls = choice.message.tool_calls.as_ref().expect("the Bash call");
        assert_eq!(calls[0].function.name, "Bash");
        assert_eq!(choice.message.content.as_deref(), Some("Let me look."));
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));

        let dropped = engine.chat_completion(request("Read")).await.unwrap();
        let choice = &dropped.choices[0];
        assert!(choice.message.tool_calls.is_none(), "{choice:?}");
        assert_eq!(choice.message.content.as_deref(), Some("Let me look."));
        assert_eq!(
            choice.finish_reason.as_deref(),
            Some("length"),
            "max_tokens ended it"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_chat_completion_with_qwen3_reasoning_parser() {
        // Qwen3: no </think> found → thinking disabled, everything is content.
        let vocab = reasoning_test_vocab();
        let parser = crate::reasoning_parser::get_reasoning_parser("qwen3", &vocab).unwrap();

        let mut engine = make_test_engine();
        engine.set_reasoning_parser(parser);
        let engine = Arc::new(engine);
        engine.spawn_step_loop();

        let request = make_chat_request();
        let response = engine.chat_completion(request).await.unwrap();

        assert_eq!(response.choices.len(), 1);
        let msg = &response.choices[0].message;
        // Qwen3: no </think> → thinking disabled, everything is content.
        assert!(
            msg.reasoning.is_none(),
            "reasoning should be None for Qwen3 thinking-disabled"
        );
        assert!(msg.content.is_some(), "content should be populated");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_chat_completion_include_reasoning_false() {
        // include_reasoning=false should suppress the reasoning field.
        let vocab = reasoning_test_vocab();
        let parser = crate::reasoning_parser::get_reasoning_parser("deepseek_r1", &vocab).unwrap();

        let mut engine = make_test_engine();
        engine.set_reasoning_parser(parser);
        let engine = Arc::new(engine);
        engine.spawn_step_loop();

        let mut request = make_chat_request();
        request.include_reasoning = false;
        let response = engine.chat_completion(request).await.unwrap();

        assert_eq!(response.choices.len(), 1);
        let msg = &response.choices[0].message;
        // Even though DeepSeek would extract reasoning, include_reasoning=false
        // suppresses it.
        assert!(
            msg.reasoning.is_none(),
            "reasoning should be None when include_reasoning=false"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_chat_completion_without_reasoning_parser() {
        // No reasoning parser set — reasoning field should always be None.
        let engine = Arc::new(make_test_engine());
        engine.spawn_step_loop();

        let request = make_chat_request();
        let response = engine.chat_completion(request).await.unwrap();

        assert_eq!(response.choices.len(), 1);
        let msg = &response.choices[0].message;
        assert!(
            msg.reasoning.is_none(),
            "reasoning should be None without parser"
        );
        assert!(msg.content.is_some(), "content should be present");
    }

    /// Gemma 4 thought-channel flood (the `launch claude` bug): the stream
    /// degrades into `thought\n<channel|>` cycles with the `<|channel>` opens
    /// dropped. Phase 1 must consume the cycles wholesale — no `thought` line
    /// may stream as text — and the final delta must fold in the parser's
    /// flush so a reply ending on the held-back word is not truncated.
    #[test]
    fn test_streaming_gemma4_flood_consumed_and_flushed() {
        let vocab = HashMap::new();
        let parser = crate::reasoning_parser::get_reasoning_parser("gemma4", &vocab).unwrap();

        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut requests = HashMap::new();
        let mut state = make_test_request_state(Some(tx));
        state.reasoning_parser_state = Some(parser.create_streaming_state());
        state.include_reasoning = true;
        requests.insert("req-1".to_string(), state);

        let steps: &[(&str, Option<FinishReason>)] = &[
            ("```bash\ngit status\n```\n", None),
            ("thought", None),
            ("\n", None),
            ("<channel|>", None),
            ("thought\n", None),
            ("<channel|>", None),
            ("one ", None),
            ("thought", Some(FinishReason::Stop)),
        ];
        for (text, finish) in steps {
            let output = EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![10],
                finish_reason: *finish,
                stop_reason: None,
                num_cached_tokens: 0,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: None,
            };
            let detok = Some((None, Some(text.to_string())));
            AsyncEngine::process_outputs_phase3(&mut requests, vec![output], vec![detok]);
        }

        let mut content_parts = Vec::new();
        while let Ok(delta) = rx.try_recv() {
            if let Some(t) = delta.text {
                content_parts.push(t);
            }
        }
        let content = content_parts.join("");
        assert_eq!(
            content, "```bash\ngit status\n```\none thought",
            "flood cycles must vanish and the final held word must flush"
        );
        assert!(
            !content.contains("<channel|>"),
            "marker text must never stream as content"
        );
    }

    #[test]
    fn test_streaming_phase3_with_reasoning_parser() {
        // Directly test process_outputs_phase3 with a RequestState that has
        // a reasoning parser state, simulating streaming deltas.
        use crate::reasoning_parser::DeepSeekR1ReasoningParser;

        let vocab = reasoning_test_vocab();
        let parser = DeepSeekR1ReasoningParser::new(&vocab).unwrap();

        let (tx, mut rx) = mpsc::unbounded_channel();

        let mut requests = HashMap::new();
        let mut state = make_test_request_state(Some(tx));
        state.reasoning_parser_state = Some(parser.create_streaming_state());
        state.include_reasoning = true;
        requests.insert("req-1".to_string(), state);

        // Simulate: no tokenizer, so delta_text is None but token IDs are present.
        // The reasoning parser needs delta_text to work, so simulate with DetokResult.
        // Since process_outputs_phase3 takes DetokResult, we provide one.

        // Simulate delta: "<think>" (single token ID 100 → skipped by parser).
        let output1 = EngineCoreOutput {
            request_id: "req-1".to_string(),
            new_token_ids: vec![100],
            finish_reason: None,
            stop_reason: None,
            num_cached_tokens: 0,
            events: None,
            new_logprobs: None,
            new_prompt_logprobs: None,
            pooler_output: None,
        };
        // A `DetokResult` no longer carries the detokenizer — it never leaves the request state.
        let detok1 = Some((None, Some("<think>".to_string())));

        AsyncEngine::process_outputs_phase3(&mut requests, vec![output1], vec![detok1]);

        // Simulate delta: "reasoning text" (token ID 10).
        let output2 = EngineCoreOutput {
            request_id: "req-1".to_string(),
            new_token_ids: vec![10],
            finish_reason: None,
            stop_reason: None,
            num_cached_tokens: 0,
            events: None,
            new_logprobs: None,
            new_prompt_logprobs: None,
            pooler_output: None,
        };
        let detok2 = Some((None, Some("reasoning text".to_string())));

        AsyncEngine::process_outputs_phase3(&mut requests, vec![output2], vec![detok2]);

        // Simulate delta: "</think>content" (token IDs 101, 20).
        let output3 = EngineCoreOutput {
            request_id: "req-1".to_string(),
            new_token_ids: vec![101, 20],
            finish_reason: None,
            stop_reason: None,
            num_cached_tokens: 0,
            events: None,
            new_logprobs: None,
            new_prompt_logprobs: None,
            pooler_output: None,
        };
        let detok3 = Some((None, Some("</think>content".to_string())));

        AsyncEngine::process_outputs_phase3(&mut requests, vec![output3], vec![detok3]);

        // Simulate delta: " done" (token ID 21) — content after reasoning ended.
        let output4 = EngineCoreOutput {
            request_id: "req-1".to_string(),
            new_token_ids: vec![21],
            finish_reason: Some(FinishReason::Stop),
            stop_reason: None,
            num_cached_tokens: 0,
            events: None,
            new_logprobs: None,
            new_prompt_logprobs: None,
            pooler_output: None,
        };
        let detok4 = Some((None, Some(" done".to_string())));

        AsyncEngine::process_outputs_phase3(&mut requests, vec![output4], vec![detok4]);

        // Collect all deltas from the channel.
        let mut reasoning_parts = Vec::new();
        let mut content_parts = Vec::new();
        while let Ok(delta) = rx.try_recv() {
            if let Some(r) = delta.reasoning {
                reasoning_parts.push(r);
            }
            if let Some(t) = delta.text {
                content_parts.push(t);
            }
        }

        assert!(
            !reasoning_parts.is_empty(),
            "should have reasoning deltas: got {:?}",
            reasoning_parts
        );
        assert!(
            !content_parts.is_empty(),
            "should have content deltas: got {:?}",
            content_parts
        );
        assert_eq!(
            content_parts.join(""),
            "content done",
            "content should be everything after </think>"
        );
    }

    /// Qwen3.6 behind Claude Code: the template opens `<think>`, so the output starts in
    /// reasoning, closes it, then calls a tool. The tool parser reads only the text past the
    /// reasoning — no reasoning and no `</think>` stream as content — and the call goes out
    /// whole, its arguments included.
    #[test]
    fn test_streaming_reasoning_then_qwen3_coder_tool_call() {
        use crate::reasoning_parser::Qwen3ReasoningParser;
        let parser = Qwen3ReasoningParser::new(&reasoning_test_vocab()).unwrap();
        let tools = crate::tool_parser::get_tool_parser("qwen3_coder").unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut requests = HashMap::new();
        let mut state = make_test_request_state(Some(tx));
        state.reasoning_parser_state = Some(parser.create_streaming_state());
        state.tool_parser_state = Some(tools.create_streaming_state());
        state.include_reasoning = true;
        requests.insert("req-1".to_string(), state);
        let pieces: [(&[u32], &str); 5] = [
            (&[10], "I will run ls."),
            (&[101, 11], "\n</think>\n\n"),
            (&[12], "<tool_call>\n<function=Bash>\n"),
            (&[13], "<parameter=command>\nls\n</parameter>\n"),
            (&[14], "</function>\n</tool_call>"),
        ];
        for (i, (ids, text)) in pieces.iter().enumerate() {
            let output = EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: ids.to_vec(),
                finish_reason: (i + 1 == pieces.len()).then_some(FinishReason::Stop),
                stop_reason: None,
                num_cached_tokens: 0,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: None,
            };
            let detok = Some((None, Some(text.to_string())));
            AsyncEngine::process_outputs_phase3(&mut requests, vec![output], vec![detok]);
        }
        let (mut content, mut reasoning, mut calls) = (String::new(), String::new(), Vec::new());
        while let Ok(delta) = rx.try_recv() {
            content.push_str(delta.text.as_deref().unwrap_or(""));
            reasoning.push_str(delta.reasoning.as_deref().unwrap_or(""));
            calls.extend(delta.tool_call_deltas.unwrap_or_default());
        }
        assert_eq!(content.trim(), "", "content: {content:?}");
        assert!(
            reasoning.contains("I will run ls."),
            "reasoning: {reasoning:?}"
        );
        assert_eq!(calls.len(), 1, "one call: {calls:?}");
        assert_eq!(calls[0].function_name.as_deref(), Some("Bash"));
        let args: serde_json::Value =
            serde_json::from_str(calls[0].function_arguments.as_deref().unwrap()).unwrap();
        assert_eq!(args["command"], "ls");
    }

    /// Gemma 4 behind Claude Code: a tools-bearing request whose reply is plain
    /// text. The terminal `<end_of_turn>` token decodes into the stream, the
    /// gemma4 parser strips it to empty and returns `None` (buffering) for the
    /// final increment — so no content delta can carry the finish. The stream
    /// must still end with a finish-bearing delta: without one the SSE response
    /// ends with no `message_delta`/`message_stop` and the client reads a
    /// truncated message (Claude Code retries without streaming, then fails).
    #[test]
    fn test_streaming_gemma4_plain_reply_ends_with_finish_delta() {
        let parser = crate::tool_parser::get_tool_parser("gemma4").unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut requests = HashMap::new();
        let mut state = make_test_request_state(Some(tx));
        state.tool_parser_state = Some(parser.create_streaming_state());
        requests.insert("req-1".to_string(), state);

        // Plain-text reply pieces, then the terminal control token with the
        // finish stamped on the same output — the shape the detokenizer
        // produces for an EOS ending on a gemma 4 reply.
        let pieces: [(&[u32], &str, Option<FinishReason>); 3] = [
            (&[10], "Hello", None),
            (&[11], ", I'm Claude.", None),
            (&[106], "<end_of_turn>", Some(FinishReason::Stop)),
        ];
        for (ids, text, finish) in pieces {
            let output = EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: ids.to_vec(),
                finish_reason: finish,
                stop_reason: None,
                num_cached_tokens: 0,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: None,
            };
            let detok = Some((None, Some(text.to_string())));
            AsyncEngine::process_outputs_phase3(&mut requests, vec![output], vec![detok]);
        }

        let mut content = String::new();
        let mut finishes = Vec::new();
        while let Ok(delta) = rx.try_recv() {
            content.push_str(delta.text.as_deref().unwrap_or(""));
            if let Some(f) = delta.finish_reason {
                finishes.push(f);
            }
        }
        assert_eq!(content, "Hello, I'm Claude.");
        assert_eq!(
            finishes,
            vec![FinishReason::Stop],
            "the stream must end with exactly one finish-bearing delta"
        );
    }

    #[test]
    fn test_streaming_phase3_include_reasoning_false() {
        // Verify include_reasoning=false suppresses reasoning in streaming deltas.
        use crate::reasoning_parser::DeepSeekR1ReasoningParser;

        let vocab = reasoning_test_vocab();
        let parser = DeepSeekR1ReasoningParser::new(&vocab).unwrap();

        let (tx, mut rx) = mpsc::unbounded_channel();

        let mut requests = HashMap::new();
        let mut state = make_test_request_state(Some(tx));
        state.reasoning_parser_state = Some(parser.create_streaming_state());
        state.include_reasoning = false; // <-- suppress reasoning
        requests.insert("req-1".to_string(), state);

        // Send <think> token.
        let output = EngineCoreOutput {
            request_id: "req-1".to_string(),
            new_token_ids: vec![100],
            finish_reason: None,
            stop_reason: None,
            num_cached_tokens: 0,
            events: None,
            new_logprobs: None,
            new_prompt_logprobs: None,
            pooler_output: None,
        };
        let detok = Some((None, Some("<think>".to_string())));
        AsyncEngine::process_outputs_phase3(&mut requests, vec![output], vec![detok]);

        // Send reasoning text.
        let output = EngineCoreOutput {
            request_id: "req-1".to_string(),
            new_token_ids: vec![10],
            finish_reason: None,
            stop_reason: None,
            num_cached_tokens: 0,
            events: None,
            new_logprobs: None,
            new_prompt_logprobs: None,
            pooler_output: None,
        };
        let detok = Some((None, Some("thinking".to_string())));
        AsyncEngine::process_outputs_phase3(&mut requests, vec![output], vec![detok]);

        // Collect deltas.
        let mut has_reasoning = false;
        while let Ok(delta) = rx.try_recv() {
            if delta.reasoning.is_some() {
                has_reasoning = true;
            }
        }

        assert!(
            !has_reasoning,
            "reasoning should be suppressed when include_reasoning=false"
        );
    }

    #[test]
    fn test_streaming_phase3_no_reasoning_parser() {
        // Without a reasoning parser, streaming deltas should have reasoning=None.
        let (tx, mut rx) = mpsc::unbounded_channel();

        let mut requests = HashMap::new();
        let state = make_test_request_state(Some(tx));
        // reasoning_parser_state is None by default.
        requests.insert("req-1".to_string(), state);

        let output = EngineCoreOutput {
            request_id: "req-1".to_string(),
            new_token_ids: vec![42],
            finish_reason: Some(FinishReason::Stop),
            stop_reason: None,
            num_cached_tokens: 0,
            events: None,
            new_logprobs: None,
            new_prompt_logprobs: None,
            pooler_output: None,
        };
        let detok = Some((None, Some("hello".to_string())));

        AsyncEngine::process_outputs_phase3(&mut requests, vec![output], vec![detok]);

        let delta = rx.try_recv().unwrap();
        assert!(
            delta.reasoning.is_none(),
            "reasoning should be None without parser"
        );
        assert_eq!(delta.text, Some("hello".to_string()));
    }

    // -----------------------------------------------------------------------
    // merge_chat_template_kwargs tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_kwargs_both_none() {
        assert!(merge_chat_template_kwargs(None, None).is_none());
    }

    #[test]
    fn test_merge_kwargs_defaults_only() {
        let mut defaults = std::collections::HashMap::new();
        defaults.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        );
        let merged = merge_chat_template_kwargs(Some(&defaults), None).unwrap();
        assert_eq!(
            merged.get("enable_thinking"),
            Some(&serde_json::Value::Bool(false))
        );
    }

    #[test]
    fn test_merge_kwargs_request_only() {
        let mut request = std::collections::HashMap::new();
        request.insert("enable_thinking".to_string(), serde_json::Value::Bool(true));
        let merged = merge_chat_template_kwargs(None, Some(&request)).unwrap();
        assert_eq!(
            merged.get("enable_thinking"),
            Some(&serde_json::Value::Bool(true))
        );
    }

    #[test]
    fn test_merge_kwargs_request_overrides_defaults() {
        let mut defaults = std::collections::HashMap::new();
        defaults.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        );
        defaults.insert("reasoning_effort".to_string(), serde_json::json!("medium"));

        let mut request = std::collections::HashMap::new();
        request.insert("enable_thinking".to_string(), serde_json::Value::Bool(true));

        let merged = merge_chat_template_kwargs(Some(&defaults), Some(&request)).unwrap();
        // Request overrides default.
        assert_eq!(
            merged.get("enable_thinking"),
            Some(&serde_json::Value::Bool(true))
        );
        // Default preserved when not overridden.
        assert_eq!(
            merged.get("reasoning_effort"),
            Some(&serde_json::json!("medium"))
        );
    }

    // -----------------------------------------------------------------------
    // chat_template_kwargs in request serde tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_chat_template_kwargs_deserialization() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "low"}
        }"#;
        let req: protocol::ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let kwargs = req.chat_template_kwargs.unwrap();
        assert_eq!(
            kwargs.get("enable_thinking"),
            Some(&serde_json::Value::Bool(false))
        );
        assert_eq!(
            kwargs.get("reasoning_effort"),
            Some(&serde_json::json!("low"))
        );
    }

    #[test]
    fn test_chat_template_kwargs_absent() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}]
        }"#;
        let req: protocol::ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.chat_template_kwargs.is_none());
    }

    #[test]
    fn test_chat_template_kwargs_empty_object() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "chat_template_kwargs": {}
        }"#;
        let req: protocol::ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.chat_template_kwargs.unwrap().is_empty());
    }

    // -----------------------------------------------------------------------
    // default_chat_template_kwargs on engine
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_chat_template_kwargs_used_in_template() {
        // Use a template that checks enable_thinking. We test at the template
        // level to avoid test-tokenizer limitations with special tokens.
        let template_str = "{% set enable_thinking = enable_thinking | default(true) %}{% for message in messages %}{{ message.content }}{% endfor %}{% if enable_thinking %}THINKING{% else %}NO_THINKING{% endif %}";
        let template = crate::chat_template::ChatTemplate::new(template_str.to_string()).unwrap();

        let messages = vec![serde_json::json!({"role": "user", "content": "Hi"})];

        // Without kwargs: enable_thinking defaults to true.
        let result = template.apply(&messages, true, None).unwrap();
        assert!(
            result.contains("THINKING"),
            "should contain THINKING with default, got: {result}"
        );
        assert!(!result.contains("NO_THINKING"));

        // With enable_thinking=false.
        let mut kwargs = std::collections::HashMap::new();
        kwargs.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        );
        let result = template
            .apply_with_kwargs(&messages, true, None, Some(&kwargs))
            .unwrap();
        assert!(
            result.contains("NO_THINKING"),
            "should contain NO_THINKING with enable_thinking=false, got: {result}"
        );
        assert!(!result.contains("\nTHINKING"));
    }

    #[test]
    fn test_default_and_request_kwargs_merge_in_engine() {
        // Test that merge_chat_template_kwargs correctly flows through
        // to chat_to_engine_request by testing merge logic directly.
        let mut defaults = std::collections::HashMap::new();
        defaults.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        );
        defaults.insert("reasoning_effort".to_string(), serde_json::json!("medium"));

        let mut request_kw = std::collections::HashMap::new();
        request_kw.insert("enable_thinking".to_string(), serde_json::Value::Bool(true));

        let merged = merge_chat_template_kwargs(Some(&defaults), Some(&request_kw)).unwrap();

        // Request overrides default.
        assert_eq!(merged["enable_thinking"], serde_json::Value::Bool(true));
        // Default preserved.
        assert_eq!(merged["reasoning_effort"], serde_json::json!("medium"));

        // Verify template renders correctly with merged kwargs.
        let template_str = "{% set enable_thinking = enable_thinking | default(true) %}{% if enable_thinking %}THINKING{% else %}NO_THINKING{% endif %}";
        let template = crate::chat_template::ChatTemplate::new(template_str.to_string()).unwrap();
        let messages = vec![serde_json::json!({"role": "user", "content": "Hi"})];
        let result = template
            .apply_with_kwargs(&messages, true, None, Some(&merged))
            .unwrap();
        assert!(
            result.contains("THINKING"),
            "merged kwargs should have enable_thinking=true"
        );
    }

    /// The tool-parsed path and the reasoning-parsed one each drop the deltas of the steps the
    /// tool parser buffers. The count still reaches the end of the stream: the last delta reports
    /// every generated token, as the `request finished` log does.
    #[test]
    fn test_streaming_buffered_tool_call_reports_every_token() {
        for reasoning in [false, true] {
            let (deltas, generated) =
                super::test_support::buffered_tool_call_deltas(reasoning, FinishReason::Stop);
            let summed: u32 = deltas.iter().map(|d| d.new_token_ids.len() as u32).sum();
            assert!(
                summed < generated,
                "reasoning={reasoning}: the parser must buffer some steps for this to test anything \
                 ({summed} of {generated} tokens on deltas)"
            );
            let last = deltas.last().expect("a delta");
            assert!(last.finish_reason.is_some(), "reasoning={reasoning}");
            assert_eq!(
                last.usage.completion_tokens, generated,
                "reasoning={reasoning}"
            );
        }
    }

    /// A tool call's finish carries the step's own reason: one cut off by `max_tokens` streams
    /// `Length`, which the encoders keep (`length`, `max_tokens`) rather than turning into a
    /// tool-call finish. It used to stream `Stop` whatever ended the request.
    #[test]
    fn test_streaming_tool_call_finish_keeps_the_steps_reason() {
        for reasoning in [false, true] {
            for finish in [FinishReason::Stop, FinishReason::Length] {
                let (deltas, _) = super::test_support::buffered_tool_call_deltas(reasoning, finish);
                assert!(deltas.iter().any(|d| d.tool_call_deltas.is_some()));
                let finishes: Vec<_> = deltas.iter().filter_map(|d| d.finish_reason).collect();
                assert_eq!(finishes, [finish], "reasoning={reasoning}");
            }
        }
    }

    /// The content a gpt-oss reply streams for `pieces` (one engine output each, the last ending
    /// it on `max_tokens`), through phases 1 and 3 as the step loop runs them.
    fn harmony_streamed_content(pieces: &[&str]) -> String {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut state = make_test_request_state(Some(tx));
        state.reasoning_parser_state =
            Some(crate::reasoning_parser::GptOssReasoningParser.create_streaming_state());
        let mut requests = HashMap::from([("req-1".to_string(), state)]);
        let last = pieces.len() - 1;
        for (i, text) in pieces.iter().enumerate() {
            let outputs = vec![EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![20 + i as u32],
                finish_reason: (i == last).then_some(FinishReason::Length),
                stop_reason: None,
                num_cached_tokens: 0,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: None,
            }];
            AsyncEngine::process_outputs_phase1(&mut requests, &outputs);
            let detok = vec![Some((None, Some(text.to_string())))];
            AsyncEngine::process_outputs_phase3(&mut requests, outputs, detok);
        }
        let mut content = String::new();
        while let Ok(delta) = rx.try_recv() {
            content.extend(delta.text);
        }
        content
    }

    /// A gpt-oss reply streams the same content the complete reply splits out, through the
    /// engine's own routing: every delta goes through the harmony scanner, content included, and
    /// the scanner's held text is flushed at the end. Reasoning-first, the engine stopped feeding
    /// it at the first content, so a `<` held there was lost and later markers streamed as text.
    #[test]
    fn a_harmony_reply_streams_the_content_the_complete_reply_splits_out() {
        let header = [
            "<|channel|>",
            "analysis",
            "<|message|>",
            "think",
            "<|end|>",
            "<|start|>",
            "assistant",
            "<|channel|>",
            "final",
            "<|message|>",
        ];
        let finals: [&[&str]; 3] = [
            // The control: content whose `<` closes in the same delta.
            &["Use", " \"<", "div", ">\""],
            // Content whose first delta ends on an unclosed `<`.
            &["\"<", "div", ">\""],
            // A second final message after content.
            &[
                "Hi",
                "<|end|>",
                "<|start|>",
                "assistant",
                "<|channel|>",
                "final",
                "<|message|>",
                " again",
            ],
        ];
        for content in finals {
            let pieces: Vec<&str> = header.iter().chain(content).copied().collect();
            let whole =
                crate::reasoning_parser::GptOssReasoningParser.extract_reasoning(&pieces.concat());
            assert_eq!(
                Some(harmony_streamed_content(&pieces)),
                whole.content,
                "{content:?}"
            );
        }
    }
}

/// Engine-produced delta streams for the HTTP encoders' tests.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// The deltas the engine streams for a qwen3_coder tool call the parser buffers for most of
    /// its steps, with the number of tokens generated. `reasoning` puts a reasoning parser in
    /// front, with a short thought first; the last step ends the request with `finish`.
    ///
    /// The buffered steps send no delta, so the deltas' `new_token_ids` add up to fewer tokens
    /// than were generated: a consumer has to read the count from [`StreamUsage`].
    pub(crate) fn buffered_tool_call_deltas(
        reasoning: bool,
        finish: FinishReason,
    ) -> (Vec<StreamDelta>, u32) {
        let mut pieces: Vec<(Vec<u32>, &str)> = Vec::new();
        if reasoning {
            pieces.push((vec![10], "I will run ls."));
            pieces.push((vec![101, 11], "\n</think>\n\n"));
        }
        pieces.extend([
            (vec![12], "<tool_call>\n"),
            (vec![13], "<function=Bash>\n"),
            (vec![14], "<parameter=command>\n"),
            (vec![15], "ls -la"),
            (vec![16], "\n</parameter>\n"),
            (vec![17], "</function>\n"),
            (vec![18], "</tool_call>"),
        ]);
        let generated = pieces.iter().map(|(ids, _)| ids.len() as u32).sum();
        let configure = |state: &mut RequestState| {
            state.tool_parser_state = Some(
                crate::tool_parser::get_tool_parser("qwen3_coder")
                    .unwrap()
                    .create_streaming_state(),
            );
            if reasoning {
                let parser = crate::reasoning_parser::Qwen3ReasoningParser::new(
                    &super::tests::reasoning_test_vocab(),
                )
                .unwrap();
                state.reasoning_parser_state = Some(parser.create_streaming_state());
            }
        };
        (stream_pieces(configure, pieces, finish), generated)
    }

    /// The deltas a request streams for `pieces` — each piece one engine output, its token ids
    /// and the text they detokenize to, the last one finishing the request with `finish` — run
    /// through phases 1 and 3 as the step loop runs them. `configure` sets the request's parsers.
    pub(super) fn stream_pieces(
        configure: impl FnOnce(&mut RequestState),
        pieces: Vec<(Vec<u32>, &str)>,
        finish: FinishReason,
    ) -> Vec<StreamDelta> {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut state = super::tests::make_test_request_state(Some(tx));
        configure(&mut state);
        let mut requests = HashMap::from([("req-1".to_string(), state)]);
        let last = pieces.len() - 1;
        for (i, (ids, text)) in pieces.into_iter().enumerate() {
            let outputs = vec![EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: ids,
                finish_reason: (i == last).then_some(finish),
                stop_reason: None,
                num_cached_tokens: 0,
                events: None,
                new_logprobs: None,
                new_prompt_logprobs: None,
                pooler_output: None,
            }];
            AsyncEngine::process_outputs_phase1(&mut requests, &outputs);
            let detok = vec![Some((None, Some(text.to_string())))];
            AsyncEngine::process_outputs_phase3(&mut requests, outputs, detok);
        }
        let mut deltas = Vec::new();
        while let Ok(delta) = rx.try_recv() {
            deltas.push(delta);
        }
        deltas
    }
}
