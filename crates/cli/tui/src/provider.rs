//! In-process goose provider backed by scratchy's serving engine.
//!
//! `ScratchyProvider` implements the goose `Provider` trait by driving
//! scratchy's `AsyncEngine` directly — the same engine the HTTP `/v1/messages`
//! handler wraps, but with no axum/server and no child process.
//!
//! Tool rendering (schemas into the prompt) and tool-call parsing (structured
//! `tool_use` out of raw generation) happen entirely inside scratchy's engine.
//! This provider only translates data shapes: goose `Message`/`Tool` in,
//! scratchy `ChatCompletionRequest` out; scratchy `ChatCompletionResponse` in,
//! goose `Message` (text + `ToolRequest`) out. The client never parses tools.

use anyhow::Result;
use async_trait::async_trait;
use goose_provider_types::base::{
    MessageStream, Provider, ProviderDescriptor, ProviderMetadata, stream_from_single_message,
};
use goose_provider_types::conversation::message::{Message, MessageContent};
use goose_provider_types::conversation::token_usage::{ProviderUsage, Usage};
use goose_provider_types::errors::ProviderError;
use goose_provider_types::model::ModelConfig;
use rmcp::model::{CallToolRequestParams, Role, Tool, object};
use scratchy_serving_api::engine::{AsyncEngine, LiveStats};
use scratchy_serving_api::init::{DownloadObserver, VllmConfig, initialize_stack};
use scratchy_serving_api::protocol;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

pub const PROVIDER_NAME: &str = "scratchy";
pub const DEFAULT_MODEL: &str = "mlx-community/Qwen2.5-3B-Instruct-4bit";
// `auto` lets the compiled-in backend factory claim it (metal / cuda / spyre all
// match `auto`), so a spyre-only build serves the TUI without erroring on a
// hard-coded `metal`. `GOOSE_SCRATCHY_DEVICE` overrides for explicit selection.
const DEFAULT_DEVICE: &str = "auto";
/// Fallback generation cap when the caller sets no `max_tokens`. Reasoning
/// models spend far more than 1k tokens on a single turn (thinking + tool
/// calls + answer), so a low cap truncates them mid-turn; keep it generous.
const DEFAULT_MAX_TOKENS: i32 = 8192;

/// One shard's download progress within a [`DownloadSnapshot`].
#[derive(Debug, Clone)]
pub struct ShardStat {
    /// Shard filename (e.g. `model-00001-of-00005.safetensors`).
    pub name: String,
    /// Bytes downloaded so far for this shard.
    pub done_bytes: u64,
    /// This shard's total size in bytes (0 until `init` reports it).
    pub total_bytes: u64,
}

/// A read-only snapshot of an in-flight model download, sampled by the TUI
/// each tick to render its own progress modal.
#[derive(Debug, Clone)]
pub struct DownloadSnapshot {
    /// Per-shard progress, sorted by name — the shards downloading in parallel
    /// (up to 8 concurrent), each rendered as its own bar.
    pub shards: Vec<ShardStat>,
    /// Bytes downloaded so far across all shards.
    pub done_bytes: u64,
    /// Total bytes to download across all shards seen so far.
    pub total_bytes: u64,
}

impl ShardStat {
    /// This shard has fully downloaded.
    pub fn is_done(&self) -> bool {
        self.total_bytes > 0 && self.done_bytes >= self.total_bytes
    }
}

#[derive(Debug, Default)]
struct DownloadInner {
    /// Whether an engine build is in progress. Gates the whole modal: set
    /// around `initialize_stack` so a cached model (no shards started) never
    /// shows it, and so it doesn't flicker off between download chunks.
    active: bool,
    /// shard filename → (bytes done, total bytes).
    shards: HashMap<String, (u64, u64)>,
    /// Last build/download failure (full error chain), shown as an error modal
    /// until the next build attempt or the user dismisses it. Kept separate
    /// from the progress state so a failed background prewarm still reports.
    error: Option<String>,
}

/// [`DownloadObserver`] sink shared between the blocking HF-download threads
/// (writers) and the TUI render loop (reader). Uses a `std::sync::Mutex`
/// because the writers are the synchronous `spawn_blocking` download threads;
/// the reader takes it with `try_lock`, mirroring `live_stats()`.
#[derive(Debug, Default)]
struct DownloadTracker {
    inner: StdMutex<DownloadInner>,
}

impl DownloadTracker {
    /// Open the modal window and clear any prior run's shards and error. Called
    /// right before `initialize_stack`.
    fn begin(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.active = true;
            inner.shards.clear();
            inner.error = None;
        }
    }

    /// Close the progress modal on success. Called after `initialize_stack`
    /// returns Ok, so the modal never outlives the build.
    fn end(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.active = false;
            inner.shards.clear();
        }
    }

    /// Record a build/download failure: drop the progress state and remember
    /// the error so the TUI can show it.
    fn fail(&self, msg: String) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.active = false;
            inner.shards.clear();
            inner.error = Some(msg);
        }
    }

    /// The last build/download failure, if any (cleared by `begin` or dismiss).
    fn error(&self) -> Option<String> {
        self.inner.try_lock().ok()?.error.clone()
    }

    /// Dismiss a shown error (user acknowledged it).
    fn clear_error(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.error = None;
        }
    }

    /// Aggregate snapshot while the build is active AND at least one shard is
    /// still downloading. Cached models never start a shard, so they show no
    /// modal; and once every shard is complete the snapshot goes back to `None`
    /// so the modal clears the moment the *download* finishes — rather than
    /// lingering through the (bar-less) weight-load/engine-build phase that
    /// runs before `end()`. Completion is derived from bytes (a shard is done
    /// once `done >= total`), which is robust to download retries re-calling
    /// `init`.
    fn snapshot(&self) -> Option<DownloadSnapshot> {
        let inner = self.inner.try_lock().ok()?;
        let any_in_flight = inner
            .shards
            .values()
            .any(|(done, total)| !(*total > 0 && *done >= *total));
        if !inner.active || !any_in_flight {
            return None;
        }
        let mut done_bytes = 0u64;
        let mut total_bytes = 0u64;
        let mut shards: Vec<ShardStat> = Vec::with_capacity(inner.shards.len());
        for (name, (done, total)) in inner.shards.iter() {
            done_bytes += *done;
            total_bytes += *total;
            shards.push(ShardStat {
                name: name.clone(),
                done_bytes: *done,
                total_bytes: *total,
            });
        }
        // Stable order so the parallel bars don't jump around between ticks.
        shards.sort_by(|a, b| a.name.cmp(&b.name));
        Some(DownloadSnapshot {
            shards,
            done_bytes,
            total_bytes,
        })
    }
}

impl DownloadObserver for DownloadTracker {
    fn on_start(&self, shard: &str, total_bytes: usize) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.shards.entry(shard.to_string()).or_insert((0, 0)).1 = total_bytes as u64;
        }
    }

    fn on_advance(&self, shard: &str, delta_bytes: usize) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.shards.entry(shard.to_string()).or_insert((0, 0)).0 += delta_bytes as u64;
        }
    }

    fn on_finish(&self, shard: &str) {
        if let Ok(mut inner) = self.inner.lock() {
            // Pin done == total so a rounding gap can't leave a finished shard
            // shy of 100%.
            if let Some(entry) = inner.shards.get_mut(shard) {
                entry.0 = entry.1;
            }
        }
    }
}

/// A running engine bound to one model spec, plus the step-loop task that keeps
/// it processing requests.
struct EngineHandle {
    spec: String,
    engine: Arc<AsyncEngine>,
    step: JoinHandle<()>,
}

pub struct ScratchyProvider {
    slot: Arc<Mutex<Option<EngineHandle>>>,
    name: String,
    /// The engine request id currently generating, so the UI can abort it
    /// (goose's cancel token never reaches the engine). Held only briefly,
    /// never across an await.
    current_request: Arc<Mutex<Option<String>>>,
    /// Model-download progress sink, handed to the engine build as the
    /// `download_observer` so HF shard progress renders in the TUI (a modal)
    /// instead of `indicatif` bars scribbling over the alt-screen.
    download: Arc<DownloadTracker>,
}

/// A process-unique engine request id we assign (so we know it and can abort it;
/// left unset, the engine would generate its own that we couldn't reference).
fn next_request_id() -> String {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    format!(
        "tui-{}",
        N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}

impl ScratchyProvider {
    pub async fn from_env() -> Result<Self> {
        Ok(Self {
            slot: Arc::new(Mutex::new(None)),
            name: PROVIDER_NAME.to_string(),
            current_request: Arc::new(Mutex::new(None)),
            download: Arc::new(DownloadTracker::default()),
        })
    }

    /// Snapshot of the in-flight model download for the TUI's progress modal.
    /// Non-blocking; `None` when no download is active (cached model, or the
    /// tracker is momentarily locked by a download thread).
    pub fn download_progress(&self) -> Option<DownloadSnapshot> {
        self.download.snapshot()
    }

    /// The last model-build/download failure (full error chain), for the TUI's
    /// error modal. `None` once dismissed or when a new build starts.
    pub fn download_error(&self) -> Option<String> {
        self.download.error()
    }

    /// Dismiss the shown build/download error (user acknowledged it).
    pub fn clear_download_error(&self) {
        self.download.clear_error();
    }

    /// Begin building the engine for `spec` (downloading weights if needed) in
    /// the background, so the work starts at TUI startup rather than on the
    /// first prompt. Fire-and-forget and idempotent with `stream`'s own
    /// `ensure_engine`: the shared `slot` lock serializes them, so whichever
    /// runs first builds and the other reuses the result. Any build error is
    /// swallowed here — the first prompt re-runs `ensure_engine` and surfaces
    /// it to the user then.
    pub fn prewarm(&self, spec: String) {
        let slot = self.slot.clone();
        let download = self.download.clone();
        tokio::spawn(async move {
            let _ = ensure_engine(&slot, &download, &spec).await;
        });
    }

    /// Abort the in-flight generation (Esc). Tells the engine to drop the
    /// request, which frees its scheduler slot and ends the awaited
    /// `chat_completion`. No-op when idle.
    pub async fn abort(&self) {
        let Some(rid) = self.current_request.lock().await.clone() else {
            return;
        };
        let engine = {
            let Ok(g) = self.slot.try_lock() else { return };
            let Some(h) = g.as_ref() else { return };
            h.engine.clone()
        };
        engine.abort_request(&rid).await;
    }

    /// Live per-request engine stats for the ⚙ panel. Non-blocking: returns
    /// `None` if the engine isn't built yet or the request map is momentarily
    /// locked by the step loop.
    pub fn live_stats(&self) -> Option<LiveStats> {
        let g = self.slot.try_lock().ok()?;
        g.as_ref()?.engine.live_stats()
    }

    /// The loaded model's max context length (once the engine is built).
    pub fn max_context(&self) -> Option<usize> {
        let g = self.slot.try_lock().ok()?;
        Some(g.as_ref()?.engine.max_model_len())
    }

    /// Decode a single token id to its display string, for the sampler-soul
    /// candidate list. Non-blocking; `None` if the engine map is momentarily
    /// locked or not yet built.
    pub fn decode_token(&self, id: u32) -> Option<String> {
        let g = self.slot.try_lock().ok()?;
        Some(g.as_ref()?.engine.decode_token(id))
    }

    /// Recent inter-token latencies (ms) for the ITL distribution panel.
    pub fn recent_itls(&self) -> Vec<f32> {
        let Ok(g) = self.slot.try_lock() else {
            return Vec::new();
        };
        g.as_ref()
            .map(|h| h.engine.recent_itls())
            .unwrap_or_default()
    }
}

fn exec_err(e: impl std::fmt::Display) -> ProviderError {
    ProviderError::ExecutionError(e.to_string())
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Load (or reuse) the engine for `spec`, building it the same way the serve
/// path does: `initialize_stack` → `spawn_step_loop`. The engine self-selects
/// its tool-call parser from the model in `initialize_stack`, so this provider
/// does no parser configuration.
async fn ensure_engine(
    slot: &Arc<Mutex<Option<EngineHandle>>>,
    download: &Arc<DownloadTracker>,
    spec: &str,
) -> Result<Arc<AsyncEngine>, ProviderError> {
    let mut guard = slot.lock().await;

    if guard.as_ref().map(|h| h.spec != spec).unwrap_or(true) {
        if let Some(old) = guard.take() {
            old.step.abort();
        }

        let config = VllmConfig {
            model: spec.to_string(),
            device: env_or("GOOSE_SCRATCHY_DEVICE", DEFAULT_DEVICE),
            dtype: "auto".to_string(),
            // The TUI is strictly one conversation at a time — an explicit width, not a default.
            max_num_seqs: Some(1),
            // Route HF shard-download progress into the TUI's own modal rather
            // than the default indicatif bars — those draw to stderr and sit
            // over the ratatui alt-screen, corrupting it.
            download_observer: Some(download.clone() as Arc<dyn DownloadObserver>),
            ..Default::default()
        };

        // The StartupProgress bar is likewise disabled (it too draws to stderr
        // over the alt-screen); the TUI renders build state itself.
        download.begin();
        let joined = tokio::task::spawn_blocking(move || {
            let progress =
                std::sync::Arc::new(scratchy_serving_api::progress::StartupProgress::new(false));
            initialize_stack(&config, Some(progress))
        })
        .await;
        // Record any build failure in the tracker (with the full error chain,
        // e.g. the "insufficient disk space" cause) so the TUI can surface it —
        // otherwise a failed build (especially the background prewarm, whose
        // Result is dropped) would just make the modal vanish with no reason.
        let stack = match joined {
            Ok(Ok(stack)) => {
                download.end();
                stack
            }
            Ok(Err(build_err)) => {
                let msg = format!("{build_err:#}");
                download.fail(msg.clone());
                return Err(exec_err(msg));
            }
            Err(join_err) => {
                let msg = join_err.to_string();
                download.fail(msg.clone());
                return Err(exec_err(msg));
            }
        };

        let engine = stack.engine;
        let step = engine.spawn_step_loop();
        *guard = Some(EngineHandle {
            spec: spec.to_string(),
            engine,
            step,
        });
    }

    guard
        .as_ref()
        .map(|h| h.engine.clone())
        .ok_or_else(|| exec_err("engine unavailable after initialization"))
}

fn concat_text(msg: &Message) -> String {
    msg.content
        .iter()
        .filter_map(|c| match c {
            MessageContent::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The assistant's tool calls in OpenAI `tool_calls` shape (arguments are a
/// JSON-encoded string, as the engine both emits and expects).
fn tool_calls_json(msg: &Message) -> Vec<Value> {
    msg.content
        .iter()
        .filter_map(|c| match c {
            MessageContent::ToolRequest(req) => {
                let call = req.tool_call.as_ref().ok()?;
                let args = call
                    .arguments
                    .as_ref()
                    .map(|a| Value::Object(a.clone()))
                    .unwrap_or_else(|| json!({}));
                Some(json!({
                    "id": req.id,
                    "type": "function",
                    "function": { "name": call.name, "arguments": args.to_string() },
                }))
            }
            _ => None,
        })
        .collect()
}

/// Tool results as OpenAI `role: "tool"` messages, keyed by the call id so the
/// model can tie each result back to the call it made.
fn tool_results_json(msg: &Message) -> Vec<Value> {
    msg.content
        .iter()
        .filter_map(|c| match c {
            MessageContent::ToolResponse(resp) => {
                let content = match &resp.tool_result {
                    Ok(r) => r
                        .content
                        .iter()
                        .filter_map(|c| c.as_text().map(|t| t.text.clone()))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    Err(e) => format!("Error: {e}"),
                };
                Some(json!({ "role": "tool", "tool_call_id": resp.id, "content": content }))
            }
            _ => None,
        })
        .collect()
}

/// Build an OpenAI-shaped chat request (the engine's native input) from goose
/// types. Constructed as JSON and deserialized so we don't depend on the full
/// `ChatCompletionRequest` field list.
fn build_request(
    spec: &str,
    system: &str,
    messages: &[Message],
    tools: &[Tool],
    model_config: &ModelConfig,
) -> Result<protocol::ChatCompletionRequest, ProviderError> {
    let mut msgs: Vec<Value> = Vec::new();
    if !system.trim().is_empty() {
        msgs.push(json!({ "role": "system", "content": system }));
    }
    for m in messages {
        match m.role {
            Role::Assistant => {
                let text = concat_text(m);
                let tool_calls = tool_calls_json(m);
                if text.trim().is_empty() && tool_calls.is_empty() {
                    continue;
                }
                let mut msg = json!({ "role": "assistant" });
                msg["content"] = if text.trim().is_empty() {
                    Value::Null
                } else {
                    json!(text)
                };
                if !tool_calls.is_empty() {
                    msg["tool_calls"] = json!(tool_calls);
                }
                msgs.push(msg);
            }
            Role::User => {
                // Tool results must precede any user text in the same turn so
                // they follow the assistant tool_calls they answer.
                msgs.extend(tool_results_json(m));
                let text = concat_text(m);
                if !text.trim().is_empty() {
                    msgs.push(json!({ "role": "user", "content": text }));
                }
            }
        }
    }

    let tools_json: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                }
            })
        })
        .collect();

    let mut req = json!({
        "model": spec,
        "messages": msgs,
        "stream": false,
        "n": 1,
    });
    if !tools_json.is_empty() {
        req["tools"] = json!(tools_json);
    }
    let max_tokens = model_config
        .max_tokens
        .filter(|t| *t > 0)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    req["max_tokens"] = json!(max_tokens);
    if let Some(t) = model_config.temperature {
        req["temperature"] = json!(t);
    }

    serde_json::from_value(req).map_err(exec_err)
}

impl ProviderDescriptor for ScratchyProvider {
    fn metadata() -> ProviderMetadata
    where
        Self: Sized,
    {
        ProviderMetadata::new(
            PROVIDER_NAME,
            "Scratchy (local)",
            "In-process local inference via the scratchy engine (no server); tools handled engine-side",
            DEFAULT_MODEL,
            vec![DEFAULT_MODEL, "mlx-community/Qwen2.5-0.5B-Instruct-4bit"],
            "https://github.com/",
            vec![],
        )
    }
}

#[async_trait]
impl Provider for ScratchyProvider {
    fn get_name(&self) -> &str {
        &self.name
    }

    fn skip_canonical_filtering(&self) -> bool {
        true
    }

    async fn stream(
        &self,
        model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let spec = model_config.model_name.clone();
        let engine = ensure_engine(&self.slot, &self.download, &spec).await?;

        // Non-streaming: take the whole response and hand it over as one
        // message. (TUI streaming was tried and dropped — the per-token TUI
        // re-render cost ~30% tok/s, and the engine's streaming tool/reasoning
        // parsers were weaker than the whole-response path.)
        let mut request = build_request(&spec, system, messages, tools, model_config)?;
        let rid = next_request_id();
        request.request_id = Some(rid.clone());
        *self.current_request.lock().await = Some(rid);
        let resp = engine.chat_completion(request).await;
        *self.current_request.lock().await = None;
        let resp = resp.map_err(exec_err)?;
        let usage = ProviderUsage::new(
            spec.clone(),
            Usage::new(
                Some(resp.usage.prompt_tokens as i32),
                resp.usage.completion_tokens.map(|c| c as i32),
                None,
            ),
        );

        let mut message = Message::assistant();
        if let Some(choice) = resp.choices.into_iter().next() {
            let msg = choice.message;
            if let Some(reasoning) = msg.reasoning.filter(|r| !r.trim().is_empty()) {
                message = message.with_thinking(reasoning, String::new());
            }
            if let Some(text) = msg.content.filter(|t| !t.is_empty()) {
                message = message.with_text(text);
            }
            for tc in msg.tool_calls.unwrap_or_default() {
                let parsed = serde_json::from_str::<Value>(&tc.function.arguments)
                    .ok()
                    .filter(|v| v.is_object());
                let call = match parsed {
                    Some(v) => {
                        CallToolRequestParams::new(tc.function.name).with_arguments(object(v))
                    }
                    None => CallToolRequestParams::new(tc.function.name),
                };
                message = message.with_tool_request(tc.id, Ok(call));
            }
        }
        Ok(stream_from_single_message(message, usage))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_modal_before_begin() {
        let t = DownloadTracker::default();
        assert!(t.snapshot().is_none());
        // A shard reporting before `begin` (shouldn't happen) still shows
        // nothing — the active gate is closed.
        t.on_start("a", 100);
        assert!(t.snapshot().is_none());
    }

    #[test]
    fn cached_model_shows_no_modal() {
        // begin/end with no shard downloads (the network-free cache fast path).
        let t = DownloadTracker::default();
        t.begin();
        assert!(t.snapshot().is_none(), "no shards started → no modal");
        t.end();
        assert!(t.snapshot().is_none());
    }

    #[test]
    fn aggregates_across_shards() {
        let t = DownloadTracker::default();
        t.begin();
        t.on_start("a", 100);
        t.on_start("b", 300);
        t.on_advance("a", 40);
        t.on_advance("b", 60);
        let s = t.snapshot().expect("active with shards");
        assert_eq!(s.done_bytes, 100);
        assert_eq!(s.total_bytes, 400);
        assert_eq!(s.shards.len(), 2);
        assert_eq!(s.shards.iter().filter(|x| x.is_done()).count(), 0);
        // Sorted by name for stable rendering.
        assert_eq!(s.shards[0].name, "a");
        assert_eq!(s.shards[1].name, "b");
    }

    #[test]
    fn finish_pins_to_total() {
        let t = DownloadTracker::default();
        t.begin();
        t.on_start("a", 100);
        t.on_advance("a", 90); // short of total
        t.on_finish("a");
        // A second shard keeps the modal open so we can inspect `a`.
        t.on_start("b", 200);
        t.on_advance("b", 10);
        let s = t.snapshot().expect("b still downloading");
        let a = s.shards.iter().find(|x| x.name == "a").unwrap();
        assert_eq!(a.done_bytes, 100, "finish pins done == total");
        assert!(a.is_done());
    }

    #[test]
    fn all_complete_clears_modal() {
        // The modal must clear the moment the download finishes, without
        // waiting for `end()` (which only fires after the whole engine build).
        let t = DownloadTracker::default();
        t.begin();
        t.on_start("a", 100);
        t.on_advance("a", 50);
        assert!(t.snapshot().is_some(), "in-flight → shown");
        t.on_advance("a", 50); // now complete
        assert!(
            t.snapshot().is_none(),
            "all shards complete → modal clears before end()"
        );
    }

    #[test]
    fn fail_sets_error_and_clears_progress() {
        let t = DownloadTracker::default();
        t.begin();
        t.on_start("a", 100);
        t.on_advance("a", 30);
        assert!(t.snapshot().is_some());
        t.fail("insufficient disk space".into());
        assert!(t.snapshot().is_none(), "progress modal gone on failure");
        assert_eq!(t.error().as_deref(), Some("insufficient disk space"));
        t.clear_error();
        assert!(t.error().is_none());
    }

    #[test]
    fn begin_clears_prior_error() {
        let t = DownloadTracker::default();
        t.fail("boom".into());
        assert!(t.error().is_some());
        t.begin();
        assert!(t.error().is_none(), "a new attempt clears the old error");
    }

    #[test]
    fn end_clears_modal() {
        let t = DownloadTracker::default();
        t.begin();
        t.on_start("a", 100);
        t.on_advance("a", 50); // still in flight
        assert!(t.snapshot().is_some());
        t.end();
        assert!(t.snapshot().is_none(), "end closes the window");
    }

    #[test]
    fn begin_clears_prior_run() {
        let t = DownloadTracker::default();
        t.begin();
        t.on_start("a", 100);
        t.on_advance("a", 100);
        t.end();
        // A second build reuses the tracker: no stale shards leak in.
        t.begin();
        assert!(t.snapshot().is_none());
        t.on_start("b", 50);
        let s = t.snapshot().expect("second run active");
        assert_eq!(s.shards.len(), 1);
        assert_eq!(s.total_bytes, 50);
    }
}
