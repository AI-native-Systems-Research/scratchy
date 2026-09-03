// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Anthropic Messages API (`/v1/messages`) — translation layer over the
//! internal chat completion engine.

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_stream::StreamExt as _;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::info;

use crate::engine::StreamDelta;
use crate::protocol;
use crate::server::AppState;

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// Content of an Anthropic message — either a plain string or array of blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// A single content block inside a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<Value>,
    },
    #[serde(rename = "image")]
    Image { source: Value },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Option<ToolResultContent>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<Value>,
    },
}

/// Content of a tool_result block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ToolResultBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolResultBlock {
    #[serde(rename = "text")]
    Text { text: String },
}

/// System parameter — string or array of system blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemParam {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemBlock {
    pub text: String,
    #[serde(default)]
    pub cache_control: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: MessageContent,
}

/// An Anthropic tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub input_schema: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<Value>,
}

/// Anthropic Messages API request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<SystemParam>,
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub metadata: Option<Value>,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<ResponseContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ---------------------------------------------------------------------------
// Conversion: Anthropic request → ChatCompletionRequest
// ---------------------------------------------------------------------------

fn convert_request(req: MessagesRequest) -> protocol::ChatCompletionRequest {
    let mut messages = Vec::new();

    // System message
    if let Some(sys) = req.system {
        let text = match sys {
            SystemParam::Text(t) => t,
            SystemParam::Blocks(blocks) => blocks
                .into_iter()
                .map(|b| b.text)
                .collect::<Vec<_>>()
                .join("\n"),
        };
        messages.push(protocol::ChatCompletionMessageParam {
            role: "system".to_string(),
            content: Some(Value::String(text)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // Convert messages
    for msg in req.messages {
        match msg.content {
            MessageContent::Text(text) => {
                messages.push(protocol::ChatCompletionMessageParam {
                    role: msg.role,
                    content: Some(Value::String(text)),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            MessageContent::Blocks(blocks) => {
                let mut text_parts = Vec::new();
                let mut tool_calls = Vec::new();
                let mut tool_results: Vec<(String, String)> = Vec::new();

                for block in blocks {
                    match block {
                        ContentBlock::Text { text, .. } => text_parts.push(text),
                        ContentBlock::Image { .. } => {}
                        ContentBlock::ToolUse { id, name, input } => {
                            tool_calls.push(protocol::ToolCall {
                                id,
                                call_type: "function".to_string(),
                                function: protocol::FunctionCall {
                                    name,
                                    arguments: serde_json::to_string(&input).unwrap_or_default(),
                                },
                            });
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            let text = match content {
                                Some(ToolResultContent::Text(t)) => t,
                                Some(ToolResultContent::Blocks(bs)) => bs
                                    .into_iter()
                                    .map(|b| match b {
                                        ToolResultBlock::Text { text } => text,
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                                None => String::new(),
                            };
                            tool_results.push((tool_use_id, text));
                        }
                    }
                }

                // Emit one "tool" message per tool result
                for (tool_use_id, text) in tool_results {
                    messages.push(protocol::ChatCompletionMessageParam {
                        role: "tool".to_string(),
                        content: Some(Value::String(text)),
                        name: None,
                        tool_calls: None,
                        tool_call_id: Some(tool_use_id),
                    });
                }

                // Emit the main message (text + tool_calls) if present
                if !text_parts.is_empty() || !tool_calls.is_empty() {
                    let content = if text_parts.is_empty() {
                        None
                    } else {
                        Some(Value::String(text_parts.join("\n")))
                    };
                    let tc = if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    };
                    messages.push(protocol::ChatCompletionMessageParam {
                        role: msg.role,
                        content,
                        name: None,
                        tool_calls: tc,
                        tool_call_id: None,
                    });
                }
            }
        }
    }

    // Convert tools
    let tools = req.tools.map(|ts| {
        ts.into_iter()
            .map(|t| protocol::ChatCompletionToolsParam {
                tool_type: "function".to_string(),
                function: protocol::FunctionDefinition {
                    name: t.name,
                    description: t.description,
                    parameters: Some(t.input_schema),
                },
            })
            .collect()
    });

    // Convert tool_choice
    let tool_choice = req.tool_choice.map(|tc| {
        if let Some(s) = tc.as_str() {
            match s {
                "auto" => Value::String("auto".to_string()),
                "any" => Value::String("required".to_string()),
                "none" => Value::String("none".to_string()),
                _ => tc,
            }
        } else if let Some(name) = tc.get("name").and_then(|n| n.as_str()) {
            serde_json::json!({
                "type": "function",
                "function": { "name": name }
            })
        } else {
            tc
        }
    });

    let stop = req.stop_sequences.map(protocol::StopCondition::Multiple);

    protocol::ChatCompletionRequest {
        model: req.model,
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        n: 1,
        max_tokens: Some(req.max_tokens),
        max_completion_tokens: None,
        stream: false,
        stream_options: None,
        stop,
        frequency_penalty: None,
        presence_penalty: None,
        logit_bias: None,
        logprobs: None,
        top_logprobs: None,
        prompt_logprobs: None,
        seed: None,
        response_format: None,
        tools,
        tool_choice,
        user: None,
        top_k: req.top_k.map(|k| k as i32),
        min_p: None,
        repetition_penalty: None,
        min_tokens: 0,
        stop_token_ids: Vec::new(),
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

// ---------------------------------------------------------------------------
// Conversion: ChatCompletionResponse → MessagesResponse
// ---------------------------------------------------------------------------

fn map_finish_reason(reason: Option<&str>) -> Option<String> {
    reason.map(|r| {
        match r {
            "stop" => "end_turn",
            "length" => "max_tokens",
            "tool_calls" => "tool_use",
            _ => "end_turn",
        }
        .to_string()
    })
}

fn convert_response(resp: protocol::ChatCompletionResponse) -> MessagesResponse {
    let choice = resp.choices.into_iter().next();

    let (content, stop_reason) = match choice {
        Some(c) => {
            let mut blocks = Vec::new();

            if let Some(text) = c.message.content.filter(|t| !t.is_empty()) {
                blocks.push(ResponseContentBlock::Text { text });
            }

            if let Some(tool_calls) = c.message.tool_calls {
                for tc in tool_calls {
                    let input: Value = serde_json::from_str(&tc.function.arguments)
                        .unwrap_or(Value::Object(serde_json::Map::new()));
                    blocks.push(ResponseContentBlock::ToolUse {
                        id: tc.id,
                        name: tc.function.name,
                        input,
                    });
                }
            }

            let stop_reason = map_finish_reason(c.finish_reason.as_deref());
            (blocks, stop_reason)
        }
        None => (vec![], Some("end_turn".to_string())),
    };

    MessagesResponse {
        id: format!("msg_{}", resp.id),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: resp.model,
        stop_reason,
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: resp.usage.prompt_tokens,
            output_tokens: resp.usage.completion_tokens.unwrap_or(0),
        },
    }
}

// ---------------------------------------------------------------------------
// Spans mode: each tool a relocatable Plus span
// ---------------------------------------------------------------------------

/// Per-tool relocatable spans for tools-bearing `/v1/messages` requests are
/// always on — each tool becomes an independently-cacheable relocatable span
/// (`Cross([system, Plus([tool₁…toolₙ]), conversation])`) with a
/// block-diagonal attention bound.
fn spans_enabled() -> bool {
    true
}

/// Render one tool definition as the text of its relocatable span.
fn render_tool(t: &AnthropicTool) -> String {
    format!(
        "Tool: {}\nDescription: {}\nInput schema: {}",
        t.name,
        t.description.as_deref().unwrap_or(""),
        serde_json::to_string(&t.input_schema).unwrap_or_default()
    )
}

/// Flatten the conversation (text + tool-result text) into ordered blocks,
/// each with its client-declared `cache_control` breakpoint flag.
fn conversation_blocks(messages: &[AnthropicMessage]) -> Vec<(String, bool)> {
    let mut parts = Vec::new();
    for m in messages {
        match &m.content {
            MessageContent::Text(t) => parts.push((t.clone(), false)),
            MessageContent::Blocks(bs) => {
                for b in bs {
                    match b {
                        ContentBlock::Text {
                            text,
                            cache_control,
                        } => parts.push((text.clone(), cache_control.is_some())),
                        ContentBlock::ToolResult {
                            content: Some(c),
                            cache_control,
                            ..
                        } => {
                            let t = match c {
                                ToolResultContent::Text(t) => t.clone(),
                                ToolResultContent::Blocks(bs) => bs
                                    .iter()
                                    .map(|x| match x {
                                        ToolResultBlock::Text { text } => text.clone(),
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                            };
                            parts.push((t, cache_control.is_some()));
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    parts
}

/// Build a SPNL query from the client's own `cache_control` declarations.
///
/// The Anthropic API's prompt-caching contract already answers "what is the
/// stable prefix": clients mark cache breakpoints, and everything up to and
/// including the LAST breakpoint (in prompt order: system → tools →
/// conversation) is content the client itself declares stable; everything
/// after it is the live turn. So every block covered by the last breakpoint
/// becomes a relocatable span (content-addressed, self-healing on change)
/// and everything after it stays ordinary full-causal prefill — no server
/// guessing about block sizes or shapes. A request with no breakpoints keeps
/// the conservative shape: fresh system, per-tool spans (tool schemas are
/// position-independent sets by definition), fresh conversation.
fn build_spnl_query(req: &MessagesRequest) -> String {
    let sys_blocks: Vec<(String, bool)> = match &req.system {
        None => vec![],
        Some(SystemParam::Text(t)) => vec![(t.clone(), false)],
        Some(SystemParam::Blocks(b)) => b
            .iter()
            .map(|x| (x.text.clone(), x.cache_control.is_some()))
            .collect(),
    };
    let tools: Vec<&AnthropicTool> = req.tools.iter().flatten().collect();
    let conv = conversation_blocks(&req.messages);

    // Prompt-order slots: system blocks, then one slot for the tools set,
    // then conversation blocks. Find the last client breakpoint.
    let t_slot = sys_blocks.len();
    let mut last_cc: Option<usize> = None;
    for (i, (_, cc)) in sys_blocks.iter().enumerate() {
        if *cc {
            last_cc = Some(i);
        }
    }
    if !tools.is_empty() && tools.iter().any(|t| t.cache_control.is_some()) {
        last_cc = Some(t_slot);
    }
    for (i, (_, cc)) in conv.iter().enumerate() {
        if *cc {
            last_cc = Some(t_slot + 1 + i);
        }
    }
    let covered = |slot: usize| last_cc.is_some_and(|c| slot <= c);
    info!(
        "[SPANS] blocks: system={} tools={} conversation={} last_breakpoint={:?}",
        sys_blocks.len(),
        tools.len(),
        conv.len(),
        last_cc
    );

    let mut cross: Vec<Value> = Vec::new();
    let mut fresh: Vec<String> = Vec::new();
    for (i, (text, _)) in sys_blocks.iter().enumerate() {
        if covered(i) {
            if !fresh.is_empty() {
                cross.push(serde_json::json!({ "system": fresh.join("\n") }));
                fresh.clear();
            }
            cross.push(serde_json::json!({ "plus": [{ "system": text }] }));
        } else {
            fresh.push(text.clone());
        }
    }
    if !fresh.is_empty() {
        cross.push(serde_json::json!({ "system": fresh.join("\n") }));
        fresh.clear();
    }
    if !tools.is_empty() {
        let tool_nodes: Vec<Value> = tools
            .iter()
            .map(|t| serde_json::json!({ "user": render_tool(t) }))
            .collect();
        cross.push(serde_json::json!({ "plus": tool_nodes }));
    }
    for (i, (text, _)) in conv.iter().enumerate() {
        if covered(t_slot + 1 + i) {
            if !fresh.is_empty() {
                cross.push(serde_json::json!({ "user": fresh.join("\n") }));
                fresh.clear();
            }
            cross.push(serde_json::json!({ "plus": [{ "user": text }] }));
        } else {
            fresh.push(text.clone());
        }
    }
    // The final Cross child is always the fresh live-turn region the model
    // generates from (empty only if the client breakpointed its whole turn).
    cross.push(serde_json::json!({ "user": fresh.join("\n") }));
    serde_json::json!({
        "g": {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature.unwrap_or(0.0),
            "input": { "cross": cross }
        }
    })
    .to_string()
}

/// Convert a raw completion response (from the spans path) into a `MessagesResponse`,
/// applying the engine's tool-call parser so the model's tool calls surface as
/// `tool_use` blocks (exactly what the flat chat path does internally).
fn completion_to_messages_response(
    resp: protocol::CompletionResponse,
    parser: Option<&std::sync::Arc<dyn crate::tool_parser::ToolCallParser>>,
) -> MessagesResponse {
    let text = resp
        .choices
        .into_iter()
        .next()
        .map(|c| c.text)
        .unwrap_or_default();

    let (content, stop_reason) = match parser {
        Some(p) => {
            let info = p.extract_tool_calls(&text);
            let mut blocks = Vec::new();
            if let Some(c) = info.content.filter(|s| !s.is_empty()) {
                blocks.push(ResponseContentBlock::Text { text: c });
            }
            for tc in info.tool_calls {
                let input: Value = serde_json::from_str(&tc.function.arguments)
                    .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
                blocks.push(ResponseContentBlock::ToolUse {
                    id: tc.id,
                    name: tc.function.name,
                    input,
                });
            }
            if blocks.is_empty() {
                blocks.push(ResponseContentBlock::Text { text });
            }
            let stop = if info.tools_called {
                "tool_use"
            } else {
                "end_turn"
            };
            (blocks, stop.to_string())
        }
        None => (
            vec![ResponseContentBlock::Text { text }],
            "end_turn".to_string(),
        ),
    };

    MessagesResponse {
        id: format!("msg_{}", resp.id),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: resp.model,
        stop_reason: Some(stop_reason),
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: resp.usage.prompt_tokens,
            output_tokens: resp.usage.completion_tokens.unwrap_or(0),
        },
    }
}

/// Replay a buffered [`MessagesResponse`] (from the spans path) as the Anthropic
/// streaming SSE sequence so `stream:true` clients (Claude Code) get a valid
/// event stream. The prefill — the part spans accelerates — already happened;
/// the decode is small, so buffering it before replaying is fine.
fn stream_buffered_messages_response(
    msg: MessagesResponse,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let mut events: Vec<(&'static str, Value)> = Vec::new();
    events.push((
        "message_start",
        serde_json::json!({
            "type": "message_start",
            "message": {
                "id": msg.id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": msg.model,
                "stop_reason": null,
                "stop_sequence": null,
                "usage": { "input_tokens": msg.usage.input_tokens, "output_tokens": 0 }
            }
        }),
    ));
    events.push(("ping", serde_json::json!({ "type": "ping" })));
    for (i, block) in msg.content.iter().enumerate() {
        match block {
            ResponseContentBlock::Text { text } => {
                events.push((
                    "content_block_start",
                    serde_json::json!({
                        "type": "content_block_start",
                        "index": i,
                        "content_block": { "type": "text", "text": "" }
                    }),
                ));
                events.push((
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": i,
                        "delta": { "type": "text_delta", "text": text }
                    }),
                ));
            }
            ResponseContentBlock::ToolUse { id, name, input } => {
                events.push((
                    "content_block_start",
                    serde_json::json!({
                        "type": "content_block_start",
                        "index": i,
                        "content_block": { "type": "tool_use", "id": id, "name": name, "input": {} }
                    }),
                ));
                events.push((
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": i,
                        "delta": { "type": "input_json_delta", "partial_json": input.to_string() }
                    }),
                ));
            }
        }
        events.push((
            "content_block_stop",
            serde_json::json!({ "type": "content_block_stop", "index": i }),
        ));
    }
    events.push((
        "message_delta",
        serde_json::json!({
            "type": "message_delta",
            "delta": { "stop_reason": msg.stop_reason, "stop_sequence": null },
            "usage": { "output_tokens": msg.usage.output_tokens }
        }),
    ));
    events.push((
        "message_stop",
        serde_json::json!({ "type": "message_stop" }),
    ));

    let sse: Vec<Result<Event, Infallible>> = events
        .into_iter()
        .map(|(name, payload)| {
            Ok(Event::default()
                .event(name)
                .data(serde_json::to_string(&payload).unwrap_or_default()))
        })
        .collect();
    Sse::new(tokio_stream::iter(sse)).keep_alive(KeepAlive::default())
}

// ---------------------------------------------------------------------------
// Handler: POST /v1/messages
// ---------------------------------------------------------------------------

/// POST /v1/messages — Anthropic Messages API.
pub async fn messages(
    State(state): State<Arc<AppState>>,
    Json(request): Json<MessagesRequest>,
) -> Response {
    info!(
        "POST /v1/messages: model={:?}, max_tokens={}, stream={}",
        request.model, request.max_tokens, request.stream
    );

    let is_stream = request.stream;

    // Spans mode: render a tools-bearing request as per-tool relocatable
    // spans. The buffered
    // completion is tool-parsed and returned as JSON or, for `stream:true`,
    // replayed as Anthropic SSE so streaming clients (Claude Code) work too.
    if spans_enabled()
        && request
            .tools
            .as_ref()
            .map(|t| !t.is_empty())
            .unwrap_or(false)
    {
        let spnl = build_spnl_query(&request);
        match crate::query::anthropic_spans_completion(&state, &spnl).await {
            Ok(resp) => {
                let msg =
                    completion_to_messages_response(resp, state.engine.tool_parser().as_ref());
                return if is_stream {
                    stream_buffered_messages_response(msg).into_response()
                } else {
                    Json(msg).into_response()
                };
            }
            Err(e) => return e.into_response(),
        }
    }

    let mut chat_request = convert_request(request);

    if is_stream {
        chat_request.stream = true;
        match state.engine.chat_completion_stream(chat_request).await {
            Ok((request_id, model, rx)) => {
                stream_messages_response(request_id, model, rx).into_response()
            }
            Err(e) => e.into_response(),
        }
    } else {
        match state.engine.chat_completion(chat_request).await {
            Ok(response) => {
                let anthropic_response = convert_response(response);
                Json(anthropic_response).into_response()
            }
            Err(e) => e.into_response(),
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming SSE
// ---------------------------------------------------------------------------

/// The currently-open Anthropic content block (at most one is open at a time).
#[derive(Clone, Copy)]
enum BlockKind {
    Text,
    /// A `tool_use` block, tagged with the originating OpenAI tool-call index so
    /// later argument fragments for the same call route to the same block.
    Tool {
        oai_index: u32,
    },
}

struct OpenBlock {
    /// Position of this block in the Anthropic `content` array.
    index: usize,
    kind: BlockKind,
}

/// Stateful translator from internal [`StreamDelta`]s to the Anthropic Messages
/// SSE event sequence (`message_start` → per-block `content_block_*` →
/// `message_delta` → `message_stop`).
///
/// Extracted from the HTTP handler so the event ordering — especially `tool_use`
/// streaming, which must use `input_json_delta`/`partial_json` and *not*
/// `text_delta` (a `text_delta` on a `tool_use` block crashes the Anthropic SDK)
/// — can be unit-tested without a live server.
struct AnthropicSseEncoder {
    msg_id: String,
    model: String,
    request_id: String,
    started: bool,
    output_tokens: u32,
    saw_tool_use: bool,
    open: Option<OpenBlock>,
    next_index: usize,
}

impl AnthropicSseEncoder {
    fn new(request_id: String, model: String) -> Self {
        Self {
            msg_id: format!("msg_{request_id}"),
            model,
            request_id,
            started: false,
            output_tokens: 0,
            saw_tool_use: false,
            open: None,
            next_index: 0,
        }
    }

    /// Close whatever block is currently open, if any.
    fn close_open(&mut self, events: &mut Vec<(&'static str, Value)>) {
        if let Some(b) = self.open.take() {
            events.push((
                "content_block_stop",
                serde_json::json!({ "type": "content_block_stop", "index": b.index }),
            ));
        }
    }

    /// Process one delta, returning the ordered `(event_name, payload)` pairs to emit.
    fn push(&mut self, delta: &StreamDelta) -> Vec<(&'static str, Value)> {
        let mut events = Vec::new();
        self.output_tokens += delta.new_token_ids.len() as u32;

        if !self.started {
            self.started = true;
            events.push((
                "message_start",
                serde_json::json!({
                    "type": "message_start",
                    "message": {
                        "id": self.msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.model,
                        "stop_reason": null,
                        "stop_sequence": null,
                        "usage": { "input_tokens": 0, "output_tokens": 0 }
                    }
                }),
            ));
            events.push(("ping", serde_json::json!({ "type": "ping" })));
        }

        // Tool-call deltas and text deltas never arrive in the same `StreamDelta`
        // (the engine sets one or the other), so handle them as alternatives.
        if let Some(tool_deltas) = delta.tool_call_deltas.as_ref() {
            self.saw_tool_use = true;
            for tc in tool_deltas {
                let cur = match self.open.as_ref().map(|b| b.kind) {
                    Some(BlockKind::Tool { oai_index }) => Some(oai_index),
                    _ => None,
                };
                // A new tool index (or switching away from a text block) opens a
                // fresh `tool_use` block carrying the call id + function name.
                if cur != Some(tc.index) {
                    self.close_open(&mut events);
                    let bidx = self.next_index;
                    self.next_index += 1;
                    let id = tc
                        .id
                        .clone()
                        .unwrap_or_else(|| format!("toolu_{}_{}", self.request_id, bidx));
                    let name = tc.function_name.clone().unwrap_or_default();
                    events.push((
                        "content_block_start",
                        serde_json::json!({
                            "type": "content_block_start",
                            "index": bidx,
                            "content_block": {
                                "type": "tool_use", "id": id, "name": name, "input": {}
                            }
                        }),
                    ));
                    self.open = Some(OpenBlock {
                        index: bidx,
                        kind: BlockKind::Tool {
                            oai_index: tc.index,
                        },
                    });
                }
                // Argument fragments stream as partial JSON, NOT text.
                if let Some(args) = tc.function_arguments.as_ref() {
                    let bidx = self.open.as_ref().map(|b| b.index).unwrap_or(0);
                    events.push((
                        "content_block_delta",
                        serde_json::json!({
                            "type": "content_block_delta",
                            "index": bidx,
                            "delta": { "type": "input_json_delta", "partial_json": args }
                        }),
                    ));
                }
            }
        } else if let Some(text) = delta.text.as_ref().filter(|t| !t.is_empty()) {
            if !matches!(self.open.as_ref().map(|b| b.kind), Some(BlockKind::Text)) {
                self.close_open(&mut events);
                let bidx = self.next_index;
                self.next_index += 1;
                events.push((
                    "content_block_start",
                    serde_json::json!({
                        "type": "content_block_start",
                        "index": bidx,
                        "content_block": { "type": "text", "text": "" }
                    }),
                ));
                self.open = Some(OpenBlock {
                    index: bidx,
                    kind: BlockKind::Text,
                });
            }
            let bidx = self.open.as_ref().map(|b| b.index).unwrap_or(0);
            events.push((
                "content_block_delta",
                serde_json::json!({
                    "type": "content_block_delta",
                    "index": bidx,
                    "delta": { "type": "text_delta", "text": text }
                }),
            ));
        }

        if delta.finish_reason.is_some() {
            self.close_open(&mut events);
            let stop_reason = if self.saw_tool_use {
                "tool_use".to_string()
            } else {
                let r = delta.finish_reason.map(|r| r.to_string());
                map_finish_reason(r.as_deref()).unwrap_or_else(|| "end_turn".to_string())
            };
            events.push((
                "message_delta",
                serde_json::json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": stop_reason, "stop_sequence": null },
                    "usage": { "output_tokens": self.output_tokens }
                }),
            ));
            events.push((
                "message_stop",
                serde_json::json!({ "type": "message_stop" }),
            ));
        }

        events
    }
}

fn stream_messages_response(
    request_id: String,
    model: String,
    rx: tokio::sync::mpsc::UnboundedReceiver<StreamDelta>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let (tx, out_rx) = tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();

    tokio::spawn(async move {
        let mut encoder = AnthropicSseEncoder::new(request_id, model);
        let mut stream = UnboundedReceiverStream::new(rx);
        while let Some(delta) = stream.next().await {
            for (event, payload) in encoder.push(&delta) {
                let _ = tx.send(Ok(Event::default()
                    .event(event)
                    .json_data(payload)
                    .unwrap()));
            }
        }
    });

    Sse::new(UnboundedReceiverStream::new(out_rx)).keep_alive(KeepAlive::default())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_simple_request() {
        let json = r#"{
            "model": "claude-3-sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}]
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, 100);
        assert_eq!(req.messages.len(), 1);
        assert!(!req.stream);
    }

    #[test]
    fn test_deserialize_content_blocks() {
        let json = r#"{
            "max_tokens": 50,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "text", "text": "Tell me more."}
                ]
            }]
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            MessageContent::Blocks(blocks) => assert_eq!(blocks.len(), 2),
            _ => panic!("expected blocks"),
        }
    }

    #[test]
    fn test_deserialize_system_string() {
        let json = r#"{
            "max_tokens": 10,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}]
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        match req.system.unwrap() {
            SystemParam::Text(t) => assert_eq!(t, "You are helpful."),
            _ => panic!("expected text system"),
        }
    }

    #[test]
    fn test_deserialize_system_blocks() {
        let json = r#"{
            "max_tokens": 10,
            "system": [{"text": "Be concise."}, {"text": "Be accurate."}],
            "messages": [{"role": "user", "content": "Hi"}]
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        match req.system.unwrap() {
            SystemParam::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert_eq!(blocks[0].text, "Be concise.");
            }
            _ => panic!("expected system blocks"),
        }
    }

    #[test]
    fn test_deserialize_tools() {
        let json = r#"{
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
            }]
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
    }

    #[test]
    fn test_deserialize_tool_result() {
        let json = r#"{
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_123", "content": "72°F"}
                ]
            }]
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            MessageContent::Blocks(blocks) => match &blocks[0] {
                ContentBlock::ToolResult { tool_use_id, .. } => {
                    assert_eq!(tool_use_id, "call_123");
                }
                _ => panic!("expected tool_result"),
            },
            _ => panic!("expected blocks"),
        }
    }

    #[test]
    fn test_deserialize_stream_flag() {
        let json = r#"{
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert!(req.stream);
    }

    #[test]
    fn test_convert_simple_request() {
        let req = MessagesRequest {
            model: Some("test-model".into()),
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: MessageContent::Text("Hello".into()),
            }],
            system: None,
            max_tokens: 100,
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            tools: None,
            tool_choice: None,
            metadata: None,
        };
        let chat = convert_request(req);
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "user");
        assert_eq!(chat.max_tokens, Some(100));
        assert_eq!(chat.temperature, Some(0.7));
    }

    #[test]
    fn test_convert_system_message() {
        let req = MessagesRequest {
            model: None,
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: MessageContent::Text("Hi".into()),
            }],
            system: Some(SystemParam::Text("Be helpful.".into())),
            max_tokens: 50,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            tools: None,
            tool_choice: None,
            metadata: None,
        };
        let chat = convert_request(req);
        assert_eq!(chat.messages.len(), 2);
        assert_eq!(chat.messages[0].role, "system");
        assert_eq!(
            chat.messages[0].content,
            Some(Value::String("Be helpful.".into()))
        );
    }

    #[test]
    fn test_convert_stop_sequences() {
        let req = MessagesRequest {
            model: None,
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: MessageContent::Text("Hi".into()),
            }],
            system: None,
            max_tokens: 50,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: Some(vec!["END".into(), "STOP".into()]),
            stream: false,
            tools: None,
            tool_choice: None,
            metadata: None,
        };
        let chat = convert_request(req);
        match chat.stop.unwrap() {
            protocol::StopCondition::Multiple(v) => {
                assert_eq!(v, vec!["END", "STOP"]);
            }
            _ => panic!("expected Multiple stop"),
        }
    }

    #[test]
    fn test_convert_tool_use_blocks() {
        let req = MessagesRequest {
            model: None,
            messages: vec![AnthropicMessage {
                role: "assistant".into(),
                content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "get_weather".into(),
                    input: serde_json::json!({"city": "NYC"}),
                }]),
            }],
            system: None,
            max_tokens: 50,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            tools: None,
            tool_choice: None,
            metadata: None,
        };
        let chat = convert_request(req);
        assert_eq!(chat.messages.len(), 1);
        let tc = chat.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(tc[0].id, "call_1");
        assert_eq!(tc[0].function.name, "get_weather");
    }

    #[test]
    fn test_convert_tool_result_blocks() {
        let req = MessagesRequest {
            model: None,
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".into(),
                    content: Some(ToolResultContent::Text("72°F".into())),
                    cache_control: None,
                }]),
            }],
            system: None,
            max_tokens: 50,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            tools: None,
            tool_choice: None,
            metadata: None,
        };
        let chat = convert_request(req);
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "tool");
        assert_eq!(chat.messages[0].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn test_convert_tools() {
        let req = MessagesRequest {
            model: None,
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: MessageContent::Text("Hi".into()),
            }],
            system: None,
            max_tokens: 50,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            tools: Some(vec![AnthropicTool {
                name: "search".into(),
                description: Some("Search the web".into()),
                input_schema: serde_json::json!({"type": "object"}),
                cache_control: None,
            }]),
            tool_choice: None,
            metadata: None,
        };
        let chat = convert_request(req);
        let tools = chat.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "search");
        assert_eq!(tools[0].tool_type, "function");
    }

    #[test]
    fn test_convert_tool_choice_any() {
        let req = MessagesRequest {
            model: None,
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: MessageContent::Text("Hi".into()),
            }],
            system: None,
            max_tokens: 50,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            tools: None,
            tool_choice: Some(Value::String("any".into())),
            metadata: None,
        };
        let chat = convert_request(req);
        assert_eq!(chat.tool_choice, Some(Value::String("required".into())));
    }

    #[test]
    fn test_convert_simple_response() {
        let resp = protocol::ChatCompletionResponse {
            id: "chatcmpl-123".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "test".into(),
            choices: vec![protocol::ChatCompletionResponseChoice {
                index: 0,
                message: protocol::ChatMessage {
                    role: "assistant".into(),
                    content: Some("Hello!".into()),
                    refusal: None,
                    tool_calls: None,
                    reasoning: None,
                },
                logprobs: None,
                finish_reason: Some("stop".into()),
                stop_reason: None,
                prompt_logprobs: None,
            }],
            system_fingerprint: None,
            usage: protocol::UsageInfo {
                prompt_tokens: 10,
                total_tokens: 15,
                completion_tokens: Some(5),
                prompt_tokens_details: None,
            },
        };
        let anthropic = convert_response(resp);
        assert_eq!(anthropic.response_type, "message");
        assert_eq!(anthropic.role, "assistant");
        assert_eq!(anthropic.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(anthropic.usage.input_tokens, 10);
        assert_eq!(anthropic.usage.output_tokens, 5);
        match &anthropic.content[0] {
            ResponseContentBlock::Text { text } => assert_eq!(text, "Hello!"),
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn test_convert_tool_calls_response() {
        let resp = protocol::ChatCompletionResponse {
            id: "chatcmpl-456".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "test".into(),
            choices: vec![protocol::ChatCompletionResponseChoice {
                index: 0,
                message: protocol::ChatMessage {
                    role: "assistant".into(),
                    content: None,
                    refusal: None,
                    tool_calls: Some(vec![protocol::ToolCall {
                        id: "call_1".into(),
                        call_type: "function".into(),
                        function: protocol::FunctionCall {
                            name: "get_weather".into(),
                            arguments: r#"{"city":"NYC"}"#.into(),
                        },
                    }]),
                    reasoning: None,
                },
                logprobs: None,
                finish_reason: Some("tool_calls".into()),
                stop_reason: None,
                prompt_logprobs: None,
            }],
            system_fingerprint: None,
            usage: protocol::UsageInfo {
                prompt_tokens: 20,
                total_tokens: 30,
                completion_tokens: Some(10),
                prompt_tokens_details: None,
            },
        };
        let anthropic = convert_response(resp);
        assert_eq!(anthropic.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(anthropic.content.len(), 1);
        match &anthropic.content[0] {
            ResponseContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "NYC");
            }
            _ => panic!("expected tool_use block"),
        }
    }

    #[test]
    fn test_convert_max_tokens_finish() {
        let resp = protocol::ChatCompletionResponse {
            id: "chatcmpl-789".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "test".into(),
            choices: vec![protocol::ChatCompletionResponseChoice {
                index: 0,
                message: protocol::ChatMessage {
                    role: "assistant".into(),
                    content: Some("truncated...".into()),
                    refusal: None,
                    tool_calls: None,
                    reasoning: None,
                },
                logprobs: None,
                finish_reason: Some("length".into()),
                stop_reason: None,
                prompt_logprobs: None,
            }],
            system_fingerprint: None,
            usage: protocol::UsageInfo {
                prompt_tokens: 5,
                total_tokens: 105,
                completion_tokens: Some(100),
                prompt_tokens_details: None,
            },
        };
        let anthropic = convert_response(resp);
        assert_eq!(anthropic.stop_reason.as_deref(), Some("max_tokens"));
    }

    #[test]
    fn test_response_serialization() {
        let resp = MessagesResponse {
            id: "msg_123".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            content: vec![ResponseContentBlock::Text { text: "Hi".into() }],
            model: "test".into(),
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 1,
                output_tokens: 1,
            },
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["type"], "message");
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][0]["text"], "Hi");
        assert_eq!(json["stop_reason"], "end_turn");
        assert_eq!(json["usage"]["input_tokens"], 1);
    }

    #[test]
    fn test_finish_reason_mapping() {
        assert_eq!(map_finish_reason(Some("stop")).as_deref(), Some("end_turn"));
        assert_eq!(
            map_finish_reason(Some("length")).as_deref(),
            Some("max_tokens")
        );
        assert_eq!(
            map_finish_reason(Some("tool_calls")).as_deref(),
            Some("tool_use")
        );
        assert_eq!(map_finish_reason(None), None);
    }

    // -----------------------------------------------------------------------
    // Streaming encoder (AnthropicSseEncoder)
    // -----------------------------------------------------------------------

    use crate::tool_parser::DeltaToolCall;
    use scratchy_core_common::FinishReason;

    fn mk_delta(
        text: Option<&str>,
        tools: Option<Vec<DeltaToolCall>>,
        finish: Option<FinishReason>,
    ) -> StreamDelta {
        StreamDelta {
            index: 0,
            new_token_ids: vec![1],
            text: text.map(str::to_string),
            finish_reason: finish,
            stop_reason: None,
            logprobs: None,
            tool_call_deltas: tools,
            reasoning: None,
        }
    }

    fn mk_tc(
        index: u32,
        id: Option<&str>,
        name: Option<&str>,
        args: Option<&str>,
    ) -> DeltaToolCall {
        DeltaToolCall {
            index,
            id: id.map(str::to_string),
            call_type: id.map(|_| "function".to_string()),
            function_name: name.map(str::to_string),
            function_arguments: args.map(str::to_string),
        }
    }

    fn event_names(events: &[(&'static str, Value)]) -> Vec<&'static str> {
        events.iter().map(|(n, _)| *n).collect()
    }

    /// The load-bearing case: Claude Code only works if tool-call arguments stream
    /// as `input_json_delta`/`partial_json` (a `text_delta` on a `tool_use` block
    /// crashes the Anthropic SDK), and the `content_block_start` carries the real
    /// tool id + name.
    #[test]
    fn test_stream_tool_use_emits_input_json_delta() {
        let mut enc = AnthropicSseEncoder::new("req1".into(), "m".into());
        let mut events = Vec::new();
        // Opening tool delta: id + name + first argument fragment.
        events.extend(enc.push(&mk_delta(
            None,
            Some(vec![mk_tc(
                0,
                Some("call_a"),
                Some("get_weather"),
                Some("{\"city\":"),
            )]),
            None,
        )));
        // Continuation fragment (no id/name).
        events.extend(enc.push(&mk_delta(
            None,
            Some(vec![mk_tc(0, None, None, Some("\"NYC\"}"))]),
            None,
        )));
        // Engine sends the finish in a separate delta with no tool deltas.
        events.extend(enc.push(&mk_delta(None, None, Some(FinishReason::Stop))));

        assert_eq!(
            event_names(&events),
            vec![
                "message_start",
                "ping",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );

        let start = &events[2].1;
        assert_eq!(start["content_block"]["type"], "tool_use");
        assert_eq!(start["content_block"]["id"], "call_a");
        assert_eq!(start["content_block"]["name"], "get_weather");

        // Every delta on the tool block is input_json_delta — never text_delta.
        let mut combined = String::new();
        for (name, payload) in &events {
            if *name == "content_block_delta" {
                assert_eq!(payload["delta"]["type"], "input_json_delta");
                assert!(payload["delta"].get("text").is_none());
                combined.push_str(payload["delta"]["partial_json"].as_str().unwrap());
            }
        }
        assert_eq!(combined, "{\"city\":\"NYC\"}");

        // stop_reason is tool_use even though the finish delta carried Stop.
        assert_eq!(events[6].1["delta"]["stop_reason"], "tool_use");
    }

    #[test]
    fn test_stream_text_basic() {
        let mut enc = AnthropicSseEncoder::new("r".into(), "m".into());
        let mut events = Vec::new();
        events.extend(enc.push(&mk_delta(Some("Hello"), None, None)));
        events.extend(enc.push(&mk_delta(Some(" world"), None, None)));
        events.extend(enc.push(&mk_delta(None, None, Some(FinishReason::Stop))));

        assert_eq!(
            event_names(&events),
            vec![
                "message_start",
                "ping",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert_eq!(events[2].1["content_block"]["type"], "text");
        assert_eq!(events[3].1["delta"]["type"], "text_delta");
        assert_eq!(events[3].1["delta"]["text"], "Hello");
        assert_eq!(events[6].1["delta"]["stop_reason"], "end_turn");
    }

    /// Parallel tool calls must land in distinct content blocks with incrementing
    /// indices — the previous implementation hardcoded index 0 and collided them.
    #[test]
    fn test_stream_parallel_tool_calls_use_distinct_indices() {
        let mut enc = AnthropicSseEncoder::new("r".into(), "m".into());
        let mut events = Vec::new();
        events.extend(enc.push(&mk_delta(
            None,
            Some(vec![mk_tc(0, Some("a"), Some("f0"), Some("{}"))]),
            None,
        )));
        events.extend(enc.push(&mk_delta(
            None,
            Some(vec![mk_tc(1, Some("b"), Some("f1"), Some("{}"))]),
            None,
        )));
        events.extend(enc.push(&mk_delta(None, None, Some(FinishReason::Stop))));

        let starts: Vec<u64> = events
            .iter()
            .filter(|(n, _)| *n == "content_block_start")
            .map(|(_, p)| p["index"].as_u64().unwrap())
            .collect();
        assert_eq!(starts, vec![0, 1]);

        let stops: Vec<u64> = events
            .iter()
            .filter(|(n, _)| *n == "content_block_stop")
            .map(|(_, p)| p["index"].as_u64().unwrap())
            .collect();
        assert_eq!(stops, vec![0, 1]);

        let names: Vec<String> = events
            .iter()
            .filter(|(n, _)| *n == "content_block_start")
            .map(|(_, p)| p["content_block"]["name"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(names, vec!["f0", "f1"]);
    }

    /// A stream that finishes with no content at all still emits a well-formed
    /// envelope (no dangling open block).
    #[test]
    fn test_stream_empty_finish() {
        let mut enc = AnthropicSseEncoder::new("r".into(), "m".into());
        let events = enc.push(&mk_delta(None, None, Some(FinishReason::Length)));
        assert_eq!(
            event_names(&events),
            vec!["message_start", "ping", "message_delta", "message_stop"]
        );
        assert_eq!(events[2].1["delta"]["stop_reason"], "max_tokens");
    }
}
