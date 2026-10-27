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
use crate::error::{ApiJson, ServeError};
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
    /// An extended-thinking block echoed back by the client (clients round-trip
    /// what the server streamed). Mapped to the assistant message's `reasoning`
    /// so the chat template re-renders it in the model's native thought format.
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
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
    pub role: AnthropicRole,
    pub content: MessageContent,
}

/// Who a message of the conversation is from. Claude Code (2.1+) sends `system` messages
/// inside `messages` too — its environment block after the first user turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicRole {
    User,
    Assistant,
    System,
}

impl AnthropicRole {
    /// The chat-completions role.
    fn chat_role(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
        }
    }
}

/// An Anthropic tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub input_schema: crate::protocol::OrderedJson,
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
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(default)]
    pub metadata: Option<Value>,
    /// Anthropic thinking control. `{"type": "adaptive"}` (what Claude Code
    /// sends) and `{"type": "enabled", …}` leave reasoning streamed back as
    /// thinking blocks; only an explicit `{"type": "disabled"}` suppresses
    /// them (the model still thinks server-side; its channels just stop
    /// being surfaced).
    #[serde(default)]
    pub thinking: Option<Value>,
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
    /// Extended thinking. The signature is synthetic (empty): a local model
    /// has no cryptographic signature to carry, but the field is emitted
    /// because clients round-trip thinking blocks verbatim.
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicUsage {
    /// Prompt tokens actually computed this request — the *uncached remainder*,
    /// NOT the whole prompt.
    ///
    /// The two protocols disagree here, and conflating them double-counts:
    ///   - OpenAI: `prompt_tokens` is the whole prompt, and
    ///     `prompt_tokens_details.cached_tokens` is a **subset** of it.
    ///   - Anthropic: `input_tokens` **excludes** the cached part, so
    ///     `input_tokens + cache_read_input_tokens` is the whole prompt.
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Prompt tokens served from the prefix cache.
    ///
    /// The number epic #158 is built on:
    /// `reprefill_ratio = input_tokens / (input_tokens + cache_read_input_tokens)`
    /// — the share of a re-sent agent prompt this turn actually paid to
    /// recompute. Block-aligned (see `StreamUsage`), so it understates reuse
    /// slightly and never overstates it.
    pub cache_read_input_tokens: u32,
}

impl AnthropicUsage {
    /// Split an OpenAI-shaped [`protocol::UsageInfo`] into Anthropic's disjoint
    /// input fields.
    ///
    /// Deliberately NOT emitting `cache_creation_input_tokens`: scratchy has no
    /// cache *write* that is separate from prefilling the prompt (no
    /// `cache_control` breakpoints, no write premium), so any value there would
    /// be invented rather than measured.
    fn from_usage(usage: &protocol::UsageInfo) -> Self {
        let cached = usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
            .unwrap_or(0);
        Self {
            // `saturating_sub` rather than `-`: unreachable on this endpoint
            // (`convert_request` pins `n: 1` and `MessagesRequest` has no field
            // mapping to `n`), but `engine.rs` sums `num_cached_tokens` across
            // n children against a per-prompt `prompt_tokens`, so the invariant
            // is a property of the caller, not of this function.
            input_tokens: usage.prompt_tokens.saturating_sub(cached),
            output_tokens: usage.completion_tokens.unwrap_or(0),
            cache_read_input_tokens: cached,
        }
    }
}

// ---------------------------------------------------------------------------
// Conversion: Anthropic request → ChatCompletionRequest
// ---------------------------------------------------------------------------

/// Where a request's last prompt-cache breakpoint is: a block of the `system` parameter, or a
/// block of a conversation message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Breakpoint {
    System(usize),
    Message(usize, usize),
}

/// The last of the client's `cache_control` breakpoints in prompt order — the system parameter's
/// blocks, then the conversation's: the furthest point it declares stable for its next request.
fn last_breakpoint(req: &MessagesRequest) -> Option<Breakpoint> {
    let marked = |b: &ContentBlock| match b {
        ContentBlock::Text { cache_control, .. }
        | ContentBlock::ToolResult { cache_control, .. } => cache_control.is_some(),
        _ => false,
    };
    let in_conversation =
        (req.messages.iter().enumerate().rev()).find_map(|(i, m)| match &m.content {
            MessageContent::Blocks(blocks) => {
                (blocks.iter().rposition(marked)).map(|j| Breakpoint::Message(i, j))
            }
            MessageContent::Text(_) => None,
        });
    in_conversation.or_else(|| match &req.system {
        Some(SystemParam::Blocks(blocks)) => {
            (blocks.iter().rposition(|b| b.cache_control.is_some())).map(Breakpoint::System)
        }
        _ => None,
    })
}

/// A request's `tool_choice` in Anthropic's forms ([`protocol::ToolChoice::from_anthropic`]), read
/// and written in them (a client posts this type too, as the e2e one does): one it does not know
/// fails the request as it is parsed. Its `disable_parallel_tool_use: true` is
/// [`protocol::CallLimit::One`], OpenAI's `parallel_tool_calls: false` (#794).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnthropicToolChoice {
    pub choice: protocol::ToolChoice,
    pub limit: protocol::CallLimit,
}

impl AnthropicToolChoice {
    /// A request's choice and call limit: with none, no choice and any number of calls.
    fn parts(tool_choice: Option<Self>) -> (Option<protocol::ToolChoice>, protocol::CallLimit) {
        tool_choice.map_or((None, protocol::CallLimit::Any), |c| {
            (Some(c.choice), c.limit)
        })
    }
}

impl From<protocol::ToolChoice> for AnthropicToolChoice {
    fn from(choice: protocol::ToolChoice) -> Self {
        Self {
            choice,
            limit: protocol::CallLimit::Any,
        }
    }
}

impl Serialize for AnthropicToolChoice {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut form = self.choice.to_anthropic();
        if self.limit == protocol::CallLimit::One {
            form["disable_parallel_tool_use"] = Value::Bool(true);
        }
        form.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AnthropicToolChoice {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let form = Value::deserialize(deserializer)?;
        let choice =
            protocol::ToolChoice::from_anthropic(&form).map_err(serde::de::Error::custom)?;
        let limit = match form.get("disable_parallel_tool_use") {
            None | Some(Value::Bool(false)) => protocol::CallLimit::Any,
            Some(Value::Bool(true)) => protocol::CallLimit::One,
            Some(other) => {
                return Err(serde::de::Error::custom(format!(
                    "tool_choice's disable_parallel_tool_use must be a boolean, not {other}"
                )));
            }
        };
        Ok(Self { choice, limit })
    }
}

fn convert_request(req: MessagesRequest) -> protocol::ChatCompletionRequest {
    let mut messages = Vec::new();

    // ⭐ THE CLIENT'S LAST PROMPT-CACHE BREAKPOINT, marked at the end of its block's text
    // (`CACHE_BREAKPOINT`): the chat path takes the mark out and keeps the prompt length that
    // reaches it (`EngineCoreRequest::cache_breakpoints`), where a model with recurrent state
    // snapshots it. Claude Code's permission classifier inserts each new transcript entry before a
    // fixed tail, so its next request shares the prompt only through the breakpoint before that
    // tail — never through the prompt's end, where every request is snapshotted.
    let breakpoint = last_breakpoint(&req);
    let mark = move |mut text: String, at: Breakpoint| {
        if breakpoint == Some(at) {
            text.push(crate::engine::CACHE_BREAKPOINT);
        }
        text
    };

    // ⭐ ONE SYSTEM MESSAGE, FIRST, AND EVERY OTHER MESSAGE WHERE IT IS. Chat templates take a system
    // message only at the start (Qwen3.5/3.6's raises "System message must be at the beginning"),
    // so the `system` parameter — and any `system` message before the conversation — is that one.
    // A `system` message INSIDE the conversation (Claude Code's environment block, and the reminder
    // it adds every turn) stays at its place, joining the user or tool message before it, as Claude
    // Code itself puts its reminders inside tool results, or opening a user turn of its own.
    //
    // ⛔ HOISTED INTO THE SYSTEM MESSAGE, EACH TURN'S REMINDER CHANGED THE PROMPT WHERE THE SYSTEM
    // MESSAGE ENDS — ahead of the whole conversation — so no turn's prefix cache reached past it:
    // measured under Claude Code on Qwen3.6, every turn re-prefilled everything after the ~17k
    // tokens of system prompt and tools. In place, each turn extends the one before it.
    let system_text = |i: usize, content: &MessageContent| match content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::Blocks(blocks) => (blocks.iter().enumerate())
            .filter_map(|(j, b)| match b {
                ContentBlock::Text { text, .. } => {
                    Some(mark(text.clone(), Breakpoint::Message(i, j)))
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    };
    let leading = (req.messages.iter())
        .take_while(|m| m.role == AnthropicRole::System)
        .count();
    let system: Vec<String> = match req.system {
        None => Vec::new(),
        Some(SystemParam::Text(t)) => vec![t],
        Some(SystemParam::Blocks(blocks)) => (blocks.into_iter().enumerate())
            .map(|(j, b)| mark(b.text, Breakpoint::System(j)))
            .collect(),
    }
    .into_iter()
    .chain(
        req.messages[..leading]
            .iter()
            .enumerate()
            .map(|(i, m)| system_text(i, &m.content)),
    )
    .collect();
    if !system.is_empty() {
        messages.push(protocol::ChatCompletionMessageParam {
            role: "system".to_string(),
            content: Some(Value::String(system.join("\n"))),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
        });
    }

    // Convert messages
    for (i, msg) in req.messages.into_iter().enumerate().skip(leading) {
        if msg.role == AnthropicRole::System {
            let text = system_text(i, &msg.content);
            match messages.last_mut() {
                Some(protocol::ChatCompletionMessageParam {
                    role,
                    content: Some(Value::String(before)),
                    ..
                }) if role == "user" || role == "tool" => {
                    before.push('\n');
                    before.push_str(&text);
                }
                _ => messages.push(protocol::ChatCompletionMessageParam {
                    role: "user".to_string(),
                    content: Some(Value::String(text)),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning: None,
                }),
            }
            continue;
        }
        match msg.content {
            MessageContent::Text(text) => {
                messages.push(protocol::ChatCompletionMessageParam {
                    role: msg.role.chat_role().to_string(),
                    content: Some(Value::String(text)),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning: None,
                });
            }
            MessageContent::Blocks(blocks) => {
                let mut text_parts = Vec::new();
                let mut thinking_parts = Vec::new();
                let mut tool_calls = Vec::new();
                let mut tool_results: Vec<(String, String)> = Vec::new();

                for (j, block) in blocks.into_iter().enumerate() {
                    match block {
                        ContentBlock::Text { text, .. } => {
                            text_parts.push(mark(text, Breakpoint::Message(i, j)))
                        }
                        ContentBlock::Image { .. } => {}
                        ContentBlock::Thinking { thinking, .. } => thinking_parts.push(thinking),
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
                            tool_results.push((tool_use_id, mark(text, Breakpoint::Message(i, j))));
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
                        reasoning: None,
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
                        role: msg.role.chat_role().to_string(),
                        content,
                        name: None,
                        tool_calls: tc,
                        tool_call_id: None,
                        reasoning: (!thinking_parts.is_empty()).then(|| thinking_parts.join("\n")),
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

    let (tool_choice, limit) = AnthropicToolChoice::parts(req.tool_choice);

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
        parallel_tool_calls: (limit == protocol::CallLimit::One).then_some(false),
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
        include_reasoning: !req
            .thinking
            .as_ref()
            .and_then(|t| t.get("type"))
            .and_then(|t| t.as_str())
            .is_some_and(|t| t == "disabled"),
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

            // Thinking comes first, mirroring the streamed block order.
            if let Some(thinking) = c.message.reasoning.filter(|t| !t.is_empty()) {
                blocks.push(ResponseContentBlock::Thinking {
                    thinking,
                    signature: String::new(),
                });
            }

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
        usage: AnthropicUsage::from_usage(&resp.usage),
    }
}

// ---------------------------------------------------------------------------
// Spans mode: each tool a relocatable Plus span
//
// EXPERIMENTAL and KNOWN BROKEN. Gated on the off-by-default `tool-spans`
// feature: `render_tool` writes its own plain text rather than going through
// the model's chat template, so the model is never put in its native
// tool-calling format, the tool parser never fires, and a request that asked
// for a `tool_use` gets prose. Tracked, with measurements, in
// https://github.com/AI-native-Systems-Research/scratchy/issues/193.
//
// The idea: tool schemas are a position-independent *set*, so each tool can be
// its own independently-cacheable relocatable span
// (`Cross([system, Plus([tool₁…toolₙ]), conversation])`) with a block-diagonal
// attention bound, instead of one ordered prefix that any edit invalidates.
// ---------------------------------------------------------------------------

/// Whether this request is served as per-tool relocatable spans.
#[cfg(feature = "tool-spans")]
fn spans_enabled(req: &MessagesRequest) -> bool {
    req.tools.as_ref().is_some_and(|t| !t.is_empty())
}

/// Render one tool definition as the text of its relocatable span.
#[cfg(feature = "tool-spans")]
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
#[cfg(feature = "tool-spans")]
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
#[cfg(feature = "tool-spans")]
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
/// `tool_use` blocks, under the request's `tool_choice` and call limit through the chat path's
/// own rule ([`crate::engine::reply_tool_calls`]).
#[cfg(feature = "tool-spans")]
fn completion_to_messages_response(
    resp: protocol::CompletionResponse,
    parser: Option<&std::sync::Arc<dyn crate::tool_parser::ToolCallParser>>,
    tool_choice: &Option<protocol::ToolChoice>,
    limit: protocol::CallLimit,
) -> MessagesResponse {
    let text = resp
        .choices
        .into_iter()
        .next()
        .map(|c| c.text)
        .unwrap_or_default();

    let (content, stop_reason) = match parser {
        Some(p) => {
            let reply = crate::engine::reply_tool_calls(p.as_ref(), text, tool_choice, limit);
            let mut blocks = Vec::new();
            if let Some(c) = reply.content.filter(|s| !s.is_empty()) {
                blocks.push(ResponseContentBlock::Text { text: c });
            }
            let stop = match reply.calls {
                Some(_) => "tool_use",
                None => "end_turn",
            };
            for tc in reply.calls.into_iter().flatten() {
                let input: Value = serde_json::from_str(&tc.function.arguments)
                    .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
                blocks.push(ResponseContentBlock::ToolUse {
                    id: tc.id,
                    name: tc.function.name,
                    input,
                });
            }
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
        usage: AnthropicUsage::from_usage(&resp.usage),
    }
}

/// Replay a buffered [`MessagesResponse`] (from the spans path) as the Anthropic
/// streaming SSE sequence so `stream:true` clients (Claude Code) get a valid
/// event stream. The prefill — the part spans accelerates — already happened;
/// the decode is small, so buffering it before replaying is fine.
#[cfg(feature = "tool-spans")]
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
                "usage": {
                    "input_tokens": msg.usage.input_tokens,
                    "output_tokens": 0,
                    "cache_read_input_tokens": msg.usage.cache_read_input_tokens
                }
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
            ResponseContentBlock::Thinking {
                thinking,
                signature,
            } => {
                events.push((
                    "content_block_start",
                    serde_json::json!({
                        "type": "content_block_start",
                        "index": i,
                        "content_block": { "type": "thinking", "thinking": "", "signature": "" }
                    }),
                ));
                events.push((
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": i,
                        "delta": { "type": "thinking_delta", "thinking": thinking }
                    }),
                ));
                events.push((
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": i,
                        "delta": { "type": "signature_delta", "signature": signature }
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

/// A [`ServeError`] as the Anthropic Messages API answers one, in its status and shape
/// (`{"type": "error", "error": {"type": …, "message": …}}`): what an Anthropic client, Claude Code
/// among them, reads an error by. `/v1/messages` answers every error this way, a body it can't
/// read (#787) included.
struct AnthropicError(ServeError);

impl IntoResponse for AnthropicError {
    fn into_response(self) -> Response {
        let kind = match &self.0 {
            ServeError::Validation(_) => "invalid_request_error",
            ServeError::PayloadTooLarge(_) => "request_too_large",
            ServeError::ModelNotFound(_) | ServeError::RequestNotFound(_) => "not_found_error",
            ServeError::Engine(_) | ServeError::EngineShutdown | ServeError::Internal(_) => {
                "api_error"
            }
        };
        let body = serde_json::json!({
            "type": "error",
            "error": { "type": kind, "message": self.0.to_string() },
        });
        (self.0.status_code(), Json(body)).into_response()
    }
}

/// POST /v1/messages — Anthropic Messages API.
pub async fn messages(
    State(state): State<Arc<AppState>>,
    body: Result<ApiJson<MessagesRequest>, ServeError>,
) -> Response {
    let request = match body {
        Ok(ApiJson(request)) => request,
        Err(e) => return AnthropicError(e).into_response(),
    };
    info!(
        "POST /v1/messages: model={:?}, max_tokens={}, stream={}",
        request.model, request.max_tokens, request.stream
    );

    let is_stream = request.stream;

    // Spans mode: render a tools-bearing request as per-tool relocatable
    // spans. The buffered completion is tool-parsed and returned as JSON or,
    // for `stream:true`, replayed as Anthropic SSE so streaming clients
    // (Claude Code) work too.
    //
    // Compiled out unless the build names the experimental `tool-spans`
    // feature — see the spans section above for why it is not shipped.
    #[cfg(feature = "tool-spans")]
    if spans_enabled(&request) {
        let (tool_choice, limit) = AnthropicToolChoice::parts(request.tool_choice.clone());
        let tools: Vec<&str> = (request.tools.iter().flatten())
            .map(|t| t.name.as_str())
            .collect();
        if let Err(e) = crate::engine::check_tool_choice(&tool_choice, &tools) {
            return AnthropicError(e).into_response();
        }
        let spnl = build_spnl_query(&request);
        match crate::query::anthropic_spans_completion(&state, &spnl).await {
            Ok(resp) => {
                let parser = state.engine.tool_parser();
                let msg =
                    completion_to_messages_response(resp, parser.as_ref(), &tool_choice, limit);
                return if is_stream {
                    stream_buffered_messages_response(msg).into_response()
                } else {
                    Json(msg).into_response()
                };
            }
            Err(e) => return AnthropicError(e).into_response(),
        }
    }

    let mut chat_request = convert_request(request);

    if is_stream {
        chat_request.stream = true;
        match state.engine.chat_completion_stream(chat_request).await {
            Ok((request_id, model, rx)) => {
                stream_messages_response(request_id, model, rx).into_response()
            }
            Err(e) => AnthropicError(e).into_response(),
        }
    } else {
        match state.engine.chat_completion(chat_request).await {
            Ok(response) => {
                let anthropic_response = convert_response(response);
                Json(anthropic_response).into_response()
            }
            Err(e) => AnthropicError(e).into_response(),
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
    /// An extended-thinking block. The signature is synthetic (empty): the
    /// model is not Anthropic's, so there is no cryptographic signature to
    /// carry — the field exists because clients round-trip it.
    Thinking,
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
    /// Uncached prompt tokens (Anthropic `input_tokens`) — see [`AnthropicUsage`].
    input_tokens: u32,
    /// Prompt tokens served from the prefix cache.
    cache_read_input_tokens: u32,
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
            input_tokens: 0,
            cache_read_input_tokens: 0,
            saw_tool_use: false,
            open: None,
            next_index: 0,
        }
    }

    /// Close whatever block is currently open, if any.
    fn close_open(&mut self, events: &mut Vec<(&'static str, Value)>) {
        if let Some(b) = self.open.take() {
            // A thinking block ends with its (here synthetic, empty)
            // signature delta before the stop event.
            if matches!(b.kind, BlockKind::Thinking) {
                events.push((
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": b.index,
                        "delta": { "type": "signature_delta", "signature": "" }
                    }),
                ));
            }
            events.push((
                "content_block_stop",
                serde_json::json!({ "type": "content_block_stop", "index": b.index }),
            ));
        }
    }

    /// Process one delta, returning the ordered `(event_name, payload)` pairs to emit.
    fn push(&mut self, delta: &StreamDelta) -> Vec<(&'static str, Value)> {
        let mut events = Vec::new();

        // Usage, refreshed from every delta (last write wins) so a cache count
        // that lands a step late still corrects itself. The output count is the
        // engine's running total, not a sum of `new_token_ids`: the steps a
        // tool parser buffers send no delta, so a sum drops most of a tool
        // call's tokens. The prompt side is split here rather than passed in,
        // because `message_start` below is emitted on the first delta and must
        // already carry the real numbers — this is the only path Claude Code
        // exercises, and it reported a literal 0.
        self.output_tokens = delta.usage.completion_tokens;
        self.cache_read_input_tokens = delta.usage.cached_tokens;
        self.input_tokens = delta
            .usage
            .prompt_tokens
            .saturating_sub(delta.usage.cached_tokens);

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
                        "usage": {
                            "input_tokens": self.input_tokens,
                            "output_tokens": 0,
                            "cache_read_input_tokens": self.cache_read_input_tokens
                        }
                    }
                }),
            ));
            events.push(("ping", serde_json::json!({ "type": "ping" })));
        }

        // Reasoning rides on the same StreamDelta as the content split out of
        // it, so handle it before (and independently of) the text/tool arms:
        // a thinking block opened here is closed by whichever arm runs next.
        if let Some(reasoning) = delta.reasoning.as_ref().filter(|r| !r.is_empty()) {
            if !matches!(
                self.open.as_ref().map(|b| b.kind),
                Some(BlockKind::Thinking)
            ) {
                self.close_open(&mut events);
                let bidx = self.next_index;
                self.next_index += 1;
                events.push((
                    "content_block_start",
                    serde_json::json!({
                        "type": "content_block_start",
                        "index": bidx,
                        "content_block": { "type": "thinking", "thinking": "", "signature": "" }
                    }),
                ));
                self.open = Some(OpenBlock {
                    index: bidx,
                    kind: BlockKind::Thinking,
                });
            }
            let bidx = self.open.as_ref().map(|b| b.index).unwrap_or(0);
            events.push((
                "content_block_delta",
                serde_json::json!({
                    "type": "content_block_delta",
                    "index": bidx,
                    "delta": { "type": "thinking_delta", "thinking": reasoning }
                }),
            ));
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
            // Only a call the model finished is `tool_use`: one cut off by `max_tokens` stays
            // `max_tokens`, so no client runs a call whose input may be incomplete.
            let stopped = delta.finish_reason == Some(scratchy_core_common::FinishReason::Stop);
            let stop_reason = if self.saw_tool_use && stopped {
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
                    // Input-side fields repeated here as well as in
                    // `message_start`: this is where ollama reports them, so one
                    // client reads the same place on both engines. Anthropic's
                    // `message_delta.usage` is cumulative-final, not additive,
                    // so restating the same values is a no-op for a conforming
                    // client.
                    "usage": {
                        "output_tokens": self.output_tokens,
                        "input_tokens": self.input_tokens,
                        "cache_read_input_tokens": self.cache_read_input_tokens
                    }
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

    /// Only a tools-bearing request takes the span path.
    #[cfg(feature = "tool-spans")]
    #[test]
    fn test_spans_only_for_tools_bearing_requests() {
        let with_tools: MessagesRequest = serde_json::from_str(
            r#"{
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "List the files"}],
            "tools": [{
                "name": "Bash",
                "description": "Run a command",
                "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}
            }]
        }"#,
        )
        .unwrap();
        assert!(spans_enabled(&with_tools));

        let req: MessagesRequest = serde_json::from_str(
            r#"{"max_tokens": 100, "messages": [{"role": "user", "content": "Hello"}]}"#,
        )
        .unwrap();
        assert!(!spans_enabled(&req));

        // An empty `tools` array is not a tools-bearing request either.
        let mut empty_tools = req;
        empty_tools.tools = Some(vec![]);
        assert!(!spans_enabled(&empty_tools));
    }

    /// The spans path's reply follows its `tool_choice` as the chat path's does (#758): a forced
    /// tool drops another's call, leaving the text around it and `end_turn`; `"none"` reads no
    /// call, so the text comes back as written; the forced tool's own call is a `tool_use` block.
    #[cfg(feature = "tool-spans")]
    #[test]
    fn the_spans_reply_follows_its_tool_choice() {
        use protocol::ToolChoice;
        let parser = crate::tool_parser::get_tool_parser("hermes").unwrap();
        let text = "Let me look.<tool_call>\n{\"name\": \"Bash\", \"arguments\": \
                    {\"command\": \"ls\"}}\n</tool_call>";
        let reply = |choice: ToolChoice| {
            let choices = vec![protocol::CompletionResponseChoice {
                index: 0,
                text: text.to_string(),
                logprobs: None,
                finish_reason: Some("stop".to_string()),
                stop_reason: None,
                prompt_logprobs: None,
            }];
            let resp = protocol::CompletionResponse::new("m".into(), choices, Default::default());
            let any = protocol::CallLimit::Any;
            let msg = completion_to_messages_response(resp, Some(&parser), &Some(choice), any);
            let blocks: Vec<String> = (msg.content.iter())
                .map(|b| match b {
                    ResponseContentBlock::Text { text } => format!("text {text}"),
                    ResponseContentBlock::ToolUse { name, .. } => format!("tool_use {name}"),
                    other => format!("{other:?}"),
                })
                .collect();
            (blocks, msg.stop_reason)
        };
        let end_turn = Some("end_turn".to_string());
        let looked = "text Let me look.".to_string();
        assert_eq!(
            reply(ToolChoice::Function("Read".into())),
            (vec![looked.clone()], end_turn.clone())
        );
        assert_eq!(
            reply(ToolChoice::None),
            (vec![format!("text {text}")], end_turn)
        );
        let bash = (
            vec![looked, "tool_use Bash".to_string()],
            Some("tool_use".into()),
        );
        assert_eq!(reply(ToolChoice::Function("Bash".into())), bash);
        assert_eq!(reply(ToolChoice::Auto), bash);
    }

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
                role: AnthropicRole::User,
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
            thinking: None,
        };
        let chat = convert_request(req);
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "user");
        assert_eq!(chat.max_tokens, Some(100));
        assert_eq!(chat.temperature, Some(0.7));
    }

    /// Claude Code 2.1's first request: the `system` blocks, then a user turn and a `system`
    /// message (its environment block). The chat request carries one system message, first — the
    /// parameter's text — and the environment block where it was, in the user turn before it.
    #[test]
    fn test_convert_system_messages_in_the_conversation() {
        let req: MessagesRequest = serde_json::from_value(serde_json::json!({
            "max_tokens": 50,
            "system": [{"type": "text", "text": "You are Claude Code."}, {"type": "text", "text": "Be concise."}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Read stack.py."}]},
                {"role": "system", "content": [{"type": "text", "text": "# Environment"}]}
            ]
        }))
        .expect("parses");
        let chat = convert_request(req);
        let roles: Vec<&str> = chat.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "user"]);
        assert_eq!(
            chat.messages[0].content,
            Some(Value::String("You are Claude Code.\nBe concise.".into()))
        );
        assert_eq!(
            chat.messages[1].content,
            Some(Value::String("Read stack.py.\n# Environment".into()))
        );
    }

    /// Claude Code's turns as it sends them: each repeats the last, then adds the assistant's tool
    /// call, its result and a `system` reminder. Converted, each turn's messages extend the turn
    /// before's exactly — what a prefix cache needs to reuse the conversation — and no system
    /// message follows the first, which chat templates reject.
    #[test]
    fn test_a_claude_code_turn_extends_the_one_before_it() {
        let turn = |n: usize| -> MessagesRequest {
            let mut messages = vec![
                serde_json::json!({"role": "user", "content": [{"type": "text", "text": "Read stack.py."}]}),
                serde_json::json!({"role": "system", "content": [{"type": "text", "text": "# Environment"}]}),
            ];
            for i in 0..n {
                messages.push(serde_json::json!({"role": "assistant", "content": [
                    {"type": "tool_use", "id": format!("t{i}"), "name": "Read", "input": {"path": "stack.py"}}
                ]}));
                messages.push(serde_json::json!({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": format!("t{i}"), "content": format!("line {i}")}
                ]}));
                messages.push(serde_json::json!({"role": "system", "content": format!("<total_tokens>{} tokens left</total_tokens>", 1000 - i)}));
            }
            serde_json::from_value(serde_json::json!({
                "max_tokens": 50,
                "system": [{"type": "text", "text": "You are Claude Code."}],
                "messages": messages,
            }))
            .expect("parses")
        };
        let converted = |n: usize| -> Vec<Value> {
            let chat = convert_request(turn(n));
            assert!(
                chat.messages.iter().skip(1).all(|m| m.role != "system"),
                "a system message only first"
            );
            (chat.messages.iter())
                .map(|m| serde_json::to_value(m).expect("serializes"))
                .collect()
        };
        for n in 0..3 {
            let (before, after) = (converted(n), converted(n + 1));
            assert_eq!(
                after[..before.len()],
                before[..],
                "turn {} extends turn {n}",
                n + 1
            );
        }
        let last = convert_request(turn(1));
        assert_eq!(
            (last.messages[3].role.as_str(), &last.messages[3].content),
            (
                "tool",
                &Some(Value::String(
                    "line 0\n<total_tokens>1000 tokens left</total_tokens>".into()
                ))
            ),
            "the reminder joins the tool result before it"
        );
    }

    /// Claude Code's permission classifier marks its transcript entries as cache breakpoints, then
    /// closes the transcript and adds its instructions. Only the LAST breakpoint is marked in the
    /// converted text, at the end of its entry, ahead of the tail every next call re-sends after a
    /// new entry.
    #[test]
    fn test_the_last_cache_breakpoint_is_marked_at_its_end() {
        let req: MessagesRequest = serde_json::from_value(serde_json::json!({
            "max_tokens": 50,
            "system": [{"type": "text", "text": "Classify.", "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "<transcript>"},
                {"type": "text", "text": "entry 1", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "entry 2", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "</transcript>"},
                {"type": "text", "text": "Answer <block> or <allow>."}
            ]}]
        }))
        .expect("parses");
        let chat = convert_request(req);
        let text = |i: usize| {
            chat.messages[i]
                .content
                .as_ref()
                .and_then(Value::as_str)
                .unwrap()
        };
        let mark = crate::engine::CACHE_BREAKPOINT;
        assert_eq!(
            text(0),
            "Classify.",
            "the system breakpoint is not the last"
        );
        assert_eq!(
            text(1),
            format!(
                "<transcript>\nentry 1\nentry 2{mark}\n</transcript>\nAnswer <block> or <allow>."
            )
        );
    }

    /// A `system` message after an assistant turn opens a user turn of its own, and one before
    /// the conversation joins the system message.
    #[test]
    fn test_a_system_message_after_the_assistant_opens_a_user_turn() {
        let req: MessagesRequest = serde_json::from_value(serde_json::json!({
            "max_tokens": 50,
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello."},
                {"role": "system", "content": "The user is away."}
            ]
        }))
        .expect("parses");
        let chat = convert_request(req);
        let got: Vec<(&str, &Option<Value>)> = (chat.messages.iter())
            .map(|m| (m.role.as_str(), &m.content))
            .collect();
        let text = |t: &str| Some(Value::String(t.into()));
        assert_eq!(
            got,
            [
                ("system", &text("Be concise.")),
                ("user", &text("Hi")),
                ("assistant", &text("Hello.")),
                ("user", &text("The user is away.")),
            ]
        );
    }

    #[test]
    fn test_convert_system_message() {
        let req = MessagesRequest {
            model: None,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
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
            thinking: None,
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
                role: AnthropicRole::User,
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
            thinking: None,
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
                role: AnthropicRole::Assistant,
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
            thinking: None,
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
                role: AnthropicRole::User,
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
            thinking: None,
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
                role: AnthropicRole::User,
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
                input_schema: serde_json::json!({"type": "object"}).into(),
                cache_control: None,
            }]),
            tool_choice: None,
            metadata: None,
            thinking: None,
        };
        let chat = convert_request(req);
        let tools = chat.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "search");
        assert_eq!(tools[0].tool_type, "function");
    }

    /// `{"type": "any"}` is `"required"`, and its `disable_parallel_tool_use: true` is
    /// `parallel_tool_calls: false` (#794).
    #[test]
    fn test_convert_tool_choice_any() {
        let req = |limit| MessagesRequest {
            model: None,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
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
            tool_choice: Some(AnthropicToolChoice {
                choice: protocol::ToolChoice::Required,
                limit,
            }),
            metadata: None,
            thinking: None,
        };
        let chat = convert_request(req(protocol::CallLimit::Any));
        assert_eq!(chat.tool_choice, Some(protocol::ToolChoice::Required));
        assert_eq!(chat.parallel_tool_calls, None);
        let chat = convert_request(req(protocol::CallLimit::One));
        assert_eq!(chat.tool_choice, Some(protocol::ToolChoice::Required));
        assert_eq!(chat.parallel_tool_calls, Some(false));
    }

    /// Anthropic's object forms read as its bare strings do, a forced tool as the named function,
    /// and `disable_parallel_tool_use: true` as one call. A form Anthropic doesn't define fails the
    /// request as it is parsed, as one naming no tool does, and so does a
    /// `disable_parallel_tool_use` that isn't a boolean.
    #[test]
    fn test_anthropic_tool_choice_forms() {
        use protocol::{CallLimit, ToolChoice};
        use serde_json::json;
        let any = AnthropicToolChoice::from;
        let one = |choice| AnthropicToolChoice {
            choice,
            limit: CallLimit::One,
        };
        let cases = [
            (json!({"type": "auto"}), any(ToolChoice::Auto)),
            (json!({"type": "any"}), any(ToolChoice::Required)),
            (json!({"type": "none"}), any(ToolChoice::None)),
            (
                json!({"type": "auto", "disable_parallel_tool_use": true}),
                one(ToolChoice::Auto),
            ),
            (
                json!({"type": "any", "disable_parallel_tool_use": true}),
                one(ToolChoice::Required),
            ),
            (
                json!({"type": "auto", "disable_parallel_tool_use": false}),
                any(ToolChoice::Auto),
            ),
            (
                json!({"type": "tool", "name": "Read"}),
                any(ToolChoice::Function("Read".into())),
            ),
            (json!("auto"), any(ToolChoice::Auto)),
            (json!("any"), any(ToolChoice::Required)),
            (json!("none"), any(ToolChoice::None)),
        ];
        for (anthropic, choice) in cases {
            let read = serde_json::from_value::<AnthropicToolChoice>(anthropic.clone());
            assert_eq!(read.ok().as_ref(), Some(&choice), "{anthropic}");
            // Written in Anthropic's form, so what a client posts reads back as itself.
            let written = serde_json::to_value(&choice).unwrap();
            let reread = serde_json::from_value::<AnthropicToolChoice>(written.clone());
            assert_eq!(reread.ok(), Some(choice), "{written}");
        }
        for bad in [
            json!({"type": "mystery"}),
            json!({"type": "tool"}),
            json!("required"),
            json!(3),
        ] {
            let err = serde_json::from_value::<AnthropicToolChoice>(bad.clone())
                .expect_err("not an Anthropic tool_choice");
            assert!(
                err.to_string().contains(r#"{"type": "tool", "name""#),
                "{bad}: {err}"
            );
        }
        let not_bool = json!({"type": "auto", "disable_parallel_tool_use": "yes"});
        let err = serde_json::from_value::<AnthropicToolChoice>(not_bool).unwrap_err();
        assert!(err.to_string().contains("must be a boolean"), "{err}");
        let request = json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 50,
            "tool_choice": {"type": "mystery"},
        });
        assert!(serde_json::from_value::<MessagesRequest>(request).is_err());
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
                cache_read_input_tokens: 0,
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

    use crate::engine::StreamUsage;
    use crate::tool_parser::DeltaToolCall;
    use scratchy_core_common::FinishReason;

    fn mk_delta(
        text: Option<&str>,
        tools: Option<Vec<DeltaToolCall>>,
        finish: Option<FinishReason>,
    ) -> StreamDelta {
        mk_delta_usage(text, tools, finish, StreamUsage::default())
    }

    /// Same, with explicit prompt-side counts for the usage assertions.
    fn mk_delta_usage(
        text: Option<&str>,
        tools: Option<Vec<DeltaToolCall>>,
        finish: Option<FinishReason>,
        usage: StreamUsage,
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
            usage,
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

    /// A tool call cut off by `max_tokens` stops `max_tokens`, not `tool_use`: its input may be
    /// incomplete, so no client should run it.
    #[test]
    fn test_stream_tool_use_cut_off_stops_max_tokens() {
        let mut enc = AnthropicSseEncoder::new("req1".into(), "m".into());
        let mut events = Vec::new();
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
        events.extend(enc.push(&mk_delta(None, None, Some(FinishReason::Length))));
        let message_delta = events.iter().find(|(name, _)| *name == "message_delta");
        assert_eq!(
            message_delta.unwrap().1["delta"]["stop_reason"],
            "max_tokens"
        );
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

    // -----------------------------------------------------------------------
    // Usage: the OpenAI -> Anthropic input-side split
    // -----------------------------------------------------------------------

    fn usage_with_cache(prompt: u32, cached: Option<u32>) -> protocol::UsageInfo {
        protocol::UsageInfo {
            prompt_tokens: prompt,
            total_tokens: prompt,
            completion_tokens: Some(0),
            prompt_tokens_details: cached.map(|c| protocol::PromptTokenUsageInfo {
                cached_tokens: Some(c),
            }),
        }
    }

    /// The headline case, with the numbers measured on ollama 0.40.0 for the
    /// same request sent twice (issue #298): 167 prompt tokens, 166 of them
    /// served from cache on the repeat. Both engines must put the same split in
    /// the same fields, or the benchmark compares different quantities.
    #[test]
    fn test_usage_splits_cached_prefix_like_ollama() {
        let u = AnthropicUsage::from_usage(&usage_with_cache(167, Some(166)));
        assert_eq!(u.input_tokens, 1);
        assert_eq!(u.cache_read_input_tokens, 166);
        // Anthropic's invariant: the input side sums to the whole prompt.
        assert_eq!(u.input_tokens + u.cache_read_input_tokens, 167);

        // And the metric that split exists to serve.
        let reprefill = u.input_tokens as f64 / (u.input_tokens + u.cache_read_input_tokens) as f64;
        assert!(
            reprefill < 0.01,
            "a 166/167 cache hit must read as near-zero reprefill, got {reprefill}"
        );
    }

    /// A cache miss must leave `input_tokens` exactly as it was before this
    /// change — the whole prompt — so nothing regresses when the cache is cold.
    #[test]
    fn test_usage_cache_miss_leaves_input_tokens_whole() {
        let u = AnthropicUsage::from_usage(&usage_with_cache(167, None));
        assert_eq!(u.input_tokens, 167);
        assert_eq!(u.cache_read_input_tokens, 0);
    }

    /// `engine.rs` sums `num_cached_tokens` across n children against a
    /// per-prompt `prompt_tokens`, so cached > prompt is representable upstream.
    /// Unreachable via `/v1/messages` (`convert_request` pins `n: 1`), but the
    /// split must not wrap if it ever becomes reachable.
    #[test]
    fn test_usage_cached_exceeding_prompt_does_not_wrap() {
        let u = AnthropicUsage::from_usage(&usage_with_cache(200, Some(300)));
        assert_eq!(u.input_tokens, 0);
        assert_eq!(u.cache_read_input_tokens, 300);
    }

    /// The regression guard for the path Claude Code actually uses: a streaming
    /// request used to report a literal `"input_tokens": 0` in `message_start`
    /// and nothing at all about the cache.
    #[test]
    fn test_stream_reports_prompt_and_cached_tokens() {
        let mut enc = AnthropicSseEncoder::new("r".into(), "m".into());
        let usage = StreamUsage {
            prompt: crate::engine::PromptIndex(0),
            prompt_tokens: 1000,
            cached_tokens: 960,
            completion_tokens: 2,
        };
        let mut events = Vec::new();
        events.extend(enc.push(&mk_delta_usage(Some("hi"), None, None, usage)));
        events.extend(enc.push(&mk_delta_usage(None, None, Some(FinishReason::Stop), usage)));

        let start = &events[0].1["message"]["usage"];
        assert_eq!(start["input_tokens"], 40, "1000 prompt - 960 cached");
        assert_eq!(start["cache_read_input_tokens"], 960);

        // Also on message_delta, where ollama reports it.
        let delta = &events
            .iter()
            .find(|(n, _)| *n == "message_delta")
            .expect("message_delta")
            .1["usage"];
        assert_eq!(delta["input_tokens"], 40);
        assert_eq!(delta["cache_read_input_tokens"], 960);
        assert_eq!(delta["output_tokens"], 2);
    }

    /// A stream with no cache hit still reports the real prompt length, rather
    /// than the 0 the encoder used to hardcode.
    #[test]
    fn test_stream_reports_prompt_tokens_without_cache_hit() {
        let mut enc = AnthropicSseEncoder::new("r".into(), "m".into());
        let usage = StreamUsage {
            prompt: crate::engine::PromptIndex(0),
            prompt_tokens: 512,
            cached_tokens: 0,
            completion_tokens: 1,
        };
        let events = enc.push(&mk_delta_usage(Some("hi"), None, None, usage));
        let start = &events[0].1["message"]["usage"];
        assert_eq!(start["input_tokens"], 512);
        assert_eq!(start["cache_read_input_tokens"], 0);
    }

    /// The Claude Code smoke run that found this: `completion_tokens=33` in the server log,
    /// `"output_tokens": 1` on the stream. The steps a tool parser buffers send no delta, so a
    /// sum of the deltas' token ids drops them; `message_delta` reports the engine's count.
    #[test]
    fn test_stream_tool_call_reports_buffered_tokens() {
        for reasoning in [false, true] {
            let (deltas, generated) = crate::engine::test_support::buffered_tool_call_deltas(
                reasoning,
                FinishReason::Stop,
            );
            let mut enc = AnthropicSseEncoder::new("r".into(), "m".into());
            let events: Vec<_> = deltas.iter().flat_map(|d| enc.push(d)).collect();
            assert!(
                events
                    .iter()
                    .any(|(_, e)| e["content_block"]["type"] == "tool_use"),
                "reasoning={reasoning}: a tool_use block"
            );
            let delta = &events
                .iter()
                .find(|(n, _)| *n == "message_delta")
                .expect("message_delta")
                .1;
            assert_eq!(delta["delta"]["stop_reason"], "tool_use");
            assert_eq!(
                delta["usage"]["output_tokens"], generated,
                "reasoning={reasoning}"
            );
        }
    }
}
