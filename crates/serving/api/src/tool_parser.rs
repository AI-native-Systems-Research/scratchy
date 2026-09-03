// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Tool call parsing for model outputs.
//!
//! When models generate text containing tool calls (e.g.
//! `<tool_call>{"name":"search","arguments":{"q":"foo"}}</tool_call>`),
//! these parsers detect and extract them into structured `ToolCall` objects.
//!
//! Supports both non-streaming (full text extraction) and streaming
//! (incremental delta) modes, matching Python vLLM behavior.
//!
//! Port of: `vllm/entrypoints/openai/tool_parsers/` (subset)

use std::sync::Arc;

use uuid::Uuid;

use crate::protocol;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Result of parsing tool calls from complete model output (non-streaming).
#[derive(Debug, Clone)]
pub struct ExtractedToolCallInfo {
    /// Whether any tool calls were found.
    pub tools_called: bool,
    /// Extracted tool calls.
    pub tool_calls: Vec<protocol::ToolCall>,
    /// Text before/outside tool calls (None if empty).
    pub content: Option<String>,
}

/// A streaming tool call delta (mirrors OpenAI DeltaToolCall).
#[derive(Debug, Clone)]
pub struct DeltaToolCall {
    /// Index of this tool call in the array.
    pub index: u32,
    /// Tool call ID (set on first delta for this tool).
    pub id: Option<String>,
    /// Call type ("function"), set on first delta.
    pub call_type: Option<String>,
    /// Function name (set once name is parsed).
    pub function_name: Option<String>,
    /// Argument fragment diff (incremental argument text).
    pub function_arguments: Option<String>,
}

/// A streaming delta that can be either content or tool calls.
#[derive(Debug, Clone)]
pub enum ToolParserDelta {
    /// Regular text content (before tool calls).
    Content(String),
    /// One or more tool call deltas.
    ToolCalls(Vec<DeltaToolCall>),
    /// Skip this token (buffering partial tags).
    None,
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Trait for non-streaming tool call extraction.
pub trait ToolCallParser: Send + Sync {
    /// Extract tool calls from complete model output text.
    fn extract_tool_calls(&self, model_output: &str) -> ExtractedToolCallInfo;

    /// Create a new per-request streaming state machine.
    fn create_streaming_state(&self) -> Box<dyn StreamingToolParserState + Send>;

    /// Whether this parser's delimiters are special/added tokens that the
    /// detokenizer would otherwise strip when `skip_special_tokens` is set
    /// (the default). Parsers whose format is built from plain text (Hermes,
    /// Granite 3.1, …) return `false`; parsers keyed on real control tokens
    /// (Gemma 4's `<|tool_call>` / `<|"|>`) return `true` so the server keeps
    /// those tokens in the text handed to `extract_tool_calls`. Without this,
    /// the delimiters vanish before parsing and no tool call can be recovered.
    fn requires_special_tokens(&self) -> bool {
        false
    }
}

/// Per-request streaming state machine for incremental tool call parsing.
pub trait StreamingToolParserState: Send {
    /// Process a new text delta.
    ///
    /// - `previous_text`: all text before this delta
    /// - `current_text`: all text including this delta (previous_text + delta_text)
    /// - `delta_text`: the new text fragment
    fn process_delta(
        &mut self,
        previous_text: &str,
        current_text: &str,
        delta_text: &str,
    ) -> ToolParserDelta;
}

// ---------------------------------------------------------------------------
// Hermes tool parser
// ---------------------------------------------------------------------------

const HERMES_TOOL_CALL_OPEN: &str = "<tool_call>";
const HERMES_TOOL_CALL_CLOSE: &str = "</tool_call>";

/// Hermes-style tool call parser.
///
/// Detects tool calls wrapped in `<tool_call>...</tool_call>` tags.
/// The content between tags should be JSON with `name` and `arguments` fields.
#[derive(Default)]
pub struct HermesToolParser;

impl HermesToolParser {
    pub fn new() -> Self {
        Self
    }
}

impl ToolCallParser for HermesToolParser {
    fn extract_tool_calls(&self, model_output: &str) -> ExtractedToolCallInfo {
        hermes_extract(model_output)
    }

    fn create_streaming_state(&self) -> Box<dyn StreamingToolParserState + Send> {
        Box::new(HermesStreamingState::new())
    }
}

/// Extract tool calls from Hermes-formatted text.
fn hermes_extract(text: &str) -> ExtractedToolCallInfo {
    let mut tool_calls = Vec::new();
    let mut content_before = String::new();

    // Find text before first <tool_call> tag.
    let first_open = text.find(HERMES_TOOL_CALL_OPEN);
    if let Some(pos) = first_open {
        let before = text[..pos].trim();
        if !before.is_empty() {
            content_before = before.to_string();
        }
    } else {
        // No tool call tags at all.
        return ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(text.to_string()),
        };
    }

    // Extract all <tool_call>...</tool_call> blocks.
    let mut search_start = 0;
    while let Some(open_pos) = text[search_start..].find(HERMES_TOOL_CALL_OPEN) {
        let open_pos = search_start + open_pos;
        let json_start = open_pos + HERMES_TOOL_CALL_OPEN.len();

        // Find closing tag or end of string (unclosed tag).
        let json_end = text[json_start..]
            .find(HERMES_TOOL_CALL_CLOSE)
            .map(|p| json_start + p)
            .unwrap_or(text.len());

        let json_str = text[json_start..json_end].trim();
        if let Some(tc) = parse_tool_call_json(json_str) {
            tool_calls.push(tc);
        }

        search_start = if json_end < text.len() {
            json_end + HERMES_TOOL_CALL_CLOSE.len()
        } else {
            text.len()
        };
    }

    if tool_calls.is_empty() {
        // Tags found but JSON was malformed — fall back to content.
        ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(text.to_string()),
        }
    } else {
        ExtractedToolCallInfo {
            tools_called: true,
            tool_calls,
            content: if content_before.is_empty() {
                None
            } else {
                Some(content_before)
            },
        }
    }
}

/// Parse a JSON string into a ToolCall.
///
/// Expects `{"name": "...", "arguments": {...}}` format.
fn parse_tool_call_json(json_str: &str) -> Option<protocol::ToolCall> {
    let val: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let name = val.get("name")?.as_str()?.to_string();
    let arguments = val.get("arguments")?;
    let arguments_str = if arguments.is_string() {
        arguments.as_str().unwrap().to_string()
    } else {
        serde_json::to_string(arguments).ok()?
    };

    Some(protocol::ToolCall {
        id: format!("call_{}", Uuid::new_v4().simple()),
        call_type: "function".to_string(),
        function: protocol::FunctionCall {
            name,
            arguments: arguments_str,
        },
    })
}

// ---------------------------------------------------------------------------
// Hermes streaming state machine
// ---------------------------------------------------------------------------

struct HermesStreamingState {
    /// Current tool call index (-1 = no tool call started).
    current_tool_id: i32,
    /// Whether the name for the current tool has been sent.
    current_tool_name_sent: bool,
    /// Previously parsed tool call JSON objects.
    prev_tool_call_arr: Vec<serde_json::Value>,
    /// Streamed argument characters for each tool call (for diffing).
    streamed_args_for_tool: Vec<String>,
    /// Buffer for partial tag tokens.
    buffer: String,
    /// Number of tool_call open tags seen so far.
    num_open_tags: usize,
    /// Number of tool_call close tags seen so far.
    num_close_tags: usize,
}

impl HermesStreamingState {
    fn new() -> Self {
        Self {
            current_tool_id: -1,
            current_tool_name_sent: false,
            prev_tool_call_arr: Vec::new(),
            streamed_args_for_tool: Vec::new(),
            buffer: String::new(),
            num_open_tags: 0,
            num_close_tags: 0,
        }
    }
}

impl StreamingToolParserState for HermesStreamingState {
    fn process_delta(
        &mut self,
        _previous_text: &str,
        current_text: &str,
        delta_text: &str,
    ) -> ToolParserDelta {
        // Append delta to buffer.
        self.buffer.push_str(delta_text);

        // Check if buffer contains partial open/close tags that need more tokens.
        if is_partial_tag(&self.buffer) {
            return ToolParserDelta::None;
        }

        // Drain the buffer.
        let text_to_process = std::mem::take(&mut self.buffer);

        // Count open/close tags in the full text so far.
        let new_open_count = current_text.matches(HERMES_TOOL_CALL_OPEN).count();
        let new_close_count = current_text.matches(HERMES_TOOL_CALL_CLOSE).count();

        // If no tool call tags yet, emit as content.
        if new_open_count == 0 {
            return ToolParserDelta::Content(text_to_process);
        }

        // Check if a new tool call just started.
        if new_open_count > self.num_open_tags {
            self.num_open_tags = new_open_count;
            self.current_tool_id += 1;
            self.current_tool_name_sent = false;
            self.prev_tool_call_arr.push(serde_json::Value::Null);
            self.streamed_args_for_tool.push(String::new());

            // The text_to_process might contain text before <tool_call>.
            // If current_tool_id == 0, there might be leading content.
            if self.current_tool_id == 0 {
                // Find content before the first <tool_call> in current_text.
                if let Some(pos) = current_text.find(HERMES_TOOL_CALL_OPEN) {
                    let before = current_text[..pos].to_string();
                    if !before.is_empty() {
                        // We already emitted this as content in previous deltas.
                        // Just skip — the content was already streamed.
                    }
                }
            }
        }

        // Update close tag count.
        self.num_close_tags = new_close_count;

        // Extract the JSON content for the current tool call.
        let tool_idx = self.current_tool_id as usize;

        // Get text after the last <tool_call> open tag.
        let after_last_open = {
            let mut start = 0;
            for _ in 0..self.num_open_tags {
                if let Some(pos) = current_text[start..].find(HERMES_TOOL_CALL_OPEN) {
                    start = start + pos + HERMES_TOOL_CALL_OPEN.len();
                }
            }
            // If there's a close tag, take text up to it.
            if let Some(close_pos) = current_text[start..].find(HERMES_TOOL_CALL_CLOSE) {
                &current_text[start..start + close_pos]
            } else {
                &current_text[start..]
            }
        };

        // Try partial JSON parse of the tool call content.
        let parsed = partial_json_parse(after_last_open.trim());

        if let Some(ref val) = parsed {
            // Try to extract name.
            if !self.current_tool_name_sent
                && let Some(name) = val.get("name").and_then(|n| n.as_str())
            {
                self.current_tool_name_sent = true;
                self.prev_tool_call_arr[tool_idx] = val.clone();

                return ToolParserDelta::ToolCalls(vec![DeltaToolCall {
                    index: tool_idx as u32,
                    id: Some(format!("call_{}", Uuid::new_v4().simple())),
                    call_type: Some("function".to_string()),
                    function_name: Some(name.to_string()),
                    function_arguments: Some(String::new()),
                }]);
            }

            // Stream arguments diff.
            if self.current_tool_name_sent {
                let current_args = val
                    .get("arguments")
                    .map(|a| {
                        if a.is_string() {
                            a.as_str().unwrap().to_string()
                        } else {
                            serde_json::to_string(a).unwrap_or_default()
                        }
                    })
                    .unwrap_or_default();

                let prev_args = &self.streamed_args_for_tool[tool_idx];
                if current_args.len() > prev_args.len() {
                    let diff = current_args[prev_args.len()..].to_string();
                    self.streamed_args_for_tool[tool_idx] = current_args;
                    self.prev_tool_call_arr[tool_idx] = val.clone();

                    return ToolParserDelta::ToolCalls(vec![DeltaToolCall {
                        index: tool_idx as u32,
                        id: None,
                        call_type: None,
                        function_name: None,
                        function_arguments: Some(diff),
                    }]);
                }
            }
        }

        ToolParserDelta::None
    }
}

/// Check if text ends with a partial `<tool_call>` or `</tool_call>` tag.
fn is_partial_tag(text: &str) -> bool {
    // Check suffixes of <tool_call> and </tool_call>.
    for tag in [HERMES_TOOL_CALL_OPEN, HERMES_TOOL_CALL_CLOSE] {
        for i in 1..tag.len() {
            if text.ends_with(&tag[..i]) {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// LLaMA JSON tool parser
// ---------------------------------------------------------------------------

const LLAMA_PYTHON_TAG: &str = "<|python_tag|>";

/// LLaMA-style JSON tool call parser.
///
/// Detects tool calls as raw JSON objects in the output, optionally
/// prefixed with `<|python_tag|>`.
#[derive(Default)]
pub struct LlamaJsonToolParser;

impl LlamaJsonToolParser {
    pub fn new() -> Self {
        Self
    }
}

impl ToolCallParser for LlamaJsonToolParser {
    fn extract_tool_calls(&self, model_output: &str) -> ExtractedToolCallInfo {
        llama_json_extract(model_output)
    }

    fn create_streaming_state(&self) -> Box<dyn StreamingToolParserState + Send> {
        Box::new(LlamaJsonStreamingState::new())
    }
}

/// Extract tool calls from LLaMA JSON-formatted text.
fn llama_json_extract(text: &str) -> ExtractedToolCallInfo {
    // Strip <|python_tag|> if present.
    let text = text
        .strip_prefix(LLAMA_PYTHON_TAG)
        .unwrap_or(text)
        .trim_start();

    // Try to find JSON objects.
    let mut tool_calls = Vec::new();
    let mut content_before = String::new();
    let mut found_first_json = false;

    let mut pos = 0;
    while pos < text.len() {
        if let Some(brace_pos) = text[pos..].find('{') {
            let abs_pos = pos + brace_pos;
            if !found_first_json {
                let before = text[..abs_pos].trim();
                if !before.is_empty() {
                    content_before = before.to_string();
                }
                found_first_json = true;
            }

            // Try to parse a JSON object starting here.
            if let Some((val, end_pos)) = try_parse_json_object(&text[abs_pos..]) {
                if let Some(tc) = json_value_to_tool_call(&val) {
                    tool_calls.push(tc);
                }
                pos = abs_pos + end_pos;
            } else {
                pos = abs_pos + 1;
            }
        } else {
            break;
        }
    }

    if tool_calls.is_empty() {
        ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(text.to_string()),
        }
    } else {
        ExtractedToolCallInfo {
            tools_called: true,
            tool_calls,
            content: if content_before.is_empty() {
                None
            } else {
                Some(content_before)
            },
        }
    }
}

/// Try to parse a JSON object from the start of `text`.
/// Returns (value, bytes_consumed) on success.
fn try_parse_json_object(text: &str) -> Option<(serde_json::Value, usize)> {
    // Use serde's streaming deserializer to find where the JSON ends.
    let mut de = serde_json::Deserializer::from_str(text).into_iter::<serde_json::Value>();
    if let Some(Ok(val)) = de.next() {
        let end = de.byte_offset();
        if val.is_object() {
            return Some((val, end));
        }
    }
    None
}

/// Convert a JSON value with "name" and "arguments" into a ToolCall.
fn json_value_to_tool_call(val: &serde_json::Value) -> Option<protocol::ToolCall> {
    let name = val.get("name")?.as_str()?.to_string();
    // Accept "parameters" as an alias when "arguments" is absent or null:
    // Llama 3.x tool calls use `{"name", "parameters"}` with no "arguments" key.
    // (Previously `val.get("arguments")?` bailed here, making the fallback dead.)
    let arguments = match val.get("arguments") {
        Some(v) if !v.is_null() => v,
        _ => val.get("parameters")?,
    };

    let arguments_str = if arguments.is_string() {
        arguments.as_str().unwrap().to_string()
    } else {
        serde_json::to_string(arguments).ok()?
    };

    Some(protocol::ToolCall {
        id: format!("call_{}", Uuid::new_v4().simple()),
        call_type: "function".to_string(),
        function: protocol::FunctionCall {
            name,
            arguments: arguments_str,
        },
    })
}

// ---------------------------------------------------------------------------
// LLaMA JSON streaming state machine
// ---------------------------------------------------------------------------

struct LlamaJsonStreamingState {
    /// Current tool call index (-1 = no tool call started).
    current_tool_id: i32,
    /// Whether the name for the current tool has been sent.
    current_tool_name_sent: bool,
    /// Streamed argument characters for each tool call.
    streamed_args_for_tool: Vec<String>,
    /// Whether we've seen the start of JSON (first `{`).
    json_started: bool,
    /// Brace depth for tracking JSON object boundaries.
    brace_depth: i32,
    /// Start position of the current JSON object in the full text.
    current_json_start: usize,
}

impl LlamaJsonStreamingState {
    fn new() -> Self {
        Self {
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
            json_started: false,
            brace_depth: 0,
            current_json_start: 0,
        }
    }
}

impl StreamingToolParserState for LlamaJsonStreamingState {
    fn process_delta(
        &mut self,
        _previous_text: &str,
        current_text: &str,
        delta_text: &str,
    ) -> ToolParserDelta {
        // Strip python tag from current_text for analysis.
        let effective_text = current_text
            .strip_prefix(LLAMA_PYTHON_TAG)
            .unwrap_or(current_text);

        // If we haven't seen a `{` yet, emit as content.
        if !self.json_started {
            if let Some(brace_pos) = effective_text.find('{') {
                self.json_started = true;
                self.brace_depth = 0;
                self.current_json_start = brace_pos;

                // Count braces in the text from brace_pos.
                for ch in effective_text[brace_pos..].chars() {
                    match ch {
                        '{' => self.brace_depth += 1,
                        '}' => self.brace_depth -= 1,
                        _ => {}
                    }
                }

                // Start a new tool call.
                self.current_tool_id += 1;
                self.current_tool_name_sent = false;
                self.streamed_args_for_tool.push(String::new());

                // Content before the first `{` in delta_text.
                let delta_stripped = delta_text
                    .strip_prefix(LLAMA_PYTHON_TAG)
                    .unwrap_or(delta_text);
                if let Some(pos) = delta_stripped.find('{') {
                    let before = &delta_stripped[..pos];
                    if !before.is_empty() {
                        return ToolParserDelta::Content(before.to_string());
                    }
                }
            } else {
                // No JSON yet, strip python_tag from delta for content.
                let delta_stripped = delta_text
                    .strip_prefix(LLAMA_PYTHON_TAG)
                    .unwrap_or(delta_text);
                if delta_stripped.is_empty() {
                    return ToolParserDelta::None;
                }
                return ToolParserDelta::Content(delta_stripped.to_string());
            }
        } else {
            // Track brace depth for new characters.
            for ch in delta_text.chars() {
                match ch {
                    '{' => self.brace_depth += 1,
                    '}' => self.brace_depth -= 1,
                    _ => {}
                }
            }

            // If depth returns to 0, current JSON object is complete.
            if self.brace_depth == 0 {
                // A new JSON object might start.
                // Check if there's another `{` after current close.
                let remainder = effective_text[self.current_json_start..].trim();
                if let Some((val, end)) = try_parse_json_object(remainder) {
                    let after = remainder[end..].trim_start();
                    if after.starts_with('{') {
                        // New tool call starting.
                        self.current_tool_id += 1;
                        self.current_tool_name_sent = false;
                        self.streamed_args_for_tool.push(String::new());
                        self.current_json_start += end + remainder[end..].find('{').unwrap_or(0);
                        self.brace_depth = 1; // for the new opening brace
                    }
                    let _ = val; // processed below via partial parse
                }
            }
        }

        // Try partial JSON parse of the current tool call's text.
        let tool_idx = self.current_tool_id as usize;
        let json_text = &effective_text[self.current_json_start..];
        let parsed = partial_json_parse(json_text.trim());

        if let Some(ref val) = parsed {
            if !self.current_tool_name_sent
                && let Some(name) = val.get("name").and_then(|n| n.as_str())
            {
                self.current_tool_name_sent = true;
                return ToolParserDelta::ToolCalls(vec![DeltaToolCall {
                    index: tool_idx as u32,
                    id: Some(format!("call_{}", Uuid::new_v4().simple())),
                    call_type: Some("function".to_string()),
                    function_name: Some(name.to_string()),
                    function_arguments: Some(String::new()),
                }]);
            }

            if self.current_tool_name_sent {
                let current_args = val
                    .get("arguments")
                    .map(|a| {
                        if a.is_string() {
                            a.as_str().unwrap().to_string()
                        } else {
                            serde_json::to_string(a).unwrap_or_default()
                        }
                    })
                    .unwrap_or_default();

                let prev_args = &self.streamed_args_for_tool[tool_idx];
                if current_args.len() > prev_args.len() {
                    let diff = current_args[prev_args.len()..].to_string();
                    self.streamed_args_for_tool[tool_idx] = current_args;

                    return ToolParserDelta::ToolCalls(vec![DeltaToolCall {
                        index: tool_idx as u32,
                        id: None,
                        call_type: None,
                        function_name: None,
                        function_arguments: Some(diff),
                    }]);
                }
            }
        }

        ToolParserDelta::None
    }
}

// ---------------------------------------------------------------------------
// Partial JSON helper
// ---------------------------------------------------------------------------

/// Attempt to parse potentially incomplete JSON by closing open braces/brackets.
///
/// Tries parsing as-is first, then attempts to fix by appending closing
/// characters for unmatched `{` and `[`.
pub fn partial_json_parse(input: &str) -> Option<serde_json::Value> {
    // Try parsing as-is first.
    if let Ok(v) = serde_json::from_str(input) {
        return Some(v);
    }

    // Track unmatched delimiters in order (outside strings).
    let mut delimiter_stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut prev_backslash = false;

    for ch in input.chars() {
        if in_string {
            if ch == '\\' && !prev_backslash {
                prev_backslash = true;
                continue;
            }
            if ch == '"' && !prev_backslash {
                in_string = false;
            }
            prev_backslash = false;
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => delimiter_stack.push('{'),
            '}' => {
                delimiter_stack.pop();
            }
            '[' => delimiter_stack.push('['),
            ']' => {
                delimiter_stack.pop();
            }
            _ => {}
        }
        prev_backslash = false;
    }

    if delimiter_stack.is_empty() && !in_string {
        return None; // Not fixable by closing delimiters.
    }

    // Try closing the string if we're inside one, then close delimiters
    // in reverse order (innermost first).
    let mut fixed = input.to_string();

    if in_string {
        fixed.push('"');
    }

    for &delim in delimiter_stack.iter().rev() {
        match delim {
            '{' => fixed.push('}'),
            '[' => fixed.push(']'),
            _ => {}
        }
    }

    serde_json::from_str(&fixed).ok()
}

// ---------------------------------------------------------------------------
// Kimi K2 tool parser
// ---------------------------------------------------------------------------

/// Markers for Kimi K2 tool call format.
const KIMI_SECTION_BEGIN: &str = "<|tool_calls_section_begin|>";
const KIMI_SECTION_BEGIN_SINGULAR: &str = "<|tool_call_section_begin|>";
const KIMI_SECTION_END: &str = "<|tool_calls_section_end|>";
const KIMI_SECTION_END_SINGULAR: &str = "<|tool_call_section_end|>";
const KIMI_CALL_BEGIN: &str = "<|tool_call_begin|>";
const KIMI_CALL_ARG_BEGIN: &str = "<|tool_call_argument_begin|>";
const KIMI_CALL_END: &str = "<|tool_call_end|>";

/// Kimi K2-style tool call parser.
///
/// Format:
/// ```text
/// <|tool_calls_section_begin|>
/// <|tool_call_begin|> functions.get_weather:0 <|tool_call_argument_begin|> {"city": "SF"} <|tool_call_end|>
/// <|tool_calls_section_end|>
/// ```
#[derive(Default)]
pub struct KimiK2ToolParser;

impl KimiK2ToolParser {
    pub fn new() -> Self {
        Self
    }
}

impl ToolCallParser for KimiK2ToolParser {
    fn extract_tool_calls(&self, model_output: &str) -> ExtractedToolCallInfo {
        kimi_k2_extract(model_output)
    }

    fn create_streaming_state(&self) -> Box<dyn StreamingToolParserState + Send> {
        Box::new(KimiK2StreamingState::new())
    }
}

/// Extract the content before the tool calls section.
fn kimi_k2_find_section_start(text: &str) -> Option<usize> {
    text.find(KIMI_SECTION_BEGIN)
        .or_else(|| text.find(KIMI_SECTION_BEGIN_SINGULAR))
}

/// Parse a Kimi K2 tool call ID into a function name.
///
/// `functions.get_weather:0` → `get_weather`
fn kimi_k2_parse_function_name(tool_call_id: &str) -> String {
    let name_part = tool_call_id
        .rsplit_once(':')
        .map(|(name, _)| name)
        .unwrap_or(tool_call_id);
    name_part
        .rsplit_once('.')
        .map(|(_, name)| name)
        .unwrap_or(name_part)
        .to_string()
}

/// Extract tool calls from Kimi K2-formatted text.
fn kimi_k2_extract(text: &str) -> ExtractedToolCallInfo {
    let section_start = match kimi_k2_find_section_start(text) {
        Some(pos) => pos,
        None => {
            return ExtractedToolCallInfo {
                tools_called: false,
                tool_calls: Vec::new(),
                content: Some(text.to_string()),
            };
        }
    };

    let content_before = text[..section_start].trim();
    let content = if content_before.is_empty() {
        None
    } else {
        Some(content_before.to_string())
    };

    // Extract individual tool calls using the markers.
    let mut tool_calls = Vec::new();
    let mut search_pos = section_start;

    while let Some(call_begin) = text[search_pos..].find(KIMI_CALL_BEGIN) {
        let call_begin = search_pos + call_begin + KIMI_CALL_BEGIN.len();

        // Find the argument section.
        let arg_begin = match text[call_begin..].find(KIMI_CALL_ARG_BEGIN) {
            Some(pos) => call_begin + pos,
            None => break,
        };
        let tool_call_id = text[call_begin..arg_begin].trim();
        let args_start = arg_begin + KIMI_CALL_ARG_BEGIN.len();

        // Find the end of this tool call.
        let call_end = text[args_start..]
            .find(KIMI_CALL_END)
            .map(|p| args_start + p)
            .unwrap_or(text.len());
        let args_str = text[args_start..call_end].trim();

        let function_name = kimi_k2_parse_function_name(tool_call_id);

        // Validate JSON arguments.
        if serde_json::from_str::<serde_json::Value>(args_str).is_ok() {
            tool_calls.push(protocol::ToolCall {
                id: format!("call_{}", Uuid::new_v4().simple()),
                call_type: "function".to_string(),
                function: protocol::FunctionCall {
                    name: function_name,
                    arguments: args_str.to_string(),
                },
            });
        }

        search_pos = if call_end < text.len() {
            call_end + KIMI_CALL_END.len()
        } else {
            text.len()
        };
    }

    if tool_calls.is_empty() {
        ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(text.to_string()),
        }
    } else {
        ExtractedToolCallInfo {
            tools_called: true,
            tool_calls,
            content,
        }
    }
}

// ---------------------------------------------------------------------------
// Kimi K2 streaming state machine
// ---------------------------------------------------------------------------

struct KimiK2StreamingState {
    /// Whether we're inside a tool_calls_section.
    in_tool_section: bool,
    /// Current tool call index (-1 = no tool call started).
    current_tool_id: i32,
    /// Whether the name for the current tool has been sent.
    current_tool_name_sent: bool,
    /// Streamed argument characters for each tool call.
    streamed_args_for_tool: Vec<String>,
    /// Buffer for partial tag tokens.
    buffer: String,
    /// Number of tool_call_begin tags seen.
    num_call_begins: usize,
    /// Number of tool_call_end tags seen.
    num_call_ends: usize,
}

impl KimiK2StreamingState {
    fn new() -> Self {
        Self {
            in_tool_section: false,
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
            buffer: String::new(),
            num_call_begins: 0,
            num_call_ends: 0,
        }
    }
}

/// Check if text ends with a partial Kimi K2 tag.
fn is_partial_kimi_tag(text: &str) -> bool {
    for tag in [
        KIMI_SECTION_BEGIN,
        KIMI_SECTION_BEGIN_SINGULAR,
        KIMI_SECTION_END,
        KIMI_SECTION_END_SINGULAR,
        KIMI_CALL_BEGIN,
        KIMI_CALL_ARG_BEGIN,
        KIMI_CALL_END,
    ] {
        for i in 1..tag.len() {
            if text.ends_with(&tag[..i]) {
                return true;
            }
        }
    }
    false
}

impl StreamingToolParserState for KimiK2StreamingState {
    fn process_delta(
        &mut self,
        _previous_text: &str,
        current_text: &str,
        delta_text: &str,
    ) -> ToolParserDelta {
        self.buffer.push_str(delta_text);

        // Wait for more tokens if we might be in a partial tag.
        if is_partial_kimi_tag(&self.buffer) {
            return ToolParserDelta::None;
        }

        let text_to_process = std::mem::take(&mut self.buffer);

        // Check if the section has started.
        if !self.in_tool_section {
            if current_text.contains(KIMI_SECTION_BEGIN)
                || current_text.contains(KIMI_SECTION_BEGIN_SINGULAR)
            {
                self.in_tool_section = true;
                // Emit any content before the section marker in this delta.
                let before_marker = if let Some(pos) = text_to_process.find("<|tool_call") {
                    &text_to_process[..pos]
                } else {
                    ""
                };
                if !before_marker.is_empty() {
                    return ToolParserDelta::Content(before_marker.to_string());
                }
                return ToolParserDelta::None;
            }
            // No section yet — emit as content.
            return ToolParserDelta::Content(text_to_process);
        }

        // We're inside the tool calls section.
        // Count tool_call_begin/end tags in full text.
        let new_begins = current_text.matches(KIMI_CALL_BEGIN).count();
        let new_ends = current_text.matches(KIMI_CALL_END).count();

        // New tool call started?
        if new_begins > self.num_call_begins {
            self.num_call_begins = new_begins;
            self.current_tool_id += 1;
            self.current_tool_name_sent = false;
            self.streamed_args_for_tool.push(String::new());
        }
        self.num_call_ends = new_ends;

        if self.current_tool_id < 0 {
            return ToolParserDelta::None;
        }

        let tool_idx = self.current_tool_id as usize;

        // Extract the current tool call's text from current_text.
        // Find the Nth tool_call_begin tag.
        let mut search = 0;
        for _ in 0..self.num_call_begins {
            if let Some(pos) = current_text[search..].find(KIMI_CALL_BEGIN) {
                search = search + pos + KIMI_CALL_BEGIN.len();
            }
        }
        let after_last_begin = &current_text[search..];

        // Try to extract function name (before <|tool_call_argument_begin|>).
        if !self.current_tool_name_sent
            && let Some(arg_pos) = after_last_begin.find(KIMI_CALL_ARG_BEGIN)
        {
            let tool_call_id = after_last_begin[..arg_pos].trim();
            if !tool_call_id.is_empty() {
                let function_name = kimi_k2_parse_function_name(tool_call_id);
                self.current_tool_name_sent = true;

                return ToolParserDelta::ToolCalls(vec![DeltaToolCall {
                    index: tool_idx as u32,
                    id: Some(format!("call_{}", Uuid::new_v4().simple())),
                    call_type: Some("function".to_string()),
                    function_name: Some(function_name),
                    function_arguments: Some(String::new()),
                }]);
            }
        }

        // Stream argument diffs.
        if self.current_tool_name_sent {
            // Get text after <|tool_call_argument_begin|>.
            if let Some(arg_start_pos) = after_last_begin.find(KIMI_CALL_ARG_BEGIN) {
                let args_text_start = arg_start_pos + KIMI_CALL_ARG_BEGIN.len();
                let args_text = if let Some(end_pos) =
                    after_last_begin[args_text_start..].find(KIMI_CALL_END)
                {
                    &after_last_begin[args_text_start..args_text_start + end_pos]
                } else {
                    &after_last_begin[args_text_start..]
                };
                let args_text = args_text.trim_start();

                let prev_args = &self.streamed_args_for_tool[tool_idx];
                if args_text.len() > prev_args.len() {
                    let diff = args_text[prev_args.len()..].to_string();
                    self.streamed_args_for_tool[tool_idx] = args_text.to_string();

                    return ToolParserDelta::ToolCalls(vec![DeltaToolCall {
                        index: tool_idx as u32,
                        id: None,
                        call_type: None,
                        function_name: None,
                        function_arguments: Some(diff),
                    }]);
                }
            }
        }

        ToolParserDelta::None
    }
}

// ---------------------------------------------------------------------------
// Mistral tool parser
// ---------------------------------------------------------------------------

const MISTRAL_BOT_TOKEN: &str = "[TOOL_CALLS]";

/// Generate a 9-character alphanumeric random ID matching Mistral's format.
fn mistral_generate_id() -> String {
    use rand::Rng;
    const ALPHANUMERIC: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut rng = rand::rng();
    (0..9)
        .map(|_| ALPHANUMERIC[rng.random_range(0..ALPHANUMERIC.len())] as char)
        .collect()
}

/// Mistral-style tool call parser.
///
/// Supports two formats:
/// - v11+: `[TOOL_CALLS]func_name{"arg":"val"}[TOOL_CALLS]func2{"arg2":"val2"}`
/// - Pre-v11: `[TOOL_CALLS] [{"name":"func","arguments":{"arg":"val"}}]`
///
/// Format is auto-detected by checking if text after `[TOOL_CALLS]` starts with `[`.
#[derive(Default)]
pub struct MistralToolParser;

impl MistralToolParser {
    pub fn new() -> Self {
        Self
    }
}

impl ToolCallParser for MistralToolParser {
    fn extract_tool_calls(&self, model_output: &str) -> ExtractedToolCallInfo {
        mistral_extract(model_output)
    }

    fn create_streaming_state(&self) -> Box<dyn StreamingToolParserState + Send> {
        Box::new(MistralStreamingState::new())
    }
}

fn mistral_extract(text: &str) -> ExtractedToolCallInfo {
    if !text.contains(MISTRAL_BOT_TOKEN) {
        return ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(text.to_string()),
        };
    }

    let parts: Vec<&str> = text.splitn(2, MISTRAL_BOT_TOKEN).collect();
    let content = parts[0];
    let rest = parts.get(1).unwrap_or(&"");

    // Auto-detect format: if rest starts with `[` (after trimming), it's pre-v11
    let trimmed_rest = rest.trim_start();
    let is_pre_v11 = trimmed_rest.starts_with('[');

    let tool_calls = if is_pre_v11 {
        // Pre-v11: `[{"name":"func","arguments":{...}}]`
        mistral_extract_pre_v11(trimmed_rest)
    } else {
        // v11+: split on [TOOL_CALLS] for multiple tools
        // rest is everything after the first [TOOL_CALLS], may contain more [TOOL_CALLS] delimiters
        let full_tool_text = &text[parts[0].len() + MISTRAL_BOT_TOKEN.len()..];
        let segments: Vec<&str> = full_tool_text.split(MISTRAL_BOT_TOKEN).collect();
        let mut calls = Vec::new();
        for segment in segments {
            if let Some(brace_pos) = segment.find('{') {
                let name = &segment[..brace_pos];
                let args = &segment[brace_pos..];
                calls.push(protocol::ToolCall {
                    id: mistral_generate_id(),
                    call_type: "function".to_string(),
                    function: protocol::FunctionCall {
                        name: name.to_string(),
                        arguments: args.to_string(),
                    },
                });
            }
        }
        calls
    };

    if tool_calls.is_empty() {
        return ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(text.to_string()),
        };
    }

    ExtractedToolCallInfo {
        tools_called: true,
        tool_calls,
        content: if content.is_empty() {
            None
        } else {
            Some(content.to_string())
        },
    }
}

fn mistral_extract_pre_v11(json_text: &str) -> Vec<protocol::ToolCall> {
    // Try direct JSON parse first
    let parsed: Result<Vec<serde_json::Value>, _> = serde_json::from_str(json_text);
    let arr = match parsed {
        Ok(arr) => arr,
        Err(_) => {
            // Fallback: find `[{...}]` pattern (matching Python's regex r"\[{.*}\]")
            let start = json_text.find("[{");
            let end = json_text.rfind("}]");
            if let (Some(s), Some(e)) = (start, end) {
                let substr = &json_text[s..e + 2];
                match serde_json::from_str::<Vec<serde_json::Value>>(substr) {
                    Ok(arr) => arr,
                    Err(_) => return Vec::new(),
                }
            } else {
                return Vec::new();
            }
        }
    };

    arr.into_iter()
        .filter_map(|val| {
            let name = val.get("name")?.as_str()?.to_string();
            let arguments = val.get("arguments")?;
            let arguments_str = if arguments.is_string() {
                arguments.as_str().unwrap().to_string()
            } else {
                serde_json::to_string(arguments).ok()?
            };
            Some(protocol::ToolCall {
                id: mistral_generate_id(),
                call_type: "function".to_string(),
                function: protocol::FunctionCall {
                    name,
                    arguments: arguments_str,
                },
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Mistral streaming state machine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MistralStreamFormat {
    Unknown,
    V11,
    PreV11,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MistralStreamState {
    WaitingForToolStart,
    ParsingName,
    ParsingArguments,
}

struct MistralStreamingState {
    format: MistralStreamFormat,
    state: MistralStreamState,
    current_tool_id: i32,
    current_tool_name: String,
    /// Buffer for accumulating text until we can determine format or complete a parse unit.
    buffer: String,
    /// For pre-v11: brace depth for JSON streaming.
    brace_depth: i32,
    /// For pre-v11: whether we're inside a JSON string.
    in_string: bool,
    /// For pre-v11: previous char was backslash (escape).
    escape_next: bool,
    /// For pre-v11: accumulated arguments JSON for current tool.
    pre_v11_args_buf: String,
    /// For pre-v11: accumulated name.
    pre_v11_key: Option<String>,
    /// For pre-v11: current JSON key being parsed.
    pre_v11_parse_state: PreV11ParseState,
    /// Whether we've seen the bot token at all.
    bot_token_seen: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreV11ParseState {
    /// Looking for the start of an object `{`.
    WaitingForObject,
    /// Inside an object, looking for keys.
    InObject,
    /// Parsing the "name" string value.
    ParsingNameValue,
    /// Parsing the "arguments" object.
    ParsingArgumentsValue,
    /// Object complete, waiting for next or end of array.
    ObjectComplete,
    /// Array complete.
    Done,
}

impl MistralStreamingState {
    fn new() -> Self {
        Self {
            format: MistralStreamFormat::Unknown,
            state: MistralStreamState::WaitingForToolStart,
            current_tool_id: -1,
            current_tool_name: String::new(),
            buffer: String::new(),
            brace_depth: 0,
            in_string: false,
            escape_next: false,
            pre_v11_args_buf: String::new(),
            pre_v11_key: None,
            pre_v11_parse_state: PreV11ParseState::WaitingForObject,
            bot_token_seen: false,
        }
    }

    /// Process v11+ format streaming.
    fn process_v11(&mut self, text: &str) -> Vec<DeltaToolCall> {
        let mut deltas = Vec::new();
        let mut remaining = text;

        loop {
            match self.state {
                MistralStreamState::WaitingForToolStart => {
                    // Look for [TOOL_CALLS] token
                    if let Some(pos) = remaining.find(MISTRAL_BOT_TOKEN) {
                        remaining = &remaining[pos + MISTRAL_BOT_TOKEN.len()..];
                        self.current_tool_id += 1;
                        self.current_tool_name.clear();
                        self.state = MistralStreamState::ParsingName;
                    } else {
                        break;
                    }
                }
                MistralStreamState::ParsingName => {
                    if let Some(brace_pos) = remaining.find('{') {
                        let name_part = &remaining[..brace_pos];
                        self.current_tool_name.push_str(name_part);
                        remaining = &remaining[brace_pos..];
                        self.state = MistralStreamState::ParsingArguments;

                        // Emit name delta with ID
                        deltas.push(DeltaToolCall {
                            index: self.current_tool_id as u32,
                            id: Some(mistral_generate_id()),
                            call_type: Some("function".to_string()),
                            function_name: Some(self.current_tool_name.clone()),
                            function_arguments: None,
                        });
                    } else {
                        // Buffer the name fragment, don't emit yet
                        self.current_tool_name.push_str(remaining);
                        break;
                    }
                }
                MistralStreamState::ParsingArguments => {
                    // Check if there's another [TOOL_CALLS] — means current tool is done
                    if let Some(pos) = remaining.find(MISTRAL_BOT_TOKEN) {
                        let args_part = &remaining[..pos];
                        if !args_part.is_empty() {
                            deltas.push(DeltaToolCall {
                                index: self.current_tool_id as u32,
                                id: None,
                                call_type: None,
                                function_name: None,
                                function_arguments: Some(args_part.to_string()),
                            });
                        }
                        remaining = &remaining[pos..]; // keep [TOOL_CALLS] for next iteration
                        self.state = MistralStreamState::WaitingForToolStart;
                    } else {
                        // All remaining text is arguments
                        if !remaining.is_empty() {
                            deltas.push(DeltaToolCall {
                                index: self.current_tool_id as u32,
                                id: None,
                                call_type: None,
                                function_name: None,
                                function_arguments: Some(remaining.to_string()),
                            });
                        }
                        break;
                    }
                }
            }
        }

        deltas
    }

    /// Process pre-v11 format streaming using brace-counting.
    fn process_pre_v11(&mut self, text: &str) -> Vec<DeltaToolCall> {
        let mut deltas = Vec::new();

        for ch in text.chars() {
            match self.pre_v11_parse_state {
                PreV11ParseState::WaitingForObject => {
                    if ch == '{' {
                        self.pre_v11_parse_state = PreV11ParseState::InObject;
                        self.current_tool_id += 1;
                        self.pre_v11_key = None;
                        self.current_tool_name.clear();
                        self.pre_v11_args_buf.clear();
                        self.brace_depth = 0;
                        self.in_string = false;
                        self.escape_next = false;
                    }
                    // skip [ , whitespace etc.
                }
                PreV11ParseState::InObject => {
                    // We're inside the top-level object, looking for "name" or "arguments" keys
                    // Simple approach: accumulate into buffer until we identify key-value pairs
                    self.buffer.push(ch);

                    // Check if we've accumulated a complete key
                    if self.buffer.contains("\"name\"") && self.buffer.ends_with(':') {
                        self.buffer.clear();
                        self.pre_v11_parse_state = PreV11ParseState::ParsingNameValue;
                        self.in_string = false;
                    } else if self.buffer.contains("\"arguments\"") && self.buffer.ends_with(':') {
                        self.buffer.clear();
                        self.pre_v11_parse_state = PreV11ParseState::ParsingArgumentsValue;
                        self.brace_depth = 0;
                        self.in_string = false;
                        self.escape_next = false;
                    } else if ch == '}' && !self.buffer.contains('"') {
                        // End of object without finding expected keys
                        self.buffer.clear();
                        self.pre_v11_parse_state = PreV11ParseState::ObjectComplete;
                    }
                }
                PreV11ParseState::ParsingNameValue => {
                    // Parse a JSON string value for the name
                    if ch == '"' && !self.in_string {
                        self.in_string = true;
                    } else if self.in_string {
                        if self.escape_next {
                            self.current_tool_name.push(ch);
                            self.escape_next = false;
                        } else if ch == '\\' {
                            self.escape_next = true;
                        } else if ch == '"' {
                            // Name complete — emit it
                            deltas.push(DeltaToolCall {
                                index: self.current_tool_id as u32,
                                id: Some(mistral_generate_id()),
                                call_type: Some("function".to_string()),
                                function_name: Some(self.current_tool_name.clone()),
                                function_arguments: None,
                            });
                            self.in_string = false;
                            self.pre_v11_parse_state = PreV11ParseState::InObject;
                            self.buffer.clear();
                        } else {
                            self.current_tool_name.push(ch);
                        }
                    }
                }
                PreV11ParseState::ParsingArgumentsValue => {
                    // Stream arguments using brace counting
                    if ch == '{' && !self.in_string {
                        self.brace_depth += 1;
                        self.pre_v11_args_buf.push(ch);
                        if self.brace_depth == 1 {
                            // Emit the opening brace as first args delta
                            deltas.push(DeltaToolCall {
                                index: self.current_tool_id as u32,
                                id: None,
                                call_type: None,
                                function_name: None,
                                function_arguments: Some("{".to_string()),
                            });
                            self.pre_v11_args_buf.clear();
                        }
                    } else if ch == '}' && !self.in_string {
                        self.brace_depth -= 1;
                        if self.brace_depth == 0 {
                            // Arguments complete
                            if !self.pre_v11_args_buf.is_empty() {
                                deltas.push(DeltaToolCall {
                                    index: self.current_tool_id as u32,
                                    id: None,
                                    call_type: None,
                                    function_name: None,
                                    function_arguments: Some(self.pre_v11_args_buf.clone()),
                                });
                                self.pre_v11_args_buf.clear();
                            }
                            deltas.push(DeltaToolCall {
                                index: self.current_tool_id as u32,
                                id: None,
                                call_type: None,
                                function_name: None,
                                function_arguments: Some("}".to_string()),
                            });
                            self.pre_v11_parse_state = PreV11ParseState::InObject;
                            self.buffer.clear();
                        } else {
                            self.pre_v11_args_buf.push(ch);
                        }
                    } else {
                        // Handle strings for correct brace counting
                        if ch == '"' && !self.escape_next {
                            self.in_string = !self.in_string;
                        }
                        self.escape_next = ch == '\\' && self.in_string && !self.escape_next;
                        if self.brace_depth > 0 {
                            self.pre_v11_args_buf.push(ch);
                        }
                    }
                }
                PreV11ParseState::ObjectComplete => {
                    if ch == '{' {
                        // Next object
                        self.pre_v11_parse_state = PreV11ParseState::InObject;
                        self.current_tool_id += 1;
                        self.current_tool_name.clear();
                        self.pre_v11_args_buf.clear();
                        self.brace_depth = 0;
                        self.buffer.clear();
                    } else if ch == ']' {
                        self.pre_v11_parse_state = PreV11ParseState::Done;
                    }
                }
                PreV11ParseState::Done => {}
            }
        }

        // Flush any accumulated args for pre-v11 in-progress arguments
        if self.pre_v11_parse_state == PreV11ParseState::ParsingArgumentsValue
            && self.brace_depth > 0
            && !self.pre_v11_args_buf.is_empty()
        {
            deltas.push(DeltaToolCall {
                index: self.current_tool_id as u32,
                id: None,
                call_type: None,
                function_name: None,
                function_arguments: Some(self.pre_v11_args_buf.clone()),
            });
            self.pre_v11_args_buf.clear();
        }

        deltas
    }
}

impl StreamingToolParserState for MistralStreamingState {
    fn process_delta(
        &mut self,
        _previous_text: &str,
        current_text: &str,
        delta_text: &str,
    ) -> ToolParserDelta {
        if delta_text.is_empty() {
            return ToolParserDelta::None;
        }

        // If we haven't seen the bot token yet, check if it's in current_text
        if !self.bot_token_seen {
            if !current_text.contains(MISTRAL_BOT_TOKEN) {
                return ToolParserDelta::Content(delta_text.to_string());
            }
            self.bot_token_seen = true;

            // Extract content before [TOOL_CALLS]
            if delta_text.contains(MISTRAL_BOT_TOKEN) {
                let parts: Vec<&str> = delta_text.splitn(2, MISTRAL_BOT_TOKEN).collect();
                if !parts[0].is_empty() {
                    // Buffer the post-bot-token text for format detection
                    let after = parts.get(1).unwrap_or(&"");
                    if !after.is_empty() {
                        self.buffer.push_str(after);
                    }
                    // Try to detect format now
                    return self.try_detect_and_flush(Some(parts[0].to_string()));
                }
                let after = parts.get(1).unwrap_or(&"");
                if !after.is_empty() {
                    self.buffer.push_str(after);
                }
            }

            return self.try_detect_and_flush(None);
        }

        // If format not yet detected, buffer and try again
        if self.format == MistralStreamFormat::Unknown {
            self.buffer.push_str(delta_text);
            return self.try_detect_and_flush(None);
        }

        // Format detected, process normally
        let deltas = match self.format {
            MistralStreamFormat::V11 => self.process_v11(delta_text),
            MistralStreamFormat::PreV11 => self.process_pre_v11(delta_text),
            MistralStreamFormat::Unknown => Vec::new(),
        };

        if deltas.is_empty() {
            ToolParserDelta::None
        } else {
            ToolParserDelta::ToolCalls(deltas)
        }
    }
}

impl MistralStreamingState {
    /// Try to detect format from buffered text. If detected, flush buffer through parser.
    fn try_detect_and_flush(&mut self, content_before: Option<String>) -> ToolParserDelta {
        let trimmed = self.buffer.trim_start();
        if trimmed.is_empty() {
            // Not enough data to detect format yet
            if let Some(content) = content_before {
                return ToolParserDelta::Content(content);
            }
            return ToolParserDelta::None;
        }

        if trimmed.starts_with('[') {
            self.format = MistralStreamFormat::PreV11;
        } else {
            self.format = MistralStreamFormat::V11;
        }

        // Flush buffered text through the appropriate parser
        // For V11, we need to prepend [TOOL_CALLS] since process_v11 expects it
        let buffered = std::mem::take(&mut self.buffer);
        let deltas = match self.format {
            MistralStreamFormat::V11 => {
                let with_token = format!("{}{}", MISTRAL_BOT_TOKEN, buffered);
                self.process_v11(&with_token)
            }
            MistralStreamFormat::PreV11 => self.process_pre_v11(&buffered),
            MistralStreamFormat::Unknown => Vec::new(),
        };

        match (content_before, deltas.is_empty()) {
            (Some(content), true) => ToolParserDelta::Content(content),
            (_, false) => ToolParserDelta::ToolCalls(deltas),
            (None, true) => ToolParserDelta::None,
        }
    }
}

// ---------------------------------------------------------------------------
// Jamba tool parser
// ---------------------------------------------------------------------------

const JAMBA_TOOL_CALLS_OPEN: &str = "<tool_calls>";
const JAMBA_TOOL_CALLS_CLOSE: &str = "</tool_calls>";

/// Jamba-style tool call parser.
///
/// Detects tool calls wrapped in `<tool_calls>...</tool_calls>` tags.
/// The content between tags is a JSON **array** of objects, each with
/// `name` and `arguments` fields.
///
/// Port of: `vllm/tool_parsers/jamba_tool_parser.py`
#[derive(Default)]
pub struct JambaToolParser;

impl JambaToolParser {
    pub fn new() -> Self {
        Self
    }
}

impl ToolCallParser for JambaToolParser {
    fn extract_tool_calls(&self, model_output: &str) -> ExtractedToolCallInfo {
        jamba_extract(model_output)
    }

    fn create_streaming_state(&self) -> Box<dyn StreamingToolParserState + Send> {
        Box::new(JambaStreamingState::new())
    }
}

/// Extract tool calls from Jamba-formatted text.
fn jamba_extract(text: &str) -> ExtractedToolCallInfo {
    // Check for the open tag.
    let Some(open_pos) = text.find(JAMBA_TOOL_CALLS_OPEN) else {
        return ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(text.to_string()),
        };
    };

    // Content before the open tag.
    let before = text[..open_pos].trim();
    let content = if before.is_empty() {
        None
    } else {
        Some(before.to_string())
    };

    // Extract the JSON array between tags.
    let json_start = open_pos + JAMBA_TOOL_CALLS_OPEN.len();
    let json_end = text[json_start..]
        .find(JAMBA_TOOL_CALLS_CLOSE)
        .map(|p| json_start + p)
        .unwrap_or(text.len());
    let json_str = text[json_start..json_end].trim();

    // Parse as a JSON array.
    let arr: Vec<serde_json::Value> = match serde_json::from_str(json_str) {
        Ok(arr) => arr,
        Err(_) => {
            return ExtractedToolCallInfo {
                tools_called: false,
                tool_calls: Vec::new(),
                content: Some(text.to_string()),
            };
        }
    };

    let tool_calls: Vec<protocol::ToolCall> = arr
        .iter()
        .filter_map(|val| {
            let name = val.get("name")?.as_str()?.to_string();
            let arguments = val.get("arguments")?;
            let arguments_str = if arguments.is_string() {
                arguments.as_str().unwrap().to_string()
            } else {
                serde_json::to_string(arguments).ok()?
            };
            Some(protocol::ToolCall {
                id: format!("call_{}", Uuid::new_v4().simple()),
                call_type: "function".to_string(),
                function: protocol::FunctionCall {
                    name,
                    arguments: arguments_str,
                },
            })
        })
        .collect();

    if tool_calls.is_empty() {
        ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(text.to_string()),
        }
    } else {
        ExtractedToolCallInfo {
            tools_called: true,
            tool_calls,
            content,
        }
    }
}

// ---------------------------------------------------------------------------
// Jamba streaming state machine
// ---------------------------------------------------------------------------

struct JambaStreamingState {
    /// Current tool call index (-1 = no tool call started).
    current_tool_id: i32,
    /// Whether the name for the current tool has been sent.
    current_tool_name_sent: bool,
    /// Previously parsed tool call array.
    prev_tool_call_arr: Vec<serde_json::Value>,
    /// Streamed argument characters for each tool call (for diffing).
    streamed_args_for_tool: Vec<String>,
    /// Buffer for partial tag tokens.
    buffer: String,
    /// Whether we've seen the open tag yet.
    seen_open_tag: bool,
}

impl JambaStreamingState {
    fn new() -> Self {
        Self {
            current_tool_id: -1,
            current_tool_name_sent: false,
            prev_tool_call_arr: Vec::new(),
            streamed_args_for_tool: Vec::new(),
            buffer: String::new(),
            seen_open_tag: false,
        }
    }
}

impl StreamingToolParserState for JambaStreamingState {
    fn process_delta(
        &mut self,
        _previous_text: &str,
        current_text: &str,
        delta_text: &str,
    ) -> ToolParserDelta {
        // Buffer delta for partial tag detection.
        self.buffer.push_str(delta_text);

        // Check for partial open/close tags.
        if is_partial_jamba_tag(&self.buffer) {
            return ToolParserDelta::None;
        }

        let text_to_process = std::mem::take(&mut self.buffer);

        // If we haven't seen the open tag yet, check for it.
        if !self.seen_open_tag {
            if current_text.contains(JAMBA_TOOL_CALLS_OPEN) {
                self.seen_open_tag = true;
                // Suppress the open tag token itself.
                if text_to_process.contains(JAMBA_TOOL_CALLS_OPEN) {
                    // There might be content before the tag in previous deltas
                    // (already streamed). Just suppress this delta.
                    return ToolParserDelta::None;
                }
            } else {
                // No tool calls yet — emit as content.
                return ToolParserDelta::Content(text_to_process);
            }
        }

        // We're inside the <tool_calls> region. Extract the parsable array.
        let parsable_arr = current_text
            .split(JAMBA_TOOL_CALLS_OPEN)
            .last()
            .unwrap_or("")
            .split(JAMBA_TOOL_CALLS_CLOSE)
            .next()
            .unwrap_or("");

        // Try partial JSON parse of the array.
        let parsed = partial_json_parse(parsable_arr.trim());

        let Some(val) = parsed else {
            return ToolParserDelta::None;
        };

        let Some(tool_call_arr) = val.as_array() else {
            return ToolParserDelta::None;
        };

        // Empty array — nothing to stream yet.
        if tool_call_arr.is_empty() {
            return ToolParserDelta::None;
        }

        // Check if a new tool call started (array grew past our cursor).
        if tool_call_arr.len() as i32 > self.current_tool_id + 1 {
            // Flush remaining args for the previous tool if any.
            let flush_delta = if self.current_tool_id >= 0 {
                let prev_idx = self.current_tool_id as usize;
                let prev_call = &tool_call_arr[prev_idx];
                let cur_args = prev_call
                    .get("arguments")
                    .map(|a| {
                        if a.is_string() {
                            a.as_str().unwrap().to_string()
                        } else {
                            serde_json::to_string(a).unwrap_or_default()
                        }
                    })
                    .unwrap_or_default();
                let prev_streamed = &self.streamed_args_for_tool[prev_idx];
                if cur_args.len() > prev_streamed.len() {
                    let diff = cur_args[prev_streamed.len()..].to_string();
                    self.streamed_args_for_tool[prev_idx] = cur_args;
                    Some(DeltaToolCall {
                        index: prev_idx as u32,
                        id: None,
                        call_type: None,
                        function_name: None,
                        function_arguments: Some(diff),
                    })
                } else {
                    None
                }
            } else {
                None
            };

            // Advance to the new tool.
            self.current_tool_id = tool_call_arr.len() as i32 - 1;
            self.current_tool_name_sent = false;
            self.streamed_args_for_tool.push(String::new());
            self.prev_tool_call_arr = tool_call_arr.clone();

            if let Some(flush) = flush_delta {
                return ToolParserDelta::ToolCalls(vec![flush]);
            }
            // Fall through to try sending the name for the new tool.
        }

        let tool_idx = self.current_tool_id as usize;
        let current_tool_call = &tool_call_arr[tool_idx];

        // Try to send the name if not yet sent.
        if !self.current_tool_name_sent {
            if let Some(name) = current_tool_call.get("name").and_then(|n| n.as_str()) {
                self.current_tool_name_sent = true;
                self.prev_tool_call_arr = tool_call_arr.clone();
                return ToolParserDelta::ToolCalls(vec![DeltaToolCall {
                    index: tool_idx as u32,
                    id: Some(format!("call_{}", Uuid::new_v4().simple())),
                    call_type: Some("function".to_string()),
                    function_name: Some(name.to_string()),
                    function_arguments: Some(String::new()),
                }]);
            }
            return ToolParserDelta::None;
        }

        // Stream arguments diff.
        let current_args = current_tool_call
            .get("arguments")
            .map(|a| {
                if a.is_string() {
                    a.as_str().unwrap().to_string()
                } else {
                    serde_json::to_string(a).unwrap_or_default()
                }
            })
            .unwrap_or_default();

        let prev_args = &self.streamed_args_for_tool[tool_idx];
        if current_args.len() > prev_args.len() {
            let diff = current_args[prev_args.len()..].to_string();
            self.streamed_args_for_tool[tool_idx] = current_args;
            self.prev_tool_call_arr = tool_call_arr.clone();

            return ToolParserDelta::ToolCalls(vec![DeltaToolCall {
                index: tool_idx as u32,
                id: None,
                call_type: None,
                function_name: None,
                function_arguments: Some(diff),
            }]);
        }

        self.prev_tool_call_arr = tool_call_arr.clone();
        ToolParserDelta::None
    }
}

/// Check if text ends with a partial `<tool_calls>` or `</tool_calls>` tag.
fn is_partial_jamba_tag(text: &str) -> bool {
    for tag in [JAMBA_TOOL_CALLS_OPEN, JAMBA_TOOL_CALLS_CLOSE] {
        for i in 1..tag.len() {
            if text.ends_with(&tag[..i]) {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Granite tool parser
// ---------------------------------------------------------------------------

/// Granite 3.0 special token prefix.
const GRANITE_BOT_TOKEN: &str = "<|tool_call|>";
/// Granite 3.1 string prefix.
const GRANITE_BOT_STRING: &str = "<tool_call>";

/// Granite-style tool call parser.
///
/// Detects tool calls prefixed by `<|tool_call|>` (Granite 3.0) or
/// `<tool_call>` (Granite 3.1), followed by a JSON array of objects
/// with `name` and `arguments` fields.
///
/// Port of: `vllm/tool_parsers/granite_tool_parser.py`
#[derive(Default)]
pub struct GraniteToolParser;

impl GraniteToolParser {
    pub fn new() -> Self {
        Self
    }
}

impl ToolCallParser for GraniteToolParser {
    fn extract_tool_calls(&self, model_output: &str) -> ExtractedToolCallInfo {
        granite_extract(model_output)
    }

    fn create_streaming_state(&self) -> Box<dyn StreamingToolParserState + Send> {
        Box::new(GraniteStreamingState::new())
    }
}

/// Strip Granite prefix tokens and leading whitespace, returning the
/// remaining text. Returns `None` if the stripped text doesn't start with `[`.
fn granite_strip_prefix(text: &str) -> Option<&str> {
    let mut s = text.trim_start();
    if let Some(rest) = s.strip_prefix(GRANITE_BOT_TOKEN) {
        s = rest.trim_start();
    }
    if let Some(rest) = s.strip_prefix(GRANITE_BOT_STRING) {
        s = rest.trim_start();
    }
    if s.starts_with('[') { Some(s) } else { None }
}

/// Extract tool calls from Granite-formatted text.
fn granite_extract(text: &str) -> ExtractedToolCallInfo {
    let Some(stripped) = granite_strip_prefix(text) else {
        return ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(text.to_string()),
        };
    };

    // Parse as a JSON array.
    let arr: Vec<serde_json::Value> = match serde_json::from_str(stripped) {
        Ok(arr) => arr,
        Err(_) => {
            return ExtractedToolCallInfo {
                tools_called: false,
                tool_calls: Vec::new(),
                content: Some(text.to_string()),
            };
        }
    };

    let tool_calls: Vec<protocol::ToolCall> = arr
        .iter()
        .filter_map(|val| {
            let name = val.get("name")?.as_str()?.to_string();
            let arguments = val.get("arguments")?;
            let arguments_str = if arguments.is_string() {
                arguments.as_str().unwrap().to_string()
            } else {
                serde_json::to_string(arguments).ok()?
            };
            Some(protocol::ToolCall {
                id: format!("call_{}", Uuid::new_v4().simple()),
                call_type: "function".to_string(),
                function: protocol::FunctionCall {
                    name,
                    arguments: arguments_str,
                },
            })
        })
        .collect();

    if tool_calls.is_empty() {
        ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(text.to_string()),
        }
    } else {
        ExtractedToolCallInfo {
            tools_called: true,
            tool_calls,
            content: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Granite streaming state machine
// ---------------------------------------------------------------------------

struct GraniteStreamingState {
    /// Current tool call index (-1 = no tool call started).
    current_tool_id: i32,
    /// Whether the name for the current tool has been sent.
    current_tool_name_sent: bool,
    /// Previously parsed tool call array.
    prev_tool_call_arr: Vec<serde_json::Value>,
    /// Streamed argument characters for each tool call (for diffing).
    streamed_args_for_tool: Vec<String>,
    /// Whether we've found the `[` start of the JSON array.
    array_started: bool,
    /// Byte offset where the JSON array begins in current_text.
    array_start_offset: usize,
}

impl GraniteStreamingState {
    fn new() -> Self {
        Self {
            current_tool_id: -1,
            current_tool_name_sent: false,
            prev_tool_call_arr: Vec::new(),
            streamed_args_for_tool: Vec::new(),
            array_started: false,
            array_start_offset: 0,
        }
    }
}

impl StreamingToolParserState for GraniteStreamingState {
    fn process_delta(
        &mut self,
        _previous_text: &str,
        current_text: &str,
        delta_text: &str,
    ) -> ToolParserDelta {
        // Find the start of the JSON array if not yet found.
        if !self.array_started {
            // Skip prefix tokens and whitespace.
            let mut s = current_text.trim_start();
            if let Some(rest) = s.strip_prefix(GRANITE_BOT_TOKEN) {
                s = rest.trim_start();
            }
            if let Some(rest) = s.strip_prefix(GRANITE_BOT_STRING) {
                s = rest.trim_start();
            }
            let offset = current_text.len() - s.len();

            if s.starts_with('[') {
                self.array_started = true;
                self.array_start_offset = offset;
            } else if s.is_empty() {
                // Still buffering prefix/whitespace.
                return ToolParserDelta::None;
            } else {
                // Not a tool call — regular content.
                return ToolParserDelta::Content(delta_text.to_string());
            }
        }

        // Parse the JSON array portion.
        let array_text = &current_text[self.array_start_offset..];
        let parsed = partial_json_parse(array_text.trim());

        let Some(val) = parsed else {
            return ToolParserDelta::None;
        };

        let Some(tool_call_arr) = val.as_array() else {
            return ToolParserDelta::None;
        };

        if tool_call_arr.is_empty() {
            return ToolParserDelta::None;
        }

        // Check completeness of the last element: if the full array_text
        // parses as valid JSON, the last element is complete.
        let last_is_complete = serde_json::from_str::<serde_json::Value>(array_text.trim()).is_ok();

        // Check if a new tool call started (array grew past cursor).
        if tool_call_arr.len() as i32 > self.current_tool_id + 1 {
            // Flush remaining args for the previous tool.
            let flush_delta = if self.current_tool_id >= 0 {
                let prev_idx = self.current_tool_id as usize;
                let prev_call = &tool_call_arr[prev_idx];
                let cur_args = prev_call
                    .get("arguments")
                    .map(|a| {
                        if a.is_string() {
                            a.as_str().unwrap().to_string()
                        } else {
                            serde_json::to_string(a).unwrap_or_default()
                        }
                    })
                    .unwrap_or_default();
                let prev_streamed = &self.streamed_args_for_tool[prev_idx];
                if cur_args.len() > prev_streamed.len() {
                    let diff = cur_args[prev_streamed.len()..].to_string();
                    self.streamed_args_for_tool[prev_idx] = cur_args;
                    Some(DeltaToolCall {
                        index: prev_idx as u32,
                        id: None,
                        call_type: None,
                        function_name: None,
                        function_arguments: Some(diff),
                    })
                } else {
                    None
                }
            } else {
                None
            };

            self.current_tool_id = tool_call_arr.len() as i32 - 1;
            self.current_tool_name_sent = false;
            self.streamed_args_for_tool.push(String::new());
            self.prev_tool_call_arr = tool_call_arr.clone();

            if let Some(flush) = flush_delta {
                return ToolParserDelta::ToolCalls(vec![flush]);
            }
            // Fall through to try sending the name.
        }

        let tool_idx = self.current_tool_id as usize;
        let current_tool_call = &tool_call_arr[tool_idx];

        // Try to send the name if not yet sent.
        if !self.current_tool_name_sent {
            if let Some(name) = current_tool_call.get("name").and_then(|n| n.as_str()) {
                self.current_tool_name_sent = true;
                self.prev_tool_call_arr = tool_call_arr.clone();
                return ToolParserDelta::ToolCalls(vec![DeltaToolCall {
                    index: tool_idx as u32,
                    id: Some(format!("call_{}", Uuid::new_v4().simple())),
                    call_type: Some("function".to_string()),
                    function_name: Some(name.to_string()),
                    function_arguments: Some(String::new()),
                }]);
            }
            return ToolParserDelta::None;
        }

        // Stream arguments diff.
        let cur_arguments = current_tool_call.get("arguments");
        if let Some(cur_args_val) = cur_arguments {
            let cur_args_json = if cur_args_val.is_string() {
                cur_args_val.as_str().unwrap().to_string()
            } else {
                serde_json::to_string(cur_args_val).unwrap_or_default()
            };

            let sent = self.streamed_args_for_tool[tool_idx].len();

            // When the tool call JSON is complete, we can send the rest.
            // When incomplete, use common-prefix diffing to avoid streaming
            // close-brackets prematurely.
            let argument_diff = if last_is_complete || tool_idx < tool_call_arr.len() - 1 {
                // Complete or not the last element — safe to send remainder.
                if cur_args_json.len() > sent {
                    Some(cur_args_json[sent..].to_string())
                } else {
                    None
                }
            } else {
                // Last element, incomplete — use common prefix with prev.
                let prev_args_val = self
                    .prev_tool_call_arr
                    .get(tool_idx)
                    .and_then(|v| v.get("arguments"));
                if let Some(prev_val) = prev_args_val {
                    let prev_args_json = if prev_val.is_string() {
                        prev_val.as_str().unwrap().to_string()
                    } else {
                        serde_json::to_string(prev_val).unwrap_or_default()
                    };
                    if cur_args_json != prev_args_json {
                        let prefix_len = find_common_prefix_len(&prev_args_json, &cur_args_json);
                        if prefix_len > sent {
                            Some(cur_args_json[sent..prefix_len].to_string())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else if cur_args_json.len() > sent {
                    // No previous args — send what we have.
                    Some(cur_args_json[sent..].to_string())
                } else {
                    None
                }
            };

            if let Some(diff) = argument_diff {
                self.streamed_args_for_tool[tool_idx].push_str(&diff);
                self.prev_tool_call_arr = tool_call_arr.clone();
                return ToolParserDelta::ToolCalls(vec![DeltaToolCall {
                    index: tool_idx as u32,
                    id: None,
                    call_type: None,
                    function_name: None,
                    function_arguments: Some(diff),
                }]);
            }
        }

        self.prev_tool_call_arr = tool_call_arr.clone();
        ToolParserDelta::None
    }
}

/// Find the length of the common prefix between two strings.
fn find_common_prefix_len(a: &str, b: &str) -> usize {
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
}

// ---------------------------------------------------------------------------
// Gemma 4 tool parser
// ---------------------------------------------------------------------------

/// Gemma 4 tool-call open token (the model emits this before every call).
const GEMMA4_CALL_OPEN: &str = "<|tool_call>";
/// Gemma 4 tool-call close token.
const GEMMA4_CALL_CLOSE: &str = "<tool_call|>";
/// Literal `call:` prefix that precedes the function name.
const GEMMA4_CALL_PREFIX: &str = "call:";
/// Special token that wraps string argument values on both sides.
const GEMMA4_STR_DELIM: &str = "<|\"|>";

/// Gemma 4-style tool call parser.
///
/// Gemma 4 (and its MoE variants, e.g. `gemma-4-26b-a4b-it`) emit tool calls
/// with dedicated control tokens — NOT JSON and NOT pythonic. Taken verbatim
/// from the model's `chat_template.jinja`, one call is:
///
/// ```text
/// <|tool_call>call:NAME{key:value,key:value}<tool_call|>
/// ```
///
/// where object keys are **bare** (unquoted), string values are wrapped on both
/// sides by the `<|"|>` special token, numbers/booleans/null are bare, and
/// nested objects/arrays use `{...}` / `[...]`. Arguments are comma-separated
/// (no space) and multiple calls are emitted back-to-back with no separator.
///
/// This is neither JSON (bare keys, `<|"|>`-delimited strings) nor pythonic
/// (`{k:v}` map, not `f(k=v)`), so no existing parser can consume it — note in
/// particular that Granite's look-alike `<|tool_call|>` has pipes on **both**
/// ends and a JSON-array body.
///
/// Because the delimiters are real added tokens, `requires_special_tokens()`
/// returns `true` so the server keeps them in the detokenized text.
#[derive(Default)]
pub struct Gemma4ToolParser;

impl Gemma4ToolParser {
    pub fn new() -> Self {
        Self
    }
}

impl ToolCallParser for Gemma4ToolParser {
    fn extract_tool_calls(&self, model_output: &str) -> ExtractedToolCallInfo {
        gemma4_extract(model_output)
    }

    fn create_streaming_state(&self) -> Box<dyn StreamingToolParserState + Send> {
        Box::new(Gemma4StreamingState::new())
    }

    fn requires_special_tokens(&self) -> bool {
        true
    }
}

/// Advance `*i` past leading whitespace in `text`.
fn gemma4_skip_ws(text: &str, i: &mut usize) {
    while let Some(ch) = text[*i..].chars().next() {
        if ch.is_whitespace() {
            *i += ch.len_utf8();
        } else {
            break;
        }
    }
}

/// Parse one Gemma 4 argument value starting at `*i`, advancing `*i` past it.
///
/// Tolerant of truncation (used by the streaming path): an unterminated
/// string/object/array yields whatever parsed so far. Handles the four shapes
/// the `format_argument` jinja macro emits: `<|"|>string<|"|>`, `{obj}`,
/// `[array]`, and bare scalars (numbers, `true`/`false`, `null`). Because
/// strings are consumed as a unit between `<|"|>` delimiters, any `{ } , :`
/// inside a string never disturbs the surrounding structure.
fn gemma4_parse_value(text: &str, i: &mut usize) -> serde_json::Value {
    use serde_json::Value;
    gemma4_skip_ws(text, i);
    let rest = &text[*i..];

    // String: <|"|> ... <|"|>
    if let Some(inner) = rest.strip_prefix(GEMMA4_STR_DELIM) {
        *i += GEMMA4_STR_DELIM.len();
        if let Some(rel) = inner.find(GEMMA4_STR_DELIM) {
            let s = &text[*i..*i + rel];
            *i += rel + GEMMA4_STR_DELIM.len();
            return Value::String(s.to_string());
        }
        // Unterminated (truncated stream): take the remainder.
        let s = &text[*i..];
        *i = text.len();
        return Value::String(s.to_string());
    }

    // Object: { key:value , ... } with bare keys.
    if rest.starts_with('{') {
        *i += 1;
        let mut map = serde_json::Map::new();
        loop {
            gemma4_skip_ws(text, i);
            if *i >= text.len() {
                break;
            }
            if text[*i..].starts_with('}') {
                *i += 1;
                break;
            }
            // Bare key runs up to the ':' separator (the first ':' — the key
            // precedes any `<|"|>`-wrapped string value that may contain ':').
            let Some(colon) = text[*i..].find(':') else {
                break;
            };
            let key = text[*i..*i + colon].trim().to_string();
            *i += colon + 1;
            let value = gemma4_parse_value(text, i);
            map.insert(key, value);
            gemma4_skip_ws(text, i);
            if text[*i..].starts_with(',') {
                *i += 1;
                continue;
            }
            if text[*i..].starts_with('}') {
                *i += 1;
            }
            break;
        }
        return Value::Object(map);
    }

    // Array: [ value , ... ]
    if rest.starts_with('[') {
        *i += 1;
        let mut arr = Vec::new();
        loop {
            gemma4_skip_ws(text, i);
            if *i >= text.len() {
                break;
            }
            if text[*i..].starts_with(']') {
                *i += 1;
                break;
            }
            arr.push(gemma4_parse_value(text, i));
            gemma4_skip_ws(text, i);
            if text[*i..].starts_with(',') {
                *i += 1;
                continue;
            }
            if text[*i..].starts_with(']') {
                *i += 1;
            }
            break;
        }
        return Value::Array(arr);
    }

    // Bare scalar: read up to the next ',', '}', or ']'.
    let end = text[*i..]
        .find([',', '}', ']'])
        .map(|r| *i + r)
        .unwrap_or(text.len());
    let raw = text[*i..end].trim();
    *i = end;
    gemma4_scalar(raw)
}

/// Interpret a bare Gemma 4 scalar token as a JSON value.
fn gemma4_scalar(raw: &str) -> serde_json::Value {
    use serde_json::Value;
    match raw {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        "null" | "none" | "None" => Value::Null,
        "" => Value::Null,
        _ => {
            if let Ok(n) = raw.parse::<i64>() {
                Value::Number(n.into())
            } else if let Ok(n) = raw.parse::<u64>() {
                Value::Number(n.into())
            } else if let Ok(f) = raw.parse::<f64>() {
                serde_json::Number::from_f64(f)
                    .map(Value::Number)
                    .unwrap_or_else(|| Value::String(raw.to_string()))
            } else {
                Value::String(raw.to_string())
            }
        }
    }
}

/// Parse the `[call:]NAME{...}` body of one call starting just after a
/// `<|tool_call>` token. Returns `(name, arguments_json, end_offset)`, or
/// `None` if there's no `{` yet (truncated) or the name is empty.
fn gemma4_parse_one_call(text: &str, start: usize) -> Option<(String, String, usize)> {
    let mut i = start;
    gemma4_skip_ws(text, &mut i);
    if text[i..].starts_with(GEMMA4_CALL_PREFIX) {
        i += GEMMA4_CALL_PREFIX.len();
    }
    let brace = text[i..].find('{')?;
    let name = text[i..i + brace].trim().to_string();
    if name.is_empty() {
        return None;
    }
    i += brace; // now at '{'
    let value = gemma4_parse_value(text, &mut i);
    let arguments = match &value {
        serde_json::Value::Object(_) => serde_json::to_string(&value).ok()?,
        _ => "{}".to_string(),
    };
    Some((name, arguments, i))
}

/// Inline whitespace (space or tab only — never a newline) that the model may
/// sprinkle inside a control token when it emits the token as ordinary text
/// instead of the real special-token id, e.g. `< |channel>` for `<|channel>`.
/// Newlines are excluded so an ordinary `<`…`>` span in prose or code split
/// across a line break is never mistaken for a control token.
fn gemma4_inline_ws(c: char) -> bool {
    c == ' ' || c == '\t'
}

/// Find the next control token described by `atoms` (its bracket/pipe/word
/// pieces, in order) in `text[from..]`, tolerating runs of inline whitespace
/// *between* atoms. Returns the byte range `[start, end)` of the whole match.
///
/// This recovers the whitespace variants the model emits when it writes a
/// special token as text (`< |channel>`, `< channel|>`) rather than the exact
/// token string, which a plain `find` / `replace` misses.
fn gemma4_tolerant_find(text: &str, atoms: &[&str], from: usize) -> Option<(usize, usize)> {
    let first = atoms[0];
    let mut search = from;
    while let Some(rel) = text[search..].find(first) {
        let start = search + rel;
        let mut p = start + first.len();
        let matched = atoms[1..].iter().all(|atom| {
            while let Some(c) = text[p..].chars().next() {
                if gemma4_inline_ws(c) {
                    p += c.len_utf8();
                } else {
                    break;
                }
            }
            let hit = text[p..].starts_with(atom);
            if hit {
                p += atom.len();
            }
            hit
        });
        if matched {
            return Some((start, p));
        }
        search = start + first.len();
    }
    None
}

/// Remove Gemma structural/control tokens that can leak into content when the
/// server keeps special tokens for tool parsing. Because `gemma4` forces
/// `skip_special_tokens=false` (so `<|tool_call>` / `<|"|>` survive), the
/// terminal `<end_of_turn>` EOS — which the engine includes in the output
/// tokens (`engine_core.rs`, `new_token_ids` retains the stop token) — would
/// otherwise render into a plain-text answer. Only the tool tokens carry
/// meaning to this parser; these turn/sentinel tokens are noise in content.
///
/// Matching is whitespace-tolerant: a small quantized MoE sometimes emits these
/// markers as ordinary text with stray spaces (`< |channel>thought < channel|>`)
/// rather than the real single special-token id, which an exact match leaks.
fn gemma4_strip_control_tokens(s: &str) -> String {
    let mut out = s.to_string();
    // Gemma 4 wraps a post-tool answer as `<|channel>NAME<channel|>ANSWER`
    // (e.g. NAME = "thought"). The answer is the content after the wrapper, so
    // drop the whole `<|channel>…<channel|>` span, leaving the answer intact.
    const CHANNEL_OPEN: [&str; 4] = ["<", "|", "channel", ">"];
    const CHANNEL_CLOSE: [&str; 4] = ["<", "channel", "|", ">"];
    while let Some((open_start, open_end)) = gemma4_tolerant_find(&out, &CHANNEL_OPEN, 0) {
        match gemma4_tolerant_find(&out, &CHANNEL_CLOSE, open_end) {
            Some((_, close_end)) => out.replace_range(open_start..close_end, ""),
            None => break,
        }
    }
    // Any remaining single control tokens (a stray half of a wrapper, plus the
    // turn/sentinel tokens), each matched whitespace-tolerantly.
    const TOKENS: [&[&str]; 7] = [
        &["<", "|", "channel", ">"],
        &["<", "channel", "|", ">"],
        &["<", "end_of_turn", ">"],
        &["<", "start_of_turn", ">"],
        &["<", "eos", ">"],
        &["<", "bos", ">"],
        &["<", "pad", ">"],
    ];
    for atoms in TOKENS {
        while let Some((start, end)) = gemma4_tolerant_find(&out, atoms, 0) {
            out.replace_range(start..end, "");
        }
    }
    out
}

/// Extract tool calls from Gemma 4-formatted text (non-streaming).
fn gemma4_extract(text: &str) -> ExtractedToolCallInfo {
    let Some(first) = text.find(GEMMA4_CALL_OPEN) else {
        return ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(gemma4_strip_control_tokens(text).trim_end().to_string()),
        };
    };

    // Strip any control tokens (e.g. a leaked `<|channel>thought<channel|>`
    // reasoning cesura) that preceded the first call — not just trim — so the
    // same markers scrubbed on the no-call path can't leak ahead of a call.
    let content_before = gemma4_strip_control_tokens(&text[..first]);
    let content_before = content_before.trim();
    let content = if content_before.is_empty() {
        None
    } else {
        Some(content_before.to_string())
    };

    let mut tool_calls = Vec::new();
    let mut i = 0;
    while let Some(rel) = text[i..].find(GEMMA4_CALL_OPEN) {
        let call_start = i + rel + GEMMA4_CALL_OPEN.len();
        match gemma4_parse_one_call(text, call_start) {
            Some((name, arguments, end)) => {
                tool_calls.push(protocol::ToolCall {
                    id: format!("call_{}", Uuid::new_v4().simple()),
                    call_type: "function".to_string(),
                    function: protocol::FunctionCall { name, arguments },
                });
                i = end;
            }
            // Malformed / truncated call — stop scanning.
            None => break,
        }
    }

    if tool_calls.is_empty() {
        ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(gemma4_strip_control_tokens(text).trim_end().to_string()),
        }
    } else {
        ExtractedToolCallInfo {
            tools_called: true,
            tool_calls,
            content,
        }
    }
}

// ---------------------------------------------------------------------------
// Gemma 4 streaming state machine
// ---------------------------------------------------------------------------

struct Gemma4StreamingState {
    /// Current tool call index (-1 = no tool call started).
    current_tool_id: i32,
    /// Whether the name for the current tool has been sent.
    current_tool_name_sent: bool,
    /// Full arguments JSON already streamed for each tool call.
    streamed_args_for_tool: Vec<String>,
    /// Buffer for partial special-token tokens.
    buffer: String,
    /// Number of `<|tool_call>` open tokens seen so far.
    num_open_tags: usize,
}

impl Gemma4StreamingState {
    fn new() -> Self {
        Self {
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
            buffer: String::new(),
            num_open_tags: 0,
        }
    }
}

/// Whether `text` ends with a partial Gemma 4 special token (need more tokens
/// before we can decide what it is).
fn is_partial_gemma4_tag(text: &str) -> bool {
    for tag in [GEMMA4_CALL_OPEN, GEMMA4_CALL_CLOSE, GEMMA4_STR_DELIM] {
        for n in 1..tag.len() {
            if text.ends_with(&tag[..n]) {
                return true;
            }
        }
    }
    false
}

impl StreamingToolParserState for Gemma4StreamingState {
    fn process_delta(
        &mut self,
        _previous_text: &str,
        current_text: &str,
        delta_text: &str,
    ) -> ToolParserDelta {
        self.buffer.push_str(delta_text);
        if is_partial_gemma4_tag(&self.buffer) {
            return ToolParserDelta::None;
        }
        let text_to_process = std::mem::take(&mut self.buffer);

        let new_open_count = current_text.matches(GEMMA4_CALL_OPEN).count();
        if new_open_count == 0 {
            // No tool call yet — stream as plain content, minus any leaked
            // control token (e.g. the terminal `<end_of_turn>`). Don't trim:
            // that would drop legitimate spaces between streamed tokens.
            let cleaned = gemma4_strip_control_tokens(&text_to_process);
            if cleaned.is_empty() {
                return ToolParserDelta::None;
            }
            return ToolParserDelta::Content(cleaned);
        }

        // A new call just started.
        if new_open_count > self.num_open_tags {
            self.num_open_tags = new_open_count;
            self.current_tool_id += 1;
            self.current_tool_name_sent = false;
            self.streamed_args_for_tool.push(String::new());
        }
        if self.current_tool_id < 0 {
            return ToolParserDelta::None;
        }
        let tool_idx = self.current_tool_id as usize;

        // Text after the last <|tool_call> open token.
        let mut after = 0;
        for _ in 0..self.num_open_tags {
            if let Some(pos) = current_text[after..].find(GEMMA4_CALL_OPEN) {
                after += pos + GEMMA4_CALL_OPEN.len();
            }
        }
        let region = &current_text[after..];
        let close_pos = region.find(GEMMA4_CALL_CLOSE);
        let call_text = match close_pos {
            Some(c) => &region[..c],
            None => region,
        };

        let mut deltas = Vec::new();

        // Emit the function name once the opening '{' has arrived.
        if !self.current_tool_name_sent {
            let mut j = 0;
            gemma4_skip_ws(call_text, &mut j);
            let name_region = if call_text[j..].starts_with(GEMMA4_CALL_PREFIX) {
                &call_text[j + GEMMA4_CALL_PREFIX.len()..]
            } else {
                &call_text[j..]
            };
            if let Some(brace) = name_region.find('{') {
                let name = name_region[..brace].trim().to_string();
                if !name.is_empty() {
                    self.current_tool_name_sent = true;
                    deltas.push(DeltaToolCall {
                        index: tool_idx as u32,
                        id: Some(format!("call_{}", Uuid::new_v4().simple())),
                        call_type: Some("function".to_string()),
                        function_name: Some(name),
                        function_arguments: Some(String::new()),
                    });
                }
            }
        }

        // Emit the full arguments JSON once the call is complete (close token
        // seen). Emitting the whole object in one delta keeps the streamed
        // fragments concatenating to exactly the final JSON — incremental
        // suffix-diffing breaks when a later key reopens the closing brace.
        if self.current_tool_name_sent
            && close_pos.is_some()
            && self.streamed_args_for_tool[tool_idx].is_empty()
            && let Some(brace) = call_text.find('{')
        {
            let mut k = brace;
            let value = gemma4_parse_value(call_text, &mut k);
            if let serde_json::Value::Object(_) = value {
                let args = serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string());
                self.streamed_args_for_tool[tool_idx] = args.clone();
                deltas.push(DeltaToolCall {
                    index: tool_idx as u32,
                    id: None,
                    call_type: None,
                    function_name: None,
                    function_arguments: Some(args),
                });
            }
        }

        if deltas.is_empty() {
            ToolParserDelta::None
        } else {
            ToolParserDelta::ToolCalls(deltas)
        }
    }
}

// ---------------------------------------------------------------------------
// Qwen3-Coder tool parser
// ---------------------------------------------------------------------------

const QWEN_TC_OPEN: &str = "<tool_call>";
const QWEN_TC_CLOSE: &str = "</tool_call>";

/// Tool parser for the Qwen3-Coder / Qwen3.6 XML call format:
/// `<tool_call><function=NAME><parameter=KEY>VALUE</parameter>…</function></tool_call>`.
#[derive(Default)]
pub struct Qwen3CoderToolParser;

impl Qwen3CoderToolParser {
    pub fn new() -> Self {
        Self
    }
}

impl ToolCallParser for Qwen3CoderToolParser {
    fn extract_tool_calls(&self, model_output: &str) -> ExtractedToolCallInfo {
        qwen3_coder_extract(model_output)
    }

    fn create_streaming_state(&self) -> Box<dyn StreamingToolParserState + Send> {
        Box::new(Qwen3CoderStreamingState::default())
    }
}

/// Coerce a raw parameter string to a JSON value (numbers/bools/JSON, else string),
/// since the XML format carries no type information.
fn qwen3_coder_coerce(s: &str) -> serde_json::Value {
    match s {
        "true" => return serde_json::Value::Bool(true),
        "false" => return serde_json::Value::Bool(false),
        "null" => return serde_json::Value::Null,
        _ => {}
    }
    if let Ok(i) = s.parse::<i64>() {
        return serde_json::Value::from(i);
    }
    if let Ok(f) = s.parse::<f64>()
        && let Some(n) = serde_json::Number::from_f64(f)
    {
        return serde_json::Value::Number(n);
    }
    let looks_json =
        (s.starts_with('{') && s.ends_with('}')) || (s.starts_with('[') && s.ends_with(']'));
    if looks_json && let Ok(v) = serde_json::from_str::<serde_json::Value>(s) {
        return v;
    }
    serde_json::Value::String(s.to_string())
}

/// Parse one `<function=NAME>…</function>` block into a ToolCall.
fn qwen3_coder_parse_call(block: &str) -> Option<protocol::ToolCall> {
    let fstart = block.find("<function=")? + "<function=".len();
    let fgt = block[fstart..].find('>')? + fstart;
    let name = block[fstart..fgt].trim().to_string();
    if name.is_empty() {
        return None;
    }

    let mut args = serde_json::Map::new();
    let mut search = fgt;
    while let Some(rel) = block[search..].find("<parameter=") {
        let ps = search + rel + "<parameter=".len();
        let Some(gt) = block[ps..].find('>') else {
            break;
        };
        let key_end = ps + gt;
        let key = block[ps..key_end].trim().to_string();
        let val_start = key_end + 1;
        let (val_end, next) = match block[val_start..].find("</parameter>") {
            Some(p) => (val_start + p, val_start + p + "</parameter>".len()),
            None => (block.len(), block.len()),
        };
        // Values are commonly wrapped in leading/trailing newlines.
        let raw = block[val_start..val_end].trim_matches(['\n', '\r']).trim();
        if !key.is_empty() {
            args.insert(key, qwen3_coder_coerce(raw));
        }
        search = next;
    }

    Some(protocol::ToolCall {
        id: format!("call_{}", Uuid::new_v4().simple()),
        call_type: "function".to_string(),
        function: protocol::FunctionCall {
            name,
            arguments: serde_json::Value::Object(args).to_string(),
        },
    })
}

/// Extract Qwen3-Coder tool calls; only closed `<tool_call>…</tool_call>` blocks count.
fn qwen3_coder_extract(text: &str) -> ExtractedToolCallInfo {
    let Some(first) = text.find(QWEN_TC_OPEN) else {
        return ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(text.to_string()),
        };
    };
    let before = text[..first].trim();
    let content = (!before.is_empty()).then(|| before.to_string());

    let mut tool_calls = Vec::new();
    let mut search = 0;
    while let Some(rel) = text[search..].find(QWEN_TC_OPEN) {
        let block_start = search + rel + QWEN_TC_OPEN.len();
        // Some Qwen3-Coder outputs omit the closing `</tool_call>` (or `</function>`)
        // on the final call, so take the block up to the next `<tool_call>` or the
        // end of the text rather than dropping the call.
        let (block_end, next) = match text[block_start..].find(QWEN_TC_CLOSE) {
            Some(p) => (block_start + p, block_start + p + QWEN_TC_CLOSE.len()),
            None => match text[block_start..].find(QWEN_TC_OPEN) {
                Some(p) => (block_start + p, block_start + p),
                None => (text.len(), text.len()),
            },
        };
        if let Some(tc) = qwen3_coder_parse_call(&text[block_start..block_end]) {
            tool_calls.push(tc);
        }
        if next <= search {
            break;
        }
        search = next;
    }

    if tool_calls.is_empty() {
        ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(text.to_string()),
        }
    } else {
        ExtractedToolCallInfo {
            tools_called: true,
            tool_calls,
            content,
        }
    }
}

/// Streaming state: emit leading content, then each tool call whole once its
/// closing tag arrives (recompute-and-diff — simple and correct for small calls).
#[derive(Default)]
struct Qwen3CoderStreamingState {
    content_emitted: usize,
    calls_emitted: usize,
}

impl StreamingToolParserState for Qwen3CoderStreamingState {
    fn process_delta(
        &mut self,
        _previous_text: &str,
        current_text: &str,
        _delta_text: &str,
    ) -> ToolParserDelta {
        let head_end = current_text
            .find(QWEN_TC_OPEN)
            .unwrap_or(current_text.len());
        if self.content_emitted < head_end {
            let new = current_text[self.content_emitted..head_end].to_string();
            self.content_emitted = head_end;
            if !new.is_empty() {
                return ToolParserDelta::Content(new);
            }
        }
        let extracted = qwen3_coder_extract(current_text);
        if extracted.tool_calls.len() > self.calls_emitted {
            let deltas = extracted.tool_calls[self.calls_emitted..]
                .iter()
                .enumerate()
                .map(|(i, tc)| DeltaToolCall {
                    index: (self.calls_emitted + i) as u32,
                    id: Some(tc.id.clone()),
                    call_type: Some("function".to_string()),
                    function_name: Some(tc.function.name.clone()),
                    function_arguments: Some(tc.function.arguments.clone()),
                })
                .collect();
            self.calls_emitted = extracted.tool_calls.len();
            return ToolParserDelta::ToolCalls(deltas);
        }
        ToolParserDelta::None
    }
}

// ---------------------------------------------------------------------------
// Parser registry
// ---------------------------------------------------------------------------

/// Get a tool call parser by name.
pub fn get_tool_parser(name: &str) -> Result<Arc<dyn ToolCallParser>, String> {
    match name {
        "hermes" => Ok(Arc::new(HermesToolParser::new())),
        "qwen3_coder" => Ok(Arc::new(Qwen3CoderToolParser::new())),
        "llama3_json" | "llama4_json" => Ok(Arc::new(LlamaJsonToolParser::new())),
        "kimi_k2" => Ok(Arc::new(KimiK2ToolParser::new())),
        "mistral" => Ok(Arc::new(MistralToolParser::new())),
        "jamba" => Ok(Arc::new(JambaToolParser::new())),
        "granite" => Ok(Arc::new(GraniteToolParser::new())),
        // Gemma 4 and its MoE variants share the same `<|tool_call>` format;
        // accept the bare `gemma` alias too.
        "gemma4" | "gemma" => Ok(Arc::new(Gemma4ToolParser::new())),
        other => Err(format!("Unknown tool call parser: {other}")),
    }
}

/// Best-effort tool-call parser selection from the model's **canonical
/// architecture** (`config.json` `model_type` / `architectures`) — NOT its
/// name or path. The tool-call format is a property of the model family/arch,
/// so keying off a quant's filename is fragile; the arch is authoritative.
/// Returns one of the names understood by [`get_tool_parser`], or `None`.
///
/// Note: architecture alone can't separate a finetune that changes the tool
/// format from its base (e.g. Hermes-on-Llama, or Qwen3-Coder-30B which shares
/// `qwen3_moe` with plain Qwen3-MoE); those need an explicit `--tool-call-parser`.
pub fn detect_tool_parser(arch: &str) -> Option<&'static str> {
    let a = arch.to_lowercase();
    if a.contains("kimi") {
        Some("kimi_k2")
    } else if a.contains("granite") {
        Some("granite")
    } else if a.contains("jamba") {
        Some("jamba")
    } else if a.contains("gemma") {
        // Gemma 4's `<|tool_call>` format (gemma ≤3 has no native tool format).
        Some("gemma4")
    } else if a.contains("mistral")
        || a.contains("mixtral")
        || a.contains("ministral")
        || a.contains("magistral")
        || a.contains("devstral")
    {
        Some("mistral")
    } else if a.contains("llama") {
        Some("llama3_json")
    } else if a.contains("qwen3_5") || a.contains("qwen3.5") {
        // Qwen3.5 / 3.6 (`qwen3_5_moe`) emit the XML `<function=…><parameter=…>` format.
        Some("qwen3_coder")
    } else if a.contains("qwen") {
        // Qwen (≤3) emits `<tool_call>{json}</tool_call>`.
        Some("hermes")
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Parser auto-detection from model id --

    #[test]
    fn detects_common_families() {
        assert_eq!(
            detect_tool_parser("meta-llama/Llama-3.2-1B-Instruct"),
            Some("llama3_json")
        );
        assert_eq!(
            detect_tool_parser("unsloth/Llama-3.2-1B-Instruct"),
            Some("llama3_json")
        );
        assert_eq!(
            detect_tool_parser("Qwen/Qwen2.5-7B-Instruct"),
            Some("hermes")
        );
        assert_eq!(
            detect_tool_parser("mistralai/Mistral-7B-Instruct-v0.3"),
            Some("mistral")
        );
        assert_eq!(
            detect_tool_parser("moonshotai/Kimi-K2-Instruct"),
            Some("kimi_k2")
        );
        assert_eq!(
            detect_tool_parser("ibm-granite/granite-3.1-8b-instruct"),
            Some("granite")
        );
        assert_eq!(
            detect_tool_parser("mlx-community/gemma-4-26b-a4b-it-4bit"),
            Some("gemma4")
        );
        assert_eq!(detect_tool_parser("google/gemma-4-12b-it"), Some("gemma4"));
    }

    #[test]
    fn unknown_family_is_none() {
        assert_eq!(detect_tool_parser("some-random/exotic-model"), None);
    }

    // -- Hermes non-streaming tests --

    #[test]
    fn test_hermes_single_tool_call() {
        let parser = HermesToolParser::new();
        let output = r#"<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
        assert_eq!(result.tool_calls[0].function.arguments, r#"{"city":"SF"}"#);
        assert_eq!(result.tool_calls[0].call_type, "function");
        assert!(result.tool_calls[0].id.starts_with("call_"));
        assert!(result.content.is_none());
    }

    #[test]
    fn test_hermes_two_tool_calls() {
        let parser = HermesToolParser::new();
        let output = r#"<tool_call>{"name":"search","arguments":{"q":"rust"}}</tool_call><tool_call>{"name":"fetch","arguments":{"url":"https://example.com"}}</tool_call>"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].function.name, "search");
        assert_eq!(result.tool_calls[1].function.name, "fetch");
    }

    #[test]
    fn test_hermes_text_before_tool_call() {
        let parser = HermesToolParser::new();
        let output = r#"I'll help you with that.
<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.content.unwrap(), "I'll help you with that.");
    }

    #[test]
    fn test_hermes_no_tool_call() {
        let parser = HermesToolParser::new();
        let output = "The weather in SF is sunny and 72°F.";
        let result = parser.extract_tool_calls(output);
        assert!(!result.tools_called);
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.content.unwrap(), output);
    }

    #[test]
    fn test_hermes_malformed_json() {
        let parser = HermesToolParser::new();
        let output = r#"<tool_call>not json at all</tool_call>"#;
        let result = parser.extract_tool_calls(output);
        assert!(!result.tools_called);
        assert!(result.tool_calls.is_empty());
        // Falls back to treating it as content.
        assert!(result.content.is_some());
    }

    #[test]
    fn test_hermes_unclosed_tag() {
        let parser = HermesToolParser::new();
        let output = r#"<tool_call>{"name":"f","arguments":{"a":1}}"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "f");
    }

    // -- LLaMA JSON non-streaming tests --

    #[test]
    fn test_llama_single_json() {
        let parser = LlamaJsonToolParser::new();
        let output = r#"{"name":"get_weather","arguments":{"city":"SF"}}"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_llama_two_json_objects() {
        let parser = LlamaJsonToolParser::new();
        let output =
            r#"{"name":"search","arguments":{"q":"rust"}}{"name":"fetch","arguments":{"url":"x"}}"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].function.name, "search");
        assert_eq!(result.tool_calls[1].function.name, "fetch");
    }

    #[test]
    fn test_llama_python_tag() {
        let parser = LlamaJsonToolParser::new();
        let output = r#"<|python_tag|>{"name":"get_weather","arguments":{"city":"SF"}}"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_llama_plain_text() {
        let parser = LlamaJsonToolParser::new();
        let output = "Just a plain text response with no JSON.";
        let result = parser.extract_tool_calls(output);
        assert!(!result.tools_called);
        assert!(result.content.is_some());
    }

    #[test]
    fn test_llama_parameters_alias() {
        // Llama 3.x tool calls use `parameters` with no `arguments` key.
        let parser = LlamaJsonToolParser::new();
        let output = r#"{"name":"shell","parameters":{"command":"echo hi"}}"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "shell");
        assert_eq!(
            result.tool_calls[0].function.arguments,
            r#"{"command":"echo hi"}"#
        );
    }

    // -- Partial JSON tests --

    #[test]
    fn test_partial_json_incomplete() {
        let input = r#"{"name": "f", "arguments": {"q": "he"#;
        let result = partial_json_parse(input);
        assert!(result.is_some());
        let val = result.unwrap();
        assert_eq!(val["name"], "f");
        assert_eq!(val["arguments"]["q"], "he");
    }

    #[test]
    fn test_partial_json_complete() {
        let input = r#"{"name": "f", "arguments": {"q": "hello"}}"#;
        let result = partial_json_parse(input);
        assert!(result.is_some());
        let val = result.unwrap();
        assert_eq!(val["name"], "f");
    }

    // -- Hermes streaming tests --

    #[test]
    fn test_hermes_streaming_basic() {
        let parser = HermesToolParser::new();
        let mut state = parser.create_streaming_state();

        // Simulate tokens for: <tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>
        let tokens = vec![
            "<tool_call>",
            r#"{"name":"#,
            r#""get_weather","#,
            r#""arguments":{"#,
            r#""city":"SF"#,
            r#"}}"#,
            "</tool_call>",
        ];

        let mut accumulated = String::new();
        let mut got_name = false;
        let mut got_args = false;

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        got_name = true;
                        assert_eq!(name, "get_weather");
                        assert_eq!(call.index, 0);
                        assert!(call.id.is_some());
                    }
                    if let Some(args) = &call.function_arguments
                        && !args.is_empty()
                    {
                        got_args = true;
                    }
                }
            }
        }

        assert!(got_name, "Should have received tool name");
        assert!(got_args, "Should have received argument fragments");
    }

    #[test]
    fn test_hermes_streaming_content_then_tool() {
        let parser = HermesToolParser::new();
        let mut state = parser.create_streaming_state();

        let mut accumulated = String::new();
        let mut got_content = false;
        let mut got_tool = false;

        // First, some content.
        let prev = accumulated.clone();
        accumulated.push_str("Sure, ");
        if let ToolParserDelta::Content(text) = state.process_delta(&prev, &accumulated, "Sure, ") {
            assert_eq!(text, "Sure, ");
            got_content = true;
        }

        // Then tool call.
        let tokens = vec![
            "<tool_call>",
            r#"{"name":"f","arguments":{}}"#,
            "</tool_call>",
        ];
        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                got_tool = true;
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        assert_eq!(name, "f");
                    }
                }
            }
        }

        assert!(got_content, "Should have received content");
        assert!(got_tool, "Should have received tool call");
    }

    // -- Registry tests --

    #[test]
    fn test_registry_hermes() {
        assert!(get_tool_parser("hermes").is_ok());
    }

    #[test]
    fn test_registry_llama3_json() {
        assert!(get_tool_parser("llama3_json").is_ok());
    }

    #[test]
    fn test_registry_llama4_json() {
        assert!(get_tool_parser("llama4_json").is_ok());
    }

    #[test]
    fn test_registry_kimi_k2() {
        assert!(get_tool_parser("kimi_k2").is_ok());
    }

    #[test]
    fn test_registry_unknown() {
        assert!(get_tool_parser("unknown").is_err());
    }

    // -- Kimi K2 non-streaming tests --

    #[test]
    fn test_kimi_k2_single_tool_call() {
        let parser = KimiK2ToolParser::new();
        let output = "<|tool_calls_section_begin|>\n<|tool_call_begin|> functions.get_weather:0 <|tool_call_argument_begin|> {\"city\": \"SF\"} <|tool_call_end|>\n<|tool_calls_section_end|>";
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
        assert_eq!(
            result.tool_calls[0].function.arguments,
            "{\"city\": \"SF\"}"
        );
        assert_eq!(result.tool_calls[0].call_type, "function");
        assert!(result.tool_calls[0].id.starts_with("call_"));
        assert!(result.content.is_none());
    }

    #[test]
    fn test_kimi_k2_two_tool_calls() {
        let parser = KimiK2ToolParser::new();
        let output = "<|tool_calls_section_begin|>\n<|tool_call_begin|> functions.search:0 <|tool_call_argument_begin|> {\"q\": \"rust\"} <|tool_call_end|>\n<|tool_call_begin|> functions.fetch:1 <|tool_call_argument_begin|> {\"url\": \"https://example.com\"} <|tool_call_end|>\n<|tool_calls_section_end|>";
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].function.name, "search");
        assert_eq!(result.tool_calls[1].function.name, "fetch");
    }

    #[test]
    fn test_kimi_k2_text_before_section() {
        let parser = KimiK2ToolParser::new();
        let output = "I'll check the weather for you.\n<|tool_calls_section_begin|>\n<|tool_call_begin|> functions.get_weather:0 <|tool_call_argument_begin|> {\"city\": \"SF\"} <|tool_call_end|>\n<|tool_calls_section_end|>";
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.content.unwrap(), "I'll check the weather for you.");
    }

    #[test]
    fn test_kimi_k2_no_tool_call() {
        let parser = KimiK2ToolParser::new();
        let output = "The weather in SF is sunny and 72°F.";
        let result = parser.extract_tool_calls(output);
        assert!(!result.tools_called);
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.content.unwrap(), output);
    }

    #[test]
    fn test_kimi_k2_singular_section_markers() {
        // Support both singular and plural section markers.
        let parser = KimiK2ToolParser::new();
        let output = "<|tool_call_section_begin|>\n<|tool_call_begin|> functions.get_weather:0 <|tool_call_argument_begin|> {\"city\": \"SF\"} <|tool_call_end|>\n<|tool_call_section_end|>";
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_kimi_k2_function_name_parsing() {
        assert_eq!(
            kimi_k2_parse_function_name("functions.get_weather:0"),
            "get_weather"
        );
        assert_eq!(kimi_k2_parse_function_name("functions.search:1"), "search");
        assert_eq!(kimi_k2_parse_function_name("get_weather:0"), "get_weather");
        assert_eq!(kimi_k2_parse_function_name("get_weather"), "get_weather");
        assert_eq!(kimi_k2_parse_function_name("a.b.c:2"), "c");
    }

    // -- Kimi K2 streaming tests --

    #[test]
    fn test_kimi_k2_streaming_basic() {
        let parser = KimiK2ToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = vec![
            "<|tool_calls_section_begin|>",
            "\n",
            "<|tool_call_begin|>",
            " functions.get_weather:0 ",
            "<|tool_call_argument_begin|>",
            " {\"city\":",
            " \"SF\"}",
            " <|tool_call_end|>",
            "\n",
            "<|tool_calls_section_end|>",
        ];

        let mut accumulated = String::new();
        let mut got_name = false;
        let mut got_args = false;

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        got_name = true;
                        assert_eq!(name, "get_weather");
                        assert_eq!(call.index, 0);
                        assert!(call.id.is_some());
                    }
                    if let Some(args) = &call.function_arguments
                        && !args.is_empty()
                    {
                        got_args = true;
                    }
                }
            }
        }

        assert!(got_name, "Should have received tool name");
        assert!(got_args, "Should have received argument fragments");
    }

    #[test]
    fn test_kimi_k2_streaming_content_then_tool() {
        let parser = KimiK2ToolParser::new();
        let mut state = parser.create_streaming_state();

        let mut accumulated = String::new();
        let mut got_content = false;
        let mut got_tool = false;

        // Content before tools.
        let prev = accumulated.clone();
        accumulated.push_str("Let me check. ");
        if let ToolParserDelta::Content(text) =
            state.process_delta(&prev, &accumulated, "Let me check. ")
        {
            assert_eq!(text, "Let me check. ");
            got_content = true;
        }

        // Tool section.
        let tokens = vec![
            "<|tool_calls_section_begin|>",
            "\n<|tool_call_begin|>",
            " functions.f:0 ",
            "<|tool_call_argument_begin|>",
            " {}",
            " <|tool_call_end|>",
            "\n<|tool_calls_section_end|>",
        ];
        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                got_tool = true;
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        assert_eq!(name, "f");
                    }
                }
            }
        }

        assert!(got_content, "Should have received content");
        assert!(got_tool, "Should have received tool call");
    }

    // -- Mistral non-streaming tests --

    #[test]
    fn test_registry_mistral() {
        assert!(get_tool_parser("mistral").is_ok());
    }

    #[test]
    fn test_registry_jamba() {
        assert!(get_tool_parser("jamba").is_ok());
    }

    #[test]
    fn test_mistral_id_format() {
        let id = mistral_generate_id();
        assert_eq!(id.len(), 9);
        assert!(id.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn test_mistral_no_tools() {
        let parser = MistralToolParser::new();
        let output = "The weather is sunny today.";
        let result = parser.extract_tool_calls(output);
        assert!(!result.tools_called);
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.content.unwrap(), output);
    }

    #[test]
    fn test_mistral_single_tool_v11() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS]get_weather{"city":"SF"}"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
        assert_eq!(result.tool_calls[0].function.arguments, r#"{"city":"SF"}"#);
        assert_eq!(result.tool_calls[0].call_type, "function");
        assert_eq!(result.tool_calls[0].id.len(), 9);
        assert!(result.content.is_none());
    }

    #[test]
    fn test_mistral_single_tool_pre_v11() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"SF"}}]"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
        assert_eq!(result.tool_calls[0].function.arguments, r#"{"city":"SF"}"#);
    }

    #[test]
    fn test_mistral_multiple_tools_v11() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS]get_weather{"city":"SF"}[TOOL_CALLS]search{"q":"rust"}"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
        assert_eq!(result.tool_calls[1].function.name, "search");
        assert_eq!(result.tool_calls[1].function.arguments, r#"{"q":"rust"}"#);
    }

    #[test]
    fn test_mistral_multiple_tools_pre_v11() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"SF"}},{"name":"search","arguments":{"q":"rust"}}]"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
        assert_eq!(result.tool_calls[1].function.name, "search");
    }

    #[test]
    fn test_mistral_content_before_tools() {
        let parser = MistralToolParser::new();
        let output = r#"Let me help you.[TOOL_CALLS]get_weather{"city":"SF"}"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.content.unwrap(), "Let me help you.");
    }

    #[test]
    fn test_mistral_complex_arguments() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS]create_event{"title":"Meeting","nested":{"key":"val\"ue"},"list":[1,2,3]}"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls[0].function.name, "create_event");
        assert!(result.tool_calls[0].function.arguments.contains("nested"));
    }

    #[test]
    fn test_mistral_pre_v11_malformed_json_fallback() {
        let parser = MistralToolParser::new();
        // Malformed JSON with extra text — the `[{...}]` pattern should be found by fallback
        let output = r#"[TOOL_CALLS] [{"name":"f","arguments":{"a":1}}] extra text"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "f");
    }

    #[test]
    fn test_mistral_pre_v11_arguments_before_name() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS] [{"arguments":{"city":"SF"},"name":"get_weather"}]"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
        assert_eq!(result.tool_calls[0].function.arguments, r#"{"city":"SF"}"#);
    }

    // -- Mistral streaming tests --

    #[test]
    fn test_mistral_streaming_no_tools() {
        let parser = MistralToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = vec!["Hello", " world", "!"];
        let mut accumulated = String::new();

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            match state.process_delta(&prev, &accumulated, token) {
                ToolParserDelta::Content(c) => assert_eq!(c, token),
                other => panic!("Expected Content, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_mistral_streaming_single_tool_v11() {
        let parser = MistralToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = vec!["[TOOL_CALLS]", "get_weather", r#"{"city":"#, r#""SF"}"#];
        let mut accumulated = String::new();
        let mut got_name = false;
        let mut args = String::new();

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        got_name = true;
                        assert_eq!(name, "get_weather");
                        assert_eq!(call.index, 0);
                        assert!(call.id.is_some());
                    }
                    if let Some(a) = &call.function_arguments {
                        args.push_str(a);
                    }
                }
            }
        }

        assert!(got_name, "Should have received function name");
        assert_eq!(args, r#"{"city":"SF"}"#);
    }

    #[test]
    fn test_mistral_streaming_multiple_tools_v11() {
        let parser = MistralToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = vec![
            "[TOOL_CALLS]",
            "get_weather",
            r#"{"city":"SF"}"#,
            "[TOOL_CALLS]",
            "search",
            r#"{"q":"rust"}"#,
        ];
        let mut accumulated = String::new();
        let mut names = Vec::new();

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        names.push(name.clone());
                    }
                }
            }
        }

        assert_eq!(names, vec!["get_weather", "search"]);
    }

    #[test]
    fn test_mistral_streaming_content_then_tool() {
        let parser = MistralToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = vec!["Sure!", "[TOOL_CALLS]", "f", r#"{"a":1}"#];
        let mut accumulated = String::new();
        let mut got_content = false;
        let mut got_tool = false;

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            match state.process_delta(&prev, &accumulated, token) {
                ToolParserDelta::Content(c) => {
                    got_content = true;
                    assert_eq!(c, "Sure!");
                }
                ToolParserDelta::ToolCalls(calls) => {
                    for call in &calls {
                        if call.function_name.is_some() {
                            got_tool = true;
                        }
                    }
                }
                _ => {}
            }
        }

        assert!(got_content);
        assert!(got_tool);
    }

    #[test]
    fn test_mistral_streaming_single_tool_pre_v11() {
        let parser = MistralToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = vec![
            "[TOOL_CALLS]",
            r#" [{"name"#,
            r#"": "get_weather", "arguments": {"#,
            r#""city": "SF""#,
            "}",
            "}]",
        ];
        let mut accumulated = String::new();
        let mut got_name = false;
        let mut got_args = false;

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        got_name = true;
                        assert_eq!(name, "get_weather");
                    }
                    if call.function_arguments.is_some() {
                        got_args = true;
                    }
                }
            }
        }

        assert!(got_name, "Should have received function name");
        assert!(got_args, "Should have received arguments");
    }

    #[test]
    fn test_mistral_streaming_multiple_tools_pre_v11() {
        let parser = MistralToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = vec![
            r#"[TOOL_CALLS] [{"name": "f1", "arguments": {"a": 1}}, {"name": "f2", "arguments": {"b": 2}}]"#,
        ];
        let mut accumulated = String::new();
        let mut names = Vec::new();

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        names.push(name.clone());
                    }
                }
            }
        }

        assert_eq!(names, vec!["f1", "f2"]);
    }

    #[test]
    fn test_mistral_streaming_one_chunk_v11() {
        let parser = MistralToolParser::new();
        let mut state = parser.create_streaming_state();

        let full = r#"[TOOL_CALLS]get_weather{"city":"SF"}"#;
        let mut got_name = false;

        match state.process_delta("", full, full) {
            ToolParserDelta::ToolCalls(calls) => {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        got_name = true;
                        assert_eq!(name, "get_weather");
                    }
                }
            }
            other => panic!("Expected ToolCalls, got {:?}", other),
        }

        assert!(got_name);
    }

    // -- Jamba non-streaming tests --

    #[test]
    fn test_jamba_single_tool_call() {
        let parser = JambaToolParser::new();
        let output =
            r#"<tool_calls>[{"name":"get_weather","arguments":{"city":"SF"}}]</tool_calls>"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
        assert_eq!(result.tool_calls[0].function.arguments, r#"{"city":"SF"}"#);
        assert_eq!(result.tool_calls[0].call_type, "function");
        assert!(result.tool_calls[0].id.starts_with("call_"));
        assert!(result.content.is_none());
    }

    #[test]
    fn test_jamba_two_tool_calls() {
        let parser = JambaToolParser::new();
        let output = r#"<tool_calls>[{"name":"search","arguments":{"q":"rust"}},{"name":"fetch","arguments":{"url":"https://example.com"}}]</tool_calls>"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].function.name, "search");
        assert_eq!(result.tool_calls[1].function.name, "fetch");
    }

    #[test]
    fn test_jamba_text_before_tool_calls() {
        let parser = JambaToolParser::new();
        let output = r#"I'll help you with that.
<tool_calls>[{"name":"get_weather","arguments":{"city":"SF"}}]</tool_calls>"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.content.unwrap(), "I'll help you with that.");
    }

    #[test]
    fn test_jamba_no_tool_call() {
        let parser = JambaToolParser::new();
        let output = "The weather in SF is sunny and 72°F.";
        let result = parser.extract_tool_calls(output);
        assert!(!result.tools_called);
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.content.unwrap(), output);
    }

    #[test]
    fn test_jamba_malformed_json() {
        let parser = JambaToolParser::new();
        let output = r#"<tool_calls>not json at all</tool_calls>"#;
        let result = parser.extract_tool_calls(output);
        assert!(!result.tools_called);
        assert!(result.tool_calls.is_empty());
        assert!(result.content.is_some());
    }

    #[test]
    fn test_jamba_unclosed_tag() {
        let parser = JambaToolParser::new();
        let output = r#"<tool_calls>[{"name":"f","arguments":{"a":1}}]"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "f");
    }

    #[test]
    fn test_jamba_string_arguments() {
        let parser = JambaToolParser::new();
        let output = r#"<tool_calls>[{"name":"f","arguments":"{\"a\":1}"}]</tool_calls>"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.arguments, r#"{"a":1}"#);
    }

    // -- Jamba streaming tests --

    #[test]
    fn test_jamba_streaming_content_then_tool() {
        let parser = JambaToolParser::new();
        let mut state = parser.create_streaming_state();

        // First token: content.
        let d1 = state.process_delta("", "Hello", "Hello");
        assert!(matches!(d1, ToolParserDelta::Content(ref s) if s == "Hello"));

        // Open tag token.
        let d2 = state.process_delta("Hello", "Hello<tool_calls>", "<tool_calls>");
        assert!(matches!(d2, ToolParserDelta::None));

        // Start of JSON array with tool name.
        let d3 = state.process_delta(
            "Hello<tool_calls>",
            r#"Hello<tool_calls>[{"name":"get_weather""#,
            r#"[{"name":"get_weather""#,
        );
        // Should get the name.
        match d3 {
            ToolParserDelta::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function_name.as_deref(), Some("get_weather"));
                assert!(calls[0].id.is_some());
            }
            other => panic!("Expected ToolCalls with name, got {:?}", other),
        }

        // Stream arguments.
        let d4 = state.process_delta(
            r#"Hello<tool_calls>[{"name":"get_weather""#,
            r#"Hello<tool_calls>[{"name":"get_weather","arguments":{"city":"SF"#,
            r#","arguments":{"city":"SF"#,
        );
        match d4 {
            ToolParserDelta::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert!(calls[0].function_arguments.is_some());
            }
            other => panic!("Expected ToolCalls with args, got {:?}", other),
        }
    }

    #[test]
    fn test_jamba_streaming_two_tools() {
        let parser = JambaToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = [
            "<tool_calls>",
            r#"[{"name":"f1""#,
            r#","arguments":{"a":1}}"#,
            r#",{"name":"f2""#,
            r#","arguments":{"b":2}}"#,
            "]</tool_calls>",
        ];

        let mut accumulated = String::new();
        let mut names = Vec::new();

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        names.push(name.clone());
                    }
                }
            }
        }

        assert_eq!(names, vec!["f1", "f2"]);
    }

    // -- Granite non-streaming tests --

    #[test]
    fn test_granite_single_tool_call_30() {
        // Granite 3.0 format: <|tool_call|> prefix.
        let parser = GraniteToolParser::new();
        let output = r#"<|tool_call|>[{"name":"get_weather","arguments":{"city":"SF"}}]"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
        assert_eq!(result.tool_calls[0].function.arguments, r#"{"city":"SF"}"#);
        assert_eq!(result.tool_calls[0].call_type, "function");
        assert!(result.tool_calls[0].id.starts_with("call_"));
        assert!(result.content.is_none());
    }

    #[test]
    fn test_granite_single_tool_call_31() {
        // Granite 3.1 format: <tool_call> prefix.
        let parser = GraniteToolParser::new();
        let output = r#"<tool_call>[{"name":"get_weather","arguments":{"city":"SF"}}]"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_granite_two_tool_calls() {
        let parser = GraniteToolParser::new();
        let output = r#"<|tool_call|>[{"name":"search","arguments":{"q":"rust"}},{"name":"fetch","arguments":{"url":"https://example.com"}}]"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].function.name, "search");
        assert_eq!(result.tool_calls[1].function.name, "fetch");
    }

    #[test]
    fn test_granite_no_tool_call() {
        let parser = GraniteToolParser::new();
        let output = "The weather in SF is sunny and 72°F.";
        let result = parser.extract_tool_calls(output);
        assert!(!result.tools_called);
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.content.unwrap(), output);
    }

    #[test]
    fn test_granite_malformed_json() {
        let parser = GraniteToolParser::new();
        let output = r#"<|tool_call|>not json"#;
        let result = parser.extract_tool_calls(output);
        assert!(!result.tools_called);
        assert!(result.content.is_some());
    }

    #[test]
    fn test_granite_whitespace_between_prefix_and_array() {
        let parser = GraniteToolParser::new();
        let output = "  <|tool_call|>  [  {\"name\":\"f\",\"arguments\":{\"a\":1}}  ]";
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "f");
    }

    #[test]
    fn test_granite_both_prefixes() {
        // Both prefixes present (unlikely but handled).
        let parser = GraniteToolParser::new();
        let output = r#"<|tool_call|><tool_call>[{"name":"f","arguments":{}}]"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
    }

    #[test]
    fn test_registry_granite() {
        assert!(get_tool_parser("granite").is_ok());
    }

    // -- Granite streaming tests --

    #[test]
    fn test_granite_streaming_basic() {
        let parser = GraniteToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = [
            "<|tool_call|>",
            r#"[{"name":"get_weather""#,
            r#","arguments":{"city":"SF"}}]"#,
        ];

        let mut accumulated = String::new();
        let mut got_name = false;
        let mut got_args = false;

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        assert_eq!(name, "get_weather");
                        got_name = true;
                    }
                    if let Some(args) = &call.function_arguments
                        && !args.is_empty()
                    {
                        got_args = true;
                    }
                }
            }
        }

        assert!(got_name);
        assert!(got_args);
    }

    #[test]
    fn test_granite_streaming_no_tool() {
        let parser = GraniteToolParser::new();
        let mut state = parser.create_streaming_state();

        let d = state.process_delta("", "Hello world", "Hello world");
        assert!(matches!(d, ToolParserDelta::Content(ref s) if s == "Hello world"));
    }

    #[test]
    fn test_granite_streaming_two_tools() {
        let parser = GraniteToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = [
            "<|tool_call|>",
            r#"[{"name":"f1""#,
            r#","arguments":{"a":1}}"#,
            r#",{"name":"f2""#,
            r#","arguments":{"b":2}}"#,
            "]",
        ];

        let mut accumulated = String::new();
        let mut names = Vec::new();

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        names.push(name.clone());
                    }
                }
            }
        }

        assert_eq!(names, vec!["f1", "f2"]);
    }

    #[test]
    fn test_granite_streaming_31_prefix() {
        // Granite 3.1 uses <tool_call> instead of <|tool_call|>.
        let parser = GraniteToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = [
            "<tool_call>",
            r#"[{"name":"f""#,
            r#","arguments":{"x":1}}]"#,
        ];

        let mut accumulated = String::new();
        let mut got_name = false;

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if call.function_name.as_deref() == Some("f") {
                        got_name = true;
                    }
                }
            }
        }

        assert!(got_name);
    }

    // -- Gemma 4 non-streaming tests --

    fn gemma4_args(tc: &crate::protocol::ToolCall) -> serde_json::Value {
        serde_json::from_str(&tc.function.arguments).expect("arguments must be valid JSON")
    }

    #[test]
    fn test_gemma4_single_string_arg() {
        let parser = Gemma4ToolParser::new();
        let out = "<|tool_call>call:get_current_weather{location:<|\"|>London<|\"|>}<tool_call|>";
        let r = parser.extract_tool_calls(out);
        assert!(r.tools_called);
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].function.name, "get_current_weather");
        assert_eq!(
            gemma4_args(&r.tool_calls[0]),
            serde_json::json!({"location": "London"})
        );
        assert!(r.content.is_none());
    }

    #[test]
    fn test_gemma4_scalar_and_bool_args() {
        let parser = Gemma4ToolParser::new();
        let out =
            "<|tool_call>call:set_temp{room:<|\"|>bedroom<|\"|>,degrees:22,eco:true}<tool_call|>";
        let r = parser.extract_tool_calls(out);
        assert!(r.tools_called);
        assert_eq!(
            gemma4_args(&r.tool_calls[0]),
            serde_json::json!({"room": "bedroom", "degrees": 22, "eco": true})
        );
    }

    #[test]
    fn test_gemma4_nested_object_and_array() {
        let parser = Gemma4ToolParser::new();
        let out = "<|tool_call>call:f{filter:{tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>],min:3},flag:false}<tool_call|>";
        let r = parser.extract_tool_calls(out);
        assert!(r.tools_called);
        assert_eq!(
            gemma4_args(&r.tool_calls[0]),
            serde_json::json!({"filter": {"tags": ["a", "b"], "min": 3}, "flag": false})
        );
    }

    #[test]
    fn test_gemma4_string_with_structural_chars_inside() {
        // A string value containing commas, braces, brackets and colons must
        // survive verbatim: the <|"|> token delimits it, so inner punctuation
        // is never treated as structure.
        let parser = Gemma4ToolParser::new();
        let out = "<|tool_call>call:say{msg:<|\"|>a, b: {c},[d]<|\"|>}<tool_call|>";
        let r = parser.extract_tool_calls(out);
        assert!(r.tools_called);
        assert_eq!(
            gemma4_args(&r.tool_calls[0]),
            serde_json::json!({"msg": "a, b: {c},[d]"})
        );
    }

    #[test]
    fn test_gemma4_empty_args() {
        let parser = Gemma4ToolParser::new();
        let out = "<|tool_call>call:now{}<tool_call|>";
        let r = parser.extract_tool_calls(out);
        assert!(r.tools_called);
        assert_eq!(r.tool_calls[0].function.name, "now");
        assert_eq!(gemma4_args(&r.tool_calls[0]), serde_json::json!({}));
    }

    #[test]
    fn test_gemma4_multiple_consecutive_calls() {
        let parser = Gemma4ToolParser::new();
        let out = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>\
                   <|tool_call>call:get_time{tz:<|\"|>Europe/Paris<|\"|>}<tool_call|>";
        let r = parser.extract_tool_calls(out);
        assert!(r.tools_called);
        assert_eq!(r.tool_calls.len(), 2);
        assert_eq!(r.tool_calls[0].function.name, "get_weather");
        assert_eq!(r.tool_calls[1].function.name, "get_time");
        assert_eq!(
            gemma4_args(&r.tool_calls[1]),
            serde_json::json!({"tz": "Europe/Paris"})
        );
    }

    #[test]
    fn test_gemma4_content_before_call() {
        let parser = Gemma4ToolParser::new();
        let out = "Let me check.<|tool_call>call:f{x:1}<tool_call|>";
        let r = parser.extract_tool_calls(out);
        assert!(r.tools_called);
        assert_eq!(r.content.as_deref(), Some("Let me check."));
    }

    #[test]
    fn test_gemma4_no_tool_call_is_content() {
        let parser = Gemma4ToolParser::new();
        let out = "The weather in London is sunny.";
        let r = parser.extract_tool_calls(out);
        assert!(!r.tools_called);
        assert!(r.tool_calls.is_empty());
        assert_eq!(
            r.content.as_deref(),
            Some("The weather in London is sunny.")
        );
    }

    #[test]
    fn test_gemma4_strips_leaked_eos_from_content() {
        // With skip_special_tokens off (required for tool tokens), a plain-text
        // answer carries a trailing <end_of_turn> — it must not leak.
        let parser = Gemma4ToolParser::new();
        let r = parser.extract_tool_calls("Sunny in London.<end_of_turn>");
        assert!(!r.tools_called);
        assert_eq!(r.content.as_deref(), Some("Sunny in London."));
    }

    #[test]
    fn test_gemma4_strips_channel_thought_wrapper() {
        // Gemma wraps a post-tool answer as `<|channel>thought<channel|>ANSWER`;
        // the wrapper is dropped and the answer kept.
        let parser = Gemma4ToolParser::new();
        let r = parser.extract_tool_calls("<|channel>thought<channel|>The answer is 42.");
        assert!(!r.tools_called);
        assert_eq!(r.content.as_deref(), Some("The answer is 42."));
    }

    #[test]
    fn test_gemma4_strips_whitespace_variant_channel() {
        // The actual bug: a small MoE emits the channel marker as ordinary text
        // with stray spaces (`< |channel>thought < channel|>`) — the exact strip
        // missed it and the raw marker leaked into the visible answer.
        let parser = Gemma4ToolParser::new();
        let r = parser.extract_tool_calls("< |channel>thought < channel|>The answer is 42.");
        assert!(!r.tools_called);
        assert_eq!(r.content.as_deref(), Some("The answer is 42."));
    }

    #[test]
    fn test_gemma4_strips_whitespace_variant_turn_token() {
        let parser = Gemma4ToolParser::new();
        let r = parser.extract_tool_calls("Done.< end_of_turn>");
        assert!(!r.tools_called);
        assert_eq!(r.content.as_deref(), Some("Done."));
    }

    #[test]
    fn test_gemma4_tolerant_strip_no_false_positive_on_plain_angles() {
        // Ordinary comparisons / generics with `<` … `>` and inner whitespace
        // must NOT be mistaken for a control token (the atoms require the exact
        // marker words like `channel` / `end_of_turn`).
        let parser = Gemma4ToolParser::new();
        let r = parser.extract_tool_calls("compute if a < b > c and Vec< T > stays intact");
        assert!(!r.tools_called);
        assert_eq!(
            r.content.as_deref(),
            Some("compute if a < b > c and Vec< T > stays intact")
        );
    }

    #[test]
    fn test_gemma4_lone_channel_marker_yields_empty_content() {
        // A bare thought cesura emitted between actions must strip to empty so
        // the provider/TUI filter it out — no empty `●` bubble (no caesura).
        let parser = Gemma4ToolParser::new();
        let r = parser.extract_tool_calls("< |channel>thought < channel|>");
        assert!(!r.tools_called);
        assert_eq!(r.content.as_deref(), Some(""));
    }

    #[test]
    fn test_gemma4_channel_marker_before_call_is_stripped() {
        // The cesura must not leak even when it precedes a valid tool call in
        // the same round (the content-before-call path, not just the no-call one).
        let parser = Gemma4ToolParser::new();
        let r = parser.extract_tool_calls(
            "< |channel>thought < channel|><|tool_call>call:list_dir{path:<|\"|>.<|\"|>}<tool_call|>",
        );
        assert!(r.tools_called);
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].function.name, "list_dir");
        assert_eq!(r.content, None);
    }

    #[test]
    fn test_gemma4_streaming_strips_leaked_eos() {
        let parser = Gemma4ToolParser::new();
        let mut state = parser.create_streaming_state();
        let mut acc = String::new();
        let mut content = String::new();
        for t in ["Sunny", " there", "<end_of_turn>"] {
            let prev = acc.clone();
            acc.push_str(t);
            if let ToolParserDelta::Content(c) = state.process_delta(&prev, &acc, t) {
                content.push_str(&c);
            }
        }
        // Inter-token spaces preserved, control token dropped.
        assert_eq!(content, "Sunny there");
    }

    #[test]
    fn test_gemma4_granite_lookalike_is_not_matched() {
        // Granite's `<|tool_call|>` (pipes BOTH ends) + JSON array must NOT be
        // parsed as a Gemma 4 call — the tokens differ by one pipe.
        let parser = Gemma4ToolParser::new();
        let out = r#"<|tool_call|>[{"name":"f","arguments":{}}]"#;
        let r = parser.extract_tool_calls(out);
        assert!(!r.tools_called);
    }

    #[test]
    fn test_gemma4_requires_special_tokens() {
        assert!(Gemma4ToolParser::new().requires_special_tokens());
        // Plain-text parsers must NOT force special-token retention.
        assert!(!HermesToolParser::new().requires_special_tokens());
        assert!(!GraniteToolParser::new().requires_special_tokens());
    }

    #[test]
    fn test_registry_gemma4() {
        assert!(get_tool_parser("gemma4").is_ok());
        assert!(get_tool_parser("gemma").is_ok());
    }

    // -- Gemma 4 streaming tests --

    #[test]
    fn test_gemma4_streaming_token_by_token() {
        let parser = Gemma4ToolParser::new();
        let mut state = parser.create_streaming_state();
        let tokens = [
            "<|tool_call>",
            "call:get_weather",
            "{city:",
            "<|\"|>",
            "SF",
            "<|\"|>",
            "}",
            "<tool_call|>",
        ];
        let mut acc = String::new();
        let mut name = None;
        let mut args = String::new();
        for t in tokens {
            let prev = acc.clone();
            acc.push_str(t);
            if let ToolParserDelta::ToolCalls(calls) = state.process_delta(&prev, &acc, t) {
                for c in &calls {
                    if let Some(n) = &c.function_name {
                        name = Some(n.clone());
                    }
                    if let Some(a) = &c.function_arguments {
                        args.push_str(a);
                    }
                }
            }
        }
        assert_eq!(name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&args).unwrap(),
            serde_json::json!({"city": "SF"})
        );
    }

    #[test]
    fn test_gemma4_streaming_whole_call_in_one_delta() {
        let parser = Gemma4ToolParser::new();
        let mut state = parser.create_streaming_state();
        let whole = "<|tool_call>call:f{x:1}<tool_call|>";
        let mut name = None;
        let mut args = String::new();
        if let ToolParserDelta::ToolCalls(calls) = state.process_delta("", whole, whole) {
            for c in &calls {
                if let Some(n) = &c.function_name {
                    name = Some(n.clone());
                }
                if let Some(a) = &c.function_arguments {
                    args.push_str(a);
                }
            }
        }
        assert_eq!(name.as_deref(), Some("f"));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&args).unwrap(),
            serde_json::json!({"x": 1})
        );
    }

    #[test]
    fn test_gemma4_streaming_no_tool_is_content() {
        let parser = Gemma4ToolParser::new();
        let mut state = parser.create_streaming_state();
        let d = state.process_delta("", "Hello there", "Hello there");
        assert!(matches!(d, ToolParserDelta::Content(ref s) if s == "Hello there"));
    }
}

#[cfg(test)]
mod qwen3_coder_tests {
    use super::*;

    #[test]
    fn extracts_xml_function_call() {
        let text = "<tool_call>\n<function=tree>\n<parameter=path>\n/Users/moosevan/git/scratchy\n</parameter>\n<parameter=depth>\n3\n</parameter>\n</function>\n</tool_call>";
        let e = qwen3_coder_extract(text);
        assert!(e.tools_called);
        assert_eq!(e.tool_calls.len(), 1);
        assert_eq!(e.tool_calls[0].function.name, "tree");
        let args: serde_json::Value =
            serde_json::from_str(&e.tool_calls[0].function.arguments).unwrap();
        assert_eq!(args["path"], "/Users/moosevan/git/scratchy");
        assert_eq!(args["depth"], 3); // coerced to a number
    }

    #[test]
    fn keeps_leading_content() {
        let text = "Let me look.\n<tool_call><function=ls><parameter=path>.</parameter></function></tool_call>";
        let e = qwen3_coder_extract(text);
        assert!(e.tools_called);
        assert_eq!(e.content.as_deref(), Some("Let me look."));
        assert_eq!(e.tool_calls[0].function.name, "ls");
    }

    #[test]
    fn plain_text_is_content() {
        let e = qwen3_coder_extract("The capital of France is Paris.");
        assert!(!e.tools_called);
        assert_eq!(
            e.content.as_deref(),
            Some("The capital of France is Paris.")
        );
    }

    #[test]
    fn tolerates_unclosed_final_call() {
        // Qwen3.6 sometimes omits the closing </tool_call>.
        let text =
            "<tool_call>\n<function=tree>\n<parameter=path>\n/some/path\n</parameter>\n</function>";
        let e = qwen3_coder_extract(text);
        assert!(e.tools_called, "unclosed call should still parse");
        assert_eq!(e.tool_calls.len(), 1);
        assert_eq!(e.tool_calls[0].function.name, "tree");
        let args: serde_json::Value =
            serde_json::from_str(&e.tool_calls[0].function.arguments).unwrap();
        assert_eq!(args["path"], "/some/path");
    }

    #[test]
    fn routes_by_architecture() {
        // model_type values from config.json.
        assert_eq!(detect_tool_parser("qwen3_5_moe"), Some("qwen3_coder"));
        assert_eq!(detect_tool_parser("qwen3"), Some("hermes"));
        assert_eq!(detect_tool_parser("gemma4"), Some("gemma4"));
        assert_eq!(detect_tool_parser("llama"), Some("llama3_json"));
        assert_eq!(detect_tool_parser("mistral"), Some("mistral"));
        // Also works on the raw architecture string.
        assert_eq!(
            detect_tool_parser("Qwen3_5MoeForConditionalGeneration"),
            Some("qwen3_coder")
        );
        assert_eq!(detect_tool_parser("LlamaForCausalLM"), Some("llama3_json"));
    }
}
