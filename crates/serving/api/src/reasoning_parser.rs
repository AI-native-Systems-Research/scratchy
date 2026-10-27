// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Reasoning content extraction from model outputs.
//!
//! When models like DeepSeek-R1 or Qwen3 generate text containing thinking
//! blocks delimited by `<think>...</think>`, these parsers extract the
//! reasoning content and return it separately from the main content.
//!
//! Supports both non-streaming (full text extraction) and streaming
//! (incremental delta) modes, matching Python vLLM behavior.
//!
//! Port of: `vllm/reasoning/basic_parsers.py`, `deepseek_r1_reasoning_parser.py`,
//! `qwen3_reasoning_parser.py`

use std::collections::HashMap;

use scratchy_core_common::sampling::ThinkTokens;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Result of extracting reasoning from complete model output (non-streaming).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedReasoning {
    /// The reasoning/thinking content (None if no reasoning was found).
    pub reasoning: Option<String>,
    /// The content after reasoning (None if empty or all reasoning).
    pub content: Option<String>,
}

/// A streaming reasoning delta — indicates what portion of a delta is
/// reasoning vs. content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasoningDelta {
    /// Delta is purely reasoning content.
    Reasoning(String),
    /// Delta is purely regular content.
    Content(String),
    /// Delta contains both reasoning and content (end-of-think within delta).
    Split {
        reasoning: Option<String>,
        content: Option<String>,
    },
    /// Nothing to emit (e.g. start/end token was the only content).
    None,
}

impl ReasoningDelta {
    /// Split into (reasoning, content) parts.
    fn into_parts(self) -> (Option<String>, Option<String>) {
        match self {
            ReasoningDelta::Reasoning(r) => (Some(r), None),
            ReasoningDelta::Content(c) => (None, Some(c)),
            ReasoningDelta::Split { reasoning, content } => (reasoning, content),
            ReasoningDelta::None => (None, None),
        }
    }

    /// Concatenate two deltas into one. The engine folds a parser's
    /// end-of-stream [`StreamingReasoningParserState::flush`] into the final
    /// output delta with this.
    pub fn merge(self, other: ReasoningDelta) -> ReasoningDelta {
        let combine = |a: Option<String>, b: Option<String>| match (a, b) {
            (Some(mut x), Some(y)) => {
                x.push_str(&y);
                Some(x)
            }
            // Covers (Some, None), (None, Some) and (None, None).
            (x, None) | (None, x) => x,
        };
        let (reasoning, content) = match (self, other) {
            (ReasoningDelta::None, x) | (x, ReasoningDelta::None) => return x,
            (a, b) => {
                let (ar, ac) = a.into_parts();
                let (br, bc) = b.into_parts();
                (combine(ar, br), combine(ac, bc))
            }
        };
        match (reasoning, content) {
            (None, None) => ReasoningDelta::None,
            (Some(r), None) => ReasoningDelta::Reasoning(r),
            (None, Some(c)) => ReasoningDelta::Content(c),
            (Some(r), Some(c)) => ReasoningDelta::Split {
                reasoning: Some(r),
                content: Some(c),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// A parser that can extract reasoning content from model output.
///
/// Mirrors Python's `ReasoningParser` / `BaseThinkingReasoningParser`.
pub trait ReasoningParser: Send + Sync {
    /// Extract reasoning from a complete model output (non-streaming).
    fn extract_reasoning(&self, model_output: &str) -> ExtractedReasoning;

    /// Create a new per-request streaming state.
    fn create_streaming_state(&self) -> Box<dyn StreamingReasoningParserState + Send>;

    /// Whether the parser needs the server to keep special tokens in the
    /// detokenized text: its markers are real added tokens that
    /// `skip_special_tokens` would strip before the parser ever runs. Mirrors
    /// [`crate::tool_parser::ToolCallParser::requires_special_tokens`].
    fn requires_special_tokens(&self) -> bool {
        false
    }

    /// The tokens this parser's thinking opens and closes on, when they are single tokens
    /// (`<think>` and `</think>`): a request's guided grammar waits for the close (#793). `None`
    /// for a parser that reads its markers as text, so the grammar applies from the first token.
    fn think_tokens(&self) -> Option<ThinkTokens>;
}

/// Per-request state for streaming reasoning extraction.
///
/// Each streaming request gets its own state to track whether we're
/// inside a thinking block or not.
pub trait StreamingReasoningParserState: Send {
    /// Process a streaming delta and return the reasoning/content routing.
    ///
    /// Arguments mirror Python's `extract_reasoning_streaming`:
    /// - `previous_text`: accumulated text before this delta
    /// - `current_text`: accumulated text including this delta
    /// - `delta_text`: the new text in this delta
    /// - `previous_token_ids`: token IDs before this delta
    /// - `current_token_ids`: token IDs including this delta
    /// - `delta_token_ids`: new token IDs in this delta
    fn process_delta(
        &mut self,
        previous_text: &str,
        current_text: &str,
        delta_text: &str,
        previous_token_ids: &[u32],
        current_token_ids: &[u32],
        delta_token_ids: &[u32],
    ) -> ReasoningDelta;

    /// Whether reasoning can re-open after content in a single generation.
    ///
    /// Most formats are strictly reasoning-first (DeepSeek, Qwen3); the engine
    /// stops routing text through the parser once content has been seen.
    /// Gemma 4's thought channels interleave — text, then a channel, then a
    /// tool call — and so can harmony's messages, so their parsers override
    /// this and keep routing every delta.
    fn interleaved(&self) -> bool {
        false
    }

    /// Release anything the parser held back against a marker that never
    /// arrived, as a final delta. Called once when the request finishes, if
    /// the parser still routes text then: always for an [`interleaved`]
    /// parser, and for a reasoning-first one only if no content came. A
    /// parser that holds text back across deltas (Gemma 4's channel-name
    /// hold, harmony's unclosed `<`) must be interleaved, and return the text
    /// here, so a reply ending on held text is not truncated.
    ///
    /// [`interleaved`]: StreamingReasoningParserState::interleaved
    fn flush(&mut self) -> ReasoningDelta {
        ReasoningDelta::None
    }
}

// ---------------------------------------------------------------------------
// DeepSeek R1 Parser
// ---------------------------------------------------------------------------

/// Reasoning parser for DeepSeek R1 model.
///
/// Uses `<think>...</think>` tokens. Handles the case where the model
/// starts reasoning without generating `<think>` (no start token).
///
/// Port of: `vllm/reasoning/deepseek_r1_reasoning_parser.py`
pub struct DeepSeekR1ReasoningParser {
    start_token: &'static str,
    end_token: &'static str,
    start_token_id: u32,
    end_token_id: u32,
}

impl DeepSeekR1ReasoningParser {
    const START_TOKEN: &'static str = "<think>";
    const END_TOKEN: &'static str = "</think>";

    pub fn new(vocab: &HashMap<String, u32>) -> Result<Self, String> {
        let start_token_id = vocab.get(Self::START_TOKEN).copied().ok_or_else(|| {
            "DeepSeekR1ReasoningParser: could not find <think> token in vocabulary".to_string()
        })?;
        let end_token_id = vocab.get(Self::END_TOKEN).copied().ok_or_else(|| {
            "DeepSeekR1ReasoningParser: could not find </think> token in vocabulary".to_string()
        })?;
        Ok(Self {
            start_token: Self::START_TOKEN,
            end_token: Self::END_TOKEN,
            start_token_id,
            end_token_id,
        })
    }
}

impl ReasoningParser for DeepSeekR1ReasoningParser {
    fn think_tokens(&self) -> Option<ThinkTokens> {
        Some(ThinkTokens {
            open: self.start_token_id,
            close: self.end_token_id,
        })
    }

    fn extract_reasoning(&self, model_output: &str) -> ExtractedReasoning {
        // Strip <think> if present.
        let output = if let Some(after) = strip_start_token(model_output, self.start_token) {
            after
        } else {
            model_output
        };

        // If no end token, everything is reasoning (model didn't finish thinking).
        if !output.contains(self.end_token) {
            return ExtractedReasoning {
                reasoning: Some(output.to_string()),
                content: None,
            };
        }

        // Split on end token, then strip any stray think tags a malformed
        // output leaves behind (e.g. a doubled `</think>`).
        let (reasoning, content) = partition_on(output, self.end_token);
        let content = content
            .replace(self.end_token, "")
            .replace(self.start_token, "");
        let content = content.trim();
        ExtractedReasoning {
            reasoning: Some(reasoning.to_string()),
            content: if content.is_empty() {
                None
            } else {
                Some(content.to_string())
            },
        }
    }

    fn create_streaming_state(&self) -> Box<dyn StreamingReasoningParserState + Send> {
        Box::new(DeepSeekR1StreamingState {
            start_token: self.start_token,
            end_token: self.end_token,
            start_token_id: self.start_token_id,
            end_token_id: self.end_token_id,
        })
    }
}

struct DeepSeekR1StreamingState {
    start_token: &'static str,
    end_token: &'static str,
    start_token_id: u32,
    end_token_id: u32,
}

impl StreamingReasoningParserState for DeepSeekR1StreamingState {
    fn process_delta(
        &mut self,
        _previous_text: &str,
        _current_text: &str,
        delta_text: &str,
        previous_token_ids: &[u32],
        _current_token_ids: &[u32],
        delta_token_ids: &[u32],
    ) -> ReasoningDelta {
        // Skip single special tokens (start or end token alone).
        if delta_token_ids.len() == 1
            && (delta_token_ids[0] == self.start_token_id
                || delta_token_ids[0] == self.end_token_id)
        {
            return ReasoningDelta::None;
        }

        let start_in_prev = previous_token_ids.contains(&self.start_token_id);
        let start_in_delta = delta_token_ids.contains(&self.start_token_id);
        let end_in_prev = previous_token_ids.contains(&self.end_token_id);
        let end_in_delta = delta_token_ids.contains(&self.end_token_id);

        // Base logic from BaseThinkingReasoningParser.
        if start_in_prev {
            if end_in_delta {
                // Split: reasoning before </think>, content after.
                let end_index = delta_text.find(self.end_token).unwrap_or(delta_text.len());
                let reasoning = &delta_text[..end_index];
                let content = &delta_text[end_index + self.end_token.len()..];
                return ReasoningDelta::Split {
                    reasoning: non_empty(reasoning),
                    content: non_empty(content),
                };
            } else if end_in_prev {
                // After thinking ended — content.
                return ReasoningDelta::Content(delta_text.to_string());
            } else {
                // Still in thinking block — reasoning.
                return ReasoningDelta::Reasoning(delta_text.to_string());
            }
        } else if start_in_delta {
            if end_in_delta {
                // Both start and end in this delta.
                let start_index = delta_text.find(self.start_token).unwrap_or(0);
                let end_index = delta_text.find(self.end_token).unwrap_or(delta_text.len());
                let reasoning = &delta_text[start_index + self.start_token.len()..end_index];
                let content = &delta_text[end_index + self.end_token.len()..];
                return ReasoningDelta::Split {
                    reasoning: non_empty(reasoning),
                    content: non_empty(content),
                };
            } else {
                // Start in delta, no end yet — reasoning.
                return ReasoningDelta::Reasoning(delta_text.to_string());
            }
        }

        // DeepSeek R1 override: no start token seen anywhere — model started
        // reasoning without <think>.
        if !start_in_prev && !start_in_delta {
            if end_in_delta {
                let end_index = delta_text.find(self.end_token).unwrap_or(delta_text.len());
                let reasoning = &delta_text[..end_index];
                let content = &delta_text[end_index + self.end_token.len()..];
                return ReasoningDelta::Split {
                    reasoning: non_empty(reasoning),
                    content: non_empty(content),
                };
            } else if end_in_prev {
                return ReasoningDelta::Content(delta_text.to_string());
            } else {
                return ReasoningDelta::Reasoning(delta_text.to_string());
            }
        }

        // Fallback: content.
        ReasoningDelta::Content(delta_text.to_string())
    }
}

// ---------------------------------------------------------------------------
// Qwen3 Parser
// ---------------------------------------------------------------------------

/// Reasoning parser for the Qwen3 model family.
///
/// Uses `<think>...</think>` tokens. The chat template typically places
/// `<think>` in the prompt, so only `</think>` appears in the generated output.
/// When thinking is disabled, no think tokens appear — everything is content.
///
/// Port of: `vllm/reasoning/qwen3_reasoning_parser.py`
pub struct Qwen3ReasoningParser {
    start_token: &'static str,
    end_token: &'static str,
    start_token_id: u32,
    end_token_id: u32,
}

impl Qwen3ReasoningParser {
    const START_TOKEN: &'static str = "<think>";
    const END_TOKEN: &'static str = "</think>";

    pub fn new(vocab: &HashMap<String, u32>) -> Result<Self, String> {
        let start_token_id = vocab.get(Self::START_TOKEN).copied().ok_or_else(|| {
            "Qwen3ReasoningParser: could not find <think> token in vocabulary".to_string()
        })?;
        let end_token_id = vocab.get(Self::END_TOKEN).copied().ok_or_else(|| {
            "Qwen3ReasoningParser: could not find </think> token in vocabulary".to_string()
        })?;
        Ok(Self {
            start_token: Self::START_TOKEN,
            end_token: Self::END_TOKEN,
            start_token_id,
            end_token_id,
        })
    }
}

impl ReasoningParser for Qwen3ReasoningParser {
    fn think_tokens(&self) -> Option<ThinkTokens> {
        Some(ThinkTokens {
            open: self.start_token_id,
            close: self.end_token_id,
        })
    }

    fn extract_reasoning(&self, model_output: &str) -> ExtractedReasoning {
        // Strip <think> if present in the generated output.
        let output = if let Some(after) = strip_start_token(model_output, self.start_token) {
            after
        } else {
            model_output
        };

        // No end token means thinking is disabled — everything is content.
        if !output.contains(self.end_token) {
            return ExtractedReasoning {
                reasoning: None,
                content: Some(output.to_string()),
            };
        }

        // Split on end token, then strip any stray think tags a malformed
        // output leaves behind (e.g. a doubled `</think>`).
        let (reasoning, content) = partition_on(output, self.end_token);
        let content = content
            .replace(self.end_token, "")
            .replace(self.start_token, "");
        let content = content.trim();
        ExtractedReasoning {
            reasoning: Some(reasoning.to_string()),
            content: if content.is_empty() {
                None
            } else {
                Some(content.to_string())
            },
        }
    }

    fn create_streaming_state(&self) -> Box<dyn StreamingReasoningParserState + Send> {
        Box::new(Qwen3StreamingState {
            start_token: self.start_token,
            end_token: self.end_token,
            start_token_id: self.start_token_id,
            end_token_id: self.end_token_id,
        })
    }
}

struct Qwen3StreamingState {
    start_token: &'static str,
    end_token: &'static str,
    start_token_id: u32,
    end_token_id: u32,
}

impl StreamingReasoningParserState for Qwen3StreamingState {
    fn process_delta(
        &mut self,
        _previous_text: &str,
        _current_text: &str,
        delta_text: &str,
        previous_token_ids: &[u32],
        _current_token_ids: &[u32],
        delta_token_ids: &[u32],
    ) -> ReasoningDelta {
        // Strip <think> from delta if present (old template / edge case).
        let delta_text = if delta_token_ids.contains(&self.start_token_id) {
            if let Some(start_idx) = delta_text.find(self.start_token) {
                &delta_text[start_idx + self.start_token.len()..]
            } else {
                delta_text
            }
        } else {
            delta_text
        };

        if delta_token_ids.contains(&self.end_token_id) {
            // End token in this delta: split reasoning from content.
            if let Some(end_index) = delta_text.find(self.end_token) {
                let reasoning = &delta_text[..end_index];
                let content = &delta_text[end_index + self.end_token.len()..];
                if reasoning.is_empty() && content.is_empty() {
                    return ReasoningDelta::None;
                }
                ReasoningDelta::Split {
                    reasoning: non_empty(reasoning),
                    content: non_empty(content),
                }
            } else {
                // end_token_id in IDs but not in text (already stripped).
                ReasoningDelta::None
            }
        } else if delta_text.is_empty() {
            // Nothing left after stripping start token.
            ReasoningDelta::None
        } else if previous_token_ids.contains(&self.end_token_id) {
            // End token already passed: everything is content now.
            ReasoningDelta::Content(delta_text.to_string())
        } else {
            // No end token yet: still in reasoning phase.
            ReasoningDelta::Reasoning(delta_text.to_string())
        }
    }
}

// ---------------------------------------------------------------------------
// GPT-OSS / Harmony Parser
// ---------------------------------------------------------------------------

/// Reasoning parser for gpt-oss / OpenAI "harmony" responses.
///
/// The assistant emits channel-tagged segments:
/// `<|channel|>analysis<|message|>…<|end|>` … `<|channel|>final<|message|>…`.
/// The `analysis` (and `commentary`) channels are reasoning; the `final`
/// channel is the answer. All harmony control markers are consumed.
///
/// Text-based (no vocab needed): the harmony markers are not guaranteed to be
/// single vocab tokens across quantized ports, so we match on their text.
pub struct GptOssReasoningParser;

/// A recognized harmony control marker. Classification is tolerant of quant
/// ports that emit the pipe on only one side: `<|channel>` (open) and the
/// stray `<channel|>` both occur in the wild alongside canonical
/// `<|channel|>` / `<|message|>`.
#[derive(PartialEq)]
enum HarmonyMarker {
    Channel,
    Message,
    Start,
    End,
    NotMarker,
}

fn classify_marker(inner: &str) -> HarmonyMarker {
    // Real harmony markers always carry a pipe; requiring one keeps plain text
    // like `<end>` in prose from being mistaken for a control token.
    if !inner.contains('|') {
        return HarmonyMarker::NotMarker;
    }
    let key: String = inner
        .chars()
        .filter(|c| *c != '|' && !c.is_whitespace())
        .collect::<String>()
        .to_lowercase();
    let (starts, ends) = (inner.starts_with('|'), inner.ends_with('|'));
    match key.as_str() {
        "message" => HarmonyMarker::Message,
        "start" => HarmonyMarker::Start,
        "end" | "return" => HarmonyMarker::End,
        // A trailing-only pipe (`<channel|>`) is used as a message delimiter by
        // some malformed ports; a leading pipe means a real channel open.
        "channel" if ends && !starts => HarmonyMarker::Message,
        "channel" => HarmonyMarker::Channel,
        _ => HarmonyMarker::NotMarker,
    }
}

fn is_reasoning_channel(name: &str) -> bool {
    let n = name.to_lowercase();
    ["analysis", "think", "thought", "reason", "commentary"]
        .iter()
        .any(|k| n.contains(k))
}

/// What the harmony text since the last marker is.
#[derive(Clone, Copy, Default)]
enum HarmonyPart {
    /// A message body (or text before any marker), routed by the open channel.
    #[default]
    Body,
    /// A channel's name, after `<|channel|>`: never emitted.
    ChannelName,
    /// A message's role, after `<|start|>` (e.g. `assistant`): never emitted.
    Role,
}

/// The longest text after a `<` the scanner holds for a `>` that may close a marker: every
/// harmony marker with room for a malformed port's stray whitespace.
const HARMONY_MARKER_MAX: usize = 24;

/// The harmony scanner both of [`GptOssReasoningParser`]'s paths run, so the streamed reply and
/// the complete one split text the same way. Fed text piece by piece; a `<` whose marker has not
/// closed yet is held until it does, or until it is too long to be one.
#[derive(Default)]
struct HarmonyScanner {
    part: HarmonyPart,
    /// The open channel is a reasoning channel ([`is_reasoning_channel`]).
    reasoning_channel: bool,
    saw_marker: bool,
    /// The channel name or role read so far.
    header: String,
    /// A `<` that may open a marker, and the text after it.
    carry: String,
}

impl HarmonyScanner {
    /// Scan `text`, appending reasoning-channel bodies to `reasoning` and the rest of the bodies
    /// to `content`. Markers, channel names and roles are consumed.
    fn feed(&mut self, text: &str, reasoning: &mut String, content: &mut String) {
        let buf = std::mem::take(&mut self.carry) + text;
        let mut rest = buf.as_str();
        while let Some(lt) = rest.find('<') {
            self.push(&rest[..lt], reasoning, content);
            let tail = &rest[lt..];
            match tail[1..].find('>') {
                Some(gt) => match classify_marker(&tail[1..1 + gt]) {
                    HarmonyMarker::NotMarker => {
                        self.push("<", reasoning, content);
                        rest = &tail[1..];
                    }
                    marker => {
                        self.mark(marker);
                        rest = &tail[gt + 2..];
                    }
                },
                None if tail.len() < HARMONY_MARKER_MAX => {
                    self.carry = tail.to_string();
                    return;
                }
                None => {
                    self.push("<", reasoning, content);
                    rest = &tail[1..];
                }
            }
        }
        self.push(rest, reasoning, content);
    }

    /// The end of the text: a held `<` that opened no marker is text.
    fn finish(&mut self, reasoning: &mut String, content: &mut String) {
        let carry = std::mem::take(&mut self.carry);
        self.push(&carry, reasoning, content);
    }

    fn mark(&mut self, marker: HarmonyMarker) {
        if let HarmonyPart::ChannelName = self.part {
            self.reasoning_channel = is_reasoning_channel(self.header.trim());
        }
        self.header.clear();
        self.saw_marker = true;
        self.part = match marker {
            HarmonyMarker::Channel => HarmonyPart::ChannelName,
            HarmonyMarker::Start => HarmonyPart::Role,
            _ => HarmonyPart::Body,
        };
    }

    fn push(&mut self, text: &str, reasoning: &mut String, content: &mut String) {
        match self.part {
            HarmonyPart::Body if self.reasoning_channel => reasoning.push_str(text),
            HarmonyPart::Body => content.push_str(text),
            HarmonyPart::ChannelName | HarmonyPart::Role => self.header.push_str(text),
        }
    }
}

/// One streamed delta's split.
fn reasoning_delta(reasoning: String, content: String) -> ReasoningDelta {
    match (reasoning.is_empty(), content.is_empty()) {
        (true, true) => ReasoningDelta::None,
        (false, true) => ReasoningDelta::Reasoning(reasoning),
        (true, false) => ReasoningDelta::Content(content),
        (false, false) => ReasoningDelta::Split {
            reasoning: Some(reasoning),
            content: Some(content),
        },
    }
}

impl ReasoningParser for GptOssReasoningParser {
    /// Harmony's channel markers are read as text: they aren't single tokens on every port.
    fn think_tokens(&self) -> Option<ThinkTokens> {
        None
    }

    fn extract_reasoning(&self, model_output: &str) -> ExtractedReasoning {
        let mut scanner = HarmonyScanner::default();
        let (mut reasoning, mut content) = (String::new(), String::new());
        scanner.feed(model_output, &mut reasoning, &mut content);
        scanner.finish(&mut reasoning, &mut content);
        if !scanner.saw_marker {
            return ExtractedReasoning {
                reasoning: None,
                content: non_empty(model_output),
            };
        }
        ExtractedReasoning {
            reasoning: non_empty(reasoning.trim()),
            content: non_empty(content.trim()),
        }
    }

    fn create_streaming_state(&self) -> Box<dyn StreamingReasoningParserState + Send> {
        Box::new(GptOssStreamingState::default())
    }

    /// The harmony markers are special tokens in gpt-oss's tokenizer (`<|channel|>`,
    /// `<|message|>`, `<|start|>`, `<|end|>`, `<|return|>`, `<|call|>`), so
    /// `skip_special_tokens` strips them before this parser reads the text, and the analysis
    /// channel came back as content (#601).
    fn requires_special_tokens(&self) -> bool {
        true
    }
}

#[derive(Default)]
struct GptOssStreamingState(HarmonyScanner);

impl StreamingReasoningParserState for GptOssStreamingState {
    fn process_delta(
        &mut self,
        _previous_text: &str,
        _current_text: &str,
        delta_text: &str,
        _previous_token_ids: &[u32],
        _current_token_ids: &[u32],
        _delta_token_ids: &[u32],
    ) -> ReasoningDelta {
        let (mut reasoning, mut content) = (String::new(), String::new());
        self.0.feed(delta_text, &mut reasoning, &mut content);
        reasoning_delta(reasoning, content)
    }

    fn flush(&mut self) -> ReasoningDelta {
        let (mut reasoning, mut content) = (String::new(), String::new());
        self.0.finish(&mut reasoning, &mut content);
        reasoning_delta(reasoning, content)
    }

    /// Every delta goes through the scanner, content included: a harmony reply can open another
    /// channel after content (a later message), and the scanner may hold a `<` that the next
    /// delta closes. Left to the engine after the first content, the held `<` was lost and any
    /// later marker streamed as text.
    fn interleaved(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Gemma 4 Parser
// ---------------------------------------------------------------------------

/// Reasoning parser for Gemma 4 (dense `gemma4*`, MoE, and `-unified` ports).
///
/// Gemma 4 wraps its reasoning in a *channel*: the model emits
/// `<|channel>thought\n{reasoning}\n<channel|>` and the answer follows the
/// closing marker. Channels interleave — the model may answer some text,
/// then open a channel, then make a tool call — and a degenerate stream can
/// re-open a channel many times without ever closing one (the
/// "hundreds of lines of `thought`" flood). Both markers are real added
/// tokens, so this parser is text-based but whitespace-tolerant (shared with
/// the Gemma 4 tool parser) to also match quantized ports that emit the
/// markers as ordinary text pieces.
pub struct Gemma4ReasoningParser;

impl Gemma4ReasoningParser {
    /// `<|channel>` — opens a channel; the channel name follows.
    const OPEN_ATOMS: [&str; 4] = ["<", "|", "channel", ">"];
    /// `<channel|>` — closes the open channel.
    const CLOSE_ATOMS: [&str; 4] = ["<", "channel", "|", ">"];
    /// Canonical marker strings, for split-across-deltas detection.
    const OPEN: &'static str = "<|channel>";
    const CLOSE: &'static str = "<channel|>";
}

impl ReasoningParser for Gemma4ReasoningParser {
    /// Its channel markers are read as text, atom by atom.
    fn think_tokens(&self) -> Option<ThinkTokens> {
        None
    }

    fn extract_reasoning(&self, model_output: &str) -> ExtractedReasoning {
        let mut state = Gemma4StreamingState::default();
        let mut reasoning = String::new();
        let mut content = String::new();
        state.scan(model_output, &mut reasoning, &mut content);
        // A complete generation cannot end inside a marker; whatever the
        // carry holds is trailing text.
        state.scan_carry(&mut reasoning, &mut content);
        ExtractedReasoning {
            reasoning: non_empty(reasoning.trim()),
            content: non_empty(content.trim()),
        }
    }

    fn create_streaming_state(&self) -> Box<dyn StreamingReasoningParserState + Send> {
        Box::new(Gemma4StreamingState::default())
    }

    fn requires_special_tokens(&self) -> bool {
        true
    }
}

/// Where the scanner sits relative to the channel markers.
#[derive(Clone, Copy, Default)]
enum Gemma4ChannelMode {
    /// Outside any channel — text is content.
    #[default]
    Outside,
    /// Between the open marker and the newline ending the channel name.
    Name,
    /// Inside the channel body; the body routes to reasoning iff `reasoning`.
    Body { reasoning: bool },
}

#[derive(Default)]
struct Gemma4StreamingState {
    mode: Gemma4ChannelMode,
    /// Channel name accumulated in [`Gemma4ChannelMode::Name`] (never emitted).
    name: String,
    /// Suffix of the previous delta that may be the start of a marker; prepended
    /// to the next delta so a marker split across deltas still matches.
    carry: String,
    /// A trailing newline held back from a thought channel's body: the format
    /// writes `{reasoning}\n<channel|>`, so a body-final newline is the
    /// *separator* and must not stream as reasoning. It is dropped when the
    /// next delta opens with a marker, and emitted as reasoning otherwise.
    pending_newline: bool,
    /// Outside-mode text held back because it may be the *name* of a channel
    /// whose open marker the degenerate stream dropped: the flood shape is
    /// `thought\n<channel|>` repeated with no `<|channel>` in sight. The
    /// held text is dropped when a close marker arrives and emitted as
    /// content otherwise.
    held: String,
}

impl Gemma4StreamingState {
    /// Scan `text`, appending reasoning-channel bodies to `reasoning` and
    /// everything else to `content`. Marker spans and channel names are
    /// consumed and never emitted.
    fn scan(&mut self, text: &str, reasoning: &mut String, content: &mut String) {
        let mut i = 0;
        while i < text.len() {
            match self.mode {
                Gemma4ChannelMode::Outside => {
                    let open = crate::tool_parser::gemma4_tolerant_find(
                        text,
                        &Gemma4ReasoningParser::OPEN_ATOMS,
                        i,
                    );
                    let close = crate::tool_parser::gemma4_tolerant_find(
                        text,
                        &Gemma4ReasoningParser::CLOSE_ATOMS,
                        i,
                    );
                    // Earliest marker: an open starts a channel; a close with
                    // no channel open is the degenerate flood's leftover.
                    let boundary = [open, close]
                        .into_iter()
                        .flatten()
                        .min_by_key(|(start, _)| *start);
                    match boundary {
                        // A close marker with no channel open — the late stage
                        // of the flood, where the stream writes
                        // `thought\n<channel|>` cycles and drops the opens.
                        // The marker is syntax and must never surface; the
                        // separator newlines before it and a trailing name
                        // line are the lost channel's name, dropped with it.
                        Some((start, end)) if Some((start, end)) == close => {
                            let mut cut = start;
                            while cut > i && text[..cut].ends_with('\n') {
                                cut -= 1;
                            }
                            let mut before = &text[i..cut];
                            let trim = trailing_reasoning_name(before);
                            if trim > 0 {
                                before = &before[..before.len() - trim];
                            }
                            self.held.clear();
                            content.push_str(before);
                            i = end;
                        }
                        // A real open: whatever was held was content, not a
                        // lost channel's name.
                        Some((start, end)) => {
                            let held = std::mem::take(&mut self.held);
                            content.push_str(&held);
                            content.push_str(&text[i..start]);
                            i = end;
                            self.mode = Gemma4ChannelMode::Name;
                            self.name.clear();
                        }
                        None => {
                            let hold = partial_marker_len(&text[i..]);
                            let emit_to = text.len() - hold;
                            let slice = &text[i..emit_to];
                            if self.held.is_empty() {
                                // Hold a trailing exact channel name (plus
                                // its newline) back: the flood writes the
                                // name in one delta and its close marker in
                                // the next.
                                let trim = trailing_reasoning_name(slice);
                                let (emit, held) = slice.split_at(slice.len() - trim);
                                content.push_str(emit);
                                self.held.push_str(held);
                            } else {
                                // Waiting on the close: separator newlines
                                // extend the hold; any other text means the
                                // hold was content after all.
                                let nl = slice.len() - slice.trim_start_matches('\n').len();
                                self.held.push_str(&slice[..nl]);
                                let rest = &slice[nl..];
                                if !rest.is_empty() {
                                    let held = std::mem::take(&mut self.held);
                                    content.push_str(&held);
                                    let trim = trailing_reasoning_name(rest);
                                    let (emit, held) = rest.split_at(rest.len() - trim);
                                    content.push_str(emit);
                                    self.held.push_str(held);
                                }
                            }
                            self.carry.push_str(&text[emit_to..]);
                            i = text.len();
                        }
                    }
                }
                Gemma4ChannelMode::Name => {
                    // The name ends at a newline, or — on a degenerate stream
                    // that never writes one — at whichever marker comes first.
                    let nl = text[i..].find('\n').map(|rel| (i + rel, i + rel));
                    let open = crate::tool_parser::gemma4_tolerant_find(
                        text,
                        &Gemma4ReasoningParser::OPEN_ATOMS,
                        i,
                    );
                    let close = crate::tool_parser::gemma4_tolerant_find(
                        text,
                        &Gemma4ReasoningParser::CLOSE_ATOMS,
                        i,
                    );
                    let boundary = [nl, open, close]
                        .into_iter()
                        .flatten()
                        .min_by_key(|(start, _)| *start);
                    match boundary {
                        // Newline: the name is complete, the body follows.
                        Some((start, end)) if Some((start, end)) == nl => {
                            self.name.push_str(&text[i..start]);
                            self.mode = Gemma4ChannelMode::Body {
                                reasoning: is_reasoning_channel(&self.name),
                            };
                            i = end + 1;
                        }
                        // Re-open before the name finished: an empty channel;
                        // restart the name.
                        Some((start, end)) if Some((start, end)) == open => {
                            self.name.clear();
                            i = end;
                        }
                        // Close right after the name: an empty channel.
                        Some((_start, end)) => {
                            self.mode = Gemma4ChannelMode::Outside;
                            i = end;
                        }
                        None => {
                            self.name.push_str(&text[i..]);
                            i = text.len();
                        }
                    }
                }
                // `is_reasoning` (not `reasoning`) so the binding does not
                // shadow the `&mut String` sink of the same name.
                Gemma4ChannelMode::Body {
                    reasoning: is_reasoning,
                } => {
                    // Resolve a newline held back from the previous delta: it
                    // is the channel separator if a marker opens this delta,
                    // body text otherwise.
                    if self.pending_newline {
                        let marker_opens = crate::tool_parser::gemma4_tolerant_find(
                            text,
                            &Gemma4ReasoningParser::CLOSE_ATOMS,
                            i,
                        )
                        .is_some_and(|(start, _)| start == i)
                            || crate::tool_parser::gemma4_tolerant_find(
                                text,
                                &Gemma4ReasoningParser::OPEN_ATOMS,
                                i,
                            )
                            .is_some_and(|(start, _)| start == i);
                        if !marker_opens {
                            if is_reasoning {
                                reasoning.push('\n');
                            } else {
                                content.push('\n');
                            }
                        }
                        self.pending_newline = false;
                    }
                    let open = crate::tool_parser::gemma4_tolerant_find(
                        text,
                        &Gemma4ReasoningParser::OPEN_ATOMS,
                        i,
                    );
                    let close = crate::tool_parser::gemma4_tolerant_find(
                        text,
                        &Gemma4ReasoningParser::CLOSE_ATOMS,
                        i,
                    );
                    // Earliest marker ends the body: the close normally, a
                    // re-open when the stream loops on channel openers.
                    let boundary = [open, close]
                        .into_iter()
                        .flatten()
                        .min_by_key(|(start, _)| *start);
                    let (start, end, next) = match boundary {
                        Some((start, end)) if Some((start, end)) == close => {
                            (start, end, Gemma4ChannelMode::Outside)
                        }
                        // Re-open: the channel implicitly closed empty; start
                        // a fresh name.
                        Some((start, end)) => {
                            self.name.clear();
                            (start, end, Gemma4ChannelMode::Name)
                        }
                        None => {
                            let hold = partial_marker_len(&text[i..]);
                            let emit_to = text.len() - hold;
                            let tail = &text[i..emit_to];
                            // Hold a thought channel's body-final newline back:
                            // the format writes `{reasoning}\n<channel|>`, so
                            // it may be the separator once the next delta
                            // arrives. Non-reasoning bodies emit verbatim.
                            if is_reasoning && let Some(stripped) = tail.strip_suffix('\n') {
                                reasoning.push_str(stripped);
                                self.pending_newline = true;
                            } else if is_reasoning {
                                reasoning.push_str(tail);
                            } else {
                                content.push_str(tail);
                            }
                            self.carry.push_str(&text[emit_to..]);
                            i = text.len();
                            continue;
                        }
                    };
                    // In a thought channel the newline directly before the
                    // marker is the format's separator, not body.
                    let body = if is_reasoning {
                        text[i..start].strip_suffix('\n').unwrap_or(&text[i..start])
                    } else {
                        &text[i..start]
                    };
                    if is_reasoning {
                        reasoning.push_str(body);
                    } else {
                        content.push_str(body);
                    }
                    self.mode = next;
                    i = end;
                }
            }
        }
    }

    /// Flush the carry as trailing text (a complete generation cannot end
    /// inside a marker). A newline still held back at flush time is body
    /// text — the separator it might have been part of never arrived.
    fn scan_carry(&mut self, reasoning: &mut String, content: &mut String) {
        // A name still held back never met its close marker: it was content.
        let held = std::mem::take(&mut self.held);
        content.push_str(&held);
        if self.pending_newline {
            match self.mode {
                Gemma4ChannelMode::Body { reasoning: r } => {
                    if r {
                        reasoning.push('\n');
                    } else {
                        content.push('\n');
                    }
                }
                _ => content.push('\n'),
            }
            self.pending_newline = false;
        }
        let rest = std::mem::take(&mut self.carry);
        if rest.is_empty() {
            return;
        }
        match self.mode {
            Gemma4ChannelMode::Body { reasoning: r } => {
                if r {
                    reasoning.push_str(&rest);
                } else {
                    content.push_str(&rest);
                }
            }
            _ => content.push_str(&rest),
        }
    }
}

/// Length of the longest suffix of `text` that is a proper prefix of either
/// Gemma 4 channel marker — text that may grow into a marker once the next
/// delta arrives.
fn partial_marker_len(text: &str) -> usize {
    let mut best = 0;
    for marker in [Gemma4ReasoningParser::OPEN, Gemma4ReasoningParser::CLOSE] {
        for n in 1..marker.len() {
            if text.ends_with(&marker[..n]) && n > best {
                best = n;
            }
        }
    }
    best
}

/// Length of the suffix of `text` that is a reasoning-channel name alone on
/// its final line, plus that line's trailing newline if present. The flood
/// stream writes the lost channel's name as a bare line — `thought\n` — so
/// Outside mode holds exactly that shape back; prose that merely ends with
/// the word ("I had a thought") does not match and streams normally.
fn trailing_reasoning_name(text: &str) -> usize {
    let body = text.strip_suffix('\n').unwrap_or(text);
    let line_start = body.rfind('\n').map(|p| p + 1).unwrap_or(0);
    if is_exact_reasoning_name(&body[line_start..]) {
        text.len() - line_start
    } else {
        0
    }
}

/// Exact channel-name match (contrast [`is_reasoning_channel`], which matches
/// names that *contain* a keyword).
fn is_exact_reasoning_name(line: &str) -> bool {
    let l = line.to_lowercase();
    ["analysis", "think", "thought", "reason", "commentary"].contains(&l.as_str())
}

impl StreamingReasoningParserState for Gemma4StreamingState {
    fn process_delta(
        &mut self,
        _previous_text: &str,
        _current_text: &str,
        delta_text: &str,
        _previous_token_ids: &[u32],
        _current_token_ids: &[u32],
        _delta_token_ids: &[u32],
    ) -> ReasoningDelta {
        let mut text = std::mem::take(&mut self.carry);
        text.push_str(delta_text);
        // A marker may straddle this delta's boundary; hold its partial start
        // back so the next delta completes it.
        let hold = partial_marker_len(&text);
        let split_at = text.len() - hold;
        self.carry = text.split_off(split_at);

        let mut reasoning = String::new();
        let mut content = String::new();
        self.scan(&text, &mut reasoning, &mut content);
        match (non_empty(&reasoning), non_empty(&content)) {
            (None, None) => ReasoningDelta::None,
            (Some(r), None) => ReasoningDelta::Reasoning(r),
            (None, Some(c)) => ReasoningDelta::Content(c),
            (Some(r), Some(c)) => ReasoningDelta::Split {
                reasoning: Some(r),
                content: Some(c),
            },
        }
    }

    fn interleaved(&self) -> bool {
        true
    }

    fn flush(&mut self) -> ReasoningDelta {
        // Whatever is held back — a channel name that never met its close
        // marker, a body-final newline, a partial marker — is trailing text
        // once the generation is over.
        let mut reasoning = String::new();
        let mut content = String::new();
        self.scan_carry(&mut reasoning, &mut content);
        match (non_empty(&reasoning), non_empty(&content)) {
            (None, None) => ReasoningDelta::None,
            (Some(r), None) => ReasoningDelta::Reasoning(r),
            (None, Some(c)) => ReasoningDelta::Content(c),
            (Some(r), Some(c)) => ReasoningDelta::Split {
                reasoning: Some(r),
                content: Some(c),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Look up a reasoning parser by name.
///
/// Requires the tokenizer vocabulary to resolve start/end token IDs.
pub fn get_reasoning_parser(
    name: &str,
    vocab: &HashMap<String, u32>,
) -> Result<Arc<dyn ReasoningParser>, String> {
    match name {
        "deepseek_r1" => Ok(Arc::new(DeepSeekR1ReasoningParser::new(vocab)?)),
        "qwen3" => Ok(Arc::new(Qwen3ReasoningParser::new(vocab)?)),
        "gpt_oss" => Ok(Arc::new(GptOssReasoningParser)),
        "gemma4" => Ok(Arc::new(Gemma4ReasoningParser)),
        other => Err(format!("Unknown reasoning parser: {other}")),
    }
}

/// Auto-select a reasoning parser from the model's **canonical architecture**
/// (`config.json` `model_type` / `architectures`), not its name/path. Only
/// architectures with a known reasoning format get a parser; everything else
/// gets `None` so their output is left untouched as content.
///
/// Caveat: `<think>` reasoning can be a finetune trait rather than an arch one
/// (e.g. R1 *distills* keep their base arch, `qwen2`/`llama`), so those aren't
/// auto-detected here and would need an explicit `--reasoning-parser`.
pub fn detect_reasoning_parser(arch: &str) -> Option<&'static str> {
    let a = arch.to_lowercase();
    if a.contains("gpt_oss") || a.contains("gpt-oss") || a.contains("gptoss") {
        Some("gpt_oss")
    } else if a.contains("gemma4") {
        // Gemma 4 (dense, MoE, and `-unified` ports) wraps reasoning in
        // `<|channel>thought … <channel|>` channels.
        Some("gemma4")
    } else if a.contains("deepseek") {
        // deepseek_v3 / R1 emit `<think>…</think>`.
        Some("deepseek_r1")
    } else if a.contains("granite_4_2") {
        // Granite 4.2 (the first granite with an untied lm_head — see
        // `model_arch` in init.rs) thinks in ` MudThink… MudThinkEnd`, Qwen3's
        // tokens, so it reuses the Qwen3 parser.
        Some("qwen3")
    } else if a.contains("qwen3") || a.contains("qwq") || a.contains("nemotron_h") {
        // Nemotron-H's template opens `<think>\n` in the prompt (`</think>` closed
        // there when thinking is off), Qwen3.5's convention.
        Some("qwen3")
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Strip the start token from the beginning of text, returning the remainder.
/// Returns None if the start token is not present.
fn strip_start_token<'a>(text: &'a str, start_token: &str) -> Option<&'a str> {
    if let Some(idx) = text.find(start_token) {
        Some(&text[idx + start_token.len()..])
    } else {
        None
    }
}

/// Partition text on the first occurrence of `sep`, returning (before, after).
fn partition_on<'a>(text: &'a str, sep: &str) -> (&'a str, &'a str) {
    if let Some(idx) = text.find(sep) {
        (&text[..idx], &text[idx + sep.len()..])
    } else {
        (text, "")
    }
}

/// Return Some(s.to_string()) if s is non-empty, None otherwise.
fn non_empty(s: &str) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    type StreamStep<'a> = (&'a str, &'a str, &'a str, &'a [u32], &'a [u32], &'a [u32]);

    /// Build a minimal vocab with <think> and </think> tokens.
    fn test_vocab() -> HashMap<String, u32> {
        let mut vocab = HashMap::new();
        vocab.insert("<think>".to_string(), 100);
        vocab.insert("</think>".to_string(), 101);
        vocab
    }

    // -- DeepSeek R1 non-streaming --

    #[test]
    fn test_deepseek_r1_basic() {
        let vocab = test_vocab();
        let parser = DeepSeekR1ReasoningParser::new(&vocab).unwrap();
        let result = parser.extract_reasoning("<think>reasoning</think>content");
        assert_eq!(
            result,
            ExtractedReasoning {
                reasoning: Some("reasoning".to_string()),
                content: Some("content".to_string()),
            }
        );
    }

    #[test]
    fn test_deepseek_r1_no_end_token() {
        let vocab = test_vocab();
        let parser = DeepSeekR1ReasoningParser::new(&vocab).unwrap();
        let result = parser.extract_reasoning("<think>reasoning only");
        assert_eq!(
            result,
            ExtractedReasoning {
                reasoning: Some("reasoning only".to_string()),
                content: None,
            }
        );
    }

    #[test]
    fn test_deepseek_r1_empty_content() {
        let vocab = test_vocab();
        let parser = DeepSeekR1ReasoningParser::new(&vocab).unwrap();
        let result = parser.extract_reasoning("<think>reasoning</think>");
        assert_eq!(
            result,
            ExtractedReasoning {
                reasoning: Some("reasoning".to_string()),
                content: None,
            }
        );
    }

    // -- Qwen3 non-streaming --

    #[test]
    fn test_qwen3_basic() {
        let vocab = test_vocab();
        let parser = Qwen3ReasoningParser::new(&vocab).unwrap();
        // No start token — chat template puts it in prompt.
        let result = parser.extract_reasoning("reasoning</think>content");
        assert_eq!(
            result,
            ExtractedReasoning {
                reasoning: Some("reasoning".to_string()),
                content: Some("content".to_string()),
            }
        );
    }

    #[test]
    fn test_qwen3_thinking_disabled() {
        let vocab = test_vocab();
        let parser = Qwen3ReasoningParser::new(&vocab).unwrap();
        let result = parser.extract_reasoning("just content");
        assert_eq!(
            result,
            ExtractedReasoning {
                reasoning: None,
                content: Some("just content".to_string()),
            }
        );
    }

    #[test]
    fn test_qwen3_with_start_token() {
        let vocab = test_vocab();
        let parser = Qwen3ReasoningParser::new(&vocab).unwrap();
        let result = parser.extract_reasoning("<think>reasoning</think>content");
        assert_eq!(
            result,
            ExtractedReasoning {
                reasoning: Some("reasoning".to_string()),
                content: Some("content".to_string()),
            }
        );
    }

    // -- Streaming: DeepSeek R1 --

    #[test]
    fn test_streaming_reasoning_then_content() {
        let vocab = test_vocab();
        let parser = DeepSeekR1ReasoningParser::new(&vocab).unwrap();
        let mut state = parser.create_streaming_state();

        // First delta: <think> token alone — should be None.
        let r = state.process_delta("", "<think>", "<think>", &[], &[100], &[100]);
        assert_eq!(r, ReasoningDelta::None);

        // Second delta: reasoning text.
        let r = state.process_delta("<think>", "<think>hello", "hello", &[100], &[100, 1], &[1]);
        assert_eq!(r, ReasoningDelta::Reasoning("hello".to_string()));

        // Third delta: more reasoning.
        let r = state.process_delta(
            "<think>hello",
            "<think>hello world",
            " world",
            &[100, 1],
            &[100, 1, 2],
            &[2],
        );
        assert_eq!(r, ReasoningDelta::Reasoning(" world".to_string()));

        // Fourth delta: </think> token alone — should be None.
        let r = state.process_delta(
            "<think>hello world",
            "<think>hello world</think>",
            "</think>",
            &[100, 1, 2],
            &[100, 1, 2, 101],
            &[101],
        );
        assert_eq!(r, ReasoningDelta::None);

        // Fifth delta: content after thinking.
        let r = state.process_delta(
            "<think>hello world</think>",
            "<think>hello world</think>answer",
            "answer",
            &[100, 1, 2, 101],
            &[100, 1, 2, 101, 3],
            &[3],
        );
        assert_eq!(r, ReasoningDelta::Content("answer".to_string()));
    }

    #[test]
    fn test_streaming_end_token_in_delta() {
        let vocab = test_vocab();
        let parser = DeepSeekR1ReasoningParser::new(&vocab).unwrap();
        let mut state = parser.create_streaming_state();

        // <think> first.
        let _ = state.process_delta("", "<think>", "<think>", &[], &[100], &[100]);

        // Delta with reasoning text.
        let _ = state.process_delta("<think>", "<think>hi", "hi", &[100], &[100, 1], &[1]);

        // Delta containing </think> and content after it.
        let r = state.process_delta(
            "<think>hi",
            "<think>hi</think>answer",
            "</think>answer",
            &[100, 1],
            &[100, 1, 101, 3],
            &[101, 3],
        );
        assert_eq!(
            r,
            ReasoningDelta::Split {
                reasoning: None,
                content: Some("answer".to_string()),
            }
        );
    }

    #[test]
    fn test_streaming_no_start_token_deepseek() {
        // DeepSeek R1 variant: model starts reasoning without <think>.
        let vocab = test_vocab();
        let parser = DeepSeekR1ReasoningParser::new(&vocab).unwrap();
        let mut state = parser.create_streaming_state();

        // First delta: reasoning without <think>.
        let r = state.process_delta("", "think", "think", &[], &[1], &[1]);
        assert_eq!(r, ReasoningDelta::Reasoning("think".to_string()));

        // More reasoning.
        let r = state.process_delta("think", "thinking", "ing", &[1], &[1, 2], &[2]);
        assert_eq!(r, ReasoningDelta::Reasoning("ing".to_string()));

        // End token appears.
        let r = state.process_delta(
            "thinking",
            "thinking</think>done",
            "</think>done",
            &[1, 2],
            &[1, 2, 101, 3],
            &[101, 3],
        );
        assert_eq!(
            r,
            ReasoningDelta::Split {
                reasoning: None,
                content: Some("done".to_string()),
            }
        );

        // Content after.
        let r = state.process_delta(
            "thinking</think>done",
            "thinking</think>done!",
            "!",
            &[1, 2, 101, 3],
            &[1, 2, 101, 3, 4],
            &[4],
        );
        assert_eq!(r, ReasoningDelta::Content("!".to_string()));
    }

    // -- Streaming: Qwen3 --

    #[test]
    fn test_streaming_qwen3_thinking_disabled() {
        // No think tokens at all — everything routes as content.
        let vocab = test_vocab();
        let parser = Qwen3ReasoningParser::new(&vocab).unwrap();
        let mut state = parser.create_streaming_state();

        let r = state.process_delta("", "hello", "hello", &[], &[1], &[1]);
        assert_eq!(r, ReasoningDelta::Reasoning("hello".to_string()));
        // Note: Without end token in previous_token_ids, Qwen3 treats as reasoning.
        // This matches Python: when thinking is disabled, the serving layer
        // detects via prompt_is_reasoning_end and routes as content.
        // But if the serving layer doesn't detect it, the parser still returns reasoning.
        // In practice, when thinking is truly disabled, the template puts
        // <think>\n\n</think>\n\n in the prompt, so end_token_id IS in previous.
    }

    #[test]
    fn test_streaming_qwen3_basic() {
        let vocab = test_vocab();
        let parser = Qwen3ReasoningParser::new(&vocab).unwrap();
        let mut state = parser.create_streaming_state();

        // Reasoning phase (no end token yet).
        let r = state.process_delta("", "thinking", "thinking", &[], &[1], &[1]);
        assert_eq!(r, ReasoningDelta::Reasoning("thinking".to_string()));

        // End token with content.
        let r = state.process_delta(
            "thinking",
            "thinking</think>answer",
            "</think>answer",
            &[1],
            &[1, 101, 2],
            &[101, 2],
        );
        assert_eq!(
            r,
            ReasoningDelta::Split {
                reasoning: None,
                content: Some("answer".to_string()),
            }
        );

        // More content.
        let r = state.process_delta(
            "thinking</think>answer",
            "thinking</think>answer!",
            "!",
            &[1, 101, 2],
            &[1, 101, 2, 3],
            &[3],
        );
        assert_eq!(r, ReasoningDelta::Content("!".to_string()));
    }

    #[test]
    fn test_streaming_qwen3_strips_start_token() {
        let vocab = test_vocab();
        let parser = Qwen3ReasoningParser::new(&vocab).unwrap();
        let mut state = parser.create_streaming_state();

        // Delta contains <think> — should be stripped.
        let r = state.process_delta("", "<think>", "<think>", &[], &[100], &[100]);
        assert_eq!(r, ReasoningDelta::None); // Empty after stripping.

        // Reasoning continues.
        let r = state.process_delta("<think>", "<think>hi", "hi", &[100], &[100, 1], &[1]);
        assert_eq!(r, ReasoningDelta::Reasoning("hi".to_string()));
    }

    // -- Registry tests --

    #[test]
    fn test_registry_deepseek_r1() {
        let vocab = test_vocab();
        assert!(get_reasoning_parser("deepseek_r1", &vocab).is_ok());
    }

    #[test]
    fn test_registry_qwen3() {
        let vocab = test_vocab();
        assert!(get_reasoning_parser("qwen3", &vocab).is_ok());
    }

    #[test]
    fn test_registry_unknown() {
        let vocab = test_vocab();
        assert!(get_reasoning_parser("unknown", &vocab).is_err());
    }

    // -- Additional edge case tests --

    #[test]
    fn test_deepseek_r1_no_start_token() {
        // Model output without <think> — everything is reasoning (no end token).
        let vocab = test_vocab();
        let parser = DeepSeekR1ReasoningParser::new(&vocab).unwrap();
        let result = parser.extract_reasoning("just thinking out loud");
        assert_eq!(
            result,
            ExtractedReasoning {
                reasoning: Some("just thinking out loud".to_string()),
                content: None,
            }
        );
    }

    #[test]
    fn test_deepseek_r1_no_start_with_end() {
        // Model output without <think> but with </think> — reasoning ends.
        let vocab = test_vocab();
        let parser = DeepSeekR1ReasoningParser::new(&vocab).unwrap();
        let result = parser.extract_reasoning("thinking</think>answer");
        assert_eq!(
            result,
            ExtractedReasoning {
                reasoning: Some("thinking".to_string()),
                content: Some("answer".to_string()),
            }
        );
    }

    #[test]
    fn test_deepseek_r1_empty_reasoning() {
        let vocab = test_vocab();
        let parser = DeepSeekR1ReasoningParser::new(&vocab).unwrap();
        let result = parser.extract_reasoning("<think></think>content");
        assert_eq!(
            result,
            ExtractedReasoning {
                reasoning: Some("".to_string()),
                content: Some("content".to_string()),
            }
        );
    }

    #[test]
    fn test_qwen3_empty_output() {
        let vocab = test_vocab();
        let parser = Qwen3ReasoningParser::new(&vocab).unwrap();
        let result = parser.extract_reasoning("");
        assert_eq!(
            result,
            ExtractedReasoning {
                reasoning: None,
                content: Some("".to_string()),
            }
        );
    }

    #[test]
    fn test_qwen3_only_end_token() {
        let vocab = test_vocab();
        let parser = Qwen3ReasoningParser::new(&vocab).unwrap();
        let result = parser.extract_reasoning("</think>");
        assert_eq!(
            result,
            ExtractedReasoning {
                reasoning: Some("".to_string()),
                content: None,
            }
        );
    }

    #[test]
    fn test_qwen3_multiline_reasoning() {
        let vocab = test_vocab();
        let parser = Qwen3ReasoningParser::new(&vocab).unwrap();
        let result = parser.extract_reasoning("line1\nline2\nline3</think>The answer is 42.");
        assert_eq!(
            result,
            ExtractedReasoning {
                reasoning: Some("line1\nline2\nline3".to_string()),
                content: Some("The answer is 42.".to_string()),
            }
        );
    }

    #[test]
    fn test_missing_vocab_token() {
        // Vocab without </think> should fail.
        let mut vocab = HashMap::new();
        vocab.insert("<think>".to_string(), 100);
        assert!(DeepSeekR1ReasoningParser::new(&vocab).is_err());
        assert!(Qwen3ReasoningParser::new(&vocab).is_err());
    }

    // -- Streaming: more edge cases --

    #[test]
    fn test_streaming_deepseek_both_tokens_in_one_delta() {
        // Both <think> and </think> in a single delta.
        let vocab = test_vocab();
        let parser = DeepSeekR1ReasoningParser::new(&vocab).unwrap();
        let mut state = parser.create_streaming_state();

        let r = state.process_delta(
            "",
            "<think>reasoning</think>content",
            "<think>reasoning</think>content",
            &[],
            &[100, 1, 101, 2],
            &[100, 1, 101, 2],
        );
        assert_eq!(
            r,
            ReasoningDelta::Split {
                reasoning: Some("reasoning".to_string()),
                content: Some("content".to_string()),
            }
        );
    }

    #[test]
    fn test_streaming_deepseek_end_token_splits_with_reasoning() {
        // Delta has reasoning text before </think> and content after.
        let vocab = test_vocab();
        let parser = DeepSeekR1ReasoningParser::new(&vocab).unwrap();
        let mut state = parser.create_streaming_state();

        // <think> first.
        let _ = state.process_delta("", "<think>", "<think>", &[], &[100], &[100]);

        // Delta with reasoning + </think> + content.
        let r = state.process_delta(
            "<think>",
            "<think>more</think>result",
            "more</think>result",
            &[100],
            &[100, 1, 101, 2],
            &[1, 101, 2],
        );
        assert_eq!(
            r,
            ReasoningDelta::Split {
                reasoning: Some("more".to_string()),
                content: Some("result".to_string()),
            }
        );
    }

    #[test]
    fn test_streaming_qwen3_end_token_no_content_after() {
        // </think> at end of delta, nothing after.
        let vocab = test_vocab();
        let parser = Qwen3ReasoningParser::new(&vocab).unwrap();
        let mut state = parser.create_streaming_state();

        // Reasoning first.
        let r = state.process_delta("", "thinking", "thinking", &[], &[1], &[1]);
        assert_eq!(r, ReasoningDelta::Reasoning("thinking".to_string()));

        // Just </think>, nothing else.
        let r = state.process_delta(
            "thinking",
            "thinking</think>",
            "</think>",
            &[1],
            &[1, 101],
            &[101],
        );
        // </think> with nothing before or after → no content to emit.
        assert_eq!(r, ReasoningDelta::None);
    }

    #[test]
    fn test_streaming_qwen3_content_after_end() {
        // Verify multiple content deltas after </think>.
        let vocab = test_vocab();
        let parser = Qwen3ReasoningParser::new(&vocab).unwrap();
        let mut state = parser.create_streaming_state();

        // Reasoning.
        let _ = state.process_delta("", "think", "think", &[], &[1], &[1]);

        // End token.
        let _ = state.process_delta(
            "think",
            "think</think>",
            "</think>",
            &[1],
            &[1, 101],
            &[101],
        );

        // Content delta 1.
        let r = state.process_delta(
            "think</think>",
            "think</think>hello",
            "hello",
            &[1, 101],
            &[1, 101, 2],
            &[2],
        );
        assert_eq!(r, ReasoningDelta::Content("hello".to_string()));

        // Content delta 2.
        let r = state.process_delta(
            "think</think>hello",
            "think</think>hello world",
            " world",
            &[1, 101, 2],
            &[1, 101, 2, 3],
            &[3],
        );
        assert_eq!(r, ReasoningDelta::Content(" world".to_string()));
    }

    #[test]
    fn test_streaming_deepseek_long_reasoning_sequence() {
        // Multiple reasoning deltas before end.
        let vocab = test_vocab();
        let parser = DeepSeekR1ReasoningParser::new(&vocab).unwrap();
        let mut state = parser.create_streaming_state();

        // <think>
        let _ = state.process_delta("", "<think>", "<think>", &[], &[100], &[100]);

        // Multiple reasoning tokens.
        for i in 0..5 {
            let prev_ids: Vec<u32> = {
                let mut ids = vec![100];
                ids.extend(1..=i);
                ids
            };
            let mut curr_ids = prev_ids.clone();
            curr_ids.push(i + 1);
            let delta_text = format!("word{i} ");
            let r = state.process_delta("", "", &delta_text, &prev_ids, &curr_ids, &[i + 1]);
            assert_eq!(r, ReasoningDelta::Reasoning(delta_text));
        }
    }

    // -- Test reasoning extraction combined with tool parsing --

    #[test]
    fn test_reasoning_then_tool_extraction() {
        // Simulates the non-streaming path: reasoning parser extracts reasoning,
        // then tool parser runs on the content portion.
        let vocab = test_vocab();
        let reasoning_parser = DeepSeekR1ReasoningParser::new(&vocab).unwrap();

        let full_output = "<think>Let me think about this</think><tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}</tool_call>";

        // Step 1: Extract reasoning.
        let extracted = reasoning_parser.extract_reasoning(full_output);
        assert_eq!(
            extracted.reasoning,
            Some("Let me think about this".to_string())
        );

        // Step 2: Content portion goes to tool parser.
        let content = extracted.content.unwrap();
        assert!(content.contains("<tool_call>"));
        assert!(content.contains("get_weather"));
    }

    #[test]
    fn test_qwen3_reasoning_then_tool_extraction() {
        let vocab = test_vocab();
        let reasoning_parser = Qwen3ReasoningParser::new(&vocab).unwrap();

        // Qwen3: no <think> in output (template puts it in prompt).
        let full_output =
            "I need to check the weather</think><tool_call>{\"name\":\"search\"}</tool_call>";

        let extracted = reasoning_parser.extract_reasoning(full_output);
        assert_eq!(
            extracted.reasoning,
            Some("I need to check the weather".to_string())
        );

        let content = extracted.content.unwrap();
        assert!(content.contains("<tool_call>"));
    }

    #[test]
    fn test_reasoning_no_tools_in_content() {
        // When reasoning is present but content has no tool calls.
        let vocab = test_vocab();
        let parser = DeepSeekR1ReasoningParser::new(&vocab).unwrap();

        let extracted =
            parser.extract_reasoning("<think>step 1, step 2, step 3</think>The answer is 42.");
        assert_eq!(
            extracted,
            ExtractedReasoning {
                reasoning: Some("step 1, step 2, step 3".to_string()),
                content: Some("The answer is 42.".to_string()),
            }
        );
    }

    // -- Streaming: full sequence simulation tests --

    #[test]
    fn test_streaming_full_sequence_deepseek() {
        // Simulate a complete DeepSeek R1 generation:
        // <think>Let me think</think>The answer is 42.
        let vocab = test_vocab();
        let parser = DeepSeekR1ReasoningParser::new(&vocab).unwrap();
        let mut state = parser.create_streaming_state();

        let steps: Vec<StreamStep<'_>> = vec![
            // (prev_text, curr_text, delta, prev_ids, curr_ids, delta_ids)
            ("", "<think>", "<think>", &[], &[100], &[100]),
            ("<think>", "<think>Let ", "Let ", &[100], &[100, 10], &[10]),
            (
                "<think>Let ",
                "<think>Let me ",
                "me ",
                &[100, 10],
                &[100, 10, 11],
                &[11],
            ),
            (
                "<think>Let me ",
                "<think>Let me think",
                "think",
                &[100, 10, 11],
                &[100, 10, 11, 12],
                &[12],
            ),
            (
                "<think>Let me think",
                "<think>Let me think</think>",
                "</think>",
                &[100, 10, 11, 12],
                &[100, 10, 11, 12, 101],
                &[101],
            ),
            (
                "<think>Let me think</think>",
                "<think>Let me think</think>The ",
                "The ",
                &[100, 10, 11, 12, 101],
                &[100, 10, 11, 12, 101, 20],
                &[20],
            ),
            (
                "<think>Let me think</think>The ",
                "<think>Let me think</think>The answer",
                "answer",
                &[100, 10, 11, 12, 101, 20],
                &[100, 10, 11, 12, 101, 20, 21],
                &[21],
            ),
        ];

        let mut reasoning_parts = Vec::new();
        let mut content_parts = Vec::new();

        for (prev_text, curr_text, delta, prev_ids, curr_ids, delta_ids) in steps {
            let r = state.process_delta(prev_text, curr_text, delta, prev_ids, curr_ids, delta_ids);
            match r {
                ReasoningDelta::Reasoning(s) => reasoning_parts.push(s),
                ReasoningDelta::Content(s) => content_parts.push(s),
                ReasoningDelta::Split { reasoning, content } => {
                    if let Some(r) = reasoning {
                        reasoning_parts.push(r);
                    }
                    if let Some(c) = content {
                        content_parts.push(c);
                    }
                }
                ReasoningDelta::None => {}
            }
        }

        assert_eq!(reasoning_parts.join(""), "Let me think");
        assert_eq!(content_parts.join(""), "The answer");
    }

    #[test]
    fn test_streaming_full_sequence_qwen3() {
        // Simulate a complete Qwen3 generation (no <think> in output):
        // reasoning text</think>content text
        let vocab = test_vocab();
        let parser = Qwen3ReasoningParser::new(&vocab).unwrap();
        let mut state = parser.create_streaming_state();

        let steps: Vec<StreamStep<'_>> = vec![
            ("", "I need ", "I need ", &[], &[10], &[10]),
            ("I need ", "I need to ", "to ", &[10], &[10, 11], &[11]),
            (
                "I need to ",
                "I need to think",
                "think",
                &[10, 11],
                &[10, 11, 12],
                &[12],
            ),
            (
                "I need to think",
                "I need to think</think>",
                "</think>",
                &[10, 11, 12],
                &[10, 11, 12, 101],
                &[101],
            ),
            (
                "I need to think</think>",
                "I need to think</think>Answer",
                "Answer",
                &[10, 11, 12, 101],
                &[10, 11, 12, 101, 20],
                &[20],
            ),
        ];

        let mut reasoning_parts = Vec::new();
        let mut content_parts = Vec::new();

        for (prev_text, curr_text, delta, prev_ids, curr_ids, delta_ids) in steps {
            let r = state.process_delta(prev_text, curr_text, delta, prev_ids, curr_ids, delta_ids);
            match r {
                ReasoningDelta::Reasoning(s) => reasoning_parts.push(s),
                ReasoningDelta::Content(s) => content_parts.push(s),
                ReasoningDelta::Split { reasoning, content } => {
                    if let Some(r) = reasoning {
                        reasoning_parts.push(r);
                    }
                    if let Some(c) = content {
                        content_parts.push(c);
                    }
                }
                ReasoningDelta::None => {}
            }
        }

        assert_eq!(reasoning_parts.join(""), "I need to think");
        assert_eq!(content_parts.join(""), "Answer");
    }

    #[test]
    fn test_streaming_qwen3_thinking_disabled_full_sequence() {
        // When thinking is disabled, end_token_id appears in previous_token_ids
        // (from the prompt template: <think>\n\n</think>\n\n).
        let vocab = test_vocab();
        let parser = Qwen3ReasoningParser::new(&vocab).unwrap();
        let mut state = parser.create_streaming_state();

        // previous_token_ids includes end_token_id (from prompt).
        let r = state.process_delta(
            "",
            "Hello",
            "Hello",
            &[100, 50, 50, 101, 50, 50], // prompt included <think>\n\n</think>\n\n
            &[100, 50, 50, 101, 50, 50, 10],
            &[10],
        );
        assert_eq!(r, ReasoningDelta::Content("Hello".to_string()));

        let r = state.process_delta(
            "Hello",
            "Hello world",
            " world",
            &[100, 50, 50, 101, 50, 50, 10],
            &[100, 50, 50, 101, 50, 50, 10, 11],
            &[11],
        );
        assert_eq!(r, ReasoningDelta::Content(" world".to_string()));
    }
}

#[cfg(test)]
mod gpt_oss_tests {
    use super::*;

    #[test]
    fn splits_analysis_and_final_channels() {
        let p = GptOssReasoningParser;
        let out = "<|channel|>analysis<|message|>weighing options<|end|>\
                   <|start|>assistant<|channel|>final<|message|>the answer";
        let r = p.extract_reasoning(out);
        assert_eq!(r.reasoning.as_deref(), Some("weighing options"));
        assert_eq!(r.content.as_deref(), Some("the answer"));
    }

    /// The reasoning and content a stream of `pieces` adds up to, its end-of-stream flush
    /// included.
    fn stream(pieces: &[&str]) -> (String, String) {
        let mut state = GptOssReasoningParser.create_streaming_state();
        let (mut reasoning, mut content) = (String::new(), String::new());
        let mut text = String::new();
        for piece in pieces {
            let previous = text.clone();
            text.push_str(piece);
            let delta = state.process_delta(&previous, &text, piece, &[], &[], &[]);
            let (r, c) = delta.into_parts();
            reasoning.extend(r);
            content.extend(c);
        }
        let (r, c) = state.flush().into_parts();
        reasoning.extend(r);
        content.extend(c);
        (reasoning, content)
    }

    /// A streamed harmony reply, one token per delta, splits as the complete reply does: channel
    /// names and the role never stream as reasoning or content (#601: they did, "analysis"
    /// first).
    #[test]
    fn a_streamed_reply_splits_as_the_complete_reply() {
        let pieces = [
            "<|channel|>",
            "analysis",
            "<|message|>",
            "weighing",
            " options",
            "<|end|>",
            "<|start|>",
            "assistant",
            "<|channel|>",
            "final",
            "<|message|>",
            "the",
            " answer",
        ];
        assert_eq!(
            stream(&pieces),
            ("weighing options".to_string(), "the answer".to_string())
        );
        let whole = GptOssReasoningParser.extract_reasoning(&pieces.concat());
        assert_eq!(whole.reasoning.as_deref(), Some("weighing options"));
        assert_eq!(whole.content.as_deref(), Some("the answer"));
    }

    /// A marker a quant port writes as several text pieces is held until it closes; a `<` that
    /// opens none is text.
    #[test]
    fn a_marker_split_across_deltas_still_matches() {
        let pieces = [
            "<|chan",
            "nel|>analysis<|mes",
            "sage|>is a < b?",
            "<|channel|>final<",
            "|message|>yes",
        ];
        assert_eq!(
            stream(&pieces),
            ("is a < b?".to_string(), "yes".to_string())
        );
        // A `<` the stream ends on opens nothing: it is text.
        assert_eq!(
            stream(&["no markers <"]),
            (String::new(), "no markers <".into())
        );
    }

    #[test]
    fn no_channels_is_all_content() {
        let p = GptOssReasoningParser;
        let r = p.extract_reasoning("plain answer, no channels");
        assert_eq!(r.reasoning, None);
        assert_eq!(r.content.as_deref(), Some("plain answer, no channels"));
    }

    #[test]
    fn tolerates_asymmetric_pipe_markers() {
        // Quantized ports emit the pipe on one side only: <|channel> … <channel|>.
        let p = GptOssReasoningParser;
        let out = "<|channel>analysis<channel|>weighing the git status\
                   <|channel>final<channel|>done";
        let r = p.extract_reasoning(out);
        assert_eq!(r.reasoning.as_deref(), Some("weighing the git status"));
        assert_eq!(r.content.as_deref(), Some("done"));
    }

    #[test]
    fn detects_by_model_id() {
        // Keyed on model_type / architecture.
        assert_eq!(detect_reasoning_parser("gpt_oss"), Some("gpt_oss"));
        assert_eq!(detect_reasoning_parser("qwen3"), Some("qwen3"));
        assert_eq!(detect_reasoning_parser("qwen3_5_moe"), Some("qwen3"));
        assert_eq!(detect_reasoning_parser("nemotron_h"), Some("qwen3"));
        assert_eq!(detect_reasoning_parser("deepseek_v3"), Some("deepseek_r1"));
        // Gemma 4's thought channels (`<|channel>thought … <channel|>`),
        // dense and MoE alike.
        assert_eq!(detect_reasoning_parser("gemma4"), Some("gemma4"));
        assert_eq!(detect_reasoning_parser("gemma4_moe"), Some("gemma4"));
        assert_eq!(detect_reasoning_parser("gemma4_unified"), Some("gemma4"));
        // Granite 4.2 (untied lm_head) thinks; ≤4.1 does not.
        assert_eq!(detect_reasoning_parser("granite_4_2"), Some("qwen3"));
        assert_eq!(detect_reasoning_parser("granite"), None);
        // Architectures with no known reasoning format get no parser.
        assert_eq!(detect_reasoning_parser("gemma3"), None);
        assert_eq!(detect_reasoning_parser("llama"), None);
        assert_eq!(detect_reasoning_parser("qwen2"), None);
    }
}

#[cfg(test)]
mod gemma4_tests {
    use super::*;

    /// Drive a streaming state token-by-token (one delta per piece) and
    /// collect (reasoning, content).
    fn stream(pieces: &[&str]) -> (String, String) {
        let parser = Gemma4ReasoningParser;
        let mut state = parser.create_streaming_state();
        let mut reasoning = String::new();
        let mut content = String::new();
        for p in pieces {
            let r = state.process_delta("", p, p, &[], &[], &[]);
            match r {
                ReasoningDelta::Reasoning(s) => reasoning.push_str(&s),
                ReasoningDelta::Content(c) => content.push_str(&c),
                ReasoningDelta::Split {
                    reasoning: Some(r),
                    content: Some(c),
                } => {
                    reasoning.push_str(&r);
                    content.push_str(&c);
                }
                ReasoningDelta::Split {
                    reasoning: Some(r),
                    content: None,
                } => reasoning.push_str(&r),
                ReasoningDelta::Split {
                    reasoning: None,
                    content: Some(c),
                } => content.push_str(&c),
                _ => {}
            }
        }
        (reasoning, content)
    }

    #[test]
    fn thought_channel_then_answer() {
        let p = Gemma4ReasoningParser;
        let r = p.extract_reasoning("<|channel>thought\nweighing options\n<channel|>the answer");
        assert_eq!(r.reasoning.as_deref(), Some("weighing options"));
        assert_eq!(r.content.as_deref(), Some("the answer"));
    }

    #[test]
    fn content_then_empty_channel_then_tool_call() {
        // The live `git status` capture: text, an empty thought channel, then
        // the tool call. The channel's markers AND its name must never surface.
        let p = Gemma4ReasoningParser;
        let out = "```bash\ngit status\n```\n<|channel>thought\n<channel|>\
                   <|tool_call>call:Bash{command:<|\"|>git status<|\"|>}<tool_call|>";
        let r = p.extract_reasoning(out);
        assert_eq!(r.reasoning, None);
        assert_eq!(
            r.content.as_deref(),
            Some(
                "```bash\ngit status\n```\n<|tool_call>call:Bash{command:<|\"|>git status<|\"|>}<tool_call|>"
            )
        );
    }

    #[test]
    fn channel_open_loop_is_fully_consumed() {
        // The reported flood: the model re-opens the thought channel hundreds
        // of times without closing. The channel *names* must never surface.
        // The trailing text is the body of the never-closed thought channel,
        // so it routes to reasoning (same convention as gpt_oss: an
        // unterminated reasoning channel runs to the end of the output).
        let p = Gemma4ReasoningParser;
        let mut out = String::new();
        for _ in 0..227 {
            out.push_str("<|channel>thought\n");
        }
        out.push_str("finally, the answer");
        let r = p.extract_reasoning(&out);
        assert_eq!(r.reasoning.as_deref(), Some("finally, the answer"));
        assert_eq!(r.content, None);
    }

    #[test]
    fn channel_open_loop_then_close_recovers_content() {
        // The same flood, but the model eventually closes the channel: the
        // answer after the close marker is content again.
        let p = Gemma4ReasoningParser;
        let mut out = String::new();
        for _ in 0..227 {
            out.push_str("<|channel>thought\n");
        }
        out.push_str("<channel|>finally, the answer");
        let r = p.extract_reasoning(&out);
        assert_eq!(r.reasoning, None);
        assert_eq!(r.content.as_deref(), Some("finally, the answer"));
    }

    #[test]
    fn no_channels_is_all_content() {
        let p = Gemma4ReasoningParser;
        let r = p.extract_reasoning("plain answer, no channels");
        assert_eq!(r.reasoning, None);
        assert_eq!(r.content.as_deref(), Some("plain answer, no channels"));
    }

    #[test]
    fn non_reasoning_channel_body_is_content() {
        // A channel whose name is not a reasoning channel (Gemma 4 only emits
        // `thought` today) routes its body as content rather than swallowing it.
        let p = Gemma4ReasoningParser;
        let r = p.extract_reasoning("<|channel>final\nvisible body\n<channel|>tail");
        assert_eq!(r.reasoning, None);
        assert_eq!(r.content.as_deref(), Some("visible body\ntail"));
    }

    #[test]
    fn whitespace_variant_markers_match() {
        // Quantized ports sometimes emit the markers as ordinary text with
        // stray spaces between the pieces.
        let p = Gemma4ReasoningParser;
        let r = p.extract_reasoning("< |channel>thought\ndeliberating\n< channel|>answer");
        assert_eq!(r.reasoning.as_deref(), Some("deliberating"));
        assert_eq!(r.content.as_deref(), Some("answer"));
    }

    #[test]
    fn streaming_matches_extract_token_by_token() {
        let pieces: Vec<&str> = vec![
            "```bash",
            "\ngit",
            " status",
            "\n",
            "```",
            "\n",
            "<|channel>",
            "thought",
            "\n",
            "<channel|>",
            "<|tool_call>",
            "call:",
            "Bash",
            "{",
            "command:",
            "<|\"|>",
            "git",
            " status",
            "<|\"|>",
            "}",
            "<tool_call|>",
        ];
        let (reasoning, content) = stream(&pieces);
        assert_eq!(reasoning, "");
        assert_eq!(
            content,
            "```bash\ngit status\n```\n<|tool_call>call:Bash{command:<|\"|>git status<|\"|>}<tool_call|>"
        );
    }

    #[test]
    fn streaming_thought_first() {
        let pieces: Vec<&str> = vec![
            "<|channel>",
            "thought\n",
            "Let",
            " me",
            " check",
            "\n",
            "<channel|>",
            "The ",
            "answer",
        ];
        let (reasoning, content) = stream(&pieces);
        assert_eq!(reasoning, "Let me check");
        assert_eq!(content, "The answer");
    }

    #[test]
    fn streaming_marker_split_across_deltas() {
        let pieces: Vec<&str> = vec!["text<", "|chan", "nel>thought\nbody\n<channel", "|>done"];
        let (reasoning, content) = stream(&pieces);
        assert_eq!(reasoning, "body");
        assert_eq!(content, "textdone");
    }

    #[test]
    fn streaming_open_loop_token_by_token() {
        // Unterminated channel: the trailing text streams as reasoning.
        let mut pieces: Vec<&str> = Vec::new();
        for _ in 0..50 {
            pieces.extend(vec!["<|channel>", "thought", "\n"]);
        }
        pieces.push("answer at last");
        let (reasoning, content) = stream(&pieces);
        assert_eq!(reasoning, "answer at last");
        assert_eq!(content, "");
    }

    #[test]
    fn streaming_open_loop_then_close_recovers_content() {
        let mut pieces: Vec<&str> = Vec::new();
        for _ in 0..50 {
            pieces.extend(vec!["<|channel>", "thought", "\n"]);
        }
        pieces.extend(vec!["<channel|>", "answer at last"]);
        let (reasoning, content) = stream(&pieces);
        assert_eq!(reasoning, "");
        assert_eq!(content, "answer at last");
    }

    #[test]
    fn bare_close_flood_is_consumed() {
        // The flood's late stage, measured on the live stream: the model
        // drops the `<|channel>` opens and emits `thought\n<channel|>`
        // cycles. Nothing may reach content or reasoning — not the names,
        // not the newlines, not the marker text.
        let p = Gemma4ReasoningParser;
        let mut out = String::new();
        for _ in 0..200 {
            out.push_str("thought\n<channel|>");
        }
        out.push_str("finally, the answer");
        let r = p.extract_reasoning(&out);
        assert_eq!(r.reasoning, None);
        assert_eq!(r.content.as_deref(), Some("finally, the answer"));
    }

    #[test]
    fn bare_close_flood_token_by_token() {
        // The same flood as the detokenizer splits it: the name, the
        // separator newline, and the close marker arrive as three deltas.
        let mut pieces: Vec<&str> = Vec::new();
        for _ in 0..50 {
            pieces.extend(vec!["thought", "\n", "<channel|>"]);
        }
        pieces.push("answer at last");
        let (reasoning, content) = stream(&pieces);
        assert_eq!(reasoning, "");
        assert_eq!(content, "answer at last");
    }

    #[test]
    fn bare_close_flood_with_sporadic_opens() {
        // The live flood still emits a proper empty channel now and then
        // between the open-less cycles.
        let mut pieces: Vec<&str> = Vec::new();
        for k in 0..60 {
            if k % 7 == 0 {
                pieces.extend(vec!["<|channel>", "thought", "\n", "<channel|>"]);
            }
            pieces.extend(vec!["thought", "\n", "<channel|>"]);
        }
        pieces.push("answer at last");
        let (reasoning, content) = stream(&pieces);
        assert_eq!(reasoning, "");
        assert_eq!(content, "answer at last");
    }

    #[test]
    fn stray_close_never_reaches_content() {
        // Marker text is syntax wherever it appears — with a channel open or
        // not — and must never stream as text.
        let (reasoning, content) = stream(&["<channel|><|channel><channel|>", "hi"]);
        assert_eq!(reasoning, "");
        assert_eq!(content, "hi");
        let p = Gemma4ReasoningParser;
        let r = p.extract_reasoning("a<channel|>b");
        assert_eq!(r.content.as_deref(), Some("ab"));
    }

    #[test]
    fn held_name_released_when_no_close_follows() {
        // `thought` as ordinary prose is held one delta and then released.
        let (reasoning, content) = stream(&["I had a ", "thought", " about it"]);
        assert_eq!(reasoning, "");
        assert_eq!(content, "I had a thought about it");
    }

    #[test]
    fn prose_containing_the_word_is_not_held() {
        // Only a line that IS the name is held; "I had a thought" streams
        // untouched even when a close marker follows in a later delta.
        let (reasoning, content) = stream(&["I had a thought", "!", "<channel|>ok"]);
        assert_eq!(reasoning, "");
        assert_eq!(content, "I had a thought!ok");
    }

    #[test]
    fn flush_releases_held_text_at_end_of_stream() {
        // A reply whose final line is the word must not lose it: the engine
        // folds the parser's flush into the final delta.
        let parser = Gemma4ReasoningParser;
        let mut state = parser.create_streaming_state();
        let _ = state.process_delta("", "", "one ", &[], &[], &[]);
        let _ = state.process_delta("", "", "thought", &[], &[], &[]);
        let flushed = state.flush();
        match flushed {
            ReasoningDelta::Content(c) => assert_eq!(c, "thought"),
            other => panic!("expected flushed content, got {other:?}"),
        }
    }

    #[test]
    fn streaming_trailing_partial_marker_flushes_on_extract() {
        // extract_reasoning is the complete-generation path: a trailing `<`
        // that streaming would hold in carry is plain text.
        let p = Gemma4ReasoningParser;
        let r = p.extract_reasoning("ends with a partial <");
        assert_eq!(r.content.as_deref(), Some("ends with a partial <"));
    }

    #[test]
    fn gemma4_declares_interleaved_and_special_tokens() {
        let parser = Gemma4ReasoningParser;
        assert!(parser.requires_special_tokens());
        assert!(parser.create_streaming_state().interleaved());
        // Harmony's messages interleave too, and its markers are special tokens in gpt-oss's
        // tokenizer.
        assert!(GptOssReasoningParser.requires_special_tokens());
        assert!(GptOssReasoningParser.create_streaming_state().interleaved());
    }
}
