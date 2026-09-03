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

impl GptOssReasoningParser {
    const CHANNEL: &'static str = "<|channel|>";
}

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

impl ReasoningParser for GptOssReasoningParser {
    fn extract_reasoning(&self, model_output: &str) -> ExtractedReasoning {
        // What the text since the last marker represents.
        enum Prev {
            Body,
            ChannelName,
            Role,
        }
        let mut reasoning = String::new();
        let mut content = String::new();
        let mut target_reasoning = false;
        let mut prev = Prev::Body;
        let mut saw_marker = false;
        let mut seg = String::new();

        let chars: Vec<char> = model_output.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '<'
                && let Some(rel) = chars[i + 1..].iter().position(|&c| c == '>')
            {
                let close = i + 1 + rel;
                let inner: String = chars[i + 1..close].iter().collect();
                let marker = classify_marker(&inner);
                if marker != HarmonyMarker::NotMarker {
                    match prev {
                        Prev::ChannelName => target_reasoning = is_reasoning_channel(seg.trim()),
                        Prev::Role => {} // the role token (e.g. "assistant") is dropped
                        Prev::Body => {
                            if target_reasoning {
                                reasoning.push_str(&seg);
                            } else {
                                content.push_str(&seg);
                            }
                        }
                    }
                    seg.clear();
                    saw_marker = true;
                    prev = match marker {
                        HarmonyMarker::Channel => Prev::ChannelName,
                        HarmonyMarker::Start => Prev::Role,
                        _ => Prev::Body,
                    };
                    i = close + 1;
                    continue;
                }
            }
            seg.push(chars[i]);
            i += 1;
        }
        // Trailing segment: only a body contributes (a dangling name/role is dropped).
        if let Prev::Body = prev {
            if target_reasoning {
                reasoning.push_str(&seg);
            } else {
                content.push_str(&seg);
            }
        }

        if !saw_marker {
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
        Box::new(GptOssStreamingState)
    }
}

struct GptOssStreamingState;

impl StreamingReasoningParserState for GptOssStreamingState {
    fn process_delta(
        &mut self,
        _previous_text: &str,
        current_text: &str,
        delta_text: &str,
        _previous_token_ids: &[u32],
        _current_token_ids: &[u32],
        _delta_token_ids: &[u32],
    ) -> ReasoningDelta {
        // Suppress fragments that carry a harmony control marker.
        if delta_text.contains("<|") || delta_text.contains("|>") {
            return ReasoningDelta::None;
        }
        // Route by the channel opened most recently in the accumulated text.
        let in_final = current_text
            .rsplit(GptOssReasoningParser::CHANNEL)
            .next()
            .map(|seg| seg.trim_start().starts_with("final"))
            .unwrap_or(false);
        if delta_text.is_empty() {
            ReasoningDelta::None
        } else if in_final {
            ReasoningDelta::Content(delta_text.to_string())
        } else {
            ReasoningDelta::Reasoning(delta_text.to_string())
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
    } else if a.contains("deepseek") {
        // deepseek_v3 / R1 emit `<think>…</think>`.
        Some("deepseek_r1")
    } else if a.contains("qwen3") || a.contains("qwq") {
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
        assert_eq!(detect_reasoning_parser("deepseek_v3"), Some("deepseek_r1"));
        // Architectures with no known reasoning format get no parser.
        assert_eq!(detect_reasoning_parser("gemma4"), None);
        assert_eq!(detect_reasoning_parser("llama"), None);
        assert_eq!(detect_reasoning_parser("qwen2"), None);
    }
}
