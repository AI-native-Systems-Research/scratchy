// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Assertion helpers for validating OpenAI-compatible API responses.

use std::collections::HashMap;

use scratchy_serving_api::protocol::{
    ChatCompletionResponse, ChatCompletionStreamResponse, CompletionResponse,
};

/// Assert that a chat completion response is well-formed.
pub fn assert_valid_chat_response(resp: &ChatCompletionResponse) {
    // ID should start with "chatcmpl-"
    assert!(
        resp.id.starts_with("chatcmpl-"),
        "id should start with 'chatcmpl-', got: {}",
        resp.id
    );
    assert_eq!(resp.object, "chat.completion");
    assert!(resp.created > 0, "created timestamp should be positive");
    assert!(!resp.model.is_empty(), "model should not be empty");
    assert!(!resp.choices.is_empty(), "should have at least one choice");

    for choice in &resp.choices {
        assert_eq!(choice.message.role, "assistant");
        // finish_reason should be set
        assert!(
            choice.finish_reason.is_some(),
            "finish_reason should be set"
        );
    }

    // Usage should be populated
    assert!(resp.usage.prompt_tokens > 0, "prompt_tokens should be > 0");
    assert!(resp.usage.total_tokens > 0, "total_tokens should be > 0");
}

/// Assert that a streaming chat completion response is well-formed.
///
/// `chunks` should be the parsed SSE chunks (excluding `[DONE]`).
pub fn assert_valid_stream(chunks: &[ChatCompletionStreamResponse]) {
    assert!(!chunks.is_empty(), "stream should have at least one chunk");

    // All chunks should have the same id and model
    let id = &chunks[0].id;
    let model = &chunks[0].model;
    for chunk in chunks {
        assert_eq!(&chunk.id, id, "all chunks should have the same id");
        assert_eq!(&chunk.model, model, "all chunks should have the same model");
        assert_eq!(chunk.object, "chat.completion.chunk");
    }

    // Last chunk should have finish_reason
    let last = chunks.last().unwrap();
    assert!(
        !last.choices.is_empty(),
        "last chunk should have at least one choice"
    );
    let last_choice = &last.choices[0];
    assert!(
        last_choice.finish_reason.is_some(),
        "last chunk should have finish_reason, got: {:?}",
        last_choice
    );
}

/// Collect all text content from stream chunks into a single string.
pub fn collect_stream_text(chunks: &[ChatCompletionStreamResponse]) -> String {
    let mut text = String::new();
    for chunk in chunks {
        for choice in &chunk.choices {
            if let Some(ref content) = choice.delta.content {
                text.push_str(content);
            }
        }
    }
    text
}

/// Strip `<token_N>` fallbacks (numeric ids printed by the API when the tokenizer
/// can't decode an out-of-vocab id) from the input text and return what's left.
fn strip_token_id_fallbacks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Look for `<token_` followed by digits then `>`.
        if bytes[i..].starts_with(b"<token_") {
            let start = i + b"<token_".len();
            let mut j = start;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > start && j < bytes.len() && bytes[j] == b'>' {
                i = j + 1;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// Assert that text output is coherent (not empty, not garbled, not just special tokens).
pub fn assert_coherent_text(text: &str, min_len: usize) {
    let trimmed = text.trim();
    assert!(
        trimmed.len() >= min_len,
        "text should be at least {min_len} chars, got {} chars: {:?}",
        trimmed.len(),
        trimmed
    );

    // Check it's not just `<unk>` or `<token_N>` tokens
    let unk_only = trimmed
        .replace("<unk>", "")
        .replace("<pad>", "")
        .trim()
        .is_empty();
    assert!(!unk_only, "text should not be only <unk>/<pad> tokens");

    // Reject `<token_N>` patterns — when the tokenizer can't decode the id,
    // the API falls back to printing the numeric token id. A full response
    // of these means inference produced out-of-vocab ids (kernel/numerical bug).
    let stripped_token_ids = strip_token_id_fallbacks(trimmed);
    assert!(
        !stripped_token_ids.trim().is_empty(),
        "text is only <token_N> fallbacks — model produced out-of-vocab ids. Text: {:?}",
        &trimmed[..trimmed.len().min(200)]
    );

    // Reject output with too few unique characters. `!!!!!!!!` and `AAAAAAAA`
    // are classic kernel-bug signatures: the model converges on a single
    // high-probability token and emits it forever. Real English output has
    // many unique characters in the first 20 chars.
    let total_chars = trimmed.chars().count();
    if total_chars >= 8 {
        let unique_chars: std::collections::HashSet<char> = trimmed.chars().collect();
        assert!(
            unique_chars.len() >= 4,
            "text has only {} unique characters across {} chars — classic kernel-bug \
             signature (repeated single token). Text: {:?}",
            unique_chars.len(),
            total_chars,
            &trimmed[..trimmed.len().min(200)]
        );
    }

    // Detect garbled output: if the majority of characters are non-ASCII,
    // CJK, or unusual Unicode, the model is likely producing garbage.
    // Real model output (even multilingual) has mostly ASCII when prompted
    // in English with English-centric test prompts.
    if total_chars >= 10 {
        let ascii_chars = trimmed.chars().filter(|c| c.is_ascii()).count();
        let ascii_ratio = ascii_chars as f64 / total_chars as f64;
        assert!(
            ascii_ratio >= 0.5,
            "text appears garbled: only {:.0}% ASCII ({} of {} chars). \
             This usually indicates a kernel or numerical bug. Text: {:?}",
            ascii_ratio * 100.0,
            ascii_chars,
            total_chars,
            &trimmed[..trimmed.len().min(200)]
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "repeated single token")]
    fn coherent_rejects_exclamation_spam() {
        assert_coherent_text("!!!!!!!!!!!!!!!!!!!!", 2);
    }

    #[test]
    #[should_panic(expected = "<token_N> fallbacks")]
    fn coherent_rejects_token_id_fallbacks() {
        assert_coherent_text("<token_0><token_0><token_0><token_0>", 2);
    }

    #[test]
    fn coherent_accepts_real_english() {
        assert_coherent_text(
            " home to which country\nFrance\nThe capital of which is",
            10,
        );
    }
}

/// Assert that a completion response is well-formed.
pub fn assert_valid_completion_response(resp: &CompletionResponse) {
    assert!(
        resp.id.starts_with("cmpl-"),
        "id should start with 'cmpl-', got: {}",
        resp.id
    );
    assert_eq!(resp.object, "text_completion");
    assert!(resp.created > 0);
    assert!(!resp.model.is_empty());
    assert!(!resp.choices.is_empty());

    for choice in &resp.choices {
        assert!(
            choice.finish_reason.is_some(),
            "finish_reason should be set"
        );
    }

    assert!(resp.usage.prompt_tokens > 0);
    assert!(resp.usage.total_tokens > 0);
}

/// Assert that tool calls are well-formed.
pub fn assert_valid_tool_calls(tool_calls: &[scratchy_serving_api::protocol::ToolCall]) {
    assert!(!tool_calls.is_empty(), "should have at least one tool call");
    for tc in tool_calls {
        assert!(!tc.id.is_empty(), "tool call id should not be empty");
        assert_eq!(
            tc.call_type, "function",
            "tool call type should be 'function'"
        );
        assert!(
            !tc.function.name.is_empty(),
            "function name should not be empty"
        );
        // Arguments should be valid JSON
        let _: serde_json::Value =
            serde_json::from_str(&tc.function.arguments).unwrap_or_else(|e| {
                panic!(
                    "tool call arguments should be valid JSON: {e}\narguments: {}",
                    tc.function.arguments
                )
            });
    }
}

/// Assert that text is parseable as JSON.
pub fn assert_json_parseable(text: &str) {
    let _: serde_json::Value = serde_json::from_str(text)
        .unwrap_or_else(|e| panic!("text should be valid JSON: {e}\ntext: {text}"));
}

// ---------------------------------------------------------------------------
// Golden reference types and logprobs comparison
// ---------------------------------------------------------------------------

/// Golden reference data generated by HF Transformers.
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct GoldenReference {
    pub model: String,
    pub max_tokens: u32,
    pub num_logprobs: u32,
    pub results: Vec<GoldenResult>,
}

/// A single prompt's golden reference output.
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct GoldenResult {
    pub prompt: String,
    /// Decoded token strings for each generated position.
    pub output_tokens: Vec<String>,
    pub output_text: String,
    /// Per-position top-N logprobs. Keys are decoded token text, values are log-probabilities.
    pub logprobs: Vec<HashMap<String, f64>>,
}

/// Output from our engine for one prompt, extracted from the completions API.
#[derive(Debug)]
pub struct EngineOutput {
    /// Decoded token strings for each generated position.
    pub output_tokens: Vec<String>,
    pub output_text: String,
    /// Per-position top-N logprobs. Keys are decoded token text, values are log-probabilities.
    pub logprobs: Vec<HashMap<String, f64>>,
}

/// Load golden references from `testdata/golden/<model_key>.json`.
pub fn load_golden_refs(model_key: &str) -> GoldenReference {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join("golden")
        .join(format!("{model_key}.json"));
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read golden ref {}: {e}", path.display()));
    serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("failed to parse golden ref {}: {e}", path.display()))
}

/// Write golden references to `testdata/golden/<model_key>.json`.
/// Used by `VLLM_UPDATE_GOLDEN` mode to regenerate from the engine.
pub fn write_golden_refs(model_key: &str, golden: &GoldenReference) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join("golden")
        .join(format!("{model_key}.json"));
    let json = serde_json::to_string_pretty(golden)
        .unwrap_or_else(|e| panic!("failed to serialize golden ref: {e}"));
    std::fs::write(&path, json)
        .unwrap_or_else(|e| panic!("failed to write golden ref {}: {e}", path.display()));
    eprintln!("Updated golden ref: {}", path.display());
}

/// Port of Python vLLM's `check_logprobs_close()`.
///
/// For each prompt, compares token-by-token using decoded token text:
/// - When tokens match → pass, continue
/// - When tokens diverge → assert each side's chosen token is in the other's top-N logprobs,
///   then break (sequences diverge from here)
///
/// Due to bf16 numerical noise accumulating across layers and positions, late-position
/// divergences (position >= `LATE_DIVERGENCE_THRESHOLD`) where the other side's token
/// is NOT in the top-N are downgraded to warnings rather than hard failures. The first
/// `MIN_EXACT_MATCH` positions must match exactly.
pub fn check_logprobs_close(golden: &GoldenResult, engine: &EngineOutput, prompt_idx: usize) {
    check_logprobs_close_with_threshold(golden, engine, prompt_idx, 10);
}

/// Variant with a configurable `late_divergence_threshold`. Use
/// lower values (e.g. 3) for quantization formats with heavier
/// numerical drift from Python vLLM's reference — BNB4 in
/// particular accumulates enough top-N-window shift within a few
/// decode steps that a strict 10-position exact-match bar isn't
/// physically achievable against `bitsandbytes.matmul_4bit` (a
/// fused-dequant-GEMM that accumulates in a different order than
/// scratchy's dequant-then-cuBLAS-GEMM, beyond the ULP-level bf16
/// noise AWQ/GPTQ see).
pub fn check_logprobs_close_with_threshold(
    golden: &GoldenResult,
    engine: &EngineOutput,
    prompt_idx: usize,
    late_divergence_threshold: usize,
) {
    // Shadow the former `const` with the runtime value to keep the
    // body below identical to the pre-existing implementation.
    #[allow(non_snake_case)]
    let LATE_DIVERGENCE_THRESHOLD: usize = late_divergence_threshold;

    let min_len = golden.output_tokens.len().min(engine.output_tokens.len());

    for idx in 0..min_len {
        let gold_tok = &golden.output_tokens[idx];
        let eng_tok = &engine.output_tokens[idx];

        if gold_tok != eng_tok {
            let gold_lp = &golden.logprobs[idx];
            let eng_lp = &engine.logprobs[idx];

            let eng_in_gold = gold_lp.contains_key(eng_tok);
            let gold_in_eng = eng_lp.contains_key(gold_tok);

            if !eng_in_gold || !gold_in_eng {
                if idx >= LATE_DIVERGENCE_THRESHOLD {
                    // Late-position divergence: bf16 noise has accumulated enough
                    // that tokens fall outside each other's top-N. Warn, don't fail.
                    eprintln!(
                        "WARN: Prompt {prompt_idx}, position {idx}: late divergence \
                         (golden={gold_tok:?}, engine={eng_tok:?}). \
                         eng_in_gold_topN={eng_in_gold}, gold_in_eng_topN={gold_in_eng}. \
                         This is expected bf16 numerical noise at position >= {LATE_DIVERGENCE_THRESHOLD}."
                    );
                    break;
                }

                // Early-position divergence where tokens aren't in each other's top-N: fail.
                assert!(
                    eng_in_gold,
                    "Prompt {prompt_idx}, position {idx}: engine token {eng_tok:?} \
                     not in golden top-N logprobs {gold_lp:?}\n\
                     Matched tokens so far: {:?}\n\
                     Golden text: {:?}\n\
                     Engine text: {:?}",
                    &golden.output_tokens[..idx],
                    golden.output_text,
                    engine.output_text,
                );
                assert!(
                    gold_in_eng,
                    "Prompt {prompt_idx}, position {idx}: golden token {gold_tok:?} \
                     not in engine top-N logprobs {eng_lp:?}\n\
                     Matched tokens so far: {:?}\n\
                     Golden text: {:?}\n\
                     Engine text: {:?}",
                    &golden.output_tokens[..idx],
                    golden.output_text,
                    engine.output_text,
                );
            }

            eprintln!(
                "WARN: Prompt {prompt_idx}, position {idx}: token mismatch \
                 (golden={gold_tok:?}, engine={eng_tok:?}), but both in each other's top-N. \
                 Sequences diverge from here."
            );
            break;
        }
    }
}

/// Extract [`EngineOutput`] from a completion response choice's logprobs.
pub fn extract_engine_output(
    choice: &scratchy_serving_api::protocol::CompletionResponseChoice,
) -> EngineOutput {
    let text = choice.text.clone();
    let lp = choice
        .logprobs
        .as_ref()
        .expect("logprobs should be present in response");

    let output_tokens: Vec<String> = lp.tokens.clone();
    let logprobs: Vec<HashMap<String, f64>> = lp
        .top_logprobs
        .iter()
        .map(|opt| {
            opt.as_ref()
                .expect("top_logprobs entry should not be None")
                .clone()
        })
        .collect();

    EngineOutput {
        output_tokens,
        output_text: text,
        logprobs,
    }
}
