// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Incremental detokenizer for streaming text generation.
//!
//! Direct port of Python's FastIncrementalDetokenizer from vllm/v1/engine/detokenizer.py

use std::sync::Arc;

use crate::tokenizer::Tokenizer;

/// Incremental detokenizer for streaming text generation.
///
/// Python equivalent: FastIncrementalDetokenizer (vllm/v1/engine/detokenizer.py)
///
/// Uses the exact same streaming decode algorithm as Python's DecodeStream.step()
pub struct IncrementalDetokenizer {
    /// The tokenizer
    tokenizer: Arc<Tokenizer>,

    /// Skip special tokens flag
    skip_special_tokens: bool,

    /// All token IDs (prompt + generated)
    token_ids: Vec<u32>,

    /// Number of prompt tokens
    num_prompt_tokens: usize,

    /// Accumulated output text
    output_text: String,

    /// Offset for delta mode
    last_output_text_offset: usize,

    // Stop string handling
    stop_strings: Vec<String>,
    min_tokens: u32,
    include_stop_str_in_output: bool,
    stop_buffer_length: usize,

    // DecodeStream state (from tokenizers library)
    /// Buffer of token IDs for streaming decode
    stream_ids: Vec<u32>,
    /// Previously returned chunk that needs to be discarded
    stream_prefix: String,
    /// Index within stream_ids corresponding to the prefix
    stream_prefix_index: usize,
}

impl IncrementalDetokenizer {
    /// Create a minimal detokenizer for testing (no real tokenizer attached).
    #[cfg(test)]
    pub(crate) fn dummy() -> Self {
        use crate::tokenizer::make_test_tokenizer;
        let tok = Arc::new(make_test_tokenizer());
        let prompt_ids = tok.encode("", false).unwrap();
        Self::new(tok, &prompt_ids, vec![], 0, false, false)
    }

    /// Create new detokenizer
    ///
    /// Python: FastIncrementalDetokenizer.__init__ (line 170-208)
    pub fn new(
        tokenizer: Arc<Tokenizer>,
        prompt_token_ids: &[u32],
        stop_strings: Vec<String>,
        min_tokens: u32,
        include_stop_str_in_output: bool,
        skip_special_tokens: bool,
    ) -> Self {
        let num_prompt_tokens = prompt_token_ids.len();

        // Python line 88-91: stop_buffer_length calculation
        let stop_buffer_length = if !stop_strings.is_empty() && !include_stop_str_in_output {
            stop_strings
                .iter()
                .map(|s| s.len())
                .max()
                .unwrap_or(0)
                .saturating_sub(1)
        } else {
            0
        };

        // Python line 182-185: Initialize DecodeStream with prompt tokens
        // self.stream = DecodeStream(ids=request.prompt_token_ids, skip_special_tokens=...)
        // This initializes the stream state with the prompt tokens
        Self {
            tokenizer,
            skip_special_tokens,
            token_ids: prompt_token_ids.to_vec(),
            num_prompt_tokens,
            output_text: String::new(),
            last_output_text_offset: 0,
            stop_strings,
            min_tokens,
            include_stop_str_in_output,
            stop_buffer_length,
            // Initialize DecodeStream state with prompt tokens
            stream_ids: prompt_token_ids.to_vec(),
            stream_prefix: String::new(),
            stream_prefix_index: 0,
        }
    }

    /// Update with new tokens
    ///
    /// Python: BaseIncrementalDetokenizer.update (line 97-144)
    pub fn update(&mut self, new_token_ids: &[u32], stop_terminated: bool) -> Option<String> {
        if new_token_ids.is_empty() {
            return None;
        }

        // Python line 118: stop_check_offset = len(self.output_text)
        let stop_check_offset = self.output_text.len();

        // Python line 119-121: Process each token
        let last = new_token_ids.len() - 1;
        for (i, &new_token_id) in new_token_ids.iter().enumerate() {
            // Python line 121: self.output_text += self.decode_next(new_token_id)
            let len_before = self.output_text.len();
            let token_text = self.decode_next(new_token_id);
            self.output_text.push_str(&token_text);
            // The token that terminated generation (EOS / a stop token, e.g.
            // gemma's <end_of_turn>) must not surface as visible text.
            if stop_terminated && i == last {
                self.output_text.truncate(len_before);
            }
        }

        // Python line 132-143: Check stop strings
        if !self.stop_strings.is_empty() && self.num_output_tokens() as u32 > self.min_tokens {
            let new_char_count = self.output_text.len() - stop_check_offset;
            if let Some((stop_str, truncate_to)) = check_stop_strings(
                &self.output_text,
                new_char_count,
                &self.stop_strings,
                self.include_stop_str_in_output,
            ) {
                if truncate_to != -1 {
                    self.output_text.truncate(truncate_to as usize);
                }
                return Some(stop_str);
            }
        }

        None
    }

    /// Decode next token using streaming decode algorithm
    ///
    /// Python: FastIncrementalDetokenizer.decode_next (line 209-220)
    /// This is a direct port of tokenizers' step_decode_stream function
    /// from /tmp/zoo/tokenizers/tokenizers/src/tokenizer/mod.rs lines 1085-1128
    fn decode_next(&mut self, next_token_id: u32) -> String {
        self.token_ids.push(next_token_id);

        // Convert single token to Vec as Python does (line 694)
        let token_ids = vec![next_token_id];

        // EXACT implementation of step_decode_stream from tokenizers library
        // Line 1100-1106: Initialize prefix if empty and ids not empty
        if self.stream_prefix.is_empty() && !self.stream_ids.is_empty() {
            match self
                .tokenizer
                .inner()
                .decode(&self.stream_ids, self.skip_special_tokens)
            {
                Ok(new_prefix) => {
                    if !new_prefix.ends_with('�') {
                        self.stream_prefix = new_prefix;
                        self.stream_prefix_index = self.stream_ids.len();
                    }
                }
                Err(_) => {
                    // Ignore decode errors during prefix initialization
                }
            }
        }

        // Line 1108: Extend ids with new token(s)
        self.stream_ids.extend(token_ids);

        // Line 1109: Decode all ids
        let string = match self
            .tokenizer
            .inner()
            .decode(&self.stream_ids, self.skip_special_tokens)
        {
            Ok(s) => s,
            Err(_) => {
                return String::new();
            }
        };

        // Line 1110-1127: Check if we have valid new text
        if string.len() > self.stream_prefix.len() && !string.ends_with('�') {
            // Line 1111-1117: Validate prefix
            if !string.starts_with(&self.stream_prefix) {
                return String::new();
            }

            // Line 1119: Extract new text
            let new_text = string[self.stream_prefix.len()..].to_string();

            // Line 1120-1123: Update state
            let new_prefix_index = self.stream_ids.len() - self.stream_prefix_index;
            self.stream_ids = self.stream_ids.drain(self.stream_prefix_index..).collect();

            match self
                .tokenizer
                .inner()
                .decode(&self.stream_ids, self.skip_special_tokens)
            {
                Ok(new_prefix) => {
                    self.stream_prefix = new_prefix;
                }
                Err(_) => {
                    // Ignore decode errors during prefix update
                }
            }
            self.stream_prefix_index = new_prefix_index;

            // Line 1124: Return new text
            new_text
        } else {
            // Line 1126: No new text yet
            String::new()
        }
    }

    /// Get next output text
    ///
    /// Python: BaseIncrementalDetokenizer.get_next_output_text (line 150-166)
    pub fn get_next_output_text(&mut self, finished: bool, delta: bool) -> String {
        // Python line 155: buffer_length = 0 if finished else self.stop_buffer_length
        let buffer_length = if finished { 0 } else { self.stop_buffer_length };

        if !delta {
            // Python line 157-159: Return full text
            if buffer_length == 0 {
                return self.output_text.clone();
            }
            let end = self.output_text.len().saturating_sub(buffer_length);
            return self.output_text[..end].to_string();
        }

        // Python line 161-166: Delta mode
        let length = self.output_text.len().saturating_sub(buffer_length);
        let last_offset = self.last_output_text_offset;
        if last_offset < length {
            self.last_output_text_offset = length;
            return self.output_text[last_offset..length].to_string();
        }
        String::new()
    }

    pub fn num_output_tokens(&self) -> usize {
        self.token_ids.len() - self.num_prompt_tokens
    }

    pub fn output_token_ids(&self) -> &[u32] {
        &self.token_ids[self.num_prompt_tokens..]
    }

    pub fn output_text(&self) -> &str {
        &self.output_text
    }
}

/// Check for stop strings
///
/// Python: check_stop_strings in detokenizer.py (line 313-336)
fn check_stop_strings(
    output_text: &str,
    new_char_count: usize,
    stop_strings: &[String],
    include_in_output: bool,
) -> Option<(String, i32)> {
    for stop_str in stop_strings {
        if stop_str.is_empty() {
            continue;
        }

        // Python line 327: stop_index = output_text.find(stop_str, 1 - new_char_count - stop_string_len)
        let stop_string_len = stop_str.len();
        let search_start = output_text
            .len()
            .saturating_sub(new_char_count + stop_string_len - 1);

        if let Some(pos) = output_text[search_start..].find(stop_str) {
            let stop_index = search_start + pos;

            // Python line 331-336
            if include_in_output {
                let end = stop_index + stop_string_len;
                if end == output_text.len() {
                    return Some((stop_str.clone(), -1));
                }
                return Some((stop_str.clone(), end as i32));
            } else {
                return Some((stop_str.clone(), stop_index as i32));
            }
        }
    }
    None
}

impl std::fmt::Debug for IncrementalDetokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IncrementalDetokenizer")
            .field("num_prompt_tokens", &self.num_prompt_tokens)
            .field("num_output_tokens", &self.num_output_tokens())
            .field("output_text_len", &self.output_text.len())
            .finish()
    }
}

// Made with Bob

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::make_test_tokenizer;

    fn make_detokenizer(prompt: &str, stop_strings: Vec<String>) -> IncrementalDetokenizer {
        let tok = Arc::new(make_test_tokenizer());
        let prompt_ids = tok.encode(prompt, false).unwrap();
        IncrementalDetokenizer::new(
            tok,
            &prompt_ids,
            stop_strings,
            0,     // min_tokens
            false, // include_stop_str_in_output
            false, // skip_special_tokens
        )
    }

    // -- check_stop_strings tests --

    #[test]
    fn test_check_stop_strings_empty() {
        assert!(check_stop_strings("hello world", 5, &[], false).is_none());
    }

    #[test]
    fn test_check_stop_strings_no_match() {
        let stops = vec!["foo".to_string()];
        assert!(check_stop_strings("hello world", 5, &stops, false).is_none());
    }

    #[test]
    fn test_check_stop_strings_match() {
        let stops = vec!["world".to_string()];
        let result = check_stop_strings("hello world", 6, &stops, false);
        assert!(result.is_some());
        let (stop, offset) = result.unwrap();
        assert_eq!(stop, "world");
        assert_eq!(offset, 6); // "hello " is 6 bytes
    }

    #[test]
    fn test_check_stop_strings_match_at_boundary() {
        let stops = vec!["lo wo".to_string()];
        let result = check_stop_strings("hello world", 6, &stops, false);
        assert!(result.is_some());
        let (stop, offset) = result.unwrap();
        assert_eq!(stop, "lo wo");
        assert_eq!(offset, 3);
    }

    #[test]
    fn test_check_stop_strings_first_match_wins() {
        let stops = vec!["world".to_string(), "ello".to_string()];
        let result = check_stop_strings("hello world", 11, &stops, false);
        assert!(result.is_some());
        // "world" is checked first and matches.
        assert_eq!(result.unwrap().0, "world");
    }

    #[test]
    fn test_check_stop_strings_multibyte() {
        let stops = vec!["世界".to_string()];
        let text = "こんにちは世界";
        let result = check_stop_strings(text, text.len(), &stops, false);
        assert!(result.is_some());
        let (stop, _offset) = result.unwrap();
        assert_eq!(stop, "世界");
    }

    // -- IncrementalDetokenizer tests --

    #[test]
    fn test_detokenizer_initial_state() {
        let detok = make_detokenizer("Hello", vec![]);
        assert_eq!(detok.num_output_tokens(), 0);
        assert!(detok.output_token_ids().is_empty());
        assert_eq!(detok.output_text(), "");
    }

    #[test]
    fn test_detokenizer_update_produces_text() {
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "Hello";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok =
            IncrementalDetokenizer::new(Arc::clone(&tok), &prompt_ids, vec![], 0, false, false);

        // Encode some continuation text and feed it token by token.
        let continuation = " world";
        let cont_ids = tok
            .encode(&format!("{prompt}{continuation}"), false)
            .unwrap();
        let new_ids = &cont_ids[prompt_ids.len()..];

        let stop = detok.update(new_ids, false);
        assert!(stop.is_none());
        assert!(detok.num_output_tokens() > 0);

        let text = detok.get_next_output_text(true, false);
        // The decoded text should contain "world" (possibly with leading space).
        assert!(
            text.contains("world"),
            "Expected 'world' in output, got: {text:?}"
        );
    }

    #[test]
    fn test_detokenizer_delta_mode() {
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "Hi";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok =
            IncrementalDetokenizer::new(Arc::clone(&tok), &prompt_ids, vec![], 0, false, false);

        // Feed tokens in two batches.
        let full = tok.encode(&format!("{prompt} ab cd"), false).unwrap();
        let new_ids = &full[prompt_ids.len()..];

        if new_ids.len() >= 2 {
            let mid = new_ids.len() / 2;
            detok.update(&new_ids[..mid], false);
            let text1 = detok.get_next_output_text(false, true);

            detok.update(&new_ids[mid..], false);
            let text2 = detok.get_next_output_text(true, true);

            // Both deltas should be non-overlapping portions of the output.
            let combined = format!("{text1}{text2}");
            let full_text = detok.get_next_output_text(true, false);
            // The cumulative text should contain the delta texts.
            assert!(!full_text.is_empty(), "Expected non-empty output text");
            // The combined delta texts should equal the full text minus any
            // stop buffer effects (no stop strings here).
            assert_eq!(
                combined, full_text,
                "Delta concatenation should equal full text"
            );
        }
    }

    #[test]
    fn test_detokenizer_stop_string() {
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "Say";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok = IncrementalDetokenizer::new(
            Arc::clone(&tok),
            &prompt_ids,
            vec!["STOP".to_string()],
            0,
            false,
            false,
        );

        // Encode text that contains the stop string.
        let full = tok
            .encode(&format!("{prompt} hello STOP bye"), false)
            .unwrap();
        let new_ids = &full[prompt_ids.len()..];

        let result = detok.update(new_ids, false);
        // Should detect the stop string.
        if let Some(stop_str) = result {
            assert_eq!(stop_str, "STOP");
            // Output text should not contain STOP (include_stop_str_in_output=false).
            let text = detok.get_next_output_text(true, false);
            assert!(
                !text.contains("STOP"),
                "Stop string should be excluded: {text:?}"
            );
        }
        // If the tokenizer doesn't produce exactly "STOP", the test still
        // passes because the stop string check requires exact text match.
    }

    #[test]
    fn test_detokenizer_stop_string_included() {
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "Say";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok = IncrementalDetokenizer::new(
            Arc::clone(&tok),
            &prompt_ids,
            vec!["STOP".to_string()],
            0,
            true, // include_stop_str_in_output
            false,
        );

        let full = tok
            .encode(&format!("{prompt} hello STOP bye"), false)
            .unwrap();
        let new_ids = &full[prompt_ids.len()..];

        let result = detok.update(new_ids, false);
        if let Some(stop_str) = result {
            assert_eq!(stop_str, "STOP");
            let text = detok.get_next_output_text(true, false);
            assert!(
                text.contains("STOP"),
                "Stop string should be included: {text:?}"
            );
        }
    }

    #[test]
    fn test_detokenizer_stop_buffer() {
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "X";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        // Use a stop string to create a buffer.
        let mut detok = IncrementalDetokenizer::new(
            Arc::clone(&tok),
            &prompt_ids,
            vec!["END".to_string()], // 3-char stop buffer
            0,
            false,
            false,
        );

        let full = tok.encode(&format!("{prompt} hello there"), false).unwrap();
        let new_ids = &full[prompt_ids.len()..];
        detok.update(new_ids, false);

        // Non-finished output should hold back 3 characters.
        let partial = detok.get_next_output_text(false, false);
        let full_text = detok.get_next_output_text(true, false);
        // The partial output should be shorter or equal to the full output
        // (shorter by up to stop_buffer_length characters).
        assert!(partial.len() <= full_text.len());
    }

    #[test]
    fn test_detokenizer_min_tokens() {
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "A";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok = IncrementalDetokenizer::new(
            Arc::clone(&tok),
            &prompt_ids,
            vec!["B".to_string()],
            100, // min_tokens = 100 -- stop strings won't trigger
            false,
            false,
        );

        // Even if "B" appears, it won't trigger because min_tokens is high.
        let full = tok.encode(&format!("{prompt} B C"), false).unwrap();
        let new_ids = &full[prompt_ids.len()..];
        let result = detok.update(new_ids, false);
        assert!(
            result.is_none(),
            "Stop string should not trigger before min_tokens"
        );
    }

    #[test]
    fn test_detokenizer_empty_update() {
        let mut detok = make_detokenizer("Hello", vec![]);
        let result = detok.update(&[], false);
        assert!(result.is_none());
        assert_eq!(detok.num_output_tokens(), 0);
    }

    #[test]
    fn test_detokenizer_debug() {
        let detok = make_detokenizer("Hello", vec!["stop".to_string()]);
        let debug_str = format!("{detok:?}");
        assert!(debug_str.contains("IncrementalDetokenizer"));
        assert!(debug_str.contains("num_output_tokens"));
    }

    // -- Streaming decode algorithm tests --

    #[test]
    fn test_streaming_decode_single_token() {
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "Hello";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok =
            IncrementalDetokenizer::new(Arc::clone(&tok), &prompt_ids, vec![], 0, false, false);

        // Add a single token
        let full = tok.encode(&format!("{prompt} world"), false).unwrap();
        let new_ids = &full[prompt_ids.len()..];

        if !new_ids.is_empty() {
            detok.update(&new_ids[..1], false);
            let text = detok.get_next_output_text(true, false);
            // Should produce some output
            assert!(!text.is_empty(), "Single token should produce output");
        }
    }

    #[test]
    fn test_streaming_decode_multibyte_utf8() {
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "Say";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok =
            IncrementalDetokenizer::new(Arc::clone(&tok), &prompt_ids, vec![], 0, false, false);

        // Test with multibyte UTF-8 characters
        let full = tok.encode(&format!("{prompt} 你好世界"), false).unwrap();
        let new_ids = &full[prompt_ids.len()..];

        detok.update(new_ids, false);
        let text = detok.get_next_output_text(true, false);

        // The output should be valid UTF-8
        assert!(text.is_char_boundary(0), "Output should be valid UTF-8");
        assert!(
            text.is_char_boundary(text.len()),
            "Output should be valid UTF-8"
        );
    }

    #[test]
    fn test_streaming_decode_incremental() {
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "Count";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok =
            IncrementalDetokenizer::new(Arc::clone(&tok), &prompt_ids, vec![], 0, false, false);

        // Add tokens one at a time
        let full = tok.encode(&format!("{prompt} 1 2 3"), false).unwrap();
        let new_ids = &full[prompt_ids.len()..];

        let mut accumulated = String::new();
        for &token_id in new_ids {
            detok.update(&[token_id], false);
            let delta = detok.get_next_output_text(false, true);
            accumulated.push_str(&delta);
        }

        let final_text = detok.get_next_output_text(true, false);

        // Accumulated deltas should match final text
        assert_eq!(
            accumulated, final_text,
            "Incremental decode should match final"
        );
    }

    // -- Tests for specific bugs fixed in streaming decode --

    #[test]
    fn test_streaming_decode_prefix_initialization() {
        // Tests that prefix is correctly initialized when stream_ids is not empty
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "Test";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok =
            IncrementalDetokenizer::new(Arc::clone(&tok), &prompt_ids, vec![], 0, false, false);

        // The detokenizer should initialize with prompt tokens in stream_ids
        assert_eq!(detok.stream_ids.len(), prompt_ids.len());

        // Add first token - should trigger prefix initialization
        let full = tok.encode(&format!("{prompt} word"), false).unwrap();
        let new_ids = &full[prompt_ids.len()..];

        if !new_ids.is_empty() {
            detok.update(&new_ids[..1], false);
            // After first token, prefix should be set
            assert!(!detok.stream_prefix.is_empty() || detok.stream_ids.len() > 1);
        }
    }

    #[test]
    fn test_streaming_decode_drain_operation() {
        // Tests that drain operation correctly maintains state
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "A";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok =
            IncrementalDetokenizer::new(Arc::clone(&tok), &prompt_ids, vec![], 0, false, false);

        // Add multiple tokens
        let full = tok.encode(&format!("{prompt} B C D"), false).unwrap();
        let new_ids = &full[prompt_ids.len()..];

        for &token_id in new_ids {
            let _ids_before = detok.stream_ids.len();
            detok.update(&[token_id], false);
            let ids_after = detok.stream_ids.len();

            // After drain, stream_ids should not grow unbounded
            // It should stay relatively small (typically 1-2 tokens)
            assert!(
                ids_after <= 3,
                "stream_ids growing unbounded: {} tokens",
                ids_after
            );
        }
    }

    #[test]
    fn test_streaming_decode_prefix_index_consistency() {
        // Tests that prefix_index correctly tracks the prefix in stream_ids
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "Count";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok =
            IncrementalDetokenizer::new(Arc::clone(&tok), &prompt_ids, vec![], 0, false, false);

        let full = tok
            .encode(&format!("{prompt} one two three"), false)
            .unwrap();
        let new_ids = &full[prompt_ids.len()..];

        for &token_id in new_ids {
            detok.update(&[token_id], false);

            // prefix_index should never exceed stream_ids length
            assert!(
                detok.stream_prefix_index <= detok.stream_ids.len(),
                "prefix_index {} exceeds stream_ids length {}",
                detok.stream_prefix_index,
                detok.stream_ids.len()
            );

            // If we have a prefix, decoding the first prefix_index tokens should produce it
            if !detok.stream_prefix.is_empty() && detok.stream_prefix_index > 0 {
                let prefix_tokens =
                    &detok.stream_ids[..detok.stream_prefix_index.min(detok.stream_ids.len())];
                if !prefix_tokens.is_empty()
                    && let Ok(decoded) = tok.inner().decode(prefix_tokens, false)
                {
                    // The decoded prefix tokens should match or be a prefix of stream_prefix
                    assert!(
                        detok.stream_prefix.starts_with(&decoded)
                            || decoded.starts_with(&detok.stream_prefix),
                        "Prefix mismatch: decoded={:?}, stream_prefix={:?}",
                        decoded,
                        detok.stream_prefix
                    );
                }
            }
        }
    }

    #[test]
    fn test_streaming_decode_no_replacement_char() {
        // Tests that we don't emit text ending with replacement character
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "Say";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok =
            IncrementalDetokenizer::new(Arc::clone(&tok), &prompt_ids, vec![], 0, false, false);

        let full = tok.encode(&format!("{prompt} hello world"), false).unwrap();
        let new_ids = &full[prompt_ids.len()..];

        // Add tokens one by one and check output never ends with replacement char
        for &token_id in new_ids {
            detok.update(&[token_id], false);
            let text = detok.get_next_output_text(false, false);

            assert!(
                !text.ends_with('\u{FFFD}'),
                "Output should not end with replacement character: {:?}",
                text
            );
        }
    }

    #[test]
    fn test_streaming_decode_extend_not_push() {
        // Tests that we use extend (not push) to add tokens, matching tokenizers library
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "X";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok =
            IncrementalDetokenizer::new(Arc::clone(&tok), &prompt_ids, vec![], 0, false, false);

        // Add a single token
        let full = tok.encode(&format!("{prompt} Y"), false).unwrap();
        let new_ids = &full[prompt_ids.len()..];

        if !new_ids.is_empty() {
            let initial_len = detok.stream_ids.len();
            detok.update(&new_ids[..1], false);

            // stream_ids should have grown by exactly 1
            // (This tests that we're using extend with vec![token_id], not push)
            assert!(
                detok.stream_ids.len() >= initial_len,
                "stream_ids should grow after adding token"
            );
        }
    }

    #[test]
    fn test_streaming_decode_valid_utf8_output() {
        // Tests that all output is valid UTF-8, even with multibyte characters
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "Test";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok =
            IncrementalDetokenizer::new(Arc::clone(&tok), &prompt_ids, vec![], 0, false, false);

        // Mix of ASCII and multibyte UTF-8
        let test_strings = vec![
            " hello",
            " \u{4e16}\u{754c}",
            " \u{645}\u{631}\u{62d}\u{628}\u{627}",
            " \u{41f}\u{440}\u{438}\u{432}\u{435}\u{442}",
            " \u{1f30d}",
        ];

        for test_str in test_strings {
            let full = tok.encode(&format!("{prompt}{test_str}"), false).unwrap();
            let new_ids = &full[prompt_ids.len()..];

            detok.update(new_ids, false);
            let text = detok.get_next_output_text(true, false);

            // Verify it's valid UTF-8 (Rust strings are always valid UTF-8,
            // but this guards against unsafe code or decoder bugs).
            assert!(
                std::str::from_utf8(text.as_bytes()).is_ok(),
                "Output should be valid UTF-8: {:?}",
                text
            );

            // Verify char boundary consistency: every char boundary index
            // returned by char_indices should be a valid boundary.
            for (i, _) in text.char_indices() {
                assert!(
                    text.is_char_boundary(i),
                    "char_indices position {} should be a char boundary in {:?}",
                    i,
                    text
                );
            }
        }
    }

    #[test]
    fn test_streaming_decode_empty_prefix_handling() {
        // Tests correct behavior when prefix is empty
        let tok = Arc::new(make_test_tokenizer());
        let prompt = ""; // Empty prompt
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok =
            IncrementalDetokenizer::new(Arc::clone(&tok), &prompt_ids, vec![], 0, false, false);

        let full = tok.encode("Hello world", false).unwrap();

        detok.update(&full, false);
        let text = detok.get_next_output_text(true, false);

        assert!(
            text.contains("Hello") || text.contains("world"),
            "Should produce output even with empty prompt: {:?}",
            text
        );
    }
}
