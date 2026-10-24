// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Tokenizer wrapper around the HuggingFace `tokenizers` crate.
//!
//! Provides encoding (text → token IDs) and decoding (token IDs → text)
//! for use in input processing and incremental detokenization.
//!
//! Port of: `vllm/tokenizers/` (subset)

use std::collections::HashSet;
use std::path::Path;

use scratchy_core_model::baked_tokenizer::{self, Baked, BakedTokenizer};
use tokenizers::tokenizer::PostProcessor as _;
use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

use crate::error::{ServeError, ServeResult};

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// A wrapper around the HuggingFace `tokenizers` tokenizer that provides
/// the core encode/decode interface needed by the serving layer.
pub struct Tokenizer {
    inner: Inner,
    /// Cached set of special token IDs for fast lookup.
    special_ids: HashSet<u32>,
    /// The EOS token ID, if present.
    eos_token_id: Option<u32>,
}

/// The `tokenizers` tokenizer a [`Tokenizer`] wraps: parsed from a `tokenizer.json`, or compiled
/// into the binary (`scratchy_core_model::baked_tokenizer`). Both are `tokenizers::TokenizerImpl`s
/// differing only in their model, so every method is the same call on either.
enum Inner {
    File(Box<HfTokenizer>),
    Baked(Box<Baked>),
}

/// `$body`, with `$t` bound to whichever tokenizer `$inner` holds.
macro_rules! with_inner {
    ($inner:expr, $t:ident => $body:expr) => {
        match $inner {
            Inner::File($t) => $body,
            Inner::Baked($t) => $body,
        }
    };
}

impl Tokenizer {
    /// Load the tokenizer of a `tokenizer.json` file: the one compiled into this binary when the
    /// build baked it (a Hub-cache file, keyed by its blob digest) — no parse, where gemma-4's
    /// file takes ~300 ms — else the file's.
    pub fn from_file(path: impl AsRef<Path>) -> ServeResult<Self> {
        let path = path.as_ref();
        if let Some(baked) =
            baked_tokenizer::cached_digest(path).and_then(|digest| BakedTokenizer::find(&digest))
        {
            match baked.load() {
                Ok(tokenizer) => {
                    tracing::info!("Tokenizer compiled in (tokenizer.json {})", baked.digest);
                    return Ok(Self::new(Inner::Baked(Box::new(tokenizer))));
                }
                Err(e) => tracing::warn!("{e}; loading {} instead", path.display()),
            }
        }
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| ServeError::Engine(format!("Failed to load tokenizer: {e}")))?;
        Ok(Self::from_hf_tokenizer(inner))
    }

    /// Create a `Tokenizer` from a raw HuggingFace `Tokenizer` instance.
    pub fn from_hf_tokenizer(inner: HfTokenizer) -> Self {
        Self::new(Inner::File(Box::new(inner)))
    }

    /// Every [`Tokenizer`], whatever its source.
    ///
    /// ⛔⛔⛔ TRUNCATION IS DISABLED HERE, AND IT MUST STAY DISABLED. A `tokenizer.json` may carry its
    /// own `truncation` stanza, and the `tokenizers` crate then applies it to every `encode` with no
    /// warning and no error. `RedHatAI/granite-3.1-8b-instruct-FP8-dynamic` ships
    /// `{"direction":"Right","max_length":2048,...}` while its `model_max_length` is `i64::MAX` and its
    /// `max_position_embeddings` is 131072 — so every prompt over 2048 tokens was silently CUT ON THE
    /// RIGHT, losing the question at the end of it.
    ///
    /// MEASURED 2026-08-12: probes built for ~2400 and ~3300 tokens both reported `prompt_tokens: 2048`
    /// and answered by continuing the passage, because the question had been cut off. It cost a whole
    /// investigation — it presented as a "2048 context wall", and the scheduler budget was blamed
    /// (exonerated: the trace prints `budget=8192` and the clamp stays at 2048).
    ///
    /// LENGTH IS THE SERVING LAYER'S DECISION, NOT THE TOKENIZER'S: `max_model_len` and the scheduler
    /// budget already bound it, and they can REFUSE. A silent right-truncation cannot — it produces a
    /// fluent answer to a question the user did not ask.
    fn new(mut inner: Inner) -> Self {
        with_inner!(&mut inner, t => {
            t.with_truncation(None)
                .expect("disabling truncation cannot fail");
        });
        let special_ids = Self::collect_special_ids(&inner);
        let eos_token_id = with_inner!(&inner, t => t
            .token_to_id("</s>")
            .or_else(|| t.token_to_id("<|endoftext|>"))
            .or_else(|| t.token_to_id("<|end_of_text|>")));
        Self {
            inner,
            special_ids,
            eos_token_id,
        }
    }

    /// Register additional special tokens that the underlying
    /// `tokenizer.json::added_tokens` list missed.
    ///
    /// HuggingFace tokenizer files split special-token configuration
    /// across two artifacts: `tokenizer.json::added_tokens` (the
    /// authoritative list the Rust `tokenizers` crate honors) and
    /// `tokenizer_config.json::additional_special_tokens` (a string
    /// list `transformers` reads at load time and merges in). Some
    /// checkpoints — notably Qwen2-VL — register
    /// `<|image_pad|>` / `<|video_pad|>` / `<|vision_pad|>` only in
    /// the latter, so the chat template's `<|image_pad|>` emission
    /// subword-splits inside the Rust loader and
    /// `expand_image_placeholders` finds zero occurrences.
    ///
    /// Mirrors `transformers`' merge: tokens already present in the
    /// vocabulary keep their existing ID and just get flagged
    /// `special` so the pre-tokenizer respects them; tokens absent
    /// from the vocabulary are appended (uncommon — flagged in a
    /// debug log).
    pub fn register_additional_special_tokens(&mut self, contents: &[String]) {
        let added: Vec<AddedToken> = contents
            .iter()
            .filter(|c| {
                // Skip entries the tokenizer already treats as
                // special — re-adding via `add_special_tokens` would
                // append a fresh ID and break vocabulary alignment.
                with_inner!(&self.inner, t => t.token_to_id(c).is_none_or(|id| {
                    let dec = t.get_added_tokens_decoder();
                    !dec.get(&id).map(|t| t.special).unwrap_or(false)
                }))
            })
            .map(|c| AddedToken::from(c.clone(), true))
            .collect();
        if added.is_empty() {
            return;
        }
        let names: Vec<&str> = added.iter().map(|t| t.content.as_str()).collect();
        tracing::info!(
            "Registering {} additional special tokens missing from tokenizer.json: {:?}",
            added.len(),
            names
        );
        let added = with_inner!(&mut self.inner, t => t.add_special_tokens(added).map(|_| ()));
        if let Err(err) = added {
            tracing::warn!("failed to register additional special tokens: {err}");
            return;
        }
        self.special_ids = Self::collect_special_ids(&self.inner);
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> ServeResult<Vec<u32>> {
        let encoding = with_inner!(&self.inner, t => t.encode(text, add_special_tokens))
            .map_err(|e| ServeError::Engine(format!("Tokenization failed: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encode text into token IDs, and for each byte offset of `marks`, the tokens before the first
    /// one that ends past it: the prompt length that reaches that point of the text and no further
    /// (a special token a post-processor appends, offset `(0, 0)`, comes after every mark).
    pub fn encode_through(
        &self,
        text: &str,
        add_special_tokens: bool,
        marks: &[usize],
    ) -> ServeResult<(Vec<u32>, Vec<u32>)> {
        let encoding = with_inner!(&self.inner, t => t.encode(text, add_special_tokens))
            .map_err(|e| ServeError::Engine(format!("Tokenization failed: {e}")))?;
        let offsets = encoding.get_offsets();
        let through = (marks.iter())
            .map(|&mark| {
                let past = offsets.iter().position(|&(_, end)| end > mark);
                past.unwrap_or(offsets.len()) as u32
            })
            .collect();
        Ok((encoding.get_ids().to_vec(), through))
    }

    /// A chat prompt the template rendered, encoded as [`Self::encode_through`] does, with the
    /// post-processor's special tokens: the one rule for every path that encodes one, so
    /// `/tokenize` counts the ids the prefill reads.
    pub fn encode_chat(&self, text: &str, marks: &[usize]) -> ServeResult<(Vec<u32>, Vec<u32>)> {
        self.encode_through(text, true, marks)
    }

    /// Decode token IDs back into text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> ServeResult<String> {
        with_inner!(&self.inner, t => t.decode(ids, skip_special_tokens))
            .map_err(|e| ServeError::Engine(format!("Detokenization failed: {e}")))
    }

    /// Get the string representation of a single token ID.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        with_inner!(&self.inner, t => t.id_to_token(id))
    }

    /// Get the token ID for a string token.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        with_inner!(&self.inner, t => t.token_to_id(token))
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        with_inner!(&self.inner, t => t.get_vocab_size(true))
    }

    /// How many special tokens encoding a single sequence (`is_pair` false) or a pair adds.
    pub fn num_special_tokens_to_add(&self, is_pair: bool) -> usize {
        with_inner!(&self.inner, t => t.get_post_processor().map_or(0, |p| p.added_tokens(is_pair)))
    }

    /// The special token IDs.
    pub fn special_token_ids(&self) -> &HashSet<u32> {
        &self.special_ids
    }

    /// Get the EOS token ID, if known.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Check if a token ID is a special token.
    pub fn is_special_token(&self, id: u32) -> bool {
        self.special_ids.contains(&id)
    }

    /// Get the full vocabulary as a token-to-id mapping.
    pub fn get_vocab(&self) -> std::collections::HashMap<String, u32> {
        with_inner!(&self.inner, t => t.get_vocab(true))
    }

    /// Collect all special token IDs from the tokenizer.
    fn collect_special_ids(tokenizer: &Inner) -> HashSet<u32> {
        let mut ids = HashSet::new();
        let added = with_inner!(tokenizer, t => t.get_added_tokens_decoder());
        for (&id, token) in &added {
            if token.special {
                ids.insert(id);
            }
        }
        ids
    }
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab_size())
            .field("eos_token_id", &self.eos_token_id)
            .field("num_special_tokens", &self.special_ids.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Create a simple BPE tokenizer for testing.
///
/// Uses a byte-level pre-tokenizer (GPT-2 style) with the full 256 byte-token
/// vocabulary. No merges — each byte is its own token. Good enough for testing
/// encode/decode roundtrips.
#[cfg(test)]
pub(crate) fn make_test_tokenizer() -> Tokenizer {
    Tokenizer::from_hf_tokenizer(test_tokenizer_json().to_string().parse().unwrap())
}

/// [`make_test_tokenizer`] with a `<s>` (id 256) its post-processor puts in front of every
/// sequence, as Llama 3's, Mistral's and Gemma's do.
#[cfg(test)]
pub(crate) fn make_test_tokenizer_with_bos() -> Tokenizer {
    let mut json = test_tokenizer_json();
    json["added_tokens"] = serde_json::json!([{
        "id": 256, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false,
        "normalized": false, "special": true
    }]);
    let bos = serde_json::json!({"SpecialToken": {"id": "<s>", "type_id": 0}});
    let seq = |id: &str| serde_json::json!({"Sequence": {"id": id, "type_id": 0}});
    json["post_processor"] = serde_json::json!({
        "type": "TemplateProcessing",
        "single": [bos, seq("A")],
        "pair": [bos, seq("A"), seq("B")],
        "special_tokens": {"<s>": {"id": "<s>", "ids": [256], "tokens": ["<s>"]}}
    });
    Tokenizer::from_hf_tokenizer(json.to_string().parse().unwrap())
}

#[cfg(test)]
fn test_tokenizer_json() -> serde_json::Value {
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;

    // Build vocabulary JSON from the byte-level alphabet.
    let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort();

    let mut vocab_entries = Vec::new();
    for (i, &c) in alphabet.iter().enumerate() {
        // Escape the character for JSON.
        let escaped = serde_json::to_string(&c.to_string()).unwrap();
        vocab_entries.push(format!("{escaped}:{i}"));
    }
    let vocab_json = format!("{{{}}}", vocab_entries.join(","));

    // Build a complete tokenizer JSON with byte-level BPE.
    let tokenizer_json = format!(
        r#"{{
            "version": "1.0",
            "model": {{
                "type": "BPE",
                "vocab": {vocab_json},
                "merges": []
            }},
            "pre_tokenizer": {{
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            }},
            "decoder": {{
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            }}
        }}"#
    );
    serde_json::from_str(&tokenizer_json).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `encode_through` counts, for each byte offset, the tokens that end at or before it: the
    /// prompt length that reaches that point and no further.
    #[test]
    fn test_tokenizer_encode_through_counts_the_tokens_before_a_mark() {
        let tok = make_test_tokenizer();
        let text = "hello world";
        let (ids, through) = tok.encode_through(text, false, &[0, 5, 11]).unwrap();
        assert_eq!(ids, tok.encode(text, false).unwrap());
        assert_eq!(through, vec![0, 5, 11]);
        let reached = tok.decode(&ids[..through[1] as usize], false).unwrap();
        assert_eq!(reached, "hello");
    }

    #[test]
    fn test_tokenizer_encode_decode_roundtrip() {
        let tok = make_test_tokenizer();
        let text = "Hello, world!";
        let ids = tok.encode(text, false).unwrap();
        assert!(!ids.is_empty());
        let decoded = tok.decode(&ids, false).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_tokenizer_encode_empty() {
        let tok = make_test_tokenizer();
        let ids = tok.encode("", false).unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn test_tokenizer_decode_empty() {
        let tok = make_test_tokenizer();
        let text = tok.decode(&[], false).unwrap();
        assert_eq!(text, "");
    }

    #[test]
    fn test_tokenizer_vocab_size() {
        let tok = make_test_tokenizer();
        // Byte-level BPE has at least 256 byte tokens.
        assert!(tok.vocab_size() >= 256);
    }

    #[test]
    fn test_tokenizer_debug() {
        let tok = make_test_tokenizer();
        let debug_str = format!("{tok:?}");
        assert!(debug_str.contains("Tokenizer"));
        assert!(debug_str.contains("vocab_size"));
    }

    #[test]
    fn test_tokenizer_multibyte_text() {
        let tok = make_test_tokenizer();
        let text = "こんにちは世界";
        let ids = tok.encode(text, false).unwrap();
        assert!(!ids.is_empty());
        let decoded = tok.decode(&ids, false).unwrap();
        assert_eq!(decoded, text);
    }
}
