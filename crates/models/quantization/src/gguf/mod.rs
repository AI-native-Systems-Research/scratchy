// SPDX-License-Identifier: Apache-2.0
//! GGUF (llama.cpp) file format support for scratchy.
//!
//! Format-level mechanics: the binary parser (in `format`), the
//! `GgufFile` reader, tokenizer reconstruction, chat-template extraction,
//! and the GGUF→HfModelConfig translator + tensor-name mapper.
//!
//! Per-arch concerns (HF arch class, qk-permute, tensor renames,
//! arch-specific metadata reads) live in each arch's
//! `configs/quantizations.json` `ggml` entry. The `scratchy-forward-compiler`
//! macro forwards each field straight into the [`register!`] call it
//! emits. There are NO per-arch arms in this crate — every per-arch
//! difference is a static-data record looked up by GGUF arch tag.

pub mod format;
mod spec;

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use tracing::info;

use scratchy_core_model::error::{ModelError, ModelResult};
use scratchy_core_model::weight::HfModelConfig;

pub use self::format::{Content, GgufDType, Shape, TensorInfo, Value, ValueType};
pub use self::spec::{GgufArchSpec, GgufDefault, apply_metadata, find_spec, lookup_rename};

/// Re-export `inventory` so the `register!` macro's expansion resolves
/// in consumer crates without their own `inventory` dep.
pub use inventory;

// ---------------------------------------------------------------------------
// GgufFile
// ---------------------------------------------------------------------------

/// Opened GGUF file with parsed header and lazy tensor reads.
pub struct GgufFile {
    pub(crate) content: Content,
}

impl GgufFile {
    /// Open and parse a GGUF file.
    pub fn open(path: impl AsRef<Path>) -> ModelResult<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|e| ModelError::Other(format!("failed to open {}: {e}", path.display())))?;
        let mut reader = BufReader::new(file);
        let content = Content::read(&mut reader).map_err(|e| {
            ModelError::Other(format!("failed to parse GGUF {}: {e}", path.display()))
        })?;
        Ok(Self { content })
    }

    /// List all tensor names in the GGUF file.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.content
            .tensor_infos
            .keys()
            .map(|s| s.as_str())
            .collect()
    }

    /// Raw GGUF metadata.
    pub fn metadata(&self) -> &HashMap<String, Value> {
        &self.content.metadata
    }

    /// Get a string metadata value.
    pub fn get_metadata_string(&self, key: &str) -> Option<&str> {
        self.content
            .metadata
            .get(key)
            .and_then(|v| v.to_string().ok())
            .map(|s| s.as_str())
    }

    /// Get a u32 metadata value (auto-upcasts from smaller int types).
    pub fn get_metadata_u32(&self, key: &str) -> Option<u32> {
        self.content.metadata.get(key).and_then(|v| {
            if let Ok(val) = v.to_u32() {
                Some(val)
            } else if let Ok(val) = v.to_u64() {
                Some(val as u32)
            } else {
                None
            }
        })
    }

    /// Get an f32 metadata value.
    ///
    /// Also handles integer-stored values (some GGUF writers store floats
    /// like `rope.freq_base` as integers).
    pub fn get_metadata_f32(&self, key: &str) -> Option<f32> {
        self.content.metadata.get(key).and_then(|v| {
            if let Ok(val) = v.to_f32() {
                Some(val)
            } else if let Ok(val) = v.to_f64() {
                Some(val as f32)
            } else if let Ok(val) = v.to_u64() {
                Some(val as f32)
            } else if let Ok(val) = v.to_u32() {
                Some(val as f32)
            } else if let Ok(val) = v.to_i32() {
                Some(val as f32)
            } else {
                None
            }
        })
    }

    /// Get the length of an array metadata value.
    pub fn get_metadata_array_len(&self, key: &str) -> Option<usize> {
        self.content.metadata.get(key).and_then(|v| {
            if let Value::Array(arr) = v {
                Some(arr.len())
            } else {
                None
            }
        })
    }

    /// Total number of tensors in the file.
    pub fn num_tensors(&self) -> usize {
        self.content.tensor_infos.len()
    }
}

// ---------------------------------------------------------------------------
// GgufModelConfig — extract HfModelConfig from GGUF metadata
// ---------------------------------------------------------------------------

/// Extract model configuration from GGUF metadata.
///
/// GGUF files embed model hyperparameters in their metadata using keys like
/// `llama.block_count`, `llama.embedding_length`, etc. This function reads
/// those values and converts them to an `HfModelConfig` for compatibility
/// with the existing init code.
/// Extract the chat template (Jinja2) string from a GGUF file's
/// metadata. GGUF embeds the same `chat_template` field that HF's
/// `tokenizer_config.json` carries — see llama.cpp's
/// `convert_hf_to_gguf.py`. Returns `None` when the GGUF doesn't
/// ship a template (rare for chat-tuned models, common for base
/// completions models).
pub fn gguf_chat_template(gguf: &GgufFile) -> Option<String> {
    gguf.get_metadata_string("tokenizer.chat_template")
        .map(str::to_string)
}

/// Build a SentencePiece (`model = "llama"`) tokenizer from GGUF
/// metadata. GGUF ships vocab strings (with `▁` already
/// substituted for spaces) and per-token f32 scores but no
/// merges. Mirrors HF's
/// `transformers/convert_slow_tokenizer.py::generate_merges` step
/// (the routine `LlamaConverter` / `GemmaConverter` /
/// `MistralConverter` invoke when materializing a fast
/// `tokenizer.json` from an SP `tokenizer.model`): for every
/// compound vocab entry, emit one merge per (left, right) split
/// where both halves are in the vocab; per-piece merges are
/// pre-sorted ascending by combined constituent rank, then a
/// stable global sort orders by `(score, len(left), len(right))`
/// descending so most-frequent parents merge first. Tokenization
/// matches `LlamaTokenizerFast` / `MistralTokenizerFast` /
/// `GemmaTokenizerFast` on the text shapes the models actually
/// see — verified against the HF reference for "What is the
/// capital of France?" across Mistral / Gemma2 / Gemma3.
///
/// Pre-tokenizer + decoder choice depends on the SP convention the
/// original tokenizer was trained with — see the dispatch on
/// `general.architecture` below.
/// Synthesize BPE merges from a SentencePiece vocab + scores.
/// Direct port of HF `transformers/convert_slow_tokenizer.py::generate_merges`.
///
/// Algorithm (verbatim from the Python):
///
/// ```text
/// for (piece, piece_score) in vocab_iteration_order:
///     local = []
///     for split in 1..codepoint_len(piece):
///         l, r = piece[:split], piece[split:]
///         if l in vocab and r in vocab:
///             local.append((l, r, piece_score))
///     local.sort(key=(vocab[l], vocab[r]))   # asc
///     merges.extend(local)
/// merges.sort(key=(score, len(l), len(r)), reverse=True)
/// ```
///
/// Per-piece local list orders constituent pairs by ascending
/// combined vocab index (most-canonical decomposition first); the
/// global stable sort puts highest-scoring parents first with
/// longer constituents leading on ties. Stable sort preserves the
/// local ordering among parents of equal score.
pub fn generate_merges(vocab_strings: &[String], scores: &[f32]) -> Vec<(String, String)> {
    use std::collections::HashMap;

    let token_to_id: HashMap<&str, u32> = vocab_strings
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i as u32))
        .collect();

    let mut all_merges: Vec<(f32, usize, usize, String, String)> = Vec::new();
    for (idx, piece) in vocab_strings.iter().enumerate() {
        let piece_score = scores[idx];
        // `range(1, len(piece))` in Python iterates codepoint positions;
        // pre-compute byte offsets for each char start so we can slice.
        let char_byte_starts: Vec<usize> = piece
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(piece.len()))
            .collect();
        let mut local: Vec<(u32, u32, &str, &str)> = Vec::new();
        // Skip index 0 (empty left) and the last entry (empty right).
        for &split in char_byte_starts
            .iter()
            .skip(1)
            .take(char_byte_starts.len().saturating_sub(2))
        {
            let left = &piece[..split];
            let right = &piece[split..];
            let (Some(&l_id), Some(&r_id)) = (token_to_id.get(left), token_to_id.get(right)) else {
                continue;
            };
            local.push((l_id, r_id, left, right));
        }
        local.sort_by_key(|t| (t.0, t.1));
        for (_, _, l, r) in local {
            all_merges.push((
                piece_score,
                l.chars().count(),
                r.chars().count(),
                l.to_string(),
                r.to_string(),
            ));
        }
    }
    all_merges.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.cmp(&a.1))
            .then(b.2.cmp(&a.2))
    });
    all_merges
        .into_iter()
        .map(|(_, _, _, l, r)| (l, r))
        .collect()
}

fn build_sentencepiece_tokenizer(gguf: &GgufFile) -> ModelResult<tokenizers::Tokenizer> {
    use tokenizers::AddedToken;
    use tokenizers::decoders::byte_fallback::ByteFallback;
    use tokenizers::decoders::fuse::Fuse;
    use tokenizers::decoders::sequence::Sequence as DecoderSequence;
    use tokenizers::decoders::strip::Strip as StripDecoder;
    use tokenizers::models::bpe::BPE;
    use tokenizers::normalizers::replace::Replace;
    use tokenizers::pre_tokenizers::metaspace::{Metaspace as MetaspacePre, PrependScheme};

    let tokens_val = gguf
        .metadata()
        .get("tokenizer.ggml.tokens")
        .ok_or_else(|| ModelError::Other("GGUF missing tokenizer.ggml.tokens".into()))?;
    let tokens = tokens_val
        .to_vec()
        .map_err(|e| ModelError::Other(format!("tokenizer.ggml.tokens: {e}")))?;
    let scores_val = gguf
        .metadata()
        .get("tokenizer.ggml.scores")
        .ok_or_else(|| ModelError::Other("GGUF missing tokenizer.ggml.scores".into()))?;
    let scores_raw = scores_val
        .to_vec()
        .map_err(|e| ModelError::Other(format!("tokenizer.ggml.scores: {e}")))?;
    if scores_raw.len() != tokens.len() {
        return Err(ModelError::Other(format!(
            "tokenizer.ggml.scores ({}) and tokens ({}) have mismatched lengths",
            scores_raw.len(),
            tokens.len()
        )));
    }
    let mut vocab_strings: Vec<String> = Vec::with_capacity(tokens.len());
    let mut scores_f32: Vec<f32> = Vec::with_capacity(tokens.len());
    for (idx, (tok, score)) in tokens.iter().zip(scores_raw.iter()).enumerate() {
        let s = tok
            .to_string()
            .map_err(|e| ModelError::Other(format!("tokenizer.ggml.tokens[{idx}]: {e}")))?;
        let v = score
            .to_f32()
            .map_err(|e| ModelError::Other(format!("tokenizer.ggml.scores[{idx}]: {e}")))?;
        vocab_strings.push(s.clone());
        scores_f32.push(v);
    }

    // Byte-fallback flag: presence of any BYTE-typed (token_type=6)
    // token in the vocab means the tokenizer encodes unknown bytes
    // via the `<0xNN>` byte tokens rather than the `unk` token. All
    // SP checkpoints we care about (Llama-2 / Mistral / Gemma) ship
    // these.
    let _byte_fallback = gguf
        .metadata()
        .get("tokenizer.ggml.token_type")
        .and_then(|v| v.to_vec().ok())
        .map(|types| {
            types.iter().any(|t| {
                t.to_u32()
                    .map(|x| x == 6)
                    .or_else(|_| t.to_i32().map(|x| x == 6))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    let merges = generate_merges(&vocab_strings, &scores_f32);

    let vocab_map: tokenizers::models::bpe::Vocab = vocab_strings
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i as u32))
        .collect();
    let bpe = BPE::builder()
        .vocab_and_merges(vocab_map, merges)
        .build()
        .map_err(|e| ModelError::Other(format!("SentencePiece-BPE build failed: {e}")))?;
    let mut tokenizer = tokenizers::Tokenizer::new(bpe);

    // Pre-tokenizer / decoder choice depends on the SP convention
    // the original tokenizer was trained with. The canonical
    // signal is `tokenizer.ggml.add_space_prefix` — llama.cpp's
    // direct mirror of SentencePiece's `add_dummy_prefix`
    // training option:
    //
    //   * `add_space_prefix = false` (Gemma family): NO metaspace
    //     prepend. HF fast tokenizer uses a Replace normalizer
    //     (`" " → "▁"`) and lets leading-no-space pieces stay
    //     un-prefixed; the decoder is `Replace + ByteFallback +
    //     Fuse` with no trailing Strip.
    //
    //   * `add_space_prefix = true` OR ABSENT (Llama-2 / Mistral /
    //     Phi-3 / etc.): Metaspace pre-tokenizer with
    //     `prepend_scheme = First` so a leading-no-space input
    //     like "What is..." tokenizes as `▁What ▁is ...`
    //     (matching the training-time tokenization). Decoder ends
    //     with `Strip(' ', start=1, stop=0)` to remove the
    //     synthesized prefix space at output.
    //
    // Earlier versions branched on `general.architecture`
    // (`gemma*`); that's strictly less reliable than the explicit
    // metadata field which every llama.cpp-converted SP GGUF
    // ships when relevant. Default to prepend (the SP default)
    // when the field is absent.
    let prepend_metaspace = gguf
        .metadata()
        .get("tokenizer.ggml.add_space_prefix")
        .and_then(|v| v.to_bool().ok())
        .unwrap_or(true);
    if !prepend_metaspace {
        let normalizer: tokenizers::NormalizerWrapper = Replace::new(" ", "▁")
            .map_err(|e| ModelError::Other(format!("Replace normalizer: {e}")))?
            .into();
        tokenizer
            .with_normalizer(Some(normalizer))
            .map_err(|e| ModelError::Other(format!("set normalizer: {e}")))?;
        let dec_replace: tokenizers::DecoderWrapper = Replace::new("▁", " ")
            .map_err(|e| ModelError::Other(format!("Replace decoder: {e}")))?
            .into();
        let dec_byte_fallback: tokenizers::DecoderWrapper = ByteFallback::new().into();
        let dec_fuse: tokenizers::DecoderWrapper = Fuse::new().into();
        let dec_seq: tokenizers::DecoderWrapper =
            DecoderSequence::new(vec![dec_replace, dec_byte_fallback, dec_fuse]).into();
        tokenizer.with_decoder(Some(dec_seq));
    } else {
        let pre: tokenizers::PreTokenizerWrapper =
            MetaspacePre::new('▁', PrependScheme::First, false).into();
        tokenizer.with_pre_tokenizer(Some(pre));
        let dec_replace: tokenizers::DecoderWrapper = Replace::new("▁", " ")
            .map_err(|e| ModelError::Other(format!("Replace decoder: {e}")))?
            .into();
        let dec_byte_fallback: tokenizers::DecoderWrapper = ByteFallback::new().into();
        let dec_fuse: tokenizers::DecoderWrapper = Fuse::new().into();
        let dec_strip: tokenizers::DecoderWrapper = StripDecoder::new(' ', 1, 0).into();
        let dec_seq: tokenizers::DecoderWrapper =
            DecoderSequence::new(vec![dec_replace, dec_byte_fallback, dec_fuse, dec_strip]).into();
        tokenizer.with_decoder(Some(dec_seq));
    }

    // Special tokens. Same logic as the BPE path: register CONTROL
    // (token_type=3) and USER_DEFINED (token_type=4) tokens as
    // special-added so the chat template's literal `<|start_header|>`-
    // style markers tokenize atomically; plus the explicit-id BOS /
    // EOS / UNK / PAD entries from metadata.
    let mut added: Vec<AddedToken> = Vec::new();
    if let Some(types_val) = gguf.metadata().get("tokenizer.ggml.token_type")
        && let Ok(types) = types_val.to_vec()
    {
        for (idx, t) in types.iter().enumerate() {
            let kind = t
                .to_u32()
                .map(|x| x as i64)
                .or_else(|_| t.to_i32().map(|x| x as i64))
                .unwrap_or(1);
            if (kind == 3 || kind == 4)
                && let Some(token_val) = tokens.get(idx)
                && let Ok(s) = token_val.to_string()
            {
                added.push(AddedToken::from(s.clone(), true));
            }
        }
    }
    let token_str_for_id = |id: u32| -> Option<String> {
        let s = tokens.get(id as usize)?.to_string().ok()?.clone();
        Some(s)
    };
    for key in [
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.pad_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.padding_token_id",
    ] {
        if let Some(id) = gguf.get_metadata_u32(key)
            && let Some(content) = token_str_for_id(id)
        {
            added.push(AddedToken::from(content, true));
        }
    }
    if !added.is_empty() {
        tokenizer
            .add_special_tokens(added)
            .map_err(|e| ModelError::Other(format!("add special tokens: {e}")))?;
    }

    Ok(tokenizer)
}

/// Build a `tokenizers::Tokenizer` from GGUF metadata. Handles the
/// `gpt2` BPE family (Llama-3, Mistral-3-style models that use
/// `pre = "llama-bpe"` etc.) — vocab from
/// `tokenizer.ggml.tokens`, merges from `tokenizer.ggml.merges`,
/// special tokens from `tokenizer.ggml.{bos,eos,pad,unk}_token_id`.
///
/// Returns `Ok(None)` for GGUFs whose `tokenizer.ggml.model` is
/// neither `"gpt2"` nor `"llama"` (these would need
/// SentencePiece/Unigram support and a different builder); the
/// caller should fall back to a sibling `tokenizer.json` for those.
pub fn gguf_tokenizer(gguf: &GgufFile) -> ModelResult<Option<tokenizers::Tokenizer>> {
    use tokenizers::AddedToken;
    use tokenizers::SplitDelimiterBehavior;
    use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::PreTokenizerWrapper;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPre;
    use tokenizers::pre_tokenizers::sequence::Sequence as PreSequence;
    use tokenizers::pre_tokenizers::split::{Split, SplitPattern};

    let Some(model) = gguf.get_metadata_string("tokenizer.ggml.model") else {
        return Ok(None);
    };
    // Llama-3 / Mistral-3 / Qwen-2/3 / etc. all use `model = "gpt2"`
    // (byte-level BPE). `model = "llama"` is the SentencePiece tokenizer
    // shared by Llama-2 / Mistral / Gemma / etc. — vocab + per-token
    // scores + optional byte-fallback. Reconstructed via the
    // `tokenizers::Unigram` model below.
    if model == "llama" {
        return build_sentencepiece_tokenizer(gguf).map(Some);
    }
    if model != "gpt2" {
        tracing::info!(
            "gguf_tokenizer: unsupported tokenizer model `{model}` (only `gpt2` BPE \
             and `llama` SentencePiece are reconstructable today); caller should \
             fall back to tokenizer.json"
        );
        return Ok(None);
    }

    // Vocab: array of strings indexed by token id.
    let tokens_val = gguf
        .metadata()
        .get("tokenizer.ggml.tokens")
        .ok_or_else(|| ModelError::Other("GGUF missing tokenizer.ggml.tokens".into()))?;
    let tokens = tokens_val
        .to_vec()
        .map_err(|e| ModelError::Other(format!("tokenizer.ggml.tokens: {e}")))?;
    let mut vocab_pairs: Vec<(String, u32)> = Vec::with_capacity(tokens.len());
    for (idx, t) in tokens.iter().enumerate() {
        let s = t
            .to_string()
            .map_err(|e| ModelError::Other(format!("tokenizer.ggml.tokens[{idx}]: {e}")))?;
        vocab_pairs.push((s.clone(), idx as u32));
    }
    let vocab: tokenizers::models::bpe::Vocab = vocab_pairs.into_iter().collect();

    // Merges: array of "left right" strings.
    let merges_val = gguf
        .metadata()
        .get("tokenizer.ggml.merges")
        .ok_or_else(|| ModelError::Other("GGUF missing tokenizer.ggml.merges".into()))?;
    let merges_raw = merges_val
        .to_vec()
        .map_err(|e| ModelError::Other(format!("tokenizer.ggml.merges: {e}")))?;
    let mut merges: Vec<(String, String)> = Vec::with_capacity(merges_raw.len());
    for (idx, m) in merges_raw.iter().enumerate() {
        let s = m
            .to_string()
            .map_err(|e| ModelError::Other(format!("tokenizer.ggml.merges[{idx}]: {e}")))?;
        let mut parts = s.splitn(2, ' ');
        let l = parts
            .next()
            .ok_or_else(|| ModelError::Other(format!("merge {idx}: empty")))?;
        let r = parts
            .next()
            .ok_or_else(|| ModelError::Other(format!("merge {idx}: missing right side `{s}`")))?;
        merges.push((l.to_string(), r.to_string()));
    }

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| ModelError::Other(format!("BPE build failed: {e}")))?;

    let mut tokenizer = tokenizers::Tokenizer::new(bpe);

    // Pre-tokenizer. `tokenizer.ggml.pre` names the family. Almost
    // every modern HF tokenizer is structured as:
    //   Sequence([Split(<arch_regex>, Isolated), ByteLevel(false, false, false)])
    // The arch differs only in the splitting regex — so the regex
    // table below is the single point of variance. ByteLevel always
    // runs with `add_prefix_space=false, trim_offsets=false,
    // use_regex=false` (the regex Split already did the splitting;
    // ByteLevel just maps bytes through the byte-to-unicode table).
    //
    // Wrong choice here is the #1 cause of gibberish output — picking
    // `ByteLevel::default()` (which has `add_prefix_space=true,
    // use_regex=true`) silently inserts a leading space on every
    // word, offsetting every token id and destroying inference.
    let pre = gguf
        .get_metadata_string("tokenizer.ggml.pre")
        .unwrap_or("default");
    let llama3_regex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    // Qwen2 / Qwen2.5 / Qwen3: same family as Llama-3 but `\p{N}`
    // instead of `\p{N}{1,3}` (digits split one-at-a-time).
    let qwen2_regex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    // GPT-2 base regex (also fits `default` and tokenizers whose
    // pre-tokenizer field is absent — the safest fallback that still
    // keeps `add_prefix_space=false` so we don't silently corrupt
    // tokenization).
    let gpt2_regex =
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
    let regex_str: &str = match pre {
        "llama-bpe" | "llama3" | "llama-v3" => llama3_regex,
        "qwen2" => qwen2_regex,
        _ => gpt2_regex,
    };
    let regex_split = Split::new(
        SplitPattern::Regex(regex_str.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| ModelError::Other(format!("pre-tokenizer regex compile ({pre}): {e}")))?;
    let bl = ByteLevelPre::new(false, false, false);
    let pre_tokenizer: PreTokenizerWrapper =
        PreSequence::new(vec![regex_split.into(), bl.into()]).into();
    tokenizer.with_pre_tokenizer(Some(pre_tokenizer));
    // Decoder: ByteLevel with `add_prefix_space=false, trim_offsets=false`
    // matches HF tokenizer.json on Llama-3/Qwen2/Qwen3/Mistral families.
    // The default ctor has `add_prefix_space=true` which strips a
    // leading space from every decoded token — visible as missing
    // spaces in user-facing output.
    tokenizer.with_decoder(Some(ByteLevelDecoder::new(false, false, false)));

    // Special tokens. GGUF marks token roles via
    // `tokenizer.ggml.token_type` (parallel to tokens array):
    //   1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED,
    //   5=UNUSED, 6=BYTE.
    // Anything that's CONTROL or USER_DEFINED needs to be registered
    // as a special added token so the tokenizer recognizes the
    // multi-byte form (e.g. `<|begin_of_text|>`) atomically rather
    // than BPE-encoding it. Without this, chat templates produce
    // nonsense input to the model.
    let mut added: Vec<AddedToken> = Vec::new();
    if let Some(types_val) = gguf.metadata().get("tokenizer.ggml.token_type")
        && let Ok(types) = types_val.to_vec()
    {
        for (idx, t) in types.iter().enumerate() {
            // GGUF token_type is typically I32 (signed) — try both.
            let kind = t
                .to_u32()
                .map(|x| x as i64)
                .or_else(|_| t.to_i32().map(|x| x as i64))
                .unwrap_or(1);
            if (kind == 3 || kind == 4)
                && let Some(token_val) = tokens.get(idx)
                && let Ok(s) = token_val.to_string()
            {
                added.push(AddedToken::from(s.clone(), true));
            }
        }
    }
    // Always add the explicit-id specials too (some GGUFs don't set
    // token_type for these or set to NORMAL).
    let token_str_for_id = |id: u32| -> Option<String> {
        let s = tokens.get(id as usize)?.to_string().ok()?.clone();
        Some(s)
    };
    for key in [
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.pad_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.padding_token_id",
    ] {
        if let Some(id) = gguf.get_metadata_u32(key)
            && let Some(content) = token_str_for_id(id)
        {
            added.push(AddedToken::from(content, true));
        }
    }
    if !added.is_empty() {
        tokenizer
            .add_special_tokens(added)
            .map_err(|e| ModelError::Other(format!("add special tokens: {e}")))?;
    }

    Ok(Some(tokenizer))
}

pub fn gguf_model_config(gguf: &GgufFile) -> ModelResult<HfModelConfig> {
    // Determine the architecture prefix (e.g., "llama", "qwen2").
    let arch = gguf
        .get_metadata_string("general.architecture")
        .ok_or_else(|| ModelError::Other("GGUF missing general.architecture metadata".to_string()))?
        .to_string();

    // Per-arch spec: declared by the canonical owner crate via
    // `configs/quantizations.json` `ggml` entry (the macro forwards
    // each field into a `register!` call). The spec carries
    // qk_permute, tensor renames, and metadata reads — every per-arch
    // knob lives in the JSON, not here. Non-canonical claimants of
    // the same gguf_arch (e.g. Mistral for `"llama"`) opt out of
    // registration via `"register_spec": false` so a single spec
    // covers the family deterministically.
    let spec = find_spec(&arch).ok_or_else(|| {
        ModelError::Other(format!(
            "no GGUF arch registered for `general.architecture = \"{arch}\"`. \
             Each arch declares its GGUF support by listing \
             `\"ggml\"` (or a `{{\"ggml\": {{...}}}}` object form for archs \
             that need overrides) in `configs/quantizations.json`."
        ))
    })?;

    // The arch hint stamped into `architectures` is the GGUF tag
    // itself (`"deepseek2"`, `"llama"`, …), not an HF class string.
    // Forward arches advertise their gguf-tag claims via
    // `ScratchyArchRegistration::gguf_archs`; the worker hands the
    // first `architectures` entry to `try_load`, which filters by
    // (`hf_arches` ∪ `gguf_archs`). Stamping the gguf tag means a
    // single tag like `"deepseek2"` fans out to every claiming forward
    // (V2 / V3-LoRA / V3-flat) without each having to declare HF
    // class aliases.
    let mut config = HfModelConfig {
        model_type: Some(arch.clone()),
        architectures: vec![arch.clone()],
        ..Default::default()
    };

    // Read hyperparameters using the architecture prefix.
    config.num_hidden_layers = gguf
        .get_metadata_u32(&format!("{arch}.block_count"))
        .map(|v| v as usize);
    config.hidden_size = gguf
        .get_metadata_u32(&format!("{arch}.embedding_length"))
        .map(|v| v as usize);
    config.num_attention_heads = gguf
        .get_metadata_u32(&format!("{arch}.attention.head_count"))
        .map(|v| v as usize);
    config.num_key_value_heads = gguf
        .get_metadata_u32(&format!("{arch}.attention.head_count_kv"))
        .map(|v| v as usize);
    config.intermediate_size = gguf
        .get_metadata_u32(&format!("{arch}.feed_forward_length"))
        .map(|v| v as usize);
    config.vocab_size = gguf
        .get_metadata_u32(&format!("{arch}.vocab_size"))
        .map(|v| v as usize);
    // Fallback: try general.vocab_size, then tokenizer.ggml.tokens array length.
    if config.vocab_size.is_none() {
        config.vocab_size = gguf
            .get_metadata_u32("general.vocab_size")
            .map(|v| v as usize);
    }
    if config.vocab_size.is_none() {
        config.vocab_size = gguf.get_metadata_array_len("tokenizer.ggml.tokens");
    }
    config.max_position_embeddings = gguf
        .get_metadata_u32(&format!("{arch}.context_length"))
        .map(|v| v as usize);
    config.rope_theta = gguf
        .get_metadata_f32(&format!("{arch}.rope.freq_base"))
        .map(|v| v as f64);
    config.rms_norm_eps = gguf
        .get_metadata_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        .map(|v| v as f64);

    // rope_scaling: parse if the GGUF records it, otherwise infer llama3 defaults
    // for Llama-3.x files (unsloth GGUFs omit the scaling keys entirely).
    // Keys follow llama.cpp convention: {arch}.rope.scaling.type and friends.
    // llama.cpp emits `rope.scaling.type = "none"` for archs whose HF
    // config has no rope_scaling key (CommandR-v01 is the canonical
    // example). Treat it as no-scaling — leaving it as a populated
    // `rope_scaling` object would fail the fingerprint check on every
    // arch whose manifest declares no rope_scaling.
    let rope_scaling_type = gguf
        .get_metadata_string(&format!("{arch}.rope.scaling.type"))
        .filter(|t| *t != "none");
    let has_scaling_keys = rope_scaling_type.is_some()
        || gguf
            .get_metadata_f32(&format!("{arch}.rope.scaling.factor"))
            .is_some();
    if has_scaling_keys {
        let rope_type = rope_scaling_type.unwrap_or("linear").to_string();
        let factor = gguf
            .get_metadata_f32(&format!("{arch}.rope.scaling.factor"))
            .unwrap_or(1.0) as f64;
        let mut scaling = serde_json::Map::new();
        // The HF key spelling for the rope-type discriminator is not
        // standardized: Llama-3 / Mistral / Phi-3 configs use
        // `rope_type`, DeepSeek-V2 / V3 (yarn) configs use `type`. The
        // safetensors `rope_scaling_hash` is computed from the literal
        // JSON object, so the runtime stamp must use the spelling
        // that arch's checkpoints use — otherwise fingerprint
        // dispatch rejects on hash mismatch.
        let type_key = if rope_type == "yarn" {
            "type"
        } else {
            "rope_type"
        };
        scaling.insert(
            type_key.to_string(),
            serde_json::Value::String(rope_type.clone()),
        );
        scaling.insert("factor".to_string(), serde_json::Value::from(factor));
        if let Some(orig) =
            gguf.get_metadata_u32(&format!("{arch}.rope.scaling.original_context_length"))
        {
            scaling.insert(
                "original_max_position_embeddings".to_string(),
                serde_json::Value::from(orig as u64),
            );
        }
        if rope_type == "llama3" {
            if let Some(lo) = gguf.get_metadata_f32(&format!("{arch}.rope.scaling.low_freq_factor"))
            {
                scaling.insert(
                    "low_freq_factor".to_string(),
                    serde_json::Value::from(lo as f64),
                );
            }
            if let Some(hi) =
                gguf.get_metadata_f32(&format!("{arch}.rope.scaling.high_freq_factor"))
            {
                scaling.insert(
                    "high_freq_factor".to_string(),
                    serde_json::Value::from(hi as f64),
                );
            }
        }
        // Yarn (DeepSeek V2 / V3) — read the four extra knobs that
        // discriminate yarn checkpoints. llama.cpp's GGUF converter
        // emits each as a top-level f32 under `{arch}.rope.scaling.*`;
        // the safetensors HF config carries them in the same nested
        // `rope_scaling` object hashed at compile time.
        if rope_type == "yarn" {
            for (gguf_key, hf_key) in [
                ("beta_fast", "beta_fast"),
                ("beta_slow", "beta_slow"),
                ("mscale", "mscale"),
                ("mscale_all_dim", "mscale_all_dim"),
            ] {
                if let Some(v) = gguf.get_metadata_f32(&format!("{arch}.rope.scaling.{gguf_key}")) {
                    scaling.insert(hf_key.to_string(), serde_json::Value::from(v as f64));
                }
            }
        }
        config.extra.insert(
            "rope_scaling".to_string(),
            serde_json::Value::Object(scaling),
        );
    }

    // Compute head_dim: try arch-specific key_length first, fall back to hidden/heads.
    if let Some(key_len) = gguf.get_metadata_u32(&format!("{arch}.attention.key_length")) {
        config.head_dim = Some(key_len as usize);
    } else if let (Some(hidden), Some(heads)) = (config.hidden_size, config.num_attention_heads)
        && heads > 0
    {
        config.head_dim = Some(hidden / heads);
    }

    // Per-arch metadata reads + defaults + Llama-3 rope-scaling
    // inference, all driven by the spec's pure-data declarations
    // (Gemma3 sliding window, Granite multipliers, etc.). No
    // per-arch arms here.
    apply_metadata(spec, gguf, &mut config);

    // EOS token ID (stored in tokenizer.ggml.eos_token_id or general.eos_token_id).
    if let Some(eos) = gguf.get_metadata_u32("tokenizer.ggml.eos_token_id") {
        config
            .extra
            .insert("eos_token_id".to_string(), serde_json::json!(eos));
    }

    info!(
        "GGUF config: arch={}, hidden={:?}, heads={:?}, kv_heads={:?}, layers={:?}, \
         intermediate={:?}, vocab={:?}, max_pos={:?}, rope_theta={:?}, rms_eps={:?}, head_dim={:?}",
        arch,
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.num_hidden_layers,
        config.intermediate_size,
        config.vocab_size,
        config.max_position_embeddings,
        config.rope_theta,
        config.rms_norm_eps,
        config.head_dim,
    );

    Ok(config)
}

// ---------------------------------------------------------------------------
// Tensor name mapping: GGUF (llama.cpp) → HuggingFace convention
// ---------------------------------------------------------------------------

/// Map a GGUF tensor name to the HuggingFace convention, given the
/// GGUF's `general.architecture` value.
///
/// llama.cpp uses names like `blk.0.attn_q.weight` while HF uses
/// `model.layers.0.self_attn.q_proj.weight`. The default rename is
/// Llama-shape and covers every arm that's globally unambiguous
/// (per-bias, MLA, MoE, etc.). When an arch's GGUF tensor name maps
/// to a different HF target than the default — Phi3's fused
/// `attn_qkv` + `ffn_up` (gate-up), Gemma2/3's pre/post norms — that
/// arch's `configs/quantizations.json` declares the override under
/// `tensor_renames`, and the spec's lookup runs first.
pub fn gguf_to_hf_name(gguf_name: &str, gguf_arch: &str) -> String {
    // Per-layer overrides apply first. The layer number is preserved
    // and only the suffix gets the override lookup.
    if let Some(rest) = gguf_name.strip_prefix("blk.")
        && let Some(dot_pos) = rest.find('.')
    {
        let layer_num = &rest[..dot_pos];
        let suffix = &rest[dot_pos + 1..];
        if let Some(spec) = find_spec(gguf_arch)
            && let Some(hf_suffix) = lookup_rename(spec, suffix)
        {
            return format!("model.layers.{layer_num}.{hf_suffix}");
        }
    }
    default_gguf_to_hf_name(gguf_name)
}

/// Llama-shape default rename. Public so per-arch overrides can fall
/// through to it for tensor names they don't override (typically all
/// of them — overrides are usually 2-4 arms).
pub fn default_gguf_to_hf_name(gguf_name: &str) -> String {
    // Global tensors.
    if gguf_name == "token_embd.weight" {
        return "model.embed_tokens.weight".to_string();
    }
    if gguf_name == "output_norm.weight" {
        return "model.norm.weight".to_string();
    }
    if gguf_name == "output.weight" {
        return "lm_head.weight".to_string();
    }

    // Per-layer tensors: blk.{i}.{suffix}
    if let Some(rest) = gguf_name.strip_prefix("blk.")
        && let Some(dot_pos) = rest.find('.')
    {
        let layer_num = &rest[..dot_pos];
        let suffix = &rest[dot_pos + 1..];

        let hf_suffix = match suffix {
            // Standard attention
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_output.weight" | "attn_o.weight" => "self_attn.o_proj.weight",
            "attn_norm.weight" => "input_layernorm.weight",
            "attn_q_norm.weight" => "self_attn.q_norm.weight",
            "attn_k_norm.weight" => "self_attn.k_norm.weight",
            // Biases (Qwen2/2.5 ships q/k/v biases; Llama / Mistral / etc. don't).
            // Without these arms the bias falls through to `model.layers.{N}.attn_q.bias`,
            // but the downstream loader looks for `model.layers.{N}.self_attn.q_proj.bias`,
            // and silently drops the bias — producing gibberish on biased archs.
            "attn_q.bias" => "self_attn.q_proj.bias",
            "attn_k.bias" => "self_attn.k_proj.bias",
            "attn_v.bias" => "self_attn.v_proj.bias",
            "attn_output.bias" | "attn_o.bias" => "self_attn.o_proj.bias",
            // MLA attention (DeepSeek V2/V3)
            "attn_q_a.weight" => "self_attn.q_a_proj.weight",
            "attn_q_a_norm.weight" => "self_attn.q_a_layernorm.weight",
            "attn_q_b.weight" => "self_attn.q_b_proj.weight",
            "attn_kv_a_mqa.weight" => "self_attn.kv_a_proj_with_mqa.weight",
            "attn_kv_a_norm.weight" => "self_attn.kv_a_layernorm.weight",
            "attn_kv_b.weight" => "self_attn.kv_b_proj.weight",
            // Standard MLP
            "ffn_gate.weight" => "mlp.gate_proj.weight",
            "ffn_up.weight" => "mlp.up_proj.weight",
            "ffn_down.weight" => "mlp.down_proj.weight",
            "ffn_norm.weight" => "post_attention_layernorm.weight",
            // MoE router gate
            "ffn_gate_inp.weight" => "mlp.gate.weight",
            // Fused 3D expert weights (kept as fused_ prefix for loader to handle)
            "ffn_gate_exps.weight" => "mlp.experts.fused_gate_exps.weight",
            "ffn_up_exps.weight" => "mlp.experts.fused_up_exps.weight",
            "ffn_down_exps.weight" => "mlp.experts.fused_down_exps.weight",
            // Shared experts (DeepSeek V2/V3)
            "ffn_gate_shexp.weight" => "mlp.shared_experts.gate_proj.weight",
            "ffn_up_shexp.weight" => "mlp.shared_experts.up_proj.weight",
            "ffn_down_shexp.weight" => "mlp.shared_experts.down_proj.weight",
            // DeepSeek V3 score correction bias
            "exp_probs_b.bias" => "mlp.gate.e_score_correction_bias",
            other => return format!("model.layers.{layer_num}.{other}"),
        };

        return format!("model.layers.{layer_num}.{hf_suffix}");
    }

    // Unknown — pass through unchanged.
    gguf_name.to_string()
}

// ---------------------------------------------------------------------------
// Multimodal GGUF: mmproj detection
// ---------------------------------------------------------------------------

/// Detect a sibling mmproj GGUF file next to a main GGUF file.
///
/// Looks for files matching `mmproj*.gguf` in the same directory as
/// `main_gguf_path`. Returns the first match, if any.
pub fn detect_mmproj_gguf(main_gguf_path: &Path) -> Option<std::path::PathBuf> {
    let parent = main_gguf_path.parent()?;
    let entries = std::fs::read_dir(parent).ok()?;
    for entry in entries.flatten() {
        let fname = entry.file_name();
        let fname_str = fname.to_string_lossy();
        if fname_str.starts_with("mmproj")
            && fname_str.ends_with(".gguf")
            && entry.path() != main_gguf_path
        {
            return Some(entry.path());
        }
    }
    None
}

/// Map an mmproj GGUF tensor name to the HuggingFace convention.
pub fn mmproj_gguf_to_hf_name(gguf_name: &str) -> String {
    // Projector tensors.
    if gguf_name == "mm.0.weight" {
        return "multi_modal_projector.mm_input_projection_weight".to_string();
    }
    if gguf_name == "mm.model_norm.weight" {
        return "multi_modal_projector.mm_soft_emb_norm.weight".to_string();
    }

    // Vision global tensors.
    if gguf_name == "v.patch_embd.weight" {
        return "vision_tower.vision_model.embeddings.patch_embedding.weight".to_string();
    }
    if gguf_name == "v.patch_embd.bias" {
        return "vision_tower.vision_model.embeddings.patch_embedding.bias".to_string();
    }
    if gguf_name == "v.position_embd.weight" {
        return "vision_tower.vision_model.embeddings.position_embedding.weight".to_string();
    }
    if gguf_name == "v.post_ln.weight" {
        return "vision_tower.vision_model.post_layernorm.weight".to_string();
    }
    if gguf_name == "v.post_ln.bias" {
        return "vision_tower.vision_model.post_layernorm.bias".to_string();
    }

    // Vision encoder per-layer tensors: v.blk.{i}.{suffix}
    if let Some(rest) = gguf_name.strip_prefix("v.blk.")
        && let Some(dot_pos) = rest.find('.')
    {
        let layer_num = &rest[..dot_pos];
        let suffix = &rest[dot_pos + 1..];
        let prefix = format!("vision_tower.vision_model.encoder.layers.{layer_num}");

        let hf_suffix = match suffix {
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_q.bias" => "self_attn.q_proj.bias",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_k.bias" => "self_attn.k_proj.bias",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_v.bias" => "self_attn.v_proj.bias",
            "attn_output.weight" => "self_attn.out_proj.weight",
            "attn_output.bias" => "self_attn.out_proj.bias",
            "ln1.weight" => "layer_norm1.weight",
            "ln1.bias" => "layer_norm1.bias",
            "ln2.weight" => "layer_norm2.weight",
            "ln2.bias" => "layer_norm2.bias",
            "ffn_up.weight" => "mlp.fc1.weight",
            "ffn_up.bias" => "mlp.fc1.bias",
            "ffn_down.weight" => "mlp.fc2.weight",
            "ffn_down.bias" => "mlp.fc2.bias",
            other => return format!("{prefix}.{other}"),
        };

        return format!("{prefix}.{hf_suffix}");
    }

    // Unknown — pass through unchanged.
    gguf_name.to_string()
}

/// Extract vision config from mmproj GGUF metadata and populate the
/// HfModelConfig extra fields needed for multimodal loading.
pub fn extract_mmproj_vision_config(
    mmproj_path: &Path,
    config: &mut HfModelConfig,
) -> ModelResult<()> {
    let gguf = GgufFile::open(mmproj_path)?;
    let meta = gguf.metadata();

    // Build vision_config JSON from mmproj metadata.
    let mut vision = serde_json::Map::new();

    // Try clip.vision.* keys (llama.cpp mmproj convention).
    if let Some(v) = meta
        .get("clip.vision.image_size")
        .and_then(|v| v.to_u32().ok())
    {
        vision.insert("image_size".to_string(), serde_json::json!(v));
    }
    if let Some(v) = meta
        .get("clip.vision.patch_size")
        .and_then(|v| v.to_u32().ok())
    {
        vision.insert("patch_size".to_string(), serde_json::json!(v));
    }
    if let Some(v) = meta
        .get("clip.vision.embedding_length")
        .and_then(|v| v.to_u32().ok())
    {
        vision.insert("hidden_size".to_string(), serde_json::json!(v));
    }
    if let Some(v) = meta
        .get("clip.vision.feed_forward_length")
        .and_then(|v| v.to_u32().ok())
    {
        vision.insert("intermediate_size".to_string(), serde_json::json!(v));
    }
    if let Some(v) = meta
        .get("clip.vision.block_count")
        .and_then(|v| v.to_u32().ok())
    {
        vision.insert("num_hidden_layers".to_string(), serde_json::json!(v));
    }
    if let Some(v) = meta
        .get("clip.vision.head_count")
        .and_then(|v| v.to_u32().ok())
    {
        vision.insert("num_attention_heads".to_string(), serde_json::json!(v));
    }

    // layer_norm_eps: try clip.vision.attention.layer_norm_epsilon.
    if let Some(v) = meta
        .get("clip.vision.attention.layer_norm_epsilon")
        .and_then(|v| v.to_f32().ok())
    {
        vision.insert("layer_norm_eps".to_string(), serde_json::json!(v as f64));
    }

    if !vision.is_empty() {
        config.extra.insert(
            "vision_config".to_string(),
            serde_json::Value::Object(vision),
        );

        // mm_tokens_per_image: compute from image_size and patch_size if available.
        if let (Some(img_size), Some(patch_size)) = (
            config
                .extra
                .get("vision_config")
                .and_then(|v| v.get("image_size"))
                .and_then(|v| v.as_u64()),
            config
                .extra
                .get("vision_config")
                .and_then(|v| v.get("patch_size"))
                .and_then(|v| v.as_u64()),
        ) {
            let patches_per_side = img_size / patch_size;
            let num_patches = patches_per_side * patches_per_side;
            if !config.extra.contains_key("mm_tokens_per_image") {
                config.extra.insert(
                    "mm_tokens_per_image".to_string(),
                    serde_json::json!(num_patches),
                );
            }
        }

        info!(
            "Extracted vision config from mmproj: {:?}",
            config.extra.get("vision_config")
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Inspect tensor shapes in a real GGUF file (run manually).
    #[test]
    #[ignore]
    fn inspect_gguf_shapes() {
        let home = std::env::var("HOME").unwrap();
        let path = format!(
            "{home}/.cache/huggingface/hub/models--unsloth--Llama-3.1-8B-Instruct-GGUF/snapshots/600b0020115fd6b17f0752848fe7b5a1be686bcd/Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        );
        if !std::path::Path::new(&path).exists() {
            eprintln!("GGUF file not found, skipping");
            return;
        }
        let gguf = GgufFile::open(&path).unwrap();
        let names = [
            "token_embd.weight",
            "output.weight",
            "output_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.ffn_norm.weight",
        ];
        for name in names {
            if let Some(info) = gguf.content.tensor_infos.get(name) {
                println!(
                    "{name}: shape={:?}, dtype={:?}",
                    info.shape.dims(),
                    info.ggml_dtype
                );
            } else {
                println!("{name}: NOT FOUND");
            }
        }
    }

    #[test]
    fn test_gguf_to_hf_name_embedding() {
        assert_eq!(
            default_gguf_to_hf_name("token_embd.weight"),
            "model.embed_tokens.weight"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_output_norm() {
        assert_eq!(
            default_gguf_to_hf_name("output_norm.weight"),
            "model.norm.weight"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_lm_head() {
        assert_eq!(default_gguf_to_hf_name("output.weight"), "lm_head.weight");
    }

    #[test]
    fn test_gguf_to_hf_name_attention() {
        assert_eq!(
            default_gguf_to_hf_name("blk.0.attn_q.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            default_gguf_to_hf_name("blk.5.attn_k.weight"),
            "model.layers.5.self_attn.k_proj.weight"
        );
        assert_eq!(
            default_gguf_to_hf_name("blk.31.attn_v.weight"),
            "model.layers.31.self_attn.v_proj.weight"
        );
        assert_eq!(
            default_gguf_to_hf_name("blk.0.attn_output.weight"),
            "model.layers.0.self_attn.o_proj.weight"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_norms() {
        assert_eq!(
            default_gguf_to_hf_name("blk.0.attn_norm.weight"),
            "model.layers.0.input_layernorm.weight"
        );
        assert_eq!(
            default_gguf_to_hf_name("blk.0.ffn_norm.weight"),
            "model.layers.0.post_attention_layernorm.weight"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_mlp() {
        assert_eq!(
            default_gguf_to_hf_name("blk.0.ffn_gate.weight"),
            "model.layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            default_gguf_to_hf_name("blk.0.ffn_up.weight"),
            "model.layers.0.mlp.up_proj.weight"
        );
        assert_eq!(
            default_gguf_to_hf_name("blk.0.ffn_down.weight"),
            "model.layers.0.mlp.down_proj.weight"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_deepseek_mla() {
        // MLA attention
        assert_eq!(
            default_gguf_to_hf_name("blk.0.attn_q_a.weight"),
            "model.layers.0.self_attn.q_a_proj.weight"
        );
        assert_eq!(
            default_gguf_to_hf_name("blk.0.attn_q_a_norm.weight"),
            "model.layers.0.self_attn.q_a_layernorm.weight"
        );
        assert_eq!(
            default_gguf_to_hf_name("blk.0.attn_q_b.weight"),
            "model.layers.0.self_attn.q_b_proj.weight"
        );
        assert_eq!(
            default_gguf_to_hf_name("blk.0.attn_kv_a_mqa.weight"),
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
        );
        assert_eq!(
            default_gguf_to_hf_name("blk.0.attn_kv_a_norm.weight"),
            "model.layers.0.self_attn.kv_a_layernorm.weight"
        );
        assert_eq!(
            default_gguf_to_hf_name("blk.0.attn_kv_b.weight"),
            "model.layers.0.self_attn.kv_b_proj.weight"
        );
        // DeepSeek uses attn_o (not attn_output)
        assert_eq!(
            default_gguf_to_hf_name("blk.0.attn_o.weight"),
            "model.layers.0.self_attn.o_proj.weight"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_deepseek_moe() {
        // Router gate
        assert_eq!(
            default_gguf_to_hf_name("blk.5.ffn_gate_inp.weight"),
            "model.layers.5.mlp.gate.weight"
        );
        // Fused expert weights
        assert_eq!(
            default_gguf_to_hf_name("blk.5.ffn_gate_exps.weight"),
            "model.layers.5.mlp.experts.fused_gate_exps.weight"
        );
        assert_eq!(
            default_gguf_to_hf_name("blk.5.ffn_up_exps.weight"),
            "model.layers.5.mlp.experts.fused_up_exps.weight"
        );
        assert_eq!(
            default_gguf_to_hf_name("blk.5.ffn_down_exps.weight"),
            "model.layers.5.mlp.experts.fused_down_exps.weight"
        );
        // Shared experts
        assert_eq!(
            default_gguf_to_hf_name("blk.5.ffn_gate_shexp.weight"),
            "model.layers.5.mlp.shared_experts.gate_proj.weight"
        );
        assert_eq!(
            default_gguf_to_hf_name("blk.5.ffn_up_shexp.weight"),
            "model.layers.5.mlp.shared_experts.up_proj.weight"
        );
        assert_eq!(
            default_gguf_to_hf_name("blk.5.ffn_down_shexp.weight"),
            "model.layers.5.mlp.shared_experts.down_proj.weight"
        );
        // Score correction bias
        assert_eq!(
            default_gguf_to_hf_name("blk.5.exp_probs_b.bias"),
            "model.layers.5.mlp.gate.e_score_correction_bias"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_unknown_passthrough() {
        assert_eq!(
            default_gguf_to_hf_name("some.unknown.tensor"),
            "some.unknown.tensor"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_unknown_layer_suffix() {
        assert_eq!(
            default_gguf_to_hf_name("blk.0.some_unknown.weight"),
            "model.layers.0.some_unknown.weight"
        );
    }

    // --- mmproj name mapping tests ---

    #[test]
    fn test_mmproj_projector_names() {
        assert_eq!(
            mmproj_gguf_to_hf_name("mm.0.weight"),
            "multi_modal_projector.mm_input_projection_weight"
        );
        assert_eq!(
            mmproj_gguf_to_hf_name("mm.model_norm.weight"),
            "multi_modal_projector.mm_soft_emb_norm.weight"
        );
    }

    #[test]
    fn test_mmproj_vision_global_names() {
        assert_eq!(
            mmproj_gguf_to_hf_name("v.patch_embd.weight"),
            "vision_tower.vision_model.embeddings.patch_embedding.weight"
        );
        assert_eq!(
            mmproj_gguf_to_hf_name("v.position_embd.weight"),
            "vision_tower.vision_model.embeddings.position_embedding.weight"
        );
        assert_eq!(
            mmproj_gguf_to_hf_name("v.post_ln.weight"),
            "vision_tower.vision_model.post_layernorm.weight"
        );
    }

    #[test]
    fn test_mmproj_vision_layer_names() {
        assert_eq!(
            mmproj_gguf_to_hf_name("v.blk.0.attn_q.weight"),
            "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            mmproj_gguf_to_hf_name("v.blk.5.attn_output.weight"),
            "vision_tower.vision_model.encoder.layers.5.self_attn.out_proj.weight"
        );
        assert_eq!(
            mmproj_gguf_to_hf_name("v.blk.0.ln1.weight"),
            "vision_tower.vision_model.encoder.layers.0.layer_norm1.weight"
        );
        assert_eq!(
            mmproj_gguf_to_hf_name("v.blk.0.ln2.weight"),
            "vision_tower.vision_model.encoder.layers.0.layer_norm2.weight"
        );
        assert_eq!(
            mmproj_gguf_to_hf_name("v.blk.0.ffn_up.weight"),
            "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight"
        );
        assert_eq!(
            mmproj_gguf_to_hf_name("v.blk.0.ffn_down.weight"),
            "vision_tower.vision_model.encoder.layers.0.mlp.fc2.weight"
        );
    }

    #[test]
    fn test_mmproj_unknown_passthrough() {
        assert_eq!(
            mmproj_gguf_to_hf_name("some.random.tensor"),
            "some.random.tensor"
        );
    }

    #[test]
    fn test_detect_mmproj_no_sibling() {
        let dir = tempfile::tempdir().unwrap();
        let main = dir.path().join("model.gguf");
        std::fs::write(&main, b"dummy").unwrap();
        assert!(detect_mmproj_gguf(&main).is_none());
    }

    #[test]
    fn test_detect_mmproj_with_sibling() {
        let dir = tempfile::tempdir().unwrap();
        let main = dir.path().join("model.gguf");
        let mmproj = dir.path().join("mmproj-model-f16.gguf");
        std::fs::write(&main, b"dummy").unwrap();
        std::fs::write(&mmproj, b"dummy").unwrap();
        let found = detect_mmproj_gguf(&main);
        assert!(found.is_some());
        assert_eq!(found.unwrap(), mmproj);
    }

    /// Synthetic-vocab unit test for `generate_merges`. Mirrors the
    /// shape of the reference Python algorithm (vocab dict iteration
    /// order = vocab order) so the assertion is exact, not "looks
    /// right". The vocab encodes a mini SentencePiece corpus where
    /// `▁the` decomposes via two valid splits and the algorithm has
    /// to keep BOTH plus pre-sort the per-piece local list by
    /// constituent rank.
    #[test]
    fn generate_merges_matches_hf_reference() {
        // Vocab order matches HF iteration order: byte-level chars
        // and unk first, then progressively longer multi-char pieces.
        let vocab: Vec<String> = vec![
            "<unk>".into(), // 0
            "▁".into(),     // 1
            "t".into(),     // 2
            "h".into(),     // 3
            "e".into(),     // 4
            "i".into(),     // 5
            "s".into(),     // 6
            "th".into(),    // 7  → ('t','h')
            "is".into(),    // 8  → ('i','s')
            "▁t".into(),    // 9  → ('▁','t')
            "▁i".into(),    // 10 → ('▁','i')
            "he".into(),    // 11 → ('h','e')
            "the".into(),   // 12 → ('th','e') | ('t','he')
            "▁is".into(),   // 13 → ('▁','is') | ('▁i','s')
            "▁the".into(), // 14 → ('▁','the') | ('▁t','he') | ('▁th','e') ; only ('▁','the') and ('▁t','he') exist as both-in-vocab
        ];
        // Score = -id approximates SP convention (more frequent = higher score).
        let scores: Vec<f32> = (0..vocab.len()).map(|i| -(i as f32)).collect();

        let merges = generate_merges(&vocab, &scores);

        // Every emitted merge must reconstruct a token in the vocab.
        let token_set: std::collections::HashSet<&str> = vocab.iter().map(|s| s.as_str()).collect();
        for (l, r) in &merges {
            let combined = format!("{l}{r}");
            assert!(
                token_set.contains(combined.as_str()),
                "merge ({l:?}, {r:?}) → {combined:?} is not a vocab entry"
            );
        }

        // Every compound vocab entry must have at least one merge
        // that produces it (otherwise BPE can't tokenize it).
        let merge_set: std::collections::HashSet<String> =
            merges.iter().map(|(l, r)| format!("{l}{r}")).collect();
        for (i, tok) in vocab.iter().enumerate() {
            if tok.chars().count() >= 2 && tok != "<unk>" {
                assert!(
                    merge_set.contains(tok),
                    "compound token #{i} {tok:?} has no producing merge"
                );
            }
        }

        // Sort order: outer sort is `(score, len_l, len_r)` DESC.
        // Highest-score parent is the first vocab entry that's
        // compound — `th` (id 7, score -7). Its single-decomposition
        // merge `('t','h')` MUST appear before any merge produced by
        // a lower-score (higher-id) parent.
        let th_pos = merges
            .iter()
            .position(|(l, r)| l == "t" && r == "h")
            .expect("('t','h') merge missing");
        // `▁the` is the lowest-score (highest-id) compound; its
        // merges should land at or near the end of the file.
        let last_the_merge = merges
            .iter()
            .rposition(|(l, r)| {
                let combined = format!("{l}{r}");
                combined == "▁the"
            })
            .expect("merge producing `▁the` missing");
        assert!(
            th_pos < last_the_merge,
            "lowest-level merge ('t','h') at {th_pos} should precede ▁the merge at {last_the_merge}"
        );

        // `▁the` has TWO valid decompositions in this vocab —
        // `('▁','the')` and `('▁t','he')`. Both must be emitted so
        // the BPE cascade can produce `▁the` regardless of which
        // intermediate forms first.
        let prods: Vec<&(String, String)> = merges
            .iter()
            .filter(|(l, r)| format!("{l}{r}") == "▁the")
            .collect();
        assert_eq!(
            prods.len(),
            2,
            "expected 2 merges producing `▁the`, got {prods:?}"
        );
    }

    /// Integration test against real GGUFs in the local HF cache.
    /// Asserts that `gguf_tokenizer` produces the SAME ids as the HF
    /// reference for a fixed prompt across every supported tokenizer
    /// shape: BPE (`gpt2`-style — Llama-3, Qwen) and SentencePiece
    /// (`llama`-style — Mistral, Gemma2, Gemma3). Reference ids are
    /// hard-coded from `transformers.AutoTokenizer.encode(prompt,
    /// add_special_tokens=False)` for each model.
    ///
    /// `#[ignore]` because it depends on the local HF cache; run
    /// manually with `cargo test -p scratchy-quantizations -- --ignored`.
    #[test]
    #[ignore]
    fn gguf_tokenizer_matches_hf_reference() {
        let prompt = "What is the capital of France?";
        let home = std::env::var("HOME").unwrap();

        // (label, gguf-path-suffix-under-HF-cache, expected-ids).
        // Reference ids come from
        // `transformers.AutoTokenizer.from_pretrained(<repo>).encode(prompt, add_special_tokens=False)`
        // run against the matching HF safetensors repo.
        let cases: &[(&str, &str, &[u32])] = &[
            // BPE (`tokenizer.ggml.model = "gpt2"`).
            (
                "Llama-3.2-1B (BPE, llama-bpe pre)",
                "models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/b69aef112e9f895e6f98d7ae0949f72ff09aa401/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                &[3923, 374, 279, 6864, 315, 9822, 30],
            ),
            (
                "Llama-3.2-3B (BPE, llama-bpe pre)",
                "models--unsloth--Llama-3.2-3B-Instruct-GGUF/snapshots/e7d0997e49c9cb00d88b4c1a6a16aa894b0bbc31/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                &[3923, 374, 279, 6864, 315, 9822, 30],
            ),
            (
                "Qwen2.5-0.5B (BPE, qwen2 pre)",
                "models--bartowski--Qwen2.5-0.5B-Instruct-GGUF/snapshots/41ba88dbac95fed2528c92514c131d73eb5a174b/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
                &[3838, 374, 279, 6722, 315, 9625, 30],
            ),
            (
                "Qwen3-0.6B (BPE, qwen2 pre)",
                "models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q4_K_M.gguf",
                &[3838, 374, 279, 6722, 315, 9625, 30],
            ),
            (
                "Granite-3.1-2B (BPE, refact pre)",
                "models--bartowski--granite-3.1-2b-instruct-GGUF/snapshots/e47b8b46c04cede00f9e19d5a846551b14b2efce/granite-3.1-2b-instruct-Q4_K_M.gguf",
                &[8197, 438, 322, 18926, 432, 45600, 49],
            ),
            // SentencePiece with Metaspace-prepend (`tokenizer.ggml.model = "llama"`,
            // non-Gemma arch).
            (
                "Mistral-7B-v0.3 (SP, Metaspace prepend)",
                "models--bartowski--Mistral-7B-Instruct-v0.3-GGUF/snapshots/61fd4167fff3ab01ee1cfe0da183fa27a944db48/Mistral-7B-Instruct-v0.3-IQ2_S.gguf",
                &[2592, 1117, 1040, 6333, 1070, 5611, 29572],
            ),
            (
                "Phi-3.5-mini (SP, Metaspace prepend)",
                "models--bartowski--Phi-3.5-mini-instruct-GGUF/snapshots/6d70da17e749a471ccb62ade694486011a75cda3/Phi-3.5-mini-instruct-Q4_K_M.gguf",
                &[1724, 338, 278, 7483, 310, 3444, 29973],
            ),
            // SentencePiece without prepend (`tokenizer.ggml.model = "llama"`,
            // gemma* arch).
            (
                "Gemma-2-2B (SP, Replace-only)",
                "models--bartowski--gemma-2-2b-it-GGUF/snapshots/855f67caed130e1befc571b52bd181be2e858883/gemma-2-2b-it-Q4_K_M.gguf",
                &[1841, 603, 573, 6037, 576, 6081, 235336],
            ),
            (
                "Gemma-3-1B (SP, Replace-only)",
                "models--unsloth--gemma-3-1b-it-GGUF/snapshots/f0b45be0aac41bd6a100a4b5734cad5f67255bfb/gemma-3-1b-it-Q4_K_M.gguf",
                &[3689, 563, 506, 5279, 529, 7001, 236881],
            ),
        ];

        let mut ran = 0;
        for (label, suffix, expected) in cases {
            let path = format!("{home}/.cache/huggingface/hub/{suffix}");
            if !std::path::Path::new(&path).exists() {
                eprintln!("[skip] {label}: {path} not found");
                continue;
            }
            let gguf = GgufFile::open(&path).expect("open gguf");
            let tok = gguf_tokenizer(&gguf)
                .expect("build tokenizer")
                .expect("tokenizer present");
            let enc = tok.encode(prompt, true).expect("encode");
            let got: Vec<u32> = enc.get_ids().to_vec();
            assert_eq!(&got, expected, "ids for {label} ({path})");
            ran += 1;
        }
        assert!(
            ran > 0,
            "no GGUFs found in HF cache — populate at least one of the test fixtures"
        );
    }

    /// E2E inference smoke. Shells out to a built `scr` binary,
    /// runs greedy "Paris" against each cached GGUF, asserts the
    /// expected substring lands in stdout. Locks down the GGUF load
    /// + dispatch + forward pipeline end-to-end on real fixtures.
    ///
    /// Skips fixtures that aren't in the local HF cache, and the
    /// whole test if `scr` isn't built. Set `VLLM_BIN=...` to point
    /// at a non-default binary path; default is the workspace's
    /// `target/release/scr`.
    ///
    /// `#[ignore]` because it depends on the local HF cache + a
    /// built CUDA binary + a GPU. Run manually:
    ///
    /// ```text
    /// cargo build --manifest-path vllm-rs/Cargo.toml -p scratchy-cli \
    ///     --features cuda --release
    /// cargo test --manifest-path vllm-rs/Cargo.toml -p scratchy-quantizations \
    ///     --release -- --ignored gguf_inference_smoke
    /// ```
    ///
    /// **Coverage gaps marked here, not skipped silently**:
    /// V2-Lite + Moonlight (DeepSeek family GGUFs) are listed but
    /// known-broken at `DeepSeekV2MoELayer::load` — the per-expert
    /// safetensors-shaped weight names don't exist in GGUF land
    /// (fused 3D `mlp.experts.fused_*_exps.weight`). Add them back
    /// once the GGUF MoE loader audit (handoff step 3) lands.
    #[test]
    #[ignore]
    fn gguf_inference_smoke() {
        let prompt = "What is the capital of France?";
        let expected = "Paris";
        let home = std::env::var("HOME").expect("HOME");

        // CARGO_MANIFEST_DIR points at vllm-rs/crates/quantization;
        // workspace target lives two up at vllm-rs/target.
        let bin = std::env::var("VLLM_BIN")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .and_then(|p| p.parent())
                    .expect("crate dir has two ancestors")
                    .join("target/release/scr")
            });
        if !bin.exists() {
            eprintln!(
                "[skip] scr binary not built at {} — \
                 run `cargo build --manifest-path vllm-rs/Cargo.toml \
                 -p scratchy-cli --features cuda --release` first",
                bin.display()
            );
            return;
        }

        // (label, gguf-path-suffix-under-HF-cache, optional
        // chat-template fixture filename relative to
        // `crates/quantization/test_fixtures/`). The chat-template
        // override is only needed for GGUFs that ship without
        // `tokenizer.chat_template` in metadata — e.g. mmnga's
        // Moonlight conversion.
        let cases: &[(&str, &str, Option<&str>)] = &[
            (
                "Llama-3.2-1B (BPE)",
                "models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/b69aef112e9f895e6f98d7ae0949f72ff09aa401/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                None,
            ),
            (
                "Mistral-7B-v0.3 (SP)",
                "models--bartowski--Mistral-7B-Instruct-v0.3-GGUF/snapshots/61fd4167fff3ab01ee1cfe0da183fa27a944db48/Mistral-7B-Instruct-v0.3-IQ2_S.gguf",
                None,
            ),
            (
                "DeepSeek-V2-Lite (MoE)",
                "models--mradermacher--DeepSeek-V2-Lite-GGUF/snapshots/0f37fdf276e8094747457f0ae4d40f2e8d2521f9/DeepSeek-V2-Lite.Q4_K_M.gguf",
                None,
            ),
            (
                "Phi-3.5-mini (fused qkv + gate_up)",
                "models--bartowski--Phi-3.5-mini-instruct-GGUF/snapshots/6d70da17e749a471ccb62ade694486011a75cda3/Phi-3.5-mini-instruct-Q4_K_M.gguf",
                None,
            ),
            (
                "Moonlight-16B-A3B (V3-flat)",
                "models--mmnga--Moonlight-16B-A3B-Instruct-gguf/snapshots/eb4728b376af0f3e168dc96d23cd21818c5738f6/Moonlight-16B-A3B-Instruct-Q4_K_M.gguf",
                Some("moonshotai_moonlight_chat_template.jinja"),
            ),
        ];

        let fixtures_dir =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_fixtures");
        let mut ran = 0;
        for (i, (label, suffix, chat_template_fixture)) in cases.iter().enumerate() {
            // Brief inter-fixture pause: each subprocess exits promptly
            // after `scr chat` produces coherent output, but the CUDA
            // driver can take a moment to fully reclaim the previous
            // process's GPU memory. Without this, the next fixture's
            // weight load can hit `CUDA_ERROR_OUT_OF_MEMORY` even when
            // total demand is well under VRAM. 2s is empirically enough
            // on L4; total inflation is bounded at ~10s for 5 fixtures.
            if i > 0 {
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
            let path = format!("{home}/.cache/huggingface/hub/{suffix}");
            if !std::path::Path::new(&path).exists() {
                eprintln!("[skip] {label}: {path} not found");
                continue;
            }
            // `timeout 180` — wraps `scr chat`. Coherent output
            // exits cleanly; garbage loops forever. 180s for the larger MoE
            // fixtures (Moonlight), 120s would suffice for the rest
            // but uniform timeout keeps the case loop simple.
            let chat_template_path: Option<std::path::PathBuf> =
                chat_template_fixture.map(|name| fixtures_dir.join(name));
            let mut argv: Vec<String> = vec![
                "180".into(),
                bin.to_str().expect("vllm bin path is utf-8").into(),
                "chat".into(),
                "--model".into(),
                path.clone(),
                "--max-tokens".into(),
                "30".into(),
                "--temperature".into(),
                "0".into(),
                "--prompt".into(),
                prompt.into(),
            ];
            if let Some(ref ct_path) = chat_template_path {
                argv.push("--chat-template".into());
                argv.push(ct_path.to_string_lossy().into_owned());
            }
            let output = std::process::Command::new("timeout")
                .args(&argv)
                .output()
                .unwrap_or_else(|e| panic!("spawn `timeout scr chat`: {e}"));
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            // Greedy "Paris" check — appears in `scr chat`'s
            // generated text on stdout. Falls back to combined
            // stdout+stderr in case future versions change which
            // stream the chat output goes to.
            let combined = format!("{stdout}\n{stderr}");
            assert!(
                combined.contains(expected),
                "{label}: expected `{expected}` in vllm output\n\
                 stdout:\n{stdout}\n\
                 stderr:\n{stderr}"
            );
            ran += 1;
        }
        assert!(
            ran > 0,
            "no GGUF inference fixtures found — populate at least one"
        );
    }
}
