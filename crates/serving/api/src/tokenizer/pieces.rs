// SPDX-License-Identifier: Apache-2.0
//! Each token's piece of text, for the decoders that join pieces end to end: what
//! [`super::Tokenizer::decode_into`] decodes from without allocating.
//!
//! `tokenizers`' decode builds a `String` per token, a `Vec` of them, the decoder's own buffers and
//! a joined result on every call, and the streaming detokenizer decodes twice per token (#1269). A
//! byte-level decoder (GPT-2's, Llama 3's, Qwen's, Granite's) and SentencePiece's byte fallback
//! (Llama 2's, Mistral's, Gemma's) turn each token into a piece that depends on that token alone and
//! then join the pieces, so the pieces are worked out once, at load, and a decode walks them into
//! the caller's buffers. Any other decoder has no table, and decodes through `tokenizers`.

use std::collections::HashMap;
use std::ops::Range;

use tokenizers::decoders::DecoderWrapper;
use tokenizers::normalizers::Replace;

/// Where one token's piece sits in its table's text, and whether `skip_special_tokens` drops it.
#[derive(Clone, Debug)]
struct Span {
    at: Range<u32>,
    special: bool,
}

impl Span {
    fn of(at: Range<usize>, special: bool) -> Self {
        let end = |i: usize| u32::try_from(i).expect("a vocabulary's text is under 4 GiB");
        Self {
            at: end(at.start)..end(at.end),
            special,
        }
    }

    fn range(&self) -> Range<usize> {
        self.at.start as usize..self.at.end as usize
    }
}

/// A byte-fallback token's piece.
#[derive(Clone, Debug)]
enum Fallback {
    /// Text, `▁` read as a space.
    Text(Span),
    /// A `<0xNN>` token: one byte, joined with the bytes of the byte tokens beside it.
    Byte { byte: u8, special: bool },
}

impl Fallback {
    fn special(&self) -> bool {
        match self {
            Self::Text(span) => span.special,
            Self::Byte { special, .. } => *special,
        }
    }
}

/// Every token's piece, and how the decoder joins them.
pub(super) struct Pieces(Table);

/// The pieces by id (`None` for an id that names no token, which `tokenizers` drops), one table
/// per way of joining them.
enum Table {
    /// Byte-level BPE (`ByteLevel`): every token is bytes, and the text is the bytes of all of them
    /// read as UTF-8, each invalid run one U+FFFD (`String::from_utf8_lossy`).
    ByteLevel {
        bytes: Vec<u8>,
        pieces: Vec<Option<Span>>,
    },
    /// SentencePiece byte fallback, the chain `Replace("▁", " ")`, `ByteFallback`, `Fuse`, then a
    /// `Strip` of up to `strip_start` leading spaces: a run of byte tokens is its UTF-8 text, or one
    /// U+FFFD per byte when it is not valid UTF-8, and any other token is its own text.
    ByteFallback {
        text: String,
        pieces: Vec<Option<Fallback>>,
        strip_start: usize,
    },
}

impl Pieces {
    /// The pieces of the ids `0..ids`, when `decoder` joins pieces end to end: `token` names each
    /// id's token as `tokenizers`' decode reads it, and `special` says whether that token is one
    /// `skip_special_tokens` drops.
    pub(super) fn of(
        decoder: Option<&DecoderWrapper>,
        ids: u32,
        token: impl Fn(u32) -> Option<String>,
        special: impl Fn(&str) -> bool,
    ) -> Option<Self> {
        match decoder? {
            DecoderWrapper::ByteLevel(_) => {
                let byte_of = byte_level_bytes();
                let mut bytes = Vec::new();
                let pieces = (0..ids)
                    .map(|id| {
                        let token = token(id)?;
                        let start = bytes.len();
                        // A character outside the byte map leaves the token's own UTF-8, as
                        // `ByteLevel`'s decode does.
                        let mapped = (token.chars())
                            .try_for_each(|c| byte_of.get(&c).map(|&b| bytes.push(b)));
                        if mapped.is_none() {
                            bytes.truncate(start);
                            bytes.extend_from_slice(token.as_bytes());
                        }
                        Some(Span::of(start..bytes.len(), special(&token)))
                    })
                    .collect();
                Some(Self(Table::ByteLevel { bytes, pieces }))
            }
            DecoderWrapper::Sequence(seq) => {
                let strip_start = byte_fallback_strip(seq.get_decoders())?;
                let mut text = String::new();
                let pieces = (0..ids)
                    .map(|id| {
                        let token = token(id)?;
                        let special = special(&token);
                        let piece = token.replace('▁', " ");
                        Some(match fallback_byte(&piece) {
                            Some(byte) => Fallback::Byte { byte, special },
                            None => {
                                let start = text.len();
                                text.push_str(&piece);
                                Fallback::Text(Span::of(start..text.len(), special))
                            }
                        })
                    })
                    .collect();
                Some(Self(Table::ByteFallback {
                    text,
                    pieces,
                    strip_start,
                }))
            }
            _ => None,
        }
    }

    /// `ids` decoded as `tokenizers`' decode would, into `out` (cleared first), `run` the scratch a
    /// byte run gathers in: allocates only to grow `run` or `out` past what they hold.
    pub(super) fn decode_into(
        &self,
        ids: &[u32],
        skip_special: bool,
        run: &mut Vec<u8>,
        out: &mut String,
    ) {
        out.clear();
        run.clear();
        match &self.0 {
            Table::ByteLevel { bytes, pieces } => {
                for span in kept(pieces, ids, skip_special, |s| s.special) {
                    run.extend_from_slice(&bytes[span.range()]);
                }
                for chunk in run.utf8_chunks() {
                    out.push_str(chunk.valid());
                    if !chunk.invalid().is_empty() {
                        out.push(char::REPLACEMENT_CHARACTER);
                    }
                }
            }
            Table::ByteFallback {
                text,
                pieces,
                strip_start,
            } => {
                for piece in kept(pieces, ids, skip_special, Fallback::special) {
                    match piece {
                        Fallback::Byte { byte, .. } => run.push(*byte),
                        Fallback::Text(span) => {
                            flush(run, out);
                            out.push_str(&text[span.range()]);
                        }
                    }
                }
                flush(run, out);
                let lead = (out.chars().take(*strip_start))
                    .take_while(|&c| c == ' ')
                    .count();
                out.drain(..lead);
            }
        }
    }
}

/// The pieces of `ids` that `tokenizers`' decode keeps: each id that names a token, less the special
/// ones when `skip_special`.
fn kept<'a, P>(
    pieces: &'a [Option<P>],
    ids: &'a [u32],
    skip_special: bool,
    special: impl Fn(&P) -> bool + 'a,
) -> impl Iterator<Item = &'a P> + 'a {
    (ids.iter())
        .filter_map(|&id| pieces.get(id as usize)?.as_ref())
        .filter(move |p| !(skip_special && special(p)))
}

/// A byte run as `ByteFallback` emits it: its text when it is valid UTF-8, else one U+FFFD per byte.
fn flush(run: &mut Vec<u8>, out: &mut String) {
    match std::str::from_utf8(run) {
        Ok(s) => out.push_str(s),
        Err(_) => out.extend(std::iter::repeat_n(char::REPLACEMENT_CHARACTER, run.len())),
    }
    run.clear();
}

/// `Some(strip_start)` when `decoders` is SentencePiece's byte-fallback chain: `Replace("▁", " ")`,
/// `ByteFallback`, `Fuse`, and an optional `Strip` of leading spaces only.
fn byte_fallback_strip(decoders: &[DecoderWrapper]) -> Option<usize> {
    let [
        DecoderWrapper::Replace(replace),
        DecoderWrapper::ByteFallback(_),
        DecoderWrapper::Fuse(_),
        strip @ ..,
    ] = decoders
    else {
        return None;
    };
    if *replace != Replace::new("▁", " ").ok()? {
        return None;
    }
    match strip {
        [] => Some(0),
        [DecoderWrapper::Strip(s)] if s.content == ' ' && s.stop == 0 => Some(s.start),
        _ => None,
    }
}

/// The byte a `<0xNN>` token stands for, by `ByteFallback`'s own test.
fn fallback_byte(token: &str) -> Option<u8> {
    if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
        u8::from_str_radix(token.get(3..5)?, 16).ok()
    } else {
        None
    }
}

/// GPT-2's byte-to-character map, inverted: the byte each character of a byte-level token stands
/// for. `tokenizers`' own (`pre_tokenizers::byte_level::bytes_char`) is private to it; this is its
/// body.
fn byte_level_bytes() -> HashMap<char, u8> {
    let mut bytes: Vec<u8> = (b'!'..=b'~')
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut chars: Vec<u32> = bytes.iter().map(|&b| u32::from(b)).collect();
    let mut n = 0;
    for b in 0..=u8::MAX {
        if !bytes.contains(&b) {
            bytes.push(b);
            chars.push(256 + n);
            n += 1;
        }
    }
    (bytes.into_iter().zip(chars))
        .filter_map(|(b, c)| Some((char::from_u32(c)?, b)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::{
        SENTENCEPIECE_WORDS, Tokenizer, make_test_sentencepiece_tokenizer,
        make_test_tokenizer_with_added,
    };

    /// Every window of ids decodes through the pieces exactly as through `tokenizers`: random
    /// windows of up to 12 ids, drawn from every id the tokenizer has and two it has not, kept and
    /// skipping special tokens. Byte runs that are valid UTF-8, cut short, and invalid all come up.
    fn decodes_as_tokenizers_does(tok: &Tokenizer, ids: u32) {
        assert!(tok.pieces.is_some(), "the decoder has a piece table");
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut draw = |n: u32| {
            state = (state.wrapping_mul(6_364_136_223_846_793_005))
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32 % n
        };
        let (mut run, mut out) = (Vec::new(), String::new());
        for _ in 0..4000 {
            let len = draw(13);
            let window: Vec<u32> = (0..len).map(|_| draw(ids + 2)).collect();
            for skip in [false, true] {
                let want = tok.decode(&window, skip).expect("tokenizers decodes");
                tok.decode_into(&window, skip, &mut run, &mut out)
                    .expect("the pieces decode");
                assert_eq!(out, want, "ids {window:?}, skip_special_tokens {skip}");
            }
        }
    }

    #[test]
    fn a_byte_level_window_decodes_as_tokenizers_does() {
        decodes_as_tokenizers_does(&make_test_tokenizer_with_added(), 259);
    }

    #[test]
    fn a_byte_fallback_window_decodes_as_tokenizers_does() {
        let ids = 258 + SENTENCEPIECE_WORDS.len() as u32;
        for strip in [true, false] {
            decodes_as_tokenizers_does(&make_test_sentencepiece_tokenizer(strip), ids);
        }
    }

    /// A decoder that does not join pieces end to end has no table, and decodes through
    /// `tokenizers` itself.
    #[test]
    fn another_decoder_has_no_table() {
        let json = serde_json::json!({
            "version": "1.0",
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
            "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
        });
        let tok = Tokenizer::from_hf_tokenizer(json.to_string().parse().unwrap());
        assert!(tok.pieces.is_none());
        let (mut run, mut out) = (Vec::new(), String::new());
        tok.decode_into(&[0, 1], false, &mut run, &mut out).unwrap();
        assert_eq!(out, tok.decode(&[0, 1], false).unwrap());
    }
}
