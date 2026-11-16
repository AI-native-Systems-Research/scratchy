// SPDX-License-Identifier: Apache-2.0
//! Each token's piece of text, baked beside the tokenizer: what a serving tokenizer decodes
//! through without allocating, read in place from the binary (#1269).
//!
//! `tokenizers`' decode builds a `String` per token, a `Vec` of them, the decoder's own buffers and
//! a joined result on every call, and a streaming detokenizer decodes twice per token. A byte-level
//! decoder (GPT-2's, Llama 3's, Qwen's, Granite's) and SentencePiece's byte fallback (Llama 2's,
//! Mistral's, Gemma's) turn each token into a piece that depends on that token alone and then join
//! the pieces, so [`bake`](super::bake) works the pieces out once, at build time, and a decode
//! walks them into the caller's buffers. Any other decoder has no table.
//!
//! The layout, in little-endian `u32` words and then bytes: `MAGIC`; the join (0 byte-level,
//! 1 byte fallback); the leading spaces a byte-fallback decode strips; the id count `n`; then for
//! each id the end of its piece in the text, its flags (`SPECIAL`, `BYTE`, `ABSENT`) in the top
//! bits; then the text.

use std::collections::HashMap;

use tokenizers::decoders::DecoderWrapper;
use tokenizers::normalizers::Replace;

const MAGIC: u32 = u32::from_le_bytes(*b"TKP1");
/// The header's words: magic, join, strip, id count.
const HEADER: usize = 4;
/// A token `skip_special_tokens` drops.
const SPECIAL: u32 = 1 << 31;
/// A byte-fallback `<0xNN>` token: its piece is its one byte.
const BYTE: u32 = 1 << 30;
/// An id that names no token, which `tokenizers`' decode drops.
const ABSENT: u32 = 1 << 29;
/// The bits of an entry that are its piece's end.
const END: u32 = ABSENT - 1;

/// How a decoder joins its tokens' pieces into text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Join {
    /// Byte-level BPE (`ByteLevel`): every piece is bytes, and the text is the bytes of all of
    /// them read as UTF-8, each invalid run one U+FFFD (`String::from_utf8_lossy`).
    ByteLevel,
    /// SentencePiece byte fallback, the chain `Replace("▁", " ")`, `ByteFallback`, `Fuse`, then a
    /// `Strip` of up to `strip_start` leading spaces: a run of byte tokens is its UTF-8 text, or one
    /// U+FFFD per byte when it is not valid UTF-8, and any other token is its own text.
    ByteFallback { strip_start: u32 },
}

impl Join {
    fn of(decoder: &DecoderWrapper) -> Option<Self> {
        match decoder {
            DecoderWrapper::ByteLevel(_) => Some(Self::ByteLevel),
            DecoderWrapper::Sequence(seq) => {
                let [
                    DecoderWrapper::Replace(replace),
                    DecoderWrapper::ByteFallback(_),
                    DecoderWrapper::Fuse(_),
                    strip @ ..,
                ] = seq.get_decoders()
                else {
                    return None;
                };
                if *replace != Replace::new("▁", " ").ok()? {
                    return None;
                }
                let strip_start = match strip {
                    [] => 0,
                    [DecoderWrapper::Strip(s)] if s.content == ' ' && s.stop == 0 => {
                        u32::try_from(s.start).ok()?
                    }
                    _ => return None,
                };
                Some(Self::ByteFallback { strip_start })
            }
            _ => None,
        }
    }
}

/// The piece table of the ids `0..ids`, in this module's layout, when `decoder` joins pieces end to
/// end: `token` names each id's token as `tokenizers`' decode reads it, and `special` says whether
/// that token is one `skip_special_tokens` drops.
pub fn build(
    decoder: Option<&DecoderWrapper>,
    ids: u32,
    token: impl Fn(u32) -> Option<String>,
    special: impl Fn(&str) -> bool,
) -> Option<Vec<u8>> {
    let join = Join::of(decoder?)?;
    let byte_of = byte_level_bytes();
    let mut ends: Vec<u32> = Vec::with_capacity(ids as usize);
    let mut text: Vec<u8> = Vec::new();
    for id in 0..ids {
        let Some(token) = token(id) else {
            ends.push(end(&text)? | ABSENT);
            continue;
        };
        let mut flags = if special(&token) { SPECIAL } else { 0 };
        match join {
            Join::ByteLevel => {
                let start = text.len();
                // A character outside the byte map leaves the token's own UTF-8, as `ByteLevel`'s
                // decode does.
                let mapped =
                    (token.chars()).try_for_each(|c| byte_of.get(&c).map(|&b| text.push(b)));
                if mapped.is_none() {
                    text.truncate(start);
                    text.extend_from_slice(token.as_bytes());
                }
            }
            Join::ByteFallback { .. } => {
                let piece = token.replace('▁', " ");
                match fallback_byte(&piece) {
                    Some(byte) => {
                        text.push(byte);
                        flags |= BYTE;
                    }
                    None => text.extend_from_slice(piece.as_bytes()),
                }
            }
        }
        ends.push(end(&text)? | flags);
    }
    let (join, strip) = match join {
        Join::ByteLevel => (0, 0),
        Join::ByteFallback { strip_start } => (1, strip_start),
    };
    let header = [MAGIC, join, strip, ids];
    let mut blob: Vec<u8> = (header.iter().chain(&ends))
        .flat_map(|w| w.to_le_bytes())
        .collect();
    blob.extend_from_slice(&text);
    Some(blob)
}

/// The text's length as a piece's end, `None` past what an entry holds.
fn end(text: &[u8]) -> Option<u32> {
    u32::try_from(text.len()).ok().filter(|&e| e <= END)
}

/// A piece table ([`build`]'s layout), read in place.
#[derive(Clone, Copy, Debug)]
pub struct Pieces<'a> {
    join: Join,
    ends: &'a [u8],
    text: &'a [u8],
}

impl<'a> Pieces<'a> {
    /// The table `blob` holds. Reads only its header: a piece whose span is not in the text reads
    /// as absent.
    pub fn parse(blob: &'a [u8]) -> Result<Self, String> {
        let word = |i: usize| {
            let bytes = blob.get(4 * i..4 * i + 4)?;
            Some(u32::from_le_bytes(<[u8; 4]>::try_from(bytes).ok()?))
        };
        let header = (0..HEADER).map(word).collect::<Option<Vec<u32>>>();
        let Some([magic, join, strip, ids]) = header.as_deref() else {
            return Err("a piece table shorter than its header".into());
        };
        if *magic != MAGIC {
            return Err("not a piece table".into());
        }
        let join = match join {
            0 => Join::ByteLevel,
            1 => Join::ByteFallback {
                strip_start: *strip,
            },
            _ => return Err(format!("a piece table of join {join}")),
        };
        let body = 4 * (HEADER + *ids as usize);
        let (Some(ends), Some(text)) = (blob.get(4 * HEADER..body), blob.get(body..)) else {
            return Err("a piece table shorter than its id count".into());
        };
        Ok(Self { join, ends, text })
    }

    /// Token `id`'s piece and its entry's flags; `None` when the id names no token.
    fn piece(&self, id: u32) -> Option<(&'a [u8], u32)> {
        let entry = |i: usize| {
            let bytes = self.ends.get(4 * i..4 * i + 4)?;
            Some(u32::from_le_bytes(<[u8; 4]>::try_from(bytes).ok()?))
        };
        let i = id as usize;
        let e = entry(i)?;
        if e & ABSENT != 0 {
            return None;
        }
        let start = match i {
            0 => 0,
            _ => entry(i - 1)? & END,
        };
        Some((self.text.get(start as usize..(e & END) as usize)?, e))
    }

    /// `ids` decoded as `tokenizers`' decode would, into `out` (cleared first), `run` the scratch
    /// a byte run gathers in: allocates only to grow `run` or `out` past what they hold.
    pub fn decode_into(
        &self,
        ids: &[u32],
        skip_special: bool,
        run: &mut Vec<u8>,
        out: &mut String,
    ) {
        out.clear();
        run.clear();
        let kept = (ids.iter())
            .filter_map(|&id| self.piece(id))
            .filter(|&(_, flags)| !(skip_special && flags & SPECIAL != 0));
        match self.join {
            Join::ByteLevel => {
                for (bytes, _) in kept {
                    run.extend_from_slice(bytes);
                }
                lossy_into(run, out);
            }
            Join::ByteFallback { strip_start } => {
                for (bytes, flags) in kept {
                    if flags & BYTE != 0 {
                        run.extend_from_slice(bytes);
                    } else {
                        flush(run, out);
                        // Always UTF-8: the bake wrote a `String`.
                        lossy_into(bytes, out);
                    }
                }
                flush(run, out);
                let lead = (out.chars().take(strip_start as usize))
                    .take_while(|&c| c == ' ')
                    .count();
                out.drain(..lead);
            }
        }
    }
}

/// `bytes` as `String::from_utf8_lossy` reads them, pushed onto `out`.
fn lossy_into(bytes: &[u8], out: &mut String) {
    for chunk in bytes.utf8_chunks() {
        out.push_str(chunk.valid());
        if !chunk.invalid().is_empty() {
            out.push(char::REPLACEMENT_CHARACTER);
        }
    }
}

/// A byte run as `ByteFallback` emits it: its text when it is valid UTF-8, else one U+FFFD per
/// byte.
fn flush(run: &mut Vec<u8>, out: &mut String) {
    match std::str::from_utf8(run) {
        Ok(s) => out.push_str(s),
        Err(_) => out.extend(std::iter::repeat_n(char::REPLACEMENT_CHARACTER, run.len())),
    }
    run.clear();
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
