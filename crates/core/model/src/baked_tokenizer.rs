// SPDX-License-Identifier: Apache-2.0
//! Tokenizers compiled into the binary.
//!
//! Loading a `tokenizer.json` parses it and hashes its whole vocabulary and merge list on every
//! launch: ~300 ms for gemma-4's (262,144 tokens, 514,906 merges, 32 MB), longer than the model
//! load it runs beside. The build bakes a tokenizer's BPE model into tables instead — read in place
//! from the binary, so loading costs nothing — and keeps the rest of it (normalizer, pre-tokenizer,
//! added tokens, post-processor, decoder) as the few KB of `tokenizer.json` they are, which
//! `tokenizers` itself deserializes. Encoding and decoding are `tokenizers`' own code but for the
//! model; [`BakedBpe`] is its BPE (`models/bpe/{model,word}.rs`, 0.23) over the tables.
//!
//! A baked tokenizer is keyed by the digest the Hub cache names the file's blob by
//! ([`cached_digest`]): a checkpoint whose `tokenizer.json` has no baked twin loads it as before.
//! [`bake`] refuses a tokenizer it cannot reproduce, checked by encoding and decoding a corpus both
//! ways, so a baked tokenizer is the file's tokenizer or the build says why not.

use std::cell::RefCell;
use std::collections::{BinaryHeap, HashMap};
use std::path::{Path, PathBuf};

use serde::de::Error as _;
use serde::{Deserialize, Deserializer};
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::ModelWrapper;
use tokenizers::models::bpe::BpeTrainer;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::{Model, Token, TokenizerImpl};

/// A baked tokenizer, ready to encode: `tokenizers`' pipeline around a [`BakedBpe`].
pub type Baked = TokenizerImpl<
    BakedBpe,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

/// Bytes aligned for the tables' `u32` reads (`include_bytes!` promises alignment 1).
#[repr(C, align(8))]
pub struct Aligned<B: ?Sized>(pub B);

/// One tokenizer the build compiled in.
pub struct BakedTokenizer {
    /// The digest of the `tokenizer.json` it was baked from ([`cached_digest`]).
    pub digest: &'static str,
    /// Its BPE model, in [`bake`]'s layout.
    pub tables: &'static Aligned<[u8]>,
    /// The rest of its `tokenizer.json`, the model stanza replaced by `{"digest": …}`.
    pub pipeline: &'static str,
}

inventory::collect!(BakedTokenizer);

impl BakedTokenizer {
    /// The tokenizer baked from the `tokenizer.json` with this digest, if the build baked it.
    pub fn find(digest: &str) -> Option<&'static BakedTokenizer> {
        inventory::iter::<BakedTokenizer>().find(|t| t.digest == digest)
    }

    pub fn load(&'static self) -> Result<Baked, String> {
        load(self.digest, &self.tables.0, self.pipeline)
    }
}

/// The digest a Hub-cache snapshot's `tokenizer.json` is stored under: the file is a link to
/// `blobs/<sha256>` (or `<git sha-1>` for a small, git-tracked one), so the key costs a
/// `readlink`, not a hash. `None` for a file that is not such a link (a local model directory).
pub fn cached_digest(tokenizer_json: &Path) -> Option<String> {
    let target: PathBuf = std::fs::read_link(tokenizer_json).ok()?;
    let name = target.file_name()?.to_str()?;
    (matches!(name.len(), 40 | 64) && name.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| name.to_string())
}

// ---------------------------------------------------------------------------------------------
// The tables. Little-endian u32s, then the string pool:
//
//   header   [u32; 8]   MAGIC, ids, pool bytes, vocab slots, merges, merge slots, flags | unk, id bits
//   offsets  [u32; ids + 1]       token `i` is pool[offsets[i]..offsets[i + 1]]
//   bytes    [u32; 256]           byte `b`'s `<0xBB>` fallback token, or EMPTY
//   vocab    [u32; vocab slots]   open addressing on `str_hash`, at most 3/4 full: an id, or EMPTY
//   merged   [u32; merges]        the token merge `r` (by rank) makes
//   pairs    [[u32; 2]; merge slots]  open addressing on `pair_hash`, at most 3/4 full: the u64
//                                     `left | right << id bits | (rank + 1) << 2·id bits`, or 0
//   unk      [u32; 1]
//   pool     [u8; pool bytes]
//
// Packed so they are read where they lie, nothing decoded at load, and probed as a hash table is:
// gemma-4's 514,906 merges take 8 bytes a slot plus 4 for what each makes.
// ---------------------------------------------------------------------------------------------

const MAGIC: u32 = u32::from_le_bytes(*b"TKB3");
const EMPTY: u32 = u32::MAX;
const HEADER: usize = 8;
const BYTE_FALLBACK: u32 = 1;
const FUSE_UNK: u32 = 2;
const IGNORE_MERGES: u32 = 4;

fn str_hash(s: &[u8]) -> u64 {
    // FNV-1a, then mixed: `slot_of` reads the high bits, which FNV leaves poorly mixed for the
    // one-character strings looked up most.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in s {
        h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
    }
    pair_hash(h)
}

fn pair_hash(key: u64) -> u64 {
    // splitmix64's finalizer.
    let mut x = key;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// `h` onto `0..n` (multiply-shift: no power-of-two table size needed).
fn slot_of(h: u64, n: usize) -> usize {
    ((h as u128 * n as u128) >> 64) as usize
}

/// Slots for `n` keys at most 3/4 full.
fn slots_for(n: usize) -> usize {
    n * 4 / 3 + 1
}

#[derive(Clone, Copy)]
struct Tables {
    offsets: &'static [u32],
    bytes: &'static [u32],
    vocab: &'static [u32],
    merged: &'static [u32],
    pairs: &'static [[u32; 2]],
    pool: &'static [u8],
    flags: u32,
    unk: u32,
    id_bits: u32,
}

impl Tables {
    fn parse(raw: &'static [u8]) -> Result<Self, String> {
        if raw.as_ptr().align_offset(4) != 0 || raw.len() < HEADER * 4 {
            return Err("baked tokenizer tables: misaligned or truncated".into());
        }
        // SAFETY: 4-aligned (checked), and every bit pattern is a valid u32.
        let words =
            unsafe { std::slice::from_raw_parts(raw.as_ptr().cast::<u32>(), raw.len() / 4) };
        let [
            magic,
            ids,
            pool_len,
            vocab_slots,
            merges,
            merge_slots,
            flags,
            id_bits,
        ] = <[u32; HEADER]>::try_from(&words[..HEADER]).expect("HEADER words");
        if magic != MAGIC {
            return Err("baked tokenizer tables: bad magic".into());
        }
        let (ids, pool_len) = (ids as usize, pool_len as usize);
        let mut at = HEADER;
        let mut take = |n: usize| -> Result<&'static [u32], String> {
            let s = words
                .get(at..at + n)
                .ok_or_else(|| "baked tokenizer tables: truncated".to_string())?;
            at += n;
            Ok(s)
        };
        let offsets = take(ids + 1)?;
        let bytes = take(256)?;
        let vocab = take(vocab_slots as usize)?;
        let merged = take(merges as usize)?;
        let pairs = take(merge_slots as usize * 2)?;
        // SAFETY: `[u32; 2]` has the alignment of `u32` and no padding.
        let pairs = unsafe {
            std::slice::from_raw_parts(pairs.as_ptr().cast::<[u32; 2]>(), merge_slots as usize)
        };
        let unk = take(1)?[0];
        let pool = raw
            .get(at * 4..at * 4 + pool_len)
            .ok_or_else(|| "baked tokenizer tables: truncated pool".to_string())?;
        Ok(Self {
            offsets,
            bytes,
            vocab,
            merged,
            pairs,
            pool,
            flags,
            unk,
            id_bits,
        })
    }

    fn token(&self, id: u32) -> Option<&'static str> {
        let i = id as usize;
        let (&start, &end) = (self.offsets.get(i)?, self.offsets.get(i + 1)?);
        // SAFETY: `bake` wrote every token as a whole `&str`.
        Some(unsafe { std::str::from_utf8_unchecked(&self.pool[start as usize..end as usize]) })
    }

    fn id(&self, token: &str) -> Option<u32> {
        let n = self.vocab.len();
        let mut slot = slot_of(str_hash(token.as_bytes()), n);
        loop {
            let id = self.vocab[slot];
            if id == EMPTY {
                return None;
            }
            if self.token(id) == Some(token) {
                return Some(id);
            }
            slot = if slot + 1 == n { 0 } else { slot + 1 };
        }
    }

    /// `(rank, merged)` of the merge of `a` then `b`, if there is one.
    fn merge(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        let key = a as u64 | (b as u64) << self.id_bits;
        let key_mask = (1u64 << (2 * self.id_bits)) - 1;
        let n = self.pairs.len();
        let mut slot = slot_of(pair_hash(key), n);
        loop {
            let [lo, hi] = self.pairs[slot];
            let e = lo as u64 | (hi as u64) << 32;
            if e == 0 {
                return None;
            }
            if e & key_mask == key {
                let rank = (e >> (2 * self.id_bits)) as u32 - 1;
                return Some((rank, self.merged[rank as usize]));
            }
            slot = if slot + 1 == n { 0 } else { slot + 1 };
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The model: `tokenizers`' BPE (models/bpe/model.rs `merge_word`, `tokenize_with_cache`;
// models/bpe/word.rs `Word::merge_all`), line for line, over the tables. Its per-thread cache
// is a speed-up only and is not ported.
// ---------------------------------------------------------------------------------------------

/// A BPE model whose vocabulary and merges live in the binary.
#[derive(Clone)]
pub struct BakedBpe {
    tables: Tables,
}

#[derive(Debug, Clone, Copy)]
struct Symbol {
    c: u32,
    prev: isize,
    next: isize,
    len: usize,
}

#[derive(Debug, Eq)]
struct Merge {
    pos: usize,
    rank: u32,
    new_id: u32,
}

impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank && self.pos == other.pos
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Merge {
    /// A min-heap on (rank, pos), as `tokenizers`' — whose equal elements are identical (a rank
    /// names one pair, so one merged id), so heap order among them cannot change a result.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.pos.cmp(&self.pos))
    }
}

struct Word(Vec<Symbol>);

impl Word {
    fn add(&mut self, c: u32, len: usize) {
        let n = self.0.len() as isize;
        let prev = match self.0.last_mut() {
            Some(last) => {
                last.next = n;
                n - 1
            }
            None => -1,
        };
        self.0.push(Symbol {
            c,
            prev,
            next: -1,
            len,
        });
    }

    fn merge_all(&mut self, tables: &Tables) {
        let symbols = &mut self.0;
        let mut queue: BinaryHeap<Merge> = symbols
            .windows(2)
            .enumerate()
            .filter_map(|(pos, w)| {
                let (rank, new_id) = tables.merge(w[0].c, w[1].c)?;
                Some(Merge { pos, rank, new_id })
            })
            .collect();
        while let Some(top) = queue.pop() {
            if symbols[top.pos].len == 0 || symbols[top.pos].next == -1 {
                continue;
            }
            let next_pos = symbols[top.pos].next as usize;
            let right = symbols[next_pos];
            // An expired entry: the pair at `pos` is no longer the one it merged.
            if tables
                .merge(symbols[top.pos].c, right.c)
                .is_none_or(|(_, new_id)| new_id != top.new_id)
            {
                continue;
            }
            let current = &mut symbols[top.pos];
            current.c = top.new_id;
            current.len += right.len;
            current.next = right.next;
            symbols[next_pos].len = 0;
            if right.next > -1 && (right.next as usize) < symbols.len() {
                symbols[right.next as usize].prev = top.pos as isize;
            }
            let current = symbols[top.pos];
            if current.prev >= 0 {
                let prev = symbols[current.prev as usize];
                if let Some((rank, new_id)) = tables.merge(prev.c, current.c) {
                    queue.push(Merge {
                        pos: current.prev as usize,
                        rank,
                        new_id,
                    });
                }
            }
            let next = current.next as usize;
            if next < symbols.len()
                && let Some((rank, new_id)) = tables.merge(current.c, symbols[next].c)
            {
                queue.push(Merge {
                    pos: top.pos,
                    rank,
                    new_id,
                });
            }
        }
        symbols.retain(|s| s.len != 0);
    }
}

impl BakedBpe {
    fn merge_word(&self, w: &str) -> Word {
        let t = &self.tables;
        let mut word = Word(Vec::with_capacity(w.len()));
        let mut unk: Option<(u32, usize)> = None;
        let mut chars = w.char_indices().peekable();
        while let Some((i, _)) = chars.next() {
            let end = chars.peek().map_or(w.len(), |&(e, _)| e);
            let s = &w[i..end];
            let byte_len = s.len();
            if let Some(id) = t.id(s) {
                if let Some((unk_id, unk_len)) = unk.take() {
                    word.add(unk_id, unk_len);
                }
                word.add(id, byte_len);
                continue;
            }
            if t.flags & BYTE_FALLBACK != 0 {
                let fallback: Option<Vec<u32>> = s
                    .bytes()
                    .map(|b| Some(t.bytes[b as usize]).filter(|&id| id != EMPTY))
                    .collect();
                if let Some(ids) = fallback {
                    for id in ids {
                        word.add(id, 1);
                    }
                    continue;
                }
            }
            if t.unk != EMPTY {
                unk = match (unk, t.flags & FUSE_UNK != 0) {
                    (Some((unk_id, unk_len)), true) => Some((unk_id, unk_len + byte_len)),
                    (Some((unk_id, unk_len)), false) => {
                        word.add(unk_id, unk_len);
                        Some((t.unk, byte_len))
                    }
                    (None, _) => Some((t.unk, byte_len)),
                };
            }
        }
        if let Some((unk_id, unk_len)) = unk {
            word.add(unk_id, unk_len);
        }
        word.merge_all(t);
        word
    }
}

impl Model for BakedBpe {
    type Trainer = BpeTrainer;

    fn tokenize(&self, sequence: &str) -> tokenizers::Result<Vec<Token>> {
        if sequence.is_empty() {
            return Ok(vec![]);
        }
        if self.tables.flags & IGNORE_MERGES != 0
            && let Some(id) = self.tables.id(sequence)
        {
            return Ok(vec![Token::new(
                id,
                sequence.to_string(),
                (0, sequence.len()),
            )]);
        }
        let mut pos = 0;
        Ok(self
            .merge_word(sequence)
            .0
            .iter()
            .map(|s| {
                let offsets = (pos, pos + s.len);
                pos += s.len;
                let value = self.tables.token(s.c).unwrap_or_default().to_string();
                Token::new(s.c, value, offsets)
            })
            .collect())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tables.id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.tables.token(id).map(str::to_string)
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        (0..self.get_vocab_size() as u32)
            .filter_map(|id| Some((self.tables.token(id)?.to_string(), id)))
            .collect()
    }

    fn get_vocab_size(&self) -> usize {
        self.tables.offsets.len() - 1
    }

    fn save(&self, _folder: &Path, _prefix: Option<&str>) -> tokenizers::Result<Vec<PathBuf>> {
        Err("a baked tokenizer is part of the binary and is not saved".into())
    }

    fn get_trainer(&self) -> BpeTrainer {
        BpeTrainer::default()
    }
}

thread_local! {
    /// The tables [`load`] is deserializing a pipeline around, for [`BakedBpe`]'s `Deserialize`.
    static LOADING: RefCell<Option<(String, Tables)>> = const { RefCell::new(None) };
}

impl<'de> Deserialize<'de> for BakedBpe {
    /// The model stanza `bake` wrote, `{"digest": …}`: the tables `load` is loading.
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Stanza {
            digest: String,
        }
        let Stanza { digest } = Stanza::deserialize(d)?;
        LOADING.with(|l| match &*l.borrow() {
            Some((loading, tables)) if *loading == digest => Ok(Self { tables: *tables }),
            _ => Err(D::Error::custom(format!(
                "baked tokenizer {digest}: deserialized outside `baked_tokenizer::load`"
            ))),
        })
    }
}

/// A baked tokenizer from its `tables` and `pipeline` ([`bake`]'s output).
pub fn load(digest: &str, tables: &'static [u8], pipeline: &str) -> Result<Baked, String> {
    let tables = Tables::parse(tables)?;
    LOADING.with(|l| *l.borrow_mut() = Some((digest.to_string(), tables)));
    let baked = serde_json::from_str::<Baked>(pipeline);
    LOADING.with(|l| *l.borrow_mut() = None);
    baked.map_err(|e| format!("baked tokenizer {digest}: {e}"))
}

// ---------------------------------------------------------------------------------------------
// Baking (the build's side).
// ---------------------------------------------------------------------------------------------

/// What [`bake`] makes of a `tokenizer.json`.
pub struct Bake {
    /// The BPE model's tables (`[`Tables`]`'s layout).
    pub tables: Vec<u8>,
    /// The rest of the file, its model stanza replaced by `{"digest": …}`.
    pub pipeline: String,
}

/// Text [`bake`] encodes and decodes both ways before it trusts its tables: ASCII words and
/// punctuation, whitespace runs, digits, code, accented / CJK / emoji text (multi-byte and
/// byte-fallback paths), and a chat turn's markup.
const CORPUS: &[&str] = &[
    "The quick brown fox jumps over the lazy dog.",
    "  leading spaces,\ttabs,\n\nnewlines   and   runs   of spaces  ",
    "Numbers: 0 7 42 1234567890 3.14159 -2.5e-10 1,000,000 0x1F 0b1010",
    "fn main() { let v: Vec<u32> = (0..10).map(|x| x * x).collect(); println!(\"{v:?}\"); }",
    "def f(x):\n    return {'a': [1, 2, 3], \"b\": None}  # comment\n",
    "Ça va très bien, merci! Größe, naïve façade, smörgåsbord, ¿qué pasa?",
    "日本語のテキスト、中文文本，한국어 텍스트。",
    "Emoji: 🙂🚀👩🏽‍💻 and symbols ∑∫√∞ ≤ ≥ ≠ ← → ★ ♥",
    "<start_of_turn>user\nWrite about the history of France.<end_of_turn>\n<start_of_turn>model\n",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nhi<|im_end|>",
    "antidisestablishmentarianism pneumonoultramicroscopicsilicovolcanoconiosis",
    "\u{0}\u{1}\u{7f} control \u{feff}bom \u{200b}zero-width",
];

/// Bake `tokenizer_json` (whose blob digest is `digest`) into tables, or say why it cannot be.
///
/// Refuses anything [`BakedBpe`] does not reproduce: a non-BPE model, BPE dropout, a continuing-
/// subword prefix or end-of-word suffix, a vocabulary whose ids are not exactly `0..n` — and any
/// tokenizer whose [`CORPUS`] encodings (ids and offsets, with and without special tokens) or
/// decodings differ from `tokenizers`' own.
pub fn bake(tokenizer_json: &str, digest: &str) -> Result<Bake, String> {
    let original: tokenizers::Tokenizer = tokenizer_json.parse().map_err(|e| format!("{e}"))?;
    let ModelWrapper::BPE(bpe) = original.get_model() else {
        return Err("not a BPE model".into());
    };
    // `tokenizers`' own canonical view: merges ordered by rank, vocabulary by id.
    let model = serde_json::to_value(bpe).map_err(|e| e.to_string())?;
    if model["dropout"].as_f64().is_some_and(|p| p != 0.0) {
        return Err("BPE dropout".into());
    }
    if !model["continuing_subword_prefix"].is_null() || !model["end_of_word_suffix"].is_null() {
        return Err("a continuing-subword prefix or end-of-word suffix".into());
    }
    let vocab: HashMap<&str, u32> = model["vocab"]
        .as_object()
        .ok_or("no vocab")?
        .iter()
        .map(|(t, id)| Some((t.as_str(), u32::try_from(id.as_u64()?).ok()?)))
        .collect::<Option<_>>()
        .ok_or("a vocab id that is not a u32")?;
    let ids = vocab.len();
    let mut by_id: Vec<Option<&str>> = vec![None; ids];
    for (&t, &id) in &vocab {
        let slot = by_id
            .get_mut(id as usize)
            .ok_or("vocab ids are not exactly 0..n")?;
        if slot.replace(t).is_some() {
            return Err("two tokens share an id".into());
        }
    }
    let by_id: Vec<&str> = by_id
        .into_iter()
        .collect::<Option<_>>()
        .ok_or("vocab ids are not exactly 0..n")?;
    let id = |t: &str| vocab.get(t).copied();

    let mut words: Vec<u32> = vec![0; HEADER];
    let mut pool: Vec<u8> = Vec::new();
    words.push(0);
    for t in &by_id {
        pool.extend_from_slice(t.as_bytes());
        words.push(u32::try_from(pool.len()).map_err(|_| "string pool over 4 GiB")?);
    }
    words.extend((0..=255u8).map(|b| id(&format!("<{b:#04X}>")).unwrap_or(EMPTY)));
    let vocab_slots = slots_for(ids);
    let mut table = vec![EMPTY; vocab_slots];
    for (i, t) in by_id.iter().enumerate() {
        let mut slot = slot_of(str_hash(t.as_bytes()), vocab_slots);
        while table[slot] != EMPTY {
            slot = if slot + 1 == vocab_slots { 0 } else { slot + 1 };
        }
        table[slot] = i as u32;
    }
    words.extend_from_slice(&table);
    let merges = model["merges"].as_array().ok_or("no merges")?;
    // Each merge by rank: (left, right, merged).
    let mut by_rank: Vec<(u32, u32, u32)> = Vec::with_capacity(merges.len());
    for m in merges {
        let (a, b) = match m {
            serde_json::Value::Array(p) if p.len() == 2 => (p[0].as_str(), p[1].as_str()),
            serde_json::Value::String(s) => s
                .split_once(' ')
                .map_or((None, None), |(a, b)| (Some(a), Some(b))),
            _ => (None, None),
        };
        let (a, b) = a.zip(b).ok_or("a merge that is not a pair of strings")?;
        let (ia, ib) = id(a)
            .zip(id(b))
            .ok_or("a merge of tokens out of the vocab")?;
        let merged = id(&format!("{a}{b}")).ok_or("a merge whose result is out of the vocab")?;
        by_rank.push((ia, ib, merged));
    }
    let id_bits = u32::BITS - (ids as u32).saturating_sub(1).leading_zeros();
    // `rank + 1` (0 marks a free slot) in what the two ids leave.
    if 2 * id_bits + (u32::BITS - (by_rank.len() as u32).leading_zeros()) > 64 {
        return Err("ids and merge ranks too wide to pack in 64 bits".into());
    }
    words.extend(by_rank.iter().map(|&(_, _, merged)| merged));
    let merge_slots = slots_for(by_rank.len());
    let mut pairs = vec![0u64; merge_slots];
    for (rank, &(a, b, _)) in by_rank.iter().enumerate() {
        let key = a as u64 | (b as u64) << id_bits;
        let mut slot = slot_of(pair_hash(key), merge_slots);
        while pairs[slot] != 0 {
            slot = if slot + 1 == merge_slots { 0 } else { slot + 1 };
        }
        pairs[slot] = key | (rank as u64 + 1) << (2 * id_bits);
    }
    words.extend(pairs.iter().flat_map(|&e| [e as u32, (e >> 32) as u32]));
    let unk = match model["unk_token"].as_str() {
        Some(t) => id(t).ok_or("an unk token out of the vocab")?,
        None => EMPTY,
    };
    words.push(unk);
    let flags = [
        (
            model["byte_fallback"].as_bool() == Some(true),
            BYTE_FALLBACK,
        ),
        (model["fuse_unk"].as_bool() == Some(true), FUSE_UNK),
        (
            model["ignore_merges"].as_bool() == Some(true),
            IGNORE_MERGES,
        ),
    ]
    .iter()
    .filter(|(on, _)| *on)
    .fold(0, |f, (_, bit)| f | bit);
    words[..HEADER].copy_from_slice(&[
        MAGIC,
        ids as u32,
        pool.len() as u32,
        vocab_slots as u32,
        by_rank.len() as u32,
        merge_slots as u32,
        flags,
        id_bits,
    ]);
    let mut tables: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    tables.extend_from_slice(&pool);

    let mut file: serde_json::Value =
        serde_json::from_str(tokenizer_json).map_err(|e| e.to_string())?;
    file["model"] = serde_json::json!({ "digest": digest });
    let pipeline = serde_json::to_string(&file).map_err(|e| e.to_string())?;

    // Trust nothing: the baked tokenizer must encode and decode the corpus as the file's does.
    let leaked: &'static [u8] = aligned_copy(&tables);
    let baked = load(digest, leaked, &pipeline)?;
    for text in CORPUS {
        for special in [false, true] {
            let want = original.encode(*text, special).map_err(|e| e.to_string())?;
            let got = baked.encode(*text, special).map_err(|e| e.to_string())?;
            if want.get_ids() != got.get_ids() || want.get_offsets() != got.get_offsets() {
                return Err(format!(
                    "encodes {text:?} differently (special tokens: {special})"
                ));
            }
            for skip in [false, true] {
                let want = original
                    .decode(want.get_ids(), skip)
                    .map_err(|e| e.to_string())?;
                let got = baked
                    .decode(got.get_ids(), skip)
                    .map_err(|e| e.to_string())?;
                if want != got {
                    return Err(format!("decodes {text:?} differently"));
                }
            }
        }
    }
    Ok(Bake { tables, pipeline })
}

/// `bytes`, copied to a 4-aligned allocation that lives for the process (a `bake`'s self-check,
/// once per tokenizer per build; tests).
fn aligned_copy(bytes: &[u8]) -> &'static [u8] {
    let mut words = vec![0u32; bytes.len().div_ceil(4)];
    // SAFETY: `words` holds at least `bytes.len()` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), words.as_mut_ptr().cast::<u8>(), bytes.len());
    }
    let words: &'static [u32] = Box::leak(words.into_boxed_slice());
    // SAFETY: `words` is `bytes.len()` initialized bytes and more, alive forever.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), bytes.len()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small byte-fallback BPE in gemma's shape (space → ▁, split on spaces, BOS added), with
    /// merges that compete for the same symbols and an unknown character.
    const TOKENIZER: &str = r#"{
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [
            {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 1, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
        "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false},
        "post_processor": {"type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {"<bos>": {"id": "<bos>", "ids": [1], "tokens": ["<bos>"]}}},
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"}, {"type": "Fuse"}]},
        "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null,
            "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": true, "ignore_merges": false,
            "vocab": {"<unk>": 0, "<bos>": 1, "▁": 2, "h": 3, "e": 4, "l": 5, "o": 6, "w": 7, "r": 8, "d": 9,
                "he": 10, "ll": 11, "hell": 12, "hello": 13, "▁w": 14, "or": 15, "▁wor": 16, "▁world": 17,
                "lo": 18, "<0xC3>": 19, "<0xA9>": 20, "<0xE2>": 21, "▁worl": 22},
            "merges": [["h", "e"], ["l", "l"], ["he", "ll"], ["hell", "o"], ["▁", "w"], ["o", "r"],
                ["▁w", "or"], ["l", "o"], ["▁wor", "l"], ["▁worl", "d"]]}
    }"#;

    #[test]
    fn bakes_a_byte_fallback_bpe_that_encodes_as_the_file_does() {
        let bake = bake(TOKENIZER, "test").expect("bakes");
        let baked = load("test", aligned_copy(&bake.tables), &bake.pipeline).expect("loads");
        let original: tokenizers::Tokenizer = TOKENIZER.parse().unwrap();
        for text in [
            "hello world",
            "hello hello",
            "héllo wörld",
            "hollow",
            "zzz hello",
            "",
        ] {
            let want = original.encode(text, true).unwrap();
            let got = baked.encode(text, true).unwrap();
            assert_eq!(got.get_ids(), want.get_ids(), "{text:?}");
            assert_eq!(got.get_offsets(), want.get_offsets(), "{text:?}");
        }
        assert_eq!(baked.get_vocab_size(true), original.get_vocab_size(true));
        assert_eq!(baked.token_to_id("hello"), Some(13));
        assert_eq!(baked.id_to_token(17).as_deref(), Some("▁world"));
    }

    #[test]
    fn refuses_what_it_cannot_reproduce() {
        let dropout = TOKENIZER.replace(r#""dropout": null"#, r#""dropout": 0.1"#);
        assert_eq!(bake(&dropout, "t").err().as_deref(), Some("BPE dropout"));
        let sparse = TOKENIZER.replace(r#""▁worl": 22"#, r#""▁worl": 30"#);
        assert_eq!(
            bake(&sparse, "t").err().as_deref(),
            Some("vocab ids are not exactly 0..n")
        );
    }

    /// gemma-4's 32 MB tokenizer from the Hub cache, baked and checked against `tokenizers` on
    /// more text than `CORPUS`; prints what loading costs each way. Needs the checkpoint cached:
    /// `cargo test -p scratchy-core-model --lib baked_tokenizer -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs mlx-community/gemma-4-26b-a4b-it-4bit in the Hub cache"]
    fn bakes_gemma4() {
        let repo = hf_hub_downloader::cache::repo_dir(
            &hf_hub_downloader::cache::default_root(),
            "mlx-community/gemma-4-26b-a4b-it-4bit",
        );
        let snapshot = std::fs::read_dir(repo.join("snapshots"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        let path = snapshot.path().join("tokenizer.json");
        let digest = cached_digest(&path).expect("a Hub-cache link");
        let json = std::fs::read_to_string(&path).unwrap();

        let t = std::time::Instant::now();
        let bake = bake(&json, &digest).expect("bakes");
        eprintln!(
            "bake (incl. its self-check): {:?}, tables {} MB",
            t.elapsed(),
            bake.tables.len() >> 20
        );
        let tables = aligned_copy(&bake.tables);
        let t = std::time::Instant::now();
        let baked = load(&digest, tables, &bake.pipeline).expect("loads");
        eprintln!("load baked: {:?}", t.elapsed());
        let t = std::time::Instant::now();
        let original = tokenizers::Tokenizer::from_file(&path).unwrap();
        eprintln!("Tokenizer::from_file: {:?}", t.elapsed());

        let text: String = CORPUS
            .iter()
            .cycle()
            .take(400)
            .copied()
            .collect::<Vec<_>>()
            .join(" ");
        for special in [false, true] {
            let (want, got) = (
                original.encode(text.as_str(), special).unwrap(),
                baked.encode(text.as_str(), special).unwrap(),
            );
            assert_eq!(got.get_ids(), want.get_ids());
            assert_eq!(got.get_offsets(), want.get_offsets());
        }
        let t = std::time::Instant::now();
        let n = original.encode(text.as_str(), false).unwrap().len();
        let (t_file, t) = (t.elapsed(), std::time::Instant::now());
        baked.encode(text.as_str(), false).unwrap();
        eprintln!(
            "encode {n} tokens: from_file's {t_file:?}, baked {:?}",
            t.elapsed()
        );
    }

    #[test]
    fn the_pipeline_only_deserializes_inside_load() {
        let bake = bake(TOKENIZER, "test").expect("bakes");
        assert!(serde_json::from_str::<Baked>(&bake.pipeline).is_err());
    }

    #[test]
    fn the_cache_key_is_the_blob_a_snapshot_links_to() {
        let dir = std::env::temp_dir().join(format!("baked-tok-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("blobs")).unwrap();
        let digest = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
        std::fs::write(dir.join("blobs").join(digest), "{}").unwrap();
        std::os::unix::fs::symlink(format!("blobs/{digest}"), dir.join("tokenizer.json")).unwrap();
        std::fs::write(dir.join("plain.json"), "{}").unwrap();
        assert_eq!(
            cached_digest(&dir.join("tokenizer.json")).as_deref(),
            Some(digest)
        );
        assert_eq!(cached_digest(&dir.join("plain.json")), None);
        let _ = std::fs::remove_dir_all(dir);
    }
}
