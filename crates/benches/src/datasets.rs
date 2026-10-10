// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Dataset loading for `scr bench` — supports ShareGPT, random prompt
//! generation, and synthetic multi-turn chats.
//!
//! Matches Python vLLM's `ShareGPTDataset.sample()` and `RandomDataset.sample()` methodology.

use std::path::Path;

use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tokenizers::Tokenizer;
use tokenizers::tokenizer::PostProcessor;

/// A single benchmark request with pre-computed token lengths.
#[derive(Debug, Clone)]
pub struct SampleRequest {
    /// The prompt text to send.
    pub prompt: String,
    /// Number of prompt tokens (after tokenization).
    pub prompt_len: usize,
    /// Expected output length (tokens).
    pub expected_output_len: usize,
}

/// Load a ShareGPT-format dataset, matching Python's `ShareGPTDataset.sample()`.
///
/// JSON format: `[{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}]`
///
/// For each conversation:
/// - First "human" turn → prompt
/// - First "gpt" turn → expected output (used for length estimation)
/// - Filter: `prompt_len >= 4`, `output_len >= 4`, `prompt_len + output_len <= max_model_len`
/// - Shuffle with seed, take `num_prompts`
pub fn load_sharegpt(
    path: &Path,
    tokenizer: &Tokenizer,
    num_prompts: usize,
    max_model_len: Option<usize>,
    seed: u64,
) -> Result<Vec<SampleRequest>> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read ShareGPT dataset from {}", path.display()))?;
    let entries: Vec<serde_json::Value> =
        serde_json::from_str(&data).context("Failed to parse ShareGPT JSON")?;

    let max_len = max_model_len.unwrap_or(usize::MAX);

    let mut samples: Vec<SampleRequest> = Vec::new();

    for entry in &entries {
        let conversations = match entry.get("conversations").and_then(|c| c.as_array()) {
            Some(c) => c,
            None => continue,
        };

        // Find first human turn and first GPT turn.
        let human_text = conversations
            .iter()
            .find(|c| c.get("from").and_then(|f| f.as_str()) == Some("human"))
            .and_then(|c| c.get("value").and_then(|v| v.as_str()));
        let gpt_text = conversations
            .iter()
            .find(|c| c.get("from").and_then(|f| f.as_str()) == Some("gpt"))
            .and_then(|c| c.get("value").and_then(|v| v.as_str()));

        let (prompt, output) = match (human_text, gpt_text) {
            (Some(h), Some(g)) => (h, g),
            _ => continue,
        };

        // Tokenize to get lengths.
        let prompt_encoding = tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
        let output_encoding = tokenizer
            .encode(output, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

        let prompt_len = prompt_encoding.get_ids().len();
        let output_len = output_encoding.get_ids().len();

        // Filter: matching Python's min thresholds.
        if prompt_len < 4 || output_len < 4 {
            continue;
        }
        if prompt_len + output_len > max_len {
            continue;
        }

        samples.push(SampleRequest {
            prompt: prompt.to_string(),
            prompt_len,
            expected_output_len: output_len,
        });
    }

    anyhow::ensure!(
        !samples.is_empty(),
        "No valid samples found in ShareGPT dataset at {}. \
         Check that the file contains conversations with 'human' and 'gpt' turns.",
        path.display()
    );

    // Shuffle deterministically with StdRng (Fisher-Yates, same algorithm as Python's
    // random.shuffle — note: PRNG differs from Python's MT19937 so order won't match
    // bit-for-bit at the same seed, but the algorithm is correct).
    shuffle_with_seed(&mut samples, seed);

    // Take up to num_prompts.
    samples.truncate(num_prompts);

    Ok(samples)
}

/// Fisher-Yates shuffle with `StdRng` (seeded), matching Python's `random.shuffle` algorithm.
fn shuffle_with_seed<T>(data: &mut [T], seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    for i in (1..data.len()).rev() {
        let j = rng.random_range(0..=i);
        data.swap(i, j);
    }
}

/// Generate random prompts matching Python's `RandomDataset.sample()`.
///
/// Algorithm matches Python exactly:
/// 1. Seed `StdRng` from `seed` (Python uses `numpy.default_rng(seed)` — PCG64,
///    which differs bit-for-bit but we follow the same call ordering).
/// 2. Sample `input_lens`, `output_lens`, `offsets` in that order (consuming the RNG
///    in the same sequence as Python's `get_sampling_params()`).
/// 3. For each request `i`:
///    - `inner = allowed[(offset[i] + i + j) % len(allowed)]` for `j` in `0..input_len`
///    - Decode → re-encode with up to 10 retries (`gen_prompt_decode_to_target_len`):
///      * if len < target: pad with random tokens from full vocab `[0, vocab_size)` (same as Python)
///      * if len > target: truncate
pub fn generate_random(
    tokenizer: &Tokenizer,
    num_requests: usize,
    input_len: usize,
    output_len: usize,
    range_ratio: f64,
    prefix_len: usize,
    seed: u64,
) -> Result<Vec<SampleRequest>> {
    anyhow::ensure!(
        (0.0..1.0).contains(&range_ratio),
        "range_ratio must be in [0, 1), got {range_ratio}"
    );

    // Build allowed tokens (exclude special tokens), matching Python.
    let vocab_size = tokenizer.get_vocab_size(true);
    let special_ids: std::collections::HashSet<u32> = tokenizer
        .get_added_vocabulary()
        .get_added_tokens_decoder()
        .iter()
        .filter(|(_, t)| t.special)
        .map(|(id, _)| *id)
        .collect();
    let allowed_tokens: Vec<u32> = (0..vocab_size as u32)
        .filter(|id| !special_ids.contains(id))
        .collect();
    let num_allowed = allowed_tokens.len();

    anyhow::ensure!(
        num_allowed > 0,
        "No non-special tokens found in tokenizer vocabulary"
    );

    let mut rng = StdRng::seed_from_u64(seed);

    // Matching Python's `get_sampling_params()`:
    //   real_input_len = max(0, input_len - num_special_tokens_to_add())
    // Python's `num_special_tokens_to_add()` queries the tokenizer's post-processor
    // for how many special tokens it prepends/appends to a single (non-pair) sequence.
    let num_special = tokenizer
        .get_post_processor()
        .map(|pp| pp.added_tokens(false))
        .unwrap_or(0);
    let real_input_len = input_len.saturating_sub(num_special);

    // Matching Python's `get_sampling_params()` RNG call order:
    //   1. rng.integers(input_low, input_high+1, size=N)  → input_lens
    //   2. rng.integers(output_low, output_high+1, size=N) → output_lens
    //   3. rng.integers(0, vocab_size, size=N)            → offsets
    let input_low = (real_input_len as f64 * (1.0 - range_ratio)).floor() as usize;
    let input_high = (real_input_len as f64 * (1.0 + range_ratio)).ceil() as usize;
    let output_low = ((output_len as f64 * (1.0 - range_ratio)).floor() as usize).max(1);
    let output_high = ((output_len as f64 * (1.0 + range_ratio)).ceil() as usize).max(1);

    let input_lens: Vec<usize> = (0..num_requests)
        .map(|_| rng.random_range(input_low..=input_high))
        .collect();
    let output_lens: Vec<usize> = (0..num_requests)
        .map(|_| rng.random_range(output_low..=output_high))
        .collect();
    // Offsets sampled from full vocab range [0, vocab_size), matching Python.
    let offsets: Vec<usize> = (0..num_requests)
        .map(|_| rng.random_range(0..vocab_size))
        .collect();

    // Generate prefix once (prefix_len=0 → empty), matching Python's `get_prefix()`.
    let prefix_token_ids: Vec<u32> = if prefix_len > 0 {
        let raw: Vec<u32> = (0..prefix_len)
            .map(|_| allowed_tokens[rng.random_range(0..num_allowed)])
            .collect();
        let (_, ids) = gen_prompt_to_target_len(tokenizer, &mut rng, raw, prefix_len, vocab_size)?;
        ids
    } else {
        Vec::new()
    };

    let mut samples = Vec::with_capacity(num_requests);
    for i in 0..num_requests {
        let req_input_len = input_lens[i];
        // inner_seq = allowed_tokens[(offset + index + arange(input_len)) % len(allowed)]
        let inner_seq: Vec<u32> = (0..req_input_len)
            .map(|j| allowed_tokens[(offsets[i] + i + j) % num_allowed])
            .collect();

        // token_sequence = prefix_token_ids + inner_seq
        let token_sequence: Vec<u32> = prefix_token_ids.iter().copied().chain(inner_seq).collect();

        let total_target_len = prefix_len + req_input_len;

        let (prompt, actual_ids) = gen_prompt_to_target_len(
            tokenizer,
            &mut rng,
            token_sequence,
            total_target_len,
            vocab_size,
        )?;

        samples.push(SampleRequest {
            prompt,
            prompt_len: actual_ids.len(),
            expected_output_len: output_lens[i],
        });
    }

    Ok(samples)
}

/// Decode token ids to text, re-encode, and iteratively adjust to `target_len`.
///
/// Matches Python's `gen_prompt_decode_to_target_len` exactly:
/// - Up to 10 retries (`max_retry = 10`)
/// - When short: extend with random tokens from `[0, vocab_size)` (full vocab, NOT allowed-only)
/// - When long: truncate to `target_len`
/// - After `remain_num_try <= 0`: break unconditionally
///
/// Returns `(prompt_string, final_token_ids)`.
fn gen_prompt_to_target_len(
    tokenizer: &Tokenizer,
    rng: &mut StdRng,
    mut token_ids: Vec<u32>,
    target_len: usize,
    vocab_size: usize,
) -> Result<(String, Vec<u32>)> {
    let mut remain: i32 = 10;
    let mut prompt;
    loop {
        prompt = tokenizer
            .decode(&token_ids, true)
            .map_err(|e| anyhow::anyhow!("Decode failed: {e}"))?;
        let encoded = tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Encode failed: {e}"))?;
        token_ids = encoded.get_ids().to_vec();

        if remain <= 0 {
            break;
        }
        if token_ids.len() == target_len {
            break;
        } else if token_ids.len() < target_len {
            // Pad with random tokens from full vocab range, matching Python:
            //   extra_tokens = rng.integers(0, vocab_size, size=needed)
            let needed = target_len - token_ids.len();
            for _ in 0..needed {
                token_ids.push(rng.random_range(0..vocab_size as u32));
            }
        } else {
            token_ids.truncate(target_len);
        }
        remain -= 1;
    }
    Ok((prompt, token_ids))
}

/// The seed of the pre-flight and warmup prompts for a run seeded `seed`.
///
/// Sent a measured prompt, they would leave it in the server's prefix cache
/// before it is timed. The high-bit mask keeps these seeds clear of the small
/// consecutive ones sweeps hand out per cell (`bench_serve_compare.sh` uses
/// base + index), so one cell's warmups are never a later cell's prompts.
pub fn throwaway_seed(seed: u64) -> u64 {
    seed ^ 0xA5A5_0000_0000_0000
}

/// Tokens in `text`, without the special tokens a server would add.
pub fn token_count(tokenizer: &Tokenizer, text: &str) -> Result<usize> {
    tokenizer
        .encode(text, false)
        .map(|e| e.get_ids().len())
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))
}

/// The shape [`generate_multi_turn`] builds. Lengths are approximate tokens.
pub struct MultiTurnSpec {
    pub conversations: usize,
    /// Requests per conversation.
    pub turns: usize,
    /// The system prompt every conversation shares.
    pub system_len: usize,
    pub user_len: usize,
    /// Each assistant reply in the history.
    pub reply_len: usize,
}

/// Seeded text and its token count under the sizing tokenizer.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SizedText {
    text: String,
    tokens: usize,
}

/// One chat: `turns` user messages and the `turns - 1` replies between them.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Conversation {
    users: Vec<SizedText>,
    replies: Vec<SizedText>,
}

/// Synthetic multi-turn chats sharing one system prompt.
///
/// The replies in the history are seeded text, not what the model said, so a
/// turn's request does not depend on the previous response and every backend
/// is sent byte-identical requests. Two conversations share only the system
/// prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiTurnDataset {
    system: SizedText,
    conversations: Vec<Conversation>,
}

impl MultiTurnDataset {
    pub fn num_conversations(&self) -> usize {
        self.conversations.len()
    }

    pub fn turns(&self) -> usize {
        self.conversations.first().map_or(0, |c| c.users.len())
    }

    /// The `messages` of conversation `conv`'s request `turn` (0-based): the
    /// system prompt, then user/assistant pairs, ending on user message `turn`.
    pub fn messages(&self, conv: usize, turn: usize) -> serde_json::Value {
        let c = &self.conversations[conv];
        let message =
            |role: &str, t: &SizedText| serde_json::json!({"role": role, "content": t.text});
        let mut messages = vec![message("system", &self.system)];
        for k in 0..=turn {
            if k > 0 {
                messages.push(message("assistant", &c.replies[k - 1]));
            }
            messages.push(message("user", &c.users[k]));
        }
        serde_json::Value::Array(messages)
    }

    /// Tokens in the contents of [`Self::messages`], without the chat template
    /// the server renders around them.
    pub fn content_tokens(&self, conv: usize, turn: usize) -> usize {
        let c = &self.conversations[conv];
        self.system.tokens
            + c.users[..=turn].iter().map(|t| t.tokens).sum::<usize>()
            + c.replies[..turn].iter().map(|t| t.tokens).sum::<usize>()
    }
}

/// Common English words. Messages are built from these and never from decoded
/// token ids, which can spell out special tokens.
const WORDS: &str = "time year people way day man thing woman life child world school state \
    family student group country problem hand part place case week company system program \
    question work government number night point home water room mother area money story fact \
    month lot right study book eye job word business issue side kind head house service friend \
    father power hour game line end member law car city community name president team minute \
    idea kid body information back parent face others level office door health person art war \
    history party result change morning reason research girl guy moment air teacher force \
    education good new first last long great little own other old big high different small \
    large next early young important few public bad same able make take see know come think \
    look want give use find tell ask seem feel try leave call";

/// Build `spec`'s conversations, sizing each message with `count_tokens`.
///
/// Each message draws whole words from its own seeded stream, and the
/// tokenizer only decides where the message ends — so a conversation is the
/// same whatever `spec.conversations` is, and the same seed gives the same
/// requests on every backend.
pub fn generate_multi_turn(
    count_tokens: &dyn Fn(&str) -> Result<usize>,
    spec: &MultiTurnSpec,
    seed: u64,
) -> Result<MultiTurnDataset> {
    anyhow::ensure!(
        spec.conversations > 0 && spec.turns > 0,
        "multi-turn needs at least one conversation of at least one turn"
    );
    anyhow::ensure!(
        spec.system_len > 0 && spec.user_len > 0 && spec.reply_len > 0,
        "multi-turn system, user and reply lengths must each be at least 1 token"
    );
    let sizer = Sizer {
        count_tokens,
        words: WORDS.split_whitespace().collect(),
        seed,
    };
    // Stream 0 is the system prompt; every other message gets the next one.
    let system = sizer.text(0, spec.system_len)?;
    let mut stream = 0;
    let mut conversations = Vec::with_capacity(spec.conversations);
    for _ in 0..spec.conversations {
        let mut users = Vec::with_capacity(spec.turns);
        let mut replies = Vec::with_capacity(spec.turns - 1);
        for turn in 0..spec.turns {
            if turn > 0 {
                stream += 1;
                replies.push(sizer.text(stream, spec.reply_len)?);
            }
            stream += 1;
            users.push(sizer.text(stream, spec.user_len)?);
        }
        conversations.push(Conversation { users, replies });
    }
    Ok(MultiTurnDataset {
        system,
        conversations,
    })
}

/// Cuts one dataset's seeded word streams to length.
struct Sizer<'a> {
    count_tokens: &'a dyn Fn(&str) -> Result<usize>,
    words: Vec<&'static str>,
    seed: u64,
}

impl Sizer<'_> {
    /// The longest run of whole words from stream `stream` that stays within
    /// `target` tokens (one word at least).
    fn text(&self, stream: u64, target: usize) -> Result<SizedText> {
        // (seed, stream) packed side by side, so no two pairs share a stream.
        let mut key = [0u8; 32];
        key[..8].copy_from_slice(&self.seed.to_le_bytes());
        key[8..16].copy_from_slice(&stream.to_le_bytes());
        let mut rng = StdRng::from_seed(key);

        // Under a tokenizer that splits on spaces every word is a token at
        // least, so `target` words reach the target; keep doubling for one
        // that doesn't.
        let mut words: Vec<&str> = Vec::new();
        let mut want = target;
        loop {
            while words.len() < want {
                words.push(self.words[rng.random_range(0..self.words.len())]);
            }
            if want >= target * 64 || (self.count_tokens)(&words.join(" "))? >= target {
                break;
            }
            want *= 2;
        }

        let (mut lo, mut hi) = (1, words.len());
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if (self.count_tokens)(&words[..mid].join(" "))? <= target {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let text = words[..lo].join(" ");
        let tokens = (self.count_tokens)(&text)?;
        Ok(SizedText { text, tokens })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One token per word: sizes messages without a real tokenizer.
    fn words(text: &str) -> Result<usize> {
        Ok(text.split_whitespace().count())
    }

    fn spec() -> MultiTurnSpec {
        MultiTurnSpec {
            conversations: 4,
            turns: 3,
            system_len: 64,
            user_len: 8,
            reply_len: 16,
        }
    }

    #[test]
    fn multi_turn_is_deterministic_per_seed() {
        let a = generate_multi_turn(&words, &spec(), 7).unwrap();
        assert_eq!(a, generate_multi_turn(&words, &spec(), 7).unwrap());
        assert_ne!(a, generate_multi_turn(&words, &spec(), 8).unwrap());
    }

    #[test]
    fn each_turn_is_the_previous_one_plus_a_reply_and_a_question() {
        let d = generate_multi_turn(&words, &spec(), 7).unwrap();
        assert_eq!(d.turns(), 3);
        for conv in 0..d.num_conversations() {
            for turn in 1..d.turns() {
                let prev = d.messages(conv, turn - 1);
                let prev = prev.as_array().unwrap();
                let next = d.messages(conv, turn);
                let next = next.as_array().unwrap();
                assert_eq!(next.len(), prev.len() + 2);
                assert_eq!(next[..prev.len()], prev[..]);
                assert_eq!(next[prev.len()]["role"], "assistant");
                assert_eq!(next[prev.len() + 1]["role"], "user");
            }
        }
    }

    #[test]
    fn conversations_share_only_the_system_prompt() {
        let d = generate_multi_turn(&words, &spec(), 7).unwrap();
        let first = d.messages(0, 0);
        assert_eq!(first[0]["role"], "system");
        for conv in 1..d.num_conversations() {
            let m = d.messages(conv, 0);
            assert_eq!(m[0], first[0]);
            assert_ne!(m[1], first[1]);
        }
    }

    #[test]
    fn warmup_conversations_do_not_share_the_measured_system_prompt() {
        let measured = generate_multi_turn(&words, &spec(), 1000).unwrap();
        let throwaway = generate_multi_turn(&words, &spec(), throwaway_seed(1000)).unwrap();
        assert_ne!(measured.messages(0, 0)[0], throwaway.messages(0, 0)[0]);
        // Nor any later sweep cell's: those are seeded base + index.
        assert!((1000..2000).all(|s| throwaway_seed(1000) != s));
    }

    #[test]
    fn messages_land_within_their_token_budget() {
        let d = generate_multi_turn(&words, &spec(), 7).unwrap();
        assert_eq!(d.content_tokens(0, 0), 64 + 8);
        assert_eq!(d.content_tokens(1, 2), 64 + 3 * 8 + 2 * 16);
        // Three tokens a word: the longest fit is 21 words (63 tokens).
        let three = |t: &str| -> Result<usize> { Ok(3 * t.split_whitespace().count()) };
        let d = generate_multi_turn(&three, &spec(), 7).unwrap();
        assert_eq!(d.system.tokens, 63);
        // Two words a token: the word draw doubles until it can reach 64.
        let half = |t: &str| -> Result<usize> { Ok(t.split_whitespace().count() / 2) };
        let d = generate_multi_turn(&half, &spec(), 7).unwrap();
        assert_eq!(d.system.tokens, 64);
    }
}
