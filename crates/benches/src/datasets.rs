// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Dataset loading for `scr bench` — supports ShareGPT and random prompt generation.
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
#[allow(dead_code)]
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
