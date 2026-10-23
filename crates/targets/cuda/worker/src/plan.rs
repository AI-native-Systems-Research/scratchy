// SPDX-License-Identifier: Apache-2.0
//! ONE STEP, PLANNED ON THE HOST — everything the CUDA worker decides about a scheduler step
//! before it plays a tape, as plain functions with no toolchain.
//!
//! The worker runs one sequence per step (`max_num_seqs` is 1). Each step it takes the one
//! scheduled request ([`scheduled_request`]), picks the bucket — the compiled tape — that holds
//! the step's tokens ([`choose_bucket`]), pads the token ids and positions to that bucket
//! ([`fill_inputs`]), hands the kernels the scheduler's block table ([`block_table_u32`]), and
//! when the step reaches the end of the request's history ([`emits_token`]) takes the greedy
//! token of its last real row ([`greedy_token`]).
//!
//! Sampling is greedy only in this build: a request asking for anything a greedy argmax cannot
//! honour is refused when it is admitted ([`check_sampling`]), with a message naming the fix.

use std::collections::HashMap;
use std::fmt;

use scratchy_core_common::SamplingParams;
use scratchy_target_cuda::tape::KvTape;

/// Why a step could not be planned, or a request is refused. Every message names what to change.
#[derive(Clone, Debug, PartialEq)]
pub enum PlanError {
    /// More than one request runs tokens this step; the worker runs one.
    ManyRequests { req_ids: Vec<String> },
    /// No compiled bucket holds `tokens` rows; the largest is `largest`.
    NoBucket { tokens: usize, largest: u32 },
    /// A position past the bucket's rows does not fit a `u32`.
    Positions { start: u32, bucket_m: u32 },
    /// Block-table entry `entry` is block `block`, which does not fit a `u32`.
    Block { entry: usize, block: usize },
    /// The result holds `elems` bf16 values, not `bucket_m` rows of logits with a row `row`.
    LogitsRow {
        elems: usize,
        bucket_m: u32,
        row: usize,
    },
    /// Token `id`'s logit is NaN, so no argmax exists.
    NanLogit { id: usize },
    /// The request carries no sampling parameters.
    NoSamplingParams,
    /// The temperature is not 0.
    NotGreedy { temperature: f64 },
    /// A repetition, presence or frequency penalty is set.
    Penalties,
    /// More than one completion per request.
    Completions { n: u32 },
    /// Per-token logprobs.
    Logprobs,
    /// Prompt logprobs.
    PromptLogprobs,
    /// A transform that moves the argmax: `field` names it.
    LogitTransform { field: &'static str },
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const GREEDY: &str = "the CUDA backend samples greedily only in this build";
        match self {
            Self::ManyRequests { req_ids } => write!(
                f,
                "the CUDA worker runs one sequence per step; max_num_seqs is 1, and this step \
                 runs {} requests ({})",
                req_ids.len(),
                req_ids.join(", ")
            ),
            Self::NoBucket { tokens, largest } => write!(
                f,
                "the step runs {tokens} tokens; the largest compiled CUDA bucket holds {largest}"
            ),
            Self::Positions { start, bucket_m } => write!(
                f,
                "a step from position {start} in a bucket of {bucket_m} rows has positions past \
                 u32"
            ),
            Self::Block { entry, block } => {
                write!(f, "block-table entry {entry} is block {block}, past u32")
            }
            Self::LogitsRow {
                elems,
                bucket_m,
                row,
            } => write!(
                f,
                "the result holds {elems} logits, not {bucket_m} rows of logits with a row {row}"
            ),
            Self::NanLogit { id } => {
                write!(
                    f,
                    "token {id}'s logit is NaN: the forward produced no argmax"
                )
            }
            Self::NoSamplingParams => write!(
                f,
                "{GREEDY} and the request carries no sampling parameters: pass --temperature 0 \
                 (or \"temperature\": 0)"
            ),
            Self::NotGreedy { temperature } => write!(
                f,
                "{GREEDY}: pass --temperature 0 (or \"temperature\": 0); this request asks for \
                 temperature {temperature}"
            ),
            Self::Penalties => write!(
                f,
                "{GREEDY}, with no penalty: drop the repetition, presence and frequency penalties \
                 (\"repetition_penalty\": 1, \"presence_penalty\": 0, \"frequency_penalty\": 0)"
            ),
            Self::Completions { n } => write!(
                f,
                "{GREEDY}, one completion per request: pass \"n\": 1; this request asks for {n}"
            ),
            Self::Logprobs => write!(
                f,
                "{GREEDY} and returns no logprobs: drop \"logprobs\" from the request"
            ),
            Self::PromptLogprobs => write!(
                f,
                "{GREEDY} and returns no prompt logprobs: drop \"prompt_logprobs\" from the \
                 request"
            ),
            Self::LogitTransform { field } => write!(
                f,
                "{GREEDY} and applies no {field}: drop \"{field}\" from the request"
            ),
        }
    }
}

impl std::error::Error for PlanError {}

/// Refuse a request whose sampling a greedy argmax of the raw logits does not honour: anything
/// but temperature 0 with no penalty, one completion, no logprobs, and no transform that moves
/// the argmax (a logit bias, an allowed-token list, bad words, a grammar). Top-k, top-p and min-p
/// pass: at temperature 0 they keep the argmax ([`SamplingParams::is_greedy`]).
pub fn check_sampling(params: Option<&SamplingParams>) -> Result<(), PlanError> {
    let p = params.ok_or(PlanError::NoSamplingParams)?;
    if p.reads_history() {
        return Err(PlanError::Penalties);
    }
    if !p.is_greedy() {
        return Err(PlanError::NotGreedy {
            temperature: p.temperature,
        });
    }
    if p.n != 1 {
        return Err(PlanError::Completions { n: p.n });
    }
    if p.logprobs.is_some() {
        return Err(PlanError::Logprobs);
    }
    if p.prompt_logprobs.is_some() {
        return Err(PlanError::PromptLogprobs);
    }
    let transforms = [
        (p.logit_bias.is_some(), "logit_bias"),
        (p.allowed_token_ids.is_some(), "allowed_token_ids"),
        (p.bad_words_token_ids.is_some(), "bad_words"),
        (p.guided_grammar.is_some(), "guided decoding"),
    ];
    match transforms.into_iter().find(|&(set, _)| set) {
        Some((_, field)) => Err(PlanError::LogitTransform { field }),
        None => Ok(()),
    }
}

/// The one request this step runs tokens for and how many, from the scheduler's
/// `num_scheduled_tokens`: `None` when the step runs none, and a refusal naming them when it runs
/// more than one.
pub fn scheduled_request(
    num_scheduled_tokens: &HashMap<String, usize>,
) -> Result<Option<(&str, usize)>, PlanError> {
    let mut running = (num_scheduled_tokens.iter()).filter(|&(_, &n)| n > 0);
    match (running.next(), running.next()) {
        (None, _) => Ok(None),
        (Some((id, &n)), None) => Ok(Some((id.as_str(), n))),
        (Some(_), Some(_)) => {
            let mut req_ids: Vec<String> = (num_scheduled_tokens.iter())
                .filter(|&(_, &n)| n > 0)
                .map(|(id, _)| id.clone())
                .collect();
            req_ids.sort();
            Err(PlanError::ManyRequests { req_ids })
        }
    }
}

/// The bucket a step of `tokens` rows runs in: of the compiled tapes' `bucket_ms` (in whatever
/// order the tapes are), the index and rows of the smallest that holds them. One token is the
/// one-row decode tape; more is the smallest prefill tape at least that long.
pub fn choose_bucket(bucket_ms: &[u32], tokens: usize) -> Result<(usize, u32), PlanError> {
    (bucket_ms.iter().copied().enumerate())
        .filter(|&(_, m)| m as usize >= tokens)
        .min_by_key(|&(_, m)| m)
        .ok_or(PlanError::NoBucket {
            tokens,
            largest: bucket_ms.iter().copied().max().unwrap_or(0),
        })
}

/// The tape's inputs for a step of `chunk` (the real token ids) from position `start` in a bucket
/// of `bucket_m` rows, into the worker's reusable buffers: the ids padded with 0 to `bucket_m`,
/// and the positions `start .. start + bucket_m`, the padding rows' included.
pub fn fill_inputs(
    chunk: &[u32],
    start: u32,
    bucket_m: u32,
    ids: &mut Vec<u32>,
    positions: &mut Vec<u32>,
) -> Result<(), PlanError> {
    debug_assert!(
        chunk.len() <= bucket_m as usize,
        "the bucket holds the chunk"
    );
    let end = start
        .checked_add(bucket_m)
        .ok_or(PlanError::Positions { start, bucket_m })?;
    ids.clear();
    ids.extend_from_slice(chunk);
    ids.resize(bucket_m as usize, 0);
    positions.clear();
    positions.extend(start..end);
    Ok(())
}

/// The scheduler's block ids, as the `u32` entries the kernels read, into `out`.
pub fn block_table_u32(blocks: &[usize], out: &mut Vec<u32>) -> Result<(), PlanError> {
    out.clear();
    for (entry, &block) in blocks.iter().enumerate() {
        out.push(u32::try_from(block).map_err(|_| PlanError::Block { entry, block })?);
    }
    Ok(())
}

/// Whether a step of `tokens` from `start` samples a token: it reaches the end of the request's
/// history of `history` tokens (the prompt's last chunk, or a decode). A prefill chunk short of
/// the prompt's end samples nothing.
pub fn emits_token(start: usize, tokens: usize, history: usize) -> bool {
    start + tokens >= history
}

/// Row `row` of a `[bucket_m, vocab]` result of bf16 logits.
pub fn logits_row(result: &[u16], bucket_m: u32, row: usize) -> Result<&[u16], PlanError> {
    let refuse = PlanError::LogitsRow {
        elems: result.len(),
        bucket_m,
        row,
    };
    let rows = bucket_m as usize;
    if rows == 0 || row >= rows || !result.len().is_multiple_of(rows) {
        return Err(refuse);
    }
    let vocab = result.len() / rows;
    Ok(&result[row * vocab..(row + 1) * vocab])
}

/// The greedy token of a row of bf16 logits: its argmax, the lowest id on a tie. A NaN logit is
/// refused rather than ranked.
pub fn greedy_token(row: &[u16]) -> Result<u32, PlanError> {
    let mut best: Option<(usize, f32)> = None;
    for (id, &bits) in row.iter().enumerate() {
        // bf16 is the top half of an f32.
        let v = f32::from_bits(u32::from(bits) << 16);
        if v.is_nan() {
            return Err(PlanError::NanLogit { id });
        }
        if best.is_none_or(|(_, b)| v > b) {
            best = Some((id, v));
        }
    }
    let (id, _) = best.ok_or(PlanError::LogitsRow {
        elems: 0,
        bucket_m: 1,
        row: 0,
    })?;
    Ok(id as u32)
}

/// The bytes one KV block of `block_size` tokens costs in a pool of `kv`'s geometry: per KV
/// layer a K and a V block of `kv_heads · block_size · head_dim` bf16. `None` on overflow.
pub fn kv_block_bytes(kv: &KvTape, block_size: usize) -> Option<usize> {
    [kv.kv_heads as usize, block_size, kv.head_dim as usize]
        .into_iter()
        .try_fold(kv.layers.len().checked_mul(2)?, |acc, f| acc.checked_mul(f))?
        .checked_mul(std::mem::size_of::<u16>())
}
