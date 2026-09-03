// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Sampling parameters for text generation, ported from `vllm/sampling_params.py`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Epsilon used to distinguish greedy from random sampling.
const SAMPLING_EPS: f64 = 1e-5;

// ---------------------------------------------------------------------------
// GuidedGrammar — constrained decoding specification
// ---------------------------------------------------------------------------

/// Specifies a grammar constraint for structured output / constrained decoding.
///
/// Stored in `SamplingParams` and used by workers to create per-request
/// `GrammarGuide` instances that mask logits during sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GuidedGrammar {
    /// `response_format: { type: "json_object" }` — output must be valid JSON.
    Json,
    /// `response_format: { type: "json_schema", json_schema: { schema: ... } }` —
    /// output must conform to the given JSON schema.
    JsonSchema { schema: serde_json::Value },
    /// `guided_regex` — output must match the given regex pattern.
    Regex { pattern: String },
    /// `guided_grammar` — output must conform to the given Lark/EBNF grammar.
    Ebnf { grammar: String },
    /// `response_format: { type: "structural_tag" }` — constrained tool/function
    /// invocations embedded in free text. The spec is a JSON-serialized object
    /// with `structures` (list of `{begin, schema, end}`) and `triggers`.
    /// Converted to a Lark grammar via `structural_tag_to_grammar()`.
    StructuralTag { spec: String },
}

// ---------------------------------------------------------------------------
// SamplingType
// ---------------------------------------------------------------------------

/// The type of sampling to use for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum SamplingType {
    /// Temperature is effectively zero -- pick the argmax token.
    Greedy = 0,
    /// Sample from the distribution (no fixed seed).
    Random = 1,
    /// Sample from the distribution with a fixed seed for reproducibility.
    RandomSeed = 2,
}

// ---------------------------------------------------------------------------
// RequestOutputKind
// ---------------------------------------------------------------------------

/// Controls how incremental output is returned to the caller.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RequestOutputKind {
    /// Return the entire output so far in every `RequestOutput`.
    #[default]
    Cumulative = 0,
    /// Return only deltas in each `RequestOutput`.
    Delta = 1,
    /// Do not return intermediate `RequestOutput`; only the final one.
    FinalOnly = 2,
}

// ---------------------------------------------------------------------------
// SamplingParams
// ---------------------------------------------------------------------------

/// Parameters that control token sampling during generation.
///
/// Mirrors the Python `SamplingParams` class from `vllm/sampling_params.py`.
/// Only the core fields used by the scheduler and sampler are included here;
/// fields that require a tokenizer or model config (e.g. `bad_words_token_ids`,
/// structured-output backends) are handled at a higher layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Number of output sequences to generate per prompt.
    pub n: u32,

    /// Penalizes tokens that have already appeared, regardless of frequency.
    /// Values in \[-2, 2\].
    pub presence_penalty: f64,

    /// Penalizes tokens proportionally to how often they have appeared.
    /// Values in \[-2, 2\].
    pub frequency_penalty: f64,

    /// Multiplicative penalty for repeated tokens. Values > 1 discourage
    /// repetition; values < 1 encourage it.
    pub repetition_penalty: f64,

    /// Softmax temperature. 0 means greedy.
    pub temperature: f64,

    /// Nucleus-sampling cumulative probability threshold. Must be in (0, 1].
    pub top_p: f64,

    /// Top-k filtering. 0 (or -1) disables.
    pub top_k: i32,

    /// Minimum probability relative to the most-likely token. Must be in [0, 1].
    pub min_p: f64,

    /// Optional random seed for reproducible sampling.
    pub seed: Option<u64>,

    /// Stop strings. Generation stops when any of these is emitted.
    pub stop: Vec<String>,

    /// Token IDs that trigger a stop.
    pub stop_token_ids: Vec<u32>,

    /// Whether to ignore the EOS token and keep generating.
    pub ignore_eos: bool,

    /// Maximum number of tokens to generate. `None` means unlimited
    /// (bounded only by the model's context length).
    pub max_tokens: Option<u32>,

    /// Minimum number of tokens to generate before allowing EOS / stop tokens.
    pub min_tokens: u32,

    /// Number of per-token log-probabilities to return.
    /// `None` means do not return logprobs. `-1` means return all.
    pub logprobs: Option<i32>,

    /// Number of per-prompt-token log-probabilities to return.
    pub prompt_logprobs: Option<i32>,

    /// Whether to detokenize the output.
    pub detokenize: bool,

    /// Whether to skip special tokens in the detokenized output.
    pub skip_special_tokens: bool,

    /// Whether to include the stop string in the generated output text.
    pub include_stop_str_in_output: bool,

    /// How incremental output is delivered to the caller.
    pub output_kind: RequestOutputKind,

    /// Per-token logit bias: add the bias value to the logit for each
    /// specified token ID before sampling.
    pub logit_bias: Option<HashMap<u32, f32>>,

    /// Grammar constraint for structured output (constrained decoding).
    /// When set, the sampler masks logits so that only tokens allowed by
    /// the grammar are sampled.
    pub guided_grammar: Option<GuidedGrammar>,

    /// When set, only these token IDs may be sampled. All other logits
    /// are masked to `-inf`. Similar to grammar masking but static.
    pub allowed_token_ids: Option<Vec<u32>>,

    /// Pre-tokenized bad word sequences. If the output ends with the prefix
    /// of a bad word, the completing token is suppressed (logit set to `-inf`).
    /// Each inner `Vec<u32>` is one bad word as a token sequence.
    pub bad_words_token_ids: Option<Vec<Vec<u32>>>,

    /// 🦭 When true, the SealPadProcessor forces pad tokens after EOS until
    /// block-aligned, so the final partial block is cacheable via prefix caching.
    #[serde(default)]
    pub seal: bool,

    /// When true, cached KV blocks from this request are deprioritized for
    /// retention. On read from the KV cache pool, volatile entries move to the
    /// front of the eviction queue (evict-first). Non-volatile entries move to
    /// the back (evict-last).
    #[serde(default)]
    pub volatile: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            n: 1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seed: None,
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            ignore_eos: false,
            max_tokens: Some(16),
            min_tokens: 0,
            logprobs: None,
            prompt_logprobs: None,
            detokenize: true,
            skip_special_tokens: true,
            include_stop_str_in_output: false,
            output_kind: RequestOutputKind::default(),
            logit_bias: None,
            guided_grammar: None,
            allowed_token_ids: None,
            bad_words_token_ids: None,
            seal: false,
            volatile: false,
        }
    }
}

/// Backend-neutral per-row inputs for a GPU sampler: the sampling params +
/// host-drawn uniforms + (optional) penalty token histories for one step's
/// non-greedy rows. Identical for cuda and metal — only the kernel dispatch that
/// consumes it differs. The worker gathers it from request state (see the GPU
/// worker base) and hands it to the backend's sampler (e.g.
/// `scratchy_target_metal::sampling::PendingSampler::prepare`). Every `Vec` is
/// length `njobs` except the two token-id arrays, which are `njobs * max_*_len`
/// row-major, padded with `vocab` (the penalty kernel ignores id == vocab).
#[derive(Debug, Default, Clone)]
pub struct GpuSampleParams {
    pub row_indices: Vec<u32>,
    pub temperatures: Vec<f32>,
    pub top_ks: Vec<i32>,
    pub top_ps: Vec<f32>,
    pub min_ps: Vec<f32>,
    pub uniforms: Vec<f32>,
    pub rep_penalties: Vec<f32>,
    pub freq_penalties: Vec<f32>,
    pub pres_penalties: Vec<f32>,
    pub output_token_ids: Vec<i32>,
    pub prompt_token_ids: Vec<i32>,
    pub max_output_len: u32,
    pub max_prompt_len: u32,
    pub any_penalty: bool,
}

/// Deterministic per-request seed for an UNSEEDED GPU-sampling request:
/// FNV-1a(req_id bytes, then `position` LE bytes) with a fixed basis. Backend-
/// AND TP-rank-independent (no process RNG, no per-rank state) so the cuda and
/// metal samplers derive the identical seed for the same `(req_id, position)`.
/// `position` is the request's generated-token count at this step (advances each
/// decode step so the seed — and thus the sample — varies). Seeded requests draw
/// from their own `StdRng` instead and never call this.
///
/// NOT `DefaultHasher` (which uses a random per-process seed for DoS protection).
pub fn fnv_seed(req_id: &str, position: u32) -> u32 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
    for b in req_id.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV prime
    }
    for b in position.to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (h >> 32) as u32 // upper 32 bits for maximum entropy
}

/// Convert a `u32` seed to a uniform in `[0, 1)` by stuffing its top 23 bits into
/// an IEEE-754 f32 mantissa — matches cuda's `gen_seed → uniform` conversion so
/// the metal and cuda samplers turn the same seed into the same uniform.
#[inline]
pub fn seed_to_uniform(seed: u32) -> f32 {
    f32::from_bits((seed >> 9) | 0x3F80_0000) - 1.0
}

impl SamplingParams {
    /// Determine the [`SamplingType`] implied by these parameters.
    ///
    /// * `temperature < EPS` => Greedy
    /// * `seed.is_some()`    => RandomSeed
    /// * otherwise           => Random
    pub fn sampling_type(&self) -> SamplingType {
        if self.temperature < SAMPLING_EPS {
            SamplingType::Greedy
        } else if self.seed.is_some() {
            SamplingType::RandomSeed
        } else {
            SamplingType::Random
        }
    }

    /// Whether plain argmax decoding is exactly correct for these params — no
    /// sampling transform (temperature, top-k / top-p / min-p, or a repetition /
    /// presence / frequency penalty) would move the argmax token. Backend-neutral:
    /// both the cuda and metal workers use this to route a row to the argmax fast
    /// path vs the on-GPU sampler. Stricter than `sampling_type() == Greedy`,
    /// which only looks at temperature.
    pub fn is_greedy(&self) -> bool {
        self.temperature < SAMPLING_EPS
            && (self.repetition_penalty - 1.0).abs() < 1e-9
            && self.presence_penalty == 0.0
            && self.frequency_penalty == 0.0
        // top-k / top-p / min-p intentionally NOT checked: at temperature ~0 the
        // distribution is one-hot on the argmax, which is always inside any
        // top-k/top-p/min-p kept set — so those filters cannot change the result.
        // (Penalties CAN move the argmax, so they stay above.)
    }

    /// Validate the sampling parameters, returning an error message on failure.
    ///
    /// This mirrors the `_verify_args` logic from the Python implementation.
    /// Uses `Cow<'static, str>` so static error messages don't allocate.
    pub fn validate(&self) -> Result<(), std::borrow::Cow<'static, str>> {
        if self.n < 1 {
            return Err(format!("n must be at least 1, got {}", self.n).into());
        }
        if !(-2.0..=2.0).contains(&self.presence_penalty) {
            return Err(format!(
                "presence_penalty must be in [-2, 2], got {}",
                self.presence_penalty
            )
            .into());
        }
        if !(-2.0..=2.0).contains(&self.frequency_penalty) {
            return Err(format!(
                "frequency_penalty must be in [-2, 2], got {}",
                self.frequency_penalty
            )
            .into());
        }
        if self.repetition_penalty <= 0.0 {
            return Err(format!(
                "repetition_penalty must be > 0, got {}",
                self.repetition_penalty
            )
            .into());
        }
        if self.temperature < 0.0 {
            return Err(
                format!("temperature must be non-negative, got {}", self.temperature).into(),
            );
        }
        if !(0.0 < self.top_p && self.top_p <= 1.0) {
            return Err(format!("top_p must be in (0, 1], got {}", self.top_p).into());
        }
        if self.top_k < -1 {
            return Err(format!(
                "top_k must be 0 (disable), or at least 1, got {}",
                self.top_k
            )
            .into());
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err(format!("min_p must be in [0, 1], got {}", self.min_p).into());
        }
        if let Some(max) = self.max_tokens
            && max < 1
        {
            return Err(format!("max_tokens must be at least 1, got {max}").into());
        }
        if self.min_tokens > 0
            && let Some(max) = self.max_tokens
            && self.min_tokens > max
        {
            return Err(format!(
                "min_tokens ({}) must be <= max_tokens ({max})",
                self.min_tokens
            )
            .into());
        }
        if let Some(lp) = self.logprobs
            && lp != -1
            && lp < 0
        {
            return Err(format!("logprobs must be non-negative or -1, got {lp}").into());
        }
        if let Some(plp) = self.prompt_logprobs
            && plp != -1
            && plp < 0
        {
            return Err(format!("prompt_logprobs must be non-negative or -1, got {plp}").into());
        }
        if let Some(ref ids) = self.allowed_token_ids
            && ids.is_empty()
        {
            return Err("allowed_token_ids must not be empty when set".into());
        }
        if let Some(ref seqs) = self.bad_words_token_ids {
            if seqs.is_empty() {
                return Err("bad_words_token_ids must not be empty when set".into());
            }
            if seqs.iter().any(|s| s.is_empty()) {
                return Err("bad_words_token_ids must not contain empty sequences".into());
            }
        }
        if !self.stop.is_empty() && !self.detokenize {
            return Err("stop strings are only supported when detokenize is true".into());
        }
        if self.stop.iter().any(|s| s.is_empty()) {
            return Err("stop cannot contain an empty string".into());
        }
        // Greedy-specific check.
        if self.temperature < SAMPLING_EPS && self.n > 1 {
            return Err(format!("n must be 1 when using greedy sampling, got {}", self.n).into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Logprobs types
// ---------------------------------------------------------------------------

/// Log-probability information for a single token position.
#[derive(Debug, Clone)]
pub struct TokenLogprob {
    /// The token ID.
    pub token_id: u32,
    /// The log-probability of this token.
    pub logprob: f32,
    /// Rank of this token in the vocabulary (1-indexed).
    pub rank: u32,
}

/// Log-probability output for a single generation step.
///
/// Contains the sampled token's logprob and the top-k alternatives.
#[derive(Debug, Clone)]
pub struct LogprobsOutput {
    /// The sampled token's log-probability info.
    pub sampled: TokenLogprob,
    /// Top-k alternative tokens (may be empty if logprobs not requested).
    pub top_logprobs: Vec<TokenLogprob>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_sampling_params() {
        let p = SamplingParams::default();
        assert_eq!(p.n, 1);
        assert_eq!(p.temperature, 1.0);
        assert_eq!(p.top_p, 1.0);
        assert_eq!(p.top_k, 0);
        assert_eq!(p.max_tokens, Some(16));
        assert!(p.stop.is_empty());
        assert!(p.stop_token_ids.is_empty());
        assert!(p.detokenize);
        assert!(p.skip_special_tokens);
    }

    #[test]
    fn test_sampling_type_greedy() {
        let p = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(p.sampling_type(), SamplingType::Greedy);
    }

    #[test]
    fn test_sampling_type_random() {
        let p = SamplingParams {
            temperature: 0.8,
            seed: None,
            ..Default::default()
        };
        assert_eq!(p.sampling_type(), SamplingType::Random);
    }

    #[test]
    fn test_sampling_type_random_seed() {
        let p = SamplingParams {
            temperature: 0.8,
            seed: Some(42),
            ..Default::default()
        };
        assert_eq!(p.sampling_type(), SamplingType::RandomSeed);
    }

    #[test]
    fn test_validate_ok() {
        let p = SamplingParams::default();
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_validate_presence_penalty_out_of_range() {
        let p = SamplingParams {
            presence_penalty: 3.0,
            ..Default::default()
        };
        let err = p.validate().unwrap_err();
        assert!(err.contains("presence_penalty"));
    }

    #[test]
    fn test_validate_frequency_penalty_out_of_range() {
        let p = SamplingParams {
            frequency_penalty: -3.0,
            ..Default::default()
        };
        let err = p.validate().unwrap_err();
        assert!(err.contains("frequency_penalty"));
    }

    #[test]
    fn test_validate_repetition_penalty_zero() {
        let p = SamplingParams {
            repetition_penalty: 0.0,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_negative_temperature() {
        let p = SamplingParams {
            temperature: -1.0,
            ..Default::default()
        };
        let err = p.validate().unwrap_err();
        assert!(err.contains("temperature"));
    }

    #[test]
    fn test_validate_top_p_zero() {
        let p = SamplingParams {
            top_p: 0.0,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_top_p_above_one() {
        let p = SamplingParams {
            top_p: 1.5,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_top_k_invalid() {
        let p = SamplingParams {
            top_k: -2,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_min_p_out_of_range() {
        let p = SamplingParams {
            min_p: 1.5,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_max_tokens_zero() {
        let p = SamplingParams {
            max_tokens: Some(0),
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_min_tokens_exceeds_max() {
        let p = SamplingParams {
            min_tokens: 100,
            max_tokens: Some(10),
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_greedy_n_greater_than_one() {
        let p = SamplingParams {
            temperature: 0.0,
            n: 2,
            ..Default::default()
        };
        let err = p.validate().unwrap_err();
        assert!(err.contains("greedy"));
    }

    #[test]
    fn test_validate_empty_stop_string() {
        let p = SamplingParams {
            stop: vec!["".into()],
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_stop_without_detokenize() {
        let p = SamplingParams {
            stop: vec!["</s>".into()],
            detokenize: false,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_logprobs_invalid() {
        let p = SamplingParams {
            logprobs: Some(-2),
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_logprobs_minus_one_ok() {
        let p = SamplingParams {
            logprobs: Some(-1),
            ..Default::default()
        };
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_validate_prompt_logprobs_invalid() {
        let p = SamplingParams {
            prompt_logprobs: Some(-5),
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_serde_roundtrip() {
        let p = SamplingParams {
            temperature: 0.7,
            top_p: 0.9,
            seed: Some(123),
            stop: vec!["STOP".into()],
            max_tokens: Some(256),
            ..Default::default()
        };
        let json = serde_json::to_string(&p).unwrap();
        let p2: SamplingParams = serde_json::from_str(&json).unwrap();
        assert_eq!(p.temperature, p2.temperature);
        assert_eq!(p.top_p, p2.top_p);
        assert_eq!(p.seed, p2.seed);
        assert_eq!(p.stop, p2.stop);
        assert_eq!(p.max_tokens, p2.max_tokens);
    }

    #[test]
    fn test_request_output_kind_default() {
        assert_eq!(RequestOutputKind::default(), RequestOutputKind::Cumulative);
    }

    #[test]
    fn test_sampling_type_repr() {
        assert_eq!(SamplingType::Greedy as u8, 0);
        assert_eq!(SamplingType::Random as u8, 1);
        assert_eq!(SamplingType::RandomSeed as u8, 2);
    }

    #[test]
    fn test_request_output_kind_repr() {
        assert_eq!(RequestOutputKind::Cumulative as u8, 0);
        assert_eq!(RequestOutputKind::Delta as u8, 1);
        assert_eq!(RequestOutputKind::FinalOnly as u8, 2);
    }

    #[test]
    fn test_guided_grammar_regex_serde_roundtrip() {
        let grammar = GuidedGrammar::Regex {
            pattern: "[0-9]+".to_string(),
        };
        let json = serde_json::to_string(&grammar).unwrap();
        let parsed: GuidedGrammar = serde_json::from_str(&json).unwrap();
        match parsed {
            GuidedGrammar::Regex { pattern } => assert_eq!(pattern, "[0-9]+"),
            other => panic!("expected Regex, got {other:?}"),
        }
    }

    #[test]
    fn test_max_tokens_none_is_unlimited() {
        let p = SamplingParams {
            max_tokens: None,
            ..Default::default()
        };
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_validate_allowed_token_ids_empty() {
        let p = SamplingParams {
            allowed_token_ids: Some(vec![]),
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_allowed_token_ids_ok() {
        let p = SamplingParams {
            allowed_token_ids: Some(vec![1, 2, 3]),
            ..Default::default()
        };
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_validate_allowed_token_ids_none_ok() {
        let p = SamplingParams {
            allowed_token_ids: None,
            ..Default::default()
        };
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_validate_bad_words_token_ids_empty() {
        let p = SamplingParams {
            bad_words_token_ids: Some(vec![]),
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_bad_words_token_ids_empty_seq() {
        let p = SamplingParams {
            bad_words_token_ids: Some(vec![vec![]]),
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_bad_words_token_ids_ok() {
        let p = SamplingParams {
            bad_words_token_ids: Some(vec![vec![1, 2], vec![3]]),
            ..Default::default()
        };
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_validate_bad_words_token_ids_none_ok() {
        let p = SamplingParams {
            bad_words_token_ids: None,
            ..Default::default()
        };
        assert!(p.validate().is_ok());
    }
}

/// Model-recommended sampling defaults from `generation_config.json`.
///
/// Mirrors Python vLLM's `generation_config="auto"` behavior
/// (`ModelConfig.get_diff_sampling_param`): the model repo's
/// `generation_config.json` supplies DEFAULT values for the sampling
/// fields below; a value the user sets explicitly on the request
/// always wins. The key set matches Python vLLM exactly
/// (`repetition_penalty`, `temperature`, `top_p`, `top_k`, `min_p`,
/// `max_new_tokens`).
///
/// Why this exists: thinker models (Qwen3.5) ship
/// `temperature 1.0 / top_k 20 / top_p 0.95` and degrade into
/// endless repetition under the neutral defaults (greedy-adjacent
/// full-vocab sampling). HF transformers and mlx-lm honor the file;
/// a serving stack that ignores it produces different (worse) output
/// from the same checkpoint.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenerationDefaults {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i32>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub max_new_tokens: Option<u32>,
}

impl GenerationDefaults {
    /// Parse `<model_dir>/generation_config.json`. Missing file or
    /// unparsable JSON ⇒ empty defaults (neutral behavior, identical
    /// to Python vLLM when the repo ships no generation config).
    pub fn from_model_dir(dir: &std::path::Path) -> Self {
        let path = dir.join("generation_config.json");
        let Ok(bytes) = std::fs::read(&path) else {
            return Self::default();
        };
        let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            return Self::default();
        };
        Self {
            temperature: json.get("temperature").and_then(|v| v.as_f64()),
            top_p: json.get("top_p").and_then(|v| v.as_f64()),
            top_k: json.get("top_k").and_then(|v| v.as_i64()).map(|v| v as i32),
            min_p: json.get("min_p").and_then(|v| v.as_f64()),
            repetition_penalty: json.get("repetition_penalty").and_then(|v| v.as_f64()),
            max_new_tokens: json
                .get("max_new_tokens")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32),
        }
    }

    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// [`SamplingParams::default`] overlaid with the model defaults —
    /// the base every request starts from before explicit user values
    /// are applied.
    pub fn as_base(&self) -> SamplingParams {
        let mut p = SamplingParams::default();
        if let Some(t) = self.temperature {
            p.temperature = t;
        }
        if let Some(t) = self.top_p {
            p.top_p = t;
        }
        if let Some(t) = self.top_k {
            p.top_k = t;
        }
        if let Some(t) = self.min_p {
            p.min_p = t;
        }
        if let Some(t) = self.repetition_penalty {
            p.repetition_penalty = t;
        }
        if let Some(t) = self.max_new_tokens {
            p.max_tokens = Some(t);
        }
        p
    }
}
